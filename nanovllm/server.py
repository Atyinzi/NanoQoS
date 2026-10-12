"""Small OpenAI-compatible HTTP server around one nano-vLLM GPU worker.

Run on Linux/WSL with CUDA, NCCL, Triton and FlashAttention installed.
"""

import argparse
import json
import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from nanovllm.sampling_params import SamplingParams

LOG = logging.getLogger(__name__)


@dataclass
class Job:
    request_id: str
    prompt: str | None
    messages: list[dict] | None
    params: SamplingParams
    deadline_seconds: float | None
    priority: int
    chat: bool
    events: queue.Queue = field(default_factory=queue.Queue)
    submitted_at: float | None = None
    seq_id: int | None = None
    rendered_text: str = ""


class Service:
    def __init__(self, model: str, max_pending: int, engine_factory=None, **engine_options):
        if max_pending <= 0:
            raise ValueError("max_pending must be positive")
        self.commands: queue.Queue = queue.Queue()
        self.slots = threading.BoundedSemaphore(max_pending)
        self.jobs: dict[str, Job] = {}
        self.seq_to_job: dict[int, Job] = {}
        self.lock = threading.Lock()
        self.completed = 0
        self.failed = 0
        self.generated_tokens = 0
        self.active = 0
        self.ttft_sum = 0.0
        self.ttft_count = 0
        self.tpot_sum = 0.0
        self.tpot_count = 0
        self.latency_buckets = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
        self.ttft_buckets = {limit: 0 for limit in self.latency_buckets}
        self.tpot_buckets = {limit: 0 for limit in self.latency_buckets}
        self.preemptions = 0
        self.prefix_cached_tokens = 0
        self.recomputed_tokens = 0
        self.kv_peak_used_blocks = 0
        self.error = None
        self.ready = threading.Event()
        self.worker = threading.Thread(target=self._loop, args=(model, engine_options, engine_factory), daemon=True)
        self.worker.start()
        self.ready.wait()
        if self.error:
            raise RuntimeError("model initialization failed") from self.error

    def submit(self, job: Job) -> bool:
        if not self.slots.acquire(blocking=False):
            return False
        job.submitted_at = time.perf_counter()
        with self.lock:
            self.jobs[job.request_id] = job
            self.active += 1
        self.commands.put(("submit", job))
        return True

    def cancel(self, request_id: str):
        self.commands.put(("cancel", request_id))

    def _finish(self, job: Job, kind: str, payload=None):
        with self.lock:
            if job.request_id not in self.jobs:
                return
            self.jobs.pop(job.request_id)
            self.active -= 1
            if kind == "error":
                self.failed += 1
            elif kind == "finish":
                self.completed += 1
            self.slots.release()
        if job.seq_id is not None:
            self.seq_to_job.pop(job.seq_id, None)
        job.events.put((kind, payload))

    def _record_completion(self, seq):
        with self.lock:
            self.preemptions += seq.preemptions
            self.prefix_cached_tokens += seq.prefix_cached_tokens
            self.recomputed_tokens += seq.recomputed_tokens
            if seq.token_times:
                ttft = seq.token_times[0] - seq.arrival_time
                self.ttft_sum += ttft
                self.ttft_count += 1
                for limit in self.latency_buckets:
                    self.ttft_buckets[limit] += ttft <= limit
            for before, after in zip(seq.token_times, seq.token_times[1:]):
                tpot = after - before
                self.tpot_sum += tpot
                self.tpot_count += 1
                for limit in self.latency_buckets:
                    self.tpot_buckets[limit] += tpot <= limit

    def _loop(self, model: str, engine_options: dict, engine_factory):
        engine = None
        try:
            if engine_factory is None:
                from nanovllm import LLM
                engine_factory = LLM
            engine = engine_factory(model, **engine_options)
            self.ready.set()
            while True:
                for job in list(self.seq_to_job.values()):
                    seq = engine.requests[job.seq_id]
                    if seq.deadline_time is not None and time.perf_counter() > seq.deadline_time:
                        engine.cancel_request(job.seq_id)
                        engine.forget_request(job.seq_id)
                        self._finish(job, "error", "deadline exceeded")
                try:
                    command, value = self.commands.get(timeout=0.02 if engine.is_finished() else 0)
                    if command == "stop":
                        break
                    if command == "cancel":
                        job = self.jobs.get(value)
                        if job:
                            if job.seq_id is not None:
                                engine.cancel_request(job.seq_id)
                                engine.forget_request(job.seq_id)
                            self._finish(job, "error", "request canceled")
                    elif command == "submit":
                        job = value
                        if job.request_id not in self.jobs:
                            continue
                        try:
                            prompt = job.prompt
                            if job.chat:
                                prompt = engine.tokenizer.apply_chat_template(
                                    job.messages, tokenize=False, add_generation_prompt=True)
                            job.seq_id = engine.add_request(
                                prompt, job.params, deadline_seconds=job.deadline_seconds,
                                priority=job.priority, arrival_time=job.submitted_at)
                            self.seq_to_job[job.seq_id] = job
                        except Exception as exc:
                            self._finish(job, "error", str(exc))
                except queue.Empty:
                    pass
                if engine.is_finished():
                    continue
                try:
                    events, _ = engine.step_events()
                except Exception as exc:
                    LOG.exception("model step failed")
                    for job in list(self.seq_to_job.values()):
                        engine.cancel_request(job.seq_id)
                        engine.forget_request(job.seq_id)
                        self._finish(job, "error", str(exc))
                    continue
                for event in events:
                    job = self.seq_to_job.get(event["seq_id"])
                    if job is None:
                        continue
                    seq = engine.requests[event["seq_id"]]
                    full_text = engine.tokenizer.decode(seq.completion_token_ids,
                                                        skip_special_tokens=True)
                    delta = full_text[len(job.rendered_text):] if full_text.startswith(job.rendered_text) else full_text
                    job.rendered_text = full_text
                    with self.lock:
                        self.generated_tokens += 1
                    job.events.put(("token", delta))
                    if event["finished"]:
                        self._record_completion(seq)
                        self._finish(job, "finish", {"text": full_text,
                                                     "prompt_tokens": seq.num_prompt_tokens,
                                                     "completion_tokens": seq.num_completion_tokens,
                                                     "finish_reason": "stop" if seq.last_token == engine.scheduler.eos
                                                     else "length"})
                        engine.forget_request(event["seq_id"])
                with self.lock:
                    self.kv_peak_used_blocks = max(
                        self.kv_peak_used_blocks,
                        engine.scheduler.block_manager.peak_used_blocks)
        except Exception as exc:
            LOG.exception("service worker failed")
            self.error = exc
            self.ready.set()
            for job in list(self.jobs.values()):
                self._finish(job, "error", str(exc))
        finally:
            if engine:
                engine.exit()

    def stop(self):
        self.commands.put(("stop", None))
        self.worker.join(timeout=30)

    def metrics(self) -> str:
        with self.lock:
            lines = [f"nanovllm_active_requests {self.active}",
                     f"nanovllm_completed_requests_total {self.completed}",
                     f"nanovllm_failed_requests_total {self.failed}",
                     f"nanovllm_generated_tokens_total {self.generated_tokens}",
                     f"nanovllm_preemptions_total {self.preemptions}",
                     f"nanovllm_prefix_cached_tokens_total {self.prefix_cached_tokens}",
                     f"nanovllm_recomputed_tokens_total {self.recomputed_tokens}",
                     f"nanovllm_kv_peak_used_blocks {self.kv_peak_used_blocks}"]
            for name, buckets, total, count in (
                ("ttft", self.ttft_buckets, self.ttft_sum, self.ttft_count),
                ("tpot", self.tpot_buckets, self.tpot_sum, self.tpot_count),
            ):
                for limit in self.latency_buckets:
                    lines.append(f'nanovllm_{name}_seconds_bucket{{le="{limit}"}} {buckets[limit]}')
                lines.append(f'nanovllm_{name}_seconds_bucket{{le="+Inf"}} {count}')
                lines.append(f"nanovllm_{name}_seconds_sum {total}")
                lines.append(f"nanovllm_{name}_seconds_count {count}")
            return "\n".join(lines) + "\n"


def make_handler(service: Service, model_name: str):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def _send(self, status: int, data: dict | str, content_type="application/json"):
            payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
            encoded = payload.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok"})
            elif self.path == "/metrics":
                self._send(200, service.metrics(), "text/plain")
            else:
                self._send(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if self.path not in ("/v1/completions", "/v1/chat/completions"):
                self._send(404, {"error": {"message": "not found"}})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1_000_000:
                    raise ValueError("invalid request size")
                body = json.loads(self.rfile.read(length))
                chat = self.path.endswith("chat/completions")
                prompt = body.get("prompt")
                messages = body.get("messages")
                if chat and (not isinstance(messages, list) or not messages):
                    raise ValueError("messages must be a nonempty list")
                if not chat and (not isinstance(prompt, str) or not prompt):
                    raise ValueError("prompt must be a nonempty string")
                max_tokens = int(body.get("max_tokens", 64))
                if max_tokens <= 0:
                    raise ValueError("max_tokens must be positive")
                params = SamplingParams(temperature=float(body.get("temperature", 1.0)),
                                        max_tokens=max_tokens)
                deadline = body.get("deadline_seconds")
                if deadline is not None:
                    deadline = float(deadline)
                    if deadline <= 0:
                        raise ValueError("deadline_seconds must be positive")
                job = Job("cmpl-" + uuid.uuid4().hex, prompt, messages, params,
                          deadline, int(body.get("priority", 0)), chat)
            except (ValueError, TypeError, json.JSONDecodeError, AssertionError) as exc:
                self._send(400, {"error": {"message": str(exc)}})
                return
            if not service.submit(job):
                self._send(429, {"error": {"message": "request queue is full"}})
                return
            stream = bool(body.get("stream", False))
            created = int(time.time())
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
            try:
                while True:
                    try:
                        kind, payload = job.events.get(timeout=120)
                    except queue.Empty:
                        service.cancel(job.request_id)
                        if not stream:
                            self._send(504, {"error": {"message": "request timed out"}})
                        break
                    if kind == "token" and stream:
                        choice = {"index": 0, "delta": {"content": payload}, "finish_reason": None}
                        if not chat:
                            choice = {"index": 0, "text": payload, "finish_reason": None}
                        chunk = {"id": job.request_id, "object": "chat.completion.chunk" if chat else "text_completion",
                                 "created": created, "model": model_name, "choices": [choice]}
                        self.wfile.write(b"data: " + json.dumps(chunk, ensure_ascii=False).encode("utf-8") + b"\n\n")
                        self.wfile.flush()
                    elif kind == "finish":
                        if stream:
                            choice = {"index": 0, "delta": {}, "finish_reason": payload["finish_reason"]} if chat else {
                                "index": 0, "text": "", "finish_reason": payload["finish_reason"]}
                            chunk = {"id": job.request_id,
                                     "object": "chat.completion.chunk" if chat else "text_completion",
                                     "created": created, "model": model_name, "choices": [choice]}
                            self.wfile.write(b"data: " + json.dumps(chunk).encode("utf-8") + b"\n\n")
                            self.wfile.write(b"data: [DONE]\n\n")
                            self.wfile.flush()
                        else:
                            choice = {"index": 0, "message": {"role": "assistant", "content": payload["text"]},
                                      "finish_reason": payload["finish_reason"]} if chat else {
                                          "index": 0, "text": payload["text"],
                                          "finish_reason": payload["finish_reason"]}
                            self._send(200, {"id": job.request_id,
                                             "object": "chat.completion" if chat else "text_completion",
                                             "created": created, "model": model_name,
                                             "choices": [choice],
                                             "usage": {"prompt_tokens": payload["prompt_tokens"],
                                                       "completion_tokens": payload["completion_tokens"],
                                                       "total_tokens": payload["prompt_tokens"] + payload["completion_tokens"]}})
                        break
                    elif kind == "error":
                        if stream:
                            self.wfile.write(b"data: " + json.dumps({"error": payload}).encode("utf-8") + b"\n\n")
                        else:
                            self._send(504 if payload == "deadline exceeded" else 500,
                                       {"error": {"message": payload}})
                        break
            except (BrokenPipeError, ConnectionResetError):
                service.cancel(job.request_id)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--policy", choices=["fcfs", "shortest", "slo"], default="fcfs")
    parser.add_argument("--max-pending", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--eager", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    service = Service(args.model, args.max_pending, scheduler_policy=args.policy,
                      max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
                      max_num_batched_tokens=args.max_num_batched_tokens,
                      gpu_memory_utilization=args.gpu_memory_utilization,
                      enforce_eager=args.eager)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service, args.model))
    try:
        LOG.info("Serving %s on http://%s:%s", args.model, args.host, args.port)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.stop()


if __name__ == "__main__":
    main()
