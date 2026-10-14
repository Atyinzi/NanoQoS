"""Arrival based benchmark. The synthetic backend tests scheduler behavior, not GPU speed."""

import argparse
import hashlib
import json
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.metrics import summarize
from nanovllm.sampling_params import SamplingParams


@dataclass
class Request:
    arrival_offset: float
    prompt: list[int]
    max_tokens: int
    deadline_seconds: float | None


def workload_sha256(requests: list[Request]) -> str:
    payload = json.dumps([asdict(request) for request in requests],
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=5, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def git_dirty() -> bool | None:
    try:
        result = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                text=True, timeout=5, check=False)
        return bool(result.stdout) if result.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def make_workload(count: int, seed: int, mode: str, rate: float,
                  shared_prefix: int, max_input: int, max_output: int,
                  deadline: float | None = None, deadline_jitter: float = 0.0) -> list[Request]:
    if count <= 0 or max_input < 2 or max_output <= 0 or shared_prefix >= max_input:
        raise ValueError("invalid workload size")
    if deadline_jitter < 0 or (deadline is not None and deadline <= deadline_jitter):
        raise ValueError("deadline must exceed deadline_jitter")
    rng = random.Random(seed)
    prefix = [rng.randrange(1, 10000) for _ in range(shared_prefix)]
    offset = 0.0
    requests = []
    for _ in range(count):
        if mode == "poisson":
            if rate <= 0:
                raise ValueError("rate must be positive")
            offset += rng.expovariate(rate)
        elif mode != "burst":
            raise ValueError("mode must be burst or poisson")
        input_len = rng.randint(max(shared_prefix + 1, 2), max_input)
        output_len = rng.randint(1, max_output)
        prompt = prefix + [rng.randrange(1, 10000) for _ in range(input_len - shared_prefix)]
        request_deadline = None if deadline is None else deadline + rng.uniform(-deadline_jitter, deadline_jitter)
        requests.append(Request(offset, prompt, output_len, request_deadline))
    return requests


def run(args):
    requests = make_workload(args.requests, args.seed, args.arrivals, args.rate,
                             args.shared_prefix, args.max_input, args.max_output,
                             args.deadline, args.deadline_jitter)
    sequences = []
    scheduler = None
    engine = None
    if args.backend == "synthetic":
        Sequence.block_size = args.block_size
        scheduler = Scheduler(SimpleNamespace(
            max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_num_batched_tokens,
            eos=-1, kvcache_block_size=args.block_size, num_kvcache_blocks=args.num_blocks,
            scheduler_policy=args.policy, aging_seconds=args.aging_seconds,
            max_consecutive_prefills=args.max_consecutive_prefills,
        ))
    else:
        from nanovllm import LLM
        engine = LLM(str(args.model), enforce_eager=args.eager, scheduler_policy=args.policy,
                     attention_backend=args.attention_backend,
                     aging_seconds=args.aging_seconds,
                     max_consecutive_prefills=args.max_consecutive_prefills,
                     max_num_seqs=args.max_num_seqs,
                     max_num_batched_tokens=args.max_num_batched_tokens,
                     max_model_len=args.max_model_len,
                     kvcache_block_size=args.block_size,
                     gpu_memory_utilization=args.gpu_memory_utilization)
    start = time.perf_counter()
    started_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    next_request = 0
    try:
        while next_request < len(requests) or not (scheduler or engine).is_finished():
            now = time.perf_counter()
            while next_request < len(requests) and requests[next_request].arrival_offset <= now - start:
                request = requests[next_request]
                params = SamplingParams(max_tokens=request.max_tokens, ignore_eos=True)
                if scheduler:
                    seq = Sequence(request.prompt, params, deadline_seconds=request.deadline_seconds)
                    scheduler.add(seq)
                else:
                    seq_id = engine.add_request(request.prompt, params,
                                                deadline_seconds=request.deadline_seconds)
                    seq = engine.requests[seq_id]
                sequences.append(seq)
                next_request += 1
            if (scheduler or engine).is_finished():
                if next_request < len(requests):
                    sleep_for = requests[next_request].arrival_offset - (time.perf_counter() - start)
                    time.sleep(max(0, min(sleep_for, 0.05)))
                continue
            if scheduler:
                batch, prefill = scheduler.schedule()
                time.sleep(args.prefill_seconds * sum(seq.num_scheduled_tokens for seq in batch)
                           if prefill else args.decode_seconds)
                scheduler.postprocess(batch, [1] * len(batch), prefill)
            else:
                engine.step_events()
        elapsed = time.perf_counter() - start
        result = {
            "schema_version": 1,
            "backend": args.backend,
            "model": str(args.model) if args.backend == "gpu" else None,
            "policy": args.policy,
            "workload_sha256": workload_sha256(requests),
            "started_at_utc": started_at_utc,
            "workload": {"requests": args.requests, "seed": args.seed, "arrivals": args.arrivals,
                         "rate": args.rate, "shared_prefix": args.shared_prefix,
                         "max_input": args.max_input, "max_output": args.max_output,
                         "deadline_seconds": args.deadline, "deadline_jitter": args.deadline_jitter},
            "engine": {"max_num_seqs": args.max_num_seqs,
                       "max_num_batched_tokens": args.max_num_batched_tokens,
                       "max_model_len": args.max_model_len,
                       "block_size": (scheduler or engine.scheduler).block_manager.block_size,
                       "num_blocks": len((scheduler or engine.scheduler).block_manager.blocks),
                       "eager": args.eager,
                       "attention_backend": args.attention_backend if engine else None,
                       "gpu_memory_utilization": args.gpu_memory_utilization if engine else None,
                       "prefill_seconds": args.prefill_seconds if scheduler else None,
                       "decode_seconds": args.decode_seconds if scheduler else None},
            "environment": {"python": platform.python_version(),
                            "platform": platform.platform(), "git_commit": git_commit(),
                            "git_dirty": git_dirty()},
            "kv_cache": {
                "peak_used_blocks": (scheduler or engine.scheduler).block_manager.peak_used_blocks,
                "cache_hits": (scheduler or engine.scheduler).block_manager.cache_hits,
                "cache_evictions": (scheduler or engine.scheduler).block_manager.cache_evictions,
            },
            "metrics": summarize(sequences, elapsed),
        }
        if engine:
            import torch
            result["environment"]["torch"] = torch.__version__
            result["environment"]["gpu"] = torch.cuda.get_device_name(0)
            result["environment"]["cuda_runtime"] = torch.version.cuda
            result["environment"]["gpu_total_bytes"] = torch.cuda.get_device_properties(0).total_memory
            for package in ("triton", "flash-attn", "transformers"):
                try:
                    result["environment"][package] = metadata.version(package)
                except metadata.PackageNotFoundError:
                    result["environment"][package] = None
            config_file = args.model / "config.json"
            if config_file.is_file():
                result["environment"]["model_config_sha256"] = hashlib.sha256(
                    config_file.read_bytes()).hexdigest()
            result["environment"]["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
        return result
    finally:
        if engine:
            engine.exit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["synthetic", "gpu"], default="synthetic")
    parser.add_argument("--model", type=Path, default=Path("Qwen"))
    parser.add_argument("--policy", choices=["fcfs", "shortest", "slo"], default="fcfs")
    parser.add_argument("--arrivals", choices=["burst", "poisson"], default="burst")
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rate", type=float, default=5.0)
    parser.add_argument("--shared-prefix", type=int, default=0)
    parser.add_argument("--max-input", type=int, default=256)
    parser.add_argument("--max-output", type=int, default=32)
    parser.add_argument("--deadline", type=float, default=None)
    parser.add_argument("--deadline-jitter", type=float, default=0.0)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--attention-backend", choices=["flash", "sdpa"], default="flash")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=256)
    parser.add_argument("--aging-seconds", type=float, default=5.0)
    parser.add_argument("--max-consecutive-prefills", type=int, default=4)
    parser.add_argument("--prefill-seconds", type=float, default=0.00001)
    parser.add_argument("--decode-seconds", type=float, default=0.001)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.backend == "gpu" and args.max_input + args.max_output > args.max_model_len:
        parser.error("max-input + max-output exceeds max-model-len")
    if args.backend == "gpu" and args.attention_backend == "sdpa" and not args.eager:
        parser.error("sdpa attention requires --eager")
    if args.backend == "synthetic" and args.max_input + args.max_output - 1 > args.block_size * args.num_blocks:
        parser.error("largest request cannot fit in the synthetic KV cache")
    if args.backend == "synthetic" and args.shared_prefix and args.shared_prefix < args.block_size:
        parser.error("shared-prefix must cover at least one full block to measure reuse")
    result = run(args)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
