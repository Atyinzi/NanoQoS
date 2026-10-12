import http.client
import json
import threading
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.server import Service, make_handler


class FakeTokenizer:
    def encode(self, prompt):
        return [ord(char) for char in prompt]

    def decode(self, tokens, skip_special_tokens=True):
        return "".join(chr(token) for token in tokens)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return " ".join(message["content"] for message in messages)


class FakeEngine:
    def __init__(self, model, **kwargs):
        Sequence.block_size = 8
        self.tokenizer = FakeTokenizer()
        self.scheduler = Scheduler(SimpleNamespace(
            max_num_seqs=4, max_num_batched_tokens=64, eos=-1,
            kvcache_block_size=8, num_kvcache_blocks=64,
            scheduler_policy="fcfs", aging_seconds=5.0,
            max_consecutive_prefills=4,
        ))
        self.requests = {}

    def add_request(self, prompt, params, deadline_seconds=None, priority=0, arrival_time=None):
        item = Sequence(self.tokenizer.encode(prompt), params,
                        deadline_seconds=deadline_seconds, priority=priority,
                        arrival_time=arrival_time)
        self.requests[item.seq_id] = item
        self.scheduler.add(item)
        return item.seq_id

    def is_finished(self):
        return self.scheduler.is_finished()

    def step_events(self):
        batch, prefill = self.scheduler.schedule()
        before = {item.seq_id: item.num_completion_tokens for item in batch}
        self.scheduler.postprocess(batch, [ord("x")] * len(batch), prefill)
        return ([{"seq_id": item.seq_id, "finished": item.is_finished}
                 for item in batch if item.num_completion_tokens > before[item.seq_id]], 0)

    def cancel_request(self, seq_id):
        return self.scheduler.cancel(seq_id)

    def forget_request(self, seq_id):
        self.requests.pop(seq_id, None)

    def exit(self):
        pass


def test_http_completion_and_metrics():
    service = Service("fake", max_pending=4, engine_factory=FakeEngine)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service, "fake"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        connection.request("POST", "/v1/completions",
                           body=json.dumps({"prompt": "hi", "max_tokens": 3}),
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        data = json.loads(response.read())
        assert response.status == 200
        assert data["choices"][0]["text"] == "xxx"
        assert data["usage"]["completion_tokens"] == 3
        connection.close()

        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        connection.request("GET", "/metrics")
        metrics = connection.getresponse().read().decode()
        assert "nanovllm_completed_requests_total 1" in metrics
        assert 'nanovllm_ttft_seconds_bucket{le="+Inf"} 1' in metrics
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        service.stop()
        thread.join(timeout=5)


def test_http_chat_stream():
    service = Service("fake", max_pending=4, engine_factory=FakeEngine)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service, "fake"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        connection.request("POST", "/v1/chat/completions",
                           body=json.dumps({"messages": [{"role": "user", "content": "hi"}],
                                            "max_tokens": 3, "stream": True}),
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        stream = response.read().decode()
        assert response.status == 200
        assert stream.count('"content": "x"') == 3
        assert "data: [DONE]" in stream
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        service.stop()
        thread.join(timeout=5)
