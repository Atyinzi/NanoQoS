import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm

from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler


class LLMEngine:

    def __init__(self, model, **kwargs):
        # GPU dependencies are loaded only when an actual model is constructed.
        from transformers import AutoTokenizer
        import torch.multiprocessing as mp
        from nanovllm.config import Config
        from nanovllm.engine.model_runner import ModelRunner
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.max_model_len = config.max_model_len
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self.requests: dict[int, Sequence] = {}
        self.closed = False
        atexit.register(self.exit)

    def exit(self):
        if self.closed:
            return
        self.closed = True
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams,
                    deadline_seconds: float | None = None, priority: int = 0,
                    arrival_time: float | None = None) -> int:
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        if len(prompt) + sampling_params.max_tokens > self.max_model_len:
            raise ValueError("prompt plus max_tokens exceeds max_model_len")
        kv_capacity = self.scheduler.block_size * len(self.scheduler.block_manager.blocks)
        # The last generated token is returned without another decode step.
        # All earlier tokens must fit in KV even if max_tokens is reached.
        if len(prompt) + sampling_params.max_tokens - 1 > kv_capacity:
            raise ValueError("prompt plus generation cannot fit in the KV cache")
        seq = Sequence(prompt, sampling_params, arrival_time=arrival_time,
                       deadline_seconds=deadline_seconds, priority=priority)
        self.scheduler.add(seq)
        self.requests[seq.seq_id] = seq
        return seq.seq_id

    def cancel_request(self, seq_id: int) -> bool:
        return self.scheduler.cancel(seq_id)

    def forget_request(self, seq_id: int):
        self.requests.pop(seq_id, None)

    def step_events(self) -> tuple[list[dict], int]:
        """Run one scheduler iteration and return newly generated token events."""
        if self.is_finished():
            return [], 0
        seqs, is_prefill = self.scheduler.schedule()
        before = {seq.seq_id: seq.num_completion_tokens for seq in seqs}
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        events = []
        for seq in seqs:
            if seq.num_completion_tokens > before[seq.seq_id]:
                events.append({
                    "seq_id": seq.seq_id,
                    "token_id": seq.last_token,
                    "finished": seq.is_finished,
                    "timestamp": seq.token_times[-1],
                })
        return events, num_tokens

    def step(self):
        events, num_tokens = self.step_events()
        outputs = [(event["seq_id"], self.requests[event["seq_id"]].completion_token_ids)
                   for event in events if event["finished"]]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        if len(sampling_params) != len(prompts):
            raise ValueError("prompts and sampling_params must have the same length")
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                self.forget_request(seq_id)
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
