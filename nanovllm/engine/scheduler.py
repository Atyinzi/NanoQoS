from __future__ import annotations

from collections import deque
from time import perf_counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.policy = config.scheduler_policy
        self.aging_seconds = config.aging_seconds
        self.max_consecutive_prefills = config.max_consecutive_prefills
        self.consecutive_prefills = 0

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def cancel(self, seq_id: int) -> bool:
        for queue in (self.waiting, self.running):
            for seq in queue:
                if seq.seq_id == seq_id:
                    queue.remove(seq)
                    if seq.block_table:
                        self.block_manager.deallocate(seq)
                    seq.cancel_time = perf_counter()
                    seq.status = SequenceStatus.FINISHED
                    return True
        return False

    def _order_waiting(self):
        if self.policy == "fcfs" or len(self.waiting) < 2:
            return
        now = perf_counter()

        def score(seq: Sequence):
            if seq.block_table:
                return float("-inf")
            waited = max(0.0, now - seq.arrival_time)
            age_bonus = waited / self.aging_seconds * self.max_num_batched_tokens
            remaining = seq.num_tokens - seq.num_cached_tokens
            if self.policy == "shortest":
                return remaining - age_bonus - seq.priority * self.max_num_batched_tokens
            deadline = seq.deadline_time if seq.deadline_time is not None else float("inf")
            return deadline - now - age_bonus / self.max_num_batched_tokens - seq.priority * self.aging_seconds

        self.waiting = deque(sorted(self.waiting, key=lambda seq: (score(seq), seq.arrival_time, seq.seq_id)))

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # Prefill is periodically deferred so active decode requests can progress.
        self._order_waiting()
        allow_prefill = not self.running or self.consecutive_prefills < self.max_consecutive_prefills
        while allow_prefill and self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
                if seq.preemptions == 0:
                    seq.prefix_cached_tokens = min(
                        num_cached_blocks * self.block_size, seq.num_prompt_tokens)
                if seq._pending_recompute_tokens:
                    seq.recomputed_tokens += max(
                        0, seq._pending_recompute_tokens - num_cached_blocks * self.block_size)
                    seq._pending_recompute_tokens = 0
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            self.consecutive_prefills += 1
            return scheduled_seqs, True

        # decode
        self.consecutive_prefills = 0
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        if not scheduled_seqs:
            raise RuntimeError("No request can progress with the available KV cache blocks")
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.preemptions += 1
        seq._pending_recompute_tokens = seq.num_cached_tokens
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            seq.record_token()
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                seq.finish_time = perf_counter()
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
