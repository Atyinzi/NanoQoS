from time import perf_counter
from types import SimpleNamespace
import pytest

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.metrics import percentile, summarize
from nanovllm.sampling_params import SamplingParams


def scheduler(policy="fcfs", blocks=32, budget=64):
    Sequence.block_size = 8
    return Scheduler(SimpleNamespace(
        max_num_seqs=8, max_num_batched_tokens=budget, eos=-1,
        kvcache_block_size=8, num_kvcache_blocks=blocks,
        scheduler_policy=policy, aging_seconds=5.0,
        max_consecutive_prefills=4,
    ))


def seq(length, output=2, **kwargs):
    return Sequence(list(range(1, length + 1)),
                    SamplingParams(max_tokens=output, ignore_eos=True), **kwargs)


def test_prefix_reuse_and_release():
    service = scheduler()
    first = seq(10, output=1)
    service.add(first)
    batch, prefill = service.schedule()
    assert prefill
    service.postprocess(batch, [99], prefill)
    assert first.is_finished
    assert len(service.block_manager.free_block_ids) == 32

    second = seq(10, output=1)
    service.add(second)
    batch, prefill = service.schedule()
    assert second.prefix_cached_tokens == 8
    assert second.num_cached_tokens == 8
    assert second.num_scheduled_tokens == 2
    service.postprocess(batch, [99], prefill)
    assert second.is_finished


def test_shortest_and_aging_change_order():
    service = scheduler("shortest", budget=8)
    long = seq(20)
    short = seq(3)
    service.add(long)
    service.add(short)
    batch, _ = service.schedule()
    assert batch == [short]

    aged = scheduler("shortest", budget=8)
    old = seq(20, arrival_time=perf_counter() - 100)
    fresh = seq(3)
    aged.add(old)
    aged.add(fresh)
    batch, _ = aged.schedule()
    assert batch == [old]


def test_slo_deadline_order_and_cancellation():
    service = scheduler("slo", budget=8)
    later = seq(3, deadline_seconds=20)
    earlier = seq(3, deadline_seconds=2)
    service.add(later)
    service.add(earlier)
    batch, _ = service.schedule()
    assert batch[0] is earlier
    assert service.cancel(later.seq_id)
    assert later.cancel_time is not None
    assert not service.cancel(later.seq_id)


def test_preemption_releases_blocks_and_counts_recompute():
    service = scheduler(blocks=2)
    first = seq(8, output=2)
    second = seq(8, output=2)
    service.add(first)
    service.add(second)
    batch, prefill = service.schedule()
    service.postprocess(batch, [99, 99], prefill)
    assert len(service.block_manager.free_block_ids) == 0
    batch, prefill = service.schedule()
    assert not prefill
    assert len(batch) == 1
    assert sum(item.preemptions for item in (first, second)) == 1
    assert len(service.block_manager.used_block_ids) <= 2


def test_metrics_are_request_level_and_deadline_aware():
    item = seq(4, output=2, arrival_time=10.0, deadline_seconds=0.5)
    item.record_token(10.1)
    item.record_token(10.3)
    item.append_token(9)
    item.append_token(9)
    item.finish_time = 10.4
    result = summarize([item], 1.0)
    assert result["ttft"]["p99_ms"] == pytest.approx(100.0)
    assert result["tpot"]["p50_ms"] == pytest.approx(200.0)
    assert result["deadline_success_fraction"] == 1.0
    assert result["slo_goodput_tokens_per_second"] == 2.0
    assert percentile([], 95) is None


def test_engine_emits_one_event_per_generated_token():
    class Runner:
        def call(self, method, batch, prefill):
            assert method == "run"
            return [77] * len(batch)

    engine = object.__new__(LLMEngine)
    engine.scheduler = scheduler()
    engine.model_runner = Runner()
    engine.requests = {}
    item = seq(3, output=2)
    engine.requests[item.seq_id] = item
    engine.scheduler.add(item)
    first, tokens = engine.step_events()
    assert tokens == 3
    assert first[0]["token_id"] == 77
    assert not first[0]["finished"]
    second, tokens = engine.step_events()
    assert tokens == -1
    assert second[0]["finished"]
    assert item.finish_time is not None
    assert engine.step_events() == ([], 0)
