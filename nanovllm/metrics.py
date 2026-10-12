"""Request metrics shared by GPU benchmarks and the CPU scheduler simulation."""

from math import ceil, floor


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    if not 0 <= percent <= 100:
        raise ValueError("percent must be between 0 and 100")
    ordered = sorted(values)
    index = (len(ordered) - 1) * percent / 100
    lower, upper = floor(index), ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def summarize(sequences, elapsed_seconds: float) -> dict:
    completed = [seq for seq in sequences if seq.finish_time is not None]
    canceled = [seq for seq in sequences if seq.cancel_time is not None]
    ttft = [seq.token_times[0] - seq.arrival_time for seq in completed if seq.token_times]
    tpot = [later - earlier for seq in completed
            for earlier, later in zip(seq.token_times, seq.token_times[1:])]
    e2e = [seq.finish_time - seq.arrival_time for seq in completed]
    waiting = [seq.token_times[0] - seq.arrival_time for seq in completed if seq.token_times]
    output_tokens = sum(seq.num_completion_tokens for seq in completed)
    prompt_tokens = sum(seq.num_prompt_tokens for seq in completed)
    cached_tokens = sum(seq.prefix_cached_tokens for seq in completed)
    deadlines = [seq for seq in completed if seq.deadline_time is not None]
    met = [seq for seq in deadlines if seq.finish_time <= seq.deadline_time]

    def distribution(values):
        return {"mean_ms": 1000 * sum(values) / len(values) if values else None,
                "p50_ms": 1000 * percentile(values, 50) if values else None,
                "p95_ms": 1000 * percentile(values, 95) if values else None,
                "p99_ms": 1000 * percentile(values, 99) if values else None}

    return {
        "submitted_requests": len(sequences),
        "completed_requests": len(completed),
        "canceled_requests": len(canceled),
        "elapsed_seconds": elapsed_seconds,
        "output_tokens": output_tokens,
        "output_tokens_per_second": output_tokens / elapsed_seconds if elapsed_seconds > 0 else None,
        "ttft": distribution(ttft),
        "tpot": distribution(tpot),
        "e2e": distribution(e2e),
        "waiting_to_first_token": distribution(waiting),
        "prefix_cached_tokens": cached_tokens,
        "prefix_cache_hit_fraction": cached_tokens / prompt_tokens if prompt_tokens else None,
        "preemptions": sum(seq.preemptions for seq in sequences),
        "recomputed_tokens": sum(seq.recomputed_tokens for seq in sequences),
        "deadline_requests": len(deadlines),
        "deadline_met_requests": len(met),
        "deadline_success_fraction": len(met) / len(deadlines) if deadlines else None,
        "slo_goodput_tokens_per_second": sum(seq.num_completion_tokens for seq in met) / elapsed_seconds
        if deadlines and elapsed_seconds > 0 else None,
    }
