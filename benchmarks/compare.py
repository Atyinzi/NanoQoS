"""Compare JSON files produced by benchmarks.run with identical workloads."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args()
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in args.results]
    reference = runs[0]
    for run in runs[1:]:
        if any(run[key] != reference[key] for key in ("backend", "model", "workload", "engine")):
            parser.error("all result files must have the same backend, model, workload and engine settings")
    print("policy       tok/s    TTFT p99 ms    TPOT p99 ms    SLO goodput tok/s    preemptions")
    for run in runs:
        metrics = run["metrics"]
        def value(number):
            return "n/a" if number is None else f"{number:.2f}"
        print(f"{run['policy']:<10} {value(metrics['output_tokens_per_second']):>9} "
              f"{value(metrics['ttft']['p99_ms']):>14} "
              f"{value(metrics['tpot']['p99_ms']):>14} "
              f"{value(metrics['slo_goodput_tokens_per_second']):>19} "
              f"{metrics['preemptions']:>12}")


if __name__ == "__main__":
    main()
