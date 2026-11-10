"""Run matched policy comparisons on Poisson-arrival synthetic workloads.

This is a controlled simulator benchmark, not an exact reproduction of MFI's
Section VI load-sweep protocol. See EXPERIMENT_PROTOCOL.md.
"""
from __future__ import annotations

# Allow running directly from a source checkout before editable installation.
import sys
from pathlib import Path as _BootstrapPath
sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1] / "src"))

import argparse
import csv
from pathlib import Path

from mig_scheduler.schedulers import (
    FirstFitScheduler, RoundRobinScheduler, BestFitBestIndexScheduler,
    WorstFitBestIndexScheduler, MFIScheduler,
)
from mig_scheduler.simulator import DiscreteEventSimulator
from mig_scheduler.workloads import generate_workload

POLICIES = (FirstFitScheduler, RoundRobinScheduler, BestFitBestIndexScheduler,
            WorstFitBestIndexScheduler, MFIScheduler)
SCENARIOS = ("uniform", "skew_small", "skew_big", "bimodal")


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched baseline matrix for MIG schedulers")
    parser.add_argument("--jobs", type=int, default=200)
    parser.add_argument("--gpus", type=int, default=10)
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--mean-interarrival", type=float, default=0.8)
    parser.add_argument("--min-duration", type=int, default=1)
    parser.add_argument("--max-duration", type=int, default=12)
    parser.add_argument("--out", type=Path, default=Path("results/m2_baseline_matrix.csv"))
    args = parser.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise SystemExit("Provide at least one seed")

    rows = []
    for scenario in SCENARIOS:
        for seed in seeds:
            jobs = generate_workload(count=args.jobs, scenario=scenario, seed=seed,
                                     mean_interarrival=args.mean_interarrival,
                                     min_duration=args.min_duration,
                                     max_duration=args.max_duration)
            for policy_cls in POLICIES:
                result = DiscreteEventSimulator(args.gpus, policy_cls()).run(jobs)
                rows.append({"scenario": scenario, "seed": seed, **result.summary()})
                print(f"{scenario:10s} seed={seed:4d} policy={result.policy:8s} "
                      f"accept={result.acceptance_rate:.3f} "
                      f"compute_util={result.summary()['mean_arrival_snapshot_compute_utilization']:.3f} "
                      f"mean_decision_ns={result.mean_decision_time_ns:.0f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader(); writer.writerows(rows)
    print(f"Saved {len(rows)} rows: {args.out.resolve()}")


if __name__ == "__main__":
    main()
