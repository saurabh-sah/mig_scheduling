"""Run matched MFI/baseline/fixed-lookahead comparisons.

All policies receive the exact same workload for each (scenario, seed). The
lookahead scheduler estimates future profile probabilities and durations from
past observations only; it does not receive future generated jobs.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mig_scheduler.schedulers import (  # noqa: E402
    BestFitBestIndexScheduler,
    FixedBudgetLookaheadScheduler,
    FirstFitScheduler,
    MFIScheduler,
    RoundRobinScheduler,
    WorstFitBestIndexScheduler,
)
from mig_scheduler.simulator import DiscreteEventSimulator  # noqa: E402
from mig_scheduler.workloads import generate_workload  # noqa: E402

SCENARIOS = ("uniform", "skew_small", "skew_big", "bimodal")
BASELINES = (FirstFitScheduler, RoundRobinScheduler, BestFitBestIndexScheduler,
             WorstFitBestIndexScheduler, MFIScheduler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=100)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--mean-interarrival", type=float, default=0.8)
    parser.add_argument("--min-duration", type=int, default=1)
    parser.add_argument("--max-duration", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--lookahead-seed", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("results/m3_lookahead_matrix.csv"))
    args = parser.parse_args()

    if args.jobs < 1 or args.gpus < 1:
        raise SystemExit("--jobs and --gpus must be >= 1")
    if args.horizon < 1 or args.rollouts < 1:
        raise SystemExit("--horizon and --rollouts must be >= 1")
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise SystemExit("Provide at least one seed")

    rows: list[dict[str, object]] = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        for seed in seeds:
            jobs = generate_workload(
                count=args.jobs, scenario=scenario, seed=seed,
                mean_interarrival=args.mean_interarrival,
                min_duration=args.min_duration, max_duration=args.max_duration,
            )
            schedulers = [factory() for factory in BASELINES]
            schedulers.append(FixedBudgetLookaheadScheduler(
                horizon=args.horizon,
                rollouts=args.rollouts,
                seed=args.lookahead_seed + scenario_index * 100_000 + seed,
            ))

            for scheduler in schedulers:
                result = DiscreteEventSimulator(args.gpus, scheduler).run(jobs)
                summary = result.summary()
                row: dict[str, object] = {
                    "scenario": scenario,
                    "seed": seed,
                    "jobs": args.jobs,
                    "gpus": args.gpus,
                    "mean_interarrival": args.mean_interarrival,
                    "min_duration": args.min_duration,
                    "max_duration": args.max_duration,
                    "horizon": args.horizon if scheduler.name == "fixed_lookahead" else 0,
                    "rollouts": args.rollouts if scheduler.name == "fixed_lookahead" else 0,
                    **summary,
                }
                rows.append(row)
                lookahead_mean = summary["mean_lookahead_value"]
                lookahead_label = f"{lookahead_mean:.3f}" if lookahead_mean is not None else "n/a"
                print(
                    f"{scenario:10s} seed={seed:4d} {scheduler.name:18s} "
                    f"accept={result.acceptance_rate:.3f} "
                    f"decision_us={result.mean_decision_time_ns / 1000:.1f} "
                    f"lookahead_mean={lookahead_label}"
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
        with args.out.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nSaved {len(rows)} matched runs to {args.out.resolve()}")


if __name__ == "__main__":
    main()
