"""Run matched baseline, fixed-lookahead, and adaptive-lookahead experiments.

Every policy sees the exact same workload for each (scenario, seed) pair.
This is an exploratory simulator experiment; it is not an exact reproduction
of any paper's complete evaluation protocol.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mig_scheduler.schedulers import (  # noqa: E402
    AdaptiveBudgetLookaheadScheduler,
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
BASELINES = (
    FirstFitScheduler,
    RoundRobinScheduler,
    BestFitBestIndexScheduler,
    WorstFitBestIndexScheduler,
    MFIScheduler,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=60)
    parser.add_argument("--gpus", type=int, default=3)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--mean-interarrival", type=float, default=0.8)
    parser.add_argument("--min-duration", type=int, default=1)
    parser.add_argument("--max-duration", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--fixed-rollouts", type=int, default=4)
    parser.add_argument("--adaptive-min-rollouts", type=int, default=4)
    parser.add_argument("--adaptive-max-rollouts", type=int, default=12)
    parser.add_argument("--adaptive-batch-size", type=int, default=4)
    parser.add_argument("--adaptive-min-history", type=int, default=18)
    parser.add_argument("--adaptive-profile-window", type=int, default=40)
    parser.add_argument("--adaptive-min-reliability", type=float, default=0.35)
    parser.add_argument("--adaptive-ambiguity-margin", type=int, default=1)
    parser.add_argument("--adaptive-max-candidates", type=int, default=8)
    parser.add_argument("--seed-offset", type=int, default=100_000)
    parser.add_argument("--out", type=Path, default=Path("results/m4_adaptive_matrix.csv"))
    args = parser.parse_args()

    if args.jobs < 1 or args.gpus < 1:
        raise SystemExit("--jobs and --gpus must be >= 1")
    if args.horizon < 1 or args.fixed_rollouts < 1:
        raise SystemExit("--horizon and --fixed-rollouts must be >= 1")
    if args.adaptive_min_rollouts < 2 or args.adaptive_max_rollouts < args.adaptive_min_rollouts:
        raise SystemExit("Adaptive rollout limits are invalid")
    if args.adaptive_batch_size < 1:
        raise SystemExit("--adaptive-batch-size must be >= 1")
    seeds = [int(piece.strip()) for piece in args.seeds.split(",") if piece.strip()]
    if not seeds:
        raise SystemExit("Provide at least one seed")

    rows: list[dict[str, object]] = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        for seed in seeds:
            jobs = generate_workload(
                count=args.jobs,
                scenario=scenario,
                seed=seed,
                mean_interarrival=args.mean_interarrival,
                min_duration=args.min_duration,
                max_duration=args.max_duration,
            )
            policies = [factory() for factory in BASELINES]
            policies.append(FixedBudgetLookaheadScheduler(
                horizon=args.horizon,
                rollouts=args.fixed_rollouts,
                seed=args.seed_offset + scenario_index * 100_000 + seed,
            ))
            policies.append(AdaptiveBudgetLookaheadScheduler(
                horizon=args.horizon,
                min_rollouts=args.adaptive_min_rollouts,
                max_rollouts=args.adaptive_max_rollouts,
                batch_size=args.adaptive_batch_size,
                seed=args.seed_offset + scenario_index * 100_000 + seed + 50_000,
                minimum_history=args.adaptive_min_history,
                profile_window=args.adaptive_profile_window,
                minimum_reliability=args.adaptive_min_reliability,
                ambiguity_margin=args.adaptive_ambiguity_margin,
                max_candidates=args.adaptive_max_candidates,
            ))

            for scheduler in policies:
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
                    "horizon": args.horizon if "lookahead" in scheduler.name else 0,
                    "fixed_rollouts": args.fixed_rollouts if scheduler.name == "fixed_lookahead" else 0,
                    "adaptive_min_rollouts": args.adaptive_min_rollouts if scheduler.name == "adaptive_lookahead" else 0,
                    "adaptive_max_rollouts": args.adaptive_max_rollouts if scheduler.name == "adaptive_lookahead" else 0,
                    **summary,
                }
                rows.append(row)
                if scheduler.name == "adaptive_lookahead":
                    rollout_text = f"{summary['adaptive_mean_total_rollouts']:.2f}"
                elif scheduler.name == "fixed_lookahead":
                    rollout_text = f"{summary['mean_lookahead_rollouts_per_decision']:.2f}"
                else:
                    rollout_text = "n/a"
                print(
                    f"{scenario:10s} seed={seed:4d} {scheduler.name:20s} "
                    f"accept={result.acceptance_rate:.3f} "
                    f"decision_us={result.mean_decision_time_ns / 1000:.1f} "
                    f"rollouts={rollout_text}"
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with args.out.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nSaved {len(rows)} matched runs to {args.out.resolve()}")


if __name__ == "__main__":
    main()
