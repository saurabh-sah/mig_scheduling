"""Command-line entry point for the baseline MIG simulator."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .schedulers import make_scheduler, FixedBudgetLookaheadScheduler, AdaptiveBudgetLookaheadScheduler
from .simulator import DiscreteEventSimulator
from .workloads import generate_workload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MIG scheduling baselines and lookahead policies.")
    parser.add_argument("--policy", choices=["mfi", "first_fit", "round_robin", "bf_bi", "wf_bi", "fixed_lookahead", "adaptive_lookahead"], default="mfi")
    parser.add_argument("--scenario", choices=["uniform", "skew_small", "skew_big", "bimodal"], default="skew_small")
    parser.add_argument("--gpus", type=int, default=10)
    parser.add_argument("--jobs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mean-interarrival", type=float, default=0.8)
    parser.add_argument("--min-duration", type=int, default=1)
    parser.add_argument("--max-duration", type=int, default=12)
    parser.add_argument("--out", type=Path, default=Path("results/latest"))
    parser.add_argument("--lookahead-horizon", type=int, default=3, help="Future arrivals simulated by lookahead policies")
    parser.add_argument("--lookahead-rollouts", type=int, default=8, help="Monte Carlo futures per candidate placement")
    parser.add_argument("--lookahead-seed", type=int, default=42, help="Independent RNG seed for lookahead rollouts")
    parser.add_argument(
        "--mfi-score-mode", choices=["all_legal_starts", "free_anchor_starts"],
        default="free_anchor_starts",
        help="MFI score interpretation used by lookahead current choices and rollout placements (default: Figure-3-conformance candidate)",
    )
    parser.add_argument(
        "--max-lookahead-candidates",
        type=int,
        default=None,
        help="Maximum candidate placements evaluated by lookahead (default: None, evaluate all feasible)",
    )
    parser.add_argument("--adaptive-min-rollouts", type=int, default=8)
    parser.add_argument("--adaptive-max-rollouts", type=int, default=32)
    parser.add_argument("--adaptive-batch-size", type=int, default=4)
    parser.add_argument("--adaptive-min-history", type=int, default=24)
    parser.add_argument("--adaptive-profile-window", type=int, default=60)
    parser.add_argument("--adaptive-min-reliability", type=float, default=0.35)
    parser.add_argument("--adaptive-ambiguity-margin", type=int, default=1)
    parser.add_argument("--adaptive-max-candidates", type=int, default=12)
    args = parser.parse_args()

    jobs = generate_workload(count=args.jobs, scenario=args.scenario, seed=args.seed,
                             mean_interarrival=args.mean_interarrival,
                             min_duration=args.min_duration, max_duration=args.max_duration)
    if args.policy == "fixed_lookahead":
        scheduler = FixedBudgetLookaheadScheduler(
            horizon=args.lookahead_horizon,
            rollouts=args.lookahead_rollouts,
            seed=args.lookahead_seed,
            score_mode=args.mfi_score_mode,
            max_candidates=args.max_lookahead_candidates,
        )
    elif args.policy == "adaptive_lookahead":
        scheduler = AdaptiveBudgetLookaheadScheduler(
            horizon=args.lookahead_horizon,
            min_rollouts=args.adaptive_min_rollouts,
            max_rollouts=args.adaptive_max_rollouts,
            batch_size=args.adaptive_batch_size,
            seed=args.lookahead_seed,
            minimum_history=args.adaptive_min_history,
            profile_window=args.adaptive_profile_window,
            minimum_reliability=args.adaptive_min_reliability,
            ambiguity_margin=args.adaptive_ambiguity_margin,
            max_candidates=args.max_lookahead_candidates if args.max_lookahead_candidates is not None else args.adaptive_max_candidates,
            score_mode=args.mfi_score_mode,
        )
    else:
        scheduler = make_scheduler(args.policy)
    result = DiscreteEventSimulator(args.gpus, scheduler).run(jobs)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(result.summary(), indent=2), encoding="utf-8")

    for filename, rows in (("jobs.csv", result.job_results), ("snapshots.csv", result.snapshots)):
        path = args.out / filename
        if rows:
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
        else:
            path.write_text("", encoding="utf-8")
    (args.out / "event_log.txt").write_text("\n".join(result.event_log) + "\n", encoding="utf-8")
    print(json.dumps(result.summary(), indent=2))
    print(f"\nWrote results to: {args.out.resolve()}")


if __name__ == "__main__":
    main()
