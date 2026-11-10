"""Matched workload-budget frontier for MFI, fixed lookahead, and adaptive lookahead."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

from mig_scheduler.schedulers import (
    AdaptiveBudgetLookaheadScheduler,
    FirstFitScheduler,
    FixedBudgetLookaheadScheduler,
    MFIScheduler,
)
from mig_scheduler.simulator import DiscreteEventSimulator
from mig_scheduler.workloads import generate_regime_shift_workload, generate_workload

SCENARIOS = (
    "uniform", "skew_small", "skew_big", "bimodal",
    "shift_small_to_big", "shift_big_to_small",
)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_jobs(scenario: str, *, jobs: int, seed: int, mean_interarrival: float,
              min_duration: int, max_duration: int, shift_fraction: float):
    kwargs = dict(count=jobs, seed=seed, mean_interarrival=mean_interarrival,
                  min_duration=min_duration, max_duration=max_duration)
    if scenario == "shift_small_to_big":
        return generate_regime_shift_workload(
            **kwargs, before_scenario="skew_small", after_scenario="skew_big",
            shift_fraction=shift_fraction,
        )
    if scenario == "shift_big_to_small":
        return generate_regime_shift_workload(
            **kwargs, before_scenario="skew_big", after_scenario="skew_small",
            shift_fraction=shift_fraction,
        )
    return generate_workload(scenario=scenario, **kwargs)


def build_policies(seed: int, horizon: int, budgets: list[int]):
    policies = [("first_fit", FirstFitScheduler()), ("mfi", MFIScheduler())]
    for budget in budgets:
        policies.append((f"fixed_h{horizon}_r{budget}", FixedBudgetLookaheadScheduler(
            horizon=horizon, rollouts=budget, seed=seed,
        )))
    for budget in budgets:
        policies.append((f"adaptive_h{horizon}_max{budget}", AdaptiveBudgetLookaheadScheduler(
            horizon=horizon, min_rollouts=2, max_rollouts=max(2, budget),
            batch_size=2, seed=seed, minimum_history=12, profile_window=32,
            stability_tv_threshold=0.60, minimum_reliability=0.35,
            ambiguity_margin=1, max_candidates=6,
        )))
    return policies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=40)
    parser.add_argument("--gpus", type=int, default=3)
    parser.add_argument("--seeds", default="11,22,33")
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--budgets", default="2,4,8")
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--mean-interarrival", type=float, default=0.8)
    parser.add_argument("--min-duration", type=int, default=1)
    parser.add_argument("--max-duration", type=int, default=12)
    parser.add_argument("--shift-fraction", type=float, default=0.5)
    parser.add_argument("--seed-offset", type=int, default=700_000)
    parser.add_argument("--out", type=Path, default=Path("results/m6_budget_frontier"))
    args = parser.parse_args()

    if args.jobs < 1 or args.gpus < 1 or args.horizon < 1:
        raise SystemExit("jobs, gpus, and horizon must be >= 1")
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    scenarios = [x.strip() for x in args.scenarios.split(",") if x.strip()]
    budgets = sorted(set(int(x.strip()) for x in args.budgets.split(",") if x.strip()))
    if not seeds or not scenarios or not budgets or any(x < 2 for x in budgets):
        raise SystemExit("Provide seeds/scenarios and rollout budgets >= 2")
    unknown = set(scenarios) - set(SCENARIOS)
    if unknown:
        raise SystemExit(f"Unknown scenarios {sorted(unknown)}; use {SCENARIOS}")

    args.out.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["out"] = str(args.out)
    config["seeds"] = seeds
    config["scenarios"] = scenarios
    config["budgets"] = budgets
    (args.out / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    rows: list[dict] = []
    total_runs = len(scenarios) * len(seeds) * (2 + 2 * len(budgets))
    completed = 0
    for scenario_index, scenario in enumerate(scenarios):
        for seed in seeds:
            workload_seed = seed + scenario_index * 1_000_003
            jobs = make_jobs(
                scenario, jobs=args.jobs, seed=workload_seed,
                mean_interarrival=args.mean_interarrival,
                min_duration=args.min_duration, max_duration=args.max_duration,
                shift_fraction=args.shift_fraction,
            )
            for policy_name, scheduler in build_policies(
                seed + args.seed_offset + scenario_index * 100_000, args.horizon, budgets
            ):
                start = perf_counter()
                result = DiscreteEventSimulator(args.gpus, scheduler).run(jobs)
                elapsed = perf_counter() - start
                summary = result.summary()
                placement_rows = [
                    row for row in result.job_results
                    if row.get("gpu_id") is not None
                    and row.get("start") is not None
                    and row.get("mfi_gpu_id") is not None
                    and row.get("mfi_start") is not None
                ]
                decisions_different_from_mfi = sum(
                    (int(row["gpu_id"]), int(row["start"]))
                    != (int(row["mfi_gpu_id"]), int(row["mfi_start"]))
                    for row in placement_rows
                )
                total_rollout_evaluations = sum(
                    int(row.get("total_rollouts") or 0) for row in result.job_results
                )
                rows.append({
                    "scenario": scenario, "seed": seed, "workload_seed": workload_seed,
                    "policy": policy_name, "jobs": len(jobs), "gpus": args.gpus,
                    "accepted": result.accepted, "rejected": result.rejected,
                    "acceptance_rate": result.acceptance_rate,
                    "decision_time_total_ms": result.total_decision_time_ns / 1e6,
                    "decision_time_mean_us": result.mean_decision_time_ns / 1e3,
                    "wall_clock_run_s": elapsed,
                    "mean_fragmentation_score_time_weighted": result.mean_time_weighted_fragmentation_score,
                    "peak_fragmentation_score": result.peak_fragmentation_score,
                    "lookahead_decisions": summary["lookahead_decisions"],
                    "adaptive_lookahead_decisions": summary["adaptive_lookahead_decisions"],
                    "adaptive_fallback_decisions": summary["adaptive_mfi_fallback_decisions"],
                    "total_rollout_evaluations": total_rollout_evaluations,
                    "rollout_evaluations_per_arrival": total_rollout_evaluations / max(result.submitted, 1),
                    "mfi_comparable_placements": len(placement_rows),
                    "placements_different_from_mfi": decisions_different_from_mfi,
                    "fraction_comparable_placements_different_from_mfi": (
                        decisions_different_from_mfi / len(placement_rows) if placement_rows else 0.0
                    ),
                    "mean_adaptive_forecast_reliability": summary["adaptive_mean_forecast_reliability"],
                })
                completed += 1
                if completed % 10 == 0 or completed == total_runs:
                    print(f"Completed {completed}/{total_runs} runs", flush=True)

    write_csv(args.out / "budget_frontier_runs.csv", rows)
    print(f"Wrote {len(rows)} matched runs to {args.out.resolve() / 'budget_frontier_runs.csv'}")


if __name__ == "__main__":
    main()
