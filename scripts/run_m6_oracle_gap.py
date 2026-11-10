"""Compare online policies with an exact offline oracle on tiny traces.

The oracle knows the complete fixed trace and is used as an upper bound only.
Default workloads are intentionally small because exact search is exponential.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter

from mig_scheduler.offline_oracle import OracleBudgetExceeded, solve_offline_optimal
from mig_scheduler.schedulers import (
    AdaptiveBudgetLookaheadScheduler,
    FirstFitScheduler,
    FixedBudgetLookaheadScheduler,
    MFIScheduler,
)
from mig_scheduler.simulator import DiscreteEventSimulator
from mig_scheduler.workloads import generate_workload


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return (float("nan"), float("nan"), float("nan"))
    mu = mean(values)
    if len(values) < 2:
        return (mu, float("nan"), float("nan"))
    se = stdev(values) / math.sqrt(len(values))
    # Approximate 95% normal interval; the report labels it exploratory.
    return mu, mu - 1.96 * se, mu + 1.96 * se


def policies(seed: int, horizon: int, fixed_rollouts: int, adaptive_cap: int):
    return [
        ("first_fit", FirstFitScheduler()),
        ("mfi", MFIScheduler()),
        (f"fixed_h{horizon}_r{fixed_rollouts}", FixedBudgetLookaheadScheduler(
            horizon=horizon, rollouts=fixed_rollouts, seed=seed,
        )),
        (f"adaptive_h{horizon}_max{adaptive_cap}", AdaptiveBudgetLookaheadScheduler(
            horizon=horizon,
            min_rollouts=min(2, adaptive_cap),
            max_rollouts=adaptive_cap,
            batch_size=min(2, adaptive_cap),
            seed=seed,
            minimum_history=4,
            profile_window=12,
            stability_tv_threshold=0.60,
            minimum_reliability=0.20,
            ambiguity_margin=1,
            max_candidates=4,
            minimum_effect=0.0,
        )),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--seeds", default="11,22,33")
    parser.add_argument("--scenarios", default="uniform,skew_small,skew_big,bimodal")
    parser.add_argument("--mean-interarrival", type=float, default=0.7)
    parser.add_argument("--min-duration", type=int, default=2)
    parser.add_argument("--max-duration", type=int, default=8)
    parser.add_argument("--max-jobs", type=int, default=12)
    parser.add_argument("--max-states", type=int, default=250000)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--fixed-rollouts", type=int, default=2)
    parser.add_argument("--adaptive-cap", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("results/m6_oracle_gap"))
    args = parser.parse_args()

    if args.jobs < 0 or args.gpus < 1:
        raise SystemExit("--jobs must be >= 0 and --gpus >= 1")
    if args.jobs > args.max_jobs:
        raise SystemExit("--jobs cannot exceed --max-jobs for exact search")
    if args.max_states < 1:
        raise SystemExit("--max-states must be >= 1")
    if args.horizon < 1 or args.fixed_rollouts < 1 or args.adaptive_cap < 2:
        raise SystemExit("Horizon/rollouts must be positive and adaptive-cap >= 2")

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    scenarios = [x.strip() for x in args.scenarios.split(",") if x.strip()]
    if not seeds or not scenarios:
        raise SystemExit("At least one seed and scenario are required")

    args.out.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["out"] = str(args.out)
    config["seeds"] = seeds
    config["scenarios"] = scenarios
    (args.out / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    run_rows: list[dict] = []
    oracle_rows: list[dict] = []
    oracle_decisions: list[dict] = []
    completed_traces = 0
    skipped_traces = 0
    overall_start = perf_counter()

    for scenario_index, scenario in enumerate(scenarios):
        for seed in seeds:
            workload_seed = seed + scenario_index * 1_000_003
            jobs = generate_workload(
                count=args.jobs,
                scenario=scenario,
                seed=workload_seed,
                mean_interarrival=args.mean_interarrival,
                min_duration=args.min_duration,
                max_duration=args.max_duration,
            )
            trace_id = f"{scenario}-seed{seed}"
            try:
                oracle = solve_offline_optimal(
                    jobs, args.gpus, max_jobs=args.max_jobs, max_states=args.max_states,
                )
            except OracleBudgetExceeded as exc:
                skipped_traces += 1
                oracle_rows.append({
                    "trace_id": trace_id, "scenario": scenario, "seed": seed,
                    "workload_seed": workload_seed, "jobs": len(jobs),
                    "status": "state_budget_exceeded", "error": str(exc),
                })
                continue

            completed_traces += 1
            oracle_rows.append({
                "trace_id": trace_id, "scenario": scenario, "seed": seed,
                "workload_seed": workload_seed, "jobs": len(jobs),
                "status": "complete", "oracle_accepted": oracle.optimal_accepted,
                "oracle_rejected": oracle.optimal_rejected,
                "oracle_acceptance_rate": oracle.acceptance_rate,
                "oracle_runtime_ms": oracle.runtime_ns / 1e6,
                "oracle_states_explored": oracle.states_explored,
                "oracle_transitions_considered": oracle.transitions_considered,
            })
            for d in oracle.decisions:
                oracle_decisions.append({
                    "trace_id": trace_id, "scenario": scenario, "seed": seed,
                    "job_id": d.job_id, "arrival_time": d.arrival_time,
                    "profile": d.profile_name, "accepted": int(d.accepted),
                    "gpu_id": d.gpu_id, "start": d.start,
                })

            for policy_name, scheduler in policies(
                seed + 55_000 + scenario_index * 1000,
                args.horizon, args.fixed_rollouts, args.adaptive_cap,
            ):
                result = DiscreteEventSimulator(args.gpus, scheduler).run(jobs)
                acc = result.accepted
                gap = oracle.optimal_accepted - acc
                # Given a correct oracle, this must not be negative.
                if gap < 0:
                    raise RuntimeError(
                        f"Oracle inconsistency on {trace_id}/{policy_name}: "
                        f"policy accepted {acc}, oracle accepted {oracle.optimal_accepted}"
                    )
                run_rows.append({
                    "trace_id": trace_id, "scenario": scenario, "seed": seed,
                    "workload_seed": workload_seed, "policy": policy_name,
                    "submitted": result.submitted, "accepted": acc,
                    "rejected": result.rejected, "acceptance_rate": result.acceptance_rate,
                    "oracle_accepted": oracle.optimal_accepted,
                    "oracle_acceptance_rate": oracle.acceptance_rate,
                    "oracle_gap_jobs": gap,
                    "oracle_gap_fraction_of_optimum": gap / max(oracle.optimal_accepted, 1),
                    "decision_time_total_ms": result.total_decision_time_ns / 1e6,
                    "decision_time_mean_us": result.mean_decision_time_ns / 1e3,
                    "lookahead_decisions": result.summary()["lookahead_decisions"],
                    "adaptive_lookahead_decisions": result.summary()["adaptive_lookahead_decisions"],
                    "adaptive_fallback_decisions": result.summary()["adaptive_mfi_fallback_decisions"],
                    "mean_total_rollouts": result.summary()["mean_lookahead_rollouts_per_decision"],
                    "oracle_states_explored": oracle.states_explored,
                    "oracle_runtime_ms": oracle.runtime_ns / 1e6,
                })

    write_csv(args.out / "online_vs_oracle_runs.csv", run_rows)
    write_csv(args.out / "oracle_trace_summary.csv", oracle_rows)
    write_csv(args.out / "oracle_decisions.csv", oracle_decisions)

    # Aggregate by policy and by scenario. Each row is a matched fixed-trace run.
    policy_summary = []
    policy_names = sorted({r["policy"] for r in run_rows})
    for name in policy_names:
        rows = [r for r in run_rows if r["policy"] == name]
        rate, lo, hi = mean_ci([float(r["acceptance_rate"]) for r in rows])
        gap, gap_lo, gap_hi = mean_ci([float(r["oracle_gap_jobs"]) for r in rows])
        time_ms, time_lo, time_hi = mean_ci([float(r["decision_time_total_ms"]) for r in rows])
        policy_summary.append({
            "policy": name, "trace_count": len(rows),
            "mean_acceptance_rate": rate, "acceptance_ci95_low_normal_approx": lo,
            "acceptance_ci95_high_normal_approx": hi,
            "mean_oracle_gap_jobs": gap, "oracle_gap_ci95_low_normal_approx": gap_lo,
            "oracle_gap_ci95_high_normal_approx": gap_hi,
            "mean_oracle_gap_fraction": mean([float(r["oracle_gap_fraction_of_optimum"]) for r in rows]) if rows else 0,
            "mean_total_decision_time_ms": time_ms,
            "decision_time_ci95_low_normal_approx_ms": time_lo,
            "decision_time_ci95_high_normal_approx_ms": time_hi,
            "mean_rollouts_metric": mean([float(r["mean_total_rollouts"] or 0) for r in rows]) if rows else 0,
        })
    write_csv(args.out / "policy_summary.csv", policy_summary)

    scenario_summary = []
    for scenario in scenarios:
        for name in policy_names:
            rows = [r for r in run_rows if r["scenario"] == scenario and r["policy"] == name]
            if not rows:
                continue
            scenario_summary.append({
                "scenario": scenario, "policy": name, "trace_count": len(rows),
                "mean_acceptance_rate": mean([float(r["acceptance_rate"]) for r in rows]),
                "mean_oracle_gap_jobs": mean([float(r["oracle_gap_jobs"]) for r in rows]),
                "mean_decision_time_ms": mean([float(r["decision_time_total_ms"]) for r in rows]),
            })
    write_csv(args.out / "scenario_summary.csv", scenario_summary)

    report = [
        "# M6 Exact Offline Oracle — Exploratory Report",
        "",
        "## Scope and semantics",
        "",
        "The oracle knows the complete fixed trace (all arrival times, profiles, and durations) and searches for the maximum number of accepted jobs under the simulator's current model. It processes arrivals in stable time order; jobs finishing at an arrival timestamp are released first; requests are accepted or rejected immediately; accepted jobs are not migrated; and no job can be delayed or retried.",
        "",
        "**This is an offline upper bound for the current simulator model, not an implementable online scheduler and not a hardware MIG emulator.** It does not prove an optimal schedule for real GPUs.",
        "",
        "## Run configuration",
        "",
        f"- Jobs per trace: {args.jobs}",
        f"- GPU count: {args.gpus}",
        f"- Scenarios: {', '.join(scenarios)}",
        f"- Seeds: {', '.join(str(s) for s in seeds)}",
        f"- Complete oracle traces: {completed_traces}",
        f"- Traces that hit search budget: {skipped_traces}",
        f"- Maximum DP states per trace: {args.max_states}",
        f"- Wall-clock experiment time: {perf_counter() - overall_start:.2f} seconds",
        "",
        "## Results by policy",
        "",
        "| Policy | Mean acceptance | Mean oracle gap (jobs) | Mean decision time per trace (ms) |",
        "|---|---:|---:|---:|",
    ]
    for row in policy_summary:
        report.append(
            f"| {row['policy']} | {row['mean_acceptance_rate']:.3f} "
            f"| {row['mean_oracle_gap_jobs']:.3f} "
            f"| {row['mean_total_decision_time_ms']:.3f} |"
        )
    report += [
        "",
        "## Interpretation guardrails",
        "",
        "- These results are exploratory and depend on the small synthetic workload traces.",
        "- Normal-approximation intervals are included as descriptive summaries; with few traces, they are not confirmatory inference.",
        "- Compare policies on the same trace (`trace_id`) and use paired differences for any claim.",
        "- An oracle gap is the gap to the exact optimum under this simulator's assumptions only.",
        "- A zero gap on tiny traces does not imply an online policy is optimal in general.",
        "- The oracle can exploit complete future information; it is an upper bound, not a fair deployment baseline.",
        "",
        "## Files",
        "",
        "- `online_vs_oracle_runs.csv`: one row per online policy and trace.",
        "- `oracle_trace_summary.csv`: exact solver runtime and DP size per trace.",
        "- `oracle_decisions.csv`: one recovered optimal schedule per trace.",
        "- `policy_summary.csv`: descriptive aggregate results.",
        "- `scenario_summary.csv`: per-scenario results.",
    ]
    (args.out / "M6_ORACLE_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"Completed oracle traces: {completed_traces}; budget-exceeded traces: {skipped_traces}")
    print(f"Wrote outputs to {args.out.resolve()}")
    if policy_summary:
        print("\nPolicy summary:")
        for row in policy_summary:
            print(
                f"{row['policy']:28s} acceptance={row['mean_acceptance_rate']:.3f} "
                f"oracle_gap={row['mean_oracle_gap_jobs']:.3f} "
                f"decision_ms={row['mean_total_decision_time_ms']:.3f}"
            )


if __name__ == "__main__":
    main()
