"""Decision-centric horizon audit for MFI vs fixed/adaptive lookahead.

This is an exploratory simulator study, not a hardware benchmark or an exact
replication of the MFI paper. Every policy receives the same workload trace.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
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
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_jobs(scenario: str, *, jobs: int, seed: int, mean_interarrival: float,
              min_duration: int, max_duration: int, shift_fraction: float):
    common = dict(count=jobs, seed=seed, mean_interarrival=mean_interarrival,
                  min_duration=min_duration, max_duration=max_duration)
    if scenario == "shift_small_to_big":
        return generate_regime_shift_workload(
            **common, before_scenario="skew_small", after_scenario="skew_big",
            shift_fraction=shift_fraction,
        )
    if scenario == "shift_big_to_small":
        return generate_regime_shift_workload(
            **common, before_scenario="skew_big", after_scenario="skew_small",
            shift_fraction=shift_fraction,
        )
    return generate_workload(scenario=scenario, **common)


def build_policies(trace_seed: int, fixed_horizons: list[int], fixed_rollouts: int,
                   adaptive_horizons: list[int], adaptive_max_rollouts: int):
    policies = [("first_fit", FirstFitScheduler()), ("mfi", MFIScheduler())]
    for horizon in fixed_horizons:
        policies.append((
            f"fixed_h{horizon}_r{fixed_rollouts}",
            FixedBudgetLookaheadScheduler(
                horizon=horizon, rollouts=fixed_rollouts, seed=trace_seed + horizon * 1009,
            ),
        ))
    for horizon in adaptive_horizons:
        policies.append((
            f"adaptive_h{horizon}_max{adaptive_max_rollouts}",
            AdaptiveBudgetLookaheadScheduler(
                horizon=horizon,
                min_rollouts=min(2, adaptive_max_rollouts),
                max_rollouts=adaptive_max_rollouts,
                batch_size=min(2, adaptive_max_rollouts),
                seed=trace_seed + horizon * 2003,
                minimum_history=12,
                profile_window=32,
                stability_tv_threshold=0.60,
                minimum_reliability=0.35,
                ambiguity_margin=1,
                max_candidates=6,
                minimum_effect=0.0,
            ),
        ))
    return policies


def parse_candidate_scores(row: dict) -> list[dict]:
    raw = row.get("candidate_scores_json") or "[]"
    try:
        entries = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    candidates = []
    for item in entries:
        if not isinstance(item, (list, tuple)) or len(item) < 5:
            continue
        gpu_id, start, delta_f, mean_value, sample_sd = item[:5]
        candidates.append({
            "gpu_id": int(gpu_id), "start": int(start), "delta_f": float(delta_f),
            "mean_future_acceptance": float(mean_value), "rollout_sd": float(sample_sd),
        })
    return candidates


def derive_decision_diagnostic(run_meta: dict, job_row: dict) -> dict:
    candidates = parse_candidate_scores(job_row)
    chosen = None
    if job_row.get("gpu_id") not in (None, "") and job_row.get("start") not in (None, ""):
        chosen = (int(job_row["gpu_id"]), int(job_row["start"]))
    mfi_gpu_raw = job_row.get("mfi_gpu_id")
    mfi_start_raw = job_row.get("mfi_start")
    mfi_choice = None
    if mfi_gpu_raw not in (None, "") and mfi_start_raw not in (None, ""):
        mfi_choice = (int(mfi_gpu_raw), int(mfi_start_raw))
    by_key = {(c["gpu_id"], c["start"]): c for c in candidates}
    mfi_candidate = by_key.get(mfi_choice)
    chosen_candidate = by_key.get(chosen)
    sorted_candidates = sorted(candidates, key=lambda c: c["mean_future_acceptance"], reverse=True)
    top_gap = (
        sorted_candidates[0]["mean_future_acceptance"] - sorted_candidates[1]["mean_future_acceptance"]
        if len(sorted_candidates) >= 2 else None
    )
    predicted_gain_vs_mfi = None
    if chosen_candidate and mfi_candidate:
        predicted_gain_vs_mfi = (
            chosen_candidate["mean_future_acceptance"] - mfi_candidate["mean_future_acceptance"]
        )
    return {
        **run_meta,
        "job_id": job_row.get("job_id"),
        "profile": job_row.get("profile"),
        "arrival_time": job_row.get("arrival_time"),
        "status": job_row.get("status"),
        "chosen_gpu_id": chosen[0] if chosen else None,
        "chosen_start": chosen[1] if chosen else None,
        "mfi_gpu_id": mfi_choice[0] if mfi_choice else None,
        "mfi_start": mfi_choice[1] if mfi_choice else None,
        "choice_differs_from_mfi": (
            int(chosen != mfi_choice) if chosen is not None and mfi_choice is not None else None
        ),
        "candidate_count_from_scores": len(candidates),
        "lookahead_value": job_row.get("lookahead_value"),
        "lookahead_std": job_row.get("lookahead_std"),
        "lookahead_horizon": job_row.get("lookahead_horizon"),
        "rollouts_per_candidate": job_row.get("rollouts_per_candidate"),
        "total_rollouts": job_row.get("total_rollouts"),
        "decision_time_ns": job_row.get("decision_time_ns"),
        "mfi_delta_fragmentation": (
            mfi_candidate["delta_f"] if mfi_candidate else None
        ),
        "chosen_delta_fragmentation": (
            chosen_candidate["delta_f"] if chosen_candidate else job_row.get("delta_fragmentation")
        ),
        "estimated_future_gain_vs_mfi_candidate": predicted_gain_vs_mfi,
        "top_two_estimated_value_gap": top_gap,
        "top_candidate_estimated_value": sorted_candidates[0]["mean_future_acceptance"] if sorted_candidates else None,
        "top_candidate_rollout_sd": sorted_candidates[0]["rollout_sd"] if sorted_candidates else None,
        "adaptive_status": job_row.get("adaptive_status"),
        "fallback_reason": job_row.get("fallback_reason"),
        "forecast_reliability": job_row.get("forecast_reliability"),
        "adaptive_stop_reason": job_row.get("adaptive_stop_reason"),
        "mfi_margin": job_row.get("mfi_margin"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=40)
    parser.add_argument("--gpus", type=int, default=3)
    parser.add_argument("--seeds", default="11,22,33,44,55")
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--fixed-horizons", default="1,2,3,5")
    parser.add_argument("--fixed-rollouts", type=int, default=4)
    parser.add_argument("--adaptive-horizons", default="2,3")
    parser.add_argument("--adaptive-max-rollouts", type=int, default=4)
    parser.add_argument("--mean-interarrival", type=float, default=0.8)
    parser.add_argument("--min-duration", type=int, default=1)
    parser.add_argument("--max-duration", type=int, default=12)
    parser.add_argument("--shift-fraction", type=float, default=0.5)
    parser.add_argument("--seed-offset", type=int, default=1_700_000)
    parser.add_argument("--out", type=Path, default=Path("results/m7_horizon_audit"))
    args = parser.parse_args()

    if args.jobs < 1 or args.gpus < 1 or args.fixed_rollouts < 1 or args.adaptive_max_rollouts < 2:
        raise SystemExit("jobs/gpus/rollouts must be valid positive values")
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    scenarios = [x.strip() for x in args.scenarios.split(",") if x.strip()]
    fixed_horizons = sorted(set(int(x) for x in args.fixed_horizons.split(",") if x.strip()))
    adaptive_horizons = sorted(set(int(x) for x in args.adaptive_horizons.split(",") if x.strip()))
    if not seeds or not scenarios or not fixed_horizons or any(h < 1 for h in fixed_horizons + adaptive_horizons):
        raise SystemExit("Provide seeds, scenarios, and positive horizons")
    unknown = set(scenarios) - set(SCENARIOS)
    if unknown:
        raise SystemExit(f"Unknown scenarios: {sorted(unknown)}")

    args.out.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["out"] = str(args.out)
    config["seeds"] = seeds
    config["scenarios"] = scenarios
    config["fixed_horizons"] = fixed_horizons
    config["adaptive_horizons"] = adaptive_horizons
    (args.out / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    run_rows: list[dict] = []
    decision_rows: list[dict] = []
    total_runs = len(scenarios) * len(seeds) * (2 + len(fixed_horizons) + len(adaptive_horizons))
    completed = 0
    total_start = perf_counter()

    for scenario_i, scenario in enumerate(scenarios):
        for seed in seeds:
            workload_seed = seed + scenario_i * 1_000_003
            jobs = make_jobs(
                scenario, jobs=args.jobs, seed=workload_seed,
                mean_interarrival=args.mean_interarrival,
                min_duration=args.min_duration, max_duration=args.max_duration,
                shift_fraction=args.shift_fraction,
            )
            trace_id = f"{scenario}-seed{seed}"
            policies = build_policies(
                seed + args.seed_offset + scenario_i * 100_000,
                fixed_horizons, args.fixed_rollouts, adaptive_horizons, args.adaptive_max_rollouts,
            )
            for policy_name, scheduler in policies:
                run_start = perf_counter()
                result = DiscreteEventSimulator(args.gpus, scheduler).run(jobs)
                run_wall = perf_counter() - run_start
                summary = result.summary()
                rows_with_mfi = [
                    row for row in result.job_results
                    if row.get("gpu_id") is not None and row.get("mfi_gpu_id") is not None
                    and row.get("mfi_start") is not None
                ]
                changed = sum(
                    (int(row["gpu_id"]), int(row["start"])) != (int(row["mfi_gpu_id"]), int(row["mfi_start"]))
                    for row in rows_with_mfi
                )
                total_rollouts = sum(int(row.get("total_rollouts") or 0) for row in result.job_results)
                meta = {
                    "trace_id": trace_id, "scenario": scenario, "seed": seed,
                    "workload_seed": workload_seed, "policy": policy_name,
                    "jobs": len(jobs), "gpus": args.gpus,
                }
                run_row = {
                    **meta,
                    "accepted": result.accepted, "rejected": result.rejected,
                    "acceptance_rate": result.acceptance_rate,
                    "decision_time_total_ms": result.total_decision_time_ns / 1e6,
                    "decision_time_mean_us": result.mean_decision_time_ns / 1e3,
                    "wall_clock_run_s": run_wall,
                    "time_weighted_fragmentation": result.mean_time_weighted_fragmentation_score,
                    "peak_fragmentation_score": result.peak_fragmentation_score,
                    "lookahead_decisions": summary["lookahead_decisions"],
                    "adaptive_lookahead_decisions": summary["adaptive_lookahead_decisions"],
                    "adaptive_fallback_decisions": summary["adaptive_mfi_fallback_decisions"],
                    "total_rollout_evaluations": total_rollouts,
                    "rollouts_per_arrival": total_rollouts / max(result.submitted, 1),
                    "mfi_comparable_accepted_placements": len(rows_with_mfi),
                    "placements_different_from_mfi": changed,
                    "fraction_placements_different_from_mfi": changed / len(rows_with_mfi) if rows_with_mfi else 0.0,
                    "mean_forecast_reliability": summary["adaptive_mean_forecast_reliability"],
                }
                run_rows.append(run_row)
                for row in result.job_results:
                    if policy_name.startswith("fixed_") or policy_name.startswith("adaptive_"):
                        decision_rows.append(derive_decision_diagnostic(meta, row))
                completed += 1
                if completed % 10 == 0 or completed == total_runs:
                    print(f"Completed {completed}/{total_runs} runs", flush=True)

    write_csv(args.out / "m7_horizon_runs.csv", run_rows)
    write_csv(args.out / "m7_decision_diagnostics.csv", decision_rows)
    elapsed = perf_counter() - total_start
    (args.out / "run_manifest.json").write_text(json.dumps({
        "runs_completed": completed, "runs_expected": total_runs,
        "elapsed_wall_clock_seconds": elapsed,
        "run_csv": "m7_horizon_runs.csv",
        "decision_csv": "m7_decision_diagnostics.csv",
    }, indent=2), encoding="utf-8")
    print(f"Wrote {len(run_rows)} run rows and {len(decision_rows)} decision rows to {args.out.resolve()}")
    print(f"Total elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
