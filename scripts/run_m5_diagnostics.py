"""Run matched scheduler comparisons and collect adaptive-policy diagnostics.

Outputs runs.csv (one row per policy/workload pair), decisions.csv (one row
per incoming job and policy), and experiment_config.json. Regime-shift workloads
are controlled synthetic stress tests, not production traces.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter

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
from mig_scheduler.workloads import generate_regime_shift_workload, generate_workload  # noqa: E402

STATIONARY_SCENARIOS = ("uniform", "skew_small", "skew_big", "bimodal")
SHIFT_SCENARIOS = ("shift_small_to_big", "shift_big_to_small")
ALL_SCENARIOS = STATIONARY_SCENARIOS + SHIFT_SCENARIOS


def _csv_write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_policies(args: argparse.Namespace, run_seed: int):
    """Return (label, scheduler) pairs with transparent gate differences."""
    policies = [
        ("first_fit", FirstFitScheduler()),
        ("round_robin", RoundRobinScheduler()),
        ("bf_bi", BestFitBestIndexScheduler()),
        ("wf_bi", WorstFitBestIndexScheduler()),
        ("mfi", MFIScheduler()),
        ("fixed_lookahead", FixedBudgetLookaheadScheduler(
            horizon=args.horizon, rollouts=args.fixed_rollouts, seed=run_seed,
        )),
    ]
    common = dict(
        horizon=args.horizon,
        min_rollouts=args.adaptive_min_rollouts,
        max_rollouts=args.adaptive_max_rollouts,
        batch_size=args.adaptive_batch_size,
        seed=run_seed,
        minimum_history=args.adaptive_min_history,
        profile_window=max(args.adaptive_profile_window, args.adaptive_min_history),
        stability_tv_threshold=args.stability_tv_threshold,
        minimum_effect=args.minimum_effect,
    )
    # These are deliberately named presets for a gate-sensitivity study, not
    # tuned winners. They vary only the two gating rules and candidate cap.
    policies.append(("adaptive_conservative", AdaptiveBudgetLookaheadScheduler(
        **common, minimum_reliability=args.conservative_reliability,
        ambiguity_margin=args.conservative_ambiguity_margin,
        max_candidates=max(2, min(4, args.adaptive_max_candidates)),
    )))
    policies.append(("adaptive_default", AdaptiveBudgetLookaheadScheduler(
        **common, minimum_reliability=args.adaptive_min_reliability,
        ambiguity_margin=args.adaptive_ambiguity_margin,
    )))
    policies.append(("adaptive_permissive", AdaptiveBudgetLookaheadScheduler(
        **common, minimum_reliability=0.0,
        ambiguity_margin=args.permissive_ambiguity_margin,
    )))
    return policies


def make_workload(scenario: str, args: argparse.Namespace, seed: int):
    common = dict(
        count=args.jobs, seed=seed, mean_interarrival=args.mean_interarrival,
        min_duration=args.min_duration, max_duration=args.max_duration,
    )
    if scenario == "shift_small_to_big":
        return generate_regime_shift_workload(
            **common, before_scenario="skew_small", after_scenario="skew_big",
            shift_fraction=args.shift_fraction,
        )
    if scenario == "shift_big_to_small":
        return generate_regime_shift_workload(
            **common, before_scenario="skew_big", after_scenario="skew_small",
            shift_fraction=args.shift_fraction,
        )
    return generate_workload(scenario=scenario, **common)


def _decision_rows(scenario: str, seed: int, workload_seed: int, policy_label: str, result) -> list[dict]:
    rows = []
    for row in result.job_results:
        mfi_gpu = row.get("mfi_gpu_id")
        mfi_start = row.get("mfi_start")
        chosen_gpu = row.get("gpu_id")
        chosen_start = row.get("start")
        has_mfi_reference = mfi_gpu is not None and mfi_start is not None
        differs = (
            has_mfi_reference and chosen_gpu is not None and chosen_start is not None
            and (int(chosen_gpu), int(chosen_start)) != (int(mfi_gpu), int(mfi_start))
        )
        rows.append({
            "scenario": scenario,
            "seed": seed,
            "workload_seed": workload_seed,
            "policy_label": policy_label,
            "job_id": row.get("job_id"),
            "arrival_time": row.get("arrival_time"),
            "profile": row.get("profile"),
            "status": row.get("status"),
            "chosen_gpu_id": chosen_gpu,
            "chosen_start": chosen_start,
            "mfi_gpu_id": mfi_gpu,
            "mfi_start": mfi_start,
            "chosen_differs_from_mfi": int(bool(differs)) if has_mfi_reference else "",
            "decision_time_ns": row.get("decision_time_ns"),
            "delta_fragmentation": row.get("delta_fragmentation"),
            "adaptive_status": row.get("adaptive_status"),
            "fallback_reason": row.get("fallback_reason"),
            "forecast_reliability": row.get("forecast_reliability"),
            "mfi_margin": row.get("mfi_margin"),
            "adaptive_stop_reason": row.get("adaptive_stop_reason"),
            "candidate_count": row.get("candidate_count"),
            "candidate_pool_size": row.get("candidate_pool_size"),
            "rollouts_per_candidate": row.get("rollouts_per_candidate"),
            "total_rollouts": row.get("total_rollouts"),
            "lookahead_value": row.get("lookahead_value"),
            "lookahead_std": row.get("lookahead_std"),
            "candidate_scores_json": row.get("candidate_scores_json", "[]"),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=60)
    parser.add_argument("--gpus", type=int, default=3)
    parser.add_argument("--seeds", default="11,22,33")
    parser.add_argument("--scenarios", default=",".join(ALL_SCENARIOS))
    parser.add_argument("--mean-interarrival", type=float, default=0.8)
    parser.add_argument("--min-duration", type=int, default=1)
    parser.add_argument("--max-duration", type=int, default=12)
    parser.add_argument("--shift-fraction", type=float, default=0.5)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--fixed-rollouts", type=int, default=4)
    parser.add_argument("--adaptive-min-rollouts", type=int, default=2)
    parser.add_argument("--adaptive-max-rollouts", type=int, default=8)
    parser.add_argument("--adaptive-batch-size", type=int, default=2)
    parser.add_argument("--adaptive-min-history", type=int, default=16)
    parser.add_argument("--adaptive-profile-window", type=int, default=32)
    parser.add_argument("--stability-tv-threshold", type=float, default=0.60)
    parser.add_argument("--adaptive-min-reliability", type=float, default=0.35)
    parser.add_argument("--adaptive-ambiguity-margin", type=int, default=1)
    parser.add_argument("--adaptive-max-candidates", type=int, default=6)
    parser.add_argument("--conservative-reliability", type=float, default=0.60)
    parser.add_argument("--conservative-ambiguity-margin", type=int, default=0)
    parser.add_argument("--permissive-ambiguity-margin", type=int, default=1000000)
    parser.add_argument("--minimum-effect", type=float, default=0.0)
    parser.add_argument("--policy-seed-offset", type=int, default=900_000)
    parser.add_argument("--out-dir", type=Path, default=Path("results/m5_diagnostics"))
    args = parser.parse_args()

    if args.jobs < 1 or args.gpus < 1:
        raise SystemExit("--jobs and --gpus must be >= 1")
    if args.horizon < 1 or args.fixed_rollouts < 1:
        raise SystemExit("--horizon and --fixed-rollouts must be >= 1")
    if args.adaptive_min_rollouts < 2 or args.adaptive_max_rollouts < args.adaptive_min_rollouts:
        raise SystemExit("Adaptive rollout limits are invalid")
    if args.adaptive_batch_size < 1:
        raise SystemExit("--adaptive-batch-size must be >= 1")
    if args.conservative_reliability < 0 or args.conservative_reliability > 1:
        raise SystemExit("--conservative-reliability must be in [0,1]")
    seeds = [int(piece.strip()) for piece in args.seeds.split(",") if piece.strip()]
    scenarios = [piece.strip() for piece in args.scenarios.split(",") if piece.strip()]
    unknown = set(scenarios) - set(ALL_SCENARIOS)
    if unknown:
        raise SystemExit(f"Unknown scenarios {sorted(unknown)}; use {ALL_SCENARIOS}")
    if not seeds or not scenarios:
        raise SystemExit("Provide at least one seed and scenario")

    run_rows: list[dict] = []
    decision_rows: list[dict] = []
    config = vars(args).copy()
    config["out_dir"] = str(args.out_dir)
    config["seeds"] = seeds
    config["scenarios"] = scenarios
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    total = len(scenarios) * len(seeds) * 9
    completed = 0
    for scenario_index, scenario in enumerate(scenarios):
        for seed in seeds:
            # Use independent deterministic workload streams across scenarios,
            # while keeping the same scenario/seed workload matched across policies.
            workload_seed = seed + scenario_index * 1_000_003
            jobs = make_workload(scenario, args, workload_seed)
            workload_signature = f"{scenario}:{seed}:{workload_seed}:{len(jobs)}"
            for policy_label, scheduler in build_policies(args, args.policy_seed_offset + scenario_index * 100_000 + seed):
                started = perf_counter()
                result = DiscreteEventSimulator(args.gpus, scheduler).run(jobs)
                wall_clock = perf_counter() - started
                summary = result.summary()
                decisions = _decision_rows(scenario, seed, workload_seed, policy_label, result)
                adaptive_decisions = [d for d in decisions if d["adaptive_status"] is not None]
                adaptive_eligible = [d for d in adaptive_decisions if d["adaptive_status"] != "no_feasible_placement"]
                adaptive_lookahead_calls = sum(d["adaptive_status"] == "lookahead" for d in adaptive_decisions)
                fallbacks = sum(d["adaptive_status"] == "mfi_fallback" for d in adaptive_decisions)
                all_lookahead_calls = sum(float(d.get("total_rollouts") or 0) > 0 for d in decisions)
                all_mfi_reference_rows = [d for d in decisions if d["chosen_differs_from_mfi"] != ""]
                all_changed = sum(int(d["chosen_differs_from_mfi"]) for d in all_mfi_reference_rows)
                diff_rows = [d for d in adaptive_decisions if d["chosen_differs_from_mfi"] != ""]
                changed = sum(int(d["chosen_differs_from_mfi"]) for d in diff_rows)
                profile_accepts: dict[str, int] = {}
                profile_rejects: dict[str, int] = {}
                for d in decisions:
                    target = profile_accepts if d["status"] == "accepted" else profile_rejects
                    target[d["profile"]] = target.get(d["profile"], 0) + 1

                run_rows.append({
                    "scenario": scenario,
                    "seed": seed,
                    "workload_seed": workload_seed,
                    "workload_signature": workload_signature,
                    "policy_label": policy_label,
                    "policy": scheduler.name,
                    "jobs": args.jobs,
                    "gpus": args.gpus,
                    "horizon": args.horizon if "lookahead" in scheduler.name else 0,
                    "fixed_rollouts": args.fixed_rollouts if scheduler.name == "fixed_lookahead" else 0,
                    "adaptive_min_rollouts": args.adaptive_min_rollouts if scheduler.name == "adaptive_lookahead" else 0,
                    "adaptive_max_rollouts": args.adaptive_max_rollouts if scheduler.name == "adaptive_lookahead" else 0,
                    "acceptance_rate": summary["acceptance_rate"],
                    "accepted": summary["accepted"],
                    "rejected": summary["rejected"],
                    "mean_decision_time_ns": summary["mean_decision_time_ns"],
                    "total_decision_time_ns": summary["total_decision_time_ns"],
                    "wall_clock_seconds": wall_clock,
                    "mean_time_weighted_fragmentation_score": summary["mean_time_weighted_fragmentation_score"],
                    "peak_fragmentation_score": summary["peak_fragmentation_score"],
                    "mean_arrival_snapshot_compute_utilization": summary["mean_arrival_snapshot_compute_utilization"],
                    "mean_arrival_snapshot_memory_utilization": summary["mean_arrival_snapshot_memory_utilization"],
                    "mean_arrival_snapshot_active_gpus": summary["mean_arrival_snapshot_active_gpus"],
                    "mean_lookahead_value": summary["mean_lookahead_value"],
                    "mean_lookahead_rollouts_per_decision": summary["mean_lookahead_rollouts_per_decision"],
                    "adaptive_lookahead_decisions": summary["adaptive_lookahead_decisions"],
                    "adaptive_mfi_fallback_decisions": summary["adaptive_mfi_fallback_decisions"],
                    "adaptive_no_feasible_decisions": summary["adaptive_no_feasible_decisions"],
                    "adaptive_mean_forecast_reliability": summary["adaptive_mean_forecast_reliability"],
                    "adaptive_mean_total_rollouts": summary["adaptive_mean_total_rollouts"],
                    "adaptive_confidence_stops": summary["adaptive_confidence_stops"],
                    "adaptive_budget_stops": summary["adaptive_budget_stops"],
                    "lookahead_decisions": all_lookahead_calls,
                    "lookahead_fraction_of_all_jobs": all_lookahead_calls / len(jobs) if jobs else 0,
                    "fraction_choices_different_from_mfi": all_changed / len(all_mfi_reference_rows) if all_mfi_reference_rows else "",
                    "lookahead_fraction_of_mfi_eligible_adaptive_jobs": adaptive_lookahead_calls / len(adaptive_eligible) if adaptive_eligible else 0,
                    "adaptive_lookahead_fraction_of_all_jobs": adaptive_lookahead_calls / len(jobs) if jobs else 0,
                    "adaptive_fallback_fraction_of_eligible_jobs": fallbacks / len(adaptive_eligible) if adaptive_eligible else 0,
                    "adaptive_lookahead_fraction_of_eligible_jobs": adaptive_lookahead_calls / len(adaptive_eligible) if adaptive_eligible else 0,
                    "adaptive_decisions_different_from_mfi": changed,
                    "adaptive_fraction_choices_different_from_mfi": changed / len(diff_rows) if diff_rows else 0,
                    "accepted_by_profile_json": json.dumps(profile_accepts, sort_keys=True),
                    "rejected_by_profile_json": json.dumps(profile_rejects, sort_keys=True),
                })
                decision_rows.extend(decisions)
                completed += 1
                print(
                    f"[{completed:>3}/{total}] {scenario:19s} seed={seed:>4} "
                    f"{policy_label:22s} accept={summary['acceptance_rate']:.3f} "
                    f"decision_us={summary['mean_decision_time_ns']/1000:.1f} "
                    f"wall_s={wall_clock:.2f}"
                )

    _csv_write(args.out_dir / "runs.csv", run_rows)
    _csv_write(args.out_dir / "decisions.csv", decision_rows)
    print(f"\nSaved {len(run_rows)} run rows to {(args.out_dir / 'runs.csv').resolve()}")
    print(f"Saved {len(decision_rows)} decision rows to {(args.out_dir / 'decisions.csv').resolve()}")
    print("Next: python scripts/analyze_m5_diagnostics.py results/m5_diagnostics")


if __name__ == "__main__":
    main()
