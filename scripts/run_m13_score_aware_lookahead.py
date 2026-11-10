"""M13: paired comparison of fixed/adaptive lookahead under both MFI score modes.

This is a controlled, reproducible simulator study, not an exact reproduction of
any paper. It uses the MFI A100 profile abstraction, synthetic workload mixes,
and limited replication to test whether lookahead conclusions depend on score
interpretation.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mig_scheduler.mfi_score_variants import MFIScoreVariantScheduler, SCORE_MODES  # noqa: E402
from mig_scheduler.paper_baselines import PaperFirstFitScheduler, PaperRoundRobinScheduler  # noqa: E402
from mig_scheduler.schedulers import (  # noqa: E402
    AdaptiveBudgetLookaheadScheduler,
    FixedBudgetLookaheadScheduler,
)
from mig_scheduler.mfi_score_variants import PaperBFWFBestIndexScheduler  # noqa: E402
from mig_scheduler.simulator import DiscreteEventSimulator  # noqa: E402
from mig_scheduler.workloads import generate_progressive_load_workload  # noqa: E402

SCENARIOS = ("uniform", "skew_small", "skew_big", "bimodal")
REGIMES = {
    "progressive": {"target_fraction": 1.0, "arrival_gap": 0.25},
    "burst": {"target_fraction": 1.5, "arrival_gap": 0.0},
}
POLICY_KEYS = (
    "paper_ff", "paper_rr", "paper_bf_bi", "paper_wf_bi",
    "mfi_all_starts", "mfi_free_anchor_score",
    "fixed_lookahead__all_legal_starts", "fixed_lookahead__free_anchor_starts",
    "adaptive_lookahead__all_legal_starts", "adaptive_lookahead__free_anchor_starts",
)
REFERENCE = "mfi_free_anchor_score"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def paired_bootstrap_ci(values: list[float], *, seed: int, samples: int = 10_000) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    estimates = []
    n = len(values)
    for _ in range(samples):
        estimates.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    estimates.sort()
    return estimates[int(0.025 * samples)], estimates[min(samples - 1, int(0.975 * samples))]


def policy_factory(policy_key: str, *, seed: int):
    if policy_key == "paper_ff":
        return PaperFirstFitScheduler()
    if policy_key == "paper_rr":
        return PaperRoundRobinScheduler()
    if policy_key == "paper_bf_bi":
        return PaperBFWFBestIndexScheduler(strategy="bf_bi")
    if policy_key == "paper_wf_bi":
        return PaperBFWFBestIndexScheduler(strategy="wf_bi")
    if policy_key == "mfi_all_starts":
        return MFIScoreVariantScheduler(score_mode="all_legal_starts")
    if policy_key == "mfi_free_anchor_score":
        return MFIScoreVariantScheduler(score_mode="free_anchor_starts")
    if policy_key.startswith("fixed_lookahead__"):
        mode = policy_key.split("__", 1)[1]
        return FixedBudgetLookaheadScheduler(
            horizon=2, rollouts=3, seed=seed, score_mode=mode,
        )
    if policy_key.startswith("adaptive_lookahead__"):
        mode = policy_key.split("__", 1)[1]
        return AdaptiveBudgetLookaheadScheduler(
            horizon=2, min_rollouts=2, max_rollouts=4, batch_size=2,
            seed=seed + 50_000, minimum_history=8, profile_window=16,
            minimum_reliability=0.20, ambiguity_margin=3,
            max_candidates=5, score_mode=mode,
        )
    raise ValueError(f"Unknown policy: {policy_key}")


def plot_results(out: Path, summary_rows: list[dict], decision_rows: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment-dependent plotting support
        (out / "PLOT_WARNING.txt").write_text(f"Plots were not generated: {exc}\n", encoding="utf-8")
        return

    for regime in REGIMES:
        rows = [row for row in summary_rows if row["regime"] == regime]
        means = {row["policy"]: row["acceptance_mean"] for row in rows}
        labels = list(POLICY_KEYS)
        values = [means.get(label, 0.0) for label in labels]
        fig, ax = plt.subplots(figsize=(13, 5.5))
        ax.bar(range(len(labels)), values)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Mean job acceptance rate")
        ax.set_title(f"M13 {regime.title()} workload: score-aware lookahead comparison")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / f"m13_acceptance_{regime}.png", dpi=160)
        fig.savefig(out / f"m13_acceptance_{regime}.svg")
        plt.close(fig)

    variants = [
        row for row in decision_rows
        if row.get("comparison") in {"fixed_score_mode_sensitivity", "adaptive_score_mode_sensitivity"}
    ]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels, values = [], []
    for regime in REGIMES:
        for comparison in ("fixed_score_mode_sensitivity", "adaptive_score_mode_sensitivity"):
            group = [r for r in variants if r["regime"] == regime and r["comparison"] == comparison]
            if group:
                labels.append(f"{regime}\n{comparison.replace('_score_mode_sensitivity', '')}")
                values.append(sum(float(r["decision_change_fraction"]) for r in group) / len(group))
    ax.bar(range(len(labels)), values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Fraction of job decisions changed")
    ax.set_title("How much the MFI score interpretation changes lookahead decisions")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "m13_score_mode_decision_sensitivity.png", dpi=160)
    fig.savefig(out / "m13_score_mode_decision_sensitivity.svg")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--repetitions", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "m13_score_aware_lookahead_4gpu_3rep")
    args = ap.parse_args()
    if args.gpus < 1 or args.repetitions < 1:
        ap.error("--gpus and --repetitions must be positive")
    args.out.mkdir(parents=True, exist_ok=True)

    raw_runs: list[dict] = []
    per_job_rows: list[dict] = []
    paired_job_diffs: list[dict] = []
    job_decisions: dict[tuple[str, str, int, str], dict[str, tuple]] = {}
    workload_ids: list[tuple[str, str, int]] = []
    start_wall = time.perf_counter()
    total = len(REGIMES) * len(SCENARIOS) * args.repetitions * len(POLICY_KEYS)
    done = 0

    for regime_index, (regime, config) in enumerate(REGIMES.items()):
        for scenario_index, scenario in enumerate(SCENARIOS):
            for rep in range(args.repetitions):
                seed = args.seed + regime_index * 10_000_019 + scenario_index * 1_000_003 + rep
                jobs = generate_progressive_load_workload(
                    gpu_count=args.gpus, scenario=scenario, seed=seed,
                    arrival_gap=config["arrival_gap"], load_basis="paper_slice_units",
                    target_fraction=config["target_fraction"],
                )
                workload_id = (regime, scenario, rep)
                workload_ids.append(workload_id)
                for policy_index, policy_key in enumerate(POLICY_KEYS):
                    scheduler = policy_factory(policy_key, seed=seed + 3_000_017)
                    result = DiscreteEventSimulator(args.gpus, scheduler).run(jobs)
                    summary = result.summary()
                    run_row = {
                        "regime": regime, "scenario": scenario, "repetition": rep,
                        "workload_seed": seed, "policy": policy_key,
                        "gpus": args.gpus, "submitted": summary["submitted"],
                        "accepted": summary["accepted"], "rejected": summary["rejected"],
                        "acceptance_rate": summary["acceptance_rate"],
                        "mean_decision_time_ns": summary["mean_decision_time_ns"],
                        "total_decision_time_ns": summary["total_decision_time_ns"],
                        "final_fragmentation_score_all_starts": summary["final_fragmentation_score"],
                        "mean_time_weighted_fragmentation_score_all_starts": summary["mean_time_weighted_fragmentation_score"],
                        "mean_compute_utilization": summary["mean_arrival_snapshot_compute_utilization"],
                        "mean_memory_utilization": summary["mean_arrival_snapshot_memory_utilization"],
                        "mean_active_gpus": summary["mean_arrival_snapshot_active_gpus"],
                        "lookahead_decisions": summary["lookahead_decisions"],
                        "adaptive_lookahead_decisions": summary["adaptive_lookahead_decisions"],
                        "adaptive_mfi_fallback_decisions": summary["adaptive_mfi_fallback_decisions"],
                        "mean_rollouts_per_lookahead_decision": summary["mean_lookahead_rollouts_per_decision"],
                        "adaptive_mean_total_rollouts": summary["adaptive_mean_total_rollouts"],
                        "trace_jobs": len(jobs), "target_fraction": config["target_fraction"],
                        "arrival_gap": config["arrival_gap"],
                        "score_mode": getattr(scheduler, "score_mode", "not_applicable"),
                    }
                    raw_runs.append(run_row)
                    decisions = {}
                    for jr in result.job_results:
                        decisions[jr["job_id"]] = (
                            jr["status"], jr.get("gpu_id"), jr.get("start")
                        )
                        per_job_rows.append({
                            "regime": regime, "scenario": scenario, "repetition": rep,
                            "workload_seed": seed, "policy": policy_key,
                            "job_id": jr["job_id"], "profile": jr["profile"],
                            "status": jr["status"], "gpu_id": jr.get("gpu_id"),
                            "start": jr.get("start"), "decision_time_ns": jr.get("decision_time_ns"),
                            "delta_fragmentation": jr.get("delta_fragmentation"),
                            "lookahead_value": jr.get("lookahead_value"),
                            "lookahead_std": jr.get("lookahead_std"),
                            "lookahead_horizon": jr.get("lookahead_horizon"),
                            "candidate_count": jr.get("candidate_count"),
                            "total_rollouts": jr.get("total_rollouts"),
                            "adaptive_status": jr.get("adaptive_status"),
                            "fallback_reason": jr.get("fallback_reason"),
                            "forecast_reliability": jr.get("forecast_reliability"),
                            "mfi_margin": jr.get("mfi_margin"),
                            "adaptive_stop_reason": jr.get("adaptive_stop_reason"),
                            "candidate_pool_size": jr.get("candidate_pool_size"),
                        })
                    job_decisions[(regime, scenario, rep, policy_key)] = decisions
                    done += 1
                    print(f"[{done}/{total}] {regime}/{scenario} rep={rep+1}/{args.repetitions} {policy_key}: {summary['accepted']}/{summary['submitted']} accepted; {summary['mean_decision_time_ns']/1e6:.2f} ms/decision", flush=True)

    write_csv(args.out / "m13_raw_runs.csv", raw_runs)
    write_csv(args.out / "m13_per_job_decisions.csv", per_job_rows)

    # Policy-level aggregation; each workload trace is the pairing unit.
    summary_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in raw_runs:
        summary_groups[(row["regime"], row["policy"])].append(row)
    summary_rows = []
    run_lookup = {(r["regime"], r["scenario"], r["repetition"], r["policy"]): r for r in raw_runs}
    for (regime, policy), rows in sorted(summary_groups.items()):
        rates = [float(r["acceptance_rate"]) for r in rows]
        summary_rows.append({
            "regime": regime, "policy": policy, "workload_traces": len(rows),
            "submitted_mean": statistics.mean(float(r["submitted"]) for r in rows),
            "accepted_mean": statistics.mean(float(r["accepted"]) for r in rows),
            "acceptance_mean": statistics.mean(rates),
            "acceptance_sd_across_traces": statistics.stdev(rates) if len(rates) > 1 else 0.0,
            "mean_decision_time_ms": statistics.mean(float(r["mean_decision_time_ns"]) for r in rows) / 1e6,
            "mean_compute_utilization": statistics.mean(float(r["mean_compute_utilization"]) for r in rows),
            "mean_memory_utilization": statistics.mean(float(r["mean_memory_utilization"]) for r in rows),
            "mean_active_gpus": statistics.mean(float(r["mean_active_gpus"]) for r in rows),
            "mean_lookahead_decisions": statistics.mean(float(r["lookahead_decisions"]) for r in rows),
            "mean_adaptive_lookahead_decisions": statistics.mean(float(r["adaptive_lookahead_decisions"]) for r in rows),
            "mean_adaptive_fallback_decisions": statistics.mean(float(r["adaptive_mfi_fallback_decisions"]) for r in rows),
            "mean_rollouts_per_lookahead_decision": statistics.mean(
                [float(r["mean_rollouts_per_lookahead_decision"]) for r in rows
                 if r["mean_rollouts_per_lookahead_decision"] is not None]
            ) if any(r["mean_rollouts_per_lookahead_decision"] is not None for r in rows) else 0.0,
        })
    write_csv(args.out / "m13_policy_summary.csv", summary_rows)

    # Paired acceptance deltas against the Figure-3-conformance MFI score candidate.
    paired_rows = []
    for regime in REGIMES:
        for policy in POLICY_KEYS:
            if policy == REFERENCE:
                continue
            deltas = [
                float(run_lookup[(regime, scenario, rep, policy)]["acceptance_rate"])
                - float(run_lookup[(regime, scenario, rep, REFERENCE)]["acceptance_rate"])
                for scenario in SCENARIOS for rep in range(args.repetitions)
            ]
            low, high = paired_bootstrap_ci(deltas, seed=args.seed + len(paired_rows) * 97)
            paired_rows.append({
                "regime": regime, "policy_vs_reference": policy,
                "reference_policy": REFERENCE, "paired_workloads": len(deltas),
                "mean_acceptance_delta": statistics.mean(deltas),
                "mean_acceptance_delta_percentage_points": 100 * statistics.mean(deltas),
                "paired_bootstrap_95ci_low": low, "paired_bootstrap_95ci_high": high,
                "paired_bootstrap_95ci_low_pp": low * 100,
                "paired_bootstrap_95ci_high_pp": high * 100,
                "wins": sum(d > 1e-12 for d in deltas),
                "losses": sum(d < -1e-12 for d in deltas),
                "ties": sum(abs(d) <= 1e-12 for d in deltas),
            })
    write_csv(args.out / "m13_paired_acceptance_vs_free_anchor_mfi.csv", paired_rows)

    # Score-mode decision sensitivity, comparing all-starts vs free-anchor on identical traces.
    for regime in REGIMES:
        for scenario in SCENARIOS:
            for rep in range(args.repetitions):
                comparisons = [
                    ("fixed_score_mode_sensitivity", "fixed_lookahead__all_legal_starts", "fixed_lookahead__free_anchor_starts"),
                    ("adaptive_score_mode_sensitivity", "adaptive_lookahead__all_legal_starts", "adaptive_lookahead__free_anchor_starts"),
                ]
                for comparison, left, right in comparisons:
                    left_map = job_decisions[(regime, scenario, rep, left)]
                    right_map = job_decisions[(regime, scenario, rep, right)]
                    shared = sorted(set(left_map) & set(right_map))
                    changed = sum(left_map[job_id] != right_map[job_id] for job_id in shared)
                    paired_job_diffs.append({
                        "regime": regime, "scenario": scenario, "repetition": rep,
                        "comparison": comparison, "left_policy": left,
                        "right_policy": right, "jobs_compared": len(shared),
                        "decision_changes": changed,
                        "decision_change_fraction": changed / len(shared) if shared else 0.0,
                        "left_acceptance_rate": run_lookup[(regime, scenario, rep, left)]["acceptance_rate"],
                        "right_acceptance_rate": run_lookup[(regime, scenario, rep, right)]["acceptance_rate"],
                        "acceptance_delta_right_minus_left": (
                            float(run_lookup[(regime, scenario, rep, right)]["acceptance_rate"])
                            - float(run_lookup[(regime, scenario, rep, left)]["acceptance_rate"])
                        ),
                    })
    write_csv(args.out / "m13_score_mode_decision_sensitivity.csv", paired_job_diffs)

    # Manifest and concise report.
    manifest = {
        "milestone": 13,
        "study": "MFI score-aware fixed/adaptive lookahead comparison",
        "source_paper": "Zambianco, Fasol, Doriguzzi-Corin, An Online Fragmentation-Aware GPU Scheduler for Multi-Tenant MIG-based Clouds",
        "source_url": "https://arxiv.org/abs/2511.18906",
        "gpus": args.gpus, "repetitions_per_profile_mix_per_regime": args.repetitions,
        "seed_base": args.seed, "regimes": REGIMES, "scenarios": list(SCENARIOS),
        "policies": list(POLICY_KEYS), "score_modes": list(SCORE_MODES),
        "runs": len(raw_runs), "per_job_decision_rows": len(per_job_rows),
        "workload_trace_count": len(workload_ids),
        "fixed_lookahead": {"horizon": 2, "rollouts_per_candidate": 3},
        "adaptive_lookahead": {
            "horizon": 2, "min_rollouts": 2, "max_rollouts": 4,
            "batch_size": 2, "max_candidates": 5, "minimum_history": 8,
            "profile_window": 16, "minimum_reliability": 0.20,
            "ambiguity_margin": 3,
        },
        "elapsed_seconds": time.perf_counter() - start_wall,
        "limitations": [
            "Synthetic A100 profile/placement model; no physical MIG hardware or Kubernetes integration.",
            "This is not an exact reproduction of the source paper's full evaluation protocol.",
            "Three repetitions per profile mix and only four GPUs make this an exploratory controlled study.",
            "The adaptive forecast gates and confidence-stopping rule are heuristics, not calibrated probabilities.",
            "Lookahead rollout future arrivals and durations are modeled from observed history with fallbacks.",
        ],
    }
    (args.out / "m13_run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    plot_results(args.out, summary_rows, paired_job_diffs)

    report = []
    report.extend([
        "# M13 — Score-Aware Lookahead Evaluation",
        "",
        "## Research question",
        "Does fixed or adaptive finite-horizon lookahead behave differently when the MFI score used for current placement and future rollout placements is switched from `all_legal_starts` to `free_anchor_starts`?",
        "",
        "## Design",
        f"- {args.gpus} simulated GPUs; {args.repetitions} repetitions for each of four profile distributions in two regimes.",
        "- Progressive regime: 100% target offered load using the paper-slice-units basis and arrival gap 0.25.",
        "- Burst regime: 150% target offered load and arrival gap 0.0; all positive-duration jobs arrive before any completion event.",
        "- Ten policies, with the identical workload trace paired across all policies.",
        "- Fixed lookahead uses horizon 2 and 3 rollouts per feasible current placement; adaptive lookahead uses horizon 2, 2–4 rollouts, a batch size of 2, and at most 5 candidates.",
        "- `free_anchor_starts` is the primary MFI score candidate because it matches the Figure 3 score values for reconstructed states; `all_legal_starts` remains a sensitivity variant.",
        "",
        "## Policy acceptance summary",
        "",
        "| Regime | Policy | Mean acceptance | SD across paired workload traces | Mean decision time (ms) | Mean lookahead decisions/trace | Mean adaptive fallbacks/trace |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in summary_rows:
        report.append(
            f"| {row['regime']} | `{row['policy']}` | {100*float(row['acceptance_mean']):.2f}% | "
            f"{100*float(row['acceptance_sd_across_traces']):.2f} pp | {float(row['mean_decision_time_ms']):.3f} | "
            f"{float(row['mean_lookahead_decisions']):.2f} | {float(row['mean_adaptive_fallback_decisions']):.2f} |"
        )
    report.extend([
        "",
        "## Paired acceptance difference vs. free-anchor MFI",
        "",
        "Differences are policy acceptance minus `mfi_free_anchor_score`, computed on the same workload trace. Intervals are percentile bootstrap intervals over workload-level paired deltas; with only 12 traces per regime in the default run, they are exploratory, not definitive.",
        "",
        "| Regime | Policy | Mean delta (percentage points) | Paired bootstrap 95% interval (pp) | Wins / losses / ties |",
        "|---|---|---:|---:|---:|",
    ])
    for row in paired_rows:
        report.append(
            f"| {row['regime']} | `{row['policy_vs_reference']}` | {float(row['mean_acceptance_delta_percentage_points']):+.2f} | "
            f"[{100*float(row['paired_bootstrap_95ci_low']):+.2f}, {100*float(row['paired_bootstrap_95ci_high']):+.2f}] | "
            f"{row['wins']} / {row['losses']} / {row['ties']} |"
        )
    report.extend([
        "",
        "## Score-mode sensitivity",
        "",
        "See `m13_score_mode_decision_sensitivity.csv` for per-workload changes in accept/reject and placement choices, and `m13_score_mode_decision_sensitivity.png` for the aggregate decision-change fractions. A decision change is a different `(status, gpu_id, start)` tuple for the same job, not necessarily a change in acceptance.",
        "",
        "## Interpretation guardrails",
        "",
        "1. Do not claim that lookahead improves MFI unless a paired acceptance interval and repeated workload results support that conclusion. A different placement is not automatically a better placement.",
        "2. The two score definitions can change candidate rankings and future rollout placement decisions. Report both; do not hide the score definition in results or code.",
        "3. Adaptive lookahead may fall back frequently; its search count, fallback reasons, and decision latency are recorded in the CSV outputs.",
        "4. The burst workload intentionally suppresses completion-driven capacity reuse. It tests packing under simultaneous demand, not realistic production arrival behavior.",
        "5. These synthetic four-GPU results are a debugging and hypothesis-screening milestone, not a general cloud-scheduling result or exact paper replication.",
        "",
        "## Recommended next decision",
        "Use these paired results to decide whether to keep lookahead in the main research contribution. If effects are small, unstable, or dominated by run-time cost, simplify the design and expand the workload/seed study before proposing novelty. If a consistent signal appears, run a larger held-out study (at least 20–30 seeds per distribution and regime) with separate calibration and test seeds.",
        "",
        "## Files",
        "- `m13_raw_runs.csv`: one row per policy/workload run.",
        "- `m13_policy_summary.csv`: aggregate acceptance, utilization and decision-cost summary.",
        "- `m13_paired_acceptance_vs_free_anchor_mfi.csv`: paired baseline deltas and bootstrap intervals.",
        "- `m13_per_job_decisions.csv`: job-level decisions and lookahead diagnostics.",
        "- `m13_score_mode_decision_sensitivity.csv`: per-trace changes between score modes.",
        "- `m13_run_manifest.json`: full configuration and limitations.",
    ])
    (args.out / "M13_EXPERIMENT_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\nWrote M13 artifacts to {args.out.resolve()}")
    print(f"Elapsed: {time.perf_counter() - start_wall:.1f}s")


if __name__ == "__main__":
    main()
