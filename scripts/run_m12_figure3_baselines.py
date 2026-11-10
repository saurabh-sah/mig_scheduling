"""M12: Figure 3 score reconstruction and paper-path baseline comparison.

This is a reproduction-oriented sensitivity study, not an exact reproduction.
It adds paper-path BF-BI/WF-BI variants that select a GPU by aggregate residual
capacity first, then attempt feasible placement on that GPU only. The existing
BF-BI/WF-BI classes remain available as global-feasible variants.
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

from mig_scheduler.mfi_score_variants import MFIScoreVariantScheduler, PaperBFWFBestIndexScheduler  # noqa: E402
from mig_scheduler.paper_baselines import PaperFirstFitScheduler, PaperRoundRobinScheduler  # noqa: E402
from mig_scheduler.profiles import MFI_WORKLOAD_DISTRIBUTIONS, PROFILES  # noqa: E402
from mig_scheduler.schedulers import BestFitBestIndexScheduler, WorstFitBestIndexScheduler  # noqa: E402
from mig_scheduler.simulator import DiscreteEventSimulator  # noqa: E402
from mig_scheduler.workloads import generate_progressive_load_workload  # noqa: E402

SCENARIOS = tuple(MFI_WORKLOAD_DISTRIBUTIONS)
POLICIES = {
    "paper_ff": PaperFirstFitScheduler,
    "paper_rr": PaperRoundRobinScheduler,
    "paper_bf_bi": lambda: PaperBFWFBestIndexScheduler(strategy="bf_bi"),
    "paper_wf_bi": lambda: PaperBFWFBestIndexScheduler(strategy="wf_bi"),
    "global_feasible_bf_bi": BestFitBestIndexScheduler,
    "global_feasible_wf_bi": WorstFitBestIndexScheduler,
    "mfi_all_starts": lambda: MFIScoreVariantScheduler(score_mode="all_legal_starts"),
    "mfi_free_anchor_score": lambda: MFIScoreVariantScheduler(score_mode="free_anchor_starts"),
}
CHECKPOINTS = tuple(sorted(set(i / 10 for i in range(1, 11)) | {0.85}))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def first_snapshot_at_or_after(snapshots, threshold):
    for row in snapshots:
        if row["offered_demand_fraction"] >= threshold:
            return row
    return snapshots[-1]


def mean(xs):
    return statistics.mean(xs) if xs else 0.0


def paired_bootstrap_ci(values, *, seed, draws=3000):
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    means = sorted(mean([values[rng.randrange(len(values))] for _ in values]) for _ in range(draws))
    return means[int(0.025 * draws)], means[min(draws - 1, int(0.975 * draws))]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpus", type=int, default=100)
    ap.add_argument("--repetitions", type=int, default=10,
                    help="Paper reports 500 runs per distribution; 10 is exploratory.")
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--arrival-gap", type=float, default=1.0)
    ap.add_argument("--target-fraction", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "m12_figure3_baseline_study")
    args = ap.parse_args()
    if args.gpus < 1 or args.repetitions < 1:
        ap.error("--gpus and --repetitions must be positive")
    args.out.mkdir(parents=True, exist_ok=True)

    raw_runs, checkpoints, per_job_rows = [], [], []
    decision_maps = {}
    start_wall = time.perf_counter()
    run_count = len(SCENARIOS) * args.repetitions * len(POLICIES)
    done = 0
    for sidx, scenario in enumerate(SCENARIOS):
        for rep in range(args.repetitions):
            seed = args.seed + sidx * 1_000_003 + rep
            jobs = generate_progressive_load_workload(
                gpu_count=args.gpus, scenario=scenario, seed=seed,
                arrival_gap=args.arrival_gap, load_basis="paper_slice_units",
                target_fraction=args.target_fraction,
            )
            for policy_name, factory in POLICIES.items():
                scheduler = factory()
                result = DiscreteEventSimulator(args.gpus, scheduler).run(jobs)
                summary = result.summary()
                row = {
                    "scenario": scenario, "repetition": rep, "seed": seed,
                    "policy": policy_name, "gpus": args.gpus,
                    "target_fraction": args.target_fraction,
                    "arrival_gap": args.arrival_gap,
                    "submitted": summary["submitted"], "accepted": summary["accepted"],
                    "rejected": summary["rejected"],
                    "acceptance_rate": summary["acceptance_rate"],
                    "mean_decision_time_ns": summary["mean_decision_time_ns"],
                    "total_decision_time_ns": summary["total_decision_time_ns"],
                    "mean_fragmentation_score_all_starts": summary["mean_time_weighted_fragmentation_score"],
                    "final_fragmentation_score_all_starts": summary["final_fragmentation_score"],
                    "mean_compute_utilization": summary["mean_arrival_snapshot_compute_utilization"],
                    "mean_memory_utilization": summary["mean_arrival_snapshot_memory_utilization"],
                    "mean_active_gpus": summary["mean_arrival_snapshot_active_gpus"],
                    "trace_jobs": len(jobs),
                }
                raw_runs.append(row)
                decision_maps[(scenario, rep, policy_name)] = {
                    jr["job_id"]: (jr["status"], jr.get("gpu_id"), jr.get("start"))
                    for jr in result.job_results
                }
                for jr in result.job_results:
                    per_job_rows.append({
                        "scenario": scenario, "repetition": rep, "seed": seed,
                        "policy": policy_name, "job_id": jr["job_id"],
                        "profile": jr["profile"], "status": jr["status"],
                        "gpu_id": jr.get("gpu_id"), "start": jr.get("start"),
                        "decision_time_ns": jr.get("decision_time_ns"),
                        "delta_fragmentation": jr.get("delta_fragmentation"),
                    })
                for checkpoint in CHECKPOINTS:
                    snap = first_snapshot_at_or_after(result.snapshots, checkpoint)
                    checkpoints.append({
                        "scenario": scenario, "repetition": rep, "seed": seed,
                        "policy": policy_name, "checkpoint_requested": checkpoint,
                        "actual_offered_demand_fraction": snap["offered_demand_fraction"],
                        "submitted_so_far": snap["submitted_so_far"],
                        "accepted_so_far": snap["accepted_so_far"],
                        "rejected_so_far": snap["rejected_so_far"],
                        "acceptance_rate_so_far": snap["acceptance_rate_so_far"],
                        "compute_utilization": snap["compute_utilization"],
                        "memory_utilization": snap["memory_utilization"],
                        "active_gpus": snap["active_gpus"],
                        "cluster_fragmentation_score_all_starts": snap["cluster_fragmentation_score"],
                    })
                done += 1
                print(f"[{done}/{run_count}] {scenario} rep={rep+1}/{args.repetitions} {policy_name} accepted={summary['accepted']}/{summary['submitted']}")

    write_csv(args.out / "m12_raw_runs.csv", raw_runs)
    write_csv(args.out / "m12_checkpoint_rows.csv", checkpoints)
    write_csv(args.out / "m12_per_job_decisions.csv", per_job_rows)

    run_lookup = {(r["scenario"], r["repetition"], r["policy"]): r for r in raw_runs}
    decision_diff_rows = []
    for scenario in SCENARIOS:
        for rep in range(args.repetitions):
            all_starts = decision_maps[(scenario, rep, "mfi_all_starts")]
            free_anchor = decision_maps[(scenario, rep, "mfi_free_anchor_score")]
            ids = sorted(set(all_starts) & set(free_anchor))
            changed = sum(all_starts[j] != free_anchor[j] for j in ids)
            decision_diff_rows.append({
                "scenario": scenario, "repetition": rep,
                "jobs_compared": len(ids), "decision_changes": changed,
                "decision_change_fraction": changed / len(ids) if ids else 0.0,
                "acceptance_delta_free_anchor_minus_all_starts": (
                    float(run_lookup[(scenario, rep, "mfi_free_anchor_score")]["acceptance_rate"])
                    - float(run_lookup[(scenario, rep, "mfi_all_starts")]["acceptance_rate"])
                ),
            })
    write_csv(args.out / "m12_mfi_score_variant_decision_changes.csv", decision_diff_rows)

    summary_rows = []
    grouped = defaultdict(list)
    for row in raw_runs:
        grouped[(row["scenario"], row["policy"])].append(row)
    for (scenario, policy), rows in sorted(grouped.items()):
        ar = [float(r["acceptance_rate"]) for r in rows]
        summary_rows.append({
            "scenario": scenario, "policy": policy, "repetitions": len(rows),
            "accepted_mean": mean([float(r["accepted"]) for r in rows]),
            "acceptance_mean": mean(ar),
            "acceptance_sd": statistics.stdev(ar) if len(ar) > 1 else 0.0,
            "mean_decision_time_ns": mean([float(r["mean_decision_time_ns"]) for r in rows]),
            "mean_fragmentation_all_starts": mean([float(r["mean_fragmentation_score_all_starts"]) for r in rows]),
            "mean_compute_utilization": mean([float(r["mean_compute_utilization"]) for r in rows]),
            "mean_memory_utilization": mean([float(r["mean_memory_utilization"]) for r in rows]),
            "mean_active_gpus": mean([float(r["mean_active_gpus"]) for r in rows]),
        })
    write_csv(args.out / "m12_policy_summary.csv", summary_rows)

    # Paired acceptance differences on the identical workload trace, compared to
    # the figure-reconstructed score policy as primary reference.
    lookup = {(r["scenario"], r["repetition"], r["policy"]): r for r in raw_runs}
    pair_rows = []
    for policy in POLICIES:
        if policy == "mfi_free_anchor_score":
            continue
        for sidx, scenario in enumerate(SCENARIOS):
            deltas = []
            for rep in range(args.repetitions):
                ref = lookup[(scenario, rep, "mfi_free_anchor_score")]
                comp = lookup[(scenario, rep, policy)]
                deltas.append(float(comp["acceptance_rate"]) - float(ref["acceptance_rate"]))
            lo, hi = paired_bootstrap_ci(deltas, seed=args.seed + sidx * 1000 + list(POLICIES).index(policy))
            pair_rows.append({
                "scenario": scenario, "policy_vs_reference": policy,
                "reference_policy": "mfi_free_anchor_score", "pairs": len(deltas),
                "mean_acceptance_delta": mean(deltas), "sd_delta": statistics.stdev(deltas) if len(deltas) > 1 else 0,
                "paired_bootstrap_ci_low": lo, "paired_bootstrap_ci_high": hi,
                "wins": sum(x > 1e-12 for x in deltas), "losses": sum(x < -1e-12 for x in deltas),
                "ties": sum(abs(x) <= 1e-12 for x in deltas),
            })
    write_csv(args.out / "m12_paired_acceptance_delta_vs_figure_score_mfi.csv", pair_rows)

    manifest = {
        "milestone": 12,
        "study": "MFI Figure 3 numerical conformance and paper-path baseline comparison",
        "source_paper": "Zambianco, Fasol, Doriguzzi-Corin, arXiv:2511.18906",
        "source_url": "https://arxiv.org/abs/2511.18906",
        "gpu_model": "A100-80GB profile/placement abstraction with separate compute/memory masks",
        "gpus": args.gpus, "repetitions_per_distribution": args.repetitions,
        "seed_base": args.seed, "arrival_gap": args.arrival_gap,
        "target_fraction": args.target_fraction, "scenarios": list(SCENARIOS),
        "policies": list(POLICIES), "checkpoint_count": len(CHECKPOINTS),
        "runs": len(raw_runs), "checkpoint_rows": len(checkpoints), "per_job_decision_rows": len(per_job_rows),
        "elapsed_seconds": time.perf_counter() - start_wall,
        "model_fidelity_note": (
            "Not an exact reproduction. The free-anchor score is a transparent interpretation that matches "
            "the two numeric fragmentation scores in Figure 3 for the reconstructed states. The paper text "
            "does not explicitly state this filtering in Algorithm 1. BF/WF resource scalarization also remains "
            "an implementation choice. All alternative policies are labelled by their operational behavior."
        ),
    }
    (args.out / "m12_run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Create concise human-readable report from the actual rows.
    with (args.out / "M12_EXPERIMENT_REPORT.md").open("w", encoding="utf-8") as f:
        f.write("# M12 Experimental Report — Figure 3 Conformance and Baseline Semantics\n\n")
        f.write("## Scope\n\n")
        f.write("This is a paper-guided sensitivity experiment, not an exact reproduction. It compares MFI score interpretations and two baseline semantics: GPU-first (paper-path) versus global feasible-candidate search.\n\n")
        f.write(f"- GPUs: {args.gpus}\n- Repetitions per distribution: {args.repetitions}\n- Offered-demand target fraction: {args.target_fraction}\n- Arrival gap: {args.arrival_gap}\n- Scenario distributions: {', '.join(SCENARIOS)}\n- Policy runs: {len(raw_runs)}\n- Per-job decision rows: {len(per_job_rows)}\n- Checkpoint rows: {len(checkpoints)}\n- Elapsed wall time: {manifest['elapsed_seconds']:.1f} s\n\n")
        f.write("## Acceptance summary\n\n")
        f.write("| Scenario | Policy | Mean acceptance | SD | Mean decision time (µs) |\n|---|---|---:|---:|---:|\n")
        for row in summary_rows:
            f.write(f"| {row['scenario']} | {row['policy']} | {row['acceptance_mean']:.4f} | {row['acceptance_sd']:.4f} | {row['mean_decision_time_ns']/1000:.2f} |\n")
        mfi_changed_mean = statistics.mean(float(row["decision_change_fraction"]) for row in decision_diff_rows) if decision_diff_rows else 0.0
        nonzero_acceptance_deltas = sum(abs(float(row["acceptance_delta_free_anchor_minus_all_starts"])) > 1e-12 for row in decision_diff_rows)
        f.write("\n## Figure 3 score reconstruction\n\n")
        f.write("The reconstructed Figure 3 states produce scores 8 and 16 under `free_anchor_starts`; `all_legal_starts` gives 14 and 23 on the same states. See `M12_FIGURE3_CONFORMANCE.md` for resource layouts and per-profile contributions.\n\n")
        f.write(f"Across matched MFI score-variant runs, an average of {mfi_changed_mean:.1%} of per-job placement/admission decisions differed. {nonzero_acceptance_deltas} of {len(decision_diff_rows)} paired scenario/seed runs had a nonzero acceptance-rate difference.\n\n")
        f.write("## Interpretation constraints\n\n")
        f.write("- The `mfi_free_anchor_score` mode matches the reported F(GPU1)=8 and F(GPU2)=16 values for the documented Figure 3 state reconstruction.\n")
        f.write("- The earlier `mfi_all_starts` mode gives different score values for those same states, showing that the paper's numeric example constrains the scoring interpretation.\n")
        f.write("- `paper_bf_bi`/`paper_wf_bi` select a GPU using aggregate residuals before checking that selected GPU's legal placement. The existing `global_feasible_*` policies consider feasible placements across all GPUs; these are different algorithms and must not share the same baseline label.\n")
        f.write("- The experiment is synthetic and uses operational assumptions. Do not describe it as reproducing the authors' original code or as a result on physical GPUs.\n")
    print(f"Saved M12 study to {args.out.resolve()}")

if __name__ == "__main__":
    main()
