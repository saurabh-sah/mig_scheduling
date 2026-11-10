"""Run a matched arrival-spacing sensitivity study for MIG baselines."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mig_scheduler.paper_baselines import (  # noqa: E402
    PaperFirstFitScheduler, PaperRoundRobinScheduler,
)
from mig_scheduler.profiles import MFI_WORKLOAD_DISTRIBUTIONS  # noqa: E402
from mig_scheduler.schedulers import (  # noqa: E402
    BestFitBestIndexScheduler, MFIScheduler, WorstFitBestIndexScheduler,
)
from mig_scheduler.simulator import DiscreteEventSimulator  # noqa: E402
from mig_scheduler.workloads import generate_progressive_load_workload  # noqa: E402

POLICY_FACTORIES = {
    "paper_ff": PaperFirstFitScheduler,
    "paper_rr": PaperRoundRobinScheduler,
    "bf_bi": BestFitBestIndexScheduler,
    "wf_bi": WorstFitBestIndexScheduler,
    "mfi": MFIScheduler,
}


def _summary_row(result, scenario, repetition, seed, arrival_gap, offered_load):
    snapshots = result.snapshots
    return {
        "scenario": scenario,
        "repetition": repetition,
        "seed": seed,
        "arrival_gap": arrival_gap,
        "offered_demand_fraction": offered_load,
        "policy": result.policy,
        "submitted": result.submitted,
        "accepted": result.accepted,
        "rejected": result.rejected,
        "acceptance_rate": result.acceptance_rate,
        "mean_decision_time_ns": result.mean_decision_time_ns,
        "total_decision_time_ns": result.total_decision_time_ns,
        "mean_arrival_compute_utilization": statistics.mean(
            row["compute_utilization"] for row in snapshots
        ) if snapshots else 0.0,
        "peak_arrival_compute_utilization": max(
            (row["compute_utilization"] for row in snapshots), default=0.0
        ),
        "mean_arrival_memory_utilization": statistics.mean(
            row["memory_utilization"] for row in snapshots
        ) if snapshots else 0.0,
        "peak_arrival_memory_utilization": max(
            (row["memory_utilization"] for row in snapshots), default=0.0
        ),
        "mean_active_gpus_at_arrivals": statistics.mean(
            row["active_gpus"] for row in snapshots
        ) if snapshots else 0.0,
        "peak_arrival_fragmentation": max(
            (row["cluster_fragmentation_score"] for row in snapshots), default=0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gaps", default="0,0.1,0.5,1.0",
                        help="Arrival-time gaps; 0 creates a burst of simultaneous arrivals")
    parser.add_argument("--out", type=Path, default=Path("results/m9_arrival_spacing_audit"))
    args = parser.parse_args()
    if args.gpus < 1 or args.repetitions < 1:
        raise SystemExit("--gpus and --repetitions must be positive")
    gaps = [float(x.strip()) for x in args.gaps.split(",") if x.strip()]
    if not gaps or any(gap < 0 for gap in gaps):
        raise SystemExit("--gaps must contain one or more non-negative numbers")

    args.out.mkdir(parents=True, exist_ok=True)
    run_rows = []
    comparison_rows = []
    total_runs = len(MFI_WORKLOAD_DISTRIBUTIONS) * args.repetitions * len(gaps) * len(POLICY_FACTORIES)
    finished = 0

    for scenario_i, scenario in enumerate(MFI_WORKLOAD_DISTRIBUTIONS):
        for rep in range(args.repetitions):
            seed = args.seed + scenario_i * 1_000_000 + rep
            # Generate once at gap=1.0 so every gap has identical profile order
            # and service durations; only arrival times change.
            base_jobs = generate_progressive_load_workload(
                gpu_count=args.gpus, scenario=scenario, seed=seed, arrival_gap=1.0
            )
            offered_load = sum(job.profile.paper_slice_units for job in base_jobs) / (7 * args.gpus)
            for gap in gaps:
                jobs = [replace(job, arrival_time=i * gap) for i, job in enumerate(base_jobs)]
                results = {}
                for policy, factory in POLICY_FACTORIES.items():
                    result = DiscreteEventSimulator(args.gpus, factory()).run(jobs)
                    results[policy] = result
                    run_rows.append(_summary_row(
                        result, scenario, rep, seed, gap, offered_load
                    ))
                    finished += 1
                    if finished % 25 == 0 or finished == total_runs:
                        print(f"completed {finished}/{total_runs} runs")

                mfi = {row["job_id"]: row for row in results["mfi"].job_results}
                bf = {row["job_id"]: row for row in results["bf_bi"].job_results}
                status_diffs = 0
                placement_diffs = 0
                jointly_accepted = 0
                for job_id in mfi:
                    a, b = mfi[job_id], bf[job_id]
                    if a["status"] != b["status"]:
                        status_diffs += 1
                    if a["status"] == "accepted" and b["status"] == "accepted":
                        jointly_accepted += 1
                        if (a["gpu_id"], a["start"]) != (b["gpu_id"], b["start"]):
                            placement_diffs += 1
                comparison_rows.append({
                    "scenario": scenario, "repetition": rep, "seed": seed,
                    "arrival_gap": gap, "submitted": len(jobs),
                    "mfi_accepted": results["mfi"].accepted,
                    "bf_bi_accepted": results["bf_bi"].accepted,
                    "acceptance_delta_mfi_minus_bf_bi": (
                        results["mfi"].acceptance_rate - results["bf_bi"].acceptance_rate
                    ),
                    "status_disagreements": status_diffs,
                    "placement_differences_when_both_accepted": placement_diffs,
                    "jointly_accepted_jobs": jointly_accepted,
                    "placement_difference_fraction": (
                        placement_diffs / jointly_accepted if jointly_accepted else 0.0
                    ),
                })

    raw_path = args.out / "m9_arrival_spacing_runs.csv"
    cmp_path = args.out / "m9_mfi_vs_bfbi_decisions.csv"
    for path, rows in ((raw_path, run_rows), (cmp_path, comparison_rows)):
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)

    grouped = {}
    keys = sorted({(r["scenario"], r["arrival_gap"], r["policy"]) for r in run_rows})
    summary_rows = []
    metrics = ("acceptance_rate", "mean_decision_time_ns", "mean_arrival_compute_utilization",
               "peak_arrival_compute_utilization", "mean_arrival_memory_utilization",
               "peak_arrival_memory_utilization", "mean_active_gpus_at_arrivals",
               "peak_arrival_fragmentation")
    for scenario, gap, policy in keys:
        group = [r for r in run_rows if r["scenario"] == scenario and r["arrival_gap"] == gap and r["policy"] == policy]
        row = {"scenario": scenario, "arrival_gap": gap, "policy": policy, "repetitions": len(group)}
        for metric in metrics:
            values = [float(r[metric]) for r in group]
            row[f"{metric}_mean"] = statistics.mean(values)
            row[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary_rows.append(row)
    summary_path = args.out / "m9_arrival_spacing_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]) if summary_rows else [])
        if summary_rows:
            writer.writeheader()
            writer.writerows(summary_rows)

    manifest = {
        "milestone": 9, "purpose": "baseline conformance and arrival-spacing sensitivity",
        "gpu_count": args.gpus, "repetitions_per_scenario": args.repetitions,
        "seed_base": args.seed, "scenarios": list(MFI_WORKLOAD_DISTRIBUTIONS),
        "arrival_gaps": gaps, "policies": list(POLICY_FACTORIES),
        "profile_sequence_and_durations_matched_across_gaps": True,
        "policy_inputs_matched_within_each_trace": True,
        "warning": "Sensitivity study, not exact replication of MFI; paper does not fully specify arrival-time sampling or publish source code.",
    }
    (args.out / "m9_run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved raw runs: {raw_path}")
    print(f"Saved MFI/BF-BI decision diagnostics: {cmp_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved manifest: {args.out / 'm9_run_manifest.json'}")


if __name__ == "__main__":
    main()
