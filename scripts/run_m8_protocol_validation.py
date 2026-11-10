"""MFI paper-protocol-oriented progressive-load study.

This is a transparent replication-oriented implementation, not a claim of exact
replication or hardware validation. It uses the repository's explicit dual
compute/memory occupancy abstraction and the MFI paper's profile distributions,
profile table, progressive offered-load idea, 100-GPU/500-repetition target,
and baseline policies.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mig_scheduler.schedulers import (  # noqa: E402
    BestFitBestIndexScheduler,
    WorstFitBestIndexScheduler,
    MFIScheduler,
)
from mig_scheduler.paper_baselines import PaperFirstFitScheduler, PaperRoundRobinScheduler  # noqa: E402
from mig_scheduler.simulator import DiscreteEventSimulator  # noqa: E402
from mig_scheduler.workloads import generate_progressive_load_workload  # noqa: E402
from mig_scheduler.profiles import MFI_WORKLOAD_DISTRIBUTIONS, PROFILES  # noqa: E402

POLICY_FACTORIES = {
    "paper_ff": PaperFirstFitScheduler,
    "paper_rr": PaperRoundRobinScheduler,
    "bf_bi": BestFitBestIndexScheduler,
    "wf_bi": WorstFitBestIndexScheduler,
    "mfi": MFIScheduler,
}
SCENARIOS = ("uniform", "skew_small", "skew_big", "bimodal")
CHECKPOINTS = tuple(i / 10 for i in range(1, 11))
METRICS = (
    "submitted_so_far",
    "accepted_so_far",
    "rejected_so_far",
    "acceptance_rate_so_far",
    "active_gpus",
    "compute_utilization",
    "memory_utilization",
    "mean_gpu_fragmentation_score",
    "cluster_fragmentation_score",
    "actual_offered_demand_fraction",
)


def first_snapshot_at_or_after(snapshots: list[dict], threshold: float) -> dict:
    for snapshot in snapshots:
        if snapshot["offered_demand_fraction"] >= threshold:
            return snapshot
    return snapshots[-1]


def run(args: argparse.Namespace) -> tuple[Path, Path, dict]:
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    output_csv = out_dir / "m8_progressive_load_raw.csv"
    manifest_path = out_dir / "m8_run_manifest.json"

    rows: list[dict] = []
    started = time.perf_counter()
    total_simulations = len(SCENARIOS) * args.repetitions * len(POLICY_FACTORIES)
    completed = 0

    # Each scenario/repetition uses one shared job trace across all policies.
    # This makes policy comparisons paired on identical offered workloads.
    for scenario_index, scenario in enumerate(SCENARIOS):
        for repetition in range(args.repetitions):
            seed = args.seed + scenario_index * 1_000_000 + repetition
            jobs = generate_progressive_load_workload(
                gpu_count=args.gpus,
                scenario=scenario,
                seed=seed,
            )
            total_offered_units = sum(job.profile.paper_slice_units for job in jobs)
            theoretical_budget_units = 7 * args.gpus
            for policy_name, factory in POLICY_FACTORIES.items():
                result = DiscreteEventSimulator(args.gpus, factory()).run(jobs)
                completed += 1
                snapshots = result.snapshots
                for checkpoint in CHECKPOINTS:
                    snap = first_snapshot_at_or_after(snapshots, checkpoint)
                    row = {
                        "scenario": scenario,
                        "repetition": repetition,
                        "seed": seed,
                        "gpu_count": args.gpus,
                        "policy": policy_name,
                        "checkpoint_requested": checkpoint,
                        "actual_offered_demand_fraction": snap["offered_demand_fraction"],
                        "trace_job_count": len(jobs),
                        "trace_total_offered_slice_units": total_offered_units,
                        "theoretical_offered_capacity_units": theoretical_budget_units,
                        "mean_decision_time_ns": result.mean_decision_time_ns,
                        "total_decision_time_ns": result.total_decision_time_ns,
                    }
                    for metric in METRICS:
                        if metric == "actual_offered_demand_fraction":
                            row[metric] = snap["offered_demand_fraction"]
                        else:
                            row[metric] = snap[metric]
                    row["whole_trace_accepted"] = result.accepted
                    row["whole_trace_rejected"] = result.rejected
                    row["whole_trace_acceptance_rate"] = result.acceptance_rate
                    rows.append(row)
                print(
                    f"[{completed}/{total_simulations}] scenario={scenario} "
                    f"rep={repetition + 1}/{args.repetitions} policy={policy_name} "
                    f"jobs={len(jobs)} acceptance={result.acceptance_rate:.3f}"
                )

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    manifest = {
        "milestone": 8,
        "study": "MFI paper-protocol-oriented progressive offered-load validation",
        "source_paper": "Zambianco, Fasol, Doriguzzi-Corin, An Online Fragmentation-Aware GPU Scheduler for Multi-Tenant MIG-based Clouds",
        "source_url": "https://arxiv.org/abs/2511.18906",
        "gpu_model_label": "A100-80GB MFI profile/placement model; explicit dual compute/memory occupancy abstraction",
        "gpus": args.gpus,
        "repetitions_per_scenario": args.repetitions,
        "seed_base": args.seed,
        "scenarios": list(SCENARIOS),
        "profile_distributions": MFI_WORKLOAD_DISTRIBUTIONS,
        "profile_table": {
            name: {
                "compute_slices": profile.compute_slices,
                "memory_slices": profile.memory_slices,
                "paper_slice_units": profile.paper_slice_units,
                "legal_starts": list(profile.legal_starts),
            }
            for name, profile in PROFILES.items()
        },
        "policies": list(POLICY_FACTORIES),
        "checkpoints": list(CHECKPOINTS),
        "jobs_per_trace_range": [
            min(row["trace_job_count"] for row in rows),
            max(row["trace_job_count"] for row in rows),
        ] if rows else None,
        "rows": len(rows),
        "elapsed_wall_seconds": time.perf_counter() - started,
        "model_fidelity_note": (
            "Replication-oriented, not exact replication: the paper does not provide original source code; "
            "this repository uses explicit separate compute and memory occupancy and its own deterministic "
            "operationalization of some baseline index tie-breaks."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_csv, manifest_path, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, default=10,
                        help="Use 100 to match the paper's cluster size (quick default is 10).")
    parser.add_argument("--repetitions", type=int, default=10,
                        help="Use 500 for the paper's reported repetition count (quick default is 10).")
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--out", type=Path, default=Path("results/m8_protocol_validation"))
    args = parser.parse_args()
    if args.gpus < 1 or args.repetitions < 1:
        raise SystemExit("--gpus and --repetitions must be positive")
    raw_csv, manifest, run_manifest = run(args)
    print(f"\nSaved raw results: {raw_csv.resolve()}")
    print(f"Saved run manifest: {manifest.resolve()}")
    print(f"Completed {run_manifest['rows']} checkpoint rows in {run_manifest['elapsed_wall_seconds']:.1f}s")
    print("Reminder: results are replication-oriented and must not be described as exact MFI reproduction.")


if __name__ == "__main__":
    main()
