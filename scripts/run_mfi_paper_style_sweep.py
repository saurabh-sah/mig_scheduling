"""Run MFI-inspired progressive offered-load sweeps.

Default repetitions are deliberately small for a quick local smoke run. For a
large reproduction-oriented run, pass --repetitions 500. This script is not
advertised as a bit-for-bit reproduction of the authors' original experiments.
"""
from __future__ import annotations

# Allow running directly from a source checkout before editable installation.
import sys
from pathlib import Path as _BootstrapPath
sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1] / "src"))

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

from mig_scheduler.schedulers import (
    FirstFitScheduler, RoundRobinScheduler, BestFitBestIndexScheduler,
    WorstFitBestIndexScheduler, MFIScheduler,
)
from mig_scheduler.simulator import DiscreteEventSimulator
from mig_scheduler.workloads import generate_progressive_load_workload

POLICIES = (FirstFitScheduler, RoundRobinScheduler, BestFitBestIndexScheduler,
            WorstFitBestIndexScheduler, MFIScheduler)
SCENARIOS = ("uniform", "skew_small", "skew_big", "bimodal")
CHECKPOINTS = tuple(i / 10 for i in range(1, 11))
METRICS = (
    "acceptance_rate_so_far", "accepted_so_far", "active_gpus",
    "compute_utilization", "memory_utilization", "mean_gpu_fragmentation_score",
)


def first_snapshot_at_or_after(snapshots: list[dict], threshold: float) -> dict:
    for snap in snapshots:
        if snap["offered_demand_fraction"] >= threshold:
            return snap
    return snapshots[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="MFI-inspired load sweep (not exact reproduction)")
    parser.add_argument("--gpus", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=10,
                        help="Use 500 for the paper's reported number of independent simulations")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out-prefix", type=Path, default=Path("results/m2_paper_style"))
    args = parser.parse_args()
    if args.gpus < 1 or args.repetitions < 1:
        raise SystemExit("--gpus and --repetitions must be positive")

    raw_rows = []
    # Every policy receives the same workload for each (scenario, repetition).
    for scenario_i, scenario in enumerate(SCENARIOS):
        for rep in range(args.repetitions):
            seed = args.seed + scenario_i * 1_000_000 + rep
            jobs = generate_progressive_load_workload(gpu_count=args.gpus, scenario=scenario, seed=seed)
            for policy_cls in POLICIES:
                result = DiscreteEventSimulator(args.gpus, policy_cls()).run(jobs)
                for threshold in CHECKPOINTS:
                    snap = first_snapshot_at_or_after(result.snapshots, threshold)
                    row = {
                        "scenario": scenario, "repetition": rep, "seed": seed,
                        "policy": result.policy, "requested_load_checkpoint": threshold,
                        "actual_offered_demand_fraction": snap["offered_demand_fraction"],
                        "submitted_so_far": snap["submitted_so_far"],
                    }
                    row.update({metric: snap[metric] for metric in METRICS})
                    row["mean_decision_time_ns"] = result.mean_decision_time_ns
                    row["total_decision_time_ns"] = result.total_decision_time_ns
                    raw_rows.append(row)
            print(f"finished scenario={scenario} repetition={rep + 1}/{args.repetitions} jobs={len(jobs)}")

    raw_path = args.out_prefix.with_name(args.out_prefix.name + "_raw.csv")
    summary_path = args.out_prefix.with_name(args.out_prefix.name + "_summary.csv")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_rows[0]) if raw_rows else [])
        if raw_rows:
            writer.writeheader(); writer.writerows(raw_rows)

    groups: dict[tuple, list[dict]] = defaultdict(list)
    key_fields = ("scenario", "policy", "requested_load_checkpoint")
    for row in raw_rows:
        groups[tuple(row[k] for k in key_fields)].append(row)
    summary_rows = []
    for key, group in sorted(groups.items()):
        row = dict(zip(key_fields, key))
        row["repetitions"] = len(group)
        for metric in (*METRICS, "mean_decision_time_ns", "total_decision_time_ns"):
            values = [float(x[metric]) for x in group]
            row[f"{metric}_mean"] = statistics.mean(values)
            row[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary_rows.append(row)
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]) if summary_rows else [])
        if summary_rows:
            writer.writeheader(); writer.writerows(summary_rows)
    print(f"Saved raw checkpoints: {raw_path.resolve()}")
    print(f"Saved aggregate means/SDs: {summary_path.resolve()}")
    print("Note: inspect EXPERIMENT_PROTOCOL.md before describing this as a reproduction.")


if __name__ == "__main__":
    main()
