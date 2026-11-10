"""Compare workload saturation definitions under a burst-admission stress test."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mig_scheduler.paper_baselines import PaperFirstFitScheduler, PaperRoundRobinScheduler  # noqa: E402
from mig_scheduler.profiles import COMPUTE_SLICE_COUNT, MEMORY_SLICE_COUNT, MFI_WORKLOAD_DISTRIBUTIONS, PROFILES  # noqa: E402
from mig_scheduler.schedulers import BestFitBestIndexScheduler, MFIScheduler, WorstFitBestIndexScheduler  # noqa: E402
from mig_scheduler.simulator import DiscreteEventSimulator  # noqa: E402
from mig_scheduler.workloads import generate_progressive_load_workload  # noqa: E402

POLICY_FACTORIES = {
    "paper_ff": PaperFirstFitScheduler,
    "paper_rr": PaperRoundRobinScheduler,
    "bf_bi": BestFitBestIndexScheduler,
    "wf_bi": WorstFitBestIndexScheduler,
    "mfi": MFIScheduler,
}
LOAD_BASES = ("paper_slice_units", "compute", "memory", "max_normalized")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=9026)
    parser.add_argument("--out", type=Path, default=Path("results/m9_load_basis_audit"))
    args = parser.parse_args()
    if args.gpus < 1 or args.repetitions < 1:
        raise SystemExit("--gpus and --repetitions must be positive")

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    n_runs = len(MFI_WORKLOAD_DISTRIBUTIONS) * len(LOAD_BASES) * args.repetitions * len(POLICY_FACTORIES)
    done = 0
    for scenario_i, scenario in enumerate(MFI_WORKLOAD_DISTRIBUTIONS):
        for basis_i, basis in enumerate(LOAD_BASES):
            for rep in range(args.repetitions):
                seed = args.seed + scenario_i * 1_000_000 + basis_i * 10_000 + rep
                # All requests arrive at t=0 for this experiment. This isolates
                # static placement/admission under a given requested load budget.
                jobs = generate_progressive_load_workload(
                    gpu_count=args.gpus, scenario=scenario, seed=seed,
                    arrival_gap=0.0, load_basis=basis,
                )
                offered_paper_units = sum(j.profile.paper_slice_units for j in jobs)
                offered_compute = sum(j.profile.compute_slices for j in jobs)
                offered_memory = sum(j.profile.memory_slices for j in jobs)
                offered = {
                    "offered_paper_units_fraction": offered_paper_units / (7 * args.gpus),
                    "offered_compute_fraction": offered_compute / (COMPUTE_SLICE_COUNT * args.gpus),
                    "offered_memory_fraction": offered_memory / (MEMORY_SLICE_COUNT * args.gpus),
                    "job_count": len(jobs),
                }
                for policy, factory in POLICY_FACTORIES.items():
                    result = DiscreteEventSimulator(args.gpus, factory()).run(jobs)
                    snapshots = result.snapshots
                    rows.append({
                        "scenario": scenario, "load_basis": basis, "repetition": rep,
                        "seed": seed, "policy": policy, **offered,
                        "accepted": result.accepted, "rejected": result.rejected,
                        "acceptance_rate": result.acceptance_rate,
                        "mean_decision_time_ns": result.mean_decision_time_ns,
                        "mean_arrival_compute_utilization": sum(s["compute_utilization"] for s in snapshots) / len(snapshots) if snapshots else 0.0,
                        "peak_arrival_compute_utilization": max((s["compute_utilization"] for s in snapshots), default=0.0),
                        "mean_arrival_memory_utilization": sum(s["memory_utilization"] for s in snapshots) / len(snapshots) if snapshots else 0.0,
                        "peak_arrival_memory_utilization": max((s["memory_utilization"] for s in snapshots), default=0.0),
                        "mean_active_gpus_at_arrivals": sum(s["active_gpus"] for s in snapshots) / len(snapshots) if snapshots else 0.0,
                    })
                    done += 1
                    if done % 20 == 0 or done == n_runs:
                        print(f"completed {done}/{n_runs} policy runs", flush=True)

    raw_path = args.out / "m9_load_basis_runs.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "milestone": 9,
        "purpose": "test how offered-load definition changes saturation and baseline outcomes",
        "gpu_count": args.gpus, "repetitions_per_scenario_and_basis": args.repetitions,
        "seed_base": args.seed, "scenarios": list(MFI_WORKLOAD_DISTRIBUTIONS),
        "load_bases": list(LOAD_BASES), "policies": list(POLICY_FACTORIES),
        "arrival_gap": 0.0,
        "warning": "Burst stress test using alternative load bases; not an exact reproduction of MFI's arrival/duration protocol.",
    }
    (args.out / "m9_load_basis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {raw_path}")


if __name__ == "__main__":
    main()
