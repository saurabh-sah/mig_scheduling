"""M10: source-protocol candidate vs explicitly labelled workload-dynamics stress tests.

This is a sensitivity study, not an exact reproduction of the MFI paper. All
modes use the same sampled profile sequence within each (scenario, seed); only
arrival timing and/or job-duration distribution change. The paper-candidate mode
uses the current MFI-inspired protocol interpretation documented in M10 docs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mig_scheduler.paper_baselines import PaperFirstFitScheduler, PaperRoundRobinScheduler
from mig_scheduler.profiles import MFI_WORKLOAD_DISTRIBUTIONS, PROFILES
from mig_scheduler.schedulers import BestFitBestIndexScheduler, MFIScheduler, WorstFitBestIndexScheduler
from mig_scheduler.simulator import DiscreteEventSimulator
from mig_scheduler.workloads import generate_progressive_load_workload

POLICY_FACTORIES = {
    "paper_ff": PaperFirstFitScheduler,
    "paper_rr": PaperRoundRobinScheduler,
    "bf_bi": BestFitBestIndexScheduler,
    "wf_bi": WorstFitBestIndexScheduler,
    "mfi": MFIScheduler,
}
MODES = (
    "paper_candidate",
    "burst_arrivals",
    "sparse_arrivals",
    "short_lifetimes",
    "long_lifetimes",
    "no_completion_before_stream_end",
)
DEFAULT_CHECKPOINTS = (0.50, 0.70, 0.85, 1.00)


def make_modes(base_jobs, seed: int) -> dict[str, list]:
    """Create sensitivity variants preserving the profile sequence and job IDs."""
    n = len(base_jobs)
    T = max(1, n)  # paper-candidate interpretation: one arrival slot per job.
    out = {"paper_candidate": list(base_jobs)}
    out["burst_arrivals"] = [replace(job, arrival_time=0.0) for job in base_jobs]
    out["sparse_arrivals"] = [replace(job, arrival_time=float(i * 2)) for i, job in enumerate(base_jobs)]

    rng_short = random.Random(seed + 10_000_019)
    short_hi = max(1, T // 4)
    out["short_lifetimes"] = [
        replace(job, arrival_time=float(i), duration=float(rng_short.randint(1, short_hi)))
        for i, job in enumerate(base_jobs)
    ]

    rng_long = random.Random(seed + 20_000_033)
    long_lo = max(1, math.ceil(0.75 * T))
    out["long_lifetimes"] = [
        replace(job, arrival_time=float(i), duration=float(rng_long.randint(long_lo, T)))
        for i, job in enumerate(base_jobs)
    ]

    # Force no positive-duration job to finish before the last arrival (at n-1).
    out["no_completion_before_stream_end"] = [
        replace(job, arrival_time=float(i), duration=float(T + 1))
        for i, job in enumerate(base_jobs)
    ]
    return out


def select_snapshot(snapshots: list[dict], checkpoint: float) -> dict:
    for snap in snapshots:
        if snap["offered_demand_fraction"] >= checkpoint:
            return snap
    return snapshots[-1]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def bootstrap_ci_mean(values: list[float], *, seed: int, n_boot: int = 3000) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean; intended for exploratory summaries."""
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(n_boot):
        means.append(statistics.mean(values[rng.randrange(n)] for __ in range(n)))
    means.sort()
    return means[int(0.025 * n_boot)], means[min(n_boot - 1, int(0.975 * n_boot))]


def build_aggregates(raw_rows: list[dict], out: Path, seed_base: int) -> tuple[list[dict], list[dict]]:
    groups = defaultdict(list)
    for row in raw_rows:
        groups[(row["scenario"], row["mode"], row["policy"], row["checkpoint"])].append(row)
    summary = []
    for (scenario, mode, policy, checkpoint), group in sorted(groups.items()):
        row = {"scenario": scenario, "mode": mode, "policy": policy,
               "checkpoint": checkpoint, "repetitions": len(group)}
        for metric in ("acceptance_rate", "accepted_so_far", "active_gpus", "compute_utilization",
                       "memory_utilization", "mean_gpu_fragmentation_score", "mean_decision_time_ns",
                       "submitted_so_far", "offered_demand_fraction"):
            values = [float(r[metric]) for r in group]
            row[f"{metric}_mean"], row[f"{metric}_sd"] = mean_sd(values)
        summary.append(row)

    # Pair MFI to BF-BI on the same profile trace and mode at each checkpoint.
    paired = defaultdict(dict)
    for row in raw_rows:
        key = (row["scenario"], row["repetition"], row["seed"], row["mode"], row["checkpoint"])
        paired[key][row["policy"]] = row
    delta_groups = defaultdict(list)
    for key, policies in paired.items():
        if "mfi" not in policies or "bf_bi" not in policies:
            continue
        delta = float(policies["mfi"]["acceptance_rate"]) - float(policies["bf_bi"]["acceptance_rate"])
        delta_groups[(key[0], key[3], key[4])].append(delta)
    deltas = []
    for (scenario, mode, checkpoint), values in sorted(delta_groups.items()):
        lo, hi = bootstrap_ci_mean(values, seed=seed_base + len(deltas) + 1)
        deltas.append({
            "scenario": scenario, "mode": mode, "checkpoint": checkpoint,
            "repetitions": len(values), "mfi_minus_bfbi_acceptance_delta_mean": statistics.mean(values),
            "paired_delta_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "paired_delta_bootstrap_ci_low": lo, "paired_delta_bootstrap_ci_high": hi,
            "positive_mfi_repetitions": sum(v > 1e-12 for v in values),
            "negative_mfi_repetitions": sum(v < -1e-12 for v in values),
            "tied_repetitions": sum(abs(v) <= 1e-12 for v in values),
        })
    write_csv(out / "m10_summary.csv", summary)
    write_csv(out / "m10_paired_mfi_minus_bfbi.csv", deltas)
    return summary, deltas


def make_plots(summary: list[dict], deltas: list[dict], out: Path) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ["Plots not generated: install the optional analysis dependency (matplotlib)."]

    messages = []
    final_cp = max(float(r["checkpoint"]) for r in summary)
    for scenario in sorted({r["scenario"] for r in summary}):
        subset = [r for r in summary if r["scenario"] == scenario and float(r["checkpoint"]) == final_cp]
        policies = list(POLICY_FACTORIES)
        modes = list(MODES)
        fig, ax = plt.subplots(figsize=(11, 5.5))
        width = 0.14
        xs = list(range(len(modes)))
        for pi, policy in enumerate(policies):
            vals = []
            for mode in modes:
                hit = next((r for r in subset if r["mode"] == mode and r["policy"] == policy), None)
                vals.append(float(hit["acceptance_rate_mean"]) if hit else 0.0)
            ax.bar([x + (pi - (len(policies)-1)/2)*width for x in xs], vals, width=width, label=policy)
        ax.set_xticks(xs, [m.replace("_", "\n") for m in modes])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Acceptance rate at final offered-load checkpoint")
        ax.set_title(f"M10 protocol/dynamics sensitivity — {scenario}")
        ax.legend(ncol=3)
        fig.tight_layout()
        file = out / f"m10_acceptance_modes_{scenario}.png"
        fig.savefig(file, dpi=160)
        plt.close(fig)
        messages.append(file.name)

    # Pooled paired delta by mode at final load; descriptive only.
    vals = []
    labels = []
    for mode in MODES:
        group = [float(r["mfi_minus_bfbi_acceptance_delta_mean"]) for r in deltas
                 if r["mode"] == mode and float(r["checkpoint"]) == final_cp]
        if group:
            vals.append(statistics.mean(group))
            labels.append(mode.replace("_", "\n"))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(range(len(vals)), vals)
    ax.axhline(0, linewidth=1)
    ax.set_xticks(range(len(vals)), labels)
    ax.set_ylabel("MFI − BF-BI acceptance rate")
    ax.set_title("Paired acceptance difference by protocol/dynamics mode")
    fig.tight_layout()
    file = out / "m10_paired_delta_by_mode.png"
    fig.savefig(file, dpi=160)
    plt.close(fig)
    messages.append(file.name)
    return messages


def render_report(out: Path, gpus: int, reps: int, seed_base: int, rows: list[dict], summary: list[dict], deltas: list[dict], plots: list[str]) -> None:
    final_cp = max(float(r["checkpoint"]) for r in summary)
    report = [
        "# M10 — MFI Protocol and Workload-Dynamics Audit",
        "",
        "## Scope",
        "",
        "This is a protocol sensitivity study, not an exact replication of MFI. It separates the paper-candidate interpretation from explicitly labelled stress tests. Every mode preserves the same generated profile sequence for each scenario/seed; variants change only arrival timing and/or durations.",
        "",
        f"- GPUs: {gpus}", f"- Repetitions per profile distribution: {reps}", f"- Seed base: {seed_base}",
        f"- Policy runs: {len(rows) // len(DEFAULT_CHECKPOINTS)}", f"- Checkpoint rows: {len(rows)}", "",
        "## Protocol modes",
        "",
        "| Mode | What changes | Status |",
        "|---|---|---|",
        "| `paper_candidate` | One arrival per slot; durations sampled uniformly from integer slots 1…T; cumulative MFI Table-I `Slice No.` reaches the offered-load target. | Closest operationalization of stated paper protocol; unresolved source ambiguities remain. |",
        "| `burst_arrivals` | Same profiles and durations; all arrivals at time 0. | Stress test only. |",
        "| `sparse_arrivals` | Same profiles and durations; inter-arrival spacing doubles. | Stress test only. |",
        "| `short_lifetimes` | Same profiles; duration uniform 1…floor(T/4). | Stress test only. |",
        "| `long_lifetimes` | Same profiles; duration uniform ceil(0.75T)…T. | Stress test only. |",
        "| `no_completion_before_stream_end` | Same profiles; duration T+1, so no positive-duration job completes before the final arrival. | Stress test only. |",
        "",
        "## Interpretation limits",
        "",
        "The MFI paper says it uses a 100-GPU homogeneous cluster, profiles from Table I, the four distributions in Table II, durations sampled from [1,T], rejected requests are not rescheduled, and 500 independent simulations per profile distribution. It does not release source code and leaves some implementation details underspecified. This repository models compute and memory occupancy separately and explicitly operationalizes arrival timing and the BF/WF GPU residual score. These results must not be represented as an exact replication.",
        "",
        "The paper's offered-load x-axis is cumulative requested `Slice No.` divided by the cluster's scalar capacity; it explicitly does not account for terminations when calculating this axis. Our stress modes intentionally preserve that axis while changing concurrent occupancy. This isolates the effect of workload dynamics from the profile sequence.",
        "",
        "## MFI versus BF-BI at final offered load",
        "",
        "| Scenario | Mode | Mean paired delta (MFI − BF-BI) | Bootstrap 95% interval | MFI positive / tie / negative repetitions |",
        "|---|---|---:|---:|---:|",
    ]
    for scenario in sorted({r["scenario"] for r in deltas}):
        for mode in MODES:
            group = [r for r in deltas if r["scenario"] == scenario and r["mode"] == mode and float(r["checkpoint"]) == final_cp]
            if not group:
                continue
            # Current runner reports each scenario/mode group as one row.
            r = group[0]
            report.append(
                f"| {scenario} | {mode} | {float(r['mfi_minus_bfbi_acceptance_delta_mean']):.4f} | "
                f"[{float(r['paired_delta_bootstrap_ci_low']):.4f}, {float(r['paired_delta_bootstrap_ci_high']):.4f}] | "
                f"{r['positive_mfi_repetitions']} / {r['tied_repetitions']} / {r['negative_mfi_repetitions']} |"
            )
    report += ["", "## Conclusions (descriptive only)", "",
               "- The `paper_candidate` rows are the main replication-oriented comparison; the other modes are stress tests and must be reported separately.",
               "- A change in acceptance across stress modes indicates sensitivity to arrival/lifetime semantics; it is not proof that the paper's protocol was implemented incorrectly.",
               "- Do not select a protocol because it creates a larger MFI-vs-baseline gap. Any mismatch with the paper requires a source-grounded explanation.",
               "- Do not claim an algorithmic improvement from this audit. The purpose is to clarify the workload dynamics and protect later experiments from protocol ambiguity.", "",
               "## Figures", ""]
    report.extend(f"- `{plot}`" for plot in plots)
    report += ["", "## Source", "", "Zambianco, Fasol, Doriguzzi-Corin, *An Online Fragmentation-Aware GPU Scheduler for Multi-Tenant MIG-based Clouds*, arXiv:2511.18906, especially Table I, Table II, Sections IV–VI. https://arxiv.org/abs/2511.18906", ""]
    (out / "M10_PROTOCOL_DYNAMICS_REPORT.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, default=20, help="Use 100 for a paper-sized cluster; default is quicker.")
    parser.add_argument("--repetitions", type=int, default=5, help="Exploratory default; use much larger for final evaluation.")
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--out", type=Path, default=Path("results/m10_protocol_dynamics"))
    args = parser.parse_args()
    if args.gpus < 1 or args.repetitions < 1:
        raise SystemExit("--gpus and --repetitions must be positive")

    args.out.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict] = []
    total = len(MFI_WORKLOAD_DISTRIBUTIONS) * args.repetitions * len(MODES) * len(POLICY_FACTORIES)
    complete = 0
    start_time = time.perf_counter()

    for scenario_i, scenario in enumerate(MFI_WORKLOAD_DISTRIBUTIONS):
        for rep in range(args.repetitions):
            seed = args.seed + scenario_i * 1_000_000 + rep
            base_jobs = generate_progressive_load_workload(
                gpu_count=args.gpus, scenario=scenario, seed=seed,
                arrival_gap=1.0, load_basis="paper_slice_units", target_fraction=1.0,
            )
            modes = make_modes(base_jobs, seed)
            for mode in MODES:
                jobs = modes[mode]
                # Ensure every experiment mode keeps the same profile sequence and job identities.
                assert [j.profile_name for j in jobs] == [j.profile_name for j in base_jobs]
                assert [j.job_id for j in jobs] == [j.job_id for j in base_jobs]
                for policy, factory in POLICY_FACTORIES.items():
                    result = DiscreteEventSimulator(args.gpus, factory()).run(jobs)
                    complete += 1
                    for cp in DEFAULT_CHECKPOINTS:
                        snap = select_snapshot(result.snapshots, cp)
                        raw_rows.append({
                            "scenario": scenario, "repetition": rep, "seed": seed,
                            "mode": mode, "policy": policy, "checkpoint": cp,
                            "gpu_count": args.gpus, "trace_jobs": len(base_jobs),
                            "duration_T_candidate": len(base_jobs),
                            "offered_demand_fraction": snap["offered_demand_fraction"],
                            "submitted_so_far": snap["submitted_so_far"],
                            "accepted_so_far": snap["accepted_so_far"],
                            "rejected_so_far": snap["rejected_so_far"],
                            "acceptance_rate": snap["acceptance_rate_so_far"],
                            "active_gpus": snap["active_gpus"],
                            "compute_utilization": snap["compute_utilization"],
                            "memory_utilization": snap["memory_utilization"],
                            "mean_gpu_fragmentation_score": snap["mean_gpu_fragmentation_score"],
                            "mean_decision_time_ns": result.mean_decision_time_ns,
                            "whole_trace_accepted": result.accepted,
                            "whole_trace_acceptance_rate": result.acceptance_rate,
                        })
                print(f"[{complete}/{total}] scenario={scenario} rep={rep+1}/{args.repetitions} mode={mode}")

    write_csv(args.out / "m10_raw_checkpoints.csv", raw_rows)
    summary, deltas = build_aggregates(raw_rows, args.out, args.seed)
    plots = make_plots(summary, deltas, args.out)
    manifest = {
        "milestone": 10,
        "study": "MFI source-protocol candidate versus workload-dynamics stress tests",
        "source_paper": "Zambianco, Fasol, Doriguzzi-Corin, An Online Fragmentation-Aware GPU Scheduler for Multi-Tenant MIG-based Clouds",
        "source_url": "https://arxiv.org/abs/2511.18906",
        "gpus": args.gpus, "repetitions_per_distribution": args.repetitions,
        "seed_base": args.seed, "scenarios": list(MFI_WORKLOAD_DISTRIBUTIONS),
        "policies": list(POLICY_FACTORIES), "modes": list(MODES),
        "checkpoints": list(DEFAULT_CHECKPOINTS), "raw_rows": len(raw_rows),
        "simulation_runs": complete,
        "elapsed_wall_seconds": time.perf_counter() - start_time,
        "duration_note": "T is operationalized as number of sampled jobs in the full scalar-offered-load trace; this interpretation is not fully specified by the paper text.",
        "arrival_note": "One job per slot in paper_candidate is an explicit operationalization of arrivals distributed in time.",
        "fidelity_note": "Exploratory protocol audit, not exact reproduction. Other modes are stress tests only.",
    }
    (args.out / "m10_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    render_report(args.out, args.gpus, args.repetitions, args.seed, raw_rows, summary, deltas, plots)
    print(f"\nSaved protocol report: {(args.out / 'M10_PROTOCOL_DYNAMICS_REPORT.md').resolve()}")
    print(f"Saved raw checkpoints: {(args.out / 'm10_raw_checkpoints.csv').resolve()}")
    print(f"Completed {complete} simulations in {manifest['elapsed_wall_seconds']:.1f}s")


if __name__ == "__main__":
    main()
