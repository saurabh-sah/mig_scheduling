"""Analyze M8 progressive-load experiment CSV and generate reproducible reports."""
from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = (
    "accepted_so_far",
    "acceptance_rate_so_far",
    "active_gpus",
    "compute_utilization",
    "memory_utilization",
    "mean_gpu_fragmentation_score",
)


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean_ci(values: list[float]) -> tuple[float, float, float, float]:
    if not values:
        return math.nan, math.nan, math.nan, math.nan
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    se = sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
    # Normal approximation, adequate for descriptive tables; paired bootstrap below
    # is preferred for policy differences.
    return mean, sd, mean - 1.96 * se, mean + 1.96 * se


def paired_bootstrap_ci(values: list[float], *, reps: int = 4000, seed: int = 20261026) -> tuple[float, float, float]:
    if not values:
        return math.nan, math.nan, math.nan
    observed = statistics.mean(values)
    if len(values) == 1:
        return observed, observed, observed
    rng = random.Random(seed)
    samples = []
    n = len(values)
    for _ in range(reps):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        samples.append(statistics.mean(sample))
    samples.sort()
    lo = samples[int(0.025 * (reps - 1))]
    hi = samples[int(0.975 * (reps - 1))]
    return observed, lo, hi


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_csv", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/m8_protocol_validation"))
    args = parser.parse_args()
    rows = read_rows(args.raw_csv)
    if not rows:
        raise SystemExit(f"No rows found in {args.raw_csv}")
    args.out.mkdir(parents=True, exist_ok=True)

    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    paired: dict[tuple[str, int, str, str], dict] = {}
    for row in rows:
        key = (row["scenario"], row["policy"], row["checkpoint_requested"])
        groups[key].append(row)
        paired[(row["scenario"], int(row["repetition"]), row["policy"], row["checkpoint_requested"])] = row

    summary = []
    for (scenario, policy, checkpoint), group in sorted(groups.items(), key=lambda item: (item[0][0], float(item[0][2]), item[0][1])):
        output = {"scenario": scenario, "policy": policy, "checkpoint_requested": float(checkpoint), "n": len(group)}
        output["mean_actual_offered_demand_fraction"] = statistics.mean(float(r["actual_offered_demand_fraction"]) for r in group)
        for metric in METRICS:
            values = [float(r[metric]) for r in group]
            mean, sd, lo, hi = mean_ci(values)
            output[f"{metric}_mean"] = mean
            output[f"{metric}_sd"] = sd
            output[f"{metric}_ci95_lo_normal"] = lo
            output[f"{metric}_ci95_hi_normal"] = hi
        output["mean_decision_time_ns"] = statistics.mean(float(r["mean_decision_time_ns"]) for r in group)
        summary.append(output)
    write_csv(args.out / "m8_progressive_load_summary.csv", summary)

    policy_names = sorted({row["policy"] for row in rows})
    scenarios = sorted({row["scenario"] for row in rows})
    checkpoints = sorted({float(row["checkpoint_requested"]) for row in rows})
    deltas = []
    for scenario in scenarios:
        for checkpoint in checkpoints:
            checkpoint_s = f"{checkpoint:.1f}"
            mfi_by_rep = {
                int(row["repetition"]): row for row in rows
                if row["scenario"] == scenario and row["policy"] == "mfi" and row["checkpoint_requested"] == checkpoint_s
            }
            for policy in policy_names:
                if policy == "mfi":
                    continue
                differences = []
                for row in rows:
                    if row["scenario"] == scenario and row["policy"] == policy and row["checkpoint_requested"] == checkpoint_s:
                        rep = int(row["repetition"])
                        if rep in mfi_by_rep:
                            differences.append(float(row["acceptance_rate_so_far"]) - float(mfi_by_rep[rep]["acceptance_rate_so_far"]))
                observed, lo, hi = paired_bootstrap_ci(differences, seed=20261026 + int(checkpoint * 10))
                deltas.append({
                    "scenario": scenario,
                    "policy": policy,
                    "checkpoint_requested": checkpoint,
                    "repetitions": len(differences),
                    "mean_paired_acceptance_delta_vs_mfi": observed,
                    "paired_bootstrap_95ci_lo": lo,
                    "paired_bootstrap_95ci_hi": hi,
                    "interval_excludes_zero": bool(lo > 0 or hi < 0),
                })
    write_csv(args.out / "m8_paired_acceptance_delta_vs_mfi.csv", deltas)

    # Aggregate per-trace final results to support a compact overview.
    final_checkpoint = max(checkpoints)
    final_rows = [r for r in rows if float(r["checkpoint_requested"]) == final_checkpoint]
    final_summary = []
    for policy in policy_names:
        group = [r for r in final_rows if r["policy"] == policy]
        acceptance = [float(r["acceptance_rate_so_far"]) for r in group]
        decision_times = [float(r["mean_decision_time_ns"]) for r in group]
        active = [float(r["active_gpus"]) for r in group]
        mean_acc, sd_acc, lo_acc, hi_acc = mean_ci(acceptance)
        final_summary.append({
            "policy": policy,
            "n_traces": len(group),
            "mean_acceptance_rate": mean_acc,
            "sd_acceptance_rate": sd_acc,
            "normal_approx_ci95_low": lo_acc,
            "normal_approx_ci95_high": hi_acc,
            "mean_decision_time_ns": statistics.mean(decision_times) if decision_times else math.nan,
            "mean_active_gpus_at_last_arrival": statistics.mean(active) if active else math.nan,
        })
    write_csv(args.out / "m8_final_checkpoint_summary.csv", final_summary)

    report_path = args.out / "M8_PROTOCOL_VALIDATION_REPORT.md"
    manifest_path = args.raw_csv.parent / "m8_run_manifest.json"
    manifest_info = ""
    if manifest_path.exists():
        import json
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_info = (
            f"- GPUs: {manifest['gpus']}\n- Repetitions per scenario: {manifest['repetitions_per_scenario']}\n"
            f"- Elapsed wall-clock seconds: {manifest['elapsed_wall_seconds']:.2f}\n"
        )
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Milestone 8 — MFI Protocol Validation Report\n\n")
        f.write("## Purpose\n\n")
        f.write("This experiment improves transparency and reproducibility of our MFI-inspired progressive-load evaluation. "
                "It does not establish an exact replication of the original paper or validate physical MIG behavior.\n\n")
        f.write("## Run configuration\n\n" + manifest_info + f"- Raw input rows: {len(rows)}\n- Scenarios: {', '.join(scenarios)}\n"
                f"- Policies: {', '.join(policy_names)}\n- Offered-load checkpoints: {', '.join(str(x) for x in checkpoints)}\n\n")
        f.write("## Final checkpoint summary\n\n")
        f.write("| Policy | Mean acceptance | SD | Mean decision time (ns) | Mean active GPUs at final arrival |\n|---|---:|---:|---:|---:|\n")
        for r in final_summary:
            f.write(f"| {r['policy']} | {r['mean_acceptance_rate']:.4f} | {r['sd_acceptance_rate']:.4f} | "
                    f"{r['mean_decision_time_ns']:.1f} | {r['mean_active_gpus_at_last_arrival']:.2f} |\n")
        f.write("\n## Interpretation rules\n\n")
        f.write("- Compare policies using identical workload seeds; the runner does this.\n")
        f.write("- Treat paired intervals as descriptive unless repetition counts and experimental design justify confirmatory inference.\n")
        f.write("- Do not compare our numeric outcomes directly with the paper's figures unless the model and protocol are matched.\n")
        f.write("- The current GPU state uses separate compute and memory occupancy arrays with legal starts from the MFI profile table. This is an explicit software abstraction.\n")
        f.write("- The progressive workload generator targets 7 scalar offered-load units per GPU using the MFI table's `Slice No.` units; duration is sampled uniformly from integer slots 1..T after T is known.\n")
        f.write("- Our index-placement tie-breakers are operationalizations where the paper provides prose but not original source code.\n\n")
        f.write("## Artifacts\n\n- `m8_progressive_load_raw.csv`: per-policy, per-repetition, per-checkpoint data.\n")
        f.write("- `m8_progressive_load_summary.csv`: means, SDs, and approximate intervals.\n")
        f.write("- `m8_paired_acceptance_delta_vs_mfi.csv`: paired differences and bootstrap intervals.\n")
        f.write("- `m8_final_checkpoint_summary.csv`: end-of-offered-load summary.\n")
        f.write("\n## Reference\n\n")
        f.write("Zambianco, Fasol, and Doriguzzi-Corin, *An Online Fragmentation-Aware GPU Scheduler for Multi-Tenant MIG-based Clouds*, arXiv:2511.18906. https://arxiv.org/abs/2511.18906\n")

    try:
        import matplotlib.pyplot as plt
        for metric, y_label, filename in (
            ("acceptance_rate_so_far", "Acceptance rate so far", "m8_acceptance_curves.png"),
            ("accepted_so_far", "Accepted workloads so far", "m8_accepted_curves.png"),
            ("mean_gpu_fragmentation_score", "Mean per-GPU fragmentation score", "m8_fragmentation_curves.png"),
        ):
            fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
            for ax, scenario in zip(axes.flat, scenarios):
                for policy in policy_names:
                    points = [r for r in summary if r["scenario"] == scenario and r["policy"] == policy]
                    points.sort(key=lambda r: float(r["checkpoint_requested"]))
                    if not points:
                        continue
                    x = [100 * float(r["checkpoint_requested"]) for r in points]
                    y = [float(r[f"{metric}_mean"]) for r in points]
                    ax.plot(x, y, marker="o", markersize=2.5, linewidth=1.2, label=policy)
                ax.set_title(scenario.replace("_", " "))
                ax.set_xlabel("Requested offered-load checkpoint (%)")
                ax.set_ylabel(y_label)
                ax.grid(True, alpha=0.25)
            handles, labels = axes.flat[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8)
            fig.tight_layout(rect=(0, 0.09, 1, 1))
            fig.savefig(args.out / filename, dpi=160)
            plt.close(fig)
    except ImportError:
        pass
    print(f"Saved analysis to {args.out.resolve()}")
    print(f"Report: {(args.out / 'M8_PROTOCOL_VALIDATION_REPORT.md').resolve()}")


if __name__ == "__main__":
    main()
