"""Aggregate M9 arrival-spacing sweep and generate plots/research report."""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("results/m9_arrival_spacing_audit"))
    args = parser.parse_args()
    raw = read_csv(args.input_dir / "m9_arrival_spacing_runs.csv")
    comparisons = read_csv(args.input_dir / "m9_mfi_vs_bfbi_decisions.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Install the optional analysis dependency: pip install -e '.[analysis]'") from exc

    policies = sorted({r["policy"] for r in raw})
    gaps = sorted({float(r["arrival_gap"]) for r in raw})
    colors = None
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for policy in policies:
        x, y, err = [], [], []
        for gap in gaps:
            rows = [r for r in raw if r["policy"] == policy and float(r["arrival_gap"]) == gap]
            vals = [float(r["acceptance_rate"]) for r in rows]
            x.append(gap)
            y.append(statistics.mean(vals))
            err.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)
        ax.errorbar(x, y, yerr=err, marker="o", capsize=3, label=policy)
    ax.set_xlabel("Inter-arrival gap (0 = burst)")
    ax.set_ylabel("Workload acceptance rate")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Acceptance sensitivity to workload arrival spacing")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.input_dir / "m9_acceptance_vs_arrival_gap.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for policy in policies:
        x, y = [], []
        for gap in gaps:
            rows = [r for r in raw if r["policy"] == policy and float(r["arrival_gap"]) == gap]
            x.append(gap)
            y.append(statistics.mean(float(r["peak_arrival_compute_utilization"]) for r in rows))
        ax.plot(x, y, marker="o", label=policy)
    ax.set_xlabel("Inter-arrival gap (0 = burst)")
    ax.set_ylabel("Peak compute occupancy at an arrival")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Peak occupancy observed by each scheduling policy")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.input_dir / "m9_peak_compute_utilization_vs_arrival_gap.png", dpi=160)
    plt.close(fig)

    # Report overall summaries averaged across scenarios and repetitions.
    out = []
    for gap in gaps:
        for policy in policies:
            rows = [r for r in raw if r["policy"] == policy and float(r["arrival_gap"]) == gap]
            accepts = [float(r["acceptance_rate"]) for r in rows]
            compute = [float(r["peak_arrival_compute_utilization"]) for r in rows]
            runtime = [float(r["mean_decision_time_ns"]) for r in rows]
            out.append({
                "arrival_gap": gap, "policy": policy, "trace_policy_runs": len(rows),
                "mean_acceptance": statistics.mean(accepts),
                "sd_acceptance_across_runs": statistics.stdev(accepts) if len(accepts) > 1 else 0.0,
                "mean_peak_compute_utilization": statistics.mean(compute),
                "mean_decision_time_ns": statistics.mean(runtime),
            })
    with (args.input_dir / "m9_overall_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)

    lines = [
        "# M9 — Baseline Conformance and Arrival-Spacing Sensitivity Report", "",
        "## Purpose", "",
        "M8 showed 100% acceptance for MFI and BF-BI at the final cumulative offered-load checkpoint. "
        "This audit tests whether the result is robust to the temporal spacing of the same profile/duration trace.", "",
        "## Design", "",
        "- Same profile order and service duration for every arrival-gap value within a (scenario, seed) trace.",
        "- Only arrival times are changed: `arrival_time = job_index * arrival_gap`.",
        "- Arrival gap 0 is a burst where all arrivals occur at time zero before any positive-duration job completes.",
        "- Policies: paper-style FF/RR operationalizations, BF-BI, WF-BI, and MFI.",
        "- Scenarios: uniform, skew-small, skew-big, and bimodal.",
        "- This is a sensitivity experiment, not exact reproduction of MFI's original arrival process.", "",
        "## Overall results", "",
        "| Arrival gap | Policy | Mean acceptance | Mean peak compute occupancy | Mean decision time (ns) |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in out:
        lines.append(
            f"| {row['arrival_gap']:.3g} | {row['policy']} | {row['mean_acceptance']:.4f} | "
            f"{row['mean_peak_compute_utilization']:.4f} | {row['mean_decision_time_ns']:.0f} |"
        )
    lines.extend([
        "", "## MFI vs BF-BI placement diagnostics", "",
        "| Arrival gap | Mean acceptance delta (MFI − BF-BI) | Mean status disagreements | Mean placement difference fraction |",
        "|---:|---:|---:|---:|",
    ])
    for gap in gaps:
        rows = [r for r in comparisons if float(r["arrival_gap"]) == gap]
        if not rows:
            continue
        lines.append(
            f"| {gap:.3g} | {statistics.mean(float(r['acceptance_delta_mfi_minus_bf_bi']) for r in rows):.4f} | "
            f"{statistics.mean(int(r['status_disagreements']) for r in rows):.2f} | "
            f"{statistics.mean(float(r['placement_difference_fraction']) for r in rows):.4f} |"
        )
    lines.extend([
        "", "## Interpretation cautions", "",
        "1. Changing the arrival gap changes concurrency and is deliberately not a direct reproduction of the paper's protocol.",
        "2. Arrival-gap 0 is a burst/admission stress test, not a realistic continuous arrival process.",
        "3. Mean acceptance across this experiment is descriptive; repetitions are exploratory unless the final design uses adequate power.",
        "4. Runtime is Python simulator overhead and is not deployment scheduler latency.",
        "5. Occupancy is a modeled resource metric, not physical GPU compute utilization.",
        "6. An acceptance difference at a given gap would show sensitivity to load timing, not by itself prove which algorithm is universally better.",
        "", "## Figures", "",
        "- `m9_acceptance_vs_arrival_gap.png`",
        "- `m9_peak_compute_utilization_vs_arrival_gap.png`",
        "", "## Sources", "",
        "- Zambianco, Fasol, Doriguzzi-Corin, *An Online Fragmentation-Aware GPU Scheduler for Multi-Tenant MIG-based Clouds*, https://arxiv.org/abs/2511.18906",
        "- Turkkan et al., *Optimal Workload Placement on Multi-Instance GPUs*, https://arxiv.org/abs/2409.06646",
    ])
    (args.input_dir / "M9_ARRIVAL_SPACING_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved plots and reports in {args.input_dir.resolve()}")


if __name__ == "__main__":
    main()
