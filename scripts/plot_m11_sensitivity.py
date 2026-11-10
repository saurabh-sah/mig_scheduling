"""Create plots from M11 sensitivity audit CSVs."""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def read(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=ROOT / "results" / "m11_rank_sensitivity_20gpu_10rep")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out = args.results
    summary = read(out / "m11_policy_summary.csv")
    changes = read(out / "m11_decision_change_summary.csv")
    scenarios = sorted({r["scenario"] for r in summary})

    # Main baseline comparison: MFI and current default BF/WF operationalizations.
    selected = {
        "mfi": "MFI",
        "bf_bi__normalized_sum__index_before_gpu": "BF-BI default",
        "wf_bi__normalized_sum__index_before_gpu": "WF-BI default",
    }
    width = 0.24
    x = np.arange(len(scenarios))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for idx, (policy_key, label) in enumerate(selected.items()):
        vals = []
        for scenario in scenarios:
            hit = next(r for r in summary if r["scenario"] == scenario and r["policy_key"] == policy_key)
            vals.append(float(hit["acceptance_mean"]) * 100)
        ax.bar(x + (idx - 1) * width, vals, width=width, label=label)
    ax.set_xticks(x, scenarios)
    ax.set_ylabel("Workload acceptance (%)")
    ax.set_title("M11: acceptance under matched synthetic workloads")
    ax.set_ylim(0, 105)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "m11_acceptance_by_baseline.png", dpi=160)
    plt.close(fig)

    # Placement-change fraction vs the current default for each family.
    selected_changes = [r for r in changes if r["policy_key"] != "bf_bi__normalized_sum__index_before_gpu" and r["policy_key"] != "wf_bi__normalized_sum__index_before_gpu"]
    keys = sorted({r["policy_key"] for r in selected_changes})
    pretty = [k.replace("bf_bi__", "BF:").replace("wf_bi__", "WF:").replace("__", " / ") for k in keys]
    matrix = []
    for scenario in scenarios:
        row = []
        for key in keys:
            hit = next((r for r in selected_changes if r["scenario"] == scenario and r["policy_key"] == key), None)
            row.append(float(hit["mean_placement_or_admission_change_fraction"]) * 100 if hit else 0)
        matrix.append(row)
    fig, ax = plt.subplots(figsize=(15, 5.5))
    img = ax.imshow(matrix, aspect="auto")
    ax.set_yticks(range(len(scenarios)), scenarios)
    ax.set_xticks(range(len(keys)), pretty, rotation=65, ha="right")
    ax.set_title("Share of job decisions changed relative to current BF-BI/WF-BI defaults")
    ax.set_xlabel("Alternative ranking / tie policy")
    fig.colorbar(img, ax=ax, label="Decision changes (%)")
    fig.tight_layout()
    fig.savefig(out / "m11_decision_change_heatmap.png", dpi=160)
    plt.close(fig)

    # Mean decision time per policy, averaged across all traces.
    policy_keys = sorted({r["policy_key"] for r in summary})
    values = []
    for key in policy_keys:
        rows = [r for r in summary if r["policy_key"] == key]
        values.append(sum(float(r["mean_decision_time_ns"]) for r in rows) / len(rows) / 1000)
    order = sorted(range(len(values)), key=lambda i: values[i])
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(range(len(order)), [values[i] for i in order])
    ax.set_xticks(range(len(order)), [policy_keys[i] for i in order], rotation=75, ha="right")
    ax.set_ylabel("Mean decision runtime (µs)")
    ax.set_title("M11: implementation runtime by ranking variant")
    fig.tight_layout()
    fig.savefig(out / "m11_runtime_by_policy.png", dpi=160)
    plt.close(fig)

    print("Wrote m11_acceptance_by_baseline.png, m11_decision_change_heatmap.png, m11_runtime_by_policy.png")


if __name__ == "__main__":
    main()
