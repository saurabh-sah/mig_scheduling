"""Summarize a matched M6 budget-frontier experiment and create plots."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, stdev


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    mu = mean(values)
    if len(values) < 2:
        return mu, float("nan"), float("nan")
    se = stdev(values) / math.sqrt(len(values))
    return mu, mu - 1.96 * se, mu + 1.96 * se


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    out = args.out_dir or args.input_csv.parent
    out.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.input_csv)
    if not rows:
        raise SystemExit("Input CSV is empty")

    # Each (scenario, seed) trace is a matched unit across all policies.
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["policy"], []).append(row)
    policy_names = sorted(grouped)
    summary_rows = []
    for policy in policy_names:
        items = grouped[policy]
        acc, acc_lo, acc_hi = mean_ci([float(r["acceptance_rate"]) for r in items])
        dec, dec_lo, dec_hi = mean_ci([float(r["decision_time_mean_us"]) for r in items])
        total, total_lo, total_hi = mean_ci([float(r["decision_time_total_ms"]) for r in items])
        wall, wall_lo, wall_hi = mean_ci([float(r["wall_clock_run_s"]) for r in items])
        summary_rows.append({
            "policy": policy, "matched_trace_count": len(items),
            "mean_acceptance_rate": acc,
            "acceptance_ci95_low_normal_approx": acc_lo,
            "acceptance_ci95_high_normal_approx": acc_hi,
            "mean_decision_time_us": dec,
            "decision_time_ci95_low_normal_approx_us": dec_lo,
            "decision_time_ci95_high_normal_approx_us": dec_hi,
            "mean_total_decision_time_ms": total,
            "total_decision_time_ci95_low_normal_approx_ms": total_lo,
            "total_decision_time_ci95_high_normal_approx_ms": total_hi,
            "mean_wall_clock_run_s": wall,
            "mean_lookahead_decisions": mean([float(r["lookahead_decisions"]) for r in items]),
            "mean_adaptive_lookahead_decisions": mean([float(r["adaptive_lookahead_decisions"]) for r in items]),
            "mean_adaptive_fallback_decisions": mean([float(r["adaptive_fallback_decisions"]) for r in items]),
            "mean_total_rollout_evaluations_per_arrival": mean([float(r["rollout_evaluations_per_arrival"] or 0) for r in items]),
            "mean_placements_different_from_mfi": mean([float(r["placements_different_from_mfi"] or 0) for r in items]),
            "mean_fraction_comparable_placements_different_from_mfi": mean([float(r["fraction_comparable_placements_different_from_mfi"] or 0) for r in items]),
            "mean_time_weighted_fragmentation": mean([float(r["mean_fragmentation_score_time_weighted"]) for r in items]),
            "mean_peak_fragmentation_score": mean([float(r["peak_fragmentation_score"]) for r in items]),
        })
    write_csv(out / "policy_budget_summary.csv", summary_rows)

    mfi_by_trace = {
        (r["scenario"], r["seed"]): r
        for r in rows if r["policy"] == "mfi"
    }
    paired_rows = []
    paired_summary = []
    for policy in policy_names:
        if policy == "mfi":
            continue
        diffs = []
        time_diffs = []
        for row in rows:
            if row["policy"] != policy:
                continue
            key = (row["scenario"], row["seed"])
            base = mfi_by_trace.get(key)
            if base is None:
                continue
            delta_acc = float(row["acceptance_rate"]) - float(base["acceptance_rate"])
            delta_time = float(row["decision_time_total_ms"]) - float(base["decision_time_total_ms"])
            diffs.append(delta_acc)
            time_diffs.append(delta_time)
            paired_rows.append({
                "scenario": row["scenario"], "seed": row["seed"], "policy": policy,
                "mfi_acceptance_rate": float(base["acceptance_rate"]),
                "policy_acceptance_rate": float(row["acceptance_rate"]),
                "paired_acceptance_delta": delta_acc,
                "mfi_decision_time_ms": float(base["decision_time_total_ms"]),
                "policy_decision_time_ms": float(row["decision_time_total_ms"]),
                "paired_decision_time_delta_ms": delta_time,
            })
        dmean, dlo, dhi = mean_ci(diffs)
        tmean, tlo, thi = mean_ci(time_diffs)
        paired_summary.append({
            "policy": policy, "paired_trace_count": len(diffs),
            "mean_paired_acceptance_delta_vs_mfi": dmean,
            "acceptance_delta_ci95_low_normal_approx": dlo,
            "acceptance_delta_ci95_high_normal_approx": dhi,
            "mean_paired_decision_time_delta_ms_vs_mfi": tmean,
            "decision_time_delta_ci95_low_normal_approx_ms": tlo,
            "decision_time_delta_ci95_high_normal_approx_ms": thi,
            "traces_with_positive_acceptance_delta": sum(x > 0 for x in diffs),
            "traces_with_zero_acceptance_delta": sum(x == 0 for x in diffs),
            "traces_with_negative_acceptance_delta": sum(x < 0 for x in diffs),
        })
    write_csv(out / "paired_vs_mfi.csv", paired_rows)
    write_csv(out / "paired_summary_vs_mfi.csv", paired_summary)

    scenario_rows = []
    for scenario in sorted({r["scenario"] for r in rows}):
        for policy in policy_names:
            items = [r for r in rows if r["scenario"] == scenario and r["policy"] == policy]
            if not items:
                continue
            scenario_rows.append({
                "scenario": scenario, "policy": policy, "trace_count": len(items),
                "mean_acceptance_rate": mean([float(r["acceptance_rate"]) for r in items]),
                "mean_decision_time_us": mean([float(r["decision_time_mean_us"]) for r in items]),
                "mean_total_decision_time_ms": mean([float(r["decision_time_total_ms"]) for r in items]),
                "mean_lookahead_decisions": mean([float(r["lookahead_decisions"]) for r in items]),
            })
    write_csv(out / "scenario_budget_summary.csv", scenario_rows)

    plot_status = "Plots not generated (matplotlib unavailable)."
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 6))
        for row in summary_rows:
            ax.scatter(row["mean_decision_time_us"], row["mean_acceptance_rate"], label=row["policy"])
            ax.annotate(row["policy"], (row["mean_decision_time_us"], row["mean_acceptance_rate"]), fontsize=8)
        ax.set_xlabel("Mean scheduling decision time (microseconds)")
        ax.set_ylabel("Mean workload acceptance rate")
        ax.set_title("Acceptance versus decision cost (matched workload traces)")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / "acceptance_vs_decision_cost.png", dpi=160)
        plt.close(fig)

        if paired_summary:
            fig, ax = plt.subplots(figsize=(10, 6))
            labels = [r["policy"] for r in paired_summary]
            values = [r["mean_paired_acceptance_delta_vs_mfi"] for r in paired_summary]
            lows = [r["mean_paired_acceptance_delta_vs_mfi"] - r["acceptance_delta_ci95_low_normal_approx"] for r in paired_summary]
            highs = [r["acceptance_delta_ci95_high_normal_approx"] - r["mean_paired_acceptance_delta_vs_mfi"] for r in paired_summary]
            y = list(range(len(labels)))
            ax.errorbar(values, y, xerr=[lows, highs], fmt="o", capsize=3)
            ax.axvline(0, linewidth=1)
            ax.set_yticks(y, labels)
            ax.set_xlabel("Paired acceptance-rate difference versus MFI")
            ax.set_title("Exploratory paired differences (normal-approximation intervals)")
            ax.grid(True, axis="x", alpha=0.25)
            fig.tight_layout()
            fig.savefig(out / "paired_acceptance_delta_vs_mfi.png", dpi=160)
            plt.close(fig)
        plot_status = "Generated acceptance-versus-cost and paired-difference plots."
    except ImportError:
        pass

    report = [
        "# M6 Budget-Frontier Analysis",
        "",
        "## Purpose",
        "",
        "Compare MFI, fixed-budget lookahead at several rollout counts, and adaptive lookahead with several maximum rollout budgets on matched synthetic workloads.",
        "This is an exploratory simulator study; it is not a hardware benchmark or exact replication of the MFI paper.",
        "",
        "## Policy summary",
        "",
        "| Policy | Mean acceptance | Mean decision time (µs) | Mean total decision time (ms) | Mean rollout evaluations / arrival | Mean placement-change fraction vs MFI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        report.append(
            f"| {r['policy']} | {r['mean_acceptance_rate']:.3f} | {r['mean_decision_time_us']:.3f} "
            f"| {r['mean_total_decision_time_ms']:.3f} | {r['mean_total_rollout_evaluations_per_arrival']:.2f} "
            f"| {r['mean_fraction_comparable_placements_different_from_mfi']:.3f} |"
        )
    report += ["", "## Paired differences versus MFI", "", "| Policy | Mean acceptance delta | Approx. 95% interval | Positive / zero / negative traces |", "|---|---:|---:|---:|"]
    for r in paired_summary:
        report.append(
            f"| {r['policy']} | {r['mean_paired_acceptance_delta_vs_mfi']:+.4f} "
            f"| [{r['acceptance_delta_ci95_low_normal_approx']:+.4f}, {r['acceptance_delta_ci95_high_normal_approx']:+.4f}] "
            f"| {r['traces_with_positive_acceptance_delta']} / {r['traces_with_zero_acceptance_delta']} / {r['traces_with_negative_acceptance_delta']} |"
        )
    report += [
        "",
        "## Interpretation rules",
        "",
        "- All policies in a scenario/seed pair receive the same generated job trace.",
        "- The reported intervals are normal-approximation descriptive intervals. They are exploratory; use a paired t interval or paired bootstrap for final claims, with enough independent seeds.",
        "- Measured wall-clock time includes Python implementation and host-machine effects. It is useful for within-run comparisons, not a deployment-grade latency guarantee.",
        "- A higher acceptance rate is not enough by itself; compare gains with decision-time overhead and the number of decisions that actually changed.",
        "- `adaptive_max2/4/8` controls a maximum rollout count, but the adaptive policy may fallback and thus use less; inspect rollout and fallback counts.",
        "- Do not tune on the same seeds used for final conclusions. Reserve held-out seeds after selecting parameters.",
        "",
        "## Artifacts",
        "",
        "- `budget_frontier_runs.csv`: all raw matched runs.",
        "- `policy_budget_summary.csv`: overall descriptive summary.",
        "- `paired_vs_mfi.csv`: trace-paired differences.",
        "- `paired_summary_vs_mfi.csv`: paired aggregate summary.",
        "- `scenario_budget_summary.csv`: scenario-specific behaviour.",
        f"- {plot_status}",
    ]
    (out / "M6_BUDGET_FRONTIER_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote analysis to {out.resolve()}")
    print(plot_status)


if __name__ == "__main__":
    main()
