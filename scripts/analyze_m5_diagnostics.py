"""Analyze Milestone-5 matched scheduling runs and create diagnostic plots/report.

The outputs are exploratory diagnostics. Small seed counts do not establish
statistical significance, and raw elapsed runtime can vary with machine load.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict, Counter
from pathlib import Path


T_CRITICAL_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(row: dict[str, str], key: str, default: float | None = None) -> float | None:
    value = row.get(key, "")
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def mean_sd(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def t_interval(values: list[float]) -> tuple[float | None, float | None]:
    """Return an exploratory 95% Student-t interval; require >=3 pairs."""
    if len(values) < 3:
        # Fewer than three paired observations are too weak to present as an
        # interval; in particular, two identical outcomes can produce a
        # misleading zero-width interval.
        return None, None
    avg = statistics.mean(values)
    sd = statistics.stdev(values)
    if sd <= 1e-12:
        # A zero sample variance is not enough to claim a zero-width inferential
        # interval from a tiny synthetic sample. Keep the observed mean, omit CI.
        return None, None
    crit = T_CRITICAL_95.get(len(values) - 1, 1.96)
    half = crit * sd / math.sqrt(len(values))
    return avg - half, avg + half


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def _maybe_make_plots(out_dir: Path, summary_rows: list[dict], paired_rows: list[dict], run_rows: list[dict]) -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "Matplotlib is not installed; CSV and Markdown outputs were created, but plots were skipped. Install with: pip install -e '.[analysis]'"

    # 1. Acceptance by scenario and policy label.
    scenarios = sorted({row["scenario"] for row in summary_rows})
    policies = sorted({row["policy_label"] for row in summary_rows})
    fig, ax = plt.subplots(figsize=(12, 6))
    for policy in policies:
        mapping = {r["scenario"]: float(r["acceptance_mean"]) for r in summary_rows if r["policy_label"] == policy and r.get("acceptance_mean") not in (None, "")}
        if mapping:
            ax.plot([s for s in scenarios if s in mapping], [mapping[s] for s in scenarios if s in mapping], marker="o", label=policy)
    ax.set_title("Acceptance rate by workload scenario")
    ax.set_ylabel("Acceptance rate")
    ax.set_xlabel("Workload scenario")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "acceptance_by_scenario.png", dpi=160)
    plt.close(fig)

    # 2. Acceptance-runtime trade-off. Plot mean runtime measured per simulator run.
    fig, ax = plt.subplots(figsize=(9, 6))
    for row in summary_rows:
        x = row.get("wall_clock_seconds_mean")
        y = row.get("acceptance_mean")
        if x not in (None, "") and y not in (None, ""):
            ax.scatter(float(x), float(y), alpha=0.8)
            ax.annotate(f"{row['scenario']}\n{row['policy_label']}", (float(x), float(y)), fontsize=6, alpha=0.8)
    ax.set_title("Acceptance versus wall-clock simulation runtime")
    ax.set_xlabel("Mean wall-clock seconds per simulation (environment-dependent)")
    ax.set_ylabel("Mean acceptance rate")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "acceptance_vs_wallclock.png", dpi=160)
    plt.close(fig)

    # 3. Adaptive policy diagnostic rates.
    adaptive = [r for r in summary_rows if str(r["policy_label"]).startswith("adaptive_")]
    if adaptive:
        xlabels = [f"{r['scenario']}\n{r['policy_label']}" for r in adaptive]
        look = [float(r.get("adaptive_lookahead_fraction_of_all_jobs") or 0) for r in adaptive]
        fallback = [float(r.get("adaptive_fallback_fraction_of_eligible_jobs") or 0) for r in adaptive]
        diff = [float(r.get("adaptive_fraction_choices_different_from_mfi") or 0) for r in adaptive]
        x = list(range(len(adaptive)))
        fig, ax = plt.subplots(figsize=(max(12, len(adaptive) * 0.7), 6))
        ax.bar([v - 0.25 for v in x], look, width=0.25, label="Lookahead / all arrivals")
        ax.bar(x, fallback, width=0.25, label="MFI fallback / eligible arrivals")
        ax.bar([v + 0.25 for v in x], diff, width=0.25, label="Choices different from MFI")
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=65, ha="right", fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Fraction")
        ax.set_title("Adaptive scheduler diagnostics")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "adaptive_diagnostics.png", dpi=160)
        plt.close(fig)

    # 4. Paired acceptance deltas against MFI.
    policies = sorted({r["policy_label"] for r in paired_rows if r["policy_label"] != "mfi"})
    values_by_policy = {
        policy: [float(row["acceptance_diff_vs_mfi"]) for row in paired_rows if row["policy_label"] == policy and row.get("acceptance_diff_vs_mfi") not in (None, "")]
        for policy in policies
    }
    values_by_policy = {p: v for p, v in values_by_policy.items() if v}
    if values_by_policy:
        fig, ax = plt.subplots(figsize=(max(9, len(values_by_policy) * 0.9), 6))
        ax.boxplot(list(values_by_policy.values()), tick_labels=list(values_by_policy.keys()), showmeans=True)
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_title("Paired acceptance-rate difference versus MFI")
        ax.set_ylabel("Acceptance-rate difference (policy minus MFI)")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "paired_acceptance_delta_vs_mfi.png", dpi=160)
        plt.close(fig)

    return "Plots created with Matplotlib. Wall-clock runtime is environment-dependent and should be reported alongside decision-time measurements."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Directory containing runs.csv and decisions.csv")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    run_dir = args.run_dir
    out_dir = args.out_dir or run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = read_csv(run_dir / "runs.csv")
    decisions = read_csv(run_dir / "decisions.csv")
    if not runs:
        raise SystemExit("runs.csv contains no result rows")

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    lookup: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in runs:
        grouped[(row["scenario"], row["policy_label"])].append(row)
        lookup[(row["scenario"], row["seed"], row["policy_label"])] = row

    summary_rows: list[dict] = []
    for (scenario, policy_label), rows in sorted(grouped.items()):
        acceptance = [float(r["acceptance_rate"]) for r in rows]
        runtime_us = [float(r["mean_decision_time_ns"]) / 1000 for r in rows]
        wall = [float(r["wall_clock_seconds"]) for r in rows]
        frag = [float(r["mean_time_weighted_fragmentation_score"]) for r in rows]
        rolls = [float(r["adaptive_mean_total_rollouts"]) for r in rows if r.get("adaptive_mean_total_rollouts") not in (None, "")]
        lookfrac = [float(r["lookahead_fraction_of_all_jobs"]) for r in rows if r.get("lookahead_fraction_of_all_jobs") not in (None, "")]
        fallfrac = [float(r["adaptive_fallback_fraction_of_eligible_jobs"]) for r in rows if r.get("adaptive_fallback_fraction_of_eligible_jobs") not in (None, "")]
        rel = [float(r["adaptive_mean_forecast_reliability"]) for r in rows if r.get("adaptive_mean_forecast_reliability") not in (None, "")]
        diff_rate = [float(r["fraction_choices_different_from_mfi"]) for r in rows if r.get("fraction_choices_different_from_mfi") not in (None, "")]
        paired_diffs = []
        for row in rows:
            mfi = lookup.get((scenario, row["seed"], "mfi"))
            if mfi:
                paired_diffs.append(float(row["acceptance_rate"]) - float(mfi["acceptance_rate"]))
        acc_mean, acc_sd = mean_sd(acceptance)
        run_mean, run_sd = mean_sd(runtime_us)
        wall_mean, wall_sd = mean_sd(wall)
        frag_mean, frag_sd = mean_sd(frag)
        diff_mean, diff_sd = mean_sd(paired_diffs)
        ci_low, ci_high = t_interval(paired_diffs)
        rolls_mean, _ = mean_sd(rolls)
        look_mean, _ = mean_sd(lookfrac)
        fall_mean, _ = mean_sd(fallfrac)
        rel_mean, _ = mean_sd(rel)
        choice_diff_mean, _ = mean_sd(diff_rate)
        summary_rows.append({
            "scenario": scenario,
            "policy_label": policy_label,
            "runs": len(rows),
            "acceptance_mean": acc_mean,
            "acceptance_sd": acc_sd,
            "acceptance_diff_vs_mfi_mean": diff_mean,
            "acceptance_diff_vs_mfi_sd": diff_sd,
            "acceptance_diff_vs_mfi_ci95_low": ci_low,
            "acceptance_diff_vs_mfi_ci95_high": ci_high,
            "paired_runs": len(paired_diffs),
            "paired_ci_status": (
                "available_approximate_t_interval" if ci_low is not None
                else "omitted_fewer_than_3_pairs_or_zero_observed_variance"
            ),
            "mean_decision_time_us_mean": run_mean,
            "mean_decision_time_us_sd": run_sd,
            "wall_clock_seconds_mean": wall_mean,
            "wall_clock_seconds_sd": wall_sd,
            "mean_time_weighted_fragmentation_mean": frag_mean,
            "mean_time_weighted_fragmentation_sd": frag_sd,
            "adaptive_mean_rollouts": rolls_mean,
            "lookahead_fraction_of_all_jobs": look_mean,
            "adaptive_lookahead_fraction_of_all_jobs": look_mean if policy_label.startswith("adaptive_") else None,
            "adaptive_fallback_fraction_of_eligible_jobs": fall_mean,
            "adaptive_mean_forecast_reliability": rel_mean,
            "fraction_choices_different_from_mfi": choice_diff_mean,
            "adaptive_fraction_choices_different_from_mfi": choice_diff_mean if policy_label.startswith("adaptive_") else None,
        })

    paired_rows: list[dict] = []
    seed_map = {(r["scenario"], r["seed"]): r for r in runs if r["policy_label"] == "mfi"}
    for row in runs:
        if row["policy_label"] == "mfi":
            continue
        mfi = seed_map.get((row["scenario"], row["seed"]))
        if not mfi:
            continue
        paired_rows.append({
            "scenario": row["scenario"], "seed": row["seed"], "policy_label": row["policy_label"],
            "policy_acceptance": row["acceptance_rate"], "mfi_acceptance": mfi["acceptance_rate"],
            "acceptance_diff_vs_mfi": float(row["acceptance_rate"]) - float(mfi["acceptance_rate"]),
            "policy_mean_decision_time_ns": row["mean_decision_time_ns"],
            "mfi_mean_decision_time_ns": mfi["mean_decision_time_ns"],
        })

    # Summarize fallback and stopping reasons using raw decision rows.
    fallback_counter: Counter[tuple[str, str, str]] = Counter()
    stop_counter: Counter[tuple[str, str, str]] = Counter()
    decision_group_counts: Counter[tuple[str, str, str, str]] = Counter()
    for row in decisions:
        label = row["policy_label"]
        if not label.startswith("adaptive_"):
            continue
        scenario = row["scenario"]
        if row.get("fallback_reason"):
            fallback_counter[(scenario, label, row["fallback_reason"])] += 1
        if row.get("adaptive_stop_reason"):
            stop_counter[(scenario, label, row["adaptive_stop_reason"])] += 1
        status = row.get("adaptive_status") or "none"
        decision_group_counts[(scenario, label, "status", status)] += 1
    fallback_rows = [
        {"scenario": s, "policy_label": p, "fallback_reason": reason, "count": n}
        for (s, p, reason), n in sorted(fallback_counter.items())
    ]
    stop_rows = [
        {"scenario": s, "policy_label": p, "stop_reason": reason, "count": n}
        for (s, p, reason), n in sorted(stop_counter.items())
    ]
    status_rows = [
        {"scenario": s, "policy_label": p, "metric": metric, "value": value, "count": n}
        for (s, p, metric, value), n in sorted(decision_group_counts.items())
    ]

    _write_csv(out_dir / "summary.csv", summary_rows)
    _write_csv(out_dir / "paired_differences_vs_mfi.csv", paired_rows)
    _write_csv(out_dir / "adaptive_fallback_reasons.csv", fallback_rows)
    _write_csv(out_dir / "adaptive_stop_reasons.csv", stop_rows)
    _write_csv(out_dir / "adaptive_status_counts.csv", status_rows)
    plot_note = _maybe_make_plots(out_dir, summary_rows, paired_rows, runs)

    # Produce a report that states observed pilot numbers without pretending they are final.
    lines = [
        "# Milestone 5 — Adaptive Scheduler Diagnostic Report",
        "",
        "## Scope",
        "",
        "This is an exploratory diagnostic report from synthetic, matched-seed simulations. It does not establish that the proposed scheduler is generally superior to MFI, does not reproduce all details of the MFI paper's published evaluation, and is not a driver-validated MIG emulator.",
        "",
        f"Runs analysed: **{len(runs)}** policy/workload executions. Decision records: **{len(decisions)}**.",
        "",
        "## Scenario-level results",
        "",
        "Acceptance differences are paired by scenario and seed against MFI. Confidence intervals are exploratory. Intervals are omitted when fewer than three paired seeds are available or when all observed paired differences are identical, because a zero-width interval from a tiny synthetic sample would be misleading. A confidence interval containing zero should not be interpreted as equivalence.",
        "",
        "| Scenario | Policy | Runs | Acceptance mean ± SD | Paired Δ vs MFI | Paired 95% CI | Mean decision time (µs) | Lookahead fraction | Different choices vs MFI |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        if r["policy_label"] not in {"mfi", "fixed_lookahead", "adaptive_conservative", "adaptive_default", "adaptive_permissive"}:
            continue
        acc = "n/a" if r["acceptance_mean"] is None else f"{r['acceptance_mean']:.3f} ± {r['acceptance_sd']:.3f}"
        delta = "n/a" if r["acceptance_diff_vs_mfi_mean"] is None else f"{r['acceptance_diff_vs_mfi_mean']:+.3f}"
        if r["acceptance_diff_vs_mfi_ci95_low"] is None:
            ci = "n/a"
        else:
            ci = f"[{r['acceptance_diff_vs_mfi_ci95_low']:+.3f}, {r['acceptance_diff_vs_mfi_ci95_high']:+.3f}]"
        time_us = "n/a" if r["mean_decision_time_us_mean"] is None else f"{r['mean_decision_time_us_mean']:.1f}"
        look = "n/a" if r["lookahead_fraction_of_all_jobs"] is None else f"{r['lookahead_fraction_of_all_jobs']:.2f}"
        diff = "n/a" if r["fraction_choices_different_from_mfi"] is None else f"{r['fraction_choices_different_from_mfi']:.2f}"
        lines.append(f"| {r['scenario']} | {r['policy_label']} | {r['runs']} | {acc} | {delta} | {ci} | {time_us} | {look} | {diff} |")
    # Descriptive pooled comparison across scenario/seed pairs. Scenario-level
    # rows above remain primary because pooled averages can hide workload effects.
    pooled_by_policy: dict[str, list[dict[str, str]]] = defaultdict(list)
    run_by_key = {(r["scenario"], r["seed"], r["policy_label"]): r for r in runs}
    for row in paired_rows:
        pooled_by_policy[row["policy_label"]].append(row)
    lines += [
        "",
        "## Descriptive cross-scenario comparison",
        "",
        "This table pools the matched scenario/seed runs to provide one compact diagnostic view. It is descriptive only; the per-scenario table above should be used to identify workload-specific effects.",
        "",
        "| Policy | Paired runs | Mean paired Δ acceptance vs MFI | Approx. 95% CI | Mean decision time (µs) | Mean decision-time ratio vs MFI | Lookahead fraction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for policy_label in sorted(pooled_by_policy):
        prs = pooled_by_policy[policy_label]
        deltas = [float(r["acceptance_diff_vs_mfi"]) for r in prs]
        delta_avg = statistics.mean(deltas) if deltas else None
        ci_l, ci_h = t_interval(deltas)
        times = [float(run_by_key[(r["scenario"], r["seed"], policy_label)]["mean_decision_time_ns"]) / 1000 for r in prs]
        ratios = []
        lookfracs = []
        for r in prs:
            own = run_by_key[(r["scenario"], r["seed"], policy_label)]
            baseline = run_by_key[(r["scenario"], r["seed"], "mfi")]
            base_time = float(baseline["mean_decision_time_ns"])
            own_time = float(own["mean_decision_time_ns"])
            if base_time > 0:
                ratios.append(own_time / base_time)
            lookfracs.append(float(own.get("lookahead_fraction_of_all_jobs") or 0))
        ci_text = "n/a" if ci_l is None else f"[{ci_l:+.3f}, {ci_h:+.3f}]"
        lines.append(
            f"| {policy_label} | {len(prs)} | {delta_avg:+.3f} | {ci_text} | "
            f"{statistics.mean(times):.1f} | {statistics.mean(ratios):.2f}× | {statistics.mean(lookfracs):.2f} |"
        )
    lines += [
        "",
        "## How to interpret the diagnostics",
        "",
        "- `adaptive_fallback_reasons.csv` shows why adaptive lookahead chose the MFI fallback (e.g. insufficient history, unstable forecast, or a non-ambiguous MFI decision).",
        "- `adaptive_stop_reasons.csv` shows whether a lookahead decision stopped after the approximate paired-confidence check or exhausted its rollout budget.",
        "- `adaptive_status_counts.csv` counts lookahead, fallback, and no-feasible-placement decisions.",
        "- `paired_differences_vs_mfi.csv` preserves run-level paired comparisons. Inspect the distribution; do not report only the average.",
        "- `acceptance_vs_wallclock.png` uses whole-run wall-clock time, which is environment-dependent. Use per-decision time alongside it.",
        "- `lookahead_fraction_of_all_jobs` counts arrival decisions with at least one rollout; it applies to fixed and adaptive lookahead, not just adaptive variants.",
        "",
        "## Limitations and next decision",
        "",
        "1. These are controlled synthetic workloads; they are not production traces.",
        "2. A profile-distribution stability score is a heuristic, not a calibrated probability of forecast correctness.",
        "3. The current approximate paired t-based stopping rule needs calibration and should not be treated as formal sequential error control.",
        "4. Results depend on the modelled MIG profile/placement abstraction and the chosen scenario and load parameters.",
        "5. A small seed count is for debugging and direction-finding only. Use more independent workload seeds and held-out evaluation before research claims.",
        "6. The adaptive method should only be retained if it creates a useful acceptance/overhead trade-off and differs meaningfully from existing work.",
        "",
        "## Artifacts",
        "",
        "- `summary.csv` — per-scenario/per-policy means, standard deviations, paired differences, and runtime metrics.",
        "- `paired_differences_vs_mfi.csv` — matched run-level differences.",
        "- `adaptive_fallback_reasons.csv`, `adaptive_stop_reasons.csv`, `adaptive_status_counts.csv` — adaptive diagnostics.",
        "- `decisions.csv` from the runner — per-arrival records and candidate-score JSON.",
        f"- Plot status: {plot_note}",
        "",
        "No scheduler is labelled a winner by this report. A result is a finding only after the experiment configuration has been predeclared, enough seeds have been run, and all policies have been compared on matched workloads.",
    ]
    (out_dir / "M5_DIAGNOSTIC_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote summary and diagnostic report to {out_dir.resolve()}")
    print(plot_note)


if __name__ == "__main__":
    main()
