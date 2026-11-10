"""Analyze M7 horizon/decision diagnostics with stratified paired bootstrap CIs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def f(row: dict, key: str, default: float = 0.0) -> float:
    try:
        v = row.get(key)
        if v in (None, "", "None"):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def stratified_bootstrap_ci(
    delta_rows: list[dict], *, scenario_key: str = "scenario", value_key: str = "delta",
    repeats: int = 5000, seed: int = 20261026,
) -> tuple[float, float, float]:
    """Equal-weight scenarios; resample traces within each scenario stratum."""
    strata: dict[str, list[float]] = defaultdict(list)
    for row in delta_rows:
        strata[str(row[scenario_key])].append(float(row[value_key]))
    strata = {k: v for k, v in strata.items() if v}
    if not strata:
        return float("nan"), float("nan"), float("nan")
    observed = mean(mean(values) for values in strata.values())
    if repeats <= 0 or not any(len(values) > 1 for values in strata.values()):
        return observed, observed, observed
    rng = random.Random(seed)
    boot = []
    for _ in range(repeats):
        stratum_means = []
        for values in strata.values():
            sampled = [rng.choice(values) for _ in range(len(values))]
            stratum_means.append(mean(sampled))
        boot.append(mean(stratum_means))
    boot.sort()
    low_index = max(0, int(0.025 * len(boot)))
    high_index = min(len(boot) - 1, int(0.975 * len(boot)))
    return observed, boot[low_index], boot[high_index]


def policy_order(policy: str):
    if policy == "first_fit": return (0, 0, 0)
    if policy == "mfi": return (1, 0, 0)
    if policy.startswith("fixed_h"):
        try:
            h = int(policy.split("_r")[0].replace("fixed_h", ""))
            return (2, h, 0)
        except ValueError:
            return (2, 99, 0)
    if policy.startswith("adaptive_h"):
        try:
            h = int(policy.split("_max")[0].replace("adaptive_h", ""))
            return (3, h, 0)
        except ValueError:
            return (3, 99, 0)
    return (9, 99, 99)


def build_summary(run_rows: list[dict], decision_rows: list[dict], output: Path,
                  bootstrap_repeats: int) -> list[dict]:
    policies = sorted({r["policy"] for r in run_rows}, key=policy_order)
    mfi_map = {(r["scenario"], r["seed"]): r for r in run_rows if r["policy"] == "mfi"}
    paired_rows = []
    paired_summaries = []
    summaries = []
    scenario_rows = []

    for policy in policies:
        own = [r for r in run_rows if r["policy"] == policy]
        acc_values = [f(r, "acceptance_rate") for r in own]
        time_values = [f(r, "decision_time_mean_us") for r in own]
        runtime_total = [f(r, "decision_time_total_ms") for r in own]
        summary = {
            "policy": policy,
            "trace_count": len(own),
            "mean_acceptance_rate": mean(acc_values) if acc_values else 0.0,
            "mean_decision_time_us": mean(time_values) if time_values else 0.0,
            "median_decision_time_us": median(time_values) if time_values else 0.0,
            "mean_total_decision_time_ms": mean(runtime_total) if runtime_total else 0.0,
            "mean_rollouts_per_arrival": mean(f(r, "rollouts_per_arrival") for r in own) if own else 0.0,
            "mean_changed_placement_fraction": mean(f(r, "fraction_placements_different_from_mfi") for r in own) if own else 0.0,
            "mean_peak_fragmentation": mean(f(r, "peak_fragmentation_score") for r in own) if own else 0.0,
        }
        summaries.append(summary)
        for scenario in sorted({r["scenario"] for r in run_rows}):
            one = [r for r in own if r["scenario"] == scenario]
            if one:
                scenario_rows.append({
                    "policy": policy, "scenario": scenario, "trace_count": len(one),
                    "mean_acceptance_rate": mean(f(r, "acceptance_rate") for r in one),
                    "mean_decision_time_us": mean(f(r, "decision_time_mean_us") for r in one),
                    "mean_total_decision_time_ms": mean(f(r, "decision_time_total_ms") for r in one),
                    "mean_changed_placement_fraction": mean(f(r, "fraction_placements_different_from_mfi") for r in one),
                })
        if policy == "mfi":
            continue
        deltas = []
        wins = ties = losses = 0
        for row in own:
            base = mfi_map.get((row["scenario"], row["seed"]))
            if not base:
                continue
            d = f(row, "acceptance_rate") - f(base, "acceptance_rate")
            dt = f(row, "decision_time_total_ms") - f(base, "decision_time_total_ms")
            wins += d > 1e-12
            losses += d < -1e-12
            ties += abs(d) <= 1e-12
            deltas.append({"scenario": row["scenario"], "seed": row["seed"], "delta": d})
            paired_rows.append({
                "policy": policy, "scenario": row["scenario"], "seed": row["seed"],
                "acceptance_rate_mfi": f(base, "acceptance_rate"),
                "acceptance_rate_policy": f(row, "acceptance_rate"),
                "paired_acceptance_delta": d,
                "decision_time_total_mfi_ms": f(base, "decision_time_total_ms"),
                "decision_time_total_policy_ms": f(row, "decision_time_total_ms"),
                "paired_decision_time_delta_ms": dt,
            })
        observed, low, high = stratified_bootstrap_ci(
            deltas, repeats=bootstrap_repeats, seed=20261026 + sum(ord(c) for c in policy)
        )
        paired_summaries.append({
            "policy": policy, "matched_trace_count": len(deltas),
            "mean_paired_acceptance_delta_vs_mfi": observed,
            "stratified_paired_bootstrap_95_low": low,
            "stratified_paired_bootstrap_95_high": high,
            "positive_zero_negative_traces": f"{wins}/{ties}/{losses}",
            "mean_decision_cost_multiplier_vs_mfi": (
                mean(f(r, "decision_time_total_ms") for r in own)
                / max(mean(f(r, "decision_time_total_ms") for r in mfi_map.values()), 1e-12)
                if own and mfi_map else float("nan")
            ),
        })

    decision_summaries = []
    lookahead_rows = [r for r in decision_rows if r.get("policy", "").startswith(("fixed_", "adaptive_"))]
    for policy in policies:
        if not (policy.startswith("fixed_") or policy.startswith("adaptive_")):
            continue
        rows = [r for r in lookahead_rows if r.get("policy") == policy]
        comparable = [r for r in rows if r.get("choice_differs_from_mfi") not in (None, "")]
        changed = [r for r in comparable if int(f(r, "choice_differs_from_mfi")) == 1]
        value_gains = [f(r, "estimated_future_gain_vs_mfi_candidate") for r in rows if r.get("estimated_future_gain_vs_mfi_candidate") not in (None, "")]
        gaps = [f(r, "top_two_estimated_value_gap") for r in rows if r.get("top_two_estimated_value_gap") not in (None, "")]
        sds = [f(r, "top_candidate_rollout_sd") for r in rows if r.get("top_candidate_rollout_sd") not in (None, "")]
        time_us = [f(r, "decision_time_ns") / 1e3 for r in rows if r.get("decision_time_ns") not in (None, "")]
        reasons: dict[str, int] = defaultdict(int)
        for r in rows:
            reason = r.get("fallback_reason")
            if reason:
                reasons[str(reason)] += 1
        decision_summaries.append({
            "policy": policy,
            "job_rows": len(rows),
            "comparable_accepted_placements": len(comparable),
            "scored_lookahead_decisions": len(value_gains),
            "placements_different_from_mfi": len(changed),
            "placement_change_fraction": len(changed) / len(comparable) if comparable else 0.0,
            "mean_estimated_future_gain_vs_mfi_candidate": mean(value_gains) if value_gains else 0.0,
            "mean_top_two_estimated_value_gap": mean(gaps) if gaps else 0.0,
            "median_top_two_estimated_value_gap": median(gaps) if gaps else 0.0,
            "mean_top_candidate_rollout_sd": mean(sds) if sds else 0.0,
            "mean_decision_time_us_from_job_rows": mean(time_us) if time_us else 0.0,
            "adaptive_fallback_reasons": json.dumps(dict(reasons), sort_keys=True),
        })

    write_csv(output / "policy_summary.csv", summaries)
    write_csv(output / "paired_vs_mfi.csv", paired_rows)
    write_csv(output / "paired_summary_vs_mfi.csv", paired_summaries)
    write_csv(output / "scenario_summary.csv", scenario_rows)
    write_csv(output / "decision_diagnostics_summary.csv", decision_summaries)

    plot_status = "matplotlib unavailable; raw CSVs retained."
    try:
        import matplotlib.pyplot as plt
        policy_x = [r["policy"] for r in summaries]
        acceptance = [r["mean_acceptance_rate"] for r in summaries]
        runtime = [r["mean_decision_time_us"] for r in summaries]
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.bar(policy_x, acceptance)
        ax.set_ylabel("Mean acceptance rate")
        ax.set_title("M7: Acceptance by policy and lookahead horizon")
        ax.tick_params(axis="x", labelrotation=35)
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout(); fig.savefig(output / "acceptance_by_policy.png", dpi=160); plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.bar(policy_x, runtime)
        ax.set_ylabel("Mean per-decision runtime (µs, implementation-dependent)")
        ax.set_title("M7: Scheduling cost by policy and lookahead horizon")
        ax.tick_params(axis="x", labelrotation=35)
        ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout(); fig.savefig(output / "runtime_by_policy.png", dpi=160); plt.close(fig)

        if paired_summaries:
            labels = [r["policy"] for r in paired_summaries]
            vals = [100 * r["mean_paired_acceptance_delta_vs_mfi"] for r in paired_summaries]
            low = [100 * (r["mean_paired_acceptance_delta_vs_mfi"] - r["stratified_paired_bootstrap_95_low"]) for r in paired_summaries]
            high = [100 * (r["stratified_paired_bootstrap_95_high"] - r["mean_paired_acceptance_delta_vs_mfi"]) for r in paired_summaries]
            fig, ax = plt.subplots(figsize=(11, 7))
            ax.errorbar(vals, range(len(labels)), xerr=[low, high], fmt="o", capsize=3)
            ax.axvline(0, linewidth=1)
            ax.set_yticks(range(len(labels)), labels)
            ax.set_xlabel("Paired acceptance-rate delta vs MFI (percentage points)")
            ax.set_title("M7: Stratified paired-bootstrap intervals (exploratory)")
            ax.grid(True, axis="x", alpha=0.25)
            fig.tight_layout(); fig.savefig(output / "paired_delta_vs_mfi.png", dpi=160); plt.close(fig)

        diag_pol = [r for r in decision_summaries]
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.bar([r["policy"] for r in diag_pol], [100*r["placement_change_fraction"] for r in diag_pol])
        ax.set_ylabel("Accepted placements differing from MFI (%)")
        ax.set_title("M7: How often lookahead changes an accepted placement")
        ax.tick_params(axis="x", labelrotation=35)
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout(); fig.savefig(output / "placement_change_fraction.png", dpi=160); plt.close(fig)
        plot_status = "Generated acceptance, runtime, paired-bootstrap, and placement-change plots."
    except ImportError:
        pass

    report_lines = [
        "# M7 Horizon and Decision-Centric Audit",
        "",
        "## Scope",
        "",
        "This is an exploratory, simulator-only analysis. It is not a physical MIG benchmark and does not constitute exact replication of all MFI paper experiments.",
        "Each policy receives the same generated workload per scenario and seed. The bootstrap resamples traces within each workload scenario and gives scenarios equal weight.",
        "The confidence intervals are exploratory because this study uses a limited synthetic seed set and multiple policy/horizon comparisons.",
        "",
        "## Configuration",
        "",
        "See `experiment_config.json`; raw traces and per-decision diagnostics are preserved in the CSV files.",
        "",
        "## Policy summary",
        "",
        "| Policy | Mean acceptance | Mean decision time (µs) | Total decision time (ms) | Changed placement fraction | Rollouts / arrival |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in summaries:
        report_lines.append(
            f"| {r['policy']} | {r['mean_acceptance_rate']:.4f} | {r['mean_decision_time_us']:.1f} | "
            f"{r['mean_total_decision_time_ms']:.3f} | {100*r['mean_changed_placement_fraction']:.2f}% | {r['mean_rollouts_per_arrival']:.2f} |"
        )
    report_lines += [
        "", "## Paired acceptance differences vs MFI", "",
        "| Policy | Mean delta (pp) | Stratified paired-bootstrap 95% interval (pp) | Traces + / = / - | Cost multiplier |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in paired_summaries:
        report_lines.append(
            f"| {r['policy']} | {100*r['mean_paired_acceptance_delta_vs_mfi']:+.3f} | "
            f"[{100*r['stratified_paired_bootstrap_95_low']:+.3f}, {100*r['stratified_paired_bootstrap_95_high']:+.3f}] | "
            f"{r['positive_zero_negative_traces']} | {r['mean_decision_cost_multiplier_vs_mfi']:.1f}× |"
        )
    report_lines += ["", "## Decision-level diagnostics", "", "| Policy | Comparable accepted placements | Scored lookahead decisions | Changed vs MFI | Mean estimated future gain over MFI candidate | Mean top-two value gap | Mean top rollout SD |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in decision_summaries:
        report_lines.append(
            f"| {r['policy']} | {r['comparable_accepted_placements']} | {r['scored_lookahead_decisions']} | {100*r['placement_change_fraction']:.2f}% | "
            f"{r['mean_estimated_future_gain_vs_mfi_candidate']:.4f} | {r['mean_top_two_estimated_value_gap']:.4f} | {r['mean_top_candidate_rollout_sd']:.4f} |"
        )
    report_lines += [
        "", "## Interpretation guardrails", "",
        "- A placement change is not automatically an improvement: compare the matched trace's final acceptance count and decision cost.",
        "- Estimated future gain is a rollout-model value difference between the selected candidate and the candidate MFI would choose. It is not a causal estimate of the realized benefit of an individual decision.",
        "- For interpretation, compare the estimated value gap with the rollout sample standard deviation. A tiny estimated advantage relative to rollout variability is weak evidence for a placement change.",
        "- Short horizons can fail to capture a delayed benefit; longer horizons can increase model error and compute cost.",
        "- Timing measures Python decision-call wall time on this machine. They are not production latency guarantees.",
        "- The current GPU state model uses separate compute and memory occupancy masks. It is a transparent software abstraction inspired by the MFI profile table, not a driver-validated physical MIG emulator.",
        "- MFI's published evaluation uses an initially empty 100-GPU cluster, progressively increasing offered load, durations sampled in scheduling slots up to saturation time T, four fixed profile distributions, and 500 independent simulations per distribution. This M7 horizon sweep uses a different fixed-job-count Poisson-arrival protocol and must not be presented as a replication of those published numbers.",
        "- The paper's BF-BI/WF-BI benchmark descriptions do not fully specify executable code; our code documents its deterministic index-cost interpretation and should not be portrayed as the original authors' implementation.",
        "",
        "## Decision rule",
        "",
        "Retain adaptive lookahead only if held-out experiments show a repeatable benefit at a defensible runtime budget. If gains remain negligible and the offline-oracle gap persists, revise the method or pivot the research question rather than adding another complexity layer.",
        "",
        f"## Plot status\n\n{plot_status}",
        "",
        "## Artifact index", "",
        "- `m7_horizon_runs.csv`: one row per policy and matched workload trace.",
        "- `m7_decision_diagnostics.csv`: per-job placement and rollout-value details.",
        "- `policy_summary.csv`, `paired_vs_mfi.csv`, `paired_summary_vs_mfi.csv`, `scenario_summary.csv`.",
        "- `decision_diagnostics_summary.csv`: rollout and placement-change analysis.",
    ]
    (output / "M7_HORIZON_AUDIT_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Wrote summaries and report to {output.resolve()}")
    print(plot_status)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_csv", type=Path)
    parser.add_argument("decisions_csv", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/m7_horizon_audit"))
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    run_rows = read_csv(args.runs_csv)
    decision_rows = read_csv(args.decisions_csv)
    build_summary(run_rows, decision_rows, args.out, args.bootstrap_repeats)


if __name__ == "__main__":
    main()
