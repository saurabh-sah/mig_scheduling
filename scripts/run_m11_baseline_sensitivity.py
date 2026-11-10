"""M11: sensitivity audit for the underspecified BF-BI/WF-BI ranking rule.

Runs matched workloads across resource-residual scalarizations and tie hierarchy
choices. This is an audit of our operationalizations, not a claim that any
variant reproduces an unavailable author implementation.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mig_scheduler.baseline_sensitivity import (  # noqa: E402
    BFWFRankingSensitivityScheduler, RANKING_MODES, TIE_POLICIES,
)
from mig_scheduler.profiles import MFI_WORKLOAD_DISTRIBUTIONS  # noqa: E402
from mig_scheduler.schedulers import MFIScheduler  # noqa: E402
from mig_scheduler.simulator import DiscreteEventSimulator  # noqa: E402
from mig_scheduler.workloads import generate_progressive_load_workload  # noqa: E402


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    return statistics.mean(values) if values else 0.0


def bootstrap_ci(values: list[float], seed: int, draws: int = 3000) -> tuple[float, float]:
    if len(values) <= 1:
        v = values[0] if values else 0.0
        return v, v
    rng = random.Random(seed)
    samples = sorted(mean([values[rng.randrange(len(values))] for _ in values]) for _ in range(draws))
    return samples[int(0.025 * draws)], samples[min(draws - 1, int(0.975 * draws))]


def policy_specs():
    specs = [("mfi", "mfi", None, None)]
    for strategy in ("bf_bi", "wf_bi"):
        for mode in RANKING_MODES:
            for tie in TIE_POLICIES:
                key = f"{strategy}__{mode}__{tie}"
                specs.append((key, strategy, mode, tie))
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--arrival-gap", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "m11_baseline_sensitivity")
    args = parser.parse_args()
    if args.gpus < 1 or args.repetitions < 1:
        parser.error("--gpus and --repetitions must be positive")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    specs = policy_specs()
    raw_rows: list[dict] = []
    per_job_rows: list[dict] = []
    decision_change_rows: list[dict] = []

    for scenario_idx, scenario in enumerate(MFI_WORKLOAD_DISTRIBUTIONS):
        for rep in range(args.repetitions):
            seed = args.seed + scenario_idx * 100_003 + rep
            jobs = generate_progressive_load_workload(
                gpu_count=args.gpus, scenario=scenario, seed=seed,
                arrival_gap=args.arrival_gap, load_basis="paper_slice_units",
            )
            baseline_job_maps: dict[str, dict[str, tuple]] = {}
            run_summaries: dict[str, dict] = {}
            for policy_idx, (policy_key, strategy, mode, tie) in enumerate(specs):
                if strategy == "mfi":
                    scheduler = MFIScheduler()
                else:
                    scheduler = BFWFRankingSensitivityScheduler(
                        strategy=strategy,
                        ranking_mode=mode or "normalized_sum",
                        tie_policy=tie or "index_before_gpu",
                    )
                result = DiscreteEventSimulator(args.gpus, scheduler).run(jobs)
                summary = result.summary()
                row = {
                    "scenario": scenario, "repetition": rep, "seed": seed,
                    "policy_key": policy_key, "strategy": strategy,
                    "ranking_mode": mode or "n/a", "tie_policy": tie or "n/a",
                    "gpu_count": args.gpus, "submitted": summary["submitted"],
                    "accepted": summary["accepted"], "rejected": summary["rejected"],
                    "acceptance_rate": summary["acceptance_rate"],
                    "mean_decision_time_ns": summary["mean_decision_time_ns"],
                    "mean_gpu_fragmentation_score": summary["mean_time_weighted_fragmentation_score"],
                    "final_fragmentation_score": summary["final_fragmentation_score"],
                    "mean_arrival_snapshot_compute_utilization": summary["mean_arrival_snapshot_compute_utilization"],
                    "mean_arrival_snapshot_memory_utilization": summary["mean_arrival_snapshot_memory_utilization"],
                    "mean_arrival_snapshot_active_gpus": summary["mean_arrival_snapshot_active_gpus"],
                    "job_count": len(jobs),
                }
                raw_rows.append(row)
                run_summaries[policy_key] = row
                map_rows = {}
                for jr in result.job_results:
                    placement = (jr.get("status"), jr.get("gpu_id"), jr.get("start"))
                    map_rows[jr["job_id"]] = placement
                    per_job_rows.append({
                        "scenario": scenario, "repetition": rep, "seed": seed,
                        "policy_key": policy_key, "job_id": jr["job_id"],
                        "profile": jr["profile"], "status": jr["status"],
                        "gpu_id": jr.get("gpu_id"), "start": jr.get("start"),
                        "decision_time_ns": jr.get("decision_time_ns"),
                    })
                baseline_job_maps[policy_key] = map_rows

            # Pair all ranking variants against current default and MFI on the exact trace.
            for strategy in ("bf_bi", "wf_bi"):
                default_key = f"{strategy}__normalized_sum__index_before_gpu"
                default_map = baseline_job_maps[default_key]
                for policy_key, spec_strategy, mode, tie in specs:
                    if spec_strategy != strategy or policy_key == default_key:
                        continue
                    other_map = baseline_job_maps[policy_key]
                    comparable = sorted(set(default_map) & set(other_map))
                    changed = sum(default_map[j] != other_map[j] for j in comparable)
                    decision_change_rows.append({
                        "scenario": scenario, "repetition": rep, "seed": seed,
                        "reference_policy": default_key, "compared_policy": policy_key,
                        "placement_or_admission_changes": changed,
                        "jobs_compared": len(comparable),
                        "placement_change_fraction": changed / len(comparable) if comparable else 0.0,
                    })

    write_csv(out / "m11_raw_runs.csv", raw_rows)
    write_csv(out / "m11_per_job_decisions.csv", per_job_rows)
    write_csv(out / "m11_decision_change_runs.csv", decision_change_rows)

    summary_rows = []
    grouped = defaultdict(list)
    for r in raw_rows:
        grouped[(r["scenario"], r["policy_key"])].append(r)
    for (scenario, key), rows in sorted(grouped.items()):
        summary_rows.append({
            "scenario": scenario, "policy_key": key, "repetitions": len(rows),
            "acceptance_mean": mean([float(r["acceptance_rate"]) for r in rows]),
            "acceptance_sd": statistics.stdev([float(r["acceptance_rate"]) for r in rows]) if len(rows) > 1 else 0.0,
            "mean_decision_time_ns": mean([float(r["mean_decision_time_ns"]) for r in rows]),
            "final_fragmentation_score_mean": mean([float(r["final_fragmentation_score"]) for r in rows]),
            "compute_utilization_mean": mean([float(r["mean_arrival_snapshot_compute_utilization"]) for r in rows]),
            "memory_utilization_mean": mean([float(r["mean_arrival_snapshot_memory_utilization"]) for r in rows]),
            "active_gpus_mean": mean([float(r["mean_arrival_snapshot_active_gpus"]) for r in rows]),
        })
    write_csv(out / "m11_policy_summary.csv", summary_rows)

    # Paired acceptance deltas vs MFI within the same scenario/replicate.
    by_run = {(r["scenario"], r["repetition"], r["policy_key"]): r for r in raw_rows}
    pair_groups = defaultdict(list)
    for r in raw_rows:
        if r["policy_key"] == "mfi":
            continue
        baseline = by_run[(r["scenario"], r["repetition"], "mfi")]
        pair_groups[(r["scenario"], r["policy_key"])].append(
            float(r["acceptance_rate"]) - float(baseline["acceptance_rate"])
        )
    paired_rows = []
    for idx, ((scenario, key), deltas) in enumerate(sorted(pair_groups.items())):
        lo, hi = bootstrap_ci(deltas, args.seed + idx)
        paired_rows.append({
            "scenario": scenario, "policy_key": key, "pairs": len(deltas),
            "acceptance_delta_vs_mfi_mean": mean(deltas),
            "paired_delta_sd": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
            "bootstrap_ci_low": lo, "bootstrap_ci_high": hi,
            "positive": sum(d > 1e-12 for d in deltas),
            "negative": sum(d < -1e-12 for d in deltas),
            "tied": sum(abs(d) <= 1e-12 for d in deltas),
        })
    write_csv(out / "m11_paired_delta_vs_mfi.csv", paired_rows)

    # Pair each BF/WF sensitivity variant against its own current operational default.
    strategy_default_rows = []
    for strategy in ("bf_bi", "wf_bi"):
        default_key = f"{strategy}__normalized_sum__index_before_gpu"
        variant_groups = defaultdict(list)
        for r in raw_rows:
            key = r["policy_key"]
            if not key.startswith(strategy + "__") or key == default_key:
                continue
            reference = by_run[(r["scenario"], r["repetition"], default_key)]
            variant_groups[(r["scenario"], key)].append(
                float(r["acceptance_rate"]) - float(reference["acceptance_rate"])
            )
        for idx, ((scenario, key), deltas) in enumerate(sorted(variant_groups.items())):
            lo, hi = bootstrap_ci(deltas, args.seed + 90_000 + idx)
            strategy_default_rows.append({
                "strategy": strategy, "scenario": scenario, "policy_key": key,
                "reference_policy": default_key, "pairs": len(deltas),
                "acceptance_delta_vs_strategy_default_mean": mean(deltas),
                "paired_delta_sd": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
                "bootstrap_ci_low": lo, "bootstrap_ci_high": hi,
                "positive": sum(d > 1e-12 for d in deltas),
                "negative": sum(d < -1e-12 for d in deltas),
                "tied": sum(abs(d) <= 1e-12 for d in deltas),
            })
    write_csv(out / "m11_paired_delta_vs_strategy_default.csv", strategy_default_rows)

    # Summarize ranking decision-change diagnostics.
    change_summary = []
    g = defaultdict(list)
    for r in decision_change_rows:
        g[(r["scenario"], r["compared_policy"])].append(float(r["placement_change_fraction"]))
    for (scenario, key), vals in sorted(g.items()):
        change_summary.append({"scenario": scenario, "policy_key": key,
                               "mean_placement_or_admission_change_fraction": mean(vals),
                               "repetitions": len(vals)})
    write_csv(out / "m11_decision_change_summary.csv", change_summary)

    manifest = {
        "milestone": 11,
        "study": "BF-BI/WF-BI resource ranking and tie hierarchy sensitivity",
        "gpus": args.gpus, "repetitions_per_profile_distribution": args.repetitions,
        "seed_base": args.seed, "arrival_gap": args.arrival_gap,
        "scenarios": list(MFI_WORKLOAD_DISTRIBUTIONS),
        "ranking_modes": list(RANKING_MODES), "tie_policies": list(TIE_POLICIES),
        "policy_runs": len(raw_rows),
        "limitation": "Sensitivity of our operationalizations; not a full replication of author code.",
        "source": "https://arxiv.org/abs/2511.18906",
    }
    (out / "m11_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Small markdown report with defaults and strongest scenario variations.
    default_names = ["mfi", "bf_bi__normalized_sum__index_before_gpu", "wf_bi__normalized_sum__index_before_gpu"]
    all_summary = { (r["scenario"], r["policy_key"]): r for r in summary_rows }
    lines = [
        "# M11 Baseline Ranking Sensitivity Report", "",
        "## Scope", "",
        "This exploratory study tests whether BF-BI/WF-BI outcomes depend on the resource-residual scalarization or tie hierarchy used in our implementation. The MFI paper defines BF-BI and WF-BI verbally (bin-packing vs load-balancing) and gives an established Best Index preference order, but does not specify a scalarization formula for the two separate compute/memory residuals or fully specify cross-GPU tie handling. This report therefore compares plausible operationalizations rather than claiming source-code equivalence.", "",
        f"- GPUs: {args.gpus}", f"- Repetitions per profile distribution: {args.repetitions}",
        f"- Policy runs: {len(raw_rows)}", f"- Workload scenarios: {', '.join(MFI_WORKLOAD_DISTRIBUTIONS)}", "",
        "## Default policy means by scenario", "",
        "| Scenario | MFI acceptance | BF-BI default | WF-BI default |", "|---|---:|---:|---:|",
    ]
    for scenario in MFI_WORKLOAD_DISTRIBUTIONS:
        vals=[]
        for k in default_names:
            hit=all_summary[(scenario,k)]
            vals.append(float(hit["acceptance_mean"]))
        lines.append(f"| {scenario} | {vals[0]:.3%} | {vals[1]:.3%} | {vals[2]:.3%} |")
    lines += ["", "## Acceptance delta versus each strategy's current default", "",
              "The default is `normalized_sum` plus `index_before_gpu`, matching the existing M10 operationalization. Values are paired variant-minus-default differences on identical traces. A zero delta means the placement changes observed elsewhere did not change acceptance on this workload set.", "",
              "| Strategy | Scenario | Variant | Mean acceptance delta | 95% bootstrap interval |", "|---|---|---|---:|---:|"]
    for r in sorted(strategy_default_rows, key=lambda x: (x["strategy"], x["scenario"], x["policy_key"])):
        lines.append(f"| {r['strategy']} | {r['scenario']} | {r['policy_key']} | {r['acceptance_delta_vs_strategy_default_mean']:+.3%} | [{r['bootstrap_ci_low']:+.3%}, {r['bootstrap_ci_high']:+.3%}] |")
    lines += ["", "## Interpretation", "",
              "1. The current compute/memory normalized-sum residual is an explicit operationalization, not a formula stated in the MFI paper.",
              "2. The profile-specific Best Index ordering is source-derived from Turkkan et al. (2024), and MFI states it uses that preference policy for BF-BI/WF-BI.",
              "3. No variant is selected merely because it improves acceptance on these traces. Doing so would tune the baseline to our result. Any eventual baseline choice must be justified from source semantics or author clarification, and sensitivity results should remain reported.",
              "4. These synthetic tests do not prove any algorithm's real-world superiority.", "",
              "## Files", "",
              "- `m11_raw_runs.csv`: per-run outcomes.",
              "- `m11_per_job_decisions.csv`: per-job placement decisions.",
              "- `m11_decision_change_runs.csv`: per-run decision changes against the current BF/WF default.",
              "- `m11_policy_summary.csv`: acceptance/runtime/fragmentation summaries.",
              "- `m11_paired_delta_vs_mfi.csv`: matched seed deltas and percentile bootstrap intervals.",
              "- `m11_decision_change_summary.csv`: fraction of job decisions that differ from the current BF/WF default.",
              "- `m11_manifest.json`: settings and audit scope.", "",
              "## Sources", "",
              "- Zambianco, Fasol, Doriguzzi-Corin, *An Online Fragmentation-Aware GPU Scheduler for Multi-Tenant MIG-based Clouds*, Sec. III–VI, Algorithms 1–2, Tables I–II: https://arxiv.org/abs/2511.18906",
              "- Turkkan et al., *Optimal Workload Placement on Multi-Instance GPUs*, Table 1 and Sec. 3.2.2: https://arxiv.org/abs/2409.06646",
              "- NVIDIA, *Supported MIG Profiles*: https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-mig-profiles.html", ""]
    (out / "M11_BASELINE_RANKING_SENSITIVITY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote M11 sensitivity study to: {out}")
    print(f"Runs: {len(raw_rows)}; raw rows: m11_raw_runs.csv")


if __name__ == "__main__":
    main()
