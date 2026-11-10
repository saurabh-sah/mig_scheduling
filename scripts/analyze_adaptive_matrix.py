"""Summarize matched adaptive-lookahead runs and paired acceptance differences vs MFI."""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def _float(row: dict[str, str], key: str) -> float | None:
    raw = row.get(key, "")
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/m4_adaptive_summary.csv"))
    args = parser.parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    rows = list(csv.DictReader(args.input.open(newline="", encoding="utf-8")))
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    paired: dict[tuple[str, int], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["scenario"], row["policy"])].append(row)
        paired[(row["scenario"], int(row["seed"]))][row["policy"]] = row

    metrics = (
        "acceptance_rate",
        "mean_decision_time_ns",
        "mean_time_weighted_fragmentation_score",
        "peak_fragmentation_score",
        "mean_lookahead_value",
        "mean_lookahead_rollouts_per_decision",
        "adaptive_mean_forecast_reliability",
        "adaptive_mean_total_rollouts",
        "adaptive_lookahead_decisions",
        "adaptive_mfi_fallback_decisions",
        "adaptive_confidence_stops",
        "adaptive_budget_stops",
    )
    out_rows: list[dict[str, object]] = []
    for (scenario, policy), policy_rows in sorted(grouped.items()):
        out: dict[str, object] = {"scenario": scenario, "policy": policy, "runs": len(policy_rows)}
        for metric in metrics:
            values = [value for row in policy_rows if (value := _float(row, metric)) is not None]
            out[f"{metric}_mean"] = statistics.mean(values) if values else ""
            out[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else (0.0 if values else "")
        paired_diffs = []
        for (pair_scenario, _seed), per_policy in paired.items():
            if pair_scenario != scenario or policy not in per_policy or "mfi" not in per_policy:
                continue
            here = _float(per_policy[policy], "acceptance_rate")
            baseline = _float(per_policy["mfi"], "acceptance_rate")
            if here is not None and baseline is not None:
                paired_diffs.append(here - baseline)
        out["acceptance_diff_vs_mfi_mean"] = statistics.mean(paired_diffs) if paired_diffs else ""
        out["acceptance_diff_vs_mfi_sd"] = statistics.stdev(paired_diffs) if len(paired_diffs) > 1 else (0.0 if paired_diffs else "")
        out_rows.append(out)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if out_rows:
        with args.out.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(out_rows[0]))
            writer.writeheader()
            writer.writerows(out_rows)
    print(f"Wrote {len(out_rows)} grouped rows to {args.out.resolve()}")


if __name__ == "__main__":
    main()
