"""Aggregate the matched fixed-lookahead experiment matrix without dependencies."""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/m3_lookahead_summary.csv"))
    args = parser.parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    with args.input.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            grouped[(row["scenario"], row["policy"])].append(row)

    metrics = [
        "acceptance_rate",
        "mean_decision_time_ns",
        "mean_time_weighted_fragmentation_score",
        "peak_fragmentation_score",
        "mean_lookahead_value",
        "mean_lookahead_rollouts_per_decision",
    ]
    output_rows = []
    for (scenario, policy), rows in sorted(grouped.items()):
        record: dict[str, object] = {
            "scenario": scenario,
            "policy": policy,
            "runs": len(rows),
        }
        for metric in metrics:
            values = []
            for row in rows:
                raw = row.get(metric, "")
                if raw not in (None, ""):
                    try:
                        values.append(float(raw))
                    except ValueError:
                        pass
            if values:
                record[f"{metric}_mean"] = statistics.mean(values)
                record[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
            else:
                record[f"{metric}_mean"] = ""
                record[f"{metric}_sd"] = ""
        output_rows.append(record)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0]) if output_rows else [])
        if output_rows:
            writer.writeheader()
            writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} grouped rows to {args.out.resolve()}")


if __name__ == "__main__":
    main()
