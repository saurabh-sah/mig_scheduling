"""Audit machine-readable MFI source-to-code assumptions and profile definitions."""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mig_scheduler.profiles import (
    COMPUTE_SLICE_COUNT, MEMORY_SLICE_COUNT, MFI_WORKLOAD_DISTRIBUTIONS,
    PROFILES, INDEX_PREFERENCE_ORDER, validate_profile_model,
)

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "7g.80gb": (7, 8, 7, (0,)),
    "4g.40gb": (4, 4, 4, (0,)),
    "3g.40gb": (3, 4, 4, (0, 4)),
    "2g.20gb": (2, 2, 2, (0, 2, 4)),
    "1g.20gb": (1, 2, 2, (0, 2, 4, 6)),
    "1g.10gb": (1, 1, 1, (0, 1, 2, 3, 4, 5, 6)),
}

def main() -> int:
    out = ROOT / "results" / "m10_source_audit"
    out.mkdir(parents=True, exist_ok=True)
    validate_profile_model()
    rows = []
    failures = []
    for name, exp in EXPECTED.items():
        p = PROFILES[name]
        observed = (p.compute_slices, p.memory_slices, p.paper_slice_units, p.legal_starts)
        passed = observed == exp
        rows.append({
            "profile": name,
            "expected_compute": exp[0], "observed_compute": p.compute_slices,
            "expected_memory": exp[1], "observed_memory": p.memory_slices,
            "expected_paper_slice_units": exp[2], "observed_paper_slice_units": p.paper_slice_units,
            "expected_legal_starts": ";".join(map(str, exp[3])),
            "observed_legal_starts": ";".join(map(str, p.legal_starts)),
            "profile_table_match": passed,
            "index_preference_covers_legal_starts": set(INDEX_PREFERENCE_ORDER[name]) == set(p.legal_starts),
        })
        if not passed:
            failures.append(f"Profile table mismatch: {name}: expected {exp}, observed {observed}")
        if set(INDEX_PREFERENCE_ORDER[name]) != set(p.legal_starts):
            failures.append(f"Index preference does not cover legal starts for {name}")
    for scenario, distribution in MFI_WORKLOAD_DISTRIBUTIONS.items():
        if abs(sum(distribution.values()) - 1.0) > 1e-9:
            failures.append(f"Distribution {scenario} does not sum to 1")
    with (out / "m10_profile_table_audit.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    report = {
        "audit": "MFI source-to-code profile/distribution structural audit",
        "source": "https://arxiv.org/abs/2511.18906",
        "compute_slice_count": COMPUTE_SLICE_COUNT,
        "memory_slice_count": MEMORY_SLICE_COUNT,
        "profile_count": len(PROFILES),
        "distribution_sums": {name: sum(prob.values()) for name, prob in MFI_WORKLOAD_DISTRIBUTIONS.items()},
        "checks": len(rows) * 2 + len(MFI_WORKLOAD_DISTRIBUTIONS),
        "failures": failures,
        "passed": not failures,
        "scope_note": "This verifies source-transcribed constants and internal consistency only; it is not hardware/driver validation or an exact reproduction of original executable code."
    }
    (out / "m10_source_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if failures else 0

if __name__ == "__main__":
    raise SystemExit(main())
