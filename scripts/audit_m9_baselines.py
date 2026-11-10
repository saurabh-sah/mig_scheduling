"""Audit MFI baseline profile metadata and Best-Index ordering.

The audit is exhaustive over all 2^7 compute-occupancy masks and 2^8
memory-occupancy masks. It compares the explicit preference ordering published
for A100-80GB in Turkkan et al. (2024) with the legacy M8 static blocking-cost
heuristic. This is a conformance check for the chosen abstraction, not a test of
physical MIG behavior or the authors' unavailable code.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mig_scheduler.profiles import (  # noqa: E402
    COMPUTE_SLICE_COUNT, MEMORY_SLICE_COUNT, INDEX_PREFERENCE_ORDER,
    PROFILE_ORDER, PROFILES, validate_profile_model,
)
from mig_scheduler.schedulers import _static_blocking_cost  # noqa: E402
from mig_scheduler.state import GPUState  # noqa: E402


def run_exhaustive_audit() -> dict:
    validate_profile_model()
    states_checked = 0
    profile_state_cases = 0
    ranked_cases = 0
    mismatch_cases = []

    # Any binary occupancy arrangement is checked, including states not
    # reachable through the simulator's allocation sequence. That broadens the
    # test for ordering consistency, but does not assert physical reachability.
    for compute_mask in range(1 << COMPUTE_SLICE_COUNT):
        for memory_mask in range(1 << MEMORY_SLICE_COUNT):
            states_checked += 1
            gpu = GPUState(0)
            gpu.compute_occupied_by = [
                "occupied" if (compute_mask >> i) & 1 else None
                for i in range(COMPUTE_SLICE_COUNT)
            ]
            gpu.memory_occupied_by = [
                "occupied" if (memory_mask >> i) & 1 else None
                for i in range(MEMORY_SLICE_COUNT)
            ]
            for profile_name in PROFILE_ORDER:
                profile = PROFILES[profile_name]
                feasible = gpu.feasible_starts(profile)
                if len(feasible) < 2:
                    continue
                profile_state_cases += 1
                reference = next(
                    start for start in INDEX_PREFERENCE_ORDER[profile_name]
                    if start in feasible
                )
                legacy = min(
                    feasible,
                    key=lambda start: (_static_blocking_cost(profile_name, start), start),
                )
                ranked_cases += 1
                if reference != legacy:
                    mismatch_cases.append({
                        "compute_mask": compute_mask,
                        "memory_mask": memory_mask,
                        "profile": profile_name,
                        "feasible_starts": feasible,
                        "reference_choice": reference,
                        "legacy_choice": legacy,
                        "legacy_costs": {
                            str(start): _static_blocking_cost(profile_name, start)
                            for start in feasible
                        },
                    })
                    if len(mismatch_cases) >= 20:
                        break
            if len(mismatch_cases) >= 20:
                break
        if len(mismatch_cases) >= 20:
            break

    return {
        "audit": "m9_index_preference_equivalence",
        "gpu_model": "A100-80GB profile/placement abstraction",
        "compute_occupancy_masks": 1 << COMPUTE_SLICE_COUNT,
        "memory_occupancy_masks": 1 << MEMORY_SLICE_COUNT,
        "states_checked": states_checked,
        "profile_state_cases_with_multiple_feasible_starts": profile_state_cases,
        "ranking_cases_compared": ranked_cases,
        "mismatch_count_reported": len(mismatch_cases),
        "mismatches": mismatch_cases,
        "reference_order": {k: list(v) for k, v in INDEX_PREFERENCE_ORDER.items()},
        "conclusion": (
            "Legacy static blocking-cost ranking agrees with the explicit reference order "
            "for all checked cases." if not mismatch_cases else
            "Legacy static blocking-cost ranking differs from the explicit reference order."
        ),
        "limitation": (
            "The exhaustive masks include states that may not correspond to reachable/legal "
            "MIG configurations. This verifies ranking consistency under this software model; "
            "it does not validate hardware placement or reproduce the published experiment."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/m9_index_order_audit.json"))
    args = parser.parse_args()
    result = run_exhaustive_audit()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in (
        "states_checked", "profile_state_cases_with_multiple_feasible_starts",
        "ranking_cases_compared", "mismatch_count_reported", "conclusion"
    )}, indent=2))
    print(f"Saved {args.out.resolve()}")
    if result["mismatches"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
