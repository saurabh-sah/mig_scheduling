"""MFI-style fragmentation metric with explicit compute and memory constraints.

The source paper describes profiles as compute/memory resource pairs and gives
an example where a 1g.20gb placement is blocked by a neighboring 1g.10gb
because the former requires two memory slices. We therefore check both
resource dimensions when deciding whether a legal placement is blocked.

This is a transparent software interpretation of the paper's abstract model,
not a claim of exact driver-level MIG emulation.
"""

from collections.abc import Sequence
from functools import lru_cache
from typing import Optional

from .profiles import (
    COMPUTE_SLICE_COUNT, MEMORY_SLICE_COUNT, PROFILE_ORDER, PROFILES, MigProfile,
)
from .state import GPUState

# Precompute profile placement masks once; scoring is invoked for every candidate.
_PROFILE_MASKS = {
    name: tuple((start,
                 sum(1 << i for i in p.occupied_compute_indices(start)),
                 sum(1 << i for i in p.occupied_memory_indices(start)))
                for start in p.legal_starts)
    for name, p in PROFILES.items()
}


def _to_mask(occupied: Sequence[Optional[str]]) -> int:
    mask = 0
    for i, owner in enumerate(occupied):
        if owner is not None:
            mask |= 1 << i
    return mask


@lru_cache(maxsize=(1 << (COMPUTE_SLICE_COUNT + MEMORY_SLICE_COUNT)))
def _fragmentation_score_from_masks(compute_mask: int, memory_mask: int) -> int:
    """Cached pure score function over the complete finite occupancy-state space.

    There are only 2^(7+8) possible occupancy-mask pairs for this profile model.
    Fragmentation scores are repeatedly requested for unchanged GPUs at arrival
    snapshots and during dry-run candidates, so memoization avoids recomputing
    the same score while preserving exactly the same metric.
    """
    free_compute = COMPUTE_SLICE_COUNT - compute_mask.bit_count()
    free_memory = MEMORY_SLICE_COUNT - memory_mask.bit_count()
    score = 0

    for profile_name in PROFILE_ORDER:
        profile = PROFILES[profile_name]
        if profile.compute_slices > free_compute or profile.memory_slices > free_memory:
            continue
        for _start, compute_placement_mask, memory_placement_mask in _PROFILE_MASKS[profile_name]:
            if (compute_mask & compute_placement_mask) or (memory_mask & memory_placement_mask):
                score += profile.memory_slices
    return score


def fragmentation_score_from_occupancy(
    compute_occupied_by: Sequence[Optional[str]],
    memory_occupied_by: Sequence[Optional[str]],
) -> int:
    if len(compute_occupied_by) != COMPUTE_SLICE_COUNT:
        raise ValueError(f"Expected {COMPUTE_SLICE_COUNT} compute positions; got {len(compute_occupied_by)}")
    if len(memory_occupied_by) != MEMORY_SLICE_COUNT:
        raise ValueError(f"Expected {MEMORY_SLICE_COUNT} memory positions; got {len(memory_occupied_by)}")

    compute_mask = _to_mask(compute_occupied_by)
    memory_mask = _to_mask(memory_occupied_by)
    return _fragmentation_score_from_masks(compute_mask, memory_mask)


def fragmentation_score(gpu: GPUState) -> int:
    return fragmentation_score_from_occupancy(gpu.compute_occupied_by, gpu.memory_occupied_by)


def fragmentation_score_breakdown_from_occupancy(
    compute_occupied_by: Sequence[Optional[str]],
    memory_occupied_by: Sequence[Optional[str]],
) -> list[dict[str, object]]:
    """Explain the MFI-style score contribution for each MIG profile.

    The paper defines workload resource requirements as a compute/memory pair,
    but Algorithm 1 writes its sufficient-free-resource gate using a compact
    scalar notation. This function makes our explicit component-wise
    interpretation inspectable: a profile contributes only when both aggregate
    compute and memory requirements fit; each legal placement blocked by either
    compute or memory occupancy then contributes ``profile.memory_slices``.

    This is a diagnostic of our operationalization, not proof that it matches
    the authors' unpublished executable implementation.
    """
    if len(compute_occupied_by) != COMPUTE_SLICE_COUNT:
        raise ValueError(f"Expected {COMPUTE_SLICE_COUNT} compute positions; got {len(compute_occupied_by)}")
    if len(memory_occupied_by) != MEMORY_SLICE_COUNT:
        raise ValueError(f"Expected {MEMORY_SLICE_COUNT} memory positions; got {len(memory_occupied_by)}")

    compute_mask = _to_mask(compute_occupied_by)
    memory_mask = _to_mask(memory_occupied_by)
    free_compute = COMPUTE_SLICE_COUNT - compute_mask.bit_count()
    free_memory = MEMORY_SLICE_COUNT - memory_mask.bit_count()
    rows: list[dict[str, object]] = []

    for profile_name in PROFILE_ORDER:
        profile = PROFILES[profile_name]
        capacity_sufficient = (
            profile.compute_slices <= free_compute
            and profile.memory_slices <= free_memory
        )
        feasible_starts: list[int] = []
        blocked_starts: list[int] = []
        if capacity_sufficient:
            for start, compute_placement_mask, memory_placement_mask in _PROFILE_MASKS[profile_name]:
                blocked = bool(
                    compute_mask & compute_placement_mask
                    or memory_mask & memory_placement_mask
                )
                (blocked_starts if blocked else feasible_starts).append(start)
        contribution = len(blocked_starts) * profile.memory_slices if capacity_sufficient else 0
        rows.append({
            "profile": profile_name,
            "free_compute": free_compute,
            "free_memory": free_memory,
            "required_compute": profile.compute_slices,
            "required_memory": profile.memory_slices,
            "capacity_sufficient_componentwise": capacity_sufficient,
            "legal_starts": list(profile.legal_starts),
            "feasible_starts": feasible_starts,
            "blocked_starts_counted": blocked_starts,
            "penalty_per_blocked_start": profile.memory_slices if capacity_sufficient else 0,
            "score_contribution": contribution,
        })
    return rows


def delta_fragmentation(
    gpu: GPUState,
    profile: MigProfile,
    start: int,
    before_score: int | None = None,
) -> tuple[int, int, int]:
    """Return (score before, score after hypothetical placement, delta)."""
    if not gpu.is_placement_feasible(profile, start):
        raise ValueError(f"Infeasible placement {profile.name}@{start}.")
    before = fragmentation_score(gpu) if before_score is None else before_score
    compute, memory = gpu.hypothetical_occupancy(profile, start)
    after = fragmentation_score_from_occupancy(compute, memory)
    return before, after, after - before
