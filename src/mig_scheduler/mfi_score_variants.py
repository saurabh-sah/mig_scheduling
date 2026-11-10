"""Explicit MFI fragmentation-score interpretations for conformance auditing.

The current project's original ``fragmentation_score`` counts every blocked
legal profile start whenever aggregate component-wise capacity is sufficient.
A second interpretation counts only candidate starts whose anchor index is
currently unoccupied, and then checks the complete compute/memory footprint.
That second interpretation reproduces the numeric F(GPU1)=8 / F(GPU2)=16
example in Figure 3 of Zambianco et al. when the states are reconstructed from
the diagram. The paper's pseudocode does not spell out this anchor filtering,
so neither mode is silently called the authors' exact implementation.
"""
from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from time import perf_counter_ns
from typing import Optional

from .fragmentation import fragmentation_score_from_occupancy
from .profiles import (
    COMPUTE_SLICE_COUNT, MEMORY_SLICE_COUNT, PROFILE_ORDER, PROFILES, MigProfile,
    index_preference_rank,
)
from .schedulers import PlacementDecision
from .state import GPUState

SCORE_MODES = ("all_legal_starts", "free_anchor_starts")


def _mask(occupied: Sequence[Optional[str]]) -> int:
    value = 0
    for i, owner in enumerate(occupied):
        if owner is not None:
            value |= 1 << i
    return value


@lru_cache(maxsize=(1 << (COMPUTE_SLICE_COUNT + MEMORY_SLICE_COUNT)))
def _free_anchor_score_from_masks(compute_mask: int, memory_mask: int) -> int:
    """Cached score for the Figure-3-conformance interpretation."""
    free_compute = COMPUTE_SLICE_COUNT - compute_mask.bit_count()
    free_memory = MEMORY_SLICE_COUNT - memory_mask.bit_count()
    total = 0
    for profile_name in PROFILE_ORDER:
        profile = PROFILES[profile_name]
        if (profile.compute_slices > free_compute
                or profile.memory_slices > free_memory):
            continue
        for start in profile.legal_starts:
            if ((compute_mask >> start) & 1) or ((memory_mask >> start) & 1):
                continue
            compute_placement = sum(1 << i for i in profile.occupied_compute_indices(start))
            memory_placement = sum(1 << i for i in profile.occupied_memory_indices(start))
            if (compute_mask & compute_placement) or (memory_mask & memory_placement):
                total += profile.memory_slices
    return total


def fragmentation_score_variant_from_occupancy(
    compute_occupied_by: Sequence[Optional[str]],
    memory_occupied_by: Sequence[Optional[str]],
    *,
    mode: str = "all_legal_starts",
) -> int:
    """Compute a named score variant without mutating a GPU state.

    ``all_legal_starts`` is the existing Milestone 11 operationalization.
    ``free_anchor_starts`` omits start indices already occupied in either
    resource dimension, then checks whether the full candidate footprint is
    blocked. This makes the score match the paper's reported Figure 3 examples
    for the reconstructed states documented in M12_FIGURE3_CONFORMANCE.md.

    Both modes use the component-wise aggregate capacity gate and add the
    profile's number of memory slices for each counted blocked placement.
    """
    if mode not in SCORE_MODES:
        raise ValueError(f"mode must be one of {SCORE_MODES}")
    if len(compute_occupied_by) != COMPUTE_SLICE_COUNT:
        raise ValueError(f"Expected {COMPUTE_SLICE_COUNT} compute positions")
    if len(memory_occupied_by) != MEMORY_SLICE_COUNT:
        raise ValueError(f"Expected {MEMORY_SLICE_COUNT} memory positions")

    if mode == "all_legal_starts":
        return fragmentation_score_from_occupancy(
            compute_occupied_by, memory_occupied_by
        )

    compute_mask = _mask(compute_occupied_by)
    memory_mask = _mask(memory_occupied_by)
    # Only two compact masks define the state, so caching is exact and safe.
    # Score is a pure function of occupancy under this named interpretation.
    return _free_anchor_score_from_masks(compute_mask, memory_mask)


def fragmentation_score_variant(gpu: GPUState, *, mode: str) -> int:
    return fragmentation_score_variant_from_occupancy(
        gpu.compute_occupied_by, gpu.memory_occupied_by, mode=mode
    )


class MFIScoreVariantScheduler:
    """MFI greedy scheduler parameterized by an explicit score interpretation."""

    def __init__(self, *, score_mode: str = "all_legal_starts") -> None:
        if score_mode not in SCORE_MODES:
            raise ValueError(f"score_mode must be one of {SCORE_MODES}")
        self.score_mode = score_mode
        self.name = "mfi" if score_mode == "all_legal_starts" else "mfi_free_anchor_score"

    def choose(self, gpus: list[GPUState], profile: MigProfile) -> Optional[PlacementDecision]:
        started = perf_counter_ns()
        best: tuple[int, int, int, int, int] | None = None
        for gpu in sorted(gpus, key=lambda item: item.gpu_id):
            if (profile.compute_slices > gpu.free_compute_slices
                    or profile.memory_slices > gpu.free_memory_slices):
                continue
            before = fragmentation_score_variant(gpu, mode=self.score_mode)
            for start in gpu.feasible_starts(profile):
                compute, memory = gpu.hypothetical_occupancy(profile, start)
                after = fragmentation_score_variant_from_occupancy(
                    compute, memory, mode=self.score_mode
                )
                delta = after - before
                candidate = (delta, gpu.gpu_id, start, before, after)
                if best is None or candidate[:3] < best[:3]:
                    best = candidate
        elapsed = perf_counter_ns() - started
        if best is None:
            return None
        delta, gpu_id, start, before, after = best
        return PlacementDecision(
            gpu_id=gpu_id, start=start, policy=self.name,
            fragmentation_before=before, fragmentation_after=after,
            delta_fragmentation=delta, decision_time_ns=elapsed,
        )


class PaperBFWFBestIndexScheduler:
    """Paper-path BF-BI/WF-BI operationalization, distinct from global-feasible search.

    First select one GPU using aggregate post-allocation compute/memory residuals.
    Only then inspect legal placements on that selected GPU in the published Best
    Index preference order. If no legal placement exists on the selected GPU,
    reject the request rather than searching another GPU. This reproduces the
    failure path illustrated in MFI Figure 3. The residual scalarization remains
    an explicit implementation choice because the paper does not provide a
    formula for combining separate compute and memory residuals.
    """

    def __init__(self, *, strategy: str) -> None:
        if strategy not in {"bf_bi", "wf_bi"}:
            raise ValueError("strategy must be 'bf_bi' or 'wf_bi'")
        self.strategy = strategy
        self.name = f"paper_{strategy}"

    @staticmethod
    def residual_score(gpu: GPUState, profile: MigProfile) -> float:
        return (
            (gpu.free_compute_slices - profile.compute_slices) / COMPUTE_SLICE_COUNT
            + (gpu.free_memory_slices - profile.memory_slices) / MEMORY_SLICE_COUNT
        )

    def choose(self, gpus: list[GPUState], profile: MigProfile) -> Optional[PlacementDecision]:
        started = perf_counter_ns()
        eligible = [
            gpu for gpu in gpus
            if (gpu.free_compute_slices >= profile.compute_slices
                and gpu.free_memory_slices >= profile.memory_slices)
        ]
        if not eligible:
            return None

        # Deterministic GPU tie-break: lowest GPU id after residual score.
        if self.strategy == "bf_bi":
            selected = min(eligible, key=lambda g: (self.residual_score(g, profile), g.gpu_id))
        else:
            selected = min(eligible, key=lambda g: (-self.residual_score(g, profile), g.gpu_id))

        feasible = selected.feasible_starts(profile)
        if not feasible:
            # Important: do not search a second GPU. Figure 3 shows that the
            # selected GPU can have aggregate capacity but no legal placement.
            return None
        start = min(feasible, key=lambda s: (index_preference_rank(profile.name, s), s))
        return PlacementDecision(
            gpu_id=selected.gpu_id, start=start, policy=self.name,
            decision_time_ns=perf_counter_ns() - started,
        )
