"""Sensitivity variants for the underspecified BF-BI/WF-BI GPU ranking rule.

MFI defines BF-BI as minimizing and WF-BI as maximizing residual resources,
then says it uses a profile-specific Best Index order from Turkkan et al. It
does not specify how separate compute/memory residuals are scalarized or how
GPU-level ties interact with index preference. These policies make those choices
explicit for audit experiments; none is claimed to be the authors' exact code.
"""
from __future__ import annotations

from time import perf_counter_ns
from typing import Optional

from .profiles import INDEX_PREFERENCE_ORDER, MigProfile, index_preference_rank
from .schedulers import PlacementDecision
from .state import GPUState

RANKING_MODES = ("normalized_sum", "normalized_max", "compute_first", "memory_first")
TIE_POLICIES = ("index_before_gpu", "gpu_before_index")


def residual_key(gpu: GPUState, profile: MigProfile, mode: str) -> tuple[float, ...]:
    """Return a dimensionless post-allocation residual-resource key.

    All values use the same A100-80GB capacity denominators as the model: seven
    compute slices and eight memory slices. ``compute_first`` and ``memory_first``
    are lexicographic alternatives; they are sensitivity checks, not claims about
    the MFI paper's intent.
    """
    if mode not in RANKING_MODES:
        raise ValueError(f"ranking mode must be one of {RANKING_MODES}")
    c = (gpu.free_compute_slices - profile.compute_slices) / 7.0
    m = (gpu.free_memory_slices - profile.memory_slices) / 8.0
    if mode == "normalized_sum":
        return (c + m,)
    if mode == "normalized_max":
        return (max(c, m),)
    if mode == "compute_first":
        return (c, m)
    return (m, c)


class BFWFRankingSensitivityScheduler:
    """BF-BI/WF-BI with explicit resource scalarization and deterministic ties."""

    def __init__(self, *, strategy: str, ranking_mode: str = "normalized_sum",
                 tie_policy: str = "index_before_gpu") -> None:
        if strategy not in {"bf_bi", "wf_bi"}:
            raise ValueError("strategy must be 'bf_bi' or 'wf_bi'")
        if ranking_mode not in RANKING_MODES:
            raise ValueError(f"ranking_mode must be one of {RANKING_MODES}")
        if tie_policy not in TIE_POLICIES:
            raise ValueError(f"tie_policy must be one of {TIE_POLICIES}")
        self.strategy = strategy
        self.ranking_mode = ranking_mode
        self.tie_policy = tie_policy
        if ranking_mode == "normalized_sum" and tie_policy == "index_before_gpu":
            # Name matches the existing Milestone 10 operationalization exactly.
            self.name = strategy
        else:
            self.name = f"{strategy}__{ranking_mode}__{tie_policy}"

    def choose(self, gpus: list[GPUState], profile: MigProfile) -> Optional[PlacementDecision]:
        started = perf_counter_ns()
        candidates: list[tuple[tuple[float, ...], int, int, int]] = []
        for gpu in gpus:
            if (profile.compute_slices > gpu.free_compute_slices
                    or profile.memory_slices > gpu.free_memory_slices):
                continue
            residual = residual_key(gpu, profile, self.ranking_mode)
            for start in gpu.feasible_starts(profile):
                rank = index_preference_rank(profile.name, start)
                candidates.append((residual, rank, gpu.gpu_id, start))
        if not candidates:
            return None

        def key(candidate: tuple[tuple[float, ...], int, int, int]):
            residual, rank, gpu_id, start = candidate
            oriented = residual if self.strategy == "bf_bi" else tuple(-x for x in residual)
            if self.tie_policy == "index_before_gpu":
                ties = (rank, gpu_id, start)
            else:
                ties = (gpu_id, rank, start)
            return (*oriented, *ties)

        _residual, _rank, gpu_id, start = min(candidates, key=key)
        return PlacementDecision(
            gpu_id=gpu_id, start=start, policy=self.name,
            decision_time_ns=perf_counter_ns() - started,
        )
