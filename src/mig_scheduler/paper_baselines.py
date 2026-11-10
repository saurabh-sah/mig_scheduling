"""Operationalizations of the MFI paper's FF/RR benchmark descriptions.

The paper states that FF selects the first GPU with enough available resources
and assigns a profile to the first available index. Its prose does not specify
all edge-case behavior as executable source code. These classes therefore make
the assumption explicit: select the first GPU with enough aggregate compute and
memory capacity; choose the earliest feasible legal start on that GPU; if no
legal start exists despite sufficient aggregate capacity, reject this request
rather than searching another GPU. This operationalization is designed to
expose the fragmentation failure described in the paper and is not asserted to
be byte-for-byte identical to the original implementation.
"""
from __future__ import annotations

from time import perf_counter_ns
from typing import Optional

from .profiles import MigProfile
from .schedulers import PlacementDecision
from .state import GPUState


def _has_aggregate_capacity(gpu: GPUState, profile: MigProfile) -> bool:
    return (
        gpu.free_compute_slices >= profile.compute_slices
        and gpu.free_memory_slices >= profile.memory_slices
    )


class PaperFirstFitScheduler:
    """MFI-paper-style FF operationalization; can reject on a fragmented first GPU."""

    name = "paper_ff"

    def choose(self, gpus: list[GPUState], profile: MigProfile) -> Optional[PlacementDecision]:
        started = perf_counter_ns()
        for gpu in sorted(gpus, key=lambda item: item.gpu_id):
            if not _has_aggregate_capacity(gpu, profile):
                continue
            starts = gpu.feasible_starts(profile)
            if not starts:
                # Selected GPU has enough aggregate resources but no legal profile
                # placement. Reject the request instead of searching later GPUs.
                return None
            return PlacementDecision(
                gpu_id=gpu.gpu_id,
                start=starts[0],
                policy=self.name,
                decision_time_ns=perf_counter_ns() - started,
            )
        return None


class PaperRoundRobinScheduler:
    """Round-robin over GPUs with the same explicit aggregate-capacity failure rule."""

    name = "paper_rr"

    def __init__(self) -> None:
        self._next_gpu_id = 0

    def choose(self, gpus: list[GPUState], profile: MigProfile) -> Optional[PlacementDecision]:
        started = perf_counter_ns()
        if not gpus:
            return None
        ordered = sorted(gpus, key=lambda item: item.gpu_id)
        offset = next((i for i, gpu in enumerate(ordered) if gpu.gpu_id >= self._next_gpu_id), 0)
        for step in range(len(ordered)):
            pos = (offset + step) % len(ordered)
            gpu = ordered[pos]
            if not _has_aggregate_capacity(gpu, profile):
                continue
            self._next_gpu_id = ordered[(pos + 1) % len(ordered)].gpu_id
            starts = gpu.feasible_starts(profile)
            if not starts:
                return None
            return PlacementDecision(
                gpu_id=gpu.gpu_id,
                start=starts[0],
                policy=self.name,
                decision_time_ns=perf_counter_ns() - started,
            )
        return None
