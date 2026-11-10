"""GPU state with separate compute and memory occupancy."""

from dataclasses import dataclass, field
from typing import Optional

from .profiles import COMPUTE_SLICE_COUNT, MEMORY_SLICE_COUNT, PROFILES, MigProfile


@dataclass(frozen=True, slots=True)
class Allocation:
    job_id: str
    profile_name: str
    start: int
    gpu_id: int

    @property
    def compute_indices(self) -> tuple[int, ...]:
        return PROFILES[self.profile_name].occupied_compute_indices(self.start)

    @property
    def memory_indices(self) -> tuple[int, ...]:
        return PROFILES[self.profile_name].occupied_memory_indices(self.start)


@dataclass
class GPUState:
    gpu_id: int
    compute_occupied_by: list[Optional[str]] = field(
        default_factory=lambda: [None] * COMPUTE_SLICE_COUNT
    )
    memory_occupied_by: list[Optional[str]] = field(
        default_factory=lambda: [None] * MEMORY_SLICE_COUNT
    )
    allocations: dict[str, Allocation] = field(default_factory=dict)

    @property
    def free_compute_slices(self) -> int:
        return sum(x is None for x in self.compute_occupied_by)

    @property
    def used_compute_slices(self) -> int:
        return COMPUTE_SLICE_COUNT - self.free_compute_slices

    @property
    def free_memory_slices(self) -> int:
        return sum(x is None for x in self.memory_occupied_by)

    @property
    def used_memory_slices(self) -> int:
        return MEMORY_SLICE_COUNT - self.free_memory_slices

    @property
    def is_active(self) -> bool:
        return bool(self.allocations)

    def is_placement_feasible(self, profile: MigProfile, start: int) -> bool:
        if start not in profile.legal_starts:
            return False
        try:
            comp = profile.occupied_compute_indices(start)
            mem = profile.occupied_memory_indices(start)
        except ValueError:
            return False
        return (
            all(self.compute_occupied_by[i] is None for i in comp)
            and all(self.memory_occupied_by[i] is None for i in mem)
        )

    def feasible_starts(self, profile: MigProfile) -> list[int]:
        return [s for s in profile.legal_starts if self.is_placement_feasible(profile, s)]

    def allocate(self, job_id: str, profile: MigProfile, start: int) -> Allocation:
        if job_id in self.allocations:
            raise ValueError(f"Job {job_id} is already allocated on GPU {self.gpu_id}.")
        if not self.is_placement_feasible(profile, start):
            raise ValueError(f"Illegal or overlapping placement: {profile.name}@{start} on GPU {self.gpu_id}.")
        allocation = Allocation(job_id, profile.name, start, self.gpu_id)
        for i in allocation.compute_indices:
            self.compute_occupied_by[i] = job_id
        for i in allocation.memory_indices:
            self.memory_occupied_by[i] = job_id
        self.allocations[job_id] = allocation
        self.assert_consistent()
        return allocation

    def release(self, job_id: str) -> Allocation:
        if job_id not in self.allocations:
            raise KeyError(f"Job {job_id} is not allocated on GPU {self.gpu_id}.")
        allocation = self.allocations.pop(job_id)
        for i in allocation.compute_indices:
            if self.compute_occupied_by[i] != job_id:
                raise RuntimeError(f"Compute state corruption releasing {job_id} at slice {i}.")
            self.compute_occupied_by[i] = None
        for i in allocation.memory_indices:
            if self.memory_occupied_by[i] != job_id:
                raise RuntimeError(f"Memory state corruption releasing {job_id} at slice {i}.")
            self.memory_occupied_by[i] = None
        self.assert_consistent()
        return allocation

    def hypothetical_occupancy(self, profile: MigProfile, start: int) -> tuple[list[Optional[str]], list[Optional[str]]]:
        if not self.is_placement_feasible(profile, start):
            raise ValueError(f"Infeasible hypothetical placement {profile.name}@{start}.")
        comp = self.compute_occupied_by.copy()
        mem = self.memory_occupied_by.copy()
        for i in profile.occupied_compute_indices(start):
            comp[i] = "<hypothetical>"
        for i in profile.occupied_memory_indices(start):
            mem[i] = "<hypothetical>"
        return comp, mem

    def assert_consistent(self) -> None:
        expected_comp: list[Optional[str]] = [None] * COMPUTE_SLICE_COUNT
        expected_mem: list[Optional[str]] = [None] * MEMORY_SLICE_COUNT
        for job_id, allocation in self.allocations.items():
            if job_id != allocation.job_id or allocation.gpu_id != self.gpu_id:
                raise RuntimeError("Allocation registry is inconsistent.")
            for i in allocation.compute_indices:
                if expected_comp[i] is not None:
                    raise RuntimeError("Overlapping compute allocation registry entries.")
                expected_comp[i] = job_id
            for i in allocation.memory_indices:
                if expected_mem[i] is not None:
                    raise RuntimeError("Overlapping memory allocation registry entries.")
                expected_mem[i] = job_id
        if expected_comp != self.compute_occupied_by:
            raise RuntimeError("Compute occupancy does not match allocation registry.")
        if expected_mem != self.memory_occupied_by:
            raise RuntimeError("Memory occupancy does not match allocation registry.")

    def visualize(self) -> str:
        comp = " ".join("." if x is None else x for x in self.compute_occupied_by)
        mem = " ".join("." if x is None else x for x in self.memory_occupied_by)
        return f"GPU {self.gpu_id} compute=[{comp}] memory=[{mem}]"
