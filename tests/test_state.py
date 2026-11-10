import pytest

from mig_scheduler.profiles import PROFILES
from mig_scheduler.state import GPUState


def test_empty_gpu_feasible_start_lists_match_profile_table():
    gpu = GPUState(0)
    assert gpu.feasible_starts(PROFILES["1g.10gb"]) == [0, 1, 2, 3, 4, 5, 6]
    assert gpu.feasible_starts(PROFILES["1g.20gb"]) == [0, 2, 4, 6]
    assert gpu.feasible_starts(PROFILES["2g.20gb"]) == [0, 2, 4]
    assert gpu.feasible_starts(PROFILES["3g.40gb"]) == [0, 4]
    assert gpu.feasible_starts(PROFILES["4g.40gb"]) == [0]
    assert gpu.feasible_starts(PROFILES["7g.80gb"]) == [0]


def test_overlapping_and_illegal_placements_are_rejected():
    gpu = GPUState(0)
    gpu.allocate("j1", PROFILES["2g.20gb"], 0)
    assert not gpu.is_placement_feasible(PROFILES["1g.10gb"], 1)
    assert not gpu.is_placement_feasible(PROFILES["1g.20gb"], 1)
    with pytest.raises(ValueError):
        gpu.allocate("j2", PROFILES["1g.10gb"], 1)


def test_release_restores_exact_occupancy():
    gpu = GPUState(0)
    original_compute = gpu.compute_occupied_by.copy()
    original_memory = gpu.memory_occupied_by.copy()
    gpu.allocate("j1", PROFILES["3g.40gb"], 4)
    assert gpu.used_compute_slices == 3
    assert gpu.used_memory_slices == 4
    gpu.release("j1")
    assert gpu.compute_occupied_by == original_compute
    assert gpu.memory_occupied_by == original_memory
    assert gpu.used_compute_slices == 0
    assert gpu.used_memory_slices == 0


def test_duplicate_allocation_cannot_overlap_registry():
    gpu = GPUState(0)
    gpu.allocate("j1", PROFILES["1g.10gb"], 0)
    with pytest.raises(ValueError):
        gpu.allocate("j1", PROFILES["1g.10gb"], 2)
