from itertools import product

from mig_scheduler.profiles import (
    COMPUTE_SLICE_COUNT,
    INDEX_PREFERENCE_ORDER,
    MEMORY_SLICE_COUNT,
    PROFILE_ORDER,
    PROFILES,
    index_preference_rank,
)
from mig_scheduler.schedulers import (
    BestFitBestIndexScheduler,
    WorstFitBestIndexScheduler,
    _static_blocking_cost,
)
from mig_scheduler.state import GPUState
from mig_scheduler.workloads import generate_progressive_load_workload


def test_reference_preference_orders_cover_every_legal_start_once():
    for name, profile in PROFILES.items():
        order = INDEX_PREFERENCE_ORDER[name]
        assert set(order) == set(profile.legal_starts)
        assert len(order) == len(set(order))
        assert sorted(index_preference_rank(name, start) for start in order) == list(range(len(order)))


def test_best_index_reference_start_for_empty_gpu():
    expected = {
        "7g.80gb": 0,
        "4g.40gb": 0,
        "3g.40gb": 4,
        "2g.20gb": 4,
        "1g.20gb": 6,
        "1g.10gb": 6,
    }
    for name, start in expected.items():
        profile = PROFILES[name]
        for scheduler in (BestFitBestIndexScheduler(), WorstFitBestIndexScheduler()):
            decision = scheduler.choose([GPUState(0)], profile)
            assert decision is not None
            assert decision.start == start


def test_best_index_uses_next_preferred_feasible_start():
    # Occupy profile 1g.10gb's preferred index 6; index 4 is next in the table.
    gpu = GPUState(0)
    gpu.allocate("occupy-6", PROFILES["1g.10gb"], 6)
    decision = BestFitBestIndexScheduler().choose([gpu], PROFILES["1g.10gb"])
    assert decision is not None
    assert decision.start == 4


def test_legacy_blocking_cost_matches_reference_order_on_empty_gpu():
    for name, order in INDEX_PREFERENCE_ORDER.items():
        costs = [_static_blocking_cost(name, start) for start in order]
        assert costs == sorted(costs)


def test_arrival_gap_changes_only_times_not_profiles_or_durations():
    base = generate_progressive_load_workload(gpu_count=4, scenario="skew_small", seed=88, arrival_gap=1.0)
    tight = generate_progressive_load_workload(gpu_count=4, scenario="skew_small", seed=88, arrival_gap=0.1)
    assert [(j.profile_name, j.duration) for j in base] == [(j.profile_name, j.duration) for j in tight]
    assert [j.arrival_time for j in base] != [j.arrival_time for j in tight]
    assert base[0].arrival_time == tight[0].arrival_time == 0.0


def test_zero_arrival_gap_creates_burst():
    jobs = generate_progressive_load_workload(gpu_count=3, scenario="uniform", seed=1, arrival_gap=0.0)
    assert {job.arrival_time for job in jobs} == {0.0}


def test_arrival_gap_negative_is_rejected():
    try:
        generate_progressive_load_workload(gpu_count=1, arrival_gap=-0.1)
    except ValueError as exc:
        assert "arrival_gap" in str(exc)
    else:
        raise AssertionError("negative arrival gap must be rejected")


def test_index_preference_ranking_agrees_with_legacy_cost_for_sampled_states():
    # Deterministic bounded sample of occupancy states; the full exhaustive audit
    # lives in scripts/audit_m9_baselines.py and its saved JSON result.
    masks = [(0, 0), (1, 1), (0b0101010, 0b01010101),
             (0b1110000, 0b11110000), (0b0001111, 0b00001111)]
    for compute_mask, memory_mask in masks:
        gpu = GPUState(0)
        gpu.compute_occupied_by = ["x" if compute_mask & (1 << i) else None for i in range(COMPUTE_SLICE_COUNT)]
        gpu.memory_occupied_by = ["x" if memory_mask & (1 << i) else None for i in range(MEMORY_SLICE_COUNT)]
        for name in PROFILE_ORDER:
            feasible = gpu.feasible_starts(PROFILES[name])
            if len(feasible) < 2:
                continue
            ref = next(start for start in INDEX_PREFERENCE_ORDER[name] if start in feasible)
            legacy = min(feasible, key=lambda start: (_static_blocking_cost(name, start), start))
            assert ref == legacy


def test_compute_load_basis_reaches_compute_capacity_budget():
    jobs = generate_progressive_load_workload(
        gpu_count=4, scenario="skew_small", seed=9, arrival_gap=0.0, load_basis="compute"
    )
    requested = sum(job.profile.compute_slices for job in jobs)
    assert requested >= 7 * 4


def test_memory_load_basis_reaches_memory_capacity_budget():
    jobs = generate_progressive_load_workload(
        gpu_count=4, scenario="skew_big", seed=10, arrival_gap=0.0, load_basis="memory"
    )
    requested = sum(job.profile.memory_slices for job in jobs)
    assert requested >= 8 * 4


def test_max_normalized_load_basis_reaches_one_resource_capacity():
    jobs = generate_progressive_load_workload(
        gpu_count=4, scenario="bimodal", seed=11, arrival_gap=0.0, load_basis="max_normalized"
    )
    compute_fraction = sum(job.profile.compute_slices for job in jobs) / (7 * 4)
    memory_fraction = sum(job.profile.memory_slices for job in jobs) / (8 * 4)
    assert max(compute_fraction, memory_fraction) >= 1.0


def test_unknown_load_basis_is_rejected():
    try:
        generate_progressive_load_workload(gpu_count=1, load_basis="fake_basis")
    except ValueError as exc:
        assert "load_basis" in str(exc)
    else:
        raise AssertionError("unknown load basis must be rejected")
