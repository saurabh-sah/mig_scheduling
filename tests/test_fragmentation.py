from mig_scheduler.fragmentation import fragmentation_score, fragmentation_score_from_occupancy, delta_fragmentation
from mig_scheduler.profiles import PROFILES
from mig_scheduler.state import GPUState


def test_empty_gpu_has_zero_fragmentation_score():
    gpu = GPUState(0)
    assert fragmentation_score(gpu) == 0


def test_fragmentation_score_nonnegative_for_all_single_1g_positions():
    profile = PROFILES["1g.10gb"]
    for start in profile.legal_starts:
        gpu = GPUState(0)
        gpu.allocate(f"job-{start}", profile, start)
        assert fragmentation_score(gpu) >= 0


def test_profile_width_and_memory_weight_are_separate():
    # 3g.40gb uses three compute positions in the occupancy model but has four
    # memory slices; memory_slices is the MFI penalty weight.
    profile = PROFILES["3g.40gb"]
    assert profile.compute_slices == 3
    assert profile.memory_slices == 4


def test_delta_fragmentation_is_consistent_with_dry_run():
    gpu = GPUState(0)
    gpu.allocate("existing", PROFILES["1g.10gb"], 1)
    profile = PROFILES["1g.20gb"]
    start = 4
    before, after, delta = delta_fragmentation(gpu, profile, start)
    hypothetical_compute, hypothetical_memory = gpu.hypothetical_occupancy(profile, start)
    assert before == fragmentation_score(gpu)
    assert after == fragmentation_score_from_occupancy(hypothetical_compute, hypothetical_memory)
    assert delta == after - before


def test_memory_overlap_blocks_1g20_even_when_compute_position_is_free():
    # MFI paper discusses 1g.20gb being blocked by a neighboring 1g.10gb
    # because the former needs two memory slices.
    gpu = GPUState(0)
    gpu.allocate("small", PROFILES["1g.10gb"], 5)
    assert 4 not in gpu.feasible_starts(PROFILES["1g.20gb"])
    assert gpu.compute_occupied_by[4] is None
    assert gpu.memory_occupied_by[5] == "small"


def test_1g20_at_start_6_occupies_two_memory_slices():
    gpu = GPUState(0)
    gpu.allocate("two_mem", PROFILES["1g.20gb"], 6)
    assert gpu.compute_occupied_by[6] == "two_mem"
    assert gpu.memory_occupied_by[6:8] == ["two_mem", "two_mem"]


def test_memory_side_fragmentation_is_counted_even_without_compute_overlap():
    gpu = GPUState(0)
    gpu.allocate("small-at-five", PROFILES["1g.10gb"], 5)
    # 1g.20gb@4 uses compute position 4 (free) but memory positions 4 and 5;
    # the memory collision must block the placement and add 2 to the score.
    score = fragmentation_score(gpu)
    assert score >= 2
    assert not gpu.is_placement_feasible(PROFILES["1g.20gb"], 4)


def test_mfi_prefers_less_fragmenting_index_for_small_profile_on_empty_gpu():
    from mig_scheduler.schedulers import MFIScheduler
    gpu = GPUState(0)
    decision = MFIScheduler().choose([gpu], PROFILES["1g.10gb"])
    assert decision is not None
    assert decision.start == 6
