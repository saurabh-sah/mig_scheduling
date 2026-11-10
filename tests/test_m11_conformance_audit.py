from random import Random

from mig_scheduler.baseline_sensitivity import BFWFRankingSensitivityScheduler, RANKING_MODES, TIE_POLICIES
from mig_scheduler.fragmentation import (
    fragmentation_score,
    fragmentation_score_breakdown_from_occupancy,
)
from mig_scheduler.profiles import PROFILE_ORDER, PROFILES
from mig_scheduler.schedulers import BestFitBestIndexScheduler, WorstFitBestIndexScheduler
from mig_scheduler.state import GPUState


def test_fragmentation_breakdown_sums_to_existing_score_on_empty_gpu():
    gpu = GPUState(0)
    rows = fragmentation_score_breakdown_from_occupancy(
        gpu.compute_occupied_by, gpu.memory_occupied_by
    )
    assert sum(row["score_contribution"] for row in rows) == fragmentation_score(gpu)
    assert all(row["score_contribution"] == 0 for row in rows)


def test_fragmentation_breakdown_catches_memory_only_overlap_example():
    # A 1g.10gb at index 5 uses compute c5 and memory m5. The 1g.20gb at
    # index 4 has free compute c4, but needs memory m4+m5, so memory overlap
    # blocks that legal placement despite enough aggregate compute and memory.
    gpu = GPUState(0)
    gpu.allocate("small-at-five", PROFILES["1g.10gb"], 5)
    rows = fragmentation_score_breakdown_from_occupancy(
        gpu.compute_occupied_by, gpu.memory_occupied_by
    )
    row = next(r for r in rows if r["profile"] == "1g.20gb")
    assert row["capacity_sufficient_componentwise"] is True
    assert 4 in row["blocked_starts_counted"]
    assert 6 in row["feasible_starts"]
    assert row["score_contribution"] == 2
    assert sum(r["score_contribution"] for r in rows) == fragmentation_score(gpu)


def test_fragmentation_breakdown_is_consistent_after_random_legal_allocations():
    rng = Random(20261026)
    for case in range(150):
        gpu = GPUState(0)
        for job_idx in range(rng.randint(0, 5)):
            profile = PROFILES[rng.choice(PROFILE_ORDER)]
            starts = gpu.feasible_starts(profile)
            if not starts:
                continue
            gpu.allocate(f"{case}-{job_idx}", profile, rng.choice(starts))
        rows = fragmentation_score_breakdown_from_occupancy(
            gpu.compute_occupied_by, gpu.memory_occupied_by
        )
        assert sum(r["score_contribution"] for r in rows) == fragmentation_score(gpu)


def test_current_default_bf_wf_are_reproducible_by_audit_variants():
    rng = Random(881122)
    for case in range(300):
        gpus = [GPUState(i) for i in range(3)]
        for gpu in gpus:
            for idx in range(rng.randint(0, 4)):
                profile = PROFILES[rng.choice(PROFILE_ORDER)]
                starts = gpu.feasible_starts(profile)
                if starts:
                    gpu.allocate(f"{case}-{gpu.gpu_id}-{idx}", profile, rng.choice(starts))
        profile = PROFILES[rng.choice(PROFILE_ORDER)]
        for original, audit, strat in (
            (BestFitBestIndexScheduler(), BFWFRankingSensitivityScheduler(strategy="bf_bi"), "bf_bi"),
            (WorstFitBestIndexScheduler(), BFWFRankingSensitivityScheduler(strategy="wf_bi"), "wf_bi"),
        ):
            a = original.choose(gpus, profile)
            b = audit.choose(gpus, profile)
            assert (None if a is None else (a.gpu_id, a.start)) == (
                None if b is None else (b.gpu_id, b.start)
            ), f"Mismatch in case={case} strategy={strat}"


def test_all_audit_modes_are_constructible_and_return_legal_placements():
    for strategy in ("bf_bi", "wf_bi"):
        for mode in RANKING_MODES:
            for tie in TIE_POLICIES:
                scheduler = BFWFRankingSensitivityScheduler(
                    strategy=strategy, ranking_mode=mode, tie_policy=tie
                )
                gpus = [GPUState(0), GPUState(1)]
                profile = PROFILES["1g.10gb"]
                decision = scheduler.choose(gpus, profile)
                assert decision is not None
                assert decision.gpu_id in {0, 1}
                assert decision.start in profile.legal_starts


def test_resource_scalarization_can_change_gpu_choice_on_same_legal_state():
    # Construct a reachable state with different compute/memory residual shape.
    # GPU 0 leaves 3 compute and 4 memory slices free; GPU 1 leaves 3 compute
    # and 3 memory slices free. For 1g.20gb the post-placement residuals create
    # a tie under max-normalized slack but not under normalized-sum slack.
    g0, g1 = GPUState(0), GPUState(1)
    g0.allocate("g0-small-3", PROFILES["1g.10gb"], 3)
    g0.allocate("g0-2g-0", PROFILES["2g.20gb"], 0)
    g0.allocate("g0-small-5", PROFILES["1g.10gb"], 5)
    g1.allocate("g1-3g-4", PROFILES["3g.40gb"], 4)
    g1.allocate("g1-small-0", PROFILES["1g.10gb"], 0)
    request = PROFILES["1g.20gb"]

    sum_choice = BFWFRankingSensitivityScheduler(
        strategy="bf_bi", ranking_mode="normalized_sum", tie_policy="gpu_before_index"
    ).choose([g0, g1], request)
    max_choice = BFWFRankingSensitivityScheduler(
        strategy="bf_bi", ranking_mode="normalized_max", tie_policy="gpu_before_index"
    ).choose([g0, g1], request)
    assert sum_choice is not None and max_choice is not None
    assert sum_choice.gpu_id == 1
    assert max_choice.gpu_id == 0
