from copy import deepcopy

import pytest

from mig_scheduler.context import ArrivalObservation, SchedulingContext
from mig_scheduler.mfi_score_variants import (
    MFIScoreVariantScheduler,
    fragmentation_score_variant,
    fragmentation_score_variant_from_occupancy,
)
from mig_scheduler.profiles import PROFILE_ORDER, PROFILES
from mig_scheduler.schedulers import AdaptiveBudgetLookaheadScheduler, FixedBudgetLookaheadScheduler
from mig_scheduler.state import GPUState


def _stable_context(size=12):
    observations = tuple(ArrivalObservation(float(i), "1g.10gb") for i in range(size))
    return SchedulingContext(
        current_time=float(size),
        past_arrivals=observations,
        completed_durations_by_profile={name: (2.0, 3.0, 4.0) for name in PROFILE_ORDER},
        running_jobs=(),
    )


@pytest.mark.parametrize("mode", ["all_legal_starts", "free_anchor_starts"])
def test_fixed_lookahead_uses_selected_score_for_candidate_deltas(mode):
    scheduler = FixedBudgetLookaheadScheduler(horizon=1, rollouts=1, seed=7, score_mode=mode)
    gpus = [GPUState(0), GPUState(1)]
    profile = PROFILES["1g.10gb"]
    candidates = scheduler._candidate_placements(gpus, profile)
    assert candidates
    candidate = candidates[0]
    gpu = gpus[candidate[1]]
    expected_before = fragmentation_score_variant(gpu, mode=mode)
    comp, mem = gpu.hypothetical_occupancy(profile, candidate[2])
    expected_after = fragmentation_score_variant_from_occupancy(comp, mem, mode=mode)
    assert candidate[0] == expected_after - expected_before
    assert scheduler._mfi.score_mode == mode


@pytest.mark.parametrize("mode", ["all_legal_starts", "free_anchor_starts"])
def test_adaptive_lookahead_uses_selected_score_and_does_not_mutate_state(mode):
    scheduler = AdaptiveBudgetLookaheadScheduler(
        horizon=1, min_rollouts=2, max_rollouts=2, batch_size=2,
        minimum_history=4, profile_window=12, minimum_reliability=0.0,
        ambiguity_margin=10_000, max_candidates=3, seed=17, score_mode=mode,
    )
    gpus = [GPUState(0), GPUState(1)]
    before = deepcopy(gpus)
    decision = scheduler.choose_with_context(gpus, PROFILES["1g.10gb"], _stable_context())
    assert decision is not None
    assert gpus == before
    assert scheduler._mfi.score_mode == mode
    assert decision.policy == "adaptive_lookahead"


def test_invalid_lookahead_score_mode_is_rejected():
    with pytest.raises(ValueError):
        FixedBudgetLookaheadScheduler(score_mode="paper_exact")
    with pytest.raises(ValueError):
        AdaptiveBudgetLookaheadScheduler(score_mode="paper_exact")


def test_free_anchor_mfi_variant_is_used_for_lookahead_without_context():
    gpus_a = [GPUState(0), GPUState(1)]
    gpus_b = [GPUState(0), GPUState(1)]
    profile = PROFILES["1g.10gb"]
    expected = MFIScoreVariantScheduler(score_mode="free_anchor_starts").choose(gpus_a, profile)
    actual = FixedBudgetLookaheadScheduler(
        horizon=1, rollouts=2, score_mode="free_anchor_starts"
    ).choose(gpus_b, profile)
    assert expected is not None and actual is not None
    assert (actual.gpu_id, actual.start) == (expected.gpu_id, expected.start)
