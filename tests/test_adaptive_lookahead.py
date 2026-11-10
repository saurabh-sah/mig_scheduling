from copy import deepcopy

import pytest

from mig_scheduler.context import ArrivalObservation, RunningJobObservation, SchedulingContext
from mig_scheduler.profiles import PROFILE_ORDER, PROFILES
from mig_scheduler.schedulers import AdaptiveBudgetLookaheadScheduler, MFIScheduler
from mig_scheduler.state import GPUState


def _context(sequence, *, duration=3.0):
    observations = tuple(
        ArrivalObservation(float(i), profile_name)
        for i, profile_name in enumerate(sequence)
    )
    duration_history = {name: (duration, duration + 1.0) for name in PROFILE_ORDER}
    return SchedulingContext(
        current_time=float(len(sequence)),
        past_arrivals=observations,
        completed_durations_by_profile=duration_history,
        running_jobs=(),
    )


def _stable_context(size=40):
    return _context(["1g.10gb"] * size)


def test_adaptive_scheduler_validates_budget_and_reliability_parameters():
    with pytest.raises(ValueError):
        AdaptiveBudgetLookaheadScheduler(min_rollouts=1)
    with pytest.raises(ValueError):
        AdaptiveBudgetLookaheadScheduler(min_rollouts=8, max_rollouts=4)
    with pytest.raises(ValueError):
        AdaptiveBudgetLookaheadScheduler(batch_size=0)
    with pytest.raises(ValueError):
        AdaptiveBudgetLookaheadScheduler(minimum_reliability=1.1)


def test_adaptive_policy_falls_back_to_mfi_without_context():
    gpus = [GPUState(0), GPUState(1)]
    profile = PROFILES["1g.10gb"]
    expected = MFIScheduler().choose(gpus, profile)
    actual = AdaptiveBudgetLookaheadScheduler().choose(gpus, profile)
    assert expected is not None and actual is not None
    assert (actual.gpu_id, actual.start) == (expected.gpu_id, expected.start)
    assert actual.adaptive_status == "mfi_fallback"
    assert actual.fallback_reason == "missing_online_context"
    assert actual.total_rollouts == 0


def test_adaptive_policy_falls_back_when_history_is_insufficient():
    scheduler = AdaptiveBudgetLookaheadScheduler(minimum_history=20, profile_window=30)
    context = _context(["1g.10gb", "1g.10gb", "2g.20gb"])
    gpus = [GPUState(0)]
    decision = scheduler.choose_with_context(gpus, PROFILES["1g.10gb"], context)
    assert decision is not None
    assert decision.adaptive_status == "mfi_fallback"
    assert decision.fallback_reason == "insufficient_history"
    assert decision.total_rollouts == 0


def test_reliability_drops_when_profile_distribution_shifts():
    sequence = ["1g.10gb"] * 20 + ["7g.80gb"] * 20
    scheduler = AdaptiveBudgetLookaheadScheduler(
        minimum_history=20,
        profile_window=40,
        stability_tv_threshold=0.6,
    )
    reliability = scheduler.forecast_reliability(_context(sequence))
    assert reliability == pytest.approx(0.0)


def test_stable_history_has_high_reliability():
    scheduler = AdaptiveBudgetLookaheadScheduler(
        minimum_history=20,
        profile_window=40,
        stability_tv_threshold=0.6,
    )
    assert scheduler.forecast_reliability(_stable_context()) == pytest.approx(1.0)


def test_adaptive_scheduler_runs_bounded_lookahead_when_gates_are_open():
    scheduler = AdaptiveBudgetLookaheadScheduler(
        horizon=2,
        min_rollouts=2,
        max_rollouts=4,
        batch_size=2,
        seed=73,
        minimum_history=4,
        profile_window=20,
        minimum_reliability=0.0,
        ambiguity_margin=10_000,  # deliberately open the ambiguity gate in this unit test
        max_candidates=3,
    )
    context = _stable_context()
    gpus = [GPUState(0), GPUState(1)]
    before = deepcopy(gpus)

    decision = scheduler.choose_with_context(gpus, PROFILES["1g.10gb"], context)

    assert decision is not None
    assert decision.adaptive_status == "lookahead"
    assert decision.fallback_reason is None
    assert decision.forecast_reliability == pytest.approx(1.0)
    assert decision.candidate_count <= 3
    assert decision.candidate_pool_size >= decision.candidate_count
    assert 2 <= decision.rollouts_per_candidate <= 4
    assert decision.rollouts_per_candidate % 2 == 0
    assert decision.total_rollouts == decision.candidate_count * decision.rollouts_per_candidate
    assert decision.adaptive_stop_reason in {
        "paired_confidence_separated", "max_rollout_budget"
    }
    assert gpus == before, "Hypothetical rollouts must not mutate live state"


def test_paired_confidence_stopping_detects_clear_winner():
    scheduler = AdaptiveBudgetLookaheadScheduler(min_rollouts=4, max_rollouts=8)
    candidate_a = (0, 0, 0, 0, 0)
    candidate_b = (1, 0, 1, 0, 1)
    candidates = {(0, 0): candidate_a, (0, 1): candidate_b}
    values = {(0, 0): [3, 3, 3, 3], (0, 1): [0, 0, 0, 0]}
    assert scheduler._winner_is_separated(values, candidates, minimum_effect=0.0)


def test_paired_confidence_stopping_does_not_claim_separation_for_ties():
    scheduler = AdaptiveBudgetLookaheadScheduler(min_rollouts=4, max_rollouts=8)
    candidate_a = (0, 0, 0, 0, 0)
    candidate_b = (1, 0, 1, 0, 1)
    candidates = {(0, 0): candidate_a, (0, 1): candidate_b}
    values = {(0, 0): [1, 2, 1, 2], (0, 1): [1, 2, 1, 2]}
    assert not scheduler._winner_is_separated(values, candidates, minimum_effect=0.0)


def test_adaptive_candidate_scores_and_selected_decision_are_consistent():
    scheduler = AdaptiveBudgetLookaheadScheduler(
        horizon=2, min_rollouts=2, max_rollouts=2, batch_size=2,
        minimum_history=4, profile_window=20, minimum_reliability=0.0,
        ambiguity_margin=10_000, max_candidates=3, seed=19,
    )
    decision = scheduler.choose_with_context(
        [GPUState(0), GPUState(1)], PROFILES["1g.10gb"], _stable_context()
    )
    assert decision is not None and decision.adaptive_status == "lookahead"
    assert decision.lookahead_value is not None
    assert decision.candidate_scores
    max_mean = max(candidate[3] for candidate in decision.candidate_scores)
    assert decision.lookahead_value == pytest.approx(max_mean)
    assert (decision.gpu_id, decision.start) in {
        (candidate[0], candidate[1])
        for candidate in decision.candidate_scores
        if candidate[3] == max_mean
    }


def test_adaptive_policy_is_registered_in_factory():
    from mig_scheduler.schedulers import make_scheduler

    assert isinstance(make_scheduler("adaptive_lookahead"), AdaptiveBudgetLookaheadScheduler)


def test_simulator_summary_tracks_adaptive_fallbacks_and_lookahead():
    from mig_scheduler.simulator import DiscreteEventSimulator, Job

    jobs = [
        Job("j0", 0.0, "1g.10gb", 2.0),
        Job("j1", 1.0, "1g.10gb", 2.0),
        Job("j2", 2.0, "2g.20gb", 2.0),
    ]
    scheduler = AdaptiveBudgetLookaheadScheduler(minimum_history=20, profile_window=30)
    result = DiscreteEventSimulator(1, scheduler).run(jobs)
    summary = result.summary()
    assert summary["adaptive_mfi_fallback_decisions"] >= 1
    assert summary["adaptive_lookahead_decisions"] == 0
    assert summary["adaptive_mean_total_rollouts"] == 0.0


def test_rejected_adaptive_arrival_records_zero_rollout_budget():
    from mig_scheduler.simulator import DiscreteEventSimulator, Job

    jobs = [
        Job("whole_gpu", 0.0, "7g.80gb", 10.0),
        Job("blocked_whole_gpu", 1.0, "7g.80gb", 1.0),
    ]
    scheduler = AdaptiveBudgetLookaheadScheduler(minimum_history=20, profile_window=30)
    result = DiscreteEventSimulator(1, scheduler).run(jobs)
    rows = {row["job_id"]: row for row in result.job_results}
    assert rows["blocked_whole_gpu"]["status"] == "rejected"
    assert rows["blocked_whole_gpu"]["total_rollouts"] == 0
    assert result.summary()["adaptive_no_feasible_decisions"] == 1
    assert result.summary()["adaptive_mean_total_rollouts"] == 0.0
