from copy import deepcopy

import pytest

from mig_scheduler.context import ArrivalObservation, RunningJobObservation, SchedulingContext
from mig_scheduler.profiles import PROFILES, PROFILE_ORDER
from mig_scheduler.schedulers import FixedBudgetLookaheadScheduler, MFIScheduler
from mig_scheduler.simulator import DiscreteEventSimulator, Job
from mig_scheduler.state import GPUState
from mig_scheduler.workloads import generate_workload


def _context(current_time=2.0):
    return SchedulingContext(
        current_time=current_time,
        past_arrivals=(
            ArrivalObservation(0.0, "1g.10gb"),
            ArrivalObservation(0.6, "1g.10gb"),
            ArrivalObservation(1.2, "2g.20gb"),
        ),
        completed_durations_by_profile={
            "1g.10gb": (2.0, 3.0),
            "2g.20gb": (4.0,),
        },
        running_jobs=(),
    )


def test_fixed_lookahead_validates_parameters():
    with pytest.raises(ValueError):
        FixedBudgetLookaheadScheduler(horizon=0)
    with pytest.raises(ValueError):
        FixedBudgetLookaheadScheduler(rollouts=0)


def test_fixed_lookahead_probabilities_use_only_past_arrivals_and_sum_to_one():
    scheduler = FixedBudgetLookaheadScheduler(horizon=2, rollouts=2, seed=7)
    names, probabilities = scheduler._profile_probabilities(_context())
    assert set(names) == set(PROFILE_ORDER)
    assert sum(probabilities) == pytest.approx(1.0)
    assert probabilities[names.index("1g.10gb")] > probabilities[names.index("7g.80gb")]


def test_fixed_lookahead_returns_feasible_candidate_and_reports_budget():
    gpus = [GPUState(0), GPUState(1)]
    scheduler = FixedBudgetLookaheadScheduler(horizon=2, rollouts=3, seed=11)
    decision = scheduler.choose_with_context(gpus, PROFILES["1g.10gb"], _context())
    assert decision is not None
    assert gpus[decision.gpu_id].is_placement_feasible(PROFILES["1g.10gb"], decision.start)
    assert decision.lookahead_value is not None
    assert decision.lookahead_std is not None
    assert decision.lookahead_horizon == 2
    assert decision.candidate_count >= 1
    assert decision.rollouts_per_candidate == 3
    assert decision.total_rollouts == decision.candidate_count * 3
    assert decision.mfi_gpu_id is not None
    assert decision.mfi_start is not None
    assert len(decision.candidate_scores) == decision.candidate_count
    best_score = max(row[3] for row in decision.candidate_scores)
    assert decision.lookahead_value == pytest.approx(best_score)


def test_fixed_lookahead_does_not_mutate_live_state_when_evaluating_candidates():
    gpus = [GPUState(0), GPUState(1)]
    gpus[0].allocate("running", PROFILES["1g.10gb"], 5)
    before = deepcopy(gpus)
    scheduler = FixedBudgetLookaheadScheduler(horizon=3, rollouts=2, seed=33)
    ctx = SchedulingContext(
        current_time=1.0,
        past_arrivals=(ArrivalObservation(0.0, "1g.10gb"),),
        completed_durations_by_profile={"1g.10gb": (5.0, 6.0)},
        running_jobs=(RunningJobObservation(
            job_id="running", profile_name="1g.10gb", arrival_time=0.0, gpu_id=0
        ),),
    )
    decision = scheduler.choose_with_context(gpus, PROFILES["1g.20gb"], ctx)
    assert decision is not None
    assert gpus == before


def test_fixed_lookahead_is_reproducible_for_same_seed_and_context():
    profile = PROFILES["2g.20gb"]
    first = FixedBudgetLookaheadScheduler(horizon=2, rollouts=4, seed=91)
    second = FixedBudgetLookaheadScheduler(horizon=2, rollouts=4, seed=91)
    decision_a = first.choose_with_context([GPUState(0)], profile, _context())
    decision_b = second.choose_with_context([GPUState(0)], profile, _context())
    assert decision_a is not None and decision_b is not None
    assert (decision_a.gpu_id, decision_a.start, decision_a.lookahead_value) == (
        decision_b.gpu_id, decision_b.start, decision_b.lookahead_value
    )


def test_fixed_lookahead_falls_back_to_mfi_without_context():
    gpus = [GPUState(0), GPUState(1)]
    profile = PROFILES["2g.20gb"]
    expected = MFIScheduler().choose(gpus, profile)
    actual = FixedBudgetLookaheadScheduler().choose(gpus, profile)
    assert expected is not None and actual is not None
    assert (actual.gpu_id, actual.start) == (expected.gpu_id, expected.start)
    assert actual.policy == "fixed_lookahead"
    assert actual.total_rollouts == 0


def test_simulator_runs_fixed_lookahead_and_records_diagnostics():
    jobs = generate_workload(count=18, scenario="skew_small", seed=123, mean_interarrival=0.7)
    scheduler = FixedBudgetLookaheadScheduler(horizon=2, rollouts=2, seed=456)
    result = DiscreteEventSimulator(2, scheduler).run(jobs)
    assert result.accepted + result.rejected == len(jobs)
    lookahead_rows = [row for row in result.job_results if row.get("lookahead_value") is not None]
    assert lookahead_rows
    assert all(row["total_rollouts"] == row["candidate_count"] * 2 for row in lookahead_rows)
    assert result.total_decision_time_ns > 0
    assert result.summary()["lookahead_decisions"] == len(lookahead_rows)
