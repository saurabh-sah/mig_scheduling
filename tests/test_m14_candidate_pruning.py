"""Tests for Milestone 14 Task 2: GPU-diverse candidate pruning in lookahead scheduling."""
from copy import deepcopy
import pytest

from mig_scheduler.context import ArrivalObservation, SchedulingContext
from mig_scheduler.mfi_score_variants import (
    fragmentation_score_variant,
    fragmentation_score_variant_from_occupancy,
)
from mig_scheduler.profiles import PROFILE_ORDER, PROFILES
from mig_scheduler.schedulers import (
    AdaptiveBudgetLookaheadScheduler,
    FixedBudgetLookaheadScheduler,
)
from mig_scheduler.simulator import DiscreteEventSimulator, Job
from mig_scheduler.state import GPUState


def _make_context(size: int = 12, duration: float = 4.0) -> SchedulingContext:
    """Create a synthetic scheduling context for testing lookahead."""
    observations = tuple(
        ArrivalObservation(float(i), "1g.10gb") for i in range(size)
    )
    return SchedulingContext(
        current_time=float(size),
        past_arrivals=observations,
        completed_durations_by_profile={
            name: (duration, duration + 1.0) for name in PROFILE_ORDER
        },
        running_jobs=(),
    )


def test_fixed_lookahead_validates_max_candidates_parameter():
    """Verify that max_candidates < 1 raises ValueError while None and >= 1 are accepted."""
    with pytest.raises(ValueError, match="max_candidates must be >= 1"):
        FixedBudgetLookaheadScheduler(max_candidates=0)
    with pytest.raises(ValueError, match="max_candidates must be >= 1"):
        FixedBudgetLookaheadScheduler(max_candidates=-5)

    s1 = FixedBudgetLookaheadScheduler(max_candidates=1)
    assert s1.max_candidates == 1
    s_none = FixedBudgetLookaheadScheduler(max_candidates=None)
    assert s_none.max_candidates is None


def test_gpu_diversity_prioritization_across_multiple_gpus():
    """Test that when max_candidates=5 on a cluster with 4 GPUs, candidates across distinct GPUs are prioritized.

    On 4 empty GPUs, 1g.10gb has 7 feasible starts per GPU (28 total).
    Without GPU diversity, a naive top-5 would take all 5 starts from GPU 0.
    With GPU diversity, all 4 GPUs must be represented in the shortlisted 5 candidates.
    """
    gpus = [GPUState(0), GPUState(1), GPUState(2), GPUState(3)]
    profile = PROFILES["1g.10gb"]
    context = _make_context()

    scheduler = FixedBudgetLookaheadScheduler(
        horizon=2,
        rollouts=2,
        seed=42,
        max_candidates=5,
    )

    decision = scheduler.choose_with_context(gpus, profile, context)
    assert decision is not None
    assert decision.candidate_count == 5
    assert decision.candidate_pool_size == 28
    assert len(decision.candidate_scores) == 5

    # Extract the GPU IDs of all evaluated candidate placements
    candidate_gpu_ids = [score[0] for score in decision.candidate_scores]
    unique_gpus = set(candidate_gpu_ids)

    # All 4 GPUs must be represented among the 5 candidates
    assert len(unique_gpus) == 4, (
        f"Expected all 4 GPUs to be represented, but got {unique_gpus}"
    )
    assert unique_gpus == {0, 1, 2, 3}
    assert decision.total_rollouts == 5 * scheduler.rollouts


def test_total_evaluated_candidates_does_not_exceed_max_candidates():
    """Test that candidate_count and total_rollouts strictly respect max_candidates."""
    gpus = [GPUState(0), GPUState(1), GPUState(2)]
    profile = PROFILES["1g.10gb"]  # 21 feasible starts total across 3 GPUs
    context = _make_context()

    for cap in [1, 2, 4, 7, 10]:
        scheduler = FixedBudgetLookaheadScheduler(
            horizon=2,
            rollouts=3,
            seed=100,
            max_candidates=cap,
        )
        decision = scheduler.choose_with_context(gpus, profile, context)
        assert decision is not None
        assert decision.candidate_count == cap
        assert decision.candidate_pool_size == 21
        assert len(decision.candidate_scores) == cap
        assert decision.total_rollouts == cap * 3


def test_when_max_candidates_is_none_all_candidates_are_evaluated():
    """Test that when max_candidates is None, all feasible candidates are evaluated (backward compatible)."""
    gpus = [GPUState(0), GPUState(1)]
    profile = PROFILES["1g.10gb"]  # 14 feasible starts
    context = _make_context()

    scheduler_none = FixedBudgetLookaheadScheduler(
        horizon=2,
        rollouts=2,
        seed=42,
        max_candidates=None,
    )
    scheduler_default = FixedBudgetLookaheadScheduler(
        horizon=2,
        rollouts=2,
        seed=42,
    )

    decision_none = scheduler_none.choose_with_context(gpus, profile, context)
    decision_default = scheduler_default.choose_with_context(gpus, profile, context)

    assert decision_none is not None and decision_default is not None
    assert decision_none.candidate_count == 14
    assert decision_none.candidate_pool_size == 14
    assert decision_none.candidate_count == decision_default.candidate_count
    assert decision_none.total_rollouts == 14 * 2
    assert (decision_none.gpu_id, decision_none.start) == (
        decision_default.gpu_id,
        decision_default.start,
    )
    assert decision_none.lookahead_value == pytest.approx(decision_default.lookahead_value)


@pytest.mark.parametrize("mode", ["all_legal_starts", "free_anchor_starts"])
def test_candidate_pruning_works_with_both_score_modes(mode):
    """Test that candidate pruning works seamlessly with free_anchor_starts and all_legal_starts."""
    gpus = [GPUState(0), GPUState(1), GPUState(2)]
    # Place a partial allocation on GPU 0 to induce non-zero fragmentation differences
    gpus[0].allocate("j0", PROFILES["2g.20gb"], 0)
    profile = PROFILES["1g.10gb"]
    context = _make_context()

    scheduler = FixedBudgetLookaheadScheduler(
        horizon=2,
        rollouts=2,
        seed=55,
        score_mode=mode,
        max_candidates=4,
    )

    decision = scheduler.choose_with_context(gpus, profile, context)
    assert decision is not None
    assert decision.candidate_count == 4
    assert len(decision.candidate_scores) == 4

    # Verify that the immediate delta values stored in candidate_scores match
    # the fragmentation score computed under this specific score_mode
    for gpu_id, start, delta, _val, _std in decision.candidate_scores:
        gpu = gpus[gpu_id]
        before = fragmentation_score_variant(gpu, mode=mode)
        comp, mem = gpu.hypothetical_occupancy(profile, start)
        after = fragmentation_score_variant_from_occupancy(comp, mem, mode=mode)
        assert delta == after - before


def test_gpu_state_remains_completely_unmutated_before_and_after_lookahead():
    """Test that GPUState instances are completely unmutated before and after lookahead scheduling."""
    gpus = [GPUState(0), GPUState(1), GPUState(2)]
    gpus[0].allocate("active_0", PROFILES["2g.20gb"], 0)
    gpus[1].allocate("active_1", PROFILES["1g.10gb"], 0)
    gpus[1].allocate("active_2", PROFILES["1g.10gb"], 1)

    before = deepcopy(gpus)
    context = _make_context()

    scheduler = FixedBudgetLookaheadScheduler(
        horizon=3,
        rollouts=4,
        seed=77,
        max_candidates=5,
    )

    decision = scheduler.choose_with_context(gpus, PROFILES["1g.10gb"], context)
    assert decision is not None

    # Strict state equality check
    assert gpus == before, "Live GPU states must not be mutated by lookahead"
    for i in range(len(gpus)):
        assert gpus[i].allocations == before[i].allocations
        assert gpus[i].free_compute_slices == before[i].free_compute_slices
        assert gpus[i].free_memory_slices == before[i].free_memory_slices
        assert gpus[i].compute_occupied_by == before[i].compute_occupied_by
        assert gpus[i].memory_occupied_by == before[i].memory_occupied_by


def test_candidate_shortlisting_when_feasible_count_less_than_max_candidates():
    """When the number of feasible placements is <= max_candidates, all are evaluated."""
    gpus = [GPUState(0)]
    # 7g.80gb has only 1 legal start on an empty GPU
    profile = PROFILES["7g.80gb"]
    context = _make_context()

    scheduler = FixedBudgetLookaheadScheduler(
        horizon=2,
        rollouts=2,
        seed=1,
        max_candidates=10,
    )
    decision = scheduler.choose_with_context(gpus, profile, context)
    assert decision is not None
    assert decision.candidate_count == 1
    assert decision.candidate_pool_size == 1
    assert len(decision.candidate_scores) == 1


def test_cli_max_lookahead_candidates_argument():
    """Verify that CLI argument --max-lookahead-candidates is parsed and passed to the scheduler."""
    import argparse
    from mig_scheduler.cli import main

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["fixed_lookahead", "adaptive_lookahead"], default="fixed_lookahead")
    parser.add_argument("--lookahead-horizon", type=int, default=3)
    parser.add_argument("--lookahead-rollouts", type=int, default=8)
    parser.add_argument("--lookahead-seed", type=int, default=42)
    parser.add_argument("--mfi-score-mode", choices=["all_legal_starts", "free_anchor_starts"], default="free_anchor_starts")
    parser.add_argument("--max-lookahead-candidates", type=int, default=None)

    args = parser.parse_args(["--max-lookahead-candidates", "6"])
    assert args.max_lookahead_candidates == 6

    scheduler = FixedBudgetLookaheadScheduler(
        horizon=args.lookahead_horizon,
        rollouts=args.lookahead_rollouts,
        seed=args.lookahead_seed,
        score_mode=args.mfi_score_mode,
        max_candidates=args.max_lookahead_candidates,
    )
    assert scheduler.max_candidates == 6


def test_end_to_end_simulator_with_pruned_lookahead():
    """Verify end-to-end discrete-event simulation with FixedBudgetLookaheadScheduler(max_candidates=3)."""
    jobs = [
        Job("j0", 0.0, "1g.10gb", 2.0),
        Job("j1", 0.5, "1g.10gb", 2.0),
        Job("j2", 1.0, "2g.20gb", 3.0),
        Job("j3", 1.5, "3g.40gb", 2.0),
    ]
    scheduler = FixedBudgetLookaheadScheduler(
        horizon=2,
        rollouts=2,
        seed=42,
        max_candidates=3,
    )
    sim = DiscreteEventSimulator(gpu_count=2, scheduler=scheduler)
    result = sim.run(jobs)
    summary = result.summary()

    assert summary["submitted"] == 4
    assert summary["accepted"] == 4
    for row in result.job_results:
        if row["status"] == "accepted":
            assert row["candidate_count"] <= 3
            assert row["candidate_pool_size"] >= row["candidate_count"]
            assert row["total_rollouts"] == row["candidate_count"] * 2
