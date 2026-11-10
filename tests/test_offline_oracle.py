from __future__ import annotations

import pytest

from mig_scheduler.offline_oracle import OracleBudgetExceeded, solve_offline_optimal
from mig_scheduler.schedulers import (
    AdaptiveBudgetLookaheadScheduler,
    FirstFitScheduler,
    FixedBudgetLookaheadScheduler,
    MFIScheduler,
)
from mig_scheduler.simulator import DiscreteEventSimulator, Job


def test_oracle_accepts_all_non_overlapping_jobs():
    jobs = [
        Job("a", 0.0, "7g.80gb", 1.0),
        Job("b", 1.0, "7g.80gb", 1.0),
        Job("c", 2.0, "7g.80gb", 1.0),
    ]
    result = solve_offline_optimal(jobs, gpu_count=1)
    assert result.optimal_accepted == 3
    assert result.acceptance_rate == 1.0
    assert all(d.accepted for d in result.decisions)


def test_oracle_respects_overlap_and_profile_geometry():
    jobs = [
        Job("a", 0.0, "7g.80gb", 10.0),
        Job("b", 1.0, "7g.80gb", 10.0),
        Job("c", 2.0, "7g.80gb", 10.0),
    ]
    result = solve_offline_optimal(jobs, gpu_count=1)
    assert result.optimal_accepted == 1
    assert sum(d.accepted for d in result.decisions) == 1


def test_oracle_is_at_least_as_good_as_online_policies_on_same_trace():
    jobs = [
        Job("a", 0.0, "2g.20gb", 5.0),
        Job("b", 0.2, "1g.20gb", 3.0),
        Job("c", 0.4, "3g.40gb", 2.0),
        Job("d", 1.0, "1g.10gb", 4.0),
        Job("e", 3.0, "4g.40gb", 2.0),
    ]
    oracle = solve_offline_optimal(jobs, gpu_count=1)
    for scheduler in (
        FirstFitScheduler(),
        MFIScheduler(),
        FixedBudgetLookaheadScheduler(horizon=2, rollouts=2, seed=4),
        AdaptiveBudgetLookaheadScheduler(
            horizon=2, min_rollouts=2, max_rollouts=2, batch_size=2, seed=4,
            minimum_history=4, profile_window=6, minimum_reliability=0.0,
            ambiguity_margin=100, max_candidates=4,
        ),
    ):
        online = DiscreteEventSimulator(1, scheduler).run(jobs)
        assert online.accepted <= oracle.optimal_accepted


def test_oracle_respects_completion_before_same_time_arrival():
    jobs = [
        Job("a", 0.0, "7g.80gb", 5.0),
        Job("b", 5.0, "7g.80gb", 1.0),
    ]
    result = solve_offline_optimal(jobs, gpu_count=1)
    assert result.optimal_accepted == 2
    assert [d.accepted for d in result.decisions] == [True, True]


def test_oracle_is_deterministic_and_recovers_a_valid_schedule():
    jobs = [
        Job("a", 0.0, "1g.10gb", 4.0),
        Job("b", 0.1, "1g.20gb", 2.0),
        Job("c", 1.0, "2g.20gb", 3.0),
        Job("d", 4.0, "7g.80gb", 1.0),
    ]
    first = solve_offline_optimal(jobs, gpu_count=2)
    second = solve_offline_optimal(jobs, gpu_count=2)
    assert first.decisions == second.decisions
    assert first.optimal_accepted == second.optimal_accepted
    assert first.states_explored == second.states_explored
    assert first.transitions_considered == second.transitions_considered
    assert sum(decision.accepted for decision in first.decisions) == first.optimal_accepted
    assert len(first.decisions) == len(jobs)


def test_oracle_rejects_duplicate_ids_and_oversized_trace():
    with pytest.raises(ValueError, match="unique"):
        solve_offline_optimal(
            [Job("x", 0, "1g.10gb", 1), Job("x", 1, "1g.10gb", 1)], 1
        )
    with pytest.raises(ValueError, match="limited"):
        solve_offline_optimal(
            [Job(str(i), float(i), "1g.10gb", 1) for i in range(3)],
            1, max_jobs=2,
        )


def test_oracle_has_explicit_state_budget():
    jobs = [Job(str(i), 0.0, "1g.10gb", 10.0) for i in range(5)]
    with pytest.raises(OracleBudgetExceeded):
        solve_offline_optimal(jobs, gpu_count=2, max_jobs=8, max_states=1)


def test_oracle_matches_independent_bruteforce_enumeration_on_four_jobs():
    """Cross-check the memoized DP against a no-memo recursive enumerator."""
    from mig_scheduler.fragmentation import delta_fragmentation
    from mig_scheduler.profiles import PROFILES
    from mig_scheduler.state import GPUState

    jobs = [
        Job("a", 0.0, "1g.10gb", 3.0),
        Job("b", 0.4, "1g.20gb", 2.0),
        Job("c", 1.1, "2g.20gb", 3.0),
        Job("d", 3.5, "4g.40gb", 1.0),
    ]
    gpu_count = 1

    def brute(i, active):
        if i >= len(jobs):
            return 0
        t = jobs[i].arrival_time
        live = tuple(sorted(
            a for a in active if jobs[a[0]].arrival_time + jobs[a[0]].duration > t
        ))
        gpus = [GPUState(k) for k in range(gpu_count)]
        for j, gid, start in live:
            gpus[gid].allocate(jobs[j].job_id, jobs[j].profile, start)
        best = brute(i + 1, live)  # reject branch
        profile = PROFILES[jobs[i].profile_name]
        for gpu in gpus:
            for start in gpu.feasible_starts(profile):
                # The exact objective is acceptance count; delta is calculated
                # only to ensure legal candidate evaluation uses the same geometry.
                delta_fragmentation(gpu, profile, start)
                next_live = tuple(sorted(live + ((i, gpu.gpu_id, start),)))
                best = max(best, 1 + brute(i + 1, next_live))
        return best

    expected = brute(0, ())
    actual = solve_offline_optimal(jobs, gpu_count=gpu_count)
    assert actual.optimal_accepted == expected
