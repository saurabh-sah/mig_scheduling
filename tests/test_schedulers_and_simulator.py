from mig_scheduler.profiles import PROFILES
from mig_scheduler.schedulers import (
    FirstFitScheduler, RoundRobinScheduler, BestFitBestIndexScheduler,
    WorstFitBestIndexScheduler, MFIScheduler,
)
from mig_scheduler.simulator import DiscreteEventSimulator, Job
from mig_scheduler.state import GPUState
from mig_scheduler.workloads import generate_workload


def test_mfi_always_returns_a_feasible_placement_when_one_exists():
    gpus = [GPUState(0), GPUState(1)]
    profile = PROFILES["2g.20gb"]
    decision = MFIScheduler().choose(gpus, profile)
    assert decision is not None
    assert gpus[decision.gpu_id].is_placement_feasible(profile, decision.start)


def test_first_fit_and_mfi_return_placement_for_empty_gpu():
    profile = PROFILES["1g.10gb"]
    for scheduler in (FirstFitScheduler(), MFIScheduler()):
        decision = scheduler.choose([GPUState(0)], profile)
        assert decision is not None
        assert decision.gpu_id == 0
        assert 0 <= decision.start <= 6


def test_event_simulation_acceptance_and_release():
    jobs = [
        Job("J1", 0.0, "2g.20gb", 5.0),
        Job("J2", 1.0, "2g.20gb", 3.0),
        Job("J3", 2.0, "1g.10gb", 2.0),
    ]
    result = DiscreteEventSimulator(1, FirstFitScheduler()).run(jobs)
    assert result.submitted == 3
    assert result.accepted + result.rejected == 3
    assert "J1" in {row["job_id"] for row in result.job_results if row["status"] == "accepted"}
    assert any("COMPLETE J1" in line for line in result.event_log)


def test_jobs_complete_before_same_time_arrival():
    jobs = [
        Job("J1", 0.0, "7g.80gb", 5.0),
        Job("J2", 5.0, "7g.80gb", 1.0),
    ]
    result = DiscreteEventSimulator(1, FirstFitScheduler()).run(jobs)
    statuses = {row["job_id"]: row["status"] for row in result.job_results}
    assert statuses == {"J1": "accepted", "J2": "accepted"}


def test_workload_generator_is_reproducible():
    first = generate_workload(count=15, scenario="skew_small", seed=123)
    second = generate_workload(count=15, scenario="skew_small", seed=123)
    assert first == second


def test_mfi_selects_minimum_delta_among_all_feasible_candidates():
    from mig_scheduler.fragmentation import delta_fragmentation

    gpus = [GPUState(0), GPUState(1)]
    gpus[0].allocate("existing-0", PROFILES["1g.10gb"], 1)
    gpus[1].allocate("existing-1", PROFILES["2g.20gb"], 0)
    requested = PROFILES["1g.20gb"]

    candidates = []
    for gpu in gpus:
        for start in gpu.feasible_starts(requested):
            before, after, delta = delta_fragmentation(gpu, requested, start)
            candidates.append((delta, gpu.gpu_id, start))

    decision = MFIScheduler().choose(gpus, requested)
    assert decision is not None
    assert (decision.delta_fragmentation, decision.gpu_id, decision.start) == min(candidates)


def test_all_published_baseline_policies_choose_only_feasible_placements():
    profile = PROFILES["1g.20gb"]
    for scheduler in (FirstFitScheduler(), RoundRobinScheduler(),
                      BestFitBestIndexScheduler(), WorstFitBestIndexScheduler(), MFIScheduler()):
        gpus = [GPUState(0), GPUState(1), GPUState(2)]
        decision = scheduler.choose(gpus, profile)
        assert decision is not None, scheduler.name
        assert gpus[decision.gpu_id].is_placement_feasible(profile, decision.start), scheduler.name


def test_best_index_prefers_index_6_for_1g10_on_empty_gpu():
    decision = BestFitBestIndexScheduler().choose([GPUState(0)], PROFILES["1g.10gb"])
    assert decision is not None
    assert decision.start == 6


def test_round_robin_advances_between_feasible_gpus():
    scheduler = RoundRobinScheduler()
    gpus = [GPUState(0), GPUState(1), GPUState(2)]
    profile = PROFILES["1g.10gb"]
    first = scheduler.choose(gpus, profile)
    second = scheduler.choose(gpus, profile)
    assert first is not None and second is not None
    assert first.gpu_id == 0
    assert second.gpu_id == 1


def test_simulator_emits_one_arrival_snapshot_per_job():
    jobs = [
        Job("J1", 0.0, "1g.10gb", 2.0),
        Job("J2", 1.0, "1g.20gb", 2.0),
        Job("J3", 4.0, "7g.80gb", 1.0),
    ]
    result = DiscreteEventSimulator(2, MFIScheduler()).run(jobs)
    assert len(result.snapshots) == len(jobs)
    for row in result.snapshots:
        assert 0.0 <= row["compute_utilization"] <= 1.0
        assert 0.0 <= row["memory_utilization"] <= 1.0
        assert 0.0 <= row["acceptance_rate_so_far"] <= 1.0
