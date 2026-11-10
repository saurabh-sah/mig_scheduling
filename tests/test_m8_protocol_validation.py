from pathlib import Path

from mig_scheduler.profiles import MFI_WORKLOAD_DISTRIBUTIONS, PROFILES
from mig_scheduler.workloads import generate_progressive_load_workload
from mig_scheduler.schedulers import MFIScheduler
from mig_scheduler.simulator import DiscreteEventSimulator


def test_progressive_load_workload_reaches_requested_budget_reproducibly():
    first = generate_progressive_load_workload(gpu_count=4, scenario="uniform", seed=101)
    second = generate_progressive_load_workload(gpu_count=4, scenario="uniform", seed=101)
    assert first == second
    assert sum(job.profile.paper_slice_units for job in first) >= 7 * 4
    assert len(first) > 0
    assert all(job.arrival_time == float(i) for i, job in enumerate(first))


def test_progressive_load_durations_are_within_uniform_integer_slot_bounds():
    jobs = generate_progressive_load_workload(gpu_count=3, scenario="skew_small", seed=202)
    T = len(jobs)
    assert all(float(d).is_integer() for d in (job.duration for job in jobs))
    assert all(1 <= job.duration <= T for job in jobs)


def test_mfi_workload_distributions_are_normalized_and_cover_profiles():
    assert set(MFI_WORKLOAD_DISTRIBUTIONS) == {"uniform", "skew_small", "skew_big", "bimodal"}
    for distribution in MFI_WORKLOAD_DISTRIBUTIONS.values():
        assert set(distribution) == set(PROFILES)
        assert abs(sum(distribution.values()) - 1.0) < 1e-9


def test_progressive_load_trace_can_run_through_every_primary_baseline():
    from mig_scheduler.schedulers import (
        FirstFitScheduler,
        RoundRobinScheduler,
        BestFitBestIndexScheduler,
        WorstFitBestIndexScheduler,
    )
    jobs = generate_progressive_load_workload(gpu_count=2, scenario="bimodal", seed=99)
    for scheduler in (
        FirstFitScheduler(), RoundRobinScheduler(), BestFitBestIndexScheduler(),
        WorstFitBestIndexScheduler(), MFIScheduler(),
    ):
        result = DiscreteEventSimulator(2, scheduler).run(jobs)
        assert result.submitted == len(jobs)
        assert result.accepted + result.rejected == len(jobs)
        assert 0.0 <= result.acceptance_rate <= 1.0
        assert len(result.snapshots) == len(jobs)


def test_m8_runner_and_analyzer_are_present():
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "run_m8_protocol_validation.py").exists()
    assert (root / "scripts" / "analyze_m8_protocol_validation.py").exists()
    assert (root / "M8_MODEL_FIDELITY_AUDIT.md").exists()


def test_paper_ff_operationalization_exposes_first_gpu_fragmentation_failure():
    from mig_scheduler.paper_baselines import PaperFirstFitScheduler, PaperRoundRobinScheduler
    from mig_scheduler.schedulers import FirstFitScheduler
    from mig_scheduler.state import GPUState

    gpus = [GPUState(0), GPUState(1)]
    for index in (0, 2, 4, 6):
        gpus[0].allocate(f"small-{index}", PROFILES["1g.10gb"], index)
    requested = PROFILES["1g.20gb"]

    # GPU 0 has enough aggregate resources, but no legal 1g.20gb placement.
    assert gpus[0].free_compute_slices >= requested.compute_slices
    assert gpus[0].free_memory_slices >= requested.memory_slices
    assert gpus[0].feasible_starts(requested) == []
    # The repository's always-search-first-feasible policy can use GPU 1.
    decision = FirstFitScheduler().choose(gpus, requested)
    assert decision is not None and decision.gpu_id == 1
    # The paper-style operationalization deliberately fails on the first
    # aggregate-capacity GPU, making the index-fragmentation failure explicit.
    assert PaperFirstFitScheduler().choose(gpus, requested) is None
    assert PaperRoundRobinScheduler().choose(gpus, requested) is None
