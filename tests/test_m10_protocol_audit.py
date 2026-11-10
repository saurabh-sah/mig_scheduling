import json
import math
from pathlib import Path

from mig_scheduler.profiles import MFI_WORKLOAD_DISTRIBUTIONS, PROFILES
from mig_scheduler.workloads import generate_progressive_load_workload
from scripts.run_m10_protocol_dynamics import MODES, make_modes

ROOT = Path(__file__).resolve().parents[1]


def test_m10_modes_preserve_profile_sequence_and_ids():
    base = generate_progressive_load_workload(gpu_count=2, scenario="skew_small", seed=11)
    modes = make_modes(base, seed=11)
    assert tuple(modes) == MODES
    for mode, jobs in modes.items():
        assert [j.job_id for j in jobs] == [j.job_id for j in base]
        assert [j.profile_name for j in jobs] == [j.profile_name for j in base]


def test_m10_paper_candidate_is_unchanged():
    base = generate_progressive_load_workload(gpu_count=2, scenario="uniform", seed=14)
    modes = make_modes(base, seed=14)
    assert modes["paper_candidate"] == base


def test_m10_arrival_modes_have_expected_timing():
    base = generate_progressive_load_workload(gpu_count=2, scenario="uniform", seed=15)
    modes = make_modes(base, seed=15)
    assert all(job.arrival_time == 0.0 for job in modes["burst_arrivals"])
    assert [job.arrival_time for job in modes["sparse_arrivals"]] == [float(i * 2) for i in range(len(base))]


def test_m10_short_and_long_lifetime_modes_obey_bounds():
    base = generate_progressive_load_workload(gpu_count=2, scenario="uniform", seed=17)
    modes = make_modes(base, seed=17)
    T = len(base)
    short_hi = max(1, T // 4)
    long_lo = max(1, math.ceil(0.75 * T))
    assert all(1 <= j.duration <= short_hi for j in modes["short_lifetimes"])
    assert all(long_lo <= j.duration <= T for j in modes["long_lifetimes"])
    assert all(j.duration == T + 1 for j in modes["no_completion_before_stream_end"])


def test_m10_source_protocol_json_is_machine_readable_and_caveated():
    protocol = json.loads((ROOT / "configs/protocols/mfi_paper_candidate.json").read_text())
    assert protocol["cluster"]["gpu_count"] == 100
    assert protocol["repetitions_per_distribution"] == 500
    assert "not an exact replication" in protocol["important_qualification"].lower()


def test_m10_stress_modes_are_explicitly_not_replication_protocol():
    config = json.loads((ROOT / "configs/protocols/mfi_stress_modes.json").read_text())
    assert "sensitivity analysis only" in config["purpose"].lower()
    assert len(config["modes"]) == len(MODES)


def test_m10_distribution_definitions_remain_normalized():
    assert set(MFI_WORKLOAD_DISTRIBUTIONS) == {"uniform", "skew_small", "skew_big", "bimodal"}
    for distribution in MFI_WORKLOAD_DISTRIBUTIONS.values():
        assert abs(sum(distribution.values()) - 1.0) < 1e-9
    assert len(PROFILES) == 6
