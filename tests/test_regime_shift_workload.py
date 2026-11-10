from collections import Counter

import pytest

from mig_scheduler.profiles import MFI_WORKLOAD_DISTRIBUTIONS
from mig_scheduler.workloads import generate_regime_shift_workload


def test_regime_shift_workload_is_reproducible_and_chronological():
    kwargs = dict(count=80, seed=19, shift_fraction=0.5)
    a = generate_regime_shift_workload(**kwargs)
    b = generate_regime_shift_workload(**kwargs)
    assert a == b
    assert len(a) == 80
    assert all(a[i].arrival_time <= a[i + 1].arrival_time for i in range(len(a) - 1))
    assert [j.job_id for j in a] == [f"shift-{i:05d}" for i in range(80)]


def test_regime_shift_uses_profile_distributions_on_each_side():
    jobs = generate_regime_shift_workload(
        count=6000, before_scenario="skew_small", after_scenario="skew_big",
        shift_fraction=0.5, seed=27,
    )
    midpoint = len(jobs) // 2
    before = Counter(job.profile_name for job in jobs[:midpoint])
    after = Counter(job.profile_name for job in jobs[midpoint:])
    small = {"1g.10gb", "1g.20gb", "2g.20gb"}
    before_small = sum(before[p] for p in small) / midpoint
    after_small = sum(after[p] for p in small) / midpoint
    assert before_small > 0.6
    assert after_small < 0.4
    assert before_small > after_small


@pytest.mark.parametrize("kwargs", [
    {"shift_fraction": 0.0},
    {"shift_fraction": 1.0},
    {"mean_interarrival": 0.0},
    {"min_duration": 5, "max_duration": 4},
    {"before_scenario": "unknown"},
])
def test_regime_shift_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        generate_regime_shift_workload(count=10, **kwargs)
