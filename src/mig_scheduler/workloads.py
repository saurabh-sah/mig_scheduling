"""Synthetic workload generation for MFI-inspired evaluation."""

from __future__ import annotations

import math
import random
from .profiles import (MFI_WORKLOAD_DISTRIBUTIONS, PROFILE_ORDER, PROFILES,
                       COMPUTE_SLICE_COUNT, MEMORY_SLICE_COUNT)
from .simulator import Job


def generate_workload(
    *, count: int, scenario: str = "skew_small", seed: int = 42,
    mean_interarrival: float = 1.0, min_duration: int = 1, max_duration: int = 12,
) -> list[Job]:
    """Generate a reproducible Poisson-arrival synthetic workload.

    This is a convenient controlled workload, not the exact MFI load-sweep protocol.
    """
    if count < 0:
        raise ValueError("count must be >= 0")
    if scenario not in MFI_WORKLOAD_DISTRIBUTIONS:
        raise ValueError(f"Unknown scenario {scenario!r}; choose from {sorted(MFI_WORKLOAD_DISTRIBUTIONS)}")
    if mean_interarrival <= 0:
        raise ValueError("mean_interarrival must be > 0")
    if min_duration < 1 or max_duration < min_duration:
        raise ValueError("Invalid duration bounds")

    rng = random.Random(seed)
    dist = MFI_WORKLOAD_DISTRIBUTIONS[scenario]
    names = list(PROFILE_ORDER)
    weights = [dist[name] for name in names]
    time = 0.0
    jobs: list[Job] = []
    for i in range(count):
        if i:
            time += rng.expovariate(1.0 / mean_interarrival)
        name = rng.choices(names, weights=weights, k=1)[0]
        duration = float(rng.randint(min_duration, max_duration))
        jobs.append(Job(f"job-{i:05d}", time, name, duration))
    return jobs


def generate_progressive_load_workload(
    *, gpu_count: int, scenario: str = "uniform", seed: int = 42,
    arrival_gap: float = 1.0, load_basis: str = "paper_slice_units",
    target_fraction: float = 1.0,
) -> list[Job]:
    """Generate a progressive offered-load sequence inspired by MFI Section VI.

    ``load_basis="paper_slice_units"`` is the existing MFI-inspired default:
    it stops when the sum of Table-I scalar Slice units reaches 7 units per GPU.
    The alternative bases are sensitivity experiments, not claims about the
    paper's protocol:

    - ``compute``: cumulative requested compute slices reach 7 per GPU;
    - ``memory``: cumulative requested memory slices reach 8 per GPU;
    - ``max_normalized``: stop when either normalized compute or memory demand
      reaches ``target_fraction`` of that dimension's cluster capacity.

    ``arrival_gap`` is independent of the load-basis choice. For a fixed seed
    and load basis, changing it changes arrival times only. A gap of zero creates
    a burst where all jobs arrive before any positive-duration job completes.

    This is a model-sensitivity generator, not an exact reproduction of MFI.
    The paper does not release its source code, and its scalar Slice load axis
    and the physical two-dimensional compute/memory model are not identical.
    """
    if gpu_count < 1:
        raise ValueError("gpu_count must be >= 1")
    if arrival_gap < 0:
        raise ValueError("arrival_gap must be >= 0")
    if not 0 < target_fraction <= 2:
        raise ValueError("target_fraction must be > 0 and <= 2")
    valid_bases = {"paper_slice_units", "compute", "memory", "max_normalized"}
    if load_basis not in valid_bases:
        raise ValueError(f"load_basis must be one of {sorted(valid_bases)}")
    if scenario not in MFI_WORKLOAD_DISTRIBUTIONS:
        raise ValueError(f"Unknown scenario {scenario!r}; choose from {sorted(MFI_WORKLOAD_DISTRIBUTIONS)}")

    rng = random.Random(seed)
    dist = MFI_WORKLOAD_DISTRIBUTIONS[scenario]
    names = list(PROFILE_ORDER)
    weights = [dist[name] for name in names]
    target_paper_units = 7 * gpu_count * target_fraction
    target_compute = COMPUTE_SLICE_COUNT * gpu_count * target_fraction
    target_memory = MEMORY_SLICE_COUNT * gpu_count * target_fraction

    sampled_profiles: list[str] = []
    offered_paper_units = 0
    offered_compute = 0
    offered_memory = 0
    max_jobs = max(100, int(20 * gpu_count * target_fraction))
    while len(sampled_profiles) < max_jobs:
        if load_basis == "paper_slice_units":
            reached = offered_paper_units >= target_paper_units
        elif load_basis == "compute":
            reached = offered_compute >= target_compute
        elif load_basis == "memory":
            reached = offered_memory >= target_memory
        else:
            compute_fraction = offered_compute / (COMPUTE_SLICE_COUNT * gpu_count)
            memory_fraction = offered_memory / (MEMORY_SLICE_COUNT * gpu_count)
            reached = max(compute_fraction, memory_fraction) >= target_fraction
        if reached and sampled_profiles:
            break
        name = rng.choices(names, weights=weights, k=1)[0]
        profile = PROFILES[name]
        sampled_profiles.append(name)
        offered_paper_units += profile.paper_slice_units
        offered_compute += profile.compute_slices
        offered_memory += profile.memory_slices
    else:
        raise RuntimeError("Workload generation did not reach the requested offered-load budget")

    # Durations are sampled only after T is known, uniformly from integer slots [1, T].
    T = len(sampled_profiles)
    return [
        Job(f"paper-{scenario}-{seed}-{i:05d}", float(i) * arrival_gap, name, float(rng.randint(1, T)))
        for i, name in enumerate(sampled_profiles)
    ]



def generate_regime_shift_workload(
    *,
    count: int,
    before_scenario: str = "skew_small",
    after_scenario: str = "skew_big",
    shift_fraction: float = 0.5,
    seed: int = 42,
    mean_interarrival: float = 1.0,
    min_duration: int = 1,
    max_duration: int = 12,
) -> list[Job]:
    """Generate a piecewise-stationary workload with one profile-mix shift.

    Jobs before ``floor(count * shift_fraction)`` are sampled from
    ``before_scenario``; later jobs are sampled from ``after_scenario``.
    Arrivals remain a Poisson process with the requested mean inter-arrival
    time. Only the synthetic workload generator knows the switch point; the
    online scheduler receives past arrivals through SchedulingContext.

    This is a controlled stress test for adaptation, not a production trace.
    """
    if count < 0:
        raise ValueError("count must be >= 0")
    valid = set(MFI_WORKLOAD_DISTRIBUTIONS)
    if before_scenario not in valid or after_scenario not in valid:
        raise ValueError(f"Scenarios must be chosen from {sorted(valid)}")
    if not 0.0 < shift_fraction < 1.0:
        raise ValueError("shift_fraction must be strictly between 0 and 1")
    if mean_interarrival <= 0:
        raise ValueError("mean_interarrival must be > 0")
    if min_duration < 1 or max_duration < min_duration:
        raise ValueError("Invalid duration bounds")

    rng = random.Random(seed)
    split_at = int(count * shift_fraction)
    time = 0.0
    jobs: list[Job] = []
    for index in range(count):
        if index:
            time += rng.expovariate(1.0 / mean_interarrival)
        scenario = before_scenario if index < split_at else after_scenario
        distribution = MFI_WORKLOAD_DISTRIBUTIONS[scenario]
        names = list(PROFILE_ORDER)
        weights = [distribution[name] for name in names]
        profile_name = rng.choices(names, weights=weights, k=1)[0]
        duration = float(rng.randint(min_duration, max_duration))
        jobs.append(Job(f"shift-{index:05d}", time, profile_name, duration))
    return jobs
