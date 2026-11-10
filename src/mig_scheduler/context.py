"""Online scheduling context visible to context-aware policies.

Only information available at the current simulated time is exposed:
- past arrivals (profiles and arrival times);
- service durations of jobs that have already completed;
- currently running jobs' identities, profile, arrival and GPU, but not their true
  completion times or true durations.

This intentionally prevents a predictive scheduler from reading future jobs or
using the current/running jobs' simulator-only service durations.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ArrivalObservation:
    arrival_time: float
    profile_name: str


@dataclass(frozen=True, slots=True)
class RunningJobObservation:
    job_id: str
    profile_name: str
    arrival_time: float
    gpu_id: int


@dataclass(frozen=True, slots=True)
class SchedulingContext:
    current_time: float
    past_arrivals: tuple[ArrivalObservation, ...]
    completed_durations_by_profile: dict[str, tuple[float, ...]]
    running_jobs: tuple[RunningJobObservation, ...]
