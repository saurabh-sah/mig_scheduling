"""Exact offline oracle for tiny workload traces.

The oracle knows the complete fixed workload trace (arrival times, requested
profiles and service durations). It maximizes the number of accepted jobs,
under the same placement and immediate-admission model used by the simulator.

It is a diagnostic upper bound, NOT a deployable online scheduler. Because the
search is combinatorial, it is intended only for tiny traces and has explicit
job/state limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter_ns
from typing import Iterable

from .fragmentation import delta_fragmentation, fragmentation_score
from .profiles import PROFILES
from .simulator import Job
from .state import GPUState


class OracleBudgetExceeded(RuntimeError):
    """Raised when exact search exceeds its configured state budget."""


@dataclass(frozen=True, slots=True)
class OracleDecision:
    job_id: str
    arrival_time: float
    profile_name: str
    accepted: bool
    gpu_id: int | None = None
    start: int | None = None


@dataclass(frozen=True, slots=True)
class OfflineOracleResult:
    submitted: int
    optimal_accepted: int
    optimal_rejected: int
    acceptance_rate: float
    decisions: tuple[OracleDecision, ...]
    states_explored: int
    transitions_considered: int
    runtime_ns: int
    objective: str = "maximize accepted jobs for a fixed, fully known workload trace"
    information: str = "offline: full future arrivals and durations are known"


ActiveAssignment = tuple[int, int, int]  # sorted-job index, gpu id, start index


def solve_offline_optimal(
    jobs: Iterable[Job],
    gpu_count: int,
    *,
    max_jobs: int = 14,
    max_states: int = 250_000,
) -> OfflineOracleResult:
    """Compute exact maximum admissions for a tiny fixed workload trace.

    Semantics match the event simulator:
    - jobs are processed in (arrival_time, input-order) order;
    - completions at time t occur before arrivals at time t;
    - requests are accepted/rejected immediately (no queue/retry);
    - allocations cannot migrate after placement;
    - all known future arrivals and service durations are available to the oracle.

    Ties between equally optimal accepting placements are broken by lower MFI
    fragmentation increment and then GPU/start order. If accepting the current
    job and rejecting it permit the same total optimum, acceptance is preferred.
    """
    if gpu_count < 1:
        raise ValueError("gpu_count must be >= 1")
    if max_jobs < 0:
        raise ValueError("max_jobs must be >= 0")
    if max_states < 1:
        raise ValueError("max_states must be >= 1")

    input_jobs = list(jobs)
    if len(input_jobs) > max_jobs:
        raise ValueError(
            f"Exact oracle limited to {max_jobs} jobs; received {len(input_jobs)}. "
            "Increase max_jobs only if the state budget is adequate."
        )
    if len({j.job_id for j in input_jobs}) != len(input_jobs):
        raise ValueError("Job identifiers must be unique")

    # Python's stable sorting preserves the input order among equal arrival times,
    # matching the simulator's insertion-order event sequence.
    ordered = sorted(enumerate(input_jobs), key=lambda pair: (pair[1].arrival_time, pair[0]))
    jobs_sorted = tuple(job for _original_index, job in ordered)
    n = len(jobs_sorted)

    start_ns = perf_counter_ns()
    states_explored = 0
    transitions_considered = 0

    def normalize(i: int, active: tuple[ActiveAssignment, ...]) -> tuple[ActiveAssignment, ...]:
        if i >= n:
            return ()
        t = jobs_sorted[i].arrival_time
        live = [
            assignment for assignment in active
            if jobs_sorted[assignment[0]].arrival_time + jobs_sorted[assignment[0]].duration > t
        ]
        return tuple(sorted(live))

    def build_gpus(active: tuple[ActiveAssignment, ...]) -> list[GPUState]:
        gpus = [GPUState(gpu_id=k) for k in range(gpu_count)]
        for job_index, gpu_id, start in active:
            job = jobs_sorted[job_index]
            gpus[gpu_id].allocate(job.job_id, job.profile, start)
        return gpus

    def candidate_placements(
        i: int, active: tuple[ActiveAssignment, ...]
    ) -> list[tuple[int, int, int, int, int]]:
        """Candidate tuple: (delta F, gpu id, start, F-before, F-after)."""
        job = jobs_sorted[i]
        gpus = build_gpus(active)
        out: list[tuple[int, int, int, int, int]] = []
        for gpu in sorted(gpus, key=lambda g: g.gpu_id):
            if (job.profile.compute_slices > gpu.free_compute_slices
                    or job.profile.memory_slices > gpu.free_memory_slices):
                continue
            before = fragmentation_score(gpu)
            for start in gpu.feasible_starts(job.profile):
                _b, after, delta = delta_fragmentation(
                    gpu, job.profile, start, before_score=before
                )
                out.append((delta, gpu.gpu_id, start, before, after))
        return out

    @lru_cache(maxsize=None)
    def dp(i: int, active: tuple[ActiveAssignment, ...]) -> int:
        nonlocal states_explored, transitions_considered
        states_explored += 1
        if states_explored > max_states:
            raise OracleBudgetExceeded(
                f"Exact oracle exceeded max_states={max_states} at job index {i}; "
                "use fewer jobs/GPUs or raise the state cap cautiously."
            )
        if i >= n:
            return 0

        # Every caller passes a canonical live-state tuple. Keeping this assertion
        # catches bugs in the recursive transition logic during development.
        if active != normalize(i, active):
            raise RuntimeError("Non-canonical active state passed into the oracle DP")

        # Branch 1: reject this request.
        reject_value = dp(i + 1, normalize(i + 1, active))
        best_value = reject_value

        # Branch 2: accept at each currently legal placement.
        for _delta, gpu_id, start, _before, _after in candidate_placements(i, active):
            transitions_considered += 1
            next_active = normalize(i + 1, active + ((i, gpu_id, start),))
            value = 1 + dp(i + 1, next_active)
            if value > best_value:
                best_value = value
        return best_value

    initial = normalize(0, ())
    optimal_accepted = dp(0, initial)

    # Recover one deterministic optimal schedule. If an accepting branch ties the
    # optimal objective, prefer acceptance, then the MFI-local tie break.
    recovered: list[OracleDecision] = []
    active = initial
    for i, job in enumerate(jobs_sorted):
        active = normalize(i, active)
        optimum_here = dp(i, active)
        reject_value = dp(i + 1, normalize(i + 1, active))
        acceptable: list[tuple[int, int, int, int, int, int]] = []
        for delta, gpu_id, start, _before, _after in candidate_placements(i, active):
            next_active = normalize(i + 1, active + ((i, gpu_id, start),))
            value = 1 + dp(i + 1, next_active)
            if value == optimum_here:
                acceptable.append((delta, gpu_id, start, value, _before, _after))
        if acceptable:
            delta, gpu_id, start, _value, _before, _after = min(acceptable)
            recovered.append(OracleDecision(
                job_id=job.job_id,
                arrival_time=job.arrival_time,
                profile_name=job.profile_name,
                accepted=True,
                gpu_id=gpu_id,
                start=start,
            ))
            active = normalize(i + 1, active + ((i, gpu_id, start),))
        else:
            # Since the DP optimum is at least reject_value and no accepting branch
            # achieves it, rejection must be optimal. If it is not, fail loudly.
            if reject_value != optimum_here:
                raise RuntimeError("Oracle reconstruction failed to find an optimal action")
            recovered.append(OracleDecision(
                job_id=job.job_id,
                arrival_time=job.arrival_time,
                profile_name=job.profile_name,
                accepted=False,
            ))
            active = normalize(i + 1, active)

    elapsed = perf_counter_ns() - start_ns
    if sum(d.accepted for d in recovered) != optimal_accepted:
        raise RuntimeError("Recovered schedule does not match the optimal objective")

    # Report decisions in the same processing order as the simulator.
    return OfflineOracleResult(
        submitted=n,
        optimal_accepted=optimal_accepted,
        optimal_rejected=n - optimal_accepted,
        acceptance_rate=(optimal_accepted / n if n else 0.0),
        decisions=tuple(recovered),
        states_explored=states_explored,
        transitions_considered=transitions_considered,
        runtime_ns=elapsed,
    )
