"""Discrete-event simulator with dual compute/memory state and arrival snapshots.

This is a research simulator using the profile/placement abstraction in MFI,
not a hardware/driver emulator. Jobs are admitted or rejected immediately;
rejected jobs are not queued or retried, matching MFI's evaluation assumption.
"""

from dataclasses import dataclass
import json
from heapq import heappush, heappop
from itertools import count
from time import perf_counter_ns
from collections import defaultdict
from typing import Any

from .profiles import PROFILES, COMPUTE_SLICE_COUNT, MEMORY_SLICE_COUNT, MigProfile
from .schedulers import Scheduler
from .context import ArrivalObservation, RunningJobObservation, SchedulingContext
from .state import GPUState
from .fragmentation import fragmentation_score


@dataclass(frozen=True, slots=True)
class Job:
    job_id: str
    arrival_time: float
    profile_name: str
    duration: float

    def __post_init__(self) -> None:
        if not self.job_id:
            raise ValueError("job_id cannot be empty")
        if self.arrival_time < 0:
            raise ValueError("arrival_time must be >= 0")
        if self.duration <= 0:
            raise ValueError("duration must be > 0")
        if self.profile_name not in PROFILES:
            raise ValueError(f"Unknown MIG profile: {self.profile_name}")

    @property
    def profile(self) -> MigProfile:
        return PROFILES[self.profile_name]


@dataclass
class SimulationResult:
    policy: str
    gpu_count: int
    submitted: int
    accepted: int
    rejected: int
    acceptance_rate: float
    total_decision_time_ns: int
    mean_decision_time_ns: float
    final_fragmentation_score: int
    mean_time_weighted_fragmentation_score: float
    peak_fragmentation_score: int
    job_results: list[dict[str, Any]]
    event_log: list[str]
    snapshots: list[dict[str, Any]]

    def summary(self) -> dict[str, Any]:
        lookahead_rows = [row for row in self.job_results if row.get("lookahead_value") is not None]
        lookahead_rollout_rows = [row for row in self.job_results if row.get("total_rollouts") is not None]
        adaptive_rows = [row for row in self.job_results if row.get("adaptive_status") is not None]
        adaptive_lookahead_rows = [row for row in adaptive_rows if row.get("adaptive_status") == "lookahead"]
        adaptive_fallback_rows = [row for row in adaptive_rows if row.get("adaptive_status") == "mfi_fallback"]
        reliability_values = [float(row["forecast_reliability"]) for row in adaptive_rows
                              if row.get("forecast_reliability") is not None]
        adaptive_stop_reasons = [row.get("adaptive_stop_reason") for row in adaptive_lookahead_rows]
        return {
            "policy": self.policy,
            "gpu_count": self.gpu_count,
            "submitted": self.submitted,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "acceptance_rate": self.acceptance_rate,
            "total_decision_time_ns": self.total_decision_time_ns,
            "mean_decision_time_ns": self.mean_decision_time_ns,
            "final_fragmentation_score": self.final_fragmentation_score,
            "mean_time_weighted_fragmentation_score": self.mean_time_weighted_fragmentation_score,
            "peak_fragmentation_score": self.peak_fragmentation_score,
            "mean_arrival_snapshot_compute_utilization": _mean(s["compute_utilization"] for s in self.snapshots),
            "mean_arrival_snapshot_memory_utilization": _mean(s["memory_utilization"] for s in self.snapshots),
            "mean_arrival_snapshot_active_gpus": _mean(s["active_gpus"] for s in self.snapshots),
            "mean_lookahead_value": _mean(row["lookahead_value"] for row in lookahead_rows) if lookahead_rows else None,
            "mean_lookahead_rollouts_per_decision": _mean(
                row["total_rollouts"] for row in lookahead_rollout_rows
            ) if lookahead_rollout_rows else None,
            "lookahead_decisions": sum(
                row.get("lookahead_value") is not None for row in self.job_results
            ),
            "adaptive_lookahead_decisions": len(adaptive_lookahead_rows),
            "adaptive_mfi_fallback_decisions": len(adaptive_fallback_rows),
            "adaptive_no_feasible_decisions": sum(
                row.get("adaptive_status") == "no_feasible_placement" for row in adaptive_rows
            ),
            "adaptive_mean_forecast_reliability": _mean(reliability_values) if reliability_values else None,
            "adaptive_mean_total_rollouts": _mean(
                row["total_rollouts"] for row in adaptive_rows if row.get("total_rollouts") is not None
            ) if adaptive_rows else None,
            "adaptive_confidence_stops": sum(reason == "paired_confidence_separated" for reason in adaptive_stop_reasons),
            "adaptive_budget_stops": sum(reason == "max_rollout_budget" for reason in adaptive_stop_reasons),
        }


def _mean(values) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


class DiscreteEventSimulator:
    """Event-driven simulator for homogeneous A100-80GB profile abstractions."""

    def __init__(self, gpu_count: int, scheduler: Scheduler):
        if gpu_count < 1:
            raise ValueError("gpu_count must be at least 1")
        self.gpu_count = gpu_count
        self.gpus = [GPUState(gpu_id=i) for i in range(gpu_count)]
        self.scheduler = scheduler

    def run(self, jobs: list[Job]) -> SimulationResult:
        jobs = list(jobs)
        if len({job.job_id for job in jobs}) != len(jobs):
            raise ValueError("Job identifiers must be unique")

        # Event format: (time, priority, sequence, type, job_id).
        # Completion events run before arrivals at the same timestamp.
        event_heap: list[tuple[float, int, int, str, str]] = []
        sequence = count()
        jobs_by_id = {job.job_id: job for job in jobs}
        for job in jobs:
            heappush(event_heap, (job.arrival_time, 1, next(sequence), "arrival", job.job_id))

        accepted_ids: set[str] = set()
        rejected_ids: set[str] = set()
        job_results: list[dict[str, Any]] = []
        event_log: list[str] = []
        snapshots: list[dict[str, Any]] = []
        decision_time_ns_total = 0
        current_time = 0.0
        first_event_time: float | None = None
        last_event_time: float | None = None
        fragmentation_area = 0.0
        peak_fragmentation = 0
        cumulative_offered_slice_units = 0
        arrivals_seen = 0

        # Only completed durations are exposed to prediction-aware schedulers.
        # Active jobs' true durations/completion times remain private to the
        # simulator so lookahead cannot accidentally receive future information.
        prior_arrivals: list[ArrivalObservation] = []
        completed_durations: dict[str, list[float]] = defaultdict(list)
        active_job_observations: dict[str, RunningJobObservation] = {}

        def cluster_fragmentation() -> int:
            return sum(fragmentation_score(gpu) for gpu in self.gpus)

        def snapshot(job: Job) -> None:
            comp_used = sum(gpu.used_compute_slices for gpu in self.gpus)
            mem_used = sum(gpu.used_memory_slices for gpu in self.gpus)
            frag = cluster_fragmentation()
            total_submitted = arrivals_seen
            snapshots.append({
                "time": current_time,
                "job_id": job.job_id,
                "profile": job.profile_name,
                "offered_slice_units": cumulative_offered_slice_units,
                "offered_demand_fraction": cumulative_offered_slice_units / (7 * self.gpu_count),
                "submitted_so_far": total_submitted,
                "accepted_so_far": len(accepted_ids),
                "rejected_so_far": len(rejected_ids),
                "acceptance_rate_so_far": len(accepted_ids) / total_submitted if total_submitted else 0.0,
                "active_gpus": sum(gpu.is_active for gpu in self.gpus),
                "compute_utilization": comp_used / (COMPUTE_SLICE_COUNT * self.gpu_count),
                "memory_utilization": mem_used / (MEMORY_SLICE_COUNT * self.gpu_count),
                "cluster_fragmentation_score": frag,
                "mean_gpu_fragmentation_score": frag / self.gpu_count,
            })

        while event_heap:
            event_time, _priority, _seq, event_type, job_id = heappop(event_heap)
            current_time = event_time
            if first_event_time is None:
                first_event_time = event_time
            if last_event_time is not None:
                fragmentation_area += cluster_fragmentation() * (event_time - last_event_time)
            last_event_time = event_time

            if event_type == "arrival":
                job = jobs_by_id[job_id]
                arrivals_seen += 1
                cumulative_offered_slice_units += job.profile.paper_slice_units
                context = SchedulingContext(
                    current_time=current_time,
                    past_arrivals=tuple(prior_arrivals),
                    completed_durations_by_profile={
                        name: tuple(values) for name, values in completed_durations.items()
                    },
                    running_jobs=tuple(active_job_observations.values()),
                )
                decision_started = perf_counter_ns()
                context_chooser = getattr(self.scheduler, "choose_with_context", None)
                if callable(context_chooser):
                    decision = context_chooser(self.gpus, job.profile, context)
                else:
                    decision = self.scheduler.choose(self.gpus, job.profile)
                # Measure the complete scheduling call externally, including a
                # failed admission (where no PlacementDecision is returned).
                decision_time_ns = perf_counter_ns() - decision_started
                decision_time_ns_total += decision_time_ns

                if decision is None:
                    rejected_ids.add(job_id)
                    is_lookahead_policy = self.scheduler.name in {"fixed_lookahead", "adaptive_lookahead"}
                    job_results.append({
                        "job_id": job_id, "arrival_time": job.arrival_time,
                        "profile": job.profile_name, "duration": job.duration,
                        "status": "rejected", "gpu_id": None, "start": None,
                        "decision_time_ns": decision_time_ns, "delta_fragmentation": None,
                        "lookahead_value": None, "lookahead_std": None,
                        "lookahead_horizon": None, "candidate_count": 0 if is_lookahead_policy else None,
                        "rollouts_per_candidate": 0 if is_lookahead_policy else None,
                        "total_rollouts": 0 if is_lookahead_policy else None,
                        "mfi_gpu_id": None, "mfi_start": None,
                        "candidate_scores_json": "[]",
                        "adaptive_status": "no_feasible_placement" if self.scheduler.name == "adaptive_lookahead" else None,
                        "fallback_reason": "no_feasible_placement" if self.scheduler.name == "adaptive_lookahead" else None,
                        "forecast_reliability": None, "mfi_margin": None,
                        "adaptive_stop_reason": "not_started" if self.scheduler.name == "adaptive_lookahead" else None,
                        "candidate_pool_size": None, "max_rollouts_per_candidate": None,
                    })
                    event_log.append(
                        f"t={current_time:.3f}: REJECT {job_id} ({job.profile_name}); no feasible placement"
                    )
                else:
                    gpu = self.gpus[decision.gpu_id]
                    gpu.allocate(job_id, job.profile, decision.start)
                    accepted_ids.add(job_id)
                    completion_time = current_time + job.duration
                    heappush(event_heap, (completion_time, 0, next(sequence), "completion", job_id))
                    active_job_observations[job_id] = RunningJobObservation(
                        job_id=job_id, profile_name=job.profile_name,
                        arrival_time=job.arrival_time, gpu_id=decision.gpu_id,
                    )
                    job_results.append({
                        "job_id": job.job_id, "arrival_time": job.arrival_time,
                        "profile": job.profile_name, "duration": job.duration,
                        "status": "accepted", "gpu_id": decision.gpu_id,
                        "start": decision.start, "decision_time_ns": decision_time_ns,
                        "fragmentation_before": decision.fragmentation_before,
                        "fragmentation_after": decision.fragmentation_after,
                        "delta_fragmentation": decision.delta_fragmentation,
                        "completion_time": completion_time,
                        "lookahead_value": decision.lookahead_value,
                        "lookahead_std": decision.lookahead_std,
                        "lookahead_horizon": decision.lookahead_horizon,
                        "candidate_count": decision.candidate_count,
                        "rollouts_per_candidate": decision.rollouts_per_candidate,
                        "total_rollouts": decision.total_rollouts,
                        "mfi_gpu_id": decision.mfi_gpu_id,
                        "mfi_start": decision.mfi_start,
                        "candidate_scores_json": json.dumps(decision.candidate_scores),
                        "adaptive_status": decision.adaptive_status,
                        "fallback_reason": decision.fallback_reason,
                        "forecast_reliability": decision.forecast_reliability,
                        "mfi_margin": decision.mfi_margin,
                        "adaptive_stop_reason": decision.adaptive_stop_reason,
                        "candidate_pool_size": decision.candidate_pool_size,
                        "max_rollouts_per_candidate": decision.max_rollouts_per_candidate,
                    })
                    event_log.append(
                        f"t={current_time:.3f}: ACCEPT {job_id} ({job.profile_name}) on GPU {decision.gpu_id} "
                        f"start={decision.start}; completes at t={completion_time:.3f}"
                        + (f"; ΔF={decision.delta_fragmentation}" if decision.delta_fragmentation is not None else "")
                    )
                # A request becomes observable after its scheduling decision;
                # future predictions can use its profile and arrival timestamp.
                prior_arrivals.append(ArrivalObservation(job.arrival_time, job.profile_name))
                snapshot(job)

            elif event_type == "completion":
                found = next((gpu for gpu in self.gpus if job_id in gpu.allocations), None)
                if found is None:
                    raise RuntimeError(f"Completion event has no allocation for {job_id}")
                allocation = found.release(job_id)
                running = active_job_observations.pop(job_id, None)
                if running is None:
                    raise RuntimeError(f"Missing online metadata for completed job {job_id}")
                # Duration enters the estimator's observable history only when
                # the job completes; the current active-job duration is hidden.
                completed_durations[running.profile_name].append(jobs_by_id[job_id].duration)
                event_log.append(
                    f"t={current_time:.3f}: COMPLETE {job_id} on GPU {allocation.gpu_id}; "
                    f"used={found.used_compute_slices}/{COMPUTE_SLICE_COUNT} compute, "
                    f"{found.used_memory_slices}/{MEMORY_SLICE_COUNT} memory slices"
                )
            else:
                raise ValueError(f"Unknown event type: {event_type}")

            peak_fragmentation = max(peak_fragmentation, cluster_fragmentation())

        total = len(jobs)
        accepted = len(accepted_ids)
        final_fragmentation = cluster_fragmentation()
        elapsed_time = (last_event_time - first_event_time) if first_event_time is not None and last_event_time is not None else 0.0
        time_weighted_fragmentation = fragmentation_area / elapsed_time if elapsed_time > 0 else 0.0

        return SimulationResult(
            policy=self.scheduler.name, gpu_count=len(self.gpus), submitted=total,
            accepted=accepted, rejected=len(rejected_ids),
            acceptance_rate=accepted / total if total else 0.0,
            total_decision_time_ns=decision_time_ns_total,
            mean_decision_time_ns=decision_time_ns_total / total if total else 0.0,
            final_fragmentation_score=final_fragmentation,
            mean_time_weighted_fragmentation_score=time_weighted_fragmentation,
            peak_fragmentation_score=peak_fragmentation,
            job_results=sorted(job_results, key=lambda row: (row["arrival_time"], row["job_id"])),
            event_log=event_log, snapshots=snapshots,
        )
