"""MIG placement policies: FF, RR, BF-BI, WF-BI, MFI, and lookahead variants.

FF/RR/BF-BI/WF-BI follow the benchmark descriptions in Zambianco et al.
(arXiv:2511.18906). The Best Index baseline uses the explicit profile-specific index preference
order tabulated by Turkkan et al. (2024), which the MFI paper says it uses for
BF-BI/WF-BI. A prior milestone used a static blocking-cost heuristic; M9 adds
an exhaustive equivalence audit showing its ranking agrees with the published
preference ordering for the current placement model.
"""

from copy import deepcopy
from dataclasses import dataclass
from heapq import heappop, heappush
import random
from functools import lru_cache
from time import perf_counter_ns
from typing import Optional, Protocol

from .context import SchedulingContext

from .fragmentation import delta_fragmentation, fragmentation_score
from .profiles import PROFILES, PROFILE_ORDER, INDEX_PREFERENCE_ORDER, index_preference_rank, MigProfile
from .state import GPUState


@dataclass(frozen=True, slots=True)
class PlacementDecision:
    gpu_id: int
    start: int
    policy: str
    fragmentation_before: int | None = None
    fragmentation_after: int | None = None
    delta_fragmentation: int | None = None
    decision_time_ns: int = 0
    lookahead_value: float | None = None
    lookahead_std: float | None = None
    lookahead_horizon: int | None = None
    candidate_count: int | None = None
    rollouts_per_candidate: int | None = None
    total_rollouts: int | None = None
    mfi_gpu_id: int | None = None
    mfi_start: int | None = None
    # (gpu_id, start, delta_fragmentation, mean_future_acceptance, rollout_std)
    candidate_scores: tuple[tuple[int, int, int, float, float], ...] = ()
    # Adaptive-lookahead diagnostics. These are None for policies that do not use it.
    adaptive_status: str | None = None
    fallback_reason: str | None = None
    forecast_reliability: float | None = None
    mfi_margin: int | None = None
    adaptive_stop_reason: str | None = None
    candidate_pool_size: int | None = None
    max_rollouts_per_candidate: int | None = None


class Scheduler(Protocol):
    name: str

    def choose(self, gpus: list[GPUState], profile: MigProfile) -> Optional[PlacementDecision]: ...


@lru_cache(maxsize=None)
def _static_blocking_cost(profile_name: str, start: int) -> float:
    """Legacy M8 score retained solely for M9 equivalence auditing.

    M9 policies now use the published explicit index-preference order. This
    helper remains so an exhaustive test can verify that the prior heuristic
    induces the same preference ranking for the current profile/placement model.
    It is not used by the BF-BI/WF-BI policies after M9.
    """
    profile = PROFILES[profile_name]
    cand_c = set(profile.occupied_compute_indices(start))
    cand_m = set(profile.occupied_memory_indices(start))
    cost = 0.0
    for other_name in PROFILE_ORDER:
        other = PROFILES[other_name]
        weight = other.memory_slices / len(other.legal_starts)
        for other_start in other.legal_starts:
            other_c = set(other.occupied_compute_indices(other_start))
            other_m = set(other.occupied_memory_indices(other_start))
            if cand_c.intersection(other_c) or cand_m.intersection(other_m):
                cost += weight
    return cost


class FirstFitScheduler:
    name = "first_fit"

    def choose(self, gpus: list[GPUState], profile: MigProfile) -> Optional[PlacementDecision]:
        started = perf_counter_ns()
        for gpu in sorted(gpus, key=lambda item: item.gpu_id):
            starts = gpu.feasible_starts(profile)
            if starts:
                return PlacementDecision(
                    gpu_id=gpu.gpu_id, start=starts[0], policy=self.name,
                    decision_time_ns=perf_counter_ns() - started,
                )
        return None


class RoundRobinScheduler:
    """Sequentially spread requests across GPUs, choosing first feasible index."""
    name = "round_robin"

    def __init__(self) -> None:
        self._next_gpu_id = 0

    def choose(self, gpus: list[GPUState], profile: MigProfile) -> Optional[PlacementDecision]:
        started = perf_counter_ns()
        if not gpus:
            return None
        ordered = sorted(gpus, key=lambda x: x.gpu_id)
        offset = next((i for i, gpu in enumerate(ordered) if gpu.gpu_id >= self._next_gpu_id), 0)
        for step in range(len(ordered)):
            pos = (offset + step) % len(ordered)
            gpu = ordered[pos]
            starts = gpu.feasible_starts(profile)
            if starts:
                self._next_gpu_id = ordered[(pos + 1) % len(ordered)].gpu_id
                return PlacementDecision(
                    gpu_id=gpu.gpu_id, start=starts[0], policy=self.name,
                    decision_time_ns=perf_counter_ns() - started,
                )
        return None


class BestFitBestIndexScheduler:
    """Best-fit GPU (least aggregate residual), using the published index order."""
    name = "bf_bi"

    def choose(self, gpus: list[GPUState], profile: MigProfile) -> Optional[PlacementDecision]:
        started = perf_counter_ns()
        candidates = []
        for gpu in gpus:
            if profile.compute_slices > gpu.free_compute_slices or profile.memory_slices > gpu.free_memory_slices:
                continue
            for start in gpu.feasible_starts(profile):
                residual = (
                    (gpu.free_compute_slices - profile.compute_slices) / 7
                    + (gpu.free_memory_slices - profile.memory_slices) / 8
                )
                preference_rank = index_preference_rank(profile.name, start)
                candidates.append((residual, preference_rank, gpu.gpu_id, start))
        if not candidates:
            return None
        residual, _rank, gpu_id, start = min(candidates)
        return PlacementDecision(
            gpu_id=gpu_id, start=start, policy=self.name,
            decision_time_ns=perf_counter_ns() - started,
        )


class WorstFitBestIndexScheduler:
    """Worst-fit GPU (most residual resources), using the published index order."""
    name = "wf_bi"

    def choose(self, gpus: list[GPUState], profile: MigProfile) -> Optional[PlacementDecision]:
        started = perf_counter_ns()
        candidates = []
        for gpu in gpus:
            if profile.compute_slices > gpu.free_compute_slices or profile.memory_slices > gpu.free_memory_slices:
                continue
            for start in gpu.feasible_starts(profile):
                residual = (
                    (gpu.free_compute_slices - profile.compute_slices) / 7
                    + (gpu.free_memory_slices - profile.memory_slices) / 8
                )
                preference_rank = index_preference_rank(profile.name, start)
                # Negative residual makes max-residual placement sort first.
                candidates.append((-residual, preference_rank, gpu.gpu_id, start))
        if not candidates:
            return None
        _neg_residual, _rank, gpu_id, start = min(candidates)
        return PlacementDecision(
            gpu_id=gpu_id, start=start, policy=self.name,
            decision_time_ns=perf_counter_ns() - started,
        )


class MFIScheduler:
    """Minimum Fragmentation Increment greedy scheduler."""
    name = "mfi"

    def choose(self, gpus: list[GPUState], profile: MigProfile) -> Optional[PlacementDecision]:
        started = perf_counter_ns()
        best: tuple[int, int, int, int, int] | None = None
        # Tuple: (delta fragmentation, GPU id, start, score before, score after).
        for gpu in sorted(gpus, key=lambda item: item.gpu_id):
            if profile.compute_slices > gpu.free_compute_slices or profile.memory_slices > gpu.free_memory_slices:
                continue
            before = None
            for start in gpu.feasible_starts(profile):
                if before is None:
                    before = fragmentation_score(gpu)
                before, after, delta = delta_fragmentation(gpu, profile, start, before_score=before)
                candidate = (delta, gpu.gpu_id, start, before, after)
                if best is None or candidate[:3] < best[:3]:
                    best = candidate
        elapsed = perf_counter_ns() - started
        if best is None:
            return None
        delta, gpu_id, start, before, after = best
        return PlacementDecision(
            gpu_id=gpu_id, start=start, policy=self.name,
            fragmentation_before=before, fragmentation_after=after,
            delta_fragmentation=delta, decision_time_ns=elapsed,
        )



@dataclass(frozen=True, slots=True)
class _FutureJob:
    arrival_offset: float
    profile_name: str
    duration: float


@dataclass(frozen=True, slots=True)
class _RolloutPlan:
    """One sampled future, shared across all current placement candidates."""
    predicted_current_duration: float
    active_residuals: tuple[tuple[str, int, float], ...]
    future_jobs: tuple[_FutureJob, ...]


class FixedBudgetLookaheadScheduler:
    """Choose a current MIG placement by simulated future acceptance.

    The scheduler considers every feasible placement for the current request.
    For each placement it runs a fixed number of finite-horizon rollouts. Future
    profiles, inter-arrivals, and durations are sampled from a model estimated
    only from prior observations. Running-job completion times are not exposed
    by the simulator; their residual lifetimes are sampled from completed-job
    history (with documented fallbacks when history is sparse).

    Within each rollout, future requests are placed by MFI. The scheduler uses
    common random numbers: the same sampled rollout plans are used to score all
    candidate placements for a given current decision.

    This is a research baseline, not the final adaptive method. `choose()` is a
    safe MFI fallback for callers that do not supply online context; the
    discrete-event simulator uses `choose_with_context()`.
    """
    name = "fixed_lookahead"

    def __init__(
        self,
        *,
        horizon: int = 3,
        rollouts: int = 8,
        seed: int = 42,
        profile_pseudocount: float = 1.0,
        default_mean_interarrival: float = 0.8,
        default_duration: float = 6.0,
        score_mode: str = "all_legal_starts",
    ) -> None:
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if rollouts < 1:
            raise ValueError("rollouts must be >= 1")
        if profile_pseudocount <= 0:
            raise ValueError("profile_pseudocount must be > 0")
        if default_mean_interarrival <= 0 or default_duration <= 0:
            raise ValueError("prior inter-arrival and duration must be > 0")
        # Local import avoids a module-import cycle: mfi_score_variants uses
        # PlacementDecision, which is defined in this module.
        from .mfi_score_variants import MFIScoreVariantScheduler, SCORE_MODES
        if score_mode not in SCORE_MODES:
            raise ValueError(f"score_mode must be one of {SCORE_MODES}")
        self.horizon = horizon
        self.rollouts = rollouts
        self.seed = seed
        self.profile_pseudocount = profile_pseudocount
        self.default_mean_interarrival = default_mean_interarrival
        self.default_duration = default_duration
        self.score_mode = score_mode
        self._rng = random.Random(seed)
        self._mfi = MFIScoreVariantScheduler(score_mode=score_mode)

    def _score_gpu(self, gpu: GPUState) -> int:
        from .mfi_score_variants import fragmentation_score_variant
        return fragmentation_score_variant(gpu, mode=self.score_mode)

    def _score_occupancy(self, compute_occupied_by, memory_occupied_by) -> int:
        from .mfi_score_variants import fragmentation_score_variant_from_occupancy
        return fragmentation_score_variant_from_occupancy(
            compute_occupied_by, memory_occupied_by, mode=self.score_mode
        )

    def _candidate_placements(
        self, gpus: list[GPUState], profile: MigProfile
    ) -> list[tuple[int, int, int, int, int]]:
        """Enumerate feasible placements using this scheduler's named MFI score."""
        candidates: list[tuple[int, int, int, int, int]] = []
        for gpu in sorted(gpus, key=lambda item: item.gpu_id):
            if (profile.compute_slices > gpu.free_compute_slices
                    or profile.memory_slices > gpu.free_memory_slices):
                continue
            before = self._score_gpu(gpu)
            for start in gpu.feasible_starts(profile):
                compute, memory = gpu.hypothetical_occupancy(profile, start)
                after = self._score_occupancy(compute, memory)
                candidates.append((after - before, gpu.gpu_id, start, before, after))
        return candidates

    def choose(self, gpus: list[GPUState], profile: MigProfile) -> Optional[PlacementDecision]:
        """MFI fallback when no past-observation context is supplied."""
        started = perf_counter_ns()
        decision = self._mfi.choose(gpus, profile)
        elapsed = perf_counter_ns() - started
        if decision is None:
            return None
        return PlacementDecision(
            gpu_id=decision.gpu_id,
            start=decision.start,
            policy=self.name,
            fragmentation_before=decision.fragmentation_before,
            fragmentation_after=decision.fragmentation_after,
            delta_fragmentation=decision.delta_fragmentation,
            decision_time_ns=elapsed,
            candidate_count=None,
            rollouts_per_candidate=0,
            total_rollouts=0,
            mfi_gpu_id=decision.gpu_id,
            mfi_start=decision.start,
        )

    def choose_with_context(
        self,
        gpus: list[GPUState],
        profile: MigProfile,
        context: SchedulingContext,
    ) -> Optional[PlacementDecision]:
        started = perf_counter_ns()
        # Candidate tuple: delta-F, GPU id, start, F-before, F-after.
        # Both the current choice and all simulated future choices use the same
        # explicitly configured score interpretation.
        candidates = self._candidate_placements(gpus, profile)

        if not candidates:
            return None

        mfi_candidate = min(candidates, key=lambda item: item[:3])
        plans = self._make_rollout_plans(profile, context)
        candidate_values: list[tuple[float, float, tuple[int, int, int, int, int]]] = []

        for candidate in candidates:
            future_counts = [self._evaluate_candidate(gpus, profile, candidate, plan, i)
                             for i, plan in enumerate(plans)]
            mean_value = sum(future_counts) / len(future_counts)
            if len(future_counts) > 1:
                variance = sum((x - mean_value) ** 2 for x in future_counts) / (len(future_counts) - 1)
                std_value = variance ** 0.5
            else:
                std_value = 0.0
            candidate_values.append((mean_value, std_value, candidate))

        # Maximize downstream acceptance. Break ties using lower immediate
        # fragmentation increment, then deterministic GPU/start ordering.
        best_value = max(item[0] for item in candidate_values)
        best_candidates = [item for item in candidate_values if abs(item[0] - best_value) < 1e-12]
        chosen_value, chosen_std, chosen = min(
            best_candidates, key=lambda item: (item[2][0], item[2][1], item[2][2])
        )
        delta, gpu_id, start, before, after = chosen
        elapsed = perf_counter_ns() - started
        return PlacementDecision(
            gpu_id=gpu_id,
            start=start,
            policy=self.name,
            fragmentation_before=before,
            fragmentation_after=after,
            delta_fragmentation=delta,
            decision_time_ns=elapsed,
            lookahead_value=chosen_value,
            lookahead_std=chosen_std,
            lookahead_horizon=self.horizon,
            candidate_count=len(candidates),
            rollouts_per_candidate=self.rollouts,
            total_rollouts=len(candidates) * self.rollouts,
            mfi_gpu_id=mfi_candidate[1],
            mfi_start=mfi_candidate[2],
            candidate_scores=tuple(
                (item[2][1], item[2][2], item[2][0], item[0], item[1])
                for item in sorted(candidate_values, key=lambda row: (row[2][1], row[2][2]))
            ),
        )

    def _profile_probabilities(self, context: SchedulingContext) -> tuple[list[str], list[float]]:
        names = list(PROFILE_ORDER)
        counts = {name: 0 for name in names}
        for observation in context.past_arrivals:
            if observation.profile_name in counts:
                counts[observation.profile_name] += 1
        denom = sum(counts.values()) + self.profile_pseudocount * len(names)
        probs = [(counts[name] + self.profile_pseudocount) / denom for name in names]
        return names, probs

    def _mean_interarrival(self, context: SchedulingContext) -> float:
        times = [item.arrival_time for item in context.past_arrivals]
        gaps = [b - a for a, b in zip(times, times[1:]) if b > a]
        if gaps:
            return sum(gaps) / len(gaps)
        return self.default_mean_interarrival

    def _duration_pool(self, profile_name: str, context: SchedulingContext) -> list[float]:
        specific = list(context.completed_durations_by_profile.get(profile_name, ()))
        if specific:
            return specific
        all_durations = [d for values in context.completed_durations_by_profile.values() for d in values]
        if all_durations:
            return all_durations
        return [self.default_duration]

    def _sample_duration(self, profile_name: str, context: SchedulingContext) -> float:
        return float(self._rng.choice(self._duration_pool(profile_name, context)))

    def _sample_residual(self, profile_name: str, elapsed: float, context: SchedulingContext) -> float:
        pool = self._duration_pool(profile_name, context)
        surviving = [duration for duration in pool if duration > elapsed]
        if not surviving:
            # Sparse history must not expose the simulator's true remaining time.
            # When the observed duration samples contain no value beyond this
            # job's elapsed age, use the configured residual-life prior.
            return max(0.25, self.default_duration)
        return max(0.05, float(self._rng.choice(surviving)) - elapsed)

    def _make_rollout_plans(
        self,
        current_profile: MigProfile,
        context: SchedulingContext,
        count: int | None = None,
    ) -> list[_RolloutPlan]:
        names, probs = self._profile_probabilities(context)
        mean_interarrival = self._mean_interarrival(context)
        plans: list[_RolloutPlan] = []
        plan_count = self.rollouts if count is None else count
        if plan_count < 0:
            raise ValueError("rollout plan count must be >= 0")
        for _ in range(plan_count):
            active_residuals = []
            for running in context.running_jobs:
                elapsed = max(0.0, context.current_time - running.arrival_time)
                residual = self._sample_residual(running.profile_name, elapsed, context)
                active_residuals.append((running.job_id, running.gpu_id, residual))

            current_duration = self._sample_duration(current_profile.name, context)
            future_jobs: list[_FutureJob] = []
            offset = 0.0
            for _job_index in range(self.horizon):
                offset += max(1e-6, self._rng.expovariate(1.0 / mean_interarrival))
                future_profile = self._rng.choices(names, weights=probs, k=1)[0]
                duration = self._sample_duration(future_profile, context)
                future_jobs.append(_FutureJob(offset, future_profile, duration))

            plans.append(_RolloutPlan(
                predicted_current_duration=current_duration,
                active_residuals=tuple(active_residuals),
                future_jobs=tuple(future_jobs),
            ))
        return plans

    def _evaluate_candidate(
        self,
        current_gpus: list[GPUState],
        current_profile: MigProfile,
        candidate: tuple[int, int, int, int, int],
        plan: _RolloutPlan,
        rollout_index: int,
    ) -> int:
        # Isolated copy: evaluating a hypothetical placement never mutates the
        # actual simulator state. The plan is shared across candidate placements.
        rollout_gpus = deepcopy(current_gpus)
        _delta, gpu_id, start, _before, _after = candidate
        current_job_id = f"__lookahead_current_{rollout_index}__"
        while any(current_job_id in gpu.allocations for gpu in rollout_gpus):
            current_job_id += "_x"
        rollout_gpus[gpu_id].allocate(current_job_id, current_profile, start)

        # Completion events are expressed as offsets from the current decision.
        completions: list[tuple[float, int, int, str]] = []
        sequence = 0
        for active_id, active_gpu_id, residual in plan.active_residuals:
            heappush(completions, (residual, sequence, active_gpu_id, active_id))
            sequence += 1
        heappush(completions, (plan.predicted_current_duration, sequence, gpu_id, current_job_id))
        sequence += 1

        accepted_future = 0
        for future_index, future_job in enumerate(plan.future_jobs):
            while completions and completions[0][0] <= future_job.arrival_offset:
                _finish_time, _seq, finish_gpu_id, finish_job_id = heappop(completions)
                if finish_job_id in rollout_gpus[finish_gpu_id].allocations:
                    rollout_gpus[finish_gpu_id].release(finish_job_id)

            future_profile = PROFILES[future_job.profile_name]
            decision = self._mfi.choose(rollout_gpus, future_profile)
            if decision is None:
                continue
            future_id = f"__lookahead_future_{rollout_index}_{future_index}__"
            while any(future_id in gpu.allocations for gpu in rollout_gpus):
                future_id += "_x"
            rollout_gpus[decision.gpu_id].allocate(future_id, future_profile, decision.start)
            heappush(
                completions,
                (future_job.arrival_offset + future_job.duration,
                 sequence, decision.gpu_id, future_id),
            )
            sequence += 1
            accepted_future += 1
        return accepted_future


class AdaptiveBudgetLookaheadScheduler(FixedBudgetLookaheadScheduler):
    """Use MFI by default, spending bounded lookahead only on uncertain decisions.

    Two gates control whether search is run:
      1. Recent profile demand must look sufficiently stable.
      2. The best and second-best MFI fragmentation increments must be close.

    When both gates pass, candidates are evaluated in batches using common
    random rollout plans. Search stops early when an approximate paired 95%
    t-interval separates the empirical best candidate from every rival, or
    stops at max_rollouts per candidate. If the evidence remains tied at the
    cap, the highest-mean candidate is selected, with MFI as the tie-breaker.

    This is a research hypothesis, not a validated optimal policy. Forecast
    stability and the normal/t approximation are heuristics that must be
    stress-tested and calibrated on held-out workloads.
    """

    name = "adaptive_lookahead"

    def __init__(
        self,
        *,
        horizon: int = 3,
        min_rollouts: int = 8,
        max_rollouts: int = 32,
        batch_size: int = 4,
        seed: int = 42,
        profile_pseudocount: float = 1.0,
        default_mean_interarrival: float = 0.8,
        default_duration: float = 6.0,
        minimum_history: int = 24,
        profile_window: int = 60,
        stability_tv_threshold: float = 0.60,
        minimum_reliability: float = 0.35,
        ambiguity_margin: int = 1,
        max_candidates: int = 12,
        minimum_effect: float = 0.0,
        score_mode: str = "all_legal_starts",
    ) -> None:
        if min_rollouts < 2:
            raise ValueError("min_rollouts must be >= 2 for uncertainty estimates")
        if max_rollouts < min_rollouts:
            raise ValueError("max_rollouts must be >= min_rollouts")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if minimum_history < 4:
            raise ValueError("minimum_history must be >= 4")
        if profile_window < minimum_history:
            raise ValueError("profile_window must be >= minimum_history")
        if stability_tv_threshold <= 0:
            raise ValueError("stability_tv_threshold must be > 0")
        if not 0 <= minimum_reliability <= 1:
            raise ValueError("minimum_reliability must be in [0, 1]")
        if ambiguity_margin < 0:
            raise ValueError("ambiguity_margin must be >= 0")
        if max_candidates < 2:
            raise ValueError("max_candidates must be >= 2")
        if minimum_effect < 0:
            raise ValueError("minimum_effect must be >= 0")

        super().__init__(
            horizon=horizon,
            rollouts=max_rollouts,
            seed=seed,
            profile_pseudocount=profile_pseudocount,
            default_mean_interarrival=default_mean_interarrival,
            default_duration=default_duration,
            score_mode=score_mode,
        )
        self.min_rollouts = min_rollouts
        self.max_rollouts = max_rollouts
        self.batch_size = batch_size
        self.minimum_history = minimum_history
        self.profile_window = profile_window
        self.stability_tv_threshold = stability_tv_threshold
        self.minimum_reliability = minimum_reliability
        self.ambiguity_margin = ambiguity_margin
        self.max_candidates = max_candidates
        self.minimum_effect = minimum_effect

    def choose(self, gpus: list[GPUState], profile: MigProfile) -> Optional[PlacementDecision]:
        """Without online observations, use the safe MFI fallback."""
        started = perf_counter_ns()
        candidates = self._candidate_placements(gpus, profile)
        if not candidates:
            return None
        mfi_candidate = min(candidates, key=lambda item: item[:3])
        return self._fallback_decision(
            mfi_candidate,
            len(candidates),
            None,
            self._mfi_margin(candidates),
            "missing_online_context",
            started,
        )

    def choose_with_context(
        self,
        gpus: list[GPUState],
        profile: MigProfile,
        context: SchedulingContext,
    ) -> Optional[PlacementDecision]:
        started = perf_counter_ns()
        all_candidates = self._candidate_placements(gpus, profile)
        if not all_candidates:
            return None
        mfi_candidate = min(all_candidates, key=lambda item: item[:3])
        reliability = self.forecast_reliability(context)
        margin = self._mfi_margin(all_candidates)

        # Do not spend compute when no alternative placement exists.
        if len(all_candidates) < 2:
            return self._fallback_decision(
                mfi_candidate, len(all_candidates), reliability, margin,
                "only_one_feasible_candidate", started,
            )
        if len(context.past_arrivals) < self.minimum_history:
            return self._fallback_decision(
                mfi_candidate, len(all_candidates), reliability, margin,
                "insufficient_history", started,
            )
        if reliability < self.minimum_reliability:
            return self._fallback_decision(
                mfi_candidate, len(all_candidates), reliability, margin,
                "forecast_unstable", started,
            )
        if margin > self.ambiguity_margin:
            return self._fallback_decision(
                mfi_candidate, len(all_candidates), reliability, margin,
                "mfi_decision_not_ambiguous", started,
            )

        candidates = self._shortlist_candidates(all_candidates)
        # Ensure the exact MFI choice is present, even if the shortlist policy changes.
        mfi_key = self._candidate_key(mfi_candidate)
        if all(self._candidate_key(c) != mfi_key for c in candidates):
            candidates[-1] = mfi_candidate
            candidates = sorted(candidates, key=lambda item: item[:3])

        values: dict[tuple[int, int], list[int]] = {
            self._candidate_key(candidate): [] for candidate in candidates
        }
        candidate_by_key = {self._candidate_key(c): c for c in candidates}
        mfi_key = self._candidate_key(mfi_candidate)
        used_rollouts = 0
        stop_reason = "max_rollout_budget"

        while used_rollouts < self.max_rollouts:
            batch_count = min(self.batch_size, self.max_rollouts - used_rollouts)
            plans = self._make_rollout_plans(profile, context, count=batch_count)
            # One sampled future is evaluated against every candidate before the
            # next future is generated: common random numbers reduce pairwise noise.
            for local_index, plan in enumerate(plans):
                rollout_index = used_rollouts + local_index
                for candidate in candidates:
                    key = self._candidate_key(candidate)
                    value = self._evaluate_candidate(
                        gpus, profile, candidate, plan, rollout_index
                    )
                    values[key].append(value)
            used_rollouts += batch_count

            if used_rollouts >= self.min_rollouts and self._winner_is_separated(
                values, candidate_by_key, minimum_effect=self.minimum_effect
            ):
                stop_reason = "paired_confidence_separated"
                break

        means_and_sds = {
            key: self._mean_and_sample_sd(samples) for key, samples in values.items()
        }
        ordered = sorted(
            candidates,
            key=lambda c: (
                -means_and_sds[self._candidate_key(c)][0],
                c[0], c[1], c[2],
            ),
        )
        chosen = ordered[0]
        chosen_key = self._candidate_key(chosen)
        chosen_mean, chosen_sd = means_and_sds[chosen_key]
        delta, gpu_id, start, before, after = chosen
        elapsed = perf_counter_ns() - started

        return PlacementDecision(
            gpu_id=gpu_id,
            start=start,
            policy=self.name,
            fragmentation_before=before,
            fragmentation_after=after,
            delta_fragmentation=delta,
            decision_time_ns=elapsed,
            lookahead_value=chosen_mean,
            lookahead_std=chosen_sd,
            lookahead_horizon=self.horizon,
            candidate_count=len(candidates),
            rollouts_per_candidate=used_rollouts,
            total_rollouts=len(candidates) * used_rollouts,
            mfi_gpu_id=mfi_candidate[1],
            mfi_start=mfi_candidate[2],
            candidate_scores=tuple(
                (
                    candidate[1], candidate[2], candidate[0],
                    means_and_sds[self._candidate_key(candidate)][0],
                    means_and_sds[self._candidate_key(candidate)][1],
                )
                for candidate in sorted(candidates, key=lambda item: (item[1], item[2]))
            ),
            adaptive_status="lookahead",
            fallback_reason=None,
            forecast_reliability=reliability,
            mfi_margin=margin,
            adaptive_stop_reason=stop_reason,
            candidate_pool_size=len(all_candidates),
            max_rollouts_per_candidate=self.max_rollouts,
        )

    def _candidate_placements(
        self, gpus: list[GPUState], profile: MigProfile
    ) -> list[tuple[int, int, int, int, int]]:
        # Defined on the base class so fixed and adaptive policies cannot drift.
        return super()._candidate_placements(gpus, profile)

    @staticmethod
    def _candidate_key(candidate: tuple[int, int, int, int, int]) -> tuple[int, int]:
        return candidate[1], candidate[2]

    @staticmethod
    def _mfi_margin(candidates: list[tuple[int, int, int, int, int]]) -> int | None:
        deltas = sorted(candidate[0] for candidate in candidates)
        return deltas[1] - deltas[0] if len(deltas) >= 2 else None

    def _shortlist_candidates(
        self, candidates: list[tuple[int, int, int, int, int]]
    ) -> list[tuple[int, int, int, int, int]]:
        """Keep the MFI-ranked top candidates while diversifying across GPUs."""
        ordered = sorted(candidates, key=lambda item: item[:3])
        chosen: list[tuple[int, int, int, int, int]] = []
        chosen_keys: set[tuple[int, int]] = set()
        represented_gpus: set[int] = set()
        # Include one best candidate per GPU first, so the cap does not usually
        # spend all its slots on multiple starts of the lowest-numbered GPU.
        for candidate in ordered:
            if candidate[1] not in represented_gpus:
                chosen.append(candidate)
                chosen_keys.add(self._candidate_key(candidate))
                represented_gpus.add(candidate[1])
                if len(chosen) >= self.max_candidates:
                    return sorted(chosen, key=lambda item: item[:3])
        for candidate in ordered:
            key = self._candidate_key(candidate)
            if key not in chosen_keys:
                chosen.append(candidate)
                chosen_keys.add(key)
                if len(chosen) >= self.max_candidates:
                    break
        return sorted(chosen, key=lambda item: item[:3])

    def _fallback_decision(
        self,
        candidate: tuple[int, int, int, int, int],
        candidate_pool_size: int,
        reliability: float | None,
        margin: int | None,
        reason: str,
        started_ns: int,
    ) -> PlacementDecision:
        delta, gpu_id, start, before, after = candidate
        return PlacementDecision(
            gpu_id=gpu_id,
            start=start,
            policy=self.name,
            fragmentation_before=before,
            fragmentation_after=after,
            delta_fragmentation=delta,
            decision_time_ns=perf_counter_ns() - started_ns,
            candidate_count=0,
            rollouts_per_candidate=0,
            total_rollouts=0,
            mfi_gpu_id=gpu_id,
            mfi_start=start,
            adaptive_status="mfi_fallback",
            fallback_reason=reason,
            forecast_reliability=reliability,
            mfi_margin=margin,
            adaptive_stop_reason="not_started",
            candidate_pool_size=candidate_pool_size,
            max_rollouts_per_candidate=self.max_rollouts,
        )

    def forecast_reliability(self, context: SchedulingContext) -> float:
        """Heuristic stability score from adjacent halves of recent profile history.

        It is not a calibrated probability. It is 0 before sufficient history;
        afterwards it is one minus the normalized total-variation distance
        between smoothed profile frequencies in the older/newer halves.
        """
        observations = list(context.past_arrivals)[-self.profile_window:]
        if len(context.past_arrivals) < self.minimum_history or len(observations) < self.minimum_history:
            return 0.0
        split = len(observations) // 2
        older = observations[:split]
        newer = observations[split:]
        names = list(PROFILE_ORDER)
        older_q = self._smoothed_profile_distribution(older, names)
        newer_q = self._smoothed_profile_distribution(newer, names)
        total_variation = 0.5 * sum(abs(a - b) for a, b in zip(older_q, newer_q))
        return max(0.0, min(1.0, 1.0 - total_variation / self.stability_tv_threshold))

    def _smoothed_profile_distribution(self, observations, names: list[str]) -> list[float]:
        counts = {name: 0 for name in names}
        for item in observations:
            if item.profile_name in counts:
                counts[item.profile_name] += 1
        pseudocount = self.profile_pseudocount
        denominator = sum(counts.values()) + pseudocount * len(names)
        return [(counts[name] + pseudocount) / denominator for name in names]

    def _profile_probabilities(self, context: SchedulingContext) -> tuple[list[str], list[float]]:
        names = list(PROFILE_ORDER)
        observations = list(context.past_arrivals)[-self.profile_window:]
        return names, self._smoothed_profile_distribution(observations, names)

    @staticmethod
    def _mean_and_sample_sd(values: list[float]) -> tuple[float, float]:
        mean_value = sum(values) / len(values)
        if len(values) < 2:
            return mean_value, 0.0
        variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
        return mean_value, variance ** 0.5

    @staticmethod
    def _t_critical_95(sample_count: int) -> float:
        # Two-sided Student-t 95% critical values. Use 1.96 for n>32.
        table = {
            2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
            7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228,
            12: 2.201, 13: 2.179, 14: 2.160, 15: 2.145, 16: 2.131,
            17: 2.120, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080,
            22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
            27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042, 31: 2.040,
            32: 2.037,
        }
        return table.get(sample_count, 1.96) if sample_count >= 2 else float("inf")

    def _winner_is_separated(
        self,
        values: dict[tuple[int, int], list[int]],
        candidate_by_key: dict[tuple[int, int], tuple[int, int, int, int, int]],
        *,
        minimum_effect: float,
    ) -> bool:
        """Approximate paired-CI stopping rule using common rollout samples."""
        keys = list(values)
        sample_count = len(values[keys[0]]) if keys else 0
        if sample_count < self.min_rollouts or len(keys) < 2:
            return False
        means = {key: sum(samples) / sample_count for key, samples in values.items()}
        best_key = sorted(
            keys,
            key=lambda key: (
                -means[key], candidate_by_key[key][0],
                candidate_by_key[key][1], candidate_by_key[key][2],
            ),
        )[0]
        critical = self._t_critical_95(sample_count)
        for other_key in keys:
            if other_key == best_key:
                continue
            differences = [a - b for a, b in zip(values[best_key], values[other_key])]
            mean_difference, sample_sd = self._mean_and_sample_sd(differences)
            lower_bound = mean_difference - critical * sample_sd / (sample_count ** 0.5)
            if lower_bound <= minimum_effect:
                return False
        return True


def make_scheduler(name: str) -> Scheduler:
    normalized = name.strip().lower().replace("-", "_")
    factories = {
        "ff": FirstFitScheduler,
        "first_fit": FirstFitScheduler,
        "rr": RoundRobinScheduler,
        "round_robin": RoundRobinScheduler,
        "bf_bi": BestFitBestIndexScheduler,
        "best_fit_best_index": BestFitBestIndexScheduler,
        "wf_bi": WorstFitBestIndexScheduler,
        "worst_fit_best_index": WorstFitBestIndexScheduler,
        "mfi": MFIScheduler,
        "fixed_lookahead": FixedBudgetLookaheadScheduler,
        "lookahead": FixedBudgetLookaheadScheduler,
        "adaptive_lookahead": AdaptiveBudgetLookaheadScheduler,
        "adaptive": AdaptiveBudgetLookaheadScheduler,
        "minimum_fragmentation_increment": MFIScheduler,
    }
    if normalized not in factories:
        raise ValueError(f"Unknown scheduler {name!r}. Available: ff, rr, bf_bi, wf_bi, mfi, fixed_lookahead, adaptive_lookahead")
    return factories[normalized]()
