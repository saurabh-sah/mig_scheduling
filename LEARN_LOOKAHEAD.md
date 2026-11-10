# Learn the Fixed-Rollout Lookahead Scheduler

This guide is for understanding and modifying the implementation, not only running it.

## One-sentence distinction

- MFI asks: *Which feasible current placement produces the smallest immediate increase in fragmentation?*
- Fixed lookahead asks: *Which feasible current placement leads to the most accepted jobs over sampled future arrivals, when those future arrivals are scheduled by MFI?*

The current job is already feasible under each candidate, so the lookahead score counts **future accepted jobs**, not the current job.

## Code-reading order

### 1. `context.py`

Read `SchedulingContext`. It carries the information a scheduler could have at the current simulated time. It has past arrivals, completed-job duration history, and metadata for currently running jobs. It does not expose the future generated workload or the true completion time/duration of active jobs.

This boundary matters because letting the scheduler read future requests would produce an oracle algorithm, not an online scheduler.

### 2. `simulator.py` — the arrival handler

At each arrival, the simulator creates a `SchedulingContext` before adding the current arrival to historical observations. It invokes `choose_with_context` when a policy offers that method; older baselines continue to use `choose`.

The simulator measures the wall-clock duration around the entire scheduler call, including rejected arrivals. This is the timing value written to experiment results. Timing values vary by machine and load, so do not interpret a single run's nanoseconds as a general performance guarantee.

At completion, the simulator releases resources and then exposes that completed job's duration to the future history. Duration information is therefore revealed only when a job completes.

### 3. `FixedBudgetLookaheadScheduler.choose_with_context`

The scheduler enumerates every feasible `(GPU, start)` candidate. For each candidate it computes its immediate fragmentation change and records the MFI choice as a baseline. It then makes a fixed set of rollout plans, scores each candidate against the same plans, and selects the candidate with the highest mean number of future arrivals accepted. Exact ties are broken by the MFI fragmentation increment, GPU ID, and start index.

### 4. `_profile_probabilities`

Let `n_p` be the number of prior observed jobs requesting profile `p`, `N` be the total number of prior arrivals, `K` be the number of modeled profiles, and `eta` be the additive pseudocount. The code estimates:

\[
\hat q_p = \frac{n_p+\eta}{N+K\eta}
\]

The default `eta=1` avoids assigning zero probability to a profile that has not appeared yet. With no history, the profile distribution is uniform.

### 5. `_make_rollout_plans`

Each rollout plan samples:

- a predicted duration for the current job;
- predicted residual times for active jobs;
- `H` future inter-arrival intervals;
- each future job's profile;
- each future job's duration.

The plans are generated from the scheduler's own seeded random generator. Each candidate uses the same plans. This is a common-random-numbers design, intended to reduce noise when comparing candidate placements.

### 6. `_evaluate_candidate`

For one candidate and one rollout:

1. Deep-copy all GPU states.
2. Apply the candidate allocation to the copy.
3. Create a completion-event heap from predicted active-job residuals and the candidate's predicted duration.
4. For each future arrival, release predicted-completed jobs first.
5. Ask MFI to place the future job.
6. If feasible, allocate it and schedule its predicted completion; otherwise count no admission.
7. Return the number of accepted future jobs.

Deep-copying ensures a hypothetical candidate cannot mutate the real simulator.

### 7. Candidate diagnostics

Each selected decision records the mean and standard deviation for the chosen candidate, the number of candidates, rollouts per candidate, total rollouts, the MFI-selected placement, and candidate-level scores. In `jobs.csv`, `candidate_scores_json` stores rows of:

`(gpu_id, start, delta_fragmentation, mean_future_acceptance, rollout_std)`

Use this record to study cases where lookahead selects a different placement from MFI.

## What can go wrong?

1. **Forecast bias:** early history can be too small or a workload mix can shift suddenly.
2. **Duration-model error:** completed-job durations may not represent active-job remaining lifetimes.
3. **Monte Carlo noise:** a few rollouts can rank candidates incorrectly.
4. **Compute cost:** total rollouts equal candidate count multiplied by rollouts per candidate. A fragmented or large cluster can create many candidates.
5. **Model fidelity:** the abstract compute/memory occupancy table is not driver-validated MIG emulation.
6. **Objective mismatch:** maximizing future accepted jobs ignores possible fairness, SLO, energy, and job-value differences.

These are research questions, not details to hide. Record them and design experiments around them.

## Exercises

1. Run MFI and fixed lookahead on exactly the same generated workload. Compare `acceptance_rate` and `mean_decision_time_ns`.
2. Set horizon to 1, then 3, then 5. Do not compare only acceptance: also compare runtime and the variance recorded for the chosen score.
3. Set rollouts to 1, 4, then 16. How does run time scale? Does the chosen placement change?
4. Inspect `candidate_scores_json` for a job where `mfi_gpu_id/mfi_start` differ from `gpu_id/start`.
5. Find an example where lookahead does not improve acceptance, or is slower without benefit. Explain why.
6. Explain why knowing a running job's actual completion time would be information leakage for an online scheduler.

## Research interpretation rule

This is a fixed-rollout baseline. It is not the adaptive confidence-budgeted method described in the broader project proposal. Only implement adaptivity after repeated matched experiments establish whether fixed lookahead has enough benefit to justify its cost.
