# Milestone 4: Adaptive-Budget Lookahead Scheduler

## 1. Goal

Milestone 4 adds an **experimental adaptive search policy** on top of the Milestone 3 fixed-budget lookahead baseline. It does not establish that the adaptive policy is novel or better. Its purpose is to make the research hypothesis executable and measurable.

The hypothesis is:

> An online MIG scheduler can sometimes improve downstream workload acceptance while using less computation than fixed-budget lookahead by spending additional search only when (a) recent profile demand appears sufficiently stable and (b) MFI's top placement choices are close under its fragmentation objective.

## 2. Decision flow

1. Enumerate feasible current placements using the existing dual compute/memory model.
2. Compute each candidate's MFI fragmentation increment.
3. Identify MFI's best candidate and the gap between the two smallest fragmentation increments.
4. Fall back to MFI when there is only one candidate, insufficient workload history, an unstable recent profile distribution, or a clear MFI winner.
5. Otherwise retain a bounded shortlist of candidates. The shortlist includes the best candidate from each GPU first, then fills remaining slots in MFI order, subject to `max_candidates`.
6. Evaluate every shortlisted candidate against the same sampled futures (common random numbers), in rollout batches.
7. After `min_rollouts` per candidate, test whether the empirical best candidate is separated from every rival using paired differences and an approximate 95% Student-t interval.
8. Stop early if separated, or stop at `max_rollouts` per candidate. If the budget is exhausted without separation, select the highest sample-mean candidate; ties go to lower MFI increment, then GPU ID/start.

## 3. Forecast distribution

The predicted profile distribution is the additively smoothed empirical frequency from the most recent `profile_window` arrivals:

\[
\hat q_p(t)=\frac{n_p+\eta}{N+K\eta}
\]

where `eta` is the pseudocount, `n_p` is the count of profile `p` in the recent window, `N` is the number of observations, and `K` is the number of profile classes.

## 4. Forecast reliability heuristic

The implementation compares smoothed profile frequencies in the older and newer halves of the recent observation window. Let their total-variation distance be:

\[
TV(\hat q_{old},\hat q_{new})=\frac12\sum_p|\hat q_{old,p}-\hat q_{new,p}|
\]

The heuristic score is:

\[
R_t=\max\left(0,1-\frac{TV(\hat q_{old},\hat q_{new})}{\theta_{TV}}\right)
\]

where `theta_TV` is `stability_tv_threshold`. The score is set to zero when history is insufficient.

**Interpretation:** this is a profile-distribution stability indicator, not a calibrated probability that the forecast is correct. It does not measure out-of-sample calibration, and parameters must be tuned on validation workloads and evaluated on held-out seeds. It may mistake sampling noise for instability, and it may miss changes that occur inside the window.

## 5. Ambiguity trigger

Let the sorted candidate fragmentation increments be `delta_1 <= delta_2 <= ...`. Define:

\[
M_t=\Delta F_{(2)}-\Delta F_{(1)}
\]

Lookahead is allowed only when `M_t <= ambiguity_margin`. This is a heuristic trigger: it detects ties or near-ties in MFI's score, but it cannot guarantee that lookahead is useful whenever it fires or useless whenever it does not.

## 6. Rollout value

For candidate action `a`, the simulator estimates the number of additional future jobs accepted over horizon `H`:

\[
\hat V_H(a)=\frac1R\sum_{r=1}^{R}Y_r(a)
\]

where `Y_r(a)` is accepted future jobs in rollout `r`, and `R` is the actual number of rollouts per candidate used before stopping. The current job is not included because it is feasible under every candidate being compared.

A rollout samples future profiles, inter-arrival times, durations, and currently running jobs' residual lifetimes using only the online context and explicit fallback priors. Future generated jobs and active jobs' true completion times are not exposed to the scheduler. MFI is used for future rollout decisions.

## 7. Adaptive stopping rule

Rollout plans are generated in batches and shared across all candidate placements. For the current empirical best candidate `a*` and any alternative `b`, form paired rollout differences:

\[
D_r=Y_r(a^*)-Y_r(b)
\]

Compute the sample mean difference and standard error. Using an approximate two-sided 95% Student-t critical value, stop when the lower confidence bound on each paired mean difference exceeds `minimum_effect`. The implementation uses common random numbers to reduce the variance of these paired comparisons.

This interval is an **engineering stopping heuristic**, not a formal guarantee of correct selection: repeated interim checks, selection of the current best from the same data, discrete bounded outcomes, and multiple candidate comparisons mean its nominal 95% coverage should not be interpreted literally. The final paper must evaluate its false-stop behaviour or replace it with a more carefully calibrated sequential procedure.

## 8. Computation bound

At most `max_candidates` placements are evaluated, with at most `max_rollouts` rollout samples per candidate. Therefore the number of candidate-rollout evaluations per scheduling decision is bounded by:

\[
B_{max}=C_{shortlist}\,R_{max}\le C_{max}R_{max}
\]

The adaptive scheduler records actual rollout count and end-to-end scheduling runtime. Runtime is measured by the simulator around the complete scheduler call; internal `decision_time_ns` is also populated for decision diagnostics.

## 9. Important baselines

- **MFI:** min-increment fragmentation heuristic, used as baseline and fallback.
- **Fixed lookahead:** evaluates a fixed number of rollouts for every candidate.
- **Adaptive lookahead:** gates search using history, a profile-distribution stability heuristic, and MFI score ambiguity; then stops in batches based on paired rollout differences.

All policies must be compared on identical workload instances. Use multiple seeds and report paired differences versus MFI.

## 10. Limitations and open questions

- The simulator's MIG profile model is an abstract implementation of the paper's profile/placement table, not a driver-validated hardware emulator.
- The stability score is heuristic and must be calibrated.
- The adaptive stop rule uses approximate confidence intervals and requires additional validation before scientific claims about error control.
- Candidate shortlisting may omit a placement with better lookahead value.
- The rollout value inherits the quality of the MFI rollout policy and the profile/duration/inter-arrival forecast.
- The method has not been shown to outperform MFI. Results may be neutral or negative.
- Literature review must verify whether closely related adaptive-budget policies have already been published before claiming novelty.
