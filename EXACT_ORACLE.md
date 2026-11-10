# Exact Offline Acceptance Oracle (Milestone 6)

## Purpose

The oracle computes the maximum number of jobs that can be accepted for a **tiny, fixed workload trace** under the simulator's current placement model. It provides a diagnostic reference for measuring how far online policies are from the best possible admission count when the entire future trace is known.

This is not an online scheduler and is not a hardware/driver-level MIG emulator.

## Formal problem

Given an ordered trace of jobs \(W=(w_1,\ldots,w_n)\), where each job has a known arrival time, requested profile, and service duration, choose for each job either rejection or one legal GPU/placement pair. Accepted allocations occupy the corresponding compute and memory indices until their completion times. The objective is:

\[
\max \sum_{j=1}^{n} z_j
\]

where \(z_j=1\) if job \(j\) is accepted and zero otherwise.

The oracle follows the simulator's rules: jobs are processed in arrival order, completions are released before arrivals at the same timestamp, admission is immediate, there is no queue/retry, and existing jobs cannot migrate.

## Dynamic-programming state

At arrival index \(i\), the state records the subset of previously accepted jobs that are still active, along with their GPU and legal placement. Completed jobs are removed before evaluating the current arrival. The recurrence enumerates rejection plus all feasible placements:

\[
V(i,S)=\max\left\{V(i+1,S_{\mathrm{live}}),\;
1+\max_{a\in A(S,w_i)}V(i+1,T(S,a))\right\}
\]

where the second branch is omitted when no legal placement exists. The objective counts accepted jobs; the MFI delta is used only as a deterministic tie-break when recovering one of multiple optimal schedules.

The memoized state count is limited because exact search is combinatorial. The default guard is 14 jobs and 250,000 memoized states. The result reports if the state budget is exceeded rather than returning an unproven optimum.

## What can be inferred

For a trace within the state budget, the oracle is exact **under this model**. A causal online scheduler's acceptance count on the same trace should not exceed the oracle. An oracle gap is a diagnostic of the value of complete future information on that trace.

## What cannot be inferred

- This is not the optimal schedule for physical MIG hardware in general; it is optimal for the explicit compute/memory placement model used here.
- The oracle knows future arrivals and durations, which online policies do not know.
- A small-trace oracle gap does not establish that one online algorithm is better than another at scale.
- The oracle optimizes admission count only; it does not optimize fairness, makespan, energy, latency, migrations, or scheduling overhead.
- If the state budget is exceeded, no optimality claim is made for that trace.
