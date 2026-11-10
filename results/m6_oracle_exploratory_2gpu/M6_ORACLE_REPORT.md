# M6 Exact Offline Oracle — Exploratory Report

## Scope and semantics

The oracle knows the complete fixed trace (all arrival times, profiles, and durations) and searches for the maximum number of accepted jobs under the simulator's current model. It processes arrivals in stable time order; jobs finishing at an arrival timestamp are released first; requests are accepted or rejected immediately; accepted jobs are not migrated; and no job can be delayed or retried.

**This is an offline upper bound for the current simulator model, not an implementable online scheduler and not a hardware MIG emulator.** It does not prove an optimal schedule for real GPUs.

## Run configuration

- Jobs per trace: 8
- GPU count: 2
- Scenarios: uniform, skew_small, skew_big, bimodal
- Seeds: 11, 22, 33
- Complete oracle traces: 11
- Traces that hit search budget: 1
- Maximum DP states per trace: 250000
- Wall-clock experiment time: 29.44 seconds

## Results by policy

| Policy | Mean acceptance | Mean oracle gap (jobs) | Mean decision time per trace (ms) |
|---|---:|---:|---:|
| adaptive_h2_max4 | 0.773 | 0.545 | 1.066 |
| first_fit | 0.773 | 0.545 | 0.089 |
| fixed_h2_r2 | 0.773 | 0.545 | 5.154 |
| mfi | 0.773 | 0.545 | 0.222 |

## Interpretation guardrails

- These results are exploratory and depend on the small synthetic workload traces.
- Normal-approximation intervals are included as descriptive summaries; with few traces, they are not confirmatory inference.
- Compare policies on the same trace (`trace_id`) and use paired differences for any claim.
- An oracle gap is the gap to the exact optimum under this simulator's assumptions only.
- A zero gap on tiny traces does not imply an online policy is optimal in general.
- The oracle can exploit complete future information; it is an upper bound, not a fair deployment baseline.

## Files

- `online_vs_oracle_runs.csv`: one row per online policy and trace.
- `oracle_trace_summary.csv`: exact solver runtime and DP size per trace.
- `oracle_decisions.csv`: one recovered optimal schedule per trace.
- `policy_summary.csv`: descriptive aggregate results.
- `scenario_summary.csv`: per-scenario results.
