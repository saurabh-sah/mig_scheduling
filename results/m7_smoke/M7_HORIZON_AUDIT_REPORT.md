# M7 Horizon and Decision-Centric Audit

## Scope

This is an exploratory, simulator-only analysis. It is not a physical MIG benchmark and does not constitute exact replication of all MFI paper experiments.
Each policy receives the same generated workload per scenario and seed. The bootstrap resamples traces within each workload scenario and gives scenarios equal weight.
The confidence intervals are exploratory because this study uses a limited synthetic seed set and multiple policy/horizon comparisons.

## Configuration

See `experiment_config.json`; raw traces and per-decision diagnostics are preserved in the CSV files.

## Policy summary

| Policy | Mean acceptance | Mean decision time (µs) | Total decision time (ms) | Changed placement fraction | Rollouts / arrival |
|---|---:|---:|---:|---:|---:|
| first_fit | 0.7708 | 9.4 | 0.113 | 0.00% | 0.00 |
| mfi | 0.7917 | 33.4 | 0.401 | 0.00% | 0.00 |
| fixed_h1_r2 | 0.7917 | 536.8 | 6.442 | 0.00% | 6.42 |
| fixed_h2_r2 | 0.7917 | 815.5 | 9.786 | 0.00% | 6.42 |
| adaptive_h2_max2 | 0.7917 | 36.8 | 0.442 | 0.00% | 0.00 |

## Paired acceptance differences vs MFI

| Policy | Mean delta (pp) | Stratified paired-bootstrap 95% interval (pp) | Traces + / = / - | Cost multiplier |
|---|---:|---:|---:|---:|
| first_fit | -2.083 | [-4.167, +0.000] | 0/3/1 | 0.3× |
| fixed_h1_r2 | +0.000 | [+0.000, +0.000] | 0/4/0 | 16.1× |
| fixed_h2_r2 | +0.000 | [+0.000, +0.000] | 0/4/0 | 24.4× |
| adaptive_h2_max2 | +0.000 | [+0.000, +0.000] | 0/4/0 | 1.1× |

## Decision-level diagnostics

| Policy | Comparable placements | Changed vs MFI | Mean estimated future gain over MFI candidate | Mean top-two value gap | Mean top rollout SD |
|---|---:|---:|---:|---:|---:|
| fixed_h1_r2 | 48 | 0.00% | 0.0000 | 0.0172 | 0.2047 |
| fixed_h2_r2 | 48 | 0.00% | 0.0000 | 0.0345 | 0.2047 |
| adaptive_h2_max2 | 48 | 0.00% | 0.0000 | 0.0000 | 0.0000 |

## Interpretation guardrails

- A placement change is not automatically an improvement: compare the matched trace's final acceptance count and decision cost.
- Estimated future gain is a rollout-model value difference between the selected candidate and the candidate MFI would choose. It is not a causal estimate of the realized benefit of an individual decision.
- Short horizons can fail to capture a delayed benefit; longer horizons can increase model error and compute cost.
- Timing measures Python decision-call wall time on this machine. They are not production latency guarantees.
- The current GPU state model uses separate compute and memory occupancy masks. It is a transparent software abstraction inspired by the MFI profile table, not a driver-validated physical MIG emulator.
- MFI's published evaluation uses an initially empty 100-GPU cluster, progressively increasing offered load, durations sampled in scheduling slots up to saturation time T, four fixed profile distributions, and 500 independent simulations per distribution. This M7 horizon sweep uses a different fixed-job-count Poisson-arrival protocol and must not be presented as a replication of those published numbers.
- The paper's BF-BI/WF-BI benchmark descriptions do not fully specify executable code; our code documents its deterministic index-cost interpretation and should not be portrayed as the original authors' implementation.

## Decision rule

Retain adaptive lookahead only if held-out experiments show a repeatable benefit at a defensible runtime budget. If gains remain negligible and the offline-oracle gap persists, revise the method or pivot the research question rather than adding another complexity layer.

## Plot status

Generated acceptance, runtime, paired-bootstrap, and placement-change plots.

## Artifact index

- `m7_horizon_runs.csv`: one row per policy and matched workload trace.
- `m7_decision_diagnostics.csv`: per-job placement and rollout-value details.
- `policy_summary.csv`, `paired_vs_mfi.csv`, `paired_summary_vs_mfi.csv`, `scenario_summary.csv`.
- `decision_diagnostics_summary.csv`: rollout and placement-change analysis.
