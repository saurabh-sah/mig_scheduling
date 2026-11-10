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
| first_fit | 0.7042 | 10.2 | 0.409 | 0.00% | 0.00 |
| mfi | 0.7025 | 31.8 | 1.273 | 0.00% | 0.00 |
| fixed_h1_r4 | 0.7042 | 1496.3 | 59.850 | 1.97% | 12.66 |
| fixed_h2_r4 | 0.7025 | 2212.5 | 88.500 | 3.27% | 12.76 |
| fixed_h3_r4 | 0.7008 | 2855.2 | 114.208 | 5.92% | 12.88 |
| fixed_h5_r4 | 0.7025 | 3782.6 | 151.302 | 8.30% | 12.65 |
| adaptive_h2_max4 | 0.7050 | 364.7 | 14.588 | 0.97% | 1.94 |
| adaptive_h3_max4 | 0.7025 | 478.6 | 19.145 | 1.08% | 2.11 |

## Paired acceptance differences vs MFI

| Policy | Mean delta (pp) | Stratified paired-bootstrap 95% interval (pp) | Traces + / = / - | Cost multiplier |
|---|---:|---:|---:|---:|
| first_fit | +0.167 | [-0.917, +1.250] | 11/7/12 | 0.3× |
| fixed_h1_r4 | +0.167 | [-0.083, +0.500] | 3/26/1 | 47.0× |
| fixed_h2_r4 | +0.000 | [-0.333, +0.333] | 2/27/1 | 69.5× |
| fixed_h3_r4 | -0.167 | [-0.833, +0.583] | 3/22/5 | 89.7× |
| fixed_h5_r4 | -0.000 | [-0.833, +0.833] | 7/14/9 | 118.8× |
| adaptive_h2_max4 | +0.250 | [+0.000, +0.500] | 3/27/0 | 11.5× |
| adaptive_h3_max4 | +0.000 | [-0.250, +0.250] | 1/28/1 | 15.0× |

## Decision-level diagnostics

| Policy | Comparable accepted placements | Scored lookahead decisions | Changed vs MFI | Mean estimated future gain over MFI candidate | Mean top-two value gap | Mean top rollout SD |
|---|---:|---:|---:|---:|---:|---:|
| fixed_h1_r4 | 845 | 845 | 2.01% | 0.0056 | 0.0357 | 0.2161 |
| fixed_h2_r4 | 843 | 843 | 3.44% | 0.0104 | 0.0468 | 0.3736 |
| fixed_h3_r4 | 841 | 841 | 6.06% | 0.0184 | 0.0599 | 0.5018 |
| fixed_h5_r4 | 843 | 843 | 8.07% | 0.0249 | 0.0684 | 0.7212 |
| adaptive_h2_max4 | 846 | 141 | 1.06% | 0.0213 | 0.0124 | 0.3900 |
| adaptive_h3_max4 | 843 | 149 | 1.19% | 0.0168 | 0.0201 | 0.5314 |

## Interpretation guardrails

- A placement change is not automatically an improvement: compare the matched trace's final acceptance count and decision cost.
- Estimated future gain is a rollout-model value difference between the selected candidate and the candidate MFI would choose. It is not a causal estimate of the realized benefit of an individual decision.
- For interpretation, compare the estimated value gap with the rollout sample standard deviation. A tiny estimated advantage relative to rollout variability is weak evidence for a placement change.
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
