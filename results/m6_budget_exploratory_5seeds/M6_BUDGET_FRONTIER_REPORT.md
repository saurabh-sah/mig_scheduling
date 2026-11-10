# M6 Budget-Frontier Analysis

## Purpose

Compare MFI, fixed-budget lookahead at several rollout counts, and adaptive lookahead with several maximum rollout budgets on matched synthetic workloads.
This is an exploratory simulator study; it is not a hardware benchmark or exact replication of the MFI paper.

## Policy summary

| Policy | Mean acceptance | Mean decision time (µs) | Mean total decision time (ms) | Mean rollout evaluations / arrival | Mean placement-change fraction vs MFI |
|---|---:|---:|---:|---:|---:|
| adaptive_h2_max2 | 0.703 | 193.293 | 7.732 | 0.95 | 0.002 |
| adaptive_h2_max4 | 0.703 | 349.417 | 13.977 | 1.97 | 0.006 |
| adaptive_h2_max8 | 0.703 | 669.077 | 26.763 | 3.99 | 0.012 |
| first_fit | 0.704 | 8.944 | 0.358 | 0.00 | 0.000 |
| fixed_h2_r2 | 0.706 | 1048.259 | 41.930 | 6.34 | 0.017 |
| fixed_h2_r4 | 0.702 | 2013.290 | 80.532 | 12.51 | 0.032 |
| fixed_h2_r8 | 0.703 | 3975.891 | 159.036 | 25.35 | 0.042 |
| mfi | 0.703 | 30.679 | 1.227 | 0.00 | 0.000 |

## Paired differences versus MFI

| Policy | Mean acceptance delta | Approx. 95% interval | Positive / zero / negative traces |
|---|---:|---:|---:|
| adaptive_h2_max2 | +0.0000 | [+0.0000, +0.0000] | 0 / 30 / 0 |
| adaptive_h2_max4 | +0.0000 | [-0.0023, +0.0023] | 1 / 28 / 1 |
| adaptive_h2_max8 | +0.0008 | [-0.0008, +0.0025] | 1 / 29 / 0 |
| first_fit | +0.0017 | [-0.0096, +0.0129] | 11 / 7 / 12 |
| fixed_h2_r2 | +0.0033 | [-0.0018, +0.0084] | 3 / 26 / 1 |
| fixed_h2_r4 | -0.0008 | [-0.0088, +0.0071] | 3 / 24 / 3 |
| fixed_h2_r8 | +0.0008 | [-0.0029, +0.0045] | 3 / 25 / 2 |

## Interpretation rules

- All policies in a scenario/seed pair receive the same generated job trace.
- The reported intervals are normal-approximation descriptive intervals. They are exploratory; use a paired t interval or paired bootstrap for final claims, with enough independent seeds.
- Measured wall-clock time includes Python implementation and host-machine effects. It is useful for within-run comparisons, not a deployment-grade latency guarantee.
- A higher acceptance rate is not enough by itself; compare gains with decision-time overhead and the number of decisions that actually changed.
- `adaptive_max2/4/8` controls a maximum rollout count, but the adaptive policy may fallback and thus use less; inspect rollout and fallback counts.
- Do not tune on the same seeds used for final conclusions. Reserve held-out seeds after selecting parameters.

## Artifacts

- `budget_frontier_runs.csv`: all raw matched runs.
- `policy_budget_summary.csv`: overall descriptive summary.
- `paired_vs_mfi.csv`: trace-paired differences.
- `paired_summary_vs_mfi.csv`: paired aggregate summary.
- `scenario_budget_summary.csv`: scenario-specific behaviour.
- Generated acceptance-versus-cost and paired-difference plots.
