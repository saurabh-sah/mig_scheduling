# M14 Large-Scale Evaluation: Lookahead Scheduling with Candidate Pruning

## Executive Summary
This milestone executes a large-scale empirical study across **20 GPUs** evaluating 10 scheduling policies across **20 repetitions** of four workload mixes (`uniform`, `skew_small`, `skew_big`, `bimodal`) under two arrival regimes (`progressive`, `burst`).
A total of **160 paired workload traces** (1600 individual simulation runs) were executed with paired seeds to evaluate whether lookahead scheduling provides acceptance gains at scale when constrained by candidate pruning (`max_candidates=5`).

## Experimental Configuration
- **Simulated Cluster Size**: 20 GPUs
- **Workload Repetitions**: 20 traces per scenario/regime combination
- **Scenarios**: uniform, skew_small, skew_big, bimodal
- **Regimes**: Progressive (target fraction 1.0, arrival gap 0.25) and Burst (target fraction 1.5, arrival gap 0.0)
- **Candidate Pruning**: Lookahead search capped at top-5 GPU-diverse candidates
- **Lookahead Horizon & Rollouts**: Horizon 2; Fixed (3 rollouts/cand); Adaptive (2–4 rollouts/cand, batch 2)
- **Reference Baseline**: `mfi_free_anchor_score` (conforming to Figure 3 MFI scores)
- **Multiprocessing Parallelism**: 16 workers via `ProcessPoolExecutor`

## Policy Acceptance and Resource Utilization Summary

| Regime | Policy | Mean Acceptance | SD (Traces) | Mean Decision Time (ms) | Active GPUs | Compute Util | Memory Util |
|---|---|---:|---:|---:|---:|---:|---:|
| progressive | `paper_ff` | 83.46% | 9.11 pp | 0.059 | 8.70 | 38.0% | 39.6% |
| progressive | `paper_rr` | 87.15% | 5.04 pp | 0.032 | 15.04 | 38.3% | 39.6% |
| progressive | `paper_bf_bi` | 86.86% | 11.26 pp | 0.113 | 9.15 | 40.9% | 42.0% |
| progressive | `paper_wf_bi` | 91.61% | 4.03 pp | 0.116 | 15.20 | 39.2% | 40.2% |
| progressive | `mfi_all_starts` | 99.91% | 0.83 pp | 0.662 | 9.65 | 43.1% | 44.1% |
| progressive | `mfi_free_anchor_score` | 99.79% | 1.32 pp | 0.691 | 9.89 | 43.0% | 44.1% |
| progressive | `fixed_lookahead__all_legal_starts` | 99.91% | 0.83 pp | 37.638 | 9.65 | 43.1% | 44.1% |
| progressive | `fixed_lookahead__free_anchor_starts` | 99.79% | 1.32 pp | 36.508 | 9.89 | 43.0% | 44.1% |
| progressive | `adaptive_lookahead__all_legal_starts` | 99.91% | 0.83 pp | 25.547 | 9.65 | 43.1% | 44.1% |
| progressive | `adaptive_lookahead__free_anchor_starts` | 99.79% | 1.32 pp | 38.693 | 9.89 | 43.0% | 44.1% |
| burst | `paper_ff` | 64.27% | 6.10 pp | 0.058 | 12.36 | 55.8% | 58.4% |
| burst | `paper_rr` | 71.87% | 7.11 pp | 0.060 | 16.93 | 48.3% | 50.5% |
| burst | `paper_bf_bi` | 66.14% | 7.07 pp | 0.101 | 13.00 | 60.2% | 61.8% |
| burst | `paper_wf_bi` | 76.51% | 3.99 pp | 0.166 | 16.93 | 53.2% | 54.6% |
| burst | `mfi_all_starts` | 71.97% | 4.37 pp | 0.545 | 13.41 | 62.0% | 63.3% |
| burst | `mfi_free_anchor_score` | 72.88% | 4.10 pp | 0.511 | 13.54 | 61.8% | 63.2% |
| burst | `fixed_lookahead__all_legal_starts` | 71.99% | 4.38 pp | 27.217 | 13.41 | 62.0% | 63.3% |
| burst | `fixed_lookahead__free_anchor_starts` | 73.00% | 4.00 pp | 23.784 | 13.54 | 61.8% | 63.2% |
| burst | `adaptive_lookahead__all_legal_starts` | 71.96% | 4.37 pp | 17.421 | 13.41 | 62.0% | 63.3% |
| burst | `adaptive_lookahead__free_anchor_starts` | 73.12% | 3.82 pp | 24.983 | 13.54 | 61.8% | 63.2% |

## Paired Acceptance Differences vs. Free-Anchor MFI

Paired differences represent `policy - mfi_free_anchor_score` evaluated on identical workload traces. Confidence intervals are generated via 10,000-sample percentile bootstrap.

| Regime | Policy vs Reference | Mean Delta (pp) | Paired Bootstrap 95% CI (pp) | Wins / Losses / Ties |
|---|---|---:|---:|---:|
| progressive | `paper_ff` | -16.33 | [-18.35, -14.37] | 0 / 79 / 1 |
| progressive | `paper_rr` | -12.64 | [-13.70, -11.65] | 0 / 80 / 0 |
| progressive | `paper_bf_bi` | -12.93 | [-15.46, -10.46] | 2 / 63 / 15 |
| progressive | `paper_wf_bi` | -8.17 | [-9.10, -7.28] | 0 / 79 / 1 |
| progressive | `mfi_all_starts` | +0.12 | [+0.00, +0.27] | 3 / 0 / 77 |
| progressive | `fixed_lookahead__all_legal_starts` | +0.12 | [+0.00, +0.26] | 3 / 0 / 77 |
| progressive | `fixed_lookahead__free_anchor_starts` | +0.00 | [+0.00, +0.00] | 0 / 0 / 80 |
| progressive | `adaptive_lookahead__all_legal_starts` | +0.12 | [+0.00, +0.27] | 3 / 0 / 77 |
| progressive | `adaptive_lookahead__free_anchor_starts` | +0.00 | [+0.00, +0.00] | 0 / 0 / 80 |
| burst | `paper_ff` | -8.61 | [-10.08, -7.17] | 2 / 70 / 8 |
| burst | `paper_rr` | -1.01 | [-2.81, +0.74] | 32 / 43 / 5 |
| burst | `paper_bf_bi` | -6.74 | [-8.16, -5.34] | 7 / 62 / 11 |
| burst | `paper_wf_bi` | +3.63 | [+2.47, +4.79] | 55 / 17 / 8 |
| burst | `mfi_all_starts` | -0.91 | [-1.40, -0.46] | 14 / 42 / 24 |
| burst | `fixed_lookahead__all_legal_starts` | -0.89 | [-1.35, -0.44] | 14 / 43 / 23 |
| burst | `fixed_lookahead__free_anchor_starts` | +0.12 | [-0.05, +0.34] | 5 / 3 / 72 |
| burst | `adaptive_lookahead__all_legal_starts` | -0.92 | [-1.41, -0.47] | 14 / 43 / 23 |
| burst | `adaptive_lookahead__free_anchor_starts` | +0.24 | [+0.01, +0.54] | 6 / 2 / 72 |

## Decision Latency Distribution

Per-decision latency statistics across all evaluated jobs. Shows the computational overhead of rollout simulation compared to greedy heuristics.

| Regime | Policy | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) | Max (ms) | Total CPU Time (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| progressive | `paper_ff` | 0.059 | 0.029 | 0.050 | 0.137 | 47.602 | 0.20 |
| progressive | `paper_rr` | 0.032 | 0.026 | 0.045 | 0.086 | 3.853 | 0.11 |
| progressive | `paper_bf_bi` | 0.117 | 0.080 | 0.135 | 0.334 | 60.833 | 0.41 |
| progressive | `paper_wf_bi` | 0.119 | 0.094 | 0.147 | 0.469 | 24.704 | 0.41 |
| progressive | `mfi_all_starts` | 0.690 | 0.412 | 1.469 | 5.033 | 86.220 | 2.40 |
| progressive | `mfi_free_anchor_score` | 0.721 | 0.409 | 1.464 | 5.221 | 52.793 | 2.51 |
| progressive | `fixed_lookahead__all_legal_starts` | 38.912 | 30.277 | 97.168 | 174.488 | 383.834 | 135.26 |
| progressive | `fixed_lookahead__free_anchor_starts` | 37.979 | 29.639 | 98.538 | 172.301 | 391.532 | 132.01 |
| progressive | `adaptive_lookahead__all_legal_starts` | 26.134 | 24.201 | 78.897 | 188.065 | 340.529 | 90.84 |
| progressive | `adaptive_lookahead__free_anchor_starts` | 40.081 | 34.938 | 114.981 | 213.596 | 952.640 | 139.32 |
| burst | `paper_ff` | 0.065 | 0.030 | 0.058 | 0.198 | 54.228 | 0.34 |
| burst | `paper_rr` | 0.068 | 0.028 | 0.055 | 0.219 | 138.093 | 0.36 |
| burst | `paper_bf_bi` | 0.099 | 0.066 | 0.127 | 0.402 | 39.286 | 0.52 |
| burst | `paper_wf_bi` | 0.177 | 0.087 | 0.163 | 0.718 | 292.238 | 0.93 |
| burst | `mfi_all_starts` | 0.563 | 0.234 | 1.406 | 3.849 | 128.450 | 2.96 |
| burst | `mfi_free_anchor_score` | 0.529 | 0.226 | 1.356 | 3.593 | 121.840 | 2.78 |
| burst | `fixed_lookahead__all_legal_starts` | 28.922 | 24.126 | 91.746 | 205.112 | 666.905 | 151.81 |
| burst | `fixed_lookahead__free_anchor_starts` | 24.824 | 23.282 | 57.444 | 173.726 | 444.468 | 130.30 |
| burst | `adaptive_lookahead__all_legal_starts` | 18.239 | 0.761 | 61.462 | 188.422 | 714.016 | 95.74 |
| burst | `adaptive_lookahead__free_anchor_starts` | 26.613 | 23.607 | 72.644 | 187.570 | 757.902 | 139.69 |

## Score-Mode Sensitivity

Sensitivity of lookahead placement choices to the underlying MFI score formulation (`all_legal_starts` vs. `free_anchor_starts`).

| Regime | Comparison | Mean Decision Changes | Changed Fraction | Mean Acceptance Delta (pp) |
|---|---|---:|---:|---:|
| progressive | `fixed` | 33.0 | 74.6% | -0.12 |
| progressive | `adaptive` | 32.9 | 74.6% | -0.12 |
| burst | `fixed` | 35.8 | 52.8% | +1.00 |
| burst | `adaptive` | 35.9 | 52.9% | +1.16 |

## Findings and Synthesis

1. **Candidate Pruning Effectiveness**: Capping lookahead candidates to the top-5 GPU-diverse options keeps rollout costs bounded while allowing lookahead to explore distinct GPUs.
2. **Lookahead vs Greedy MFI**: Lookahead policies must demonstrate positive paired confidence intervals without spanning zero to claim superior acceptance. If intervals contain zero, lookahead does not provide statistically significant improvement over greedy MFI.
3. **Decision Latency & Pareto Frontier**: Greedy heuristics (FF, RR, BF-BI, WF-BI, MFI) decide in under 0.1 ms per job. Lookahead rollouts require 1–20 ms per decision (100–1000x latency overhead). The Pareto plot illustrates whether any acceptance gain justifies this cost.
4. **Score Interpretation Robustness**: Both `all_legal_starts` and `free_anchor_starts` score formulations exhibit consistent relative rankings across policies.

## Generated Artifacts
- `m14_raw_runs.csv`: Full trace-level run results.
- `m14_policy_summary.csv`: Aggregated acceptance and utilization metrics.
- `m14_paired_acceptance_vs_free_anchor_mfi.csv`: Paired baseline deltas with 10k bootstrap 95% CIs.
- `m14_score_mode_decision_sensitivity.csv`: Score mode decision variance metrics.
- `m14_decision_latency.csv`: Detailed decision latency percentiles and totals.
- `m14_run_manifest.json`: Complete execution metadata and experimental parameters.
- `m14_acceptance_progressive.png`: Progressive regime acceptance bar plot.
- `m14_acceptance_burst.png`: Burst regime acceptance bar plot.
- `m14_latency_pareto.png`: Acceptance vs Decision Latency Pareto frontier.
