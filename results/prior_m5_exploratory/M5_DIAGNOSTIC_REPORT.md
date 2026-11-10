# Milestone 5 — Adaptive Scheduler Diagnostic Report

## Scope

This is an exploratory diagnostic report from synthetic, matched-seed simulations. It does not establish that the proposed scheduler is generally superior to MFI, does not reproduce all details of the MFI paper's published evaluation, and is not a driver-validated MIG emulator.

Runs analysed: **270** policy/workload executions. Decision records: **16200**.

## Scenario-level results

Acceptance differences are paired by scenario and seed against MFI. Confidence intervals are exploratory. Intervals are omitted when fewer than three paired seeds are available or when all observed paired differences are identical, because a zero-width interval from a tiny synthetic sample would be misleading. A confidence interval containing zero should not be interpreted as equivalence.

| Scenario | Policy | Runs | Acceptance mean ± SD | Paired Δ vs MFI | Paired 95% CI | Mean decision time (µs) | Lookahead fraction | Different choices vs MFI |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bimodal | adaptive_conservative | 5 | 0.700 ± 0.078 | -0.003 | [-0.013, +0.006] | 360.2 | 0.07 | 0.02 |
| bimodal | adaptive_default | 5 | 0.710 ± 0.075 | +0.007 | [-0.012, +0.025] | 1159.0 | 0.13 | 0.02 |
| bimodal | adaptive_permissive | 5 | 0.707 ± 0.073 | +0.003 | [-0.006, +0.013] | 3413.8 | 0.40 | 0.03 |
| bimodal | fixed_lookahead | 5 | 0.697 ± 0.065 | -0.007 | [-0.030, +0.017] | 2412.1 | 0.70 | 0.02 |
| bimodal | mfi | 5 | 0.703 ± 0.072 | +0.000 | n/a | 36.1 | 0.00 | n/a |
| shift_big_to_small | adaptive_conservative | 5 | 0.700 ± 0.081 | +0.003 | [-0.006, +0.013] | 169.4 | 0.03 | 0.01 |
| shift_big_to_small | adaptive_default | 5 | 0.700 ± 0.081 | +0.003 | [-0.006, +0.013] | 843.2 | 0.11 | 0.03 |
| shift_big_to_small | adaptive_permissive | 5 | 0.717 ± 0.086 | +0.020 | [-0.014, +0.054] | 2951.6 | 0.41 | 0.06 |
| shift_big_to_small | fixed_lookahead | 5 | 0.707 ± 0.089 | +0.010 | [-0.001, +0.021] | 1716.0 | 0.71 | 0.04 |
| shift_big_to_small | mfi | 5 | 0.697 ± 0.085 | +0.000 | n/a | 30.9 | 0.00 | n/a |
| shift_small_to_big | adaptive_conservative | 5 | 0.677 ± 0.073 | +0.000 | n/a | 328.2 | 0.06 | 0.01 |
| shift_small_to_big | adaptive_default | 5 | 0.677 ± 0.073 | +0.000 | n/a | 767.5 | 0.10 | 0.01 |
| shift_small_to_big | adaptive_permissive | 5 | 0.670 ± 0.066 | -0.007 | [-0.025, +0.012] | 2042.6 | 0.29 | 0.02 |
| shift_small_to_big | fixed_lookahead | 5 | 0.673 ± 0.069 | -0.003 | [-0.013, +0.006] | 2112.5 | 0.67 | 0.01 |
| shift_small_to_big | mfi | 5 | 0.677 ± 0.073 | +0.000 | n/a | 29.9 | 0.00 | n/a |
| skew_big | adaptive_conservative | 5 | 0.573 ± 0.063 | +0.000 | n/a | 87.9 | 0.03 | 0.00 |
| skew_big | adaptive_default | 5 | 0.577 ± 0.060 | +0.003 | [-0.006, +0.013] | 435.4 | 0.06 | 0.01 |
| skew_big | adaptive_permissive | 5 | 0.557 ± 0.053 | -0.017 | [-0.063, +0.030] | 806.7 | 0.20 | 0.02 |
| skew_big | fixed_lookahead | 5 | 0.570 ± 0.067 | -0.003 | [-0.013, +0.006] | 763.4 | 0.57 | 0.02 |
| skew_big | mfi | 5 | 0.573 ± 0.063 | +0.000 | n/a | 21.7 | 0.00 | n/a |
| skew_small | adaptive_conservative | 5 | 0.883 ± 0.047 | +0.000 | n/a | 562.6 | 0.11 | 0.02 |
| skew_small | adaptive_default | 5 | 0.883 ± 0.047 | +0.000 | n/a | 1870.6 | 0.21 | 0.02 |
| skew_small | adaptive_permissive | 5 | 0.883 ± 0.047 | +0.000 | n/a | 4673.2 | 0.55 | 0.03 |
| skew_small | fixed_lookahead | 5 | 0.880 ± 0.062 | -0.003 | [-0.026, +0.019] | 3781.9 | 0.88 | 0.02 |
| skew_small | mfi | 5 | 0.883 ± 0.047 | +0.000 | n/a | 50.1 | 0.00 | n/a |
| uniform | adaptive_conservative | 5 | 0.697 ± 0.075 | +0.000 | n/a | 233.0 | 0.05 | 0.00 |
| uniform | adaptive_default | 5 | 0.693 ± 0.077 | -0.003 | [-0.013, +0.006] | 781.6 | 0.11 | 0.01 |
| uniform | adaptive_permissive | 5 | 0.700 ± 0.072 | +0.003 | [-0.006, +0.013] | 1893.5 | 0.33 | 0.05 |
| uniform | fixed_lookahead | 5 | 0.707 ± 0.069 | +0.010 | [-0.009, +0.029] | 1713.0 | 0.71 | 0.04 |
| uniform | mfi | 5 | 0.697 ± 0.075 | +0.000 | n/a | 29.6 | 0.00 | n/a |

## Descriptive cross-scenario comparison

This table pools the matched scenario/seed runs to provide one compact diagnostic view. It is descriptive only; the per-scenario table above should be used to identify workload-specific effects.

| Policy | Paired runs | Mean paired Δ acceptance vs MFI | Approx. 95% CI | Mean decision time (µs) | Mean decision-time ratio vs MFI | Lookahead fraction |
|---|---:|---:|---:|---:|---:|---:|
| adaptive_conservative | 30 | +0.000 | [-0.002, +0.002] | 290.2 | 8.59× | 0.06 |
| adaptive_default | 30 | +0.002 | [-0.001, +0.005] | 976.2 | 28.63× | 0.12 |
| adaptive_permissive | 30 | +0.001 | [-0.008, +0.009] | 2630.2 | 75.40× | 0.36 |
| bf_bi | 30 | -0.002 | [-0.006, +0.002] | 14.5 | 0.45× | 0.00 |
| first_fit | 30 | -0.006 | [-0.018, +0.006] | 9.5 | 0.31× | 0.00 |
| fixed_lookahead | 30 | +0.001 | [-0.005, +0.006] | 2083.2 | 59.87× | 0.71 |
| round_robin | 30 | -0.018 | [-0.034, -0.002] | 9.5 | 0.31× | 0.00 |
| wf_bi | 30 | -0.003 | [-0.018, +0.013] | 16.1 | 0.49× | 0.00 |

## How to interpret the diagnostics

- `adaptive_fallback_reasons.csv` shows why adaptive lookahead chose the MFI fallback (e.g. insufficient history, unstable forecast, or a non-ambiguous MFI decision).
- `adaptive_stop_reasons.csv` shows whether a lookahead decision stopped after the approximate paired-confidence check or exhausted its rollout budget.
- `adaptive_status_counts.csv` counts lookahead, fallback, and no-feasible-placement decisions.
- `paired_differences_vs_mfi.csv` preserves run-level paired comparisons. Inspect the distribution; do not report only the average.
- `acceptance_vs_wallclock.png` uses whole-run wall-clock time, which is environment-dependent. Use per-decision time alongside it.
- `lookahead_fraction_of_all_jobs` counts arrival decisions with at least one rollout; it applies to fixed and adaptive lookahead, not just adaptive variants.

## Limitations and next decision

1. These are controlled synthetic workloads; they are not production traces.
2. A profile-distribution stability score is a heuristic, not a calibrated probability of forecast correctness.
3. The current approximate paired t-based stopping rule needs calibration and should not be treated as formal sequential error control.
4. Results depend on the modelled MIG profile/placement abstraction and the chosen scenario and load parameters.
5. A small seed count is for debugging and direction-finding only. Use more independent workload seeds and held-out evaluation before research claims.
6. The adaptive method should only be retained if it creates a useful acceptance/overhead trade-off and differs meaningfully from existing work.

## Artifacts

- `summary.csv` — per-scenario/per-policy means, standard deviations, paired differences, and runtime metrics.
- `paired_differences_vs_mfi.csv` — matched run-level differences.
- `adaptive_fallback_reasons.csv`, `adaptive_stop_reasons.csv`, `adaptive_status_counts.csv` — adaptive diagnostics.
- `decisions.csv` from the runner — per-arrival records and candidate-score JSON.
- Plot status: Plots created with Matplotlib. Wall-clock runtime is environment-dependent and should be reported alongside decision-time measurements.

No scheduler is labelled a winner by this report. A result is a finding only after the experiment configuration has been predeclared, enough seeds have been run, and all policies have been compared on matched workloads.
