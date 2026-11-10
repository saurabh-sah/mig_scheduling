# Milestone 5 Findings and Research Decision

## What was run

This exploratory run used the Milestone 5 simulator and matched workloads across nine policy variants: First Fit, Round Robin, Best Fit–Best Index, Worst Fit–Best Index, MFI, fixed-budget lookahead, and conservative/default/permissive adaptive lookahead.

- 6 synthetic workload scenarios: uniform, skew-small, skew-big, bimodal, small-to-big regime shift, and big-to-small regime shift.
- 5 requested seeds per scenario.
- 60 arrivals per workload and 3 simulated GPUs.
- 270 policy/workload executions (6 scenarios × 5 seeds × 9 policies).
- 16,200 per-arrival diagnostic rows.
- Fixed lookahead horizon: 2 future arrivals, 4 rollouts per candidate.
- Adaptive lookahead: 2–8 rollouts per candidate, batches of 2, at most 6 candidates, with conservative/default/permissive gate presets.

These are synthetic, controlled results—not production traces and not an exact reproduction of every MFI paper experiment.

## Descriptive results across the six scenarios

The values below are unweighted averages across the six scenario-level means. Scenario-level results are in `results/m5_exploratory/M5_DIAGNOSTIC_REPORT.md` and `summary.csv`.

| Policy | Mean acceptance | Mean paired acceptance difference vs MFI | Approx. pooled 95% interval | Mean decision time | Ratio to MFI decision time | Arrivals using lookahead |
|---|---:|---:|---:|---:|---:|---:|
| MFI | 0.705 | baseline | n/a | 33.0 µs | 1.00× | 0% |
| Fixed lookahead | 0.706 | +0.001 | [-0.005, +0.006] | 2,083.2 µs | 59.87× | 70.6% |
| Adaptive conservative | 0.705 | +0.000 | [-0.002, +0.002] | 290.2 µs | 8.59× | 5.9% |
| Adaptive default | 0.707 | +0.002 | [-0.001, +0.005] | 976.2 µs | 28.63× | 12.1% |
| Adaptive permissive | 0.706 | +0.001 | [-0.008, +0.009] | 2,630.2 µs | 75.40× | 36.3% |

**Interpretation:** the pilot does not show a reliable workload-acceptance improvement from fixed or adaptive lookahead. All approximate pooled intervals for those methods include zero. The interval is descriptive and must not be treated as a final confirmatory statistical result. These data are enough to question whether the present method is worth further complexity; they are not enough to conclude that lookahead can never help.

## Important observations

1. **Fixed lookahead is expensive in this implementation.** It used lookahead on about 71% of arrivals and averaged roughly 60 times MFI's measured decision time.
2. **Conservative adaptive lookahead rarely searches.** It used lookahead on about 6% of arrivals and achieved nearly the same acceptance as MFI, but still had overhead from calculating its gates and diagnostics.
3. **Default adaptive lookahead searches on about 12% of arrivals.** Its average acceptance was only 0.2 percentage points higher than MFI in this small experiment, with an interval crossing zero.
4. **Permissive adaptive lookahead searches more often but did not establish a reliable gain.** It used lookahead on about 36% of arrivals and averaged about 75 times MFI's decision time. It was worse than MFI in some individual scenarios and better in others.
5. **The workload matters.** The big-to-small regime shift showed a positive point estimate for the permissive variant, while the small-to-big shift showed a negative point estimate. These differences are exploratory and uncertain.
6. **The method changes relatively few decisions.** Inspect `fraction_choices_different_from_mfi` in `summary.csv` and individual rows in `decisions.csv` to see which selected placements differ from MFI. Invoking lookahead does not guarantee that the chosen placement will differ or improve the final outcome.

## What we can and cannot claim

### Supported by this pilot

- The pipeline runs on six controlled synthetic workload scenarios.
- Each policy in a scenario/seed pair sees the same generated jobs.
- Adaptive fallback, rollout, chosen-placement, and runtime diagnostics can be measured.
- Under the current parameters, the extra search cost is much clearer than any acceptance gain.

### Not supported by this pilot

- That adaptive lookahead is better than MFI.
- That the proposed gate or stopping rule is optimal.
- That results generalize to production GPU clusters.
- That the simulator exactly reproduces the hardware behavior or all published MFI results.
- That the research gap is novel; the latest related literature must still be checked method by method.

## Recommended next step: validate before adding another algorithm

Do **not** immediately add more heuristics or a neural model. First do a small validation milestone:

1. Review the exact MFI experimental protocol and document every difference from our simulator.
2. Add a tiny-instance offline/reference search to estimate the best possible acceptance for a short, known workload sequence. This is diagnostic ground truth—not a deployable online scheduler.
3. Increase independent seeds (for example, 20–30 per scenario), separate tuning seeds from held-out seeds, and repeat the paired evaluation.
4. Compare methods at matched decision-time budgets, not only at their default settings.
5. Determine whether the adaptive gates actually predict decisions where lookahead can change a meaningful outcome.
6. Re-check recent related work before retaining confidence-budgeted lookahead as the claimed research contribution.

If the acceptance/overhead trade-off remains weak, keep the tested simulator and pivot the research question. That is preferable to adding complexity to an unpromising method.
