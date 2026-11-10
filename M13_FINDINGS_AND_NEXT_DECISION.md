# M13 Findings and Next Research Decision

## What changed

Milestone 13 makes the existing fixed-budget and adaptive-budget lookahead schedulers explicitly **score-aware**. A scheduler configured with `score_mode="free_anchor_starts"` now uses that same score interpretation for (1) ranking the current request's feasible placements, (2) selecting MFI placement decisions inside every hypothetical future rollout, and (3) falling back to a greedy MFI placement when adaptive lookahead declines to search. The previous `all_legal_starts` interpretation remains selectable and is still the default for backward compatibility.

This consistency matters: comparing lookahead variants while scoring current placements with one interpretation and future rollouts with another would confound the policy comparison.

## Experiment design

- **240 simulations**: 10 policy variants × 24 workload traces.
- **Cluster**: 4 simulated A100-profile GPUs.
- **Workloads**: uniform, skew-small, skew-big, and bimodal profile mixtures; 3 repetitions for each mixture in each of two regimes.
- **Progressive-load regime**: 100% target cumulative demand in the paper-slice-units accounting basis, inter-arrival gap 0.25.
- **Burst regime**: 150% target cumulative demand, inter-arrival gap 0.0; positive-duration jobs arrive before any completion can free capacity.
- **Fixed lookahead**: horizon 2, 3 Monte Carlo rollouts per feasible current placement.
- **Adaptive lookahead**: horizon 2, 2–4 rollouts, batch size 2, at most 5 candidate placements. The profile-stability and ambiguity gates remain heuristics.
- All policies see the exact same generated workload within each paired trace.

This is an exploratory simulator study, not an exact reproduction of the original paper or a production-cluster evaluation.

## Result 1 — no reliable lookahead acceptance improvement established

Mean acceptance across the 12 workload traces in each regime:

| Regime | Policy | Mean acceptance | Mean decision time |
|---|---|---:|---:|
| Progressive | `mfi_all_starts` | 95.56% | 0.037 ms |
| Progressive | `mfi_free_anchor_score` | 93.70% | 0.039 ms |
| Progressive | fixed lookahead, all-starts score | 95.56% | 2.638 ms |
| Progressive | fixed lookahead, free-anchor score | 94.17% | 2.562 ms |
| Progressive | adaptive lookahead, all-starts score | 95.56% | 0.220 ms |
| Progressive | adaptive lookahead, free-anchor score | 94.17% | 0.323 ms |
| Burst | `mfi_all_starts` | 73.28% | 0.033 ms |
| Burst | `mfi_free_anchor_score` | 73.47% | 0.034 ms |
| Burst | fixed lookahead, all-starts score | 73.28% | 2.578 ms |
| Burst | fixed lookahead, free-anchor score | 72.54% | 2.537 ms |
| Burst | adaptive lookahead, all-starts score | 73.28% | 0.148 ms |
| Burst | adaptive lookahead, free-anchor score | 73.47% | 0.227 ms |

The fixed-lookahead acceptance rate matches the corresponding greedy MFI score mode in both regimes, except the free-anchor burst case, which is lower by about 0.93 percentage points. Adaptive lookahead matches its corresponding greedy MFI acceptance in this sample. None of the paired bootstrap intervals for lookahead versus the free-anchor reference establishes a reliable positive improvement; the intervals are wide or include zero, and most paired traces are ties.

Fixed lookahead's mean decision cost is about **2.5–2.7 ms per request** versus about **0.033–0.039 ms** for greedy MFI—roughly 66–78× higher in this measured run. Adaptive lookahead is cheaper than fixed lookahead but still has a higher mean decision cost than greedy MFI, and its gates usually fall back instead of searching.

## Result 2 — score interpretation changes placements even when acceptance barely changes

Across paired traces, comparing the `all_legal_starts` and `free_anchor_starts` modes changed approximately:

| Policy family | Progressive: decision tuples changed | Burst: decision tuples changed | Mean acceptance delta (free-anchor minus all-starts) |
|---|---:|---:|---:|
| Fixed lookahead | 55.3% | 54.7% | −1.39 pp progressive; −0.74 pp burst |
| Adaptive lookahead | 55.0% | 58.0% | −1.39 pp progressive; +0.19 pp burst |

A decision change means the same job had a different `(status, gpu_id, start)` tuple. It is not necessarily an admission change. These results justify keeping both score modes explicit in research logs; they do **not** establish that either mode is universally superior.

## Result 3 — progressive load is more discriminative after increasing arrival pressure

The initial M13 trial with inter-arrival gap 1.0 admitted every request for both MFI variants, which left no acceptance headroom for the comparison. The frozen M13 run instead uses gap 0.25, while retaining the same cumulative offered-demand target. This creates a more informative stress level: acceptance varies across policies and seeds. The burst regime remains a separate simultaneous-arrival stress case.

## Recommendation

Do **not** claim that fixed or adaptive lookahead improves the MFI baseline at this stage. Current results show substantial extra computation without a robust acceptance gain. Keep the implementation for controlled ablation, but prioritize:

1. a larger paired seed study (at least 20–30 workload traces per distribution and regime);
2. reporting search/fallback frequency and decision latency jointly with acceptance;
3. held-out workload tests before tuning the adaptive gates;
4. considering a candidate-pruning or early-stop method only after the present baseline comparison is reproducible.

The score-aware implementation and all job-level outputs are retained so the current experiment can be reproduced and challenged rather than accepted as a definitive conclusion.

## Validation

- `python -m pytest -q`: **107 passed**.
- `python scripts/validate_baseline.py`: profile validation, memory-overlap regression, and randomized MFI argmin consistency all pass. The validation script explicitly says these checks test implementation consistency, not replication of published performance results.
- CLI exposes `--mfi-score-mode {all_legal_starts,free_anchor_starts}`.

## Main result files

See `results/m13_score_aware_lookahead_4gpu_3rep/` for the raw runs, per-job choices, paired bootstrap comparisons, score-mode decision sensitivity, plots, run manifest, and full experiment report.
