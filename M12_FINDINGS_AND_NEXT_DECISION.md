# M12 Findings and next research decision

## What changed

Milestone 12 addresses a high-impact issue from the previous audit: whether our implementation reproduces the baseline decision path and the numeric score example in MFI Figure 3.

It adds:

1. A Figure 3 score reconstruction test.
2. Two explicitly named MFI score variants: `mfi_all_starts` and `mfi_free_anchor_score`.
3. Paper-path BF-BI/WF-BI implementations that choose a GPU based on aggregate residuals before checking placements on that GPU.
4. Retained global-feasible BF-BI/WF-BI variants under distinct labels.
5. Matched synthetic experiments, per-job decision logs, checkpoint rows, paired bootstrap summaries, figures, and run manifests.

All tests: **101 passed**.

## Experiment A — paper-guided progressive load

- GPU model: MFI A100-80GB abstraction, separate compute and memory masks.
- Cluster: 100 GPUs.
- Workload distributions: uniform, skew-small, skew-big, bimodal.
- Repetitions: 5 per distribution.
- Policies: 8.
- Total policy runs: 160.
- Per-arrival/checkpoint rows: 1,760, including the exact 85% cumulative offered-load checkpoint.
- Workload streams are matched across policies within a scenario and seed.

### Full-trace mean acceptance, macro-averaged over the four distributions

| Policy | Mean acceptance |
|---|---:|
| Paper-path FF | 75.24% |
| Paper-path RR | 96.57% |
| Paper-path BF-BI | 79.89% |
| Paper-path WF-BI | 99.00% |
| Global-feasible BF-BI | 100.00% |
| Global-feasible WF-BI | 99.01% |
| MFI, all-legal-start score | 100.00% |
| MFI, free-anchor score | 100.00% |

These are our synthetic simulator results, not values copied from the MFI paper and not a physical GPU measurement.

### Mean acceptance at the first arrival checkpoint reaching 85% offered demand

| Workload | Paper FF | Paper RR | Paper BF-BI | Paper WF-BI | Both MFI variants |
|---|---:|---:|---:|---:|---:|
| Uniform | 80.16% | 96.80% | 80.89% | 99.22% | 100.00% |
| Skew-small | 63.53% | 96.13% | 64.69% | 98.54% | 100.00% |
| Skew-big | 86.91% | 98.31% | 94.57% | 100.00% | 100.00% |
| Bimodal | 76.36% | 97.88% | 80.61% | 99.66% | 100.00% |

## Experiment B — burst/high-demand stress test

- Cluster: 20 GPUs.
- Offered scalar demand target: 150% of nominal cluster slice units.
- Arrival gap: zero (all jobs arrive before any positive-duration workload can complete).
- Repetitions: 5 per distribution.
- Policy runs: 160.

Macro-average acceptance:

| Policy | Mean acceptance |
|---|---:|
| Paper-path FF | 63.89% |
| Paper-path RR | 73.42% |
| Paper-path BF-BI | 64.94% |
| Paper-path WF-BI | 75.04% |
| Global-feasible BF-BI | 71.24% |
| Global-feasible WF-BI | 76.64% |
| MFI, all-legal-start score | 71.03% |
| MFI, free-anchor score | 71.67% |

The free-anchor variant exceeds the all-starts variant by **0.64 percentage points** on average. Across 20 paired scenario/seed comparisons, a paired bootstrap interval is approximately **−0.81 to +1.88 percentage points**, so this small sample does not establish a statistically reliable improvement. It does show that the score interpretation can change the acceptance outcome in bursty, high-demand conditions: 13 of 20 paired runs have nonzero acceptance differences.

## Decision-relevant insight 1 — Figure 3 strongly constrains the score interpretation

For reconstructed states consistent with Figure 3:

| State | Paper's reported score | Free-anchor interpretation | All-legal-starts interpretation |
|---|---:|---:|---:|
| GPU 1 | 8 | **8** | 14 |
| GPU 2 | 16 | **16** | 23 |

The paper's reported values are reproduced only by the free-anchor interpretation among these two implementations. The paper's pseudocode does not explicitly spell out that anchor filtering, so this is a strong conformance clue—not proof of the unavailable original implementation.

## Decision-relevant insight 2 — Our old BF/WF baseline was a different, stronger policy

The earlier BF-BI/WF-BI classes searched globally across all feasible candidates. This avoids the rejection path illustrated in Figure 3. The new paper-path variants choose a GPU based on aggregate residuals first and reject if it has no feasible start. The new variants show the qualitatively expected failure under fragmentation; the global-feasible versions remain available as alternative heuristics.

## Decision-relevant insight 3 — Score differences can change placements without changing acceptance

In the 100-GPU progressive-load study, the two MFI score modes select different per-job placements/admissions for an average of about **84.0%** of jobs, while both have 100% overall acceptance. Therefore placement disagreement alone is not evidence of a performance gain. The burst stress test is more discriminative, but the sample is still exploratory.

## Limitations

- The MFI paper reports 500 independent simulations per profile distribution; our 100-GPU study uses five.
- The model uses separate compute/memory occupancy masks. The exact GPU driver semantics are not emulated.
- The paper does not specify a complete executable baseline implementation or all tie-breaking rules.
- The workload generator implements a transparent interpretation of the cumulative offered-load axis, arrival spacing, and service durations; it is not claimed to be the exact original protocol.
- The residual-resource scalarization for BF/WF remains an explicit implementation choice.
- The score reconstruction uses a state configuration inferred from Figure 3's diagram and utilization labels; the figure doesn't print every start index as machine-readable text.

## Recommendation

Use `mfi_free_anchor_score` as the primary **Figure-3-conformance candidate**, keep `mfi_all_starts` as a sensitivity variant, and use the paper-path BF-BI/WF-BI classes as the paper-oriented baselines. Keep the global-feasible variants labelled separately.

Do not claim exact replication. The next research milestone should use this frozen protocol to compare the fixed/adaptive lookahead policies against the figure-conforming MFI baseline on a high-demand stress suite, with enough independent seeds to estimate confidence intervals. It should not add another algorithm before this comparison is run.
