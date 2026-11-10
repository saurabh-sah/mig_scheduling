# M12 Experimental Report — Figure 3 Conformance and Baseline Semantics

## Scope

This is a paper-guided sensitivity experiment, not an exact reproduction. It compares MFI score interpretations and two baseline semantics: GPU-first (paper-path) versus global feasible-candidate search.

- GPUs: 20
- Repetitions per distribution: 5
- Offered-demand target fraction: 1.5
- Arrival gap: 0.0
- Per-job decision rows: 10176
- Scenario distributions: uniform, skew_small, skew_big, bimodal
- Policy runs: 160
- Checkpoint rows: 1760
- Elapsed wall time: 2.8 s

## Acceptance summary

| Scenario | Policy | Mean acceptance | SD | Mean decision time (µs) |
|---|---|---:|---:|---:|
| bimodal | global_feasible_bf_bi | 0.7164 | 0.0498 | 72.58 |
| bimodal | global_feasible_wf_bi | 0.7463 | 0.0475 | 101.04 |
| bimodal | mfi_all_starts | 0.7060 | 0.0523 | 162.02 |
| bimodal | mfi_free_anchor_score | 0.7068 | 0.0365 | 150.03 |
| bimodal | paper_bf_bi | 0.6233 | 0.0459 | 24.80 |
| bimodal | paper_ff | 0.6147 | 0.0940 | 11.92 |
| bimodal | paper_rr | 0.6849 | 0.0307 | 11.16 |
| bimodal | paper_wf_bi | 0.7457 | 0.0487 | 40.21 |
| skew_big | global_feasible_bf_bi | 0.7107 | 0.0291 | 42.96 |
| skew_big | global_feasible_wf_bi | 0.7107 | 0.0208 | 60.64 |
| skew_big | mfi_all_starts | 0.7107 | 0.0291 | 90.69 |
| skew_big | mfi_free_anchor_score | 0.7233 | 0.0137 | 87.71 |
| skew_big | paper_bf_bi | 0.7063 | 0.0307 | 20.74 |
| skew_big | paper_ff | 0.6648 | 0.0456 | 10.86 |
| skew_big | paper_rr | 0.6767 | 0.0352 | 9.34 |
| skew_big | paper_wf_bi | 0.7107 | 0.0208 | 23.00 |
| skew_small | global_feasible_bf_bi | 0.7209 | 0.0454 | 82.44 |
| skew_small | global_feasible_wf_bi | 0.8185 | 0.0146 | 113.93 |
| skew_small | mfi_all_starts | 0.7231 | 0.0430 | 160.41 |
| skew_small | mfi_free_anchor_score | 0.7295 | 0.0403 | 157.44 |
| skew_small | paper_bf_bi | 0.6356 | 0.0389 | 24.71 |
| skew_small | paper_ff | 0.6132 | 0.0390 | 11.79 |
| skew_small | paper_rr | 0.8143 | 0.0126 | 12.18 |
| skew_small | paper_wf_bi | 0.7614 | 0.0367 | 32.23 |
| uniform | global_feasible_bf_bi | 0.7014 | 0.0372 | 58.02 |
| uniform | global_feasible_wf_bi | 0.7900 | 0.0178 | 80.87 |
| uniform | mfi_all_starts | 0.7013 | 0.0348 | 121.48 |
| uniform | mfi_free_anchor_score | 0.7073 | 0.0572 | 118.89 |
| uniform | paper_bf_bi | 0.6324 | 0.0256 | 21.54 |
| uniform | paper_ff | 0.6628 | 0.0537 | 12.67 |
| uniform | paper_rr | 0.7608 | 0.0117 | 10.12 |
| uniform | paper_wf_bi | 0.7838 | 0.0196 | 27.22 |

## Interpretation constraints

- The `mfi_free_anchor_score` mode matches the reported F(GPU1)=8 and F(GPU2)=16 values for the documented Figure 3 state reconstruction.
- The earlier `mfi_all_starts` mode gives different score values for those same states, showing that the paper's numeric example constrains the scoring interpretation.
- `paper_bf_bi`/`paper_wf_bi` select a GPU using aggregate residuals before checking that selected GPU's legal placement. The existing `global_feasible_*` policies consider feasible placements across all GPUs; these are different algorithms and must not share the same baseline label.
- The experiment is synthetic and uses operational assumptions. Do not describe it as reproducing the authors' original code or as a result on physical GPUs.

## Figure 3 score reconstruction

For the reconstructed Figure 3 states, `mfi_free_anchor_score` returns the reported scores `F(GPU1)=8` and `F(GPU2)=16`; the earlier `mfi_all_starts` interpretation returns 14 and 23. The reconstructed states and per-profile score contributions are documented in the repository-level `M12_FIGURE3_CONFORMANCE.md`. This matches the numeric example for these states but does not prove equivalence to unavailable author code.

## MFI score-variant decision effects

The two score interpretations changed an average of **52.6%** of per-job placement/admission decisions across matched scenario/seed runs. 13 of 20 paired runs had a nonzero acceptance-rate difference. Across 20 matched scenario/seed pairs, free-anchor minus all-starts acceptance averaged **0.64 percentage points**; the paired bootstrap interval was approximately **-0.80 to 1.86 percentage points**, including zero. This does not establish a reliable improvement.

