# M12 Experimental Report — Figure 3 Conformance and Baseline Semantics

## Scope

This is a paper-guided sensitivity experiment, not an exact reproduction. It compares MFI score interpretations and two baseline semantics: GPU-first (paper-path) versus global feasible-candidate search.

- GPUs: 100
- Repetitions per distribution: 5
- Offered-demand target fraction: 1.0
- Arrival gap: 1.0
- Per-job decision rows: 34928
- Scenario distributions: uniform, skew_small, skew_big, bimodal
- Policy runs: 160
- Checkpoint rows: 1760
- Elapsed wall time: 33.9 s

## Acceptance summary

| Scenario | Policy | Mean acceptance | SD | Mean decision time (µs) |
|---|---|---:|---:|---:|
| bimodal | global_feasible_bf_bi | 1.0000 | 0.0000 | 535.25 |
| bimodal | global_feasible_wf_bi | 0.9933 | 0.0094 | 543.72 |
| bimodal | mfi_all_starts | 1.0000 | 0.0000 | 1155.53 |
| bimodal | mfi_free_anchor_score | 1.0000 | 0.0000 | 1160.34 |
| bimodal | paper_bf_bi | 0.7979 | 0.0263 | 121.59 |
| bimodal | paper_ff | 0.7536 | 0.0569 | 17.12 |
| bimodal | paper_rr | 0.9700 | 0.0084 | 15.98 |
| bimodal | paper_wf_bi | 0.9933 | 0.0094 | 116.54 |
| skew_big | global_feasible_bf_bi | 1.0000 | 0.0000 | 361.30 |
| skew_big | global_feasible_wf_bi | 1.0000 | 0.0000 | 364.72 |
| skew_big | mfi_all_starts | 1.0000 | 0.0000 | 804.28 |
| skew_big | mfi_free_anchor_score | 1.0000 | 0.0000 | 783.81 |
| skew_big | paper_bf_bi | 0.9283 | 0.0208 | 113.27 |
| skew_big | paper_ff | 0.8484 | 0.0246 | 17.36 |
| skew_big | paper_rr | 0.9722 | 0.0176 | 13.83 |
| skew_big | paper_wf_bi | 1.0000 | 0.0000 | 117.55 |
| skew_small | global_feasible_bf_bi | 1.0000 | 0.0000 | 634.25 |
| skew_small | global_feasible_wf_bi | 0.9794 | 0.0070 | 746.98 |
| skew_small | mfi_all_starts | 1.0000 | 0.0000 | 1402.14 |
| skew_small | mfi_free_anchor_score | 1.0000 | 0.0000 | 1383.25 |
| skew_small | paper_bf_bi | 0.6667 | 0.0709 | 126.66 |
| skew_small | paper_ff | 0.6245 | 0.0690 | 18.73 |
| skew_small | paper_rr | 0.9586 | 0.0187 | 17.69 |
| skew_small | paper_wf_bi | 0.9801 | 0.0076 | 146.36 |
| uniform | global_feasible_bf_bi | 1.0000 | 0.0000 | 490.56 |
| uniform | global_feasible_wf_bi | 0.9878 | 0.0108 | 589.81 |
| uniform | mfi_all_starts | 1.0000 | 0.0000 | 1101.76 |
| uniform | mfi_free_anchor_score | 1.0000 | 0.0000 | 1074.80 |
| uniform | paper_bf_bi | 0.8025 | 0.0426 | 117.22 |
| uniform | paper_ff | 0.7833 | 0.0487 | 18.99 |
| uniform | paper_rr | 0.9621 | 0.0106 | 16.28 |
| uniform | paper_wf_bi | 0.9868 | 0.0126 | 130.53 |

## Interpretation constraints

- The `mfi_free_anchor_score` mode matches the reported F(GPU1)=8 and F(GPU2)=16 values for the documented Figure 3 state reconstruction.
- The earlier `mfi_all_starts` mode gives different score values for those same states, showing that the paper's numeric example constrains the scoring interpretation.
- `paper_bf_bi`/`paper_wf_bi` select a GPU using aggregate residuals before checking that selected GPU's legal placement. The existing `global_feasible_*` policies consider feasible placements across all GPUs; these are different algorithms and must not share the same baseline label.
- The experiment is synthetic and uses operational assumptions. Do not describe it as reproducing the authors' original code or as a result on physical GPUs.

## Figure 3 score reconstruction

For the reconstructed Figure 3 states, `mfi_free_anchor_score` returns the reported scores `F(GPU1)=8` and `F(GPU2)=16`; the earlier `mfi_all_starts` interpretation returns 14 and 23. The reconstructed states and per-profile score contributions are documented in the repository-level `M12_FIGURE3_CONFORMANCE.md`. This matches the numeric example for these states but does not prove equivalence to unavailable author code.

## MFI score-variant decision effects

The two score interpretations changed an average of **84.0%** of per-job placement/admission decisions across matched scenario/seed runs. 0 of 20 paired runs had a nonzero acceptance-rate difference. In this progressive-load sample, both score modes accepted every submitted job despite choosing different placements.

