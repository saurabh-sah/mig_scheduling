# M9 — Load-Basis Sensitivity Report

## Purpose

M8 used the paper Table-I scalar Slice units to define cumulative offered load, while the simulator tracks compute and memory independently. This experiment asks whether that choice changes apparent saturation and baseline outcomes.

## Method

- Cluster size and arrival order are fixed for a given basis/scenario/seed.
- All jobs arrive at time zero, so this is a burst admission stress test; it is not the paper’s online-arrival protocol.
- `paper_slice_units`: stop when Table-I scalar units reach 7 per GPU.
- `compute`: stop when requested compute slices reach 7 per GPU.
- `memory`: stop when requested memory slices reach 8 per GPU.
- `max_normalized`: stop when either normalized compute or memory demand reaches 100% of its corresponding capacity.
- Same policy implementations: paper-style FF/RR, BF-BI, WF-BI, MFI.

## Results

| Load basis | Policy | Mean acceptance | Mean offered paper units | Mean compute demand | Mean memory demand | Peak compute occupancy | Peak memory occupancy |
|---|---|---:|---:|---:|---:|---:|---:|
| paper_slice_units | bf_bi | 1.0000 | 1.001 | 0.909 | 0.927 | 0.909 | 0.927 |
| paper_slice_units | mfi | 1.0000 | 1.001 | 0.909 | 0.927 | 0.909 | 0.927 |
| paper_slice_units | paper_ff | 0.7356 | 1.001 | 0.909 | 0.927 | 0.768 | 0.803 |
| paper_slice_units | paper_rr | 0.8138 | 1.001 | 0.909 | 0.927 | 0.606 | 0.635 |
| paper_slice_units | wf_bi | 0.8896 | 1.001 | 0.909 | 0.927 | 0.695 | 0.713 |
| compute | bf_bi | 0.9567 | 1.119 | 1.003 | 1.029 | 0.951 | 0.977 |
| compute | mfi | 0.9576 | 1.119 | 1.003 | 1.029 | 0.952 | 0.978 |
| compute | paper_ff | 0.7406 | 1.119 | 1.003 | 1.029 | 0.838 | 0.885 |
| compute | paper_rr | 0.8138 | 1.119 | 1.003 | 1.029 | 0.676 | 0.718 |
| compute | wf_bi | 0.8845 | 1.119 | 1.003 | 1.029 | 0.766 | 0.793 |
| memory | bf_bi | 0.9781 | 1.089 | 0.978 | 1.002 | 0.953 | 0.977 |
| memory | mfi | 0.9779 | 1.089 | 0.978 | 1.002 | 0.952 | 0.977 |
| memory | paper_ff | 0.7462 | 1.089 | 0.978 | 1.002 | 0.834 | 0.877 |
| memory | paper_rr | 0.8062 | 1.089 | 0.978 | 1.002 | 0.656 | 0.697 |
| memory | wf_bi | 0.8979 | 1.089 | 0.978 | 1.002 | 0.767 | 0.793 |
| max_normalized | bf_bi | 0.9716 | 1.084 | 0.988 | 1.002 | 0.953 | 0.967 |
| max_normalized | mfi | 0.9720 | 1.084 | 0.988 | 1.002 | 0.952 | 0.966 |
| max_normalized | paper_ff | 0.7347 | 1.084 | 0.988 | 1.002 | 0.840 | 0.873 |
| max_normalized | paper_rr | 0.7919 | 1.084 | 0.988 | 1.002 | 0.636 | 0.667 |
| max_normalized | wf_bi | 0.8774 | 1.084 | 0.988 | 1.002 | 0.740 | 0.756 |

## Interpretation cautions

1. This is a sensitivity experiment to expose the consequences of load-axis definitions, not a reproduction of the MFI paper.
2. The paper uses a scalar Slice column but also discusses separate compute and memory footprints; its complete source code is not available here.
3. The burst setup intentionally removes job turnover during arrivals. It is a stress test for placement/admission, not a realistic temporal model.
4. These results help diagnose a benchmark model; they do not establish a universally better scheduler.

## Figure

- `m9_acceptance_by_load_basis.png`

## Sources

- MFI paper: https://arxiv.org/abs/2511.18906
- Optimal Workload Placement on MIG: https://arxiv.org/abs/2409.06646
