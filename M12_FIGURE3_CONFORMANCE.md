# M12 — Reconstructing MFI Figure 3

## Purpose

The MFI paper reports two numerical fragmentation scores in its Figure 3 discussion:

- GPU 1: `F(1) = 8`
- GPU 2: `F(2) = 16`

The paper explains the score as a weighted count of unfeasible MIG profile configurations, with each blocked configuration weighted by the profile's memory-slice count. The figure caption and surrounding text describe a 3g.40gb workload being rejected on one selected GPU even though a different GPU can accommodate it. Source: [MFI paper, arXiv:2511.18906](https://arxiv.org/abs/2511.18906), Section V and Figure 3.

## Reconstructed states

The figure does not print every occupied index in plain text, so this is a reconstruction consistent with its diagram and utilization labels, not a claim that we have recovered the authors' internal state exactly.

### GPU 1

Allocations:

- `1g.10gb` at start 2
- `1g.10gb` at start 3

Compute occupancy is `2/7 = 28.6%`, consistent with the figure's approximately 30% label. A `3g.40gb` workload can fit at start 4, while its start-0 configuration is blocked.

### GPU 2

Allocations:

- `2g.20gb` at start 2
- `1g.10gb` at start 5

Compute occupancy is `3/7 = 42.9%`, consistent with the figure's approximately 43% label. The `3g.40gb` placements at starts 0 and 4 are both blocked.

## Score interpretation tested

Two explicit score interpretations are retained in code.

1. **`all_legal_starts`**: for each profile with sufficient aggregate compute and memory, count every blocked legal start, including starts whose anchor index is already occupied.
2. **`free_anchor_starts`**: for each profile with sufficient aggregate compute and memory, consider only legal starts whose anchor index is unoccupied in both resource dimensions; then test the full profile footprint for overlap. A free anchor can still have a larger profile footprint blocked by a later occupied resource unit.

Both add `profile.memory_slices` for each counted blocked placement. Both retain the component-wise aggregate resource gate. The second interpretation is not stated as a separate line in Algorithm 1, but it matches the two numeric scores in the paper for the reconstructed states.

| Reconstructed state | Paper-reported score | `free_anchor_starts` | `all_legal_starts` |
|---|---:|---:|---:|
| GPU 1 | 8 | **8** | 14 |
| GPU 2 | 16 | **16** | 23 |

## Contribution breakdown

| Profile | GPU 1: free-anchor contribution | GPU 2: free-anchor contribution |
|---|---:|---:|
| `7g.80gb` | 0 | 0 |
| `4g.40gb` | 4 | 4 |
| `3g.40gb` | 4 | 8 |
| `2g.20gb` | 0 | 2 |
| `1g.20gb` | 0 | 2 |
| `1g.10gb` | 0 | 0 |
| **Total** | **8** | **16** |

This resolves one previously important discrepancy: the previous score implementation over-counted blocked starts whose anchor indices were already occupied, relative to the values implied by this example.

## What this proves—and does not prove

It proves that our two named software interpretations produce different score values and that `free_anchor_starts` reproduces the two reported numbers for this particular reconstructed state.

It does **not** prove that the free-anchor interpretation is identical to the authors' original code, because the paper does not provide the executable implementation and the index filtering is implicit rather than explicit in the pseudocode. Therefore:

- use `mfi_free_anchor_score` as the **Figure-3-conformance candidate**;
- keep `mfi_all_starts` as the **literal all-legal-start sensitivity variant**;
- never silently overwrite one with the other;
- include the interpretation in the methodology whenever reporting results.

Executable regression tests: `tests/test_m12_figure3_conformance.py`.
