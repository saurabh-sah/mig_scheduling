# Changelog — Milestone 11

- Added `fragmentation_score_breakdown_from_occupancy` for inspectable per-profile MFI-style score contributions.
- Added `baseline_sensitivity.py` with explicit normalized-sum, normalized-max, compute-first, and memory-first BF-BI/WF-BI variants.
- Added explicit tie hierarchy variants: index-before-GPU (existing M10 behavior) and GPU-before-index (staged placement interpretation).
- Added tests confirming the source-aligned memory-conflict case, decomposition consistency, default-baseline equivalence, and a state where scalarization changes the chosen GPU.
- Added a matched-run sensitivity experiment with 680 policy executions across four MFI profile distributions.
- Added MFI gate audit, BF/WF ranking audit, findings/decision memo, and a ready-to-send author inquiry draft.
- No new research algorithm added; no claim of superior acceptance made.
