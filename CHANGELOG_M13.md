# Changelog — Milestone 13

- Made fixed and adaptive lookahead explicitly accept `score_mode="all_legal_starts"` or `score_mode="free_anchor_starts"`.
- Unified candidate scoring so the current placement, tie-breaking reference, hypothetical future placements, and adaptive fallback all use the same selected score mode.
- Exposed `--mfi-score-mode` in the CLI.
- Added score-aware lookahead regression tests; total suite is now 107 passing tests.
- Added a matched 240-run experiment comparing paper-path FF/RR/BF-BI/WF-BI, both MFI modes, fixed lookahead under both modes, and adaptive lookahead under both modes.
- Added progressive-load and 150% simultaneous-arrival burst regimes, per-job logs, policy summaries, paired bootstrap comparisons, score-mode sensitivity analysis, manifests, and plots.
- Findings currently do not establish reliable acceptance improvement from lookahead; fixed lookahead incurs substantially higher decision latency in this simulator.
- Documented that this is an exploratory four-GPU synthetic study, not an exact reproduction or production benchmark.
