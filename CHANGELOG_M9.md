# Changelog — Milestone 9

- Added the explicit A100-80GB Best Index preference ordering from Turkkan et al. (2024).
- Updated BF-BI and WF-BI to use the explicit reference preference ranks after residual-resource selection.
- Retained the former static blocking-cost score as an equivalence audit helper.
- Added exhaustive validation of that ordering over all compute/memory occupancy-mask pairs.
- Extended progressive workload generation with alternative offered-load bases while preserving `paper_slice_units` as the default.
- Added arrival-gap and load-basis sensitivity runners and analysis scripts.
- Added reports, manifests, CSV outputs, and plots for both studies.
- Memoized fragmentation scores over the finite bitmask state space to reduce repeated computation without changing the metric.
- Added M9 tests for index preferences and load-basis generation.
