# Project Status — Milestone 13

## Current milestone

**Score-aware fixed/adaptive lookahead comparison across two explicit MFI score interpretations.**

## Repository status

- [x] MFI-inspired simulator and separate compute/memory occupancy model retained.
- [x] Figure-3-conformance score candidate retained as `free_anchor_starts`; prior all-legal-start score retained as sensitivity mode.
- [x] Fixed-budget lookahead candidate ranking and rollout placements use the same named score mode.
- [x] Adaptive-budget lookahead candidate ranking, rollout placements, and fallback use the same named score mode.
- [x] CLI exposes `--mfi-score-mode`.
- [x] Matched 240-run experiment completed across 24 synthetic workload traces, four profile distributions, and progressive/burst regimes.
- [x] Per-job decisions, paired acceptance deltas, bootstrap intervals, score-mode sensitivity, plots, manifests and report generated.
- [x] Automated tests: **107 passed**.
- [x] `scripts/validate_baseline.py` passes all implementation-consistency checks.

## Current findings

- The score mode changes about 55% of lookahead job decision tuples in this experiment.
- Fixed lookahead has about 2.6–2.8 ms mean decision time per request, compared with roughly 0.035–0.041 ms for greedy MFI.
- Adaptive lookahead is cheaper than fixed lookahead, but the search gates commonly fall back and average acceptance matches its score-matched greedy MFI in this sample.
- The paired study does **not** establish a reliable positive acceptance improvement from lookahead.

See `M13_FINDINGS_AND_NEXT_DECISION.md` and `results/m13_score_aware_lookahead_4gpu_3rep/M13_EXPERIMENT_REPORT.md`.

## Current recommendation

Do not claim lookahead is an improvement yet. Keep both MFI score modes explicit, retain lookahead for ablation, and prioritize a larger paired workload/seed study that reports acceptance, decision latency, and the proportion of decisions using lookahead together.

## Known limitations

- Four GPUs and three repetitions per profile distribution/regime are exploratory.
- Workloads are synthetic, based on the paper-slice-units offered-load proxy.
- This is not an exact reproduction of the original paper's 500-run study, and no physical GPU hardware or Kubernetes scheduling integration is exercised.
- Adaptive forecast reliability and its stopping rule are heuristics, not calibrated probabilities.
- A changed placement decision is not necessarily a changed admission or a performance gain.
