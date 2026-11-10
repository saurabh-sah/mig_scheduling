# Changelog

## Milestone 9

- Added the explicit A100-80GB Best Index preference ordering from Turkkan et al. (2024) and switched BF-BI/WF-BI to use its ranks.
- Exhaustively audited the prior blocking-cost rank against the explicit reference order across 32,768 compute/memory masks; no ranking mismatch was found in 21,142 multi-candidate cases.
- Added alternative offered-load bases and arrival-spacing sensitivity experiments, plus CSVs, manifests, plots, and analysis reports.
- Memoized fragmentation-score computations over the finite occupancy-mask state space.
- Added model-fidelity documentation explaining why cumulative scalar offered demand can differ from requested compute/memory demand and concurrent occupancy.
- Added M9 tests for index-order and load-basis assumptions.

## Milestone 6

- Added exact offline acceptance oracle for tiny known traces.
- Added memoized dynamic programming with maximum-job and maximum-state guards.
- Added oracle decisions output and online-policy gap calculations.
- Added independent brute-force cross-check tests.
- Added fixed/adaptive lookahead rollout-budget frontier runner.
- Added paired analysis against MFI, scenario summaries, and plots.
- Added research go/no-go decision guide and explicit oracle limitations.
- Retained all previous M1–M5 schedulers and diagnostics.

## M6 experimental update

- Added 240-run matched budget frontier with 5 seeds across six scenarios.
- Added per-policy placement-difference counts versus MFI.
- Added M6 findings and explicit go/no-go discussion based on current exploratory results.

## M10 — Source protocol and dynamics audit

- Added a paper-to-code conformance matrix and machine-readable protocol files.
- Added matched-profile experimental modes varying arrivals and lifetimes.
- Added source-to-code profile/distribution audit, raw CSV, manifests, summaries, paired deltas, bootstrap intervals, and plots.
- Confirmed the source profile/distribution checks and all 90 tests pass.
- Did not add a new scheduler; MFI/BF-BI gap remains unresolved under the current paper-candidate interpretation.

## Milestone 11 — MFI gate and BF/WF ranking audit

- Added an inspectable per-profile fragmentation-score breakdown with explicit component-wise compute/memory capacity gating.
- Added a source-aligned `1g.10gb@5` / `1g.20gb@4` memory-overlap example, plus score-decomposition invariants.
- Added BF-BI/WF-BI scalarization sensitivity modes: normalized sum, normalized max, compute-first, and memory-first.
- Added tie-hierarchy alternatives to expose whether index preference or GPU ID wins when residual resource scores tie.
- Verified that the default audit variants reproduce the existing BF-BI/WF-BI decisions over 300 deterministic randomized legal-state cases.
- Ran 680 matched policy simulations across 4 profile distributions and 10 repetitions per distribution.
- Added paired acceptance deltas against MFI and each policy family's default, per-job placement-change data, decision-time summaries, plots, and run manifest.
- Added an author clarification email draft; no email has been sent.
- No new scheduling algorithm was added and no superiority claim was established.


## Milestone 12 — Figure 3 conformance and paper-path baseline semantics

- Added a second explicitly named MFI score interpretation that reproduces Figure 3's reported scores 8 and 16 for reconstructed states; retained the previous all-starts interpretation as a sensitivity mode.
- Added a reconstructed Figure 3 regression suite and checked the reported utilization labels against resource occupancy.
- Added paper-path BF-BI/WF-BI operationalizations that choose a GPU by aggregate residual score before checking local feasible starts.
- Renamed/retained global-feasible BF-BI/WF-BI variants distinctly so they are not confused with the paper-path baselines.
- Added 100-GPU progressive offered-load sensitivity (5 repetitions/distribution) and a separate 20-GPU, 150%-demand burst stress test.
- Added per-job decision comparisons, paired bootstrap summaries, exact 85% offered-load checkpoint, manifests, and plots.
- Kept score scalarization, original arrival semantics, and exact code-level equivalence explicitly unresolved; no exact-reproduction claim is made.
