# Changelog — Milestone 12

- Added `mfi_free_anchor_score`, an MFI score interpretation that matches the two numeric score values reported in Figure 3 for reconstructed GPU states.
- Preserved `mfi_all_starts` as a sensitivity alternative instead of silently changing the previous score implementation.
- Added `PaperBFWFBestIndexScheduler`, which chooses the selected GPU by aggregate residual score first and checks placements on that GPU, matching the qualitative failure mechanism in Figure 3.
- Kept global-feasible BF-BI/WF-BI under separate labels to prevent baseline conflation.
- Added Figure 3 conformance regression tests and paper-path selection/rejection tests.
- Added a 100-GPU progressive-load experiment with five repetitions per distribution and an exact 85% offered-demand checkpoint.
- Added a 20-GPU burst/high-demand stress study with 150% cumulative offered demand.
- Added per-job decision rows, MFI score-variant decision-difference output, paired bootstrap summaries, run manifests, and plots.
- Documented all remaining operational assumptions and the fact that this is not an exact reproduction of the paper's unavailable executable implementation.
