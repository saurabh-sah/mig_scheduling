# Milestone 7 changelog

- Added `scripts/run_m7_horizon_audit.py` for matched per-trace sweep of fixed horizons 1, 2, 3, 5 and adaptive horizons 2, 3.
- Added per-job candidate diagnostics: lookahead value, sample dispersion, MFI candidate score, chosen candidate score, estimated gain vs MFI, top-two value gap, placement changes, runtime, rollout count, and adaptive fallback/stop reasons.
- Added `scripts/analyze_m7_horizon_audit.py` to compute per-policy and per-scenario summaries, paired acceptance differences, and scenario-stratified paired-bootstrap intervals.
- Added an MFI model/protocol audit and clear warnings that current experiments are MFI-inspired, not exact paper replication or hardware/driver-level MIG emulation.
- Added tests for diagnostic parsing, deterministic bootstrap behavior, and report analysis utilities.
