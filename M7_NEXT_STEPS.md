# M7: Decision-Centric Diagnostics and Horizon Sensitivity

## What this milestone adds

1. Re-runs matched workload traces over multiple lookahead horizons.
2. Logs per-job candidate values, the selected placement, the MFI placement, the estimated value gap, rollout variance, fallback reasons, and scheduling time.
3. Uses a scenario-stratified paired bootstrap interval for acceptance deltas versus MFI.
4. Produces decision-level, per-policy, per-scenario, and paired summaries plus plots.
5. Includes a separate model/protocol audit so findings are not incorrectly described as exact replication of MFI.

## Default experiment

- Scenarios: uniform, skew-small, skew-big, bimodal, small-to-big shift, big-to-small shift.
- Seeds: 11, 22, 33, 44, 55.
- Jobs per trace: 40.
- GPUs: 3.
- Fixed lookahead horizons: H = 1, 2, 3, 5; 4 rollouts per candidate.
- Adaptive horizons: H = 2, 3; max 4 rollouts per candidate.
- Baselines: First Fit and MFI.
- Expected runs: 6 scenarios × 5 seeds × 8 policies = 240.

## Run

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[test,analysis]"
pytest -q
python scripts\run_m7_horizon_audit.py --out results\m7_horizon_audit
python scripts\analyze_m7_horizon_audit.py results\m7_horizon_audit\m7_horizon_runs.csv results\m7_horizon_audit\m7_decision_diagnostics.csv --out results\m7_horizon_audit
```

## Read the outputs in this order

1. `M7_HORIZON_AUDIT_REPORT.md`
2. `paired_summary_vs_mfi.csv`
3. `decision_diagnostics_summary.csv`
4. `scenario_summary.csv`
5. `m7_decision_diagnostics.csv` when diagnosing a specific job/seed

## Research interpretation

The estimated rollout-value advantage is not a causal measure of a realized job's outcome. It is a model-based estimate used to make the decision. To test whether it helps, compare whole-trace outcomes under identical generated workloads and report both acceptance and overhead.

If horizons 1–5 do not produce a repeatable acceptance improvement, or if their overhead is high relative to a tiny gain, do not keep increasing the horizon indefinitely. Use the result as evidence to revise or abandon the current hypothesis.
