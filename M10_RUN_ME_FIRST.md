# M10 — Run Me First

## 1. Install

From PowerShell in the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[test,analysis]"
```

## 2. Run verification

```powershell
pytest -q
python scripts\validate_baseline.py
python scripts\audit_m9_baselines.py --out results\m9_index_order_audit.json
```

## 3. Run the M10 protocol/dynamics matrix

Quick run:

```powershell
python scripts\run_m10_protocol_dynamics.py --gpus 10 --repetitions 1 --out results\m10_smoke
```

Exploratory run included with the milestone uses a modest cluster and repetition count. For a larger paper-sized run, use:

```powershell
python scripts\run_m10_protocol_dynamics.py --gpus 100 --repetitions 50 --out results\m10_100gpu_50rep
```

This larger run is compute-intensive. The MFI paper reports 500 simulations per profile distribution; do not call a smaller run a full reproduction.

## 4. Inspect outputs

- `m10_raw_checkpoints.csv`: raw metrics per scenario, mode, repetition, policy and offered-load checkpoint.
- `m10_summary.csv`: means and standard deviations.
- `m10_paired_mfi_minus_bfbi.csv`: paired MFI minus BF-BI acceptance differences and percentile bootstrap intervals.
- `m10_manifest.json`: configuration and assumptions.
- `M10_PROTOCOL_DYNAMICS_REPORT.md`: automatically generated findings.
- `m10_acceptance_modes_*.png` and `m10_paired_delta_by_mode.png`: plots.

## Interpretation warning

`paper_candidate` is the closest current operationalization, not an exact reproduction. Burst, sparse, short-lifetime, long-lifetime, and no-completion modes are stress tests. Never mix them into the primary replication-oriented claim without labeling them.
