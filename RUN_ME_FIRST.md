# Run Me First — M9

## 1. Install and test

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[test,analysis]"
pytest -q
python scripts\validate_baseline.py
```

Expected: **83 tests pass**; `validate_baseline.py` reports profile validation, the memory-overlap regression case, and MFI argmin consistency checks.

## 2. Run exhaustive index-order audit

```powershell
python scripts\audit_m9_baselines.py --out results\m9_index_order_audit.json
```

Expected: 32,768 occupancy-mask pairs checked; 21,142 profile/state cases compared; zero mismatches.

## 3. Reproduce arrival-spacing sensitivity

```powershell
python scripts\run_m9_arrival_spacing_sweep.py --gpus 100 --repetitions 2 --seed 2026 --gaps 0,0.1,0.5,1.0 --out results\m9_arrival_spacing_audit
python scripts\analyze_m9_arrival_spacing_sweep.py --input-dir results\m9_arrival_spacing_audit
```

## 4. Reproduce load-basis sensitivity

```powershell
python scripts\run_m9_load_basis_sweep.py --gpus 100 --repetitions 2 --seed 9026 --out results\m9_load_basis_audit
python scripts\analyze_m9_load_basis_sweep.py --input-dir results\m9_load_basis_audit
```

## 5. Read before interpreting

Open these in order:

1. `M9_FINDINGS_AND_NEXT_DECISION.md`
2. `M9_MODEL_FIDELITY_AUDIT.md`
3. `M9_BASELINE_CONFORMANCE.md`
4. `M9_EXPERIMENT_PROTOCOL.md`

The included 100-GPU studies have only two repetitions per scenario. They are exploratory, not confirmatory. The burst run is a sensitivity test, not the original paper's online arrival process.
