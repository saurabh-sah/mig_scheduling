# Run Me First — Milestone 11

M11 audits the MFI fragmentation-score capacity gate and the BF-BI/WF-BI resource ranking rule. It intentionally adds no new research scheduler.

## Install (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[test,analysis]"
pytest -q
python scripts\validate_baseline.py
python scripts\audit_m11_resource_gate.py
```

## Inspect the score gate

Read `M11_MFI_RESOURCE_GATE_AUDIT.md`, then open:

- `src/mig_scheduler/fragmentation.py`
- `tests/test_m11_conformance_audit.py`

The diagnostic `fragmentation_score_breakdown_from_occupancy(...)` shows the component-wise free-compute/free-memory gate and contribution from each profile.

## Re-run BF-BI/WF-BI sensitivity

A small smoke run:

```powershell
python scripts\run_m11_baseline_sensitivity.py --gpus 4 --repetitions 2 --seed 20261026 --out results\m11_smoke
```

The included exploratory run:

```powershell
python scripts\run_m11_baseline_sensitivity.py --gpus 20 --repetitions 10 --seed 20261026 --out results\m11_rank_sensitivity_20gpu_10rep
python scripts\plot_m11_sensitivity.py --results results\m11_rank_sensitivity_20gpu_10rep
```

The runner compares the current normalized-sum residual with normalized-max, compute-first, and memory-first alternatives, each under two tie hierarchies. It uses identical workloads for all policies within each scenario/seed.

## Important caveat

The results measure sensitivity to explicit implementations of an underspecified rule. They are not a reproduction of unpublished author code. Do not select a baseline variant just because it gives a preferred research result. See `M11_FINDINGS_AND_NEXT_DECISION.md` and `M11_AUTHOR_QUERY_DRAFT.md`.
