# M13 — Run Me First

## 1. Environment

Python 3.11+ is recommended. Install the repository in editable mode from the project root:

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[test,analysis]"
```

## 2. Validate the code

```bash
python -m pytest -q
python scripts/validate_baseline.py
```

Expected current test result: `107 passed`. Baseline validation checks profile definitions, the 1g.20gb memory-overlap regression, and MFI argmin consistency over randomized states. It does not validate paper-level performance replication.

## 3. Reproduce the main M13 experiment

```bash
python scripts/run_m13_score_aware_lookahead.py \
  --gpus 4 \
  --repetitions 3 \
  --seed 20261026 \
  --out results/m13_score_aware_lookahead_4gpu_3rep
```

The script runs 240 simulations over 24 paired workload traces. The current committed output is the generated result for those parameters. It includes:

- `m13_raw_runs.csv` — one row per policy/workload run;
- `m13_per_job_decisions.csv` — admission, placement and lookahead diagnostics;
- `m13_policy_summary.csv` — acceptance, utilization and decision time summaries;
- `m13_paired_acceptance_vs_free_anchor_mfi.csv` — paired acceptance deltas and bootstrap intervals;
- `m13_score_mode_decision_sensitivity.csv` — decision changes between score interpretations;
- `m13_acceptance_progressive.png` and `m13_acceptance_burst.png` — acceptance plots;
- `m13_score_mode_decision_sensitivity.png` — sensitivity visualization;
- `m13_run_manifest.json` — seeds, parameters, policy list and limitations;
- `M13_EXPERIMENT_REPORT.md` — the generated full report.

## 4. Try one score mode from the CLI

```bash
python -m mig_scheduler.cli --policy fixed_lookahead --mfi-score-mode free_anchor_starts
python -m mig_scheduler.cli --policy adaptive_lookahead --mfi-score-mode all_legal_starts
```

## 5. How to read the result

Treat acceptance as a workload-paired metric. Report decision latency and search/fallback frequency alongside acceptance. A different placement tuple is not itself an improvement, and a small positive mean with a paired interval that includes zero is not evidence of a robust benefit. Use `M13_FINDINGS_AND_NEXT_DECISION.md` for the current recommendation.
