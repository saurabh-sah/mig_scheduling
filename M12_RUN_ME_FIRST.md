# Run Me First — Milestone 12

M12 resolves an important baseline-conformance issue. It does not add a new research scheduler.

## 1. Install and test (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[test,analysis]"
pytest -q
python scripts\validate_baseline.py
```

## 2. Read the key documents

1. `M12_FIGURE3_CONFORMANCE.md` — how the Figure 3 scores are reconstructed.
2. `M12_BASELINE_SEMANTICS.md` — why paper-path and global-feasible BF/WF are separate.
3. `M12_FINDINGS_AND_NEXT_DECISION.md` — results and recommendation.
4. `M12_EXPERIMENT_PROTOCOL.md` — exact settings and reproducible commands.

## 3. Run a small smoke test

```powershell
python scripts\run_m12_figure3_baselines.py --gpus 4 --repetitions 2 --seed 12345 --out results\m12_smoke
python scripts\plot_m12_results.py --results results\m12_smoke
```

## 4. Run the included experiment configurations

Main paper-guided sensitivity run (100 GPUs, 5 repetitions per distribution):

```powershell
python scripts\run_m12_figure3_baselines.py --gpus 100 --repetitions 5 --seed 20261026 --out results\m12_figure3_baseline_study_100gpu_5rep
python scripts\plot_m12_results.py --results results\m12_figure3_baseline_study_100gpu_5rep
```

Burst/high-demand stress test (20 GPUs, 5 repetitions per distribution):

```powershell
python scripts\run_m12_figure3_baselines.py --gpus 20 --repetitions 5 --seed 20261101 --arrival-gap 0 --target-fraction 1.5 --out results\m12_burst_150pct_20gpu_5rep
python scripts\plot_m12_results.py --results results\m12_burst_150pct_20gpu_5rep
```

The burst case is deliberately a stress test, not the MFI paper protocol.

## 5. Interpret cautiously

- `mfi_free_anchor_score` matches the paper's Figure 3 scores 8 and 16 for the reconstructed states.
- `mfi_all_starts` remains as a score sensitivity variant.
- `paper_bf_bi`/`paper_wf_bi` choose a GPU before checking its legal starts; the `global_feasible_*` variants search globally and are different policies.
- The current study uses five repetitions, not the paper's 500. It is exploratory.
- Passing tests validates code properties, not exact agreement with unpublished author code or actual GPU drivers.
