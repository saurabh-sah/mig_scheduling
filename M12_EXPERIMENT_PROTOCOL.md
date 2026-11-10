# M12 Experiment Protocol

## A. Main paper-guided run

Command:

```powershell
python scripts\run_m12_figure3_baselines.py --gpus 100 --repetitions 5 --seed 20261026 --out results\m12_figure3_baseline_study_100gpu_5rep
python scripts\plot_m12_results.py --results results\m12_figure3_baseline_study_100gpu_5rep
```

Configuration:

- 100 GPU instances of the A100-80GB profile/placement abstraction.
- Four profile distributions from MFI Table II.
- Five independent seeds per distribution.
- Eight policies.
- Same workload stream is replayed under every policy for each scenario/seed.
- Progressive offered scalar load stops when sum of the paper's `Slice` units reaches the target fraction of `7 * GPU_count`.
- `arrival_gap=1.0`; each workload receives a duration sampled uniformly from integers `[1,T]` with this implementation using `T = number of generated jobs`.
- Checkpoints are 10%, 20%, ..., 100%, plus an exact 85% requested-load checkpoint (the stored snapshot is the first arrival at or above 85%).

This is a **paper-guided sensitivity run**, not exact replication. The MFI paper reports 500 runs per distribution; use 5 as an exploratory validation sample. A final paper should use substantially more seeds after the benchmark semantics are frozen.

## B. Burst/high-demand stress run

```powershell
python scripts\run_m12_figure3_baselines.py --gpus 20 --repetitions 5 --seed 20261101 --arrival-gap 0 --target-fraction 1.5 --out results\m12_burst_150pct_20gpu_5rep
python scripts\plot_m12_results.py --results results\m12_burst_150pct_20gpu_5rep
```

This is not a reproduction of MFI's experiment. It is a stress test in which the cumulative offered demand reaches 150% of nominal scalar slice capacity, all jobs arrive at time zero, and no positive-duration job can finish before later arrivals are processed. It helps expose differences between policies under high contention.

## C. Policies and their precise semantics

- `paper_ff`: paper-path FF operationalization.
- `paper_rr`: paper-path RR operationalization.
- `paper_bf_bi`: rank GPUs by normalized residual sum first; then choose the preferred feasible start only on the selected GPU; reject if none exists.
- `paper_wf_bi`: same process, but maximize normalized residual sum.
- `global_feasible_bf_bi` / `global_feasible_wf_bi`: stronger variants that rank candidates across all feasible placements in the entire cluster.
- `mfi_all_starts`: MFI using the pre-M12 score interpretation.
- `mfi_free_anchor_score`: MFI using the score interpretation that reconstructs the Figure 3 numeric examples.

The paper does not completely specify scalarization of residual compute and memory for BF/WF, nor all tie behavior. M12 documents those assumptions explicitly.

## D. Metrics

Primary: accepted workloads / submitted workloads.

Secondary: mean decision time in Python, final/mean fragmentation score under the common all-starts metric, compute and memory occupancy, active GPU count, decision differences between MFI score modes, and acceptance at offered-load checkpoints.

Decision-time numbers depend on this Python implementation and host, and must not be interpreted as GPU hardware latency.

## E. Statistical analysis

- Preserve raw run-level CSVs and per-job decisions.
- Match all policies on the same workload trace for each scenario/seed.
- Use paired comparisons, not independent unpaired averages, for policy deltas.
- Report confidence intervals; do not claim significance if the interval includes zero.
- Keep main paper-guided and stress-test results separate.

## F. Files generated

- `m12_raw_runs.csv`
- `m12_checkpoint_rows.csv`
- `m12_per_job_decisions.csv`
- `m12_mfi_score_variant_decision_changes.csv`
- `m12_policy_summary.csv`
- `m12_paired_acceptance_delta_vs_figure_score_mfi.csv`
- `m12_run_manifest.json`
- `M12_EXPERIMENT_REPORT.md`
- PNG and SVG plots
