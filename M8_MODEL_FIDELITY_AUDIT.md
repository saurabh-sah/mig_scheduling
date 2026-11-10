# Milestone 8 — MFI Model Fidelity and Progressive-Load Protocol Audit

## Why this milestone exists

Milestone 7 showed that fixed and adaptive lookahead do not yet produce a convincing acceptance-versus-overhead improvement. Before adding more algorithmic complexity, we need to establish a more defensible reference experiment against the published MFI evaluation.

This milestone adds a progressive-offered-load study, a run manifest, aggregation/paired comparison scripts, plots, and tests. It does **not** claim exact reproduction of MFI.

## Facts taken from the MFI paper

Primary source: Zambianco, Fasol, and Doriguzzi-Corin, *An Online Fragmentation-Aware GPU Scheduler for Multi-Tenant MIG-based Clouds*, arXiv:2511.18906, https://arxiv.org/abs/2511.18906.

The paper states that it:

- models a homogeneous cluster of 100 A100 GPUs;
- progressively loads an initially empty cluster toward its resource capacity;
- samples workload lifetimes uniformly from integer scheduling slots `[1, T]`, where `T` is the number of slots required to saturate the offered-demand budget;
- tests four workload-profile distributions: uniform, skew-small, skew-big, and bimodal;
- compares MFI against First Fit (FF), Round Robin (RR), Best Fit–Best Index (BF-BI), and Worst Fit–Best Index (WF-BI);
- reports allocated/scheduled workloads, acceptance rate, active GPUs, resource utilization, and average cluster fragmentation;
- averages 500 independent simulations for each profile distribution;
- normalizes the reported metrics for plotting.

The profile distribution table and legal starts are recorded in `configs/a100_80gb_mfi.json` and `src/mig_scheduler/profiles.py`. The paper describes FF as selecting the first GPU with enough available resources, and RR as selecting the next such GPU. Its examples indicate that a scheduler can choose a GPU with sufficient aggregate resources yet fail due to incompatible legal indices.

## What this repository implements

`python scripts/run_m8_protocol_validation.py` produces a progressive offered-load trace for each scenario/repetition and runs every baseline on the same trace. It records the first observed snapshot at or beyond 10%, 20%, ..., 100% cumulative offered load.

The run manifest records the hardware label, profile table, distributions, seed base, policies, and repetition count. For FF/RR, this milestone adds explicit paper-style operationalizations: select the first/round-robin GPU with enough aggregate compute and memory, use its earliest feasible legal start, and reject if that selected GPU has no legal start rather than searching further. This is designed to expose the paper-described index-fragmentation failure, but remains an inferred implementation because no reference source code is published. The analysis script emits summary data, paired acceptance differences against MFI, and plots.

## Critical distinctions from the paper

| Dimension | Published MFI description | Repository implementation | Consequence |
|---|---|---|---|
| Cluster scale | 100 GPUs | Configurable; quick run defaults to 10 | Use `--gpus 100` for paper-scale cluster size |
| Repetitions | 500 per distribution | Configurable; quick run defaults to 10 | Quick outcomes are exploratory only |
| Workload load pattern | Progressive offered-load toward capacity | Progressive workload generated from scalar `Slice No.` units | Similar protocol structure, generated independently |
| Duration | Integer scheduling slots sampled from `[1,T]` | Uniform integer duration from 1 through generated trace length `T` | Matches the textual model at a high level |
| Hardware state | Paper uses slice-index abstraction and describes compute/memory slice constraints | Explicit separate compute and memory occupancy arrays with MFI legal starts | Model is a transparent interpretation, not a driver-validated MIG emulator |
| Baselines | FF, RR, BF-BI, WF-BI | All five policies implemented; FF/RR use explicit aggregate-capacity-first operationalizations, and BF-BI/WF-BI use deterministic index tie-breaks | Do not claim identity with unpublished source code |
| Resource utilization | Described as allocated slices/resources and normalized in paper plots | Report compute utilization and memory utilization separately | Do not equate these fields directly with paper's plotted scalar metric |
| Original code | No official executable source included in paper | Independent implementation | Replication-oriented validation, not bit-for-bit replication |

## Model issue we are not hiding

MIG exposes seven SM slices and eight memory slices on the A100-80GB profile model described in the paper. Profiles can have different compute and memory footprints; for example, `1g.20gb` uses one compute slice and two memory slices, and `3g.40gb` uses three compute slices and four memory slices. The paper's fragmentation score weights failed placements by memory-slice footprint, while its notation also uses a scalar slice-index occupancy abstraction. This repository makes the separate dimensions explicit and applies each profile's legal start indices to both masks.

That is a reasonable, auditable interpretation, but it is still an interpretation. The report labels the output MFI-style / replication-oriented until we have reconciled all model details with the original authors' source or an unambiguous formal description.

## Reproduction commands

Quick run (for code validation):

```powershell
python scripts\run_m8_protocol_validation.py --gpus 10 --repetitions 10 --out results\m8_quick
python scripts\analyze_m8_protocol_validation.py results\m8_quick\m8_progressive_load_raw.csv --out results\m8_quick
```

Paper-scale run (100 GPUs and 500 repetitions per profile distribution):

```powershell
python scripts\run_m8_protocol_validation.py --gpus 100 --repetitions 500 --out results\m8_paper_scale
python scripts\analyze_m8_protocol_validation.py results\m8_paper_scale\m8_progressive_load_raw.csv --out results\m8_paper_scale
```

The paper-scale command is computationally expensive. Run a quick experiment and estimate wall time before starting it. All policies within a repetition use the same workload trace, allowing paired comparisons.

## Interpretation rules

- Do not compare results from the 10-GPU quick run directly with the paper's 100-GPU figures.
- Do not call this experiment an exact reproduction.
- Do not call allocated compute/memory occupancy physical GPU utilization.
- Do not claim MFI is beaten unless paired results, variability, workload-specific results, and scheduling overhead all support that claim.
- The workload is synthetic. We have no production trace proving that these profile distributions or service lifetimes represent a real cloud deployment.

## Success criterion for this milestone

This milestone is successful if it produces reproducible progressive-load experiments, records the full configuration, compares all primary baselines on matched traces, and makes the remaining fidelity differences explicit. It does not require a particular policy to win.
