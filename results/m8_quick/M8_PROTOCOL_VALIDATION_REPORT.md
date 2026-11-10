# Milestone 8 — MFI Protocol Validation Report

## Purpose

This experiment improves transparency and reproducibility of our MFI-inspired progressive-load evaluation. It does not establish an exact replication of the original paper or validate physical MIG behavior.

## Run configuration

- GPUs: 10
- Repetitions per scenario: 10
- Elapsed wall-clock seconds: 0.88
- Raw input rows: 2000
- Scenarios: bimodal, skew_big, skew_small, uniform
- Policies: bf_bi, mfi, paper_ff, paper_rr, wf_bi
- Offered-load checkpoints: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0

## Final checkpoint summary

| Policy | Mean acceptance | SD | Mean decision time (ns) | Mean active GPUs at final arrival |
|---|---:|---:|---:|---:|
| bf_bi | 1.0000 | 0.0000 | 60949.6 | 6.17 |
| mfi | 1.0000 | 0.0000 | 177248.3 | 6.15 |
| paper_ff | 0.8579 | 0.0833 | 9318.0 | 5.50 |
| paper_rr | 0.9602 | 0.0380 | 11079.1 | 8.62 |
| wf_bi | 0.9837 | 0.0286 | 63265.8 | 9.38 |

## Interpretation rules

- Compare policies using identical workload seeds; the runner does this.
- Treat paired intervals as descriptive unless repetition counts and experimental design justify confirmatory inference.
- Do not compare our numeric outcomes directly with the paper's figures unless the model and protocol are matched.
- The current GPU state uses separate compute and memory occupancy arrays with legal starts from the MFI profile table. This is an explicit software abstraction.
- The progressive workload generator targets 7 scalar offered-load units per GPU using the MFI table's `Slice No.` units; duration is sampled uniformly from integer slots 1..T after T is known.
- Our index-placement tie-breakers are operationalizations where the paper provides prose but not original source code.

## Artifacts

- `m8_progressive_load_raw.csv`: per-policy, per-repetition, per-checkpoint data.
- `m8_progressive_load_summary.csv`: means, SDs, and approximate intervals.
- `m8_paired_acceptance_delta_vs_mfi.csv`: paired differences and bootstrap intervals.
- `m8_final_checkpoint_summary.csv`: end-of-offered-load summary.

## Reference

Zambianco, Fasol, and Doriguzzi-Corin, *An Online Fragmentation-Aware GPU Scheduler for Multi-Tenant MIG-based Clouds*, arXiv:2511.18906. https://arxiv.org/abs/2511.18906
