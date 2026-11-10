# M11 Baseline Ranking Sensitivity Report

## Scope

This exploratory study tests whether BF-BI/WF-BI outcomes depend on the resource-residual scalarization or tie hierarchy used in our implementation. The MFI paper defines BF-BI and WF-BI verbally (bin-packing vs load-balancing) and gives an established Best Index preference order, but does not specify a scalarization formula for the two separate compute/memory residuals or fully specify cross-GPU tie handling. This report therefore compares plausible operationalizations rather than claiming source-code equivalence.

- GPUs: 4
- Repetitions per profile distribution: 2
- Policy runs: 136
- Workload scenarios: uniform, skew_small, skew_big, bimodal

## Default policy means by scenario

| Scenario | MFI acceptance | BF-BI default | WF-BI default |
|---|---:|---:|---:|
| uniform | 100.000% | 100.000% | 100.000% |
| skew_small | 100.000% | 100.000% | 100.000% |
| skew_big | 100.000% | 100.000% | 100.000% |
| bimodal | 100.000% | 100.000% | 93.750% |

## Largest alternatives by acceptance change vs current BF-BI default

Values are within-scenario paired means across the configured repetitions. Positive means the variant accepted a larger fraction of the same trace than the current BF-BI operationalization.

| Scenario | Variant | Mean acceptance delta vs MFI | 95% bootstrap interval | Mean decision time (ns) |
|---|---|---:|---:|---:|
| uniform | bf_bi__compute_first__gpu_before_index | +0.000% | [+0.000%, +0.000%] | 20462 |
| uniform | bf_bi__compute_first__index_before_gpu | +0.000% | [+0.000%, +0.000%] | 14302 |
| skew_small | bf_bi__compute_first__gpu_before_index | +0.000% | [+0.000%, +0.000%] | 25434 |
| skew_small | bf_bi__compute_first__index_before_gpu | +0.000% | [+0.000%, +0.000%] | 24821 |
| skew_big | bf_bi__compute_first__gpu_before_index | +0.000% | [+0.000%, +0.000%] | 17198 |
| skew_big | bf_bi__compute_first__index_before_gpu | +0.000% | [+0.000%, +0.000%] | 16218 |
| bimodal | bf_bi__compute_first__gpu_before_index | +0.000% | [+0.000%, +0.000%] | 31723 |
| bimodal | bf_bi__compute_first__index_before_gpu | +0.000% | [+0.000%, +0.000%] | 28007 |

## Interpretation

1. The current compute/memory normalized-sum residual is an explicit operationalization, not a formula stated in the MFI paper.
2. The profile-specific Best Index ordering is source-derived from Turkkan et al. (2024), and MFI states it uses that preference policy for BF-BI/WF-BI.
3. No variant is selected merely because it improves acceptance on these traces. Doing so would tune the baseline to our result. Any eventual baseline choice must be justified from source semantics or author clarification, and sensitivity results should remain reported.
4. These synthetic tests do not prove any algorithm's real-world superiority.

## Files

- `m11_raw_runs.csv`: per-run outcomes.
- `m11_per_job_decisions.csv`: per-job placement decisions.
- `m11_decision_change_runs.csv`: per-run decision changes against the current BF/WF default.
- `m11_policy_summary.csv`: acceptance/runtime/fragmentation summaries.
- `m11_paired_delta_vs_mfi.csv`: matched seed deltas and percentile bootstrap intervals.
- `m11_decision_change_summary.csv`: fraction of job decisions that differ from the current BF/WF default.
- `m11_manifest.json`: settings and audit scope.

## Sources

- Zambianco, Fasol, Doriguzzi-Corin, *An Online Fragmentation-Aware GPU Scheduler for Multi-Tenant MIG-based Clouds*, Sec. III–VI, Algorithms 1–2, Tables I–II: https://arxiv.org/abs/2511.18906
- Turkkan et al., *Optimal Workload Placement on Multi-Instance GPUs*, Table 1 and Sec. 3.2.2: https://arxiv.org/abs/2409.06646
- NVIDIA, *Supported MIG Profiles*: https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-mig-profiles.html
