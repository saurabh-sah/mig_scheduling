# M10 — MFI Protocol and Workload-Dynamics Audit

## Scope

This is a protocol sensitivity study, not an exact replication of MFI. It separates the paper-candidate interpretation from explicitly labelled stress tests. Every mode preserves the same generated profile sequence for each scenario/seed; variants change only arrival timing and/or durations.

- GPUs: 20
- Repetitions per profile distribution: 10
- Seed base: 20261026
- Policy runs: 1200
- Checkpoint rows: 4800

## Protocol modes

| Mode | What changes | Status |
|---|---|---|
| `paper_candidate` | One arrival per slot; durations sampled uniformly from integer slots 1…T; cumulative MFI Table-I `Slice No.` reaches the offered-load target. | Closest operationalization of stated paper protocol; unresolved source ambiguities remain. |
| `burst_arrivals` | Same profiles and durations; all arrivals at time 0. | Stress test only. |
| `sparse_arrivals` | Same profiles and durations; inter-arrival spacing doubles. | Stress test only. |
| `short_lifetimes` | Same profiles; duration uniform 1…floor(T/4). | Stress test only. |
| `long_lifetimes` | Same profiles; duration uniform ceil(0.75T)…T. | Stress test only. |
| `no_completion_before_stream_end` | Same profiles; duration T+1, so no positive-duration job completes before the final arrival. | Stress test only. |

## Interpretation limits

The MFI paper says it uses a 100-GPU homogeneous cluster, profiles from Table I, the four distributions in Table II, durations sampled from [1,T], rejected requests are not rescheduled, and 500 independent simulations per profile distribution. It does not release source code and leaves some implementation details underspecified. This repository models compute and memory occupancy separately and explicitly operationalizes arrival timing and the BF/WF GPU residual score. These results must not be represented as an exact replication.

The paper's offered-load x-axis is cumulative requested `Slice No.` divided by the cluster's scalar capacity; it explicitly does not account for terminations when calculating this axis. Our stress modes intentionally preserve that axis while changing concurrent occupancy. This isolates the effect of workload dynamics from the profile sequence.

## MFI versus BF-BI at final offered load

| Scenario | Mode | Mean paired delta (MFI − BF-BI) | Bootstrap 95% interval | MFI positive / tie / negative repetitions |
|---|---|---:|---:|---:|
| bimodal | paper_candidate | 0.0000 | [0.0000, 0.0000] | 0 / 10 / 0 |
| bimodal | burst_arrivals | 0.0023 | [0.0000, 0.0070] | 1 / 9 / 0 |
| bimodal | sparse_arrivals | 0.0000 | [0.0000, 0.0000] | 0 / 10 / 0 |
| bimodal | short_lifetimes | 0.0000 | [0.0000, 0.0000] | 0 / 10 / 0 |
| bimodal | long_lifetimes | 0.0023 | [0.0000, 0.0070] | 1 / 9 / 0 |
| bimodal | no_completion_before_stream_end | 0.0023 | [0.0000, 0.0070] | 1 / 9 / 0 |
| skew_big | paper_candidate | 0.0000 | [0.0000, 0.0000] | 0 / 10 / 0 |
| skew_big | burst_arrivals | 0.0000 | [0.0000, 0.0000] | 0 / 10 / 0 |
| skew_big | sparse_arrivals | 0.0000 | [0.0000, 0.0000] | 0 / 10 / 0 |
| skew_big | short_lifetimes | 0.0000 | [0.0000, 0.0000] | 0 / 10 / 0 |
| skew_big | long_lifetimes | 0.0000 | [0.0000, 0.0000] | 0 / 10 / 0 |
| skew_big | no_completion_before_stream_end | 0.0000 | [0.0000, 0.0000] | 0 / 10 / 0 |
| skew_small | paper_candidate | 0.0000 | [0.0000, 0.0000] | 0 / 10 / 0 |
| skew_small | burst_arrivals | 0.0000 | [0.0000, 0.0000] | 0 / 10 / 0 |
| skew_small | sparse_arrivals | 0.0000 | [0.0000, 0.0000] | 0 / 10 / 0 |
| skew_small | short_lifetimes | 0.0000 | [0.0000, 0.0000] | 0 / 10 / 0 |
| skew_small | long_lifetimes | 0.0000 | [0.0000, 0.0000] | 0 / 10 / 0 |
| skew_small | no_completion_before_stream_end | 0.0000 | [0.0000, 0.0000] | 0 / 10 / 0 |
| uniform | paper_candidate | 0.0000 | [0.0000, 0.0000] | 0 / 10 / 0 |
| uniform | burst_arrivals | 0.0000 | [0.0000, 0.0000] | 0 / 10 / 0 |
| uniform | sparse_arrivals | 0.0000 | [0.0000, 0.0000] | 0 / 10 / 0 |
| uniform | short_lifetimes | 0.0000 | [0.0000, 0.0000] | 0 / 10 / 0 |
| uniform | long_lifetimes | 0.0000 | [0.0000, 0.0000] | 0 / 10 / 0 |
| uniform | no_completion_before_stream_end | 0.0000 | [0.0000, 0.0000] | 0 / 10 / 0 |

## Conclusions (descriptive only)

- The `paper_candidate` rows are the main replication-oriented comparison; the other modes are stress tests and must be reported separately.
- A change in acceptance across stress modes indicates sensitivity to arrival/lifetime semantics; it is not proof that the paper's protocol was implemented incorrectly.
- Do not select a protocol because it creates a larger MFI-vs-baseline gap. Any mismatch with the paper requires a source-grounded explanation.
- Do not claim an algorithmic improvement from this audit. The purpose is to clarify the workload dynamics and protect later experiments from protocol ambiguity.

## Figures

- `m10_acceptance_modes_bimodal.png`
- `m10_acceptance_modes_skew_big.png`
- `m10_acceptance_modes_skew_small.png`
- `m10_acceptance_modes_uniform.png`
- `m10_paired_delta_by_mode.png`

## Source

Zambianco, Fasol, Doriguzzi-Corin, *An Online Fragmentation-Aware GPU Scheduler for Multi-Tenant MIG-based Clouds*, arXiv:2511.18906, especially Table I, Table II, Sections IV–VI. https://arxiv.org/abs/2511.18906
