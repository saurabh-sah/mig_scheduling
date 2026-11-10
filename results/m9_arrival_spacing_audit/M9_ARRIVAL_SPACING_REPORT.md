# M9 — Baseline Conformance and Arrival-Spacing Sensitivity Report

## Purpose

M8 showed 100% acceptance for MFI and BF-BI at the final cumulative offered-load checkpoint. This audit tests whether the result is robust to the temporal spacing of the same profile/duration trace.

## Design

- Same profile order and service duration for every arrival-gap value within a (scenario, seed) trace.
- Only arrival times are changed: `arrival_time = job_index * arrival_gap`.
- Arrival gap 0 is a burst where all arrivals occur at time zero before any positive-duration job completes.
- Policies: paper-style FF/RR operationalizations, BF-BI, WF-BI, and MFI.
- Scenarios: uniform, skew-small, skew-big, and bimodal.
- This is a sensitivity experiment, not exact reproduction of MFI's original arrival process.

## Overall results

| Arrival gap | Policy | Mean acceptance | Mean peak compute occupancy | Mean decision time (ns) |
|---:|---|---:|---:|---:|
| 0 | bf_bi | 1.0000 | 0.9129 | 407598 |
| 0 | mfi | 1.0000 | 0.9129 | 1043498 |
| 0 | paper_ff | 0.7035 | 0.7523 | 23097 |
| 0 | paper_rr | 0.8165 | 0.6170 | 18285 |
| 0 | wf_bi | 0.9018 | 0.7200 | 489126 |
| 0.1 | bf_bi | 1.0000 | 0.8705 | 406221 |
| 0.1 | mfi | 1.0000 | 0.8705 | 1045318 |
| 0.1 | paper_ff | 0.7133 | 0.7295 | 23141 |
| 0.1 | paper_rr | 0.8351 | 0.6239 | 18672 |
| 0.1 | wf_bi | 0.9110 | 0.6998 | 488937 |
| 0.5 | bf_bi | 1.0000 | 0.6902 | 446615 |
| 0.5 | mfi | 1.0000 | 0.6902 | 1193358 |
| 0.5 | paper_ff | 0.7371 | 0.5896 | 19461 |
| 0.5 | paper_rr | 0.9213 | 0.6014 | 15126 |
| 0.5 | wf_bi | 0.9445 | 0.5886 | 513405 |
| 1 | bf_bi | 1.0000 | 0.4805 | 518738 |
| 1 | mfi | 1.0000 | 0.4805 | 1335711 |
| 1 | paper_ff | 0.7252 | 0.4039 | 16893 |
| 1 | paper_rr | 0.9585 | 0.4437 | 16121 |
| 1 | wf_bi | 0.9901 | 0.4611 | 553502 |

## MFI vs BF-BI placement diagnostics

| Arrival gap | Mean acceptance delta (MFI − BF-BI) | Mean status disagreements | Mean placement difference fraction |
|---:|---:|---:|---:|
| 0 | 0.0000 | 0.00 | 0.2543 |
| 0.1 | 0.0000 | 0.00 | 0.2571 |
| 0.5 | 0.0000 | 0.00 | 0.3475 |
| 1 | 0.0000 | 0.00 | 0.4386 |

## Interpretation cautions

1. Changing the arrival gap changes concurrency and is deliberately not a direct reproduction of the paper's protocol.
2. Arrival-gap 0 is a burst/admission stress test, not a realistic continuous arrival process.
3. Mean acceptance across this experiment is descriptive; repetitions are exploratory unless the final design uses adequate power.
4. Runtime is Python simulator overhead and is not deployment scheduler latency.
5. Occupancy is a modeled resource metric, not physical GPU compute utilization.
6. An acceptance difference at a given gap would show sensitivity to load timing, not by itself prove which algorithm is universally better.

## Figures

- `m9_acceptance_vs_arrival_gap.png`
- `m9_peak_compute_utilization_vs_arrival_gap.png`

## Sources

- Zambianco, Fasol, Doriguzzi-Corin, *An Online Fragmentation-Aware GPU Scheduler for Multi-Tenant MIG-based Clouds*, https://arxiv.org/abs/2511.18906
- Turkkan et al., *Optimal Workload Placement on Multi-Instance GPUs*, https://arxiv.org/abs/2409.06646
