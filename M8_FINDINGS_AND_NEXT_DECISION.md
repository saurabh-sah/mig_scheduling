# M8 Findings and Next Research Decision

## What was actually run

### Quick progressive-load experiment

- 10 GPUs
- 10 independent repetitions for each of four profile distributions
- 40 workload traces
- Five policies per trace: paper-style FF/RR operationalizations, BF-BI, WF-BI, and MFI
- 200 policy/trace simulations and 2,000 policy/checkpoint rows

### Larger-cluster exploratory experiment

- 100 GPUs (same cluster size as the MFI paper)
- 3 repetitions per distribution, only 12 workload traces
- Same five policies on matched traces
- 60 policy/trace simulations and 600 checkpoint rows

The 100-GPU run is an exploratory sanity check, **not** the paper's 500-repetition evaluation. The full 100-GPU/500-repetition command is available, but it was not completed in this milestone.

## Final-checkpoint results

### 10 GPUs, 10 repetitions per profile distribution

| Policy | Mean acceptance | Standard deviation across traces | Mean decision time (ns) |
|---|---:|---:|---:|
| MFI | 100.00% | 0.00 percentage points | 177,248 |
| BF-BI | 100.00% | 0.00 percentage points | 60,950 |
| WF-BI | 98.37% | 2.86 percentage points | 63,266 |
| Paper-style RR operationalization | 96.02% | 3.80 percentage points | 11,079 |
| Paper-style FF operationalization | 85.79% | 8.33 percentage points | 9,318 |

### 100 GPUs, 3 repetitions per profile distribution

| Policy | Mean acceptance | Standard deviation across traces | Mean decision time (ns) |
|---|---:|---:|---:|
| MFI | 100.00% | 0.00 percentage points | 1,584,119 |
| BF-BI | 100.00% | 0.00 percentage points | 500,454 |
| WF-BI | 99.17% | 0.81 percentage points | 535,436 |
| Paper-style RR operationalization | 96.24% | 1.00 percentage point | 16,207 |
| Paper-style FF operationalization | 73.76% | 8.78 percentage points | 17,240 |

Decision runtime is wall-clock Python time on the execution environment and is not comparable to real scheduler latency or reported paper timings.

## What these results mean — and do not mean

1. The model and experiment runner are executable and reproducible; the tests validate their software invariants.
2. MFI and BF-BI achieve 100% acceptance in these samples. Therefore, this experiment does **not** reproduce the paper's reported MFI-vs-BF-BI performance gap.
3. The paper-style FF/RR operationalizations show more rejections, but this difference depends on our explicit rule: select the first (or next round-robin) GPU with enough aggregate resources, and reject if that GPU has no legal placement rather than searching further. This is a defensible interpretation of the paper's failure description, but it is not verified against the authors' source code and may make those baselines harsher than their original implementation.
4. Three repetitions per profile distribution at 100 GPUs are far too few for a confirmatory claim. The results are descriptive only.
5. The resource model is an explicit dual compute/memory abstraction using the paper's profile footprints and legal starts. It is not a driver-level MIG emulator. The offered-load protocol resembles the paper's progressive-load design, but we have not established bit-for-bit or metric-for-metric equivalence.
6. We do not infer that MFI is universally optimal, nor that the proposed lookahead approach is promising. We infer that the benchmark/model fidelity needs further work before the original algorithm comparisons can support a research novelty claim.

## Model and protocol issue revealed by M8

The paper's prose says FF chooses the first GPU with enough available resources, RR distributes over GPUs, and profile index choice can fail because valid MIG placement indices are constrained. It also describes BF-BI and WF-BI using a Best Index preference that reserves scarce starts for profiles with few legal options. The original code is not provided, and not every tie-breaking detail is fully executable from the prose. This means baseline details can materially alter the acceptance results.

Our next engineering task should be a **baseline-conformance audit**, not another proposed algorithm:

- derive tiny hand-checkable states from the paper's figures and profile table;
- document the behavior implied by each possible interpretation of FF/RR/BF-BI/WF-BI;
- test whether results are sensitive to those interpretations;
- identify the exact protocol choices needed before claiming an MFI replication;
- if necessary, ask the paper authors for their reference code or clarification.

## Recommendation

**Do not move to another algorithm yet.** The work so far shows that a code-and-paper protocol audit is necessary. Complete that audit before spending further time on lookahead or adaptive scheduling. Once the comparison protocol is defensible, rerun a adequately powered experiment and return to the research-gap decision.

## Reproducibility artifacts

- `results/m8_quick/m8_run_manifest.json`
- `results/m8_quick/m8_progressive_load_raw.csv`
- `results/m8_quick/m8_progressive_load_summary.csv`
- `results/m8_quick/m8_paired_acceptance_delta_vs_mfi.csv`
- `results/m8_quick/M8_PROTOCOL_VALIDATION_REPORT.md`
- `results/m8_100gpu_3rep/m8_run_manifest.json`
- `results/m8_100gpu_3rep/m8_progressive_load_raw.csv`
- `results/m8_100gpu_3rep/m8_progressive_load_summary.csv`
- `results/m8_100gpu_3rep/m8_paired_acceptance_delta_vs_mfi.csv`
- `results/m8_100gpu_3rep/M8_PROTOCOL_VALIDATION_REPORT.md`
