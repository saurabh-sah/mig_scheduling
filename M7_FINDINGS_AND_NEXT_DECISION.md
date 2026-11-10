# M7 Findings and Research Decision

## Executive decision

**Do not add more complexity to lookahead yet.** This sweep shows that longer horizons change more placements, but that does not translate into a dependable acceptance improvement. Adaptive lookahead is cheaper than fixed lookahead but still costs about 11–15 times the MFI decision runtime in this implementation, and its observed acceptance gain is only a few jobs over all traces.

The current evidence does not justify claiming that confidence-budgeted lookahead improves MFI. It also does not show that the broad MIG scheduling area is exhausted. It shows that this particular version of the method has not earned its complexity.

## Experiment performed

- 6 workload scenarios: uniform, skew-small, skew-big, bimodal, small-to-big regime shift, and big-to-small regime shift.
- 5 workload seeds per scenario: 30 matched traces.
- 40 jobs per trace on 3 simulated GPUs.
- 8 policies per trace: First Fit, MFI, fixed lookahead H=1/2/3/5 with 4 rollouts per candidate, and adaptive lookahead H=2/3 capped at 4 rollouts per candidate.
- 240 policy/trace runs; 7,200 lookahead-policy per-job diagnostic rows.
- Every policy within a given scenario/seed used the same generated job trace.
- Confidence intervals use a scenario-stratified paired bootstrap; they are exploratory because there are only five seeds per scenario and several comparisons.

## Headline results

| Policy | Mean acceptance | Paired delta vs MFI | Stratified paired bootstrap 95% interval | Mean decision time | Runtime multiplier vs MFI |
|---|---:|---:|---:|---:|---:|
| First Fit | 70.42% | +0.167 percentage points | [-0.917, +1.250] pp | 10.2 µs | 0.3× |
| MFI | 70.25% | baseline | — | 31.8 µs | 1.0× |
| Fixed lookahead H=1, R=4 | 70.42% | +0.167 pp | [-0.083, +0.500] pp | 1,496.3 µs | 47.0× |
| Fixed lookahead H=2, R=4 | 70.25% | 0.000 pp | [-0.333, +0.333] pp | 2,212.5 µs | 69.5× |
| Fixed lookahead H=3, R=4 | 70.08% | -0.167 pp | [-0.833, +0.583] pp | 2,855.2 µs | 89.7× |
| Fixed lookahead H=5, R=4 | 70.25% | approximately 0.000 pp | [-0.833, +0.833] pp | 3,782.6 µs | 118.8× |
| Adaptive lookahead H=2, max R=4 | 70.50% | +0.250 pp | [0.000, +0.500] pp | 364.7 µs | 11.5× |
| Adaptive lookahead H=3, max R=4 | 70.25% | 0.000 pp | [-0.250, +0.250] pp | 478.6 µs | 15.0× |

`pp` means percentage points, not relative percentage improvement. Adaptive H=2 accepted 846 of 1,200 submitted jobs; MFI accepted 843. That is three additional jobs across the 30 traces, not a large demonstrated gain. The bootstrap interval begins at zero, so this should not be presented as a reliable positive effect.

## What the decision diagnostics show

### 1. Longer horizon changes more decisions, but does not improve acceptance

Among accepted placements for which MFI had a comparable placement:

- H=1 fixed lookahead changed 2.01% of placements.
- H=2 changed 3.44%.
- H=3 changed 6.06%.
- H=5 changed 8.07%.

However, the paired acceptance outcome did not improve as the horizon increased. H=3 was slightly worse than MFI on the pooled mean; H=5 returned to approximately the same acceptance as MFI while requiring nearly 119 times the decision runtime. Changing more placements is therefore not evidence of better scheduling.

### 2. The estimated advantage is small relative to rollout variation

Mean estimated future-acceptance gain over the MFI candidate rose from about 0.0056 jobs at H=1 to 0.0249 jobs at H=5. The mean rollout sample standard deviation of the top candidate rose from about 0.216 to 0.721 jobs. The median top-two estimated-value gap was zero for all listed configurations.

These diagnostics suggest the estimated candidate differences are often small and noisy under four rollouts. Increasing the horizon may produce more variance and computation without establishing a more reliable winner. This is a plausible explanation, not a causal proof of why each placement was made.

### 3. Adaptive gating avoids much of the fixed cost, but the practical gain remains small

- Adaptive H=2 invoked lookahead on 141 of 1,200 job arrivals; adaptive H=3 did so on 149.
- They changed 9 and 10 accepted placements respectively relative to MFI across the experiment.
- Adaptive H=2's acceptance was better on 3 traces and tied on 27, with no negative traces; the absolute effect remained only three extra accepted jobs overall.
- Adaptive H=3 produced no aggregate acceptance gain.

This suggests the adaptive gates save computation compared with always-on lookahead, but the current policy still does not have compelling evidence of a useful acceptance-versus-runtime improvement.

## Model and protocol caution

These are results for the current synthetic simulator. The MFI paper reports an initially empty 100-GPU cluster, progressive offered-load evaluation, durations sampled in slots from 1 to saturation time T, four profile distributions, and 500 independent simulations per distribution. Our current sweep uses 40-job Poisson-arrival traces on three GPUs. It is not an exact replication of MFI's reported 10% improvement. See `M7_MODEL_AND_PROTOCOL_AUDIT.md` and the original paper: https://arxiv.org/abs/2511.18906.

The GPU state model explicitly tracks separate compute and memory occupancy. This is an intentional, documented software abstraction, not a driver-validated MIG emulator. The benchmark BF-BI/WF-BI index preference is an operationalization of the paper's prose rather than an assertion that the authors' unpublished implementation has been reproduced.

## Recommendation

**Research status: modify or pivot, not “keep optimizing the same lookahead.”**

Before investing in adaptive-lookahead refinements, do one of the following:

1. **Validation-first path:** Reconcile the placement/fragmentation abstraction with the MFI paper, reproduce its progressive-load protocol as closely as possible, and preserve this horizon experiment as a separate synthetic stress test.
2. **Question-change path:** Keep the tested simulator and baselines, but move to a different measurable objective or decision mechanism with a clearer gap. Any proposal involving conditional optimization or migration must be compared carefully with OrigaMIG (https://arxiv.org/abs/2610.06646) and prior migration/reconfiguration work.

Do not add an LSTM, RL, a larger rollout budget, or more heuristics simply to make the method seem advanced. First state the exact failure of the current method that the next experiment is designed to address.

## Reproducibility artifacts

- `results/m7_horizon_audit/experiment_config.json`
- `results/m7_horizon_audit/m7_horizon_runs.csv`
- `results/m7_horizon_audit/m7_decision_diagnostics.csv`
- `results/m7_horizon_audit/paired_vs_mfi.csv`
- `results/m7_horizon_audit/paired_summary_vs_mfi.csv`
- `results/m7_horizon_audit/decision_diagnostics_summary.csv`
- `results/m7_horizon_audit/M7_HORIZON_AUDIT_REPORT.md`
- `results/m7_horizon_audit/*.png`
