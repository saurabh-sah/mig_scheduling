# M6 Findings and Research Decision

## Executive summary

Milestone 6 adds an exact offline admission oracle for tiny traces and a matched rollout-budget sweep. The current evidence does **not** establish that fixed or adaptive lookahead improves MFI enough to justify its added computation. Lookahead sometimes changes a placement and occasionally improves acceptance in individual workload scenarios, but the pooled differences are small, unstable, and costly to compute.

The next step should not be another layer of algorithmic complexity. First inspect why rollout value estimates so often choose the same placement as MFI, validate the simulator/model against the paper's protocol, and decide whether a distinct research question remains.

## Experiment 1 — Rollout-budget frontier

### Configuration

- 40 jobs per trace; 3 simulated GPUs.
- Six synthetic scenarios: uniform, skew-small, skew-big, bimodal, small-to-big regime shift, and big-to-small regime shift.
- Five seeds per scenario; 30 matched traces total.
- Policies: First Fit, MFI, fixed lookahead at 2/4/8 rollouts per candidate, and adaptive lookahead capped at 2/4/8 rollouts per candidate.
- Horizon: 2 future arrivals.
- All policies in each scenario/seed pair saw the same generated workload.

The CSV output and plots are stored in `results/m6_budget_exploratory_5seeds/`.

### Pooled descriptive results

| Policy | Mean acceptance | Mean decision time | Mean total decision time | Rollout evaluations per arrival | Placement differences from MFI |
|---|---:|---:|---:|---:|---:|
| First Fit | 0.704 | 8.9 µs | 0.358 ms | 0.00 | n/a |
| MFI | 0.703 | 30.7 µs | 1.227 ms | 0.00 | baseline |
| Fixed lookahead, R=2 | 0.706 | 1048.3 µs | 41.930 ms | 6.34 | 1.7% |
| Fixed lookahead, R=4 | 0.702 | 2013.3 µs | 80.532 ms | 12.51 | 3.2% |
| Fixed lookahead, R=8 | 0.703 | 3975.9 µs | 159.036 ms | 25.35 | 4.2% |
| Adaptive, max R=2 | 0.703 | 193.3 µs | 7.732 ms | 0.95 | 0.2% |
| Adaptive, max R=4 | 0.703 | 349.4 µs | 13.977 ms | 1.97 | 0.6% |
| Adaptive, max R=8 | 0.703 | 669.1 µs | 26.763 ms | 3.99 | 1.2% |

Acceptance is the fraction of submitted synthetic jobs accepted under the current simulator model. Decision timing is host- and implementation-dependent. Placement differences are calculated only where the lookahead method reports an MFI-reference placement and the job is accepted.

### Paired acceptance differences versus MFI

| Policy | Mean paired difference | Approx. 95% normal interval | Positive / zero / negative traces |
|---|---:|---:|---:|
| First Fit | +0.0017 | [-0.0096, +0.0129] | 11 / 7 / 12 |
| Fixed lookahead, R=2 | +0.0033 | [-0.0018, +0.0084] | 3 / 26 / 1 |
| Fixed lookahead, R=4 | -0.0008 | [-0.0088, +0.0071] | 3 / 24 / 3 |
| Fixed lookahead, R=8 | +0.0008 | [-0.0029, +0.0045] | 3 / 25 / 2 |
| Adaptive, max R=2 | +0.0000 | [0.0000, 0.0000] | 0 / 30 / 0 |
| Adaptive, max R=4 | +0.0000 | [-0.0023, +0.0023] | 1 / 28 / 1 |
| Adaptive, max R=8 | +0.0008 | [-0.0008, +0.0025] | 1 / 29 / 0 |

These normal-approximation intervals are descriptive, not a final confirmatory statistical analysis. Every lookahead interval includes zero. Fixed R=2 has the largest mean gain among lookahead settings, but the interval includes no improvement and the decision-time cost is around 34 times MFI in this implementation. More rollouts increase runtime without a dependable acceptance gain.

### Scenario dependence

There are different point estimates by scenario. For example, fixed R=2 is higher than MFI in the bimodal and uniform workload means but lower or unchanged in some other scenarios. The regime-shift scenarios do not show a consistent win. With only five seeds per scenario and discrete 40-job acceptance rates, these observations are hypothesis-generating only. They do not support a claim of scenario-specific superiority yet.

## Experiment 2 — Exact offline oracle on tiny traces

The oracle solves for the maximum accepted-job count with complete knowledge of the fixed trace, using the same profile/placement model and simulator semantics. It considers rejection and every legal current placement, then uses memoized dynamic programming over active accepted allocations. It is not an online policy.

### One GPU, 8 jobs

- 12 traces: four profile distributions × three seeds.
- All 12 traces completed inside the 250,000-state limit.
- Mean online acceptance was about 0.531.
- Mean oracle gap was about 0.833 jobs per 8-job trace.

This indicates that the current online policies can leave admission opportunity unused on some tiny traces, but it does not identify which online algorithm will close the gap.

### Two GPUs, 8 jobs

- 12 traces requested; 11 completed and one exceeded the 250,000-state cap.
- Among completed traces, mean online acceptance was about 0.773 and mean oracle gap about 0.545 jobs.
- Oracle runtime varied substantially by trace; one bimodal trace took about 10.3 seconds and explored over 234,000 memoized states.

This confirms both the usefulness and the combinatorial limit of the oracle. The oracle gap is exact only for traces that finish within the state budget and only under the current model. It is not an optimum for physical GPUs in general.

## What the evidence supports

1. The experiment pipeline can compare multiple policies on identical workload traces and quantify both admissions and scheduling cost.
2. The current lookahead implementation changes only a small fraction of accepted placements relative to MFI.
3. For the current two-arrival horizon and synthetic settings, extra search costs far more runtime than the acceptance differences justify.
4. An offline optimum is computable for tiny traces and provides a useful diagnostic bound, but exact search grows quickly with trace complexity.

## What the evidence does not support

- That the proposed adaptive scheduler is better than MFI.
- That MFI is always better than simpler baselines such as First Fit.
- That predictive lookahead is a novel idea in MIG scheduling.
- That results generalize to real MIG hardware, real cluster traces, or a production scheduler.
- That the simulator exactly reproduces every MFI experiment.

## Recommended next action

Do **not** add another learning model or heuristic immediately. Conduct a diagnostic pass:

1. For decisions where lookahead changed a placement, compare candidate rollout means and variances; identify whether the change was driven by stable differences or noisy small-integer rollout outcomes.
2. Increase horizon only on a small subset and measure whether it changes choices enough to reduce the oracle gap; log computation cost and avoid cherry-picking.
3. Use a held-out workload set after choosing any parameters.
4. Revalidate the model/experimental differences from the MFI paper.
5. Keep adaptive lookahead only if it shows a repeatable acceptance-versus-cost advantage. Otherwise pivot the research question, not just the algorithm name.

## Reproducibility artifacts

- `results/m6_oracle_exploratory/`: one-GPU, 8-job oracle study.
- `results/m6_oracle_exploratory_2gpu/`: two-GPU oracle study, including the state-budget-exceeded trace.
- `results/m6_budget_exploratory_5seeds/`: 240 policy runs, raw CSV, config, summaries, paired differences, and plots.
- `EXACT_ORACLE.md`: objective, DP semantics, and caveats.
- `M6_GO_NO_GO.md`: decision criteria for continuation/pivot.
