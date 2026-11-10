# M9 Findings and Research Decision

## What M9 completed

1. Replaced the inferred Best Index tie-break with the explicit A100-80GB preference order from Turkkan et al. (2024), which MFI identifies as the basis for its BF-BI/WF-BI baselines.
2. Exhaustively compared that reference ordering against the previous M8 static blocking-cost ranking across 32,768 compute/memory occupancy-mask pairs. Across 21,142 profile/state cases with more than one feasible start, the chosen starts matched in all cases.
3. Added an arrival-spacing sensitivity experiment for the same profile/duration traces.
4. Added a load-basis sensitivity experiment comparing paper-scalar, compute, memory, and maximum-normalized resource budgets.
5. Added raw CSV outputs, manifests, summary tables, plots, test coverage, and caveats.
6. Improved fragmentation-score computation using memoization over the finite occupancy-mask state space; outputs are unchanged, but the experiment harness runs more efficiently.

## Main result

The most important observation is not that MFI wins or loses. It is that our current workload/model interpretation can make the principal comparison uninformative.

At 100 GPUs, two repetitions per scenario, using the scalar `Slice No.` offered-load target:

- MFI mean acceptance: 100% across each tested arrival gap;
- BF-BI mean acceptance: 100% across each tested arrival gap;
- the two policies still made different placements on approximately 25%–44% of jobs that both accepted;
- at arrival gap 1.0, mean peak compute occupancy was only about 48%; at a burst gap of 0, it was about 91%.

A separate burst load-basis sweep showed that scalar-slice demand of about 1.0× the seven-unit-per-GPU target represented only about 0.91× cumulative compute demand and 0.93× memory demand under the current dual-resource profile model. When the workload instead reached compute capacity, acceptance for MFI was about 95.8%; when it reached memory capacity, it was about 97.8%.

These are exploratory synthetic results. They show sensitivity to model definitions; they do not show a general MFI improvement or prove which load basis the original paper intended.

## What this means for the research direction

Do not add a new scheduler yet. First, stabilize the simulator's workload semantics and write an explicit conformance table that records:

- source-paper definition;
- exact implementation;
- ambiguity or assumption;
- independent test;
- effect on reported metrics.

Then choose two clearly separated evaluation tracks:

**Track A — replication-oriented:** follow the MFI paper's stated load-axis and duration protocol as closely as the text allows; state every remaining ambiguity and seek author clarification/source code if practical.

**Track B — model stress tests:** separately vary arrival spacing and concurrent resource-demand basis to establish how the policies behave when actual compute/memory occupancy approaches capacity.

Only after Track A is defensible should we resume the claim that any proposed method improves on MFI.

## Go/no-go recommendation

**GO for baseline validation; HOLD new algorithm development.** The current evidence suggests the first blocker is experimental-model fidelity, not lack of scheduling methods. CBL/lookahead is not justified yet, because the baseline comparison can still be made uninformative by offered-load semantics.

## Reproducibility artifacts

- `results/m9_index_order_audit.json`
- `results/m9_arrival_spacing_audit/m9_arrival_spacing_runs.csv`
- `results/m9_arrival_spacing_audit/m9_mfi_vs_bfbi_decisions.csv`
- `results/m9_arrival_spacing_audit/m9_arrival_spacing_summary.csv`
- `results/m9_arrival_spacing_audit/M9_ARRIVAL_SPACING_REPORT.md`
- `results/m9_load_basis_audit/m9_load_basis_runs.csv`
- `results/m9_load_basis_audit/m9_load_basis_summary.csv`
- `results/m9_load_basis_audit/M9_LOAD_BASIS_REPORT.md`

## Sources

- Zambianco et al., *An Online Fragmentation-Aware GPU Scheduler for Multi-Tenant MIG-based Clouds*, https://arxiv.org/abs/2511.18906
- Turkkan et al., *Optimal Workload Placement on Multi-Instance GPUs*, https://arxiv.org/abs/2409.06646
