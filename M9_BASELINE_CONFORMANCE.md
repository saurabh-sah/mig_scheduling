# M9 — Baseline conformance audit

## Purpose

Milestone 8 used a deterministic static blocking-cost function to operationalize the Best Index tie-break for BF-BI/WF-BI. The MFI paper says its MIG-aware baselines use a prior index-selection preference that reserves scarce profile indices. The cited source, Turkkan et al. (2024), publishes an explicit order for each A100-80GB profile.

M9 replaces the heuristic tie-break in the baseline implementations with that explicit preference table. The previous cost score is retained as a test-only audit helper so that the change can be checked.

## Reference index preference table

From Table 1 in *Optimal Workload Placement on Multi-Instance GPUs*:

| Profile | Legal starts | Preference order |
|---|---|---|
| 7g.80gb | 0 | 0 |
| 4g.40gb | 0 | 0 |
| 3g.40gb | 0, 4 | 4, 0 |
| 2g.20gb | 0, 2, 4 | 4, 0, 2 |
| 1g.20gb | 0, 2, 4, 6 | 6, 4, 0, 2 |
| 1g.10gb | 0, 1, 2, 3, 4, 5, 6 | 6, 4, 5, 0, 1, 2, 3 |

The legal-start set and preference order are distinct concepts. The first enumerates allowed placements; the second ranks allowed placements when several are feasible.

### Remaining GPU-selection operationalization

The current BF-BI/WF-BI implementation computes a normalized dual-resource residual:

\[
R(a)=\frac{C_{free}-C_p}{7}+\frac{M_{free}-M_p}{8}.
\]

BF-BI minimizes this value; WF-BI maximizes it. The cited MFI prose says to minimize/maximize available resources after allocation but does not provide a fully executable scalarization for the separate compute and memory dimensions. Thus the index order is now explicit and source-based, while the GPU-level residual formula remains a documented implementation choice. This does not turn the baseline into the authors' exact code.

## Exhaustive result

`python scripts/audit_m9_baselines.py --out results/m9_index_order_audit.json` examined all occupancy-mask pairs in the software state space:

- 128 compute occupancy masks;
- 256 memory occupancy masks;
- 32,768 mask pairs total;
- 21,142 profile/state cases with multiple feasible placements;
- zero ranking mismatches between the previous M8 blocking-cost ranking and the explicit reference ordering.

This means the replacement does not change the local placement preference for any state checked under this model, while making the rule explicit, sourced, and easier to explain. The mask enumeration also includes abstract states that may not be physically reachable, so this is a software-ordering equivalence result, not hardware validation.

## Remaining baseline ambiguities

The MFI paper provides prose definitions for FF, RR, BF-BI, and WF-BI but no accompanying executable source. We therefore continue to label the paper-style FF/RR policies as explicit operationalizations. The tests can show what our implementation does, but cannot prove that it matches the authors' unavailable code.

## Sources

- Zambianco, Fasol, Doriguzzi-Corin, *An Online Fragmentation-Aware GPU Scheduler for Multi-Tenant MIG-based Clouds*, https://arxiv.org/abs/2511.18906
- Turkkan et al., *Optimal Workload Placement on Multi-Instance GPUs*, https://arxiv.org/abs/2409.06646
