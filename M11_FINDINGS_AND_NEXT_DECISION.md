# M11 Findings and Research Decision

## What this milestone did

M11 did not add another scheduler. It audited two pieces that were still affecting confidence in our baseline comparisons:

1. the free-resource gate used when counting a blocked legal MIG placement in the MFI fragmentation score;
2. how BF-BI/WF-BI combine residual compute and memory resources and break ties between GPUs versus preferred placement indices.

## Findings

### 1. MFI fragmentation gate

The paper defines the workload resource requirement as a compute/memory pair in its system model, but Algorithm 1 writes the sufficient-resource condition using a compact scalar expression. Our component-wise gate is a sensible interpretation of the stated two-resource model: a profile is counted as fragmented only if both aggregate compute and memory would be sufficient, but its legal placements are blocked by occupied resources.

A new diagnostic exposes per-profile score contributions. A hand-checkable test now confirms the paper-described memory-conflict behavior: `1g.10gb` at index 5 blocks `1g.20gb` at index 4 through memory overlap, even though aggregate compute and memory capacity remains sufficient; the blocked placement contributes two memory slices.

**Still unresolved:** the published mathematical notation does not fully specify how the vector resource requirement maps onto the scalar free-slice quantity and compact index range. Our interpretation is documented; exact equivalence to the authors' implementation is not proven.

### 2. BF-BI/WF-BI ranking

The MFI paper explains BF-BI as minimizing remaining resources and WF-BI as maximizing them, and states it uses the Best Index preference policy. It does not give an explicit formula for scalarizing residual compute and residual memory or fully specify cross-GPU tie hierarchy.

M11 preserves the existing default and tests alternatives: normalized sum, normalized maximum, compute-first lexicographic, memory-first lexicographic, and two tie hierarchies. Tests confirm the current baseline class and the audit class's default agree on 300 deterministic randomized states for each of BF-BI and WF-BI.

### 3. Matched sensitivity study

Study settings: 20 GPUs, 10 repetitions per profile distribution, four MFI distributions, 680 policy runs total.

- MFI: 100% acceptance in all four scenario means for this synthetic protocol.
- Current BF-BI default: 100% acceptance in all four scenario means.
- Current WF-BI default: average 98.12% across the four scenario means.
- BF-BI scalarization/tie variants changed about 3.40% of decisions on average relative to the current BF-BI default, yet acceptance remained 100% here.
- WF-BI variants changed about 2.29% of decisions on average relative to the current WF-BI default, while acceptance remained very close to the default.

These results show that ranking details can change placements even when aggregate acceptance does not change. They do not identify which variant matches the authors' implementation, and they do not demonstrate a new scheduling method.

## Decision

**Do not select a different baseline merely because it performs better on our synthetic traces.** We need source semantics or author clarification to resolve the remaining implementation choices. The best next research action is to send the prepared author inquiry in `M11_AUTHOR_QUERY_DRAFT.md`. It asks for clarification of the two-resource score gate, BF-BI/WF-BI resource ranking and tie hierarchy, timing parameter \(T\), and source availability.

Until the authors respond—or we explicitly decide to define and justify an independent variant—the report should label the implementation an MFI-inspired reference implementation rather than an exact reproduction.

## Go/no-go

- **MFI score:** component-wise compute/memory gate is defensible and now inspectable/tested.
- **BF/WF index order:** source-derived from Turkkan et al. Table 1.
- **BF/WF GPU scalarization and tie hierarchy:** still implementation-specific; disclose and sensitivity-test.
- **New algorithm:** not warranted by this audit alone; resume research-design work after baseline specification is frozen.

## Source availability check

A targeted search of the MFI arXiv landing page and the web for an author-linked GitHub implementation did not surface a public implementation link. This is not proof that code is unavailable privately or under an unindexed repository. A draft source/clarification request is included in `M11_AUTHOR_QUERY_DRAFT.md`; it has **not** been sent.
