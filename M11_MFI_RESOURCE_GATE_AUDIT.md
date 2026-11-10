# M11 — MFI Fragmentation-Score Resource-Gate Audit

## Objective

Resolve as far as the published text permits how the MFI fragmentation metric decides whether a profile has enough aggregate resources to be counted as fragmented, and make the implemented interpretation testable.

## Source facts

The MFI paper's system model defines a workload's MIG resource requirement as a compute/memory pair, \(r_w(p)=\langle r^{comp},r^{mem}\rangle\) (Section IV). Its Algorithm 1 then writes the sufficient-resource gate as \(r_w(p) \leq \Delta S_m\), where \(\Delta S_m\) is introduced as the amount of unused slices. If the gate passes, it examines each legal starting index; each infeasible placement adds the profile's memory-slice count to the score (Algorithm 1; arXiv version 1, pp. 3–4).

The prose and tables also explicitly distinguish compute and memory footprints. In Turkkan et al.'s A100-80GB Table 1, for example, `1g.20gb` requires 1 compute slice and 2 memory slices, `3g.40gb` requires 3 compute and 4 memory slices, and `7g.80gb` requires 7 compute and 8 memory slices. Legal placement starts and their preference order are also listed there (Section 3.2.2 explains that the order is empirically derived with `nvidia-smi`).

Sources:

- Zambianco, Fasol, Doriguzzi-Corin, *An Online Fragmentation-Aware GPU Scheduler for Multi-Tenant MIG-based Clouds*, Section IV and Algorithm 1: https://arxiv.org/abs/2511.18906
- Turkkan et al., *Optimal Workload Placement on Multi-Instance GPUs*, Table 1 and Section 3.2.2: https://arxiv.org/abs/2409.06646
- NVIDIA MIG supported profiles: https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-mig-profiles.html

## Our explicit interpretation

The implementation uses a component-wise capacity test:

\[
C^{free}_m \geq C_p \quad\land\quad M^{free}_m \geq M_p
\]

If the test passes, it examines every legal start for profile \(p\). A start is blocked if its compute mask overlaps occupied compute positions **or** its memory mask overlaps occupied memory positions. Each blocked legal start contributes \(M_p\), the profile's number of memory slices, to the score.

This interpretation follows the paper's explicit two-dimensional resource pair while resolving the compressed inequality in Algorithm 1 into separate compute and memory checks. It is the most physically transparent interpretation of the written model; it is **not proof of bit-for-bit equivalence** to the authors' executable code.

## Hand-checkable source-aligned test

Occupy `1g.10gb` at start 5. That allocation occupies compute position 5 and memory position 5. For `1g.20gb` at start 4, compute position 4 is free, but memory positions 4 and 5 are required and position 5 is occupied. Therefore the placement is blocked by memory overlap.

The GPU still has 6 free compute positions and 7 free memory positions, so the aggregate component-wise capacity gate passes for `1g.20gb` (which needs 1 compute and 2 memory slices). The blocked placement contributes 2 to the fragmentation score. `1g.20gb` at start 6 remains feasible in this state because it consumes compute position 6 and memory positions 6 and 7.

The executable test is `test_fragmentation_breakdown_catches_memory_only_overlap_example`.

## Finding

The current component-wise gate is internally consistent with the system model's compute/memory pair and passes the source-aligned memory-overlap test. The paper's Algorithm 1 notation remains ambiguous because it compares the resource pair to a scalar unused-slice count and uses a compact index interval, whereas the physical profile table has asymmetric compute/memory footprints. We cannot remove this uncertainty without clarification from the authors or an implementation that reproduces their exact internal representation.

The diagnostic function `fragmentation_score_breakdown_from_occupancy` now exposes, for each profile, raw free compute/memory, the capacity-gate outcome, feasible starts, blocked starts counted, per-start penalty, and total contribution. Its per-profile contributions are unit-tested to sum to the existing score.

## Machine-readable audit output

Run `python scripts/audit_m11_resource_gate.py`. It writes `results/m11_resource_gate_audit/m11_resource_gate_example.json`, including the complete profile-by-profile score breakdown for the memory-overlap example. This lets reviewers inspect the actual score contributions instead of relying only on prose.
