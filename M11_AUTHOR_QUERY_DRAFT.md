# Draft inquiry to the MFI authors

**Subject:** Clarification / implementation request for MFI fragmentation score and BF-BI/WF-BI baselines

Dear Dr. Zambianco, Dr. Fasol, and Dr. Doriguzzi-Corin,

I am a student working on a Cloud Computing research project studying MIG-aware GPU scheduling. I am independently implementing the MFI baseline from your paper, *An Online Fragmentation-Aware GPU Scheduler for Multi-Tenant MIG-based Clouds* (arXiv:2511.18906), and would be grateful for clarification on a few implementation details to ensure our comparison is faithful.

1. In Section IV and Algorithm 1, the workload resource requirement is introduced as a compute/memory pair, while the fragmentation-score gate is written as \(r_w(p)\leq\Delta S_m\), and blocked placements are evaluated over a compact range of indices. How should this gate and the blocked-placement test be interpreted for asymmetric profiles such as `1g.20gb` (1 compute slice, 2 memory slices) and `3g.40gb` (3 compute slices, 4 memory slices)? Is the implementation intended to compare compute and memory availability component-wise?
2. For BF-BI and WF-BI in Section VI, which exact residual-resource metric is used to rank GPUs when compute and memory have different residual capacities? Is there a published scalarization or tie-breaking rule beyond the stated bin-packing/load-balancing descriptions?
3. When GPU residual scores tie, is the GPU selected first and then the profile-specific Best Index applied within that GPU, or is the index preference part of the global candidate ordering?
4. Could you clarify the precise generation of arrival slots and the definition of \(T\) used to sample workload durations from \([1,T]\)?
5. Is source code or an implementation artifact available for the MFI score and the FF/RR/BF-BI/WF-BI baselines? Access to it would help us avoid over-interpreting details omitted from the paper.

Our current interpretation uses separate compute and memory occupancy masks, a component-wise aggregate-capacity gate, the memory-slice count as the penalty for each blocked legal profile placement, and the index preference order from Turkkan et al. (2024). For BF-BI/WF-BI, we have kept the resource scalarization explicitly labelled as an implementation choice and are testing alternatives rather than claiming exact equivalence.

Thank you for your time and for the useful work. Any clarification you can provide would help us document the baseline accurately.

Best regards,

[Your name]
[Course / Department / Institution]
