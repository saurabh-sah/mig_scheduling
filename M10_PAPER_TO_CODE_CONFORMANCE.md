# M10 — MFI paper-to-code conformance matrix

## Purpose

This document freezes what is supported by the MFI paper, what our implementation operationalizes, and what remains unresolved. It is a prerequisite for credible evaluation—not a claim that our simulator is already a full reproduction.

Primary source: Zambianco, Fasol, Doriguzzi-Corin, *An Online Fragmentation-Aware GPU Scheduler for Multi-Tenant MIG-based Clouds*, arXiv:2511.18906, https://arxiv.org/abs/2511.18906.

## Conformance matrix

| Topic | Paper evidence | Current implementation | Status | Consequence |
|---|---|---|---|---|
| Hardware/profile family | Section III and Table I describe A100 MIG profiles and legal placement indices. | `src/mig_scheduler/profiles.py` uses the six Table I profile classes and legal start sets. Compute/memory footprints are represented separately. | Table-derived; dual-resource mask is a physical interpretation | Validate all profiles against one GPU model; do not call this a driver-level emulator. |
| Table I `Slice No.` | Table I gives 7, 4, 4, 2, 2, 1 for the six profiles. | Stored as `paper_slice_units`, separate from compute and memory occupancy. | Direct transcription of table values | Used for the paper-style offered-load axis only; not used as a replacement for physical feasibility. |
| Table II distributions | Table II provides exact Uniform, Skew-small, Skew-big and Bimodal probabilities. | `MFI_WORKLOAD_DISTRIBUTIONS` contains those probabilities. | Direct transcription | Same distributions should be used for each matched policy comparison. |
| Cluster scale | Section VI says a homogeneous cluster of `M=100` A100 GPUs. | All runners accept `--gpus`; default smoke-test scale may be smaller. | Configurable; runs with fewer GPUs are exploratory | Large-scale reproduction-oriented run should use 100 GPUs. |
| Offered-load axis | Section VI says requested cumulative GPU demand before considering terminations, relative to cluster capacity; it explicitly notes actual utilization is lower. Table I's `Slice No.` is the paper's scalar profile resource column. | `offered_demand_fraction = cumulative Slice No. / (7 * gpu_count)`. Snapshots use the first arrival at/after a checkpoint. | Strong interpretation of the paper's stated axis | Do not confuse offered demand with instantaneous utilization or accepted demand. |
| Arrival process | Section IV says workload arrivals are distributed over time and scheduled online FIFO. Section VI refers to scheduling slots. | `paper_candidate` treats one sampled job as one slot (`arrival_time = slot_index`). | Explicit operationalization; exact inter-arrival rule not specified in source | Keep it separate from burst/sparse sensitivity tests. |
| Duration process | Section VI states durations are sampled in `[1,T]`, with `T` related to scheduling slots required to saturate cluster capacity. | Duration is discrete uniform `[1, number_of_jobs_in_full_scalar-load_trace]`. | Explicit interpretation of `T`; needs confirmation if author clarification/source code becomes available | This assumption can change overlap and fragmentation dynamics. |
| Resource occupancy | Section III describes 7 compute slices and 8 memory slices, including profile footprints with mismatched compute/memory. Section IV defines workload resource needs as a pair. | Separate compute and memory occupancy masks; each profile consumes a range from its legal start in both dimensions. | Physically motivated interpretation | Paper's pseudocode compresses the placement/fragmentation model into a single `x_{m,i}` index notation, so exact formal equivalence cannot be proven from text alone. |
| Fragmentation score | Algorithm 1 adds the profile's memory-slice count for each infeasible placement, considering profiles for which sufficient free resources remain. | `fragmentation.py` tests both raw compute and raw memory sufficiency, then charges each blocked legal placement by `profile.memory_slices`. | Faithful intent; dual-resource feasibility is explicit operationalization | Add a source-example conformance test where the paper state can be reconstructed unambiguously; do not infer unseen states from a figure. |
| MFI decision | Algorithm 2 dry-runs every feasible profile placement and picks the minimum change in score. | `MFIScheduler` evaluates all feasible GPU/start options and chooses minimum `delta_fragmentation`. | Algorithmic conformance testable | Tests establish agreement with our formalized algorithm, not original unpublished executable code. |
| FF and RR | Section VI says FF chooses the first GPU with enough resources and first available index; RR distributes sequentially across available GPUs, using first feasible index. | `PaperFirstFitScheduler` and `PaperRoundRobinScheduler` implement these definitions. | Operationalized from prose | Tie ordering and some failure branches may still depend on implementation convention. |
| BF-BI and WF-BI | Section VI gives bin-packing / load-balancing meanings and states an established best-index preference policy is used. | Index preference order is taken from Turkkan et al. (2024); GPU-level ranking combines normalized compute and memory residuals. | Index order source-based; GPU scalarization is an implementation choice | This unresolved scalarization can affect comparisons with MFI. |
| Rejection semantics | Section VI says rejected jobs are not rescheduled during the simulation. | Simulator rejects immediately when no feasible placement exists. | Directly supported | This is an intentionally simplified acceptance-study model. |
| Metrics | Section VI reports acceptance, allocated workload count, active GPUs, allocated resource utilization, and mean fragmentation score. | We record acceptance, accepted jobs, active GPUs, compute/memory occupancy, fragmentation, and scheduling runtime. | Partial; resource utilization is split into compute/memory dimensions and runtime is additional | Label simulator occupancy precisely; it is not measured GPU compute utilization. |
| Replications | Section VI states 500 independent simulations per profile distribution. | Runner supports configurable repetitions; packaged M10 runs use fewer repetitions for exploration. | Not yet replicated at full repetition count | Do not compare small-run means as if they were the paper's definitive result. |
| Normalization | Section VI says metrics are normalized by their maximum for comparing schemes/distributions. | Raw measurements are retained; summaries can be normalized separately. | Raw-first reporting improves auditability; normalized reproduction plot not yet identical | Preserve raw data and show the normalization equation for any paper-facing graph. |

## Main ambiguity to resolve

The paper's scalar `Slice No.` load axis is suitable for reconstructing the published x-axis, but it is not the same thing as the two-dimensional physical resource footprint. The model must preserve two separate concepts:

1. **Offered scalar demand:** cumulative `Slice No.` units divided by `7 × GPU count`, before accounting for completions, matching the paper's description of the x-axis.
2. **Concurrent physical occupancy:** compute and memory slices currently allocated after arrival and completion events, represented separately in the simulator.

These must not be conflated. Choosing compute or memory load as the main axis simply because it produces more baseline failures would be methodologically invalid.

## Sources

- MFI paper, Sections III–VI, Table I, Table II, Algorithms 1–2: https://arxiv.org/abs/2511.18906
- Turkkan et al., *Optimal Workload Placement on Multi-Instance GPUs*: https://arxiv.org/abs/2409.06646
- NVIDIA MIG profile documentation: https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-mig-profiles.html
