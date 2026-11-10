# M7 Model and Protocol Audit

## Purpose

This audit prevents us from describing our current simulator as an exact reproduction of the MFI paper or as a hardware MIG emulator. Milestone 7 adds decision-level diagnostics and horizon sensitivity, but does not change the core placement model.

## What is grounded in the MFI paper

Primary source: Zambianco, Fasol, and Doriguzzi-Corin, *An Online Fragmentation-Aware GPU Scheduler for Multi-Tenant MIG-based Clouds*, arXiv:2511.18906, https://arxiv.org/abs/2511.18906.

The paper defines fragmentation relative to legal MIG profile starts and builds a score by evaluating failed placements for profiles, weighting each failed configuration by the profile's memory-slice count. MFI dry-runs feasible allocations and chooses the GPU/start position with the minimum fragmentation-score increment. The paper lists six A100-80GB profile classes and their allowed indices: 7g.80gb at {0}, 4g.40gb at {0}, 3g.40gb at {0,4}, 2g.20gb at {0,2,4}, 1g.20gb at {0,2,4,6}, and 1g.10gb at {0,1,2,3,4,5,6}. See the paper's system model and algorithm sections.

The paper's evaluation protocol is not the same as our current simulator's main protocol. It describes an initially empty cluster of 100 A100 GPUs that is progressively loaded toward capacity. Workload durations are sampled in scheduling slots from 1 to T, where T is the number of arrival slots required to reach the offered resource budget. It evaluates uniform, skew-small, skew-big, and bimodal profile distributions and averages 500 independent simulations per distribution. Its reported metrics include acceptance, allocated workloads, active GPUs, resource utilization, and average cluster fragmentation. See the paper's Section VI: https://arxiv.org/abs/2511.18906.

## What our simulator currently does

- Represents each GPU using separate seven-position compute and eight-position memory occupancy arrays.
- Uses the MFI profile names and legal start-index table.
- Tests both compute and memory non-overlap when checking a placement.
- Computes an MFI-style fragmentation score that weights blocked placements by the profile's memory-slice count.
- Runs a discrete-event online simulator: Poisson-distributed inter-arrivals, integer-valued synthetic durations, immediate admission/rejection, and release at completion.
- Uses 1–3 GPUs and fixed-size workload traces for most diagnostic experiments.
- Implements BF-BI/WF-BI with a documented static blocking-cost tie-breaker because the paper describes the best-index priority in prose rather than publishing a full executable definition.

## Why the difference matters

The dual compute/memory occupancy representation is a transparent resource model, but it is not a driver-validated MIG configuration engine. The MFI paper presents a scalar slice/index abstraction in its algorithm and describes its feasibility and weighting rules. Our two-axis interpretation makes the compute and memory constraints explicit, but this means a result from our simulator must be called **MFI-style** or **MFI-inspired**, not an exact replication of the reported 10% improvement.

Similarly, the synthetic Poisson-arrival experiments are useful for controlled comparisons of our own policies, but their rates cannot be compared numerically with the paper's progressive offered-load plots.

## What is required for a stronger reproduction claim

1. Reconcile the paper's scalar slice abstraction, profile-specific starts, compute slices, and memory slices with its Table I and equations.
2. If that reconciliation remains ambiguous, report the chosen interpretation explicitly and test alternative reasonable abstractions.
3. Recreate the progressive offered-load protocol, lifetime sampling, cluster size, and number of independent runs as closely as practical.
4. Implement the benchmark placement-index rule exactly where the paper specifies it; otherwise label the code as our own operationalization.
5. Report differences from the original protocol and avoid comparing percentages across incompatible experimental settings.

## Sources checked

- MFI paper: https://arxiv.org/abs/2511.18906
- OrigaMIG (newer related work): https://arxiv.org/abs/2610.06646
- NVIDIA MIG user guide: https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-mig-profiles.html

OrigaMIG studies heterogeneous multi-host placement, CPU/RAM constraints, admission, consolidation, and migration. It is a necessary related-work check for any proposed adaptive search or selective optimization contribution.
