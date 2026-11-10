# M9 — Model-fidelity audit: what does “100% offered load” mean?

## Executive finding

In the M8 simulator, the progressive offered-load sequence stops when the sum of the paper's scalar `Slice No.` column reaches seven units per GPU. The physical resource model, however, tracks seven compute slices and eight memory slices separately. For profiles whose compute and memory footprints differ, the scalar offered-load budget does not imply that either modeled dimension has reached capacity.

This is a material modelling issue because M8 observed 100% acceptance for MFI and BF-BI. M9 measures resource demand directly and demonstrates that the same scalar load target corresponds to lower cumulative compute and memory demand in the dual-resource model.

The MFI paper itself says its load axis is based on requested resources before accounting for terminations, and notes that actual utilization is lower. It does not publish executable source or unambiguously specify every detail needed to reproduce the arrival process. Therefore, the current implementation remains *paper-inspired*, not an exact replication.

## Definitions used in M9

For a trace with profile requests \(p_1,\ldots,p_n\), the three cumulative demand measures are:

\[
D_{paper}=\sum_{j=1}^{n} s_{paper}(p_j),
\qquad D_C=\sum_{j=1}^{n} c(p_j),
\qquad D_M=\sum_{j=1}^{n} m(p_j),
\]

where `paper_slice_units` preserves the MFI Table-I `Slice No.` column, \(c(p)\) is compute slices, and \(m(p)\) is memory slices.

The normalized demand ratios are:

\[
\rho_{paper}=\frac{D_{paper}}{7G},\quad
\rho_C=\frac{D_C}{7G},\quad
\rho_M=\frac{D_M}{8G},
\]

where \(G\) is the number of GPUs.

These are **offered cumulative demand** ratios. They are not the same as concurrent occupancy after job completions, nor are they measured physical GPU compute utilization.

## Arrival-spacing sensitivity

`results/m9_arrival_spacing_audit/` varies only the arrival gap for a fixed trace of profile types and durations. At gap 0, every job arrives at simulation time zero before any positive-duration workload can complete. Larger gaps allow more short-duration jobs to finish during the arrival stream.

For the 100-GPU, two-repetition exploratory run:

- MFI and BF-BI both accepted every request at each tested gap (0, 0.1, 0.5, 1.0).
- Their placement choices differed on a meaningful fraction of jointly accepted jobs: about 25%–44% depending on arrival gap.
- Mean peak compute occupancy fell from roughly 91% at gap 0 to roughly 48% at gap 1 for the scalar-load trace.

Therefore, 100% acceptance does not mean the policies made identical decisions, and the current workload is not driving the modeled cluster to full concurrent occupancy in the same way for each arrival spacing.

## Load-basis sensitivity

`results/m9_load_basis_audit/` creates burst workloads whose cumulative demand reaches one of four targets:

1. `paper_slice_units`: the original scalar target of 7 units per GPU;
2. `compute`: 7 compute slices per GPU;
3. `memory`: 8 memory slices per GPU;
4. `max_normalized`: whichever normalized compute or memory dimension reaches capacity first.

For the 100-GPU, two-repetition exploratory study, under the `paper_slice_units` basis the trace demanded about 90.9% of compute and 92.7% of memory capacity cumulatively and MFI/BF-BI accepted 100%. Under the `compute` basis, cumulative requested compute was about 100.3% of capacity and memory about 102.9%, and acceptance fell to about 95.8% for MFI. Under the `memory` basis, memory demand was about 100.2% and MFI acceptance was about 97.8%.

These results demonstrate sensitivity to the workload budget definition. They do **not** establish that the compute or memory basis is the correct reproduction of the published experiment; the correct protocol must be inferred from the paper's model and clarified where underspecified.

## Main conclusion

Before continuing to optimize the scheduler, the experiment must clearly label its load axis and distinguish:

- cumulative offered demand;
- concurrent resource occupancy;
- allocated compute/memory occupancy;
- accepted-job rate.

The next research step should be to decide which of these is the paper's intended load axis and which alternative scenarios are used as robustness tests. We should not select the load definition because it creates a desired MFI-vs-baseline gap.

## References

- Zambianco et al., MFI paper, especially the model and evaluation sections: https://arxiv.org/abs/2511.18906
- NVIDIA's A100-80GB supported profile information: https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-mig-profiles.html
