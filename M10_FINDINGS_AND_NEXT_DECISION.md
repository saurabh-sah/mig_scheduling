# M10 Findings and Next Decision

## What was built

- A source-to-code conformance matrix for the MFI paper's profile table, profile probabilities, load axis, arrival and duration assumptions, fragmentation score, baselines, metrics, and repetition count.
- Machine-readable protocol configs for the paper-candidate interpretation and separate dynamics stress tests.
- An experiment runner that holds the profile sequence and job IDs fixed within each `(scenario, seed)` and varies only arrival timing and/or service durations.
- 1,200 policy executions in the included exploratory study: 4 distributions × 10 seeds × 6 modes × 5 policies. Each run records 4 offered-load checkpoints (4,800 checkpoint rows total).
- Source profile/distribution audit, raw CSVs, aggregate tables, paired MFI-minus-BF-BI differences, descriptive bootstrap intervals, plots, and a manifest.

## Main finding

Under the `paper_candidate` mode in the included 20-GPU, 10-repetition experiment, **MFI and BF-BI tied on acceptance in every repetition at all tested final offered-load summaries**. This persists across uniform, skew-small, skew-big, and bimodal distributions. That means this particular implementation and protocol still do not create the performance gap we expected from the MFI paper's figures.

This is a diagnostic result, not a claim about the original paper. It indicates our implementation is not yet sufficiently discriminative to use for an algorithm-improvement claim.

## Dynamics sensitivity

At the final offered-load checkpoint, the mean arrival-snapshot compute/memory occupancy under `paper_candidate` was approximately:

| Distribution | Compute occupancy | Memory occupancy | MFI / BF-BI acceptance |
|---|---:|---:|---:|
| Uniform | 47.2% | 48.0% | 100% / 100% |
| Skew-small | 41.9% | 44.9% | 100% / 100% |
| Skew-big | 50.4% | 50.6% | 100% / 100% |
| Bimodal | 50.3% | 50.1% | 100% / 100% |

The stress modes show why semantics matter:

- **Burst arrivals / no completion before stream end:** peak/final arrival occupancy is near 85–94% of the modeled dimensions; small MFI–BF-BI differences appear in some repetitions.
- **Sparse arrivals:** much lower concurrent occupancy (roughly 22–28% at the final checkpoint) because jobs finish between arrivals.
- **Short lifetimes:** lower occupancy (roughly 12–17%).
- **Long lifetimes:** higher occupancy (roughly 76–86%).

The profile stream is kept identical within each seed, so these differences are attributable to the arrival/lifetime transformation in this simulator. These modes are **stress tests, not evidence that the paper uses any of these alternatives**.

## What the result does and does not mean

It does mean that:

1. The `Slice No.` offered-load axis should remain distinct from current compute/memory occupancy.
2. The interpreted value of `T` and the arrival schedule materially affect resource dynamics.
3. The current baseline comparison is not yet a trustworthy test bed for claiming an MFI improvement.

It does not mean that MFI's published findings are wrong, that our two-resource model is wrong, or that changing the protocol to produce a gap would be acceptable.

## Research decision

**Hold new scheduler development.** Before Milestone 11, prioritise:

1. Inspect and document the exact `r_w(p)` / free-slice gate semantics in the MFI fragmentation-score pseudocode and compare them to our physical dual-resource interpretation.
2. Audit the BF-BI/WF-BI GPU ranking function: the paper gives prose for the resource residual but does not fully define the scalarization across compute and memory resources. Keep the current formula labelled as an implementation choice.
3. If possible, request the authors' executable source or clarification on the definition of `T`, the slot-arrival model, and the resource residual scalarization.
4. Build a small trace-by-trace conformance test with a manually documented legal MIG placement state and then make the model/protocol choice explicit.
5. Only resume lookahead research once the paper-candidate baseline has a sufficiently discriminative and explainable evaluation.

## Reproducibility

Included run: `results/m10_protocol_dynamics_20gpu_10rep/`

- `m10_manifest.json` — seeds, policies, modes, and assumptions.
- `m10_raw_checkpoints.csv` — raw checkpoint rows for all policy runs.
- `m10_summary.csv` — means and standard deviations.
- `m10_paired_mfi_minus_bfbi.csv` — paired deltas and bootstrap intervals.
- `M10_PROTOCOL_DYNAMICS_REPORT.md` — generated report.
- `m10_acceptance_modes_*.png` — policy acceptance by mode and scenario.
- `m10_paired_delta_by_mode.png` — paired MFI–BF-BI differences by mode.

The structural source audit passes and the software test suite passes. This is still an independent simulator, not an NVIDIA driver emulator or exact reproduction of the unpublished original experiment code.
