# M10 next-step decision

## Why this milestone comes before a new scheduler

M8–M9 showed that our current MFI/BF-BI acceptance gap is often negligible under the current offered-load and lifetime interpretation. This does not prove that MFI has no benefit. Before claiming that, we need to separate the paper's scalar offered-load axis from active physical occupancy and quantify sensitivity to job arrival and lifetime dynamics.

## New work in M10

1. Make the source-to-code conformance ledger explicit.
2. Add a single command for paired comparisons across one paper-candidate mode and five clearly labelled workload-dynamics stress modes.
3. Preserve the same profile sequence and job identifiers within each `(scenario, seed)` so timing/lifetime effects are isolated.
4. Record all arrival checkpoints, per-policy metrics, and paired MFI-minus-BF-BI deltas.
5. Bootstrap the paired mean delta descriptively and generate plots.
6. Keep raw CSV and a machine-readable run manifest.

## Go/no-go criteria

**Go to algorithm research only after:**
- profile/placement legality tests pass;
- the MFI and baselines are explainable line by line;
- the paper-candidate protocol is frozen and documented;
- stress-test modes remain separate from the reproduction-oriented result;
- the exact small-run outputs can be regenerated from fixed seeds.

If paper-candidate behavior remains different from the paper's figures, the next action is to isolate the unresolved implementation choices (especially BF/WF GPU scoring and the interpretation of `T`) and, if possible, ask the authors for code or clarification. Do not make changes solely to create a bigger performance gap.

## Not claimed

M10 does not claim exact reproduction, a new scheduling method, a statistically validated MFI improvement, or hardware performance. It is an audit and sensitivity experiment.
