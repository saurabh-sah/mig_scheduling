# M6 Research Decision: What the Current Evidence Says

## Why this milestone exists

Milestone 5 suggested that fixed/adaptive lookahead incurs substantial scheduling overhead without a dependable acceptance gain over MFI. Before adding any more algorithmic components, M6 introduces (1) an exact offline acceptance oracle for tiny traces and (2) a rollout-budget frontier experiment.

## Primary question

How far are the current online methods from the optimal acceptance count on tiny traces, and does spending additional scheduling time on lookahead close that gap enough to justify the cost?

## Required evidence

1. Unit-test and cross-check the offline oracle against independent brute-force enumeration on very small traces.
2. Verify that the oracle's accepted count is never below the accepted count of any tested online policy on the same trace.
3. Sweep fixed lookahead rollout budgets and adaptive maximum budgets on identical workload traces.
4. Report acceptance and actual scheduling runtime together, plus per-scenario paired differences versus MFI.
5. Retain raw CSVs, configs, seeds, and generated plots.

## Decision rules

- **Continue with adaptive lookahead** only if it produces a reproducible acceptance improvement or reaches comparable acceptance at lower measured decision cost than a fixed-budget alternative, on held-out traces.
- **Modify the method** if the offline oracle shows an appreciable gap but lookahead is unable to recover it. Inspect whether the gap is due to the prediction model, too-short horizon, candidate shortlisting, or objective mismatch.
- **Pivot the research question** if online policies nearly saturate the oracle on meaningful traces, or if improvements require disproportionate decision cost. Potential pivots include selective optimization (which must be carefully differentiated from OrigaMIG) or a different operational objective such as migration-aware recovery.
- Do not claim a universal improvement based on an isolated seed or scenario.

## Experimental guardrails

- The oracle is offline and is never counted as a deployable scheduler.
- Use paired workloads: same arrival times, profiles, and durations across policies.
- Separate algorithm tuning from held-out evaluation seeds.
- Show scenario-level results as well as pooled summaries.
- Runtime is implementation- and machine-dependent; interpret it as an internal comparison, not production latency.
- The model's legal placement table must be cited and versioned; this remains a software abstraction, not a CUDA/NVIDIA driver emulator.
