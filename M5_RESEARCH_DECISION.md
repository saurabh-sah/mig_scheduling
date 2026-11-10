# Milestone 5: What this experiment can and cannot tell us

## Purpose

Milestone 5 does not add another scheduler. It measures the current policies and diagnoses the adaptive policy's behavior before we spend more effort on a new algorithm.

## Research questions

1. Does fixed-budget lookahead change accepted workload counts versus MFI on identical job streams?
2. Does adaptive lookahead invoke search often enough to matter, or mostly fall back to MFI?
3. When it invokes lookahead, does it choose a different placement from MFI?
4. Does adaptive lookahead reduce decision cost relative to always-on fixed lookahead?
5. Do these answers differ under uniform, skew-small, skew-big, bimodal, and piecewise regime-shift workloads?

## What a meaningful result looks like

Do not decide based on one best acceptance rate. Examine, for each scenario:

- paired acceptance-rate difference versus MFI by seed;
- paired 95% interval as an exploratory uncertainty measure;
- mean decision time and total lookahead rollouts;
- lookahead-use and MFI-fallback rates;
- why fallback happened;
- how often selected placements differ from MFI;
- acceptance broken down by requested profile.

Adaptive lookahead is promising only if it achieves a useful acceptance/runtime trade-off on held-out seeds, not just because it reduces the number of rollouts.

## Gate interpretation

- If lookahead is almost never invoked, the gate is too conservative or its trigger is not useful. Check fallback reasons before changing thresholds.
- If lookahead is invoked frequently but nearly always chooses the MFI placement, its estimated value may not justify its runtime.
- If lookahead changes decisions but not acceptance, it may be changing placement without improving the selected objective.
- If acceptance increases but runtime rises sharply, compare the method against fixed-lookahead at matched runtime budgets.
- If results reverse after a regime shift, inspect forecast reliability and adaptation delay.
- If no method consistently beats MFI, that is a legitimate finding; report it honestly and revise or abandon the proposed contribution.

## Important caveats

1. A regime-shift workload is a controlled synthetic stress test, not a trace from a production cluster.
2. The simulator is an abstraction of the profile/placement model, not a driver-validated hardware emulator.
3. Forecast reliability is a heuristic based on distribution stability; it is not the probability that a forecast is correct.
4. The adaptive stopping rule uses an approximate paired interval and has not been calibrated as a formal sequential statistical procedure.
5. Wall-clock timing depends on the execution environment. Per-decision timing should be measured across repeated runs on the same machine.
6. Use held-out seeds after tuning. Do not tune the adaptive thresholds on final evaluation seeds.
7. The `Best Index` baseline is an explicit operationalization of the paper's prose, not the authors' original source code.
8. These pilot runs support engineering decisions, not state-of-the-art claims.

## Go/no-go decision after this milestone

Continue toward a research contribution only if (a) the literature review confirms a specific gap, (b) the baseline is defensibly validated, and (c) repeated matched experiments show a useful improvement or a clear, publishable trade-off. Otherwise keep the reusable simulator and change the question rather than adding arbitrary algorithmic complexity.
