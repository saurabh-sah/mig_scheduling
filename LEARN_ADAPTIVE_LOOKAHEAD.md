# Learn the Adaptive Scheduler: Guided Walkthrough

This note is a learning companion, not a replacement for reading the code.

## Learning objective

You should be able to answer four questions after reading this milestone:

1. When does the scheduler decide not to use lookahead?
2. How is recent workload-distribution stability estimated?
3. How are future-placement candidates scored?
4. How does the algorithm stop spending computation?

## 1. Read in this order

1. `src/mig_scheduler/context.py` — what information is visible at a scheduling decision.
2. `src/mig_scheduler/schedulers.py` — `AdaptiveBudgetLookaheadScheduler` constructor and `choose_with_context`.
3. `_candidate_placements` — same resource legality and MFI score as the existing baseline.
4. `forecast_reliability` — profile frequencies in adjacent halves of the recent history.
5. `_shortlist_candidates` — limits the number of candidate placements evaluated.
6. The rollout loop in `choose_with_context` — creates a batch of futures and applies the same futures to each candidate.
7. `_winner_is_separated` — paired-difference stopping rule.
8. `src/mig_scheduler/simulator.py` — how diagnostic fields are saved in job-level output.
9. `tests/test_adaptive_lookahead.py` — examples of expected behaviour.

## 2. Questions to answer in your own notes

- Why is MFI always a fallback?
- Why are future jobs generated inside the scheduler rather than passed from the simulator?
- What is the purpose of common random numbers?
- Why do we sample durations for running jobs rather than read their true completion times?
- Why can a profile-distribution shift lower reliability?
- What is the difference between `candidate_pool_size` and `candidate_count`?
- What happens if no candidate is statistically separated before the budget cap?
- Why is the reported 95% interval only an approximate stopping heuristic?
- Can a small candidate shortlist miss the best future-acceptance placement?
- Why must the same random workload be used across policies for a matched comparison?

## 3. First runs

Run a clean smoke test:

```powershell
python -m mig_scheduler.cli --policy adaptive_lookahead --scenario skew_small --gpus 2 --jobs 40 --seed 42 --lookahead-horizon 2 --adaptive-min-rollouts 2 --adaptive-max-rollouts 6 --adaptive-batch-size 2 --adaptive-min-history 12 --adaptive-profile-window 24 --adaptive-min-reliability 0.2 --adaptive-max-candidates 4 --out results\m4_adaptive_smoke
```

Inspect `results/m4_adaptive_smoke/jobs.csv` and answer:

- What fraction of accepted jobs used lookahead?
- What fraction fell back to MFI?
- Which fallback reason was most common?
- What was the average and maximum rollout count per decision?
- Was the candidate selected by lookahead the same as the recorded MFI candidate?

Then compare policies on matched traces:

```powershell
python scripts\run_adaptive_matrix.py --jobs 40 --gpus 2 --seeds 1,2 --horizon 2 --fixed-rollouts 2 --adaptive-min-rollouts 2 --adaptive-max-rollouts 6 --adaptive-batch-size 2 --adaptive-min-history 12 --adaptive-profile-window 24 --adaptive-min-reliability 0.2 --adaptive-max-candidates 4 --out results\m4_smoke_matrix.csv
python scripts\analyze_adaptive_matrix.py results\m4_smoke_matrix.csv --out results\m4_smoke_summary.csv
```

These are smoke-test parameters, not research defaults. A small run only verifies that the code executes; it cannot establish scheduler superiority.

## 4. Make one controlled change at a time

Try changes to `minimum_reliability`, `ambiguity_margin`, `max_candidates`, and the rollout budgets separately. Do not tune them against your final test seeds. Reserve separate workload seeds for tuning and final evaluation.

## 5. What would falsify the hypothesis?

Examples:

- Adaptive lookahead rarely triggers, so it behaves almost exactly like MFI.
- It triggers often but does not increase acceptance.
- It improves acceptance but costs substantially more scheduling time than fixed lookahead.
- The forecast stability score is unreliable after workload changes.
- The apparent gain disappears across repeated matched seeds.

These are valid results. Record them rather than changing evaluation settings until the method appears better.
