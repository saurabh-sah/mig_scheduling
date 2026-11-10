# M13 — Score-Aware Lookahead Evaluation

## Research question
Does fixed or adaptive finite-horizon lookahead behave differently when the MFI score used for current placement and future rollout placements is switched from `all_legal_starts` to `free_anchor_starts`?

## Design
- 4 simulated GPUs; 3 repetitions for each of four profile distributions in two regimes.
- Progressive regime: 100% target offered load using the paper-slice-units basis and arrival gap 0.25.
- Burst regime: 150% target offered load and arrival gap 0.0; all positive-duration jobs arrive before any completion event.
- Ten policies, with the identical workload trace paired across all policies.
- Fixed lookahead uses horizon 2 and 3 rollouts per feasible current placement; adaptive lookahead uses horizon 2, 2–4 rollouts, a batch size of 2, and at most 5 candidates.
- `free_anchor_starts` is the primary MFI score candidate because it matches the Figure 3 score values for reconstructed states; `all_legal_starts` remains a sensitivity variant.

## Policy acceptance summary

| Regime | Policy | Mean acceptance | SD across paired workload traces | Mean decision time (ms) | Mean lookahead decisions/trace | Mean adaptive fallbacks/trace |
|---|---|---:|---:|---:|---:|---:|
| burst | `adaptive_lookahead__all_legal_starts` | 73.28% | 7.08 pp | 0.148 | 0.92 | 9.17 |
| burst | `adaptive_lookahead__free_anchor_starts` | 73.47% | 6.24 pp | 0.227 | 1.50 | 8.67 |
| burst | `fixed_lookahead__all_legal_starts` | 73.28% | 7.08 pp | 2.578 | 10.08 | 0.00 |
| burst | `fixed_lookahead__free_anchor_starts` | 72.54% | 7.94 pp | 2.537 | 10.08 | 0.00 |
| burst | `mfi_all_starts` | 73.28% | 7.08 pp | 0.033 | 0.00 | 0.00 |
| burst | `mfi_free_anchor_score` | 73.47% | 6.24 pp | 0.034 | 0.00 | 0.00 |
| burst | `paper_bf_bi` | 67.97% | 7.71 pp | 0.010 | 0.00 | 0.00 |
| burst | `paper_ff` | 68.33% | 10.26 pp | 0.008 | 0.00 | 0.00 |
| burst | `paper_rr` | 70.88% | 6.77 pp | 0.008 | 0.00 | 0.00 |
| burst | `paper_wf_bi` | 73.84% | 6.28 pp | 0.010 | 0.00 | 0.00 |
| progressive | `adaptive_lookahead__all_legal_starts` | 95.56% | 8.08 pp | 0.220 | 0.92 | 7.50 |
| progressive | `adaptive_lookahead__free_anchor_starts` | 94.17% | 11.20 pp | 0.323 | 1.50 | 6.83 |
| progressive | `fixed_lookahead__all_legal_starts` | 95.56% | 8.08 pp | 2.638 | 8.42 | 0.00 |
| progressive | `fixed_lookahead__free_anchor_starts` | 94.17% | 11.20 pp | 2.562 | 8.33 | 0.00 |
| progressive | `mfi_all_starts` | 95.56% | 8.08 pp | 0.037 | 0.00 | 0.00 |
| progressive | `mfi_free_anchor_score` | 93.70% | 11.05 pp | 0.039 | 0.00 | 0.00 |
| progressive | `paper_bf_bi` | 87.84% | 12.15 pp | 0.013 | 0.00 | 0.00 |
| progressive | `paper_ff` | 87.55% | 12.44 pp | 0.009 | 0.00 | 0.00 |
| progressive | `paper_rr` | 83.10% | 7.83 pp | 0.008 | 0.00 | 0.00 |
| progressive | `paper_wf_bi` | 87.71% | 9.87 pp | 0.011 | 0.00 | 0.00 |

## Paired acceptance difference vs. free-anchor MFI

Differences are policy acceptance minus `mfi_free_anchor_score`, computed on the same workload trace. Intervals are percentile bootstrap intervals over workload-level paired deltas; with only 12 traces per regime in the default run, they are exploratory, not definitive.

| Regime | Policy | Mean delta (percentage points) | Paired bootstrap 95% interval (pp) | Wins / losses / ties |
|---|---|---:|---:|---:|
| progressive | `paper_ff` | -6.16 | [-11.76, -1.39] | 0 / 4 / 8 |
| progressive | `paper_rr` | -10.60 | [-15.70, -5.95] | 0 / 9 / 3 |
| progressive | `paper_bf_bi` | -5.86 | [-13.66, +1.11] | 1 / 4 / 7 |
| progressive | `paper_wf_bi` | -5.99 | [-10.12, -2.22] | 0 / 5 / 7 |
| progressive | `mfi_all_starts` | +1.85 | [+0.00, +5.09] | 2 / 0 / 10 |
| progressive | `fixed_lookahead__all_legal_starts` | +1.85 | [+0.00, +5.09] | 2 / 0 / 10 |
| progressive | `fixed_lookahead__free_anchor_starts` | +0.46 | [+0.00, +1.39] | 1 / 0 / 11 |
| progressive | `adaptive_lookahead__all_legal_starts` | +1.85 | [+0.00, +5.09] | 2 / 0 / 10 |
| progressive | `adaptive_lookahead__free_anchor_starts` | +0.46 | [+0.00, +1.39] | 1 / 0 / 11 |
| burst | `paper_ff` | -5.14 | [-9.73, -1.45] | 0 / 5 / 7 |
| burst | `paper_rr` | -2.58 | [-6.42, +1.12] | 2 / 5 / 5 |
| burst | `paper_bf_bi` | -5.50 | [-11.01, +0.13] | 2 / 7 / 3 |
| burst | `paper_wf_bi` | +0.38 | [-3.50, +4.40] | 4 / 4 / 4 |
| burst | `mfi_all_starts` | -0.19 | [-3.80, +3.77] | 3 / 5 / 4 |
| burst | `fixed_lookahead__all_legal_starts` | -0.19 | [-3.76, +3.75] | 3 / 5 / 4 |
| burst | `fixed_lookahead__free_anchor_starts` | -0.93 | [-2.78, +0.00] | 0 / 1 / 11 |
| burst | `adaptive_lookahead__all_legal_starts` | -0.19 | [-3.66, +3.82] | 3 / 5 / 4 |
| burst | `adaptive_lookahead__free_anchor_starts` | +0.00 | [+0.00, +0.00] | 0 / 0 / 12 |

## Score-mode sensitivity

See `m13_score_mode_decision_sensitivity.csv` for per-workload changes in accept/reject and placement choices, and `m13_score_mode_decision_sensitivity.png` for the aggregate decision-change fractions. A decision change is a different `(status, gpu_id, start)` tuple for the same job, not necessarily a change in acceptance.

## Interpretation guardrails

1. Do not claim that lookahead improves MFI unless a paired acceptance interval and repeated workload results support that conclusion. A different placement is not automatically a better placement.
2. The two score definitions can change candidate rankings and future rollout placement decisions. Report both; do not hide the score definition in results or code.
3. Adaptive lookahead may fall back frequently; its search count, fallback reasons, and decision latency are recorded in the CSV outputs.
4. The burst workload intentionally suppresses completion-driven capacity reuse. It tests packing under simultaneous demand, not realistic production arrival behavior.
5. These synthetic four-GPU results are a debugging and hypothesis-screening milestone, not a general cloud-scheduling result or exact paper replication.

## Recommended next decision
Use these paired results to decide whether to keep lookahead in the main research contribution. If effects are small, unstable, or dominated by run-time cost, simplify the design and expand the workload/seed study before proposing novelty. If a consistent signal appears, run a larger held-out study (at least 20–30 seeds per distribution and regime) with separate calibration and test seeds.

## Files
- `m13_raw_runs.csv`: one row per policy/workload run.
- `m13_policy_summary.csv`: aggregate acceptance, utilization and decision-cost summary.
- `m13_paired_acceptance_vs_free_anchor_mfi.csv`: paired baseline deltas and bootstrap intervals.
- `m13_per_job_decisions.csv`: job-level decisions and lookahead diagnostics.
- `m13_score_mode_decision_sensitivity.csv`: per-trace changes between score modes.
- `m13_run_manifest.json`: full configuration and limitations.
