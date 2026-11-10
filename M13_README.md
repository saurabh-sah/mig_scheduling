# MIG Scheduling Research — Milestone 13

**Milestone 13: Score-aware fixed and adaptive lookahead evaluation.**

M13 revisits the existing bounded lookahead policies after M12 identified two plausible MFI fragmentation-score interpretations. The implementation now keeps a score mode consistent across the current-placement ranking, hypothetical future placements, and adaptive fallback. The earlier score implementation remains available for sensitivity analysis.

## Start here

- `M13_FINDINGS_AND_NEXT_DECISION.md` — experiment results and next research recommendation.
- `M13_SCORE_MODE_DESIGN.md` — precise behavior of the two score modes and code examples.
- `M13_RUN_ME_FIRST.md` — environment, validation, CLI use, and commands to reproduce M13.
- `results/m13_score_aware_lookahead_4gpu_3rep/M13_EXPERIMENT_REPORT.md` — full generated report.
- `M12_FINDINGS_AND_NEXT_DECISION.md` and `M12_FIGURE3_CONFORMANCE.md` — prior baseline-conformance findings.

## What changed in M13

1. `FixedBudgetLookaheadScheduler` accepts a `score_mode` and uses it throughout current-candidate scoring and future rollouts.
2. `AdaptiveBudgetLookaheadScheduler` does the same, including its greedy fallback path.
3. The CLI includes `--mfi-score-mode {all_legal_starts,free_anchor_starts}` and defaults to the Figure-3-conformance candidate `free_anchor_starts`; the Python constructors keep the legacy all-starts default for backward compatibility.
4. New tests guard against a score-mode mismatch between current candidates and rollout placements.
5. The 240-run experiment compares 10 policies on 24 paired synthetic workload traces, with both progressive-load and simultaneous-arrival burst conditions.

## Main result

The score interpretation changes about 55% of lookahead decision tuples in this small study. Fixed lookahead adds roughly 2.6–2.8 ms of decision time compared with around 0.035–0.041 ms for greedy MFI. The experiment does not yet establish a robust acceptance benefit, so no such claim should be made.

## Validate and reproduce

```bash
python -m pip install -e ".[test,analysis]"
python -m pytest -q
python scripts/validate_baseline.py
python scripts/run_m13_score_aware_lookahead.py --gpus 4 --repetitions 3 --seed 20261026 --out results/m13_score_aware_lookahead_4gpu_3rep
```

## Source paper

Zambianco, Fasol, and Doriguzzi-Corin, *An Online Fragmentation-Aware GPU Scheduler for Multi-Tenant MIG-based Clouds*, arXiv:2511.18906: https://arxiv.org/abs/2511.18906.

The repository is a synthetic profile/placement simulator. It is not an exact reproduction of the paper's complete evaluation protocol and does not require a physical MIG GPU.
