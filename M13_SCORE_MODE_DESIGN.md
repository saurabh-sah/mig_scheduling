# M13 Score-Mode Semantics

## Supported modes

- `all_legal_starts`: the prior project operationalization. It counts each blocked legal start when component-wise aggregate compute and memory capacity is sufficient.
- `free_anchor_starts`: the Figure-3-conformance candidate. It skips candidate starts whose anchor index is already occupied in either resource dimension, then checks the complete placement footprint. For reconstructed Figure 3 states, this yields the reported numeric scores 8 and 16.

Neither name claims to be the authors' confirmed executable implementation. `free_anchor_starts` is the main Figure-3-conformance candidate; `all_legal_starts` is retained as a required sensitivity case.

## Lookahead consistency contract

For both `FixedBudgetLookaheadScheduler` and `AdaptiveBudgetLookaheadScheduler`, `score_mode` now controls all score-based work:

1. calculating the fragmentation increment for every feasible current-placement candidate;
2. selecting the greedy MFI reference candidate and breaking rollout-value ties;
3. choosing future placements inside simulated rollouts;
4. selecting the greedy fallback when adaptive lookahead does not run or when no online context is supplied.

The mode can be passed in Python:

```python
from mig_scheduler.schedulers import FixedBudgetLookaheadScheduler

scheduler = FixedBudgetLookaheadScheduler(
    horizon=2,
    rollouts=3,
    seed=123,
    score_mode="free_anchor_starts",
)
```

For adaptive lookahead, the same argument is available:

```python
from mig_scheduler.schedulers import AdaptiveBudgetLookaheadScheduler

scheduler = AdaptiveBudgetLookaheadScheduler(
    horizon=2,
    min_rollouts=2,
    max_rollouts=4,
    batch_size=2,
    score_mode="free_anchor_starts",
)
```

The scheduler constructors retain `all_legal_starts` as their Python default for backward compatibility. The CLI defaults to the primary `free_anchor_starts` Figure-3-conformance candidate; use `--mfi-score-mode all_legal_starts` to request the sensitivity mode explicitly.

## Why the mode must be logged

The M13 experiment found that switching score mode changed about 55% of fixed-lookahead job decision tuples and 55–58% of adaptive-lookahead tuples in this sample. Acceptance often remained the same, and where it changed the direction depended on the workload regime. Mode choice is therefore a material experimental factor, not a hidden implementation detail.
