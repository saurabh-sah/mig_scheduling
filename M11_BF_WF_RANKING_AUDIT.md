# M11 — BF-BI / WF-BI Ranking Rule Audit

## What the paper specifies

The MFI paper defines:

- **BF-BI:** a bin-packing strategy that selects GPUs to minimize remaining available resources after allocation.
- **WF-BI:** a load-balancing strategy that selects GPUs to maximize remaining available resources after allocation.
- **Best Index (BI):** a profile-specific start-index preference order that tries to preserve the locations needed by profiles with fewer legal placements.

The index preference policy is described in Section VI of MFI. MFI cites Turkkan et al. (2024), which provides the index preference order in Table 1 and explains in Section 3.2.2 how the authors empirically derived it using `nvidia-smi`.

Sources:

- MFI Section VI: https://arxiv.org/abs/2511.18906
- Turkkan et al., Table 1 and Section 3.2.2: https://arxiv.org/abs/2409.06646

## Preference order used

The current profile-specific order is transcribed from Turkkan et al.'s A100-80GB Table 1:

| Profile | Legal starts in preference order |
|---|---|
| `7g.80gb` | 0 |
| `4g.40gb` | 0 |
| `3g.40gb` | 4, 0 |
| `2g.20gb` | 4, 0, 2 |
| `1g.20gb` | 6, 4, 0, 2 |
| `1g.10gb` | 6, 4, 5, 0, 1, 2, 3 |

The exact order is tested by `test_reference_preference_orders_cover_every_legal_start_once` and related tests. The audit scheduler's default setting was also compared with the Milestone 10 BF-BI/WF-BI classes across 300 deterministic randomized occupancy cases for each strategy; all choices matched.

## What remains unspecified in MFI's prose

The cluster model has two resource dimensions, compute and memory. MFI says BF-BI minimizes, and WF-BI maximizes, the amount of available resources remaining after allocation, but the text does not state an explicit scalar formula for combining compute and memory residuals. Nor does it spell out cross-GPU tie handling when multiple GPUs have equal residual scores.

Milestone 10 operationalized the resource residual as:

\[
R_{sum}=\frac{C^{free}-C_p}{7}+\frac{M^{free}-M_p}{8}
\]

and its candidate tuple puts the Best Index rank before the GPU ID when residual scores tie. Both are implementation choices, not explicit formulas in MFI.

## Sensitivity variants implemented

`src/mig_scheduler/baseline_sensitivity.py` keeps the existing default class untouched and provides explicit audit variants:

1. `normalized_sum`: sum of post-allocation compute and memory residual fractions (the existing operationalization).
2. `normalized_max`: maximum of the two post-allocation residual fractions.
3. `compute_first`: lexicographic compute residual, then memory residual.
4. `memory_first`: lexicographic memory residual, then compute residual.

Two deterministic tie hierarchies are tested:

- `index_before_gpu`: BI rank first, then GPU ID (existing implementation ordering).
- `gpu_before_index`: GPU ID first among equal resource scores, then BI rank within the chosen GPU (a staged GPU-then-index interpretation).

For BF-BI, the selected resource score is minimized; for WF-BI it is maximized. The Best Index rank is then applied as a deterministic tie-breaker.

These variants are **sensitivity tests**, not claims that one particular mode is specified by the paper. We retain the current default rather than selecting an alternative just because it looks better on our own traces.

## Exploratory results

The included study uses 20 modeled GPUs, 10 repetitions for each of the four MFI profile distributions, and the same generated workload trace for every policy within a scenario/seed pair: 680 policy runs total.

- MFI and all eight BF-BI variants accept 100% of jobs for all four distributions in this particular synthetic experiment.
- The current WF-BI default averages 98.12% acceptance over the four scenario means; all WF-BI ranking variants are very close in acceptance.
- Alternative scalarizations/tie rules do change some per-job GPU/index choices. Across the reported scenario-level decision-change summaries, the average fraction of job decisions differing from the current BF-BI default is approximately 3.40% for BF-BI variants and 2.29% for WF-BI variants.
- Because acceptance does not improve for BF-BI in this run and MFI itself accepts all generated jobs, this experiment does not establish any improved scheduling policy. It establishes that tie/resource-ranking choices can alter placement decisions without necessarily altering acceptance under a given workload/load regime.

These are synthetic simulator outcomes, not measurements on physical GPUs and not a reproduction of the original authors' code.
