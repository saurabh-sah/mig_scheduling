# M12 — Baseline semantics after the Figure 3 audit

## 1. The bug in our earlier comparison

The earlier `BestFitBestIndexScheduler` and `WorstFitBestIndexScheduler` implementation built a candidate list from **all feasible placements across all GPUs** and selected the best candidate from that global list.

That is a sensible stronger heuristic, but it is not the failure mechanism illustrated in MFI Figure 3. There, the selected GPU has enough aggregate resources but cannot place the request at a legal MIG configuration, and the request is rejected even though another GPU could accommodate it. The paper's prose describes BF-BI/WF-BI as GPU-selection policies followed by a Best Index placement rule; Figure 3 demonstrates that the choice can fail under MIG placement constraints.

Therefore, our previous global-search implementation was too strong to label simply as the paper's BF-BI/WF-BI baseline.

## 2. M12 implementation split

The repository now distinguishes two algorithm families.

### Paper-path operationalizations

- `paper_ff`
- `paper_rr`
- `paper_bf_bi`
- `paper_wf_bi`

The M12 BF-BI/WF-BI policies first select one GPU among GPUs with sufficient aggregate compute and memory, using an explicit normalized residual score. They then search legal starts on that selected GPU in the profile-specific Best Index order. If the selected GPU has no feasible start, the request is rejected; the policy does not fall back to another GPU.

This behavior is consistent with the Figure 3 failure mechanism. It is still an operationalization, not a byte-for-byte reproduction of unavailable source.

### Global-feasible variants

- `global_feasible_bf_bi`
- `global_feasible_wf_bi`

These choose from the feasible placements across the whole cluster. They are retained as meaningful alternative heuristics, but should not be presented as the paper's BF-BI/WF-BI implementation.

## 3. Remaining baseline choices

The source paper states that BF-BI minimizes residual resources and WF-BI maximizes residual resources, but it does not provide an explicit scalar formula combining separate compute and memory residuals. M12 keeps the normalized-sum score:

\[
R(g,p)=\frac{C_g^{free}-C_p}{7}+\frac{M_g^{free}-M_p}{8}
\]

BF-BI chooses the smallest `R`; WF-BI chooses the largest. Ties resolve to the lowest GPU ID. Within the selected GPU, the algorithm selects the feasible legal start with the best published profile-specific index rank.

This is named and documented so it can be varied later, rather than presented as an author-confirmed detail.

## 4. Decision rules

For paper-path BF-BI/WF-BI:

1. Build the set of GPUs with enough aggregate compute and memory.
2. Rank those GPUs by the declared residual score.
3. Select one GPU.
4. Enumerate feasible legal starts **on that GPU only**.
5. If none exist, reject the request.
6. Otherwise choose the feasible start with the best index-preference rank and allocate.

For global-feasible BF-BI/WF-BI:

1. Enumerate all feasible `(GPU, start)` pairs cluster-wide.
2. Rank candidates by GPU residual score, then index rank and deterministic tie-breakers.
3. Select the best feasible candidate.

The distinction between steps 3–5 in the paper-path method and step 1–3 in the global-feasible method explains why the two policies can have very different acceptance rates.
