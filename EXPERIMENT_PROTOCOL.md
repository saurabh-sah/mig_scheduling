# Experiment Protocol Index — M11

For the baseline conformance and ranking sensitivity work, use:

- `M11_MFI_RESOURCE_GATE_AUDIT.md` — what is source-supported and how the score gate is interpreted.
- `M11_BF_WF_RANKING_AUDIT.md` — the source-derived Best Index order, what the MFI paper leaves unspecified, and the sensitivity variants.
- `M11_FINDINGS_AND_NEXT_DECISION.md` — findings and go/no-go decision.

The included M11 experiment uses 20 modeled A100-80GB GPUs, 10 repetitions per MFI profile distribution, identical generated job traces across policies within each scenario/seed, a cumulative paper-style scalar offered-load target, and four residual-resource rankings under two tie hierarchies for both BF-BI and WF-BI. These are sensitivity tests, not exact reproduction results.

Do not use the best-performing sensitivity variant as the baseline without source justification or author clarification. Retain the current normalized-sum/index-before-GPU policy as the default until then.
