# MIG Scheduling Research

A research and simulation repository for studying online placement policies for NVIDIA Multi-Instance GPU (MIG) workloads. The project builds an executable profile-placement simulator, audits paper-oriented baseline semantics, resolves two plausible interpretations of the MFI fragmentation score, evaluates fixed and adaptive bounded-lookahead policies with GPU-diverse candidate pruning, and benchmarks 10 policies across a 20-GPU cluster (1,600 simulations).

> **Scope:** This is a synthetic MIG profile/placement simulator. It is not a production Kubernetes scheduler, does not run jobs on physical MIG GPUs, and is not an exact reproduction of the cited paper's full evaluation protocol.

## Research Question & Final Conclusion

*Can fragmentation-aware online placement be improved with bounded lookahead, and are the observed results robust to score interpretations and workload arrival patterns?*

**Final Scientific Conclusion:**
1. **Online Lookahead Yields No Meaningful Gain at Scale:** Across 160 paired workload traces (1,600 simulation runs) on a 20-GPU cluster, fixed and adaptive lookahead achieve identical acceptance to greedy MFI under progressive load (0 wins, 0 losses, 80 ties; paired delta 0.00 pp). Under burst load, lookahead achieves 73.00% vs. 72.88% (+0.12 pp, bootstrap 95% CI [-0.05, +0.34] pp, crossing zero).
2. **Computational Overhead Is Prohibitive:** Lookahead incurs a **50×–70× latency penalty** (~24–40 ms per decision vs. ~0.5–0.7 ms for greedy MFI; total CPU time 130–152 s vs. 2.5–3.0 s), even when restricted to the top-5 GPU-diverse candidates.
3. **Score Formulation Drives Decisions Without Changing Acceptance:** Filtering out occupied anchor indices (`free_anchor_starts`) reproduces published MFI Figure 3 scores ($F=8, 16$) while literal all-starts enumeration shifts 74.6% of placement choices with negligible acceptance impact (-0.12 pp).
4. **Spreading Beats Packing Under Burst Arrivals:** Under simultaneous concurrent arrivals (inter-arrival gap 0.0), Worst-Fit Best-Index (`paper_wf_bi`) achieves **76.51%** acceptance, outperforming MFI's **72.88%** by **+3.63 pp** (bootstrap 95% CI [+2.47, +4.79] pp, 55 wins / 17 losses / 8 ties). When arrival waves are simultaneous, spreading jobs across GPUs preserves contiguous multi-slice blocks across the cluster, avoiding catastrophic head-of-line blocking.

---

## Start Here

1. [`M14_FINDINGS_AND_NEXT_DECISION.md`](M14_FINDINGS_AND_NEXT_DECISION.md) — comprehensive synthesis of Milestone 14 findings, scaled 20-GPU benchmark, and final research recommendations.
2. [`results/m14_large_scale_eval_20gpu_20rep/M14_EXPERIMENT_REPORT.md`](results/m14_large_scale_eval_20gpu_20rep/M14_EXPERIMENT_REPORT.md) — full M14 experimental report, tables, and Pareto metrics.
3. [`paper/mig_scheduling_ieee.pdf`](paper/mig_scheduling_ieee.pdf) ([Source: `mig_scheduling_ieee.typ`](paper/mig_scheduling_ieee.typ)) — publication-grade, 8-page IEEE conference paper manuscript ("*On the Limits of Online Lookahead for Fragmentation-Aware MIG GPU Scheduling*").
4. [`M13_FINDINGS_AND_NEXT_DECISION.md`](M13_FINDINGS_AND_NEXT_DECISION.md) and [`M13_SCORE_MODE_DESIGN.md`](M13_SCORE_MODE_DESIGN.md) — score-aware lookahead design and sensitivity.
5. [`M12_FINDINGS_AND_NEXT_DECISION.md`](M12_FINDINGS_AND_NEXT_DECISION.md) and [`M12_FIGURE3_CONFORMANCE.md`](M12_FIGURE3_CONFORMANCE.md) — Figure 3 score reconstruction and baseline conformance.
6. [`presentation/`](presentation/) — 29-slide presentation deck with speaker notes and PDF preview.

---

## Repository Layout

- `src/` — simulator, MIG profile model, placement logic, scheduling policies, and score implementations.
- `tests/` — unit and regression tests for scheduling semantics, score modes, and lookahead consistency (119 automated tests passing).
- `scripts/` — experiment runners, validation, and plotting utilities.
- `paper/` — Typst source manuscript (`mig_scheduling_ieee.typ`), compiled PDF (`mig_scheduling_ieee.pdf`), and figures.
- `configs/` — experiment configuration files.
- `results/` — raw runs, per-job decisions, summaries, manifests, reports, and figures for all milestones.
- `presentation/` — generated presentation assets.

---

## Setup and Validation

Use Python 3.10+ (managed via `uv` or standard virtual environment):

```bash
# Run full automated test suite (119 tests)
uv run --with pytest pytest -q

# Run baseline consistency validation
uv run python scripts/validate_baseline.py
```

---

## Reproducing Milestone 14

The scaled 20-GPU benchmark (160 traces, 1,600 simulations) can be reproduced via:

```bash
uv run python scripts/run_m14_large_scale_eval.py \
  --gpus 20 \
  --repetitions 20 \
  --workers 16 \
  --out results/m14_large_scale_eval_20gpu_20rep
```

### Compiling the Research Paper

The IEEE conference manuscript is typeset in [Typst](https://typst.app) and can be recompiled directly:

```bash
uv run --with typst python -c "import typst; typst.compile('paper/mig_scheduling_ieee.typ', output='paper/mig_scheduling_ieee.pdf')"
```

---

## Source Paper

Zambianco, Fasol, and Doriguzzi-Corin, *An Online Fragmentation-Aware GPU Scheduler for Multi-Tenant MIG-based Clouds*, arXiv:2511.18906: <https://arxiv.org/abs/2511.18906>.
