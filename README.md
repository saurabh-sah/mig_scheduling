# MIG Scheduling Research

A research and simulation repository for studying online placement policies for NVIDIA Multi-Instance GPU (MIG) workloads. The project builds an executable profile-placement simulator, audits paper-oriented baseline semantics, studies two plausible interpretations of the MFI fragmentation score, and evaluates fixed/adaptive bounded-lookahead policies.

> **Scope:** This is a synthetic MIG profile/placement simulator. It is not a production Kubernetes scheduler, does not run jobs on a physical MIG GPU, and is not an exact reproduction of the cited paper's full evaluation protocol.

## Research question

Can fragmentation-aware online placement be improved with bounded lookahead, and are the observed results robust to the interpretation of the MFI fragmentation score and to workload arrival patterns?

## Start here

1. [`M13_FINDINGS_AND_NEXT_DECISION.md`](M13_FINDINGS_AND_NEXT_DECISION.md) — latest findings and the next research decision.
2. [`M13_README.md`](M13_README.md) — M13 implementation summary and reproduction instructions.
3. [`M13_SCORE_MODE_DESIGN.md`](M13_SCORE_MODE_DESIGN.md) — exact behavior of the `all_legal_starts` and `free_anchor_starts` score modes.
4. [`results/m13_score_aware_lookahead_4gpu_3rep/M13_EXPERIMENT_REPORT.md`](results/m13_score_aware_lookahead_4gpu_3rep/M13_EXPERIMENT_REPORT.md) — M13 experiment report.
5. [`M12_FINDINGS_AND_NEXT_DECISION.md`](M12_FINDINGS_AND_NEXT_DECISION.md) and [`M12_FIGURE3_CONFORMANCE.md`](M12_FIGURE3_CONFORMANCE.md) — baseline and Figure 3 conformance audit.
6. [`presentation/`](presentation/) — professor-facing PowerPoint, its slide-by-slide generation brief, and PDF preview.

## Repository layout

- `src/` — simulator, MIG profile model, placement logic, scheduling policies, and score implementations.
- `tests/` — unit and regression tests for scheduling semantics and score/lookahead consistency.
- `scripts/` — experiment runners, validation, and plotting utilities.
- `configs/` — experiment configuration files.
- `results/` — raw runs, per-job decisions, summaries, manifests, reports, and figures from prior milestones.
- `M*_*.md`, `CHANGELOG*.md` — milestone methods, audits, findings, and research decisions.
- `presentation/` — generated presentation assets.

## Setup and validation

Use Python 3.10+ (the project metadata specifies the supported version range):

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test,analysis]"
python -m pytest -q
python scripts/validate_baseline.py
```

The milestone validation checks implementation consistency, including profile-model checks, a memory-overlap regression case, and deterministic randomized MFI argmin consistency. Passing these checks does not establish replication of published performance results.

## Reproducing M13

The recorded M13 experiment can be rerun with:

```bash
python scripts/run_m13_score_aware_lookahead.py \\
  --gpus 4 \\
  --repetitions 3 \\
  --seed 20261026 \\
  --out results/m13_score_aware_lookahead_4gpu_3rep
```

The CLI exposes `--mfi-score-mode {all_legal_starts,free_anchor_starts}`. The default CLI mode is `free_anchor_starts`, retained as the Figure-3-conformance candidate; the alternate mode remains available for sensitivity analysis. Consult `M13_RUN_ME_FIRST.md` for full reproduction instructions and environment details.

## Current evidence — interpret cautiously

- M12's reconstructed Figure 3 states yield fragmentation scores 8 and 16 under `free_anchor_starts`. This is a conformance clue, not proof of the authors' exact implementation.
- M13 compares 10 policy variants over 240 simulations and 24 paired synthetic workload traces, including progressive-load and simultaneous-arrival burst conditions.
- Score interpretation changes roughly 55% of lookahead decision tuples in the recorded M13 study.
- Fixed lookahead adds materially more decision latency than greedy MFI in this small setup.
- The current experiments do **not** establish a robust acceptance improvement from lookahead. Larger paired experiments are needed before making that claim.

Always quote the per-experiment reports and manifests for exact settings and figures. Do not present simulated decision latency as GPU execution latency or the results as a physical-cluster benchmark.

## Source paper

Zambianco, Fasol, and Doriguzzi-Corin, *An Online Fragmentation-Aware GPU Scheduler for Multi-Tenant MIG-based Clouds*, arXiv:2511.18906: <https://arxiv.org/abs/2511.18906>.

## Presentation

The `presentation/` folder contains:

- `MIG_Scheduling_Project_Presentation.pptx` — editable 29-slide presentation with speaker notes.
- `MIG_Scheduling_Project_Presentation_Brief.docx` — slide-by-slide source brief used to generate the deck.
- `MIG_Scheduling_Project_Presentation.pdf` — rendered PDF preview.

## Research status

M13 is an evaluation and consistency milestone, not a claim that a new policy has been proven superior. The next recommended work is a larger paired evaluation with more independent traces, retaining both MFI score modes and reporting acceptance, decision latency, and adaptive-search activation/fallback rates.
