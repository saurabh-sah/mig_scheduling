# Project Status — Milestone 14 (Final)

## Milestone 14 Status: COMPLETE

**Large-scale empirical evaluation of lookahead and baseline policies with GPU-diverse candidate pruning, publication manuscript generation, and final scientific conclusions.**

## Repository Status

- [x] **Dual-Resource MIG Architecture:** 7 compute slices, 8 memory slices, separate occupancy bitvectors, and six legal A100-80GB profiles modeled faithfully.
- [x] **Baseline & Paper Conformance:** Paper-path First-Fit, Round-Robin, Best-Fit Best-Index, and Worst-Fit Best-Index implemented and verified against Figure 3 rejection semantics.
- [x] **Figure-3 Conformance Resolution:** Mathematical scoring discrepancy resolved: `free_anchor_starts` reproduces published scores $F(1)=8, F(2)=16$, whereas `all_legal_starts` overcounts occupied anchors ($F(1)=14, F(2)=23$).
- [x] **Bounded Lookahead & Candidate Pruning:** Fixed-budget ($H=2, R=3$) and adaptive-budget lookahead implemented with GPU-diverse pruning ($K_{\text{max}} = 5$) and common random numbers.
- [x] **Large-Scale Benchmark:** 20 GPUs, 160 matched workload traces, 1,600 simulations, 10 policies, across progressive and burst arrival regimes.
- [x] **Automated Testing & Validation:** 119 tests passing (`pytest -q`), `scripts/validate_baseline.py` passes all consistency checks.
- [x] **Publication Manuscript:** 8-page IEEE conference paper compiled in `paper/mig_scheduling_ieee.pdf`.

---

## Final Scientific Conclusions

1. **Online Lookahead Yields No Statistically Meaningful Acceptance Gain at Scale:**
   - *Progressive streaming load:* Greedy MFI achieves 99.79% (free-anchor) and 99.91% (all-starts). Lookahead matches MFI identically (0 wins, 0 losses, 80 ties; paired delta 0.00 pp, bootstrap 95% CI [0.00, 0.00]).
   - *Burst load:* Fixed lookahead achieves 73.00% vs. 72.88% for free-anchor MFI (+0.12 pp, bootstrap 95% CI [-0.05, +0.34] pp, crossing zero). Adaptive lookahead achieves 73.12% (+0.24 pp, 6 wins, 2 losses, 72 ties).
2. **Computational Overhead Is Prohibitive for Online Deployments:**
   - Greedy MFI decides in **0.5–0.7 ms** per job; classical baselines decide in **0.03–0.18 ms**.
   - Lookahead requires **24–40 ms** per decision (a **50×–70× latency inflation**; P99 latency exceeding 170–214 ms; total CPU time 130–152 s vs. 2.5–3.0 s).
3. **Score Interpretation Alters Placements Without Changing Acceptance:**
   - Switching between `all_legal_starts` and `free_anchor_starts` alters **74.6%** of placement decisions in progressive load and **52.8%** in burst load, yet yields negligible acceptance difference (-0.12 pp in progressive, ~1.0 pp in burst).
4. **Spreading Beats Packing Under Concurrent Burst Arrivals:**
   - Under simultaneous arrivals (inter-arrival gap 0.0), Worst-Fit Best-Index (`paper_wf_bi`) achieves **76.51%** acceptance, outperforming MFI's **72.88%** by **+3.63 pp** (bootstrap 95% CI [+2.47, +4.79] pp; 55 wins / 17 losses / 8 ties).
   - *Mechanistic cause:* Greedy bin-packing packs early arrivals densely onto few GPUs, destroying contiguous 4-slice and 7-slice anchor regions. Load-spreading distributes early jobs across distinct GPUs, preserving contiguous headroom across the cluster.

---

## Final Research Recommendations

1. **Reject Online Lookahead:** Abandon online Monte Carlo rollout search in production schedulers due to negative operational ROI.
2. **Maintain Greedy MFI as Online Standard:** Greedy MFI (`mfi_free_anchor_score`) remains the superior general-purpose online heuristic for streaming arrivals, achieving Pareto optimality in latency and acceptance.
3. **Adopt Load-Spreading in Burst Regimes:** Multi-tenant clusters subject to batch or gang arrivals should employ Worst-Fit Best-Index or implement regime-aware meta-schedulers that switch from MFI to Worst-Fit during arrival spikes.
4. **Retain Framework for Offline Benchmarking:** The simulator, lookahead rollouts, and exact offline dynamic programming oracle provide valuable tools for offline trace analysis and theoretical bounding.

---

## Artifacts & Links

- Synthesis Document: [`M14_FINDINGS_AND_NEXT_DECISION.md`](M14_FINDINGS_AND_NEXT_DECISION.md)
- Experiment Report: [`results/m14_large_scale_eval_20gpu_20rep/M14_EXPERIMENT_REPORT.md`](results/m14_large_scale_eval_20gpu_20rep/M14_EXPERIMENT_REPORT.md)
- Manuscript PDF: [`paper/mig_scheduling_ieee.pdf`](paper/mig_scheduling_ieee.pdf)
- Manuscript Typst Source: [`paper/mig_scheduling_ieee.typ`](paper/mig_scheduling_ieee.typ)
