# Milestone 3 Method: Fixed-Rollout Lookahead

## Purpose

Milestone 3 adds a comparison policy to the existing simulator. It is **not** the proposed adaptive/uncertainty-aware contribution yet. The aim is to measure whether finite-horizon search adds value over MFI and what computational overhead it incurs.

## Online information boundary

At a scheduling decision, the policy receives:

- the current GPU allocation states;
- past workload arrivals (arrival time and requested profile);
- service durations for jobs that have already completed;
- currently running jobs' profile, arrival time, and GPU, but **not** their actual completion time or duration.

The policy does not receive future generated jobs or the current job's true duration. This is intentional: those fields are known to the simulator but should not leak into the scheduler's forecasts.

## Forecast model

For each profile \(p\), estimate its next-arrival probability with additive smoothing:

\[
\hat q_p = \frac{n_p + \eta}{N + K\eta}
\]

where \(n_p\) is the number of previously observed arrivals requesting profile \(p\), \(N\) is the number of past arrivals, \(K\) is the number of supported profiles, and \(\eta>0\) is the pseudocount. The implementation uses \(\eta=1\) by default, so the cold-start forecast is uniform.

The mean inter-arrival interval is estimated from positive gaps between observed arrivals. If no positive gap exists, a configurable prior is used. Service durations are sampled from completed historical durations for the same profile; when none exist, the model falls back to the pool of all completed durations, then to a configurable duration prior.

For an active job, the scheduler samples a predicted residual lifetime conditional on a historical duration exceeding the job's elapsed age. If no such observation exists, it uses the configured residual-duration prior. This fallback is intentionally simple and should be tested as a sensitivity parameter in future work.

## Candidate scoring

For the current request, enumerate every legal feasible placement \(a\). Each placement creates a hypothetical state \(S_a\). For each candidate, simulate \(R\) sampled future traces of \(H\) future arrivals. In each rollout:

1. Clone the current cluster state.
2. Apply the candidate placement.
3. Give running jobs and the candidate job sampled completion estimates.
4. Generate future profile arrivals, inter-arrivals, and durations from the online model.
5. Release predicted-completed jobs before each future arrival.
6. Use MFI for each simulated future scheduling decision.
7. Count accepted future jobs.

The estimated downstream value is:

\[
\hat V_H(a)=\frac{1}{R}\sum_{r=1}^{R}Y_r(a)
\]

where \(Y_r(a)\) is the count of future jobs accepted in rollout \(r\) after the current placement \(a\). The current job is not included in this count because it is feasible under all candidates.

Choose the candidate with the highest estimated value. Exact ties are broken by smaller immediate MFI fragmentation increment, then GPU ID and start index. The MFI action is recorded for analysis.

## Common random numbers

The same sampled rollout plans are used to score all current candidate placements within one decision. This reduces variance in pairwise candidate comparison because each candidate sees the same sampled future workload traces.

## Budget meaning

`--lookahead-rollouts R` fixes the number of rollouts **per feasible candidate**, and `--lookahead-horizon H` fixes the number of future arrivals per rollout. The total rollout count per scheduling decision is therefore `candidate_count * R`; it is not a fixed global wall-clock budget. This makes the baseline easy to implement and compare, but motivates the next milestone: adaptively allocating a total search budget across candidates.

## Limitations

- The simulator is an abstract profile/placement model, not driver-validated MIG emulation.
- The service-time model is based on completed history and simple priors.
- Future profile demand is estimated from prior arrivals; the method does not know the actual future workload.
- Lookahead calls the current MFI implementation inside each rollout, so its quality depends on that rollout policy.
- The candidate state is cloned for every evaluation, making computation grow with candidate count, horizon, and rollout count.
- The method is a baseline experiment, not evidence of research novelty or a demonstrated improvement.

## Main command

```powershell
python -m mig_scheduler.cli --policy fixed_lookahead --scenario skew_small --gpus 4 --jobs 100 --seed 42 --lookahead-horizon 3 --lookahead-rollouts 4 --out results\m3_demo
```

## Matched experiment matrix

```powershell
python scripts\run_lookahead_matrix.py --jobs 100 --gpus 4 --seeds 1,2,3 --horizon 3 --rollouts 4 --out results\m3_matrix.csv
python scripts\analyze_lookahead_matrix.py results\m3_matrix.csv --out results\m3_summary.csv
```

Use small settings first. The lookahead policy can be significantly slower than greedy baselines. Do not interpret a smoke test as a research result.
