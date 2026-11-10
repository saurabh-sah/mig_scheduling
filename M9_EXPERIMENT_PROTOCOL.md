# M9 Experiment Protocol

## A. Best-Index conformance check

```powershell
python scripts\audit_m9_baselines.py --out results\m9_index_order_audit.json
```

This enumerates all compute/memory occupancy bitmask pairs and verifies whether the prior M8 static blocking-cost index ranking matches the explicit reference order from Turkkan et al. (2024). It is an exhaustive property check of this software abstraction, not a driver-level hardware test.

## B. Arrival-spacing sensitivity

Quick local run:

```powershell
python scripts\run_m9_arrival_spacing_sweep.py --gpus 20 --repetitions 3 --gaps 0,0.1,0.5,1.0 --out results\m9_gap_quick
python scripts\analyze_m9_arrival_spacing_sweep.py --input-dir results\m9_gap_quick
```

Recreate the included 100-GPU exploratory run:

```powershell
python scripts\run_m9_arrival_spacing_sweep.py --gpus 100 --repetitions 2 --seed 2026 --gaps 0,0.1,0.5,1.0 --out results\m9_arrival_spacing_audit
python scripts\analyze_m9_arrival_spacing_sweep.py --input-dir results\m9_arrival_spacing_audit
```

For a fixed scenario/repetition seed, the workload profile sequence and service durations are generated once and then arrival times are scaled by `arrival_gap`. Gap 0 is a burst stress case, not a realistic inter-arrival process.

## C. Load-basis sensitivity

Quick local run:

```powershell
python scripts\run_m9_load_basis_sweep.py --gpus 20 --repetitions 3 --out results\m9_load_basis_quick
python scripts\analyze_m9_load_basis_sweep.py --input-dir results\m9_load_basis_quick
```

Recreate the included 100-GPU exploratory run:

```powershell
python scripts\run_m9_load_basis_sweep.py --gpus 100 --repetitions 2 --seed 9026 --out results\m9_load_basis_audit
python scripts\analyze_m9_load_basis_sweep.py --input-dir results\m9_load_basis_audit
```

The load bases are:

- `paper_slice_units`: MFI-inspired scalar Slice budget;
- `compute`: cumulative requested compute slices;
- `memory`: cumulative requested memory slices;
- `max_normalized`: whichever of normalized compute or memory demand reaches its capacity first.

Every load-basis experiment is a burst stress test (arrival gap 0). Do not compare it as if it were the paper's original arrival-time protocol.

## Reporting rules

- The included runs use two repetitions per scenario/basis or scenario/gap and are exploratory.
- Report raw run-level outcomes and the configuration manifest.
- Keep cumulative offered demand separate from concurrent occupancy.
- Call compute and memory measurements modeled occupancy, not physical GPU utilization.
- Do not make an algorithmic novelty/performance claim from this audit.
