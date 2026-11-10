"""Independent internal checks for MFI's argmin decision and dual-resource model."""
from __future__ import annotations

# Allow running directly from a source checkout before editable installation.
import sys
from pathlib import Path as _BootstrapPath
sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1] / "src"))

import random
from mig_scheduler.fragmentation import delta_fragmentation
from mig_scheduler.profiles import PROFILES, PROFILE_ORDER, validate_profile_model
from mig_scheduler.schedulers import MFIScheduler
from mig_scheduler.state import GPUState


def random_cluster(seed: int, gpu_count: int = 3) -> list[GPUState]:
    rng = random.Random(seed)
    gpus = [GPUState(i) for i in range(gpu_count)]
    next_id = 0
    for _ in range(rng.randint(0, 20)):
        gpu = rng.choice(gpus)
        profile = PROFILES[rng.choice(PROFILE_ORDER)]
        starts = gpu.feasible_starts(profile)
        if starts:
            gpu.allocate(f"v{next_id}", profile, rng.choice(starts))
            next_id += 1
    return gpus


def main() -> None:
    validate_profile_model()

    # Source-motivated regression case: a 1g.10gb at index 5 occupies memory
    # slice 5 and blocks a 1g.20gb at index 4, which requires memory slices 4–5.
    gpu = GPUState(0)
    gpu.allocate("existing-small", PROFILES["1g.10gb"], 5)
    assert 4 not in gpu.feasible_starts(PROFILES["1g.20gb"])

    # Verify the MFI policy returns the global argmin of delta F across every
    # feasible (GPU, placement) candidate for deterministic randomized states.
    checked = 0
    for seed in range(250):
        gpus = random_cluster(seed)
        rng = random.Random(seed + 9000)
        requested = PROFILES[rng.choice(PROFILE_ORDER)]
        candidates = []
        for gpu in gpus:
            for start in gpu.feasible_starts(requested):
                _before, _after, delta = delta_fragmentation(gpu, requested, start)
                candidates.append((delta, gpu.gpu_id, start))
        decision = MFIScheduler().choose(gpus, requested)
        if not candidates:
            assert decision is None
        else:
            assert decision is not None
            assert (decision.delta_fragmentation, decision.gpu_id, decision.start) == min(candidates)
        checked += 1

    print("PASS: profile model validation")
    print("PASS: 1g.20gb memory-overlap regression case")
    print(f"PASS: MFI argmin consistency over {checked} deterministic randomized states")
    print("These checks validate implementation consistency, not replication of published performance results.")


if __name__ == "__main__":
    main()
