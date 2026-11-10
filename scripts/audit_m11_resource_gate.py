"""Emit a machine-readable MFI score-gate audit example."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mig_scheduler.fragmentation import (  # noqa: E402
    fragmentation_score, fragmentation_score_breakdown_from_occupancy,
)
from mig_scheduler.profiles import PROFILES  # noqa: E402
from mig_scheduler.state import GPUState  # noqa: E402


def main():
    out = ROOT / "results" / "m11_resource_gate_audit"
    out.mkdir(parents=True, exist_ok=True)
    gpu = GPUState(0)
    gpu.allocate("small-at-index-5", PROFILES["1g.10gb"], 5)
    rows = fragmentation_score_breakdown_from_occupancy(
        gpu.compute_occupied_by, gpu.memory_occupied_by
    )
    payload = {
        "source": "Zambianco et al., arXiv:2511.18906, Algorithm 1 and Section IV",
        "source_url": "https://arxiv.org/abs/2511.18906",
        "interpretation": "component-wise compute and memory capacity gate; blocked legal placements weighted by memory-slice count",
        "gpu_state": {
            "compute_occupied_by": gpu.compute_occupied_by,
            "memory_occupied_by": gpu.memory_occupied_by,
            "free_compute_slices": gpu.free_compute_slices,
            "free_memory_slices": gpu.free_memory_slices,
        },
        "fragmentation_score": fragmentation_score(gpu),
        "profile_breakdown": rows,
        "key_check": next(r for r in rows if r["profile"] == "1g.20gb"),
        "caveat": "The paper's formula uses compact scalar notation for a resource pair; this is a transparent interpretation, not proof of equivalence to authors' executable implementation.",
    }
    path = out / "m11_resource_gate_example.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
