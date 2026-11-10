"""MIG profile metadata for the MFI A100-80GB model.

The model keeps compute and memory occupancy separate. This is important for
profiles such as 1g.20gb (one compute slice, two memory slices) and 3g.40gb
(three compute slices, four memory slices). Legal placement starts follow
Table I in Zambianco et al. (arXiv:2511.18906).

The field ``paper_slice_units`` preserves the scalar ``Slice`` column in that
table for plotting offered load in paper-inspired experiments. It is not used
as a substitute for compute/memory feasibility.
"""

from dataclasses import dataclass
from typing import Final

COMPUTE_SLICE_COUNT: Final[int] = 7
MEMORY_SLICE_COUNT: Final[int] = 8


@dataclass(frozen=True, slots=True)
class MigProfile:
    name: str
    compute_slices: int
    memory_slices: int
    legal_starts: tuple[int, ...]
    paper_slice_units: int

    def occupied_compute_indices(self, start: int) -> tuple[int, ...]:
        if start not in self.legal_starts:
            raise ValueError(f"Start index {start} is not legal for {self.name}.")
        indices = tuple(range(start, start + self.compute_slices))
        if any(i < 0 or i >= COMPUTE_SLICE_COUNT for i in indices):
            raise ValueError(f"Compute placement {self.name}@{start} is outside the GPU.")
        return indices

    def occupied_memory_indices(self, start: int) -> tuple[int, ...]:
        if start not in self.legal_starts:
            raise ValueError(f"Start index {start} is not legal for {self.name}.")
        indices = tuple(range(start, start + self.memory_slices))
        if any(i < 0 or i >= MEMORY_SLICE_COUNT for i in indices):
            raise ValueError(f"Memory placement {self.name}@{start} is outside the GPU.")
        return indices


GPU_MODEL: Final[str] = "A100-80GB (MFI profile/placement model; explicit compute and memory occupancy)"

# Table I fields: profile's compute footprint, memory footprint, legal placement
# indices, and scalar Slice units shown in the paper's table.
PROFILES: Final[dict[str, MigProfile]] = {
    "7g.80gb": MigProfile("7g.80gb", 7, 8, (0,), 7),
    "4g.40gb": MigProfile("4g.40gb", 4, 4, (0,), 4),
    "3g.40gb": MigProfile("3g.40gb", 3, 4, (0, 4), 4),
    "2g.20gb": MigProfile("2g.20gb", 2, 2, (0, 2, 4), 2),
    "1g.20gb": MigProfile("1g.20gb", 1, 2, (0, 2, 4, 6), 2),
    "1g.10gb": MigProfile("1g.10gb", 1, 1, (0, 1, 2, 3, 4, 5, 6), 1),
}

PROFILE_ORDER: Final[tuple[str, ...]] = tuple(PROFILES.keys())

# Explicit Best-Index preference order for A100-80GB, as documented in Table 1
# of Turkkan et al., "Optimal Workload Placement on Multi-Instance GPUs" (2024).
# This is a preference ordering, distinct from the set of legal starts above.
# MFI describes using this policy for its BF-BI/WF-BI baselines.
INDEX_PREFERENCE_ORDER: Final[dict[str, tuple[int, ...]]] = {
    "7g.80gb": (0,),
    "4g.40gb": (0,),
    "3g.40gb": (4, 0),
    "2g.20gb": (4, 0, 2),
    "1g.20gb": (6, 4, 0, 2),
    "1g.10gb": (6, 4, 5, 0, 1, 2, 3),
}


def index_preference_rank(profile_name: str, start: int) -> int:
    """Return the documented preference rank for a legal A100-80GB start."""
    order = INDEX_PREFERENCE_ORDER[profile_name]
    try:
        return order.index(start)
    except ValueError as exc:
        raise ValueError(f"Start {start} is not legal for profile {profile_name}.") from exc

MFI_WORKLOAD_DISTRIBUTIONS: Final[dict[str, dict[str, float]]] = {
    "uniform": {name: 1 / 6 for name in PROFILE_ORDER},
    "skew_small": {
        "7g.80gb": 0.05, "4g.40gb": 0.10, "3g.40gb": 0.10,
        "2g.20gb": 0.20, "1g.20gb": 0.25, "1g.10gb": 0.30,
    },
    "skew_big": {
        "7g.80gb": 0.30, "4g.40gb": 0.25, "3g.40gb": 0.20,
        "2g.20gb": 0.10, "1g.20gb": 0.10, "1g.10gb": 0.05,
    },
    "bimodal": {
        "7g.80gb": 0.30, "4g.40gb": 0.15, "3g.40gb": 0.05,
        "2g.20gb": 0.05, "1g.20gb": 0.15, "1g.10gb": 0.30,
    },
}


def validate_profile_model() -> None:
    if len(PROFILES) != 6:
        raise ValueError("Expected the six profile classes listed in MFI Table I.")
    for name, p in PROFILES.items():
        if p.name != name:
            raise ValueError(f"Profile key/name mismatch for {name}.")
        if min(p.compute_slices, p.memory_slices, p.paper_slice_units) <= 0:
            raise ValueError(f"Invalid resource demand in profile {name}.")
        if not p.legal_starts:
            raise ValueError(f"No legal starts for {name}.")
        for start in p.legal_starts:
            p.occupied_compute_indices(start)
            p.occupied_memory_indices(start)
        if set(INDEX_PREFERENCE_ORDER.get(name, ())) != set(p.legal_starts):
            raise ValueError(f"Index preference must order each legal start exactly once for {name}.")
        if len(INDEX_PREFERENCE_ORDER[name]) != len(set(INDEX_PREFERENCE_ORDER[name])):
            raise ValueError(f"Duplicate index in preference order for {name}.")
    for scenario, dist in MFI_WORKLOAD_DISTRIBUTIONS.items():
        if set(dist) != set(PROFILES):
            raise ValueError(f"Distribution {scenario} does not cover all profiles.")
        if any(p < 0 for p in dist.values()) or abs(sum(dist.values()) - 1) > 1e-9:
            raise ValueError(f"Invalid probabilities in distribution {scenario}.")
