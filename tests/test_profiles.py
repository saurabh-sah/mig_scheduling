from mig_scheduler.profiles import (
    COMPUTE_SLICE_COUNT,
    MFI_WORKLOAD_DISTRIBUTIONS,
    PROFILES,
    validate_profile_model,
)


def test_profile_model_is_internally_consistent():
    validate_profile_model()
    assert COMPUTE_SLICE_COUNT == 7
    assert len(PROFILES) == 6


def test_mfi_legal_start_indices_are_encoded():
    expected = {
        "7g.80gb": (0,),
        "4g.40gb": (0,),
        "3g.40gb": (0, 4),
        "2g.20gb": (0, 2, 4),
        "1g.20gb": (0, 2, 4, 6),
        "1g.10gb": (0, 1, 2, 3, 4, 5, 6),
    }
    assert {name: profile.legal_starts for name, profile in PROFILES.items()} == expected


def test_mfi_distributions_sum_to_one():
    for distribution in MFI_WORKLOAD_DISTRIBUTIONS.values():
        assert abs(sum(distribution.values()) - 1.0) < 1e-9


def test_compute_and_memory_footprints_are_distinct():
    assert PROFILES["1g.20gb"].compute_slices == 1
    assert PROFILES["1g.20gb"].memory_slices == 2
    assert PROFILES["3g.40gb"].compute_slices == 3
    assert PROFILES["3g.40gb"].memory_slices == 4
    assert PROFILES["7g.80gb"].compute_slices == 7
    assert PROFILES["7g.80gb"].memory_slices == 8
