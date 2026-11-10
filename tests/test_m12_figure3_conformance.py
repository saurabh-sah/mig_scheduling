"""Regression tests anchored to the numeric and qualitative MFI Figure 3 example."""
from mig_scheduler.mfi_score_variants import (
    MFIScoreVariantScheduler,
    PaperBFWFBestIndexScheduler,
    fragmentation_score_variant,
)
from mig_scheduler.profiles import PROFILES
from mig_scheduler.schedulers import BestFitBestIndexScheduler, WorstFitBestIndexScheduler
from mig_scheduler.state import GPUState


def _state(allocations, gpu_id=0):
    gpu = GPUState(gpu_id)
    for job_id, profile_name, start in allocations:
        gpu.allocate(job_id, PROFILES[profile_name], start)
    return gpu


def test_reconstructed_figure3_gpu1_score_matches_reported_eight():
    # Two 1g.10gb instances at starts 2 and 3 leave anchor 0 free, but block
    # 4g and 3g profiles beginning at 0. This configuration matches the 30%
    # compute-occupancy diagram and the published F(GPU1)=8 example.
    gpu = _state([
        ("small-a", "1g.10gb", 2),
        ("small-b", "1g.10gb", 3),
    ])
    assert fragmentation_score_variant(gpu, mode="free_anchor_starts") == 8
    # The earlier all-blocked-legal-starts interpretation gives a different
    # value, which is precisely the conformance discrepancy M12 documents.
    assert fragmentation_score_variant(gpu, mode="all_legal_starts") == 14


def test_reconstructed_figure3_gpu2_score_matches_reported_sixteen():
    # 2g.20gb at start 2 plus 1g.10gb at start 5 uses three compute slices
    # (about 43%) and blocks starts 0/4 for 3g.40gb. The free-anchor score
    # reproduces the paper's reported F(GPU2)=16.
    gpu = _state([
        ("medium", "2g.20gb", 2),
        ("small", "1g.10gb", 5),
    ])
    assert fragmentation_score_variant(gpu, mode="free_anchor_starts") == 16
    assert fragmentation_score_variant(gpu, mode="all_legal_starts") == 23


def test_paper_bf_bi_selects_best_fit_gpu_then_rejects_if_fragmented():
    # GPU 0 has lower residual capacity but no legal 3g.40gb placement;
    # GPU 1 can host the request at start 4. Paper-path BF-BI picks GPU 0
    # by aggregate residual first, then rejects rather than globally searching.
    g0 = _state([
        ("g0-medium", "2g.20gb", 2),
        ("g0-small", "1g.10gb", 5),
    ])
    g0.gpu_id = 0
    g1 = _state([
        ("g1-small-a", "1g.10gb", 2),
        ("g1-small-b", "1g.10gb", 3),
    ], gpu_id=1)
    request = PROFILES["3g.40gb"]

    paper_path = PaperBFWFBestIndexScheduler(strategy="bf_bi").choose([g0, g1], request)
    global_path = BestFitBestIndexScheduler().choose([g0, g1], request)
    assert paper_path is None
    assert global_path is not None and global_path.gpu_id == 1 and global_path.start == 4


def test_paper_wf_bi_selects_worst_fit_gpu_then_rejects_if_fragmented():
    # GPU 0 has more residual resources and no legal 3g placement because
    # starts 0 and 4 are both occupied. GPU 1 has a legal placement at 4.
    g0 = _state([
        ("g0-small-a", "1g.10gb", 0),
        ("g0-small-b", "1g.10gb", 4),
    ], gpu_id=0)
    g1 = _state([("g1-large", "4g.40gb", 0)], gpu_id=1)
    request = PROFILES["3g.40gb"]

    paper_path = PaperBFWFBestIndexScheduler(strategy="wf_bi").choose([g0, g1], request)
    global_path = WorstFitBestIndexScheduler().choose([g0, g1], request)
    assert paper_path is None
    assert global_path is not None and global_path.gpu_id == 1 and global_path.start == 4


def test_mfi_score_variant_scheduler_returns_feasible_placement():
    gpus = [GPUState(0), GPUState(1)]
    gpus[0].allocate("occupied", PROFILES["1g.10gb"], 2)
    request = PROFILES["3g.40gb"]
    for mode in ("all_legal_starts", "free_anchor_starts"):
        decision = MFIScoreVariantScheduler(score_mode=mode).choose(gpus, request)
        assert decision is not None
        assert gpus[decision.gpu_id].is_placement_feasible(request, decision.start)
