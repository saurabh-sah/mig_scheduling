from scripts.analyze_m7_horizon_audit import stratified_bootstrap_ci
from scripts.run_m7_horizon_audit import derive_decision_diagnostic, parse_candidate_scores


def test_candidate_score_parser_accepts_serialized_candidate_tuple():
    row = {"candidate_scores_json": '[[0, 2, 1, 3.5, 0.7], [1, 4, 2, 2.0, 0.5]]'}
    parsed = parse_candidate_scores(row)
    assert len(parsed) == 2
    assert parsed[0]["gpu_id"] == 0
    assert parsed[0]["start"] == 2
    assert parsed[0]["mean_future_acceptance"] == 3.5


def test_decision_diagnostic_computes_estimated_gain_and_placement_change():
    meta = {"trace_id": "uniform-1", "scenario": "uniform", "seed": 1, "policy": "fixed_h2_r4"}
    job = {
        "job_id": "job-1", "profile": "1g.10gb", "arrival_time": 1.0,
        "status": "accepted", "gpu_id": 1, "start": 2,
        "mfi_gpu_id": 0, "mfi_start": 0,
        "candidate_scores_json": '[[0, 0, 1, 1.5, 0.4], [1, 2, 2, 2.5, 0.6]]',
        "lookahead_value": 2.5, "lookahead_std": 0.6, "total_rollouts": 8,
        "decision_time_ns": 1000,
    }
    row = derive_decision_diagnostic(meta, job)
    assert row["choice_differs_from_mfi"] == 1
    assert row["estimated_future_gain_vs_mfi_candidate"] == 1.0
    assert row["top_two_estimated_value_gap"] == 1.0


def test_stratified_bootstrap_ci_is_deterministic():
    rows = [
        {"scenario": "a", "delta": 0.1},
        {"scenario": "a", "delta": 0.2},
        {"scenario": "b", "delta": -0.1},
        {"scenario": "b", "delta": 0.0},
    ]
    first = stratified_bootstrap_ci(rows, repeats=1000, seed=7)
    second = stratified_bootstrap_ci(rows, repeats=1000, seed=7)
    assert first == second
    assert abs(first[0] - 0.05) < 1e-12
    assert first[1] <= first[0] <= first[2]


def test_bootstrap_single_trace_returns_point_estimate():
    rows = [{"scenario": "one", "delta": 0.125}]
    assert stratified_bootstrap_ci(rows, repeats=1000, seed=1) == (0.125, 0.125, 0.125)


def test_decision_diagnostic_marks_rejected_job_as_not_comparable():
    meta = {"trace_id": "uniform-1", "scenario": "uniform", "seed": 1, "policy": "fixed_h2_r4"}
    job = {
        "job_id": "job-rejected", "profile": "7g.80gb", "arrival_time": 1.0,
        "status": "rejected", "gpu_id": None, "start": None,
        "mfi_gpu_id": None, "mfi_start": None, "candidate_scores_json": "[]",
    }
    row = derive_decision_diagnostic(meta, job)
    assert row["choice_differs_from_mfi"] is None
