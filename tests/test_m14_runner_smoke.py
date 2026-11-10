"""Smoke test for Milestone 14 large-scale evaluation runner."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "run_m14_large_scale_eval.py"


def test_m14_runner_smoke_execution(tmp_path: Path):
    """Run run_m14_large_scale_eval.py with --smoke flag in a temp directory and verify outputs."""
    out_dir = tmp_path / "m14_smoke_out"
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--smoke",
        "--out",
        str(out_dir),
        "--workers",
        "2",
        "--max-lookahead-candidates",
        "5",
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Runner failed with exit code {res.returncode}.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    # Required CSV and metadata files
    expected_files = [
        "m14_raw_runs.csv",
        "m14_policy_summary.csv",
        "m14_paired_acceptance_vs_free_anchor_mfi.csv",
        "m14_score_mode_decision_sensitivity.csv",
        "m14_decision_latency.csv",
        "m14_run_manifest.json",
        "M14_EXPERIMENT_REPORT.md",
        "m14_acceptance_progressive.png",
        "m14_acceptance_burst.png",
        "m14_latency_pareto.png",
    ]

    for fname in expected_files:
        fpath = out_dir / fname
        assert fpath.exists(), f"Expected artifact {fname} was not created"
        assert fpath.stat().st_size > 0, f"Expected artifact {fname} is empty"

    # 1. Verify m14_raw_runs.csv
    with (out_dir / "m14_raw_runs.csv").open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        # 2 regimes * 4 scenarios * 1 repetition * 10 policies = 80 rows
        assert len(reader) == 80, f"Expected 80 raw runs, got {len(reader)}"
        assert reader[0]["gpus"] == "2"
        assert reader[0]["max_candidates"] == "5"

    # 2. Verify m14_policy_summary.csv
    with (out_dir / "m14_policy_summary.csv").open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        # 2 regimes * 10 policies = 20 summary rows
        assert len(reader) == 20, f"Expected 20 policy summary rows, got {len(reader)}"
        policies = {r["policy"] for r in reader}
        assert "fixed_lookahead__all_legal_starts" in policies
        assert "adaptive_lookahead__free_anchor_starts" in policies

    # 3. Verify m14_paired_acceptance_vs_free_anchor_mfi.csv
    with (out_dir / "m14_paired_acceptance_vs_free_anchor_mfi.csv").open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        # 2 regimes * 9 policies (excluding reference) = 18 rows
        assert len(reader) == 18, f"Expected 18 paired comparison rows, got {len(reader)}"
        for row in reader:
            assert "paired_bootstrap_95ci_low" in row
            assert "paired_bootstrap_95ci_high" in row
            assert float(row["paired_workloads"]) == 4.0  # 4 scenarios * 1 rep

    # 4. Verify m14_score_mode_decision_sensitivity.csv
    with (out_dir / "m14_score_mode_decision_sensitivity.csv").open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        # 2 regimes * 4 scenarios * 1 rep * 2 comparisons = 16 rows
        assert len(reader) == 16, f"Expected 16 sensitivity rows, got {len(reader)}"

    # 5. Verify m14_decision_latency.csv
    with (out_dir / "m14_decision_latency.csv").open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        # 2 regimes * 10 policies = 20 rows
        assert len(reader) == 20, f"Expected 20 latency rows, got {len(reader)}"
        for row in reader:
            assert float(row["mean_decision_time_ms"]) >= 0.0
            assert float(row["median_decision_time_ms"]) >= 0.0
            assert int(row["total_decisions"]) > 0

    # 6. Verify m14_run_manifest.json
    with (out_dir / "m14_run_manifest.json").open("r", encoding="utf-8") as f:
        manifest = json.load(f)
        assert manifest["milestone"] == 14
        assert manifest["gpus"] == 2
        assert manifest["runs"] == 80
        assert manifest["max_lookahead_candidates"] == 5
        assert len(manifest["policies"]) == 10

    # 7. Verify M14_EXPERIMENT_REPORT.md
    report_text = (out_dir / "M14_EXPERIMENT_REPORT.md").read_text(encoding="utf-8")
    assert "# M14 Large-Scale Evaluation" in report_text
    assert "Policy Acceptance and Resource Utilization Summary" in report_text
    assert "Paired Acceptance Differences vs. Free-Anchor MFI" in report_text
    assert "Decision Latency Distribution" in report_text


def test_m14_runner_argument_validation():
    """Verify that invalid command line arguments are rejected."""
    bad_invocations = [
        ["--gpus", "0"],
        ["--repetitions", "-1"],
        ["--workers", "0"],
        ["--max-lookahead-candidates", "0"],
    ]

    for flags in bad_invocations:
        cmd = [sys.executable, str(SCRIPT_PATH), *flags]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode != 0, f"Expected failure for flags {flags}, but succeeded"
