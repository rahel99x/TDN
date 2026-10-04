"""Run preservation, manifest integrity and an actual small neural workflow."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tdn.research.protocol import digest, file_digest, readable_summary, verify_artifacts

ROOT = Path(__file__).resolve().parents[1]


def test_protocol_inspection_does_not_import_torch():
    result = subprocess.run([sys.executable, "-c", "import sys; import tdn.research.protocol; assert 'torch' not in sys.modules"],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_completed_artifacts_detect_tampering(tmp_path):
    protocol = {"version": 1}
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    (tmp_path / "summary.json").write_text('{"status":"COMPLETED"}')
    (tmp_path / "dataset.pt").write_bytes(b"test fixture, never loaded")
    manifest = {"version": 1, "protocol_sha256": digest(protocol), "source_tree_sha256": "source",
                "files": {name: file_digest(tmp_path / name) for name in ("protocol.json", "summary.json", "dataset.pt")}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "COMPLETED").write_text(file_digest(tmp_path / "manifest.json"))
    assert verify_artifacts(tmp_path, source_tree_sha256="source") == manifest
    with pytest.raises(ValueError, match="source changed"):
        verify_artifacts(tmp_path, source_tree_sha256="changed")
    (tmp_path / "dataset.pt").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_artifacts(tmp_path)


def test_research_cli_preserves_existing_run(tmp_path):
    run_dir = tmp_path / "existing"
    run_dir.mkdir()
    marker = run_dir / "important.txt"
    marker.write_text("preserve this result")
    result = subprocess.run([sys.executable, "scripts/research.py", "run", "--smoke", "--run-dir", str(run_dir)],
                            cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "fresh run directory" in result.stderr
    assert marker.read_text() == "preserve this result"
    assert sorted(path.name for path in run_dir.iterdir()) == ["important.txt"]


def test_research_cli_actual_neural_smoke(tmp_path):
    run_dir = tmp_path / "research-smoke"
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run([sys.executable, "scripts/research.py", "run", "--smoke", "--run-dir", str(run_dir)],
                            cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = verify_artifacts(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["status"] == "COMPLETED" and summary["device"] == "cpu"
    assert summary["actual_neural_training"] and len(summary["training"]) == 8
    successful = [row for row in summary["training"] if row["status"] == "COMPLETED"]
    assert successful and all(row["parameters_changed"] for row in successful)
    assert all(not row["diagnostics_seen_during_training"] for row in summary["training"])
    protocol = json.loads((run_dir / "protocol.json").read_text())
    assert len(protocol["parents"]) == 9
    assert "dataset.pt" in manifest["files"]
    report = (run_dir / "summary.txt").read_text()
    assert "summary.txt" in manifest["files"] and report in result.stdout
    assert "CPU development research" in report and "Eligible matched-accuracy comparisons:" in report
    for row in summary["training"]:
        assert f"{row['family']}: {row['status']}" in report
        if row.get("error"):
            assert row["error"] in report
    assert "heldout.json" in report and "frontier.json" in report
    # Re-running must fail before overwriting any immutable result.
    old_digest = file_digest(run_dir / "manifest.json")
    rerun = subprocess.run([sys.executable, "scripts/research.py", "run", "--smoke", "--run-dir", str(run_dir)],
                           cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    assert rerun.returncode != 0 and file_digest(run_dir / "manifest.json") == old_digest


def test_gpu_summary_distinguishes_frozen_benchmark_from_training():
    report = readable_summary({"status": "COMPLETED", "device": "cuda",
                               "headroom": {"passed": True, "case_ids": ["diagnostic-one"]},
                               "diagnostic_parent_count": 3, "comparisons": [], "failed_trajectories": 2})
    assert "GPU development research" in report and "Frozen CPU checkpoints" in report
    assert "Completed neural training:" not in report
    assert "PASS (1/3" in report and "Invalid timed trajectories retained: 2" in report


def test_summary_explicitly_reports_initialization_and_numerical_failure():
    report = readable_summary({"status": "COMPLETED", "device": "cpu",
                               "training": [{"family": "fixed_undamped", "status": "COMPLETED", "selected_step": 0},
                                            {"family": "transport", "status": "NUMERICAL_FAILURE", "selected_step": None,
                                             "error": "Nonfinite neural gradient"}],
                               "headroom": {"passed": False, "case_ids": []},
                               "diagnostic_parent_count": 3, "comparisons": [], "failed_trajectories": 2})
    assert "Completed neural training: 1/2" in report
    assert "selected initialization (step 0)" in report
    assert "transport: NUMERICAL_FAILURE; no selected checkpoint; Nonfinite neural gradient" in report
    assert "NO HEADROOM (0/3" in report
