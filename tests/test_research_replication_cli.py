"""Versioned artifact sealing and actual bounded replication CLI integration."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from tdn.research.protocol import digest, file_digest, verify_artifacts
from tdn.research.replication_protocol import DEFAULT

ROOT = Path(__file__).resolve().parents[1]


def test_replication_manifest_versions_must_match(tmp_path):
    protocol = {"version": 2, "config": {"protocol_version": 2}}
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    (tmp_path / "summary.json").write_text('{"status":"COMPLETED"}')
    for name in ("replication.json", "normalization.json", "references.json"):
        (tmp_path / name).write_text("{}")
    (tmp_path / "dataset.pt").write_bytes(b"never unpickled")
    manifest = {"version": 2, "protocol_sha256": digest(protocol), "source_tree_sha256": "source",
                "files": {name: file_digest(tmp_path / name)
                          for name in ("protocol.json", "summary.json", "dataset.pt", "replication.json",
                                       "normalization.json", "references.json")}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "COMPLETED").write_text(file_digest(tmp_path / "manifest.json"))
    assert verify_artifacts(tmp_path) == manifest
    manifest["version"] = 1
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "COMPLETED").write_text(file_digest(tmp_path / "manifest.json"))
    with pytest.raises(ValueError, match="versions differ"):
        verify_artifacts(tmp_path)


def test_replication_cli_rejects_wrong_workflow_before_creating_output(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(DEFAULT))
    run_dir = tmp_path / "wrong-suite"
    result = subprocess.run([sys.executable, "scripts/research.py", "run", "--config", str(config),
                             "--run-dir", str(run_dir), "--expected-suite", "neural-benchmarks"],
                            cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and "Configuration suite differs" in result.stderr
    assert not run_dir.exists()


def test_replication_cli_actual_smoke_seals_all_seed_blocks(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(DEFAULT))
    run_dir = tmp_path / "replication-smoke"
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run([sys.executable, "scripts/research.py", "run", "--config", str(config),
                             "--run-dir", str(run_dir), "--smoke", "--expected-suite", "neural-replication"],
                            cwd=ROOT, env=env, capture_output=True, text=True, timeout=210)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = verify_artifacts(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    detail = json.loads((run_dir / "replication.json").read_text())
    protocol = json.loads((run_dir / "protocol.json").read_text())
    assert manifest["version"] == protocol["version"] == 2
    assert summary["status"] == "COMPLETED" and summary["device"] == "cpu"
    assert summary["actual_neural_training"] is True
    assert summary["diagnostic_parent_count"] == 9 and summary["completed_block_count"] == 9
    assert len(summary["training"]) == len(detail["endpoint_rows"]) == 15
    assert len(detail["comparison_rows"]) == 12
    assert all(row["rollout_expected"] == 36 and row["heldout_one_expected"] == 18
               and row["heldout_two_expected"] == 18 for row in detail["endpoint_rows"])
    assert all(not row["diagnostics_seen_during_training"] for row in summary["training"])
    blocks = list(run_dir.glob("replicates/seed-*/blocks/block-*/frontier.json"))
    assert len(blocks) == 9
    assert all(path.relative_to(run_dir).as_posix() in manifest["files"] for path in blocks)
    assert "Smoke integration fixture" in (run_dir / "summary.txt").read_text()
    previous = file_digest(run_dir / "manifest.json")
    rerun = subprocess.run([sys.executable, "scripts/research.py", "run", "--config", str(config),
                            "--run-dir", str(run_dir), "--smoke"],
                           cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    assert rerun.returncode != 0 and file_digest(run_dir / "manifest.json") == previous
