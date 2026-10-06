"""Real CPU preflight, immutable output directories and sealed manifests."""
import json
import os
from pathlib import Path
import subprocess
import sys

from tdn.analysis.interactions.protocol import ARTIFACTS, digest, file_digest

ROOT = Path(__file__).resolve().parents[1]


def test_interaction_help_needs_no_scientific_imports():
    child = subprocess.run([sys.executable, "-S", "scripts/interaction_screen.py", "--help"],
                           cwd=ROOT, text=True, capture_output=True)
    assert child.returncode == 0, child.stderr
    assert "--smoke" in child.stdout


def test_cpu_smoke_sealed_and_completed_run_protected(tmp_path):
    output = tmp_path / "screen"
    env = {key: value for key, value in os.environ.items() if key != "TDN_TOWER_DIR"}
    command = [sys.executable, "scripts/interaction_screen.py", "--run-dir", str(output), "--smoke"]
    child = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=120)
    assert child.returncode == 0, child.stdout + child.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    assert (output / "COMPLETED").read_text().strip() == file_digest(output / "manifest.json")
    assert set(manifest["files"]) == set(ARTIFACTS)
    for relative, expected in manifest["files"].items():
        assert file_digest(output / relative) == expected
    protocol = json.loads((output / "protocol.json").read_text())
    canonical = json.loads((output / "interaction-screen.json").read_text())
    assert manifest["protocol_sha256"] == digest(protocol)
    assert canonical["training_performed"] is False and canonical["training_attempted"] is False
    assert canonical["status"] == canonical["computational_status"] == "COMPLETED"
    assert canonical["scientific_outcome"] in ("OBSERVED_MIXED", "INCONCLUSIVE")
    assert canonical["schema"] == "tdn.interaction-screen/v1"
    assert canonical["device"] == "cpu"
    for name, plan in protocol["plans"].items():
        assert {row["case_id"] for row in canonical["rows"] if row["panel"] == name} == set(plan["expected_case_ids"])
    assert sum(len(row["metrics"]) for row in canonical["rows"]) < 12000
    previous = file_digest(output / "manifest.json")
    child = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=30)
    assert child.returncode != 0 and "preserve prior results" in child.stderr
    assert file_digest(output / "manifest.json") == previous


def test_existing_empty_directory_is_not_claimed(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    child = subprocess.run([sys.executable, "scripts/interaction_screen.py", "--run-dir", str(output), "--smoke"],
                           cwd=ROOT, text=True, capture_output=True, timeout=30)
    assert child.returncode != 0
    assert "fresh nonexistent run directory" in child.stderr
    assert list(output.iterdir()) == []
