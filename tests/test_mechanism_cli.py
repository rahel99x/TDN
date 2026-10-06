"""Real CPU entrypoint: preflight, complete artifacts and immutable attempts."""
import json
import os
from pathlib import Path
import subprocess
import sys

from tdn.analysis.mechanisms.protocol import ARTIFACTS, digest, file_digest

ROOT = Path(__file__).resolve().parents[1]


def test_help_needs_no_scientific_packages():
    child = subprocess.run([sys.executable, "-S", "scripts/mechanism_audit.py", "--help"],
                           cwd=ROOT, text=True, capture_output=True)
    assert child.returncode == 0, child.stderr
    assert "--smoke" in child.stdout


def test_cpu_smoke_seals_all_cases_and_refuses_overwrite(tmp_path):
    output = tmp_path / "audit"
    env = {key: value for key, value in os.environ.items() if key != "TDN_TOWER_DIR"}
    command = [sys.executable, "scripts/mechanism_audit.py", "--run-dir", str(output), "--smoke"]
    child = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=90)
    assert child.returncode == 0, child.stdout + child.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    assert (output / "COMPLETED").read_text().strip() == file_digest(output / "manifest.json")
    assert set(manifest["files"]) == set(ARTIFACTS)
    for relative, expected in manifest["files"].items():
        assert file_digest(output / relative) == expected
    protocol = json.loads((output / "protocol.json").read_text())
    canonical = json.loads((output / "mechanism-audit.json").read_text())
    assert manifest["protocol_sha256"] == digest(protocol)
    assert canonical["training_attempted"] is False
    assert canonical["status"] == "COMPLETED"
    for name, plan in protocol["plans"].items():
        assert {r["case_id"] for r in canonical["rows"] if r["panel"] == name} == set(plan["expected_case_ids"])
    previous = file_digest(output / "manifest.json")
    child = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=30)
    assert child.returncode != 0
    assert "preserve prior results" in child.stderr
    assert file_digest(output / "manifest.json") == previous
