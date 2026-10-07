"""The stage CLI enforces frozen lineage and protects full fresh confirmation."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("agenda_stage_cli", ROOT / "scripts/agenda.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


@pytest.mark.parametrize("values,stage", [([], "prepare"), (["prepare=x"], "prepare"),
    (["structure=x", "structure=y"], "prepare"), (["structure="], "prepare"),
    (["unknown=x"], "prepare"), (["structure"], "prepare")])
def test_missing_duplicate_or_out_of_order_prerequisites_are_rejected(values, stage):
    with pytest.raises(ValueError, match="Prerequisite|preceding"):
        cli.parse_prerequisites(values, stage)


def test_exact_earlier_chain_is_required():
    assert cli.parse_prerequisites([], "structure") == {}
    assert cli.parse_prerequisites(["prepare=y", "structure=x"], "controls") == {
        "prepare": Path("y"), "structure": Path("x")}


def test_software_identity_allows_job_change_but_rejects_runtime_change():
    base = dict(python="3.13.13", executable="/home/rahel/TDN/.venv/bin/python", venv=True,
                torch="2.10.0+cu126", numpy="2.2", scipy="1.15", torch_cuda_runtime="12.6",
                git_commit="a", source_tree_sha256="b", slurm_job_id="1")
    assert cli.compatible_software(base) == cli.compatible_software({**base, "slurm_job_id": "2"})
    assert cli.compatible_software(base) != cli.compatible_software({**base, "torch": "2.9"})


def _command(args):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("TDN_", "SLURM_"))}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
    return subprocess.run([sys.executable, str(ROOT / "scripts/agenda.py"), *args], cwd=ROOT,
                          env=env, text=True, capture_output=True, timeout=90)


@pytest.mark.parametrize("extra,message", [(["--profile", "full"], "full confirmation"),
    (["--profile", "smoke", "--device", "cuda"], "CPU smoke/development")])
def test_local_cli_rejects_full_or_cuda_before_creating_outputs(tmp_path, extra, message):
    path = tmp_path / "forbidden"
    response = _command(["--stage", "structure", "--run-dir", str(path), "--local-root", str(ROOT), *extra])
    assert response.returncode != 0
    assert message in response.stderr
    assert not path.exists()


def test_actual_structural_cli_seals_evidence_and_rejects_tampered_successor(tmp_path):
    path = tmp_path / "structure"
    response = _command(["--stage", "structure", "--profile", "smoke", "--run-dir", str(path),
                         "--local-root", str(ROOT)])
    assert response.returncode == 0, response.stdout + response.stderr
    from tdn.analysis.agenda.artifacts import verify_stage
    manifest = verify_stage(path, stage="structure", profile="smoke")
    assert manifest["benchmark_suite"] == "agenda"
    assert json.loads((path / "summary.json").read_text())["correctness_failures"] == 0
    summary = json.loads((path / "summary.json").read_text())
    summary["correctness_failures"] = 1
    (path / "summary.json").write_text(json.dumps(summary))
    target = tmp_path / "prepare"
    failed = _command(["--stage", "prepare", "--profile", "smoke", "--run-dir", str(target),
        "--local-root", str(ROOT), "--prerequisite-dir", f"structure={path}"])
    assert failed.returncode != 0
    assert "changed" in failed.stderr or "differs" in failed.stderr
    assert not target.exists()
