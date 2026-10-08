"""Frontier CLI cohort policy and wrapper provenance seals."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("frontier_stage_cli_test", ROOT / "scripts/frontier.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


@pytest.mark.parametrize("values,stage", [([], "screen"), (["prepare=x"], "screen"),
    (["audit=x", "audit=y"], "screen"), (["audit="], "screen"),
    (["unknown=x"], "screen"), (["audit"], "screen")])
def test_required_prerequisite_chain_is_not_optional(values, stage):
    with pytest.raises(ValueError, match="Prerequisite|preceding"):
        cli.parse_prerequisites(values, stage)


def test_final_report_gets_all_stage_paths_even_when_missing():
    values = [f"{stage}=missing/{stage}" for stage in cli.STAGES[:-1]]
    assert set(cli.parse_prerequisites(values, "report")) == set(cli.STAGES[:-1])
    assert cli.parse_prerequisites([], "audit") == {}


def test_software_identity_permits_new_jobs_but_not_library_or_source_change():
    base = {"python": "3.13.13", "executable": "/home/rahel/TDN/.venv/bin/python", "venv": True,
        "torch": "2.10.0+cu126", "numpy": "2.2", "scipy": "1.15", "torch_cuda_runtime": "12.6",
        "git_commit": "a", "source_tree_sha256": "b", "slurm_job_id": "1"}
    assert cli.compatible_software(base) == cli.compatible_software({**base, "slurm_job_id": "2"})
    assert cli.compatible_software(base) != cli.compatible_software({**base, "source_tree_sha256": "c"})
    assert cli.compatible_software(base) != cli.compatible_software({**base, "torch": "other"})


def _command(args):
    env = {key: value for key, value in os.environ.items() if not key.startswith(("TDN_", "SLURM_"))}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
    return subprocess.run([sys.executable, str(ROOT / "scripts/frontier.py"), *args], cwd=ROOT,
                          env=env, text=True, capture_output=True, timeout=90)


@pytest.mark.parametrize("extra,message", [(["--profile", "full"], "full confirmation"),
    (["--profile", "smoke", "--device", "cuda"], "CPU smoke/development")])
def test_local_cli_cannot_run_fresh_full_cohort_or_cuda(tmp_path, extra, message):
    destination = tmp_path / "forbidden"
    result = _command(["--stage", "audit", "--run-dir", str(destination), "--local-root", str(ROOT), *extra])
    assert result.returncode != 0 and message in result.stderr
    assert not destination.exists()


@pytest.fixture
def sealed_stage(tmp_path, monkeypatch):
    # Unit-isolate the wrapper's seal from the numerical engine's independent
    # artifact verifier. This fixture never supplies scientific success evidence.
    engine = ModuleType("tdn.analysis.frontier.engine")
    engine.verify_science = lambda protocol, path: {"fixture": "science verifier isolated"}
    monkeypatch.setitem(sys.modules, engine.__name__, engine)
    protocol = {"profile": "smoke", "fixture": "wrapper provenance only"}
    execution = {"protocol_sha256": cli.digest(protocol), "software": {"source_tree_sha256": "source"}}
    for name, payload in (("execution.json", execution), ("protocol.json", protocol),
                          ("stage.json", {"status": "COMPLETED"}), ("science_manifest.json", {"fixture": True})):
        (tmp_path / name).write_text(json.dumps(payload))
    cli.seal_execution(tmp_path, protocol)
    return tmp_path, protocol


def test_execution_seal_binds_actual_protocol_source_and_completion(sealed_stage):
    path, protocol = sealed_stage
    result = cli.verify_execution(path, protocol, source_tree_sha256="source")
    assert result["execution"]["software"]["source_tree_sha256"] == "source"
    with pytest.raises(ValueError, match="source differs"):
        cli.verify_execution(path, protocol, source_tree_sha256="other")
    with pytest.raises(ValueError, match="protocol differs"):
        cli.verify_execution(path, {**protocol, "profile": "full"})


@pytest.mark.parametrize("name", ["execution.json", "protocol.json", "stage.json", "science_manifest.json"])
def test_execution_seal_detects_any_bound_file_change(sealed_stage, name):
    path, protocol = sealed_stage
    (path / name).write_text("{}")
    with pytest.raises(ValueError, match="evidence changed"):
        cli.verify_execution(path, protocol)


def test_execution_seal_cannot_omit_a_bound_file(sealed_stage):
    path, protocol = sealed_stage
    seal = json.loads((path / "workflow-seal.json").read_text())
    del seal["files"]["execution.json"]
    (path / "workflow-seal.json").write_text(json.dumps(seal))
    with pytest.raises(ValueError, match="incomplete file inventory"):
        cli.verify_execution(path, protocol)


def test_execution_seal_is_not_interchangeable_with_earlier_programs(sealed_stage):
    path, protocol = sealed_stage
    seal = json.loads((path / "workflow-seal.json").read_text())
    assert seal["schema"] == "tdn.frontier/v1"
    seal["schema"] = "tdn.roadmap/v1"
    (path / "workflow-seal.json").write_text(json.dumps(seal))
    with pytest.raises(ValueError, match="protocol differs"):
        cli.verify_execution(path, protocol)
