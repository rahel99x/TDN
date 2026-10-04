"""Desktop orchestration guards, without subprocess numerical work or CUDA."""
from argparse import Namespace
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import pytest

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tdn_desktop_launcher", PROJECT / "scripts" / "desktop.py")
desktop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(desktop)


@pytest.fixture
def project(monkeypatch, tmp_path):
    root = tmp_path / "desktop project with spaces"
    (root / "configs").mkdir(parents=True)
    shutil.copyfile(PROJECT / "configs" / "desktop-smoke.yaml", root / "configs" / "desktop-smoke.yaml")
    monkeypatch.setattr(desktop, "ROOT", root)
    monkeypatch.setattr(os, "environ", os.environ.copy())
    for name in ("SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_JOB_ACCOUNT", "CONDA_PREFIX"):
        os.environ.pop(name, None)
    monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
    monkeypatch.setattr(desktop, "validate_venv", lambda: None)
    from tdn.runtime import metadata
    monkeypatch.setattr(metadata, "software_metadata", lambda: {
        "source_tree_sha256": "unchanged-source", "python": "3.12.1", "torch": "2.10.0+cpu",
        "numpy": "2.2.6", "scipy": "1.15.3", "torch_cuda_runtime": None,
    })
    return root


def args(**updates):
    values = dict(config="configs/desktop-smoke.yaml", device="cpu", cuda_device=0,
                  max_steps=12, run_id="smoke-test", dry_run=False, resume=False)
    values.update(updates)
    return Namespace(**values)


@pytest.mark.parametrize("name", ["SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_JOB_ACCOUNT", "CONDA_PREFIX"])
def test_reject_cluster_or_conda_before_creating_cache(project, monkeypatch, name):
    monkeypatch.setenv(name, "active")
    with pytest.raises(ValueError, match="Slurm|standalone"):
        desktop.prepare_environment()
    assert not (project / ".runtime").exists()


def test_bootstrap_redirects_cached_temp_and_install_targets(project, monkeypatch):
    monkeypatch.setenv("PIP_TARGET", "outside-project")
    monkeypatch.setenv("PIP_PREFIX", "outside-project")
    monkeypatch.setenv("PYTHONUSERBASE", "outside-project")
    monkeypatch.setenv("PYTHONPATH", "outside-project")
    monkeypatch.setattr(tempfile, "tempdir", str(project.parent / "old-system-temp"))
    desktop.prepare_environment()
    assert Path(tempfile.gettempdir()).is_relative_to(project)
    for name in ("PIP_TARGET", "PIP_PREFIX", "PYTHONUSERBASE", "PYTHONPATH"):
        assert name not in os.environ
    assert os.environ["PIP_CONFIG_FILE"] == os.devnull


def test_desktop_bootstrap_refuses_carc_root_before_cache_creation(monkeypatch):
    monkeypatch.setattr(desktop, "ROOT", Path("/home1/aadaniel/projects/TDN").resolve())
    with pytest.raises(ValueError, match="fixed CARC"):
        desktop.prepare_environment()


def test_refuses_escaped_config_before_starting_any_stage(project):
    with pytest.raises(ValueError, match="escapes"):
        desktop.run(args(config=str(project.parent / "other.yaml")))
    assert not (project / "runs").exists()


def test_explicit_step_budget_guard_precedes_stages(project):
    with pytest.raises(ValueError, match="budget"):
        desktop.run(args(max_steps=11))
    assert not (project / "runs").exists()


def test_dry_run_lists_serial_commands_without_creating_run(project, capsys):
    assert desktop.run(args(dry_run=True)) == 0
    output = capsys.readouterr().out
    assert "cpu-tests -> audit -> generate -> calibrate -> train -> evaluate -> benchmark" in output
    assert "best.pt" in output
    assert not (project / "runs").exists()


def test_failed_stage_stops_all_successors(project, monkeypatch):
    executed = []
    def failure(command, log):
        executed.append(command)
        return 9
    monkeypatch.setattr(desktop, "execute", failure)
    assert desktop.run(args()) == 9
    assert len(executed) == 1
    report = json.loads((project / "runs" / "smoke-test" / "pipeline.json").read_text())
    assert report["status"] == "FAILED"
    assert report["stages"]["cpu-tests"]["exit_code"] == 9
    assert not (project / "runs" / "smoke-test" / "audit").exists()
    attempts = list((project / "runs" / "smoke-test" / "tower").glob("*/summary.json"))
    assert len(attempts) == 1
    tower = json.loads(attempts[0].read_text())
    assert tower["state"] == "FAILED" and tower["exit_code"] == 9
    assert "job_id" not in tower and "cpus" not in tower
    assert "TDN_TOWER_DIR" not in os.environ


def test_resume_refuses_changed_software_before_any_execution(project, monkeypatch):
    monkeypatch.setattr(desktop, "execute", lambda command, log: 9)
    assert desktop.run(args()) == 9
    manifest_path = project / "runs" / "smoke-test" / "pipeline.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["software"]["torch"] = "other-version"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(desktop, "execute", lambda *arguments: pytest.fail("Resume started an incompatible stage"))
    with pytest.raises(ValueError, match="software differs"):
        desktop.run(args(resume=True))


def test_resume_refuses_missing_checkpoint_for_interrupted_training(project, monkeypatch):
    monkeypatch.setattr(desktop, "execute", lambda command, log: 9)
    assert desktop.run(args()) == 9
    manifest_path = project / "runs" / "smoke-test" / "pipeline.json"
    manifest = json.loads(manifest_path.read_text())
    for stage in ("cpu-tests", "audit", "generate", "calibrate"):
        manifest["stages"][stage] = {"status": "COMPLETED"}
        stage_dir = project / "runs" / "smoke-test" / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / "COMPLETED").write_text("completed")
    manifest["stages"]["train"] = {"status": "PAUSED_NEEDS_RESUME"}
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(desktop, "execute", lambda *arguments: pytest.fail("Interrupted train restarted without its checkpoint"))
    with pytest.raises(ValueError, match="no last.pt"):
        desktop.run(args(resume=True))


def test_missing_best_records_failed_unexecuted_evaluation(project, monkeypatch):
    monkeypatch.setattr(desktop, "execute", lambda command, log: 0)
    with pytest.raises(ValueError, match="no feasible"):
        desktop.run(args())
    report = json.loads((project / "runs" / "smoke-test" / "pipeline.json").read_text())
    assert report["status"] == "FAILED"
    assert report["stages"]["train"]["status"] == "COMPLETED"
    assert report["stages"]["evaluate"]["actually_ran"] is False


def test_desktop_retry_retains_distinct_tower_attempts(project, monkeypatch):
    monkeypatch.setattr(desktop, "execute", lambda *args: 75)
    assert desktop.run(args()) == 75
    assert desktop.run(args(resume=True)) == 75
    attempts = list((project / "runs" / "smoke-test" / "tower").glob("*/summary.json"))
    records = [json.loads(path.read_text()) for path in attempts]
    assert len(records) == 2 and len({row["id"] for row in records}) == 2
    assert all(row["state"] == "INTERRUPTED" for row in records)
    assert sorted(row["attempt"] for row in records) == [1, 2]


def test_desktop_pytest_does_not_inherit_production_metrics(project, monkeypatch):
    captured = []

    class Child:
        stdout = []
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def wait(self): return 0

    def popen(command, **kwargs):
        captured.append(kwargs["env"])
        return Child()

    monkeypatch.setattr(desktop.subprocess, "Popen", popen)
    monkeypatch.setenv("TDN_TOWER_DIR", "production-report")
    assert desktop.execute([sys.executable, "-m", "pytest", "-q"], project / "test.log") == 0
    assert "TDN_TOWER_DIR" not in captured[0]
    assert os.environ["TDN_TOWER_DIR"] == "production-report"
