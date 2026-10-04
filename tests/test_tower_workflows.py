"""Allocation reporting integration; scheduler identity is mocked explicitly."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(f"tower_test_{name}", ROOT / "scripts" / f"{name}.py")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    base = tmp_path / "workflow"
    base.mkdir()
    for name in ("SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "TDN_TOWER_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "41001")
    monkeypatch.setenv("SLURM_STEP_ID", "0")
    monkeypatch.setenv("SLURM_JOB_ACCOUNT", "anakano_81")
    return {"run_id": "tower-fixture", "run_dir": str(base), "source_sha256": "a" * 64,
            "config_sha256": "b" * 64, "source_tree_sha256": "c" * 64,
            "profile": "smoke", "action": "start", "smoke": True,
            "resources": {"cpu": {"partition": "main", "cpus": 4, "mem_gib": 16, "walltime": "00:30:00"},
                          "gpu": {"partition": "gpu", "cpus": 4, "mem_gib": 16, "walltime": "00:30:00"}}}


def fake_publish(monkeypatch, result=None):
    calls = []
    def publish(report, sources):
        calls.append((report, sources))
        return {"numerical_failures": 1} if result is None else result
    monkeypatch.setitem(sys.modules, "tdn.tower_analytics", SimpleNamespace(publish_outputs=publish))
    return calls


def test_grouped_attempt_is_bound_to_verified_job_and_has_exact_logs(workflow, monkeypatch):
    cw = module("carc_workflow")
    monkeypatch.setattr(cw, "actual_policy", lambda **kwargs: None)
    monkeypatch.setattr(cw, "ownership", lambda *args: {"JobState": "RUNNING"})
    first = cw.worker_report_begin(workflow, "cpu")
    inventory = json.loads((first / "run.json").read_text())
    logs = json.loads((first / "logs.json").read_text())
    assert first.parent == Path(workflow["run_dir"]) / "tower"
    assert inventory["job_id"] == logs["job_id"] == "41001"
    assert inventory["resources"]["time_seconds"] == 1800
    assert inventory["resources"]["mem_bytes"] == 16 * 1024 ** 3
    assert inventory["resources"]["gpus"] == 0
    assert any(row["path"].endswith("cpu-41001.err") for row in logs["logs"])
    with pytest.raises(ValueError, match="already has a Tower attempt"):
        cw.worker_report_begin(workflow, "cpu")
    monkeypatch.setenv("SLURM_JOB_ID", "41002")
    second = cw.worker_report_begin(workflow, "gpu")
    assert second != first
    assert json.loads((second / "run.json").read_text())["resources"]["gpus"] == 1
    assert len([path for path in first.parent.iterdir() if path.is_dir()]) == 2


def test_unowned_allocation_creates_no_tower_files(workflow, monkeypatch):
    cw = module("carc_workflow")
    monkeypatch.setattr(cw, "actual_policy", lambda **kwargs: None)
    monkeypatch.setattr(cw, "ownership", lambda *args: {"JobState": "PENDING"})
    with pytest.raises(ValueError, match="running allocation"):
        cw.worker_report_begin(workflow, "cpu")
    assert not (Path(workflow["run_dir"]) / "tower").exists()


def test_status_discovers_exact_job_reports_without_writes(workflow, monkeypatch, capsys):
    cw = module("carc_workflow")
    first = cw.begin_workflow_report(workflow, "cpu")
    monkeypatch.setenv("SLURM_JOB_ID", "41002")
    second = cw.begin_workflow_report(workflow, "gpu")
    before = {path: path.read_bytes() for path in first.parent.rglob("*") if path.is_file()}
    monkeypatch.setattr(cw, "jobs_for", lambda *args: {"cpu": [{"job_id": "41001"}], "gpu": []})
    monkeypatch.setattr(cw, "scheduler_state", lambda *args: "COMPLETED")
    cw.status(workflow)
    output = capsys.readouterr().out
    assert f"TDN_TOWER_DIR={first}" in output
    assert str(second) not in output
    assert before == {path: path.read_bytes() for path in first.parent.rglob("*") if path.is_file()}


@pytest.mark.parametrize("exit_code,interrupted,state", [(0, False, "COMPLETED"), (75, False, "INTERRUPTED"),
                                                        (143, True, "INTERRUPTED"), (7, False, "FAILED")])
def test_grouped_terminal_preserves_exit_and_scientific_outcome(workflow, monkeypatch, exit_code, interrupted, state):
    cw = module("carc_workflow")
    monkeypatch.setattr(cw, "actual_policy", lambda **kwargs: None)
    monkeypatch.setattr(cw, "ownership", lambda *args: {"JobState": "RUNNING"})
    calls = fake_publish(monkeypatch)
    report = cw.worker_report_begin(workflow, "cpu")
    monkeypatch.setenv("TDN_TOWER_DIR", str(report))
    cw.worker_report_finish(workflow, "cpu", exit_code, interrupted=interrupted)
    summary = json.loads((report / "summary.json").read_text())
    assert summary["state"] == state and summary["exit_code"] == exit_code
    assert summary["runtime_seconds"] >= 0
    assert summary["results"]["numerical_failures"] == 1
    assert calls == [(report, [Path(workflow["run_dir"])])]


def test_recovered_stage_does_not_emit_or_create_reporting(workflow, monkeypatch):
    cw = module("carc_workflow")
    monkeypatch.setattr(cw, "reporting_api", lambda: pytest.fail("Recovery imported reporting"))
    monkeypatch.setenv("TDN_TOWER_DIR", "unrelated")
    cw.mark(workflow, "cpu", "audit", "COMPLETED", 0, recovered=True)
    assert not (Path(workflow["run_dir"]) / "tower").exists()


def configure_research_worker(workflow, monkeypatch, *, exit_code=0, reporting_error=False):
    rw = module("research_workflow")
    monkeypatch.setattr(rw, "verify_worker", lambda *args: {"verified": True})
    monkeypatch.setattr(rw, "software_report", lambda *args: {"verified": True})
    monkeypatch.setattr(rw, "load", lambda *args: workflow)
    monkeypatch.setattr(rw, "verify_experiment", lambda *args: None)
    monkeypatch.setattr(rw, "worker_commands", lambda *args: [("tests", ["pytest"]), ("experiment", ["science"])])
    fake_publish(monkeypatch, {"numerical_failures": 1})
    launches = []
    class Child:
        def __init__(self, args, **kwargs):
            launches.append((args, kwargs["env"]))
        def wait(self, timeout=None):
            return exit_code
        def poll(self):
            return exit_code
    monkeypatch.setattr(rw.subprocess, "Popen", Child)
    if reporting_error:
        monkeypatch.setattr(rw.cw, "finish_workflow_report", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("report unavailable")))
    return rw, launches


def test_research_child_env_and_completed_scientific_failure(workflow, monkeypatch):
    monkeypatch.setenv("TDN_TOWER_DIR", "original-caller-value")
    rw, launches = configure_research_worker(workflow, monkeypatch)
    assert rw.worker(workflow, "cpu") == 0
    assert "TDN_TOWER_DIR" not in launches[0][1]
    report = Path(launches[1][1]["TDN_TOWER_DIR"])
    assert json.loads((report / "summary.json").read_text())["state"] == "COMPLETED"
    assert json.loads((report / "summary.json").read_text())["results"]["numerical_failures"] == 1
    assert os.environ["TDN_TOWER_DIR"] == "original-caller-value"
    assert not (Path(workflow["run_dir"]) / "experiment" / "tower").exists()


@pytest.mark.parametrize("code,reporting_error", [(7, False), (75, False), (7, True)])
def test_research_failure_retains_child_code_and_stops_successors(workflow, monkeypatch, code, reporting_error):
    rw, launches = configure_research_worker(workflow, monkeypatch, exit_code=code, reporting_error=reporting_error)
    with pytest.raises(rw.WorkerFailure) as captured:
        rw.worker(workflow, "cpu")
    assert captured.value.exit_code == code
    assert len(launches) == 1
    state = json.loads(rw.state_path(workflow).read_text())
    assert state["exit_code"] == code
    assert state["status"] == ("INTERRUPTED" if code == 75 else "FAILED")
    assert "TDN_TOWER_DIR" not in os.environ


def test_research_cli_preserves_failed_worker_code(monkeypatch):
    rw = module("research_workflow")
    monkeypatch.setattr(rw, "load", lambda *args: {})
    monkeypatch.setattr(rw, "worker", lambda *args: (_ for _ in ()).throw(rw.WorkerFailure("child failed", 7)))
    assert rw.main(["worker", "--workflow", "unused", "--phase", "cpu"]) == 7


def test_research_reporting_failure_is_not_silent_success(workflow, monkeypatch):
    rw, _ = configure_research_worker(workflow, monkeypatch, reporting_error=True)
    with pytest.raises(rw.WorkerFailure) as captured:
        rw.worker(workflow, "cpu")
    assert captured.value.exit_code != 0
