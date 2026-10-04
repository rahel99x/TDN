"""Reporting invariants: factual evidence, independent attempts, safe files."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tdn import reporting as report


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "ROOT", tmp_path)
    for key in ("TDN_TOWER_DIR", "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID"):
        monkeypatch.delenv(key, raising=False)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "work.py").write_text("print('science')\n")
    (tmp_path / "runs").mkdir()
    return tmp_path


def begin(project, **kwargs):
    return report.begin_report(project / "runs" / "science", name="TDN/research/cpu",
                               script="scripts/work.py", parameters={"protocol": "v1"}, **kwargs)


def test_exclusive_retry_identity_and_no_science_changes(project):
    science = project / "runs/science"
    science.mkdir()
    original = b'{"scientific_status":"NUMERICAL_FAILURE"}\n'
    (science / "summary.json").write_bytes(original)
    first, second = begin(project), begin(project)
    a, b = report.read_json(first / "run.json"), report.read_json(second / "run.json")
    assert first != second and a["run_id"] != b["run_id"]
    assert a["attempt"] == 1 and b["attempt"] == 2
    assert a["experiment_id"] == b["experiment_id"]
    assert (science / "summary.json").read_bytes() == original
    assert list(science.iterdir()) == [science / "summary.json"]
    assert a["resources"] == {} and "job_id" not in a
    assert a["provenance"]["script"] == "scripts/work.py"


def test_finish_once_and_preserve_scientific_failure(project):
    path = begin(project)
    data = report.finish_report(path, state="COMPLETED", exit_code=0, runtime_seconds=1.25,
                                results={"transport": "NUMERICAL_FAILURE"})
    assert data["state"] == "COMPLETED"
    assert data["results"]["transport"] == "NUMERICAL_FAILURE"
    assert data["runtime_seconds"] == 1.25
    assert not {"cpus", "gpus", "memory_bytes", "cpu_seconds"} & data.keys()
    original = (path / "summary.json").read_bytes()
    with pytest.raises(ValueError, match="terminal"):
        report.finish_report(path, state="FAILED", exit_code=1)
    assert (path / "summary.json").read_bytes() == original


def test_metrics_filter_unknowns_keep_negative_zero_and_progress(project, monkeypatch):
    path = begin(project)
    monkeypatch.setenv("TDN_TOWER_DIR", str(path))
    report.emit({"loss": .5, "delta": -2, "zero": 0, "missing": None,
                 "infinite": float("inf"), "nan": float("nan"), "bool": True, "text": "1"},
                phase="training/family", step=1, completed=1, total=4, unit="steps")
    raw = (path / "metrics.jsonl").read_bytes()
    assert raw.endswith(b"\n")
    row = json.loads(raw.splitlines()[-1])
    assert row["metrics"] == {"loss": .5, "delta": -2, "zero": 0}
    assert row["progress"] == {"completed": 1, "total": 4, "unit": "steps"}
    assert set(row) == {"t", "metrics", "phase", "step", "progress"}
    with pytest.raises(ValueError, match="decrease"):
        report.emit({"loss": .2}, phase="training/family", step=0)
    with pytest.raises(ValueError, match="exceeds"):
        report.emit({}, phase="bad", completed=2, total=1)


def test_distinct_series_budget_across_lines(project, monkeypatch):
    path = begin(project)
    monkeypatch.setenv("TDN_TOWER_DIR", str(path))
    report.emit({f"m{i}": i for i in range(63)}, phase="samples")
    before = (path / "metrics.jsonl").read_bytes()
    with pytest.raises(ValueError, match="64 distinct"):
        report.emit({"one_too_many": 1}, phase="samples")
    assert (path / "metrics.jsonl").read_bytes() == before


def test_unattached_metrics_do_not_create_files(project):
    before = set(project.rglob("*"))
    report.emit({"loss": 1}, phase="training")
    assert set(project.rglob("*")) == before


def test_job_binding_and_array_identity(project, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "124")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "120")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "3")
    path = begin(project)
    assert report.read_json(path / "run.json")["job_id"] == "120_3"
    assert report.read_json(path / "logs.json")["job_id"] == "120_3"
    with pytest.raises(ValueError, match="differs"):
        begin(project, job_id="124")
    monkeypatch.setenv("TDN_TOWER_DIR", str(path))
    assert report.attach_report(project / "runs/child")[1] is False
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "4")
    with pytest.raises(ValueError, match="different job"):
        report.attach_report(project / "runs/child")


def test_grouped_logs_exact_paths_and_duplicate_refusal(project):
    path = begin(project)
    scheduler = project / "runs" / "scheduler.out"
    scheduler.write_text("scheduler output\n")
    relative = os.path.relpath(scheduler, path)
    report.register_log(path, "scheduler.stdout", relative, group="Scheduler", label="CPU stdout")
    report.register_log(path, "application.stderr", "outputs/future.err", group="Application")
    index = report.read_json(path / "logs.json")
    assert len(index["logs"]) == 2
    with pytest.raises(ValueError, match="unique"):
        report.register_log(path, "other", str(scheduler))
    with pytest.raises(ValueError, match="unique"):
        report.register_log(path, "scheduler.stdout", "outputs/other.log")
    with pytest.raises(ValueError, match="globs"):
        report.register_log(path, "glob", "outputs/*.log")


def test_symlinks_and_outside_paths_refused(project):
    path = begin(project)
    target = path / "outputs/target.json"
    target.write_text("{}")
    (path / "outputs/link.json").symlink_to(target)
    with pytest.raises((ValueError, OSError)):
        report.read_json(path / "outputs/link.json")
    with pytest.raises(ValueError):
        report.atomic_json(path / "outputs/link.json", {"new": 1})
    with pytest.raises(ValueError):
        report.register_log(path, "link", "outputs/link.json")
    with pytest.raises(ValueError):
        report.read_json(project.parent / "outside.json")
    (project / "runs/science").mkdir()
    (project / "runs/science/protocol.json").write_text("{}")
    with pytest.raises(ValueError, match="outside science"):
        begin(project, report_parent=project / "runs/science/tower")


def test_explicit_coordinator_tower_child_allowed(project):
    path = begin(project, report_parent=project / "runs/science/tower")
    assert path.parent == project / "runs/science/tower"


def test_duplicate_json_nonfinite_and_failed_atomic_replacement(project):
    path = project / "runs/value.json"
    path.write_text('{"x":1,"x":2}')
    with pytest.raises(ValueError, match="duplicate"):
        report.read_json(path)
    path.write_text('{"x":1e999}')
    with pytest.raises(ValueError):
        report.read_json(path)
    path.write_text('{"keep":1}')
    with pytest.raises(ValueError):
        report.atomic_json(path, {"bad": float("nan")})
    assert path.read_text() == '{"keep":1}'
    assert not list(path.parent.glob(".value.json.*"))


def test_planning_retains_failure_and_omits_synthetic_scaling(project):
    completed, failed = begin(project), begin(project)
    report.finish_report(completed, state="COMPLETED", exit_code=0, runtime_seconds=2)
    report.finish_report(failed, state="INTERRUPTED", exit_code=75, runtime_seconds=1)
    bundle = report.export_planning([completed, failed], reference_run=completed)
    assert [r["state"] for r in bundle["history"]] == ["COMPLETED", "INTERRUPTED"]
    assert bundle["query"]["name"] == "TDN/research/cpu"
    assert "cpus" not in bundle["query"] and bundle["scaling"] == []
    with pytest.raises(ValueError, match="duplicate"):
        report.export_planning([completed, completed])
    destination = project / "runs/science.json"
    destination.write_text('{"science":true}')
    with pytest.raises(ValueError, match="non-planning"):
        report.export_planning([completed], output=destination)
    assert destination.read_text() == '{"science":true}'


def test_summary_invalid_completion_does_not_seal_report(project):
    path = begin(project)
    with pytest.raises(ValueError, match="exit_code zero"):
        report.finish_report(path, state="COMPLETED", exit_code=1)
    assert not (path / "summary.json").exists()
    assert report.read_json(path / "run.json")["state"] == "RUNNING"
    report.finish_report(path, state="FAILED", exit_code=1)


def test_early_reports_keep_verified_software_in_planning_identity(project):
    first, second = begin(project), begin(project)
    for path, fingerprint in ((first, "a" * 64), (second, "b" * 64)):
        report.finish_report(path, state="COMPLETED", exit_code=0,
                             observed_parameters={"software_sha256": fingerprint})
    originals = {(path / "summary.json"): (path / "summary.json").read_bytes() for path in (first, second)}
    bundle = report.export_planning([first, second], reference_run=first)
    assert [row["parameters"]["software_sha256"] for row in bundle["history"]] == ["a" * 64, "b" * 64]
    assert bundle["query"]["parameters"]["software_sha256"] == "a" * 64
    assert all(path.read_bytes() == original for path, original in originals.items())


def test_observed_software_cannot_replace_requested_fingerprint(project):
    path = report.begin_report(project / "runs/science", name="TDN/research/cpu", script="scripts/work.py",
                               parameters={"software_sha256": "a" * 64})
    with pytest.raises(ValueError, match="cannot overwrite"):
        report.finish_report(path, state="COMPLETED", exit_code=0,
                             observed_parameters={"software_sha256": "b" * 64})
    assert not (path / "summary.json").exists()
    assert report.read_json(path / "run.json")["state"] == "RUNNING"


def test_scheduler_reconciliation_keeps_application_runtime_and_outcome(project):
    path = begin(project, job_id="123")
    report.finish_report(path, state="INTERRUPTED", exit_code=75, runtime_seconds=2,
                         results={"checkpoint": "retained"})
    evidence = {"job_id": "123", "state": "TIMEOUT", "exit_code": "0:15", "elapsed_seconds": 1800,
                "command": ["sacct", "-nP", "-j", "123", "--format=JobIDRaw,State,ExitCode,ElapsedRaw"]}
    summary = report.reconcile_report(path, evidence)
    assert summary["state"] == "TIMEOUT" and summary["runtime_seconds"] == 2
    assert summary["metadata"]["application_state"] == "INTERRUPTED"
    assert summary["metadata"]["application_exit_code"] == 75
    assert summary["results"]["checkpoint"] == "retained"
    assert report.reconcile_report(path, evidence) == summary
    with pytest.raises(ValueError, match="exact job_id"):
        report.reconcile_report(path, {**evidence, "job_id": "123.batch"})


def test_reconciliation_without_summary_keeps_unknown_duration(project):
    path = begin(project, job_id="124")
    summary = report.reconcile_report(path, {"job_id": "124", "state": "CANCELLED", "exit_code": "0:15",
        "elapsed_seconds": 30, "command": ["sacct", "-j", "124"]})
    assert "runtime_seconds" not in summary and "end" not in summary
    assert summary["metadata"]["application_state"] == "RUNNING"


@pytest.mark.parametrize("application_state", ["FAILED", "INTERRUPTED", "RUNNING"])
def test_successful_scheduler_exit_cannot_repair_application_failure(project, application_state):
    path = begin(project, job_id="125")
    if application_state != "RUNNING":
        report.finish_report(path, state=application_state, exit_code=1, runtime_seconds=3)
    summary = report.reconcile_report(path, {"job_id": "125", "state": "COMPLETED", "exit_code": "0:0",
        "elapsed_seconds": 4, "command": ["sacct", "-j", "125"]})
    assert summary["state"] == ("UNKNOWN" if application_state == "RUNNING" else application_state)
    assert summary["metadata"]["scheduler_reconciliation"]["evidence"]["state"] == "COMPLETED"
    if application_state != "RUNNING":
        assert summary["exit_code"] == 1


def test_planning_rejects_copies_of_same_job_or_import(project):
    first, second = begin(project, job_id="123"), begin(project, job_id="123")
    for path in (first, second):
        report.finish_report(path, state="COMPLETED", exit_code=0)
    with pytest.raises(ValueError, match="scheduler observation"):
        report.export_planning([first, second])
    third, fourth = begin(project), begin(project)
    for path in (third, fourth):
        report.finish_report(path, state="FAILED", exit_code=1, metadata={"source_observation_id": "abc"})
    with pytest.raises(ValueError, match="imported observation"):
        report.export_planning([third, fourth])


def test_reporting_import_does_not_import_numerical_packages():
    process = subprocess.run([sys.executable, "-c",
        "import sys; import tdn.reporting; assert not {'torch','numpy','yaml'} & sys.modules.keys()"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert process.returncode == 0, process.stderr


def test_all_six_vendored_schemas_with_generated_reports(project):
    jsonschema = pytest.importorskip("jsonschema", reason="optional schema validation development dependency")
    path = begin(project)
    report.finish_report(path, state="FAILED", exit_code=1, runtime_seconds=.1)
    planning = report.export_planning([path], reference_run=path)
    repo = Path(__file__).resolve().parents[1]
    schemas = repo / ".tower/schemas"
    cases = {"run": report.read_json(path / "run.json"), "logs": report.read_json(path / "logs.json"),
             "summary": report.read_json(path / "summary.json"), "planning": planning,
             "metrics": json.loads((path / "metrics.jsonl").read_text().splitlines()[0]),
             "output-contract": json.loads((repo / ".tower/contracts/outputs.v1.json").read_text())}
    from referencing import Registry, Resource
    registry = Registry().with_resources((
        file.as_uri(), Resource.from_contents(json.loads(file.read_text())))
        for file in schemas.glob("*.schema.json"))
    for name, value in cases.items():
        file = schemas / f"{name}.v1.schema.json"
        schema = {**json.loads(file.read_text()), "$id": file.as_uri()}
        jsonschema.Draft202012Validator(schema, registry=registry).validate(value)
