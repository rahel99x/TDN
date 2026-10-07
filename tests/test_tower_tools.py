"""Project-only Tower tooling: bounded paths, truthful history and exact jobs."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tdn_tower_tools", ROOT / "scripts/tower.py")
tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tools)


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def report_fixture(root, relative="runs/unit/tower/attempt-1", *, state="COMPLETED", job=None):
    report = root / relative
    report.mkdir(parents=True)
    manifest = {"schema": "tower.run/v1", "run_id": report.name,
                "experiment_id": "test", "attempt": 1, "state": state}
    logs = {"schema": "tower.logs/v1", "run_id": report.name, "logs": []}
    summary = {"schema": "tower.summary/v1", "id": report.name, "name": "tdn/test", "state": state}
    if job:
        manifest["job_id"] = summary["job_id"] = logs["job_id"] = job
    write_json(report / "run.json", manifest)
    write_json(report / "logs.json", logs)
    if state != "RUNNING":
        write_json(report / "summary.json", summary)
    (report / "metrics.jsonl").write_text('{"t":1234,"metrics":{"loss":0.1},"phase":"training","step":1}\n')
    return report


def test_discovery_uses_report_layouts_without_cache_recursion(tmp_path):
    grouped = report_fixture(tmp_path)
    standalone = report_fixture(tmp_path, "runs/experiment-tower/attempt-2")
    stage = report_fixture(tmp_path, "runs/workflow/train-tower/attempt-3")
    report_fixture(tmp_path, ".cache/deep/tower/ignored")
    (tmp_path / "runs/evil").symlink_to(tmp_path / ".cache", target_is_directory=True)
    paths = [path for path, _ in tools.find_reports(tmp_path)]
    assert paths == sorted((grouped, standalone, stage))
    assert [path for path, _ in tools.find_reports(tmp_path, "runs/unit")] == [grouped]
    assert [path for path, _ in tools.find_reports(tmp_path, standalone)] == [standalone]


def test_discovery_stops_at_scan_budget(tmp_path, monkeypatch):
    (tmp_path / "runs").mkdir()
    for index in range(4):
        (tmp_path / "runs" / str(index)).mkdir()
    monkeypatch.setattr(tools, "MAX_ENTRIES", 2)
    with pytest.raises(ValueError, match="discovery limit"):
        tools.find_reports(tmp_path)


@pytest.mark.parametrize("path", ["../outside", "/outside", "runs/*", "runs/../unit", "runs//unit"])
def test_paths_refuse_escapes_and_globs(tmp_path, path):
    with pytest.raises(ValueError):
        tools.project_path(tmp_path, path)


def test_paths_refuse_report_symlink(tmp_path):
    report = report_fixture(tmp_path)
    linked = tmp_path / "alias"
    linked.symlink_to(report, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        tools.project_path(tmp_path, linked)


def test_show_is_exact_and_never_launches(tmp_path, monkeypatch, capsys):
    report = report_fixture(tmp_path)
    monkeypatch.setattr(tools, "ROOT", tmp_path)
    monkeypatch.setattr(tools.subprocess, "run", lambda *a, **k: pytest.fail("show must not launch Tower"))
    assert tools.main(["show", str(report)]) == 0
    assert capsys.readouterr().out.strip() == (
        f"tower --profile carc --config {tmp_path}/.tower/config.json --workdir {report} --tab research --research-view experiment")


def test_fedora_report_uses_its_local_tower_profile(tmp_path, monkeypatch, capsys):
    report = report_fixture(tmp_path)
    manifest = tools.read_json(report / "run.json")
    manifest["parameters"] = {"execution_mode": "desktop-slurm"}
    write_json(report / "run.json", manifest)
    monkeypatch.setattr(tools, "ROOT", tmp_path)
    assert tools.main(["show", str(report)]) == 2
    assert "configuration is missing" in capsys.readouterr().err
    (tmp_path / ".tower").mkdir()
    config = tmp_path / ".tower/fedora-slurm.json"
    write_json(config, {"user": "rahel", "profiles": {"desktop-slurm": {}}})
    monkeypatch.setattr(tools.subprocess, "run", lambda *a, **kw: pytest.fail("show must not launch Tower"))
    assert tools.main(["show", str(report)]) == 0
    command = capsys.readouterr().out
    assert "--profile desktop-slurm" in command
    assert f"--config {config}" in command
    assert "--profile carc" not in command


def test_latest_selects_report_creation_and_announces_identity(tmp_path, monkeypatch, capsys):
    older = report_fixture(tmp_path, "runs/older/tower/attempt-1", job="111")
    newer = report_fixture(tmp_path, "runs/newer/tower/attempt-2", state="FAILED", job="222")
    for report, timestamp in ((older, 10), (newer, 20)):
        manifest = tools.read_json(report / "run.json")
        manifest.update(start=timestamp, name="tdn/test", metadata={"source_directory": report.parent.parent.relative_to(tmp_path).as_posix()})
        write_json(report / "run.json", manifest)
    monkeypatch.setattr(tools, "ROOT", tmp_path)
    assert tools.main(["show", "latest"]) == 0
    output = capsys.readouterr()
    assert str(newer) in output.out
    assert "job: 222" in output.err and "state: FAILED" in output.err


def test_latest_with_no_report_does_not_guess_a_workflow(tmp_path):
    with pytest.raises(ValueError, match="No Tower reports"):
        tools.selected_report(tmp_path, "latest")


def test_inspection_is_labeled_structural_and_allows_live_summary_absent(tmp_path):
    report = report_fixture(tmp_path, state="RUNNING")
    result = tools.inspect_report(tmp_path, report)
    assert result["validation"] == "tdn_structural_checks_only"
    assert result["terminal_summary_present"] is False
    assert result["metric_rows"] == 1


@pytest.mark.parametrize("row", [
    '{"t":1,"metrics":{"x":NaN}}\n',
    '{"t":1,"metrics":{"x":true}}\n',
    '{"t":1,"metrics":{"x":1,"x":2}}\n',
    '{"t":1,"metrics":{},"progress":{"completed":3,"total":2}}\n',
    '{"t":1,"metrics":{},"step":-1}\n',
    '{"t":1,"metrics":{}}',
])
def test_inspection_rejects_malformed_metrics(tmp_path, row):
    report = report_fixture(tmp_path)
    (report / "metrics.jsonl").write_text(row)
    with pytest.raises(ValueError):
        tools.inspect_report(tmp_path, report)


def test_inspection_rejects_metrics_symlink(tmp_path):
    report = report_fixture(tmp_path)
    target = report / "metrics.jsonl"
    target.unlink()
    target.symlink_to(report / "run.json")
    with pytest.raises(OSError):
        tools.inspect_report(tmp_path, report)


def test_native_validation_uses_explicit_project_bindings(tmp_path, monkeypatch):
    report = report_fixture(tmp_path)
    calls = []
    monkeypatch.setattr(tools, "ROOT", tmp_path)
    monkeypatch.setattr(tools.shutil, "which", lambda _: "/opt/tower/bin/tower")
    monkeypatch.setattr(tools.subprocess, "run", lambda command, **kw: calls.append((command, kw)) or SimpleNamespace(returncode=0))
    assert tools.main(["validate", str(report), "--native"]) == 0
    assert calls[0][0] == ["tower", "--no-state", "--no-plugins", "--config", str(tmp_path / ".tower/config.json"),
                          "run", "validate", str(tmp_path / ".tower/contracts/outputs.v1.json"), str(report)]
    assert calls[0][1]["cwd"] == tmp_path


def test_accounting_requires_exact_allocation_and_preserves_signal(tmp_path):
    calls = []
    def runner(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="123.batch|OUT_OF_MEMORY|0:9|8\n123|CANCELLED by 42|0:15|9\n", stderr="")
    result = tools.accounting_record("123", runner=runner)
    assert result["state"] == "CANCELLED"
    assert result["exit_code"] == "0:15"
    assert calls[0][calls[0].index("--jobs") + 1] == "123"
    assert "--allocations" in calls[0]


@pytest.mark.parametrize("output", ["123.batch|COMPLETED|0:0|3\n", "123|RUNNING|0:0|3\n",
                                      "123|OUT_OF_ME+|0:9|3\n", "123|FAILED|1:0|3\n123|FAILED|1:0|3\n"])
def test_accounting_does_not_guess_missing_ambiguous_or_live_states(output):
    runner = lambda *a, **k: SimpleNamespace(returncode=0, stdout=output, stderr="")
    with pytest.raises(ValueError):
        tools.accounting_record("123", runner=runner)


def test_accounting_refuses_shell_injection_or_steps():
    for job in ("123; true", "123.batch", "123_[1-4]", None):
        with pytest.raises(ValueError):
            tools.accounting_record(job, runner=lambda *a, **k: pytest.fail("invalid job queried"))


@pytest.mark.parametrize("state", ["COMPLETED", "FAILED", "PAUSED_NEEDS_RESUME"])
def test_historical_import_preserves_original_files_and_unknown_facts(tmp_path, state):
    source = tmp_path / "runs/history/experiment"
    source.mkdir(parents=True)
    original = {"stage": "research-run", "status": state, "actually_ran": True,
                "elapsed_seconds": 12.25, "software": {"execution_mode": "local-cpu"}}
    write_json(source / "stage.json", original)
    before = (source / "stage.json").read_bytes()
    report = tools.import_report(tmp_path, source)
    summary = tools.read_json(report / "summary.json")
    manifest = tools.read_json(report / "run.json")
    assert (source / "stage.json").read_bytes() == before
    assert set(source.iterdir()) == {source / "stage.json"}
    assert summary["runtime_seconds"] == 12.25
    assert summary["state"] == ("INTERRUPTED" if state == "PAUSED_NEEDS_RESUME" else state)
    assert summary["metadata"]["imported_observation"] is True
    assert summary["metadata"]["independent_repeat"] is False
    assert all(key not in summary for key in ("job_id", "cpus", "gpus", "memory_bytes", "start", "end", "script_sha256"))
    assert "start" not in manifest and "end" not in manifest
    metrics = json.loads((report / "metrics.jsonl").read_text())
    assert metrics["phase"] == "historical-report-export" and metrics["metrics"] == {}
    assert tools.inspect_report(tmp_path, report)["terminal_summary_present"]


def test_import_requires_terminal_evidence_before_creating_report(tmp_path):
    source = tmp_path / "runs/live"
    source.mkdir(parents=True)
    write_json(source / "stage.json", {"stage": "train", "status": "RUNNING", "actually_ran": True})
    with pytest.raises(ValueError, match="terminal"):
        tools.import_report(tmp_path, source)
    assert not (source.parent / "live-tower").exists()


def test_repeated_historical_import_is_not_an_independent_observation(tmp_path):
    source = tmp_path / "runs/history"
    source.mkdir(parents=True)
    write_json(source / "stage.json", {"stage": "train", "status": "FAILED", "actually_ran": True})
    first = tools.import_report(tmp_path, source)
    second = tools.import_report(tmp_path, source)
    assert first != second
    assert tools.read_json(first / "summary.json")["metadata"]["source_observation_id"] == tools.read_json(second / "summary.json")["metadata"]["source_observation_id"]
    with pytest.raises(ValueError, match="duplicate|observation"):
        tools.reporting.export_planning([first, second], output=tmp_path / "reports/planning.json")


def test_export_preserves_explicit_reference_and_destination(tmp_path, monkeypatch):
    first = report_fixture(tmp_path)
    second = report_fixture(tmp_path, "runs/unit/tower/attempt-2")
    monkeypatch.setattr(tools, "ROOT", tmp_path)
    calls = []
    monkeypatch.setattr(tools.reporting, "export_planning", lambda reports, **kwargs: calls.append((reports, kwargs)) or {})
    assert tools.main(["export", str(first), str(second), "--reference", str(second)]) == 0
    assert calls == [([first, second], {"reference_run": second, "output": tmp_path / "reports/planning.json"})]


def test_read_only_cli_does_not_import_torch():
    code = f"import runpy,sys; sys.argv=['tower.py','--help'];\ntry: runpy.run_path({str(ROOT / 'scripts/tower.py')!r},run_name='__main__')\nexcept SystemExit: pass\nassert 'torch' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
