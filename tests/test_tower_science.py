"""Reporting preserves scientific identities, outcomes and sealed evidence."""
from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"tower_test_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def isolated(monkeypatch):
    import torch
    from tdn.runtime import metadata, precision, preflight, storage
    monkeypatch.delenv("TDN_TOWER_DIR", raising=False)
    for name in ("SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_JOB_ACCOUNT",
                 "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID"):
        monkeypatch.delenv(name, raising=False)
    software = metadata.software_metadata()
    monkeypatch.setattr(metadata, "software_metadata", lambda: software)
    monkeypatch.setattr(storage, "configure_storage", lambda: ROOT)
    monkeypatch.setattr(preflight, "verify_runtime", lambda *args: None)
    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _: None)
    monkeypatch.setattr(precision, "reference_precision", lambda: None)


def only_report(source):
    directories = [path for path in source.with_name(source.name + "-tower").iterdir() if path.is_dir()]
    assert len(directories) == 1
    return directories[0]


@pytest.mark.parametrize("error,code,state", [(FloatingPointError("deliberate numerical failure"), 1, "FAILED"),
                                             (InterruptedError("signal requested"), 75, "INTERRUPTED")])
def test_research_failure_retains_exit_and_terminal_report(tmp_path, monkeypatch, isolated, error, code, state):
    from tdn.research import experiment
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(experiment, "run", fail)
    source = tmp_path / "research"
    assert load_script("research").main(["run", "--smoke", "--run-dir", str(source)]) == code
    stage = json.loads((source / "stage.json").read_text())
    summary = json.loads((only_report(source) / "summary.json").read_text())
    assert stage["status"] == summary["state"] == state
    assert summary["exit_code"] == code
    assert summary["metadata"]["scientific_status"] == state
    assert not (source / "COMPLETED").exists()
    assert "TDN_TOWER_DIR" not in os.environ
    assert "job_id" not in summary and "cpus" not in summary and "gpus" not in summary


def test_inherited_report_is_not_finalized_by_child(tmp_path, monkeypatch, isolated):
    from tdn import reporting
    from tdn.analysis import light_screen
    source = tmp_path / "light"
    report = reporting.begin_report(tmp_path / "parent", name="test-parent", script="scripts/light_screen.py")
    monkeypatch.setenv("TDN_TOWER_DIR", str(report))
    def complete(config, destination, device):
        reporting.emit({"accepted_reference_cases": 0}, phase="light-screen", completed=6, total=6, unit="cases")
        result = {"status": "COMPLETED", "headroom_passed_cases": 0}
        (destination / "summary.json").write_text(json.dumps(result))
        (destination / "summary.txt").write_text("No numerical headroom.\n")
        return result
    monkeypatch.setattr(light_screen, "run", complete)
    assert load_script("light_screen").main(["--config", str(ROOT / "configs/carc-light.yaml"),
                                            "--run-dir", str(source)]) == 0
    assert json.loads((report / "run.json").read_text())["state"] == "RUNNING"
    assert not (report / "summary.json").exists()
    assert os.environ["TDN_TOWER_DIR"] == str(report)
    assert not source.with_name(source.name + "-tower").exists()
    assert any(json.loads(line)["metrics"].get("accepted_reference_cases") == 0
               for line in (report / "metrics.jsonl").read_text().splitlines())


@pytest.mark.parametrize("publication_failure", [False, True])
def test_training_pause_keeps_exit_75_and_checkpoint_status(tmp_path, monkeypatch, isolated, publication_failure):
    import tdn.cli as cli
    import tdn.train
    from tdn.runtime.metadata import write_json
    from tdn import tower_analytics
    monkeypatch.setattr(cli, "configure_storage", lambda: ROOT)
    def pause(config, dataset, run, **kwargs):
        write_json(run / "training_result.json", {"status": "PAUSED_NEEDS_RESUME", "global_step": 3})
        raise SystemExit(75)
    monkeypatch.setattr(tdn.train, "train", pause)
    if publication_failure:
        def fail(*args, **kwargs):
            raise OSError("controlled report disk failure")
        monkeypatch.setattr(tower_analytics, "publish_outputs", fail)
    source = tmp_path / "train"
    with pytest.raises(SystemExit) as caught:
        cli.main(["train", "--config", str(ROOT / "configs/smoke.yaml"), "--run-dir", str(source)])
    assert caught.value.code == 75
    assert json.loads((source / "stage.json").read_text())["status"] == "PAUSED_NEEDS_RESUME"
    report = only_report(source)
    if not publication_failure:
        summary = json.loads((report / "summary.json").read_text())
        assert summary["state"] == "INTERRUPTED" and summary["exit_code"] == 75
        assert summary["metadata"]["scientific_status"] == "PAUSED_NEEDS_RESUME"
    else:
        assert not (report / "summary.json").exists()
    assert "TDN_TOWER_DIR" not in os.environ


def test_report_export_failure_does_not_certify_reporting_success(tmp_path, monkeypatch, isolated):
    from tdn.analysis import light_screen
    from tdn import tower_analytics
    source = tmp_path / "light"
    def complete(config, destination, device):
        result = {"status": "COMPLETED", "headroom_passed_cases": 0}
        (destination / "summary.json").write_text(json.dumps(result))
        (destination / "summary.txt").write_text("No numerical headroom.\n")
        return result
    def fail(*args, **kwargs):
        raise OSError("controlled report disk failure")
    monkeypatch.setattr(light_screen, "run", complete)
    monkeypatch.setattr(tower_analytics, "publish_outputs", fail)
    assert load_script("light_screen").main(["--config", str(ROOT / "configs/carc-light.yaml"),
                                            "--run-dir", str(source)]) == 1
    assert (source / "COMPLETED").exists()
    assert json.loads((source / "stage.json").read_text())["status"] == "COMPLETED"
    assert not (only_report(source) / "summary.json").exists()
    assert "TDN_TOWER_DIR" not in os.environ


def test_actual_research_smoke_remains_sealed_after_tower_export(tmp_path):
    from tdn.research.protocol import verify_artifacts
    source = tmp_path / "research-smoke"
    env = {name: value for name, value in os.environ.items() if name != "TDN_TOWER_DIR" and not name.startswith("SLURM_")}
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    child = subprocess.run([sys.executable, "scripts/research.py", "run", "--smoke", "--run-dir", str(source)],
                           cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
    assert child.returncode == 0, child.stdout + child.stderr
    manifest = verify_artifacts(source)
    assert all("tower" not in name for name in manifest["files"])
    report = only_report(source)
    summary = json.loads((report / "summary.json").read_text())
    assert summary["schema"] == "tower.summary/v1" and summary["state"] == "COMPLETED"
    assert summary["exit_code"] == 0 and "job_id" not in summary
    rows = [json.loads(line) for line in (report / "metrics.jsonl").read_text().splitlines()]
    phases = {row["phase"] for row in rows}
    assert {"research-references", "research-evaluation"}.issubset(phases)
    assert any(phase.startswith("research-train-") for phase in phases)
    assert any(phase.startswith("research-validation-") for phase in phases)
    assert all(math.isfinite(value) for row in rows for value in row["metrics"].values())
    assert any((report / "outputs").iterdir())
