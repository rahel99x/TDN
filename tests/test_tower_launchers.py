"""Manual job reporting uses real identity evidence and preserves child exits."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tdn_tower_task", ROOT / "scripts/tower_task.py")
task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)


def test_manual_allocation_context_uses_exact_owned_job(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "1234")
    monkeypatch.setenv("SLURM_STEP_ID", "0")
    monkeypatch.setenv("TDN_DEVICE", "cuda")
    output = ("JobId=1234 JobName=tdn-gpu-tests Account=anakano_81 UserId=aadaniel(99) "
              "NumCPUs=4 NumNodes=1 Partition=gpu QOS=normal AllocTRES=cpu=4,gres/gpu=1 "
              f"StdOut={ROOT}/logs/test.out StdErr={ROOT}/logs/test.err")
    monkeypatch.setattr(task.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 0, output, ""))
    job, resources, logs = task.scheduler_context("gpu-tests")
    assert job == "1234" and resources["cpus"] == 4 and resources["gpus"] == 1
    assert "memory_bytes" not in resources and "cpu_seconds" not in resources
    assert logs[0]["path"] == f"{ROOT}/logs/test.out"
    with pytest.raises(ValueError, match="identity"):
        task.scheduler_context("train")


def test_manual_context_omits_unknown_gpu_and_requires_task(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "1234")
    monkeypatch.setenv("SLURM_STEP_ID", "0")
    monkeypatch.setenv("TDN_DEVICE", "cuda")
    output = "JobId=1234 JobName=tdn-train Account=anakano_81 UserId=aadaniel(99)"
    monkeypatch.setattr(task.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 0, output, ""))
    assert task.scheduler_context("train")[1] == {"account": "anakano_81"}
    monkeypatch.delenv("SLURM_STEP_ID")
    with pytest.raises(ValueError, match="srun"):
        task.scheduler_context("train")


@pytest.mark.parametrize("code,state", [(0, "COMPLETED"), (17, "FAILED"), (75, "INTERRUPTED")])
def test_manual_wrapper_preserves_exit_and_output_evidence(tmp_path, monkeypatch, code, state):
    source = tmp_path / "manual"
    source.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text("purpose: development\n")
    for key in tuple(os.environ):
        if key.startswith("SLURM_") or key == "TDN_TOWER_DIR":
            monkeypatch.delenv(key)
    monkeypatch.setattr(task, "scheduler_context", lambda stage: ("1234", {"gpus": 0}, []))
    result = task.execute([sys.executable, "-c", f"raise SystemExit({code})"],
                          source=source, config=config, stage="cpu-tests")
    assert result == code
    paths = list((tmp_path / "manual-tower").glob("*/summary.json"))
    assert len(paths) == 1
    summary = json.loads(paths[0].read_text())
    assert summary["state"] == state and summary["exit_code"] == code
    assert summary["job_id"] == "1234"
    assert "cpu_seconds" not in summary and "memory_bytes" not in summary
    assert (paths[0].parent / "outputs/artifacts.json").is_file()
    assert "TDN_TOWER_DIR" not in os.environ


def test_manual_wrapper_does_not_mask_original_failure(tmp_path, monkeypatch):
    from tdn import tower_analytics
    source = tmp_path / "manual"
    source.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text("purpose: development\n")
    for key in tuple(os.environ):
        if key.startswith("SLURM_") or key == "TDN_TOWER_DIR":
            monkeypatch.delenv(key)
    monkeypatch.setattr(task, "scheduler_context", lambda stage: ("1234", {}, []))
    monkeypatch.setattr(tower_analytics, "publish_outputs", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    assert task.execute([sys.executable, "-c", "raise SystemExit(17)"],
                        source=source, config=config, stage="cpu-tests") == 17
