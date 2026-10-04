"""Grouped-worker unit tests with explicit isolated root/scheduler substitutes."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def worker_project(tmp_path, monkeypatch):
    """Only copied scripts see simulated allocations; no scheduler is invoked."""
    root = tmp_path / "project"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    for name in ("carc_worker.sh", "carc_cpu.sbatch", "carc_gpu.sbatch"):
        (scripts / name).write_text((ROOT / "scripts" / name).read_text())
    common = (ROOT / "scripts/common.sh").read_text()
    common = common.replace(
        "TDN_CARC_ROOT=/home1/aadaniel/projects/TDN",
        "TDN_CARC_ROOT=" + shlex.quote(str(root)),
    )
    # Runtime policy remains active in the copied library. The interpreter
    # substitute loads/validates real config but never calls a scientific stage.
    common += "\ntdn_python() { printf '%s\\n' " + shlex.quote(sys.executable) + "; }\n"
    (scripts / "common.sh").write_text(common)
    commands = root / "commands"
    commands.mkdir()
    (commands / "id").write_text("#!/usr/bin/env bash\nprintf 'aadaniel\\n'\n")
    (commands / "id").chmod(0o755)
    (scripts / "carc_workflow.py").write_text(
        f"import sys\nsys.path.insert(0, {str(ROOT)!r})\n" +
        """import argparse, json, os, signal, time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('command')
p.add_argument('--workflow',required=True)
p.add_argument('--phase',required=True)
p.add_argument('--stage')
p.add_argument('--status')
p.add_argument('--exit-code',type=int)
p.add_argument('--interrupted',action='store_true')
a=p.parse_args()
root=Path(a.workflow).parent
calls=root/'calls.json'
items=json.loads(calls.read_text()) if calls.exists() else []
items.append(vars(a))
calls.write_text(json.dumps(items))
if a.command=='worker-report-begin':
    from tdn.reporting import begin_report
    report=begin_report(root, name='TDN/mocked-grouped-worker', script='scripts/carc_worker.sh',
                        job_id=os.environ['SLURM_JOB_ID'], report_parent=root/'tower',
                        metadata={'validation_scope':'mocked scheduler unit test'})
    (root/'tower-start.json').write_text(json.dumps({'started':time.monotonic()}))
    print(report)
if a.command=='worker-report-finish':
    from tdn.reporting import finish_report
    started=json.loads((root/'tower-start.json').read_text())['started']
    state='INTERRUPTED' if a.interrupted or a.exit_code==75 else 'FAILED' if a.exit_code else 'COMPLETED'
    finish_report(Path(os.environ['TDN_TOWER_DIR']), state=state,
                  runtime_seconds=time.monotonic()-started, exit_code=a.exit_code,
                  results={'validation_scope':'mocked scheduler unit test'})
if a.command=='worker-verify' and os.environ.get('MOCK_VERIFY_FAIL')=='1':
    raise SystemExit(2)
if a.command=='worker-venv-ready':
    raise SystemExit(0 if (root/'venv.ready').exists() else 1)
if a.command=='worker-stage-complete':
    raise SystemExit(0 if a.stage in os.environ.get('MOCK_COMPLETED','').split(',') else 1)
if (a.command=='worker-mark' and a.status=='COMPLETED'
    and a.stage==os.environ.get('MOCK_SIGNAL_AFTER_STAGE')):
    os.kill(os.getppid(), signal.SIGUSR1)
"""
    )
    (scripts / "run_stage.sh").write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s|%s|%s\\n' "$TDN_STAGE" "$TDN_DEVICE" "$TDN_RESUME" >> "$TDN_WORKFLOW_ROOT/executed.txt"
if [[ "$TDN_STAGE" == setup ]]; then touch "$TDN_WORKFLOW_ROOT/venv.ready"; fi
if [[ "$TDN_STAGE" == "${MOCK_WAIT_STAGE:-}" ]]; then
    stop=false
    trap 'stop=true' USR1 TERM
    touch "$TDN_WORKFLOW_ROOT/child.started"
    while [[ "$stop" == false ]]; do read -r -t 0.05 ignored || true; done
    touch "$TDN_WORKFLOW_ROOT/checkpoint.paused"
    exit 75
fi
if [[ "$TDN_STAGE" == "${MOCK_FAIL_STAGE:-}" ]]; then exit "${MOCK_FAIL_CODE:-17}"; fi
"""
    )
    base = root / "runs" / "unit"
    base.mkdir(parents=True)
    (base / "config.yaml").write_text((ROOT / "configs/smoke.yaml").read_text())
    (base / "workflow.json").write_text("{}\n")
    (base / "venv.ready").touch()
    for name in list(os.environ):
        if name.startswith(("SLURM_", "TDN_", "MOCK_")) or name in (
            "CONDA_PREFIX", "CONDA_SHLVL", "CARC_ACCOUNT",
        ):
            monkeypatch.delenv(name, raising=False)
    values = {
        "PATH": str(commands) + os.pathsep + os.environ["PATH"],
        "CARC_ACCOUNT": "anakano_81",
        "SLURM_JOB_ID": "700001",
        "SLURM_STEP_ID": "0",
        "SLURM_JOB_ACCOUNT": "anakano_81",
        "TDN_WORKFLOW": str(base / "workflow.json"),
        "TDN_WORKFLOW_ROOT": str(base),
        "TDN_WORKFLOW_PHASE": "cpu",
        "TDN_WORKFLOW_ACTION": "start",
        "TDN_CONFIG": str(base / "config.yaml"),
        "TDN_DATASET": str(base / "dataset"),
        "TDN_SETUP_MODE": "auto",
        "TDN_PILOT_BUDGET": "12",
        "TDN_RESUME": "none",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    return root, base


def run_worker(project, **environment):
    root, _base = project
    env = os.environ.copy()
    env.update(environment)
    return subprocess.run(
        ["bash", str(root / "scripts/carc_worker.sh")],
        env=env, text=True, capture_output=True, timeout=30,
    )


def executed(base):
    path = base / "executed.txt"
    return [line.split("|") for line in path.read_text().splitlines()] if path.exists() else []


def marks(base):
    path = base / "calls.json"
    calls = json.loads(path.read_text()) if path.exists() else []
    return [
        (item["stage"], item["status"], item["exit_code"])
        for item in calls if item["command"] == "worker-mark"
    ]


def test_cpu_auto_reuses_ready_venv_and_finishes_three_stages(worker_project):
    _root, base = worker_project
    result = run_worker(worker_project)
    assert result.returncode == 0, result.stderr
    assert [item[0] for item in executed(base)] == ["cpu-tests", "audit", "generate"]
    assert ("setup", "COMPLETED", 0) in marks(base)
    assert all(item[1:] == ["cpu", "none"] for item in executed(base))
    summaries = list((base / "tower").glob("tdn-*/summary.json"))
    assert len(summaries) == 1
    assert json.loads(summaries[0].read_text())["state"] == "COMPLETED"


@pytest.mark.parametrize("mode,initial_ready", [("auto", False), ("always", True)])
def test_cpu_allocated_setup_runs_when_requested(worker_project, mode, initial_ready):
    _root, base = worker_project
    if not initial_ready:
        (base / "venv.ready").unlink()
    result = run_worker(worker_project, TDN_SETUP_MODE=mode)
    assert result.returncode == 0, result.stderr
    assert [item[0] for item in executed(base)] == ["setup", "cpu-tests", "audit", "generate"]
    assert ("setup", "RUNNING", 0) in marks(base)
    assert ("setup", "COMPLETED", 0) in marks(base)


def test_cpu_never_does_not_install_missing_venv(worker_project):
    _root, base = worker_project
    (base / "venv.ready").unlink()
    result = run_worker(worker_project, TDN_SETUP_MODE="never")
    assert result.returncode == 2
    assert executed(base) == []
    assert "exact project venv is not ready" in result.stderr


def test_setup_only_installs_on_cpu_without_optimizer_budget(worker_project):
    _root, base = worker_project
    (base / "venv.ready").unlink()
    # Package setup does not claim the configuration has passed scientific tests.
    (base / "config.yaml").write_text("not a training configuration\n")
    result = run_worker(worker_project, TDN_WORKFLOW_ACTION="setup", TDN_PILOT_BUDGET="")
    assert result.returncode == 0, result.stderr
    assert executed(base) == [["setup", "cpu", "none"]]
    assert "setup-only" in result.stdout


def test_gpu_executes_all_stages_with_only_selected_checkpoint_for_reports(worker_project):
    _root, base = worker_project
    resume = str(base / "train/checkpoints/last.pt")
    result = run_worker(worker_project, TDN_WORKFLOW_PHASE="gpu", TDN_RESUME=resume)
    assert result.returncode == 0, result.stderr
    assert executed(base) == [
        ["gpu-tests", "cuda", "none"],
        ["calibrate", "cuda", "none"],
        ["train", "cuda", resume],
        ["evaluate", "cuda", str(base / "train/checkpoints/best.pt")],
        ["benchmark", "cuda", str(base / "train/checkpoints/best.pt")],
    ]


@pytest.mark.parametrize("code,status", [("17", "FAILED"), ("75", "PAUSED_NEEDS_RESUME")])
def test_failed_or_paused_train_stops_successors(worker_project, code, status):
    _root, base = worker_project
    result = run_worker(
        worker_project, TDN_WORKFLOW_PHASE="gpu",
        MOCK_FAIL_STAGE="train", MOCK_FAIL_CODE=code,
    )
    assert result.returncode == int(code), result.stderr
    assert [item[0] for item in executed(base)] == ["gpu-tests", "calibrate", "train"]
    assert ("train", status, int(code)) in marks(base)
    assert not any(item[0] in ("evaluate", "benchmark") for item in marks(base))
    summary = json.loads(next((base / "tower").glob("tdn-*/summary.json")).read_text())
    assert summary["exit_code"] == int(code)
    assert summary["state"] == ("INTERRUPTED" if code == "75" else "FAILED")


def test_completed_prerequisites_are_preserved_on_checked_resume(worker_project):
    _root, base = worker_project
    result = run_worker(
        worker_project, TDN_WORKFLOW_PHASE="gpu", MOCK_COMPLETED="gpu-tests,calibrate",
    )
    assert result.returncode == 0, result.stderr
    assert [item[0] for item in executed(base)] == ["train", "evaluate", "benchmark"]
    assert "preserved verified completed stage gpu-tests" in result.stdout
    assert not any(item[0] in ("gpu-tests", "calibrate") for item in marks(base))


def test_gpu_never_installs_packages_when_venv_is_missing(worker_project):
    _root, base = worker_project
    (base / "venv.ready").unlink()
    result = run_worker(worker_project, TDN_WORKFLOW_PHASE="gpu")
    assert result.returncode == 2
    assert executed(base) == []


def test_strict_config_checks_optimizer_budget_before_compute(worker_project):
    _root, base = worker_project
    config = yaml.safe_load((base / "config.yaml").read_text())
    config["training"]["max_steps"] = 13
    (base / "config.yaml").write_text(yaml.safe_dump(config))
    result = run_worker(worker_project)
    assert result.returncode != 0
    assert "exceeds the explicitly authorized optimizer budget" in result.stderr
    assert executed(base) == []


@pytest.mark.parametrize(
    "values,expected",
    [
        ({"SLURM_JOB_ACCOUNT": "other_lab"}, "required charging account"),
        ({"TDN_WORKFLOW_PHASE": "gpu", "TDN_WORKFLOW_ACTION": "setup"}, "requires a CPU phase"),
        ({"MOCK_VERIFY_FAIL": "1"}, ""),
    ],
)
def test_guard_failure_starts_no_stage(worker_project, values, expected):
    _root, base = worker_project
    result = run_worker(worker_project, **values)
    assert result.returncode != 0
    assert expected in result.stderr
    assert executed(base) == []


def test_worker_rejects_different_workflow_layout(worker_project):
    _root, base = worker_project
    result = run_worker(worker_project, TDN_DATASET=str(base / "different-dataset"))
    assert result.returncode == 2
    assert "immutable workflow layout" in result.stderr
    assert executed(base) == []


@pytest.mark.parametrize("stop_signal", [signal.SIGUSR1, signal.SIGTERM])
def test_signal_during_train_wait_preserves_child_pause_status(worker_project, stop_signal):
    root, base = worker_project
    env = os.environ.copy()
    env.update(TDN_WORKFLOW_PHASE="gpu", MOCK_WAIT_STAGE="train")
    child = subprocess.Popen(
        ["bash", str(root / "scripts/carc_worker.sh")],
        env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 15
        while not (base / "child.started").exists():
            if child.poll() is not None:
                stdout, stderr = child.communicate()
                pytest.fail(f"Worker exited before signal: {stdout}\n{stderr}")
            if time.monotonic() > deadline:
                pytest.fail("Mocked worker startup exceeded 15 seconds")
            time.sleep(0.01)
        child.send_signal(stop_signal)
        stdout, stderr = child.communicate(timeout=10)
        assert child.returncode == 75, stdout + stderr
    finally:
        if child.poll() is None:
            child.terminate()
            child.communicate(timeout=10)
    assert (base / "checkpoint.paused").exists()
    assert ("train", "PAUSED_NEEDS_RESUME", 75) in marks(base)
    assert [item[0] for item in executed(base)] == ["gpu-tests", "calibrate", "train"]


def test_usr1_between_stages_starts_no_successor(worker_project):
    _root, base = worker_project
    result = run_worker(
        worker_project, TDN_WORKFLOW_PHASE="gpu", MOCK_SIGNAL_AFTER_STAGE="gpu-tests",
    )
    assert result.returncode == 75, result.stdout + result.stderr
    assert [item[0] for item in executed(base)] == ["gpu-tests"]
    assert ("gpu-tests", "COMPLETED", 0) in marks(base)
    assert "no successor stage started" in result.stderr


def test_shared_venv_lock_blocks_another_phase_until_child_finishes(worker_project):
    root, base = worker_project
    env = os.environ.copy()
    env.update(TDN_WORKFLOW_PHASE="gpu", MOCK_WAIT_STAGE="train")
    child = subprocess.Popen(
        ["bash", str(root / "scripts/carc_worker.sh")],
        env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 15
        while not (base / "child.started").exists():
            if child.poll() is not None:
                stdout, stderr = child.communicate()
                pytest.fail(f"First phase exited before lock check: {stdout}\n{stderr}")
            if time.monotonic() > deadline:
                pytest.fail("Mocked locked phase startup exceeded 15 seconds")
            time.sleep(0.01)
        blocked = run_worker(worker_project, TDN_WORKFLOW_PHASE="cpu")
        assert blocked.returncode == 2, blocked.stdout + blocked.stderr
        assert "Another CARC phase is using this checkout" in blocked.stderr
        assert [item[0] for item in executed(base)] == ["gpu-tests", "calibrate", "train"]
        child.send_signal(signal.SIGUSR1)
        stdout, stderr = child.communicate(timeout=10)
        assert child.returncode == 75, stdout + stderr
    finally:
        if child.poll() is None:
            child.terminate()
            child.communicate(timeout=10)
    # The lock is released after the safe child exit, without deleting a run.
    ready = run_worker(worker_project, TDN_WORKFLOW_PHASE="cpu")
    assert ready.returncode == 0, ready.stdout + ready.stderr
    assert [item[0] for item in executed(base)][-3:] == ["cpu-tests", "audit", "generate"]


def test_grouped_batch_files_share_one_srun_per_phase():
    cpu = (ROOT / "scripts/carc_cpu.sbatch").read_text()
    gpu = (ROOT / "scripts/carc_gpu.sbatch").read_text()
    assert cpu.count("exec srun ") == gpu.count("exec srun ") == 1
    assert "export TDN_WORKFLOW_PHASE=cpu" in cpu
    assert "export TDN_WORKFLOW_PHASE=gpu" in gpu
    assert "--gpus-per-task" not in cpu
    assert "srun --ntasks=1 --gpus-per-task=a100:1" in gpu
    assert "#SBATCH --constraint=a100-40gb" in gpu
