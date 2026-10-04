"""CPU light automation using explicit scheduler and stage mocks, never CUDA."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from test_carc_workflow import controller, fake_software, parsed, persist
from test_carc_workers import worker_project, run_worker, executed, marks
from test_carc_integration import carc_integration, controller as integration_controller, run_phase, stage_records

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(os.name == "nt", reason="CARC launchers require POSIX Bash")


@pytest.fixture
def light_controller(controller):
    (controller.ROOT / "configs/carc-light.yaml").write_text("purpose: development\n")
    return controller


def light_plan(module, *args):
    return module.prepare(parsed(module, "light", "--run-id", "light-unit", *args))


def test_light_preview_is_one_cpu_job_and_creates_no_files(light_controller, monkeypatch, capsys):
    module = light_controller
    monkeypatch.setattr(module, "command", lambda *a, **kw: pytest.fail("Dry preview executed a command"))
    assert module.main(["light", "--run-id", "light-preview"]) == 0
    output = capsys.readouterr().out
    assert output.count("sbatch --parsable") == 1
    assert "setup -> light-tests -> light-screen" in output
    assert "--cpus-per-task=2" in output and "--mem=8G" in output and "--time=00:15:00" in output
    assert "--account=anakano_81" in output and "--partition=main" in output
    assert "--gpus" not in output and "carc_gpu.sbatch" not in output and "--dependency" not in output
    assert not any((module.ROOT / name).exists() for name in ("runs", ".cache", ".venv"))


@pytest.mark.parametrize("options,reason", [
    (["--profile", "pilot"], "fixed CPU-only profile"),
    (["--profile", "smoke"], "fixed CPU-only profile"),
    (["--pilot-budget", "13"], "fixed scope"),
    (["--config", "configs/carc-smoke.yaml"], "only configs/carc-light.yaml"),
    (["--setup", "always"], "setup=never"),
    (["--setup", "auto"], "setup=never"),
])
def test_light_refuses_expanded_or_installing_scope(light_controller, options, reason):
    with pytest.raises(ValueError, match=reason):
        light_plan(light_controller, *options)
    assert not (light_controller.ROOT / "runs").exists()


def test_start_light_alias_and_setup_rejection(light_controller):
    module = light_controller
    workflow = module.prepare(parsed(module, "start", "--profile", "light", "--run-id", "light-alias"))
    assert workflow["action"] == "light" and workflow["setup"] == "never"
    assert module.phase_stages(workflow, "cpu") == ("setup", "light-tests", "light-screen")
    assert module.phase_stages(workflow, "gpu") == ()
    with pytest.raises(ValueError, match="existing venv"):
        module.prepare(parsed(module, "setup", "--profile", "light"))


def test_light_submission_has_only_one_live_cpu_check_and_sbatch(light_controller, monkeypatch):
    module = light_controller
    calls = []
    def check(workflow, phase):
        calls.append(("check", phase))
        return {"phase": phase}
    def submit(args, **kwargs):
        assert args[0] == "sbatch"
        calls.append(("submit", kwargs["env"]["TDN_WORKFLOW_PHASE"]))
        assert kwargs["env"]["TDN_SETUP_MODE"] == "never"
        assert kwargs["env"]["TDN_WORKFLOW_ACTION"] == "light"
        assert "--gpus-per-task=a100:1" not in args and "--dependency" not in " ".join(args)
        return SimpleNamespace(stdout="8201\n", returncode=0)
    monkeypatch.setattr(module, "live_check", check)
    monkeypatch.setattr(module, "command", submit)
    assert module.main(["light", "--run-id", "light-submit", "--submit"]) == 0
    assert calls == [("check", "cpu"), ("submit", "cpu")]
    workflow = module.load_workflow(module.workflow_path("light-submit"))
    assert module.workflow_phases(workflow) == ("cpu",)
    assert workflow["resources"] == {"cpu": {"partition": "main", "cpus": 2, "mem_gib": 8, "walltime": "00:15:00"}}
    assert module.jobs_for(workflow)["gpu"] == []


def test_light_cannot_reach_gpu_worker_or_training_resume(light_controller, monkeypatch):
    module = light_controller
    workflow = light_plan(module)
    monkeypatch.setattr(module, "actual_policy", lambda **kw: None)
    monkeypatch.setattr(module, "ownership", lambda *a: pytest.fail("Light checked GPU ownership"))
    with pytest.raises(ValueError, match="no GPU phase"):
        module.worker_verify(workflow, "gpu")
    with pytest.raises(ValueError, match="no GPU phase"):
        module.scheduler_args(workflow, "gpu")
    with pytest.raises(ValueError, match="no training checkpoint"):
        module.resume(parsed(module, "resume", "light-unit"), workflow)


def test_light_restart_preserves_only_cpu_scope(light_controller, monkeypatch, capsys):
    module = light_controller
    workflow = light_plan(module)
    persist(module, workflow)
    before = {p: p.read_bytes() for p in Path(workflow["run_dir"]).rglob("*") if p.is_file()}
    monkeypatch.setattr(module, "command", lambda *a, **kw: pytest.fail("Dry restart contacted scheduler"))
    assert module.main(["restart", "light-unit", "--run-id", "light-restart"]) == 0
    output = capsys.readouterr().out
    assert output.count("sbatch --parsable") == 1 and "profile=light" in output
    assert "carc_gpu.sbatch" not in output and "setup -> light-tests -> light-screen" in output
    assert before == {p: p.read_bytes() for p in Path(workflow["run_dir"]).rglob("*") if p.is_file()}
    assert not (module.ROOT / "runs/light-restart").exists()
    assert module.main(["restart", "light-unit", "--profile", "pilot", "--pilot-budget", "1000"]) == 2


def test_light_status_does_not_invent_gpu_stages(light_controller, monkeypatch, capsys):
    module = light_controller
    workflow = light_plan(module)
    persist(module, workflow)
    monkeypatch.setattr(module, "scheduler_state", lambda *a: pytest.fail("No submitted jobs to inspect"))
    module.status(workflow)
    output = capsys.readouterr().out
    assert "light-tests: NOT_STARTED" in output and "light-screen: NOT_STARTED" in output
    assert "gpu:" not in output and "train:" not in output and "generate:" not in output


def test_light_manifest_rejects_added_gpu_resource(light_controller):
    module = light_controller
    workflow = light_plan(module)
    path = persist(module, workflow)
    workflow["resources"]["gpu"] = {"cpus": 4, "mem_gib": 16, "walltime": "00:45:00"}
    module.atomic_json(path, workflow)
    with pytest.raises(ValueError, match="only 2 CPUs"):
        module.load_workflow(path)


def test_light_test_proof_requires_zero_exit_and_current_environment(light_controller):
    module = light_controller
    workflow = light_plan(module)
    persist(module, workflow)
    fake_software(module, workflow)
    assert not module.completed(workflow, "cpu", "light-tests")
    module.mark(workflow, "cpu", "light-tests", "COMPLETED", 0)
    assert module.completed(workflow, "cpu", "light-tests")
    module.atomic_json(module.software_record(workflow), {"config_hash": "changed"})
    assert not module.completed(workflow, "cpu", "light-tests")


def test_light_screen_completion_requires_actual_cpu_report_and_marker(light_controller):
    module = light_controller
    workflow = light_plan(module)
    persist(module, workflow)
    fake_software(module, workflow)
    run = Path(workflow["run_dir"]) / "light-screen"
    run.mkdir()
    payload = {"stage": "light-screen", "status": "COMPLETED", "actually_ran": True,
               "config_hash": "normalized-config", "device": "cpu",
               "software": {"source_tree_sha256": workflow["source_tree_sha256"],
                            "python": "3.11.9", "torch": "2.10.0+cu126"}}
    module.atomic_json(run / "stage.json", payload)
    assert not module.completed(workflow, "cpu", "light-screen")
    (run / "COMPLETED").write_text("normalized-config\n")
    assert module.completed(workflow, "cpu", "light-screen")
    payload["device"] = "cuda"
    module.atomic_json(run / "stage.json", payload)
    assert not module.completed(workflow, "cpu", "light-screen")


@pytest.fixture
def light_worker(worker_project, monkeypatch):
    root, base = worker_project
    (base / "config.yaml").write_text((ROOT / "configs/carc-light.yaml").read_text())
    monkeypatch.setenv("TDN_WORKFLOW_ACTION", "light")
    monkeypatch.setenv("TDN_SETUP_MODE", "never")
    return root, base


def test_light_worker_reuses_venv_then_only_runs_focused_stages(light_worker):
    _root, base = light_worker
    result = run_worker(light_worker)
    assert result.returncode == 0, result.stderr
    assert executed(base) == [["light-tests", "cpu", "none"], ["light-screen", "cpu", "none"]]
    assert ("setup", "COMPLETED", 0) in marks(base)
    assert "CPU-only light" in result.stdout


@pytest.mark.parametrize("mode", ["auto", "always"])
def test_light_worker_never_installs_even_if_injected(light_worker, mode):
    _root, base = light_worker
    result = run_worker(light_worker, TDN_SETUP_MODE=mode)
    assert result.returncode == 2 and "setup=never" in result.stderr
    assert executed(base) == []


def test_light_worker_missing_venv_stops_before_tests(light_worker):
    _root, base = light_worker
    (base / "venv.ready").unlink()
    result = run_worker(light_worker)
    assert result.returncode == 2 and "exact project venv is not ready" in result.stderr
    assert executed(base) == []


def test_light_worker_refuses_gpu_before_any_stage(light_worker):
    _root, base = light_worker
    result = run_worker(light_worker, TDN_WORKFLOW_PHASE="gpu")
    assert result.returncode == 2 and "no GPU phase" in result.stderr
    assert executed(base) == []


def test_light_worker_failure_does_not_start_science_screen(light_worker):
    _root, base = light_worker
    result = run_worker(light_worker, MOCK_FAIL_STAGE="light-tests")
    assert result.returncode == 17
    assert [item[0] for item in executed(base)] == ["light-tests"]
    assert ("light-tests", "FAILED", 17) in marks(base)


def test_light_worker_rejects_weakened_headroom_before_any_tests(light_worker):
    _root, base = light_worker
    config = yaml.safe_load((base / "config.yaml").read_text())
    config["validation"]["require_headroom"] = False
    (base / "config.yaml").write_text(yaml.safe_dump(config))
    result = run_worker(light_worker)
    assert result.returncode != 0
    assert executed(base) == []


@pytest.fixture
def light_integration(carc_integration):
    root, _environment = carc_integration
    # Copy execution code solely to validate the prescribed light config.
    # The existing fixture substitutes all scientific stages and scheduler facts.
    for name in ("tdn", "reference"):
        shutil.copytree(ROOT / name, root / name, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(ROOT / "configs/carc-light.yaml", root / "configs/carc-light.yaml")
    return carc_integration


@pytest.mark.parametrize("fail_tests", [False, True])
def test_actual_controller_batch_worker_coupling_is_only_one_cpu_phase(light_integration, fail_tests):
    root, _environment = light_integration
    submission = integration_controller(light_integration, "light", "--run-id", "integrated-unit", "--submit")
    assert submission.returncode == 0, submission.stdout + submission.stderr
    submitted = [json.loads(line) for line in (root / "mock-submissions.jsonl").read_text().splitlines()]
    assert len(submitted) == 1
    assert submitted[0]["env"]["TDN_WORKFLOW_PHASE"] == "cpu"
    assert "--time=00:15:00" in submitted[0]["args"]
    assert not any("gpu" in item or "dependency" in item for item in submitted[0]["args"])
    updates = {"MOCK_FAIL_STAGE": "light-tests"} if fail_tests else {}
    result = run_phase(light_integration, submitted, "cpu", **updates)
    assert result.returncode == (17 if fail_tests else 0), result.stdout + result.stderr
    base = root / "runs/integrated-unit"
    records = stage_records(base, "cpu")
    assert records["setup"]["status"] == "COMPLETED"
    entries = [json.loads(line) for line in (base / "mock-executed.jsonl").read_text().splitlines()]
    assert [entry["stage"] for entry in entries] == (["light-tests"] if fail_tests else ["light-tests", "light-screen"])
    assert all(entry["device"] == "cpu" for entry in entries)
    assert records["light-tests"]["status"] == ("FAILED" if fail_tests else "COMPLETED")
    assert ("light-screen" in records) is (not fail_tests)
    assert not (base / "state/gpu").exists()
    assert len((root / "mock-srun.txt").read_text().splitlines()) == 1
    status = integration_controller(light_integration, "status", "integrated-unit")
    assert status.returncode == 0 and "gpu" not in status.stdout


@pytest.fixture
def stage_project(worker_project, monkeypatch):
    root, base = worker_project
    (root / "scripts/run_stage.sh").write_text((ROOT / "scripts/run_stage.sh").read_text())
    # Dispatch runs inside the grouped worker's existing report. Exercise that
    # real attachment contract; manual reporting has separate launcher tests.
    from tdn.reporting import begin_report
    report = begin_report(base, name="TDN/mocked-light-dispatch", script="scripts/run_stage.sh",
                          job_id=os.environ["SLURM_JOB_ID"], report_parent=base / "tower",
                          metadata={"validation_scope": "mocked scheduler stage dispatch test"})
    monkeypatch.setenv("TDN_TOWER_DIR", str(report))
    python = root / "commands/stage-python"
    python.write_text("#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$TDN_WORKFLOW_ROOT/python-args.txt\"\n"
                      "printf '%s' \"${TDN_TOWER_DIR:-}\" > \"$TDN_WORKFLOW_ROOT/python-tower-env.txt\"\n")
    python.chmod(0o755)
    common = root / "scripts/common.sh"
    with common.open("a") as stream:
        stream.write("\ntdn_python() { printf '%s\\n' " + shlex.quote(str(python)) + "; }\n")
    return root, base


@pytest.mark.parametrize("stage", ["light-tests", "light-screen"])
def test_light_stage_rejects_cuda_without_running_preflight(stage_project, stage):
    root, base = stage_project
    env = dict(os.environ, TDN_STAGE=stage, TDN_DEVICE="cuda", TDN_RUN_DIR=str(base / stage))
    result = subprocess.run(["bash", str(root / "scripts/run_stage.sh")], env=env, text=True, capture_output=True)
    assert result.returncode == 2 and "require a CPU task" in result.stderr
    assert not (base / "python-args.txt").exists()


def test_light_tests_dispatch_only_scientific_regressions(stage_project):
    root, base = stage_project
    env = dict(os.environ, TDN_STAGE="light-tests", TDN_DEVICE="cpu", TDN_RUN_DIR=str(base / "light-tests"))
    result = subprocess.run(["bash", str(root / "scripts/run_stage.sh")], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    args = (base / "python-args.txt").read_text().splitlines()
    assert args[:3] == ["-m", "pytest", "-q"]
    assert args[3:7] == ["tests/test_light_screen.py", "tests/test_temporal_oracle_screen.py",
                         "tests/test_light_screen_cli.py", "tests/test_analysis.py"]
    assert args[7:] == ["--basetemp", str(base / "light-tests/pytest-work"),
                       "--junitxml", str(base / "light-tests/pytest-results.xml")]
    assert (base / "python-tower-env.txt").read_text() == ""


def test_light_screen_dispatch_uses_cpu_wrapper_without_dataset_or_training(stage_project):
    root, base = stage_project
    env = dict(os.environ, TDN_STAGE="light-screen", TDN_DEVICE="cpu", TDN_RUN_DIR=str(base / "light-screen"))
    result = subprocess.run(["bash", str(root / "scripts/run_stage.sh")], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert (base / "python-args.txt").read_text().splitlines() == [
        "-u", str(root / "scripts/light_screen.py"), "--config", str(base / "config.yaml"),
        "--run-dir", str(base / "light-screen"),
    ]
    assert (base / "python-tower-env.txt").read_text() == os.environ["TDN_TOWER_DIR"]
