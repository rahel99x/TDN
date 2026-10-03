"""Isolated controller tests: no allocations, installation or GPU claims."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def controller(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("carc_controller", ROOT / "scripts" / "carc_workflow.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "project with spaces"
    for name in ("tdn", "reference", "scripts", "configs", "tests"):
        (root / name).mkdir(parents=True)
        (root / name / "source.txt").write_text(f"{name} source\n")
    for name in ("carc-smoke.yaml", "pilot.yaml"):
        (root / "configs" / name).write_text("purpose: development\n")
    (root / "requirements.txt").write_text("numpy==2.2.6\n")
    (root / "pyproject.toml").write_text("[project]\nname='tdn'\n")
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "CARC_ROOT", root)
    monkeypatch.setattr(module.getpass, "getuser", lambda: "aadaniel")
    for name in ("TDN_LOCAL_TEST_ROOT", "TDN_EXECUTION_MODE", "TDN_PROJECT_ROOT", "SLURM_JOB_ID", "SLURM_STEP_ID",
                 "SLURM_JOB_ACCOUNT", "CONDA_PREFIX", "CONDA_SHLVL", "CARC_ACCOUNT", "TORCH_VERSION", "TORCH_WHEEL_INDEX",
                 "TDN_CPU_PARTITION"):
        monkeypatch.delenv(name, raising=False)
    return module


def parsed(module, *args):
    return module.parser().parse_args(args)


def plan(module, identifier="carc-test", **options):
    args = parsed(module, "start", "--run-id", identifier)
    for key, value in options.items():
        setattr(args, key, value)
    return module.prepare(args)


def persist(module, workflow):
    base = Path(workflow["run_dir"])
    base.mkdir(parents=True)
    Path(workflow["config_path"]).write_bytes(Path(workflow["original_config"]).read_bytes())
    module.atomic_json(base / "workflow.json", workflow)
    module.atomic_json(base / "jobs.json", {"cpu": [], "gpu": []})
    return base / "workflow.json"


def fake_software(module, workflow):
    payload = {"config_hash": "normalized-config", "python": "3.11.9", "packages": {"torch": "2.10.0+cu126"}}
    module.atomic_json(module.software_record(workflow), payload)
    return payload


def cli_proof(module, workflow, stage):
    base = Path(workflow["run_dir"]) / stage
    base.mkdir(parents=True, exist_ok=True)
    (base / "COMPLETED").write_text("normalized-config\n")
    module.atomic_json(base / "stage.json", {"stage": stage, "status": "COMPLETED", "actually_ran": True,
                       "config_hash": "normalized-config", "device": "cpu" if stage in ("audit", "generate") else "cuda",
                       "software": {"source_tree_sha256": workflow["source_tree_sha256"], "python": "3.11.9", "torch": "2.10.0+cu126"}})


def completed_stages(module, monkeypatch, workflow, phase, stages):
    monkeypatch.setattr(module, "venv_ready", lambda *args: {})
    for stage in stages:
        if stage not in ("setup", "cpu-tests", "gpu-tests"):
            cli_proof(module, workflow, stage)
        module.mark(workflow, phase, stage, "COMPLETED", 0)


def test_default_preview_is_two_jobs_without_side_effects(controller, monkeypatch, capsys):
    monkeypatch.setattr(controller, "command", lambda *args, **kwargs: pytest.fail("Dry run used external command"))
    assert controller.main(["start", "--run-id", "carc-preview"]) == 0
    output = capsys.readouterr().out
    assert output.count("sbatch --parsable") == 2
    assert "cpu-tests -> audit -> generate" in output
    assert "gpu-tests -> calibrate -> train -> evaluate -> benchmark" in output
    assert "--account=anakano_81" in output and "--partition=main" in output
    assert "--gpus-per-task=a100:1" in output and "--constraint=a100-40gb" in output
    assert not (controller.ROOT / "runs").exists()
    assert not (controller.ROOT / ".venv").exists()
    assert not (controller.ROOT / ".cache").exists()


@pytest.mark.skipif(os.name == "nt", reason="CARC controller locks require POSIX fcntl")
def test_controller_lock_rejects_concurrent_controller_then_releases(controller):
    first = controller.controller_lock()
    try:
        with pytest.raises(ValueError, match="Another CARC controller"):
            controller.controller_lock()
    finally:
        first.close()
    following = controller.controller_lock()
    following.close()


def test_submit_policy_is_checked_before_creating_controller_lock(controller, monkeypatch):
    monkeypatch.setattr(controller, "actual_policy", lambda **kwargs: (_ for _ in ()).throw(ValueError("wrong CARC user")))
    assert controller.main(["start", "--run-id", "blocked", "--submit"]) == 2
    assert not (controller.ROOT / ".cache").exists()


def test_setup_only_preview_one_cpu_request(controller, capsys):
    assert controller.main(["setup", "--run-id", "carc-setup"]) == 0
    output = capsys.readouterr().out
    assert output.count("sbatch --parsable") == 1
    assert "--gpus-per-task" not in output
    assert "cpu: setup" in output


@pytest.mark.parametrize("version,index", [
    ("2.10.0", "https://download.pytorch.org/whl/cu126"),
    ("2.10.0+cpu", "https://download.pytorch.org/whl/cu126"),
    ("2.10.0+cu128", "https://download.pytorch.org/whl/cu126"),
    ("2.10.0+cu126", "http://download.pytorch.org/whl/cu126"),
])
def test_exact_cuda_build_and_official_index_required(controller, monkeypatch, version, index):
    monkeypatch.setenv("TORCH_VERSION", version)
    monkeypatch.setenv("TORCH_WHEEL_INDEX", index)
    with pytest.raises(ValueError, match="exact CUDA"):
        plan(controller)


def test_pilot_requires_explicit_budget_and_uses_larger_resources(controller):
    with pytest.raises(ValueError, match="explicit"):
        plan(controller, profile="pilot")
    workflow = plan(controller, profile="pilot", pilot_budget=1000)
    assert workflow["resources"]["cpu"] == {"partition": "main", "cpus": 8, "mem_gib": 32, "walltime": "02:00:00"}
    assert workflow["resources"]["gpu"]["mem_gib"] == 64
    with pytest.raises(ValueError, match="1-1000"):
        plan(controller, profile="pilot", pilot_budget=1001)


def test_submission_live_checks_before_jobs_and_chains_actual_id(controller, monkeypatch):
    calls = []
    def fake_check(workflow, phase):
        calls.append(("check", phase))
        return {"account": "anakano_81", "phase": phase}
    def fake_command(args, **kwargs):
        assert args[0] == "sbatch"
        phase = kwargs["env"]["TDN_WORKFLOW_PHASE"]
        calls.append(("submit", phase, args, kwargs["env"]))
        return SimpleNamespace(stdout="7001;discovery\n" if phase == "cpu" else "7002\n", returncode=0)
    monkeypatch.setattr(controller, "live_check", fake_check)
    monkeypatch.setattr(controller, "command", fake_command)
    assert controller.main(["start", "--run-id", "carc-live", "--submit"]) == 0
    assert [entry[:2] for entry in calls] == [("check", "cpu"), ("check", "gpu"), ("submit", "cpu"), ("submit", "gpu")]
    gpu = calls[-1]
    assert "--dependency=afterok:7001" in gpu[2]
    assert "--kill-on-invalid-dep=yes" in gpu[2]
    assert "--comment=tdn:carc-live:gpu" in gpu[2]
    assert gpu[3]["TDN_WORKFLOW_ACTION"] == "start"
    assert gpu[3]["TDN_CONFIG"].endswith("/carc-live/config.yaml")
    base = controller.ROOT / "runs" / "carc-live"
    assert controller.read_json(base / "jobs.json")["cpu"][0]["job_id"] == "7001"
    assert controller.read_json(base / "jobs.json")["gpu"][0]["job_id"] == "7002"
    assert controller.read_json(controller.ROOT / "runs" / ".carc-latest.json")["run_id"] == "carc-live"


def test_second_submission_failure_keeps_cpu_id_without_cancel(controller, monkeypatch):
    monkeypatch.setattr(controller, "live_check", lambda *args: {})
    def fake_command(args, **kwargs):
        assert args[0] == "sbatch", "Controller cancelled work after partial submission"
        if kwargs["env"]["TDN_WORKFLOW_PHASE"] == "gpu":
            raise ValueError("scheduler submission failed")
        return SimpleNamespace(stdout="7011\n", returncode=0)
    monkeypatch.setattr(controller, "command", fake_command)
    assert controller.main(["start", "--run-id", "carc-partial", "--submit"]) == 2
    jobs = controller.read_json(controller.ROOT / "runs" / "carc-partial" / "jobs.json")
    assert jobs["cpu"][0]["job_id"] == "7011" and jobs["gpu"] == []


@pytest.mark.parametrize("result", ["Submitted batch job 20", "20\n21", "0", "12;discovery;bad"])
def test_ambiguous_scheduler_output_never_chained(controller, monkeypatch, result):
    workflow = plan(controller)
    persist(controller, workflow)
    monkeypatch.setattr(controller, "command", lambda *args, **kwargs: SimpleNamespace(stdout=result))
    with pytest.raises(ValueError, match="ambiguous"):
        controller.submit_phase(workflow, "cpu")
    assert controller.jobs_for(workflow) == {"cpu": [], "gpu": []}


def test_source_and_config_are_immutable_but_status_can_read_stale_run(controller):
    workflow = plan(controller)
    path = persist(controller, workflow)
    Path(workflow["config_path"]).write_text("changed config\n")
    with pytest.raises(ValueError, match="configuration changed"):
        controller.load_workflow(path)
    assert controller.load_workflow(path, verify=False)["run_id"] == "carc-test"
    Path(workflow["config_path"]).write_bytes(Path(workflow["original_config"]).read_bytes())
    (controller.ROOT / "requirements.txt").write_text("numpy==9.9.9\n")
    with pytest.raises(ValueError, match="source changed"):
        controller.load_workflow(path)


def test_generated_pycache_does_not_invalidate_source(controller):
    before = controller.source_hash()
    cache = controller.ROOT / "scripts" / "__pycache__"
    cache.mkdir()
    (cache / "ignored.pyc").write_bytes(b"generated")
    assert controller.source_hash() == before


def test_storage_symlink_escape_rejected_before_directory_creation(controller, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (controller.ROOT / "runs").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="leaves the project"):
        plan(controller)
    assert not list(outside.iterdir())


@pytest.mark.parametrize("field,value", [("UserId", "other(8)"), ("Account", "anakano_81_extra"),
                                         ("Comment", "tdn:other:gpu"), ("JobName", "tdn-gpu-tests")])
def test_cancel_requires_exact_scheduler_ownership(controller, monkeypatch, field, value):
    workflow = plan(controller)
    persist(controller, workflow)
    controller.atomic_json(Path(workflow["run_dir"]) / "jobs.json", {"cpu": [], "gpu": [{"job_id": "900"}]})
    fields = {"JobId": "900", "UserId": "aadaniel(8)", "Account": "anakano_81", "Comment": "tdn:carc-test:gpu",
              "JobName": "tdn-carc-gpu", "JobState": "PENDING"}
    fields[field] = value
    def fake_command(args, **kwargs):
        assert args[0] == "scontrol", "Unowned scheduler job was cancelled"
        return SimpleNamespace(returncode=0, stdout=" ".join(f"{k}={v}" for k, v in fields.items()))
    monkeypatch.setattr(controller, "command", fake_command)
    with pytest.raises(ValueError, match="ownership"):
        controller.cancel(workflow, submit=True)


def test_cancel_only_owned_pending_job_and_preserves_running(controller, monkeypatch):
    workflow = plan(controller)
    persist(controller, workflow)
    controller.atomic_json(Path(workflow["run_dir"]) / "jobs.json", {"cpu": [{"job_id": "901"}], "gpu": [{"job_id": "902"}]})
    calls = []
    def fake_command(args, **kwargs):
        calls.append(args)
        if args[0] == "scancel":
            return SimpleNamespace(returncode=0, stdout="")
        job, phase, state = ("901", "cpu", "RUNNING") if args[3] == "901" else ("902", "gpu", "PENDING")
        return SimpleNamespace(returncode=0, stdout=f"JobId={job} UserId=aadaniel(8) Account=anakano_81 Comment=tdn:carc-test:{phase} JobName=tdn-carc-{phase} JobState={state}")
    monkeypatch.setattr(controller, "command", fake_command)
    controller.cancel(workflow, submit=True)
    assert [call for call in calls if call[0] == "scancel"] == [["scancel", "--state=PENDING", "902"]]


def test_cancel_dry_run_does_not_contact_scheduler(controller, monkeypatch):
    workflow = plan(controller)
    persist(controller, workflow)
    controller.atomic_json(Path(workflow["run_dir"]) / "jobs.json", {"cpu": [{"job_id": "902"}], "gpu": []})
    monkeypatch.setattr(controller, "command", lambda *args, **kwargs: pytest.fail("Dry cancellation contacted scheduler"))
    controller.cancel(workflow)


def test_gpu_requires_all_cpu_prerequisites_and_software(controller, monkeypatch):
    workflow = plan(controller)
    persist(controller, workflow)
    fake_software(controller, workflow)
    monkeypatch.setattr(controller, "actual_policy", lambda **kwargs: None)
    monkeypatch.setattr(controller, "ownership", lambda *args: {"JobState": "RUNNING"})
    monkeypatch.setattr(controller, "venv_ready", lambda *args: {})
    with pytest.raises(ValueError, match="CPU prerequisites"):
        controller.worker_verify(workflow, "gpu")
    completed_stages(controller, monkeypatch, workflow, "cpu", controller.CPU_STAGES)
    controller.worker_verify(workflow, "gpu")


def test_worker_checks_live_running_phase_without_waiting_for_id_publication(controller, monkeypatch):
    workflow = plan(controller)
    persist(controller, workflow)
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_STEP_ID", "0")
    monkeypatch.setenv("SLURM_JOB_ACCOUNT", "anakano_81")
    observed = []
    def ownership(workflow, phase, job):
        observed.append((phase, job))
        return {"JobState": "RUNNING"}
    monkeypatch.setattr(controller, "ownership", ownership)
    controller.worker_verify(workflow, "cpu")
    assert observed == [("cpu", "12345")]
    assert controller.jobs_for(workflow) == {"cpu": [], "gpu": []}
    monkeypatch.setattr(controller, "ownership", lambda *args: {"JobState": "PENDING"})
    with pytest.raises(ValueError, match="running phase"):
        controller.worker_verify(workflow, "cpu")


@pytest.mark.parametrize("job,step", [("batch", "0"), ("1", "batch"), ("0", "0")])
def test_worker_rejects_non_task_allocation_identifiers(controller, monkeypatch, job, step):
    monkeypatch.setenv("SLURM_JOB_ID", job)
    monkeypatch.setenv("SLURM_STEP_ID", step)
    monkeypatch.setenv("SLURM_JOB_ACCOUNT", "anakano_81")
    with pytest.raises(ValueError, match="numeric"):
        controller.actual_policy(worker=True)


def test_cli_completion_is_recovered_only_with_current_proof(controller):
    workflow = plan(controller)
    persist(controller, workflow)
    fake_software(controller, workflow)
    cli_proof(controller, workflow, "audit")
    record = controller.stage_record(workflow, "cpu", "audit")
    assert controller.completed(workflow, "cpu", "audit", recover=False) is False
    assert not record.exists()
    assert controller.completed(workflow, "cpu", "audit") is True
    assert controller.read_json(record)["recovered_cli_proof"] is True
    (Path(workflow["run_dir"]) / "audit" / "COMPLETED").unlink()
    assert controller.completed(workflow, "cpu", "audit") is False


def test_resume_requires_certified_paused_checkpoint_and_preserves_config(controller, monkeypatch):
    workflow = plan(controller)
    persist(controller, workflow)
    fake_software(controller, workflow)
    completed_stages(controller, monkeypatch, workflow, "cpu", controller.CPU_STAGES)
    completed_stages(controller, monkeypatch, workflow, "gpu", ("gpu-tests", "calibrate"))
    args = parsed(controller, "resume", "carc-test")
    with pytest.raises(ValueError, match="certified paused"):
        controller.resume(args, workflow)
    checkpoint = Path(workflow["run_dir"]) / "train" / "checkpoints" / "last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"unit-test checkpoint certificate, no Torch payload")
    latest = checkpoint.parent / "latest.json"
    controller.atomic_json(latest, {"path": "last.pt", "status": "RUNNING", "sha256": controller.digest(checkpoint)})
    with pytest.raises(ValueError, match="PAUSED_NEEDS_RESUME"):
        controller.resume(args, workflow)
    controller.atomic_json(latest, {"path": "last.pt", "status": "PAUSED_NEEDS_RESUME", "sha256": controller.digest(checkpoint)})
    monkeypatch.setattr(controller, "command", lambda *args, **kwargs: pytest.fail("Dry resume contacted scheduler or interpreter"))
    before = {path: path.read_bytes() for path in Path(workflow["run_dir"]).rglob("*") if path.is_file()}
    controller.resume(args, workflow)
    assert before == {path: path.read_bytes() for path in Path(workflow["run_dir"]).rglob("*") if path.is_file()}
    args.config = "configs/pilot.yaml"
    with pytest.raises(ValueError, match="preserves the original"):
        controller.resume(args, workflow)


def test_resume_blocks_active_or_unknown_jobs(controller, monkeypatch):
    workflow = plan(controller)
    persist(controller, workflow)
    fake_software(controller, workflow)
    completed_stages(controller, monkeypatch, workflow, "cpu", controller.CPU_STAGES)
    completed_stages(controller, monkeypatch, workflow, "gpu", ("gpu-tests", "calibrate"))
    checkpoint = Path(workflow["run_dir"]) / "train" / "checkpoints" / "last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"certificate")
    controller.atomic_json(checkpoint.parent / "latest.json", {"path": "last.pt", "status": "PAUSED_NEEDS_RESUME", "sha256": controller.digest(checkpoint)})
    controller.atomic_json(Path(workflow["run_dir"]) / "jobs.json", {"cpu": [], "gpu": [{"job_id": "995"}]})
    monkeypatch.setattr(controller, "scheduler_state", lambda job: "RUNNING")
    with pytest.raises(ValueError, match="active"):
        controller.resume(parsed(controller, "resume", "carc-test", "--submit"), workflow)


def test_status_uses_terminal_scheduler_state_for_stale_local_running_record(controller, monkeypatch, capsys):
    workflow = plan(controller)
    persist(controller, workflow)
    controller.atomic_json(Path(workflow["run_dir"]) / "jobs.json", {"cpu": [{"job_id": "999"}], "gpu": []})
    controller.mark(workflow, "cpu", "cpu-tests", "RUNNING", 0)
    monkeypatch.setattr(controller, "scheduler_state", lambda job: "TIMEOUT")
    controller.status(workflow)
    output = capsys.readouterr().out
    assert "cpu job 999: TIMEOUT" in output and "cpu-tests: INTERRUPTED" in output
    assert "saved record RUNNING" in output


def test_real_venv_symlink_interpreter_is_allowed_but_prefix_checked(controller, monkeypatch):
    (controller.ROOT / ".venv" / "bin").mkdir(parents=True)
    (controller.ROOT / ".venv" / "pyvenv.cfg").write_text("home = standalone-python\n")
    (controller.ROOT / ".venv" / "bin" / "python").symlink_to(Path(__import__("sys").executable))
    workflow = plan(controller)
    calls = []
    def fake_command(args, **kwargs):
        calls.append(args)
        assert "sys.prefix" in args[2] and "conda-meta" in args[2]
        assert "torch" in args[2] and "import torch" not in args[2]
        return SimpleNamespace(stdout=json.dumps({"python": "3.11.9"}))
    monkeypatch.setattr(controller, "command", fake_command)
    assert controller.venv_report(workflow) == {"python": "3.11.9"}
    assert calls[0][0] == controller.ROOT / ".venv" / "bin" / "python"


def test_changed_software_cannot_reuse_baseline(controller, monkeypatch):
    workflow = plan(controller)
    persist(controller, workflow)
    fake_software(controller, workflow)
    monkeypatch.setattr(controller, "venv_report", lambda workflow: {"packages": {"torch": "2.10.0+cpu"}})
    with pytest.raises(ValueError, match="software environment changed"):
        controller.venv_ready(workflow, "gpu")
