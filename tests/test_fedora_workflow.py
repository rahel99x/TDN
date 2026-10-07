"""Fedora scheduler contracts and actual Tower I/O; no live Slurm/GPU claim."""
from __future__ import annotations

from copy import deepcopy
import getpass
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
    spec = importlib.util.spec_from_file_location("fedora_workflow_test", ROOT / "scripts" / "fedora_workflow.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "project with spaces"
    for name in ("tdn", "reference", "scripts", "configs", "tests"):
        (root / name).mkdir(parents=True)
        (root / name / "source.txt").write_text(name + "\n")
    for name in ("test_premix_gpu.py", "test_premix_numerics.py", "test_fedora_workflow.py"):
        (root / "tests" / name).write_text("def test_ok(): pass\n")
    (root / "scripts" / "fedora_slurm_worker.sh").write_bytes((ROOT / "scripts" / "fedora_slurm_worker.sh").read_bytes())
    (root / "requirements.txt").write_text("numpy==2.2.6\n")
    (root / "pyproject.toml").write_text("[project]\nname='tdn'\n")
    for owner in (module, module.pw, module.rw, module.cw):
        monkeypatch.setattr(owner, "ROOT", root)
    for key in ("TDN_LOCAL_TEST_ROOT", "TDN_EXECUTION_MODE", "TDN_PROJECT_ROOT", "SLURM_JOB_ID", "SLURM_STEP_ID",
                "SLURM_JOB_ACCOUNT", "CONDA_PREFIX", "CONDA_SHLVL", "TDN_SLURM_CONFIG", "TDN_TOWER_DIR"):
        monkeypatch.delenv(key, raising=False)
    profile = {"schema_version": 1, "kind": "desktop-slurm", "root": str(root), "user": getpass.getuser(),
               "cpu_partition": "workstation", "gpu_partition": "workstation", "account": None,
               "gpu_gres": "gpu:rtx4090:1", "expected_gpu_name": "RTX 4090", "gpu_vram_gib": 24,
               "torch_version": "2.10.0+cu126", "torch_wheel_index": "https://download.pytorch.org/whl/cu126"}
    module.cw.atomic_json(root / ".tdn" / "fedora-slurm.json", profile)
    return module


def plan(module, name="test", *flags):
    return module.prepare(module.parser().parse_args(["plan", "--run-id", name, *flags]))


def freeze(module, workflow):
    base = Path(workflow["run_dir"])
    base.mkdir(parents=True)
    (base / "logs").mkdir()
    for path, value in ((base / "premix-workflow.json", workflow),
                        (base / "protocol.json", module.declaration(workflow["profile"])),
                        (base / "slurm-profile.json", workflow["slurm_profile"]), (base / "jobs.json", [])):
        module.cw.atomic_json(path, value)
    return base


def seal(module, workflow, stage, *, execution_mode="desktop-slurm", profile_hash=None):
    base = Path(workflow["run_dir"]) / stage
    base.mkdir(parents=True)
    files = {"summary.json": {"status": "COMPLETED"}, "protocol.json": {"stage": stage},
             "execution.json": {"stage": stage, "profile": workflow["profile"], "execution_mode": execution_mode,
                "device": "cuda" if stage == "neural" else "cpu",
                "slurm_profile_sha256": profile_hash or workflow["slurm_profile_sha256"]}}
    if stage == "prepare":
        files["dataset_manifest.json"] = {"fixture": True}
        (base / "dataset.pt").write_bytes(b"fixture, not scientific evidence")
    for name, value in files.items():
        module.cw.atomic_json(base / name, value)
    manifest = {"version": 1, "benchmark_suite": "premix", "stage": stage, "profile": workflow["profile"],
                "source_tree_sha256": workflow["source_tree_sha256"],
                "protocol_sha256": module.pw.canonical_hash(files["protocol.json"]),
                "files": {path.name: module.cw.digest(path) for path in base.iterdir()}}
    module.cw.atomic_json(base / "manifest.json", manifest)
    (base / "COMPLETED").write_text(module.cw.digest(base / "manifest.json") + "\n")
    return base


@pytest.mark.parametrize("flags", [[], ["--smoke"]])
def test_plan_is_read_only_and_desktop_bounded(controller, monkeypatch, capsys, flags):
    monkeypatch.setattr(controller.cw, "command", lambda *a, **kw: pytest.fail("plan called scheduler"))
    assert controller.main(["plan", *flags]) == 0
    output = capsys.readouterr().out
    assert output.count("sbatch --parsable") == 4
    assert output.count("--time=00:30:00") == 4
    assert output.count("--gres=none") == 3 and output.count("--gres=gpu:rtx4090:1") == 1
    assert "anakano_81" not in output and "a100" not in output and "--account=" not in output
    assert "CPU stages run sequentially" in output
    assert not (controller.ROOT / "runs").exists() and not (controller.ROOT / ".cache").exists()


def test_explicit_account_only(controller):
    workflow = plan(controller)
    workflow["slurm_profile"]["account"] = "local-research"
    assert "--account=local-research" in controller.scheduler_args(workflow, "accuracy")


def mock_submission(module, monkeypatch, *, fail_at=None):
    calls = []
    monkeypatch.setattr(module.rw, "software_report", lambda *a: {"packages": {"torch": "fixture"}})
    def command(args, **kwargs):
        calls.append((args, kwargs))
        if args[0] == "scontrol":
            return SimpleNamespace(stdout="PartitionName=workstation State=UP MaxTime=1-00:00:00")
        count = len([row for row in calls if row[0][0] == "sbatch"])
        if count == fail_at:
            raise ValueError("simulated submission failure")
        return SimpleNamespace(stdout=str(9000 + count))
    monkeypatch.setattr(module.cw, "command", command)
    return calls


def test_run_freezes_profile_and_submits_serial_cpu_dag(controller, monkeypatch):
    calls = mock_submission(controller, monkeypatch)
    assert controller.main(["run", "--run-id", "serial"]) == 0
    workflow = controller.load(controller.workflow_path("latest"))
    batches = [(args, kwargs) for args, kwargs in calls if args[0] == "sbatch"]
    assert len(batches) == 4
    expected = [None, "afterok:9001", "afterok:9002", "afterok:9001:9002:9003"]
    for (args, kwargs), dependency in zip(batches, expected):
        assert [item for item in args if item.startswith("--dependency=")] == (["--dependency=" + dependency] if dependency else [])
        assert kwargs["env"]["TDN_EXECUTION_MODE"] == "desktop-slurm"
        assert kwargs["env"]["TDN_SLURM_CONFIG"] == workflow["slurm_profile_path"]
    assert Path(workflow["slurm_profile_path"]).stat().st_mode & 0o222 == 0
    assert (controller.ROOT / "runs" / ".fedora-premix-latest.json").is_file()
    assert not (controller.ROOT / "runs" / ".premix-latest.json").exists()


def test_partial_submission_is_preserved_without_neural_or_retry(controller, monkeypatch):
    calls = mock_submission(controller, monkeypatch, fail_at=2)
    assert controller.main(["run", "--run-id", "partial"]) == 2
    workflow = controller.load(controller.workflow_path("latest"))
    state = controller.cw.read_json(Path(workflow["run_dir"]) / "state" / "submission.json")
    assert state["status"] == "FAILED" and len(state["submitted_jobs"]) == 1
    assert len([row for row in calls if row[0][0] == "sbatch"]) == 2


def test_submission_drops_ambient_scheduler_defaults(controller, monkeypatch):
    stale = {"SBATCH_ACCOUNT": "anakano_81", "SBATCH_CONSTRAINT": "a100-40gb", "SBATCH_QOS": "external",
             "SBATCH_GRES": "gpu:a100:4", "SBATCH_ARRAY_INX": "1-100", "SRUN_GRES": "gpu:a100:4",
             "SRUN_NTASKS": "30", "CARC_ACCOUNT": "anakano_81", "TORCH_CUDA_ARCH_LIST": "8.0"}
    for key, value in stale.items():
        monkeypatch.setenv(key, value)
    calls = mock_submission(controller, monkeypatch)
    assert controller.main(["run", "--run-id", "clean-env"]) == 0
    for args, kwargs in calls:
        if args[0] == "sbatch":
            assert stale.keys().isdisjoint(kwargs["env"])
            assert not any(item.startswith("--account=") for item in args)


@pytest.mark.parametrize("stage,dependency", [("accuracy", "afterok:1"), ("scaling", "afterok:1:2"),
                                            ("neural", "afterok:1:1:2"), ("neural", "afterany:1:2:3")])
def test_dependency_contract(controller, stage, dependency):
    with pytest.raises(ValueError, match="dependency"):
        controller.scheduler_args(plan(controller), stage, dependency)


@pytest.mark.parametrize("env", [{"SLURM_JOB_ID": "123"}, {"TDN_EXECUTION_MODE": "desktop"},
                                {"TDN_LOCAL_TEST_ROOT": "spoof"}, {"CONDA_PREFIX": "/conda"}])
def test_submission_rejects_ambient_bypass(controller, monkeypatch, env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(controller.cw, "command", lambda *a, **kw: pytest.fail("unsafe scheduler request"))
    assert controller.main(["run"]) == 2


def test_changed_frozen_profile_blocks_execution(controller):
    workflow = plan(controller)
    base = freeze(controller, workflow)
    controller.cw.atomic_json(base / "slurm-profile.json", {**workflow["slurm_profile"], "account": "changed"})
    with pytest.raises(ValueError, match="profile changed"):
        controller.load(base / "premix-workflow.json")


def test_changed_source_blocks_execution(controller):
    workflow = plan(controller)
    base = freeze(controller, workflow)
    (controller.ROOT / "tdn" / "source.txt").write_text("changed\n")
    with pytest.raises(ValueError, match="source changed"):
        controller.load(base / "premix-workflow.json")


@pytest.mark.parametrize("stage", ["accuracy", "scaling", "prepare", "neural"])
def test_stage_seal_accepts_only_desktop_profile(controller, stage):
    workflow = plan(controller)
    freeze(controller, workflow)
    seal(controller, workflow, stage)
    assert controller.verify_stage(workflow, stage)["stage"] == stage


@pytest.mark.parametrize("changes", [{"execution_mode": "carc"}, {"profile_hash": "0" * 64}])
def test_stage_seal_rejects_other_environment(controller, changes):
    workflow = plan(controller)
    freeze(controller, workflow)
    seal(controller, workflow, "neural", **changes)
    with pytest.raises(ValueError, match="execution_mode|profile differs"):
        controller.verify_stage(workflow, "neural")


def test_worker_commands_use_desktop_preflight_and_real_gpu_tests(controller):
    workflow = plan(controller)
    commands = controller.worker_commands(workflow, "neural")
    assert [name for name, _ in commands] == ["preflight", "gpu-tests", "experiment"]
    assert commands[0][1][1].endswith("fedora_gpu_preflight.py")
    assert "gpu" in commands[1][1] and str(controller.ROOT / "tests" / "test_premix_gpu.py") in commands[1][1]
    cpu = controller.worker_commands(workflow, "prepare")
    assert str(controller.ROOT / "tests" / "test_fedora_workflow.py") in cpu[0][1]
    assert "--device" in commands[-1][1] and commands[-1][1][-3] == "cuda"


def test_unit_tests_do_not_inherit_production_mode_but_gpu_tests_do(controller):
    env = {"TDN_EXECUTION_MODE": "desktop-slurm", "TDN_SLURM_CONFIG": "frozen", "SLURM_JOB_ID": "9001"}
    gpu = dict(env)
    controller.prepare_test_environment(env, "tests")
    controller.prepare_test_environment(gpu, "gpu-tests")
    assert env == {"SLURM_JOB_ID": "9001"}
    assert gpu["TDN_EXECUTION_MODE"] == "desktop-slurm"


def test_allocation_checks_job_profile_comment_and_workdir(controller, monkeypatch):
    workflow = plan(controller)
    base = freeze(controller, workflow)
    monkeypatch.setenv("SLURM_JOB_ID", "9001")
    monkeypatch.setenv("TDN_SLURM_CONFIG", str(base / "slurm-profile.json"))
    verified = []
    monkeypatch.setattr(controller, "runtime_allocation", lambda *args, **kw: verified.append((args, kw)))
    good = (f"JobId=9001 JobName=tdn-fedora-accuracy UserId={getpass.getuser()}(1000) "
            f"Comment={controller.comment(workflow, 'accuracy')} JobState=RUNNING Partition=workstation WorkDir={controller.ROOT}")
    monkeypatch.setattr(controller.cw, "command", lambda *a, **kw: SimpleNamespace(stdout=good))
    controller.verify_allocation(workflow, "accuracy")
    assert verified[0][0] == ("cpu",)
    for wrong in (good.replace("tdn-fedora-accuracy", "other"), good.replace(str(controller.ROOT), "/elsewhere"),
                  good.replace(workflow["slurm_profile_sha256"], "0" * 64)):
        monkeypatch.setattr(controller.cw, "command", lambda *a, **kw: SimpleNamespace(stdout=wrong))
        with pytest.raises(ValueError, match="running stage"):
            controller.verify_allocation(workflow, "accuracy")


@pytest.fixture
def reporting(controller, monkeypatch):
    from tdn import reporting
    monkeypatch.setattr(reporting, "ROOT", controller.ROOT)
    for key in ("TDN_TOWER_DIR", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID"):
        monkeypatch.delenv(key, raising=False)
    return reporting


@pytest.mark.parametrize("stage", ["accuracy", "scaling", "prepare", "neural"])
def test_actual_tower_boundary_accepts_frozen_workflow_sidecar(controller, reporting, monkeypatch, stage):
    workflow = plan(controller)
    base = freeze(controller, workflow)
    monkeypatch.setenv("SLURM_JOB_ID", "9001")
    report = controller.begin_report(workflow, stage)
    assert report.parent == base / "tower" and not (base / stage).exists()
    run = reporting.read_json(report / "run.json")
    assert run["parameters"]["execution_mode"] == "desktop-slurm"
    assert run["parameters"]["slurm_profile_sha256"] == workflow["slurm_profile_sha256"]
    assert run["metadata"]["source_directory"] == str((base / stage).relative_to(controller.ROOT))
    source = seal(controller, workflow, stage)
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    monkeypatch.setenv("TDN_TOWER_DIR", str(report))
    assert reporting.attach_report(source) == (report, False)
    controller.pw.finish_report(workflow, stage, report, state="COMPLETED", runtime_seconds=.1, exit_code=0)
    summary = reporting.read_json(report / "summary.json")
    assert summary["state"] == "COMPLETED" and summary["results"]["reporting_omission_count"] == 0
    assert before == {path.name: path.read_bytes() for path in source.iterdir()}


def test_actual_tower_startup_failure_is_visible(controller, reporting, monkeypatch):
    workflow = plan(controller)
    base = freeze(controller, workflow)
    monkeypatch.setenv("SLURM_JOB_ID", "9001")
    monkeypatch.setenv("SLURM_STEP_ID", "0")
    monkeypatch.setattr(controller, "verify_allocation", lambda *a: None)
    def broken_venv(*args):
        raise ValueError("Existing venv differs from pinned dependencies")
    monkeypatch.setattr(controller.rw, "software_report", broken_venv)
    with pytest.raises(controller.rw.WorkerFailure, match="venv differs"):
        controller.worker(workflow, "accuracy")
    state = controller.cw.read_json(base / "state" / "accuracy.json")
    assert state["status"] == "FAILED" and state["stage"] == "startup"
    reports = controller.cw.tower_reports_for_job(workflow, "9001")
    assert len(reports) == 1 and reporting.read_json(reports[0] / "summary.json")["state"] == "FAILED"
    assert not (base / "accuracy").exists()


@pytest.mark.parametrize("script", ["fedora_slurm.sh", "fedora_slurm_worker.sh"])
def test_shell_syntax(script):
    subprocess.run(["bash", "-n", str(ROOT / "scripts" / script)], check=True)
