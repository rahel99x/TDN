"""Policy tests with mocked scheduler/device evidence, not actual GPU validation."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tdn.runtime import desktop_slurm as policy
from tdn.runtime import preflight, storage


pytestmark = pytest.mark.skipif(not hasattr(os, "getuid"), reason="Fedora Slurm runtime requires POSIX process identity")


@pytest.fixture
def desktop_slurm(monkeypatch, tmp_path):
    root = tmp_path / "checkout with spaces"
    root.mkdir()
    monkeypatch.setattr(storage, "__file__", str(root / "tdn/runtime/storage.py"))
    monkeypatch.delenv("TDN_PROJECT_ROOT", raising=False)
    monkeypatch.delenv("TDN_SLURM_CONFIG", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    for name in ("SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_JOB_ACCOUNT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TDN_EXECUTION_MODE", "desktop-slurm")
    monkeypatch.setattr(policy.getpass, "getuser", lambda: "desktop_user")
    monkeypatch.setattr(policy.os, "getuid", lambda: 1000)
    monkeypatch.setattr(sys, "prefix", str(root / ".venv"))
    monkeypatch.setattr(sys, "base_prefix", str(root / "standalone-python"))
    profile = {"schema_version": 1, "kind": "desktop-slurm", "root": str(root),
               "user": "desktop_user", "cpu_partition": "local", "gpu_partition": "local-gpu",
               "account": None, "gpu_gres": "gpu:rtx4090:1", "expected_gpu_name": "RTX 4090",
               "gpu_vram_gib": 24, "torch_version": "2.10.0+cu126",
               "torch_wheel_index": "https://download.pytorch.org/whl/cu126"}
    path = root / ".tdn/fedora-slurm.json"
    path.parent.mkdir()
    path.write_text(json.dumps(profile))
    return root, profile


def scheduler(monkeypatch, root, *, device="cpu", job_changes=None, step_changes=None):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_STEP_ID", "0")
    partition = "local-gpu" if device == "cuda" else "local"
    job = {"JobId": "123", "UserId": "desktop_user(1000)", "JobState": "RUNNING",
           "Partition": partition, "Account": "default", "WorkDir": str(root), "NumNodes": "1",
           "AllocTRES": "cpu=4,mem=16G,node=1,gres/gpu=1,gres/gpu:rtx4090=1"}
    step = {"StepId": "123.0", "UserId": "desktop_user(1000)", "State": "RUNNING",
            "Partition": partition, "Tasks": "1", "Nodes": "1",
            "TRES": "cpu=4,mem=16G,node=1,gres/gpu=1"}
    job.update(job_changes or {})
    step.update(step_changes or {})
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        assert command in (["scontrol", "show", "job", "123", "-o"],
                           ["scontrol", "show", "step", "123.0", "-o"])
        data = job if command[2] == "job" else step
        return SimpleNamespace(returncode=0, stdout=" ".join(f"{key}={value}" for key, value in data.items()), stderr="")

    monkeypatch.setattr(policy.subprocess, "run", run)
    return commands


def mock_cuda(monkeypatch, *, name="NVIDIA GeForce RTX 4090", total_gib=24, free_gib=20):
    import torch
    monkeypatch.setattr(torch, "__version__", "2.10.0+cu126")
    monkeypatch.setattr(torch.version, "cuda", "12.6")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_device", lambda index: None)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: SimpleNamespace(
        name=name, major=8, minor=9, total_memory=total_gib * 2**30))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index: (free_gib * 2**30, total_gib * 2**30))
    return torch


def test_profile_loading_and_mode_are_explicit(desktop_slurm):
    root, profile = desktop_slurm
    assert policy.load_profile() == profile
    assert storage.project_root() == root
    assert preflight.execution_mode() == "desktop-slurm"


@pytest.mark.parametrize("key,value", [
    ("schema_version", True), ("schema_version", 2), ("kind", "desktop"), ("user", "other"),
    ("root", "/home/somewhere-else"), ("root", "."), ("cpu_partition", "local,other"),
    ("gpu_partition", "--bad"), ("account", ""), ("account", "one two"),
    ("gpu_gres", "gpu:2"), ("gpu_gres", "gpu:rtx4090:0"), ("expected_gpu_name", None),
    ("expected_gpu_name", ""), ("gpu_vram_gib", True), ("gpu_vram_gib", float("nan")),
    ("gpu_vram_gib", 0), ("torch_version", "latest"), ("torch_version", "2.10.0+cpu"),
    ("torch_version", "2.10.0"),
    ("torch_version", "2.10.0+cu128"), ("torch_wheel_index", "https://download.pytorch.org/whl/cpu"),
    ("torch_wheel_index", "https://example.com/cu126"), ("python", "python3"), ("unexpected", 1),
])
def test_malformed_profiles_fail_closed(desktop_slurm, key, value):
    _, profile = desktop_slurm
    profile[key] = value
    with pytest.raises(ValueError):
        policy.validate_profile(profile)


def test_alternative_contained_profile_and_symlink_escape(desktop_slurm, monkeypatch, tmp_path):
    root, profile = desktop_slurm
    alternate = root / "alternate.json"
    alternate.write_text(json.dumps(profile))
    monkeypatch.setenv("TDN_SLURM_CONFIG", "alternate.json")
    assert policy.load_profile() == profile
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps(profile))
    alternate.unlink()
    alternate.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        policy.load_profile()


def test_duplicate_profile_fields_rejected(desktop_slurm):
    root, profile = desktop_slurm
    path = root / ".tdn/fedora-slurm.json"
    path.write_text(json.dumps(profile)[:-1] + ', "user": "desktop_user"}')
    with pytest.raises(ValueError, match="Duplicate"):
        policy.load_profile()


def test_profile_import_has_no_torch_requirement():
    root = Path(__file__).resolve().parents[1]
    code = "import sys; import tdn.runtime.desktop_slurm; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], cwd=root, check=True, capture_output=True)


def test_cpu_runtime_reads_actual_job_and_step_without_cuda(desktop_slurm, monkeypatch):
    root, _ = desktop_slurm
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CPU runtime probed CUDA"))
    commands = scheduler(monkeypatch, root)
    preflight.verify_runtime("cpu", "premix-accuracy")
    assert len(commands) == 2


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_allocation_accepts_slurm_allocnode_sid_field(desktop_slurm, monkeypatch, device):
    root, _ = desktop_slurm
    commands = scheduler(monkeypatch, root, device=device)
    scheduler_run = policy.subprocess.run

    def actual_format(command, **kwargs):
        result = scheduler_run(command, **kwargs)
        if command[2] == "job":
            partition = "local-gpu" if device == "cuda" else "local"
            result.stdout = result.stdout.replace(
                f"Partition={partition} ",
                f"Partition={partition} AllocNode:Sid=system:8749 ")
        return result

    monkeypatch.setattr(policy.subprocess, "run", actual_format)
    evidence = policy.verify_allocation(device)
    assert evidence["job"]["Partition"] == ("local-gpu" if device == "cuda" else "local")
    assert evidence["job"]["AllocNode:Sid"] == "system:8749"
    assert evidence["job"]["WorkDir"] == str(root)
    assert len(commands) == 2


def test_numeric_slurm_step_owner_is_checked_by_actual_uid(desktop_slurm, monkeypatch):
    root, _ = desktop_slurm
    scheduler(monkeypatch, root, step_changes={"UserId": "1000"})
    preflight.verify_runtime("cpu", "premix-accuracy")
    scheduler(monkeypatch, root, step_changes={"UserId": "1001"})
    with pytest.raises(ValueError, match="srun step"):
        preflight.verify_runtime("cpu", "premix-accuracy")


def test_login_config_validation_does_not_run_compute(desktop_slurm, monkeypatch):
    monkeypatch.setattr(policy.subprocess, "run", lambda *a, **kw: pytest.fail("Config validation probed scheduler"))
    preflight.verify_runtime("cpu", "validate-config")
    with pytest.raises(ValueError, match="allocated compute"):
        preflight.verify_runtime("cuda", "validate-config")


@pytest.mark.parametrize("job_id,step_id", [("", ""), ("123", "batch"), ("123", "extern"), ("123;bad", "0")])
def test_invented_or_batch_step_flags_do_not_authorize_compute(desktop_slurm, monkeypatch, job_id, step_id):
    monkeypatch.setenv("SLURM_JOB_ID", job_id)
    monkeypatch.setenv("SLURM_STEP_ID", step_id)
    with pytest.raises(ValueError, match="numeric srun"):
        preflight.verify_runtime("cpu", "premix-accuracy")


@pytest.mark.parametrize("level,key,value", [
    ("job", "JobId", "456"), ("job", "JobState", "PENDING"),
    ("job", "UserId", "desktop_user(9999)"), ("job", "UserId", "other(1000)"),
    ("job", "Partition", "other"), ("job", "WorkDir", "/home/somewhere-else"),
    ("job", "NumNodes", "2"), ("step", "Nodes", "2"), ("step", "Tasks", "2"),
    ("step", "StepId", "123.batch"), ("step", "State", "COMPLETED"),
    ("step", "UserId", "other(1000)"), ("step", "Partition", "other"),
])
def test_runtime_rejects_mismatched_scheduler_evidence(desktop_slurm, monkeypatch, level, key, value):
    root, _ = desktop_slurm
    scheduler(monkeypatch, root, **{f"{level}_changes": {key: value}})
    with pytest.raises(ValueError, match="desktop Slurm|Desktop Slurm"):
        preflight.verify_runtime("cpu", "premix-accuracy")


def test_configured_account_must_match_scheduler(desktop_slurm, monkeypatch):
    root, profile = desktop_slurm
    scheduler(monkeypatch, root)
    profile["account"] = "my-account"
    with pytest.raises(ValueError, match="account"):
        policy.verify_allocation("cpu", profile=profile)


@pytest.mark.parametrize("failure", ["unavailable", "timeout", "returncode"])
def test_unverifiable_scheduler_is_not_authorization(desktop_slurm, monkeypatch, failure):
    root, _ = desktop_slurm
    scheduler(monkeypatch, root)

    def fail(command, **kwargs):
        if failure == "unavailable":
            raise FileNotFoundError("scontrol")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 15)
        return SimpleNamespace(returncode=1, stdout="", stderr="Invalid job id")

    monkeypatch.setattr(policy.subprocess, "run", fail)
    with pytest.raises(ValueError, match="Cannot verify"):
        policy.verify_allocation("cpu")


def test_gpu_runtime_requires_actual_one_gpu_allocation_and_visible_gpu(desktop_slurm, monkeypatch):
    root, profile = desktop_slurm
    scheduler(monkeypatch, root, device="cuda")
    mock_cuda(monkeypatch)
    preflight.verify_runtime("cuda", "premix-neural")
    report = policy.verify_cuda_device(profile)
    assert report["execution_mode"] == "desktop-slurm"
    assert report["soft_reserved_budget_bytes"] == 15 * 2**30  # .75 of free, not host RAM
    assert report["memory_policy"] == {"soft_cap_bytes": 18 * 2**30, "soft_fraction": .75, "hard_fraction": .9}


def test_busy_display_gpu_hard_limit_stops_before_computation(desktop_slurm, monkeypatch):
    _, profile = desktop_slurm
    mock_cuda(monkeypatch, free_gib=2)
    with pytest.raises(ValueError, match="hard dedicated-memory"):
        policy.verify_cuda_device(profile)


@pytest.mark.parametrize("key,value", [("AllocTRES", "cpu=4,gres/gpu=2"), ("AllocTRES", "cpu=4"),
                                      ("TRES", "cpu=4,gres/gpu=0"), ("TRES", "cpu=4")])
def test_gpu_task_must_have_one_scheduler_assigned_gpu(desktop_slurm, monkeypatch, key, value):
    root, _ = desktop_slurm
    scheduler(monkeypatch, root, device="cuda", **{("job_changes" if key == "AllocTRES" else "step_changes"): {key: value}})
    with pytest.raises(ValueError, match="allocate exactly one"):
        policy.verify_allocation("cuda")


@pytest.mark.parametrize("record,expected", [
    ({"Gres": "gpu:rtx4090:1(IDX:0)"}, True), ({"TresPerNode": "gres/gpu:rtx4090:1"}, True),
    ({"AllocTRES": "cpu=4,gres/gpu=1,gres/gpu:rtx4090=1"}, True),
    ({"AllocTRES": "cpu=4,gres/gpu:a=1,gres/gpu:b=1"}, False),
    ({"AllocTRES": "gres/gpu=1", "TresPerNode": "gres/gpu:2"}, False),
])
def test_slurm_gpu_evidence_formats(record, expected):
    assert policy._one_gpu(record) is expected


@pytest.mark.parametrize("issue", ["unavailable", "multiple", "name", "vram", "version", "cuda-runtime", "memory"])
def test_incompatible_gpu_or_wheel_rejected(desktop_slurm, monkeypatch, issue):
    _, profile = desktop_slurm
    torch = mock_cuda(monkeypatch, name="A100" if issue == "name" else "NVIDIA GeForce RTX 4090",
                      total_gib=40 if issue == "vram" else 24)
    if issue == "unavailable":
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    elif issue == "multiple":
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    elif issue == "version":
        monkeypatch.setattr(torch, "__version__", "2.9.0+cu126")
    elif issue == "cuda-runtime":
        monkeypatch.setattr(torch.version, "cuda", "12.8")
    elif issue == "memory":
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index: (25 * 2**30, 24 * 2**30))
    with pytest.raises(ValueError):
        policy.verify_cuda_device(profile)


def test_standalone_project_venv_remains_required(desktop_slurm, monkeypatch):
    root, _ = desktop_slurm
    scheduler(monkeypatch, root)
    monkeypatch.setattr(sys, "prefix", str(root / "other-venv"))
    with pytest.raises(ValueError, match="project root/.venv"):
        preflight.verify_runtime("cpu", "premix-accuracy")
    monkeypatch.setattr(sys, "prefix", str(root / ".venv"))
    monkeypatch.setenv("CONDA_PREFIX", "/home/user/conda")
    with pytest.raises(ValueError, match="standalone"):
        preflight.verify_runtime("cpu", "premix-accuracy")


def test_new_mode_cannot_relax_carc_or_implicit_local_policy(desktop_slurm, monkeypatch):
    root, _ = desktop_slurm
    scheduler(monkeypatch, root)
    monkeypatch.delenv("TDN_EXECUTION_MODE")
    with pytest.raises(ValueError, match="CARC runtime root"):
        storage.project_root()
    monkeypatch.setenv("TDN_EXECUTION_MODE", "desktop-slurm")
    monkeypatch.setattr(storage, "__file__", str(storage.CARC_ROOT / "tdn/runtime/storage.py"))
    with pytest.raises(ValueError, match="cannot bypass"):
        storage.project_root()
    with pytest.raises(ValueError, match="cannot bypass"):
        policy.validate_profile({}, root=storage.CARC_ROOT)
