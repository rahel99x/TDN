"""Desktop bootstrap policy without installation, real CUDA or live Slurm."""
from __future__ import annotations

import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def bootstrap(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("fedora_setup_test", ROOT / "scripts/fedora_setup.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "Fedora project with spaces"
    root.mkdir()
    monkeypatch.setattr(module, "ROOT", root)
    for name in ("CONDA_PREFIX", "CONDA_SHLVL", "SLURM_JOB_ID", "SLURM_STEP_ID", "TDN_SLURM_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    return module


def observed(stdout, returncode=0):
    return {"stdout": stdout, "stderr": "", "returncode": returncode}


def interpreter():
    return {"executable": "/usr/bin/python3.13", "base_executable": "/usr/bin/python3.13",
            "version": [3, 13, 13], "bits": 64, "is_venv": False, "is_conda": False}


def configured(module, monkeypatch, *flags):
    monkeypatch.setattr(module, "inspect_python", lambda _: interpreter())
    monkeypatch.setattr(module, "command", lambda _: observed("desktop*|up|gpu:rtx4090:1(S:0)"))
    assert module.main(["configure", "--python", "python3.13", *flags]) == 0
    return json.loads((module.ROOT / module.PROFILE).read_text())


def test_doctor_is_read_only_without_dependencies_or_scheduler(bootstrap, monkeypatch, capsys):
    monkeypatch.setattr(bootstrap, "command", lambda argv: {**observed("", None), "command": argv})
    monkeypatch.setattr(bootstrap.shutil, "which", lambda _: None)
    monkeypatch.setattr(bootstrap, "profile_module", lambda: pytest.fail("doctor loaded runtime dependencies"))
    assert bootstrap.main(["doctor"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["partitions"] == [] and report["gpus"] == []
    assert report["associations"]["command"][5:7] == ["where", f"user={bootstrap.getpass.getuser()}"]
    assert list(bootstrap.ROOT.iterdir()) == []
    assert "no numerical or GPU readiness claim" in report["scope"]


def test_partition_discovery_preserves_typed_gres_and_default(bootstrap):
    rows = bootstrap.partitions("work*|up|gpu:rtx4090:1(S:0)\nwork*|up|gpu:rtx4090:1(S:0)\ncpu|up|(null)")
    assert len(rows) == 2
    assert bootstrap.choose_partition(rows, None, gpu=True) == "work"
    assert bootstrap.choose_partition(rows, None, gpu=False) == "work"
    assert bootstrap.choose_partition(rows, "cpu", gpu=False) == "cpu"
    with pytest.raises(ValueError, match="GPU GRES"):
        bootstrap.choose_partition(rows, "cpu", gpu=True)


def test_ambiguous_or_down_partitions_need_explicit_selection(bootstrap):
    rows = bootstrap.partitions("one|up|gpu:1\ntwo|up|gpu:1\ndown*|down|gpu:1")
    with pytest.raises(ValueError, match="unambiguous"):
        bootstrap.choose_partition(rows, None, gpu=True)
    with pytest.raises(ValueError, match="absent or not UP"):
        bootstrap.choose_partition(rows, "down", gpu=True)
    assert bootstrap.choose_partition(rows, "two", gpu=True) == "two"


def test_configure_binds_current_checkout_user_and_never_inherits_carc(bootstrap, monkeypatch):
    monkeypatch.setenv("CARC_ACCOUNT", "anakano_81")
    profile = configured(bootstrap, monkeypatch)
    assert profile["root"] == str(bootstrap.ROOT)
    assert profile["user"] == bootstrap.getpass.getuser()
    assert profile["account"] is None
    assert profile["python"] == "/usr/bin/python3.13"
    assert profile["gpu_vram_gib"] == 24
    tower = json.loads((bootstrap.ROOT / ".tower/fedora-slurm.json").read_text())
    assert "account" not in tower
    assert tower["profiles"]["desktop-slurm"]["gpu_sampling"] is False
    assert tower["research"]["contract"] == str(bootstrap.ROOT / ".tower/contracts/outputs.v1.json")
    assert not (bootstrap.ROOT / ".tower/config.json").exists()


def test_configure_explicit_local_account_and_overwrite_guard(bootstrap, monkeypatch):
    profile = configured(bootstrap, monkeypatch, "--account", "desktoplab")
    assert profile["account"] == "desktoplab"
    before = (bootstrap.ROOT / bootstrap.PROFILE).read_bytes()
    assert bootstrap.main(["configure", "--account", "different"]) == 2
    assert (bootstrap.ROOT / bootstrap.PROFILE).read_bytes() == before
    assert bootstrap.main(["configure", "--replace", "--account", "different"]) == 0


@pytest.mark.parametrize("name", ["CONDA_PREFIX", "CONDA_SHLVL", "SLURM_JOB_ID", "SLURM_STEP_ID"])
def test_configure_rejects_active_environment_without_writes(bootstrap, monkeypatch, name):
    monkeypatch.setenv(name, "1")
    assert bootstrap.main(["configure"]) == 2
    assert list(bootstrap.ROOT.iterdir()) == []


def test_carc_cannot_be_reconfigured_as_desktop(bootstrap, monkeypatch):
    monkeypatch.setattr(bootstrap, "ROOT", bootstrap.CARC_ROOT)
    with pytest.raises(ValueError, match="CARC"):
        bootstrap.standalone()


@pytest.mark.parametrize("version,index", [
    ("2.10", "https://download.pytorch.org/whl/cu126"),
    ("2.10.0", "https://download.pytorch.org/whl/cu126"),
    ("2.10.0+cu128", "https://download.pytorch.org/whl/cu126"),
    ("2.10.0+cu126", "https://example.org/whl/cu126"),
    ("2.10.0+cpu", "https://download.pytorch.org/whl/cpu"),
    ("2.10.0+cu999", "https://download.pytorch.org/whl/cu999"),
])
def test_invalid_or_unreviewed_wheel_selection_rejected(bootstrap, version, index):
    with pytest.raises(ValueError):
        bootstrap.wheel_check(version, index)


def gpu_profile(module):
    return {"torch_version": module.TORCH_VERSION, "torch_wheel_index": module.TORCH_INDEX,
            "expected_gpu_name": "RTX 4090", "gpu_vram_gib": 24}


def test_driver_inventory_requires_matching_gpu_memory_and_driver(bootstrap):
    report = bootstrap.check_driver(gpu_profile(bootstrap), observed("0, NVIDIA GeForce RTX 4090, 580.95.05, 24564"))
    assert report["cuda_build"] == "cu126"
    assert "checked inside" in report["scope"]
    for text, message in [
        ("0, NVIDIA GeForce RTX 4090, 550.54.14, 24564", "below"),
        ("0, NVIDIA GeForce RTX 4090, 580.95.05, 16384", "memory"),
        ("0, NVIDIA A100, 580.95.05, 40960", "configured GPU"),
    ]:
        with pytest.raises(ValueError, match=message):
            bootstrap.check_driver(gpu_profile(bootstrap), observed(text))


def test_setup_checks_driver_before_any_install_or_cache(bootstrap, monkeypatch):
    profile = configured(bootstrap, monkeypatch)
    monkeypatch.setattr(bootstrap, "observe_gpu", lambda: observed("0, RTX 4090, 550.1.1, 24576"))
    monkeypatch.setattr(bootstrap, "execute", lambda *_: pytest.fail("unsupported driver started installation"))
    assert bootstrap.main(["setup"]) == 2
    assert not (bootstrap.ROOT / ".venv").exists()
    assert not (bootstrap.ROOT / ".runtime").exists()


def test_shared_worker_lock_prevents_setup_and_reconfiguration(bootstrap, monkeypatch):
    configured(bootstrap, monkeypatch)
    monkeypatch.setattr(bootstrap, "observe_gpu", lambda: observed("0, RTX 4090, 580.95.05, 24576"))
    lock = (bootstrap.ROOT / ".cache/carc-phase.lock").open("a+b")
    try:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        assert bootstrap.main(["setup"]) == 2
        assert bootstrap.main(["configure", "--replace"]) == 2
    finally:
        lock.close()
    assert not (bootstrap.ROOT / ".venv").exists()
    assert not (bootstrap.ROOT / ".runtime").exists()


def test_install_env_keeps_all_writes_under_project_and_clears_pip_overrides(bootstrap, monkeypatch):
    for name in ("PIP_TARGET", "PIP_PREFIX", "PIP_USER", "PIP_LOG", "PIP_EXTRA_INDEX_URL", "PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE"):
        monkeypatch.setenv(name, "untrusted-external-location")
    env = bootstrap.install_environment()
    for name in ("PIP_TARGET", "PIP_PREFIX", "PIP_USER", "PIP_LOG", "PIP_EXTRA_INDEX_URL", "PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE"):
        assert name not in env
    for name in ("TMPDIR", "TMP", "TEMP", "PIP_CACHE_DIR", "XDG_CACHE_HOME", "TORCH_HOME", "CUDA_CACHE_PATH", "PYTHONPYCACHEPREFIX"):
        assert Path(env[name]).is_relative_to(bootstrap.ROOT)
    assert env["PIP_CONFIG_FILE"] == os.devnull


def test_symlink_escape_rejected_before_profile_write(bootstrap, tmp_path):
    (bootstrap.ROOT / ".tdn").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        bootstrap.inside(bootstrap.PROFILE)


def test_existing_incompatible_venv_is_preserved(bootstrap, monkeypatch):
    prefix = bootstrap.ROOT / ".venv"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "pyvenv.cfg").write_text("original")
    (prefix / "bin/python").write_text("original")
    report = {"prefix": str(prefix), "base": "/usr", "executable": "/usr/bin/python3.12", "version": [3, 12, 14]}
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=json.dumps(report)))
    with pytest.raises(ValueError, match="incompatible"):
        bootstrap.verify_existing_venv(prefix, interpreter(), {})
    assert (prefix / "pyvenv.cfg").read_text() == "original"


@pytest.mark.parametrize("change", [{"version": [3, 14, 0]}, {"bits": 32}, {"is_conda": True}, {"is_venv": True}])
def test_inspect_python_rejects_unsupported_bases(bootstrap, monkeypatch, change):
    monkeypatch.setattr(bootstrap.shutil, "which", lambda _: "/usr/bin/python3")
    monkeypatch.setattr(bootstrap, "command", lambda _: observed(json.dumps({**interpreter(), **change})))
    with pytest.raises(ValueError, match="standalone 64-bit"):
        bootstrap.inspect_python("python3")


def test_python313_is_supported(bootstrap, monkeypatch):
    monkeypatch.setattr(bootstrap.shutil, "which", lambda _: "/usr/bin/python3.13")
    monkeypatch.setattr(bootstrap, "command", lambda _: observed(json.dumps(interpreter())))
    assert bootstrap.inspect_python("python3.13")["version"] == [3, 13, 13]


def test_setup_wrapper_has_valid_bash_syntax():
    subprocess.run(["bash", "-n", str(ROOT / "scripts/fedora_setup.sh")], check=True)


def test_setup_commands_and_software_record_are_project_contained(bootstrap, monkeypatch):
    profile = configured(bootstrap, monkeypatch)
    (bootstrap.ROOT / "requirements.txt").write_text("numpy==2.2.6\n")
    monkeypatch.setattr(bootstrap, "observe_gpu", lambda: observed("0, RTX 4090, 580.95.05, 24576"))
    calls = []

    def execute(argv, env):
        calls.append(argv)
        assert Path(env["TMPDIR"]).is_relative_to(bootstrap.ROOT)
        if argv[1:3] == ["-m", "venv"]:
            (Path(argv[3]) / "bin").mkdir(parents=True)

    def run(argv, **kwargs):
        if argv[-2:] == ["pip", "freeze"]:
            return SimpleNamespace(stdout="torch==2.10.0+cu126\n")
        assert "torch.version.cuda" in argv[-1]
        assert "cuda.is_available" not in argv[-1]
        return SimpleNamespace(stdout=json.dumps({"python": "3.13.13", "torch_cuda_runtime": "12.6",
            "python_executable": str(bootstrap.ROOT / ".venv/bin/python"),
            "venv": str(bootstrap.ROOT / ".venv"), "packages": {"torch": "2.10.0+cu126"}}))

    monkeypatch.setattr(bootstrap, "execute", execute)
    monkeypatch.setattr(bootstrap.subprocess, "run", run)
    assert bootstrap.main(["setup"]) == 0
    assert len(calls) == 6
    assert calls[0] == ["/usr/bin/python3.13", "-m", "venv", str(bootstrap.ROOT / ".venv")]
    assert calls[2][-3:] == ["torch==2.10.0+cu126", "--index-url", bootstrap.TORCH_INDEX]
    assert calls[-1][-3:] == ["-m", "pip", "check"]
    report = json.loads((bootstrap.ROOT / ".tdn/fedora-environment.json").read_text())
    assert report["status"] == "COMPLETED" and report["configured_profile"] == profile
    assert "checked in allocations" in report["scope"]
    assert not (bootstrap.ROOT / "requirements/environment-versions.json").exists()
