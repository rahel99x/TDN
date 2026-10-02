"""Desktop runtime policy without requiring a Windows host or CUDA device."""
import copy
import os
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace

import pytest

from tdn.config import load_config
from tdn.runtime import preflight, storage
from tdn.runtime.signal_handling import StopRequest, ignore_worker_warning

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def desktop(monkeypatch, tmp_path):
    for name in ("SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_JOB_ACCOUNT", "CONDA_PREFIX"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("TDN_PROJECT_ROOT", raising=False)
    # Model a separate desktop checkout even when pytest runs at the CARC root.
    # Keep the real root resolver and its CARC/Slurm rejection checks active.
    monkeypatch.setattr(storage, "__file__", str(tmp_path / "tdn/runtime/storage.py"))
    monkeypatch.setenv("TDN_EXECUTION_MODE", "desktop")
    monkeypatch.setattr(sys, "prefix", str(tmp_path / ".venv"))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "standalone-python"))
    # configure_storage mutates these variables; retain test isolation.
    for name in ("TMPDIR", "TMP", "TEMP", "PIP_CACHE_DIR", "XDG_CACHE_HOME", "TORCH_HOME",
                 "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "TORCH_EXTENSIONS_DIR",
                 "CUDA_CACHE_PATH", "MPLCONFIGDIR", "PYTHONPYCACHEPREFIX", "MPLBACKEND",
                 "PYTHONNOUSERSITE", "CUBLAS_WORKSPACE_CONFIG"):
        monkeypatch.setenv(name, os.environ.get(name, ""))
    return tmp_path


def test_desktop_cpu_does_not_require_or_probe_cuda(desktop, monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CPU preflight probed CUDA"))
    preflight.verify_runtime("cpu", "train")
    assert preflight.execution_mode() == "desktop"


def test_desktop_requires_its_own_standalone_venv(desktop, monkeypatch):
    monkeypatch.setattr(sys, "prefix", str(desktop / "another-venv"))
    with pytest.raises(ValueError, match="project root/.venv"):
        preflight.verify_runtime("cpu", "train")
    monkeypatch.setattr(sys, "prefix", str(desktop / ".venv"))
    monkeypatch.setenv("CONDA_PREFIX", str(desktop / "conda"))
    with pytest.raises(ValueError, match="standalone"):
        preflight.verify_runtime("cpu", "train")


def test_desktop_rejects_conda_base_without_active_conda_env(desktop, monkeypatch, tmp_path):
    base = tmp_path / "custom-python"
    (base / "conda-meta").mkdir(parents=True)
    monkeypatch.setattr(sys, "base_prefix", str(base))
    with pytest.raises(ValueError, match="standalone"):
        preflight.verify_runtime("cpu", "train")


@pytest.mark.parametrize("name", ["SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_JOB_ACCOUNT"])
def test_desktop_never_bypasses_slurm(desktop, monkeypatch, name):
    monkeypatch.setenv(name, "123")
    with pytest.raises(ValueError, match="cannot bypass"):
        storage.project_root()
    with pytest.raises(ValueError, match="cannot bypass"):
        preflight.verify_runtime("cpu", "train")


def test_desktop_never_bypasses_carc_root(desktop, monkeypatch):
    monkeypatch.setattr(preflight, "project_root", lambda: storage.CARC_ROOT)
    with pytest.raises(ValueError, match="cannot bypass"):
        preflight.verify_runtime("cpu", "train")


def test_ordinary_local_cuda_still_rejected(desktop, monkeypatch):
    monkeypatch.delenv("TDN_EXECUTION_MODE")
    with pytest.raises(ValueError, match="actual CARC"):
        preflight.verify_runtime("cuda", "train")


def test_carc_still_requires_allocation_and_a100(desktop, monkeypatch):
    import torch
    monkeypatch.delenv("TDN_EXECUTION_MODE")
    monkeypatch.setattr(preflight, "project_root", lambda: storage.CARC_ROOT)
    monkeypatch.setattr(sys, "prefix", str(storage.CARC_ROOT / ".venv"))
    monkeypatch.setattr(preflight.getpass, "getuser", lambda: "aadaniel")
    with pytest.raises(ValueError, match="through srun"):
        preflight.verify_runtime("cpu", "train")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_STEP_ID", "0")
    monkeypatch.setenv("SLURM_JOB_ACCOUNT", "other-lab")
    with pytest.raises(ValueError, match="anakano_81"):
        preflight.verify_runtime("cpu", "train")
    monkeypatch.setenv("SLURM_JOB_ACCOUNT", "anakano_81")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda _: SimpleNamespace(name="GeForce RTX", total_memory=8 * 2**30, major=8, minor=6))
    with pytest.raises(ValueError, match="full A100"):
        preflight.verify_runtime("cuda", "train")


def test_desktop_selects_consumer_gpu_among_multiple_devices(desktop, monkeypatch):
    import torch
    selected = []
    monkeypatch.setenv("TDN_DESKTOP_CUDA_DEVICE", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_device", selected.append)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda _: SimpleNamespace(name="GeForce RTX", total_memory=8 * 2**30, major=8, minor=6))
    preflight.verify_runtime("cuda", "train")
    assert selected == [1]


@pytest.mark.parametrize("value", ["-1", "bad", "1.0", "2"])
def test_invalid_desktop_device_indices(desktop, monkeypatch, value):
    import torch
    monkeypatch.setenv("TDN_DESKTOP_CUDA_DEVICE", value)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="index|nonnegative"):
        preflight.verify_runtime("cuda", "train")


def test_cached_temp_directory_is_reset(desktop, monkeypatch, tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.setattr(storage, "project_root", lambda: root)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "old-system-temp"))
    assert storage.configure_storage() == root
    assert Path(tempfile.gettempdir()) == root / ".runtime" / "tmp"
    for name in ("TMPDIR", "TEMP", "TMP", "TORCH_HOME", "PYTHONPYCACHEPREFIX"):
        assert Path(os.environ[name]).is_relative_to(root)


def test_storage_refuses_redirected_temp_directory(desktop, monkeypatch, tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / ".runtime").mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (root / ".runtime" / "tmp").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Windows account lacks symlink permission")
    monkeypatch.setattr(storage, "project_root", lambda: root)
    with pytest.raises(ValueError, match="escapes"):
        storage.configure_storage()


def test_desktop_ctrl_c_is_a_safe_stop_request(desktop):
    original = signal.getsignal(signal.SIGINT)
    with StopRequest() as stop:
        signal.raise_signal(signal.SIGINT)
        assert stop.requested
        assert stop.signal_number == signal.SIGINT
    assert signal.getsignal(signal.SIGINT) == original


def test_stop_handler_supports_platform_without_sigusr1(desktop, monkeypatch):
    monkeypatch.delattr(signal, "SIGUSR1", raising=False)
    with StopRequest() as stop:
        assert signal.SIGINT in stop.previous
        assert signal.SIGTERM in stop.previous


def test_desktop_ctrl_c_checkpoints_after_complete_optimizer_update(desktop, monkeypatch, tmp_path):
    import torch
    import tdn.train.loop as loop
    from tdn.data import generate_dataset
    from tdn.train import load_checkpoint
    config = load_config(ROOT / "configs" / "smoke.yaml")
    config["problem"]["grid"] = [4, 4]
    config["data"].update(train_count=2, validation_count=1, diagnostic_count=1)
    config["training"].update(max_steps=3, validation_every_steps=3)
    data = tmp_path / "data"
    run = tmp_path / "training"
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        generate_dataset(config, data)
        original = loop._regression_backward

        def interrupt_during_update(*args, **kwargs):
            loss = original(*args, **kwargs)
            signal.raise_signal(signal.SIGINT)
            return loss

        monkeypatch.setattr(loop, "_regression_backward", interrupt_during_update)
        with pytest.raises(SystemExit) as paused:
            loop.train(config, data, run)
        assert paused.value.code == 75
        checkpoint = load_checkpoint(run)
        assert checkpoint["status"] == "PAUSED_NEEDS_RESUME"
        assert checkpoint["global_step"] == 1
        assert checkpoint["sampler"]["committed_cursor"] == 1
        assert checkpoint["scheduler_state"]["last_epoch"] == 1
        assert checkpoint["optimizer_state"]["state"]
        assert all(float(state["step"]) == 1 for state in checkpoint["optimizer_state"]["state"].values())
    finally:
        torch.set_num_threads(old_threads)


def test_worker_ignores_desktop_console_interrupt(desktop, monkeypatch):
    installed = []
    monkeypatch.setattr(signal, "signal", lambda number, handler: installed.append((number, handler)))
    ignore_worker_warning()
    assert (signal.SIGINT, signal.SIG_IGN) in installed


def test_desktop_calibration_measures_eager_pipeline_without_optimized_passes(desktop, monkeypatch, tmp_path):
    import torch
    from tdn.runtime.calibration import calibrate
    monkeypatch.setattr(torch, "compile", lambda *_args, **_kwargs: pytest.fail("Desktop calibration compiled"))
    monkeypatch.setattr(torch, "autocast", lambda *_args, **_kwargs: pytest.fail("Desktop calibration used autocast"))
    config = load_config(ROOT / "configs" / "smoke.yaml")
    config["problem"]["grid"] = [4, 4]
    report = calibrate(config, tmp_path / "calibration", "cpu")
    assert report["passed"]
    assert report["execution_mode"] == "desktop"
    assert report["device"] == "cpu"
    assert not report["optimized_eligible"]
    assert all(report["precision"]["eager_finite_checks"].values())
    for name in ("bf16", "compile", "combined"):
        assert report["precision"][name + "_status"] == "UNRUN"
        assert report["precision"][name + "_passed"] is False
    assert report["memory"]["cuda_status"] == "UNRUN"
    assert report["memory"]["cuda_budget_passed"] is None
    assert {"teacher_fp64", "split_fp32", "features_fp32", "encode_fp32", "decode_fp32",
            "short_full_domain_rollout", "full_domain_training_backward"} <= report["phase_profiles"].keys()
    assert all(p["cuda_event_seconds_raw"] is None for p in report["phase_profiles"].values())
    assert (tmp_path / "calibration" / "calibration.json").is_file()


@pytest.mark.parametrize("section,value", [("compile_mode", "default"), ("network_autocast", "bfloat16")])
def test_desktop_calibration_rejects_unvalidated_optimizations(desktop, tmp_path, section, value):
    from tdn.runtime.calibration import calibrate
    config = copy.deepcopy(load_config(ROOT / "configs" / "smoke.yaml"))
    config["precision"][section] = value
    with pytest.raises(ValueError, match="eager FP32"):
        calibrate(config, tmp_path / "calibration", "cpu")
