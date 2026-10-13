"""Fresh-process inference must use the trained model's precision contract."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from tdn.analysis.response import training as tr
from tdn.analysis.response.models import create_model


def checkpoint_fixture(path, target="fd_same_grid", device="cpu"):
    """Nonzero direct predictions expose backend changes hidden by an anchor."""
    config = dict(family="direct_fno", width=16, depth=2, modes=4,
                  normalization=dict(mean_scale=.01, centered_scale=.01,
                                     full_mean=.4, full_scale=.2, floor=1e-7))
    torch.manual_seed(1170321)
    model = create_model(config).to(device)
    size = 32
    x = torch.arange(size, device=device, dtype=torch.float32)/size
    u = (.4+.09*torch.cos(2*torch.pi*x[:, None])+.03*torch.sin(4*torch.pi*x[None, :]+.3))[None, None]
    sample = dict(u0=u, record=dict(parent_id="reload-probe", split="train", grid=[size, size],
        lengths=[1., 1.], horizon=.06, kappa=.01, reaction_rate=3., target_kind=target))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad()
        (tr.predict(model, sample)-(u+.01)).square().mean().backward()
        optimizer.step()
    tr.atomic_torch_save(dict(config=config, normalization=config["normalization"],
                             state_dict=model.state_dict()), path/"selected.pt")
    return sample


@pytest.mark.parametrize("deterministic", (False, True))
@pytest.mark.parametrize("cudnn_benchmark,cudnn_deterministic", ((False, False), (True, True)))
def test_reload_uses_ieee_in_both_processes_and_matches_thread_settings(tmp_path, deterministic, cudnn_benchmark, cudnn_deterministic):
    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    old_cudnn = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=deterministic)
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic
        sample = checkpoint_fixture(tmp_path)
        # Set the backend's actual fresh-process default that caused the CUDA
        # direct-FNO failures. This CPU check verifies policy, not CUDA arithmetic.
        if hasattr(torch.backends, "fp32_precision"):
            torch.backends.cudnn.conv.fp32_precision = "tf32"
        else:
            torch.backends.cudnn.allow_tf32 = True
        result = tr.checkpoint_reload_probe(tmp_path, sample)
        assert result["status"] == "PASS"
        assert result["exact_equal"] and result["maximum_difference"] == 0.
        assert result["runtime_equal"] and result["state_equal"] and result["checkpoint_equal"]
        assert result["source_equal"]
        assert result["parent_source_tree_sha256"] == result["child_source_tree_sha256"] == tr.software_metadata()["source_tree_sha256"]
        for runtime in (result["parent_runtime"], result["child_runtime"]):
            assert runtime["threads"] == 1
            assert runtime["deterministic_algorithms"] == deterministic
            assert runtime["deterministic_warn_only"] == deterministic
            assert runtime["cudnn_benchmark"] == cudnn_benchmark
            assert runtime["cudnn_deterministic"] == cudnn_deterministic
            settings = runtime["precision"]
            expected = "ieee" if settings["api"] == "fp32_precision" else False
            assert settings["cuda_matmul"] == expected
            assert settings["cudnn_convolution"] == expected
        assert json.loads((tmp_path/"checkpoint_reload.json").read_text()) == result
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn_only)
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = old_cudnn
        tr.reference_precision()


@pytest.mark.parametrize("corruption", ["output", "state", "runtime", "source"])
def test_reload_remains_fail_closed_for_corruption(tmp_path, monkeypatch, corruption):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        sample = checkpoint_fixture(tmp_path)
        launch = tr.subprocess.run

        def corrupt(*args, **kwargs):
            completed = launch(*args, **kwargs)
            if "tdn.analysis.response.training" not in args[0]:
                return completed
            assert completed.returncode == 0, completed.stderr
            if corruption == "output":
                file = tmp_path/"reload-output.pt"
                value = torch.load(file, weights_only=True)
                value.flatten()[0] += .01
                torch.save(value, file)
            else:
                file = tmp_path/"reload-runtime.json"
                value = json.loads(file.read_text())
                if corruption == "state":
                    value["state_sha256"] = "0"*64
                elif corruption == "source":
                    value["source_tree_sha256"] = "0"*64
                else:
                    value["runtime"]["threads"] += 1
                file.write_text(json.dumps(value))
            return completed

        monkeypatch.setattr(tr.subprocess, "run", corrupt)
        result = tr.checkpoint_reload_probe(tmp_path, sample)
        assert result["status"] == "FAIL"
        if corruption == "output":
            assert not result["exact_equal"] and result["maximum_difference"] > .009
        elif corruption == "state":
            assert not result["state_equal"]
        elif corruption == "source":
            assert not result["source_equal"]
        else:
            assert not result["runtime_equal"]
    finally:
        torch.set_num_threads(old_threads)


def test_stale_parent_source_rejected_before_model_or_child(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Stale parent source must be rejected before loading a checkpoint")
    monkeypatch.setattr(tr, "load_model", forbidden)
    actual = tr.software_metadata()["source_tree_sha256"]
    stale = ("0" if actual[0] != "0" else "1")+actual[1:]
    launch = tr.subprocess.run
    def no_reload(command, *args, **kwargs):
        assert "tdn.analysis.response.training" not in command
        return launch(command, *args, **kwargs)
    monkeypatch.setattr(tr.subprocess, "run", no_reload)
    with pytest.raises(tr.SourceChangedError, match="before checkpoint reload"):
        tr.checkpoint_reload_probe(tmp_path, {}, expected_source_tree_sha256=stale)
    assert not list(tmp_path.iterdir())


def test_actual_reload_child_rejects_stale_source_before_prediction(tmp_path):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        checkpoint_fixture(tmp_path)
        runtime_path = tmp_path/"runtime.json"
        runtime_path.write_text(json.dumps(tr._checkpoint_runtime()))
        actual = tr.software_metadata()["source_tree_sha256"]
        stale = ("0" if actual[0] != "0" else "1")+actual[1:]
        command = [sys.executable, "-m", "tdn.analysis.response.training", "reload", str(tmp_path),
                   "cpu", "1", "1", str(runtime_path), stale]
        completed = subprocess.run(command, cwd=Path(tr.__file__).resolve().parents[3],
                                   capture_output=True, text=True, timeout=60, check=False)
        assert completed.returncode != 0
        assert "SourceChangedError" in completed.stderr and "before checkpoint reload child" in completed.stderr
        assert actual in completed.stderr and stale in completed.stderr
        assert not (tmp_path/"reload-output.pt").exists()
        assert not (tmp_path/"reload-runtime.json").exists()
    finally:
        torch.set_num_threads(old_threads)


def test_engine_binds_entry_source_to_stage_context(tmp_path, monkeypatch):
    from tdn.analysis.response import engine
    from tdn.analysis.response.protocol import build_protocol
    source = "a"*64
    monkeypatch.setattr(engine, "_inside", lambda p: Path(p).resolve())
    monkeypatch.setattr(engine, "software_metadata", lambda: dict(source_tree_sha256=source))
    def dispatch(ctx):
        assert ctx.source_tree_sha256 == source
        raise RuntimeError("TEST_STOP_AFTER_FROZEN_SOURCE_BINDING")
    monkeypatch.setattr(engine, "_dispatch", dispatch)
    protocol = build_protocol("smoke", local_smoke=True, targets=["fd_same_grid"])
    with pytest.raises(RuntimeError, match="TEST_STOP_AFTER_FROZEN_SOURCE_BINDING"):
        engine.run_stage(protocol, "audit", tmp_path, device="cpu")
    summary = json.loads((tmp_path/"summary.json").read_text())
    assert summary["source_tree_sha256"] == source and summary["status"] == "FAILED"
