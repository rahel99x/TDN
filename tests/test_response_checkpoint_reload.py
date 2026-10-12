"""Fresh-process inference must use the trained model's precision contract."""
import json

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


def test_reload_uses_ieee_in_both_processes_and_matches_thread_settings(tmp_path):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
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
        for runtime in (result["parent_runtime"], result["child_runtime"]):
            assert runtime["threads"] == 1
            settings = runtime["precision"]
            expected = "ieee" if settings["api"] == "fp32_precision" else False
            assert settings["cuda_matmul"] == expected
            assert settings["cudnn_convolution"] == expected
        assert json.loads((tmp_path/"checkpoint_reload.json").read_text()) == result
    finally:
        torch.set_num_threads(old_threads)
        tr.reference_precision()


@pytest.mark.parametrize("corruption", ["output", "state", "runtime"])
def test_reload_remains_fail_closed_for_corruption(tmp_path, monkeypatch, corruption):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        sample = checkpoint_fixture(tmp_path)
        launch = tr.subprocess.run

        def corrupt(*args, **kwargs):
            completed = launch(*args, **kwargs)
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
        else:
            assert not result["runtime_equal"]
    finally:
        torch.set_num_threads(old_threads)
