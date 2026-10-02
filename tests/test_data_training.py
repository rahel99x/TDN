"""Real paired labels, replay, live rollouts, and safe signal serialization."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

import numpy as np
import pytest
import torch
import yaml

from tdn.data import DatasetStore, generate_dataset, make_parent_split
from tdn.train import load_checkpoint, model_from_checkpoint, train
from tdn.train.loop import _geometry, _normalization


@pytest.fixture
def project_directory():
    # Independent of pytest's default temporary root, forbidden on this project.
    path = Path(__file__).resolve().parents[1] / "results" / "test-work" / uuid.uuid4().hex
    path.mkdir(parents=True)
    yield path
    shutil.rmtree(path)


@pytest.fixture
def configuration():
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml").read_text())
    config["problem"]["grid"] = [6, 6]
    config["data"].update(train_count=2, validation_count=1, diagnostic_count=1,
                          confirmatory_count=2, rollout_windows=[1, 2, 4])
    config["training"].update(max_steps=6, checkpoint_every_steps=2, validation_every_steps=3,
                               microbatch_cells=13)
    return config


def test_parent_split_deterministic_disjoint_sealed(configuration):
    split = make_parent_split(configuration)
    assert split == make_parent_split(copy.deepcopy(configuration))
    seeds = [p["seed"] for parents in split.values() for p in parents]
    assert len(set(seeds)) == len(seeds)
    ids = [p["parent_id"] for parents in split.values() for p in parents]
    assert len(set(ids)) == len(ids)


def test_fp64_defect_and_immutable_checksums(configuration, project_directory):
    path = project_directory / "data"
    generate_dataset(configuration, path)
    data = DatasetStore(path)
    arrays = data.arrays("train")
    np.testing.assert_array_equal(arrays["defects_fp64"], arrays["teacher"] - arrays["split"])
    np.testing.assert_array_equal(arrays["defects"], arrays["defects_fp64"].astype(np.float32))
    assert isinstance(arrays["states"], np.memmap)
    assert data.manifest["confirmatory"]["status"] == "SEALED"
    with pytest.raises(ValueError, match="sealed"):
        data.arrays("confirmatory")
    records = data.manifest["groups"]["train"]["samples"]
    assert records[0]["input_quantization_state_norm"] > 0
    for record in records:
        for reference in record["references"]:
            assert reference["accepted"]
            assert reference["uncertainty"] <= .05 * max(reference["defect_norm"], 1e-10)
            assert len(reference["refinement_substeps"]) == 3
            assert reference["input_quantization_defect_error"] >= 0
    assert generate_dataset(configuration, path) == path / "manifest.json"
    changed = copy.deepcopy(configuration)
    changed["problem"]["kappa"] *= 2
    with pytest.raises(ValueError, match="immutable"):
        generate_dataset(changed, path)
    label_path = path / data.manifest["groups"]["train"]["arrays"]["defects"]["path"]
    with label_path.open("r+b") as stream:
        stream.seek(-1, 2)
        value = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([value[0] ^ 1]))
    with pytest.raises(ValueError, match="checksum"):
        DatasetStore(path)


def test_normalization_only_training_parents(configuration, project_directory):
    first = project_directory / "first"
    second = project_directory / "second"
    generate_dataset(configuration, first)
    changed = copy.deepcopy(configuration)
    changed["data"]["validation_count"] = 3
    changed["data"]["diagnostic_count"] = 2
    generate_dataset(changed, second)
    one = _normalization(configuration, DatasetStore(first), _geometry(configuration))
    two = _normalization(changed, DatasetStore(second), _geometry(changed))
    assert one["mean"] == two["mean"]
    assert one["std"] == two["std"]
    assert one["defect_scale"] == two["defect_scale"]
    assert one["fit_split"] == "train"


def test_uninterrupted_resume_bitwise_and_fresh_process(configuration, project_directory):
    data_path = project_directory / "data"
    generate_dataset(configuration, data_path)
    whole = project_directory / "whole"
    resumed = project_directory / "resumed"
    one = train(configuration, data_path, whole)
    paused = train(configuration, data_path, resumed, max_steps=2)
    assert paused["status"] == "PAUSED_BUDGET"
    two = train(configuration, data_path, resumed, resume=resumed / "checkpoints" / "last.pt")
    assert one["history"] == two["history"]
    assert one["validation"] == two["validation"]
    a = load_checkpoint(whole)
    b = load_checkpoint(resumed)
    for name, value in a["model_state"].items():
        assert torch.equal(value, b["model_state"][name]), name
    assert a["sampler"] == b["sampler"]
    assert torch.equal(a["rng_state"]["torch_cpu"], b["rng_state"]["torch_cpu"])
    assert (resumed / "checkpoints" / "last.previous.pt").exists()
    model = model_from_checkpoint(b)
    features = torch.linspace(-1, 1, len(b["normalization"]["mean"])).reshape(1, -1)
    expected = model(features, torch.tensor(.1)).detach().numpy()
    output = project_directory / "serialized.npy"
    code = "\n".join([
        "import sys,numpy as np,torch",
        "from tdn.train import load_checkpoint,model_from_checkpoint",
        "payload=load_checkpoint(sys.argv[1])",
        "model=model_from_checkpoint(payload)",
        "features=torch.linspace(-1,1,len(payload['normalization']['mean'])).reshape(1,-1)",
        "np.save(sys.argv[2],model(features,torch.tensor(.1)).detach().numpy())"])
    result = subprocess.run([sys.executable, "-c", code, str(resumed), str(output)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    np.testing.assert_array_equal(np.load(output), expected)
    last = resumed / "checkpoints" / "last.pt"
    with last.open("r+b") as stream:
        stream.seek(-1, 2)
        value = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([value[0] ^ 1]))
    with pytest.raises(ValueError, match="checksum"):
        load_checkpoint(last)


@pytest.mark.parametrize("windows", [2, 4])
def test_short_rollout_training_and_chunk_weighting(configuration, project_directory, windows):
    config = copy.deepcopy(configuration)
    config["training"].update(max_steps=2, rollout_windows=windows)
    data = project_directory / "data"
    generate_dataset(config, data)
    small = train(config, data, project_directory / "small")
    large_config = copy.deepcopy(config)
    large_config["training"]["microbatch_cells"] = 128
    large = train(large_config, data, project_directory / "large")
    assert all(np.isfinite(point["loss"]) for point in small["history"])
    a = load_checkpoint(project_directory / "small")
    b = load_checkpoint(project_directory / "large")
    for name in a["model_state"]:
        torch.testing.assert_close(a["model_state"][name], b["model_state"][name], atol=3e-6, rtol=3e-5)
    assert a["training_phase"] == f"{windows}_window"
    assert small["committed_cursor"] == large["committed_cursor"] == 2


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="POSIX Slurm warning signal")
def test_real_signal_commits_and_exits_75(configuration, project_directory):
    data = project_directory / "data"
    generate_dataset(configuration, data)
    config = copy.deepcopy(configuration)
    config["training"]["max_steps"] = 5000
    config_path = project_directory / "signal_config.json"
    config_path.write_text(json.dumps(config))
    ready = project_directory / "ready"
    run = project_directory / "signal"
    code = "\n".join([
        "import json,sys,torch",
        "from pathlib import Path",
        "import tdn.train.loop as loop",
        "torch.set_num_threads(1)",
        "original=loop._regression_backward",
        "def wrapped(*args,**kwargs):",
        "    Path(sys.argv[4]).write_text('handler installed')",
        "    return original(*args,**kwargs)",
        "loop._regression_backward=wrapped",
        "loop.train(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]),Path(sys.argv[3]))"])
    child = subprocess.Popen([sys.executable, "-c", code, str(config_path), str(data), str(run), str(ready)],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.monotonic() + 25
    try:
        while not ready.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(.02)
        assert ready.exists(), "Training failed to reach a handler-protected boundary"
        os.kill(child.pid, signal.SIGUSR1)
        stdout, stderr = child.communicate(timeout=25)
        assert child.returncode == 75, stdout + stderr
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate()
    payload = load_checkpoint(run)
    assert payload["status"] == "PAUSED_NEEDS_RESUME"
    assert payload["sampler"]["committed_cursor"] == payload["global_step"]
    assert payload["global_step"] >= 1
    result = json.loads((run / "training_result.json").read_text())
    assert result["exit_code"] == 75
