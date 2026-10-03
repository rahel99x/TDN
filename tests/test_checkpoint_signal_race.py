"""A stop during checkpoint publication must serialize a resumable status."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import numpy as np
import pytest
import torch

from tdn.config import load_config
from tdn.data import generate_dataset
from tdn.data.provenance import file_hash
from tdn.train import load_checkpoint, train
import tdn.train.loop as loop


pytestmark = pytest.mark.skipif(
    not hasattr(signal, "SIGUSR1"), reason="POSIX Slurm warning signal"
)


def _assert_identical(actual, expected):
    """Compare optimizer/RNG trees without accepting numerical drift."""
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_identical(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for value, reference in zip(actual, expected):
            _assert_identical(value, reference)
    else:
        assert actual == expected


@pytest.mark.parametrize("arrival", ["payload_construction", "after_publication"])
@pytest.mark.parametrize(
    ("boundary", "interrupted_step", "budget", "published_status"),
    [("periodic", 1, None, "RUNNING"),
     ("budget", 1, 1, "PAUSED_BUDGET"),
     ("final", 3, None, "COMPLETE")],
)
def test_signal_during_checkpoint_pauses_and_replays(
        tmp_path, monkeypatch, arrival, boundary, interrupted_step, budget, published_status):
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml")
    config["problem"]["grid"] = [4, 4]
    config["data"].update(train_count=2, validation_count=1, diagnostic_count=1)
    config["training"].update(
        max_steps=3, checkpoint_every_steps=1, validation_every_steps=3
    )
    data = tmp_path / "data"
    generate_dataset(config, data)
    run = tmp_path / "interrupted"

    # A completed prefix supplies the exact committed state expected
    # at the interruption, independently of the signal publication path.
    prefix = tmp_path / "prefix"
    train(config, data, prefix, max_steps=interrupted_step)
    expected_prefix = load_checkpoint(prefix)
    injected = False
    first_published = None
    first_best_hash = None

    with monkeypatch.context() as patch:
        original_save = loop.save_checkpoint

        def stop_after_publication(run_dir, payload, *, best=False):
            nonlocal injected, first_published, first_best_hash
            result = original_save(run_dir, payload, best=best)
            if first_published is None and payload["global_step"] == interrupted_step:
                first_published = load_checkpoint(run_dir)
                selected = run_dir / "checkpoints" / "best.pt"
                if selected.exists():
                    first_best_hash = file_hash(selected)
                if arrival == "after_publication":
                    assert first_published["status"] == published_status
                    injected = True
                    os.kill(os.getpid(), signal.SIGUSR1)
            return result

        patch.setattr(loop, "save_checkpoint", stop_after_publication)
        if arrival == "payload_construction":
            original_report = loop._MemoryBudget.report
            report_count = 0

            def stop_during_payload(memory):
                nonlocal injected, report_count
                report_count += 1
                if not injected and report_count == interrupted_step:
                    injected = True
                    os.kill(os.getpid(), signal.SIGUSR1)
                return original_report(memory)

            patch.setattr(loop._MemoryBudget, "report", stop_during_payload)

        with pytest.raises(SystemExit) as stopped:
            train(config, data, run, max_steps=budget)
        assert stopped.value.code == 75

    assert injected
    paused = load_checkpoint(run)
    latest = json.loads((run / "checkpoints" / "latest.json").read_text())
    result = json.loads((run / "training_result.json").read_text())
    assert paused["status"] == latest["status"] == result["status"] == "PAUSED_NEEDS_RESUME"
    assert latest["sha256"] == file_hash(run / "checkpoints" / "last.pt")
    assert paused["global_step"] == latest["global_step"] == result["global_step"] == interrupted_step
    assert paused["sampler"]["committed_cursor"] == interrupted_step
    assert paused["scheduler_state"]["last_epoch"] == interrupted_step
    assert all(state["step"].item() == interrupted_step
               for state in paused["optimizer_state"]["state"].values())
    assert result["exit_code"] == 75
    assert result["signal"] == signal.SIGUSR1
    assert len(paused["history"]) == interrupted_step
    if boundary != "final":
        assert paused["validation"] is None
        assert not (run / "checkpoints" / "best.pt").exists()
    for key in ("model_state", "optimizer_state", "scheduler_state", "sampler", "history", "rng_state"):
        _assert_identical(paused[key], expected_prefix[key])
        _assert_identical(paused[key], first_published[key])
    if first_best_hash is not None:
        assert latest["best"]["sha256"] == first_best_hash
        assert file_hash(run / "checkpoints" / "best.pt") == first_best_hash
        selected = load_checkpoint(run / "checkpoints" / "best.pt")
        assert selected["validation"]["feasible"]
        _assert_identical(selected["model_state"], paused["model_state"])

    if boundary == "final":
        assert first_best_hash is not None
        resumed = train(config, data, run, resume=run / "checkpoints")
        complete = load_checkpoint(run)
        latest = json.loads((run / "checkpoints" / "latest.json").read_text())
        result = json.loads((run / "training_result.json").read_text())
        assert resumed["status"] == complete["status"] == latest["status"] == result["status"] == "COMPLETE"
        assert resumed["global_step"] == complete["global_step"] == latest["global_step"] == 3
        assert latest["sha256"] == file_hash(run / "checkpoints" / "last.pt")
        assert latest["best"]["sha256"] == first_best_hash
        assert file_hash(run / "checkpoints" / "best.pt") == first_best_hash
        for key in ("model_state", "optimizer_state", "scheduler_state", "sampler",
                    "history", "rng_state", "validation"):
            _assert_identical(complete[key], paused[key])
        assert resumed["validation"] == paused["validation"]
        return

    # The same configured three-step trajectory must finish its remaining two
    # updates exactly, without changing the budget or regenerating the data.
    if boundary == "periodic" and arrival == "after_publication":
        config_path = tmp_path / "resume-config.json"
        config_path.write_text(json.dumps(config))
        code = "\n".join([
            "import json,sys,torch",
            "from pathlib import Path",
            "from tdn.train import train",
            "torch.set_num_threads(int(sys.argv[4]))",
            "torch.set_num_interop_threads(int(sys.argv[5]))",
            "run=Path(sys.argv[3])",
            "train(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]),run,resume=run/'checkpoints')",
        ])
        child = subprocess.run(
            [sys.executable, "-c", code, str(config_path), str(data), str(run),
             str(torch.get_num_threads()), str(torch.get_num_interop_threads())],
            capture_output=True, text=True, timeout=60, check=False,
        )
        assert child.returncode == 0, child.stdout + child.stderr
        resumed = json.loads((run / "training_result.json").read_text())
    else:
        resumed = train(config, data, run, resume=run / "checkpoints")
    uninterrupted_run = tmp_path / "uninterrupted"
    uninterrupted = train(config, data, uninterrupted_run)
    assert resumed["status"] == uninterrupted["status"] == "COMPLETE"
    assert resumed["global_step"] == uninterrupted["global_step"] == 3
    assert resumed["history"] == uninterrupted["history"]
    assert resumed["validation"] == uninterrupted["validation"]
    final = load_checkpoint(run)
    reference = load_checkpoint(uninterrupted_run)
    for key in ("model_state", "optimizer_state", "scheduler_state", "sampler", "history", "rng_state"):
        _assert_identical(final[key], reference[key])
    latest = json.loads((run / "checkpoints" / "latest.json").read_text())
    if final["validation"]["feasible"]:
        assert latest["best"]["sha256"] == file_hash(run / "checkpoints" / "best.pt")
        selected = load_checkpoint(run / "checkpoints" / "best.pt")
        assert selected["validation"]["feasible"]
        assert selected["global_step"] == 3
        _assert_identical(selected["model_state"], final["model_state"])
