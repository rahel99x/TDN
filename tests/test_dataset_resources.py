"""Dataset readers release file mappings before returning or propagating errors."""
from __future__ import annotations

from pathlib import Path
import signal

import numpy as np
import pytest
import yaml

from tdn.analysis.workflow import evaluate
from tdn.data import DatasetStore, generate_dataset
from tdn.train import load_checkpoint, train


@pytest.fixture
def tiny_dataset(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "smoke.yaml").read_text())
    config["problem"]["grid"] = [4, 4]
    config["model"]["width"] = 8
    config["data"].update(train_count=2, validation_count=1, diagnostic_count=1)
    config["training"].update(max_steps=1, validation_every_steps=1,
                              checkpoint_every_steps=1, microbatch_cells=16)
    config["horizons"].update(values=[0.02, 0.04], rollout_time=0.08,
                             evaluation_steps=[0.04], mixed_factors=[0.5, 1.5])
    config["teacher"]["base_substeps"] = 8
    config["validation"]["bootstrap_samples"] = 10
    path = tmp_path / "dataset"
    generate_dataset(config, path)
    return config, path


@pytest.fixture
def mapped_files(monkeypatch):
    """Keep mappings alive so garbage collection cannot make the test pass."""
    mappings = []
    original = DatasetStore.arrays

    def tracked_arrays(self, split):
        arrays = original(self, split)
        for array in arrays.values():
            mapping = array._mmap
            if not any(existing is mapping for existing in mappings):
                mappings.append(mapping)
        return arrays

    monkeypatch.setattr(DatasetStore, "arrays", tracked_arrays)
    return mappings


def assert_mappings_closed(mappings):
    assert mappings, "The operation must actually open dataset mappings"
    assert all(mapping.closed for mapping in mappings)


def test_dataset_context_closes_on_error_and_samples_own_their_data(tiny_dataset, mapped_files):
    _, path = tiny_dataset
    with pytest.raises(RuntimeError, match="injected reader failure") as saved_error:
        with DatasetStore(path) as store:
            sample = store.sample("train", 0)
            arrays = store.arrays("train")
            expected = sample["state"].copy()
            assert not np.shares_memory(sample["state"], arrays["states"])
            assert not np.shares_memory(sample["teacher"], arrays["teacher"])
            raise RuntimeError("injected reader failure")
    assert saved_error.value.__traceback__ is not None
    assert_mappings_closed(mapped_files)
    np.testing.assert_array_equal(sample["state"], expected)
    store.close()  # Closing an already closed reader is safe.
    assert_mappings_closed(mapped_files)


def test_training_closes_mappings_before_return(tiny_dataset, tmp_path, mapped_files):
    config, path = tiny_dataset
    result = train(config, path, tmp_path / "train")
    assert result["status"] == "COMPLETE"
    assert result["global_step"] == 1
    assert_mappings_closed(mapped_files)


def test_partial_shard_load_failure_releases_already_opened_mappings(tiny_dataset, monkeypatch):
    _, path = tiny_dataset
    store = DatasetStore(path)
    mappings = []
    original_load = np.load

    def fail_second_load(*args, **kwargs):
        if mappings:
            raise OSError("injected second-shard read failure")
        array = original_load(*args, **kwargs)
        mappings.append(array._mmap)
        return array

    monkeypatch.setattr(np, "load", fail_second_load)
    with pytest.raises(OSError, match="injected second-shard read failure") as saved_error:
        store.arrays("train")
    assert saved_error.value.__traceback__ is not None
    assert_mappings_closed(mappings)
    with pytest.raises(RuntimeError, match="closed"):
        store.sample("train", 0)
    store.close()


@pytest.mark.parametrize("pause", [False, True], ids=["error", "safe-boundary-pause"])
def test_training_closes_mappings_with_live_exception_traceback(
        tiny_dataset, tmp_path, mapped_files, monkeypatch, pause):
    import tdn.train.loop as loop

    config, path = tiny_dataset
    original = loop._regression_backward

    def interrupted_backward(*args, **kwargs):
        if not pause:
            raise RuntimeError("injected training failure")
        result = original(*args, **kwargs)
        signal.raise_signal(signal.SIGTERM)
        return result

    monkeypatch.setattr(loop, "_regression_backward", interrupted_backward)
    expected_error = SystemExit if pause else RuntimeError
    with pytest.raises(expected_error) as saved_error:
        train(config, path, tmp_path / "train")
    assert saved_error.value.__traceback__ is not None
    assert_mappings_closed(mapped_files)
    if pause:
        assert saved_error.value.code == 75
        payload = load_checkpoint(tmp_path / "train")
        assert payload["status"] == "PAUSED_NEEDS_RESUME"
        assert payload["global_step"] == 1
        assert payload["sampler"]["committed_cursor"] == 1
    else:
        assert str(saved_error.value) == "injected training failure"


def test_evaluation_closes_mappings_before_return(tiny_dataset, tmp_path, mapped_files):
    config, path = tiny_dataset
    result = evaluate(config, path, None, tmp_path / "evaluate")
    assert result["stage"] == "evaluate"
    assert_mappings_closed(mapped_files)


def test_evaluation_closes_mappings_with_live_exception_traceback(
        tiny_dataset, tmp_path, mapped_files, monkeypatch):
    import tdn.analysis.workflow as workflow

    config, path = tiny_dataset

    def failing_candidates(store, split, *args, **kwargs):
        store.sample(split, 0)
        raise RuntimeError("injected evaluation failure")

    monkeypatch.setattr(workflow, "_candidate_records", failing_candidates)
    with pytest.raises(RuntimeError, match="injected evaluation failure") as saved_error:
        evaluate(config, path, None, tmp_path / "evaluate")
    assert saved_error.value.__traceback__ is not None
    assert_mappings_closed(mapped_files)
