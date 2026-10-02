"""Actual matched-budget training, evaluation, and failure retention."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import uuid

import numpy as np
import pytest
import yaml

from tdn.analysis.learned_controls import FAMILIES, compare_learned_controls
from tdn.data import generate_dataset
from tdn.data.provenance import canonical_hash
from tdn.train import load_checkpoint


@pytest.fixture
def control_config():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "smoke.yaml").read_text())
    config["problem"]["grid"] = [4, 4]
    config["model"]["width"] = 8
    config["data"].update(train_count=2, validation_count=1, diagnostic_count=1)
    config["training"].update(max_steps=2, validation_every_steps=1,
                               checkpoint_every_steps=1, microbatch_cells=16)
    config["horizons"].update(values=[0.02, 0.04], rollout_time=0.08,
                              evaluation_steps=[0.04], mixed_factors=[0.5, 1.5])
    config["teacher"]["base_substeps"] = 8
    config["validation"]["bootstrap_samples"] = 10
    return config


@pytest.fixture
def control_directory():
    path = Path(__file__).resolve().parents[1] / "results" / "test-work" / uuid.uuid4().hex
    path.mkdir(parents=True)
    yield path
    shutil.rmtree(path)


def test_six_controls_really_train_and_evaluate_identical_design(control_config, control_directory):
    dataset = control_directory / "dataset"
    generate_dataset(control_config, dataset)
    original = copy.deepcopy(control_config)
    result = compare_learned_controls(control_config, dataset, control_directory / "comparison")
    assert control_config == original
    assert result["status"] == "COMPLETE"
    assert result["device"] == "cpu"
    assert [record["family"] for record in result["results"]] == list(FAMILIES)
    assert len(result["accuracy_time_frontier"]) == 6
    saved = json.loads((control_directory / "comparison" / "learned_controls.json").read_text())
    assert saved == result
    for record in result["results"]:
        assert record["status"].startswith("COMPLETE_")
        assert record["training"]["global_step"] == 2
        assert record["trained_head_nonzero_count"] > 0
        assert record["training"]["source_hash"] == result["source_provenance"]["python_source_hash"]
        assert record["training"]["dataset_hash"] == result["dataset_hash"]
        assert record["training"]["split_hash"] == result["split_hash"]
        assert record["normalization_hash"] == result["normalization_hash"]
        checkpoint = load_checkpoint(Path(record["selected_checkpoint"]))
        expected = copy.deepcopy(original)
        expected["model"]["family"] = record["family"]
        assert checkpoint["config"] == expected
        assert record["config_hash"] == canonical_hash(expected)
        assert np.isfinite(record["evaluation"]["summary"]["worst_measured_error"])
        assert record["evaluation"]["summary"]["median_wall_seconds"] > 0
        assert all(np.isfinite(point["loss"]) for point in record["training"]["history"])
        evaluation_json = json.loads(Path(record["evaluation"]["path"]).read_text())
        assert evaluation_json["checkpoint"]["selection"] == record["checkpoint_selection"]

def test_cli_comparison_stage_metadata_does_not_block_control_directory(control_config, control_directory):
    import subprocess,sys
    dataset=control_directory/"dataset"
    generate_dataset(control_config,dataset)
    cfg=control_directory/"config.yaml";cfg.write_text(yaml.safe_dump(control_config))
    run=control_directory/"cli-comparison"
    root=Path(__file__).resolve().parents[1]
    command=[sys.executable,"-m","tdn.cli","compare","--config",str(cfg),"--dataset",str(dataset),"--run-dir",str(run)]
    result=subprocess.run(command,cwd=root,capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stderr
    stage=json.loads((run/"stage.json").read_text())
    assert stage["status"]=="COMPLETED" and stage["actually_ran"]
    summary=json.loads((run/"controls"/"learned_controls.json").read_text())
    assert summary["status"]=="COMPLETE"
    assert len(summary["results"])==6


def test_failed_family_is_retained_and_other_controls_continue(control_config, control_directory, monkeypatch):
    import tdn.analysis.learned_controls as controls
    dataset = control_directory / "dataset"
    generate_dataset(control_config, dataset)
    original_train = controls.train

    def failing_train(configuration, *args, **kwargs):
        if configuration["model"]["family"] == "rational":
            raise FloatingPointError("injected control failure")
        return original_train(configuration, *args, **kwargs)

    monkeypatch.setattr(controls, "train", failing_train)
    result = compare_learned_controls(control_config, dataset, control_directory / "comparison")
    assert result["status"] == "COMPLETE_WITH_FAILURES"
    assert result["failed_families"] == ["rational"]
    rational = next(row for row in result["results"] if row["family"] == "rational")
    assert rational["failure"]["phase"] == "training"
    assert "injected control failure" in rational["failure"]["message"]
    assert result["results"][-1]["family"] == "temporal_mlp"
    assert result["results"][-1]["status"].startswith("COMPLETE_")


def test_infeasible_latest_checkpoint_is_labelled_failure_diagnostic(control_config, control_directory):
    dataset = control_directory / "dataset"
    generate_dataset(control_config, dataset)
    config = copy.deepcopy(control_config)
    config["validation"]["tolerance"] = 1e-14
    result = compare_learned_controls(config, dataset, control_directory / "comparison")
    assert result["status"] == "COMPLETE"
    for record in result["results"]:
        assert record["status"] == "COMPLETE_INFEASIBLE"
        assert record["training"]["best_checkpoint"] is None
        assert record["checkpoint_selection"] == "last_failure_diagnostic_only"
        evaluation = json.loads(Path(record["evaluation"]["path"]).read_text())
        assert evaluation["checkpoint"]["selection"] == "last_failure_diagnostic_only"
        assert evaluation["checkpoint"]["validation_checkpoint_feasible"] is False


@pytest.mark.parametrize("change,device", (({}, "cuda"), ({"purpose": "confirmatory"}, "cpu"),
                                          ({"training": {"max_steps": 65}}, "cpu"),
                                          ({"problem": {"grid": [128, 128]}}, "cpu")))
def test_unapproved_campaigns_block_before_artifact_creation(control_config, control_directory, change, device):
    config = copy.deepcopy(control_config)
    for key, value in change.items():
        if isinstance(value, dict):
            config[key].update(value)
        else:
            config[key] = value
    target = control_directory / "comparison"
    with pytest.raises(ValueError):
        compare_learned_controls(config, control_directory / "absent-dataset", target, device=device)
    assert not target.exists()
