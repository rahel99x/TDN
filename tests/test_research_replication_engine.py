"""Actual bounded replication, split isolation, frozen weights and provenance."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil

import pytest
import torch

from tdn.research import experiment, replication
from tdn.research.protocol import digest, file_digest, verify_artifacts
from tdn.research.replication_protocol import DEFAULT, make_protocol, validate_config


def smoke_protocol():
    return make_protocol(validate_config(DEFAULT, smoke=True), smoke=True, software={}, command=["engine-test"])


def seal(path: Path, protocol: dict) -> None:
    files = {item.relative_to(path).as_posix(): file_digest(item) for item in path.rglob("*")
             if item.is_file() and item.name not in ("manifest.json", "COMPLETED") and ".partial" not in item.name}
    manifest = {"version": protocol["version"], "protocol_sha256": digest(protocol),
                "source_tree_sha256": "test-fixture", "files": files}
    (path / "manifest.json").write_text(json.dumps(manifest))
    (path / "COMPLETED").write_text(file_digest(path / "manifest.json") + "\n")


@pytest.fixture(scope="module")
def actual_replication(tmp_path_factory):
    path = tmp_path_factory.mktemp("actual-replication")
    protocol = smoke_protocol()
    (path / "protocol.json").write_text(json.dumps(protocol))
    calls = {"generate": 0, "normalization": 0, "train": [], "evaluation": [], "samples": {}}
    original_generate, original_normalization = replication.generate_data, replication.training_normalization
    original_train, original_evaluate = replication.train_family, replication.evaluate
    original_loss = experiment.physical_loss
    normalizations = []
    current = []
    def generate(*args, **kwargs):
        calls["generate"] += 1
        return original_generate(*args, **kwargs)
    def normalize(*args, **kwargs):
        calls["normalization"] += 1
        return original_normalization(*args, **kwargs)
    def train(*args, **kwargs):
        normalizations.append(args[4])
        current[:] = [(kwargs["replicate_id"], args[0])]
        value = original_train(*args, **kwargs)
        calls["train"].append((kwargs["replicate_id"], args[0]))
        return value
    def loss(model, parent, h, *args, **kwargs):
        if not kwargs.get("audit", False):
            calls["samples"].setdefault(current[0], []).append((parent["parent_id"], h))
        return original_loss(model, parent, h, *args, **kwargs)
    def evaluate(*args, **kwargs):
        # Every candidate must already have a terminal training record; even
        # another seed's diagnostics cannot affect a later checkpoint.
        assert len(calls["train"]) == 15
        calls["evaluation"].append(kwargs["replication"])
        return original_evaluate(*args, **kwargs)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(replication, "generate_data", generate)
            monkeypatch.setattr(replication, "training_normalization", normalize)
            monkeypatch.setattr(replication, "train_family", train)
            monkeypatch.setattr(replication, "evaluate", evaluate)
            monkeypatch.setattr(experiment, "physical_loss", loss)
            summary = replication.run(protocol, path)
    finally:
        torch.set_num_threads(old_threads)
    assert all(value is normalizations[0] for value in normalizations)
    seal(path, protocol)
    return path, protocol, summary, calls


def test_actual_replication_generates_once_freezes_before_diagnostics_and_seals(actual_replication):
    path, protocol, summary, calls = actual_replication
    assert summary["status"] == "COMPLETED" and summary["device"] == "cpu"
    assert summary["actual_neural_training"] and summary["training_attempted"]
    assert calls["generate"] == calls["normalization"] == 1
    assert len(calls["train"]) == len(summary["training"]) == 15
    assert len(calls["evaluation"]) == summary["completed_block_count"] == 9
    assert all(not record["diagnostics_seen_during_training"] for record in summary["training"])
    assert all(record["status"] in ("COMPLETED", "NUMERICAL_FAILURE") for record in summary["training"])
    manifest = verify_artifacts(path)
    assert len([name for name in manifest["files"] if name.endswith("/frontier.json")]) == 9
    references = json.loads((path / "references.json").read_text())["records"]
    assert len(references) == 75 and all(row["accepted"] for row in references)
    detail = json.loads((path / "replication.json").read_text())
    assert len(detail["endpoint_rows"]) == 15 and len(detail["comparison_rows"]) == 12
    assert detail["diagnostic_parent_count"] == 9
    assert detail["actual_neural_training"] and detail["training_attempted"]
    assert detail["trained_checkpoint_count"] == summary["trained_checkpoint_count"]
    assert detail["selected_checkpoint_count"] == summary["selected_checkpoint_count"]
    assert summary["trained_checkpoint_count"] == sum(row["trained_checkpoint"] for row in detail["endpoint_rows"])
    assert summary["selected_checkpoint_count"] == sum(row["status"] == "COMPLETED" for row in summary["training"])


def test_optimizer_parent_horizon_schedule_is_paired_within_each_seed(actual_replication):
    _, protocol, _, calls = actual_replication
    for replicate in protocol["replicates"]:
        schedules = [calls["samples"][replicate["replicate_id"], family] for family in protocol["config"]["families"]]
        assert len(schedules[0]) == 2
        assert all(schedule == schedules[0] for schedule in schedules)
        assert all(parent.startswith("train-") for parent, _ in schedules[0])


def test_block_outputs_are_bounded_identified_and_keep_shared_parent_pairing(actual_replication):
    path, protocol, _, _ = actual_replication
    parent_sets = {}
    for replicate in protocol["replicates"]:
        for block in range(3):
            root = path / "replicates" / replicate["replicate_id"] / "blocks" / f"block-{block}"
            frontier = json.loads((root / "frontier.json").read_text())
            heldout = json.loads((root / "heldout.json").read_text())
            comparison = json.loads((root / "neural-comparisons.json").read_text())
            assert frontier["replication"] == heldout["replication"] == comparison["replication"] == {**replicate, "diagnostic_block": block}
            parents = {row["parent_id"] for row in frontier["rows"]}
            assert len(parents) == 3
            assert parent_sets.setdefault(block, parents) == parents
            assert comparison["baseline_families"] == ["generic_mlp", "residual_cnn_split", "unet_split", "fno_split"]
            assert len(frontier["rows"]) <= 120 and len(heldout["rows"]) <= 60
            assert all(item.stat().st_size < 1 << 20 for item in root.glob("*.json"))
    assert len(set.union(*parent_sets.values())) == 9


def test_frozen_loader_preserves_all_selected_parameters_and_normalization(actual_replication):
    path, protocol, summary, _ = actual_replication
    models = replication._frozen_models(protocol, path, summary["training"], "cpu")
    for record in summary["training"]:
        if record["status"] != "COMPLETED":
            assert record["family"] not in models[record["replicate_id"]]
            continue
        checkpoint_path = path / "replicates" / record["replicate_id"] / "checkpoints" / f"{record['family']}.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = models[record["replicate_id"]][record["family"]].state_dict()
        assert state.keys() == checkpoint["state_dict"].keys()
        assert all(torch.equal(state[name], value) for name, value in checkpoint["state_dict"].items())
        assert checkpoint["training_seed"] == record["initialization_seed"] == record["training_seed"]
        assert checkpoint["sample_schedule_seed"] == record["sample_schedule_seed"]


def test_frozen_loader_rejects_cross_seed_records_and_wrong_checkpoint_hash(actual_replication):
    path, protocol, summary, _ = actual_replication
    altered = copy.deepcopy(summary["training"])
    altered[0]["training_seed"] += 1
    with pytest.raises(ValueError, match="seed/family plan"):
        replication._frozen_models(protocol, path, altered, "cpu")
    altered = copy.deepcopy(summary["training"])
    next(record for record in altered if record["status"] == "COMPLETED")["checkpoint_sha256"] = "corrupted"
    with pytest.raises(ValueError, match="checkpoint hash"):
        replication._frozen_models(protocol, path, altered, "cpu")


def test_frozen_loader_rejects_different_normalization_buffers(actual_replication, tmp_path):
    path, protocol, summary, _ = actual_replication
    source = tmp_path / "changed-normalization"
    shutil.copytree(path, source)
    records = copy.deepcopy(summary["training"])
    record = next(row for row in records if row["status"] == "COMPLETED")
    checkpoint_path = source / "replicates" / record["replicate_id"] / "checkpoints" / f"{record['family']}.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    key = next(name for name in checkpoint["state_dict"] if name.endswith("feature_mean"))
    checkpoint["state_dict"][key].add_(1)
    experiment.atomic_torch_save(checkpoint, checkpoint_path)
    record["checkpoint_sha256"] = file_digest(checkpoint_path)
    with pytest.raises(ValueError, match="normalization differs"):
        replication._frozen_models(protocol, source, records, "cpu")


def test_public_gpu_api_rejects_initialization_only_source_without_loading_cuda(actual_replication, tmp_path, monkeypatch):
    path, protocol, summary, _ = actual_replication
    source = tmp_path / "initialization-source"
    shutil.copytree(path, source)
    changed = copy.deepcopy(summary)
    for row in changed["training"]:
        if row["status"] == "COMPLETED":
            row.update(selected_step=0, selected_parameters_changed=False)
    experiment.write_json(source / "summary.json", changed)
    seal(source, protocol)
    target = tmp_path / "gpu-target"
    target.mkdir()
    def unexpected_cuda(*args, **kwargs):
        pytest.fail("Initialization-only source reached GPU checkpoint loading")
    monkeypatch.setattr(replication, "_frozen_models", unexpected_cuda)
    with pytest.raises(ValueError, match="same seed"):
        replication.benchmark(protocol, source, target)
    outcome = json.loads((target / "summary.json").read_text())
    assert outcome["status"] == "FAILED" and outcome["device"] == "cuda"
    assert not outcome["training_attempted"] and not outcome["actual_neural_training"]
    assert "training" not in outcome and len(outcome["source_training"]) == 15
    assert not (target / "dataset.pt").exists() and not (target / "COMPLETED").exists()
    detail = json.loads((target / "replication.json").read_text())
    assert not detail["training_attempted"] and not detail["actual_neural_training"]


@pytest.mark.parametrize("corruption", ["protocol", "seed", "block", "hash", "duplicate"])
def test_replication_dataset_rejects_wrong_plan_or_actual_states(actual_replication, corruption):
    path, protocol, _, _ = actual_replication
    data = torch.load(path / "dataset.pt", map_location="cpu", weights_only=True)
    if corruption == "protocol":
        data["protocol_sha256"] = "other"
    elif corruption == "seed":
        data["parents"][0]["seed"] += 2000
    elif corruption == "block":
        next(parent for parent in data["parents"] if parent["split"] == "diagnostic")["diagnostic_block"] = 2
    elif corruption == "hash":
        data["parents"][0]["initial"].add_(.001)
    else:
        data["parents"][0]["initial"] = data["parents"][1]["initial"].clone()
        data["parents"][0]["initial_state_sha256"] = data["parents"][1]["initial_state_sha256"]
    with pytest.raises(ValueError):
        replication._validate_dataset(data, protocol)


def test_interruption_retains_partial_training_and_explicit_missing_endpoints(tmp_path, monkeypatch):
    protocol = smoke_protocol()
    class Stop:
        requested = True
        signal_number = 10
    with pytest.raises(InterruptedError):
        replication.run(protocol, tmp_path, stop=Stop())
    summary = json.loads((tmp_path / "summary.json").read_text())
    detail = json.loads((tmp_path / "replication.json").read_text())
    assert summary["status"] == "INTERRUPTED" and not summary["training_attempted"]
    assert len(summary["training"]) == 15 and all(row["status"] == "NOT_STARTED" for row in summary["training"])
    assert detail["observed_block_result_count"] == 0
    assert all(row["heldout_one_missing"] == row["heldout_one_expected"] for row in detail["endpoint_rows"])
    assert not (tmp_path / "COMPLETED").exists()


def test_checkpoint_counts_distinguish_initialization_and_substantiated_training(tmp_path):
    protocol = smoke_protocol()
    records = replication._training_plan(protocol)
    records[0].update(status="COMPLETED", selected_step=0, selected_parameters_changed=False)
    records[1].update(status="COMPLETED", selected_step=1, selected_parameters_changed=True)
    records[2].update(status="COMPLETED", selected_step=1, selected_parameters_changed=False)
    outcome = replication._write_outcome(protocol, tmp_path, records, [], status="COMPLETED", device="cpu",
        start=__import__("time").monotonic(), training_attempted=True)
    assert outcome["selected_checkpoint_count"] == 3 and outcome["trained_checkpoint_count"] == 1
    assert outcome["trained_checkpoint_count"] == sum(row["trained_checkpoint"] for row in outcome["endpoint_rows"])
    detail = json.loads((tmp_path / "replication.json").read_text())
    assert detail["selected_checkpoint_count"] == 3 and detail["trained_checkpoint_count"] == 1
    assert detail["training_attempted"] and not detail["actual_neural_training"]


def test_mid_block_timeout_retains_actual_partial_observations(actual_replication, tmp_path, monkeypatch):
    path, protocol, summary, _ = actual_replication
    replicate = protocol["replicates"][0]
    source = path / "replicates" / replicate["replicate_id"] / "blocks/block-0"
    frontier = json.loads((source / "frontier.json").read_text())["rows"]
    heldout = json.loads((source / "heldout.json").read_text())["rows"]
    parent = frontier[0]["parent_id"]
    def interrupted_evaluate(models, dataset, protocol, block_dir, budget, device, **kwargs):
        experiment.write_json(block_dir / "frontier.json", {"rows": [row for row in frontier if row["parent_id"] == parent]})
        experiment.write_json(block_dir / "heldout.json", {"rows": [row for row in heldout if row["parent_id"] == parent]})
        raise TimeoutError("Injected bounded interruption after a persisted parent")
    monkeypatch.setattr(replication, "evaluate", interrupted_evaluate)
    dataset = torch.load(path / "dataset.pt", map_location="cpu", weights_only=True)
    blocks = []
    with pytest.raises(TimeoutError):
        replication._evaluate_blocks(protocol, dataset, {replicate["replicate_id"]: {}}, summary["training"],
                                     tmp_path, experiment.Budget(30), "cpu", blocks)
    assert len(blocks) == 1 and blocks[0]["status"] == "PARTIAL"
    assert {row["parent_id"] for row in blocks[0]["frontier"]} == {parent}
    outcome = replication._write_outcome(protocol, tmp_path, summary["training"], blocks, status="INTERRUPTED",
        device="cpu", start=__import__("time").monotonic(), training_attempted=True)
    assert outcome["completed_block_count"] == 0 and outcome["partial_block_count"] == 1
    assert outcome["observed_block_result_count"] == 1
    row = next(row for row in outcome["endpoint_rows"] if row["replicate_id"] == replicate["replicate_id"]
               and row["family"] == "reaction_clock")
    assert row["rollout_missing"] == row["rollout_expected"] - 4
    assert row["heldout_one_missing"] == row["heldout_one_expected"] - 2


def test_validation_failure_provenance_identifies_validation_not_last_training_case(tmp_path, monkeypatch):
    from test_research_experiment import simple_dataset
    import tdn.research
    dataset, geometry = simple_dataset()
    for parent in dataset["parents"]:
        parent["references"] = {experiment.horizon_key(h): {"state": torch.full_like(parent["initial"], .7)} for h in (.02, .04)}
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(.5))
        def set_normalization(self, *_):
            return self
        def forward(self, state, *_):
            return self.scale.expand_as(state)
        def audit_step(self, state, *_):
            result = self.scale.expand_as(state)
            return result, {"injected_invalid": state - 2}
    monkeypatch.setattr(tdn.research, "build_research_model", lambda *args, **kwargs: TinyModel())
    config = {**DEFAULT, "grid": [4, 4], "max_steps": 1, "validation_every": 1,
              "train_horizons": [.02], "heldout_horizons": [.02]}
    model, record = experiment.train_family("tiny", dataset, {"config": config}, tmp_path,
        (torch.zeros(12), torch.ones(12)), experiment.Budget(30), initialization_seed=74011,
        sample_schedule_seed=74012, replicate_id="seed-74011")
    assert model is None and record["status"] == "NUMERICAL_FAILURE"
    assert record["failure_context"] == "checkpoint_selection" and "failure_stages" not in record
    assert [row["step"] for row in record["validation_failures"]] == [0, 1]
    assert all(row["parent_id"].startswith("validation-") and row["h"] == .02 and "injected_invalid" in row["error"]
               for row in record["validation_failures"])
