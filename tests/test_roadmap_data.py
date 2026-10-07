"""Cohort leakage, accepted FP64 teachers and immutable dataset tests."""
import json

import pytest
import torch

from tdn.analysis.roadmap.core import Context
from tdn.analysis.roadmap.data import (
    _validate_cohorts, _worker_count, load_bank, prepare, reference_horizons,
    verify_frozen_training,
)
from tdn.analysis.roadmap.protocol import build_protocol
from tdn.research.experiment import Budget, atomic_torch_save
from tdn.research.protocol import digest, file_digest
from tdn.runtime.metadata import write_json


def small_protocol():
    p = build_protocol("smoke")
    p["parents"] = [next(x for x in p["parents"] if x["split"] == split)
                    for split in ("train", "validation", "calibration", "confirmation")]
    p["train_horizons"] = [0.01]
    p["validation_horizons"] = [0.015]
    p["confirm_schedules"] = [[0.01, 0.02], [0.02, 0.01]]
    p["continuum_parent_ids"] = []
    p["models"] = ["source"]
    p["seeds"] = [74011]
    p["data"]["teacher_workers"] = 1
    return p


def frozen_training(path, p):
    path.mkdir(parents=True)
    checkpoint = path / "checkpoints/source.pt"
    atomic_torch_save({"protocol_sha256": digest(p), "selection_split": "validation", "state_dict": {},
                       "family": "source", "seed": 74011}, checkpoint)
    write_json(path / "training_plan.json", {"plan": "unit test frozen declaration"})
    record = {"family": "source", "seed": 74011, "model_id": "source-seed-74011",
              "checkpoint": "checkpoints/source.pt", "checkpoint_sha256": file_digest(checkpoint)}
    write_json(path / "catalog.json", {"protocol_sha256": digest(p), "records": [record],
        "frozen_before_confirmation": True, "selection_split": "validation",
        "training_plan_sha256": file_digest(path / "training_plan.json")})
    write_json(path / "freeze.json", {"protocol_sha256": digest(p), "catalog_sha256": file_digest(path / "catalog.json"),
        "checkpoint_hashes": {record["model_id"]: record["checkpoint_sha256"]}})


def context(p, stage, path, prerequisites=None):
    return Context(p, stage, path, prerequisites or {}, "cpu", Budget(90))


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_real_banks_keep_confirmation_uncreated_and_same_fields_on_grids(tmp_path):
    p = small_protocol()
    output = tmp_path / "prepare"
    ctx = context(p, "prepare", output)
    summary = prepare(ctx)
    assert summary["parents"] == {"train": 1, "validation": 1, "calibration": 1}
    assert not (output / "confirmation.pt").exists()
    reader = context(p, "train", tmp_path / "train-reader", {"prepare": output})
    bank = load_bank(reader, "calibration")
    parent = bank[0]
    torch.testing.assert_close(parent["states"]["8"], parent["states"]["16"][..., ::2, ::2], rtol=0, atol=0)
    assert all(r["accepted"] for r in parent["references"].values())
    assert {row["horizon"] for row in parent["references"].values()} == {0.01, 0.02, 0.03}
    assert len(ctx.rows) == 3
    assert all(row["metrics"]["parameters"] == 0 for row in ctx.rows)
    with pytest.raises(ValueError, match="Confirmation access forbidden"):
        load_bank(reader, "confirmation")
    with pytest.raises(FileExistsError):
        prepare(ctx)


def test_confirmation_generation_and_loading_require_same_frozen_models(tmp_path):
    p = small_protocol()
    train = tmp_path / "train"
    frozen_training(train, p)
    generator = context(p, "confirm_prepare", tmp_path / "confirm_prepare", {"train": train})
    prepare(generator, confirmation=True)
    reader = context(p, "confirm", tmp_path / "confirm", {"train": train, "confirm_prepare": generator.path})
    assert len(load_bank(reader, "confirmation")) == 1
    with (train / "checkpoints/source.pt").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="Frozen checkpoint bytes changed"):
        load_bank(reader, "confirmation")


def test_confirmation_freeze_check_precedes_any_teacher(tmp_path, monkeypatch):
    p = small_protocol()
    train = tmp_path / "train"
    frozen_training(train, p)
    freeze = json.loads((train / "freeze.json").read_text())
    freeze["catalog_sha256"] = "0" * 64
    write_json(train / "freeze.json", freeze)
    calls = []
    monkeypatch.setattr("tdn.analysis.roadmap.data._parent", lambda *args: calls.append(True))
    ctx = context(p, "confirm_prepare", tmp_path / "confirm_prepare", {"train": train})
    with pytest.raises(ValueError, match="must freeze"):
        prepare(ctx, confirmation=True)
    assert calls == []


def test_dataset_tampering_is_rejected_before_tensor_use(tmp_path):
    p = small_protocol()
    ctx = context(p, "prepare", tmp_path / "prepare")
    prepare(ctx)
    reader = context(p, "train", tmp_path / "train", {"prepare": ctx.path})
    with (ctx.path / "calibration.pt").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="artifact digest differs"):
        load_bank(reader, "train")


def test_resealed_parent_parameters_cannot_override_protocol(tmp_path):
    p = small_protocol()
    ctx = context(p, "prepare", tmp_path / "prepare")
    prepare(ctx)
    data = torch.load(ctx.path / "train.pt", weights_only=True)
    data["parents"][0]["kappa"] *= 2
    atomic_torch_save(data, ctx.path / "train.pt")
    manifest = json.loads((ctx.path / "data_manifest.json").read_text())
    manifest["artifacts"]["train.pt"] = file_digest(ctx.path / "train.pt")
    write_json(ctx.path / "data_manifest.json", manifest)
    reader = context(p, "train", tmp_path / "train", {"prepare": ctx.path})
    with pytest.raises(ValueError, match="Parent parameters differ"):
        load_bank(reader, "train")


def test_cross_split_phase_cluster_reuse_is_rejected():
    p = small_protocol()
    p["parents"][1]["field_cluster"] = p["parents"][0]["field_cluster"]
    with pytest.raises(ValueError, match="split-disjoint"):
        _validate_cohorts(p)


def test_intermediate_horizons_are_explicit_and_endpoint_only_continuum():
    p = small_protocol()
    assert reference_horizons(p, "confirmation") == [0.01, 0.02, 0.03]
    assert reference_horizons(p, "confirmation", include_intermediate=False) == [0.03]
    assert reference_horizons(p, "train") == [0.01, 0.02]


def test_worker_count_respects_allocation_and_eight_core_cap(tmp_path, monkeypatch):
    p = small_protocol()
    p["data"]["teacher_workers"] = 16
    ctx = context(p, "prepare", tmp_path)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "3")
    assert _worker_count(ctx, 100) == 3
    monkeypatch.delenv("SLURM_CPUS_PER_TASK")
    assert _worker_count(ctx, 100) == 8


def test_freeze_requires_every_declared_family_seed(tmp_path):
    p = small_protocol()
    p["models"].append("rank1")
    frozen_training(tmp_path / "train", p)
    ctx = context(p, "confirm_prepare", tmp_path / "confirm_prepare", {"train": tmp_path / "train"})
    with pytest.raises(ValueError, match="Every declared family"):
        verify_frozen_training(ctx)


def test_spawn_worker_transport_and_partial_artifacts(tmp_path):
    p = small_protocol()
    p["data"]["teacher_workers"] = 2
    ctx = context(p, "prepare", tmp_path / "prepare")
    summary = prepare(ctx)
    assert summary["teacher_workers"] == 2
    assert not list(ctx.path.glob("*.partial.pt"))
    reader = context(p, "train", tmp_path / "train", {"prepare": ctx.path})
    assert len(load_bank(reader, "validation")) == 1
