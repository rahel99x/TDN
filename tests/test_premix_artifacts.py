"""Independent stage-envelope, dependency-closure and coverage regressions."""
from __future__ import annotations

import json

import pytest

from tdn.analysis.premix import artifacts as a
from tdn.runtime.metadata import write_json

SOURCE = "a" * 64


def evidence(root, stage="accuracy", *, device="cpu"):
    root.mkdir(parents=True, exist_ok=True)
    protocol = {"profile": "smoke"}
    summary = {"profile": "smoke", "status": "COMPLETED"}
    contents = {}
    if stage in ("accuracy", "scaling"):
        expected = {table: [table + "/1"] for table in a._TABLE_IDS}
        protocol.update(config={}, plans={"expected_ids": expected})
        summary.update(panel=stage, coverage={table: {"expected": 1, "reported": 1} for table in expected})
        contents["config.json"] = {}
        contents["metrics.json"] = {"status": "COMPLETED", "profile": "smoke", "panel": stage,
            **{table: [{identity: expected[table][0]}] for table, identity in a._TABLE_IDS.items()}}
        (root / "summary.txt").write_text("Scientific summary fixture\n")
    elif stage == "prepare":
        protocol["parents"] = [{"parent_id": "train-1", "split": "train", "horizons": [.1]}]
        summary.update(stage="dataset", device=device)
        (root / "dataset.pt").write_bytes(b"fixture; no torch deserialization needed for seal checks")
        contents["references.json"] = {"rows": [{"parent_id": "train-1", "horizon": .1, "accepted": True}]}
        contents["normalization.json"] = {"parent_ids": ["train-1"], "split": "train", "protocol_sha256": a.digest(protocol)}
    else:
        protocol.update(seeds=[11], families=["premix"], classical=["strang"],
            parents=[{"parent_id": "p0", "split": "diagnostic", "regime": "smooth"}],
            step_counts=[1, 2], long_step_counts=[2], targets=[.1])
        summary.update(stage="neural", device=device)
        checkpoint = root / "checkpoints" / "seed-11-premix.pt"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"checkpoint fixture")
        contents["training.json"] = {"rows": [{"seed": 11, "family": "premix", "status": "COMPLETED",
            "checkpoint_sha256": a.file_digest(checkpoint)}]}
        contents["candidates.json"] = {"rows": [{"parent_id": "p0", "seed": seed, "family": family, "steps": steps}
            for seed, family in ((11, "premix"), (None, "strang")) for steps in (1, 2)]}
        contents["frontiers.json"] = {"rows": [{"parent_id": "p0", "seed": seed, "family": family, "norm": norm, "target": .1}
            for seed, family in ((11, "premix"), (None, "strang")) for norm in ("rms", "max")]}
        contents["checkpoint_freeze.json"] = {"protocol_sha256": a.digest(protocol), "selection_split": "validation",
            "all_training_completed_before_diagnostics": True, "artifacts": {"checkpoints/seed-11-premix.pt": a.file_digest(checkpoint)}}
        contents["dataset_source.json"] = {}
        contents["memory.json"] = {}
    contents.update({"protocol.json": protocol, "summary.json": summary,
        "execution.json": {"stage": stage, "profile": "smoke", "device": device,
            "execution_mode": "carc" if device == "cuda" else "local-cpu", "protocol_sha256": a.digest(protocol),
            "software": {"source_tree_sha256": SOURCE}}})
    for name, value in contents.items():
        write_json(root / name, value)
    refresh_inner(root, stage)
    return root


def refresh_inner(root, stage):
    if stage not in ("prepare", "neural"):
        return
    name = "dataset_manifest.json" if stage == "prepare" else "neural_manifest.json"
    names = (a._REQUIRED[stage] - {"execution.json", name}) | {
        path.relative_to(root).as_posix() for path in root.glob("checkpoints/*.pt")}
    write_json(root / name, {"protocol_sha256": a.digest(json.loads((root / "protocol.json").read_text())),
        "artifacts": {filename: a.file_digest(root / filename) for filename in names}})


def mutate(root, filename, change):
    value = json.loads((root / filename).read_text())
    change(value)
    write_json(root / filename, value)


def seal(root, stage="accuracy"):
    return a.seal_stage(root, stage=stage, profile="smoke", source_tree_sha256=SOURCE)


@pytest.mark.parametrize("stage", a.STAGES)
def test_complete_stage_envelope_and_inner_closure_verify(tmp_path, stage):
    root = evidence(tmp_path, stage)
    manifest = seal(root, stage)
    assert a.verify_stage(root, stage=stage, profile="smoke", source_tree_sha256=SOURCE) == manifest


@pytest.mark.parametrize("field,value", [("stage", "prepare"), ("profile", "full"),
    ("protocol_sha256", "b" * 64), ("device", "cuda"), ("execution_mode", "desktop"),
    ("software", {"source_tree_sha256": "b" * 64})])
def test_execution_mismatch_cannot_be_sealed(tmp_path, field, value):
    evidence(tmp_path)
    mutate(tmp_path, "execution.json", lambda item: item.update({field: value}))
    with pytest.raises(ValueError, match="execution"):
        seal(tmp_path)
    assert not (tmp_path / "COMPLETED").exists()


def test_cpu_and_carc_cuda_neural_are_distinct_valid_envelopes(tmp_path):
    for device in ("cpu", "cuda"):
        root = evidence(tmp_path / device, "neural", device=device)
        seal(root, "neural")
        assert a.verify_stage(root)["stage"] == "neural"


@pytest.mark.parametrize("stage", a.STAGES)
def test_desktop_slurm_envelope_requires_its_own_profile_fingerprint(tmp_path, stage):
    root = evidence(tmp_path, stage, device="cuda" if stage == "neural" else "cpu")
    mutate(root, "execution.json", lambda item: item.update(execution_mode="desktop-slurm"))
    with pytest.raises(ValueError, match="profile fingerprint"):
        seal(root, stage)
    mutate(root, "execution.json", lambda item: item.update(slurm_profile_sha256="d" * 64))
    seal(root, stage)
    assert a.verify_stage(root)["stage"] == stage


@pytest.mark.parametrize("stage,filename", [("accuracy", "config.json"), ("scaling", "metrics.json"),
    ("prepare", "normalization.json"), ("neural", "checkpoint_freeze.json")])
def test_mandatory_scientific_files_cannot_be_omitted(tmp_path, stage, filename):
    evidence(tmp_path, stage)
    (tmp_path / filename).unlink()
    with pytest.raises(ValueError, match="inventory"):
        seal(tmp_path, stage)


def test_pending_dataset_partial_pt_cannot_be_sealed(tmp_path):
    evidence(tmp_path, "prepare")
    (tmp_path / "dataset.partial.pt").write_bytes(b"unfinished")
    with pytest.raises(ValueError, match="Unfinished"):
        seal(tmp_path, "prepare")


@pytest.mark.parametrize("table", a._TABLE_IDS)
def test_declared_numerical_coverage_is_checked_independently(tmp_path, table):
    evidence(tmp_path)
    mutate(tmp_path, "metrics.json", lambda item: item[table].clear())
    with pytest.raises(ValueError, match="coverage"):
        seal(tmp_path)


@pytest.mark.parametrize("filename", ["training.json", "candidates.json", "frontiers.json"])
def test_neural_declared_coverage_cannot_be_truncated(tmp_path, filename):
    evidence(tmp_path, "neural")
    mutate(tmp_path, filename, lambda item: item["rows"].pop())
    # Recompute only the inner dependency hashes: coverage still has to agree
    # with the fixed protocol, not merely with a hash of the current bytes.
    refresh_inner(tmp_path, "neural")
    with pytest.raises(ValueError, match="coverage|frozen set"):
        seal(tmp_path, "neural")


def test_inner_dependency_hash_mismatch_prevents_sealing(tmp_path):
    evidence(tmp_path, "prepare")
    mutate(tmp_path, "normalization.json", lambda item: item.update(extra="changed after inner sealing"))
    with pytest.raises(ValueError, match="inner artifact inventory"):
        seal(tmp_path, "prepare")


def test_unaccepted_reference_and_diagnostic_normalization_are_rejected(tmp_path):
    root = evidence(tmp_path / "reference", "prepare")
    mutate(root, "references.json", lambda item: item["rows"][0].update(accepted=False))
    refresh_inner(root, "prepare")
    with pytest.raises(ValueError, match="unaccepted"):
        seal(root, "prepare")
    root = evidence(tmp_path / "normalization", "prepare")
    mutate(root, "normalization.json", lambda item: item.update(split="diagnostic"))
    refresh_inner(root, "prepare")
    with pytest.raises(ValueError, match="training-only"):
        seal(root, "prepare")


def test_strict_manifest_version_and_required_inventory_on_verification(tmp_path):
    evidence(tmp_path)
    seal(tmp_path)
    mutate(tmp_path, "manifest.json", lambda item: item.update(version=True))
    (tmp_path / "COMPLETED").write_text(a.file_digest(tmp_path / "manifest.json"))
    with pytest.raises(ValueError, match="Unsupported"):
        a.verify_stage(tmp_path)


def test_parent_directory_symlink_inside_root_is_not_an_artifact(tmp_path):
    evidence(tmp_path)
    (tmp_path / "actual").mkdir()
    (tmp_path / "actual/file.json").write_text("{}")
    (tmp_path / "linked").symlink_to(tmp_path / "actual", target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe"):
        a._artifact(tmp_path, "linked/file.json")


def test_nonfinite_json_does_not_become_scientific_evidence(tmp_path):
    evidence(tmp_path)
    (tmp_path / "metrics.json").write_text('{"stat": NaN}')
    with pytest.raises(ValueError):
        seal(tmp_path)
