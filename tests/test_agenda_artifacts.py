"""Seal genuine numerical structural evidence and reject incomplete closure."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tdn.analysis.agenda.artifacts import digest, file_digest, seal_stage, verify_stage
from tdn.analysis.agenda.protocol import build_protocol
from tdn.runtime.metadata import write_json


SOURCE = "1" * 64


@pytest.fixture(scope="module")
def genuine_structural_evidence(tmp_path_factory):
    from tdn.analysis.agenda.structural import run
    root = tmp_path_factory.mktemp("agenda-real-structure") / "structure"
    protocol = build_protocol("smoke")
    write_json(root / "protocol.json", protocol)
    summary = run(protocol, root)
    assert summary["status"] == "COMPLETED", summary
    write_json(root / "execution.json", {"stage": "structure", "profile": "smoke", "protocol_sha256": digest(protocol),
        "execution_mode": "local-cpu", "device": "cpu", "software": {"source_tree_sha256": SOURCE}, "prerequisites": {}})
    seal_stage(root, stage="structure", profile="smoke", source_tree_sha256=SOURCE)
    return root


@pytest.fixture
def evidence(genuine_structural_evidence, tmp_path):
    return Path(shutil.copytree(genuine_structural_evidence, tmp_path / "structure"))


@pytest.fixture(scope="module")
def genuine_prepared_evidence(genuine_structural_evidence):
    from tdn.analysis.agenda.engine import prepare
    root = genuine_structural_evidence.parent / "prepare"
    protocol = build_protocol("smoke")
    write_json(root / "execution.json", {"stage": "prepare", "profile": "smoke", "protocol_sha256": digest(protocol),
        "execution_mode": "local-cpu", "device": "cpu", "software": {"source_tree_sha256": SOURCE},
        "prerequisites": {"structure": {"run_dir": str(genuine_structural_evidence),
                                         "manifest_sha256": file_digest(genuine_structural_evidence / "manifest.json")}}})
    result = prepare(protocol, root)
    assert result["status"] == "COMPLETED", result
    seal_stage(root, stage="prepare", profile="smoke", source_tree_sha256=SOURCE)
    return root


@pytest.fixture
def prepared(genuine_prepared_evidence, tmp_path):
    destination = tmp_path / "collected"
    destination.mkdir()
    shutil.copytree(genuine_prepared_evidence.parent / "structure", destination / "structure")
    return Path(shutil.copytree(genuine_prepared_evidence, destination / "prepare"))


def _unseal(path):
    (path / "manifest.json").unlink()
    (path / "COMPLETED").unlink()


def _rewrite_cases(path, transform):
    rows = json.loads((path / "rows.json").read_text())
    transform(rows["rows"])
    write_json(path / "rows.json", rows)
    cases = json.loads((path / "cases.json").read_text())
    cases["rows"] = rows["rows"]
    write_json(path / "cases.json", cases)


def test_genuine_completed_structural_response_has_strict_hash_and_size_closure(evidence):
    manifest = verify_stage(evidence, stage="structure", profile="smoke", source_tree_sha256=SOURCE)
    assert manifest["benchmark_suite"] == "agenda"
    for name, record in manifest["files"].items():
        assert record == {"sha256": file_digest(evidence / name), "bytes": (evidence / name).stat().st_size}
    assert (evidence / "COMPLETED").read_text().strip() == file_digest(evidence / "manifest.json")
    with pytest.raises(ValueError, match="Preserve"):
        seal_stage(evidence, stage="structure", profile="smoke", source_tree_sha256=SOURCE)


def test_local_full_evidence_cannot_be_resealed_as_native_confirmation(evidence):
    _unseal(evidence)
    protocol = build_protocol("full")
    write_json(evidence / "protocol.json", protocol)
    execution = json.loads((evidence / "execution.json").read_text())
    execution.update(profile="full", protocol_sha256=digest(protocol))
    write_json(evidence / "execution.json", execution)
    with pytest.raises(ValueError, match="mode/device"):
        seal_stage(evidence, stage="structure", profile="full", source_tree_sha256=SOURCE)


@pytest.mark.parametrize("operation", ("alter", "missing", "extra", "partial", "symlink"))
def test_verify_rejects_altered_missing_extra_unfinished_and_symlink_science(evidence, operation):
    target = evidence / "fields.npz"
    if operation == "alter":
        target.write_bytes(target.read_bytes() + b"altered")
    elif operation == "missing":
        target.unlink()
    elif operation == "extra":
        (evidence / "unsealed-result.json").write_text("{}")
    elif operation == "partial":
        (evidence / "unfinished.partial").write_text("incomplete")
    else:
        (evidence / "escape.json").symlink_to(evidence.parent / "outside.json")
    with pytest.raises(ValueError):
        verify_stage(evidence)


def test_mutable_lifecycle_is_separate_and_tower_cannot_enter_science(evidence):
    (evidence / "stage.json").write_text('{"status":"COMPLETED"}')
    verify_stage(evidence)
    (evidence / "tower").mkdir()
    with pytest.raises(ValueError, match="Tower"):
        verify_stage(evidence)


@pytest.mark.parametrize("kwargs", ({"stage": "prepare"}, {"profile": "full"}, {"source_tree_sha256": "2" * 64}))
def test_verifier_binds_expected_stage_profile_and_source(evidence, kwargs):
    with pytest.raises(ValueError, match="differs"):
        verify_stage(evidence, **kwargs)


@pytest.mark.parametrize("mutation", ("remove", "duplicate", "relabel_required", "required_fail"))
def test_resealing_cannot_hide_required_failure_or_declared_coverage(evidence, mutation):
    _unseal(evidence)
    def mutate(rows):
        required = next(row for row in rows if row["required"])
        if mutation == "remove":
            rows.remove(required)
        elif mutation == "duplicate":
            rows.append(deepcopy(required))
        elif mutation == "relabel_required":
            required["required"] = False
        else:
            required["outcome"] = "FAIL"
    _rewrite_cases(evidence, mutate)
    with pytest.raises(ValueError):
        seal_stage(evidence, stage="structure", profile="smoke", source_tree_sha256=SOURCE)
    assert not (evidence / "manifest.json").exists()
    assert not (evidence / "COMPLETED").exists()


@pytest.mark.parametrize("mutation", ("source", "profile", "foreign_protocol", "forged_prior", "incomplete"))
def test_resealing_rejects_foreign_execution_protocol_or_lineage(evidence, mutation):
    _unseal(evidence)
    execution = json.loads((evidence / "execution.json").read_text())
    if mutation == "source":
        execution["software"]["source_tree_sha256"] = "2" * 64
    elif mutation == "profile":
        execution["profile"] = "full"
    elif mutation == "foreign_protocol":
        protocol = json.loads((evidence / "protocol.json").read_text())
        protocol["train_horizons"] = [.1]
        write_json(evidence / "protocol.json", protocol)
        execution["protocol_sha256"] = digest(protocol)
    elif mutation == "forged_prior":
        execution["prerequisites"] = {"prepare": {"run_dir": str(evidence.parent / "prepare"), "manifest_sha256": "2" * 64}}
    else:
        summary = json.loads((evidence / "summary.json").read_text())
        summary["status"] = "INCOMPLETE"
        write_json(evidence / "summary.json", summary)
    write_json(evidence / "execution.json", execution)
    with pytest.raises(ValueError):
        seal_stage(evidence, stage="structure", profile="smoke", source_tree_sha256=SOURCE)


def test_completed_marker_and_declared_file_size_cannot_be_forged(evidence):
    manifest = json.loads((evidence / "manifest.json").read_text())
    manifest["files"]["fields.npz"]["bytes"] += 1
    write_json(evidence / "manifest.json", manifest)
    (evidence / "COMPLETED").write_text(file_digest(evidence / "manifest.json") + "\n")
    with pytest.raises(ValueError, match="inventory changed"):
        verify_stage(evidence)


def test_inner_science_manifest_must_bind_actual_science_even_before_outer_sealing(evidence):
    _unseal(evidence)
    names = [p.name for p in evidence.iterdir() if p.is_file() and p.name != "stage.json"]
    write_json(evidence / "science_manifest.json", {"protocol_sha256": digest(build_protocol("smoke")),
        "artifacts": {name: file_digest(evidence / name) for name in names}})
    (evidence / "fields.npz").write_bytes((evidence / "fields.npz").read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Inner agenda artifact differs"):
        seal_stage(evidence, stage="structure", profile="smoke", source_tree_sha256=SOURCE)


def test_controller_artifact_import_has_no_tensor_scientific_dependencies():
    command = "import sys; import tdn.analysis.agenda.artifacts; assert not ({'torch','numpy','scipy'} & set(sys.modules))"
    result = subprocess.run([sys.executable, "-c", command], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_real_prepared_banks_and_relocated_lineage_verify(prepared):
    manifest = verify_stage(prepared, stage="prepare", profile="smoke", source_tree_sha256=SOURCE)
    assert {"train.pt", "validation.pt", "calibration.pt", "confirmation.pt"} <= manifest["files"].keys()
    assert manifest["prerequisites"]["structure"]["manifest_sha256"] == file_digest(prepared.parent / "structure" / "manifest.json")


@pytest.mark.parametrize("mutation", ("previous_bytes", "manifest_identity", "omitted_prerequisite", "software_mismatch", "normalization_cohort"))
def test_actual_prepare_chain_rejects_predecessor_or_training_lineage_changes(prepared, mutation):
    if mutation == "previous_bytes":
        predecessor = prepared.parent / "structure" / "fields.npz"
        predecessor.write_bytes(predecessor.read_bytes() + b"tampered")
        with pytest.raises(ValueError):
            verify_stage(prepared)
        return
    _unseal(prepared)
    execution = json.loads((prepared / "execution.json").read_text())
    if mutation == "manifest_identity":
        execution["prerequisites"]["structure"]["manifest_sha256"] = "2" * 64
    elif mutation == "omitted_prerequisite":
        execution["prerequisites"] = {}
    elif mutation == "software_mismatch":
        execution["software"]["python"] = "foreign-runtime"
    else:
        normalization = json.loads((prepared / "normalization.json").read_text())
        normalization["split"] = "validation"
        write_json(prepared / "normalization.json", normalization)
    write_json(prepared / "execution.json", execution)
    with pytest.raises(ValueError):
        seal_stage(prepared, stage="prepare", profile="smoke", source_tree_sha256=SOURCE)
    assert not (prepared / "COMPLETED").exists()
