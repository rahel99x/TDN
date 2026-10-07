"""Audit evidence must cover the immutable plan before science can start."""
import json

import pytest

from tdn.analysis.consistency import artifacts
from tdn.analysis.consistency.protocol import build_protocol
from tdn.runtime.metadata import write_json


def audit_evidence(path):
    path.mkdir(parents=True, exist_ok=True)
    protocol = build_protocol("smoke")
    required = set(protocol["audit_required_case_ids"])
    rows = [{"check_id": identity, "required": identity in required,
             "status": "PASS" if identity in required else "OBSERVED"}
            for identity in protocol["audit_case_ids"]]
    documents = {
        "protocol.json": protocol,
        "execution.json": {"stage": "audit", "profile": "smoke", "device": "cpu",
                           "execution_mode": "local-cpu", "protocol_sha256": artifacts.digest(protocol),
                           "software": {"source_tree_sha256": "a" * 64}},
        "summary.json": {"status": "COMPLETED", "stage": "audit", "profile": "smoke", "device": "cpu",
                         "coverage": {"expected": len(rows), "reported": len(rows)},
                         "fixture_scope": "seal contract only; no numerical measurements"},
        "checks.json": {"schema": protocol["schema"], "rows": rows},
    }
    for name, value in documents.items():
        write_json(path / name, value)
    return documents


def test_exact_audit_coverage_and_source_seal(tmp_path):
    audit_evidence(tmp_path)
    result = artifacts.seal_stage(tmp_path, stage="audit", profile="smoke", source_tree_sha256="a" * 64)
    assert result["benchmark_suite"] == "consistency"
    artifacts.verify_stage(tmp_path, stage="audit", profile="smoke", source_tree_sha256="a" * 64)
    with pytest.raises(ValueError, match="source_tree"):
        artifacts.verify_stage(tmp_path, source_tree_sha256="b" * 64)
    with pytest.raises(ValueError, match="Preserve"):
        artifacts.seal_stage(tmp_path, stage="audit", profile="smoke", source_tree_sha256="a" * 64)
    (tmp_path / "checks.json").write_text("tampered")
    with pytest.raises(ValueError, match="changed"):
        artifacts.verify_stage(tmp_path)


@pytest.mark.parametrize("change", ["missing", "duplicate", "required_flag", "failed", "wrong_profile", "wrong_device"])
def test_incomplete_or_relabelled_audit_cannot_be_sealed(tmp_path, change):
    documents = audit_evidence(tmp_path)
    rows = documents["checks.json"]["rows"]
    required = next(row for row in rows if row["required"])
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[-1] = rows[0]
    elif change == "required_flag":
        required["required"] = False
        required["status"] = "OBSERVED"
    elif change == "failed":
        required["status"] = "FAILED"
    elif change == "wrong_profile":
        documents["protocol.json"]["updates"] += 1
    else:
        documents["execution.json"]["device"] = "cuda"
    for name, value in documents.items():
        write_json(tmp_path / name, value)
    with pytest.raises(ValueError):
        artifacts.seal_stage(tmp_path, stage="audit", profile="smoke", source_tree_sha256="a" * 64)
    assert not (tmp_path / "COMPLETED").exists()


def test_foreign_suite_seal_and_symlinks_are_rejected(tmp_path):
    audit_evidence(tmp_path)
    artifacts.seal_stage(tmp_path, stage="audit", profile="smoke", source_tree_sha256="a" * 64)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["benchmark_suite"] = "premix"
    write_json(tmp_path / "manifest.json", manifest)
    (tmp_path / "COMPLETED").write_text(artifacts.file_digest(tmp_path / "manifest.json"))
    with pytest.raises(ValueError, match="Unsupported"):
        artifacts.verify_stage(tmp_path)
    (tmp_path / "checks.json").unlink()
    (tmp_path / "checks.json").symlink_to(tmp_path / "protocol.json")
    manifest["benchmark_suite"] = "consistency"
    write_json(tmp_path / "manifest.json", manifest)
    (tmp_path / "COMPLETED").write_text(artifacts.file_digest(tmp_path / "manifest.json"))
    with pytest.raises(ValueError, match="unsafe"):
        artifacts.verify_stage(tmp_path)
