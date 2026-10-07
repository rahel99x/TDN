"""Real numerical-audit execution and fail-closed aggregate evidence."""
import json
import shutil

import pytest
import torch

from tdn.analysis.roadmap import engine
from tdn.analysis.roadmap.core import check
from tdn.analysis.roadmap.protocol import build_protocol
from tdn.research.protocol import digest, file_digest
from tdn.runtime.metadata import write_json


SOURCE = "7" * 64


@pytest.fixture(autouse=True)
def stable_test_source(monkeypatch):
    # Concurrent coding must not pretend to be a native source-stability test.
    # The real final CLI run verifies the actual executable fingerprint.
    monkeypatch.setattr(engine, "software_metadata", lambda: {"source_tree_sha256": SOURCE, "execution_evidence": "unit_test"})
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def audit(tmp_path):
    protocol = build_protocol("smoke")
    path = tmp_path / "audit"
    summary = engine.run_stage(protocol, "audit", path)
    return protocol, path, summary


def _rehash_artifacts(path):
    manifest = json.loads((path / engine.MANIFEST).read_text())
    manifest["artifacts"] = {name: file_digest(path / name) for name in manifest["artifacts"]}
    write_json(path / engine.MANIFEST, manifest)
    (path / "COMPLETED").write_text(digest(manifest) + "\n")


def test_actual_numerical_audit_is_sealed_and_repeat_is_refused(audit):
    protocol, path, summary = audit
    assert summary["status"] == "COMPLETED"
    assert summary["experiment_count"] >= 10
    seal = engine.verify_science(protocol, path)
    assert seal["stage"] == "audit"
    assert seal["source_tree_sha256"] == SOURCE
    rows = [json.loads(line) for line in (path / "rows.jsonl").read_text().splitlines()]
    assert any("C3" in row["combination_ids"] for row in rows)
    assert all(row["assessment"]["proof_status"] == "NOT_A_PROOF" for row in rows)
    with pytest.raises(FileExistsError, match="Preserve"):
        engine.run_stage(protocol, "audit", path)


@pytest.mark.parametrize("artifact", ["review.csv", "rows.jsonl", "COMPLETED"])
def test_changed_scientific_output_or_completion_marker_is_rejected(audit, artifact):
    protocol, path, _ = audit
    with (path / artifact).open("a") as handle:
        handle.write("changed")
    with pytest.raises(ValueError):
        engine.verify_science(protocol, path)


def test_unlisted_scientific_artifact_cannot_hide_outside_seal(audit):
    protocol, path, _ = audit
    (path / "omitted-failed-experiment.json").write_text('{"status":"FAILED"}')
    with pytest.raises(ValueError):
        engine.verify_science(protocol, path)


def test_resealed_row_with_wrong_stage_or_ids_is_rejected(audit):
    protocol, path, _ = audit
    rows = [json.loads(line) for line in (path / "rows.jsonl").read_text().splitlines()]
    rows[0]["stage"] = "confirm"
    rows[0]["mechanism_ids"] = ["M999"]
    rows[0]["row_sha256"] = digest({k: v for k, v in rows[0].items() if k != "row_sha256"})
    (path / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    write_json(path / "rows.json", {"schema": "tdn.roadmap-rows/v1", "rows": rows})
    _rehash_artifacts(path)
    with pytest.raises(ValueError):
        engine.verify_science(protocol, path)


def test_missing_stages_remain_all_29_visible_na_records(tmp_path):
    protocol = build_protocol("smoke")
    path = tmp_path / "report"
    summary = engine.run_stage(protocol, "report", path)
    data = json.loads((path / "mechanism_summary.json").read_text())
    assert summary["status"] == "COMPLETED"
    assert summary["details"]["verified_experiments"] == 0
    assert not summary["details"]["all_stage_evidence_verified"]
    assert {r["id"] for r in data["records"]} == {*(f"M{i:02}" for i in range(24)), *(f"C{i}" for i in range(5))}
    assert all(r["verdict"] == "NA" and r["score_1_100"] == 1 and r["evidence_coverage"] == 0 for r in data["records"])
    assert all(r["missing_stages"] for r in data["records"])
    assert engine.verify_science(protocol, path)["stage"] == "report"


def test_partial_report_credits_only_actual_per_mechanism_evidence(audit, tmp_path):
    protocol, audit_path, summary = audit
    path = tmp_path / "report"
    engine.run_stage(protocol, "report", path, prerequisites={"audit": audit_path})
    data = json.loads((path / "mechanism_summary.json").read_text())
    records = {row["id"]: row for row in data["records"]}
    assert data["verified_experiment_count"] == summary["experiment_count"]
    assert records["M02"]["experiment_count"] > 0
    assert "confirm" in records["M02"]["missing_stages"]
    assert records["M14"]["experiment_count"] == 0 and records["M14"]["verdict"] == "NA"
    assert records["C3"]["experiment_count"] > 0
    assert records["C4"]["experiment_count"] == 0


def test_worker_rejected_provenance_is_not_credited_even_when_science_hashes_match(audit, tmp_path):
    protocol, audit_path, _ = audit
    path = tmp_path / "report"
    path.mkdir()
    write_json(path / "execution.json", {"prerequisites": {"audit": {"verification": "INVALID", "error": "allocation mismatch"}}})
    engine.run_stage(protocol, "report", path, prerequisites={"audit": audit_path})
    data = json.loads((path / "mechanism_summary.json").read_text())
    assert data["verified_experiment_count"] == 0
    assert "allocation mismatch" in data["stage_status"]["audit"]["reason"]
    assert all(r["verdict"] == "NA" for r in data["records"])


def test_present_cli_execution_without_valid_wrapper_seal_is_not_credited(audit, tmp_path):
    protocol, audit_path, _ = audit
    write_json(audit_path / "execution.json", {"allocation": "deliberately-unsealed"})
    engine.run_stage(protocol, "report", tmp_path / "report", prerequisites={"audit": audit_path})
    data = json.loads((tmp_path / "report/mechanism_summary.json").read_text())
    assert data["verified_experiment_count"] == 0
    assert data["stage_status"]["audit"]["status"] == "MISSING_OR_INVALID"


def test_invalid_wrapper_hash_is_not_credited_with_valid_scientific_seal(tmp_path):
    protocol = build_protocol("smoke")
    path = tmp_path / "audit"
    path.mkdir()
    write_json(path / "execution.json", {"execution_evidence": "unit_fixture"})
    engine.run_stage(protocol, "audit", path)
    write_json(path / "stage.json", {"status": "COMPLETED"})
    wrapper = {"protocol_sha256": digest(protocol), "files": {
        name: file_digest(path / name) for name in ("execution.json", "protocol.json", "stage.json", engine.MANIFEST)}}
    wrapper["files"]["execution.json"] = "0" * 64
    write_json(path / "workflow-seal.json", wrapper)
    assert engine.verify_science(protocol, path)["stage"] == "audit"
    engine.run_stage(protocol, "report", tmp_path / "report", prerequisites={"audit": path})
    data = json.loads((tmp_path / "report/mechanism_summary.json").read_text())
    assert data["verified_experiment_count"] == 0
    assert "Execution seal" in data["stage_status"]["audit"]["reason"]


def test_failure_retains_completed_rows_without_success_marker(tmp_path, monkeypatch):
    from tdn.analysis.roadmap import numerical_audit
    def interrupted(ctx):
        ctx.record("completed-before-stop", ["M02"], checks=[check("finite", True, True, "eq", category="math")])
        raise TimeoutError("declared numerical budget exhausted")
    monkeypatch.setattr(numerical_audit, "run", interrupted)
    path = tmp_path / "audit"
    with pytest.raises(TimeoutError):
        engine.run_stage(build_protocol("smoke"), "audit", path)
    summary = json.loads((path / "summary.json").read_text())
    assert summary["status"] == "INTERRUPTED"
    assert summary["experiment_count"] == 1
    assert len((path / "rows.jsonl").read_text().splitlines()) == 1
    assert not (path / "COMPLETED").exists()
    assert not (path / engine.MANIFEST).exists()


def test_missing_prerequisite_and_full_local_confirmation_are_rejected(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="Missing required stage"):
        engine.run_stage(build_protocol("smoke"), "prepare", tmp_path / "prepare")
    monkeypatch.setenv("TDN_EXECUTION_MODE", "desktop")
    with pytest.raises(ValueError, match="native allocated Fedora"):
        engine.run_stage(build_protocol("full"), "confirm_prepare", tmp_path / "confirm_prepare")


def test_symlinked_scientific_artifact_is_rejected(audit, tmp_path):
    protocol, path, _ = audit
    outside = tmp_path / "saved-review.csv"
    shutil.copyfile(path / "review.csv", outside)
    (path / "review.csv").unlink()
    (path / "review.csv").symlink_to(outside)
    with pytest.raises(ValueError):
        engine.verify_science(protocol, path)
