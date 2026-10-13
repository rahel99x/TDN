"""Bulk science, failed workers and lossless portable research evidence.

Small forced budgets exercise the native aggregate failure without allocating
the original 539 MB document or treating controlled job metadata as CUDA work.
"""
import hashlib
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from tdn.analysis.response import diagnostic_metrics as metadata
from tdn.analysis.response import diagnostic_report as transport
from tdn.analysis.tangent_response import report, report_artifacts
from tdn.research.protocol import digest
from tdn.runtime.metadata import write_json


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture(autouse=True)
def isolated_reporting_environment(monkeypatch):
    monkeypatch.delenv("TDN_TANGENT_WORKFLOW", raising=False)
    monkeypatch.delenv("TDN_RESPONSE_RAVEN_DIR", raising=False)


def _budgets(monkeypatch, *, page=16 << 10, chunk=4 << 10):
    monkeypatch.setattr(report_artifacts, "PAGE_BYTES", page)
    monkeypatch.setattr(report_artifacts, "CHUNK_BYTES", chunk)
    monkeypatch.setattr(metadata, "MAX_METADATA_BYTES", 64 << 10)


def _context(tmp_path, stage="report", *, legacy=False):
    stages = (["legacy"] if legacy else []) + ["compression", "report"]
    units = {name: dict(kind="mechanism" if name == "legacy" else name, device="cpu") for name in stages}
    protocol = dict(profile="pilot", execution_scope="NATIVE_RESEARCH", stages=stages, units=units)
    path = tmp_path / "tangent-response-large-fixture" / stage
    path.mkdir(parents=True)
    return SimpleNamespace(path=path, stage=stage, protocol=protocol, unit=units[stage],
                           device="cpu", sourcehash="a" * 64, prereqs={})


def _compression_raw(size=45 << 10):
    rows = [dict(query_id=f"query-{i}", parent_id=f"parent-{i // 3}", method="fixed_shared_basis",
                 status="RECORDED", rms=(i + 1) / 1000, maximum=(i + 1) / 500) for i in range(12)]
    payload = dict(action_rows=rows, coverage=dict(requested=14, recorded=12, missing=2),
                   comment="Controlled serialization fixture, not numerical research")
    # Include multibyte characters at arbitrary segment boundaries. This bulk
    # annotation does not create a giant compiled measurement row.
    payload["annotation"] = "é🙂" * (size // 6)
    return (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def _seal(context, raw, *, extra=None):
    base = context.path.parent / "compression"
    base.mkdir(exist_ok=True)
    (base / "compression_results.json").write_bytes(raw)
    write_json(base / "summary.json", dict(status="COMPLETED", elapsed_seconds=86.1,
               scientific_outcome="MEASUREMENTS_RECORDED", device="cpu"))
    write_json(base / "decision.json", dict(gate="B", decision="STOP", eligible=False,
               denominators=dict(requested=14, eligible=12, missing=2)))
    for name, payload in (extra or {}).items():
        (base / name).write_bytes(payload)
    artifacts = {p.name: _sha(p.read_bytes()) for p in base.iterdir() if p.is_file()}
    write_json(base / "science_manifest.json", dict(schema="tdn.tangent-response-science/v1",
               stage="compression", profile=context.protocol["profile"], protocol_sha256=digest(context.protocol),
               source_tree_sha256=context.sourcehash, artifacts=artifacts))
    (base / "COMPLETED").write_text(_sha((base / "science_manifest.json").read_bytes()) + "\n")
    context.prereqs["compression"] = base
    return base


def _terminal(context, *, status="FAILED", code=2):
    return dict(status=status, native_worker_finalized=True, native_worker_runtime_seconds=95.,
                native_worker_exit_code=code, native_worker_error="Controlled observer failure" if code else None,
                science_body_status="COMPLETED", science_body_elapsed_seconds=86.1)


def _segmented(tmp_path, monkeypatch):
    _budgets(monkeypatch)
    output = tmp_path / "segments"
    output.mkdir()
    raw = _compression_raw()
    source_file = tmp_path / "compression_results.json"
    source_file.write_bytes(raw)
    payload = report_artifacts.ScientificPayload(source_file, tmp_path, _sha(raw))
    pages = report_artifacts.Pages(output, "tangent-response-segment-fixture", report.CONFIG,
                                  dict(source_tree_sha256="a" * 64, protocol_sha256="b" * 64))
    index, descriptor = report_artifacts.source_segments(pages, source_file.name, payload)
    pages.finish()
    source = dict(path=index["path"], representation="exact-segments", source_descriptor_sha256=index["sha256"],
                  sha256=_sha(raw), bytes=len(raw))
    return output, source, descriptor, raw


@pytest.mark.parametrize("page,chunk,size", [(16 << 10, 4 << 10, 45 << 10),
                                            (64 << 10, 32 << 10, 150 << 10)])
def test_registered_sealed_compression_uses_separate_bound_without_relaxing_metadata(
        tmp_path, monkeypatch, page, chunk, size):
    _budgets(monkeypatch, page=page, chunk=chunk)
    context = _context(tmp_path)
    raw = _compression_raw(size)
    base = _seal(context, raw)
    assert len(raw) > page
    assert report_artifacts.SCIENCE_PAYLOAD_BYTES == 1 << 30
    if len(raw) > metadata.MAX_METADATA_BYTES:
        with pytest.raises(ValueError, match="bounded regular file"):
            metadata.read_metadata(base / "compression_results.json", root=base)

    data, copies = report.collect(context)
    assert metadata.MAX_METADATA_BYTES == 64 << 10
    assert data["summary"]["completed_science_units"] == 1
    assert data["summary"]["unresolved_source_units"] == 0
    assert len(data["tables"]["action_rows"]) == 12
    assert data["decisions"][0]["decision"] == "STOP"
    assert data["decisions"][0]["eligible"] is False
    source = next(s for s in data["sources"] if s["name"] == "compression_results.json")
    payload = copies[source["path"]]
    assert isinstance(payload, report_artifacts.ScientificPayload)
    assert payload.sha256 == source["sha256"] == _sha(raw)
    assert b"".join(part for _, part in payload.chunks()) == raw


def test_unregistered_large_json_cannot_bypass_generic_metadata_bound(tmp_path, monkeypatch):
    _budgets(monkeypatch)
    context = _context(tmp_path)
    _seal(context, _compression_raw(100), extra={"unregistered.json": _compression_raw(80 << 10)})
    data, copies = report.collect(context)
    row = next(r for r in data["tables"]["units"] if r["stage"] == "compression")
    assert row["status"] == "UNRESOLVED_SOURCE"
    assert "bounded regular file" in row["reason"]
    assert not data["sources"] and not data["decisions"] and not copies
    assert metadata.MAX_METADATA_BYTES == 64 << 10


def test_native_finalizer_preserves_completed_body_and_actual_failed_worker(tmp_path, monkeypatch):
    _budgets(monkeypatch)
    context = _context(tmp_path, "compression")
    base = _seal(context, _compression_raw())
    scientific_summary = (base / "summary.json").read_bytes()
    marker = (base / "COMPLETED").read_bytes()
    attempt = report.start_attempt(context, job_id="2891", worker_managed=True)
    report.finish_attempt(context, dict(status="COMPLETED", elapsed_seconds=86.1))
    assert json.loads((attempt / "run.json").read_text())["state"] == "RUNNING"
    report.finish_attempt(context, _terminal(context))

    run = json.loads((attempt / "run.json").read_text())
    summary = json.loads((attempt / "summary.json").read_text())
    assert run["state"] == summary["state"] == "FAILED"
    assert run["job_id"] == summary["job_id"] == "2891"
    assert summary["exit_code"] == 2 and summary["runtime_seconds"] == 95.
    assert summary["results"]["science_body_status"] == "COMPLETED"
    assert summary["results"]["science_body_elapsed_seconds"] == 86.1
    assert (base / "summary.json").read_bytes() == scientific_summary
    assert (base / "COMPLETED").read_bytes() == marker
    index = json.loads((attempt / "outputs/source_index.json").read_text())
    source = next(s for s in index["sources"] if s["source_name"] == "compression_results.json")
    descriptor_raw = (attempt / source["path"]).read_bytes()
    descriptor = json.loads(descriptor_raw)
    assert source["sha256"] == _sha(descriptor_raw)
    assert source["source_sha256"] == descriptor["source_sha256"] == _sha((base / "compression_results.json").read_bytes())
    assert descriptor["science_seal_status"] == "VERIFIED"
    assert descriptor["availability"] == "ORIGINAL_NATIVE_PROJECT_ONLY"
    assert source["representation"] == "native-source-descriptor"
    # The worker holds provenance, not a second copy of the full aggregate.
    assert not any(p.name.endswith(".txt") for p in (attempt / "outputs/large-science").rglob("*"))


def test_failed_unsealed_aggregate_retains_bytes_without_claiming_valid_science(tmp_path, monkeypatch):
    _budgets(monkeypatch)
    context = _context(tmp_path, "compression")
    raw = _compression_raw()[:-2]  # An interrupted JSON document is still failure evidence.
    (context.path / "compression_results.json").write_bytes(raw)
    write_json(context.path / "summary.json", dict(status="FAILED", elapsed_seconds=12., error="Interrupted producer"))
    attempt = report.start_attempt(context, job_id="2891", worker_managed=True)
    report.finish_attempt(context, _terminal(context))
    summary = json.loads((attempt / "summary.json").read_text())
    assert summary["state"] == "FAILED" and summary["results"]["science_body_status"] == "FAILED"
    index = json.loads((attempt / "outputs/source_index.json").read_text())
    source = next(s for s in index["sources"] if s["source_name"] == "compression_results.json")
    assert source["science_seal_status"] == "UNSEALED_FAILURE_EVIDENCE"
    assert source["science_manifest_sha256"] is None
    citation = dict(path=source["local_path"], representation=source["representation"],
                    source_descriptor_sha256=source["sha256"], sha256=source["source_sha256"], bytes=source["bytes"])
    assert b"".join(report_artifacts.source_bytes(attempt / "outputs/unsealed-science", citation)) == raw
    assert not (context.path / "COMPLETED").exists()


def test_bulk_export_failure_still_publishes_actual_terminal_lifecycle(tmp_path, monkeypatch):
    _budgets(monkeypatch)
    context = _context(tmp_path, "compression")
    _seal(context, _compression_raw())
    attempt = report.start_attempt(context, job_id="2891", worker_managed=True)
    monkeypatch.setattr(report_artifacts, "SCIENCE_PAYLOAD_BYTES", 32 << 10)
    with pytest.raises(ValueError, match="bounded|bound"):
        report.finish_attempt(context, _terminal(context))
    summary = json.loads((attempt / "summary.json").read_text())
    assert json.loads((attempt / "run.json").read_text())["state"] == summary["state"] == "FAILED"
    assert summary["exit_code"] == 2
    assert summary["results"]["science_body_status"] == "COMPLETED"


def test_source_bytes_recovers_exact_utf8_source_and_enforces_separate_size_bound(tmp_path, monkeypatch):
    output, source, descriptor, raw = _segmented(tmp_path, monkeypatch)
    assert b"".join(report_artifacts.source_bytes(output, source)) == raw
    assert len(descriptor["segments"]) > 1
    for part in descriptor["segments"]:
        assert (output / part["path"]).read_bytes().decode("utf-8")
        assert part["bytes"] <= report_artifacts.CHUNK_BYTES + 4
    monkeypatch.setattr(report_artifacts, "SCIENCE_PAYLOAD_BYTES", len(raw) - 1)
    with pytest.raises(ValueError, match="segment identity"):
        b"".join(report_artifacts.source_bytes(output, source))
    original = tmp_path / "compression_results.json"
    with pytest.raises(ValueError, match="separate 1 GiB"):
        report_artifacts.ScientificPayload(original, tmp_path, _sha(raw))


@pytest.mark.parametrize("damage,match", [
    ("corrupt_part", "segment hash"), ("missing_part", None),
    ("byte_gap", "byte gap"), ("oversize_part", "exceed their bound"),
    ("whole_hash", "whole hash"), ("descriptor_hash", "science seal"),
])
def test_source_bytes_rejects_changed_missing_or_misdeclared_segments(tmp_path, monkeypatch, damage, match):
    output, source, descriptor, _ = _segmented(tmp_path, monkeypatch)
    first = output / descriptor["segments"][0]["path"]
    if damage == "corrupt_part":
        raw = first.read_bytes()
        first.write_bytes(b"!" + raw[1:])
    elif damage == "missing_part":
        first.unlink()
    else:
        if damage == "byte_gap":
            descriptor["segments"][0]["byte_offset"] = 1
        elif damage == "oversize_part":
            descriptor["segments"][0]["bytes"] = report_artifacts.CHUNK_BYTES + 5
        elif damage == "whole_hash":
            source["sha256"] = descriptor["source_sha256"] = "0" * 64
        elif damage == "descriptor_hash":
            descriptor["source_bytes"] += 1
        path = output / source["path"]
        write_json(path, descriptor)
        if damage != "descriptor_hash":
            source["source_descriptor_sha256"] = _sha(path.read_bytes())
    with pytest.raises((ValueError, OSError), match=match):
        b"".join(report_artifacts.source_bytes(output, source))


def test_source_bytes_refuses_unsegmented_source_above_page_bound(tmp_path, monkeypatch):
    _budgets(monkeypatch)
    raw = _compression_raw()
    file = tmp_path / "large.json"
    file.write_bytes(raw)
    original_open = Path.open

    class NoContentRead:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def fileno(self):
            return self.stream.fileno()

        def read(self, *args):
            pytest.fail("Oversized inline source was read before rejecting its size")

    def guarded_open(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        return NoContentRead(stream) if path == file else stream

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "read_bytes", lambda *args: pytest.fail("Inline source used an unbounded read_bytes"))
    with pytest.raises(ValueError, match="bounded regular-file"):
        b"".join(report_artifacts.source_bytes(tmp_path, dict(path="large.json", sha256=_sha(raw), bytes=len(raw))))


@pytest.mark.parametrize("declared_bytes", [1, 1 << 40])
def test_inline_source_bytes_rejects_wrong_expected_size(tmp_path, monkeypatch, declared_bytes):
    _budgets(monkeypatch)
    raw = b'{"scope":"bounded inline source"}\n'
    (tmp_path / "inline.json").write_bytes(raw)
    monkeypatch.setattr(Path, "read_bytes", lambda *args: pytest.fail("Inline source used an unbounded read_bytes"))
    with pytest.raises(ValueError, match="differs from its original bytes"):
        b"".join(report_artifacts.source_bytes(tmp_path, dict(path="inline.json", sha256=_sha(raw), bytes=declared_bytes)))


def _trusted_raven(monkeypatch):
    root = Path(os.environ.get("TDN_RAVEN_TEST_ROOT", "/workspace/Raven"))
    if not (root / "raven/projects.py").is_file():
        pytest.skip("Trusted optional Raven checkout unavailable")
    monkeypatch.syspath_prepend(str(root))
    return [importlib.import_module("raven." + name) for name in
            ("projects", "reporting", "metrics", "artifacts", "artifact_pages")]


def test_portable_review_retains_large_source_and_incomplete_worker_after_relocation(tmp_path, monkeypatch):
    projects, reporting, metrics, artifacts, artifact_pages = _trusted_raven(monkeypatch)
    _budgets(monkeypatch, page=64 << 10, chunk=32 << 10)
    context = _context(tmp_path, legacy=True)
    raw = _compression_raw(150 << 10)
    _seal(context, raw)
    worker_context = SimpleNamespace(**vars(context))
    worker_context.stage = "compression"
    worker_context.path = context.path.parent / "compression"
    worker_context.unit = context.protocol["units"]["compression"]
    worker = report.start_attempt(worker_context, job_id="2891", worker_managed=True)
    report.finish_attempt(worker_context, _terminal(worker_context))

    legacy_context = SimpleNamespace(**vars(context))
    legacy_context.stage = "legacy"
    legacy_context.path = context.path.parent / "legacy"
    legacy_context.unit = context.protocol["units"]["legacy"]
    legacy = report.start_attempt(legacy_context, job_id="2890", worker_managed=True)
    (legacy / "logs/research.log").write_text("Original large artifact finalization failed\n")
    state_path = context.path.parent / "state/legacy.json"
    state_path.parent.mkdir()
    write_json(state_path, dict(status="FAILED", slurm_job_id="2890", exit_code=2,
               error="Terminal native reporting/seal failed: bounded metadata"))
    original_inventory = (legacy / "run.json").read_bytes()
    original_state = state_path.read_bytes()
    assert not (legacy / "summary.json").exists()

    report.start_attempt(context, job_id="2893", worker_managed=True)
    result = report.run(context)
    assert result["scientific_decisions"][0]["decision"] == "STOP"
    assert (legacy / "run.json").read_bytes() == original_inventory
    assert state_path.read_bytes() == original_state
    assert not (legacy / "summary.json").exists()
    manifest = transport.verify_portable(context.path / report.BUNDLE)
    assert legacy.relative_to(metadata.raven_root(context)).as_posix() in manifest["excluded_attempts"]

    destination = tmp_path / "different-machine" / "relocated"
    destination.mkdir(parents=True)
    with zipfile.ZipFile(context.path / report.BUNDLE) as archive:
        for name in archive.namelist():
            metadata.relative_path(name)
            file = destination / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(archive.read(name))
    overview = destination / "runs" / (context.path.parent.name + "-analysis") / "outputs"
    index = json.loads((overview / "tangent_research.json").read_text())
    source = next(s for s in report_artifacts.read_section(overview, index, "sources")
                  if s["name"] == "compression_results.json")
    assert source["representation"] == "exact-segments"
    assert b"".join(report_artifacts.source_bytes(overview, source)) == raw
    assert len(report_artifacts.read_rows(overview, index, "action_rows")) == 12
    incomplete = report_artifacts.read_section(overview, index, "incomplete_attempts")[0]
    assert incomplete["original_raven_state"] == "RUNNING"
    assert incomplete["actual_native_worker_state"] == "FAILED" and incomplete["job_id"] == "2890"
    assert "summary.json" in incomplete["missing_declared_paths"]
    inventory_source = next(s for s in incomplete["files"] if s["original_name"] == "run.json")
    state_source = next(s for s in incomplete["files"] if s["original_name"] == "native-worker-state")
    assert b"".join(report_artifacts.source_bytes(overview, inventory_source)) == original_inventory
    assert b"".join(report_artifacts.source_bytes(overview, state_source)) == original_state
    assert not (destination / "runs" / legacy.name).exists()

    checked = 0
    for config in destination.rglob(".raven/config.json"):
        project = config.parent.parent
        discovered = projects.discover_project(str(project))
        assert not discovered["warnings"]
        for entry in discovered["runs"]:
            run = projects.validate_inventory(entry["inventory"])
            attempt = project / "runs" / run["run_id"]
            summary = reporting._summary(json.loads((attempt / "summary.json").read_text()))
            if run["run_id"] == worker.name:
                assert summary["state"] == "FAILED" and summary["job_id"] == "2891"
            for line in (attempt / "metrics.jsonl").read_text().splitlines():
                metrics._record(json.loads(line))
            contract = json.loads((attempt / "outputs/contract.json").read_text())
            assert artifacts.validate_contract(contract, str(attempt))["valid"]
            for specification in contract["outputs"]:
                page = artifact_pages.read_page(str(attempt), specification)
                assert page["status"] == "ready"
            checked += 1
    assert checked > 2


def test_failed_attempt_with_dangling_log_is_excluded_and_retained_unchanged(tmp_path, monkeypatch):
    _budgets(monkeypatch, page=64 << 10, chunk=32 << 10)
    context = _context(tmp_path, legacy=True)
    legacy_context = SimpleNamespace(**vars(context))
    legacy_context.stage = "legacy"
    legacy_context.path = context.path.parent / "legacy"
    legacy_context.unit = context.protocol["units"]["legacy"]
    legacy = report.start_attempt(legacy_context, job_id="2890", worker_managed=True)
    terminal = _terminal(legacy_context)
    terminal.update(science_body_status="NOT_STARTED", science_body_elapsed_seconds=None)
    report.finish_attempt(legacy_context, terminal)
    inventory = json.loads((legacy / "run.json").read_text())
    assert inventory["state"] == "FAILED"
    # This reproduces a failure hidden below a valid top-level log_index path.
    for field, pointer in inventory["paths"].items():
        target = legacy / pointer
        assert target.is_dir() and any(target.iterdir()) if field in ("outputs", "logs", "passports") else target.is_file()
    logs = json.loads((legacy / "logs.json").read_text())
    missing_log = "logs/missing-scheduler.err"
    logs["logs"].append(dict(id="missing-scheduler", path=missing_log, label="Scheduler stderr", group="Execution"))
    write_json(legacy / "logs.json", logs)
    original_inventory = (legacy / "run.json").read_bytes()
    original_summary = (legacy / "summary.json").read_bytes()
    original_log_index = (legacy / "logs.json").read_bytes()
    state_path = context.path.parent / "state/legacy.json"
    state_path.parent.mkdir()
    write_json(state_path, dict(status="FAILED", slurm_job_id="2890", exit_code=2, error="Missing scheduler log"))

    report.start_attempt(context, job_id="2893", worker_managed=True)
    report.run(context)
    manifest = transport.verify_portable(context.path / report.BUNDLE)
    assert legacy.relative_to(metadata.raven_root(context)).as_posix() in manifest["excluded_attempts"]
    overview = metadata.raven_root(context) / "runs" / (context.path.parent.name + "-analysis") / "outputs"
    index = json.loads((overview / "tangent_research.json").read_text())
    incomplete = report_artifacts.read_section(overview, index, "incomplete_attempts")[0]
    assert incomplete["original_raven_state"] == incomplete["actual_native_worker_state"] == "FAILED"
    assert incomplete["missing_declared_paths"] == [missing_log]
    source = next(row for row in incomplete["files"] if row["original_name"] == "logs.json")
    assert "log-index.raw.json" in source["path"]
    assert b"".join(report_artifacts.source_bytes(overview, source)) == original_log_index
    assert (legacy / "run.json").read_bytes() == original_inventory
    assert (legacy / "summary.json").read_bytes() == original_summary
    assert (legacy / "logs.json").read_bytes() == original_log_index
    assert not (legacy / missing_log).exists()
