"""Portable evidence recovery checks; these fixtures perform no scientific run."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def native(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "scaling_export_under_test", ROOT / "scripts/response_preconditioner_scaling_native.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    return module


@pytest.fixture
def terminal_run(native, tmp_path):
    run = tmp_path / "runs" / ("launch-" + "b" * 32 + "-a1")
    (run / "outputs/native").mkdir(parents=True)
    (run / "outputs/study/queries").mkdir(parents=True)
    (run / "run.json").write_text(json.dumps(
        dict(schema="raven.run/v1", run_id=run.name, state="COMPLETED")
    ))
    (run / "outputs/native/status.json").write_text(json.dumps(
        dict(schema=native.SCHEMA, status="COMPLETED")
    ))
    return run


def digest_stream(stream):
    digest = hashlib.sha256()
    count = 0
    while block := stream.read(1 << 20):
        digest.update(block)
        count += len(block)
    return dict(bytes=count, sha256=digest.hexdigest())


def test_export_recovers_orphan_bundle_above_old_64_mib_cap_and_reassembles(
    native, terminal_run, monkeypatch
):
    """A failed packing attempt can leave a large ZIP without packing.json."""
    run = terminal_run
    orphan = run / "outputs/native/portable-study-evidence.zip"
    block = b"retained scientific evidence\n" * 4096
    remaining = 65 << 20
    digest = hashlib.sha256()
    with orphan.open("xb") as stream:
        while remaining:
            piece = block[:min(remaining, len(block))]
            stream.write(piece)
            digest.update(piece)
            remaining -= len(piece)
    before = orphan.stat()
    (run / "outputs/native/status.json").write_text(json.dumps(
        dict(schema=native.SCHEMA, status="FAILED", error="Packing metadata exceeded old cap")
    ))
    monkeypatch.setattr(native, "UPLOAD_PART_BYTES", 32 << 10)

    folder = native.export_review(str(run))
    index = json.loads((folder / "index.json").read_text())
    row = next(row for row in index["files"] if row["path"] == orphan.relative_to(run).as_posix())
    assert row == dict(path=orphan.relative_to(run).as_posix(), bytes=65 << 20, sha256=digest.hexdigest())
    assert index["omitted_duplicates"] == []
    assert len(index["parts"]) > 1
    assert not (folder / index["archive"]).exists()
    joined = bytearray()
    for part in index["parts"]:
        raw = (folder / part["path"]).read_bytes()
        assert len(raw) == part["bytes"] <= 32 << 10
        assert hashlib.sha256(raw).hexdigest() == part["sha256"]
        joined.extend(raw)
    assert len(joined) == index["archive_bytes"]
    assert hashlib.sha256(joined).hexdigest() == index["archive_sha256"]
    with zipfile.ZipFile(io.BytesIO(joined)) as bundle:
        with bundle.open(run.name + "/" + row["path"]) as stream:
            assert digest_stream(stream) == {key: row[key] for key in ("bytes", "sha256")}
    after = orphan.stat()
    assert (after.st_ino, after.st_size, after.st_mtime_ns) == (
        before.st_ino, before.st_size, before.st_mtime_ns
    )
    assert not (run / "outputs/native/packing.json").exists()


def test_full_scale_inventory_exceeds_small_metadata_cap_without_relaxing_default(
    native, terminal_run
):
    """Exercise both real inventories at the declared scale, using tiny payloads."""
    run = terminal_run
    study = run / "outputs/study"
    payload = b"fixture\n"
    checksum = hashlib.sha256(payload).hexdigest()
    for number in range(7680):
        name = f"train-p{number:04d}-context-000-calibration-source-000-reference-action.npz"
        (study / "queries" / name).write_bytes(payload)

    packed = native.pack_study_evidence({"run": run}, study)
    packing_path = run / "outputs/native/packing.json"
    assert (1 << 20) < packing_path.stat().st_size <= native.INVENTORY_BYTES
    assert len(packed["files"]) == 7680
    assert all(row["bytes"] == len(payload) and row["sha256"] == checksum for row in packed["files"])
    with zipfile.ZipFile(run / packed["archive"]) as bundle:
        assert len(bundle.namelist()) == 7681
        embedded = json.loads(bundle.read("packing-manifest.json"))
        assert embedded["files"] == packed["files"]
        assert bundle.read(packed["files"][0]["path"]) == payload
        assert bundle.read(packed["files"][-1]["path"]) == payload

    folder = native.export_review(str(run))
    index_path = folder / "index.json"
    assert (1 << 20) < index_path.stat().st_size <= native.INVENTORY_BYTES
    index = json.loads(index_path.read_text())
    assert index["file_count"] == 7683
    assert len(index["omitted_duplicates"]) == 1
    assert index["omitted_duplicates"][0]["path"] == packed["archive"]
    assert len(index["parts"]) == 1
    with zipfile.ZipFile(folder / index["archive"]) as bundle:
        assert len(bundle.namelist()) == index["file_count"]
        for row in index["files"]:
            assert bundle.getinfo(run.name + "/" + row["path"]).file_size == row["bytes"]
        assert run.name + "/" + packed["archive"] not in bundle.namelist()
    with pytest.raises(ValueError):
        native.write_json(run / "oversized-status.json", {"status": "x" * (1 << 20)}, root=native.ROOT)
    assert not (run / "oversized-status.json").exists()


def test_stream_hash_and_copy_are_chunked_and_exact(native, tmp_path):
    path = tmp_path / "evidence.bin"
    raw = bytes(range(256)) * 8193
    path.write_bytes(raw)

    class Sink(io.BytesIO):
        lengths = []

        def write(self, block):
            self.lengths.append(len(block))
            return super().write(block)

    sink = Sink()
    expected = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    assert native._stream_evidence(path, root=tmp_path, limit=len(raw), sink=sink) == expected
    assert native._stream_evidence(path, root=tmp_path, limit=len(raw)) == expected
    assert sink.getvalue() == raw
    assert len(sink.lengths) >= 3 and max(sink.lengths) <= native.IO_CHUNK_BYTES


def test_stream_rejects_oversize_before_copying(native, tmp_path):
    path = tmp_path / "evidence.bin"
    path.write_bytes(b"12345678")
    sink = io.BytesIO()
    with pytest.raises(ValueError):
        native._stream_evidence(path, root=tmp_path, limit=7, sink=sink)
    assert sink.getvalue() == b""


@pytest.mark.parametrize("change", ["grow", "same_size", "replace"])
def test_stream_detects_evidence_changed_while_copying(native, tmp_path, change):
    path = tmp_path / "evidence.bin"
    raw = b"a" * (2 * native.IO_CHUNK_BYTES)
    path.write_bytes(raw)
    original_stat = path.stat()

    class MutatingSink:
        changed = False
        total = 0

        def write(self, block):
            self.total += len(block)
            if not self.changed:
                self.changed = True
                if change == "grow":
                    with path.open("ab") as stream:
                        stream.write(b"growth")
                elif change == "same_size":
                    with path.open("r+b") as stream:
                        stream.seek(native.IO_CHUNK_BYTES)
                        stream.write(b"changed")
                    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 1_000_000))
                else:
                    path.rename(tmp_path / "original-evidence.bin")
                    path.write_bytes(raw)
            return len(block)

    sink = MutatingSink()
    with pytest.raises(ValueError):
        native._stream_evidence(path, root=tmp_path, limit=len(raw), sink=sink)
    assert sink.changed and sink.total <= len(raw)


@pytest.mark.parametrize("kind", ["symlink", "parent_symlink", "fifo", "directory", "outside"])
def test_stream_rejects_nonregular_or_unconfined_evidence(native, tmp_path, kind):
    root = tmp_path / "evidence"
    root.mkdir()
    target = root / "regular.bin"
    target.write_bytes(b"evidence")
    path = root / "untrusted"
    if kind == "symlink":
        path.symlink_to(target)
    elif kind == "parent_symlink":
        path.symlink_to(root, target_is_directory=True)
        path = path / "regular.bin"
    elif kind == "fifo":
        os.mkfifo(path)
    elif kind == "directory":
        path.mkdir()
    else:
        path = tmp_path / "outside.bin"
        path.write_bytes(b"outside")
    with pytest.raises((ValueError, OSError)):
        native._stream_evidence(path, root=root, limit=1024)


@pytest.mark.parametrize("operation", ["pack", "export"])
def test_aggregate_caps_still_reject_without_mutating_originals(native, terminal_run, monkeypatch, operation):
    run = terminal_run
    paths = [run / "outputs/study/queries" / f"q{number}.npz" for number in range(2)]
    for path in paths:
        path.write_bytes(b"x" * 64)
    if operation == "pack":
        monkeypatch.setattr(native, "PACK_TOTAL_BYTES", 127)
        with pytest.raises(ValueError):
            native.pack_study_evidence({"run": run}, run / "outputs/study")
        assert not (run / "outputs/native/packing.json").exists()
    else:
        total = sum(path.stat().st_size for path in run.rglob("*") if path.is_file())
        monkeypatch.setattr(native, "REVIEW_TOTAL_BYTES", total - 1)
        with pytest.raises(ValueError):
            native.export_review(str(run))
        assert not list((native.ROOT / ".runtime/response-preconditioner-scaling-exports").glob("*/index.json"))
    assert all(path.read_bytes() == b"x" * 64 for path in paths)


@pytest.mark.parametrize("raven_state,native_state", [("RUNNING", "COMPLETED"), ("COMPLETED", "RUNNING")])
def test_export_still_requires_both_terminal_records(native, terminal_run, raven_state, native_state):
    run = terminal_run
    (run / "run.json").write_text(json.dumps(
        dict(schema="raven.run/v1", run_id=run.name, state=raven_state)
    ))
    (run / "outputs/native/status.json").write_text(json.dumps(
        dict(schema=native.SCHEMA, status=native_state)
    ))
    with pytest.raises(ValueError, match="terminal"):
        native.export_review(str(run))
    assert not (native.ROOT / ".runtime/response-preconditioner-scaling-exports").exists()
