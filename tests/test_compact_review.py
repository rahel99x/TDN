"""Exact-byte, bounded upload and safe restoration checks for review exports.

These are artifact transport tests. They do not establish scientific success or
replace the original experiment's prerequisite/seal validation.
"""
from __future__ import annotations

import hashlib
import gzip
import copy
import importlib.util
import io
import json
from pathlib import Path
import random
import sys
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def exporter(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "compact_review_test", ROOT / "scripts" / "compact_review.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    project = tmp_path / "project with spaces"
    project.mkdir()
    monkeypatch.setattr(module, "ROOT", project)
    return module


@pytest.fixture
def run(exporter):
    base = exporter.ROOT / "runs" / "fedora-portfolio-fixture"
    files = {
        "protocol.json": b'{"source_commit":"c5c9a85","seed":711}\n',
        "jobs.json": b'[{"stage":"train","job_id":"123","state":"FAILED"}]\n',
        "report/summary.json": b'{"status":"COMPLETED","scientific_outcome":"NA"}\n',
        "report/measurements.csv": b"method,error,seconds\r\nours,0.012,0.004\r\ntheirs,0.008,0.007\r\n",
        "report/failure.json": b'{"status":"FAILED","error":"out of memory"}\n',
        "report/timing.jsonl": b'{"method":"ours","seconds":0.004}\n',
        "report/raw-table.json.gz": gzip.compress(b'[{"error":0.012}]\r\n', mtime=0),
        "logs/train.err": b"failed trial retained\n",
        "logs/train.out": b"original output\n",
        "state/source-seal.json": b'{"sha256":"historical-source"}\n',
        "models/selected.pt": b"checkpoint-bytes\x00\x80\xff" * 29,
        "models/last.pth": b"last-checkpoint\x00\x80\xff" * 7,
        "references/fields.npz": b"reference-array\x00\xff" * 17,
        "references/state.npy": b"npy-fixture\x00\xff" * 9,
        "report/atlas.png": b"png-rendering" * 19,
        "report/atlas.pdf": b"pdf-rendering" * 13,
        "report/diagram.svg": b"<svg><!-- rendered --></svg>\n",
        "report/preview.jpg": b"jpeg-rendering" * 17,
        "report/preview.jpeg": b"jpeg-rendering" * 17,
        "report/preview.webp": b"webp-rendering" * 11,
        "models/selected.safetensors": b"tensor-bytes\x00\xff" * 15,
        "report/empty.csv": b"",
    }
    for relative, data in files.items():
        path = base / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    # Identical evidence must reconstruct under both original names without
    # paying for two compressed payloads. The CSV is deliberately CRLF sealed.
    (base / "report/measurements-copy.csv").write_bytes(files["report/measurements.csv"])
    return base


def snapshot(base):
    return {
        str(path.relative_to(base)): path.read_bytes()
        for path in base.rglob("*")
        if path.is_file()
    }


def pack(exporter, run, name="bundle", *extra):
    output = exporter.ROOT / name
    if "--mode" not in extra:
        extra = ("--mode", "full", *extra)
    result = exporter.main([
        "pack", str(run), "--output", str(output), "--part-mib", "1", *extra
    ])
    assert result == 0
    index_path = output / "index.json"
    assert index_path.is_file()
    return output, index_path, json.loads(index_path.read_text())


def archive_bytes(output, index):
    return b"".join((output / row["path"]).read_bytes() for row in index["parts"])


def restored_run(destination, original):
    return destination / "runs" / original.name


def test_plan_does_not_write_or_require_current_scientific_source(exporter, run):
    before = snapshot(exporter.ROOT)
    assert exporter.main(["plan", str(run)]) == 0
    assert snapshot(exporter.ROOT) == before


def test_full_roundtrip_retains_exact_sealed_bytes_and_failure_evidence(exporter, run):
    original = snapshot(run)
    output, index_path, index = pack(exporter, run)
    assert exporter.main(["verify", str(index_path)]) == 0
    destination = exporter.ROOT / "restored"
    assert exporter.main(["unpack", str(index_path), "--output", str(destination)]) == 0
    assert snapshot(restored_run(destination, run)) == original
    assert snapshot(run) == original
    assert index["manifest_sha256"]
    for row in index["parts"]:
        data = (output / row["path"]).read_bytes()
        assert row["bytes"] == len(data)
        assert row["sha256"] == hashlib.sha256(data).hexdigest()
        assert len(data) <= 1024 * 1024
    with tarfile.open(fileobj=io.BytesIO(archive_bytes(output, index)), mode="r:xz") as tar:
        members = tar.getmembers()
        assert members[0].name == "REVIEW-MANIFEST.json"
        manifest = tar.extractfile(members[0]).read()
        assert hashlib.sha256(manifest).hexdigest() == index["manifest_sha256"]
        copies = [member for member in members if member.name.endswith((
            "/report/measurements.csv", "/report/measurements-copy.csv"
        ))]
        assert len(copies) == 2
        assert sum(member.islnk() for member in copies) == 1


def test_review_mode_keeps_all_raw_text_but_explicitly_omits_heavy_payloads(exporter, run):
    original = snapshot(run)
    output, index_path, index = pack(exporter, run, "review", "--mode", "review")
    destination = exporter.ROOT / "review-restored"
    assert exporter.main(["unpack", str(index_path), "--output", str(destination)]) == 0
    retained = snapshot(restored_run(destination, run))
    excluded = {".pt", ".pth", ".npz", ".npy", ".safetensors", ".png", ".pdf", ".svg", ".jpg", ".jpeg", ".webp"}
    assert retained == {
        name: data for name, data in original.items() if Path(name).suffix not in excluded
    }
    with tarfile.open(fileobj=io.BytesIO(archive_bytes(output, index)), mode="r:xz") as tar:
        manifest = json.loads(tar.extractfile("REVIEW-MANIFEST.json").read())
    entries = {row["path"]: row for row in manifest["files"]}
    for name, data in original.items():
        # A review subset must disclose absent checkpoints/references/figures,
        # as well as providing the byte-exact retained measurements.
        row = entries[f"runs/{run.name}/{name}"]
        assert row["bytes"] == len(data)
        assert row["sha256"] == hashlib.sha256(data).hexdigest()
        assert row["included"] is (Path(name).suffix not in excluded)
        if not row["included"]:
            assert row["reason"]
    assert snapshot(run) == original


def test_chunk_limits_and_corruption_are_checked_before_restoration(exporter, run):
    payload = random.Random(73419).randbytes(2 * 1024 * 1024 + 12345)
    (run / "models" / "incompressible.pt").write_bytes(payload)
    output, index_path, index = pack(exporter, run)
    assert len(index["parts"]) >= 3
    assert all(0 < row["bytes"] <= 1024 * 1024 for row in index["parts"])
    damaged = output / index["parts"][1]["path"]
    data = bytearray(damaged.read_bytes())
    data[len(data) // 2] ^= 1
    damaged.write_bytes(data)
    assert exporter.main(["verify", str(index_path)]) != 0
    destination = exporter.ROOT / "corrupt-restored"
    assert exporter.main(["unpack", str(index_path), "--output", str(destination)]) != 0
    assert not destination.exists()


def test_missing_chunk_is_not_silently_accepted(exporter, run):
    output, index_path, index = pack(exporter, run)
    (output / index["parts"][0]["path"]).unlink()
    assert exporter.main(["verify", str(index_path)]) != 0


def test_manifest_hash_is_checked(exporter, run):
    _, index_path, index = pack(exporter, run)
    index["manifest_sha256"] = "0" * 64
    index_path.write_text(json.dumps(index))
    assert exporter.main(["verify", str(index_path)]) != 0


def test_export_never_overwrites_existing_destination(exporter, run):
    output, _, _ = pack(exporter, run)
    original = snapshot(output)
    assert exporter.main(["pack", str(run), "--output", str(output)]) != 0
    assert snapshot(output) == original


def test_restore_never_overwrites_existing_destination(exporter, run):
    _, index_path, _ = pack(exporter, run)
    destination = exporter.ROOT / "owned"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_bytes(b"existing evidence\r\n")
    assert exporter.main(["unpack", str(index_path), "--output", str(destination)]) != 0
    assert snapshot(destination) == {"keep.txt": b"existing evidence\r\n"}


def test_output_inside_source_is_rejected(exporter, run):
    original = snapshot(run)
    assert exporter.main(["pack", str(run), "--output", str(run / "recursive")]) != 0
    assert snapshot(run) == original


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_source_symlinks_are_rejected(exporter, run, kind):
    target = run / ("protocol.json" if kind == "file" else "report")
    (run / "linked-evidence").symlink_to(target, target_is_directory=(kind == "directory"))
    assert exporter.main(["pack", str(run), "--output", str(exporter.ROOT / "bad-link")]) != 0


def test_source_outside_project_is_rejected(exporter):
    source = exporter.ROOT.parent / "external-run"
    source.mkdir()
    (source / "summary.json").write_text("{}\n")
    assert exporter.main(["pack", str(source), "--output", str(exporter.ROOT / "external-copy")]) != 0


def test_output_outside_project_is_rejected(exporter, run):
    output = exporter.ROOT.parent / "external-export"
    assert exporter.main(["pack", str(run), "--output", str(output)]) != 0
    assert not output.exists()


def test_restore_outside_project_is_rejected(exporter, run):
    _, index_path, _ = pack(exporter, run)
    output = exporter.ROOT.parent / "external-restore"
    assert exporter.main(["unpack", str(index_path), "--output", str(output)]) != 0
    assert not output.exists()


def test_part_paths_cannot_escape_bundle(exporter, run):
    output, index_path, index = pack(exporter, run)
    row = index["parts"][0]
    escaped = output.parent / "escaped-part"
    escaped.write_bytes((output / row["path"]).read_bytes())
    row["path"] = "../escaped-part"
    index_path.write_text(json.dumps(index))
    assert exporter.main(["verify", str(index_path)]) != 0


def test_missing_run_fails_without_creating_bundle(exporter):
    output = exporter.ROOT / "missing-export"
    assert exporter.main([
        "pack", str(exporter.ROOT / "runs" / "missing"), "--output", str(output)
    ]) != 0
    assert not output.exists()


def test_default_is_compact_review_and_latest_uses_portfolio_pointer(exporter, run):
    (exporter.ROOT / "runs" / ".fedora-portfolio-latest.json").write_text(
        json.dumps({"run_id": run.name})
    )
    output = exporter.ROOT / "default-review"
    assert exporter.main(["pack", "latest", "--output", str(output)]) == 0
    destination = exporter.ROOT / "default-restored"
    assert exporter.main([
        "unpack", str(output / "index.json"), "--output", str(destination)
    ]) == 0
    retained = snapshot(restored_run(destination, run))
    assert "report/measurements.csv" in retained
    assert "report/raw-table.json.gz" in retained
    assert "models/selected.pt" not in retained
    assert "report/atlas.png" not in retained


def test_symlinked_output_ancestor_cannot_escape_project(exporter, run):
    outside = exporter.ROOT.parent / "outside-destination"
    outside.mkdir()
    (exporter.ROOT / "aliased-output").symlink_to(outside, target_is_directory=True)
    assert exporter.main([
        "pack", str(run), "--output", str(exporter.ROOT / "aliased-output" / "bundle")
    ]) != 0
    assert not list(outside.iterdir())


def rewrite_tar(output, index_path, mutate):
    """Build a checksum-consistent hostile archive, rather than corrupt a chunk."""
    index = json.loads(index_path.read_text())
    original = archive_bytes(output, index)
    result = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(original), mode="r:xz") as source:
        with tarfile.open(fileobj=result, mode="w:xz", format=tarfile.PAX_FORMAT) as target:
            for original_member in source.getmembers():
                payload = source.extractfile(original_member).read() if original_member.isfile() else None
                member = copy.copy(original_member)
                mutate(member)
                target.addfile(member, io.BytesIO(payload) if member.isfile() and payload is not None else None)
    data = result.getvalue()
    for row in index["parts"]:
        (output / row["path"]).unlink()
    name = index["archive"]
    (output / name).write_bytes(data)
    index["bytes"] = len(data)
    index["sha256"] = hashlib.sha256(data).hexdigest()
    index["parts"] = [{"path": name, "bytes": len(data), "sha256": index["sha256"]}]
    index_path.write_text(json.dumps(index))


def test_rewritten_valid_tar_helper_preserves_integrity(exporter, run):
    output, index_path, _ = pack(exporter, run)
    rewrite_tar(output, index_path, lambda member: None)
    assert exporter.main(["verify", str(index_path)]) == 0


@pytest.mark.parametrize("attack", ["traversal", "absolute", "hardlink", "symlink", "fifo"])
def test_checksum_consistent_hostile_tar_cannot_restore(exporter, run, attack):
    output, index_path, _ = pack(exporter, run)

    def mutate(member):
        if not member.name.endswith("/logs/train.out"):
            return
        if attack == "traversal":
            member.name = "runs/../../escape.txt"
        elif attack == "absolute":
            member.name = str(exporter.ROOT / "escape.txt")
        else:
            member.size = 0
            member.type = {"hardlink": tarfile.LNKTYPE, "symlink": tarfile.SYMTYPE, "fifo": tarfile.FIFOTYPE}[attack]
            member.linkname = "../../escape.txt"

    rewrite_tar(output, index_path, mutate)
    # Independent part/archive hashes are valid, so rejection must inspect the
    # declared payload and safe member types/paths rather than merely checksums.
    exporter.check_parts(index_path)
    assert exporter.main(["verify", str(index_path)]) != 0
    destination = exporter.ROOT / "hostile-restore"
    assert exporter.main(["unpack", str(index_path), "--output", str(destination)]) != 0
    assert not destination.exists()
    assert not (exporter.ROOT / "escape.txt").exists()


def recovery_origin(exporter, run):
    origin = exporter.ROOT / "runs" / "original-portfolio"
    (origin / "train-A-000").mkdir(parents=True)
    (origin / "train-A-000" / "trials.csv").write_bytes(b"loss,status\r\n0.23,FAILED\r\n")
    (origin / "portfolio-workflow.json").write_text("{}\n")
    stage_paths = {"train-A-000": str(origin / "train-A-000")}
    bridge = {
        "origin_workflow_path": str(origin / "portfolio-workflow.json"),
        "origin_roots": [str(origin)],
        "stage_paths": stage_paths,
        "resume_paths": {},
    }
    recovery = run / "recovery.json"
    recovery.write_text(json.dumps(bridge))
    descriptor = {
        "manifest_path": str(recovery),
        "sha256": hashlib.sha256(recovery.read_bytes()).hexdigest(),
        "stage_paths": stage_paths,
        "resume_paths": {},
    }
    (run / "portfolio-workflow.json").write_text(json.dumps({"recovery": descriptor}))
    return origin, bridge, descriptor


def test_verified_recovery_closure_keeps_original_cost_and_failure_evidence(exporter, run):
    origin, _, _ = recovery_origin(exporter, run)
    original_bytes = snapshot(origin)
    _, index_path, _ = pack(exporter, run)
    destination = exporter.ROOT / "recovery-restored"
    assert exporter.main(["unpack", str(index_path), "--output", str(destination)]) == 0
    assert snapshot(restored_run(destination, origin)) == original_bytes
    assert snapshot(restored_run(destination, run)) == snapshot(run)


@pytest.mark.parametrize("defect", ["missing-origin", "tampered-bridge", "descriptor-path-mismatch"])
def test_unverifiable_recovery_closure_blocks_export(exporter, run, defect):
    origin, bridge, descriptor = recovery_origin(exporter, run)
    if defect == "missing-origin":
        origin.rename(origin.with_name("hidden-origin"))
    elif defect == "tampered-bridge":
        bridge["origin_roots"] = []
        (run / "recovery.json").write_text(json.dumps(bridge))
    else:
        descriptor["stage_paths"] = {}
        (run / "portfolio-workflow.json").write_text(json.dumps({"recovery": descriptor}))
    destination = exporter.ROOT / "bad-recovery-export"
    assert exporter.main(["pack", str(run), "--output", str(destination)]) != 0
    assert not destination.exists()


@pytest.mark.parametrize("relative", ["logs/train.out", "models/selected.pt"])
def test_mid_export_changes_fail_without_publishing_partial_bundle(exporter, run, monkeypatch, relative):
    original_addfile = exporter.tarfile.TarFile.addfile
    changed = False

    def mutate_after_read(archive, member, fileobj=None):
        nonlocal changed
        result = original_addfile(archive, member, fileobj)
        # Mutate after the inventory is sealed. The checkpoint is omitted from
        # review, but its provenance must still describe the same source bytes.
        if member.name.endswith("/logs/train.out") and not changed:
            path = run / relative
            path.write_bytes(path.read_bytes() + b"changed during export")
            changed = True
        return result

    monkeypatch.setattr(exporter.tarfile.TarFile, "addfile", mutate_after_read)
    destination = exporter.ROOT / "mutating-export"
    assert exporter.main(["pack", str(run), "--output", str(destination)]) != 0
    assert changed
    assert not destination.exists()
    assert not list(exporter.ROOT.glob("mutating-export.partial-*"))


@pytest.mark.parametrize("command", ["plan", "pack"])
def test_unreadable_directory_fails_instead_of_silently_omitting_evidence(exporter, run, monkeypatch, command):
    original_scandir = exporter.os.scandir

    def unreadable(path):
        if Path(path) == run / "report":
            raise PermissionError("review evidence is unreadable")
        return original_scandir(path)

    monkeypatch.setattr(exporter.os, "scandir", unreadable)
    destination = exporter.ROOT / "unreadable-export"
    options = ["--output", str(destination)] if command == "pack" else []
    assert exporter.main([command, str(run), *options]) != 0
    assert not destination.exists()


def test_recovery_change_between_discovery_and_inventory_cannot_omit_new_ancestor(exporter, run, monkeypatch):
    _, bridge, descriptor = recovery_origin(exporter, run)
    new_origin = exporter.ROOT / "runs" / "newly-referenced-origin"
    new_origin.mkdir()
    (new_origin / "costs.csv").write_bytes(b"seconds,status\r\n45,FAILED\r\n")
    original_roots = exporter.run_roots
    changed = False

    def add_ancestor_after_discovery(value):
        nonlocal changed
        selected = original_roots(value)
        if not changed:
            # A concurrently refreshed bridge is internally consistent and its
            # new bytes are captured by inventory. Only rediscovering closure
            # can detect that this first selected root list misses an ancestor.
            bridge["origin_roots"].append(str(new_origin))
            recovery = run / "recovery.json"
            recovery.write_text(json.dumps(bridge))
            descriptor["sha256"] = hashlib.sha256(recovery.read_bytes()).hexdigest()
            (run / "portfolio-workflow.json").write_text(json.dumps({"recovery": descriptor}))
            changed = True
        return selected

    monkeypatch.setattr(exporter, "run_roots", add_ancestor_after_discovery)
    destination = exporter.ROOT / "changed-ancestry-export"
    assert exporter.main(["pack", str(run), "--output", str(destination)]) != 0
    assert changed
    assert not destination.exists()
    assert not list(exporter.ROOT.glob("changed-ancestry-export.partial-*"))
