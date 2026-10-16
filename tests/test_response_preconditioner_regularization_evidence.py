"""Retained-bank provenance, bounded parsing and protected target access."""
from copy import deepcopy
import io
import json
from pathlib import Path
import stat
import zipfile

import numpy as np
import pytest

from tdn.analysis.response_preconditioner import regularization_evidence as evidence


def _json(value):
    return (json.dumps(value, sort_keys=True)+"\n").encode()


def _npz(values):
    output = io.BytesIO()
    np.savez_compressed(output, **values)
    return output.getvalue()


def _fixture(root, monkeypatch, mutation=None):
    """Private forged miniature archive; production pins have no override API."""
    run = "launch-private-evidence-fixture-a1"
    source = dict(files=[dict(path="tdn/trusted.py", sha256=evidence._sha(b"trusted"))])
    source["fingerprint"] = evidence._sha(evidence._canonical(source["files"]))
    protocol = dict(schema="private.fixture", device="cpu", parent_counts=[1])
    protocol_sha = evidence._sha(evidence._canonical(protocol))
    parents = [dict(parent_id=s+"-parent", split=s) for s in ("train", "validation", "diagnostic")]
    slots = [("train", "calibration"), ("validation", "withheld"), ("train", "withheld"), ("diagnostic", "withheld")]
    files, queries = {}, []
    for split, role in slots:
        qid = split+"-parent-context-00-"+role+"-00"
        arrays = {k: np.ones(shape, dtype=np.float32 if k == "initial" else np.float64)
                  for k, shape in evidence.QUERY_SHAPES.items()}
        if mutation == "wrong-array-shape" and role == "calibration":
            arrays["design"] = np.zeros((100, 100))
        if mutation == "object-array" and role == "calibration":
            arrays["teacher"] = np.full((1, 1, 5, 5), "untrusted", dtype=object)
        content = _npz(arrays)
        path = "queries/"+qid+".npz"
        files[path] = content
        queries.append(dict(query_id=qid, parent_id=split+"-parent", split=split,
            role=role, array_path=path, array_sha256=evidence._sha(content)))
        if role == "withheld":
            files["predictions/"+qid+".npz"] = _npz({k: np.ones((1, 1, 5, 5)) for k in evidence.PREDICTION_METHODS})
    files.update({
        "parents.json": _json(parents),
        "parents.npz": _npz({p["parent_id"]: np.ones((1, 1, 5, 5), dtype=np.float32) for p in parents}),
        "query_index.json": _json(queries),
        "protocol.json": _json(protocol if mutation != "wrong-protocol" else dict(protocol, extra=True)),
        "source.json": _json(source),
        "source_after.json": _json(source if mutation != "changed-source" else dict(source, fingerprint="0"*64)),
        "summary.json": _json(dict(status="COMPLETED", scientific_promotion=False, device="cpu",
            reference_count=4, protocol_sha256=protocol_sha, source_fingerprint=source["fingerprint"])),
        "selection.json": _json(dict(schema="historical.p1", all_selections_sealed=True)),
    })
    manifest = [dict(path=p, bytes=len(raw), sha256=evidence._sha(raw)) for p, raw in files.items()]
    if mutation == "changed-manifest":
        manifest[0]["sha256"] = "0"*64
    files["artifacts.json"] = _json(dict(files=manifest))
    rows = [dict(path="outputs/study/"+p, bytes=len(raw), sha256=evidence._sha(raw)) for p, raw in files.items()]
    destination = root / evidence.ARCHIVE_RELATIVE
    destination.parent.mkdir(parents=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path, raw in files.items():
            name = run+"/outputs/study/"+path
            if mutation == "traversal" and path == "parents.json":
                name = run+"/../outside"
            if mutation == "symlink-member" and path == "parents.json":
                entry = zipfile.ZipInfo(name)
                entry.create_system = 3
                entry.external_attr = (stat.S_IFLNK | 0o777) << 16
                bundle.writestr(entry, raw)
            else:
                if mutation == "changed-array-bytes" and path == queries[0]["array_path"]:
                    raw = raw[:-1] + bytes([raw[-1] ^ 1])
                bundle.writestr(name, raw)
    archive_raw = destination.read_bytes()
    inventory = dict(run_id=run, archive_sha256=evidence._sha(archive_raw), archive_bytes=len(archive_raw),
        file_count=len(rows), uncompressed_bytes=sum(r["bytes"] for r in rows), files=rows,
        parts=[dict(path="private.part001", bytes=len(archive_raw), sha256=evidence._sha(archive_raw))])
    index_raw = _json(inventory)
    (root / evidence.INDEX_RELATIVE).write_bytes(index_raw)
    monkeypatch.setattr(evidence, "PINNED", dict(archive_sha256=inventory["archive_sha256"],
        index_sha256=evidence._sha(index_raw), archive_bytes=len(archive_raw), index_bytes=len(index_raw),
        file_count=len(rows), uncompressed_bytes=inventory["uncompressed_bytes"], run_id=run,
        protocol_sha256=protocol_sha, source_fingerprint=source["fingerprint"],
        study_files=len(manifest), query_count=len(queries), parent_count=len(parents)))
    return queries


def _open(root, monkeypatch, mutation=None):
    queries = _fixture(root, monkeypatch, mutation)
    identity = evidence.input_identity(root)
    output = root / "new-study" / "inputs"
    return evidence.RetainedEvidence(root, output, expected_identity=identity), queries


def _release(reader, *, sealed=True, schema="tdn.response_preconditioner.regularization.selection.v1"):
    path = reader.output.parent / "selection.json"
    path.write_bytes(_json(dict(schema=schema, all_selections_sealed=sealed)))
    reader.release_protected(path, evidence._sha(path.read_bytes()))


def test_prepare_verifies_real_committed_identity_without_arrays_or_extraction(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    def forbidden(*args, **kwargs):
        raise AssertionError("Preparation must not load numeric arrays")
    monkeypatch.setattr(np, "load", forbidden)
    result = evidence.input_identity(root)
    assert result["archive_sha256"] == "caaacb862df0e272763340b69f523e66884c40e32013cd1ec6f0ff46eb89b6fb"
    assert result["query_count"] == 2688 and result["original_native_status"] == "FAILED"
    assert result["numeric_arrays_loaded"] is False and result["uploaded_code_executed"] is False


def test_queries_release_separately_and_retain_only_consumed_portable_arrays(tmp_path, monkeypatch):
    reader, q = _open(tmp_path, monkeypatch)
    with reader:
        assert reader.parents[0]["split"] == "train"
        assert reader.read_parent("train-parent").dtype == np.float32
        for row in q[:2]:
            loaded = reader.read_query(row["query_id"])
            assert loaded["metadata"] == row
            assert loaded["arrays"]["design"].shape == (40, 1, 1, 5, 5)
        for row in q[2:]:
            with pytest.raises(ValueError, match="persisted selections"):
                reader.read_query(row["query_id"])
            assert not (reader.output / row["array_path"]).exists()
        with pytest.raises(ValueError, match="persisted selections"):
            reader.read_predictions(q[1]["query_id"])
        assert reader.read_original_selection()["schema"] == "historical.p1"
        _release(reader)
        for row in q[2:]:
            assert reader.read_query(row["query_id"])["metadata"] == row
        predicted = reader.read_predictions(q[-1]["query_id"])
        assert set(predicted) == set(evidence.PREDICTION_METHODS)
    receipt = json.loads((reader.output / "input-receipt.json").read_text())
    assert receipt["copied_numeric_queries"] == 4
    assert receipt["protected_release"]["access_count_before_release"] == 2
    assert receipt["original_archive_copied"] is False
    assert not list(reader.output.rglob("*.zip"))
    assert len((reader.output / "access.jsonl").read_text().splitlines()) == 5


@pytest.mark.parametrize("mutation", ("wrong-protocol", "changed-source", "changed-manifest", "traversal", "symlink-member"))
def test_forged_pinned_fixture_rejects_invalid_provenance_or_zip_structure(tmp_path, monkeypatch, mutation):
    _fixture(tmp_path, monkeypatch, mutation)
    with pytest.raises(ValueError):
        evidence.input_identity(tmp_path)
    assert not (tmp_path / "new-study").exists()


@pytest.mark.parametrize("target", ("archive", "index"))
def test_unknown_archive_or_index_bytes_fail_closed(tmp_path, monkeypatch, target):
    _fixture(tmp_path, monkeypatch)
    path = tmp_path / (evidence.ARCHIVE_RELATIVE if target == "archive" else evidence.INDEX_RELATIVE)
    content = path.read_bytes()
    path.write_bytes(content[:-1]+bytes([content[-1] ^ 1]))
    with pytest.raises(ValueError, match="pinned|hash mismatch"):
        evidence.input_identity(tmp_path)


@pytest.mark.parametrize("mutation", ("changed-array-bytes", "wrong-array-shape", "object-array"))
def test_array_hash_shape_and_object_payload_rejected_before_use(tmp_path, monkeypatch, mutation):
    reader, queries = _open(tmp_path, monkeypatch, mutation)
    with reader:
        with pytest.raises(ValueError, match="pinned hash|shape/dtype"):
            reader.read_query(queries[0]["query_id"])


def test_frozen_input_identity_fresh_output_and_symlink_confinement(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch)
    identity = evidence.input_identity(tmp_path)
    bad = deepcopy(identity)
    bad["protocol_sha256"] = "0"*64
    with pytest.raises(ValueError, match="frozen native contract"):
        evidence.RetainedEvidence(tmp_path, tmp_path/"bad", expected_identity=bad)
    assert not (tmp_path/"bad").exists()
    existing = tmp_path/"existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="fresh output"):
        evidence.RetainedEvidence(tmp_path, existing, expected_identity=identity)
    link = tmp_path/"link"
    link.symlink_to(existing, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        evidence.RetainedEvidence(tmp_path, link/"inputs", expected_identity=identity)


@pytest.mark.parametrize("sealed,schema", ((False, "tdn.response_preconditioner.regularization.selection.v1"), (True, "historical.p1")))
def test_release_requires_new_all_selected_schema(tmp_path, monkeypatch, sealed, schema):
    reader, q = _open(tmp_path, monkeypatch)
    with reader:
        with pytest.raises(ValueError, match="all selections sealed"):
            _release(reader, sealed=sealed, schema=schema)
        with pytest.raises(ValueError, match="persisted selections"):
            reader.read_query(q[-1]["query_id"])


def test_release_requires_persisted_exact_file_and_cannot_use_historical_selection(tmp_path, monkeypatch):
    reader, q = _open(tmp_path, monkeypatch)
    with reader:
        reader.read_original_selection()
        path = reader.output/"selection.json"
        with pytest.raises(ValueError, match="beside inputs"):
            reader.release_protected(path, evidence._sha(path.read_bytes()))
        path = reader.output.parent/"selection.json"
        path.write_bytes(_json(dict(schema="tdn.response_preconditioner.regularization.selection.v1", all_selections_sealed=True)))
        with pytest.raises(ValueError, match="file seal"):
            reader.release_protected(path, "0"*64)
        exposed = reader.queries
        exposed[-1]["split"] = "train"
        exposed[-1]["role"] = "calibration"
        with pytest.raises(ValueError, match="persisted selections"):
            reader.read_query(q[-1]["query_id"])


def test_mutated_new_selection_blocks_further_protected_access(tmp_path, monkeypatch):
    reader, q = _open(tmp_path, monkeypatch)
    _release(reader)
    path = reader.output.parent/"selection.json"
    path.write_bytes(path.read_bytes()+b" ")
    with pytest.raises(ValueError, match="changed after"):
        reader.read_query(q[-1]["query_id"])
    with pytest.raises(ValueError, match="changed after"):
        reader.close()


def test_mutated_portable_input_copy_and_open_archive_are_detected(tmp_path, monkeypatch):
    reader, q = _open(tmp_path, monkeypatch)
    reader.read_query(q[0]["query_id"])
    path = reader.output/q[0]["array_path"]
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="copy was mutated"):
        reader.read_query(q[0]["query_id"])
    with pytest.raises(ValueError, match="copy was mutated"):
        reader.close()
    second = evidence.RetainedEvidence(tmp_path, tmp_path/"another-study"/"inputs", expected_identity=evidence.input_identity(tmp_path))
    archive = tmp_path/evidence.ARCHIVE_RELATIVE
    with archive.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="archive changed"):
        second.read_query(q[0]["query_id"])
    with pytest.raises(ValueError, match="archive changed"):
        second.close()


def test_retained_read_budgets_apply_before_copy_or_numeric_allocation(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch)
    identity = evidence.input_identity(tmp_path)
    monkeypatch.setattr(evidence, "INPUT_LIMIT", 8)
    with pytest.raises(ValueError, match="byte allowance"):
        evidence.RetainedEvidence(tmp_path, tmp_path/"new-study"/"inputs", expected_identity=identity)
    monkeypatch.setattr(evidence, "MEMBER_LIMIT", 8)
    with pytest.raises(ValueError, match="read allowance"):
        evidence.input_identity(tmp_path)
