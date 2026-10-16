"""Pinned, read-only access to the inspected P1 bank for a bounded follow-up.

Preparation hashes the committed archive and provenance; it does not parse any
numeric array. Workers copy only provenance and arrays they actually consume.
Protected targets remain unreadable until a persisted selection is sealed.
Nothing in the historical archive is imported or executed; no git history is
needed to establish the pinned input identity.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import zipfile

import numpy as np


ARCHIVE_RELATIVE = "results/native-response-preconditioner-scaling-2944/native-review.zip"
INDEX_RELATIVE = "results/native-response-preconditioner-scaling-2944/upload-index.json"
PINNED = dict(
    archive_sha256="caaacb862df0e272763340b69f523e66884c40e32013cd1ec6f0ff46eb89b6fb",
    index_sha256="43e44377ffb99f0ac4c056132b551e17736b61f35af26352b1650ec1692480b3",
    archive_bytes=91404958, index_bytes=1350584, file_count=7757,
    uncompressed_bytes=320826795,
    run_id="launch-0e144975eb4b433492fe55d75d209ef2-a1",
    protocol_sha256="bda3d88005a242cb7c97a4ab8e75c71ff76aad99f7e302dd4988003219e47e64",
    source_fingerprint="53758e02afc69d1e665caed9f971ae8735ba8830f5c7487ec904f6395101e1bc",
    study_files=7717, query_count=2688, parent_count=48,
)
SCHEMA = "tdn.response_preconditioner.regularization.inputs.v1"
CHUNK = 1 << 20
INDEX_LIMIT = 16 << 20
MEMBER_LIMIT = 32 << 20
INPUT_LIMIT = 96 << 20
STUDY_PREFIX = "outputs/study/"
PROVENANCE = ("protocol.json", "source.json", "source_after.json", "summary.json", "artifacts.json")
QUERY_SHAPES = dict(initial=(1, 1, 5, 5), source=(1, 1, 5, 5), teacher=(1, 1, 5, 5),
    refined=(1, 1, 5, 5), matched_mean=(1, 1, 5, 5), matched_mean_refined=(1, 1, 5, 5),
    base=(1, 1, 5, 5), branches=(5, 1, 1, 5, 5), features=(8,), design=(40, 1, 1, 5, 5))
PREDICTION_METHODS = tuple(f"fit-p{p:03d}-s{s:02d}" for p in (8, 16, 32) for s in (2, 4, 8)) + (
    "ordered_2", "ordered_8", "lawson_2", "lawson_4", "lawson_8")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "Duplicate retained JSON key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError("Nonfinite retained JSON value: " + value)

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def _relative(name):
    _require(isinstance(name, str) and name and len(name) < 4096, "Invalid evidence path")
    path = PurePosixPath(name)
    _require(not path.is_absolute() and ".." not in path.parts and "\\" not in name
             and "\x00" not in name and path.as_posix() == name,
             "Unsafe evidence path: " + name)
    return path


def _confined(path, root):
    path, root = Path(path).absolute(), Path(root).absolute()
    _require(path.is_relative_to(root) and ".." not in path.parts, "Evidence path is outside repository")
    for component in (path, *path.parents):
        _require(not component.is_symlink(), "Evidence paths cannot traverse symlinks")
    return path


def _signature(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _read_file(path, root, limit):
    path = _confined(path, root)
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        _require(stat.S_ISREG(before.st_mode) and before.st_size <= limit, "Evidence must be a bounded regular file")
        data = stream.read(limit + 1)
        _require(len(data) == before.st_size and _signature(before) == _signature(os.fstat(stream.fileno()))
                 == _signature(path.stat(follow_symlinks=False)), "Evidence file changed while reading")
    return data


class _Archive:
    """Stable descriptor for the exact reviewed ZIP, with bounded member reads."""

    def __init__(self, repository):
        self.root = Path(repository).absolute()
        self.stream = self.bundle = None
        try:
            index_raw = _read_file(self.root / INDEX_RELATIVE, self.root, INDEX_LIMIT)
            _require(len(index_raw) == PINNED["index_bytes"] and _sha(index_raw) == PINNED["index_sha256"],
                     "Index differs from the pinned P1 review")
            self.index = _json(index_raw)
            _require(all(self.index[key] == PINNED[key] for key in
                         ("run_id", "archive_sha256", "archive_bytes", "file_count", "uncompressed_bytes")),
                     "Retained P1 archive identity mismatch")
            self.rows = {}
            for row in self.index["files"]:
                name = _relative(row["path"]).as_posix()
                _require(name not in self.rows and type(row["bytes"]) is int and row["bytes"] >= 0
                         and isinstance(row["sha256"], str) and len(row["sha256"]) == 64,
                         "Invalid or duplicate retained inventory row")
                self.rows[name] = row
            _require(len(self.rows) == PINNED["file_count"]
                     and sum(r["bytes"] for r in self.rows.values()) == PINNED["uncompressed_bytes"],
                     "Retained P1 file roster mismatch")
            self.path = _confined(self.root / ARCHIVE_RELATIVE, self.root)
            descriptor = os.open(self.path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
            self.stream = os.fdopen(descriptor, "rb")
            before = os.fstat(self.stream.fileno())
            _require(stat.S_ISREG(before.st_mode) and before.st_size == PINNED["archive_bytes"],
                     "Archive must have the pinned regular-file size")
            self.signature = _signature(before)
            whole = hashlib.sha256()
            for part in self.index["parts"]:
                _require(type(part["bytes"]) is int and 0 < part["bytes"] <= PINNED["archive_bytes"], "Invalid archive part size")
                remaining, digest = part["bytes"], hashlib.sha256()
                while remaining:
                    block = self.stream.read(min(CHUNK, remaining))
                    _require(bool(block), "Truncated archive part")
                    remaining -= len(block)
                    digest.update(block)
                    whole.update(block)
                _require(digest.hexdigest() == part["sha256"], "Retained archive part hash mismatch")
            _require(not self.stream.read(1) and whole.hexdigest() == PINNED["archive_sha256"],
                     "Archive differs from the pinned P1 review")
            self.stable()
            self.stream.seek(0)
            self.bundle = zipfile.ZipFile(self.stream)
            entries = self.bundle.infolist()
            _require(len(entries) == PINNED["file_count"], "ZIP member roster mismatch")
            self.entries = {}
            for entry in entries:
                name = _relative(entry.filename).as_posix()
                prefix = PINNED["run_id"] + "/"
                _require(name.startswith(prefix), "Unexpected retained ZIP root")
                relative = name[len(prefix):]
                _require(relative in self.rows and relative not in self.entries and not entry.is_dir()
                         and not entry.flag_bits & 1 and stat.S_IFMT(entry.external_attr >> 16) in (0, stat.S_IFREG)
                         and entry.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                         and entry.file_size == self.rows[relative]["bytes"],
                         "Unsupported or mismatched ZIP member: " + name)
                self.entries[relative] = entry
            _require(self.entries.keys() == self.rows.keys(), "Incomplete retained ZIP roster")
            self.metadata_raw = {name: self.read(STUDY_PREFIX + name) for name in PROVENANCE}
            self.metadata = {name: _json(raw) for name, raw in self.metadata_raw.items()}
            manifest = self.metadata["artifacts.json"]
            self.study_rows = {}
            for row in manifest["files"]:
                name = _relative(row["path"]).as_posix()
                _require(name not in self.study_rows and STUDY_PREFIX + name in self.rows,
                         "Duplicate or unknown study artifact")
                exported = self.rows[STUDY_PREFIX + name]
                _require(row["bytes"] == exported["bytes"] and row["sha256"] == exported["sha256"],
                         "Study manifest differs from pinned upload inventory")
                self.study_rows[name] = row
            _require(len(self.study_rows) == PINNED["study_files"], "Study artifact roster mismatch")
            source, after = self.metadata["source.json"], self.metadata["source_after.json"]
            protocol, summary = self.metadata["protocol.json"], self.metadata["summary.json"]
            _require(source == after and source["fingerprint"] == PINNED["source_fingerprint"]
                     == _sha(_canonical(source["files"])), "Unexpected or changed P1 numerical source")
            _require(_sha(_canonical(protocol)) == PINNED["protocol_sha256"]
                     == summary["protocol_sha256"] and summary["source_fingerprint"] == PINNED["source_fingerprint"],
                     "Unexpected retained P1 protocol/source binding")
            _require(summary["status"] == "COMPLETED" and summary["scientific_promotion"] is False
                     and summary["device"] == "cpu" and summary["reference_count"] == PINNED["query_count"],
                     "Retained P1 study is not the completed bounded CPU bank")
            self.stable()
        except BaseException:
            self.close(check=False)
            raise

    def stable(self):
        _require(self.stream is not None, "Retained archive reader is closed")
        _confined(self.path, self.root)
        _require(self.signature == _signature(os.fstat(self.stream.fileno()))
                 == _signature(self.path.stat(follow_symlinks=False)), "Retained archive changed during access")

    def read(self, relative):
        self.stable()
        _relative(relative)
        _require(relative in self.entries, "Unknown retained file")
        row, entry = self.rows[relative], self.entries[relative]
        _require(entry.file_size <= MEMBER_LIMIT, "Retained member exceeds bounded read allowance")
        with self.bundle.open(entry) as stream:
            content = stream.read(entry.file_size + 1)
        _require(len(content) == row["bytes"] and _sha(content) == row["sha256"],
                 "Retained member differs from its pinned hash: " + relative)
        self.stable()
        return content

    def identity(self):
        return dict(schema=SCHEMA, archive_path=ARCHIVE_RELATIVE, index_path=INDEX_RELATIVE,
            **deepcopy(PINNED), study_manifest_sha256=_sha(self.metadata_raw["artifacts.json"]),
            numerical_source_manifest_sha256=_sha(self.metadata_raw["source.json"]),
            numerical_study_status="COMPLETED", original_native_status="FAILED",
            development_only=True, uploaded_code_executed=False, numeric_arrays_loaded=False)

    def close(self, *, check=True):
        try:
            if check and self.stream is not None:
                self.stable()
        finally:
            if self.bundle is not None:
                self.bundle.close()
            if self.stream is not None:
                self.stream.close()
            self.stream = self.bundle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close(check=exc_type is None)


def input_identity(repository):
    """Verify immutable P1 identity/provenance without extracting numeric data."""
    with _Archive(repository) as archive:
        return archive.identity()


def _array_archive(content, expected):
    """Reject object arrays and oversized/forged NPY headers before allocation."""
    with zipfile.ZipFile(io.BytesIO(content)) as bundle:
        names = {name + ".npy" for name in expected}
        entries = bundle.infolist()
        _require(len(entries) == len(names) and {x.filename for x in entries} == names
                 and sum(x.file_size for x in entries) <= 128 << 10,
                 "Unexpected or oversized retained array roster")
        for entry in entries:
            _require(not entry.flag_bits & 1 and not entry.is_dir()
                     and stat.S_IFMT(entry.external_attr >> 16) in (0, stat.S_IFREG), "Unsupported retained NPY entry")
            with bundle.open(entry) as stream:
                version = np.lib.format.read_magic(stream)
                _require(version in ((1, 0), (2, 0)), "Unsupported retained NPY version")
                read_header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                               else np.lib.format.read_array_header_2_0)
                shape, fortran, dtype = read_header(stream, max_header_size=4096)
                spec = expected[entry.filename[:-4]]
                _require(shape == spec[0] and dtype == np.dtype(spec[1]) and not dtype.hasobject
                         and not fortran and entry.file_size - stream.tell() == math_product(shape)*dtype.itemsize,
                         "Retained array shape/dtype differs from frozen contract")
    with np.load(io.BytesIO(content), allow_pickle=False) as arrays:
        result = {name: arrays[name].copy() for name in expected}
    _require(all(np.isfinite(value).all() for value in result.values()), "Nonfinite retained array")
    return result


def math_product(shape):
    product = 1
    for value in shape:
        product *= value
    return product


class RetainedEvidence:
    """Access targets through read_query; protected labels require a file seal.

    ``output`` must be a fresh directory inside the repository, conventionally
    ``outputs/study/inputs``. Call this only after the native launch/preflight
    gates. Metadata can be inspected before release; target arrays cannot.
    """

    def __init__(self, repository, output, *, expected_identity):
        self.root = Path(repository).absolute()
        self.output = _confined(output, self.root)
        _require(self.output != self.root and not self.output.exists(), "Retained inputs require a fresh output directory")
        self._archive = _Archive(self.root)
        self._written, self._access, self._prediction_access, self._release = {}, [], [], None
        self._bytes = 0
        try:
            self.identity = self._archive.identity()
            _require(self.identity == expected_identity, "Retained inputs differ from frozen native contract")
            self._queries = _json(self._archive.read(STUDY_PREFIX + "query_index.json"))
            self._parents = _json(self._archive.read(STUDY_PREFIX + "parents.json"))
            self._by_query = {q["query_id"]: q for q in self._queries}
            self._by_parent = {p["parent_id"]: p for p in self._parents}
            _require(len(self._by_query) == len(self._queries) == PINNED["query_count"]
                     and len(self._by_parent) == len(self._parents) == PINNED["parent_count"], "Retained data roster mismatch")
            for q in self._queries:
                _require(q["parent_id"] in self._by_parent and q["split"] in ("train", "validation", "diagnostic")
                         and q["split"] == self._by_parent[q["parent_id"]]["split"]
                         and q["role"] in ("calibration", "withheld")
                         and (q["role"] == "withheld" or q["split"] == "train")
                         and q["array_path"] == "queries/" + q["query_id"] + ".npz"
                         and q["array_path"] in self._archive.study_rows
                         and q["array_sha256"] == self._archive.study_rows[q["array_path"]]["sha256"],
                         "Retained query routing/hash differs from its sealed inventory")
            self.output.mkdir(parents=True, exist_ok=False)
            for name in PROVENANCE + ("query_index.json", "parents.json", "parents.npz"):
                self._retain(name, self._archive.read(STUDY_PREFIX + name))
            self._persist()
        except BaseException:
            self._archive.close(check=False)
            raise

    @property
    def queries(self):
        return deepcopy(self._queries)

    @property
    def parents(self):
        return deepcopy(self._parents)

    def _retain(self, relative, content):
        row = self._archive.rows[STUDY_PREFIX + relative]
        _require(_sha(content) == row["sha256"] and len(content) == row["bytes"], "Input copy differs from retained hash")
        target = _confined(self.output / _relative(relative), self.root)
        if relative in self._written:
            _require(_read_file(target, self.root, MEMBER_LIMIT) == content, "Retained input copy was mutated")
            return
        _require(self._bytes + len(content) <= INPUT_LIMIT, "Portable retained inputs exceed byte allowance")
        target.parent.mkdir(parents=True, exist_ok=True)
        _confined(target, self.root)
        with target.open("xb") as stream:
            stream.write(content)
        self._bytes += len(content)
        self._written[relative] = dict(path=relative, bytes=len(content), sha256=row["sha256"])

    def _persist(self):
        receipt = dict(schema=SCHEMA, identity=self.identity, files=list(self._written.values()),
            retained_bytes=self._bytes, copied_numeric_queries=sum(k.startswith("queries/") for k in self._written),
            accessed_query_ids=list(self._access), protected_release=self._release,
            accessed_prediction_ids=list(self._prediction_access),
            original_archive_copied=False, original_prediction_arrays_copied=bool(self._prediction_access),
            scope="Pinned original P1 provenance and consumed query/prediction arrays only; no uploaded code executed")
        target = _confined(self.output / "input-receipt.json", self.root)
        with target.open("w") as stream:
            stream.write(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False)+"\n")

    def _log_access(self, kind, query_id):
        target = _confined(self.output / "access.jsonl", self.root)
        with target.open("a") as stream:
            stream.write(json.dumps(dict(kind=kind, query_id=query_id,
                protected_released=self._release is not None), sort_keys=True)+"\n")
        if (len(self._access) + len(self._prediction_access)) % 64 == 0:
            self._persist()

    def _check_release(self):
        _require(self._release is not None, "Protected retained truth requires persisted selections")
        path = _confined(self.output.parent / self._release["selection_path"], self.root)
        _require(_signature(path.stat(follow_symlinks=False)) == self._release_signature,
                 "Persisted selection changed after protected release")

    def read_parent(self, parent_id):
        """Initial physical states are deployable inputs, not protected truth."""
        _require(parent_id in self._by_parent, "Unknown retained parent")
        content = self._archive.read(STUDY_PREFIX + "parents.npz")
        self._retain("parents.npz", content)
        arrays = _array_archive(content, {pid: ((1, 1, 5, 5), "float32") for pid in self._by_parent})
        return arrays[parent_id]

    def read_query(self, query_id):
        _require(query_id in self._by_query, "Unknown retained query")
        q = self._by_query[query_id]
        protected = q["split"] == "diagnostic" or (q["split"] == "train" and q["role"] == "withheld")
        _require(not protected or self._release is not None, "Protected retained truth requires persisted selections")
        if protected:
            self._check_release()
        content = self._archive.read(STUDY_PREFIX + q["array_path"])
        self._retain(q["array_path"], content)
        arrays = _array_archive(content, {k: (shape, "float32" if k == "initial" else "float64") for k, shape in QUERY_SHAPES.items()})
        if protected:
            self._check_release()
        self._access.append(query_id)
        self._log_access("query", query_id)
        return dict(metadata=deepcopy(q), arrays=arrays)

    def read_original_selection(self):
        """Historical fitted coefficients are provenance, never a new release."""
        content = self._archive.read(STUDY_PREFIX + "selection.json")
        self._retain("selection.json", content)
        return _json(content)

    def read_predictions(self, query_id):
        """Replay baselines only after the new-run selections are persisted."""
        _require(self._release is not None, "Historical prediction access requires persisted selections")
        self._check_release()
        _require(query_id in self._by_query and self._by_query[query_id]["role"] == "withheld",
                 "Predictions exist only for retained withheld queries")
        relative = "predictions/" + query_id + ".npz"
        _require(relative in self._archive.study_rows, "Prediction is absent from the sealed study inventory")
        content = self._archive.read(STUDY_PREFIX + relative)
        self._retain(relative, content)
        arrays = _array_archive(content, {k: ((1, 1, 5, 5), "float64") for k in PREDICTION_METHODS})
        self._check_release()
        self._prediction_access.append(query_id)
        self._log_access("predictions", query_id)
        return arrays

    def release_protected(self, selection_path, selection_sha256):
        _require(self._release is None, "Protected retained truth already released")
        path = _confined(selection_path, self.root)
        _require(path.parent == self.output.parent and path.name == "selection.json",
                 "Selection seal must be selection.json beside inputs in this same new study directory")
        content = _read_file(path, self.root, MEMBER_LIMIT)
        selection = _json(content)
        _require(_sha(content) == selection_sha256, "Persisted selection differs from its file seal")
        _require(selection.get("schema") == "tdn.response_preconditioner.regularization.selection.v1"
                 and selection.get("all_selections_sealed") is True,
                 "Protected truth requires all selections sealed")
        self._release = dict(selection_path=path.relative_to(self.output.parent).as_posix(),
            selection_sha256=selection_sha256, selection_file_sha256=_sha(content),
            access_count_before_release=len(self._access), source="persisted new-run selection, not historical P1 selection")
        self._release_signature = _signature(path.stat(follow_symlinks=False))
        self._persist()

    def close(self):
        if self._archive.stream is None:
            return
        try:
            for name, row in self._written.items():
                content = _read_file(self.output / name, self.root, MEMBER_LIMIT)
                _require(len(content) == row["bytes"] and _sha(content) == row["sha256"],
                         "Retained input copy was mutated")
            if self._release is not None:
                self._check_release()
            self._persist()
        finally:
            self._archive.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
