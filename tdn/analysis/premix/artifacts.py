"""Sealed stage artifacts shared by the bounded premix program.

Scientific stage outputs are immutable after completion. Tower exports and
stage lifecycle records are separate, so reporting can be refreshed safely.
Hashes detect accidental changes; these local manifests are not signatures.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from tdn.runtime.metadata import write_json

STAGES = ("accuracy", "scaling", "prepare", "neural")
_SHA = re.compile(r"[a-f0-9]{64}\Z")
_BASE = {"protocol.json", "summary.json", "execution.json"}
_REQUIRED = {
    "accuracy": _BASE | {"config.json", "metrics.json", "summary.txt"},
    "scaling": _BASE | {"config.json", "metrics.json", "summary.txt"},
    "prepare": _BASE | {"dataset.pt", "dataset_manifest.json", "normalization.json", "references.json"},
    "neural": _BASE | {"training.json", "candidates.json", "frontiers.json", "checkpoint_freeze.json",
                         "neural_manifest.json", "dataset_source.json", "memory.json"},
}
_TABLE_IDS = {"candidate_rows": "candidate_id", "frontier_rows": "frontier_id",
              "reference_rows": "reference_id", "parity_rows": "parity_id"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _artifact(root, relative):
    if not isinstance(relative, str):
        raise ValueError("Unsafe premix artifact path")
    part = Path(relative)
    if part.is_absolute() or ".." in part.parts:
        raise ValueError("Unsafe premix artifact path")
    path = root / part
    if (not path.resolve().is_relative_to(root.resolve()) or not path.is_file()
            or any((root / Path(*part.parts[:index])).is_symlink() for index in range(1, len(part.parts) + 1))):
        raise ValueError(f"Missing or unsafe premix artifact: {relative}")
    return path


def _read(root, relative):
    value = json.loads(_artifact(root, relative).read_text())
    # Python's permissive JSON reader accepts NaN/Infinity, which cannot form
    # canonical scientific evidence even if their file bytes were hashed.
    digest(value)
    if not isinstance(value, dict):
        raise ValueError(f"Premix {relative} must be a JSON object")
    return value


def _labels(stage, profile, source):
    if stage not in STAGES or profile not in ("smoke", "full"):
        raise ValueError("Unknown premix stage/profile")
    if not isinstance(source, str) or not _SHA.fullmatch(source):
        raise ValueError("Premix source_tree_sha256 must be a SHA256 fingerprint")


def _rows(root, filename):
    rows = _read(root, filename).get("rows")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Premix {filename} requires a row list")
    return rows


def _identities(rows, keys):
    identities = [tuple(row.get(key) for key in keys) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Premix scientific table contains duplicate identities")
    return set(identities)


def _validate_payload(root, manifest):
    """Check the stage envelope, declared coverage and inner artifact closure."""
    stage, profile, source = (manifest[key] for key in ("stage", "profile", "source_tree_sha256"))
    files = manifest["files"]
    if not _REQUIRED[stage] <= files.keys():
        raise ValueError("Premix stage artifact inventory is incomplete")
    protocol, execution, summary = (_read(root, name) for name in
                                    ("protocol.json", "execution.json", "summary.json"))
    protocol_hash = digest(protocol)
    if protocol_hash != manifest["protocol_sha256"]:
        raise ValueError("Premix protocol differs from its seal")
    expected = {"stage": stage, "profile": profile, "protocol_sha256": protocol_hash}
    if any(execution.get(key) != value for key, value in expected.items()):
        raise ValueError("Premix execution stage/profile/protocol differs from its seal")
    if execution.get("software", {}).get("source_tree_sha256") != source:
        raise ValueError("Premix execution source differs from its seal")
    device, mode = execution.get("device"), execution.get("execution_mode")
    if (device not in ("cpu", "cuda") or (stage != "neural" and device != "cpu")
            or mode not in ("local-cpu", "carc") or (device == "cuda" and mode != "carc")):
        raise ValueError("Premix execution device/mode is inconsistent with its stage")
    if summary.get("status") != "COMPLETED" or summary.get("profile") != profile:
        raise ValueError("Premix summary did not complete the declared profile")
    if stage in ("accuracy", "scaling"):
        if summary.get("panel") != stage:
            raise ValueError("Premix numerical summary panel differs")
        metrics = _read(root, "metrics.json")
        if metrics.get("status") != "COMPLETED" or metrics.get("profile") != profile or metrics.get("panel") != stage:
            raise ValueError("Premix canonical numerical status/profile/panel differs")
        expected_ids = protocol.get("plans", {}).get("expected_ids")
        coverage = summary.get("coverage")
        for table, identity in _TABLE_IDS.items():
            rows = metrics.get(table)
            if not isinstance(rows, list):
                raise ValueError("Premix numerical canonical table is missing")
            actual = _identities(rows, (identity,))
            if expected_ids is not None and actual != {(value,) for value in expected_ids.get(table, [])}:
                raise ValueError("Premix numerical declared coverage is incomplete")
            if coverage is not None and coverage.get(table) != {"expected": len(rows), "reported": len(rows)}:
                raise ValueError("Premix numerical coverage summary differs from its tables")
        if protocol.get("config") is not None and _read(root, "config.json") != protocol["config"]:
            raise ValueError("Premix numerical configuration differs from its protocol")
    else:
        if summary.get("stage") != ("dataset" if stage == "prepare" else "neural") or summary.get("device") != device:
            raise ValueError("Premix summary stage/device differs from execution")
        inner_name = "dataset_manifest.json" if stage == "prepare" else "neural_manifest.json"
        inner = _read(root, inner_name)
        if inner.get("protocol_sha256") != protocol_hash or not isinstance(inner.get("artifacts"), dict):
            raise ValueError("Premix inner artifact manifest belongs to another protocol")
        inner_required = (_REQUIRED[stage] - {"execution.json", inner_name})
        if not inner_required <= inner["artifacts"].keys():
            raise ValueError("Premix inner artifact inventory is incomplete")
        for name, fingerprint in inner["artifacts"].items():
            if files.get(name) != fingerprint:
                raise ValueError("Premix inner artifact inventory differs from outer seal")
        if stage == "prepare" and "parents" in protocol:
            references = _rows(root, "references.json")
            expected_references = {(parent["parent_id"], horizon)
                for parent in protocol["parents"] for horizon in parent["horizons"]}
            if (_identities(references, ("parent_id", "horizon")) != expected_references
                    or any(row.get("accepted") is not True for row in references)):
                raise ValueError("Premix dataset references are incomplete or unaccepted")
            normalization = _read(root, "normalization.json")
            train_ids = [parent["parent_id"] for parent in protocol["parents"] if parent["split"] == "train"]
            if (normalization.get("parent_ids") != train_ids or normalization.get("split") != "train"
                    or normalization.get("protocol_sha256") != protocol_hash):
                raise ValueError("Premix normalization does not identify the training-only fit")
        if stage == "neural":
            training, candidates, frontiers = (_rows(root, name) for name in
                                              ("training.json", "candidates.json", "frontiers.json"))
            freeze = _read(root, "checkpoint_freeze.json")
            if (freeze.get("protocol_sha256") != protocol_hash or freeze.get("selection_split") != "validation"
                    or freeze.get("all_training_completed_before_diagnostics") is not True
                    or not isinstance(freeze.get("artifacts"), dict)):
                raise ValueError("Premix checkpoints were not frozen using validation before diagnostics")
            if any(row.get("status") not in ("COMPLETED", "NUMERICAL_FAILURE") for row in training):
                raise ValueError("Premix training attempts did not all finish")
            expected_checkpoints = {f"checkpoints/seed-{row['seed']}-{row['family']}.pt": row.get("checkpoint_sha256")
                for row in training if row["status"] == "COMPLETED"}
            if freeze["artifacts"] != expected_checkpoints:
                raise ValueError("Premix completed training checkpoints differ from the frozen set")
            for name, fingerprint in freeze["artifacts"].items():
                if files.get(name) != fingerprint or inner["artifacts"].get(name) != fingerprint:
                    raise ValueError("Premix frozen checkpoint differs from artifact inventories")
            if all(key in protocol for key in ("seeds", "families", "classical", "parents", "step_counts", "long_step_counts", "targets")):
                pairs = {(seed, family) for seed in protocol["seeds"] for family in protocol["families"]}
                if _identities(training, ("seed", "family")) != pairs:
                    raise ValueError("Premix training coverage differs from the declared seed/family plan")
                methods = pairs | {(None, family) for family in protocol["classical"]}
                diagnostic = [parent for parent in protocol["parents"] if parent["split"] == "diagnostic"]
                expected_candidates = {(parent["parent_id"], seed, family, steps)
                    for parent in diagnostic for seed, family in methods
                    for steps in protocol["long_step_counts"] if parent["regime"] == "long_rollout"}
                expected_candidates.update((parent["parent_id"], seed, family, steps)
                    for parent in diagnostic if parent["regime"] != "long_rollout" for seed, family in methods
                    for steps in protocol["step_counts"])
                if _identities(candidates, ("parent_id", "seed", "family", "steps")) != expected_candidates:
                    raise ValueError("Premix diagnostic candidate coverage differs from the declared plan")
                expected_frontiers = {(parent["parent_id"], seed, family, norm, target)
                    for parent in diagnostic for seed, family in methods for norm in ("rms", "max") for target in protocol["targets"]}
                if _identities(frontiers, ("parent_id", "seed", "family", "norm", "target")) != expected_frontiers:
                    raise ValueError("Premix frontier coverage differs from the declared plan")


def seal_stage(run_dir, *, stage, profile, source_tree_sha256):
    root = Path(run_dir)
    _labels(stage, profile, source_tree_sha256)
    if (root / "COMPLETED").exists() or (root / "manifest.json").exists():
        raise ValueError("Preserve existing premix seals")
    if _read(root, "summary.json").get("status") != "COMPLETED":
        raise ValueError("Incomplete premix stages cannot be sealed")
    files = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if "tower" in relative.parts or relative.as_posix() in ("stage.json", "manifest.json", "COMPLETED"):
            continue
        if path.is_symlink():
            raise ValueError("Premix artifacts cannot be symlinks")
        if path.is_file():
            if ".partial" in path.suffixes:
                raise ValueError("Unfinished premix artifact remains")
            files[relative.as_posix()] = file_digest(path)
    manifest = {"version": 1, "benchmark_suite": "premix", "stage": stage,
                "profile": profile, "source_tree_sha256": source_tree_sha256,
                "protocol_sha256": digest(_read(root, "protocol.json")), "files": files}
    _validate_payload(root, manifest)
    write_json(root / "manifest.json", manifest)
    (root / "COMPLETED").write_text(file_digest(root / "manifest.json") + "\n")
    return manifest


def verify_stage(run_dir, *, stage=None, profile=None, source_tree_sha256=None):
    root = Path(run_dir)
    path = _artifact(root, "manifest.json")
    if _artifact(root, "COMPLETED").read_text().strip() != file_digest(path):
        raise ValueError("Premix completion marker differs from its manifest")
    manifest = _read(root, "manifest.json")
    if type(manifest.get("version")) is not int or manifest["version"] != 1 or manifest.get("benchmark_suite") != "premix":
        raise ValueError("Unsupported premix manifest")
    _labels(manifest.get("stage"), manifest.get("profile"), manifest.get("source_tree_sha256"))
    for name, expected in (("stage", stage), ("profile", profile), ("source_tree_sha256", source_tree_sha256)):
        if expected is not None and manifest.get(name) != expected:
            raise ValueError(f"Premix predecessor {name} differs")
    files = manifest.get("files")
    if not isinstance(files, dict) or not _REQUIRED[manifest["stage"]] <= files.keys():
        raise ValueError("Premix manifest has an incomplete inventory")
    for relative, expected in files.items():
        if not isinstance(expected, str) or not _SHA.fullmatch(expected):
            raise ValueError("Premix artifact fingerprint is not SHA256")
        if file_digest(_artifact(root, relative)) != expected:
            raise ValueError(f"Premix artifact changed: {relative}")
    _validate_payload(root, manifest)
    return manifest
