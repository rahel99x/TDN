"""Sealed evidence for the separately declared bounded consistency program."""
from __future__ import annotations

from pathlib import Path
import re

from tdn.analysis.premix import artifacts as shared
from tdn.runtime.metadata import write_json

digest, file_digest = shared.digest, shared.file_digest
STAGES = ("audit", "prepare", "neural")
_SHA = re.compile(r"[a-f0-9]{64}\Z")
_REQUIRED = {
    "audit": {"protocol.json", "summary.json", "execution.json", "checks.json"},
    "prepare": shared._REQUIRED["prepare"],
    "neural": shared._REQUIRED["neural"] | {"comparisons.json", "summary.txt", "consistency.json", "target_coverage.json"},
}


def _labels(stage, profile, source):
    if stage not in STAGES or profile not in ("smoke", "full"):
        raise ValueError("Unknown consistency stage/profile")
    if not isinstance(source, str) or not _SHA.fullmatch(source):
        raise ValueError("Consistency source_tree_sha256 requires a SHA256 fingerprint")


def _validate(root, manifest):
    from .protocol import validate_protocol
    stage = manifest["stage"]
    protocol = shared._read(root, "protocol.json")
    validate_protocol(protocol)
    if protocol["profile"] != manifest["profile"] or digest(protocol) != manifest["protocol_sha256"]:
        raise ValueError("Consistency protocol differs from its seal")
    execution = shared._read(root, "execution.json")
    expected = {"stage": stage, "profile": manifest["profile"], "protocol_sha256": digest(protocol)}
    if (any(execution.get(key) != value for key, value in expected.items())
            or execution.get("software", {}).get("source_tree_sha256") != manifest["source_tree_sha256"]):
        raise ValueError("Consistency execution identity differs from its seal")
    mode, device = execution.get("execution_mode"), execution.get("device")
    if (mode not in ("local-cpu", "desktop-slurm") or device not in ("cpu", "cuda")
            or (stage != "neural" and device != "cpu")
            or (device == "cuda" and mode != "desktop-slurm")):
        raise ValueError("Consistency execution device/mode is inconsistent")
    if mode == "desktop-slurm" and not _SHA.fullmatch(str(execution.get("slurm_profile_sha256", ""))):
        raise ValueError("Consistency Slurm execution requires its profile fingerprint")
    if stage != "audit":
        # Dataset/checkpoint closure and exact planned coverage are the same
        # evidence rules as the earlier experiment, with a distinct protocol.
        shared._validate_payload(root, manifest)
        if not _SHA.fullmatch(str(execution.get("audit_manifest_sha256", ""))):
            raise ValueError("Consistency science requires the preceding audit fingerprint")
        if stage == "neural":
            checks = shared._rows(root, "consistency.json")
            cases = ("constant", "zero_reaction", "zero_diffusion", "zero_horizon")
            expected = {(seed, family, case) for seed in protocol["seeds"]
                        for family in protocol["families"] for case in cases}
            if shared._identities(checks, ("seed", "family", "case")) != expected:
                raise ValueError("Selected-checkpoint consistency coverage is incomplete")
            records = {(row["seed"], row["family"]): row for row in shared._rows(root, "training.json")}
            for row in checks:
                required = row["family"].endswith(("_gated", "_moment")) or row["case"] == "zero_horizon"
                trained = records[row["seed"], row["family"]]
                if (type(row.get("required")) is not bool or row["required"] != required
                        or row.get("after_checkpoint_freeze") is not True
                        or row.get("diagnostic_parents_used") is not False
                        or row.get("checkpoint_selection") != trained.get("selection")):
                    raise ValueError("Selected-checkpoint consistency identity differs")
                if trained["status"] == "COMPLETED" and required and row.get("status") != "PASS":
                    raise ValueError("Required selected-checkpoint consistency check did not pass")
                if trained["status"] != "COMPLETED" and row.get("status") != "TRAINING_FAILED":
                    raise ValueError("Failed training cannot supply a consistency pass")
            freeze = shared._read(root, "checkpoint_freeze.json")
            if freeze.get("diagnostic_parent_ids") != [p["parent_id"] for p in protocol["parents"] if p["split"] == "diagnostic"]:
                raise ValueError("Frozen checkpoint diagnostic cohort differs")
        return
    summary = shared._read(root, "summary.json")
    if (summary.get("status") != "COMPLETED" or summary.get("stage") != "audit"
            or summary.get("profile") != manifest["profile"] or summary.get("device") != "cpu"):
        raise ValueError("Consistency audit did not complete")
    rows = shared._rows(root, "checks.json")
    identifiers = shared._identities(rows, ("check_id",))
    if identifiers != {(value,) for value in protocol["audit_case_ids"]}:
        raise ValueError("Consistency audit declared coverage is incomplete")
    if summary.get("coverage") != {"expected": len(rows), "reported": len(rows)}:
        raise ValueError("Consistency audit coverage summary differs")
    required = set(protocol["audit_required_case_ids"])
    if any(type(row.get("required")) is not bool or row["required"] != (row["check_id"] in required) for row in rows):
        raise ValueError("Consistency audit required checks differ from the protocol")
    if not rows or any(row.get("required") and row.get("status") != "PASS" for row in rows):
        raise ValueError("Required consistency checks did not pass")


def seal_stage(run_dir, *, stage, profile, source_tree_sha256):
    root = Path(run_dir)
    _labels(stage, profile, source_tree_sha256)
    if (root / "manifest.json").exists() or (root / "COMPLETED").exists():
        raise ValueError("Preserve existing consistency seals")
    if shared._read(root, "summary.json").get("status") != "COMPLETED":
        raise ValueError("Incomplete consistency stages cannot be sealed")
    files = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if "tower" in relative.parts or relative.as_posix() in ("stage.json", "manifest.json", "COMPLETED"):
            continue
        if path.is_symlink():
            raise ValueError("Consistency artifacts cannot be symlinks")
        if path.is_file():
            if ".partial" in path.suffixes:
                raise ValueError("Unfinished consistency artifact remains")
            files[relative.as_posix()] = file_digest(path)
    if not _REQUIRED[stage] <= files.keys():
        raise ValueError("Consistency artifact inventory is incomplete")
    manifest = {"version": 1, "benchmark_suite": "consistency", "stage": stage, "profile": profile,
                "source_tree_sha256": source_tree_sha256, "protocol_sha256": digest(shared._read(root, "protocol.json")),
                "files": files}
    _validate(root, manifest)
    write_json(root / "manifest.json", manifest)
    (root / "COMPLETED").write_text(file_digest(root / "manifest.json") + "\n")
    return manifest


def verify_stage(run_dir, *, stage=None, profile=None, source_tree_sha256=None):
    root = Path(run_dir)
    path = shared._artifact(root, "manifest.json")
    if shared._artifact(root, "COMPLETED").read_text().strip() != file_digest(path):
        raise ValueError("Consistency completion marker differs from its manifest")
    manifest = shared._read(root, "manifest.json")
    if type(manifest.get("version")) is not int or manifest["version"] != 1 or manifest.get("benchmark_suite") != "consistency":
        raise ValueError("Unsupported consistency manifest")
    _labels(manifest.get("stage"), manifest.get("profile"), manifest.get("source_tree_sha256"))
    for key, value in (("stage", stage), ("profile", profile), ("source_tree_sha256", source_tree_sha256)):
        if value is not None and manifest.get(key) != value:
            raise ValueError(f"Consistency predecessor {key} differs")
    files = manifest.get("files")
    if not isinstance(files, dict) or not _REQUIRED[manifest["stage"]] <= files.keys():
        raise ValueError("Consistency artifact inventory is incomplete")
    for relative, expected in files.items():
        if not isinstance(expected, str) or not _SHA.fullmatch(expected):
            raise ValueError("Consistency artifact fingerprint is not SHA256")
        if file_digest(shared._artifact(root, relative)) != expected:
            raise ValueError(f"Consistency artifact changed: {relative}")
    _validate(root, manifest)
    return manifest
