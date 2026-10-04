"""Bounded, read-only projections of TDN science artifacts for Tower.

This module never imports Torch, loads a checkpoint, changes a scientific file,
or treats application completion as scientific success. Paths in the inventory
and tables resolve relative to the *report directory*, including ``../`` paths.
The contract itself only names exact files under the report's ``outputs/``.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import time
import uuid
import xml.etree.ElementTree as ET

from tdn.reporting import atomic_json, read_json, register_log

MAX_FILES = 1024
MAX_ENTRIES = 4096
MAX_DEPTH = 12
MAX_READ_BYTES = 16 << 20
MAX_FILE_BYTES = 1 << 20
MAX_INVENTORY_BYTES = 512 << 10
MAX_TABLE_BYTES = 2 << 20
MAX_ROWS = 10000
MAX_SECONDS = 10
MAX_OMISSIONS = 128
EXCLUDED = {".git", ".venv", "__pycache__", ".cache", ".runtime", ".pytest_cache",
            "pytest-work", "pytest_work", "cache", "caches", "passports"}
KINDS = {".json": "json", ".csv": "csv", ".jsonl": "jsonl", ".txt": "text",
         ".log": "log", ".out": "log", ".err": "log", ".stdout": "log",
         ".stderr": "log", ".png": "image", ".svg": "image", ".pdf": "pdf",
         ".pt": "checkpoint", ".pth": "checkpoint", ".npz": "array_archive",
         ".npy": "array", ".yaml": "configuration", ".yml": "configuration",
         ".xml": "test_report", ".md": "text", ".gz": "archive", ".zip": "archive"}
TEXT_KINDS = {"json", "csv", "jsonl", "text", "log", "configuration", "test_report"}
PREFIX = ["source_path", "source_record"]
COLUMNS = {
    "frontier": PREFIX + ["case_id", "parent_id", "split", "family", "h", "sequence", "device",
        "status", "failed", "feasible", "error", "error_upper", "reference_uncertainty",
        "wall_seconds_median", "cuda_event_seconds_median", "peak_allocated_bytes",
        "peak_reserved_bytes", "host_process_peak_rss_bytes", "timing_scope", "reason"],
    "training": PREFIX + ["record_type", "family", "status", "step", "selected_step", "loss",
        "training_loss", "validation_loss", "admissible_validation", "selected_parameters_changed", "reason"],
    "heldout": PREFIX + ["case_id", "parent_id", "family", "h", "device", "status",
        "one_step_rms", "one_step_upper", "one_step_reference_uncertainty", "two_step_rms",
        "two_step_upper", "two_step_reference_uncertainty", "heldout_absolute_rms",
        "heldout_relative_rms_with_noise_floor", "reason"],
    "references": PREFIX + ["case_id", "parent_id", "role", "h", "accepted", "reason", "uncertainty",
        "defect_norm", "substeps", "refinement_substeps", "refinement_differences", "observed_order"],
    "gates": PREFIX + ["case_id", "gate", "passed", "decision", "scope", "reason"],
    "stages": PREFIX + ["stage", "status", "device", "execution_mode", "actually_ran", "elapsed_seconds", "reason"],
    "tests": PREFIX + ["suite", "test", "classname", "status", "elapsed_seconds", "reason"],
}


def _safe_directory(path):
    path = Path(os.path.abspath(path))
    for component in reversed((path, *path.parents)):
        if not stat.S_ISDIR(component.lstat().st_mode):
            raise ValueError(f"directory required; symlinks refused: {component}")
    return path


def _relative(path, report_dir):
    return Path(os.path.relpath(path, report_dir)).as_posix()


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _dict(value):
    return value if isinstance(value, dict) else {}


def _records(value):
    return value if isinstance(value, list) else []


def _pick(value, names):
    return {name: value[name] for name in names if name in value}


class _Budget:
    def __init__(self):
        self.deadline = time.monotonic() + MAX_SECONDS
        self.read_bytes = 0
        self.entries = 0
        self.omissions = []
        self.omission_count = 0

    def omit(self, path, reason):
        self.omission_count += 1
        if len(self.omissions) < MAX_OMISSIONS:
            self.omissions.append({"path": str(path)[:4096], "reason": str(reason)[:300]})

    def expired(self):
        return time.monotonic() >= self.deadline

    def read(self, path, expected):
        if expected.st_size > MAX_FILE_BYTES or self.read_bytes + expected.st_size > MAX_READ_BYTES:
            raise ValueError("per-file or total read budget exceeded")
        if self.expired():
            raise ValueError("analytics time budget exceeded")
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_size != expected.st_size:
                raise ValueError("source is not the inventoried regular file")
            raw = bytearray()
            while len(raw) <= MAX_FILE_BYTES:
                part = os.read(fd, min(65536, MAX_FILE_BYTES + 1 - len(raw)))
                if not part:
                    break
                raw.extend(part)
                self.read_bytes += len(part)
                if self.read_bytes > MAX_READ_BYTES or self.expired():
                    raise ValueError("analytics read/time budget exceeded")
            after = os.fstat(fd)
            named = path.lstat()
            identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
            if (len(raw) != before.st_size or identity(expected) != identity(before)
                    or identity(before) != identity(after) or identity(after) != identity(named)):
                raise ValueError("source changed while reporting")
            return bytes(raw)
        finally:
            os.close(fd)


def _walk(root, report_dir, budget):
    stack = [(root, 0)]
    while stack:
        parent, depth = stack.pop()
        if budget.expired() or budget.entries >= MAX_ENTRIES:
            budget.omit(_relative(parent, report_dir), "traversal time/entry budget exhausted; remaining subtree omitted")
            break
        try:
            _safe_directory(parent)
            with os.scandir(parent) as scan:
                entries = []
                for entry in scan:
                    budget.entries += 1
                    if budget.entries > MAX_ENTRIES:
                        budget.omit(_relative(parent, report_dir), "directory entry budget exhausted")
                        break
                    entries.append(entry)
            for entry in sorted(entries, key=lambda item: item.name):
                path = Path(entry.path)
                relative = _relative(path, report_dir)
                try:
                    info = entry.stat(follow_symlinks=False)
                    if stat.S_ISLNK(info.st_mode):
                        budget.omit(relative, "symlink refused")
                    elif stat.S_ISDIR(info.st_mode):
                        if (path == report_dir or entry.name in EXCLUDED
                                or entry.name.lower().startswith(("tower", "pytest-"))):
                            budget.omit(relative, "report/cache/work directory excluded")
                        elif depth >= MAX_DEPTH:
                            budget.omit(relative, "maximum traversal depth exceeded")
                        else:
                            stack.append((path, depth + 1))
                    elif stat.S_ISREG(info.st_mode):
                        if (not relative.isprintable() or "\\" in relative or len(relative) > 4096):
                            budget.omit(relative, "unsupported nonportable artifact path")
                        else:
                            yield path, info
                    else:
                        budget.omit(relative, "non-regular artifact refused")
                except OSError as exc:
                    budget.omit(relative, f"artifact stat unavailable: {exc.__class__.__name__}")
        except (OSError, ValueError) as exc:
            budget.omit(_relative(parent, report_dir), f"source directory unavailable: {exc.__class__.__name__}")


def _decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f"nonfinite JSON value {value}")
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    # A valid exponent can overflow a float without invoking parse_constant.
    stack = [value]
    count = 0
    while stack:
        item = stack.pop()
        count += 1
        if count > 100000:
            raise ValueError("JSON value budget exceeded")
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("nonfinite JSON number")
        if isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return value


def _atomic_csv(path, rows, columns, budget):
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, columns, lineterminator="\n")
    writer.writeheader()
    count = 0
    # JSON null and booleans have explicit spellings; an absent key is blank.
    for row in rows:
        converted = {key: (json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                          if not isinstance(value, str) else value) for key, value in row.items() if key in columns}
        previous = out.tell()
        writer.writerow(converted)
        if out.tell() > MAX_TABLE_BYTES // 4:  # conservative UTF-8 byte bound
            out.seek(previous)
            out.truncate()
            budget.omit(path.name, "table byte budget exhausted; remaining rows remain in source artifacts")
            break
        count += 1
    raw = out.getvalue().encode("utf-8")
    if path.exists() and not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("CSV destination must be a regular file")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return count


class _Projection:
    def __init__(self, budget, existing_paths):
        self.budget = budget
        self.existing_paths = existing_paths
        self.tables = {key: [] for key in COLUMNS}
        self.science = []
        self.row_count = 0

    def add(self, table, source, pointer, values):
        if self.row_count >= MAX_ROWS:
            if self.row_count == MAX_ROWS:
                self.budget.omit(source, "canonical row budget exhausted; originals retained")
                self.row_count += 1
            return
        self.tables[table].append({"source_path": source, "source_record": pointer, **values})
        self.row_count += 1

    def frontier(self, source, pointer, row, **context):
        if not isinstance(row, dict):
            return
        values = {**context, **_pick(row, ["case_id", "parent_id", "split", "family", "h", "sequence", "device",
                    "status", "failed", "feasible", "error", "error_upper", "reference_uncertainty"])}
        for old, new in [("method", "family"), ("candidate_h", "h"), ("failure_reason", "reason")]:
            if old in row:
                values[new] = row[old]
        timing = _dict(row.get("timing", row.get("performance")))
        values.update(_pick(timing, ["wall_seconds_median", "cuda_event_seconds_median", "peak_allocated_bytes",
                                  "peak_reserved_bytes", "host_process_peak_rss_bytes"]))
        if "includes" in timing:
            values["timing_scope"] = timing["includes"]
        reference = _dict(row.get("reference"))
        if "uncertainty" in reference:
            values["reference_uncertainty"] = reference["uncertainty"]
        if "reason" in row:
            values["reason"] = row["reason"]
        self.add("frontier", source, pointer, values)

    def training(self, source, pointer, row):
        values = _pick(row, ["family", "status", "selected_step", "selected_parameters_changed"])
        values.update({"record_type": "final"})
        for old, new in [("steps", "step"), ("global_step", "step"), ("best_validation_loss", "validation_loss"),
                         ("last_training_loss", "training_loss"), ("error", "reason")]:
            if old in row:
                values[new] = row[old]
        self.add("training", source, pointer, values)
        for index, item in enumerate(_records(row.get("history"))):
            if isinstance(item, dict):
                self.add("training", source, f"{pointer}/history/{index}", {
                    **_pick(row, ["family", "status"]), "record_type": "history",
                    **_pick(item, ["step", "loss", "training_loss", "validation_loss", "admissible_validation"])})

    def reference(self, source, pointer, row, **context):
        if isinstance(row, dict):
            self.add("references", source, pointer, {**context, **_pick(row, COLUMNS["references"])})

    def gate(self, source, pointer, name, row, **context):
        if isinstance(row, dict):
            self.add("gates", source, pointer, {**context, "gate": name,
                     **_pick(row, ["passed", "decision", "scope", "reason"])})

    def consume(self, path, source, value):
        if not isinstance(value, dict):
            return
        name = path.name
        if name == "stage.json":
            row = _pick(value, COLUMNS["stages"])
            if "error" in value:
                row["reason"] = value["error"]
            self.add("stages", source, "", row)
        elif name == "frontier.json":
            for i, row in enumerate(_records(value.get("rows"))):
                self.frontier(source, f"/rows/{i}", row)
        elif name == "heldout.json":
            for i, row in enumerate(_records(value.get("rows"))):
                if isinstance(row, dict):
                    flat = _pick(row, ["parent_id", "family", "h", "device", "status", "reason"])
                    for part in ("one_step", "two_step"):
                        flat.update({part + "_" + k: v for k, v in _dict(row.get(part)).items()
                                     if k in {"rms", "upper", "reference_uncertainty"}})
                    self.add("heldout", source, f"/rows/{i}", flat)
        elif name == "references.json":
            for i, row in enumerate(_records(value.get("records"))):
                self.reference(source, f"/records/{i}", row)
        elif name == "gates.json":
            for gate, row in value.items():
                self.gate(source, "/" + gate, gate, row)
        elif name == "training_result.json" or path.parent.name == "training":
            if "history" in value or "status" in value:
                self.training(source, "", value)
        elif name == "summary.json" and "comparisons" in value:
            comparisons = [v for v in _records(value.get("comparisons")) if isinstance(v, dict)]
            eligible = [v for v in comparisons if v.get("eligible") is True]
            speeds = [v["speedup"] for v in eligible if _finite(v.get("speedup"))]
            failed = [v for v in _records(value.get("training")) if isinstance(v, dict)
                      and v.get("status") == "NUMERICAL_FAILURE"]
            summary = {"source_path": source, "kind": "architecture_research",
                **_pick(value, ["status", "device", "scope", "selection_scope", "headroom_passed", "trained_family_count",
                               "diagnostic_parent_count", "failed_trajectories", "accuracy_tolerance", "actual_neural_training"]),
                "comparison_count": len(comparisons), "eligible_comparison_count": len(eligible),
                "twenty_percent_faster_count": sum(v.get("twenty_percent_faster") is True for v in eligible),
                "speed_advantage_observed": any(v > 1 for v in speeds) if speeds else None,
                "numerical_failure_count": len(failed),
                "numerical_failure_families": [v.get("family") for v in failed]}
            self.science.append(summary)
            self.gate(source, "/headroom", "numerical_headroom", _dict(value.get("headroom")))
            for i, row in enumerate(_records(value.get("training"))):
                if isinstance(row, dict) and path.parent / "training" / f"{row.get('family')}.json" not in self.existing_paths:
                    self.training(source, f"/training/{i}", row)
        elif name == "summary.json" and value.get("stage") == "light-screen":
            self.science.append({"source_path": source, "kind": "light_screen", **_pick(value, ["status", "device", "scope",
                "declared_cases", "completed_cases", "references_passed_cases", "headroom_passed_cases", "temporal_passed_cases",
                "decision", "pilot_authorized", "confirmatory_authorized"])})
            for i, case in enumerate(_records(value.get("cases"))):
                if not isinstance(case, dict):
                    continue
                pointer = f"/cases/{i}"
                context = _pick(case, ["case_id"])
                for j, row in enumerate(_records(case.get("local_references"))):
                    self.reference(source, f"{pointer}/local_references/{j}", row, **context)
                self.reference(source, pointer + "/global_reference", case.get("global_reference"), role="global", **context)
                for j, row in enumerate(_records(case.get("classical_methods"))):
                    self.frontier(source, f"{pointer}/classical_methods/{j}", row, **context)
                for gate in ("numerical_headroom_screen", "temporal_representation_screen"):
                    self.gate(source, pointer + "/" + gate, gate, case.get(gate), **context)
                models = _dict(_dict(case.get("temporal_oracle")).get("models"))
                for family, row in models.items():
                    if isinstance(row, dict):
                        self.add("heldout", source, pointer + "/temporal_oracle/models/" + family,
                            {**context, "family": family, **_pick(row, ["heldout_absolute_rms", "heldout_relative_rms_with_noise_floor"])})
        elif name in {"evaluation.json", "benchmark.json"}:
            self.science.append({"source_path": source, "kind": "solver_evaluation",
                **_pick(value, ["stage", "status", "device", "scope", "claim", "tolerance", "all_methods_tolerance_feasible"])})
            for family, batches in _dict(value.get("validation_candidates")).items():
                for i, batch in enumerate(_records(batches)):
                    for j, row in enumerate(_records(_dict(batch).get("outcomes"))):
                        self.frontier(source, f"/validation_candidates/{family}/{i}/outcomes/{j}", row, family=family)
            for family, rows in _dict(value.get("diagnostic_outcomes")).items():
                for i, row in enumerate(_records(rows)):
                    self.frontier(source, f"/diagnostic_outcomes/{family}/{i}", row, family=family)

    def junit(self, source, raw):
        # External entities are unsupported by ElementTree; reject declarations
        # as well so internal entity expansion cannot amplify bounded input.
        if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
            raise ValueError("XML entity/DOCTYPE declarations are unsupported")
        root = ET.fromstring(raw)
        if root.tag not in {"testsuites", "testsuite"}:
            return
        for suite_index, suite in enumerate(root.iter("testsuite")):
            for test_index, case in enumerate(suite.findall("testcase")):
                status, reason = "PASSED", None
                for tag, outcome in (("failure", "FAILED"), ("error", "ERROR"), ("skipped", "SKIPPED")):
                    element = case.find(tag)
                    if element is not None:
                        status, reason = outcome, element.get("message", element.text or "")[:1024]
                        break
                values = {"suite": suite.get("name", ""), "test": case.get("name", ""),
                          "classname": case.get("classname", ""), "status": status, "reason": reason}
                if "time" in case.attrib:
                    try:
                        elapsed = float(case.attrib["time"])
                        if math.isfinite(elapsed) and elapsed >= 0:
                            values["elapsed_seconds"] = elapsed
                    except ValueError:
                        pass
                self.add("tests", source, f"testsuite[{suite_index}]/testcase[{test_index}]", values)


def publish_outputs(report_dir: Path, source_dirs: list[Path]) -> dict:
    """Publish bounded projections and return compact, honest scientific results.

    The report must already exist and have a Tower ``logs.json`` index. Missing,
    malformed, oversized and unsupported source artifacts remain explicit in
    the inventory/omissions. Source manifests' hashes are declared evidence,
    never falsely described as verified when a large file was not read.
    """
    report_dir = _safe_directory(report_dir)
    outputs = report_dir / "outputs"
    outputs.mkdir(exist_ok=True)
    _safe_directory(outputs)
    budget = _Budget()
    roots, seen = [], set()
    for raw in source_dirs:
        root = Path(os.path.abspath(raw))
        if root == report_dir or report_dir in root.parents:
            raise ValueError("scientific source cannot be the reporting directory or its child")
        if root in seen:
            continue
        seen.add(root)
        try:
            roots.append(_safe_directory(root))
        except (OSError, ValueError) as exc:
            budget.omit(_relative(root, report_dir), f"source unavailable: {exc}")
    files, inventory, documents, xml_documents, used_paths = [], [], {}, {}, set()
    inventory_bytes = 0
    for root in roots:
        for path, info in _walk(root, report_dir, budget):
            if path in used_paths:
                continue
            used_paths.add(path)
            if len(files) >= MAX_FILES:
                budget.omit(_relative(root, report_dir), "artifact count budget exhausted; remaining files omitted")
                break
            relative = _relative(path, report_dir)
            entry = {"path": relative, "kind": KINDS.get(path.suffix.lower(), "other"), "bytes": info.st_size}
            raw = None
            if info.st_size <= MAX_FILE_BYTES and budget.read_bytes + info.st_size <= MAX_READ_BYTES and not budget.expired():
                try:
                    _safe_directory(path.parent)
                    raw = budget.read(path, info)
                    entry.update(sha256=hashlib.sha256(raw).hexdigest(), hash_status="computed")
                except (OSError, ValueError) as exc:
                    entry["hash_status"] = "unavailable"
                    budget.omit(relative, str(exc))
            else:
                entry["hash_status"] = "not_read_budget"
            if raw is not None and path.suffix.lower() == ".json":
                try:
                    documents[path] = _decode(raw)
                except (ValueError, UnicodeError, RecursionError) as exc:
                    budget.omit(relative, f"JSON not projected: {exc}")
            elif raw is not None and path.suffix.lower() == ".xml":
                xml_documents[path] = raw
            size = len(json.dumps(entry, ensure_ascii=False).encode("utf-8"))
            if inventory_bytes + size > MAX_INVENTORY_BYTES:
                documents.pop(path, None)
                xml_documents.pop(path, None)
                budget.omit(relative, "inventory byte budget exhausted; remaining artifacts omitted")
                break
            inventory_bytes += size
            files.append(path)
            inventory.append(entry)
    # Preserve existing declared hashes without reading or unpickling large files.
    declared = {}
    for path, value in documents.items():
        if path.name == "manifest.json" and isinstance(value, dict):
            for key, digest in _dict(value.get("files")).items():
                if (isinstance(key, str) and not Path(key).is_absolute() and ".." not in Path(key).parts
                        and isinstance(digest, str) and re.fullmatch(r"[0-9a-fA-F]{64}", digest)):
                    declared[path.parent / key] = (digest.lower(), _relative(path, report_dir))
    for path, entry in zip(files, inventory):
        if path in declared:
            digest, manifest = declared[path]
            entry["declared_sha256"] = digest
            entry["hash_manifest"] = manifest
            if "sha256" not in entry:
                entry["sha256"] = digest
                entry["hash_status"] = "declared_unverified"
            elif entry["sha256"] != digest:
                entry["hash_status"] = "computed_manifest_mismatch"
                budget.omit(entry["path"], "computed hash disagrees with source manifest")
    projection = _Projection(budget, set(files))
    for path, value in documents.items():
        if budget.expired():
            budget.omit(_relative(path, report_dir), "projection time budget exhausted")
            break
        projection.consume(path, _relative(path, report_dir), value)
    for path, raw in xml_documents.items():
        try:
            projection.junit(_relative(path, report_dir), raw)
        except (ValueError, ET.ParseError) as exc:
            budget.omit(_relative(path, report_dir), f"JUnit not projected: {exc}")
    table_rows = {name: _atomic_csv(outputs / f"{name}.csv", rows, COLUMNS[name], budget)
                  for name, rows in projection.tables.items()}
    metadata = {"schema": "tdn.tower.analytics-tables/v1", "path_base": "Tower report directory",
        "null_encoding": "literal null = observed source null; empty cell = absent field; booleans = true/false",
        "source_record": "JSON Pointer into source_path; source files are authoritative and are never rewritten",
        "units": {"h": "model physical time", "error": "source-defined state error (research RMS)",
                  "*_rms": "state RMS", "*_seconds*": "seconds", "*_bytes": "bytes",
                  "loss": "source-defined training objective; not comparable across protocols",
                  "refinement_substeps": "JSON array of integer step counts"},
        "interpretation": "No cross-run aggregation, scheduler inference, imputation, or scientific gate relaxation. "
                          "History rows retain null losses and failed/invalid outcomes. Empty tables mean no projected rows.",
        "columns": COLUMNS, "rows": table_rows}
    atomic_json(outputs / "tables.json", metadata)
    # Prefer actual stdout/stderr and readable summaries; all other text artifacts remain inventoried.
    log_index = read_json(report_dir / "logs.json") if (report_dir / "logs.json").exists() else {"logs": []}
    logs = _records(_dict(log_index).get("logs"))
    registered = {os.path.abspath(report_dir / row["path"]) for row in logs
                  if isinstance(row, dict) and isinstance(row.get("path"), str)}
    candidates = sorted(zip(files, inventory), key=lambda pair: (
        0 if pair[0].suffix.lower() in {".out", ".err", ".log", ".stdout", ".stderr"} else
        1 if pair[0].name in {"summary.txt", "stage.json", "summary.json"} else 2, str(pair[0])))
    for path, entry in candidates:
        if entry["kind"] not in TEXT_KINDS or str(path) in registered:
            continue
        if len(registered) >= 256:
            budget.omit(entry["path"], "log index capacity reached; artifact remains inventoried")
            continue
        try:
            register_log(report_dir, "science." + hashlib.sha256(entry["path"].encode()).hexdigest()[:20], entry["path"],
                         label=path.name[:160], group="Scientific outputs", description="Original TDN artifact; read-only location")
            registered.add(str(path))
        except (ValueError, OSError) as exc:
            budget.omit(entry["path"], f"log registration unavailable: {exc}")
    # Keep the final return small enough for Tower summaries and planning cohorts.
    science, science_bytes = [], 0
    for row in projection.science[:32]:
        size = len(json.dumps(row, ensure_ascii=False).encode("utf-8"))
        if science_bytes + size > 64 << 10:
            break
        science.append(row)
        science_bytes += size
    if len(projection.science) > len(science):
        budget.omit("outputs/results.json", "scientific summary limit reached; originals and tables retained")
    failure_rows = [row for row in projection.tables["training"] if row.get("record_type") == "final"
                    and row.get("status") == "NUMERICAL_FAILURE"]
    outcomes = set()
    if failure_rows:
        outcomes.add("NUMERICAL_FAILURE")
    for row in science:
        if row.get("headroom_passed") is False or row.get("headroom_passed_cases") == 0:
            outcomes.add("NO_NUMERICAL_HEADROOM")
        if row.get("speed_advantage_observed") is False:
            outcomes.add("NO_SPEED_ADVANTAGE_OBSERVED")
        if row.get("numerical_failure_count", 0):
            outcomes.add("NUMERICAL_FAILURE")
    results = {"schema": "tdn.tower.scientific-results/v1", "application_completion_is_scientific_success": False,
        "scientific_outcomes": sorted(outcomes), "reports": science,
        "numerical_failure_records": len(failure_rows), "table_rows": table_rows,
        "artifact_count": len(inventory), "reporting_omission_count": budget.omission_count,
        "science_sources_mutated": False}
    if projection.tables["gates"]:
        results["gate_outcomes"] = {
            "passed": sum(row.get("passed") is True for row in projection.tables["gates"]),
            "failed": sum(row.get("passed") is False for row in projection.tables["gates"]),
            "unknown": sum(row.get("passed") is not True and row.get("passed") is not False
                           for row in projection.tables["gates"])}
    if projection.tables["tests"]:
        results["test_outcomes"] = {status: sum(row.get("status") == status for row in projection.tables["tests"])
                                    for status in ("PASSED", "FAILED", "ERROR", "SKIPPED")}
    artifact_report = {"schema": "tdn.tower.artifacts/v1", "path_base": "Tower report directory",
        "source_dirs": [_relative(root, report_dir) for root in roots], "artifacts": inventory,
        "omissions": budget.omissions, "omission_count": budget.omission_count,
        "omissions_list_truncated": budget.omission_count > len(budget.omissions),
        "limits": {"files": MAX_FILES, "directory_entries": MAX_ENTRIES, "depth": MAX_DEPTH,
                   "read_bytes": MAX_READ_BYTES, "per_file_read_bytes": MAX_FILE_BYTES,
                   "source_read_seconds": MAX_SECONDS, "canonical_rows": MAX_ROWS},
        "observed_read_bytes": budget.read_bytes,
        "hash_semantics": "computed = bytes read here; declared_unverified = original manifest assertion only; "
                          "not_read_budget = no hash available within budgets. Source paths may name large binary files; "
                          "these are not copied or deserialized."}
    atomic_json(outputs / "artifacts.json", artifact_report)
    atomic_json(outputs / "results.json", results)
    return results
