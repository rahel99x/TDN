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
# GPU frontier rows retain raw wall/CUDA repetitions as well as their medians.
MAX_FRONTIER_FILE_BYTES = 2 << 20
# Frozen interaction screens retain complete per-method work/error records.
MAX_INTERACTION_FILE_BYTES = 2 << 20
# CPU work-precision candidates retain repeated timing and work records in JSON.
# Only this exact canonical filename receives the larger read allowance.
MAX_WORK_PRECISION_FILE_BYTES = 8 << 20
MAX_WORK_PRECISION_JSON_VALUES = 300000
# Compact spatial scaling keeps independent per-member errors and raw repeats.
# Its dedicated reserves cannot consume the ordinary analytics budgets.
MAX_COMPACT_SPATIAL_FILE_BYTES = 24 << 20
MAX_COMPACT_SPATIAL_JSON_VALUES = 1500000
MAX_COMPACT_SPATIAL_ROWS = 16000
MAX_COMPACT_SPATIAL_TABLE_BYTES = 8 << 20
MAX_INVENTORY_BYTES = 512 << 10
MAX_TABLE_BYTES = 2 << 20
MAX_ROWS = 10000
# Mechanism cases expose many scalar diagnostics. This separate reserve keeps
# their complete projection without consuming the existing science row budget.
MAX_MECHANISM_TABLE_BYTES = 8 << 20
MAX_MECHANISM_ROWS = 12000
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
REPLICATION_CONTEXT = ["replicate_id", "training_seed", "sample_schedule_seed", "diagnostic_block"]
COLUMNS = {
    "frontier": PREFIX + REPLICATION_CONTEXT + ["case_id", "parent_id", "split", "family", "h", "sequence", "device",
        "status", "failed", "feasible", "error", "error_upper", "reference_uncertainty",
        "wall_seconds_median", "cuda_event_seconds_median", "peak_allocated_bytes",
        "peak_reserved_bytes", "host_process_peak_rss_bytes", "timing_scope", "reason"],
    "training": PREFIX + REPLICATION_CONTEXT + ["record_type", "family", "status", "step", "selected_step", "loss",
        "training_loss", "validation_loss", "admissible_validation", "selected_parameters_changed", "reason",
        "parameter_count", "training_seconds", "benchmark_role", "learning_rate", "optimizer",
        "maximum_optimizer_updates", "architecture_track", "architecture_backbone", "failure_context",
        "validation_failures", "failures", "failure_stages"],
    "heldout": PREFIX + REPLICATION_CONTEXT + ["case_id", "parent_id", "family", "h", "device", "status",
        "one_step_rms", "one_step_upper", "one_step_reference_uncertainty", "two_step_rms",
        "two_step_upper", "two_step_reference_uncertainty", "heldout_absolute_rms",
        "heldout_relative_rms_with_noise_floor", "reason"],
    "references": PREFIX + ["case_id", "parent_id", "role", "h", "accepted", "reason", "uncertainty",
        "defect_norm", "substeps", "refinement_substeps", "refinement_differences", "observed_order"],
    "gates": PREFIX + ["case_id", "gate", "passed", "decision", "scope", "reason"],
    "stages": PREFIX + ["stage", "status", "device", "execution_mode", "actually_ran", "elapsed_seconds", "reason"],
    "tests": PREFIX + ["suite", "test", "classname", "status", "elapsed_seconds", "reason"],
    "neural_comparisons": PREFIX + REPLICATION_CONTEXT + ["parent_id", "tdn_family", "baseline_family", "baseline_kind",
        "tdn_training_status", "baseline_training_status", "tdn_training_selection", "baseline_training_selection",
        "tdn_selected_step", "baseline_selected_step", "tdn_optimizer_steps", "baseline_optimizer_steps",
        "tdn_status", "baseline_status", "eligible", "tdn_h", "baseline_h", "tdn_error", "baseline_error",
        "tdn_error_upper", "baseline_error_upper", "tdn_wall_seconds", "baseline_wall_seconds",
        "tdn_device", "baseline_device", "speedup", "speed_outcome", "tdn_feasible_h_count",
        "baseline_feasible_h_count", "tdn_missing_h_count", "baseline_missing_h_count", "tdn_invalid_h_count",
        "baseline_invalid_h_count", "tdn_heldout_missing_h_count", "baseline_heldout_missing_h_count",
        "tdn_heldout_invalid_h_count", "baseline_heldout_invalid_h_count", "ineligible_reasons",
        "tdn_failure_reason", "baseline_failure_reason"],
    "neural_accuracy": PREFIX + REPLICATION_CONTEXT + ["tdn_family", "baseline_family", "baseline_kind", "tdn_training_selection",
        "baseline_training_selection", "metric", "cases", "eligible", "wins", "losses", "ties", "ineligible",
        "failure_reasons"],
    "replication": PREFIX + REPLICATION_CONTEXT + ["family", "status", "selection", "selected_step",
        "optimizer_steps", "parameter_count", "training_seconds", "failure_reason", "parent_count",
        *[prefix + "_" + field for prefix in ("rollout", "heldout_one", "heldout_two")
          for field in ("expected", "completed", "invalid", "missing", "pass", "invalid_reasons")],
        "feasible_parent_count", "robust_joint_parent_pass_count", "trained_checkpoint",
        "robust_joint_trained_parent_pass_count", "worst_rollout_upper_error", "worst_one_step_upper_error",
        "worst_two_step_upper_error", "worst_rollout", "worst_heldout_one", "worst_heldout_two",
        "regime_counts", "state_counts"],
    "replication_comparisons": PREFIX + REPLICATION_CONTEXT + ["tdn_family", "baseline_family",
        "expected_parent_count", "eligible", "wins", "losses", "ties", "ineligible", "failure_reasons",
        "speedup_median", "speedup_min", "speedup_max", "trained_pair", "tdn_training_selection",
        "baseline_training_selection", "block_rows",
        *[prefix + "_" + field for prefix in ("same_h_rollout", "heldout_one", "heldout_two")
          for field in ("expected", "eligible", "wins", "losses", "ties", "ineligible")]],
    "mechanisms": PREFIX + ["case_record", "case_id", "panel", "mechanism", "test", "variant",
        "kind", "outcome", "device", "training_attempted", "metric", "value", "value_type", "inputs", "note"],
    "work_precision": PREFIX + ["candidate_id", "case_id", "case_role", "variant", "nsteps", "h",
        "final_time", "grid", "dtype", "device", "training_attempted", "status", "trajectory_completed",
        "completed_steps", "finite", "in_bounds", "error_rms", "error_max", "error_mean",
        "error_upper_rms", "error_upper_max_estimate", "reference_accepted", "reference_uncertainty_rms",
        "reference_uncertainty_max_estimate", "preparation_seconds", "prepared_median_seconds",
        "prepared_min_seconds", "prepared_max_seconds", "setup_inclusive_median_seconds",
        "timing_repeats_record", "warmup_record", "work_per_rollout_record", "cache_metadata_record",
        "failure_reason"],
    "tolerance_frontiers": PREFIX + ["frontier_id", "case_id", "case_role", "variant", "norm", "tolerance",
        "device", "training_attempted", "status", "selected_candidate_id", "selected_candidate_record",
        "selected_nsteps", "selected_error", "reference_uncertainty", "adjusted_error", "prepared_seconds",
        "setup_inclusive_seconds", "setup_selected_candidate_id", "setup_selected_candidate_record",
        "setup_selected_nsteps", "setup_selected_seconds", "setup_selected_error", "setup_selected_adjusted_error",
        "feasible_candidate_count", "selection_scope"],
    "work_precision_checks": PREFIX + ["record_type", "reference_id", "parity_id", "case_id", "case_role",
        "device", "training_attempted", "reference_accepted", "uncertainty_rms", "uncertainty_max_estimate",
        "reference_n", "reference_2n", "reference_4n", "difference_n_2n", "difference_2n_4n", "observed_order",
        "reference_reason", "attempts", "rhs_evaluations", "reference_seconds", "check", "dtype", "nodes",
        "nsteps", "max_difference", "rms_difference", "allowed_absolute_difference", "passed", "status",
        "failure_reason"],
}
COMPACT_SPATIAL_TABLES = ("compact_spatial", "compact_spatial_frontiers", "compact_spatial_checks")
for _table, _base in zip(COMPACT_SPATIAL_TABLES,
                         ("work_precision", "tolerance_frontiers", "work_precision_checks")):
    COLUMNS[_table] = COLUMNS[_base] + ["panel", "batch_size", "cells"]
COLUMNS["compact_spatial"] += ["error_spatial_rms", "error_spatial_max", "error_mean_abs",
    "error_rms_worst_member", "error_max_worst_member", "throughput_members_per_second", "member_errors_record"]
COLUMNS["compact_spatial_checks"] += ["variant", "reference_defect_rms", "reference_defect_max",
    "relative_rms_when_resolved", "relative_resolution_threshold", "mean_difference_abs",
    "spatial_rms_difference", "member_references_record"]
COLUMNS["compact_spatial_frontiers"] += ["throughput_members_per_second"]
MECHANISM_KINDS = ("correctness", "representation", "scientific", "negative_control")
MECHANISM_OUTCOMES = ("PASS", "FAIL", "OBSERVED", "INCONCLUSIVE", "EXPECTED_LIMITATION")
NEURAL_METRICS = ("matched_tolerance_speed", "same_h_rollout_rms", "same_h_rollout_upper_error",
                  "heldout_one_step_rms", "heldout_two_step_rms", "heldout_one_step_upper_error",
                  "heldout_two_step_upper_error")


def _safe_directory(path):
    path = Path(os.path.abspath(path))
    for component in reversed((path, *path.parents)):
        if not stat.S_ISDIR(component.lstat().st_mode):
            raise ValueError(f"directory required; symlinks refused: {component}")
    return path


def _relative(path, report_dir):
    return Path(os.path.relpath(path, report_dir)).as_posix()


def _finite(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _dict(value):
    return value if isinstance(value, dict) else {}


def _records(value):
    return value if isinstance(value, list) else []


def _pick(value, names):
    return {name: value[name] for name in names if name in value}


def _file_read_limit(path):
    if path.name == "frontier.json":
        return MAX_FRONTIER_FILE_BYTES
    if path.name == "interaction-screen.json":
        return MAX_INTERACTION_FILE_BYTES
    if path.name == "work-precision.json":
        return MAX_WORK_PRECISION_FILE_BYTES
    if path.name == "compact-spatial.json":
        return MAX_COMPACT_SPATIAL_FILE_BYTES
    return MAX_FILE_BYTES


def _work_precision_row_supported(row, columns):
    """Keep null distinct from absent, while refusing structured scalar cells."""
    text = {"candidate_id", "frontier_id", "reference_id", "parity_id", "case_id", "case_role", "variant",
            "dtype", "status", "failure_reason", "selected_candidate_id", "setup_selected_candidate_id",
            "norm", "selection_scope",
            "reference_reason", "check", "device", "record_type", "panel"}
    boolean = {"trajectory_completed", "finite", "in_bounds", "reference_accepted", "passed", "training_attempted"}
    if row.get("device", "cpu") != "cpu" or row.get("training_attempted", False) is not False:
        return False
    for key in columns:
        if key not in row or row[key] is None or key in PREFIX or key.endswith("_record"):
            continue
        item = row[key]
        if key == "grid":
            if not isinstance(item, list) or not item or any(type(n) is not int or n <= 0 for n in item):
                return False
        elif key in text:
            if not isinstance(item, str):
                return False
        elif key in boolean:
            if type(item) is not bool:
                return False
        elif not _finite(item):
            return False
    return True


def _compact_reference_supported(row):
    """Reference acceptance and uncertainty summarize all declared members."""
    batch = row.get("batch_size")
    members = row.get("member_references")
    if (type(batch) is not int or batch <= 0 or not isinstance(members, list)
            or len(members) != batch or any(not isinstance(item, dict) for item in members)
            or [item.get("member_index") for item in members] != list(range(batch))
            or any(type(item.get("reference_accepted")) is not bool for item in members)):
        return False
    if row.get("reference_accepted") is not all(item["reference_accepted"] for item in members):
        return False
    for key in ("uncertainty_rms", "uncertainty_max_estimate"):
        values = [item.get(key) for item in members]
        if any(item is not None and (not _finite(item) or item < 0) for item in values):
            return False
        if row.get(key) != (max(values) if all(item is not None for item in values) else None):
            return False
        if row["reference_accepted"] and any(item is None for item in values):
            return False
    return True


def _compact_candidate_members_supported(row, reference):
    """Avoid a pooled batch error or a mismatched member uncertainty sum."""
    if (not isinstance(reference, dict) or not _compact_reference_supported(reference)
            or row.get("batch_size") != reference.get("batch_size")
            or row.get("reference_accepted") != reference.get("reference_accepted")):
        return False
    members = row.get("member_errors")
    if not isinstance(members, list):
        return False
    for suffix in ("rms", "max_estimate"):
        if row.get("reference_uncertainty_" + suffix) != reference.get("uncertainty_" + suffix):
            return False
    errors = ("error_rms", "error_max", "error_spatial_rms", "error_spatial_max",
              "error_upper_rms", "error_upper_max_estimate")
    if row.get("trajectory_completed") is not True or row.get("finite") is not True:
        return not members and all(row.get(key) is None for key in (*errors, "error_mean", "error_mean_abs"))
    if (len(members) != row["batch_size"] or any(not isinstance(item, dict) for item in members)
            or [item.get("member_index") for item in members] != list(range(row["batch_size"]))):
        return False
    for item, ref in zip(members, reference["member_references"]):
        for key in (*errors, "error_mean"):
            if item.get(key) is not None and (not _finite(item[key]) or (key != "error_mean" and item[key] < 0)):
                return False
        for norm, suffix in (("rms", "rms"), ("max", "max_estimate")):
            error, uncertainty = item.get("error_" + norm), ref.get("uncertainty_" + suffix)
            expected = error + uncertainty if error is not None and uncertainty is not None else None
            if item.get("error_upper_" + suffix) != expected:
                return False
    for key in errors:
        values = [item.get(key) for item in members]
        if row.get(key) != (max(values) if all(item is not None for item in values) else None):
            return False
    means = [item.get("error_mean") for item in members]
    mean = max(means, key=abs) if all(item is not None for item in means) else None
    return (row.get("error_mean") == mean and row.get("error_mean_abs") == (abs(mean) if mean is not None else None)
            and row.get("error_rms_worst_member") == row.get("error_rms")
            and row.get("error_max_worst_member") == row.get("error_max"))


class _Budget:
    def __init__(self):
        self.deadline = time.monotonic() + MAX_SECONDS
        self.read_bytes = 0
        self.compact_read_bytes = 0
        self.entries = 0
        self.omissions = []
        self.omission_count = 0

    def omit(self, path, reason):
        self.omission_count += 1
        if len(self.omissions) < MAX_OMISSIONS:
            self.omissions.append({"path": str(path)[:4096], "reason": str(reason)[:300]})

    def expired(self):
        return time.monotonic() >= self.deadline

    def can_read(self, path, size):
        if size > _file_read_limit(path):
            return False
        if path.name == "compact-spatial.json":
            return self.compact_read_bytes + size <= MAX_COMPACT_SPATIAL_FILE_BYTES
        return self.read_bytes - self.compact_read_bytes + size <= MAX_READ_BYTES

    def read(self, path, expected):
        file_limit = _file_read_limit(path)
        if not self.can_read(path, expected.st_size):
            raise ValueError("per-file or total read budget exceeded")
        if self.expired():
            raise ValueError("analytics time budget exceeded")
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_size != expected.st_size:
                raise ValueError("source is not the inventoried regular file")
            raw = bytearray()
            while len(raw) <= file_limit:
                part = os.read(fd, min(65536, file_limit + 1 - len(raw)))
                if not part:
                    break
                raw.extend(part)
                self.read_bytes += len(part)
                if path.name == "compact-spatial.json":
                    self.compact_read_bytes += len(part)
                if (self.read_bytes - self.compact_read_bytes > MAX_READ_BYTES
                        or self.compact_read_bytes > MAX_COMPACT_SPATIAL_FILE_BYTES or self.expired()):
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


def _decode(raw, *, max_values=100000):
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
        if count > max_values:
            raise ValueError("JSON value budget exceeded")
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("nonfinite JSON number")
        if isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return value


def _atomic_csv(path, rows, columns, budget, *, max_bytes=None):
    max_bytes = MAX_TABLE_BYTES if max_bytes is None else max_bytes
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, columns, lineterminator="\n")
    writer.writeheader()
    encoded_bytes = len(out.getvalue().encode("utf-8"))
    count = 0
    # JSON null and booleans have explicit spellings; an absent key is blank.
    for row in rows:
        converted = {key: (json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                          if not isinstance(value, str) else value) for key, value in row.items() if key in columns}
        previous = out.tell()
        writer.writerow(converted)
        out.seek(previous)
        encoded_row_bytes = len(out.read().encode("utf-8"))
        if encoded_bytes + encoded_row_bytes > max_bytes:
            out.seek(previous)
            out.truncate()
            budget.omit(path.name, "table byte budget exhausted; remaining rows remain in source artifacts")
            break
        encoded_bytes += encoded_row_bytes
        out.seek(0, io.SEEK_END)
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
        self.mechanism_row_count = 0
        self.compact_row_count = 0

    def add(self, table, source, pointer, values):
        if table in COMPACT_SPATIAL_TABLES:
            if self.compact_row_count >= MAX_COMPACT_SPATIAL_ROWS:
                if self.compact_row_count == MAX_COMPACT_SPATIAL_ROWS:
                    self.budget.omit(source, "compact-spatial row budget exhausted; originals retained")
                    self.compact_row_count += 1
                return
            self.compact_row_count += 1
        elif table == "mechanisms":
            if self.mechanism_row_count >= MAX_MECHANISM_ROWS:
                if self.mechanism_row_count == MAX_MECHANISM_ROWS:
                    self.budget.omit(source, "mechanisms row budget exhausted; originals retained")
                    self.mechanism_row_count += 1
                return
            self.mechanism_row_count += 1
        else:
            if self.row_count >= MAX_ROWS:
                if self.row_count == MAX_ROWS:
                    self.budget.omit(source, "canonical row budget exhausted; originals retained")
                    self.row_count += 1
                return
            self.row_count += 1
        self.tables[table].append({"source_path": source, "source_record": pointer, **values})

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

    def training(self, source, pointer, row, record_type="final", **context):
        context = {**context, **_pick(row, REPLICATION_CONTEXT)}
        values = {**context, **_pick(row, ["family", "status", "selected_step", "selected_parameters_changed", "parameter_count",
                            "training_seconds", "benchmark_role", "learning_rate", "optimizer", "maximum_optimizer_updates",
                            "failure_context", "validation_failures", "failure_stages"])}
        architecture = _dict(row.get("architecture"))
        values.update({"architecture_" + name: architecture[name] for name in ("track", "backbone")
                       if name in architecture})
        values.update({"record_type": record_type})
        for old, new in [("steps", "step"), ("global_step", "step"), ("best_validation_loss", "validation_loss"),
                         ("last_training_loss", "training_loss"), ("error", "reason")]:
            if old in row:
                values[new] = row[old]
        self.add("training", source, pointer, values)
        for index, item in enumerate(_records(row.get("history"))):
            if isinstance(item, dict):
                self.add("training", source, f"{pointer}/history/{index}", {
                    **context, **_pick(row, ["family", "status"]),
                    "record_type": "history" if record_type == "final" else "source_history",
                    **_pick(item, ["step", "loss", "training_loss", "validation_loss", "admissible_validation", "failures"])})

    def reference(self, source, pointer, row, **context):
        if isinstance(row, dict):
            self.add("references", source, pointer, {**context, **_pick(row, COLUMNS["references"])})

    def gate(self, source, pointer, name, row, **context):
        if isinstance(row, dict):
            self.add("gates", source, pointer, {**context, "gate": name,
                     **_pick(row, ["passed", "decision", "scope", "reason"])})

    def neural(self, source, value, **replication):
        """Project declared neural comparisons without requiring classical headroom.

        Selection at initialization and failed training remain separate from
        timing eligibility. Counts of trained-baseline wins are therefore
        distinct from counts against all recorded baseline checkpoints.
        """
        if value.get("version") != 1 or not all(isinstance(value.get(key), list)
                for key in ("comparisons", "aggregates", "training", "parents")):
            self.budget.omit(source, "neural comparison version or required record lists unsupported")
            return
        training = {row.get("family"): row for row in value["training"] if isinstance(row, dict)
                    and isinstance(row.get("family"), str)}
        parents = {(parent.get("parent_id"), row.get("family")): row for parent in value["parents"]
                   if isinstance(parent, dict) and isinstance(parent.get("parent_id"), str)
                   for row in _records(parent.get("families")) if isinstance(row, dict)
                   and isinstance(row.get("family"), str)}
        comparison_records = []
        for index, row in enumerate(value["comparisons"]):
            if isinstance(row, dict) and all(isinstance(row.get(key), str)
                    for key in ("parent_id", "tdn_family", "baseline_family")):
                comparison_records.append((index, row))
            else:
                self.budget.omit(source, "neural comparison row has unsupported parent/family labels")
        baseline_kinds = {}
        comparisons = [row for _, row in comparison_records]
        for index, row in comparison_records:
            values = {**replication, **_pick(row, ["parent_id", "tdn_family", "baseline_family", "baseline_kind", "eligible",
                                "speedup", "speed_outcome", "ineligible_reasons"])}
            baseline_kinds[(row.get("tdn_family"), row.get("baseline_family"))] = row.get("baseline_kind")
            for side in ("tdn", "baseline"):
                family = row.get(side + "_family")
                selected = training.get(family, {})
                coverage = parents.get((row.get("parent_id"), family), {})
                best = _dict(row.get(side + "_best"))
                for field in ("h", "error", "error_upper", "wall_seconds", "device"):
                    values[side + "_" + field] = best.get(field)
                for field in ("status", "feasible_h_count", "missing_h_count", "invalid_h_count",
                              "heldout_missing_h_count", "heldout_invalid_h_count"):
                    values[side + "_" + field] = coverage.get(field)
                values[side + "_training_status"] = selected.get("status")
                values[side + "_training_selection"] = selected.get("selection",
                    row.get(side + "_training_selection", coverage.get("training_selection")))
                for field in ("selected_step", "optimizer_steps", "failure_reason"):
                    values[side + "_" + field] = selected.get(field)
            self.add("neural_comparisons", source, f"/comparisons/{index}", values)
        for index, row in enumerate(value["aggregates"]):
            if not isinstance(row, dict) or not all(isinstance(row.get(key), str)
                    for key in ("tdn_family", "baseline_family")):
                self.budget.omit(source, "neural aggregate row has unsupported family labels")
                continue
            context = {**replication, **_pick(row, ["tdn_family", "baseline_family", "baseline_training_selection"])}
            context["baseline_kind"] = baseline_kinds.get((row.get("tdn_family"), row.get("baseline_family")))
            context["tdn_training_selection"] = _dict(training.get(row.get("tdn_family"))).get("selection")
            for metric in NEURAL_METRICS:
                counts = row.get(metric)
                if isinstance(counts, dict):
                    self.add("neural_accuracy", source, f"/aggregates/{index}/{metric}", {**context,
                        "metric": metric, **_pick(counts, ["cases", "eligible", "wins", "losses", "ties",
                                                          "ineligible", "failure_reasons"])})
        eligible = [row for row in comparisons if row.get("eligible") is True]
        trained = [row for row in eligible
                   if _dict(training.get(row.get("baseline_family"))).get("selection") == "TRAINED_CHECKPOINT"]
        wins = sum(row.get("speed_outcome") == "WIN" for row in eligible)
        trained_wins = sum(row.get("speed_outcome") == "WIN" for row in trained)
        self.science.append({"source_path": source, "kind": "neural_solver_comparison",
            **replication,
            **_pick(value, ["diagnostic_parent_count", "scope", "cost_scope", "accuracy_scope", "selection_scope"]),
            "neural_comparison_count": len(comparisons), "neural_eligible_comparison_count": len(eligible),
            "neural_speed_advantage_count": wins,
            "neural_speed_advantage_observed": wins > 0 if eligible else None,
            "neural_trained_baseline_eligible_comparison_count": len(trained),
            "neural_trained_baseline_speed_advantage_count": trained_wins,
            "neural_trained_baseline_speed_advantage_observed": trained_wins > 0 if trained else None,
            "neural_baseline_initialization_comparison_count": sum(
                _dict(training.get(row.get("baseline_family"))).get("selection") == "SELECTED_INITIALIZATION"
                for row in comparisons),
            "neural_baseline_training_failure_comparison_count": sum(
                _dict(training.get(row.get("baseline_family"))).get("selection") == "TRAINING_FAILURE"
                for row in comparisons)})

    def replication(self, source, value):
        """Export the canonical prespecified endpoint document once.

        Diagnostic parents are reused across training seeds. Per-seed records
        stay separate and do not imply three independent diagnostic populations.
        """
        if (value.get("version") != 1 or value.get("protocol_version") != 2
                or value.get("benchmark_suite") != "neural-replication"
                or not all(isinstance(value.get(key), list) for key in
                           ("replicates", "families", "endpoint_rows", "comparison_rows"))
                or len(value["replicates"]) != 3 or len(value["families"]) != 5
                or len(value["endpoint_rows"]) > 15 or len(value["comparison_rows"]) > 12):
            self.budget.omit(source, "replication version or required record lists unsupported")
            return
        plans = {}
        for row in value["replicates"]:
            if (not isinstance(row, dict) or not isinstance(row.get("replicate_id"), str)
                    or row["replicate_id"] in plans
                    or any(type(row.get(key)) is not int for key in ("training_seed", "sample_schedule_seed"))):
                self.budget.omit(source, "replication seed declarations unsupported")
                return
            plans[row["replicate_id"]] = row
        families = value["families"]
        if not all(isinstance(family, str) for family in families) or len(set(families)) != len(families):
            self.budget.omit(source, "replication family declarations unsupported")
            return
        for table, field, identity in (("replication", "endpoint_rows", ("family",)),
                                      ("replication_comparisons", "comparison_rows", ("tdn_family", "baseline_family"))):
            seen = set()
            for index, row in enumerate(value[field]):
                if not isinstance(row, dict):
                    self.budget.omit(source, "replication summary row unsupported")
                    continue
                plan = plans.get(row.get("replicate_id")) if isinstance(row.get("replicate_id"), str) else None
                if (plan is None or any(type(row.get(key)) is not int or row[key] != plan[key]
                                        for key in ("training_seed", "sample_schedule_seed"))
                        or any(row.get(key) not in families for key in identity)):
                    self.budget.omit(source, "replication summary row has undeclared seed/family context")
                    continue
                key = (row["replicate_id"], *(row[name] for name in identity))
                if key in seen:
                    self.budget.omit(source, "duplicate canonical replication summary row")
                    continue
                seen.add(key)
                values = _pick(row, [column for column in COLUMNS[table] if column not in PREFIX])
                if table == "replication_comparisons":
                    for metric in ("same_h_rollout", "heldout_one", "heldout_two"):
                        values.update({metric + "_" + name: count for name, count in
                                       _dict(row.get(metric)).items()
                                       if metric + "_" + name in COLUMNS[table]})
                self.add(table, source, f"/{field}/{index}", values)
        self.science.append({"source_path": source, "kind": "neural_replication",
            **_pick(value, ["status", "device", "smoke", "scope", "diagnostic_parent_count", "diagnostic_blocks",
                "replicates", "families", "expected_block_result_count", "observed_block_result_count",
                "training_record_count", "endpoint_scope", "timing_scope", "pairing_scope", "classical_scope",
                "missing_scope", "selection_scope", "training_attempted", "actual_neural_training"]),
            "endpoint_record_count": len(value["endpoint_rows"]),
            "comparison_record_count": len(value["comparison_rows"]),
            "diagnostic_parent_interpretation": "The same declared diagnostic parents are evaluated for each "
                "training seed; seed-by-parent measurements are not independent diagnostic parents.",
            "scientific_gate_authorized": False})

    def mechanisms(self, source, value, *, suite="mechanism-audit"):
        """Project only the canonical training-free audit, including non-results.

        Each metric points to its scalar in the original JSON; case_record
        preserves the case context. Empty metric dictionaries get one explicit
        row so an inconclusive case cannot disappear from the readable table.
        Scientific observations and expected limitations are not unit-test or
        neural-training results.
        """
        if (value.get("schema") != f"tdn.{suite}/v1"
                or value.get("benchmark_suite") != suite
                or value.get("status") not in ("COMPLETED", "INCOMPLETE", "FAILED")
                or value.get("device") != "cpu" or value.get("training_attempted") is not False
                or not isinstance(value.get("rows"), list)):
            self.budget.omit(source, "mechanism audit schema, CPU/training-free scope or record list unsupported")
            return
        seen, accepted = set(), []
        for index, row in enumerate(value["rows"]):
            if (not isinstance(row, dict)
                    or any(not isinstance(row.get(key), str) or not row[key]
                           for key in ("case_id", "panel", "mechanism", "test", "variant"))
                    or row.get("kind") not in MECHANISM_KINDS
                    or row.get("outcome") not in MECHANISM_OUTCOMES
                    or not isinstance(row.get("metrics"), dict)
                    or not isinstance(row.get("inputs"), dict) or not isinstance(row.get("note"), str)):
                self.budget.omit(source, f"mechanism audit row {index} has unsupported identity or fields")
                continue
            identity = (row["panel"], row["case_id"])
            if identity in seen:
                self.budget.omit(source, f"duplicate canonical mechanism case at row {index}")
                continue
            seen.add(identity)
            if any(not isinstance(key, str) or not key
                   or not (item is None or isinstance(item, (str, bool)) or _finite(item))
                   for key, item in row["metrics"].items()):
                self.budget.omit(source, f"mechanism audit row {index} has non-scalar or nonfinite metrics")
                continue
            accepted.append(row)
            pointer = f"/rows/{index}"
            context = {**_pick(row, ["case_id", "panel", "mechanism", "test", "variant", "kind", "outcome",
                                   "inputs", "note"]), "case_record": pointer,
                       "device": value["device"], "training_attempted": False}
            if not row["metrics"]:
                self.add("mechanisms", source, pointer, {**context, "value_type": "absent"})
            for metric_index, (metric, item) in enumerate(row["metrics"].items()):
                escaped = metric.replace("~", "~0").replace("/", "~1")
                value_type = ("null" if item is None else "boolean" if isinstance(item, bool)
                              else "string" if isinstance(item, str) else "number")
                self.add("mechanisms", source, pointer + "/metrics/" + escaped,
                         {**context, "metric": metric, "value": item, "value_type": value_type,
                          # Keep every scalar and its full inputs, but do not repeat
                          # long interaction notes dozens of times per case. Every
                          # row retains case_record into the authoritative JSON.
                          "note": "" if suite == "interaction-screen" and metric_index else row["note"]})
        by_kind = {kind: {outcome: sum(row["kind"] == kind and row["outcome"] == outcome for row in accepted)
                         for outcome in MECHANISM_OUTCOMES} for kind in MECHANISM_KINDS}
        resources = {key: item for key, item in _dict(value.get("resources")).items()
                     if isinstance(key, str) and (item is None or isinstance(item, (str, bool)) or _finite(item))}
        self.science.append({"source_path": source, "kind": suite.replace("-", "_"),
            "benchmark_suite": suite, "status": value["status"],
            "device": "cpu", "training_attempted": False, "actual_neural_training": False,
            "source_case_count": len(value["rows"]), "valid_case_count": len(accepted),
            "unsupported_case_count": len(value["rows"]) - len(accepted),
            "metric_count": sum(len(row["metrics"]) for row in accepted),
            "no_metric_case_count": sum(not row["metrics"] for row in accepted),
            "outcome_counts": {outcome: sum(row["outcome"] == outcome for row in accepted)
                               for outcome in MECHANISM_OUTCOMES},
            "kind_outcome_counts": by_kind, "resources": resources,
            **_pick(value, [key for key in ("scientific_outcome", "reference_scope", "timing_scope", "work_scope")
                           if isinstance(value.get(key), str)]),
            **_pick(value, [key for key in ("elapsed_seconds", "host_process_peak_rss_bytes", "numerical_budget_seconds")
                           if _finite(value.get(key))]),
            "counts_scope": "Valid source cases, before table row/byte budgets; metric rows are not independent cases.",
            "interpretation": "Training-free correctness checks, representation probes and scientific observations. "
                "Negative controls and expected limitations are explicit. PASS is not trained-model superiority; "
                "INCONCLUSIVE and missing metrics provide no affirmative evidence.",
            "scientific_gate_authorized": False})

    def work_precision(self, source, value, *, suite="work-precision"):
        """Project compact canonical records; raw timings stay in the source.

        A frontier is a reported posthoc choice, never an inferred scientific
        success. Unsupported rows are counted and omitted, and valid failures
        retain their original status, nulls and provenance.
        """
        compact = suite == "compact-spatial"
        kind = "compact_spatial" if compact else "work_precision"
        candidate_table, frontier_table, checks_table = (COMPACT_SPATIAL_TABLES if compact else
            ("work_precision", "tolerance_frontiers", "work_precision_checks"))
        roles = (("reused_review", "new_diagnostic", "scaling_control", "historical_control")
                 if compact else ("fresh", "historical_control"))
        selection_scope = "posthoc_declared_grid_worst_member" if compact else "posthoc_declared_grid"
        fields = ("candidate_rows", "frontier_rows", "reference_rows", "parity_rows")
        summary = {"source_path": source, "kind": kind, "benchmark_suite": suite,
            "status": "UNSUPPORTED", "unsupported_document_count": 0, "unsupported_row_count": 0,
            "scientific_gate_authorized": False, "neural_superiority_claim": False,
            "selection_scope": selection_scope,
            "interpretation": "CPU training-free observations; posthoc declared-grid selection is not a "
                "deployable step controller. Failed or infeasible rows provide no affirmative evidence.",
            "counts_scope": "Supported source rows before shared row and per-table byte budgets; "
                "raw timings and complete records remain authoritative in source JSON."}
        if (not isinstance(value, dict) or value.get("schema") != f"tdn.{suite}/v1"
                or value.get("benchmark_suite") != suite
                or value.get("status") not in ("COMPLETED", "INCOMPLETE", "FAILED")
                or value.get("computational_status", value.get("status")) != value.get("status")
                or value.get("device") != "cpu" or value.get("training_attempted") is not False
                or value.get("training_performed", False) is not False
                or (compact and value.get("scientific_outcome") not in ("OBSERVED_MIXED", "INCONCLUSIVE"))
                or not all(isinstance(value.get(field), list) for field in fields)):
            self.budget.omit(source, f"{suite} schema, terminal status, CPU/training-free scope or lists unsupported")
            summary["unsupported_document_count"] = 1
            self.science.append(summary)
            return
        summary.update(status=value["status"], device="cpu", training_attempted=False,
                       actual_neural_training=False)
        if compact:
            summary["scientific_outcome"] = value["scientific_outcome"]
        references = {row.get("case_id"): row for row in value["reference_rows"]
                      if isinstance(row, dict) and isinstance(row.get("case_id"), str)} if compact else {}
        candidates, supported = {}, {}
        for field, table, id_field, labels, statuses in (
            ("candidate_rows", candidate_table, "candidate_id", ("case_id", "case_role", "variant"),
             ("VALID", "INVALID", "INCOMPLETE", "FAILED")),
            ("frontier_rows", frontier_table, "frontier_id", ("case_id", "case_role", "variant", "norm"),
             ("FEASIBLE", "NO_FEASIBLE_CANDIDATE", "INCONCLUSIVE")),
            ("reference_rows", checks_table, "reference_id", ("case_id", "case_role"), None),
            ("parity_rows", checks_table, "parity_id", ("case_id", "check", "dtype"),
             ("PASS", "FAIL", "UNAVAILABLE", "OBSERVED") if compact else ("PASS", "FAIL", "UNAVAILABLE")),
        ):
            seen_ids, seen_keys, accepted = set(), set(), []
            for index, row in enumerate(value[field]):
                pointer = f"/{field}/{index}"
                if (not isinstance(row, dict)
                        or any(not isinstance(row.get(key), str) or not row[key] for key in (id_field, *labels))
                        or (statuses is not None and row.get("status") not in statuses)
                        or ("case_role" in labels and row["case_role"] not in roles)
                        or not _work_precision_row_supported(row, COLUMNS[table])):
                    self.budget.omit(source, f"{suite} {pointer} has unsupported identity, status or scalar fields")
                    continue
                if field == "candidate_rows":
                    valid = (type(row.get("nsteps")) is int and row["nsteps"] > 0
                        and all(key not in row or isinstance(row[key], kind) for key, kind in
                                (("timing_repeats", list), ("work_per_rollout", dict), ("cache_metadata", dict)))
                        and (row.get("warmup") is None or isinstance(row["warmup"], dict)))
                    identity = (row["case_id"], row["variant"], row.get("nsteps"))
                    if compact:
                        valid = (valid and row.get("panel") in ("accuracy", "scaling")
                            and type(row.get("batch_size")) is int and row["batch_size"] > 0
                            and type(row.get("cells")) is int and row["cells"] > 0
                            and isinstance(row.get("grid"), list) and bool(row["grid"])
                            and row["cells"] == math.prod(row["grid"])
                            and all(row.get(key) is None or (_finite(row[key]) and row[key] >= 0)
                                for key in ("error_spatial_rms", "error_spatial_max", "error_mean_abs",
                                    "error_rms_worst_member", "error_max_worst_member",
                                    "throughput_members_per_second"))
                            and _compact_candidate_members_supported(row, references.get(row["case_id"])))
                elif field == "frontier_rows":
                    valid = (row["norm"] in ("rms", "max") and _finite(row.get("tolerance"))
                        and row["tolerance"] > 0 and row.get("selection_scope") == selection_scope)
                    identity = (row["case_id"], row["variant"], row["norm"], row.get("tolerance"))
                    if row["status"] == "FEASIBLE":
                        selected = candidates.get(row.get("selected_candidate_id"))
                        valid = valid and selected is not None
                        if selected is not None:
                            candidate = selected[1]
                            valid = valid and (candidate["status"] == "VALID"
                                and candidate.get("trajectory_completed") is True
                                and candidate.get("finite") is True and candidate.get("in_bounds") is True
                                and candidate.get("reference_accepted") is True
                                and candidate["case_id"] == row["case_id"]
                                and candidate["case_role"] == row["case_role"]
                                and candidate["variant"] == row["variant"]
                                and candidate["nsteps"] == row.get("selected_nsteps")
                                and all(_finite(row.get(key)) and row[key] >= 0 for key in
                                        ("selected_error", "reference_uncertainty", "adjusted_error",
                                         "prepared_seconds", "setup_inclusive_seconds"))
                                and row["adjusted_error"] <= row["tolerance"]
                                and type(row.get("feasible_candidate_count")) is int
                                and row["feasible_candidate_count"] > 0)
                            if compact:
                                valid = valid and (row.get("selected_error") == candidate.get("error_" + row["norm"])
                                    and row.get("adjusted_error") == candidate.get(
                                        "error_upper_rms" if row["norm"] == "rms" else "error_upper_max_estimate"))
                    else:
                        valid = valid and row.get("selected_candidate_id") is None
                    if row.get("setup_selected_candidate_id") is not None:
                        setup_selected = candidates.get(row["setup_selected_candidate_id"])
                        valid = valid and row["status"] == "FEASIBLE" and setup_selected is not None
                        if setup_selected is not None:
                            candidate = setup_selected[1]
                            valid = valid and (candidate["status"] == "VALID"
                                and candidate.get("trajectory_completed") is True
                                and candidate.get("finite") is True and candidate.get("in_bounds") is True
                                and candidate.get("reference_accepted") is True
                                and candidate["case_id"] == row["case_id"]
                                and candidate["case_role"] == row["case_role"]
                                and candidate["variant"] == row["variant"]
                                and candidate["nsteps"] == row.get("setup_selected_nsteps")
                                and all(_finite(row.get(key)) and row[key] >= 0 for key in
                                        ("setup_selected_seconds", "setup_selected_error",
                                         "setup_selected_adjusted_error"))
                                and row["setup_selected_adjusted_error"] <= row["tolerance"])
                            if compact:
                                valid = valid and (row.get("setup_selected_error") == candidate.get("error_" + row["norm"])
                                    and row.get("setup_selected_adjusted_error") == candidate.get(
                                        "error_upper_rms" if row["norm"] == "rms" else "error_upper_max_estimate"))
                elif field == "reference_rows":
                    valid = type(row.get("reference_accepted")) is bool
                    if compact:
                        valid = valid and _compact_reference_supported(row)
                    identity = (row["case_id"],)
                else:
                    valid = (type(row.get("nodes")) is int and row["nodes"] > 0
                        and type(row.get("nsteps")) is int and row["nsteps"] > 0
                        and row.get("passed") is {"PASS": True, "FAIL": False, "UNAVAILABLE": None,
                                                "OBSERVED": None}[row["status"]])
                    identity = (row["case_id"], row["check"], row["dtype"], row.get("nodes"), row.get("nsteps"))
                    if compact:
                        identity += (row.get("variant"),)
                        if row["status"] == "OBSERVED":
                            valid = (valid and row["check"] == "approximation"
                                     and isinstance(row.get("variant"), str) and bool(row["variant"]))
                        elif row["check"] == "approximation":
                            valid = valid and row["status"] == "UNAVAILABLE"
                if not valid:
                    self.budget.omit(source, f"{suite} {pointer} has inconsistent candidate, selection or check fields")
                    continue
                if row[id_field] in seen_ids or identity in seen_keys:
                    self.budget.omit(source, f"duplicate canonical {suite} identity at {pointer}")
                    continue
                seen_ids.add(row[id_field])
                seen_keys.add(identity)
                accepted.append(row)
                values = _pick(row, [key for key in COLUMNS[table] if key not in PREFIX
                                    and not key.endswith("_record")])
                values.update(device="cpu", training_attempted=False)
                if field == "candidate_rows":
                    candidates[row[id_field]] = (pointer, row)
                    for key in ("timing_repeats", "warmup", "work_per_rollout", "cache_metadata",
                                *(("member_errors",) if compact else ())):
                        if key in row:
                            values[key + "_record"] = pointer + "/" + key
                elif field == "frontier_rows" and row.get("selected_candidate_id") in candidates:
                    values["selected_candidate_record"] = candidates[row["selected_candidate_id"]][0]
                    if row.get("setup_selected_candidate_id") in candidates:
                        values["setup_selected_candidate_record"] = candidates[row["setup_selected_candidate_id"]][0]
                elif field in ("reference_rows", "parity_rows"):
                    values["record_type"] = "reference" if field == "reference_rows" else "parity"
                    if compact and field == "reference_rows":
                        values["member_references_record"] = pointer + "/member_references"
                self.add(table, source, pointer, values)
            supported[field] = accepted
            name = field.removesuffix("_rows")
            summary[f"source_{name}_count"] = len(value[field])
            summary[f"supported_{name}_count"] = len(accepted)
            summary[f"unsupported_{name}_count"] = len(value[field]) - len(accepted)
            summary["unsupported_row_count"] += len(value[field]) - len(accepted)
            if statuses is not None:
                summary[f"{name}_status_counts"] = {status: sum(row["status"] == status for row in accepted)
                                                    for status in statuses}
        summary["rejected_reference_count"] = sum(row["reference_accepted"] is False
                                                   for row in supported["reference_rows"])
        self.science.append(summary)

    def consume(self, path, source, value):
        if path.name == "compact-spatial.json":
            self.work_precision(source, value, suite="compact-spatial")
            return
        if path.name == "work-precision.json":
            self.work_precision(source, value)
            return
        if not isinstance(value, dict):
            return
        name = path.name
        replication = _pick(_dict(value.get("replication")), REPLICATION_CONTEXT)
        if name == "mechanism-audit.json":
            self.mechanisms(source, value)
        elif name == "interaction-screen.json":
            self.mechanisms(source, value, suite="interaction-screen")
        elif name == "replication.json":
            self.replication(source, value)
        elif name == "neural-comparisons.json":
            self.neural(source, value, **replication)
        elif name == "stage.json":
            row = _pick(value, COLUMNS["stages"])
            if "error" in value:
                row["reason"] = value["error"]
            self.add("stages", source, "", row)
        elif name == "frontier.json":
            for i, row in enumerate(_records(value.get("rows"))):
                self.frontier(source, f"/rows/{i}", row, **replication)
        elif name == "heldout.json":
            for i, row in enumerate(_records(value.get("rows"))):
                if isinstance(row, dict):
                    flat = {**replication, **_pick(row, ["parent_id", "family", "h", "device", "status", "reason"])}
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
                self.training(source, "", value, **replication)
        elif name == "summary.json" and value.get("benchmark_suite") == "neural-replication":
            # Endpoint/comparison records are exported only from replication.json.
            # Frozen GPU runs describe source checkpoints and do not retrain them.
            field = "source_training" if isinstance(value.get("source_training"), list) else "training"
            for index, row in enumerate(_records(value.get(field))):
                if not isinstance(row, dict):
                    continue
                identity, family = row.get("replicate_id"), row.get("family")
                if not isinstance(identity, str) or not isinstance(family, str):
                    self.budget.omit(source, "replication training summary has unsupported seed/family context")
                    continue
                original = path.parent / "replicates" / identity / "training" / f"{family}.json"
                if original not in self.existing_paths:
                    self.training(source, f"/{field}/{index}", row,
                                  record_type="source_checkpoint" if field == "source_training" else "final")
        elif name == "summary.json" and "comparisons" in value:
            comparisons = [v for v in _records(value.get("comparisons")) if isinstance(v, dict)]
            eligible = [v for v in comparisons if v.get("eligible") is True]
            speeds = [v["speedup"] for v in eligible if _finite(v.get("speedup"))]
            failed = [v for v in _records(value.get("training")) if isinstance(v, dict)
                      and v.get("status") == "NUMERICAL_FAILURE"]
            summary = {"source_path": source, "kind": "architecture_research",
                **replication,
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
    # Exact premix schemas have a dedicated bounded parser. Preserve its
    # verified provenance without reading those files through ordinary limits
    # or interpreting the same training observations a second time.
    from tdn.premix_reporting import publish_premix_outputs
    premix_sources = {}
    premix = publish_premix_outputs(report_dir, roots, delegated_sources=premix_sources)
    from tdn.consistency_reporting import publish_consistency_outputs
    consistency_sources = {}
    consistency = publish_consistency_outputs(report_dir, roots, delegated_sources=consistency_sources)
    # Legacy isolated workflow fixtures omit modules from later programs. Load
    # the new publisher only for explicitly present canonical agenda row files.
    agenda_sources = {}
    agenda = None
    agenda_stages = ("structure", "prepare", "controls", "optimize", "compression", "kernel", "confirm", "policy")
    if any((root / "rows.json").exists() or (root / "rows.json").is_symlink() or any(
            (root / stage / "rows.json").exists() or (root / stage / "rows.json").is_symlink()
            for stage in agenda_stages) for root in roots):
        from tdn.agenda_reporting import publish_agenda_outputs
        agenda = publish_agenda_outputs(report_dir, roots, delegated_sources=agenda_sources)
    delegated_sources = {**premix_sources, **consistency_sources, **agenda_sources}
    # The premix parser has its own explicit time budget.
    budget.deadline = time.monotonic() + MAX_SECONDS
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
            if path in delegated_sources:
                delegated = delegated_sources[path]
                suite = "agenda" if path in agenda_sources else "consistency" if path in consistency_sources else "premix"
                signature = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
                if signature == delegated["signature"]:
                    entry.update(sha256=delegated["sha256"], hash_status="computed_" + suite + "_projection",
                                 projection=f"outputs/{suite}-tables.json")
                else:
                    entry["hash_status"] = "unavailable"
                    budget.omit(relative, "source changed after " + suite + " projection")
            elif budget.can_read(path, info.st_size) and not budget.expired():
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
                    values_limit = {"work-precision.json": MAX_WORK_PRECISION_JSON_VALUES,
                                    "compact-spatial.json": MAX_COMPACT_SPATIAL_JSON_VALUES}.get(path.name, 100000)
                    documents[path] = _decode(raw, max_values=values_limit)
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
                if (isinstance(digest, dict) and set(digest) == {"sha256", "bytes"}
                        and type(digest["bytes"]) is int and digest["bytes"] >= 0):
                    digest = digest["sha256"]
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
    for path in files:
        if path.name in ("work-precision.json", "compact-spatial.json") and path not in documents:
            # An unreadable/oversized canonical artifact must not resemble an
            # empty successful experiment in the scientific summary.
            projection.work_precision(_relative(path, report_dir), None, suite=path.stem)
    for path, value in documents.items():
        if budget.expired():
            budget.omit(_relative(path, report_dir), "projection time budget exhausted")
            break
        projection.consume(path, _relative(path, report_dir), value)
    for path in files:
        if (path.name == "compact-spatial.json" and not any(
                row["kind"] == "compact_spatial" and row["source_path"] == _relative(path, report_dir)
                for row in projection.science)):
            projection.work_precision(_relative(path, report_dir), None, suite="compact-spatial")
    for path, raw in xml_documents.items():
        try:
            projection.junit(_relative(path, report_dir), raw)
        except (ValueError, ET.ParseError) as exc:
            budget.omit(_relative(path, report_dir), f"JUnit not projected: {exc}")
    table_rows = {name: _atomic_csv(outputs / f"{name}.csv", rows, COLUMNS[name], budget,
                                  max_bytes=MAX_COMPACT_SPATIAL_TABLE_BYTES if name in COMPACT_SPATIAL_TABLES
                                  else MAX_MECHANISM_TABLE_BYTES if name == "mechanisms" else MAX_TABLE_BYTES)
                  for name, rows in projection.tables.items()}
    # Report source support separately from actual emitted rows: truncation must
    # never resemble a complete compact-spatial experiment to a consumer.
    for summary in projection.science:
        if summary["kind"] != "compact_spatial":
            continue
        projected = {name: sum(row["source_path"] == summary["source_path"] for row in
                              projection.tables[table][:table_rows[table]]
                              if row.get("record_type", name) == name)
                     for name, table in (("candidate", "compact_spatial"),
                                         ("frontier", "compact_spatial_frontiers"),
                                         ("reference", "compact_spatial_checks"),
                                         ("parity", "compact_spatial_checks"))}
        summary["projected_counts"] = projected
        summary["reporting_complete"] = (not summary["unsupported_document_count"] and
            all(projected[name] == summary.get(f"source_{name}_count") for name in projected))
    metadata = {"schema": "tdn.tower.analytics-tables/v1", "path_base": "Tower report directory",
        "null_encoding": "literal null = observed source null; empty cell = absent field; booleans = true/false",
        "source_record": "JSON Pointer into source_path; source files are authoritative and are never rewritten",
        "units": {"h": "model physical time", "error": "source-defined state error (research RMS)",
                  "*_rms": "state RMS", "*_seconds*": "seconds", "*_bytes": "bytes",
                  "loss": "source-defined training objective; not comparable across protocols",
                  "refinement_substeps": "JSON array of integer step counts",
                  "tdn_h": "model physical time", "baseline_h": "model physical time",
                  "tdn_error": "state RMS", "baseline_error": "state RMS",
                  "tdn_error_upper": "state RMS plus accepted-reference uncertainty",
                  "baseline_error_upper": "state RMS plus accepted-reference uncertainty",
                  "speedup": "baseline complete-rollout wall seconds / TDN complete-rollout wall seconds",
                  "neural_accuracy.cases": "parent pairs for speed; parent x horizon pairs for error",
                  "neural_accuracy.eligible": "eligible pairs for the specified metric",
                  "neural_accuracy.wins": "pairs won by TDN; no statistical significance claim",
                  "*_selected_step": "validation-selected optimizer step; 0 means initialization",
                  "*_optimizer_steps": "completed optimizer updates", "*_h_count": "declared horizon counts",
                  "parameter_count": "trainable and frozen model parameter elements",
                  "training_seconds": "actual training wall seconds including validation/checkpoint overhead",
                  "maximum_optimizer_updates": "declared maximum optimizer updates per family",
                  "learning_rate": "declared optimizer learning rate",
                  "replicate_id": "paired training-seed identifier; not an independent diagnostic population",
                  "training_seed": "initialization random seed paired across families",
                  "sample_schedule_seed": "training-parent/horizon sampling random seed paired across families",
                  "diagnostic_block": "prespecified fresh diagnostic-parent block index",
                  "replication.*_expected": "prespecified parent x horizon endpoint count for this training seed",
                  "replication.*_pass": "finite completed upper error <= fixed tolerance; missing/invalid never passes",
                  "replication.parent_count": "distinct diagnostic parents reused across training seeds",
                  "replication.robust_joint_parent_pass_count": "parents passing both heldout horizons at both one/two steps",
                  "replication_comparisons.speedup_*": "baseline wall seconds / reaction-clock wall seconds within one seed",
                  "replication_comparisons.trained_pair": "both same-seed models selected a positive step with changed parameters",
                  "mechanisms.value": "source-defined scalar metric; metric name and inputs define its meaning",
                  "mechanisms.source_record": "JSON Pointer to the metric scalar, or case object when metrics are empty",
                  "mechanisms.case_record": "JSON Pointer to the original case object",
                  "mechanisms.value_type": "number, boolean, string, null, or absent (empty metric dictionary)"},
        "interpretation": "No cross-run aggregation, scheduler inference, imputation, or scientific gate relaxation. "
                          "History rows retain null losses and failed/invalid outcomes. Empty tables mean no projected rows.",
        "neural_comparison_interpretation": "Speed eligibility uses each family's fastest feasible declared-grid "
            "complete rollout on the same device, independently of classical headroom. WIN/LOSS/TIE are from TDN's "
            "perspective; speedup > 1 means TDN is faster. Error counts use finite completed same-h or held-out "
            "trajectories regardless of tolerance. Checkpoint selection and failures are explicit; comparisons against "
            "initialization are not evidence of superiority over a trained baseline. No overall neural solver ranking.",
        "replication_interpretation": "replication.csv and replication_comparisons.csv project only canonical "
            "replication.json. Full-rollout and one/two-step transient endpoints remain separate. The same diagnostic "
            "parents are reused across three training seeds; repeated classical timings and seed-by-parent rows "
            "are not additional independent parents. Missing/invalid results and initialization remain explicit. "
            "Per-seed speed ratios use each family's fastest feasible declared grid entry, a post-hoc diagnostic "
            "rather than a deployable step selector. No significance, SOTA, ranking or scientific gate authorization.",
        "mechanism_interpretation": "mechanisms.csv projects only canonical mechanism-audit.json or "
            "interaction-screen.json, never their "
            "panel files or summary duplicates. One row represents one scalar metric, not one independent case. "
            "Empty metrics remain explicit. Correctness, representation, scientific observations and negative "
            "controls retain separate kinds and outcomes. A completed training-free audit does not demonstrate "
            "trained-model accuracy or efficiency; expected limitations do not imply failed unit tests. "
            "Interaction-screen case notes appear on the first metric row only; case_record on every row "
            "resolves the complete original note and inputs without duplicating long prose.",
        "work_precision_interpretation": "work_precision.csv has one row per candidate and "
            "tolerance_frontiers.csv one row per case/method/norm/tolerance, projected only from canonical "
            "work-precision.json. work_precision_checks.csv retains reference and parity checks. "
            "Source JSON remains authoritative for raw repeated timings, warmups, work counters and cache metadata; "
            "*_record fields are JSON Pointers within source_path. Selection is posthoc over the declared step "
            "grid, not a deployable step controller. selected_* identifies the fastest prepared candidate; "
            "setup_selected_* independently identifies the fastest setup-inclusive candidate. "
            "Failed, infeasible and incomplete outcomes provide no "
            "affirmative evidence. Training-free CPU measurements make no neural superiority, GPU, significance "
            "or scientific gate claim. Max-error uncertainty is an estimate; norms remain separate.",
        "compact_spatial_interpretation": "compact_spatial.csv, compact_spatial_frontiers.csv and "
            "compact_spatial_checks.csv project only canonical compact-spatial.json, retaining complete "
            "raw repetitions and member records through source JSON pointers. Batches reuse a coefficient "
            "case; their members are not independent cases. RMS and spatial RMS use the worst member, "
            "not a pooled batch RMS; error_mean_abs is the worst absolute member mean error. Adjusted "
            "errors maximize each member's error plus that same member's uncertainty, not the sum of "
            "separately maximized errors and uncertainties. A frontier "
            "is posthoc over the declared grid, not a deployable step controller. Accuracy and scaling "
            "panels and reused/new diagnostic roles remain distinct. Computational completion and source "
            "scientific_outcome are separate; reporting_complete is false if any projected row is omitted. "
            "No neural/FNO, GPU, continuum convergence or independent-replication advantage is established.",
        "limits": {"ordinary_tables_shared_rows": MAX_ROWS, "ordinary_table_bytes": MAX_TABLE_BYTES,
                   "table_overrides": {"mechanisms": {"rows": MAX_MECHANISM_ROWS,
                                                       "bytes": MAX_MECHANISM_TABLE_BYTES}},
                   "combined_rows": MAX_ROWS + MAX_MECHANISM_ROWS},
        "columns": COLUMNS, "rows": table_rows}
    if any(row["kind"] == "compact_spatial" for row in projection.science):
        metadata["limits"].update(compact_spatial_shared_rows=MAX_COMPACT_SPATIAL_ROWS,
                                 combined_rows=MAX_ROWS + MAX_MECHANISM_ROWS + MAX_COMPACT_SPATIAL_ROWS)
        metadata["limits"]["table_overrides"].update({name: {
            "rows": MAX_COMPACT_SPATIAL_ROWS, "bytes": MAX_COMPACT_SPATIAL_TABLE_BYTES}
            for name in COMPACT_SPATIAL_TABLES})
    atomic_json(outputs / "tables.json", metadata)
    # Prefer actual stdout/stderr and readable summaries; all other text artifacts remain inventoried.
    log_index = read_json(report_dir / "logs.json") if (report_dir / "logs.json").exists() else {"logs": []}
    logs = _records(_dict(log_index).get("logs"))
    registered = {os.path.abspath(report_dir / row["path"]) for row in logs
                  if isinstance(row, dict) and isinstance(row.get("path"), str)}
    if any(row.get("kind") in ("mechanism_audit", "interaction_screen") for row in projection.science):
        target = outputs / "mechanisms.csv"
        if str(target) not in registered:
            if len(registered) < 256:
                try:
                    register_log(report_dir, "analytics.mechanisms", "outputs/mechanisms.csv",
                                 label="Mechanism and interaction metrics", group="Scientific outputs",
                                 description="Bounded scalar projection; source pointers retain original case evidence")
                    registered.add(str(target))
                except (ValueError, OSError) as exc:
                    budget.omit("outputs/mechanisms.csv", f"log registration unavailable: {exc}")
            else:
                budget.omit("outputs/mechanisms.csv", "log index capacity reached")
    for kind, tables in (("work_precision", (("work_precision", "Work-precision candidates"),
                            ("tolerance_frontiers", "Posthoc tolerance frontiers"),
                            ("work_precision_checks", "Work-precision reference and parity checks"))),
                         ("compact_spatial", (("compact_spatial", "Compact spatial candidates"),
                            ("compact_spatial_frontiers", "Compact spatial posthoc tolerance frontiers"),
                            ("compact_spatial_checks", "Compact spatial reference and parity checks")))):
        if not any(row.get("kind") == kind for row in projection.science):
            continue
        for name, label in tables:
            target = outputs / f"{name}.csv"
            if str(target) in registered:
                continue
            if len(registered) >= 256:
                budget.omit(f"outputs/{name}.csv", "log index capacity reached")
                continue
            try:
                register_log(report_dir, f"analytics.{name}", f"outputs/{name}.csv", label=label,
                             group="Scientific outputs", description="Compact CPU training-free projection; "
                             "JSON pointers retain raw timing and scientific evidence")
                registered.add(str(target))
            except (ValueError, OSError) as exc:
                budget.omit(f"outputs/{name}.csv", f"log registration unavailable: {exc}")
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
        if row.get("kind") == "work_precision":
            if (row["status"] != "COMPLETED" or row.get("unsupported_row_count", 0)
                    or row.get("unsupported_document_count", 0)):
                outcomes.add("WORK_PRECISION_INCOMPLETE")
        if row.get("kind") == "compact_spatial":
            if row["status"] != "COMPLETED" or not row.get("reporting_complete"):
                outcomes.add("COMPACT_SPATIAL_INCOMPLETE")
            if row.get("scientific_outcome") == "INCONCLUSIVE":
                outcomes.add("COMPACT_SPATIAL_INCONCLUSIVE")
        if row.get("kind") in ("mechanism_audit", "interaction_screen"):
            prefix = "INTERACTION" if row["kind"] == "interaction_screen" else "MECHANISM"
            if row["kind_outcome_counts"]["correctness"]["FAIL"]:
                outcomes.add(f"{prefix}_CORRECTNESS_FAILURE")
            if row["kind_outcome_counts"]["negative_control"]["FAIL"]:
                outcomes.add(f"{prefix}_NEGATIVE_CONTROL_FAILURE")
            if row["status"] != "COMPLETED" or row["unsupported_case_count"]:
                outcomes.add("INTERACTION_SCREEN_INCOMPLETE" if prefix == "INTERACTION" else "MECHANISM_AUDIT_INCOMPLETE")
    results = {"schema": "tdn.tower.scientific-results/v1", "application_completion_is_scientific_success": False,
        "scientific_outcomes": sorted(outcomes), "reports": science,
        "numerical_failure_records": len(failure_rows), "table_rows": table_rows,
        "artifact_count": len(inventory), "reporting_omission_count": budget.omission_count,
        "science_sources_mutated": False}
    neural = [row for row in science if row.get("kind") == "neural_solver_comparison"]
    if neural:
        if any(row.get("replicate_id") for row in neural):
            results["neural_observation_scope"] = (
                "Counts include separate seed-by-parent observations; shared diagnostic parents are not independent "
                "across training seeds. Canonical replication tables retain per-seed outcomes.")
        for field in ("neural_comparison_count", "neural_eligible_comparison_count", "neural_speed_advantage_count",
                      "neural_trained_baseline_eligible_comparison_count", "neural_trained_baseline_speed_advantage_count",
                      "neural_baseline_initialization_comparison_count", "neural_baseline_training_failure_comparison_count"):
            results[field] = sum(row[field] for row in neural)
        for field, eligibility in (("neural_speed_advantage_observed", "neural_eligible_comparison_count"),
                ("neural_trained_baseline_speed_advantage_observed", "neural_trained_baseline_eligible_comparison_count")):
            results[field] = any(row[field] is True for row in neural) if results[eligibility] else None
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
                   "per_file_read_bytes_overrides": {"frontier.json": MAX_FRONTIER_FILE_BYTES,
                                                      "interaction-screen.json": MAX_INTERACTION_FILE_BYTES,
                                                      "work-precision.json": MAX_WORK_PRECISION_FILE_BYTES},
                   "json_values": 100000,
                   "json_values_overrides": {"work-precision.json": MAX_WORK_PRECISION_JSON_VALUES},
                   "source_read_seconds": MAX_SECONDS, "canonical_rows": MAX_ROWS,
                   "canonical_rows_scope": "Shared ordinary science tables; mechanisms has a separate bounded reserve",
                   "mechanisms_rows": MAX_MECHANISM_ROWS,
                   "combined_canonical_rows": MAX_ROWS + MAX_MECHANISM_ROWS,
                   "table_bytes": MAX_TABLE_BYTES,
                   "table_bytes_overrides": {"mechanisms.csv": MAX_MECHANISM_TABLE_BYTES}},
        "observed_read_bytes": budget.read_bytes,
        "hash_semantics": "computed = bytes read here; computed_premix_projection = bytes read by the dedicated "
                          "premix parser with unchanged file identity; computed_consistency_projection = bytes read by the dedicated consistency parser with unchanged file identity; computed_agenda_projection = bytes read by the dedicated agenda parser with unchanged file identity; declared_unverified = original manifest assertion only; "
                          "not_read_budget = no hash available within budgets. Source paths may name large binary files; "
                          "these are not copied or deserialized."}
    if any(path.name == "compact-spatial.json" for path in files):
        limits = artifact_report["limits"]
        limits.update(compact_spatial_read_bytes=MAX_COMPACT_SPATIAL_FILE_BYTES,
                      combined_read_bytes=MAX_READ_BYTES + MAX_COMPACT_SPATIAL_FILE_BYTES,
                      compact_spatial_rows=MAX_COMPACT_SPATIAL_ROWS,
                      combined_canonical_rows=MAX_ROWS + MAX_MECHANISM_ROWS + MAX_COMPACT_SPATIAL_ROWS)
        limits["per_file_read_bytes_overrides"]["compact-spatial.json"] = MAX_COMPACT_SPATIAL_FILE_BYTES
        limits["json_values_overrides"]["compact-spatial.json"] = MAX_COMPACT_SPATIAL_JSON_VALUES
        limits["table_bytes_overrides"].update({name + ".csv": MAX_COMPACT_SPATIAL_TABLE_BYTES
                                               for name in COMPACT_SPATIAL_TABLES})
        artifact_report["observed_compact_spatial_read_bytes"] = budget.compact_read_bytes
    atomic_json(outputs / "artifacts.json", artifact_report)
    if premix is not None:
        results["premix"] = premix
        results["reporting_omission_count"] += premix["reporting_omission_count"]
        if not premix["reporting_complete"]:
            results["scientific_outcomes"] = sorted(set(results["scientific_outcomes"]) |
                                                    {"PREMIX_REPORTING_INCOMPLETE"})
    if consistency is not None:
        results["consistency"] = consistency
        results["reporting_omission_count"] += consistency["reporting_omission_count"]
        if not consistency["reporting_complete"]:
            results["scientific_outcomes"] = sorted(set(results["scientific_outcomes"]) |
                                                    {"CONSISTENCY_REPORTING_INCOMPLETE"})
    if agenda is not None:
        results["agenda"] = agenda
        results["reporting_omission_count"] += agenda["reporting_omission_count"]
        results["scientific_outcomes"] = sorted(set(results["scientific_outcomes"]) | {
            source["scientific_outcome"] for source in agenda["sources"]
            if isinstance(source.get("scientific_outcome"), str)})
        if not agenda["reporting_complete"]:
            results["scientific_outcomes"] = sorted(set(results["scientific_outcomes"]) |
                                                    {"AGENDA_REPORTING_INCOMPLETE"})
    atomic_json(outputs / "results.json", results)
    return results
