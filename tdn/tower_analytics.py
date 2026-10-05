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
}
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
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _dict(value):
    return value if isinstance(value, dict) else {}


def _records(value):
    return value if isinstance(value, list) else []


def _pick(value, names):
    return {name: value[name] for name in names if name in value}


def _file_read_limit(path):
    return MAX_FRONTIER_FILE_BYTES if path.name == "frontier.json" else MAX_FILE_BYTES


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
        file_limit = _file_read_limit(path)
        if expected.st_size > file_limit or self.read_bytes + expected.st_size > MAX_READ_BYTES:
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
        if encoded_bytes + encoded_row_bytes > MAX_TABLE_BYTES:
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

    def consume(self, path, source, value):
        if not isinstance(value, dict):
            return
        name = path.name
        replication = _pick(_dict(value.get("replication")), REPLICATION_CONTEXT)
        if name == "replication.json":
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
            if info.st_size <= _file_read_limit(path) and budget.read_bytes + info.st_size <= MAX_READ_BYTES and not budget.expired():
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
                  "replication_comparisons.trained_pair": "both same-seed models selected a positive step with changed parameters"},
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
                   "per_file_read_bytes_overrides": {"frontier.json": MAX_FRONTIER_FILE_BYTES},
                   "source_read_seconds": MAX_SECONDS, "canonical_rows": MAX_ROWS},
        "observed_read_bytes": budget.read_bytes,
        "hash_semantics": "computed = bytes read here; declared_unverified = original manifest assertion only; "
                          "not_read_budget = no hash available within budgets. Source paths may name large binary files; "
                          "these are not copied or deserialized."}
    atomic_json(outputs / "artifacts.json", artifact_report)
    atomic_json(outputs / "results.json", results)
    return results
