"""Bounded, source-linked Tower views of the premix experiment.

Only the project's outputs are adapted.  The Tower application is untouched.
Canonical JSON stays authoritative: these tables never train a model, select a
checkpoint, infer scientific success from completion, or count repeated seeds
and step sizes as independent parents.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import time
import uuid

from tdn.reporting import atomic_json, read_json, register_log

MAX_FILE_BYTES = 48 << 20
MAX_READ_BYTES = 128 << 20
MAX_JSON_VALUES = 3_000_000
MAX_ROWS = 40_000
MAX_TABLE_BYTES = 16 << 20
MAX_CELL_BYTES = 8192
MAX_SECONDS = 30
MAX_DIRECTORIES = 32
MAX_OMISSIONS = 128
STAGES = ("accuracy", "scaling", "prepare", "neural", "experiment", "audit")
FILES = ("metrics.json", "references.json", "training.json", "candidates.json", "frontiers.json", "comparisons.json",
         "summary.json", "protocol.json", "execution.json", "stage.json", "checks.json", "consistency.json")
SCHEMAS = {"tdn.premix-mechanisms/v1", "tdn.premix-neural/v1", "tdn.premix-comparisons/v1"}
PREFIX = ["source_path", "source_record", "stage", "record_type"]
GROUP_COUNTS = ("total", "ours_feasible", "baseline_feasible", "both_feasible", "timing_eligible",
    "ours_only_feasible", "baseline_only_feasible", "trained_pair_eligible", "initialization_in_eligible_pair",
    "ours_feasible_baseline_unresolved", "baseline_feasible_ours_unresolved",
    "incomplete_or_invalid", "ours_wins", "ours_ties", "ours_losses", "trained_pair_wins", "trained_pair_ties",
    "trained_pair_losses", "range_separated_wins", "range_separated_losses")
CONTEXT = ["candidate_id", "frontier_id", "reference_id", "parity_id", "case_id", "parent_id",
           "case_role", "panel", "split", "regime", "category", "distribution", "pattern", "grid", "batch_size",
           "cells", "seed", "family", "variant", "device", "dtype", "state_precision"]
COLUMNS = {
    "premix_candidates": PREFIX + CONTEXT + [
        "status", "training_attempted", "steps", "nsteps", "h", "horizon", "final_time",
        "trajectory_completed", "completed_steps", "finite", "in_bounds", "admissible", "memory_ok", "setup_seconds",
        "error_rms", "error_max", "error_mean", "error_mean_abs", "error_spatial_rms", "error_spatial_max",
        "error_upper_rms", "error_upper_max_estimate", "reference_accepted", "reference_uncertainty_rms",
        "reference_uncertainty_max_estimate", "preparation_seconds", "prepared_median_seconds",
        "prepared_min_seconds", "prepared_max_seconds", "setup_inclusive_median_seconds",
        "wall_seconds_median", "cuda_event_seconds_median", "peak_allocated_bytes", "peak_reserved_bytes",
        "throughput_members_per_second", "work_fft_total", "work_fft_total_fields", "work_fft_transformed_cells",
        "work_pair_product_cells", "work_pair_kernel_values", "work_pair_chunks", "work_parent_chunks",
        "cached_tensor_bytes",
        "parameter_count", "selected_step", "training_status", "failure_reason", "reason",
        "member_errors_record", "timing_record", "warmup_record", "work_record", "cache_metadata_record"],
    "premix_frontiers": PREFIX + CONTEXT + [
        "status", "norm", "target", "tolerance", "selected_steps", "selected_candidate_id", "setup_selected_candidate_id",
        "seconds", "error_upper", "prepared_median_seconds", "setup_inclusive_median_seconds",
        "selected_nsteps", "setup_selected_nsteps", "selected_error_upper", "setup_selected_error_upper",
        "selected_error", "adjusted_error", "reference_uncertainty", "prepared_seconds", "setup_inclusive_seconds",
        "setup_selected_seconds", "setup_selected_error", "setup_selected_adjusted_error", "feasible_candidate_count",
        "selection_scope", "reference_accepted", "reference_uncertainty_rms", "reference_uncertainty_max_estimate",
        "throughput_members_per_second", "failure_reason", "reason"],
    "premix_checks": PREFIX + CONTEXT + [
        "status", "check", "passed", "accepted", "reference_accepted", "horizon", "final_time", "h",
        "nsteps", "uncertainty_rms", "uncertainty_max_estimate", "reference_defect_rms", "reference_defect_max",
        "max_difference", "rms_difference", "allowed_absolute_difference", "relative_rms_when_resolved",
        "relative_resolution_threshold", "mean_difference_abs", "spatial_rms_difference",
        "retained_low_output_rms_difference", "reference_seconds",
        "reference_n", "reference_2n", "reference_4n", "difference_n_2n", "difference_2n_4n", "observed_order",
        "rhs_evaluations", "nodes",
        "source_cache_id", "reference_state_sha256", "member_references_record", "attempts_record",
        "failure_reason", "reason", "reference_reason"],
    "premix_training": PREFIX + ["family", "seed", "status", "selection", "step", "steps", "selected_step",
        "selected_parameters_changed", "parameter_count", "training_seconds", "training_loss", "validation_loss",
        "best_validation_loss", "sample_schedule_sha256", "diagnostics_seen_during_training",
        "admissible_validation", "checkpoint_sha256", "architecture_record", "failure_reason", "reason"],
    "premix_comparisons": PREFIX + ["comparison_id", "parent_id", "regime", "category", "distribution", "grid", "seed",
        "ours", "baseline", "norm", "target", "status", "ours_feasible", "baseline_feasible", "timing_eligible",
        "ours_selection", "baseline_selection", "trained_pair_eligible", "outcome", "speedup_baseline_over_ours",
        "robust_speed_win", "robust_speed_loss", "range_status", "issues_record",
        *[side + "_" + field for side in ("ours", "baseline") for field in ("status", "selected_step",
          "selected_steps", "seconds", "frontier_record", "candidate_record", "training_record",
          "raw_min_seconds", "raw_max_seconds", "repeat_count", "device")]],
    "premix_groups": PREFIX + ["group_scope", "group", "seed", "ours", "baseline", "norm", "target",
        *["count_" + field for field in GROUP_COUNTS], "parent_ids_record"],
}
LABELS = {"premix_candidates": "Premix accuracy, cost and stress cases",
          "premix_frontiers": "Premix posthoc tolerance frontiers",
          "premix_checks": "Premix reference quality and mechanism checks",
          "premix_training": "Premix training and validation selection",
          "premix_comparisons": "Premix paired matched-tolerance comparisons",
          "premix_groups": "Premix predeclared category and regime endpoints"}


def _directory(path):
    path = Path(os.path.abspath(path))
    for component in reversed((path, *path.parents)):
        if not stat.S_ISDIR(component.lstat().st_mode):
            raise ValueError(f"directory required; symlinks refused: {component}")
    return path


def _relative(path, report):
    return Path(os.path.relpath(path, report)).as_posix()


class _Budget:
    def __init__(self):
        self.started = time.monotonic()
        self.read_bytes = 0
        self.omissions = []
        self.omission_count = 0
        self.snapshots = {}

    def omit(self, path, reason):
        self.omission_count += 1
        if len(self.omissions) < MAX_OMISSIONS:
            self.omissions.append({"path": str(path)[:4096], "reason": str(reason)[:300]})

    def read(self, path, maximum=MAX_FILE_BYTES):
        _directory(path.parent)
        expected = path.lstat()
        if not stat.S_ISREG(expected.st_mode):
            raise ValueError("source must be a regular file; symlinks refused")
        if expected.st_size > maximum or self.read_bytes + expected.st_size > MAX_READ_BYTES:
            raise ValueError("premix source byte budget exceeded")
        if time.monotonic() - self.started > MAX_SECONDS:
            raise ValueError("premix source time budget exceeded")
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
        try:
            before = os.fstat(fd)
            raw = bytearray()
            while len(raw) <= maximum:
                part = os.read(fd, min(65536, maximum + 1 - len(raw)))
                if not part:
                    break
                raw.extend(part)
                self.read_bytes += len(part)
                if self.read_bytes > MAX_READ_BYTES or time.monotonic() - self.started > MAX_SECONDS:
                    raise ValueError("premix source read budget exceeded")
            after = os.fstat(fd)
            named = path.lstat()
            identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
            if (not stat.S_ISREG(before.st_mode) or len(raw) != before.st_size
                    or len(raw) > maximum or identity(expected) != identity(before)
                    or identity(before) != identity(after) or identity(after) != identity(named)):
                raise ValueError("premix source changed while reporting")
            self.snapshots[path] = identity(named)
            return bytes(raw)
        finally:
            os.close(fd)


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
    data = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    stack, count = [data], 0
    while stack:
        value = stack.pop()
        count += 1
        if count > MAX_JSON_VALUES:
            raise ValueError("premix JSON value budget exceeded")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("nonfinite JSON number")
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    if not isinstance(data, dict):
        raise ValueError("canonical premix document must be an object")
    return data


def _cell(value):
    if not isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    # One physical line per CSV row; JSON pointers preserve the original string.
    value = "".join("\\n" if char == "\n" else "\\r" if char == "\r" else
                    "\\t" if char == "\t" else f"\\u{ord(char):04x}" if ord(char) < 32 else char
                    for char in value)
    if value.lstrip().startswith(("=", "+", "-", "@")):
        value = "'" + value
    return value


def _csv(path, rows, columns, budget):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, columns, lineterminator="\n")
    writer.writeheader()
    size, count = len(stream.getvalue().encode()), 0
    if size > MAX_TABLE_BYTES:
        raise ValueError("premix table budget cannot contain the CSV header")
    for row in rows:
        start = stream.tell()
        writer.writerow({key: _cell(value) for key, value in row.items() if key in columns})
        stream.seek(start)
        row_size = len(stream.read().encode())
        if count >= MAX_ROWS or size + row_size > MAX_TABLE_BYTES:
            stream.seek(start)
            stream.truncate()
            budget.omit(path.name, "premix table budget exceeded; remaining rows retained in canonical JSON")
            break
        size += row_size
        count += 1
        stream.seek(0, io.SEEK_END)
    if path.exists() or path.is_symlink():
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError("CSV destination must be a regular file")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(stream.getvalue().encode())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return count


def _rows(document, key, path, budget):
    rows = document.get(key)
    if not isinstance(rows, list):
        budget.omit(path, f"canonical {key} must be an array")
        return []
    return rows


def _row(table, raw, pointer, source, stage, record_type, tables, budget, *, columns=None):
    columns = COLUMNS if columns is None else columns
    if not isinstance(raw, dict):
        budget.omit(f"{source}#{pointer}", "canonical row must be an object")
        return
    if len(tables[table]) >= MAX_ROWS:
        budget.omit(source, f"{table} row budget exceeded; raw rows retained")
        return
    selected = {key: value for key, value in raw.items() if key in columns[table] and key not in PREFIX}
    structured = {key for key, value in selected.items() if isinstance(value, (dict, list)) and key != "grid"}
    if "target_mean" in structured and "target_mean_record" in columns[table]:
        selected.pop("target_mean")
        structured.remove("target_mean")
    if structured:
        budget.omit(f"{source}#{pointer}", f"structured scalar fields refused: {sorted(structured)}")
        return
    if "grid" in selected and (not isinstance(selected["grid"], list)
            or not 1 <= len(selected["grid"]) <= 3 or any(type(n) is not int or n < 1 for n in selected["grid"])):
        budget.omit(f"{source}#{pointer}", "invalid grid")
        return
    aliases = {"rms": "error_rms", "max_error": "error_max", "mean_error": "error_mean",
               "spatial_rms": "error_spatial_rms", "upper_rms": "error_upper_rms",
               "upper_max": "error_upper_max_estimate", "uncertainty_rms": "reference_uncertainty_rms",
               "uncertainty_max_bound": "reference_uncertainty_max_estimate"}
    if table == "premix_checks":
        aliases = {"uncertainty_max_bound": "uncertainty_max_estimate"}
    for old, new in aliases.items():
        if old in raw and new in columns[table] and new not in selected:
            value = raw[old]
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
                budget.omit(f"{source}#{pointer}", f"invalid numeric field {old}")
                return
            selected[new] = value
    timing = raw.get("timing")
    if table == "premix_candidates":
        for source_dict, pairs in ((timing, [(key, key) for key in ("wall_seconds_median", "cuda_event_seconds_median",
                                   "peak_allocated_bytes", "peak_reserved_bytes")]),
                (raw.get("work_per_rollout"), [(key, "work_" + key) for key in ("fft_total", "fft_total_fields",
                    "fft_transformed_cells", "pair_product_cells", "pair_kernel_values", "pair_chunks", "parent_chunks")]),
                (raw.get("cache_metadata"), [("cached_tensor_bytes", "cached_tensor_bytes")])):
            if isinstance(source_dict, dict):
                for key, column in pairs:
                    if key in source_dict:
                        value = source_dict[key]
                        if value is not None and (type(value) not in (int, float) or value < 0):
                            budget.omit(f"{source}#{pointer}", f"invalid numeric cost field {key}")
                            return
                        selected[column] = value
    if table == "premix_groups":
        counts = raw.get("counts")
        if not isinstance(counts, dict) or any(type(counts.get(key)) is not int or counts[key] < 0 for key in GROUP_COUNTS):
            budget.omit(f"{source}#{pointer}", "invalid premix grouped counts")
            return
        selected.update({"count_" + key: counts[key] for key in GROUP_COUNTS})
    for key, column in (("member_errors", "member_errors_record"), ("timing", "timing_record"),
                        ("timing_repeats", "timing_record"), ("work_per_rollout", "work_record"),
                        ("warmup", "warmup_record"),
                        ("cache_metadata", "cache_metadata_record"), ("member_references", "member_references_record"),
                        ("attempts", "attempts_record"), ("architecture", "architecture_record"),
                        ("issues", "issues_record"), ("parent_ids", "parent_ids_record"),
                        *[(key, key + "_record") for key in ("observed_orders", "missing_parameter_gradients",
                          "nonfinite_parameter_gradients", "time_gradients", "target_mean", "final_mean",
                          "ours_diagnostic_issues", "baseline_diagnostic_issues", "mean_state", "variance",
                          "physical_mean_derivative")]):
        if key in raw and column in columns[table]:
            selected[column] = pointer + "/" + key
    selected.update(source_path=source, source_record=pointer, stage=stage,
                    record_type=record_type)
    if any(len(_cell(value).encode()) > MAX_CELL_BYTES for value in selected.values()):
        budget.omit(f"{source}#{pointer}", "premix CSV cell budget exceeded; raw row retained")
        return
    tables[table].append(selected)


def _seal(root, raw_documents, budget):
    """Verify only the canonical files read here, never deserialize checkpoints."""
    if not (root / "COMPLETED").exists():
        return "UNSEALED"
    try:
        raw = budget.read(root / "manifest.json", 1 << 20)
        marker = budget.read(root / "COMPLETED", 256).decode().strip()
        if marker != hashlib.sha256(raw).hexdigest():
            raise ValueError("COMPLETED does not match the manifest")
        manifest = _decode(raw)
        files = manifest.get("files")
        if not isinstance(files, dict):
            raise ValueError("manifest has no file digest mapping")
        for name, content in raw_documents.items():
            # Coordinator lifecycle metadata changes after sealing by design.
            if name in {"stage.json", "execution.json"} and name not in files:
                continue
            expected = files.get(name)
            if isinstance(expected, dict):
                expected = expected.get("sha256")
            if expected != hashlib.sha256(content).hexdigest():
                raise ValueError(f"canonical source {name} differs from or is absent from the seal")
        return "PROJECTED_CANONICAL_FILES_VERIFIED"
    except (OSError, ValueError, UnicodeError) as error:
        budget.omit(root, f"premix seal verification failed: {error}")
        return "INVALID"


def _consistency(root, documents, budget, *, neural_schema="tdn.premix-neural/v1",
                 comparison_schema="tdn.premix-comparisons/v1"):
    """Cross-check declared counts against canonical arrays, without inference."""
    before = budget.omission_count
    summary = documents.get("summary.json", {})
    mechanism = documents.get("metrics.json", {})
    if mechanism.get("schema") == "tdn.premix-mechanisms/v1":
        if summary.get("status") == "COMPLETED" and summary.get("status") != mechanism.get("status"):
            budget.omit(root, "summary status differs from canonical metrics status")
        coverage = summary.get("coverage", {})
        if not isinstance(coverage, dict):
            budget.omit(root, "summary coverage must be an object")
            coverage = {}
        ids = {"candidate_rows": "candidate_id", "frontier_rows": "frontier_id",
               "reference_rows": "reference_id", "parity_rows": "parity_id"}
        for key, identity in ids.items():
            rows = mechanism.get(key)
            if not isinstance(rows, list):
                continue  # The row projector records the malformed array.
            stated = coverage.get(key)
            if stated is not None and (not isinstance(stated, dict) or stated.get("reported") != len(rows)
                    or (summary.get("status") == "COMPLETED" and stated.get("expected") != len(rows))):
                budget.omit(root, f"summary coverage differs from canonical {key}")
            identifiers = [row.get(identity) for row in rows if isinstance(row, dict) and identity in row]
            if any(not isinstance(value, str) for value in identifiers):
                budget.omit(root, f"invalid {identity}")
            elif len(set(identifiers)) != len(identifiers):
                budget.omit(root, f"duplicate canonical {identity}")
    counts = summary.get("counts", {})
    if not isinstance(counts, dict):
        budget.omit(root, "summary counts must be an object")
        return False
    neural = {name: doc.get("rows") for name, doc in documents.items()
              if doc.get("schema") == neural_schema and isinstance(doc.get("rows"), list)}
    def check(name, actual):
        if name in counts and (type(counts[name]) is not int or counts[name] != actual):
            budget.omit(root, f"summary {name} differs from canonical rows")
    if "references.json" in neural:
        rows = neural["references.json"]
        check("references", len(rows))
        check("accepted_references", sum(isinstance(row, dict) and row.get("accepted") is True for row in rows))
        check("parents", len({row.get("parent_id") for row in rows
                              if isinstance(row, dict) and isinstance(row.get("parent_id"), str)}))
    if "training.json" in neural:
        rows = neural["training.json"]
        check("training_records", len(rows))
        for count, key, value in (("trained_checkpoints", "selection", "TRAINED_CHECKPOINT"),
                                  ("selected_initializations", "selection", "SELECTED_INITIALIZATION"),
                                  ("numerical_training_failures", "status", "NUMERICAL_FAILURE")):
            check(count, sum(isinstance(row, dict) and row.get(key) == value for row in rows))
    if "candidates.json" in neural:
        rows = neural["candidates.json"]
        check("candidates", len(rows))
        check("candidate_rows", len(rows))
        check("completed_candidates", sum(isinstance(row, dict) and row.get("status") == "COMPLETED" for row in rows))
        # Diagnostic coverage can be intentionally partial after interruption.
        if summary.get("status") == "COMPLETED":
            check("diagnostic_parents", len({row.get("parent_id") for row in rows
                if isinstance(row, dict) and isinstance(row.get("parent_id"), str)}))
    if neural_schema == "tdn.consistency-neural/v1":
        if "consistency.json" in neural:
            check("trained_consistency_checks", len(neural["consistency.json"]))
        if "checks.json" in neural:
            rows = neural["checks.json"]
            identifiers = [row.get("check_id") for row in rows if isinstance(row, dict)]
            if (any(not isinstance(value, str) or not value for value in identifiers)
                    or len(identifiers) != len(rows) or len(set(identifiers)) != len(identifiers)):
                budget.omit(root, "missing or duplicate structural check identity")
            if "case_count" in summary and (type(summary["case_count"]) is not int or summary["case_count"] != len(rows)):
                budget.omit(root, "audit case count differs from canonical checks")
            coverage = summary.get("coverage")
            if coverage is not None and (not isinstance(coverage, dict) or coverage.get("reported") != len(rows)
                    or (summary.get("status") == "COMPLETED" and coverage.get("expected") != len(rows))):
                budget.omit(root, "audit coverage differs from canonical checks")
            outcomes = summary.get("outcomes")
            if outcomes is not None:
                actual = {}
                for row in rows:
                    if isinstance(row, dict):
                        status = row.get("status")
                        if isinstance(status, str):
                            actual[status] = actual.get(status, 0) + 1
                if not isinstance(outcomes, dict) or outcomes != actual:
                    budget.omit(root, "audit outcomes differ from canonical checks")
            if "correctness_failures" in summary:
                failures = sum(isinstance(row, dict) and row.get("status") == "FAILED" for row in rows)
                if type(summary["correctness_failures"]) is not int or summary["correctness_failures"] != failures:
                    budget.omit(root, "audit correctness failures differ from canonical checks")
    comparisons = documents.get("comparisons.json", {})
    if comparisons.get("schema") == comparison_schema and isinstance(comparisons.get("rows"), list):
        rows = comparisons["rows"]
        coverage = comparisons.get("coverage", {})
        declared = coverage.get("comparisons", {}) if isinstance(coverage, dict) else {}
        if not isinstance(declared, dict) or declared.get("reported") != len(rows) or declared.get("expected") != len(rows):
            budget.omit(root, "comparison coverage differs from canonical rows")
        counts = comparisons.get("counts", {})
        if not isinstance(counts, dict) or counts.get("total") != len(rows):
            budget.omit(root, "comparison total differs from canonical rows")
        identifiers = [row.get("comparison_id") for row in rows if isinstance(row, dict)]
        if any(not isinstance(value, str) for value in identifiers) or len(set(identifiers)) != len(identifiers):
            budget.omit(root, "missing or duplicate comparison identity")
    return budget.omission_count == before


def publish_premix_outputs(report_dir, source_dirs, *, delegated_sources=None,
                          _suite="premix", _schemas=None, _extra_columns=None,
                          _neural_schema="tdn.premix-neural/v1",
                          _comparison_schema="tdn.premix-comparisons/v1"):
    """Return a small premix result, or None when no premix sources are present.

    Separate tables, limits and manifests preserve existing Tower exports. Any
    unsupported, truncated or corrupted source is explicit; it cannot become a
    successful scientific outcome. Sources and model checkpoints are read-only.
    """
    schemas = SCHEMAS if _schemas is None else set(_schemas)
    columns = {name: list(values) for name, values in COLUMNS.items()}
    for name, extra in (_extra_columns or {}).items():
        columns[name].extend(key for key in extra if key not in columns[name])
    output_name = lambda name: name.replace("premix", _suite, 1)
    report = _directory(report_dir)
    budget = _Budget()
    directories, seen = [], set()
    for supplied in source_dirs:
        root = _directory(supplied)
        if root == report or report in root.parents:
            raise ValueError("premix source cannot be the reporting directory or its child")
        for candidate in (root, *(root / name for name in STAGES)):
            if candidate in seen or not candidate.exists():
                continue
            seen.add(candidate)
            if len(directories) >= MAX_DIRECTORIES:
                budget.omit(candidate, "premix directory limit exceeded")
                continue
            try:
                directories.append(_directory(candidate))
            except (OSError, ValueError) as error:
                budget.omit(candidate, str(error))
    tables = {name: [] for name in COLUMNS}
    sources, inventory = [], []
    recognized = False
    for root in directories:
        documents, raw_documents, errors = {}, {}, []
        for name in FILES:
            path = root / name
            if not path.exists() and not path.is_symlink():
                continue
            try:
                raw = budget.read(path)
                documents[name] = _decode(raw)
                raw_documents[name] = raw
            except (OSError, ValueError, UnicodeError, RecursionError) as error:
                errors.append((path, str(error)))
        canonical = {name: doc for name, doc in documents.items() if doc.get("schema") in schemas}
        is_premix = bool(canonical) or any(str(doc.get("schema", "")).startswith("tdn." + _suite)
                        or _suite in str(doc.get("benchmark_suite", "")) for doc in documents.values())
        if not is_premix:
            continue
        recognized = True
        for path, reason in errors:
            budget.omit(_relative(path, report), reason)
        for name, document in documents.items():
            if name in {"metrics.json", "references.json", "training.json", "candidates.json", "frontiers.json", "comparisons.json", "checks.json", "consistency.json"} and document.get("schema") not in schemas:
                budget.omit(_relative(root / name, report), "unsupported canonical premix schema")
        summary = documents.get("summary.json", {})
        stage = summary.get("stage") or documents.get("metrics.json", {}).get("panel") or root.name
        if not isinstance(stage, str):
            budget.omit(root, "premix stage must be text")
            stage = root.name
        before = {table: len(rows) for table, rows in tables.items()}
        for name, doc in canonical.items():
            source = _relative(root / name, report)
            digest = hashlib.sha256(raw_documents[name]).hexdigest()
            inventory.append({"path": source, "bytes": len(raw_documents[name]), "sha256": digest,
                              "hash_status": "computed", "schema": doc["schema"]})
            if delegated_sources is not None:
                delegated_sources[root / name] = {"sha256": digest,
                    "signature": budget.snapshots[root / name], "schema": doc["schema"]}
            if doc["schema"] == "tdn.premix-mechanisms/v1" and name == "metrics.json":
                specifications = (("candidate_rows", "premix_candidates", "candidate"),
                                  ("frontier_rows", "premix_frontiers", "frontier"),
                                  ("reference_rows", "premix_checks", "reference"),
                                  ("parity_rows", "premix_checks", "parity"))
            elif doc["schema"] == _neural_schema and name in {
                    "candidates.json", "frontiers.json", "references.json", "training.json"}:
                table, kind = {"candidates.json": ("premix_candidates", "candidate"),
                               "frontiers.json": ("premix_frontiers", "frontier"),
                               "references.json": ("premix_checks", "reference"),
                               "training.json": ("premix_training", "final")}[name]
                specifications = (("rows", table, kind),)
            elif doc["schema"] == _comparison_schema and name == "comparisons.json":
                specifications = (("rows", "premix_comparisons", "comparison"),
                                  ("groups", "premix_groups", "group"))
            elif _suite == "consistency" and name in {"checks.json", "consistency.json"}:
                specifications = (("rows", "premix_checks", "structural" if name == "checks.json" else "postfreeze"),)
            else:
                continue
            for key, table, kind in specifications:
                raw_rows = _rows(doc, key, source, budget)
                for index, raw_row in enumerate(raw_rows):
                    pointer = f"/{key}/{index}"
                    _row(table, raw_row, pointer, source, stage, kind, tables, budget, columns=columns)
                    if table == "premix_training" and isinstance(raw_row, dict):
                        history = raw_row.get("history", [])
                        if not isinstance(history, list):
                            budget.omit(source + "#" + pointer, "training history must be an array")
                            continue
                        for step, record in enumerate(history):
                            # Every plotted history observation retains its exact source pointer.
                            if isinstance(record, dict):
                                record = {"family": raw_row.get("family"), "seed": raw_row.get("seed"), **record}
                            _row(table, record, f"{pointer}/history/{step}", source, stage,
                                 "history", tables, budget, columns=columns)
        status = summary.get("status", documents.get("metrics.json", {}).get("status", "UNKNOWN"))
        if status == "COMPLETED":
            expected_files = {"neural": {"training.json", "candidates.json", "frontiers.json"},
                              "prepare": {"references.json"}, "dataset": {"references.json"},
                              "accuracy": {"metrics.json"}, "scaling": {"metrics.json"}, "audit": {"checks.json"}}.get(stage, set())
            if _suite == "consistency" and stage == "neural":
                expected_files |= {"comparisons.json", "consistency.json"}
            for missing in sorted(expected_files - canonical.keys()):
                budget.omit(_relative(root / missing, report), "completed premix stage lacks required canonical table")
        consistent = _consistency(root, documents, budget, neural_schema=_neural_schema,
                                  comparison_schema=_comparison_schema)
        seal = _seal(root, raw_documents, budget)
        if status == "COMPLETED" and seal == "UNSEALED":
            budget.omit(root, "source declares completion without a COMPLETED seal")
        sources.append({"source_dir": _relative(root, report), "stage": stage, "status": status,
                        "canonical_status": documents.get("metrics.json", {}).get("status"),
                        "scientific_outcome": summary.get("scientific_outcome",
                            documents.get("metrics.json", {}).get("scientific_outcome")),
                        "seal_status": seal, "declared_counts_consistent": consistent,
                        "projected_rows": {output_name(name): len(rows) - before[name] for name, rows in tables.items()}})
    if not recognized:
        return None
    outputs = report / "outputs"
    outputs.mkdir(exist_ok=True)
    _directory(outputs)
    row_counts = {output_name(name): _csv(outputs / (output_name(name) + ".csv"), rows, columns[name], budget)
                  for name, rows in tables.items()}
    logs = read_json(report / "logs.json").get("logs", [])
    registered = {item.get("path") for item in logs if isinstance(item, dict)}
    for name, label in LABELS.items():
        path = f"outputs/{output_name(name)}.csv"
        if path in registered:
            continue
        try:
            register_log(report, "analytics." + output_name(name), path, label=label.replace("Premix", _suite.title()), group="Scientific outputs",
                         description=_suite.title() + " canonical JSON projection; predeclared regimes and paired parent IDs retained")
        except (OSError, ValueError) as error:
            budget.omit(path, f"log registration unavailable: {error}")
    metadata = {"schema": f"tdn.{_suite}.tower-tables/v1", "path_base": "Tower report directory",
                "columns": {output_name(name): value for name, value in columns.items()}, "rows": row_counts,
                "units": {"*_seconds": "seconds", "errors": "absolute solution units", "grid": "cells per axis"},
                "source_record": "RFC 6901 pointer into source_path; source_sha256 below hashes each canonical source once",
                "source_sha256": {item["path"]: item["sha256"] for item in inventory},
                "empty_cells": "absent field; JSON null is the literal null; booleans are true or false",
                "string_encoding": "Control characters escaped; formula-leading strings prefixed with apostrophe; canonical JSON unchanged",
                "normalized_aliases": {"rms": "error_rms", "max_error": "error_max", "mean_error": "error_mean",
                    "spatial_rms": "error_spatial_rms", "upper_rms": "error_upper_rms", "upper_max": "error_upper_max_estimate",
                    "uncertainty_rms": "reference_uncertainty_rms (candidates)",
                    "uncertainty_max_bound": "reference_uncertainty_max_estimate (candidates), uncertainty_max_estimate (checks)"},
                "nested_cost_fields": "work_* fields come from work_per_rollout; cached_tensor_bytes from cache_metadata; neural wall/CUDA/memory timing fields from timing",
                "comparison_pointers": "ours/baseline_*_record resolve into sibling frontiers.json, candidates.json or training.json as named; other *_record pointers resolve into source_path",
                "group_fields": "count_* come from each group's counts; parent_ids_record retains the paired parent list; grouped counts are descriptive endpoints",
                "scope": "Periodic logistic reaction-diffusion. FNO is a representative shared-physics hybrid, not a reproduction of the FNO paper.",
                "interpretation": "Regimes are declared before evaluation. Repeated parents across seeds, grids, step counts and methods are paired observations, not independent evidence. Frontiers are reference-informed posthoc selections, not deployable adaptive policies.",
                "limits": {"rows_per_table": MAX_ROWS, "bytes_per_table": MAX_TABLE_BYTES,
                           "bytes_per_cell": MAX_CELL_BYTES,
                           "per_file_read_bytes": MAX_FILE_BYTES, "total_read_bytes": MAX_READ_BYTES,
                           "json_values_per_document": MAX_JSON_VALUES, "source_read_seconds": MAX_SECONDS}}
    atomic_json(outputs / (_suite + "-tables.json"), metadata)
    artifact_report = {"schema": f"tdn.{_suite}.tower-artifacts/v1", "path_base": "Tower report directory",
                       "artifacts": inventory, "observed_read_bytes": budget.read_bytes,
                       "omissions": budget.omissions, "omission_count": budget.omission_count,
                       "omissions_list_truncated": budget.omission_count > len(budget.omissions),
                       "seal_scope": "COMPLETED, manifest and the canonical JSON files read by this publisher only; model and dataset binaries are never deserialized or independently verified here"}
    atomic_json(outputs / (_suite + "-artifacts.json"), artifact_report)
    result = {"schema": f"tdn.{_suite}.tower-results/v1", "application_completion_is_scientific_success": False,
              "reporting_complete": budget.omission_count == 0, "reporting_omission_count": budget.omission_count,
              "table_rows": row_counts, "sources": sources, "science_sources_mutated": False,
              "claim_scope": "Bounded mechanism diagnostics and representative neural comparison; no automatic FNO-paper or out-of-domain superiority claim"}
    atomic_json(outputs / (_suite + "-results.json"), result)
    return result
