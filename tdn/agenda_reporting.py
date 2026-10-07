"""Finite, source-linked Tower views of the research agenda.

The canonical stage rows remain authoritative. Reporting never trains a model,
selects a checkpoint, evaluates a policy, or derives scientific success from
application completion. The Tower application is used without modification.
"""
from __future__ import annotations

import hashlib
import csv
import io
import json
import os
from pathlib import Path
import stat
import uuid

from tdn.premix_reporting import (_Budget, _cell, _decode, _directory,
                                 _relative, _seal)
from tdn.reporting import atomic_json, read_json, register_log

SCHEMA = "tdn.agenda-rows/v1"
STAGES = ("structure", "prepare", "controls", "optimize", "compression", "kernel", "confirm", "policy")
MAX_READ_BYTES = 64 << 20
MAX_FILE_BYTES = 64 << 20
MAX_ROWS = 40_000
MAX_TABLE_BYTES = 16 << 20
MAX_OUTPUT_BYTES = 64 << 20
MAX_DIRECTORIES = 16
PREFIX = ["source_path", "source_record", "source_sha256", "stage", "record_type"]
CONTEXT = ["row_id", "case_id", "reference_id", "candidate_id", "comparison_id", "trial_id", "parent_id", "physical_parent_id",
           "continuous_field_id", "grid_size", "panel", "question_id", "row_kind", "cohort", "schedule_index",
           "split", "regime", "family", "variant", "seed", "grid", "track", "status", "selection", "checkpoint_selection", "required", "passed",
           "reason", "failure_reason", "device", "dtype", "h", "horizon", "nsteps", "reaction", "diffusion",
           "mean", "amplitude", "variance", "phase", "spectral_gap", "maximum_eigenvalue"]
ERROR = ["rms", "max_error", "error_rms", "error_max", "signed_mean_error", "mean_error", "spatial_rms", "spatial_max", "upper_rms",
         "upper_max", "uncertainty_rms", "uncertainty_max", "uncertainty_max_bound", "reference_accepted", "reference_seconds",
         "base_rms", "base_max_error", "base_spatial_rms", "rms_vs_base", "spatial_vs_base", "spatial_regression",
         "observed_order", "relative_error", "residual_rms", "quadratic_error", "epsilon", "uncertainty_after_scaling",
         "base_error_rms", "base_error_max", "base_error_mean", "base_error_spatial_rms",
         "teacher_error_rms", "amplitude_uncertainty", "spectral_gap_h", "largest_eigenvalue_h",
         "reaction_first_rms", "diffusion_first_rms", "reference_uncertainty", "reaction_first_limit_bias_rms",
         "coupled_to_homogenized_rms", "kernel_relative_rms", "field_relative_rms", "kernel_uncertainty_rms",
         "error_rms_vs_base", "error_max_vs_base", "mean_error_vs_base", "spatial_rms_vs_base",
         "selected_error_upper", "selection_rule", "target", "norm", "feasible", "in_bounds", "finite"]
COST = ["seconds", "timing_seconds", "wall_seconds", "wall_seconds_median", "cuda_event_seconds_median", "training_seconds",
        "setup_seconds", "prepared_seconds", "peak_allocated_bytes", "peak_reserved_bytes", "host_peak_rss_bytes",
        "soft_vram_budget_bytes", "fft_total", "fft_forward", "fft_inverse", "fft_fields", "fft_transformed_cells",
        "product_cells", "rhs_evaluations", "attempts", "completed_steps", "steps", "complete_rollout", "preparation_included",
        "coarse_seconds", "doubling_extra_seconds", "defect_extra_seconds", "all_estimator_attempt_seconds",
        "host_attempt_elapsed_seconds", "coarse_steps", "doubling_steps", "defect_steps", "defect_jvps"]
POINTERS = ["metrics_record", "timing_record", "work_record", "architecture_record", "history_record",
            "physics_record", "sources_record", "response_record", "spectrum_record", "phases_record",
            "schedule_record", "criteria_record", "validation_by_regime_record", "gradient_by_regime_record",
            "correction_by_regime_record", "fit_record", "singular_values_record", "rank_errors_record",
            "acceptance_record", "attempt_timings_record", "fallback_record", "issues_record", "question_ids_record",
            "factors_record", "same_step_ratios_record", "fft_counts_record", "training_record", "checkpoint_record",
            "candidate_record", "reference_record", "frontier_record", "oracle_spans_record", "rank_summary_record",
            "gradient_history_record", "schedule_record", "selected_schedule_record", "timing_samples_record",
            "estimates_record", "estimate_record", "costs_record", "classical_controls_record", "rank_gate_record"]
TRAIN = ["selection", "selected_step", "step", "updates", "batch_size", "learning_rate", "training_loss",
         "validation_loss", "initial_validation_loss", "best_validation_loss", "gradient_norm", "gradient_max",
         "correction_rms", "correction_max", "defect_rms", "parameter_count", "selected_parameters_changed",
         "checkpoint_sha256", "checkpoint_selection", "sample_schedule_sha256", "diagnostics_seen_during_training",
         "gradient_l2_before_clip", "gradient_l2_after_clip", "endpoint_correction_rms", "parameters_changed",
         "fresh_cohort_opened", "hyperparameter_source", "base_selection_trial"]
CAPACITY = ["parameter_count", "width", "depth", "spectral_layers", "retained_modes", "support", "rank",
            "requested_rank", "minimum_rank", "selected_rank", "rank_gate_passed", "input_support", "output_support",
            "high_frequency_bypass", "physical_gate_bypass", "actual_fft_support", "fft_count_per_step",
            "pair_symmetry_error", "real_output_error", "phase_error", "rank_gate_threshold", "rank_residual",
            "source_count", "source_before_filter", "time_decoder", "base_orientation", "fused_half_steps",
            "response_rank", "defect_separable_terms", "promising", "paired_selection_inconclusive",
            "source_parameter_count", "parameter_relative_difference", "source_timing_seconds", "cost_ratio",
            "actual_support_matches", "matching_data"]
POLICY = ["accepted", "rejected", "false_accept", "true_error", "estimated_error", "acceptance_threshold",
          "rejected_steps", "accepted_steps", "fallback_used", "fallback_reason", "fallback_seconds",
          "rejected_seconds", "accepted_seconds", "complete_seconds", "attempt_seconds", "rejected_work",
          "accepted_work", "fallback_work", "total_work", "reference_informed_selection", "policy_frozen",
          "estimator", "unequal_steps", "unseen_steps", "true_error_rms", "true_error_max",
          "false_accept_rms", "false_accept_max", "false_accept_joint", "false_accept_definitely_resolved", "accepted_attempt", "rejected_attempts",
          "fallback", "target_rms", "target_max", "joint_target_verified", "estimator_target", "attempt",
          "charged_seconds", "cumulative_seconds", "calibration_sha256", "checkpoint_freeze_sha256"]
COMPARE = ["ours", "baseline", "ours_selection", "baseline_selection", "trained_pair_eligible",
           "ours_feasible", "baseline_feasible", "both_feasible", "speedup_baseline_over_ours", "outcome",
           "ours_seconds", "baseline_seconds", "ours_error", "baseline_error", "ours_spatial_regression",
           "baseline_spatial_regression", "criteria_frozen", "diagnostic_fresh"]


def _columns(*groups):
    return list(dict.fromkeys([*PREFIX, *CONTEXT, *(item for group in groups for item in group), *POINTERS]))


COLUMNS = {
    "agenda_cases": _columns(ERROR, COST, CAPACITY),
    "agenda_references": _columns(ERROR, COST),
    "agenda_training": _columns(TRAIN, ERROR, COST, CAPACITY),
    "agenda_optimization": _columns(TRAIN, ERROR, COST),
    "agenda_capacity": _columns(CAPACITY, COST, TRAIN),
    "agenda_kernels": _columns(CAPACITY, ERROR, COST),
    "agenda_confirmation": _columns(ERROR, COST, CAPACITY, COMPARE),
    "agenda_policy": _columns(POLICY, ERROR, COST),
    "agenda_comparisons": _columns(COMPARE, ERROR, COST, CAPACITY),
}
TABLE_LIMITS = {name: {"rows": MAX_ROWS, "bytes": MAX_TABLE_BYTES} for name in COLUMNS}
# Full confirmation crosses 32 physical parents, three grids and several
# unequal schedules; retain its candidates and both error-norm frontiers.
TABLE_LIMITS["agenda_confirmation"] = {"rows": 80_000, "bytes": 48 << 20}
TABLE_FOR_KIND = {"case": "agenda_cases", "check": "agenda_cases", "reference": "agenda_references",
    "training": "agenda_training", "history": "agenda_training", "optimization": "agenda_optimization",
    "capacity": "agenda_capacity", "kernel": "agenda_kernels", "rank": "agenda_kernels",
    "confirmation": "agenda_confirmation", "frontier": "agenda_confirmation", "policy": "agenda_policy",
    "policy_calibration": "agenda_policy", "comparison": "agenda_comparisons"}


class _AgendaBudget(_Budget):
    def read(self, path, maximum=MAX_FILE_BYTES):
        if self.read_bytes + path.lstat().st_size > MAX_READ_BYTES:
            raise ValueError("agenda source read budget exceeded")
        return super().read(path, maximum)


def has_agenda_sources(roots):
    """Cheap recognition before importing new science-specific integrations.

    Older isolated controller fixtures intentionally do not copy new modules.
    This predicate is therefore also implemented inline in tower_analytics.
    """
    return any((Path(root) / "rows.json").exists() or any(
        (Path(root) / stage / "rows.json").exists() for stage in STAGES) for root in roots)


def _project(raw, pointer, source, digest, stage, tables, budget):
    if not isinstance(raw, dict):
        budget.omit(source + "#" + pointer, "agenda row must be an object")
        return
    kind = raw.get("record_type")
    if not isinstance(kind, str) or not kind:
        budget.omit(source + "#" + pointer, "agenda row requires a record_type")
        return
    table = TABLE_FOR_KIND.get(kind, "agenda_cases")
    selected = {}
    for key, value in raw.items():
        if key in COLUMNS[table] and key not in PREFIX:
            if isinstance(value, (dict, list)) and key != "grid":
                continue
            selected[key] = value
        if isinstance(value, (dict, list)) and key + "_record" in COLUMNS[table]:
            escaped = key.replace("~", "~0").replace("/", "~1")
            selected[key + "_record"] = pointer + "/" + escaped
    # Explicitly retained raw measurements can live in small named objects.
    # No aliases, units, success values, or missing values are invented here.
    for group in ("metrics", "timing", "work", "architecture"):
        value = raw.get(group)
        if isinstance(value, dict):
            for key, item in value.items():
                if key in COLUMNS[table] and key not in selected and key not in PREFIX and not isinstance(item, (dict, list)):
                    selected[key] = item
    selected.update(source_path=source, source_record=pointer, source_sha256=digest, stage=stage, record_type=kind)
    if any(len(_cell(value).encode()) > 8192 for value in selected.values()):
        budget.omit(source + "#" + pointer, "agenda CSV cell budget exceeded; complete canonical row retained")
        return
    if len(tables[table]) >= TABLE_LIMITS[table]["rows"]:
        budget.omit(source, "agenda row budget exceeded; complete canonical rows retained")
        return
    tables[table].append(selected)


def _header_bytes(columns):
    stream = io.StringIO(newline="")
    csv.DictWriter(stream, columns, lineterminator="\n").writeheader()
    return len(stream.getvalue().encode())


def _csv(path, items, columns):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, columns, lineterminator="\n")
    writer.writeheader()
    count = 0
    for item in items:
        writer.writerow({key: _cell(value) for key, value in item.items() if key in columns})
        count += 1
    if path.exists() or path.is_symlink():
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError("agenda CSV destination must be a regular file")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(stream.getvalue().encode())
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return count


def _fitting_rows(items, columns, allowance, budget, name):
    """Enforce the aggregate allowance before atomic CSV publication."""
    used = 0
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, columns, lineterminator="\n")
    for item in items:
        stream.seek(0); stream.truncate(0)
        writer.writerow({key: _cell(value) for key, value in item.items() if key in columns})
        size = len(stream.getvalue().encode())
        if used + size > allowance:
            budget.omit(name, "agenda table or aggregate byte budget exceeded; canonical rows retained")
            break
        used += size
        yield item


def publish_agenda_outputs(report_dir, source_dirs, *, delegated_sources=None):
    """Read-only canonical publication, with separate finite agenda budgets."""
    report = _directory(report_dir)
    budget = _AgendaBudget()
    directories, seen = [], set()
    for supplied in source_dirs:
        root = _directory(supplied)
        if root == report or report in root.parents:
            raise ValueError("agenda source cannot be the reporting directory or its child")
        for candidate in (root, *(root / stage for stage in STAGES)):
            if candidate in seen or not candidate.exists():
                continue
            seen.add(candidate)
            if len(directories) >= MAX_DIRECTORIES:
                budget.omit(candidate, "agenda directory budget exceeded")
                continue
            directories.append(_directory(candidate))
    tables = {name: [] for name in COLUMNS}
    inventory, sources = [], []
    recognized = False
    for root in directories:
        if not (root / "rows.json").exists() and not (root / "rows.json").is_symlink():
            continue
        recognized = True
        documents, contents = {}, {}
        for name in ("rows.json", "summary.json", "protocol.json", "execution.json", "stage.json"):
            path = root / name
            if not path.exists() and not path.is_symlink():
                continue
            try:
                raw = budget.read(path)
                document = _decode(raw)
                documents[name], contents[name] = document, raw
            except (OSError, ValueError, UnicodeError, RecursionError) as error:
                budget.omit(_relative(path, report), str(error))
        document = documents.get("rows.json", {})
        summary = documents.get("summary.json", {})
        if "summary.json" not in documents:
            budget.omit(root, "agenda canonical summary is missing or unreadable")
        stage = document.get("stage", summary.get("stage", root.name))
        rows = document.get("rows")
        if document.get("schema") != SCHEMA or stage not in STAGES or not isinstance(rows, list):
            budget.omit(_relative(root / "rows.json", report), "unsupported agenda rows schema, stage, or row list")
            rows = []
        source = _relative(root / "rows.json", report)
        digest = hashlib.sha256(contents["rows.json"]).hexdigest() if "rows.json" in contents else None
        before = {name: len(items) for name, items in tables.items()}
        for index, raw in enumerate(rows):
            _project(raw, f"/rows/{index}", source, digest, stage, tables, budget)
        identities = [row.get("row_id") for row in rows if isinstance(row, dict) and "row_id" in row]
        if identities and (any(not isinstance(item, str) or not item for item in identities)
                           or len(set(identities)) != len(identities)):
            budget.omit(source, "invalid or duplicate agenda row_id")
        counts = summary.get("counts", {})
        if isinstance(counts, dict) and "rows" in counts and (type(counts["rows"]) is not int or counts["rows"] != len(rows)):
            budget.omit(source, "agenda summary row count differs from canonical rows")
        coverage = summary.get("coverage")
        if isinstance(coverage, dict) and "rows" in coverage:
            declared = coverage["rows"]
            if (not isinstance(declared, dict) or declared.get("reported") != len(rows)
                    or (summary.get("status") == "COMPLETED" and declared.get("expected") != len(rows))):
                budget.omit(source, "agenda summary coverage differs from canonical rows")
        for name, raw in contents.items():
            path = root / name
            item = {"path": _relative(path, report), "sha256": hashlib.sha256(raw).hexdigest(),
                    "bytes": len(raw), "hash_status": "computed", "schema": documents[name].get("schema")}
            inventory.append(item)
            if delegated_sources is not None:
                delegated_sources[path] = {"sha256": item["sha256"], "signature": budget.snapshots[path],
                                           "schema": item["schema"]}
        status = summary.get("status", "UNKNOWN")
        seal = _seal(root, contents, budget)
        if status == "COMPLETED" and seal == "UNSEALED":
            budget.omit(root, "agenda completion requires a canonical source seal")
        if "stage" in summary and summary["stage"] != stage:
            budget.omit(root, "agenda canonical stage and summary stage differ")
        sources.append({"source_dir": _relative(root, report), "stage": stage, "status": status,
            "scientific_outcome": summary.get("scientific_outcome"), "seal_status": seal,
            "projected_rows": {name: len(items) - before[name] for name, items in tables.items()}})
    if not recognized:
        return None
    outputs = report / "outputs"
    outputs.mkdir(exist_ok=True)
    _directory(outputs)
    row_counts, output_bytes = {}, 0
    header_bytes = {name: _header_bytes(columns) for name, columns in COLUMNS.items()}
    remaining = MAX_OUTPUT_BYTES - sum(header_bytes.values())
    if remaining < 0:
        raise ValueError("agenda aggregate allowance cannot contain table headers")
    for name, items in tables.items():
        path = outputs / (name + ".csv")
        fitting = _fitting_rows(items, COLUMNS[name], min(TABLE_LIMITS[name]["bytes"] - header_bytes[name], remaining), budget, name)
        row_counts[name] = _csv(path, fitting, COLUMNS[name])
        remaining -= path.stat().st_size - header_bytes[name]
        output_bytes += path.stat().st_size
    if output_bytes > MAX_OUTPUT_BYTES:
        raise ValueError("agenda aggregate output accounting disagrees with published table bytes")
    registered = {item.get("path") for item in read_json(report / "logs.json").get("logs", []) if isinstance(item, dict)}
    for name in COLUMNS:
        path = "outputs/" + name + ".csv"
        if path not in registered:
            try:
                register_log(report, "analytics." + name, path, label=name.replace("_", " ").title(),
                             group="Scientific outputs", description="Agenda canonical rows, exact source pointers and hashes")
            except (OSError, ValueError) as error:
                budget.omit(path, f"log registration unavailable: {error}")
    atomic_json(outputs / "agenda-tables.json", {"schema": "tdn.agenda.tower-tables/v1", "columns": COLUMNS,
        "rows": row_counts, "units": {"*_seconds": "seconds", "*_bytes": "bytes", "errors": "solution units"},
        "source_sha256": {item["path"]: item["sha256"] for item in inventory},
        "source_record": "RFC 6901 pointer into source_path; the row SHA hashes its entire canonical document",
        "nested_records": "*_record points into the same source document; full row pointers retain unprojected fields",
        "empty_cells": "Absent field; null is literal null, booleans are true/false. No missing success or failure inferred.",
        "scope": "Per-stage finite development and frozen fresh confirmation. Parent/grid/seed repetitions remain paired.",
        "limits": {"read_bytes": MAX_READ_BYTES, "per_file_bytes": MAX_FILE_BYTES, "rows_per_table": MAX_ROWS,
                   "bytes_per_table": MAX_TABLE_BYTES, "table_overrides": TABLE_LIMITS,
                   "aggregate_table_bytes": MAX_OUTPUT_BYTES}})
    atomic_json(outputs / "agenda-artifacts.json", {"schema": "tdn.agenda.tower-artifacts/v1", "artifacts": inventory,
        "observed_read_bytes": budget.read_bytes, "omissions": budget.omissions, "omission_count": budget.omission_count,
        "seal_scope": "The canonical JSON documents read by this publisher; no checkpoints or binary states are deserialized"})
    result = {"schema": "tdn.agenda.tower-results/v1", "table_rows": row_counts, "sources": sources,
        "reporting_complete": budget.omission_count == 0, "reporting_omission_count": budget.omission_count,
        "application_completion_is_scientific_success": False, "science_sources_mutated": False,
        "claim_scope": "Scientific outcomes are canonical stage statements; no FNO-paper or solver superiority is inferred"}
    atomic_json(outputs / "agenda-results.json", result)
    return result


__all__ = ["publish_agenda_outputs", "COLUMNS", "SCHEMA", "STAGES"]
