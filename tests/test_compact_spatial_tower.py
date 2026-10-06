"""Compact spatial reporting retains bounded, complete, non-affirmative evidence."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from tdn import tower_analytics as analytics
from test_work_precision_tower import document as work_precision_document, make_report, read_rows, resolve


TABLES = analytics.COMPACT_SPATIAL_TABLES


def document():
    value = work_precision_document()
    value.update(schema="tdn.compact-spatial/v1", benchmark_suite="compact-spatial",
                 scientific_outcome="OBSERVED_MIXED")
    for field in ("candidate_rows", "frontier_rows", "reference_rows", "parity_rows"):
        for row in value[field]:
            row.update(case_role="new_diagnostic", panel="accuracy", batch_size=4, cells=16)
    for row in value["candidate_rows"]:
        valid = row["status"] == "VALID"
        row.update(error_spatial_rms=.00009 if valid else None,
                   error_spatial_max=.00019 if valid else None,
                   error_mean_abs=.00001 if valid else None,
                   error_rms_worst_member=row["error_rms"], error_max_worst_member=row["error_max"],
                   throughput_members_per_second=4 / .003 if valid else None)
        row["member_errors"] = [{"member_index": index, **{key: row[key] for key in (
            "error_rms", "error_max", "error_mean", "error_spatial_rms", "error_spatial_max",
            "error_upper_rms", "error_upper_max_estimate")}} for index in range(4)] if valid else []
    for row in value["frontier_rows"]:
        row["selection_scope"] = "posthoc_declared_grid_worst_member"
    for row in value["reference_rows"]:
        row["member_references"] = [{"member_index": index, "reference_accepted": True,
            "uncertainty_rms": row["uncertainty_rms"],
            "uncertainty_max_estimate": row["uncertainty_max_estimate"]} for index in range(4)]
    parity = value["parity_rows"][0]
    parity["variant"] = "gl3"
    for variant in ("compact_midpoint", "compact_secant", "compact_local", "compact_filter"):
        value["parity_rows"].append({**parity,
            "parity_id": f"fresh-0/approximation/{variant}", "check": "approximation",
            "variant": variant, "passed": None, "status": "OBSERVED", "max_difference": .002,
            "rms_difference": .001, "reference_defect_rms": .003, "reference_defect_max": .004,
            "relative_rms_when_resolved": 1 / 3, "mean_difference_abs": 1e-15,
            "spatial_rms_difference": .001, "relative_resolution_threshold": 1e-12})
    return value


def report(tmp_path, payload=None):
    return make_report(tmp_path, document() if payload is None else payload, name="compact-spatial.json")


def test_complete_export_preserves_diagnostics_provenance_nulls_and_terminal_outcome(tmp_path):
    payload = document()
    source, target, canonical = report(tmp_path, payload)
    before = hashlib.sha256(canonical.read_bytes()).hexdigest()
    result = analytics.publish_outputs(target, [source])
    summary, = result["reports"]
    assert result["reporting_omission_count"] == 0
    assert result["scientific_outcomes"] == []
    assert summary["reporting_complete"] is True
    assert summary["scientific_outcome"] == "OBSERVED_MIXED"
    assert summary["projected_counts"] == {"candidate": 2, "frontier": 2, "reference": 1, "parity": 5}
    assert summary["parity_status_counts"] == {"PASS": 1, "FAIL": 0, "UNAVAILABLE": 0, "OBSERVED": 4}
    assert summary["scientific_gate_authorized"] is summary["neural_superiority_claim"] is False
    for table in TABLES:
        for row in read_rows(target, table):
            original = resolve(payload, row["source_record"])
            assert original["case_id"] == row["case_id"]
            assert original["panel"] == row["panel"]
            assert int(row["batch_size"]) == original["batch_size"]
            assert row["source_path"].endswith("/compact-spatial.json")
    candidate, failed = read_rows(target, "compact_spatial")
    assert candidate["error_rms"] == candidate["error_rms_worst_member"]
    assert candidate["timing_repeats_record"] == "/candidate_rows/0/timing_repeats"
    assert candidate["member_errors_record"] == "/candidate_rows/0/member_errors"
    assert failed["error_spatial_rms"] == failed["throughput_members_per_second"] == "null"
    assert failed["setup_inclusive_median_seconds"] == ""
    checks = read_rows(target, "compact_spatial_checks")
    assert checks[0]["member_references_record"] == "/reference_rows/0/member_references"
    assert len([row for row in checks if row["status"] == "OBSERVED" and row["passed"] == "null"]) == 4
    assert result["table_rows"]["work_precision"] == result["table_rows"]["mechanisms"] == 0
    metadata = json.loads((target / "outputs/tables.json").read_text())
    assert "worst member" in metadata["compact_spatial_interpretation"]
    contract = json.loads((Path(__file__).parents[1] / ".tower/contracts/outputs.v1.json").read_text())
    for table in TABLES:
        entry, = [row for row in contract["outputs"] if row["path"] == f"outputs/{table}.csv"]
        assert entry["columns"] == analytics.COLUMNS[table] == metadata["columns"][table]
        assert entry["required"] is False and entry["max_rows"] == analytics.MAX_COMPACT_SPATIAL_ROWS
        assert entry["max_bytes"] == analytics.MAX_COMPACT_SPATIAL_TABLE_BYTES
    analytics.publish_outputs(target, [source])
    logs = json.loads((target / "logs.json").read_text())["logs"]
    assert sum(row["id"].startswith("analytics.compact_spatial") for row in logs) == 3
    assert hashlib.sha256(canonical.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("field,value", [
    ("schema", "tdn.compact-spatial/v2"), ("benchmark_suite", "work-precision"),
    ("device", "cuda"), ("training_attempted", True), ("training_performed", True),
    ("status", "RUNNING"), ("scientific_outcome", "SUPERIOR"), ("scientific_outcome", None),
    ("candidate_rows", None), ("frontier_rows", {}), ("parity_rows", {}),
])
def test_unsupported_document_is_explicitly_incomplete(tmp_path, field, value):
    payload = document()
    payload[field] = value
    source, target, canonical = report(tmp_path, payload)
    original = canonical.read_bytes()
    result = analytics.publish_outputs(target, [source])
    assert result["scientific_outcomes"] == ["COMPACT_SPATIAL_INCOMPLETE"]
    assert result["reports"][0]["reporting_complete"] is False
    assert all(result["table_rows"][table] == 0 for table in TABLES)
    assert canonical.read_bytes() == original


@pytest.mark.parametrize("field,key,value", [
    ("candidate_rows", "panel", "other"), ("candidate_rows", "cells", 17),
    ("candidate_rows", "batch_size", True), ("candidate_rows", "error_spatial_rms", -.1),
    ("candidate_rows", "error_rms_worst_member", {"value": .1}),
    ("candidate_rows", "throughput_members_per_second", "fast"),
    ("candidate_rows", "case_role", "fresh"), ("frontier_rows", "selected_candidate_id", "missing"),
    ("parity_rows", "status", "OBSERVED"), ("parity_rows", "passed", None),
])
def test_malformed_rows_cannot_become_complete_evidence(tmp_path, field, key, value):
    payload = document()
    payload[field][0][key] = value
    source, target, _ = report(tmp_path, payload)
    result = analytics.publish_outputs(target, [source])
    assert result["scientific_outcomes"] == ["COMPACT_SPATIAL_INCOMPLETE"]
    assert result["reports"][0]["unsupported_row_count"] >= 1
    assert result["reports"][0]["reporting_complete"] is False


def test_inconclusive_science_is_preserved_independently_of_computational_completion(tmp_path):
    payload = document()
    payload["scientific_outcome"] = "INCONCLUSIVE"
    rejected = copy.deepcopy(payload["reference_rows"][0])
    rejected.update(case_id="excluded-reference", reference_id="excluded-reference/reference",
                    reference_accepted=False, uncertainty_rms=None)
    for row in rejected["member_references"]:
        row.update(reference_accepted=False, uncertainty_rms=None)
    payload["reference_rows"].append(rejected)
    source, target, _ = report(tmp_path, payload)
    result = analytics.publish_outputs(target, [source])
    assert result["scientific_outcomes"] == ["COMPACT_SPATIAL_INCONCLUSIVE"]
    assert result["reports"][0]["status"] == "COMPLETED"
    assert result["reports"][0]["reporting_complete"] is True
    assert result["reports"][0]["rejected_reference_count"] == 1


def test_frontier_pairs_each_error_with_its_own_member_uncertainty(tmp_path):
    payload = document()
    reference = payload["reference_rows"][0]
    for member, rms, maximum in zip(reference["member_references"],
                                     (1e-5, 6e-4, 0., 0.), (2e-5, 7e-4, 0., 0.)):
        member.update(uncertainty_rms=rms, uncertainty_max_estimate=maximum)
    reference.update(uncertainty_rms=6e-4, uncertainty_max_estimate=7e-4)
    for candidate in payload["candidate_rows"]:
        candidate.update(reference_uncertainty_rms=6e-4, reference_uncertainty_max_estimate=7e-4)
    candidate = payload["candidate_rows"][0]
    for member, ref, error in zip(candidate["member_errors"], reference["member_references"],
                                  (8e-4, 1e-4, 2e-4, 3e-4)):
        member.update(error_rms=error, error_max=2*error, error_mean=0., error_spatial_rms=error,
                      error_spatial_max=2*error, error_upper_rms=error+ref["uncertainty_rms"],
                      error_upper_max_estimate=2*error+ref["uncertainty_max_estimate"])
    for key in ("error_rms", "error_max", "error_spatial_rms", "error_spatial_max", "error_upper_rms",
                "error_upper_max_estimate"):
        candidate[key] = max(row[key] for row in candidate["member_errors"])
    candidate.update(error_mean=0., error_mean_abs=0., error_rms_worst_member=candidate["error_rms"],
                     error_max_worst_member=candidate["error_max"])
    frontier = payload["frontier_rows"][0]
    frontier.update(selected_error=candidate["error_rms"], reference_uncertainty=6e-4,
                    adjusted_error=candidate["error_upper_rms"], setup_selected_error=candidate["error_rms"],
                    setup_selected_adjusted_error=candidate["error_upper_rms"])
    assert frontier["adjusted_error"] < frontier["tolerance"] < frontier["selected_error"] + frontier["reference_uncertainty"]
    source, target, canonical = report(tmp_path, payload)
    result = analytics.publish_outputs(target, [source])
    assert result["reports"][0]["reporting_complete"] is True
    assert result["reporting_omission_count"] == 0
    assert float(read_rows(target, "compact_spatial_frontiers")[0]["adjusted_error"]) == frontier["adjusted_error"]
    # A relabeled upper error that loses this pairing must fail projection.
    candidate["member_errors"][0]["error_upper_rms"] = candidate["error_rms"] + reference["uncertainty_rms"]
    canonical.write_text(json.dumps(payload))
    result = analytics.publish_outputs(target, [source])
    assert result["reports"][0]["reporting_complete"] is False
    assert result["scientific_outcomes"] == ["COMPACT_SPATIAL_INCOMPLETE"]


def test_partial_member_references_and_unstarted_endpoint_are_retained(tmp_path):
    payload = document()
    payload.update(status="INCOMPLETE", computational_status="INCOMPLETE", scientific_outcome="INCONCLUSIVE")
    reference = payload["reference_rows"][0]
    reference.update(reference_accepted=False, uncertainty_rms=None, uncertainty_max_estimate=None)
    reference["member_references"][-1].update(reference_accepted=False, uncertainty_rms=None,
                                             uncertainty_max_estimate=None)
    candidate = payload["candidate_rows"][1]
    candidate.update(status="INCOMPLETE", reference_accepted=False, reference_uncertainty_rms=None,
                     reference_uncertainty_max_estimate=None, warmup=None, timing_repeats=[])
    payload.update(candidate_rows=[candidate], frontier_rows=[], parity_rows=[])
    source, target, _ = report(tmp_path, payload)
    result = analytics.publish_outputs(target, [source])
    assert result["reporting_omission_count"] == 0
    assert result["reports"][0]["reporting_complete"] is True
    assert result["scientific_outcomes"] == ["COMPACT_SPATIAL_INCOMPLETE", "COMPACT_SPATIAL_INCONCLUSIVE"]
    row, = read_rows(target, "compact_spatial")
    assert row["status"] == "INCOMPLETE" and row["reference_accepted"] == "false"
    assert row["error_rms"] == "null" and row["member_errors_record"] == "/candidate_rows/0/member_errors"


@pytest.mark.parametrize("limit_name,value", [("MAX_COMPACT_SPATIAL_ROWS", 1),
                                               ("MAX_COMPACT_SPATIAL_TABLE_BYTES", 1000)])
def test_output_budget_cannot_silently_truncate_a_complete_experiment(tmp_path, monkeypatch, limit_name, value):
    source, target, canonical = report(tmp_path)
    original = canonical.read_bytes()
    monkeypatch.setattr(analytics, limit_name, value)
    result = analytics.publish_outputs(target, [source])
    assert result["reporting_omission_count"] > 0
    assert result["reports"][0]["reporting_complete"] is False
    assert result["scientific_outcomes"] == ["COMPACT_SPATIAL_INCOMPLETE"]
    assert canonical.read_bytes() == original


def test_canonical_has_separate_read_reserve_and_exact_filename_override(tmp_path):
    source, target, canonical = report(tmp_path)
    size = canonical.stat().st_size
    budget = analytics._Budget()
    budget.read_bytes = analytics.MAX_READ_BYTES
    assert len(budget.read(canonical, canonical.stat())) == size
    assert budget.compact_read_bytes == size
    other = source / "Compact-spatial.json"
    other.write_bytes(canonical.read_bytes())
    with pytest.raises(ValueError, match="read budget exceeded"):
        budget.read(other, other.stat())
    assert analytics._file_read_limit(other) == analytics.MAX_FILE_BYTES
    canonical.write_bytes(canonical.read_bytes() + b" " * analytics.MAX_COMPACT_SPATIAL_FILE_BYTES)
    result = analytics.publish_outputs(target, [source])
    assert result["scientific_outcomes"] == ["COMPACT_SPATIAL_INCOMPLETE"]
    assert result["reports"][0]["reporting_complete"] is False


def test_full_scale_fixture_retains_all_rows_repeats_and_member_evidence(tmp_path):
    payload = document()
    candidate, frontier, reference = (copy.deepcopy(payload[key][0])
        for key in ("candidate_rows", "frontier_rows", "reference_rows"))
    payload.update(candidate_rows=[], frontier_rows=[], reference_rows=[], parity_rows=[])
    # 60 cases x 10 methods x 6 schedules: 3600 candidates, 4800 frontiers.
    for case in range(60):
        case_id = f"scaling/case-{case:03d}/batch16"
        payload["reference_rows"].append({**reference, "case_id": case_id,
            "reference_id": case_id + "/reference", "panel": "scaling", "batch_size": 16,
            "member_references": [{**reference["member_references"][0], "member_index": i} for i in range(16)]})
        for method in range(10):
            variant = f"control-{method}"
            for nsteps in (4, 8, 16, 32, 64, 128):
                row = copy.deepcopy(candidate)
                row.update(case_id=case_id, variant=variant, nsteps=nsteps, panel="scaling", batch_size=16,
                           candidate_id=f"{case_id}/{variant}/n{nsteps}")
                row["timing_repeats"] *= 2
                row["member_errors"] = [{**candidate["member_errors"][0], "member_index": i} for i in range(16)]
                payload["candidate_rows"].append(row)
            for norm in ("rms", "max"):
                for tolerance in (.001, .002, .004, .008):
                    payload["frontier_rows"].append({**frontier, "case_id": case_id, "variant": variant,
                        "panel": "scaling", "batch_size": 16, "norm": norm, "tolerance": tolerance,
                        "selected_candidate_id": f"{case_id}/{variant}/n8",
                        "setup_selected_candidate_id": f"{case_id}/{variant}/n8",
                        "selected_error": candidate["error_" + norm],
                        "adjusted_error": candidate["error_upper_rms" if norm == "rms" else "error_upper_max_estimate"],
                        "setup_selected_error": candidate["error_" + norm],
                        "setup_selected_adjusted_error": candidate["error_upper_rms" if norm == "rms" else "error_upper_max_estimate"],
                        "frontier_id": f"{case_id}/{variant}/{norm}/t{tolerance}"})
    source, target, canonical = report(tmp_path, payload)
    original = hashlib.sha256(canonical.read_bytes()).hexdigest()
    assert analytics.MAX_WORK_PRECISION_FILE_BYTES < canonical.stat().st_size < analytics.MAX_COMPACT_SPATIAL_FILE_BYTES
    result = analytics.publish_outputs(target, [source])
    assert result["reporting_omission_count"] == 0
    assert result["reports"][0]["reporting_complete"] is True
    assert result["reports"][0]["projected_counts"] == {"candidate": 3600, "frontier": 4800,
                                                       "reference": 60, "parity": 0}
    assert hashlib.sha256(canonical.read_bytes()).hexdigest() == original
    assert all((target / "outputs" / f"{table}.csv").stat().st_size < analytics.MAX_COMPACT_SPATIAL_TABLE_BYTES
               for table in TABLES)
