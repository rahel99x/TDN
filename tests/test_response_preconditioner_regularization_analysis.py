"""Synthetic audit-report tests: no teacher acquisition or scientific fitting."""
from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest

from tdn.analysis.response_preconditioner.regularization_analysis import metric, render, summarize


def _row(query="q0", parent="p0", **changes):
    row = dict(query_id=query, parent_id=parent, parent_family="smooth_spectral",
               split="diagnostic", role="withheld", family="constant", start=.02, stop=.04)
    return dict(row, **changes)


def _metric(error, *, query="q0", parent="p0", method="learned", **metadata):
    error = np.asarray(error, dtype=float)
    teacher = np.ones_like(error) * 4
    return metric(_row(query, parent), method, teacher + error, teacher,
                  teacher - 2, teacher - 1, .01, .02,
                  matched_mean_uncertainty_rms=.01, kind="selected", **metadata)


def test_physical_error_components_and_signed_mean():
    value = _metric([1., 3.])
    assert value["status"] == "MEASURED"
    assert value["mse"] == 5.
    assert value["rms"] == math.sqrt(5)
    assert value["maximum"] == 3.
    assert value["mean_error"] == 2.
    assert value["mean_component_mse"] == 4.
    assert value["spatial_centered_mse"] == value["centered_mse"] == 1.
    assert value["spatial_centered_rms"] == 1.
    assert value["relative_reference_rms"] == math.sqrt(5) / 4
    assert value["relative_remainder_rms"] == math.sqrt(5) / 2
    assert value["relative_feedback_rms"] == math.sqrt(5)
    assert value["feedback_floor_rms"] == .2
    assert value["reference_uncertainty_is_bound"] is False


def test_uniform_physical_array_shapes_are_supported():
    array = np.arange(6.).reshape(2, 3)
    value = metric(_row(), "m", array, np.zeros((2, 3)), np.zeros((2, 3)),
                   None, 0., 0.)
    assert value["mse"] == np.mean(array ** 2)
    assert value["relative_reference_rms"] is None
    assert value["reference_status"] == "UNRESOLVED_NEAR_NULL"


@pytest.mark.parametrize("bad", [None, [np.nan, 0], [np.inf, 0], [1j, 0], [], ["bad", "data"]])
def test_invalid_prediction_retains_failed_row_without_nonfinite_payload(bad):
    result = metric(_row(), "m", bad, [1., 1.], [0., 0.], [0., 0.], .01, .01)
    assert result["status"] == "FAILED"
    assert result["query_id"] == "q0"
    assert result["mse"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("uncertainty", [-1., float("nan"), float("inf"), True, "bad"])
def test_invalid_uncertainty_is_a_retained_failure(uncertainty):
    result = metric(_row(), "m", [1., 1.], [1., 1.], [0., 0.], None, uncertainty, .01)
    assert result["status"] == "FAILED"


def test_mismatched_shapes_and_overflow_fail_without_partial_metrics():
    shape = metric(_row(), "m", [1., 2.], [[1., 2.]], [[0., 0.]], None, .01, .01)
    overflow = metric(_row(), "m", [1e308, 1e308], [-1e308, -1e308], [0., 0.], None, .01, .01)
    for result in (shape, overflow):
        assert result["status"] == "FAILED"
        assert result["rms"] is None
        json.dumps(result, allow_nan=False)


def test_near_null_ratio_has_explicit_denominator_and_strict_floor():
    result = metric(_row(), "m", [1.], [.1], [0.], [0.], .01, .01,
                    matched_mean_uncertainty_rms=0.)
    assert result["reference_rms"] == result["reference_floor_rms"] == .1
    assert result["reference_status"] == "UNRESOLVED_NEAR_NULL"
    assert result["relative_reference_rms"] is None
    assert result["relative_remainder_rms"] is None
    assert result["relative_feedback_rms"] is None


def test_feedback_uncertainty_is_not_silently_assumed_zero():
    result = metric(_row(), "m", [3., 3.], [2., 2.], [0., 0.], [1., 1.], .01, .02)
    assert result["status"] == "MEASURED"
    assert result["feedback_rms"] == 1.
    assert result["feedback_status"] == "UNRESOLVED_REFERENCE"
    assert result["feedback_floor_rms"] is None
    assert result["relative_feedback_rms"] is None
    assert result["relative_remainder_rms"] == .5


def test_missing_reference_estimate_does_not_erase_physical_error():
    result = metric(_row(), "m", [3., 3.], [2., 2.], [0., 0.], None, None, None)
    assert result["status"] == "MEASURED"
    assert result["mse"] == 1.
    assert result["relative_reference_rms"] is None
    assert result["reference_status"] == "UNRESOLVED_REFERENCE"


def test_opposing_query_means_never_become_spatial_error():
    result = summarize([_metric([1., 1.], query="a"), _metric([-1., -1.], query="b")])
    for row in [result["learning_curves"][0], result["parent_metrics"][0]]:
        assert row["mean_error"] == 0.
        assert row["mse"] == row["mean_component_mse"] == 1.
        assert row["spatial_centered_mse"] == row["spatial_centered_rms"] == 0.


def test_parent_weighting_is_independent_of_descendant_query_count():
    rows = [_metric([1., 1.], query=f"a{i}", parent="a") for i in range(3)]
    rows += [_metric([3., 3.], query="b", parent="b")]
    result = summarize(rows)["learning_curves"][0]
    assert result["mse"] == (1 + 9) / 2
    assert result["mean_error"] == 2.
    assert result["mean_component_mse"] == 5.
    assert result["spatial_centered_mse"] == 0.
    assert result["expected"] == result["measured"] == 4
    assert result["expected_parent_count"] == result["complete_parent_count"] == 2


def test_mixed_mean_and_centered_components_decompose_at_both_levels():
    rows = [_metric([0., 2.], query="a0", parent="a"),
            _metric([-2., 0.], query="a1", parent="a"),
            _metric([0., 6.], query="b0", parent="b")]
    result = summarize(rows)
    for row in result["learning_curves"] + result["parent_metrics"]:
        assert row["mse"] == pytest.approx(row["mean_component_mse"] + row["spatial_centered_mse"])
    group = result["learning_curves"][0]
    assert group["mse"] == 10.
    assert group["mean_component_mse"] == group["spatial_centered_mse"] == 5.
    assert group["mean_error"] == 1.5


def test_planned_failed_and_missing_rows_preserve_all_denominators():
    rows = [_metric([1., 1.], query="ok", parent="a")]
    for query, status in [("missing", "NOT_EVALUATED"), ("failure", "FAILED"), ("unresolved", "UNRESOLVED")]:
        rows.append(dict(_row(query, "b"), method="learned", kind="selected", status=status,
                         mse=None, rms=None, training_membership="not_applicable"))
    result = summarize(rows)["learning_curves"][0]
    assert (result["expected"], result["measured"], result["failed"], result["missing"], result["unresolved"]) == (4, 1, 1, 1, 1)
    assert not result["complete"]
    assert result["mse"] is None and result["mean_component_mse"] is None
    assert result["available_parent_balanced_rms"] == 1.
    assert result["missing_parent_ids"] == ["b"]


@pytest.mark.parametrize("field,value", [("mse", float("nan")), ("rms", 1e308),
                                         ("mean_component_mse", 10.), ("spatial_centered_mse", "bad")])
def test_invalid_measured_rows_count_as_failed(field, value):
    row = _metric([1., 1.])
    row[field] = value
    result = summarize([row])["learning_curves"][0]
    assert result["failed"] == result["invalid_measured_queries"] == 1
    assert result["measured"] == 0
    assert result["rms"] is None


def test_ratio_aggregation_cannot_hide_unresolved_query_denominator():
    good = _metric([1., 1.], query="a")
    near = metric(_row("b"), "learned", [1.], [0.], [0.], [0.], .01, .01,
                  matched_mean_uncertainty_rms=.01, kind="selected")
    group = summarize([good, near])["learning_curves"][0]
    assert group["complete"]
    assert group["reference_rms"] > group["reference_floor_rms"]
    assert group["reference_status"] == "UNRESOLVED_MEMBER"
    assert group["relative_reference_rms"] is None


def test_cohort_memberships_and_roles_stay_separate():
    a = _metric([1., 1.], query="a", training_membership="in_cohort")
    b = _metric([2., 2.], query="b", training_membership="outside_cohort")
    a["split"] = b["split"] = "train"
    c = dict(a, query_id="c", role="calibration")
    assert len(summarize([a, b, c])["learning_curves"]) == 3


def test_duplicate_and_cross_split_rows_are_rejected():
    a = _metric([1., 1.])
    with pytest.raises(ValueError, match="Duplicate"):
        summarize([a, a])
    b = dict(a, query_id="q1", split="train")
    with pytest.raises(ValueError, match="cross splits"):
        summarize([a, b])


def test_method_metadata_inconsistency_is_rejected():
    a = _metric([1., 1.], query="a", training_parent_count=8)
    b = _metric([1., 1.], query="b", training_parent_count=32)
    with pytest.raises(ValueError, match="Inconsistent method metadata"):
        summarize([a, b])


def test_render_csv_and_only_finite_bounded_research_rows(tmp_path):
    rows = [_metric([1., 3.])]
    selection = {"cells": [{"cell_id": "p8s8", "candidates": [
        dict(name="ridge", family="coefficient_ridge", ridge=1e-10, validation_mse=1., eligible=True,
             diagnostics={"data_loss": .5, "ridge_loss": .2, "arbitrary": 99., "gradient_norm": float("nan")}),
        dict(name="failed", family="action_energy", ridge=1e-6, validation_mse=None, eligible=False,
             diagnostics=None)]}]}
    result = render(tmp_path, rows, selection,
                    {"fitting_seconds": 2., "solver_seconds": 1., "validation_seconds": .5,
                     "penalty_setup_seconds": .1, "original_reference_seconds": 100.,
                     "component_scope": "Overlapping costs", "gpu_utilization": 100}, plots=False)
    assert result["learning_curves"][0]["mse"] == 5.
    with (tmp_path / "metrics.csv").open() as stream:
        assert list(csv.DictReader(stream))[0]["mean_component_mse"] == "4.0"
    series = json.loads((tmp_path / "regularization_metrics.json").read_text())
    assert len(series) == 4
    for row in series:
        assert set(row) == {"phase_suffix", "step", "values"}
        assert len(row["phase_suffix"]) <= 128
        assert all(isinstance(value, (int, float)) and math.isfinite(value) for value in row["values"].values())
        assert not {"arbitrary", "gpu_utilization"} & row["values"].keys()
    assert series[2]["values"]["eligible"] == 0
    costs = series[-1]["values"]
    assert costs["solver_seconds"] == 1.
    assert costs["original_reference_seconds"] == 100.
    assert "component_scope" not in costs


def test_research_series_budget_rejects_instead_of_truncating(tmp_path):
    selection = {"cells": [dict(cell_id="cell", candidates=[dict(family="f", eligible=False)] * 501)]}
    with pytest.raises(ValueError, match="budget exceeded"):
        render(tmp_path, [], selection, {}, plots=False)


def test_empty_render_retains_empty_tables_and_no_false_success(tmp_path):
    assert render(tmp_path, [], {}, {}, plots=False) == {"learning_curves": [], "parent_metrics": []}
    assert json.loads((tmp_path / "regularization_metrics.json").read_text()) == []


def test_optional_figures_use_only_retained_synthetic_metrics(tmp_path):
    candidate = dict(name="ridge", family="coefficient_ridge", ridge=1e-6, validation_mse=.5, eligible=True)
    selection = dict(cells=[dict(cell_id="p8s8", candidates=[candidate], selected={"coefficient_ridge": candidate})])
    render(tmp_path, [_metric([1., 3.])], selection, {}, plots=True)
    assert (tmp_path / "figures" / "diagnostic_errors.png").read_bytes().startswith(b"\x89PNG")
    assert "<svg" in (tmp_path / "figures" / "diagnostic_errors.svg").read_text()
    assert (tmp_path / "figures" / "validation_paths.png").read_bytes().startswith(b"\x89PNG")
