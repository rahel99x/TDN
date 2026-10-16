"""Independent objective, split, missing-data and parent-unit checks for P1."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from tdn.analysis.response_preconditioner import scaling_analysis as analysis


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def row(parent, split, source, *, context=0, cells=5, role=None):
    role = role or ("calibration" if split == "train" else "withheld")
    seed = sum(map(ord, parent)) * 100 + context * 10 + source
    generator = np.random.default_rng(seed)
    design = generator.normal(size=(cells, 40)) * .2
    coefficients = np.linspace(-.02, .03, 40)
    base = generator.normal(size=cells)
    teacher = base + design @ coefficients
    return dict(query_id=f"{parent}-c{context}-{role}-{source}", parent_id=parent,
                parent_family="smooth" if parent[-1] in "02" else "localized",
                split=split, context_id=f"{parent}-c{context}", source_index=source,
                family="dense_random", role=role, start=0., stop=.08,
                status="ACCEPTED", design=design, teacher=teacher, base=base,
                matched_mean=base - .1,
                reference=dict(accepted=True, uncertainty_rms=1e-12, uncertainty_maximum=2e-12,
                               matched_mean=dict(accepted=True, uncertainty_rms=1e-12,
                                                 uncertainty_maximum=2e-12)))


def fixture_rows():
    training = [row(f"t{p}", "train", s) for p in range(2) for s in range(2)]
    validation = [row(f"v{p}", "validation", s) for p in range(2) for s in range(2)]
    diagnostic = [row(f"d{p}", "diagnostic", s) for p in range(2) for s in range(2)]
    return training, validation, diagnostic


def fit(training=None, validation=None, **kwargs):
    t, v, _ = fixture_rows()
    options = dict(training_parent_ids=["t0", "t1"], parent_counts=(1, 2), source_counts=(1, 2),
                   expected_contexts_per_parent=1, expected_withheld_sources=2)
    options.update(kwargs)
    return analysis.fit_nested_models(t if training is None else training,
                                      v if validation is None else validation, **options)


def test_nested_cohorts_use_declared_parent_order_and_original_candidates():
    training, validation, _ = fixture_rows()
    result = fit(training[::-1], validation)
    assert result["all_selections_sealed"]
    assert [c["model_id"] for c in result["cells"]] == [
        "fit-p001-s01", "fit-p001-s02", "fit-p002-s01", "fit-p002-s02"]
    for cell in result["cells"]:
        assert [c["name"] for c in cell["candidates"]] == ["zero_readout", "ridge_1e-10", "ridge_1e-06"]
        assert len(cell["train_query_ids"]) == cell["training_parent_count"] * cell["calibration_source_count"]
        assert not cell["diagnostic_truth_accessed"]
        assert not cell["training_withheld_truth_accessed"]
        for candidate in cell["candidates"][1:]:
            assert candidate["diagnostics"]["original_penalty_preserved"]
            assert candidate["diagnostics"]["parent_count"] == cell["training_parent_count"]
            assert candidate["fitting_seconds"] >= 0
    assert result["cells"][0]["training_parent_ids"] == ["t0"]
    json.dumps(result, allow_nan=False)


def test_coefficients_match_independent_augmented_numpy_objective_with_query_and_parent_mass():
    training, validation, _ = fixture_rows()
    training[0] = row("t0", "train", 0, cells=3)
    training[1] = row("t0", "train", 1, cells=7)
    result = fit(training, validation)
    cell = result["cells"][-1]
    design = np.concatenate([r["design"] for r in training])
    target = np.concatenate([r["teacher"] - r["base"] for r in training])
    omega = np.concatenate([np.full(r["teacher"].size, .25 / r["teacher"].size) for r in training])
    ridge = 1e-6
    augmented = np.vstack([np.sqrt(omega)[:, None] * design, np.sqrt(ridge) * np.eye(40)])
    rhs = np.concatenate([np.sqrt(omega) * target, np.zeros(40)])
    independent = np.linalg.lstsq(augmented, rhs, rcond=None)[0]
    candidate = cell["candidates"][2]
    np.testing.assert_allclose(candidate["coefficients"], independent, atol=2e-13, rtol=2e-11)
    parent_losses = []
    for parent in ("v0", "v1"):
        parent_losses.append(np.mean([np.mean((r["base"] + r["design"] @ independent - r["teacher"]) ** 2)
                                      for r in validation if r["parent_id"] == parent]))
    assert candidate["validation_loss"] == pytest.approx(np.mean(parent_losses), rel=1e-9)


def test_validation_equal_parent_mass_is_not_cell_count_weighting():
    training, validation, _ = fixture_rows()
    for index, r in enumerate(validation):
        cells = 1 if r["parent_id"] == "v0" else 9
        replacement = row(r["parent_id"], "validation", r["source_index"], cells=cells)
        replacement["teacher"] = replacement["base"] + (1 if cells == 1 else 3)
        validation[index] = replacement
    zero = fit(training, validation)["cells"][0]["candidates"][0]
    assert zero["validation_loss"] == 5.


def test_initialization_wins_exact_ties_and_seal_excludes_timing_only():
    training, validation, diagnostic = fixture_rows()
    for r in training + validation:
        r["teacher"] = r["base"].copy()
    cell = fit(training, validation)["cells"][0]
    assert cell["selected"]["name"] == "zero_readout"
    cell["fitting_seconds"] += 7
    cell["candidates"][0]["fitting_seconds"] += 2
    assert analysis.evaluate_model(cell, diagnostic)
    cell["selected"]["coefficients"][0] = 1.
    with pytest.raises(ValueError, match="seal"):
        analysis.evaluate_model(cell, diagnostic)


def test_unresolved_source_blocks_only_cells_that_would_use_it_and_preserves_denominator():
    training, validation, _ = fixture_rows()
    training[-1].update(status="UNRESOLVED", teacher=None)
    result = fit(training, validation)
    assert not result["all_selections_sealed"]
    assert [c["status"] for c in result["cells"]] == ["SELECTED", "SELECTED", "SELECTED", "BLOCKED"]
    blocked = result["cells"][-1]
    assert blocked["training_accounting"]["planned"] == 4
    assert blocked["training_accounting"]["unresolved"] == 1
    assert blocked["selection_seal"] is None and not blocked["candidates"]


@pytest.mark.parametrize("change", ["missing_source", "missing_parent", "missing_context", "nonfinite"])
def test_incomplete_or_invalid_training_is_never_silently_fitted(change):
    training, validation, _ = fixture_rows()
    kwargs = {}
    if change == "missing_source":
        training.pop()
    elif change == "missing_parent":
        training = training[:2]
    elif change == "missing_context":
        kwargs["expected_contexts_per_parent"] = 2
    else:
        training[-1]["design"][0, 0] = np.nan
    result = fit(training, validation, **kwargs)
    assert result["cells"][-1]["status"] == "BLOCKED"
    assert result["cells"][-1]["reasons"]
    json.dumps(result, allow_nan=False)


def test_missing_validation_source_blocks_every_cell():
    training, validation, _ = fixture_rows()
    result = fit(training, validation[:-1])
    assert all(c["status"] == "BLOCKED" for c in result["cells"])
    assert all("Incomplete validation sources" in c["reasons"][0] for c in result["cells"])


@pytest.mark.parametrize("contamination", ["diagnostic", "training_withheld", "parent_leak", "duplicate"])
def test_illegal_access_and_duplicate_queries_rejected(contamination):
    training, validation, diagnostic = fixture_rows()
    if contamination == "diagnostic":
        validation = diagnostic
    elif contamination == "training_withheld":
        training[0]["role"] = "withheld"
    elif contamination == "parent_leak":
        validation[0]["parent_id"] = "t0"
    else:
        training.append(training[0])
    with pytest.raises(ValueError):
        fit(training, validation)


def test_ineligible_solver_and_failed_candidate_retained_without_selection(monkeypatch):
    original = analysis.fit_parent_balanced_ridge
    def checked(*args, ridge, **kwargs):
        if ridge == 1e-6:
            raise ArithmeticError("fixture numerical failure")
        result = original(*args, ridge=ridge, **kwargs)
        result["diagnostics"]["optimality_verified"] = False
        return result
    monkeypatch.setattr(analysis, "fit_parent_balanced_ridge", checked)
    cell = fit()["cells"][0]
    assert cell["selected"]["name"] == "zero_readout"
    assert not cell["candidates"][1]["eligible"]
    assert "fixture numerical failure" in cell["candidates"][2]["error"]


def test_budget_stops_before_next_cell_and_callback_preserves_completed_cell():
    completed, calls = [], []
    def check():
        calls.append(None)
        if len(calls) == 4:
            raise TimeoutError("budget exhausted")
    with pytest.raises(TimeoutError, match="budget"):
        fit(budget_check=check, on_cell=completed.append)
    assert len(completed) == 1
    assert completed[0]["selection_seal"]


def test_metrics_preserve_denominators_uncertainty_and_training_membership():
    training, _, diagnostic = fixture_rows()
    cell = fit()["cells"][0]
    output = analysis.evaluate_model(cell, diagnostic + training)
    first = output[0]
    r = diagnostic[0]
    prediction = r["base"] + r["design"] @ cell["selected"]["coefficients"]
    expected = np.sqrt(np.mean((prediction - r["teacher"]) ** 2))
    assert first["rms"] == expected
    assert first["maximum"] == np.max(np.abs(prediction - r["teacher"]))
    assert first["relative_remainder_rms"] == pytest.approx(expected / np.sqrt(np.mean((r["teacher"]-r["base"])**2)))
    assert first["relative_feedback_rms"] is not None
    assert not first["reference_uncertainty_is_bound"]
    assert [r["training_membership"] for r in output[-4:]] == ["in_cohort", "in_cohort", "outside_cohort", "outside_cohort"]
    assert first["training_membership"] == "not_training_split"


def test_near_null_and_unresolved_feedback_never_report_relative_signal_error():
    _, _, diagnostic = fixture_rows()
    diagnostic[0]["teacher"] = diagnostic[0]["base"].copy()
    diagnostic[1]["reference"]["matched_mean"]["accepted"] = False
    records = analysis.evaluate_model(fit()["cells"][0], diagnostic)
    assert records[0]["remainder_status"] == "UNRESOLVED_NEAR_NULL"
    assert records[0]["relative_remainder_rms"] is None
    assert records[1]["feedback_status"] == "UNRESOLVED_REFERENCE"
    assert records[1]["relative_feedback_rms"] is None
    aggregate = analysis.summarize_learning_curves(records, expected_parent_ids={"diagnostic": ["d0", "d1"]})
    assert aggregate["groups"][0]["error_to_remainder_rms"] is None
    assert aggregate["groups"][0]["error_to_feedback_rms"] is None


@pytest.mark.parametrize("status", ["NOT_ACQUIRED", "UNRESOLVED", "FAILED", "NONFINITE"])
def test_evaluation_missing_failed_nonfinite_queries_stay_visible(status):
    _, _, diagnostic = fixture_rows()
    if status == "NONFINITE":
        diagnostic[0]["base"][0] = np.inf
    else:
        diagnostic[0]["status"] = status
    records = analysis.evaluate_model(fit()["cells"][0], diagnostic)
    assert len(records) == 4 and records[0]["status"] != "MEASURED"
    result = analysis.summarize_learning_curves(records, expected_parent_ids={"diagnostic": ["d0", "d1"]})
    assert not result["groups"][0]["complete"]
    assert result["groups"][0]["rms"] is None
    assert result["groups"][0]["complete_parent_count"] == 1
    assert len(result["parent_rows"]) == 2
    json.dumps(result, allow_nan=False)


def metric(parent, query, rms, method="fit-p001-s01", parents=1, sources=1, **kwargs):
    return dict(parent_id=parent, query_id=query, method=method, split="diagnostic", role="withheld",
                training_parent_count=parents, calibration_source_count=sources,
                status="MEASURED", rms=rms, maximum=2*rms, **kwargs)


def test_aggregation_weights_parents_equally_and_retains_unseen_expected_parent():
    rows = [metric("d0", "q0", 1.)] + [metric("d1", f"q{i}", 3.) for i in range(1, 10)]
    result = analysis.summarize_learning_curves(rows, expected_parent_ids={"diagnostic": ["d0", "d1"]})
    group = result["groups"][0]
    assert group["rms"] == pytest.approx(np.sqrt(5.))
    assert group["maximum"] == 6.
    assert group["median_parent_rms"] == 2.
    assert group["worst_parent_rms"] == 3.
    assert not group["descendant_queries_are_independent"]
    missing = analysis.summarize_learning_curves(rows, expected_parent_ids={"diagnostic": ["d0", "d1", "d2"]})
    assert missing["groups"][0]["rms"] is None
    assert missing["groups"][0]["missing_parent_ids"] == ["d2"]


def test_both_learning_axes_pair_fixed_diagnostic_parents_descriptively():
    rows = []
    for parents in (1, 2):
        for sources in (1, 2):
            method = f"fit-p{parents:03d}-s{sources:02d}"
            rows.extend([metric("d0", "q0", 1 / parents / sources, method, parents, sources),
                         metric("d1", "q1", 2 / parents / sources, method, parents, sources)])
    result = analysis.summarize_learning_curves(rows, expected_parent_ids={"diagnostic": ["d0", "d1"]})
    assert len(result["paired_parent_count"]) == len(result["paired_source_count"]) == 2
    for pair in result["paired_parent_count"] + result["paired_source_count"]:
        assert pair["complete"] and pair["paired_parent_count"] == 2
        assert pair["improved_parent_fraction"] == 1.
        assert pair["median_rms_difference"] < 0
        assert pair["descriptive_only"]
        assert all(p["improvement_ratio"] == 2. for p in pair["parents"])
    assert not result["automatic_expansion"] and not result["scientific_promotion"]


def test_training_membership_views_are_separate_and_never_make_paired_learning_claims():
    training, _, _ = fixture_rows()
    metrics = []
    for cell in fit()["cells"]:
        metrics += analysis.evaluate_model(cell, training)
    result = analysis.summarize_learning_curves(metrics, expected_parent_ids={"train": ["t0", "t1"]})
    assert {g["training_membership"] for g in result["groups"]} == {"in_cohort", "outside_cohort"}
    assert not result["paired_parent_count"] and not result["paired_source_count"]


@pytest.mark.parametrize("problem", ["duplicate", "unknown_split", "cross_split", "unknown_parent"])
def test_summary_rejects_ambiguous_denominators(problem):
    rows = [metric("d0", "q0", 1.)]
    expected = {"diagnostic": ["d0"]}
    if problem == "duplicate":
        rows *= 2
    elif problem == "unknown_split":
        rows[0]["split"] = "validation"
    elif problem == "cross_split":
        expected["train"] = ["d0"]
    else:
        rows[0]["parent_id"] = "other"
    with pytest.raises(ValueError):
        analysis.summarize_learning_curves(rows, expected_parent_ids=expected)


@pytest.mark.parametrize("kwargs", [dict(parent_counts=(2, 1)), dict(source_counts=(1, 1)),
                                     dict(ridge_menu=(1e-8,)), dict(training_parent_ids=["t0", "t0"]),
                                     dict(expected_withheld_sources=0)])
def test_unfrozen_or_ambiguous_fit_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        fit(**kwargs)


def test_authoritative_fresh_prediction_supplies_every_metric_and_does_not_fallback():
    _, _, diagnostic = fixture_rows()
    cell = fit()["cells"][0]
    fresh = {r["query_id"]: r["teacher"] + .25 for r in diagnostic}
    records = analysis.evaluate_model(cell, diagnostic, predictions=fresh)
    for record in records:
        assert record["rms"] == pytest.approx(.25)
        assert record["mse"] == pytest.approx(.25**2)
        assert record["maximum"] == pytest.approx(.25)
        assert record["mean_error"] == pytest.approx(.25)
        assert record["prediction_scope"] == "fresh_action"
    del fresh[diagnostic[0]["query_id"]]
    missing = analysis.evaluate_model(cell, diagnostic, predictions=fresh)[0]
    assert missing["status"] == "FAILED"
    assert "Missing authoritative prediction" in missing["error"]


def test_classical_prediction_scoring_needs_no_affine_design():
    _, _, diagnostic = fixture_rows()
    r = diagnostic[0]
    r.pop("design")
    scored = analysis.evaluate_prediction(r, r["teacher"] - .5, method="lawson_2")
    assert scored["status"] == "MEASURED"
    assert scored["rms"] == pytest.approx(.5)
    assert scored["mse"] == pytest.approx(.25)
    assert scored["mean_error"] == pytest.approx(-.5)
    assert scored["training_parent_count"] is None
    assert scored["training_membership"] == "not_applicable"


def test_parent_centered_error_and_interval_strata_use_physical_signed_means():
    _, _, diagnostic = fixture_rows()
    scored = []
    for index, r in enumerate(diagnostic):
        offset = 1. if r["parent_id"] == "d0" else -1.
        scored.append(analysis.evaluate_prediction(r, r["teacher"] + offset, method="ordered_2"))
    report = analysis.summarize_learning_curves(scored, expected_parent_ids={"diagnostic": ["d0", "d1"]})
    group = report["groups"][0]
    assert group["rms"] == pytest.approx(1.)
    assert group["mean_error"] == pytest.approx(0., abs=1e-15)
    assert group["centered_rms"] == pytest.approx(1.)
    assert set(group["strata"]["interval"]) == {"0:0.08"}
    assert report["metric_accounting"]["measured"] == 4


def test_semantically_duplicate_query_ids_are_rejected():
    training, validation, _ = fixture_rows()
    duplicate = deepcopy(training[0])
    duplicate["query_id"] += "-duplicate"
    with pytest.raises(ValueError, match="slot"):
        fit(training + [duplicate], validation)


def test_unrepresentable_squared_error_is_counted_invalid_instead_of_overflowing():
    report = analysis.summarize_learning_curves([metric("d0", "q0", 1e200)],
                                               expected_parent_ids={"diagnostic": ["d0"]})
    assert report["metric_accounting"]["invalid_measured"] == 1
    assert report["groups"][0]["rms"] is None


def test_entire_missing_expected_validation_parent_blocks_all_selections():
    training, validation, _ = fixture_rows()
    result = fit(training, validation[:2], expected_validation_parent_ids=["v0", "v1"])
    assert not result["all_selections_sealed"]
    assert all(c["status"] == "BLOCKED" for c in result["cells"])
    assert all("Incomplete expected validation parent coverage" in c["reasons"] for c in result["cells"])


def test_pending_evaluation_counts_as_missing_and_cannot_form_complete_parent():
    r = metric("d0", "q0", 1.)
    r["status"] = "NOT_EVALUATED"
    result = analysis.summarize_learning_curves([r], expected_parent_ids={"diagnostic": ["d0"]})
    assert result["parent_rows"][0]["missing_queries"] == 1
    assert not result["parent_rows"][0]["complete"]
    assert result["groups"][0]["rms"] is None
