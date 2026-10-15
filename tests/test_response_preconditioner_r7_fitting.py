"""Independent linear-algebra checks for R7 source-shared diagnostic fits."""
import json

import numpy as np
import pytest

from tdn.analysis.response_preconditioner.r7_fitting import (
    fit_local_projection, fit_local_ridge, ridge_sensitivity,
)


def fixture():
    rng = np.random.default_rng(951)
    A = rng.normal(size=(21, 5))
    b = A @ np.array([.3, -.7, .02, 1.1, -.2]) + rng.normal(size=21) * .01
    test_A = rng.normal(size=(17, 5))
    weights = np.linspace(.2, 2.3, len(b))
    return A, b, test_A, weights


def test_projection_matches_independent_qr_and_physical_objective():
    A, b, _, weights = fixture()
    actual = fit_local_projection(A, b, row_weights=weights)
    omega = weights / weights.sum()
    B = np.sqrt(omega)[:, None] * A
    d = np.sqrt(omega) * b
    Q, R = np.linalg.qr(B)
    expected = np.linalg.solve(R, Q.T @ d)
    np.testing.assert_allclose(actual["coefficients"], expected, atol=3e-15, rtol=3e-14)
    diag = actual["diagnostics"]
    residual = A @ expected - b
    assert diag["data_loss"] == pytest.approx(omega @ residual**2, rel=3e-13)
    assert diag["calibration_rms"] == pytest.approx(np.sqrt(omega @ residual**2))
    assert diag["calibration_signed_mean"] == pytest.approx(omega @ residual)
    assert diag["normal_residual_l2"] < 3e-15
    assert diag["numerical_rank"] == 5 and not diag["retained_subspace_only"]
    assert not diag["certificate"] and diag["offline_label_informed"]
    json.dumps(diag, allow_nan=False)


def test_projection_declares_rank_deficiency_and_minimum_norm():
    A, b, _, _ = fixture()
    A[:, 3] = A[:, 0]
    A[:, 4] = 0
    result = fit_local_projection(A, b)
    expected = np.linalg.lstsq(A, b, rcond=np.finfo(float).eps * max(A.shape))[0]
    np.testing.assert_allclose(result["coefficients"], expected, atol=3e-15, rtol=3e-14)
    c = result["coefficients"]
    assert c[0] == pytest.approx(c[3], abs=2e-15)
    assert c[4] == 0
    diag = result["diagnostics"]
    assert diag["numerical_rank"] == 3 and diag["retained_subspace_only"]
    assert diag["condition_number"] is None
    assert diag["truncated_singular_values"] == 2


def test_projection_cutoff_reports_retained_not_original_optimum():
    A = np.diag([1., .1, 1e-3, 1e-5, 1e-9])
    b = np.ones(5)
    result = fit_local_projection(A, b, rank_rtol=1e-4)
    np.testing.assert_allclose(result["coefficients"], [1, 10, 1000, 0, 0])
    diag = result["diagnostics"]
    assert diag["numerical_rank"] == 3
    assert diag["normal_residual_l2"] > 1e-6
    assert diag["retained_normal_residual_l2"] < 2e-16
    assert diag["weighted_label_to_coefficient_gain"] == pytest.approx(1000 * np.sqrt(5))
    assert diag["rank_threshold"] == pytest.approx(1e-4 / np.sqrt(5))


@pytest.mark.parametrize("rows", [3, 9])
def test_zero_projection_and_zero_ridge_design_are_finite(rows):
    A = np.zeros((rows, 5))
    b = np.ones(rows)
    projection = fit_local_projection(A, b)
    ridge = fit_local_ridge(A, b, global_ridge=.1, context_weight=.5, features=[1., 2.])
    for fit in (projection, ridge):
        np.testing.assert_array_equal(fit["coefficients"], np.zeros(5))
        assert fit["diagnostics"]["numerical_rank"] == 0
        assert fit["diagnostics"]["calibration_rms"] == pytest.approx(1.)
        json.dumps(fit["diagnostics"], allow_nan=False)


def test_underdetermined_projection_keeps_minimum_norm_scope():
    A, b, _, _ = fixture()
    result = fit_local_projection(A[:3], b[:3])
    expected = np.linalg.lstsq(A[:3], b[:3], rcond=None)[0]
    np.testing.assert_allclose(result["coefficients"], expected, atol=2e-15, rtol=3e-14)
    assert result["diagnostics"]["numerical_rank"] == 3
    assert result["diagnostics"]["retained_subspace_only"]


@pytest.mark.parametrize("ridge", [1e-10, .03])
def test_induced_ridge_matches_original_augmented_problem(ridge):
    A, b, _, weights = fixture()
    A *= np.array([1e-3, 1, 1e2, 1e-8, 3])
    eta = np.array([1, .2, -.3, .8, .01, .9, -.1, .2])
    pi = 1 / 12
    result = fit_local_ridge(A, b, global_ridge=ridge, context_weight=pi,
                             features=eta, row_weights=weights)
    lam = ridge / (pi * (eta @ eta))
    omega = weights / weights.sum()
    B = np.sqrt(omega)[:, None] * A
    d = np.sqrt(omega) * b
    augmented = np.vstack((B, np.sqrt(lam) * np.eye(5)))
    rhs = np.concatenate((d, np.zeros(5)))
    Q, R = np.linalg.qr(augmented)
    expected = np.linalg.solve(R, Q.T @ rhs)
    np.testing.assert_allclose(result["coefficients"], expected, atol=2e-10, rtol=3e-10)
    diag = result["diagnostics"]
    assert diag["ridge"] == pytest.approx(lam)
    assert diag["objective"] == pytest.approx(np.linalg.norm(B @ expected - d)**2 + lam * (expected @ expected))
    assert diag["objective"] <= diag["zero_objective"]
    assert diag["optimality_gap_diagnostic"] < 1e-14
    assert not diag["singular_value_truncation"] and not diag["same_as_joint_global_objective"]
    json.dumps(diag, allow_nan=False)


def test_minimum_W_norm_identity_matches_40_variable_single_context_fit():
    A, b, _, weights = fixture()
    eta = np.array([1., .4, .2, .3, -.4, .7, .1, .2])
    ridge, pi = .03, .4
    result = fit_local_ridge(A, b, global_ridge=ridge, context_weight=pi,
                             features=eta, row_weights=weights)
    W = result["minimum_norm_W"]
    np.testing.assert_allclose(W @ eta, result["coefficients"], atol=1e-15)
    assert np.sum(W**2) == pytest.approx(np.sum(result["coefficients"]**2) / (eta @ eta))
    global_A = (A[:, :, None] * eta).reshape(len(A), 40)
    root = np.sqrt(pi * weights / weights.sum())
    augmented = np.vstack((root[:, None] * global_A, np.sqrt(ridge) * np.eye(40)))
    rhs = np.concatenate((root * b, np.zeros(40)))
    full_weights = np.linalg.lstsq(augmented, rhs, rcond=None)[0].reshape(5, 8)
    np.testing.assert_allclose(W, full_weights, rtol=3e-14, atol=2e-15)


def test_normalized_local_weights_invariant_to_complete_source_duplication():
    A, b, _, weights = fixture()
    kw = dict(global_ridge=.03, context_weight=1/12, features=[1, .4])
    actual = fit_local_ridge(A, b, row_weights=weights, **kw)
    duplicated = fit_local_ridge(np.tile(A, (2, 1)), np.tile(b, 2),
                                 row_weights=np.tile(weights, 2), **kw)
    np.testing.assert_allclose(actual["coefficients"], duplicated["coefficients"], atol=3e-15, rtol=3e-14)
    assert actual["diagnostics"]["objective"] == pytest.approx(duplicated["diagnostics"]["objective"])


def test_sensitivity_maps_predict_actual_refits_and_leading_ball_attains_bound():
    A, b, At, weights = fixture()
    lam, epsilon = .07, 2e-5
    test_weights = np.linspace(.2, 1.4, len(At))
    result = ridge_sensitivity(A, At, ridge=lam, row_weights=weights,
                               test_row_weights=test_weights, uncertainty_radius=epsilon,
                               independent_reference_difference=np.linspace(-1e-5, 3e-5, len(A)))
    diag, arrays = result["diagnostics"], result["arrays"]
    omega = weights / weights.sum()
    B = np.sqrt(omega)[:, None] * A
    # Direct normal-equation solve is an independent reference on this
    # deliberately well-conditioned fixture; production uses SVD only.
    inverse = np.linalg.solve(B.T @ B + lam * np.eye(5), B.T)
    np.testing.assert_allclose(arrays["weighted_label_to_coefficients"], inverse, rtol=3e-14, atol=3e-15)
    base = inverse @ (np.sqrt(omega) * b)
    for index, delta in enumerate(arrays["weighted_label_deltas"]):
        refit = inverse @ (np.sqrt(omega) * b + delta)
        np.testing.assert_allclose(refit - base, arrays["coefficient_deltas"][index], atol=7e-16)
        np.testing.assert_allclose(At @ (refit - base), arrays["prediction_deltas"][index], atol=2e-15)
    assert diag["scenarios"][0]["withheld_prediction_rms"] == pytest.approx(diag["worst_ball_withheld_rms"], rel=5e-15)
    assert diag["scenarios"][0]["weighted_label_l2"] == pytest.approx(epsilon)
    assert diag["coefficient_gain_bound_numerically_consistent"]
    assert diag["weighted_label_to_coefficient_gain"] <= 1 / (2 * np.sqrt(lam))
    assert not diag["uncertainty_is_certified"]
    json.dumps(diag, allow_nan=False)


def test_prediction_gain_includes_test_mass_normalization_and_duplication():
    A, _, At, weights = fixture()
    actual = ridge_sensitivity(A, At, ridge=.03, row_weights=weights, uncertainty_radius=.01)
    duplicated = ridge_sensitivity(A, np.tile(At, (3, 1)), ridge=.03,
                                    row_weights=weights, uncertainty_radius=.01)
    key = "weighted_label_to_withheld_rms_gain"
    assert actual["diagnostics"][key] == pytest.approx(duplicated["diagnostics"][key], rel=3e-14)
    S = actual["arrays"]["weighted_label_to_coefficients"]
    expected = np.linalg.norm(At @ S, ord=2) / np.sqrt(len(At))
    assert actual["diagnostics"][key] == pytest.approx(expected)


def test_reference_radius_random_stream_and_unscaled_independent_difference():
    A, _, At, weights = fixture()
    uncertainty = np.linspace(1e-7, 4e-7, len(A))
    difference = np.full(len(A), 3e-5)
    kw = dict(ridge=.03, row_weights=weights, reference_uncertainty=uncertainty,
              independent_reference_difference=difference, seed=321, random_trials=2)
    first = ridge_sensitivity(A, At, **kw)
    second = ridge_sensitivity(A, At, **kw)
    for key in first["arrays"]:
        np.testing.assert_array_equal(first["arrays"][key], second["arrays"][key])
    diag = first["diagnostics"]
    expected = np.sqrt((weights / weights.sum()) @ uncertainty**2)
    assert diag["uncertainty_radius"] == pytest.approx(expected)
    assert len(diag["scenarios"]) == 4
    reference_scenario = diag["scenarios"][-1]
    assert reference_scenario["weighted_label_l2"] == pytest.approx(3e-5)
    assert not reference_scenario["rescaled_to_radius"]
    assert not reference_scenario["within_scenario_ball"]
    np.testing.assert_allclose(first["arrays"]["weighted_label_deltas"][-1],
                               np.sqrt(weights / weights.sum()) * difference)
    other = ridge_sensitivity(A, At, **{**kw, "seed": 322})
    np.testing.assert_array_equal(first["arrays"]["weighted_label_deltas"][0], other["arrays"]["weighted_label_deltas"][0])
    assert not np.array_equal(first["arrays"]["weighted_label_deltas"][1], other["arrays"]["weighted_label_deltas"][1])


def test_zero_radius_zero_design_and_general_global_coefficient_count():
    result = ridge_sensitivity(np.zeros((3, 40)), np.zeros((7, 40)), ridge=.1,
                               uncertainty_radius=0, random_trials=0)
    diag = result["diagnostics"]
    assert diag["coefficient_count"] == 40
    assert diag["worst_ball_withheld_rms"] == 0
    assert diag["scenarios"][0]["prediction_gain_observed"] is None
    assert diag["scenarios"][0]["within_scenario_ball"]
    json.dumps(diag, allow_nan=False)


@pytest.mark.parametrize("change", [
    {"global_ridge": 0}, {"global_ridge": float("nan")}, {"context_weight": 0},
    {"context_weight": 1.1}, {"features": [0, 0]}, {"features": []},
    {"features": [1, float("inf")]}, {"row_weights": np.zeros(21)},
])
def test_invalid_local_ridge_contract_rejected(change):
    A, b, _, _ = fixture()
    kwargs = dict(global_ridge=.01, context_weight=1/12, features=[1, .3])
    with pytest.raises(ValueError):
        fit_local_ridge(A, b, **{**kwargs, **change})


@pytest.mark.parametrize("change", [
    {"ridge": 0}, {"uncertainty_radius": -1}, {"uncertainty_radius": float("nan")},
    {"reference_uncertainty": np.ones(21)}, {"random_trials": -1},
    {"random_trials": 1.5}, {"seed": True}, {"seed": -1},
    {"test_row_weights": np.zeros(17)}, {"independent_reference_difference": np.ones(20)},
])
def test_invalid_sensitivity_contract_rejected(change):
    A, _, At, _ = fixture()
    kwargs = dict(ridge=.01, uncertainty_radius=1e-6)
    with pytest.raises(ValueError):
        ridge_sensitivity(A, At, **{**kwargs, **change})


@pytest.mark.parametrize("bad", ["columns", "target", "nan", "rank", "weight"])
def test_invalid_projection_contract_rejected(bad):
    A, b, _, weights = fixture()
    kwargs = dict(row_weights=weights)
    if bad == "columns":
        A = A[:, :4]
    elif bad == "target":
        b = b[:, None]
    elif bad == "nan":
        A[0, 0] = np.nan
    elif bad == "rank":
        kwargs["rank_rtol"] = -1
    elif bad == "weight":
        weights[0] = 0
    with pytest.raises(ValueError):
        fit_local_projection(A, b, **kwargs)
