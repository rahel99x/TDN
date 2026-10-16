"""Independent small-map and analytic tests for retained-label sensitivity."""
import json

import numpy as np
import pytest

from tdn.analysis.response_preconditioner.regularization_sensitivity import sensitivity


def call(A=None, E=None, delta=None, *, family="coefficient_ridge", ridge=.2,
         radius=.03, error=.4, remainder=.7, uncertainty=.002, bw=None):
    A = np.diag([2., 1., .5]) if A is None else np.asarray(A)
    E = np.eye(A.shape[1]) if E is None else np.asarray(E)
    delta = np.linspace(.001, .003, A.shape[0]) if delta is None else np.asarray(delta)
    return sensitivity(A, bw, E, delta, radius, error, remainder, uncertainty,
                       family=family, ridge=ridge)


@pytest.mark.parametrize("family,ridge", [("coefficient_ridge", .2), ("action_energy", .3), ("projection", 0.)])
def test_thin_factor_matches_independent_tiny_dense_augmented_map(family, ridge):
    rng = np.random.default_rng(108)
    A = rng.normal(size=(7, 3))
    E = rng.normal(size=(5, 3))
    delta = rng.normal(size=7)*.002
    if family == "coefficient_ridge":
        augmented = np.vstack((A, np.sqrt(ridge)*np.eye(3)))
        rhs = np.vstack((np.eye(7), np.zeros((3, 7))))
        coefficient_map = np.linalg.lstsq(augmented, rhs, rcond=None)[0]
    else:
        coefficient_map = np.linalg.lstsq(A, np.eye(7), rcond=None)[0]/(1+ridge)
    dense = E @ coefficient_map
    result = call(A, E, delta, family=family, ridge=ridge)
    assert result["prediction_operator_gain"] == pytest.approx(np.linalg.norm(dense, 2), rel=2e-13)
    assert result["actual_reference_perturbation_prediction_rms"] == pytest.approx(np.linalg.norm(dense@delta), rel=2e-13)
    assert result["estimated_ball_prediction_rms"] == pytest.approx(.03*np.linalg.norm(dense, 2), rel=2e-13)
    assert result["operator_factor_columns"] == 3
    assert not result["dense_prediction_map_materialized"]
    assert result["status"] == "ESTIMATED_NOT_CERTIFIED"
    assert not result["error_certificate"] and not result["scientific_eligibility_gate"]
    json.dumps(result, allow_nan=False)


def test_known_diagonal_gain_and_weighted_norm_is_not_divided_by_row_count():
    # D=[2/5,1/2,0.5/1.25]; E=diag(1,3,1), hence gain=1.5.
    result = call(E=np.diag([1., 3., 1.]), delta=[0., .02, 0.], ridge=1.)
    assert result["prediction_operator_gain"] == pytest.approx(1.5)
    assert result["actual_reference_perturbation_prediction_rms"] == pytest.approx(.03)
    assert result["estimated_ball_prediction_rms"] == pytest.approx(.045)
    assert result["estimated_ball_prediction_to_validation_error_rms"] == pytest.approx(.045/.4)
    assert result["estimated_ball_prediction_to_validation_remainder_rms"] == pytest.approx(.045/.7)
    assert result["estimated_ball_prediction_to_validation_uncertainty_rms"] == pytest.approx(.045/.002)


def test_action_energy_has_exact_uniform_projection_shrinkage():
    projection = call(family="projection", ridge=0.)
    action = call(family="action_energy", ridge=9.)
    assert action["prediction_operator_gain"] == pytest.approx(projection["prediction_operator_gain"]/10)
    assert action["actual_reference_perturbation_prediction_rms"] == pytest.approx(
        projection["actual_reference_perturbation_prediction_rms"]/10)


def test_weak_positive_ridge_keeps_direction_below_numerical_rank_threshold():
    A = np.diag([1., 1e-16])
    result = call(A, np.eye(2), [0., 1e-18], ridge=1e-18, radius=1e-18)
    assert result["training_design_numerical_rank"] == 1
    assert result["prediction_operator_gain"] == pytest.approx(1e-16/(1e-32+1e-18))
    assert result["actual_reference_perturbation_prediction_rms"] > 0
    assert "no truncation" in result["singular_value_policy"]


def test_projection_uses_explicit_machine_epsilon_cutoff():
    result = call(np.diag([1., 1e-16]), np.eye(2), [0., 1e-18], family="projection", ridge=None)
    assert result["training_design_numerical_rank"] == 1
    assert result["prediction_operator_gain"] == 1.
    assert result["actual_reference_perturbation_prediction_rms"] == 0.


def test_rank_deficient_and_underdetermined_coefficient_ridge_remains_available():
    A = np.array([[1., 0., 0.], [2., 0., 0.]])
    result = call(A, np.eye(3), [.01, .02], ridge=.1)
    augmented = np.vstack((A, np.sqrt(.1)*np.eye(3)))
    rhs = np.vstack((np.eye(2), np.zeros((3, 2))))
    dense = np.linalg.lstsq(augmented, rhs, rcond=None)[0]
    assert result["training_design_numerical_rank"] == 1
    assert result["prediction_operator_gain"] == pytest.approx(np.linalg.norm(dense, 2))
    assert result["status"] == "ESTIMATED_NOT_CERTIFIED"


def test_rank_deficient_action_energy_does_not_invent_minimum_norm_prior():
    result = call(np.diag([1., 0.]), np.eye(2), [0., 0.], family="action_energy", ridge=.1)
    assert result["status"] == "UNAVAILABLE_RANK_DEFICIENT"
    assert result["prediction_operator_gain"] is None
    assert result["actual_reference_perturbation_prediction_rms"] is None
    assert not result["scientific_eligibility_gate"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("family,ridge", [("coefficient_ridge", 1e-18), ("projection", 0.)])
def test_zero_design_has_zero_prediction_gain(family, ridge):
    result = call(np.zeros((5, 3)), np.eye(3), np.zeros(5), family=family, ridge=ridge)
    assert result["training_design_numerical_rank"] == 0
    assert result["prediction_operator_gain"] == 0.
    assert result["estimated_ball_prediction_rms"] == 0.


def test_zero_radius_and_zero_denominators_are_explicit():
    result = call(delta=np.zeros(3), radius=0., error=0., remainder=0., uncertainty=0.)
    assert result["estimated_ball_prediction_rms"] == 0.
    assert result["actual_reference_perturbation_prediction_rms"] == 0.
    assert result["actual_reference_difference_within_estimated_ball"]
    assert result["estimated_ball_prediction_to_validation_error_rms"] is None
    assert result["estimated_ball_prediction_to_validation_remainder_rms"] is None
    assert result["estimated_ball_prediction_to_validation_uncertainty_rms"] is None
    json.dumps(result, allow_nan=False)


def test_actual_reference_difference_outside_estimated_ball_is_reported_not_hidden():
    result = call(delta=[.1, .2, .3], radius=0.)
    assert result["estimated_ball_prediction_rms"] == 0.
    assert not result["actual_reference_difference_within_estimated_ball"]
    assert result["actual_reference_perturbation_prediction_rms"] > 0
    assert result["status"] == "ESTIMATED_NOT_CERTIFIED"


def test_sensitivity_is_independent_of_training_labels():
    a = call(bw=[1., 2., 3.])
    b = call(bw=[100., -200., 300.])
    assert a == b


def test_actual_perturbation_is_bounded_by_gain_times_its_own_weighted_norm():
    rng = np.random.default_rng(44)
    for _ in range(4):
        A = rng.normal(size=(13, 4))
        E = rng.normal(size=(9, 4))
        delta = rng.normal(size=13)*.01
        result = call(A, E, delta, ridge=1e-18, radius=np.linalg.norm(delta))
        assert result["actual_reference_perturbation_prediction_rms"] <= result["estimated_ball_prediction_rms"]*(1+1e-13)


def test_only_thin_svd_factors_are_materialized(monkeypatch):
    original = np.linalg.svd
    observed = []

    def record(matrix, *args, **kwargs):
        observed.append(matrix.shape)
        assert matrix.shape[1] == 3
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", record)
    rng = np.random.default_rng(92)
    call(rng.normal(size=(201, 3)), rng.normal(size=(301, 3)), np.zeros(201))
    assert observed == [(201, 3), (301, 3)]


def test_nonfinite_computed_prediction_factor_fails_instead_of_writing_infinity():
    with pytest.raises(ArithmeticError, match="Nonfinite sensitivity"):
        call([[1e-160]], [[1e160]], [0.], ridge=1e-320)


@pytest.mark.parametrize("replacement", [np.nan, np.inf, -1., True])
def test_invalid_uncertainty_radius_rejected(replacement):
    with pytest.raises(ValueError):
        call(radius=replacement)


@pytest.mark.parametrize("family,ridge", [("coefficient_ridge", 0.), ("action_energy", -1.),
                                          ("projection", 1.), ("coefficient_ridge", np.nan),
                                          ("unknown", .1), ("coefficient_ridge", True)])
def test_invalid_family_or_strength_rejected(family, ridge):
    with pytest.raises(ValueError):
        call(family=family, ridge=ridge)


@pytest.mark.parametrize("kwargs", [
    {"A": np.zeros((3, 41))}, {"A": np.zeros((0, 3))}, {"A": [1., 2.], "E": np.eye(2)},
    {"A": np.array([[1., np.nan], [0., 1.]])}, {"A": np.eye(2, dtype=complex)},
    {"E": np.ones((4, 2))}, {"E": np.zeros((0, 3))}, {"E": np.array([[1., 2., np.inf]])},
    {"delta": [1., 2.]}, {"delta": [1., 2., np.nan]}, {"bw": [1., 2.]},
    {"error": -1.}, {"remainder": np.inf}, {"uncertainty": np.nan},
])
def test_invalid_shapes_numeric_arrays_and_denominators_rejected(kwargs):
    with pytest.raises(ValueError):
        call(**kwargs)
