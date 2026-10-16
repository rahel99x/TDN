"""Thin-factor sensitivity of a fixed retained-data regularized response fit.

This module does not fit coefficients, acquire references, or choose a model.
It maps perturbations of *weighted training labels* to weighted validation
predictions. The retained empirical uncertainty radius defines a diagnostic
ball, not a certified error enclosure or a scientific eligibility criterion.
"""
from __future__ import annotations

import json
import math

import numpy as np


def _array(value, name, *, ndim=None, shape=None):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} has an incompatible shape")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _scalar(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a real scalar")
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return result


def _norm(value):
    """Euclidean norm without squaring tiny/large values before scaling."""
    scale = float(np.max(np.abs(value))) if value.size else 0.
    return 0. if scale == 0 else scale * float(np.linalg.norm(value / scale))


def sensitivity(Aw, bw, validation_design_weighted,
                training_reference_difference_weighted, training_uncertainty_radius,
                validation_error_rms, validation_remainder_rms, validation_uncertainty_rms,
                *, family, ridge):
    r"""Report prediction sensitivity without an n_validation-by-n_training map.

    ``Aw=sqrt(W_train) A_train`` and ``E=validation_design_weighted`` already
    include positive parent/query/cell weights summing to one on each bank.
    Therefore the Euclidean norm of a weighted prediction is physical RMS;
    this function must not divide it by another square root of row count.
    ``bw`` is optional (None), otherwise checked for compatible finite shape;
    the derivative is independent of labels. ``training_reference_difference``
    is weighted with the *same* training weights. The empirical radius is
    ``sqrt(sum_query parent_query_weight * uncertainty_rms_query**2)``.

    With a thin SVD ``Aw=U diag(sigma) V.T``, the derivative is ``E V D U.T``.
    Its spectral norm equals that of ``E V D``, with at most 40 columns:

    * coefficient_ridge: D_j = sigma_j / (sigma_j**2 + ridge), no truncation;
    * action_energy: D_j = 1 / ((1+ridge)*sigma_j), requires full column rank;
    * projection: D_j = 1/sigma_j above eps64*max(Aw.shape)*sigma_max, otherwise
      zero. This explicit TSVD convention is diagnostic, not model selection.

    ``ridge`` is lambda for coefficient ridge and dimensionless alpha for
    action-energy regularization; projection requires 0 or None. Numerical
    rank is reported for all families, but does not discard directions from
    positive coefficient ridge. A rank-deficient action-energy objective is
    reported unavailable rather than silently adding a minimum-norm prior.

    Returned ratios separate validation error, target remainder, and reference
    uncertainty. A zero denominator yields None. No sensitivity threshold is
    used to accept/reject a scientifically valid fit, and large coefficient
    magnitude is not an eligibility test.
    """
    design = _array(Aw, "Aw", ndim=2)
    if min(design.shape, default=0) < 1 or design.shape[1] > 40:
        raise ValueError("Aw must have positive dimensions and at most 40 columns")
    rows, columns = design.shape
    validation = _array(validation_design_weighted, "validation_design_weighted", ndim=2)
    if validation.shape[0] < 1 or validation.shape[1] != columns:
        raise ValueError("Validation design must have positive rows and matching columns")
    if bw is not None:
        _array(bw, "bw", shape=(rows,))
    difference = _array(training_reference_difference_weighted,
                        "training_reference_difference_weighted", shape=(rows,))
    radius = _scalar(training_uncertainty_radius, "training_uncertainty_radius")
    denominators = {
        "validation_error_rms": _scalar(validation_error_rms, "validation_error_rms"),
        "validation_remainder_rms": _scalar(validation_remainder_rms, "validation_remainder_rms"),
        "validation_uncertainty_rms": _scalar(validation_uncertainty_rms, "validation_uncertainty_rms"),
    }
    if family not in ("coefficient_ridge", "action_energy", "projection"):
        raise ValueError("Unknown sensitivity family")
    strength = (0. if ridge is None else _scalar(ridge, "ridge"))
    if (family == "projection" and strength != 0) or (family != "projection" and strength <= 0):
        raise ValueError("Positive penalty required except for zero-penalty projection")

    left, singular, right_t = np.linalg.svd(design, full_matrices=False)
    if not np.isfinite(singular).all():
        raise ArithmeticError("Nonfinite singular values in sensitivity diagnostic")
    threshold = float(np.finfo(np.float64).eps * max(design.shape) * singular[0])
    retained = singular > threshold
    rank = int(np.count_nonzero(retained))
    delta_norm = _norm(difference)
    if not math.isfinite(delta_norm):
        raise ArithmeticError("Nonfinite reference perturbation norm")
    result = dict(
        schema="tdn.response_preconditioner.regularization.sensitivity.v1",
        status="ESTIMATED_NOT_CERTIFIED", family=family, ridge=strength,
        training_matrix_shape=list(design.shape), validation_matrix_shape=list(validation.shape),
        training_design_numerical_rank=rank, training_design_rank_threshold=threshold,
        rank_convention="eps64 * max(Aw.shape) * largest singular value",
        singular_values=singular.tolist(),
        singular_value_policy=("all positive-ridge singular directions; no truncation"
                               if family == "coefficient_ridge" else
                               "explicit numerical-rank TSVD diagnostic" if family == "projection" else
                               "all directions; full column rank required"),
        training_uncertainty_radius=radius,
        actual_reference_difference_weighted_norm=delta_norm,
        actual_reference_difference_within_estimated_ball=bool(
            delta_norm <= radius + 64*np.finfo(float).eps*max(radius, delta_norm)),
        prediction_operator_gain=None, estimated_ball_prediction_rms=None,
        actual_reference_perturbation_prediction_rms=None,
        error_certificate=False, scientific_eligibility_gate=False,
        dense_prediction_map_materialized=False, operator_factor_columns=int(singular.size),
        scope="weighted validation prediction change, fixed design/penalty and estimated training-label ball",
        uncertainty_scope="empirical reference uncertainty; not a proven admissible-error enclosure",
        **denominators,
    )
    for prefix in ("estimated_ball_prediction", "actual_reference_perturbation_prediction"):
        for denominator in denominators:
            result[prefix+"_to_"+denominator] = None
    if family == "action_energy" and rank != columns:
        result.update(status="UNAVAILABLE_RANK_DEFICIENT",
                      reason="Action-energy penalty shares the data nullspace; no unique coefficient minimizer")
        return result

    if family == "coefficient_ridge":
        root = np.hypot(singular, math.sqrt(strength))
        factors = (singular/root)/root
    else:
        factors = np.zeros_like(singular)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            factors[retained] = (1./singular[retained])/(1.+strength)
    with np.errstate(over="ignore", invalid="ignore"):
        # Never multiply the two outer rectangular factors into a dense label map.
        thin = (validation @ right_t.T) * factors[None, :]
        actual_change = thin @ (left.T @ difference)
    if not np.isfinite(thin).all() or not np.isfinite(actual_change).all():
        raise ArithmeticError("Nonfinite sensitivity factors or prediction change")
    gain = float(np.linalg.svd(thin, compute_uv=False)[0])
    ball_rms = radius * gain
    actual_rms = _norm(actual_change)
    if not all(math.isfinite(v) for v in (gain, ball_rms, actual_rms, delta_norm)):
        raise ArithmeticError("Nonfinite sensitivity magnitude")
    result.update(prediction_operator_gain=gain, estimated_ball_prediction_rms=ball_rms,
                  actual_reference_perturbation_prediction_rms=actual_rms)
    for prefix, value in (("estimated_ball_prediction", ball_rms),
                          ("actual_reference_perturbation_prediction", actual_rms)):
        for denominator, magnitude in denominators.items():
            ratio = value/magnitude if magnitude else None
            if ratio is not None and not math.isfinite(ratio):
                raise ArithmeticError("Nonfinite sensitivity ratio")
            result[prefix+"_to_"+denominator] = ratio
    json.dumps(result, allow_nan=False)
    return result
