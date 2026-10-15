"""Source-shared R7 projections, induced local ridge, and label sensitivity.

All solves use NumPy FP64 SVD independently of the deployable Torch fit.  These
five-coefficient oracles use calibration labels and are not deployed models.
No function in this module reads a withheld target or chooses a ridge value.
The uncertainty ball is a perturbation scenario, not a certified teacher bound.
"""
from __future__ import annotations

import math
import json

import numpy as np


def _array(value, name, *, ndim=None):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    result = np.asarray(value, dtype=np.float64)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite real values")
    return result


def _computed(value, name):
    if not np.isfinite(value).all():
        raise ArithmeticError(f"Nonfinite {name}; FP64 solve cannot be retained")
    return value


def _checked_diagnostics(value):
    try:
        json.dumps(value, allow_nan=False)
    except (ValueError, TypeError) as error:
        raise ArithmeticError("Nonfinite or nonportable FP64 diagnostics") from error
    return value


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and positive")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _weights(value, rows, name):
    q = np.ones(rows, dtype=np.float64) if value is None else _array(value, name, ndim=1)
    if q.shape != (rows,) or (q <= 0).any():
        raise ValueError(f"{name} requires one strictly positive weight per row")
    # This normalization is part of the declared objective.  Dividing by max
    # first avoids overflow in a sum of otherwise valid physical masses.
    q = q / q.max()
    q = q / q.sum()
    if (q <= 0).any():
        raise ArithmeticError("Positive physical weights underflowed; no implicit row deletion")
    return q


def _design(value, name, *, columns=None):
    A = _array(value, name, ndim=2)
    if min(A.shape) < 1 or (columns is not None and A.shape[1] != columns):
        expected = f" and {columns} columns" if columns is not None else ""
        raise ValueError(f"{name} must have nonempty rows{expected}")
    return A


def _calibration(design, target, row_weights):
    A = _design(design, "design", columns=5)
    b = _array(target, "target", ndim=1)
    if b.shape != (A.shape[0],):
        raise ValueError("target must have one value per calibration row")
    omega = _weights(row_weights, len(b), "row_weights")
    root = np.sqrt(omega)
    B = _computed(root[:, None] * A, "weighted design")
    d = _computed(root * b, "weighted target")
    return A, b, omega, B, d


def _spectrum(B, rank_rtol=None):
    if rank_rtol is None:
        rank_rtol = np.finfo(np.float64).eps * max(B.shape)
    if isinstance(rank_rtol, (bool, np.bool_)):
        raise ValueError("rank_rtol must be finite and between zero and one")
    rank_rtol = float(rank_rtol)
    if not math.isfinite(rank_rtol) or not 0 <= rank_rtol < 1:
        raise ValueError("rank_rtol must be finite and between zero and one")
    U, s, Vh = np.linalg.svd(B, full_matrices=False)
    threshold = rank_rtol * s[0]
    keep = s > threshold
    rank = int(keep.sum())
    return U, s, Vh, keep, dict(
        singular_values=s.tolist(), numerical_rank=rank,
        rank_rtol=rank_rtol, rank_threshold=float(threshold),
        rank_convention="retain singular values strictly greater than rank_rtol*s_max",
        rank_rtol_default="float64_epsilon * max(rows,columns)",
        full_column_rank=rank == B.shape[1],
        unidentified_coefficient_dimensions=B.shape[1] - rank,
        condition_number=(float(s[0] / s[-1]) if rank == B.shape[1] else None),
        retained_condition_number=(float(s[0] / s[keep][-1]) if rank else None),
    )


def _ridge_map(B, ridge):
    _, _, _, _, spectrum = _spectrum(B)
    # The direct data-SVD filter s/(s*s+lambda) loses accuracy in very small
    # columns next to O(100) commutators.  Solve the equivalent augmented SVD
    # with its *original* penalty transformed under column equilibration.
    # Its first n pseudoinverse columns map weighted data labels to c.  There
    # is no normal-equation inverse, TSVD, or change of ridge coordinates.
    scales = _computed(np.hypot(np.linalg.norm(B, axis=0), math.sqrt(ridge)),
                       "augmented column scales")
    penalty = math.sqrt(ridge) / scales
    if (penalty <= 0).any():
        raise ArithmeticError("Scaled ridge penalty underflowed; no implicit rank truncation")
    augmented = np.vstack((B / scales, np.diag(penalty)))
    U, singular, Vh = np.linalg.svd(augmented, full_matrices=False)
    if (singular <= 0).any():
        raise ArithmeticError("Augmented ridge system lost numerical rank")
    S = _computed(((Vh.T / singular) @ U[:len(B)].T) / scales[:, None],
                  "weighted-label sensitivity map")
    gain = np.linalg.svd(S, compute_uv=False)
    spectrum.update(augmented_column_scales=scales.tolist(),
                    augmented_singular_values=singular.tolist(),
                    augmented_condition_number=float(singular[0] / singular[-1]),
                    original_penalty_preserved=True)
    return S, gain, spectrum


def _residual_diagnostics(A, b, omega, B, d, c):
    residual = _computed(A @ c - b, "calibration residual")
    weighted = _computed(B @ c - d, "weighted calibration residual")
    normal = _computed(B.T @ weighted, "normal residual")
    return normal, dict(
        calibration_rms=float(np.linalg.norm(weighted)),
        calibration_maximum=float(np.max(np.abs(residual))),
        calibration_signed_mean=float(omega @ residual),
        calibration_centered_rms=float(np.sqrt(omega @ (residual - omega @ residual) ** 2)),
        normal_residual_l2=float(np.linalg.norm(normal)),
        coefficient_l2=float(np.linalg.norm(c)),
        data_loss=float(weighted @ weighted),
        zero_objective=float(d @ d),
    )


def fit_local_projection(design, target, *, row_weights=None, rank_rtol=None):
    """Project calibration remainder labels onto five source-shared branches.

    ``design`` is [physical rows,5]; ``target`` is U z - P z on exactly those
    calibration rows.  Weights are positive physical masses normalized to sum
    one, giving an aggregate physical RMS objective.  A numerical SVD cutoff
    is explicit: the answer is minimum-norm in the retained numerical subspace,
    not an exact rank statement or a lower bound on withheld-source error.
    """
    A, b, omega, B, d = _calibration(design, target, row_weights)
    U, s, Vh, keep, spectrum = _spectrum(B, rank_rtol)
    inverse = np.zeros_like(s)
    inverse[keep] = 1 / s[keep]
    S = _computed((Vh.T * inverse) @ U.T, "projection pseudoinverse")
    c = _computed(S @ d, "projection coefficients")
    normal, residual = _residual_diagnostics(A, b, omega, B, d, c)
    diagnostics = dict(
        schema="tdn.response_preconditioner.r7.local_projection.v1",
        status="SOLVED", solver="numpy_float64_svd_with_declared_numerical_subspace",
        rows=A.shape[0], coefficient_count=5, weighting="positive physical rows normalized to sum one",
        **spectrum, **residual,
        retained_normal_residual_l2=float(np.linalg.norm(Vh[keep] @ normal)),
        weighted_label_to_coefficient_gain=float(np.max(inverse)),
        truncated_singular_values=int((~keep).sum()),
        retained_subspace_only=not spectrum["full_column_rank"],
        minimum_norm_in_retained_subspace=True, ridge=0.0,
        objective=residual["data_loss"], offline_label_informed=True,
        certificate=False,
        scope="sampled calibration-source projection; not a full-operator or withheld-source bound",
    )
    return dict(coefficients=c, row_weights=omega,
                weighted_label_to_coefficients=S, diagnostics=_checked_diagnostics(diagnostics))


def fit_local_ridge(design, target, *, global_ridge, context_weight, features,
                    row_weights=None):
    """Fit the one-context induced penalty, relaxing global context coupling.

    The local loss is normalized to unit mass.  For context mass pi_q and
    feature vector eta, lambda_c = global_ridge/(pi_q*||eta||^2).  This follows
    from min_{W eta=c} ||W||_F^2 = ||c||^2/||eta||^2.  Independent local fits do
    not preserve the cross-context constraints or joint global objective.
    """
    global_ridge = _positive(global_ridge, "global_ridge")
    context_weight = _positive(context_weight, "context_weight")
    if context_weight > 1:
        raise ValueError("context_weight must be at most one in the normalized global objective")
    eta = _array(features, "features", ndim=1)
    if eta.size == 0:
        raise ValueError("features must be nonempty")
    norm2 = float(_computed(eta @ eta, "feature squared norm"))
    if norm2 <= 0:
        raise ValueError("features must have positive squared norm")
    ridge = global_ridge / context_weight / norm2
    if not math.isfinite(ridge) or ridge <= 0:
        raise ArithmeticError("Induced local ridge is not positive and finite")
    A, b, omega, B, d = _calibration(design, target, row_weights)
    S, gain, spectrum = _ridge_map(B, ridge)
    c = _computed(S @ d, "ridge coefficients")
    normal, residual = _residual_diagnostics(A, b, omega, B, d, c)
    gradient = _computed(2 * (normal + ridge * c), "ridge gradient")
    ridge_loss = float(_computed(ridge * (c @ c), "ridge loss"))
    gap = float(_computed((gradient @ gradient) / (4 * ridge), "gradient gap diagnostic"))
    diagnostics = dict(
        schema="tdn.response_preconditioner.r7.local_ridge.v1",
        status="SOLVED", solver="numpy_float64_augmented_equilibrated_svd_without_truncation",
        rows=A.shape[0], coefficient_count=5, weighting="positive physical rows normalized to sum one",
        **spectrum, **residual,
        global_ridge=global_ridge, context_weight=context_weight,
        feature_squared_norm=norm2, ridge=ridge,
        induced_penalty="global_ridge / (context_weight * feature_squared_norm)",
        ridge_loss=ridge_loss, objective=residual["data_loss"] + ridge_loss,
        gradient_l2=float(np.linalg.norm(gradient)),
        optimality_gap_diagnostic=gap, singular_value_truncation=False,
        weighted_label_to_coefficient_gain=float(np.max(gain)),
        objective_scope="one-context induced penalty; independent contexts relax global coupling",
        same_as_joint_global_objective=False, offline_label_informed=True,
        certificate=False,
    )
    minimum_norm_W = _computed(c[:, None] * eta[None, :] / norm2, "minimum norm W")
    return dict(coefficients=c, row_weights=omega,
                weighted_label_to_coefficients=S, minimum_norm_W=minimum_norm_W,
                diagnostics=_checked_diagnostics(diagnostics))


def ridge_sensitivity(design, test_design, *, ridge, row_weights=None,
                      test_row_weights=None, reference_uncertainty=None,
                      uncertainty_radius=None, independent_reference_difference=None,
                      seed=2026101507, random_trials=3):
    """Measure label-to-coefficient and label-to-withheld-RMS amplification.

    This accepts any coefficient count, so it also applies to the global W.
    Input labels are weighted d=sqrt(omega)*b.  ``reference_uncertainty`` is a
    nonnegative row-scale vector giving epsilon=||sqrt(omega)*u||_2; alternatively
    provide an explicit weighted-label ``uncertainty_radius``.  Neither proves
    a teacher error bound.  Test rows receive separately normalized physical
    masses, so ||sqrt(test_omega)*A_test*delta_c|| is aggregate physical RMS.

    The independent reference difference is an actual unweighted calibration
    label difference, retained at its measured magnitude even outside the ball.
    The leading and fixed-seed random directions use radius epsilon.  Perturbation
    arrays are numerical replay data; JSON-safe records are in ``diagnostics``.
    No test target is accepted, and no ridge or model selection happens here.
    """
    ridge = _positive(ridge, "ridge")
    A = _design(design, "design")
    At = _design(test_design, "test_design", columns=A.shape[1])
    omega = _weights(row_weights, len(A), "row_weights")
    test_omega = _weights(test_row_weights, len(At), "test_row_weights")
    if (reference_uncertainty is None) == (uncertainty_radius is None):
        raise ValueError("Supply exactly one of reference_uncertainty or uncertainty_radius")
    if reference_uncertainty is not None:
        uncertainty = _array(reference_uncertainty, "reference_uncertainty", ndim=1)
        if uncertainty.shape != (len(A),) or (uncertainty < 0).any():
            raise ValueError("reference_uncertainty requires one nonnegative scale per calibration row")
        epsilon = float(_computed(np.linalg.norm(np.sqrt(omega) * uncertainty),
                                  "reference-sized label radius"))
        radius_source = "l2 norm of sqrt(normalized physical mass)*reference_uncertainty rows"
    else:
        if isinstance(uncertainty_radius, (bool, np.bool_)):
            raise ValueError("uncertainty_radius must be finite and nonnegative")
        epsilon = float(uncertainty_radius)
        if not math.isfinite(epsilon) or epsilon < 0:
            raise ValueError("uncertainty_radius must be finite and nonnegative")
        radius_source = "explicit weighted calibration-label L2 radius"
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (isinstance(random_trials, (bool, np.bool_))
            or not isinstance(random_trials, (int, np.integer)) or not 0 <= random_trials <= 32):
        raise ValueError("random_trials must be an integer between zero and 32")
    B = _computed(A * np.sqrt(omega)[:, None], "weighted design")
    S, gain, spectrum = _ridge_map(B, ridge)
    normalized_test = _computed(At * np.sqrt(test_omega)[:, None], "normalized test design")
    prediction_map = _computed(normalized_test @ S, "normalized prediction sensitivity map")
    _, prediction_singular, prediction_Vh = np.linalg.svd(prediction_map, full_matrices=False)
    prediction_gain = float(prediction_singular[0])
    coefficient_gain = float(np.max(gain))
    theoretical_bound = 1 / (2 * math.sqrt(ridge))
    deltas = [epsilon * prediction_Vh[0]]
    names = ["leading_withheld_rms_direction"]
    generator = np.random.default_rng(int(seed))
    for index in range(random_trials):
        direction = generator.normal(size=len(A))
        direction /= np.linalg.norm(direction)
        deltas.append(epsilon * direction)
        names.append(f"fixed_seed_random_{index:02d}")
    if independent_reference_difference is not None:
        difference = _array(independent_reference_difference,
                            "independent_reference_difference", ndim=1)
        if difference.shape != (len(A),):
            raise ValueError("independent_reference_difference must have one calibration value per row")
        deltas.append(np.sqrt(omega) * difference)
        names.append("independent_reference_difference_unscaled")
    delta_d = _computed(np.stack(deltas), "weighted label perturbations")
    delta_c = _computed(delta_d @ S.T, "coefficient perturbations")
    delta_prediction = _computed(delta_c @ At.T, "prediction perturbations")
    delta_normalized_prediction = _computed(delta_prediction * np.sqrt(test_omega),
                                            "normalized prediction perturbations")
    scenarios = []
    for index, name in enumerate(names):
        size = float(np.linalg.norm(delta_d[index]))
        prediction_size = float(np.linalg.norm(delta_normalized_prediction[index]))
        scenarios.append(dict(
            name=name, weighted_label_l2=size,
            coefficient_l2=float(np.linalg.norm(delta_c[index])),
            withheld_prediction_rms=prediction_size,
            withheld_prediction_maximum=float(np.max(np.abs(delta_prediction[index]))),
            prediction_gain_observed=(prediction_size / size if size else None),
            within_scenario_ball=bool(size <= epsilon * (1 + 16 * np.finfo(float).eps)),
            rescaled_to_radius=name != "independent_reference_difference_unscaled",
        ))
    diagnostics = dict(
        schema="tdn.response_preconditioner.r7.ridge_sensitivity.v1",
        ridge=ridge, calibration_rows=len(A), withheld_rows=len(At),
        coefficient_count=A.shape[1], **spectrum,
        weighted_label_to_coefficient_gain=coefficient_gain,
        coefficient_gain_exact_arithmetic_upper_bound=theoretical_bound,
        coefficient_gain_bound_numerically_consistent=bool(
            coefficient_gain <= theoretical_bound * (1 + 16 * np.finfo(float).eps)),
        weighted_label_to_withheld_rms_gain=prediction_gain,
        uncertainty_radius=epsilon, uncertainty_radius_source=radius_source,
        worst_ball_coefficient_l2=coefficient_gain * epsilon,
        worst_ball_withheld_rms=prediction_gain * epsilon,
        uncertainty_set="aggregate weighted calibration-label L2 ball; not per-source error bounds",
        uncertainty_is_certified=False, seed=int(seed), random_trials=int(random_trials),
        output_norm="sqrt(sum normalized positive physical test-row mass * error^2)",
        input_norm="Euclidean norm after multiplying labels by sqrt(normalized calibration mass)",
        perturbation_scope="calibration labels only; withheld targets neither read nor fitted",
        certificate=False, scenarios=scenarios,
    )
    return dict(diagnostics=_checked_diagnostics(diagnostics), arrays=dict(
        row_weights=omega, test_row_weights=test_omega,
        weighted_label_to_coefficients=S,
        weighted_label_to_normalized_prediction=prediction_map,
        weighted_label_deltas=delta_d, coefficient_deltas=delta_c,
        prediction_deltas=delta_prediction,
        normalized_prediction_deltas=delta_normalized_prediction,
    ))
