"""Stable conditional fitting with the original positive-ridge objective.

There is no neural optimizer here. A fixed-feature model is affine in its
readout; fitting that readout is a strictly convex quadratic problem. The
reported gradient gap is an FP64 numerical diagnostic, not an interval proof.
All scaling is fit on the supplied training rows only. Callers must enforce
parent-disjoint validation/diagnostic splits and retain those identities.
"""
from __future__ import annotations

import math

import torch


def _finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    return value


def fit_parent_balanced_ridge(design, target, parent_ids, *, ridge,
                              row_weights=None, optimality_rtol=1e-8):
    r"""Minimize sum_i omega_i (A_i w-b_i)^2 + ridge*||w||^2.

    ``design`` has shape [rows,coefficients], ``target`` shape [rows], and each
    parent ID is a nonempty string. Positive physical row weights q_i (default
    one) become omega_i=q_i/(P*sum_{k in parent(i)} q_k), giving each of P
    parents equal total weight. For nodal normalized physical L2, use equal
    within-field cell weights; heterogeneous grids need their own cell masses.
    All supplied rows must belong to training parents; no split is inferred.

    Set scale_j=hypot(||sqrt(omega)*A[:,j]||, sqrt(ridge)), the norm of
    column j in the original augmented system. Thus a tiny or zero data
    column cannot make its scaled penalty larger than one. Write x=scale*w
    and solve the FP64 CPU augmented system
        [sqrt(omega)*A/scale; sqrt(ridge)*diag(1/scale)] x
            ~= [sqrt(omega)*b; 0].
    SVD divides by *every* positive augmented singular value. No TSVD cutoff,
    row deletion, or hidden regularization changes the original minimizer.
    The transformed penalty is essential: penalizing ||x|| instead would fit
    a different objective. Even zero design columns have unique optimum w=0.

    Return ``coefficients`` as an FP64 CPU tensor and JSON-safe ``diagnostics``.
    Conditional strict convexity gives loss(w)-loss(w*) <= ||grad||^2/(4*ridge)
    in exact arithmetic. Here objective/gradient and this gap are evaluated in
    ordinary FP64 and explicitly labelled numerical diagnostics.
    """
    if isinstance(ridge, bool) or not math.isfinite(float(ridge)) or float(ridge) <= 0:
        raise ValueError("ridge must be finite and strictly positive")
    if not math.isfinite(float(optimality_rtol)) or float(optimality_rtol) <= 0:
        raise ValueError("optimality_rtol must be finite and positive")
    lam = float(ridge)
    # Construct directly in FP64: list -> default-FP32 -> FP64 would irreversibly
    # round precise scalar data before the numerical solve even starts.
    A = torch.as_tensor(design, device="cpu", dtype=torch.float64).detach()
    b = torch.as_tensor(target, device="cpu", dtype=torch.float64).detach()
    if A.ndim != 2 or min(A.shape) < 1 or b.shape != (A.shape[0],):
        raise ValueError("Expected nonempty design [rows,coefficients] and target [rows]")
    _finite(A, "design"); _finite(b, "target")
    parents = list(parent_ids)
    if len(parents) != A.shape[0] or any(not isinstance(x, str) or not x for x in parents):
        raise ValueError("One nonempty string parent ID is required per row")
    q = (torch.ones_like(b) if row_weights is None else
         torch.as_tensor(row_weights, device="cpu", dtype=torch.float64).detach())
    if q.shape != b.shape:
        raise ValueError("row_weights must have shape [rows]")
    _finite(q, "row_weights")
    if bool((q <= 0).any()):
        raise ValueError("Every row weight must be positive; dropping rows is not implicit")
    unique = sorted(set(parents))
    omega = torch.empty_like(q)
    parent_counts = {}
    for parent in unique:
        mask = torch.tensor([p == parent for p in parents], dtype=torch.bool)
        total = _finite(q[mask].sum(), "parent row-weight sum")
        omega[mask] = (q[mask] / total) / len(unique)
        parent_counts[parent] = int(mask.sum())
    if not bool(torch.isfinite(omega).all()) or bool((omega <= 0).any()):
        raise ArithmeticError("Parent balancing lost a positive row weight; no implicit row deletion")
    root_weights = omega.sqrt()
    weighted_A = _finite(A * root_weights[:, None], "weighted design")
    weighted_b = _finite(b * root_weights, "weighted target")
    data_column_norms = _finite(torch.linalg.vector_norm(weighted_A, dim=0),
                                "training column norms")
    root_ridge = math.sqrt(lam)
    scales = _finite(torch.hypot(data_column_norms,
                                torch.full_like(data_column_norms, root_ridge)),
                     "augmented column scales")
    scaled_A = _finite(weighted_A / scales, "scaled design")
    penalty = _finite(root_ridge / scales, "scaled ridge penalty")
    if bool((penalty <= 0).any()):
        raise ArithmeticError("Scaled ridge penalty underflowed; original objective is not representable")
    augmented = torch.cat((scaled_A, torch.diag(penalty)), dim=0)
    rhs = torch.cat((weighted_b, torch.zeros(A.shape[1], dtype=torch.float64)))
    U, singular, Vh = torch.linalg.svd(augmented, full_matrices=False)
    _finite(singular, "augmented singular values")
    if bool((singular <= 0).any()):
        raise ArithmeticError("Augmented ridge system lost numerical rank; no silent TSVD")
    scaled_coefficients = Vh.mT @ ((U.mT @ rhs) / singular)
    coefficients = _finite(scaled_coefficients / scales, "ridge coefficients")
    residual = _finite(weighted_A @ coefficients - weighted_b, "fit residual")
    data_loss = _finite(torch.dot(residual, residual), "data loss")
    ridge_loss = _finite(lam * torch.dot(coefficients, coefficients), "ridge loss")
    objective = _finite(data_loss + ridge_loss, "objective")
    zero_objective = _finite(torch.dot(weighted_b, weighted_b), "zero objective")
    gradient = _finite(2 * (weighted_A.mT @ residual + lam * coefficients), "gradient")
    gradient_norm = _finite(torch.linalg.vector_norm(gradient), "gradient norm")
    gap = _finite(gradient_norm.square() / (4 * lam), "optimality gap diagnostic")
    tolerance = float(optimality_rtol) * float(zero_objective) + 1e-28
    train_singular = torch.linalg.svdvals(weighted_A)
    # This threshold is diagnostic only. No value was discarded in the solve.
    rank_threshold = torch.finfo(torch.float64).eps * max(A.shape) * train_singular[0]
    rank = int((train_singular > rank_threshold).sum())
    design_condition = (float(train_singular[0] / train_singular[-1])
                        if rank == A.shape[1] else None)
    if design_condition is not None and not math.isfinite(design_condition):
        design_condition = None
    augmented_condition = float(singular[0] / singular[-1])
    if not math.isfinite(augmented_condition):
        augmented_condition = None
    diagnostics = dict(
        schema="tdn.response_preconditioner.ridge.v1", status="SOLVED",
        rows=A.shape[0], coefficients=A.shape[1], parent_count=len(unique),
        parent_row_counts=parent_counts, parent_total_weight=1. / len(unique),
        weighting="positive physical rows normalized to equal total mass per parent",
        precision="float64", device="cpu", solver="augmented_svd_without_truncation",
        ridge=lam, column_scales=scales.tolist(),
        column_scaling_scope="supplied_training_rows_only",
        column_scaling_strategy="original_augmented_column_l2_norm",
        training_column_norms=data_column_norms.tolist(),
        zero_training_columns=int((data_column_norms == 0).sum()),
        minimum_column_scale=float(scales.min()), maximum_column_scale=float(scales.max()),
        minimum_scaled_ridge_penalty=float(penalty.min()),
        maximum_scaled_ridge_penalty=float(penalty.max()),
        original_penalty_preserved=True, singular_value_truncation=False,
        augmented_singular_values=singular.tolist(),
        augmented_condition_number=augmented_condition,
        original_design_singular_values=train_singular.tolist(),
        original_design_numerical_rank=rank, original_design_condition_number=design_condition,
        data_loss=float(data_loss), ridge_loss=float(ridge_loss), objective=float(objective),
        zero_objective=float(zero_objective), objective_improvement=float(zero_objective - objective),
        gradient_norm=float(gradient_norm), optimality_gap_upper_bound=float(gap),
        optimality_tolerance=tolerance,
        optimality_verified=bool(float(gap) <= tolerance),
        baseline_nonregression_verified=bool(float(objective) <= float(zero_objective) + tolerance),
        certificate=False, diagnostic_arithmetic="ordinary_float64_not_outward_rounded",
        guarantee_scope="conditional fixed-feature regularized training objective only",
    )
    return dict(coefficients=coefficients, diagnostics=diagnostics)
