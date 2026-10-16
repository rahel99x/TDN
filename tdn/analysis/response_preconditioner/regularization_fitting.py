"""Conditional FP64 generalized-ridge fits for a frozen retained design bank.

No data acquisition, validation selection or architecture lives here. Designs
have one physical scalar observation per row, exactly as in ``fitting.py``.
The penalty and prior are declared inputs, not inferred from diagnostic data.
All ranks, gradients and gaps are ordinary FP64 numerical diagnostics.
"""
from __future__ import annotations

import math

import torch


def _tensor(value, name):
    # Construct list inputs directly in FP64; never round through default FP32.
    original = torch.as_tensor(value)
    if original.is_complex():
        raise ValueError(f"{name} must be real")
    result = torch.as_tensor(value, dtype=torch.float64, device="cpu").detach()
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must be finite")
    return result


def _finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise ArithmeticError(f"Nonfinite {name}")
    return value


def _inputs(design, target, parent_ids, row_weights):
    A, b = _tensor(design, "design"), _tensor(target, "target")
    if A.ndim != 2 or min(A.shape) < 1 or b.shape != (A.shape[0],):
        raise ValueError("Expected nonempty design [scalar_rows, coefficients] and target [scalar_rows]")
    parents = list(parent_ids)
    if len(parents) != A.shape[0] or any(not isinstance(p, str) or not p for p in parents):
        raise ValueError("One nonempty parent ID is required per scalar row")
    q = torch.ones_like(b) if row_weights is None else _tensor(row_weights, "row_weights")
    if q.shape != b.shape or bool((q <= 0).any()):
        raise ValueError("row_weights must have shape [scalar_rows] and be strictly positive")
    unique = sorted(set(parents))
    omega = torch.empty_like(q)
    counts = {}
    for parent in unique:
        mask = torch.tensor([p == parent for p in parents], dtype=torch.bool)
        total = _finite(q[mask].sum(), "parent weight sum")
        omega[mask] = (q[mask] / total) / len(unique)
        counts[parent] = int(mask.sum())
    if not bool(torch.isfinite(omega).all()) or bool((omega <= 0).any()):
        raise ArithmeticError("Parent normalization lost a finite positive row weight")
    root = omega.sqrt()
    Aw = _finite(A * root[:, None], "weighted design")
    bw = _finite(b * root, "weighted target")
    return A, b, Aw, bw, counts


def _spectrum(matrix):
    U, singular, Vh = torch.linalg.svd(matrix, full_matrices=False)
    _finite(singular, "singular values")
    threshold = float(torch.finfo(torch.float64).eps * max(matrix.shape) * singular[0])
    retained = singular > threshold
    rank = int(retained.sum())
    full_column_rank = rank == matrix.shape[1]
    condition = float(singular[0] / singular[-1]) if full_column_rank else None
    if condition is not None and not math.isfinite(condition):
        condition = None
    return U, singular, Vh, retained, dict(singular_values=singular.tolist(), rank=rank,
        rank_threshold=threshold, condition_number=condition, full_column_rank=full_column_rank)


def _minimum_norm(U, singular, Vh, retained, rhs):
    # This truncation is permitted only in explicitly ineligible diagnostics.
    if not bool(retained.any()):
        return torch.zeros(Vh.shape[1], dtype=torch.float64)
    return _finite(Vh[retained].mT @ ((U[:, retained].mT @ rhs) / singular[retained]),
                   "diagnostic minimum-norm coefficients")


def _metadata(A, counts):
    return dict(rows=A.shape[0], coefficients=A.shape[1], parent_count=len(counts),
        parent_row_counts=counts, parent_total_weight=1./len(counts),
        weighting="positive physical scalar rows normalized to equal total mass per parent",
        precision="float64", device="cpu", certificate=False,
        diagnostic_arithmetic="ordinary_float64_not_outward_rounded",
        rank_convention="eps64 * max(matrix.shape) * largest singular value; fixed, no rank tuning")


def fit_parent_balanced_tikhonov(design, target, parent_ids, *, ridge,
                                penalty=None, prior=None, row_weights=None,
                                optimality_rtol=1e-8):
    r"""Fit ``||sqrt(W)(Aw-b)||² + ridge*||L(w-w0)||²``.

    Shapes: A=[scalar_rows,p], b=[scalar_rows], L=[penalty_rows,p], w0=[p].
    Defaults L=I and w0=0 reproduce ordinary ridge. Positive physical row
    weights are normalized to equal total mass per parent, identically to
    ``fit_parent_balanced_ridge``. The caller repeats each query's parent ID
    for every physical scalar row and supplies the appropriate cell masses.

    We scale columns of the complete original augmented system B=[sqrt(W)A;
    sqrt(ridge)L], including both its penalty and RHS=[sqrt(W)b;sqrt(ridge)Lw0].
    Eligible fits divide by every scaled singular value, without truncation.
    A numerically deficient augmented system yields an explicitly ineligible
    original-coordinate minimum-norm diagnostic and never a selected candidate.

    For an identifiable fit let mu=sigma_min(B)^2. The exact-arithmetic bound
    J(w)-J(w*) <= ||grad J||²/(4*mu) uses the complete Hessian's eigenvalue
    2*mu; using ridge alone would be incorrect for arbitrary L. Reported values
    are evaluated in FP64, and caller-side validation/prediction checks remain
    necessary. Numerical eligibility here is not scientific promotion.
    """
    if isinstance(ridge, bool) or not math.isfinite(float(ridge)) or float(ridge) <= 0:
        raise ValueError("ridge must be finite and strictly positive")
    if (isinstance(optimality_rtol, bool) or not math.isfinite(float(optimality_rtol))
            or float(optimality_rtol) <= 0):
        raise ValueError("optimality_rtol must be finite and strictly positive")
    lam = float(ridge)
    A, b, Aw, bw, counts = _inputs(design, target, parent_ids, row_weights)
    p = A.shape[1]
    L = torch.eye(p, dtype=torch.float64) if penalty is None else _tensor(penalty, "penalty")
    w0 = torch.zeros(p, dtype=torch.float64) if prior is None else _tensor(prior, "prior")
    if L.ndim != 2 or L.shape[0] < 1 or L.shape[1] != p:
        raise ValueError("penalty must have shape [positive penalty_rows, coefficients]")
    if w0.shape != (p,):
        raise ValueError("prior must have shape [coefficients]")
    root_lam = math.sqrt(lam)
    penalty_matrix = _finite(root_lam * L, "weighted penalty")
    penalty_rhs = _finite(penalty_matrix @ w0, "prior penalty right-hand side")
    B = torch.cat((Aw, penalty_matrix))
    rhs = torch.cat((bw, penalty_rhs))
    data_norms = _finite(torch.linalg.vector_norm(Aw, dim=0), "data column norms")
    penalty_norms = _finite(torch.linalg.vector_norm(penalty_matrix, dim=0), "penalty column norms")
    raw_scales = _finite(torch.hypot(data_norms, penalty_norms), "augmented column scales")
    # A genuinely zero objective column remains zero, with an explicit unit
    # coordinate scale; its null direction will make the fit ineligible.
    scales = torch.where(raw_scales == 0, torch.ones_like(raw_scales), raw_scales)
    Bs = _finite(B / scales, "column-scaled augmented system")
    Ud, sd, Vhd, kept_d, data_spec = _spectrum(Aw)
    U, singular, Vh, kept, original_spec = _spectrum(B)
    Us, ss, Vhs, kept_s, scaled_spec = _spectrum(Bs)
    _, _, _, _, penalty_spec = _spectrum(L)
    unique = original_spec["full_column_rank"] and scaled_spec["full_column_rank"]
    if unique:
        coeff = _finite((Vhs.mT @ ((Us.mT @ rhs) / ss)) / scales, "coefficients")
        solver = "column_scaled_augmented_svd_without_truncation"
    else:
        coeff = _minimum_norm(U, singular, Vh, kept, rhs)
        solver = "ineligible_original_coordinate_truncated_svd_diagnostic"
    data_residual = _finite(Aw @ coeff - bw, "data residual")
    penalty_residual = _finite(penalty_matrix @ (coeff-w0), "penalty residual")
    data_loss = _finite(torch.dot(data_residual, data_residual), "data loss")
    penalty_loss = _finite(torch.dot(penalty_residual, penalty_residual), "penalty loss")
    objective = _finite(data_loss+penalty_loss, "objective")
    zero_objective = _finite(torch.dot(bw, bw)+torch.dot(penalty_rhs, penalty_rhs), "zero objective")
    prior_residual = _finite(Aw @ w0-bw, "prior data residual")
    prior_objective = _finite(torch.dot(prior_residual, prior_residual), "prior objective")
    baseline = min(float(zero_objective), float(prior_objective))
    normal = _finite(Aw.mT @ data_residual + penalty_matrix.mT @ penalty_residual, "normal residual")
    gradient_norm = _finite(2*torch.linalg.vector_norm(normal), "gradient norm")
    mu = float(singular[-1].square()) if unique else None
    if mu is not None and (not math.isfinite(mu) or mu <= 0):
        mu = None
    hessian_min = 2*mu if mu is not None else None
    if hessian_min is not None and not math.isfinite(hessian_min):
        hessian_min = None
    curvature_finite = mu is not None and hessian_min is not None
    # Evaluate (||gradient||/(2*sigma_min(B)))² rather than squaring the
    # gradient first: the latter can overflow although the final gap is finite.
    gap = float(((gradient_norm/singular[-1])/2).square()) if curvature_finite else None
    if gap is not None and not math.isfinite(gap):
        gap = None
    tolerance = float(optimality_rtol)*baseline+1e-28
    optimality = bool(gap is not None and gap <= tolerance)
    nonregression = bool(float(objective) <= baseline+tolerance)
    reasons = []
    if not original_spec["full_column_rank"]:
        reasons.append("ORIGINAL_AUGMENTED_SYSTEM_NUMERICALLY_RANK_DEFICIENT")
    if not scaled_spec["full_column_rank"]:
        reasons.append("SCALED_AUGMENTED_SYSTEM_NUMERICALLY_RANK_DEFICIENT")
    if not curvature_finite:
        reasons.append("NO_FINITE_POSITIVE_NUMERICAL_CURVATURE")
    if not optimality:
        reasons.append("NUMERICAL_OPTIMALITY_GAP_NOT_VERIFIED")
    if not nonregression:
        reasons.append("ZERO_OR_PRIOR_BASELINE_NONREGRESSION_NOT_VERIFIED")
    # d(Aw*w)/d(bw) = Aw*(B^T B)^-1*Aw^T. For full rank B,
    # its trace is ||U_data||_F²; no nrows-by-nrows hat matrix is allocated.
    effective_df = float(U[:A.shape[0]].square().sum()) if unique else None
    diag = dict(schema="tdn.response_preconditioner.tikhonov.v1", **_metadata(A, counts),
        status="SOLVED" if unique else "NON_IDENTIFIABLE_DIAGNOSTIC", solver=solver,
        ridge=lam, penalty_rows=L.shape[0], penalty_rank=penalty_spec["rank"],
        penalty_nullity=p-penalty_spec["rank"], penalty_singular_values=penalty_spec["singular_values"],
        penalty_rank_threshold=penalty_spec["rank_threshold"],
        prior_coefficients=w0.tolist(), penalty_matrix=L.tolist(),
        original_penalty_preserved=True, prior_rhs_preserved=True,
        column_scales=scales.tolist(), training_column_norms=data_norms.tolist(),
        weighted_penalty_column_norms=penalty_norms.tolist(),
        column_scaling_scope="supplied_training_rows_and_declared_penalty_only",
        column_scaling_strategy="original_augmented_column_l2_norm",
        zero_objective_columns=int((raw_scales == 0).sum()),
        singular_value_truncation=not unique,
        original_design_singular_values=data_spec["singular_values"],
        original_design_numerical_rank=data_spec["rank"],
        original_design_rank_threshold=data_spec["rank_threshold"],
        original_design_condition_number=data_spec["condition_number"],
        original_augmented_singular_values=original_spec["singular_values"],
        original_augmented_numerical_rank=original_spec["rank"],
        original_augmented_rank_threshold=original_spec["rank_threshold"],
        original_augmented_condition_number=original_spec["condition_number"],
        augmented_singular_values=scaled_spec["singular_values"],
        augmented_numerical_rank=scaled_spec["rank"],
        augmented_rank_threshold=scaled_spec["rank_threshold"],
        augmented_condition_number=scaled_spec["condition_number"],
        generalized_effective_degrees_of_freedom=effective_df,
        effective_degrees_of_freedom_scope="trace of training weighted-prediction smoother for fixed L,w0; not physical dimension",
        data_loss=float(data_loss), penalty_loss=float(penalty_loss), ridge_loss=float(penalty_loss),
        objective=float(objective), zero_objective=float(zero_objective), prior_objective=float(prior_objective),
        baseline_objective=baseline, objective_improvement=baseline-float(objective),
        coefficient_l2=float(_finite(torch.linalg.vector_norm(coeff), "coefficient norm")),
        prior_displacement_l2=float(_finite(torch.linalg.vector_norm(coeff-w0), "prior displacement norm")),
        normal_residual_norm=float(torch.linalg.vector_norm(normal)), gradient_norm=float(gradient_norm),
        minimum_normal_eigenvalue=mu, minimum_hessian_eigenvalue=hessian_min,
        optimality_gap_upper_bound=gap, optimality_tolerance=tolerance,
        optimality_verified=optimality, baseline_nonregression_verified=nonregression,
        eligible=not reasons, reasons=reasons, diagnostic_only=not unique,
        eligibility_scope="numerical fixed-training-objective checks only; caller must evaluate validation and prediction stability",
        guarantee_scope="conditional declared fixed-feature generalized training objective only")
    return dict(coefficients=coeff, diagnostics=diag)


def fit_parent_balanced_projection(design, target, parent_ids, *, row_weights=None):
    """Unregularized original-coordinate minimum-norm SVD projection diagnostic.

    Numerical singular values below eps64*max(shape)*sigma_max are explicitly
    discarded. This is a descriptive lower-error control on the *training*
    design and is always excluded from deployment/validation selection.
    """
    A, b, Aw, bw, counts = _inputs(design, target, parent_ids, row_weights)
    U, singular, Vh, kept, spec = _spectrum(Aw)
    coeff = _minimum_norm(U, singular, Vh, kept, bw)
    residual = _finite(Aw @ coeff-bw, "projection residual")
    normal = _finite(Aw.mT @ residual, "projection normal residual")
    loss = _finite(torch.dot(residual, residual), "projection data loss")
    diag = dict(schema="tdn.response_preconditioner.projection.v1", **_metadata(A, counts),
        status="PROJECTION_DIAGNOSTIC", solver="original_coordinate_rank_revealing_svd",
        ridge=0., data_loss=float(loss), penalty_loss=0., objective=float(loss),
        zero_objective=float(_finite(torch.dot(bw, bw), "projection zero objective")),
        coefficient_l2=float(_finite(torch.linalg.vector_norm(coeff), "projection coefficient norm")),
        original_design_singular_values=spec["singular_values"],
        original_design_numerical_rank=spec["rank"], original_design_rank_threshold=spec["rank_threshold"],
        original_design_condition_number=spec["condition_number"],
        singular_value_truncation=int(kept.sum()) < A.shape[1],
        retained_singular_values=int(kept.sum()),
        gradient_norm=float(2*torch.linalg.vector_norm(normal)),
        normal_residual_norm=float(torch.linalg.vector_norm(normal)),
        generalized_effective_degrees_of_freedom=float(spec["rank"]),
        optimality_gap_upper_bound=None, optimality_verified=False,
        eligible=False, deployable=False, diagnostic_only=True,
        reasons=["UNREGULARIZED_PROJECTION_EXCLUDED_FROM_DEPLOYMENT"],
        guarantee_scope="rank-thresholded training-design projection only; no generalization or operator-capacity certificate")
    return dict(coefficients=coeff, diagnostics=diag)
