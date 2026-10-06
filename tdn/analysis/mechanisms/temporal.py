"""Training-free temporal representation audits for logistic reaction--diffusion.

These are small deterministic, one-dimensional discrete Fourier calculations.
The least-squares coefficients are parent-specific oracle diagnostics, never
trained-model accuracy. The perturbation audit concerns the exact coefficient
of epsilon**2, not the error of a finite-amplitude PDE prediction.
"""
from __future__ import annotations

from functools import lru_cache
import math
from typing import Callable

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp
from scipy.linalg import expm


FIT_HORIZONS = (.0125, .025, .05, .1, .2, .3)
VALIDATION_HORIZONS = (.0375, .15)
TEST_HORIZONS = (.01875, .075, .24, .4)
VARIANTS = ("scalar", "transported", "output_phi", "reaction_pair")
GRID = 64
AMPLITUDES = (.02, .01, .005)


def plan(config: dict) -> dict:
    """Freeze all cases and partitions before inspecting numerical results."""
    smoke = bool(config.get("smoke", False))
    regimes = ((.002, 1.), (.02, 6.)) if smoke else ((.0005, 1.), (.002, 6.), (.02, 1.), (.02, 6.))
    pairs = ((1, 3), (2, 2), (1, -1), (3, -1)) if smoke else ((1, 3), (2, 2), (1, 1), (3, 5), (1, -1), (3, -1))
    ids = []
    for order in (3, 4):
        ids.extend(f"phi-{order}-{dtype}" for dtype in ("float64", "float32"))
        ids.extend(f"forcing-{order}-{decay:g}" for decay in (0., 3., 100.))
    ids.append("phi-long-horizon")
    for ri, _ in enumerate(regimes):
        for p, q in pairs:
            prefix = f"r{ri}-p{p}-q{q}"
            ids.extend((f"{prefix}-reference", f"{prefix}-linear"))
            ids.extend(f"{prefix}-{variant}" for variant in VARIANTS)
        ids.append(f"collision-r{ri}")
    ids.extend(f"finite-amplitude-p{p}-q{q}" for p, q in pairs)
    return {"expected_case_ids": ids, "grid": GRID, "dimension": 1,
            "fit_horizons": list(FIT_HORIZONS), "validation_horizons": list(VALIDATION_HORIZONS),
            "test_horizons": list(TEST_HORIZONS), "regimes": [{"kappa": k, "rate": r} for k, r in regimes],
            "pairs": [list(pair) for pair in pairs], "coefficient_count": 2, "variants": list(VARIANTS),
            "base_state": .3, "fit_objective": "least squares of defect/h^3, normalized columns; no label-dependent model selection",
            "phi_quadrature_orders": [192, 384], "interaction_quadrature_orders": [96, 192],
            "reference": "refined Gaussian quadrature and independent DOP853 coefficient ODE",
            "finite_amplitude_check": {"epsilons": list(AMPLITUDES), "kappa": .002, "rate": 2., "horizon": .2,
                                       "expected_even_residual_order": 4, "accepted_order_interval": [3.7, 4.3]},
            "assumptions": ["Periodic one-dimensional discrete Laplacian on a unit interval.",
                            "Exact first- and second-variation coefficients about a homogeneous logistic solution.",
                            "Transported DC columns are identical: two nominal coefficients have effective rank one; retain this expected limitation and its prediction errors.",
                            "Pair/background information enriches reaction_pair beyond the output-frequency-only models."]}


def phi_negative(x, order: int, dtype=np.float64):
    """Stable phi_order(-x) for nonnegative x, preserving requested FP32/64.

    phi_j(z) = sum_{n>=0} z**n/(n+j)!. A Horner series avoids
    cancellation near zero; recurrence in reciprocal x avoids large powers.
    This is a correctness implementation, not a GPU/performance claim.
    """
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or order not in (1, 2, 3, 4):
        raise ValueError("phi order must be 1, 2, 3 or 4")
    if np.dtype(dtype) not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("phi dtype must be float32 or float64")
    x = np.asarray(x, dtype=dtype)
    if not np.all(np.isfinite(x)) or np.any(x < 0):
        raise ValueError("phi expects finite nonnegative x")
    small = np.minimum(x, dtype(2.0))
    series = np.full_like(x, 1.0 / math.factorial(40 + order))
    for n in range(39, -1, -1):
        series = -small * series + dtype(1.0 / math.factorial(n + order))
    safe = np.maximum(x, dtype(2.0))
    inverse = dtype(1.0) / safe
    recurrence = -np.expm1(-safe) * inverse
    for j in range(2, order + 1):
        recurrence = (dtype(1.0 / math.factorial(j - 1)) - recurrence) * inverse
    return np.where(x <= dtype(2.0), series, recurrence)


@lru_cache(maxsize=8)
def _quadrature(order: int):
    nodes, weights = leggauss(order)
    return .5 * (nodes + 1), .5 * weights


def _reaction(b: float, r: float, h, dtype=np.float64):
    h = np.asarray(h, dtype=dtype)
    b = np.asarray(b, dtype=dtype)
    decay = np.exp(-dtype(r) * h)
    denominator = b + (dtype(1.) - b) * decay
    value = b / denominator
    derivative = decay / denominator**2
    quadratic = -decay * (-np.expm1(-dtype(r) * h)) / denominator**3
    return value, derivative, quadratic


def eigenvalue(mode: int, grid: int = GRID) -> float:
    """Positive eigenvalue of minus periodic second-difference Laplacian."""
    return float(4 * grid**2 * np.sin(np.pi * mode / grid)**2)


def interaction_coefficients(h, *, p: int, q: int, kappa: float,
                             rate: float, base: float = .3,
                             quadrature_order: int = 96):
    """Exact and Strang normalized quadratic coefficients at mode p+q.

    Signed modes include p+q=0 and difference modes. For distinct absolute
    modes, the mixed contribution of cos(p x)+cos(q x) to mode p+q has
    multiplicity one. This mixed contribution may overlap a self harmonic;
    four-sign polarization isolates it. For |p|==|q|, use one cosine:
    its square has coefficient one half at both DC and mode 2|p|. Returned
    coefficients divide out that multiplicity, permitting comparison.
    All factors use the discrete Laplacian, not continuum eigenvalues.
    """
    h = np.atleast_1d(np.asarray(h, dtype=np.float64))
    theta, weights = _quadrature(quadrature_order)
    time = h[:, None] * theta
    _, jh, _ = _reaction(base, rate, h)
    _, js, _ = _reaction(base, rate, time)
    lm, lpq = eigenvalue(p + q), eigenvalue(p) + eigenvalue(q)
    # Factored this way, all diffusion exponentials decay; no growing
    # exp[-kappa*(lp+lq-lm)*s] is formed before cancellation.
    kernel = (jh[:, None] * js * np.exp(-kappa * lm * (h[:, None] - time))
              * np.exp(-kappa * lpq * time))
    exact = -rate * h * (kernel @ weights)
    middle, j1, k1 = _reaction(base, rate, h / 2)
    _, j2, k2 = _reaction(middle, rate, h / 2)
    split = j2 * k1 * np.exp(-kappa * lm * h) + k2 * j1**2 * np.exp(-kappa * lpq * h)
    return exact, split


def interaction_ode(h: float, *, p: int, q: int, kappa: float,
                    rate: float, base: float = .3) -> float:
    """Independent coupled perturbation ODE reference (normalized coefficient)."""
    lp, lq, lm = (eigenvalue(k) for k in (p, q, p + q))

    def rhs(_time, state):
        b, vp, vq, w = state
        sensitivity = rate * (1 - 2 * b)
        return (rate * b * (1 - b), (sensitivity - kappa * lp) * vp,
                (sensitivity - kappa * lq) * vq,
                (sensitivity - kappa * lm) * w - rate * vp * vq)

    solution = solve_ivp(rhs, (0., h), (base, 1., 1., 0.),
                         method="DOP853", rtol=2e-12, atol=2e-14)
    if not solution.success:
        raise RuntimeError(f"Interaction reference failed: {solution.message}")
    return float(solution.y[3, -1])


def finite_amplitude_scaling(p: int, q: int, check_budget: Callable[[], None]) -> dict:
    """Verify analytic coefficients against the full nonlinear discrete PDE.

    Even/polarized residuals after removing the quadratic term must scale
    as epsilon**4. Polarization matters when a difference mode coincides
    with a self harmonic: (3,-1) generates mode2, also generated by 1+1.
    """
    kappa, rate, horizon, base = .002, 2., .2, .3
    x = np.arange(GRID) / GRID
    first, second = (np.cos(2 * np.pi * mode * x) for mode in (p, q))
    mode = p + q
    projection = np.cos(2 * np.pi * mode * x)
    repeated = abs(p) == abs(q)
    modes = np.fft.rfftfreq(GRID) * GRID
    diffusion_multiplier = np.exp(-4 * kappa * GRID**2 * np.sin(np.pi * modes / GRID)**2 * horizon)

    def reaction(u, time):
        return u / (u + (1 - u) * np.exp(-rate * time))

    def rhs(_time, u):
        return kappa * GRID**2 * (np.roll(u, 1) + np.roll(u, -1) - 2 * u) + rate * u * (1 - u)

    exact, split = interaction_coefficients([horizon], p=p, q=q, kappa=kappa, rate=rate, base=base)
    references = (float(exact[0]), float(split[0]))
    residuals = [[], []]
    metrics = {"coupled_coefficient": references[0], "split_coefficient": references[1],
               "output_mode": mode, "polarized_mixed_coefficient": not repeated,
               "normalized_multiplicity": .5 if repeated else 1.}
    for ei, epsilon in enumerate(AMPLITUDES):
        # Repeated absolute modes mean a SINGLE cosine, including p=-q.
        # Distinct modes use four signs to eliminate self-harmonic terms.
        signs = ((-1, 0), (1, 0)) if repeated else ((-1, -1), (-1, 1), (1, -1), (1, 1))
        accumulators = [np.zeros(GRID), np.zeros(GRID)]
        for sp, sq in signs:
            check_budget()
            initial = base + epsilon * (sp * first + sq * second)
            solution = solve_ivp(rhs, (0., horizon), initial, method="DOP853", rtol=3e-13, atol=2e-14)
            if not solution.success:
                raise RuntimeError(f"Finite-amplitude PDE reference failed: {solution.message}")
            half_reacted = reaction(initial, horizon / 2)
            diffused = np.fft.irfft(np.fft.rfft(half_reacted) * diffusion_multiplier, n=GRID)
            split_solution = reaction(diffused, horizon / 2)
            weight = 1 if repeated else sp * sq
            accumulators[0] += weight * solution.y[:, -1]
            accumulators[1] += weight * split_solution
        if repeated:
            # Two signed solutions contain twice the homogeneous background.
            accumulators = [value - 2 * reaction(base, horizon) for value in accumulators]
            denominator = epsilon**2 * .5 * (2 if mode == 0 else 1)
        else:
            # The polarized sum is 4 eps^2 times the mixed cosine coefficient;
            # nonzero-cosine projection has mean square 1/2.
            denominator = 2 * epsilon**2
        for label, accumulator, reference, residual in zip(("coupled", "split"), accumulators, references, residuals):
            coefficient = float(np.mean(accumulator * projection) / denominator)
            value = abs(coefficient - reference) * epsilon**2
            residual.append(value)
            metrics[f"{label}_epsilon_{ei}_coefficient"] = coefficient
            metrics[f"{label}_epsilon_{ei}_even_residual"] = value
    for label, values in zip(("coupled", "split"), residuals):
        for i in range(len(values) - 1):
            metrics[f"{label}_residual_order_{i}"] = float(np.log(values[i] / values[i + 1]) / np.log(AMPLITUDES[i] / AMPLITUDES[i + 1])) if min(values[i:i + 2]) > 0 else None
    return metrics


def basis(h, variant: str, *, p: int, q: int, kappa: float,
          rate: float, base: float = .3, dtype=np.float64):
    """Two fixed coefficient columns; rates and choices do not use labels."""
    h = np.asarray(h, dtype=dtype)
    lm = eigenvalue(p + q)
    z = -np.expm1(-dtype(rate) * h)
    if variant == "scalar":
        return np.stack((z**3, z**4), axis=-1)
    if variant == "transported":
        return np.stack((z**3 * np.exp(-dtype(kappa * lm) * h), z**3), axis=-1)
    if variant == "output_phi":
        return np.stack((6 * h**3 * phi_negative(dtype(kappa * lm) * h, 3, dtype),
                         24 * h**4 * phi_negative(dtype(kappa * lm) * h, 4, dtype)), axis=-1)
    if variant == "reaction_pair":
        # This enriched diagnostic knows the generating pair and background.
        # It is not an h-independent local encoder or an implemented solver.
        theta, weights = _quadrature(96)
        theta, weights = theta.astype(dtype), weights.astype(dtype)
        time = h[:, None] * theta
        _, jh, _ = _reaction(base, rate, h, dtype)
        _, js, _ = _reaction(base, rate, time, dtype)
        kernel = (jh[:, None] * js
                  * np.exp(-dtype(kappa * lm) * (h[:, None] - time))
                  * np.exp(-dtype(kappa * (eigenvalue(p) + eigenvalue(q))) * time))
        return np.stack((3 * h**3 * (kernel @ (weights * theta**2)),
                         4 * h**4 * (kernel @ (weights * theta**3))), axis=-1).astype(dtype)
    raise ValueError(f"Unknown temporal basis: {variant}")


def fit_coefficients(fit_h, fit_targets, variant: str, **parameters):
    """Fit exactly two coefficients using fitting labels only.

    The objective is squared error in defect/h**3. Column normalization
    improves numerical conditioning without changing the least-squares span.
    Validation and test arrays deliberately are not accepted by this API.
    """
    h = np.asarray(fit_h, dtype=np.float64)
    target = np.asarray(fit_targets, dtype=np.float64)
    if h.ndim != 1 or len(h) < 3 or target.shape != h.shape:
        raise ValueError("Fit needs at least three aligned scalar observations")
    if (not np.isfinite(h).all() or not np.isfinite(target).all()
            or np.any(h <= 0) or len(np.unique(h)) != len(h)):
        raise ValueError("Fit horizons and targets must be finite, positive and unique")
    matrix = basis(h, variant, **parameters) / h[:, None]**3
    scales = np.linalg.norm(matrix, axis=0)
    scales = np.maximum(scales, np.finfo(np.float64).tiny)
    scaled = matrix / scales
    normalized, _, rank, singular = np.linalg.lstsq(scaled, target / h**3, rcond=1e-12)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else None
    return normalized / scales, {"rank": int(rank), "condition_number": condition,
                                  "coefficient_l2": float(np.linalg.norm(normalized / scales))}


def _row(case_id, test, variant, kind, outcome, metrics, inputs=None, note=""):
    return {"case_id": case_id, "mechanism": "temporal_response", "test": test,
            "variant": variant, "kind": kind, "outcome": outcome,
            "metrics": metrics, "inputs": inputs or {}, "note": note}


def run(config: dict, check_budget: Callable[[], None]) -> dict:
    """Execute a bounded deterministic panel; caller owns files and deadline."""
    rows = []
    frozen_plan = plan(config)
    x = np.array([0., 1e-12, 1e-6, .1, 1., 2., 2.01, 10., 100.])
    theta, weights = _quadrature(192)
    for order in (3, 4):
        check_budget()
        reference = np.exp(-x[:, None] * (1 - theta)) @ (weights * theta**(order - 1)) / math.factorial(order - 1)
        theta2, weights2 = _quadrature(384)
        refined = np.exp(-x[:, None] * (1 - theta2)) @ (weights2 * theta2**(order - 1)) / math.factorial(order - 1)
        uncertainty = float(np.max(np.abs(reference - refined)))
        for dtype in (np.float64, np.float32):
            measured = phi_negative(x, order, dtype).astype(np.float64)
            error = float(np.max(np.abs(measured - refined) / refined))
            limit = 3e-12 if dtype == np.float64 else 4e-6
            rows.append(_row(f"phi-{order}-{np.dtype(dtype).name}", "phi_quadrature", np.dtype(dtype).name,
                             "correctness", "PASS" if error <= limit and uncertainty < 1e-13 else "FAIL",
                             {"maximum_relative_error": error, "quadrature_difference": uncertainty,
                              "relative_error_limit": limit, "zero_value": float(measured[0])},
                             {"order": order, "x": x.tolist(), "quadrature_orders": [192, 384]}))
        # y'=-lambda*y+t**(j-1)/(j-1)!: augmented linear system has exact
        # matrix-exponential solution h**j*phi_j(-lambda*h).
        for decay, horizon in ((0., .3), (3., .7), (100., .4)):
            check_budget()
            matrix = np.zeros((order + 1, order + 1))
            matrix[0, 0], matrix[0, order] = -decay, 1.
            for j in range(2, order + 1):
                matrix[j, j - 1] = 1.
            start = np.zeros(order + 1)
            start[1] = 1.
            exact = float((expm(horizon * matrix) @ start)[0])
            prediction = float(horizon**order * phi_negative(decay * horizon, order))
            error = abs(prediction - exact)
            rows.append(_row(f"forcing-{order}-{decay:g}", "manufactured_polynomial_forcing", "matrix_exponential",
                             "correctness", "PASS" if error < 3e-13 else "FAIL",
                             {"absolute_error": error, "analytic_response": prediction, "reference": exact},
                             {"order": order, "decay": decay, "horizon": horizon}))

    # h**3 phi3(-lambda*h) grows as h**2 for lambda>0; no equilibrium claim.
    values = [float(h**3 * phi_negative(h, 3)) for h in (10., 100., 1000.)]
    rows.append(_row("phi-long-horizon", "long_horizon_growth", "output_phi", "negative_control",
                     "EXPECTED_LIMITATION", {"response_h10": values[0], "response_h100": values[1],
                                             "response_h1000": values[2]},
                     {"decay": 1.}, "Unbounded raw response: a phi-filtered cubic correction does not itself preserve long-time equilibrium or bounds."))

    regimes = [(regime["kappa"], regime["rate"]) for regime in frozen_plan["regimes"]]
    pairs = frozen_plan["pairs"]
    all_h = np.array(FIT_HORIZONS + VALIDATION_HORIZONS + TEST_HORIZONS)
    fit_count, val_count = len(FIT_HORIZONS), len(VALIDATION_HORIZONS)
    for ri, (kappa, rate) in enumerate(regimes):
        for p, q in pairs:
            check_budget()
            parameters = {"p": p, "q": q, "kappa": kappa, "rate": rate, "base": .3}
            prefix = f"r{ri}-p{p}-q{q}"
            exact, split = interaction_coefficients(all_h, **parameters, quadrature_order=96)
            refined, _ = interaction_coefficients(all_h, **parameters, quadrature_order=192)
            quad_difference = float(np.max(np.abs(exact - refined)))
            ode_error = abs(interaction_ode(.4, **parameters) - float(refined[-1]))
            accepted = quad_difference < 2e-12 and ode_error < 3e-11
            rows.append(_row(f"{prefix}-reference", "quadratic_interaction_reference", "quadrature_vs_ode", "correctness",
                             "PASS" if accepted else "FAIL",
                             {"quadrature_difference": quad_difference, "ode_absolute_difference": ode_error,
                              "maximum_quadratic_defect": float(np.max(np.abs(refined - split)))},
                             parameters, "Normalized epsilon-squared Fourier coefficient, not a finite-amplitude solution error."))
            middle, j1, _ = _reaction(.3, rate, all_h / 2)
            _, j2, _ = _reaction(middle, rate, all_h / 2)
            _, jh, _ = _reaction(.3, rate, all_h)
            linear_error = float(np.max(np.abs((j1 * j2 - jh) * np.exp(-kappa * eigenvalue(p) * all_h))))
            rows.append(_row(f"{prefix}-linear", "homogeneous_first_variation", "strang", "correctness",
                             "PASS" if linear_error < 3e-14 else "FAIL", {"maximum_absolute_defect": linear_error},
                             parameters, "Strang has exact first variation about a homogeneous trajectory; linear sine propagation cannot establish a temporal advantage."))
            targets = refined - split
            for variant in VARIANTS:
                check_budget()
                coefficients, fit_info = fit_coefficients(all_h[:fit_count], targets[:fit_count], variant, **parameters)
                design = basis(all_h, variant, **parameters)
                prediction = design @ coefficients
                predicted32 = basis(all_h, variant, dtype=np.float32, **parameters) @ coefficients.astype(np.float32)
                errors = np.abs(prediction - targets)
                scale = max(float(np.max(np.abs(targets))), 1e-30)
                reference_floor = max(quad_difference, ode_error, 1e-14)
                metrics = {**fit_info,
                           "nominal_coefficient_count": 2,
                           "effective_coefficient_count": fit_info["rank"],
                           "two_coefficients_identifiable": fit_info["rank"] == 2,
                           "fit_max_absolute_error": float(errors[:fit_count].max()),
                           "validation_max_absolute_error": float(errors[fit_count:fit_count + val_count].max()),
                           "test_max_absolute_error": float(errors[fit_count + val_count:].max()),
                           "test_scaled_max_error": float(errors[fit_count + val_count:].max() / scale),
                           "test_extrapolation_absolute_error": float(errors[-1]),
                           "fp32_max_absolute_disagreement": float(np.max(np.abs(predicted32 - prediction))),
                           "maximum_defect_amplitude": scale,
                           "maximum_term_cancellation": float(np.max(np.sum(np.abs(design * coefficients), axis=1) / np.maximum(np.abs(prediction), reference_floor))),
                           "reference_error_floor": reference_floor,
                           "coefficient_0": float(coefficients[0]), "coefficient_1": float(coefficients[1])}
                for partition, indexes in (("validation", range(fit_count, fit_count + val_count)),
                                           ("test", range(fit_count + val_count, len(all_h)))):
                    for index, position in enumerate(indexes):
                        for name, value in (("horizon", all_h[position]), ("reference", targets[position]),
                                            ("prediction", prediction[position]), ("absolute_error", errors[position])):
                            metrics[f"{partition}_{index}_{name}"] = float(value)
                reference_resolved = accepted and scale > 100 * reference_floor
                known_dc_degeneracy = variant == "transported" and p + q == 0 and fit_info["rank"] == 1
                if reference_resolved and known_dc_degeneracy:
                    outcome = "EXPECTED_LIMITATION"
                    note = "At output mode zero, diffusion transport is identity: both transported columns are identical. Two nominal coefficients are nonidentifiable with only one effective coefficient. Predictive errors are retained; this is not a matched-two-effective-coefficient comparison."
                else:
                    outcome = "OBSERVED" if reference_resolved and fit_info["rank"] == 2 else "INCONCLUSIVE"
                    note = "Exactly two nominal parent-specific coefficients; fit-only defect/h^3 least squares. Validation does not select a variant; all test errors retained. Reaction-pair knows additional pair/background information. Unexpected rank deficiency or unresolved reference remains inconclusive."
                rows.append(_row(f"{prefix}-{variant}", "heldout_oracle_basis", variant, "representation",
                                 outcome, metrics, parameters, note))
        # Same output mode, different generating pairs; amplitude-normalized.
        left = interaction_coefficients(all_h, p=1, q=3, kappa=kappa, rate=rate)[0]
        right = interaction_coefficients(all_h, p=2, q=2, kappa=kappa, rate=rate)[0]
        difference = float(np.max(np.abs(left - right)))
        rows.append(_row(f"collision-r{ri}", "same_output_mode_different_input_pairs", "quadratic_response", "scientific",
                         "OBSERVED", {"maximum_normalized_response_difference": difference,
                                      "output_eigenvalue": eigenvalue(4),
                                      "pair_13_eigenvalue_sum": eigenvalue(1) + eigenvalue(3),
                                      "pair_22_eigenvalue_sum": 2 * eigenvalue(2)},
                         {"kappa": kappa, "rate": rate, "pairs": [[1, 3], [2, 2]]},
                         "Different pairwise diffusion sums affect temporal response despite an identical output mode. This does not prove a fitted two-coefficient output-only basis must lose."))
    for p, q in pairs:
        check_budget()
        metrics = finite_amplitude_scaling(p, q, check_budget)
        orders = [value for key, value in metrics.items() if "residual_order" in key]
        outcome = "PASS" if all(value is not None and 3.7 <= value <= 4.3 for value in orders) else "FAIL"
        rows.append(_row(f"finite-amplitude-p{p}-q{q}", "finite_amplitude_residual_scaling", "coupled_pde_and_strang", "correctness",
                         outcome, metrics, {"p": p, "q": q, **frozen_plan["finite_amplitude_check"]},
                         "Three amplitudes test fourth-order even residuals after subtracting the analytic quadratic term. Four-sign polarization removes overlapping self harmonics; DC uses its constant-mode normalization."))
    return {"panel": "temporal", "status": "COMPLETED", "rows": rows,
            "summary": {"rows": len(rows), "correctness_failures": sum(row["kind"] == "correctness" and row["outcome"] == "FAIL" for row in rows),
                        "oracle_rows": sum(row["kind"] == "representation" for row in rows),
                        "trained_models": 0, "optimizer_steps": 0},
            "plan": frozen_plan,
            "limitations": ["Perturbative one-dimensional interaction coefficients are not finite-amplitude PDE solver results or trained-model performance.",
                            "Per-parent oracle coefficients provide optimistic representation diagnostics; there is no cross-parent generalization claim.",
                            "All variants have two nominal coefficients; transported DC has identical columns and only one effective coefficient. Reaction_pair has explicit generating-pair/background information unavailable to an output-frequency-only decoder.",
                            "The fixed-grid small-step model does not establish mesh-uniform order, long-time stability, positivity, or A100 speed.",
                            "FP32 results evaluate fixed FP64-fitted coefficients, not FP32 training or GPU execution."]}
