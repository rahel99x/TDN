"""Raw convergence evidence and per-anchor temporal representation oracles."""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares


def fit_order(horizons, errors, *, noise_floor: float = 0.0,
              reference_uncertainty=None, max_h: float | None = None,
              min_points: int = 3) -> dict:
    h, e = np.asarray(horizons, float), np.asarray(errors, float)
    if h.ndim != 1 or e.shape != h.shape or min_points < 3:
        raise ValueError("Fit needs matching vectors and at least three points")
    ref = np.zeros_like(e) if reference_uncertainty is None else np.asarray(reference_uncertainty, float)
    if ref.shape != e.shape:
        raise ValueError("Reference uncertainty shape mismatch")
    mask = np.isfinite(e) & (e > 10 * np.maximum(noise_floor, ref)) & (e > 0) & (h > 0)
    if max_h is not None:
        mask &= h <= max_h
    result = {"raw_horizons": h.tolist(), "raw_errors": e.tolist(),
              "raw_reference_uncertainty": ref.tolist(), "fit_mask": mask.tolist(),
              "selection_rule": "positive errors > 10 * max(noise_floor, reference uncertainty); predeclared max_h",
              "noise_floor": noise_floor, "max_h": max_h, "slope": None, "intercept": None,
              "minimum_points": min_points, "valid": False}
    if mask.sum() < min_points:
        result["reason"] = "Fewer than three points above the audited reference/roundoff floor"
        return result
    log_h, log_e = np.log(h[mask]), np.log(e[mask])
    if np.unique(log_h).size < min_points:
        result["reason"] = "Fewer than three distinct horizons"
        return result
    slope, intercept = np.polyfit(log_h, log_e, 1)
    predicted = slope * log_h + intercept
    denominator = float(np.sum((log_e - log_e.mean()) ** 2))
    result.update(slope=float(slope), intercept=float(intercept), valid=True,
                  r_squared=float(1 - np.sum((log_e - predicted) ** 2) / denominator) if denominator else None,
                  fitted_h_range=[float(h[mask].min()), float(h[mask].max())])
    return result


def tier_a_audit() -> dict:
    """Exact noncommuting matrix control, plus commuting/zero limits."""
    A = np.array([[-.3, .8], [-.1, -.2]])
    B = np.array([[-1., .1], [.4, -.8]])
    u = np.array([.7, -.4])
    leading = np.zeros_like(A)
    for p in range(4):
        for q in range(4 - p):
            r = 3 - p - q
            leading += (np.linalg.matrix_power(A / 2, p) / math.factorial(p)
                        @ (np.linalg.matrix_power(B, q) / math.factorial(q))
                        @ (np.linalg.matrix_power(A / 2, r) / math.factorial(r)))
    E3 = np.linalg.matrix_power(A + B, 3) / 6 - leading
    horizons = 1 / np.asarray([8, 16, 32, 64], float)
    local, local_anchor, global_base, global_anchor = [], [], [], []
    for h in horizons:
        S = expm(h * A / 2) @ expm(h * B) @ expm(h * A / 2)
        C = S + h**3 * E3
        reference = expm(h * (A + B)) @ u
        local.append(float(np.linalg.norm(S @ u - reference)))
        local_anchor.append(float(np.linalg.norm(C @ u - reference)))
        count = int(round(1 / h))
        global_base.append(float(np.linalg.norm(np.linalg.matrix_power(S, count) @ u - expm(A + B) @ u)))
        global_anchor.append(float(np.linalg.norm(np.linalg.matrix_power(C, count) @ u - expm(A + B) @ u)))
    fits = {"split_local": fit_order(horizons, local, noise_floor=1e-14),
            "anchored_local": fit_order(horizons, local_anchor, noise_floor=1e-14),
            "split_global": fit_order(horizons, global_base, noise_floor=1e-14),
            "anchored_global": fit_order(horizons, global_anchor, noise_floor=1e-14)}
    limits = {}
    for label, left, right in [("commuting", A, .4 * A), ("zero_reaction", A, np.zeros_like(A)),
                               ("zero_diffusion", np.zeros_like(B), B), ("zero_advection", A, B)]:
        # The two-operator control has no third advection operator by construction.
        if label == "zero_advection":
            limits[label] = {"scope": "two-operator model contains no advection"}
            continue
        difference = expm(.1 * left / 2) @ expm(.1 * right) @ expm(.1 * left / 2) - expm(.1 * (left + right))
        limits[label] = {"error": float(np.linalg.norm(difference)), "passed": bool(np.linalg.norm(difference) < 1e-13)}
    passed = (all(f["valid"] for f in fits.values()) and
              2.8 < fits["split_local"]["slope"] < 3.2 and
              3.7 < fits["anchored_local"]["slope"] < 4.3 and
              1.8 < fits["split_global"]["slope"] < 2.2 and
              2.8 < fits["anchored_global"]["slope"] < 3.2 and
              all(v.get("passed", True) for v in limits.values()))
    return {"scope": "tiny exact semidiscrete linear controls, not PDE or GPU evidence", "passed": passed,
            "A": A.tolist(), "B": B.tolist(), "commutator_norm": float(np.linalg.norm(A @ B - B @ A)),
            "leading_matrix": E3.tolist(), "fits": fits, "limits": limits}


def _psi_numpy(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    polynomial = np.full_like(x, 2 / math.factorial(17))
    for degree in range(13, -1, -1):
        polynomial = polynomial * x + 2 * (-1)**degree / math.factorial(degree + 3)
    safe = np.maximum(x, .5)
    inverse = (1 / safe) * (1 - 2 / safe - 2 * np.expm1(-safe) / safe**2)
    return np.where(x <= .5, polynomial, inverse)


MAX_TEMPORAL_ORACLE_MODES = 16
_RATE_BOUNDS = (1e-4, 1e4)


def _horizon_vector(values, name: str, *, minimum: int = 1) -> np.ndarray:
    values = np.asarray(values, float)
    if (values.ndim != 1 or len(values) < minimum or not np.isfinite(values).all()
            or (values <= 0).any()):
        raise ValueError(f"{name} must contain at least {minimum} finite positive horizons")
    ordered = np.sort(values)
    if np.isclose(ordered[:-1], ordered[1:], rtol=1e-12, atol=0).any():
        raise ValueError(f"{name} must contain distinct horizons")
    return values


def temporal_screen_horizons(horizons, *, modes: int = 4) -> tuple[list[float], list[float]]:
    """Expand fitting coverage without moving the original held-out targets.

    Sorted original adjacent midpoints and the 1.25-times-maximum extrapolation remain
    held out. Odd subdivisions add only fitting points, chosen before labels.
    Four original horizons/four modes produce ten fit and four held-out points.
    """
    original = np.sort(_horizon_vector(horizons, "Original fit horizons", minimum=3))
    if isinstance(modes, bool) or not isinstance(modes, (int, np.integer)) or not 1 <= modes <= MAX_TEMPORAL_ORACLE_MODES:
        raise ValueError(f"Temporal oracle modes must be an integer in [1, {MAX_TEMPORAL_ORACLE_MODES}]")
    heldout = np.r_[(original[:-1] + original[1:]) / 2, 1.25 * original[-1]]
    _horizon_vector(heldout, "Generated held-out horizons")
    fitting = list(original)
    # Start with thirds even when the original grid is already overdetermined;
    # this gives the audit a consistent, predeclared richer temporal coverage.
    denominator = 3
    while denominator == 3 or len(fitting) <= modes:
        for left, right in zip(original[:-1], original[1:]):
            for numerator in range(1, denominator):
                candidate = float(left + (right - left) * numerator / denominator)
                if (not np.isclose(candidate, heldout, rtol=1e-12, atol=0).any()
                        and not np.isclose(candidate, fitting, rtol=1e-12, atol=0).any()):
                    fitting.append(candidate)
        denominator += 2
        if denominator > 35 and len(fitting) <= modes:
            raise ValueError("Original horizons are too close to expand a disjoint fitting grid")
    return sorted(float(h) for h in fitting), heldout.tolist()


def temporal_oracle_fits(horizons, defects, heldout_horizons, heldout_defects, *,
                         fixed_rates=(.1, 1., 10., 100.), t_ref: float = 1.,
                         noise_floor: float = 1e-10) -> dict:
    """Fit teacher amplitudes independently at each spatial output (oracle only).

    These fits use labels inaccessible to a deployment encoder. They diagnose
    temporal representation headroom and are never reported as learned solvers.
    """
    h = _horizon_vector(horizons, "Fit horizons", minimum=3)
    test_h = _horizon_vector(heldout_horizons, "Held-out horizons")
    y, test_y = np.asarray(defects, float), np.asarray(heldout_defects, float)
    if (y.ndim < 1 or test_y.ndim < 1 or y.shape[0] != len(h)
            or test_y.shape[0] != len(test_h) or y.shape[1:] != test_y.shape[1:]
            or y.size == 0 or not np.isfinite(y).all() or not np.isfinite(test_y).all()):
        raise ValueError("Finite aligned fitting and held-out targets with matching output shape required")
    if np.isclose(h[:, None], test_h[None, :], rtol=1e-12, atol=0).any():
        raise ValueError("Fit and held-out horizons must be disjoint")
    if not np.isfinite(t_ref) or t_ref <= 0 or not np.isfinite(noise_floor) or noise_floor <= 0:
        raise ValueError("Finite positive t_ref and noise_floor required")
    shape = y.shape[1:]
    y = y.reshape(len(h), -1)
    test_y = test_y.reshape(len(test_h), -1)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        normalized = y / h[:, None]**3
        tau, test_tau = h / t_ref, test_h / t_ref
    if not np.isfinite(normalized).all() or not np.isfinite(tau).all() or not np.isfinite(test_tau).all():
        raise ValueError("Scaled fitting targets and dimensionless horizons must be finite")
    rates = np.asarray(fixed_rates, float)
    if (rates.ndim != 1 or not 1 <= len(rates) <= MAX_TEMPORAL_ORACLE_MODES
            or not np.isfinite(rates).all() or (rates < 0).any()
            or (rates > _RATE_BOUNDS[1]).any()):
        raise ValueError(f"One to {MAX_TEMPORAL_ORACLE_MODES} finite fixed rates in [0, {_RATE_BOUNDS[1]:g}] required")
    if np.unique(rates).size != len(rates):
        raise ValueError("Distinct fixed rates required")
    output = {"scope": "per-anchor teacher-informed representation oracle; not deployable, not trained encoder",
              "fit_horizons": h.tolist(), "heldout_horizons": test_h.tolist(),
              "output_shape": list(shape), "h_independent_coefficients_per_anchor": True,
              "fit_and_heldout_horizons_disjoint": True,
              "selection_rule": "rate optimization uses fitting labels only; held-out labels are evaluation only",
              "models": {}}
    def record(name: str, X: np.ndarray, X_test: np.ndarray, metadata: dict | None = None) -> None:
        if not np.isfinite(X).all() or not np.isfinite(X_test).all():
            raise ValueError("Temporal design matrices must be finite")
        coefficients, _, rank, _ = np.linalg.lstsq(X, normalized, rcond=1e-12)
        prediction = (X_test @ coefficients) * test_h[:, None]**3
        fit = (X @ coefficients) * h[:, None]**3
        modes = X_test[:, :, None] * coefficients[None, :, :] * test_h[:, None, None]**3
        absolute = np.sqrt(np.mean((prediction - test_y)**2, axis=1))
        magnitude = np.sqrt(np.mean(test_y**2, axis=1))
        cancellation = np.sum(np.abs(modes), axis=1) / (np.abs(prediction) + noise_floor)
        condition = float(np.linalg.cond(X))
        output["models"][name] = {"design_condition_number": condition if np.isfinite(condition) else None, "rank": int(rank),
            "columns": X.shape[1], "fit_rms": float(np.sqrt(np.mean((fit - y)**2))),
            "amplitude_fit_overdetermined": bool(len(h) > X.shape[1]),
            "amplitude_design_full_rank": bool(rank == X.shape[1]),
            "fit_residual_degrees_of_freedom_per_output": int(len(h) - X.shape[1]),
            "heldout_absolute_rms": absolute.tolist(),
            "heldout_relative_rms_with_noise_floor": (absolute / np.maximum(magnitude, noise_floor)).tolist(),
            "coefficient_min": float(coefficients.min()), "coefficient_max": float(coefficients.max()),
            "cancellation_ratio_max": float(cancellation.max()),
            "cancellation_ratio_median": float(np.median(cancellation)), **(metadata or {})}
    record("fixed_rate", _psi_numpy(tau[:, None] * rates), _psi_numpy(test_tau[:, None] * rates),
           {"rates_reference_time_units": rates.tolist(), "rate_tuning": False})
    count = len(rates)
    record("polynomial", np.stack([tau**p for p in range(count)], axis=1),
           np.stack([test_tau**p for p in range(count)], axis=1), {"anchored_powers": list(range(3, 3 + count))})
    # A fixed denominator dictionary gives a pole-free rational control with an
    # identical number of fitted coefficients. No singular denominator is hidden.
    denominator_rates = np.geomspace(.1, 100., count)
    record("rational", 1 / (1 + tau[:, None] * denominator_rates),
           1 / (1 + test_tau[:, None] * denominator_rates),
           {"form": "h^3 sum a_j/(1+b_j*h/t_ref), b_j>=0 fixed", "denominator_rates": denominator_rates.tolist()})
    # Bounded, shared-rate variable projection, with the same amplitude columns.
    # A square or underdetermined amplitude fit can interpolate every fitting
    # label for many rate choices, so its rates carry no optimization evidence.
    residual_calls = 0
    residual_scale = max(float(np.sqrt(np.mean(normalized**2))), noise_floor)
    def residual(log_rates: np.ndarray) -> np.ndarray:
        nonlocal residual_calls
        residual_calls += 1
        X = _psi_numpy(tau[:, None] * np.exp(log_rates))
        coefficients = np.linalg.lstsq(X, normalized, rcond=1e-12)[0]
        return ((X @ coefficients - normalized) / residual_scale).ravel()

    def jacobian(log_rates: np.ndarray) -> np.ndarray:
        # A larger symmetric log-rate increment avoids subtractive cancellation
        # in the projected least-squares amplitudes. It is fixed before fitting.
        step = 1e-4
        columns = []
        for column in range(len(log_rates)):
            direction = np.zeros_like(log_rates)
            direction[column] = step
            columns.append((residual(log_rates + direction) - residual(log_rates - direction)) / (2 * step))
        return np.stack(columns, axis=1)

    initial_log_rates = np.log(np.maximum(rates, _RATE_BOUNDS[0]))
    structural = bool(len(h) > len(rates) and output["models"]["fixed_rate"]["amplitude_design_full_rank"]
                      and (len(h) - len(rates)) * normalized.shape[1] >= len(rates))
    if structural:
        initial_residual = residual(initial_log_rates)
        initial_rates = np.exp(initial_log_rates)
    else:
        # No optimizer runs: describe the reported fixed-rate fallback, including
        # a legal zero rate, rather than an unused log-rate-clamped initialization.
        initial_design = _psi_numpy(tau[:, None] * rates)
        initial_amplitudes = np.linalg.lstsq(initial_design, normalized, rcond=1e-12)[0]
        initial_residual = ((initial_design @ initial_amplitudes - normalized) / residual_scale).ravel()
        initial_rates = rates.copy()
    initial_cost = float(np.sum(initial_residual**2) / 2)
    learned_rates = rates.copy()
    metadata = {"rate_fit_structurally_identifiable": structural,
                "rate_fit_identifiable": False, "informative_rate_fit": False,
                "rate_function_evaluations": 0, "optimizer_success": False,
                "rate_tuning_budget": 40, "rate_optimizer_nfev_budget": 40,
                "rate_tuning_budget_convention": "SciPy optimizer nfev; finite-difference Jacobian calls are counted separately in rate_residual_evaluations",
                "rate_residual_evaluation_upper_bound": 1 + 40 * (1 + 2 * len(rates)),
                "rate_jacobian_log_step": 1e-4, "rate_bounds_reference_time_units": list(_RATE_BOUNDS),
                "initial_rates_reference_time_units": initial_rates.tolist(),
                "initial_normalized_fit_cost": initial_cost, "final_normalized_fit_cost": initial_cost,
                "rate_jacobian_rank": None, "rate_jacobian_singular_values": [],
                "shared_fitted_rate_parameters": len(rates),
                "rate_fit_reason": "Rate optimization skipped: amplitude fit is saturated, rank deficient, or has insufficient residual degrees of freedom"}
    if structural:
        learned = least_squares(residual, initial_log_rates, jac=jacobian,
                                bounds=tuple(np.log(_RATE_BOUNDS)), max_nfev=40)
        learned_rates = np.clip(np.exp(learned.x), *_RATE_BOUNDS)
        singular_values = np.linalg.svd(learned.jac, compute_uv=False)
        threshold = max(1e-10, float(singular_values[0]) * 1e-6) if len(singular_values) else 1e-10
        rate_rank = int(np.sum(singular_values > threshold))
        identifiable = bool(rate_rank == len(rates))
        improved = bool(initial_cost - learned.cost > max(1e-24, initial_cost * 1e-8))
        informative = bool(learned.success and learned.nfev > 1 and identifiable and improved)
        metadata.update(rate_fit_identifiable=identifiable, informative_rate_fit=informative,
                        optimizer_success=bool(learned.success), rate_function_evaluations=int(learned.nfev),
                        final_normalized_fit_cost=float(learned.cost), rate_jacobian_rank=rate_rank,
                        rate_jacobian_singular_values=singular_values.tolist(),
                        rate_jacobian_rank_threshold=threshold,
                        rate_fit_reason="Successful informative fitting-only rate optimization" if informative
                        else "Rate optimization did not establish a successful, improved, locally identifiable multi-evaluation fit")
    metadata.update(rates_reference_time_units=learned_rates.tolist(), rate_residual_evaluations=residual_calls,
                    limitation="Local projected-Jacobian rank is a numerical screen, not a global identifiability guarantee; held-out errors do not tune rates")
    record("learned_rate_oracle", _psi_numpy(tau[:, None] * learned_rates),
           _psi_numpy(test_tau[:, None] * learned_rates),
           metadata)
    return output


def configure_plot_cache(root: Path) -> None:
    """Prevent matplotlib/fontconfig from falling back to a system temp root."""
    cache = Path(root).resolve() / ".cache"
    mpl = cache / "matplotlib"
    mpl.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl)
    os.environ["XDG_CACHE_HOME"] = str(cache)


def plot_orders(fits: dict, path: Path, *, title: str) -> None:
    configure_plot_cache(path.parent)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, ax = plt.subplots()
    for name, fit in fits.items():
        h, e = np.asarray(fit["raw_horizons"]), np.asarray(fit["raw_errors"])
        positive = e > 0
        ax.loglog(h[positive], e[positive], "o-", label=name)
        mask = np.asarray(fit["fit_mask"], bool)
        if fit["valid"]:
            ax.loglog(h[mask], np.exp(fit["intercept"]) * h[mask]**fit["slope"], "--",
                      label=f"{name} fit slope={fit['slope']:.3f}")
    ax.set(xlabel="macrostep h", ylabel="discrete weighted error", title=title)
    ax.legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
