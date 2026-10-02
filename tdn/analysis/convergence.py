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


def temporal_oracle_fits(horizons, defects, heldout_horizons, heldout_defects, *,
                         fixed_rates=(.1, 1., 10., 100.), t_ref: float = 1.,
                         noise_floor: float = 1e-10) -> dict:
    """Fit teacher amplitudes independently at each spatial output (oracle only).

    These fits use labels inaccessible to a deployment encoder. They diagnose
    temporal representation headroom and are never reported as learned solvers.
    """
    h, test_h = np.asarray(horizons, float), np.asarray(heldout_horizons, float)
    y, test_y = np.asarray(defects, float), np.asarray(heldout_defects, float)
    if len(h) < 3 or y.shape[0] != len(h) or test_y.shape[0] != len(test_h):
        raise ValueError("Need at least three training horizons and aligned held-out targets")
    if (h <= 0).any() or (test_h <= 0).any() or t_ref <= 0:
        raise ValueError("Positive horizons and fixed t_ref required")
    shape = y.shape[1:]
    y = y.reshape(len(h), -1)
    test_y = test_y.reshape(len(test_h), -1)
    normalized = y / h[:, None]**3
    tau, test_tau = h / t_ref, test_h / t_ref
    rates = np.asarray(fixed_rates, float)
    if (rates < 0).any() or rates.ndim != 1 or not len(rates):
        raise ValueError("Nonnegative fixed rates required")
    output = {"scope": "per-anchor teacher-informed representation oracle; not deployable, not trained encoder",
              "fit_horizons": h.tolist(), "heldout_horizons": test_h.tolist(),
              "output_shape": list(shape), "h_independent_coefficients_per_anchor": True,
              "models": {}}
    def record(name: str, X: np.ndarray, X_test: np.ndarray, metadata: dict | None = None) -> None:
        coefficients, _, rank, _ = np.linalg.lstsq(X, normalized, rcond=1e-12)
        prediction = (X_test @ coefficients) * test_h[:, None]**3
        fit = (X @ coefficients) * h[:, None]**3
        modes = X_test[:, :, None] * coefficients[None, :, :] * test_h[:, None, None]**3
        absolute = np.sqrt(np.mean((prediction - test_y)**2, axis=1))
        magnitude = np.sqrt(np.mean(test_y**2, axis=1))
        cancellation = np.sum(np.abs(modes), axis=1) / (np.abs(prediction) + noise_floor)
        output["models"][name] = {"design_condition_number": float(np.linalg.cond(X)), "rank": int(rank),
            "columns": X.shape[1], "fit_rms": float(np.sqrt(np.mean((fit - y)**2))),
            "heldout_absolute_rms": absolute.tolist(),
            "heldout_relative_rms_with_noise_floor": (absolute / np.maximum(magnitude, noise_floor)).tolist(),
            "coefficient_min": float(coefficients.min()), "coefficient_max": float(coefficients.max()),
            "cancellation_ratio_max": float(cancellation.max()),
            "cancellation_ratio_median": float(np.median(cancellation)), **(metadata or {})}
    record("fixed_rate", _psi_numpy(tau[:, None] * rates), _psi_numpy(test_tau[:, None] * rates),
           {"rates_reference_time_units": rates.tolist(), "rate_tuning": False})
    count = min(len(rates), len(h))
    record("polynomial", np.stack([tau**p for p in range(count)], axis=1),
           np.stack([test_tau**p for p in range(count)], axis=1), {"anchored_powers": list(range(3, 3 + count))})
    # A fixed denominator dictionary gives a pole-free rational control with an
    # identical number of fitted coefficients. No singular denominator is hidden.
    denominator_rates = np.geomspace(.1, 100., count)
    record("rational", 1 / (1 + tau[:, None] * denominator_rates),
           1 / (1 + test_tau[:, None] * denominator_rates),
           {"form": "h^3 sum a_j/(1+b_j*h/t_ref), b_j>=0 fixed", "denominator_rates": denominator_rates.tolist()})
    # Bounded, shared-rate variable projection, with the same amplitude columns.
    def residual(log_rates: np.ndarray) -> np.ndarray:
        X = _psi_numpy(tau[:, None] * np.exp(log_rates))
        coefficients = np.linalg.lstsq(X, normalized, rcond=1e-12)[0]
        return ((X @ coefficients - normalized) / max(float(np.sqrt(np.mean(normalized**2))), noise_floor)).ravel()
    learned = least_squares(residual, np.log(np.maximum(rates, 1e-4)),
                            bounds=(np.log(1e-4), np.log(1e4)), max_nfev=40)
    learned_rates = np.exp(learned.x)
    record("learned_rate_oracle", _psi_numpy(tau[:, None] * learned_rates),
           _psi_numpy(test_tau[:, None] * learned_rates),
           {"rates_reference_time_units": learned_rates.tolist(), "rate_function_evaluations": learned.nfev,
            "optimizer_success": bool(learned.success), "rate_tuning_budget": 40,
            "limitation": "four horizons and four coefficients can interpolate; held-out errors decide representation"})
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
