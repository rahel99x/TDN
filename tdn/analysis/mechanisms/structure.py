"""Small, training-free probes of operator structure and deployment claims.

These are mathematical controls, not Burgers/Navier--Stokes benchmarks, a
trained neural operator, or a certified adaptive neural solver.  Negative
controls deliberately show why a desirable identity alone is insufficient.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable

import numpy as np
import torch
from scipy.signal import resample

from tdn.numerics.operators import rhs, weighted_norm
from tdn.numerics.reference import reference_step
from tdn.numerics.splitting import split_step
from tdn.numerics.subflows import diffusion_step, reaction_step
from tdn.numerics.types import Equation, Geometry


def filtered_square(values: np.ndarray, oversampling: int = 4) -> np.ndarray:
    """Interpolate, square, and Fourier-project to the original 1D grid.

    The experiments exclude Nyquist input/output modes so that the real-even
    Nyquist convention cannot obscure their independent analytic answer.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 4 or oversampling < 2:
        raise ValueError("expected a 1D periodic grid and oversampling >= 2")
    return resample(resample(values, values.size * oversampling) ** 2, values.size)


def additive_axis_projection(multiplier: np.ndarray) -> np.ndarray:
    """Frobenius-orthogonal projection onto a(k_x) + b(k_y)."""
    multiplier = np.asarray(multiplier, dtype=np.float64)
    if multiplier.ndim != 2:
        raise ValueError("expected a two-axis scalar multiplier")
    return multiplier.mean(1, keepdims=True) + multiplier.mean(0, keepdims=True) - multiplier.mean()


def rank_projection(multiplier: np.ndarray, rank: int) -> np.ndarray:
    """Best matrix-rank approximation; an oracle, not a trained F-FNO."""
    multiplier = np.asarray(multiplier, dtype=np.float64)
    if multiplier.ndim != 2 or not 1 <= rank <= min(multiplier.shape):
        raise ValueError("rank must fit a two-axis multiplier")
    left, singular, right = np.linalg.svd(multiplier, full_matrices=False)
    return (left[:, :rank] * singular[:rank]) @ right[:rank]


def compatible_projection(vector: np.ndarray) -> np.ndarray:
    """Periodic projection using forward gradient and backward divergence.

    This is a scalar-potential grid identity, not a Navier--Stokes time step.
    The two velocity components are on the same algebraic periodic array;
    no claim about a full staggered-grid solver or physical boundaries follows.
    """
    vector = np.asarray(vector, dtype=np.float64)
    if vector.ndim != 3 or vector.shape[0] != 2 or min(vector.shape[1:]) < 4:
        raise ValueError("expected vector shape [2, n_x, n_y], n >= 4")
    nx, ny = vector.shape[1:]
    qx = (np.exp(2j * np.pi * np.fft.fftfreq(nx)) - 1)[:, None] * nx
    qy = (np.exp(2j * np.pi * np.fft.fftfreq(ny)) - 1)[None, :] * ny
    lap = -(np.abs(qx) ** 2 + np.abs(qy) ** 2)
    velocity = np.fft.fftn(vector, axes=(-2, -1))
    divergence = -(np.conj(qx) * velocity[0] + np.conj(qy) * velocity[1])
    potential = np.zeros_like(divergence)
    np.divide(divergence, lap, out=potential, where=lap != 0)
    projected = np.stack((velocity[0] - qx * potential, velocity[1] - qy * potential))
    return np.fft.ifftn(projected, axes=(-2, -1)).real


def _divergence(vector: np.ndarray) -> np.ndarray:
    nx, ny = vector.shape[1:]
    return nx * (vector[0] - np.roll(vector[0], 1, axis=0)) + ny * (
        vector[1] - np.roll(vector[1], 1, axis=1))


def heun_step(value: float, step: float, rate: float) -> float:
    """Explicit trapezoidal rule for y' = -rate*y (two RHS evaluations)."""
    first = -rate * value
    return value + step * (first - rate * (value + step * first)) / 2


def scalar_controller(tolerance: float, check_budget: Callable[[], None], *,
                      rate: float = 2.0, end: float = 1.0,
                      initial_step: float = 0.75) -> dict:
    """Counted Heun step-doubling demonstrator, with local error density.

    Accept the two-half-step value; charge all six RHS evaluations on every
    attempted step, including rejects.  Neither the estimator nor accumulated
    local targets constitute a rigorous global error bound.
    """
    if not all(math.isfinite(v) and v > 0 for v in (tolerance, rate, end, initial_step)):
        raise ValueError("controller parameters must be finite and positive")
    time, value, step = 0.0, 1.0, min(initial_step, end)
    attempts = accepted = rejected = 0
    min_step, maximum_accepted_estimate_ratio = end, 0.0
    while time < end:
        check_budget()
        if attempts >= 10000:
            raise RuntimeError("scalar controller exceeded its independent attempt bound")
        step = min(step, end - time)
        if step < 1e-12 * end or time + step == time:
            raise RuntimeError("scalar controller reached minimum step")
        coarse = heun_step(value, step, rate)
        fine = heun_step(heun_step(value, step / 2, rate), step / 2, rate)
        estimate = abs(fine - coarse) / 3  # p=2, error estimate for finer value.
        target = tolerance * step / end
        attempts += 1
        if estimate <= target:
            accepted += 1
            value = fine
            time = end if step == end - time else time + step
            min_step = min(min_step, step)
            maximum_accepted_estimate_ratio = max(maximum_accepted_estimate_ratio, estimate / target)
        else:
            rejected += 1
        factor = 4.0 if estimate == 0 else min(4.0, max(0.2, 0.9 * (target / estimate) ** (1 / 3)))
        step *= factor
    actual_error = abs(value - math.exp(-rate * end))
    return {"accepted_steps": accepted, "rejected_steps": rejected, "attempted_steps": attempts,
            "rhs_evaluations": 6 * attempts, "final_time": time, "requested_final_time": end,
            "minimum_accepted_step": min_step, "absolute_final_error": actual_error,
            "requested_tolerance": tolerance, "final_tolerance_met": actual_error <= tolerance,
            "maximum_accepted_estimate_ratio": maximum_accepted_estimate_ratio}


def _rms(values) -> float:
    return float(np.sqrt(np.mean(np.asarray(values) ** 2)))


def _row(case_id: str, mechanism: str, test: str, variant: str, kind: str,
         outcome: str, metrics: dict, inputs: dict, note: str) -> dict:
    return dict(case_id=case_id, mechanism=mechanism, test=test, variant=variant,
                kind=kind, outcome=outcome, metrics=metrics, inputs=inputs, note=note)


def _alias_rows(check_budget: Callable[[], None]) -> list[dict]:
    rows = []
    n = 16
    x = np.arange(n) / n
    for mode in (3, 6):
        for phase in (0.0, 0.37):
            check_budget()
            values = np.cos(2 * np.pi * mode * x + phase)
            target = np.full(n, 0.5)
            resolved = 2 * mode < n / 2
            if resolved:
                target += 0.5 * np.cos(4 * np.pi * mode * x + 2 * phase)
            naive, filtered = values ** 2, filtered_square(values)
            for name, output in (("coarse_square", naive), ("oversample_filter", filtered)):
                error = _rms(output - target)
                negative = not resolved and name == "coarse_square"
                outcome = ("EXPECTED_LIMITATION" if error > 0.3 else "FAIL") if negative else (
                    "PASS" if error < 2e-13 else "FAIL")
                rows.append(_row(
                    f"alias-square-k{mode}-p{phase}-{name}", "alias_control", "analytic_squared_harmonic",
                    name, "negative_control" if negative else "correctness", outcome,
                    {"rms_error_to_lowpass_target": error, "mean_error": abs(float(output.mean()) - 0.5),
                     "discarded_physical_high_band_energy": 0.0 if resolved else 0.125},
                    {"grid": n, "input_mode": mode, "phase": phase, "oversampling": 4},
                    "Target is the common resolvable Fourier projection, not the unfiltered continuum square; "
                    "filtering deliberately discards the unresolved physical harmonic."))
    check_budget()
    values = 0.4 + 0.2 * np.cos(2 * np.pi * 6 * x + 0.2)
    shifted_error = _rms(filtered_square(np.roll(values, 3)) - np.roll(filtered_square(values), 3))
    fine_x = np.arange(32) / 32
    fine = 0.4 + 0.2 * np.cos(2 * np.pi * 6 * fine_x + 0.2)
    grid_error = _rms(filtered_square(values) - resample(filtered_square(fine), n))
    constant_error = _rms(filtered_square(np.full(n, 0.3)) - 0.09)
    rows.append(_row("alias-resampling-identities", "alias_control", "resampling_identities",
                     "oversample_filter", "correctness", "PASS" if max(shifted_error, grid_error, constant_error) < 2e-13 else "FAIL",
                     {"integer_translation_rms_error": shifted_error, "common_band_grid_rms_error": grid_error,
                      "constant_rms_error": constant_error}, {"coarse_grid": n, "fine_grid": 32, "integer_shift": 3},
                     "The fine output is projected to the coarse resolvable band before comparison; "
                     "this checks the resampling algebra, not neural resolution transfer."))
    return rows


def _factorization_rows(check_budget: Callable[[], None]) -> list[dict]:
    rows = []
    k = np.arange(-4, 5, dtype=float)
    x, y = k[:, None], k[None, :]
    targets = {
        "additive_axes": np.exp(-0.2 * x * x) + np.exp(-0.3 * y * y),
        "separable_product": np.exp(-0.2 * x * x) * np.exp(-0.3 * y * y),
        "oblique_ridge": np.exp(-0.2 * (x + y) ** 2),
    }
    for name, target in targets.items():
        check_budget()
        for representation in ("axis_additive", "rank_1", "rank_3", "dense_rank_9"):
            output = additive_axis_projection(target) if representation == "axis_additive" else rank_projection(
                target, {"rank_1": 1, "rank_3": 3, "dense_rank_9": 9}[representation])
            relative = _rms(target - output) / _rms(target)
            exact = representation == "dense_rank_9" or (name == "additive_axes" and representation in (
                "axis_additive", "rank_3")) or (name == "separable_product" and representation in ("rank_1", "rank_3"))
            rows.append(_row(f"factorization-{name}-{representation}", "spectral_factorization",
                             "linear_multiplier_projection", representation, "correctness" if exact else "representation",
                             ("PASS" if relative < 2e-13 else "FAIL") if exact else "OBSERVED",
                             {"relative_frobenius_error": relative, "rms_error": _rms(target - output)},
                             {"target": name, "modes_per_axis": 9},
                             "Best linear scalar projection only: axis-additive and low-rank product classes differ. "
                             "This is not a bound on a deep nonlinear F-FNO and measures no trained accuracy or latency."))
    return rows


def _composition_rows(check_budget: Callable[[], None], tolerance: float, smoke: bool) -> list[dict]:
    rows = []
    n = 8
    geometry, equation = Geometry((n, n), (1.0, 1.0)), Equation(0.02, 3.0)
    x = torch.arange(n, dtype=torch.float64) / n
    xx, yy = torch.meshgrid(x, x, indexing="ij")
    initial = (0.4 + 0.16 * torch.cos(2 * torch.pi * xx) + 0.08 * torch.sin(4 * torch.pi * yy))[None, None]
    norm = lambda value: float(weighted_norm(value))
    with torch.no_grad():
        for total in ((0.12,) if smoke else (0.06, 0.12, 0.24)):
            states = []
            for count in (32, 64, 128):
                check_budget()
                states.append(reference_step(initial, total, equation, geometry, count))
            differences = [norm(states[j + 1] - states[j]) for j in range(2)]
            uncertainty = max(differences[-1], 64 * torch.finfo(torch.float64).eps)
            resolved = (differences[1] <= differences[0] / 4 or max(differences) < 2e-14) and uncertainty < tolerance / 100
            truth = states[-1]
            for variant in ("strang", "identity", "diffusion_only"):
                check_budget()
                def evolve(value, horizon):
                    if variant == "strang":
                        return split_step(value, horizon, equation, geometry)
                    if variant == "diffusion_only":
                        return diffusion_step(value, horizon, equation, geometry)
                    return value.clone()
                full = evolve(initial, total)
                first = total * 0.37
                composed = evolve(evolve(initial, first), total - first)
                reverse = evolve(evolve(initial, total - first), first)
                eps = 1e-5
                time_derivative = (evolve(initial, total + eps) - evolve(initial, total - eps)) / (2 * eps)
                error = norm(full - truth)
                residual = norm(time_derivative - rhs(full, equation, geometry))
                semigroup = norm(full - composed)
                negative = variant != "strang"
                outcome = "OBSERVED"
                if not resolved:
                    outcome = "INCONCLUSIVE"
                elif negative:
                    outcome = "EXPECTED_LIMITATION" if semigroup < 2e-13 and error > tolerance and residual > tolerance else "FAIL"
                rows.append(_row(f"composition-{total}-{variant}", "composition_and_pde", "unequal_step_and_true_error",
                                 variant, "negative_control" if negative else "scientific", outcome,
                                 {"composition_rms": semigroup, "reverse_partition_rms": norm(full - reverse),
                                  "true_rms_error": error, "reference_uncertainty": uncertainty,
                                  "reference_accepted": resolved, "pde_residual_rms": residual,
                                  "zero_step_rms": norm(evolve(initial, 0) - initial)},
                                 {"grid": n, "horizon": total, "first_step_fraction": 0.37, "kappa": 0.02,
                                  "reaction_rate": 3.0, "reference_substeps": 128, "time_derivative_epsilon": eps},
                                 "Same-grid coupled RK4 at three refinement levels; full last refinement difference is retained. "
                                 "Identity and exact diffusion are perfect semigroups that solve the wrong coupled equation. "
                                 "Finite-difference PDE residual is diagnostic, not a certified error bound."))
    return rows


def _balance_rows(check_budget: Callable[[], None]) -> list[dict]:
    check_budget()
    n = 12
    x = torch.arange(n, dtype=torch.float64) / n
    u = (0.45 + 0.25 * torch.cos(2 * torch.pi * x))[None, None]
    geometry, equation = Geometry((n,), (1.0,)), Equation(0.02, 3.0)
    diffusion_error = abs(float(diffusion_step(u, 0.2, equation, geometry).mean() - u.mean()))
    source = equation.reaction_rate * u * (1 - u)
    source_error = abs(float(rhs(u, equation, geometry).mean() - source.mean()))
    # Independent Gauss--Legendre integration of the exact reaction trajectory.
    nodes, weights = np.polynomial.legendre.leggauss(16)
    integral = 0.0
    for node, weight in zip(nodes, weights):
        check_budget()
        state = reaction_step(u, 0.15 * (float(node) + 1), equation)
        integral += 0.15 * float(weight) * float((equation.reaction_rate * state * (1 - state)).mean())
    gain = float((reaction_step(u, 0.3, equation) - u).mean())
    rows = [_row("balance-reaction-diffusion", "equation_specific_structure", "source_aware_mean_balance",
                 "periodic_rd", "correctness", "PASS" if max(diffusion_error, source_error, abs(gain - integral)) < 2e-13 else "FAIL",
                 {"diffusion_mean_error": diffusion_error, "coupled_mean_derivative_error": source_error,
                  "reaction_mean_gain": gain, "integrated_source_error": abs(gain - integral)},
                 {"grid": n, "kappa": 0.02, "reaction_rate": 3.0, "reaction_horizon": 0.3},
                 "Diffusion conserves the periodic mean; logistic reaction changes it. "
                 "The finite-time source integral here is for the exact reaction subflow, not the coupled RD trajectory.")]
    flux = np.random.default_rng(6041).normal(size=n)
    divergence = n * (flux - np.roll(flux, 1))
    independent_left = flux + 0.1
    bad_divergence = n * (flux - np.roll(independent_left, 1))
    error = abs(float(divergence.mean()))
    rows.append(_row("balance-shared-face-flux", "equation_specific_structure", "shared_face_cancellation",
                     "periodic_flux_algebra", "correctness", "PASS" if error < 2e-13 else "FAIL",
                     {"shared_face_mean_divergence": error, "unshared_face_mean_divergence": abs(float(bad_divergence.mean()))},
                     {"grid": n, "rng_seed": 6041},
                     "Algebraic shared-face telescoping only. Arbitrary fluxes do not establish consistency, entropy stability, "
                     "or Burgers accuracy; the unshared-face example shows why the cancellation matters."))
    check_budget()
    vector = np.random.default_rng(6042).normal(size=(2, n, n))
    projected = compatible_projection(vector)
    divergence_error = _rms(_divergence(projected))
    idempotence_error = _rms(compatible_projection(projected) - projected)
    mean_error = float(np.max(np.abs(projected.mean(axis=(1, 2)) - vector.mean(axis=(1, 2)))))
    rows.append(_row("balance-compatible-projection", "equation_specific_structure", "compatible_gradient_divergence",
                     "periodic_projection_algebra", "correctness", "PASS" if max(divergence_error, idempotence_error, mean_error) < 2e-12 else "FAIL",
                     {"divergence_rms": divergence_error, "idempotence_rms": idempotence_error,
                      "mean_velocity_error": mean_error, "input_energy": float(np.mean(vector ** 2)),
                      "output_energy": float(np.mean(projected ** 2))}, {"grid": n, "rng_seed": 6042},
                     "Compatible forward-gradient/backward-divergence algebra only, using a periodic scalar-potential projection; "
                     "this is not a Navier--Stokes solver or benchmark."))
    return rows


def _controller_rows(check_budget: Callable[[], None], tolerance: float) -> list[dict]:
    rows = []
    for step in (0.2, 0.1, 0.05):
        check_budget()
        coarse = heun_step(1, step, 2)
        fine = heun_step(heun_step(1, step / 2, 2), step / 2, 2)
        error = abs(fine - math.exp(-2 * step))
        estimate = abs(fine - coarse) / 3
        rows.append(_row(f"estimator-heun-{step}", "adaptive_stepping", "step_doubling_known_order",
                         "heun_order_two", "scientific", "OBSERVED",
                         {"finer_actual_error": error, "finer_estimated_error": estimate,
                          "estimate_to_actual_ratio": estimate / error, "rhs_evaluations": 6},
                         {"horizon": step, "rate": 2.0, "assumed_order": 2},
                         "The Richardson denominator 2**p-1 estimates the two-half-step local error only in the asymptotic regime."))
    step = 0.2
    wrong_full = math.exp(-step)
    wrong_fine = math.exp(-step / 2) ** 2
    estimate = abs(wrong_fine - wrong_full) / 3
    actual = abs(wrong_fine - math.exp(-2 * step))
    rows.append(_row("estimator-shared-generator-bias", "adaptive_stepping", "shared_bias_negative_control",
                     "exact_wrong_generator", "negative_control", "EXPECTED_LIMITATION" if estimate < 2e-15 and actual > tolerance else "FAIL",
                     {"estimated_error": estimate, "actual_error": actual, "tolerance_met_by_estimator": estimate <= tolerance,
                      "true_tolerance_met": actual <= tolerance}, {"horizon": step, "true_rate": 2.0, "used_rate": 1.0},
                     "Two agreeing exact wrong-generator predictions fool step doubling; a small estimated error is not proof of correctness."))
    metrics = scalar_controller(tolerance, check_budget)
    rows.append(_row("controller-counted-scalar", "adaptive_stepping", "counted_accept_reject_controller",
                     "heun_step_doubling", "scientific", "OBSERVED", metrics,
                     {"rate": 2.0, "initial_value": 1.0, "initial_step": 0.75, "safety_factor": 0.9},
                     "Accepted two-half-step values; all six RHS calls per attempt, including rejected work, are charged. "
                     "A local target proportional to step length is used; this scalar demonstration does not implement or certify a neural controller."))
    return rows


def plan(config: dict) -> dict:
    """Declare every case and numerical control without computing a result."""
    smoke = bool(config.get("smoke", False))
    horizons = [0.12] if smoke else [0.06, 0.12, 0.24]
    ids = [f"alias-square-k{mode}-p{phase}-{variant}"
           for mode in (3, 6) for phase in (0.0, 0.37)
           for variant in ("coarse_square", "oversample_filter")]
    ids.append("alias-resampling-identities")
    ids.extend(f"factorization-{target}-{representation}"
               for target in ("additive_axes", "separable_product", "oblique_ridge")
               for representation in ("axis_additive", "rank_1", "rank_3", "dense_rank_9"))
    ids.extend(f"composition-{horizon}-{variant}" for horizon in horizons
               for variant in ("strang", "identity", "diffusion_only"))
    ids.extend(("balance-reaction-diffusion", "balance-shared-face-flux", "balance-compatible-projection"))
    ids.extend(f"estimator-heun-{step}" for step in (0.2, 0.1, 0.05))
    ids.extend(("estimator-shared-generator-bias", "controller-counted-scalar"))
    return {
        "panel": "structure", "expected_case_ids": ids,
        "device": "cpu", "dtype": "float64", "training_performed": False,
        "alias_control": {
            "grid": 16, "fine_grid": 32, "oversampling": 4, "modes": [3, 6],
            "phases": [0.0, 0.37], "integer_translation": 3,
            "target": "analytic square projected onto the coarse representable Fourier band",
            "negative_control": "coarse nonlinear evaluation aliases the unresolved k=12 harmonic",
            "absolute_identity_tolerance": 2e-13,
        },
        "spectral_factorization": {
            "axis_modes": list(range(-4, 5)),
            "targets": {"additive_axes": "exp(-0.2*kx^2)+exp(-0.3*ky^2)",
                        "separable_product": "exp(-0.2*kx^2)*exp(-0.3*ky^2)",
                        "oblique_ridge": "exp(-0.2*(kx+ky)^2)"},
            "representations": ["axis_additive", "rank_1", "rank_3", "dense_rank_9"],
            "relative_exact_control_tolerance": 2e-13,
            "scope": "best linear scalar projections, not deep nonlinear F-FNO capability",
        },
        "composition_and_pde": {
            "grid": [8, 8], "lengths": [1.0, 1.0], "kappa": 0.02, "reaction_rate": 3.0,
            "initial_state": "0.4+0.16*cos(2*pi*x)+0.08*sin(4*pi*y)",
            "horizons": horizons, "partition_fraction": 0.37, "reference_counts": [32, 64, 128],
            "reference_acceptance": "last difference <= first/4 or roundoff floor; uncertainty < tolerance/100",
            "time_derivative_epsilon": 1e-5, "controls": ["strang", "identity", "diffusion_only"],
            "negative_controls": "identity and exact diffusion preserve composition while missing coupled dynamics",
        },
        "equation_specific_structure": {
            "grid": 12, "kappa": 0.02, "reaction_rate": 3.0,
            "initial_state": "0.45+0.25*cos(2*pi*x)", "diffusion_horizon": 0.2,
            "reaction_horizon": 0.3, "source_quadrature_order": 16,
            "flux_rng_seed": 6041, "projection_rng_seed": 6042,
            "negative_control": "independent left/right face fluxes differ by 0.1",
            "projection": "periodic forward gradient/backward divergence, unit square",
            "scope": "source-aware RD and algebraic conservation/projection identities, no external PDE benchmark",
        },
        "adaptive_stepping": {
            "scalar_equation": "y'=-2*y, y(0)=1", "estimator_horizons": [0.2, 0.1, 0.05],
            "assumed_order": 2, "fine_error_denominator": 3,
            "negative_control": "exact flow of y'=-y evaluated against true y'=-2*y",
            "controller_final_time": 1.0, "controller_initial_step": 0.75,
            "controller_safety": 0.9, "controller_factor_bounds": [0.2, 4.0],
            "controller_attempt_limit": 10000, "rhs_evaluations_per_attempt": 6,
            "tolerance": float(config.get("tolerance", 0.002)),
            "scope": "counted scalar demonstrator without global error certification",
        },
    }


def run(config: dict, check_budget: Callable[[], None]) -> dict:
    """Run bounded deterministic structural probes without an optimizer."""
    tolerance = float(config.get("tolerance", 0.002))
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    rows = []
    for probe in (_alias_rows, _factorization_rows):
        check_budget()
        rows.extend(probe(check_budget))
    rows.extend(_composition_rows(check_budget, tolerance, bool(config.get("smoke", False))))
    rows.extend(_balance_rows(check_budget))
    rows.extend(_controller_rows(check_budget, tolerance))
    return {"panel": "structure", "status": "COMPLETED", "rows": rows,
            "summary": {"row_count": len(rows), "outcome_counts": dict(Counter(row["outcome"] for row in rows)),
                        "training_performed": False, "device": "cpu", "arithmetic": "float64"},
            "limitations": [
                "Algebraic and synthetic representation tests do not measure trained FNO/TDN accuracy or runtime superiority.",
                "Low-rank scalar multiplier approximation is not a bound on deep nonlinear factorized neural operators.",
                "Composition and error estimates can be exactly satisfied by an incorrect evolution; negative controls remain explicit.",
                "PDE references and projection identities use periodic same-grid operators; no spatial-convergence or external PDE benchmark is claimed.",
                "Filtered outputs are compared on a common resolvable band; discarded physical harmonics are reported, not declared accurately predicted.",
                "The counted scalar controller is a demonstrator, not a deployable adaptive neural integrator."]}
