"""Three bounded, separately seeded development experiments, not confirmation.

The experiments study a representation, an information limit, and allocation of
work. They deliberately do not relabel a second-Picard term as a complete PDE
solver, measured CPU time as GPU time, or new-to-project ideas as new mathematics.
All Fourier sums below are non-aliased Galerkin sums on explicitly listed modes.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np
from numpy.polynomial.chebyshev import chebfit, chebval
from scipy.integrate import solve_ivp

from tdn.analysis.frontier.core import check


THRESHOLDS = {
    "X01": {"quadrature_absolute_error": 1e-10, "decoder_relative_rms": 1e-3,
            "break_even_queries": 32},
    "X02": {"resolved_twin_difference": 1e-12, "target_separation": 1e-8,
            "memory_to_instantaneous_rmse": .5},
    "X03": {"relative_error": .01, "speedup": 1.10},
}


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        return _plain(value.item())
    if isinstance(value, complex):
        return [value.real, value.imag]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write(path, value):
    Path(path).write_text(json.dumps(_plain(value), indent=2, allow_nan=False) + "\n")


def _sum_modes(values, indices, count):
    return (np.bincount(indices, weights=values.real, minlength=count)
            + 1j * np.bincount(indices, weights=values.imag, minlength=count))


@dataclass(frozen=True)
class PairGeometry:
    modes: np.ndarray
    outputs: np.ndarray
    p: np.ndarray
    q: np.ndarray
    output_index: np.ndarray
    eigenvalues: np.ndarray
    pair_rate: np.ndarray
    output_rate: np.ndarray
    resolved_rates: np.ndarray


def pair_geometry(max_mode=15, cutoff=6, diffusion=.01, growth=.2):
    """Integer, continuous-symbol Galerkin geometry; no circular convolution."""
    if max_mode < 1 or not 0 <= cutoff <= 2 * max_mode or diffusion < 0:
        raise ValueError("Invalid finite spectral-pair geometry")
    modes = np.arange(-max_mode, max_mode + 1)
    outputs = np.arange(-cutoff, cutoff + 1)
    p, q = np.meshgrid(np.arange(len(modes)), np.arange(len(modes)), indexing="ij")
    wave = modes[p] + modes[q]
    keep = abs(wave) <= cutoff
    p, q, wave = p[keep], q[keep], wave[keep]
    eigenvalues = growth - diffusion * modes.astype(float) ** 2
    return PairGeometry(modes, outputs, p, q, wave + cutoff, eigenvalues,
                        eigenvalues[p] + eigenvalues[q], growth - diffusion * wave ** 2,
                        growth - diffusion * outputs.astype(float)**2)


def interaction_kernel(h, output_rate, pair_rate):
    """Integral_0^h exp((h-s)*a+s*b) ds, stably including a=b and h=0."""
    if not np.isfinite(h) or h < 0:
        raise ValueError("A query needs a finite nonnegative horizon")
    a, b = np.broadcast_arrays(np.asarray(output_rate), np.asarray(pair_rate))
    z = h * abs(b - a)
    quotient = np.divide(-np.expm1(-z), z, out=np.ones_like(z, dtype=float), where=z != 0)
    return h * np.exp(h * np.maximum(a, b)) * quotient


@lru_cache(maxsize=8)
def _quadrature_rule(nodes):
    x, w = np.polynomial.legendre.leggauss(nodes)
    return (x + 1) / 2, w / 2


def pair_query(coefficients, geometry, h, *, products=None, active=None, nodes=None):
    """Second-Picard quadratic term; this is not the nonlinear solution."""
    coeff = np.asarray(coefficients, dtype=complex)
    if coeff.shape != geometry.modes.shape:
        raise ValueError("Coefficient field and geometry differ")
    selected = np.ones(len(geometry.p), dtype=bool) if active is None else np.asarray(active, dtype=bool)
    if selected.shape != geometry.p.shape:
        raise ValueError("Pair selector and geometry differ")
    values = coeff[geometry.p[selected]] * coeff[geometry.q[selected]] if products is None else products[selected]
    a, b = geometry.output_rate[selected], geometry.pair_rate[selected]
    if nodes is None:
        kernel = interaction_kernel(h, a, b)
    else:
        s, w = _quadrature_rule(nodes)
        kernel = h * (w @ np.exp(h * ((1 - s[:, None]) * a + s[:, None] * b)))
    return _sum_modes(values * kernel, geometry.output_index[selected], len(geometry.outputs))


def fft_quadrature(coefficients, geometry, h, nodes=4):
    """Comparable FFT opportunity: batched, non-aliased physical-space products."""
    s, w = _quadrature_rule(nodes)
    n = 4 * int(max(abs(geometry.modes))) + 1
    transformed = np.zeros((nodes, n), dtype=complex)
    transformed[:, geometry.modes % n] = coefficients[None, :] * np.exp(h*s[:, None]*geometry.eigenvalues)
    fields = np.fft.ifft(transformed, axis=1, norm="forward")
    square = np.fft.fft(fields*fields, axis=1, norm="forward")[:, geometry.outputs % n]
    return h*np.sum(w[:, None]*np.exp(h*(1-s[:, None])*geometry.resolved_rates)*square, axis=0)


def temporal_basis(geometry, horizon=.2, degree=12):
    """State-independent Chebyshev representation of the exact pair kernel."""
    if horizon <= 0 or degree < 1:
        raise ValueError("A temporal basis needs a positive interval and degree")
    # Fit K(h)/h, not K(h), so the exact h=0 limit is structural.
    x = np.cos(np.pi * (np.arange(degree + 1) + .5) / (degree + 1))
    h = horizon * (x + 1) / 2
    values = np.stack([interaction_kernel(t, geometry.output_rate, geometry.pair_rate) / t for t in h])
    return {"coefficients": chebfit(x, values, degree), "horizon": float(horizon),
            "degree": degree}


def encode_temporal(coefficients, geometry, basis):
    products = coefficients[geometry.p] * coefficients[geometry.q]
    encoded = np.stack([_sum_modes(products * weights, geometry.output_index, len(geometry.outputs))
                        for weights in basis["coefficients"]])
    return {"coefficients": encoded, "horizon": basis["horizon"], "degree": basis["degree"]}


def temporal_query(encoded, h):
    if not np.isfinite(h) or not 0 <= h <= encoded["horizon"]:
        raise ValueError("Temporal query is outside the sealed encoding interval; rebuild the basis")
    return h * chebval(2 * h / encoded["horizon"] - 1, encoded["coefficients"])


def dense_quadratic_solution(coefficients, geometry, horizon):
    """Classical DOP853 dense output for exactly the same lifted quadratic task."""
    n, k = len(geometry.modes), len(geometry.outputs)
    def rhs(t, state):
        y, z = state[:n], state[n:]
        forcing = _sum_modes(y[geometry.p] * y[geometry.q], geometry.output_index, k)
        return np.r_[geometry.eigenvalues * y, geometry.resolved_rates * z + forcing]
    solution = solve_ivp(rhs, (0., horizon), np.r_[coefficients, np.zeros(k, dtype=complex)],
                         method="DOP853", dense_output=True, rtol=1e-11, atol=1e-13)
    if not solution.success:
        raise RuntimeError(f"Classical dense-output construction failed: {solution.message}")
    return solution


def _field(geometry, rng, kind="mixed"):
    m = int(max(abs(geometry.modes)))
    coefficients = np.zeros(len(geometry.modes), dtype=complex)
    coefficients[m] = .1
    for k in range(1, m + 1):
        if kind == "smooth":
            amplitude = .07 * np.exp(-k / 2)
        elif kind == "sparse":
            amplitude = .03 if k in (3, 7, min(m, 13)) else 0.
        else:
            amplitude = .06 / (1 + k) ** rng.uniform(.2, 1.2)
        value = amplitude * np.exp(1j * rng.uniform(-np.pi, np.pi))
        coefficients[m + k], coefficients[m - k] = value, value.conjugate()
    return coefficients


def _paired_timing(calls, repeats, rng, budget):
    """CPU timings, randomized interleaving; first call and warm repeats separate."""
    raw, cold = {name: [] for name in calls}, {}
    for name in rng.permutation(list(calls)):
        budget.check(); start = time.perf_counter(); calls[name](); cold[name] = time.perf_counter() - start
    for round_id in range(repeats):
        for order, name in enumerate(rng.permutation(list(calls))):
            budget.check(); start = time.perf_counter(); calls[name](); seconds = time.perf_counter() - start
            raw[name].append({"round": round_id, "order": order, "seconds": seconds})
    return {name: {"cold_seconds": cold[name], "median_seconds": statistics.median(v["seconds"] for v in rows),
                   "raw": rows, "device": "cpu", "scope": "wall clock, no profiler, first invocation excluded from warm"}
            for name, rows in raw.items()}


def _relative_error(actual, expected):
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), np.finfo(float).tiny))


def _emit(ctx, rows, prototype_id, experiment_id, method, metrics, checks, *, role="Ours", config=None,
          interpretation="", evidence=None, status="COMPLETED"):
    row = dict(schema="tdn.portfolio-prototype/v1", prototype_id=prototype_id,
               experiment_id=experiment_id, method=method, role=role, status=status,
               metrics=_plain(metrics), checks=checks, config=_plain(config or {}),
               interpretation=interpretation, evidence=evidence or [])
    rows.append(row)
    ctx.record(experiment_id, [prototype_id], metrics=row["metrics"], checks=checks,
               config=dict(row["config"], method=method, role=role), evidence=row["evidence"], status=status)
    _write(ctx.path / "prototype_rows.json", rows)


def _temporal(ctx, rows, seed):
    rng = np.random.default_rng(seed)
    geometry = pair_geometry()
    coefficients = _field(geometry, rng)
    horizon, degree, repeats = .2, 12, 9
    quadrature_setup = {}
    for nodes in (2, 4):
        _quadrature_rule.cache_clear()
        start = time.perf_counter(); _quadrature_rule(nodes); quadrature_setup[nodes] = time.perf_counter() - start
    _quadrature_rule(2)
    start = time.perf_counter(); basis = temporal_basis(geometry, horizon, degree); basis_seconds = time.perf_counter() - start
    start = time.perf_counter(); encoded = encode_temporal(coefficients, geometry, basis); encode_seconds = time.perf_counter() - start
    start = time.perf_counter(); products = coefficients[geometry.p] * coefficients[geometry.q]; exact_build_seconds = time.perf_counter() - start
    start = time.perf_counter(); dense = dense_quadratic_solution(coefficients, geometry, horizon); dense_build_seconds = time.perf_counter() - start
    horizons = np.r_[0., np.linspace(.003, horizon, 19)]
    query_rows, maximum_errors = [], {}
    for h in horizons:
        ctx.budget.check()
        answer = pair_query(coefficients, geometry, float(h), products=products)
        methods = {"exact_cached_pairs": lambda: pair_query(coefficients, geometry, h, products=products),
                   "uncached_pairs": lambda: pair_query(coefficients, geometry, h),
                   "compact_temporal_encoding": lambda: temporal_query(encoded, h),
                   "classical_dense_output": lambda: dense.sol(h)[len(coefficients):],
                   "gauss_two": lambda: pair_query(coefficients, geometry, h, nodes=2),
                   "gauss_four": lambda: pair_query(coefficients, geometry, h, nodes=4),
                   "fft_gauss_four": lambda: fft_quadrature(coefficients, geometry, h, nodes=4)}
        timings = _paired_timing(methods, repeats, rng, ctx.budget)
        for method, call in methods.items():
            prediction = call()
            relative = _relative_error(prediction, answer) if h else float(np.linalg.norm(prediction))
            maximum_errors[method] = max(maximum_errors.get(method, 0.), relative)
            query_rows.append(dict(method=method, horizon=float(h), relative_rms=relative,
                                   absolute_max=float(np.max(abs(prediction - answer))),
                                   timing=timings[method], reference="closed_form_pair_integral"))
    direct64 = pair_query(coefficients, geometry, horizon, nodes=64)
    exact = pair_query(coefficients, geometry, horizon)
    absolute_check = float(np.max(abs(exact - direct64)))
    per_query = {method: statistics.median(v["timing"]["median_seconds"] for v in query_rows
                                         if v["method"] == method and v["horizon"] > 0)
                 for method in maximum_errors}
    builds = {"compact_temporal_encoding": basis_seconds + encode_seconds,
              "exact_cached_pairs": exact_build_seconds, "classical_dense_output": dense_build_seconds,
              "uncached_pairs": 0., "gauss_two": quadrature_setup[2], "gauss_four": quadrature_setup[4],
              "fft_gauss_four": quadrature_setup[4]}
    amortization = []
    for count in (1, 2, 4, 8, 16, 32, 64, 128):
        for method in builds:
            amortization.append(dict(method=method, query_count=count,
                                     total_seconds=builds[method] + count * per_query[method],
                                     existing_operator_basis_total_seconds=(encode_seconds if method == "compact_temporal_encoding" else builds[method]) + count*per_query[method],
                                     cost_basis="measured build plus count times median measured query, not a batch benchmark"))
    differences = per_query["exact_cached_pairs"] - per_query["compact_temporal_encoding"]
    break_even = (max(1, math.ceil((builds["compact_temporal_encoding"] - exact_build_seconds) / differences))
                  if differences > 0 else None)
    refresh_break_even = max(1, math.ceil((encode_seconds-exact_build_seconds)/differences)) if differences > 0 else None
    _write(ctx.path / "temporal_query_rows.json", query_rows)
    _write(ctx.path / "temporal_amortization.json", {"rows": amortization, "build_seconds": builds,
           "basis_seconds": basis_seconds, "state_refresh_seconds": encode_seconds,
           "exact_cache_build_seconds": exact_build_seconds, "classical_dense_nfev": dense.nfev,
           "assumptions": "Fixed state, physical parameters, grid, and [0,H]. New state refreshes encoding; changed operator/H rebuilds basis.",
           "independent_horizons": False, "statistical_unit": "one synthetic development field"})
    _write(ctx.path / "temporal_encoding.json", {"basis": basis, "encoding": encoded,
           "memory_bytes": {"basis": basis["coefficients"].nbytes, "encoding": encoded["coefficients"].nbytes,
                            "pair_products": products.nbytes}, "seed": seed})
    for method in maximum_errors:
        checks = [check("same_quadratic_task", absolute_check, THRESHOLDS["X01"]["quadrature_absolute_error"], category="math"),
                  check("relative_query_error", maximum_errors[method], THRESHOLDS["X01"]["decoder_relative_rms"], category="gap")]
        if method == "compact_temporal_encoding":
            checks.append(check("break_even_32_queries", break_even, THRESHOLDS["X01"]["break_even_queries"], category="utility",
                                reason="NA if cached exact pairs are not slower per query; single-state CPU development diagnostic"))
        _emit(ctx, rows, "X01", f"X01/{method}", method,
              {"maximum_relative_rms": maximum_errors[method], "warm_query_seconds": per_query[method],
               "build_seconds": builds[method], "break_even_queries": break_even if method == "compact_temporal_encoding" else None,
               "operator_basis_reused_break_even_queries": refresh_break_even if method == "compact_temporal_encoding" else None,
               "pair_count": len(geometry.p), "degree": degree, "query_count": len(horizons)}, checks,
              role="Ours" if method == "compact_temporal_encoding" else "Theirs" if method == "classical_dense_output" else "Analytic control",
              config={"seed": seed, "max_mode": 15, "cutoff": 6, "horizon": horizon, "degree": degree,
                      "track": "quadratic_Galerkin_response_only", "device": "cpu"},
              interpretation="Encoding is a fixed approximation of a known kernel, not a trained neural PDE solver. Query timings do not establish full-solver advantage.",
              evidence=["temporal_query_rows.json", "temporal_amortization.json", "temporal_encoding.json"])
    return {"prototype": "X01", "break_even_queries": break_even, "maximum_relative_rms": maximum_errors,
            "decision": "ADVANCE_REPEATED_QUERY_PILOT" if break_even is not None and break_even <= 32 and maximum_errors["compact_temporal_encoding"] <= 1e-3
                         else "RETAIN_REPRESENTATION_RESULT; COST_GATE_NOT_MET"}


def twin_coefficients(max_mode, mean, amplitude_a, amplitude_b, phase_a, phase_b, sign=1, high_mode=9):
    if high_mode + 1 > max_mode or high_mode < 3 or sign not in (-1, 1):
        raise ValueError("Twin modes must be unresolved and lie in the Galerkin state")
    result = np.zeros(2 * max_mode + 1, dtype=complex); result[max_mode] = mean
    for k, value in ((high_mode, .5 * amplitude_a * np.exp(1j * phase_a)),
                     (high_mode + 1, .5 * sign * amplitude_b * np.exp(1j * phase_b))):
        result[max_mode + k], result[max_mode - k] = value, value.conjugate()
    return result


def galerkin_solution(coefficients, h, diffusion=.01, reaction=.5):
    """Independent adaptive reference for a finite non-aliased logistic equation."""
    max_mode = (len(coefficients) - 1) // 2
    geometry = pair_geometry(max_mode, max_mode, diffusion, reaction)
    def rhs(t, state):
        square = _sum_modes(state[geometry.p] * state[geometry.q], geometry.output_index, len(state))
        return geometry.eigenvalues * state - reaction * square
    result = solve_ivp(rhs, (0., h), coefficients, method="DOP853", rtol=2e-12, atol=2e-14)
    if not result.success:
        raise RuntimeError(f"Galerkin reference failed: {result.message}")
    return result.y[:, -1]


def galerkin_rk4(coefficients, h, steps, diffusion=.01, reaction=.5):
    """Independent fixed-step cross-check of the finite equation, not its continuum limit."""
    m = (len(coefficients)-1)//2
    geometry = pair_geometry(m, m, diffusion, reaction)
    def rhs(state):
        return geometry.eigenvalues*state - reaction*_sum_modes(
            state[geometry.p]*state[geometry.q], geometry.output_index, len(state))
    state, dt = coefficients.copy(), h/steps
    for _ in range(steps):
        a = rhs(state); b = rhs(state+.5*dt*a); c = rhs(state+.5*dt*b); d = rhs(state+dt*c)
        state += dt*(a+2*b+2*c+d)/6
    return state


def _closure(ctx, rows, seed):
    rng = np.random.default_rng(seed)
    parent_count = 16 if ctx.protocol.get("profile") == "smoke" else 40
    train_count = parent_count // 2
    dt, h, m = .002, .01, 15
    closure_rows = []; reference_seconds = memory_seconds = 0.; reference_crosschecks = []
    for parent in range(parent_count):
        ctx.budget.check()
        mean, a, b = rng.uniform(.25, .65), rng.uniform(.03, .16), rng.uniform(.03, .16)
        phase_a, phase_b = rng.uniform(-np.pi, np.pi, 2)
        for sign in (-1, 1):
            state = twin_coefficients(m, mean, a, b, phase_a, phase_b, sign)
            start = time.perf_counter(); future = galerkin_solution(state, h); reference_seconds += time.perf_counter() - start
            start = time.perf_counter(); previous = galerkin_solution(state, -dt); memory_seconds += time.perf_counter() - start
            if parent < 2:
                for reference_h, reference in ((h, future), (-dt, previous)):
                    start = time.perf_counter()
                    coarse = galerkin_rk4(state, reference_h, 8)
                    fine = galerkin_rk4(state, reference_h, 16)
                    reference_seconds += time.perf_counter()-start
                    reference_crosschecks.append(dict(parent=parent, twin=sign, h=reference_h,
                        rk4_refinement_max=float(np.max(abs(coarse-fine))),
                        dop853_vs_rk4_max=float(np.max(abs(reference-fine)))))
            target, memory = future[m + 1], -previous[m + 1] / dt
            closure_rows.append(dict(parent_id=f"X02-development-{seed}-{parent:03d}", twin=sign,
                split="fit" if parent < train_count else "heldout_development", mean=mean,
                variance=(a*a+b*b)/2, target=[target.real, target.imag],
                memory=[memory.real, memory.imag], resolved_coefficients=state[m-2:m+3].tolist(),
                full_coefficients=state.tolist(),
                oracle_product=[(sign*a*b*np.exp(1j*(phase_b-phase_a))).real, (sign*a*b*np.exp(1j*(phase_b-phase_a))).imag]))
    train = [v for v in closure_rows if v["split"] == "fit"]
    def instantaneous(v):
        return [1., v["mean"], v["mean"]**2, v["variance"], v["mean"]*v["variance"]]
    x = np.array([instantaneous(v) for v in train]); y = np.array([v["target"] for v in train])
    start = time.perf_counter(); instant_weights = np.linalg.lstsq(x, y, rcond=None)[0]
    # Shared coefficients for real/imaginary equations preserve rotation equivariance.
    memory_x = np.array([[component, v["mean"]*component] for v in train for component in v["memory"]])
    memory_y = y.reshape(-1)
    memory_weights = np.linalg.lstsq(memory_x, memory_y, rcond=None)[0]
    fit_seconds = time.perf_counter() - start
    for row in closure_rows:
        memory = np.array(row["memory"])
        row["predictions"] = {"instantaneous_fit": (np.array(instantaneous(row)) @ instant_weights).tolist(),
                              "memory_persistence": (h*memory).tolist(),
                              "memory_fitted": ((memory_weights[0] + row["mean"]*memory_weights[1])*memory).tolist(),
                              "full_state_instantaneous_oracle": (-.5*.5*h*np.array(row["oracle_product"])).tolist()}
        row["errors"] = {method: float(np.linalg.norm(np.array(prediction)-row["target"]))
                         for method, prediction in row["predictions"].items()}
    heldout = [v for v in closure_rows if v["split"] != "fit"]
    methods = list(heldout[0]["predictions"])
    rms = {method: float(np.sqrt(np.mean([v["errors"][method]**2 for v in heldout]))) for method in methods}
    twin_gaps, lowerbounds, coarse_diffs = [], [], []
    for index in range(0, len(heldout), 2):
        a, b = heldout[index:index+2]
        twin_gaps.append(float(np.linalg.norm(np.array(a["target"])-b["target"])))
        lowerbounds.append(twin_gaps[-1]/2)
        coarse_diffs.append(float(np.max(abs(np.array(a["resolved_coefficients"])-b["resolved_coefficients"])) ))
    sample = heldout[0]
    timings = _paired_timing({"instantaneous_fit": lambda: np.array(instantaneous(sample)) @ instant_weights,
        "memory_persistence": lambda: h*np.array(sample["memory"]),
        "memory_fitted": lambda: (memory_weights[0]+sample["mean"]*memory_weights[1])*np.array(sample["memory"])},
        15, rng, ctx.budget)
    _write(ctx.path / "closure_rows.json", closure_rows)
    _write(ctx.path / "closure_fit.json", {"instantaneous_weights": instant_weights, "memory_weights": memory_weights,
           "fit_seconds": fit_seconds, "reference_generation_seconds": reference_seconds,
           "history_acquisition_seconds": memory_seconds, "history_acquisition_per_state_seconds": memory_seconds/len(closure_rows),
           "timings": timings, "horizon": h, "history_interval": dt, "independent_fit_parents": train_count,
           "reference_crosschecks": reference_crosschecks,
           "independent_heldout_parents": parent_count-train_count,
           "lower_bound_rms_any_deterministic_instantaneous_predictor": float(np.sqrt(np.mean(np.square(lowerbounds)))),
           "scope": "Synthetic twin states, short smooth backward histories; future task is discrete Galerkin, not continuum. Not a general memory closure."})
    for method in methods:
        ratio = rms[method]/max(rms["instantaneous_fit"], np.finfo(float).tiny)
        _emit(ctx, rows, "X02", f"X02/{method}", method,
              {"heldout_rms": rms[method], "relative_to_instantaneous_fit": ratio,
               "minimum_twin_target_separation": min(twin_gaps), "resolved_twin_difference": max(coarse_diffs),
               "heldout_parents": parent_count-train_count, "fit_parents": train_count,
               "warm_query_seconds": timings.get(method, {}).get("median_seconds"),
               "history_acquisition_per_state_seconds": memory_seconds/len(closure_rows)},
              [check("identical_resolved_information", max(coarse_diffs), 1e-12, category="math"),
               check("different_future_targets", min(twin_gaps), 1e-8, relation="gt", category="math"),
               check("independent_reference_crosscheck", max(v["dop853_vs_rk4_max"] for v in reference_crosschecks), 1e-10,
                     category="correctness", reason="Fixed-step RK4 8/16 refinement and adaptive DOP853 on a declared four-state subset"),
               check("memory_improves_heldout_rmse", ratio, .5, category="gap",
                     applicable=method in ("memory_persistence", "memory_fitted"),
                     reason="Instantaneous and full-state oracle controls test information, not the memory utility claim")],
              role="Ours" if method == "memory_fitted" else "Analytic control",
              config={"seed": seed, "dt": dt, "horizon": h, "max_mode": m, "resolved_cutoff": 2,
                      "split_unit": "independent twin parent", "target": "first complex resolved Fourier coefficient"},
              interpretation="Memory reveals information absent from identical resolved states. History is measured from backward-generated short trajectories; acquisition is separately charged. No universal closure or solver speed claim.",
              evidence=["closure_rows.json", "closure_fit.json"])
    return {"prototype": "X02", "heldout_rms": rms,
            "decision": "ADVANCE_CAUSAL_HISTORY_PILOT" if rms["memory_fitted"] <= .5*rms["instantaneous_fit"] else "STOP_CURRENT_MEMORY_FEATURE"}


def selection_features(coefficients, geometry):
    magnitudes = abs(coefficients); energy = magnitudes**2; total = max(float(energy.sum()), 1e-30)
    probabilities = energy / total
    return np.array([1., float(-np.sum(probabilities*np.log(np.maximum(probabilities, 1e-30)))) / np.log(len(energy)),
                     float(np.dot(energy, abs(geometry.modes))) / total / max(abs(geometry.modes)),
                     float(magnitudes.sum()**2 / total) / len(energy)])


def select_modes(coefficients, geometry, rank):
    """Select conjugate pairs together, preserving real-valued outputs."""
    m = max(abs(geometry.modes)); energy = abs(coefficients)**2
    scores = energy[m+1:] + energy[:m][::-1]
    order = np.argsort(-scores, kind="stable")
    chosen = np.zeros(len(coefficients), dtype=bool); chosen[m] = True
    k = order[:int(rank)]+1; chosen[m+k] = True; chosen[m-k] = True
    return chosen[geometry.p] & chosen[geometry.q]


def heuristic_rank(coefficients, tolerance=.01):
    """L1 tail control before forming pair products; absolute kernel-bound proxy."""
    m = (len(coefficients)-1)//2
    masses = abs(coefficients[m+1:]) + abs(coefficients[:m][::-1])
    sums = abs(coefficients[m]) + np.r_[0., np.cumsum(np.sort(masses)[::-1])]
    residual_fraction = 1 - sums**2/max(sums[-1]**2, 1e-30)
    return max(1, int(np.flatnonzero(residual_fraction <= tolerance)[0]))


def predicted_rank(coefficients, geometry, weights, candidates):
    estimate = float(selection_features(coefficients, geometry) @ weights)
    return int(next((r for r in candidates if r >= estimate), candidates[-1]))


def _selection(ctx, rows, seed):
    rng = np.random.default_rng(seed); geometry = pair_geometry(63, 8, .004, .2)
    candidates, h, tolerance = [2, 4, 8, 16, 32, 63], .03, THRESHOLDS["X03"]["relative_error"]
    count = 18 if ctx.protocol.get("profile") == "smoke" else 48
    states = []
    for parent in range(count):
        ctx.budget.check(); regime = ("smooth", "sparse", "rough")[parent % 3]
        coefficient = _field(geometry, rng, regime)
        reference = pair_query(coefficient, geometry, h)
        errors = {rank: _relative_error(pair_query(coefficient, geometry, h, active=select_modes(coefficient, geometry, rank)), reference)
                  for rank in candidates}
        oracle_rank = next(rank for rank in candidates if errors[rank] <= tolerance)
        states.append(dict(parent_id=f"X03-development-{seed}-{parent:03d}", regime=regime,
                           split="fit" if parent < count//2 else "heldout_development",
                           coefficients=coefficient, reference=reference, errors=errors, oracle_rank=oracle_rank))
    fit = [v for v in states if v["split"] == "fit"]
    start = time.perf_counter()
    weights = np.linalg.lstsq(np.array([selection_features(v["coefficients"], geometry) for v in fit]),
                              np.array([v["oracle_rank"] for v in fit]), rcond=None)[0]
    fit_seconds = time.perf_counter()-start
    selection_rows = []
    for state in states:
        ctx.budget.check(); coefficients, reference = state["coefficients"], state["reference"]
        selectors = {"deterministic_l1_tail": lambda: heuristic_rank(coefficients, tolerance),
                     "fitted_rank_rule": lambda: predicted_rank(coefficients, geometry, weights, candidates)}
        ranks = {name: call() for name, call in selectors.items()}
        masks = {name: select_modes(coefficients, geometry, rank) for name, rank in ranks.items()}
        calls = {"full_pairs": lambda: pair_query(coefficients, geometry, h)}
        for method, selector in selectors.items():
            def selected_call(selector=selector):
                active = select_modes(coefficients, geometry, selector())
                return pair_query(coefficients, geometry, h, active=active)
            calls[method] = selected_call
        timing = _paired_timing(calls, 9, rng, ctx.budget)
        selector_timing = _paired_timing(selectors, 9, rng, ctx.budget)
        for method in calls:
            mask = masks.get(method, np.ones(len(geometry.p), dtype=bool))
            prediction = calls[method]()
            selection_rows.append(dict(parent_id=state["parent_id"], split=state["split"], regime=state["regime"],
                 method=method, rank=ranks.get(method, 63), oracle_rank=state["oracle_rank"],
                 pair_fraction=float(mask.mean()), pair_count=int(mask.sum()), full_pair_count=len(mask),
                 relative_error=_relative_error(prediction, reference), timing=timing[method],
                 selection_timing=selector_timing.get(method),
                 speedup=timing["full_pairs"]["median_seconds"]/timing[method]["median_seconds"],
                 scope="End-to-end includes magnitude features, sorting/prediction, mask, products, exponentials and summation; full pairs get the same cached geometry"))
    _write(ctx.path / "selection_rows.json", selection_rows)
    _write(ctx.path / "selection_fit.json", {"weights": weights, "fit_seconds": fit_seconds,
           "features": ["bias", "spectral_entropy", "energy_weighted_frequency", "l1_squared_over_l2_squared"],
           "candidate_positive_mode_counts": candidates, "tolerance": tolerance, "seed": seed,
           "fit_parents": len(fit), "heldout_parents": count-len(fit),
           "target": "minimum candidate rank meeting development relative error; ground truth is never consulted by deployed selector",
           "limitation": "Least squares is not a certified tolerance controller. Absolute L1-tail proxy is not a relative error guarantee under cancellation."})
    _write(ctx.path / "selection_states.json", [dict(v, features=selection_features(v["coefficients"], geometry)) for v in states])
    # Separate correlated phase audit: never pooled into independent-field means.
    phase_rng = np.random.default_rng(seed+7000)
    anchor = _field(geometry, phase_rng, "rough"); midpoint = len(anchor)//2
    phase_rows = []
    for phase_id in range(8):
        state = anchor.copy()
        phases = np.zeros(midpoint) if phase_id == 0 else phase_rng.uniform(-np.pi, np.pi, midpoint)
        state[midpoint+1:] = abs(state[midpoint+1:])*np.exp(1j*phases)
        state[:midpoint] = state[midpoint+1:][::-1].conj()
        reference = pair_query(state, geometry, h)
        ranks = {"deterministic_l1_tail": heuristic_rank(state, tolerance),
                 "fitted_rank_rule": predicted_rank(state, geometry, weights, candidates)}
        phase_rows.append(dict(phase_id=phase_id, parent_id=f"X03-phase-audit-{seed+7000}",
            split="heldout_development_phase_audit", features=selection_features(state, geometry),
            coefficients=state, ranks=ranks, errors={method: _relative_error(
                pair_query(state, geometry, h, active=select_modes(state, geometry, rank)), reference)
                for method, rank in ranks.items()}))
    _write(ctx.path / "selection_phase_audit.json", phase_rows)
    feature_difference = float(np.max(np.ptp(np.array([v["features"] for v in phase_rows]), axis=0)))
    phase_ranges = {method: {"min_error": min(v["errors"][method] for v in phase_rows),
                            "max_error": max(v["errors"][method] for v in phase_rows)} for method in ranks}
    decisions = {}
    for method in calls:
        subset = [v for v in selection_rows if v["split"] != "fit" and v["method"] == method]
        max_error = max(v["relative_error"] for v in subset)
        speedup = float(np.exp(np.mean(np.log([v["speedup"] for v in subset]))))
        violations = sum(v["relative_error"] > tolerance for v in subset)
        checks = [check("real_conjugate_selection", max(float(np.max(abs(pair_query(state["coefficients"], geometry, h,
                         active=select_modes(state["coefficients"], geometry, 8)) -
                         pair_query(state["coefficients"], geometry, h, active=select_modes(state["coefficients"], geometry, 8))[::-1].conj())))
                         for state in states[:2]), 1e-12, category="math"),
                  check("heldout_relative_error", max_error, tolerance, category="gap"),
                  check("complete_selection_speedup", speedup, 1.10, relation="ge", category="utility",
                        applicable=method != "full_pairs", reason="CPU only; all selection work charged, no matched-accuracy full solver claim")]
        _emit(ctx, rows, "X03", f"X03/{method}", method,
              {"max_heldout_relative_error": max_error, "geometric_mean_speedup": speedup,
               "violations": violations, "heldout_parents": len(subset),
               "mean_pair_fraction": float(np.mean([v["pair_fraction"] for v in subset])),
               "warm_query_seconds": float(np.median([v["timing"]["median_seconds"] for v in subset]))}, checks,
              role="Ours" if method == "fitted_rank_rule" else "Analytic control",
              config={"seed": seed, "max_mode": 63, "output_cutoff": 8, "horizon": h,
                      "relative_tolerance": tolerance, "fit_parents": len(fit)},
              interpretation="Work is selected before complex pair products and transport kernels. Selection overhead and cancellation can eliminate gains; failures are retained.",
              evidence=["selection_rows.json", "selection_fit.json", "selection_states.json"])
        decisions[method] = ("ANALYTIC_REFERENCE_ONLY" if method == "full_pairs" else
                             "ADVANCE" if violations == 0 and speedup >= 1.10 else "STOP_CURRENT_SELECTOR")
    _emit(ctx, rows, "X03", "X03/phase_information_audit", "phase_information_audit",
          {"feature_difference": feature_difference, "phase_error_ranges": phase_ranges,
           "independent_amplitude_parents": 1, "paired_phase_variants": 8},
          [check("phase_blind_features", feature_difference, 1e-12, category="math"),
           check("phase_invariance_is_not_accuracy", None, None, category="gap",
                 reason="Diagnostic observation without a predeclared gap-success threshold; not pooled with independent fields")],
          role="Analytic control", interpretation="Same magnitude features across phase variants do not encode cancellation. Inspect per-phase errors; no universal predictor-impossibility claim follows from this one amplitude parent.",
          evidence=["selection_phase_audit.json"])
    return {"prototype": "X03", "decision": decisions}


def run(ctx):
    """Run only registered prototypes for this independently budgeted CPU unit."""
    if str(ctx.device) != "cpu":
        raise ValueError("Protected first-stage prototypes explicitly measure CPU; do not mislabel them GPU experiments")
    unit = ctx.protocol.get("units", {}).get(ctx.stage, {})
    selected = unit.get("prototype_ids", ["X01", "X02", "X03"])
    if not selected or any(name not in THRESHOLDS for name in selected) or len(set(selected)) != len(selected):
        raise ValueError("Prototype selection must contain distinct registered X01/X02/X03 identities")
    ctx.path.mkdir(parents=True, exist_ok=True)
    fallback_seed = {"smoke": 54_100_000, "development": 54_200_000, "full": 54_300_000}.get(ctx.protocol.get("profile"), 54_100_000)
    base_seed = ctx.protocol.get("exploration", {}).get("seed", fallback_seed)
    declared = ctx.protocol.get("exploration", {})
    expected = {"X01": {"parity_tolerance": 1e-10, "relative_rmse": .001, "break_even_queries": 32},
                "X02": {"resolved_twin_tolerance": 1e-12, "target_separation": 1e-8, "memory_rmse_ratio": .5},
                "X03": {"relative_error": .01, "total_speedup": 1.10}}
    if any(name in declared and declared[name] != expected[name] for name in selected):
        raise ValueError("Declared prototype thresholds differ from this frozen implementation")
    _write(ctx.path / "prototype_registration.json", {"thresholds": THRESHOLDS, "prototype_ids": selected,
           "seeds": {name: base_seed+int(name[1:])*100 for name in selected}, "device": "cpu",
           "data_scope": "synthetic development only; no confirmation fields or prior diagnostic fields",
           "novelty": "new to TDN prototypes; no literature novelty claim", "protected_path": "C"})
    rows, outcomes = [], []
    for name in selected:
        ctx.budget.check()
        try:
            outcomes.append({"X01": _temporal, "X02": _closure, "X03": _selection}[name](ctx, rows, base_seed+int(name[1:])*100))
        except (TimeoutError, KeyboardInterrupt):
            _write(ctx.path / "prototype_summary.json", {"status": "INTERRUPTED", "outcomes": outcomes, "rows": len(rows)})
            raise
        except (ArithmeticError, ValueError, RuntimeError) as error:
            _emit(ctx, rows, name, f"{name}/failure", "prototype_execution", {"error": f"{type(error).__name__}: {error}"},
                  [check("prototype_math", None, None, category="math", reason=str(error)),
                   check("prototype_gap", None, None, category="gap", reason="Execution did not supply evidence")], status="FAILED",
                  interpretation="Explicit prototype failure retained; no positive scientific conclusion")
            outcomes.append({"prototype": name, "decision": "FAILED_IMPLEMENTATION_OR_NUMERICS", "error": str(error)})
    summary = {"status": "COMPLETED" if all(v.get("decision") != "FAILED_IMPLEMENTATION_OR_NUMERICS" for v in outcomes) else "COMPLETED_WITH_FAILURES",
               "outcomes": outcomes, "rows": len(rows), "scientific_scope": "CPU development diagnostics, not confirmation"}
    _write(ctx.path / "prototype_summary.json", summary)
    return summary
