"""Three bounded formulation probes on a non-aliased 1D Galerkin equation.

These are development experiments, not 2D PDE confirmation.  All histories are
generated forwards, model selection sees validation parents only, and numerical
correctness is scored separately from measured utility.  Standard rational
functions and variational equations are controls, not novelty claims.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np
from scipy.integrate import solve_ivp

from tdn.analysis.frontier.core import check, score_checks


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, complex):
        return [value.real, value.imag]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write(path, value):
    Path(path).write_text(json.dumps(_plain(value), indent=2, allow_nan=False) + "\n")


def _emit(ctx, rows, prototype, identity, method, metrics, checks, *, role="Ours", config=None):
    row = dict(schema="tdn.advance-prototype/v1", prototype_id=prototype,
               experiment_id=identity, method=method, role=role, metrics=_plain(metrics),
               checks=checks, assessment=score_checks(checks), config=_plain(config or {}),
               evidence=[f"{prototype}_{ {'XH': 'history', 'XR': 'resolvent', 'XT': 'tangent'}[prototype]}.json"],
               scope="bounded 1D Galerkin development; independent parent is the statistical unit")
    rows.append(row)
    ctx.record(identity, [prototype], metrics=row["metrics"], checks=checks,
               config=dict(row["config"], method=method, role=role), evidence=row["evidence"])
    _write(ctx.path / "prototype_rows.json", rows)


def convolution(a, b):
    """Exact finite convolution projected to the original symmetric modes."""
    a, b = np.asarray(a, complex), np.asarray(b, complex)
    if a.shape != b.shape or a.ndim != 1 or len(a) % 2 != 1:
        raise ValueError("Galerkin fields need matching odd one-dimensional mode arrays")
    k = len(a) // 2
    return np.convolve(a, b)[k:3*k+1]


def rhs(a, diffusion=.02, reaction=.7):
    k = len(a) // 2
    modes = np.arange(-k, k+1)
    return (reaction-diffusion*modes*modes)*a - reaction*convolution(a, a)


def tangent_rhs(a, v, diffusion=.02, reaction=.7):
    k = len(a) // 2
    modes = np.arange(-k, k+1)
    return (reaction-diffusion*modes*modes)*v - 2*reaction*convolution(a, v)


def project(a, cutoff):
    k = len(a) // 2
    if not 0 <= cutoff <= k:
        raise ValueError("Projection cutoff exceeds the field")
    return np.array(a[k-cutoff:k+cutoff+1], copy=True)


def real_coordinates(a):
    """Nonredundant coordinates for conjugate-symmetric coefficients."""
    k = len(a) // 2
    return np.r_[a[k].real, a[k+1:].real, a[k+1:].imag]


def complex_coordinates(x):
    x = np.asarray(x, float)
    if x.ndim != 1 or len(x) % 2 != 1:
        raise ValueError("Real spectral coordinates need an odd vector")
    k = len(x) // 2
    positive = x[1:k+1] + 1j*x[k+1:]
    return np.r_[positive[::-1].conj(), x[0]+0j, positive]


def field(seed, max_mode=8):
    """Positive smooth fields with independently randomized unresolved phase."""
    rng = np.random.default_rng(seed)
    a = np.zeros(2*max_mode+1, complex)
    a[max_mode] = rng.uniform(.3, .6)
    for k in range(1, max_mode+1):
        amplitude = rng.uniform(.003, .016) / (1+.1*k)
        value = amplitude*np.exp(1j*rng.uniform(-np.pi, np.pi))
        a[max_mode+k], a[max_mode-k] = value, value.conjugate()
    return a


def integrate(a, horizon, *, diffusion=.02, reaction=.7, tangent=None, rtol=2e-11, atol=2e-13):
    """DOP853 on an explicitly non-aliased finite ODE; never backward history."""
    if not np.isfinite(horizon) or horizon <= 0:
        raise ValueError("Only positive forward integration horizons are allowed")
    n = len(a)
    initial = a if tangent is None else np.r_[a, tangent]
    def fun(t, y):
        base = rhs(y[:n], diffusion, reaction)
        return base if tangent is None else np.r_[base, tangent_rhs(y[:n], y[n:], diffusion, reaction)]
    start = time.perf_counter()
    solution = solve_ivp(fun, (0., horizon), initial, method="DOP853", dense_output=True,
                         rtol=rtol, atol=atol)
    elapsed = time.perf_counter()-start
    if not solution.success or not np.isfinite(solution.y).all():
        raise RuntimeError(f"Forward Galerkin reference failed: {solution.message}")
    return solution, elapsed


def rk4(a, h, diffusion=.02, reaction=.7):
    k1 = rhs(a, diffusion, reaction)
    k2 = rhs(a+h*k1/2, diffusion, reaction)
    k3 = rhs(a+h*k2/2, diffusion, reaction)
    k4 = rhs(a+h*k3, diffusion, reaction)
    return a+h*(k1+2*k2+2*k3+k4)/6


def error_metrics(a, reference):
    error = a-reference
    k = len(a)//2
    # Parseval RMS on [0,2pi]; oversampled maximum is an estimate, not a bound.
    decoder = np.exp(1j*np.outer(np.arange(257)*2*np.pi/257, np.arange(-k, k+1)))
    physical = decoder @ error
    state = (decoder @ a).real
    return dict(error_rms=float(np.linalg.norm(error)), error_max=float(np.max(abs(physical))),
                mean_error=float(abs(error[k])), finite=bool(np.isfinite(a).all()),
                imaginary_residual=float(np.max(abs(a-a[::-1].conj()))),
                minimum_state=float(np.min(state)), maximum_state=float(np.max(state)),
                observed_interval_violation=float(max(0., -np.min(state), np.max(state)-1.)))


def rk4_reference(a, horizon, diffusion=.02, reaction=.7, steps=128):
    """Independent fixed-step ODE implementation, not a continuum certificate."""
    tick = time.perf_counter()
    state = a.copy()
    for _ in range(steps):
        state = rk4(state, horizon/steps, diffusion, reaction)
    return state, time.perf_counter()-tick


def _ridge(x, y, ridge):
    x, y = np.asarray(x), np.asarray(y)
    scale = np.maximum(np.sqrt(np.mean(x*x, axis=0)), 1e-10)
    z = x/scale
    # SVD regularization avoids normal-equation conditioning loss.
    u, singular, vt = np.linalg.svd(z, full_matrices=False)
    weights = (vt.T*(singular/(singular*singular+ridge))) @ (u.T@y)
    return dict(scale=scale, weights=weights, singular_values=singular, ridge=ridge)


def _predict(model, x):
    return (np.asarray(x)/model["scale"]) @ model["weights"]


def _features(current, previous, h, diffusion, reaction, memory):
    x = real_coordinates(current)
    base = np.r_[1., x, x*x]
    if memory:
        # Known previous coarse state and step; no future teacher is consulted.
        secant = real_coordinates((current-rk4(previous, h, diffusion, reaction))/h)
        base = np.r_[base, secant]
    return base


def _select_model(training, validation, memory):
    options = []
    for ridge in (1e-10, 1e-6, 1e-2):
        key = "features_memory" if memory else "features_instant"
        model = _ridge([v[key] for v in training], [v["target"] for v in training], ridge)
        loss = float(np.mean([np.mean((_predict(model, v[key])-v["target"])**2) for v in validation]))
        options.append((loss, ridge, model))
    _, _, selected = min(options, key=lambda v: (v[0], v[1]))
    return selected, [{"ridge": ridge, "validation_mse": loss} for loss, ridge, _ in options]


def _history(ctx, rows, settings):
    seed = settings["seed"]
    cutoff, max_mode, h, start_time = 2, settings["max_mode"], .01, .1
    diffusion, reaction = .02, .7
    counts = settings["history_counts"]
    splits = {name: [seed+offset+i for i in range(count)]
              for name, count, offset in zip(("train", "validation", "test"), counts, (0, 1000, 2000))}
    raw = dict(schema="tdn.advance-history/v1", settings=settings, parent_ids=splits,
               rows=[], training_rows=[], selection={}, teacher_records=[],
               target="projected fine Galerkin endpoint minus coarse RK4, divided by h",
               startup="forward teacher observations at t=0.09 and 0.10; startup cost charged once per parent/scenario",
               limitations=["1D finite-mode development, not continuum convergence",
                            "maximum error uses 257 evaluation points", "test parents are development, not confirmation"])
    training, validation = [], []
    for split, bank in (("train", training), ("validation", validation)):
        for parent in splits[split]:
            ctx.budget.check()
            sol, seconds = integrate(field(parent, max_mode), .3, diffusion=diffusion, reaction=reaction)
            raw["teacher_records"].append(dict(teacher_id=f"XH/{split}/{parent}", parent_id=parent,
                                              split=split, seconds=seconds, nfev=sol.nfev))
            for t in (.10, .16, .22):
                previous, current, future = [project(sol.sol(t+shift), cutoff) for shift in (-h, 0, h)]
                example = dict(parent_id=parent, split=split, observed_time=t,
                    history_time=t-h, target_time=t+h,
                    target=real_coordinates((future-rk4(current, h, diffusion, reaction))/h),
                    features_instant=_features(current, previous, h, diffusion, reaction, False),
                    features_memory=_features(current, previous, h, diffusion, reaction, True))
                bank.append(example)
    raw["training_rows"] = training+validation
    models = {}
    training_start = time.perf_counter()
    for memory, name in ((False, "instantaneous_fit"), (True, "fitted_memory")):
        models[name], raw["selection"][name] = _select_model(training, validation, memory)
    fit_seconds = time.perf_counter()-training_start
    raw["models"] = models
    raw["fit_and_validation_seconds"] = fit_seconds
    scenarios = [("normal", h, diffusion, 0.), ("step_shift", 2*h, diffusion, 0.),
                 ("coefficient_shift", h, 3*diffusion, 0.)]
    scenarios += [(f"history_noise_{noise:g}", h, diffusion, noise) for noise in settings["noise_levels"] if noise > 0]
    if settings["full"]:
        scenarios += [("combined_shift", 2*h, 3*diffusion, 1e-4)]
    methods = ("coarse_rk4", "state_persistence", "instantaneous_fit", "secant_memory", "fitted_memory")
    refinement_errors = []
    for scenario, step, kappa, noise in scenarios:
        for parent in splits["test"]:
            ctx.budget.check()
            initial = field(parent, max_mode)
            horizon = start_time+step*settings["rollout_steps"]
            sol, teacher_seconds = integrate(initial, horizon, diffusion=kappa, reaction=reaction)
            startup_sol, startup_seconds = integrate(initial, start_time, diffusion=kappa, reaction=reaction)
            # Independent tighter solve cross-check on every test trajectory.
            refined, refine_seconds = integrate(initial, horizon, diffusion=kappa, reaction=reaction,
                                                  rtol=2e-13, atol=2e-15)
            uncertainty = float(np.max(abs(sol.sol(np.linspace(0, horizon, 11))-refined.sol(np.linspace(0, horizon, 11)))))
            independent, independent_seconds = rk4_reference(initial, horizon, kappa, reaction)
            independent_error = float(np.max(abs(independent-sol.y[:, -1])))
            uncertainty = max(uncertainty, independent_error)
            refinement_errors.append(uncertainty)
            raw["teacher_records"].append(dict(teacher_id=f"XH/{parent}/{scenario}", parent_id=parent, split="test", scenario=scenario,
                seconds=teacher_seconds, refinement_seconds=refine_seconds, startup_seconds=startup_seconds,
                independent_rk4_seconds=independent_seconds, independent_rk4_steps=128,
                independent_rk4_endpoint_error=independent_error,
                max_coefficient_refinement_error=uncertainty, nfev=sol.nfev))
            previous = project(startup_sol.sol(start_time-step), cutoff)
            current = project(startup_sol.sol(start_time), cutoff)
            rng = np.random.default_rng(parent+int(noise*1e8))
            # Noisy causal observations apply equally to every method; neither
            # a noise-free current state nor a fresh teacher is injected later.
            previous += complex_coordinates(rng.normal(0., noise, len(previous)))
            current += complex_coordinates(rng.normal(0., noise, len(current)))
            for method in methods:
                prev, state, cumulative_seconds = previous.copy(), current.copy(), 0.
                for step_id in range(1, settings["rollout_steps"]+1):
                    tick = time.perf_counter()
                    if method == "state_persistence":
                        prediction = state.copy()
                    else:
                        base = rk4(state, step, kappa, reaction)
                        if method == "coarse_rk4":
                            prediction = base
                        elif method == "secant_memory":
                            prediction = base + (state-rk4(prev, step, kappa, reaction))
                        else:
                            features = _features(state, prev, step, kappa, reaction, method == "fitted_memory")
                            prediction = base + step*complex_coordinates(_predict(models[method], features))
                    elapsed = time.perf_counter()-tick
                    cumulative_seconds += elapsed
                    target = project(sol.sol(start_time+step_id*step), cutoff)
                    raw["rows"].append(dict(parent_id=parent, split="test", scenario=scenario,
                        method=method, rollout_step=step_id, h=step, diffusion=kappa, history_noise=noise,
                        trajectory_id=f"XH/{parent}/{scenario}/{method}", teacher_id=f"XH/{parent}/{scenario}",
                        time=start_time+step_id*step, **error_metrics(prediction, target),
                        inference_seconds=elapsed, cumulative_inference_seconds=cumulative_seconds,
                        startup_seconds=startup_seconds, complete_seconds=startup_seconds+cumulative_seconds,
                        reference_uncertainty=uncertainty, teacher_seconds=teacher_seconds,
                        information="causal coarse observations then autonomous predictions; no future teacher"))
                    prev, state = state, prediction
    raw["fitting_cost_scope"] = "All ridge candidates and validation, excluding separately itemized teachers"
    raw["cost_aggregation"] = "Sum unique teacher_records once; startup and cumulative times repeat across rows, not additive"
    _write(ctx.path/"XH_history.json", raw)
    for scenario, _, _, _ in scenarios:
        final = [v for v in raw["rows"] if v["scenario"] == scenario and v["rollout_step"] == settings["rollout_steps"]]
        base = {v["parent_id"]: v for v in final if v["method"] == "coarse_rk4"}
        deterministic = {v["parent_id"]: v for v in final if v["method"] == "secant_memory"}
        for method in methods:
            selected = [v for v in final if v["method"] == method]
            ratio = statistics.median(v["error_rms"]/max(base[v["parent_id"]]["error_rms"], 1e-15) for v in selected)
            memory_ratio = statistics.median(v["error_rms"]/max(deterministic[v["parent_id"]]["error_rms"], 1e-15) for v in selected)
            metrics = dict(independent_parents=len(selected), error_rms=statistics.median(v["error_rms"] for v in selected),
                error_max=max(v["error_max"] for v in selected), ratio_to_coarse_rk4=ratio,
                ratio_to_deterministic_memory=memory_ratio, mean_error=statistics.median(v["mean_error"] for v in selected),
                startup_seconds=statistics.median(v["startup_seconds"] for v in selected),
                inference_seconds=statistics.median(v["cumulative_inference_seconds"] for v in selected),
                complete_seconds=statistics.median(v["complete_seconds"] for v in selected),
                fit_and_validation_seconds=fit_seconds if method in models else 0.,
                parameters=int(models[method]["weights"].size) if method in models else 0)
            _emit(ctx, rows, "XH", f"XH/{scenario}/{method}", method, metrics, [
                check("reference-refinement", max(refinement_errors), 1e-9, category="math"),
                check("finite-autonomous-rollout", all(v["finite"] for v in selected), True, "eq", category="math"),
                check("twenty-percent-error-reduction", ratio, .8, category="gap",
                      applicable=method != "coarse_rk4",
                      reason="The baseline cannot establish improvement over itself" if method == "coarse_rk4" else ""),
                check("beats-deterministic-memory", memory_ratio, .9, category="gap",
                      applicable=method == "fitted_memory", required=method == "fitted_memory"),
                check("complete-cost-advantage", None, None, category="utility",
                      reason="Unequal information: causal history startup is charged, but no deployment workload is established")],
                role="Ours" if method == "fitted_memory" else "Analytic control" if method != "instantaneous_fit" else "Fitted control",
                config=dict(scenario=scenario, rollout_steps=settings["rollout_steps"], cutoff=cutoff, max_mode=max_mode))
    return raw


def rational_response(z, method):
    """E(z), phi(z) with E=1+z*phi; all use the exact h=0 identity.

    The Padé (1,1) response is A-stable but not L-stable.  Its negative stiff
    limit is -1 and is deliberately retained as a falsification case.
    """
    z = np.asarray(z, float)
    if not np.isfinite(z).all() or np.any(z > 0):
        raise ValueError("These probes split diffusion only and require finite z <= 0")
    if method == "exponential_etd1":
        exponential = np.exp(z)
        phi = np.divide(np.expm1(z), z, out=np.ones_like(z), where=z != 0)
    elif method == "resolvent_backward_euler":
        exponential, phi = 1/(1-z), 1/(1-z)
    elif method == "pade_11":
        phi = 1/(1-z/2)
        exponential = (1+z/2)*phi
    elif method == "pade_02":
        denominator = 1-z+z*z/2
        exponential, phi = 1/denominator, (1-z/2)/denominator
    else:
        raise ValueError(f"Unknown response {method}")
    return exponential, phi


def etd_step(a, h, diffusion, reaction, method, cached=None):
    if h < 0 or not np.isfinite(h):
        raise ValueError("Only nonnegative ETD steps are supported")
    modes = np.arange(-(len(a)//2), len(a)//2+1)
    e, phi = rational_response(-diffusion*modes*modes*h, method) if cached is None else cached
    return e*a+h*phi*reaction*(a-convolution(a, a))


def _paired_cpu(calls, repeats, rng, budget):
    values, first = {name: [] for name in calls}, {}
    for name in rng.permutation(list(calls)):
        budget.check(); tick = time.perf_counter(); calls[name](); first[name] = time.perf_counter()-tick
    for round_id in range(repeats):
        for order, name in enumerate(rng.permutation(list(calls))):
            budget.check(); tick = time.perf_counter(); calls[name](); elapsed = time.perf_counter()-tick
            values[name].append(dict(round=round_id, order=order, seconds=elapsed))
    return {name: dict(first_invocation_seconds=first[name], median_seconds=statistics.median(v["seconds"] for v in data),
                       rounds=data, repeats=repeats, scope="CPU wall clock; first invocation is not process-cold latency")
            for name, data in values.items()}


def _resolvent(ctx, rows, settings):
    methods = ("exponential_etd1", "resolvent_backward_euler", "pade_11", "pade_02")
    parents = [settings["seed"]+4000+i for i in range(settings["test_parents"])]
    raw = dict(schema="tdn.advance-resolvent/v1", settings=settings, parent_ids=parents,
               rows=[], response_rows=[], timing_rows=[], teacher_records=[],
               reference="forward nonlinear non-aliased Galerkin DOP853; diffusion-only ETD decomposition",
               cost_aggregation="Sum unique teacher_records once; per-method rows repeat teacher cost by teacher_id",
               novelty="standard rational/ETD controls, new-to-project formulation probe only")
    z = np.r_[0., -np.logspace(-10, 6, 81)]
    for method in methods:
        e, phi = rational_response(z, method)
        raw["response_rows"].extend(dict(method=method, z=float(zi), exponential=float(ei), phi=float(pi),
            exact_exponential=float(np.exp(zi)), exact_phi=float(rational_response(np.array([zi]), methods[0])[1][0]))
            for zi, ei, pi in zip(z, e, phi))
    for parent in parents:
        a = field(parent, settings["max_mode"])
        for kappa in (.02, 2.):
            horizon = .2
            reference, reference_seconds = integrate(a, horizon, diffusion=kappa)
            tighter, tighter_seconds = integrate(a, horizon, diffusion=kappa, rtol=2e-13, atol=2e-15)
            target = reference.sol(horizon)
            uncertainty = float(np.max(abs(target-tighter.sol(horizon))))
            independent, independent_seconds = rk4_reference(a, horizon, kappa)
            independent_error = float(np.max(abs(target-independent)))
            uncertainty = max(uncertainty, independent_error)
            teacher_id = f"XR/{parent}/{kappa:g}"
            raw["teacher_records"].append(dict(teacher_id=teacher_id, parent_id=parent, diffusion=kappa,
                horizon=horizon, teacher_seconds=reference_seconds, refinement_seconds=tighter_seconds,
                independent_rk4_seconds=independent_seconds, independent_rk4_endpoint_error=independent_error,
                reference_uncertainty=uncertainty))
            for steps in (1, 2, 4, 8):
                h = horizon/steps
                for method in methods:
                    ctx.budget.check()
                    tick = time.perf_counter()
                    cached = rational_response(-kappa*np.arange(-settings["max_mode"], settings["max_mode"]+1)**2*h, method)
                    build_seconds = time.perf_counter()-tick
                    state = a.copy(); tick = time.perf_counter()
                    for _ in range(steps):
                        state = etd_step(state, h, kappa, .7, method, cached)
                    seconds = time.perf_counter()-tick
                    raw["rows"].append(dict(parent_id=parent, teacher_id=teacher_id, method=method, h=h, steps=steps,
                        diffusion=kappa, **error_metrics(state, target), build_seconds=build_seconds,
                        inference_seconds=seconds, complete_seconds=seconds+build_seconds,
                        teacher_seconds=reference_seconds, refinement_seconds=tighter_seconds,
                        independent_rk4_seconds=independent_seconds, independent_rk4_endpoint_error=independent_error,
                        reference_uncertainty=uncertainty))
    # Equivalent caching opportunities: all methods cache their diffusion
    # response and nonlinear state product, and all refresh after a new state.
    a = field(parents[0], settings["max_mode"])
    modes = np.arange(-settings["max_mode"], settings["max_mode"]+1)
    dense, dense_build_seconds = integrate(a, .2)
    dense_refined, dense_refinement_seconds = integrate(a, .2, rtol=2e-13, atol=2e-15)
    for queries in settings["query_counts"]:
        horizons = np.linspace(.005, .2, queries)
        truth = [dense.sol(h) for h in horizons]
        def dense_complete():
            built, _ = integrate(a, .2)
            return [built.sol(h) for h in horizons]
        dense_timing = _paired_cpu({"complete_build_and_queries": dense_complete,
            "cached_queries_only": lambda: [dense.sol(h) for h in horizons]}, settings["timing_repeats"],
            np.random.default_rng(settings["seed"]+queries), ctx.budget)
        dense_error = max(float(np.linalg.norm(value-dense_refined.sol(h))/np.linalg.norm(dense_refined.sol(h)))
                          for h, value in zip(horizons, truth))
        raw["timing_rows"].append(dict(method="classical_dop853_dense", queries=queries,
            build_seconds=dense_build_seconds, maximum_relative_error=dense_error,
            reference_refinement_seconds=dense_refinement_seconds, reference_is_self=False,
            quality_qualified=dense_error <= settings["relative_tolerance"],
            qualification_reason="Tighter-tolerance DOP853 check; separate endpoint RK4 checks in PDE rows",
            **dense_timing))
        for method in methods:
            def complete():
                nonlinear = .7*(a-convolution(a, a))
                kernels = [rational_response(-.02*modes*modes*h, method) for h in horizons]
                return [e*a+h*phi*nonlinear for h, (e, phi) in zip(horizons, kernels)]
            tick = time.perf_counter()
            nonlinear = .7*(a-convolution(a, a))
            kernels = [rational_response(-.02*modes*modes*h, method) for h in horizons]
            build_seconds = time.perf_counter()-tick
            def cached_queries():
                return [e*a+h*phi*nonlinear for h, (e, phi) in zip(horizons, kernels)]
            def uncached():
                return [etd_step(a, float(h), .02, .7, method) for h in horizons]
            relative_error = max(float(np.linalg.norm(value-expected)/np.linalg.norm(expected))
                                 for value, expected in zip(cached_queries(), truth))
            timing = _paired_cpu({"complete_build_and_queries": complete, "cached_queries_only": cached_queries,
                                 "uncached_queries": uncached}, settings["timing_repeats"],
                                 np.random.default_rng(settings["seed"]+queries), ctx.budget)
            raw["timing_rows"].append(dict(method=method, queries=queries, build_seconds=build_seconds,
                maximum_relative_error=relative_error, relative_tolerance=settings["relative_tolerance"],
                quality_qualified=relative_error <= settings["relative_tolerance"], **timing))
    _write(ctx.path/"XR_resolvent.json", raw)
    for method in methods:
        selected = [v for v in raw["rows"] if v["method"] == method]
        baseline = {(v["parent_id"], v["diffusion"], v["steps"]): v for v in raw["rows"] if v["method"] == methods[0]}
        ratios = [v["error_rms"]/baseline[v["parent_id"], v["diffusion"], v["steps"]]["error_rms"] for v in selected]
        costs = [baseline[v["parent_id"], v["diffusion"], v["steps"]]["complete_seconds"]/v["complete_seconds"] for v in selected]
        e, phi = rational_response(z, method)
        identity = float(np.max(abs(e-1-z*phi)))
        _emit(ctx, rows, "XR", f"XR/{method}", method,
            dict(independent_parents=len(parents), error_rms=statistics.median(v["error_rms"] for v in selected),
                 error_max=max(v["error_max"] for v in selected), ratio_to_exponential_etd1=statistics.median(ratios),
                 descriptive_complete_speedup=statistics.median(costs),
                 stiff_limit_magnitude=float(abs(e[-1])), reference_uncertainty=max(v["reference_uncertainty"] for v in selected)),
            [check("exponential-phi-consistency", identity, 2e-15, category="math"),
             check("independent-reference-crosscheck", max(v["reference_uncertainty"] for v in selected), 1e-8, category="math"),
             check("small-step-consistency", abs(float(rational_response(np.array([-1e-8]), method)[1][0])-1), 2e-8, category="math"),
             check("negative-axis-stiff-damping", float(abs(e[-1])), 1e-4, category="gap"),
             check("nonlinear-error-noninferiority", max(ratios), 1.05, category="gap"),
             check("matched-accuracy-deployment-speed", None, None, category="utility",
                   reason="Per-call exploratory timers do not establish a validation-selected accuracy-cost frontier")],
            role="Analytic control", config=dict(formulation="diffusion ETD1 with rational response", max_mode=settings["max_mode"]))
    return raw


def _tangent(ctx, rows, settings):
    parents = [settings["seed"]+6000+i for i in range(settings["test_parents"])]
    k, cutoff = settings["max_mode"], 2
    raw = dict(schema="tdn.advance-tangent/v1", settings=settings, parent_ids=parents, rows=[], twin_rows=[], solve_records=[],
               exact_mean_law="m' = r(m-m^2-variance); variance=sum_{k != 0}|a_k|^2",
               tangent_equation="v'=(r-kappa*k^2)v-2r P_K(a*v)",
               cost_aggregation="Sum unique solve_records once; row teacher_seconds aliases the main variational solve, not additional work",
               scope="Jacobian and information-loss diagnostics; not a learned deployment solver")
    def solve_record(a, horizon, parent, role, **kwargs):
        solution, seconds = integrate(a, horizon, **kwargs)
        raw["solve_records"].append(dict(solve_id=f"XT/{parent}/{len(raw['solve_records']):06d}",
            parent_id=parent, horizon=horizon, role=role, seconds=seconds, nfev=solution.nfev,
            tangent="tangent" in kwargs, direction="forward"))
        return solution, seconds
    for parent in parents:
        a = field(parent, k)
        rng = np.random.default_rng(parent+9000)
        perturbation = complex_coordinates(rng.normal(size=2*k+1))
        perturbation[k-cutoff:k+cutoff+1] = 0
        perturbation /= np.linalg.norm(perturbation)
        for horizon in (.02, .2, .5):
            ctx.budget.check()
            solution, seconds = solve_record(a, horizon, parent, "main_variational", tangent=perturbation)
            endpoint, tangent = solution.y[:len(a), -1], solution.y[len(a):, -1]
            finite_differences = []
            for epsilon in settings["finite_difference_epsilons"]:
                plus, _ = solve_record(a+epsilon*perturbation, horizon, parent, f"finite_difference_plus_{epsilon:g}")
                minus, _ = solve_record(a-epsilon*perturbation, horizon, parent, f"finite_difference_minus_{epsilon:g}")
                fd = (plus.y[:, -1]-minus.y[:, -1])/(2*epsilon)
                finite_differences.append(dict(epsilon=epsilon,
                    relative_error=float(np.linalg.norm(fd-tangent)/np.linalg.norm(tangent))))
            half, _ = solve_record(a, horizon/2, parent, "semigroup_first", tangent=perturbation)
            composed, _ = solve_record(half.y[:len(a), -1], horizon/2, parent, "semigroup_second", tangent=half.y[len(a):, -1])
            semigroup = float(np.linalg.norm(composed.y[len(a):, -1]-tangent)/np.linalg.norm(tangent))
            variance = float(np.sum(abs(a)**2)-abs(a[k])**2)
            expected_mean = .7*(a[k].real-a[k].real**2-variance)
            # Second identity checks d(mean derivative)/da along the perturbation.
            expected_tangent_mean = .7*(perturbation[k] - 2*np.sum(a*perturbation[::-1]))
            raw["rows"].append(dict(parent_id=parent, h=horizon, teacher_seconds=seconds,
                tangent_fd_relative_error=min(v["relative_error"] for v in finite_differences),
                tangent_fd_worst_relative_error=max(v["relative_error"] for v in finite_differences),
                finite_difference_sweep=finite_differences, semigroup_relative_error=semigroup,
                mean_identity_error=float(abs(rhs(a)[k]-expected_mean)),
                tangent_mean_identity_error=float(abs(tangent_rhs(a, perturbation)[k]-expected_tangent_mean)),
                initial_coarse_tangent_norm=float(np.linalg.norm(project(perturbation, cutoff))),
                coarse_tangent_norm=float(np.linalg.norm(project(tangent, cutoff))),
                full_tangent_norm=float(np.linalg.norm(tangent)),
                endpoint_realness=float(np.max(abs(endpoint-endpoint[::-1].conj())))))
        # Same resolved state and same unresolved energy, different unresolved
        # phases. A scalar variance augmentation fixes instantaneous mean law,
        # but cannot generally identify the spatial low-mode response.
        twin = a.copy()
        phase = np.exp(1j*.8)
        for mode in range(cutoff+1, k+1):
            twin[k+mode] *= phase
            twin[k-mode] = twin[k+mode].conjugate()
        left, _ = solve_record(a, .2, parent, "twin_left")
        right, _ = solve_record(twin, .2, parent, "twin_right")
        raw["twin_rows"].append(dict(parent_id=parent,
            coarse_initial_difference=float(np.linalg.norm(project(a-twin, cutoff))),
            unresolved_energy_difference=float(abs(np.sum(abs(a)**2)-np.sum(abs(twin)**2))),
            mean_derivative_difference=float(abs(rhs(a)[k]-rhs(twin)[k])),
            coarse_future_difference=float(np.linalg.norm(project(left.y[:, -1]-right.y[:, -1], cutoff))),
            spatial_rhs_difference=float(np.linalg.norm(project(rhs(a)-rhs(twin), cutoff)))))
    raw["total_solve_seconds"] = sum(v["seconds"] for v in raw["solve_records"])
    _write(ctx.path/"XT_tangent.json", raw)
    selected = raw["rows"]
    _emit(ctx, rows, "XT", "XT/variational-flow", "fine_galerkin_variational_flow",
        dict(independent_parents=len(parents), maximum_fd_relative_error=max(v["tangent_fd_worst_relative_error"] for v in selected),
             maximum_semigroup_relative_error=max(v["semigroup_relative_error"] for v in selected),
             maximum_mean_identity_error=max(v["mean_identity_error"] for v in selected),
             minimum_hidden_to_resolved_sensitivity=min(v["coarse_tangent_norm"] for v in selected),
             maximum_full_tangent_norm=max(v["full_tangent_norm"] for v in selected)), [
        check("variational-finite-difference-sweep", max(v["tangent_fd_worst_relative_error"] for v in selected), settings["parity_tolerance"], category="math"),
        check("semigroup-chain-rule", max(v["semigroup_relative_error"] for v in selected), 1e-8, category="math"),
        check("logistic-mean-identity", max(v["mean_identity_error"] for v in selected), 1e-13, category="math"),
        check("mean-tangent-identity", max(v["tangent_mean_identity_error"] for v in selected), 1e-13, category="math"),
        check("unresolved-information-reaches-resolved-modes", min(v["coarse_tangent_norm"] for v in selected), 1e-8, "ge", category="gap"),
        check("deployment-benefit", None, None, category="utility", reason="Derivative information diagnostic, not an inference accelerator")],
        role="Analytic control")
    twins = raw["twin_rows"]
    _emit(ctx, rows, "XT", "XT/variance-augmentation-information-limit", "coarse_state_plus_variance",
        dict(independent_parents=len(parents), maximum_feature_difference=max(v["coarse_initial_difference"]+v["unresolved_energy_difference"] for v in twins),
             minimum_future_separation=min(v["coarse_future_difference"] for v in twins)), [
        check("identical-augmented-state", max(v["coarse_initial_difference"]+v["unresolved_energy_difference"] for v in twins), 1e-13, category="math"),
        check("identical-instantaneous-mean-law", max(v["mean_derivative_difference"] for v in twins), 1e-13, category="math"),
        check("scalar-variance-identifies-spatial-future", max(v["coarse_future_difference"] for v in twins), 1e-8, category="gap",
              reason="BAD is the useful negative result: scalar energy cannot recover unresolved phase")],
        role="Analytic control")
    return raw


def run(ctx):
    if ctx.device != "cpu":
        raise ValueError("Advance exploration explicitly measures CPU; no GPU relabeling")
    ctx.path = Path(ctx.path); ctx.path.mkdir(parents=True, exist_ok=True)
    full = ctx.protocol.get("profile") == "full"
    settings = dict(full=full, seed=88_000_000 if full else 86_000_000, max_mode=8 if full else 6,
        history_counts=[12, 6, 12] if full else [4, 2, 4], test_parents=12 if full else 4,
        rollout_steps=20 if full else 6, timing_repeats=11 if full else 5,
        query_counts=[1, 4, 16, 64] if full else [1, 4, 16], noise_levels=[0., 1e-4, 1e-3],
        finite_difference_epsilons=[1e-4, 1e-5, 1e-6], parity_tolerance=1e-6, relative_tolerance=1e-3)
    # The effective settings are always serialized. Only recognized declared
    # keys may override defaults, preventing untracked workload expansion.
    declaration = ctx.protocol.get("exploration", {})
    settings.update({key: declaration[key] for key in settings if key in declaration})
    history, response, tangent = (declaration.get(key, {}) for key in ("XH", "XR", "XT"))
    settings["history_counts"] = [history.get("train_parents", settings["history_counts"][0]),
        history.get("validation_parents", settings["history_counts"][1]),
        history.get("test_parents", settings["history_counts"][2])]
    for key in ("noise_levels", "rollout_steps"):
        settings[key] = history.get(key, settings[key])
    for key in ("query_counts", "relative_tolerance"):
        settings[key] = response.get(key, settings[key])
    settings["test_parents"] = tangent.get("parents", settings["test_parents"])
    for key in ("finite_difference_epsilons", "parity_tolerance"):
        settings[key] = tangent.get(key, settings[key])
    if (settings["seed"] < 80_000_000 or not 4 <= settings["max_mode"] <= 16 or
        not 1 <= settings["test_parents"] <= 32 or not 1 <= settings["rollout_steps"] <= 40 or
        len(settings["history_counts"]) != 3 or any(not 2 <= count <= 32 for count in settings["history_counts"]) or
        not 3 <= settings["timing_repeats"] <= 31 or any(not 1 <= count <= 128 for count in settings["query_counts"])):
        raise ValueError("Exploration settings exceed the declared bounded development envelope")
    unit = getattr(ctx, "unit", ctx.protocol.get("units", {}).get(ctx.stage, {}))
    prototype = unit.get("prototype_id")
    if prototype is None and len(unit.get("prototype_ids", [])) == 1:
        prototype = unit["prototype_ids"][0]
    if prototype not in ("XH", "XR", "XT"):
        raise ValueError("Exploration unit must declare exactly one XH/XR/XT prototype")
    rows = []
    started = time.perf_counter()
    raw = {"XH": _history, "XR": _resolvent, "XT": _tangent}[prototype](ctx, rows, settings)
    raw_name = f"{prototype}_{ {'XH': 'history', 'XR': 'resolvent', 'XT': 'tangent'}[prototype]}.json"
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    _write(ctx.path/"exploration_provenance.json", dict(schema="tdn.advance-exploration-provenance/v1",
        prototype_id=prototype, source_file="tdn/analysis/advance/exploration.py", source_sha256=source_sha,
        raw_file=raw_name, raw_sha256=hashlib.sha256((ctx.path/raw_name).read_bytes()).hexdigest(),
        settings=settings, device="cpu", seed=settings["seed"], parent_ids=raw["parent_ids"],
        interpretation="development-only; numerical checks are not proofs and repeated times are not independent parents"))
    return dict(status="COMPLETED", prototype_id=prototype, experiment_rows=len(rows),
                numerical_seconds=time.perf_counter()-started, independent_parents=raw["parent_ids"],
                raw_artifacts=[raw_name, "prototype_rows.json", "exploration_provenance.json"],
                scientific_outcome="BOUNDED_DEVELOPMENT_EVIDENCE; utility may be BAD or NA")
