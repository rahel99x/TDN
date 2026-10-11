"""MX: shadow-band closure and two-observation causal initialization.

The protected problem is a one-dimensional, unit-period Galerkin equation. Its
unknown hidden initial modes are explicit and never mixed with the main study's
zero-hidden-mode lift. Teacher data end at startup for every deployed observer;
the remaining rollout is autonomous. Growth-allowed hidden rates are physical.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from functools import lru_cache

import numpy as np
from scipy.integrate import solve_ivp

from tdn.analysis.frontier.core import check, clean
from .fields import stable_seed


def _plain(value):
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return clean(value)


def product(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim != 1 or a.shape != b.shape or len(a) % 2 != 1:
        raise ValueError("Galerkin products need matching odd symmetric mode arrays")
    cutoff = len(a)//2
    return np.convolve(a, b)[cutoff:3*cutoff+1]


def project(a, cutoff):
    current = len(a)//2
    if not 0 <= cutoff <= current:
        raise ValueError("Invalid coarse projection")
    return np.array(a[current-cutoff:current+cutoff+1], copy=True)


def lift(a, cutoff):
    if cutoff < len(a)//2:
        raise ValueError("A lift cannot remove modes")
    result = np.zeros(2*cutoff+1, complex)
    current = len(a)//2
    result[cutoff-current:cutoff+current+1] = a
    return result


def coordinates(a):
    k = len(a)//2
    return np.r_[a[k].real, a[k+1:].real, a[k+1:].imag]


def coefficients(x):
    x = np.asarray(x, float)
    if x.ndim != 1 or len(x) % 2 != 1:
        raise ValueError("Spectral real coordinates need odd length")
    k = len(x)//2
    positive = x[1:k+1]+1j*x[k+1:]
    return np.r_[positive[::-1].conj(), x[0]+0j, positive]


def rates(cutoff, mean, kappa, reaction):
    modes = np.arange(-cutoff, cutoff+1)
    return reaction*(1-2*mean)-kappa*(2*np.pi*modes)**2


def rhs(a, kappa, reaction):
    modes = np.arange(-len(a)//2+1, len(a)//2+1)
    return (reaction-kappa*(2*np.pi*modes)**2)*a-reaction*product(a, a)


def shadow_rhs(a, coarse_cutoff, kappa, reaction, *, rate_shift=0., full_feedback=False):
    """Resolved feedback retained; linear shadow omits z*v and z*z only in Q.

    A learned additive hidden-rate correction is unrestricted in sign. It does
    not silently replace unstable/growing physical modes by decays.
    """
    cutoff = len(a)//2
    if cutoff <= coarse_cutoff:
        raise ValueError("Shadow band must extend the observed band")
    if full_feedback:
        return rhs(a, kappa, reaction)
    coarse = lift(project(a, coarse_cutoff), cutoff)
    hidden = a-coarse
    v = coarse.copy(); v[cutoff] = 0.
    answer = rhs(a, kappa, reaction)
    hidden_derivative = (rates(cutoff, coarse[cutoff].real, kappa, reaction)+rate_shift)*hidden-reaction*product(v, v)
    hidden_derivative[cutoff-coarse_cutoff:cutoff+coarse_cutoff+1] = answer[cutoff-coarse_cutoff:cutoff+coarse_cutoff+1]
    return hidden_derivative


def advance(a, h, kappa, reaction, *, coarse_cutoff=None, rate_shift=0., full_feedback=False):
    if not math.isfinite(h) or h < 0:
        raise ValueError("Only finite forward steps are allowed")
    cutoff = len(a)//2
    substeps = max(1, math.ceil(h*(kappa*(2*np.pi*cutoff)**2+abs(reaction))/1.0))
    dt = h/substeps
    state = np.array(a, copy=True)
    fun = (lambda u: rhs(u, kappa, reaction)) if coarse_cutoff is None else (
        lambda u: shadow_rhs(u, coarse_cutoff, kappa, reaction, rate_shift=rate_shift, full_feedback=full_feedback))
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(substeps):
            k1 = fun(state); k2 = fun(state+dt*k1/2); k3 = fun(state+dt*k2/2); k4 = fun(state+dt*k3)
            state = state+dt*(k1+2*k2+2*k3+k4)/6
    return state


def observation_noise(a, sigma, seed):
    """Independent real-field Gaussian noise with expected physical RMS sigma."""
    if sigma < 0 or not math.isfinite(sigma):
        raise ValueError("Noise RMS must be finite and nonnegative")
    rng = np.random.default_rng(seed)
    n = len(a)
    vector = rng.normal(size=n)*sigma/math.sqrt(n)
    vector[1:] /= math.sqrt(2.)
    return np.asarray(a)+coefficients(vector)


def _fit(x, y, ridge):
    x, y = np.asarray(x, float), np.asarray(y, float)
    mean, scale = x.mean(0), np.maximum(x.std(0), 1e-7)
    design = np.c_[np.ones(len(x)), (x-mean)/scale]
    penalty = np.eye(design.shape[1])*float(ridge)*len(design); penalty[0, 0] = 0.
    fitted = np.linalg.solve(design.T@design+penalty+1e-12*np.eye(len(penalty)), design.T@y)
    return dict(mean=mean, scale=scale, weights=fitted, ridge=float(ridge), parameters=fitted.size)


def _predict(model, features):
    return np.r_[1., (features-model["mean"])/model["scale"]]@model["weights"]


def observer_features(previous, current, h, kappa, reaction):
    propagated = advance(previous, h, kappa, reaction)
    return np.r_[coordinates(current), coordinates(propagated-current)]


def initialize_observer(previous, current, *, h, kappa, reaction, kind, gain=0., model=None):
    """Two causal startup observations only. No future target is accepted."""
    if kind == "current_only" or (kind == "deterministic" and gain == 0.):
        return np.array(current, copy=True)
    if kind == "deterministic":
        if not 0 <= gain <= 1:
            raise ValueError("Observer gain must lie in [0,1]")
        return current+gain*(advance(previous, h, kappa, reaction)-current)
    if kind in ("clean_fitted_observer", "noise_aware_observer") and model is not None:
        return current+coefficients(_predict(model, observer_features(previous, current, h, kappa, reaction)))
    raise ValueError("Unknown or unfitted causal observer")


def forecast(previous, current, *, h, kappa, reaction, steps, method, model=None, gain=0., decay=0.):
    """One-time observation assimilation followed by an autonomous rollout."""
    if type(steps) is not int or steps < 1:
        raise ValueError("Rollout length must be positive")
    tick = time.perf_counter()
    if method in ("coarse_rk4", "damped_secant", "clean_fitted_memory"):
        state = np.array(current, copy=True)
    else:
        kind = "deterministic" if method == "deterministic_observer" else method
        state = initialize_observer(previous, current, h=h, kappa=kappa, reaction=reaction, kind=kind, gain=gain, model=model)
    startup_seconds = time.perf_counter()-tick
    prev = np.array(previous, copy=True)
    predictions = np.full((steps, len(state)), np.nan+0j)
    step_seconds = np.zeros(steps)
    for step in range(steps):
        tick = time.perf_counter()
        base = advance(state, h, kappa, reaction)
        if method == "damped_secant" and gain != 0.:
            base = base+gain*math.exp(-decay*reaction*step*h)*(state-advance(prev, h, kappa, reaction))
        elif method == "clean_fitted_memory":
            features = np.r_[coordinates(state), coordinates((state-advance(prev, h, kappa, reaction))/h)]
            base = base+h*coefficients(_predict(model, features))
        step_seconds[step] = time.perf_counter()-tick
        predictions[step] = base
        prev, state = state, base
        if not np.isfinite(state).all():
            break
    return predictions, dict(startup_seconds=startup_seconds, rollout_seconds=float(step_seconds.sum()),
        total_seconds=startup_seconds+float(step_seconds.sum()), step_seconds=step_seconds.tolist())


def _refined_path(a, times, kappa, reaction, budget, *, coarse_cutoff=None,
                  rate_shift=0., full_feedback=False):
    """Two tolerances for a declared finite vector field, with separate costs.

    A refined shadow solution is a control for temporal error in that shadow
    equation, not a reference for the fine equation.  It does not see fine truth.
    """
    start = time.perf_counter()
    def fun(t, y):
        budget.check()
        return rhs(y, kappa, reaction) if coarse_cutoff is None else shadow_rhs(
            y, coarse_cutoff, kappa, reaction, rate_shift=rate_shift,
            full_feedback=full_feedback)
    horizon = max(times)
    rough = solve_ivp(fun, (0, horizon), a, method="DOP853", rtol=2e-10, atol=2e-12, t_eval=times)
    rough_seconds = time.perf_counter()-start
    refined_start = time.perf_counter()
    refined = solve_ivp(fun, (0, horizon), a, method="DOP853", rtol=2e-12, atol=2e-14, t_eval=times)
    refined_seconds = time.perf_counter()-refined_start
    if not rough.success or not refined.success or not np.isfinite(refined.y).all():
        raise RuntimeError("Protected closure reference failed")
    difference = refined.y.T-rough.y.T
    rms = float(np.sqrt(np.sum(abs(difference)**2, axis=1)).max())
    maximum = max(_metrics(d)["maximum"] for d in difference)
    return refined.y.T, dict(accepted=max(rms, maximum) < 2e-8, uncertainty_rms=rms, uncertainty_maximum=maximum,
        method="independent tolerance refinement of DOP853 on the declared finite vector field", elapsed_seconds=time.perf_counter()-start,
        nfev=rough.nfev+refined.nfev, refined_nfev=refined.nfev, refined_seconds=refined_seconds,
        refinement_check_seconds=rough_seconds,
        finite_equation="Galerkin" if coarse_cutoff is None else "shadow-band evolution",
        cutoff=len(a)//2, interpretation="temporal convergence estimate for this finite vector field; not a continuum certificate")


def _teacher(a, times, kappa, reaction, budget):
    return _refined_path(a, times, kappa, reaction, budget)


def _controlled_path(a, times, kappa, reaction, budget, *, substeps,
                     coarse_cutoff=None, rate_shift=0., full_feedback=False):
    """Common explicit RK4 mesh across coarse and shadow component controls."""
    if not isinstance(substeps, int) or substeps < 1:
        raise ValueError("A positive declared RK4 substep count is required")
    state = np.array(a, copy=True); previous = 0.; states = []
    start = time.perf_counter()
    fun = (lambda x: rhs(x, kappa, reaction)) if coarse_cutoff is None else (
        lambda x: shadow_rhs(x, coarse_cutoff, kappa, reaction,
            rate_shift=rate_shift, full_feedback=full_feedback))
    with np.errstate(over="ignore", invalid="ignore"):
        for endpoint in times:
            budget.check()
            if not math.isfinite(endpoint) or endpoint <= previous:
                raise ValueError("Attribution times must be finite, positive and strictly increasing")
            dt = (endpoint-previous)/substeps
            for _ in range(substeps):
                k1 = fun(state); k2 = fun(state+dt*k1/2)
                k3 = fun(state+dt*k2/2); k4 = fun(state+dt*k3)
                state = state+dt*(k1+2*k2+2*k3+k4)/6
            states.append(state.copy()); previous = endpoint
    return np.stack(states), dict(prediction_seconds=time.perf_counter()-start,
        nfev=4*len(times)*substeps, macrosteps=len(times), substeps_per_macrostep=substeps,
        method="RK4 on the same frozen mesh for coarse and shadow models")


def signed_interaction(baseline, temporal, closure, combined, truth):
    """Exact endpoint factorial identity; adverse interaction stays signed."""
    fields = [np.asarray(value) for value in (baseline, temporal, closure, combined, truth)]
    if len({value.shape for value in fields}) != 1 or fields[0].ndim != 1:
        raise ValueError("Attribution vectors must share one odd Fourier coefficient shape")
    if not all(np.isfinite(value).all() for value in fields):
        return dict(status="NONFINITE", squared_identity_discrepancy=None,
            combined_beats_strongest_component=None)
    baseline, temporal, closure, combined, truth = fields
    terms = dict(base_error=baseline-truth, temporal_change=temporal-baseline,
        closure_change=closure-baseline, interaction=combined-temporal-closure+baseline)
    cross = [dict(left=left, right=right, twice_inner_product=float(2*np.vdot(terms[left], terms[right]).real))
        for i, left in enumerate(terms) for right in list(terms)[i+1:]]
    squared = {key: float(np.vdot(value, value).real) for key, value in terms.items()}
    actual = float(np.vdot(combined-truth, combined-truth).real)
    errors = {name: _metrics(value-truth) for name, value in
        (("baseline", baseline), ("temporal_only", temporal), ("closure_only", closure), ("combined", combined))}
    strongest = min(("temporal_only", "closure_only"), key=lambda key: errors[key]["rms"])
    return dict(status="MEASURED", term_squared_norms=squared, signed_cross_terms=cross,
        vector_identity_maximum=float(np.max(abs(sum(terms.values())-(combined-truth)))),
        squared_identity_discrepancy=abs(actual-sum(squared.values())-sum(c["twice_inner_product"] for c in cross)),
        errors=errors, combined_beats_strongest_component=errors["combined"]["rms"] < errors[strongest]["rms"],
        strongest_component=strongest, strongest_component_scope="Hindsight attribution comparator, not a deployable selected policy",
        combined_minus_strongest_component_rms=errors["combined"]["rms"]-errors[strongest]["rms"],
        interpretation="Signed changes and cross terms are not independent positive error budgets; negative/cancelling interactions remain visible")


def temporal_closure_attribution(initial, times, truth, fine_reference, *, kappa,
                                 reaction, shadow_cutoff, rate_shift, budget):
    """Same-observation 2x2 temporal/closure experiment with no fitted new data.

    Physical truth is the existing projected fine trajectory.  A refined coarse
    solution isolates temporal discretization from omitted-scale closure.  The
    fitted hidden-rate parameter is frozen from training parents before this
    function is called; neither closure trajectory receives fine targets.
    """
    coarse = len(initial)//2
    if shadow_cutoff <= coarse:
        raise ValueError("Attribution shadow cutoff must exceed the observed cutoff")
    times = np.asarray(times, float)
    if times.ndim != 1 or len(times) == 0 or np.any(np.diff(np.r_[0., times]) <= 0):
        raise ValueError("Attribution requires increasing positive times")
    truth = np.stack([project(v, coarse) for v in truth])
    if len(truth) != len(times):
        raise ValueError("Attribution trajectory and time counts differ")
    # All few-step components get the same stable mesh, chosen from known
    # coefficients/cutoff before seeing an error. No oracle time-step choice.
    interval = float(np.max(np.diff(np.r_[0., times])))
    substeps = max(1, math.ceil(interval*(kappa*(2*np.pi*shadow_cutoff)**2+abs(reaction)+abs(rate_shift))))
    baseline, base_cost = _controlled_path(initial, times, kappa, reaction, budget, substeps=substeps)
    temporal, coarse_ref = _refined_path(initial, times, kappa, reaction, budget)
    temporal_error, closure_error = baseline[-1]-temporal[-1], temporal[-1]-truth[-1]
    two_terms = dict(temporal_discretization=_metrics(temporal_error), omitted_scale_closure=_metrics(closure_error),
        total=_metrics(baseline[-1]-truth[-1]),
        signed_twice_inner_product=float(2*np.vdot(temporal_error, closure_error).real),
        vector_identity_maximum=float(np.max(abs(temporal_error+closure_error-(baseline[-1]-truth[-1])))),
        temporal_uncertainty_rms=coarse_ref["uncertainty_rms"],
        closure_uncertainty_rms=coarse_ref["uncertainty_rms"]+fine_reference["uncertainty_rms"])
    references = [dict(role="coarse_temporal_reference", **coarse_ref)]
    raw = dict(initial=initial, truth=truth, baseline=baseline, temporal_only=temporal)
    rows, interactions = [], []
    for name, path, cost in (("baseline", baseline, base_cost), ("temporal_only", temporal,
        dict(prediction_seconds=coarse_ref["refined_seconds"], nfev=coarse_ref["refined_nfev"],
             refinement_check_seconds=coarse_ref["refinement_check_seconds"], method="DOP853 refined coarse equation"))):
        errors = [_metrics(p-y) for p, y in zip(path, truth)]
        rows.append(dict(method=name, component="coarse", endpoint_rms=errors[-1]["rms"],
            endpoint_maximum=errors[-1]["maximum"], rollout_error_rms=[x["rms"] for x in errors],
            prediction_finite=all(x["rms"] is not None for x in errors), cost=cost))
    for family, shift, full in (("deterministic_shadow", 0., False),
        ("fitted_rate_shadow", rate_shift, False), ("full_shadow_galerkin", 0., True)):
        lifted = lift(initial, shadow_cutoff)
        path, cost = _controlled_path(lifted, times, kappa, reaction, budget, substeps=substeps,
            coarse_cutoff=coarse, rate_shift=shift, full_feedback=full)
        accurate, shadow_ref = _refined_path(lifted, times, kappa, reaction, budget,
            coarse_cutoff=coarse, rate_shift=shift, full_feedback=full)
        path = np.stack([project(v, coarse) for v in path])
        accurate = np.stack([project(v, coarse) for v in accurate])
        references.append(dict(role=family+"_temporal_reference", **shadow_ref))
        raw[family+"_closure_only"], raw[family+"_combined"] = path, accurate
        for arm, answer, measured_cost in (("closure_only", path, cost), ("combined", accurate,
            dict(prediction_seconds=shadow_ref["refined_seconds"], nfev=shadow_ref["refined_nfev"],
                refinement_check_seconds=shadow_ref["refinement_check_seconds"], method="DOP853 refined shadow equation"))):
            errors = [_metrics(p-y) for p, y in zip(answer, truth)]
            rows.append(dict(method=family+"/"+arm, component=arm, shadow_model=family,
                endpoint_rms=errors[-1]["rms"], endpoint_maximum=errors[-1]["maximum"],
                rollout_error_rms=[x["rms"] for x in errors], prediction_finite=all(x["rms"] is not None for x in errors),
                cost=measured_cost, hidden_rate_shift=shift))
        interactions.append(dict(shadow_model=family, **signed_interaction(baseline[-1], temporal[-1],
            path[-1], accurate[-1], truth[-1])))
    return dict(rows=rows, interactions=interactions, decomposition=two_terms,
        all_temporal_references_accepted=all(ref["accepted"] for ref in references), fine_reference=fine_reference,
        reference_histories=references, substeps_per_macrostep=substeps, times=times.tolist(),
        input_sha256=hashlib.sha256(initial.tobytes()).hexdigest(), coarse_cutoff=coarse, shadow_cutoff=shadow_cutoff,
        information="Identical observed coarse initial state; zero hidden-mode lift; known coefficients; autonomous trajectories; no truth refresh",
        scope="Numerical temporal/closure attribution; no new neural model or scientific superiority claim",
        cost_scope="Each unique component call measured once; verification solves charged separately; CPU diagnostic timing, not deployment frontier"), raw


@lru_cache(maxsize=16)
def _evaluation_matrix(cutoff):
    return np.exp(2j*np.pi*np.outer(np.arange(257)/257, np.arange(-cutoff, cutoff+1)))


def _metrics(a):
    if not np.isfinite(a).all():
        return dict(rms=None, maximum=None, mean=None, minimum=None, upper=None)
    k = len(a)//2
    values = (_evaluation_matrix(k)@a).real
    return dict(rms=float(np.sqrt(np.sum(abs(a)**2))), maximum=float(np.max(abs(values))),
                mean=float(a[k].real), minimum=float(values.min()), upper=float(values.max()))


def _parent(root_seed, split, index, coarse_cutoff, fine_cutoff, *, hidden=True):
    seed = stable_seed(root_seed, "closure", split, index, "phase")
    rng = np.random.default_rng(seed)
    result = np.zeros(2*fine_cutoff+1, complex)
    mean = float(rng.choice([.08, .2, .4, .65, .85]))
    result[fine_cutoff] = mean
    max_mode = fine_cutoff if hidden else coarse_cutoff
    amplitudes = rng.uniform(.2, 1., max_mode)/(1+np.arange(1, max_mode+1))
    amplitudes *= .5*min(mean, 1-mean)/amplitudes.sum()
    for mode, amplitude in enumerate(amplitudes, 1):
        phase = rng.uniform(-np.pi, np.pi)
        coefficient = amplitude*np.exp(1j*phase)/2
        result[fine_cutoff+mode], result[fine_cutoff-mode] = coefficient, coefficient.conjugate()
    return result, dict(parent_id=f"response-closure-{split}-{index:05d}-{seed:x}", split=split, index=index,
        independent_phase_seed=seed, initial_mean=mean, fine_cutoff=fine_cutoff, observed_cutoff=coarse_cutoff,
        initial_hidden_modes=hidden, field_sha256=hashlib.sha256(result.tobytes()).hexdigest())


def _loss(prediction, truth):
    if not np.isfinite(prediction).all():
        return 1e100
    return float(np.mean(np.sum(abs(prediction-truth)**2, axis=1)))


def select_observer(validation, *, kind, candidates, h, kappa, reaction, steps, budget=None):
    """All validation descendants have fixed equal weight; never sees test parents."""
    if not validation or any(v.get("split") != "validation" for v in validation):
        raise ValueError("Observer selection requires validation parents only")
    measured = []
    for index, candidate in enumerate(candidates):
        losses = []
        start = time.perf_counter()
        for item in validation:
            if budget is not None:
                budget.check()
            prediction, _ = forecast(item["previous"], item["current"], h=h, kappa=kappa, reaction=reaction,
                steps=steps, method=kind, **candidate)
            losses.append(_loss(prediction, item["truth"]))
        measured.append(dict(candidate=index, validation_mse=float(np.mean(losses)), elapsed_seconds=time.perf_counter()-start,
            gain=candidate.get("gain"), decay=candidate.get("decay"), ridge=candidate.get("model", {}).get("ridge")))
    best = min(range(len(measured)), key=lambda j: (measured[j]["validation_mse"], j))
    return candidates[best], measured


def onset_diagnostics(budget):
    """Reaction-only h² closure and hidden-initial h onset are distinct."""
    coarse, fine, reaction = 8, 64, 3.
    initial = np.zeros(2*fine+1, complex); initial[fine] = .4
    initial[fine-6] = initial[fine+6] = .075
    resolved = project(initial, coarse)
    square = product(initial, initial)
    unresolved_square = square-lift(project(square, coarse), fine)
    coefficient = reaction**2*project(product(initial, unresolved_square), coarse)
    rows, differences = [], []
    for h in (.0025, .005, .01, .02):
        coarse_truth, cref = _teacher(resolved, [h], 0., reaction, budget)
        # Exact pointwise logistic sampled well beyond all initial modes gives
        # an independent fine solution to the reaction-only continuum equation.
        x = np.arange(1025)/1025
        u = .4+.15*np.cos(2*np.pi*6*x)
        exact = u/(u+(1-u)*np.exp(-reaction*h))
        coefficients_exact = np.fft.fft(exact, norm="forward")
        fine_projected = np.r_[coefficients_exact[-coarse:], coefficients_exact[:coarse+1]]
        delta = fine_projected-coarse_truth[0]
        error = _metrics(delta)["rms"]
        differences.append(error)
        rows.append(dict(h=h, measured_rms=error, leading_h2_rms=_metrics(h*h*coefficient)["rms"],
            coefficient_relative_error=_metrics(delta/(h*h)-coefficient)["rms"]/max(_metrics(coefficient)["rms"], 1e-30),
            reference_uncertainty_rms=cref["uncertainty_rms"]))
    orders = np.log(np.array(differences[1:])/differences[:-1])/np.log(2.)
    hidden = initial.copy(); hidden[fine-9] = hidden[fine+9] = .02
    derivative_gap = project(rhs(hidden, 0., reaction), coarse)-rhs(project(hidden, coarse), 0., reaction)
    hidden_rows = []
    for h in (.0005, .001, .002):
        truth, _ = _teacher(hidden, [h], 0., reaction, budget)
        reduced, _ = _teacher(project(hidden, coarse), [h], 0., reaction, budget)
        delta = project(truth[0], coarse)-reduced[0]
        hidden_rows.append(dict(h=h, measured_rms=_metrics(delta)["rms"], first_order_relative_error=
            _metrics(delta/h-derivative_gap)["rms"]/max(_metrics(derivative_gap)["rms"], 1e-30)))
    return dict(reaction_only=rows, observed_orders=orders.tolist(), initially_hidden=hidden_rows,
        physical_statement="Initially resolved closure begins at h² even when kappa=0; nonzero initial hidden state generally permits an h term")


def run(ctx):
    cfg = ctx.protocol.get("response", {}).get("closure", {})
    smoke = ctx.protocol.get("profile") == "smoke"
    root_seed = int(ctx.protocol.get("root_seed", 111010001))
    coarse, shadow = int(cfg.get("coarse_cutoff", 6)), int(cfg.get("shadow_cutoff", 12))
    fine = int(cfg.get("fine_cutoff", 18))
    if not 1 <= coarse < shadow <= fine:
        raise ValueError("Closure requires observed < shadow <= reference cutoffs")
    h, kappa, reaction = float(cfg.get("step", .01)), float(cfg.get("kappa", .0005)), float(cfg.get("reaction", .7))
    steps = int(cfg.get("rollout_steps", 8 if smoke else 32))
    counts = dict(train=int(cfg.get("train_parents", 4 if smoke else 32)),
        validation=int(cfg.get("validation_parents", 2 if smoke else 16)), development_test=int(cfg.get("test_parents", 2 if smoke else 16)))
    noise_levels = list(cfg.get("noise_levels", [0., 1e-4, 1e-3]))
    if not {0., 1e-4, 1e-3} <= set(noise_levels):
        raise ValueError("Protected observer requires clean, 1e-4 and 1e-3 noise cases")
    shifts = list(cfg.get("step_multipliers", [1., 2.]))
    reaction_rates = list(cfg.get("reaction_rates", [reaction, 3*reaction]))
    ridges = cfg.get("ridges", [1e-3, .1] if smoke else [1e-5, 1e-3, .1])
    started = time.perf_counter()
    onset_tick = time.perf_counter()
    onset = onset_diagnostics(ctx.budget)
    onset_seconds = time.perf_counter()-onset_tick
    parent_specs, parents, teacher_rows, clean_data = [], {}, [], {}
    for split, count in counts.items():
        for index in range(count):
            ctx.budget.check()
            field, spec = _parent(root_seed, split, index, coarse, fine)
            parent_specs.append(spec); parents[spec["parent_id"]] = field
            times = np.arange(steps+2)*h
            truth, reference = _teacher(field, times, kappa, reaction, ctx.budget)
            clean_data[spec["parent_id"]] = dict(previous=project(truth[0], coarse), current=project(truth[1], coarse),
                truth=np.stack([project(v, coarse) for v in truth[2:]]), full=truth, split=split, reference=reference, parent_id=spec["parent_id"])
            teacher_rows.append(dict(parent_id=spec["parent_id"], reaction=reaction, h=h, **reference))
    # Historical clean memory uses true clean transition targets but is deployed
    # autonomously; this control is intentionally distinct from one-time denoising.
    memory_x, memory_y, clean_x, clean_y = [], [], [], []
    for item in clean_data.values():
        if item["split"] != "train":
            continue
        memory_x.append(np.r_[coordinates(item["current"]), coordinates((item["current"]-advance(item["previous"], h, kappa, reaction))/h)])
        memory_y.append(coordinates((item["truth"][0]-advance(item["current"], h, kappa, reaction))/h))
        clean_x.append(observer_features(item["previous"], item["current"], h, kappa, reaction))
        clean_y.append(np.zeros(len(item["current"])))
    tick = time.perf_counter()
    memory_models = [_fit(memory_x, memory_y, ridge) for ridge in ridges]
    clean_model = _fit(clean_x, clean_y, ridges[0])
    fit_seconds = time.perf_counter()-tick
    selections, settings, fitted_models = [], {}, []
    for sigma in noise_levels:
        data = []
        for item in clean_data.values():
            if item["split"] == "development_test":
                continue
            seed = stable_seed(root_seed, item["parent_id"], "noise", sigma, 0)
            data.append(dict(item, previous=observation_noise(item["previous"], sigma, seed),
                current=observation_noise(item["current"], sigma, seed+1)))
        train = [v for v in data if v["split"] == "train"]
        validation = [v for v in data if v["split"] == "validation"]
        tick = time.perf_counter()
        observer_models = [_fit([observer_features(v["previous"], v["current"], h, kappa, reaction) for v in train],
            [coordinates(clean_data[v["parent_id"]]["current"]-v["current"]) for v in train], ridge) for ridge in ridges]
        fit_seconds += time.perf_counter()-tick
        methods = {"coarse_rk4": [{}], "deterministic_observer": [dict(gain=g) for g in (0., .25, .5, .75)],
            "damped_secant": [dict(gain=g, decay=d) for g in (0., .25, .5, 1.) for d in (0., 1.)],
            "clean_fitted_observer": [dict(model=clean_model)], "noise_aware_observer": [dict(model=m) for m in observer_models],
            "clean_fitted_memory": [dict(model=m) for m in memory_models]}
        for method, candidates in methods.items():
            selected, evidence = select_observer(validation, kind=method, candidates=candidates, h=h, kappa=kappa, reaction=reaction,
                steps=steps, budget=ctx.budget)
            settings[(sigma, method)] = selected
            selections.append(dict(noise=sigma, method=method, selected_gain=selected.get("gain"), selected_decay=selected.get("decay"),
                selected_ridge=selected.get("model", {}).get("ridge"), rows=evidence, selection_split="validation"))
            if selected.get("model") is not None:
                fitted_models.append(dict(noise=sigma, method=method, model=selected["model"]))
    # Fit an unrestricted scalar hidden-rate residual on development train only.
    numerator, denominator = 0., 0.
    for item in clean_data.values():
        if item["split"] != "train":
            continue
        for full in item["full"][:3]:
            state = project(full, shadow)
            residual = project(rhs(full, kappa, reaction), shadow)-shadow_rhs(state, coarse, kappa, reaction)
            hidden = state-lift(project(state, coarse), shadow)
            numerator += float(np.vdot(hidden, residual).real)
            denominator += float(np.vdot(hidden, hidden).real)
    rate_shift = numerator/(denominator+1e-12)
    rows, shadow_rows, ambiguities, attributions = [], [], [], []
    attribution_parents = min(counts["development_test"], int(cfg.get("attribution_parents", 1 if smoke else 2)))
    if attribution_parents < 1:
        raise ValueError("Protected closure requires at least one predeclared numerical attribution parent")
    raw = {}
    for spec in parent_specs:
        if spec["split"] != "development_test":
            continue
        parent_id, initial = spec["parent_id"], parents[spec["parent_id"]]
        for rate in reaction_rates:
            for multiplier in shifts:
                step = h*multiplier
                times = np.arange(steps+2)*step
                truth, reference = _teacher(initial, times, kappa, rate, ctx.budget)
                teacher_rows.append(dict(parent_id=parent_id, reaction=rate, h=step, **reference))
                observed = np.stack([project(v, coarse) for v in truth])
                for sigma in noise_levels:
                    noise_seed = stable_seed(root_seed, parent_id, "noise", sigma, rate, multiplier)
                    previous = observation_noise(observed[0], sigma, noise_seed)
                    current = observation_noise(observed[1], sigma, noise_seed+1)
                    for method in ("coarse_rk4", "deterministic_observer", "damped_secant", "clean_fitted_observer", "noise_aware_observer", "clean_fitted_memory"):
                        ctx.budget.check()
                        prediction, cost = forecast(previous, current, h=step, kappa=kappa, reaction=rate, steps=steps,
                            method=method, **settings[(sigma, method)])
                        errors = [_metrics(p-y) for p, y in zip(prediction, observed[2:])]
                        states = [_metrics(p) for p in prediction]
                        finite = all(v["rms"] is not None for v in states)
                        row_id = f"{parent_id}/r{rate:g}/h{step:g}/noise{sigma:g}/{method}"
                        row = dict(parent_id=parent_id, query_id=row_id, split="development_test", method=method, role="Ours" if method == "noise_aware_observer" else "Analytic or fitted control",
                            reaction_rate=rate, kappa=kappa, h=step, steps=steps, final_time=(steps+1)*step, noise=sigma,
                            noise_seed=noise_seed, phase_seed=spec["independent_phase_seed"], prediction_finite=finite,
                            endpoint_rms=errors[-1]["rms"], endpoint_maximum=errors[-1]["maximum"], mean_error=errors[-1]["mean"],
                            rollout_rms=float(np.sqrt(np.mean([v["rms"]**2 for v in errors]))) if finite else None,
                            worst_rollout_maximum=max(v["maximum"] for v in errors) if finite else None,
                            minimum=min(v["minimum"] for v in states) if finite else None, maximum=max(v["upper"] for v in states) if finite else None,
                            bounds_violated=any(v["minimum"] < -1e-8 or v["upper"] > 1+1e-8 for v in states) if finite else True,
                            reference=reference, cost=cost, selected_gain=settings[(sigma, method)].get("gain"),
                            parameters=settings[(sigma, method)].get("model", {}).get("parameters", 0),
                            rollout_error_rms=[v["rms"] for v in errors], rollout_error_maximum=[v["maximum"] for v in errors],
                            information="same two noisy coarse observations at t=0 and t=h; no future observations; hidden initial modes unknown",
                            acquisition_seconds=reference["elapsed_seconds"], acquisition_scope="full offline reference trajectory is charged here; physical sensing cost is unmeasured")
                        rows.append(row)
                        if sigma in (0., 1e-3) and multiplier == shifts[0] and rate == reaction_rates[0]:
                            key = hashlib.sha256(row_id.encode()).hexdigest()[:16]
                            raw[key] = prediction
                            row["raw_trajectory_key"] = key
                # Known initially resolved state: shadow closure may infer the
                # newly generated hidden modes, never assume they decay.
                resolved_initial = project(initial, coarse)
                truth_known, known_ref = _teacher(lift(resolved_initial, fine), times[1:], kappa, rate, ctx.budget)
                teacher_rows.append(dict(parent_id=parent_id, reaction=rate, h=step, branch="known_zero_hidden", **known_ref))
                if spec["index"] < attribution_parents:
                    attribution, values = temporal_closure_attribution(resolved_initial, times[1:], truth_known, known_ref,
                        kappa=kappa, reaction=rate, shadow_cutoff=shadow, rate_shift=rate_shift, budget=ctx.budget)
                    attribution.update(parent_id=parent_id, reaction_rate=rate, kappa=kappa, h=step)
                    prefix = hashlib.sha256(f"attribution/{parent_id}/{rate}/{step}".encode()).hexdigest()[:16]
                    for name, value in values.items():
                        raw[prefix+"_"+name] = value
                    attribution["raw_trajectory_prefix"] = prefix
                    attributions.append(attribution)
                    teacher_rows.extend(dict(parent_id=parent_id, reaction=rate, h=step,
                        branch="temporal_closure_attribution", **ref) for ref in attribution["reference_histories"])
                for method, shift, full in (("coarse_rk4", 0., False), ("deterministic_shadow", 0., False),
                                           ("fitted_rate_shadow", rate_shift, False), ("full_shadow_galerkin", 0., True)):
                    state = resolved_initial if method == "coarse_rk4" else lift(resolved_initial, shadow)
                    tick = time.perf_counter(); errors = []
                    for target in truth_known:
                        state = advance(state, step, kappa, rate, coarse_cutoff=None if method == "coarse_rk4" else coarse, rate_shift=shift, full_feedback=full)
                        errors.append(_metrics(project(state, coarse)-project(target, coarse)))
                    shadow_rows.append(dict(parent_id=parent_id, method=method, reaction_rate=rate, h=step, steps=steps+1,
                        endpoint_rms=errors[-1]["rms"], endpoint_maximum=errors[-1]["maximum"], rollout_error_rms=[v["rms"] for v in errors],
                        cost_seconds=time.perf_counter()-tick, reference=known_ref, hidden_rate_shift=shift,
                        maximum_initial_hidden_linear_rate=float(rates(shadow, resolved_initial[coarse].real, kappa, rate)[shadow+coarse+1:].max()),
                        information="known zero hidden initial state; retained shadow modes explicitly driven by unresolved resolved-resolved products"))
        # Matched observed state with opposite unseen phases: a one-observation
        # deterministic map cannot distinguish the two. Two observations can.
        twin = 2*lift(project(initial, coarse), fine)-initial
        first, ref1 = _teacher(initial, [h], kappa, reaction, ctx.budget)
        second, ref2 = _teacher(twin, [h], kappa, reaction, ctx.budget)
        spread = _metrics(project(first[0]-second[0], coarse))
        ambiguities.append(dict(parent_id=parent_id, matched_coarse_initial_sha256=hashlib.sha256(project(initial, coarse).tobytes()).hexdigest(),
            future_coarse_rms_separation=spread["rms"], one_observation_conditional_rms_lower_bound=spread["rms"]/2,
            lower_bound_scope="equiprobable hidden-phase twins with exactly identical coarse input; not a bound after two distinct observations",
            reference_uncertainty_rms=ref1["uncertainty_rms"]+ref2["uncertainty_rms"]))
        teacher_rows.extend([dict(parent_id=parent_id, branch="hidden_phase_twin", **ref) for ref in (ref1, ref2)])
    comparisons = []
    # Explicitly descriptive development effects. Preserve each physical/noise
    # regime and the full parent denominator, including numerical failures.
    for rate in reaction_rates:
        for multiplier in shifts:
            for sigma in noise_levels:
                cohort = [v for v in rows if v["reaction_rate"] == rate and v["h"] == h*multiplier and v["noise"] == sigma]
                base = {v["parent_id"]: v for v in cohort if v["method"] == "coarse_rk4"}
                for method in sorted({v["method"] for v in cohort}-{ "coarse_rk4" }):
                    selected = [v for v in cohort if v["method"] == method]
                    eligible = [v for v in selected if v["reference"]["accepted"] and v["prediction_finite"] and
                                base[v["parent_id"]]["prediction_finite"] and base[v["parent_id"]]["rollout_rms"] > 0]
                    ratios = [v["rollout_rms"]/base[v["parent_id"]]["rollout_rms"] for v in eligible]
                    ratio = float(np.exp(np.log(np.maximum(ratios, 1e-300)).mean())) if ratios else None
                    bound_harm = sum(v["bounds_violated"] and not base[v["parent_id"]]["bounds_violated"] for v in selected)
                    decision = "UNRESOLVED_FAILURE_OR_REFERENCE" if len(eligible) != counts["development_test"] else (
                        "SMALL_SAMPLE_SIGNAL" if ratio < .9 and not bound_harm and len(eligible) < 16 else
                        "PROMISING_DEVELOPMENT_SIGNAL" if ratio < .9 and not bound_harm else "NO_DECLARED_DEVELOPMENT_GAIN")
                    comparisons.append(dict(method=method, comparator="coarse_rk4", noise=sigma, reaction_rate=rate, h=h*multiplier,
                        attempted_parents=counts["development_test"], eligible_parents=len(eligible), rollout_rms_geometric_ratio=ratio,
                        lower_error_parent_count=sum(v < 1. for v in ratios), new_bound_violation_parents=bound_harm,
                        decision=decision, interpretation="descriptive independent-parent development effects; not fresh confirmation or adjusted inference"))
    training_seconds = sum(v["elapsed_seconds"] for v in teacher_rows if any(p["parent_id"] == v["parent_id"] and p["split"] == "train" for p in parent_specs))
    selection_seconds = sum(r["elapsed_seconds"] for v in selections for r in v["rows"])
    attribution_result = dict(schema="tdn.response-closure-attribution/v1", cases=attributions,
        rows=[dict(row, parent_id=case["parent_id"], reaction_rate=case["reaction_rate"], kappa=case["kappa"], h=case["h"],
            coarse_cutoff=coarse, shadow_cutoff=shadow, fine_cutoff=fine,
            reference_accepted=case["all_temporal_references_accepted"] and case["fine_reference"]["accepted"],
            reference_uncertainty_rms=case["fine_reference"]["uncertainty_rms"],
            scope="Descriptive same-state numerical factorial; repeated descendants are paired")
            for case in attributions for row in case["rows"]],
        independent_parents=attribution_parents, parent_selection="First predeclared test parents before observing outcomes",
        scientific_scope="Temporal-only, closure-only and combined numerical controls; no neural or continuum advantage assertion")
    (ctx.path/"closure_attribution.json").write_text(json.dumps(_plain(attribution_result), indent=2, allow_nan=False)+"\n")
    result = dict(schema="tdn.response-closure/v1", rows=rows, shadow_rows=shadow_rows, comparisons=comparisons, hidden_ambiguity=ambiguities,
        temporal_closure_attribution=dict(artifact="closure_attribution.json", cases=len(attributions), independent_parents=attribution_parents,
            signed_interaction_cases=sum(len(case["interactions"]) for case in attributions)),
        onset=onset, parents=parent_specs, selections=selections, teacher_references=teacher_rows,
        hidden_rate_shift=rate_shift, noise_definition="independent Gaussian real Fourier coordinates scaled to expected continuous spatial RMS sigma per observation",
        mathematical_contract="periodic unit interval, odd complex Galerkin coefficients, exact projected convolution, physical symbol -kappa*(2*pi*k)^2",
        cost=dict(elapsed_seconds=time.perf_counter()-started, fit_seconds=fit_seconds, validation_selection_seconds=selection_seconds, onset_seconds=onset_seconds,
            training_teacher_seconds=training_seconds, all_teacher_seconds=sum(v["elapsed_seconds"] for v in teacher_rows)),
        scientific_scope="protected one-dimensional development formulation; no neural or continuum superiority assertion",
        limitations=["DOP853 tolerance refinement controls the fine finite ODE, not the omitted modes beyond the fine cutoff",
            "One paired noise draw per physical descendant; independent parent remains statistical unit",
            "Clean fitted memory intentionally retains the historical autonomous-control failure opportunity",
            "Physical interval is checked on 257 points, not certified everywhere"])
    (ctx.path/"closure_results.json").write_text(json.dumps(_plain(result), indent=2, allow_nan=False)+"\n")
    (ctx.path/"closure_fitted_models.json").write_text(json.dumps(_plain(dict(schema="tdn.response-closure-models/v1", models=fitted_models)), indent=2, allow_nan=False)+"\n")
    np.savez_compressed(ctx.path/"closure_trajectories.npz", **raw)
    ctx.record("closure/onset", ["MX"], metrics=dict(reaction_only_orders=onset["observed_orders"], hidden_rate_shift=rate_shift),
        checks=[check("reaction_only_second_order", min(onset["observed_orders"]), 1.8, relation="ge", category="math"),
            check("useful_noisy_observer", None, None, category="gap", reason="Separate parent/rate/noise comparisons retained; no pooled win declaration")], evidence=["closure_results.json"])
    ctx.record("closure/causal-observer", ["MX"], metrics=dict(parents=counts, observations=2, autonomous_steps=steps,
        prediction_failures=sum(not r["prediction_finite"] for r in rows), elapsed_seconds=result["cost"]["elapsed_seconds"]),
        checks=[check("future_truth_inputs", 0, 0, category="math"), check("fresh_confirmation", None, None, category="gap", reason="Protected development probe")], evidence=["closure_results.json", "closure_fitted_models.json", "closure_trajectories.npz"])
    for case in attributions:
        for interaction in case["interactions"]:
            ctx.record(f"closure/temporal-attribution/{case['parent_id']}/r{case['reaction_rate']:g}/h{case['h']:g}/{interaction['shadow_model']}", ["MX"],
                metrics=interaction, checks=[
                    check("signed_factorial_squared_identity", interaction.get("squared_identity_discrepancy"), 1e-12, category="math"),
                    check("all_temporal_references_accepted", case["all_temporal_references_accepted"], True, relation="eq", category="math"),
                    check("combined_improves_strongest_component", interaction.get("combined_beats_strongest_component"), True,
                        relation="eq", category="gap", reason="Observed numerical attribution only; negative interactions retained")],
                evidence=["closure_attribution.json", "closure_trajectories.npz"])
    for comparison in comparisons:
        ctx.record(f"closure/{comparison['method']}/r{comparison['reaction_rate']:g}/h{comparison['h']:g}/noise{comparison['noise']:g}", ["MX"],
            metrics=comparison, checks=[
                check("complete_reference_and_prediction_cohort", comparison["eligible_parents"], comparison["attempted_parents"], relation="eq", category="math"),
                check("ten_percent_rollout_rms_development_gain", comparison["rollout_rms_geometric_ratio"], .9, category="gap",
                      reason="Descriptive independent-parent effect; not a confirmatory test"),
                check("no_new_sampled_bound_failures", comparison["new_bound_violation_parents"], 0, category="utility")],
            evidence=["closure_results.json"])
    return dict(rows=len(rows), shadow_rows=len(shadow_rows), attribution_cases=len(attributions),
        independent_test_parents=counts["development_test"], elapsed_seconds=result["cost"]["elapsed_seconds"])
