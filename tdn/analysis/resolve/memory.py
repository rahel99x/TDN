"""RX: causal, noise-aware coarse memory on fresh 1D Galerkin parents.

This is a development formulation probe.  It does not assert a continuum
closure, neural advantage, or benefit on the main two-dimensional benchmark.
The fine equation is projected only for observation and scoring; deployment
receives two coarse startup observations and subsequently evolves autonomously.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from tdn.analysis.advance import exploration as old
from tdn.analysis.frontier.core import check


METHODS = ("coarse_rk4", "plain_secant", "validated_damped_secant",
           "clean_fitted_memory", "noise_aware_fitted_memory")


def noisy_observations(previous, current, sigma, seed):
    """Independent coordinate noise at each of two causal observation times.

    Sigma is the standard deviation of each independent real Fourier coordinate,
    not pointwise physical RMS.  Conjugate symmetry is preserved exactly.
    """
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("Observation noise must be finite and nonnegative")
    rng = np.random.default_rng(seed)
    return tuple(np.asarray(a) + old.complex_coordinates(rng.normal(0., sigma, len(a)))
                 for a in (previous, current))


def forecast(previous, current, *, h, diffusion, reaction, steps, method,
             model=None, gain=1., decay=0.):
    """Predict with no reference/teacher argument or fresh observation access.

    The deterministic correction is gain*exp(-decay*r*t) times the previous
    one-step coarse defect.  It includes plain secant (1,0) and RK4 (gain=0).
    Fitted controls use precisely the historical causal state/secant features.
    """
    if method not in METHODS or h <= 0 or steps < 1:
        raise ValueError("Unknown memory method or invalid forward schedule")
    if not 0 <= gain <= 1 or decay < 0:
        raise ValueError("Deterministic damping must be nonnegative and at most one")
    if method.endswith("fitted_memory") and model is None:
        raise ValueError("Fitted memory needs a frozen model")
    prev, state = np.array(previous, copy=True), np.array(current, copy=True)
    trajectory = np.full((steps, len(state)), np.nan, dtype=complex)
    seconds = np.zeros(steps)
    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(steps):
            tick = time.perf_counter()
            base = old.rk4(state, h, diffusion, reaction)
            if method == "coarse_rk4":
                prediction = base
            elif method in ("plain_secant", "validated_damped_secant"):
                weight = 1. if method == "plain_secant" else gain*np.exp(-decay*reaction*k*h)
                prediction = base + weight*(state-old.rk4(prev, h, diffusion, reaction))
            else:
                features = old._features(state, prev, h, diffusion, reaction, True)
                prediction = base + h*old.complex_coordinates(old._predict(model, features))
            seconds[k] = time.perf_counter()-tick
            trajectory[k] = prediction
            prev, state = state, prediction
            if not np.isfinite(state).all():
                break
    return trajectory, seconds


def select_damping(validation, gains, decays, *, h, diffusion, reaction, steps, budget=None):
    """Select only on supplied, explicitly labeled validation trajectories."""
    if not validation or any(v.get("split") != "validation" for v in validation):
        raise ValueError("Damping selection requires validation parents only")
    candidates = []
    for gain in gains:
        for decay in decays:
            if budget is not None:
                budget.check()
            loss = _validation_loss(validation, "validated_damped_secant", h=h,
                diffusion=diffusion, reaction=reaction, steps=steps, gain=gain, decay=decay)
            candidates.append(dict(gain=float(gain), decay=float(decay), validation_log_rms_ratio=loss))
    best = min(candidates, key=lambda v: (v["validation_log_rms_ratio"], v["gain"], v["decay"]))
    return dict(gain=best["gain"], decay=best["decay"]), candidates


def _validation_loss(validation, method, **kwargs):
    values = []
    for item in validation:
        prediction, _ = forecast(item["previous"], item["current"], method=method, **kwargs)
        if not np.isfinite(prediction).all():
            return float("inf")
        # Every validation parent/noise scenario has equal weight. All horizon
        # points contribute; sigma cannot dominate just because it has units.
        error = np.sqrt(np.mean(np.sum(abs(prediction-item["truth"])**2, axis=1)))
        values.append(np.log((error+1e-12)/(item["coarse_trajectory_rms"]+1e-12)))
    return float(np.mean(values))


def _select_fit(training, validation, ridges, settings, *, clean):
    candidates, models = [], []
    for ridge in ridges:
        model = old._ridge([v["features"] for v in training], [v["target"] for v in training], ridge)
        if clean:
            # Historical recipe: clean one-step regression selected on clean
            # one-step validation, retained even if its rollout behavior loses.
            loss = float(np.mean([np.mean((old._predict(model, v["features"])-v["target"])**2)
                                  for v in validation]))
        else:
            loss = _validation_loss(validation, "noise_aware_fitted_memory", model=model,
                h=settings["step"], diffusion=settings["diffusion"], reaction=settings["reaction"],
                steps=settings["rollout_steps"])
        candidates.append(dict(ridge=float(ridge), validation_loss=loss))
        models.append(model)
    index = min(range(len(candidates)), key=lambda i: (candidates[i]["validation_loss"], candidates[i]["ridge"]))
    return models[index], candidates


def _settings(ctx):
    full = ctx.protocol.get("profile") == "full"
    settings = dict(seed=96000000, train_parents=32 if full else 4,
        validation_parents=12 if full else 2, test_parents=12 if full else 2,
        max_mode=8, cutoff=2, step=.02, diffusion=.02, reaction=.7,
        start_time=.1, rollout_steps=32 if full else 6, noise_levels=[0., 1e-4, 1e-3],
        noise_replicates=2 if full else 1, ridges=[1e-10, 1e-6, 1e-2, 1.],
        damp_gains=[0., .25, .5, .75, 1.], decay_rates=[0., 1., 4.],
        rate_multipliers=[1., 3.], reference_tolerance=1e-9)
    settings.update(ctx.protocol.get("resolve", {}).get("memory", {}))
    if settings["cutoff"] < 1 or settings["max_mode"] <= settings["cutoff"]:
        raise ValueError("Memory needs unresolved fine modes and a positive coarse cutoff")
    if not 0 < settings["step"] <= settings["start_time"]:
        raise ValueError("Startup history must have nonnegative forward time")
    if any(settings[key] < 1 or settings[key] >= 1000 for key in
           ("train_parents", "validation_parents", "test_parents", "noise_replicates", "rollout_steps")):
        raise ValueError("Memory counts must fit the disjoint bounded seed blocks")
    if settings["diffusion"] < 0 or settings["reaction"] <= 0:
        raise ValueError("Memory physics requires nonnegative diffusion and positive reaction")
    if not {0., 1e-4, 1e-3}.issubset(settings["noise_levels"]):
        raise ValueError("Memory must retain clean, 1e-4 and 1e-3 noise controls")
    if any(v < 0 or not math.isfinite(v) for v in settings["noise_levels"]):
        raise ValueError("Memory noise must be finite and nonnegative")
    if not settings["rate_multipliers"] or any(v <= 0 for v in settings["rate_multipliers"]):
        raise ValueError("Reaction shifts must be positive")
    return settings


def _teacher(parent, settings, reaction, ledger, split):
    h, start = settings["step"], settings["start_time"]
    horizon = max(start+.12+h, start+settings["rollout_steps"]*h)
    initial = old.field(parent, settings["max_mode"])
    sol, seconds = old.integrate(initial, horizon, diffusion=settings["diffusion"], reaction=reaction)
    record = dict(parent_id=parent, split=split, reaction=reaction, seconds=seconds,
                  nfev=int(sol.nfev), horizon=horizon, purpose="training" if split == "train" else "validation" if split == "validation" else "evaluation")
    if split == "development_test":
        fine, finer_seconds = old.integrate(initial, horizon, diffusion=settings["diffusion"],
            reaction=reaction, rtol=2e-13, atol=2e-15)
        times = start+np.arange(settings["rollout_steps"]+1)*h
        uncertainty = float(np.max(np.linalg.norm(sol.sol(times)-fine.sol(times), axis=0)))
        independent, independent_seconds = old.rk4_reference(initial, horizon,
            settings["diffusion"], reaction, steps=512)
        independent_error = float(np.linalg.norm(independent-fine.sol(horizon)))
        record.update(refinement_seconds=finer_seconds, independent_rk4_seconds=independent_seconds,
            independent_rk4_steps=512, reference_uncertainty=max(uncertainty, independent_error))
    # This is separately timed, forward startup acquisition. It never integrates
    # beyond deployment start. Other reference generation is an offline cost.
    startup, startup_seconds = old.integrate(initial, start, diffusion=settings["diffusion"], reaction=reaction)
    record["startup_seconds"] = startup_seconds
    ledger.append(record)
    return sol, startup, record


def _observation_seed(seed, parent, replicate, observation_index=0):
    # Independent from field-phase RNG. All methods and all sigma values use
    # the same standardized draw for a given parent/replicate/observation.
    return int(np.random.SeedSequence([seed, parent, replicate, observation_index, 0x5258]).generate_state(1)[0])


def _regression_examples(sol, parent, split, settings, noise_levels):
    h, cutoff = settings["step"], settings["cutoff"]
    examples = []
    for observation_index, at in enumerate(settings["start_time"]+np.array([0., .06, .12])):
        previous, current, future = [old.project(sol.sol(at+shift), cutoff) for shift in (-h, 0., h)]
        for sigma in noise_levels:
            for rep in range(settings["noise_replicates"]):
                seed = _observation_seed(settings["seed"], parent, rep, observation_index+1)
                p, c = noisy_observations(previous, current, sigma, seed)
                examples.append(dict(parent_id=parent, split=split, noise=sigma, noise_seed=seed,
                    history_time=at-h, observed_time=at, target_time=at+h,
                    features=old._features(c, p, h, settings["diffusion"], settings["reaction"], True),
                    target=old.real_coordinates((future-old.rk4(c, h, settings["diffusion"], settings["reaction"]))/h)))
    return examples


def run(ctx):
    """Execute a bounded RX unit; all selection completes before test teachers."""
    if str(ctx.device) != "cpu":
        raise ValueError("The protected memory prototype requires explicit CPU execution")
    settings = _settings(ctx)
    ctx.path.mkdir(parents=True, exist_ok=True)
    splits = {split: [settings["seed"]+offset+i for i in range(settings[key])]
        for split, key, offset in (("train", "train_parents", 0), ("validation", "validation_parents", 1000),
                                  ("development_test", "test_parents", 2000))}
    raw = dict(schema="tdn.resolve-memory/v1", settings=settings, parent_ids=splits, rows=[],
        teacher_records=[], training_rows=[], selection={}, startup_observations=[],
        scope="Fresh 1D finite Galerkin development only; no confirmatory or 2D solver claim",
        target="projected fine Galerkin dynamics; closure differs from solving the coarse Galerkin equation",
        uncertainty="Temporal refinement plus independent RK4 cross-check; estimate, not a certificate",
        noise_units="Standard deviation per independent real Fourier coordinate; same pair per method",
        causality="Two forward startup observations, then autonomous predictions; no future truth refresh",
        cost_aggregation="Unique teacher ledger entries once; repeated row startup/fit costs are not additive")
    train_clean, train_noisy, validation_clean, validation_rollouts = [], [], [], []
    h, cutoff = settings["step"], settings["cutoff"]
    common = dict(h=h, diffusion=settings["diffusion"], reaction=settings["reaction"], steps=settings["rollout_steps"])
    for split in ("train", "validation"):
        for parent in splits[split]:
            ctx.budget.check()
            sol, startup, _ = _teacher(parent, settings, settings["reaction"], raw["teacher_records"], split)
            clean = _regression_examples(sol, parent, split, settings, [0.])
            if split == "train":
                train_clean.extend(clean)
                train_noisy.extend(_regression_examples(sol, parent, split, settings, settings["noise_levels"]))
            else:
                validation_clean.extend(clean)
                truth = np.array([old.project(sol.sol(settings["start_time"]+k*h), cutoff)
                                  for k in range(1, settings["rollout_steps"]+1)])
                for sigma in settings["noise_levels"]:
                    for rep in range(settings["noise_replicates"]):
                        seed = _observation_seed(settings["seed"], parent, rep)
                        p, c = noisy_observations(old.project(startup.sol(settings["start_time"]-h), cutoff),
                            old.project(startup.sol(settings["start_time"]), cutoff), sigma, seed)
                        baseline, _ = forecast(p, c, method="coarse_rk4", **common)
                        validation_rollouts.append(dict(parent_id=parent, split=split, noise=sigma, noise_seed=seed,
                            previous=p, current=c, truth=truth,
                            coarse_trajectory_rms=float(np.sqrt(np.mean(np.sum(abs(baseline-truth)**2, axis=1))))))
    tick = time.perf_counter()
    damping, raw["selection"]["validated_damped_secant"] = select_damping(validation_rollouts,
        settings["damp_gains"], settings["decay_rates"], budget=ctx.budget, **common)
    costs = {"validated_damped_secant": time.perf_counter()-tick}
    models = {}
    for method, training, validation, clean in (("clean_fitted_memory", train_clean, validation_clean, True),
            ("noise_aware_fitted_memory", train_noisy, validation_rollouts, False)):
        ctx.budget.check(); tick = time.perf_counter()
        models[method], raw["selection"][method] = _select_fit(training, validation, settings["ridges"], settings, clean=clean)
        costs[method] = time.perf_counter()-tick
    raw.update(models=models, selected_damping=damping, fit_and_selection_seconds=costs,
        training_rows=train_noisy, selection_parent_ids=splits["validation"],
        selection_objectives={"clean_fitted_memory": "Historical clean one-step validation MSE",
            "noise_aware_fitted_memory": "Mean log trajectory RMS ratio to coarse RK4 on all validation noise levels",
            "validated_damped_secant": "Same noisy validation trajectory objective as noise-aware memory"})
    frozen = dict(models=models, damping=damping, selection=raw["selection"], parent_ids=splits)
    old._write(ctx.path/"memory_selection.json", frozen)
    raw["selection_sha256"] = hashlib.sha256((ctx.path/"memory_selection.json").read_bytes()).hexdigest()
    # Test truth is generated only now, after immutable selection is written.
    for multiplier in settings["rate_multipliers"]:
        reaction = settings["reaction"]*multiplier
        for parent in splits["development_test"]:
            ctx.budget.check()
            sol, startup, teacher = _teacher(parent, settings, reaction, raw["teacher_records"], "development_test")
            times = settings["start_time"]+np.arange(1, settings["rollout_steps"]+1)*h
            targets = np.array([old.project(sol.sol(at), cutoff) for at in times])
            for sigma in settings["noise_levels"]:
                for rep in range(settings["noise_replicates"]):
                    seed = _observation_seed(settings["seed"], parent, rep)
                    previous, current = noisy_observations(old.project(startup.sol(settings["start_time"]-h), cutoff),
                        old.project(startup.sol(settings["start_time"]), cutoff), sigma, seed)
                    scenario = f"reaction-{multiplier:g}/noise-{sigma:g}"
                    observation_id = f"{parent}/{scenario}/rep-{rep}"
                    raw["startup_observations"].append(dict(observation_id=observation_id, parent_id=parent,
                        noise_seed=seed, previous=previous, current=current, targets=targets, times=times))
                    for method in METHODS:
                        ctx.budget.check()
                        kwargs = dict(model=models.get(method))
                        if method == "validated_damped_secant":
                            kwargs.update(damping)
                        prediction, seconds = forecast(previous, current, h=h, diffusion=settings["diffusion"],
                            reaction=reaction, steps=settings["rollout_steps"], method=method, **kwargs)
                        cumulative = 0.
                        for step_id, (state, target, elapsed) in enumerate(zip(prediction, targets, seconds), 1):
                            cumulative += elapsed
                            metrics = old.error_metrics(state, target)
                            raw["rows"].append(dict(parent_id=parent, split="development_test", scenario=scenario,
                                observation_id=observation_id, noise_replicate=rep, noise_seed=seed,
                                method=method, rollout_step=step_id, h=h, hr=h*reaction,
                                reaction=reaction, reaction_multiplier=multiplier, noise=sigma,
                                state=state, **metrics, inference_seconds=elapsed,
                                cumulative_inference_seconds=cumulative, startup_seconds=teacher["startup_seconds"],
                                complete_seconds=teacher["startup_seconds"]+cumulative,
                                reference_uncertainty=teacher["reference_uncertainty"]))
    raw["offline_costs"] = dict(
        unique_teacher_and_reference_seconds=sum(v["seconds"]+v.get("refinement_seconds", 0.)+
            v.get("independent_rk4_seconds", 0.) for v in raw["teacher_records"]),
        unique_startup_acquisition_seconds=sum(v["startup_seconds"] for v in raw["teacher_records"]),
        all_fit_and_selection_seconds=sum(costs.values()),
        unmeasured="Device power, monetary rate and whole scheduler allocation cost; prototype has no deployment claim")
    old._write(ctx.path/"memory_results.json", raw)
    summaries = []
    for scenario in sorted({v["scenario"] for v in raw["rows"]}):
        final = [v for v in raw["rows"] if v["scenario"] == scenario and v["rollout_step"] == settings["rollout_steps"]]
        paired = {(v["parent_id"], v["noise_replicate"], v["method"]): v for v in final}
        for method in METHODS:
            selected = [v for v in final if v["method"] == method]
            whole_rollout = [v for v in raw["rows"] if v["scenario"] == scenario and v["method"] == method]
            def ratios(control):
                return [v["error_rms"]/max(paired[(v["parent_id"], v["noise_replicate"], control)]["error_rms"], 1e-15)
                        for v in selected]
            # Repeat draws remain nested in parent. These are descriptive
            # medians, not confidence intervals or extra independent fields.
            metrics = dict(independent_parents=len(splits["development_test"]), paired_noise_replicates=settings["noise_replicates"],
                endpoint_rms_median=float(np.median([v["error_rms"] for v in selected])),
                endpoint_max_worst=float(np.max([v["error_max"] for v in selected])),
                endpoint_mean_error_median=float(np.median([v["mean_error"] for v in selected])),
                rollout_rms_worst=float(np.max([v["error_rms"] for v in whole_rollout])),
                rollout_max_worst=float(np.max([v["error_max"] for v in whole_rollout])),
                rollout_mean_error_worst=float(np.max([v["mean_error"] for v in whole_rollout])),
                endpoint_ratio_to_coarse=float(np.median(ratios("coarse_rk4"))),
                endpoint_ratio_to_damped=float(np.median(ratios("validated_damped_secant"))),
                nonfinite_trajectories=sum(not v["finite"] for v in selected),
                observed_interval_violation=max(v["observed_interval_violation"] for v in whole_rollout),
                inference_seconds_median=float(np.median([v["cumulative_inference_seconds"] for v in selected])),
                startup_seconds_median=float(np.median([v["startup_seconds"] for v in selected])),
                complete_seconds_median=float(np.median([v["complete_seconds"] for v in selected])),
                fit_and_selection_seconds=costs.get(method, 0.),
                parameters=int(models[method]["weights"].size) if method in models else 2 if method == "validated_damped_secant" else 0,
                reference_uncertainty=max(v["reference_uncertainty"] for v in selected))
            checks = [check("temporal-reference-cross-check", metrics["reference_uncertainty"], settings["reference_tolerance"], category="math"),
                check("finite-autonomous-rollout", metrics["nonfinite_trajectories"], 0, "eq", category="math"),
                check("twenty-percent-rms-reduction-vs-coarse", metrics["endpoint_ratio_to_coarse"], .8,
                    applicable=method != "coarse_rk4", category="gap"),
                check("ten-percent-rms-reduction-vs-validated-damping", metrics["endpoint_ratio_to_damped"], .9,
                    applicable=method == "noise_aware_fitted_memory", category="gap"),
                check("complete-deployment-cost-advantage", None, None, category="utility",
                    reason="Development only; startup, selection and teachers recorded but no accuracy-qualified deployment workload")]
            identity = f"RX/{scenario}/{method}"
            role = "Ours" if method == "noise_aware_fitted_memory" else "Fitted control" if method == "clean_fitted_memory" else "Analytic control"
            ctx.record(identity, ["RX"], metrics=old._plain(metrics), checks=checks,
                config=dict(method=method, role=role, scenario=scenario, selection_sha256=raw["selection_sha256"],
                    scope=raw["scope"]), evidence=["memory_results.json", "memory_selection.json"])
            summaries.append(dict(method=method, scenario=scenario, metrics=old._plain(metrics)))
    old._write(ctx.path/"memory_summary.json", summaries)
    return dict(status="COMPLETED", prototype_id="RX", independent_test_parents=len(splits["development_test"]),
                rows=len(raw["rows"]), scenarios=len(settings["rate_multipliers"])*len(settings["noise_levels"]),
                scientific_outcome="DEVELOPMENT_EVIDENCE_ONLY", evidence=["memory_results.json", "memory_selection.json", "memory_summary.json"])
