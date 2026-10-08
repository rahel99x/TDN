"""Frozen, individually timed solver policies and function-level risk audit.

Acceptance reads states, physics and a sealed calibration only. Confirmation
truth is opened after each decision. Both the empirical operational policy and
the finite-sample conformal policy are retained; insufficient conformal samples
produce NA coverage and classical fallback, never a made-up finite bound.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from tdn.numerics import Equation, Geometry
from tdn.numerics.operators import rhs
from tdn.analysis.agenda.physics import diffusion_first_step
from tdn.research.experiment import horizon_key
from tdn.runtime.metadata import write_json

from .statistics import (amortization_summary, assert_disjoint_cohorts, conformal_quantile, cost_distribution,
                         independent_function_scores, paired_cluster_bootstrap,
                         selective_risk_summary)

NORMS = ("rms", "max")


def endpoint_difference(a, b):
    delta = a.detach().double() - b.detach().double()
    return {"rms": float(delta.square().mean().sqrt()), "max": float(delta.abs().max())}


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def _timed(function, device):
    _sync(device)
    started = time.perf_counter()
    value = function()
    _sync(device)
    return value, time.perf_counter() - started


def _check(budget):
    if budget is not None:
        budget.check()


def _schedule(horizon, steps):
    return [float(horizon) / steps] * steps


def rollout(model, initial, schedule, equation, geometry, budget=None):
    value = initial
    with torch.no_grad():
        for step in schedule:
            _check(budget)
            value = model(value, step, equation, geometry).detach()
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError("Nonfinite proposed trajectory")
    return value


class BareDiffusionFirst(torch.nn.Module):
    def forward(self, value, step, equation, geometry):
        return diffusion_first_step(value, step, equation, geometry)

    def architecture_metadata(self):
        return {"family": "df_base", "trainable_parameters": 0, "known_physical_solver": True}


def interior_residual_indicator(model, initial, schedule, equation, geometry, budget=None):
    """Stability-weighted 2/4-node residual quadrature with refinement diagnostic.

    On the bounded nodal FD logistic equation the log norm is <= reaction_rate.
    The reconstruction must stay in [0,1]. Quadrature error and unsampled state
    excursions are not bounded here, so this remains an empirical indicator.
    Identity jumps are charged explicitly; a wrong-generator semigroup is not
    silently approved because its compositions happen to agree.
    """
    horizon = sum(schedule)
    estimates = {}
    samples_in_domain = True
    jump_estimates = {norm: 0. for norm in NORMS}
    for nodes in (2, 4):
        xi, wi = np.polynomial.legendre.leggauss(nodes)
        quadrature = tuple(zip((xi + 1) / 2, wi / 2))
        total = {norm: 0. for norm in NORMS}
        state, elapsed = initial.detach(), 0.
        for h in schedule:
            _check(budget)
            with torch.no_grad():
                zero = model(state, 0., equation, geometry)
            jumps = endpoint_difference(zero, state)
            for norm in NORMS:
                total[norm] += math.exp(equation.reaction_rate * (horizon - elapsed)) * jumps[norm]
                if nodes == 4:
                    jump_estimates[norm] += jumps[norm]
            for node, weight in quadrature:
                _check(budget)
                step = torch.tensor(h * float(node), device=state.device, dtype=state.dtype)
                with torch.enable_grad():
                    value, derivative = torch.autograd.functional.jvp(
                        lambda query: model(state, query, equation, geometry), step,
                        torch.ones_like(step), create_graph=False)
                samples_in_domain = samples_in_domain and bool(((value >= -1e-7) & (value <= 1 + 1e-7)).all())
                defect = derivative.detach() - rhs(value.detach(), equation, geometry)
                norms = endpoint_difference(defect, torch.zeros_like(defect))
                growth = math.exp(equation.reaction_rate * (horizon - elapsed - float(step)))
                for norm in NORMS:
                    total[norm] += h * float(weight) * growth * norms[norm]
            with torch.no_grad():
                state = model(state, h, equation, geometry).detach()
            elapsed += h
        estimates[nodes] = total
    discrepancy = {norm: abs(estimates[4][norm] - estimates[2][norm]) for norm in NORMS}
    return {norm: estimates[4][norm] + 2 * discrepancy[norm] for norm in NORMS}, {
        "quadrature_nodes": [2, 4], "quadrature_refinement_difference": discrepancy,
        "identity_jump": jump_estimates, "sampled_reconstruction_in_domain": samples_in_domain,
        "log_norm_upper": equation.reaction_rate,
        "scope": "empirical stability-weighted residual quadrature for the nodal FD equation",
        "deterministic_certificate": False,
        "unproved_requirements": ["quadrature remainder bound", "between-node admissible reconstruction"],
        "jvp_calls": 6 * len(schedule)}


def spatial_discrepancy(model, initial, schedule, equation, geometry, sampler, budget=None):
    """Compare a fresh 2N sample of the same continuous function to the N solve.

    Full discrepancy is used without a Richardson /3 assumption. Calibration
    against continuum truth is separate. This is not a deterministic bound.
    """
    fine_grid = tuple(2 * n for n in geometry.grid)
    fine_initial = sampler(fine_grid).to(device=initial.device, dtype=initial.dtype)
    fine = rollout(model, fine_initial, schedule, equation, Geometry(fine_grid, geometry.lengths), budget)
    coarse = rollout(model, initial, schedule, equation, geometry, budget)
    restriction = (slice(None), slice(None)) + (slice(None, None, 2),) * geometry.ndim
    return endpoint_difference(fine[restriction], coarse), {
        "coarse_grid": list(geometry.grid), "fine_grid": list(fine_grid),
        "same_continuous_function": True, "richardson_order_assumed": None,
        "deterministic_certificate": False, "fine_steps": len(schedule), "coarse_recomputed_steps": len(schedule)}


def estimate_attempt(model, initial, schedule, equation, geometry, *, estimator="step_doubling",
                     spatial_guard=False, sampler=None, budget=None):
    """Execute exactly the estimator requested and record all component costs."""
    started = time.perf_counter()
    value, proposal_seconds = _timed(lambda: rollout(model, initial, schedule, equation, geometry, budget), initial.device)
    if estimator == "step_doubling":
        fine, estimator_seconds = _timed(lambda: rollout(model, initial,
            [h / 2 for h in schedule for _ in range(2)], equation, geometry, budget), initial.device)
        temporal, diagnostics = endpoint_difference(value, fine), {
            "richardson_order_assumed": None, "deterministic_certificate": False,
            "estimator_extra_steps": 2 * len(schedule)}
    elif estimator == "interior_residual":
        (temporal, diagnostics), estimator_seconds = _timed(
            lambda: interior_residual_indicator(model, initial, schedule, equation, geometry, budget), initial.device)
    else:
        raise ValueError(f"Unknown policy estimator: {estimator}")
    spatial, spatial_seconds, spatial_diagnostics = {norm: 0. for norm in NORMS}, 0., None
    if spatial_guard:
        if sampler is None:
            raise ValueError("A spatial guard requires the immutable continuous-function sampler")
        (spatial, spatial_diagnostics), spatial_seconds = _timed(
            lambda: spatial_discrepancy(model, initial, schedule, equation, geometry, sampler, budget), initial.device)
    total = {norm: temporal[norm] + spatial[norm] for norm in NORMS}
    if any(not math.isfinite(v) for v in total.values()):
        raise FloatingPointError("Nonfinite policy indicator")
    return value, {"temporal": temporal, "spatial": spatial, "total": total,
                   "temporal_diagnostics": diagnostics, "spatial_diagnostics": spatial_diagnostics}, {
        "proposal_seconds": proposal_seconds, "estimator_seconds": estimator_seconds,
        "spatial_seconds": spatial_seconds, "attempt_elapsed_seconds": time.perf_counter() - started,
        "proposal_steps": len(schedule)}


def calibrate_envelope(rows, *, alpha=.01, safety_factor=1.1):
    """One joint norm/query maximum per independent function, before quantile."""
    if safety_factor < 1 or not math.isfinite(safety_factor):
        raise ValueError("Empirical safety factor must be finite and at least one")
    scores = []
    for row in rows:
        if row.get("split") != "calibration":
            raise ValueError("Only the calibration cohort may fit a policy")
        ratio = max(float(row["upper_error"][norm]) / max(float(row["indicator"][norm]), 1e-12) for norm in NORMS)
        scores.append({"field_cluster": row["field_cluster"], "score": ratio})
    functions = independent_function_scores(scores)
    if not functions:
        raise ValueError("Calibration requires at least one independent function")
    quantile = conformal_quantile([row["score"] for row in functions], alpha)
    return {"conformal": quantile, "empirical_multiplier": max(1., max((r["score"] for r in functions), default=1.)) * safety_factor,
            "function_scores": functions, "query_rows": len(rows), "calibration_only": True,
            "empirical_scope": "maximum observed joint error/indicator ratio times declared safety factor; no finite-sample guarantee",
            "score_scope": "joint maximum over the frozen calibration query bank and both error norms"}


def acceptance_decision(value, indicator, envelope, targets, *, calibration_mode="empirical"):
    if calibration_mode == "conformal":
        multiplier = envelope["conformal"]["quantile"]
    elif calibration_mode == "empirical":
        multiplier = envelope["empirical_multiplier"]
    else:
        raise ValueError("Unknown calibration mode")
    if multiplier is None:
        return False, {"reason": "INSUFFICIENT_INDEPENDENT_FUNCTIONS", "calibrated_upper": None}
    if calibration_mode == "conformal" and not envelope["conformal"].get("exchangeability_supported", True):
        return False, {"reason": "EXCHANGEABILITY_NOT_ESTABLISHED", "calibrated_upper": None,
                       "scope": "finite quantile alone does not transfer across coefficient or field-distribution shift"}
    bound = {norm: float(multiplier) * max(float(indicator[norm]), 1e-12) for norm in NORMS}
    physical = bool(torch.isfinite(value).all()) and bool(((value >= -1e-7) & (value <= 1 + 1e-7)).all())
    decision = physical and all(math.isfinite(bound[norm]) and bound[norm] <= targets[norm] for norm in NORMS)
    return bool(decision), {"reason": "ACCEPTED" if decision else "INDICATOR_OR_PHYSICAL_REJECTION",
                            "calibrated_upper": bound, "physical_sample_admissible": physical,
                            "calibration_mode": calibration_mode, "deterministic_certificate": False}


def continuum_fallback_solve(initial, horizon, equation, geometry, substeps, *, budget=None):
    """Device-native Lawson RK4 for the deployed dealiased continuum control.

    The independent reference uses SciPy resampling on CPU. Deployment instead
    uses the tested Torch Fourier interpolation, including even-grid Nyquist
    splitting, so a CUDA decision keeps its fallback work on the same device.
    Both discretizations use >3/2-padded quadratic products and the same RK
    stages. This operational solve is not an independent reference or a bound.
    """
    from tdn.numerics.operators import check_shape
    from .numerics import dealiased_product

    check_shape(initial, geometry)
    if type(substeps) is not int or substeps < 1:
        raise ValueError("Continuum fallback requires a positive integer substep count")
    if not math.isfinite(float(horizon)) or horizon < 0:
        raise ValueError("Continuum fallback requires a finite nonnegative horizon")
    dt = float(horizon) / substeps
    eigenvalues = torch.zeros(geometry.grid, dtype=initial.dtype, device=initial.device)
    for axis, (n, dx) in enumerate(zip(geometry.grid, geometry.dx)):
        frequency = 2 * torch.pi * torch.fft.fftfreq(n, d=dx, dtype=initial.dtype, device=initial.device)
        shape = [1] * geometry.ndim
        shape[axis] = n
        eigenvalues = eigenvalues - equation.kappa * frequency.square().reshape(shape)
    full, half = torch.exp(dt * eigenvalues), torch.exp(dt * eigenvalues / 2)
    axes = tuple(range(2, initial.ndim))

    def flow(value, multiplier):
        return torch.fft.ifftn(torch.fft.fftn(value, dim=axes) * multiplier, dim=axes).real

    def nonlinear(value):
        return equation.reaction_rate * (value - dealiased_product(value, value))

    value = initial
    with torch.no_grad():
        for index in range(substeps):
            if index % 8 == 0:
                _check(budget)
            k1 = nonlinear(value)
            k2 = nonlinear(flow(value + dt * k1 / 2, half))
            k3 = nonlinear(flow(value, half) + dt * k2 / 2)
            k4 = nonlinear(flow(value, full) + dt * flow(k3, half))
            value = flow(value, full) + dt / 6 * (flow(k1, full) + 2 * flow(k2 + k3, half) + k4)
        _check(budget)
    return value


def adaptive_classical(initial, horizon, equation, geometry, targets, *, track="discrete", sampler=None,
                       max_refinements=5, budget=None):
    """A bounded operational refinement controller, independent of truth.

    Refines time from 1 to 2^k steps, using undivided step-doubling differences.
    Continuum mode advances the dealiased spectral equation on 2N/4N samples;
    it audits a spatial discrepancy too. A small indicator remains empirical.
    All trial/rejected solves are performed and charged, not fixed eightfold work.
    """
    if type(max_refinements) is not int or max_refinements < 1:
        raise ValueError("Classical refinement cap must be a positive integer")
    all_attempts = []
    started = time.perf_counter()
    last = initial
    for level in range(max_refinements):
        _check(budget)
        count = 2 ** level
        if track == "discrete":
            model = BareDiffusionFirst()
            coarse = rollout(model, initial, _schedule(horizon, count), equation, geometry, budget)
            last = rollout(model, initial, _schedule(horizon, 2 * count), equation, geometry, budget)
            temporal = endpoint_difference(last, coarse)
            spatial = {norm: 0. for norm in NORMS}
        elif track == "continuum":
            if sampler is None:
                raise ValueError("Continuum classical controller needs a continuous sampler")
            fine_grid = tuple(2 * n for n in geometry.grid)
            fine_initial = sampler(fine_grid).to(device=initial.device, dtype=initial.dtype)
            high_grid = tuple(4 * n for n in geometry.grid)
            high_initial = sampler(high_grid).to(device=initial.device, dtype=initial.dtype)
            fine_geometry = Geometry(fine_grid, geometry.lengths)
            high_geometry = Geometry(high_grid, geometry.lengths)
            coarse = continuum_fallback_solve(fine_initial, horizon, equation, fine_geometry, count, budget=budget)
            fine = continuum_fallback_solve(fine_initial, horizon, equation, fine_geometry, 2 * count, budget=budget)
            high = continuum_fallback_solve(high_initial, horizon, equation, high_geometry, 2 * count, budget=budget)
            r2 = (slice(None), slice(None)) + (slice(None, None, 2),) * geometry.ndim
            r4 = (slice(None), slice(None)) + (slice(None, None, 4),) * geometry.ndim
            last = high[r4]
            temporal = endpoint_difference(fine[r2], coarse[r2])
            spatial = endpoint_difference(last, fine[r2])
        else:
            raise ValueError("Unknown classical target track")
        physical = bool(torch.isfinite(last).all()) and bool(((last >= -1e-7) & (last <= 1 + 1e-7)).all())
        accepted = physical and all(temporal[norm] + spatial[norm] <= targets[norm] * .5 for norm in NORMS)
        all_attempts.append({"level": level, "coarse_steps": count, "fine_steps": 2 * count,
                             "temporal": temporal, "spatial": spatial, "indicator_accepted": accepted})
        if accepted:
            break
    _sync(initial.device)
    return last, {"controller_status": "INDICATOR_ACCEPTED" if accepted else "REFINEMENT_CAP_EXHAUSTED",
                  "controller_accepted": bool(accepted), "refinements": len(all_attempts), "attempts": all_attempts,
                  "elapsed_seconds": time.perf_counter() - started, "track": track,
                  "solver": "torch_dealiased_lawson_rk4" if track == "continuum" else "diffusion_first_strang",
                  "compute_device": str(initial.device), "compute_dtype": str(initial.dtype),
                  "deterministic_certificate": False, "truth_used_in_decision": False}


def deploy_policy(models, model_order, initial, horizon, equation, geometry, envelopes, targets, *,
                  estimator="step_doubling", spatial_guard=True, routing=True, calibration_mode="empirical",
                  step_sizes=(.1, .05), max_attempts=2, max_refinements=5, track="continuum", sampler=None, budget=None):
    """Standalone policy with no shared proposal/fallback work across policies."""
    _sync(initial.device)
    if str(initial.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(initial.device)
    started = time.perf_counter()
    route_started = time.perf_counter()
    order = list(model_order)
    if not routing:
        order = order[-1:]  # Frozen single proposed solver, no cheap-bank screening.
    if not order:
        raise ValueError("The policy solver bank is empty")
    route_seconds = time.perf_counter() - route_started
    attempts = []
    selected, final, last_failure = None, None, None
    for attempt_index, step_size in enumerate(tuple(step_sizes)[:max_attempts]):
        count = max(1, math.ceil(horizon / float(step_size) - 1e-12))
        schedule = _schedule(horizon, count)
        for model_id in order:
            _check(budget)
            envelope = envelopes[model_id]
            if model_id not in models or envelope.get("unavailable", False):
                attempts.append({"model_id": model_id, "attempt": attempt_index, "status": "VALIDATED_CHECKPOINT_UNAVAILABLE",
                                 "charged_seconds": 0., "decision_seconds": 0., "accepted": False})
                continue
            # A missing formal rank is knowable before an expensive proposal.
            # Still execute and charge the fallback below, as a real deployment.
            if calibration_mode == "conformal" and (envelope["conformal"]["quantile"] is None or not envelope["conformal"].get("exchangeability_supported", True)):
                attempts.append({"model_id": model_id, "attempt": attempt_index, "status": ("NO_FINITE_CONFORMAL_QUANTILE" if envelope["conformal"]["quantile"] is None else "EXCHANGEABILITY_NOT_ESTABLISHED"),
                                 "charged_seconds": 0., "decision_seconds": 0., "accepted": False})
                continue
            attempt_started = time.perf_counter()
            try:
                value, indicators, costs = estimate_attempt(models[model_id], initial, schedule, equation, geometry,
                    estimator=estimator, spatial_guard=spatial_guard, sampler=sampler, budget=budget)
                decision_started = time.perf_counter()
                accepted, decision = acceptance_decision(value, indicators["total"], envelope, targets, calibration_mode=calibration_mode)
                decision_seconds = time.perf_counter() - decision_started
                attempts.append({"model_id": model_id, "attempt": attempt_index, "schedule": schedule,
                    "accepted": accepted, "status": decision["reason"], "indicators": indicators, "decision": decision,
                    "costs": costs, "decision_seconds": decision_seconds,
                    "charged_seconds": time.perf_counter() - attempt_started})
                if accepted:
                    selected, final = model_id, value
                    break
            except (FloatingPointError, OverflowError) as error:
                _sync(initial.device)
                last_failure = str(error)
                attempts.append({"model_id": model_id, "attempt": attempt_index, "status": "NUMERICAL_FAILURE",
                    "accepted": False, "error": last_failure, "charged_seconds": time.perf_counter() - attempt_started})
        if final is not None:
            break
    fallback, fallback_seconds = None, 0.
    if final is None:
        (final, fallback), fallback_seconds = _timed(lambda: adaptive_classical(initial, horizon, equation, geometry,
            targets, track=track, sampler=sampler, max_refinements=max_refinements, budget=budget), initial.device)
    _sync(initial.device)
    elapsed = time.perf_counter() - started
    proposal = sum(a.get("costs", {}).get("proposal_seconds", 0.) for a in attempts)
    estimator_cost = sum(a.get("costs", {}).get("estimator_seconds", 0.) + a.get("costs", {}).get("spatial_seconds", 0.) for a in attempts)
    rejected = sum(a["charged_seconds"] for a in attempts if not a["accepted"])
    return final, {"accepted": selected is not None, "selected_model": selected, "fallback": selected is None,
        "attempts": attempts, "fallback_details": fallback, "standalone_seconds": elapsed,
        "proposal_seconds": proposal, "estimator_seconds": estimator_cost, "route_seconds": route_seconds,
        "fallback_seconds": fallback_seconds, "rejected_work_seconds": rejected,
        "rejected_attempts": sum(not a["accepted"] for a in attempts),
        "accounted_component_seconds": sum(a["charged_seconds"] for a in attempts) + fallback_seconds + route_seconds,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(initial.device) if str(initial.device).startswith("cuda") else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(initial.device) if str(initial.device).startswith("cuda") else None,
        "unattributed_loop_overhead_seconds": max(0., elapsed - sum(a["charged_seconds"] for a in attempts) - fallback_seconds - route_seconds),
        "timing_scope": "standalone end-to-end invocation; proposals, route, estimates, rejected work and fallback all included",
        "truth_used_in_decision": False, "calibration_mode": calibration_mode, "last_numerical_failure": last_failure}


def _upper_error(value, reference):
    """Compare predictions to the intact FP64 teacher with explicit metadata.

    Downcasting the teacher first can erase an actual sub-FP32 error. Missing
    uncertainty or acceptance is not evidence of a perfect accepted teacher.
    """
    if reference.get("accepted") is not True:
        raise ValueError("Policy calibration/audit requires explicit accepted=True on an independent reference")
    teacher = reference.get("state")
    if not isinstance(teacher, torch.Tensor) or teacher.dtype != torch.float64:
        raise ValueError("Policy calibration/audit requires an FP64 reference state")
    if teacher.shape != value.shape or not bool(torch.isfinite(teacher).all()):
        raise ValueError("Policy reference state must be finite and match the predicted shape")
    uncertainty = {}
    for norm, key in (("rms", "uncertainty_rms"), ("max", "uncertainty_max_bound")):
        if key not in reference or reference[key] is None or isinstance(reference[key], bool):
            raise ValueError(f"Policy reference requires explicit finite nonnegative {key}")
        try:
            number = float(reference[key])
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"Policy reference requires explicit finite nonnegative {key}") from error
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"Policy reference requires explicit finite nonnegative {key}")
        uncertainty[norm] = number
    difference = endpoint_difference(value, teacher.to(device=value.device, dtype=torch.float64))
    return {"error": difference, "uncertainty": uncertainty,
            "upper": {norm: difference[norm] + uncertainty[norm] for norm in NORMS},
            "lower": {norm: max(0., difference[norm] - uncertainty[norm]) for norm in NORMS},
            "reference_uncertainty_is_certificate": False}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _parent_subset(parents, max_clusters=None, max_parents=None):
    if max_clusters is not None:
        allowed = sorted({str(p["field_cluster"]) for p in parents})[:int(max_clusters)]
        parents = [p for p in parents if str(p["field_cluster"]) in allowed]
    return list(parents[:int(max_parents)]) if max_parents is not None else list(parents)


def _model_family(identifier, model):
    try:
        metadata = model.architecture_metadata()
    except AttributeError:
        metadata = {}
    return str(metadata.get("family", identifier.split("/")[0]))


def run(ctx):
    """Run M07/M08/M09/M18 and the preregistered C2 ablation matrix."""
    from .core import check
    from .data import load_bank
    from .neural import load_frozen_models
    from tdn.analysis.agenda.data import continuous_field

    path = Path(ctx.path)
    path.mkdir(parents=True, exist_ok=True)
    if any((path / name).exists() for name in ("calibration.json", "calibration-progress.jsonl", "policy-details.jsonl")):
        raise FileExistsError("Preserve existing policy evidence and start a fresh stage")
    opts = ctx.protocol.get("policies", {})
    if "train" not in ctx.prerequisites:
        raise ValueError("The policy stage requires the sealed training prerequisite")
    setup_started = time.perf_counter()
    available = load_frozen_models(ctx)
    allowed_families = opts.get("model_families", ["source", "rank2", "cheap_fno"])
    seed_limit = int(opts.get("model_seed_limit", 1))
    if seed_limit < 1:
        raise ValueError("At least one preregistered model seed slot is required")
    models = {"df_base": BareDiffusionFirst().to(ctx.device)}
    families = defaultdict(list)
    requested_seeds = [int(seed) for seed in ctx.protocol["seeds"][:seed_limit]]
    requested_slots = [{"family": family, "seed": seed, "model_id": f"{family}-seed-{seed}"}
                       for family in allowed_families for seed in requested_seeds]
    eligible_by_slot = {}
    for identifier, model in sorted(available.items()):
        family = _model_family(identifier, model)
        selection = getattr(model, "selection_metadata", {})
        seed = selection.get("seed")
        if seed is None and "-seed-" in identifier:
            seed = int(identifier.rsplit("-seed-", 1)[1])
        if family not in allowed_families or seed not in requested_seeds:
            continue
        slot = (family, int(seed))
        if slot in eligible_by_slot:
            raise ValueError("Multiple validated checkpoints occupy the same frozen policy seed slot")
        eligible_by_slot[slot] = identifier
        families[family].append(identifier)
        models[identifier] = model
    missing_slots = [slot for slot in requested_slots if (slot["family"], slot["seed"]) not in eligible_by_slot]
    for slot in missing_slots:
        ctx.record(f"unavailable-policy-slot/{slot['model_id']}", ["M06", "M07", "M09", "M10"], combination_ids=["C2"],
            metrics={"model_id": slot["model_id"], "available": False},
            checks=[check("requested_validated_checkpoint", None, True, relation="eq", category="gap", applicable=False,
                          reason="This exact preregistered family/seed has no validated checkpoint; another seed cannot replace it"),
                    check("learned_policy_mathematics", None, True, relation="eq", category="math", applicable=False,
                          reason="Missing learned predictions supply no mathematical evidence")],
            config=slot, status="NOT_APPLICABLE", evidence={"fallback": "actual adaptive classical controller retained for every missing variant"})
    setup_seconds = time.perf_counter() - setup_started
    targets = {"rms": float(opts.get("rms_target", 2e-4)), "max": float(opts.get("max_target", 2e-4))}
    if any(not math.isfinite(x) or x <= 0 for x in targets.values()):
        raise ValueError("Policy targets must be positive and finite")
    grids = [int(n[0] if isinstance(n, (list, tuple)) else n) for n in opts.get("grids", [ctx.protocol["train_grid"]])]
    default_h = sum(ctx.protocol.get("confirm_schedules", [[.1]])[0])
    horizons = [float(h) for h in opts.get("horizons", [default_h])]
    estimator_names = list(opts.get("estimators", ["step_doubling", "interior_residual"]))
    tracks = list(opts.get("tracks", ["discrete", "continuum"]))
    step_sizes = tuple(float(h) for h in opts.get("attempted_step_sizes", [.1, .05]))
    if not step_sizes or any(h <= 0 or not math.isfinite(h) for h in step_sizes):
        raise ValueError("Policy step sizes must be positive and finite")
    max_attempts = int(opts.get("max_attempts", 2))
    max_refinements = int(opts.get("max_refinements", opts.get("fallback_max_refinements", 5)))
    if max_attempts < 1:
        raise ValueError("Policy attempt cap must be positive")
    alpha = float(opts.get("alpha", .01))
    calibration = _parent_subset(load_bank(ctx, "calibration"), opts.get("max_calibration_clusters"), opts.get("max_calibration_parents"))
    train = load_bank(ctx, "train")
    validation = load_bank(ctx, "validation")
    split_check = assert_disjoint_cohorts(train=train, validation=validation, calibration=calibration)
    calibration_started = time.perf_counter()
    calibration_rows = []
    envelopes = {}
    # Both raw error tracks are retained. The temporal-only ablation calibrates
    # on the discrete target; it deliberately cannot claim continuum coverage.
    for identifier, model in models.items():
        for estimator in estimator_names:
            for spatial_guard in (False, True):
                own = []
                for parent in calibration:
                    equation = Equation(parent["kappa"], parent["reaction_rate"])
                    for n in grids:
                        geometry = Geometry((n, n), (1., 1.))
                        initial = parent["states"][str(n)].to(device=ctx.device, dtype=torch.float32)
                        sampler = lambda grid, parent=parent: continuous_field(parent, grid)
                        for horizon in horizons:
                            for step_size in step_sizes[:max_attempts]:
                                ctx.budget.check()
                                schedule = _schedule(horizon, max(1, math.ceil(horizon / step_size - 1e-12)))
                                value, indicators, costs = estimate_attempt(model, initial, schedule, equation, geometry,
                                    estimator=estimator, spatial_guard=spatial_guard, sampler=sampler, budget=ctx.budget)
                                track = "continuum" if spatial_guard else "discrete"
                                reference_key = f"{track}:{n}:{horizon_key(horizon)}"
                                if reference_key not in parent["references"]:
                                    raise ValueError(f"Calibration reference missing for frozen query {reference_key}")
                                audit = _upper_error(value, parent["references"][reference_key])
                                own.append({"model_id": identifier, "estimator": estimator, "spatial_guard": spatial_guard,
                                    "parent_id": parent["parent_id"], "field_cluster": str(parent["field_cluster"]),
                                    "split": "calibration", "grid": n, "horizon": horizon, "schedule": schedule,
                                    "track": track, "indicator": indicators["total"], "upper_error": audit["upper"],
                                    "temporal_indicator": indicators["temporal"], "spatial_indicator": indicators["spatial"], "costs": costs})
                                with (path / "calibration-progress.jsonl").open("a") as calibration_log:
                                    calibration_log.write(json.dumps(own[-1], allow_nan=False, separators=(",", ":")) + "\n")
                                    calibration_log.flush()
                key = f"{identifier}|{estimator}|spatial={int(spatial_guard)}"
                envelopes[key] = calibrate_envelope(own, alpha=alpha, safety_factor=float(opts.get("empirical_safety_factor", opts.get("safety_factor", 1.1))))
                calibration_rows.extend(own)
                fitted = envelopes[key]
                ctx.record(f"calibration/{identifier}/{estimator}/spatial-{int(spatial_guard)}", ["M08", "M09", "M18"], combination_ids=["C2"],
                    metrics={"model_id": identifier, "query_rows": len(own),
                             "independent_functions": fitted["conformal"]["independent_functions"],
                             "empirical_multiplier": fitted["empirical_multiplier"], "parameters": sum(p.numel() for p in model.parameters()),
                             "observed_max_upper_rms": max(r["upper_error"]["rms"] for r in own),
                             "observed_max_upper_max": max(r["upper_error"]["max"] for r in own),
                             "attempt_seconds": sum(r["costs"]["attempt_elapsed_seconds"] for r in own)},
                    checks=[check("independent_calibration_function_count", fitted["conformal"]["independent_functions"], 2, relation="ge", category="math"),
                            check("calibration_is_not_confirmation_evidence", None, True, relation="eq", category="gap", applicable=False,
                                  reason="Fitting an envelope cannot establish held-out accuracy or population coverage"),
                            check("fit_uses_only_calibration_split", all(r["split"] == "calibration" for r in own), True, relation="eq", category="correctness")],
                    config={"estimator": estimator, "spatial_guard": spatial_guard, "alpha": alpha,
                            "empirical_safety_factor": float(opts.get("empirical_safety_factor", opts.get("safety_factor", 1.1)))},
                    evidence={"observations": "calibration-progress.jsonl", "formal_proof": False})
    # A data-dependent router can choose among every bank model. Separate
    # model-wise q_alpha values do not imply coverage of the selected result.
    # Freeze one joint maximum over all declared candidates/estimators/steps.
    joint_functions = independent_function_scores([
        score for envelope in envelopes.values() for score in envelope["function_scores"]])
    joint_quantile = conformal_quantile([score["score"] for score in joint_functions], alpha)
    joint_quantile["exchangeability_supported"] = bool(opts.get("exchangeability_assumed", False))
    joint_quantile["exchangeability_scope"] = "User-declared distributional assumption only; roadmap includes unseen physics and distribution shifts, so default is unsupported"
    joint_quantile["numerical_score_coverage_assumptions_met"] = joint_quantile["quantile"] is not None and joint_quantile["exchangeability_supported"]
    joint_quantile["teacher_error_bound_certified"] = False
    joint_quantile["guarantee_available"] = False
    joint_quantile["teacher_scope"] = "Calibration uses reference-refinement uncertainty estimates, not proved error bounds; no certified PDE error coverage"
    for envelope in envelopes.values():
        envelope["per_model_conformal_diagnostic_only"] = envelope["conformal"]
        envelope["conformal"] = joint_quantile
        envelope["router_joint_scope"] = "maximum over every frozen bank model, estimator, spatial guard, step, grid, horizon and norm"
    calibration_seconds = time.perf_counter() - calibration_started
    model_costs = {identifier: float(np.median([r["costs"]["proposal_seconds"] for r in calibration_rows if r["model_id"] == identifier]))
                   for identifier in models}
    bank_order = sorted(models, key=lambda identifier: (model_costs[identifier], identifier))
    # Candidate is frozen by declared family order/seed, never fresh labels.
    default_candidate = next((families[f][0] for f in allowed_families if families.get(f)), bank_order[-1])
    serialized = {"schema": "tdn.roadmap.policy-calibration/v1", "envelopes": envelopes, "rows": calibration_rows,
        "calibration_parent_ids": [p["parent_id"] for p in calibration],
        "calibration_field_clusters": sorted({str(p["field_cluster"]) for p in calibration}),
        "disjoint_development_cohorts": split_check, "frozen_model_ids": list(models), "model_families": dict(families),
        "requested_model_slots": requested_slots, "unavailable_model_slots": missing_slots,
        "selection_metadata": {identifier: getattr(model, "selection_metadata", {}) for identifier, model in models.items()},
        "model_order": bank_order, "default_candidate": default_candidate, "order_basis": "calibration proposal median time only",
        "grids": grids, "horizons": horizons, "targets": targets, "alpha": alpha,
        "joint_router_function_scores": joint_functions, "joint_router_conformal_quantile": joint_quantile,
        "calibration_seconds": calibration_seconds, "model_setup_seconds": setup_seconds,
        "calibration_timings_are_shared_setup": True,
        "empirical_safety_factor": float(opts.get("empirical_safety_factor", opts.get("safety_factor", 1.1))), "confirmation_seen_during_calibration": False,
        "unexamined_seed_policies": "Only the preregistered first model_seed_limit seeds; no policy claim for remaining seeds",
        "coverage_scope": "empirical policies have no finite-sample guarantee; conformal requires a finite function-level rank and exchangeability"}
    write_json(path / "calibration.json", serialized)
    calibration_sha = _sha(path / "calibration.json")
    # Do not move this load above serialization. No calibration fit may inspect
    # confirmation states or labels, even if the final label is unfavorable.
    fresh = _parent_subset(load_bank(ctx, "confirmation"), opts.get("max_confirmation_clusters", 6), opts.get("max_confirmation_parents"))
    separation = assert_disjoint_cohorts(train=train, validation=validation, calibration=calibration, confirmation=fresh)
    groups = {identifier: ["df_base", identifier] for identifier in models if identifier != "df_base"}
    groups.update({slot["model_id"]: [slot["model_id"]] for slot in missing_slots})
    groups["solver_bank"] = bank_order
    variants = []
    for group, order in groups.items():
        for estimator in estimator_names:
            for spatial_guard in (False, True):
                for routing in (False, True):
                    variants.append({"group": group, "order": ([default_candidate] if group == "solver_bank" and not routing else order), "estimator": estimator,
                        "spatial_guard": spatial_guard, "routing": routing, "calibration_mode": "empirical"})
    # Execute the formal policy too. At n<99/alpha=.01 it cheaply detects the
    # unavailable quantile, then actually executes and charges classical work.
    variants.append({"group": "solver_bank", "order": bank_order, "estimator": estimator_names[0],
                     "spatial_guard": True, "routing": True, "calibration_mode": "conformal"})
    endpoints = []
    controls = []
    sequence = 0
    stream = path / "policy-details.jsonl"
    if stream.exists():
        raise ValueError("A policy stage cannot append to an existing result stream")
    with stream.open("x") as log:
        for parent in fresh:
            equation = Equation(parent["kappa"], parent["reaction_rate"])
            for n in grids:
                geometry = Geometry((n, n), (1., 1.))
                initial = parent["states"][str(n)].to(device=ctx.device, dtype=torch.float32)
                sampler = lambda grid, parent=parent: continuous_field(parent, grid)
                for horizon in horizons:
                    for track in tracks:
                        key = f"{track}:{n}:{horizon_key(horizon)}"
                        if key not in parent["references"]:
                            if track == "continuum" and parent["parent_id"] not in ctx.protocol.get("continuum_parent_ids", []):
                                ctx.record(f"continuum-not-declared-{sequence}", ["M01", "M09", "M18"], combination_ids=["C2"],
                                    metrics={"parent_id": parent["parent_id"], "track": track, "grid": n, "horizon": horizon},
                                    checks=[check("continuum_reference_scope", None, True, relation="eq", category="gap", applicable=False,
                                                  reason="This parent was outside the predeclared continuum teacher subset"),
                                            check("continuum_total_error_evidence", None, True, relation="eq", category="math", applicable=False,
                                                  reason="No continuum truth is substituted from the discrete target")],
                                    status="NOT_APPLICABLE")
                                sequence += 1
                                continue
                            raise ValueError(f"Confirmation reference missing for frozen query {key}")
                        ctx.budget.check()
                        (control, control_info), control_seconds = _timed(lambda: adaptive_classical(initial, horizon, equation,
                            geometry, targets, track=track, sampler=sampler, max_refinements=max_refinements, budget=ctx.budget), ctx.device)
                        # Baseline accuracy is also examined only after the
                        # operational controller finishes its own decisions.
                        control_audit = _upper_error(control, parent["references"][key])
                        control_good = all(control_audit["upper"][norm] <= targets[norm] for norm in NORMS)
                        common = {"parent_id": parent["parent_id"], "field_cluster": str(parent["field_cluster"]),
                                  "grid": n, "horizon": horizon, "track": track, "regime": parent.get("regime"),
                                  "kappa": equation.kappa, "reaction_rate": equation.reaction_rate}
                        controls.append({**common, "seconds": control_seconds, "joint_accuracy": control_good,
                                         "audit": control_audit, "controller": control_info})
                        ctx.record(f"classical-{sequence}", ["M07", "M08"], combination_ids=["C2"],
                            metrics={**common, "standalone_seconds": control_seconds, "joint_accuracy": control_good,
                                     "upper_rms": control_audit["upper"]["rms"], "upper_max": control_audit["upper"]["max"],
                                     "refinements": control_info["refinements"], "parameters": 0},
                            checks=[check("classical_operational_accuracy", control_good, True, relation="eq", category="gap"),
                                    check("controller_converged", control_info["controller_accepted"], True, relation="eq", category="correctness")],
                            config={"target_track": track, "targets": targets, "max_refinements": max_refinements},
                            evidence={"scope": "empirical adaptive classical controller; no reference used in decisions"})
                        sequence += 1
                        for variant in variants:
                            ctx.budget.check()
                            unavailable_envelope = {"unavailable": True, "empirical_multiplier": None,
                                                    "conformal": {**joint_quantile, "quantile": None}}
                            own_envelopes = {identifier: envelopes.get(f"{identifier}|{variant['estimator']}|spatial={int(variant['spatial_guard'])}", unavailable_envelope)
                                             for identifier in variant["order"]}
                            value, outcome = deploy_policy(models, variant["order"], initial, horizon, equation, geometry,
                                own_envelopes, targets, estimator=variant["estimator"], spatial_guard=variant["spatial_guard"],
                                routing=variant["routing"], calibration_mode=variant["calibration_mode"],
                                step_sizes=step_sizes, max_attempts=max_attempts, max_refinements=max_refinements,
                                track=track, sampler=sampler, budget=ctx.budget)
                            # This is the first access to truth for this policy
                            # invocation; its route/accept/reject decisions have ended.
                            audit = _upper_error(value, parent["references"][key])
                            good = all(audit["upper"][norm] <= targets[norm] for norm in NORMS)
                            variant_id = f"{variant['group']}|{variant['estimator']}|spatial={int(variant['spatial_guard'])}|route={int(variant['routing'])}|{variant['calibration_mode']}"
                            selected_model = outcome["selected_model"]
                            candidate_available = all(identifier in models for identifier in variant["order"])
                            training_failures = {identifier: getattr(models[identifier], "selection_metadata", {}).get("numerical_failure")
                                for identifier in variant["order"] if identifier in models
                                and getattr(models[identifier], "selection_metadata", {}).get("numerical_failure") is not None}
                            selected_parameters = sum(p.numel() for p in models[selected_model].parameters()) if selected_model else 0
                            attempted_model_ids = {attempt["model_id"] for attempt in outcome["attempts"] if "costs" in attempt}
                            parameters = sum(p.numel() for identifier in attempted_model_ids for p in models[identifier].parameters())
                            row = {**common, "experiment_id": f"policy-{sequence}", "variant_id": variant_id, "variant": variant, **outcome,
                                   "false_accept": bool(outcome["accepted"] and not good),
                                   "false_accept_definitely_resolved": bool(outcome["accepted"] and any(audit["lower"][norm] > targets[norm] for norm in NORMS)),
                                   "accuracy_status": "VERIFIED_BY_REFINEMENT_ESTIMATE" if good else "RESOLVED_FAILURE_BY_REFINEMENT_ESTIMATE" if any(audit["lower"][norm] > targets[norm] for norm in NORMS) else "UNRESOLVED",
                                   "false_accept_meaning": "conservative possible false acceptance: estimated upper error exceeds target; see resolved and unresolved labels",
                                   "joint_accuracy": good,
                                   "audit": audit, "calibration_sha256": calibration_sha, "parameters": parameters,
                                   "requested_candidate_available": candidate_available, "prior_training_failures": training_failures,
                                   "fully_requested_solver_bank_available": not missing_slots,
                                   "selected_parameters": selected_parameters, "attempted_model_ids": sorted(attempted_model_ids),
                                   "parameter_count_scope": "unique modules actually proposed; includes rejected proposals; bank residency is separate",
                                   "classical_seconds": control_seconds, "classical_joint_accuracy": control_good,
                                   "cost_over_classical": outcome["standalone_seconds"] / max(control_seconds, 1e-12),
                                   "seed": str(variant["group"]).split("seed-")[-1] if "seed-" in variant["group"] else "frozen_bank"}
                            endpoints.append(row)
                            log.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
                            log.flush()
                            finite = all(e["conformal"]["quantile"] is not None for e in own_envelopes.values())
                            scope_valid = track == "discrete" or variant["spatial_guard"]
                            checks = [check("requested_candidate_checkpoint_available", True if candidate_available else None, True,
                                relation="eq", category="gap", applicable=candidate_available,
                                reason="A missing learned checkpoint is NA; its classical fallback is measured independently"),
                                check("requested_solver_bank_complete", True if not missing_slots else None, True,
                                relation="eq", category="gap", applicable=not missing_slots,
                                reason="Missing preregistered neural slots cannot be replaced by other seeds"),
                                check("prior_training_completed_without_numerical_failure", not training_failures, True,
                                relation="eq", category="correctness", reason="Measuring a retained validated checkpoint does not erase its training failure"),
                                check("conformal_exchangeability_scope", True if joint_quantile["exchangeability_supported"] else None, True,
                                relation="eq", category="math", applicable=variant["calibration_mode"] == "conformal" and joint_quantile["exchangeability_supported"],
                                reason="The protocol includes coefficient/distribution shift; no exchangeability guarantee is established"),
                                check("audited_joint_accuracy", good, True, relation="eq", category="gap"),
                                check("false_acceptance", row["false_accept"], False, relation="eq", category="gap"),
                                check("separate_spatial_target", scope_valid, True, relation="eq", category="math"),
                                check("full_work_faster_than_accurate_classical", row["cost_over_classical"], 1., category="utility",
                                      applicable=bool(control_good and good), reason="Cost comparison requires both outputs to pass the same target"),
                                check("component_times_fit_standalone_total", outcome["accounted_component_seconds"], outcome["standalone_seconds"] * 1.001,
                                      category="correctness"),
                                check("finite_conformal_function_rank", finite if finite else None, True, relation="eq", category="math",
                                      applicable=variant["calibration_mode"] == "conformal" and finite,
                                      reason="Insufficient independent calibration functions produces NA, not a finite guarantee")]
                            ctx.record(f"policy-{sequence}", ["M01", "M06", "M07", "M08", "M09", "M10", "M15", "M18"], combination_ids=["C2"],
                                metrics={**common, "variant_id": variant_id, "accepted": outcome["accepted"], "fallback": outcome["fallback"],
                                    "joint_accuracy": good, "false_accept": row["false_accept"], "parameters": parameters,
                                    "requested_candidate_available": candidate_available, "prior_training_failures": training_failures,
                                    "selected_parameters": selected_parameters, "attempted_model_ids": sorted(attempted_model_ids),
                                    "standalone_seconds": outcome["standalone_seconds"], "cost_over_classical": row["cost_over_classical"],
                                    "peak_allocated_bytes": outcome["peak_allocated_bytes"], "peak_reserved_bytes": outcome["peak_reserved_bytes"],
                                    "unattributed_loop_overhead_seconds": outcome["unattributed_loop_overhead_seconds"],
                                    "bank_parameters": sum(p.numel() for m in models.values() for p in m.parameters()),
                                    "proposal_seconds": outcome["proposal_seconds"], "route_seconds": outcome["route_seconds"],
                                    "estimator_seconds": outcome["estimator_seconds"], "rejected_work_seconds": outcome["rejected_work_seconds"],
                                    "fallback_seconds": outcome["fallback_seconds"], "rejected_attempts": outcome["rejected_attempts"],
                                    "upper_rms": audit["upper"]["rms"], "upper_max": audit["upper"]["max"]},
                                checks=checks, config={**variant, "targets": targets, "step_sizes": list(step_sizes)},
                                evidence={"detail_log": "policy-details.jsonl", "calibration_sha256": calibration_sha,
                                          "formal_proof": False, "uncertainty_is_certificate": False,
                                          "decision_without_truth": True})
                            sequence += 1
    if _sha(path / "calibration.json") != calibration_sha:
        raise ValueError("Frozen calibration was mutated after confirmation")
    grouped = defaultdict(list)
    for row in endpoints:
        grouped[(row["variant_id"], row["track"])].append(row)
    summaries = []
    offline_components = {"model_loading": setup_seconds, "calibration": calibration_seconds}
    offline_source_files = {}
    for stage_name in ("prepare", "train"):
        summary_file = Path(ctx.prerequisites[stage_name]) / "summary.json" if stage_name in ctx.prerequisites else None
        payload = json.loads(summary_file.read_text()) if summary_file is not None and summary_file.is_file() else {}
        offline_components[f"shared_bank_{stage_name}"] = payload.get("elapsed_seconds")
        offline_source_files[stage_name] = str(summary_file) if summary_file is not None else None
    for (variant_id, track), rows in grouped.items():
        risks = selective_risk_summary(rows, target_risk=float(opts.get("target_conditional_risk", .01)))
        cost = cost_distribution([r["standalone_seconds"] for r in rows])
        base_cost = cost_distribution([r["classical_seconds"] for r in rows])
        paired = paired_cluster_bootstrap([{**r, "difference": r["standalone_seconds"] - r["classical_seconds"]} for r in rows],
            repeats=int(opts.get("bootstrap_repeats", 400)))
        summary = {"variant_id": variant_id, "track": track, "risk": risks, "cost": cost, "classical_cost": base_cost,
            "paired_cost_difference_bootstrap": paired, "joint_accuracy_rows": sum(r["joint_accuracy"] for r in rows),
            "total_rows": len(rows), "false_accepts": sum(r["false_accept"] for r in rows),
            "resolved_false_accepts": sum(r["false_accept_definitely_resolved"] for r in rows),
            "unresolved_accuracy_rows": sum(r["accuracy_status"] == "UNRESOLVED" for r in rows),
            "false_accept_meaning": "conservative possible failures against estimated reference upper errors; resolved/unresolved counts are separate",
            "total_cost_over_classical": cost["total_seconds"] / max(base_cost["total_seconds"], 1e-12),
            "p95_cost_over_classical": cost["p95_seconds"] / max(base_cost["p95_seconds"], 1e-12),
            "training_teacher_amortization": amortization_summary(offline_components, cost["mean_seconds"], base_cost["mean_seconds"],
                matched_accuracy=all(r["joint_accuracy"] and r["classical_joint_accuracy"] for r in rows)),
            "offline_source_summaries": offline_source_files}
        summaries.append(summary)
        all_good = all(r["joint_accuracy"] and r["classical_joint_accuracy"] for r in rows)
        ctx.record(f"policy-summary-{sequence}", ["M01", "M06", "M07", "M08", "M09", "M10", "M15", "M18"], combination_ids=["C2"],
            metrics=summary, checks=[
                check("finite_measured_break_even", summary["training_teacher_amortization"]["break_even_queries"], 0,
                      relation="ge", category="utility", applicable=summary["training_teacher_amortization"]["status"] == "FINITE_BREAK_EVEN",
                      reason="Break-even is NA without matched accuracy, complete offline costs and strictly positive per-query savings"),
                check("all_audited_queries_accurate", summary["joint_accuracy_rows"], len(rows), relation="eq", category="gap"),
                check("total_cost_beats_classical", summary["total_cost_over_classical"], 1., category="utility", applicable=all_good),
                check("p95_cost_beats_classical", summary["p95_cost_over_classical"], 1., category="utility", applicable=all_good),
                check("observed_bad_accepted_functions", risks["bad_accepted_functions"], 0, category="gap",
                      applicable=risks["accepted_independent_functions"] > 0),
                check("independent_cluster_risk_bound", risks["one_sided_binomial_upper"], risks["target_risk"], category="math",
                      applicable=bool(opts.get("independent_iid_confirmation", False)) and risks["accepted_independent_functions"] >= risks["zero_failure_functions_needed"],
                      reason="Insufficient independent accepted functions or unestablished iid sampling gives NA for the requested 1% population-risk statement"),
                check("independent_fields_for_inference", paired["independent_fields"], 2, relation="ge", category="math"),
                check("three_paired_training_seed_policy_replication", paired["training_seeds"] if paired["training_seeds"] >= 3 else None,
                      3, relation="ge", category="math", applicable=paired["training_seeds"] >= 3,
                      reason="Bounded policy scope uses frozen first seeds; no population inference over untested seeds")],
            evidence={"calibration_sha256": calibration_sha, "claim_scope": "bounded empirical policy audit, no mathematical proof"})
        sequence += 1
    quantiles = [value["conformal"] for value in envelopes.values()]
    summary = {"status": "COMPLETED", "models": list(models), "policy_variants": len(variants), "endpoints": len(endpoints),
        "requested_model_slots": requested_slots, "unavailable_model_slots": missing_slots,
        "policy_bank_status": "CLASSICAL_ONLY_NO_ELIGIBLE_NEURAL_MODEL" if len(models) == 1 else "PARTIAL_NEURAL_BANK" if missing_slots else "COMPLETE_REQUESTED_BANK",
        "calibration_rows": len(calibration_rows), "calibration_independent_functions": len(serialized["calibration_field_clusters"]),
        "confirmation_independent_functions": len({str(p["field_cluster"]) for p in fresh}), "cohort_separation": separation,
        "conformal_status": "FINITE_QUANTILE" if all(q["quantile"] is not None for q in quantiles) else "INSUFFICIENT_INDEPENDENT_FUNCTIONS",
        "alpha": alpha, "exchangeability_supported": joint_quantile["exchangeability_supported"],
        "conformal_guarantee_available": joint_quantile["guarantee_available"], "model_setup_seconds": setup_seconds, "calibration_seconds": calibration_seconds,
        "calibration_sha256": calibration_sha, "summaries": summaries,
        "false_accepts": sum(r["false_accept"] for r in endpoints),
        "resolved_false_accepts": sum(r["false_accept_definitely_resolved"] for r in endpoints),
        "unresolved_accuracy_rows": sum(r["accuracy_status"] == "UNRESOLVED" for r in endpoints),
        "false_accept_meaning": "conservative possible failures against numerical-reference upper estimates, not proved wrong outputs",
        "offline_components_seconds": offline_components, "offline_source_summaries": offline_source_files,
        "all_rejected_and_fallback_work_charged": True,
        "timing_scope": "separate full standalone policy invocations in one worker process; not independent-process timing replication",
        "formal_proof": False, "empirical_policies_are_certificates": False,
        "incomplete_work": ["99 independent calibration functions required for finite alpha=.01 conformal rank",
                            "299 zero-failure independent accepted functions required for a 95% upper risk bound <=1%",
                            "multiple independent timing processes and full paired-seed policy replication"]}
    write_json(path / "policy-summaries.json", {"summaries": summaries, "controls": controls})
    write_json(path / "policy-summary.json", summary)
    ctx.record("calibration-and-independence", ["M09", "M18"], combination_ids=["C0", "C2"],
        metrics={"calibration_independent_functions": summary["calibration_independent_functions"],
                 "confirmation_independent_functions": summary["confirmation_independent_functions"],
                 "alpha": alpha, "conformal_status": summary["conformal_status"], "calibration_seconds": calibration_seconds,
                 "model_setup_seconds": setup_seconds},
        checks=[check("cohorts_disjoint", separation["disjoint"], True, relation="eq", category="correctness"),
                check("frozen_calibration_unchanged", _sha(path / "calibration.json"), calibration_sha, relation="eq", category="correctness"),
                check("finite_function_conformal_quantile", True if summary["conformal_status"] == "FINITE_QUANTILE" else None,
                      True, relation="eq", category="math", applicable=summary["conformal_status"] == "FINITE_QUANTILE",
                      reason="n<99 at alpha=.01 is unavailable finite coverage, recorded NA")],
        evidence={"calibration_file": "calibration.json", "calibration_sha256": calibration_sha,
                  "formal_proof": False, "marginal_is_not_selective_risk": True})
    return summary


def validate_policy_artifacts(path):
    """Independently recheck policy log arithmetic and decision provenance.

    Recomputes decisions, cohort-quantile counts, errors, risk/counts and full
    cost totals from the retained observations. This checks reporting integrity,
    not whether floating-point teachers are mathematically exact solutions.
    """
    path = Path(path)
    calibration = json.loads((path / "calibration.json").read_text())
    summary = json.loads((path / "policy-summary.json").read_text())
    digest = _sha(path / "calibration.json")
    if digest != summary["calibration_sha256"] or calibration["confirmation_seen_during_calibration"]:
        raise ValueError("Policy calibration seal/provenance changed")
    joint = independent_function_scores([score for envelope in calibration["envelopes"].values()
                                         for score in envelope["function_scores"]])
    expected_quantile = conformal_quantile([score["score"] for score in joint], calibration["alpha"])
    actual_quantile = calibration["joint_router_conformal_quantile"]
    for key in ("quantile", "rank", "independent_functions", "status"):
        if actual_quantile[key] != expected_quantile[key]:
            raise ValueError("Policy function-level joint quantile/count differs")
    details = [json.loads(line) for line in (path / "policy-details.jsonl").read_text().splitlines() if line]
    if len(details) != summary["endpoints"]:
        raise ValueError("Policy endpoint log is incomplete")
    targets = calibration["targets"]
    score_rows = {}
    if (path / "rows.jsonl").is_file():
        from .core import validate_row
        score_rows = {record["experiment_id"]: validate_row(record) for record in
                      (json.loads(line) for line in (path / "rows.jsonl").read_text().splitlines() if line)}
    bound_scored_rows = 0
    seen, groups = set(), defaultdict(list)
    for row in details:
        identity = (row["variant_id"], row["parent_id"], row["grid"], row["horizon"], row["track"])
        if identity in seen:
            raise ValueError("Policy endpoint identity repeated")
        seen.add(identity)
        if row["calibration_sha256"] != digest or row["truth_used_in_decision"]:
            raise ValueError("Policy decision was not based on the sealed calibration")
        if row["accepted"] != (row["selected_model"] is not None) or row["fallback"] == row["accepted"]:
            raise ValueError("Policy accepted/fallback identity differs")
        verified = all(row["audit"]["upper"][norm] <= targets[norm] for norm in NORMS)
        resolved = any(row["audit"]["lower"][norm] > targets[norm] for norm in NORMS)
        if row["joint_accuracy"] != verified or row["false_accept"] != bool(row["accepted"] and not verified):
            raise ValueError("Policy accuracy/false-accept log disagrees with its audit")
        if row["false_accept_definitely_resolved"] != bool(row["accepted"] and resolved):
            raise ValueError("Policy resolved-failure label differs")
        variant = row["variant"]
        for attempt in row["attempts"]:
            if "decision" not in attempt:
                if attempt["accepted"]:
                    raise ValueError("An accepted policy attempt lacks decision evidence")
                continue
            envelope = calibration["envelopes"][f"{attempt['model_id']}|{variant['estimator']}|spatial={int(variant['spatial_guard'])}"]
            multiplier = (envelope["conformal"]["quantile"] if row["calibration_mode"] == "conformal" else envelope["empirical_multiplier"])
            bounds = {norm: float(multiplier) * max(attempt["indicators"]["total"][norm], 1e-12) for norm in NORMS} if multiplier is not None else None
            decision = bool(bounds is not None and attempt["decision"]["physical_sample_admissible"]
                            and all(bounds[norm] <= targets[norm] for norm in NORMS))
            if row["calibration_mode"] == "conformal" and not envelope["conformal"]["exchangeability_supported"]:
                decision = False
            if attempt["accepted"] != decision or bounds != attempt["decision"]["calibrated_upper"]:
                raise ValueError("Policy accept/reject operands changed")
        total = sum(attempt["charged_seconds"] for attempt in row["attempts"]) + row["fallback_seconds"] + row["route_seconds"]
        rejected = sum(attempt["charged_seconds"] for attempt in row["attempts"] if not attempt["accepted"])
        if not math.isclose(total, row["accounted_component_seconds"], rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("Policy total component cost differs")
        if not math.isclose(rejected, row["rejected_work_seconds"], rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("Policy rejected work was omitted")
        if total > row["standalone_seconds"] * 1.001 or row["standalone_seconds"] < 0:
            raise ValueError("Policy component costs exceed its measured full invocation")
        if row.get("experiment_id"):
            scored = score_rows.get(row["experiment_id"])
            if scored is None:
                raise ValueError("Policy detail has no scored experiment log")
            expected_metrics = {key: row[key] for key in ("accepted", "fallback", "joint_accuracy", "false_accept", "parameters",
                "standalone_seconds", "cost_over_classical", "rejected_work_seconds", "fallback_seconds")}
            expected_metrics.update(upper_rms=row["audit"]["upper"]["rms"], upper_max=row["audit"]["upper"]["max"])
            if any(scored["metrics"].get(key) != value for key, value in expected_metrics.items()):
                raise ValueError("Scored policy metrics differ from the detailed observations")
            checks = {item["check_id"]: item for item in scored["checks"]}
            operands = {"audited_joint_accuracy": row["joint_accuracy"], "false_acceptance": row["false_accept"],
                        "full_work_faster_than_accurate_classical": row["cost_over_classical"],
                        "component_times_fit_standalone_total": row["accounted_component_seconds"]}
            if any(checks.get(key, {}).get("measured") != value for key, value in operands.items()):
                raise ValueError("Policy gap/math scoring operands differ from observed audit values")
            bound_scored_rows += 1
        groups[(row["variant_id"], row["track"])].append(row)
    summaries = json.loads((path / "policy-summaries.json").read_text())["summaries"]
    if {(item["variant_id"], item["track"]) for item in summaries} != set(groups):
        raise ValueError("Policy summary groups differ from the actual observations")
    for item in summaries:
        rows = groups[(item["variant_id"], item["track"])]
        observed_risk = selective_risk_summary(rows, confidence=item["risk"]["confidence"], target_risk=item["risk"]["target_risk"])
        if observed_risk != item["risk"] or cost_distribution([r["standalone_seconds"] for r in rows]) != item["cost"]:
            raise ValueError("Policy risk/cost summary differs from raw observations")
    return {"status": "VERIFIED", "endpoints": len(details), "policy_groups": len(groups),
            "calibration_independent_functions": expected_quantile["independent_functions"],
            "scored_experiment_rows_bound_to_observations": bound_scored_rows,
            "calibration_sha256": digest, "verification_scope": "logged arithmetic, frozen decisions, joint function counts and cost inclusion; not a PDE proof"}
