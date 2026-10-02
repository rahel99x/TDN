"""Executable scientific audits and evaluation with explicit evidence boundaries."""
from __future__ import annotations

import json
import hashlib
import math
import time
from pathlib import Path

import numpy as np
import torch

from tdn.numerics import Equation, Geometry, choose_substeps, reference_step, refined_reference, split_step, weighted_norm
from .convergence import configure_plot_cache, fit_order, plot_orders, temporal_oracle_fits, tier_a_audit
from .influence import derivative_audit, same_observation_audit
from .profiling import measure
from .statistics import parent_summary, paired_parent_comparison


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _initial(geometry: Geometry, *, device="cpu") -> torch.Tensor:
    grids = torch.meshgrid(*[torch.arange(n, dtype=torch.float64, device=device) / n
                             for n in geometry.grid], indexing="ij")
    state = torch.full(geometry.grid, .45, dtype=torch.float64, device=device)
    for dimension, coordinate in enumerate(grids):
        state = state + .08 / (dimension + 1) * torch.sin(2 * math.pi * coordinate)
    return state[None, None]


def _parameters(config: dict, *, tiny=False) -> tuple[Equation, Geometry]:
    problem = config["problem"]
    grid = tuple(min(int(n), 8) for n in problem["grid"]) if tiny else tuple(problem["grid"])
    return Equation(float(problem["kappa"]), float(problem["reaction_rate"])), Geometry(grid, tuple(problem["lengths"]))


def _rms(value: torch.Tensor, geometry: Geometry) -> float:
    return float(weighted_norm(value, geometry).detach().cpu())


def _safe_state(state: torch.Tensor) -> None:
    if not torch.isfinite(state).all().item():
        raise FloatingPointError("Nonfinite physical state")
    # Accept roundoff only; never clamp and count an inaccurate state as success.
    slack = 32 * torch.finfo(state.dtype).eps
    if state.min().item() < -slack or state.max().item() > 1 + slack:
        raise FloatingPointError("Logistic physical state outside validated [0,1] interval")


def step_schedule(total_time: float, base_h: float, factors=(1.,)) -> list[float]:
    """Exact physical end time, with one explicit shorter final step."""
    if total_time <= 0 or base_h <= 0 or not factors or any(f <= 0 for f in factors):
        raise ValueError("Positive horizon, macrostep and nonempty factors required")
    steps, current = [], 0.
    while current < total_time:
        h = min(base_h * factors[len(steps) % len(factors)], total_time - current)
        if h <= max(total_time, 1.) * 1e-14:
            break
        steps.append(h)
        current += h
        if len(steps) > 1000000:
            raise ValueError("Schedule exceeds one million macrosteps")
    if steps:
        steps[-1] += total_time - sum(steps)
    return steps


def _rollout(state: torch.Tensor, steps: list[float], equation: Equation, geometry: Geometry,
             method: str, *, model=None, chunk_size=32768, rtol=1e-3,
             atol=1e-5, rk4_multiplier=1) -> tuple[torch.Tensor, dict]:
    cost = {"accepted_macrosteps": 0, "rejected_macrosteps": 0, "split_calls": 0,
            "rk4_substeps": 0, "model_calls": 0, "fallback_steps": 0, "simulated_time": 0.}
    if method == "adaptive_split":
        # Classical step doubling, sequential local error allocation. No teacher
        # access at deployment, and every rejected/accepted solve is charged.
        for requested in steps:
            remaining, h = requested, requested
            while remaining > max(requested, 1.) * 1e-14:
                h = min(h, remaining)
                big = split_step(state, h, equation, geometry, differentiable=False)
                half = split_step(state, h / 2, equation, geometry, differentiable=False)
                small = split_step(half, h / 2, equation, geometry, differentiable=False)
                cost["split_calls"] += 3
                error = _rms((small - big) / 3, geometry)
                allowed = atol * h / sum(steps) + rtol * _rms(small, geometry) * h / sum(steps)
                if error <= allowed or h < requested * 1e-7:
                    if error > allowed:
                        raise FloatingPointError("Adaptive split failed local tolerance at minimum step")
                    state = small
                    _safe_state(state)
                    remaining -= h
                    cost["accepted_macrosteps"] += 1
                    cost["simulated_time"] += h
                    h = min(remaining, h * min(2., max(.5, .9 * (allowed / max(error, 1e-30))**(1 / 3))))
                else:
                    cost["rejected_macrosteps"] += 1
                    h *= max(.1, .9 * (allowed / error)**(1 / 3))
        return state, cost
    for h in steps:
        if method == "split":
            state = split_step(state, h, equation, geometry, differentiable=False)
            cost["split_calls"] += 1
        elif method == "richardson_split":
            full = split_step(state, h, equation, geometry, differentiable=False)
            half = split_step(state, h / 2, equation, geometry, differentiable=False)
            halves = split_step(half, h / 2, equation, geometry, differentiable=False)
            state = (4 * halves - full) / 3
            cost["split_calls"] += 3
        elif method == "coupled_rk4":
            substeps = max(1, choose_substeps(h, equation, geometry)) * rk4_multiplier
            state = reference_step(state, h, equation, geometry, substeps=substeps)
            cost["rk4_substeps"] += substeps
        elif method == "learned":
            from tdn.models.solver import corrected_step
            state = corrected_step(state, h, equation, geometry, model, chunk_size=chunk_size)
            cost["model_calls"] += 1
            cost["split_calls"] += 1
        else:
            raise ValueError(f"Unsupported deployed method {method}")
        _safe_state(state)
        cost["accepted_macrosteps"] += 1
        cost["simulated_time"] += h
    return state, cost


def audit(config: dict, run_dir: Path, device="cpu") -> dict:
    """Run tiny Tier-A/B numerical, derivative, information and temporal audits."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    equation, geometry = _parameters(config, tiny=True)
    state = _initial(geometry, device=device)
    teacher_config = config["teacher"]
    base_substeps = int(teacher_config["base_substeps"])
    noise_floor = float(teacher_config["noise_floor"])
    error_fraction = float(teacher_config["error_fraction"])
    total = min(float(config["horizons"]["rollout_time"]), .64)
    # Predeclared smooth-asymptotic grid; no retrospective two-point fit.
    horizons = total / np.asarray([8, 16, 32, 64], float)
    tier_a = tier_a_audit()
    plot_orders(tier_a["fits"], run_dir / "tier_a_orders.png", title="Exact noncommuting linear control")
    refinements, local_errors, global_errors, classical_errors, uncertainties = [], [], [], [], []
    reference_count = max(base_substeps, choose_substeps(total, equation, geometry))
    reference_T_coarse = reference_step(state, total, equation, geometry, substeps=2 * reference_count)
    reference_T = reference_step(state, total, equation, geometry, substeps=4 * reference_count)
    global_uncertainty = _rms(reference_T - reference_T_coarse, geometry)
    for h in horizons:
        count = max(base_substeps, choose_substeps(float(h), equation, geometry))
        levels = [reference_step(state, float(h), equation, geometry, substeps=factor * count) for factor in (1, 2, 4)]
        differences = [_rms(levels[1] - levels[0], geometry), _rms(levels[2] - levels[1], geometry)]
        base = split_step(state, float(h), equation, geometry, differentiable=False)
        defect = _rms(levels[-1] - base, geometry)
        final_tolerance = float(config["validation"]["tolerance"])
        allowed = min(error_fraction * defect, float(teacher_config["tolerance_fraction"]) * final_tolerance)
        accepted = differences[-1] <= max(noise_floor, allowed)
        refinements.append({"h": float(h), "substeps": [count, 2 * count, 4 * count],
                            "refinement_differences": differences, "defect_norm": defect,
                            "uncertainty_bound_convention": "conservative finest two-level difference, not extrapolated error estimate",
                            "observed_refinement_ratio": differences[0] / max(differences[1], noise_floor),
                            "permitted_uncertainty": allowed, "accepted": accepted,
                            "roundoff_dominated": defect <= 10 * noise_floor})
        uncertainties.append(differences[-1])
        local_errors.append(defect)
        steps = step_schedule(total, float(h))
        full, _ = _rollout(state.clone(), steps, equation, geometry, "split")
        classical, _ = _rollout(state.clone(), steps, equation, geometry, "richardson_split")
        global_errors.append(_rms(full - reference_T, geometry))
        classical_errors.append(_rms(classical - reference_T, geometry))
    logistic_fits = {"split_local": fit_order(horizons, local_errors, noise_floor=noise_floor,
                                             reference_uncertainty=uncertainties),
                    "split_global": fit_order(horizons, global_errors, noise_floor=noise_floor,
                                               reference_uncertainty=[global_uncertainty] * len(horizons)),
                    "richardson_global": fit_order(horizons, classical_errors, noise_floor=noise_floor,
                                                    reference_uncertainty=[global_uncertainty] * len(horizons))}
    plot_orders(logistic_fits, run_dir / "logistic_orders.png", title="Periodic semidiscrete logistic reaction diffusion")
    classical_cost_profile = []
    for operational_h in sorted({float(h) for h in config["horizons"]["evaluation_steps"]}):
        schedule = step_schedule(total, operational_h)
        for method in ("split", "adaptive_split", "coupled_rk4", "richardson_split"):
            row = {"method": method, "h": operational_h, "total_time": total, "dtype": "float64",
                   "failed": False, "error": None}
            try:
                def operation():
                    with torch.inference_mode():
                        return _rollout(state.clone(), schedule, equation, geometry, method,
                                        rtol=0., atol=config["validation"]["tolerance"] * .5)
                (final_state, counters), cost = measure(operation, device=device, warmup=1, repeats=3,
                                                       memory_policy=config["runtime"])
                row.update(error=_rms(final_state - reference_T, geometry), counters=counters, performance=cost)
            except (RuntimeError, ValueError, FloatingPointError) as error:
                row.update(failed=True, failure_reason=f"{type(error).__name__}: {error}")
            classical_cost_profile.append(row)
    largest_h = float(max(config["horizons"]["values"]))
    derivative_count = 4 * max(base_substeps, choose_substeps(largest_h, equation, geometry))
    derivative = derivative_audit(state, equation, geometry, largest_h, substeps=derivative_count, seed=config["seed"])
    information = same_observation_audit(state, equation, geometry, largest_h, substeps=derivative_count,
                                         t_ref=config["problem"]["t_ref"], U_ref=config["problem"]["U_ref"])
    information_regimes = [{"regime": "central", "equation": {"kappa": equation.kappa, "reaction_rate": equation.reaction_rate},
                            "information": information, "derivatives": derivative}]
    kappa_range = config["problem"].get("kappa_range", [equation.kappa] * 2)
    rate_range = config["problem"].get("reaction_rate_range", [equation.reaction_rate] * 2)
    for index, name in enumerate(("lower_endpoint", "upper_endpoint")):
        physical = Equation(float(kappa_range[index]), float(rate_range[index]))
        generator = torch.Generator(device=state.device).manual_seed(int(config["seed"]) + 101 + index)
        # Independent bounded initial realizations use legal local observations.
        # Their high-frequency components make hidden influence less favorable
        # than a single smooth sine field without pretending to be a population.
        parent_state = .35 + .25 * torch.rand(state.shape, dtype=state.dtype, device=state.device, generator=generator)
        frozen_count = 4 * max(base_substeps, choose_substeps(largest_h, physical, geometry))
        parent_derivatives = derivative_audit(parent_state, physical, geometry, largest_h,
                                               substeps=frozen_count, seed=config["seed"] + index + 1)
        parent_information = same_observation_audit(parent_state, physical, geometry, largest_h,
                                                     substeps=frozen_count, t_ref=config["problem"]["t_ref"],
                                                     U_ref=config["problem"]["U_ref"])
        information_regimes.append({"regime": name, "initial_realization_seed": config["seed"] + 101 + index,
                                    "equation": {"kappa": physical.kappa, "reaction_rate": physical.reaction_rate},
                                    "information": parent_information, "derivatives": parent_derivatives})
    fit_h = [float(h) for h in config["horizons"]["values"]]
    heldout_h = [(a + b) / 2 for a, b in zip(fit_h[:-1], fit_h[1:])] + [1.25 * max(fit_h)]
    fit_defects, heldout_defects = [], []
    fit_teacher_audits = []
    for h in fit_h + heldout_h:
        count = max(base_substeps, choose_substeps(h, equation, geometry))
        result = refined_reference(state, h, equation, geometry, substeps=count,
                                   error_fraction=error_fraction, tolerance=config["validation"]["tolerance"] * teacher_config["tolerance_fraction"],
                                   noise_floor=noise_floor)
        fit_teacher_audits.append({"h": h, "accepted": result.accepted,
                                   "uncertainty": result.uncertainty, "defect_norm": result.defect_norm,
                                   "substeps": result.substeps})
        defect = result.state - split_step(state, h, equation, geometry, differentiable=False)
        (fit_defects if h in fit_h else heldout_defects).append(defect.detach().cpu().numpy())
    oracle = temporal_oracle_fits(fit_h, np.stack(fit_defects), heldout_h, np.stack(heldout_defects),
                                  fixed_rates=config["model"].get("fixed_rates", [.1, 1., 10., 100.]),
                                  t_ref=config["problem"]["t_ref"], noise_floor=noise_floor)
    reference_pass = all(x["accepted"] for x in refinements + fit_teacher_audits)
    order_pass = logistic_fits["split_local"]["valid"] and logistic_fits["split_global"]["valid"]
    order_pass = bool(order_pass and 2.7 < logistic_fits["split_local"]["slope"] < 3.3 and
                       1.7 < logistic_fits["split_global"]["slope"] < 2.3)
    # A cheap fourth-order classical correction can close the entire apparent
    # split gap. Numerical correctness is not itself scientific headroom.
    tolerance = config["validation"]["tolerance"]
    largest_operational_h = max(float(h) for h in config["horizons"]["evaluation_steps"])
    largest_profile = {row["method"]: row for row in classical_cost_profile if row["h"] == largest_operational_h}
    headroom = bool(max(local_errors) > 10 * noise_floor and
                    largest_profile["split"].get("error") is not None and
                    largest_profile["richardson_split"].get("error") is not None and
                    largest_profile["split"]["error"] > tolerance and
                    largest_profile["richardson_split"]["error"] > tolerance)
    temporal_errors = {name: max(value["heldout_relative_rms_with_noise_floor"])
                       for name, value in oracle["models"].items()}
    best_classical = min(temporal_errors["polynomial"], temporal_errors["rational"])
    best_temporal = min(temporal_errors["fixed_rate"], temporal_errors["learned_rate_oracle"])
    temporal_advantage = bool(best_temporal < .8 * best_classical)
    information_budget = float(tolerance) * float(config["validation"]["reference_fraction"])
    information_screen = all(item["information"]["feature_equality_exact"] and item["derivatives"]["passed"] and
                             item["information"]["deterministic_local_defect_worst_case_error_lower_bound"] <= information_budget
                             for item in information_regimes)
    gates = {
        "G1": {"passed": bool(tier_a["passed"] and derivative["passed"]), "scope": "linear order and discrete derivative controls; temporal-core unit suite is separately required"},
        "G2": {"passed": bool(reference_pass and order_pass and headroom), "numerical_correctness_passed": bool(reference_pass and order_pass),
               "material_headroom_observed": headroom,
               "screening_rule": "largest proposed macrostep misses development tolerance for both split and costed classical Richardson correction, and defects exceed floor",
               "reason": "Measured classical errors/costs screen for headroom; one tiny probe cannot establish learned efficiency"},
        "G3": {"passed": bool(information_screen), "exact_same_observation_test_passed": information["feature_equality_exact"],
               "empirical_screen_only": True, "independent_initial_realizations": len(information_regimes),
               "proposed_hidden_input_error_budget": information_budget,
               "budget_rule": "development numerical tolerance * validation.reference_fraction",
               "reason": "Sampled exact-hidden-pair lower bounds below a proposed budget and derivative audits pass; this screens against immediate impossibility, not a universal sufficiency certificate"},
        "G4": {"passed": bool(reference_pass and temporal_advantage), "oracle_temporal_advantage_observed": temporal_advantage,
               "reason": "Teacher-informed per-anchor fits assess representation only; deployable feature-to-coefficient learning remains untested"},
    }
    payload = {"stage": "audit", "evidence_category": "newly measured result", "device": str(device),
               "config_hash": hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
               "scope": "FP64 tiny-grid CPU/GPU selected device; no CARC access or GPU claim follows from CPU",
               "problem": {"grid": list(geometry.grid), "lengths": list(geometry.lengths),
                           "kappa": equation.kappa, "reaction_rate": equation.reaction_rate,
                           "split": "reaction_half_diffusion_full_reaction_half",
                           "diffusion": "exact discrete periodic FFT", "teacher": "coupled RK4, three refinements"},
               "tier_a": tier_a, "teacher_refinements": refinements, "global_teacher_uncertainty": global_uncertainty,
               "logistic_convergence": logistic_fits, "derivatives": derivative, "information": information,
               "classical_cost_profile": classical_cost_profile,
               "information_regimes": information_regimes,
               "temporal_oracle": oracle, "temporal_teacher_audits": fit_teacher_audits, "gates": gates,
               "decision": "continue bounded development diagnostics" if reference_pass and order_pass else "stop: numerical audit failed",
               "scientific_claim": "No learned accuracy or efficiency result has been established"}
    _write(run_dir / "audit.json", payload)
    _write(run_dir / "gates.json", gates)
    _write(run_dir / "teacher_refinement.json", {"three_level_audits": refinements})
    _plot_refinements(refinements, run_dir / "teacher_refinement.png")
    return payload


def _plot_refinements(refinements, path):
    configure_plot_cache(Path(path).parent)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, ax = plt.subplots()
    for row in refinements:
        ax.loglog(row["substeps"][1:], np.maximum(row["refinement_differences"], 1e-17), "o-", label=f"h={row['h']:.5g}")
    ax.set(xlabel="finer RK4 substeps", ylabel="successive FP64 state difference", title="Three-level coupled teacher refinement")
    ax.legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _dataset_parent(store, split: str, index: int, config: dict, device):
    sample = store.sample(split, index)
    physical = sample["parameters"]
    equation = Equation(float(physical["kappa"]), float(physical["reaction_rate"]))
    geometry = Geometry(tuple(config["problem"]["grid"]), tuple(config["problem"]["lengths"]))
    dtype = {"float32": torch.float32, "float64": torch.float64}[config["precision"]["state"]]
    state = torch.as_tensor(np.array(sample["state"], copy=True), device=device, dtype=dtype)
    return sample["parent_id"], sample["metadata"], state, equation, geometry


def _reference_final(state, equation, geometry, config):
    total = float(config["horizons"]["rollout_time"])
    count = max(int(config["teacher"]["base_substeps"]), choose_substeps(total, equation, geometry))
    start = time.perf_counter()
    result = refined_reference(state.double(), total, equation, geometry, substeps=count,
                               error_fraction=config["teacher"]["error_fraction"],
                               tolerance=config["validation"]["tolerance"] * config["teacher"]["tolerance_fraction"],
                               noise_floor=config["teacher"]["noise_floor"])
    if not result.accepted:
        raise RuntimeError(f"Full-rollout teacher failed refinement: {result.reason}")
    return result.state, {"uncertainty": result.uncertainty, "substeps": result.substeps,
                          "refinement_counts": list(result.refinement_substeps),
                          "refinement_differences": list(result.refinement_differences),
                          "reference_seconds_excluded_from_deployed_cost": time.perf_counter() - start,
                          "initial_state_contract": "reference starts from identical deployed dtype input promoted to FP64"}


def _sequence_policies(config):
    """Both unseen h and mixed steps keep the identical final physical time."""
    return [("constant", (1.,)), ("mixed", tuple(config["horizons"]["mixed_factors"])),
            ("unseen_horizon", (.75,))]


def _candidate_records(store, split, config, method, h, *, device, model=None,
                       benchmark_mode=False, reference_cache=None):
    outcomes = []
    tolerance = float(config["validation"]["tolerance"])
    if reference_cache is None:
        reference_cache = {}
    for index in range(store.count(split)):
        parent_id, sample_metadata, state, equation, geometry = _dataset_parent(store, split, index, config, device)
        anchor_key = (parent_id, sample_metadata["anchor_index"])
        reference_error = None
        if anchor_key not in reference_cache:
            try:
                reference_cache[anchor_key] = _reference_final(state, equation, geometry, config)
            except (RuntimeError, ValueError, FloatingPointError) as error:
                reference_error = f"{type(error).__name__}: {error}"
        teacher, teacher_metadata = reference_cache.get(anchor_key, (None, {"accepted": False, "failure_reason": reference_error}))
        for sequence_id, factors in _sequence_policies(config):
            steps = step_schedule(float(config["horizons"]["rollout_time"]), h, factors)
            record = {"parent_id": parent_id, "split": split, "method": method, "candidate_h": h,
                      "anchor_index": sample_metadata["anchor_index"], "anchor_time": sample_metadata["anchor_time"],
                      "sequence": sequence_id, "actual_h_sequence": steps,
                      "physical_horizon": sum(steps), "precision": config["precision"]["state"],
                      "reference": teacher_metadata, "failed": True, "error": None, "failure_reason": None}
            if reference_error:
                record["failure_reason"] = f"Teacher unavailable: {reference_error}"
                outcomes.append(record)
                continue
            try:
                def operation():
                    with torch.inference_mode():
                        return _rollout(state.clone(), steps, equation, geometry, method, model=model,
                                        chunk_size=config["model"].get("chunk_size", 32768),
                                        rtol=0., atol=tolerance * .5)
                (computed, counters), performance = measure(operation, device=device,
                                                            warmup=1 if benchmark_mode else 0,
                                                            repeats=3 if benchmark_mode else 1,
                                                            memory_policy=config["runtime"])
                error = _rms(computed.double() - teacher, geometry)
                record.update(error=error, failed=error > tolerance,
                              failure_reason="fixed numerical tolerance exceeded" if error > tolerance else None,
                              counters=counters, performance=performance,
                              physical_observables={"mean_state": float(computed.double().mean()),
                                                    "minimum": float(computed.min()), "maximum": float(computed.max()),
                                                    "reference_mean": float(teacher.mean()),
                                                    "mean_absolute_error": float((computed.double() - teacher).abs().mean()),
                                                    "maximum_absolute_error": float((computed.double() - teacher).abs().max())})
                if performance.get("soft_budget_passed") is False or performance.get("hard_device_memory_warning"):
                    record.update(failed=True, failure_reason="measured CUDA memory budget exceeded")
            except (RuntimeError, ValueError, FloatingPointError) as error:
                record["failure_reason"] = f"{type(error).__name__}: {error}"
            outcomes.append(record)
    return outcomes


def _aggregate(outcomes, config):
    parents = sorted({row["parent_id"] for row in outcomes})
    errors = []
    for parent in parents:
        rows = [row for row in outcomes if row["parent_id"] == parent]
        errors.append(None if any(row["failed"] for row in rows) else max(row["error"] for row in rows))
    summary = parent_summary(errors, bootstrap_samples=config["validation"]["bootstrap_samples"], seed=config["seed"])
    measured = [row["performance"]["wall_seconds_median"] for row in outcomes if "performance" in row]
    summary.update(method=outcomes[0]["method"], parent_ids=parents,
                   median_wall_seconds=float(np.median(measured)) if measured else None,
                   worst_measured_error=max((row["error"] for row in outcomes if row["error"] is not None), default=None),
                   required_sequences=[name for name, _ in _sequence_policies(config)])
    return summary


def _load_learned(checkpoint, config, store, device):
    if checkpoint is None:
        return None, None
    from tdn.train import load_checkpoint, model_from_checkpoint
    payload = load_checkpoint(Path(checkpoint), map_location="cpu")
    # Changing the immutable parent split or data provenance invalidates a
    # saved model. Never silently evaluate a checkpoint on a different dataset.
    for field, expected in (("split_hash", store.split_hash), ("dataset_hash", store.manifest_hash)):
        saved = payload.get(field)
        if saved is not None and saved != expected:
            raise ValueError(f"Checkpoint {field} does not match the dataset")
    model = model_from_checkpoint(payload, device=device)
    if config["precision"]["state"] == "float64":
        model = model.double()
    model.eval()
    trained_family = payload.get("config", {}).get("model", {}).get("family", config["model"]["family"])
    return model, {"path": str(Path(checkpoint).resolve()), "family": trained_family,
                   "selection": "one previously validation-selected checkpoint; unchanged for all parents/horizons",
                   "deployment_precision": config["precision"]["state"],
                   "normalization": "checkpoint training-parent statistics",
                   "training_global_step": payload.get("global_step")}


def _run_evaluation(config, dataset_dir, checkpoint, run_dir, device, *, benchmark_mode):
    from tdn.data import DatasetStore
    store = DatasetStore(Path(dataset_dir), verify=True)
    for field in ("family", "grid", "lengths", "t_ref", "U_ref", "periodic"):
        if store.manifest["problem"].get(field) != config["problem"].get(field):
            raise ValueError(f"Evaluation configuration {field} differs from immutable dataset problem")
    if store.count("validation") < 1 or store.count("diagnostic") < 1:
        raise ValueError("Validation tuning and independent diagnostic parents are required")
    # Diagnostic development evaluation cannot open a sealed confirmatory set.
    # A separate protocol must freeze tolerance, sample size and deployed policy.
    model, checkpoint_metadata = _load_learned(checkpoint, config, store, device)
    methods = ["split", "adaptive_split", "coupled_rk4", "richardson_split"]
    if model is not None:
        methods.append("learned")
    validation_candidates, selected, diagnostic, summaries = {}, {}, {}, {}
    reference_cache = {}
    candidate_steps = sorted({float(h) for h in config["horizons"]["evaluation_steps"]})
    for method in methods:
        trials = []
        for h in candidate_steps:
            rows = _candidate_records(store, "validation", config, method, h, device=device,
                                      model=model, benchmark_mode=benchmark_mode, reference_cache=reference_cache)
            summary = _aggregate(rows, config)
            trials.append({"candidate_h": h, "summary": summary, "outcomes": rows})
        feasible = [trial for trial in trials if trial["summary"]["failed_parents"] == 0]
        if feasible:
            winner = min(feasible, key=lambda item: item["summary"]["median_wall_seconds"])
            selection_reason = "fastest validation policy satisfying tolerance and all declared sequences"
        else:
            # Retain a policy for failure diagnosis; this is not tolerance-matched
            # success and cannot support an efficiency comparison.
            winner = min(trials, key=lambda item: (item["summary"]["failed_parents"],
                                                  item["summary"]["worst_measured_error"] if item["summary"]["worst_measured_error"] is not None else math.inf))
            selection_reason = "no feasible validation policy; minimum-failure policy retained for diagnostics"
        validation_candidates[method] = trials
        selected[method] = {"macrostep": winner["candidate_h"], "validation_feasible": bool(feasible),
                            "selection_reason": selection_reason, "validation_summary": winner["summary"]}
        diagnostic[method] = _candidate_records(store, "diagnostic", config, method, winner["candidate_h"],
                                                device=device, model=model, benchmark_mode=benchmark_mode,
                                                reference_cache=reference_cache)
        summaries[method] = _aggregate(diagnostic[method], config)
    comparisons = {}
    if "learned" in summaries:
        for baseline in methods[:-1]:
            independent_parents = summaries[baseline]["parent_ids"]
            base_times, learned_times = [], []
            for parent in independent_parents:
                base_rows = [r for r in diagnostic[baseline] if r["parent_id"] == parent]
                learned_rows = [r for r in diagnostic["learned"] if r["parent_id"] == parent]
                base_times.append(np.nan if any(r["failed"] for r in base_rows) else np.mean([r["performance"]["wall_seconds_median"] for r in base_rows]))
                learned_times.append(np.nan if any(r["failed"] for r in learned_rows) else np.mean([r["performance"]["wall_seconds_median"] for r in learned_rows]))
            comparisons[baseline] = paired_parent_comparison(base_times, learned_times,
                                                             bootstrap_samples=config["validation"]["bootstrap_samples"], seed=config["seed"])
    all_valid = all(not summaries[method]["failed_parents"] and selected[method]["validation_feasible"] for method in methods)
    payload = {"stage": "benchmark" if benchmark_mode else "evaluate", "evidence_category": "newly measured result",
               "config_hash": hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
               "scope": "development diagnostic parents only; sealed confirmatory parents remain unopened",
               "device": str(device), "tolerance": config["validation"]["tolerance"],
               "tolerance_provenance": config["validation"]["tolerance_provenance"],
               "physical_horizon": config["horizons"]["rollout_time"], "dataset_hash": store.manifest_hash,
               "split_hash": store.split_hash, "checkpoint": checkpoint_metadata,
               "validation_candidates": validation_candidates, "selected_policies": selected,
               "diagnostic_outcomes": diagnostic, "summaries": summaries,
               "parent_paired_runtime_comparisons": comparisons, "all_methods_tolerance_feasible": all_valid,
               "baseline_coverage": {"implemented": methods,
                   "neural_control_policy": "Run the same pipeline separately for fixed_rate, polynomial, rational, generic_mlp and taylor configs with equal training/tuning budgets",
                   "unsupported_optional": ["KAN", "anchored nonlinear discrete e3", "Gray-Scott", "advection", "distributed HALO"],
                   "competitive_coupled_method": "stability-resolved coupled RK4; stiff regimes may require IMEX before confirmation"},
               "training_amortization": {"available": False, "reason": "All dataset/training/setup costs must be measured and a positive matched-tolerance runtime gain established"},
               "claim": "Measured development frontiers only; no confirmatory efficiency claim. GPU results require recorded CUDA execution on the reported hardware."}
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write(run_dir / ("benchmark.json" if benchmark_mode else "evaluation.json"), payload)
    _write(run_dir / "evaluation.json", payload)
    _plot_frontiers(payload, run_dir)
    return payload


def _plot_frontiers(payload, run_dir):
    configure_plot_cache(run_dir)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, ax = plt.subplots()
    memory_figure, memory_ax = plt.subplots()
    memory_points = False
    for method, trials in payload["validation_candidates"].items():
        time_points, error_points, memory_values = [], [], []
        for trial in trials:
            rows = trial["outcomes"]
            if not rows or any("performance" not in row or row["error"] is None for row in rows):
                continue
            time_points.append(np.mean([row["performance"]["wall_seconds_median"] for row in rows]))
            error_points.append(max(row["error"] for row in rows))
            memory_values.append(max(row["performance"]["peak_reserved_bytes"] or 0 for row in rows))
        if time_points:
            ax.loglog(time_points, error_points, "o-", label=method)
        if any(memory_values):
            memory_points = True
            memory_ax.loglog(np.asarray(memory_values) / 1024**3, error_points, "o-", label=method)
    ax.axhline(payload["tolerance"], linestyle="--", color="black", label="prespecified development tolerance")
    ax.set(xlabel="full-rollout median wall seconds", ylabel="worst validation-parent/sequence weighted error",
           title="Validation tuning frontier, matched precision and final time")
    ax.legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(run_dir / "accuracy_time_frontier.png")
    plt.close(figure)
    if memory_points:
        memory_ax.axhline(payload["tolerance"], linestyle="--", color="black")
        memory_ax.set(xlabel="warm peak reserved CUDA GiB", ylabel="worst validation weighted error",
                      title="Measured CUDA accuracy-memory frontier")
        memory_ax.legend(fontsize="small")
    else:
        memory_ax.text(.5, .5, "CUDA memory unavailable: this run used CPU.\nHost lifetime peak RSS is recorded in JSON.",
                       ha="center", va="center", transform=memory_ax.transAxes)
        memory_ax.set_axis_off()
    memory_figure.tight_layout()
    memory_figure.savefig(run_dir / "accuracy_memory_frontier.png")
    plt.close(memory_figure)


def evaluate(config: dict, dataset_dir: Path, checkpoint: Path | None, run_dir: Path, device="cpu") -> dict:
    return _run_evaluation(config, dataset_dir, checkpoint, run_dir, device, benchmark_mode=False)


def benchmark(config: dict, dataset_dir: Path, checkpoint: Path | None, run_dir: Path, device="cpu") -> dict:
    return _run_evaluation(config, dataset_dir, checkpoint, run_dir, device, benchmark_mode=True)
