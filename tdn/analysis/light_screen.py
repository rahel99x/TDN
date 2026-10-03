"""Prespecified, CPU-only physical and temporal screening without training.

The six cases are an inexpensive search for regimes worth a full scientific
audit. They do not implement G1/G3 and cannot authorize a pilot or confirmation.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from tdn.config import config_hash, validate_config
from tdn.numerics import Equation, Geometry, choose_substeps, refined_reference, split_step
from .convergence import temporal_oracle_fits
from .profiling import measure
from .workflow import _rms, _rollout, _safe_state, _write, step_schedule


PROTOCOL_VERSION = 1
REGIMES = (("baseline", .01, 2.), ("intermediate", .03, 6.), ("stiff", .1, 12.))
INITIAL_STATES = (("mixed_frequency", 101), ("bounded_random", 102))
FIT_HORIZONS = (.02, .03, .045, .065, .095, .14, .21, .32)
HELDOUT_HORIZONS = (.025, .055, .11, .18, .27, .4)
FIXED_RATES = (.1, 1., 10., 100.)
TOTAL_TIME = .64
OPERATIONAL_H = .32
TOLERANCE = .002
METHODS = ("split", "richardson_split")
MAX_SCREEN_SECONDS = 600
MAX_REFINEMENT_SUBSTEPS = 324
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def validate_light_config(config: dict, device: str = "cpu") -> dict:
    """Reject configuration changes that could expand or weaken this protocol."""
    checked = validate_config(config)
    if str(device) != "cpu":
        raise ValueError("The light screen is CPU-only; no GPU work is requested")
    if checked["purpose"] != "development" or checked["runtime"]["confirmatory_authorized"]:
        raise ValueError("Light screening is development-only and cannot authorize confirmation")
    if checked["data"]["confirmatory_count"] != 0:
        raise ValueError("Light screening cannot open confirmatory parents")
    problem = checked["problem"]
    if (problem["grid"] != [8, 8] or problem["lengths"] != [1., 1.]
            or problem["t_ref"] != 1. or problem["U_ref"] != 1.):
        raise ValueError("Light screening fixes the grid at 8x8 and reference scales at one")
    if (checked["validation"]["tolerance"] != TOLERANCE
            or checked["validation"]["require_headroom"] is not True):
        raise ValueError("Keep the unchanged 0.002 tolerance and require_headroom=true")
    if not 1 <= checked["training"]["max_steps"] <= 12:
        raise ValueError("The light configuration must retain a bounded <=12-step development budget")
    if (checked["precision"]["teacher"] != "float64"
            or checked["precision"]["state"] != "float32"
            or checked["precision"]["compile_mode"] != "eager"
            or checked["precision"]["network_autocast"] != "none"
            or checked["precision"]["tf32"] is not False):
        raise ValueError("Retain eager FP32 configuration, TF32 off, and FP64 teachers")
    if tuple(checked["model"]["fixed_rates"]) != FIXED_RATES:
        raise ValueError("Light screening fixes four rate columns at 0.1, 1, 10, 100")
    teacher = checked["teacher"]
    if teacher != {"base_substeps": 16, "error_fraction": .05,
                   "tolerance_fraction": .1, "noise_floor": 1e-10}:
        raise ValueError("Light screening uses its fixed three-level teacher budget")
    if checked["runtime"]["intraop_threads"] != 1 or checked["runtime"]["interop_threads"] != 1:
        raise ValueError("Light screening fixes one intraop and one interop thread")
    return checked


def initial_state(name: str, seed: int, geometry: Geometry) -> torch.Tensor:
    """Two declared bounded states, reused across physical regimes."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if name == "mixed_frequency":
        x, y = torch.meshgrid(*[torch.arange(n, dtype=torch.float64) / n
                               for n in geometry.grid], indexing="ij")
        phases = 2 * math.pi * torch.rand(3, generator=generator, dtype=torch.float64)
        state = (.5 + .15 * torch.sin(2 * math.pi * x + phases[0])
                 + .1 * torch.cos(2 * math.pi * y + phases[1])
                 + .1 * torch.sin(6 * math.pi * (x + y) + phases[2]))
    elif name == "bounded_random":
        state = .15 + .7 * torch.rand(geometry.grid, generator=generator, dtype=torch.float64)
    else:
        raise ValueError(f"Unknown prespecified initial state {name}")
    state = state[None, None]
    _safe_state(state)
    return state


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def make_plan(config: dict) -> dict:
    checked = validate_light_config(config)
    geometry = Geometry((8, 8), (1., 1.))
    states = {}
    for name, seed in INITIAL_STATES:
        state = initial_state(name, seed, geometry)
        states[name] = {"seed": seed, "shape": list(state.shape),
                        "state_sha256": hashlib.sha256(state.numpy().tobytes()).hexdigest(),
                        "minimum": float(state.min()), "maximum": float(state.max()),
                        "values": state[0, 0].tolist()}
    plan = {
        "protocol_version": PROTOCOL_VERSION, "config_hash": config_hash(checked),
        "device": "cpu", "screen_state_dtype": "float64", "configuration_state_dtype": "float32",
        "grid": [8, 8], "lengths": [1., 1.], "tolerance": TOLERANCE,
        "initial_states": states,
        "cases": [{"case_id": f"{regime}-{name}-{seed}", "regime": regime,
                   "kappa": kappa, "reaction_rate": rate, "initial_state": name,
                   "seed": seed, "initial_state_sha256": states[name]["state_sha256"]}
                  for regime, kappa, rate in REGIMES for name, seed in INITIAL_STATES],
        "fit_horizons": list(FIT_HORIZONS), "heldout_horizons": list(HELDOUT_HORIZONS),
        "fixed_rates": list(FIXED_RATES), "temporal_amplitude_columns": 4,
        "rate_tuning_max_nfev": 40, "operational_h": OPERATIONAL_H,
        "screen_walltime_budget_seconds": MAX_SCREEN_SECONDS,
        "maximum_finest_reference_substeps": MAX_REFINEMENT_SUBSTEPS,
        "rollout_time": TOTAL_TIME, "methods": list(METHODS),
        "cost_protocol": {"first_use": 1, "warmup": 1, "steady_state_repeats": 3},
        "teacher": dict(checked["teacher"]),
        "headroom_rule": "both fixed-h classical rollout errors minus reference uncertainty exceed unchanged tolerance",
        "temporal_rule": "eligible temporal worst held-out relative error upper bound < 0.8 * best classical lower bound",
        "eligibility_rule": "overdetermined full-rank amplitudes; learned rates additionally require informative_rate_fit",
        "heldout_rule": "held-out labels are scored only, never passed to the fit residual or rate optimizer",
        "case_accounting": "all six fixed cases retained; two initial states reused across three regimes; not six independent parents",
        "scope": "bounded development screen only; no dataset generation, training, GPU, G1/G3 audit or pilot authorization",
    }
    return {**plan, "plan_hash": _digest(plan)}


def _finite(value) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _reference(state, horizon, equation, geometry, teacher) -> tuple[torch.Tensor, dict]:
    base_count = max(teacher["base_substeps"], choose_substeps(horizon, equation, geometry))
    if 4 * base_count > MAX_REFINEMENT_SUBSTEPS:
        raise ValueError("Reference count exceeds the prespecified light-screen bound")
    result = refined_reference(state, horizon, equation, geometry, substeps=base_count,
                               error_fraction=teacher["error_fraction"],
                               tolerance=TOLERANCE * teacher["tolerance_fraction"],
                               noise_floor=teacher["noise_floor"])
    record = {"h": horizon, "accepted": bool(result.accepted), "reason": result.reason,
              "uncertainty": _finite(result.uncertainty), "defect_norm": _finite(result.defect_norm),
              "substeps": result.substeps, "refinement_substeps": list(result.refinement_substeps),
              "refinement_differences": [_finite(value) for value in result.refinement_differences],
              "observed_order": None if result.observed_order is None else _finite(result.observed_order),
              "converged": bool(result.converged),
              "uncertainty_convention": "full finest two-level difference plus FP64 rounding allowance"}
    return result.state, record


def headroom_screen(methods: list[dict], reference: dict) -> dict:
    by_name = {row["method"]: row for row in methods}
    complete = set(by_name) == set(METHODS) and all(not row["failed"] for row in methods)
    uncertainty = reference["uncertainty"]
    passed = bool(reference["accepted"] and uncertainty is not None and complete
                  and all(by_name[name]["error"] - uncertainty > TOLERANCE for name in METHODS))
    return {"passed": passed, "classical_methods_completed": complete,
            "tolerance": TOLERANCE, "reference_uncertainty": uncertainty,
            "rule": "both split and Richardson errors minus global reference uncertainty exceed 0.002",
            "scope": "numerical headroom at the fixed proposed h; no learned efficiency evidence"}


def temporal_screen(oracle: dict | None, references: list[dict]) -> dict:
    """Use fixed held-out comparisons with conservative reference intervals."""
    result = {"passed": False, "eligible_temporal_models": [], "models": {},
              "minimum_relative_improvement": .2,
              "rule": "best eligible temporal upper bound < 0.8 * best polynomial/rational lower bound",
              "scope": "teacher-informed representation oracle only; no deployable encoder is fitted"}
    if oracle is None or not all(row["accepted"] for row in references):
        result["reason"] = "Teacher references incomplete or rejected; oracle comparison unavailable"
        return result
    heldout = [row for row in references if row["role"] == "heldout"]
    magnitudes = np.asarray([row["defect_norm"] for row in heldout], float)
    uncertainty = np.asarray([row["uncertainty"] for row in heldout], float)
    for name, model in oracle["models"].items():
        absolute = np.asarray(model["heldout_absolute_rms"], float)
        eligible = bool(model.get("amplitude_fit_overdetermined") and model.get("amplitude_design_full_rank"))
        if name == "learned_rate_oracle":
            eligible = eligible and bool(model.get("informative_rate_fit"))
        lower = np.maximum(absolute - uncertainty, 0) / np.maximum(magnitudes + uncertainty, 1e-10)
        upper = (absolute + uncertainty) / np.maximum(magnitudes - uncertainty, 1e-10)
        result["models"][name] = {"eligible": eligible,
            "heldout_relative_rms_lower_bound": lower.tolist(),
            "heldout_relative_rms_upper_bound": upper.tolist(),
            "worst_relative_rms_lower_bound": float(lower.max()),
            "worst_relative_rms_upper_bound": float(upper.max())}
    controls = [name for name in ("polynomial", "rational")
                if name in result["models"] and result["models"][name]["eligible"]]
    temporal = [name for name in ("fixed_rate", "learned_rate_oracle")
                if name in result["models"] and result["models"][name]["eligible"]]
    result["eligible_temporal_models"] = temporal
    if len(controls) != 2 or not temporal:
        result["reason"] = "An informative temporal fit and both matched-amplitude classical controls are required"
        return result
    classical_lower = min(result["models"][name]["worst_relative_rms_lower_bound"] for name in controls)
    temporal_upper = min(result["models"][name]["worst_relative_rms_upper_bound"] for name in temporal)
    resolved = all(row["defect_norm"] is not None and row["uncertainty"] is not None
                   and row["defect_norm"] > 10 * max(1e-10, row["uncertainty"]) for row in references)
    result.update(best_classical_lower_bound=classical_lower, best_temporal_upper_bound=temporal_upper,
                  defects_resolved_above_reference_floor=resolved,
                  passed=bool(resolved and temporal_upper < .8 * classical_lower))
    result["reason"] = "Fixed held-out error and reference-uncertainty comparison; unsuccessful cases retained"
    return result


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.perf_counter() >= deadline:
        raise TimeoutError("Light-screen walltime budget exceeded; retain partial results and use a fresh run")


def _case(config: dict, declared: dict, geometry: Geometry, deadline: float | None = None) -> dict:
    state = initial_state(declared["initial_state"], declared["seed"], geometry)
    equation = Equation(declared["kappa"], declared["reaction_rate"])
    teacher = config["teacher"]
    references, labels = [], {"fit": [], "heldout": []}
    for role, horizons in (("fit", FIT_HORIZONS), ("heldout", HELDOUT_HORIZONS)):
        for horizon in horizons:
            _check_deadline(deadline)
            reference, row = _reference(state, horizon, equation, geometry, teacher)
            row["role"] = role
            references.append(row)
            defect = reference - split_step(state, horizon, equation, geometry, differentiable=False)
            labels[role].append(defect.numpy())
    _check_deadline(deadline)
    reference_T, global_reference = _reference(state, TOTAL_TIME, equation, geometry, teacher)
    methods = []
    for method in METHODS:
        _check_deadline(deadline)
        row = {"method": method, "h": OPERATIONAL_H, "total_time": TOTAL_TIME,
               "state_dtype": "float64", "failed": False, "error": None}
        try:
            def operation():
                with torch.inference_mode():
                    return _rollout(state.clone(), step_schedule(TOTAL_TIME, OPERATIONAL_H),
                                    equation, geometry, method)
            (final_state, counters), performance = measure(operation, device="cpu", warmup=1, repeats=3)
            row.update(error=_rms(final_state - reference_T, geometry), counters=counters, performance=performance)
            if not math.isfinite(row["error"]):
                row.update(failed=True, error=None, failure_reason="Nonfinite classical error against reference")
        except FloatingPointError as error:
            row.update(failed=True, failure_reason=f"{type(error).__name__}: {error}")
        methods.append(row)
    reference_pass = bool(global_reference["accepted"] and all(row["accepted"] for row in references))
    oracle = None
    if reference_pass:
        _check_deadline(deadline)
        oracle = temporal_oracle_fits(FIT_HORIZONS, np.stack(labels["fit"]),
                                     HELDOUT_HORIZONS, np.stack(labels["heldout"]),
                                     fixed_rates=FIXED_RATES, t_ref=1., noise_floor=teacher["noise_floor"])
    headroom = headroom_screen(methods, global_reference)
    temporal = temporal_screen(oracle, references)
    candidate = bool(reference_pass and headroom["passed"] and temporal["passed"])
    return {**declared, "reference_passed": reference_pass, "local_references": references,
            "global_reference": global_reference, "classical_methods": methods,
            "temporal_oracle": oracle, "numerical_headroom_screen": headroom,
            "temporal_representation_screen": temporal,
            "candidate_requires_full_audit": candidate,
            "decision": "candidate_requires_full_audit" if candidate else "no_joint_candidate",
            "pilot_authorized": False}


def run(config: dict, run_dir: str | Path, device: str = "cpu") -> dict:
    """Execute all six declared cases; science failures remain completed results."""
    checked = validate_light_config(config, device)
    target = Path(run_dir).resolve()
    if not target.is_relative_to(PROJECT_ROOT) or target == PROJECT_ROOT:
        raise ValueError("Light-screen output must stay inside the project checkout")
    target.mkdir(parents=True, exist_ok=True)
    if any((target / name).exists() for name in ("plan.json", "summary.json", "cases")):
        raise FileExistsError("Preserve prior light-screen results; use a fresh run directory")
    plan = make_plan(checked)
    _write(target / "plan.json", plan)  # Persist every case/rule before teacher work.
    started = time.perf_counter()
    geometry = Geometry((8, 8), (1., 1.))
    cases = []
    for declared in plan["cases"]:
        _check_deadline(started + MAX_SCREEN_SECONDS)
        case = _case(checked, declared, geometry, deadline=started + MAX_SCREEN_SECONDS)
        case["plan_hash"] = plan["plan_hash"]
        _write(target / "cases" / f"{declared['case_id']}.json", case)
        cases.append(case)
    candidates = [case["case_id"] for case in cases if case["candidate_requires_full_audit"]]
    _check_deadline(started + MAX_SCREEN_SECONDS)
    report = {"stage": "light-screen", "status": "COMPLETED",
              "evidence_category": "newly measured result", "device": "cpu",
              "config_hash": plan["config_hash"], "plan_hash": plan["plan_hash"],
              "scope": plan["scope"], "case_accounting": plan["case_accounting"],
              "declared_cases": len(plan["cases"]), "completed_cases": len(cases),
              "references_passed_cases": sum(case["reference_passed"] for case in cases),
              "headroom_passed_cases": sum(case["numerical_headroom_screen"]["passed"] for case in cases),
              "temporal_passed_cases": sum(case["temporal_representation_screen"]["passed"] for case in cases),
              "candidate_requires_full_audit": candidates,
              "decision": "full_audit_needed_for_candidates" if candidates else "no_joint_candidate",
              "pilot_authorized": False, "confirmatory_authorized": False,
              "screen_seconds": time.perf_counter() - started, "cases": cases,
              "next_step": "Review all cases; any promising regime needs full G1-G4 audit and representative short CUDA comparison before expanding training"}
    _write(target / "summary.json", report)
    lines = ["TDN bounded CPU light screen", f"Completed: {len(cases)}/6 fixed cases",
             f"Accepted references: {report['references_passed_cases']}/6",
             f"Numerical headroom: {report['headroom_passed_cases']}/6",
             f"Temporal advantage: {report['temporal_passed_cases']}/6",
             "Joint candidates requiring a full audit: " + (", ".join(candidates) or "none"),
             "No pilot or confirmatory work is authorized by this screen.", ""]
    (target / "summary.txt").write_text("\n".join(lines))
    return report
