"""A calibrated whole-interval acceptance diagnostic with audited costs.

This is a bounded prototype, not a certified adaptive integrator. Every
estimator threshold is frozen using an independent calibration bank before
fresh policy labels are opened. Rejected work, estimator solves and fallback
solves are all included in cost; composition agreement is never a certificate.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
import time

import torch

from tdn.numerics import Equation, Geometry
from tdn.research.experiment import horizon_key
from tdn.research.protocol import digest, file_digest
from tdn.runtime.metadata import write_json

from .data import load_bank, load_normalization


ESTIMATORS = ("step_doubling", "splitting_defect", "validated_envelope")


def refined_schedule(schedule, factor):
    return [h / factor for h in schedule for _ in range(factor)]


def endpoint_difference(a, b):
    d = a.detach().double() - b.detach().double()
    return {"rms": float(d.square().mean().sqrt()), "max": float(d.abs().max())}


def fit_envelope(rows, estimator, *, safety_factor=1.1):
    """Maximal calibration ratio, with explicit zero-estimate handling.

    There is no Richardson denominator: unknown finite/stiff order prevents
    assuming that a step-doubling difference estimates one specific order.
    """
    values = {}
    for norm in ("rms", "max"):
        ratios = []
        for row in rows:
            estimate, actual = row["estimates"][estimator][norm], row[f"upper_{norm}"]
            if not math.isfinite(actual) or not math.isfinite(estimate):
                raise FloatingPointError("Nonfinite policy calibration observation")
            ratios.append(actual / max(estimate, 1e-12))
        values[norm] = max(1., max(ratios, default=1.)) * safety_factor
    return values


def accepted(estimate, envelope, rms_target, max_target):
    return envelope["rms"] * max(estimate["rms"], 1e-12) <= rms_target and envelope["max"] * max(estimate["max"], 1e-12) <= max_target


def policy_schedule(horizon, step_size):
    count = max(1, math.ceil(round(horizon / step_size, 12)))
    return [horizon / count] * count


def _defect_proxy(model, initial, schedule, equation, geometry, budget=None):
    """Actual flow residual, estimated by an exact autodiff horizon JVP.

    Sum h*||d_h Phi_h(u)-F(Phi_h(u))|| over attempted substeps. This is
    neither a rigorous stability-weighted integral nor an order certificate.
    Empirical calibration must expose its reliability independently.
    """
    from tdn.numerics.operators import rhs
    value, estimates = initial, {"rms": 0., "max": 0.}
    for h in schedule:
        if budget:
            budget.check()
        state = value.detach()
        step = torch.tensor(h, dtype=state.dtype, device=state.device)
        value, derivative = torch.autograd.functional.jvp(
            lambda query: model(state, query, equation, geometry), step, torch.ones_like(step), create_graph=False)
        residual = derivative - rhs(value, equation, geometry)
        estimates["rms"] += h * float(residual.detach().double().square().mean().sqrt())
        estimates["max"] += h * float(residual.detach().abs().max())
        value = value.detach()
    return estimates


def _attempt(model, initial, schedule, equation, geometry, device, budget):
    from .engine import timed_rollout
    start = time.perf_counter()
    coarse, coarse_cost = timed_rollout(model, initial, schedule, equation, geometry, device=device, budget=budget)
    fine, fine_cost = timed_rollout(model, initial, refined_schedule(schedule, 2), equation, geometry, device=device, budget=budget)
    from .engine import _sync
    _sync(device)
    defect_start = time.perf_counter()
    defect = _defect_proxy(model, initial, schedule, equation, geometry, budget)
    _sync(device)
    defect_seconds = time.perf_counter() - defect_start
    doubling = endpoint_difference(coarse, fine)
    return coarse, {"step_doubling": doubling,
                    "splitting_defect": defect, "validated_envelope": {norm: max(doubling[norm], defect[norm]) for norm in ("rms", "max")}}, {
        "coarse_seconds": coarse_cost["timing_seconds"], "doubling_extra_seconds": fine_cost["timing_seconds"],
        "defect_extra_seconds": defect_seconds,
        "all_estimator_attempt_seconds": coarse_cost["timing_seconds"] + fine_cost["timing_seconds"] + defect_seconds,
        "host_attempt_elapsed_seconds": time.perf_counter() - start,
        "coarse_steps": len(schedule), "doubling_steps": 2 * len(schedule), "defect_jvps": len(schedule)}


def _used_cost(cost, estimator):
    return cost["coarse_seconds"] + (cost["doubling_extra_seconds"] if estimator in ("step_doubling", "validated_envelope") else 0) + (
        cost["defect_extra_seconds"] if estimator in ("splitting_defect", "validated_envelope") else 0)


def run_policy(protocol, dataset, prerequisites, path, device, budget):
    from .engine import errors, load_selected_models, select_confirmation, _catalog, _table, timed_rollout
    path = Path(path)
    options = protocol.get("policy", {})
    limit = options.get("model_limit", 3)
    # Mechanism slots are frozen in the protocol; selection uses validation
    # scores, never confirmation feasibility or timing wins.
    records = select_confirmation(protocol, _catalog(protocol, prerequisites))
    preferred = options.get("families", ["source", "source_time", "source_closure"])
    records = [min((r for r in records if r["family"] == family), key=lambda r: (r["best_validation_loss"], r["trial_id"]))
               for family in preferred if any(r["family"] == family for r in records)][:limit]
    if not records:
        write_json(path / "calibration.json", {"protocol_sha256": digest(protocol), "status": "NO_ELIGIBLE_CHECKPOINT", "records": [],
            "cohort": "calibration", "checkpoint_sha256": {}, "envelopes": {}, "rows": [],
            "calibration_parent_ids": [p["parent_id"] for p in protocol["parents"] if p["split"] == "calibration"],
            "fresh_confirmation_seen_during_calibration": False})
        write_json(path / "attempts.json", {"rows": []})
        write_json(path / "policy_endpoints.json", {"rows": []})
        write_json(path / "policy_summaries.json", {"rows": []})
        _table(path, "policy", [{"record_type": "policy", "question_ids": ["Q5"], "status": "NO_ELIGIBLE_CHECKPOINT"}])
        return {"models": 0, "attempts": 0, "false_accepts": 0, "policy_status": "NO_ELIGIBLE_CHECKPOINT"}
    models = load_selected_models(protocol, records, load_normalization(protocol, dataset), device)
    calibration = load_bank(protocol, dataset, "calibration")
    schedules = protocol.get("confirm_schedules", protocol.get("confirmation_schedules", []))
    horizons = sorted({round(sum(schedule), 12) for schedule in schedules})
    step_sizes = options.get("attempted_step_sizes", [.09, .045, .0225, .01125])
    policy_schedules = [policy_schedule(horizon, step_size) for horizon in horizons for step_size in step_sizes]
    grids = protocol["grids"]
    calibration_rows, frozen = [], {}
    with torch.no_grad():
        for record in records:
            model, identifier = models[record["trial_id"]], record["trial_id"]
            for parent in calibration:
                equation = Equation(parent["kappa"], parent["reaction_rate"])
                for grid in grids:
                    n = grid if isinstance(grid, int) else grid[0]
                    geometry = Geometry((n, n), (1., 1.))
                    initial = parent["states"][str(n)].to(device=device, dtype=torch.float32)
                    for schedule_index, schedule in enumerate(policy_schedules):
                        budget.check()
                        value, estimates, costs = _attempt(model, initial, schedule, equation, geometry, device, budget)
                        reference = parent["references"][f"discrete:{n}:{horizon_key(sum(schedule))}"]
                        calibration_rows.append({"record_type": "policy_calibration", "question_ids": ["Q5"],
                            "trial_id": identifier, "parent_id": parent["parent_id"], "grid_size": n,
                            "schedule_index": schedule_index, "cohort": "calibration", "estimates": estimates, **costs, **errors(value, reference)})
            own = [row for row in calibration_rows if row["trial_id"] == identifier]
            frozen[identifier] = {estimator: fit_envelope(own, estimator, safety_factor=options.get("safety_factor", 2.)) for estimator in ESTIMATORS}
    frozen_record = {"protocol_sha256": digest(protocol), "envelopes": frozen, "rows": calibration_rows,
        "calibration_parent_ids": [p["parent_id"] for p in calibration],
        "checkpoint_sha256": {r["trial_id"]: r["checkpoint_sha256"] for r in records},
        "fresh_confirmation_seen_during_calibration": False,
        "rule": "max independent calibration upper-error/estimate ratio times frozen protocol safety factor; zero estimates denominator1e-12",
        "scope": "empirical whole-interval acceptance; no certification or extrapolation guarantee"}
    write_json(path / "calibration.json", frozen_record)
    frozen_sha = file_digest(path / "calibration.json")
    # The fresh tensors are first loaded only after the acceptance calibration
    # is written and hashed; later reports cannot update the envelope.
    fresh = load_bank(protocol, dataset, "confirmation")
    rows, attempts = [], []
    actual_attempt_work = 0.
    rms_target = options.get("rms_target", options.get("primary_target", 2e-4))
    max_target = options.get("max_target", options.get("primary_target", 2e-4))
    max_attempts = min(options.get("max_attempts", 3), len(step_sizes))
    with torch.no_grad():
        for record in records:
            model, identifier = models[record["trial_id"]], record["trial_id"]
            for parent in fresh:
                equation = Equation(parent["kappa"], parent["reaction_rate"])
                for grid in grids:
                    n = grid if isinstance(grid, int) else grid[0]
                    geometry = Geometry((n, n), (1., 1.))
                    initial = parent["states"][str(n)].to(device=device, dtype=torch.float32)
                    for schedule_index, horizon in enumerate(horizons):
                        schedule = policy_schedule(horizon, step_sizes[0])
                        pending = set(ESTIMATORS)
                        trajectories, work, rejected = {}, {key: 0. for key in ESTIMATORS}, {key: 0 for key in ESTIMATORS}
                        for attempt in range(max_attempts):
                            if not pending:
                                break
                            budget.check()
                            schedule_now = policy_schedule(horizon, step_sizes[attempt])
                            attempt_start = time.perf_counter()
                            try:
                                value, estimates, costs = _attempt(model, initial, schedule_now, equation, geometry, device, budget)
                                failed = False
                            except (FloatingPointError, ValueError) as error:
                                from .engine import _sync
                                _sync(device)
                                value, estimates, costs = None, None, {"all_estimator_attempt_seconds": time.perf_counter() - attempt_start}
                                failed = True
                                failure = str(error)
                            actual_attempt_work += costs["all_estimator_attempt_seconds"]
                            for estimator in tuple(pending):
                                decision = False if failed else accepted(estimates[estimator], frozen[identifier][estimator], rms_target, max_target)
                                charge = costs.get("all_estimator_attempt_seconds", 0.) if failed else _used_cost(costs, estimator)
                                work[estimator] += charge
                                attempts.append({"record_type": "policy", "question_ids": ["Q5", "Q7"], "row_kind": "attempt",
                                    "trial_id": identifier, "family": record["family"], "parent_id": parent["parent_id"],
                                    "grid_size": n, "schedule_index": schedule_index, "estimator": estimator, "attempt": attempt,
                                    "status": "NUMERICAL_FAILURE" if failed else "ACCEPTED" if decision else "REJECTED",
                                    "schedule": schedule_now, "estimate": estimates[estimator] if estimates else None,
                                    "charged_seconds": charge, "cumulative_seconds": work[estimator], "timing_seconds": charge,
                                    "calibration_sha256": frozen_sha, "costs": costs, **({"error": failure} if failed else {})})
                                if decision:
                                    trajectories[estimator] = (value, attempt, False)
                                    pending.remove(estimator)
                                else:
                                    rejected[estimator] += 1
                        fallback_seconds, fallback_value = 0., None
                        if pending:
                            fallback_schedule = refined_schedule(schedule, options.get("fallback_refinement", 8))
                            fallback_value, fallback_cost = timed_rollout(None, initial, fallback_schedule, equation, geometry,
                                family=options.get("fallback", "gl3_fused"), device=device, budget=budget)
                            fallback_seconds = fallback_cost["timing_seconds"]
                            for estimator in pending:
                                work[estimator] += fallback_seconds
                                trajectories[estimator] = (fallback_value, max_attempts, True)
                        # Both classical complete-solver controls are measured,
                        # even if the learned estimator happened to accept.
                        classical_costs, classical_values = {}, {}
                        for family in ("etdrk4", "gl3_fused"):
                            baseline_value, timing = timed_rollout(None, initial, refined_schedule(schedule, options.get("fallback_refinement", 8)),
                                equation, geometry, family=family, device=device, budget=budget)
                            classical_costs[family] = {"timing_seconds": timing["timing_seconds"],
                                **errors(baseline_value, parent["references"][f"discrete:{n}:{horizon_key(sum(schedule))}"])}
                            classical_values[family] = baseline_value
                        for estimator, (value, accepted_attempt, fallback) in trajectories.items():
                            for track in ("discrete", "continuum"):
                                key = f"{track}:{n}:{horizon_key(sum(schedule))}"
                                if key not in parent["references"]:
                                    continue
                                metrics = errors(value, parent["references"][key])
                                # Acceptance is calibrated to the discrete
                                # equation only; continuum transfer is audited
                                # separately and can fail without relabeling.
                                false_rms = metrics["upper_rms"] > rms_target
                                false_max = metrics["upper_max"] > max_target
                                rows.append({"record_type": "policy", "question_ids": ["Q5", "Q7"], "row_kind": "endpoint",
                                    "trial_id": identifier, "family": record["family"], "checkpoint_selection": record["selection"],
                                    "parent_id": parent["parent_id"], "continuous_field_id": parent.get("continuous_field_id", parent["parent_id"]),
                                    "grid_size": n, "track": track, "schedule_index": schedule_index, "estimator": estimator,
                                    "factors": {key: parent.get(key) for key in ("mean", "variance", "amplitude", "spectrum", "phase", "kappa", "reaction_rate")},
                                    "regime": parent.get("spectrum"),
                                    "status": "FALLBACK" if fallback else "ACCEPTED", "fallback": fallback,
                                    "accepted_attempt": accepted_attempt, "rejected_attempts": rejected[estimator],
                                    "timing_seconds": work[estimator], "fallback_seconds": fallback_seconds if fallback else 0.,
                                    "classical_controls": {family: {"timing_seconds": classical_costs[family]["timing_seconds"],
                                        **errors(classical_values[family], parent["references"][key])} for family in classical_costs},
                                    "calibration_sha256": frozen_sha,
                                    "false_accept_rms": bool(not fallback and false_rms), "false_accept_max": bool(not fallback and false_max),
                                    "false_accept_joint": bool(not fallback and (false_rms or false_max)),
                                    "false_accept_definitely_resolved": bool(not fallback and
                                        (metrics["error_rms"] - metrics["uncertainty_rms"] > rms_target or metrics["error_max"] - metrics["uncertainty_max"] > max_target)),
                                    "target_rms": rms_target, "target_max": max_target,
                                    "joint_target_verified": metrics["upper_rms"] <= rms_target and metrics["upper_max"] <= max_target,
                                    "estimator_target": "discrete", **metrics})
                    _table(path, "policy", calibration_rows + attempts + rows)
    if file_digest(path / "calibration.json") != frozen_sha:
        raise ValueError("Fresh policy evaluation altered its frozen calibration")
    write_json(path / "attempts.json", {"rows": attempts})
    write_json(path / "policy_endpoints.json", {"rows": rows})
    policy_groups = {}
    for row in rows:
        policy_groups.setdefault((row["trial_id"], row["estimator"], row["track"]), []).append(row)
    summaries = []
    for (identifier, estimator, track), group in policy_groups.items():
        eligible = [row for row in group if not row["fallback"]]
        rate = sum(row["false_accept_joint"] for row in eligible) / len(eligible) if eligible else None
        regimes = {name: [row for row in group if row["regime"] == name] for name in {row["regime"] for row in group}}
        summaries.append({"record_type": "policy", "question_ids": ["Q5", "Q7"], "row_kind": "policy_summary",
            "trial_id": identifier, "estimator": estimator, "track": track,
            "status": "PASS" if eligible and rate <= protocol["success_criteria"]["maximum_false_acceptance_rate"] else "FAIL" if eligible else "NO_ACCEPTS",
            "accepted": len(eligible), "fallbacks": sum(row["fallback"] for row in group),
            "false_accepts": sum(row["false_accept_joint"] for row in eligible), "false_accept_rate": rate,
            "rejected_attempts": sum(row["rejected_attempts"] for row in group),
            "joint_target_coverage": sum(row["joint_target_verified"] for row in group) / len(group),
            "timing_seconds": sum(row["timing_seconds"] for row in group),
            "per_regime": {name: {"endpoints": len(values), "joint_verified": sum(row["joint_target_verified"] for row in values),
                "accepted": sum(not row["fallback"] for row in values), "false_accepts": sum(row["false_accept_joint"] for row in values)} for name, values in regimes.items()},
            "median_cost_over_classical": {family: statistics.median(row["timing_seconds"] / row["classical_controls"][family]["timing_seconds"] for row in group)
                for family in ("etdrk4", "gl3_fused")},
            "timing_scope": "component-attributed hypothetical separate-policy solver cost; actual shared benchmark work reported separately"})
    write_json(path / "policy_summaries.json", {"rows": summaries})
    _table(path, "policy", calibration_rows + attempts + rows + summaries)
    discrete = [r for r in rows if r["track"] == "discrete"]
    accepted_rows = [r for r in discrete if not r["fallback"]]
    return {"models": len(records), "calibration_rows": len(calibration_rows), "attempts": len(attempts),
        "policy_endpoints": len(rows), "accepted": len(accepted_rows), "fallbacks": sum(r["fallback"] for r in discrete),
        "false_accepts": sum(r["false_accept_joint"] for r in accepted_rows),
        "false_accept_rate": sum(r["false_accept_joint"] for r in accepted_rows) / len(accepted_rows) if accepted_rows else None,
        "false_accepts_rms": sum(r["false_accept_rms"] for r in accepted_rows),
        "false_accepts_max": sum(r["false_accept_max"] for r in accepted_rows),
        "all_rejected_work_charged": True, "calibration_sha256": frozen_sha,
        "timing_scope": "component-attributed separate-policy costs; shared benchmark computes all estimator components, actual benchmark time reported separately",
        "actual_attempt_benchmark_seconds": actual_attempt_work,
        "effective_attempt_cap": max_attempts, "frozen_step_sizes": step_sizes,
        "scientific_outcome": "FALSE_ACCEPTANCE_GATE_PASS" if accepted_rows and sum(r["false_accept_joint"] for r in accepted_rows) / len(accepted_rows) <= protocol["success_criteria"]["maximum_false_acceptance_rate"] else "FALSE_ACCEPTANCE_GATE_FAIL_OR_NO_ACCEPTS",
        "uncertainty_is_certificate": False, "policy_status": "EMPIRICAL_UNCERTIFIED_ACCEPTANCE_AUDIT"}
