"""Conditional empirical deployment with fresh fields and charged rejection.

The architecture is fixed before confirmation. A per-track utility gate can
block this experiment; it cannot choose a new winner using confirmation truth.
Calibration fits error-indicator envelopes only. Neither an empirical envelope
nor step agreement is a deterministic PDE error certificate.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
import time

import torch

from tdn.numerics import Geometry
from tdn.analysis.roadmap.numerics import fourier_resample
from tdn.analysis.roadmap.statistics import (assert_disjoint_cohorts, conformal_quantile,
    independent_function_scores, selective_risk_summary)
from tdn.research.protocol import digest
from tdn.runtime.metadata import write_json
from .core import check
from .measurement import measure_paired, field_cluster_interval, supported_amortization


def _timed(call, device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = call()
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    return result, time.perf_counter() - start


def difference(a, b):
    value = (a.detach().double() - b.detach().double()).abs()
    if not bool(torch.isfinite(value).all()):
        return {"rms": None, "max": None}
    return {"rms": float(value.square().mean().sqrt()), "max": float(value.max())}


def can_accept(indicator, envelope, target):
    if not math.isfinite(target) or target <= 0:
        raise ValueError("Policy target must be positive and finite")
    return all(indicator.get(k) is not None and envelope.get(k) is not None
               and math.isfinite(indicator[k]) and math.isfinite(envelope[k])
               and indicator[k] >= 0 and envelope[k] >= 0
               and indicator[k] * envelope[k] <= target for k in ("rms", "max"))


def cost_margin(proposal, estimator, acceptance, classical):
    values = (proposal, estimator, acceptance, classical)
    if any(v is None or not math.isfinite(v) for v in values) or not 0 <= acceptance <= 1:
        return None
    if min(proposal, estimator) < 0 or classical <= 0:
        return None
    return acceptance * classical - proposal - estimator


@torch.no_grad()
def proposal(model, initial, horizon, eq, geom, *, spatial, sampler, budget, device):
    from .neural import rollout
    coarse, proposal_seconds = _timed(lambda: rollout(model, initial, [horizon], eq, geom, budget)[0], device)
    def estimate():
        refined = rollout(model, initial, [horizon / 2] * 2, eq, geom, budget)[0]
        temporal = difference(coarse, refined)
        spatial_indicator = {"rms": 0., "max": 0.}
        if spatial:
            fine_geom = Geometry(tuple(2 * n for n in geom.grid), geom.lengths)
            fine_initial = sampler(fine_geom.grid[0]).to(device=device, dtype=initial.dtype)
            fine = rollout(model, fine_initial, [horizon], eq, fine_geom, budget)[0]
            spatial_indicator = difference(coarse, fourier_resample(fine, geom.grid))
        indicator = {k: temporal[k] + spatial_indicator[k] if temporal[k] is not None and spatial_indicator[k] is not None else None
                     for k in ("rms", "max")}
        return indicator, temporal, spatial_indicator
    (indicator, temporal, spatial_indicator), estimator_seconds = _timed(estimate, device)
    # A floor prevents finite calibration ratios hiding near-zero indicators.
    indicator = {k: max(v, 1e-8) if v is not None else None for k, v in indicator.items()}
    return coarse, dict(indicator=indicator, temporal_indicator=temporal, spatial_indicator=spatial_indicator,
        proposal_seconds=proposal_seconds, estimator_seconds=estimator_seconds,
        spatial_sampler_scope="same declared continuous initial function; sampling and transfer charged to estimator")


@torch.no_grad()
def classical_fallback(model, initial, horizon, eq, geom, *, target, max_refinements, budget):
    """Operational ETDRK4 temporal refinement, with no reference access."""
    from .neural import rollout
    if type(max_refinements) is not int or max_refinements < 1:
        raise ValueError("A positive bounded refinement count is required")
    previous = rollout(model, initial, [horizon], eq, geom, budget)[0]
    total_steps = 1
    error = {"rms": None, "max": None}
    for level in range(1, max_refinements + 1):
        count = 2 ** level
        current = rollout(model, initial, [horizon / count] * count, eq, geom, budget)[0]
        total_steps += count
        error = difference(current, previous)
        if all(error[k] is not None and error[k] <= target / 4 for k in error):
            return current, dict(converged=True, estimated_error=error, steps=total_steps,
                scope="temporal agreement; spatial/teacher accuracy audited separately")
        previous = current
    return previous, dict(converged=False, estimated_error=error, steps=total_steps,
        scope="bounded refinement exhausted; no guaranteed accuracy")


@torch.no_grad()
def deploy(model, classical, initial, horizon, eq, geom, *, envelope, target, spatial,
           sampler, max_refinements, budget, device):
    candidate, info = proposal(model, initial, horizon, eq, geom, spatial=spatial,
                               sampler=sampler, budget=budget, device=device)
    physical = bool(torch.isfinite(candidate).all()) and bool(((candidate >= -1e-7) & (candidate <= 1+1e-7)).all())
    accepted = physical and can_accept(info["indicator"], envelope, target)
    fallback_seconds, fallback_info = 0., None
    value = candidate
    if not accepted:
        (value, fallback_info), fallback_seconds = _timed(lambda: classical_fallback(classical, initial, horizon,
            eq, geom, target=target, max_refinements=max_refinements, budget=budget), device)
    return value, dict(**info, accepted=accepted, fallback=not accepted,
        fallback_seconds=fallback_seconds, fallback_info=fallback_info,
        attributed_seconds=info["proposal_seconds"] + info["estimator_seconds"] + fallback_seconds,
        physical_interval_admissible=physical, clipping=False,
        decision_uses_reference=False, deterministic_certificate=False)


def _envelope(rows, safety, alpha):
    if not math.isfinite(safety) or safety < 1:
        raise ValueError("The empirical envelope safety factor must be at least one")
    if any(r.get("split") != "calibration" for r in rows):
        raise ValueError("Policy envelope fitting accepts calibration fields only")
    values = {}
    for norm in ("rms", "max"):
        scores = [dict(field_cluster=r["field_cluster"], score=r[f"ratio_{norm}"]) for r in rows
                  if r.get(f"ratio_{norm}") is not None]
        clustered = independent_function_scores(scores)
        complete = len(scores) == len(rows) and bool(rows)
        values[norm] = dict(empirical=max((s["score"] for s in clustered), default=0.) * safety if complete else None,
                            conformal=conformal_quantile([s["score"] for s in clustered], alpha),
                            independent_function_scores=clustered, complete=complete)
    return dict(empirical={k: values[k]["empirical"] for k in values}, diagnostics=values,
                scope="calibration-only empirical maximum envelope; no distribution-shift or conditional-risk guarantee")


def _blocked(ctx, track, gate):
    reason = ("No resolved development headroom for this target track and deployment grid" if
              not gate.get("development_headroom", False) else
              "No preregistered matched-accuracy classical saving for the frozen candidate in this target track")
    ctx.record(f"{track}/blocked", ["G5"], status="BLOCKED", metrics={"deployment_executed": False},
        config={"track": track, "reason": reason, "prerequisite_gate": gate},
        checks=[check("measured-solver-margin", None, True, "eq", category="utility", reason=reason),
                check("conditional-risk-unmeasured", None, True, "eq", category="math"),
                check("conditional-deployment-unmeasured", None, True, "eq", category="gap")])
    return dict(track=track, status="BLOCKED", reason=reason, deployment_executed=False)


def run(ctx):
    from .data import load_split, field_state
    from .models import make_model
    from .neural import load_models, eq_geom, reference, endpoint_errors
    gate = json.loads((ctx.prerequisites["confirm"] / "gate.json").read_text())
    options = ctx.protocol["policies"]
    n, horizon, target = options["grid"], options["horizon"], options["target"]
    rows, calibrations, outcomes = [], {}, []
    enabled = {track: gate.get("tracks", {}).get(track, {}) for track in ctx.protocol["tracks"]}
    screening = json.loads((ctx.prerequisites["screen"] / "screen.json").read_text())
    if screening.get("selection_split") != "development":
        raise ValueError("Deployment headroom must be assessed on development fields")
    if not math.isclose(float(screening["target"]), float(target), rel_tol=1e-12):
        raise ValueError("Deployment and development headroom targets differ")
    enabled = {track: {**item, "development_headroom": any(
        cell["track"] == track and cell["grid"] == n and cell["outcome"] == "headroom"
        for cell in screening["conditions_detail"])} for track, item in enabled.items()}
    for item in enabled.values():
        item["solver_utility_eligible"] = bool(item.get("solver_utility_eligible") is True and item["development_headroom"])
    write_json(ctx.path / "policy_rows.json", {"rows": rows})
    if not any(v.get("solver_utility_eligible") is True for v in enabled.values()):
        outcomes = [_blocked(ctx, track, value) for track, value in enabled.items()]
        write_json(ctx.path / "calibration.json", {"tracks": {}, "status": "BLOCKED_BEFORE_CALIBRATION"})
        write_json(ctx.path / "gate.json", dict(gate="G5", status="BLOCKED", tracks=outcomes,
            scientific_outcome="NO_DEPLOYMENT_CLAIM; UTILITY_PREREQUISITE_NOT_MET"))
        return dict(status="BLOCKED", deployment_executed=False, tracks=outcomes)
    loading_started = time.monotonic()
    models = load_models(ctx)
    model_loading_seconds = time.monotonic() - loading_started
    catalog = json.loads((ctx.prerequisites["train"] / "catalog.json").read_text())
    for track, eligibility in enabled.items():
        if eligibility.get("solver_utility_eligible"):
            identifier = eligibility["selected_policy_model_id"]
            if catalog["selected_policy_models"].get(track) != identifier or identifier not in models:
                raise ValueError("Policy model must be fixed and available before confirmation")
    calibration, parents = load_split(ctx, "calibration"), load_split(ctx, "policy")
    assert_disjoint_cohorts(calibration=calibration, policy=parents)
    offline_calibration_seconds = 0.
    for track, eligibility in enabled.items():
        if eligibility.get("solver_utility_eligible") is not True:
            outcomes.append(_blocked(ctx, track, eligibility)); continue
        model_id = eligibility["selected_policy_model_id"]
        model = models[model_id]
        classical = make_model("etdrk4", track, ctx.protocol["model_config"]).to(ctx.device).eval()
        calibration_started = time.monotonic()
        for mode in options["modes"]:
            spatial = mode == "temporal_spatial"
            fitted = []
            for parent in calibration:
                ctx.budget.check()
                eq, geom = eq_geom(parent, n)
                initial = parent["states"][str(n)].to(ctx.device, dtype=torch.float32)
                value, info = proposal(model, initial, horizon, eq, geom, spatial=spatial,
                    sampler=lambda size, p=parent: field_state(p, size), budget=ctx.budget, device=ctx.device)
                errors = endpoint_errors(value, reference(parent, n, horizon, track))
                fitted.append(dict(field_cluster=parent["field_cluster"], parent_id=parent["parent_id"],
                    split=parent["split"], **info,
                    **{f"ratio_{norm}": errors[f"upper_{norm}"] / info["indicator"][norm]
                       if errors["reference_accepted"] and errors[f"upper_{norm}"] is not None and info["indicator"][norm] is not None else None
                       for norm in ("rms", "max")}))
            envelope = _envelope(fitted, options["empirical_safety_factor"], options["alpha"])
            calibrations[f"{track}/{mode}"] = dict(model_id=model_id, rows=fitted, **envelope)
        offline_calibration_seconds += time.monotonic() - calibration_started
        frozen_calibration = {key: value for key, value in calibrations.items() if key.startswith(track + "/")}
        calibration_hash = digest(frozen_calibration)
        calibration_file = f"calibration-{track}.json"
        if (ctx.path / calibration_file).exists():
            raise ValueError("Cannot replace a frozen track calibration")
        write_json(ctx.path / calibration_file, frozen_calibration)
        write_json(ctx.path / "calibration.json", {"tracks": calibrations, "seconds": offline_calibration_seconds,
            "selection_split": "calibration", "frozen_per_track_before_policy": True,
            "immutable_track_files": {t: f"calibration-{t}.json" for t in enabled if (ctx.path / f"calibration-{t}.json").exists()}})
        for parent in parents:
            eq, geom = eq_geom(parent, n)
            initial = parent["states"][str(n)].to(ctx.device, dtype=torch.float32)
            args = dict(target=target, max_refinements=options["max_refinements"], budget=ctx.budget)
            calls = {"classical": lambda: classical_fallback(classical, initial, horizon, eq, geom, **args)}
            for mode in options["modes"]:
                envelope = calibrations[f"{track}/{mode}"]["empirical"]
                calls[mode] = lambda mode=mode, envelope=envelope: deploy(model, classical, initial, horizon, eq, geom,
                    envelope=envelope, spatial=mode == "temporal_spatial", sampler=lambda size: field_state(parent, size),
                    device=ctx.device, **args)
            # Keep small per-call component metadata, not every GPU output.
            component_samples = {name: [] for name in calls}
            def capture(name, call):
                value, info = call()
                component_samples[name].append(dict(info))
                return value, info
            captured_calls = {name: (lambda name=name, call=call: capture(name, call)) for name, call in calls.items()}
            answers, timings = measure_paired(captured_calls, device=ctx.device, repeats=ctx.protocol["timing_repeats"],
                warmup=ctx.protocol["timing_warmup"], seed=parent["seed"], budget=ctx.budget)
            # Fresh reference is used only AFTER every candidate made its decision.
            ref = reference(parent, n, horizon, track)
            classical_errors = endpoint_errors(answers["classical"][0], ref)
            classical_accurate = classical_errors["reference_accepted"] and classical_errors["finite"] and max(
                classical_errors["upper_rms"], classical_errors["upper_max"]) <= target
            for mode in options["modes"]:
                value, last_info = answers[mode]
                cost = timings["methods"][mode]
                component_index = min(range(len(cost["samples_seconds"])),
                    key=lambda i: abs(cost["samples_seconds"][i] - cost["median_seconds"]))
                warm_infos = component_samples[mode][1 + timings["warmup"]:]
                if any((v["accepted"], v["fallback"]) != (last_info["accepted"], last_info["fallback"]) for v in warm_infos):
                    raise ValueError("Policy decisions changed across identical warm timing repetitions")
                info = warm_infos[component_index]
                sample_seconds = cost["samples_seconds"][component_index]
                if info["attributed_seconds"] > sample_seconds * 1.001:
                    raise ValueError("Policy attributed work exceeds its complete measured call")
                errors = endpoint_errors(value, ref)
                accurate = errors["reference_accepted"] and errors["finite"] and max(errors["upper_rms"], errors["upper_max"]) <= target
                control_cost = timings["methods"]["classical"]
                row = dict(parent_id=parent["parent_id"], field_cluster=parent["field_cluster"], regime=parent["regime"],
                    seed=model.selection_metadata["seed"], model_id=model_id, track=track, mode=mode, grid=n,
                    final_time=horizon, target=target, **info, **errors, accurate=bool(accurate),
                    false_accept=bool(info["accepted"] and not accurate), classical_accurate=bool(classical_accurate),
                    accuracy_status="REFERENCE_UNACCEPTED" if not errors["reference_accepted"] else "NONFINITE" if not errors["finite"] else "ACCURATE" if accurate else "TARGET_EXCEEDED",
                    false_accept_scope="conservative possible failure; reference-unaccepted cases remain unresolved, not established inaccuracies",
                    classical_errors=classical_errors, classical_seconds=control_cost["median_seconds"],
                    cost_seconds=cost["median_seconds"], cost_ratio=cost["median_seconds"] / control_cost["median_seconds"],
                    cost=cost, classical_cost=control_cost, timing_rounds=timings["rounds"],
                    component_sample_index=component_index, component_sample_seconds=sample_seconds,
                    component_unattributed_seconds=max(0., sample_seconds-info["attributed_seconds"]),
                    component_scope="one actual warm sample closest to the median; components are not independently averaged; output is final deterministic repeat",
                    calibration_sha256=calibration_hash, calibration_file=calibration_file, status="COMPLETED")
                rows.append(row)
                write_json(ctx.path / "policy_rows.json", {"rows": rows})
                ctx.record(f"{track}/{mode}/{parent['parent_id']}", ["G5"], metrics=row,
                    config={"model_id": model_id, "mode": mode, "track": track, "target": target},
                    checks=[check("decision-independent-of-reference", info["decision_uses_reference"], False, "eq", category="correctness"),
                            check("joint-accuracy", bool(accurate), True, "eq", category="gap"),
                            check("no-false-accept", row["false_accept"], False, "eq", category="math"),
                            check("complete-component-accounting", info["attributed_seconds"], sample_seconds * 1.001,
                                  "le", category="correctness"),
                            check("complete-cost-improves", row["cost_ratio"] if accurate and classical_accurate else None,
                                  1 / ctx.protocol["practical_speedup"], "le", category="utility")])
        if digest(json.loads((ctx.path / calibration_file).read_text())) != calibration_hash:
            raise ValueError("Frozen calibration changed during policy execution")
        outcomes.append(dict(track=track, status="MEASURED", deployment_executed=True))
    summaries = []
    offline = {}
    for stage in ("prepare", "train", "confirm_prepare"):
        path = ctx.prerequisites.get(stage)
        summary_path = Path(path) / "summary.json" if path is not None else None
        summary = json.loads(summary_path.read_text()) if summary_path is not None and summary_path.is_file() else {}
        offline[stage] = summary.get("elapsed_seconds")
    offline["calibration"] = offline_calibration_seconds
    offline["model_loading"] = model_loading_seconds
    for track in ctx.protocol["tracks"]:
        for mode in options["modes"]:
            group = [r for r in rows if r["track"] == track and r["mode"] == mode]
            if not group:
                continue
            matched = all(r["accurate"] and r["classical_accurate"] for r in group)
            interval = field_cluster_interval([dict(field_cluster=r["field_cluster"], seed=r["seed"],
                difference=r["classical_seconds"] - r["cost_seconds"]) for r in group],
                repeats=ctx.protocol["bootstrap_replicates"])
            acceptance = statistics.mean(r["accepted"] for r in group)
            risk = selective_risk_summary(group, target_risk=options["alpha"])
            risk.update(formal_conditional_risk_status="NA", iid_comparable_field_assumption_verified=False,
                        distribution_shift_guarantee=False)
            item = dict(track=track, mode=mode, endpoints=len(group), matched_accuracy=matched,
                acceptance_fraction=acceptance, cost_margin_seconds=cost_margin(
                    statistics.mean(r["proposal_seconds"] for r in group), statistics.mean(r["estimator_seconds"] for r in group),
                    acceptance, statistics.mean(r["classical_seconds"] for r in group)),
                risk=risk, saving_interval=interval,
                amortization=supported_amortization(offline, statistics.mean(r["cost_seconds"] for r in group),
                    statistics.mean(r["classical_seconds"] for r in group), matched_accuracy=matched,
                    accepted_neural_queries=sum(r["accepted"] and models[r["model_id"]].selection_metadata["updates_selected"] > 0 for r in group),
                    saving_lower=interval["lower"]))
            summaries.append(item)
            ctx.record(f"{track}/{mode}/summary", ["G5"], metrics=item,
                config={"track": track, "mode": mode, "alpha": options["alpha"]},
                checks=[check("fresh-field-matched-accuracy", matched, True, "eq", category="gap"),
                    check("positive-measured-cost-margin", item["cost_margin_seconds"], 0., "gt", category="utility"),
                    check("positive-field-cluster-saving", interval["lower"] if matched else None, 0., "gt", category="utility"),
                    check("formal-conditional-risk-guarantee", None, options["alpha"], category="math",
                          reason="Finite heterogeneous empirical fields do not establish iid exchangeability or a distribution-shift conditional-risk guarantee")])
    write_json(ctx.path / "policy_summary.json", {"rows": summaries, "offline_cost_scope": "conservative shared preparation/training plus calibration; diagnostic execution excluded from amortization"})
    write_json(ctx.path / "gate.json", {"gate": "G5", "status": "MEASURED", "tracks": outcomes,
        "deployment_executed": bool(rows), "scientific_outcome": "EMPIRICAL_FRESH_FIELD_ACCURACY_AND_FULL_COST; NO_FORMAL_GUARANTEE"})
    return dict(endpoints=len(rows), tracks=outcomes, false_accepts=sum(r["false_accept"] for r in rows),
                deployment_executed=bool(rows))


def validate_policy_artifacts(path):
    """A blocked gate cannot masquerade as measured deployment success."""
    path = Path(path)
    gate = json.loads((path / "gate.json").read_text())
    rows = json.loads((path / "policy_rows.json").read_text())["rows"]
    if gate["status"] == "BLOCKED" and rows:
        raise ValueError("Blocked policy stage cannot contain deployment endpoints")
    for row in rows:
        if row["decision_uses_reference"] or row["fallback"] == row["accepted"]:
            raise ValueError("Policy decision lineage is inconsistent")
        if row["false_accept"] != bool(row["accepted"] and not row["accurate"]):
            raise ValueError("False acceptance label differs from the measured endpoint")
        if not math.isclose(row["cost_ratio"], row["cost_seconds"] / row["classical_seconds"], rel_tol=1e-12):
            raise ValueError("Policy cost ratio differs from complete measured work")
        if not math.isclose(row["attributed_seconds"], sum(row[k] for k in ("proposal_seconds", "estimator_seconds", "fallback_seconds")), rel_tol=1e-12):
            raise ValueError("Policy omits rejected or fallback work")
        if row["attributed_seconds"] > row["component_sample_seconds"] * 1.001:
            raise ValueError("Policy component work exceeds its measured representative call")
        sample = row["cost"]["samples_seconds"][row["component_sample_index"]]
        if sample != row["component_sample_seconds"]:
            raise ValueError("Policy representative component sample differs from raw timing")
        calibration = json.loads((path / row["calibration_file"]).read_text())
        if digest(calibration) != row["calibration_sha256"]:
            raise ValueError("Policy frozen calibration differs from its row provenance")
    return {"endpoints": len(rows), "gate_status": gate["status"]}
