"""Gate two: preserved easy/stress strata, spatial floors and solver headroom."""
from __future__ import annotations

from collections import defaultdict
import math

import torch

from tdn.analysis.agenda.physics import fourier_resample
from tdn.analysis.roadmap.core import check
from tdn.numerics import Equation
from tdn.research.experiment import horizon_key
from tdn.runtime.metadata import write_json
from .data import prepare, load_split, geometry
from .measurement import measure_paired
from .models import make_model
from .neural import rollout as complete_rollout


def summarize_headroom(candidates, target):
    """One independent question per parent/grid/track/final time, not per row."""
    groups = defaultdict(list)
    for row in candidates:
        groups[(row["parent_id"], row["grid"], row["track"], row["final_time"])].append(row)
    inventory, strata = [], defaultdict(lambda: dict(conditions=0, easy=0, headroom=0, unresolved=0, infeasible=0))
    for key, rows in sorted(groups.items()):
        def feasible(row):
            return row["reference_accepted"] and not row.get("numerical_failure", False) and all(
                row.get(k) is not None and math.isfinite(row[k]) and row[k] <= target for k in ("upper_rms", "upper_max"))
        resolved = all(row["reference_accepted"] for row in rows)
        good = [r for r in rows if feasible(r)]
        one = any(r["family"] == "df" and r["steps"] == 1 and feasible(r) for r in rows)
        outcome = "unresolved" if not resolved else "infeasible" if not good else "easy" if one else "headroom"
        fastest = min(good, key=lambda r: r["cost_seconds"]) if good else None
        regime = rows[0]["regime"]
        strata[regime]["conditions"] += 1; strata[regime][outcome] += 1
        inventory.append(dict(parent_id=key[0], grid=key[1], track=key[2], final_time=key[3], regime=regime,
            outcome=outcome, target=target, best_classical_family=fastest["family"] if fastest else None,
            best_classical_steps=fastest["steps"] if fastest else None,
            best_classical_seconds=fastest["cost_seconds"] if fastest else None))
    count = sum(r["outcome"] == "headroom" for r in inventory)
    return dict(target=target, conditions=len(inventory), headroom_conditions=count,
        solver_utility_eligible=count > 0, selection_split="development",
        eligibility_scope="at least one resolved development condition requires more than one bare DF step; no efficiency claim",
        neural_research_allowed=True, strata=dict(strata), conditions_detail=inventory)


def run(ctx):
    options, p = ctx.protocol["screening"], ctx.protocol
    prepared = prepare(ctx, splits=["development"])
    bank = load_split(ctx, "development")
    candidates, spatial, timings = [], [], []
    for parent in bank:
        equation = Equation(parent["kappa"], parent["reaction_rate"])
        for n in options["grids"]:
            initial = parent["states"][str(n)].to(device=ctx.device, dtype=torch.float64)
            geom = geometry(parent, n)
            for h in options["horizons"]:
                for track in p["tracks"]:
                    ref = parent["references"][f"{track}:{n}:{horizon_key(h)}"]
                    calls, metadata = {}, {}
                    for family in ("df", "etdrk4"):
                        for steps in options["steps"]:
                            model = make_model(family, track, p.get("model_config", {})).to(device=ctx.device, dtype=torch.float64)
                            key = f"{family}/{steps}"
                            def rollout(model=model, steps=steps):
                                with torch.no_grad():
                                    return complete_rollout(model, initial, [h / steps] * steps,
                                                            equation, geom, ctx.budget)[0]
                            calls[key] = rollout; metadata[key] = (family, steps)
                    outputs, timing = measure_paired(calls, device=ctx.device,
                        repeats=max(2, p["timing_repeats"]), warmup=max(1, p["timing_warmup"]),
                        seed=p.get("timing_seed", 0) + len(timings), budget=ctx.budget)
                    timing_id = f"{parent['parent_id']}/{track}/{n}/{horizon_key(h)}"
                    timings.append(dict(timing_id=timing_id, **timing))
                    for key, output in outputs.items():
                        family, steps = metadata[key]
                        finite = bool(torch.isfinite(output).all())
                        errors = output.detach().double().cpu() - ref["state"]
                        rms = float(errors.square().mean().sqrt()) if finite else None
                        maximum = float(errors.abs().max()) if finite else None
                        upper_rms = rms + ref["uncertainty_rms"] if finite and ref["accepted"] else None
                        upper_max = maximum + ref["uncertainty_max_bound"] if finite and ref["accepted"] else None
                        row = dict(parent_id=parent["parent_id"], field_cluster=parent["field_cluster"],
                            regime=parent["regime"], grid=n, track=track, final_time=h,
                            family=family, steps=steps, schedule=[h / steps] * steps,
                            error_rms=rms, error_max=maximum, upper_rms=upper_rms, upper_max=upper_max,
                            numerical_failure=not finite, reference_accepted=ref["accepted"],
                            cost_seconds=timing["methods"][key]["median_seconds"], cost=timing["methods"][key])
                        candidates.append(row)
                        ctx.record(f"classical/{timing_id}/{key}", ["G2"], metrics=row,
                            config={"target_convention": ref["target_convention"], "precision": "FP64", "regime": parent["regime"],
                                    "timer": "paired randomized warm latency; first invocation and warmup retained separately"},
                            checks=[check("finite-output", finite, True, "eq", category="correctness"),
                                check("independent-reference-accepted", ref["accepted"], True, "eq", category="math"),
                                check("joint-rms-target", upper_rms, options["target"], category="gap"),
                                check("joint-maximum-target", upper_max, options["target"], category="gap")])
                discrete = parent["references"][f"discrete:{n}:{horizon_key(h)}"]
                continuum = parent["references"][f"continuum:{n}:{horizon_key(h)}"]
                accepted = discrete["accepted"] and continuum["accepted"]
                delta = discrete["state"] - continuum["state"]
                floor_rms = float(delta.square().mean().sqrt())
                floor_max = float(delta.abs().max())
                uncertainty = discrete["uncertainty_rms"] + continuum["uncertainty_rms"] if accepted else None
                record = dict(parent_id=parent["parent_id"], field_cluster=parent["field_cluster"], regime=parent["regime"],
                    grid=n, horizon=h, spatial_difference_rms=floor_rms, spatial_difference_max=floor_max,
                    accepted=accepted, reference_uncertainty_rms=uncertainty,
                    resolved_spatial_floor_lower=max(0., floor_rms - uncertainty) if accepted else None)
                spatial.append(record)
                ctx.record(f"spatial-floor/{parent['parent_id']}/{n}/{horizon_key(h)}", ["G2"], metrics=record,
                    checks=[check("both-target-teachers-accepted", accepted, True, "eq", category="math"),
                        check("spatial-floor-below-requested-target", floor_rms + uncertainty if accepted else None,
                              options["target"], category="gap")])
        # Same continuous field, fixed physical final time and common projection.
        for h in options["horizons"]:
            for coarse, fine in zip(options["grids"][:-1], options["grids"][1:]):
                first = parent["references"][f"discrete:{coarse}:{horizon_key(h)}"]
                second = parent["references"][f"discrete:{fine}:{horizon_key(h)}"]
                change = first["state"] - fourier_resample(second["state"], (coarse, coarse))
                ctx.record(f"grid-refinement/{parent['parent_id']}/{coarse}-{fine}/{horizon_key(h)}", ["G2"],
                    metrics={"difference_rms": float(change.square().mean().sqrt()), "difference_max": float(change.abs().max()),
                             "coarse_grid": coarse, "fine_grid": fine, "final_time": h},
                    checks=[check("paired-physical-grid-refinement", fine, 2 * coarse, "eq", category="math"),
                            check("both-grid-teachers-accepted", first["accepted"] and second["accepted"], True, "eq", category="gap")])
    summary = summarize_headroom(candidates, options["target"])
    summary.update(schema="tdn.frontier-screen/v1", prepared=prepared, candidates=candidates,
        spatial_floors=spatial, all_targets=[summarize_headroom(candidates, target) for target in p["targets"]],
        no_confirmation_used=True, paper_reproduction=False)
    write_json(ctx.path / "screen.json", summary)
    write_json(ctx.path / "screen-timing.json", timings)
    ctx.record("screen/development-headroom", ["G2"], metrics={k: v for k, v in summary.items() if k not in ("candidates", "spatial_floors", "all_targets")},
        checks=[check("development-only-selection", summary["selection_split"], "development", "eq", category="math"),
                check("nontrivial-classical-work-exists", summary["headroom_conditions"], 0, "gt", category="gap")])
    return {k: v for k, v in summary.items() if k not in ("candidates", "spatial_floors", "all_targets", "conditions_detail")}
