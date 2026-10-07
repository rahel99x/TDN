"""Accuracy headroom, spatial consistency, and charged scaling controls."""
from __future__ import annotations

import math
import statistics
import time

import torch

from tdn.numerics import Equation, Geometry, choose_substeps, refined_reference
from tdn.analysis.agenda.data import continuous_field, _continuum_reference
from tdn.analysis.agenda.engine import rollout
from tdn.analysis.agenda.physics import diffusion_first_step
from tdn.research.protocol import digest
from .core import check

CLASSICAL = ("strang_diffusion_first", "strang_reaction_first", "etdrk4", "gl3_fused")


def discrepancy(a, b):
    difference = a.detach().cpu().double() - b.detach().cpu().double()
    return dict(rms=float(difference.square().mean().sqrt()), max=float(difference.abs().max()))


def reference(u, h, equation, geometry, ctx):
    count = max(8, choose_substeps(h, equation, geometry))
    started = time.perf_counter()
    result = None
    for _ in range(ctx.protocol["reference_attempts"]):
        ctx.budget.check()
        if 4 * count > ctx.protocol["max_finest_substeps"]:
            break
        result = refined_reference(u.double().cpu(), h, equation, geometry, count,
            tolerance=20 * ctx.protocol["reference_tolerance"] / math.sqrt(u.numel()), check=ctx.budget.check)
        if result.accepted:
            break
        count *= 2
    return result, time.perf_counter() - started


def run_headroom(ctx):
    from .numerics import fourier_resample
    p, options = ctx.protocol, ctx.protocol["headroom"]
    parents = [v for v in p["parents"] if v["split"] == "train"][:options["clusters"]]
    candidates, spatial_rows, fusion_errors = [], [], []
    for parent in parents:
        equation = Equation(parent["kappa"], parent["reaction_rate"])
        for h in options["horizons"]:
            grid_values, reference_rows, reference_uncertainties = {}, {}, {}
            try:
                continuum, continuum_meta = _continuum_reference(parent, h, p, ctx.budget)
            except FloatingPointError as error:
                continuum, continuum_meta = None, {"accepted": False, "failure": str(error)}
            for n in options["grids"]:
                ctx.budget.check()
                geometry, u = Geometry((n, n), (1., 1.)), continuous_field(parent, n)
                ref, ref_seconds = reference(u, h, equation, geometry, ctx)
                ref_ok = ref is not None and ref.accepted
                if ref_ok:
                    grid_values[n] = ref.state
                    reference_uncertainties[n] = ref.uncertainty
                    spatial = discrepancy(ref.state, fourier_resample(continuum, (n, n))) if continuum is not None else {"rms": None, "max": None}
                    reference_rows[n] = spatial
                    if continuum is not None:
                        spatial_rows.append(dict(parent_id=parent["parent_id"], n=n, h=h, **spatial))
                else:
                    spatial = {"rms": None, "max": None}
                for family in CLASSICAL:
                    for count in options["steps"]:
                        schedule = [h / count] * count
                        failure = None
                        try:
                            output, costs = ctx.measure(lambda: rollout(None, u, schedule, equation, geometry,
                                family=family, budget=ctx.budget), warmup=p["timing_warmup"])
                            finite = bool(torch.isfinite(output).all())
                            errors = discrepancy(output, ref.state) if ref_ok else {"rms": None, "max": None}
                        except (FloatingPointError, ValueError) as error:
                            finite, costs, errors, failure = False, {}, {"rms": None, "max": None}, str(error)
                        uncertainty = ref.uncertainty if ref_ok else None
                        upper = {norm: errors[norm] + uncertainty * (1 if norm == "rms" else n)
                                 if ref_ok and errors[norm] is not None else None for norm in ("rms", "max")}
                        passes = finite and ref_ok and all(v <= options["target"] for v in upper.values())
                        row = dict(parent_id=parent["parent_id"], field_cluster=parent["field_cluster"], grid=n,
                            horizon=h, family=family, steps=count, feasible=passes, reference_accepted=ref_ok, upper=upper, costs=costs)
                        candidates.append(row)
                        ctx.record(f"classical/{parent['parent_id']}/{n}/{h}/{family}/{count}", ["M00", "M16"],
                            metrics={**costs, "parameters": 0, "error_rms": errors["rms"], "error_max": errors["max"],
                                "upper_rms": upper["rms"], "upper_max": upper["max"], "teacher_seconds": ref_seconds,
                                "shared_reference_id": f"FD/{parent['parent_id']}/N{n}/h{h}",
                                "teacher_seconds_scope": "one shared teacher per condition; repeated metadata must not be summed across solver rows",
                                "teacher_seconds_charged": ref_seconds if family == CLASSICAL[0] and count == options["steps"][0] else 0.},
                            config={"family": family, "grid": n, "schedule": schedule, "parent_id": parent["parent_id"],
                                "target": "FD/nodal", "precision": "float64", "failure": failure},
                            checks=[check("finite-trajectory", finite, True, "eq", category="correctness"),
                                check("independent-refinement-accepted", ref_ok, True, "eq", category="math"),
                                check("matched-rms-target", upper["rms"], options["target"], category="gap"),
                                check("matched-max-target", upper["max"], options["target"], category="gap")])
                # Compare exactly the same variable-step DF map before/after
                # fusion. Corrections are never moved across a heat stage.
                schedule = [h / 6, h / 3, h / 2]
                unfused = lambda: _unfused_df(u, schedule, equation, geometry)
                fused = lambda: rollout(None, u, schedule, equation, geometry, family="strang_diffusion_first")
                a, ca = ctx.measure(unfused); b, cb = ctx.measure(fused)
                parity = discrepancy(a, b)["max"]; fusion_errors.append(parity)
                ctx.record(f"df-fusion/{parent['parent_id']}/{n}/{h}", ["M00", "M15"],
                    metrics={**cb, "unfused_seconds": ca["median_seconds"], "speedup": ca["median_seconds"] / cb["median_seconds"],
                        "parity_max": parity, "parameters": 0, "heat_calls_fused": len(schedule) + 1, "heat_calls_unfused": 2 * len(schedule)},
                    config={"schedule": schedule, "grid": n, "field_cluster": parent["field_cluster"]},
                    checks=[check("same-map-fusion", parity, 2e-12, category="math"),
                            check("fusion-saves-heat-calls", len(schedule) + 1, 2 * len(schedule), "lt", category="gap"),
                            check("measured-fusion-speedup", ca["median_seconds"] / cb["median_seconds"], 1., "ge", category="utility")])
            if all(n in grid_values for n in options["grids"]):
                n, n2, n4 = options["grids"]
                d1 = discrepancy(grid_values[n], fourier_resample(grid_values[n2], (n, n)))
                d2 = discrepancy(fourier_resample(grid_values[n2], (n, n)), fourier_resample(grid_values[n4], (n, n)))
                for norm in ("rms", "max"):
                    ratio = d1[norm] / max(d2[norm], 1e-15)
                    refinement_noise = (reference_uncertainties[n2] + reference_uncertainties[n4]) * (1 if norm == "rms" else n)
                    resolved = d2[norm] > max(1e-12, 10 * refinement_noise) and 2.5 <= ratio <= 6.
                    estimate = 4 * d2[norm] / 3 if resolved else None
                    ctx.record(f"spatial/{parent['parent_id']}/{h}/{norm}", ["M01", "M05"],
                        metrics={"parameters": 0, "coarse_difference": d1[norm], "fine_difference": d2[norm],
                            "ratio": ratio, "refinement_noise_estimate": refinement_noise, "estimated_2N_error": estimate, "estimated_4N_error": estimate / 4 if estimate else None,
                            "continuum_discrepancy_coarse": reference_rows[n][norm],
                            "continuum_spatial_uncertainty": continuum_meta.get("spatial_resolution_difference_" + norm)},
                        config={"norm": norm, "grids": options["grids"], "same_continuous_field": parent["continuous_field_id"],
                            "error_estimate_is_certificate": False, "teacher_target": "dealiased spectral continuum estimate"},
                        checks=[check("matched-continuous-field", True, True, "eq", category="correctness"),
                            check("observed-second-order-regime", resolved, True, "eq", category="math"),
                            check("spatial-budget-at-2N", estimate, options["target"] / 2, category="gap",
                                reason="Richardson estimate only when refinement is resolved and approximately second order")])
    headroom = summarize_headroom(candidates)
    eligible = headroom["eligible_conditions"]
    ctx.record("C0/headroom-spatial-independent-controls", ["M00", "M01", "M05", "M16", "M18", "M22"], combination_ids=["C0"],
        metrics={"parameters": 0, "independent_development_clusters": len({v['field_cluster'] for v in parents}),
            **headroom, "spatial_audit_rows": len(spatial_rows), "maximum_fusion_parity_error": max(fusion_errors, default=0.)},
        config={"classical_controls": list(CLASSICAL), "headroom_options": options,
            "eligibility": "accepted independent teacher and at least one accurate classical candidate",
            "cohort": "training/development; never confirmation", "neural_only_comparison": "separate confirm stage",
            "published_paper_reproduction": False},
        evidence=["rows.jsonl"], checks=[check("source-declaration-bound", bool(p.get("mechanisms")), True, "eq", category="correctness"),
            check("fusion-preserves-df-map", max(fusion_errors, default=math.inf), 2e-12, category="math"),
            check("some-classical-accuracy-headroom", headroom["conditions_needing_more_than_one_bare_df_step"] if eligible else None, 1, "ge", category="gap",
                  reason="Unresolved references or an entirely infeasible classical grid cannot establish headroom"),
            check("complete-eligible-classical-controls", headroom["unresolved_conditions"], 0, "eq", category="utility",
                  applicable=bool(eligible), reason="Unresolved conditions remain visible and are excluded from headroom evidence")])
    return dict(classical_candidates=len(candidates), **headroom)


def summarize_headroom(candidates):
    """Headroom is identifiable only against a resolved, feasible comparator."""
    grouped = {}
    for row in candidates:
        grouped.setdefault((row["parent_id"], row["grid"], row["horizon"]), []).append(row)
    eligible, missing_reference, infeasible, needs_more = 0, 0, 0, 0
    for rows in grouped.values():
        accepted = all(row.get("reference_accepted", False) for row in rows)
        feasible = any(row["feasible"] for row in rows)
        if not accepted:
            missing_reference += 1
        elif not feasible:
            infeasible += 1
        else:
            eligible += 1
            needs_more += not any(row["family"] == "strang_diffusion_first" and row["steps"] == 1 and row["feasible"] for row in rows)
    return dict(conditions=len(grouped), eligible_conditions=eligible,
        conditions_needing_more_than_one_bare_df_step=needs_more,
        accepted_classical_coverage=eligible, unresolved_reference_conditions=missing_reference,
        no_feasible_classical_conditions=infeasible, unresolved_conditions=missing_reference + infeasible,
        headroom_conditions=needs_more)


def _unfused_df(u, schedule, equation, geometry):
    for h in schedule:
        u = diffusion_first_step(u, h, equation, geometry)
    return u


class PreparedCache:
    """Reuse immutable coefficients only for an identical physical operator.

    State values and batch count are deliberately absent from the key; every
    cached primitive validates geometry/dtype/device on every call. Time,
    equation, lengths/grid, precision, device and solver family are all keyed.
    The cache lives inside one charged experiment, never across timing calls.
    """
    def __init__(self):
        self.entries = {}
        self.hits = 0
        self.preparations = 0

    def get(self, example, h, equation, geometry, family):
        if family not in {"etdrk4", "gl3_fused"}:
            raise ValueError("Prepared cache experiment supports ETDRK4 and fused GL3")
        if isinstance(h, torch.Tensor) and (h.ndim or h.requires_grad):
            raise ValueError("Cache requires a frozen scalar horizon")
        step = float(h)
        if not math.isfinite(step) or step < 0:
            raise ValueError("Cache horizon must be finite and nonnegative")
        key = (family, step, equation.kappa, equation.reaction_rate,
               geometry.grid, geometry.lengths, example.dtype, str(example.device))
        if key in self.entries:
            self.hits += 1
            return self.entries[key]
        from tdn.research.work_precision import prepare_step
        from tdn.research.compact_spatial import prepare_compact_step
        factory = prepare_compact_step if family == "gl3_fused" else prepare_step
        prepared = factory(example, step, equation, geometry, family)
        self.entries[key] = prepared
        self.preparations += 1
        return prepared

    @property
    def metadata(self):
        return {"coefficient_preparations": self.preparations, "cache_hits": self.hits,
                "cached_tensor_bytes": sum(value.metadata["cached_tensor_bytes"] for value in self.entries.values()),
                "state_dependent_cache": False, "cache_entries": len(self.entries)}


def prepared_workload(initial, schedule, equation, geometry, family, *, cached=True, queries=3, budget=None):
    """Charge cold coefficient construction plus every query/rollout call."""
    if type(queries) is not int or queries < 1 or not schedule:
        raise ValueError("Positive queries and a nonempty schedule are required")
    cache = PreparedCache()
    outputs, preparations, tensor_bytes = [], 0, 0
    for _ in range(queries):
        value = initial.clone()
        for h in schedule:
            if budget:
                budget.check()
            owner = cache if cached else PreparedCache()
            method = owner.get(value, h, equation, geometry, family)
            value = method(value)
            if not cached:
                preparations += owner.preparations
                tensor_bytes = max(tensor_bytes, owner.metadata["cached_tensor_bytes"])
        outputs.append(value)
    metadata = cache.metadata if cached else {"coefficient_preparations": preparations, "cache_hits": 0,
                                              "cached_tensor_bytes": tensor_bytes, "state_dependent_cache": False, "cache_entries": 0}
    return torch.stack(outputs), {**metadata, "queries": queries, "steps_per_query": len(schedule),
        "cost_scope": "coefficient construction plus all complete query rollouts; no persistent warm cache"}


def run_preparation_reuse(ctx, parent):
    n = ctx.protocol["train_grid"]
    equation = Equation(parent["kappa"], parent["reaction_rate"])
    geometry = Geometry((n, n), (1., 1.))
    initial = continuous_field(parent, n).to(device=ctx.device, dtype=torch.float32)
    horizon = float(sum(ctx.protocol["confirm_schedules"][0]))
    schedule = [horizon / 4, horizon / 2, horizon / 4]
    for family in ("etdrk4", "gl3_fused"):
        fresh, fresh_cost = ctx.measure(lambda: prepared_workload(initial, schedule, equation, geometry,
            family, cached=False, queries=3, budget=ctx.budget), repeats=ctx.protocol["scaling"]["repeats"])
        reused, reused_cost = ctx.measure(lambda: prepared_workload(initial, schedule, equation, geometry,
            family, cached=True, queries=3, budget=ctx.budget), repeats=ctx.protocol["scaling"]["repeats"])
        error = discrepancy(fresh[0], reused[0])["max"]
        speedup = fresh_cost["median_seconds"] / reused_cost["median_seconds"]
        ctx.record(f"preparation-reuse/{family}", ["M15"],
            metrics={**reused_cost, "parameters": 0, "fresh_cost": fresh_cost, "reused_cost": reused_cost,
                "fresh_preparations": fresh[1]["coefficient_preparations"],
                "reused_preparations": reused[1]["coefficient_preparations"],
                "coefficient_cache_bytes": reused[1]["cached_tensor_bytes"],
                "cache_hits": reused[1]["cache_hits"], "parity_max": error, "speedup": speedup},
            config={"family": family, "grid": n, "schedule": schedule, "queries": 3,
                "cost_scope": reused[1]["cost_scope"], "state_dependent_cache": False},
            checks=[check("same-operator-identical-output", error, 5e-6, category="math"),
                    check("fewer-coefficient-preparations", reused[1]["coefficient_preparations"], fresh[1]["coefficient_preparations"], "lt", category="gap"),
                    check("charged-complete-workload-speedup", speedup, 1., "ge", category="utility")])
    # Execute key changes and compare each entry to an independently fresh
    # preparation, preventing a stale coefficient cache from passing via count.
    cache = PreparedCache()
    cases = [(initial, schedule[0], equation, geometry),
             (initial, schedule[1], equation, geometry),
             (initial, schedule[0], Equation(equation.kappa * 1.5 + .001, equation.reaction_rate), geometry),
             (initial, schedule[0], Equation(equation.kappa, equation.reaction_rate + .5), geometry),
             (initial, schedule[0], equation, Geometry((n, n), (1.5, 1.))),
             (continuous_field(parent, n + 2).to(initial), schedule[0], equation, Geometry((n + 2, n + 2), (1., 1.))),
             (initial.double(), schedule[0], equation, geometry)]
    def invalidate_workload():
        owner, errors = PreparedCache(), []
        for value, step, eq, geo in cases:
            a = owner.get(value, step, eq, geo, "etdrk4")(value)
            b = PreparedCache().get(value, step, eq, geo, "etdrk4")(value)
            errors.append(discrepancy(a, b)["max"])
        owner.get(initial, schedule[0], equation, geometry, "etdrk4")
        return owner.metadata, max(errors)
    (metadata, parity), cost = ctx.measure(invalidate_workload, repeats=1)
    ctx.record("preparation-reuse/operator-invalidation", ["M15", "M22"],
        metrics={**cost, **metadata, "parity_max": parity, "parameters": 0},
        config={"changed_keys": ["h", "kappa", "reaction_rate", "physical_lengths", "grid", "dtype"],
                "device_is_keyed": True, "unobserved_device_migration": "NA; this run uses one actual device"},
        checks=[check("fresh-parity-after-each-key-change", parity, 1e-12, category="math"),
                check("distinct-physical-cache-entries", metadata["cache_entries"], len(cases), "eq", category="correctness"),
                check("unchanged-key-reused", metadata["cache_hits"], 1, "eq", category="gap")])
    return 3


def select_scaling_models(models, families=("source", "rank1", "rank2", "cheap_fno", "deep_fno", "direct_fno", "c1_rank2", "c2_rank2")):
    """Deterministic first metadata seed, independent of checkpoint ID spelling."""
    selected = {}
    for family in families:
        matches = [(key, model) for key, model in models.items() if getattr(model, "family", None) == family]
        def key(item):
            seed = getattr(item[1], "selection_metadata", {}).get("seed")
            return (int(seed) if seed is not None else math.inf, item[0])
        if matches:
            identity, model = min(matches, key=key)
            selected[identity] = model
    return selected


def _reference_errors(output, reference):
    if reference is None or not reference.get("accepted", False):
        return None
    values = discrepancy(output, reference["state"])
    for norm, name in (("rms", "uncertainty_rms"), ("max", "uncertainty_max_bound")):
        uncertainty = float(reference[name])
        if not math.isfinite(uncertainty) or uncertainty < 0:
            return None
        values["upper_" + norm] = values[norm] + uncertainty
    return values


def run_scaling(ctx):
    from .neural import load_frozen_models
    from .data import load_bank
    from tdn.research.experiment import horizon_key
    models = load_frozen_models(ctx)
    selected = select_scaling_models(models)
    if not selected:
        raise ValueError("No declared trained-family models available for scaling")
    options, count, accuracy_rows, throughput_rows = ctx.protocol["scaling"], 0, 0, 0
    parents = load_bank(ctx, "confirmation")
    if not parents:
        raise ValueError("Scaling needs a frozen confirmation engineering field")
    parent = parents[0]
    reuse_rows = run_preparation_reuse(ctx, parent)
    equation = Equation(parent["kappa"], parent["reaction_rate"])
    schedule = ctx.protocol["confirm_schedules"][0]
    horizon = horizon_key(sum(schedule))
    target = float(ctx.protocol["targets"][0])
    for n in options["grids"]:
        for batch in options["batches"]:
            if count >= options["maximum_cases"]:
                break
            count += 1
            ctx.budget.check()
            initial = continuous_field(parent, n).to(device=ctx.device, dtype=torch.float32).repeat(batch, 1, 1, 1)
            geometry = Geometry((n, n), (1., 1.))
            teacher = parent["references"].get(f"discrete:{n}:{horizon}")
            with torch.no_grad():
                base, baseline = ctx.measure(lambda: rollout(None, initial, schedule, equation, geometry,
                    family="strang_diffusion_first", budget=ctx.budget), repeats=options["repeats"], warmup=1)
                base_errors = _reference_errors(base, teacher)
                for identity, model in selected.items():
                    try:
                        result, cost = ctx.measure(lambda: rollout(model, initial, schedule, equation, geometry, budget=ctx.budget),
                            repeats=options["repeats"], warmup=1)
                        finite = bool(torch.isfinite(result).all())
                        errors = _reference_errors(result, teacher)
                        known_accuracy = errors is not None and base_errors is not None
                        accurate = known_accuracy and all(values["upper_" + norm] <= target for values in (errors, base_errors) for norm in ("rms", "max"))
                        accuracy_rows += known_accuracy
                        throughput_rows += not known_accuracy
                        scope = ("same-grid independent accepted FD teacher; repeated field in batch, not independent parents" if known_accuracy
                                 else "throughput only; independent accepted same-grid teacher absent for this workload")
                        ctx.record(f"scaling/{identity}/{n}/b{batch}", ["M15", "M16", "M03"],
                            metrics={**cost, "parameters": sum(v.numel() for v in model.parameters()),
                                "batch": batch, "seconds_per_field": cost["median_seconds"] / batch,
                                "base_seconds": baseline["median_seconds"], "speedup_over_bare_df": baseline["median_seconds"] / cost["median_seconds"],
                                "same_step_base_difference_max": discrepancy(result, base)["max"],
                                "teacher_errors": errors, "base_teacher_errors": base_errors, "joint_accuracy": bool(accurate)},
                            config={"model_id": identity, "family": model.family, "seed": getattr(model, "selection_metadata", {}).get("seed"),
                                "grid": n, "schedule": schedule, "batch": batch, "target": target,
                                "precision": "float32 TF32 off", "accuracy_scope": scope},
                            checks=[check("finite-batched-output", finite, True, "eq", category="correctness"),
                                check("teacher-backed-rms", errors["upper_rms"] if errors else None, target, category="math", reason=scope),
                                check("teacher-backed-max", errors["upper_max"] if errors else None, target, category="math", reason=scope),
                                check("accuracy-matched-rollout-latency-improvement", baseline["median_seconds"] / cost["median_seconds"], 1., "ge", category="gap",
                                      applicable=bool(accurate), reason="Latency advantage is accuracy-matched only when both outputs pass the independent teacher"),
                                check("memory-observed", cost["peak_allocated_bytes"], ctx.protocol["gpu_memory"]["soft_cap_gib"] * 2**30,
                                    category="utility", reason="CPU execution has no CUDA memory observation" if ctx.device == "cpu" else "")])
                    except torch.cuda.OutOfMemoryError as error:
                        torch.cuda.empty_cache()
                        ctx.record(f"scaling/{identity}/{n}/b{batch}", ["M15", "M03"], status="FAILED",
                            metrics={"parameters": sum(v.numel() for v in model.parameters())},
                            config={"model_id": identity, "grid": n, "batch": batch, "failure": str(error)},
                            checks=[check("fits-dedicated-vram", False, True, "eq", category="gap"),
                                check("accuracy-unavailable-after-oom", None, None, category="math")])
    return dict(workload_cases=count, models=len(selected), preparation_reuse_rows=reuse_rows,
                teacher_checked_rows=accuracy_rows, throughput_only_rows=throughput_rows,
                scope="engineering throughput; accepted bank references used where available, explicit accuracy NA elsewhere")
