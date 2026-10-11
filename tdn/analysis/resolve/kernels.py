"""Frozen interaction kernels and separately charged execution profiling.

Two distinct hypotheses are tested. ``quad4_batched`` is *the same* GL4 rule
with node work batched and shared transforms; it cannot earn a representation
claim. ``quad3_midpoint_elided`` is a different, degree-five quadrature rule.
Its midpoint contributes exactly zero to the zero-subtracted DF defect, so
only two off-centre responses are evaluated. The implicit midpoint weight is
retained in the mathematical rule. Neither change supplies a stability proof.
"""
from __future__ import annotations

import statistics
import time

import torch
from torch import nn

from tdn.analysis.frontier import numerics as physical
from tdn.analysis.roadmap.numerics import background_response, quadrature, quadratic_df_defect
from tdn.numerics.operators import broadcast_h


def effective_rule(nodes):
    """Return only nonzero response nodes, plus their complete GL rule."""
    if nodes not in (2, 3, 4):
        raise ValueError("Resolve kernels support the frozen GL2, GL3 and GL4 rules")
    positions, weights = quadrature(nodes)
    active = [(x, w) for x, w in zip(positions, weights) if x != .5]
    return dict(nodes=nodes, positions=list(positions), weights=list(weights),
                active_positions=[x for x, _ in active], active_weights=[w for _, w in active],
                midpoint_weight=sum(w for x, w in zip(positions, weights) if x == .5),
                active_response_count=len(active), polynomial_exactness=2*nodes-1,
                weight_scope="Complete GL weights sum to one; omitted midpoint response is identically zero")


def batched_quadratic_defect(u, h, equation, geometry, *, track="discrete", nodes=4):
    """Same finite-step coefficient as sequential GLn, up to roundoff.

The centered input transform is shared across all quadrature nodes. Their
transported products and the split midpoint are transformed as one batch;
weighted transported spectra are reduced before the final inverse transform.
Continuum products use the same Nyquist-aware dealiased projection as controls.
There is no state caching, compression, precision reduction or hidden teacher.
"""
    step = physical.validate(u, h, geometry, track)
    rule = effective_rule(nodes)
    if equation.kappa == 0 or equation.reaction_rate == 0:
        return u * 0 + step * 0
    axes = tuple(range(2, u.ndim))
    mean = u.mean(axes, keepdim=True)
    centered = u - mean
    constant = (u == u[(slice(None), slice(None), *([slice(0, 1)]*geometry.ndim))]).flatten(2).all(-1)
    constant = constant.reshape(u.shape[0], 1, *([1]*geometry.ndim))
    mean = torch.where(constant, torch.full_like(mean, .5), mean)
    shape = (u.shape[0], 1, *([1]*geometry.ndim))
    step = step.expand(shape)
    positions = torch.tensor(rule["active_positions"]+[.5], dtype=u.dtype, device=u.device)
    weights = torch.tensor(rule["active_weights"], dtype=u.dtype, device=u.device)
    count, batch = len(positions), u.shape[0]
    node_shape = (count, *([1]*u.ndim))
    node_times = positions.reshape(node_shape) * step.unsqueeze(0)
    symbol = physical.diffusion_symbol(u, equation, geometry, track)
    initial = torch.fft.fftn(centered, dim=axes)
    # Leading extra dimension indexes nodes; flatten it with independent fields
    # only when calling the scalar-field Galerkin product implementation.
    propagated = torch.fft.ifftn(initial.unsqueeze(0)*torch.exp(node_times*symbol),
                                 dim=tuple(range(3, u.ndim+1))).real
    flat = propagated.reshape(count*batch, *u.shape[1:])
    products = physical.product(flat, flat, track).reshape(count, *u.shape)
    transformed = torch.fft.fftn(products, dim=tuple(range(3, u.ndim+1)))
    transported = transformed * torch.exp((step.unsqueeze(0)-node_times)*symbol)
    c = mean.unsqueeze(0).expand(count, *mean.shape).reshape(count*batch, *mean.shape[1:])
    s = node_times.reshape(count*batch, *step.shape[1:])
    _, q, _ = background_response(c, s, equation.reaction_rate)
    q = q.reshape(count, *mean.shape)[:-1]
    weighted = weights.reshape(count-1, *([1]*u.ndim))*q
    spectrum = (weighted*(transported[:-1]-transported[-1:])).sum(0)
    result = torch.fft.ifftn(spectrum, dim=axes).real
    _, qh, _ = background_response(mean, step, equation.reaction_rate)
    answer = -equation.reaction_rate*qh*step*result
    return torch.where(constant | (step == 0), torch.zeros_like(answer), answer)


class BatchedQuadratureModel(nn.Module):
    """Uncompressed DF plus a frozen normalized physical interaction rule."""
    def __init__(self, track="discrete", *, nodes=4, reaction_substeps=4, cache_entries=32):
        super().__init__()
        if track not in physical.TRACKS:
            raise ValueError("Explicit spatial target required")
        effective_rule(nodes)
        if type(reaction_substeps) is not int or reaction_substeps < 1:
            raise ValueError("Positive reaction subdivisions required")
        self.track, self.nodes = track, nodes
        self.reaction_substeps = reaction_substeps
        self.cache = physical.CoefficientCache(cache_entries)
        self.family = f"quad{nodes}_batched"

    def clear_cache(self):
        self.cache.clear()

    @property
    def cache_metadata(self):
        return self.cache.metadata

    def _apply(self, fn, recurse=True):
        self.clear_cache()
        return super()._apply(fn, recurse=recurse)

    def correction_components(self, u, h, equation, geometry):
        base = physical.df_step(u, h, equation, geometry, self.track,
                                reaction_substeps=self.reaction_substeps, cache=self.cache)
        increment = batched_quadratic_defect(u, h, equation, geometry, track=self.track, nodes=self.nodes)
        return dict(base=base, increment=increment)

    def forward(self, u, h, equation, geometry):
        parts = self.correction_components(u, h, equation, geometry)
        return torch.where(broadcast_h(h, u) == 0, u, parts["base"]+parts["increment"])

    def architecture_metadata(self):
        return dict(family=self.family, ownership="Ours execution prototype" if self.nodes != 3 else "Ours numerical prototype",
            track=self.track, parameters=0, learned=False, rule=effective_rule(self.nodes),
            same_rule_as_sequential=True, state_cache=False, compression=False,
            exact_nulls=["constant", "time=0", "diffusion=0", "reaction=0"],
            temporal_order_claim="No full-method or stiff-uniform order claim",
            novelty="New to this execution study; no literature novelty claim", stability_certificate=False)


class SequentialGL3(BatchedQuadratureModel):
    """Reference execution of all three nodes, retaining the zero midpoint."""
    def __init__(self, track, **options):
        super().__init__(track, nodes=3, **options)
        self.family = "quad3_sequential"

    def correction_components(self, u, h, equation, geometry):
        base = physical.df_step(u, h, equation, geometry, self.track,
                                reaction_substeps=self.reaction_substeps, cache=self.cache)
        return dict(base=base, increment=quadratic_df_defect(u, h, equation, geometry, nodes=3, target=self.track))


def method_roster(track, *, reaction_substeps=4):
    from tdn.analysis.advance.models import make_model
    models = {name:make_model(name, track, dict(reaction_substeps=reaction_substeps, quad_nodes=4, cubic_nodes=4))
              for name in ("df", "quad2_full", "quad4_full", "analytic_quad_cubic")}
    models.update(quad3_sequential=SequentialGL3(track, reaction_substeps=reaction_substeps),
                  quad2_batched=BatchedQuadratureModel(track, nodes=2, reaction_substeps=reaction_substeps),
                  quad3_midpoint_elided=BatchedQuadratureModel(track, nodes=3, reaction_substeps=reaction_substeps),
                  quad4_batched=BatchedQuadratureModel(track, nodes=4, reaction_substeps=reaction_substeps))
    return models


def profile_call(call, *, device="cpu"):
    """One separately charged operator trace, never a deployment timing sample.

    Counts are PyTorch operator invocations. A batched FFT invocation may
    transform many fields; input shapes are retained to prevent conflating
    invocation reduction with equivalent arithmetic reduction.
    """
    gpu = torch.device(device).type == "cuda"
    activities = [torch.profiler.ProfilerActivity.CPU]
    if gpu:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.no_grad(), torch.profiler.profile(activities=activities, record_shapes=True) as trace:
        answer = call()
        if gpu:
            torch.cuda.synchronize(device)
    elapsed = time.perf_counter()-start
    rows = []
    for event in trace.key_averages(group_by_input_shape=True):
        rows.append(dict(operator=event.key, calls=event.count, input_shapes=event.input_shapes,
                         self_cpu_microseconds=event.self_cpu_time_total,
                         self_device_microseconds=getattr(event, "self_device_time_total", None) if gpu else None))
    groups = {
        "fft_dispatch": lambda name:name.startswith("aten::_fft_"),
        "products_and_scalar_multiplication": lambda name:name in ("aten::mul", "aten::square", "aten::pow"),
        "reductions": lambda name:name in ("aten::mean", "aten::sum", "aten::all", "aten::amax"),
        "allocations": lambda name:name.startswith("aten::empty"),
        "copies_or_transfers": lambda name:name in ("aten::_to_copy", "aten::copy_", "aten::to"),
        "scalar_reads": lambda name:name in ("aten::item", "aten::_local_scalar_dense"),
    }
    return dict(profile_seconds=elapsed, operators=rows,
        invocation_counts={key:sum(r["calls"] for r in rows if predicate(r["operator"])) for key,predicate in groups.items()},
        output_finite=bool(torch.isfinite(answer).all()),
        scope="Separate warm operator profile. Counts include scalar products and validation; not a FLOP or physical product count.",
        gpu_utilization=None, energy_joules=None, device_kernel_launch_count=None,
        limitation="Profiler overhead is charged separately; CPU times are not synchronized CUDA kernel latency")


def _rollout(model, u, horizon, steps, equation, geometry):
    # Same loop for every method: no private DF fusion or output reuse.
    answer = u
    for _ in range(steps):
        answer = model(answer, horizon/steps, equation, geometry)
    return answer


def run(ctx):
    """Bounded development profiling; deliberately no confirmation selection."""
    from tdn.analysis.advance.diagnostics import probe_field
    from tdn.analysis.advance.metrics import endpoint_metrics, measure_paired_metrics
    from tdn.analysis.frontier.data import lawson_reference
    from tdn.analysis.frontier.measurement import endpoint_eligibility
    from tdn.analysis.frontier.core import check, clean
    from tdn.numerics import Equation, Geometry
    from tdn.runtime.metadata import write_json

    options = ctx.protocol["resolve"]["kernels"]
    track, n = ctx.unit["track"], ctx.unit["grid"]
    geom = Geometry((n, n), (1., 1.))
    horizon = float(options["horizon"])
    equation = Equation(float(options.get("kappa", .004)), float(options.get("reaction_rate", 3.)))
    models = method_roster(track, reaction_substeps=options.get("reaction_substeps", 4))
    models = {name:model.to(device=ctx.device, dtype=torch.float32).eval() for name,model in models.items()}
    rows, profiles, timings, references, comparisons, representations = [], [], [], [], [], []
    ownership = {name:("Theirs" if name == "df" else "Ours numerical prototype" if name == "quad3_midpoint_elided"
        else "Ours execution prototype" if name.endswith("_batched") else "Analytic control") for name in models}
    fields = [(regime, phase) for regime in options["regimes"] for phase in range(options.get("phase_replicates", 1))]
    for field_index, (regime, phase) in enumerate(fields):
        ctx.budget.check()
        seed = options["seed"]+field_index
        cpu = probe_field(n, regime, options.get("amplitude", .04), seed)
        tick = time.perf_counter()
        steps = options["reference_steps"]
        teacher_coarse = lawson_reference(cpu, horizon, equation, geom, steps, track, check_budget=ctx.budget.check)
        teacher_low = lawson_reference(cpu, horizon, equation, geom, 2*steps, track, check_budget=ctx.budget.check)
        teacher = lawson_reference(cpu, horizon, equation, geom, 4*steps, track, check_budget=ctx.budget.check)
        delta = teacher-teacher_low
        coarse_delta = float((teacher_low-teacher_coarse).abs().max())
        finest_delta = float(delta.abs().max())
        contracting = finest_delta <= max(1e-12, 1.1*coarse_delta)
        ref = dict(accepted=finest_delta <= options.get("reference_tolerance", 2e-7) and contracting,
            uncertainty_rms=float(delta.square().mean().sqrt()), uncertainty_max_bound=float(delta.abs().max()),
            coarse_difference_max=coarse_delta, finest_difference_max=finest_delta,
            refinement_contracting=contracting, roundoff_floor=1e-12,
            teacher_seconds=time.perf_counter()-tick, temporal_steps=[steps, 2*steps, 4*steps], spatial_grid=n,
            target="Same-grid specified FD/nodal or Galerkin equation; not spatially converged continuum truth",
            uncertainty_is_certificate=False)
        parent = f"resolve-kernel-{track}-{n}-{seed}-{regime}"
        references.append(dict(parent_id=parent, **ref))
        u = cpu.to(device=ctx.device, dtype=torch.float32)
        for count in options["schedules"]:
            ctx.budget.check()
            calls = {name:(lambda model=model:_rollout(model, u, horizon, count, equation, geom)) for name,model in models.items()}
            # First inspect the unmodified implementations. Profiling precedes
            # latency, and all model coefficient caches are then cleared.
            if field_index == 0 and count == options["schedules"][0]:
                for name in ("df", "quad2_full", "quad4_full", "analytic_quad_cubic", "quad3_sequential",
                             "quad2_batched", "quad3_midpoint_elided", "quad4_batched"):
                    ctx.budget.check()
                    if ctx.device == "cuda": torch.cuda.synchronize(ctx.device)
                    warm_start = time.perf_counter()
                    with torch.no_grad(): calls[name]()
                    if ctx.device == "cuda": torch.cuda.synchronize(ctx.device)
                    warm_seconds = time.perf_counter()-warm_start
                    profiles.append(dict(method=name, track=track, grid=n, schedule_steps=count,
                                         warmup_seconds=warm_seconds,
                                         **profile_call(calls[name], device=ctx.device)))
            for model in models.values(): model.clear_cache()
            answers, timing = measure_paired_metrics(calls, device=ctx.device,
                repeats=options["repeats"], warmup=options["warmup"], seed=seed+count,
                budget=ctx.budget, memory_probe=True)
            timings.append(dict(parent_id=parent, schedule_steps=count, **timing))
            group = {}
            for name, answer in answers.items():
                metrics = endpoint_metrics(answer, teacher, cpu, equation, geom, track, ref)
                measured = timing["methods"][name]
                row = dict(parent_id=parent, field_cluster=parent, method=name, model_id=name,
                    ownership=ownership[name],
                    regime=regime, phase_replicate=phase, track=track, grid=n, schedule_steps=count, schedule=[horizon/count]*count,
                    final_time=horizon, cost_seconds=measured["median_seconds"],
                    first_invocation_seconds=measured["cold_seconds"], raw_seconds=measured["samples_seconds"],
                    memory=measured["memory"], cache=models[name].cache_metadata,
                    parameters=0, training_seconds=0., reference_seconds=ref["teacher_seconds"],
                    **{key:value for key,value in metrics.items() if key not in ("grid", "track")})
                row["accuracy_status"] = endpoint_eligibility(row, options["rms_target"], options["max_target"])
                row["scope"] = "Frozen development methods; measured complete-call latency, no validation-selected or confirmatory advantage"
                rows.append(row); group[name] = row
                ctx.record(f"kernel/{regime}/{phase}/{count}/{name}", ["R4", "R5"], metrics=row, checks=[
                    check("reference-accepted", ref["accepted"], True, "eq", category="math"),
                    check("finite", metrics["finite"], True, "eq", category="correctness"),
                    check("rms-target", metrics["upper_rms"], options["rms_target"]),
                    check("maximum-target", metrics["upper_max"], options["max_target"])])
            for optimized, baseline in (("quad2_batched", "quad2_full"), ("quad4_batched", "quad4_full"),
                                        ("quad3_midpoint_elided", "quad3_sequential")):
                error = float((answers[optimized]-answers[baseline]).abs().max())
                eligible = all(group[key]["accuracy_status"] == "ELIGIBLE" for key in (optimized,baseline))
                ratios = [a/b for a,b in zip(group[baseline]["raw_seconds"], group[optimized]["raw_seconds"])]
                compare = dict(parent_id=parent, grid=n, track=track, schedule_steps=count,
                    candidate=optimized, comparator=baseline, maximum_parity_error=error,
                    both_accuracy_eligible=eligible, paired_round_speed_ratios=ratios,
                    median_paired_speed_ratio=statistics.median(ratios),
                    accuracy_qualified_speed_ratio=statistics.median(ratios) if eligible else None,
                    interpretation="Same mathematical rule execution comparison, not a representation or population speed guarantee")
                comparisons.append(compare)
                ctx.record(f"execution/{regime}/{phase}/{count}/{optimized}", ["R4", "R5"], metrics=compare, checks=[
                    check("fp32-parity", error, options.get("parity_tolerance", 3e-6), category="correctness"),
                    check("qualified-development-speed", compare["accuracy_qualified_speed_ratio"], options.get("practical_speedup", 1.2), "ge", category="gap")])
            for control in ("quad2_batched", "quad4_batched", "analytic_quad_cubic"):
                candidate = group["quad3_midpoint_elided"]
                comparator = group[control]
                eligible = candidate["accuracy_status"] == comparator["accuracy_status"] == "ELIGIBLE"
                ratios = [a/b for a,b in zip(comparator["raw_seconds"], candidate["raw_seconds"])]
                row = dict(parent_id=parent, track=track, grid=n, schedule_steps=count,
                    candidate="quad3_midpoint_elided", comparator=control,
                    candidate_error_rms=candidate["error_rms"], comparator_error_rms=comparator["error_rms"],
                    control_over_candidate_rms_ratio=(comparator["error_rms"]/candidate["error_rms"]
                        if candidate["error_rms"] else None),
                    both_accuracy_eligible=eligible, paired_round_speed_ratios=ratios,
                    accuracy_qualified_speed_ratio=statistics.median(ratios) if eligible else None,
                    scope="Matched fixed schedule development comparison; no monotonic quadrature, population or validation-frontier guarantee")
                representations.append(row)
                ctx.record(f"representation/{regime}/{phase}/{count}/{control}", ["R4", "R5"], metrics=row, checks=[
                    check("both-meet-target", eligible if ref["accepted"] else None, True, "eq", category="gap"),
                    check("qualified-development-speed", row["accuracy_qualified_speed_ratio"], options.get("practical_speedup", 1.2), "ge", category="gap")])
            write_json(ctx.path/"kernel_rows.json", dict(rows=clean(rows), comparisons=clean(comparisons),
                representation_comparisons=clean(representations), references=clean(references)))
            write_json(ctx.path/"profiling.json", dict(profiles=clean(profiles), paired_timing=clean(timings),
                prototype=effective_rule(3), profile_before_optimized_comparison=True,
                profile_total_seconds=sum(p["profile_seconds"] for p in profiles),
                profile_warmup_total_seconds=sum(p["warmup_seconds"] for p in profiles),
                complete_campaign_cost="Use scheduler accounting plus separately recorded references, profiles, warmups, probes and repeated calls",
                monetary_cost=None, energy_joules=None))
    return dict(methods=list(models), endpoint_rows=len(rows), execution_comparisons=len(comparisons),
        representation_comparisons=len(representations),
        profiling_scope="One separately charged representative development field; no extrapolated kernel cost model",
        new_representation="GL3 rule with provably zero midpoint response elided; two active off-centre nodes",
        no_training=True, no_confirmation_access=True)
