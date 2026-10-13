"""Physical, parent-aware evaluation; no teacher data enters an inference call.

Endpoint composition re-encodes intermediate states. Tangent composition is a
different observation along one declared anchor path. Refinement differences,
balance residuals and sensitivity checks are diagnostics, not certificates.
"""
from __future__ import annotations

from collections import defaultdict
import math
import time
from typing import Callable, Mapping

import torch

from .contracts import Geometry, Physics, physical_metrics as contract_metrics
from tdn.analysis.response import numerics as n
from tdn.analysis.response.trajectory import moment_rhs

TARGETS = ("fd_same_grid", "galerkin_same_grid")
SHIFT_AXES = ("mean", "amplitude", "phase", "frequency", "horizon", "reaction", "diffusion", "resolution")


def sample_context(sample):
    record = sample["record"]
    target = record["target_kind"]
    if target not in TARGETS:
        raise ValueError("UNSUPPORTED_TARGET: projected-fine dynamics require a separate lift/closure contract")
    geom = sample.get("geom") or Geometry(tuple(record["grid"]), tuple(record["lengths"]))
    eq = sample.get("eq") or Physics(record["kappa"], record["reaction_rate"])
    return record, eq, geom, target


def teacher(sample):
    return sample.get("teacher64", sample.get("target"))


def reference_status(sample):
    record = sample.get("record", {})
    ref = sample.get("reference", record.get("reference", {}))
    accepted = sample.get("accepted", sample.get("reference_accepted", ref.get("accepted", False)))
    return bool(accepted), ref


def physical_metrics(error, geom, target, *, maximum_oversampling=2):
    if target not in TARGETS:
        raise ValueError("Unsupported same-equation physical norm")
    error = error.detach().double()
    if not bool(torch.isfinite(error).all()):
        return dict(finite=False, mse=None, rms=None, maximum=None, mean_error=None, centered_mse=None,
                    centered_rms=None, mean_mse=None, maximum_scope="sampled nodal maximum")
    if type(maximum_oversampling) is not int or not 1<=maximum_oversampling<=16:
        raise ValueError("Sampled maximum oversampling must be a declared bounded integer")
    values = contract_metrics(error,geom,target,maximum_oversampling=maximum_oversampling)
    return dict(values,finite=True,mean_mse=values["mean_squared"],
                norm="FD nodal L2" if target == "fd_same_grid" else "Galerkin physical interpolant L2, Nyquist mass weighted",
                maximum_scope="sampled physical interpolant maximum; not a continuous supremum certificate" if target == "galerkin_same_grid" else "sampled nodal maximum")


def endpoint_metrics(prediction, sample):
    record, eq, geom, target = sample_context(sample)
    truth = teacher(sample)
    if truth is None or truth.dtype != torch.float64:
        raise ValueError("Physical endpoint evaluation requires retained FP64 teacher")
    if prediction.shape != truth.shape:
        raise ValueError("Prediction and teacher geometries differ")
    result = physical_metrics(prediction.double()-truth.to(prediction.device), geom, target,
                              maximum_oversampling=int(record.get("maximum_oversampling",2)))
    _, ref = reference_status(sample)
    ur = ref.get("uncertainty_rms", sample.get("uncertainty_rms"))
    um = ref.get("uncertainty_maximum", sample.get("uncertainty_maximum"))
    for name, value in [("rms", ur), ("maximum", um)]:
        if value is not None and (not math.isfinite(float(value)) or float(value) < 0):
            raise ValueError("Reference uncertainty must be finite and nonnegative")
        result["uncertainty_"+name] = None if value is None else float(value)
        result["conservative_"+name] = (result[name]+float(value)
                                         if value is not None and result[name] is not None else None)
    if result["finite"]:
        pm, tv = n.mean(prediction.double()), n.mean(truth.to(prediction.device))
        pv = n.inner(prediction.double()-pm, prediction.double()-pm, geom, target)
        vv = n.inner(truth.to(prediction.device)-tv, truth.to(prediction.device)-tv, geom, target)
        result.update(variance_error=float((pv-vv).mean()), prediction_mean=float(pm.mean()),
                      prediction_variance=float(pv.mean()), teacher_mean=float(tv.mean()), teacher_variance=float(vv.mean()))
    return result


def final_rounding_metrics(prediction, parts, geom, target):
    """Attribute the deployed-field difference from its FP64 physical assembly."""
    value = parts.get("final_rounding")
    physical = parts.get("physical_prediction")
    if value is None and torch.is_tensor(physical):
        value = prediction.double()-physical
    return dict(final_rounding_rms=physical_metrics(value,geom,target)["rms"] if torch.is_tensor(value) else None,
        final_rounding_available=torch.is_tensor(value),
        final_rounding_scope="deployed output minus FP64 physical assembly; ordinary output also includes anchor/assembly differences")


def _identity(record):
    return {k: record.get(k) for k in ("parent_id", "query_id", "split", "target_kind", "grid", "lengths", "horizon",
                                      "regime", "shape_sha256", "active_frequency", "design_shift", "resolution_scope")}


def assert_parent_disjoint(samples):
    """Reject split leakage, including phase twins sharing a declared shape hash."""
    parents, shapes = defaultdict(set), defaultdict(set)
    for sample in samples:
        r = sample["record"]
        if not r.get("parent_id") or not r.get("split"):
            raise ValueError("Evaluation requires parent and split identity")
        parents[r["parent_id"]].add(r["split"])
        shape = r.get("shape_sha256", r.get("translation_invariant_shape_hash"))
        if shape:
            shapes[shape].add(r["split"])
    if any(len(v)>1 for v in (*parents.values(), *shapes.values())):
        raise ValueError("Parent/shape leakage across training, validation or diagnostic splits")


def parent_summary(rows, requested=None):
    """Equal parent weighting, equal within-parent queries; failures stay counted."""
    grouped = defaultdict(list)
    for row in rows:
        if row.get("status") == "MEASURED" and row.get("finite", False):
            grouped[row["parent_id"]].append(row)
    requested = rows if requested is None else requested
    result = dict(requested_queries=len(requested), measured_queries=sum(len(v) for v in grouped.values()),
                  requested_parents=len({r["parent_id"] for r in requested}), measured_parents=len(grouped),
                  weighting="equal parent; equal query within parent", failures=len(requested)-sum(len(v) for v in grouped.values()),
                  population_scope="declared development parents; paired descendants are not independent replicates")
    for key in ("mse", "rms", "mean_mse", "centered_mse", "centered_rms"):
        result["parent_mean_"+key] = (sum(sum(r[key] for r in v)/len(v) for v in grouped.values())/len(grouped)
                                       if grouped else None)
    maxima = [r["maximum"] for v in grouped.values() for r in v]
    result["worst_sampled_maximum"] = max(maxima) if maxima else None
    return result


def evaluate_endpoints(model, samples, *, method="model"):
    samples = list(samples)
    assert_parent_disjoint(samples)
    rows = []
    for sample in samples:
        record = sample["record"]
        row = dict(_identity(record), method=method, status="UNRESOLVED_REFERENCE")
        if record["target_kind"] not in TARGETS:
            row.update(status="UNSUPPORTED_TARGET", reason="No same-equation projected-fine closure contract")
        elif not reference_status(sample)[0]:
            row["reason"] = "Requested teacher unresolved; retained in denominator"
        else:
            try:
                _, eq, geom, target = sample_context(sample)
                # This is the complete inference interface. Labels and future
                # anchor/action arrays are deliberately absent.
                prediction = model(sample["u0"], float(record["horizon"]), eq, geom, target)
                row.update(endpoint_metrics(prediction, sample), output_dtype=str(prediction.dtype),
                           inference_scope="complete model call; no supplied path/residual/teacher")
                row.update(final_rounding_metrics(prediction,getattr(model,"last_parts",{}),geom,target))
                row["status"] = "MEASURED" if row["finite"] else "FAILED_NONFINITE"
            except (RuntimeError, ValueError, FloatingPointError, OverflowError) as exc:
                row.update(status="FAILED", failure_type=type(exc).__name__, reason=str(exc))
        rows.append(row)
    summaries = {}
    for split, target in sorted({(r["split"], r["target_kind"]) for r in rows}):
        subset = [r for r in rows if (r["split"], r["target_kind"]) == (split, target)]
        summaries[f"{split}/{target}"] = parent_summary(subset)
    return dict(rows=rows, summaries=summaries)


def _teacher_at(sample, at, providers):
    if abs(at-float(sample["record"]["horizon"])) < 1e-13:
        return sample
    if callable(providers):
        # A teacher always evolves the ORIGINAL actual input, never the
        # predicted state used by the student macrostep.
        value = providers(sample, at)
    else:
        value = None
        for key, candidate in (providers or {}).items():
            if abs(float(key)-at) < 1e-13:
                value = candidate
                break
    if value is None:
        return None
    record = dict(sample["record"], horizon=float(at))
    return {**sample, **value, "record":record}


def _balance(previous, current, h, eq, geom, target):
    p, c = moment_rhs(previous.double(), eq, geom, target), moment_rhs(current.double(), eq, geom, target)
    return dict(mean_balance_residual=float(((c["mean"]-p["mean"])/h-(p["mean_derivative"]+c["mean_derivative"])/2).abs().mean()),
                variance_balance_residual=float(((c["variance"]-p["variance"])/h-(p["variance_derivative"]+c["variance_derivative"])/2).abs().mean()),
                diffusion_variance_rate=float((2*n.inner(n.center(current.double()), n.generator(n.center(current.double()),eq,geom,target),geom,target)).mean()),
                balance_scope="trapezoidal endpoint moment residual; temporal discretization diagnostic, not enforced identity")


def evaluate_rollout(model, sample, steps, intermediate_teachers=None, *, method="model", bounded_fd=False):
    if steps not in (2, 4, 8):
        raise ValueError("Initial rollout panel declares 2, 4 or 8 macrosteps")
    record, eq, geom, target = sample_context(sample)
    horizon = float(record["horizon"])
    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("Rollout final time must be positive and finite")
    state = sample["u0"]
    input_dtype = state.dtype
    interval_applicable = (target == "fd_same_grid" and bounded_fd and bool(((state>=0)&(state<=1)).all()))
    rows, acquisition_seconds = [], 0.
    for step in range(1, steps+1):
        row = dict(_identity(record), method=method, macrosteps=steps, step=step,
                   physical_time=horizon*step/steps, status="UNRESOLVED_REFERENCE")
        before = state
        try:
            prediction = model(before, horizon/steps, eq, geom, target)
            finite = bool(torch.isfinite(prediction).all())
            row.update(finite=finite, reencoded_current_state=True, input_dtype=str(before.dtype), output_dtype=str(prediction.dtype))
            if not finite:
                row.update(status="FAILED_NONFINITE", reason="Autonomous rollout stopped; no state clipping")
                rows.append(row)
                break
            row.update(_balance(before, prediction, horizon/steps, eq, geom, target))
            row.update(interval_diagnostic_applicable=interval_applicable,
                       interval_violation=(float(torch.maximum((-prediction).clamp_min(0),(prediction-1).clamp_min(0)).max())
                                           if interval_applicable else None),
                       interval_scope="bounded FD assumptions" if interval_applicable else "NA; no Galerkin nodal maximum principle or unverified input bounds")
            start = time.perf_counter()
            reference = _teacher_at(sample, row["physical_time"], intermediate_teachers)
            acquisition_seconds += time.perf_counter()-start
            if reference is not None and reference_status(reference)[0]:
                row.update(endpoint_metrics(prediction, reference), status="MEASURED")
            else:
                row["reason"] = "Independent original-input teacher missing at this intermediate time"
            state = prediction.to(input_dtype)
            row["restart_rounding_rms"] = physical_metrics(state.double()-prediction.double(),geom,target)["rms"]
        except (RuntimeError, ValueError, FloatingPointError, OverflowError, TimeoutError, InterruptedError) as exc:
            row.update(status="TIME_CENSORED" if isinstance(exc,(TimeoutError,InterruptedError)) else "FAILED", failure_type=type(exc).__name__, reason=str(exc))
            rows.append(row)
            break
        rows.append(row)
    measured_steps = {r["step"] for r in rows if r["status"] == "MEASURED"}
    return dict(rows=rows, requested_steps=steps, measured_steps=len(measured_steps),
                missing_steps=[i for i in range(1,steps+1) if i not in measured_steps],
                teacher_acquisition_seconds=acquisition_seconds,
                teacher_scope="independent evolution of original actual FP32 input at every cumulative time",
                restart_precision=str(input_dtype), inference_scope="one complete re-encoded call per actual macrostep")


def nonlinear_endpoint_composition(model, u0, h, eq, geom, target, *, fraction=.5):
    if not (math.isfinite(float(h)) and h>=0 and 0<fraction<1):
        raise ValueError("Composition requires nonnegative horizon and an interior split")
    direct = model(u0,h,eq,geom,target)
    intermediate = model(u0,h*fraction,eq,geom,target)
    composed = model(intermediate.to(u0.dtype),h*(1-fraction),eq,geom,target)
    return dict(physical_metrics(composed.double()-direct.double(),geom,target),
                kind="nonlinear_endpoint_composition", reencoded_current_state=True,
                teacher_used=False, split_fraction=fraction,
                interpretation="composition defect; distinct from endpoint accuracy and stability")


def same_path_tangent_composition(action: Callable, probe, q, s, h, geom, target, *, path_identity):
    if not path_identity or not all(math.isfinite(float(t)) for t in (q,s,h)) or not 0<=q<=s<=h:
        raise ValueError("Tangent composition requires one declared path and ordered times")
    first = action(probe,q,s)
    composed = action(first,s,h)
    direct = action(probe,q,h)
    return dict(physical_metrics(composed.double()-direct.double(),geom,target),
                kind="same_path_tangent_composition", path_identity=str(path_identity),
                q=float(q), s=float(s), h=float(h), reencoding=False,
                interpretation="one fixed anchor path; not nonlinear endpoint composition")


def sensitivity_panel(function, state, direction, geom, target, *, epsilons=(1e-4,1e-5)):
    if state.shape != direction.shape or not bool(torch.isfinite(direction).all()):
        raise ValueError("Sensitivity probe shape/values invalid")
    with torch.enable_grad():
        u = state.detach().clone().requires_grad_(True)
        _, jvp = torch.autograd.functional.jvp(function,u,direction,create_graph=False,strict=False)
    rows = []
    for epsilon in epsilons:
        if not math.isfinite(float(epsilon)) or epsilon <= 0:
            raise ValueError("Finite-difference epsilon must be positive and finite")
        difference = (function(state+epsilon*direction)-function(state-epsilon*direction))/(2*epsilon)
        row = physical_metrics(jvp.double()-difference.double(),geom,target)
        scale = physical_metrics(jvp,geom,target)["rms"]
        row.update(epsilon=float(epsilon), jvp_rms=scale, relative_rms=(row["rms"]/scale if scale else None),
                   scope="central finite-difference agreement; epsilon/roundoff sensitivity, not a global stability proof")
        rows.append(row)
    return dict(rows=rows, independent_probe=True)


def finite_product_constant(geom,target):
    """Conservative normalized-L2 bilinear constant; explicitly grid dependent."""
    if target not in TARGETS:
        raise ValueError("Finite product bound requires the declared same-grid equation")
    count=math.prod(geom.grid) if target=="fd_same_grid" else math.prod(size+1 if size%2==0 else size for size in geom.grid)
    return math.sqrt(count)


def conditional_response_bound(*,action_error_integral,nonlinear_squared_integral,reaction,growth_bound,geom,target,
                               path_bound=0.,residual_bound=0.,quadrature_bound=0.,arithmetic_bound=0.):
    """Conditional integral inequality, never promote sampled estimates to bounds.

    action_error_integral bounds integral ||(U-U_tilde)rho||. The nonlinear
    squared integral bounds integral ||e(s)||²; growth_bound bounds ||U(t,s)||
    over that interval. Remaining terms must already be in endpoint norm units.
    """
    values=(action_error_integral,nonlinear_squared_integral,reaction,growth_bound,path_bound,residual_bound,quadrature_bound,arithmetic_bound)
    if any(not math.isfinite(float(v)) or v<0 for v in values):
        raise ValueError("Conditional bounds require finite nonnegative inputs and explicit assumptions")
    constant=finite_product_constant(geom,target)
    nonlinear=reaction*constant*growth_bound*nonlinear_squared_integral
    return dict(norm_bound=action_error_integral+nonlinear+path_bound+residual_bound+quadrature_bound+arithmetic_bound,
                response_approximation=action_error_integral,nonlinear_feedback=nonlinear,finite_product_constant=constant,
                path=path_bound,residual=residual_bound,quadrature=quadrature_bound,arithmetic=arithmetic_bound,
                grid=list(geom.grid),target_kind=target,certificate=False,
                scope="conditional inequality; caller must control continuous-time state/response and arithmetic errors; sampled refinements are estimates")


def signed_error_decomposition(components,geom,target):
    if not components:
        raise ValueError("Squared-error decomposition requires declared component vectors")
    names=list(components)
    fields=[components[name].double() for name in names]
    if any(v.shape!=fields[0].shape or not bool(torch.isfinite(v).all()) for v in fields):
        raise ValueError("Decomposition vectors must share finite geometry")
    diagonal={name:float(n.inner(field,field,geom,target).mean()) for name,field in zip(names,fields)}
    cross=[]
    for i,name in enumerate(names):
        for j in range(i+1,len(names)):
            cross.append(dict(left=name,right=names[j],signed_twice_inner=float(2*n.inner(fields[i],fields[j],geom,target).mean())))
    total=sum(fields[1:],fields[0])
    exact=float(n.inner(total,total,geom,target).mean())
    reconstructed=sum(diagonal.values())+sum(c["signed_twice_inner"] for c in cross)
    return dict(component_mse=diagonal,signed_cross_terms=cross,total_mse=exact,reconstructed_mse=reconstructed,
                algebraic_difference=exact-reconstructed,conservative_norm_sum=sum(math.sqrt(v) for v in diagonal.values()),
                scope="signed vector identity; component cancellation is retained, not an error certificate")


def run(samples, methods: Mapping[str,Callable], config=None):
    """JSON-ready endpoint and explicitly bounded auxiliary evaluation panels."""
    config = dict(config or {})
    samples = list(samples)
    assert_parent_disjoint(samples)
    outputs = {name:evaluate_endpoints(model,samples,method=name) for name,model in methods.items()}
    rollouts, compositions = [], []
    selected = [s for s in samples if s["record"].get("split") == "diagnostic"][:int(config.get("auxiliary_parents",0))]
    for name, model in methods.items():
        for sample in selected:
            if sample["record"]["target_kind"] not in TARGETS:
                continue
            for steps in config.get("rollout_steps",(2,4,8)):
                rollouts.append(dict(method=name, parent_id=sample["record"]["parent_id"],
                                     **evaluate_rollout(model,sample,int(steps),config.get("teacher_provider"),method=name,
                                                        bounded_fd=bool(config.get("bounded_fd",False)))))
            _,eq,geom,target = sample_context(sample)
            compositions.append(dict(method=name, parent_id=sample["record"]["parent_id"],
                **nonlinear_endpoint_composition(model,sample["u0"],float(sample["record"]["horizon"]),eq,geom,target)))
    return dict(schema="tdn.tangent-response-evaluation/v1", endpoints=outputs, rollouts=rollouts,
                endpoint_composition=compositions, requested_queries=len(samples),
                shift_axes=list(SHIFT_AXES), evidence_scope="DEVELOPMENT_ONLY_NO_CONFIRMATORY_CLAIMS",
                auxiliary_selection="prospectively configured first diagnostic parents; never selected by error or speed",
                resolution_contract="record fixed_physics separately from grid_matched_stiffness; do not pool as convergence")
