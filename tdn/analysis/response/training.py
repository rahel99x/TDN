"""Response trainability, honest selection, and exactly resumable optimization.

The supervised objective is evaluated after promotion of the deployed output to
FP64.  No target, reference, or anchor label is passed to the inference model.
Each optimizer commit saves model/optimizer/RNG/sampler state.  A scheduled
continuation is not a new trial or a failed scientific result.
"""
from __future__ import annotations

import copy
from collections import defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from tdn.research.experiment import atomic_torch_save
from tdn.research.protocol import digest, file_digest
from tdn.runtime.metadata import write_json, software_metadata
from tdn.runtime.precision import reference_precision
from tdn.analysis.frontier.core import check, clean
from .training_control import (control_config, learning_rate, full_parent_plan, is_head,
                               reporting_control, whole_history_summary, sampled_curve_rows)

SCHEMA = "tdn.response-training/v1"
CORE_REGIMES = ("low_mean", "high_mean", "large_amplitude", "strong_diffusion", "strong_reaction", "phase_twin")
TERMINAL = ("COMPLETED_UPDATES", "TIME_CENSORED", "FAILED_NONFINITE", "UNRESOLVED_SCREEN")
TINY_PHASES = ("tiny_individual", "tiny_joint", "diagnostic_individual", "diagnostic_joint")


def factorial_settings():
    """All settings have equal opportunity; no performance-dependent pruning."""
    return [dict(id=f"lr{lr:g}-clip{clip}-scale{scale:g}", lr=lr,
                 clip=clip, loss_multiplier=scale, weight_decay=0., eps=1e-8,
                 betas=[.9, .999])
            for lr, clip, scale in itertools.product((1e-3, 3e-4), (None, 1.), (1., .01))]


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _record(sample):
    return sample["record"]


def _parent(sample):
    return str(_record(sample)["parent_id"])


def _accepted(sample):
    record = _record(sample)
    return bool(sample.get("reference_accepted", record.get("reference", {}).get("accepted", False)))


def _reference(sample):
    return _record(sample).get("reference", {})


def _assert_development(samples):
    forbidden = [s for s in samples if _record(s).get("split") not in ("train", "validation")]
    if forbidden:
        raise ValueError("Training and selection must not access test/confirmation labels")
    by_parent = defaultdict(set)
    for sample in samples:
        by_parent[_parent(sample)].add(_record(sample)["split"])
    if any(len(splits) != 1 for splits in by_parent.values()):
        raise ValueError("A parent occurs in more than one training/validation split")


def _parent_mean(values):
    grouped = defaultdict(list)
    for parent, value in values:
        grouped[parent].append(float(value))
    return float(np.mean([np.mean(v) for v in grouped.values()])) if grouped else None


def _weighted_quantile(values, probability):
    """Each parent has total weight one, irrespective of its query count."""
    grouped = defaultdict(list)
    for parent, value in values:
        grouped[parent].append(float(value))
    pairs = sorted((value, 1./len(rows)) for rows in grouped.values() for value in rows)
    if not pairs:
        raise ValueError("No training values for normalization")
    threshold = probability*len(grouped)
    cumulative = 0.
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= threshold:
            return float(value)
    return float(pairs[-1][0])


def _physical_mse(value, record=None):
    if record is None or record.get("target_kind", "fd_same_grid") == "fd_same_grid":
        return value.square().mean()
    from .numerics import inner
    from .references import equation_geometry
    _, geom = equation_geometry(record)
    return inner(value, value, geom, record["target_kind"]).mean()


def _physical_coordinates(value, record):
    """Isometric coordinates for declared physical L2, including Nyquist mass."""
    dimension = len(record["grid"])
    axes = tuple(range(value.ndim-dimension, value.ndim))
    if record["target_kind"] == "fd_same_grid":
        return value.flatten(-dimension)/math.sqrt(math.prod(record["grid"]))
    from .numerics import mass_weights
    from .references import equation_geometry
    _, geom = equation_geometry(record)
    spectrum = torch.fft.fftn(value, dim=axes, norm="forward")*mass_weights(value, geom).sqrt()
    return torch.cat([spectrum.real.flatten(-dimension), spectrum.imag.flatten(-dimension)], dim=-1)


def training_normalization(samples, *, minimum_floor=1e-8):
    """Robust frozen scales, using only accepted *training* FP64 labels.

    The floor is the largest training reference estimate or observed anchor
    rounding, and half an FP32 epsilon at the largest training state scale.
    It is a normalization safeguard, not a rigorous bound on model error.
    """
    _assert_development(samples)
    rows = [s for s in samples if _record(s)["split"] == "train" and _accepted(s)]
    if not rows:
        raise ValueError("No accepted training references")
    means, centered, full, within_field_scales = [], [], [], []
    floor = float(minimum_floor)
    for sample in rows:
        target = sample["target"].double()
        if sample["target"].dtype != torch.float64:
            raise ValueError("Response training targets must retain FP64 labels")
        if not bool(torch.isfinite(target).all()):
            raise ValueError("Accepted training reference contains nonfinite values")
        deployed = sample["anchor_fp32"].double()
        residual = target-deployed
        parent = _parent(sample)
        mean = residual.mean()
        means.append((parent, abs(float(mean))))
        centered.append((parent, float(_physical_mse(residual-mean, _record(sample)).sqrt())))
        full.append((parent, float(target.mean())))
        within_field_scales.append((parent, float(_physical_mse(target-target.mean(), _record(sample)).sqrt())))
        reference = _reference(sample)
        uncertainty = max(float(reference.get("uncertainty_rms", 0.) or 0.),
                          float(reference.get("uncertainty_maximum", 0.) or 0.))
        rounding = float((deployed-sample["anchor_fp64"].double()).abs().max())
        output_resolution = .5*torch.finfo(torch.float32).eps*max(1., float(target.abs().max()))
        floor = max(floor, uncertainty, rounding, output_resolution)
    full_mean = _parent_mean(full)
    full_scales = [(_parent(sample), float(_physical_mse(sample["target"].double()-full_mean, _record(sample)).sqrt()))
                   for sample in rows]
    return dict(mean_scale=max(floor, _weighted_quantile(means, .75)),
                centered_scale=max(floor, _weighted_quantile(centered, .75)),
                full_mean=full_mean, full_scale=max(floor, _weighted_quantile(full_scales, .75)),
                within_field_centered_scale=max(floor, _weighted_quantile(within_field_scales, .75)),
                full_scale_estimator="parent-weighted 75th percentile of physical RMS(target - global training mean), including between-parent means",
                floor=floor, estimator="parent-weighted training 75th percentile; full mean is parent mean",
                floor_policy="max(reference uncertainty, observed FP32-anchor error, half eps at state scale, fixed floor)",
                label_contract="FP64 teacher for actual input minus actual FP32 anchor promoted to FP64",
                train_parent_ids=sorted({_parent(s) for s in rows}),
                accepted_training_queries=len(rows), rejected_training_queries=sum(not _accepted(s) for s in samples
                    if _record(s)["split"] == "train"))


def apply_normalization_mode(normalization, mode="separate"):
    """An explicit scaling ablation, never described as covariance whitening."""
    if mode not in ("separate", "common"):
        raise ValueError("Normalization mode must be separate or common")
    result = copy.deepcopy(normalization)
    result["normalization_mode"] = mode
    result["unmodified_channel_scales"] = {key:normalization[key] for key in ("mean_scale", "centered_scale")}
    if mode == "common":
        result["mean_scale"] = result["centered_scale"] = max(normalization["mean_scale"], normalization["centered_scale"])
    result["transform_kind"] = "frozen diagonal channel rescaling; no covariance whitening"
    return result


def configure_objective(normalization, family, mode="family"):
    """Use an appropriate direct full-target objective; retain a shared ablation.

    Output-feature scales and auxiliary source scales remain independently
    recorded.  A residual-sized denominator for a direct order-one prediction
    is an explicit ablation, never the default competitor training recipe.
    """
    result = copy.deepcopy(normalization)
    aliases = {"family":"full_target" if family == "direct_fno" else "residual_channels",
               "shared_residual":"shared_physical"}
    selected = aliases.get(mode, mode)
    if selected not in ("full_target", "residual_channels", "shared_physical", "physical"):
        raise ValueError("Unknown Response training objective normalization")
    if selected == "physical":
        mean_scale = centered_scale = 1.
    elif selected == "full_target":
        mean_scale = centered_scale = result["full_scale"]
    else:
        mean_scale, centered_scale = result["mean_scale"], result["centered_scale"]
    result.update(objective_mode=selected, objective_mean_scale=mean_scale,
                  objective_centered_scale=centered_scale,
                  objective_contract="FP64 physical endpoint error; family-appropriate frozen training scales; unnormalized errors reported separately")
    return result


def _objective_scales(normalization):
    return (normalization.get("objective_mean_scale", normalization["mean_scale"]),
            normalization.get("objective_centered_scale", normalization["centered_scale"]))


def predict(model, sample):
    """Inference accepts a stripped sample containing only state and physics."""
    from .references import equation_geometry
    record = _record(sample)
    eq, geom = equation_geometry(record)
    return model(sample["u0"], float(record["horizon"]), eq, geom, record["target_kind"])


def physical_loss(prediction, target, normalization, record=None):
    error = prediction.double()-target.double()
    dims = tuple(range(2, error.ndim))
    mean = error.mean(dim=dims, keepdim=True)
    mean_scale, centered_scale = _objective_scales(normalization)
    return (.5*(mean/mean_scale).square().mean()
            + .5*_physical_mse((error-mean)/centered_scale, record))


def endpoint_row(prediction, sample, normalization):
    error = prediction.detach().double()-sample["target"].double()
    delta = sample["target"].double()-sample["anchor_fp32"].double()
    mean = float(error.mean())
    centered = float(_physical_mse(error-error.mean(), _record(sample)).sqrt())
    reference = _reference(sample)
    uncertainty = float(reference.get("uncertainty_rms", 0.) or 0.)
    maximum_uncertainty = float(reference.get("uncertainty_maximum", 0.) or 0.)
    record = _record(sample)
    rms = float(_physical_mse(error, record).sqrt())
    maximum = float(error.abs().max())
    defect = float(_physical_mse(delta, record).sqrt())
    physical_defect = sample["target"].double()-sample["anchor_fp64"].double()
    roundoff = sample["anchor_fp32"].double()-sample["anchor_fp64"].double()
    quantization = sample["target"].double()-sample["target"].float().double()
    # Physical Galerkin L2 is not the separable nodal norm: pointwise rounding
    # gives an attainable reference approximation there, not a physical bound.
    nearest_fp32_sampled_rms = float(quantization.square().mean().sqrt())
    return dict(parent_id=_parent(sample), query_id=record.get("query_id"), regime=record.get("regime", "unspecified"),
                tiny_regime=record.get("tiny_regime"),
                target_kind=record["target_kind"], horizon=record["horizon"], split=record["split"],
                rms=rms, maximum=maximum, mean_error=mean, centered_rms=centered,
                sampled_rms=float(error.square().mean().sqrt()), maximum_scope="sampled nodal maximum",
                norm_contract="nodal L2" if record["target_kind"] == "fd_same_grid" else "physical interpolant L2; Nyquist mass weighting",
                mse=rms*rms, objective=float(physical_loss(prediction, sample["target"], normalization, record)),
                anchor_rms=defect, anchor_maximum=float(delta.abs().max()), anchor_mse=defect*defect,
                candidate_over_anchor_rms=rms/defect if defect > 0 else None,
                physical_defect_rms=float(_physical_mse(physical_defect, record).sqrt()),
                deployment_roundoff_rms=float(_physical_mse(roundoff, record).sqrt()),
                deployment_roundoff_maximum=float(roundoff.abs().max()),
                nearest_fp32_sampled_rms=nearest_fp32_sampled_rms,
                fp32_resolution_scope="minimum nodal RMS for FP32 output; Galerkin physical norm is not a separable rounding bound",
                target_defect_rms=defect, floor_limited=defect <= normalization["floor"],
                floor=normalization["floor"], reference_accepted=_accepted(sample),
                uncertainty_rms=uncertainty, uncertainty_maximum=maximum_uncertainty,
                conservative_rms=rms+uncertainty, conservative_maximum=maximum+maximum_uncertainty,
                finite=bool(torch.isfinite(prediction).all()))


def evaluate_training(model, samples, normalization, budget=None, criteria=None):
    was_training = model.training
    model.eval()
    rows = []
    with torch.no_grad():
        for sample in samples:
            if budget is not None:
                budget.check()
            rows.append(endpoint_row(predict(model, sample), sample, normalization))
    model.train(was_training)
    limits = criteria or {}
    rms_limit, max_limit = float(limits.get("rms", 2e-5)), float(limits.get("maximum", 2e-5))
    if rms_limit <= 0 or max_limit <= 0:
        raise ValueError("Physical accuracy criteria must be positive")
    criterion = dict(rms_target=rms_limit, maximum_target=max_limit,
        all_queries_passed=bool(rows) and all(r["reference_accepted"] and r["conservative_rms"] <= rms_limit
                                           and r["conservative_maximum"] <= max_limit for r in rows),
        passed_queries=sum(r["reference_accepted"] and r["conservative_rms"] <= rms_limit
                           and r["conservative_maximum"] <= max_limit for r in rows),
        worst_tolerance_ratio=max((max(r["conservative_rms"]/rms_limit,
                                      r["conservative_maximum"]/max_limit) for r in rows), default=None))
    return dict(rows=rows, queries=len(rows), parents=len({_parent(s) for s in samples}),
                physical_criterion=criterion,
                objective=_parent_mean([(r["parent_id"], r["objective"]) for r in rows]),
                rms=_parent_mean([(r["parent_id"], r["rms"]) for r in rows]),
                maximum=max((r["maximum"] for r in rows), default=None),
                mse=_parent_mean([(r["parent_id"], r["mse"]) for r in rows]),
                mean_absolute=_parent_mean([(r["parent_id"], abs(r["mean_error"])) for r in rows]),
                centered_rms=_parent_mean([(r["parent_id"], r["centered_rms"]) for r in rows]))


def _unavailable_metrics(reason):
    return dict(rows=[], queries=0, parents=0, objective=None, rms=None, maximum=None,
                mse=None, mean_absolute=None, centered_rms=None, unavailable_reason=reason)


def _state_hash(state):
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        result.update(name.encode())
        result.update(str(value.dtype).encode())
        result.update(str(tuple(value.shape)).encode())
        result.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def _rng_state(sampler):
    np_state = np.random.get_state()
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                python=random.getstate(), sampler=sampler.getstate(),
                numpy=dict(name=np_state[0], keys=np_state[1].tolist(), position=np_state[2],
                           has_gauss=np_state[3], cached_gaussian=np_state[4]))


def _restore_rng(state, sampler):
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Exact continuation requires the same visible CUDA device count")
        torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    sampler.setstate(state["sampler"])
    n = state["numpy"]
    np.random.set_state((n["name"], np.asarray(n["keys"], dtype=np.uint32), n["position"], n["has_gauss"], n["cached_gaussian"]))


def _gradient_norm(model, *, head=None):
    values = []
    for name, parameter in model.named_parameters():
        parameter_is_head = is_head(name)
        if parameter.grad is not None and (head is None or head == parameter_is_head):
            values.append(parameter.grad.detach().abs().double().square().sum())
    return float(torch.stack(values).sum().sqrt()) if values else 0.


def _movement(model, initial, *, head=None):
    values = []
    for name, parameter in model.named_parameters():
        parameter_is_head = is_head(name)
        if head is None or head == parameter_is_head:
            values.append((parameter.detach()-initial[name].to(parameter.device)).abs().double().square().sum())
    return float(torch.stack(values).sum().sqrt()) if values else 0.


def _rollout_loss(model, sample, normalization, steps):
    from .references import equation_geometry
    record = _record(sample)
    if record["target_kind"] == "galerkin_projected_fine":
        raise ValueError("Autonomous coarse rollout training is a separate closure experiment")
    eq, geom = equation_geometry(record)
    u = sample["u0"]
    parameter_dtype = next(model.parameters(), sample["u0"]).dtype
    for index in range(steps):
        u = model(u, float(record["horizon"])/steps, eq, geom, record["target_kind"])
        if index+1 < steps:
            # FP64 assembly is an endpoint arithmetic control. Autonomous
            # re-encoding returns to network precision without detaching.
            u = u.to(dtype=parameter_dtype)
    return physical_loss(u, sample["target"], normalization, record)


def _source_loss(model, sample, normalization):
    from .references import equation_geometry
    if "source_times" not in sample or "source_targets" not in sample:
        raise ValueError("Source-supervised arm requires explicitly prepared, costed FP64 source labels")
    if sample["source_targets"].dtype != torch.float64:
        raise ValueError("Source targets must retain FP64 labels")
    contract = _record(sample).get("source_supervision", {})
    if not contract.get("accepted", False):
        raise ValueError("Source supervision has no accepted mathematical target contract")
    eq, geom = equation_geometry(_record(sample))
    encoded = model.encode(sample["u0"], eq, geom, _record(sample)["target_kind"])
    values = [model.source_terms(encoded, float(t)).double() for t in sample["source_times"]]
    output = torch.stack(values)
    target = sample["source_targets"]
    if output.shape != target.shape:
        raise ValueError("Source-label node/spatial shape differs from model source")
    error = (output-target)/(normalization["centered_scale"]/model.t_ref)
    return torch.stack([_physical_mse(node, _record(sample)) for node in error]).mean()


def _consistency_loss(model, sample, normalization):
    """A separate mean/variance balance penalty for the actual finite equation.

    This is not imposed on projected-fine closure. A small residual alone is
    never a correctness or accuracy certificate.
    """
    from .references import equation_geometry
    from . import numerics as n
    record = _record(sample)
    if record["target_kind"] == "galerkin_projected_fine":
        raise ValueError("Coarse mean-balance penalty cannot suppress projected-fine closure")
    eq, geom = equation_geometry(record)
    time_input = torch.tensor(float(record["horizon"]), dtype=sample["u0"].dtype,
                              device=sample["u0"].device, requires_grad=True)
    prediction = model(sample["u0"], time_input, eq, geom, record["target_kind"])
    mean = n.mean(prediction)
    derivative = torch.autograd.grad(mean.sum(), time_input, create_graph=True)[0]
    variance = n.mean(n.product(n.center(prediction), n.center(prediction), geom, record["target_kind"]))
    balance = eq.reaction_rate*(mean*(1-mean)-variance)
    return ((derivative-balance)/(normalization["mean_scale"]/getattr(model, "t_ref", .1))).square().mean()


def _loss_gradient_norm(loss, parameters):
    if not loss.requires_grad:
        return 0.
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return math.sqrt(sum(float(g.detach().abs().double().square().sum()) for g in gradients if g is not None))


def _jsonl(path, rows):
    target = Path(path)
    temporary = target.with_suffix(target.suffix+".partial")
    temporary.write_text("".join(json.dumps(clean(row), sort_keys=True, allow_nan=False)+"\n" for row in rows))
    temporary.replace(target)


def _emit_live(row, config, *, force=False, phase="training"):
    if not __import__("os").environ.get("TDN_RESPONSE_RAVEN_DIR"):
        return
    from .diagnostic_metrics import emit_live
    policy = reporting_control(config)
    if (policy["enabled"] and not force and row["update"] != 0
        and row["update"] % policy["live_every"] != 0 and row.get("validation_loss") is None):
        return
    names = ("train_loss", "train_rms", "train_maximum", "endpoint_loss", "full_train_mse", "full_train_rms",
             "full_train_maximum", "full_train_worst_tolerance_ratio", "validation_loss", "validation_rms",
             "validation_maximum", "validation_mean_absolute", "validation_centered_rms", "gradient_norm",
             "gradient_norm_after_clip", "head_gradient_norm", "trunk_gradient_norm", "learning_rate", "examples_seen",
             "parameter_movement", "trunk_movement", "optimization_seconds")
    emit_live({name:row.get(name) for name in names}, phase=phase, step=row["update"],
              progress=dict(completed=row["update"], total=int(config["updates"]), unit="logical updates"))


def fit_correction_gains(samples, normalization, config, *, minimum_gain=1e-8):
    """Freeze parent-weighted RMS complete-basis gains using training inputs only."""
    from .references import equation_geometry
    from .temporal_basis import correction_gains
    if not math.isfinite(minimum_gain) or minimum_gain <= 0:
        raise ValueError("Minimum correction gain must be positive and finite")
    _assert_development(samples)
    rows = [sample for sample in samples if _record(sample)["split"] == "train" and _accepted(sample)]
    if not rows:
        raise ValueError("Gain normalization requires accepted training inputs")
    result = copy.deepcopy(normalization)
    gains = {key: [] for key in ("mean_gain", "centered_gain")}
    with torch.no_grad():
        for sample in rows:
            record = _record(sample)
            eq, geom = equation_geometry(record)
            value = correction_gains(sample["u0"], record["horizon"], eq, geom, record["target_kind"],
                                    family=config["family"], config={**config.get("model", {}), "normalization":normalization})
            for key in gains:
                tensor = torch.as_tensor(value[key]).double()
                if not bool(torch.isfinite(tensor).all()):
                    raise ValueError("Nonfinite complete correction gain")
                gains[key].append((_parent(sample), float(tensor.square().mean())))
    result.update({key:max(float(minimum_gain), math.sqrt(_parent_mean(values))) for key, values in gains.items()})
    result["gain_normalization"] = dict(estimator="parent-weighted train RMS of complete input-only basis gain before channel scales",
        minimum_gain=float(minimum_gain), training_parent_ids=sorted({_parent(s) for s in rows}),
        accepted_training_queries=len(rows), fit_access="accepted training input/physics/horizons only; no validation or teacher values",
        coupled_source_gain_scope="nominal physical gate/time envelope; not the learned-source Jacobian")
    return result


def _correction_summary(model, samples, budget=None):
    """Budgeted selected-checkpoint instrumentation, retaining actual partial work."""
    if not hasattr(model, "correction_diagnostics"):
        return dict(status="NA", reason="Model does not expose correction diagnostics")
    from .references import equation_geometry
    family = getattr(model, "family", None)
    if family == "temporal":
        raw_scope = "Historical temporal decoded raw response includes its analytic time response; not a pre-basis coefficient"
    elif family in ("adaptive_temporal", "spatial_temporal"):
        raw_scope = "Raw endpoint dictionary coefficients before time basis and physical channel scaling"
    elif family == "coupled_source":
        raw_scope = "Encoded mean coefficients and centered source field-times-coefficients before time integration/physical source scaling"
    else:
        raw_scope = "Raw endpoint readouts before time and physical channel scaling"
    rows = []
    status, reason = "MEASURED", None
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for sample in samples:
                if budget is not None:
                    budget.check()
                record = _record(sample)
                eq, geom = equation_geometry(record)
                values = model.correction_diagnostics(sample["u0"], record["horizon"], eq, geom, record["target_kind"])
                row = dict(parent_id=_parent(sample), query_id=record.get("query_id"), horizon=record["horizon"],
                    raw_units_scope=raw_scope)
                for key, value in values.items():
                    if torch.is_tensor(value):
                        row[key] = dict(rms=float(value.detach().double().square().mean().sqrt()),
                                        maximum=float(value.detach().abs().max()), finite=bool(torch.isfinite(value).all()))
                    elif isinstance(value, (str, float, int, bool)):
                        row[key] = value
                correction, output = values.get("total_correction"), values.get("output")
                if torch.is_tensor(correction) and torch.is_tensor(output):
                    output32 = output.float()
                    infinity = torch.full_like(output32, float("inf"))
                    spacing = (torch.nextafter(output32, infinity)-output32).double().abs()
                    row["sub_ulp_correction_fraction"] = float((correction.double().abs() < spacing).double().mean())
                    row["fp32_output_spacing_rms"] = float(spacing.square().mean().sqrt())
                    row["resolution_scope"] = "Adjacent FP32 output-spacing proxy; sub-ULP fraction is not an erasure rate"
                rows.append(row)
    except (TimeoutError, InterruptedError) as exc:
        status, reason = "UNRESOLVED", "Bounded correction diagnostic incomplete: "+str(exc)
    finally:
        model.train(was_training)
    return dict(status=status, reason=reason, rows=rows, queries=len(rows), requested_queries=len(samples),
                raw_units_scope=raw_scope, cross_family_raw_comparison="Raw magnitudes are not comparable across families/scaling contracts",
                scope="selected checkpoint, training inputs only; raw heads/pre-addition correction/assembly resolution")


def _components(model, sample, normalization, config):
    endpoint = physical_loss(predict(model, sample), sample["target"], normalization, _record(sample))
    rollout = _rollout_loss(model, sample, normalization, int(config.get("rollout_steps", 2))) if config.get("rollout_weight", 0.) else endpoint.new_zeros(())
    source = _source_loss(model, sample, normalization) if config.get("source_aux_weight", 0.) else endpoint.new_zeros(())
    consistency = _consistency_loss(model, sample, normalization) if config.get("consistency_weight", 0.) else endpoint.new_zeros(())
    return endpoint, rollout, source, consistency


def train_trial(model, train, validation, config, normalization, path, *,
                identity, resume=None, budget=None, max_chunk_updates=None,
                chunk_seconds=None, final_chunk=True):
    """Train one declared trial, or continue it without resetting its RNG/order.

    Public for independent resume tests. ``max_chunk_updates`` is an execution
    partition, never part of the scientific trial identity or selection rule.
    Validation occurs only at declared update indices (plus initialization),
    so changing chunk boundaries cannot change the selected checkpoint.
    """
    # TorchVersion is a str subclass that weights_only deliberately rejects.
    # Persist only plain JSON metadata, never arbitrary configuration objects.
    identity = json.loads(json.dumps(identity, sort_keys=True, allow_nan=False))
    controls = control_config(config)
    reporting = reporting_control(config)
    accuracy_criteria = config.get("physical_criteria", {})
    normalization = configure_objective(normalization, config.get("family"), config.get("objective_mode", "family"))
    _assert_development(train+validation)
    if any(_record(s)["split"] != "train" for s in train) or any(_record(s)["split"] != "validation" for s in validation):
        raise ValueError("Incorrect fitting/selection split")
    if not train or not validation or not all(_accepted(s) for s in train+validation):
        raise ValueError("Training requires nonempty accepted train and validation references")
    root = Path(path); root.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters(), torch.empty((), device=train[0]["u0"].device)).device
    reference_precision()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    setting = config["setting"]
    seed = int(config.get("seed", 0))
    sampler = random.Random(seed)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("Trainable trial has no trainable parameters")
    if controls["head_warmup_updates"] and not any(is_head(name) and parameter.requires_grad for name, parameter in model.named_parameters()):
        raise ValueError("Head warmup requires an explicitly named trainable correction head")
    optimizer = torch.optim.AdamW(parameters, lr=float(setting["lr"]),
        betas=tuple(setting.get("betas", [.9, .999])), eps=float(setting.get("eps", 1e-8)),
        weight_decay=float(setting.get("weight_decay", 0.)))
    updates = int(config["updates"])
    validation_every = int(config.get("validation_every", max(1, updates//10)))
    validation_indices = set(range(validation_every, updates+1, validation_every)) | {updates}
    model.train()
    saved = torch.load(resume, weights_only=True, map_location="cpu") if resume is not None else None
    if saved is not None and saved["identity"] != identity:
        raise ValueError("Training continuation source/config/software/data identity changed")
    if saved is not None and saved.get("training_controls", control_config(config)) != controls:
        raise ValueError("Training continuation logical update controls changed")
    if saved is not None and saved.get("reporting_controls", reporting) != reporting:
        raise ValueError("Training continuation reporting cadence changed")
    if saved is None:
        initial_state = copy.deepcopy(model.state_dict())
        history, completed, examples, selected_update = [], 0, 0, 0
        selected_state = copy.deepcopy(initial_state)
        initial = best = initial_train = None
        attainable_train = None; attainable_update = 0
        criterion_train = None; criterion_update = 0; criterion_state = copy.deepcopy(initial_state)
        cumulative = dict(total_seconds=0., optimization_seconds=0., validation_seconds=0.,
                          checkpoint_seconds=0., diagnostic_seconds=0.)
        work_counters = dict(optimization_query_attempts=0, backward_calls=0, optimizer_step_attempts=0, committed_optimizer_updates=0)
        pending_validation = False
    else:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        initial_state, selected_state = saved["initial_state"], saved["selected_state"]
        history, completed, examples = saved["history"], saved["update"], saved["examples"]
        selected_update = saved["selected_update"]
        initial, best, initial_train = saved["initial"], saved["best"], saved["initial_train"]
        attainable_train = saved.get("attainable_train", initial_train)
        attainable_update = saved.get("attainable_update", 0)
        criterion_train = saved.get("criterion_train", initial_train)
        criterion_update = saved.get("criterion_update", 0)
        criterion_state = saved.get("criterion_state", initial_state)
        cumulative = saved["cost"]
        work_counters = saved.get("work_counters", dict(optimization_query_attempts=examples,
            backward_calls=completed, optimizer_step_attempts=completed, committed_optimizer_updates=completed))
        pending_validation = saved["pending_validation"]
        _restore_rng(saved["rng"], sampler)
    start = time.perf_counter(); previous_seconds = cumulative["total_seconds"]
    chunk_start_update = completed
    prior_attempts = saved.get("attempts", []) if saved else []
    attempt = dict(index=len(prior_attempts), started_unix=time.time(), start_update=completed,
                   resume_from=str(resume) if resume else None, status="RUNNING")
    attempts = prior_attempts+[attempt]
    write_json(root/"attempt.json", attempt)
    parent_samples = defaultdict(list)
    for index, sample in enumerate(train):
        parent_samples[_parent(sample)].append(index)
    parent_ids = sorted(parent_samples)

    def elapsed():
        return previous_seconds+time.perf_counter()-start

    def checkpoint():
        tick = time.perf_counter()
        cumulative["total_seconds"] = elapsed()
        atomic_torch_save(dict(schema=SCHEMA, identity=identity, model=model.state_dict(),
            optimizer=optimizer.state_dict(), initial_state=initial_state, selected_state=selected_state,
            normalization=normalization, update=completed, examples=examples, selected_update=selected_update,
            initial=initial, best=best, initial_train=initial_train, history=history,
            attainable_train=attainable_train, attainable_update=attainable_update,
            criterion_train=criterion_train, criterion_update=criterion_update, criterion_state=criterion_state,
            work_counters=work_counters,
            training_controls=controls, reporting_controls=reporting, scheduler_state=dict(completed_updates=completed,
                next_learning_rate=learning_rate(controls, completed)),
            pending_validation=pending_validation, rng=_rng_state(sampler), cost=cumulative,
            attempts=attempts), root/"journal.pt")
        cumulative["checkpoint_seconds"] += time.perf_counter()-tick

    def validate():
        nonlocal best, selected_state, selected_update, pending_validation, attainable_train, attainable_update, criterion_train, criterion_update, criterion_state
        tick = time.perf_counter(); _sync(device)
        value = evaluate_training(model, validation, normalization, budget, accuracy_criteria)
        _sync(device); cumulative["validation_seconds"] += time.perf_counter()-tick
        if not math.isfinite(value["objective"]):
            raise FloatingPointError("Nonfinite validation objective")
        # Exact ties retain the earlier checkpoint; neither test truth nor
        # chunk boundaries affect model selection.
        if best is None or value["objective"] < best["objective"]:
            best, selected_state, selected_update = value, copy.deepcopy(model.state_dict()), completed
        if completed > 0 and config.get("phase") in TINY_PHASES:
            tick = time.perf_counter()
            fit = evaluate_training(model, train, normalization, budget, accuracy_criteria)
            cumulative["diagnostic_seconds"] += time.perf_counter()-tick
            if attainable_train is None or fit["mse"] < attainable_train["mse"]:
                attainable_train, attainable_update = fit, completed
            if criterion_train is None or fit["physical_criterion"]["worst_tolerance_ratio"] < criterion_train["physical_criterion"]["worst_tolerance_ratio"]:
                criterion_train, criterion_update = fit, completed
                criterion_state = copy.deepcopy(model.state_dict())
            history[-1].update(full_train_mse=fit["mse"], full_train_loss=fit["objective"],
                               full_train_rms=fit["rms"], full_train_maximum=fit["maximum"])
            history[-1]["full_train_worst_tolerance_ratio"] = fit["physical_criterion"]["worst_tolerance_ratio"]
        history[-1].update(validation_loss=value["objective"], validation_rms=value["rms"],
                           validation_maximum=value["maximum"], validation_mean_absolute=value["mean_absolute"],
                           validation_centered_rms=value["centered_rms"])
        _emit_live(history[-1], config)
        pending_validation = False
        checkpoint()
        return value

    status = "CONTINUATION_REQUIRED"; failure = None; optimization_tick = None
    try:
        if initial is None:
            tick = time.perf_counter()
            initial_train = evaluate_training(model, train, normalization, budget, accuracy_criteria)
            attainable_train = initial_train
            criterion_train = initial_train
            cumulative["diagnostic_seconds"] += time.perf_counter()-tick
            history.append(dict(update=0, train_loss=initial_train["objective"],
                train_rms=initial_train["rms"], train_maximum=initial_train["maximum"], elapsed_seconds=elapsed(),
                gradient_norm=None, examples_seen=0, initialization=True))
            initial = validate()
            checkpoint()
        elif pending_validation:
            validate()
        while completed < updates:
            if budget is not None:
                budget.check()
            if elapsed() >= float(config.get("total_seconds", float("inf"))):
                status = "TIME_CENSORED"; break
            if chunk_seconds is not None and time.perf_counter()-start >= chunk_seconds:
                break
            if max_chunk_updates is not None and completed-chunk_start_update >= max_chunk_updates:
                break
            plan = full_parent_plan(parent_samples) if controls["batch_mode"] == "full_parent" else [
                (sampler.choice(parent_samples[sampler.choice(parent_ids)]), 1./int(config.get("batch_size", 1)))
                for _ in range(int(config.get("batch_size", 1)))]
            indices = [index for index, _ in plan]
            _sync(device); tick = time.perf_counter()
            optimization_tick = tick
            current_lr = learning_rate(controls, completed)
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            optimizer.zero_grad(set_to_none=True)
            endpoint_losses, rollout_losses, source_losses, consistency_losses = [], [], [], []
            full_values = [0., 0., 0., 0.]
            for index, weight in plan:
                sample = train[index]
                work_counters["optimization_query_attempts"] += 1
                if controls["batch_mode"] == "full_parent":
                    components = _components(model, sample, normalization, config)
                    sample_loss = components[0]+float(config.get("rollout_weight", 0.))*components[1]+float(config.get("source_aux_weight", 0.))*components[2]+float(config.get("consistency_weight", 0.))*components[3]
                    scaled_sample = sample_loss*(weight*float(setting.get("loss_multiplier", 1.)))
                    if not torch.isfinite(scaled_sample):
                        raise FloatingPointError("Nonfinite training loss")
                    scaled_sample.backward()
                    work_counters["backward_calls"] += 1
                    for position, value in enumerate(components):
                        full_values[position] += weight*float(value.detach())
                    continue
                endpoint_losses.append(physical_loss(predict(model, sample), sample["target"], normalization, _record(sample)))
                if float(config.get("rollout_weight", 0.)):
                    rollout_losses.append(_rollout_loss(model, sample, normalization, int(config.get("rollout_steps", 2))))
                if float(config.get("source_aux_weight", 0.)):
                    source_losses.append(_source_loss(model, sample, normalization))
                if float(config.get("consistency_weight", 0.)):
                    consistency_losses.append(_consistency_loss(model, sample, normalization))
            if controls["batch_mode"] == "full_parent":
                endpoint, auxiliary, source, consistency = [torch.tensor(value, dtype=torch.float64, device=device) for value in full_values]
            else:
                endpoint = torch.stack(endpoint_losses).mean()
                auxiliary = torch.stack(rollout_losses).mean() if rollout_losses else endpoint.new_zeros(())
                source = torch.stack(source_losses).mean() if source_losses else endpoint.new_zeros(())
                consistency = torch.stack(consistency_losses).mean() if consistency_losses else endpoint.new_zeros(())
            loss = (endpoint+float(config.get("rollout_weight", 0.))*auxiliary
                    +float(config.get("source_aux_weight", 0.))*source
                    +float(config.get("consistency_weight", 0.))*consistency)
            scaled = loss*float(setting.get("loss_multiplier", 1.))
            if not torch.isfinite(scaled):
                raise FloatingPointError("Nonfinite training loss")
            endpoint_gradient = _loss_gradient_norm(endpoint, parameters) if source_losses or rollout_losses or consistency_losses else None
            auxiliary_gradient = _loss_gradient_norm(auxiliary, parameters) if rollout_losses else None
            source_gradient = _loss_gradient_norm(source, parameters) if source_losses else None
            consistency_gradient = _loss_gradient_norm(consistency, parameters) if consistency_losses else None
            if controls["batch_mode"] != "full_parent":
                scaled.backward()
                work_counters["backward_calls"] += 1
            pre = _gradient_norm(model)
            head, trunk = _gradient_norm(model, head=True), _gradient_norm(model, head=False)
            if not math.isfinite(pre):
                raise FloatingPointError("Nonfinite training gradient")
            head_only = completed < controls["head_warmup_updates"]
            if head_only:
                for name, parameter in model.named_parameters():
                    if not is_head(name):
                        parameter.grad = None
            optimizer_gradient = _gradient_norm(model)
            clip = setting.get("clip")
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(parameters, float(clip), error_if_nonfinite=True)
            post = _gradient_norm(model)
            work_counters["optimizer_step_attempts"] += 1
            optimizer.step()
            if not all(bool(torch.isfinite(p).all()) for p in parameters):
                raise FloatingPointError("Nonfinite optimizer parameter")
            _sync(device); cumulative["optimization_seconds"] += time.perf_counter()-tick
            optimization_tick = None
            completed += 1; examples += len(indices)
            work_counters["committed_optimizer_updates"] += 1
            pending_validation = completed in validation_indices
            history.append(dict(update=completed, train_loss=float(loss.detach()), endpoint_loss=float(endpoint.detach()),
                rollout_loss=float(auxiliary.detach()) if config.get("rollout_weight", 0.) else None,
                source_aux_loss=float(source.detach()) if config.get("source_aux_weight", 0.) else None,
                consistency_loss=float(consistency.detach()) if config.get("consistency_weight", 0.) else None,
                endpoint_gradient_norm=endpoint_gradient, rollout_gradient_norm=auxiliary_gradient,
                source_gradient_norm=source_gradient, consistency_gradient_norm=consistency_gradient,
                scaled_objective=float(scaled.detach()), gradient_norm=pre, gradient_norm_after_clip=post,
                head_gradient_norm=head, trunk_gradient_norm=trunk,
                optimizer_gradient_norm=optimizer_gradient,
                clipped=clip is not None and optimizer_gradient > float(clip), learning_rate=current_lr,
                batch_mode=controls["batch_mode"], head_only_warmup=head_only,
                logical_updates=completed, query_visits=len(indices), query_weights=[weight for _, weight in plan],
                parent_ids=[_parent(train[i]) for i in indices], horizons=[_record(train[i])["horizon"] for i in indices],
                query_ids=[_record(train[i]).get("query_id") for i in indices],
                phase_variants=[_record(train[i]).get("phase_variant", "original") for i in indices],
                examples_seen=examples, elapsed_seconds=elapsed(), optimization_seconds=cumulative["optimization_seconds"],
                parameter_movement=_movement(model, initial_state), trunk_movement=_movement(model, initial_state, head=False)))
            checkpoint()  # Commit sampler/RNG exactly with this completed update.
            if not reporting["enabled"] or not pending_validation:
                _emit_live(history[-1], config)
            if pending_validation:
                validate()
        if completed >= updates:
            status = "COMPLETED_UPDATES"
        elif status != "TIME_CENSORED" and final_chunk:
            status = "TIME_CENSORED"
    except (TimeoutError, InterruptedError) as exc:
        status = "TIME_CENSORED" if final_chunk else "CONTINUATION_REQUIRED"
        failure = dict(type=type(exc).__name__, message=str(exc), scientific_failure=False)
    except FloatingPointError as exc:
        status = "FAILED_NONFINITE"
        failure = dict(type=type(exc).__name__, message=str(exc), scientific_failure=True)
    finally:
        if optimization_tick is not None:
            cumulative["optimization_seconds"] += time.perf_counter()-optimization_tick
        attempt.update(status=status, end_update=completed, finished_unix=time.time(),
                       elapsed_seconds=time.perf_counter()-start)
        checkpoint()
        write_json(root/"attempt.json", attempt)
    last_state = copy.deepcopy(model.state_dict())
    tick = time.perf_counter()
    try:
        final_train = evaluate_training(model, train, normalization, budget, accuracy_criteria)
    except (TimeoutError, InterruptedError) as exc:
        final_train = _unavailable_metrics(type(exc).__name__+": bounded final training diagnostic incomplete")
    if (status != "CONTINUATION_REQUIRED" and final_train["mse"] is not None
        and (attainable_train is None or final_train["mse"] < attainable_train["mse"])):
        attainable_train, attainable_update = final_train, completed
    if (status != "CONTINUATION_REQUIRED" and final_train["mse"] is not None
        and (criterion_train is None or final_train["physical_criterion"]["worst_tolerance_ratio"] < criterion_train["physical_criterion"]["worst_tolerance_ratio"])):
        criterion_train, criterion_update = final_train, completed
        criterion_state = copy.deepcopy(last_state)
    cumulative["diagnostic_seconds"] += time.perf_counter()-tick
    model.load_state_dict(selected_state)
    # Diagnostics are separate from the optimizer clock but included in total
    # trial work. They do not select any parameter/checkpoint.
    tick = time.perf_counter()
    if selected_update == 0 and initial_train is not None:
        selected_train = initial_train
    elif attainable_update == selected_update and attainable_train is not None:
        selected_train = attainable_train
    else:
        try:
            selected_train = evaluate_training(model, train, normalization, budget, accuracy_criteria)
        except (TimeoutError, InterruptedError) as exc:
            selected_train = _unavailable_metrics(type(exc).__name__+": bounded selected training diagnostic incomplete")
    selected_validation = best if best is not None else _unavailable_metrics("No completed validation observation")
    try:
        correction_diagnostics = _correction_summary(model, train, budget=budget) if config.get("record_correction_diagnostics", False) else dict(status="NA", reason="Not requested")
    except (TimeoutError, InterruptedError) as exc:
        correction_diagnostics = dict(status="UNRESOLVED", reason=str(exc))
    cumulative["diagnostic_seconds"] += time.perf_counter()-tick
    selected = dict(schema=SCHEMA, identity=identity, config=config.get("model", {}), normalization=normalization,
                    state_dict=selected_state, update=selected_update, family=config.get("family"),
                    target_kind=config.get("target_kind"), seed=seed,
                    architecture_metadata=model.architecture_metadata() if hasattr(model, "architecture_metadata") else None)
    atomic_torch_save(selected, root/"selected.pt")
    if config.get("phase") in TINY_PHASES:
        atomic_torch_save({**selected, "state_dict":criterion_state, "update":criterion_update,
            "checkpoint_role":"criterion-aware training diagnostic only; never deployment or validation selection"},
            root/"criterion-diagnostic.pt")
    model.load_state_dict(last_state)
    cumulative["total_seconds"] = elapsed()
    attempt.update(finished_unix=time.time(), elapsed_seconds=time.perf_counter()-start)
    checkpoint()
    write_json(root/"attempt.json", attempt)
    result = dict(schema=SCHEMA, trial_id=config.get("id", digest(identity)), family=config.get("family"),
        target_kind=config.get("target_kind"), phase=config.get("phase"), seed=seed, regime=config.get("regime"),
        variant_id=config.get("variant_id", "base"), data_count=config.get("data_count"),
        model_config=config.get("model", {}), primary=config.get("primary", config.get("phase") == "main"),
        status=status, failure=failure, updates=completed, requested_updates=updates,
        requested_seconds=config.get("total_seconds"), selected_update=selected_update,
        selected_initialization=selected_update == 0, selection_reason="minimum validation objective; exact ties retain earliest checkpoint",
        initial_state_sha256=_state_hash(initial_state), selected_state_sha256=_state_hash(selected_state),
        selected_checkpoint="selected.pt", selected_checkpoint_sha256=file_digest(root/"selected.pt"),
        journal="journal.pt", journal_sha256=file_digest(root/"journal.pt"),
        initial_train=initial_train or _unavailable_metrics("No completed initialization diagnostic"),
        selected_train=selected_train, final_train=final_train,
        attainable_train=attainable_train or _unavailable_metrics("No completed training diagnostic"), attainable_update=attainable_update,
        attainable_selection="minimum full-training parent-mean MSE at declared validation updates plus final iterate; diagnostic only",
        criterion_train=criterion_train or _unavailable_metrics("No completed training criterion diagnostic"),
        criterion_update=criterion_update,
        criterion_state_sha256=_state_hash(criterion_state),
        criterion_checkpoint="criterion-diagnostic.pt" if config.get("phase") in TINY_PHASES else None,
        criterion_checkpoint_sha256=file_digest(root/"criterion-diagnostic.pt") if config.get("phase") in TINY_PHASES else None,
        criterion_selection="minimum worst conservative RMS/max tolerance ratio across the entire training bank at declared observations; one checkpoint, diagnostic only",
        correction_diagnostics=correction_diagnostics, training_controls=controls,
        effective_updates=completed, examples_seen=examples,
        work=dict(logical_optimizer_updates=completed, endpoint_forward_queries=examples,
            **work_counters,
            optimizer_backward_queries=examples if controls["batch_mode"] == "full_parent" else completed,
            rollout_forward_queries=examples*int(config.get("rollout_steps", 2)) if config.get("rollout_weight", 0.) else 0,
            full_parent_weighting="each independent parent total weight one, queries averaged within parent" if controls["batch_mode"] == "full_parent" else None,
            validation_queries=sum(r.get("validation_loss") is not None for r in history)*len(validation),
            diagnostic_scope="includes initialization, full-train scheduled diagnostics, final/selected checks and optional correction instrumentation"),
        initial_validation=initial,
        selected_validation=selected_validation, normalization=normalization, setting=setting,
        cost=cumulative, chunk_seconds=time.perf_counter()-start, attempts=attempts,
        total_parameters=sum(p.numel() for p in model.parameters()), trainable_parameters=sum(p.numel() for p in parameters),
        gradient_norm_scope="scaled objective, pre-clipping; heads identified by readout/head/gate/output/amplitude names",
        component_gradient_norm_scope="NA in full_parent accumulation; aggregate head/trunk/total gradients remain measured" if controls["batch_mode"] == "full_parent" else "unscaled component objectives when auxiliary supervision is enabled",
        clipped_fraction=float(np.mean([r["clipped"] for r in history if "clipped" in r])) if completed else None,
        maximum_trunk_gradient=max((r.get("trunk_gradient_norm", 0.) for r in history), default=0.),
        trunk_movement=_movement(model, initial_state, head=False),
        precision=dict(parameters="float32", labels="float64", loss="float64", tf32=False),
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
        training_parent_ids=sorted({_parent(s) for s in train}), validation_parent_ids=sorted({_parent(s) for s in validation}),
        gradient_horizon=int(config.get("rollout_steps", 2)) if config.get("rollout_weight", 0.) else 1,
        rollout_weight=float(config.get("rollout_weight", 0.)), truncated_backpropagation=False)
    result["source_aux_weight"] = float(config.get("source_aux_weight", 0.))
    result["consistency_weight"] = float(config.get("consistency_weight", 0.))
    result["normalization_mode"] = config.get("normalization_mode", "separate")
    result["objective_mode"] = normalization["objective_mode"]
    result["supervision"] = "endpoint_plus_extra_source_labels" if config.get("source_aux_weight", 0.) else "endpoint_only"
    for row in history:
        row.update(trial_id=result["trial_id"], model_id=result["trial_id"], family=config.get("family"),
                   target_kind=config.get("target_kind"), seed=seed, phase=config.get("phase"))
        row.update(variant_id=config.get("variant_id", "base"), data_count=config.get("data_count"))
        row.update(objective_mode=normalization["objective_mode"],
                   objective_mean_scale=normalization["objective_mean_scale"],
                   objective_centered_scale=normalization["objective_centered_scale"])
    # Exact private history/checkpoint cadence is independent of reporting.
    prior_curves = [json.loads(line) for line in (root/"learning_curves.jsonl").read_text().splitlines()] if (root/"learning_curves.jsonl").exists() else []
    if reporting["enabled"]:
        public_rows, plans, export = sampled_curve_rows(history, reporting,
            chunk_start_update=chunk_start_update, prior_rows=prior_curves, pending_validation=pending_validation)
        result["learning_curve_export"] = export
        result["optimization_history_summary"] = whole_history_summary(history)
        if reporting["compact_public_query_plan"]:
            write_json(root/"query_plan.json", plans)
            result["learning_curve_export"]["query_plan_file_sha256"] = file_digest(root/"query_plan.json")
        _jsonl(root/"learning_curves.jsonl", public_rows)
        if history:
            _emit_live(history[-1], config, force=True, phase="training-terminal")
    else:
        # Legacy public exports retain every observation and their query arrays.
        retained = {int(row["update"]):row for row in prior_curves}
        retained.update({row["update"]:row for row in history if row["update"] > chunk_start_update or chunk_start_update == 0})
        _jsonl(root/"learning_curves.jsonl", [retained[key] for key in sorted(retained)])
    write_json(root/"trial.json", clean(result))
    write_json(root/"training_results.json", clean(dict(schema=SCHEMA, trials=[result])))
    return result


def _conditioned_affine_solve(a, b, *, ridge, rcond):
    """Normalize columns before a truncated FP64 SVD, then undo that scaling."""
    if not math.isfinite(ridge) or ridge < 0 or not math.isfinite(rcond) or not 0 < rcond < 1:
        raise ValueError("Affine ridge/rcond must be finite, nonnegative/strictly between zero and one")
    column_norms = a.square().sum(0).sqrt()
    nonzero = column_norms > 0
    safe_norms = torch.where(nonzero, column_norms, torch.ones_like(column_norms))
    standardized = a/safe_norms
    u, singular, vh = torch.linalg.svd(standardized, full_matrices=False)
    keep = singular > (singular[0]*rcond if len(singular) else 0.)
    denominator = singular.square()+ridge
    inverse = torch.zeros_like(singular)
    inverse[keep] = singular[keep]/denominator[keep]
    scaled = vh.T@(inverse*(u.T@b))
    coefficients = scaled/safe_norms
    coefficients[~nonzero] = 0.
    raw_singular = torch.linalg.svdvals(a)
    rank = int(keep.sum())
    return coefficients, dict(column_scales=column_norms.tolist(), singular_values=singular.tolist(),
        raw_singular_values=raw_singular.tolist(), rank=rank, discarded_singular_directions=int(len(singular)-rank),
        condition_number=float(singular[0]/singular[-1]) if rank == a.shape[1] else None,
        retained_condition_number=float(singular[0]/singular[keep][-1]) if rank else None,
        solve_contract="parent-weighted objective; unit-column FP64 truncated SVD; ridge applies in normalized coordinates",
        raw_coefficient_l2=float(coefficients.norm()), normalized_coefficient_l2=float(scaled.norm()))


def fit_fixed_features(samples, normalization, *, ridge=0., rcond=1e-10):
    """A train-only affine control with fixed local physical-state features.

    Frozen analytic features (not the neural encoder) are fitted to the same
    mean/centered objective by an FP64 SVD. It establishes attainable residual
    in this declared feature span, not a lower bound for a nonlinear network.
    """
    _assert_development(samples)
    train = [s for s in samples if _record(s)["split"] == "train" and _accepted(s)]
    if not train:
        raise ValueError("No accepted training labels for fixed-feature control")
    parent_counts = defaultdict(int)
    for sample in train:
        parent_counts[_parent(sample)] += 1
    matrices, targets = [], []
    mean_scale, centered_scale = _objective_scales(normalization)
    def features(sample):
        u = sample["u0"].double(); v = u-u.mean()
        h = float(_record(sample)["horizon"])
        # Identity-at-zero is structural, but h^3 or physical-null masks are
        # deliberately absent: this control also diagnoses closure labels.
        return torch.stack([torch.ones_like(u)*h, v*h, (v.square()-v.square().mean())*h,
                            (v.pow(3)-v.pow(3).mean())*h, torch.ones_like(u)*h*h, v*h*h])
    def design(sample):
        basis = features(sample)
        means = basis.flatten(1).mean(1)
        spatial = _physical_coordinates(basis-means.reshape(-1, *([1]*(basis.ndim-1))), _record(sample)).flatten(1).T
        weight = 1./math.sqrt(len(parent_counts)*parent_counts[_parent(sample)])
        a = torch.cat([means[None]/mean_scale, spatial/centered_scale], 0)*weight
        residual = sample["target"].double()-sample["anchor_fp32"].double()
        b = torch.cat([residual.mean()[None]/mean_scale,
                       _physical_coordinates(residual-residual.mean(), _record(sample)).flatten()/centered_scale])*weight
        return a.cpu(), b.cpu()
    for sample in train:
        a, b = design(sample); matrices.append(a); targets.append(b)
    a, b = torch.cat(matrices), torch.cat(targets)
    coefficients, solve = _conditioned_affine_solve(a, b, ridge=ridge, rcond=rcond)
    rows = []
    for sample in samples:
        if not _accepted(sample):
            continue
        prediction = sample["anchor_fp32"].double()+torch.einsum("k,k...->...", coefficients.to(sample["u0"].device), features(sample))
        rows.append(endpoint_row(prediction, sample, normalization))
    return clean(dict(schema=SCHEMA, control="fixed_analytic_features_affine_readout", precision="FP64 assembly oracle; deployed FP32 addition is separate",
        affine=True, mean_propagator_learned=False, ridge=ridge, rcond=rcond,
        feature_names=["h", "h*v", "h*center(v²)", "h*center(v³)", "h²", "h²*v"],
        coefficients=coefficients.tolist(), **solve,
        gram=(a.T@a).tolist(), train_objective=float((a@coefficients-b).square().sum()*.5),
        rows=rows, training_parent_ids=sorted(parent_counts), target_access="train-only fit; validation diagnostic only"))


def fit_frozen_encoder(model, samples, normalization, *, ridge=0., rcond=1e-10):
    """Exact affine-head attribution conditional on frozen neural features.

    Only FP64 least squares is claimed: deployed FP32 output rounding is
    measured separately after installing the coefficients into the model.
    """
    from .references import equation_geometry
    _assert_development(samples)
    if (not hasattr(model, "affine_features") or not hasattr(model, "set_affine_coefficients")
        or getattr(model, "family", None) not in ("direct_fno", "residual_fno") or getattr(model, "numerical_remainder", False)):
        return dict(schema=SCHEMA, control="frozen_random_neural_encoder_affine_head", status="NA",
                    family=getattr(model, "family", None), reason="Architecture has no exact affine coefficient interface")
    train = [s for s in samples if _record(s)["split"] == "train" and _accepted(s)]
    if not train:
        raise ValueError("No accepted training labels for frozen encoder")
    counts = defaultdict(int)
    for sample in train:
        counts[_parent(sample)] += 1
    design, labels = [], []
    mean_scale, centered_scale = _objective_scales(normalization)
    model.eval()
    with torch.no_grad():
        for sample in train:
            eq, geom = equation_geometry(_record(sample))
            base, basis = model.affine_features(sample["u0"], _record(sample)["horizon"], eq, geom, _record(sample)["target_kind"])
            basis, residual = basis.double().cpu(), (sample["target"].double()-base.double()).cpu()
            means = basis.flatten(1).mean(1)
            centered = _physical_coordinates(basis-means.reshape(-1, *([1]*(basis.ndim-1))), _record(sample)).flatten(1).T
            factor = 1./math.sqrt(len(counts)*counts[_parent(sample)])
            design.append(torch.cat([means[None]/mean_scale, centered/centered_scale])*factor)
            labels.append(torch.cat([residual.mean()[None]/mean_scale,
                _physical_coordinates(residual-residual.mean(), _record(sample)).flatten()/centered_scale])*factor)
        a, b = torch.cat(design), torch.cat(labels)
        coefficients, solve = _conditioned_affine_solve(a, b, ridge=ridge, rcond=rcond)
        fp64_rows = []
        for sample in samples:
            if not _accepted(sample):
                continue
            eq, geom = equation_geometry(_record(sample))
            base, basis = model.affine_features(sample["u0"], _record(sample)["horizon"], eq, geom, _record(sample)["target_kind"])
            design_prediction = base.double()+torch.einsum("k,k...->...", coefficients.to(basis.device), basis.double())
            fp64_rows.append(endpoint_row(design_prediction, sample, normalization))
        model.set_affine_coefficients(coefficients)
        rows = [endpoint_row(predict(model, sample), sample, normalization) for sample in samples if _accepted(sample)]
    return clean(dict(schema=SCHEMA, control="frozen_random_neural_encoder_affine_head", family=model.family,
        affine=True, fitted_mean_propagator=False, ridge=ridge, rcond=rcond, coefficients=coefficients.tolist(),
        **solve, gram=(a.T@a).tolist(), fp64_design_rows=fp64_rows,
        fp64_affine_train_objective=float(.5*(a@coefficients-b).square().sum()),
        rows=rows, deployed_assembly="Actual model FP32 output, distinct from FP64 affine design optimum",
        training_parent_ids=sorted(counts), target_access="training-only solve; validation diagnostic only"))


def load_model(path, device="cpu"):
    from .models import create_model
    # This is also an entry point in fresh Python processes. Backend settings
    # do not travel with a state_dict or the subprocess environment: cuDNN
    # defaults to allowing TF32 even though the training worker disables it.
    # Reapply the declared IEEE FP32 policy before constructing/invoking a model.
    reference_precision()
    root = Path(path)
    checkpoint = root/"selected.pt" if root.is_dir() else root
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = {**saved["config"], "normalization": saved["normalization"]}
    model = create_model(config).to(device=device, dtype=torch.float32)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, {k: v for k, v in saved.items() if k != "state_dict"}


def _checkpoint_runtime():
    """Record the arithmetic/execution settings used by both reload processes."""
    if hasattr(torch.backends, "fp32_precision"):
        precision = dict(api="fp32_precision", global_fp32=torch.backends.fp32_precision,
            cuda_matmul=torch.backends.cuda.matmul.fp32_precision,
            cudnn_convolution=torch.backends.cudnn.conv.fp32_precision)
    else:
        precision = dict(api="allow_tf32", cuda_matmul=torch.backends.cuda.matmul.allow_tf32,
            cudnn_convolution=torch.backends.cudnn.allow_tf32)
    return dict(precision=precision, threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(), cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled())


def _checkpoint_reload_child(directory, destination, *, threads, interop_threads, runtime_path=None):
    directory = Path(directory)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(interop_threads)
    if runtime_path is not None:
        declared = json.loads(Path(runtime_path).read_text())
        keys = ("deterministic_algorithms", "deterministic_warn_only", "cudnn_benchmark", "cudnn_deterministic")
        if any(type(declared.get(key)) is not bool for key in keys):
            raise ValueError("Checkpoint probe requires declared boolean backend settings")
        torch.use_deterministic_algorithms(declared["deterministic_algorithms"], warn_only=declared["deterministic_warn_only"])
        torch.backends.cudnn.benchmark = declared["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = declared["cudnn_deterministic"]
    # load_model still enforces IEEE FP32, irrespective of this execution policy.
    loaded, _ = load_model(directory, destination)
    probe = torch.load(directory/"reload-input.pt", weights_only=True, map_location="cpu")
    probe["u0"] = probe["u0"].to(destination)
    with torch.no_grad():
        output = predict(loaded, probe).cpu()
    atomic_torch_save(output, directory/"reload-output.pt")
    write_json(directory/"reload-runtime.json", dict(runtime=_checkpoint_runtime(),
        checkpoint_sha256=file_digest(directory/"selected.pt"),
        state_sha256=_state_hash(loaded.state_dict())))


def checkpoint_reload_probe(path, sample, *, device="cpu"):
    """Actually reload project state-dict bytes in another Python process."""
    root = Path(path)
    model, _ = load_model(root, device)
    stripped = dict(u0=sample["u0"].detach().cpu(), record={k: v for k, v in _record(sample).items()
        if k not in ("reference", "target", "teacher", "anchor_fp32", "anchor_fp64")})
    with torch.no_grad():
        expected = predict(model, dict(stripped, u0=stripped["u0"].to(device))).cpu()
    probe_path = root/"reload-input.pt"
    atomic_torch_save(stripped, probe_path)
    output_path = root/"reload-output.pt"
    runtime = _checkpoint_runtime()
    runtime_path = root/"reload-parent-runtime.json"
    write_json(runtime_path, runtime)
    command = [sys.executable, "-m", "tdn.analysis.response.training", "reload", str(root.resolve()), str(device),
               str(runtime["threads"]), str(runtime["interop_threads"]), str(runtime_path.resolve())]
    started = time.perf_counter()
    child = subprocess.run(command, capture_output=True, text=True, check=False, timeout=120)
    if child.returncode:
        raise RuntimeError("Fresh-process checkpoint reload failed: "+child.stderr[-4000:])
    actual = torch.load(output_path, weights_only=True, map_location="cpu")
    receipt = json.loads((root/"reload-runtime.json").read_text())
    difference = float((actual.double()-expected.double()).abs().max())
    exact_equal = torch.equal(actual, expected)
    runtime_equal = runtime == receipt["runtime"]
    checkpoint_sha256 = file_digest(root/"selected.pt")
    state_equal = _state_hash(model.state_dict()) == receipt["state_sha256"]
    checkpoint_equal = checkpoint_sha256 == receipt["checkpoint_sha256"]
    result = dict(status="PASS" if exact_equal and runtime_equal and state_equal and checkpoint_equal else "FAIL",
                  maximum_difference=difference, exact_equal=exact_equal,
                  runtime_equal=runtime_equal, state_equal=state_equal, checkpoint_equal=checkpoint_equal,
                  parent_runtime=runtime, child_runtime=receipt["runtime"],
                  elapsed_seconds=time.perf_counter()-started,
                  checkpoint_sha256=checkpoint_sha256, input_sha256=file_digest(probe_path),
                  parent_runtime_sha256=file_digest(runtime_path),
                  independent_process=True, device=str(device))
    write_json(root/"checkpoint_reload.json", result)
    return result


def _trial_config(ctx):
    unit = ctx.unit
    declared = ctx.protocol["trials"]
    if isinstance(declared, dict):
        result = copy.deepcopy(declared[unit["trial_id"]])
    else:
        result = copy.deepcopy(next(t for t in declared if t["id"] == unit["trial_id"]))
    if result.get("setting_from"):
        selection = json.loads((Path(ctx.prerequisites[result["setting_from"]])/"selection.json").read_text())
        key = f"{result['target_kind']}/{result['family']}"
        choices = selection["selected_settings"].get(key, [])
        rank = int(result.get("setting_rank", 0))
        if rank >= len(choices):
            raise ValueError("Requested optimizer setting was not selected by a completed factorial")
        result["setting"] = choices[rank]["setting"]
    if isinstance(result.get("setting"), str):
        result["setting"] = next(s for s in factorial_settings() if s["id"] == result["setting"])
    if "setting" not in result:
        raise ValueError("An explicit optimizer setting or validated selector is required")
    result.setdefault("model", {**ctx.protocol.get("model_config", {}), "family": result["family"]})
    result["model"].setdefault("family", result["family"])
    settings = ctx.protocol.get("training", {})
    result.setdefault("screen_stage", ctx.unit.get("screen_stage"))
    result.setdefault("physical_criteria", dict(rms=float(settings.get("target_rms", 2e-5)), maximum=float(settings.get("target_max", 2e-5))))
    for key in ("batch_size", "batch_mode", "validation_every", "rollout_weight", "rollout_steps", "lr_schedule",
                "head_warmup_updates", "record_correction_diagnostics", "inference_benchmark",
                "public_curve_every", "live_every", "compact_public_query_plan"):
        if key in settings:
            result.setdefault(key, settings[key])
    if result.get("phase") == "rollout":
        result.setdefault("rollout_weight", .1)
        result.setdefault("rollout_steps", 2)
    return result


def _data(ctx, config):
    from .references import load_samples
    stages = ctx.unit.get("data_stages")
    if stages is None:
        stages = list(ctx.unit.get("train_stage_ids", []))+list(ctx.unit.get("validation_stage_ids", []))
    if not stages:
        stages = [name for name, path in ctx.prerequisites.items() if (Path(path)/"dataset.json").exists()]
    samples = [sample for stage in stages for sample in load_samples(ctx.prerequisites[stage], ctx.device)]
    _assert_development(samples)
    samples = [s for s in samples if _record(s)["target_kind"] == config["target_kind"]]
    screen_stage = config.get("screen_stage")
    if screen_stage:
        if screen_stage not in ctx.prerequisites:
            raise ValueError("Screening must be a sealed training prerequisite")
        screen_path = Path(ctx.prerequisites[screen_stage])/"screening.json"
        screen = json.loads(screen_path.read_text())
        if screen.get("schema") != "tdn.response-screening/v1":
            raise ValueError("Unsupported Response screening schema")
        target = config["target_kind"]
        selected = screen.get("selected_parents", {}).get(target, {})
        regimes = [config["regime"]] if config.get("phase") in ("tiny_individual", "diagnostic_individual") else list(CORE_REGIMES)
        requested = [(split, regime, selected.get(split, {}).get(regime)) for split in ("train", "validation") for regime in regimes]
        missing = [dict(split=split, regime=regime, reason="No screen-eligible parent") for split, regime, pid in requested if not pid]
        choices = {(split, pid):regime for split, regime, pid in requested if pid}
        observed = {(_record(s)["split"], _parent(s)) for s in samples}
        missing += [dict(split=split, regime=regime, parent_id=pid, reason="Screen-selected parent missing from prepared datasets")
                    for split, regime, pid in requested if pid and (split, pid) not in observed]
        config["screening"] = dict(stage=screen_stage, sha256=file_digest(screen_path),
            status="UNRESOLVED_SCREEN" if missing else "ELIGIBLE", requested_regimes=regimes,
            selected_parents=selected, missing=missing, target_group=screen.get("target_groups", {}).get(target, {}))
        samples = [dict(s, record={**_record(s), "tiny_regime":choices[(_record(s)["split"], _parent(s))]})
                   for s in samples if (_record(s)["split"], _parent(s)) in choices]
    seen = set()
    for sample in samples:
        record = _record(sample)
        key = (record["parent_id"], record["target_kind"], tuple(record["grid"]), record["horizon"],
               record.get("phase_variant", "original"), record.get("descendant_sha256"))
        if key in seen:
            raise ValueError("Duplicated prepared training query")
        seen.add(key)
    if config.get("phase") in TINY_PHASES:
        samples = [s for s in samples if _record(s)["split"] != "train" or
                   _record(s).get("tiny_regime", _record(s).get("regime")) in CORE_REGIMES]
    if config.get("phase") in ("tiny_individual", "diagnostic_individual"):
        samples = [s for s in samples if _record(s)["split"] != "train" or
                   _record(s).get("tiny_regime", _record(s).get("regime")) == config["regime"]]
    count = config.get("data_count")
    if count:
        groups = defaultdict(list)
        for sample in samples:
            if _record(sample)["split"] == "train":
                groups[_record(sample).get("tiny_regime", _record(sample).get("regime", "unspecified"))].append(_parent(sample))
        ordered = []
        groups = {key: sorted(set(value), key=lambda pid: digest(["nested-response-subsets", pid])) for key, value in groups.items()}
        while any(groups.values()):
            for regime in sorted(groups):
                if groups[regime]:
                    ordered.append(groups[regime].pop(0))
        if len(ordered) < int(count):
            raise ValueError("Declared training-parent count exceeds prepared parent coverage")
        selected = set(ordered[:int(count)])
        samples = [s for s in samples if _record(s)["split"] != "train" or _parent(s) in selected]
    if config.get("phase") in ("tiny_individual", "diagnostic_individual"):
        ids = sorted({_parent(s) for s in samples if _record(s)["split"] == "train"})
        if ids:
            samples = [s for s in samples if _record(s)["split"] != "train" or _parent(s) == ids[0]]
    return samples, stages


def _anchor_labels(samples, config):
    """Accounted preprocessing for an explicitly changed physical anchor."""
    name = config.get("model", {}).get("anchor", "df")
    substeps = int(config.get("model", {}).get("reaction_substeps", 8))
    if config.get("source_aux_weight", 0.):
        for sample in samples:
            contract = _record(sample).get("source_supervision", {})
            if contract.get("anchor") != name or contract.get("reaction_substeps") != substeps:
                raise ValueError("Source labels require the exact model anchor and reaction_substeps identity")
    if name == "df" and substeps == 8:
        return samples
    from .references import equation_geometry
    from .numerics import anchor
    result = []
    with torch.no_grad():
        for sample in samples:
            record = _record(sample)
            eq, geom = equation_geometry(record)
            u = sample["u0"]
            result.append(dict(sample, anchor_fp32=anchor(u, record["horizon"], eq, geom, record["target_kind"],
                name=name, reaction_substeps=substeps), anchor_fp64=anchor(u.double(), record["horizon"], eq, geom,
                record["target_kind"], name=name, reaction_substeps=substeps),
                record={**record, "training_anchor":name, "training_anchor_substeps":substeps}))
    return result


def _stage_record(ctx, identity, metrics, gap=None, correct=True):
    if identity in getattr(ctx, "identities", set()):
        return
    mechanisms = ctx.unit.get("mechanism_ids", list(ctx.protocol.get("mechanisms", {"M3": {}}))[:1])
    ctx.record(identity, mechanisms, metrics=clean(metrics), checks=[
        check("training-numerically-finite", correct, True, "eq", category="math"),
        check("resolved-trainability", gap, True, "eq", category="gap",
              reason="Training gate is distinct from held-out scientific advantage")])


def _postfit_inference_benchmark(path, validation, config, trial, *, device, budget=None):
    """Time only the generated validation-selected checkpoint, with a fixed query."""
    declaration = config.get("inference_benchmark")
    if not declaration:
        return None
    if not isinstance(declaration, dict):
        raise ValueError("inference_benchmark must declare bounded repeats and warmup")
    repeats, warmup = int(declaration.get("repeats", 2)), int(declaration.get("warmup", 1))
    if repeats <= 0 or warmup < 0:
        raise ValueError("Inference repeats must be positive and warmup nonnegative")
    tick = time.perf_counter()
    unavailable = dict(schema="tdn.response-diagnostic-performance/v1", status="NA", requested_repeats=repeats,
        requested_warmup=warmup, checkpoint_role="validation-selected checkpoint",
        fresh_process="NOT_MEASURED", transfers="NOT_INCLUDED", p99=None,
        scientific_claim="Inference timing alone does not establish utility or amortization")
    if trial["status"] != "COMPLETED_UPDATES":
        result = {**unavailable, "reason":"Trial has not completed its declared optimization budget"}
    elif not validation:
        result = {**unavailable, "reason":"No accepted validation query"}
    else:
        # Selection uses immutable identifiers, never measured quality or speed.
        sample = min(validation, key=lambda value:(str(_record(value).get("query_id", "")), _parent(value),
            tuple(_record(value)["grid"]), float(_record(value)["horizon"]), str(_record(value).get("phase_variant", "original"))))
        try:
            if budget is not None:
                budget.check()
            from .diagnostic_performance import benchmark_model
            # This file was just generated by this trial. Uploaded or external
            # historical checkpoint payloads are never used by this hook.
            selected_model, selected = load_model(path, device=device)
            result = benchmark_model(selected_model, [sample], selected["normalization"],
                repeats=repeats, warmup=warmup, budget=budget)
            result.update(checkpoint_role="validation-selected checkpoint", selected_update=selected["update"],
                checkpoint_sha256=file_digest(Path(path)/"selected.pt"),
                query_selection="minimum canonical accepted validation query identifier; independent of quality and timing")
        except (TimeoutError, InterruptedError) as exc:
            result = {**unavailable, "status":"UNRESOLVED", "reason":"Bounded post-fit benchmark incomplete: "+str(exc)}
    result["postprocessing_seconds"] = time.perf_counter()-tick
    write_json(Path(path)/"inference_cost.json", clean(result))
    return result


def run_training(ctx):
    from .models import create_model
    setup_start = time.perf_counter()
    config = _trial_config(ctx)
    previous = ctx.unit.get("previous_chunk")
    if previous:
        previous_path = Path(ctx.prerequisites[previous])
        prior = json.loads((previous_path/"trial.json").read_text())
        if prior["status"] in TERMINAL:
            for name in ("selected.pt", "journal.pt", "trial.json", "fixed_features.json", "fixed_encoder.json", "fixed_encoder.pt", "checkpoint_reload.json", "criterion-diagnostic.pt", "inference_cost.json", "query_plan.json"):
                if (previous_path/name).exists():
                    shutil.copy2(previous_path/name, Path(ctx.path)/name)
            prior["continuation_noop"] = True
            prior["continuation_noop_seconds"] = time.perf_counter()-setup_start
            prior["continued_from_stage"] = previous
            write_json(Path(ctx.path)/"trial.json", prior)
            write_json(Path(ctx.path)/"training_results.json", dict(schema=SCHEMA, trials=[prior]))
            (Path(ctx.path)/"learning_curves.jsonl").write_text("")
            _stage_record(ctx, "continuation/noop/"+prior["trial_id"],
                          dict(trial_status=prior["status"], previous=previous, seconds=prior["continuation_noop_seconds"]))
            return dict(trial_id=prior["trial_id"], trial_status=prior["status"], continuation_noop=True)
    samples, data_stages = _data(ctx, config)
    if config.get("screening", {}).get("status") == "UNRESOLVED_SCREEN":
        unavailable = _unavailable_metrics("No complete declared resolvable screening bank")
        row = dict(schema=SCHEMA, trial_id=config.get("id"), family=config["family"], target_kind=config["target_kind"],
            phase=config.get("phase"), seed=config["seed"], regime=config.get("regime"), variant_id=config.get("variant_id", "base"),
            status="UNRESOLVED_SCREEN", updates=0, requested_updates=config["updates"], requested_seconds=config.get("total_seconds"),
            selected_update=0, selected_initialization=True, screening=config["screening"], initial_train=unavailable,
            selected_train=unavailable, final_train=unavailable, attainable_train=unavailable, initial_validation=unavailable,
            selected_validation=unavailable, criterion_train=unavailable, setting=config["setting"], attempts=[],
            model_config=config.get("model", {}), cost=dict(total_seconds=time.perf_counter()-setup_start),
            failure=dict(scientific_failure=False, type="UNRESOLVED_SCREEN", message="Required teacher coverage or headroom unavailable"),
            scientific_claim="No optimizer or model fit attempted; missing screening regimes retain denominator weight")
        inference = _postfit_inference_benchmark(ctx.path, [], config, row, device=ctx.device)
        if inference is not None:
            row["inference_cost"] = inference
            row["postprocessing_cost"] = dict(inference_benchmark_seconds=inference["postprocessing_seconds"])
        row["stage_total_seconds"] = time.perf_counter()-setup_start
        row["total_cost_including_postprocessing_seconds"] = row["stage_total_seconds"]
        write_json(Path(ctx.path)/"trial.json", row)
        write_json(Path(ctx.path)/"training_results.json", dict(schema=SCHEMA, trials=[row]))
        (Path(ctx.path)/"learning_curves.jsonl").write_text("")
        _stage_record(ctx, "trial/"+str(row["trial_id"]), dict(status=row["status"], screening=row["screening"]), gap=None)
        return dict(trial_id=row["trial_id"], trial_status=row["status"], updates=0, continuation_required=False)
    samples = _anchor_labels(samples, config)
    normalization = training_normalization(samples, minimum_floor=float(ctx.protocol.get("training", {}).get("normalization_floor", 1e-8)))
    screen_floor = config.get("screening", {}).get("target_group", {}).get("normalization_floor")
    if screen_floor is not None:
        screen_floor = float(screen_floor)
        if not math.isfinite(screen_floor) or screen_floor <= 0:
            raise ValueError("Screening normalization floor must be positive and finite")
        normalization["floor"] = max(normalization["floor"], screen_floor)
        for key in ("mean_scale", "centered_scale", "full_scale", "within_field_centered_scale"):
            normalization[key] = max(normalization[key], normalization["floor"])
        normalization["screening_floor"] = screen_floor
    normalization = apply_normalization_mode(normalization, config.get("normalization_mode", "separate"))
    normalization = configure_objective(normalization, config["family"], config.get("objective_mode", "family"))
    if config.get("model", {}).get("gain_mode", config.get("model", {}).get("gain_normalization", "legacy")) == "frozen":
        normalization = fit_correction_gains(samples, normalization, config)
    config["model"]["normalization"] = normalization
    torch.manual_seed(int(config["seed"])); random.seed(int(config["seed"])); np.random.seed(int(config["seed"]) % (2**32))
    model = create_model(config["model"]).to(device=ctx.device, dtype=torch.float32)
    train = [s for s in samples if _record(s)["split"] == "train" and _accepted(s)]
    validation = [s for s in samples if _record(s)["split"] == "validation" and _accepted(s)]
    metadata = software_metadata()
    identity = dict(protocol=digest(ctx.protocol), trial=config, data={name:file_digest(Path(ctx.prerequisites[name])/"dataset.json")
                    for name in data_stages}, source=ctx.protocol.get("source_tree_sha256"),
                    software={key:metadata.get(key) for key in ("source_tree_sha256", "python", "torch", "numpy", "scipy", "torch_cuda_runtime")},
                    torch=str(torch.__version__), device_type=torch.device(ctx.device).type)
    if config.get("screen_stage"):
        identity["screening"] = config["screening"]["sha256"]
    resume = Path(ctx.prerequisites[previous])/"journal.pt" if previous else None
    if (Path(ctx.path)/"journal.pt").exists():
        resume = Path(ctx.path)/"journal.pt"
    final = int(ctx.unit.get("chunk_index", 0))+1 >= int(ctx.unit.get("chunk_count", 1))
    setup_seconds = time.perf_counter()-setup_start
    row = train_trial(model, train, validation, config, normalization, ctx.path, identity=identity,
        resume=resume, budget=ctx.budget, chunk_seconds=ctx.unit.get("seconds"), final_chunk=final)
    if row["status"] != "CONTINUATION_REQUIRED" and config.get("phase") in TINY_PHASES:
        tick = time.perf_counter()
        fixed = fit_fixed_features(samples, normalization)
        fixed["elapsed_seconds"] = time.perf_counter()-tick
        write_json(Path(ctx.path)/"fixed_features.json", fixed)
        if config["family"] in ("direct_fno", "residual_fno") and not config["model"].get("numerical_remainder", False):
            tick = time.perf_counter()
            # Recreate the exact initial random encoder, not the trained one.
            torch.manual_seed(int(config["seed"]))
            frozen = create_model({**config["model"], "frozen_encoder":True}).to(device=ctx.device, dtype=torch.float32)
            fixed_encoder = fit_frozen_encoder(frozen, samples, normalization)
            fixed_encoder["elapsed_seconds"] = time.perf_counter()-tick
            write_json(Path(ctx.path)/"fixed_encoder.json", fixed_encoder)
            atomic_torch_save(dict(config={**config["model"], "frozen_encoder":True}, state_dict=frozen.state_dict(),
                normalization=normalization, family=config["family"], target_kind=config["target_kind"], seed=config["seed"]),
                Path(ctx.path)/"fixed_encoder.pt")
        else:
            fixed_encoder = dict(schema=SCHEMA, control="frozen_random_neural_encoder_affine_head", status="NA",
                family=config["family"], reason="Architecture has no exact affine coefficient interface", elapsed_seconds=0.)
            write_json(Path(ctx.path)/"fixed_encoder.json", fixed_encoder)
    if row["status"] == "COMPLETED_UPDATES":
        reload = checkpoint_reload_probe(ctx.path, train[0], device=ctx.device)
        row["checkpoint_reload"] = reload
        if reload["status"] != "PASS":
            raise ValueError("Selected checkpoint fails fresh-process output parity: "
                f"maximum_difference={reload['maximum_difference']:.9g}, "
                f"runtime_equal={reload['runtime_equal']}, state_equal={reload['state_equal']}, "
                f"checkpoint_equal={reload['checkpoint_equal']}; inspect "
                f"{Path(ctx.path)/'checkpoint_reload.json'}")
    row["accepted_training_queries"] = len(train)
    row["requested_training_queries"] = sum(_record(s)["split"] == "train" for s in samples)
    row["accepted_validation_queries"] = len(validation)
    row["requested_validation_queries"] = sum(_record(s)["split"] == "validation" for s in samples)
    row["reference_coverage_rows"] = [dict(parent_id=_parent(s), query_id=_record(s).get("query_id"),
        split=_record(s)["split"], tiny_regime=_record(s).get("tiny_regime"), horizon=_record(s)["horizon"],
        reference_accepted=_accepted(s), reference_status=_reference(s).get("status")) for s in samples]
    if config.get("screening"):
        row["screening"] = config["screening"]
    inference = _postfit_inference_benchmark(ctx.path, validation, config, row, device=ctx.device, budget=ctx.budget)
    if inference is not None:
        row["inference_cost"] = inference
    postprocessing = dict(fixed_features_seconds=fixed["elapsed_seconds"] if "fixed" in locals() else 0.,
        fixed_encoder_seconds=fixed_encoder["elapsed_seconds"] if "fixed_encoder" in locals() else 0.,
        checkpoint_reload_seconds=row.get("checkpoint_reload", {}).get("elapsed_seconds", 0.),
        inference_benchmark_seconds=inference["postprocessing_seconds"] if inference is not None else 0.)
    row["postprocessing_cost"] = postprocessing
    row["stage_setup_seconds"] = setup_seconds
    row["stage_total_seconds"] = time.perf_counter()-setup_start
    row["total_cost_including_postprocessing_seconds"] = row["cost"]["total_seconds"]+sum(postprocessing.values())+setup_seconds
    write_json(Path(ctx.path)/"trial.json", clean(row))
    write_json(Path(ctx.path)/"training_results.json", clean(dict(schema=SCHEMA, trials=[row])))
    _stage_record(ctx, "trial/"+row["trial_id"], dict(status=row["status"], updates=row["updates"],
        selected_update=row["selected_update"], selected_initialization=row["selected_initialization"],
        validation_rms=row["selected_validation"]["rms"], cost=row["cost"]), correct=row["status"] != "FAILED_NONFINITE")
    return clean(dict(trial_id=row["trial_id"], trial_status=row["status"], updates=row["updates"],
                      selected_update=row["selected_update"], continuation_required=row["status"] == "CONTINUATION_REQUIRED"))


def _latest_trials(ctx):
    trials = {}
    for stage, path in ctx.prerequisites.items():
        file = Path(path)/"trial.json"
        if file.exists():
            row = json.loads(file.read_text())
            row["stage"] = stage
            row["path"] = str(path)
            previous = trials.get(row["trial_id"])
            if previous is None or (row["updates"], len(row.get("attempts", []))) > (previous["updates"], len(previous.get("attempts", []))):
                trials[row["trial_id"]] = row
    return list(trials.values())


def select_settings(rows, *, maximum_settings=2, required_count=8):
    groups = defaultdict(list)
    for row in rows:
        if row["phase"] == "factorial":
            groups[f"{row['target_kind']}/{row['family']}"].append(row)
    result = {}; diagnostics = {}
    for key, group in groups.items():
        settings = {digest(r["setting"]):r for r in group}
        if len(settings) != required_count or any(r["status"] not in TERMINAL for r in group):
            diagnostics[key] = dict(status="BLOCKED", reason="Missing distinct completed factorial opportunities", observed=len(settings), required=required_count)
            result[key] = []; continue
        opportunities = {(r["requested_updates"], r.get("requested_seconds")) for r in group}
        if len(opportunities) != 1:
            raise ValueError("Factorial settings did not receive equal declared opportunity")
        eligible = [r for r in group if r["status"] != "FAILED_NONFINITE" and r["selected_validation"]["objective"] is not None]
        eligible.sort(key=lambda r:(r["selected_validation"]["objective"], r["selected_validation"]["rms"],
                                    r["selected_validation"]["maximum"], digest(r["setting"])))
        chosen = eligible[:maximum_settings]
        result[key] = [dict(setting=r["setting"], trial_id=r["trial_id"], objective=r["selected_validation"]["objective"],
            status=r["status"], tied_objective_trial_ids=[x["trial_id"] for x in eligible
                 if x["selected_validation"]["objective"] == r["selected_validation"]["objective"]]) for r in chosen]
        diagnostics[key] = dict(status="SELECTED" if chosen else "BLOCKED", settings=len(settings),
            tie_rule="validation objective, physical RMS, maximum, canonical setting hash; up to two retained",
            source="validation only; initial checkpoints eligible", time_censored=sum(r["status"] == "TIME_CENSORED" for r in group))
    return dict(schema=SCHEMA, selected_settings=result, diagnostics=diagnostics)


def run_select(ctx):
    rows = _latest_trials(ctx)
    declarations = ctx.protocol.get("trials", {})
    declarations = list(declarations.values()) if isinstance(declarations, dict) else declarations
    if ctx.unit.get("selection_phase", "factorial") == "factorial":
        result = select_settings(rows, maximum_settings=int(ctx.unit.get("maximum_settings", 2)),
                                 required_count=int(ctx.unit.get("required_settings", 8)))
        result["selected_models"] = [dict(family=key.split("/", 1)[1], target_kind=key.split("/", 1)[0],
            setting=choice["setting"], setting_rank=index) for key, choices in result["selected_settings"].items()
            for index, choice in enumerate(choices)]
        expected_groups = sorted({f"{r['target_kind']}/{r['family']}" for r in declarations if r["phase"] == "factorial"})
        for group in expected_groups:
            if group not in result["selected_settings"]:
                result["selected_settings"][group] = []
                result["diagnostics"][group] = dict(status="BLOCKED", reason="Entire declared factorial group is missing", observed=0)
        result["missing"] = [dict(group=group, reason=result["diagnostics"][group].get("reason", "No eligible setting"))
                             for group in expected_groups if not result["selected_settings"][group]]
    else:
        groups = defaultdict(list)
        expected = defaultdict(list)
        for spec in declarations:
            if spec["phase"] not in ("factorial", *TINY_PHASES):
                key = (spec["target_kind"], spec["family"], spec.get("variant_id", "base"), int(spec.get("data_count") or 0), spec["seed"])
                expected[key].append(spec["id"])
                groups[key] = []
        for row in rows:
            if row["phase"] not in ("factorial", *TINY_PHASES):
                groups[(row["target_kind"], row["family"], row.get("variant_id", "base"),
                        int(row.get("data_count") or 0), row["seed"])].append(row)
        selected = []; missing = []
        for key, trials in sorted(groups.items()):
            candidates = [row for row in trials if row["status"] in ("COMPLETED_UPDATES", "TIME_CENSORED")
                          and row["selected_validation"]["objective"] is not None]
            candidates.sort(key=lambda row:(row["selected_validation"]["objective"], row["trial_id"]))
            if not candidates:
                missing.append(dict(target_kind=key[0], family=key[1], variant_id=key[2], data_count=key[3], seed=key[4],
                                    expected_trial_ids=expected.get(key, []),
                                    reason="No completed or time-censored finite selected checkpoint"))
                continue
            absent = sorted(set(expected.get(key, []))-{row["trial_id"] for row in trials})
            if absent:
                missing.append(dict(target_kind=key[0], family=key[1], variant_id=key[2], data_count=key[3], seed=key[4],
                                    expected_trial_ids=absent, reason="Missing declared optimizer opportunity within group"))
            row = candidates[0]
            selected.append(dict(target_kind=key[0], family=key[1], variant_id=key[2], data_count=key[3], seed=key[4],
                method_id=f"{key[1]}::{key[2]}::n{key[3]}",
                trial_status=row["status"], trial_id=row["trial_id"], primary=row.get("primary", False),
                trial_directory=row["path"], stage=row["stage"], setting=row["setting"],
                checkpoint_sha256=row["selected_checkpoint_sha256"], validation_objective=row["selected_validation"]["objective"],
                selected_initialization=row["selected_initialization"]))
        result = dict(schema=SCHEMA, selected_models=selected, missing=missing,
                      selected_settings={}, expected_group_count=len(expected), observed_trial_count=len(rows),
                      selection_contract="validation objective within family/target/variant/data-count/seed; every declared group retained")
    write_json(Path(ctx.path)/"selection.json", result)
    _stage_record(ctx, "optimizer/selection", dict(groups=len(result["selected_settings"]),
        selected=sum(len(x) for x in result["selected_settings"].values())), gap=None)
    return result


def trainability_gate(rows, *, seeds, regimes=CORE_REGIMES, rms_target=2e-5, max_target=2e-5, profile="pilot", expected_groups=None):
    if len(set(seeds)) != len(seeds) or len(set(regimes)) != 6:
        raise ValueError("M3 requires distinct seeds and exactly six declared resolvable regimes")
    individual = [r for r in rows if r["phase"] in ("tiny_individual", "diagnostic_individual")]
    groups = sorted(set(expected_groups or ()) | {(r["target_kind"], r["family"]) for r in individual})
    outcomes = []; cells = []
    for target, family in groups:
        seed_rows = []
        for seed in seeds:
            successes = 0; unresolved = 0; floors = 0
            for regime in regimes:
                matches = [r for r in individual if (r["target_kind"],r["family"],r["seed"],r.get("regime")) == (target,family,seed,regime)]
                if len(matches) > 1:
                    raise ValueError("Gate requires one preselected tiny trial per family/seed/regime")
                row = matches[0] if matches else None
                cell = dict(target_kind=target, family=family, seed=seed, regime=regime, status="MISSING", passed=False)
                if row and row["status"] == "UNRESOLVED_SCREEN":
                    cell.update(status="UNRESOLVED_SCREEN", trial_id=row["trial_id"], screening=row.get("screening"))
                    row = None
                if row and row["status"] == "FAILED_NONFINITE":
                    cell.update(status="FAILED_NONFINITE", trial_id=row["trial_id"])
                    row = None
                if row and (row.get("initial_train", {}).get("mse") is None or
                            row.get("attainable_train", row.get("final_train", row["selected_train"])).get("mse") is None):
                    cell.update(status="UNMEASURED_TRAINING_DIAGNOSTIC", trial_id=row["trial_id"])
                    row = None
                if row:
                    initial = row["initial_train"]["mse"]
                    achieved = row.get("attainable_train", row.get("final_train", row["selected_train"]))
                    final = achieved["mse"]
                    endpoints = achieved["rows"]
                    floor = all(e["floor_limited"] for e in endpoints)
                    absolute = all(e["conservative_rms"] <= rms_target and e["conservative_maximum"] <= max_target for e in endpoints)
                    reduction = initial/final if final > 0 else None
                    resolvable = all(e["reference_accepted"] for e in endpoints) and not floor
                    met = (final <= initial/100.) and absolute and resolvable
                    finished = row["status"] == "COMPLETED_UPDATES"
                    cell.update(status="FLOOR_LIMITED" if floor else row["status"],
                        passed=met and finished, initial_mse=initial, final_mse=final,
                        reduction_factor=reduction, physical_target_passed=absolute, floor_limited=floor,
                        selected_initialization=row["selected_initialization"], trial_id=row["trial_id"],
                        trunk_movement=row.get("trunk_movement"), maximum_trunk_gradient=row.get("maximum_trunk_gradient"))
                    cell["attainable_update"] = row.get("attainable_update", row.get("updates"))
                    floors += int(floor)
                successes += int(cell["passed"])
                unresolved += int(cell["status"] in ("MISSING", "TIME_CENSORED", "CONTINUATION_REQUIRED", "FLOOR_LIMITED", "UNMEASURED_TRAINING_DIAGNOSTIC", "UNRESOLVED_SCREEN"))
                cells.append(cell)
            joint = [r for r in rows if r["phase"] in ("tiny_joint", "diagnostic_joint") and
                     (r["target_kind"], r["family"], r["seed"]) == (target, family, seed)]
            if len(joint) > 1:
                raise ValueError("Gate requires one preselected joint tiny trial per family/seed")
            joint_passes = 0; joint_cells = []
            for regime in regimes:
                item = dict(regime=regime, status="MISSING", passed=False)
                if joint:
                    trial = joint[0]
                    if trial["status"] == "UNRESOLVED_SCREEN":
                        item.update(status="UNRESOLVED_SCREEN", screening=trial.get("screening"))
                    initial_rows = [e for e in trial["initial_train"]["rows"] if e.get("tiny_regime", e.get("regime")) == regime]
                    achieved = trial.get("attainable_train", trial.get("final_train", trial["selected_train"]))
                    final_rows = [e for e in achieved["rows"] if e.get("tiny_regime", e.get("regime")) == regime]
                    if initial_rows and len(initial_rows) == len(final_rows):
                        initial_mse = _parent_mean([(e["parent_id"], e["mse"]) for e in initial_rows])
                        final_mse = _parent_mean([(e["parent_id"], e["mse"]) for e in final_rows])
                        limited = all(e["floor_limited"] for e in final_rows)
                        absolute = all(e["conservative_rms"] <= rms_target and e["conservative_maximum"] <= max_target for e in final_rows)
                        resolved = all(e["reference_accepted"] for e in final_rows) and not limited
                        item.update(status="FLOOR_LIMITED" if limited else trial["status"], initial_mse=initial_mse,
                            final_mse=final_mse, passed=resolved and absolute and final_mse <= initial_mse/100.
                            and trial["status"] == "COMPLETED_UPDATES", attainable_update=trial.get("attainable_update"))
                joint_passes += int(item["passed"])
                joint_cells.append(item)
            joint_unresolved = sum(r["status"] in ("MISSING", "TIME_CENSORED", "CONTINUATION_REQUIRED", "FLOOR_LIMITED", "UNRESOLVED_SCREEN") for r in joint_cells)
            seed_rows.append(dict(seed=seed, passed_regimes=successes, joint_passed_regimes=joint_passes,
                joint_regimes=joint_cells, required=5, total_regimes=len(regimes),
                unresolved=unresolved+joint_unresolved, floor_limited=floors,
                passed=successes >= 5 and joint_passes >= 5))
        passed = len(seeds) >= 3 and all(r["passed"] for r in seed_rows) and profile != "smoke"
        outcomes.append(dict(target_kind=target, family=family, passed=passed, seeds=seed_rows,
            status="PASS" if passed else "UNRESOLVED" if any(s["unresolved"] for s in seed_rows) or profile == "smoke" else "FAIL"))
    return dict(schema=SCHEMA, milestone="M3", gate_passed=bool(outcomes) and all(r["passed"] for r in outcomes),
        groups=outcomes, cells=cells, profile=profile, required_seeds=list(seeds),
        criterion="100-fold training MSE reduction and conservative physical RMS/max targets in >=5/6 resolved regimes for all 3 seeds, both individual and joint fits",
        floor_policy="separate absolute checks; floor cases do not replace six resolvable cases",
        scientific_claim="Attainable training fit diagnostic only; deployment checkpoint remains validation-selected; not held-out superiority",
        joint_trials=[r["trial_id"] for r in rows if r["phase"] in ("tiny_joint", "diagnostic_joint")])


def run_gate(ctx):
    settings = ctx.protocol.get("training", {})
    result = trainability_gate(_latest_trials(ctx), seeds=ctx.protocol.get("seeds", settings.get("seeds", [])),
        regimes=ctx.unit.get("regimes", settings.get("tiny_regimes", CORE_REGIMES)),
        rms_target=float(settings.get("target_rms", 2e-5)), max_target=float(settings.get("target_max", 2e-5)),
        profile=ctx.protocol.get("profile", "pilot"),
        expected_groups=list(itertools.product(ctx.protocol.get("targets", []), settings.get("primary_families", []))))
    write_json(Path(ctx.path)/"gate.json", result)
    _stage_record(ctx, "M3/diverse-trainability", dict(groups=result["groups"], gate_passed=result["gate_passed"]), gap=result["gate_passed"])
    return result


def run(ctx):
    kind = ctx.unit["kind"]
    if kind in ("gate", "trainability_gate"):
        return run_gate(ctx)
    if kind in ("select", "optimizer_select"):
        return run_select(ctx)
    return run_training(ctx)


if __name__ == "__main__":
    if len(sys.argv) not in (6, 7) or sys.argv[1] != "reload":
        raise SystemExit("Internal checkpoint probe: reload PATH DEVICE THREADS INTEROP_THREADS [RUNTIME_JSON]")
    _checkpoint_reload_child(sys.argv[2], sys.argv[3], threads=int(sys.argv[4]), interop_threads=int(sys.argv[5]),
                             runtime_path=sys.argv[6] if len(sys.argv) == 7 else None)
