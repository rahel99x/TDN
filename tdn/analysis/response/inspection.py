"""Bounded, read-only inspection of a coupled model on a declared input.

Channel factors are not causal or identifiable components: rescaling,
permutation, and rotations can preserve their combined response. Effective
rank here describes observed fields on one state, not a population rank.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from tdn.runtime.metadata import write_json
from tdn.research.protocol import file_digest, digest
from . import numerics as n
from .references import equation_geometry
from .training import _state_hash

SCHEMA = "tdn.response-source-inspection/v1"


def _array(value):
    return value.detach().cpu().numpy()


def _rms(value, geometry, target):
    return float(n.inner(value.double(), value.double(), geometry, target).clamp_min(0).mean().sqrt())


def field_rank(fields, geometry, target, *, rtol=1e-6):
    """Singular values in the physical spatial inner product, including Nyquist."""
    if fields.ndim != geometry.ndim+3 or fields.shape[1:3] != (1, 1):
        raise ValueError("Inspection rank expects [channels,1,1,*grid] fields")
    count = fields.shape[0]
    gram = torch.empty((count, count), dtype=torch.float64)
    for i in range(count):
        for j in range(i+1):
            value = n.inner(fields[i].double(), fields[j].double(), geometry, target).mean()
            gram[i, j] = gram[j, i] = value.cpu()
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0).flip(0)
    singular = eigenvalues.sqrt()
    total = float(eigenvalues.sum())
    if total == 0.:
        entropy_rank, stable_rank, rank = 0., 0., 0
    else:
        probabilities = eigenvalues[eigenvalues > 0]/total
        entropy_rank = float(torch.exp(-(probabilities*probabilities.log()).sum()))
        stable_rank = total/float(eigenvalues[0])
        rank = int((singular > singular[0]*rtol).sum())
    return dict(singular_values=singular.tolist(), gram=gram.tolist(), numerical_rank=rank,
        effective_rank_entropy=entropy_rank, stable_rank=stable_rank, relative_singular_cutoff=rtol,
        total_physical_energy=total, norm="physical interpolant L2" if target != "fd_same_grid" else "nodal L2")


def spectral_bands(fields, geometry, target):
    """Per-channel energy in fixed physical cycles/unit, not grid fractions."""
    spectrum = torch.fft.fftn(fields.double(), dim=tuple(range(2, fields.ndim)), norm="forward")
    radial = torch.zeros(geometry.grid, dtype=torch.float64, device=fields.device)
    for axis, (size, spacing) in enumerate(zip(geometry.grid, geometry.dx)):
        wave = torch.fft.fftfreq(size, d=spacing, device=fields.device, dtype=torch.float64)
        shape = [1]*geometry.ndim; shape[axis] = size
        radial = radial+wave.reshape(shape).square()
    radial = radial.sqrt()
    mass = torch.ones_like(radial) if target == "fd_same_grid" else n.mass_weights(fields.double(), geometry)
    power = spectrum.abs().square()*mass
    edges = (0., 2., 4., 8., 16., float("inf"))
    return [dict(lower_cycles_per_unit=left, upper_cycles_per_unit=right if math.isfinite(right) else None,
                 channel_energy=power[..., (radial >= left) & (radial < right)].sum(-1).flatten().cpu().tolist())
            for left,right in zip(edges[:-1], edges[1:])]


def mean_balance(model, encoded, time_value, prediction):
    """Inspect the actual decoded mean path; bounded clocks need not match variance.

    Differentiation includes the implemented finite mean-path quadrature. The
    latent clock rate by itself is not substituted for this derivative.
    """
    h = float(time_value)
    def differentiate(function, *, identity_branch=False):
        method, difference, step = "autograd of implemented trajectory", None, None
        try:
            if identity_branch and h == 0:
                raise RuntimeError("Literal h=0 identity branch requires right derivative")
            with torch.enable_grad():
                time_input = torch.tensor(h, dtype=encoded.u.dtype, device=encoded.u.device, requires_grad=True)
                value = function(time_input)
                derivative = float(torch.autograd.grad(value.double().mean(), time_input)[0].detach())
            if not math.isfinite(derivative):
                raise FloatingPointError("Nonfinite mean derivative")
        except (RuntimeError, FloatingPointError):
            # Second-order differences with explicit halving sensitivity;
            # this is a diagnostic, not an error certificate.
            step = max(h, model.t_ref, 1e-3)*torch.finfo(encoded.u.dtype).eps**(1/3)
            def estimate(delta):
                def at(t):
                    return float(function(t).double().mean())
                if h >= delta:
                    return (at(h+delta)-at(h-delta))/(2*delta)
                return (-3*at(h)+4*at(h+delta)-at(h+2*delta))/(2*delta)
            coarse, derivative = estimate(step), estimate(step/2)
            difference = abs(derivative-coarse)
            method = "second-order finite difference with step halving"
            if not math.isfinite(derivative):
                raise FloatingPointError("Nonfinite numerical mean derivative")
        return derivative, method, difference, step
    derivative, method, difference, step = differentiate(lambda t:model.mean_path(encoded,t))
    decoded_derivative, decoded_method, decoded_difference, decoded_step = differentiate(
        lambda t:model.decode(encoded,t), identity_branch=True)
    physical_mean = float(n.mean(prediction.double()).mean())
    path_mean = float(model.mean_path(encoded, h).double().mean())
    centered = n.center(prediction.double())
    variance = float(n.inner(centered, centered, encoded.geom, encoded.target).mean())
    rate = float(encoded.eq.reaction_rate)
    balance = rate*(physical_mean*(1-physical_mean)-variance)
    same_grid = encoded.target in ("fd_same_grid", "galerkin_same_grid")
    result = dict(mean_path_derivative=derivative, mean_derivative_method=method,
        mean_derivative_halving_difference=difference, mean_derivative_coarse_step=step,
        reconstructed_mean_derivative=decoded_derivative, reconstructed_mean_derivative_method=decoded_method,
        reconstructed_mean_derivative_halving_difference=decoded_difference,
        reconstructed_mean_derivative_coarse_step=decoded_step,
        reconstructed_physical_variance=variance,
        reconstructed_mean_minus_mean_path=physical_mean-path_mean,
        mean_ode_residual=decoded_derivative-balance if same_grid else None,
        mean_path_ode_residual=derivative-rate*(path_mean*(1-path_mean)-variance) if same_grid else None,
        mean_ode_residual_scope="Actual decoded-mean derivative minus finite-equation RHS from reconstructed field; consistency is a penalty, not an identity or accuracy certificate" if same_grid else "NA: unresolved fine-field variance contributes a closure term",
        clock_variance_status="NA", clock_implied_variance=None,
        clock_implied_minus_physical_variance=None,
        clock_variance_relative_gap=None, clock_variance_relative_gap_denominator=None,
        clock_variance_relative_gap_definition="(implied-physical)/max(abs(implied),abs(physical)); NA when both are zero")
    reason = None
    if model.mean_parameterization != "clock":
        reason = "No bounded clock parameterization"
    elif encoded.target != "fd_same_grid":
        reason = "Bounded clock relation only declared for nodal FD"
    elif rate <= 0:
        reason = "Implied-variance inversion requires positive reaction rate"
    elif bool(((encoded.u < 0) | (encoded.u > 1)).any()) or bool(((prediction < 0) | (prediction > 1)).any()):
        reason = "Initial or reconstructed field violates bounded [0,1] clock conditions"
    elif not 0 < path_mean < 1:
        reason = "Mean is on or outside the bounded clock interior"
    else:
        implied = path_mean*(1-path_mean)-derivative/rate
        denominator = max(abs(implied), abs(variance))
        result.update(clock_variance_status="MEASURED", clock_implied_variance=implied,
            clock_implied_minus_physical_variance=implied-variance,
            clock_variance_relative_gap=(implied-variance)/denominator if denominator else None,
            clock_variance_relative_gap_denominator=denominator,
            clock_implied_variance_is_nonnegative=implied >= 0)
    result["clock_variance_reason"] = reason
    return result


def inspect_model(model, sample, *, fractions=(0., .25, .5, .75, 1.), budget=None):
    """Return numerical metadata and arrays without reading a teacher label."""
    if getattr(model, "family", None) != "coupled_source":
        raise ValueError("Source channel inspection requires a coupled_source model")
    record = sample["record"]
    if any(n < 1 or n > 128 for n in record["grid"]):
        raise ValueError("Representative source arrays are bounded to grids of at most 128 per axis")
    if not fractions or len(fractions) > 33 or any(not math.isfinite(float(t)) or not 0 <= t <= 1 for t in fractions):
        raise ValueError("Inspection fractions must lie in the declared query horizon")
    eq, geom = equation_geometry(record)
    u = sample["u0"]
    if u.shape[:2] != (1, 1):
        raise ValueError("Inspection uses exactly one declared parent, scalar field")
    previous_mode = model.training
    model.eval()
    before = _state_hash(model.state_dict())
    start = time.perf_counter()
    try:
        with torch.no_grad():
            encoded = model.encode(u, eq, geom, record["target_kind"])
            initial_components = model.source_components(encoded, 0.)
            fields = initial_components["fields"]
            coefficients = initial_components["coefficients"]
            actual_rates = .01+19.99*torch.sigmoid(model.source_rate_logits) if model.learned_time_rates else model.source_rates
            rates_used = model.source_representation != "direct_interpolation" and model.source_time_basis != "polynomial"
            arrays = dict(input=_array(u), source_fields=_array(fields),
                source_coefficients=_array(coefficients), source_rates=_array(actual_rates),
                mean_coefficients=_array(encoded.mean_coefficients))
            sources, predictions, anchors, effective, basis_values, time_rows, reconstruction = [], [], [], [], [], [], []
            for fraction in fractions:
                if budget is not None:
                    budget.check()
                h = float(record["horizon"])*float(fraction)
                components = model.source_components(encoded, h)
                temporal = components["temporal_basis"]
                weights = components["effective_coefficients"]
                source = model.source_terms(encoded, h)
                rebuilt = n.center((fields*weights).sum(1, keepdim=True))
                delta = (source.double()-rebuilt.double()).abs().max()
                prediction = model.decode(encoded, h)
                anchor = model._anchor(encoded, h)
                correction = prediction.double()-anchor.double()
                learned_mean = model.mean_path(encoded, h)
                if not all(bool(torch.isfinite(value).all()) for value in (fields, weights, source, prediction, anchor, learned_mean)):
                    raise FloatingPointError("Nonfinite source inspection; no finite response or rank claim is available")
                rows = dict(time=h, fraction=float(fraction), learned_mean=float(learned_mean.mean()),
                    anchor_mean=float(anchor.double().mean()), prediction_mean=float(prediction.double().mean()),
                    mean_correction=float(correction.mean()), centered_correction_rms=_rms(n.center(correction), geom, record["target_kind"]),
                    full_correction_rms=_rms(correction, geom, record["target_kind"]),
                    maximum_correction=float(correction.abs().max()), source_rms=_rms(source, geom, record["target_kind"]),
                    source_mean=float(source.double().mean()), reconstruction_maximum_difference=float(delta),
                    source_spectral_bands=spectral_bands(source, geom, record["target_kind"]),
                    effective_source_coefficients=_array(weights).reshape(-1).tolist(),
                    temporal_basis=_array(temporal).reshape(-1).tolist(),
                    finite=bool(torch.isfinite(prediction).all() and torch.isfinite(source).all()))
                rows.update(mean_balance(model, encoded, h, prediction))
                time_rows.append(rows)
                sources.append(source); predictions.append(prediction); anchors.append(anchor)
                effective.append(weights); basis_values.append(temporal); reconstruction.append(float(delta))
            sources_tensor = torch.stack(sources)
            field_channels = fields.movedim(1, 0).unsqueeze(2)
            active = (fields*effective[-1]).movedim(1, 0).unsqueeze(2)
            source_spectrum = torch.fft.fftn(fields.double(), dim=tuple(range(2, fields.ndim)), norm="forward")
            combined_spectrum = torch.fft.fftn(sources_tensor.double(), dim=tuple(range(3, sources_tensor.ndim)), norm="forward")
            arrays.update(time=np.asarray([row["time"] for row in time_rows], dtype=np.float64),
                effective_coefficients=_array(torch.stack(effective)), temporal_basis=_array(torch.stack(basis_values)),
                combined_source=_array(sources_tensor), prediction=_array(torch.stack(predictions)),
                anchor=_array(torch.stack(anchors)), correction=_array(torch.stack(predictions).double()-torch.stack(anchors).double()),
                source_spectrum_real=_array(source_spectrum.real), source_spectrum_imag=_array(source_spectrum.imag),
                combined_source_spectrum_real=_array(combined_spectrum.real), combined_source_spectrum_imag=_array(combined_spectrum.imag))
            symbol = -n.diffusion_symbol(u, eq, geom, record["target_kind"])
            metadata = dict(schema=SCHEMA, status="COMPLETED", parent_id=record["parent_id"], query_id=record.get("query_id"),
                target_kind=record["target_kind"], grid=list(record["grid"]), lengths=list(record["lengths"]),
                horizon=float(record["horizon"]), kappa=float(record["kappa"]), reaction_rate=float(record["reaction_rate"]),
                input_sha256=record.get("input_sha256"), phase_variant=record.get("phase_variant", "original"),
                mean=float(u.double().mean()), minimum=float(u.min()), maximum=float(u.max()),
                variance=float(n.inner(n.center(u.double()), n.center(u.double()), geom, record["target_kind"]).mean()),
                reaction_horizon=float(record["reaction_rate"])*float(record["horizon"]),
                maximum_diffusion_horizon=float(symbol.max())*float(record["horizon"]),
                source_rank_requested=model.rank, source_time_basis=model.source_time_basis,
                effective_time_basis="piecewise_cubic_interpolation" if model.source_representation == "direct_interpolation" else model.source_time_basis,
                source_representation=model.source_representation, source_decoder=model.source_decoder,
                learned_time_rates=model.learned_time_rates, source_rates=_array(actual_rates).tolist(),
                source_rates_used=rates_used,
                source_rates_scope="Active temporal rates" if rates_used else "Unused declared buffer; effective temporal basis does not use these rates",
                mean_parameterization=model.mean_parameterization, source_feedback=model.source_feedback,
                assembly=model.assembly, numerical_remainder=model.numerical_remainder,
                source_field_spectral_bands=spectral_bands(fields, geom, record["target_kind"]),
                source_field_rank=field_rank(field_channels, geom, record["target_kind"]),
                active_final_rank=field_rank(active, geom, record["target_kind"]),
                observed_source_time_rank=field_rank(sources_tensor, geom, record["target_kind"]),
                times=time_rows, parameter_state_sha256=before,
                elapsed_seconds=time.perf_counter()-start, teacher_access=False,
                maximum_source_reconstruction_difference=max(reconstruction),
                zero_active_source=all(row["source_rms"] == 0 for row in time_rows),
                interpretation="Conditional on one declared input. Channel factors have scale/permutation/rotation ambiguity; no causal channel identification or population-rank claim.")
    finally:
        model.train(previous_mode)
    after = _state_hash(model.state_dict())
    if after != before:
        raise ValueError("Read-only source inspection changed learned parameters or buffers")
    metadata["parameters_unchanged"] = True
    return metadata, arrays


def run(ctx, samples, methods):
    root = Path(ctx.path); root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter(); rows = []
    settings = ctx.protocol.get("response", {}).get("inspection", {})
    seconds = float(settings.get("seconds", 60.))
    fractions = tuple(settings.get("fractions", (0., .25, .5, .75, 1.)))
    if not samples:
        result = dict(schema=SCHEMA, rows=[], status="NOT_MEASURED", reason="No declared input parent", elapsed_seconds=0.)
        write_json(root/"source_diagnostics.json", result)
        return result
    sample = samples[0]
    selected = [(name, method) for name, method in methods.items()
                if getattr(getattr(method, "model", None), "family", None) == "coupled_source"]
    if not selected:
        rows.append(dict(status="NOT_MEASURED", family="coupled_source", parent_id=sample["record"]["parent_id"],
                         reason="No selected coupled-source prototype checkpoint; analytic and endpoint models do not expose learned source channels"))
    class InspectionBudget:
        def check(self):
            ctx.budget.check()
            if time.perf_counter()-started >= seconds:
                raise TimeoutError("Bounded source inspection budget exceeded")
    bound = InspectionBudget()
    for name, method in selected:
        metadata = getattr(method, "metadata", {})
        identity = dict(model_id=name, method=metadata.get("method_id", name), family=metadata.get("family", "coupled_source"),
                        seed=metadata.get("seed"), variant_id=metadata.get("variant_id"), checkpoint_path=metadata.get("checkpoint_path"),
                        selection_sha256=metadata.get("selection_sha256"))
        try:
            bound.check()
            row, arrays = inspect_model(method.model, sample, fractions=fractions, budget=bound)
            filename = "source-inspection-"+digest([name, sample["record"].get("query_id"), row["parameter_state_sha256"]])[:20]+".npz"
            destination = root/filename
            temporary = destination.with_suffix(".npz.partial")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **arrays)
            temporary.replace(destination)
            row.update(arrays=filename, arrays_sha256=file_digest(destination),
                array_shapes={key:list(value.shape) for key,value in arrays.items()},
                array_dtypes={key:str(value.dtype) for key,value in arrays.items()})
        except (TimeoutError, InterruptedError) as exc:
            row = dict(status="CENSORED", reason=str(exc), parent_id=sample["record"]["parent_id"])
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            row = dict(status="NOT_MEASURED" if any(g > 128 for g in sample["record"]["grid"]) else "FAILED",
                       reason=f"{type(exc).__name__}: {exc}", parent_id=sample["record"]["parent_id"])
        rows.append(dict(row, **identity))
        write_json(root/"source_diagnostics.json", dict(schema=SCHEMA, rows=rows,
            expected_models=[name for name,_ in selected], elapsed_seconds=time.perf_counter()-started,
            selection="first declared input query, all selected coupled models; no outcome-informed parent selection",
            array_contract="numeric-only NPZ; load with allow_pickle=False; bounded spatial dimensions <=128"))
    result = dict(schema=SCHEMA, rows=rows, expected_models=[name for name,_ in selected],
        elapsed_seconds=time.perf_counter()-started, selection="first declared input query; no outcome-informed selection",
        source_scope="Parameter/source inspection, not an accuracy evaluation or timing benchmark",
        array_contract="numeric-only NPZ; load with allow_pickle=False; bounded spatial dimensions <=128")
    write_json(root/"source_diagnostics.json", result)
    return result
