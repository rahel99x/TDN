"""Matrix-free response-action compression, with explicit oracle attribution.

Training snapshots have O(probes * grid) storage; no grid-by-grid propagator is
assembled.  Source-to-action factors are tested on previously unused directions.
Projection of a known action into a spatial POD is a separately named oracle,
and cannot be mistaken for a deployable operator or an operator-rank result.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import time
from typing import Callable

import numpy as np
import torch

from tdn.analysis.response import numerics as n
from tdn.analysis.response.contracts import Physics
from tdn.numerics.types import Geometry


SUPPORTED_TARGETS = ("fd_same_grid", "galerkin_same_grid")


def _target(target):
    if target not in SUPPORTED_TARGETS:
        raise ValueError("Tangent compression supports same-grid FD/Galerkin only; projected-fine closure is unsupported")


def _validate_fields(value, geom, target):
    _target(target)
    if tuple(value.shape[-geom.ndim:]) != tuple(geom.grid):
        raise ValueError("Field shape differs from declared compression geometry")
    if value.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(value).all()):
        raise ValueError("Compression fields must be finite FP32/FP64 tensors")


def physical_whiten(value, geom, target):
    """An invertible real transform whose Euclidean norm is physical RMS.

    Unlike flattening nodal samples directly, the Galerkin transform includes
    half-mass weights on each even-grid Nyquist axis. No centering removes mean.
    """
    _validate_fields(value, geom, target)
    count = math.prod(geom.grid)
    if target == "fd_same_grid":
        return value/math.sqrt(count)
    axes = tuple(range(value.ndim-geom.ndim, value.ndim))
    mass = n.mass_weights(value, geom)
    spectrum = torch.fft.fftn(value, dim=axes, norm="ortho")
    return torch.fft.ifftn(spectrum*mass.sqrt(), dim=axes, norm="ortho").real/math.sqrt(count)


def physical_unwhiten(value, geom, target):
    _validate_fields(value, geom, target)
    count = math.prod(geom.grid)
    if target == "fd_same_grid":
        return value*math.sqrt(count)
    axes = tuple(range(value.ndim-geom.ndim, value.ndim))
    mass = n.mass_weights(value, geom)
    spectrum = torch.fft.fftn(value*math.sqrt(count), dim=axes, norm="ortho")
    return torch.fft.ifftn(spectrum/mass.sqrt(), dim=axes, norm="ortho").real


def field_metrics(value, geom, target):
    """Physical mean/centered/RMS and an explicitly sampled maximum."""
    _validate_fields(value, geom, target)
    rows = value.double().reshape(-1, 1, *geom.grid)
    means = n.mean(rows)
    rms = n.inner(rows, rows, geom, target).clamp_min(0).mean().sqrt()
    centered = n.center(rows)
    crms = n.inner(centered, centered, geom, target).clamp_min(0).mean().sqrt()
    samples = rows if target == "fd_same_grid" else n.resample(rows, tuple(4*k for k in geom.grid))
    return dict(rms=float(rms), mse=float(rms.square()), mean_rms=float(means.square().mean().sqrt()),
                centered_rms=float(crms), maximum=float(samples.abs().max()),
                maximum_scope="FD nodal maximum" if target == "fd_same_grid" else "4x interpolant sampled maximum, not continuous supremum",
                norm="normalized physical L2; Galerkin Nyquist masses included")


def _sample_weights(weights, count, *, dtype, device):
    value = torch.ones(count, dtype=dtype, device=device) if weights is None else torch.as_tensor(weights, dtype=dtype, device=device).reshape(-1)
    if value.numel() != count or not bool(torch.isfinite(value).all()) or bool((value < 0).any()) or not bool(value.sum() > 0):
        raise ValueError("Finite nonnegative query weights with positive sum required")
    return value/value.sum()


def conditional_fit(design, target, geom, target_kind, *, ridge=0., rcond=1e-10, weights=None):
    """Stable affine coefficient subproblem; it is not a nonlinear optimum.

    ``design`` is [K, *target.shape]. Prefix rows receive the supplied weights;
    callers must supply parent weights when parents have unequal query counts.
    Columns are equilibrated in the physical metric. Ridge acts on these frozen
    equilibrated coordinates, and its value/coordinate convention is retained.
    """
    tick = time.perf_counter()
    _validate_fields(target, geom, target_kind)
    _validate_fields(design, geom, target_kind)
    if design.ndim != target.ndim+1 or tuple(design.shape[1:]) != tuple(target.shape) or design.shape[0] < 1:
        raise ValueError("Design requires [coefficients, *target.shape]")
    if not math.isfinite(ridge) or ridge < 0 or not math.isfinite(rcond) or not 0 <= rcond < 1:
        raise ValueError("Nonnegative ridge and relative cutoff in [0,1) required")
    k, pixels = design.shape[0], math.prod(geom.grid)
    rows = target.numel()//pixels
    w = _sample_weights(weights, rows, dtype=torch.float64, device=target.device)
    a = physical_whiten(design.double(), geom, target_kind).reshape(k, rows, pixels)
    a = (a*w.sqrt()[None, :, None]).reshape(k, -1).T
    b = (physical_whiten(target.double(), geom, target_kind).reshape(rows, pixels)*w.sqrt()[:, None]).reshape(-1)
    norms = torch.linalg.vector_norm(a, dim=0)
    scales = torch.where(norms > 0, norms, torch.ones_like(norms))
    equilibrated = a/scales
    u, singular, vh = torch.linalg.svd(equilibrated, full_matrices=False)
    cutoff = float(singular[0])*rcond if singular.numel() else 0.
    keep = singular > cutoff
    inverse = torch.zeros_like(singular)
    inverse[keep] = singular[keep]/(singular[keep].square()+ridge)
    coefficients = (vh.T@(inverse*(u.T@b)))/scales
    prediction = torch.einsum("k,k...->...", coefficients, design.double())
    deployed = torch.einsum("k,k...->...", coefficients.float(), design.float())
    retained = singular[keep]
    return dict(coefficients=coefficients, prediction=prediction, deployed_prediction=deployed,
                rank=int(keep.sum()), singular_values=singular.tolist(), rcond=float(rcond),
                cutoff=cutoff, ridge=float(ridge), column_norms=norms.tolist(), column_scales=scales.tolist(),
                gram=(a.T@a).tolist(), equilibrated_gram=(equilibrated.T@equilibrated).tolist(),
                coefficient_norm=float(coefficients.norm()),
                retained_condition_ratio=float(retained[0]/retained[-1]) if retained.numel() else None,
                physical_error=field_metrics(prediction-target.double(), geom, target_kind),
                deployment_error=field_metrics(deployed.double()-target.double(), geom, target_kind),
                deployment_sensitivity=field_metrics(deployed.double()-prediction, geom, target_kind),
                fit_seconds=time.perf_counter()-tick, svd_calls=1,
                ridge_scope="squared coefficient norm in column-equilibrated coordinates",
                optimum_scope="FP64 conditional affine fit with fixed factors/path; no global nonlinear optimum claim")


def fit_pod(fields, geom, target, rank, *, split="train", weights=None, rcond=1e-10):
    """Train-only physical POD; a spatial-vector approximation, not an operator."""
    if split != "train":
        raise ValueError("Shared compression bases use train parents only")
    _validate_fields(fields, geom, target)
    if type(rank) is not int or rank not in (2, 4, 8):
        raise ValueError("Declared response ranks are 2, 4, and 8")
    if not math.isfinite(rcond) or not 0 <= rcond < 1:
        raise ValueError("Finite relative cutoff in [0,1) required")
    pixels = math.prod(geom.grid)
    values = fields.double().reshape(-1, 1, *geom.grid)
    w = _sample_weights(weights, len(values), dtype=torch.float64, device=fields.device)
    flat = physical_whiten(values, geom, target).reshape(len(values), pixels)
    _, singular, vh = torch.linalg.svd(flat*w.sqrt()[:, None], full_matrices=False)
    cutoff = float(singular[0])*rcond if singular.numel() else 0.
    effective = int((singular > cutoff).sum())
    retained = min(rank, effective)
    basis = physical_unwhiten(vh[:retained].reshape(retained, 1, *geom.grid), geom, target)
    total = float(singular.square().sum())
    return dict(basis=basis, requested_rank=rank, retained_rank=retained, singular_values=singular.tolist(),
                cutoff=cutoff, rcond=float(rcond), training_split=split,
                discarded_energy_fraction=float(singular[retained:].square().sum())/total if total else None,
                training_sha256=hashlib.sha256(values.detach().cpu().numpy().tobytes()).hexdigest(),
                basis_scope="Shared spatial output/source basis; does not establish full operator rank")


@dataclass
class ResponseFactors:
    """Finite-rank map for tested source/time context, never the complete U."""
    alpha: torch.Tensor
    beta: torch.Tensor
    geometry: Geometry
    target: str
    metadata: dict
    temporal_beta: torch.Tensor | None = None

    def to_payload(self):
        """Tensor/primitive-only payload, compatible with weights_only loading."""
        return dict(alpha=self.alpha.detach(), beta=self.beta.detach(),
                    temporal_beta=self.temporal_beta.detach() if self.temporal_beta is not None else None,
                    grid=list(self.geometry.grid), lengths=list(self.geometry.lengths),
                    target=self.target, metadata=dict(self.metadata))

    @classmethod
    def from_payload(cls, payload):
        geom = Geometry(tuple(payload["grid"]), tuple(payload["lengths"]))
        _target(payload["target"])
        alpha, beta = payload["alpha"], payload["beta"]
        _validate_fields(alpha, geom, payload["target"]); _validate_fields(beta, geom, payload["target"])
        if alpha.shape != beta.shape or alpha.ndim != geom.ndim+2 or alpha.shape[1] != 1 or len(alpha) > 8:
            raise ValueError("Saved response factors must have matched rank-by-field shapes")
        temporal = payload.get("temporal_beta")
        if temporal is not None and (temporal.ndim != alpha.ndim+1 or temporal.shape[0] != len(alpha) or
                temporal.shape[1] not in (1, 2, 3, 4) or tuple(temporal.shape[2:]) != tuple(alpha.shape[1:])):
            raise ValueError("Saved temporal factor shape differs from declared response rank")
        if temporal is not None:
            _validate_fields(temporal, geom, payload["target"])
        return cls(alpha, beta, geom, payload["target"], dict(payload["metadata"]), temporal)

    def apply(self, source, s=None, h=None, *, deployment=False):
        _validate_fields(source, self.geometry, self.target)
        if tuple(source.shape[-self.geometry.ndim-1:-self.geometry.ndim]) != (1,):
            raise ValueError("Scalar-channel source required")
        dtype = torch.float32 if deployment else torch.float64
        value = source.to(dtype)
        alpha, beta = self.alpha.to(dtype), self.beta.to(dtype)
        if not len(alpha):
            return torch.zeros_like(value)
        source_rows = physical_whiten(value, self.geometry, self.target).reshape(-1, math.prod(self.geometry.grid))
        if self.temporal_beta is None:
            beta_rows = physical_whiten(beta, self.geometry, self.target).reshape(len(beta), math.prod(self.geometry.grid))
            scores = source_rows@beta_rows.T
        else:
            if s is None or h is None:
                raise ValueError("Time-conditioned response factors require source and endpoint times")
            channels = self.temporal_beta.shape[1]
            factors = self.temporal_beta.to(dtype)
            beta_rows = physical_whiten(factors, self.geometry, self.target).reshape(len(alpha)*channels, math.prod(self.geometry.grid))
            temporal = time_features(s, h, t_ref=self.metadata["t_ref"], channels=channels, dtype=dtype, device=source.device)
            if temporal.shape[0] not in (1, len(source_rows)):
                raise ValueError("Times must be scalar or one pair per source field")
            scores = ((source_rows@beta_rows.T).reshape(len(source_rows), len(alpha), channels)*temporal[:, None]).sum(-1)
        return (scores@alpha.reshape(len(alpha), -1)).reshape_as(value)


def time_features(s, h, *, t_ref=.12, channels=4, dtype=torch.float64, device=None):
    """Signed coefficient dictionary with an exact zero diagonal; no forced decay."""
    if channels not in (1, 2, 3, 4) or not math.isfinite(t_ref) or t_ref <= 0:
        raise ValueError("Declare one to four temporal channels and positive reference time")
    start, stop = torch.broadcast_tensors(torch.as_tensor(s, dtype=dtype, device=device), torch.as_tensor(h, dtype=dtype, device=device))
    if not bool(torch.isfinite(start).all()) or not bool(torch.isfinite(stop).all()) or bool((start < 0).any()) or bool((stop < start).any()):
        raise ValueError("Finite ordered nonnegative response intervals required")
    left, right, delta = start.reshape(-1)/t_ref, stop.reshape(-1)/t_ref, (stop-start).reshape(-1)/t_ref
    channels_list = (delta, delta*left/(1+left), delta*right/(1+right), delta.square()/(1+delta))
    return torch.stack(channels_list[:channels], dim=1)


def fit_source_map(sources, actions, geom, target, rank, *, split="train", basis_kind="response_action_pod", weights=None, ridge=0., rcond=1e-10, oracle=False, time_pairs=None, temporal_channels=4, t_ref=.12):
    """Fit a finite-rank source->(U-G) map from charged action snapshots.

    With ``time_pairs``, declared causal time features enter the input regression
    without dividing labels by any time/gate. Output spatial rank remains J;
    temporal-channel count is reported separately. Without pairs it is a static
    same-context map, not an invented interpolation across arbitrary intervals.
    Per-state oracle fitting is explicitly nondeployable even if source probes
    used for evaluation were withheld from its local fit.
    """
    tick = time.perf_counter()
    if split != "train" and not oracle:
        raise ValueError("Shared source maps fit training parents only")
    if sources.shape != actions.shape or sources.ndim != geom.ndim+2 or sources.shape[1] != 1:
        raise ValueError("Matching [snapshots,1,*grid] sources/actions required")
    _validate_fields(sources, geom, target); _validate_fields(actions, geom, target)
    if basis_kind not in ("source_energy_pod", "response_action_pod", "endpoint_weighted_pod"):
        raise ValueError("Unknown shared basis control")
    if not math.isfinite(ridge) or ridge < 0:
        raise ValueError("Nonnegative finite ridge required")
    basis = fit_pod(sources if basis_kind == "source_energy_pod" else actions, geom, target, rank,
                    split="train", weights=weights, rcond=rcond)
    pixels = math.prod(geom.grid)
    x = physical_whiten(sources.double(), geom, target).reshape(len(sources), pixels)
    y = physical_whiten(actions.double(), geom, target).reshape(len(actions), pixels)
    temporal = None
    if time_pairs is not None:
        pairs = torch.as_tensor(time_pairs, dtype=x.dtype, device=x.device)
        if pairs.shape != (len(sources), 2):
            raise ValueError("One declared source/endpoint time pair per action required")
        temporal = time_features(pairs[:, 0], pairs[:, 1], t_ref=t_ref, channels=temporal_channels, dtype=x.dtype, device=x.device)
        x = (x[:, None]*temporal[:, :, None]).reshape(len(sources), -1)
    alpha = basis["basis"]
    if not len(alpha):
        beta = alpha.clone(); singular = torch.zeros(0, dtype=x.dtype, device=x.device)
        cutoff, effective, coefficients = 0., 0, x.new_zeros((0, x.shape[1]))
    else:
        q = physical_whiten(alpha, geom, target).reshape(len(alpha), pixels)
        w = _sample_weights(weights, len(x), dtype=x.dtype, device=x.device).sqrt()[:, None]
        u, singular, vh = torch.linalg.svd(x*w, full_matrices=False)
        cutoff = float(singular[0])*rcond if singular.numel() else 0.
        keep = singular > cutoff
        inverse = torch.zeros_like(singular)
        inverse[keep] = singular[keep]/(singular[keep].square()+ridge)
        coefficients = ((vh.T*inverse)@(u.T@((y@q.T)*w))).T
        shaped = coefficients.reshape(len(alpha), temporal_channels if temporal is not None else 1, 1, *geom.grid)
        full_beta = physical_unwhiten(shaped, geom, target)
        beta = full_beta[:, 0]
        effective = int(keep.sum())
    retained = singular[singular > cutoff]
    metadata = {key: val for key, val in basis.items() if key != "basis"}
    metadata.update(basis_kind=basis_kind, source_singular_values=singular.tolist(), source_effective_rank=effective,
                    source_cutoff=cutoff, ridge=float(ridge), beta_norms=[field_metrics(b, geom, target)["rms"] for b in beta],
                    source_norms=[field_metrics(s, geom, target)["rms"] for s in sources],
                    alpha_gram=(physical_whiten(alpha, geom, target).reshape(len(alpha), pixels)@physical_whiten(alpha, geom, target).reshape(len(alpha), pixels).T).tolist() if len(alpha) else [],
                    beta_gram=(coefficients@coefficients.T).tolist(), coefficient_norm=float(coefficients.norm()),
                    source_gram_scope="Factor-coordinate Grams retained; no probe-by-probe or grid-by-grid dense Gram allocated",
                    retained_source_condition_ratio=float(retained[0]/retained[-1]) if retained.numel() else None,
                    deployable=not oracle, state_oracle=bool(oracle),
                    fit_scope="Per-state action-label oracle; nondeployable" if oracle else "Train-only fixed shared source-to-action regression with declared time dictionary" if temporal is not None else "Train-only static source-to-action regression; same time context only",
                    rank_claim="Finite-rank correction on tested source distribution, not uniform rank of U-G")
    metadata.update(fit_seconds=time.perf_counter()-tick, fit_snapshots=len(sources), svd_calls=2 if len(alpha) else 1,
                    t_ref=float(t_ref), temporal_channels=temporal_channels if temporal is not None else 0,
                    time_feature_gram=(temporal.T@temporal).tolist() if temporal is not None else None,
                    time_dictionary="delta * [1, left/(1+left), right/(1+right), delta/(1+delta)] with delta=(h-s)/t_ref" if temporal is not None else None,
                    matrix_free_storage_scope="O(snapshot_count * spatial_dofs + retained_rank * spatial_dofs), no dense spatial propagator")
    if temporal is not None and len(alpha):
        metadata["temporal_beta_norms"] = [[field_metrics(field, geom, target)["rms"] for field in channels] for channels in full_beta]
    metadata["alpha_norms"] = [field_metrics(field, geom, target)["rms"] for field in alpha]
    return ResponseFactors(alpha, beta, geom, target, metadata, full_beta if temporal is not None and len(alpha) else None)


def _seed(seed, *identity):
    raw = json.dumps([int(seed), *identity], separators=(",", ":"), sort_keys=True).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**63-1)


def predeclare_probes(u0, geom, target, parent_id, query_id, horizon, *, split="train", seed=0,
                     source_nodes=4, band_probes=4, random_probes=4, source_fractions=None):
    """Freeze distributions/roles before action evaluation, with signed probes.

    Actual residual slots are materialized from the declared anchor before any
    action runs. Independent Fourier/random directions have no action inputs.
    Probe counts are paired work, not numbers of independent field samples.
    """
    _validate_fields(u0, geom, target)
    if u0.shape[0] != 1 or u0.shape[1] != 1 or not math.isfinite(horizon) or horizon < 0:
        raise ValueError("One parent and a finite nonnegative horizon required")
    if source_nodes < 1 or band_probes < 0 or random_probes < 0:
        raise ValueError("Finite positive source-node count and nonnegative probe counts required")
    x, weights = np.polynomial.legendre.leggauss(source_nodes)
    nodes = (x+1)*horizon/2; weights = weights*horizon/2
    if source_fractions is not None:
        fractions = tuple(float(v) for v in source_fractions)
        if not fractions or len(set(fractions)) != len(fractions) or any(not math.isfinite(v) or not 0 <= v < 1 for v in fractions):
            raise ValueError("Unique declared source fractions in [0,1) required")
        slots = [(index, s, weight, True, False) for index, (s, weight) in enumerate(zip(nodes, weights))]
        slots += [(source_nodes+index, horizon*fraction, 0., False, True) for index, fraction in enumerate(fractions)]
    else:
        slots = [(index, s, weight, True, True) for index, (s, weight) in enumerate(zip(nodes, weights))]
    result = []
    axes = [torch.arange(size, dtype=torch.float64, device=u0.device)/size for size in geom.grid]
    mesh = torch.meshgrid(*axes, indexing="ij")
    for node_index, s, weight, residual_slot, independent_slot in slots:
        directions = [("residual", 0, None, "fit")] if residual_slot else []
        for index in range(band_probes if independent_slot else 0):
            axis = index % geom.ndim
            frequency = 0 if index == 0 else (1 if index < 2 else max(1, geom.grid[axis]//3))
            phase_seed = _seed(seed, parent_id, query_id, node_index, "fourier", index)
            phase = 2*math.pi*(phase_seed % 1000003)/1000003 if frequency else 0.
            vector = torch.cos(2*math.pi*frequency*mesh[axis]+phase).reshape_as(u0)
            sign = -1. if index % 2 else 1.
            directions.append(("fourier_band", index, sign*vector, "fit" if index % 2 == 0 else "heldout_direction"))
        for index in range(random_probes if independent_slot else 0):
            generator = torch.Generator(device="cpu").manual_seed(_seed(seed, parent_id, query_id, node_index, "random", index))
            vector = (2*torch.randint(0, 2, tuple(u0.shape), generator=generator, dtype=torch.int64)-1).double().to(u0.device)
            directions.append(("signed_random", index, vector, "fit" if index % 2 == 0 else "heldout_direction"))
        for kind, index, source, role in directions:
            if source is not None:
                magnitude = field_metrics(source, geom, target)["rms"]
                source = source/magnitude if magnitude else source
            identity = dict(parent_id=str(parent_id), query_id=str(query_id), split=split, target_kind=target,
                            horizon=float(horizon), source_time=float(s), source_node=node_index, quadrature_weight=float(weight),
                            probe_kind=kind, probe_index=index, probe_role=role,
                            source_fraction=float(s/horizon) if horizon else 0.,
                            seed=_seed(seed, parent_id, query_id, node_index, kind, index), geometry=list(geom.grid), lengths=list(geom.lengths))
            identity["probe_id"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
            result.append({**identity, "source": source})
    return result


def _default_action(path, eq, geom, target, source, s, h, *, steps, tolerance, max_steps=None, budget=None):
    from .tangent import response_action
    tick = time.perf_counter()
    if not math.isfinite(tolerance) or tolerance <= 0 or type(steps) is not int or steps < 1:
        raise ValueError("Positive finite action tolerance and integer work required")
    cap = 2*steps if max_steps is None else max_steps
    if type(cap) is not int or cap < steps:
        raise ValueError("Action refinement cap must cover initial steps")
    values, calls, committed, requested, failure = {}, 0, 0, 0, None
    for mode in ("tangent", "mean"):
        count, coarse, history = steps, None, []
        difference = None
        while count <= cap:
            calls += 1; requested += count
            try:
                fine = response_action(path, eq, geom, target, source, s, h, steps=count, mode=mode, budget=budget)
            except (TimeoutError, InterruptedError, FloatingPointError) as error:
                failure = type(error).__name__+": "+str(error)
                fine = coarse
                break
            committed += count
            if coarse is not None:
                difference = field_metrics(fine-coarse, geom, target)
            history.append(dict(steps=count, difference=difference))
            if difference is not None and max(difference["rms"], difference["maximum"]) <= tolerance/2:
                break
            coarse, count = fine, 2*count
        values[mode] = fine
        values[mode+"_uncertainty"] = difference
        values[mode+"_refinements"] = history
        if failure is not None:
            break
    for mode in ("tangent", "mean"):
        values.setdefault(mode, None)
        values.setdefault(mode+"_uncertainty", None)
    available = failure is None and all(values[mode+"_uncertainty"] is not None for mode in ("tangent", "mean"))
    uncertainty = sum(values[mode+"_uncertainty"]["rms"] for mode in ("tangent", "mean")) if available else None
    maximum_uncertainty = sum(values[mode+"_uncertainty"]["maximum"] for mode in ("tangent", "mean")) if available else None
    values.update(uncertainty=uncertainty, maximum_uncertainty=maximum_uncertainty,
                  resolved=available and max(uncertainty, maximum_uncertainty) <= tolerance,
                  reason=failure,
                  work=dict(homogeneous_solves=calls, requested_tangent_steps=requested, completed_tangent_steps=committed,
                            wall_seconds=time.perf_counter()-tick), uncertainty_kind="independent coarse/fine refinement difference estimate, not certificate")
    return values


def collect_actions(path, eq, geom, target, probes, *, action_callback=None, steps=16, tolerance=1e-10, max_steps=None, budget=None):
    """Materialize every residual before actions; preserve unresolved labels."""
    prepared = []
    for probe in probes:
        item = dict(probe)
        try:
            if budget is not None:
                budget.check()
            if item["source"] is None:
                item["source"] = path.evaluate(item["source_time"])["residual"].detach().double()
                item["residual_evaluations"] = 1
        except (TimeoutError, InterruptedError, FloatingPointError) as error:
            item["preparation_failure"] = type(error).__name__+": "+str(error)
        prepared.append(item)
    records = []
    for item in prepared:
        if item.get("preparation_failure") or item["source"] is None:
            item.update(tangent=None, mean=None, action=None, uncertainty=None, status="UNRESOLVED",
                        reason=item.get("preparation_failure", "Residual preparation unavailable"),
                        source_metrics=field_metrics(item["source"], geom, target) if item["source"] is not None else None,
                        action_metrics=None)
            records.append(item)
            continue
        measured = (_default_action(path, eq, geom, target, item["source"], item["source_time"], item["horizon"],
                                   steps=steps, tolerance=tolerance, max_steps=max_steps, budget=budget) if action_callback is None else
                    action_callback(path, eq, geom, target, item["source"], item["source_time"], item["horizon"]))
        if not isinstance(measured, dict) or "tangent" not in measured or "mean" not in measured:
            raise ValueError("Action callback must return measured tangent/mean fields and uncertainty metadata")
        if measured["tangent"] is None or measured["mean"] is None:
            item.update(measured, action=None, source_metrics=field_metrics(item["source"], geom, target),
                        action_metrics=None, status="UNRESOLVED")
            records.append(item)
            continue
        if measured["tangent"].shape != item["source"].shape or measured["mean"].shape != item["source"].shape:
            raise ValueError("Measured actions must preserve source shape")
        uncertainty = measured.get("uncertainty")
        if uncertainty is None or not math.isfinite(float(uncertainty)) or float(uncertainty) < 0:
            measured["resolved"] = False
            measured["uncertainty"] = None
        correction = measured["tangent"]-measured["mean"]
        item.update(measured, action=correction, source_metrics=field_metrics(item["source"], geom, target),
                    action_metrics=field_metrics(correction, geom, target),
                    status="RESOLVED" if measured.get("resolved", False) else "UNRESOLVED")
        records.append(item)
    return records


def _direction_diagnostics(records, geom, target, rcond=1e-10):
    """Check direction independence on each prospectively fixed state/time menu.

    No probe is dropped or redrawn after this calculation. Gram sizes are probe
    counts, never grid dimensions. A direction diagnostic does not grant action
    accuracy or count a probe as an independent population sample.
    """
    menus = {}
    for rec in records:
        menus.setdefault((rec["parent_id"], rec["query_id"], rec["source_node"]), []).append(rec)
    summaries = []
    pixels = math.prod(geom.grid)
    for identity, group in menus.items():
        all_requested = len(group)
        group = [rec for rec in group if rec["source"] is not None]
        if not group:
            summaries.append(dict(parent_id=identity[0], query_id=identity[1], source_node=identity[2],
                                  requested_directions=all_requested, available_directions=0, status="UNRESOLVED"))
            continue
        x = physical_whiten(torch.cat([rec["source"] for rec in group]), geom, target).reshape(len(group), pixels)
        norms = x.norm(dim=1)
        normalized = x/torch.where(norms > 0, norms, torch.ones_like(norms))[:, None]
        _, singular, _ = torch.linalg.svd(normalized, full_matrices=False)
        cut = float(singular[0])*rcond if singular.numel() else 0.
        fit = normalized[[index for index, rec in enumerate(group) if rec["probe_role"] == "fit"]]
        _, fit_singular, fit_vh = torch.linalg.svd(fit, full_matrices=False)
        fit_cut = float(fit_singular[0])*rcond if fit_singular.numel() else 0.
        fit_basis = fit_vh[fit_singular > fit_cut]
        for index, rec in enumerate(group):
            novelty = normalized[index]-(normalized[index]@fit_basis.T)@fit_basis
            rec["source_novelty_relative_to_fit"] = float(novelty.norm()) if norms[index] > 0 else None
            rec["source_linearly_independent_of_fit"] = rec["probe_role"] == "heldout_direction" and float(novelty.norm()) > rcond
        summaries.append(dict(parent_id=identity[0], query_id=identity[1], source_node=identity[2],
                              requested_directions=all_requested, available_directions=len(group), retained_direction_rank=int((singular > cut).sum()),
                              normalized_source_gram=(normalized@normalized.T).tolist(), singular_values=singular.tolist(),
                              relative_cutoff=rcond, independent_population_units=1))
    return summaries


def _weights(records, *, endpoint=False):
    counts = {}
    for rec in records:
        counts[rec["parent_id"]] = counts.get(rec["parent_id"], 0)+1
    values = [1/counts[rec["parent_id"]]*(abs(rec["quadrature_weight"])**2 if endpoint else 1.) for rec in records]
    if not any(values):
        values = [1/counts[rec["parent_id"]] for rec in records]
    return values


def _clean_metadata(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _clean_metadata(val) for key, val in value.items() if key not in ("source", "action", "tangent", "mean", "basis")}
    if isinstance(value, (list, tuple)):
        return [_clean_metadata(v) for v in value]
    return value


def _sample_contract(sample):
    rec = sample.get("record", sample)
    geom = sample.get("geom")
    eq = sample.get("eq")
    if geom is None or eq is None:
        from tdn.analysis.response.references import equation_geometry
        old_eq, old_geom = equation_geometry(rec)
        geom = old_geom if geom is None else geom
        eq = old_eq if eq is None else eq
    return rec, eq, geom


def _endpoint_quadrature_check(context, residuals, *, steps, tolerance, budget=None):
    """Independently compare source quadrature with a forced-equation solve.

    The forced response has its own time refinement. Its difference from the
    action sum estimates source quadrature error; neither estimate is a bound.
    No exact future path or teacher is consumed by these inference primitives.
    """
    from .tangent import forced_response
    if not residuals or any(r["status"] != "RESOLVED" for r in residuals):
        return dict(status="UNRESOLVED", reason="Unresolved residual action nodes"), {}
    tick = time.perf_counter()
    path, eq, geom, target = (context[k] for k in ("path", "eq", "geom", "target"))
    h = float(context["record"]["horizon"])
    errors, refinements = {}, {}
    attempted, requested = 0, 0
    try:
        for mode, key in (("tangent", "tangent"), ("mean", "mean")):
            attempted += 1; requested += steps
            coarse = forced_response(path, eq, geom, target, h, steps=steps, mode=mode, budget=budget)
            attempted += 1; requested += 2*steps
            fine = forced_response(path, eq, geom, target, h, steps=2*steps, mode=mode, budget=budget)
            action_sum = -sum(r["quadrature_weight"]*r[key] for r in residuals)
            errors[mode] = field_metrics(action_sum-fine, geom, target)
            refinements[mode] = field_metrics(fine-coarse, geom, target)
    except (TimeoutError, InterruptedError, FloatingPointError) as error:
        return (dict(status="UNRESOLVED", reason=type(error).__name__+": "+str(error),
                     source_quadrature_differences=errors, independent_forced_time_refinements=refinements),
                dict(forced_solves=attempted, requested_forced_steps=requested, wall_seconds=time.perf_counter()-tick))
    rms = sum(errors[mode]["rms"]+refinements[mode]["rms"] for mode in errors)
    maximum = sum(errors[mode]["maximum"]+refinements[mode]["maximum"] for mode in errors)
    return (dict(status="RESOLVED" if max(rms, maximum) <= tolerance else "UNRESOLVED",
                uncertainty_rms=rms, uncertainty_maximum=maximum,
                source_quadrature_differences=errors, independent_forced_time_refinements=refinements,
                tolerance=float(tolerance), uncertainty_is_certificate=False,
                uncertainty_scope="independent forced-solve/action-quadrature comparison plus forced time refinement"),
           dict(forced_solves=4, requested_forced_steps=6*steps, wall_seconds=time.perf_counter()-tick))


def collect_experiment_actions(samples, config=None, *, action_callback=None,
                               endpoint_reference_callback=None, precomputed_actions=None,
                               precomputed_endpoint_checks=None, acquisition_receipts=None, budget=None):
    """Bounded per-query acquisition, separate from shared global factor fitting.

    Controllers may acquire one immutable query per allocation, then supply its
    sealed labels to ``run_experiment`` without numerical reacquisition. Source
    and artifact seals belong to the controller; frozen probe identities and
    roles are independently rebound here and acquisition work remains visible.
    """
    from .tangent import AnchorPath
    cfg = dict(config or {})
    cfg.setdefault("steps", cfg.get("initial_steps", 16))
    cfg.setdefault("action_tolerance", cfg.get("tolerance", 1e-10))
    cfg.setdefault("quadrature_tolerance", cfg.get("tolerance", 1e-10))
    cfg.setdefault("band_probes", 2*cfg.get("probes_per_type", 2))
    cfg.setdefault("random_probes", 2*cfg.get("probes_per_type", 2))
    cfg.setdefault("action_absolute_floor", cfg.get("nontrivial_action_threshold", 1e-10))
    ranks = tuple(cfg.get("ranks", (2, 4, 8)))
    if not ranks or any(r not in (2, 4, 8) for r in ranks):
        raise ValueError("Compression ranks must be a declared subset of 2/4/8")
    plans, contexts, requested = [], {}, []
    seen_parents = {}
    for sample in samples:
        rec, eq, geom = _sample_contract(sample)
        target = rec["target_kind"]
        if target not in SUPPORTED_TARGETS:
            requested.append(dict(parent_id=rec["parent_id"], query_id=rec["query_id"], status="NOT_APPLICABLE", reason="Projected-fine requires a separate hidden-state/forcing contract"))
            continue
        parent, query, split = rec["parent_id"], rec["query_id"], rec["split"]
        if parent in seen_parents and seen_parents[parent] != split:
            raise ValueError("Parent identity crosses compression splits")
        seen_parents[parent] = split
        key = (target, tuple(geom.grid), tuple(geom.lengths))
        path = sample.get("path") or AnchorPath(sample["u0"].double(), eq, geom, target,
                                               reaction_substeps=cfg.get("reaction_substeps", 8), budget=budget)
        plan = predeclare_probes(sample["u0"], geom, target, parent, query, float(rec["horizon"]), split=split,
                                seed=cfg.get("seed", 0), source_nodes=cfg.get("source_nodes", 4),
                                band_probes=cfg["band_probes"], random_probes=cfg["random_probes"],
                                source_fractions=cfg.get("source_fractions"))
        contexts[(parent, query)] = dict(sample=sample, record=rec, eq=eq, geom=geom, target=target, path=path, key=key)
        plans.append((contexts[(parent, query)], plan))
        requested.append(dict(parent_id=parent, query_id=query, split=split, status="REQUESTED", probes=len(plan)))
    # Hash every prospectively declared distribution before collecting any action.
    plan_rows = [_clean_metadata(p) for _, probes in plans for p in probes]
    plan_hash = hashlib.sha256(json.dumps(plan_rows, sort_keys=True, allow_nan=False).encode()).hexdigest()
    indexed = None
    if precomputed_actions is not None:
        supplied = precomputed_actions.get("_actions", []) if isinstance(precomputed_actions, dict) else precomputed_actions
        indexed = {row["probe_id"]: row for row in supplied}
        if len(indexed) != len(supplied) or set(indexed) != {row["probe_id"] for row in plan_rows}:
            raise ValueError("Precomputed actions must exactly cover the frozen probe plan without duplicate identities")
    checks = {}
    if precomputed_endpoint_checks is not None:
        if isinstance(precomputed_endpoint_checks, dict):
            checks = precomputed_endpoint_checks
        else:
            checks = {(r["parent_id"], r["query_id"]): r["check"] for r in precomputed_endpoint_checks}
    records = []
    work = dict(action_probes=0, homogeneous_solves=0, requested_tangent_steps=0, residual_evaluations=0, wall_seconds=0., endpoint_residual_nodes=0)
    for context, probes in plans:
        rows = ([dict(indexed[p["probe_id"]]) for p in probes] if indexed is not None else
                collect_actions(context["path"], context["eq"], context["geom"], context["target"], probes,
                               action_callback=action_callback, steps=cfg.get("steps", 16),
                               tolerance=cfg.get("action_tolerance", 1e-10), max_steps=cfg.get("max_steps"), budget=budget))
        for row, planned in zip(rows, probes):
            if any(row.get(key) != planned.get(key) for key in ("parent_id", "query_id", "split", "source_time", "horizon", "probe_role", "probe_kind", "seed")):
                raise ValueError("Precomputed action metadata differs from frozen probe contract")
            row["context_key"] = context["key"]
            uncertainty = row.get("uncertainty")
            row["nontrivial_threshold"] = max(cfg.get("action_absolute_floor", 1e-10),
                                               cfg.get("action_relative_floor", 1e-8)*row["source_metrics"]["rms"],
                                               10*float(uncertainty)) if uncertainty is not None else None
            row["nontrivial"] = row["status"] == "RESOLVED" and row["nontrivial_threshold"] is not None and row["action_metrics"]["rms"] > row["nontrivial_threshold"]
            work["action_probes"] += 1
            work["residual_evaluations"] += row.get("residual_evaluations", 0)
            for name, value in row.get("work", {}).items():
                work[name] = work.get(name, 0)+value
        records.extend(rows)
    endpoint_checks = {}
    for identity, context in contexts.items():
        residuals = [r for r in records if (r["parent_id"], r["query_id"]) == identity and r["probe_kind"] == "residual"]
        if indexed is not None:
            check, charged = checks.get(identity, dict(status="UNRESOLVED", reason="Precomputed endpoint/source-quadrature check missing")), {}
        else:
            check, charged = (_endpoint_quadrature_check(context, residuals, steps=cfg.get("steps", 16),
                            tolerance=cfg.get("quadrature_tolerance", 1e-10), budget=budget) if endpoint_reference_callback is None else
                          endpoint_reference_callback(context, residuals))
        endpoint_checks[identity] = check
        for name, value in charged.items():
            work[name] = work.get(name, 0)+value
    receipts = list(acquisition_receipts or [])
    if indexed is not None:
        for receipt in receipts:
            for name, value in receipt.get("work", {}).items():
                if name in ("forced_solves", "requested_forced_steps"):
                    work[name] = work.get(name, 0)+value
        work["reused_numerical_acquisition"] = True
        work["acquisition_receipt_count"] = len(receipts)
    path_work = [{"parent_id": identity[0], "query_id": identity[1], **context["path"].work()} for identity, context in contexts.items()]
    original_path_work = ([dict(row) for receipt in receipts for row in receipt.get("path_work", [])]
                          if indexed is not None else path_work)
    return dict(probe_plan_sha256=plan_hash, probe_plan=plan_rows,
                measurement_rows=[_clean_metadata(r) for r in records],
                endpoint_quadrature_checks=[dict(parent_id=identity[0], query_id=identity[1], check=check) for identity, check in endpoint_checks.items()],
                work=work, path_work=path_work, acquisition_path_work=original_path_work,
                acquisition_path_work_status="RECORDED" if indexed is None or len(original_path_work) == len(contexts) else "PARTIAL_OR_UNAVAILABLE",
                acquisition_receipts=receipts,
                coverage=dict(requested_queries=requested, requested_probes=len(records),
                              resolved_probes=sum(r["status"] == "RESOLVED" for r in records),
                              nontrivial_probes=sum(r["nontrivial"] for r in records), independent_parent_count=len(seen_parents)),
                _actions=records, _contexts=contexts, _endpoint_checks=endpoint_checks, _config=cfg)


def run_experiment(samples, config=None, *, action_callback=None, endpoint_reference_callback=None,
                   precomputed_actions=None, precomputed_endpoint_checks=None,
                   acquisition_receipts=None, budget=None):
    """Train-only factors and held-out evaluation, optionally from sealed labels.

    Private replay tensors live in ``_actions``/``_factors``; public rows are
    JSON-ready. This function records evidence and leaves campaign decisions to
    the controller, while retaining the numerical acquisition receipts.
    """
    collected = collect_experiment_actions(samples, config, action_callback=action_callback,
                  endpoint_reference_callback=endpoint_reference_callback, precomputed_actions=precomputed_actions,
                  precomputed_endpoint_checks=precomputed_endpoint_checks, acquisition_receipts=acquisition_receipts, budget=budget)
    cfg, records, contexts, endpoint_checks = (collected[k] for k in ("_config", "_actions", "_contexts", "_endpoint_checks"))
    work = dict(collected["work"])
    ranks = tuple(cfg.get("ranks", (2, 4, 8)))
    factors, metadata, action_rows, endpoint_rows = {}, [], [], []
    keys = sorted({rec["context_key"] for rec in records})
    direction_summaries = []
    for key in keys:
        group = [rec for rec in records if rec["context_key"] == key]
        geom = Geometry(key[1], key[2]); target = key[0]
        diagnostic_tick = time.perf_counter()
        direction_summaries.extend(_direction_diagnostics(group, geom, target, cfg.get("rcond", 1e-10)))
        work["source_direction_diagnostic_seconds"] = work.get("source_direction_diagnostic_seconds", 0.)+time.perf_counter()-diagnostic_tick
        train = [rec for rec in group if rec["split"] == "train" and rec["probe_role"] == "fit" and rec["status"] == "RESOLVED"]
        for rank in ranks:
            menus = [(kind, train, f"shared_{kind}_rank{rank}", "reusable_operator")
                     for kind in ("source_energy_pod", "response_action_pod", "endpoint_weighted_pod")]
            menus.append(("response_action_pod", [r for r in train if r["probe_kind"] == "residual"],
                          f"task_residual_response_pod_rank{rank}", "task_residual_family"))
            for kind, fitting, name, compression_object in menus:
                if not fitting:
                    metadata.append(dict(method=name, context=list(key), status="UNRESOLVED", reason="No accepted training action labels"))
                    continue
                model = fit_source_map(torch.cat([r["source"] for r in fitting]), torch.cat([r["action"] for r in fitting]), geom, target, rank,
                                       basis_kind=kind, weights=_weights(fitting, endpoint=kind == "endpoint_weighted_pod"),
                                       ridge=cfg.get("ridge", 0.), rcond=cfg.get("rcond", 1e-10),
                                       time_pairs=[(r["source_time"], r["horizon"]) for r in fitting],
                                       temporal_channels=cfg.get("temporal_channels", 4), t_ref=cfg.get("t_ref", .12))
                model.metadata["compression_object"] = compression_object
                work["factor_fit_seconds"] = work.get("factor_fit_seconds", 0.)+model.metadata["fit_seconds"]
                work["factor_svd_calls"] = work.get("factor_svd_calls", 0)+model.metadata["svd_calls"]
                factors[(key, name)] = model
                metadata.append(dict(method=name, context=list(key), status="FITTED", **model.metadata))
            # Same-state representation oracles fit only that state's fit directions.
            for parent in sorted({r["parent_id"] for r in group}):
                local = [r for r in group if r["parent_id"] == parent and r["probe_role"] == "fit" and r["status"] == "RESOLVED"]
                if local:
                    name = f"state_oracle_response_pod_rank{rank}"
                    model = fit_source_map(torch.cat([r["source"] for r in local]), torch.cat([r["action"] for r in local]), geom, target, rank,
                                           split=local[0]["split"], weights=_weights(local), oracle=True,
                                           ridge=cfg.get("ridge", 0.), rcond=cfg.get("rcond", 1e-10),
                                           time_pairs=[(r["source_time"], r["horizon"]) for r in local],
                                           temporal_channels=cfg.get("temporal_channels", 4), t_ref=cfg.get("t_ref", .12))
                    model.metadata["compression_object"] = "reusable_operator_state_oracle"
                    work["factor_fit_seconds"] = work.get("factor_fit_seconds", 0.)+model.metadata["fit_seconds"]
                    work["factor_svd_calls"] = work.get("factor_svd_calls", 0)+model.metadata["svd_calls"]
                    factors[(key, name, parent)] = model
                    metadata.append(dict(method=name, parent_id=parent, context=list(key), status="FITTED", **model.metadata))
        methods = sorted({identity[1] for identity in factors if identity[0] == key})
        for name in methods:
            for row in group:
                model = factors.get((key, name, row["parent_id"])) or factors.get((key, name))
                if model is None:
                    continue
                public = _clean_metadata(row)
                if row["status"] == "RESOLVED":
                    prediction = model.apply(row["source"], row["source_time"], row["horizon"])
                    deployed = model.apply(row["source"], row["source_time"], row["horizon"], deployment=True)
                    error = field_metrics(prediction-row["action"], geom, target)
                    public.update(error=error, relative_error=error["rms"]/row["action_metrics"]["rms"] if row["nontrivial"] else None,
                                  deployment_error=field_metrics(deployed.double()-row["action"], geom, target),
                                  deployment_sensitivity=field_metrics(deployed.double()-prediction, geom, target))
                public.update(method=name, deployable=model.metadata["deployable"],
                              compression_object=model.metadata["compression_object"],
                              evaluation_scope="heldout_parent" if row["split"] != "train" else row["probe_role"],
                              requested_rank=model.metadata["requested_rank"], effective_rank=model.metadata["retained_rank"])
                action_rows.append(public)
            for parent, query in sorted({(r["parent_id"], r["query_id"]) for r in group}):
                residuals = [r for r in group if r["parent_id"] == parent and r["query_id"] == query and r["probe_kind"] == "residual"]
                context = contexts[(parent, query)]
                model = factors.get((key, name, parent)) or factors.get((key, name))
                if model is None:
                    continue
                public = dict(parent_id=parent, query_id=query, split=context["record"]["split"], target_kind=target,
                              horizon=float(context["record"]["horizon"]), method=name, deployable=model.metadata["deployable"],
                              requested_rank=model.metadata["requested_rank"], effective_rank=model.metadata["retained_rank"],
                              compression_object=model.metadata["compression_object"],
                              source_nodes=len(residuals), source_quadrature="declared Gauss-Legendre physical residual action sum; same nodes for all methods")
                if not residuals or any(r["status"] != "RESOLVED" for r in residuals) or "target" not in context["sample"]:
                    public.update(status="UNRESOLVED", reason="Missing/unresolved residual response actions or physical endpoint teacher")
                else:
                    anchor = context["path"].evaluate(public["horizon"])["state"]
                    work["residual_evaluations"] += 1
                    mean_correction = -sum(r["quadrature_weight"]*r["mean"] for r in residuals)
                    oracle_correction = -sum(r["quadrature_weight"]*r["tangent"] for r in residuals)
                    compressed_correction = mean_correction-sum(r["quadrature_weight"]*model.apply(r["source"], r["source_time"], r["horizon"]) for r in residuals)
                    deployed_correction = mean_correction.float()-sum(r["quadrature_weight"]*model.apply(r["source"], r["source_time"], r["horizon"], deployment=True) for r in residuals)
                    teacher = context["sample"]["target"].double()
                    eg = field_metrics(anchor+mean_correction-teacher, geom, target)
                    eu = field_metrics(anchor+oracle_correction-teacher, geom, target)
                    el = field_metrics(anchor+compressed_correction-teacher, geom, target)
                    gain = eg["rms"]-eu["rms"]
                    uncertainty = sum(abs(r["quadrature_weight"])*float(r.get("uncertainty", math.inf)) for r in residuals)
                    deployed_anchor = context["sample"].get("deployed_anchor32", context["sample"].get("anchor_fp32"))
                    if deployed_anchor is None:
                        deployed_anchor = n.anchor(context["sample"]["u0"].float(), public["horizon"], context["eq"], geom, target,
                                                   name=context["path"].anchor_name, reaction_substeps=cfg.get("reaction_substeps", 8))
                        work["deployed_anchor_evaluations"] = work.get("deployed_anchor_evaluations", 0)+1
                    if deployed_anchor.dtype != torch.float32:
                        raise ValueError("Actual deployment anchor must be FP32")
                    deployed_prediction = deployed_anchor+deployed_correction
                    quadrature_check = endpoint_checks[(parent, query)]
                    quadrature_uncertainty = quadrature_check.get("uncertainty_rms")
                    uncertainty = uncertainty+float(quadrature_uncertainty) if quadrature_uncertainty is not None and math.isfinite(float(quadrature_uncertainty)) else None
                    public.update(status=quadrature_check["status"] if uncertainty is not None else "UNRESOLVED", mean_error=eg, oracle_error=eu, compressed_error=el,
                                  propagated_residual_error=field_metrics(compressed_correction-oracle_correction, geom, target),
                                  deployment_sensitivity=field_metrics(deployed_correction.double()-compressed_correction, geom, target),
                                  arithmetic_bridge=field_metrics(anchor-deployed_anchor.double(), geom, target),
                                  final_fp32_addition_error=field_metrics(deployed_prediction.double()-(deployed_anchor.double()+deployed_correction.double()), geom, target),
                                  deployed_endpoint_error=field_metrics(deployed_prediction.double()-teacher, geom, target),
                                  oracle_gain=gain, retained_gain=(eg["rms"]-el["rms"])/gain if uncertainty is not None and gain > 10*uncertainty else None,
                                  uncertainty=uncertainty, quadrature_check=quadrature_check,
                                  uncertainty_scope="action/source-quadrature/forced integration refinement; path/teacher uncertainty additionally required by controller",
                                  endpoint_precision="FP64 physical mechanism; FP32 compressed correction sensitivity separately measured")
                    work["endpoint_residual_nodes"] += len(residuals)
                endpoint_rows.append(public)
    path_work = [{"parent_id": identity[0], "query_id": identity[1], **context["path"].work()} for identity, context in contexts.items()]
    previous_work = {(row["parent_id"], row["query_id"]): row for row in collected["path_work"]}
    counter_keys = ("path_queries", "anchor_evaluations", "derivative_evaluations", "residual_evaluations", "mean_queries", "cache_hits", "cached_nodes")
    aggregation_path_work = []
    for row in path_work:
        before = previous_work.get((row["parent_id"], row["query_id"]), {})
        delta = dict(row)
        for key in counter_keys:
            delta[key] = row.get(key, 0)-before.get(key, 0)
        delta["scope"] = "New aggregation/evaluation path work only; acquisition retained separately"
        aggregation_path_work.append(delta)
    unavailable = [dict(method=name, status="NOT_APPLICABLE", reason="No learned/frozen-encoder factor producer supplied to compression; Stage C must acquire/fit its own labels")
                   for name in ("frozen_random_encoder_readout", "learned_encoder")]
    return dict(probe_plan_sha256=collected["probe_plan_sha256"], probe_plan=collected["probe_plan"], measurement_rows=[_clean_metadata(r) for r in records],
                direction_summaries=direction_summaries, action_rows=action_rows, endpoint_rows=endpoint_rows,
                method_rows=metadata+unavailable, coverage=dict(requested_queries=collected["coverage"]["requested_queries"], requested_probes=len(records),
                    resolved_probes=sum(r["status"] == "RESOLVED" for r in records),
                    nontrivial_probes=sum(r["nontrivial"] for r in records), independent_parent_count=collected["coverage"]["independent_parent_count"],
                    untested_directions="All directions outside the prospectively declared finite Fourier/random/residual menu; no uniform operator claim"),
                work=work, path_work=path_work, aggregation_path_work=aggregation_path_work,
                acquisition_path_work=collected["acquisition_path_work"],
                acquisition_path_work_status=collected["acquisition_path_work_status"],
                acquisition_receipts=collected["acquisition_receipts"],
                endpoint_quadrature_checks=collected["endpoint_quadrature_checks"], _actions=records, _factors=factors)
