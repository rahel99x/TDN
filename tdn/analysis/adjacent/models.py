"""Adjacent-study interaction-origin channels, without changing TDN models.

LL/LH/HH label which input scales form a signed quadratic interaction. Their
axis is not spatial and is never transformed. All channels share one physical
background and retain the zero-subtracted finite-time DF correction. Bounded
mixing preserves physical nulls; it does not establish stability or positivity.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn

from tdn.analysis.frontier import numerics
from tdn.analysis.portfolio.models import MODEL_SPECS as PORTFOLIO_SPECS
from tdn.analysis.portfolio.models import make_model as portfolio_model, physical_features
from tdn.analysis.roadmap.numerics import background_response, lowpass, quadrature
from tdn.numerics.operators import broadcast_h

CHANNEL_NAMES = ("LL", "LH", "HH")
CHANNEL_FAMILIES = ("channel_fixed", "channel_global", "channel_affine", "channel_neural", "feature_capacity")
MODEL_SPECS = {key: dict(value) for key, value in PORTFOLIO_SPECS.items()}
for _name, _fit, _role in (("channel_fixed", "frozen", "Analytic control (Ours decomposition)"),
        ("channel_global", "channel_lstsq", "Ours fitted numerical"),
        ("channel_affine", "channel_affine_lstsq", "Ours fitted numerical"),
        ("channel_neural", "gradient", "Ours"), ("feature_capacity", "gradient", "Ours capacity control")):
    MODEL_SPECS[_name] = dict(role=_role, ownership=_role, fit=_fit, trainable=_fit == "gradient",
        nodes=2, compression="output", path="adjacent", description=(
        "Ordinary pointwise feature partition of the normalized correction" if _name == "feature_capacity"
        else "Fixed-background input-origin LL/LH/HH physical interaction channels"))
FAMILIES = tuple(MODEL_SPECS)
TRAINABLE_FAMILIES = tuple(k for k, v in MODEL_SPECS.items() if v["trainable"])


def gain_bounds(family):
    """Actual effective multiplicative bounds, including family differences.

    Endpoints for tanh families are the closure of the attainable open range.
    None means that no single bounded multiplicative-gain description applies.
    """
    if family not in MODEL_SPECS:
        raise ValueError(f"Unknown family: {family}")
    if family == "historical_half":
        return (.5, .5)
    if family in ("fno_small", "fno_standard", "direct_fno", "analytic_quad_cubic", "df", "rf", "etdrk4"):
        return None
    if family == "residual_quad2":
        return (.75, 1.25)
    if family == "band_gain":
        return (.5, 1.5)
    if family in ("channel_fixed", "quad2_fixed", "quad2_nodes", "quad2_input", "quad2_full", "quad4_fixed", "quad4_full"):
        return (1., 1.)
    return (.25, 1.75)


def _directions(u, split_modes):
    if type(split_modes) is not int or split_modes < 0:
        raise ValueError("Input split cutoff must be a nonnegative integer")
    mean = u.mean((-2, -1), keepdim=True)
    v = u - mean
    low = lowpass(v, split_modes)
    return mean, v, low, v - low


def _background(background, v):
    value = torch.as_tensor(background, dtype=v.dtype, device=v.device)
    if value.ndim == 0:
        value = value.expand(v.shape[0], 1, 1, 1)
    if tuple(value.shape) != (v.shape[0], 1, 1, 1) or not bool(torch.isfinite(value).all()):
        raise ValueError("One finite, spatially constant background per parent required")
    # Avoid a formal divergent Jacobian times an identically zero direction.
    null = (v == 0).flatten(2).all(-1)[..., None, None]
    return torch.where(null, torch.full_like(value, .5), value)


def fixed_background_quadratic(v, background, h, equation, geometry, *, track="discrete", nodes=2):
    """Transparent Q_{m,h}(v); does not recenter v or recompute m.

    This is a homogeneous quadratic form at a fixed background. Calling a
    state-based correction separately on each channel would change m and is
    deliberately avoided. Inputs are scalar fields [batch,1,nx,ny].
    """
    step = numerics.validate(v, h, geometry, track)
    if geometry.ndim != 2:
        raise ValueError("Interaction channels require a two-dimensional grid")
    positions, weights = quadrature(nodes)
    mean = _background(background, v)
    if equation.kappa == 0 or equation.reaction_rate == 0:
        return v * 0 + step * 0
    _, qh, _ = background_response(mean, step, equation.reaction_rate)
    heat = lambda z, dt: numerics.heat_step(z, dt, equation, geometry, track)
    midpoint = heat(v, step / 2)
    split = heat(numerics.product(midpoint, midpoint, track), step / 2)
    result = torch.zeros_like(v)
    for position, weight in zip(positions, weights):
        s = step * position
        transported = heat(v, s)
        propagated = heat(numerics.product(transported, transported, track), step - s)
        _, qs, _ = background_response(mean, s, equation.reaction_rate)
        result = result + weight * qs * (propagated - split)
    result = -equation.reaction_rate * qh * step * result
    constant = (v == v[..., :1, :1]).flatten(2).all(-1)[..., None, None]
    return torch.where(constant | (step == 0), torch.zeros_like(result), result)


def interaction_channels(u, h, equation, geometry, *, track="discrete", split_modes=2,
                         nodes=2, output_modes=None, implementation="shared"):
    """Return [batch,3,nx,ny] channels LL, LH, HH, with signed cross term.

    ``reference`` evaluates Q(L), Q(H), Q(L+H)-Q(L)-Q(H) at one background.
    ``shared`` transports L/H together and forms (L², 2LH, H²) with the
    track's product, retaining the midpoint subtraction for each product.
    Batched spatial FFTs operate only on x/y, never on the channel axis.
    Output compression is optional and applied after forming interactions.
    """
    step = numerics.validate(u, h, geometry, track)
    if geometry.ndim != 2:
        raise ValueError("Interaction channels require a two-dimensional grid")
    if implementation not in ("shared", "reference"):
        raise ValueError("Implementation must be shared or reference")
    positions, weights = quadrature(nodes)
    mean, v, lo, hi = _directions(u, split_modes)
    if implementation == "reference":
        fun = lambda z: fixed_background_quadratic(z, mean, step, equation, geometry, track=track, nodes=nodes)
        ll, hh, whole = fun(lo), fun(hi), fun(v)
        result = torch.cat((ll, whole - ll - hh, hh), 1)
    else:
        if equation.kappa == 0 or equation.reaction_rate == 0:
            result = (u * 0 + step * 0).expand(-1, 3, -1, -1)
        else:
            mean = _background(mean, v)
            symbol = numerics.diffusion_symbol(u, equation, geometry, track)
            # Shared coefficient and FFT launches; no pairwise-mode tensor.
            pair = torch.cat((lo, hi), 1)
            heat = lambda z, dt: numerics.apply_multiplier(z, torch.exp(dt * symbol))
            def products(z):
                left, right = z[:, :1], z[:, 1:]
                return torch.cat((numerics.product(left, left, track),
                                  2 * numerics.product(left, right, track),
                                  numerics.product(right, right, track)), 1)
            midpoint = heat(pair, step / 2)
            split = heat(products(midpoint), step / 2)
            result = torch.zeros_like(split)
            for position, weight in zip(positions, weights):
                s = step * position
                _, qs, _ = background_response(mean, s, equation.reaction_rate)
                result = result + weight * qs * (heat(products(heat(pair, s)), step - s) - split)
            _, qh, _ = background_response(mean, step, equation.reaction_rate)
            result = -equation.reaction_rate * qh * step * result
    if output_modes is not None:
        result = lowpass(result, output_modes)
    constant = (u == u[..., :1, :1]).flatten(2).all(-1)[..., None, None]
    return torch.where(constant | (step == 0), torch.zeros_like(result), result)


class InteractionModel(nn.Module):
    def __init__(self, family, track="discrete", *, modes=4, split_modes=2, quad_nodes=2,
                 hidden=8, reaction_substeps=4, output_compression=True, cache_entries=32,
                 width=None, depth=None, t_ref=.1, cubic_nodes=4):
        super().__init__()
        if family not in CHANNEL_FAMILIES or track not in numerics.TRACKS:
            raise ValueError("Unknown adjacent channel model or target track")
        if any(type(n) is not int or n < 1 for n in (modes, quad_nodes, hidden, reaction_substeps)):
            raise ValueError("Positive integer dimensions, nodes and subdivisions required")
        if type(split_modes) is not int or split_modes < 0 or not isinstance(output_compression, bool):
            raise ValueError("Nonnegative input cutoff and boolean output compression required")
        quadrature(quad_nodes)
        self.family, self.track, self.modes = family, track, modes
        self.split_modes, self.quad_nodes = split_modes, quad_nodes
        self.hidden, self.reaction_substeps = hidden, reaction_substeps
        self.output_compression = output_compression
        self.spec = dict(MODEL_SPECS[family])
        self.spec.update(nodes=quad_nodes, compression="output" if output_compression else "none")
        self.cache = numerics.CoefficientCache(cache_entries)
        if family == "channel_global":
            self.register_buffer("gain_logits", torch.zeros(3, dtype=torch.float64))
        if family == "channel_affine":
            self.register_buffer("affine_coefficients", torch.zeros(4, 3, dtype=torch.float64))
        if family in ("channel_neural", "feature_capacity"):
            self.conditioner = nn.Sequential(nn.Linear(3, hidden), nn.SiLU(), nn.Linear(hidden, 3))
            nn.init.zeros_(self.conditioner[-1].weight)
            nn.init.zeros_(self.conditioner[-1].bias)

    def _apply(self, fn, recurse=True):
        self.clear_cache()
        return super()._apply(fn, recurse=recurse)

    def clear_cache(self):
        self.cache.clear()

    @property
    def cache_metadata(self):
        return self.cache.metadata

    def channels(self, u, h, equation, geometry):
        if self.family != "feature_capacity":
            return interaction_channels(u, h, equation, geometry, track=self.track,
                split_modes=self.split_modes, nodes=self.quad_nodes)
        mean = u.mean((-2, -1), keepdim=True)
        q = fixed_background_quadratic(u - mean, mean, h, equation, geometry,
                                       track=self.track, nodes=self.quad_nodes)
        # Ordinary local features, not input-origin interaction labels. The
        # partition sums to one and preserves the analytic initialization.
        # Fixed dimensionless logistic-state scaling avoids a singular std.
        z = u - mean
        partition = torch.softmax(torch.cat((z, -z, z.square()), 1), dim=1)
        return q * partition

    def response(self, u, h, equation, geometry):
        step = numerics.validate(u, h, geometry, self.track)
        features = (physical_features(u, step, equation, geometry, self.track)
                    if self.family not in ("channel_fixed", "channel_global") else
                    torch.empty((u.shape[0], 0), dtype=u.dtype, device=u.device))
        if self.family == "channel_global":
            latent = self.gain_logits.to(u.dtype).expand(u.shape[0], -1)
        elif self.family == "channel_affine":
            design = torch.cat((torch.ones_like(features[:, :1]), features), 1)
            latent = design @ self.affine_coefficients.to(u.dtype)
        elif hasattr(self, "conditioner"):
            latent = self.conditioner(features.to(self.conditioner[0].weight.dtype)).to(u.dtype)
        else:
            latent = torch.zeros((u.shape[0], 3), dtype=u.dtype, device=u.device)
        return dict(gain=1 + .75 * torch.tanh(latent), features=features,
                    channels=CHANNEL_NAMES if self.family != "feature_capacity" else ("feature+", "feature-", "feature-square"))

    def correction_components(self, u, h, equation, geometry):
        step = numerics.validate(u, h, geometry, self.track)
        base = numerics.df_step(u, step, equation, geometry, self.track,
                               reaction_substeps=self.reaction_substeps, cache=self.cache)
        channels = self.channels(u, step, equation, geometry)
        if self.family == "channel_fixed":
            increment = channels.sum(1, keepdim=True)
        else:
            gains = self.response(u, step, equation, geometry)["gain"]
            increment = (channels * gains[..., None, None]).sum(1, keepdim=True)
        if self.output_compression:
            increment = lowpass(increment, self.modes)
        return dict(base=base, increment=increment)

    def forward(self, u, h, equation, geometry):
        pieces = self.correction_components(u, h, equation, geometry)
        return torch.where(broadcast_h(h, u) == 0, u, pieces["base"] + pieces["increment"])

    @torch.no_grad()
    def fit_least_squares(self, samples: Iterable, *, ridge=1e-8, peak_weight=.1,
                          budget=None, max_iterations=100):
        """Non-neural fit using training samples only and actual bounded gains.

        Scale-equilibrated LS initializes a MSE+maximum-square objective fit.
        Conditioning/rank refer to the sampled response design, not a claim of
        globally identifiable coefficients. Fitting transfers are offline cost.
        """
        import numpy as np
        from scipy.optimize import minimize
        if self.family not in ("channel_global", "channel_affine"):
            raise ValueError("This fitter applies to global or affine channel controls")
        if not math.isfinite(ridge) or ridge < 0 or not math.isfinite(peak_weight) or peak_weight < 0:
            raise ValueError("Nonnegative finite ridge and peak weight required")
        if type(max_iterations) is not int or max_iterations < 1:
            raise ValueError("Positive fitting iteration limit required")
        feature_count = 1 if self.family == "channel_global" else 4
        fields, design_rows, targets = [], [], []
        for sample in samples:
            if budget is not None:
                budget.check()
            u, h, equation, geometry, target = sample[:5]
            weight = 1. if len(sample) == 5 else float(sample[5])
            if target.shape != u.shape or not math.isfinite(weight) or weight <= 0:
                raise ValueError("Matching target and positive finite weight required")
            step = numerics.validate(u, h, geometry, self.track)
            base = numerics.df_step(u, step, equation, geometry, self.track,
                reaction_substeps=self.reaction_substeps, cache=self.cache)
            channels = self.channels(u, step, equation, geometry)
            if self.output_compression:
                channels = lowpass(channels, self.modes)
            features = physical_features(u, step, equation, geometry, self.track)
            features = torch.cat((torch.ones_like(features[:, :1]), features), 1)[:, :feature_count]
            for index in range(u.shape[0]):
                f = features[index].double().cpu().numpy()
                q = channels[index].flatten(1).T.double().cpu().numpy()
                d = (target[index] - base[index]).flatten().double().cpu().numpy()
                if not all(np.isfinite(item).all() for item in (f, q, d)):
                    raise ValueError("Finite training features, responses and targets required")
                fields.append((f, q, d, weight))
                scale = math.sqrt(weight / q.shape[0])
                design_rows.append((q[:, None, :] * f[None, :, None]).reshape(q.shape[0], -1) * scale)
                targets.append((d - q.sum(1)) * scale)
        if not fields:
            raise ValueError("At least one training sample required")
        design, target = np.concatenate(design_rows), np.concatenate(targets)
        scales = np.maximum(np.linalg.norm(design, axis=0), np.finfo(float).eps)
        balanced = design / scales
        singular = np.linalg.svd(balanced, compute_uv=False)
        rank = int(np.linalg.matrix_rank(balanced))
        augmented = np.vstack((balanced, math.sqrt(ridge) * np.eye(design.shape[1])))
        coefficients = np.linalg.lstsq(augmented, np.concatenate((target, np.zeros(design.shape[1]))), rcond=None)[0] / scales
        initializer = (coefficients / .75).reshape(feature_count, 3)
        evaluations = 0
        def objective(flat):
            nonlocal evaluations
            if budget is not None:
                budget.check()
            evaluations += 1
            beta = flat.reshape(feature_count, 3)
            total, gradient = 0., np.zeros_like(beta)
            for f, q, d, weight in fields:
                response = np.tanh(f @ beta)
                gain = 1 + .75 * response
                residual = q @ gain - d
                maximum = int(np.argmax(residual ** 2))
                total += weight * (np.mean(residual ** 2) + peak_weight * residual[maximum] ** 2)
                dg = 2 * weight * (q.T @ residual / len(residual) + peak_weight * q[maximum] * residual[maximum])
                gradient += np.outer(f, dg * .75 * (1 - response ** 2))
            return total / len(fields), gradient.flatten() / len(fields)
        zeros = np.zeros_like(initializer).flatten()
        initial = objective(zeros)[0]
        normalization = max(initial, np.finfo(float).tiny)
        result = minimize(lambda z: tuple(v / normalization for v in objective(z)), initializer.flatten(),
                          jac=True, method="L-BFGS-B", options={"maxiter": max_iterations, "ftol": 1e-13, "gtol": 1e-10})
        candidates = (zeros, initializer.flatten(), result.x)
        selected = min(candidates, key=lambda z: objective(z)[0])
        destination = self.gain_logits if self.family == "channel_global" else self.affine_coefficients
        destination.copy_(torch.as_tensor(selected.reshape(destination.shape), dtype=destination.dtype, device=destination.device))
        return dict(fit="bounded_channel_matched_MSE_plus_peak", samples=len(fields), columns=design.shape[1], rank=rank,
            identifiable=rank == design.shape[1], singular_values=singular.tolist(),
            condition_number=float(singular[0] / singular[-1]) if len(singular) and singular[-1] > 0 else None,
            initial_weighted_sse=initial, fitted_weighted_sse=objective(selected)[0],
            ridge_initializer_only=ridge, peak_weight=peak_weight, gain_bounds=list(gain_bounds(self.family)),
            optimizer_success=bool(result.success), optimizer_message=str(result.message), evaluations=evaluations,
            matched_objective=True, reference_inputs_during_inference=False)

    def parameter_report(self, u=None, h=None, equation=None, geometry=None):
        result = dict(family=self.family, trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
            stored_parameters=sum(p.numel() for p in self.parameters()),
            fitted_buffer_coefficients=3 if self.family == "channel_global" else 12 if self.family == "channel_affine" else 0,
            gain_bounds=list(gain_bounds(self.family)), state_dict={name: value.detach().cpu().tolist() for name, value in self.state_dict().items()},
            identifiability="Inspect sampled channel rank; different gains can represent the same correction when channels are collinear")
        if u is not None:
            result["response"] = {key: value.detach().cpu().tolist() if torch.is_tensor(value) else value
                                  for key, value in self.response(u, h, equation, geometry).items()}
        return result

    def architecture_metadata(self):
        return dict(family=self.family, track=self.track, **self.spec,
            parameters=sum(p.numel() for p in self.parameters()),
            trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
            fitted_buffer_coefficients=self.parameter_report()["fitted_buffer_coefficients"],
            modes=self.modes, split_modes=self.split_modes, physical_core="diffusion_first_strang",
            reaction_target="nodal_logistic" if self.track == "discrete" else "quadratic_dealiased_galerkin",
            physical_background="one fixed spatial mean for all channels", channel_axis="information, not space; never FFT transformed",
            channels=CHANNEL_NAMES if self.family != "feature_capacity" else ("feature+", "feature-", "feature-square"),
            state_inputs=[] if self.family in ("channel_fixed", "channel_global") else
                ["mean", "log1p(h*r)", "log1p(h*active_diffusion_rate)"],
            ordinary_feature_partition="softmax(v,-v,v^2) times full quadratic correction" if self.family == "feature_capacity" else None,
            gain_parameterization="1+0.75*tanh(response); initialized to one", gain_bounds=list(gain_bounds(self.family)),
            shared_transports=self.family != "feature_capacity", explicit_pair_tensor=False, output_compression=self.output_compression,
            published_FNO_reproduction=False, reference_or_future_inputs=False, parameter_matching_claim=False,
            compute_matching_claim=False, clipping=False, stability_certificate=False, invariant_interval_guarantee=False,
            exact_correction_nulls=["constant", "reaction=0", "diffusion=0", "time=0"],
            actual_work="full DF, split FFT, transported signed products, optional features and mixing, output projection")


def make_model(family, track="discrete", config=None):
    config = dict(config or {})
    if family in CHANNEL_FAMILIES:
        return InteractionModel(family, track, **config)
    # Shared experiment config explicitly includes adjacent-only controls;
    # these never alter the historical portfolio implementation.
    config.pop("split_modes", None)
    config.pop("output_compression", None)
    return portfolio_model(family, track, config)
