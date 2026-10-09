"""Controlled normalized interaction rules and small physical conditioners.

The historical half-normalized arm is imported unchanged. New attribution
arms share the same DF backbone, physical quadratic formula, input and output
operators; only explicitly named treatments differ. Normalized quadrature
weights sum to one *before* an optional learned gain. Neither bounded gains
nor physical nulls imply stability, positivity, or a global error certificate.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn

from tdn.analysis.frontier import numerics
from tdn.analysis.frontier.models import make_model as historical_model
from tdn.analysis.roadmap.numerics import lowpass, quadrature, quadratic_df_defect
from tdn.numerics.operators import broadcast_h, check_shape
from tdn.numerics.subflows import reaction_step


def _spec(role, *, fit="frozen", nodes=0, compression="output", path="A", description=""):
    return dict(role=role, ownership=role, fit=fit, trainable=fit == "gradient", nodes=nodes,
                compression=compression, path=path, description=description)


MODEL_SPECS = {
    "historical_half": _spec("Ours control", nodes=2, description="Unchanged historical 0.5-total-weight frozen rank1"),
    "quad2_fixed": _spec("Analytic control", nodes=2, description="Normalized fixed GL2; output cutoff"),
    "quad2_amplitude": _spec("Ours fitted numerical", fit="amplitude_lstsq", nodes=2, description="One bounded least-squares amplitude; fixed GL2 nodes"),
    "quad2_nodes": _spec("Ours", fit="gradient", nodes=2, description="One fitted symmetric node pair; gain one"),
    "quad2_joint": _spec("Ours", fit="gradient", nodes=2, description="Global node and amplitude fit"),
    "quad2_linear": _spec("Ours fitted numerical", fit="linear_lstsq", nodes=2, description="Non-neural bounded affine-feature gain, LS-initialized matched-loss fit"),
    "quad2_conditioned": _spec("Ours", fit="gradient", nodes=2, description="Node/amplitude plus three-feature scalar conditioner"),
    "quad2_input": _spec("Analytic control", nodes=2, compression="input_and_output", description="Pure premature-input-compression ablation"),
    "quad2_full": _spec("Analytic control", nodes=2, compression="none", description="Normalized GL2 without output cutoff"),
    "quad2_conditioned_full": _spec("Ours", fit="gradient", nodes=2, compression="none", description="Conditioned GL2 without output cutoff"),
    "quad4_fixed": _spec("Analytic control", nodes=4, description="Normalized fixed GL4; output cutoff"),
    "quad4_full": _spec("Analytic control", nodes=4, compression="none", description="Normalized GL4 without output cutoff"),
    "quad4_conditioned": _spec("Ours", fit="gradient", nodes=4, description="Two symmetric pairs, fixed normalized GL4 weights; conditioned gain"),
    "analytic_quad_cubic": _spec("Analytic control", nodes=4, compression="none", description="Existing full GL4 quadratic plus GL4 cubic formula"),
    "df": _spec("Theirs", compression="none", description="Diffusion-first Strang"),
    "rf": _spec("Theirs", compression="none", description="Reaction-first Strang, same track and substeps"),
    "etdrk4": _spec("Theirs", compression="none", description="Cox-Matthews ETDRK4"),
    "fno_small": _spec("Theirs", fit="gradient", compression="spectral_and_full_local", description="Local hybrid FNO adaptation, not published reproduction"),
    "fno_standard": _spec("Theirs", fit="gradient", compression="spectral_and_full_local", description="Larger local hybrid FNO adaptation, not published reproduction"),
    "direct_fno": _spec("Theirs", fit="gradient", compression="spectral_and_full_local", description="Direct local neural FNO control"),
    "residual_quad2": _spec("Ours", fit="gradient", nodes=2, path="B", description="Bounded small residual gain atop fixed normalized GL2"),
    "conditioned_rich": _spec("Ours", fit="gradient", nodes=2, path="B", description="Seven nondimensional global features, scalar gain"),
    "band_gain": _spec("Ours", fit="gradient", nodes=2, path="B", description="Three conditioned output-band gains atop fixed normalized GL2"),
}
FAMILIES = tuple(MODEL_SPECS)
TRAINABLE_FAMILIES = tuple(k for k, v in MODEL_SPECS.items() if v["trainable"])
FIT_FAMILIES = tuple(k for k, v in MODEL_SPECS.items() if v["fit"] != "frozen")
PHYSICAL_FAMILIES = tuple(k for k, v in MODEL_SPECS.items() if v["nodes"])
_DELEGATED = {"historical_half": "rank1_frozen", "analytic_quad_cubic": "analytic_quad_cubic",
              "df": "df", "etdrk4": "etdrk4", "fno_small": "fno_small",
              "fno_standard": "fno_standard", "direct_fno": "direct_fno"}
_CONDITIONED = {"quad2_conditioned", "quad2_conditioned_full", "quad4_conditioned",
                "residual_quad2", "conditioned_rich", "band_gain"}
_LEARN_NODES = {"quad2_nodes", "quad2_joint", "quad2_conditioned", "quad2_conditioned_full",
                "quad4_conditioned", "conditioned_rich"}
_LEARN_AMPLITUDE = {"quad2_amplitude", "quad2_joint", "quad2_conditioned",
                    "quad2_conditioned_full", "quad4_conditioned", "conditioned_rich"}


def physical_features(u, step, equation, geometry, track, *, rich=False):
    """Dimensionless per-field statistics; no reference or future state input.

    Rich features add variance, absolute centered amplitude, spectral rate
    spread and nonlinear strength. State is dimensionless logistic density.
    Active diffusion is measured using the declared spatial generator, not a
    substituted continuum symbol. The additional multiplier is charged work.
    """
    axes = tuple(range(2, u.ndim))
    mean = u.mean(axes, keepdim=True)
    centered = u - mean
    variance = centered.square().mean(axes, keepdim=True)
    diffused = numerics.apply_multiplier(centered, numerics.diffusion_symbol(u, equation, geometry, track))
    energy = -(centered * diffused).mean(axes, keepdim=True)
    tiny = torch.finfo(u.dtype).tiny
    active = torch.where(variance > 0, energy / variance.clamp_min(tiny), torch.zeros_like(variance)).clamp_min(0)
    hr = (step * equation.reaction_rate).expand_as(mean)
    items = [mean, torch.log1p(hr), torch.log1p(step * active)]
    if rich:
        # Variance-free constants receive exactly zero high-order statistics.
        spread_squared = (diffused + active * centered).square().mean(axes, keepdim=True)
        spread = torch.sqrt(torch.where(variance > 0, spread_squared / variance.clamp_min(tiny),
                                        torch.zeros_like(variance)).clamp_min(0) + torch.finfo(u.dtype).eps**2)
        items += [variance, centered.abs().mean(axes, keepdim=True), torch.log1p(step * spread),
                  torch.log1p(hr * torch.sqrt(variance + torch.finfo(u.dtype).eps**2))]
    return torch.cat(items, dim=1).flatten(1)


class PortfolioModel(nn.Module):
    """Transparent attribution family with normalized analytic initialization."""
    def __init__(self, family, track="discrete", *, modes=4, width=None, depth=None,
                 t_ref=.1, reaction_substeps=4, quad_nodes=4, cubic_nodes=4,
                 cache_entries=32, hidden=8):
        super().__init__()
        if family not in MODEL_SPECS or track not in numerics.TRACKS:
            raise ValueError("Unknown portfolio family or target track")
        if any(type(x) is not int or x < 1 for x in (modes, reaction_substeps, quad_nodes, cubic_nodes, hidden)):
            raise ValueError("Positive integer model dimensions and subdivisions required")
        if not math.isfinite(t_ref) or t_ref <= 0:
            raise ValueError("Positive finite reference time required")
        self.family, self.track, self.modes = family, track, modes
        self.reaction_substeps, self.hidden = reaction_substeps, hidden
        self.quad_nodes, self.cubic_nodes = quad_nodes, cubic_nodes
        self.spec = MODEL_SPECS[family]
        self.cache = numerics.CoefficientCache(cache_entries)
        if family in _DELEGATED:
            self.delegate = historical_model(_DELEGATED[family], track, dict(modes=modes, width=width,
                depth=depth, t_ref=t_ref, reaction_substeps=reaction_substeps, quad_nodes=quad_nodes,
                cubic_nodes=cubic_nodes, cache_entries=cache_entries))
            # Generic DF rollout fuses adjacent heat subflows via model.cache;
            # keep that cache identical to the delegated one-step cache so
            # metadata, preparation costs and invalidation include both paths.
            self.cache = self.delegate.cache
            return
        if family == "rf":
            return
        count = self.spec["nodes"]
        nodes, weights = quadrature(count)
        # Strictly symmetric nodes retain O(h^3) at a fixed spatial operator.
        # Interior restriction avoids accidentally collapsing to the midpoint.
        left = torch.tensor(nodes[:count // 2], dtype=torch.float64)
        scaled = (left - .005) / .49
        logits = torch.log(scaled / (1 - scaled))
        if family in _LEARN_NODES:
            self.node_logits = nn.Parameter(logits)
        else:
            self.register_buffer("node_logits", logits)
        self.register_buffer("normalized_weights", torch.tensor(weights, dtype=torch.float64))
        if family in _LEARN_AMPLITUDE:
            self.amplitude_logit = nn.Parameter(torch.zeros((), dtype=torch.float64), requires_grad=family != "quad2_amplitude")
        else:
            self.register_buffer("amplitude_logit", torch.zeros((), dtype=torch.float64))
        if family in _CONDITIONED:
            self.conditioner = nn.Sequential(nn.Linear(7 if family == "conditioned_rich" else 3, hidden),
                nn.SiLU(), nn.Linear(hidden, 3 if family == "band_gain" else 1))
            nn.init.zeros_(self.conditioner[-1].weight)
            nn.init.zeros_(self.conditioner[-1].bias)
        if family == "quad2_linear":
            self.register_buffer("linear_coefficients", torch.zeros(4, dtype=torch.float64))

    def _apply(self, fn, recurse=True):
        self.clear_cache()
        return super()._apply(fn, recurse=recurse)

    def clear_cache(self):
        self.cache.clear()
        if hasattr(self, "delegate"):
            self.delegate.clear_cache()

    @property
    def cache_metadata(self):
        return self.delegate.cache_metadata if hasattr(self, "delegate") else self.cache.metadata

    def nodes_weights(self, u):
        left = .005 + .49 * torch.sigmoid(self.node_logits)
        positions = torch.cat((left, 1 - left.flip(0))).to(dtype=u.dtype, device=u.device)
        return positions, self.normalized_weights.to(dtype=u.dtype, device=u.device)

    def _amplitude(self):
        return 1 + .75 * torch.tanh(self.amplitude_logit)

    def _analytic_increment(self, u, step, equation, geometry):
        state = lowpass(u, self.modes) if self.spec["compression"] == "input_and_output" else u
        positions, weights = self.nodes_weights(u)
        answer = quadratic_df_defect(state, step, equation, geometry, target=self.track,
                                    node_positions=positions, node_weights=weights)
        if self.spec["compression"] != "none":
            answer = lowpass(answer, self.modes)
        constant = (state == state[..., :1, :1]).flatten(2).all(-1)[..., None, None]
        return torch.where(constant | (step == 0), torch.zeros_like(answer), answer)

    def response(self, u, h, equation, geometry):
        """Observable effective response for attribution and identifiability.

        Node/amplitude and network outputs are not independently identifiable
        from effective gain alone. Report actual effective weights/gains, not
        parameter count as a claim of identifiable learned degrees of freedom.
        """
        step = numerics.validate(u, h, geometry, self.track)
        if hasattr(self, "delegate") or self.family == "rf":
            return {"kind": "delegated_or_classical"}
        features = physical_features(u, step, equation, geometry, self.track,
                                     rich=self.family == "conditioned_rich")
        gain = self._amplitude().to(u.dtype).expand(u.shape[0], 1)
        if hasattr(self, "conditioner"):
            raw = self.conditioner(features.to(self.conditioner[0].weight.dtype)).to(u.dtype)
            if self.family in ("residual_quad2", "band_gain"):
                scale = .25 if self.family == "residual_quad2" else .5
                gain = 1 + scale * torch.tanh(raw)
            else:
                # Same admissible effective amplitude as amplitude-only and
                # joint controls. A wider scalar range must not masquerade as
                # useful state dependence. Global logit + network bias retain
                # an explicit additive gauge measured in response diagnostics.
                gain = 1 + .75 * torch.tanh(self.amplitude_logit.to(u.dtype) + raw)
        elif self.family == "quad2_linear":
            design = torch.cat((torch.ones_like(features[:, :1]), features), 1)
            gain = 1 + .75 * torch.tanh(design @ self.linear_coefficients.to(u.dtype)[:, None])
        positions, weights = self.nodes_weights(u)
        return dict(features=features, gain=gain, nodes=positions, normalized_weights=weights,
                    effective_weights=gain[..., None] * weights)

    def correction_components(self, u, h, equation, geometry):
        check_shape(u, geometry)
        if geometry.ndim != 2:
            raise ValueError("Portfolio architectures require a two-dimensional periodic field")
        step = numerics.validate(u, h, geometry, self.track)
        if hasattr(self, "delegate"):
            return self.delegate.correction_components(u, step, equation, geometry)
        if self.family == "rf":
            reaction = (lambda v, dt: reaction_step(v, dt, equation)) if self.track == "discrete" else (
                lambda v, dt: numerics.galerkin_reaction_step(v, dt, equation, substeps=self.reaction_substeps))
            base = reaction(numerics.heat_step(reaction(u, step / 2), step, equation, geometry,
                                              self.track, cache=self.cache), step / 2)
            return dict(base=torch.where(step == 0, u, base), increment=torch.zeros_like(u))
        base = numerics.df_step(u, step, equation, geometry, self.track,
                               reaction_substeps=self.reaction_substeps, cache=self.cache)
        increment = self._analytic_increment(u, step, equation, geometry)
        # Fixed rules avoid unnecessary feature transforms; they receive the
        # same coefficient-cache opportunity as every learned/classical rule.
        if self.family in _CONDITIONED or self.family == "quad2_linear":
            gain = self.response(u, step, equation, geometry)["gain"]
            if self.family == "band_gain":
                # DC / lower band / upper retained band partition exactly.
                # No second nonlinear multiplication or changed cutoff hides
                # additional information in this spatial-adaptation treatment.
                dc = increment.mean((-2, -1), keepdim=True).expand_as(increment)
                low = lowpass(increment, max(1, self.modes // 2))
                pieces = (dc, low - dc, increment - low)
                increment = sum(gain[:, i, None, None, None] * piece for i, piece in enumerate(pieces))
            else:
                increment = increment * gain[:, :, None, None]
        else:
            increment = increment * self._amplitude().to(u.dtype)
        return dict(base=base, increment=increment)

    def forward(self, u, h, equation, geometry):
        parts = self.correction_components(u, h, equation, geometry)
        return torch.where(broadcast_h(h, u) == 0, u, parts["base"] + parts["increment"])

    def linear_design(self, u, h, equation, geometry):
        """Gain-one prediction and unconstrained LS-initializer design, P=1/4.

        The affine forward model subsequently applies tanh, so this is the
        initializer's design, not a claim that deployed output is affine in
        coefficients. The final fit optimizes the exact bounded forward gain.
        """
        if self.family not in ("quad2_linear", "quad2_amplitude"):
            raise ValueError("Linear design is defined only for affine or scalar amplitude fitting")
        step = numerics.validate(u, h, geometry, self.track)
        base = numerics.df_step(u, step, equation, geometry, self.track,
                               reaction_substeps=self.reaction_substeps, cache=self.cache)
        increment = self._analytic_increment(u, step, equation, geometry)
        features = (physical_features(u, step, equation, geometry, self.track)
                    if self.family == "quad2_linear" else torch.empty((u.shape[0], 0), dtype=u.dtype, device=u.device))
        features = torch.cat((torch.ones((u.shape[0], 1), dtype=u.dtype, device=u.device), features), 1)
        return base + increment, increment * features[:, :, None, None]

    @torch.no_grad()
    def fit_least_squares(self, samples: Iterable, *, ridge=1e-8, peak_weight=.1,
                          budget=None, max_iterations=100):
        """LS-initialize, then fit the exact bounded neural training objective.

        Samples are (u,h,equation,geometry,target[,weight]). Each field has
        equal total squared-error weight independent of resolution. Scaling
        the columns before ridge avoids a spurious all-zero fit when physical
        defects are small. Both controls minimize weighted mean-square plus
        peak_weight * maximum-square error, with the same gain range as the
        conditioned model. Ridge applies only to initialization, not the final
        objective. CPU fitting/transfers occur once and are charged offline;
        inference is ordinary device-native Torch with no host optimization.
        This historical API name is retained for orchestration compatibility.
        """
        import numpy as np
        from scipy.optimize import minimize, minimize_scalar
        if self.family not in ("quad2_linear", "quad2_amplitude"):
            raise ValueError("Closed-form fitting is defined only for affine or scalar amplitude fitting")
        if not math.isfinite(ridge) or ridge < 0 or not math.isfinite(peak_weight) or peak_weight < 0:
            raise ValueError("Ridge and peak weight must be finite and nonnegative")
        if type(max_iterations) is not int or max_iterations < 1:
            raise ValueError("Positive fitting iteration limit required")
        designs, targets, field_data = [], [], []
        columns = 4 if self.family == "quad2_linear" else 1
        for sample in samples:
            if budget is not None: budget.check()
            u, h, equation, geometry, target = sample[:5]
            weight = 1. if len(sample) == 5 else float(sample[5])
            if not math.isfinite(weight) or weight <= 0 or target.shape != u.shape:
                raise ValueError("Matching target and finite positive sample weight required")
            prediction, basis = self.linear_design(u, h, equation, geometry)
            scale = math.sqrt(weight / math.prod(u.shape[2:]))
            designs.append((basis.movedim(1, -1).reshape(-1, columns) * scale).double())
            targets.append(((target - prediction).flatten() * scale).double())
            raw_features = (physical_features(u, numerics.validate(u, h, geometry, self.track), equation, geometry, self.track)
                            if columns == 4 else torch.empty((u.shape[0], 0), dtype=u.dtype, device=u.device))
            raw_features = torch.cat((torch.ones((u.shape[0], 1), dtype=u.dtype, device=u.device), raw_features), 1)
            for i in range(u.shape[0]):
                field_data.append((raw_features[i].double().cpu().numpy(), basis[i, 0].double().flatten().cpu().numpy(),
                                   (target[i] - prediction[i]).double().flatten().cpu().numpy(), weight))
        if not designs:
            raise ValueError("At least one training sample is required")
        design, target = torch.cat(designs), torch.cat(targets)
        if not bool(torch.isfinite(design).all() & torch.isfinite(target).all()):
            raise ValueError("Least squares inputs must be finite")
        scales = design.square().sum(0).sqrt().clamp_min(torch.finfo(torch.float64).eps)
        equilibrated = design / scales
        singular = torch.linalg.svdvals(equilibrated)
        rank = int(torch.linalg.matrix_rank(equilibrated))
        matrix = torch.cat((equilibrated, math.sqrt(ridge) * torch.eye(columns, device=design.device, dtype=design.dtype)))
        values = torch.cat((target, torch.zeros(columns, device=target.device, dtype=target.dtype)))
        # GPU gels requires full column rank; positive ridge supplies it. For
        # zero ridge use SVD pseudoinverse, consistent with rank diagnostics.
        coefficients = torch.linalg.lstsq(matrix, values).solution if ridge else torch.linalg.pinv(equilibrated) @ target
        coefficients = coefficients / scales
        ls_coefficients = coefficients.detach().cpu().numpy()
        evaluations = 0
        def objective(latent, *, scalar=False):
            nonlocal evaluations
            if budget is not None: budget.check()
            evaluations += 1
            value, gradient = 0., np.zeros(1 if scalar else columns)
            for features, increment, residual_target, weight in field_data:
                if scalar:
                    delta, derivative = float(latent[0]) - 1, np.ones(1)
                else:
                    response = math.tanh(float(features @ latent))
                    delta, derivative = .75 * response, .75 * (1 - response**2) * features
                residual = delta * increment - residual_target
                maximum = int(np.argmax(residual**2))
                value += weight * (float(np.mean(residual**2)) + peak_weight * residual[maximum]**2)
                derivative_gain = 2 * weight * (float(np.mean(residual * increment))
                                   + peak_weight * residual[maximum] * increment[maximum])
                gradient += derivative_gain * derivative
            return value / len(field_data), gradient / len(field_data)

        # Scale the objective uniformly so a small physical defect cannot make
        # the optimizer declare convergence before evaluating useful updates.
        initial_latent = np.zeros(columns)
        initial_objective = objective(np.array([1.]) if columns == 1 else initial_latent,
                                      scalar=columns == 1)[0]
        objective_scale = max(initial_objective, np.finfo(float).tiny)
        if columns == 1:
            raw = 1 + float(ls_coefficients[0])
            initializer = min(1.75 - 1e-7, max(.25 + 1e-7, raw))
            def scalar_objective(value): return objective(np.array([value]), scalar=True)[0] / objective_scale
            optimized = minimize_scalar(scalar_objective, method="bounded", bounds=(.25 + 1e-7, 1.75 - 1e-7),
                                        options={"maxiter": max_iterations, "xatol": 1e-12})
            candidates = [1., initializer, float(optimized.x)]
            amplitude = min(candidates, key=scalar_objective)
            self.amplitude_logit.copy_(torch.as_tensor(math.atanh((amplitude - 1) / .75),
                                                     dtype=self.amplitude_logit.dtype, device=self.amplitude_logit.device))
            extra = dict(unconstrained_amplitude=raw, fitted_amplitude=amplitude,
                         amplitude_clamped=initializer != raw, amplitude_bounds=[.25, 1.75],
                         initializer_objective=objective(np.array([initializer]), scalar=True)[0])
            final_objective = objective(np.array([amplitude]), scalar=True)[0]
            fitted_delta = torch.full((len(target),), amplitude - 1, dtype=target.dtype, device=target.device)
            fitted_sse = float((design[:, 0] * fitted_delta - target).square().sum())
            reported_coefficients = [amplitude - 1]
        else:
            feature_matrix = np.stack([v[0] for v in field_data])
            suggested_delta = feature_matrix @ ls_coefficients
            suggested_latent = np.arctanh(np.clip(suggested_delta / .75, -1 + 1e-7, 1 - 1e-7))
            initializer = np.linalg.lstsq(feature_matrix, suggested_latent, rcond=None)[0]
            def affine_objective(value):
                cost, gradient = objective(value)
                return cost / objective_scale, gradient / objective_scale
            optimized = minimize(affine_objective, initializer, method="L-BFGS-B", jac=True,
                options={"maxiter": max_iterations, "maxfun": 5 * max_iterations, "ftol": 1e-13,
                         "gtol": 1e-9, "maxls": 40})
            candidates = [initial_latent, initializer, np.asarray(optimized.x)]
            selected = min(candidates, key=lambda value: objective(value)[0])
            self.linear_coefficients.copy_(torch.as_tensor(selected, dtype=self.linear_coefficients.dtype,
                                                          device=self.linear_coefficients.device))
            extra = dict(initializer_objective=objective(initializer)[0], coefficient_space="affine pre-tanh gain logits")
            final_objective = objective(selected)[0]
            fitted_sse = sum(weight * np.mean((.75 * np.tanh(features @ selected) * increment - truth)**2)
                             for features, increment, truth, weight in field_data)
            reported_coefficients = selected.tolist()
        return dict(fit="training_only_bounded_affine_matched_objective" if columns == 4 else "training_only_bounded_scalar_matched_objective",
            rows=len(target), fields=len(field_data), rank=rank, columns=columns, identifiable=rank == columns,
            ridge_initializer_only=float(ridge), peak_weight=float(peak_weight), **extra,
            effective_gain_bounds=[.25, 1.75], optimizer="L-BFGS-B" if columns == 4 else "bounded_scalar",
            optimizer_success=bool(optimized.success), optimizer_message=str(optimized.message),
            optimizer_iterations=int(optimized.nit), objective_evaluations=evaluations,
            maximum_iterations=max_iterations, objective="mean over fields of weight*(mean(error^2)+peak_weight*max(error^2))",
            initial_training_objective=float(initial_objective), fitted_training_objective=float(final_objective),
            scientific_success=False, inference_uses_host_optimizer=False,
            singular_values=singular.cpu().tolist(), ls_initializer_coefficients=ls_coefficients.tolist(), coefficients=reported_coefficients,
            initial_weighted_sse=float(target.square().sum()),
            fitted_weighted_sse=float(fitted_sse))

    fit_linear = fit_least_squares

    def parameter_report(self, u=None, h=None, equation=None, geometry=None):
        report = dict(family=self.family, trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
            stored_parameters=sum(p.numel() for p in self.parameters()),
            fitted_buffer_coefficients=4 if self.family == "quad2_linear" else 0,
            state_dict={k: v.detach().cpu().tolist() if v.numel() <= 4096 else
                        dict(shape=list(v.shape), numel=v.numel(), min=float(v.detach().min()), max=float(v.detach().max()),
                             mean=float(v.detach().mean()), norm=float(v.detach().norm()), full_values="sealed checkpoint")
                        for k, v in self.state_dict().items()},
            identifiability="Node, amplitude and conditioner may compensate; effective responses and data Jacobian rank must be audited")
        if not hasattr(self, "delegate") and self.family != "rf":
            prototype = self.normalized_weights.reshape(1, 1, 1, -1)
            nodes, weights = self.nodes_weights(prototype)
            report.update(nodes=nodes.detach().cpu().tolist(), normalized_weights=weights.cpu().tolist(),
                          weight_sum=float(weights.sum()), global_amplitude=float(self._amplitude().detach()))
        if u is not None:
            response = self.response(u, h, equation, geometry)
            report["response"] = {k: v.detach().cpu().tolist() if torch.is_tensor(v) else v for k, v in response.items()}
        return report

    def architecture_metadata(self):
        info = dict(family=self.family, track=self.track, **self.spec,
            parameters=sum(p.numel() for p in self.parameters()),
            trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
            fitted_buffer_coefficients=4 if self.family == "quad2_linear" else 0,
            modes=self.modes, physical_core="reaction_first_strang" if self.family == "rf" else "diffusion_first_strang",
            reaction_target="nodal_logistic" if self.track == "discrete" else "quadratic_dealiased_galerkin",
            reaction_flow="exact_logistic" if self.track == "discrete" else "projected_RK4_not_exact_logistic",
            correction_input="compressed" if self.family == "quad2_input" else "full_state",
            published_FNO_reproduction=False, continuum_truth_claim=False, clipping=False,
            invariant_interval_guarantee=False, stability_certificate=False,
            state_inputs=["mean", "log1p(h*r)", "log1p(h*active_diffusion_rate)"],
            gain_parameterization="1+0.75*tanh(global+conditioner); same effective (0.25,1.75) range as scalar-only control, not normalized final weights",
            quadrature_normalization="unit sum before optional gain", node_parameterization="symmetric interior nodes: .005+.49*sigmoid(logit)",
            reference_or_future_inputs=False, parameter_matching_claim=False, compute_matching_claim=False,
            exact_correction_nulls=["constant", "reaction=0", "diffusion=0", "time=0"],
            actual_work="full-field DF, transported quadratic products, optional features and output projection; count all in timing")
        if hasattr(self, "delegate"):
            original = self.delegate.architecture_metadata()
            info.update({key: original[key] for key in ("physical_core", "exact_correction_nulls", "width", "depth", "fno_full_grid_local_path")})
            info["delegated_historical_family"] = self.delegate.family
            if self.family in ("fno_small", "fno_standard", "direct_fno"):
                info["state_inputs"] = ["full field", "centered full field", "h/t_ref", "r*t_ref",
                                        "kappa*t_ref/Lx^2", "kappa*t_ref/Ly^2", "1/nx", "1/ny"]
                info["gain_parameterization"] = "full-grid neural residual, scaled by h^3 for hybrid or h for direct"
            if self.family == "historical_half":
                info.update(quadrature_normalization="historical frozen total weight 0.5", gain_parameterization="unchanged historical frozen implementation")
            if self.family == "analytic_quad_cubic":
                info.update(nodes=self.quad_nodes, cubic_nodes=self.cubic_nodes,
                            gain_parameterization="fixed unit-weight analytic quadratic and cubic responses")
        if self.family == "quad2_linear":
            info["gain_parameterization"] = "1+0.75*tanh(affine([mean,log1p(h*r),log1p(h*active_rate)])); four fitted coefficients, same gain range and training loss as conditioner"
        elif self.family == "residual_quad2":
            info["gain_parameterization"] = "1+0.25*tanh(conditioner); fixed GL2 nodes and amplitude"
        elif self.family == "conditioned_rich":
            info["state_inputs"] += ["variance", "mean_abs_centered", "log1p(h*diffusion_rate_spread)", "log1p(h*r*std)"]
        elif self.family == "band_gain":
            info["gain_parameterization"] = "three gains 1+0.5*tanh(conditioner), for DC/lower/upper retained bands"
        if self.family == "quad2_amplitude":
            info["gain_parameterization"] = "training-only bounded scalar fit to matched MSE+peak objective; LS initializer; amplitude in (0.25,1.75)"
        if self.family not in _CONDITIONED | {"historical_half", "quad2_linear"} and self.family not in {"fno_small", "fno_standard", "direct_fno"}:
            info["state_inputs"] = []
        return info


def make_model(family, track="discrete", config=None):
    config = dict(config or {})
    allowed = {"modes", "width", "depth", "t_ref", "reaction_substeps", "quad_nodes", "cubic_nodes", "cache_entries", "hidden"}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"Unknown portfolio architecture settings: {sorted(unknown)}")
    return PortfolioModel(family, track, **config)
