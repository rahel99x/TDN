"""A bounded next-step architecture roster; historical controls stay unchanged."""
from __future__ import annotations

import torch
from torch import nn

from tdn.analysis.frontier import numerics as physical
from tdn.analysis.frontier.models import FrontierModel
from tdn.analysis.portfolio.models import make_model as portfolio_model
from tdn.numerics.operators import broadcast_h
from .numerics import cubic_commutator_defect, deployable_defect_scale


def _spec(role, *, trainable=False, description=""):
    return dict(role=role, ownership=role, fit="gradient" if trainable else "frozen",
                trainable=trainable, description=description)


MODEL_SPECS = {
    "df": _spec("Theirs", description="Unchanged diffusion-first Strang control"),
    "etdrk4": _spec("Theirs", description="Unchanged Cox–Matthews ETDRK4 control"),
    "quad2_full": _spec("Analytic control", description="Normalized full-output GL2"),
    "quad4_full": _spec("Analytic control", description="Normalized full-output GL4"),
    "analytic_quad_cubic": _spec("Analytic control", description="Full nested Volterra quadratic+cubic control"),
    "quad2_conditioned": _spec("Ours historical control", trainable=True, description="Unchanged scalar-conditioned output-compressed GL2"),
    "commutator_raw": _spec("Ours numerical", description="Full GL2 plus raw leading h^3 cubic DF defect"),
    "commutator_cubic": _spec("Ours numerical", description="Full GL2 plus heat-filtered leading cubic defect; fixed unit gains"),
    "two_basis": _spec("Ours", trainable=True, description="Two bounded global gains on identical full GL2 and filtered cubic bases"),
    "fno_legacy": _spec("Theirs", trainable=True, description="Historical local hybrid FNO with nonzero output initialization"),
    "fno_zero": _spec("Theirs adapted control", trainable=True, description="Same hybrid FNO with baseline-preserving zero residual head"),
    "fno_scaled": _spec("Theirs adapted control", trainable=True, description="Zero-head FNO with deployable dimensionless physical defect scaling"),
    "fno_mean": _spec("Theirs adapted control", trainable=True, description="Scaled zero-head FNO with separate mean and centered residual outputs"),
}
FAMILIES = tuple(MODEL_SPECS)
TRAINABLE_FAMILIES = tuple(f for f, spec in MODEL_SPECS.items() if spec["trainable"])
PHYSICAL_FAMILIES = ("quad2_full", "quad4_full", "analytic_quad_cubic", "quad2_conditioned",
                     "commutator_raw", "commutator_cubic", "two_basis", "fno_scaled", "fno_mean")
ZERO_HEAD_FAMILIES = ("fno_zero", "fno_scaled", "fno_mean")
_CONTROL_FAMILIES = ("df", "etdrk4", "quad2_full", "quad4_full", "analytic_quad_cubic", "quad2_conditioned")
_MODEL_KEYS = {"modes", "width", "depth", "t_ref", "reaction_substeps", "quad_nodes", "cubic_nodes", "cache_entries", "hidden"}


def _state_report(model):
    return {name: value.detach().cpu().tolist() if value.numel() <= 4096 else
            dict(shape=list(value.shape), numel=value.numel(), norm=float(value.detach().norm()),
                 full_values="sealed checkpoint") for name, value in model.state_dict().items()}


class CubicBasisModel(nn.Module):
    def __init__(self, family, track, config):
        super().__init__()
        self.family, self.track = family, track
        self.base_rule = portfolio_model("quad2_full", track, config)
        self.reaction_substeps = self.base_rule.reaction_substeps
        self.modes = self.base_rule.modes
        self.transport = "none" if family == "commutator_raw" else "symmetric_heat"
        if family == "two_basis":
            self.gain_logits = nn.Parameter(torch.zeros(2, dtype=torch.float64))
        else:
            self.register_buffer("gain_logits", torch.zeros(2, dtype=torch.float64))

    @property
    def cache(self):
        return self.base_rule.cache

    @property
    def cache_metadata(self):
        return self.base_rule.cache_metadata

    def clear_cache(self):
        self.base_rule.clear_cache()

    def gains(self):
        return 1 + .75 * torch.tanh(self.gain_logits)

    def correction_components(self, u, h, equation, geometry):
        parts = self.base_rule.correction_components(u, h, equation, geometry)
        cubic = cubic_commutator_defect(u, h, equation, geometry, track=self.track, transport=self.transport)
        gains = self.gains().to(u.dtype)
        return dict(base=parts["base"], increment=gains[0] * parts["increment"] + gains[1] * cubic,
                    quadratic=parts["increment"], cubic=cubic, effective_gains=gains)

    def forward(self, u, h, equation, geometry):
        parts = self.correction_components(u, h, equation, geometry)
        return torch.where(broadcast_h(h, u) == 0, u, parts["base"] + parts["increment"])

    def architecture_metadata(self):
        return dict(family=self.family, track=self.track, **MODEL_SPECS[self.family],
                    parameters=sum(p.numel() for p in self.parameters()),
                    trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
                    physical_core="diffusion_first_strang", correction_input="full_state", output_compression=False,
                    gain_bounds=[.25, 1.75], coefficient_initialization=[1., 1.], cubic_transport=self.transport,
                    cubic_scope="leading h^3 centered-amplitude cubic Volterra defect; fixed-grid asymptotic only",
                    temporal_order_claim="no new full-method order or stiff-order claim",
                    projected_products_associative=False, invariant_interval_guarantee=False,
                    clipping=False, stability_certificate=False, published_FNO_reproduction=False,
                    future_or_reference_inputs=False,
                    exact_correction_nulls=["constant", "reaction=0", "diffusion=0", "time=0"])

    def parameter_report(self, *args):
        return dict(family=self.family, effective_gains=self.gains().detach().cpu().tolist(),
                    trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
                    cubic_transport=self.transport, state_dict=_state_report(self),
                    identifiability="Two basis coefficients need independent response directions; assess data Gram matrix")


class AdvanceFNO(FrontierModel):
    """Same local/spectral FNO trunk with controlled residual parameterizations.

    Zero initialization blocks first-backward trunk gradients intentionally.
    The output head learns first; trunk gradients activate after a nonzero head
    update. This is baseline-preserving optimization, not an assertion that all
    layers receive gradients on the first step.
    """
    def __init__(self, family, track, config):
        self.advance_family = family
        options = {k: v for k, v in config.items() if k != "hidden"}
        super().__init__("fno_small", track, **options)
        if family in ZERO_HEAD_FAMILIES:
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)
        if family == "fno_mean":
            # A spatially constant bias would be annihilated by centering and
            # be an unidentifiable duplicate mean parameter. Do not fit it.
            self.head[-1] = nn.Conv2d(self.width, 1, 1, bias=False)
            nn.init.zeros_(self.head[-1].weight)
            self.mean_head = nn.Sequential(nn.Linear(self.width, self.width), nn.GELU(), nn.Linear(self.width, 1))
            nn.init.zeros_(self.mean_head[-1].weight)
            nn.init.zeros_(self.mean_head[-1].bias)
        self._legacy_metadata = super().architecture_metadata()
        self.family = family

    def correction_components(self, u, h, equation, geometry):
        step = physical.validate(u, h, geometry, self.track)
        if geometry.ndim != 2:
            raise ValueError("Advance FNOs require two-dimensional periodic fields")
        base = physical.df_step(u, step, equation, geometry, self.track,
                                reaction_substeps=self.reaction_substeps, cache=self.cache)
        features = self._features(u, step, equation, geometry).to(self.lift.weight.dtype)
        encoded = self.blocks(self.lift(features))
        raw = self.head(encoded).to(u.dtype)
        if self.advance_family in ("fno_legacy", "fno_zero"):
            return dict(base=base, increment=(step / self.t_ref).pow(3) * raw)
        scale = deployable_defect_scale(u, step, equation, geometry, track=self.track)
        if self.advance_family == "fno_mean":
            centered = raw - raw.mean((-2, -1), keepdim=True)
            mean = self.mean_head(encoded.mean((-2, -1))).to(u.dtype)[:, :, None, None]
            return dict(base=base, increment=scale * (centered + mean),
                        centered_increment=scale * centered, mean_increment=scale * mean,
                        output_scale=scale)
        return dict(base=base, increment=scale * raw, output_scale=scale)

    def architecture_metadata(self):
        info = dict(self._legacy_metadata)
        family = self.advance_family
        info.update(family=family, **MODEL_SPECS[family], baseline_preserving_initialization=family in ZERO_HEAD_FAMILIES,
                    zero_head_gradient_semantics="head gradients first; trunk gradients after head activation" if family in ZERO_HEAD_FAMILIES else "legacy nonzero head",
                    physical_residual_scale="h^3*r*variance*active_rate*(r+active_rate)" if family in ("fno_scaled", "fno_mean") else "(h/t_ref)^3",
                    mean_split=family == "fno_mean", reference_or_future_scale_inputs=False,
                    centered_output_bias=False if family == "fno_mean" else True,
                    scale_is_error_estimator=False, clipping=False, stability_certificate=False,
                    published_FNO_reproduction=False)
        if family in ("fno_scaled", "fno_mean"):
            info["exact_correction_nulls"] = ["constant", "reaction=0", "diffusion=0", "time=0"]
        return info

    def parameter_report(self, *args):
        return dict(family=self.advance_family, trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
                    state_dict=_state_report(self), zero_head=self.advance_family in ZERO_HEAD_FAMILIES,
                    output_scaling=self.architecture_metadata()["physical_residual_scale"])


def make_model(family, track="discrete", config=None):
    if family not in MODEL_SPECS or track not in physical.TRACKS:
        raise ValueError("Unknown advance family or equation track")
    config = dict(config or {})
    unknown = set(config) - _MODEL_KEYS
    if unknown:
        raise ValueError("Unknown advance model settings: " + ", ".join(sorted(unknown)))
    if family in _CONTROL_FAMILIES:
        return portfolio_model(family, track, config)
    if family in ("commutator_raw", "commutator_cubic", "two_basis"):
        return CubicBasisModel(family, track, config)
    return AdvanceFNO(family, track, config)
