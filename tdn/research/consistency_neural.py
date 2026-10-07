"""Paired commutator-gated and moment-calibrated hybrid corrections.

The unmodified arms are constructed by ``premix_neural.build_model``. New
arms retain those backbones and the complete physical Strang state. The gate
enforces three exact commuting limits, while a separate mean head can learn
a legitimate reaction-induced mean correction. Neither constraint is a claim
of an exact mean ODE, mean conservation, stability, or architecture advantage.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tdn.features.local import extract_features
from tdn.numerics.operators import broadcast_h, check_shape
from tdn.numerics.splitting import split_step

from . import premix_neural
from .reaction import capacity_update


FAMILIES = ("premix", "fno", "premix_gated", "fno_gated", "premix_moment",
            "fno_moment", "precompress")
CONSTRAINED_FAMILIES = ("premix_gated", "fno_gated", "premix_moment", "fno_moment")


def _reference_scales(t_ref: float, U_ref: float) -> None:
    if not all(math.isfinite(float(value)) and float(value) > 0
               for value in (t_ref, U_ref)):
        raise ValueError("Reference scales must be finite and positive")


def discrete_logistic_commutator(u: Tensor, equation, geometry) -> Tensor:
    """Return R'(u)D(u)-D(R(u)) for the declared logistic discrete problem.

    C_i = r sum_j w_ij (u_j-u_i)^2, w_ij=kappa/dx_d**2 for
    each of the two periodic neighbors in direction d. This nonnegative
    state/time**2 field vanishes exactly for homogeneous fields, r=0 and
    kappa=0. Its dependence on perturbation amplitude is exactly quadratic.
    """
    check_shape(u, geometry)
    result = torch.zeros_like(u)
    for axis, dx in enumerate(geometry.dx, start=2):
        weight = equation.reaction_rate * equation.kappa / dx**2
        plus = torch.roll(u, -1, axis) - u
        minus = torch.roll(u, 1, axis) - u
        result = result + weight * (plus.square() + minus.square())
    return result


def commutator_gate(u: Tensor, equation, geometry, *, t_ref: float = 1.,
                    U_ref: float = 1.) -> Tensor:
    """A dimensionless local gate tanh(t_ref**2*C/U_ref), independent of h."""
    _reference_scales(t_ref, U_ref)
    return torch.tanh(t_ref**2 * discrete_logistic_commutator(u, equation, geometry) / U_ref)


def _spatial_mean(value: Tensor) -> Tensor:
    return value.mean(tuple(range(2, value.ndim)), keepdim=True)


def _mean_tensor(value: Tensor | float, state: Tensor) -> Tensor:
    result = torch.as_tensor(value, dtype=state.dtype, device=state.device)
    if result.ndim == 1 and result.shape[0] == state.shape[0]:
        result = result.reshape(state.shape[0], *([1] * (state.ndim - 1)))
    if result.ndim != 0 and result.shape != (state.shape[0], *([1] * (state.ndim - 1))):
        raise ValueError("Mean quantities must be scalar or one scalar per parent")
    return result


def calibrate_mean(state: Tensor, target_mean: Tensor | float) -> Tensor:
    """Boundedly redistribute a candidate to its requested physical mean.

    For target>=mean(y), add (1-y)*(target-mean(y))/(1-mean(y));
    otherwise subtract y*(mean(y)-target)/mean(y). Each branch is a
    convex redistribution on [0,1] and its spatial mean is the target.
    Denominators at saturated endpoints are regularized only when zero;
    finite physical inputs and target in [0,1] are required. No clipping or
    cross-parent pooling is used. The branch boundary is piecewise smooth.
    """
    if state.ndim < 3 or state.shape[1] != 1 or not state.is_floating_point():
        raise ValueError("Mean calibration expects floating [batch,1,*grid]")
    target = _mean_tensor(target_mean, state)
    current = _spatial_mean(state)
    change = target - current
    capacity = torch.where(change >= 0, 1 - state, state)
    mean_capacity = torch.where(change >= 0, 1 - current, current)
    safe = torch.where(mean_capacity > 0, mean_capacity, torch.ones_like(mean_capacity))
    # The ratio precedes multiplication to avoid rounding outside capacity
    # for extreme but admissible requested means.
    return state + capacity * (change / safe)


def mean_shape_update(base: Tensor, mean_increment: Tensor | float,
                      spatial_increment: Tensor) -> Tensor:
    """Capacity-map shape, then calibrate its mean to the scalar head target.

    The spatial proposal is explicitly centered, but the signed-capacity map
    may generate DC. Calibration therefore happens after that physical map.
    The mean target itself is a bounded learned correction to mean(base).
    """
    if spatial_increment.shape != base.shape:
        raise ValueError("Spatial increment must match the physical base")
    scalar = _mean_tensor(mean_increment, base)
    target = capacity_update(_spatial_mean(base), scalar)
    centered = spatial_increment - _spatial_mean(spatial_increment)
    candidate = capacity_update(base, centered)
    return calibrate_mean(candidate, target)


class ConsistencySolver(nn.Module):
    """One existing backbone with a paired gate and optional mean/shape heads."""

    def __init__(self, family: str, *, width: int = 16, modes: int = 4,
                 normalization=None, t_ref: float = 1., U_ref: float = 1.):
        super().__init__()
        if family not in CONSTRAINED_FAMILIES:
            raise ValueError(f"Unsupported consistency neural family: {family}")
        self.family = family
        self.base_family = family.split("_", 1)[0]
        self.moment = family.endswith("_moment")
        self.backbone = premix_neural.build_model(
            self.base_family, width=width, modes=modes, normalization=normalization,
            t_ref=t_ref, U_ref=U_ref)
        self.t_ref, self.U_ref = self.backbone.t_ref, self.backbone.U_ref
        if self.moment:
            self.mean_head = nn.Conv2d(width, 1, 1)
            nn.init.zeros_(self.mean_head.weight)
            nn.init.zeros_(self.mean_head.bias)

    @property
    def head(self):
        return self.backbone.head

    @property
    def feature_mean(self):
        return self.backbone.feature_mean

    @property
    def feature_std(self):
        return self.backbone.feature_std

    def set_normalization(self, mean, std):
        self.backbone.set_normalization(mean, std)
        return self

    def _latent(self, features: Tensor) -> Tensor:
        if self.base_family == "premix":
            return self.backbone.spectral(self.backbone.interactions(features))
        return F.silu(self.backbone.project(self.backbone.blocks(self.backbone.lift(features))))

    def correction_components(self, u: Tensor, h: float | Tensor, equation,
                              geometry) -> dict[str, Tensor]:
        """Expose physical and learned terms for independent structural audits."""
        if geometry.ndim != 2:
            raise ValueError("Consistency neural models require two-dimensional geometry")
        check_shape(u, geometry)
        step = broadcast_h(h, u)
        features = extract_features(u, equation, geometry, t_ref=self.t_ref, U_ref=self.U_ref)
        features = ((features - self.feature_mean.to(features.dtype)) /
                    self.feature_std.to(features.dtype)).movedim(-1, 1)
        horizon = (step / self.t_ref).expand(u.shape[0], 1, *geometry.grid)
        inputs = torch.cat((features, horizon), dim=1).to(self.head.weight.dtype)
        latent = self._latent(inputs)
        raw_spatial = self.head(latent).to(u.dtype)
        gate = commutator_gate(u, equation, geometry, t_ref=self.t_ref, U_ref=self.U_ref)
        factor = self.U_ref * (step / self.t_ref).pow(3)
        spatial_increment = factor * gate * raw_spatial
        base = split_step(u, step, equation, geometry)
        result = dict(base=base, gate=gate, raw_spatial=raw_spatial,
                      spatial_increment=spatial_increment)
        if self.moment:
            # A head on the per-parent pooled latent is a separate scalar
            # branch, not the mean of a capacity-limited spatial correction.
            raw_mean = self.mean_head(_spatial_mean(latent)).to(u.dtype)
            mean_increment = factor * _spatial_mean(gate) * raw_mean
            result.update(raw_mean=raw_mean, mean_increment=mean_increment,
                          target_mean=capacity_update(_spatial_mean(base), mean_increment))
        return result

    def forward(self, u: Tensor, h: float | Tensor, equation, geometry) -> Tensor:
        terms = self.correction_components(u, h, equation, geometry)
        if self.moment:
            return mean_shape_update(terms["base"], terms["mean_increment"], terms["spatial_increment"])
        return capacity_update(terms["base"], terms["spatial_increment"])

    def architecture_metadata(self) -> dict:
        metadata = self.backbone.architecture_metadata()
        metadata.update(
            family=self.family,
            constraint_backbone=self.base_family,
            commutator="C_i=r*sum_neighbors(kappa/dx_d^2)*(u_j-u_i)^2 = R'(u)D(u)-D(R(u))",
            gate="tanh(t_ref^2*C_i/U_ref), local and independent of queried h",
            exact_correction_limits=["homogeneous input", "reaction_rate=0", "kappa=0", "h=0"],
            small_amplitude_behavior="gate is quadratic to leading order; C is exactly quadratic",
            increment="U_ref*(h/t_ref)^3*gate*raw_spatial",
            physical_output_bandlimited=False,
            raw_output_bandlimited=self.base_family == "premix",
            gate_frequency_effect="nonlinear full-grid gate multiplication adds output frequencies",
            global_reductions="none beyond backbone Fourier transforms",
            mean_dynamics="reaction changes the mean; no mean conservation or exact mean ODE assertion",
        )
        if self.moment:
            metadata.update(
                scalar_mean_head="separate pointwise head of per-parent mean latent",
                initialization="zero spatial output head and zero scalar mean head",
                spatial_head="original backbone head; center its gated increment per parent",
                target_mean="capacity_update(mean(Strang base), U_ref*(h/t_ref)^3*mean(gate)*raw_mean)",
                physical_mean_calibration="after spatial signed-capacity map, bounded affine capacity redistribution to target",
                global_reductions="per-parent latent/gate/spatial means and physical mean calibration; no cross-parent pooling",
                output_form="signed-capacity spatial map followed by bounded per-parent mean calibration",
                increment="separate cubic commutator-gated scalar mean and centered spatial increments",
                mean_claim="calibrated to learned bounded target, not exact physical mean",
                differentiability="piecewise smooth; redistribution branch at target_mean=mean(candidate)",
                capacity_comparison="one additional width+1 scalar mean head parameters versus paired gated arm",
            )
        return metadata


def build_model(family: str, *, width: int = 16, modes: int = 4,
                normalization=None, t_ref: float = 1., U_ref: float = 1.) -> nn.Module:
    """Build a declared arm; current controls exactly use the original factory."""
    if family in ("premix", "fno", "precompress"):
        return premix_neural.build_model(family, width=width, modes=modes,
                                        normalization=normalization, t_ref=t_ref, U_ref=U_ref)
    if family in CONSTRAINED_FAMILIES:
        return ConsistencySolver(family, width=width, modes=modes,
                                 normalization=normalization, t_ref=t_ref, U_ref=U_ref)
    raise ValueError(f"Unsupported consistency neural family: {family}")


__all__ = ["FAMILIES", "CONSTRAINED_FAMILIES", "ConsistencySolver", "build_model",
           "discrete_logistic_commutator", "commutator_gate", "calibrate_mean", "mean_shape_update"]
