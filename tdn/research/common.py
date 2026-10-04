"""Shared, step-size-independent components for bounded research experiments.

These experimental full-state models deliberately live outside the established
pilot/checkpoint schema. Physical features and exact-limit factors stay live in
the autograd graph during rollouts; normalization is fitted on training parents.
"""
from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import Tensor, nn

from tdn.features.local import extract_features, feature_count
from tdn.numerics.operators import check_shape


def _scales(t_ref: float, U_ref: float) -> None:
    if not all(math.isfinite(float(x)) and float(x) > 0 for x in (t_ref, U_ref)):
        raise ValueError("Reference scales must be finite, positive and independent of h")


def validate_normalization(mean, std, count: int, *, dtype=None, device=None) -> tuple[Tensor, Tensor]:
    mean = torch.as_tensor(mean, dtype=dtype, device=device)
    std = torch.as_tensor(std, dtype=dtype, device=device)
    if tuple(mean.shape) != (count,) or tuple(std.shape) != (count,):
        raise ValueError("Normalization must contain one mean and scale per feature")
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
        raise ValueError("Feature means must be finite and scales finite and positive")
    return mean, std


class LocalEncoder(nn.Module):
    """One hidden layer, zero amplitude head, no queried horizon input."""

    def __init__(self, outputs: int, *, width: int = 16, t_ref: float = 1.,
                 U_ref: float = 1., ndim: int = 2):
        super().__init__()
        _scales(t_ref, U_ref)
        if type(outputs) is not int or type(width) is not int or min(outputs, width) <= 0:
            raise ValueError("Encoder output count and width must be positive integers")
        count = feature_count(ndim)
        self.ndim, self.t_ref, self.U_ref = ndim, float(t_ref), float(U_ref)
        self.body = nn.Sequential(nn.Linear(count, width), nn.SiLU())
        self.head = nn.Linear(width, outputs)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("feature_mean", torch.zeros(count))
        self.register_buffer("feature_std", torch.ones(count))

    def set_normalization(self, mean, std):
        mean, std = validate_normalization(mean, std, self.feature_mean.numel(),
                                           dtype=self.feature_mean.dtype,
                                           device=self.feature_mean.device)
        with torch.no_grad():
            self.feature_mean.copy_(mean)
            self.feature_std.copy_(std)
        return self

    def forward(self, u: Tensor, equation, geometry) -> Tensor:
        if geometry.ndim != self.ndim:
            raise ValueError("Geometry dimension differs from the encoder feature schema")
        features = extract_features(u, equation, geometry, t_ref=self.t_ref, U_ref=self.U_ref)
        normalized = ((features - self.feature_mean.to(features.dtype)) /
                      self.feature_std.to(features.dtype))
        value = self.head(self.body(normalized.to(self.head.weight.dtype)))
        return value.movedim(-1, 1)


def fit_feature_normalization(samples: Iterable[tuple[Tensor, object, object]], *,
                              t_ref: float = 1., U_ref: float = 1.) -> tuple[Tensor, Tensor]:
    """Fit one shared normalization from explicitly supplied training states.

    Each item is (state, equation, geometry). Accumulate centered second moments
    in FP64; validation/diagnostic data must never be passed by the caller.
    """
    _scales(t_ref, U_ref)
    count = 0
    mean = variance_sum = None
    with torch.no_grad():
        for state, equation, geometry in samples:
            features = extract_features(state.detach().cpu().double(), equation, geometry,
                                        t_ref=t_ref, U_ref=U_ref)
            flat = features.reshape(-1, features.shape[-1])
            n = flat.shape[0]
            batch_mean = flat.mean(0)
            batch_sum = (flat - batch_mean).square().sum(0)
            if not torch.isfinite(flat).all():
                raise ValueError("Training features must be finite")
            if mean is None:
                mean, variance_sum, count = batch_mean, batch_sum, n
            else:
                if mean.shape != batch_mean.shape:
                    raise ValueError("All training states must use the same feature schema")
                delta = batch_mean - mean
                variance_sum = variance_sum + batch_sum + delta.square() * count * n / (count + n)
                mean = mean + delta * n / (count + n)
                count += n
    if count == 0:
        raise ValueError("Cannot fit normalization without training parents")
    std = torch.sqrt(torch.clamp(variance_sum / count, min=0.))
    std = torch.where(std > 1e-6, std, torch.ones_like(std))
    return mean, std


def commuting_gate(u: Tensor, equation, geometry, *, t_ref: float = 1.,
                   U_ref: float = 1.) -> Tensor:
    """Bounded radius-two variation factor, exactly zero in commuting limits.

    The second pass includes diagonal neighbors in multiple dimensions. A
    radius-one or axial-only radius-two gate can incorrectly erase a nonzero
    nested commutator at the center of an otherwise flat local patch.
    This is an architectural constraint, not a nonlinear stability certificate.
    """
    _scales(t_ref, U_ref)
    check_shape(u, geometry)
    physical = u if u.dtype == torch.float64 else u.float()
    if equation.kappa == 0 or equation.reaction_rate == 0:
        return physical * 0.
    variation = torch.zeros_like(physical)
    for axis in range(2, physical.ndim):
        for shift in (-1, 1):
            delta = (torch.roll(physical, shift, axis) - physical) / U_ref
            variation = variation + delta.square()
    extended = variation
    for axis in range(2, physical.ndim):
        extended = extended + torch.roll(variation, 1, axis) + torch.roll(variation, -1, axis)
    factor = (equation.kappa * t_ref * sum(dx**-2 for dx in geometry.dx) *
              equation.reaction_rate * t_ref)
    scaled = factor * extended
    return scaled / (1. + scaled)
