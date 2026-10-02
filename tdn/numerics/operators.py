"""Periodic central differences on the declared semidiscrete grid."""
from __future__ import annotations

import torch
from torch import Tensor

from .types import Equation, Geometry


def check_shape(u: Tensor, geometry: Geometry) -> None:
    if not u.is_floating_point():
        raise TypeError("physical states must be real floating-point tensors")
    if u.ndim != geometry.ndim + 2 or u.shape[1] != 1:
        raise ValueError("expected scalar state shape [batch, 1, *grid]")
    if tuple(u.shape[2:]) != geometry.grid or u.shape[0] < 1:
        raise ValueError("state does not match the declared geometry")


def broadcast_h(h: float | Tensor, u: Tensor) -> Tensor:
    """Preserve tensor time gradients; accept a scalar or one h per parent."""
    step = torch.as_tensor(h, dtype=u.dtype, device=u.device)
    if step.ndim == 1:
        if step.shape[0] != u.shape[0]:
            raise ValueError("a vector h must contain one horizon per batch item")
        step = step.reshape(step.shape[0], *([1] * (u.ndim - 1)))
    if step.ndim != 0 and tuple(step.shape) != (u.shape[0], *([1] * (u.ndim - 1))):
        raise ValueError("h must be scalar, [batch], or [batch,1,...,1]")
    return step


def laplacian(u: Tensor, geometry: Geometry) -> Tensor:
    check_shape(u, geometry)
    result = torch.zeros_like(u)
    for axis, dx in enumerate(geometry.dx, start=2):
        result = result + (torch.roll(u, -1, axis) - 2 * u + torch.roll(u, 1, axis)) / dx**2
    return result


def rhs(u: Tensor, equation: Equation, geometry: Geometry) -> Tensor:
    """Coupled field, used by the independent reference, not split subflows."""
    return equation.kappa * laplacian(u, geometry) + equation.reaction_rate * u * (1 - u)


def weighted_norm(v: Tensor, geometry: Geometry | None = None) -> Tensor:
    """Volume-normalized L2, averaged over parents: sqrt(mean(v**2)).

    For a scalar uniform grid, W_cell = cell_volume / domain_volume. Thus the
    metric has state units, includes all cells, and does not grow with grid
    resolution. Every parent receives equal weight. There are no output masks
    or species rescalings in this scalar problem. FP64 reduction is retained
    for diagnostics even if the state is FP32.
    """
    if geometry is not None:
        check_shape(v, geometry)
    return v.to(torch.float64).square().mean().sqrt()
