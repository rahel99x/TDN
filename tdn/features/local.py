"""Legal periodic radius-one inputs; requested time never enters the encoder.

The reference supports one scalar logistic reaction–diffusion field. Physical
states are [batch, 1, *grid]; model features are [batch, *grid, features].
The streaming path gathers exactly the same stencil without allocating a
whole-domain feature matrix. Neither path pools across the domain.
"""
from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import Tensor

SUPPORT_RADIUS = 1
FEATURE_SCHEMA_VERSION = 1


def feature_count(ndim: int) -> int:
    if ndim not in (1, 2, 3):
        raise ValueError("only 1-D, 2-D, and 3-D scalar RD features are supported")
    return 4 + 4 * ndim


def feature_names(ndim: int) -> tuple[str, ...]:
    feature_count(ndim)
    return ("state", *(f"scaled_gradient_{d}" for d in range(ndim)),
            *(f"scaled_second_derivative_{d}" for d in range(ndim)),
            "scaled_reaction", "scaled_reaction_derivative",
            *(f"diffusion_timescale_ratio_{d}" for d in range(ndim)),
            "reaction_timescale_ratio",
            *(f"geometry_spacing_ratio_{d}" for d in range(ndim)))


def _validate(u: Tensor, geometry, t_ref: float, U_ref: float) -> None:
    feature_count(geometry.ndim)
    if u.ndim != geometry.ndim + 2 or u.shape[1] != 1:
        raise ValueError("scalar state must have shape [batch, 1, *grid]")
    if tuple(u.shape[2:]) != tuple(geometry.grid):
        raise ValueError("state grid does not match geometry")
    if not u.is_floating_point():
        raise TypeError("state must be floating point")
    if t_ref <= 0 or U_ref <= 0:
        raise ValueError("fixed physical reference scales must be positive")


def _assemble(center: Tensor, gradients: list[Tensor], second: list[Tensor],
              equation, geometry, t_ref: float, U_ref: float) -> Tensor:
    dtype = torch.float64 if center.dtype == torch.float64 else torch.float32
    center = center.to(dtype)
    gradients = [v.to(dtype) for v in gradients]
    second = [v.to(dtype) for v in second]
    reaction = equation.reaction_rate * center * (1.0 - center)
    derivative = equation.reaction_rate * (1.0 - 2.0 * center)
    constant = lambda value: torch.full_like(center, float(value))
    fields = [center / U_ref, *gradients, *second,
              t_ref * reaction / U_ref, t_ref * derivative,
              *(constant(equation.kappa * t_ref / dx**2) for dx in geometry.dx),
              constant(equation.reaction_rate * t_ref),
              *(constant(dx / length) for dx, length in zip(geometry.dx, geometry.lengths))]
    return torch.stack(fields, dim=-1)


def extract_features(u: Tensor, equation, geometry, *, t_ref: float = 1.0,
                     U_ref: float = 1.0) -> Tensor:
    """Construct scaled state/derivative/physics features at radius one.

    dx*D_d u and dx²*D_dd u use periodic centered differences. Physical
    coefficients and fixed geometry enter through dimensionless ratios. All
    state derivatives remain differentiable during student rollouts.
    """
    _validate(u, geometry, t_ref, U_ref)
    center = u[:, 0]
    center = center if center.dtype == torch.float64 else center.float()
    gradients, second = [], []
    for axis in range(geometry.ndim):
        plus = torch.roll(center, shifts=-1, dims=axis + 1)
        minus = torch.roll(center, shifts=1, dims=axis + 1)
        gradients.append((plus - minus) / (2.0 * U_ref))
        second.append((plus - 2.0 * center + minus) / U_ref)
    return _assemble(center, gradients, second, equation, geometry, t_ref, U_ref)


def feature_chunk(u: Tensor, equation, geometry, start: int, stop: int, *,
                  t_ref: float = 1.0, U_ref: float = 1.0) -> Tensor:
    """Return flattened [start:stop] features via bounded periodic gathers."""
    _validate(u, geometry, t_ref, U_ref)
    flat = u[:, 0].reshape(-1)
    if start < 0 or stop < start or stop > flat.numel():
        raise ValueError("feature chunk outside flattened state")
    indices = torch.arange(start, stop, device=u.device)
    center = flat[indices]
    center = center if center.dtype == torch.float64 else center.float()
    gradients, second = [], []
    stride = 1
    for axis in reversed(range(geometry.ndim)):
        count = geometry.grid[axis]
        position = (indices // stride) % count
        plus_index = indices + torch.where(position == count - 1, -(count - 1) * stride, stride)
        minus_index = indices + torch.where(position == 0, (count - 1) * stride, -stride)
        plus, minus = flat[plus_index].to(center.dtype), flat[minus_index].to(center.dtype)
        gradients.insert(0, (plus - minus) / (2.0 * U_ref))
        second.insert(0, (plus - 2.0 * center + minus) / U_ref)
        stride *= count
    return _assemble(center, gradients, second, equation, geometry, t_ref, U_ref)


def iter_feature_chunks(u: Tensor, equation, geometry, *, chunk_size: int = 32768,
                        t_ref: float = 1.0, U_ref: float = 1.0) -> Iterator[tuple[int, int, Tensor]]:
    if chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    count = u.numel()
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        yield start, stop, feature_chunk(u, equation, geometry, start, stop,
                                        t_ref=t_ref, U_ref=U_ref)
