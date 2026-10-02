"""Exact logistic and exact discrete periodic diffusion subflows.

FFT diffusion is a global operator. This code makes no finite-halo claim.
Logistic values are never clipped. Endpoint sensitivities at u=0 are exp(t):
if that sensitivity is unrepresentable it is explicitly infinite, although
the endpoint value remains exactly zero. Derivative audits use representable
sensitivities and the interior of the declared admissible interval.
"""
from __future__ import annotations

from functools import lru_cache

import torch
from torch import Tensor

from .operators import broadcast_h, check_shape
from .types import Equation, Geometry


def _reaction_derivatives(u: Tensor, t: Tensor, out: Tensor) -> tuple[Tensor, Tensor]:
    q = torch.exp(-t)
    denominator = u + (1 - u) * q
    degenerate = (q == 0) & (u == 0)
    safe = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
    # Sequential division avoids squaring a tiny denominator to zero.
    du = (q / safe) / safe
    du = torch.where(degenerate, torch.full_like(du, float("inf")), du)
    return du, out * (1 - out)


def _derivative_product(coefficient: Tensor, tangent: Tensor) -> Tensor:
    # A zero direction has zero sensitivity even at a representational overflow.
    safe_coefficient = torch.where((tangent == 0) & torch.isinf(coefficient),
                                   torch.zeros_like(coefficient), coefficient)
    return safe_coefficient * tangent


class _LogisticFlow(torch.autograd.Function):
    """Analytic backward/JVP keeps endpoint derivatives numerically deliberate."""
    generate_vmap_rule = True

    @staticmethod
    def forward(u: Tensor, t: Tensor) -> Tensor:
        q = torch.exp(-t)
        denominator = u + (1 - u) * q
        # Only the exact underflow degeneracy is regularized internally.
        safe = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
        return u / safe

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        u, t = inputs
        ctx.save_for_backward(u, t, output)
        ctx.save_for_forward(u, t, output)

    @staticmethod
    def backward(ctx, grad: Tensor) -> tuple[Tensor, Tensor]:
        u, t, out = ctx.saved_tensors
        du, dt = _reaction_derivatives(u, t, out)
        return (_derivative_product(du, grad).sum_to_size(u.shape),
                _derivative_product(dt, grad).sum_to_size(t.shape))

    @staticmethod
    def jvp(ctx, tangent_u: Tensor | None, tangent_t: Tensor | None) -> Tensor:
        u, t, out = ctx.saved_tensors
        du, dt = _reaction_derivatives(u, t, out)
        answer = torch.zeros_like(out)
        if tangent_u is not None:
            answer = answer + _derivative_product(du, tangent_u)
        if tangent_t is not None:
            answer = answer + _derivative_product(dt, tangent_t)
        return answer


def reaction_step(u: Tensor, h: float | Tensor, equation: Equation) -> Tensor:
    """Exact R_h on [0,1]; supports autograd, gradgrad, and torch.func JVP."""
    if equation.reaction_rate == 0:
        return u + broadcast_h(h, u) * 0
    return _LogisticFlow.apply(u, equation.reaction_rate * broadcast_h(h, u))


@lru_cache(maxsize=32)
def _eigenvalues_cached(geometry: Geometry, device: str, dtype: torch.dtype) -> Tensor:
    """Cache geometry eigenvalues, not h-dependent exponential multipliers."""
    result = torch.zeros(geometry.grid, device=device, dtype=dtype)
    for axis, (n, dx) in enumerate(zip(geometry.grid, geometry.dx)):
        k = torch.arange(n, device=device, dtype=dtype)
        eigenvalues = -4 * torch.sin(torch.pi * k / n).square() / dx**2
        shape = [1] * geometry.ndim
        shape[axis] = n
        result = result + eigenvalues.reshape(shape)
    return result


def diffusion_eigenvalues(geometry: Geometry, u: Tensor) -> Tensor:
    return _eigenvalues_cached(geometry, str(u.device), u.dtype)


def diffusion_step(u: Tensor, h: float | Tensor, equation: Equation,
                   geometry: Geometry) -> Tensor:
    check_shape(u, geometry)
    step = broadcast_h(h, u)
    if equation.kappa == 0:
        return u + step * 0
    eigenvalues = diffusion_eigenvalues(geometry, u)
    multiplier = torch.exp(equation.kappa * step * eigenvalues)
    axes = tuple(range(2, u.ndim))
    return torch.fft.ifftn(torch.fft.fftn(u, dim=axes) * multiplier, dim=axes).real
