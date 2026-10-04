"""Shared, confluent temporal responses for bounded research experiments.

The latent responses avoid differencing almost equal decay modes. Their
positive rates and equilibrium envelope do not imply nonlinear solver
stability. In particular, these additive models do not preserve state bounds.
"""
from __future__ import annotations

import math
from functools import lru_cache

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from reference.temporal_core import psi3
from tdn.numerics.operators import broadcast_h
from tdn.numerics.splitting import split_step

from .common import LocalEncoder, commuting_gate


def _powers(x: Tensor, count: int) -> Tensor:
    """0,...,count-1 powers by binary products, with regular zero derivatives."""
    value = torch.stack((torch.ones_like(x), x), dim=-1)
    multiplier = x * x
    while value.shape[-1] < count:
        value = torch.cat((value, value * multiplier.unsqueeze(-1)), dim=-1)
        multiplier = multiplier * multiplier
    return value[..., :count]


@lru_cache(maxsize=24)
def _series_coefficients(orders: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    # Powers are of z/8: neither coefficients nor powers overflow at z<=8.
    return torch.tensor([[(n + 1) * (n + 2) * 8.0**n / math.factorial(n + k + 3)
                          for k in range(orders)] for n in range(49)],
                        device=device, dtype=dtype)


def confluent_basis(rho: Tensor, tau: Tensor, *, orders: int = 4) -> Tensor:
    r"""Evaluate J_0,...,J_(orders-1) without subtracting adjacent rates.

    J_k = int_0^tau exp[-rho*(tau-s)] (tau-s)^k s^2 ds / k!.

    Inputs broadcast and the final output axis indexes k. FP64 is preserved;
    lower precision inputs are promoted to FP32. The caller must supply finite
    nonnegative rho and tau. All inactive branches remain finite at zero.

    For z=rho*tau <= 8 a positive shifted Taylor series avoids cancellation.
    For larger z, incomplete-gamma moments use a bounded exponential polynomial
    and inverse powers. This is a differentiable correctness implementation,
    not a claim that its small shared basis evaluation is a fused GPU kernel.
    """
    if type(orders) is not int or not 1 <= orders <= 5:
        raise ValueError("orders must be an integer between one and five")
    if not rho.is_floating_point() or not tau.is_floating_point():
        raise TypeError("rho and tau must be floating-point tensors")
    dtype = torch.float64 if torch.float64 in (rho.dtype, tau.dtype) else torch.float32
    rho, tau = torch.broadcast_tensors(rho.to(dtype), tau.to(device=rho.device, dtype=dtype))
    z = rho * tau
    small = z.clamp(min=0.0, max=8.0)
    # At z<=8, 48 terms leave substantially less than FP64 rounding error.
    # This tiny shared reduction deliberately avoids autocast/TF32 matmul:
    # encoder precision choices must not silently lower the temporal accuracy.
    series = (_powers(small / 8.0, 49).unsqueeze(-1)
              * _series_coefficients(orders, z.device, dtype)).sum(dim=-2)
    small_value = torch.exp(-small).unsqueeze(-1) * series

    large = z.clamp(min=8.0)
    inv = large.reciprocal()
    # For z>80 this polynomial tail is below working precision even in FP64.
    # Capping only this tail avoids inf*0 for large finite FP32 rates; the
    # inverse powers retain the actual z and its derivatives.
    tail_z = large.clamp(max=80.0)
    term = torch.exp(-tail_z)
    tail_sum = term
    moments = [inv * (1.0 - tail_sum)]
    for m in range(1, orders + 2):
        term = term * tail_z / m
        tail_sum = tail_sum + term
        moments.append(math.factorial(m) * inv.pow(m + 1) * (1.0 - tail_sum))
    moments = torch.stack(moments, dim=-1)
    factorials = z.new_tensor([math.factorial(k) for k in range(orders)])
    large_value = (moments[..., :orders] - 2.0 * moments[..., 1:orders + 1]
                   + moments[..., 2:orders + 2]) / factorials
    value = torch.where((z <= 8.0).unsqueeze(-1), small_value, large_value)
    powers = torch.stack([tau.pow(k + 3) for k in range(orders)], dim=-1)
    return powers * value


class _SharedBasisTDN(nn.Module):
    """Common h-independent encoder and exact commuting-limit factor."""

    def __init__(self, *, width: int, t_ref: float, U_ref: float,
                 decay_multiplier: float, ndim: int) -> None:
        super().__init__()
        if not math.isfinite(t_ref) or t_ref <= 0:
            raise ValueError("t_ref must be finite and positive")
        if not math.isfinite(U_ref) or U_ref <= 0:
            raise ValueError("U_ref must be finite and positive")
        if not math.isfinite(decay_multiplier) or decay_multiplier < 0:
            raise ValueError("decay_multiplier must be finite and nonnegative")
        self.t_ref, self.U_ref = float(t_ref), float(U_ref)
        self.decay_multiplier = float(decay_multiplier)
        self.encoder = LocalEncoder(4, width=width, t_ref=t_ref, U_ref=U_ref, ndim=ndim)

    def set_normalization(self, mean: Tensor, std: Tensor) -> None:
        """Install training-parent-only feature statistics."""
        self.encoder.set_normalization(mean, std)

    def encode(self, u: Tensor, equation, geometry) -> Tensor:
        amplitudes = self.encoder(u, equation, geometry)
        return amplitudes * commuting_gate(u, equation, geometry,
                                          t_ref=self.t_ref, U_ref=self.U_ref)

    def temporal_basis(self, tau: Tensor, equation) -> Tensor:
        raise NotImplementedError

    def forward(self, u: Tensor, h: float | Tensor, equation, geometry) -> Tensor:
        amplitudes = self.encode(u, equation, geometry)
        step = broadcast_h(h, u)
        # Evaluate four functions once per parent horizon, never per grid cell.
        tau = step.reshape(-1) / self.t_ref
        basis = self.temporal_basis(tau, equation)
        shape = (basis.shape[0], 4, *([1] * geometry.ndim))
        correction = self.U_ref * (amplitudes * basis.reshape(shape)).sum(dim=1, keepdim=True)
        return split_step(u, step, equation, geometry) + correction


class ConfluentTDN(_SharedBasisTDN):
    """Four confluent moments and one learned rate shared by the whole model.

    The initial experiment shares that rate across all its training regimes,
    a stricter restriction than the earlier per-parent oracle fits. The rate,
    amplitude encoder and normalization use training parents only.
    ``decay_multiplier=0`` is the undamped ablation.
    """

    def __init__(self, width: int = 16, t_ref: float = 1.0, U_ref: float = 1.0,
                 decay_multiplier: float = 1.0, rho_init: float = 10.0,
                 ndim: int = 2) -> None:
        super().__init__(width=width, t_ref=t_ref, U_ref=U_ref,
                         decay_multiplier=decay_multiplier, ndim=ndim)
        if not math.isfinite(rho_init) or rho_init <= 0:
            raise ValueError("rho_init must be finite and positive")
        # Stable inverse softplus; no cap on the learned positive rate.
        raw = rho_init + math.log(-math.expm1(-rho_init))
        self.raw_rho = nn.Parameter(torch.tensor(raw))

    @property
    def rho(self) -> Tensor:
        return F.softplus(self.raw_rho)

    def temporal_basis(self, tau: Tensor, equation) -> Tensor:
        gamma = self.decay_multiplier * equation.reaction_rate * self.t_ref
        return torch.exp(-gamma * tau).unsqueeze(-1) * confluent_basis(self.rho, tau)


class FixedDecayTDN(_SharedBasisTDN):
    """Four ordinary fixed-rate modes with a matched physical decay envelope.

    This mandatory control separates equilibrium damping from confluence.
    ``decay_multiplier`` supports the declared 0, 1 and 2 ablations.
    """

    def __init__(self, width: int = 16, t_ref: float = 1.0, U_ref: float = 1.0,
                 decay_multiplier: float = 2.0, ndim: int = 2) -> None:
        super().__init__(width=width, t_ref=t_ref, U_ref=U_ref,
                         decay_multiplier=decay_multiplier, ndim=ndim)
        self.register_buffer("fixed_rates", torch.tensor([0.1, 1.0, 10.0, 100.0]))

    def temporal_basis(self, tau: Tensor, equation) -> Tensor:
        gamma = self.decay_multiplier * equation.reaction_rate * self.t_ref
        return (torch.exp(-gamma * tau) * tau.pow(3)).unsqueeze(-1) * psi3(
            tau.unsqueeze(-1) * self.fixed_rates)
