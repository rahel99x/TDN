"""CPU-testable temporal units, not an implemented ADR solver or a CUDA benchmark.

Shapes: amplitudes [..., C, M], rates broadcastable to amplitudes,
        h scalar or broadcastable after the caller adds channel/mode axes.
Rates and amplitudes MUST be independent of the queried h for the stated jets.
All quantities may be nondimensional. A wrapper must restore physical units.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def psi3(x: Tensor) -> Tensor:
    """Integral int_0^1 exp(-x*(1-r))*r**2 dr for finite x >= 0.

    Preserve float64; promote lower precision to float32. The caller validates
    nonnegative finite inputs outside a compiled hot path. Both torch.where
    branches are finite, including at x=0, to avoid NaN gradients.
    This is a correctness reference; profile before replacing it with a kernel.
    """
    if not x.is_floating_point():
        raise TypeError("psi3 expects a real floating-point tensor")
    y = x if x.dtype == torch.float64 else x.float()
    cut = 0.5
    s = torch.clamp(y, min=0.0, max=cut)
    degree = 14
    value = torch.full_like(s, 2.0 * (-1.0)**degree / math.factorial(degree + 3))
    for n in range(degree - 1, -1, -1):
        value = value * s + 2.0 * (-1.0)**n / math.factorial(n + 3)
    # Inverse form avoids x**3 overflow at large x. Its small-x cancellation
    # is bypassed by the polynomial branch. Do not use an unclamped 0/0 branch.
    safe = torch.clamp(y, min=cut)
    inv = safe.reciprocal()
    large = inv * (1.0 - 2.0 * inv - 2.0 * torch.expm1(-safe) * inv.square())
    return torch.where(y <= cut, value, large)


def mode_response(amplitudes: Tensor, rates: Tensor, h: Tensor) -> Tensor:
    """Return each mode's value without reducing the last (mode) axis."""
    dtype = torch.float64 if amplitudes.dtype == torch.float64 else torch.float32
    a = amplitudes.to(dtype)
    lam = rates.to(device=a.device, dtype=dtype)
    step = h.to(device=a.device, dtype=dtype)
    return step.pow(3) * a * psi3(lam * step)


def temporal_defect(amplitudes: Tensor, rates: Tensor, h: Tensor) -> Tensor:
    return mode_response(amplitudes, rates, h).sum(dim=-1)


def temporal_jets(amplitudes: Tensor, rates: Tensor, h: Tensor) -> tuple[Tensor, ...]:
    """E, dE/dh, d2E/dh2, d3E/dh3, holding encoded parameters fixed."""
    z = mode_response(amplitudes, rates, h)
    a = amplitudes.to(z.dtype)
    lam = rates.to(device=z.device, dtype=z.dtype)
    step = h.to(device=z.device, dtype=z.dtype)
    first = -lam * z + step.square() * a
    second = -lam * first + 2.0 * step * a
    third = -lam * second + 2.0 * a
    return tuple(t.sum(dim=-1) for t in (z, first, second, third))


def anchored_amplitudes(free: Tensor, leading_defect: Tensor) -> Tensor:
    """Given M-1 free modes, enforce sum_m a_m = 3*e3 exactly in arithmetic.

    free [..., C, M-1]; leading_defect [..., C]. Floating-point cancellation
    must still be measured. This does not compute the correct e3 for a solver.
    """
    last = 3.0 * leading_defect - free.sum(dim=-1)
    return torch.cat((free, last.unsqueeze(-1)), dim=-1)


@dataclass(frozen=True)
class EncodedModes:
    amplitudes: Tensor
    rates: Tensor


class TemporalParameterMLP(nn.Module):
    """Small h-independent parameter generator used as an architectural reference.

    Input [..., F]. Outputs [..., C, M] amplitudes and [..., 1, M] rates.
    Zero amplitude initialization gives zero correction. Initial rate gradients
    are consequently zero; this is expected, not a broken autograd graph.
    No batch statistics, dropout, coordinates, or queried step enter by default.
    """
    def __init__(self, features: int, channels: int, modes: int = 4,
                 width: int = 64, anchored: bool = False) -> None:
        super().__init__()
        if min(features, channels, modes, width) <= 0:
            raise ValueError("all dimensions must be positive")
        if anchored and modes < 2:
            raise ValueError("this reference's learned anchor uses at least 2 modes")
        self.channels, self.modes, self.anchored = channels, modes, anchored
        self.body = nn.Sequential(nn.Linear(features, width), nn.SiLU(),
                                  nn.Linear(width, width), nn.SiLU())
        count = modes - 1 if anchored else modes
        self.amplitude = nn.Linear(width, channels * count)
        self.rate = nn.Linear(width, modes)
        nn.init.zeros_(self.amplitude.weight)
        nn.init.zeros_(self.amplitude.bias)
        nn.init.zeros_(self.rate.weight)
        desired = torch.logspace(-1, 2, modes)
        # Stable inverse softplus, desired = softplus(bias).
        with torch.no_grad():
            self.rate.bias.copy_(desired + torch.log(-torch.expm1(-desired)))

    def encode(self, features: Tensor, leading_defect: Tensor | None = None) -> EncodedModes:
        hidden = self.body(features)
        count = self.modes - 1 if self.anchored else self.modes
        # Preserve float64 for audits; explicit FP32 head output under BF16 AMP.
        raw_a = self.amplitude(hidden)
        raw_lam = self.rate(hidden)
        if raw_a.dtype != torch.float64:
            raw_a = raw_a.float()
            raw_lam = raw_lam.float()
        amplitudes = raw_a.reshape(*features.shape[:-1], self.channels, count)
        if self.anchored:
            if leading_defect is None:
                raise ValueError("anchored model requires the independently audited e3")
            amplitudes = anchored_amplitudes(amplitudes, leading_defect.to(amplitudes.dtype))
        rates = F.softplus(raw_lam).unsqueeze(-2)
        return EncodedModes(amplitudes, rates)

    def forward(self, features: Tensor, h: Tensor,
                leading_defect: Tensor | None = None) -> Tensor:
        modes = self.encode(features, leading_defect)
        return temporal_defect(modes.amplitudes, modes.rates, h)
