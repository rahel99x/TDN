"""Transport a discrete splitting-defect correction through the paid FFT.

The convention is ``[F, G] = F'G - G'F`` for the semidiscrete vector
fields A = kappa*Delta_d and B = r*u*(1-u).  For the repository's
R(h/2) D(h) R(h/2) split, ``exact - split = h**3 * e3 + O(h**4)``.
The coefficient uses the *discrete* commutators below; substituting a
continuum product rule would give a different coefficient.

This is an experimental forward composition, not a positivity-preserving
method.  No clipping, inverse diffusion, fallback, or extra FFT is hidden
inside it.  The experiment engine must reject inadmissible stages/rollouts.
All state-dependent features remain live in the autograd graph.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tdn.numerics.operators import broadcast_h, check_shape
from tdn.numerics.splitting import split_step
from tdn.numerics.subflows import diffusion_step, reaction_step
from tdn.numerics.types import Equation, Geometry

from .common import LocalEncoder


@dataclass(frozen=True)
class CommutatorFields:
    """Physical fields with C in state/time**2, ZA/ZB/e3 in state/time**3."""

    C: Tensor
    ZA: Tensor
    ZB: Tensor
    e3: Tensor


def _difference_laplacian(u: Tensor, geometry: Geometry) -> Tensor:
    """Difference-first evaluation makes spatial constants exactly zero."""
    result = torch.zeros_like(u)
    for axis, dx in enumerate(geometry.dx, start=2):
        result = result + ((torch.roll(u, -1, axis) - u)
                           + (torch.roll(u, 1, axis) - u)) / dx**2
    return result


def commutator_fields(u: Tensor, equation: Equation,
                      geometry: Geometry) -> CommutatorFields:
    """Return C=[A,B], ZA=[A,C], ZB=[B,C], and the leading split defect.

    The C identity is a sum of neighbor differences squared, and C'[v]
    uses differences of v.  ZA has radius two, ZB radius one.  The formulas
    vanish exactly for a uniform field or either zero physical rate.
    Neighbor multiplicity on a two-point periodic axis is intentional and
    agrees with the actual central-difference Laplacian.
    """
    check_shape(u, geometry)
    kappa, rate = equation.kappa, equation.reaction_rate
    a = kappa * _difference_laplacian(u, geometry)
    b = rate * u * (1 - u)
    c = torch.zeros_like(u)
    c_prime_a = torch.zeros_like(u)
    c_prime_b = torch.zeros_like(u)
    for axis, dx in enumerate(geometry.dx, start=2):
        weight = kappa * rate / dx**2
        for shift in (-1, 1):
            du = torch.roll(u, shift, axis) - u
            da = torch.roll(a, shift, axis) - a
            db = torch.roll(b, shift, axis) - b
            c = c - weight * du.square()
            c_prime_a = c_prime_a - 2 * weight * du * da
            c_prime_b = c_prime_b - 2 * weight * du * db
    za = kappa * _difference_laplacian(c, geometry) - c_prime_a
    zb = rate * (1 - 2 * u) * c - c_prime_b
    return CommutatorFields(c, za, zb, -za / 12 - zb / 24)


def e3_anchor_step(u: Tensor, h: float | Tensor, equation: Equation,
                   geometry: Geometry) -> Tensor:
    """Classical, untrained post-split h**3 e3 control (no stage limiter)."""
    step = broadcast_h(h, u)
    fields = commutator_fields(u, equation, geometry)
    return split_step(u, step, equation, geometry) + step**3 * fields.e3


class LeadingDefectStep(nn.Module):
    """Parameter-free control; its local fourth order is not global fourth order."""

    def forward(self, u: Tensor, h: float | Tensor, equation: Equation,
                geometry: Geometry) -> Tensor:
        return e3_anchor_step(u, h, equation, geometry)


class TransportTDN(nn.Module):
    """Anchored early injection and late bypass with two shared damping rates.

    The h-independent encoder has five heads: an early/late mixing logit
    and four signed bounded dimensionless coefficients.  With tau=h/t_ref,
    q_in = h**3/(1+lambda_in*tau)**4
           * (alpha*e3 + tau*(a1*ZA + a2*ZB)),
    and the analogous late field uses (1-alpha), b1, b2, lambda_late.
    Fixed U_ref normalizes encoder inputs; it cancels from the physical
    commutator basis after nondimensionalization.

    A zero head is deliberately *anchored*, not the original split:
    alpha=1/2 and a1=a2=b1=b2=0.  Both positive rates are global parameters,
    never functions of h or of the cell.  An exact cubic local correction
    alone generally gives third-order global convergence.
    """

    def __init__(self, width: int = 16, t_ref: float = 1.0,
                 U_ref: float = 1.0, *, coefficient_bound: float = 4.0,
                 damping_rate: float = 1.0, ndim: int = 2) -> None:
        super().__init__()
        for name, value in (("t_ref", t_ref), ("U_ref", U_ref),
                            ("coefficient_bound", coefficient_bound),
                            ("damping_rate", damping_rate)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.t_ref = float(t_ref)
        self.U_ref = float(U_ref)
        self.coefficient_bound = float(coefficient_bound)
        self.encoder = LocalEncoder(outputs=5, width=width,
                                    t_ref=t_ref, U_ref=U_ref, ndim=ndim)
        # Stable inverse softplus, also for initialization far from one.
        inverse = damping_rate + math.log(-math.expm1(-damping_rate))
        self.raw_damping_rates = nn.Parameter(torch.full((2,), inverse))

    @property
    def damping_rates(self) -> Tensor:
        return F.softplus(self.raw_damping_rates)

    def set_normalization(self, mean: Tensor, std: Tensor) -> None:
        """Install statistics fitted only on the experiment's training parents."""
        self.encoder.set_normalization(mean, std)

    def _apply_step(self, u: Tensor, h: float | Tensor, equation: Equation,
                    geometry: Geometry) -> tuple[Tensor, dict[str, Tensor]]:
        check_shape(u, geometry)
        step = broadcast_h(h, u)
        tau = step / self.t_ref
        heads = self.encoder(u, equation, geometry)
        alpha = torch.sigmoid(heads[:, :1])
        coefficients = self.coefficient_bound * torch.tanh(heads[:, 1:])
        a1, a2, b1, b2 = coefficients.split(1, dim=1)
        fields = commutator_fields(u, equation, geometry)
        rates = self.damping_rates.to(u)
        early_weight = step**3 / (1 + rates[0] * tau)**4
        late_weight = step**3 / (1 + rates[1] * tau)**4
        q_in = early_weight * (alpha * fields.e3
                               + tau * (a1 * fields.ZA + a2 * fields.ZB))
        q_late = late_weight * ((1 - alpha) * fields.e3
                                + tau * (b1 * fields.ZA + b2 * fields.ZB))
        first = reaction_step(u, step / 2, equation)
        injected = first + q_in
        middle = diffusion_step(injected, step, equation, geometry)
        final = reaction_step(middle, step / 2, equation)
        output = final + q_late
        return output, {"reaction_first": first, "injected": injected,
                        "diffused": middle, "reaction_final": final,
                        "output": output}

    def forward(self, u: Tensor, h: float | Tensor, equation: Equation,
                geometry: Geometry) -> Tensor:
        return self._apply_step(u, h, equation, geometry)[0]

    def audit_step(self, u: Tensor, h: float | Tensor, equation: Equation,
                   geometry: Geometry) -> tuple[Tensor, dict[str, Tensor]]:
        """Expose every physical stage without extra transforms or clipping.

        The caller checks finite values and interval bounds using its declared
        floating-point tolerance.  Returning stages makes the audit explicit
        without device synchronization or silently detaching training graphs.
        """
        return self._apply_step(u, h, equation, geometry)
