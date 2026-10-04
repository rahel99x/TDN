"""Research corrections in logistic coordinates, with explicit controls.

The clock core has an intentional boundary expressivity limitation: a finite
O(h**3) clock perturbation cannot supply an O(h**3) state correction at an
initially zero cell whose split intermediate is O(h). ``HybridReactionTDN``
is a separate experimental extension, not a remedy with a uniform order
guarantee. Neither interval preservation nor these constructions imply a
nonlinear stability theorem or an efficiency advantage.

All encoders are independent of h. Differentiate the complete forward map to
obtain time derivatives; additive-model frozen temporal jets do not apply.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from tdn.numerics.operators import broadcast_h, check_shape
from tdn.numerics.subflows import diffusion_step, reaction_step
from tdn.numerics.types import Equation, Geometry

from .common import LocalEncoder, commuting_gate


def _clock_terms(u: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    # Keep both inactive exponentials safe, including during backward/JVP.
    positive = t >= 0
    q = torch.exp(torch.where(positive, -t, t))
    denominator = torch.where(positive, u + (1 - u) * q,
                              (1 - u) + u * q)
    safe = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
    return positive, q, safe


class _SignedLogisticFlow(torch.autograd.Function):
    """Stable signed clock with analytic endpoint sensitivities.

    Positive clocks use precisely the arithmetic of the existing logistic
    subflow. Negative clocks use exp(t), so polynomial control failures are
    not obscured by a gratuitous exp(-t) overflow. Unrepresentable endpoint
    sensitivities remain infinite, just as in the physical subflow.
    """

    generate_vmap_rule = True

    @staticmethod
    def forward(u: Tensor, t: Tensor) -> Tensor:
        positive, q, safe = _clock_terms(u, t)
        value = torch.where(positive, u / safe, (u * q) / safe)
        # At t << 0, u=1 gives the exact underflow degeneracy 0/0.
        return torch.where((~positive) & (q == 0) & (u == 1), u, value)

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        u, t = inputs
        ctx.save_for_backward(u, t, output)
        ctx.save_for_forward(u, t, output)

    @staticmethod
    def _derivatives(u: Tensor, t: Tensor, out: Tensor) -> tuple[Tensor, Tensor]:
        positive, q, safe = _clock_terms(u, t)
        du = (q / safe) / safe
        degenerate = (q == 0) & torch.where(positive, u == 0, u == 1)
        du = torch.where(degenerate, torch.full_like(du, float("inf")), du)
        return du, out * (1 - out)

    @staticmethod
    def _product(coefficient: Tensor, tangent: Tensor) -> Tensor:
        coefficient = torch.where((tangent == 0) & torch.isinf(coefficient),
                                  torch.zeros_like(coefficient), coefficient)
        return coefficient * tangent

    @staticmethod
    def backward(ctx, grad: Tensor) -> tuple[Tensor, Tensor]:
        u, t, out = ctx.saved_tensors
        du, dt = _SignedLogisticFlow._derivatives(u, t, out)
        return (_SignedLogisticFlow._product(du, grad).sum_to_size(u.shape),
                _SignedLogisticFlow._product(dt, grad).sum_to_size(t.shape))

    @staticmethod
    def jvp(ctx, tangent_u: Tensor | None, tangent_t: Tensor | None) -> Tensor:
        u, t, out = ctx.saved_tensors
        du, dt = _SignedLogisticFlow._derivatives(u, t, out)
        answer = torch.zeros_like(out)
        if tangent_u is not None:
            answer = answer + _SignedLogisticFlow._product(du, tangent_u)
        if tangent_t is not None:
            answer = answer + _SignedLogisticFlow._product(dt, tangent_t)
        return answer


def reaction_clock_step(w: Tensor, clock: Tensor) -> Tensor:
    """Exact logistic map for a dimensionless, potentially signed clock.

    Interval preservation assumes w is in [0,1]; FFT roundoff is never clipped.
    """
    return _SignedLogisticFlow.apply(w, clock)


def capacity_update(v: Tensor, increment: Tensor) -> Tensor:
    """Apply a signed bounded increment to a state in [0,1].

    P(v,d) = v + m*d/(m+abs(d)), with m the available directional capacity.
    The signed form has derivative one in d at d=0 for interior states;
    replacing it with two ReLUs can freeze zero-initialized amplitude heads.
    This map is generally C1, not C2, at the interior sign transition.
    """
    capacity = torch.where(increment >= 0, 1 - v, v)
    denominator = capacity + increment.abs()
    safe = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    # Divide the signed increment first: its ratio is in [-1,1]. Multiplying
    # increment*capacity first can round the eventual correction just beyond
    # capacity for very large increments, producing tiny negative states.
    return v + capacity * (increment / safe)


def interior_gate(u: Tensor, epsilon: float = 0.05) -> Tensor:
    """A fixed C1 convex gate, one on [epsilon,1-epsilon], zero at endpoints."""
    if not math.isfinite(epsilon) or not 0 < epsilon < 0.5:
        raise ValueError("interior gate epsilon must lie strictly between 0 and 0.5")

    def smoothstep(x: Tensor) -> Tensor:
        return torch.where(x <= 0, torch.zeros_like(x),
                           torch.where(x >= 1, torch.ones_like(x), x.square() * (3 - 2 * x)))

    return smoothstep(u / epsilon) * smoothstep((1 - u) / epsilon)


class _ReactionBase(nn.Module):
    def __init__(self, outputs: int, *, width: int = 16, t_ref: float = 1.0,
                 U_ref: float = 1.0, amplitude_cap: float = 8.0,
                 ndim: int = 2) -> None:
        super().__init__()
        for name, value in (("t_ref", t_ref), ("U_ref", U_ref),
                            ("amplitude_cap", amplitude_cap)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.t_ref, self.U_ref = float(t_ref), float(U_ref)
        self.amplitude_cap = float(amplitude_cap)
        self.encoder = LocalEncoder(outputs, width=width, t_ref=t_ref, U_ref=U_ref,
                                    ndim=ndim)

    def set_normalization(self, mean: Tensor, std: Tensor) -> None:
        """Install statistics fitted exclusively on training parents."""
        self.encoder.set_normalization(mean, std)

    def amplitudes(self, u: Tensor, equation: Equation, geometry: Geometry) -> Tensor:
        raw = self.encoder(u, equation, geometry)
        gate = commuting_gate(u, equation, geometry, t_ref=self.t_ref, U_ref=self.U_ref)
        return self.amplitude_cap * torch.tanh(raw) * gate

    def _middle(self, u: Tensor, h: float | Tensor, equation: Equation,
                geometry: Geometry) -> tuple[Tensor, Tensor]:
        check_shape(u, geometry)
        step = broadcast_h(h, u)
        middle = diffusion_step(reaction_step(u, step / 2, equation), step,
                                equation, geometry)
        return middle, step


class ReactionClockTDN(_ReactionBase):
    """Two bounded amplitude heads perturb the final physical reaction clock."""

    def __init__(self, width: int = 16, t_ref: float = 1.0, U_ref: float = 1.0,
                 amplitude_cap: float = 8.0, ndim: int = 2) -> None:
        super().__init__(2, width=width, t_ref=t_ref, U_ref=U_ref,
                         amplitude_cap=amplitude_cap, ndim=ndim)

    def _basis(self, step: Tensor, equation: Equation) -> tuple[Tensor, Tensor]:
        z = -torch.expm1(-equation.reaction_rate * step)
        return z.pow(3), z.pow(4)

    def forward(self, u: Tensor, h: float | Tensor, equation: Equation,
                geometry: Geometry) -> Tensor:
        amplitudes = self.amplitudes(u, equation, geometry)
        middle, step = self._middle(u, h, equation, geometry)
        cubic, quartic = self._basis(step, equation)
        eta = amplitudes[:, :1] * cubic + amplitudes[:, 1:2] * quartic
        return reaction_clock_step(middle, equation.reaction_rate * step / 2 + eta)


class AdditiveClockTDN(ReactionClockTDN):
    """Matched two-head/time-basis control using an additive state correction.

    This deliberately has no interval-preservation guarantee; failed state
    checks belong in reports. U_ref converts the dimensionless shift to state.
    """

    def forward(self, u: Tensor, h: float | Tensor, equation: Equation,
                geometry: Geometry) -> Tensor:
        amplitudes = self.amplitudes(u, equation, geometry)
        middle, step = self._middle(u, h, equation, geometry)
        cubic, quartic = self._basis(step, equation)
        eta = amplitudes[:, :1] * cubic + amplitudes[:, 1:2] * quartic
        return reaction_step(middle, step / 2, equation) + self.U_ref * eta


class PolynomialClockTDN(ReactionClockTDN):
    """Optional coordinate-matched control without the saturating time basis.

    The amplitudes are bounded but the polynomial clock is not. Long-time
    equilibrium behavior is therefore a testable failure mode of this control.
    """

    def _basis(self, step: Tensor, equation: Equation) -> tuple[Tensor, Tensor]:
        tau = step / self.t_ref
        return tau.pow(3), tau.pow(4)


class HybridReactionTDN(_ReactionBase):
    """Experimental boundary extension; retain separate results from the core.

    Four amplitude heads share one encoder. A fixed, h-independent gate blends
    the clock map with a decaying capacity correction. No uniform cubic-order
    guarantee is claimed when the available capacity is O(h**3) or smaller.
    """

    def __init__(self, width: int = 16, t_ref: float = 1.0, U_ref: float = 1.0,
                 amplitude_cap: float = 8.0, epsilon: float = 0.05,
                 ndim: int = 2) -> None:
        super().__init__(4, width=width, t_ref=t_ref, U_ref=U_ref,
                         amplitude_cap=amplitude_cap, ndim=ndim)
        if not math.isfinite(epsilon) or not 0 < epsilon < 0.5:
            raise ValueError("interior gate epsilon must lie strictly between 0 and 0.5")
        self.epsilon = float(epsilon)

    def forward(self, u: Tensor, h: float | Tensor, equation: Equation,
                geometry: Geometry) -> Tensor:
        amplitudes = self.amplitudes(u, equation, geometry)
        middle, step = self._middle(u, h, equation, geometry)
        z = -torch.expm1(-equation.reaction_rate * step)
        eta = amplitudes[:, :1] * z.pow(3) + amplitudes[:, 1:2] * z.pow(4)
        clock = reaction_clock_step(middle, equation.reaction_rate * step / 2 + eta)
        base = reaction_step(middle, step / 2, equation)
        tau = step / self.t_ref
        increment = (self.U_ref * torch.exp(-equation.reaction_rate * step)
                     * tau.pow(3) * (amplitudes[:, 2:3] + tau * amplitudes[:, 3:4]))
        boundary = capacity_update(base, increment)
        gate = interior_gate(u, self.epsilon)
        # Difference form retains bitwise base equivalence at zero amplitudes.
        return base + gate * (clock - base) + (1 - gate) * (boundary - base)
