"""Prepared classical controls and a Parseval evaluation of the raw GL mean.

The spectral mean is the mean of the finite-amplitude *quadratic* GL defect,
not the exact nonlinear mean. It is applied additively, before any bounding.
Preparation freezes one scalar horizon, geometry, dtype and device. Only
state-independent tensors are retained; every call validates its new state
and recomputes its mean, logistic Jacobians and reaction weights. These are
forward numerical controls, with no prepared-time/autograd contract.

Work counts successful operations over a whole batch, including operations
completed before a later failure. Setup performs no transforms. Cached tensor
bytes measure retained coefficient storage, not allocator/peak process memory.
The spectral mean uses FP64 coefficient/reduction arithmetic for FP32 inputs
to protect the cancellation of its O(h^2) terms; its one FFT and final output
retain the input dtype. This mixed arithmetic is included in measured work.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from tdn.numerics.subflows import diffusion_eigenvalues
from tdn.numerics.types import Equation, Geometry
from tdn.research.interaction import (
    Work, _GL, _center, _counts, _finish, _nodes, _validate, phi,
)


VARIANTS = ("strang", "etdrk2", "etdrk4", "gl3", "gl5",
            "gl3_mean_full", "gl3_mean_spectral")


@dataclass(frozen=True)
class _ReactionConstants:
    time: Tensor
    q: Tensor
    change: Tensor
    small: Tensor

    @classmethod
    def at(cls, time: Tensor, rate: float) -> _ReactionConstants:
        return cls(time, torch.exp(-rate * time), torch.expm1(-rate * time),
                   rate * time < .5)

    def denominator(self, c: Tensor) -> Tensor:
        return torch.where(self.small, 1 + (1 - c) * self.change,
                           c + (1 - c) * self.q)

    def jacobian(self, c: Tensor, rate: float) -> Tensor:
        denominator = self.denominator(c)
        degenerate = (denominator == 0) & (c == 0)
        safe = torch.where(degenerate, torch.ones_like(denominator), denominator)
        value = (self.q / safe) / safe
        tail = (self.q == 0) & (c > 0) & (c < 1)
        log_safe = torch.where(tail, denominator, torch.ones_like(denominator))
        value = torch.where(tail, torch.exp(-rate * self.time - 2 * torch.log(log_safe)), value)
        return torch.where(degenerate, torch.full_like(value, float("inf")), value)


class _Coefficients:
    def __init__(self, example: Tensor, step: Tensor, equation: Equation,
                 geometry: Geometry, variant: str, nodes: int | None = None):
        self.step = step
        self.zero_time = bool((step == 0).all())
        self.axes = tuple(range(2, example.ndim))
        self.rate = equation.reaction_rate
        self.kappa = equation.kappa
        self.variant = variant
        self.nodes = nodes if nodes is not None else (5 if variant == "gl5" else 3)
        if self.zero_time:
            return
        eigenvalues = diffusion_eigenvalues(geometry, example)
        spectrum = equation.kappa * eigenvalues
        # Match the direct split/ETD multiplication order, including FP32
        # rounding. GL's variation uses its separately formed spectrum.
        z = step * equation.kappa * eigenvalues
        self.E = torch.exp(z)
        if variant.startswith("etdrk"):
            if self.rate:
                p1, p2 = phi(z, 1), phi(z, 2)
                if variant == "etdrk4":
                    p3 = phi(z, 3)
                    self.final_weights = (p1 - 3 * p2 + 4 * p3,
                                          2 * p2 - 4 * p3, -p2 + 4 * p3)
                    self.E2 = torch.exp(z / 2)
                    self.Q = (step / 2) * phi(z / 2, 1)
                else:
                    self.b1, self.b2 = step * p1, step * p2
            return
        self.half = _ReactionConstants.at(step / 2, self.rate)
        self.split_q = self.half.q
        if variant == "strang" or not self.rate or not self.kappa:
            return
        spectral = variant == "gl3_mean_spectral"
        if spectral:
            # Keep the split arithmetic unchanged while protecting the raw
            # mean's cancellation. The expensive transform remains FP32.
            step = step.to(torch.float64)
            spectrum = spectrum.to(torch.float64)
            self.half = _ReactionConstants.at(step / 2, self.rate)
        self.full = _ReactionConstants.at(step, self.rate)
        self.weight_factor = (step / 2) * phi(-self.rate * (step / 2))
        self.quadrature = []
        locations, weights = _GL[self.nodes]
        for x, weight in zip(locations, weights):
            s = step * ((1 + x) / 2)
            self.quadrature.append((step * weight / 2,
                                    _ReactionConstants.at(s, self.rate),
                                    torch.expm1((2 * s if spectral else s) * spectrum),
                                    None if spectral else torch.exp((step - s) * spectrum)))
        self.endpoint_increment = torch.expm1((2 * step if spectral else step) * spectrum)

    def weight(self, c: Tensor) -> Tensor:
        dhalf, dfull = self.half.denominator(c), self.full.denominator(c)
        degenerate = (dfull == 0) & (c == 0)
        safe_half = torch.where(degenerate, torch.ones_like(dhalf), dhalf)
        safe_full = torch.where(degenerate, torch.ones_like(dfull), dfull)
        value = (self.half.q / safe_half) * (self.weight_factor / safe_full)
        tail = ((self.half.q == 0) | ~torch.isfinite(value)) & (c > 0) & (c < 1)
        log_half = torch.where(tail, dhalf, torch.ones_like(dhalf))
        log_full = torch.where(tail, dfull, torch.ones_like(dfull))
        tail_time = torch.where(tail, self.full.time / 2, torch.ones_like(self.full.time))
        log_value = (-self.rate * tail_time + torch.log(-torch.expm1(-self.rate * tail_time))
                     - math.log(self.rate) - torch.log(log_half) - torch.log(log_full))
        value = torch.where(tail, torch.exp(log_value), value)
        return torch.where(degenerate, torch.full_like(value, float("inf")), value)


class _Evaluation:
    def __init__(self, coefficients: _Coefficients, counts: dict[str, int]):
        self.coefficients = coefficients
        self.counts = counts

    def fft(self, value: Tensor) -> Tensor:
        result = torch.fft.fftn(value, dim=self.coefficients.axes)
        self.counts["fft_forward"] += 1
        return result

    def inverse(self, value: Tensor) -> Tensor:
        result = torch.fft.ifftn(value, dim=self.coefficients.axes).real
        self.counts["fft_inverse"] += 1
        return result

    def split(self, u: Tensor) -> Tensor:
        p = self.coefficients

        def reaction(value):
            if not p.rate:
                return value + p.step * 0
            denominator = value + (1 - value) * p.split_q
            safe = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
            result = value / safe
            self.counts["reaction_evaluations"] += 1
            return result

        first = reaction(u)
        second = self.inverse(self.fft(first) * p.E) if p.kappa else first + p.step * 0
        return reaction(second)

    def defect(self, u: Tensor, *, spectral: bool) -> Tensor:
        p = self.coefficients
        c, v, uniform = _center(u)
        if not p.kappa or not p.rate or p.zero_time or bool(uniform.all()):
            return c * 0 if spectral else u * 0
        safe_c = torch.where(uniform, torch.full_like(c, .5), c)
        vhat = self.fft(v)
        if spectral:
            cells = math.prod(u.shape[2:])
            power = (vhat.real.double().square() + vhat.imag.double().square()) / cells**2
            safe_c = safe_c.double()
            self.counts["nonlinear_evaluations"] += 1

            def variation(increment):
                return (power * increment).sum(dim=p.axes, keepdim=True)
        else:
            square = v.square()
            self.counts["nonlinear_evaluations"] += 1
            square_hat = self.fft(square)

            def variation(increment):
                dv = self.inverse(increment * vhat)
                dsquare = self.inverse(increment * square_hat)
                answer = 2 * v * dv + dv.square() - dsquare
                self.counts["nonlinear_evaluations"] += 1
                return answer

        integral = torch.zeros_like(c if spectral else u)
        for weight, reaction, increment, transport in p.quadrature:
            value = variation(increment)
            if not spectral:
                value = self.inverse(transport * self.fft(value))
            jacobian = reaction.jacobian(safe_c, p.rate)
            self.counts["reaction_jacobian_evaluations"] += 1
            integral = integral + weight * jacobian * value
            self.counts["quadrature_evaluations"] += 1
        reaction_weight = p.weight(safe_c)
        self.counts["reaction_weight_evaluations"] += 1
        integral = integral - reaction_weight * variation(p.endpoint_increment)
        jacobian = p.full.jacobian(safe_c, p.rate)
        self.counts["reaction_jacobian_evaluations"] += 1
        result = -p.rate * jacobian * integral
        return torch.where(uniform | (p.step == 0), torch.zeros_like(result), result).to(u.dtype)

    def etd(self, u: Tensor) -> Tensor:
        p = self.coefficients
        if not p.rate:
            return self.inverse(self.fft(u) * p.E) if p.kappa else u + p.step * 0
        fft = self.fft if p.kappa else lambda x: x
        inverse = self.inverse if p.kappa else lambda x: x

        def nonlinear(value):
            result = p.rate * value * (1 - value)
            self.counts["nonlinear_evaluations"] += 1
            return fft(result)

        initial, n0 = fft(u), nonlinear(u)
        if p.variant == "etdrk2":
            stage_hat = p.E * initial + p.b1 * n0
            n1 = nonlinear(inverse(stage_hat))
            return inverse(stage_hat + p.b2 * (n1 - n0))
        a_hat = p.E2 * initial + p.Q * n0
        na = nonlinear(inverse(a_hat))
        b_hat = p.E2 * initial + p.Q * na
        nb = nonlinear(inverse(b_hat))
        c_hat = p.E2 * a_hat + p.Q * (2 * nb - n0)
        nc = nonlinear(inverse(c_hat))
        w0, wab, wc = p.final_weights
        return inverse(p.E * initial + p.step * (w0 * n0 + wab * (na + nb) + wc * nc))


def _cache_tensors(value: Any):
    if isinstance(value, Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _cache_tensors(child)
    elif isinstance(value, (_Coefficients, _ReactionConstants)):
        for child in vars(value).values():
            yield from _cache_tensors(child)


class PreparedStep:
    """Reusable forward step with immutable scalar-time coefficient ownership.

    Batch size may change; the spatial grid, dtype and device may not. No
    example state or state-derived tensor is retained in this object.
    """

    def __init__(self, example: Tensor, h: float | Tensor, equation: Equation,
                 geometry: Geometry, variant: str):
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if not isinstance(example, Tensor):
            raise TypeError("example must be a Tensor")
        step = _validate(example, h, equation, geometry)
        if step.ndim != 0:
            raise ValueError("prepared h must be a scalar")
        if step.requires_grad:
            raise ValueError("prepared h must be frozen, without time gradients")
        self._dtype, self._device = example.dtype, example.device
        self._geometry, self._equation = geometry, equation
        self._coefficients = _Coefficients(example, step.detach().clone(), equation, geometry, variant)

    @property
    def metadata(self) -> dict[str, Any]:
        p = self._coefficients
        tensors = list(_cache_tensors(p))
        storages = {tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
                    for tensor in tensors}
        return {"variant": p.variant, "h": float(p.step), "dtype": str(self._dtype),
                "device": str(self._device), "grid": list(self._geometry.grid),
                "cached_tensor_bytes": sum(storages.values()),
                "cache_tensor_count": len({id(tensor) for tensor in tensors}),
                "state_dependent_cache": False, "validation_per_call": True,
                "spectral_reduction_dtype": "torch.float64" if p.variant == "gl3_mean_spectral" else None,
                "setup_fft_total": 0}

    def __call__(self, u: Tensor, work: Work | None = None) -> Tensor:
        if not isinstance(u, Tensor):
            raise TypeError("state must be a Tensor")
        if u.dtype != self._dtype or u.device != self._device:
            raise ValueError("state dtype and device must match the prepared example")
        p = self._coefficients
        _validate(u, p.step, self._equation, self._geometry)
        counts = _counts()
        evaluation = _Evaluation(p, counts)
        try:
            if p.zero_time:
                return u + p.step * 0
            if p.variant.startswith("etdrk"):
                return evaluation.etd(u)
            base = evaluation.split(u)
            if p.variant == "strang":
                return base
            increment = evaluation.defect(u, spectral=p.variant == "gl3_mean_spectral")
            if p.variant == "gl3_mean_full":
                increment = increment.mean(dim=p.axes, keepdim=True)
            return base + increment
        finally:
            _finish(work, counts)


def prepare_step(example: Tensor, h: float | Tensor, equation: Equation,
                 geometry: Geometry, variant: str) -> PreparedStep:
    """Prepare one fixed scalar step for an equally cached warmed comparison."""
    return PreparedStep(example, h, equation, geometry, variant)


def _spectral_call(u: Tensor, h: float | Tensor, equation: Equation,
                   geometry: Geometry, nodes: int, work: Work | None,
                   include_split: bool) -> Tensor:
    _nodes(nodes)
    if not isinstance(u, Tensor):
        raise TypeError("state must be a Tensor")
    step = _validate(u, h, equation, geometry)
    # Both quadratures use the same spectral algorithm; there is only a GL3
    # prepared mean variant because that is the frozen work-precision control.
    p = _Coefficients(u, step, equation, geometry, "gl3_mean_spectral", nodes=nodes)
    counts = _counts()
    evaluation = _Evaluation(p, counts)
    try:
        if p.zero_time:
            return u + step * 0 if include_split else u.mean(dim=p.axes, keepdim=True) * 0
        increment = evaluation.defect(u, spectral=True)
        if not include_split:
            return increment
        return torch.where(step == 0, u, evaluation.split(u) + increment)
    finally:
        _finish(work, counts)


def spectral_mean_defect(u: Tensor, h: float | Tensor, equation: Equation,
                         geometry: Geometry, *, nodes: int = 3,
                         work: Work | None = None) -> Tensor:
    """Raw GL mean, shape [batch,1,1,...], using one FFT and no inverse.

    With N spatial cells, P=abs(fftn(u-mean(u)))**2/N**2 and
    D(s)=sum(P*expm1(2*s*kappa*lambda)), this returns
    -r*J(h)*(sum_j omega_j*J(s_j)*D(s_j)-W2*D(h)).
    """
    return _spectral_call(u, h, equation, geometry, nodes, work, False)


def spectral_mean_step(u: Tensor, h: float | Tensor, equation: Equation,
                       geometry: Geometry, *, nodes: int = 3,
                       work: Work | None = None) -> Tensor:
    """Add the raw spectral mean defect to Strang (three active transforms)."""
    return _spectral_call(u, h, equation, geometry, nodes, work, True)
