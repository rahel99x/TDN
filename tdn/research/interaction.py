"""Training-free quadratic interaction corrections and exponential RK controls.

The correction is the coupled quadratic variation about ``mean(u)`` minus
the R(h/2) D(h) R(h/2) quadratic variation on the *same discrete grid*.
It is a finite-amplitude truncation, not an exact nonlinear solver or a
mesh-uniform order claim. No learned coefficient or variation gate is used.

Work dictionaries count calls over the entire batched tensor, not individual
parents: every FFT/IFFT, quadrature node, exact logistic map, logistic
Jacobian/weight evaluation, and nonlinear squared-field/RHS evaluation.
Counters describe the actual path, including skipped exact limits, and add to
an optional supplied dictionary for rollout accounting. They are not a runtime
model. Validation is part of each public solver call. Exact-limit shortcuts
target forward evaluation; time derivatives through h=0 are not audited.
"""
from __future__ import annotations

import math
from collections.abc import MutableMapping

import torch
from torch import Tensor

from tdn.numerics.invariants import validate_state
from tdn.numerics.operators import broadcast_h, check_shape, laplacian
from tdn.numerics.splitting import split_step
from tdn.numerics.subflows import diffusion_eigenvalues, diffusion_step
from tdn.numerics.types import Equation, Geometry
from tdn.research.reaction import capacity_update


_GL = {
    3: ((-.7745966692414834, 0., .7745966692414834),
        (5 / 9, 8 / 9, 5 / 9)),
    5: ((-.9061798459386640, -.5384693101056831, 0.,
         .5384693101056831, .9061798459386640),
        (.2369268850561891, .4786286704993665, .5688888888888889,
         .4786286704993665, .2369268850561891)),
}
_COUNTERS = ("fft_forward", "fft_inverse", "fft_total",
             "quadrature_evaluations", "reaction_evaluations",
             "reaction_jacobian_evaluations", "reaction_weight_evaluations",
             "nonlinear_evaluations", "laplacian_evaluations")
Work = MutableMapping[str, int]


def _counts() -> dict[str, int]:
    return dict.fromkeys(_COUNTERS, 0)


def _finish(work: Work | None, counts: dict[str, int]) -> None:
    counts["fft_total"] = counts["fft_forward"] + counts["fft_inverse"]
    if work is not None:
        for key, value in counts.items():
            work[key] = work.get(key, 0) + value


def _nodes(nodes: int) -> None:
    if type(nodes) is not int or nodes not in _GL:
        raise ValueError("nodes must be the integer 3 or 5")


def _validate(u: Tensor, h: float | Tensor, equation: Equation,
              geometry: Geometry) -> Tensor:
    if not isinstance(equation, Equation) or not isinstance(geometry, Geometry):
        raise TypeError("equation and geometry must be Equation and Geometry")
    check_shape(u, geometry)
    if u.dtype not in (torch.float32, torch.float64):
        raise TypeError("interaction and ETD controls require FP32 or FP64")
    validate_state(u)
    step = broadcast_h(h, u)
    if not bool(torch.isfinite(step).all()) or bool((step < 0).any()):
        raise ValueError("h must be finite and nonnegative")
    return step


def phi(z: Tensor, order: int = 1) -> Tensor:
    """Stable phi_k(z)=sum_j z**j/(j+k)!, for k in {1,2,3}.

    A Taylor polynomial at |z| <= 1 avoids cancellation in FP32 and FP64;
    expm1 and recurrence handle the negative diffusion spectrum elsewhere.
    """
    if type(order) is not int or order not in (1, 2, 3):
        raise ValueError("phi order must be the integer 1, 2, or 3")
    if not isinstance(z, Tensor) or z.dtype not in (torch.float32, torch.float64):
        raise TypeError("phi requires a real FP32 or FP64 tensor")
    if not bool(torch.isfinite(z).all()):
        raise ValueError("phi argument must be finite")
    small = z.abs() <= 1
    # Only evaluate the polynomial on its convergence patch. This also avoids
    # overflowing an inactive branch for very stiff negative eigenvalues.
    zs = torch.where(small, z, torch.zeros_like(z))
    series = torch.full_like(zs, 1 / math.factorial(22 + order))
    for k in range(21, -1, -1):
        series = series * zs + 1 / math.factorial(k + order)
    safe = torch.where(small, -torch.ones_like(z), z)
    value = torch.expm1(safe) / safe
    for k in range(2, order + 1):
        value = (value - 1 / math.factorial(k - 1)) / safe
    return torch.where(small, series, value)


def _reaction_terms(c: Tensor, h: Tensor, rate: float) -> tuple[Tensor, Tensor]:
    q = torch.exp(-rate * h)
    # expm1 preserves the small-time change even when exp(-r h) rounds to 1.
    denominator = 1 + (1 - c) * torch.expm1(-rate * h)
    # At large times the sum above can cancel to zero for small c; use the
    # positive summands there. Both expressions are the same denominator.
    denominator = torch.where(rate * h < .5, denominator, c + (1 - c) * q)
    return q, denominator


def reaction_jacobian(c: Tensor, h: float | Tensor, rate: float) -> Tensor:
    """R'_h(c), using decaying exponentials and sequential divisions."""
    step = torch.as_tensor(h, dtype=c.dtype, device=c.device)
    if rate == 0:
        return torch.ones_like(c + step)
    q, denominator = _reaction_terms(c, step, rate)
    degenerate = (denominator == 0) & (c == 0)
    safe = torch.where(degenerate, torch.ones_like(denominator), denominator)
    value = (q / safe) / safe
    # A tiny q can underflow while q/D² remains representable for small c.
    # Only that tail needs the log-domain expression; the direct expression
    # preserves small-time accuracy around J=1.
    tail = (q == 0) & (c > 0) & (c < 1)
    log_safe = torch.where(tail, denominator, torch.ones_like(denominator))
    value = torch.where(tail, torch.exp(-rate * step - 2 * torch.log(log_safe)), value)
    return torch.where(degenerate, torch.full_like(value, float("inf")), value)


def reaction_weight(c: Tensor, h: float | Tensor, rate: float) -> Tensor:
    """Analytic W2=integral_(h/2)^h R'_s(c) ds, without subtracting W's.

    W2=(q_half-q_full)/(r D_half D_full). The expm1/phi form
    retains the r -> 0 limit without a division by a tiny reaction rate.
    """
    step = torch.as_tensor(h, dtype=c.dtype, device=c.device)
    if rate == 0:
        return torch.ones_like(c) * step / 2
    half = step / 2
    q_half, denominator_half = _reaction_terms(c, half, rate)
    _, denominator_full = _reaction_terms(c, step, rate)
    degenerate = (denominator_full == 0) & (c == 0)
    safe_half = torch.where(degenerate, torch.ones_like(denominator_half), denominator_half)
    safe_full = torch.where(degenerate, torch.ones_like(denominator_full), denominator_full)
    value = (q_half / safe_half) * (half * phi(-rate * half) / safe_full)
    tail = ((q_half == 0) | ~torch.isfinite(value)) & (c > 0) & (c < 1)
    log_half = torch.where(tail, denominator_half, torch.ones_like(denominator_half))
    log_full = torch.where(tail, denominator_full, torch.ones_like(denominator_full))
    tail_time = torch.where(tail, half, torch.ones_like(half))
    log_value = (-rate * tail_time + torch.log(-torch.expm1(-rate * tail_time))
                 - math.log(rate) - torch.log(log_half) - torch.log(log_full))
    value = torch.where(tail, torch.exp(log_value), value)
    return torch.where(degenerate, torch.full_like(value, float("inf")), value)


def _center(u: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    axes = tuple(range(2, u.ndim))
    c = u.mean(dim=axes, keepdim=True)
    first = u[(slice(None), slice(None)) + (slice(0, 1),) * len(axes)]
    uniform = (u == first).all(dim=axes, keepdim=True)
    # An arithmetic mean can round differently from its constant input.
    # This explicit exact-limit branch is not an amplitude-dependent gate.
    return c, torch.where(uniform, torch.zeros_like(u), u - c), uniform


def interaction_work(nodes: int = 3, *, include_split: bool = True) -> dict[str, int]:
    """Declared full nondegenerate batched-call work; use work={} for actuals."""
    _nodes(nodes)
    counts = _counts()
    counts.update(fft_forward=nodes + 2 + int(include_split),
                  fft_inverse=3 * nodes + 2 + int(include_split),
                  quadrature_evaluations=nodes,
                  reaction_evaluations=2 * int(include_split),
                  reaction_jacobian_evaluations=nodes + 1,
                  reaction_weight_evaluations=1,
                  nonlinear_evaluations=nodes + 2)
    _finish(None, counts)
    return counts


def _defect(u: Tensor, step: Tensor, equation: Equation, geometry: Geometry,
            nodes: int, counts: dict[str, int]) -> Tensor:
    c, v, uniform = _center(u)
    if equation.kappa == 0 or equation.reaction_rate == 0 or bool((step == 0).all()) or bool(uniform.all()):
        return u * 0
    axes = tuple(range(2, u.ndim))
    spectrum = equation.kappa * diffusion_eigenvalues(geometry, u)

    def fft(x):
        counts["fft_forward"] += 1
        return torch.fft.fftn(x, dim=axes)

    def inverse(x):
        counts["fft_inverse"] += 1
        return torch.fft.ifftn(x, dim=axes).real

    v_hat, square_hat = fft(v), fft(v.square())
    counts["nonlinear_evaluations"] += 1

    def variation(t):
        increment = torch.expm1(t * spectrum)
        dv = inverse(increment * v_hat)
        dsquare = inverse(increment * square_hat)
        counts["nonlinear_evaluations"] += 1
        return 2 * v * dv + dv.square() - dsquare

    # Uniform endpoint parents must not introduce 0*infinity into a mixed
    # batch at large r h. Their exact quadratic variation is identically zero.
    safe_c = torch.where(uniform, torch.full_like(c, .5), c)
    integral = torch.zeros_like(u)
    locations, weights = _GL[nodes]
    for x, weight in zip(locations, weights):
        s = step * ((1 + x) / 2)
        transported = inverse(torch.exp((step - s) * spectrum) * fft(variation(s)))
        integral = integral + (step * weight / 2) * reaction_jacobian(safe_c, s, equation.reaction_rate) * transported
        counts["quadrature_evaluations"] += 1
        counts["reaction_jacobian_evaluations"] += 1
    # V_s=(E_s v)^2-E_s(v²), so E_(h-s)V_s=F_s-B and -V_h=B-C.
    integral = integral - reaction_weight(safe_c, step, equation.reaction_rate) * variation(step)
    result = -equation.reaction_rate * reaction_jacobian(safe_c, step, equation.reaction_rate) * integral
    counts["reaction_weight_evaluations"] += 1
    counts["reaction_jacobian_evaluations"] += 1
    return torch.where(uniform | (step == 0), torch.zeros_like(result), result)


def interaction_defect(u: Tensor, h: float | Tensor, equation: Equation,
                       geometry: Geometry, *, nodes: int = 3,
                       work: Work | None = None) -> Tensor:
    """Anchored GL3/GL5 quadratic state defect; no coordinate or extra gate."""
    _nodes(nodes)
    step = _validate(u, h, equation, geometry)
    counts = _counts()
    answer = _defect(u, step, equation, geometry, nodes, counts)
    _finish(work, counts)
    return answer


def interaction_step(u: Tensor, h: float | Tensor, equation: Equation,
                     geometry: Geometry, *, nodes: int = 3,
                     coordinate: str = "additive", mean_mode: str = "full",
                     work: Work | None = None) -> Tensor:
    """Add the defect after Strang, optionally applying directional capacity.

    ``projected`` explicitly clips the additive result and is only a baseline.
    ``mean_only`` and ``zero_mean`` decompose the *raw* correction before the
    coordinate map. Bounds of capacity remain conditional on a bounded split
    base; FFT roundoff is never silently repaired in the capacity variant.
    """
    _nodes(nodes)
    if coordinate not in ("additive", "capacity", "projected"):
        raise ValueError("coordinate must be additive, capacity, or projected")
    if mean_mode not in ("full", "zero_mean", "mean_only"):
        raise ValueError("mean_mode must be full, zero_mean, or mean_only")
    step = _validate(u, h, equation, geometry)
    counts = _counts()
    if bool((step == 0).all()):
        _finish(work, counts)
        return u + step * 0
    base = split_step(u, step, equation, geometry)
    counts["reaction_evaluations"] = 2 if equation.reaction_rate else 0
    counts["fft_forward"] = counts["fft_inverse"] = 1 if equation.kappa else 0
    increment = _defect(u, step, equation, geometry, nodes, counts)
    if mean_mode != "full":
        mean = increment.mean(dim=tuple(range(2, u.ndim)), keepdim=True)
        increment = mean if mean_mode == "mean_only" else increment - mean
    if coordinate == "additive":
        answer = base + increment
    elif coordinate == "capacity":
        answer = capacity_update(base, increment)
    else:
        answer = (base + increment).clamp(0, 1)
    answer = torch.where(step == 0, u, answer)
    _finish(work, counts)
    return answer


def interaction_cubic_coefficient(u: Tensor, equation: Equation,
                                  geometry: Geometry, *, work: Work | None = None) -> Tensor:
    """The analytic h³ coefficient of the quadratic interaction defect."""
    _validate(u, 0., equation, geometry)
    counts = _counts()
    c, v, uniform = _center(u)
    if equation.kappa == 0 or equation.reaction_rate == 0 or bool(uniform.all()):
        _finish(work, counts)
        return u * 0

    def A(x):
        counts["laplacian_evaluations"] += 1
        return equation.kappa * laplacian(x, geometry)

    av, av2 = A(v), A(v.square())
    v1 = 2 * v * av - av2
    v2 = 2 * av.square() + 2 * v * A(av) - A(av2)
    counts["nonlinear_evaluations"] = 2
    result = -equation.reaction_rate * (A(v1) / 6 - v2 / 12
             - equation.reaction_rate * (1 - 2 * c) * v1 / 24)
    _finish(work, counts)
    return torch.where(uniform, torch.zeros_like(result), result)


def scalar_defect(u: Tensor, h: float | Tensor, equation: Equation,
                  geometry: Geometry, *, work: Work | None = None) -> Tensor:
    """Prescribed scalar h³ control of the same quadratic coefficient."""
    step = _validate(u, h, equation, geometry)
    counts = _counts()
    if bool((step == 0).all()):
        _finish(work, counts)
        return u * 0
    answer = step.pow(3) * interaction_cubic_coefficient(u, equation, geometry, work=counts)
    _finish(work, counts)
    return answer


def output_phi_defect(u: Tensor, h: float | Tensor, equation: Equation,
                      geometry: Geometry, *, work: Work | None = None) -> Tensor:
    """Prescribed h³*6phi_3(hA) control, filtering output frequency only."""
    step = _validate(u, h, equation, geometry)
    counts = _counts()
    coefficient = interaction_cubic_coefficient(u, equation, geometry, work=counts)
    if bool((step == 0).all()) or equation.kappa == 0 or equation.reaction_rate == 0 or not bool(coefficient.any()):
        _finish(work, counts)
        return u * 0
    axes = tuple(range(2, u.ndim))
    multiplier = 6 * phi(step * equation.kappa * diffusion_eigenvalues(geometry, u), 3)
    answer = step.pow(3) * torch.fft.ifftn(multiplier * torch.fft.fftn(coefficient, dim=axes), dim=axes).real
    counts["fft_forward"] += 1
    counts["fft_inverse"] += 1
    _finish(work, counts)
    return answer


def _etd_step(u: Tensor, h: float | Tensor, equation: Equation,
              geometry: Geometry, order: int, work: Work | None) -> Tensor:
    step = _validate(u, h, equation, geometry)
    counts = _counts()
    if bool((step == 0).all()):
        _finish(work, counts)
        return u + step * 0
    if equation.reaction_rate == 0:
        answer = diffusion_step(u, step, equation, geometry)
        counts["fft_forward"] = counts["fft_inverse"] = int(equation.kappa > 0)
        _finish(work, counts)
        return torch.where(step == 0, u, answer)
    axes = tuple(range(2, u.ndim))

    def fft(x):
        if equation.kappa == 0:
            return x
        counts["fft_forward"] += 1
        return torch.fft.fftn(x, dim=axes)

    def inverse(x):
        if equation.kappa == 0:
            return x
        counts["fft_inverse"] += 1
        return torch.fft.ifftn(x, dim=axes).real

    def N(x):
        counts["nonlinear_evaluations"] += 1
        return fft(equation.reaction_rate * x * (1 - x))

    z = step * equation.kappa * diffusion_eigenvalues(geometry, u)
    E = torch.exp(z)
    p1, p2 = phi(z, 1), phi(z, 2)
    initial, n0 = fft(u), N(u)
    if order == 2:
        stage_hat = E * initial + step * p1 * n0
        n1 = N(inverse(stage_hat))
        answer = inverse(stage_hat + step * p2 * (n1 - n0))
    else:
        E2, Q = torch.exp(z / 2), (step / 2) * phi(z / 2, 1)
        a_hat = E2 * initial + Q * n0
        na = N(inverse(a_hat))
        b_hat = E2 * initial + Q * na
        nb = N(inverse(b_hat))
        c_hat = E2 * a_hat + Q * (2 * nb - n0)
        nc = N(inverse(c_hat))
        p3 = phi(z, 3)
        answer = inverse(E * initial + step * ((p1 - 3 * p2 + 4 * p3) * n0
                         + (2 * p2 - 4 * p3) * (na + nb)
                         + (-p2 + 4 * p3) * nc))
    _finish(work, counts)
    return torch.where(step == 0, u, answer)


def etdrk2_step(u: Tensor, h: float | Tensor, equation: Equation,
                geometry: Geometry, *, work: Work | None = None) -> Tensor:
    """Second-order exponential Heun, with the discrete diffusion spectrum."""
    return _etd_step(u, h, equation, geometry, 2, work)


def etdrk4_step(u: Tensor, h: float | Tensor, equation: Equation,
                geometry: Geometry, *, work: Work | None = None) -> Tensor:
    """Cox--Matthews ETDRK4 with stable phi functions and four RHS calls."""
    return _etd_step(u, h, equation, geometry, 4, work)
