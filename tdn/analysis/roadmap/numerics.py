"""Finite-step physical mechanisms for the new roadmap program.

The default target is the periodic FD/nodal equation. ``target='continuum'``
selects a Fourier/Galerkin quadratic target explicitly; neither convention is
silently substituted for the other. All response coefficients use the Taylor
coefficient convention (second derivative / 2, third derivative / 6).
"""
from __future__ import annotations

from functools import lru_cache
import math
from typing import Callable

import numpy as np
import torch
from torch import Tensor

from tdn.numerics import Equation, Geometry
from tdn.numerics.operators import broadcast_h, check_shape, laplacian, rhs
from tdn.numerics.subflows import diffusion_step, reaction_step
from tdn.analysis.agenda.physics import diffusion_first_step, moment_derivatives


@lru_cache(maxsize=16)
def quadrature(nodes: int):
    if type(nodes) is not int or not 1 <= nodes <= 128:
        raise ValueError("quadrature nodes must be an integer in [1, 128]")
    x, w = np.polynomial.legendre.leggauss(nodes)
    return tuple((x + 1) / 2), tuple(w / 2)


def _axes(u):
    return tuple(range(2, u.ndim))


def _step(h, u):
    value = broadcast_h(h, u)
    if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
        raise ValueError("finite nonnegative horizons are required")
    return value


def background_response(c: Tensor, h, reaction_rate: float):
    """Stable homogeneous state, derivative q and integral of q on [0,h]."""
    step = broadcast_h(h, c)
    z = torch.exp(-reaction_rate * step)
    den = c + (1 - c) * z
    # A constant zero field at an underflowed positive time has zero centered
    # direction. Preserve that value without claiming finite endpoint Jacobian.
    safe = torch.where(den == 0, torch.ones_like(den), den)
    q = (z / safe) / safe
    integral = step if reaction_rate == 0 else -torch.expm1(-reaction_rate * step) / (reaction_rate * safe)
    return c / safe, q, integral


def fourier_resample(u: Tensor, grid: tuple[int, ...]) -> Tensor:
    """Differentiable real Fourier interpolation, including Nyquist splitting.

    This is the torch equivalent of scipy.signal.resample along each axis.
    Norm='forward' makes coefficients independent of the sampling count.
    """
    if len(grid) != u.ndim - 2 or any(type(n) is not int or n < 2 for n in grid):
        raise ValueError("a positive spatial grid is required")
    answer = u
    for axis, size in enumerate(grid, start=2):
        old = answer.shape[axis]
        if old == size:
            continue
        spectrum = torch.fft.rfft(answer, dim=axis, norm="forward")
        shape = list(spectrum.shape)
        shape[axis] = size // 2 + 1
        output = torch.zeros(shape, dtype=spectrum.dtype, device=u.device)
        common = min(old, size)
        index = [slice(None)] * u.ndim
        index[axis] = slice(0, common // 2 + 1)
        output[tuple(index)] = spectrum[tuple(index)]
        if common % 2 == 0:
            nyquist = [slice(None)] * u.ndim
            nyquist[axis] = common // 2
            if size < old:
                output[tuple(nyquist)] = output[tuple(nyquist)] * 2
            else:
                output[tuple(nyquist)] = output[tuple(nyquist)] * .5
        answer = torch.fft.irfft(output, n=size, dim=axis, norm="forward")
    return answer


def dealiased_product(a: Tensor, b: Tensor) -> Tensor:
    """Exact retained quadratic Fourier product with >3/2 padding.

    It resolves a quadratic RHS. It does not assert that an exact sampled
    logistic subflow, with infinitely many harmonics, is dealiased.
    """
    if a.shape != b.shape or a.ndim < 3:
        raise ValueError("matching scalar spatial fields are required")
    grid = tuple(a.shape[2:])
    padded = tuple(3 * n // 2 + 1 for n in grid)
    return fourier_resample(fourier_resample(a, padded) * fourier_resample(b, padded), grid)


def continuum_diffusion_step(u, h, equation, geometry):
    check_shape(u, geometry)
    step = _step(h, u)
    lam = torch.zeros(geometry.grid, dtype=u.dtype, device=u.device)
    for axis, (n, dx) in enumerate(zip(geometry.grid, geometry.dx)):
        k = 2 * torch.pi * torch.fft.fftfreq(n, d=dx, dtype=u.dtype, device=u.device)
        shape = [1] * geometry.ndim
        shape[axis] = n
        lam = lam - equation.kappa * k.square().reshape(shape)
    return torch.fft.ifftn(torch.fft.fftn(u, dim=_axes(u)) * torch.exp(step * lam), dim=_axes(u)).real


def _target(target):
    if target == "discrete":
        return diffusion_step, lambda a, b: a * b
    if target == "continuum":
        return continuum_diffusion_step, dealiased_product
    raise ValueError("target must be discrete or continuum")


def quadratic_df_defect(u, h, equation, geometry, *, nodes=4, target="discrete",
                        node_positions=None, node_weights=None):
    """Zero-subtracted exact finite-h quadratic DF coefficient on centered u.

    A finite quadrature approximates the exact coefficient. A midpoint-only
    rule is identically zero. Learned signed weights preserve the physical
    nullspace but are not called an exact quadrature. Batched c and h remain
    differentiable, including learned interior quadrature nodes/weights.
    """
    check_shape(u, geometry)
    step = _step(h, u)
    heat, product = _target(target)
    if equation.kappa == 0 or equation.reaction_rate == 0:
        return u * 0 + step * 0
    c = u.mean(_axes(u), keepdim=True)
    v = u - c
    # Constant endpoint fields have zero response even when their formal
    # homogeneous derivative overflows; avoid inf*0 in this null branch.
    c = torch.where(v.abs().amax(_axes(u), keepdim=True) == 0, torch.full_like(c, .5), c)
    _, qh, _ = background_response(c, step, equation.reaction_rate)
    if (node_positions is None) != (node_weights is None):
        raise ValueError("provide both node positions and weights")
    positions, weights = quadrature(nodes) if node_positions is None else (node_positions, node_weights)
    if len(positions) != len(weights) or not len(positions):
        raise ValueError("nonempty matching quadrature positions/weights required")
    midpoint = heat(v, step / 2, equation, geometry)
    split_shape = heat(product(midpoint, midpoint), step / 2, equation, geometry)
    result = torch.zeros_like(u)
    for position, weight in zip(positions, weights):
        node = torch.as_tensor(position, dtype=u.dtype, device=u.device)
        if not bool(torch.isfinite(node).all()) or bool(((node < 0) | (node > 1)).any()):
            raise ValueError("quadrature positions must lie in [0,1]")
        s = step * node
        _, qs, _ = background_response(c, s, equation.reaction_rate)
        propagated = heat(v, s, equation, geometry)
        shape = heat(product(propagated, propagated), step - s, equation, geometry)
        result = result + torch.as_tensor(weight, dtype=u.dtype, device=u.device) * qs * (shape - split_shape)
    return -equation.reaction_rate * qh * step * result


def cubic_df_defect(u, h, equation, geometry, *, nodes=6, target="discrete"):
    """Causal cubic Volterra coefficient minus the exact DF cubic coefficient.

    Nested integration obeys 0 <= tau <= s <= h. The split term is subtracted
    INSIDE the integral; commuting limits hold independently of quadrature.
    Cubic Galerkin products mean sequential projected quadratic interactions,
    as required by that target's variational hierarchy.
    """
    check_shape(u, geometry)
    step = _step(h, u)
    heat, product = _target(target)
    if equation.kappa == 0 or equation.reaction_rate == 0:
        return u * 0 + step * 0
    c = u.mean(_axes(u), keepdim=True)
    v = u - c
    # Constant endpoint fields have zero response even when their formal
    # homogeneous derivative overflows; avoid inf*0 in this null branch.
    c = torch.where(v.abs().amax(_axes(u), keepdim=True) == 0, torch.full_like(c, .5), c)
    _, qh, _ = background_response(c, step, equation.reaction_rate)
    midpoint = heat(v, step / 2, equation, geometry)
    split_shape = heat(product(midpoint, product(midpoint, midpoint)), step / 2, equation, geometry)
    result = torch.zeros_like(u)
    positions, weights = quadrature(nodes)
    for outer, outer_weight in zip(positions, weights):
        s = step * float(outer)
        _, qs, _ = background_response(c, s, equation.reaction_rate)
        first_s = heat(v, s, equation, geometry)
        inner = torch.zeros_like(u)
        for node, weight in zip(positions, weights):
            tau = s * float(node)
            _, qt, _ = background_response(c, tau, equation.reaction_rate)
            first_t = heat(v, tau, equation, geometry)
            second_s = heat(product(first_t, first_t), s - tau, equation, geometry)
            shape = heat(product(first_s, second_s), step - s, equation, geometry)
            inner = inner + float(weight) * qt * (shape - split_shape)
        result = result + float(outer_weight) * qs * s * inner
    return 2 * equation.reaction_rate**2 * qh * step * result


def lowpass(u: Tensor, cutoff: int):
    """Physical integer Fourier-mode cutoff, applied identically across grids."""
    if type(cutoff) is not int or cutoff < 0:
        raise ValueError("cutoff must be a nonnegative integer")
    mask = torch.ones(tuple(u.shape[2:]), dtype=torch.bool, device=u.device)
    for axis, n in enumerate(u.shape[2:]):
        k = torch.fft.fftfreq(n, device=u.device) * n
        shape = [1] * (u.ndim - 2)
        shape[axis] = n
        mask = mask & (k.abs().reshape(shape) <= cutoff)
    return torch.fft.ifftn(torch.fft.fftn(u, dim=_axes(u)) * mask, dim=_axes(u)).real


def signed_source(u, equation, geometry):
    """Exact discrete [A,B]=r(L(u²)-2uLu); transport accepts signed weights.

    The commutator itself is nonnegative for this cooperative FD stencil.
    A signed dictionary also includes its centered and propagated responses.
    """
    check_shape(u, geometry)
    # Neighbor squared differences avoid catastrophic constant-field subtraction.
    result = torch.zeros_like(u)
    for axis, dx in enumerate(geometry.dx, start=2):
        result = result + ((torch.roll(u, 1, axis) - u).square()
                           + (torch.roll(u, -1, axis) - u).square()) / dx**2
    return equation.reaction_rate * equation.kappa * result


def source_transport(u, h, equation, geometry, *, cutoff, placement="before", weight=1.):
    """Pure source-before/after-input-compression ablation without a bypass.

    Both variants output the same retained band. ``before`` forms the source
    from full input before compression; ``after`` forms it from compressed u.
    A full identical DF base can be added by the caller, never a source bypass.
    """
    _step(h, u)
    if placement not in {"before", "after"}:
        raise ValueError("placement must be before or after")
    source = signed_source(u if placement == "before" else lowpass(u, cutoff), equation, geometry)
    return weight * lowpass(diffusion_step(source, h, equation, geometry), cutoff)


def match_mean_bounded(u, target_mean):
    """Replace zero mode once, with the largest feasible centered amplitude."""
    axes = _axes(u)
    mean = torch.as_tensor(target_mean, dtype=u.dtype, device=u.device)
    if bool(((mean < 0) | (mean > 1)).any()):
        raise ValueError("target mean must lie in [0,1]")
    v = u - u.mean(axes, keepdim=True)
    negative = (-v.amin(axes, keepdim=True)).clamp_min(torch.finfo(u.dtype).tiny)
    positive = v.amax(axes, keepdim=True).clamp_min(torch.finfo(u.dtype).tiny)
    scale = torch.minimum(torch.ones_like(mean), torch.minimum(mean / negative, (1 - mean) / positive))
    return mean + scale * v, scale


def dynamic_moment_step(u, h, equation, geometry, *, steps=16):
    """Feasible dynamic mean/variance approximation with inward binary flux.

    Initial rates equal the discrete moment identities. Later spectral energy
    and skewness are modeled by a diffused initial shape rescaled to current
    variance; this is an explicit closure, NOT an exact future moment law.
    Exact frozen positive flux updates keep z=V/[m(1-m)] in [0,1].
    """
    check_shape(u, geometry)
    step = _step(h, u)
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    axes = _axes(u)
    if equation.kappa == 0 or equation.reaction_rate == 0:
        endpoint = (reaction_step(u, step, equation) if equation.kappa == 0
                    else diffusion_step(u, step, equation, geometry))
        m = endpoint.mean(axes, keepdim=True)
        variance = (endpoint - m).square().mean(axes, keepdim=True)
        capacity = m * (1 - m)
        z = torch.where(capacity > 0, variance / capacity.clamp_min(torch.finfo(u.dtype).tiny), torch.zeros_like(capacity))
        return {"mean": m, "variance": variance, "normalized_variance": z,
                "closure": "exact_commuting_subflow_moments"}
    m = u.mean(axes, keepdim=True)
    v = u - m
    variance = v.square().mean(axes, keepdim=True)
    tiny = torch.finfo(u.dtype).eps
    cap = m * (1 - m)
    z = torch.where(cap > 0, variance / cap.clamp_min(tiny), torch.zeros_like(cap)).clamp(0, 1)
    dt = step / steps
    for i in range(steps):
        shape = diffusion_step(v, step * (i / steps), equation, geometry)
        shape_var = shape.square().mean(axes, keepdim=True)
        gamma = -(shape * (equation.kappa * laplacian(shape, geometry))).mean(axes, keepdim=True) / shape_var.clamp_min(tiny)
        current_var = z * m * (1 - m)
        shape_skew = shape.pow(3).mean(axes, keepdim=True) / shape_var.clamp_min(tiny)
        skew = shape_skew * torch.sqrt(current_var / shape_var.clamp_min(tiny))
        skew = torch.minimum(1 - m, torch.maximum(-m, skew))
        # z=1 is a Bernoulli field: its skewness ratio is exactly 1-2m.
        skew = torch.where(z >= 1 - 8 * tiny, 1 - 2 * m, skew)
        rate = z * (-2 * gamma + equation.reaction_rate * ((1 - 2 * m) * (1 + z) - 2 * skew))
        influx = torch.where(z < 1 - tiny, rate.clamp_min(0) / (1 - z).clamp_min(tiny), torch.zeros_like(z))
        outflux = torch.where(z > tiny, (-rate).clamp_min(0) / z.clamp_min(tiny), torch.zeros_like(z))
        total = influx + outflux
        decay = torch.exp(-total * dt)
        znext = z * decay + torch.where(total > tiny, influx / total.clamp_min(tiny), torch.zeros_like(total)) * (-torch.expm1(-total * dt))
        # m'=r*m*(1-m)*(1-z), whose frozen-z update is exact logistic.
        m = reaction_step(m, dt * (1 - z), equation)
        z = znext.clamp(0, 1)
    variance = z * m * (1 - m)
    return {"mean": m, "variance": variance, "normalized_variance": z,
            "closure": "dynamic_heat_shape_rescaled_moments_with_feasible_flux"}


def finite_amplitude_step(u, h, equation, geometry, *, include_mean=False,
                          include_quadratic=True, include_cubic=False,
                          nodes=6, moment_steps=16):
    """C3 ablation: DF + physical coefficients; optional single mean replacement."""
    base = diffusion_first_step(u, h, equation, geometry)
    correction = torch.zeros_like(base)
    if include_quadratic:
        correction = correction + quadratic_df_defect(u, h, equation, geometry, nodes=nodes)
    if include_cubic:
        correction = correction + cubic_df_defect(u, h, equation, geometry, nodes=nodes)
    if include_mean:
        closure = dynamic_moment_step(u, h, equation, geometry, steps=moment_steps)
        # Remove the entire tentative zero mode before installing closure mean.
        answer, _ = match_mean_bounded(base + correction, closure["mean"])
        return answer
    return base + correction


def rk4_step(u, h, equation, geometry, *, field=rhs):
    dt = _step(h, u)
    k1 = field(u, equation, geometry)
    k2 = field(u + dt * k1 / 2, equation, geometry)
    k3 = field(u + dt * k2 / 2, equation, geometry)
    k4 = field(u + dt * k3, equation, geometry)
    return u + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def order_preserving_rk4(u, h, equation, geometry, *, residual: Callable | None = None):
    """Classical RK4 plus bounded h^5 residual: fixed-grid fourth order only.

    ``residual(u, equation, geometry)`` may be a trainable module and must be
    independent of h. Tanh bounds it. This does not imply stiff/A-stability.
    """
    dt = _step(h, u)
    answer = rk4_step(u, dt, equation, geometry)
    return answer if residual is None else answer + dt.pow(5) * torch.tanh(residual(u, equation, geometry))


def embedded_rk4(u, h, equation, geometry, *, residual=None):
    """Actual step-doubling estimator with all three RK4 steps charged."""
    coarse = order_preserving_rk4(u, h, equation, geometry, residual=residual)
    half = order_preserving_rk4(u, broadcast_h(h, u) / 2, equation, geometry, residual=residual)
    fine = order_preserving_rk4(half, broadcast_h(h, u) / 2, equation, geometry, residual=residual)
    return fine, (coarse - fine) / 15


def physical_diagnostics(stepper, u, v, h, equation, geometry, *, theta=.5):
    """Sampled monotonicity, concavity and e^(rh) growth; never certificates."""
    if not 0 <= theta <= 1:
        raise ValueError("theta must lie in [0,1]")
    lo, hi = torch.minimum(u, v), torch.maximum(u, v)
    a, b = stepper(lo, h, equation, geometry), stepper(hi, h, equation, geometry)
    fu, fv = stepper(u, h, equation, geometry), stepper(v, h, equation, geometry)
    mixed = stepper(theta * u + (1 - theta) * v, h, equation, geometry)
    initial = float((u - v).abs().max())
    growth = float((fu - fv).abs().max()) / max(initial, torch.finfo(u.dtype).eps)
    return {"monotonicity_violation": float((a - b).clamp_min(0).max()),
            "concavity_violation": float((theta * fu + (1 - theta) * fv - mixed).clamp_min(0).max()),
            "growth_ratio": growth, "growth_envelope": math.exp(equation.reaction_rate * float(h)),
            "bound_violation": max(float(torch.maximum((-value).clamp_min(0), (value - 1).clamp_min(0)).max())
                                   for value in (a, b, fu, fv, mixed)),
            "evidence_scope": "sampled_numerical_diagnostics_not_a_global_certificate"}
