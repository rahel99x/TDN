"""Track-explicit differentiable numerical primitives for the five-gate study.

``discrete`` is the periodic FD/nodal equation. ``continuum`` advances the
spectral Galerkin quadratic equation on the supplied grid; it is still a finite
spatial approximation, not the continuum truth. In particular its reaction
flow is integrated with projected RK4, never sampled pointwise logistic.
"""
from __future__ import annotations

from collections import OrderedDict
import math

import torch

from tdn.numerics.operators import broadcast_h, check_shape
from tdn.numerics.subflows import diffusion_eigenvalues, reaction_step
from tdn.research.interaction import phi
from tdn.analysis.roadmap.numerics import dealiased_product, fourier_resample

TRACKS = ("discrete", "continuum")


def validate(u, h, geometry, track):
    check_shape(u, geometry)
    if u.dtype not in (torch.float32, torch.float64):
        raise TypeError("Frontier numerical work requires FP32 or FP64")
    if track not in TRACKS:
        raise ValueError("Track must be discrete or continuum")
    step = broadcast_h(h, u)
    if not bool(torch.isfinite(step).all()) or bool((step < 0).any()):
        raise ValueError("Finite nonnegative time steps are required")
    return step


def diffusion_symbol(u, equation, geometry, track="discrete"):
    """Physical diffusion generator; both grids and domain lengths matter."""
    if track == "discrete":
        return equation.kappa * diffusion_eigenvalues(geometry, u)
    if track != "continuum":
        raise ValueError("Track must be discrete or continuum")
    result = torch.zeros(geometry.grid, dtype=u.dtype, device=u.device)
    for axis, (n, dx) in enumerate(zip(geometry.grid, geometry.dx)):
        frequency = 2 * torch.pi * torch.fft.fftfreq(n, d=dx, device=u.device, dtype=u.dtype)
        shape = [1] * geometry.ndim
        shape[axis] = n
        result = result - equation.kappa * frequency.square().reshape(shape)
    return result


def product(a, b, track="discrete"):
    if track == "discrete":
        return a * b
    if track == "continuum":
        return dealiased_product(a, b)
    raise ValueError("Track must be discrete or continuum")


def apply_multiplier(u, multiplier):
    axes = tuple(range(2, u.ndim))
    return torch.fft.ifftn(torch.fft.fftn(u, dim=axes) * multiplier, dim=axes).real


def reaction_rhs(u, equation, track="discrete"):
    return equation.reaction_rate * (u - product(u, u, track))


def rhs(u, equation, geometry, track="discrete"):
    validate(u, 0., geometry, track)
    return apply_multiplier(u, diffusion_symbol(u, equation, geometry, track)) + reaction_rhs(u, equation, track)


class CoefficientCache:
    """Bounded immutable operator coefficients, never states or predictions.

    Every family receives this same option. Cold preparation is observable in
    metadata and must be charged separately from warmed inference. Tensor time
    gradients and batched steps bypass caching. No numerical precision changes
    are made to disguise reference or timing differences.
    """
    def __init__(self, max_entries=32):
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("Cache capacity must be a positive integer")
        self.max_entries = max_entries
        self.clear()

    def clear(self):
        self.entries = OrderedDict()
        self.hits = self.preparations = self.evictions = self.bypasses = 0

    def get(self, kind, u, step, equation, geometry, track, factory):
        if step.ndim or step.requires_grad:
            self.bypasses += 1
            return factory()
        key = (kind, float(step), equation.kappa, equation.reaction_rate,
               geometry.grid, geometry.lengths, track, str(u.device), u.dtype)
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key]
        value = factory()
        self.preparations += 1
        self.entries[key] = value
        if len(self.entries) > self.max_entries:
            self.entries.popitem(last=False)
            self.evictions += 1
        return value

    @property
    def metadata(self):
        tensors = []
        for item in self.entries.values():
            tensors.extend(item.values() if isinstance(item, dict) else [item])
        return dict(coefficient_preparations=self.preparations, cache_hits=self.hits,
                    cache_entries=len(self.entries), cache_evictions=self.evictions,
                    gradient_or_batch_bypasses=self.bypasses,
                    cached_tensor_bytes=sum(t.numel() * t.element_size() for t in tensors),
                    state_dependent_cache=False, cold_preparation_must_be_charged=True)


def heat_step(u, h, equation, geometry, track="discrete", *, cache=None):
    step = validate(u, h, geometry, track)
    if equation.kappa == 0:
        return u + step * 0
    factory = lambda: torch.exp(step * diffusion_symbol(u, equation, geometry, track))
    multiplier = factory() if cache is None else cache.get("heat", u, step, equation, geometry, track, factory)
    answer = apply_multiplier(u, multiplier)
    return torch.where(step == 0, u, answer)


def galerkin_reaction_step(u, h, equation, *, substeps=4):
    """RK4 for r(u-P_N(u²)); no clipping and no exact-flow claim.

    Subdivision depends only on declared r and h. The constant minimum bounds
    reaction integration error; additional r*h-based subdivision is explicitly
    included in whole-call cost. This is not an unconditional stability claim.
    """
    if type(substeps) is not int or substeps < 1:
        raise ValueError("Reaction substeps must be a positive integer")
    step = broadcast_h(h, u)
    if not bool(torch.isfinite(step).all()) or bool((step < 0).any()):
        raise ValueError("Finite nonnegative time steps are required")
    if equation.reaction_rate == 0:
        return u + step * 0
    count = max(substeps, math.ceil(float(step.detach().max()) * equation.reaction_rate / .25))
    dt = step / count
    value = u
    for _ in range(count):
        a = reaction_rhs(value, equation, "continuum")
        b = reaction_rhs(value + dt * a / 2, equation, "continuum")
        c = reaction_rhs(value + dt * b / 2, equation, "continuum")
        d = reaction_rhs(value + dt * c, equation, "continuum")
        value = value + dt * (a + 2 * b + 2 * c + d) / 6
    return torch.where(step == 0, u, value)


def df_step(u, h, equation, geometry, track="discrete", *, reaction_substeps=4, cache=None):
    step = validate(u, h, geometry, track)
    middle = heat_step(u, step / 2, equation, geometry, track, cache=cache)
    middle = (reaction_step(middle, step, equation) if track == "discrete" else
              galerkin_reaction_step(middle, step, equation, substeps=reaction_substeps))
    result = heat_step(middle, step / 2, equation, geometry, track, cache=cache)
    return torch.where(step == 0, u, result)


def etdrk4_step(u, h, equation, geometry, track="discrete", *, cache=None):
    """Cox–Matthews ETDRK4, stable phi functions and the declared product."""
    step = validate(u, h, geometry, track)
    if equation.reaction_rate == 0:
        return heat_step(u, step, equation, geometry, track, cache=cache)

    def prepare():
        z = step * diffusion_symbol(u, equation, geometry, track)
        p1, p2, p3 = phi(z, 1), phi(z, 2), phi(z, 3)
        return dict(E=torch.exp(z), E2=torch.exp(z / 2), Q=step / 2 * phi(z / 2, 1),
                    w0=step * (p1 - 3 * p2 + 4 * p3),
                    wab=step * (2 * p2 - 4 * p3), wc=step * (-p2 + 4 * p3))

    p = prepare() if cache is None else cache.get("etdrk4", u, step, equation, geometry, track, prepare)
    axes = tuple(range(2, u.ndim))
    fft = lambda x: torch.fft.fftn(x, dim=axes)
    inverse = lambda x: torch.fft.ifftn(x, dim=axes).real
    nonlinear = lambda x: fft(reaction_rhs(x, equation, track))
    initial, n0 = fft(u), nonlinear(u)
    a = p["E2"] * initial + p["Q"] * n0
    na = nonlinear(inverse(a))
    b = p["E2"] * initial + p["Q"] * na
    nb = nonlinear(inverse(b))
    c = p["E2"] * a + p["Q"] * (2 * nb - n0)
    nc = nonlinear(inverse(c))
    value = inverse(p["E"] * initial + p["w0"] * n0 + p["wab"] * (na + nb) + p["wc"] * nc)
    return torch.where(step == 0, u, value)
