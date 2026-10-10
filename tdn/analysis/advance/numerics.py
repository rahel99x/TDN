"""Cheap cubic interaction bases with explicit fixed-grid asymptotic scope.

The cubic formula is the leading h^3 term of the *existing* centered-amplitude
Volterra DF defect. It is not a new third/fourth-order full solver, and it is
not uniform in h times the diffusion eigenvalues. Projected quadratic products
are not associative: the Galerkin formula must retain their nesting.
"""
from __future__ import annotations

import torch

from tdn.analysis.frontier import numerics as physical
from tdn.numerics.operators import laplacian


def diffusion_generator(u, equation, geometry, track="discrete"):
    """Apply the declared generator, using the local FD stencil when possible."""
    if track == "discrete":
        return equation.kappa * laplacian(u, geometry)
    if track != "continuum":
        raise ValueError("Track must be discrete or continuum")
    return physical.apply_multiplier(u, physical.diffusion_symbol(u, equation, geometry, track))


def cubic_commutator_defect(u, h, equation, geometry, *, track="discrete", transport="none"):
    r"""Leading centered cubic DF defect; raw or explicitly heat-filtered.

    With v=u-mean(u), A the diffusion generator, and B the target's symmetric
    quadratic product, the coefficient is

    r^2 h^3/6 [B(Av,B(v,v))-2B(v,B(v,Av))
               +2B(v,A B(v,v))-A B(v,B(v,v))].

    This follows by expanding the nested causal cubic Volterra integral and
    its split subtraction to first order in transport time. For a nodal
    product it simplifies to r^2 h^3/6[-v^2 Av+2v A(v^2)-A(v^3)].
    For an unprojected continuum Laplacian this is -r^2*kappa*h^3/3*v|grad v|^2;
    applying that differential product identity to a truncated Galerkin algebra
    would be incorrect.

    ``symmetric_heat`` evaluates H(h/2) C3[H(h/2)v]. This preserves the leading
    coefficient and cubic amplitude order but is a heuristic finite-step
    regularization, not an exact Volterra coefficient or stiff-order proof.
    It can suppress useful high-to-low interactions and has its own raw control.
    """
    step = physical.validate(u, h, geometry, track)
    if transport not in ("none", "symmetric_heat"):
        raise ValueError("Cubic transport must be none or symmetric_heat")
    if equation.kappa == 0 or equation.reaction_rate == 0:
        return u * 0 + step * 0
    axes = tuple(range(2, u.ndim))
    v = u - u.mean(axes, keepdim=True)
    if transport == "symmetric_heat":
        v = physical.heat_step(v, step / 2, equation, geometry, track)
    av = diffusion_generator(v, equation, geometry, track)
    product = lambda a, b: physical.product(a, b, track)
    square = product(v, v)
    a_square = diffusion_generator(square, equation, geometry, track)
    cube = product(v, square)
    a_cube = diffusion_generator(cube, equation, geometry, track)
    if track == "discrete":
        coefficient = -square * av + 2 * v * a_square - a_cube
    else:
        coefficient = (product(av, square) - 2 * product(v, product(v, av))
                       + 2 * product(v, a_square) - a_cube)
    answer = equation.reaction_rate**2 * step**3 * coefficient / 6
    if transport == "symmetric_heat":
        answer = physical.heat_step(answer, step / 2, equation, geometry, track)
    constant = (u == u[(...,) + (slice(0, 1),) * geometry.ndim]).flatten(2).all(-1)
    constant = constant.reshape(u.shape[0], 1, *([1] * geometry.ndim))
    return torch.where(constant | (step == 0), torch.zeros_like(answer), answer)


def deployable_defect_scale(u, h, equation, geometry, *, track="discrete"):
    """Dimensionless state-local residual scale, with no labels or target calls.

    h^3*r*variance*lambda_active*(r+lambda_active) reflects leading split-defect
    dimensions. It is an amplitude scale, not an error estimator or certificate.
    Feature extraction is executed and charged at each call. Scaling uses the
    declared generator, domain lengths and field, not an n-dependent surrogate.
    """
    step = physical.validate(u, h, geometry, track)
    axes = tuple(range(2, u.ndim))
    v = u - u.mean(axes, keepdim=True)
    variance = v.square().mean(axes, keepdim=True)
    if equation.kappa == 0 or equation.reaction_rate == 0:
        return variance * 0 + step * 0
    energy = -(v * diffusion_generator(v, equation, geometry, track)).mean(axes, keepdim=True)
    active = torch.where(variance > 0, energy / variance.clamp_min(torch.finfo(u.dtype).tiny),
                         torch.zeros_like(variance)).clamp_min(0)
    scale = step**3 * equation.reaction_rate * variance * active * (equation.reaction_rate + active)
    constant = (u == u[(...,) + (slice(0, 1),) * geometry.ndim]).flatten(2).all(-1)
    constant = constant.reshape(u.shape[0], 1, *([1] * geometry.ndim))
    return torch.where(constant | (variance == 0) | (step == 0), torch.zeros_like(scale), scale)
