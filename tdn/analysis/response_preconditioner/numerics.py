"""Bounded finite-difference response controls for the new preconditioner study.

These actions approximate the response about a supplied, deployable smooth
anchor path. They are not its nonlinear solver Jacobian. All factors are
independent of the source direction, and no teacher/source-label lookup occurs.
The work dictionary counts this module's actions; ``AnchorPath.work()`` records
the separate cost of constructing/evaluating the path (including cache reuse).
"""
from __future__ import annotations

import torch

from tdn.analysis.response import numerics as n

TARGET = 'fd_same_grid'


def _count(work, key, amount=1):
    if work is not None:
        work[key] = work.get(key, 0) + amount


def _field(value, like, geom, name):
    if not torch.is_tensor(value):
        raise ValueError(f'{name} must be a physical field tensor')
    n.validate(value, 0., geom, TARGET)
    if value.shape != like.shape or value.dtype != like.dtype or value.device != like.device:
        raise ValueError(f'{name} must match source shape, dtype and device')
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f'{name} must be finite')
    return value


def _validate(z, eq, geom, target):
    if target != TARGET:
        raise ValueError('UNSUPPORTED_TARGET: response preconditioner supports fd_same_grid only')
    n.validate(z, 0., geom, target)
    if not bool(torch.isfinite(z).all()):
        raise ValueError('Source must be finite')
    for name in ('kappa', 'reaction_rate'):
        value = n.scalar(getattr(eq, name), z)
        if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
            raise ValueError(f'{name} must be finite and nonnegative')


def _path_contract(path, z, start, eq, geom, *, stop=None):
    if hasattr(path, 'target') and path.target != TARGET:
        raise ValueError('UNSUPPORTED_TARGET: anchor path must use fd_same_grid')
    if hasattr(path, 'geom') and path.geom != geom:
        raise ValueError('Anchor path geometry must match response geometry')
    if hasattr(path, 'eq'):
        for name in ('kappa', 'reaction_rate'):
            path_value, action_value = getattr(path.eq, name), getattr(eq, name)
            left, right = torch.broadcast_tensors(n.scalar(path_value, z), n.scalar(action_value, z))
            if not torch.equal(left, right):
                raise ValueError(f'Anchor path {name} must match response physics')
            if (torch.is_grad_enabled() and path_value is not action_value
                    and any(torch.is_tensor(value) and value.requires_grad
                            for value in (path_value, action_value))):
                raise ValueError('Differentiable physics must use the same tensors as the anchor path')
    # A detached AnchorPath cache is valid for diagnostics, but cannot provide
    # the complete state/time/physics derivatives required by a trained model.
    inputs = (getattr(path, 'u0', None), start, stop, eq.kappa, eq.reaction_rate)
    if (torch.is_grad_enabled() and getattr(path, 'create_graph', True) is False
            and any(torch.is_tensor(x) and x.requires_grad for x in inputs)):
        raise ValueError('Differentiable anchor dependence requires create_graph=True')


def _finite(value):
    if not bool(torch.isfinite(value).all()):
        raise ValueError('Nonfinite response action; no clipping or silent fallback')
    return value


def _zero_value(value, exact, duration):
    """Literal zero-time value while retaining the computed right derivative."""
    return torch.where(duration == 0, exact.detach() + (value-value.detach()), value)


def _heat(z, delta, eq, geom, work):
    _count(work, 'heat_actions')
    _count(work, 'fft_transforms', 2)
    return n.heat(z, delta, eq, geom, TARGET)


def ordered_response(path, z, start, stop, eq, geom, *, panels=2,
                     potential_nodes=2, work=None, target=TARGET):
    """Time-ordered potential/heat/potential panels, GL-integrated potential.

    With ``V_j = integral r(1-2a(q)) dq`` on each panel, apply
    ``exp(V_j/2) H_dt exp(V_j/2)`` in chronological order. The default is
    exactly two panels and two Gauss--Legendre nodes per panel. Refinement is
    explicit and bounded; quadrature is not exact for general moving paths.
    Scalar or per-parent tensor times and physics retain their full graph.
    """
    _validate(z, eq, geom, target)
    if type(panels) is not int or panels not in (1, 2, 4, 8):
        raise ValueError('Ordered response panels must be one of 1, 2, 4, 8')
    if type(potential_nodes) is not int or potential_nodes not in (1, 2, 3, 4):
        raise ValueError('Potential quadrature nodes must be one of 1, 2, 3, 4')
    begin = n.validate(z, start, geom, target)
    end = n.validate(z, stop, geom, target)
    duration = end-begin
    if bool((duration < 0).any()):
        raise ValueError('Reversed response interval')
    _path_contract(path, z, begin, eq, geom, stop=end)
    dt = duration/panels
    rate = n.scalar(eq.reaction_rate, z)
    answer = z
    positions, weights = n.quadrature(potential_nodes)
    for panel in range(panels):
        integral = torch.zeros_like(z)
        for point, weight in zip(positions, weights):
            _count(work, 'potential_path_queries')
            state = _field(path(begin+(panel+point)*dt), z, geom, 'Anchor state')
            integral = integral + weight*rate*(1-2*state)
        multiplier = torch.exp(dt*integral/2)
        answer = multiplier*_heat(multiplier*answer, dt, eq, geom, work)
    return _finite(_zero_value(answer, z, duration))


def commutator_actions(path, z, start, eq, geom, *, work=None, target=TARGET):
    """Return ``[L,[L,V]]z, [V,[L,V]]z, [L,Vdot]z`` with seven FD stencils.

    ``V=r(1-2a(start))`` and ``Vdot=-2r*a'(start)``. The derivative is that of
    the actual supplied path, not the equation RHS substituted for it. These
    features are neither damped nor a claim of grid-uniform stiff stability.
    """
    _validate(z, eq, geom, target)
    begin = n.validate(z, start, geom, target)
    _path_contract(path, z, begin, eq, geom)
    if not callable(getattr(path, 'evaluate', None)):
        raise ValueError('Commutators require a path.evaluate(time) state/derivative pair')
    _count(work, 'potential_path_queries')
    _count(work, 'derivative_path_queries')
    row = path.evaluate(begin)
    state = _field(row['state'], z, geom, 'Anchor state')
    derivative = _field(row['derivative'], z, geom, 'Anchor derivative')
    rate = n.scalar(eq.reaction_rate, z)
    potential, potential_dot = rate*(1-2*state), -2*rate*derivative

    def L(value):
        _count(work, 'laplacian_actions')
        return n.generator(value, eq, geom, TARGET)

    lz = L(z)
    lvz = L(potential*z)
    llz = L(lz)
    llvz = L(lvz)
    lvlz = L(potential*lz)
    lvvz = L(potential.square()*z)
    lvdz = L(potential_dot*z)
    first = llvz-2*lvlz+potential*llz
    second = 2*potential*lvz-potential.square()*lz-lvvz
    third = lvdz-potential_dot*lz
    return tuple(_finite(value) for value in (first, second, third))


def two_sided_heat(z, b, c, delta, eq, geom, *, work=None, target=TARGET):
    """Apply ``D_b H_delta D_c z``; the caller keeps b/c source-independent.

    Input-side multiplication precedes diffusion and output-side multiplication
    follows it. No coefficient positivity or physical low-rank claim is made.
    """
    _validate(z, eq, geom, target)
    step = n.validate(z, delta, geom, target)
    _field(b, z, geom, 'Output factor')
    _field(c, z, geom, 'Input factor')
    answer = b*_heat(c*z, step, eq, geom, work)
    return _finite(_zero_value(answer, b*c*z, step))
