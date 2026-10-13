"""Matrix-free full-field tangent and forced error evolution.

The smooth FP64 path is generated from the actual initial state. It is an
explicitly charged diagnostic/inference operation, never a future teacher.
All evolution uses an exact diffusion Lawson step, and never allocates a dense
spatial operator. Response, path interpolation and teacher refinement have
separate interfaces and uncertainty records.
"""
from __future__ import annotations

import math
import time
import torch

from tdn.analysis.response import numerics as n
from tdn.analysis.response.trajectory import anchor_path

TARGETS = ('fd_same_grid', 'galerkin_same_grid')


def validate_target(target):
    if target not in TARGETS:
        raise ValueError('UNSUPPORTED_TARGET: Tangent Response requires same-grid FD or Galerkin; projected fine needs explicit closure forcing')
    return target


def _check(budget):
    if budget is not None:
        budget.check()


def physical_norms(value, geom, target):
    validate_target(target)
    if not bool(torch.isfinite(value).all()):
        return dict(rms=None, maximum=None, mean=None, mean_error=None, centered_rms=None,
                    mean_mse=None, centered_mse=None, mean_squared_error_fraction=None)
    total = n.inner(value, value, geom, target).mean()
    mean_mse = n.mean(value).square().mean()
    centered = n.inner(n.center(value), n.center(value), geom, target).mean()
    return dict(rms=float(total.sqrt().detach()),
                maximum=float(value.abs().max().detach()), mean=float(n.mean(value).abs().mean().detach()),
                mean_error=float(n.mean(value).mean().detach()), mean_mse=float(mean_mse.detach()),
                centered_rms=float(centered.sqrt().detach()), centered_mse=float(centered.detach()),
                mean_squared_error_fraction=float((mean_mse/total).detach()) if bool(total > 0) else None)


def finite_product_constant(geom, target):
    """A finite-grid physical-L2 product constant, not grid-uniform stability.

    FD uses ||v||_infinity<=sqrt(N)||v||_nodal. The Galerkin interpolant's
    orthonormal real basis has evaluation norm bounded by sqrt(product(n+1))
    on even axes and sqrt(n) on odd axes. Orthogonal product projection is
    nonexpansive in the declared physical inner product.
    """
    validate_target(target)
    return math.sqrt(math.prod(size+(size%2==0 and target=='galerkin_same_grid') for size in geom.grid))


def conditional_first_correction_bound(response_error_integral, nonlinear_error_integral, eq, geom, target, *,
                                       residual_path_error=0., quadrature_error=0., arithmetic_error=0.):
    """Conditional norm inequality with separately supplied controlled terms.

    response_error_integral bounds integral||(U-Utheta)rho||; nonlinear term
    bounds integral M(t,s)||e(s)||², with an independently justified growth M.
    This function does NOT turn sampled values or refinement into certificates.
    """
    values = (response_error_integral, nonlinear_error_integral, residual_path_error, quadrature_error, arithmetic_error)
    if any(not math.isfinite(float(value)) or float(value)<0 for value in values):
        raise ValueError('Conditional bound terms require finite nonnegative controlled quantities')
    rate = float(torch.as_tensor(eq.reaction_rate,dtype=torch.float64).detach())
    constant = finite_product_constant(geom,target)
    nonlinear = rate*constant*float(nonlinear_error_integral)
    return dict(bound=float(response_error_integral)+nonlinear+float(residual_path_error)+float(quadrature_error)+float(arithmetic_error),
                response_approximation=float(response_error_integral), nonlinear_feedback=nonlinear,
                residual_path=float(residual_path_error),quadrature=float(quadrature_error),arithmetic=float(arithmetic_error),
                product_constant=constant, grid_uniform=False, certified=False,
                assumptions='Exact finite equation; physical norm; valid response/growth and integral controls supplied separately; samples alone do not establish them')


def signed_error_decomposition(first, second, geom, target):
    """Retain cancellation instead of summing component squared errors."""
    a = float(n.inner(first,first,geom,target).mean())
    b = float(n.inner(second,second,geom,target).mean())
    cross = float(2*n.inner(first,second,geom,target).mean())
    total = float(n.inner(first+second,first+second,geom,target).mean())
    return dict(first_mse=a,second_mse=b,signed_cross_term=cross,total_mse=total,
                algebraic_gap=total-a-b-cross,scope='Paired vector identity; cancellation permitted')


def conditional_growth_bound(eq, interval, *, anchor_lower_bound):
    """Conditional ||U(t,s)||<=exp(r(1-2*lower)*(t-s)).

    Lower must bound the full path at all spatial points and intervening times.
    Galerkin nodal minima do not establish this interpolant/path assumption.
    Diffusion is dissipative; no universal reaction contraction is imposed.
    """
    rate=float(torch.as_tensor(eq.reaction_rate,dtype=torch.float64).detach())
    duration,lower=float(interval),float(anchor_lower_bound)
    if not math.isfinite(duration) or duration<0 or not math.isfinite(lower):
        raise ValueError('Conditional growth requires finite interval and global path bound')
    exponent=rate*(1-2*lower)*duration
    if not math.isfinite(exponent) or exponent>math.log(torch.finfo(torch.float64).max):
        raise OverflowError('Conditional growth is not representable; no clipping')
    representable=exponent>=math.log(torch.finfo(torch.float64).tiny)
    return dict(log_growth=exponent,growth=math.exp(exponent) if representable else None,
        status='REPRESENTABLE' if representable else 'POSITIVE_BOUND_UNDERFLOWS',certified=False,
        assumption='Global spatial/time anchor lower bound supplied separately; samples alone are insufficient')


def tangent_action(a, z, eq, geom, target):
    """A_a z=Lz+rz-2rB(a,z), including spatial-mean coupling."""
    validate_target(target); n.validate(a, 0., geom, target); n.validate(z, 0., geom, target)
    return n.generator(z, eq, geom, target)+n.scalar(eq.reaction_rate, z)*(z-2*n.product(a, z, geom, target))


def mean_action(a, z, eq, geom, target):
    validate_target(target)
    return n.generator(z, eq, geom, target)+n.scalar(eq.reaction_rate, z)*(1-2*n.mean(a))*z


def spatial_feedback(a, z, eq, geom, target):
    validate_target(target)
    return -2*n.scalar(eq.reaction_rate, z)*n.product(n.center(a), z, geom, target)


class AnchorPath:
    """Fixed-work smooth path with cache/work counters, independent of horizon.

    ``create_graph=True`` is required when differentiating time/physics/path
    dependence. Detached caches are used only in ordinary numerical diagnostics.
    """
    def __init__(self, u0, eq, geom, target, *, anchor_name='df', reaction_substeps=8,
                 create_graph=False, budget=None):
        validate_target(target)
        if u0.dtype != torch.float64:
            raise ValueError('Smooth mathematical paths require FP64 initial values')
        n.validate(u0, 0., geom, target)
        if type(reaction_substeps) is not int or reaction_substeps < 1:
            raise ValueError('Fixed positive reaction work is required')
        self.u0, self.eq, self.geom, self.target = u0, eq, geom, target
        self.anchor_name, self.reaction_substeps = anchor_name, reaction_substeps
        self.create_graph, self.budget = bool(create_graph), budget
        self._cache = {}
        tensors = [u0]+[value for value in (eq.kappa, eq.reaction_rate) if torch.is_tensor(value)]
        self._identity = []
        for value in tensors:
            try: version = value._version
            except RuntimeError: version = None
            self._identity.append((value, version, value.clone() if version is None else None))
        self._work = dict(path_queries=0, anchor_evaluations=0, derivative_evaluations=0,
                          residual_evaluations=0, mean_queries=0, cache_hits=0)

    def _validate_identity(self):
        for value, version, snapshot in self._identity:
            try: current = value._version
            except RuntimeError: current = None
            if current != version or snapshot is not None and not torch.equal(value, snapshot):
                raise ValueError('Stale anchor path: input or physics changed; regenerate its fixed work')

    def _key(self, t):
        value = torch.as_tensor(t)
        if self.create_graph or value.requires_grad:
            return None
        return (tuple(value.shape), tuple(value.detach().cpu().reshape(-1).tolist()))

    def __call__(self, t):
        self._validate_identity()
        _check(self.budget); self._work['path_queries'] += 1
        key = self._key(t)
        if key is not None and key in self._cache:
            self._work['cache_hits'] += 1
            return self._cache[key]['state']
        step = n.validate(self.u0, t, self.geom, self.target)
        state = n.anchor(self.u0, step, self.eq, self.geom, self.target,
                         name=self.anchor_name, reaction_substeps=self.reaction_substeps)
        # Preserve literal initial values without killing the smooth path's
        # right-hand time derivative or doubling its input derivative.
        initial_value = self.u0.detach()+(state-state.detach())
        state = torch.where(step == 0, initial_value, state)
        self._work['anchor_evaluations'] += 1
        if key is not None:
            self._cache[key] = dict(state=state.detach())
        return state

    def evaluate(self, t):
        self._validate_identity()
        _check(self.budget); self._work['path_queries'] += 1
        key = self._key(t)
        if key is not None and 'residual' in self._cache.get(key, {}):
            self._work['cache_hits'] += 1
            return self._cache[key]
        value = anchor_path(self.u0, t, self.eq, self.geom, self.target,
                            name=self.anchor_name, reaction_substeps=self.reaction_substeps,
                            create_graph=self.create_graph)
        step = n.validate(self.u0, t, self.geom, self.target)
        initial_value = self.u0.detach()+(value['state']-value['state'].detach())
        state = torch.where(step == 0, initial_value, value['state'])
        derivative = value['derivative']
        result = dict(state=state, derivative=derivative,
                      residual=derivative-n.rhs(state, self.eq, self.geom, self.target))
        self._work['anchor_evaluations'] += 1; self._work['derivative_evaluations'] += 1
        self._work['residual_evaluations'] += 1
        if key is not None:
            result = {k: v.detach() for k, v in result.items()}; self._cache[key] = result
        return result

    def mean(self, t):
        self._work['mean_queries'] += 1
        return n.mean(self(t))

    def work(self):
        return dict(self._work, path='smooth_fixed_work_anchor', anchor=self.anchor_name,
                    reaction_substeps=self.reaction_substeps, cached_nodes=len(self._cache))


def residual(path, t):
    return path.evaluate(t)['residual']


def source_nodes(nodes):
    from tdn.analysis.roadmap.numerics import quadrature
    return quadrature(nodes)


class SampledAnchorPath:
    """Fixed-node C1 cubic Hermite path; derivative and residual are consistent.

    This is an explicitly different approximate path, used only to measure
    source/path sensitivity with tangent integration work held fixed.
    """
    def __init__(self, path, h, *, nodes):
        if type(nodes) is not int or nodes < 2 or path.u0.shape[0] != 1:
            raise ValueError('Sampled path requires at least2 nodes and one parent/query')
        stop = n.scalar(h, path.u0)
        if not bool(torch.isfinite(stop).all()) or not bool((stop > 0).all()):
            raise ValueError('Sampled path requires a positive finite endpoint')
        self.u0, self.eq, self.geom, self.target = path.u0, path.eq, path.geom, path.target
        self.h, self.nodes = stop, nodes
        self.times = [stop*i/(nodes-1) for i in range(nodes)]
        self.samples = [path.evaluate(t) for t in self.times]
        self.queries = 0

    def evaluate(self, t):
        self.queries += 1
        step = n.validate(self.u0, t, self.geom, self.target)
        if bool((step > self.h).any()):
            raise ValueError('Sampled path query is outside its declared horizon')
        index = int(((step/self.h)*(self.nodes-1)).floor().clamp(max=self.nodes-2).detach())
        dt = self.times[index+1]-self.times[index]; x = (step-self.times[index])/dt
        left, right = self.samples[index], self.samples[index+1]
        a, b, da, db = left['state'], right['state'], left['derivative'], right['derivative']
        state = (2*x**3-3*x**2+1)*a+(x**3-2*x**2+x)*dt*da+(-2*x**3+3*x**2)*b+(x**3-x**2)*dt*db
        derivative = (6*x**2-6*x)*a/dt+(3*x**2-4*x+1)*da+(-6*x**2+6*x)*b/dt+(3*x**2-2*x)*db
        return dict(state=state, derivative=derivative,
                    residual=derivative-n.rhs(state, self.eq, self.geom, self.target))

    def __call__(self, t):
        return self.evaluate(t)['state']

    def mean(self, t):
        return n.mean(self(t))

    def work(self):
        return dict(path='sampled_C1_cubic_Hermite', nodes=self.nodes, path_queries=self.queries)


def _lawson_step(state, t, dt, nonlinear, eq, geom, target):
    H = lambda value, amount: n.heat(value, amount, eq, geom, target)
    k1 = nonlinear(t, state)
    k2 = nonlinear(t+dt/2, H(state+dt*k1/2, dt/2))
    k3 = nonlinear(t+dt/2, H(state, dt/2)+dt*k2/2)
    k4 = nonlinear(t+dt, H(state, dt)+dt*H(k3, dt/2))
    return H(state, dt)+dt/6*(H(k1, dt)+2*H(k2+k3, dt/2)+k4)


def _evolve(initial, start, stop, nonlinear, eq, geom, target, *, steps, return_history=False, budget=None):
    validate_target(target)
    if initial.dtype != torch.float64:
        raise ValueError('Tangent numerical evolution requires FP64')
    if type(steps) is not int or not 1 <= steps <= 1_000_000:
        raise ValueError('Declare a bounded fixed positive integration count')
    a, b = n.validate(initial, start, geom, target), n.validate(initial, stop, geom, target)
    if bool((b < a).any()):
        raise ValueError('Reversed response intervals are invalid')
    dt = (b-a)/steps; state = initial
    states, times = [state], [a]
    for index in range(steps):
        _check(budget)
        t = a+dt*index
        state = _lawson_step(state, t, dt, nonlinear, eq, geom, target)
        if not bool(torch.isfinite(state).all()):
            raise FloatingPointError('Nonfinite tangent/forced evolution; no clipping')
        if return_history:
            states.append(state); times.append(a+dt*(index+1))
    boundary_value = initial.detach()+(state-state.detach())
    state = torch.where(b == a, boundary_value, state)
    if not return_history:
        return state
    states[-1] = state
    return dict(endpoint=state, states=torch.stack(states), times=torch.stack(times),
                steps=steps, rhs_evaluations=4*steps, method='matrix_free_Lawson_RK4_exact_diffusion')


def _reaction_linear(a, z, eq, geom, target, mode):
    r = n.scalar(eq.reaction_rate, z)
    if mode == 'mean':
        return r*(1-2*n.mean(a))*z
    if mode != 'tangent':
        raise ValueError('Response mode must be tangent or mean')
    return r*(z-2*n.product(a, z, geom, target))


def response_action(path, eq, geom, target, z, s, h, *, steps=32, mode='tangent',
                    return_history=False, budget=None):
    return _evolve(z, s, h, lambda t, value: _reaction_linear(path(t), value, eq, geom, target, mode),
                   eq, geom, target, steps=steps, return_history=return_history, budget=budget)


def matched_mean_action(path, eq, geom, target, z, s, h, *, steps=32, return_history=False, budget=None):
    return response_action(path, eq, geom, target, z, s, h, steps=steps, mode='mean',
                           return_history=return_history, budget=budget)


def _interpolate_history(history, t, eq, geom, target, nonlinear):
    # Cubic Hermite interpolation of the PREVIOUS Picard trajectory. It does
    # not inspect the current iterate or a future solution teacher.
    times, values = history['times'], history['states']
    fraction = ((t-times[0])/(times[-1]-times[0]).clamp_min(torch.finfo(t.dtype).tiny))*history['steps']
    index = fraction.floor().long().clamp(max=history['steps']-1)
    if index.numel() != 1:
        raise ValueError('Picard dense trajectory currently requires one parent/query')
    i = int(index.detach()); dt = times[i+1]-times[i]; x = (t-times[i])/dt.clamp_min(torch.finfo(t.dtype).tiny)
    y0, y1 = values[i], values[i+1]
    f0 = n.generator(y0, eq, geom, target)+nonlinear(times[i], y0)
    f1 = n.generator(y1, eq, geom, target)+nonlinear(times[i+1], y1)
    return (2*x**3-3*x**2+1)*y0+(x**3-2*x**2+x)*dt*f0+(-2*x**3+3*x**2)*y1+(x**3-x**2)*dt*f1


def forced_response(path, eq, geom, target, h, *, steps=32, mode='tangent', nonlinear_iterations=0,
                    full_error=False, return_history=False, budget=None):
    if type(nonlinear_iterations) is not int or nonlinear_iterations not in (0, 1, 2):
        raise ValueError('Only separately declared zero/one/two Picard iterations are supported')
    if full_error and (mode != 'tangent' or nonlinear_iterations):
        raise ValueError('Full error oracle requires tangent dynamics and no Picard iteration flag')
    zero = torch.zeros_like(path.u0)
    def linear(t, value):
        row = path.evaluate(t)
        return _reaction_linear(row['state'], value, eq, geom, target, mode)-row['residual']
    if full_error:
        fun = lambda t, value: linear(t, value)-n.scalar(eq.reaction_rate, value)*n.product(value, value, geom, target)
        return _evolve(zero, 0., h, fun, eq, geom, target, steps=steps, return_history=return_history, budget=budget)
    previous = _evolve(zero, 0., h, linear, eq, geom, target, steps=steps,
                       return_history=bool(nonlinear_iterations) or return_history, budget=budget)
    previous_fun = linear
    for _ in range(nonlinear_iterations):
        history, old_fun = previous, previous_fun
        def picard(t, value, history=history, old_fun=old_fun):
            old = _interpolate_history(history, t, eq, geom, target, old_fun)
            return linear(t, value)-n.scalar(eq.reaction_rate, old)*n.product(old, old, geom, target)
        previous = _evolve(zero, 0., h, picard, eq, geom, target, steps=steps, return_history=True, budget=budget)
        previous_fun = picard
    return previous if return_history or not nonlinear_iterations else previous['endpoint']


def refine_solve(solve, *, initial_steps=16, max_steps=4096, tolerance=1e-10, budget=None):
    if type(initial_steps) is not int or type(max_steps) is not int or not 1 <= initial_steps <= max_steps:
        raise ValueError('Declare ordered positive fixed refinement counts')
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Refinement tolerance must be positive and finite')
    previous = None; history = []; last = None; uncertainty = dict(rms=None, maximum=None)
    start = time.monotonic(); accepted = False; reason = 'REFINEMENT_CAP'
    steps = initial_steps
    try:
        while steps <= max_steps:
            _check(budget); last = solve(steps); _check(budget)
            if previous is not None:
                uncertainty = solve.norms(last-previous)
                accepted = all(uncertainty[k] is not None and uncertainty[k] <= tolerance for k in ('rms', 'maximum'))
            history.append(dict(steps=steps, difference=uncertainty.copy()))
            if accepted:
                reason = 'INDEPENDENT_TIME_REFINEMENT'; break
            previous = last; steps *= 2
    except (TimeoutError, InterruptedError, FloatingPointError) as error:
        reason = type(error).__name__+': '+str(error)
    return last, dict(accepted=accepted, status='ACCEPTED' if accepted else 'UNRESOLVED', reason=reason,
                     uncertainty_rms=uncertainty['rms'], uncertainty_maximum=uncertainty['maximum'],
                     refinements=history, elapsed_seconds=time.monotonic()-start, uncertainty_is_certificate=False)


def refined_action(path, eq, geom, target, z, s, h, *, mode='tangent', initial_steps=16, max_steps=4096,
                   tolerance=1e-10, budget=None):
    solve = lambda steps: response_action(path, eq, geom, target, z, s, h, steps=steps, mode=mode, budget=budget)
    solve.norms = lambda value: physical_norms(value, geom, target)
    return refine_solve(solve, initial_steps=initial_steps, max_steps=max_steps, tolerance=tolerance, budget=budget)


def refined_forced(path, eq, geom, target, h, *, mode='tangent', nonlinear_iterations=0, full_error=False,
                   initial_steps=16, max_steps=4096, tolerance=1e-10, budget=None):
    solve = lambda steps: forced_response(path, eq, geom, target, h, steps=steps, mode=mode,
                                         nonlinear_iterations=nonlinear_iterations, full_error=full_error, budget=budget)
    solve.norms = lambda value: physical_norms(value, geom, target)
    return refine_solve(solve, initial_steps=initial_steps, max_steps=max_steps, tolerance=tolerance, budget=budget)


def source_quadrature(path, eq, geom, target, h, *, nodes=8, steps=32,
                      mode='tangent', budget=None, _work=None):
    """Actual analytic-path GL source integral, -integral U(h,s) rho_a(s) ds.

    Homogeneous actions keep their declared integration work while source
    nodes vary. No compact or Hermite approximation replaces the deployed
    fixed-work analytic anchor path. All actions remain matrix-free.
    """
    validate_target(target)
    if not isinstance(path, AnchorPath):
        raise ValueError('Analytic source quadrature requires the actual AnchorPath')
    if path.target != target or path.geom != geom:
        raise ValueError('Source quadrature geometry and target differ from the anchor path')
    if mode not in ('mean', 'tangent'):
        raise ValueError('Response mode must be tangent or mean')
    if type(steps) is not int or not 1 <= steps <= 1_000_000:
        raise ValueError('Declare bounded positive fixed homogeneous action work')
    positions, weights = source_nodes(nodes)
    stop = n.validate(path.u0, h, geom, target)
    path._validate_identity(); _check(budget)
    result = torch.zeros_like(path.u0)
    for position, weight in zip(positions, weights):
        _check(budget)
        source_time = stop*position
        rho = residual(path, source_time)
        if _work is not None: _work['source_residual_queries'] += 1
        if _work is not None: _work['homogeneous_actions_started'] += 1
        value = response_action(path, eq, geom, target, rho, source_time, stop,
                                steps=steps, mode=mode, budget=budget)
        if _work is not None: _work['homogeneous_actions_completed'] += 1
        result = result-stop*weight*value
    return result


def refined_source_quadrature(path, eq, geom, target, h, *, steps=32, mode='tangent',
                              initial_nodes=8, max_nodes=64, tolerance=1e-10, budget=None):
    """Refine actual GL forcing opportunities at fixed response integration work."""
    # Validate the complete node menu before a null can bypass any work.
    source_nodes(initial_nodes); source_nodes(max_nodes)
    if type(initial_nodes) is not int or type(max_nodes) is not int or initial_nodes > max_nodes:
        raise ValueError('Declare ordered positive bounded source quadrature counts')
    before = path.work()
    work = dict(source_residual_queries=0, homogeneous_actions_started=0, homogeneous_actions_completed=0)
    solve = lambda nodes: source_quadrature(path, eq, geom, target, h,
                                           nodes=nodes, steps=steps, mode=mode, budget=budget, _work=work)
    solve.norms = lambda value: physical_norms(value, geom, target)
    answer, metadata = refine_solve(solve, initial_steps=initial_nodes, max_steps=max_nodes,
                                    tolerance=tolerance, budget=budget)
    after = path.work()
    path_work = {key: after[key]-before[key] for key in before if isinstance(before[key], int)}
    metadata.update(axis='actual_analytic_path_Gauss_Legendre_source_nodes',
        path='smooth_fixed_work_anchor', fixed_tangent_steps=steps, mode=mode,
        source_nodes_evaluated=work['source_residual_queries'],
        homogeneous_actions=work['homogeneous_actions_completed'],
        homogeneous_actions_started=work['homogeneous_actions_started'],
        homogeneous_rhs_evaluations=path_work['path_queries']-work['source_residual_queries'],
        path_work=path_work,
        dense=False, unused_path_interpolation=False)
    return answer, metadata


def refined_path_response(path, eq, geom, target, h, *, steps=32, mode='tangent', initial_nodes=8,
                          max_nodes=256, tolerance=1e-10, budget=None):
    if bool((n.scalar(h, path.u0) == 0).all()):
        return torch.zeros_like(path.u0), dict(accepted=True, status='ACCEPTED', uncertainty_rms=0.,
            uncertainty_maximum=0., refinements=[], axis='null zero-time path', uncertainty_is_certificate=False)
    solve = lambda nodes: forced_response(SampledAnchorPath(path, h, nodes=nodes), eq, geom, target, h,
                                           steps=steps, mode=mode, budget=budget)
    solve.norms = lambda value: physical_norms(value, geom, target)
    answer, meta = refine_solve(solve, initial_steps=initial_nodes, max_steps=max_nodes,
                                tolerance=tolerance, budget=budget)
    meta.update(axis='anchor/source interpolation nodes', fixed_tangent_steps=steps,
                interpolation='C1 cubic Hermite; derivative of the same interpolated state')
    return answer, meta
