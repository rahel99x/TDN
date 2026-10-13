"""Independent NumPy/SciPy finite equations and separately refined controls.

NumPy diffusion, scipy Fourier resampling, split-path sensitivities and DOP853
are implemented independently of the Torch primitives. Dense matrices are
verification-only and refused beyond 64 spatial degrees of freedom.
"""
from __future__ import annotations

import math
import time
import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.signal import resample

from tdn.analysis.response import numerics as n
from .tangent import (AnchorPath, validate_target, physical_norms, refined_forced,
                      response_action, refined_action, refine_solve, refined_path_response,
                      refined_source_quadrature, signed_error_decomposition, _evolve)


class NumpyEquation:
    def __init__(self, eq, geom, target):
        validate_target(target)
        self.geom, self.target = geom, target
        self.kappa = float(torch.as_tensor(eq.kappa, dtype=torch.float64).detach().cpu())
        self.rate = float(torch.as_tensor(eq.reaction_rate, dtype=torch.float64).detach().cpu())
        self.symbol = np.zeros(geom.grid, dtype=np.float64)
        for axis, (size, length) in enumerate(zip(geom.grid, geom.lengths)):
            modes = np.fft.fftfreq(size)*size
            term = (-4*(size/length)**2*np.sin(np.pi*modes/size)**2 if target == 'fd_same_grid'
                    else -(2*np.pi*modes/length)**2)
            shape = [1]*geom.ndim; shape[axis] = size
            self.symbol += self.kappa*term.reshape(shape)
        self.work = dict(rhs_calls=0, product_calls=0, heat_calls=0, anchor_calls=0, path_derivative_calls=0)

    def product(self, a, b):
        self.work['product_calls'] += 1
        if a.shape != self.geom.grid or b.shape != self.geom.grid:
            raise ValueError('Independent oracle fields require the declared spatial shape')
        if self.target == 'fd_same_grid':
            return a*b
        # A different padding size/construction from the Torch >3/2 product.
        fine_a, fine_b = a, b
        for axis, size in enumerate(self.geom.grid):
            fine_a = resample(fine_a, 2*size+1, axis=axis)
            fine_b = resample(fine_b, 2*size+1, axis=axis)
        answer = fine_a*fine_b
        for axis, size in enumerate(self.geom.grid):
            answer = resample(answer, size, axis=axis)
        return answer

    def diffusion(self, z):
        return np.fft.ifftn(np.fft.fftn(z)*self.symbol).real

    def heat(self, z, t):
        self.work['heat_calls'] += 1
        return np.fft.ifftn(np.fft.fftn(z)*np.exp(t*self.symbol)).real

    def reaction(self, z):
        return self.rate*(z-self.product(z, z))

    def reaction_tangent(self, a, z):
        return self.rate*(z-2*self.product(a, z))

    def rhs(self, z):
        self.work['rhs_calls'] += 1
        return self.diffusion(z)+self.reaction(z)

    def tangent(self, a, z, *, mode='tangent'):
        if mode == 'mean':
            return self.diffusion(z)+self.rate*(1-2*a.mean())*z
        if mode != 'tangent':
            raise ValueError('Independent response mode must be tangent or mean')
        return self.diffusion(z)+self.reaction_tangent(a, z)

    def anchor_path(self, u0, t, *, reaction_substeps=8):
        """Independent DF state and analytic derivative of its fixed RK4 path."""
        if type(reaction_substeps) is not int or reaction_substeps < 1 or not math.isfinite(t) or t < 0:
            raise ValueError('Independent path requires fixed positive work and finite time')
        self.work['anchor_calls'] += 1; self.work['path_derivative_calls'] += 1
        state = self.heat(u0, t/2); derivative = self.diffusion(state)/2
        if self.target == 'fd_same_grid':
            decay = np.exp(-self.rate*t); denominator = state+(1-state)*decay
            if np.any(denominator <= 0):
                raise FloatingPointError('Independent logistic path reaches a pole')
            following = state/denominator
            derivative = decay/denominator**2*derivative+self.rate*following*(1-following)
            state = following
        else:
            dt, ddt = t/reaction_substeps, 1/reaction_substeps
            for _ in range(reaction_substeps):
                k1 = self.reaction(state); dk1 = self.reaction_tangent(state, derivative)
                a = state+dt*k1/2; da = derivative+ddt*k1/2+dt*dk1/2
                k2 = self.reaction(a); dk2 = self.reaction_tangent(a, da)
                b = state+dt*k2/2; db = derivative+ddt*k2/2+dt*dk2/2
                k3 = self.reaction(b); dk3 = self.reaction_tangent(b, db)
                c = state+dt*k3; dc = derivative+ddt*k3+dt*dk3
                k4 = self.reaction(c); dk4 = self.reaction_tangent(c, dc)
                state = state+dt*(k1+2*k2+2*k3+k4)/6
                derivative = derivative+ddt*(k1+2*k2+2*k3+k4)/6+dt*(dk1+2*dk2+2*dk3+dk4)/6
        answer = self.heat(state, t/2)
        slope = self.diffusion(answer)/2+self.heat(derivative, t/2)
        if t == 0:
            answer, slope = u0.copy(), self.rhs(u0)
        return dict(state=answer, derivative=slope, residual=slope-self.rhs(answer))

    def dense_tangent(self, a):
        size = math.prod(self.geom.grid)
        if size > 64:
            raise ValueError('Dense tangent verification is limited to at most64 spatial degrees of freedom')
        columns = [self.tangent(a, np.eye(1, size, index, dtype=np.float64).reshape(self.geom.grid)).reshape(-1)
                   for index in range(size)]
        return np.stack(columns, axis=1)


def _numpy_initial(u):
    value = u.detach().cpu().double().numpy()
    if value.shape[:2] != (1, 1):
        raise ValueError('Independent adaptive oracle requires one scalar parent per solve')
    return value[0, 0].copy()


def independent_solve(u0, h, eq, geom, target, *, kind='teacher', z=None, s=0., mode='tangent',
                      reaction_substeps=8, tolerance=1e-12, check=None):
    equation = NumpyEquation(eq, geom, target); initial = _numpy_initial(u0)
    start, stop = float(torch.as_tensor(s, dtype=torch.float64).detach()), float(torch.as_tensor(h, dtype=torch.float64).detach())
    if not math.isfinite(start) or not math.isfinite(stop) or not 0 <= start <= stop:
        raise ValueError('Independent solve rejects reversed or invalid intervals')
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Independent tolerance must be positive and finite')
    state = initial if kind == 'teacher' else _numpy_initial(z) if kind == 'action' else np.zeros_like(initial)
    if kind not in ('teacher', 'action', 'forced', 'full_error'):
        raise ValueError('Unknown independent oracle control')
    def fun(t, flat):
        if check is not None: check()
        value = flat.reshape(geom.grid)
        if kind == 'teacher':
            answer = equation.rhs(value)
        else:
            path = equation.anchor_path(initial, t, reaction_substeps=reaction_substeps)
            answer = equation.tangent(path['state'], value, mode=mode)
            if kind in ('forced', 'full_error'):
                answer = answer-path['residual']
            if kind == 'full_error':
                answer = answer-equation.rate*equation.product(value, value)
        return answer.reshape(-1)
    if start == stop:
        return torch.from_numpy(state.copy())[None, None], dict(accepted=True, nfev=0, work=equation.work, method='independent_numpy_DOP853', dense=False)
    solution = solve_ivp(fun, (start, stop), state.reshape(-1), method='DOP853',
                         rtol=max(2.3e-14, tolerance*.1), atol=max(1e-15, tolerance*.02), t_eval=[stop])
    if not solution.success or solution.y.shape[1] != 1 or not np.isfinite(solution.y).all():
        raise FloatingPointError('Independent DOP853 did not produce a finite endpoint')
    return torch.from_numpy(solution.y[:, -1].reshape(geom.grid).copy())[None, None], dict(
        accepted=True, nfev=int(solution.nfev), work=equation.work, method='independent_numpy_DOP853', dense=False,
        component_tolerance=tolerance, uncertainty_is_certificate=False)


def _value(config, key, default):
    config = config or {}
    return config.get(key, config.get('reference', {}).get(key, default))


def refined_teacher(u0, h, eq, geom, target, *, config=None, budget=None):
    validate_target(target)
    if u0.dtype not in (torch.float32, torch.float64):
        raise ValueError('Teacher input must be an actual FP32 state or its FP64 promotion')
    initial = u0.detach().cpu().float().double()
    eq = n.Physics(torch.as_tensor(eq.kappa, dtype=torch.float64).detach().cpu(),
                   torch.as_tensor(eq.reaction_rate, dtype=torch.float64).detach().cpu())
    tol = float(_value(config, 'tolerance', 1e-10))
    limit = float(_value(config, 'max_seconds_per_query', 180.)); start = time.monotonic()
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError('Teacher acquisition requires a finite positive time cap')
    def check():
        if budget is not None: budget.check()
        if time.monotonic()-start >= limit:
            raise TimeoutError('Independent teacher query allowance exhausted')
    class Deadline:
        def check(self): check()
    solve = lambda steps: _evolve(initial, 0., h, lambda t, value:n.reaction_rhs(value, eq, geom, target),
                                  eq, geom, target, steps=steps, budget=Deadline())
    solve.norms = lambda error: physical_norms(error, geom, target)
    candidate, refinement = refine_solve(solve,
        initial_steps=int(_value(config, 'initial_steps', 8)), max_steps=int(_value(config, 'max_steps', 4096)),
        tolerance=tol*.25, budget=Deadline())
    candidate = initial if candidate is None else candidate.detach()
    cross, disagreement = None, dict(rms=None, maximum=None)
    accepted = False; reason = refinement['reason']
    if refinement['accepted']:
        try:
            independent, cross = independent_solve(initial, h, eq, geom, target, tolerance=tol*.1, check=check)
            disagreement = physical_norms(candidate-independent, geom, target)
            combined = {k: float(refinement['uncertainty_'+k])+disagreement[k] for k in ('rms', 'maximum')}
            accepted = all(value <= tol for value in combined.values())
            reason = 'TIME_AND_INDEPENDENT_SOLVER_BUDGET_ACCEPTED' if accepted else 'COMBINED_TEACHER_BUDGET_UNRESOLVED'
        except (TimeoutError, InterruptedError, FloatingPointError) as error:
            reason = type(error).__name__+': '+str(error); combined = dict(rms=None, maximum=None)
    else:
        combined = dict(rms=None, maximum=None)
    return candidate, dict(accepted=accepted, status='ACCEPTED' if accepted else 'UNRESOLVED', reason=reason,
        uncertainty_rms=combined['rms'], uncertainty_maximum=combined['maximum'], tolerance=tol,
        temporal_refinement=refinement, cross_solver=cross, cross_solver_difference=disagreement,
        component_error_budget=dict(temporal_estimate=tol*.25, cross_solver_requested_tolerance=tol*.1,
                                    combined_acceptance=tol, aggregation='temporal estimate + cross-solver discrepancy'),
        input_contract='Actual FP32 input promoted to FP64', elapsed_seconds=time.monotonic()-start,
        uncertainty_is_certificate=False, matrix_free=True)


def run_mechanism(samples, config=None, *, budget=None):
    """JSON rows plus private response tensors; no downstream promotion here."""
    config = dict(config or {}); rows = []; private = []; start = time.monotonic()
    tol = float(config.get('response_tolerance', 1e-10))
    total_budget = float(config.get('total_uncertainty_budget', 1e-10))
    if not math.isfinite(total_budget) or total_budget <= 0:
        raise ValueError('Combined mechanism uncertainty budget must be positive and finite')
    def finite_uncertainty(value):
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value >= 0)
    options = dict(initial_steps=int(config.get('response_initial_steps', 8)),
                   max_steps=int(config.get('response_max_steps', 4096)), tolerance=tol, budget=budget)
    for sample in samples:
        if budget is not None: budget.check()
        record = sample['record']; target = validate_target(record['target_kind'])
        eq = sample.get('eq', n.Physics(record['kappa'], record['reaction_rate']))
        geom = sample.get('geom', n.Geometry(tuple(record['grid']), tuple(record['lengths'])))
        u = sample['u0'].detach().double(); h = float(record['horizon'])
        teacher = sample.get('teacher64', sample.get('target'))
        if teacher is None:
            teacher, reference = refined_teacher(sample['u0'], h, eq, geom, target, config=config, budget=budget)
        else:
            reference = sample.get('reference', record.get('reference', {}))
        path = AnchorPath(u, eq, geom, target, reaction_substeps=int(config.get('reaction_substeps', 8)), budget=budget)
        a = path(h); endpoints = {'anchor': a}; uncertainties = {}
        controls = [('G', dict(mode='mean')), ('U', dict(mode='tangent')),
                    ('nonlinear1', dict(mode='tangent', nonlinear_iterations=1)),
                    ('nonlinear2', dict(mode='tangent', nonlinear_iterations=2)),
                    ('full_error_oracle', dict(full_error=True))]
        corrections = {}
        for name, control in controls:
            correction, metadata = refined_forced(path, eq, geom, target, h, **control, **options)
            uncertainties[name] = metadata
            if correction is not None:
                corrections[name] = correction; endpoints[name] = a+correction
        # Refine the same analytic-path GL source integral used by deployment,
        # separately from tangent response time work. An unused compact path
        # approximation is not a prerequisite for this analytic path.
        path_uncertainties = {}; interpolation_diagnostics = {}
        for name, mode in (('G', 'mean'), ('U', 'tangent')):
            fixed_steps = uncertainties[name]['refinements'][-1]['steps'] if uncertainties[name]['refinements'] else options['initial_steps']
            source_value, path_meta = refined_source_quadrature(path, eq, geom, target, h, mode=mode, steps=fixed_steps,
                initial_nodes=int(config.get('source_nodes', 8)), max_nodes=int(config.get('max_source_nodes', 64)),
                tolerance=float(config.get('path_tolerance', tol)), budget=budget)
            if source_value is not None and name in corrections:
                path_meta['difference_to_direct_forced'] = physical_norms(source_value-corrections[name],geom,target)
                disagreement=path_meta['difference_to_direct_forced']
                checked=all(finite_uncertainty(disagreement[key]) and disagreement[key]<=total_budget
                            for key in ('rms','maximum'))
                path_meta['direct_forced_crosscheck']=dict(accepted=checked,
                    status='RESOLVED' if checked else 'UNRESOLVED',
                    uncertainty_rms=disagreement['rms'],uncertainty_maximum=disagreement['maximum'],
                    tolerance=total_budget,
                    interpretation='Different integral route on the same analytic path; may include response time error, not an independent implementation certificate')
            path_uncertainties[name] = path_meta
            if config.get('path_interpolation_diagnostic', False):
                _, interpolation_meta = refined_path_response(path, eq, geom, target, h, mode=mode, steps=fixed_steps,
                    initial_nodes=int(config.get('source_nodes', 8)),max_nodes=int(config.get('max_source_nodes', 64)),
                    tolerance=float(config.get('path_tolerance', tol)),budget=budget)
                interpolation_meta['acceptance_role']='Optional unused-path diagnostic; excluded from analytic-path acceptance and uncertainty'
                interpolation_diagnostics[name]=interpolation_meta
        independent_path = NumpyEquation(eq, geom, target)
        derivative_checks = []
        for fraction in (0., .25, .5, .75, 1.):
            time_point = h*fraction
            torch_row = path.evaluate(time_point)
            numpy_row = independent_path.anchor_path(_numpy_initial(u), time_point, reaction_substeps=path.reaction_substeps)
            derivative_checks.append(dict(time=time_point,
                state=physical_norms(torch_row['state']-torch.from_numpy(numpy_row['state'])[None,None].to(u),geom,target),
                derivative=physical_norms(torch_row['derivative']-torch.from_numpy(numpy_row['derivative'])[None,None].to(u),geom,target),
                residual=physical_norms(torch_row['residual']-torch.from_numpy(numpy_row['residual'])[None,None].to(u),geom,target)))
        # State discrepancies have field units; derivative/residual discrepancies
        # need a time factor. This sampled estimate is not a propagated error
        # certificate or a bound on unobserved intermediate path values.
        independent_path_uncertainty = {}
        for key in ('rms', 'maximum'):
            values = [check[quantity][key] for check in derivative_checks
                      for quantity in ('state', 'derivative', 'residual')]
            independent_path_uncertainty[key] = (
                max(check['state'][key] for check in derivative_checks)
                + h*(max(check['derivative'][key] for check in derivative_checks)
                     + max(check['residual'][key] for check in derivative_checks))
                if all(finite_uncertainty(value) for value in values) else None)
        independent_path_resolved = all(finite_uncertainty(value)
            and value <= float(config.get('path_tolerance', tol))
            for value in independent_path_uncertainty.values())
        # The existing improved batched GL4 is performed here, never supplied
        # as a free cached teacher feature. Classical teacher acquisition cost
        # remains its own recorded reference workload.
        endpoints['GL4'] = n.anchor(u, h, eq, geom, target, name='shared_gl4', reaction_substeps=path.reaction_substeps)
        endpoints['classical'] = teacher
        metrics = {name: physical_norms(value-teacher.to(value), geom, target) for name, value in endpoints.items()}
        controls_status = {}
        teacher_uncertainty = {key:reference.get('uncertainty_'+key) for key in ('rms','maximum')}
        for name, metric in metrics.items():
            time_meta = uncertainties.get(name)
            source_meta = path_uncertainties.get(name)
            crosscheck=source_meta.get('direct_forced_crosscheck') if source_meta else None
            uses_path = name not in ('GL4', 'classical')
            individually_accepted = (reference.get('accepted') and
                (name in ('anchor','GL4','classical') or time_meta and time_meta['accepted']) and
                (source_meta is None or source_meta['accepted']) and
                (source_meta is None or crosscheck and crosscheck['accepted']) and
                (not uses_path or independent_path_resolved))
            for key in ('rms','maximum'):
                values = [teacher_uncertainty[key], time_meta.get('uncertainty_'+key,0.) if time_meta else 0.,
                          source_meta.get('uncertainty_'+key,0.) if source_meta else 0.,
                          crosscheck.get('uncertainty_'+key) if crosscheck else None if source_meta else 0.,
                          independent_path_uncertainty[key] if uses_path else 0.]
                metric['uncertainty_'+key] = sum(values) if all(finite_uncertainty(value) for value in values) else None
            within_total = all(finite_uncertainty(metric['uncertainty_'+key])
                and metric['uncertainty_'+key] <= total_budget
                and finite_uncertainty(metric[key]) for key in ('rms', 'maximum'))
            status = 'RESOLVED' if individually_accepted and within_total else 'UNRESOLVED'
            metric['uncertainty_budget'] = total_budget
            metric['uncertainty_acceptance'] = 'finite_nonnegative_combined_RMS_and_maximum_within_total_budget'
            metric['status'] = status; controls_status[name] = status
        deployed = sample.get('deployed_anchor32',sample.get('anchor_fp32'))
        if deployed is None:
            deployed = n.anchor(sample['u0'].float(),h,eq,geom,target,reaction_substeps=path.reaction_substeps)
        physical = teacher.to(a)-a; bridge = a-deployed.to(a); deployment = teacher.to(a)-deployed.to(a)
        row = dict(parent_id=record['parent_id'], query_id=record.get('query_id'), horizon=h, target_kind=target,
                   split=record['split'], reference=reference, controls=metrics, response_uncertainties=uncertainties,
                   source_path_uncertainties=path_uncertainties, independent_path_checks=derivative_checks,
                   optional_path_interpolation_diagnostics=interpolation_diagnostics,
                   independent_path_uncertainty=dict(uncertainty_rms=independent_path_uncertainty['rms'],
                       uncertainty_maximum=independent_path_uncertainty['maximum'],
                       status='RESOLVED' if independent_path_resolved else 'UNRESOLVED',
                       aggregation='sampled max state discrepancy + horizon*(max derivative discrepancy + max residual discrepancy)',
                       uncertainty_is_certificate=False), total_uncertainty_budget=total_budget,
                   independent_path_work=independent_path.work, control_statuses=controls_status,
                   primary_status='RESOLVED' if all(controls_status.get(name)=='RESOLVED' for name in ('G','U')) else 'UNRESOLVED',
                   primary_case=record.get('primary_case',True), intervention=record.get('intervention','original'),
                   role=record.get('role','primary'), regime=record.get('regime'),
                   physical_defect=physical_norms(physical,geom,target), arithmetic_bridge=physical_norms(bridge,geom,target),
                   deployment_defect=physical_norms(deployment,geom,target),
                   bridge_identity_maximum=float((deployment-physical-bridge).abs().max()),
                   path_work=path.work(), dense=False, maximum_scope='sampled nodal maximum',
                   status='RESOLVED' if all(value=='RESOLVED' for value in controls_status.values()) else 'UNRESOLVED')
        row.update({key:record[key] for key in (
            'grid','lengths','kappa','reaction_rate','cohort','panel','phase_shift','extrapolation',
            'input_sha256','active_frequency','diffusion_number','reaction_number') if key in record})
        if 'U' in corrections and 'G' in corrections:
            row['linear_response_and_remainder_decomposition'] = signed_error_decomposition(
                physical-corrections['U'],corrections['U']-corrections['G'],geom,target)
        rows.append(row); private.append(dict(record=record, u0=sample['u0'], smooth_anchor64=a,
                                             teacher64=teacher, corrections=corrections, endpoints=endpoints,
                                             physical_defect=physical,arithmetic_bridge=bridge,deployment_defect=deployment))
    return dict(rows=rows, private_tensors=private, elapsed_seconds=time.monotonic()-start,
                scientific_claim='Diagnostic physical controls; mechanism promotion requires the separately frozen gate')
