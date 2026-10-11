"""Differentiable reference paths and exact/approximate response identities."""
from __future__ import annotations
import torch
from tdn.analysis.roadmap.numerics import quadrature
from .numerics import (anchor, rhs, generator, product, mean, center, inner,
                       scalar, validate, heat, integrate)
from .kernels import q_ratio
from .contracts import Physics


def anchor_path(u, t, eq, geom, target, *, name='df', reaction_substeps=8, create_graph=True):
    """Smooth fixed-work anchor path, not the derivative of rounded FP32 deployment.

    JVP differentiates the declared finite RK4 reaction path for Galerkin; no
    detached horizon-dependent subdivision count changes its definition.
    """
    with torch.inference_mode(False), torch.enable_grad():
        u = u.clone() if torch.is_inference(u) else u
        step = validate(u, t, geom, target)
        step = step.clone() if torch.is_inference(step) else step
        kap = eq.kappa.clone() if torch.is_tensor(eq.kappa) and torch.is_inference(eq.kappa) else eq.kappa
        rate = eq.reaction_rate.clone() if torch.is_tensor(eq.reaction_rate) and torch.is_inference(eq.reaction_rate) else eq.reaction_rate
        eq = Physics(kap, rate)
        state, derivative = torch.autograd.functional.jvp(
            lambda time: anchor(u, time, eq, geom, target, name=name,
                                reaction_substeps=reaction_substeps),
            step, torch.ones_like(step), create_graph=create_graph, strict=False)
        residual = derivative-rhs(state, eq, geom, target)
    return dict(state=state, derivative=derivative, residual=residual,
                path='fixed_substep_smooth_anchor', reaction_substeps=reaction_substeps)


def reference_path(u, t, eq, geom, target, *, substeps, method='lawson', create_graph=True):
    """Differentiable numerical reference with a declared fixed count, not a certificate."""
    with torch.inference_mode(False), torch.enable_grad():
        u = u.clone() if torch.is_inference(u) else u
        step = validate(u, t, geom, target)
        step = step.clone() if torch.is_inference(step) else step
        kap = eq.kappa.clone() if torch.is_tensor(eq.kappa) and torch.is_inference(eq.kappa) else eq.kappa
        rate = eq.reaction_rate.clone() if torch.is_tensor(eq.reaction_rate) and torch.is_inference(eq.reaction_rate) else eq.reaction_rate
        eq = Physics(kap, rate)
        state, derivative = torch.autograd.functional.jvp(
            lambda time: integrate(u, time, eq, geom, target, substeps=substeps, method=method),
            step, torch.ones_like(step), create_graph=create_graph, strict=False)
    return dict(state=state, derivative=derivative, residual=derivative-rhs(state, eq, geom, target),
                substeps=substeps, method=method)


def full_error_identity(truth, reference, reference_derivative, eq, geom, target, *, closure_forcing=None):
    """Two independently assembled sides of e'=f(u)-a'. Same equation only.

    For a projected-fine truth, callers must supply fine fields/geometry, or add
    its separately measured closure forcing. Coarse fields alone do not satisfy
    this identity with the coarse Galerkin RHS.
    """
    if target == 'galerkin_projected_fine' and closure_forcing is None:
        raise ValueError('Projected-fine error identity needs explicit projected closure forcing')
    closure = torch.zeros_like(truth) if closure_forcing is None else closure_forcing
    e = truth-reference
    rho = reference_derivative-rhs(reference, eq, geom, target)
    rate = scalar(eq.reaction_rate, truth)
    direct = rhs(truth, eq, geom, target)+closure-reference_derivative
    expanded = generator(e, eq, geom, target)+rate*e-2*rate*product(reference, e, geom, target)-rate*product(e, e, geom, target)-rho+closure
    return dict(error=e, residual=rho, direct=direct, expanded=expanded, difference=direct-expanded)


def moment_rhs(u, eq, geom, target):
    m, v = mean(u), center(u)
    V = inner(v, v, geom, target)
    rate = scalar(eq.reaction_rate, u)
    mean_rate = rate*(m*(1-m)-V)
    variance_rate = 2*inner(v, generator(v, eq, geom, target), geom, target)+2*rate*(1-2*m)*V-2*rate*inner(v, product(v, v, geom, target), geom, target)
    return dict(mean=m, variance=V, mean_derivative=mean_rate, variance_derivative=variance_rate)


def actual_source(truth, reference, reference_derivative, eq, geom, target, *, closure_forcing=None):
    """Exact centered source including mean mismatch and nonlinear feedback.

    This is teacher-only diagnostic supervision, never an inference input.
    """
    if target == 'galerkin_projected_fine' and closure_forcing is None:
        raise ValueError('Projected-fine source needs explicit projected closure forcing')
    closure = torch.zeros_like(truth) if closure_forcing is None else closure_forcing
    m, ma = mean(truth), mean(reference)
    va, d = center(reference), center(truth)-center(reference)
    rho = reference_derivative-rhs(reference, eq, geom, target)
    rate = scalar(eq.reaction_rate, truth)
    forcing = -center(rho)-2*rate*(m-ma)*va-rate*center(2*product(va, d, geom, target)+product(d, d, geom, target))
    return dict(source=center(forcing+closure), spatial_error=d, mean=m, reference_mean=ma,
                residual=rho, centered_residual=center(rho))


def mean_propagator(mean_path, s, h, eq, like, *, nodes=8):
    """Scalar exp(r integral_s^h [1-2m(t)]dt), GL approximation to mean clock."""
    start, stop = scalar(s, like), scalar(h, like)
    integral = torch.zeros_like(mean(like))
    for x, weight in zip(*quadrature(nodes)):
        integral = integral+weight*(1-2*mean_path(start+(stop-start)*x))
    return torch.exp(scalar(eq.reaction_rate, like)*(stop-start)*integral)


def source_response(sources, h, eq, geom, target, mean_path, *, nodes=8, mean_nodes=8):
    """Approximate coupled variation-of-constants integral with explicit GL work.

    sources(s) returns [B,1,*grid]; mean_path(s) returns [B,1,1,...]. Source
    projection is structural. Source and mean call counts are nodes and
    nodes*mean_nodes. There is no access to future truth in this interface.
    """
    probe = sources(torch.as_tensor(h)*0)
    step = validate(probe, h, geom, target)
    value = torch.zeros_like(probe)
    for x, weight in zip(*quadrature(nodes)):
        s = step*x
        field = center(sources(s))
        factor = mean_propagator(mean_path, s, step, eq, field, nodes=mean_nodes)
        value = value+weight*factor*heat(field, step-s, eq, geom, target)
    return center(step*value)


def homogeneous_residual_response(u0, h, eq, geom, target, *, name='df', nodes=16, reaction_substeps=8, create_graph=False):
    """First homogeneous-response correction only; omitted feedback is nonzero."""
    step = validate(u0, h, geom, target)
    c0 = mean(u0)
    correction = torch.zeros_like(u0)
    for x, weight in zip(*quadrature(nodes)):
        s = step*x
        rho = anchor_path(u0, s, eq, geom, target, name=name,
                          reaction_substeps=reaction_substeps, create_graph=create_graph)['residual']
        correction = correction-weight*q_ratio(c0, step, s, eq.reaction_rate)*heat(rho, step-s, eq, geom, target)
    return step*correction


def conditional_clock(m0, means, variances, r, weights):
    """Clock from supplied trajectory quadrature, conditional on physical [0,1].

    ``means`` and ``variances`` contain corresponding per-parent observations;
    ``weights`` include physical time-interval length. This teacher diagnostic
    does not infer interpolant bounds from nodal extrema. An inconsistent
    variance is rejected rather than clipped into the admissible interval.
    """
    if not (len(means) == len(variances) == len(weights)) or not len(means):
        raise ValueError('Clock quadrature requires matching nonempty observations and weights')
    if bool(((m0 <= 0) | (m0 >= 1) | ~torch.isfinite(m0)).any()):
        raise ValueError('Conditional clock requires an interior initial mean')
    integral = torch.zeros_like(m0)
    for m, V, weight in zip(means, variances, weights):
        if bool(((m <= 0) | (m >= 1) | (V < 0) | (V > m*(1-m)+1e-12) |
                 ~torch.isfinite(m) | ~torch.isfinite(V)).any()):
            raise ValueError('Variance/mean violates the conditional physical interval contract')
        w = scalar(weight, m0)
        if not bool(torch.isfinite(w).all()) or bool((w < 0).any()):
            raise ValueError('Clock quadrature weights must be finite and nonnegative')
        integral = integral+w*(1-V/(m*(1-m)))
    return torch.sigmoid(torch.logit(m0)+scalar(r, m0)*integral)
