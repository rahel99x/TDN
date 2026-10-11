"""Stable scalar kernels and finite-time interaction controls.

A converged quadrature coefficient is not an exact full finite-amplitude defect.
No clipped kernel is used to hide an unrepresentable response sensitivity.
"""
from __future__ import annotations
import math
import torch
from tdn.analysis.roadmap.numerics import quadrature
from .numerics import scalar, logistic


def phi(z, order=1):
    """Entire phi_k(z)=sum z^j/(j+k)!; stable value and derivative at zero."""
    if type(order) is not int or order not in (1, 2, 3):
        raise ValueError('Supported phi orders are 1, 2 and 3')
    z = torch.as_tensor(z)
    small = z.abs() < .1
    safe = torch.where(small, torch.ones_like(z), z)
    regular = torch.expm1(safe)
    for k in range(1, order):
        regular = regular-safe**k/math.factorial(k)
    regular = regular/safe**order
    series_z = torch.where(small, z, torch.zeros_like(z))
    term = torch.ones_like(z)/math.factorial(order)
    series = term
    for j in range(1, 20):
        term = term*series_z/(order+j)
        series = series+term
    return torch.where(small, series, regular)


def quadratic_mean_kernel(x):
    """phi_1(-2x)-exp(-x), including the x^2/6 cancellation at zero."""
    x = torch.as_tensor(x)
    small = x.abs() < .1
    series_x = torch.where(small, x, torch.zeros_like(x))
    term = torch.zeros_like(x)
    for j in range(2, 22):
        term = term+((-2)**j/math.factorial(j+1)-(-1)**j/math.factorial(j))*series_x**j
    safe = torch.where(small, torch.ones_like(x), x)
    return torch.where(small, term, phi(-2*safe)-torch.exp(-safe))


def background(c0, t, r):
    """Exact homogeneous logistic coefficients on 0<=c0<=1, t,r>=0.

    Endpoint sensitivities may be infinite when the exact value overflows the
    dtype; this is explicit, not clipped. D may underflow at c0=0 while log_q
    and log_q_ratio remain meaningful. Interior values retain physics gradients.
    """
    if not torch.is_tensor(c0) or not c0.is_floating_point():
        c0 = torch.as_tensor(c0, dtype=torch.float64)
    if bool(((c0 < 0) | (c0 > 1) | ~torch.isfinite(c0)).any()):
        raise ValueError('Homogeneous logistic response requires 0<=c0<=1')
    step = scalar(t, c0)
    rate = scalar(r, c0)
    if not bool(torch.isfinite(step).all()) or not bool(torch.isfinite(rate).all()) or bool(((step < 0) | (rate < 0)).any()):
        raise ValueError('Nonnegative time and reaction rate required')
    z = rate*step
    e = torch.exp(-z)
    D = c0+(1-c0)*e
    underflow = D == 0
    safe_D = torch.where(underflow, torch.ones_like(D), D)
    log_D = torch.where(underflow & (c0 == 0), -z, torch.log(safe_D))
    log_q = -z-2*log_D
    q = torch.exp(log_q)
    c = logistic(c0, step, rate)
    beta = -torch.expm1(-z)/safe_D
    beta = torch.where(underflow, torch.full_like(beta, float('inf')), beta)
    return dict(c=c, q=q, log_q=log_q, D=D, log_D=log_D, beta=beta,
                c_derivative=rate*c*(1-c), q_derivative=rate*(1-2*c)*q)


def log_q_ratio(c0, t, s, r):
    """log(q(t)/q(s)); avoid ratios of two underflowed sensitivities."""
    return background(c0, t, r)['log_q']-background(c0, s, r)['log_q']


def q_ratio(c0, t, s, r):
    return torch.exp(log_q_ratio(c0, t, s, r))


def nested_cubic_defect(u, h, eq, geom, target, *, nodes=8):
    from .numerics import heat, product, mean, center, validate
    step = validate(u, h, geom, target)
    c, v = mean(u), center(u)
    vh = heat(v, step/2, eq, geom, target)
    split = heat(product(vh, product(vh, vh, geom, target), geom, target), step/2, eq, geom, target)
    positions, weights = quadrature(nodes)
    outer = torch.zeros_like(u)
    for x, wx in zip(positions, weights):
        s = step*x
        vs = heat(v, s, eq, geom, target)
        inside = torch.zeros_like(u)
        for y, wy in zip(positions, weights):
            tau = s*y
            vt = heat(v, tau, eq, geom, target)
            nested = product(vs, heat(product(vt, vt, geom, target), s-tau, eq, geom, target), geom, target)
            response = heat(nested, step-s, eq, geom, target)-split
            inside = inside+wy*background(c, tau, eq.reaction_rate)['q']*response
        outer = outer+wx*background(c, s, eq.reaction_rate)['q']*s*inside
    return 2*scalar(eq.reaction_rate, u)**2*background(c, step, eq.reaction_rate)['q']*step*outer


def leading_cubic_defect(u, h, eq, geom, target, *, filtered=False):
    from .numerics import heat, product, center, generator, validate
    step = validate(u, h, geom, target)
    v = center(u)
    if filtered:
        v = heat(v, step/2, eq, geom, target)
    B = lambda a, b: product(a, b, geom, target)
    L = lambda a: generator(a, eq, geom, target)
    square, lv = B(v, v), L(v)
    value = B(lv, square)-2*B(v, B(v, lv))+2*B(v, L(square))-L(B(v, square))
    value = scalar(eq.reaction_rate, u)**2*step**3/6*value
    return heat(value, step/2, eq, geom, target) if filtered else value


def mean_quadratic_spectral(u, h, eq, geom, target, *, nodes=32):
    """Independent spectral reduction of the uncompressed quadratic mean.

    This avoids assembling any transported physical product. GL refinement is
    still required separately from parity with the full quadratic field.
    """
    from .numerics import validate, center, mean, axes, mass_weights, diffusion_symbol
    step = validate(u, h, geom, target)
    c, v = mean(u), center(u)
    lam = -diffusion_symbol(u, eq, geom, target)
    energy = torch.fft.fftn(v, dim=axes(v), norm='forward').abs().square()
    mass = torch.ones_like(lam) if target == 'fd_same_grid' else mass_weights(u, geom)
    integral = torch.zeros_like(energy)
    for x, w in zip(*quadrature(nodes)):
        s = step*x
        integral = integral+w*background(c, s, eq.reaction_rate)['q']*(torch.exp(-2*lam*s)-torch.exp(-lam*step))
    value = (mass*energy*integral).sum(axes(u), keepdim=True)
    return -scalar(eq.reaction_rate, u)*background(c, step, eq.reaction_rate)['q']*step*value
