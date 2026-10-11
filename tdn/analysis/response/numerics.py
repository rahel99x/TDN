"""Differentiable finite-equation primitives, without historical target aliases.

For projected-fine targets these operators act on the *supplied* geometry.
An actual projected-fine teacher must evolve its declared finer geometry and
project; applying this coarse RHS is not an exact closure model.
"""
from __future__ import annotations
import torch
from tdn.numerics.operators import broadcast_h, check_shape, laplacian
from tdn.numerics.subflows import _LogisticFlow
from tdn.numerics.types import Geometry
from tdn.analysis.roadmap.numerics import fourier_resample, dealiased_product, quadrature
from .contracts import Physics, validate_target


def axes(u):
    return tuple(range(2, u.ndim))


def scalar(value, u):
    return broadcast_h(value, u)


def validate(u, h, geom, target):
    validate_target(target)
    check_shape(u, geom)
    if geom.ndim not in (1, 2) or u.dtype not in (torch.float32, torch.float64):
        raise ValueError('Response supports scalar FP32/FP64 fields in one or two dimensions')
    step = scalar(h, u)
    if not bool(torch.isfinite(step).all()) or bool((step < 0).any()):
        raise ValueError('Finite nonnegative horizons required')
    return step


def mean(u):
    """Per-parent spatial mean, shape [B,1,1,...]; never average parents."""
    return u.mean(axes(u), keepdim=True)


def center(u):
    return u - mean(u)


def resample(u, grid):
    return fourier_resample(u, tuple(grid))


def product(a, b, geom, target):
    validate_target(target)
    check_shape(a, geom)
    check_shape(b, geom)
    if a.shape != b.shape:
        raise ValueError('Quadratic factors must be matching fields')
    return a*b if target == 'fd_same_grid' else dealiased_product(a, b)


def mass_weights(u, geom):
    """Continuous interpolant masses: half on each even-grid Nyquist axis."""
    weight = torch.ones(geom.grid, dtype=u.dtype, device=u.device)
    for axis, n in enumerate(geom.grid):
        if n % 2 == 0:
            index = [slice(None)]*geom.ndim
            index[axis] = n//2
            weight[tuple(index)] *= .5
    return weight


def inner(a, b, geom, target):
    """Per-parent normalized physical L2 inner product, retaining spatial axes."""
    validate_target(target)
    check_shape(a, geom)
    check_shape(b, geom)
    if a.shape != b.shape:
        raise ValueError('Inner product requires matching fields')
    if target == 'fd_same_grid':
        return mean(a*b)
    fa = torch.fft.fftn(a, dim=axes(a), norm='forward')
    fb = torch.fft.fftn(b, dim=axes(b), norm='forward')
    return (mass_weights(a, geom)*(fa.conj()*fb).real).sum(axes(a), keepdim=True)


def diffusion_symbol(u, eq, geom, target):
    validate_target(target)
    check_shape(u, geom)
    value = torch.zeros(geom.grid, dtype=u.dtype, device=u.device)
    for axis, (n, dx) in enumerate(zip(geom.grid, geom.dx)):
        if target == 'fd_same_grid':
            k = torch.arange(n, dtype=u.dtype, device=u.device)
            term = -4*torch.sin(torch.pi*k/n).square()/dx**2
        else:
            term = -(2*torch.pi*torch.fft.fftfreq(n, d=dx, dtype=u.dtype, device=u.device)).square()
        shape = [1]*geom.ndim
        shape[axis] = n
        value = value + term.reshape(shape)
    return scalar(eq.kappa, u)*value


def generator(u, eq, geom, target):
    validate_target(target)
    if target == 'fd_same_grid':
        return scalar(eq.kappa, u)*laplacian(u, geom)
    return torch.fft.ifftn(torch.fft.fftn(u, dim=axes(u))*diffusion_symbol(u, eq, geom, target), dim=axes(u)).real


def heat(u, h, eq, geom, target):
    step = validate(u, h, geom, target)
    out = torch.fft.ifftn(torch.fft.fftn(u, dim=axes(u))*torch.exp(step*diffusion_symbol(u, eq, geom, target)), dim=axes(u)).real
    # Do not branch on kappa=0: its derivative is the generator even there.
    return out


def reaction_rhs(u, eq, geom, target):
    return scalar(eq.reaction_rate, u)*(u-product(u, u, geom, target))


def rhs(u, eq, geom, target):
    return generator(u, eq, geom, target)+reaction_rhs(u, eq, geom, target)


def logistic(u, h, r):
    """Exact nodal flow; no clipping. Poles outside its admissible domain are invalid."""
    rate, step = scalar(r, u), scalar(h, u)
    if not bool(torch.isfinite(rate).all()) or not bool(torch.isfinite(step).all()) or bool(((rate < 0) | (step < 0)).any()):
        raise ValueError('Finite nonnegative logistic time and reaction required')
    t = rate*step
    den = u + (1-u)*torch.exp(-t)
    if bool(((den <= 0) & (u != 0)).any()):
        raise ValueError('Logistic subflow reaches a pole outside its admissible interval')
    return _LogisticFlow.apply(u, t)


def reaction(u, h, eq, geom, target, *, substeps=8):
    step = validate(u, h, geom, target)
    if target == 'fd_same_grid':
        return logistic(u, step, eq.reaction_rate)
    if type(substeps) is not int or substeps < 1:
        raise ValueError('A fixed positive reaction subdivision count is required')
    dt, state = step/substeps, u
    f = lambda x: reaction_rhs(x, eq, geom, target)
    for _ in range(substeps):
        k1 = f(state)
        k2 = f(state+dt*k1/2)
        k3 = f(state+dt*k2/2)
        k4 = f(state+dt*k3)
        state = state+dt*(k1+2*k2+2*k3+k4)/6
    return state


def df(u, h, eq, geom, target, *, reaction_substeps=8):
    """Exact nodal Strang, or declared fixed-RK4 projected reaction Strang."""
    return heat(reaction(heat(u, h/2 if torch.is_tensor(h) else h*.5, eq, geom, target),
                         h, eq, geom, target, substeps=reaction_substeps),
                h/2 if torch.is_tensor(h) else h*.5, eq, geom, target)


def quadratic_defect(u, h, eq, geom, target, *, nodes=4, shared=True):
    """Finite-time centered quadratic coefficient; GL quadrature is approximate.

    Shared mode mirrors the previously verified batched GL4 implementation,
    while preserving tensor-valued physics and using the same projected product.
    No state-dependent cache and no float conversion of physics are used.
    """
    from .kernels import background
    step = validate(u, h, geom, target)
    c, v = mean(u), center(u)
    constant = (v == 0).all(axes(v), keepdim=True)
    c = torch.where(constant, torch.full_like(c, .5), c)
    positions, weights = quadrature(nodes)
    if not shared:
        vh = heat(v, step/2, eq, geom, target)
        split = heat(product(vh, vh, geom, target), step/2, eq, geom, target)
        total = torch.zeros_like(u)
        for x, weight in zip(positions, weights):
            s = step*x
            vs = heat(v, s, eq, geom, target)
            response = heat(product(vs, vs, geom, target), step-s, eq, geom, target)-split
            total = total+weight*background(c, s, eq.reaction_rate)['q']*response
    else:
        batch = u.shape[0]
        step = step.expand(batch, 1, *([1]*geom.ndim))
        xs = torch.tensor((*positions, .5), dtype=u.dtype, device=u.device)
        ts = xs.reshape(-1, *([1]*u.ndim))*step.unsqueeze(0)
        symbol = diffusion_symbol(u, eq, geom, target)
        spectrum = torch.fft.fftn(v, dim=axes(v))
        node_axes = tuple(range(3, u.ndim+1))
        transported = torch.fft.ifftn(spectrum.unsqueeze(0)*torch.exp(ts*symbol), dim=node_axes).real
        flat = transported.reshape((nodes+1)*batch, *u.shape[1:])
        squares = product(flat, flat, geom, target).reshape(nodes+1, *u.shape)
        spectra = torch.fft.fftn(squares, dim=node_axes)*torch.exp((step.unsqueeze(0)-ts)*symbol)
        total_spectrum = torch.zeros_like(spectrum)
        for j, weight in enumerate(weights):
            total_spectrum = total_spectrum+weight*background(c, step*positions[j], eq.reaction_rate)['q']*(spectra[j]-spectra[-1])
        total = torch.fft.ifftn(total_spectrum, dim=axes(v)).real
    answer = -scalar(eq.reaction_rate, u)*background(c, step, eq.reaction_rate)['q']*step*total
    return answer


def etdrk4_step(u, h, eq, geom, target):
    """Cox–Matthews ETDRK4 with entire kernels and live tensor physics."""
    from .kernels import phi
    step = validate(u, h, geom, target)
    z = step*diffusion_symbol(u, eq, geom, target)
    E, E2 = torch.exp(z), torch.exp(z/2)
    Q = step/2*phi(z/2, 1)
    p1, p2, p3 = phi(z, 1), phi(z, 2), phi(z, 3)
    w0 = step*(p1-3*p2+4*p3)
    wab = step*(2*p2-4*p3)
    wc = step*(-p2+4*p3)
    fft = lambda x: torch.fft.fftn(x, dim=axes(u))
    inverse = lambda x: torch.fft.ifftn(x, dim=axes(u)).real
    N = lambda x: fft(reaction_rhs(x, eq, geom, target))
    initial, n0 = fft(u), N(u)
    a = E2*initial+Q*n0; na = N(inverse(a))
    b = E2*initial+Q*na; nb = N(inverse(b))
    c = E2*a+Q*(2*nb-n0); nc = N(inverse(c))
    return inverse(E*initial+w0*n0+wab*(na+nb)+wc*nc)


def anchor(u, h, eq, geom, target, name='df', *, reaction_substeps=8, cubic_nodes=8):
    """Named controls share target, physics, output precision and input state.

    Galerkin split reactions use the explicitly recorded fixed subdivision
    count. Quadratic/cubic coefficients are quadrature approximations, not a
    full-flow correction certificate. All constituent work is performed here.
    """
    if name == 'etdrk4':
        return etdrk4_step(u, h, eq, geom, target)
    if name == 'rf':
        step = validate(u, h, geom, target)
        first = reaction(u, step/2, eq, geom, target, substeps=reaction_substeps)
        return reaction(heat(first, step, eq, geom, target), step/2, eq, geom, target, substeps=reaction_substeps)
    if name not in ('df', 'shared_gl4', 'gl4', 'sequential_gl4', 'gl8', 'gl16', 'quadratic_cubic'):
        raise ValueError('Unknown Response physical anchor')
    value = df(u, h, eq, geom, target, reaction_substeps=reaction_substeps)
    if name != 'df':
        nodes = {'gl8':8, 'gl16':16}.get(name, 4)
        value = value+quadratic_defect(u, h, eq, geom, target, nodes=nodes, shared=name!='sequential_gl4')
    if name == 'quadratic_cubic':
        from .kernels import nested_cubic_defect
        value = value+nested_cubic_defect(u, h, eq, geom, target, nodes=cubic_nodes)
    return value


def rk4_step(u, h, eq, geom, target):
    step = validate(u, h, geom, target)
    f = lambda x: rhs(x, eq, geom, target)
    k1 = f(u)
    k2 = f(u+step*k1/2)
    k3 = f(u+step*k2/2)
    k4 = f(u+step*k3)
    return u+step*(k1+2*k2+2*k3+k4)/6


def lawson_step(u, h, eq, geom, target):
    """Fourth-order explicit Lawson step with nonnegative transport intervals."""
    dt = validate(u, h, geom, target)
    H = lambda x, t: heat(x, t, eq, geom, target)
    N = lambda x: reaction_rhs(x, eq, geom, target)
    k1 = N(u)
    a = H(u+dt*k1/2, dt/2)
    k2 = N(a)
    b = H(u, dt/2)+dt*k2/2
    k3 = N(b)
    c = H(u, dt)+dt*H(k3, dt/2)
    k4 = N(c)
    return H(u, dt)+dt*(H(k1, dt)+2*H(k2+k3, dt/2)+k4)/6


def integrate(u, h, eq, geom, target, *, substeps, method='lawson'):
    """Differentiable fixed-work path. Refinement acceptance belongs to references."""
    if type(substeps) is not int or substeps < 1 or method not in ('lawson', 'rk4'):
        raise ValueError('Positive fixed substeps and lawson/rk4 method required')
    dt = validate(u, h, geom, target)/substeps
    state = u
    call = lawson_step if method == 'lawson' else rk4_step
    for _ in range(substeps):
        state = call(state, dt, eq, geom, target)
    return state
