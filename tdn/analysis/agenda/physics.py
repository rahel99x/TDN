"""Independent physical response diagnostics for periodic logistic diffusion.

Every Fourier multiplier is for the declared central-difference Laplacian.
Products are nodal products (cyclic convolution), not a dealiased continuum
equation. Variational quadrature is independent of the coupled RK4 teachers.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch

from tdn.numerics import Equation, Geometry
from tdn.numerics.operators import check_shape, laplacian, rhs
from tdn.numerics.reference import ReferenceResult, choose_substeps, reference_step
from tdn.numerics.subflows import diffusion_step, reaction_step


def discrete_eigenvalues(geometry, state, equation=None):
    """Laplacian eigenvalues, or diffusion eigenvalues when equation is given."""
    result = torch.zeros(geometry.grid, dtype=state.dtype, device=state.device)
    for axis, (n, dx) in enumerate(zip(geometry.grid, geometry.dx)):
        k = torch.arange(n, dtype=state.dtype, device=state.device)
        shape = [1] * geometry.ndim
        shape[axis] = n
        result = result - (4 * torch.sin(torch.pi * k / n).square() / dx**2).reshape(shape)
    return result if equation is None else result * equation.kappa


def background_logistic_response(c, h, reaction_rate):
    """Return homogeneous flow, first derivative q and quadratic coefficient.

    The quadratic coefficient is one half of the second derivative with
    respect to c, not the second derivative itself. Stable nonpositive
    exponentials avoid overflow for positive reaction times.
    """
    if not 0 < float(c) < 1:
        raise ValueError("Quadratic response requires an interior background")
    if float(h) < 0 or float(reaction_rate) < 0:
        raise ValueError("Response times and reaction rates must be nonnegative")
    z = math.exp(-float(h) * float(reaction_rate))
    den = float(c) + (1 - float(c)) * z
    q = (z / den) / den
    return float(c) / den, q, -q * (1 - z) / den


@lru_cache(maxsize=12)
def _quadrature(n):
    if type(n) is not int or n < 1:
        raise ValueError("Quadrature rank must be a positive integer")
    x, w = np.polynomial.legendre.leggauss(n)
    return (x + 1) / 2, w / 2


def coupled_quadratic_response(v, c, h, equation, geometry, *, quadrature_nodes=64):
    """Coefficient of epsilon² in flow(c+epsilon*v), by variation of constants.

    b(h)=-r*q(h)*integral q(s)*D_(h-s)[(D_s v)^2] ds.
    This includes all aliased input pairs and their phases. A quadrature node
    is one physically separable input-input-output kernel term.
    """
    check_shape(v, geometry)
    if not math.isfinite(float(h)) or h < 0:
        raise ValueError("Response horizon must be finite and nonnegative")
    _, qh, _ = background_logistic_response(c, h, equation.reaction_rate)
    answer = torch.zeros_like(v)
    if h == 0 or equation.reaction_rate == 0:
        return answer
    nodes, weights = _quadrature(quadrature_nodes)
    for node, weight in zip(nodes, weights):
        s = float(node) * h
        _, qs, _ = background_logistic_response(c, s, equation.reaction_rate)
        first = diffusion_step(v, s, equation, geometry)
        answer = answer - equation.reaction_rate * qh * h * float(weight) * qs * diffusion_step(
            first.square(), h - s, equation, geometry)
    return answer


def split_quadratic_response(v, c, h, equation, geometry, *, orientation="reaction_first"):
    """Analytic epsilon² coefficients of the two exact Strang orientations."""
    check_shape(v, geometry)
    if orientation == "diffusion_first":
        _, _, coefficient = background_logistic_response(c, h, equation.reaction_rate)
        return coefficient * diffusion_step(
            diffusion_step(v, h / 2, equation, geometry).square(), h / 2, equation, geometry)
    if orientation != "reaction_first":
        raise ValueError("Unknown splitting orientation")
    c1, q1, b1 = background_logistic_response(c, h / 2, equation.reaction_rate)
    _, q2, b2 = background_logistic_response(c1, h / 2, equation.reaction_rate)
    linear = q1 * diffusion_step(v, h, equation, geometry)
    return q2 * b1 * diffusion_step(v.square(), h, equation, geometry) + b2 * linear.square()


def diffusion_first_step(u, h, equation, geometry):
    """D(h/2) R(h) D(h/2), preserving the declared nodal target."""
    return diffusion_step(reaction_step(diffusion_step(u, h / 2, equation, geometry), h,
                                       equation), h / 2, equation, geometry)


def moment_derivatives(u, equation, geometry):
    """Exact discrete mean/variance source terms, with one scalar per parent."""
    check_shape(u, geometry)
    axes = tuple(range(2, u.ndim))
    mean = u.mean(axes, keepdim=True)
    v = u - mean
    variance = v.square().mean(axes, keepdim=True)
    third = v.pow(3).mean(axes, keepdim=True)
    energy = (v * (equation.kappa * laplacian(v, geometry))).mean(axes, keepdim=True)
    return dict(mean=mean, variance=variance, third_centered_moment=third,
                diffusion_energy=energy,
                mean_derivative=equation.reaction_rate * (mean - mean.square() - variance),
                variance_derivative=2 * energy + 2 * equation.reaction_rate * (1 - 2 * mean) * variance
                - 2 * equation.reaction_rate * third)


def symmetric_quadratic_reference(v, c, h, equation, geometry, *, epsilon=.04,
                                  substeps=None, check=None):
    """Symmetric FP64 coupled teachers with amplitude/refinement uncertainty.

    q_e=(F(c+e*v)+F(c-e*v)-2F(c))/(2e²), at e,e/2,e/4. Richardson
    extrapolation in amplitude removes the leading e² remainder. Uncertainty retains the
    full amplitude difference and full final temporal difference *after*
    division by e², including a subtraction-roundoff allowance. It is an
    estimate, never a certificate.
    """
    check_shape(v, geometry)
    if v.shape[0] != 1 or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Reference requires one direction and positive epsilon")
    v = v.to(dtype=torch.float64, device="cpu")
    if epsilon * float(v.abs().max()) >= min(c, 1 - c):
        raise ValueError("Symmetric states must remain strictly interior")
    count = max(16, choose_substeps(h, equation, geometry)) if substeps is None else substeps
    states = torch.cat(tuple(x for scale in (1., .5, .25) for x in
                             (c + epsilon * scale * v, c - epsilon * scale * v,
                              torch.full_like(v, c))))
    estimates, raw_estimates = [], []
    for n in (count, 2 * count, 4 * count):
        answer = reference_step(states, h, equation, geometry, n, check=check)
        coefficients = [(answer[3 * index:3 * index + 1] + answer[3 * index + 1:3 * index + 2]
                         - 2 * answer[3 * index + 2:3 * index + 3]) / (2 * (epsilon * scale)**2)
                        for index, scale in enumerate((1., .5, .25))]
        estimates.append((4 * coefficients[2] - coefficients[1]) / 3)
        raw_estimates.append(coefficients)
    rms = lambda x: float(x.double().square().mean().sqrt())
    temporal = estimates[-1] - estimates[-2]
    amplitude = raw_estimates[-1][2] - raw_estimates[-1][1]
    coarse_extrapolated = (4 * raw_estimates[-1][1] - raw_estimates[-1][0]) / 3
    extrapolation_change = estimates[-1] - coarse_extrapolated
    rounding = 128 * torch.finfo(torch.float64).eps / (epsilon / 4)**2
    difference = [rms(estimates[i + 1] - estimates[i]) for i in range(2)]
    resolved = difference[-1] <= max(difference[0] / 4, rounding)
    return dict(response=estimates[-1], uncertainty_rms=max(rms(temporal), rms(amplitude), rms(extrapolation_change), rounding),
                uncertainty_max=max(float(temporal.abs().max()), float(amplitude.abs().max()), float(extrapolation_change.abs().max()), rounding),
                temporal_uncertainty_rms=rms(temporal), amplitude_uncertainty_rms=rms(amplitude),
                amplitude_extrapolation_change_rms=rms(extrapolation_change),
                rounding_allowance=rounding, epsilon=epsilon, epsilon_levels=[epsilon, epsilon / 2, epsilon / 4],
                refinement_counts=[count, 2 * count, 4 * count], refinement_differences=difference,
                accepted=resolved and bool(torch.isfinite(estimates[-1]).all()))


def quadratic_pair_kernel(c, h, equation, geometry, state, *, quadrature_nodes=64,
                          orientation=None):
    """Small 1D mode-pair matrix with cyclic output index m=(k+l) mod N.

    For a defect, subtract the requested split's analytic two-term kernel.
    This is an offline response oracle, not a trained neural weight tensor.
    """
    if geometry.ndim != 1:
        raise ValueError("Offline dense pair matrix is deliberately bounded to one dimension")
    n = geometry.grid[0]
    lam = discrete_eigenvalues(geometry, state, equation)
    k = torch.arange(n, device=state.device)
    output = (k[:, None] + k[None, :]) % n
    ls = lam[:, None] + lam[None, :]
    lm = lam[output]
    _, qh, _ = background_logistic_response(c, h, equation.reaction_rate)
    matrix = torch.zeros((n, n), dtype=state.dtype, device=state.device)
    for node, weight in zip(*_quadrature(quadrature_nodes)):
        s = float(node) * h
        _, qs, _ = background_logistic_response(c, s, equation.reaction_rate)
        matrix = matrix - equation.reaction_rate * qh * h * float(weight) * qs * torch.exp(
            lm * (h - s) + ls * s)
    if orientation == "reaction_first":
        c1, q1, b1 = background_logistic_response(c, h / 2, equation.reaction_rate)
        _, q2, b2 = background_logistic_response(c1, h / 2, equation.reaction_rate)
        matrix = matrix - q2 * b1 * torch.exp(lm * h) - b2 * q1**2 * torch.exp(ls * h)
    elif orientation == "diffusion_first":
        _, _, b = background_logistic_response(c, h, equation.reaction_rate)
        matrix = matrix - b * torch.exp((ls + lm) * h / 2)
    elif orientation is not None:
        raise ValueError("Unknown splitting orientation")
    return matrix


def apply_pair_kernel(v, matrix):
    """Direct bounded cyclic convolution for independent phase/alias tests."""
    if v.ndim != 3 or v.shape[0:2] != (1, 1) or matrix.shape != (v.shape[-1], v.shape[-1]):
        raise ValueError("Pair application requires one 1D direction and a square mode matrix")
    n = v.shape[-1]
    modes = torch.fft.fft(v[0, 0]) / n
    output = torch.zeros(n, dtype=modes.dtype, device=v.device)
    for k in range(n):
        for l in range(n):
            output[(k + l) % n] += modes[k] * modes[l] * matrix[k, l]
    return torch.fft.ifft(output * n).real.reshape_as(v)


def fourier_resample(u, grid):
    """CPU real Fourier interpolation/restriction with explicit Nyquist split.

    scipy's resampling convention splits an even-grid Nyquist coefficient
    when interpolating and recombines it when restricting. This function is
    for independent FP64 teachers, never a learned differentiable branch.
    """
    from scipy.signal import resample
    if u.device.type != "cpu" or len(grid) != u.ndim - 2:
        raise ValueError("Fourier teacher resampling requires a matching CPU field")
    result = u.detach().double().numpy()
    for axis, size in enumerate(grid, start=2):
        result = resample(result, int(size), axis=axis)
    return torch.from_numpy(np.ascontiguousarray(result)).to(dtype=u.dtype)


def dealiased_square(u, geometry):
    """3/2-padded real product restricted to the input Fourier support.

    Uses padding strictly greater than 3N/2 to avoid the endpoint alias at
    an even input Nyquist mode. This defines a Galerkin spectral target;
    it deliberately differs from the project's nodal FD equation.
    """
    check_shape(u, geometry)
    padded_grid = tuple(3 * n // 2 + 1 for n in geometry.grid)
    lifted = fourier_resample(u, padded_grid)
    return fourier_resample(lifted.square(), geometry.grid)


def continuum_rhs(u, equation, geometry):
    """Fourier Laplacian plus explicitly dealiased logistic source (CPU)."""
    check_shape(u, geometry)
    axes = tuple(range(2, u.ndim))
    lam = torch.zeros(geometry.grid, dtype=u.dtype, device=u.device)
    for axis, (n, dx) in enumerate(zip(geometry.grid, geometry.dx)):
        frequency = 2 * torch.pi * torch.fft.fftfreq(n, d=dx, dtype=u.dtype, device=u.device)
        shape = [1] * geometry.ndim
        shape[axis] = n
        lam = lam - frequency.square().reshape(shape)
    linear = torch.fft.ifftn(torch.fft.fftn(u, dim=axes) * lam, dim=axes).real
    return equation.kappa * linear + equation.reaction_rate * (u - dealiased_square(u, geometry))


def continuum_reference_step(u, h, equation, geometry, substeps, *, check=None):
    if type(substeps) is not int or substeps < 1 or u.device.type != "cpu":
        raise ValueError("Continuum reference requires CPU and positive fixed count")
    answer = u.double()
    dt = float(h) / substeps
    for index in range(substeps):
        if check is not None and index % 16 == 0:
            check()
        k1 = continuum_rhs(answer, equation, geometry)
        k2 = continuum_rhs(answer + dt * k1 / 2, equation, geometry)
        k3 = continuum_rhs(answer + dt * k2 / 2, equation, geometry)
        k4 = continuum_rhs(answer + dt * k3, equation, geometry)
        answer = answer + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    if check is not None:
        check()
    return answer


def choose_continuum_substeps(h, equation, geometry):
    stiffness = equation.kappa * sum((math.pi * n / length)**2
                                    for n, length in zip(geometry.grid, geometry.lengths))
    return max(1, math.ceil(2 * float(h) * (stiffness + equation.reaction_rate)))


def refined_continuum_reference(u, h, equation, geometry, substeps=None, tolerance=None,
                               *, error_fraction=.05, noise_floor=1e-10, check=None):
    """Three-level coupled spectral RK4; uncertainty is temporal, not spatial.

    The caller must separately compare spatial resolutions and include their
    disagreement in the common fine-grid continuum target's uncertainty.
    Acceptance checks finite states and temporal refinement; spectral nodal
    interval excursions are visible, and no clipping is performed.
    """
    if not math.isfinite(float(h)) or h < 0 or not 0 < error_fraction < 1:
        raise ValueError("Invalid continuum teacher horizon or error fraction")
    if tolerance is not None and (not math.isfinite(tolerance) or tolerance <= 0):
        raise ValueError("Continuum teacher tolerance must be positive")
    base = choose_continuum_substeps(h, equation, geometry) if substeps is None else substeps
    counts = (base, 2 * base, 4 * base)
    with torch.no_grad():
        states = [continuum_reference_step(u, h, equation, geometry, n, check=check) for n in counts]
    rms = lambda x: float(x.double().square().mean().sqrt())
    differences = tuple(rms(states[i + 1] - states[i]) for i in range(2))
    rounding = 64 * torch.finfo(torch.float64).eps * max(1., rms(states[-1]))
    finite = all(bool(torch.isfinite(x).all()) for x in states)
    converged = finite and (max(differences) <= rounding or differences[-1] <= differences[0] / 4)
    uncertainty = max(differences[-1], rounding)
    from tdn.numerics.splitting import split_step
    defect = rms(states[-1] - split_step(u.double(), h, equation, geometry))
    accepted = converged and uncertainty <= error_fraction * max(defect, noise_floor)
    accepted = accepted and (tolerance is None or uncertainty <= error_fraction * tolerance)
    order = math.log2(differences[0] / differences[1]) if min(differences) > 0 else None
    reasons = []
    if not finite:
        reasons.append("nonfinite_refinement")
    if not converged:
        reasons.append("unresolved_temporal_refinement")
    if uncertainty > error_fraction * max(defect, noise_floor):
        reasons.append("uncertainty_exceeds_defect_budget")
    if tolerance is not None and uncertainty > error_fraction * tolerance:
        reasons.append("uncertainty_exceeds_tolerance_budget")
    return ReferenceResult(states[-1], uncertainty, defect, accepted, counts[-1], counts,
                           differences, order, converged, ",".join(reasons) if reasons else "accepted")
