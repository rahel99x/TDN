"""Independent response, moment and discretization checks for the new agenda."""
import math

import pytest
import torch

from tdn.analysis.agenda.physics import (
    apply_pair_kernel, background_logistic_response, continuum_rhs,
    coupled_quadratic_response, dealiased_square, diffusion_first_step,
    discrete_eigenvalues, fourier_resample, moment_derivatives,
    quadratic_pair_kernel, refined_continuum_reference,
    split_quadratic_response, symmetric_quadratic_reference,
    continuum_reference_step,
)
from tdn.numerics import Equation, Geometry
from tdn.numerics.operators import laplacian, rhs
from tdn.numerics.splitting import split_step
from tdn.numerics.subflows import reaction_step


def direction(n=16):
    x = torch.arange(n, dtype=torch.float64) / n
    return (.7 * torch.cos(2 * math.pi * 2 * x + .3)
            + .3 * torch.sin(2 * math.pi * 5 * x - .2)).reshape(1, 1, n)


@pytest.mark.parametrize("c,h,r", [(.2, .1, 2.), (.7, 10., 100.), (.4, 0., 3.), (.3, .4, 0.)])
def test_background_response_derivatives_independently(c, h, r):
    value, q, b = background_logistic_response(c, h, r)
    x = torch.tensor(c, dtype=torch.float64, requires_grad=True)
    y = reaction_step(x.reshape(1, 1, 1), h, Equation(0., r)).sum()
    first = torch.autograd.grad(y, x, create_graph=True)[0]
    second = torch.autograd.grad(first, x)[0] if first.requires_grad else torch.zeros_like(x)
    assert value == pytest.approx(float(y.detach()), abs=2e-15)
    assert q == pytest.approx(float(first.detach()), abs=2e-15)
    assert b == pytest.approx(float(second.detach()) / 2, abs=2e-15)


@pytest.mark.parametrize("grid", [(16,), (9, 10)])
def test_discrete_eigenvalues_match_spatial_matrix_modes(grid):
    geometry = Geometry(grid, tuple(1. for _ in grid))
    axes = torch.meshgrid(*(torch.arange(n, dtype=torch.float64) / n for n in grid), indexing="ij")
    state = torch.cos(2 * math.pi * sum((i + 1) * a for i, a in enumerate(axes))).reshape(1, 1, *grid)
    index = tuple(i + 1 for i in range(len(grid)))
    eigenvalue = discrete_eigenvalues(geometry, state)[index]
    assert torch.allclose(laplacian(state, geometry), eigenvalue * state, atol=2e-12, rtol=1e-12)


def test_quadratic_response_agrees_with_independent_symmetric_coupled_teacher():
    geometry = Geometry((16,), (1.,))
    equation = Equation(.01, 3.)
    v = direction()
    analytic = coupled_quadratic_response(v, .4, .12, equation, geometry)
    independent = symmetric_quadratic_reference(v, .4, .12, equation, geometry,
                                               epsilon=.02)
    error = float((analytic - independent["response"]).square().mean().sqrt())
    assert independent["accepted"]
    assert error < independent["uncertainty_rms"]
    assert error < 2e-8
    assert independent["amplitude_uncertainty_rms"] > independent["rounding_allowance"]


@pytest.mark.parametrize("orientation", ["reaction_first", "diffusion_first"])
def test_analytic_split_quadratic_matches_symmetric_actual_split(orientation):
    geometry = Geometry((16,), (1.,))
    equation = Equation(.01, 3.)
    v, c, h, epsilon = direction(), .4, .12, .0001
    flow = split_step if orientation == "reaction_first" else diffusion_first_step
    numerical = (flow(c + epsilon * v, h, equation, geometry)
                 + flow(c - epsilon * v, h, equation, geometry)
                 - 2 * flow(torch.full_like(v, c), h, equation, geometry)) / (2 * epsilon**2)
    analytic = split_quadratic_response(v, c, h, equation, geometry, orientation=orientation)
    assert torch.allclose(analytic, numerical, atol=2e-8, rtol=2e-6)


def test_pair_kernel_retains_phases_symmetry_and_nodal_aliases():
    geometry = Geometry((16,), (1.,))
    equation = Equation(.003, 2.)
    v = direction()
    matrix = quadratic_pair_kernel(.3, .2, equation, geometry, v)
    direct = apply_pair_kernel(v, matrix)
    variation = coupled_quadratic_response(v, .3, .2, equation, geometry)
    assert torch.allclose(direct, variation, atol=3e-15, rtol=2e-13)
    assert torch.equal(matrix, matrix.T)
    shifted = torch.roll(v, 3, -1)
    assert torch.allclose(apply_pair_kernel(shifted, matrix), torch.roll(direct, 3, -1), atol=2e-15)


def test_pair_rank_is_physical_quadrature_not_plain_matrix_svd():
    geometry = Geometry((16,), (1.,))
    equation = Equation(.003, 2.)
    v = direction()
    truth = quadratic_pair_kernel(.4, .2, equation, geometry, v, quadrature_nodes=128)
    errors = []
    for rank in (2, 4, 8):
        approximation = quadratic_pair_kernel(.4, .2, equation, geometry, v, quadrature_nodes=rank)
        errors.append(float((truth - approximation).square().mean().sqrt()))
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 1e-10


def test_mean_variance_identity_exact_in_2d_with_nonzero_third_moment():
    geometry = Geometry((9, 10), (1., 1.))
    x, y = torch.meshgrid(torch.arange(9, dtype=torch.float64) / 9,
                          torch.arange(10, dtype=torch.float64) / 10, indexing="ij")
    state = (.4 + .07 * (torch.cos(2 * math.pi * x) + torch.cos(4 * math.pi * x)
                        + .3 * torch.sin(2 * math.pi * y))).reshape(1, 1, 9, 10)
    equation = Equation(.003, 2.)
    terms = moment_derivatives(state, equation, geometry)
    derivative = rhs(state, equation, geometry)
    assert abs(float(terms["third_centered_moment"])) > 1e-5
    assert float(terms["mean_derivative"]) == pytest.approx(float(derivative.mean()), abs=1e-15)
    assert float(terms["variance_derivative"]) == pytest.approx(
        float((2 * (state - state.mean()) * derivative).mean()), abs=1e-15)


def test_dealiased_product_distinguishes_nodal_equation_near_nyquist():
    n = 16
    geometry = Geometry((n,), (1.,))
    x = torch.arange(n, dtype=torch.float64) / n
    v = torch.cos(2 * math.pi * 7 * x).reshape(1, 1, n)
    nodal, galerkin = v.square(), dealiased_square(v, geometry)
    assert torch.allclose(galerkin, torch.full_like(v, .5), atol=3e-15)
    assert float((nodal - galerkin).square().mean().sqrt()) > .3


@pytest.mark.parametrize("grid", [(16,), (8, 10)])
def test_fourier_interpolation_handles_real_even_nyquist(grid):
    state = torch.randn((1, 1, *grid), generator=torch.Generator().manual_seed(23), dtype=torch.float64)
    enlarged = fourier_resample(state, tuple(2 * n for n in grid))
    restricted = fourier_resample(enlarged, grid)
    assert torch.allclose(restricted, state, atol=2e-15, rtol=2e-15)


def test_continuum_rhs_uniform_and_resolved_product_differ_from_fd_laplacian():
    geometry = Geometry((16,), (1.,))
    uniform = torch.full((1, 1, 16), .4, dtype=torch.float64)
    equation = Equation(.003, 2.)
    assert torch.allclose(continuum_rhs(uniform, equation, geometry), rhs(uniform, equation, geometry), atol=2e-15)
    x = torch.arange(16, dtype=torch.float64) / 16
    state = (.4 + .1 * torch.cos(2 * math.pi * x)).reshape(1, 1, 16)
    expected = -.003 * (2 * math.pi)**2 * (state - .4) + 2 * state * (1 - state)
    assert torch.allclose(continuum_rhs(state, equation, geometry), expected, atol=5e-15)
    assert float((continuum_rhs(state, equation, geometry) - rhs(state, equation, geometry)).abs().max()) > 1e-4


def test_refined_continuum_teacher_has_independent_temporal_uncertainty():
    geometry = Geometry((8,), (1.,))
    state = .4 + .1 * direction(8)
    teacher = refined_continuum_reference(state, .05, Equation(.003, 2.), geometry,
                                         substeps=16, tolerance=1e-6)
    assert teacher.accepted
    assert teacher.uncertainty < 1e-8
    assert teacher.refinement_substeps == (16, 32, 64)


def test_strong_diffusion_correct_orientation_limit_and_wrong_mean_bias():
    geometry = Geometry((16,), (1.,))
    state = .4 + .2 * direction()
    equation = Equation(100., 3.)
    h = .2
    correct = reaction_step(state.mean(-1, keepdim=True).expand_as(state), h, equation)
    wrong = reaction_step(reaction_step(state, h / 2, equation).mean(-1, keepdim=True).expand_as(state), h / 2, equation)
    assert torch.allclose(diffusion_first_step(state, h, equation, geometry), correct, atol=3e-15)
    assert torch.allclose(split_step(state, h, equation, geometry), wrong, atol=3e-15)
    assert float((correct - wrong).abs().max()) > .001


def test_reference_rejects_outside_interval_perturbations():
    with pytest.raises(ValueError, match="strictly interior"):
        symmetric_quadratic_reference(direction(), .01, .1, Equation(.003, 2.),
                                      Geometry((16,), (1.,)), epsilon=.1)


def test_independent_continuum_lawson_teacher_matches_explicit_coupled_spectral_rk4():
    from tdn.analysis.agenda.data import continuum_ifrk4
    geometry = Geometry((8, 8), (1., 1.))
    x, y = torch.meshgrid(torch.arange(8, dtype=torch.float64) / 8,
                          torch.arange(8, dtype=torch.float64) / 8, indexing="ij")
    state = (.4 + .05 * torch.cos(2 * math.pi * (x + 2 * y))
             + .04 * torch.sin(2 * math.pi * (2 * x - y))).reshape(1, 1, 8, 8)
    equation = Equation(.003, 2.)
    lawson = continuum_ifrk4(state, .12, equation, 32)
    independent = continuum_reference_step(state, .12, equation, geometry, 128)
    assert torch.allclose(lawson, independent, atol=2e-11, rtol=2e-11)
