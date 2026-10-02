from __future__ import annotations

import math

import pytest
import torch
from scipy.linalg import expm

from tdn.numerics import (Equation, Geometry, choose_substeps, diffusion_step,
                         laplacian, reaction_step, reference_step,
                         refined_reference, rhs, split_step, validate_state,
                         weighted_norm)
from tdn.numerics.linear import (control_matrices, exact_matrix, leading_matrix,
                                 rollout_convergence, split_matrix)


def state(grid=(16,)):
    geometry = Geometry(grid, (1.0,) * len(grid))
    axes = [torch.arange(n, dtype=torch.float64) / n for n in grid]
    coords = torch.meshgrid(*axes, indexing="ij")
    u = 0.45 + sum(0.12 / len(grid) * torch.cos(2 * torch.pi * x) for x in coords)
    return u.reshape(1, 1, *grid), geometry


def test_discrete_eigenmode_and_diffusion_match_same_grid():
    u, geometry = state((14, 12))
    equation = Equation(0.02, 0.0)
    h = 0.17
    lap = laplacian(u, geometry)
    first_mode = (u - 0.45)[..., :, 0]  # both modes are checked through coupled RK4 below
    assert first_mode.isfinite().all()
    assert abs(float(lap.mean())) < 3e-14
    exact = diffusion_step(u, h, equation, geometry)
    coupled = reference_step(u, h, equation, geometry, 100)
    assert torch.allclose(exact, coupled, atol=1e-12, rtol=1e-11)
    assert float(exact.mean()) == pytest.approx(float(u.mean()), abs=1e-15)
    n = 16
    geometry = Geometry((n,), (1.0,))
    k = 3
    mode = torch.cos(2 * torch.pi * k * torch.arange(n, dtype=torch.float64) / n)[None, None]
    eigenvalue = -4 * math.sin(math.pi * k / n)**2 / geometry.dx[0]**2
    assert torch.allclose(laplacian(mode, geometry), eigenvalue * mode, atol=2e-12, rtol=1e-12)
    expected = torch.exp(torch.tensor(equation.kappa * h * eigenvalue)) * mode
    assert torch.allclose(diffusion_step(mode, h, equation, geometry), expected.double(),
                          atol=2e-8, rtol=2e-7)


def test_logistic_endpoints_underflow_and_endpoint_derivatives():
    u = torch.tensor([[[0.0, 0.2, 1.0]]], dtype=torch.float64, requires_grad=True)
    equation = Equation(0, 1)
    out = reaction_step(u, 1000.0, equation)
    assert torch.equal(out, torch.tensor([[[0.0, 1.0, 1.0]]], dtype=torch.float64))
    gradient, = torch.autograd.grad(out.sum(), u)
    assert torch.isinf(gradient[..., 0]).all()
    assert torch.equal(gradient[..., 1:], torch.zeros_like(gradient[..., 1:]))
    for h in (0.0, 0.2, 30.0, 690.0):
        endpoints = torch.tensor([[[0.0, 1.0]]], dtype=torch.float64, requires_grad=True)
        values = reaction_step(endpoints, h, equation)
        derivative, = torch.autograd.grad(values.sum(), endpoints)
        assert torch.allclose(values, endpoints, atol=0, rtol=0)
        assert float(derivative[0, 0, 0]) == pytest.approx(math.exp(h), rel=5e-15)
        assert float(derivative[0, 0, 1]) == pytest.approx(math.exp(-h), rel=5e-15)


def test_logistic_grad_and_gradgrad_and_time_jvp():
    u = torch.tensor([[[0.15, 0.4, 0.85]]], dtype=torch.float64, requires_grad=True)
    h = torch.tensor(0.21, dtype=torch.float64, requires_grad=True)
    f = lambda initial, step: reaction_step(initial, step, Equation(0, 2))
    assert torch.autograd.gradcheck(f, (u, h))
    assert torch.autograd.gradgradcheck(f, (u, h))
    out, dh = torch.func.jvp(lambda step: f(u, step), (h,), (torch.ones_like(h),))
    assert torch.allclose(dh, 2 * out * (1 - out), atol=1e-14, rtol=1e-13)


def test_coupled_reference_refines_at_fourth_order_and_accepts():
    u, geometry = state()
    equation = Equation(0.02, 2)
    h = 0.25
    n = choose_substeps(h, equation, geometry)
    levels = [reference_step(u, h, equation, geometry, n * j) for j in (1, 2, 4)]
    d1 = float(weighted_norm(levels[1] - levels[0]))
    d2 = float(weighted_norm(levels[2] - levels[1]))
    assert d1 / d2 > 14
    teacher = refined_reference(u.float(), h, equation, geometry, n, tolerance=1e-5)
    assert teacher.accepted
    assert teacher.state.dtype == torch.float64
    assert teacher.refinement_substeps == (n, 2 * n, 4 * n)
    assert teacher.uncertainty < 0.05 * teacher.defect_norm
    rejected = refined_reference(u, 1.0, Equation(0.1, 5), geometry, 1)
    assert not rejected.accepted


def test_split_local_and_rollout_orders_on_nonlinear_rd():
    u, geometry = state()
    equation = Equation(0.02, 2)
    horizons = (0.12, 0.06, 0.03, 0.015)
    errors = []
    for h in horizons:
        n = choose_substeps(h, equation, geometry)
        teacher = refined_reference(u, h, equation, geometry, n)
        assert teacher.accepted
        errors.append(float(weighted_norm(teacher.state - split_step(u, h, equation, geometry))))
    orders = [math.log2(a / b) for a, b in zip(errors, errors[1:])]
    assert min(orders[-2:]) > 2.8
    truth = reference_step(u, 0.5, equation, geometry, 600)
    rollout_errors = []
    for n in (4, 8, 16, 32):
        got = u
        for _ in range(n):
            got = split_step(got, 0.5 / n, equation, geometry)
        rollout_errors.append(float(weighted_norm(got - truth)))
    assert min(math.log2(a / b) for a, b in zip(rollout_errors, rollout_errors[1:])) > 1.9


def test_exact_tier_a_noncommuting_anchor_sign_and_order():
    A, B, u0 = control_matrices()
    assert torch.linalg.vector_norm(A @ B - B @ A) > 0.1
    # Independent SciPy Padé exponential checks the torch control backend.
    oracle_exact = torch.from_numpy(expm(0.2 * (A + B).numpy()))
    oracle_split = torch.from_numpy(expm(0.1 * A.numpy()) @ expm(0.2 * B.numpy()) @ expm(0.1 * A.numpy()))
    assert torch.allclose(exact_matrix(A, B, 0.2), oracle_exact, atol=2e-15, rtol=2e-15)
    assert torch.allclose(split_matrix(A, B, 0.2), oracle_split, atol=2e-15, rtol=2e-15)
    e3 = leading_matrix(A, B)
    h = 1e-3
    oracle = (exact_matrix(A, B, h) - split_matrix(A, B, h)) / h**3
    assert torch.allclose(oracle, e3, atol=1e-4, rtol=5e-3)
    split = rollout_convergence(A, B, u0)
    anchor = rollout_convergence(A, B, u0, anchored=True)
    assert min(split["observed_orders"][-2:]) > 1.95
    assert 2.8 < min(anchor["observed_orders"][-2:]) < 3.2
    assert anchor["errors"][-1] < split["errors"][-1] / 10


def test_commuting_linear_and_zero_rd_controls():
    A = torch.diag(torch.tensor([-0.2, -0.7], dtype=torch.float64))
    B = torch.diag(torch.tensor([-1.0, -0.4], dtype=torch.float64))
    zeros = torch.zeros_like(A)
    for first, second in ((A, B), (A, zeros), (zeros, B), (zeros, zeros)):
        assert torch.allclose(exact_matrix(first, second, 0.5), split_matrix(first, second, 0.5),
                              atol=4e-15, rtol=4e-15)
        assert torch.allclose(leading_matrix(first, second), zeros, atol=2e-16, rtol=0)
    u, geometry = state()
    for equation in (Equation(0, 0), Equation(0, 2), Equation(0.02, 0)):
        reference = reference_step(u, 0.15, equation, geometry, 160)
        assert torch.allclose(split_step(u, 0.15, equation, geometry), reference, atol=2e-12, rtol=1e-11)
    uniform = torch.full_like(u, 0.3)
    assert torch.allclose(split_step(uniform, 0.2, Equation(0.02, 2), geometry),
                          reaction_step(uniform, 0.2, Equation(0, 2)), atol=1e-15, rtol=1e-14)


@pytest.mark.parametrize("method", ["split", "reference"])
def test_state_jvp_vjp_dot_and_centered_fd_sweep(method):
    u, geometry = state((8,))
    equation = Equation(0.02, 2)
    function = ((lambda x: split_step(x, 0.15, equation, geometry)) if method == "split"
                else (lambda x: reference_step(x, 0.15, equation, geometry, 24)))
    generator = torch.Generator().manual_seed(9)
    v = torch.randn(u.shape, generator=generator, dtype=u.dtype)
    w = torch.randn(u.shape, generator=generator, dtype=u.dtype)
    out, jv = torch.func.jvp(function, (u,), (v,))
    _, pullback = torch.func.vjp(function, u)
    vjp, = pullback(w)
    assert torch.allclose((jv * w).sum(), (v * vjp).sum(), atol=2e-12, rtol=2e-11)
    fd_errors = []
    for eps in (1e-2, 1e-3, 1e-4):
        fd = (function(u + eps * v) - function(u - eps * v)) / (2 * eps)
        fd_errors.append(float(weighted_norm(fd - jv)))
    assert fd_errors[1] < fd_errors[0] / 80
    assert fd_errors[2] < fd_errors[1] / 50
    assert out.isfinite().all()


def test_per_parent_horizon_and_time_gradients():
    u, geometry = state((8,))
    batch = u.expand(2, -1, -1).clone()
    h = torch.tensor([0.0, 0.1], dtype=torch.float64, requires_grad=True)
    result = split_step(batch, h, Equation(0.02, 2), geometry)
    assert torch.allclose(result[0], batch[0], atol=2e-15, rtol=0)
    derivative, = torch.autograd.grad(result.sum(), h)
    assert torch.isfinite(derivative).all()
    assert float(derivative[0]) == pytest.approx(float(rhs(batch[:1], Equation(0.02, 2), geometry).sum()), rel=1e-13)
    assert not split_step(batch.requires_grad_(), h, Equation(0.02, 2), geometry,
                          differentiable=False).requires_grad


def test_split_state_time_gradgrad_preserves_all_subflows():
    u, geometry = state((4,))
    u.requires_grad_()
    h = torch.tensor(0.03, dtype=torch.float64, requires_grad=True)
    function = lambda initial, step: split_step(initial, step, Equation(0.02, 2), geometry)
    assert torch.autograd.gradcheck(function, (u, h), atol=2e-6, rtol=1e-4)
    assert torch.autograd.gradgradcheck(function, (u, h), atol=2e-6, rtol=1e-4)


def test_teacher_fail_closed_on_declared_tolerance_budget():
    u, geometry = state((8,))
    equation = Equation(0.02, 2)
    teacher = refined_reference(u, 0.2, equation, geometry, 32, tolerance=1e-15)
    assert not teacher.accepted
    assert "tolerance_budget" in teacher.reason
    # State values remain the measured fine reference, even when rejected.
    assert torch.allclose(teacher.state, reference_step(u, 0.2, equation, geometry, 128),
                          atol=0, rtol=0)


def test_admissibility_fails_without_clipping_and_norm_resolution_independent():
    u, geometry = state()
    before = u.clone()
    validate_state(u)
    assert torch.equal(u, before)
    for bad in (-1e-5, 1.00001, math.nan, math.inf):
        with pytest.raises(ValueError):
            validate_state(torch.tensor([bad], dtype=torch.float64))
    one = torch.ones(1, 1, 16, dtype=torch.float64)
    fine = torch.ones(1, 1, 32, dtype=torch.float64)
    assert float(weighted_norm(one, geometry)) == float(weighted_norm(fine, Geometry((32,), (1.0,)))) == 1.0
