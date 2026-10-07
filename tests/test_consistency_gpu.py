"""Actual allocated-CUDA consistency maps, limits, gradients and optimizer.

Skipped tests supply no GPU validation. TDN_REQUIRE_GPU_TESTS=1 additionally
makes missing CUDA or Slurm allocation a collection error in conftest.py;
the fixture validates the actual allocation and task-visible hardware.
Manufactured-target updates establish execution parity, not PDE superiority.
"""
from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from tdn.numerics import Equation, Geometry
from tdn.numerics.splitting import split_step
from tdn.research.consistency_neural import (
    CONSTRAINED_FAMILIES, FAMILIES, build_model, calibrate_mean,
    commutator_gate, discrete_logistic_commutator, mean_shape_update,
)
from tdn.research.reaction import capacity_update


pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not torch.cuda.is_available(),
                                 reason="No CUDA device; consistency GPU coverage remains unvalidated")]


@pytest.fixture(autouse=True)
def allocated_ieee_cuda():
    from tdn.runtime.preflight import verify_runtime
    from tdn.runtime.precision import reference_precision
    verify_runtime("cuda", "consistency-gpu-tests")
    reference_precision()
    torch.manual_seed(571903)
    torch.cuda.manual_seed_all(571903)


def _state(dtype):
    x = 2 * torch.pi * torch.arange(8, dtype=dtype) / 8
    y = 2 * torch.pi * torch.arange(10, dtype=dtype) / 10
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    pattern = .09 * torch.cos(2 * xx + .2) + .03 * torch.sin(3 * yy - .3)
    return torch.stack((.23 + pattern, .72 - .8 * pattern))[:, None]


def _fixture():
    return Equation(.013, 3.2), Geometry((8, 10), (1.2, .8))


def _learned_cpu(family, dtype):
    model = build_model(family, width=3, modes=2, t_ref=.2).to(dtype=dtype)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(.2 * torch.randn_like(parameter))
        if family.endswith("_moment"):
            model.mean_head.bias.fill_(.6)
    return model


@pytest.mark.parametrize("family", CONSTRAINED_FAMILIES)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_allocated_consistency_complete_map_cpu_gpu_values_and_all_gradients(family, dtype):
    cpu = _learned_cpu(family, dtype)
    gpu = deepcopy(cpu).cuda()
    equation, geometry = _fixture()
    u = _state(dtype).requires_grad_()
    horizon = torch.tensor([.07, .13], dtype=dtype, requires_grad=True)
    expected = cpu(u, horizon, equation, geometry)
    cpu_gradients = torch.autograd.grad(expected.square().mean(), (u, horizon, *cpu.parameters()))
    gpu_u = u.detach().cuda().requires_grad_()
    gpu_h = horizon.detach().cuda().requires_grad_()
    actual = gpu(gpu_u, gpu_h, equation, geometry)
    gpu_gradients = torch.autograd.grad(actual.square().mean(), (gpu_u, gpu_h, *gpu.parameters()))
    out_tol = dict(rtol=3e-5, atol=2e-6) if dtype == torch.float32 else dict(rtol=3e-10, atol=2e-11)
    grad_tol = dict(rtol=4e-3, atol=3e-5) if dtype == torch.float32 else dict(rtol=4e-8, atol=3e-10)
    torch.testing.assert_close(actual.cpu(), expected, **out_tol)
    assert ((actual >= 0) & (actual <= 1)).all()
    for expected_gradient, actual_gradient in zip(cpu_gradients, gpu_gradients):
        assert torch.isfinite(actual_gradient).all()
        torch.testing.assert_close(actual_gradient.cpu(), expected_gradient, **grad_tol)
    if family.endswith("_moment"):
        target = gpu.correction_components(gpu_u, gpu_h, equation, geometry)["target_mean"]
        torch.testing.assert_close(actual.mean((-2, -1), keepdim=True), target,
                                   rtol=0., atol=5 * torch.finfo(dtype).eps)


@pytest.mark.parametrize("family", CONSTRAINED_FAMILIES)
@pytest.mark.parametrize("limit", ("constant", "zero_reaction", "zero_diffusion", "zero_time"))
def test_allocated_arbitrary_trained_weights_obey_exact_physical_limits(family, limit):
    model = _learned_cpu(family, torch.float64).cuda()
    u = _state(torch.float64).cuda()
    equation, geometry = _fixture()
    horizon = .11
    if limit == "constant":
        u = torch.cat((torch.full_like(u[:1], .17), torch.full_like(u[:1], .83)))
    elif limit == "zero_reaction":
        equation = Equation(equation.kappa, 0.)
    elif limit == "zero_diffusion":
        equation = Equation(0., equation.reaction_rate)
    else:
        horizon = 0.
    actual = model(u, horizon, equation, geometry)
    expected = split_step(u, horizon, equation, geometry)
    torch.testing.assert_close(actual, expected, rtol=0., atol=0.)


@pytest.mark.parametrize("family", FAMILIES)
def test_allocated_zero_heads_preserve_the_complete_strang_state(family):
    model = build_model(family, width=3, modes=2, t_ref=.2).double().cuda()
    u = _state(torch.float64).cuda()
    equation, geometry = _fixture()
    torch.testing.assert_close(model(u, .11, equation, geometry), split_step(u, .11, equation, geometry),
                               rtol=0., atol=0.)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_allocated_commutator_gate_cpu_parity_and_quadratic_scaling(dtype):
    u = _state(dtype)
    equation, geometry = _fixture()
    expected = discrete_logistic_commutator(u, equation, geometry)
    actual = discrete_logistic_commutator(u.cuda(), equation, geometry)
    tolerance = dict(rtol=3e-6, atol=2e-7) if dtype == torch.float32 else dict(rtol=3e-13, atol=2e-14)
    torch.testing.assert_close(actual.cpu(), expected, **tolerance)
    cpu_gate = commutator_gate(u, equation, geometry, t_ref=.2)
    gpu_gate = commutator_gate(u.cuda(), equation, geometry, t_ref=.2)
    torch.testing.assert_close(gpu_gate.cpu(), cpu_gate, **tolerance)
    perturbed = .5 + .1 * (u.cuda() - .5)
    doubled = .5 + .2 * (u.cuda() - .5)
    first = discrete_logistic_commutator(perturbed, equation, geometry)
    second = discrete_logistic_commutator(doubled, equation, geometry)
    torch.testing.assert_close(second, 4 * first,
                               rtol=5e-5 if dtype == torch.float32 else 4e-13,
                               atol=2e-9 if dtype == torch.float32 else 2e-15)


def test_allocated_mean_shape_and_redistribution_values_gradients_and_extreme_bounds():
    base = torch.tensor([.03, .21, .58, .93], dtype=torch.float64).reshape(1, 1, 2, 2).requires_grad_()
    spatial = torch.tensor([-.6, .25, -.1, .45], dtype=torch.float64).reshape_as(base).requires_grad_()
    scalar = torch.tensor([.17], dtype=torch.float64, requires_grad=True)
    expected = mean_shape_update(base, scalar, spatial)
    cpu_gradients = torch.autograd.grad(expected.square().sum(), (base, scalar, spatial))
    gpu_base, gpu_scalar, gpu_spatial = [value.detach().cuda().requires_grad_()
                                        for value in (base, scalar, spatial)]
    actual = mean_shape_update(gpu_base, gpu_scalar, gpu_spatial)
    gpu_gradients = torch.autograd.grad(actual.square().sum(), (gpu_base, gpu_scalar, gpu_spatial))
    torch.testing.assert_close(actual.cpu(), expected, rtol=3e-13, atol=2e-14)
    for cpu_gradient, gpu_gradient in zip(cpu_gradients, gpu_gradients):
        assert torch.isfinite(gpu_gradient).all()
        torch.testing.assert_close(gpu_gradient.cpu(), cpu_gradient, rtol=3e-12, atol=3e-13)
    for mean in (0., .001, .73, .999, 1.):
        calibrated = calibrate_mean(gpu_base, mean)
        assert ((calibrated >= 0) & (calibrated <= 1)).all()
        torch.testing.assert_close(calibrated.mean(), torch.tensor(mean, dtype=base.dtype, device="cuda"),
                                   rtol=0., atol=3e-16)


@pytest.mark.parametrize("family", CONSTRAINED_FAMILIES)
def test_allocated_optimizer_progress_and_parameter_updates_match_cpu(family):
    cpu = build_model(family, width=3, modes=2, t_ref=.2).double()
    gpu = deepcopy(cpu).cuda()
    state = _state(torch.float64)
    equation, geometry = _fixture()
    with torch.no_grad():
        target = capacity_update(split_step(state, .15, equation, geometry), .003 * torch.ones_like(state))
    initial = {name: parameter.detach().clone() for name, parameter in cpu.named_parameters()}
    cpu_optimizer = torch.optim.SGD(cpu.parameters(), lr=.07)
    gpu_optimizer = torch.optim.SGD(gpu.parameters(), lr=.07)
    history = []
    for _ in range(4):
        losses = []
        for model, optimizer, device in ((cpu, cpu_optimizer, "cpu"), (gpu, gpu_optimizer, "cuda")):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(state.to(device), .15, equation, geometry) - target.to(device)).square().mean()
            loss.backward()
            assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                       for parameter in model.parameters())
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        assert losses[0] == pytest.approx(losses[1], rel=2e-8, abs=1e-12)
        history.append(losses)
    assert history[-1][1] < history[0][1]
    assert any(not torch.equal(parameter, initial[name]) for name, parameter in cpu.named_parameters())
    for expected_parameter, actual_parameter in zip(cpu.parameters(), gpu.parameters()):
        torch.testing.assert_close(actual_parameter.cpu(), expected_parameter, rtol=4e-8, atol=3e-10)
