"""Actual allocated-CUDA premix values, gradients, identities and optimizer checks.

A skipped suite supplies no GPU evidence. The CARC launcher additionally sets
TDN_REQUIRE_GPU_TESTS=1, making unavailable CUDA/a real task a collection error.
Small manufactured-target optimizer checks validate execution and CPU parity;
they are not PDE accuracy, convergence or architecture advantage results.
"""
from __future__ import annotations

from copy import deepcopy
import os

import pytest
import torch

from tdn.numerics.types import Equation, Geometry
from tdn.numerics.splitting import split_step
from tdn.research.premix import VARIANTS, prepare_premix_step
from tdn.research.premix_neural import FAMILIES, build_model, project_modes
from tdn.research.reaction import capacity_update


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(),
                       reason="No CUDA device; premix GPU coverage remains unvalidated"),
    pytest.mark.skipif(os.environ.get("TDN_EXECUTION_MODE") == "desktop",
                       reason="Premix CARC GPU checks require a real allocated CARC task"),
]


@pytest.fixture(autouse=True)
def allocated_ieee_gpu():
    from tdn.runtime.preflight import verify_runtime
    from tdn.runtime.precision import reference_precision
    verify_runtime("cuda", "premix-gpu-tests")
    reference_precision()
    torch.manual_seed(85019)
    torch.cuda.manual_seed_all(85019)


def _state(dtype, grid=(12, 16)):
    x = 2 * torch.pi * torch.arange(grid[0], dtype=dtype) / grid[0]
    y = 2 * torch.pi * torch.arange(grid[1], dtype=dtype) / grid[1]
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    pattern = .03 * torch.cos(3 * xx + .27) + .024 * torch.sin(4 * xx - .19) + .017 * torch.cos(yy)
    return torch.stack((.32 + pattern, .69 - .8 * pattern))[:, None]


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_allocated_premix_numerical_values_gradients_and_work(variant, dtype):
    u = _state(dtype).requires_grad_()
    equation, geometry = Equation(.012, 2.9), Geometry((12, 16), (1.2, .9))
    cpu = prepare_premix_step(u, .08, equation, geometry, variant, modes=2, chunk_size=1)
    gpu_u = u.detach().cuda().requires_grad_()
    gpu = prepare_premix_step(gpu_u, .08, equation, geometry, variant, modes=2, chunk_size=1)
    cpu_work, gpu_work = {}, {}
    expected, actual = cpu(u, cpu_work), gpu(gpu_u, gpu_work)
    expected_gradient, = torch.autograd.grad(expected.square().mean(), u)
    actual_gradient, = torch.autograd.grad(actual.square().mean(), gpu_u)
    tolerance = dict(rtol=3e-5, atol=2e-6) if dtype == torch.float32 else dict(rtol=3e-10, atol=2e-12)
    torch.testing.assert_close(actual.cpu(), expected, **tolerance)
    torch.testing.assert_close(actual_gradient.cpu(), expected_gradient, **tolerance)
    assert torch.isfinite(actual_gradient).all()
    assert cpu_work == gpu_work
    raw_tolerance = dict(rtol=4e-4, atol=2e-9) if dtype == torch.float32 else dict(rtol=2e-9, atol=4e-17)
    torch.testing.assert_close(gpu.defect(gpu_u).cpu(), cpu.defect(u), **raw_tolerance)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_allocated_premix_learned_full_map_cpu_gpu_gradients(family, dtype):
    model = build_model(family, width=4, modes=2, t_ref=.2).to(dtype=dtype)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.startswith("head.") or name.startswith("local_head."):
                parameter.add_(.002 * torch.randn_like(parameter))
    gpu_model = deepcopy(model).cuda()
    equation, geometry = Equation(.008, 2.4), Geometry((12, 16), (1.2, .9))
    u = _state(dtype).requires_grad_()
    h = torch.tensor([.06, .09], dtype=dtype, requires_grad=True)
    expected = model(u, h, equation, geometry)
    cpu_gradients = torch.autograd.grad(expected.square().mean(), (u, h, *model.parameters()))
    gpu_u, gpu_h = u.detach().cuda().requires_grad_(), h.detach().cuda().requires_grad_()
    actual = gpu_model(gpu_u, gpu_h, equation, geometry)
    gpu_gradients = torch.autograd.grad(actual.square().mean(), (gpu_u, gpu_h, *gpu_model.parameters()))
    out_tol = dict(rtol=3e-5, atol=2e-6) if dtype == torch.float32 else dict(rtol=3e-10, atol=2e-11)
    grad_tol = dict(rtol=3e-3, atol=3e-5) if dtype == torch.float32 else dict(rtol=3e-8, atol=3e-10)
    torch.testing.assert_close(actual.cpu(), expected, **out_tol)
    for cpu_gradient, gpu_gradient in zip(cpu_gradients, gpu_gradients):
        assert torch.isfinite(gpu_gradient).all()
        torch.testing.assert_close(gpu_gradient.cpu(), cpu_gradient, **grad_tol)


@pytest.mark.parametrize("family", FAMILIES)
def test_allocated_zero_heads_preserve_full_state_base_and_zero_time(family):
    u = _state(torch.float64).cuda()
    model = build_model(family, width=4, modes=2, t_ref=.2).double().cuda()
    equation, geometry = Equation(.009, 2.8), Geometry((12, 16), (1.2, .9))
    expected = split_step(u, .08, equation, geometry)
    torch.testing.assert_close(model(u, .08, equation, geometry), expected, rtol=0., atol=0.)
    torch.testing.assert_close(model(u, 0., equation, geometry), u, rtol=0., atol=2e-15)


def test_allocated_nonlinear_product_preserves_high_high_to_low_mode():
    x = torch.arange(32, dtype=torch.float64, device="cuda") * (2 * torch.pi / 32)
    u = (torch.cos(9 * x + .31) + torch.cos(10 * x - .17))[None, None, :, None].expand(1, 1, 32, 16)
    after = project_modes(u.square(), 2)
    before = project_modes(u, 2).square()
    expected = (1 + torch.cos(x - .48))[None, None, :, None].expand_as(after)
    torch.testing.assert_close(after, expected, rtol=3e-12, atol=4e-14)
    assert float(before.abs().max()) < 1e-25


@pytest.mark.parametrize("family", FAMILIES)
def test_allocated_tiny_optimizer_updates_match_cpu(family):
    dtype = torch.float64
    cpu = build_model(family, width=4, modes=2, t_ref=.2).double()
    gpu = deepcopy(cpu).cuda()
    equation, geometry = Equation(.006, 2.1), Geometry((12, 16), (1.2, .9))
    state = _state(dtype)
    with torch.no_grad():
        target = capacity_update(split_step(state, .15, equation, geometry), torch.full_like(state, .004))
    initial = {name: parameter.detach().clone() for name, parameter in cpu.named_parameters()}
    cpu_optimizer = torch.optim.SGD(cpu.parameters(), lr=.03)
    gpu_optimizer = torch.optim.SGD(gpu.parameters(), lr=.03)
    history = []
    for _ in range(4):
        losses = []
        for model, optimizer, device in ((cpu, cpu_optimizer, "cpu"), (gpu, gpu_optimizer, "cuda")):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(state.to(device), .15, equation, geometry)
            loss = (prediction - target.to(device)).square().mean()
            loss.backward()
            assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                       for parameter in model.parameters())
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        assert losses[0] == pytest.approx(losses[1], rel=1e-7, abs=1e-12)
        history.append(losses)
    assert history[-1][1] < history[0][1]
    assert any(not torch.equal(parameter, initial[name]) for name, parameter in cpu.named_parameters())
    for cpu_parameter, gpu_parameter in zip(cpu.parameters(), gpu.parameters()):
        torch.testing.assert_close(gpu_parameter.cpu(), cpu_parameter, rtol=3e-8, atol=3e-10)
