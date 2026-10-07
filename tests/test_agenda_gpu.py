"""Native allocated CUDA checks; skipped cloud tests establish no GPU evidence."""
from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from tdn.numerics import Equation, Geometry
from tdn.research.agenda_neural import FAMILIES, build_model, physical_base


pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not torch.cuda.is_available(),
                                 reason="No actual CUDA; agenda GPU behavior remains unvalidated")]


@pytest.fixture(autouse=True)
def allocated_ieee_cuda():
    from tdn.runtime.preflight import verify_runtime
    from tdn.runtime.precision import reference_precision
    verify_runtime("cuda", "agenda-gpu-tests")
    reference_precision()
    torch.manual_seed(571917)
    torch.cuda.manual_seed_all(571917)


def state(dtype):
    x = 2 * torch.pi * torch.arange(8, dtype=dtype) / 8
    y = 2 * torch.pi * torch.arange(10, dtype=dtype) / 10
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    v = .06 * torch.cos(2 * xx + .2) + .025 * torch.sin(3 * yy - .3)
    return torch.stack((.23 + v, .72 - .6 * v))[:, None]


def equation_geometry():
    return Equation(.013, 3.2), Geometry((8, 10), (1.2, .8))


def learned(family, dtype, orientation="reaction-first"):
    model = build_model(family, width=3, modes=2, t_ref=.2,
                        base_orientation=orientation).to(dtype=dtype)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(.15 * torch.randn_like(parameter))
    return model


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_allocated_agenda_cpu_cuda_complete_map_and_all_gradients(family, dtype):
    cpu = learned(family, dtype)
    gpu = deepcopy(cpu).cuda()
    equation, geometry = equation_geometry()
    u = state(dtype).requires_grad_()
    h = torch.tensor([.07, .12], dtype=dtype, requires_grad=True)
    expected = cpu(u, h, equation, geometry)
    cpu_gradients = torch.autograd.grad(expected.square().mean(), (u, h, *cpu.parameters()))
    gpu_u, gpu_h = u.detach().cuda().requires_grad_(), h.detach().cuda().requires_grad_()
    actual = gpu(gpu_u, gpu_h, equation, geometry)
    gpu_gradients = torch.autograd.grad(actual.square().mean(), (gpu_u, gpu_h, *gpu.parameters()))
    out_tol = dict(rtol=4e-5, atol=3e-6) if dtype == torch.float32 else dict(rtol=3e-9, atol=2e-11)
    grad_tol = dict(rtol=5e-3, atol=4e-5) if dtype == torch.float32 else dict(rtol=3e-7, atol=3e-10)
    torch.testing.assert_close(actual.cpu(), expected, **out_tol)
    assert ((actual >= 0) & (actual <= 1)).all()
    for cpu_gradient, gpu_gradient in zip(cpu_gradients, gpu_gradients):
        assert torch.isfinite(gpu_gradient).all()
        torch.testing.assert_close(gpu_gradient.cpu(), cpu_gradient, **grad_tol)
    if "endpoint" in family or "closure" in family:
        target = gpu.correction_components(gpu_u, gpu_h, equation, geometry)["target_mean"]
        torch.testing.assert_close(actual.mean((-2, -1), keepdim=True), target,
                                   atol=5 * torch.finfo(dtype).eps, rtol=0.)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("limit", ("constant", "reaction", "diffusion", "time"))
def test_allocated_nonzero_weights_preserve_agenda_correction_limits(family, limit):
    model = learned(family, torch.float64).cuda()
    u = state(torch.float64).cuda()
    equation, geometry = equation_geometry()
    h = .09
    if limit == "constant":
        u = torch.cat((torch.full_like(u[:1], .17), torch.full_like(u[:1], .83)))
    elif limit == "reaction":
        equation = Equation(equation.kappa, 0.)
    elif limit == "diffusion":
        equation = Equation(0., equation.reaction_rate)
    else:
        h = 0.
    terms = model.correction_components(u, h, equation, geometry)
    assert torch.count_nonzero(terms["spatial_increment"]) == 0
    if "mean_increment" in terms:
        assert torch.count_nonzero(terms["mean_increment"]) == 0
    torch.testing.assert_close(model(u, h, equation, geometry), physical_base(u, h, equation, geometry),
                               rtol=0., atol=0.)


@pytest.mark.parametrize("family", ("source", "source_time_closure", "local_endpoint", "pair_rank4_closure"))
def test_allocated_diffusion_first_limits_and_initialization(family):
    equation, geometry = equation_geometry()
    u = state(torch.float64).cuda()
    model = build_model(family, width=3, modes=2, t_ref=.2,
                        base_orientation="diffusion-first").double().cuda()
    torch.testing.assert_close(model(u, .09, equation, geometry),
                               physical_base(u, .09, equation, geometry, "diffusion-first"), rtol=0., atol=0.)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(.15 * torch.randn_like(parameter))
    constant = torch.full_like(u, .4)
    torch.testing.assert_close(model(constant, .09, equation, geometry),
                               physical_base(constant, .09, equation, geometry, "diffusion-first"), rtol=0., atol=0.)


@pytest.mark.parametrize("family", ("source", "source_time", "source_closure", "fno_source", "pair_rank4"))
def test_allocated_agenda_optimizer_moves_heads_and_matches_cpu(family):
    cpu = build_model(family, width=3, modes=2, t_ref=.2).double()
    gpu = deepcopy(cpu).cuda()
    u = state(torch.float64)
    equation, geometry = equation_geometry()
    target = physical_base(u, .15, equation, geometry) + .001 * torch.cos(u * 13)
    optimizers = (torch.optim.SGD(cpu.parameters(), lr=.1), torch.optim.SGD(gpu.parameters(), lr=.1))
    history = []
    for _ in range(4):
        losses = []
        for model, optimizer, device in ((cpu, optimizers[0], "cpu"), (gpu, optimizers[1], "cuda")):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(u.to(device), .15, equation, geometry) - target.to(device)).square().mean()
            loss.backward()
            assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                       for parameter in model.parameters())
            optimizer.step()
            losses.append(loss.detach().cpu().item())
        assert losses[0] == pytest.approx(losses[1], rel=2e-7, abs=1e-12)
        history.append(losses)
    assert history[-1][1] < history[0][1]
    for cpu_parameter, gpu_parameter in zip(cpu.parameters(), gpu.parameters()):
        torch.testing.assert_close(gpu_parameter.cpu(), cpu_parameter, rtol=4e-7, atol=5e-10)


@pytest.mark.parametrize("family", ("pair_rank2", "pair_rank4", "pair_rank8"))
def test_allocated_pair_branch_retains_quadratic_amplitude_and_real_output(family):
    model = learned(family, torch.float64).cuda()
    u = state(torch.float64).cuda()
    equation, geometry = equation_geometry()
    v = u - u.mean((-2, -1), keepdim=True)
    first = model.correction_components(.4 + .01 * v, .1, equation, geometry)["spatial_increment"]
    second = model.correction_components(.4 + .02 * v, .1, equation, geometry)["spatial_increment"]
    assert first.is_floating_point() and torch.isfinite(first).all()
    assert first.abs().max() > 1e-13
    torch.testing.assert_close(second, 4 * first, rtol=3e-8, atol=2e-14)
