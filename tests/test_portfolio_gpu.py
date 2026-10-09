"""Mandatory actual-CUDA portfolio checks; CPU skips are never GPU evidence."""
from copy import deepcopy
import os

import pytest
import torch

from tdn.analysis.portfolio.models import FAMILIES, PHYSICAL_FAMILIES, TRAINABLE_FAMILIES, make_model
from tdn.numerics import Equation, Geometry

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module", autouse=True)
def bounded_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture(autouse=True)
def native_cuda():
    if not torch.cuda.is_available():
        if os.environ.get("TDN_REQUIRE_PORTFOLIO_GPU_TESTS") == "1":
            pytest.fail("Required portfolio CUDA suite has no actual CUDA device")
        pytest.skip("No CUDA; this skip is not GPU validation")
    from tdn.runtime.preflight import verify_runtime
    from tdn.runtime.precision import reference_precision
    verify_runtime("cuda", "portfolio-gpu-tests")
    reference_precision()
    torch.manual_seed(5300173)
    torch.cuda.manual_seed_all(5300173)


def state(device="cpu", dtype=torch.float64):
    x = torch.arange(8, dtype=dtype, device=device) / 8
    return (.43 + .08 * torch.cos(2 * torch.pi * x[:, None] + .2)
            + .03 * torch.sin(4 * torch.pi * x[None, :] - .4))[None, None]


def model(family, track):
    value = make_model(family, track, dict(width=3, depth=2, modes=2, reaction_substeps=2, cubic_nodes=2)).double()
    with torch.no_grad():
        for p in value.parameters():
            if p.requires_grad: p.copy_(torch.randn_like(p) * .1)
    return value


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("track", ("discrete", "continuum"))
def test_portfolio_cuda_limits(track, family):
    eq, geo = Equation(.003, 2.), Geometry((8, 8), (1., 1.))
    for dtype in (torch.float32, torch.float64):
        cpu, u = model(family, track).to(dtype), state(dtype=dtype)
        gpu = deepcopy(cpu).cuda()
        with torch.no_grad():
            torch.testing.assert_close(gpu(u.cuda(), .03, eq, geo).cpu(), cpu(u, .03, eq, geo),
                atol=8e-7 if dtype == torch.float32 else 2e-11, rtol=5e-6 if dtype == torch.float32 else 2e-9)
            torch.testing.assert_close(gpu(u.cuda(), 0., eq, geo).cpu(), u, atol=0, rtol=0)
            if family in PHYSICAL_FAMILIES:
                for null_eq in (Equation(0., 2.), Equation(.003, 0.)):
                    delta = gpu.correction_components(u.cuda(), .03, null_eq, geo)["increment"]
                    torch.testing.assert_close(delta, torch.zeros_like(delta), atol=0, rtol=0)


@pytest.mark.parametrize("family", TRAINABLE_FAMILIES)
@pytest.mark.parametrize("track", ("discrete", "continuum"))
def test_portfolio_cuda_gradients(track, family):
    cpu = model(family, track)
    gpu = deepcopy(cpu).cuda()
    u, v = state().requires_grad_(), state("cuda").requires_grad_()
    h = torch.tensor(.03, dtype=torch.float64, requires_grad=True)
    hg = h.detach().cuda().requires_grad_()
    eq, geo = Equation(.003, 2.), Geometry((8, 8), (1., 1.))
    cg = torch.autograd.grad(cpu(u, h, eq, geo).square().mean(), (u, h, *(p for p in cpu.parameters() if p.requires_grad)))
    gg = torch.autograd.grad(gpu(v, hg, eq, geo).square().mean(), (v, hg, *(p for p in gpu.parameters() if p.requires_grad)))
    for a, b in zip(cg, gg):
        assert torch.isfinite(b).all()
        torch.testing.assert_close(b.cpu(), a, atol=3e-9, rtol=3e-5)


@pytest.mark.parametrize("family", ("quad2_amplitude", "quad2_linear"))
def test_portfolio_cuda_fitted_control(family):
    cpu, u = model(family, "discrete"), state()
    gpu, eq, geo = deepcopy(cpu).cuda(), Equation(.01, 2.), Geometry((8, 8), (1., 1.))
    samples = []
    for index in range(6):
        initial, h = u + index * .02, .02 + index * .01
        baseline, design = cpu.linear_design(initial, h, eq, geo)
        target = baseline + .1 * design.sum(1, keepdim=True)
        samples.append((initial, h, eq, geo, target))
    first = cpu.fit_least_squares(samples)
    second = gpu.fit_least_squares([(a.cuda(), h, eq, geo, target.cuda()) for a, h, eq, geo, target in samples])
    assert first["rank"] == second["rank"]
    torch.testing.assert_close(gpu(u.cuda(), .04, eq, geo).cpu(), cpu(u, .04, eq, geo), atol=2e-11, rtol=2e-9)


def test_portfolio_cuda_visible_allocation():
    assert torch.cuda.device_count() >= 1 and torch.cuda.get_device_properties(0).total_memory > 0
    assert os.environ.get("TDN_EXECUTION_MODE") == "desktop-slurm"
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_STEP_ID")
