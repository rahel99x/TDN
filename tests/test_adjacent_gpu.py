"""Actual-CUDA channel checks; collected or skipped cases are not validation."""
from copy import deepcopy
import os

import pytest
import torch

from tdn.analysis.adjacent.models import CHANNEL_FAMILIES, interaction_channels, make_model
from tdn.numerics import Equation, Geometry

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture(autouse=True)
def native_cuda():
    if not torch.cuda.is_available():
        if os.environ.get("TDN_REQUIRE_ADJACENT_GPU_TESTS") == "1":
            pytest.fail("Required adjacent CUDA suite has no actual CUDA device")
        pytest.skip("No CUDA; this is not GPU verification")
    from tdn.runtime.preflight import verify_runtime
    from tdn.runtime.precision import reference_precision
    verify_runtime("cuda", "adjacent-gpu-tests")
    reference_precision()
    torch.manual_seed(8900131)
    torch.cuda.manual_seed_all(8900131)


def state(device="cpu", dtype=torch.float64):
    x = torch.arange(12, dtype=dtype, device=device) / 12
    return (.43 + .07 * torch.cos(2 * torch.pi * (x[:, None] + x[None, :]) + .4)
            + .04 * torch.cos(6 * torch.pi * x[:, None] - .3)
            + .025 * torch.sin(8 * torch.pi * x[None, :] + .2))[None, None]


def model(family, track):
    value = make_model(family, track, dict(modes=3, split_modes=1, quad_nodes=2, reaction_substeps=2)).double()
    with torch.no_grad():
        for p in value.parameters():
            p.copy_(torch.randn_like(p) * .2)
        if hasattr(value, "gain_logits"):
            value.gain_logits.copy_(torch.tensor([.3, -.7, .2]))
        if hasattr(value, "affine_coefficients"):
            value.affine_coefficients.fill_(.1)
    return value


@pytest.mark.parametrize("family", CHANNEL_FAMILIES)
@pytest.mark.parametrize("track", ("discrete", "continuum"))
def test_adjacent_cuda_limits(track, family):
    eq, geo = Equation(.012, 2.3), Geometry((12, 12), (1., 1.))
    for dtype in (torch.float32, torch.float64):
        cpu, u = model(family, track).to(dtype), state(dtype=dtype)
        gpu = deepcopy(cpu).cuda()
        torch.testing.assert_close(gpu(u.cuda(), .05, eq, geo).cpu(), cpu(u, .05, eq, geo),
            atol=8e-7 if dtype == torch.float32 else 2e-11, rtol=5e-6 if dtype == torch.float32 else 2e-9)
        torch.testing.assert_close(gpu(u.cuda(), 0., eq, geo).cpu(), u, atol=0, rtol=0)
        for initial, null_eq, h in ((u, Equation(0., 2.3), .05), (u, Equation(.012, 0.), .05),
                                   (torch.full_like(u, .6), eq, .05), (u, eq, 0.)):
            delta = gpu.correction_components(initial.cuda(), h, null_eq, geo)["increment"]
            torch.testing.assert_close(delta, torch.zeros_like(delta), atol=0, rtol=0)


@pytest.mark.parametrize("family", ("channel_neural", "feature_capacity"))
@pytest.mark.parametrize("track", ("discrete", "continuum"))
def test_adjacent_cuda_gradients(track, family):
    cpu = model(family, track)
    gpu = deepcopy(cpu).cuda()
    u, v = state().requires_grad_(), state("cuda").requires_grad_()
    h = torch.tensor(.05, dtype=torch.float64, requires_grad=True)
    hg = h.detach().cuda().requires_grad_()
    eq, geo = Equation(.012, 2.3), Geometry((12, 12), (1., 1.))
    first = torch.autograd.grad(cpu(u, h, eq, geo).square().mean(), (u, h, *cpu.parameters()))
    second = torch.autograd.grad(gpu(v, hg, eq, geo).square().mean(), (v, hg, *gpu.parameters()))
    for a, b in zip(first, second):
        assert torch.isfinite(b).all()
        torch.testing.assert_close(b.cpu(), a, atol=3e-9, rtol=3e-5)


@pytest.mark.parametrize("nodes", (2, 4))
@pytest.mark.parametrize("track", ("discrete", "continuum"))
def test_adjacent_cuda_polarization(track, nodes):
    u, eq, geo = state("cuda"), Equation(.012, 2.3), Geometry((12, 12), (1., 1.))
    actual = interaction_channels(u, .05, eq, geo, track=track, split_modes=1, nodes=nodes)
    reference = interaction_channels(u, .05, eq, geo, track=track, split_modes=1, nodes=nodes, implementation="reference")
    cpu = interaction_channels(u.cpu(), .05, eq, geo, track=track, split_modes=1, nodes=nodes)
    torch.testing.assert_close(actual, reference, atol=2e-15, rtol=5e-8)
    torch.testing.assert_close(actual.cpu(), cpu, atol=2e-15, rtol=5e-8)


@pytest.mark.parametrize("family", ("channel_global", "channel_affine"))
def test_adjacent_cuda_fitted_control(family):
    cpu, u = model(family, "discrete"), state()
    gpu, eq, geo = deepcopy(cpu).cuda(), Equation(.012, 2.3), Geometry((12, 12), (1., 1.))
    samples = []
    for index in range(8):
        initial, h = u + index * .01, .03 + index * .004
        parts = cpu.correction_components(initial, h, eq, geo)
        target = parts["base"] + 1.05 * parts["increment"]
        samples.append((initial, h, eq, geo, target))
    first = cpu.fit_least_squares(samples, max_iterations=30)
    second = gpu.fit_least_squares([(a.cuda(), h, eq, geo, target.cuda()) for a, h, eq, geo, target in samples], max_iterations=30)
    assert first["rank"] == second["rank"]
    torch.testing.assert_close(gpu(u.cuda(), .05, eq, geo).cpu(), cpu(u, .05, eq, geo), atol=3e-10, rtol=2e-8)


def test_adjacent_cuda_visible_allocation():
    assert torch.cuda.device_count() >= 1 and torch.cuda.get_device_properties(0).total_memory > 0
    assert os.environ.get("TDN_EXECUTION_MODE") == "desktop-slurm"
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_STEP_ID")
