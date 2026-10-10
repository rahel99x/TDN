"""Mandatory actual-CUDA readiness at both requested resolutions."""
from copy import deepcopy
import os

import pytest
import torch

from tdn.analysis.adjacent.models import interaction_channels, make_model
from tdn.analysis.adjacent.resolution_checks import audit_field
from tdn.analysis.frontier.models import FNOBlock
from tdn.numerics import Equation, Geometry

pytestmark = pytest.mark.gpu
CASES = [(n, track) for n in (64, 128) for track in ("discrete", "continuum")]
IDS = [f"{n}-{track}" for n, track in CASES]


@pytest.fixture(autouse=True)
def native_cuda():
    if not torch.cuda.is_available():
        if os.environ.get("TDN_REQUIRE_ADJACENT_GPU_TESTS") == "1":
            pytest.fail("Required resolution suite has no actual CUDA device")
        pytest.skip("No CUDA; collection is not native verification")
    from tdn.runtime.preflight import verify_runtime
    from tdn.runtime.precision import reference_precision
    verify_runtime("cuda", "adjacent-resolution-gpu-tests")
    reference_precision()
    torch.manual_seed(941781)
    torch.cuda.manual_seed_all(941781)
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize("grid,track", CASES, ids=IDS)
def test_resolution_cuda_large_grid_contract(grid, track):
    u = audit_field(grid, device="cuda")
    eq, geo = Equation(.004, 3.), Geometry((grid, grid), (1., 1.))
    shared = interaction_channels(u, .06, eq, geo, track=track, split_modes=4,
                                  output_modes=8, nodes=2)
    reference = interaction_channels(u, .06, eq, geo, track=track, split_modes=4,
                                     output_modes=8, nodes=2, implementation="reference")
    torch.testing.assert_close(shared, reference, atol=2e-12, rtol=2e-8)
    model = make_model("channel_neural", track, dict(modes=8, split_modes=4)).cuda().double()
    with torch.no_grad():
        model.conditioner[-1].weight.fill_(.03)
        model.conditioner[-1].bias.copy_(u.new_tensor([.02, -.03, .04]))
    assert shared.shape == (1, 3, grid, grid)
    for null_eq in (Equation(0., 3.), Equation(.004, 0.)):
        delta = model.correction_components(u, .06, null_eq, geo)["increment"]
        torch.testing.assert_close(delta, torch.zeros_like(delta), atol=0, rtol=0)
    torch.testing.assert_close(model(u, 0., eq, geo), u, atol=0, rtol=0)
    loss = model(u, .06, eq, geo).square().mean()
    gradients = torch.autograd.grad(loss, tuple(model.parameters()))
    assert all(torch.isfinite(g).all() for g in gradients)
    assert any(bool(g.abs().max() > 0) for g in gradients)


@pytest.mark.parametrize("grid,track", CASES, ids=IDS)
def test_resolution_cuda_fno_local_path(grid, track):
    model = make_model("fno_standard", track, dict(modes=8, width=16, depth=3)).cuda().float()
    blocks = [module for module in model.modules() if isinstance(module, FNOBlock)]
    assert blocks, "The comparator must retain full-grid local FNO paths"
    # Test the actual block's high-frequency response beyond its retained
    # spectral modes. Removing the local path would make this gradient zero.
    block = blocks[0]
    x = torch.arange(grid, device="cuda", dtype=torch.float32) / grid
    value = torch.cos(2 * torch.pi * 13 * x)[None, None, :, None].expand(1, 16, grid, grid).clone().requires_grad_()
    output = block(value)
    assert output.shape == value.shape and torch.isfinite(output).all()
    grad = torch.autograd.grad(output.square().mean(), block.local.weight)[0]
    assert torch.isfinite(grad).all() and bool(grad.abs().max() > 0)


@pytest.mark.parametrize("grid,track", CASES, ids=IDS)
def test_resolution_cuda_precision(grid, track):
    eq, geo = Equation(.004, 3.), Geometry((grid, grid), (1., 1.))
    cpu = make_model("channel_neural", track, dict(modes=8, split_modes=4)).float()
    with torch.no_grad():
        cpu.conditioner[-1].bias.copy_(torch.tensor([.1, -.1, .05]))
    gpu = deepcopy(cpu).cuda()
    u = audit_field(grid, dtype=torch.float32)
    with torch.no_grad():
        expected = cpu(u, .06, eq, geo)
        actual = gpu(u.cuda(), .06, eq, geo).cpu()
    torch.testing.assert_close(actual, expected, atol=8e-7, rtol=5e-6)
    assert not torch.backends.cuda.matmul.allow_tf32
