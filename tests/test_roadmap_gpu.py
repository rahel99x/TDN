"""Mandatory native CUDA suite; a CPU skip is never GPU validation."""
from copy import deepcopy
import os

import pytest
import torch

from tdn.analysis.roadmap.protocol import MODEL_IDS
from tdn.analysis.roadmap.models import build_model, fractional_feature
from tdn.analysis.roadmap import numerics
from tdn.numerics import Equation, Geometry

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def native_cuda():
    if not torch.cuda.is_available():
        if os.environ.get("TDN_REQUIRE_ROADMAP_GPU_TESTS") == "1":
            pytest.fail("Required roadmap CUDA tests have no actual CUDA device")
        pytest.skip("No real CUDA; CPU checks do not establish native GPU evidence")
    from tdn.runtime.preflight import verify_runtime
    from tdn.runtime.precision import reference_precision
    verify_runtime("cuda", "roadmap-gpu-tests")
    reference_precision()
    torch.manual_seed(312051)
    torch.cuda.manual_seed_all(312051)


def state(device="cpu", n=8):
    x = torch.arange(n, dtype=torch.float64, device=device) / n
    value = .45 + .07 * torch.cos(2 * torch.pi * x[:, None] + .2) + .03 * torch.sin(4 * torch.pi * x[None, :] - .1)
    return value[None, None]


def model(family):
    value = build_model(family, {"width": 4, "modes": 1, "t_ref": .2}).double()
    with torch.no_grad():
        for parameter in value.parameters():
            parameter.copy_(.04 * torch.randn_like(parameter))
    return value


@pytest.mark.parametrize("family", MODEL_IDS)
def test_roadmap_cuda_model_limits(family):
    cpu = model(family); gpu = deepcopy(cpu).cuda()
    geometry, equation = Geometry((8, 8), (1., 1.)), Equation(.003, 2.)
    u = state()
    with torch.no_grad():
        expected = cpu(u, .03, equation, geometry)
        actual = gpu(u.cuda(), .03, equation, geometry)
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-7, atol=2e-10)
        torch.testing.assert_close(gpu(u.cuda(), 0., equation, geometry).cpu(), u, rtol=0., atol=2e-14)
        assert torch.isfinite(actual).all()
        if family not in ("direct_fno", "source_embedded"):
            for eq in (Equation(0., 2.), Equation(.003, 0.)):
                terms = gpu.correction_components(u.cuda(), .03, eq, geometry)
                # Every physically anchored learned residual must vanish,
                # even after arbitrary nonzero weights were assigned.
                correction = terms.get("increment", terms.get("spatial_increment"))
                if correction is not None:
                    torch.testing.assert_close(correction, torch.zeros_like(correction), atol=1e-14, rtol=0.)
                else:
                    from tdn.analysis.agenda.physics import diffusion_first_step
                    torch.testing.assert_close(gpu(u.cuda(), .03, eq, geometry),
                                               diffusion_first_step(u.cuda(), .03, eq, geometry), atol=2e-13, rtol=0.)


@pytest.mark.parametrize("family", MODEL_IDS)
def test_roadmap_cuda_model_gradient(family):
    cpu = model(family); gpu = deepcopy(cpu).cuda()
    geometry, equation = Geometry((8, 8), (1., 1.)), Equation(.003, 2.)
    u, v = state().requires_grad_(), state("cuda").requires_grad_()
    h, hg = torch.tensor(.03, dtype=torch.float64, requires_grad=True), torch.tensor(.03, dtype=torch.float64, device="cuda", requires_grad=True)
    out = cpu(u, h, equation, geometry); actual = gpu(v, hg, equation, geometry)
    grads = torch.autograd.grad(out.square().mean(), (u, h, *cpu.parameters()), allow_unused=True)
    cuda_grads = torch.autograd.grad(actual.square().mean(), (v, hg, *gpu.parameters()), allow_unused=True)
    for first, second in zip(grads, cuda_grads):
        assert (first is None) == (second is None)
        if first is not None:
            assert torch.isfinite(second).all()
            torch.testing.assert_close(second.cpu(), first, rtol=2e-5, atol=2e-9)
    assert any(v is not None and float(v.abs().max()) > 0 for v in cuda_grads[2:])


@pytest.mark.parametrize("kind", ("quadratic", "cubic", "dealias", "moments"))
def test_roadmap_cuda_numerical_parity(kind):
    geometry, equation = Geometry((8, 8), (1., 1.)), Equation(.003, 2.)
    def apply(u):
        if kind == "quadratic": return numerics.quadratic_df_defect(u, .03, equation, geometry, nodes=4)
        if kind == "cubic": return numerics.cubic_df_defect(u, .03, equation, geometry, nodes=4)
        if kind == "dealias": return numerics.dealiased_product(u, u)
        return numerics.dynamic_moment_step(u, .03, equation, geometry)["mean"]
    expected, actual = apply(state()), apply(state("cuda"))
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-7, atol=2e-11)
    assert torch.isfinite(actual).all()


def test_roadmap_cuda_fractional_physical_scale():
    small, large = state("cuda", 8), state("cuda", 16)
    a = fractional_feature(small, Geometry((8, 8), (1., 1.)), 1.25)
    b = fractional_feature(large, Geometry((16, 16), (1., 1.)), 1.25)
    torch.testing.assert_close(b[..., ::2, ::2], a, rtol=2e-10, atol=2e-11)


def test_roadmap_cuda_allocation_visible():
    assert torch.cuda.device_count() >= 1
    props = torch.cuda.get_device_properties(0)
    assert props.total_memory > 0
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_STEP_ID")
    assert os.environ.get("TDN_EXECUTION_MODE") == "desktop-slurm"
