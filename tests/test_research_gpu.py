"""Actual allocated A100 parity checks for the experimental full-state maps."""
from copy import deepcopy
import os

import pytest
import torch

from tdn.numerics import Equation, Geometry
from tdn.research import NEURAL_BASELINE_FAMILIES, OPTIONAL_FAMILIES, RESEARCH_FAMILIES, build_research_model


pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not torch.cuda.is_available(),
                                 reason="No CUDA device; research GPU parity remains unvalidated"),
              pytest.mark.skipif(os.environ.get("TDN_EXECUTION_MODE") == "desktop",
                                 reason="Research A100 workflow requires an actual CARC allocation")]


@pytest.fixture(autouse=True)
def allocated_a100():
    from tdn.runtime.preflight import verify_runtime
    from tdn.runtime.precision import reference_precision
    verify_runtime("cuda", "research-gpu-tests")
    reference_precision()


@pytest.mark.parametrize("family", RESEARCH_FAMILIES + OPTIONAL_FAMILIES + NEURAL_BASELINE_FAMILIES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_research_allocated_gpu_forward_and_full_gradients(family, dtype):
    torch.manual_seed(8021)
    model = build_research_model(family).to(dtype=dtype)
    # Exercise learned corrections, not only the zero-head base flow.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".head." in name or name.startswith("head.") or ".amplitude." in name:
                parameter.add_(.002 * torch.randn_like(parameter))
    gpu_model = deepcopy(model).cuda()
    u = (.2 + .6 * torch.rand(2, 1, 8, 8, dtype=dtype)).requires_grad_()
    h = torch.tensor([.015, .035], dtype=dtype, requires_grad=True)
    equation, geometry = Equation(.01, 2.), Geometry((8, 8), (1., 1.))
    expected = model(u, h, equation, geometry)
    cpu_gradients = torch.autograd.grad(expected.square().mean(),
                                       (u, h, *model.parameters()))
    gpu_u = u.detach().cuda().requires_grad_()
    gpu_h = h.detach().cuda().requires_grad_()
    actual = gpu_model(gpu_u, gpu_h, equation, geometry)
    gpu_gradients = torch.autograd.grad(actual.square().mean(),
                                       (gpu_u, gpu_h, *gpu_model.parameters()))
    output_tolerance = dict(atol=2e-6, rtol=2e-5) if dtype == torch.float32 else dict(atol=2e-11, rtol=2e-10)
    gradient_tolerance = dict(atol=2e-5, rtol=2e-3) if dtype == torch.float32 else dict(atol=2e-10, rtol=2e-8)
    torch.testing.assert_close(actual.cpu(), expected, **output_tolerance)
    for first, second in zip(cpu_gradients, gpu_gradients):
        assert torch.isfinite(second).all()
        torch.testing.assert_close(second.cpu(), first, **gradient_tolerance)


def test_research_allocated_gpu_fp32_confluent_origin_and_branches():
    from tdn.research.confluent import confluent_basis
    z = torch.cat((torch.zeros(1), torch.logspace(-7, 4, 151)))
    expected = confluent_basis(z, torch.ones_like(z))
    actual = confluent_basis(z.cuda(), torch.ones_like(z, device="cuda"))
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-8, rtol=2e-5)
    assert torch.isfinite(actual).all()
