"""Mandatory native CUDA checks; a skipped suite is never GPU validation."""
from copy import deepcopy
import os

import pytest
import torch

from tdn.analysis.advance.models import make_model
from tdn.analysis.resolve.kernels import BatchedQuadratureModel, batched_quadratic_defect
from tdn.analysis.roadmap.numerics import quadratic_df_defect
from tdn.numerics import Equation, Geometry

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module", autouse=True)
def threads():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def actual_cuda():
    if not torch.cuda.is_available():
        if os.environ.get("TDN_REQUIRE_RESOLVE_GPU_TESTS") == "1":
            pytest.fail("Required Resolve CUDA suite has no CUDA device")
        pytest.skip("No actual CUDA; not GPU validation")
    from tdn.runtime.preflight import verify_runtime
    from tdn.runtime.precision import reference_precision
    verify_runtime("cuda", "resolve-gpu-tests")
    reference_precision()
    torch.manual_seed(9700013); torch.cuda.manual_seed_all(9700013)


def field(device="cpu", dtype=torch.float64):
    x = torch.arange(8, dtype=dtype, device=device)/8
    return (.4+.08*torch.cos(2*torch.pi*x[:, None]+.3)
            +.04*torch.sin(4*torch.pi*x[None, :]+.2)
            +.017*torch.cos(2*torch.pi*(x[:, None]+x[None, :])))[None, None]


@pytest.mark.parametrize("track", ("discrete", "continuum"))
@pytest.mark.parametrize("precision", ("float32", "float64"))
def test_resolve_cuda_kernel_parity(precision, track):
    dtype = getattr(torch, precision)
    cpu = torch.cat((field(dtype=dtype), field(dtype=dtype)*.6+.1))
    cpu[..., ::2, ::2] += .003
    gpu = cpu.cuda(); h = torch.tensor([.06, .04], dtype=dtype)
    eq, geo = Equation(.008, 4.), Geometry((8, 8), (1.3, .8))
    for nodes in (2, 3, 4):
        expected = quadratic_df_defect(cpu, h, eq, geo, nodes=nodes, target=track)
        actual = batched_quadratic_defect(gpu, h.cuda(), eq, geo, nodes=nodes, track=track)
        native_sequential = quadratic_df_defect(gpu, h.cuda(), eq, geo, nodes=nodes, target=track)
        tolerance = 8e-9 if dtype == torch.float32 else 3e-15
        torch.testing.assert_close(actual.cpu(), expected, atol=tolerance, rtol=1e-3 if dtype == torch.float32 else 2e-8)
        torch.testing.assert_close(actual, native_sequential, atol=tolerance, rtol=1e-3 if dtype == torch.float32 else 2e-8)


@pytest.mark.parametrize("track", ("discrete", "continuum"))
def test_resolve_cuda_kernel_nulls(track):
    u, geo = field("cuda"), Geometry((8, 8), (1., 1.))
    for nodes in (2, 3, 4):
        for state, h, eq in ((u, 0., Equation(.008, 3.)), (u, .1, Equation(0., 3.)),
                             (u, .1, Equation(.008, 0.)), (u*0+.7, .1, Equation(.008, 3.))):
            increment = batched_quadratic_defect(state, h, eq, geo, track=track, nodes=nodes)
            assert torch.count_nonzero(increment) == 0
        model = BatchedQuadratureModel(track, nodes=nodes).double().cuda()
        torch.testing.assert_close(model(u, 0., Equation(.008, 3.), geo), u, atol=0., rtol=0.)


@pytest.mark.parametrize("track", ("discrete", "continuum"))
def test_resolve_cuda_kernel_gradients(track):
    eq, geo = Equation(.008, 4.), Geometry((8, 8), (1., 1.))
    for nodes in (2, 3, 4):
        cpu, gpu = field().requires_grad_(), field("cuda").requires_grad_()
        h = torch.tensor(.07, dtype=torch.float64, requires_grad=True)
        hg = h.detach().cuda().requires_grad_()
        c = batched_quadratic_defect(cpu, h, eq, geo, track=track, nodes=nodes)
        g = batched_quadratic_defect(gpu, hg, eq, geo, track=track, nodes=nodes)
        cg = torch.autograd.grad(c.square().sum(), (cpu, h))
        gg = torch.autograd.grad(g.square().sum(), (gpu, hg))
        for actual, expected in zip(gg, cg):
            assert torch.isfinite(actual).all() and actual.abs().max() > 0
            torch.testing.assert_close(actual.cpu(), expected, atol=1e-14, rtol=3e-7)


@pytest.mark.parametrize("track", ("discrete", "continuum"))
@pytest.mark.parametrize("family", ("two_basis", "fno_scaled", "fno_mean"))
def test_resolve_cuda_learning_parity(family, track):
    eq, geo = Equation(.008, 3.), Geometry((8, 8), (1., 1.))
    cpu = make_model(family, track, dict(width=4, depth=2, modes=2)).double()
    # Activate FNO trunk gradients beyond its deliberate zero-head initialization.
    with torch.no_grad():
        for p in cpu.parameters():
            p.add_(torch.randn_like(p)*.03)
    gpu = deepcopy(cpu).cuda()
    uc, ug = field().requires_grad_(), field("cuda").requires_grad_()
    ac, ag = cpu(uc, .06, eq, geo), gpu(ug, .06, eq, geo)
    torch.testing.assert_close(ag.cpu(), ac, atol=3e-11, rtol=3e-9)
    cg = torch.autograd.grad(ac.square().sum(), (uc, *cpu.parameters()))
    gg = torch.autograd.grad(ag.square().sum(), (ug, *gpu.parameters()))
    for actual, expected in zip(gg, cg):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual.cpu(), expected, atol=2e-10, rtol=3e-5)
    torch.testing.assert_close(gpu(ug, 0., eq, geo), ug, atol=0., rtol=0.)


@pytest.mark.parametrize("track", ("discrete", "continuum"))
@pytest.mark.parametrize("family", ("fno_scaled", "fno_mean"))
def test_resolve_cuda_zero_head_activation(family, track):
    eq, geo, u = Equation(.008, 3.), Geometry((8, 8), (1., 1.)), field("cuda")
    model = make_model(family, track, dict(width=4, depth=2, modes=2)).double().cuda()
    parts = model.correction_components(u, .06, eq, geo)
    assert torch.count_nonzero(parts["increment"]) == 0
    target = parts["base"].detach()+.01*u
    (model(u, .06, eq, geo)-target).square().mean().backward()
    assert model.head[-1].weight.grad.norm() > 0
    assert model.lift.weight.grad.count_nonzero() == 0
    if family == "fno_mean":
        assert model.mean_head[-1].weight.grad.norm() > 0
    with torch.no_grad():
        model.head[-1].weight.add_(-.01*model.head[-1].weight.grad/model.head[-1].weight.grad.norm())
        if family == "fno_mean":
            model.mean_head[-1].weight.add_(-.01*model.mean_head[-1].weight.grad/model.mean_head[-1].weight.grad.norm())
    model.zero_grad(set_to_none=True)
    (model(u, .06, eq, geo)-target).square().mean().backward()
    assert torch.isfinite(model.lift.weight.grad).all() and model.lift.weight.grad.norm() > 0


def test_resolve_cuda_allocation():
    assert os.environ.get("TDN_EXECUTION_MODE") == "desktop-slurm"
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_STEP_ID")
    assert torch.cuda.device_count() >= 1
    assert torch.cuda.get_device_properties(0).total_memory > 0


def test_resolve_cuda_timing_memory():
    from tdn.analysis.advance.metrics import measure_paired_metrics
    u, eq, geo = field("cuda", torch.float32), Equation(.008, 3.), Geometry((8, 8), (1., 1.))
    model = BatchedQuadratureModel("discrete", nodes=4).float().cuda()
    _, evidence = measure_paired_metrics({"batched":lambda:model(u, .06, eq, geo)}, device="cuda", repeats=2, warmup=1)
    measured = evidence["methods"]["batched"]
    assert len(measured["samples_seconds"]) == 2
    assert measured["memory"]["status"] == "MEASURED_SEPARATE_PROBE"
    assert measured["memory"]["absolute_peak_allocated_bytes"] >= measured["memory"]["baseline_allocated_bytes"]
    assert measured["tail_statistics"]["p95_seconds"] is None
    assert evidence["energy_joules"] is None
