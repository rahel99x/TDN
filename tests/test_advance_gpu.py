"""Mandatory actual-CUDA checks; collected/skipped cases are not GPU evidence."""
from copy import deepcopy
import os

import pytest
import torch

from tdn.analysis.advance.models import (FAMILIES, PHYSICAL_FAMILIES,
    TRAINABLE_FAMILIES, ZERO_HEAD_FAMILIES, make_model)
from tdn.numerics import Equation, Geometry

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module", autouse=True)
def bounded_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture(autouse=True)
def native_cuda():
    if not torch.cuda.is_available():
        if os.environ.get("TDN_REQUIRE_ADVANCE_GPU_TESTS") == "1":
            pytest.fail("Required advance CUDA checks have no actual CUDA device")
        pytest.skip("No actual CUDA; this skip is not GPU validation")
    from tdn.runtime.preflight import verify_runtime
    from tdn.runtime.precision import reference_precision
    verify_runtime("cuda", "advance-gpu-tests")
    reference_precision()
    torch.manual_seed(730029)
    torch.cuda.manual_seed_all(730029)


def state(device="cpu", dtype=torch.float64):
    x = torch.arange(8, dtype=dtype, device=device)/8
    return (.43+.12*torch.cos(2*torch.pi*x[:,None]+.2)
            +.06*torch.sin(4*torch.pi*x[None,:]-.4))[None,None]


def model(family, track):
    return make_model(family, track, dict(width=3,depth=2,modes=2,reaction_substeps=2,cubic_nodes=2)).double()


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("track", ("discrete", "continuum"))
def test_advance_cuda_limits(track, family):
    eq, geo = Equation(.003,2.), Geometry((8,8),(1.,1.))
    for dtype in (torch.float32,torch.float64):
        cpu, u = model(family,track).to(dtype),state(dtype=dtype)
        with torch.no_grad():
            for p in cpu.parameters():
                if p.requires_grad: p.copy_(torch.randn_like(p)*.1)
        gpu = deepcopy(cpu).cuda()
        with torch.no_grad():
            torch.testing.assert_close(gpu(u.cuda(),.03,eq,geo).cpu(),cpu(u,.03,eq,geo),
                atol=8e-7 if dtype==torch.float32 else 2e-11,rtol=5e-6 if dtype==torch.float32 else 2e-9)
            torch.testing.assert_close(gpu(u.cuda(),0.,eq,geo).cpu(),u,atol=0,rtol=0)
            if family in PHYSICAL_FAMILIES:
                for v, null_eq in [(u,Equation(0.,2.)),(u,Equation(.003,0.)),(u*0+.7,eq)]:
                    delta=gpu.correction_components(v.cuda(),.03,null_eq,geo)["increment"]
                    torch.testing.assert_close(delta,torch.zeros_like(delta),atol=0,rtol=0)


@pytest.mark.parametrize("family", TRAINABLE_FAMILIES)
@pytest.mark.parametrize("track", ("discrete", "continuum"))
def test_advance_cuda_gradients(track, family):
    eq, geo = Equation(.01,3.),Geometry((8,8),(1.,1.))
    cpu = model(family,track)
    if family in ZERO_HEAD_FAMILIES:
        first = deepcopy(cpu).cuda()
        u=state("cuda")
        baseline=first.correction_components(u,.08,eq,geo)["base"].detach()
        (first(u,.08,eq,geo)-(baseline+.01*u)).square().mean().backward()
        assert first.head[-1].weight.grad.norm()>0
        assert first.lift.weight.grad.count_nonzero()==0
        with torch.no_grad():
            heads=[first.head[-1]]+([first.mean_head[-1]] if family=="fno_mean" else [])
            for head in heads:
                for p in head.parameters():
                    assert p.grad is not None and p.grad.norm()>0
                    p.add_(-.01*p.grad/p.grad.norm())
        first.zero_grad(set_to_none=True)
        (first(u,.08,eq,geo)-(baseline+.01*u)).square().mean().backward()
        assert torch.isfinite(first.lift.weight.grad).all() and first.lift.weight.grad.norm()>0
        cpu.load_state_dict({key:value.cpu() for key,value in first.state_dict().items()})
    else:
        with torch.no_grad():
            for p in cpu.parameters():
                if p.requires_grad:p.copy_(torch.randn_like(p)*.1)
    gpu=deepcopy(cpu).cuda()
    u,v=state().requires_grad_(),state("cuda").requires_grad_()
    h=torch.tensor(.03,dtype=torch.float64,requires_grad=True)
    hg=h.detach().cuda().requires_grad_()
    cg=torch.autograd.grad(cpu(u,h,eq,geo).square().mean(),(u,h,*(p for p in cpu.parameters() if p.requires_grad)))
    gg=torch.autograd.grad(gpu(v,hg,eq,geo).square().mean(),(v,hg,*(p for p in gpu.parameters() if p.requires_grad)))
    for a,b in zip(cg,gg):
        assert torch.isfinite(b).all() and b.abs().max()>0
        torch.testing.assert_close(b.cpu(),a,atol=3e-9,rtol=5e-5)


def test_advance_cuda_visible_allocation():
    assert torch.cuda.device_count()>=1 and torch.cuda.get_device_properties(0).total_memory>0
    assert os.environ.get("TDN_EXECUTION_MODE")=="desktop-slurm"
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_STEP_ID")


def test_advance_cuda_metrics_and_memory():
    """Native transfers, Nyquist-aware energy gradients and isolated peak probes."""
    from tdn.analysis.advance.metrics import (endpoint_metrics, energy_value,
        measure_paired_metrics, physical_inner_product)
    from tdn.analysis.frontier.numerics import rhs
    eq, geo = Equation(.01, 3.), Geometry((8,8),(1.,1.))
    u = .43 + .02*torch.randn((1,1,8,8),dtype=torch.float64)
    target = .97*u + .015
    prediction = target + .0002*torch.randn_like(u)
    reference = dict(accepted=True,uncertainty_rms=1e-10,uncertainty_max_bound=2e-10)
    for track in ("discrete", "continuum"):
        cpu_metrics = endpoint_metrics(prediction,target,u,eq,geo,track,reference)
        gpu_metrics = endpoint_metrics(prediction.cuda(),target.cuda(),u.cuda(),eq,geo,track,reference)
        # Both metric paths intentionally detach to exactly the same CPU FP64
        # values. Transfer/evaluation is not hidden inside inference latency.
        assert cpu_metrics == gpu_metrics
        cpu, gpu = u.clone().requires_grad_(), u.cuda().requires_grad_()
        ec, eg = energy_value(cpu,eq,geo,track), energy_value(gpu,eq,geo,track)
        gc, gg = torch.autograd.grad(ec,cpu)[0], torch.autograd.grad(eg,gpu)[0]
        torch.testing.assert_close(eg.cpu(),ec,atol=2e-12,rtol=2e-10)
        torch.testing.assert_close(gg.cpu(),gc,atol=2e-11,rtol=2e-9)
        derivative = rhs(gpu,eq,geo,track)
        expected = -physical_inner_product(derivative,derivative,geo,track)
        torch.testing.assert_close((gg*derivative).sum(),expected,atol=2e-11,rtol=2e-9)
    device_state = u.cuda()
    calls = {"small":lambda:device_state.square()+device_state,
             "large":lambda:torch.ones((256,256),device="cuda",dtype=torch.float32).square()}
    _, timing = measure_paired_metrics(calls,device="cuda",repeats=3,warmup=1,seed=730029,memory_probe=True)
    assert timing["memory_probe_seconds"]>0
    for method in timing["methods"].values():
        memory = method["memory"]
        assert memory["status"] == "MEASURED_SEPARATE_PROBE" and memory["probe_seconds"]>0
        for kind in ("allocated", "reserved"):
            peak, baseline = memory[f"absolute_peak_{kind}_bytes"], memory[f"baseline_{kind}_bytes"]
            assert peak >= baseline >= 0
            assert memory[f"incremental_peak_{kind}_bytes"] == peak-baseline
        assert method["tail_statistics"]["p95_seconds"] is None
        assert method["tail_statistics"]["p99_seconds"] is None
