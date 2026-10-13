"""Allocated CUDA numerical cases for the new full-field tangent mechanism."""
from copy import deepcopy
import os
import pytest
import torch

from tdn.analysis.response import numerics as n
from tdn.analysis.tangent_response import tangent as t
from test_tangent_response_numerics import case, TARGETS

pytestmark = pytest.mark.gpu


@pytest.fixture(scope='module', autouse=True)
def one_thread():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture(autouse=True)
def actual_allocation():
    if not torch.cuda.is_available():
        if os.environ.get('TDN_REQUIRE_TANGENT_RESPONSE_GPU_TESTS')=='1':
            pytest.fail('Required Tangent Response suite has no actual CUDA device')
        pytest.skip('No actual CUDA; not Tangent Response GPU verification')
    from tdn.runtime.preflight import verify_runtime
    from tdn.runtime.precision import reference_precision
    verify_runtime('cuda','tangent-response-gpu-tests');reference_precision()


@pytest.mark.parametrize('target', TARGETS)
@pytest.mark.parametrize('size', (5,6))
def test_tangent_cuda_operator_odd_even_parity(size,target):
    a,z,eq,geom=case(size)
    expected=t.tangent_action(a,z,eq,geom,target)
    actual=t.tangent_action(a.cuda(),z.cuda(),eq,geom,target)
    torch.testing.assert_close(actual.cpu(),expected,atol=3e-11,rtol=3e-10)
    shifted=t.tangent_action(torch.roll(a.cuda(),(1,-2),(-2,-1)),torch.roll(z.cuda(),(1,-2),(-2,-1)),eq,geom,target)
    torch.testing.assert_close(shifted,torch.roll(actual,(1,-2),(-2,-1)),atol=3e-11,rtol=3e-10)


@pytest.mark.parametrize('target', TARGETS)
def test_tangent_cuda_homogeneous_action_parity(target):
    a,z,eq,geom=case();cpu=t.AnchorPath(a,eq,geom,target);gpu=t.AnchorPath(a.cuda(),eq,geom,target)
    actual=t.response_action(gpu,eq,geom,target,z.cuda(),.01,.05,steps=8)
    expected=t.response_action(cpu,eq,geom,target,z,.01,.05,steps=8)
    torch.testing.assert_close(actual.cpu(),expected,atol=3e-11,rtol=3e-10)


@pytest.mark.parametrize('target', TARGETS)
def test_tangent_cuda_forced_full_error_parity(target):
    a,_,eq,geom=case();cpu=t.AnchorPath(a,eq,geom,target,reaction_substeps=4);gpu=t.AnchorPath(a.cuda(),eq,geom,target,reaction_substeps=4)
    for mode, full in (('mean',False),('tangent',False),('tangent',True)):
        expected=t.forced_response(cpu,eq,geom,target,.04,steps=8,mode=mode,full_error=full)
        actual=t.forced_response(gpu,eq,geom,target,.04,steps=8,mode=mode,full_error=full)
        torch.testing.assert_close(actual.cpu(),expected,atol=3e-11,rtol=3e-8)
        if not full:
            expected_source=t.source_quadrature(cpu,eq,geom,target,.04,nodes=4,steps=8,mode=mode)
            actual_source=t.source_quadrature(gpu,eq,geom,target,.04,nodes=4,steps=8,mode=mode)
            torch.testing.assert_close(actual_source.cpu(),expected_source,atol=3e-11,rtol=3e-8)


@pytest.mark.parametrize('target', TARGETS)
def test_tangent_cuda_physics_time_gradients(target):
    a,z,_,geom=case();a=a.cuda();z=z.cuda()
    k=torch.tensor(.013,dtype=a.dtype,device='cuda',requires_grad=True)
    r=torch.tensor(2.4,dtype=a.dtype,device='cuda',requires_grad=True)
    h=torch.tensor(.04,dtype=a.dtype,device='cuda',requires_grad=True)
    eq=n.Physics(k,r);path=t.AnchorPath(a,eq,geom,target,create_graph=True)
    actual=t.response_action(path,eq,geom,target,z,0.,h,steps=4)
    grads=torch.autograd.grad(actual.square().sum(),(k,r,h))
    assert all(torch.isfinite(g) and g.abs()>0 for g in grads)
    source=t.source_quadrature(path,eq,geom,target,h,nodes=3,steps=4)
    source_grads=torch.autograd.grad(source.square().sum(),(k,r,h))
    assert all(torch.isfinite(g) and g.abs()>0 for g in source_grads)


@pytest.mark.parametrize('target', TARGETS)
def test_tangent_cuda_physical_nyquist_norm(target):
    geom=n.Geometry((6,6),(1.,1.));x=torch.arange(6,device='cuda',dtype=torch.float64)
    z=torch.cos(torch.pi*x)[:,None].expand(6,6)[None,None]
    expected=1. if target=='fd_same_grid' else .5
    torch.testing.assert_close(n.inner(z,z,geom,target),torch.full((1,1,1,1),expected,dtype=z.dtype,device=z.device),atol=2e-13,rtol=2e-13)


@pytest.mark.parametrize('target', TARGETS)
def test_tangent_cuda_nulls_identity_and_mean_coupling(target):
    a,z,eq,geom=case();a=a.cuda();z=z.cuda();path=t.AnchorPath(a,eq,geom,target)
    torch.testing.assert_close(t.response_action(path,eq,geom,target,z,.04,.04,steps=2),z,atol=0,rtol=0)
    heat_eq=n.Physics(eq.kappa,0.);linear=t.AnchorPath(a,heat_eq,geom,target)
    torch.testing.assert_close(t.response_action(linear,heat_eq,geom,target,z,0.,.04,steps=4),n.heat(z,.04,heat_eq,geom,target),atol=3e-12,rtol=3e-12)
    assert n.mean(t.tangent_action(a,z,eq,geom,target)).abs().max()>.1
    assert n.mean(t.mean_action(a,z,eq,geom,target)).abs().max()<1e-12


@pytest.mark.parametrize('target', TARGETS)
def test_tangent_cuda_same_path_composition(target):
    a,z,eq,geom=case();a=a.cuda();z=z.cuda();path=t.AnchorPath(a,eq,geom,target)
    first=t.response_action(path,eq,geom,target,z,.01,.03,steps=4)
    composed=t.response_action(path,eq,geom,target,first,.03,.05,steps=4)
    direct=t.response_action(path,eq,geom,target,z,.01,.05,steps=8)
    torch.testing.assert_close(composed,direct,atol=3e-12,rtol=3e-12)


def test_tangent_cuda_actual_allocated_device():
    assert os.environ.get('TDN_EXECUTION_MODE')=='desktop-slurm'
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_STEP_ID')
    assert torch.ones(1,device='cuda').device.type=='cuda'
