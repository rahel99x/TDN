"""Mandatory actual-device Response validation, never simulated by CPU imports."""
from copy import deepcopy
import io
import os
import pytest
import torch
from tdn.numerics.types import Geometry
from tdn.analysis.response.contracts import Physics
from tdn.analysis.response.models import create_model
from tdn.analysis.response import numerics as n

pytestmark = pytest.mark.gpu
TARGETS = ('fd_same_grid', 'galerkin_same_grid', 'galerkin_projected_fine')
LEARNED = ('residual_fno', 'direct_fno', 'temporal', 'coupled_source')


@pytest.fixture(scope='module', autouse=True)
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture(autouse=True)
def actual_cuda():
    if not torch.cuda.is_available():
        if os.environ.get('TDN_REQUIRE_RESPONSE_GPU_TESTS') == '1':
            pytest.fail('Required Response suite has no actual CUDA device')
        pytest.skip('No actual CUDA; not GPU validation')
    from tdn.runtime.preflight import verify_runtime
    from tdn.runtime.precision import reference_precision
    verify_runtime('cuda', 'response-gpu-tests')
    reference_precision()
    torch.manual_seed(1190031); torch.cuda.manual_seed_all(1190031)


def field(size=7, dtype=torch.float64):
    x = torch.arange(size, dtype=dtype)/size
    return (.31+.09*torch.cos(2*torch.pi*x[:,None])+.03*torch.sin(4*torch.pi*x[None,:]+.3))[None,None]


def make(family, *, dtype=torch.float64):
    return create_model(dict(family=family,width=4,depth=1,modes=2,rank=2,
        quadrature_nodes=2,mean_nodes=2,reaction_substeps=2)).to(dtype=dtype)


def activate(model):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if 'head' in name or 'amplitude' in name:
                p.add_(.04*torch.randn_like(p))


@pytest.mark.parametrize('target', TARGETS)
@pytest.mark.parametrize('family', LEARNED)
@pytest.mark.parametrize('size', (6, 7))
def test_response_cuda_real_odd_even_parity(size, family, target):
    u = field(size); geom = Geometry((size,size),(1.,1.)); eq=Physics(.013,2.4)
    cpu=make(family); activate(cpu); gpu=deepcopy(cpu).cuda()
    expected=cpu(u,.03,eq,geom,target)
    actual=gpu(u.cuda(),.03,eq,geom,target)
    torch.testing.assert_close(actual.cpu(),expected,atol=2e-11,rtol=2e-9)
    shifted=gpu(torch.roll(u.cuda(),(1,-2),(-2,-1)),.03,eq,geom,target)
    torch.testing.assert_close(shifted,torch.roll(actual,(1,-2),(-2,-1)),atol=2e-11,rtol=2e-9)


@pytest.mark.parametrize('target', TARGETS)
@pytest.mark.parametrize('family', LEARNED)
def test_response_cuda_head_and_trunk_gradients(family, target):
    u=field().cuda();geom=Geometry((7,7),(1.,1.));eq=Physics(.013,2.4)
    model=make(family).cuda()
    reference=n.anchor(u,.05,eq,geom,target,reaction_substeps=2)+.01+.01*torch.randn_like(u)
    optimizer=torch.optim.SGD(model.parameters(),lr=10.)
    (model(u,.05,eq,geom,target)-reference).square().mean().backward()
    heads=[p for name,p in model.named_parameters() if 'head' in name or 'amplitude' in name]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in heads)
    assert sum(p.grad.abs().sum() for p in heads)>0
    if family=='coupled_source':
        assert model.mean_head.weight.grad.norm()>0 and model.source_head.weight.grad.norm()>0
    optimizer.step();optimizer.zero_grad()
    (model(u,.05,eq,geom,target)-reference).square().mean().backward()
    trunk=[p for name,p in model.named_parameters() if 'encoder' in name or 'body' in name]
    assert sum(p.grad.abs().sum() for p in trunk if p.grad is not None)>0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize('target', TARGETS)
def test_response_cuda_tensor_physics_and_query_parity(target):
    u=field().repeat(2,1,1,1).cuda().requires_grad_()
    k=torch.tensor([.01,.02],device='cuda',dtype=u.dtype,requires_grad=True)
    r=torch.tensor([2.,3.],device='cuda',dtype=u.dtype,requires_grad=True)
    h=torch.tensor([.03,.04],device='cuda',dtype=u.dtype,requires_grad=True)
    eq=Physics(k,r);geom=Geometry((7,7),(1.,1.));model=make('coupled_source').cuda();activate(model)
    enc=model.encode(u,eq,geom,target);prediction=model.decode(enc,h)
    grads=torch.autograd.grad(prediction.square().sum(),(u,k,r,h))
    assert all(torch.isfinite(g).all() and g.norm()>0 for g in grads)
    queries=torch.stack((h,h*.5),dim=1)
    many=model.decode(enc,queries)
    torch.testing.assert_close(many[:,0],prediction,atol=0,rtol=0)
    torch.testing.assert_close(model.decode(enc,queries.flip(1)).flip(1),many,atol=0,rtol=0)
    parts=[model(u[i:i+1],h[i],Physics(k[i],r[i]),geom,target) for i in range(2)]
    torch.testing.assert_close(torch.cat(parts),prediction,atol=2e-11,rtol=2e-9)


@pytest.mark.parametrize('dtype_name', ('float32','float64'))
def test_response_cuda_reload_cache_and_precision(dtype_name):
    dtype=getattr(torch,dtype_name);u=field(dtype=dtype).cuda()
    eq=Physics(.013,2.4);geom=Geometry((7,7),(1.,1.));model=make('coupled_source',dtype=dtype).cuda();activate(model)
    result=model(u,.03,eq,geom,'fd_same_grid')
    stream=io.BytesIO();torch.save(model.state_dict(),stream);stream.seek(0)
    restored=make('coupled_source',dtype=dtype).cuda();restored.load_state_dict(torch.load(stream,weights_only=True))
    torch.testing.assert_close(restored(u,.03,eq,geom,'fd_same_grid'),result,atol=0,rtol=0)
    encoded=model.encode(u,eq,geom,'fd_same_grid')
    with torch.no_grad(): next(model.parameters()).add_(.01)
    with pytest.raises(ValueError,match='Stale'):model.decode(encoded,.03)
    assert result.dtype==dtype and torch.isfinite(result).all()
    assert not torch.backends.cuda.matmul.allow_tf32


@pytest.mark.parametrize('target', TARGETS)
def test_response_cuda_direct_fno_fresh_process_reload(tmp_path, target):
    from test_response_checkpoint_reload import checkpoint_fixture
    from tdn.analysis.response.training import checkpoint_reload_probe
    # Native-sized 32² FP32 convolutions with a trained nonzero output head;
    # same-process state_dict copies and FP64 parity cannot expose this failure.
    sample = checkpoint_fixture(tmp_path, target, device='cuda')
    result = checkpoint_reload_probe(tmp_path, sample, device='cuda')
    assert result['status'] == 'PASS', result
    assert result['independent_process'] and result['exact_equal']
    assert result['maximum_difference'] == 0.
    assert result['runtime_equal'] and result['state_equal'] and result['checkpoint_equal']


def test_response_cuda_clock_and_physical_nulls():
    u=field().cuda();eq=Physics(.013,2.4);geom=Geometry((7,7),(1.,1.))
    model=make('coupled_source').cuda();activate(model)
    for physical in (Physics(0.,2.4),Physics(.013,0.)):
        torch.testing.assert_close(model(u,.03,physical,geom,'fd_same_grid'),n.anchor(u,.03,physical,geom,'fd_same_grid'),atol=2e-12,rtol=2e-12)
    torch.testing.assert_close(model(u,0.,eq,geom,'fd_same_grid'),u,atol=2e-12,rtol=2e-12)
    clock=create_model(dict(family='coupled_source',width=4,depth=1,rank=2,quadrature_nodes=2,mean_nodes=2,mean_parameterization='clock')).double().cuda()
    with pytest.raises(ValueError,match='bounded nodal'):clock(u,.03,eq,geom,'galerkin_same_grid')
    enc=clock.encode(u,eq,geom,'fd_same_grid')
    assert ((clock.mean_path(enc,.03)>=0)&(clock.mean_path(enc,.03)<=1)).all()


def test_response_cuda_actual_allocation():
    assert os.environ.get('TDN_EXECUTION_MODE')=='desktop-slurm'
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_STEP_ID')
    value=torch.randn(8,device='cuda')
    assert value.device.type=='cuda' and torch.cuda.get_device_properties(0).total_memory>0


@pytest.mark.parametrize('target', TARGETS)
def test_response_cuda_cached_classical_dense_and_batch(target):
    from tdn.analysis.response.performance import CachedDF, FixedMeshDenseOutput
    u=field().cuda();eq=Physics(.013,2.4);geom=Geometry((7,7),(1.,1.))
    cache=CachedDF(u,eq,geom,target,reaction_substeps=2)
    torch.testing.assert_close(cache.decode(.03),n.df(u,.03,eq,geom,target,reaction_substeps=2),atol=2e-12,rtol=2e-12)
    dense=FixedMeshDenseOutput(u,.04,eq,geom,target,steps=4)
    expected=FixedMeshDenseOutput(u.cpu(),.04,eq,geom,target,steps=4)
    torch.testing.assert_close(dense.decode(.017).cpu(),expected.decode(.017),atol=2e-12,rtol=2e-11)
    model=make('coupled_source').cuda();activate(model)
    one=model(u,.03,eq,geom,target)
    batch=model(u.expand(4,-1,-1,-1).clone(),.03,eq,geom,target)
    torch.testing.assert_close(batch,one.expand_as(batch),atol=2e-12,rtol=2e-10)


def test_response_cuda_competitor_auxiliary_source():
    u=field().cuda();eq=Physics(.013,2.4);geom=Geometry((7,7),(1.,1.))
    model=create_model(dict(family='residual_fno',width=4,depth=1,modes=2,rank=2,auxiliary_source_head=True)).double().cuda()
    encoded=model.encode(u,eq,geom,'fd_same_grid')
    predicted=model.source_terms(encoded,.03)
    (predicted-.01*n.center(u)).square().mean().backward()
    assert model.auxiliary_head.weight.grad.norm()>0 and torch.isfinite(model.auxiliary_head.weight.grad).all()

@pytest.mark.parametrize('target',TARGETS)
def test_response_cuda_cached_decoder(target):
    u=field().cuda().requires_grad_();geom=Geometry((7,7),(1.,1.));eq=Physics(.013,2.4)
    for representation in ('learned_basis','fixed_physical','direct_interpolation'):
        config=dict(family='coupled_source',width=4,depth=1,modes=2,rank=2,quadrature_nodes=2,mean_nodes=2,
                    reaction_substeps=2,source_representation=representation)
        transparent=create_model(config).double().cuda();activate(transparent)
        cached=create_model(dict(config,source_decoder='cached_spectral')).double().cuda()
        cached.load_state_dict(transparent.state_dict())
        y=transparent(u,.037,eq,geom,target);z=cached(u,.037,eq,geom,target)
        torch.testing.assert_close(y,z,rtol=2e-10,atol=2e-12)
        dy=torch.autograd.grad(y.square().sum(),(u,*transparent.parameters()),allow_unused=True)
        dz=torch.autograd.grad(z.square().sum(),(u,*cached.parameters()),allow_unused=True)
        for a,b in zip(dy,dz):
            assert (a is None)==(b is None)
            if a is not None:torch.testing.assert_close(a,b,rtol=2e-8,atol=2e-10)
        enc=cached.encode(u,eq,geom,target)
        many=cached.decode(enc,torch.tensor([[.037,.01]],device='cuda'))
        torch.testing.assert_close(many[:,0],z,rtol=2e-8,atol=2e-10)
        with torch.no_grad():next(cached.parameters()).add_(.01)
        with pytest.raises(ValueError,match='Stale'):cached.decode(enc,.03)


def test_response_cuda_remainder_and_assembly():
    u=torch.full_like(field(dtype=torch.float32),.4).cuda();geom=Geometry((7,7),(1.,1.));eq=Physics(0.,2.4)
    config=dict(family='coupled_source',width=4,depth=1,modes=2,rank=2,quadrature_nodes=2,mean_nodes=2,
                numerical_remainder=True,source_decoder='cached_spectral')
    a=create_model(config).cuda();b=create_model(dict(config,assembly='fp64')).cuda()
    with torch.no_grad():a.remainder_head.bias.fill_(1e-10)
    b.load_state_dict(a.state_dict())
    low=a(u,.03,eq,geom,'fd_same_grid');high=b(u,.03,eq,geom,'fd_same_grid')
    assert high.dtype==torch.float64 and low.dtype==torch.float32 and (high-low.double()).abs().max()>0
    torch.testing.assert_close(b(u,0.,eq,geom,'fd_same_grid'),u.double(),rtol=0,atol=0)
