import pytest
import torch
from tdn.analysis.response.models import create_model
from tdn.analysis.response.performance import CachedDF, FixedMeshDenseOutput, measure_model_workloads
from tdn.analysis.response.contracts import Physics
from tdn.analysis.response import numerics as n
from tdn.numerics.types import Geometry


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def case():
    x=torch.arange(7,dtype=torch.float64)/7
    u=(.3+.05*torch.cos(2*torch.pi*x[:,None])+.03*torch.sin(2*torch.pi*x[None,:]))[None,None]
    return u,Physics(.01,2.),Geometry((7,7),(1.,1.))


@pytest.mark.parametrize('target',['fd_same_grid','galerkin_same_grid'])
def test_response_cached_classical_parity_gradients_and_invalidation(target):
    u,eq,geom=case();u.requires_grad_()
    h=torch.tensor(.05,dtype=u.dtype,requires_grad=True)
    cache=CachedDF(u,eq,geom,target,reaction_substeps=2)
    actual=cache.decode(h);expected=n.df(u,h,eq,geom,target,reaction_substeps=2)
    torch.testing.assert_close(actual,expected,atol=3e-15,rtol=3e-14)
    a=torch.autograd.grad(actual.square().sum(),(u,h),retain_graph=True)
    b=torch.autograd.grad(expected.square().sum(),(u,h))
    for got,want in zip(a,b):torch.testing.assert_close(got,want,atol=1e-13,rtol=1e-12)
    with torch.no_grad():u.add_(.01)
    with pytest.raises(ValueError,match='Stale'):cache.decode(h)


def test_response_dense_output_refinement_query_order_and_endpoints():
    u,eq,geom=case();h=.2
    coarse=FixedMeshDenseOutput(u,h,eq,geom,'fd_same_grid',steps=4)
    fine=FixedMeshDenseOutput(u,h,eq,geom,'fd_same_grid',steps=8)
    ref=FixedMeshDenseOutput(u,h,eq,geom,'fd_same_grid',steps=32)
    times=torch.tensor([[0.,.037,.131,.2]],dtype=u.dtype)
    a,b,c=coarse.decode(times),fine.decode(times),ref.decode(times)
    assert (b-c).norm()<(a-c).norm()
    torch.testing.assert_close(coarse.decode(times.flip(1)).flip(1),a,atol=0,rtol=0)
    torch.testing.assert_close(a[:,0],u,atol=0,rtol=0)
    torch.testing.assert_close(a[:,-1],coarse.states[-1],atol=0,rtol=0)
    with pytest.raises(ValueError,match='exceeds'):coarse.decode(.21)


def test_response_workload_panel_has_measured_reuse_batch_and_raw_rounds():
    u,eq,geom=case()
    model=create_model(dict(family='temporal',width=4,depth=1,rank=2)).double()
    result=measure_model_workloads(model,u,.03,eq,geom,'fd_same_grid',repeats=1,warmup=0,
        query_counts=(1,2),batch_sizes=(1,4),profile=False,dense_steps=4)
    assert result['status']=='COMPLETED'
    assert len(result['reuse_panels'])==2 and len(result['batch_panels'])==2
    assert len(result['raw_rounds'])==22
    assert all(p['reuse_parity']['maximum']<1e-13 for p in result['reuse_panels'])
    assert all(p['df_reuse_parity']['maximum']<1e-13 for p in result['reuse_panels'])
    assert all(p['independent_accuracy_sample_count']==1 for p in result['batch_panels'])
    assert result['reuse_panels'][0]['final_endpoint_error'] is None


def test_response_workload_budget_retains_explicit_partial_result():
    class Stop:
        def check(self):raise TimeoutError('bounded test')
    u,eq,geom=case();model=create_model(dict(family='temporal',width=4,depth=1,rank=2)).double()
    result=measure_model_workloads(model,u,.03,eq,geom,'fd_same_grid',budget=Stop())
    assert result['status']=='PARTIAL_BUDGET' and not result['raw_rounds']


def test_response_query_accuracy_uses_prepared_truth_and_missing_stays_missing():
    u,eq,geom=case();model=create_model(dict(family='temporal',width=4,depth=1,rank=2)).double()
    record=dict(target_kind='fd_same_grid',horizon=.03,grid=[7,7],lengths=[1.,1.],
        kappa=.01,reaction_rate=2.,parent_id='independent-parent',query_id='endpoint',
        reference=dict(accepted=True,uncertainty_rms=1e-12,uncertainty_maximum=1e-12))
    teacher=n.integrate(u,.03,eq,geom,'fd_same_grid',substeps=32)
    sample=dict(u0=u,record=record,target=teacher)
    result=measure_model_workloads(model,u,.03,eq,geom,'fd_same_grid',repeats=1,warmup=0,
        query_counts=(1,2),batch_sizes=(),query_teachers=[sample],profile=False,dense_steps=4)
    first,second=result['reuse_panels']
    assert len(first['query_accuracy'])==6 and all(row['reference_accepted'] for row in first['query_accuracy'])
    assert len(second['query_accuracy'])==12
    missing=[row for row in second['query_accuracy'] if row['accuracy_qualification']=='NOT_MEASURED']
    assert len(missing)==6 and all(row['error_rms'] is None for row in missing)
    assert not any(second['all_queries_qualified'].values())
