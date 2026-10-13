import pytest
import torch
from tdn.analysis.response.models import create_model
from tdn.analysis.response.diagnostic_performance import benchmark_model
from tdn.analysis.response.numerics import anchor,Physics,Geometry


@pytest.mark.parametrize('family',['direct_fno','residual_fno','temporal','adaptive_temporal','spatial_temporal','coupled_source'])
def test_selected_model_complete_and_cached_call_measured_on_actual_cpu(family):
    u=torch.linspace(.15,.65,36).reshape(1,1,6,6)
    geom=Geometry((6,6),(1.,1.));eq=Physics(.003,3.)
    normalization=dict(mean_scale=.01,centered_scale=.01,full_scale=.4,full_mean=.4,floor=1e-7)
    model=create_model(dict(family=family,width=4,depth=1,rank=4,normalization=normalization))
    sample=dict(u0=u,target=anchor(u,.04,eq,geom,'fd_same_grid').double(),
        anchor_fp32=anchor(u,.04,eq,geom,'fd_same_grid'),reference_accepted=True,
        anchor_fp64=anchor(u.double(),.04,eq,geom,'fd_same_grid'),
        record=dict(parent_id='validation-parent',query_id='fixed-query',target_kind='fd_same_grid',split='validation',
            horizon=.04,kappa=.003,reaction_rate=3.,grid=[6,6],lengths=[1.,1.],reference=dict(accepted=True,uncertainty_rms=0.,uncertainty_maximum=0.)))
    result=benchmark_model(model,[sample],normalization,repeats=2,warmup=0)
    assert result['device']=='cpu' and result['status']=='MEASURED'
    assert result['query_id']=='fixed-query' and len(result['timing_rows'])==4
    assert result['complete_call_median_seconds']>0 and result['p99_seconds'] is None
    assert result['utility_status'].startswith('NOT_ESTABLISHED')
    assert bool(result['cached_decode_rows'])==model.architecture_metadata()['reusable_encoding']


def test_missing_reference_is_na_and_unsafe_repetitions_rejected():
    model=create_model(dict(family='temporal',width=4,depth=1))
    assert benchmark_model(model,[],{},repeats=2)['status']=='NA'
    with pytest.raises(ValueError):benchmark_model(model,[],{},repeats=1001)
