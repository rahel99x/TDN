"""Learned-decoder quadrature checks preserve frozen selection and raw gaps."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch
from tdn.analysis.response.models import create_model
from tdn.analysis.response import quadrature_audit as audit
from tdn.analysis.response import numerics as n
from tdn.analysis.response.references import equation_geometry
from tdn.research.experiment import Budget


@pytest.fixture(autouse=True)
def one_thread():
    previous=torch.get_num_threads();torch.set_num_threads(1);torch.manual_seed(74513)
    yield
    torch.set_num_threads(previous)


def sample(parent='first',h=.06,target='fd_same_grid',phase='original'):
    x=torch.arange(6)/6
    u=(.33+.08*torch.cos(2*torch.pi*x[:,None])+.02*torch.sin(4*torch.pi*x[None,:]+.2))[None,None]
    r=dict(parent_id=parent,query_id=f'{parent}/{h}/{phase}',target_kind=target,grid=[6,6],lengths=[1.,1.],
        horizon=h,kappa=.004,reaction_rate=3.,input_sha256='declared-input',phase_variant=phase,
        reference=dict(accepted=True,status='ACCEPTED',uncertainty_rms=1e-10,uncertainty_maximum=1e-9))
    eq,geom=equation_geometry(r)
    truth=n.integrate(u.double(),h,eq,geom,target,substeps=32)
    return dict(u0=u,target=truth,record=r)


def method(decoder='quadrature',representation='learned_basis',seed=1):
    m=create_model(dict(family='coupled_source',width=3,depth=1,modes=1,rank=2,quadrature_nodes=4,mean_nodes=4,
        reaction_substeps=2,source_decoder=decoder,source_representation=representation))
    with torch.no_grad():
        for name,p in m.named_parameters():
            if 'head' in name:p.add_(.15*torch.randn_like(p))
    f=lambda x:None;f.model=m;f.metadata=dict(method_id=f'coupled_source::{decoder}::{representation}',seed=seed,
        variant_id='primary' if decoder=='quadrature' else decoder,checkpoint_path='selected.pt',selection_sha256='selection')
    return f


def context(tmp_path,seconds=60):
    protocol=dict(dimension=2,units={'references':dict(target_kind='fd_same_grid',grid=6,parent_ids=['first','second'],horizons=[.02,.06])},
        response={'quadrature_audit':{'seconds':seconds}})
    return SimpleNamespace(path=tmp_path,protocol=protocol,unit={'data_stages':['references']},budget=Budget(90))


@pytest.mark.parametrize('decoder,representation',[('quadrature','learned_basis'),('cached_spectral','learned_basis'),('cached_spectral','direct_interpolation')])
def test_nonzero_model_refinement_preserves_original_and_accepted_reference(tmp_path,decoder,representation):
    f=method(decoder,representation);m=f.model;m.train();s=sample();eq,geom=equation_geometry(s['record'])
    old=deepcopy(m.state_dict());enc=m.encode(s['u0'],eq,geom,'fd_same_grid');before=m.decode(enc,.06)
    result=audit.run(context(tmp_path),[sample('second'),sample(h=.02),s],{'chosen':f})
    assert result['status']=='COMPLETED'
    assert len(result['rows'])==6 and all(r['parent_id']=='first' and r['horizon']==.06 for r in result['rows'])
    assert all(r['reference_error'] is not None and r['reference_accepted'] for r in result['rows'])
    assert all(r['finest_completed_joint_nodes']==32 and r['dtype']=='torch.float64' for r in result['rows'])
    assert all(r['elapsed_seconds']>0 for r in result['rows'])
    by_nodes={(r['source_nodes'],r['mean_nodes']):r for r in result['rows']}
    if representation!='direct_interpolation':
        assert by_nodes[(4,4)]['difference_rms_vs_finest']>0
        assert by_nodes[(8,8)]['difference_rms_vs_finest']<by_nodes[(4,4)]['difference_rms_vs_finest']
    assert by_nodes[(32,32)]['difference_rms_vs_finest']==0
    assert result['arithmetic_comparisons'][0]['status']=='COMPLETED'
    assert result['arithmetic_comparisons'][0]['difference_rms']>0
    assert m.training and m.nodes==m.mean_nodes==4
    assert all(torch.equal(value,m.state_dict()[key]) for key,value in old.items())
    torch.testing.assert_close(m.decode(enc,.06),before,rtol=0,atol=0)
    assert json.loads((tmp_path/'quadrature_refinement.json').read_text())['rows']==result['rows']


def test_missing_declared_original_parent_is_not_replaced_or_phase_twinned(tmp_path):
    result=audit.run(context(tmp_path),[sample('second'),sample('first',phase='phase_twin')],{'chosen':method()})
    assert result['status']=='PARTIAL'
    assert len(result['rows'])==6 and all(r['status']=='NOT_MEASURED' for r in result['rows'])
    assert all(r['parent_id']=='first' and r['difference_rms_vs_finest'] is None for r in result['rows'])
    assert result['arithmetic_comparisons']==[]


def test_configuration_aliases_are_not_imputed_and_cap_retains_denominator(tmp_path):
    a,b=method(seed=1),method(seed=2)
    with torch.no_grad():b.model.source_head.bias.add_(.2)
    result=audit.run(context(tmp_path,seconds=0),[sample()],{'seed-2':b,'seed-1':a})
    assert result['selected_models']==['seed-1']
    assert result['aliases'][0]['model_id']=='seed-2' and result['aliases'][0]['status']=='NOT_MEASURED'
    assert all(r['status']=='CENSORED' for r in result['rows']) and len(result['rows'])==6
    assert all(r['difference_rms_vs_finest'] is None for r in result['rows'])


def test_physical_nyquist_norm_and_unaccepted_reference_are_explicit(tmp_path):
    s=sample(target='galerkin_same_grid');eq,geom=equation_geometry(s['record'])
    alternation=(-1.)**torch.arange(6,dtype=torch.float64)
    corner=(alternation[:,None]*alternation[None,:])[None,None]
    norms=audit._norms(corner,geom,'galerkin_same_grid')
    assert norms['physical_rms']==pytest.approx(.5) and norms['nodal_rms']==pytest.approx(1.)
    ctx=context(tmp_path);ctx.protocol['units']['references']['target_kind']='galerkin_same_grid'
    s['record']['reference']['accepted']=False
    result=audit.run(ctx,[s],{'chosen':method('cached_spectral')})
    assert all(r['reference_error'] is None and not r['reference_accepted'] for r in result['rows'])


def test_no_coupled_checkpoint_explicit_not_measured(tmp_path):
    result=audit.run(context(tmp_path),[sample()],{})
    assert result['status']=='NOT_MEASURED' and result['rows'][0]['status']=='NOT_MEASURED'
