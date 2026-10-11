"""Focused panel contracts, independent controls and complete-call accounting."""
from types import SimpleNamespace
import json
import numpy as np
import pytest
import torch
from tdn.analysis.response import numerical_controls as controls
from tdn.analysis.response import numerics as n
from tdn.analysis.response.contracts import Physics
from tdn.numerics.types import Geometry
from tdn.analysis.premix.neural import _RunBudget


def test_numerical_control_menu_distinguishes_etdrk4_from_lawson_and_refinements():
    specs=controls.control_specs()
    assert len({s['id'] for s in specs})==len(specs)==29
    menus={m:{s['steps'] for s in specs if s['method']==m} for m in ('df','rf','etdrk4','shared_gl4','lawson4')}
    assert all(menu=={1,2,4,8} for menu in menus.values())
    x=torch.arange(9,dtype=torch.float64)/9;u=(.4+.2*torch.cos(2*torch.pi*x))[None,None,:]
    eq=Physics(.04,8.);geom=Geometry((9,),(1.,))
    etd=next(s for s in specs if s['id']=='etdrk4/steps-1')
    lawson=next(s for s in specs if s['id']=='lawson4/steps-1')
    a=controls.apply_control(u,.12,eq,geom,'fd_same_grid',etd)
    b=controls.apply_control(u,.12,eq,geom,'fd_same_grid',lawson)
    assert (a-b).abs().max()>1e-7
    torch.testing.assert_close(a,n.etdrk4_step(u,.12,eq,geom,'fd_same_grid'))


def make_sample(accepted=True):
    x=torch.arange(7,dtype=torch.float32)/7;u=(.4+.12*torch.cos(2*torch.pi*x))[None,None,:]
    eq=Physics(.01,3.);geom=Geometry((7,),(1.,));h=.06
    target=n.integrate(u.double(),h,eq,geom,'fd_same_grid',substeps=64)
    r=dict(parent_id='held-fixed',query_id='query-fixed',split='validation',grid=[7],lengths=[1.],horizon=h,
        kappa=.01,reaction_rate=3.,target_kind='fd_same_grid',reference=dict(accepted=accepted,
        status='ACCEPTED' if accepted else 'UNRESOLVED_REFERENCE',uncertainty_rms=1e-12,uncertainty_maximum=2e-12))
    return dict(u0=u,target=target,anchor_fp32=n.anchor(u,h,eq,geom,'fd_same_grid'),
        anchor_fp64=n.anchor(u.double(),h,eq,geom,'fd_same_grid'),reference_accepted=accepted,record=r)


def context(tmp_path):
    records=[]
    ctx=SimpleNamespace(path=tmp_path,unit=dict(parent_ids=['held-fixed'],grid=[7],horizons=[.06],target_kind='fd_same_grid',data_stages=['ref']),
        prerequisites={'ref':tmp_path/'ref'},device='cpu',budget=_RunBudget(120,None,'cpu'),
        protocol=dict(profile='smoke',dimension=1,train_grid=7,validation_horizons=[.06],cost=dict(repeats=2,warmup=0),
        evaluation=dict(rms_target=2e-5)),record=lambda *args,**kwargs:records.append((args,kwargs)))
    return ctx,records


def test_numerical_control_declared_unresolved_parent_is_never_replaced(monkeypatch,tmp_path):
    sample=make_sample(False);extra=make_sample(True);extra['record']=extra['record']|{'parent_id':'easy-replacement','query_id':'replacement'}
    monkeypatch.setattr(controls,'load_samples',lambda path,device:[sample,extra])
    ctx,_=context(tmp_path)
    keys,found=controls._load_declared_samples(ctx)
    assert len(keys)==len(found)==1
    assert not found[keys[0]]['reference_accepted']
    ctx.unit['parent_ids']=['missing-fixed']
    keys,found=controls._load_declared_samples(ctx)
    assert len(keys)==1 and not found


def test_numerical_control_panel_saves_physical_deployed_randomized_and_refined_evidence(monkeypatch,tmp_path):
    sample=make_sample();monkeypatch.setattr(controls,'load_samples',lambda path,device:[sample])
    ctx,records=context(tmp_path)
    summary=controls.run(ctx)
    payload=json.loads((tmp_path/'numerical_controls.json').read_text())
    assert summary['rows']==29 and summary['timing_samples']==58
    assert payload['complete'] and len(payload['comparisons'])==4
    assert all(r['physical']['prediction_finite'] and r['deployed']['prediction_finite'] for r in payload['rows'])
    assert all(r['cost']['repeats_completed']==2 and r['cost']['median_seconds']>0 for r in payload['rows'])
    orders=[[r['method'] for r in payload['timing_samples'] if r['round']==j] for j in range(2)]
    assert set(orders[0])==set(orders[1]) and orders[0]!=orders[1]
    assert all(r['parity_with_first_call']['maximum']==0 for r in payload['timing_samples'])
    parity=next(x for x in payload['comparisons'] if x['kind']=='same_rule_parity')
    assert parity['fp64_disagreement']['maximum']<1e-12
    with np.load(tmp_path/'numerical_predictions.npz',allow_pickle=False) as arrays:
        assert len(arrays.files)==58
    assert not [c for _,row in records for c in row['checks'] if c['verdict']=='BAD']


def test_signed_error_decomposition_retains_destructive_cross_terms():
    geom=Geometry((9,),(1.,));x=torch.arange(9,dtype=torch.float64)/9
    shape=torch.cos(2*torch.pi*x)[None,None,:]
    fine=torch.full_like(shape,.5)
    galerkin=fine+shape*.2
    discrete=galerkin-shape*.3
    prediction=discrete+shape*.11
    result=controls.error_decomposition(prediction,discrete,galerkin,fine,geom)
    metrics=result['metrics']
    assert metrics['vector_reconstruction_maximum']<1e-16
    assert metrics['squared_identity_discrepancy']<1e-16
    assert metrics['sum_signed_cross_terms']<0
    assert metrics['sum_component_squared_norms']>100*metrics['total_squared_norm']
    assert len(metrics['cross_terms'])==3
    assert any(row['cosine']<-.999 for row in metrics['cross_terms'])
    assert any(row['cosine']>.999 for row in metrics['cross_terms'])


def test_counterreference_and_spectral_stress_panel_are_measured_on_same_input(tmp_path):
    from tdn.analysis.response.fields import make_parent,sample_parent,digest,array_digest
    from tdn.analysis.response.references import generate_reference
    parent=make_parent(83098,'validation',2,regime='triads',dimension=2)
    parent.update(kappa=.003,reaction_rate=3.)
    parent['parent_sha256']=digest({k:v for k,v in parent.items() if k!='parent_sha256'})
    u=sample_parent(parent,[8,8]).float();geom=Geometry((8,8),(1.,1.));eq=Physics(.003,3.);h=.04
    options=dict(tolerance=1e-8,max_steps=256,initial_steps=8,max_seconds_per_query=30,max_grid=32,spatial_factors=[2,4],crosscheck=True)
    teacher,meta=generate_reference(u,h,eq,geom,'fd_same_grid',options)
    record=dict(parent_id=parent['parent_id'],query_id='q-decomp',split='validation',phase_variant='original',
        grid=[8,8],lengths=[1.,1.],horizon=h,kappa=.003,reaction_rate=3.,target_kind='fd_same_grid',reference=meta,
        input_sha256=array_digest(u))
    sample=dict(u0=u,target=teacher,record=record,reference_accepted=meta['accepted'])
    rows=[];ctx=SimpleNamespace(path=tmp_path,unit={'target_kind':'fd_same_grid'},device='cpu',budget=_RunBudget(120,None,'cpu'),
        protocol=dict(parents=[parent],root_seed=83098,reference=options,evaluation={'rms_target':2e-5}),
        record=lambda *args,**kwargs:rows.append((args,kwargs)))
    summary=controls.run_auxiliary_panels(ctx,{('fixed',):sample})
    assert summary['stress_rows']==24 and summary['decomposition_rows']==4
    decomposition=json.loads((tmp_path/'decomposition.json').read_text())
    assert {r['input_sha256'] for r in decomposition['teachers'].values()}=={array_digest(u)}
    assert all(row['metrics']['vector_reconstruction_maximum']<1e-13 for row in decomposition['rows'])
    stress=json.loads((tmp_path/'spectral_stress.json').read_text())
    assert stress['complete'] and len(stress['fields'])==2
    assert all(row['cluster_parent_id']==parent['parent_id'] for row in stress['rows'])
    assert {field['parent']['regime'] for field in stress['fields']}=={'broadband','grid_relative'}
    assert not [c for _,row in rows for c in row['checks'] if c['verdict']=='BAD']


def test_auxiliary_panels_do_not_replace_absent_first_declared_parent(tmp_path):
    ctx=SimpleNamespace(unit=dict(parent_ids=['missing-first','easy-second']),path=tmp_path)
    replacement=dict(record={'parent_id':'easy-second'})
    result=controls.run_auxiliary_panels(ctx,{('second',):replacement})
    assert result['status']=='NOT_MEASURED'
    assert 'no replacement' in result['reason']
