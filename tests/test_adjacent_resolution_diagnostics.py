"""Reference bank integrity and actual independent numerical teacher checks."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tdn.analysis.adjacent import resolution_diagnostics as rd
from tdn.analysis.adjacent.protocol import build_protocol
from tdn.analysis.roadmap.numerics import fourier_resample
from tdn.numerics import Equation,Geometry


@pytest.fixture
def protocol():
    return build_protocol('resolution-smoke')


def test_fixed_parent_is_same_physical_field_on_64_and_128(protocol):
    for index in range(7):
        a=rd.resolution_parent(protocol,'train',index,64)
        b=rd.resolution_parent(protocol,'train',index,128)
        assert a==b
        u=rd.sample_resolution_parent(a,64); fine=rd.sample_resolution_parent(b,128)
        torch.testing.assert_close(fourier_resample(fine,(64,64)),u,atol=1e-14,rtol=0)
        assert u.min()>0 and u.max()<1
        assert float((u-u.mean()).square().mean().sqrt())==pytest.approx(a['rms'],abs=1e-15)


def test_relative_stress_actually_adds_scales_and_keeps_cluster(protocol):
    a=rd.resolution_parent(protocol,'diagnostic',3,64,regime='resolution_relative_highpair')
    b=rd.resolution_parent(protocol,'diagnostic',3,128,regime='resolution_relative_highpair')
    assert max(i for i,j in a['modes'])==24
    assert max(i for i,j in b['modes'])==48
    assert a['parent_id']!=b['parent_id']
    assert a['field_cluster']==b['field_cluster']
    assert a['bandwidth_policy']==b['bandwidth_policy']=='resolution_relative'


def test_independent_parents_vary_power_and_splits_are_disjoint(protocol):
    a=rd.resolution_parent(protocol,'train',0)
    b=rd.resolution_parent(protocol,'train',7)
    assert not np.allclose(a['amplitudes'],b['amplitudes'])
    c=rd.resolution_parent(protocol,'evaluation',0)
    assert a['field_cluster']!=c['field_cluster'] and a['seed']!=c['seed']
    assert a['parent_sha256']!=c['parent_sha256']
    with pytest.raises(ValueError,match='Nyquist'): rd.sample_resolution_parent(rd.resolution_parent(protocol,'train',1),32)
    altered=copy.deepcopy(a); altered['mean']+=.01
    with pytest.raises(ValueError,match='hash'): rd.sample_resolution_parent(altered,64)


@pytest.mark.parametrize('track',['discrete','continuum'])
def test_lawson_teacher_has_real_independent_crosscheck(protocol,track):
    parent=rd.resolution_parent(protocol,'train',0)
    u=rd.sample_resolution_parent(parent,8); equation,geometry=rd._physics(parent,8)
    reference,meta,difference=rd.converged_reference(u,.04,equation,geometry,track,protocol['resolution']['reference'])
    assert meta['accepted'] and meta['reference_accepted']
    assert meta['independent_crosscheck']['method']=='Cox-Matthews ETDRK4'
    assert meta['refinement_substeps']==[8,16,32]
    assert meta['uncertainty_rms']>=meta['independent_crosscheck']['difference_from_lawson']['error_rms']
    assert reference.dtype==torch.float64 and difference.shape==u.shape
    assert meta['reference_seconds']>0


def _tiny_unit(protocol):
    return dict(grid=8,track='discrete',split='train',field_indices=[0])


def test_bank_roundtrip_hash_and_duplicate_protection(protocol,tmp_path,monkeypatch):
    monkeypatch.setattr(rd,'_bank_identity',lambda p,u:dict(protocol_sha256=rd.digest(p),source_tree_sha256='test-source'))
    result=rd.prepare_reference_bank(protocol,_tiny_unit(protocol),tmp_path)
    assert result['reference_count']==result['accepted_references']==1
    row=rd.iterate_bank_entries(tmp_path)[0]
    u,truth,meta=rd.load_bank_entry([tmp_path],row['parent_id'],'discrete',8,.04)
    torch.testing.assert_close(u,rd.sample_resolution_parent(meta['parent'],8),rtol=0,atol=0)
    assert truth.shape==u.shape
    assert meta['parent']['index']==0 and meta['parent']['domain']==[1.,1.]
    with pytest.raises(ValueError,match='one matching'): rd.load_bank_entry([tmp_path,tmp_path],row['parent_id'],'discrete',8,.04)
    with pytest.raises(ValueError,match='Preserve'): rd.prepare_reference_bank(protocol,_tiny_unit(protocol),tmp_path,resume=False)
    file=tmp_path/row['reference_file']; file.write_bytes(file.read_bytes()+b'tamper')
    with pytest.raises(ValueError,match='hash'): rd.iterate_bank_entries(tmp_path)


def test_reference_journal_retains_completed_horizon_and_resumes(protocol,tmp_path,monkeypatch):
    p=copy.deepcopy(protocol);p['resolution']['training_horizons']=[.02,.04]
    monkeypatch.setattr(rd,'_bank_identity',lambda p,u:dict(protocol_sha256=rd.digest(p),source_tree_sha256='stable'))
    original=rd.converged_reference
    def interrupt(initial,h,*args,**kwargs):
        if h==.04: raise TimeoutError('bounded test interruption')
        return original(initial,h,*args,**kwargs)
    monkeypatch.setattr(rd,'converged_reference',interrupt)
    with pytest.raises(TimeoutError):rd.prepare_reference_bank(p,_tiny_unit(p),tmp_path)
    raw=json.loads((tmp_path/'reference_bank.json').read_text())
    assert raw['status']=='INTERRUPTED' and len(raw['entries'])==1
    first_hash=raw['entries'][0]['reference_sha256']
    with pytest.raises(ValueError,match='completed reference'):rd.iterate_bank_entries(tmp_path)
    changed=copy.deepcopy(p);changed['resolution']['rms']*=.9
    with pytest.raises(ValueError,match='differs'):rd.prepare_reference_bank(changed,_tiny_unit(p),tmp_path,resume=True)
    monkeypatch.setattr(rd,'converged_reference',original)
    result=rd.prepare_reference_bank(p,_tiny_unit(p),tmp_path,resume=True)
    rows=rd.iterate_bank_entries(tmp_path)
    assert len(rows)==2 and rows[0]['reference_sha256']==first_hash
    raw=json.loads((tmp_path/'reference_bank.json').read_text())
    assert len(raw['attempts'])==2 and raw['attempts'][0]['error'].startswith('TimeoutError')
    assert result['accepted_references']==2


def test_bounded_reference_cannot_claim_acceptance_without_refinement(protocol):
    parent=rd.resolution_parent(protocol,'train',0);u=rd.sample_resolution_parent(parent,8)
    equation,geometry=rd._physics(parent,8);options=dict(protocol['resolution']['reference'],max_substeps=8)
    _,meta,_=rd.converged_reference(u,.04,equation,geometry,'discrete',options)
    assert not meta['reference_accepted'] and meta['independent_crosscheck'] is None


def test_diagnostic_floor_rows_use_matched_nodes_and_bounded_oracles(protocol,tmp_path):
    p=copy.deepcopy(protocol);p['resolution']['model_config'].update(modes=2,split_modes=1)
    p['resolution'].update(physical_cutoff=2,split_cutoff=1)
    ctx=SimpleNamespace(protocol=p,path=tmp_path,unit=dict(grid=8,track='discrete'),budget=SimpleNamespace(check=lambda:None))
    ex=rd._Diagnostics(ctx);parent=rd.resolution_parent(p,'train',0)
    u=rd.sample_resolution_parent(parent,8);eq,geo=rd._physics(parent,8)
    ref,meta,_=rd.converged_reference(u,.04,eq,geo,'discrete',p['resolution']['reference'])
    ex.floors(parent,u,ref,meta,.04)
    rows=json.loads((tmp_path/'resolution_diagnostic_rows.json').read_text())
    oracles=[r for r in rows if r['category']=='oracle']
    assert len(oracles)==16
    assert {r['nodes'] for r in oracles}=={2,4}
    assert {r['output_modes'] for r in oracles}=={2,None}
    for row in oracles:
        assert row['oracle']['effective_rank']<=row['oracle']['algebraic_rank']
        if row['method']=='origin_channels_oracle':
            assert all(.25-1e-10<=c<=1.75+1e-10 for c in row['oracle']['coefficients'])
    floor=next(r for r in rows if r['category']=='compression_floor')
    assert floor['maximum_norm_floor'] is None and not floor['deployable']
    assert (tmp_path/floor['array_file']).is_file()


def test_diagnostic_endpoint_parity_collision_and_autonomous_paths(protocol,tmp_path):
    p=copy.deepcopy(protocol);p['resolution']['model_config'].update(modes=2,split_modes=1)
    p['resolution'].update(physical_cutoff=2,split_cutoff=1)
    ctx=SimpleNamespace(protocol=p,path=tmp_path,unit=dict(grid=8,track='continuum'),budget=SimpleNamespace(check=lambda:None))
    ex=rd._Diagnostics(ctx);parent=rd.resolution_parent(p,'train',0)
    u=rd.sample_resolution_parent(parent,8);eq,geo=rd._physics(parent,8)
    ref,meta,_=rd.converged_reference(u,.04,eq,geo,'continuum',p['resolution']['reference'])
    ex.endpoint(parent,u,ref,meta,.04,schedules=[1,2],save=False)
    ex.collisions(parent);ex.amplitude(parent);ex.trajectories(parent)
    assert {'D02','D03','D05','D06'} <= {r['diagnostic'] for r in ex.rows}
    endpoints=[r for r in ex.rows if r['category']=='endpoint']
    assert len(endpoints)==16
    assert all(len(r['latency_samples_seconds'])==2 and r['timing_dtype']=='float64' for r in endpoints)
    for row in endpoints:
        for target,qualified in row['qualified_targets'].items():
            expected=row['reference_accepted'] and row['error_rms']+row['reference_uncertainty']<=float(target) and row['error_max']+row['reference_uncertainty_max']<=float(target)
            assert qualified is expected
    parity=[r for r in ex.rows if r['category']=='amplitude_parity']
    assert len(parity)==3 and all(r['background_integrator_error_rms']>=0 for r in parity)
    traces=[r for r in ex.rows if r['category']=='rollout_trace']
    assert len(traces)==4
    with np.load(tmp_path/traces[0]['array_file'],allow_pickle=False) as data:
        assert list(data['times'])==[0.,.04,.08]
        assert data['prediction'].shape==(3,1,8,8)


def test_completed_inner_bank_recovers_outer_interruption_without_regeneration(protocol,tmp_path,monkeypatch):
    import shutil
    monkeypatch.setattr(rd,'_bank_identity',lambda p,u:dict(protocol_sha256=rd.digest(p),source_tree_sha256='stable'))
    original=tmp_path/'original';recovered=tmp_path/'recovered'
    expected=rd.prepare_reference_bank(protocol,_tiny_unit(protocol),original)
    # Outer orchestration failed after the completed inner journal was saved.
    (original/'summary.json').write_text(json.dumps(dict(status='INTERRUPTED',error='budget check after dispatch')))
    before={p.name:p.read_bytes() for p in original.iterdir()}
    shutil.copytree(original,recovered)
    def must_not_run(*args,**kwargs):raise AssertionError('Completed teachers must be reused')
    monkeypatch.setattr(rd,'converged_reference',must_not_run)
    assert rd.prepare_reference_bank(protocol,_tiny_unit(protocol),recovered,resume=True)==expected
    assert {p.name:p.read_bytes() for p in original.iterdir()}==before
    assert {p.name:p.read_bytes() for p in recovered.iterdir()}==before
    journal=json.loads((recovered/'reference_bank.json').read_text())
    assert len(journal['attempts'])==1
    journal['entries']=[]
    (recovered/'reference_bank.json').write_text(json.dumps(journal))
    with pytest.raises(ValueError,match='inventory is incomplete'):
        rd.prepare_reference_bank(protocol,_tiny_unit(protocol),recovered,resume=True)


def test_projected_teacher_uncertainty_measures_actual_projection(protocol):
    parent=rd.resolution_parent(protocol,'train',0);u=rd.sample_resolution_parent(parent,16)
    eq,geo=rd._physics(parent,16)
    _,meta,difference=rd.converged_reference(u,.04,eq,geo,'continuum',protocol['resolution']['reference'],projection_grid=8)
    projected=meta['projected_temporal_uncertainty']
    actual=rd.error_metrics(fourier_resample(difference,(8,8)))
    assert projected['grid']==8
    assert projected['differences']['lawson_refinement']==actual
    assert projected['uncertainty_rms']>=max(r['error_rms'] for r in projected['differences'].values())
    assert projected['uncertainty_max_bound']>=max(r['error_max'] for r in projected['differences'].values())


@pytest.mark.parametrize('spatial_difference,accepted,contrast_accepted',[(1e-3,False,False),(1e-12,True,True),(.95e-9,True,False)])
def test_spatial_refinement_needs_space_convergence_and_charges_all_teacher_uncertainties(protocol,tmp_path,monkeypatch,spatial_difference,accepted,contrast_accepted):
    p=copy.deepcopy(protocol);p['resolution']['reference']['tolerance']=1e-9
    ctx=SimpleNamespace(protocol=p,path=tmp_path,unit=dict(grid=8,track='continuum'),budget=SimpleNamespace(check=lambda:None))
    ex=rd._Diagnostics(ctx);parent=rd.resolution_parent(p,'train',0)
    coarse=torch.full((1,1,8,8),.4,dtype=torch.float64)
    coarse_meta=dict(reference_accepted=True,uncertainty_rms=3e-11,uncertainty_max_bound=4e-11)
    def teacher(parent,h,n,*,projection_grid):
        assert projection_grid==8
        value=.4 if n==16 else .4+spatial_difference
        fine=torch.full((1,1,n,n),value,dtype=torch.float64)
        temporal=1e-11 if n==16 else 2e-11
        meta=dict(reference_accepted=True,uncertainty_rms=temporal,uncertainty_max_bound=temporal,
            projected_temporal_uncertainty=dict(grid=8,uncertainty_rms=temporal,uncertainty_max_bound=temporal))
        return fine,fine,meta,torch.full_like(fine,temporal)
    monkeypatch.setattr(ex,'teacher',teacher)
    ex.spatial(parent,.04,coarse,coarse_meta)
    row=ex.rows[-1]
    assert row['temporal_reference_accepted']
    assert row['spatial_resolution_accepted'] is accepted
    assert row['contrast_reference_accepted'] is contrast_accepted
    assert row['reference_accepted'] is contrast_accepted
    assert row['refined_reference_uncertainty_rms']==pytest.approx(spatial_difference+3e-11,abs=1e-16)
    assert row['reference_uncertainty']==pytest.approx(spatial_difference+6e-11,abs=1e-16)
    assert row['reference_uncertainty_max']==pytest.approx(spatial_difference+7e-11,abs=1e-16)
    assert row['fine_grids']==[16,32]


@pytest.mark.parametrize('intervals,resolved,expected',[
    ([[.4,.5],[1.,1.1]],True,True),
    ([[.4,1.2],[1.,1.1]],True,False),
    ([None,[1.,1.1]],True,None),
    ([[.4,.5],[1.,1.1]],False,None),
])
def test_feature_collision_requires_individually_resolved_feasible_fits(intervals,resolved,expected):
    rows=[dict(acceptable_gain_interval=v,oracle=dict(resolved=resolved),features=[.43,.2,.3],reference_accepted=True) for v in intervals]
    result=rd._collision_assessment(rows)
    assert result['response_insufficiency_demonstrated'] is expected
    if expected is None:
        assert result['collision_inference']=='NA'
        assert result['collision_inference_reason']
    if intervals[0] is None:
        assert result['acceptable_intervals_overlap'] is None
        assert 'scalar-family' in result['collision_inference_reason']


@pytest.mark.parametrize('grid',[64,128])
def test_joint_teacher_refines_crosscheck_after_lawson_accepts(protocol,grid):
    parent=rd.resolution_parent(protocol,'evaluation',1,grid)
    initial=rd.sample_resolution_parent(parent,grid);eq,geo=rd._physics(parent,grid)
    options=protocol['resolution']['reference']
    _,meta,_=rd.converged_reference(initial,.04,eq,geo,'continuum',options,projection_grid=grid//2)
    assert meta['reference_accepted']
    assert meta['refinement_substeps'][-1]==128
    assert len(meta['joint_refinements'])==2
    rejected,accepted=meta['joint_refinements']
    assert rejected['temporal_accepted'] and not rejected['reference_accepted']
    assert rejected['independent_crosscheck']['own_refinement']['error_max']>options['tolerance']
    assert accepted['reference_accepted']
    assert max(meta['uncertainty_rms'],meta['uncertainty_max_bound'])<=options['tolerance']
    assert all('projected_temporal_uncertainty' in trial for trial in meta['joint_refinements'])
    assert meta['reference_seconds']>=sum(trial['trial_elapsed_seconds'] for trial in meta['joint_refinements'])
    assert meta['joint_refinement_stop']=='accepted'


def test_joint_teacher_crosscheck_failure_honors_existing_substep_limit(protocol,monkeypatch):
    parent=rd.resolution_parent(protocol,'train',0);initial=rd.sample_resolution_parent(parent,8)
    eq,geo=rd._physics(parent,8);options=dict(protocol['resolution']['reference'],max_substeps=64)
    actual=rd.numerics.etdrk4_step;steps=[]
    def biased_etd(state,h,*args,**kwargs):
        steps.append(h)
        return actual(state,h,*args,**kwargs)+1e-7*h
    monkeypatch.setattr(rd.numerics,'etdrk4_step',biased_etd)
    _,meta,_=rd.converged_reference(initial,.04,eq,geo,'discrete',options)
    assert not meta['reference_accepted']
    assert meta['joint_refinement_stop']=='declared_substep_limit'
    assert meta['refinement_substeps'][-1]==64
    assert len(meta['joint_refinements'])==2
    assert max(round(.04/h) for h in steps)==64
    assert all(max(trial['refinement_substeps'])<=64 for trial in meta['joint_refinements'])
