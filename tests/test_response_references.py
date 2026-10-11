"""Actual-input teacher contracts and independent reference failure checks."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from tdn.analysis.response.fields import make_parent,sample_parent,validate_parent_splits,stable_seed
from tdn.analysis.response.references import generate_reference,temporal_reference,prepare_dataset,load_samples,equation_geometry
from tdn.analysis.response.contracts import Physics
from tdn.numerics.types import Geometry


def test_parent_independence_and_translation_duplicate_detection():
    a=make_parent(92,'train',0);b=make_parent(92,'validation',0)
    assert a['parent_id']!=b['parent_id'];assert a['translation_invariant_shape_hash']!=b['translation_invariant_shape_hash']
    assert a==make_parent(92,'train',0)
    translated=dict(a,parent_id='different-id',split='validation')
    with pytest.raises(ValueError,match='translated'):validate_parent_splits([a,translated])
    for i in range(16):
        u=sample_parent(make_parent(92,'train',i),17)
        assert u.min()>=0 and u.max()<=1


@pytest.mark.parametrize('target',['fd_same_grid','galerkin_same_grid'])
def test_actual_fp32_input_teacher_matches_independent_logistic_constant(target):
    u=torch.full((1,1,5,5),.123456789,dtype=torch.float32)
    eq=Physics(.03,3.);geom=Geometry((5,5),(1.,1.));h=.06
    value,meta=generate_reference(u,h,eq,geom,target,dict(initial_steps=4,max_steps=256,tolerance=1e-10))
    exact=u.double()/(u.double()+(1-u.double())*np.exp(-3*h))
    assert meta['accepted'];assert meta['cross_solver']['success']
    torch.testing.assert_close(value,exact,atol=1e-10,rtol=0)
    # The intended decimal continuous field is not the actual FP32 input.
    unrelated=.123456789/(.123456789+(1-.123456789)*np.exp(-3*h))
    assert float((value-unrelated).abs().max())>1e-10


def test_refinement_cap_does_not_accept_an_unchecked_teacher():
    u=sample_parent(make_parent(90,'train',2),8)
    value,meta=temporal_reference(u,.06,Physics(.03,8.),Geometry((8,8),(1.,1.)),'fd_same_grid',dict(initial_steps=8,max_steps=8,tolerance=1e-10))
    assert not meta['accepted'];assert len(meta['temporal_refinements'])==1
    assert meta['uncertainty_rms'] is None


def test_projected_fine_lifts_actual_input_and_retains_space_refinements():
    u=sample_parent(make_parent(99,'train',0),5).float()
    value,meta=generate_reference(u,.02,Physics(.003,.5),Geometry((5,5),(1.,1.)),'galerkin_projected_fine',
        dict(initial_steps=4,max_steps=64,tolerance=1e-8,spatial_factors=[2,4],max_grid=20))
    assert len(meta['spatial_refinements'])==2
    assert 'actual deployed FP32' in meta['initial_lifting']
    assert not meta['continuum_claim'];assert value.dtype==torch.float64


def test_dataset_preserves_physical_and_actual_anchor_labels(tmp_path):
    p=make_parent(7,'train',0)
    protocol=dict(root_seed=7,parents=[p],targets=['fd_same_grid'],grids=[5],train_horizons=[.03],
        reference=dict(initial_steps=4,max_steps=128,tolerance=1e-8,crosscheck=True))
    manifest=prepare_dataset(protocol,tmp_path,'train')
    sample=load_samples(tmp_path)[0]
    assert manifest['requested_queries']==1
    assert sample['u0'].dtype==torch.float32 and sample['target'].dtype==torch.float64
    torch.testing.assert_close(sample['deployment_label'],sample['target']-sample['anchor_fp32'].double(),atol=0,rtol=0)
    torch.testing.assert_close(sample['physical_label'],sample['target']-sample['anchor_fp64'],atol=0,rtol=0)
    arrays=tmp_path/manifest['records'][0]['array_file'];arrays.write_bytes(arrays.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='bytes changed'):load_samples(tmp_path)


def test_scientific_missing_reference_is_kept_in_dataset(tmp_path):
    p=make_parent(7,'train',0)
    protocol=dict(root_seed=7,parents=[p],targets=['fd_same_grid'],grids=[5],train_horizons=[.03],
        reference=dict(initial_steps=8,max_steps=8,tolerance=1e-10))
    m=prepare_dataset(protocol,tmp_path,'train')
    assert m['accepted_references']==0 and m['unresolved_references']==1
    assert not load_samples(tmp_path)[0]['reference_accepted']

@pytest.mark.parametrize('n',[5,6])
def test_nonconstant_galerkin_independent_solver_and_nyquist(n):
    axis=torch.arange(n,dtype=torch.float64)/n
    u=(.4+.12*torch.cos(2*torch.pi*axis)+.03*torch.cos(2*torch.pi*(n//2)*axis))[None,None].float()
    value,meta=generate_reference(u,.04,Physics(.003,3.),Geometry((n,),(1.,)),'galerkin_same_grid',
        dict(initial_steps=4,max_steps=256,tolerance=2e-9,max_seconds_per_query=5))
    assert meta['accepted'],meta
    assert meta['cross_solver']['difference']['maximum']<2e-9


def test_extra_source_supervision_is_separate_costed_and_centered(tmp_path):
    p=make_parent(7,'train',0)
    protocol=dict(root_seed=7,parents=[p],targets=['fd_same_grid'],grids=[5],train_horizons=[.03],source_supervision=True,
        reference=dict(initial_steps=4,max_steps=128,tolerance=1e-8,crosscheck=True))
    prepare_dataset(protocol,tmp_path,'train')
    sample=load_samples(tmp_path)[0]
    source=sample['source_targets']
    assert source.dtype==torch.float64 and source.shape==(3,1,1,5,5)
    torch.testing.assert_close(source.mean((-2,-1)),torch.zeros_like(source.mean((-2,-1))),atol=1e-14,rtol=0)
    meta=sample['record']['source_supervision']
    assert meta['accepted'] and meta['elapsed_seconds']>0 and meta['extra_supervision']
    assert meta['anchor']=='df' and meta['reaction_substeps']==8
    assert sample['record']['elapsed_seconds']>=sample['record']['endpoint_preparation_seconds']+meta['elapsed_seconds']


def test_grid_relative_stress_is_distinct_same_parent_bounded_descendant(tmp_path):
    from tdn.analysis.response.fields import grid_relative_parent
    p=make_parent(92,'development_test',5,regime='triads')
    a=grid_relative_parent(p,16);b=grid_relative_parent(p,32)
    assert a['parent_id']==b['parent_id']==p['parent_id']
    assert a['terms']!=b['terms'] and a['descriptive_only']
    for specimen,n in [(a,16),(b,32)]:
        u=sample_parent(specimen,n)
        assert 0<=u.min()<=u.max()<=1
        assert all(abs(k)<n//2 for term in specimen['terms'] for k in term['mode'])
    protocol=dict(root_seed=92,parents=[p],targets=['fd_same_grid'],grids=[16],development_test_horizons=[.001],
        phase_variants=['original','phase_twin'],reference=dict(initial_steps=4,max_steps=128,tolerance=1e-7))
    primary=prepare_dataset(protocol,tmp_path/'base','development_test')
    stress=prepare_dataset(protocol,tmp_path/'stress','development_test',unit=dict(panel='grid_relative_stress',descriptive_only=True))
    assert stress['requested_queries']==2
    assert {r['query_id'] for r in primary['records']}.isdisjoint(r['query_id'] for r in stress['records'])
    assert {r['parent_id'] for r in stress['records']}=={p['parent_id']}
    assert all(r['panel']=='grid_relative_stress' and r['descriptive_only'] for r in stress['records'])


def test_physical_interventions_bind_query_menu_holdouts_and_reference_budget(tmp_path):
    p=make_parent(19,'development_test',0)
    protocol=dict(root_seed=19,parents=[p],targets=['fd_same_grid'],grids=[4],development_test_horizons=[.001],
        phase_variants=['original'],reference=dict(initial_steps=4,max_steps=128,tolerance=1e-7,max_seconds_per_query=120))
    unit=dict(panel='physical_interventions',descriptive_only=True,
        intervention_variants=['original','reaction_interpolation','reaction_extrapolation'],reference_max_seconds_per_query=2.)
    manifest=prepare_dataset(protocol,tmp_path,'development_test',unit=unit)
    assert manifest['requested_queries']==3
    assert manifest['identity']['reference_options']['max_seconds_per_query']==2.
    assert len({r['query_id'] for r in manifest['records']})==3
    assert {r['parent_id'] for r in manifest['records']}=={p['parent_id']}
    for r in manifest['records']:
        assert r['intervention_metadata']['variant']==r['intervention_variant']
        assert r['descriptive_only'] and r['reference']['accepted']
