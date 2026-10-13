"""Screening retains rejected candidate/query denominators and physical scope."""
import copy
import json

import pytest
import torch

from tdn.analysis.response.fields import make_parent
from tdn.analysis.response.references import prepare_dataset, load_samples
from tdn.analysis.response.screening import screen_candidates, run_screen


def _protocol():
    parents=[];candidates={'fd_same_grid':{'train':{},'validation':{}}}
    for split in ('train','validation'):
        pair=[make_parent(98200001,split,index) for index in (0,8)]
        parents.extend(pair);candidates['fd_same_grid'][split]['low_mean']=[p['parent_id'] for p in pair]
    return dict(parents=parents,targets=['fd_same_grid'],dimension=2,train_grid=4,
        train_horizons=[.03,.06],validation_horizons=[.03,.06],phase_variants=['original','phase_twin'],
        screening=dict(candidate_parents=candidates,required_regimes=['low_mean'],minimum_floor=1e-7))


def _record(parent,horizon,phase,*,accepted=True,defect=.01,quantization=1e-7):
    norm=lambda value:dict(rms=value,maximum=value)
    return dict(parent_id=parent['parent_id'],split=parent['split'],tiny_regime='low_mean',
        target_kind='fd_same_grid',grid=[4,4],horizon=horizon,phase_variant=phase,
        intervention_variant='original',query_id=f"{parent['parent_id']}-{horizon}-{phase}",
        reference=dict(accepted=accepted,status='ACCEPTED' if accepted else 'UNRESOLVED_REFERENCE',
            uncertainty_rms=1e-10,uncertainty_maximum=1e-10),
        precision=dict(deployment_defect=norm(defect),physical_defect=norm(defect*.9),
            teacher_output_rounding=norm(quantization),assembly_rounding_floor=norm(quantization),
            anchor_difference=norm(1e-8),state_scale=1.))


def _manifest(protocol):
    return dict(records=[_record(p,h,v) for p in protocol['parents']
        for h in [.03,.06] for v in ['original','phase_twin']])


def test_first_eligible_candidate_is_selected_but_every_candidate_is_evaluated():
    p=_protocol();manifest=_manifest(p)
    rejected=p['parents'][0]['parent_id']
    for r in manifest['records']:
        if r['parent_id']==rejected:r['reference']['accepted']=False
    result=screen_candidates(p,[manifest]);selected=result['selected_parents']['fd_same_grid']
    assert result['eligible'] and selected['train']['low_mean']==p['parents'][1]['parent_id']
    assert selected['validation']['low_mean']==p['parents'][2]['parent_id']
    assert len(result['candidates'])==4
    assert [c['selection_status'] for c in result['candidates']]==[
        'INELIGIBLE','SELECTED','SELECTED','ELIGIBLE_NOT_SELECTED']
    assert result['denominators']['requested_candidate_queries']==16
    assert result['denominators']['accepted_reference_queries']==12
    assert result['denominators']['unresolved_reference_queries']==4
    assert all(q['unresolved_reasons']==['UNRESOLVED_REFERENCE'] for q in result['candidates'][0]['queries'])


def test_one_missing_query_prevents_candidate_selection_and_preserves_requested_slot():
    p=_protocol();manifest=_manifest(p);missing=manifest['records'].pop(0)
    result=screen_candidates(p,[manifest])
    candidate=result['candidates'][0]
    assert candidate['requested_queries']==4 and candidate['recorded_queries']==3
    assert candidate['queries'][0]['query_id'] is None
    assert 'MISSING_QUERY_RECORD' in candidate['unresolved_reasons']
    assert result['denominators']['requested_candidate_queries']==16
    assert result['denominators']['recorded_candidate_queries']==15
    assert result['denominators']['missing_query_records']==1
    assert result['selected_parents']['fd_same_grid']['train']['low_mean']!=missing['parent_id']


def test_no_eligible_candidate_remains_explicitly_unresolved():
    p=_protocol();manifest=_manifest(p)
    for r in manifest['records']:
        if r['split']=='train':r['precision']['deployment_defect']['rms']=1e-8
    result=screen_candidates(p,[manifest])
    assert not result['eligible'] and result['status']=='UNRESOLVED_SCREEN'
    assert result['selected_parents']['fd_same_grid']['train']=={}
    assert result['target_groups']['fd_same_grid']['unresolved_reasons']==['train/low_mean:NO_ELIGIBLE_CANDIDATE']
    assert len(result['candidates'])==4 and result['denominators']['requested_candidate_queries']==16


def test_duplicate_and_unexpected_records_are_retained_and_cannot_pass():
    p=_protocol();manifest=_manifest(p)
    manifest['records'].append(copy.deepcopy(manifest['records'][0]))
    extra=copy.deepcopy(manifest['records'][0]);extra['horizon']=.09;extra['query_id']='undeclared'
    manifest['records'].append(extra)
    result=screen_candidates(p,[manifest]);candidate=result['candidates'][0]
    assert not candidate['eligible']
    assert result['denominators']['duplicate_query_records']==1
    assert result['denominators']['unexpected_query_records']==1
    assert len(candidate['queries'][0]['duplicate_records'])==2
    assert candidate['unexpected_queries'][0]['query_id']=='undeclared'
    assert result['denominators']['recorded_candidate_queries']==18


def test_headroom_uses_uncertainty_margin_and_retains_galerkin_benchmark_qualification():
    p=_protocol();manifest=_manifest(p)
    for r in manifest['records']:
        r['precision']['deployment_defect']['rms']=15e-7
    result=screen_candidates(p,[manifest]);query=result['candidates'][0]['queries'][0]
    assert not result['eligible'] and query['nearest_fp32_mse_reduction_benchmark']>100
    assert query['rms_headroom_ratio']<20
    assert 'INSUFFICIENT_CONSERVATIVE_RMS_HEADROOM' in query['unresolved_reasons']
    p['targets']=['galerkin_same_grid']
    p['screening']['candidate_parents']['galerkin_same_grid']=p['screening']['candidate_parents'].pop('fd_same_grid')
    for r in manifest['records']:
        r['target_kind']='galerkin_same_grid'
        r['precision']['physical_norm_components']={k:r['precision'][k] for k in
            ('deployment_defect','physical_defect','teacher_output_rounding','assembly_rounding_floor')}
    query=screen_candidates(p,[manifest])['candidates'][0]['queries'][0]
    assert 'not a rigorous global physical-L2 optimum' in query['output_quantization_scope']


def test_complete_declared_regime_scope_is_required():
    p=_protocol();p['screening'].pop('required_regimes')
    with pytest.raises(ValueError,match='complete declared regime'):screen_candidates(p,[_manifest(p)])


def test_floor_uses_training_metadata_and_scope_mismatch_cannot_pass():
    p=_protocol();manifest=_manifest(p)
    for r in manifest['records']:
        if r['split']=='validation':r['precision']['anchor_difference']['maximum']=.1
    result=screen_candidates(p,[manifest])
    assert result['target_groups']['fd_same_grid']['normalization_floor']==1e-7
    # Validation diagnostics must not determine the frozen training scale.
    assert result['target_groups']['fd_same_grid']['floor_components']['anchor_roundoff']==1e-8
    manifest['records'][0]['tiny_regime']='high_mean'
    result=screen_candidates(p,[manifest])
    assert 'QUERY_REGIME_SCOPE_MISMATCH' in result['candidates'][0]['unresolved_reasons']
    assert result['selected_parents']['fd_same_grid']['train']['low_mean']==p['parents'][1]['parent_id']


def test_real_preparation_records_the_actual_fp32_representability_benchmark(tmp_path):
    p=make_parent(7,'train',0)
    protocol=dict(parents=[p],targets=['fd_same_grid'],grids=[4],train_horizons=[.03],
        reference=dict(initial_steps=4,max_steps=128,tolerance=1e-8,crosscheck=True))
    manifest=prepare_dataset(protocol,tmp_path,'train');sample=load_samples(tmp_path)[0]
    rounding=sample['target'].float().double()-sample['target']
    precision=manifest['records'][0]['precision']
    assert precision['physical_norm_components']['teacher_output_rounding']['rms']==float(rounding.square().mean().sqrt())
    assert precision['physical_norm_components']['deployment_defect']['rms']==float(sample['deployment_label'].square().mean().sqrt())
    assert precision['physical_norm_components']['physical_defect']['rms']==float(sample['physical_label'].square().mean().sqrt())
    assert 'exact nodal representability optimum for FD' in precision['output_quantization_scope']
    # Casting to nearest FP32 is no worse at any node than the neighboring
    # representable float. This verifies the actual output representation,
    # independently of the screen's metadata arithmetic.
    nearest=sample['target'].float();above=torch.nextafter(nearest,torch.full_like(nearest,float('inf')))
    below=torch.nextafter(nearest,torch.full_like(nearest,float('-inf')))
    assert bool((rounding.abs()<=(above.double()-sample['target']).abs()).all())
    assert bool((rounding.abs()<=(below.double()-sample['target']).abs()).all())


def test_run_screen_writes_portable_json_without_loading_numeric_arrays(tmp_path):
    p=_protocol();data=tmp_path/'reference';data.mkdir();(data/'dataset.json').write_text(json.dumps(_manifest(p)))
    output=tmp_path/'screen';output.mkdir()
    class Context:
        protocol=p;unit=dict(data_stages=['reference']);path=output;prerequisites=dict(reference=data);budget=None
    summary=run_screen(Context())
    assert summary['eligible']
    saved=json.loads((output/'screening.json').read_text())
    assert saved['policy']['data_access'].startswith('JSON metadata only')
    assert saved['denominators']['selected_requested_queries']==8
