"""Guard reporting semantics: smoothing is display only and provenance survives."""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tdn.analysis.adjacent import resolution_plotting as plotting
from tdn.analysis.adjacent import resolution_report as report


def put(root, name, value):
    root.mkdir(parents=True, exist_ok=True)
    content='\n'.join(json.dumps(row) for row in value)+'\n' if name.endswith('.jsonl') else json.dumps(value)
    (root/name).write_text(content)


def context(root, **kwargs):
    values=dict(path=root/'report',stage='report',device='cpu',prerequisites={},stage_failures={},
        protocol={'profile':'resolution-smoke','units':{'train':{'kind':'resolution_train','grid':64,'track':'discrete'},
            'evaluate':{'kind':'resolution_evaluate','grid':64,'track':'discrete'},
            'aggregate':{'kind':'resolution_aggregate'},'report':{'kind':'report'}},
            'resolution':{'grids':[64,128],'tracks':['discrete','continuum']}},budget=SimpleNamespace(check=lambda:None))
    values.update(kwargs)
    return SimpleNamespace(**values)


def test_pchip_retains_observations_and_explicit_missing_interval():
    curve=plotting.smooth_curve([0,1,2,3,4,5],[2.,1.,.6,None,.4,.3])
    assert len(curve['segments'])==2 and len(curve['gaps'])==1
    assert curve['segments'][0]['method'].startswith('PCHIP')
    assert max(curve['segments'][0]['x'])==2 and min(curve['segments'][1]['x'])==4
    for segment in curve['segments']:
        for row in segment['observed']:
            i=segment['x'].index(row['x'])
            assert segment['y'][i]==pytest.approx(row['y'])


def test_absent_expected_updates_break_curve_and_duplicate_x_rejected():
    curve=plotting.smooth_curve([0,1,4,5],[1.,.8,.7,.6],expected_step=1)
    assert len(curve['segments'])==2
    with pytest.raises(ValueError,match='Duplicate'):
        plotting.smooth_curve([1,1],[1,2])


def test_observed_envelope_never_crosses_or_silently_imputes_values():
    rng=np.random.default_rng(38)
    for _ in range(20):
        median=rng.uniform(.1,2,6);low=median-rng.uniform(.01,.1,6);high=median+rng.uniform(.01,.1,6)
        result=plotting.smooth_curve(list(range(6)),median.tolist(),lower=low.tolist(),upper=high.tolist())
        seg=result['segments'][0]
        assert np.all(np.asarray(seg['lower'])<=seg['y']) and np.all(np.asarray(seg['y'])<=seg['upper'])
        assert [row['y'] for row in seg['observed']]==median.tolist()
    assert 'not confidence intervals' in result['meaning']


def test_nonpositive_log_values_leave_visible_gaps():
    result=plotting.smooth_curve([0,1,2],[1.,0.,.1],positive=True)
    assert len(result['segments'])==2
    assert result['gaps'][0]['reason']=='missing/nonpositive observation'


def test_dense_windows_preserve_all_raw_and_empty_windows():
    raw=[{'update':i,'loss':1.+.4*np.sin(i)} for i in range(100) if not 40<=i<70]
    windows=plotting.observed_windows(raw,'update','loss',maximum_windows=20)
    assert sum(r['observation_count'] for r in windows)==len(raw)
    assert any(r['median'] is None for r in windows)
    assert sum(len(r['raw']) for r in windows)==len(raw)


def test_spatial_smoothing_keeps_same_array_coordinates_and_limits():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    field=np.arange(12,dtype=np.float64).reshape(3,4);original=field.copy()
    fig,axes=plt.subplots(1,2)
    raw,rawmeta=plotting.spatial_image(axes[0],field,domain=(2.,3.),vmin=0,vmax=11)
    smooth,smoothmeta=plotting.spatial_image(axes[1],field,display='bicubic',domain=(2.,3.),vmin=0,vmax=11)
    assert np.array_equal(field,original) and np.array_equal(raw.get_array(),smooth.get_array())
    assert rawmeta['array_sha256']==smoothmeta['array_sha256']==hashlib.sha256(field.tobytes()).hexdigest()
    assert raw.get_clim()==smooth.get_clim() and raw.get_extent()==smooth.get_extent()
    assert raw.get_extent()==[-.25,1.75,-.5,2.5]
    assert 'display only' in smoothmeta['label'] and 'overshoot' in smoothmeta['warning']
    plt.close(fig)


def test_error_aliases_do_not_relabel_physical_roughness():
    view=report.view({'metrics':{'rms':.05,'maximum':.8,'variance':.1},'error_max':.002})
    assert view['error_rms'] is None and view['error_max']==.002
    assert view['rms']==.05 and view['maximum']==.8


def test_canonical_sources_avoid_summary_freeze_and_aggregate_double_count(tmp_path):
    train=tmp_path/'train';freeze=tmp_path/'freeze';evaluate=tmp_path/'evaluate';aggregate=tmp_path/'aggregate'
    model={'model_id':'m','family':'channel_neural','selection_status':'SELECTED_INITIALIZATION'}
    endpoint={'model_id':'m','family':'channel_neural','error_rms':.01}
    for folder in (train,freeze):put(folder,'catalog.json',[model])
    for folder in (evaluate,aggregate):put(folder,'evaluation-rows.json',[endpoint]);put(folder,'evaluation-summary.json',{'rows':[endpoint]})
    ctx=context(tmp_path,prerequisites={'train':train,'freeze':freeze,'evaluate':evaluate,'aggregate':aggregate})
    ctx.protocol['units']['freeze']={'kind':'resolution_freeze'}
    data=report.collect(ctx)
    assert len(data['catalog'])==1 and len(data['evaluation'])==1 and len(data['partial_evaluation'])==1
    assert data['evaluation'][0]['_source']['unit']=='aggregate'
    assert data['evaluation_inventory']=='verified aggregate'
    assert len([a for a in data['auxiliary'] if a['path'].endswith('evaluation-summary.json')])==2
    assert data['catalog'][0]['_source']['sha256']==hashlib.sha256((train/'catalog.json').read_bytes()).hexdigest()


def test_absent_aggregate_keeps_verified_shards_and_failure_denominators(tmp_path):
    root=tmp_path/'evaluate';put(root,'evaluation-rows.json',[{'error_rms':.2}])
    data=report.collect(context(tmp_path,prerequisites={'evaluate':root},stage_failures={'train':{'status':'FAILED','error':'timeout','elapsed_seconds':4}}))
    assert len(data['evaluation'])==1 and 'absent' in data['evaluation_inventory']
    assert report.view(data['evaluation'][0])['N']==64
    status={r['unit']:r for r in data['stage_status']}
    assert status['train']['computational_status']=='FAILED' and status['aggregate']['scientific_outcome']=='NA'
    assert status['train']['elapsed_seconds']==4 and all(r['monetary_cost'] is None for r in status.values())


def test_representative_spatial_selection_not_best_error_and_safe_source(tmp_path):
    root=tmp_path/'evaluate';root.mkdir()
    np.savez(root/'fields.npz',u=np.arange(64).reshape(8,8),error=np.ones((8,8)))
    rows=[{'parent_id':name,'family':'channel_neural','error_rms':error,'array_file':'fields.npz','array_keys':{'input':'u','residual':'error'}}
          for name,error in [('p1',.001),('p0',2.)]]
    put(root,'evaluation-rows.json',rows)
    data=report.collect(context(tmp_path,prerequisites={'evaluate':root}))
    assert data['patterns'][0]['parent_id']=='p0'
    assert data['patterns'][0]['arrays']['input'][0][1]==1
    assert data['patterns'][0]['_source']['sha256']==hashlib.sha256((root/'fields.npz').read_bytes()).hexdigest()
    rows[0]['array_file']='../outside.npz';rows[0]['parent_id']='before'
    put(root,'evaluation-rows.json',rows)
    with pytest.raises(ValueError,match='escapes'):report.collect(context(tmp_path,prerequisites={'evaluate':root}))


def test_sources_reject_nonfinite_and_symlink(tmp_path):
    root=tmp_path/'train';root.mkdir();(root/'catalog.json').write_text('[{"x":NaN}]')
    with pytest.raises(ValueError,match='Nonfinite'):report.collect(context(tmp_path,prerequisites={'train':root}))
    (root/'catalog.json').unlink();(root/'catalog.json').symlink_to(tmp_path/'unknown')
    with pytest.raises(ValueError,match='bounded'):report.collect(context(tmp_path,prerequisites={'train':root}))


def test_minimal_render_is_source_bound_and_does_not_change_any_metrics(tmp_path):
    data=report.collect(context(tmp_path));original=copy.deepcopy(data)
    output=tmp_path/'figures';manifest=report.build_figures(data,output,_test_dpi=35,_test_page_limit=2)
    assert data==original and manifest['metrics_changed_by_smoothing'] is False
    assert len(manifest['panels'])==2 and manifest['test_preview'] is True
    assert (output/'resolution-atlas.pdf').stat().st_size>1000
    for name,sha in manifest['artifacts'].items():assert hashlib.sha256((output/name).read_bytes()).hexdigest()==sha
    with gzip.open(output/'chart-data.json.gz','rt') as stream:assert json.load(stream)==data
    assert 'not confidence intervals' in (output/'index.html').read_text()
    assert all(p['reading_guide'] and p['scope'] for p in manifest['panels'])


def test_styles_and_response_extraction_keep_roles_and_gains_separate():
    assert report.style('channel_neural')['marker']=='^'
    assert report.style('fno_small')['marker']=='o'
    assert report.style('quad2_fixed')['marker']=='s'
    records=[{'family':'channel_neural','response_probes':[{'report':{'state_dict':{'weight':[9,8]},'response':{'gain':[1.1,.9,1.], 'features':[.4,.3,.2]}}}]}]
    rows=report._response_rows(records)
    assert [r['gain'] for r in rows]==[1.1,.9,1.]


def test_real_diagnostic_singleton_axes_and_aggregate_spatial_origin(tmp_path):
    shard=tmp_path/'evaluate';aggregate=tmp_path/'aggregate';shard.mkdir()
    field=np.arange(64,dtype=np.float64).reshape(1,1,8,8)
    np.savez(shard/'fields.npz',initial=field,error=field*.01,trajectory=np.zeros((3,1,8,8)))
    sha=hashlib.sha256((shard/'fields.npz').read_bytes()).hexdigest()
    row={'parent_id':'p0','family':'channel_neural','grid':64,'track':'discrete','array_file':'fields.npz',
         'array_keys':{'initial':'initial','residual':'error','trajectory':'trajectory'},'array_sha256':sha}
    put(shard,'evaluation-rows.json',[row]);put(aggregate,'evaluation-rows.json',[row])
    data=report.collect(context(tmp_path,prerequisites={'evaluate':shard,'aggregate':aggregate}))
    pattern=data['patterns'][0]
    assert pattern['_source']['unit']=='evaluate' and pattern['row_source']['unit']=='evaluate'
    assert np.asarray(pattern['arrays']['initial']).shape==(8,8)
    assert pattern['array_shapes']['initial']=={'original_shape':[1,1,8,8],'display_shape':[8,8]}
    assert 'trajectory' not in pattern['arrays']
    assert len(data['evaluation'])==1 and len(data['partial_evaluation'])==1


def test_spatial_observation_document_digest_and_response_model_context(tmp_path):
    shard=tmp_path/'evaluate';train=tmp_path/'train';shard.mkdir()
    np.savez(shard/'spatial.npz',u=np.ones((1,1,8,8)))
    sha=hashlib.sha256((shard/'spatial.npz').read_bytes()).hexdigest()
    put(shard,'spatial-observations.json',{'array_sha256':sha,'source_parent':{'domain':[2.,3.],'split':'evaluation'},
        'rows':[{'family':'channel_neural','array_file':'spatial.npz','array_keys':{'initial':'u'}}]})
    put(shard,'parameter-responses.json',[{'model_id':'m','response':{'gain':[[1.,1.1,1.2]]}}])
    put(train,'catalog.json',[{'model_id':'m','family':'channel_neural','role':'Ours','track':'discrete'}])
    data=report.collect(context(tmp_path,prerequisites={'evaluate':shard,'train':train}))
    assert data['patterns'][0]['domain']==[2.,3.]
    assert data['patterns'][0]['_source']['sha256']==sha
    assert data['responses'][0]['family']=='channel_neural'
    assert [r['gain'] for r in report._response_rows(data['responses'])]==[1.,1.1,1.2]


def test_actual_diagnostic_aliases_preserve_timestep_and_refinement_meaning():
    row=report.view({'category':'endpoint','h':.12,'T':.12,'steps':4,'reference_uncertainty_rms':1e-8})
    assert row['h']==.03 and row['uncertainty']==1e-8
    row=report.view({'category':'spatial_refinement','fine_grids':[128,256],'refinement_difference_rms':2e-8,'error_rms':.001})
    assert row['spatial_difference_rms']==2e-8 and row['error_rms']==.001 and row['refinement_grid']==256


def test_complete_bounded_empty_atlas_has_explicit_missing_evidence(tmp_path):
    data=report.collect(context(tmp_path));original=copy.deepcopy(data)
    manifest=report.build_figures(data,tmp_path/'complete',_test_dpi=18)
    assert len(manifest['panels'])==28 and data==original
    assert all(panel['reading_guide'] for panel in manifest['panels'])


def test_descriptive_summaries_do_not_imply_computed_statistical_intervals(tmp_path):
    source=tmp_path/'aggregate'
    put(source,'descriptive-comparisons.json',[{'grid':64,'track':'discrete','model_id':'m',
        'median_error_rms':.001,'independent_parent_denominator':12,'scientific_verdict':'NA'}])
    data=report.collect(context(tmp_path,prerequisites={'aggregate':source}))
    assert len(data['claims'])==1 and data['statistical_intervals']['status']=='NA_NOT_COMPUTED'
    assert 'no parent-cluster intervals' in data['interval_estimand']
    assert 'scientific_verdict' in data['claims'][0] and data['claims'][0]['scientific_verdict']=='NA'
