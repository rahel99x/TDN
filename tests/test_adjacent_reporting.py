"""Scientific report integrity, not screenshot or implementation-mirror tests."""
from __future__ import annotations
import copy
import csv
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from tdn.analysis.adjacent import report


def put(root, name, value):
    root.mkdir(parents=True, exist_ok=True)
    raw = '\n'.join(json.dumps(r) for r in value) + '\n' if name.endswith('.jsonl') else json.dumps(value)
    (root / name).write_text(raw)


def context(root, **kw):
    values = dict(path=root/'report', stage='report', device='cpu', prerequisites={}, stage_failures={},
        protocol={'profile':'smoke','units':{k:{'seconds':10} for k in ('D01','D05','pilot','evaluate','report')}},
        budget=SimpleNamespace(check=lambda:None))
    values.update(kw); return SimpleNamespace(**values)


def test_collect_preserves_negative_evidence_and_exact_source(tmp_path):
    source = tmp_path/'D01'
    row = {'diagnostic':'D01','method':'bounded_scalar_oracle','error_rms':.1,'verdict':'BAD'}
    put(source,'diagnostic_rows.json',[row])
    put(source,'rows.jsonl',[{'experiment_id':'case','checks':[{'category':'utility','verdict':'BAD'}]}])
    put(source,'summary.json',{'status':'COMPLETED','elapsed_seconds':2,'scientific_outcome':'INCONCLUSIVE'})
    data = report.collect(context(tmp_path, prerequisites={'D01':source}, stage_failures={'pilot':{'status':'FAILED','error':'timeout'}}))
    record = data['diagnostics'][0]
    assert record['verdict'] == 'BAD'
    assert record['_source']['sha256'] == hashlib.sha256((source/'diagnostic_rows.json').read_bytes()).hexdigest()
    assert record['_source']['pointer'] == '/0'
    assert data['checks'][0]['_source']['pointer'] == '/0/checks/0'
    status = {r['unit']:r for r in data['stage_status']}
    assert status['D01']['scientific_outcome'] == 'INCONCLUSIVE'
    assert status['pilot']['status'] == 'FAILED' and status['evaluate']['status'] == 'NA'
    assert all(r['monetary_cost'] is None and r['energy_joules'] is None for r in status.values())


def test_all_named_tables_are_preserved_with_pointer(tmp_path):
    root = tmp_path/'D01'
    put(root,'bundle.json',{'rows':[{'method':'one'}],'other':[{'method':'two'}], 'metadata':'scope'})
    data = report.collect(context(tmp_path, prerequisites={'D01':root}))
    assert {r['method'] for r in data['diagnostics']} == {'one','two'}
    assert {r['_source']['pointer'] for r in data['diagnostics']} == {'/rows/0','/other/0'}


def test_roles_and_max_error_never_confuse_physical_maximum():
    assert report.role('channels_neural') == 'Ours'
    assert report.style('channels_neural')['marker'] == '^'
    assert report.style('fno_standard')['marker'] == 'o'
    assert report.style('fixed_recombination')['marker'] == 's'
    assert report.style('channel_oracle')['marker'] == 's'
    assert report.role('anything', {'method_role':'Analytic control'}) == 'Analytic control'
    original = {'maximum':.85,'error_max':.01,'error_rms':.005,'coefficient_reference':{'real':-.2,'imag':.1}}
    view = report.flatten(original)
    assert view['max_error'] == .01 and view['maximum'] == .85 and view['target_real'] == -.2
    assert 'target_real' not in original
    assert report.style('channels_neural')['linewidth'] < 1


def test_range_plot_is_continuous_but_does_not_bridge_missing_windows():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    rows = [{'family':'channels_neural','track':'discrete','phase':'train','train_count':8,
             'model_id':'m','seed':1,'update':i,'loss':1 if i%2 else -1}
            for i in range(40) if not 15 <= i < 25]
    ranges = report._learning_ranges(rows,'update','loss',max_windows=8)
    assert any(row['observation_count']==0 for row in ranges)
    fig, ax = plt.subplots(); report.range_plot(ax,ranges)
    bands = [c for c in ax.collections if isinstance(c,PolyCollection)]
    assert len(bands)==2 and all(b.get_alpha()==.10 for b in bands)
    boundaries = [line for line in ax.lines if len(line.get_xdata())>2]
    assert len(boundaries)==4
    assert all(all(b>a for a,b in zip(line.get_xdata(),line.get_xdata()[1:])) for line in boundaries)
    assert min(min(line.get_ydata()) for line in boundaries)==-1
    plt.close(fig)


def test_tampered_tower_artifacts_keep_raw_evidence_but_revoke_score(tmp_path):
    from tdn.adjacent_reporting import publish_outputs, _digest
    from tdn.reporting import begin_report
    source = tmp_path/'D01'; source.mkdir()
    row = {'experiment_id':'x','assessment':{'verdict':'GOOD','score_1_100':100},'metrics':{'error':.1}}
    put(source,'rows.jsonl',[row])
    manifest = {'schema':'tdn.adjacent-science/v1','artifacts':{'rows.jsonl':hashlib.sha256((source/'rows.jsonl').read_bytes()).hexdigest()}}
    put(source,'science_manifest.json',manifest); (source/'COMPLETED').write_text(_digest(manifest)+'\n')
    tower = Path(begin_report(source,name='TDN/adjacent/D01',script='scripts/adjacent.py',parameters={'benchmark_suite':'adjacent'},report_parent=tmp_path/'tower'))
    first = publish_outputs(tower,[source]); assert first['sealed_source_records']==1
    row['metrics']['error'] = .2; put(source,'rows.jsonl',[row])
    second = publish_outputs(tower,[source]); assert second['unverified_diagnostic_records']==1
    with (tower/second['pages'][0]['path']).open() as stream: display=next(csv.DictReader(stream))
    assert display['raw_verdict']=='GOOD' and display['verdict']=='NA'
    assert json.loads(display['record_json'])['metrics']['error']==.2
    assert display['source_pointer']=='/0'


def test_csv_cells_safe_and_raw_numeric_sign_retained(tmp_path):
    data={name:[] for name in report.TABLES}
    data['diagnostics']=[{'method':'=formula()','metrics':{'signed':-1}}]
    report.write_tables(data,tmp_path/'tables')
    with (tmp_path/'tables/diagnostics.csv').open() as stream: row=next(csv.DictReader(stream))
    assert row['method']=="'=formula()" and json.loads(row['metrics'])['signed']==-1


def test_empty_atlas_displays_na_and_retains_exact_raw_data(tmp_path):
    data = report.collect(context(tmp_path)); original=copy.deepcopy(data)
    output = tmp_path/'figures'; manifest=report.build_figures(data,output)
    assert data==original
    assert len(manifest['panels'])>=30
    assert all(p['scope'] and p['reading_guide'] for p in manifest['panels'])
    assert (output/'adjacent-atlas.pdf').stat().st_size>1000
    with gzip.open(output/'chart-data.json.gz','rt') as stream: assert json.load(stream)==data
    assert 'not confidence intervals' in (output/'index.html').read_text()
    assert manifest['counts']['diagnostics']==0


def test_sources_reject_symlinks_and_nonfinite_json(tmp_path):
    root=tmp_path/'D01';root.mkdir();target=tmp_path/'target.json';target.write_text('[]')
    (root/'diagnostic_rows.json').symlink_to(target)
    with pytest.raises(ValueError,match='bounded'):report.collect(context(tmp_path,prerequisites={'D01':root}))
    (root/'diagnostic_rows.json').unlink();(root/'diagnostic_rows.json').write_text('[{"value": NaN}]')
    with pytest.raises(ValueError,match='Nonfinite'):report.collect(context(tmp_path,prerequisites={'D01':root}))


def test_known_missing_curve_measurement_leaves_nan_gap():
    import math
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots()
    rows=[{'method':'channel_neural','parent_id':'p','track':'discrete','time':t,'error_rms':v}
          for t,v in ((0.,1.),(1.,None),(2.,.1))]
    report._scatter(ax,rows,'time','rms',connect=True)
    assert len(ax.lines)==1
    assert math.isnan(ax.lines[0].get_ydata()[1])
    plt.close(fig)


def test_original_feature_collision_not_replaced_by_rich_feature_distance():
    view=report.flatten({'feature_differences':{'original':1e-15,'rich':.3},'shared_error_rms':2.,'separate_error_rms':1.})
    assert view['feature_distance']==1e-15
    assert view['shared_response_penalty']==1.


def test_categorical_floors_keep_both_equation_tracks_visible():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots()
    report._categorical(ax,[{'method':'quad2_fixed','track':track,'error_rms':.1} for track in ('discrete','continuum')],'rms')
    labels=ax.get_legend_handles_labels()[1]
    assert any('[discrete]' in label for label in labels)
    assert any('[continuum]' in label for label in labels)
    plt.close(fig)
