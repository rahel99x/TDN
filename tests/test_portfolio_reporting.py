"""Reporting preserves evidence, source identity, missingness and visual semantics."""
from __future__ import annotations

import copy
import csv
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tdn.analysis.portfolio import report


def context(tmp_path, **kw):
    defaults=dict(protocol={'profile':'smoke','units':{'audit':{'kind':'audit'},'train-A-000':{'kind':'train'},
        'freeze':{'kind':'freeze'},'confirm-000':{'kind':'confirm'},'aggregate':{'kind':'aggregate'},'report':{'kind':'report'}}},
        stage='report',path=tmp_path/'report',prerequisites={},device='cpu',stage_failures={},
        budget=SimpleNamespace(check=lambda:None))
    defaults.update(kw);return SimpleNamespace(**defaults)


def put(root,name,value):
    root.mkdir(exist_ok=True,parents=True)
    raw='\n'.join(json.dumps(r) for r in value)+'\n' if name.endswith('.jsonl') else json.dumps(value)
    (root/name).write_text(raw)


def test_collection_retains_failures_missing_stages_and_source_pointer(tmp_path):
    base=tmp_path/'audit'
    row={'experiment_id':'negative','metrics':{'value':2},'checks':[{'check_id':'x','verdict':'BAD'}],
         'assessment':{'verdict':'BAD'}}
    put(base,'rows.jsonl',[row]);put(base,'summary.json',{'status':'COMPLETED','elapsed_seconds':1})
    ctx=context(tmp_path,prerequisites={'audit':base},stage_failures={'train-A-000':{'status':'FAILED','error':'timeout'}})
    data=report.collect(ctx)
    assert data['experiments'][0]['assessment']['verdict']=='BAD'
    source=data['experiments'][0]['_source']
    assert source['sha256']==hashlib.sha256((base/'rows.jsonl').read_bytes()).hexdigest()
    assert source['line']==1 and source['pointer']=='/0'
    assert data['checks'][0]['_source']['pointer']=='/0/checks/0'
    statuses={r['unit']:r for r in data['stage_status']}
    assert statuses['train-A-000']['status']=='FAILED'
    assert statuses['train-A-000']['elapsed_seconds'] is None
    assert statuses['aggregate']['status']=='NA'
    assert statuses['aggregate']['monetary_cost'] is None


def test_freeze_and_aggregate_do_not_double_count_partition_or_train_rows(tmp_path):
    roots={k:tmp_path/k for k in ('train-A-000','freeze','confirm-000','aggregate')}
    for k in ('train-A-000','freeze'):put(roots[k],'catalog.json',{'records':[{'model_id':'m','family':'quad2_fixed'}]})
    for k in ('confirm-000','aggregate'):put(roots[k],'endpoint_rows.json',{'rows':[{'family':'quad2_fixed','field_cluster':'p','upper_rms':.1}]})
    data=report.collect(context(tmp_path,prerequisites=roots))
    assert len(data['catalog'])==len(data['endpoints'])==1
    assert not data['partial_endpoints']
    assert data['catalog'][0]['_source']['unit']=='freeze'
    assert data['endpoints'][0]['_source']['unit']=='aggregate'
    assert len(data['sources'])==4
    partial=report.collect(context(tmp_path,prerequisites={'confirm-000':roots['confirm-000']}))
    assert len(partial['partial_endpoints'])==1 and not partial['endpoints']


def test_source_symlink_is_rejected_and_csv_formulas_are_escaped(tmp_path):
    source=tmp_path/'file.json';source.write_text('{}')
    link=tmp_path/'link.json';link.symlink_to(source)
    with pytest.raises(ValueError,match='regular bounded'):report._read_source(link)
    data={k:[] for k in report.TABLES};data['experiments']=[{'experiment_id':'=bad()','metrics':{'negative':-1},'_source':{'pointer':'/0'}}]
    report.write_tables(data,tmp_path/'tables')
    with (tmp_path/'tables/experiments.csv').open() as stream:row=next(csv.DictReader(stream))
    assert row['experiment_id']=="'=bad()"
    assert json.loads(row['metrics'])=={'negative':-1}


def test_roles_distinguish_analytic_rules_from_ours_and_external():
    for name in ('quad2_fixed','quad2_full','quad4_fixed','analytic_quad_cubic'):
        assert report.role(name)=='Analytic control'
        assert report.style(name)['marker']=='s'
    for name in ('quad2_conditioned','quad2_linear','historical_half','band_gain'):
        assert report.role(name)=='Ours' and report.style(name)['marker']=='^'
    for name in ('fno_small','rf','etdrk4'):
        assert report.role(name)=='Theirs' and report.style(name)['marker']=='o'
    assert report.style('quad2_joint')==report.style('quad2_joint')
    assert report.style('quad2_joint')['linewidth']<1


def test_continuous_ranges_preserve_gaps_extrema_and_track_identity():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    curves=[]
    for track in ('discrete','continuum'):
        for x in range(40):
            if 15<=x<25:continue
            curves.append({'family':'quad2_conditioned','track':track,'phase':'tuning','train_count':8,
                           'model_id':'m','seed':1,'update':x,'train_loss':(-1 if x%2 else 1)})
    original=copy.deepcopy(curves)
    ranges=report._learning_ranges(curves,'update','train_loss',max_windows=8)
    assert curves==original
    values=[r for r in ranges if r['track']=='discrete']
    assert sum(r['observation_count'] for r in values)==30
    assert any(r['observation_count']==0 for r in values)
    fig,ax=plt.subplots();report.range_plot(ax,values,'update','train_loss')
    from matplotlib.collections import PolyCollection
    bands=[c for c in ax.collections if isinstance(c,PolyCollection)]
    assert len(bands)==2
    assert all(c.get_alpha()==.10 for c in bands)
    boundaries=[line for line in ax.lines if len(line.get_xdata())>2]
    assert len(boundaries)==4
    assert all(all(b>a for a,b in zip(line.get_xdata(),line.get_xdata()[1:])) for line in boundaries)
    assert min(min(line.get_ydata()) for line in boundaries)==-1
    assert max(max(line.get_ydata()) for line in boundaries)==1
    assert ax.get_yscale()=='linear'
    plt.close(fig)


def test_amortization_requires_positive_margin_and_does_not_invent_money():
    data={'catalog':[{'model_id':'m','total_training_seconds':10}],
          'locked_frontiers':[{'model_id':'m','candidate_seconds':2,'comparator_seconds':1},
                              {'model_id':'m','candidate_seconds':1,'comparator_seconds':3}]}
    result=report.amortization(data)
    assert result[0]['status']=='NA' and result[0]['training_only_break_even_queries'] is None
    assert result[1]['training_only_break_even_queries']==5
    assert all(r['monetary_cost'] is None for r in result)


def test_empty_atlas_explicit_na_and_exact_raw_chart_source(tmp_path):
    data=report.collect(context(tmp_path))
    before=copy.deepcopy(data)
    manifest=report.build_figures(data,tmp_path/'figures')
    assert data==before
    assert len(manifest['panels'])>=25
    assert manifest['counts']['endpoints']==0
    with gzip.open(tmp_path/'figures/chart-data.json.gz','rt') as stream:assert json.load(stream)==data
    assert all(p['reading_guide'] and p['scope'] for p in manifest['panels'])
    assert (tmp_path/'figures/portfolio-atlas.pdf').stat().st_size>1000
    assert 'not confidence intervals' in (tmp_path/'figures/index.html').read_text()


def test_tower_source_bound_projection_retains_tampered_raw_but_revokes_score(tmp_path):
    from tdn.portfolio_reporting import publish_outputs,_digest
    from tdn.reporting import begin_report
    source=tmp_path/'audit';source.mkdir()
    record={'experiment_id':'e','assessment':{'verdict':'GOOD','score_1_100':100},'metrics':{'a':1}}
    put(source,'rows.jsonl',[record])
    manifest={'schema':'tdn.portfolio-science/v1','artifacts':{
        'rows.jsonl':hashlib.sha256((source/'rows.jsonl').read_bytes()).hexdigest()}}
    put(source,'science_manifest.json',manifest);(source/'COMPLETED').write_text(_digest(manifest)+'\n')
    tower=Path(begin_report(source,name='TDN/portfolio/audit',script='scripts/portfolio.py',
                           parameters={'benchmark_suite':'portfolio'},report_parent=tmp_path/'tower'))
    first=publish_outputs(tower,[source]);assert first['sealed_source_records']==1
    page=first['pages'][0]
    with (tower/page['path']).open() as stream:row=next(csv.DictReader(stream))
    assert row['verdict']=='GOOD' and row['source_pointer']=='/0'
    record['metrics']['a']=2;put(source,'rows.jsonl',[record])
    second=publish_outputs(tower,[source]);assert second['unverified_diagnostic_records']==1
    with (tower/second['pages'][0]['path']).open() as stream:row=next(csv.DictReader(stream))
    assert row['raw_verdict']=='GOOD' and row['verdict']=='NA' and row['score_1_100']=='1'
    assert json.loads(row['record_json'])['metrics']['a']==2
    assert len([r for r in json.loads((tower/'logs.json').read_text())['logs'] if r['id']=='portfolio-analytics'])==1


def test_tower_pages_do_not_drop_any_row_and_unknown_money_remains_null(tmp_path):
    from tdn.portfolio_reporting import publish_outputs
    from tdn.reporting import begin_report
    source=tmp_path/'audit';source.mkdir()
    put(source,'rows.jsonl',[{'experiment_id':f'case/{i}','metrics':{'cost':None}} for i in range(130)])
    tower=Path(begin_report(source,name='TDN/portfolio/audit',script='scripts/portfolio.py',
                           parameters={'benchmark_suite':'portfolio'},report_parent=tmp_path/'tower'))
    result=publish_outputs(tower,[source]);assert sum(p['rows'] for p in result['pages'])==130
    assert all(p['rows']<=128 and p['bytes']<=256<<10 for p in result['pages'])
    rows=[]
    for page in result['pages']:
        with (tower/page['path']).open() as stream:rows.extend(csv.DictReader(stream))
    assert {json.loads(r['record_json'])['experiment_id'] for r in rows}=={f'case/{i}' for i in range(130)}
    assert all(json.loads(r['record_json'])['metrics']['cost'] is None for r in rows)


def test_frozen_family_line_styles_are_distinct_and_timing_aliases_keep_raw_values():
    from tdn.analysis.portfolio.models import MODEL_SPECS
    styles=[report.style(name) for name in MODEL_SPECS]
    identities={(s['color'],s['marker'],s['linestyle']) for s in styles}
    assert len(identities)==len(styles)
    row={'family':'quad2_fixed','final_time':.2,'cost_seconds':.003,'timing':{'median_seconds':.003},
         'errors':[{'upper_rms':.1,'upper_max':.2},{'upper_rms':.3,'upper_max':.4}]}
    original=copy.deepcopy(row);view=report.flatten(row)
    assert view['horizon']==.2 and view['median_seconds']==.003 and view['upper_rms']==.3
    assert row==original


def test_primary_claims_preserve_exact_gates_and_trace_each_claim(tmp_path):
    aggregate=tmp_path/'aggregate'
    claims={'accuracy':[{'candidate_family':'quad2_conditioned','control_family':'quad2_fixed','verdict':'NA',
                        'unresolved_pairs':2,'required_ratio':1.1,'denominator':8}],
            'cost':[{'candidate_family':'quad2_conditioned','control_family':'df','verdict':'BAD',
                     'required_speedup':1.2,'candidate_joint_field_coverage':.5}],
            'scientific_success_is_not_job_completion':True,'no_published_fno_superiority_claim':True}
    put(aggregate,'claims.json',claims)
    ctx=context(tmp_path,prerequisites={'aggregate':aggregate},
                stage_failures={'train-A-000':{'status':'FAILED','elapsed_seconds':123.}})
    data=report.collect(ctx)
    assert data['claims_accuracy'][0]['verdict']=='NA'
    assert data['claims_cost'][0]['verdict']=='BAD'
    assert data['claims_accuracy'][0]['_source']['pointer']=='/accuracy/0'
    assert data['claims_cost'][0]['_source']['pointer']=='/cost/0'
    assert data['claims_metadata'][0]['no_published_fno_superiority_claim'] is True
    assert next(r for r in data['stage_status'] if r['unit']=='train-A-000')['elapsed_seconds']==123.
    assert claims['accuracy'][0]=={k:v for k,v in data['claims_accuracy'][0].items() if k!='_source'}


def test_shared_legend_is_external_and_prototype_cost_failure_stays_visible():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax in axes:
        for i in range(23):ax.plot([0,1],[i,i+1],label=f'Ours: model-{i}',linewidth=.5)
        ax.legend()
    report._layout_figure(fig,'Title','Lower is better.','Test scope.')
    assert all(ax.get_legend() is None for ax in axes)
    assert len(fig.legends)==1 and len(fig.legends[0].get_texts())==23
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=fig.bbox
    for text in fig.texts:
        box=text.get_window_extent(renderer)
        assert bounds.x0<=box.x0 and box.x1<=bounds.x1
        assert bounds.y0<=box.y0 and box.y1<=bounds.y1
    legend=fig.legends[0].get_window_extent(renderer)
    assert all(legend.y1<ax.get_window_extent(renderer).y0 for ax in axes)
    plt.close(fig)
    rows=report.prototype_assessments([{'status':'COMPLETED','method':'compact_temporal_encoding','checks':[
        {'category':'math','verdict':'GOOD'},{'category':'gap','verdict':'GOOD'},
        {'category':'utility','verdict':'BAD'}]}])
    assert rows[0]['status']=='COMPLETED'
    assert rows[0]['display_science']=={'math':'GOOD','gap':'GOOD','utility':'BAD'}
