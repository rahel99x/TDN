"""Synthetic projection tests; these fixtures are not numerical experiments."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tdn.analysis.resolve.core import Context, check, write_reviews
from tdn.analysis.resolve.protocol import build_protocol, digest
from tdn.reporting import begin_report
from tdn.runtime.metadata import write_json
from tdn.tower_analytics import publish_outputs


def seal(source, protocol, manifest=None):
    stage = source.name
    manifest = manifest or dict(schema='tdn.resolve-science/v1', stage=stage,
        kind=protocol['units'][stage]['kind'], unit_sha256=digest(protocol['units'][stage]),
        source_tree_sha256='a'*64, software={'source_tree_sha256':'a'*64}, device='cpu')
    write_json(source/'protocol.json', protocol)
    manifest['protocol_sha256'] = digest(protocol)
    for name in ('summary.json','execution.json','stage.json'):
        value = json.loads((source/name).read_text())
        value['protocol_sha256'] = digest(protocol)
        write_json(source/name,value)
    exclude = {'science_manifest.json','COMPLETED','execution.json','stage.json','workflow-seal.json'}
    manifest['artifacts'] = {str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source.rglob('*') if p.is_file() and p.name not in exclude}
    write_json(source/'science_manifest.json',manifest)
    (source/'COMPLETED').write_text(digest(manifest)+'\n')
    write_json(source/'workflow-seal.json',dict(schema='tdn.resolve/v1',protocol_sha256=digest(protocol),
        files={name:hashlib.sha256((source/name).read_bytes()).hexdigest()
            for name in ('execution.json','stage.json','protocol.json','science_manifest.json')}))
    return manifest


def fixture(tmp_path, *, stage='fairness-discrete-000', nested_figure=True):
    protocol=build_protocol('smoke')
    source=tmp_path/stage;source.mkdir()
    ctx=Context(protocol,stage,source,{},'cpu',SimpleNamespace(check=lambda:None))
    ctx.record('synthetic/only',['R3'],metrics={'parameters':2,'median_seconds':.002},
        checks=[check('identity',0.,1e-10,category='math'),check('coverage',3,3,'eq',category='gap')])
    write_reviews(source,ctx.rows)
    identity=dict(stage=stage,profile='smoke',protocol_sha256=digest(protocol),device='cpu')
    write_json(source/'summary.json',dict(identity,schema='tdn.resolve/v1',status='COMPLETED',
        source_tree_sha256='a'*64,experiment_count=1,scientific_outcome='DEVELOPMENT_EVIDENCE_ONLY'))
    write_json(source/'execution.json',dict(identity,software={'source_tree_sha256':'a'*64}))
    write_json(source/'stage.json',dict(identity,status='COMPLETED'))
    curves=[dict(model_id='discrete/fno_scaled/n2/seed1',family='fno_scaled',track='discrete',
        seed=1,train_count=2,phase='long_update',trial_id=trial,update=update,
        optimizer_control={'id':trial,'clip_grad_norm':1. if trial=='clipped' else None},
        train_loss=.1/(update+1),validation_loss=.2 if update==0 else None,
        elapsed_seconds=.01+update) for trial in ('clipped','unclipped') for update in (0,1)]
    write_json(source/'learning_curves.json',dict(rows=curves,observed_range_is_confidence_interval=False))
    if nested_figure:
        (source/'figures').mkdir()
        (source/'figures/fixture.pdf').write_bytes(b'%PDF-1.4\n%%EOF\n')
    manifest=seal(source,protocol)
    report=Path(begin_report(source,name='TDN/resolve/fixture',script='scripts/resolve.py',
        parameters={'fixture_only':True},report_parent=tmp_path/'tower'))
    return source,report,protocol,manifest


def pages(report):
    index=json.loads((report/'outputs/frontier-tables.json').read_text())
    pages=[]
    for catalog in index['page_catalogs']:
        pages.extend(json.loads((report/catalog['path']).read_text())['pages'])
    return index,pages


def table(report,name):
    rows=[]
    for page in pages(report)[1]:
        if page['table']==name:
            with (report/page['path']).open(newline='') as stream:rows.extend(csv.DictReader(stream))
    return rows


def test_resolve_seals_dynamic_units_and_complete_learning_trials_are_projected(tmp_path):
    source,report,_,_=fixture(tmp_path)
    before={p:p.read_bytes() for p in source.rglob('*') if p.is_file()}
    result=publish_outputs(report,[source])
    assert result['frontier']['reporting_complete'],result
    index,_=pages(report)
    assert index['expected_rows']==index['published_rows']
    assert index['sources'][0]['program']=='resolve'
    assert index['sources'][0]['scientific_scope']=='DEVELOPMENT_ONLY_NO_CONFIRMATORY_CLAIMS'
    assert table(report,'experiments')[0]['evidence_status']=='VERIFIED_CANONICAL'
    assert table(report,'experiments')[0]['verdict']=='GOOD'
    curves=table(report,'learning_curves')
    assert len(curves)==4 and {r['trial_id'] for r in curves}=={'clipped','unclipped'}
    assert {r['source_record'] for r in curves}=={'/rows/0','/rows/1','/rows/2','/rows/3'}
    lineage=[r for r in table(report,'metric_lineage') if r['observation_kind']=='learning_curve']
    assert len({r['phase'] for r in lineage})==2
    assert all(r['phase'].startswith('resolve/learning/') for r in lineage)
    assert before=={p:p.read_bytes() for p in source.rglob('*') if p.is_file()}
    assert result['frontier']['figure_count']==1
    logs=json.loads((report/'logs.json').read_text())['logs']
    assert any(r['group']=='Resolve analytical figures' and r['path'].endswith('fixture.pdf') for r in logs)


@pytest.mark.parametrize('defect',['unit_hash','source','summary_source','wrapper_schema','confirmation','wrong_family'])
def test_changed_semantic_bindings_never_credit_scores(tmp_path,defect):
    source,report,protocol,manifest=fixture(tmp_path)
    if defect=='unit_hash':manifest['unit_sha256']='b'*64
    elif defect=='source':manifest['software']['source_tree_sha256']='b'*64
    elif defect=='summary_source':
        row=json.loads((source/'summary.json').read_text());row['source_tree_sha256']='b'*64
        write_json(source/'summary.json',row)
    elif defect=='confirmation':protocol['confirmation']['enabled']=True
    elif defect=='wrong_family':manifest['schema']='tdn.frontier-science-manifest/v1'
    seal(source,protocol,manifest)
    if defect=='wrapper_schema':
        row=json.loads((source/'workflow-seal.json').read_text());row['schema']='tdn.frontier/v1'
        write_json(source/'workflow-seal.json',row)
    result=publish_outputs(report,[source])
    assert not result['frontier']['reporting_complete']
    rows=table(report,'experiments')
    assert rows[0]['evidence_status'].startswith('INVALID') and rows[0]['verdict']=='NA'
    assert rows[0]['raw_verdict']=='GOOD'


def test_learning_json_requires_trial_identity_and_cannot_merge_tied_optimizers(tmp_path):
    source,report,protocol,manifest=fixture(tmp_path)
    curves=json.loads((source/'learning_curves.json').read_text())
    curves['rows'][2].pop('trial_id');write_json(source/'learning_curves.json',curves)
    seal(source,protocol,manifest)
    result=publish_outputs(report,[source])
    assert not result['frontier']['reporting_complete']
    assert table(report,'learning_curves')==[]
    assert table(report,'experiments')[0]['evidence_status']=='VERIFIED_CANONICAL'


def test_coordinator_discovers_only_declared_safe_resolve_unit_names(tmp_path):
    from tdn.frontier_reporting import _candidates
    protocol=build_protocol('smoke')
    write_json(tmp_path/'protocol.json',{'scientific_protocol':protocol})
    candidates=set(_candidates(tmp_path))
    assert {tmp_path/name for name in protocol['units']}<=candidates
    protocol['units']['../escape']={}
    write_json(tmp_path/'protocol.json',{'scientific_protocol':protocol})
    candidates=set(_candidates(tmp_path))
    assert tmp_path/'../escape' not in candidates
    assert tmp_path/'fairness-discrete-000' not in candidates


def test_resolve_actual_report_root_layout_registers_png_and_pdf_without_source_edits(tmp_path):
    source,report,protocol,manifest=fixture(tmp_path,stage='report',nested_figure=False)
    (source/'fit-discrete-fit.png').write_bytes(bytes.fromhex('89504e470d0a1a0a'))
    (source/'fit-discrete-fit.pdf').write_bytes(b'%PDF-1.4\n%%EOF\n')
    write_json(source/'figures.json',[{'name':'fit-discrete-fit',
        'png':'fit-discrete-fit.png','pdf':'fit-discrete-fit.pdf','caption':'Synthetic projection fixture'}])
    seal(source,protocol,manifest)
    before={p:p.read_bytes() for p in source.rglob('*') if p.is_file()}
    result=publish_outputs(report,[source]);index,_=pages(report)
    assert result['frontier']['reporting_complete']
    assert result['frontier']['figure_count']==2
    figures=index['figures']
    assert {r['format'] for r in figures}=={'image','pdf'}
    assert all(r['hash_status']=='computed_projection' and r['evidence_status']=='VERIFIED_CANONICAL' for r in figures)
    for row in figures:
        assert row['sha256']==hashlib.sha256((report/row['path']).read_bytes()).hexdigest()
    logs=json.loads((report/'logs.json').read_text())['logs']
    assert {r['path'] for r in figures}<={r['path'] for r in logs if r['group']=='Resolve analytical figures'}
    assert before=={p:p.read_bytes() for p in source.rglob('*') if p.is_file()}


@pytest.mark.parametrize('defect',['changed_bytes','undeclared','unsealed_source','missing_file'])
def test_resolve_report_root_figures_need_verified_content_and_source(tmp_path,defect):
    source,report,protocol,manifest=fixture(tmp_path,stage='report',nested_figure=False)
    panel=source/'stage-costs.png';panel.write_bytes(bytes.fromhex('89504e470d0a1a0a'))
    if defect!='undeclared':seal(source,protocol,manifest)
    if defect=='changed_bytes':panel.write_bytes(b'changed image')
    if defect=='unsealed_source':(source/'COMPLETED').unlink()
    if defect=='missing_file':panel.unlink()
    result=publish_outputs(report,[source]);index,_=pages(report)
    assert result['frontier']['figure_count']==0 and index['figures']==[]
    assert not result['frontier']['reporting_complete']


def test_resolve_report_root_discovery_does_not_expand_to_other_stages(tmp_path):
    source,report,protocol,manifest=fixture(tmp_path,nested_figure=False)
    (source/'incidental.png').write_bytes(bytes.fromhex('89504e470d0a1a0a'))
    seal(source,protocol,manifest)
    result=publish_outputs(report,[source])
    assert result['frontier']['reporting_complete'] and result['frontier']['figure_count']==0
