"""Complete, honest frontier graphs and compact unchanged-Tower projections."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tdn.analysis.frontier.core import Context, check, write_reviews
from tdn.analysis.frontier.protocol import build_protocol
from tdn.analysis.frontier import report as graphs
from tdn.reporting import begin_report
from tdn.research.protocol import digest
from tdn.runtime.metadata import write_json
from tdn.tower_analytics import publish_outputs


def fixture(tmp_path, count=3):
    source = tmp_path / 'audit'; source.mkdir()
    protocol = build_protocol('smoke')
    ctx = Context(protocol, 'audit', source, {}, 'cpu', SimpleNamespace(check=lambda: None))
    for index in range(count):
        ctx.record(f'case/{index}', ['G1'], metrics={'parameters': 43, 'median_seconds': .01,
            'nested': {'a/b~c': [1.,2.,3.]}, 'samples': list(range(100))},
            config={'family': 'rank1', 'grid': 8, 'seed': 41},
            checks=[check('identity', 0., 1.e-8, category='math'),
                    check('cost', 2., 1., category='gap'),
                    check('unmeasured-utility', None, 1., category='utility')])
    write_reviews(source, ctx.rows)
    write_json(source / 'protocol.json', protocol)
    write_json(source / 'summary.json', {'schema': protocol['schema'], 'stage': 'audit', 'status': 'COMPLETED',
        'experiment_count': count, 'elapsed_seconds': 1., 'scientific_outcome': 'BOUNDED_EVIDENCE'})
    manifest = {'schema': 'tdn.frontier-science-manifest/v1', 'stage': 'audit',
        'protocol_sha256': digest(protocol), 'source_tree_sha256': 'a'*64,
        'artifacts': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()}}
    write_json(source / 'science_manifest.json', manifest)
    (source / 'COMPLETED').write_text(digest(manifest)+'\n')
    tower = Path(begin_report(source, name='TDN/frontier/audit', script='scripts/frontier.py',
        parameters={'benchmark_suite': 'frontier'}, report_parent=tmp_path / 'tower'))
    return source, tower, protocol, ctx.rows


def pages(tower):
    index = json.loads((tower / 'outputs/frontier-tables.json').read_text())
    result = []
    for catalog in index['page_catalogs']:
        path = tower / catalog['path']
        assert catalog['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
        result.extend(json.loads(path.read_text())['pages'])
    return index, result


def table(tower, name):
    result = []
    for page in pages(tower)[1]:
        if page['table'] == name:
            with (tower / page['path']).open(newline='') as stream:
                result.extend(csv.DictReader(stream))
    return result


def test_complete_nested_values_reduce_page_duplication_without_losing_metrics(tmp_path):
    source, tower, protocol, rows = fixture(tmp_path, 129)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    result = publish_outputs(tower, [source])
    assert result['frontier']['reporting_complete'], result
    assert 'roadmap' not in result and 'agenda' not in result
    index, catalogs = pages(tower)
    assert index['expected_rows'] == index['published_rows']
    assert index['published_rows']['experiments'] == 129
    assert index['published_rows']['checks'] == 387
    assert index['published_rows']['values'] == 3*129
    values = table(tower, 'values')
    nested = next(r for r in values if r['field'] == '/metrics')
    assert json.loads(nested['value'])['nested']['a/b~c'] == [1.,2.,3.]
    assert all(row['evidence_status'] == 'VERIFIED_CANONICAL' for row in table(tower,'experiments'))
    assert all(page['rows'] <= 128 and page['bytes'] <= 262144 for page in catalogs)
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}


def test_tampered_or_missing_seal_cannot_contribute_affirmative_scores(tmp_path):
    source, tower, _, _ = fixture(tmp_path)
    (source/'COMPLETED').unlink()
    result = publish_outputs(tower, [source])
    assert not result['frontier']['reporting_complete']
    assert all(r['verdict']=='NA' and r['score_1_100']=='1' for r in table(tower,'experiments'))
    assert all(r['raw_verdict']=='BAD' for r in table(tower,'experiments'))


def test_nested_graph_groups_keep_final_times_separate():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [{'track':'discrete', 'horizon':.2}, {'track':'discrete','horizon':.8}]
    seen=[]
    fig=graphs._facets(plt,rows,('track','horizon'),lambda ax, vals: seen.append(vals),'fixed time')
    assert len(seen)==2 and all(len({r['horizon'] for r in group})==1 for group in seen)
    plt.close(fig)


def test_gate_missing_coverage_is_na_and_observed_failure_is_not_hidden(tmp_path):
    _,_,protocol,rows=fixture(tmp_path)
    data={'protocol':protocol,'rows':rows}
    summary={r['id']:r for r in graphs.aggregate(data)}
    assert set(summary)=={'G1','G2','G3','G4','G5'}
    assert summary['G1']['verdict']=='BAD'
    assert summary['G3']['verdict']=='NA'
    assert summary['G3']['missing_stages']


def test_complete_graph_export_retains_loss_points_checks_and_na(tmp_path):
    _,_,protocol,rows=fixture(tmp_path)
    curves=[{'model_id':'discrete/rank1/n2/seed1', 'family':'rank1','track':'discrete','phase':'final',
        'seed':1,'train_count':2,'update':i,'train_loss':.1/(i+1), 'validation_loss':.2/(i+1),
        'validation_rms':.02/(i+1), 'elapsed_seconds':i+.1,'examples_seen':i+1,'parameters':43} for i in range(3)]
    data={'schema':'tdn.frontier-chart-data/v1','profile':'smoke','protocol':protocol,'rows':rows,
        'learning_curves':curves,'catalog':[],'confirmation':[],'comparisons':[],'scaling':[], 'policy':[],
        'costs':[],'sources':[],'stage_status':{'audit':{'status':'VERIFIED'}}}
    path=tmp_path/'figures'
    manifest=graphs.build_figures(data,path)
    assert len(manifest['panels'])>=27
    assert manifest['experiment_count']==3 and manifest['check_count']==9
    assert manifest['loss_observations']==3 and manifest['downsampled'] is False
    assert manifest['unavailable_panels_are_na']
    with gzip.open(path/'chart-data.json.gz','rt') as stream:
        restored=json.load(stream)
    assert restored==data
    with gzip.open(path/'check-cell-index.json.gz','rt') as stream:
        cells=json.load(stream)
    assert len(cells)==9 and {r['verdict'] for r in cells}=={'GOOD','BAD','NA'}
    assert (path/'frontier-atlas.pdf').read_bytes().startswith(b'%PDF')
    for panel in manifest['panels']:
        assert hashlib.sha256((path/panel['path']).read_bytes()).hexdigest()==panel['sha256']
        assert panel['path'] in (path/'index.html').read_text()
    assert (path/'frontier-overview.png').stat().st_size>1000


def test_invalid_late_source_rolls_back_previously_loaded_rows(tmp_path, monkeypatch):
    from tdn.runtime.metadata import software_metadata
    source,_,protocol,_=fixture(tmp_path)
    # Validating the sources is the engine's responsibility; simulate a verified
    # manifest then an unreadable supplemental catalog. No rows may be credited.
    (source/'catalog.json').write_text('{malformed')
    ctx=SimpleNamespace(protocol=protocol,path=tmp_path/'report',prerequisites={'audit':source},budget=SimpleNamespace(check=lambda:None))
    ctx.path.mkdir()
    data=graphs.collect(ctx,verifier=lambda protocol,path:{'stage':'audit','source_tree_sha256':software_metadata()['source_tree_sha256']})
    assert data['rows']==[] and data['stage_status']['audit']['status']=='MISSING_OR_INVALID'


def test_oversized_supplemental_curve_inventory_uses_frontier_reserve(tmp_path):
    source,tower,_,_=fixture(tmp_path)
    supplemental=source/'timing_rounds.json'
    supplemental.write_text(json.dumps({'samples':[.2]*300000}))
    manifest=json.loads((source/'science_manifest.json').read_text())
    manifest['artifacts'][supplemental.name]=hashlib.sha256(supplemental.read_bytes()).hexdigest()
    write_json(source/'science_manifest.json',manifest)
    (source/'COMPLETED').write_text(digest(manifest)+'\n')
    result=publish_outputs(tower,[source])
    assert result['frontier']['reporting_complete'],result
    inventory=json.loads((tower/'outputs/artifacts.json').read_text())['artifacts']
    entry=next(r for r in inventory if r['path'].endswith('timing_rounds.json'))
    assert entry['hash_status']=='computed_frontier_projection'


@pytest.mark.parametrize('mutation',['wrong-wrapper-stage','unexpected-wrapper-file','mixed-manifest-chain'])
def test_report_rejects_mislabelled_wrapper_and_mixed_prerequisite_lineage(tmp_path, mutation):
    from tdn.runtime.metadata import software_metadata
    source,_,protocol,_=fixture(tmp_path)
    source_hash=software_metadata()['source_tree_sha256']
    metadata={'stage':'train' if mutation=='wrong-wrapper-stage' else 'audit', 'profile':'smoke',
        'protocol_sha256':digest(protocol),'software':{'source_tree_sha256':source_hash}}
    write_json(source/'execution.json',metadata)
    write_json(source/'stage.json',{**metadata,'status':'COMPLETED'})
    files={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in
           ('execution.json','protocol.json','stage.json','science_manifest.json')}
    if mutation=='unexpected-wrapper-file':files['spurious.txt']='a'*64
    write_json(source/'workflow-seal.json',{'schema':'tdn.frontier/v1','schema_version':1,
        'protocol_sha256':digest(protocol),'files':files})
    ctx=SimpleNamespace(protocol=protocol,path=tmp_path/'report',prerequisites={'audit':source},
        budget=SimpleNamespace(check=lambda:None))
    ctx.path.mkdir()
    data=graphs.collect(ctx,verifier=lambda protocol,path:{'stage':'audit','source_tree_sha256':source_hash,
        'prerequisites':{'some-other-run':'b'*64} if mutation=='mixed-manifest-chain' else {}})
    assert data['rows']==[]
    assert data['stage_status']['audit']['status']=='MISSING_OR_INVALID'


def test_junit_occurrences_keep_skips_and_incomplete_artifacts_explicit(tmp_path):
    from tdn.runtime.metadata import software_metadata
    source,_,protocol,_=fixture(tmp_path)
    native=tmp_path/'reporter-tests'/'audit';native.mkdir(parents=True)
    (native/'tests.xml').write_text('<testsuite><testcase name="same" time=".1"/><testcase name="same"><skipped/></testcase><testcase name="broken"><failure/></testcase></testsuite>')
    other=tmp_path/'reporter-tests'/'train';other.mkdir(parents=True)
    (other/'tests.xml').write_text('<testsuite>')
    ctx=SimpleNamespace(protocol=protocol,path=tmp_path/'report',prerequisites={'audit':source},budget=SimpleNamespace(check=lambda:None))
    ctx.path.mkdir()
    data=graphs.collect(ctx,verifier=lambda protocol,path:{'stage':'audit','source_tree_sha256':software_metadata()['source_tree_sha256'],'prerequisites':{}})
    assert [r['status'] for r in data['software_tests']]==['PASSED','SKIPPED','FAILED']
    assert len({r['occurrence'] for r in data['software_tests']})==3
    assert data['software_test_sources'][1]['status']=='INVALID_OR_INCOMPLETE'
    assert json.loads((ctx.path/'junit-snapshot.json').read_text())['occurrences']==data['software_tests']


def test_native_tower_reads_and_validates_every_frontier_page(tmp_path):
    import os
    import subprocess
    import sys
    root=Path(__file__).resolve().parents[1]
    native=root/'.runtime/tower-current'
    if not (native/'tower/artifact_pages.py').is_file():
        pytest.skip('Optional unchanged Tower checkout unavailable')
    source,tower,_,_=fixture(tmp_path,129)
    curve=source/'learning_curves.jsonl'
    curve.write_text(json.dumps({'model_id':'discrete/rank1/n2/seed1','phase':'final','update':1,
        'family':'rank1','track':'discrete','seed':1,'train_count':2,'train_loss':.3,
        'validation_loss':None,'elapsed_seconds':.02,'examples_seen':4})+'\n')
    seal=json.loads((source/'science_manifest.json').read_text())
    seal['artifacts'][curve.name]=hashlib.sha256(curve.read_bytes()).hexdigest()
    write_json(source/'science_manifest.json',seal)
    (source/'COMPLETED').write_text(digest(seal)+'\n')
    assert publish_outputs(tower,[source])['frontier']['reporting_complete']
    program=r'''
import importlib.util,json,pathlib,sys
from tower.artifact_pages import read_page
from tower.artifacts import load_contract,validate_contract
from tower.metrics import _record
root=pathlib.Path(sys.argv[1])
index=json.loads((root/'outputs/frontier-tables.json').read_text())
for item in index['output_contracts']:
 contract=load_contract(root/item['path'])
 result=validate_contract(contract,root,max_bytes=64<<20,max_entries=4096)
 assert result['valid'],result
 for output in contract['outputs']:
  page=read_page(str(root),output)
  assert page['status']=='ready' and not page['has_next'] and not page['truncated'],page
for catalog in index['page_catalogs']:
 for page in json.loads((root/catalog['path']).read_text())['pages']:
  if page['table']=='metrics':
   for line in (root/page['path']).read_text().splitlines():_record(json.loads(line))
spec=importlib.util.spec_from_file_location('native_validator',sys.argv[2])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
result=module.validate_frontier_pages(root,load_contract,validate_contract)
assert result['valid'] and result['page_count']==index['page_count'],result
first=json.loads((root/index['page_catalogs'][0]['path']).read_text())['pages'][0]
with (root/first['path']).open('a') as stream:stream.write('tampered')
assert not module.validate_frontier_pages(root,load_contract,validate_contract)['valid']
print('Every frontier page and metric record accepted by unchanged Tower')
'''
    process=subprocess.run([sys.executable,'-c',program,str(tower),str(root/'scripts/tower_native_validate.py')],
        cwd=native,env={**os.environ,'PYTHONPATH':str(native),'PYTHONDONTWRITEBYTECODE':'1'},
        text=True,capture_output=True,timeout=60)
    assert process.returncode==0,process.stdout+process.stderr


def test_native_frontier_command_selects_complete_indexed_validation(tmp_path,monkeypatch):
    import importlib.util
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('frontier_tower_tools',root/'scripts/tower.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    report=tmp_path/'report';report.mkdir()
    write_json(report/'run.json',{'parameters':{'benchmark_suite':'frontier'}})
    monkeypatch.setattr(helper,'tower_profile',lambda *a:('desktop-slurm',tmp_path/'config.json'))
    monkeypatch.setattr(helper.shutil,'which',lambda name:'/native/bin/tower')
    monkeypatch.setattr(helper,'tower_interpreter',lambda exe:('/native/bin/python',None))
    command,_=helper.native_validation_command(root,report)
    assert command[-2:]==['--suite','frontier']
    assert command[command.index('--max-bytes')+1]==str(64<<20)


def test_every_loss_observation_has_strict_native_point_and_exact_lineage(tmp_path):
    source,tower,_,_=fixture(tmp_path)
    observations=[]
    for update in range(130):
        observations.append({'model_id':'discrete/rank1/n8/seed1','family':'rank1','track':'discrete',
            'seed':1,'train_count':8,'phase':'final','update':update,'train_loss':.5/(update+1),
            'validation_loss':.4/(update+1) if update%10==0 else None,
            'validation_rms':.03/(update+1) if update%10==0 else None,
            'elapsed_seconds':update*.02,'examples_seen':update*4,'update_seconds':.02,'parameters':43})
    path=source/'learning_curves.jsonl'
    path.write_text(''.join(json.dumps(row)+'\n' for row in observations))
    manifest=json.loads((source/'science_manifest.json').read_text())
    manifest['artifacts'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(source/'science_manifest.json',manifest)
    (source/'COMPLETED').write_text(digest(manifest)+'\n')
    result=publish_outputs(tower,[source])
    assert result['frontier']['reporting_complete']
    index,catalogs=pages(tower)
    assert index['learning_metric_points']==130
    assert index['published_rows']['learning_curves']==130
    assert index['published_rows']['metrics']==133
    assert index['published_rows']['metric_lineage']==133
    points=[]
    for page in catalogs:
        if page['table']=='metrics':points.extend(json.loads(line) for line in (tower/page['path']).read_text().splitlines())
    assert all(set(p)=={'t','phase','step','metrics'} for p in points)
    learning=[p for p in points if p['phase'].startswith('frontier/learning/')]
    assert len(learning)==130
    assert 'validation_loss' not in learning[1]['metrics']
    assert learning[10]['metrics']['validation_rms']==observations[10]['validation_rms']
    assert learning[10]['metrics']['examples_seen']==40
    lineage=table(tower,'metric_lineage')
    selected=next(r for r in lineage if r['observation_kind']=='learning_curve' and r['step']=='10')
    assert selected['source_line']=='11' and selected['train_count']=='8' and selected['seed']=='1'
    assert selected['trial_phase']=='final' and selected['model_id']=='discrete/rank1/n8/seed1'
    assert points[int(selected['metric_ordinal'])]==learning[10]
