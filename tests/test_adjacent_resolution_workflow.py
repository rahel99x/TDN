"""Frozen large-grid planning and launch boundaries; mocked Slurm is not CUDA evidence."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest
from tdn.analysis.adjacent.protocol import build_protocol, digest, validate_protocol

ROOT = Path(__file__).resolve().parents[1]

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

fixtures = module('resolution_workflow_fixtures', ROOT/'tests/test_adjacent_workflow.py')
controller = fixtures.controller
cli = module('resolution_cli_fixtures', ROOT/'scripts/adjacent.py')


@pytest.mark.parametrize('profile,expected', [
    ('smoke', '66d453bcbbd3332cc6c19d2d116fa5f6ff6ed443b724257796864466087ef43f'),
    ('development', '9c2004ffe18a74b9d2162a6f43631f1b674711fa9d2c0ea9aee4d7f296ebe97b'),
    ('full', '2bf1d41456d16e220199a025bab11b534c0fad35190b9ee8a1890f3ee9bad120'),
])
def test_historical_scientific_protocol_is_byte_equivalent(profile, expected):
    assert digest(build_protocol(profile)) == expected


@pytest.mark.parametrize('profile,grids,parents', [
    ('resolution-smoke', [64,128], {'train':1,'validation':1,'evaluation':2}),
    ('resolution-full', [64,128], {'train':16,'validation':8,'evaluation':12}),
    ('resolution64', [64], {'train':16,'validation':8,'evaluation':12}),
    ('resolution128', [128], {'train':16,'validation':8,'evaluation':12}),
])
def test_bounded_dag_freshness_and_complete_paired_coverage(profile, grids, parents):
    p = build_protocol(profile); validate_protocol(p)
    r, units = p['resolution'], p['units']
    assert r['grids'] == grids and r['splits'] == parents
    assert r['paired_grids'] == r['cross_grid_transfer'] == (len(grids)==2)
    assert r['rms_target'] == r['max_target'] == 2e-5
    assert r['tolerances'] == ([2e-5] if profile=='resolution-smoke' else [1e-4,2e-5,1e-6])
    assert p['confirmation']['primary_comparisons'] == 3*2*len(grids)
    assert 'bounded development' in p['scientific_scope']
    assert len(p['gpu_test_cases']) == len(set(p['gpu_test_cases'])) == 108
    for grid in grids:
        for track in r['tracks']:
            for split, count in parents.items():
                banks = [u for u in units.values() if u['kind']=='resolution_prepare'
                         and u['grid']==grid and u['track']==track and u['split']==split]
                ids = [i for u in banks for i in u['field_indices']]
                assert sorted(ids) == list(range(count)) and len(ids)==len(set(ids))
                assert all(u['resumable'] for u in banks)
                assert all(('freeze' in u['dependencies']) == (split=='evaluation') for u in banks)
    trained = {name for name,u in units.items() if u['kind']=='resolution_train'}
    evaluated = {name for name,u in units.items() if u['kind']=='resolution_evaluate'}
    assert set(units['freeze']['dependencies'])==trained
    assert set(units['aggregate']['dependencies'])=={'freeze',*evaluated}
    assert set(units['report']['dependencies'])==set(units)-{'report'}
    for u in units.values():
        assert u['cpus']==4 and u['mem_gib']==32 and 0<u['seconds']<=1800
        h,m,s=map(int,u['walltime'].split(':')); assert h*3600+m*60+s<=38*60
    budget=p['adjacent_budget']
    assert budget['protected_exploration_seconds']/budget['discretionary_seconds']==.2
    assert budget['summed_science_ceiling_seconds']==sum(u['seconds'] for u in units.values())
    assert r['reference']['target']=='same_grid' and r['reference']['spatial_case_limit']<=2
    assert r['model_config']['modes']==8 and r['model_config']['split_modes']==4
    assert r['exploration']['query_counts']==[1,4,8]
    assert set(r['exploration']['prototypes'])=={'HR-E01','HR-E02'}
    p['resolution']['grids']=[32]
    with pytest.raises(ValueError,match='immutable'): validate_protocol(p)


@pytest.mark.parametrize('extra,profile', [([], 'resolution-full'),(['--smoke'],'resolution-smoke'),
    (['--full','--grid','64'],'resolution64'),(['--full','--grid','128'],'resolution128')])
def test_resolution_plan_is_read_only(controller,monkeypatch,capsys,extra,profile):
    monkeypatch.setattr(controller.cw,'command',lambda *a,**kw:pytest.fail('Plan queried scheduler'))
    assert controller.main(['--resolution','plan',*extra])==0
    assert capsys.readouterr().out.count('sbatch --parsable')==len(build_protocol(profile)['units'])
    assert not (controller.ROOT/'runs').exists()


@pytest.mark.parametrize('args', [['--resolution','plan','--smoke','--grid','64'],
                                 ['--resolution','plan','--development'],['plan','--grid','64']])
def test_resolution_options_cannot_change_smoke_scope(controller,args):
    assert controller.main(args)==2
    assert not (controller.ROOT/'runs').exists()


def test_resolution_submission_has_own_latest_and_defers_existing_work(controller,monkeypatch):
    fixtures.runner(controller,monkeypatch)
    original=controller.cw.command
    def command(args,**kwargs):
        if args[0]=='squeue': return SimpleNamespace(stdout='123|tdn-portfolio-train\n124|tdn-adjacent-pilot\n125|other-project\n')
        return original(args,**kwargs)
    monkeypatch.setattr(controller.cw,'command',command)
    controller.cw.atomic_json(controller.ROOT/'runs'/controller.POINTER,{'run_id':'historical'})
    assert controller.main(['--resolution','run','--smoke','--run-id','resolution-fixture'])==0
    w=controller.load(controller.workflow_path('latest',resolution=True))
    assert w['profile']=='resolution-smoke' and w['defer_job_ids']==['123','124']
    assert controller.cw.read_json(controller.ROOT/'runs'/controller.POINTER)=={'run_id':'historical'}
    jobs=controller.cw.read_json(Path(w['run_dir'])/'jobs.json')
    assert jobs[0]['dependency']=='afterany:123:124'
    assert jobs[-1]['dependency']=='afterany:'+':'.join(j['job_id'] for j in jobs[:-1])
    by_stage={j['stage']:j['job_id'] for j in jobs}
    for j in jobs:
        if j['stage'].startswith('prepare-evaluation'):
            assert by_stage['freeze'] in j['dependency'].split(',')[0].split(':')
    assert controller.main(['status','resolution-fixture'])==2


def test_resolution_gpu_suite_is_exact_and_isolated(controller):
    for filename in ('test_adjacent_resolution_gpu.py','test_adjacent_resolution_exploration_gpu.py'):
        (controller.ROOT/'tests'/filename).write_text('# native test fixture\n')
    w=controller.prepare(controller.parser().parse_args(['--resolution','plan','--smoke']))
    commands=dict(controller.worker_commands(w,'train-n64-discrete-ours'))
    assert any(s.endswith('test_adjacent_resolution_gpu.py') for s in commands['gpu-tests'])
    assert commands['check-gpu-tests'][-2:]==['--profile','resolution-smoke']
    names=build_protocol('resolution-smoke')['gpu_test_cases'];path=controller.ROOT/'resolution.xml'
    def write(names):
        tree=ElementTree.Element('testsuites');suite=ElementTree.SubElement(tree,'testsuite',tests=str(len(names)),failures='0',errors='0',skipped='0')
        for name in names:
            group='tests.test_portfolio_gpu' if name.startswith('test_portfolio_') else 'tests.test_adjacent_resolution_exploration_gpu' if name.startswith('test_resolution_cuda_exploration') else 'tests.test_adjacent_resolution_gpu' if name.startswith('test_resolution_') else 'tests.test_adjacent_gpu'
            ElementTree.SubElement(suite,'testcase',classname=group,name=name)
        ElementTree.ElementTree(tree).write(path)
    write(names); assert controller.check_gpu_tests(path,profile='resolution-smoke')['test_cases']==108
    with pytest.raises(ValueError): controller.check_gpu_tests(path)
    write(names[:-1])
    with pytest.raises(ValueError): controller.check_gpu_tests(path,profile='resolution-smoke')


@pytest.mark.parametrize('profile',['resolution-full','resolution64','resolution128'])
def test_resolution_native_profiles_rejected_locally_before_writes(tmp_path,profile):
    target=tmp_path/'forbidden'
    env={k:v for k,v in os.environ.items() if not k.startswith(('TDN_','SLURM_'))}
    result=subprocess.run([sys.executable,str(ROOT/'scripts/adjacent.py'),'--profile',profile,'--stage','audit',
        '--run-dir',str(target),'--local-root',str(ROOT)],cwd=ROOT,env=env,capture_output=True,text=True)
    assert result.returncode!=0 and 'CPU smoke/development' in result.stderr
    assert not target.exists()


@pytest.mark.parametrize('stage',['prepare-train-n64-discrete-p000','train-n64-discrete-ours','evaluate-n64-discrete-p000'])
def test_journal_resume_requires_exact_resolution_identity(tmp_path,stage):
    p=build_protocol('resolution-smoke');software={'python':'fixture','source_tree_sha256':'source'}
    lineage={'audit':{'workflow_seal_sha256':'fixture'}}
    execution={'stage':stage,'profile':p['profile'],'protocol_sha256':digest(p),'software':software,
               'execution_mode':'local-cpu','device':'cpu','prerequisites':lineage}
    (tmp_path/'execution.json').write_text(json.dumps(execution))
    (tmp_path/'stage.json').write_text('{"status":"INTERRUPTED"}')
    kwargs=dict(stage=stage,protocol=p,software=software,mode='local-cpu',device='cpu',lineage=lineage)
    assert cli.check_resume(tmp_path,**kwargs)==execution
    with pytest.raises(ValueError,match='differs'): cli.check_resume(tmp_path,**{**kwargs,'lineage':{}})
    (tmp_path/'COMPLETED').write_text('bad')
    with pytest.raises(ValueError,match='Completed'): cli.check_resume(tmp_path,**kwargs)


@pytest.mark.parametrize('script',['fedora_adjacent_resolution.sh','adjacent_resolution_local.sh'])
def test_new_wrapper_shell_syntax(script):
    result=subprocess.run(['bash','-n',str(ROOT/'scripts'/script)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_resolution_recovery_preserves_profile_and_reference_journals(controller,monkeypatch):
    origin=controller.prepare(controller.parser().parse_args(['--resolution','plan','--full','--grid','64','--run-id','origin']))
    base=fixtures.freeze(controller,origin)
    audit=base/'audit';audit.mkdir();(audit/'workflow-seal.json').write_text('synthetic seal')
    name='prepare-train-n64-discrete-p000';interrupted=base/name;interrupted.mkdir()
    (interrupted/'execution.json').write_text('{"fixture":"interrupted"}')
    (interrupted/'stage.json').write_text('{"status":"INTERRUPTED"}')
    (interrupted/'reference-parent.npz').write_bytes(b'unit-test teacher journal, never scientific evidence')
    controller.cw.atomic_json(base/'jobs.json',[{'stage':'audit','job_id':'50'},{'stage':name,'job_id':'51'}])
    before={str(p.relative_to(base)):p.read_bytes() for p in base.rglob('*') if p.is_file()}
    monkeypatch.setattr(controller.cw,'scheduler_state',lambda _: 'COMPLETED')
    monkeypatch.setattr(controller,'verify_stage',lambda *a,**k:{'execution':{'stage':'audit'}})
    # Fingerprint with ordinary bytes in this fixture; production retains contained_path checks.
    def fingerprint(path):
        return digest({str(p.relative_to(path)):p.read_bytes().hex() for p in Path(path).rglob('*') if p.is_file()})
    monkeypatch.setattr(controller,'stage_cli',lambda:SimpleNamespace(
        verify_execution=lambda *a,**k:{'execution':{'stage':'audit'}},resume_fingerprint=fingerprint))
    calls=fixtures.runner(controller,monkeypatch)
    old=controller.cw.command
    def command(args,**kwargs):
        if args[0]=='squeue':return SimpleNamespace(stdout='310|tdn-portfolio-report\n')
        return old(args,**kwargs)
    monkeypatch.setattr(controller.cw,'command',command)
    assert controller.main(['--resolution','recover','origin','--run-id','recovered'])==0
    recovered=controller.load(controller.workflow_path('latest',resolution=True))
    assert recovered['profile']=='resolution64'
    assert recovered['recovery']['resume_paths']=={name:str(interrupted)}
    assert recovered['recovery']['stage_paths']=={'audit':str(audit)}
    jobs=controller.cw.read_json(Path(recovered['run_dir'])/'jobs.json')
    assert jobs[0]['stage']==name and jobs[0]['dependency']=='afterany:310'
    commands=dict(controller.worker_commands(recovered,name))
    assert commands['experiment'][-2:]==['--resume-from',str(interrupted)]
    assert {str(p.relative_to(base)):p.read_bytes() for p in base.rglob('*') if p.is_file()}==before
    assert not any(cmd[0]=='scancel' for cmd,_ in calls)


@pytest.mark.parametrize('profile',['resolution-smoke','resolution-full','resolution64','resolution128'])
def test_resolution_snapshot_and_strong_cubic_configuration(profile):
    from tdn.analysis.adjacent.resolution_protocol import model_config_for
    from tdn.analysis.adjacent.models import make_model
    p=build_protocol(profile)
    assert json.loads((ROOT/'configs/adjacent'/f'{profile}.json').read_text())==p
    assert model_config_for(p,'channel_neural')['quad_nodes']==2
    cubic=model_config_for(p,'analytic_quad_cubic')
    assert cubic['quad_nodes']==cubic['cubic_nodes']==4
    actual=make_model('analytic_quad_cubic','discrete',cubic).architecture_metadata()
    assert actual['nodes']==4 and actual['cubic_nodes']==4
