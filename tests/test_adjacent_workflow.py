"""Mock Slurm tests exercise allocation contracts only, never certify CUDA science."""
from __future__ import annotations
import getpass
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tarfile
from types import SimpleNamespace
from xml.etree import ElementTree
import pytest
from tdn.analysis.adjacent.protocol import build_protocol

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def controller(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('adjacent_workflow_test',ROOT/'scripts/adjacent_workflow.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    root=tmp_path/'project with spaces'
    for name in ('tdn','reference','scripts','configs','tests'):
        (root/name).mkdir(parents=True);(root/name/'source.txt').write_text(name+'\n')
    for name in ('test_adjacent_gpu.py','test_portfolio_gpu.py','test_adjacent_models.py','test_desktop_slurm_runtime.py'):
        (root/'tests'/name).write_text('def test_ok(): pass\n')
    for name in ('adjacent_worker.sh','adjacent.py'):
        (root/'scripts'/name).write_bytes((ROOT/'scripts'/name).read_bytes())
    (root/'requirements.txt').write_text('numpy\n');(root/'pyproject.toml').write_text('[project]\nname="tdn"\n')
    for owner in (mod,mod.fw,mod.pw,mod.rw,mod.cw): monkeypatch.setattr(owner,'ROOT',root)
    for key in tuple(os.environ):
        if key.startswith(('TDN_','SLURM_')) or key in ('CONDA_PREFIX','CONDA_SHLVL'): monkeypatch.delenv(key,raising=False)
    profile={'schema_version':1,'kind':'desktop-slurm','root':str(root),'user':getpass.getuser(),
        'cpu_partition':'local','gpu_partition':'local','account':None,'gpu_gres':'gpu:1',
        'expected_gpu_name':'RTX 4090','gpu_vram_gib':24,'torch_version':'2.10.0+cu126','torch_wheel_index':'https://download.pytorch.org/whl/cu126'}
    mod.cw.atomic_json(root/'.tdn/fedora-slurm.json',profile)
    return mod


def plan(mod,name='test',profile='smoke'):
    return mod.prepare(mod.parser().parse_args(['plan','--run-id',name,'--'+profile]))


def freeze(mod,workflow):
    base=Path(workflow['run_dir']);base.mkdir(parents=True);(base/'logs').mkdir()
    for file,value in ((mod.MANIFEST,workflow),('protocol.json',mod.declaration(workflow['profile'])),
        ('slurm-profile.json',workflow['slurm_profile']),('jobs.json',[]),('state/submission-software.json',{'packages':{'torch':'fixture'}})):
        mod.cw.atomic_json(base/file,value)
    return base


def runner(mod,monkeypatch,fail_at=None):
    calls=[];monkeypatch.setattr(mod.rw,'software_report',lambda *a:{'packages':{'torch':'fixture'}})
    def command(args,**kwargs):
        calls.append((args,kwargs))
        if args[0]=='scontrol': return SimpleNamespace(stdout='PartitionName=local State=UP MaxTime=7-00:00:00')
        if args[0]=='squeue': return SimpleNamespace(stdout='')
        if args[0]=='sinfo': return SimpleNamespace(stdout='system|16|110000|gpu:1')
        count=sum(row[0][0]=='sbatch' for row in calls)
        if count==fail_at: raise ValueError('submission failed')
        return SimpleNamespace(stdout=str(9000+count))
    monkeypatch.setattr(mod.cw,'command',command)
    return calls


@pytest.mark.parametrize('profile',['smoke','development','full'])
def test_plan_reads_only_and_bounds_each_unit(controller,monkeypatch,capsys,profile):
    monkeypatch.setattr(controller.cw,'command',lambda *a,**kw:pytest.fail('plan queried scheduler'))
    assert controller.main(['plan','--'+profile])==0
    output=capsys.readouterr().out
    assert output.count('sbatch --parsable')==len(build_protocol(profile)['units'])
    assert 'no pending-job cap' in output
    assert not (controller.ROOT/'runs').exists()
    workflow=plan(controller,profile=profile)
    for unit in controller.physical_stages(workflow):
        res=controller.resource_for(workflow,unit)
        assert res['cpus']<=8 and res['mem_gib']<=48 and controller.wall_seconds(res['walltime'])<=2700


def test_resource_serialization_does_not_propagate_unrelated_failure(controller):
    workflow=plan(controller)
    jobs=[{'stage':'audit','job_id':'1'},{'stage':'D01','job_id':'2'}]
    assert controller.job_dependency(workflow,'D01',jobs)=='afterok:1,afterany:2'
    jobs.append({'stage':'D01','job_id':'3'})
    assert controller.job_dependency(workflow,'D05',jobs)=='afterok:1,afterany:3'
    assert controller.job_dependency(workflow,'report',jobs)=='afterany:1:2:3'
    assert controller.job_dependency(workflow,'audit',[]) is None


def test_submission_preserves_complete_dag_and_afterany_report(controller,monkeypatch):
    calls=runner(controller,monkeypatch)
    assert controller.main(['run','--smoke','--run-id','submitted'])==0
    workflow=controller.load(controller.workflow_path('submitted'))
    jobs=controller.cw.read_json(Path(workflow['run_dir'])/'jobs.json')
    assert [row['stage'] for row in jobs]==list(controller.physical_stages(workflow))
    assert jobs[2]['dependency']=='afterok:9001,afterany:9002'
    assert jobs[-1]['dependency']=='afterany:'+':'.join(row['job_id'] for row in jobs[:-1])
    for cmd,options in calls:
        if cmd[0]=='sbatch':
            assert '--kill-on-invalid-dep=yes' in cmd and '--signal=USR1@120' in cmd
            assert options['env']['TDN_ADJACENT_DEVICE'] in ('cpu','cuda')


def test_partial_submission_still_schedules_report(controller,monkeypatch):
    calls=runner(controller,monkeypatch,fail_at=4)
    assert controller.main(['run','--smoke','--run-id','partial'])==2
    jobs=controller.cw.read_json(controller.ROOT/'runs/partial/jobs.json')
    assert [row['stage'] for row in jobs]==['audit','D01','D05','report']
    assert jobs[-1]['dependency']=='afterany:9001:9002:9003'
    assert not any(cmd[0]=='scancel' for cmd,_ in calls)


def test_adjacent_waits_for_main_campaign(controller,monkeypatch):
    runner(controller,monkeypatch)
    original=controller.cw.command
    def command(args,**kwargs):
        if args[0]=='squeue':
            return SimpleNamespace(stdout='100|tdn-portfolio-train-A-000\n101|tdn-portfolio-report\n200|other-project\n')
        return original(args,**kwargs)
    monkeypatch.setattr(controller.cw,'command',command)
    assert controller.main(['run','--smoke','--run-id','deferred','--after-job','99'])==0
    workflow=controller.load(controller.workflow_path('deferred'))
    assert workflow['defer_job_ids']==['99','100','101']
    jobs=controller.cw.read_json(Path(workflow['run_dir'])/'jobs.json')
    assert jobs[0]['dependency']=='afterany:99:100:101'
    assert not any('200' in (j['dependency'] or '') for j in jobs)


def test_every_gpu_unit_requires_complete_suite_and_real_preflight(controller):
    workflow=plan(controller)
    for name,row in controller.units(workflow).items():
        commands=dict(controller.worker_commands(workflow,name))
        if row['device']=='cuda':
            assert {'preflight','gpu-tests','check-gpu-tests','experiment'}<=set(commands)
            assert '--gres=gpu:1' in controller.scheduler_args(workflow,name)
        if name=='D05':
            assert commands['experiment'].count('--prerequisite-dir')==1
    env={'PYTEST_ADDOPTS':'--ignore=tests','TDN_ADJACENT_WORKFLOW':'bad','TDN_EXECUTION_MODE':'desktop-slurm'}
    controller.prepare_test_environment(env,'gpu-tests')
    assert env['TDN_REQUIRE_ADJACENT_GPU_TESTS']=='1' and 'PYTEST_ADDOPTS' not in env and 'TDN_ADJACENT_WORKFLOW' not in env
    assert env['TDN_EXECUTION_MODE']=='desktop-slurm'


def junit(path,names,outcome=None):
    root=ElementTree.Element('testsuites');suite=ElementTree.SubElement(root,'testsuite',tests=str(len(names)),failures='0',errors='0',skipped='0')
    for i,name in enumerate(names):
        case=ElementTree.SubElement(suite,'testcase',classname='tests.test_portfolio_gpu' if name.startswith('test_portfolio_') else 'tests.test_adjacent_gpu',name=name)
        if i==0 and outcome: ElementTree.SubElement(case,outcome)
    ElementTree.ElementTree(root).write(path)


def test_gpu_check_requires_exact_testcase_identity(controller):
    names=build_protocol('full')['gpu_test_cases'];path=controller.ROOT/'gpu.xml'
    junit(path,names);assert controller.check_gpu_tests(path)['test_cases']==len(names)
    for values in (names[:-1],names+[names[0]],names[:-1]+['invented']):
        junit(path,values)
        with pytest.raises(ValueError): controller.check_gpu_tests(path)
    for outcome in ('skipped','failure','error'):
        junit(path,names,outcome)
        with pytest.raises(ValueError): controller.check_gpu_tests(path)


def test_unknown_stage_cannot_create_startup_artifact(controller,monkeypatch):
    workflow=plan(controller);freeze(controller,workflow)
    monkeypatch.setattr(controller.pw,'worker',lambda *a,**kw:pytest.fail('worker ran'))
    with pytest.raises(ValueError): controller.worker(workflow,'../../outside')
    assert not (controller.ROOT/'runs/outside').exists()


def test_full_dependency_failure_reporting_is_explicit_na(controller,capsys):
    workflow=plan(controller);freeze(controller,workflow)
    with pytest.raises(ValueError,match='missing or invalid'):controller.validate_results(workflow)
    result=json.loads(capsys.readouterr().out)
    assert set(result['units'])==set(controller.units(workflow))
    assert all(row['validation']=='MISSING' and row['scientific_outcome']=='NA' for row in result['units'].values())


@pytest.mark.parametrize('state',['RUNNING','PENDING','UNKNOWN','COMPLETING'])
def test_recovery_refuses_live_or_unknown_origin(controller,monkeypatch,state):
    workflow=plan(controller);base=freeze(controller,workflow)
    controller.cw.atomic_json(base/'jobs.json',[{'stage':'audit','job_id':'1'}])
    monkeypatch.setattr(controller.cw,'scheduler_state',lambda job:state)
    with pytest.raises(ValueError,match='terminal'): controller.require_terminal_origin(workflow)


def test_collect_keeps_failures_and_splits_with_hashes(controller):
    workflow=plan(controller);base=freeze(controller,workflow)
    (base/'logs/failure.err').write_text('retained failure\n'*100)
    archive=controller.collect(workflow,part_bytes=1024,accounting=False)
    index=controller.cw.read_json(str(archive)+'.index.json')
    assert index['sha256']==controller.cw.digest(archive)
    if index['parts']:
        assert b''.join((archive.parent/part['path']).read_bytes() for part in index['parts'])==archive.read_bytes()
    with tarfile.open(archive) as stream: assert workflow['run_id']+'/logs/failure.err' in stream.getnames()


@pytest.mark.parametrize('filename',['fedora_adjacent.sh','adjacent_worker.sh','adjacent_local.sh'])
def test_shell_syntax(filename):
    result=subprocess.run(['bash','-n',str(ROOT/'scripts'/filename)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_allocation_checks_exact_job_resource_and_submission(controller,monkeypatch):
    workflow=plan(controller);base=freeze(controller,workflow);stage='pilot';res=controller.resource_for(workflow,stage)
    controller.cw.atomic_json(base/'jobs.json',[{'stage':stage,'job_id':'123'}])
    allocation={'job_id':'123','job':{'JobId':'123','JobName':'tdn-adjacent-'+stage,'Comment':controller.comment(workflow,stage),
        'JobState':'RUNNING','Partition':'local','WorkDir':str(controller.ROOT),'UserId':getpass.getuser()+'(1000)',
        'AllocTRES':f"cpu={res['cpus']},mem={res['mem_gib']}G,gres/gpu=1",'TimeLimit':res['walltime']},
        'step':{'TRES':f"cpu={res['cpus']},gres/gpu=1"}}
    monkeypatch.setattr(controller,'runtime_allocation',lambda *a,**kw:allocation)
    controller.verify_allocation(workflow,stage)
    for key,value in [('JobName','other'),('Partition','wrong'),('TimeLimit','01:00:00'),('AllocTRES','cpu=4,mem=64G')]:
        prior=allocation['job'][key];allocation['job'][key]=value
        with pytest.raises(ValueError):controller.verify_allocation(workflow,stage)
        allocation['job'][key]=prior
    controller.cw.atomic_json(base/'jobs.json',[{'stage':stage,'job_id':'456'}])
    with pytest.raises(ValueError,match='recorded'):controller.verify_allocation(workflow,stage)


def test_recovery_plans_only_unfinished_units_and_preserves_origin(controller,monkeypatch):
    origin=plan(controller,'origin');base=freeze(controller,origin)
    (base/'audit').mkdir();(base/'audit/workflow-seal.json').write_text('sealed')
    controller.cw.atomic_json(base/'jobs.json',[{'stage':'audit','job_id':'1'}])
    original=(base/controller.MANIFEST).read_bytes()
    monkeypatch.setattr(controller.cw,'scheduler_state',lambda *a:'COMPLETED')
    monkeypatch.setattr(controller,'verify_stage',lambda *a,**kw:{'execution':{'stage':'audit'}})
    module=SimpleNamespace(verify_execution=lambda *a,**kw:{'execution':{'stage':'audit'}})
    monkeypatch.setattr(controller,'stage_cli',lambda:module)
    calls=runner(controller,monkeypatch)
    assert controller.main(['recover','origin','--run-id','continued'])==0
    new=controller.load(controller.workflow_path('continued'))
    assert controller.stage_path(new,'audit')==base/'audit'
    jobs=controller.cw.read_json(Path(new['run_dir'])/'jobs.json')
    assert 'audit' not in [row['stage'] for row in jobs]
    assert jobs[0]['stage']=='D01' and jobs[0]['dependency'] is None
    assert (base/controller.MANIFEST).read_bytes()==original
    assert not (Path(new['run_dir'])/'audit').exists()
    assert len([c for c,_ in calls if c[0]=='sbatch'])==len(controller.units(new))-1
    (base/'audit/workflow-seal.json').write_text('changed')
    with pytest.raises(ValueError,match='changed'):controller.verify_recovery_bridge(new)


def test_recovery_rejects_modified_execution_source_before_submission(controller,monkeypatch):
    origin=plan(controller,'origin');base=freeze(controller,origin)
    controller.cw.atomic_json(base/'jobs.json',[{'stage':'audit','job_id':'1'}])
    (controller.ROOT/'scripts/source.txt').write_text('changed')
    calls=runner(controller,monkeypatch)
    assert controller.main(['recover','origin','--run-id','unsafe'])==2
    assert not (controller.ROOT/'runs/unsafe').exists()
    assert not any(cmd[0]=='sbatch' for cmd,_ in calls)


def test_accounting_exact_ids_separates_steps_and_no_invented_cost(controller):
    workflow=plan(controller);base=freeze(controller,workflow)
    controller.cw.atomic_json(base/'jobs.json',[{'stage':'audit','job_id':'1'},{'stage':'pilot','job_id':'2'}])
    calls=[]
    def run(cmd,**kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0,stderr='',stdout='1|COMPLETED|0:0|10|00:00:02|cpu=4,mem=32G||40\n1.0|COMPLETED|0:0|9|00:00:02|cpu=4,mem=32G|20M|36\n2|FAILED|75:0|30|00:00:20|cpu=4,mem=32G,gres/gpu=1||120\n')
    result=controller.scheduler_accounting(workflow,runner=run)
    assert calls[0][calls[0].index('--jobs')+1]=='1,2'
    assert result['status']=='RECORDED' and result['all_allocations_terminal']
    assert result['monetary_cost'] is None and result['records'][1]['record_kind']=='step'
    assert result['records'][2]['State']=='FAILED' and result['records'][2]['logical_stage']=='train'
    bad=controller.scheduler_accounting(workflow,runner=lambda *a,**kw:SimpleNamespace(returncode=0,stderr='',stdout='999|COMPLETED|0:0|1|1|cpu=1||1\n'))
    assert bad['status']=='UNAVAILABLE' and bad['records']==[]
