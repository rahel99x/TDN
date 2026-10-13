"""Mock Slurm tests exercise allocation contracts only, never certify CUDA science."""
from __future__ import annotations
import getpass
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace
from xml.etree import ElementTree
import pytest
from tdn.analysis.response.protocol import build_protocol

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def controller(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('response_workflow_test',ROOT/'scripts/response_workflow.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    root=tmp_path/'project with spaces'
    for name in ('tdn','reference','scripts','configs','tests'):
        (root/name).mkdir(parents=True);(root/name/'source.txt').write_text(name+'\n')
    for name in ('test_response_gpu.py','test_response_cli.py','test_desktop_slurm_runtime.py'):
        (root/'tests'/name).write_text('def test_ok(): pass\n')
    for name in ('response_worker.sh','response.py'):
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
    for file,value in ((mod.MANIFEST,workflow),('protocol.json',mod.declaration(workflow)),
        ('slurm-profile.json',workflow['slurm_profile']),('jobs.json',[]),('state/submission-software.json',{'packages':{'torch':'fixture'}})):
        mod.cw.atomic_json(base/file,value)
    return base


def runner(mod,monkeypatch,fail_at=None):
    calls=[];monkeypatch.setattr(mod.rw,'software_report',lambda *a:{'packages':{'torch':'fixture'}})
    def command(args,**kwargs):
        calls.append((args,kwargs))
        if args[0]=='scontrol': return SimpleNamespace(stdout='PartitionName=local State=UP MaxTime=7-00:00:00')
        if args[0]=='sinfo': return SimpleNamespace(stdout='system|16|110000|gpu:1')
        count=sum(row[0][0]=='sbatch' for row in calls)
        if count==fail_at: raise ValueError('submission failed')
        return SimpleNamespace(stdout=str(9000+count))
    monkeypatch.setattr(mod.cw,'command',command)
    return calls


@pytest.mark.parametrize('profile',['smoke','pilot','development'])
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
        assert res['cpus']<=8 and res['mem_gib']<=48 and controller.wall_seconds(res['walltime'])<=2100


def test_resource_serialization_does_not_propagate_unrelated_failure(controller):
    workflow=plan(controller)
    fit=next(name for name,row in controller.units(workflow).items() if row['kind']=='train')
    jobs=[{'stage':'audit','job_id':'1'},{'stage':fit,'job_id':'2'}]
    assert controller.job_dependency(workflow,'fields',jobs)=='afterok:1,afterany:2'
    jobs.append({'stage':'fields','job_id':'3'})
    exploratory=next(name for name,row in controller.units(workflow).items() if row['kind']=='closure')
    assert controller.job_dependency(workflow,exploratory,jobs)=='afterok:1,afterany:3'
    assert controller.job_dependency(workflow,'report',jobs)=='afterany:1:2:3'
    assert controller.job_dependency(workflow,'audit',[]) is None


def test_native_smoke_cost_plan_schedules_bounded_complete_work(controller):
    targets=['fd_same_grid','galerkin_same_grid','galerkin_projected_fine']
    workflow=controller.prepare(controller.parser().parse_args(['plan','--smoke','--run-id','cost-budget','--targets',*targets]))
    rows=controller.units(workflow)
    costs=[name for name,row in rows.items() if row['kind']=='cost']
    assert len(costs)==3
    for name in costs:
        res=controller.resource_for(workflow,name)
        args=controller.scheduler_args(workflow,name)
        assert rows[name]['seconds']==1800 and res['walltime']=='00:33:00'
        assert controller.wall_seconds(res['walltime'])-rows[name]['seconds']==180
        assert '--time=00:33:00' in args and '--signal=USR1@120' in args
        assert '--cpus-per-task=4' in args and '--mem=32G' in args and '--gres=gpu:1' in args
    assert controller.scientific_protocol(workflow)['gpu_test_cases']==controller.scientific_protocol('smoke')['gpu_test_cases']
    assert not (controller.ROOT/'runs').exists()


def test_submission_preserves_complete_dag_and_afterany_report(controller,monkeypatch):
    calls=runner(controller,monkeypatch)
    assert controller.main(['run','--smoke','--run-id','submitted'])==0
    workflow=controller.load(controller.workflow_path('submitted'))
    jobs=controller.cw.read_json(Path(workflow['run_dir'])/'jobs.json')
    assert [row['stage'] for row in jobs]==list(controller.physical_stages(workflow))
    assert jobs[1]['dependency']=='afterok:9001'
    assert jobs[2]['dependency']=='afterok:9002'
    assert jobs[-1]['dependency']=='afterany:'+':'.join(row['job_id'] for row in jobs[:-1])
    for cmd,options in calls:
        if cmd[0]=='sbatch':
            assert '--kill-on-invalid-dep=yes' in cmd and '--signal=USR1@120' in cmd
            assert options['env']['TDN_RESPONSE_DEVICE'] in ('cpu','cuda')


def test_partial_submission_still_schedules_report(controller,monkeypatch):
    calls=runner(controller,monkeypatch,fail_at=4)
    assert controller.main(['run','--smoke','--run-id','partial'])==2
    jobs=controller.cw.read_json(controller.ROOT/'runs/partial/jobs.json')
    expected=list(build_protocol('smoke')['units'])[:3]+['report']
    assert [row['stage'] for row in jobs]==expected
    assert jobs[-1]['dependency']=='afterany:9001:9002:9003'
    assert not any(cmd[0]=='scancel' for cmd,_ in calls)


def test_every_gpu_unit_requires_complete_suite_and_real_preflight(controller):
    workflow=plan(controller)
    for name,row in controller.units(workflow).items():
        commands=dict(controller.worker_commands(workflow,name))
        if row['device']=='cuda':
            assert {'preflight','gpu-tests','check-gpu-tests','experiment'}<=set(commands)
            assert '--gres=gpu:1' in controller.scheduler_args(workflow,name)
        if row['kind']=='closure':
            assert commands['experiment'].count('--prerequisite-dir')==1
    env={'PYTEST_ADDOPTS':'--ignore=tests','TDN_RESPONSE_WORKFLOW':'bad','TDN_EXECUTION_MODE':'desktop-slurm'}
    controller.prepare_test_environment(env,'gpu-tests')
    assert env['TDN_REQUIRE_RESPONSE_GPU_TESTS']=='1' and 'PYTEST_ADDOPTS' not in env and 'TDN_RESPONSE_WORKFLOW' not in env
    assert env['TDN_EXECUTION_MODE']=='desktop-slurm'


def junit(path,names,outcome=None):
    root=ElementTree.Element('testsuites');suite=ElementTree.SubElement(root,'testsuite',tests=str(len(names)),failures='0',errors='0',skipped='0')
    for i,name in enumerate(names):
        case=ElementTree.SubElement(suite,'testcase',classname='tests.test_response_gpu',name=name)
        if i==0 and outcome: ElementTree.SubElement(case,outcome)
    ElementTree.ElementTree(root).write(path)


def test_gpu_check_requires_exact_testcase_identity(controller):
    names=build_protocol('smoke')['gpu_test_cases'];path=controller.ROOT/'gpu.xml'
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


@pytest.mark.parametrize('filename',['fedora_response.sh','response_worker.sh','response_local.sh'])
def test_shell_syntax(filename):
    result=subprocess.run(['bash','-n',str(ROOT/'scripts'/filename)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_allocation_checks_exact_job_resource_and_submission(controller,monkeypatch):
    workflow=plan(controller);base=freeze(controller,workflow)
    stage=next(name for name,row in controller.units(workflow).items() if row['device']=='cuda')
    res=controller.resource_for(workflow,stage)
    controller.cw.atomic_json(base/'jobs.json',[{'stage':stage,'job_id':'123'}])
    allocation={'job_id':'123','job':{'JobId':'123','JobName':'tdn-response-'+stage,'Comment':controller.comment(workflow,stage),
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
    assert jobs[0]['stage']=='fields' and jobs[0]['dependency'] is None
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
    training=next(name for name,row in controller.units(workflow).items() if row['kind']=='train')
    controller.cw.atomic_json(base/'jobs.json',[{'stage':'audit','job_id':'1'},{'stage':training,'job_id':'2'}])
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


@pytest.mark.parametrize('key,value', [
    ('cpus', 9), ('cpus', True), ('mem_gib', 49), ('walltime', '00:36:00'),
    ('walltime', '00:02:00'), ('walltime', '00:60:00'), ('seconds', 1801),
])
def test_protocol_resource_mutations_cannot_exceed_desktop_envelope(controller,key,value):
    protocol=copy.deepcopy(build_protocol('smoke'))
    protocol['units']['fields'][key]=value
    with pytest.raises(ValueError): controller.validate_resources(protocol)


@pytest.mark.parametrize('run_id', ['../outside','a/b','a;true','a\nname',''])
def test_unsafe_run_identifiers_rejected_without_writes(controller,run_id):
    # An omitted identifier requests an autogenerated safe name; the explicit
    # unsafe strings are refused by the controller before any submission.
    if not run_id:
        with pytest.raises(ValueError): controller.pw.identifier(run_id)
    else:
        assert controller.main(['plan','--run-id',run_id,'--smoke'])==2
    assert not (controller.ROOT/'runs').exists()


def test_prior_campaign_flags_cannot_contaminate_cpu_or_gpu_tests(controller):
    original={'TDN_REQUIRE_PORTFOLIO_GPU_TESTS':'1','TDN_REQUIRE_GPU_TESTS':'1',
        'TDN_PORTFOLIO_WORKFLOW':'old','TDN_RESPONSE_WORKFLOW':'new',
        'TDN_EXECUTION_MODE':'desktop-slurm','SLURM_JOB_ID':'123'}
    cpu=original.copy();controller.prepare_test_environment(cpu,'tests')
    assert cpu=={'SLURM_JOB_ID':'123'}
    gpu=original.copy();controller.prepare_test_environment(gpu,'gpu-tests')
    assert gpu=={'TDN_EXECUTION_MODE':'desktop-slurm','SLURM_JOB_ID':'123','TDN_REQUIRE_RESPONSE_GPU_TESTS':'1'}


def test_cpu_allocation_refuses_unexpected_gpu(controller,monkeypatch):
    workflow=plan(controller);base=freeze(controller,workflow);stage='audit'
    res=controller.resource_for(workflow,stage)
    controller.cw.atomic_json(base/'jobs.json',[{'stage':stage,'job_id':'123'}])
    allocation={'job_id':'123','job':{'JobId':'123','JobName':'tdn-response-'+stage,
        'Comment':controller.comment(workflow,stage),'JobState':'RUNNING','Partition':'local',
        'WorkDir':str(controller.ROOT),'UserId':getpass.getuser()+'(1000)',
        'AllocTRES':f"cpu={res['cpus']},mem={res['mem_gib']}G,gres/gpu=1",'TimeLimit':res['walltime']},
        'step':{'TRES':f"cpu={res['cpus']},gres/gpu=1"}}
    monkeypatch.setattr(controller,'runtime_allocation',lambda *a,**kw:allocation)
    with pytest.raises(ValueError,match='unexpectedly allocated a GPU'):controller.verify_allocation(workflow,stage)


def test_compact_wrapper_responses_only_response_latest_pointer(tmp_path):
    root=tmp_path/'desktop with spaces';(root/'scripts').mkdir(parents=True);(root/'runs').mkdir()
    (root/'scripts/fedora_response.sh').write_bytes((ROOT/'scripts/fedora_response.sh').read_bytes())
    (root/'scripts/compact_review.py').write_text('import json,sys; print(json.dumps(sys.argv[1:]))\n')
    for campaign,run in [('response','response-chosen'),('portfolio','wrong-portfolio-run')]:
        (root/f'runs/.fedora-{campaign}-latest.json').write_text(json.dumps({'run_id':run}))
    environment={**os.environ,'TDN_PYTHON':sys.executable}
    result=subprocess.run(['bash',str(root/'scripts/fedora_response.sh'),'compact','latest','--mode','review'],
        env=environment,capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)==['pack','response-chosen','--mode','review']
    explicit=subprocess.run(['bash',str(root/'scripts/fedora_response.sh'),'compact','explicit-rerun'],
        env=environment,capture_output=True,text=True,check=True)
    assert json.loads(explicit.stdout)==['pack','explicit-rerun']


def test_list_uses_only_own_coordinator_metadata(controller,monkeypatch,capsys):
    workflow=plan(controller,'own-run');freeze(controller,workflow)
    other=controller.ROOT/'runs/other-run';other.mkdir()
    (other/'advance-workflow.json').write_text('{}')
    monkeypatch.setattr(controller,'scientific_protocol',lambda *a:pytest.fail('list imported science'))
    monkeypatch.setattr(controller.cw,'command',lambda *a,**kw:pytest.fail('list queried scheduler'))
    assert controller.main(['list'])==0
    output=capsys.readouterr().out
    assert 'own-run\tsmoke' in output and 'other-run' not in output


def test_wrapper_prefers_project_venv_without_shell_activation(tmp_path):
    root=tmp_path/'desktop with spaces';(root/'scripts').mkdir(parents=True)
    (root/'.venv/bin').mkdir(parents=True)
    (root/'scripts/fedora_response.sh').write_bytes((ROOT/'scripts/fedora_response.sh').read_bytes())
    python=root/'.venv/bin/python'
    python.write_text('#!/usr/bin/env bash\nprintf "PROJECT_VENV:%s\\n" "$*"\n')
    python.chmod(0o755)
    env={k:v for k,v in os.environ.items() if k not in ('TDN_PYTHON','VIRTUAL_ENV')}
    result=subprocess.run(['bash',str(root/'scripts/fedora_response.sh'),'status','latest'],
        env=env,capture_output=True,text=True,check=True)
    assert result.stdout.startswith('PROJECT_VENV:') and 'response_workflow.py status latest' in result.stdout


def test_old_campaign_flags_are_not_exported_into_new_worker(controller,monkeypatch):
    workflow=plan(controller)
    for key in ('TDN_ADVANCE_WORKFLOW','TDN_ADVANCE_STAGE','TDN_PORTFOLIO_WORKFLOW','TDN_REQUIRE_ADVANCE_GPU_TESTS'):
        monkeypatch.setenv(key,'stale')
    env=controller.submission_environment(workflow,'audit')
    assert env['TDN_RESPONSE_STAGE']=='audit'
    assert not any(key.startswith(('TDN_ADVANCE_','TDN_PORTFOLIO_','TDN_REQUIRE_')) for key in env)


def test_submission_blocks_uncalibrated_research_before_any_scheduler_request(controller,monkeypatch):
    monkeypatch.setattr(controller.cw,'command',lambda *a,**kw:pytest.fail('queried scheduler before binding'))
    assert controller.main(['run','--pilot','--run-id','uncalibrated'])==2
    assert not (controller.ROOT/'runs').exists()


def test_native_calibration_ignores_short_integration_trials_and_binds_bytes(controller,monkeypatch):
    workflow=plan(controller);base=freeze(controller,workflow)
    monkeypatch.setattr(controller,'verify_stage',lambda *a:None)
    for stage,unit in controller.units(workflow).items():
        if unit['kind']!='train' or unit.get('phase')!='factorial':continue
        trial=workflow['scientific_protocol']['trials'][unit['trial_id']]
        path=base/stage;path.mkdir()
        controller.cw.atomic_json(path/'trial.json',{'family':trial['family'],'target_kind':trial['target_kind'],
            'updates':64,'status':'COMPLETED_UPDATES','cost':{'total_seconds':6.4},'total_cost_including_postprocessing_seconds':12.8})
        controller.cw.atomic_json(path/'execution.json',{'execution_mode':'desktop-slurm','device':'cuda'})
        (path/'workflow-seal.json').write_text('fixtureseal')
        jp=controller.junit_path(workflow,stage);jp.parent.mkdir(parents=True)
        junit(jp,build_protocol()['gpu_test_cases'])
    result=controller.read_calibration(workflow['run_id'])
    assert len(result['seconds_per_update'])==8 and set(result['seconds_per_update'].values())=={.2}
    assert len(result['sources'])==8 and result['verified_native_gpu']
    first=next(iter(result['sources'].values()))
    changed=Path(first['path'])/'trial.json';row=controller.cw.read_json(changed)
    row['updates']=2;controller.cw.atomic_json(changed,row)
    with pytest.raises(ValueError,match='64'):controller.read_calibration(workflow['run_id'])


def test_gate_cannot_promote_smoke_or_incomplete_or_failed_competitor(controller,monkeypatch):
    with pytest.raises(ValueError,match='smoke'):controller.gate_results(plan(controller))
    workflow=plan(controller,profile='pilot');base=freeze(controller,workflow)
    monkeypatch.setattr(controller,'verify_stage',lambda *a:None)
    groups=[{'target_kind':t,'family':f,'passed':True} for t in workflow['scientific_protocol']['targets'] for f in ('direct_fno','residual_fno','temporal')]
    settings={r['target_kind']+'/'+r['family']:[{'setting':{'lr':.001}}] for r in groups}
    controller.cw.atomic_json(base/'gate/gate.json',{'gate_passed':False,'groups':groups})
    controller.cw.atomic_json(base/'optimizer-selection/selection.json',{'selected_settings':settings})
    assert controller.gate_results(workflow)['eligible_for_development'] is False
    controller.cw.atomic_json(base/'gate/gate.json',{'gate_passed':True,'groups':groups[:-1]})
    with pytest.raises(ValueError,match='incomplete'):controller.gate_results(workflow)
    controller.cw.atomic_json(base/'gate/gate.json',{'gate_passed':True,'groups':groups})
    assert controller.gate_results(workflow)['eligible_for_development'] is True


def test_freeze_creates_verified_checkpoint_derivative_without_mutating_science(controller,monkeypatch):
    workflow=plan(controller,profile='development');base=freeze(controller,workflow)
    monkeypatch.setattr(controller,'verify_stage',lambda *a:None)
    monkeypatch.setattr(controller,'verify_external_bindings',lambda *a:None)
    stage=next(k for k,v in controller.units(workflow).items() if v['kind']=='train')
    directory=base/stage;directory.mkdir();(directory/'selected.pt').write_bytes(b'fixture-checkpoint-bytes')
    row=dict(stage=stage,family='direct_fno',target_kind='fd_same_grid',seed=1,variant_id='primary',data_count=256,
        method_id='direct_fno::primary::n256',trial_directory=str(directory),checkpoint_sha256=controller.cw.digest(directory/'selected.pt'))
    controller.cw.atomic_json(base/'selection/selection.json',{'selected_models':[row],'missing':[]})
    controller.cw.atomic_json(base/'aggregate/aggregate.json',{'freeze_eligible':True})
    prior=(directory/'selected.pt').read_bytes();output=controller.ROOT/'outputs'/'locked'
    controller.freeze_results(workflow,output)
    descriptor=controller.read_freeze(output/'freeze.json')
    frozen=controller.cw.read_json(output/'freeze.json')
    copy_path=output/frozen['models'][0]['checkpoint']
    assert copy_path.read_bytes()==prior and (directory/'selected.pt').read_bytes()==prior
    assert descriptor['sha256']==controller.cw.digest(output/'freeze.json')
    assert descriptor['evaluation']==workflow['scientific_protocol']['evaluation']
    copy_path.chmod(0o644);copy_path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='checkpoint'):controller.read_freeze(output/'freeze.json')


def test_accounting_pending_is_not_reported_final_cost(controller):
    workflow=plan(controller);base=freeze(controller,workflow)
    controller.cw.atomic_json(base/'jobs.json',[{'stage':'report','job_id':'1'}])
    result=controller.scheduler_accounting(workflow,runner=lambda *a,**kw:SimpleNamespace(returncode=0,stderr='',
        stdout='1|RUNNING|0:0|17|00:00:02|cpu=2,mem=32G||34\n'))
    assert result['status']=='PENDING_ACCOUNTING' and not result['all_allocations_terminal']
    assert result['monetary_cost'] is None
