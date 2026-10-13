"""Native controller stubs check policy; none establish hardware readiness."""
from __future__ import annotations

import copy
from contextlib import nullcontext
import getpass
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import tangent_response_workflow as workflow
from tdn.analysis.tangent_response.protocol import build_protocol


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


@pytest.fixture
def native(tmp_path,monkeypatch):
    root=tmp_path/'project with spaces';root.mkdir()
    site=dict(schema_version=1,kind='desktop-slurm',root=str(root),user=getpass.getuser(),cpu_partition='local',
        gpu_partition='local',account=None,gpu_gres='gpu:1',expected_gpu_name='RTX 4090',gpu_vram_gib=24,
        torch_version='2.10.0+cu126',torch_wheel_index='https://download.pytorch.org/whl/cu126')
    monkeypatch.setattr(workflow,'write',put)
    monkeypatch.setattr(workflow,'load_profile',lambda **_:site)
    monkeypatch.setattr(workflow.cw,'source_hash',lambda **_: 'a'*64)
    return workflow.NativeWorkflow(root),site


def args(**values):
    result=dict(command='plan',profile='pilot',target='fd_same_grid',phase='mechanism',run_id='native-test',
        prerequisite=[],calibration=None,no_worker_telemetry=False,worker_telemetry='cpu,memory,gpu',sampling_seconds=5)
    result.update(values);return SimpleNamespace(**result)


def test_plan_is_read_only_and_discloses_downstream_unavailable(native,capsys):
    api,site=native;w=api.prepare(args());api.submit(w,plan=True)
    assert not (api.root/'runs').exists()
    assert 'UNAVAILABLE' in capsys.readouterr().out
    assert all(workflow.shared.wall_seconds(r['walltime'])<=2100 for r in w['resources'].values())
    assert w['scientific_protocol']['training']['updates']==300
    assert len(w['scientific_protocol']['training']['seeds'])==1
    assert w['slurm_profile']==site


def test_uncalibrated_learning_plan_has_no_training_allocation(native,capsys):
    api,_=native;w=api.prepare(args(phase='learning'));api.submit(w,plan=True)
    assert not any(u['kind']=='training' for u in api.units(w).values())
    assert all(t['allocation_status']=='UNAVAILABLE_UNTIL_NATIVE_CALIBRATION' for t in api.protocol(w)['trials'].values())
    assert not (api.root/'runs').exists()
    with pytest.raises(ValueError,match='gate'):
        api.prepare(args(command='run',phase='learning'))


@pytest.mark.parametrize('phase',['compression','fit_prepare','calibration','utility'])
def test_downstream_native_work_requires_literal_passing_gate(native,phase):
    api,_=native
    with pytest.raises(ValueError,match='gate'):
        api.prepare(args(command='run',phase=phase))


def test_worker_environment_removes_captured_coordinator_and_other_program_overrides(native,monkeypatch):
    api,_=native;w=api.prepare(args())
    for key in ('RAVEN_RUN_DIR','RAVEN_LAUNCH_PLAN','SBATCH_PARTITION','TDN_RESPONSE_WORKFLOW','TDN_TOWER_DIR','TDN_REQUIRE_GPU_TESTS','TDN_TANGENT_CAPTURED_SCRIPT','CARC_ACCOUNT'):
        monkeypatch.setenv(key,'outer-value')
    env=api.environment(w,'audit')
    assert not any(k.startswith(('RAVEN_','SBATCH_','TDN_RESPONSE_','TDN_REQUIRE_')) for k in env)
    assert 'TDN_TOWER_DIR' not in env and 'CARC_ACCOUNT' not in env and 'TDN_TANGENT_CAPTURED_SCRIPT' not in env
    assert env['TDN_TANGENT_STAGE']=='audit' and env['TDN_EXECUTION_MODE']=='desktop-slurm'


def test_real_scheduler_commands_bind_exact_unit_and_afterany_report(native):
    api,_=native;w=api.prepare(args());stages=api.stages(w);first=stages[0]
    command=api.scheduler_args(w,first)
    assert '--chdir='+str(api.root) in command
    assert '--comment='+api.comment(w,first) in command
    assert '--gres=gpu:1' in command and '--cpus-per-task=4' in command
    jobs=[dict(stage=first,job_id='101'),dict(stage=stages[1],job_id='102')]
    assert api.dependency(w,'report',jobs)=='afterany:101:102'
    assert api.dependency(w,stages[2],jobs).startswith('afterok:102')
    assert api.dependency(w,'decision',jobs).startswith('afterany:102')


@pytest.mark.parametrize('change',[{'cpus':9},{'device':'cuda','cpus':8},{'mem_gib':49},{'walltime':'00:36:00'},{'seconds':1801}])
def test_native_resource_envelope_cannot_expand(native,change):
    api,site=native;p=build_protocol('pilot');p['units']['audit'].update(change)
    with pytest.raises(ValueError,match='envelope'):api.resources(p,site)


def test_cuda_roster_rejects_skip_omission_duplicates_and_wrong_namespace(native,tmp_path):
    api,_=native;p=dict(gpu_test_cases=[('tests.test_tangent_response_gpu','case_a'),('tests.test_tangent_response_gpu','case_b')])
    good='<testcase classname="tests.test_tangent_response_gpu" name="case_a"/><testcase classname="tests.test_tangent_response_gpu" name="case_b"/>'
    file=tmp_path/'tests.xml';file.write_text('<testsuites><testsuite>'+good+'</testsuite></testsuites>')
    assert api.check_gpu_tests(file,p)['test_cases']==2
    for body in (good.replace('case_b','case_a'),good.replace('case_b','old_response_case'),good.replace('name="case_b"/>','name="case_b"><skipped/></testcase>'),good.split('/>')[0]+'/>'):
        file.write_text('<testsuites><testsuite>'+body+'</testsuite></testsuites>')
        with pytest.raises(ValueError):api.check_gpu_tests(file,p)


def test_historical_status_loading_does_not_compare_current_source(native):
    api,_=native;w=api.prepare(args());base=Path(w['run_dir']);put(base/workflow.MANIFEST,w);put(base/'protocol.json',api.protocol(w))
    changed=copy.deepcopy(w);changed['source_sha256']='b'*64;put(base/workflow.MANIFEST,changed)
    assert api.load(base/workflow.MANIFEST,verify=False)['source_sha256']=='b'*64
    with pytest.raises(ValueError,match='Source changed'):api.load(base/workflow.MANIFEST)


class CampaignNative:
    def __init__(self,root,decisions):self.root=root;self.decisions=decisions;self.planned=[];self.workflows={}
    path=lambda self,path:Path(path)
    read=lambda self,path:json.loads(Path(path).read_text())
    def prepare(self,args):
        assert (self.root/'runs/.tangent-response-campaigns'/args.campaign_run_id/'receipt.json').is_file()
        self.planned.append(args.phase)
        w=dict(run_id=args.run_id,phase=args.phase,run_dir=str(self.root/'runs'/args.run_id),campaign_run_id=args.campaign_run_id)
        self.workflows[args.run_id]=w;return w
    def submit(self,w):pass
    def status(self,w):return dict(terminal=True,completed=True,failed_units=[],report_complete=True)
    def validate_results(self,w):return dict(validation='VERIFIED')
    def decision(self,w):return dict(gate=w['phase'],decision=self.decisions.get(w['phase'],'PASS'),eligible=self.decisions.get(w['phase'],'PASS')=='PASS')
    def units(self,w):return {'decision':{'kind':'decision'}}


@pytest.mark.parametrize('outcome',['STOP','UNRESOLVED','IMPLEMENTATION_ONLY'])
def test_scientific_negative_is_completed_evidence_and_never_expands(tmp_path,monkeypatch,outcome):
    monkeypatch.setattr(workflow,'write',put);monkeypatch.setattr(workflow.cw,'source_hash',lambda **_: 'a'*64)
    api=CampaignNative(tmp_path,{'mechanism':outcome});runner=workflow.CampaignCoordinator(api,args(command='campaign',run_id='parent'),sleep=lambda _:None)
    packed=[];runner.compact=packed.append
    result=runner.execute()
    assert api.planned==['mechanism'] and result['status']=='COMPLETED' and result['decision']==outcome
    assert set(result['not_submitted'])==set(workflow.PHASES[1:]) and packed==['parent']


def test_fit_screen_negative_blocks_calibration_even_after_a_b_pass(tmp_path,monkeypatch):
    monkeypatch.setattr(workflow,'write',put);monkeypatch.setattr(workflow.cw,'source_hash',lambda **_: 'a'*64)
    api=CampaignNative(tmp_path,{'fit_prepare':'UNRESOLVED'});runner=workflow.CampaignCoordinator(api,args(command='campaign',run_id='parent'),sleep=lambda _:None)
    runner.compact=lambda _:None;result=runner.execute()
    assert api.planned==['mechanism','compression','fit_prepare'] and 'calibration' in result['not_submitted']


def test_full_eligible_campaign_has_distinct_children_and_no_auto_retry(tmp_path,monkeypatch):
    monkeypatch.setattr(workflow,'write',put);monkeypatch.setattr(workflow.cw,'source_hash',lambda **_: 'a'*64)
    api=CampaignNative(tmp_path,{});runner=workflow.CampaignCoordinator(api,args(command='campaign',run_id='parent'),sleep=lambda _:None)
    runner.compact=lambda _:None;result=runner.execute()
    assert api.planned==list(workflow.PHASES)
    assert len(set(result['run_ids'].values()))==6 and result['automatic_retry'] is False


def test_native_infrastructure_failure_stops_with_preserved_receipt(tmp_path,monkeypatch):
    monkeypatch.setattr(workflow,'write',put);monkeypatch.setattr(workflow.cw,'source_hash',lambda **_: 'a'*64)
    api=CampaignNative(tmp_path,{})
    api.status=lambda _:dict(terminal=True,completed=False,failed_units=['references'],report_complete=True)
    runner=workflow.CampaignCoordinator(api,args(command='campaign',run_id='parent'),sleep=lambda _:None);runner.compact=lambda _:None
    with pytest.raises(ValueError,match='did not complete'):runner.execute()
    assert api.planned==['mechanism'] and runner.receipt['status']=='FAILED'
    assert (tmp_path/'runs/parent/campaign-receipt.json').is_file()


def test_calibration_requires_all_exact_arms_actual_cuda_and_64_commits(native,monkeypatch):
    api,_=native;w=api.prepare(args(phase='calibration'));p=api.protocol(w)
    monkeypatch.setattr(api,'binding',lambda *a,**k:{'decision':{'gate':'CALIBRATION','decision':'PASS','eligible':True}})
    monkeypatch.setattr(api,'load',lambda _:w);checked=[];monkeypatch.setattr(api,'check_gpu_tests',lambda path,protocol:checked.append(path))
    paths=[]
    for stage,u in api.units(w).items():
        if u['kind']=='calibration':
            path=api.stage_path(w,stage)/'calibration.json';paths.append(path)
            put(path,dict(native=True,device='cuda',status='COMPLETED_UPDATES',completed_updates=64,
                workload_sha256=u['workload_sha256'],seconds_per_update=.2,arm=u['arm']))
    calibrated=api.read_calibration(w['run_id'])
    assert len(checked)==2 and set(calibrated['data']['measurements'])==set(p['training']['arms'])
    learning=build_protocol('pilot',phase='learning',calibration=calibrated)
    assert all(t['allocation_status']=='FROZEN' for t in learning['trials'].values())
    row=json.loads(paths[0].read_text());row['completed_updates']=63;put(paths[0],row)
    with pytest.raises(ValueError,match='64 actual'):api.read_calibration(w['run_id'])
    row['completed_updates']=64;row['workload_sha256']='wrong-layout';put(paths[0],row)
    with pytest.raises(ValueError,match='exact declared workload'):api.read_calibration(w['run_id'])
    row['workload_sha256']=api.units(w)[paths[0].parent.name]['workload_sha256'];row['native']=False;put(paths[0],row)
    with pytest.raises(ValueError,match='64 actual'):api.read_calibration(w['run_id'])


def test_partial_submission_saves_intent_and_afterany_report_without_retry(native,monkeypatch):
    api,_=native;w=api.prepare(args(command='run'))
    import tdn.analysis.tangent_response.protocol as protocol
    monkeypatch.setattr(protocol,'validate_protocol',lambda *a,**k:None)
    monkeypatch.setattr(workflow.fw,'controller_policy',lambda:None)
    monkeypatch.setattr(workflow.shared,'check_capacity',lambda _: {'checked':'stub'})
    monkeypatch.setattr(workflow.rw,'acquire_venv_lock',lambda:nullcontext())
    monkeypatch.setattr(workflow.rw,'software_report',lambda _: {'software':'stub'})
    submitted=[]
    def command(command,**kwargs):
        submitted.append(command)
        if len(submitted)==2:raise OSError('explicit scheduler failure')
        return SimpleNamespace(stdout=str(100+len(submitted)))
    api.command=command
    with pytest.raises(OSError,match='scheduler failure'):api.submit(w)
    base=Path(w['run_dir']);jobs=json.loads((base/'jobs.json').read_text())
    assert [(j['stage'],j['job_id']) for j in jobs]==[('audit','101'),('report','103')]
    assert jobs[1]['dependency']=='afterany:101' and len(submitted)==3
    assert json.loads((base/'state/submission-intents/prepare.json').read_text())['status']=='SUBMITTING'
    with pytest.raises(ValueError,match='Unresolved submission intent'):api.terminal_jobs(w)


def test_worker_observer_spans_tests_and_terminal_seal(native,monkeypatch):
    api,_=native;w=api.prepare(args());stage='audit';base=Path(w['run_dir']);ctxpath=base/stage
    from tdn.analysis.tangent_response import report,telemetry
    order=[];attempt=base/'raven/runs/audit';attempt.mkdir(parents=True)
    monkeypatch.setenv('SLURM_JOB_ID','101');monkeypatch.setenv('SLURM_STEP_ID','0')
    monkeypatch.setattr(api,'verify_allocation',lambda *a:order.append('allocation'))
    monkeypatch.setattr(report,'start_attempt',lambda *a,**k:order.append('raven-start') or attempt)
    monkeypatch.setattr(report,'finish_attempt',lambda *a,**k:order.append('raven-finish'))
    monkeypatch.setattr(telemetry,'start_recording',lambda *a,**k:order.append('observer-start') or object())
    monkeypatch.setattr(telemetry,'stop_recording',lambda *a,**k:order.append('observer-stop') or {'status':'partial'})
    def allocated_worker(bound,unit,*,backend):
        backend.verify_allocation(bound,unit);backend.begin_report(bound,unit)
        order.extend(['correctness-tests','science-body'])
        ctxpath.mkdir(parents=True);put(ctxpath/'summary.json',{'status':'COMPLETED'})
        put(ctxpath/'science_manifest.json',{'science':'sealed'})
        put(api.state(w,stage),dict(status='COMPLETED',exit_code=0,source_sha256=w['source_sha256'],protocol_sha256=w['protocol_sha256'],slurm_job_id='101'))
        put(base/'state/audit-software.json',{'native':'stub'});put(base/'state/submission-software.json',{'native':'stub'})
        api.junit(w,stage).parent.mkdir(parents=True);api.junit(w,stage).write_text('<testsuite tests="1"/>')
        put(base/'audit-gpu-preflight.json',{'scope':'explicit stub, no hardware readiness claim'})
        backend.finish_report(bound,unit,attempt,runtime_seconds=2.)
        return 0
    monkeypatch.setattr(workflow.pw,'worker',allocated_worker)
    assert api.worker(w,stage)==0
    assert order==['allocation','raven-start','observer-start','correctness-tests','science-body','observer-stop','raven-finish']
    seal=json.loads((ctxpath/'workflow-seal.json').read_text())
    assert seal['slurm_job_id']=='101' and seal['slurm_profile_sha256']==w['slurm_profile_sha256']
    assert api.state(w,stage).relative_to(api.root).as_posix() in seal['artifacts']


def test_reporting_failure_invalidates_native_completion(native,monkeypatch):
    api,_=native;w=api.prepare(args());stage='audit';base=Path(w['run_dir']);ctxpath=base/stage
    from tdn.analysis.tangent_response import report,telemetry
    attempt=base/'raven/runs/audit';attempt.mkdir(parents=True)
    monkeypatch.setenv('SLURM_JOB_ID','101');monkeypatch.setattr(report,'start_attempt',lambda *a,**k:attempt)
    def failed_finish(*a,**k):raise OSError('report write failed')
    monkeypatch.setattr(report,'finish_attempt',failed_finish);monkeypatch.setattr(telemetry,'start_recording',lambda *a,**k:None)
    def allocated_worker(bound,unit,*,backend):
        backend.begin_report(bound,unit);ctxpath.mkdir();put(api.state(w,stage),{'status':'COMPLETED','exit_code':0});return 0
    monkeypatch.setattr(workflow.pw,'worker',allocated_worker)
    with pytest.raises(workflow.rw.WorkerFailure,match='reporting/seal failed'):api.worker(w,stage)
    assert json.loads(api.state(w,stage).read_text())['status']=='FAILED' and not (ctxpath/'workflow-seal.json').exists()
