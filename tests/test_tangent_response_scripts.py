"""Shell forwarding and Raven root checks use explicit command stubs."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def shell(tmp_path):
    root=tmp_path/'project with spaces';(root/'scripts').mkdir(parents=True)
    for name in ('fedora_tangent_response.sh','tangent_response_campaign.sh','tangent_response_archive.sh'):
        shutil.copyfile(ROOT/'scripts'/name,root/'scripts'/name)
    stub=root/'python stub';stub.write_text('#!/usr/bin/env python3\nimport json,os,sys\nfrom pathlib import Path\np=Path(os.environ["STUB_LOG"])\nwith p.open("a") as f:f.write(json.dumps({"args":sys.argv[1:],"snapshot":os.environ.get("TDN_TANGENT_CAPTURED_SCRIPT")})+"\\n")\nraise SystemExit(int(os.environ.get("STUB_EXIT","0")))\n');stub.chmod(0o755)
    env={k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','RAVEN_','TDN_'))}
    env.update(TDN_PYTHON=str(stub),STUB_LOG=str(root/'calls.jsonl'))
    return root,env


def invoke(shell,name,*args,**env):
    root,base=shell
    return subprocess.run(['bash',str(root/'scripts'/name),*args],env={**base,**env},text=True,capture_output=True)


def calls(shell):return [json.loads(line) for line in (shell[0]/'calls.jsonl').read_text().splitlines()]


def test_native_default_plan_and_explicit_fields_forward_with_spaces(shell):
    assert invoke(shell,'fedora_tangent_response.sh').returncode==0
    assert calls(shell)[0]['args']==[str(shell[0]/'scripts/tangent_response_workflow.py'),'plan']
    assert invoke(shell,'fedora_tangent_response.sh','run','--profile','pilot','--target','galerkin_same_grid','--run-id','fresh-id').returncode==0
    assert calls(shell)[1]['args'][-6:]==['--profile','pilot','--target','galerkin_same_grid','--run-id','fresh-id']


@pytest.mark.parametrize('name',['fedora_tangent_response.sh','tangent_response_campaign.sh','tangent_response_archive.sh'])
def test_underlying_failure_is_not_hidden(shell,name):
    assert invoke(shell,name,'--plan',STUB_EXIT='19').returncode==19


def test_campaign_forwards_only_one_fresh_coordinator(shell):
    assert invoke(shell,'tangent_response_campaign.sh','--profile','pilot','--run-id','parent').returncode==0
    assert calls(shell)[0]['args']==[str(shell[0]/'scripts/tangent_response_workflow.py'),'campaign','--profile','pilot','--run-id','parent']


def test_coordinator_rejects_outer_native_allocation(shell):
    assert invoke(shell,'tangent_response_campaign.sh','--run-id','parent',SLURM_JOB_ID='123').returncode==2
    assert not (shell[0]/'calls.jsonl').exists()


def test_workers_never_run_without_actual_scheduler_context(shell):
    assert invoke(shell,'fedora_tangent_response.sh','_worker-task').returncode!=0
    assert not (shell[0]/'calls.jsonl').exists()


def test_captured_raven_script_uses_original_project_root_not_snapshot_ancestors(shell):
    root,env=shell;captured=root/'.raven/launches/token/bin/captured.sh';captured.parent.mkdir(parents=True)
    shutil.copyfile(root/'scripts/tangent_response_campaign.sh',captured)
    plan=root/'.raven/launches/token/plan.json';plan.write_text(json.dumps({'project_root':str(root)}))
    outer=root/'runs/launch-token-a1';(outer/'logs').mkdir(parents=True)
    result=subprocess.run(['bash',str(captured),'--run-id','parent'],env={**env,'RAVEN_LAUNCH_PLAN':str(plan),'RAVEN_RUN_DIR':str(outer)},text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    observed=calls(shell)
    assert observed[0]['args']==[str(root/'scripts/tangent_response_raven.py'),'check','--snapshot',str(captured)]
    assert observed[1]['args'][0]==str(root/'scripts/tangent_response_workflow.py')
    assert observed[1]['snapshot']==str(captured)


def test_archive_exact_campaign_forwarding(shell):
    assert invoke(shell,'tangent_response_archive.sh','--campaign','parent','--plan').returncode==0
    assert calls(shell)[0]['args']==[str(shell[0]/'scripts/tangent_response_archive.py'),'--campaign','parent','--plan']


@pytest.fixture
def worker_transport(shell):
    """Only shell transport is stubbed; no allocation verifier/science runs."""
    root,env=shell;(root/'.venv/bin').mkdir(parents=True);shutil.copyfile(root/'python stub',root/'.venv/bin/python')
    (root/'.venv/bin/python').chmod(0o755)
    common=root/'scripts/common.sh'
    common.write_text('''TDN_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
printf '%s\\n' "$TDN_REPO_ROOT" >> "${STUB_COMMON_LOG:?}"
tdn_prepare_env() { mkdir -p -- "$TDN_REPO_ROOT/.cache"; }
''')
    stub_bin=root/'stub-bin';stub_bin.mkdir()
    srun=stub_bin/'srun'
    srun.write_text('''#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
with Path(os.environ['STUB_SRUN_LOG']).open('a') as stream:
    stream.write(json.dumps({'args':sys.argv[1:],'root':os.environ.get('TDN_PROJECT_ROOT')})+'\\n')
# This is a transport fixture, not a scheduler. The Python target is a recorder;
# it never imports or invokes the production allocation verifier/science engine.
os.environ['SLURM_STEP_ID']='0'
command=sys.argv[sys.argv.index('bash'):]
os.execvpe(command[0],command,os.environ)
''');srun.chmod(0o755)
    spool=root/'spool/job-fixture/slurm_script';spool.parent.mkdir(parents=True)
    shutil.copyfile(root/'scripts/fedora_tangent_response.sh',spool)
    workflow=root/'runs/transport-fixture/tangent-response-workflow.json';workflow.parent.mkdir(parents=True)
    workflow.write_text(json.dumps({'test_fixture':True,'native_readiness':False}))
    env.update(PATH=str(stub_bin)+os.pathsep+env['PATH'],STUB_COMMON_LOG=str(root/'common-roots.log'),
        STUB_SRUN_LOG=str(root/'srun-calls.jsonl'),TDN_PROJECT_ROOT=str(root),TDN_REPO_ROOT=str(root),
        TDN_EXECUTION_MODE='desktop-slurm',TDN_TANGENT_WORKFLOW=str(workflow),
        SLURM_JOB_ID='101',TDN_FEDORA_GPU_GRES='gpu:1')
    return root,env,spool


def transport_launch(fixture,entry='_worker-batch',**changes):
    root,env,spool=fixture;values={**env,**changes}
    values={key:value for key,value in values.items() if value is not None}
    return subprocess.run(['bash',str(spool),entry],env=values,text=True,capture_output=True,timeout=20)


@pytest.mark.parametrize('stage,device,cpus,gres',[('audit','cuda','4','gpu:1'),('decision','cpu','8','none'),('report','cpu','8','none')])
def test_spooled_batch_routes_original_checkout_through_task_transport(worker_transport,stage,device,cpus,gres):
    root,env,spool=worker_transport
    result=transport_launch(worker_transport,TDN_TANGENT_STAGE=stage,TDN_TANGENT_DEVICE=device,TDN_TANGENT_CPUS=cpus)
    assert result.returncode==0,result.stderr
    assert (root/'common-roots.log').read_text().splitlines()==[str(root),str(root)]
    srun=json.loads((root/'srun-calls.jsonl').read_text())
    assert srun['root']==str(root)
    assert '--gres='+gres in srun['args'] and '--cpus-per-task='+cpus in srun['args']
    assert srun['args'][-3:]==['bash',str(root/'scripts/fedora_tangent_response.sh'),'_worker-task']
    assert calls((root,env))==[{'args':[str(root/'scripts/tangent_response_workflow.py'),'worker',
        '--workflow',env['TDN_TANGENT_WORKFLOW'],'--stage',stage],'snapshot':None}]
    assert not (spool.parent.parent/'scripts').exists()


@pytest.mark.parametrize('invalid',['missing_project','missing_repo','relative','mismatch','symlink','parent_symlink'])
def test_spooled_worker_rejects_invalid_declared_roots_before_transport(worker_transport,invalid):
    root,env,spool=worker_transport;changes=dict(TDN_TANGENT_STAGE='report',TDN_TANGENT_DEVICE='cpu',TDN_TANGENT_CPUS='8')
    if invalid=='missing_project':changes['TDN_PROJECT_ROOT']=None
    if invalid=='missing_repo':changes['TDN_REPO_ROOT']=None
    if invalid=='relative':changes.update(TDN_PROJECT_ROOT='relative-project',TDN_REPO_ROOT='relative-project')
    if invalid=='mismatch':changes['TDN_REPO_ROOT']=str(root/'other')
    if invalid=='symlink':
        alias=root.parent/'project-alias';alias.symlink_to(root.name);changes.update(TDN_PROJECT_ROOT=str(alias),TDN_REPO_ROOT=str(alias))
    if invalid=='parent_symlink':
        alias=root.parent/'parent-alias';alias.symlink_to(root.parent);path=alias/root.name;changes.update(TDN_PROJECT_ROOT=str(path),TDN_REPO_ROOT=str(path))
    result=transport_launch(worker_transport,**changes)
    assert result.returncode!=0 and not (root/'common-roots.log').exists()
    assert not (root/'srun-calls.jsonl').exists() and not (root/'calls.jsonl').exists()


@pytest.mark.parametrize('invalid',['missing_job','wrong_mode','missing_step','nonnumeric_step'])
def test_spooled_worker_preserves_scheduler_and_mode_guards(worker_transport,invalid):
    root,env,spool=worker_transport;changes=dict(TDN_TANGENT_STAGE='report',TDN_TANGENT_DEVICE='cpu',TDN_TANGENT_CPUS='8');entry='_worker-batch'
    if invalid=='missing_job':changes['SLURM_JOB_ID']=None
    if invalid=='wrong_mode':changes['TDN_EXECUTION_MODE']='local_cpu'
    if invalid in ('missing_step','nonnumeric_step'):
        entry='_worker-task';changes['SLURM_STEP_ID']=None if invalid=='missing_step' else 'batch'
    result=transport_launch(worker_transport,entry,**changes)
    assert result.returncode!=0 and not (root/'srun-calls.jsonl').exists() and not (root/'calls.jsonl').exists()


def test_controller_help_keeps_source_root_despite_other_exported_root(shell):
    root,env=shell;result=invoke(shell,'fedora_tangent_response.sh','--help',TDN_PROJECT_ROOT=str(root/'other'),TDN_REPO_ROOT=str(root/'other'))
    assert result.returncode==0 and calls(shell)[0]['args']==[str(root/'scripts/tangent_response_workflow.py'),'--help']
