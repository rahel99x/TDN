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
