"""Exact ownership and byte-preserving archive tests with scheduler stubs."""
import getpass
import json
from pathlib import Path
import socket

import pytest

from scripts import tangent_response_archive as archive
from scripts.tangent_response_workflow import CAMPAIGN, MANIFEST, POINTER, process_identity


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


class Scheduler:
    def __init__(self,state='CANCELLED',edit=None):self.state=state;self.edit=edit;self.calls=0
    def observe(self,jobs,*,user,root,run_ids):
        self.calls+=1
        if self.edit:self.edit(self.calls)
        return {j['job_id']:dict(state=self.state,user=user,workdir=str(root),job_name=j['expected_name'],comment=j['expected_comment'],account='') for j in jobs}


@pytest.fixture
def campaign(tmp_path):
    root=tmp_path/'project with spaces';root.mkdir();parent='pilot-parent';names=['child-a','child-b']
    site=dict(schema_version=1,kind='desktop-slurm',root=str(root),user=getpass.getuser(),cpu_partition='local',
        gpu_partition='local',account=None,gpu_gres='gpu:1',expected_gpu_name='RTX 4090',gpu_vram_gib=24,
        torch_version='2.10.0+cu126',torch_wheel_index='https://download.pytorch.org/whl/cu126')
    receipt=root/'runs/.tangent-response-campaigns'/parent/'receipt.json'
    put(receipt,dict(schema='tdn.tangent-response-campaign/v1',run_id=parent,project_root=str(root),status='FAILED',created_at='2026-10-13T12:00:00+00:00',
        coordinator=dict(pid=2147483647,host=socket.gethostname(),start_ticks='999'),run_ids={'mechanism':names[0],'compression':names[1]},replaced_run_ids={}))
    put(root/'runs'/parent/CAMPAIGN,dict(run_id=parent,receipt_path=str(receipt)))
    put(root/'runs'/parent/'campaign-receipt.json',json.loads(receipt.read_text()))
    for number,name in enumerate(names):
        base=root/'runs'/name
        put(base/MANIFEST,dict(kind='desktop-slurm-tangent-response',execution_mode='desktop-slurm',root=str(root),run_id=name,run_dir=str(base),campaign_run_id=parent,
            slurm_profile=site,slurm_profile_sha256='a'*64,source_sha256='old-source',scientific_protocol={'units':{'audit':{'kind':'audit'}}}))
        put(base/'jobs.json',[dict(stage='audit',job_id=str(101+number))]);(base/'retained-checkpoint.pt').write_bytes(b'\0original\xff')
    put(root/'runs'/POINTER,dict(run_id=parent));put(root/'runs/.fedora-response-latest.json',dict(run_id='old-response'))
    put(root/'runs/unrelated/evidence.json',{'keep':True})
    return root,parent,receipt,site,names


def api(campaign,**kwargs):
    root,_,_,site,_=campaign
    return archive.TangentArchive(root,site=site,scheduler=kwargs.pop('scheduler',Scheduler()),**kwargs)


def test_plan_lists_exact_roots_without_mutation(campaign):
    root,parent,receipt,_,names=campaign;result=api(campaign).archive(plan=True)
    assert result['action']=='PLAN_ONLY' and result['submitted_jobs']==2
    assert {m['kind'] for m in result['moves']}=={'native_run','campaign_root','campaign_receipt_directory','tangent_latest_pointer'}
    assert receipt.is_file() and not (root/'runs/archived-tangent-response').exists()


def test_cancelled_and_partial_artifacts_are_all_preserved(campaign):
    root,_,_,_,_=campaign;before={p.relative_to(root):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    result=api(campaign).archive();dest=root/result['archive_directory']
    assert result['status']=='ARCHIVED'
    for move in result['moves']:
        source=Path(move['original']);target=dest/move['archived']
        assert not (root/source).exists()
        for original,raw in before.items():
            if original==source:assert target.read_bytes()==raw
            elif original.is_relative_to(source):assert (target/original.relative_to(source)).read_bytes()==raw
    assert (root/'runs/unrelated/evidence.json').is_file() and (root/'runs/.fedora-response-latest.json').is_file()


@pytest.mark.parametrize('state',['RUNNING','PENDING','COMPLETING','UNKNOWN'])
def test_actual_nonterminal_or_unknown_jobs_block(campaign,state):
    with pytest.raises(ValueError,match='terminal'):api(campaign,scheduler=Scheduler(state)).archive()
    assert campaign[2].is_file()


def test_alive_coordinator_blocks_terminal_job_archive(campaign):
    value=json.loads(campaign[2].read_text());value['coordinator']=process_identity();put(campaign[2],value)
    with pytest.raises(ValueError,match='stopped'):api(campaign).archive()


def test_receipt_live_status_blocks_even_dead_coordinator(campaign):
    value=json.loads(campaign[2].read_text());value['status']='RUNNING';put(campaign[2],value)
    with pytest.raises(ValueError,match='receipt must be terminal'):api(campaign).archive()


def test_other_campaign_pointer_and_original_source_remain_distinct(campaign):
    root,_,_,_,_=campaign;put(root/'runs'/POINTER,dict(run_id='new-campaign'))
    result=api(campaign).archive()
    assert (root/'runs'/POINTER).is_file() and result['native_source_fingerprints']=='PRESERVED_WITHOUT_CURRENT_SOURCE_COMPARISON'


def test_wrong_exact_association_blocks_unrelated_root(campaign):
    root,_,_,_,names=campaign;path=root/'runs'/names[1]/MANIFEST;w=json.loads(path.read_text());w['campaign_run_id']='another';put(path,w)
    with pytest.raises(ValueError,match='exact campaign'):api(campaign).archive()


def test_symlink_jobs_metadata_cannot_be_archived(campaign):
    root,_,_,_,names=campaign;path=root/'runs'/names[0]/'jobs.json';saved=path.with_name('retained.json');path.rename(saved);path.symlink_to(saved.name)
    with pytest.raises(ValueError,match='symlink'):api(campaign).archive()


def test_absent_never_submitted_child_is_named_and_not_invented(campaign):
    root,_,_,_,names=campaign;(root/'runs'/names[1]).rename(root/'preserved-fixture')
    result=api(campaign).archive();assert result['absent_run_ids']==[names[1]] and result['submitted_jobs']==1
    assert (root/'preserved-fixture').is_dir()


def test_metadata_change_during_final_scheduler_check_prevents_all_moves(campaign):
    def edit(call):
        if call==2:
            value=json.loads(campaign[2].read_text());value['changed']=True;put(campaign[2],value)
    with pytest.raises(ValueError,match='changed'):api(campaign,scheduler=Scheduler(edit=edit)).archive()
    assert campaign[2].is_file()


def test_move_failure_rolls_back_without_overwrite_or_deletion(campaign,monkeypatch):
    original=archive.transport.rename_noreplace;calls=0
    def rename(*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==2:raise OSError('injected failure')
        return original(*args,**kwargs)
    monkeypatch.setattr(archive.transport,'rename_noreplace',rename)
    with pytest.raises(OSError,match='injected'):api(campaign).archive()
    assert campaign[2].is_file() and (campaign[0]/'runs'/campaign[4][0]/'retained-checkpoint.pt').is_file()


def test_new_scheduler_live_census_catches_unrecorded_native_submission(monkeypatch):
    scheduler=archive.TangentScheduler()
    monkeypatch.setattr(scheduler,'command',lambda _: '999|tdn-tangent-response:child-a:audit:hash|tdn-tangent-response-audit|/project\n')
    with pytest.raises(ValueError,match='live submitted'):scheduler.observe([],user='owner',root=Path('/project'),run_ids=['child-a'])


def test_nonempty_foreign_comment_is_never_accepted(campaign):
    class Foreign(Scheduler):
        def observe(self,*args,**kwargs):
            rows=super().observe(*args,**kwargs);rows['101']['comment']='another-project';return rows
    with pytest.raises(ValueError,match='ownership'):api(campaign,scheduler=Foreign()).archive()


def raven(campaign,state='FAILED'):
    root,_,receipt,_,_=campaign;token='exact-token';name='launch-'+token+'-a1'
    value=json.loads(receipt.read_text());value['raven_launch']=dict(project_root=str(root),coordinator_run_id=name,coordinator_run_dir='runs/'+name,launch_token=token);put(receipt,value)
    base=root/'runs'/name
    put(base/'run.json',dict(schema='raven.run/v1',run_id=name,state=state,job_id=None,provenance={'script':'scripts/tangent_response_campaign.sh'},
        metadata={'automation':{'execution':'local','project_root':str(root),'plan':{'token':token,'project_root':str(root),'script_relative':'scripts/tangent_response_campaign.sh'}}}))
    put(base/'summary.json',dict(schema='raven.summary/v1',id=name,state=state));return base


def test_exact_outer_raven_records_are_checked_and_preserved(campaign):
    base=raven(campaign);before=(base/'run.json').read_bytes();result=api(campaign).archive()
    assert result['raven_coordinator']['action']=='PRESERVED' and (base/'run.json').read_bytes()==before


def test_live_outer_raven_run_prevents_all_native_moves(campaign):
    raven(campaign,'RUNNING')
    with pytest.raises(ValueError,match='matching terminal'):api(campaign).archive()
    assert campaign[2].is_file()


def test_outer_raven_allocation_cannot_masquerade_as_local_coordinator(campaign):
    base=raven(campaign);path=base/'run.json';value=json.loads(path.read_text());value['job_id']='999';put(path,value)
    with pytest.raises(ValueError,match='matching terminal'):api(campaign).archive()
