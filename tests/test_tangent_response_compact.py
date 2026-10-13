"""Tangent export includes literal phase/upstream/recovery roots only."""
import hashlib
import importlib.util
import json
import getpass
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]


def put(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


@pytest.fixture
def exporter(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('tangent_compact_test',ROOT/'scripts/compact_review.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    root=tmp_path/'project with spaces';root.mkdir();monkeypatch.setattr(module,'ROOT',root);return module


def test_parent_selects_declared_present_phases_and_recovery_origins(exporter):
    root=exporter.ROOT;parent=root/'runs/parent';receipt=root/'runs/.tangent-response-campaigns/parent/receipt.json'
    put(parent/'tangent-response-campaign.json',dict(run_id='parent',receipt_path=str(receipt)))
    put(receipt,dict(schema='tdn.tangent-response-campaign/v1',run_id='parent',run_ids={'mechanism':'a','compression':'never-submitted'},replaced_run_ids={'mechanism':['origin']}))
    put(root/'runs/a/evidence.json',{'value':'accepted'});put(root/'runs/origin/evidence.json',{'value':'failed'});put(root/'runs/unrelated/evidence.json',{})
    initial,roots=exporter.run_roots('parent')
    assert initial==parent and {p.name for p in roots}=={'parent','a','origin'}


def test_exact_upstream_manifest_hash_checked_without_loading_code(exporter):
    root=exporter.ROOT;origin=root/'runs/upstream/tangent-response-workflow.json';put(origin,{'retained':'historical'})
    put(root/'runs/current/tangent-response-workflow.json',{'scientific_protocol':{'prerequisites':{'mechanism':{'workflow_path':str(origin),'workflow_sha256':hashlib.sha256(origin.read_bytes()).hexdigest()}}}})
    assert {p.name for p in exporter.run_roots('current')[1]}=={'current','upstream'}
    put(origin,{'changed':True})
    with pytest.raises(ValueError,match='binding changed'):exporter.run_roots('current')


def test_old_default_latest_pointer_is_unchanged(exporter):
    root=exporter.ROOT;put(root/'runs/.fedora-portfolio-latest.json',{'run_id':'old'});put(root/'runs/.fedora-tangent-response-latest.json',{'run_id':'new'})
    put(root/'runs/old/evidence.json',{});put(root/'runs/new/evidence.json',{})
    assert exporter.run_roots('latest')[0].name=='old'


def test_campaign_receipt_cannot_escape_exact_hidden_inventory(exporter):
    root=exporter.ROOT;receipt=root/'other/receipt.json';put(receipt,dict(schema='tdn.tangent-response-campaign/v1',run_id='parent',run_ids={}))
    put(root/'runs/parent/tangent-response-campaign.json',dict(run_id='parent',receipt_path=str(receipt)))
    with pytest.raises(ValueError,match='exact inventory'):exporter.run_roots('parent')


@pytest.fixture
def native_cli(tmp_path,monkeypatch):
    from scripts import tangent_response_workflow as workflow
    from tdn.analysis.tangent_response.protocol import build_protocol
    root=tmp_path/'native project with spaces';root.mkdir();api=workflow.NativeWorkflow(root)
    site=dict(schema_version=1,kind='desktop-slurm',root=str(root),user=getpass.getuser(),cpu_partition='local',gpu_partition='local',
        account=None,gpu_gres='gpu:1',expected_gpu_name='RTX 4090',gpu_vram_gib=24,torch_version='2.10.0+cu126',
        torch_wheel_index='https://download.pytorch.org/whl/cu126')
    run='tangent-response-child';base=root/'runs'/run;p=build_protocol('pilot',run_id=run)
    stored=dict(schema_version=1,kind='desktop-slurm-tangent-response',execution_mode='desktop-slurm',run_id=run,
        root=str(root),run_dir=str(base),profile='pilot',phase='mechanism',scientific_protocol=p,
        protocol_sha256=workflow.digest(p),slurm_profile=site,slurm_profile_sha256=workflow.digest(site),
        resources=api.resources(p,site),source_sha256='a'*64,source_tree_sha256='b'*64)
    put(base/workflow.MANIFEST,stored)
    parent='tangent-response-parent';receipt=root/'runs/.tangent-response-campaigns'/parent/'receipt.json'
    put(receipt,dict(schema='tdn.tangent-response-campaign/v1',run_id=parent,project_root=str(root),run_ids={'mechanism':run},status='COMPLETED',gates={}))
    put(root/'runs'/parent/workflow.CAMPAIGN,dict(run_id=parent,receipt_path=str(receipt)))
    monkeypatch.setattr(workflow,'NativeWorkflow',lambda:api)
    return workflow,api,parent,run


@pytest.mark.parametrize('identity',['campaign','child'])
@pytest.mark.parametrize('public_mode,exporter_mode',[('review','review'),('replay','full')])
def test_native_compact_main_forwards_exact_identity_and_exporter_mode(native_cli,monkeypatch,identity,public_mode,exporter_mode):
    workflow,api,parent,child=native_cli;selected=parent if identity=='campaign' else child;calls=[]
    monkeypatch.setattr(workflow.subprocess,'call',lambda command,**kwargs:calls.append((command,kwargs)) or 0)
    assert workflow.main(['compact',selected,'--mode',public_mode])==0
    assert calls==[([str(api.root/'.venv/bin/python'),str(api.root/'scripts/compact_review.py'),'pack',selected,'--mode',exporter_mode],{'cwd':api.root})]


@pytest.mark.parametrize('identity',['campaign','child'])
def test_native_compact_main_preserves_exporter_failure(native_cli,monkeypatch,identity):
    workflow,api,parent,child=native_cli;selected=parent if identity=='campaign' else child;calls=[]
    monkeypatch.setattr(workflow.subprocess,'call',lambda command,**kwargs:calls.append(command) or 23)
    assert workflow.main(['compact',selected,'--mode','replay'])==23
    assert len(calls)==1 and calls[0][-2:]==['--mode','full']
