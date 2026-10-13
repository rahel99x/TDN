"""Completed native chunk reuse is tested with generated CPU journals only."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from tdn.analysis.tangent_response import continuation as reuse,training
from tdn.analysis.tangent_response.contracts import digest
from tdn.analysis.tangent_response.protocol import build_protocol

ROOT=Path(__file__).resolve().parents[1]


def write(path,value):path.write_text(json.dumps(value)+'\n')


def hash_file(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(origin,p,stage,source):
    write(origin/'protocol.json',p)
    execution=dict(stage=stage,profile=p['profile'],source_tree_sha256=source,protocol_sha256=digest(p),device='cpu',execution_mode='local_cpu')
    write(origin/'execution.json',execution);write(origin/'summary.json',{'status':'COMPLETED','scope':'CPU test fixture'})
    manifest=dict(schema='tdn.tangent-response-science/v1',stage=stage,kind='training',profile=p['profile'],
        protocol_sha256=digest(p),unit_sha256=digest(p['units'][stage]),source_tree_sha256=source,
        artifacts={path.name:hash_file(path) for path in origin.iterdir() if path.is_file() and path.name not in ('science_manifest.json','COMPLETED')})
    write(origin/'science_manifest.json',manifest);(origin/'COMPLETED').write_text(hash_file(origin/'science_manifest.json')+'\n')


@pytest.fixture
def fitted(tmp_path,monkeypatch):
    old=torch.get_num_threads();torch.set_num_threads(1)
    monkeypatch.setattr(reuse,'ROOT',tmp_path)
    spec=importlib.util.spec_from_file_location('continuation_training_fixture',ROOT/'tests/test_tangent_response_training.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    data=module.samples();cfg=module.config(updates=2);cfg['source_tree_sha256']='a'*64
    origin=tmp_path/'origin';result=training.fit(deepcopy(data),cfg,output_path=origin)
    p=build_protocol('cpu-smoke');stage=next(s for s in p['stages'] if p['units'][s]['kind']=='training')
    seal(origin,p,stage,cfg['source_tree_sha256'])
    yield origin,tmp_path/'current',cfg,data,result,p,stage
    torch.set_num_threads(old)


def test_completed_work_copies_exact_bytes_without_optimizer_or_final_observations(fitted,monkeypatch):
    origin,current,cfg,data,trial,p,stage=fitted
    monkeypatch.setattr(training,'fit',lambda *a,**k:pytest.fail('No new optimizer or observation fit may run'))
    original={path.name:path.read_bytes() for path in origin.iterdir() if path.is_file()}
    current.mkdir();write(current/'execution.json',{'current':'engine metadata already exists'})
    result=reuse.reuse_completed_trial(origin,current,cfg,data,'cpu',cfg['source_tree_sha256'])
    assert result['continuation_noop'] is True and result['new_committed_updates']==0
    assert result['cost']==trial['cost'] and result['work']==trial['work'] and result['status']==trial['status']
    assert result['selected_checkpoint_sha256']==trial['selected_checkpoint_sha256']
    for name in reuse.COPY_ARTIFACTS:
        if name in original:assert (current/name).read_bytes()==original[name]
    assert not (current/'trial.json').exists() and not (current/'science_manifest.json').exists()
    assert json.loads((current/'execution.json').read_text())=={'current':'engine metadata already exists'}
    assert {path.name:path.read_bytes() for path in origin.iterdir() if path.is_file()}==original
    receipt=json.loads((current/'continuation_reuse.json').read_text())
    assert receipt['origin_trial_sha256']==hash_file(origin/'trial.json')


def test_partial_journal_returns_none_for_normal_exact_resume(fitted):
    origin,current,cfg,data,*_=fitted
    journal=torch.load(origin/'journal.pt',weights_only=True);journal['update']=1;torch.save(journal,origin/'journal.pt')
    assert reuse.reuse_completed_trial(origin,current,cfg,data,'cpu',cfg['source_tree_sha256']) is None
    assert not current.exists()


@pytest.mark.parametrize('change',['config','data','software','source','device'])
def test_completed_identity_mismatch_never_copies(fitted,monkeypatch,change):
    origin,current,cfg,data,*_=fitted;cfg=deepcopy(cfg);data=deepcopy(data);device='cpu';source='a'*64
    if change=='config':cfg['lr']=.007
    if change=='data':data[0]['u0']+=.001
    if change=='software':
        runtime=reuse.software_metadata();runtime['torch']='other-version';monkeypatch.setattr(reuse,'software_metadata',lambda:runtime)
    if change=='source':source='b'*64
    if change=='device':device='cuda'
    with pytest.raises(ValueError,match='identity changed'):reuse.reuse_completed_trial(origin,current,cfg,data,device,source)
    assert not current.exists()


def test_unsealed_or_changed_origin_cannot_reuse(fitted):
    origin,current,cfg,data,*_=fitted;(origin/'selected.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='artifact changed'):reuse.reuse_completed_trial(origin,current,cfg,data,'cpu','a'*64)
    assert not current.exists()


def test_missing_required_artifact_cannot_be_hidden_by_new_manifest(fitted):
    origin,current,cfg,data,_,p,stage=fitted;(origin/'criterion-diagnostic.pt').unlink();seal(origin,p,stage,'a'*64)
    with pytest.raises(ValueError,match='required sealed artifacts'):reuse.reuse_completed_trial(origin,current,cfg,data,'cpu','a'*64)


def test_existing_target_is_preserved_and_no_other_files_copy(fitted):
    origin,current,cfg,data,*_=fitted;current.mkdir();(current/'selected.pt').write_bytes(b'keep')
    with pytest.raises(ValueError,match='no overwrites'):reuse.reuse_completed_trial(origin,current,cfg,data,'cpu','a'*64)
    assert (current/'selected.pt').read_bytes()==b'keep' and not (current/'journal.pt').exists()


def test_source_and_target_symlinks_are_rejected(fitted):
    origin,current,cfg,data,*_=fitted;link=origin.parent/'linked-origin';link.symlink_to(origin.name)
    with pytest.raises(ValueError,match='symlinks'):reuse.reuse_completed_trial(link,current,cfg,data,'cpu','a'*64)
    current.symlink_to(origin.name)
    with pytest.raises(ValueError,match='symlinks'):reuse.reuse_completed_trial(origin,current,cfg,data,'cpu','a'*64)


def test_completed_counter_above_declared_work_is_not_new_success(fitted):
    origin,current,cfg,data,*_=fitted;journal=torch.load(origin/'journal.pt',weights_only=True);journal['update']=3;torch.save(journal,origin/'journal.pt')
    with pytest.raises(ValueError,match='exceeds'):reuse.reuse_completed_trial(origin,current,cfg,data,'cpu','a'*64)


def test_repeated_noop_preserves_original_cost_work_and_journal(fitted):
    origin,current,cfg,data,trial,p,stage=fitted
    first=reuse.reuse_completed_trial(origin,current,cfg,data,'cpu','a'*64)
    write(current/'trial.json',first);write(current/'training_results.json',{'trials':[first]});seal(current,p,stage,'a'*64)
    following=current.parent/'following';second=reuse.reuse_completed_trial(current,following,cfg,data,'cpu','a'*64)
    assert second['cost']==trial['cost'] and second['work']==trial['work']
    assert (following/'journal.pt').read_bytes()==(origin/'journal.pt').read_bytes()


def test_known_post_checkpoint_write_cost_is_preserved(fitted):
    origin,current,cfg,data,trial,*_=fitted;journal=torch.load(origin/'journal.pt',weights_only=True)
    assert trial['cost']['checkpoint_seconds']>=journal['cost']['checkpoint_seconds']
    result=reuse.reuse_completed_trial(origin,current,cfg,data,'cpu','a'*64)
    assert result['cost']['checkpoint_seconds']==trial['cost']['checkpoint_seconds']
