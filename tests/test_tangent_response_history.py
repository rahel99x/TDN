"""Historical byte verification does not authorize obsolete execution plans."""
from copy import deepcopy
import getpass
import json
from pathlib import Path

import pytest

from scripts import tangent_response_workflow as native
from tdn.analysis.tangent_response import engine,protocol


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


def old_builder(monkeypatch):
    current=protocol.build_protocol
    def older(*a,**k):
        # A real former default; no source files or CUDA/device checks change.
        k['_validate']=False
        result=current(*a,**k);result['mechanism']['max_source_nodes']=16;return result
    monkeypatch.setattr(protocol,'build_protocol',older)
    return current


def test_old_cpu_source_cap_remains_verifiable_but_cannot_execute_fresh(tmp_path,monkeypatch):
    current=old_builder(monkeypatch)
    p=protocol.build_protocol('cpu-smoke',worker_telemetry=())
    assert p['mechanism']['max_source_nodes']==16
    root=tmp_path/'old-audit';engine.execute('audit',p,root)
    old_source=json.loads((root/'execution.json').read_text())['source_tree_sha256']
    original={f.relative_to(root):f.read_bytes() for f in root.rglob('*') if f.is_file()}
    monkeypatch.setattr(protocol,'build_protocol',current)
    assert current('cpu-smoke')['mechanism']['max_source_nodes']==64
    with pytest.raises(ValueError,match='Frozen protocol'):engine.verify_science(root,p,sourcehash=old_source)
    assert engine.verify_science(root,p,sourcehash=old_source,historical=True)['protocol_sha256']==protocol.digest(p)
    assert engine.verify_science(root,sourcehash=old_source,historical=True)['source_tree_sha256']==old_source
    assert {f.relative_to(root):f.read_bytes() for f in root.rglob('*') if f.is_file()}==original
    with pytest.raises(ValueError,match='Frozen protocol'):engine.execute('audit',p,tmp_path/'fresh-obsolete')
    assert not (tmp_path/'fresh-obsolete').exists()
    with pytest.raises(ValueError,match='explicit sealed source'):engine.verify_science(root,p,historical=True)
    with pytest.raises(ValueError,match='source identity'):engine.verify_science(root,p,sourcehash='0'*64,historical=True)
    (root/'audit.json').write_text('{}\n')
    with pytest.raises(ValueError,match='artifact changed'):engine.verify_science(root,p,sourcehash=old_source,historical=True)


@pytest.mark.parametrize('change',['backward_dependency','zero_cpu','gpu_cpu','over_time','scope','report_roster','duplicate_unit'])
def test_historical_protocol_still_enforces_structure_and_resources(change):
    p=protocol.build_protocol('cpu-smoke');p['mechanism']['max_source_nodes']=16
    if change=='backward_dependency':p['units']['audit']['dependencies']=['report']
    if change=='zero_cpu':p['units']['audit']['cpus']=0
    if change=='gpu_cpu':p['profile']='pilot';p['execution_scope']='NATIVE_RESEARCH';p['phase']='mechanism';p['units']['audit'].update(device='cuda',cpus=8)
    if change=='over_time':p['units']['audit']['walltime']='00:36:00'
    if change=='scope':p['execution_scope']='NATIVE_RESEARCH'
    if change=='report_roster':p['units']['report']['dependencies']=[]
    if change=='duplicate_unit':p['stages'].append('report')
    with pytest.raises(ValueError):protocol.validate_protocol(p,canonical=False,require_binding=False)


def test_historical_validation_mode_cannot_authorize_bindings():
    p=protocol.build_protocol('pilot');p['mechanism']['max_source_nodes']=16
    with pytest.raises(ValueError,match='cannot authorize'):protocol.validate_protocol(p,canonical=False)


@pytest.fixture
def historical_native(tmp_path,monkeypatch):
    root=tmp_path/'native project';root.mkdir();api=native.NativeWorkflow(root)
    site=dict(schema_version=1,kind='desktop-slurm',root=str(root),user=getpass.getuser(),cpu_partition='local',gpu_partition='local',
        account=None,gpu_gres='gpu:1',expected_gpu_name='RTX 4090',gpu_vram_gib=24,torch_version='2.10.0+cu126',
        torch_wheel_index='https://download.pytorch.org/whl/cu126')
    p=protocol.build_protocol('pilot',run_id='old-native');p['mechanism']['max_source_nodes']=16
    base=root/'runs/old-native'
    w=dict(schema_version=1,kind='desktop-slurm-tangent-response',execution_mode='desktop-slurm',run_id='old-native',
        root=str(root),run_dir=str(base),phase='mechanism',profile='pilot',scientific_protocol=p,
        protocol_path=str(base/'protocol.json'),protocol_sha256=native.digest(p),slurm_profile=site,slurm_profile_sha256=native.digest(site),
        source_sha256='a'*64,source_tree_sha256='b'*64,resources=api.resources(p,site))
    put(base/native.MANIFEST,w);put(base/'protocol.json',p);put(base/'jobs.json',[])
    monkeypatch.setattr(native,'load_profile',lambda **k:site);monkeypatch.setattr(native.cw,'source_hash',lambda **k:'c'*64)
    return api,w


def test_native_readonly_load_status_use_frozen_protocol_not_current_defaults(historical_native):
    api,w=historical_native;loaded=api.load(api.workflow_path(w['run_id']),verify=False)
    assert loaded==w and api.status(loaded)['completed'] is False
    with pytest.raises(ValueError,match='Frozen protocol'):api.load(api.workflow_path(w['run_id']))
    with pytest.raises(ValueError,match='Frozen protocol'):api.binding(w['run_id'])
    with pytest.raises(ValueError,match='Frozen protocol'):api.recover(w['run_id'],'obsolete-recovery',plan=True)
    assert not (api.root/'runs/obsolete-recovery').exists()


def test_native_readonly_science_verifier_receives_explicit_historical_mode(historical_native,monkeypatch):
    api,w=historical_native;stage='decision';path=api.stage_path(w,stage)
    put(path/'execution.json',dict(stage=stage,profile=w['profile'],device='cpu',execution_mode='desktop-slurm'))
    calls=[]
    api.science_verifier=lambda path,p,**options:calls.append(options) or {'scope':'explicit controller stub'}
    api.verify_science(w,stage,historical=True)
    api.verify_science(w,stage)
    assert calls==[{'sourcehash':'b'*64,'historical':True},{'sourcehash':'b'*64}]
    received=[]
    monkeypatch.setattr(api,'verify_stage',lambda w,stage,**options:received.append(options))
    put(path/'decision.json',dict(decision='UNRESOLVED',eligible=False,gate='A'))
    assert api.decision(w,historical=True)['decision']=='UNRESOLVED'
    api.validate_results(w,historical=True)
    assert received and all(options=={'historical':True} for options in received)


def test_frozen_protocol_digest_remains_mandatory_for_historical_native(historical_native):
    api,w=historical_native;changed=deepcopy(w);changed['scientific_protocol']['mechanism']['max_source_nodes']=8
    put(api.workflow_path(w['run_id']),changed)
    with pytest.raises(ValueError,match='Frozen protocol changed'):api.load(api.workflow_path(w['run_id']),verify=False)
