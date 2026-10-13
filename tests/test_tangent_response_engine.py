"""Science identity and retained-attempt behavior, independent of numerical benefit."""
from pathlib import Path
import copy
import json
import pytest

from tdn.analysis.tangent_response import engine
from tdn.analysis.tangent_response.protocol import build_protocol


def test_real_audit_seal_retains_cpu_scope_and_rejects_changed_bytes(tmp_path):
    p=build_protocol('cpu-smoke',worker_telemetry=())
    root=tmp_path/'audit'
    result=engine.execute('audit',p,root)
    assert result['status']=='COMPLETED'
    sealed=engine.verify_science(root,p)
    assert sealed['profile']=='cpu-smoke'
    assert json.loads((root/'audit.json').read_text())['native_readiness'] is False
    with pytest.raises(ValueError,match='exists'):
        engine.execute('audit',p,root)
    (root/'audit.json').write_text('{}\n')
    with pytest.raises(ValueError,match='artifact changed'):
        engine.verify_science(root,p)


def test_science_marker_cannot_substitute_other_manifest(tmp_path):
    p=build_protocol('cpu-smoke',worker_telemetry=())
    root=tmp_path/'audit';engine.execute('audit',p,root)
    (root/'COMPLETED').write_text('0'*64+'\n')
    with pytest.raises(ValueError,match='seal'):
        engine.verify_science(root,p)


def test_missing_ancestor_cannot_create_completed_measurement(tmp_path):
    p=build_protocol('cpu-smoke',worker_telemetry=())
    root=tmp_path/'references'
    with pytest.raises(ValueError,match='Missing declared ancestor'):
        engine.execute('references-000',p,root)
    assert not (root/'COMPLETED').exists()


def test_private_payload_rejects_arbitrary_objects_and_preserves_existing(tmp_path):
    path=tmp_path/'private.pt'
    with pytest.raises(TypeError,match='tensors and primitives'):
        engine.save_private(path,object())
    engine.save_private(path,{'records':[{'status':'UNRESOLVED'}]})
    assert engine.load_private(path)['records'][0]['status']=='UNRESOLVED'
    with pytest.raises(ValueError,match='Preserve'):
        engine.save_private(path,{'records':[]})


def test_native_body_cannot_launch_from_local_process(tmp_path,monkeypatch):
    p=build_protocol('pilot')
    monkeypatch.delenv('TDN_EXECUTION_MODE',raising=False)
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    monkeypatch.setattr(engine.torch.cuda,'is_available',lambda:True)
    with pytest.raises(RuntimeError,match='verified desktop Slurm'):
        engine.execute('audit',p,tmp_path/'native',device='cuda')
    assert not (tmp_path/'native').exists()


def test_calibration_observer_uses_verified_producer_hardware(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from tdn.analysis.tangent_response.protocol import training_workload
    p=build_protocol('pilot',phase='calibration')
    paths=[]
    for arm in p['training']['arms']:
        path=tmp_path/arm;path.mkdir()
        (path/'calibration.json').write_text(json.dumps({'arm':arm,'status':'COMPLETED_UPDATES',
            'native':True,'device':'cuda','completed_updates':64,'seconds_per_update':1.,
            'workload_sha256':training_workload(p,arm)}))
        paths.append((arm,path))
    monkeypatch.setattr(engine,'_paths',lambda ctx,kind:paths if kind=='calibration' else [])
    ctx=SimpleNamespace(protocol=p,device='cpu',sourcehash='a'*64)
    result=engine._decision(ctx,'CALIBRATION')
    assert result['decision']=='PASS' and result['eligible'] is True


def test_failure_observer_completes_without_promoting_missing_science(tmp_path):
    p=build_protocol('cpu-smoke',worker_telemetry=())
    root=tmp_path/'decision'
    result=engine.execute('decision',p,root)
    decision=json.loads((root/'decision.json').read_text())
    assert result['status']=='COMPLETED'
    assert decision['decision']=='IMPLEMENTATION_ONLY'
    assert decision['underlying_scientific_decision']=='UNRESOLVED'
    assert not decision['eligible']
    assert engine.verify_science(root,p)['unresolved_prerequisites']
