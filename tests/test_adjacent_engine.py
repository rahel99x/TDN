"""Scientific integrity tests use explicitly synthetic dispatch, not fake results."""
import json
import pytest
from tdn.analysis.adjacent import engine
from tdn.analysis.adjacent.protocol import build_protocol, digest, validate_protocol
from tdn.analysis.adjacent.core import check
from tdn.runtime.metadata import write_json

@pytest.fixture
def dispatch(monkeypatch):
    metadata={'source_tree_sha256':'a'*64,'python':'unit-test'}
    monkeypatch.setattr(engine,'software_metadata',lambda:metadata.copy())
    def run(ctx):
        ctx.record('synthetic-'+ctx.stage,['D01'],metrics={'unit_test_only':True},checks=[
            check('synthetic-math',True,True,'eq',category='math'),check('no-science',None,None)])
        write_json(ctx.path/'payload.json',{'unit_test_only':True,'stage':ctx.stage})
        return {'unit_test_only':True}
    monkeypatch.setattr(engine,'_dispatch',run)
    return metadata

def campaign(root,p,stages):
    paths={}
    for stage in stages:
        path=root/stage
        engine.run_stage(p,stage,path,prerequisites={k:paths[k] for k in p['units'][stage]['dependencies']})
        paths[stage]=path
    return paths

def test_sealed_exact_inventory_and_no_overwrite(tmp_path,dispatch):
    p=build_protocol();paths=campaign(tmp_path,p,p['units'])
    for stage,path in paths.items():
        manifest=engine.verify_science(p,path,source_tree_sha256='a'*64)
        assert manifest['unit_sha256']==digest(p['units'][stage])
    with pytest.raises(FileExistsError):engine.run_stage(p,'audit',paths['audit'])

@pytest.mark.parametrize('tamper',['bytes','extra','missing','symlink','marker'])
def test_tampering_fails(tmp_path,dispatch,tamper):
    p=build_protocol();base=campaign(tmp_path,p,['audit'])['audit']
    if tamper=='bytes':(base/'payload.json').write_text('{}')
    elif tamper=='extra':(base/'extra.json').write_text('{}')
    elif tamper=='missing':(base/'payload.json').unlink()
    elif tamper=='symlink':(base/'link').symlink_to(base/'payload.json')
    else:(base/'COMPLETED').write_text('wrong')
    with pytest.raises((ValueError,OSError)):engine.verify_science(p,base)

def test_mixed_lineages_rejected(tmp_path,dispatch):
    p=build_protocol()
    a=campaign(tmp_path/'a',p,['audit','D01','D05'])
    b=campaign(tmp_path/'b',p,['audit','D01','D05'])
    # Differing audit bytes establish genuinely distinct ancestry.
    # Synthetic identity is identical unless input payload differs, so use a
    # separately resealed random operational scientific artifact.
    (b['audit']/'payload.json').write_text('{"independent_fixture":true}')
    with pytest.raises(ValueError):
        engine.run_stage(p,'D02',tmp_path/'mixed',prerequisites={'audit':b['audit'],'D01':a['D01'],'D05':a['D05']})

def test_changed_software_fails(tmp_path,dispatch):
    p=build_protocol();base=campaign(tmp_path,p,['audit'])['audit'];dispatch['python']='changed'
    with pytest.raises(ValueError,match='software'):
        engine.run_stage(p,'D01',tmp_path/'next',prerequisites={'audit':base})

def test_report_missing_is_na(tmp_path,dispatch):
    p=build_protocol();path=tmp_path/'report'
    summary=engine.run_stage(p,'report',path,stage_failures={'pilot':{'status':'INTERRUPTED'}})
    assert summary['status']=='COMPLETED'
    assert engine.verify_science(p,path)['prerequisites']=={}

@pytest.mark.parametrize('profile',['smoke','development','full'])
def test_budget_separate_and_protected(profile):
    p=build_protocol(profile);validate_protocol(p)
    budget=p['adjacent_budget']
    assert budget['protected_exploration_seconds']/budget['discretionary_seconds']==.2
    assert sum(u['seconds'] for u in p['units'].values() if u['kind']=='explore')==budget['protected_exploration_seconds']
    assert sum(u['seconds'] for u in p['units'].values() if u['kind']=='train')==budget['architecture_seconds']
    assert max(int(u['walltime'].split(':')[1]) for u in p['units'].values())<=45
    assert budget['transfer_from_main_campaign']==0
    assert len(p['gpu_test_cases'])==92 and len(set(p['gpu_test_cases']))==92
    p['pilot']['updates']+=1
    with pytest.raises(ValueError):validate_protocol(p)

def test_full_cannot_run_on_cloud_cpu(tmp_path,dispatch,monkeypatch):
    monkeypatch.delenv('TDN_EXECUTION_MODE',raising=False)
    with pytest.raises(ValueError,match='native allocated'):
        engine.run_stage(build_protocol('full'),'audit',tmp_path/'no')
