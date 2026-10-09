"""Portfolio scope, execution seals and interruption reuse; no mock scientific claims."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
import pytest
from tdn.analysis.portfolio.protocol import build_protocol

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('portfolio_cli_test',ROOT/'scripts/portfolio.py')
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)


def test_ancestors_are_dag_not_every_earlier_stage():
    p=build_protocol('smoke')
    assert cli.ancestors(p,'explore-X01')==('audit',)
    assert set(cli.ancestors(p,'train-A-000'))=={'audit','prepare'}
    assert set(cli.ancestors(p,'report'))==set(p['units'])-{'report'}
    assert cli.parse_prerequisites(['audit=x'],p,'explore-X01')=={'audit':Path('x')}
    for values in ([],['audit=x','audit=y'],['audit='],['audit=x','diagnose=y']):
        with pytest.raises(ValueError): cli.parse_prerequisites(values,p,'explore-X01')


@pytest.mark.parametrize('extra',[['--profile','full'],['--profile','smoke','--device','cuda']])
def test_local_full_and_cuda_rejected_before_writes(tmp_path,extra):
    target=tmp_path/'forbidden'
    env={k:v for k,v in os.environ.items() if not k.startswith(('TDN_','SLURM_'))}
    result=subprocess.run([sys.executable,str(ROOT/'scripts/portfolio.py'),'--stage','audit','--run-dir',str(target),
        '--local-root',str(ROOT),*extra],env=env,cwd=ROOT,capture_output=True,text=True)
    assert result.returncode!=0 and 'CPU smoke/development' in result.stderr
    assert not target.exists()


@pytest.fixture
def sealed(tmp_path,monkeypatch):
    engine=ModuleType('tdn.analysis.portfolio.engine')
    engine.verify_science=lambda *a,**kw:{'stage':'audit','source_tree_sha256':'source'}
    monkeypatch.setitem(sys.modules,engine.__name__,engine)
    protocol={'profile':'smoke'}
    for name,value in [('execution.json',{'stage':'audit','protocol_sha256':cli.digest(protocol),'software':{'source_tree_sha256':'source'}}),
        ('protocol.json',protocol),('stage.json',{'status':'COMPLETED'}),('science_manifest.json',{'fixture':True})]:
        (tmp_path/name).write_text(json.dumps(value))
    cli.seal_execution(tmp_path,protocol)
    return tmp_path,protocol


def test_bound_seal_and_source(sealed):
    path,p=sealed
    assert cli.verify_execution(path,p,source_tree_sha256='source')['science']['stage']=='audit'
    with pytest.raises(ValueError): cli.verify_execution(path,p,source_tree_sha256='wrong')
    with pytest.raises(ValueError): cli.verify_execution(path,{'profile':'full'})


@pytest.mark.parametrize('name',['execution.json','protocol.json','stage.json','science_manifest.json'])
def test_each_bound_file_is_verified(sealed,name):
    path,p=sealed;(path/name).write_text('{}')
    with pytest.raises(ValueError,match='evidence changed'): cli.verify_execution(path,p)


def test_resume_requires_exact_lineage_and_never_completed(tmp_path):
    p=build_protocol('smoke');stage='train-A-000';software={'python':'3.13','source_tree_sha256':'s'}
    lineage={'audit':{'run_dir':'audit','workflow_seal_sha256':'x'}}
    ex={'stage':stage,'profile':'smoke','protocol_sha256':cli.digest(p),'execution_mode':'local-cpu','device':'cpu','prerequisites':lineage,'software':software}
    (tmp_path/'execution.json').write_text(json.dumps(ex));(tmp_path/'stage.json').write_text('{"status":"INTERRUPTED"}')
    kwargs=dict(stage=stage,protocol=p,software=software,mode='local-cpu',device='cpu',lineage=lineage)
    assert cli.check_resume(tmp_path,**kwargs)==ex
    with pytest.raises(ValueError): cli.check_resume(tmp_path,**{**kwargs,'lineage':{}})
    with pytest.raises(ValueError): cli.check_resume(tmp_path,**{**kwargs,'software':{'source_tree_sha256':'changed'}})
    (tmp_path/'workflow-seal.json').write_text('{}')
    with pytest.raises(ValueError,match='Completed'): cli.check_resume(tmp_path,**kwargs)


def test_resume_fingerprint_keeps_attempt_and_checkpoint_bytes(tmp_path,monkeypatch):
    import tdn.runtime.storage as storage
    monkeypatch.setattr(storage,'contained_path',lambda p:Path(p))
    (tmp_path/'journal.json').write_text('{}');(tmp_path/'checkpoint.pt').write_bytes(b'one')
    first=cli.resume_fingerprint(tmp_path)
    (tmp_path/'checkpoint.pt').write_bytes(b'two')
    assert cli.resume_fingerprint(tmp_path)!=first
    (tmp_path/'bad').symlink_to(tmp_path/'checkpoint.pt')
    with pytest.raises(ValueError,match='symlink'): cli.resume_fingerprint(tmp_path)


def test_native_cli_requires_recorded_worker_binding(monkeypatch):
    monkeypatch.delenv('TDN_PORTFOLIO_WORKFLOW',raising=False)
    monkeypatch.delenv('TDN_PORTFOLIO_STAGE',raising=False)
    with pytest.raises(ValueError,match='worker binding'):
        cli.verify_native_binding('train-A-000','full','cuda')


def test_failed_stage_retains_bounded_operational_cost_as_unverified(tmp_path):
    (tmp_path/'stage.json').write_text(json.dumps({'status':'INTERRUPTED','elapsed_seconds':17.2,'slurm_job_id':'123'}))
    record=cli.failed_stage_diagnostic(tmp_path,'seal missing')
    assert record['verification']=='INVALID' and record['evidence_status']=='UNVERIFIED_DIAGNOSTIC'
    assert record['elapsed_seconds']==17.2 and record['status']=='INTERRUPTED' and record['slurm_job_id']=='123'
    assert record['operational_source']['sha256']==cli.file_digest(tmp_path/'stage.json')
    for invalid in (None,-1,True,float('nan'),float('inf')):
        (tmp_path/'stage.json').write_text(json.dumps({'elapsed_seconds':invalid}))
        assert 'elapsed_seconds' not in cli.failed_stage_diagnostic(tmp_path,'bad')
    assert cli.failed_stage_diagnostic(tmp_path/'absent','missing')['verification']=='MISSING'
