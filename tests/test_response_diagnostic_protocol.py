"""Diagnostic work contracts, native binding and failure-aware orchestration."""
import copy
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest

from tdn.analysis.response.protocol import build_protocol, validate_protocol, ancestors, GPU_TEST_CASES
from tdn.analysis.response.diagnostic_protocol import variants, workload_hash, DIAGNOSTIC_GPU_CASES
from tests.test_response_workflow import controller, plan, freeze, runner


def calibration(p):
    return dict(execution_mode='desktop-slurm',verified_native_gpu=True,
        variant_measurements={t['calibration_key']:dict(updates=64,seconds_per_update=.01,
            workload_sha256=t['calibration_workload_sha256']) for t in p['trials'].values()})


def test_cpu_matrix_covers_every_variant_and_target_without_native_claim():
    p=build_protocol('diagnostic-smoke',local_smoke=True)
    assert len(p['diagnostic']['roster'])==34
    assert len(p['trials'])==97
    assert {u['device'] for u in p['units'].values()}=={'cpu'}
    assert p['execution_scope']=='CPU_IMPLEMENTATION_SMOKE'
    assert p['diagnostic']['authorizes_development'] is False
    assert set(p['units']['report']['dependencies'])==set(p['units'])-{'report'}
    assert len(p['unsupported_cells'])==5
    assert all(t['updates']==2 for t in p['trials'].values())
    from tdn.analysis.response.training_control import control_config
    assert all(control_config(t)['requested_updates']==2 for t in p['trials'].values())
    for name in p['stages']:
        assert set(p['units'][name]['dependencies']) <= set(ancestors(p,name))


def test_native_exact_workload_calibration_and_changed_basis_rejected():
    smoke=build_protocol('diagnostic-smoke',targets=['fd_same_grid'])
    assert {t['updates'] for t in smoke['trials'].values()}=={64}
    assert all(t['screen_stage']=='screening' for t in smoke['trials'].values())
    assert smoke['training']['record_correction_diagnostics'] is True
    assert smoke['training']['public_curve_every']==40
    p=build_protocol('diagnostic',targets=['fd_same_grid'],calibration=calibration(smoke))
    validate_protocol(p)
    p['allowance']['calibrated_forecast']['verified_native_gpu']=False
    with pytest.raises(ValueError,match='native'):validate_protocol(p)
    p['allowance']['calibrated_forecast']=calibration(smoke)
    first=next(iter(p['trials'].values()))
    p['allowance']['calibrated_forecast']['variant_measurements'][first['calibration_key']]['workload_sha256']='0'*64
    with pytest.raises(ValueError,match='matching'):validate_protocol(p)


def test_full_matrix_has_three_seed_complete_primary_and_focused_ablations():
    smoke=build_protocol('diagnostic-smoke',targets=['fd_same_grid'])
    p=build_protocol('diagnostic',targets=['fd_same_grid'],calibration=calibration(smoke))
    primary=[t for t in p['trials'].values() if t['primary']]
    assert len(primary)==7*3*7
    assert len(p['trials'])==228
    assert all(t['screen_stage']=='screening' for t in p['trials'].values())
    for u in p['units'].values():
        h,m,s=map(int,u['walltime'].split(':'))
        assert 0<u['seconds']<=1800 and u['seconds']+120<=h*3600+m*60+s<=2100
    assert all(t['setting']['eps']==1e-12 for t in p['trials'].values() if t.get('objective_mode')=='physical')


def test_unbound_diagnostic_cannot_submit_or_relabel_historical_gate():
    p=build_protocol('diagnostic',targets=['fd_same_grid'])
    with pytest.raises(ValueError,match='calibration'):validate_protocol(p)
    p=build_protocol('diagnostic-smoke',local_smoke=True,targets=['fd_same_grid'])
    p['diagnostic']['authorizes_development']=True
    with pytest.raises(ValueError,match='authorize'):validate_protocol(p)
    with pytest.raises(ValueError,match='Local'):build_protocol('diagnostic',local_smoke=True)


def test_screen_candidate_order_parent_disjoint_and_component_budget():
    p=build_protocol('diagnostic-smoke')
    train={q['parent_id'] for q in p['parents'] if q['split']=='train'}
    validation={q['parent_id'] for q in p['parents'] if q['split']=='validation'}
    assert train.isdisjoint(validation)
    for target in p['targets']:
        for split,groups in p['screening']['candidate_parents'][target].items():
            assert len(groups)==6 and all(len(candidates)==2 for candidates in groups.values())
    b=p['reference']['projected_error_budget']
    assert max(a+c+b['spatial_tolerance'] for a,c in zip(b['temporal_tolerances'],b['temporal_tolerances'][1:]))<=p['reference']['tolerance']


def test_complete_cuda_manifest_cannot_be_filtered():
    p=build_protocol('diagnostic-smoke',local_smoke=True)
    assert len(p['gpu_test_cases'])==97 and p['gpu_test_cases']==GPU_TEST_CASES+DIAGNOSTIC_GPU_CASES
    p['gpu_test_cases'].pop()
    with pytest.raises(ValueError,match='97'):validate_protocol(p)


def test_observer_metrics_wait_afterany_and_worker_runs_new_suite(controller):
    for name in ('test_response_diagnostic_gpu.py',):
        (controller.ROOT/'tests'/name).write_text('def test_structure(): pass\n')
    workflow=plan(controller,profile='diagnostic-smoke')
    jobs=[{'stage':'audit','job_id':'1'},{'stage':'fields','job_id':'2'}]
    assert controller.job_dependency(workflow,'diagnostic-metrics',jobs)=='afterany:1:2'
    stage=next(s for s,u in controller.units(workflow).items() if u['kind']=='train')
    commands=controller.worker_commands(workflow,stage)
    command=next(c for name,c in commands if name=='gpu-tests')
    assert str(controller.ROOT/'tests/test_response_diagnostic_gpu.py') in command


def test_status_json_handles_partial_submission_and_pending(controller,monkeypatch,capsys):
    workflow=plan(controller,profile='diagnostic-smoke');base=freeze(controller,workflow)
    controller.cw.atomic_json(base/'jobs.json',[{'stage':'audit','job_id':'9001'}])
    monkeypatch.setattr(controller.cw,'scheduler_state',lambda job:'PENDING')
    result=controller.status(workflow,json_output=True)
    assert result['terminal'] is False and result['completed'] is False
    assert 'report' in result['failed_units']
    monkeypatch.setattr(controller.cw,'scheduler_state',lambda job:'FAILED')
    result=controller.status(workflow,json_output=True)
    assert result['terminal'] is True and result['report_complete'] is False
    assert 'audit' in result['failed_units']


def test_diagnostic_latest_pointer_does_not_overwrite_response(controller,monkeypatch):
    controller.cw.atomic_json(controller.ROOT/'runs'/controller.POINTER,{'run_id':'historical'})
    controller.cw.atomic_json(controller.ROOT/'runs/.fedora-response-diagnostic-latest.json',{'run_id':'new-diagnostic'})
    monkeypatch.setenv('TDN_RESPONSE_POINTER','.fedora-response-diagnostic-latest.json')
    assert controller.workflow_path('latest').parent.name=='new-diagnostic'
    assert controller.cw.read_json(controller.ROOT/'runs'/controller.POINTER)['run_id']=='historical'


def test_native_junit_validator_requires_new_cases_and_no_skips(controller):
    p=build_protocol('diagnostic-smoke')
    root=ElementTree.Element('testsuite',tests='97')
    for module,names in [('tests.test_response_gpu',GPU_TEST_CASES),('tests.test_response_diagnostic_gpu',DIAGNOSTIC_GPU_CASES)]:
        for name in names:ElementTree.SubElement(root,'testcase',classname=module,name=name)
    path=controller.ROOT/'complete.xml';ElementTree.ElementTree(root).write(path)
    assert controller.check_gpu_tests(path,protocol=p)['test_cases']==97
    ElementTree.SubElement(root[-1],'skipped');ElementTree.ElementTree(root).write(path)
    with pytest.raises(ValueError):controller.check_gpu_tests(path,protocol=p)


def test_recovery_reruns_completed_failure_observers(controller,monkeypatch):
    (controller.ROOT/'tests/test_response_diagnostic_gpu.py').write_text('def test_structure(): pass\n')
    origin=plan(controller,'observer-origin',profile='diagnostic-smoke');base=freeze(controller,origin)
    for name in ('audit','diagnostic-metrics'):
        (base/name).mkdir();(base/name/'workflow-seal.json').write_text('preserved sealed observer')
    controller.cw.atomic_json(base/'jobs.json',[{'stage':'audit','job_id':'1'},
        {'stage':'fields','job_id':'2'},{'stage':'diagnostic-metrics','job_id':'3'}])
    monkeypatch.setattr(controller.cw,'scheduler_state',lambda job:'FAILED' if job=='2' else 'COMPLETED')
    monkeypatch.setattr(controller,'verify_stage',lambda *a,**kw:{'execution':{'stage':'audit'}})
    monkeypatch.setattr(controller,'stage_cli',lambda:SimpleNamespace(verify_execution=lambda *a,**kw:{'execution':{'stage':'audit'}}))
    runner(controller,monkeypatch)
    assert controller.main(['recover','observer-origin','--run-id','observer-recovered'])==0
    recovered=controller.load(controller.workflow_path('observer-recovered'))
    assert set(recovered['recovery']['stage_paths'])=={'audit'}
    jobs=controller.cw.read_json(Path(recovered['run_dir'])/'jobs.json')
    assert 'diagnostic-metrics' in {r['stage'] for r in jobs}
    assert (base/'diagnostic-metrics/workflow-seal.json').read_text()=='preserved sealed observer'


def test_native_calibration_rejects_incomplete_postfit_measurements(controller,monkeypatch):
    args=controller.parser().parse_args(['plan','--diagnostic-smoke','--run-id','calibration-postfit','--targets','fd_same_grid'])
    origin=controller.prepare(args);base=freeze(controller,origin)
    monkeypatch.setattr(controller,'verify_stage',lambda *a,**kw:None)
    monkeypatch.setattr(controller,'check_gpu_tests',lambda *a,**kw:None)
    paths=[]
    for ident,trial in controller.scientific_protocol(origin)['trials'].items():
        stages=[s for s,u in controller.units(origin).items() if u.get('trial_id')==ident]
        last=max(stages,key=lambda s:controller.units(origin)[s]['chunk_index']);path=base/last
        path.mkdir();(path/'workflow-seal.json').write_text('parser fixture; no GPU claim')
        row=dict(status='COMPLETED_UPDATES',updates=64,accepted_training_queries=48,requested_training_queries=48,
            accepted_validation_queries=36,requested_validation_queries=36,total_cost_including_postprocessing_seconds=1.,
            correction_diagnostics=dict(status='MEASURED',queries=48,requested_queries=48),inference_cost=dict(status='MEASURED'))
        controller.cw.atomic_json(path/'trial.json',row)
        controller.cw.atomic_json(path/'execution.json',dict(execution_mode='desktop-slurm',device='cuda'))
        paths.append(path)
    assert len(controller.read_diagnostic_calibration(origin)['variant_measurements'])==34
    row=controller.cw.read_json(paths[0]/'trial.json');row['correction_diagnostics']['queries']=47
    controller.cw.atomic_json(paths[0]/'trial.json',row)
    with pytest.raises(ValueError,match='correction'):controller.read_diagnostic_calibration(origin)
    row['correction_diagnostics']['queries']=48;row['inference_cost']['status']='UNRESOLVED'
    controller.cw.atomic_json(paths[0]/'trial.json',row)
    with pytest.raises(ValueError,match='inference'):controller.read_diagnostic_calibration(origin)


@pytest.mark.parametrize('failure',['preflight','after_science','tower_startup'])
def test_native_worker_records_failure_even_outside_science_cli(controller,monkeypatch,failure):
    """Mock allocation lifecycle checks reporting; it provides no CUDA evidence."""
    import json
    from tdn.analysis.response import diagnostic_metrics as dm
    workflow=plan(controller,'worker-'+failure,profile='diagnostic-smoke');base=freeze(controller,workflow)
    stage=next(s for s,u in controller.units(workflow).items() if u['kind']=='train')
    monkeypatch.setenv('SLURM_JOB_ID','9001');monkeypatch.setenv('SLURM_STEP_ID','0')
    monkeypatch.setenv('TDN_RESPONSE_RAVEN_DIR','prior-context')
    monkeypatch.setattr(controller,'verify_allocation',lambda *a:None)
    def tower_begin(*args):
        if failure=='tower_startup':raise OSError('Tower startup failed')
        return base/'tower'
    monkeypatch.setattr(controller,'begin_report',tower_begin)
    monkeypatch.setattr(controller,'finish_report',lambda *a,**kw:None)
    (base/'logs'/(stage+'-9001.out')).write_text('Native mock worker failure; no GPU science ran\n')
    def allocated_worker(bound,unit,*,backend):
        backend.verify_allocation(bound,unit)
        record=dict(status='RUNNING',stage='startup',exit_code=0,slurm_job_id='9001',slurm_step_id='0')
        controller.cw.atomic_json(controller.pw.state_path(bound,unit),record)
        report=None
        try:
            report=backend.begin_report(bound,unit)
            if failure=='after_science':
                ctx=SimpleNamespace(path=base/unit,protocol=controller.scientific_protocol(bound),stage=unit,
                    unit=controller.units(bound)[unit],device='cuda')
                worker_source=json.loads((dm.attempt_path(ctx)/'run.json').read_text())['metadata']['source_tree_sha256']
                dm.start_attempt(ctx,job_id='9001',sourcehash=worker_source)
                ctx.path.mkdir()
                body=dict(status='COMPLETED',elapsed_seconds=1.25,experiment_count=0,scientific_outcome='Mock science body only')
                controller.cw.atomic_json(ctx.path/'summary.json',body)
                dm.finish_attempt(ctx,body)
                assert json.loads((dm.attempt_path(ctx)/'run.json').read_text())['state']=='RUNNING'
            raise ValueError('Mock '+failure+' failure')
        except Exception:
            controller.cw.atomic_json(controller.pw.state_path(bound,unit),dict(record,status='FAILED',exit_code=2,error=failure))
            if report is not None:backend.finish_report(bound,unit,report,state='FAILED',runtime_seconds=4.5,exit_code=2)
            raise
    monkeypatch.setattr(controller.pw,'worker',allocated_worker)
    with pytest.raises((ValueError,OSError)):controller.worker(workflow,stage)
    ctx=SimpleNamespace(path=base/stage,protocol=controller.scientific_protocol(workflow),stage=stage,device='cuda')
    attempt=dm.attempt_path(ctx)
    run=json.loads((attempt/'run.json').read_text());summary=json.loads((attempt/'summary.json').read_text())
    assert run['state']==summary['state']=='FAILED' and run['job_id']=='9001'
    assert summary['exit_code']==2
    assert summary['results']['science_body_status']==('COMPLETED' if failure=='after_science' else 'NOT_STARTED')
    assert run['metadata']['native_gpu_verification']=='NOT_INFERRED_FROM_REPORTING'
    assert controller.os.environ['TDN_RESPONSE_RAVEN_DIR']=='prior-context'
    if failure!='after_science':assert not (base/stage).exists()
