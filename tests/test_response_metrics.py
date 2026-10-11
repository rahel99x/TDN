"""Independent decomposition, failure denominators and no imputed cost."""
import pytest
import torch
from tdn.analysis.response.metrics import endpoint_metrics,accuracy_status,uncertainty_interval,summarize_endpoints
from tdn.analysis.response.cost import break_even,paired_timing
from tdn.analysis.response.evaluation import inference_input


def record():
    return dict(parent_id='p',query_id='q',target_kind='fd_same_grid',grid=[4,4],lengths=[1.,1.],kappa=.003,reaction_rate=3.,horizon=.04,
        reference=dict(accepted=True,uncertainty_rms=1e-9,uncertainty_maximum=2e-9,status='ACCEPTED'))


def test_mean_centered_identity_and_absolute_failure_denominators():
    target=torch.full((1,1,4,4),.3,dtype=torch.float64);delta=torch.arange(16,dtype=torch.float64).reshape(1,1,4,4)*.001
    a=endpoint_metrics(target+delta,target,record=record())
    assert a['mean_centered_decomposition_residual']==pytest.approx(0,abs=1e-16)
    assert a['mean_error']==pytest.approx(.0075)
    bad=endpoint_metrics(torch.full_like(target,float('nan')),target,record=record())
    unresolved=endpoint_metrics(target,target,record=record()|{'reference':{'accepted':False}})
    summary=summarize_endpoints([a,bad,unresolved])
    assert summary['attempted_queries']==3 and summary['accepted_reference_queries']==2
    assert summary['prediction_failures_with_valid_reference']==1 and summary['unresolved_reference_queries']==1
    assert summary['tail_quantiles']['0.99'] is None


def test_uncertainty_straddles_threshold_instead_of_false_pass():
    reference=dict(accepted=True,uncertainty_rms=2e-6,uncertainty_maximum=2e-6)
    assert accuracy_status(1.9e-5,1.9e-5,reference)=='UNCERTAINTY_STRADDLES_TARGET'
    assert accuracy_status(1e-5,1e-5,reference)=='QUALIFIED'
    assert accuracy_status(3e-5,1e-5,reference)=='FAILED_TARGET'
    assert uncertainty_interval(1e-5,None)['upper'] is None


def test_missing_cost_is_unavailable_never_zero():
    assert break_even(None,1.,.001,.01,accuracy_equivalent=True)['status']=='NOT_MEASURED'
    assert break_even(10.,1.,.01,.001,accuracy_equivalent=True)['status']=='NO_BREAK_EVEN'
    assert break_even(10.,1.,.001,.01,accuracy_equivalent=True)['queries']==pytest.approx(1000)
    assert break_even(10.,1.,.001,.01,accuracy_equivalent=False)['status']=='INELIGIBLE'


def test_timing_records_paired_order_and_real_samples():
    x=torch.arange(10.)
    results=paired_timing({'a':lambda:x+1,'b':lambda:x.square()},repeats=3,warmup=1,seed=5)
    assert len(results['raw_rounds'])==6
    for i in range(3):assert {r['method'] for r in results['raw_rounds'] if r['round']==i}=={'a','b'}
    assert all(r['seconds']>0 for r in results['raw_rounds'])
    assert results['summaries']['a']['fresh_process_seconds'] is None
    assert results['p99'] is None


def test_inference_view_has_no_teacher_or_future_handles():
    sample={'u0':torch.zeros(1,1,4,4),'target':object(),'anchor_fp64':object(),
        'record':record()|{'future':object(),'reference':object(),'normalization_teacher':object()}}
    view=inference_input(sample)
    assert set(view)=={'u0','record'}
    assert not {'future','reference','normalization_teacher'}&set(view['record'])


def test_even_nyquist_interpolant_norm_differs_from_sampled_norm():
    x=(-1.)**torch.arange(4,dtype=torch.float64);err=x[None,None,:,None]*x[None,None,None,:]
    r=record()|{'target_kind':'galerkin_same_grid'}
    result=endpoint_metrics(err,torch.zeros_like(err),record=r)
    assert result['error_rms']==pytest.approx(1.)
    assert result['interpolant_error_rms']==pytest.approx(.5)
    assert result['centered_rms']==pytest.approx(.5)
    assert result['nodal_centered_rms']==pytest.approx(1.)
    assert abs(result['mean_centered_decomposition_residual'])<1e-15
    mixed=endpoint_metrics(err+.3,torch.zeros_like(err),anchor=torch.zeros_like(err),record=r)
    assert mixed['rms']**2==pytest.approx(mixed['mean_error']**2+mixed['centered_rms']**2)
    assert mixed['mean_fraction_squared_error']==pytest.approx(.09/.34)
    assert mixed['correction_rms']==pytest.approx(mixed['rms'])


def test_failed_prediction_retains_norm_for_registered_statistics():
    r=endpoint_metrics(None,torch.ones(1,1,4,4),record=record())
    assert r['norm']=='nodal_rms' and r['reference_accepted'] and not r['prediction_finite']


def test_bound_violation_uses_declared_rounding_slack():
    t=torch.ones(1,1,4,4,dtype=torch.float64)
    tiny=endpoint_metrics(t+1e-8,t,record=record())
    assert tiny['reference_bounded'] and tiny['interval_violation']>0 and not tiny['bound_violation']
    large=endpoint_metrics(t+1e-4,t,record=record())
    assert large['bound_violation']


@pytest.mark.parametrize('steps',[1,8])
def test_fresh_process_measures_real_classical_startup(tmp_path,steps):
    from tdn.analysis.response.evaluation import classical_methods
    from tdn.analysis.response.cost import fresh_process_probe
    inp={'u0':torch.full((1,1,4,4),.3),'record':record()}
    method=classical_methods(inp['record'],[steps])[f'analytic_df/steps-{steps}']
    expected=method(inp)
    result=fresh_process_probe(method,inp,tmp_path,'cpu',expected=expected)
    assert result['status']=='COMPLETED'
    assert result['parent_observed_total_seconds']>=result['first_prediction_seconds']>0
    assert result['reload_maximum_difference']<1e-7


def test_evaluation_keeps_each_schedule_seed_and_missing_model_denominator(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    from tdn.analysis.response import evaluation as ev
    ref=record()|{'split':'development_test','regime':'triads','phase_variant':'original'}
    sample={'u0':torch.full((1,1,4,4),.3),'target':torch.full((1,1,4,4),.31,dtype=torch.float64),
        'anchor_fp32':torch.full((1,1,4,4),.3),'record':ref,'reference_accepted':True}
    def no_truth(inp):
        assert set(inp)=={'u0','record'} and 'reference' not in inp['record']
        return inp['u0']+float(inp['record']['horizon'])*.01
    no_truth.metadata=dict(family='probe',method_id='probe::primary::n4',seed=1)
    def analytic(inp):return inp['u0']
    analytic.metadata=dict(family='analytic_df',method_id='analytic_df',seed=None)
    monkeypatch.setattr(ev,'load_stage_samples',lambda ctx:[sample])
    monkeypatch.setattr(ev,'load_selected_methods',lambda ctx,target:{'probe::primary::n4/seed-1':no_truth})
    monkeypatch.setattr(ev,'classical_methods',lambda *args:{'analytic_df/steps-1':analytic})
    monkeypatch.setattr(ev,'run_rollout_panel',lambda *args:[])
    monkeypatch.setattr(ev,'run_sensitivity_panel',lambda *args:[])
    monkeypatch.setattr(ev,'run_dense_panel',lambda *args:[])
    from tdn.analysis.response import diagnostics
    monkeypatch.setattr(diagnostics,'run',lambda *args:{})
    from tdn.analysis.response import inspection
    monkeypatch.setattr(inspection,'run',lambda *args:{})
    trial=lambda seed:dict(family='probe',phase='main',variant_id='primary',data_count=4,target_kind='fd_same_grid',seed=seed)
    protocol=dict(profile='development',seeds=[1,2],parent_counts={'train':4},trials={'a':trial(1),'b':trial(2)},
        evaluation={'frozen_schedules':[1,2,4,8]},units={'refs':{'parent_ids':['p'],'grid':4,'horizons':[.04]}},dimension=2)
    ctx=SimpleNamespace(path=tmp_path,protocol=protocol,unit={'target_kind':'fd_same_grid','data_stages':['refs'],'mechanism_ids':['M6']},
        prerequisites={},device='cpu',budget=SimpleNamespace(check=lambda:None),record=lambda *args,**kwargs:None)
    ev.run_evaluate(ctx)
    rows=json.loads((tmp_path/'endpoint_metrics.json').read_text())['rows']
    assert len(rows)==4*(1+2+1) # learned seed1 + analytic repeated2 + missing learned seed2
    assert {r['steps'] for r in rows}=={1,2,4,8}
    assert sum(r['status']=='NOT_MEASURED' for r in rows)==4
    registry=json.loads((tmp_path/'query_registry.json').read_text())
    assert registry['expected_seed_map']['probe::primary::n4']==[1,2]
    assert len(registry['query_roles'])==4
    assert len(json.loads((tmp_path/'spatial_examples.json').read_text())['rows'])==2


def test_learned_cost_schedules_count_complete_autonomous_calls():
    from tdn.analysis.response.cost import scheduled_methods
    calls=[]
    def step(sample):
        calls.append(float(sample['record']['horizon']))
        return sample['u0']+sample['record']['horizon']
    step.metadata=dict(method_id='probe',family='probe',seed=1)
    methods=scheduled_methods({'probe/seed-1':step},[1,2,4,8])
    sample={'u0':torch.zeros(1,1,4,4),'record':record()}
    answer=methods['probe/seed-1/steps-4'](sample)
    assert calls==[.01]*4
    torch.testing.assert_close(answer,torch.full_like(answer,.04))


def test_tolerance_frontiers_do_not_impute_cost_or_promote_ambiguous_accuracy():
    from tdn.analysis.response.cost import accuracy_cost_frontiers
    base=dict(parent_id='p',query_id='q',target_kind='fd_same_grid',grid=[4,4],seed=None,steps=1,norm='nodal_rms',
        reference_accepted=True,uncertainty_rms=1e-6,uncertainty_maximum=1e-6,timing_status='COMPLETED',measured_rounds=20,requested_rounds=20)
    analytic=base|dict(method='analytic_df',model_id='analytic_df/steps-1',family='analytic_df',rms=8e-6,error_max=9e-6,median_seconds=.002)
    learned=base|dict(method='coupled_source',model_id='coupled_source/seed-1/steps-1',family='coupled_source',rms=8e-6,error_max=9e-6,median_seconds=.001)
    unavailable=base|dict(method='missing',model_id='missing/steps-1',family='missing',rms=0.,error_max=0.,median_seconds=None)
    offline=dict(family_training_tuning_seconds={'coupled_source':10.},shared_preparation_selection_seconds=2.,missing=[])
    result=accuracy_cost_frontiers([analytic,learned,unavailable],[5e-6,1e-5],offline)
    assert all(row['selected'] is None for row in result['rows'] if row['tolerance']==5e-6)
    assert next(row for row in result['rows'] if row['tolerance']==1e-5 and row['category']=='all')['selected']['method']=='coupled_source'
    amort=result['amortization_rows'][0]
    assert amort['conditional_representative_break_even']['queries']==pytest.approx(12000.)
    assert amort['complete_campaign_break_even_queries'] is None
    unresolved=accuracy_cost_frontiers([learned|dict(reference_accepted=False)],[1e-5],offline)
    assert not unresolved['amortization_rows']


def test_offline_accounting_deduplicates_continuations_and_keeps_setup(tmp_path):
    import json
    from types import SimpleNamespace
    from tdn.analysis.response.cost import offline_accounting
    paths={}
    for i,total in enumerate([10.,20.]):
        path=tmp_path/str(i);path.mkdir();paths[str(i)]=path
        (path/'trial.json').write_text(json.dumps(dict(trial_id='same',family='probe',cost={'total_seconds':total},
            stage_setup_seconds=2.,postprocessing_cost={'reload':1.})))
    ctx=SimpleNamespace(prerequisites=paths,protocol={'units':{}})
    result=offline_accounting(ctx)
    assert result['family_training_tuning_seconds']['probe']==26.
    assert result['accounted_total_seconds']==26.
    assert not result['complete_campaign_cost_available']


def test_cost_warm_partition_restricts_roster_phase_and_cold_schedule(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    from tdn.analysis.response import evaluation as ev,cost
    calls=[];cold=[]
    def model(sample):
        assert sample['record']['query_id']=='original'
        calls.append(sample['record']['horizon']);return sample['u0']
    model.metadata=dict(family='probe',method_id='probe::primary::n4',seed=1)
    sample=dict(u0=torch.full((1,1,4,4),.3),target=torch.full((1,1,4,4),.3,dtype=torch.float64),
        record=record()|dict(query_id='original',split='validation',phase_variant='original'))
    twin={**sample,'record':sample['record']|dict(query_id='twin',phase_variant='phase_twin')}
    monkeypatch.setattr(ev,'load_stage_samples',lambda *args,**kwargs:[sample,twin])
    def load(ctx,target,selection_filter=None):
        assert selection_filter=={('probe::primary::n4',1),('missing::primary::n4',1)}
        return {'probe::primary::n4/seed-1':model}
    monkeypatch.setattr(ev,'load_selected_methods',load)
    monkeypatch.setattr(ev,'classical_methods',lambda *args:{})
    monkeypatch.setattr(cost,'_run_performance',lambda *args:pytest.fail('Warm partition must not repeat dense benchmarks'))
    def fresh(method,*args,**kwargs):
        cold.append(method.metadata['steps']);return dict(status='COMPLETED',parent_observed_total_seconds=.1)
    monkeypatch.setattr(cost,'fresh_process_probe',fresh)
    protocol=dict(cost={'repeats':2,'warmup':0,'fresh_process':True},evaluation={'frozen_schedules':[1,2,4,8],'tolerances':[2e-5]},units={})
    ctx=SimpleNamespace(path=tmp_path,protocol=protocol,unit=dict(target_kind='fd_same_grid',grid=4,cost_workload='warm',
        cost_models=[dict(method_id='probe::primary::n4',seed=1),dict(method_id='missing::primary::n4',seed=1)]),
        prerequisites={},device='cpu',budget=SimpleNamespace(check=lambda:None),record=lambda *args,**kwargs:None)
    cost.run_cost(ctx)
    rows=json.loads((tmp_path/'timing.json').read_text())['rows']
    assert len(rows)==4 and {r['steps'] for r in rows}=={1,2,4,8}
    assert cold==[1]
    assert all(r['fresh_process_seconds'] is None for r in rows if r['steps']>1)
    coverage=json.loads((tmp_path/'timing_coverage.json').read_text())['rows'][0]
    assert len(coverage['missing_selected_methods'])==4
    assert coverage['representative_only'] and coverage['alias_timing_status'].startswith('NOT_MEASURED')


def test_cost_dense_partition_skips_endpoint_sweep_and_cold_start(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    from tdn.analysis.response import evaluation as ev,cost
    sample=dict(u0=torch.full((1,1,4,4),.3),target=torch.full((1,1,4,4),.3,dtype=torch.float64),
        record=record()|dict(split='validation',phase_variant='original'))
    monkeypatch.setattr(ev,'load_stage_samples',lambda *args,**kwargs:[sample])
    monkeypatch.setattr(ev,'load_selected_methods',lambda *args,**kwargs:{})
    monkeypatch.setattr(cost,'paired_timing',lambda *args,**kwargs:pytest.fail('Dense stage must not run ordinary endpoint timing'))
    monkeypatch.setattr(cost,'fresh_process_probe',lambda *args,**kwargs:pytest.fail('Dense stage must not run cold-start sweep'))
    ctx=SimpleNamespace(path=tmp_path,protocol={'cost':{'repeats':20},'units':{}},unit=dict(target_kind='fd_same_grid',grid=4,
        cost_workload='dense',cost_models=[dict(method_id='missing::primary::n4',seed=1)]),
        prerequisites={},device='cpu',budget=SimpleNamespace(check=lambda:None),record=lambda *args,**kwargs:None)
    result=cost.run_cost(ctx)
    assert result['models']==0 and result['missing_models']==['missing::primary::n4/seed-1']
    assert json.loads((tmp_path/'performance.json').read_text())['rows']==[]
    assert json.loads((tmp_path/'timing.json').read_text())['fresh_process_status']=='NOT_APPLICABLE'
