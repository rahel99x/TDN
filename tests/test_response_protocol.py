"""Work, scope, and opportunity contracts, without mocked scientific success."""
import copy
import subprocess
import sys
from pathlib import Path
import pytest
from tdn.analysis.response.protocol import build_protocol,validate_protocol,ancestors,GPU_TEST_CASES,FAMILIES,TINY_REGIMES

ROOT=Path(__file__).resolve().parents[1]

def calibration():
 return {'execution_mode':'desktop-slurm','verified_native_gpu':True,'seconds_per_update':{f+'/'+t:.1 for f in (*FAMILIES,'coupled_source') for t in ('fd_same_grid','galerkin_same_grid')}}

@pytest.mark.parametrize('profile',['smoke','pilot','development','confirmation'])
def test_profiles_bound_units_keep_separate_targets_and_failure_report(profile):
 p=build_protocol(profile)
 assert len(p['parents'])==len({x['parent_id'] for x in p['parents']})
 assert set(p['units']['report']['dependencies'])==set(p['units'])-{'report'}
 assert p['confirmation']['enabled']==(profile=='confirmation')
 for u in p['units'].values():
  assert u['seconds']<=1800 and u['mem_gib']<=48 and u['cpus']<=8
  assert u['device']!='cuda' or u['cpus']==4
 assert 'endpoint_tdn' not in {x['family'] for x in p['trials'].values()}

def test_smoke_runs_every_stage_but_cannot_promote():
 p=build_protocol('smoke',local_smoke=True)
 assert {u['kind'] for u in p['units'].values()} >= {'audit','fields','references','train','select','gate','compression','closure','evaluate','cost','aggregate','report','numerical_controls'}
 assert p['execution_scope']=='CPU_IMPLEMENTATION_SMOKE' and p['train_grid']==8
 assert p['response']['compression']['ranks']==[2,4,8]
 assert p['response']['closure']['noise_levels']==[0,1e-4,1e-3]
 assert not p['confirmation']['enabled']

def test_native_calibration_has_real_work_and_no_duplicate_family():
 p=build_protocol('smoke')
 trials=[t for t in p['trials'].values() if t['phase']=='factorial']
 assert len(trials)==8 and {t['updates'] for t in trials}=={64} and p['train_grid']==32
 assert len(TINY_REGIMES)==6 and 'precision_floor' not in TINY_REGIMES

@pytest.mark.parametrize('profile',['pilot','development','confirmation'])
def test_unbound_research_cannot_submit(profile):
 with pytest.raises(ValueError):validate_protocol(build_protocol(profile))

def test_every_declared_target_requires_its_own_native_calibration():
 p=build_protocol('pilot',calibration=calibration())
 validate_protocol(p)
 p['allowance']['calibrated_forecast']['seconds_per_update'].pop('direct_fno/fd_same_grid')
 with pytest.raises(ValueError,match='Every declared family'):validate_protocol(p)

def test_update_and_six_hour_work_allowance_are_frozen_in_exact_chunks():
 p=build_protocol('pilot',calibration=calibration(),trial_seconds={'temporal':21600},updates={'temporal':25000})
 temporal=[t for t in p['trials'].values() if t['family']=='temporal']
 assert {t['updates'] for t in temporal}=={25000}
 for t in temporal:
  chunks=[u for u in p['units'].values() if u.get('trial_id')==t['id']]
  assert sum(u['seconds'] for u in chunks)==21600 and len(chunks)==15
  for a,b in zip(chunks,chunks[1:]):assert b['previous_chunk']==a['id'] and a['id'] in b['dependencies']
 assert all(t['updates']!=25000 for t in p['trials'].values() if t['family']!='temporal')

@pytest.mark.parametrize('kwargs',[{'updates':{'oops':3}},{'updates':{'temporal':0}},{'trial_seconds':{'oops':3600}},{'chunk_seconds':1900}])
def test_invalid_work_bounds_refused(kwargs):
 with pytest.raises(ValueError):build_protocol('pilot',**kwargs)

def test_development_focused_ablations_keep_identity_and_precompute_cost_labels():
 p=build_protocol('development')
 variants={t['variant_id'] for t in p['trials'].values()}
 assert {'rank2','rank8','power_only','frozen_encoder','source_supervised','mean_variance_consistency','short_rollout','data-32','data-64','data-128','standard_capacity'}<=variants
 for u in p['units'].values():
  if u['kind']=='cost':
   assert u['data_stages'] and all(p['units'][x]['device']=='cpu' and p['units'][x]['grid']==u['grid'] for x in u['data_stages'])
 assert any(u.get('source_supervision') for u in p['units'].values())
 assert any(u.get('grid')==128 and u['kind']=='evaluate' for u in p['units'].values())
 assert all('::primary::n256' in x['candidate'] for x in p['evaluation']['comparisons'] if x['scope']=='primary')

@pytest.mark.parametrize('mutation',[lambda p:p['units']['audit'].update(seconds=2000),lambda p:p['units']['audit'].update(dependencies=['report']),lambda p:p.update(scientific_scope='FROZEN_CONFIRMATION')])
def test_scope_and_resource_tampering_refused(mutation):
 p=build_protocol();mutation(p)
 with pytest.raises(ValueError):validate_protocol(p)

def test_native_mandatory_gpu_cases_match_actual_collection():
 result=subprocess.run([sys.executable,'-m','pytest','--collect-only','-q','tests/test_response_gpu.py'],cwd=ROOT,text=True,capture_output=True,check=True)
 cases=[line.split('::',1)[1] for line in result.stdout.splitlines() if line.startswith('tests/test_response_gpu.py::')]
 assert len(cases)==54 and cases==GPU_TEST_CASES


def test_json_key_sorting_preserves_declared_dag_order():
 import json
 from tdn.analysis.response.protocol import plan_units
 p=build_protocol();sorted_copy=json.loads(json.dumps(p,sort_keys=True))
 validate_protocol(sorted_copy)
 assert [u['id'] for u in plan_units(sorted_copy)]==p['stages']


def test_focused_competitor_opportunities_and_dense_reference_bank():
 p=build_protocol('development')
 specs=list(p['trials'].values())
 for target in p['targets']:
  matched=[t for t in specs if t['target_kind']==target and t['variant_id']=='source_supervised']
  assert {t['family'] for t in matched}=={'coupled_source','residual_fno','direct_fno'}
  assert all(t['source_aux_weight']==1. for t in matched)
  assert all(t['model'].get('auxiliary_source_head') for t in matched if t['family']!='coupled_source')
  assert any(t['family']=='residual_fno' and t['variant_id']=='shared_gl4_anchor' and t['model']['anchor']=='shared_gl4' for t in specs)
  assert any(t['variant_id']=='rollout4' and t['rollout_steps']==4 for t in specs)
  for pair in p['evaluation']['comparisons']:
   if pair['scope']=='exploratory' and '::primary::' not in pair['candidate']:
    assert len(pair['seeds'])==1
 for unit in p['units'].values():
  if unit['kind']=='cost':
   refs=[p['units'][r] for r in unit['data_stages']]
   times=[h for ref in refs for h in ref['horizons']]
   assert times==unit['dense_times'] and len(times)==16
   assert all(len(ref['parent_ids'])==1 and len(ref['horizons'])==4 for ref in refs)
 assert p['response']['compression']['max_descendants_per_parent']==2


def test_grid_relative_stress_is_separate_descriptive_frozen_panel():
 p=build_protocol('development')
 stress=[u for u in p['units'].values() if u.get('panel')=='grid_relative_stress']
 assert len(stress)==16
 for u in stress:
  assert u['descriptive_only'] is True and u['grid'] in (64,128)
  if u['kind']=='references':assert len(u['parent_ids'])==1 and u['horizons']==[.02,.08]
  else:assert all(p['units'][r]['panel']=='grid_relative_stress' for r in u['data_stages'])


def test_native_fine_smoke_can_refine_and_source_teacher_cost_fits_declared_unit():
 native=build_protocol('smoke',targets=['galerkin_projected_fine'])
 local=build_protocol('smoke',local_smoke=True,targets=['galerkin_projected_fine'])
 assert native['reference']['max_grid']==4*native['train_grid']==128
 assert local['reference']['max_grid']==4*local['train_grid']==32
 p=build_protocol('development')
 for u in p['units'].values():
  if u['kind']=='references':
   assert u['reference_max_seconds_per_query']*u['requested_reference_queries']*u['source_teacher_calls_per_query']<=.8*u['seconds']+1e-9
 assert any(t['family']=='direct_fno' and t['variant_id']=='shared_physical_objective' and t['objective_mode']=='shared_physical' for t in p['trials'].values())
 assert any('direct_fno::shared_physical_objective::n256'==c['candidate'] and c['scope']=='exploratory' for c in p['evaluation']['comparisons'])


def test_cached_source_representation_arms_have_contained_component_controls():
 p=build_protocol('development');specs=list(p['trials'].values())
 for target in p['targets']:
  variants={t['variant_id']:t for t in specs if t['target_kind']==target and t['family']=='coupled_source'}
  for name in ('cached_spectral','cached_fixed_physical','cached_direct_interpolation','cached_learned_rates'):
   assert variants[name]['model']['source_decoder']=='cached_spectral'
   assert variants[name]['forecast_components']['anchor']==1.
  assert variants['cached_fixed_physical']['model']['source_representation']=='fixed_physical'
  assert variants['cached_direct_interpolation']['model']['source_representation']=='direct_interpolation'
  assert variants['cached_learned_rates']['model']['learned_time_rates']
  pairs=[c for c in p['evaluation']['comparisons'] if c['targets']==[target]]
  for name in ('cached_fixed_physical','cached_direct_interpolation','cached_learned_rates'):
   assert any(c['candidate']==f'coupled_source::{name}::n256' and c['comparator']=='coupled_source::cached_spectral::n256' and c['scope']=='exploratory' for c in pairs)


def test_clock_consistency_is_fd_only_with_clock_contained_component_control():
 p=build_protocol('development',targets=['fd_same_grid','galerkin_same_grid','galerkin_projected_fine'])
 arms=[t for t in p['trials'].values() if t['variant_id']=='clock_mean_consistency']
 assert len(arms)==1
 arm=arms[0]
 assert arm['family']=='coupled_source' and arm['target_kind']=='fd_same_grid'
 assert arm['model']['mean_parameterization']=='clock' and arm['consistency_weight']==.1
 for target in ('galerkin_same_grid','galerkin_projected_fine'):
  assert not any(t['target_kind']==target and t['variant_id'] in ('clock_mean','clock_mean_consistency') for t in p['trials'].values())
  assert {c['variant_id'] for c in p['unsupported_cells'] if c['target_kind']==target} >= {'clock_mean','clock_mean_consistency'}
 pairs=[c for c in p['evaluation']['comparisons'] if c['candidate']=='coupled_source::clock_mean_consistency::n256']
 assert len(pairs)==1
 assert pairs[0]['comparator']=='coupled_source::clock_mean::n256'
 assert pairs[0]['targets']==['fd_same_grid'] and pairs[0]['scope']=='exploratory' and pairs[0]['seeds']==[p['seeds'][0]]


def test_physical_interventions_are_held_out_and_budget_actual_descendants():
 p=build_protocol('development');units=[u for u in p['units'].values() if u.get('panel')=='physical_interventions']
 assert len(units)==32
 for u in units:
  assert u['descriptive_only'] is True
  if u['kind']=='references':
   assert u['split']=='development_test' and len(u['parent_ids'])==1 and u['phase_variants']==['original']
   assert u['requested_reference_queries']==2*len(u['intervention_variants'])
 assert {'numerical_remainder','fp64_assembly'}<={t['variant_id'] for t in p['trials'].values()}


def test_cost_partition_is_declared_architecture_only_and_aliases_are_unmeasured():
 from tdn.analysis.response.protocol import cost_representatives,dense_representatives
 p=build_protocol('development')
 for target in p['targets']:
  groups=cost_representatives(p,target)
  signatures=[g['representative']['architecture_sha256'] for g in groups]
  assert len(signatures)==len(set(signatures))
  assert {r['family'] for r in dense_representatives(p,target)}=={'direct_fno','residual_fno','temporal','coupled_source'}
  assert all(r['seed']==min(p['seeds']) and '::primary::n256' in r['method_id'] for r in dense_representatives(p,target))
  for group in groups:
   rep=group['representative']
   assert all(a['timing_status']=='NOT_MEASURED' and a['median_seconds'] is None for a in group['aliases'])
   assert not any((a['method_id'],a['seed'])==(rep['method_id'],rep['seed']) for a in group['aliases'])
  q=copy.deepcopy(p)
  q['trials']=dict(reversed(list(q['trials'].items())))
  for trial in q['trials'].values():
   trial['validation_loss']=-12345.
   trial['model']['normalization']={'arbitrary_unseen_constant':12345.}
  assert cost_representatives(q,target)==groups
 for unit in p['units'].values():
  if unit['kind']!='cost':continue
  assert unit['phase_variants']==['original']
  if unit['cost_workload']=='warm':assert 1<=len(unit['cost_models'])<=2
  elif unit['cost_workload']=='dense':assert len(unit['cost_models'])==4
  else:pytest.fail('Development must separate dense and warm workloads')
  assert all(p['units'][r]['phase_variants']==['original'] for r in unit['data_stages'])
 smoke=build_protocol('smoke',local_smoke=True)
 assert all(u['cost_workload']=='all' and len(u['cost_models'])==4 for u in smoke['units'].values() if u['kind']=='cost')


def test_artifact_plan_accounts_declared_reference_arrays_without_claiming_disk_bound():
 p=build_protocol('development')
 storage=p['budget_plan']['artifact_storage']
 refs=[u for u in p['units'].values() if u['kind']=='references']
 expected=0
 for u in refs:
  arrays=u['grid']**p['dimension']*(2*4+5*8)
  if u.get('source_supervision'):arrays+=3*u['grid']**p['dimension']*8+3*8
  expected+=u['requested_reference_queries']*arrays
 assert storage['declared_uncompressed_reference_array_bytes']==expected
 assert storage['declared_reference_queries']==sum(u['requested_reference_queries'] for u in refs)
 assert storage['checkpoint_continuation_units']==sum(u['kind']=='train' for u in p['units'].values())
 assert storage['worst_case_bytes'] is None and storage['actual_bytes'] is None
 assert storage['worst_case_status']=='UNAVAILABLE_NOT_A_FILESYSTEM_LIMIT'
 assert storage['unestimated_components'] and 'Every retained attempt' in storage['recovery_scope']
