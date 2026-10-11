"""Independent native Resolve fairness review; reads only verified canonical artifacts."""
from pathlib import Path
import collections,json,math,statistics,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import review_io as io
BASE=Path(__file__).resolve().parent
P=io.load('fairness-discrete-000/protocol.json')
parents={x['parent_id']:x for x in P['parents']}
reports=[io.load(n) for n in sorted(io.names()) if n.endswith('/fairness_results.json')]
trials=[]; selections=[]; metrics=[]; curves={}
for report in reports:
 stage=report['stage']
 for rec in report['records']:
  selections.append(dict(stage=stage,track=report['track'],seed=report['seed'],**{k:v for k,v in rec.items() if k not in ('trials','metric_rows')}))
  for trial in rec['trials']:trials.append(dict(stage=stage,**trial))
 for row in io.load(stage+'/fairness_metrics.json')['rows']:
  metrics.append(dict(seed=report['seed'],regime=parents[row['parent_id']]['regime'],**row))
 for row in io.load(stage+'/learning_curves.json')['rows']:curves.setdefault(row['trial_id'],[]).append(row)
def geo(values):return math.exp(statistics.mean(math.log(max(v,1e-300)) for v in values))
def span(values):return dict(minimum=min(values),median=statistics.median(values),maximum=max(values)) if values else None
def group(rows, keys):
 out=collections.defaultdict(list)
 for row in rows:out[tuple(row[k] for k in keys)].append(row)
 return out
def counts(rows,key):return dict(collections.Counter(r[key] for r in rows))
# Reconstruct selection and tiny diagnostics independently.
selection_checks=[]; tiny_checks=[]
for report in reports:
 for rec in report['records']:
  if not rec['trials']:continue
  eligible=[t for t in rec['trials'] if t['phase'] in ('long_update','equal_time') and t['checkpoint_validated'] and not t['failure'] and math.isfinite(t['validation_objective'])]
  winner=min(eligible,key=lambda t:(t['validation_objective'],t['trial_id']))
  selection_checks.append(dict(stage=report['stage'],family=rec['family'],matched=winner['trial_id']==rec['selected_trial_id']))
  for t in rec['trials']:
   if t['tiny_gate']:
    expected=t['validation_objective']<=.01 and t['validation_objective']/t['initial_validation_objective']<=.01
    tiny_checks.append(dict(trial_id=t['trial_id'],matched=expected==(t['tiny_gate']['verdict']=='GOOD')))
trial_groups=[]
for key,rows in group([dict(t,optimizer=t['optimizer_control']['id']) for t in trials],['track','family','phase','optimizer']).items():
 trial_groups.append(dict(zip(['track','family','phase','optimizer'],key),trials=len(rows),
  stopping=counts(rows,'stopping_reason'),selected=counts(rows,'selection_status'),
  updates=span([r['updates_completed'] for r in rows]),selected_updates=span([r['updates_selected'] for r in rows]),
  objective=span([r['validation_objective'] for r in rows]),seconds=span([r['training_seconds'] for r in rows]),
  clip_fraction=span([r['diagnostics']['gradient_clipped_fraction'] for r in rows]),
  tiny_verdicts=dict(collections.Counter(r['tiny_gate']['verdict'] for r in rows if r['tiny_gate'])),
  parameter_delta={g:span([r['parameter_movement'][g]['delta_l2'] for r in rows if g in r['parameter_movement']]) for g in ['mean_head','spatial_head','trunk_or_basis']}))
model_summary=[]
for (track,family),rows in group(selections,['track','family']).items():
 data=dict(track=track,family=family,seeds=len(rows),selection=counts(rows,'selection_status'))
 for split in ['train','validation']:
  values=[r['selected_metrics'][split] for r in rows if split in r['selected_metrics']]
  if values:data[split]={k:span([r[k] for r in values]) for k in ['objective','field_mean_rms','field_mean_absolute_mean_error','worst_spatial_maximum']}
 model_summary.append(data)
# Query-paired contrasts use repeated seeds/horizons, then independent-field means.
comparisons=[];regimes=[];horizons=[]
for track in ['discrete','continuum']:
 val=[r for r in metrics if r['track']==track and r['split']=='validation']
 mapped={(r['family'],r['seed'],r['parent_id'],r['horizon']):r for r in val}
 for candidate in ['fno_scaled','fno_mean','two_basis']:
  for control in ['df','quad2_full','quad4_full']:
   paired=[]
   for r in val:
    if r['family']!=candidate:continue
    c=mapped[(control,r['seed'],r['parent_id'],r['horizon'])]
    paired.append(dict(field=r['field_cluster'],seed=r['seed'],regime=r['regime'],horizon=r['horizon'],
      ratio=c['error_rms']/r['error_rms'],max_ratio=c['error_max']/r['error_max'],
      separated_win=c['error_rms']-c['reference_uncertainty_rms']>r['error_rms']+r['reference_uncertainty_rms'],
      candidate_error=r['error_rms'],control_error=c['error_rms']))
   byfield=group(paired,['field'])
   comparisons.append(dict(track=track,candidate=candidate,control=control,queries=len(paired),fields=len(byfield),
      geometric_control_over_candidate_rms=geo([r['ratio'] for r in paired]),
      arithmetic_control_over_candidate_rms=statistics.mean(r['control_error'] for r in paired)/statistics.mean(r['candidate_error'] for r in paired),
      field_geomean_wins=sum(geo([r['ratio'] for r in sub])>1 for sub in byfield.values()),
      reference_separated_query_wins=sum(r['separated_win'] for r in paired),
      worst_query_rms_regression=max(1/r['ratio'] for r in paired),
      uncertainty_scope='Descriptive validation-selected comparisons; no unbiased confirmation interval'))
   for reg,rr in group(paired,['regime']).items():
    regimes.append(dict(track=track,candidate=candidate,control=control,regime=reg[0],fields=len({r['field'] for r in rr}),queries=len(rr),
      geometric_control_over_candidate_rms=geo([r['ratio'] for r in rr]),
      arithmetic_control_over_candidate_rms=statistics.mean(r['control_error'] for r in rr)/statistics.mean(r['candidate_error'] for r in rr),
      separated_query_wins=sum(r['separated_win'] for r in rr),worst_rms_regression=max(1/r['ratio'] for r in rr)))
   for horizon,rr in group(paired,['horizon']).items():
    horizons.append(dict(track=track,candidate=candidate,control=control,horizon=horizon[0],fields=len({r['field'] for r in rr}),queries=len(rr),
      geometric_control_over_candidate_rms=geo([r['ratio'] for r in rr]),
      arithmetic_control_over_candidate_rms=statistics.mean(r['control_error'] for r in rr)/statistics.mean(r['candidate_error'] for r in rr)))
# Error energy / spectral decomposition is aggregate signal energy, not a sample CI.
error_energy=[]
for (track,family,split),rows in group(metrics,['track','family','split']).items():
 total=sum(r['error_rms']**2 for r in rows)
 error_energy.append(dict(track=track,family=family,split=split,queries=len(rows),
   mean_component_energy_fraction=sum(r['mean_error_rms']**2 for r in rows)/total,
   spectral_band_energy_fractions={label:sum(next(b['error_rms'] for b in r['spectral_bands'] if b['label']==label)**2 for r in rows)/total for label in ['low','middle','high']},
   interval_violation_queries=sum(r['physical_interval_violations']>0 for r in rows),
   energy_increase_queries=sum(r['energy_increase']>1e-12 for r in rows),
   maximum_reference_uncertainty=max(r['reference_uncertainty_rms'] for r in rows),
   floor_active_queries=sum(r['normalization_scale']==1e-12 for r in rows)))
# Conditioning derivation from the exact, orthogonal initial Fourier modes.
# S=h^3*r*variance*lambda_active*(r+lambda_active); normalized raw-output
# sensitivity is S/sqrt(DF-MSE-or-floor). This is not the full network Hessian.
conditioning=[]
for track in ['discrete','continuum']:
 for row in metrics:
  if row['track']!=track or row['family']!='fno_scaled' or row['seed']!=P['seeds'][0]:continue
  q=parents[row['parent_id']];n=32
  modes=[tuple(t['mode']) for t in q['field_terms']]
  assert len(set(modes))==len(modes) and all(max(map(abs,k))<n/2 for k in modes)
  lam=q['kappa']*sum(t['weight']**2*(4*n*n*sum(math.sin(math.pi*k/n)**2 for k in t['mode']) if track=='discrete' else 4*math.pi**2*sum(k*k for k in t['mode'])) for t in q['field_terms'])/sum(t['weight']**2 for t in q['field_terms'])
  h=row['horizon'];s=h**3*q['reaction_rate']*q['variance']*lam*(q['reaction_rate']+lam)
  sensitivity=s/math.sqrt(row['normalization_scale'])
  conditioning.append(dict(track=track,split=row['split'],parent_id=row['parent_id'],regime=q['regime'],horizon=h,active_rate=lam,physical_scale=s,
     normalized_raw_output_sensitivity=sensitivity,loss_curvature_scale=sensitivity**2,
     source='Derived in FP64 from exact declared Fourier coefficients and recorded normalization; no new trained-model execution'))
condition_groups=[]
for key,rows in group(conditioning,['track','split','regime']).items():
 condition_groups.append(dict(zip(['track','split','regime'],key),queries=len(rows),sensitivity=span([r['normalized_raw_output_sensitivity'] for r in rows])))
curve_summary=[]
for t in trials:
 cc=curves[t['trial_id']];loss=[r['train_loss'] for r in cc if r['train_loss'] is not None]; vv=[r['validation_loss'] for r in cc if r['validation_loss'] is not None]
 curve_summary.append(dict(stage=t['stage'],family=t['family'],track=t['track'],optimizer=t['optimizer_control']['id'],phase=t['phase'],
   trial_id=t['trial_id'],curve_rows=len(cc),train_min=min(loss),train_last=loss[-1],train_last50_median=statistics.median(loss[-50:]),
   initial_validation=vv[0],max_validation_over_initial=max(vv)/vv[0],last_validation_over_initial=vv[-1]/vv[0],
   selected_validation_over_initial=t['validation_objective']/t['initial_validation_objective'],
   caveat='Training curve is changing minibatch objective, not a fixed full-training loss'))
# Four post-hoc CPU arithmetic cross-checks of the derived scale, not new
# trained-model trials or a replacement for native timings.
import torch
from tdn.analysis.frontier.data import field_state
from tdn.analysis.advance.numerics import deployable_defect_scale
from tdn.numerics import Equation,Geometry
torch.set_num_threads(1)
crosschecks=[]
for track in ['discrete','continuum']:
 values=[a for a in conditioning if a['track']==track]
 for a in [min(values,key=lambda x:x['normalized_raw_output_sensitivity']),max(values,key=lambda x:x['normalized_raw_output_sensitivity'])]:
  q=parents[a['parent_id']];u=field_state(q,32).float()
  scale=float(deployable_defect_scale(u,a['horizon'],Equation(q['kappa'],q['reaction_rate']),Geometry((32,32),(1.,1.)),track=track))
  crosschecks.append(dict(track=track,parent_id=q['parent_id'],horizon=a['horizon'],analytic_fp64=a['physical_scale'],reconstructed_cpu_fp32=scale,relative_difference=abs(scale/a['physical_scale']-1)))
result=dict(schema='independent-resolve-fairness-review/v1',sources=[r['stage']+'/fairness_results.json' for r in reports],
 scope=dict(tracks=2,training_fields=64,validation_fields=32,paired_seeds=P['seeds'],grid=32,training_horizons=P['train_horizons'],validation_horizons=P['validation_horizons'],
  validation_selection_bias=True,fresh_confirmation=False,methods_fno='Local physical-backbone FNO adaptations; not authoritative paper reproduction'),
 totals=dict(stages=len(reports),trials=len(trials),phase_counts=counts(trials,'phase'),failures=sum(bool(t['failure']) for t in trials),
  training_seconds=sum(r['training_seconds'] for r in reports),optimizer_seconds=sum(r['optimizer_seconds'] for r in reports),
  primary_selected_fits=sum(r['selection_status']=='FITTED_CHECKPOINT' for r in selections),
  tiny_verdicts=dict(collections.Counter(t['tiny_gate']['verdict'] for t in trials if t['tiny_gate'])),
  actual_updates=sum(t['updates_completed'] for t in trials),examples_seen=sum(t['examples_seen'] for t in trials),
  metric_rows=len(metrics),curves=sum(len(c) for c in curves.values()),finite_metrics=sum(r['finite'] for r in metrics),accepted_references=sum(r['reference_accepted'] for r in metrics)),
 selection_checks=selection_checks,tiny_gate_checks=tiny_checks,model_summary=model_summary,trial_groups=trial_groups,
 selected_models=[{k:v for k,v in r.items() if k!='selected_metrics'} for r in selections],comparisons=comparisons,regime_comparisons=regimes,horizon_comparisons=horizons,
 error_energy=error_energy,conditioning=conditioning,conditioning_groups=condition_groups,
 conditioning_implementation_crosschecks=crosschecks,curve_summary=curve_summary)
(BASE/'fairness-review.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(result['totals'],indent=2))
print('model summary')
for r in model_summary:print(r['track'],r['family'],r['selection'],r['validation']['field_mean_rms'],r['validation']['objective'])
print('comparison')
for r in comparisons:print(r['track'],r['candidate'],r['control'],round(r['geometric_control_over_candidate_rms'],3),round(r['arithmetic_control_over_candidate_rms'],3),r['field_geomean_wins'])
