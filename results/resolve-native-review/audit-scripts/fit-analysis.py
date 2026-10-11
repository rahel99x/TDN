"""Post-hoc descriptive review of sealed native coefficient fits; no fitting."""
from pathlib import Path
from collections import defaultdict
import json,math,statistics
import numpy as np
import review_io as io
BASE=Path(__file__).resolve().parent

def group(rows,key):
 out=defaultdict(list)
 for r in rows:out[key(r)].append(r)
 return out

def avg(rows,key):
 return float(np.mean([np.mean([r[key] for r in rs]) for rs in group(rows,lambda r:r['field_cluster']).values()]))
def summary(rows):
 return dict(fields=len(set(r['field_cluster'] for r in rows)),queries=len(rows),
  mean_rms=avg(rows,'error_rms'),mean_maximum=avg(rows,'error_max'),worst_maximum=max(r['error_max'] for r in rows),
  mean_centered_rms=avg(rows,'centered_rms'),mean_absolute_mean_error=avg(rows,'mean_error'),
  mean_objective=avg(rows,'objective'), joint_tolerance_passes=sum(r['upper_rms']<=2e-5 and r['upper_max']<=2e-5 for r in rows),
  fields_all_horizons_pass=sum(all(r['upper_rms']<=2e-5 and r['upper_max']<=2e-5 for r in rs) for rs in group(rows,lambda r:r['field_cluster']).values()),
  below_1e6_count=sum(r['error_rms']<1e-6 for r in rows),
  interval_violations=sum(r['physical_interval_violations'] for r in rows),
  energy_increases=sum(r['energy_increase']>1e-12 for r in rows))
def pair(rows,ctl,cand='two_gain_direct',boot=True):
 key=lambda r:(r['field_cluster'],r['horizon'])
 a={key(r):r for r in rows if r['method']==ctl};b={key(r):r for r in rows if r['method']==cand}
 assert a.keys()==b.keys()
 fields=sorted({k[0] for k in a});fm=[]
 for f in fields:
  keys=[k for k in a if k[0]==f]
  fm.append((np.mean([a[k]['error_rms'] for k in keys]),np.mean([b[k]['error_rms'] for k in keys])))
 fm=np.array(fm);ratios=[a[k]['error_rms']/b[k]['error_rms'] for k in a]
 out=dict(control=ctl,candidate=cand,fields=len(fields),queries=len(a),ratio_of_field_means=float(fm[:,0].mean()/fm[:,1].mean()),
   geometric_query_ratio=float(np.exp(np.mean(np.log(ratios)))),field_wins=int((fm[:,0]>fm[:,1]).sum()),field_losses=int((fm[:,0]<fm[:,1]).sum()),
   query_wins=sum(x>1 for x in ratios),query_losses=sum(x<1 for x in ratios),min_query_ratio=min(ratios),max_query_ratio=max(ratios),
   uncertainty_adjusted_ratio=avg([dict(v,error_rms=max(v['error_rms']-v['reference_uncertainty_rms'],0)) for v in a.values()],'error_rms')/avg([dict(v,error_rms=v['error_rms']+v['reference_uncertainty_rms']) for v in b.values()],'error_rms'))
 if boot:
  rng=np.random.default_rng(19031);ix=rng.integers(0,len(fields),(10000,len(fields)))
  bs=fm[ix].mean(axis=1);values=bs[:,0]/bs[:,1]
  out['posthoc_field_bootstrap_95_ratio_interval']=np.quantile(values,[.025,.975]).tolist()
  out['interval_scope']='Post-hoc descriptive independent-field percentile bootstrap, 10000 resamples; frozen model only, not retraining or multiple-comparison adjusted'
 worst=sorted(a,key=lambda k:a[k]['error_rms']/b[k]['error_rms'])[:5]
 out['worst_relative_cases']=[dict(parent_id=a[k]['parent_id'],regime=a[k]['regime'],horizon=k[1],control_rms=a[k]['error_rms'],candidate_rms=b[k]['error_rms'],ratio=a[k]['error_rms']/b[k]['error_rms']) for k in worst]
 return out

out={}
for track in ['discrete','continuum']:
 fit=io.load(f'fit-{track}/fit_results.json');rows=io.load(f'fit-{track}/fit_endpoints.json')['rows'];basis=io.load(f'fit-{track}/fit_basis_cases.json')['rows']
 native=fit['validation'];methods=sorted(native)
 for method in methods:
  computed=summary([r for r in rows if r['method']==method]);assert math.isclose(computed['mean_rms'],native[method]['validation_rms'],rel_tol=1e-14)
  assert math.isclose(computed['mean_objective'],native[method]['validation_objective'],rel_tol=1e-14)
 controls=[m for m in methods if m!='two_gain_direct']
 groups={label:{str(k):{m:summary([r for r in rs if r['method']==m]) for m in methods} for k,rs in group(rows,selector).items()} for label,selector in [('by_regime',lambda r:r['regime']),('by_horizon',lambda r:r['horizon'])]}
 pairs={ctl:pair(rows,ctl) for ctl in controls}
 scopes={label:{str(k):{ctl:pair(rs,ctl,boot=False) for ctl in controls} for k,rs in group(rows,selector).items()} for label,selector in [('by_regime',lambda r:r['regime']),('by_horizon',lambda r:r['horizon'])]}
 basis_groups={}
 for split,bs in group(basis,lambda r:r['split']).items():
  basis_groups[split]={}
  for regime,rs in group(bs,lambda r:r['regime']).items():
   cosine=np.abs([r['quadratic_cubic_cosine'] for r in rs]); q=np.array([r['quadratic_rms'] for r in rs]);c=np.array([r['cubic_rms'] for r in rs])
   basis_groups[split][regime]=dict(queries=len(rs),fields=len(set(r['field_cluster'] for r in rs)),absolute_cosine_min_median_max=[float(x) for x in [cosine.min(),np.median(cosine),cosine.max()]],
    quadratic_rms_min_median_max=[float(x) for x in [q.min(),np.median(q),q.max()]],cubic_rms_min_median_max=[float(x) for x in [c.min(),np.median(c),c.max()]],
    both_bases_below_1e8=int(((q<1e-8)&(c<1e-8)).sum()))
 refs=[r for r in rows if r['method']=='two_gain_direct']
 basis_errors=[r for r in rows if r['fp32_assembly_max_difference'] is not None]
 direct=fit['fits']['two_gain_direct'];adam=fit['fits']['two_gain_adam']; eig=fit['train_gram']['eigenvalues'][0]
 out[track]=dict(source=f'fit-{track}/fit_results.json',fits={m:{k:v for k,v in values.items() if k!='curves'} for m,values in fit['fits'].items()},
  optimizer=dict(adam_minus_direct=adam['objective']-direct['objective'],adam_minus_lower=adam['objective']-direct['lower_bound'],direct_numerical_parameter_distance_upper_estimate=math.sqrt(direct['objective_gap']/eig),
    direct_gain_changes=[a-1 for a in direct['gains']],training_loss_reduction=(1-direct['objective']/fit['fits']['fixed_unit']['objective']),
    direct_training_ms=1000*direct['elapsed_seconds'],adam_training_ms=1000*adam['elapsed_seconds']),
  train_gram=fit['train_gram'],validation_gram=fit['validation_gram'],models={m:summary([r for r in rows if r['method']==m]) for m in methods},
  comparisons=pairs,grouped_models=groups,grouped_comparisons=scopes,basis_groups=basis_groups,
  reference=dict(accepted=sum(r['reference_accepted'] for r in refs),queries=len(refs),maximum_rms_uncertainty=max(r['reference_uncertainty_rms'] for r in refs),
    max_uncertainty_over_candidate_error=max(r['reference_uncertainty_rms']/r['error_rms'] for r in refs),
    assembly_difference_max=max(r['fp32_assembly_max_difference'] for r in basis_errors),
    assembly_difference_median=float(np.median([r['fp32_assembly_max_difference'] for r in basis_errors])),
    assembly_difference_over_error_max=max(r['fp32_assembly_max_difference']/r['error_rms'] for r in basis_errors)),
  stage_attempt_seconds=fit['attempt_costs']['observed_total_seconds'])
(BASE/'fit-review.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
for t,x in out.items():
 print(t,'OPT',x['optimizer'],'REF',x['reference'])
 for m,r in x['models'].items():print(m,round(r['mean_rms'],10),'pass',r['joint_tolerance_passes'],r['fields_all_horizons_pass'])
 for m,p in x['comparisons'].items():print(m,round(p['ratio_of_field_means'],4),p['field_wins'],p['query_wins'],'CI',p['posthoc_field_bootstrap_95_ratio_interval'])
 for axis,gs in x['grouped_comparisons'].items():
  for scope,cs in gs.items(): print(axis,scope,'fixed,GL2,GL4,nested',[round(cs[c]['ratio_of_field_means'],4) for c in ['fixed_unit','quad2_full','quad4_full','analytic_quad_cubic']])
 print('basis overlap',x['basis_groups']['validation']['overlapping_triads'])
