"""Independent descriptive audit of sealed native Advance data; no model changes."""
import json,math,statistics as st
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from review_io import load,BASE
p=load('aggregate/protocol.json'); rows=load('aggregate/endpoint_rows.json')['rows']; locked=load('aggregate/locked_frontiers.json')['rows']; claims=load('aggregate/claims.json')
target=p['primary_target']
def eligible(r):return bool(r and r.get('finite') and r.get('reference_accepted') and r.get('upper_rms',math.inf)<=target and r.get('upper_max',math.inf)<=target)
def gmean(values):return float(np.exp(np.mean(np.log(values)))) if len(values) else None
def quantile(values,q):return float(np.quantile(values,q)) if len(values) else None
def boot(obs):
 d=defaultdict(list)
 for a,b,v in obs:d[(str(a),str(b))].append(v)
 f=sorted({a for a,b in d});s=sorted({b for a,b in d})
 out={'fields':len(f),'seeds':len(s),'rows':len(obs)}
 if any((a,b) not in d for a in f for b in s):return dict(out,status='INCOMPLETE_CROSSED_DESIGN',mean=None,lower=None,upper=None)
 if not d:return dict(out,status='NO_OBSERVATIONS',mean=None,lower=None,upper=None)
 x=np.array([[np.mean(d[(a,b)]) for b in s] for a in f]);rng=np.random.default_rng(73421);vals=[]
 for i in range(1000):
  ix=rng.integers(0,len(f),len(f));iy=rng.integers(0,len(s),len(s));vals.append(x[ix[:,None],iy[None,:]].mean())
 lo,hi=np.quantile(vals,[.025,.975]);return dict(out,status='DESCRIPTIVE_INTERVAL',mean=float(x.mean()),lower=float(lo),upper=float(hi))
primary=[r for r in rows if r['primary']]
lookup={(r['family'],r['track'],r['train_count'],r['seed'],r['parent_id'],r['grid'],r['schedule_id']):r for r in rows}
def matched(left,family,count=None):
 count=left['train_count'] if count is None else count
 return lookup.get((family,left['track'],count,left['seed'],left['parent_id'],left['grid'],left['schedule_id'])) or lookup.get((family,left['track'],0,None,left['parent_id'],left['grid'],left['schedule_id']))
def summary(rr):
 fields=defaultdict(list)
 for r in rr:fields[r['field_cluster']].append(eligible(r))
 return dict(rows=len(rr),fields=len(fields),eligible_rows=sum(map(eligible,rr)),all_queries_pass_fields=sum(all(v) for v in fields.values()),gmean_rms=gmean([r['error_rms'] for r in rr]),median_rms=st.median(r['error_rms'] for r in rr),p90_rms=quantile([r['error_rms'] for r in rr],.9),max_rms=max(r['error_rms'] for r in rr),max_error=max(r['error_max'] for r in rr),median_cost=st.median(r['cost_seconds'] for r in rr),mean_bias_mse_fraction=float(np.mean([(r['mean_error']/r['error_rms'])**2 for r in rr])),physical_violation_rows=sum(r['physical_interval_violations']>0 for r in rr),energy_increase_rows=sum(r['energy_increase']>1e-12 for r in rr),below_1e7_rows=sum(r['error_rms']<1e-7 for r in rr))
def group_summary(rr,keys):
 d=defaultdict(list)
 for r in rr:d[tuple(r[k] for k in keys)].append(r)
 return [dict(zip(keys,k),**summary(v)) for k,v in sorted(d.items())]
# Reconstruct each primary claim independently, including hierarchical pairing.
checks=[]
for c in claims['accuracy']:
 rr=[r for r in primary if r['family']=='two_basis' and r['train_count']==64 and r['track']==c['track'] and r['final_time']==c['final_time']]
 pp=[(a,matched(a,c['control_family'])) for a in rr];assert all(b for a,b in pp)
 assert all(a['reference_accepted'] and b['reference_accepted'] and a['finite'] and b['finite'] for a,b in pp)
 obs=[(a['field_cluster'],a['seed'],math.log(b['error_rms']/a['error_rms'])) for a,b in pp]; bi=boot(obs)
 assert all(abs(c['interval'][k]-bi[k])<1e-12 for k in ['mean','lower','upper'])
 ratio=math.exp(bi['mean']);coverage=(sum(eligible(a) for a,b in pp)-sum(eligible(b) for a,b in pp))/len(pp)
 verdict='GOOD' if ratio>=1.1 and bi['lower']>0 and coverage>=-.05 else 'BAD';assert verdict==c['verdict']
 fields=defaultdict(list)
 for a,b in pp:fields[a['field_cluster']].append(math.log(b['error_rms']/a['error_rms']))
 checks.append(dict(track=c['track'],time=c['final_time'],control=c['control_family'],ratio=ratio,ratio_ci=[math.exp(bi['lower']),math.exp(bi['upper'])],verdict=verdict,fields_improved=sum(np.mean(v)>0 for v in fields.values()),independent_fields=len(fields),coverage_difference=coverage,candidate_passes=c['candidate_passes'],control_passes=c['control_passes'],denominator=len(pp)))
# Reconstruct cost claims: mandatory coverage first, then eligible-pair contrast.
costchecks=[]
for c in claims['cost']:
 rr=[r for r in locked if r['family']=='two_basis' and r['train_count']==64 and r['track']==c['track'] and r['final_time']==c['final_time'] and r['rms_target']==target and r['max_target']==target]
 rhs=[r for r in locked if r['family']==c['control_family'] and r['train_count'] in(0,64) and r['track']==c['track'] and r['final_time']==c['final_time'] and r['rms_target']==target and r['max_target']==target]
 ix={(r['seed'],r['parent_id'],r['grid']):r for r in rhs};aok=defaultdict(list);bok=defaultdict(list);obs=[]
 for a in rr:
  b=ix.get((a['seed'],a['parent_id'],a['grid'])) or ix.get((None,a['parent_id'],a['grid']));assert b
  yes=a['status']=='ELIGIBLE';other=b['status']=='ELIGIBLE';aok[a['field_cluster']].append(yes);bok[a['field_cluster']].append(other)
  if yes and other:obs.append((a['field_cluster'],a['seed'],math.log(b['cost_seconds']/a['cost_seconds'])))
 ca=np.mean([all(v) for v in aok.values()]);cb=np.mean([all(v) for v in bok.values()]);bi=boot(obs)
 assert ca==c['candidate_joint_field_coverage'] and cb==c['control_joint_field_coverage']
 for k in ['mean','lower','upper']:
  if bi[k] is None:assert c['interval'][k] is None
  else:assert abs(bi[k]-c['interval'][k])<1e-12
 verdict='BAD' if ca<.95 or ca<cb-.05 else 'NA' if bi['lower'] is None else 'GOOD' if math.exp(bi['mean'])>=1.2 and bi['lower']>0 else 'BAD';assert verdict==c['verdict']
 costchecks.append(dict(track=c['track'],time=c['final_time'],control=c['control_family'],candidate_fields_passing=int(ca*64),control_fields_passing=int(cb*64),fields=64,paired_rows=len(obs),ratio=c['ratio'],ratio_ci=[math.exp(bi[k]) if bi[k] is not None else None for k in ['lower','upper']],interval_status=bi['status'],verdict=verdict))
# Exploratory subgroups; no inferential intervals, same frozen primary schedules.
pairs=[]
for a in primary:
 if a['family']!='two_basis' or a['train_count']!=64:continue
 for family in p['models']:
  if family=='two_basis':continue
  b=matched(a,family)
  assert b
  pairs.append(dict(track=a['track'],final_time=a['final_time'],control=family,grid=a['grid'],regime=a['regime'],distribution=a['distribution'],field=a['field_cluster'],seed=a['seed'],logratio=math.log(b['error_rms']/a['error_rms']),candidate_pass=eligible(a),control_pass=eligible(b)))
def pairgroups(rr,keys):
 d=defaultdict(list)
 for r in rr:d[tuple(r[k] for k in keys)].append(r)
 out=[]
 for k,v in sorted(d.items()):
  ff=defaultdict(list)
  for r in v:ff[r['field']].append(r['logratio'])
  out.append(dict(zip(keys,k),ratio=float(np.exp(np.mean([np.mean(x) for x in ff.values()]))),median_field_ratio=float(np.median([np.exp(np.mean(x)) for x in ff.values()])),fields=len(ff),rows=len(v),fields_improved=sum(np.mean(x)>0 for x in ff.values()),candidate_passes=sum(x['candidate_pass'] for x in v),control_passes=sum(x['control_pass'] for x in v)))
 return out
# n16 versus n64 comparisons use the same seed and primary schedule.
data_pairs=[]
for a in primary:
 if a['family'] not in ['two_basis','fno_scaled'] or a['train_count']!=16:continue
 b=matched(a,a['family'],64);assert b
 data_pairs.append(dict(track=a['track'],family=a['family'],final_time=a['final_time'],grid=a['grid'],regime=a['regime'],distribution=a['distribution'],field=a['field_cluster'],seed=a['seed'],logratio=math.log(b['error_rms']/a['error_rms']),candidate_pass=eligible(a),control_pass=eligible(b)))
# Frozen choices and field coverage retain unselected methods in denominator.
ld=defaultdict(list)
for r in locked:
 if r['rms_target']==target and r['max_target']==target:ld[(r['track'],r['family'],r['train_count'],r['final_time'])].append(r)
lf=[]
for key,v in sorted(ld.items()):
 f=defaultdict(list)
 for r in v:f[r['field_cluster']].append(r['status']=='ELIGIBLE')
 costs=[r['cost_seconds'] for r in v if r['status']=='ELIGIBLE'];schedules=Counter(str(r['schedule_id']) for r in v)
 lf.append(dict(zip(['track','family','train_count','final_time'],key),rows=len(v),statuses=dict(Counter(r['status'] for r in v)),all_queries_pass_fields=sum(all(x) for x in f.values()),fields=len(f),median_eligible_cost=st.median(costs) if costs else None,schedules=dict(schedules)))
out=dict(scope='Independent exact-byte reanalysis; subgroup and extra-model comparisons exploratory, not new locked claims.',denominators=dict(endpoint_rows=len(rows),primary_rows=len(primary),locked_rows=len(locked),models=len({r['model_id'] for r in rows}),families=len({r['family'] for r in rows}),independent_fields=len({r['field_cluster'] for r in rows}),numerical_failure_rows=sum(not r['finite'] for r in rows),reference_unaccepted_rows=sum(not r['reference_accepted'] for r in rows),physical_violation_rows=sum(r['physical_interval_violations']>0 for r in rows),energy_increase_rows=sum(r['energy_increase']>1e-12 for r in rows)),primary_accuracy_checks=checks,primary_cost_checks=costchecks,primary_method_summary=group_summary(primary,['track','family','train_count','final_time']),secondary_all_controls=pairgroups(pairs,['track','final_time','control']),secondary_regime=pairgroups(pairs,['track','final_time','control','regime']),secondary_distribution=pairgroups(pairs,['track','final_time','control','distribution']),secondary_grid=pairgroups(pairs,['track','final_time','control','grid']),secondary_data_efficiency=pairgroups(data_pairs,['track','family','final_time']),secondary_data_efficiency_distribution=pairgroups(data_pairs,['track','family','final_time','distribution']),locked_method_summary=lf)

# Post-hoc sensitivity to FP32-sized errors: no new inferential claim.
sensitivity=[]
for threshold in [1e-7,1e-6]:
 for tr in ['discrete','continuum']:
  for tt in [.12,.24]:
   for control in ['commutator_cubic','quad2_full','quad4_full','analytic_quad_cubic','fno_scaled']:
    rr=[]
    for a in primary:
     if a['family']!='two_basis' or a['train_count']!=64 or a['track']!=tr or a['final_time']!=tt:continue
     b=matched(a,control)
     if min(a['error_rms'],b['error_rms'])>=threshold:rr.append((a,b))
    ff=defaultdict(list)
    for a,b in rr:ff[a['field_cluster']].append(math.log(b['error_rms']/a['error_rms']))
    sensitivity.append(dict(track=tr,time=tt,control=control,minimum_both_rms=threshold,rows=len(rr),fields=len(ff),ratio=float(np.exp(np.mean([np.mean(v) for v in ff.values()]))) if ff else None))
out['secondary_roundoff_sensitivity']=sensitivity

# Sensitivity envelope using reported reference estimates, not certificates.
reference_sensitivity=[]
for c in claims['accuracy']:
 pp=[]
 for a in primary:
  if a['family']=='two_basis' and a['train_count']==64 and a['track']==c['track'] and a['final_time']==c['final_time']:
   pp.append((a,matched(a,c['control_family'])))
 obslo=[];obshi=[];potential_sign_changes=0
 for a,b in pp:
  ua=a['reference_uncertainty_rms'];ub=b['reference_uncertainty_rms']
  obslo.append((a['field_cluster'],a['seed'],math.log((b['error_rms']-ub)/(a['error_rms']+ua))))
  obshi.append((a['field_cluster'],a['seed'],math.log((b['error_rms']+ub)/(a['error_rms']-ua))))
  potential_sign_changes+=abs(b['error_rms']-a['error_rms'])<=ua+ub
 lo=boot(obslo);hi=boot(obshi)
 reference_sensitivity.append(dict(track=c['track'],time=c['final_time'],control=c['control_family'],ratio_lower_estimate=math.exp(lo['mean']),ratio_upper_estimate=math.exp(hi['mean']),bootstrap_lower_of_lower_estimate=math.exp(lo['lower']),bootstrap_upper_of_upper_estimate=math.exp(hi['upper']),sign_sensitive_rows=potential_sign_changes,rows=len(pp),scope='Conservative triangle-inequality sensitivity assuming reported reference estimates bound error; estimates are not certificates, endpoints share one target, and this is not a formal confidence interval.'))
out['secondary_reference_sensitivity']=reference_sensitivity
(BASE/'statistics-results.json').write_text(json.dumps(out,indent=2,default=lambda x:x.item()))
print(json.dumps(out['denominators']))
print('Verified all20 accuracy claims and all20 cost claims, including exact bootstrap intervals.')
print('Primary verdicts',Counter(r['verdict'] for r in checks),Counter(r['verdict'] for r in costchecks))
print('Two-basis16 versus64',json.dumps(out['secondary_data_efficiency'],default=lambda x:x.item()))

