from pathlib import Path
from collections import defaultdict, Counter
import csv,gzip,json,math,statistics,sys
import numpy as np
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE));sys.path.insert(0,str(ROOT))
from review_io import load
from tdn.analysis.portfolio.statistics import paired_cluster_interval_fast
p=load('protocol.json')['scientific_protocol']
rows=load('aggregate/endpoint_rows.json')['rows']
metrics=['error_rms','error_max','centered_rms','mean_error','spectral_low_rms','spectral_high_rms','upper_rms','upper_max','cost_seconds','cold_seconds','reference_uncertainty_rms','reference_uncertainty_max']

def quant(vals):
 a=np.asarray([x for x in vals if x is not None and math.isfinite(x)],dtype=float)
 if len(a)==0:return {'count':0}
 return dict(count=len(a),mean=float(a.mean()),median=float(np.median(a)),p90=float(np.quantile(a,.9)),p99=float(np.quantile(a,.99)),maximum=float(a.max()),minimum=float(a.min()))

def eligible(r,t=2e-5):return bool(r.get('finite',True) and r.get('reference_accepted') and r.get('upper_rms') is not None and r.get('upper_max') is not None and r['upper_rms']<=t and r['upper_max']<=t)
def summary(rs,identity):
 fields=defaultdict(list)
 for r in rs:fields[r['field_cluster']].append(eligible(r))
 d={**identity,'rows':len(rs),'independent_fields':len(fields),'seeds':sorted({r['seed'] for r in rs if r.get('seed') is not None}),'finite_rows':sum(r.get('finite',True) for r in rs),'accepted_reference_rows':sum(r.get('reference_accepted',False) for r in rs),'status_counts':dict(Counter(r.get('status','not_recorded') for r in rs)),'passes':{str(t):sum(eligible(r,t) for r in rs) for t in p['targets']},'every_query_passes_fields':sum(all(v) for v in fields.values()),'interval_violation_rows':sum(bool(r.get('physical_interval_violations')) for r in rs),'interval_violation_entries':sum(r.get('physical_interval_violations') or 0 for r in rs),'minimum_output':min((r['minimum'] for r in rs if r.get('minimum') is not None),default=None),'maximum_output':max((r['maximum'] for r in rs if r.get('maximum') is not None),default=None),'metrics':{m:quant([r.get(m) for r in rs]) for m in metrics},'mean_squared_error_fraction':quant([(r['mean_error']/r['error_rms'])**2 for r in rs if r.get('error_rms',0)>0 and r.get('mean_error') is not None]),'high_spectral_squared_error_fraction':quant([(r['spectral_high_rms']/r['error_rms'])**2 for r in rs if r.get('error_rms',0)>0 and r.get('spectral_high_rms') is not None])}
 return d

grand=summary(rows,{'scope':'all endpoint rows; not independent observations'})
output={'metadata':{'run':'fedora-portfolio-20261009T183307014599Z','source':'aggregate/endpoint_rows.json','fresh_statistical_unit':'24 independent continuous fields; 3 crossed training seeds for fitted families','review_status':'posthoc descriptive review, no fresh protocol or novel confirmatory claims','primary_target':p['primary_target'],'targets':p['targets'],'grids':p['grids'],'model_instances':len({r['model_id'] for r in rows}),'families':sorted({r['family'] for r in rows})},'grand':grand}
for key,dimension in [('family',None),('grid','grid'),('regime','regime'),('distribution','distribution'),('horizon','final_time'),('seed','seed')]:
 groups=defaultdict(list)
 for r in rows:
  fixed=(r['family'],r['track'],r['train_count'])
  if dimension:fixed+=(r[dimension],)
  groups[fixed].append(r)
 tables=[]
 for identity,rs in sorted(groups.items(),key=lambda v:repr(v[0])):
  base=dict(zip(['family','track','train_count']+([dimension] if dimension else []),identity))
  tables.append(summary(rs,{**base,'schedules':'all'}))
  if dimension not in ('seed','final_time'):
   tables.append(summary([r for r in rs if r['primary']],{**base,'schedules':'primary'}))
 output[key]=tables
 print(key,len(tables),flush=True)
# Analyze short-rollout intermediate states separately from endpoints.
inter=[]
for r in rows:
 for i,mid in enumerate(r['intermediates']):
  inter.append({**{k:r[k] for k in ('model_id','family','track','train_count','seed','parent_id','field_cluster','regime','grid','final_time','schedule_id','distribution')},**mid,'intermediate_index':i+1,'parent_endpoint_error_rms':r['error_rms']})
output['intermediate_grand']=summary(inter,{'scope':'independently referenced intermediate observations for primary schedules only'})
groups=defaultdict(list)
for r in inter:groups[(r['family'],r['track'],r['train_count'])].append(r)
output['intermediates']=[summary(rs,dict(zip(['family','track','train_count'],k))) for k,rs in sorted(groups.items())]
# Matched pair aggregate all 24fields (frozen controls repeated against paired trainedseed, never luckyseedselected).
lookup={(r['family'],r['track'],r['train_count'],r['seed'],r['parent_id'],r['grid'],r['schedule_id']):r for r in rows}
contrasts=[(f,'quad2_fixed') for f in output['metadata']['families'] if f!='quad2_fixed']+[
 ('quad2_fixed','historical_half'),('quad2_fixed','quad2_input'),('quad4_fixed','quad2_fixed'),('quad2_full','quad2_fixed'),('quad4_full','quad4_fixed'),('quad2_conditioned','quad2_amplitude'),('quad2_conditioned','quad2_nodes'),('quad2_conditioned','quad2_joint'),('quad2_conditioned','quad2_linear'),('quad2_conditioned_full','quad2_conditioned'),('quad4_conditioned','quad2_conditioned'),('residual_quad2','quad2_conditioned'),('conditioned_rich','quad2_conditioned'),('band_gain','quad2_conditioned'),('analytic_quad_cubic','quad4_full'),('etdrk4','df')]
contrasts=list(dict.fromkeys(contrasts))
output['matched_review_contrasts']=[]
for left,right in contrasts:
 for track in p['tracks']:
  left_rows=[r for r in rows if r['family']==left and r['track']==track and r['train_count'] in (0,32)]
  for scope in ('all','primary'):
   obs={m:[] for m in ('error_rms','error_max','mean_error','centered_rms','cost_seconds')};pairs=[];left_pass=right_pass=0
   for a in left_rows:
    if scope=='primary' and not a['primary']:continue
    match=(a['parent_id'],a['grid'],a['schedule_id'])
    b=lookup.get((right,track,a['train_count'],a['seed'],*match)) or lookup.get((right,track,0,None,*match))
    if b is None:continue
    pairs.append((a,b));left_pass+=eligible(a);right_pass+=eligible(b)
    for m in obs:
     if a.get(m) is not None and b.get(m) is not None and a[m]>0 and b[m]>0 and a['reference_accepted'] and b['reference_accepted']:
      obs[m].append({'field_cluster':a['field_cluster'],'seed':a['seed'] if a['seed'] is not None else b['seed'],'difference':math.log(b[m]/a[m])})
   if not pairs:continue
   result={'candidate':left,'control':right,'track':track,'schedules':scope,'data_size':32 if any(a['train_count']==32 for a,b in pairs) else 0,'paired_rows':len(pairs),'candidate_passes':left_pass,'control_passes':right_pass,'metric_orientation':'control/candidate; >1 candidate lower error or cost','scope':'posthoc wholecohort paired descriptive analysis; not new confirmatory claim','metrics':{}}
   for m,values in obs.items():
    interval=paired_cluster_interval_fast(values,repeats=1000,minimum_fields=5)
    result['metrics'][m]={'ratio':math.exp(interval['mean']) if interval.get('mean') is not None else None,'lower':math.exp(interval['lower']) if interval.get('lower') is not None else None,'upper':math.exp(interval['upper']) if interval.get('upper') is not None else None,'status':interval['status'],'independent_fields':interval['independent_fields'],'training_seeds':interval['training_seeds'],'query_rows':interval['query_rows']}
   output['matched_review_contrasts'].append(result)
print('contrasts',len(output['matched_review_contrasts']),flush=True)
# Original declaredclaims preserved verbatim for terminaldecisioncounts.
output['claims']=load('aggregate/claims.json')
# Independent endpoint norm identities should hold before aggregation.
output['metric_consistency']={'max_relative_RMS_mean_centered_square_discrepancy':max(abs(r['error_rms']**2-r['mean_error']**2-r['centered_rms']**2)/max(r['error_rms']**2,1e-30) for r in rows if r.get('error_rms')),'max_relative_RMS_spectral_square_discrepancy':max(abs(r['error_rms']**2-r['spectral_low_rms']**2-r['spectral_high_rms']**2)/max(r['error_rms']**2,1e-30) for r in rows if r.get('error_rms'))}
# Compact exactnumericprojection for followup reviews; rawsourcevalues retained, no rounding or replacement.
with gzip.open(BASE/'statistics-endpoints-compact.csv.gz','wt',newline='') as fh:
 fields=['model_id','family','track','seed','train_count','parent_id','field_cluster','regime','distribution','grid','schedule_id','final_time','primary','status','finite','reference_accepted','physical_interval_violations','minimum','maximum',*metrics]
 writer=csv.DictWriter(fh,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
# Locked/posthoc frontier coverage byfamily/track/data/horizon/target; coarsebestbutnotselectiontruth.
comparisons=load('aggregate/comparisons.json')
for source in ('locked_frontiers','posthoc_frontiers'):
 groups=defaultdict(list)
 for r in comparisons[source]:groups[(r['family'],r['track'],r['train_count'],r['final_time'],r['rms_target'])].append(r)
 tables=[]
 for key,rs in sorted(groups.items()):
  good='ELIGIBLE' if source=='locked_frontiers' else 'FEASIBLE'
  fields=defaultdict(list)
  for r in rs:fields[r['field_cluster']].append(r['status']==good)
  tables.append({**dict(zip(['family','track','train_count','final_time','target'],key)),'rows':len(rs),'statuses':dict(Counter(r['status'] for r in rs)),'passed_rows':sum(r['status']==good for r in rs),'independent_fields':len(fields),'all_seed_grid_passed_fields':sum(all(v) for v in fields.values()),'eligible_cost_seconds':quant([r['cost_seconds'] for r in rs if r['status']==good]),'selected_schedules':dict(Counter(r['schedule_id'] for r in rs))})
 output[source]=tables
output['original_subgroup_comparison_count']=len(comparisons['matched_configuration'])
# Retain summarized allpairmetrics with source3fieldCIcaveat.
with gzip.open(BASE/'statistics-all-pair-subgroups.json.gz','wt') as fh:json.dump(comparisons['matched_configuration'],fh,separators=(',',':'),allow_nan=False)
(BASE/'statistics-review.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
print('Wrote',BASE/'statistics-review.json',flush=True)
