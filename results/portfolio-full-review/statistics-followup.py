from pathlib import Path
from collections import defaultdict,Counter
import sys,json,math,statistics
BASE=Path(__file__).resolve().parent;sys.path[:0]=[str(BASE),str(BASE.parents[1])]
from review_io import load
from tdn.analysis.portfolio.statistics import paired_cluster_interval_fast
rows=load('aggregate/endpoint_rows.json')['rows'];p=load('protocol.json')['scientific_protocol']
lookup={(r['family'],r['track'],r['train_count'],r['seed'],r['parent_id'],r['grid'],r['schedule_id']):r for r in rows}
def eligible(r,t=2e-5):return r['reference_accepted'] and r['finite'] and r['upper_rms']<=t and r['upper_max']<=t
result={'data_efficiency':[],'violations':[],'worst_cases':[],'locked_positive_savings':[]}
for fam in sorted({r['family'] for r in rows if r['train_count']==32}):
 for track in p['tracks']:
  for scope in ('all','primary'):
   left=[r for r in rows if r['family']==fam and r['track']==track and r['train_count']==32 and (scope=='all' or r['primary'])]
   obs=[];passes8=passes32=0;pairs=[]
   for a in left:
    b=lookup[(fam,track,8,a['seed'],a['parent_id'],a['grid'],a['schedule_id'])]
    pairs.append((a,b));passes32+=eligible(a);passes8+=eligible(b)
    obs.append({'field_cluster':a['field_cluster'],'seed':a['seed'],'difference':math.log(b['error_rms']/a['error_rms'])})
   ci=paired_cluster_interval_fast(obs,repeats=1000,minimum_fields=5)
   result['data_efficiency'].append({'family':fam,'track':track,'schedules':scope,'ratio_rms_8_over_32':math.exp(ci['mean']),'lower':math.exp(ci['lower']),'upper':math.exp(ci['upper']),'passes8':passes8,'passes32':passes32,'denominator':len(pairs),'independent_fields':ci['independent_fields'],'training_seeds':ci['training_seeds']})
for (fam,track,n),rs in __import__('itertools').groupby(sorted(rows,key=lambda r:(r['family'],r['track'],r['train_count'])),key=lambda r:(r['family'],r['track'],r['train_count'])):
 rs=list(rs);bad=[r for r in rs if r['physical_interval_violations']];inter=[m for r in rs for m in r['intermediates'] if m['physical_interval_violations']]
 result['violations'].append({'family':fam,'track':track,'train_count':n,'endpoint_rows':len(rs),'violation_endpoint_rows':len(bad),'violation_field_clusters':len({r['field_cluster'] for r in bad}),'violation_regimes':dict(Counter(r['regime'] for r in bad)),'maximum':max(r['maximum'] for r in rs),'minimum':min(r['minimum'] for r in rs),'intermediate_violation_rows':len(inter)})
 if n in (0,32):
  for metric in ('error_rms','error_max','mean_error'):
   r=max(rs,key=lambda r:r[metric]);result['worst_cases'].append({k:r[k] for k in ('family','track','train_count','model_id','parent_id','regime','distribution','grid','final_time','schedule_id','schedule','error_rms','error_max','mean_error','spectral_high_rms','maximum')}|{'metric_selected':metric,'scope':'posthoc worst observed endpoint; no universal worst-case claim'})
comparisons=load('aggregate/comparisons.json');front= [r for r in comparisons['locked_frontiers'] if r['rms_target']==2e-5 and r['train_count']==32]
lookup={(r['family'],r['track'],r['seed'],r['parent_id'],r['grid'],r['final_time']):r for r in front}
for track in p['tracks']:
 for t in (.12,.24):
  for competitor in ('fno_small','fno_standard'):
   for seed in [*p['seeds'],None]:
    pairs=[]
    for a in front:
     if a['family']!='quad2_conditioned' or a['track']!=track or a['final_time']!=t or (seed is not None and a['seed']!=seed):continue
     b=lookup[(competitor,track,a['seed'],a['parent_id'],a['grid'],t)]
     if a['status']=='ELIGIBLE' and b['status']=='ELIGIBLE':pairs.append((a,b))
    if pairs:
     result['locked_positive_savings'].append({'candidate':'quad2_conditioned','control':competitor,'track':track,'final_time':t,'seed':seed,'pairs':len(pairs),'fields':len({a['field_cluster'] for a,b in pairs}),'mean_candidate_seconds':statistics.mean(a['cost_seconds'] for a,b in pairs),'mean_control_seconds':statistics.mean(b['cost_seconds'] for a,b in pairs),'mean_saving_seconds':statistics.mean(b['cost_seconds']-a['cost_seconds'] for a,b in pairs),'positive_saving_rows':sum(b['cost_seconds']>a['cost_seconds'] for a,b in pairs),'caveat':'Conditional on both frozen schedules meeting both tolerances, no fallback/estimator; descriptive, not universal economics'})
(BASE/'statistics-followup.json').write_text(json.dumps(result,indent=2)+'\n')
print('DATAEFF')
for r in result['data_efficiency']:
 if r['schedules']=='all':print(r)
print('VIOLATIONS')
for r in result['violations']:
 if r['violation_endpoint_rows'] or r['intermediate_violation_rows']:print(r)
