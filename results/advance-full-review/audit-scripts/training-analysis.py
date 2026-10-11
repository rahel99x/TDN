"""Descriptive training audit of verified native bytes; does not train or retune."""
from collections import Counter,defaultdict
import json,statistics,math
from pathlib import Path
from review_io import load,lines,BASE
C=load('freeze/catalog.json')['records']; P=load('audit/protocol.json'); U=sorted({r['training_unit'] for r in C})
def extent(v):
 v=[x for x in v if x is not None and math.isfinite(x)]
 return dict(n=len(v),minimum=min(v),median=statistics.median(v),maximum=max(v)) if v else dict(n=0)
def slim(q):
 keys=['model_id','family','track','seed','train_count','parameters','trainable_parameters','phase','trial_id','selection_status','initial_validation_objective','validation_objective','updates_requested','updates_completed','updates_selected','examples_seen','training_seconds','total_training_seconds','optimizer_seconds','time_cap_seconds','stopping_reason','equal_time_status','time_cap_overshoot_seconds','learning_rate','optimizer_control','failure','training_samples','independent_training_fields','initial_df_max_difference']
 out={k:q[k] for k in keys if k in q};out['effective_gains']=q.get('parameter_report',{}).get('effective_gains')
 out['validation_reduction_fraction']=1-q['validation_objective']/q['initial_validation_objective'] if q.get('initial_validation_objective') else None
 return out
A=[q for u in U for q in load(u+'/fairness_diagnostics.json')['rows']];T=[q for r in C for q in r['trials']]
res=dict(protocol_training=P['training'],counts=dict(instances=len(C),trainable=sum(r['trainable_parameters']>0 for r in C),selection=dict(Counter(r['selection_status'] for r in C)),trial_phases=dict(Counter(r['phase'] for r in T)),tiny_diagnostics=len(A)),instances=[slim(q) for q in C],tiny=[slim(q) for q in A],trials=[slim(q) for q in T],grouped=[])
for track in P['tracks']:
 for fam in P['models']:
  for n in sorted({q['train_count'] for q in C if q['track']==track and q['family']==fam}):
   c=[q for q in C if q['track']==track and q['family']==fam and q['train_count']==n]
   res['grouped'].append(dict(track=track,family=fam,train_count=n,instances=len(c),selection=dict(Counter(q['selection_status'] for q in c)),selected_loss=extent([q['validation_objective'] for q in c]),reduction_fraction=extent([1-q['validation_objective']/q['initial_validation_objective'] for q in c]),total_seconds=extent([q['total_training_seconds'] for q in c]),parameters=sorted({q['trainable_parameters'] for q in c}),quadratic_gain=extent([q.get('parameter_report',{}).get('effective_gains',[None,None])[0] for q in c]),cubic_gain=extent([q.get('parameter_report',{}).get('effective_gains',[None,None])[1] for q in c])))
res['phase_totals']={p:dict(trials=len(t),updates=sum(x['updates_completed'] for x in t),elapsed_seconds=sum(x['training_seconds'] for x in t),optimizer_seconds=sum(x['optimizer_seconds'] for x in t),stopping_reasons=dict(Counter(x['stopping_reason'] for x in t)),seconds=extent([x['training_seconds'] for x in t]),updates_range=extent([x['updates_completed'] for x in t]),failures=sum(x['failure'] is not None for x in t)) for p in ['frozen','tuning','final','equal_time','tiny_overfit'] if (t:=[x for x in T+A if x['phase']==p])}
res['curves']=[]; buckets=defaultdict(list)
for u in U:
 for r in lines(u+'/learning_curves.jsonl'):
  if r['gradient_norm'] is not None:buckets[(r['track'],r['family'],r['train_count'],r['phase'],r['optimizer_control']['id'])].append(r)
for (track,fam,n,phase,control),c in sorted(buckets.items()):
 g=[q['gradient_norm'] for q in c];clip=c[0]['optimizer_control']['clip_grad_norm'];v=[q for q in c if q['validation_loss'] is not None]
 res['curves'].append(dict(track=track,family=fam,train_count=n,phase=phase,optimizer_control=control,rows=len(c),gradient=extent(g),clip_fraction=sum(x>clip for x in g)/len(g) if clip else None,zero_gradient_fraction=sum(x==0 for x in g)/len(g),train_loss=extent([x['train_loss'] for x in c]),last10_train_loss=extent([x['train_loss'] for x in c if x['update']>790]),validation_loss=extent([x['validation_loss'] for x in v])))
res['source_files']=['freeze/catalog.json','audit/protocol.json']+[u+'/'+f for u in U for f in ['fairness_diagnostics.json','learning_curves.jsonl','catalog.json']]
(BASE/'training-analysis.json').write_text(json.dumps(res,indent=2)+'\n')
print(json.dumps({'counts':res['counts'],'phase_totals':res['phase_totals'],'training':P['training']},indent=2))
for q in res['grouped']:
 if q['train_count']:
  print(q['track'],q['family'],q['train_count'],q['selection'],'val',q['selected_loss']['median'],'reduction',q['reduction_fraction']['median'],'gains',q['quadratic_gain'],q['cubic_gain'])
