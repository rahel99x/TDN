"""Read-only operational analysis of exact canonical native Advance evidence."""
import collections, datetime, json, math, statistics, xml.etree.ElementTree as ET
from review_io import BASE, load, names, raw
D=collections.defaultdict

def hhmm(s):
 return f'{int(s)//3600}:{int(s)%3600//60:02d}:{int(s)%60:02d}'
def duration(s):
 chunks=s.split(':'); return sum(float(c)*60**i for i,c in enumerate(chunks[::-1]))
def group_summary(rows,keys,stats):
 groups=D(list)
 for r in rows: groups[tuple(r.get(k) for k in keys)].append(r)
 return [dict(zip(keys,k),**stats(v)) for k,v in sorted(groups.items(),key=lambda x:str(x[0]))]
def desc(values):
 values=[v for v in values if v is not None]
 return dict(n=len(values),median=statistics.median(values) if values else None,min=min(values) if values else None,max=max(values) if values else None)

workflow=load('advance-workflow.json'); jobs=load('jobs.json'); accounting=load('state/scheduler-accounting.json')
alloc=[r for r in accounting['records'] if r['record_kind']=='allocation']
steps=[r for r in accounting['records'] if r['record_kind']=='step']
assert len(alloc)==len(jobs)==49
assert {r['JobIDRaw'] for r in alloc}=={r['job_id'] for r in jobs}
assert all(r['State']=='COMPLETED' and r['ExitCode']=='0:0' for r in alloc)
jobmap={r['stage']:r for r in jobs}
gpu=[r for r in alloc if 'gres/gpu=1' in r['AllocTRES']]
assert len(gpu)==23
summaries={r['stage']:load(r['stage']+'/summary.json') for r in jobs}
for r in alloc:
 r['numerical_seconds']=summaries[r['workflow_stage']]['elapsed_seconds']
 r['cpu_seconds']=duration(r['TotalCPU'])
 r['requested_wall_seconds']=duration(workflow['resources'][r['workflow_stage']]['walltime'])
 assert r['elapsed_seconds']<=r['requested_wall_seconds']<=2700
category=group_summary(alloc,['logical_stage'],lambda rs:dict(jobs=len(rs),allocation_seconds=sum(r['elapsed_seconds'] for r in rs),gpu_seconds=sum(r['elapsed_seconds'] for r in rs if 'gres/gpu=1' in r['AllocTRES']),allocated_cpu_seconds=sum(r['allocated_cpu_seconds'] for r in rs),actual_cpu_seconds=sum(r['cpu_seconds'] for r in rs),numerical_seconds=sum(r['numerical_seconds'] for r in rs),max_allocation_seconds=max(r['elapsed_seconds'] for r in rs)))
checks=[]
for job in jobs:
 n='reporter-tests/'+job['stage']+'/tests.xml'
 if n not in names():continue
 xml=ET.fromstring(raw(n)); cases=list(xml.iter('testcase'))
 checks.append(dict(stage=job['stage'],cases=len(cases),names=[c.get('name') for c in cases],failures=sum(c.find('failure') is not None for c in cases),errors=sum(c.find('error') is not None for c in cases),skips=sum(c.find('skipped') is not None for c in cases)))
gpuchecks=[c for c in checks if c['stage']!='audit']
assert len(gpuchecks)==23 and all(c['cases']==40 and c['failures']==c['errors']==c['skips']==0 for c in gpuchecks)
assert all(set(c['names'])==set(gpuchecks[0]['names']) for c in gpuchecks)

memory=[dict(stage=k,**s['memory']) for k,s in summaries.items() if s['memory']['enabled']]
rss=[]
for r in steps:
 value=r['MaxRSS']
 if value and value[-1] in 'KMG':rss.append((float(value[:-1])*{'K':1024,'M':1024**2,'G':1024**3}[value[-1]],r['workflow_stage'],r['JobIDRaw']))
report_done=load('state/report.json')['updated_at']
start=datetime.datetime.fromisoformat(workflow['created_at']); end=datetime.datetime.fromisoformat(report_done)
operation=dict(accounting_status=accounting['status'],all_allocations_terminal=accounting['all_allocations_terminal'],allocation_count=len(alloc),gpu_allocation_count=len(gpu),allocation_seconds=sum(r['elapsed_seconds'] for r in alloc),gpu_allocation_seconds=sum(r['elapsed_seconds'] for r in gpu),actual_cpu_seconds=sum(r['cpu_seconds'] for r in alloc),allocated_cpu_seconds=sum(r['allocated_cpu_seconds'] for r in alloc),science_stage_seconds=sum(r['numerical_seconds'] for r in alloc),workflow_elapsed_seconds=(end-start).total_seconds(),max_job=max(alloc,key=lambda r:r['elapsed_seconds']),categories=category,tests=[{k:v for k,v in c.items() if k!='names'} for c in checks],max_peak_allocated_bytes=max(m['peak_allocated_bytes'] for m in memory),max_peak_reserved_bytes=max(m['peak_reserved_bytes'] for m in memory),max_device_used_bytes=max(m['peak_sampled_device_used_bytes'] for m in memory),minimum_soft_budget_bytes=min(m['soft_budget_bytes'] for m in memory),max_step_rss=sorted(rss,reverse=True)[:5],failed_allocations=0,failures_interruption_cost_seconds=0,monetary_cost=None,electricity_joules=None)

references=[]
for n in names():
 if '/attempt-' not in n or not n.endswith('.json') or '/parent-' not in n:continue
 p=load(n)
 if 'references' not in p:continue
 for key,r in p['references'].items():
  references.append(dict(stage=n.split('/')[0],split=p['split'],parent_id=p['parent_id'],track=r['track'],grid=r['grid'],horizon=r['horizon'],teacher_seconds=r['teacher_seconds'],accepted=r['accepted']))
refsummary=group_summary(references,['split','track'],lambda rs:dict(references=len(rs),accepted=sum(r['accepted'] for r in rs),summed_teacher_task_seconds=sum(r['teacher_seconds'] for r in rs)))

catalog=load('freeze/catalog.json')['records']; trials=[]
for r in catalog:
 for t in r['trials']:trials.append(t)
 audit=r.get('initialization_audit')
 if audit: print('AUDIT_KEY')
train_costs=group_summary(catalog,['track','family','train_count'],lambda rs:dict(models=len(rs),total_training_and_tuning_seconds=sum(r.get('total_training_seconds',r['training_seconds']) for r in rs),selected_final_trial_seconds=sum(r['training_seconds'] for r in rs),selected_final_optimizer_seconds=sum(r['optimizer_seconds'] for r in rs)))
trialsummary=group_summary(trials,['track','phase'],lambda rs:dict(trials=len(rs),seconds=sum(r['training_seconds'] for r in rs),optimizer_seconds=sum(r['optimizer_seconds'] for r in rs),failures=sum(bool(r['failure']) for r in rs),stopping=dict(collections.Counter(r['stopping_reason'] for r in rs))))

scale=load('scaling/scaling_rows.json')['rows']; measured=[r for r in scale if r['status']=='MEASURED']; missing=[r for r in scale if r['status']!='MEASURED']
scaletable=group_summary(measured,['track','grid','batch_size','family'],lambda rs:dict(measurement_rows=len(rs),parent_fields=len({p for r in rs for p in r['parent_ids']}),accuracy_qualified_rows=sum(r['accuracy_qualified'] for r in rs),latency_ms=desc([1000*r['cost_seconds'] for r in rs]),per_field_ms=desc([1000*r['amortized_seconds_per_field'] for r in rs]),fields_per_second=desc([r['fields_per_second'] for r in rs]),max_upper_rms=max(m['upper_rms'] for r in rs for m in r['per_field']),max_upper_max=max(m['upper_max'] for r in rs for m in r['per_field'])))
# Compare batching only on its four reused low/nonstiff fields; not against a different field mix.
lookup={(r['track'],r['grid'],r['family'],r['parent_ids'][0]):r for r in measured if r['batch_size']==1}
batchgain=[]
for r in measured:
 if r['batch_size']!=4:continue
 single=[lookup[(r['track'],r['grid'],r['family'],p)] for p in r['parent_ids']]
 serial=sum(s['cost_seconds'] for s in single)
 batchgain.append(dict(track=r['track'],grid=r['grid'],family=r['family'],parent_ids=r['parent_ids'],serial_single_field_seconds=serial,batch_seconds=r['cost_seconds'],batch_throughput_gain=serial/r['cost_seconds'],batch_accuracy_qualified=r['accuracy_qualified'],single_accuracy_qualified=all(s['accuracy_qualified'] for s in single)))

# Exact eligible frozen-schedule joins; no best seed, oracle schedule, or rank pooling.
locked=load('aggregate/locked_frontiers.json')['rows']
keys=('parent_id','grid','track','final_time','rms_target','max_target')
def key(r):return tuple(r.get(k) for k in keys)
lookup={(*key(r),r['family'],r.get('seed'),r.get('train_count')):r for r in locked}
assert len(lookup)==len(locked)
byid={r['model_id']:r for r in catalog}
comparators=['df','etdrk4','quad2_full','quad4_full','analytic_quad_cubic','commutator_cubic','quad2_conditioned','fno_scaled']
parent_metadata={p['parent_id']:p for p in load('aggregate/protocol.json')['parents']}
econ=[]
for r in locked:
 if r['family']!='two_basis' or r['train_count']!=64:continue
 for family in comparators:
  control=lookup.get((*key(r),family,r.get('seed'),r.get('train_count')),lookup.get((*key(r),family,None,0)))
  eligible=r['status']=='ELIGIBLE' and control is not None and control['status']=='ELIGIBLE'
  saving=control['cost_seconds']-r['cost_seconds'] if eligible else None
  training=byid[r['model_id']]['total_training_seconds']
  econ.append(dict(track=r['track'],grid=r['grid'],regime=parent_metadata[r['parent_id']]['regime'],distribution=parent_metadata[r['parent_id']]['distribution'],final_time=r['final_time'],target=r['rms_target'],family=family,seed=r['seed'],parent_id=r['parent_id'],candidate_eligible=r['status']=='ELIGIBLE',control_eligible=control is not None and control['status']=='ELIGIBLE',paired_eligible=eligible,saving=saving,speed_ratio=control['cost_seconds']/r['cost_seconds'] if eligible else None,training_only_break_even=math.ceil(training/saving) if saving is not None and saving>0 else None,whole_campaign_allocation_scenario=math.ceil(operation['allocation_seconds']/saving) if saving is not None and saving>0 else None))
def esummary(rs):
 eligible=[r for r in rs if r['paired_eligible']]
 return dict(rows=len(rs),eligible_pairs=len(eligible),positive_savings=sum(r['saving']>0 for r in eligible),candidate_pass_rows=sum(r['candidate_eligible'] for r in rs),control_pass_rows=sum(r['control_eligible'] for r in rs),descriptive_paired_speed_ratio=desc([r['speed_ratio'] for r in eligible]),training_only_break_even=desc([r['training_only_break_even'] for r in eligible]),whole_campaign_allocation_scenario=desc([r['whole_campaign_allocation_scenario'] for r in eligible]))
econtable=group_summary(econ,['track','final_time','target','family'],esummary)
econregime=group_summary(econ,['track','grid','regime','distribution','final_time','target','family'],esummary)

timingstats=[]
for path in ['aggregate/timing_rounds.json','scaling/timing_rounds.json']:
 timing=load(path)['rows']; methods=[v for r in timing for v in r['methods'].values()]
 rounds=[round for r in timing for round in r['rounds']]
 assert all(set(round['order'])==set(round['samples_seconds']) and len(round['order'])==len(set(round['order'])) for round in rounds)
 timingstats.append(dict(source=path,groups=len(timing),methods_calls=len(methods),randomized_rounds=len(rounds),warm_measured_calls=sum(len(m['samples_seconds']) for m in methods),first_calls=len(methods),warmup_calls=sum(m['warmup'] for m in methods),memory_probes=len(methods),warm_measured_seconds=sum(m['total_seconds'] for m in methods),first_call_seconds=sum(m['cold_seconds'] for m in methods),warmup_seconds=sum(m['warmup_seconds'] for m in methods),memory_probe_seconds=sum(m['memory']['probe_seconds'] for m in methods),all_tail_p95_null=all(m['tail_statistics']['p95_seconds'] is None for m in methods),all_tail_p99_null=all(m['tail_statistics']['p99_seconds'] is None for m in methods),repeats=dict(collections.Counter(m['repeats'] for m in methods)),max_absolute_peak_allocated_bytes=max(m['memory']['absolute_peak_allocated_bytes'] for m in methods),max_incremental_peak_allocated_bytes=max(m['memory']['incremental_peak_allocated_bytes'] for m in methods)))

result=dict(operations=operation,references=refsummary,training=train_costs,trials=trialsummary,scaling_manifest=load('scaling/scaling_manifest.json'),scaling_missing=missing,scaling=scaletable,batch_gain=batchgain,economics=econtable,economics_regime=econregime,timing=timingstats,primary_cost_claims=load('aggregate/claims.json')['cost'],scope='Read-only exact native artifacts, a764670; economic scenarios are conditional per workload, not a new locked claim; full per-method offline attribution remains unavailable.')
(BASE/'costs-review.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['operations','references','trials','timing']},indent=2))
