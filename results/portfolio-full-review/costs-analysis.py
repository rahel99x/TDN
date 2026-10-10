"""Read-only accounting and scaling review of exact native portfolio artifacts."""
from collections import Counter, defaultdict
from pathlib import Path
import json, math, re, statistics
import review_io as evidence

BASE = Path(__file__).resolve().parent
protocol = evidence.load('scaling/protocol.json')
accounting = evidence.load('state/scheduler-accounting.json')
allocations = [r for r in accounting['records'] if r['record_kind'] == 'allocation']
steps = [r for r in accounting['records'] if r['record_kind'] == 'step']

def cpu_seconds(value):
    days, clock = (value.split('-', 1) if '-' in value else ('0', value))
    out = 0.
    for x in clock.split(':'): out = out * 60 + float(x)
    return out + int(days) * 86400

def rss_bytes(value):
    if not value: return None
    match = re.fullmatch(r'([0-9.]+)([KMGT]?)', value)
    return float(match[1]) * 1024 ** ('KMGT'.index(match[2])+1 if match[2] else 0)

def summarize(rows):
    elapsed = sum(r['elapsed_seconds'] for r in rows)
    cpu = sum(cpu_seconds(r['TotalCPU']) for r in rows)
    allocated = sum(r['allocated_cpu_seconds'] for r in rows)
    return dict(jobs=len(rows), elapsed_allocation_seconds=elapsed,
                allocated_cpu_seconds=allocated, total_cpu_seconds=cpu,
                cpu_allocation_efficiency=cpu/allocated if allocated else None,
                gpu_allocation_seconds=sum(r['elapsed_seconds'] for r in rows if 'gres/gpu=1' in r['AllocTRES']))

bykind = defaultdict(list)
for r in allocations: bykind[r['logical_stage']].append(r)
stage_rows = []
for r in allocations:
    summary = evidence.load(r['workflow_stage'] + '/summary.json')
    matching_steps = [s for s in steps if s['allocation_job_id'] == r['JobIDRaw']]
    peak = max((rss_bytes(s['MaxRSS']) or 0 for s in matching_steps), default=None)
    stage_rows.append(dict(stage=r['workflow_stage'], kind=r['logical_stage'],
        job_id=r['JobIDRaw'], allocated_seconds=r['elapsed_seconds'],
        cpu_seconds=cpu_seconds(r['TotalCPU']), allocated_cpu_seconds=r['allocated_cpu_seconds'],
        cpu_efficiency=cpu_seconds(r['TotalCPU'])/r['allocated_cpu_seconds'],
        peak_step_rss_bytes=peak, numerical_seconds=summary['elapsed_seconds'],
        nonnumerical_seconds=r['elapsed_seconds']-summary['elapsed_seconds'],
        memory=summary.get('memory'), state=r['State'], recorded_step_count=len(matching_steps)))

scaling = evidence.load('scaling/scaling_rows.json')['rows']
measured = [r for r in scaling if 'model_id' in r]
grouped = defaultdict(list)
for r in measured: grouped[(r['grid'],r['track'],r['batch_size'])].append(r)
comparisons=[]
for (grid,track,batch), rows in grouped.items():
    models={r['family']:r for r in rows}; primary=models['quad2_conditioned']
    def brief(r):
        return dict(family=r['family'], latency_ms=r['cost_seconds']*1000,
            throughput_fields_per_second=r['samples_per_second'],
            first_invocation_ms=r['cold_seconds']*1000,
            worst_rms=max(e['error_rms'] for e in r['errors']),
            worst_max=max(e['error_max'] for e in r['errors']),
            rms_median=statistics.median(e['error_rms'] for e in r['errors']),
            status=r['status'], repeats=r['raw_timing']['repeats'],
            spread_max_over_min=max(r['raw_timing']['methods'][r['model_id']]['samples_seconds'])/min(r['raw_timing']['methods'][r['model_id']]['samples_seconds']))
    comparisons.append(dict(grid=grid,track=track,batch=batch,final_time=primary['final_time'],
        steps=primary['step_count'], parent_ids=primary['parent_ids'],
        methods=[brief(r) for r in rows],
        conditioned_over_fixed_latency=primary['cost_seconds']/models['quad2_fixed']['cost_seconds'],
        conditioned_over_etdrk4_latency=primary['cost_seconds']/models['etdrk4']['cost_seconds'],
        conditioned_over_df_latency=primary['cost_seconds']/models['df']['cost_seconds'],
        paired_group_peak_allocated_bytes=primary['raw_timing']['peak_allocated_bytes'],
        paired_group_peak_reserved_bytes=primary['raw_timing']['peak_reserved_bytes'],
        teacher_seconds_once=primary['teacher_seconds']))

profiles=evidence.load('diagnose/profile_rows.json'); profile_summaries=[]
for regime in protocol['diagnostics']['regimes']:
    for family in protocol['diagnostics']['profile_families']:
        rows=[r for r in profiles if r['regime']==regime and r['family']==family]
        wall=next(r['seconds']for r in rows if r['component']=='warm_end_to_end')
        components={r['component']:r['seconds'] for r in rows if r['measurement_scope'].startswith('instrumented')}
        total=sum(components.values())
        profile_summaries.append(dict(regime=regime,family=family,warm_ms=wall*1000,
            instrumented_self_cpu_seconds=total,component_fraction_of_instrumented={k:v/total for k,v in components.items()},
            feature_wall_ms=next((r['seconds']*1000 for r in rows if r['component']=='standalone_feature_extraction'),None)))

catalog=evidence.load('freeze/catalog.json')['records']
training=[]
for family in sorted({r['family']for r in catalog}):
    records=[r for r in catalog if r['family']==family]
    trials=[t for r in records for t in r.get('trials',[])]
    training.append(dict(family=family,models=len(records),trials=len(trials),
        trial_wall_seconds=sum(t['training_seconds']for t in trials),
        optimizer_seconds=sum(t.get('optimizer_seconds',0)for t in trials),
        completed_updates=sum(t['updates_completed']for t in trials),
        failed_trials=sum(bool(t.get('failure'))for t in trials),
        initialization_selections=sum(r['selection_status']=='SELECTED_INITIALIZATION'for r in records)))

references=[]
for name in evidence.names():
    if re.match(r'(prepare|confirm-prepare-[0-9]+)/parent-[0-9]+/attempt-[0-9]+/parent-[0-9]+\.json$', name):
        references.append(evidence.load(name))

output=dict(scope='Fresh read-only audit of native c5c9a85 portfolio; no new inference or test-truth selection',
    total=summarize(allocations), by_stage_kind={k:summarize(v)for k,v in bykind.items()},
    all_allocations_terminal=accounting['all_allocations_terminal'],
    allocation_states=dict(Counter(r['State']for r in allocations)),
    accounting_records=len(accounting['records']),step_rows=len(steps),
    step_states=dict(Counter(r['State']for r in steps)),stage_rows=stage_rows,
    scientific_seconds=sum(r['numerical_seconds']for r in stage_rows),
    max_step_rss_bytes=max(r['peak_step_rss_bytes']for r in stage_rows),
    gpu_utilization=None,energy_joules=None,monetary_cost=None,
    scaling=dict(measured_method_cells=len(measured),status_counts=dict(Counter(r['status']for r in scaling)),
        qualified_field_method_cells=sum(len(r['errors'])for r in measured if r['status']=='ACCURACY_QUALIFIED'),
        unique_fields=len({p for r in measured for p in r['parent_ids']}),
        omitted_cases=[r for r in scaling if 'model_id'not in r],comparisons=comparisons,
        unique_case_teacher_seconds=sum(r['teacher_seconds_once']for r in comparisons)),
    training_cost=training,profiles=profile_summaries,
    caveats=['Allocation elapsed summed across jobs, not calendar makespan or money.',
        'Actual TotalCPU / reserved CPU-seconds measures CPU reservation use, not GPU utilization.',
        'All allocation MaxRSS fields blank; reported host maximum is maximum observed task-step RSS, not summed process RSS.',
        'Paired-group GPU memory peaks are not per-model memory attribution.',
        'Scaling includes largest declared batch=4 only; batch=1 cells were capped by protocol.',
        'Three warm paired rounds do not establish p95 or p99 tails; p95 is observed maximum here.',
        'First invocation is new timing group within a warm process, not end-to-end service cold start.',
        'CPU component profiles use small FP64 initialized models, not trained GPU deployment.',
        'Amortization reporter expects candidate_seconds/comparator_seconds absent in locked-frontier rows; NA is not itself evidence of no margin.',
        'Offline costs, failed attempts, reference generation, preprocessing and deployment verification are distinct; all included jobs completed, separate accidental rerun not in this accounting.'])
(BASE/'costs-summary.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps({k:output[k] for k in ('total','by_stage_kind','scientific_seconds','max_step_rss_bytes')},indent=2))
print('Wrote',BASE/'costs-summary.json')
