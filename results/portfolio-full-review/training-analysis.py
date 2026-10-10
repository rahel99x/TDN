"""Post-hoc audit of exact native portfolio training evidence; source untouched."""
import collections
import json
import math
import statistics as st
from pathlib import Path
from review_io import load, lines, names

BASE=Path(__file__).resolve().parent
def stat(values):
    values=[v for v in values if v is not None and math.isfinite(v)]
    return dict(n=len(values), min=min(values), median=st.median(values), max=max(values), mean=st.mean(values)) if values else None
def tally(values): return dict(collections.Counter(values))
def geo(values): return math.exp(st.mean(math.log(v) for v in values if v>0))

catalog=load('freeze/catalog.json')['records']
trials=[t for row in catalog for t in row['trials']]
curves=[row for path in names() if path.startswith('train-') and path.endswith('/learning_curves.jsonl') for row in lines(path)]
by_trial=collections.defaultdict(list)
for row in curves: by_trial[row['trial_id']].append(row)
def summary(rows):
    selected_curve=[]
    for row in rows:
        found=[c for c in by_trial[row['trial_id']] if c['update']==row['updates_selected'] and c.get('validation_loss') is not None]
        selected_curve.extend(found)
    return dict(instances=len(rows), selection=tally(r['selection_status'] for r in rows),
        parameters=sorted(set(r['parameters'] for r in rows)),trainable_parameters=sorted(set(r['trainable_parameters'] for r in rows)),
        fitted_coefficients=sorted(set(r['fitted_coefficients'] for r in rows)),
        validation_objective=stat([r['validation_objective'] for r in rows]),
        validation_rms=stat([r['validation_rms'] for r in selected_curve]),
        validation_max=stat([r['validation_max'] for r in selected_curve]),
        improvement_factor_over_initial=stat([r['initial_validation_objective']/r['validation_objective'] for r in rows if r['validation_objective']]),
        selected_updates=stat([r['updates_selected'] for r in rows]),
        selected_phase=tally(r['phase'] for r in rows),
        training_seconds_total=sum(r['total_training_seconds'] for r in rows),
        selected_trial_seconds=stat([r['training_seconds'] for r in rows]),
        selected_optimizer_control=tally(r['optimizer_control']['id'] for r in rows),
        selected_learning_rate=tally(r['learning_rate'] for r in rows),
        failures=tally(str(r['failure']) for r in rows),
        convergence=tally(r['fit_convergence_status'] for r in rows))

result=dict(instances=len(catalog),trials=len(trials),curve_rows=len(curves),
    selection=tally(r['selection_status'] for r in catalog),
    training_seconds=sum(r['total_training_seconds'] for r in catalog),
    optimizer_seconds=sum(r['optimizer_seconds'] for r in trials),
    trial_failures=tally(str(t['failure']) for t in trials),
    truncated_gradient_trials=[{k:r[k] for k in ('model_id','trial_id','phase','updates_requested','updates_completed','training_seconds','failure')} for r in trials if r['fit']=='gradient' and r['updates_completed']<r['updates_requested']],
    by_family={f:summary([r for r in catalog if r['family']==f]) for f in sorted({r['family'] for r in catalog})},
    by_track_family_size={'/'.join(map(str,(t,f,n))):summary([r for r in catalog if (r['track'],r['family'],r['train_count'])==(t,f,n)]) for t,f,n in sorted({(r['track'],r['family'],r['train_count']) for r in catalog})},
    data_efficiency=[], learned_parameters=[], optimization=[])
for family in sorted({r['family'] for r in catalog if r['train_count']}):
    for track in ('discrete','continuum'):
        low={r['seed']:r for r in catalog if (r['track'],r['family'],r['train_count'])==(track,family,8)}
        high={r['seed']:r for r in catalog if (r['track'],r['family'],r['train_count'])==(track,family,32)}
        ratios=[low[s]['validation_objective']/high[s]['validation_objective'] for s in low]
        result['data_efficiency'].append(dict(family=family,track=track,metric='validation loss n8/n32; >1 favors more data', ratios=ratios, geometric_ratio=geo(ratios)))
for row in catalog:
    report=row['parameter_report']
    if report.get('nodes'):
        probes=[]
        for probe in row['response_probes']:
            response=probe.get('response') or {}
            probes.append({k:probe[k] for k in ('split','parent_id','horizon')}|{k:response.get(k) for k in ('features','gain','effective_weights')})
        result['learned_parameters'].append({k:row[k] for k in ('model_id','family','track','train_count','seed','selection_status')}|{k:report.get(k) for k in ('nodes','global_amplitude','normalized_weights')}|dict(probes=probes, fit_diagnostics=row.get('fit_diagnostics')))
for family in sorted({r['family'] for r in catalog}):
    family_curves=[c for c in curves if c['family']==family and c.get('gradient_norm') is not None]
    clipped=[c for c in family_curves if c['optimizer_control'].get('clip_grad_norm') is not None]
    final_trials=[t for t in trials if t['family']==family and t['phase']=='final']
    result['optimization'].append(dict(family=family,updates=len(family_curves),gradient_norm=stat([c['gradient_norm'] for c in family_curves]),
        clip_fraction=sum(c['gradient_norm']>c['optimizer_control']['clip_grad_norm'] for c in clipped)/len(clipped) if clipped else None,
        clipped_updates=len(clipped),final_trials=len(final_trials),final_training_seconds=stat([r['training_seconds'] for r in final_trials]),
        final_validation_gain=stat([r['initial_validation_objective']/r['validation_objective'] for r in final_trials if r['validation_objective']]),
        t_refs=sorted(set(r['effective_config'].get('t_ref') for r in trials if r['family']==family)),
        nonfinite_train_loss_rows=sum(c.get('train_loss') is not None and not math.isfinite(c['train_loss']) for c in family_curves)))
(BASE/'training-summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('learned_parameters','by_track_family_size','by_family')},indent=2))
