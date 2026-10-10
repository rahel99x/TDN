"""Post-hoc CPU algebraic response inspection, not new PDE accuracy evidence."""
import json
import math
import statistics
from pathlib import Path
import torch
from review_io import load
from tdn.analysis.portfolio.models import make_model, physical_features
from tdn.analysis.frontier.data import field_state, geometry
from tdn.analysis.frontier.neural import eq_geom
from tdn.numerics.operators import broadcast_h

torch.set_num_threads(1)
BASE=Path(__file__).resolve().parent
protocol=load('train-A-000/protocol.json')
catalog=load('freeze/catalog.json')['records']
families={'quad2_conditioned','conditioned_rich','band_gain','residual_quad2','quad4_conditioned','quad2_linear'}
rows=[r for r in catalog if r['family'] in families and r['train_count']==32 and (r['family']!='quad2_linear' or r['seed']==6100011)]
parents=[p for p in protocol['parents'] if p['split'] in ('train','validation','confirmation')]
horizons=[.03,.06,.10,.12,.24]
cache={}
def features(parent,n,h,track,rich):
    key=(parent['parent_id'],n,h,track,rich)
    if key not in cache:
        u=field_state(parent,n).float();eq,geom=eq_geom(parent,n)
        cache[key]=physical_features(u,broadcast_h(h,u),eq,geom,track,rich=rich)
    return cache[key]
def gain(model,x):
    if model.family=='quad2_linear':
        design=torch.cat((torch.ones_like(x[:,:1]),x),1)
        return 1+.75*torch.tanh(design@model.linear_coefficients[:,None])
    raw=model.conditioner(x)
    if model.family in ('band_gain','residual_quad2'):
        return 1+(.5 if model.family=='band_gain' else .25)*torch.tanh(raw)
    return 1+.75*torch.tanh(model.amplitude_logit+raw)
def stat(x):return dict(min=min(x),max=max(x),mean=statistics.mean(x),stdev=statistics.pstdev(x))
answers=[];max_probe_discrepancy=0.
for row in rows:
    model=make_model(row['family'],row['track'],row['effective_config']).float().eval()
    state={k:torch.tensor(v,dtype=torch.float32) for k,v in row['parameter_report']['state_dict'].items()}
    model.load_state_dict(state,strict=True)
    with torch.no_grad():
        for probe in row['response_probes']:
            if not probe.get('response'):continue
            x=torch.tensor(probe['response']['features'],dtype=torch.float32)
            discrepancy=(gain(model,x)-torch.tensor(probe['response']['gain'])).abs().max().item()
            max_probe_discrepancy=max(max_probe_discrepancy,discrepancy)
        observations=[]
        for parent in parents:
            for n in (32,64):
                for h in horizons:
                    x=features(parent,n,h,row['track'],row['family']=='conditioned_rich')
                    value=gain(model,x)[0].tolist()
                    observations.append(dict(parent_id=parent['parent_id'],split=parent['split'],regime=parent['regime'],grid=n,h=h,gain=value))
    summaries=[]
    for split in ('train','validation','confirmation'):
        for grid in (32,64):
            selected=[r for r in observations if r['split']==split and r['grid']==grid]
            summaries.append(dict(split=split,grid=grid,independent_fields=len({r['parent_id'] for r in selected}),
                components=[stat([r['gain'][j] for r in selected]) for j in range(len(selected[0]['gain']))]))
    answers.append({k:row[k] for k in ('model_id','family','track','train_count','seed')}|dict(summaries=summaries,observations=observations))
assert max_probe_discrepancy<2e-6,max_probe_discrepancy
out=dict(scope='EXPLORATORY POST-HOC CPU RESPONSE AUDIT; no integration, references, fitting, performance timings or accuracy claims; original parameters reconstructed from exact report tensors; original initial fields regenerated from protocol',
    horizons=horizons,grids=[32,64],max_saved_probe_discrepancy=max_probe_discrepancy,records=answers)
(BASE/'training-response-audit.json').write_text(json.dumps(out,indent=2)+'\n')
print('models',len(answers),'max saved-probe discrepancy',max_probe_discrepancy)
for row in answers:
    print(row['model_id'],[(s['grid'],s['components']) for s in row['summaries'] if s['split']=='confirmation'])
