"""Post-hoc worst independent-field contrasts; initial-mode stiffness is diagnostic."""
import json,math,statistics as st
from collections import defaultdict
from review_io import load,BASE
r=load('aggregate/endpoint_rows.json')['rows'];p=load('aggregate/protocol.json');parents={x['parent_id']:x for x in p['parents']};ix={(x['family'],x['track'],x['seed'],x['train_count'],x['parent_id'],x['grid'],x['schedule_id']):x for x in r};pairs=[]
for a in r:
 if not (a['primary'] and a['family']=='two_basis' and a['train_count']==64):continue
 par=parents[a['parent_id']];n=a['grid'];s=max(a['schedule']);k=par['kappa'];modes=[v['mode'] for v in par['field_terms']]
 lam=max(sum((2*math.pi*m[i]/par['lengths'][i])**2 if a['track']=='continuum' else (2*n/par['lengths'][i]*math.sin(math.pi*m[i]/n))**2 for i in range(2)) for m in modes)*k
 for c in ['quad2_full','quad4_full','commutator_cubic']:
  b=ix[(c,a['track'],None,0,a['parent_id'],n,a['schedule_id'])];pairs.append(dict(control=c,track=a['track'],field=a['parent_id'],regime=a['regime'],distribution=a['distribution'],grid=n,seed=a['seed'],time=a['final_time'],error=a['error_rms'],control_error=b['error_rms'],ratio=b['error_rms']/a['error_rms'],h_max_diffusion_initial_modes=s*lam,h_reaction=s*par['reaction_rate'],kappa=k,reaction_rate=par['reaction_rate'],mean=par['mean'],variance=par['variance']))
out=[];dd=defaultdict(list)
for x in pairs:dd[tuple(x[k] for k in ['control','track','time','field'])].append(x)
for key,v in dd.items():
 z=dict(v[0]);z['ratio']=math.exp(st.mean(math.log(x['ratio']) for x in v));z['error']=max(x['error'] for x in v);z['control_error']=max(x['control_error'] for x in v);z['h_max_diffusion_initial_modes_range']=[min(x['h_max_diffusion_initial_modes'] for x in v),max(x['h_max_diffusion_initial_modes'] for x in v)];z['query_count']=len(v)
 for k in ['grid','seed','h_max_diffusion_initial_modes']:z.pop(k)
 out.append(z)
summary={c:sorted([x for x in out if x['control']==c],key=lambda x:x['ratio'])[:8] for c in ['quad2_full','quad4_full','commutator_cubic']}
(BASE/'statistics-worst-cases.json').write_text(json.dumps(summary,indent=2))
