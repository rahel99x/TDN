"""Immutable continuous-field parents, prospective queries and private payloads."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch

from tdn.runtime.metadata import write_json
from tdn.analysis.response.numerics import anchor, diffusion_symbol
from .contracts import Physics, Geometry, PrecisionContract, digest, equation_geometry, physical_metrics


def field_spec(parent, base_grid):
    rng=np.random.default_rng(parent['seed']);regime=parent['regime']
    high=max(1,min(base_grid)//2-1)
    modes=([(1,0),(0,2),(1,1),(2,1)] if regime!='high_frequency' else
           [(high,0),(0,high),(high,1),(1,high)])
    if regime=='localized':modes=[(1,0),(0,1),(2,0),(0,2),(1,1),(2,1)]
    if regime=='overlapping':modes=[(1,0),(0,1),(1,1),(2,1)]
    weights=rng.uniform(.4,1.,len(modes))*rng.choice([-1.,1.],len(modes))
    phases=rng.uniform(-math.pi,math.pi,len(modes))
    weights=weights/np.abs(weights).sum()
    mean_value={'low_mean':.12,'mid_mean':.5,'high_mean':.85}.get(regime,.45)
    amplitude=.32 if regime=='large_amplitude' else .22 if regime in ('localized','overlapping','mixed') else .08
    amplitude=min(amplitude,.8*min(mean_value,1-mean_value))
    shape=dict(modes=[list(k) for k in modes],weights=weights.tolist(),phases=phases.tolist())
    return dict(**shape,mean=mean_value,amplitude=amplitude,shape_sha256=digest(shape),
                construction='Continuous trigonometric field with coefficient-l1 amplitude bound; no nodal clipping',
                active_frequency=list(max(modes,key=lambda k:sum(x*x for x in k))))


def materialize_parent(parent,spec,grid,*,phase_shift=0.,device='cpu'):
    axes=[torch.arange(n,dtype=torch.float64,device=device)/n for n in grid]
    xx=torch.meshgrid(*axes,indexing='ij')
    value=torch.zeros(grid,dtype=torch.float64,device=device)
    for index,(mode,weight,phase) in enumerate(zip(spec['modes'],spec['weights'],spec['phases'])):
        # Designed relative phase changes preserve every modal magnitude.
        phase=phase+2*math.pi*phase_shift*(index%2)
        angle=sum(2*math.pi*k*x for k,x in zip(mode,xx))+phase
        value=value+weight*torch.cos(angle)
    result=(spec['mean']+spec['amplitude']*value)[None,None].float()
    if not bool(torch.isfinite(result).all()) or bool((result<0).any()) or bool((result>1).any()):
        raise ValueError('Declared bounded continuous initial-field construction failed')
    return result


def parent_bank(protocol,cohort='mechanism',*,prior_shape_hashes=()):
    parents=protocol['parents'] if cohort=='mechanism' else protocol['fit_parents']
    base=protocol['geometry']['grids'][0] if cohort=='mechanism' else protocol['fit_geometry']['grid']
    seen={};rows=[];prior=set(prior_shape_hashes)
    for parent in parents:
        spec=field_spec(parent,base);shape=spec['shape_sha256']
        if shape in prior or shape in seen and seen[shape]!=parent['parent_id']:
            raise ValueError('Independent parent shape is reused across candidates/cohorts/splits')
        seen[shape]=parent['parent_id'];rows.append(dict(**parent,field_spec=spec,shape_sha256=shape))
    return dict(schema='tdn.tangent-response-parents/v1',cohort=cohort,parents=rows,
                parent_counts=protocol['parent_counts'] if cohort=='mechanism' else protocol['fit_parent_counts'],
                protocol_sha256=digest(protocol),identity_scope='Independent continuous field; phase/horizon/grid/physics are paired descendants')


def make_query(protocol,bank,spec,*,device='cpu'):
    menu=protocol['queries'] if bank['cohort']=='mechanism' else protocol['fit_queries']
    if spec not in menu or bank.get('protocol_sha256')!=digest(protocol):
        raise ValueError('Query and parent bank must match the prospectively frozen protocol')
    parent=next(p for p in bank['parents'] if p['parent_id']==spec['parent_id'])
    record=copy.deepcopy(spec);field=parent['field_spec']
    geom=Geometry(tuple(record['grid']),tuple(record['lengths']))
    u0=materialize_parent(parent,field,geom.grid,phase_shift=record.get('phase_shift',0.),device=device)
    if record.get('coefficient_rule')=='active_generator_at_base_grid':
        base=Geometry(tuple(record['base_grid']),tuple(record['lengths']))
        probe=torch.zeros((1,1,*base.grid),dtype=torch.float64,device=device)
        symbol=diffusion_symbol(probe,Physics(1.,0.),base,record['target_kind'])
        mode=tuple(k%n for k,n in zip(field['active_frequency'],base.grid))
        active=abs(float(symbol[mode]))
        if active<=0:raise ValueError('Physics intervention requires a nonzero declared active generator')
        record['kappa']=record['diffusion_number']/(.12*active)
        record['reaction_rate']=record['reaction_number']/.12
        record['active_generator_magnitude_at_unit_kappa']=active
    record.update(shape_sha256=parent['shape_sha256'],active_frequency=field['active_frequency'],
        field_spec_sha256=digest(field),input_sha256=hashlib.sha256(u0.detach().cpu().numpy().tobytes()).hexdigest(),
        input_dtype='float32',primary_case=record.get('panel')=='primary',
        role=record.get('panel','primary'),intervention=record.get('panel')!='primary')
    eq=Physics(record['kappa'],record['reaction_rate'])
    return dict(record=record,u0=u0,eq=eq,geom=geom)


def add_reference(sample,protocol,*,budget=None):
    from .oracle import refined_teacher
    config={**protocol['reference'],'max_seconds_per_query':protocol['reference']['seconds_per_query']}
    teacher,reference=refined_teacher(sample['u0'],sample['record']['horizon'],sample['eq'],sample['geom'],
        sample['record']['target_kind'],config=config,budget=budget)
    h=sample['record']['horizon'];target=sample['record']['target_kind'];substeps=protocol['anchor']['reaction_substeps']
    a64=anchor(sample['u0'].double(),h,sample['eq'],sample['geom'],target,name=protocol['anchor']['name'],reaction_substeps=substeps)
    a32=anchor(sample['u0'],h,sample['eq'],sample['geom'],target,name=protocol['anchor']['name'],reaction_substeps=substeps)
    values=PrecisionContract().labels(teacher,a64,a32)
    sample.update(teacher64=teacher,target=teacher,reference=reference,smooth_anchor64=a64,anchor_fp64=a64,
                  deployed_anchor32=a32,anchor_fp32=a32,**values)
    sample['record']['reference']=reference
    return sample


def save_samples(path,samples):
    """NPZ numerical payloads use allow_pickle=False; labels never enter inference."""
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    if (path/'samples.npz').exists() or (path/'samples.json').exists():
        raise ValueError('Preserve existing private numerical samples; choose a fresh stage')
    records=[];arrays={}
    aliases={'target':'teacher64','anchor_fp64':'smooth_anchor64','anchor_fp32':'deployed_anchor32'}
    for index,sample in enumerate(samples):
        metadata={'record':sample['record'],'reference':sample.get('reference'),
                  'oracle_metadata':sample.get('oracle_metadata'),'tensor_keys':[],'action_labels':[]}
        for key,value in sample.items():
            if torch.is_tensor(value) and key not in aliases:
                arrays[f's{index}/{key}']=value.detach().cpu().numpy();metadata['tensor_keys'].append(key)
        for action_index,label in enumerate(sample.get('action_labels',[])):
            item={k:v for k,v in label.items() if not torch.is_tensor(v)}
            item['tensor_keys']=[]
            for key,value in label.items():
                if torch.is_tensor(value):
                    arrays[f's{index}/a{action_index}/{key}']=value.detach().cpu().numpy();item['tensor_keys'].append(key)
            metadata['action_labels'].append(item)
        records.append(metadata)
    temporary=path/'samples.npz.partial'
    with temporary.open('xb') as stream:np.savez_compressed(stream,**arrays)
    temporary.replace(path/'samples.npz')
    write_json(path/'samples.json',dict(schema='tdn.tangent-response-samples/v1',samples=records,
        numerical_payload='samples.npz',numerical_payload_sha256=hashlib.sha256((path/'samples.npz').read_bytes()).hexdigest(),
        aliases=aliases,label_scope='Reference and oracle supervision only; never model inference arguments'))


def load_samples(path,*,device='cpu'):
    path=Path(path)
    if path.is_symlink() or (path/'samples.json').is_symlink() or (path/'samples.npz').is_symlink():
        raise ValueError('Private numerical payloads cannot traverse symlinks')
    metadata=json.loads((path/'samples.json').read_text())
    if metadata.get('schema')!='tdn.tangent-response-samples/v1' or hashlib.sha256((path/'samples.npz').read_bytes()).hexdigest()!=metadata['numerical_payload_sha256']:
        raise ValueError('Private numerical sample payload identity changed')
    result=[]
    with np.load(path/'samples.npz',allow_pickle=False) as arrays:
        for index,item in enumerate(metadata['samples']):
            sample={k:copy.deepcopy(item[k]) for k in ('record','reference','oracle_metadata') if item.get(k) is not None}
            for key in item['tensor_keys']:sample[key]=torch.as_tensor(arrays[f's{index}/{key}'].copy(),device=device)
            for alias,key in metadata['aliases'].items():
                if key in sample:sample[alias]=sample[key]
            sample['eq'],sample['geom']=equation_geometry(sample['record']);sample['action_labels']=[]
            for action_index,label in enumerate(item.get('action_labels',[])):
                action={k:copy.deepcopy(v) for k,v in label.items() if k!='tensor_keys'}
                for key in label['tensor_keys']:action[key]=torch.as_tensor(arrays[f's{index}/a{action_index}/{key}'].copy(),device=device)
                sample['action_labels'].append(action)
            result.append(sample)
    return result


def screening(samples,protocol):
    """First eligible candidate, requiring every declared horizon; never loss/speed."""
    groups={};rows=[];selected=[]
    for sample in samples:groups.setdefault(sample['record']['parent_id'],[]).append(sample)
    for split,count in protocol['fit_parent_counts'].items():
        for slot in range(count):
            choices=[p for p in protocol['fit_parents'] if p['split']==split and p['slot']==slot]
            chosen=None
            for parent in choices:
                queries=groups.get(parent['parent_id'],[]);reasons=[]
                required=[q for q in protocol['fit_queries'] if q['parent_id']==parent['parent_id']]
                if {s['record']['query_id'] for s in queries}!={q['query_id'] for q in required}:reasons.append('MISSING_REQUESTED_QUERY')
                for sample in queries:
                    ref=sample.get('reference',{});oracle=sample.get('oracle_metadata',{})
                    if not ref.get('accepted') or not oracle.get('accepted'):reasons.append('UNRESOLVED_REFERENCE_OR_TANGENT')
                    if 'physical_defect' not in sample:reasons.append('MISSING_PHYSICAL_DEFECT');continue
                    metrics=physical_metrics(sample['physical_defect'],sample['geom'],protocol['target'])
                    uncertainty=ref.get('uncertainty_rms')
                    if uncertainty is None or metrics['rms']<10*uncertainty:reasons.append('REFERENCE_HEADROOM')
                    spacing=torch.nextafter(sample['deployed_anchor32'],torch.full_like(sample['deployed_anchor32'],math.inf))-sample['deployed_anchor32']
                    if float(spacing.abs().max())>protocol['performance']['primary_tolerance']*.1:reasons.append('FP32_OUTPUT_RESOLUTION')
                if len(queries)!=len({s['record']['query_id'] for s in queries}):reasons.append('DUPLICATE_REQUESTED_QUERY')
                eligible=not reasons and bool(queries)
                if eligible and chosen is None:chosen=parent['parent_id'];selected+=queries
                rows.append(dict(parent_id=parent['parent_id'],split=split,slot=slot,candidate=parent['candidate'],
                    requested_queries=len(required),observed_queries=len(queries),eligible=eligible,
                    selected=chosen==parent['parent_id'],reasons=sorted(set(reasons))))
    complete=len({s['record']['parent_id'] for s in selected})==sum(protocol['fit_parent_counts'].values())
    return selected,dict(schema='tdn.tangent-response-screening/v1',status='ACCEPTED' if complete else 'UNRESOLVED',
        requested_parent_slots=sum(protocol['fit_parent_counts'].values()),accepted_parent_slots=len({s['record']['parent_id'] for s in selected}),
        candidates=rows,selected_parent_ids=sorted({s['record']['parent_id'] for s in selected}),
        selection='First prospectively declared eligible candidate; no trained loss or speed')
