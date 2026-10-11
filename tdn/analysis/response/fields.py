"""Versioned independent continuous-field parents and grouped descendants."""
from __future__ import annotations
import hashlib
import json
import math
import numpy as np
import torch

GENERATOR='tdn.response-fields/v1'
REGIMES=('isolated','two_modes','triads','smooth','peaks','holes','interface')

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def stable_seed(root_seed,*ids):
    return int(digest([int(root_seed),*ids])[:16],16) % (2**63-1)

def make_parent(root_seed,split,index,regime=None,dimension=2,lengths=None,max_mode=3):
    if dimension not in (1,2) or type(index) is not int or index<0:
        raise ValueError('Response fields support one or two dimensions and nonnegative parent indices')
    tiny_regime=('low_mean','high_mean','large_amplitude','strong_diffusion','strong_reaction','phase_twin','precision_floor','exact_null')[index%8]
    regime=regime or ('triads' if tiny_regime=='phase_twin' else REGIMES[index%len(REGIMES)])
    if regime not in REGIMES:raise ValueError('Unknown field regime')
    lengths=list(lengths or [1.]*dimension)
    if len(lengths)!=dimension or any(not math.isfinite(x) or x<=0 for x in lengths):raise ValueError('Invalid box lengths')
    seed=stable_seed(root_seed,split,index,'shape'); rng=np.random.default_rng(seed)
    # Independent shape amplitudes and physical draws prevent translated copies
    # of a fixed mode from masquerading as independent train/test fields.
    mean=float(rng.choice([.03,.08,.15,.35,.5,.65,.85,.92,.97]))
    amplitude=float(min(mean,1-mean)*rng.uniform(.1,.94))
    if dimension==1:
        modes=[(1,),(2,),(3,)]
    else:
        modes=[(1,0),(0,1),(1,1),(2,1),(1,-1),(2,0),(0,2),(3,1)]
    if type(max_mode) is not int or max_mode<3 or max_mode>64:raise ValueError('Bounded physical bandwidth must be 3..64')
    if max_mode>3 and regime in ('smooth','peaks','holes','interface'):
        candidates=[(k,) for k in range(4,max_mode+1)] if dimension==1 else [(int(a),int(b)) for a in range(1,max_mode+1) for b in range(-max_mode,max_mode+1) if max(abs(a),abs(b))>3]
        take=min(len(candidates),16)
        for i in rng.choice(len(candidates),take,replace=False):modes.append(candidates[int(i)])
    if regime=='isolated':modes=modes[:1]
    elif regime=='two_modes':modes=modes[:2]
    elif regime=='triads':modes=modes[:3 if dimension==2 else 3]
    if regime in ('peaks','holes','interface'):
        center=rng.uniform(0,1,dimension)
        phases=[float(-2*math.pi*np.dot(mode,center)) for mode in modes]
        weights=np.array([math.exp(-rng.uniform(.08,.4)*sum(k*k for k in mode)) for mode in modes])
        if regime=='holes':weights=-weights
        if regime=='interface':weights=np.array([(-1.)**j/(1+sum(abs(k) for k in m)) for j,m in enumerate(modes)])
    else:
        slope=float(rng.uniform(.5,3));anisotropy=rng.uniform(.5,2,dimension)
        phases=rng.uniform(-math.pi,math.pi,len(modes)).tolist()
        weights=np.array([rng.uniform(.4,1.4)/(sum(a*k*k for a,k in zip(anisotropy,m))**(slope/2)) for m in modes])
    weights=weights/np.abs(weights).sum()
    if tiny_regime=='low_mean':mean=float(rng.uniform(.02,.05));amplitude=float(mean*rng.uniform(.2,.85))
    elif tiny_regime=='high_mean':mean=float(rng.uniform(.95,.98));amplitude=float((1-mean)*rng.uniform(.2,.85))
    elif tiny_regime=='large_amplitude':mean=.5;amplitude=.46
    elif tiny_regime=='precision_floor':amplitude=1e-7
    elif tiny_regime=='exact_null':mean=float(rng.uniform(.03,.97));amplitude=0.
    terms=[dict(mode=list(m),amplitude=float(amplitude*w),phase=float(p)) for m,w,p in zip(modes,weights,phases)]
    physics_rng=np.random.default_rng(stable_seed(root_seed,split,index,'physics'))
    parent=dict(generator=GENERATOR,parent_id=f'response-{split}-{index:06d}-{seed:016x}',split=split,index=index,
        seed=seed,regime=regime,tiny_regime=tiny_regime,dimension=dimension,lengths=lengths,mean=mean,amplitude_bound=amplitude,terms=terms,
        kappa=float(physics_rng.choice([3e-4,.003,.03])),reaction_rate=float(physics_rng.choice([.5,3.,8.,25.,60.])),
        shape_scope='fixed physical Fourier modes; analytically bounded without clipping',rejected_draws=0,bandwidth_max_mode=max_mode)
    if tiny_regime=='strong_diffusion':parent['kappa']=.03
    if tiny_regime=='strong_reaction':parent['reaction_rate']=60.
    parent['translation_invariant_shape_hash']=digest(dict(mean=mean,terms=sorted((tuple(t['mode']),abs(t['amplitude'])) for t in terms)))
    parent['parent_sha256']=digest(parent)
    return parent

def parent_specs(protocol,split):
    declared=[p for p in protocol.get('parents',[]) if p.get('split')==split]
    if declared:
        result=[]
        for i,p in enumerate(declared):
            if 'terms' in p:result.append(p)
            else:
                q=make_parent(protocol.get('root_seed',protocol.get('seed',98100000)),split,p.get('index',i),
                    p.get('regime'),p.get('dimension',protocol.get('dimension',2)),p.get('lengths',protocol.get('lengths')),max_mode=protocol.get('field_max_mode',max(3,protocol.get('train_grid',8)//4)))
                q.update({k:v for k,v in p.items() if k not in ('parent_sha256','translation_invariant_shape_hash')})
                q['parent_sha256']=digest({k:v for k,v in q.items() if k!='parent_sha256'});result.append(q)
        return result
    counts=protocol.get('parent_counts',protocol.get('data',{}).get('parent_counts',{}))
    count=counts.get(split, {'train':4,'validation':2,'development':2,'development_test':2}.get(split,2))
    return [make_parent(protocol.get('root_seed',protocol.get('seed',98100000)),split,i,
        dimension=protocol.get('dimension',2),lengths=protocol.get('lengths'),max_mode=protocol.get('field_max_mode',max(3,protocol.get('train_grid',8)//4))) for i in range(count)]

def validate_parent_splits(parents):
    ids={};shapes={}
    for p in parents:
        if p['parent_id'] in ids:raise ValueError('Duplicate parent ID')
        ids[p['parent_id']]=p['split']
        shape=p.get('translation_invariant_shape_hash')
        if shape in shapes and shapes[shape]!=p['split']:
            raise ValueError('Identical or translated-only shape appears across splits')
        shapes[shape]=p['split']
    return True

def sample_parent(parent,grid,*,dtype=torch.float64,device='cpu'):
    grid=(grid,)*parent['dimension'] if isinstance(grid,int) else tuple(grid)
    if len(grid)!=parent['dimension'] or any(type(n) is not int or n<2 for n in grid):raise ValueError('Invalid field grid')
    axes=torch.meshgrid(*[torch.arange(n,dtype=torch.float64,device=device)/n for n in grid],indexing='ij')
    u=torch.full(grid,parent['mean'],dtype=torch.float64,device=device)
    for term in parent['terms']:
        phase=sum(k*x for k,x in zip(term['mode'],axes))*2*math.pi+term['phase']
        u=u+term['amplitude']*torch.cos(phase)
    return u[None,None].to(dtype)

def array_digest(value):
    a=value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def phase_variant(parent,variant):
    """Paired phase-only descendant, never a new independent parent."""
    import copy
    if variant not in ('original','phase_twin'):raise ValueError('Unknown paired phase intervention')
    result=copy.deepcopy(parent);result['phase_variant']=variant
    has_triad=all(any(tuple(t['mode'])==mode for t in result['terms']) for mode in
        ([(1,),(2,),(3,)] if parent['dimension']==1 else [(1,0),(0,1),(1,1)]))
    if variant=='phase_twin':
        index=2 if len(result['terms'])>=3 else 0
        result['terms'][index]['phase']+=math.pi
    result['phase_scope']='relative-triad-phase intervention at identical continuous modal amplitudes' if has_triad else 'translation-only or nonclosed-mode control; no higher-interaction claim'
    result['descendant_sha256']=digest({k:v for k,v in result.items() if k!='descendant_sha256'})
    return result


def spectral_stress_parent(parent,coarse_grid,kind='grid_relative'):
    """A paired, explicitly different physical workload, never grid transfer.

    Frequencies stay fixed during teacher refinement of this declared case.
    They intentionally change when *coarse_grid* is changed for grid-relative
    stress. Every field remains bounded analytically by its coefficient L1 sum.
    """
    import copy
    if kind not in ('grid_relative','broadband') or coarse_grid<8:raise ValueError('Declared stress needs grid >=8')
    p=copy.deepcopy(parent);n=int(coarse_grid);k=max(2,n//2-1)
    modes=[(k,),(k-1,),(1,)] if p['dimension']==1 else [(k,0),(k-1,1),(1,-1),(1,0),(0,k),(1,k-1)]
    if kind=='broadband':
        upper=max(3,n//3)
        modes=[(j,) for j in range(1,upper+1)] if p['dimension']==1 else [(j,j%3-1) for j in range(1,upper+1)]+[(0,j) for j in range(1,upper+1)]
    rng=np.random.default_rng(stable_seed(parent['seed'],kind,n));weights=rng.uniform(.5,1.5,len(modes));weights/=weights.sum()
    amplitude=min(.35,min(p['mean'],1-p['mean'])*.85)
    p['terms']=[dict(mode=list(m),amplitude=float(amplitude*w),phase=float(phi)) for m,w,phi in zip(modes,weights,rng.uniform(-math.pi,math.pi,len(modes)))]
    p.update(regime=kind,amplitude_bound=amplitude,frequency_scope='declared coarse-grid-relative stress; fixed frequencies during reference refinement',declared_coarse_grid=n,
        stress_parent_id=parent['parent_id'],parent_id=parent['parent_id']+f'/{kind}/n{n}',independence_scope='paired descendant workload; not an additional independent parent')
    p['translation_invariant_shape_hash']=digest(dict(mean=p['mean'],terms=sorted((tuple(t['mode']),abs(t['amplitude'])) for t in p['terms'])))
    p['parent_sha256']=digest({k:v for k,v in p.items() if k!='parent_sha256'})
    return p


def grid_relative_parent(parent,coarse_grid):
    """Scale the same declared Fourier pattern toward the deployment cutoff.

    This changes the physical field with grid and therefore is a stress
    descendant of the original parent, never a resolution-transfer example.
    Signed integer modes, phases and amplitudes are retained; modes are capped
    below Nyquist to avoid phase-dependent collapse of the continuous power.
    """
    import copy
    n=int(coarse_grid)
    if n<8:raise ValueError('Grid-relative stress requires grid >=8')
    p=copy.deepcopy(parent);scale=max(1,n//8);cap=n//2-1
    for term in p['terms']:
        term['mode']=[int(np.sign(k))*min(abs(k)*scale,cap) for k in term['mode']]
    p.update(panel='grid_relative_stress',descriptive_only=True,declared_coarse_grid=n,mode_scale=scale,
        frequency_scope='Physical field changes with coarse grid; fixed integer modes during reference refinement',
        independence_scope='paired stress descendant of unchanged parent_id; not an additional independent parent')
    p['descendant_sha256']=digest({k:v for k,v in p.items() if k!='descendant_sha256'})
    return p
