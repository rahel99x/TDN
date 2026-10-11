"""Endpoint and uncertainty metrics with explicit missing/failure denominators."""
from __future__ import annotations
import math
import numpy as np
import torch


def norm_metrics(x):
    if not bool(torch.isfinite(x).all()):return dict(rms=None,maximum=None,mean=None,centered_rms=None)
    x=x.double();mean=x.mean();centered=x-mean
    return dict(rms=float(x.square().mean().sqrt()),maximum=float(x.abs().max()),mean=float(mean),centered_rms=float(centered.square().mean().sqrt()))

def uncertainty_interval(error,uncertainty):
    if error is None or uncertainty is None or not math.isfinite(error) or not math.isfinite(uncertainty) or min(error,uncertainty)<0:
        return {'lower':None,'upper':None,'status':'NOT_MEASURED','reason':'Finite norm and deterministic uncertainty estimate required'}
    return {'lower':max(0.,error-uncertainty),'upper':error+uncertainty,'status':'ESTIMATED_SENSITIVITY_NOT_CONFIDENCE_INTERVAL'}

def accuracy_status(rms,maximum,reference,tolerance=2e-5):
    if not reference.get('accepted',False):return 'UNRESOLVED_REFERENCE'
    intervals=[uncertainty_interval(rms,reference.get('uncertainty_rms')),uncertainty_interval(maximum,reference.get('uncertainty_maximum'))]
    if any(x['upper'] is None for x in intervals):return 'NOT_MEASURED'
    if all(x['upper']<=tolerance for x in intervals):return 'QUALIFIED'
    if any(x['lower']>tolerance for x in intervals):return 'FAILED_TARGET'
    return 'UNCERTAINTY_STRADDLES_TARGET'

def endpoint_metrics(prediction,target,*,anchor=None,u0=None,record=None,tolerance=2e-5,relative_floor=1e-8):
    record=record or {};reference=record.get('reference',{});result={k:record.get(k) for k in
        ('parent_id','query_id','split','regime','tiny_regime','shape_regime','phase_variant','phase_scope','intervention_variant','intervention_metadata','panel','descriptive_only','frequency_scope','target_kind','grid','lengths','horizon','kappa','reaction_rate')}
    result['norm']='nodal_rms' if record.get('target_kind')=='fd_same_grid' else 'trigonometric_interpolant_rms'
    result['maximum_norm']='sampled_nodal_maximum; not a continuous supremum certificate'
    result['precision_audit']=record.get('precision')
    result['phase_matching']=record.get('phase_matching')
    result.update(reference_accepted=bool(reference.get('accepted',False)),reference_uncertainty_rms=reference.get('uncertainty_rms'),
        reference_uncertainty_maximum=reference.get('uncertainty_maximum'),reference_status=reference.get('status','NOT_MEASURED'))
    finite=prediction is not None and bool(torch.isfinite(prediction).all())
    target_finite=target is not None and bool(torch.isfinite(target).all())
    result.update(prediction_finite=finite,target_finite=target_finite,status='COMPLETED' if finite else 'FAILED',
        error_rms=None,error_max=None,rms=None,maximum=None,mean_error=None,centered_rms=None,relative_rms=None,accuracy_qualification='NOT_MEASURED',
        uncertainty_rms=reference.get('uncertainty_rms'),uncertainty_maximum=reference.get('uncertainty_maximum'),bound_violation=None,reference_bounded=None)
    if not finite or not target_finite:
        result['unavailable_reason']='Nonfinite or missing prediction' if not finite else 'Nonfinite or missing teacher';return result
    if prediction.shape!=target.shape:raise ValueError('Predictions and teachers must have identical shapes')
    p=prediction.double();t=target.double();e=p-t;m=norm_metrics(e)
    result.update(error_rms=m['rms'],error_max=m['maximum'],mean_error=m['mean'],absolute_mean_error=abs(m['mean']),
        centered_rms=m['centered_rms'],relative_rms=m['rms']/max(relative_floor,float(t.square().mean().sqrt())),
        relative_denominator_floor=relative_floor,mean_fraction_squared_error=(m['mean']**2/m['rms']**2 if m['rms']>0 else None),
        mean_centered_decomposition_residual=m['rms']**2-m['mean']**2-m['centered_rms']**2,
        prediction_min=float(p.min()),prediction_max=float(p.max()),reference_min=float(t.min()),reference_max=float(t.max()),prediction_mean=float(p.mean()),reference_mean=float(t.mean()),
        interval_violation=float(torch.maximum(-p,p-1).clamp_min(0).max()),
        reference_interval_violation=float(torch.maximum(-t,t-1).clamp_min(0).max()),
        prediction_variance=float((p-p.mean()).square().mean()),reference_variance=float((t-t.mean()).square().mean()),
        rms_uncertainty_interval=uncertainty_interval(m['rms'],reference.get('uncertainty_rms')),
        max_uncertainty_interval=uncertainty_interval(m['maximum'],reference.get('uncertainty_maximum')),
        accuracy_qualification=accuracy_status(m['rms'],m['maximum'],reference,tolerance))
    result.update(rms=result['error_rms'],maximum=result['error_max'],uncertainty_rms=reference.get('uncertainty_rms'),uncertainty_maximum=reference.get('uncertainty_maximum'),bound_violation=result['interval_violation']>max(8*torch.finfo(torch.float32).eps,reference.get('uncertainty_maximum') or 0.),reference_bounded=result['reference_interval_violation']<=max(8*torch.finfo(torch.float32).eps,reference.get('uncertainty_maximum') or 0.))
    if not reference.get('accepted',False):result['unavailable_reason']='Errors relative to unresolved teacher are diagnostic only'
    if anchor is not None:
        a=anchor.double();correction=p-a;defect=t-a;c=norm_metrics(correction);d=norm_metrics(defect)
        den=float(correction.square().sum().sqrt()*defect.square().sum().sqrt())
        result.update(anchor_error_rms=d['rms'],correction_rms=c['rms'],target_defect_rms=d['rms'],
            correction_defect_ratio=(c['rms']/d['rms'] if d['rms']>0 else None),
            residual_alignment=(float((correction*defect).sum())/den if den>0 else None),
            represented_zero_correction_fraction=float((p==a).double().mean()),
            fp32_assembly_lost_fraction=None,
            fp32_assembly_loss_unavailable_reason='Pre-rounding correction components are not supplied; rounded prediction minus anchor cannot identify lost corrections')
    axes=tuple(range(2,p.ndim));lengths=record.get('lengths',[1.]*(p.ndim-2))
    errhat=torch.fft.fftn(e,dim=axes,norm='forward');truehat=torch.fft.fftn(t,dim=axes,norm='forward')
    phat=torch.fft.fftn(p,dim=axes,norm='forward');freq2=torch.zeros(p.shape[2:],device=p.device,dtype=p.dtype)
    for axis,(n,L) in enumerate(zip(p.shape[2:],lengths)):
        shape=[1]*(p.ndim-2);shape[axis]=n;k=torch.fft.fftfreq(n,d=L/n,device=p.device,dtype=p.dtype)
        freq2+=k.square().reshape(shape)
    bands=[]
    for lower,upper in [(0.,2.),(2.,4.),(4.,8.),(8.,None)]:
        mask=freq2>=lower*lower
        if upper is not None:mask=mask&(freq2<upper*upper)
        bands.append(dict(lower_cycles_per_length=lower,upper_cycles_per_length=upper,
            coefficient_error_energy=float(errhat[...,mask].abs().square().sum()),modes=int(mask.sum())))
    result['spectral_bands']=bands
    spectral_den=float(phat.abs().square().sum().sqrt()*truehat.abs().square().sum().sqrt())
    result['spectral_complex_coherence']=float((phat.conj()*truehat).sum().abs())/spectral_den if spectral_den>0 else None
    if record.get('target_kind') in ('galerkin_same_grid','galerkin_projected_fine'):
        weights=torch.ones(p.shape[2:],dtype=p.dtype,device=p.device)
        for axis,n in enumerate(p.shape[2:]):
            if n%2==0:
                index=[slice(None)]*(p.ndim-2);index[axis]=n//2;weights[tuple(index)]*=.5
        result['interpolant_error_rms']=float((errhat.abs().square()*weights).sum(axes).mean().sqrt())
    else:result['interpolant_error_rms']=None
    result['rms']=result['interpolant_error_rms'] if result['interpolant_error_rms'] is not None else result['error_rms']
    result['norm']='trigonometric_interpolant_rms' if result['interpolant_error_rms'] is not None else 'nodal_rms'
    result['maximum_norm']='sampled_nodal_maximum; not a continuous supremum certificate'
    result['accuracy_qualification']=accuracy_status(result['rms'],result['maximum'],reference,tolerance)
    # Preserve sampled diagnostics explicitly, but use a single compatible norm
    # for the primary decomposition and correction attribution.  Even Nyquist
    # modes distinguish the interpolant norm from the nodal norm.
    result['error_rms_norm']='nodal_rms'
    for key in ('centered_rms','relative_rms','mean_fraction_squared_error','mean_centered_decomposition_residual',
                'rms_uncertainty_interval','anchor_error_rms','correction_rms','target_defect_rms','correction_defect_ratio','residual_alignment'):
        if key in result:result['nodal_'+key]=result[key]
    if result['interpolant_error_rms'] is not None:
        def physical_inner(a,b):
            ah=torch.fft.fftn(a,dim=axes,norm='forward');bh=torch.fft.fftn(b,dim=axes,norm='forward')
            return float(((ah.conj()*bh).real*weights).sum(axes).mean())
        centered2=max(0.,physical_inner(e-e.mean(),e-e.mean()))
        result['centered_rms']=math.sqrt(centered2)
        result['relative_rms']=result['rms']/max(relative_floor,math.sqrt(max(0.,physical_inner(t,t))))
        if anchor is not None:
            correction=p-anchor.double();defect=t-anchor.double()
            correction2=max(0.,physical_inner(correction,correction));defect2=max(0.,physical_inner(defect,defect))
            result.update(anchor_error_rms=math.sqrt(defect2),correction_rms=math.sqrt(correction2),target_defect_rms=math.sqrt(defect2),
                correction_defect_ratio=math.sqrt(correction2/defect2) if defect2>0 else None,
                residual_alignment=physical_inner(correction,defect)/math.sqrt(correction2*defect2) if correction2*defect2>0 else None)
    result['mean_fraction_squared_error']=result['mean_error']**2/result['rms']**2 if result['rms']>0 else None
    result['mean_centered_decomposition_residual']=result['rms']**2-result['mean_error']**2-result['centered_rms']**2
    result['rms_uncertainty_interval']=uncertainty_interval(result['rms'],reference.get('uncertainty_rms'))
    try:
        from .references import equation_geometry
        from .numerics import rhs,inner,product
        eq,geom=equation_geometry(record);kind=record['target_kind']
        def energy(x):
            reaction=eq.reaction_rate*(x-product(x,x,geom,kind));linear=rhs(x,eq,geom,kind)-reaction
            return float((-.5*inner(x,linear,geom,kind)-.5*eq.reaction_rate*inner(x,x,geom,kind)
                         +eq.reaction_rate/3*inner(x,product(x,x,geom,kind),geom,kind)).sum())
        result['prediction_energy']=energy(p);result['reference_energy']=energy(t)
        result['prediction_physical_variance']=float(inner(p-p.mean(),p-p.mean(),geom,kind).mean())
        result['reference_physical_variance']=float(inner(t-t.mean(),t-t.mean(),geom,kind).mean())
        result['energy_change']=energy(p)-energy(u0.double()) if u0 is not None else None
        result['energy_scope']='Compatible finite-equation energy; projected-fine coarse energy is a diagnostic, not exact projected dynamics'
    except (ImportError,KeyError,ValueError,TypeError) as exc:
        result['prediction_energy']=None;result['reference_energy']=None;result['energy_unavailable_reason']=str(exc)
    return result

def summarize_endpoints(rows):
    attempted=len(rows);accepted=[r for r in rows if r.get('reference_accepted')]
    valid=[r for r in accepted if r.get('prediction_finite') and r.get('error_rms') is not None]
    values=np.array([r['error_rms'] for r in valid])
    result=dict(attempted_queries=attempted,attempted_parents=len({r.get('parent_id') for r in rows}),
        accepted_reference_queries=len(accepted),unresolved_reference_queries=attempted-len(accepted),
        prediction_failures_with_valid_reference=len(accepted)-len(valid),finite_valid_queries=len(valid),
        qualified_queries=sum(r.get('accuracy_qualification')=='QUALIFIED' for r in rows),
        error_rms_mean=float(values.mean()) if len(values) else None,error_rms_median=float(np.median(values)) if len(values) else None,
        worst_observed_rms=float(values.max()) if len(values) else None,
        missing_values_are_zero=False,universal_leaderboard=False)
    # Never label sparse sample percentiles as stable tail estimates.
    result['tail_quantiles']={str(q):float(np.quantile(values,q)) if len(values)*(1-q)>=5 else None for q in [.9,.95,.99]}
    return result
