"""Bounded FP64 teachers for the actual deployed input; safe auditable datasets."""
from __future__ import annotations
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from .fields import parent_specs,sample_parent,array_digest,digest,validate_parent_splits,phase_variant,grid_relative_parent

TARGETS=('fd_same_grid','galerkin_same_grid','galerkin_projected_fine')

def equation_geometry(record):
    from .numerics import Physics
    from tdn.numerics.types import Geometry
    return Physics(record['kappa'],record['reaction_rate']),Geometry(tuple(record['grid']),tuple(record['lengths']))

def _norms(x):
    if not bool(torch.isfinite(x).all()):return {'rms':None,'maximum':None}
    return {'rms':float(x.double().square().mean().sqrt()),'maximum':float(x.double().abs().max())}

def _physical_norms(x, geometry, target):
    """Physical interpolant L2, with sampled maximum kept explicitly separate."""
    from .numerics import inner
    if not bool(torch.isfinite(x).all()):return {'rms':None,'maximum':None}
    return {'rms':float(inner(x.double(),x.double(),geometry,target).mean().clamp_min(0).sqrt()),
        'maximum':float(x.double().abs().max())}

def projected_error_budget(options):
    """Validate an optional component allocation without relaxing acceptance.

    The historical default uses the total tolerance for each temporal solve.
    New diagnostic protocols may allocate separate temporal tolerances for each
    declared fine grid and a spatial tolerance. Every adjacent pair must fit
    within the unchanged total uncertainty allowance.
    """
    allocation=options.get('projected_error_budget')
    if allocation is None:return None
    if not isinstance(allocation,dict):raise ValueError('Projected error budget must be a mapping')
    factors=list(options.get('spatial_factors',[2,4]))
    temporal=allocation.get('temporal_tolerances')
    if not isinstance(temporal,(list,tuple)) or len(temporal)!=len(factors) or len(factors)<2:
        raise ValueError('Temporal component tolerances must match at least two spatial factors')
    if any(type(f) is not int or f<2 for f in factors) or any(b<=a for a,b in zip(factors,factors[1:])):
        raise ValueError('Projected spatial factors must be strictly increasing integers >=2')
    def positive(value):
        if isinstance(value,bool):raise ValueError('Reference budgets must be positive finite numbers')
        try:value=float(value)
        except (TypeError,ValueError):raise ValueError('Reference budgets must be positive finite numbers') from None
        if not math.isfinite(value) or value<=0:raise ValueError('Reference budgets must be positive finite numbers')
        return value
    total=positive(options.get('tolerance',1e-10));spatial=positive(allocation.get('spatial_tolerance'))
    temporal=[positive(x) for x in temporal]
    if any(math.fsum([a,b,spatial])>total for a,b in zip(temporal,temporal[1:])):
        raise ValueError('Adjacent temporal and spatial budgets must sum to at most total tolerance')
    return dict(total_tolerance=total,temporal_tolerances=temporal,spatial_tolerance=spatial,
        adjacent_combined_tolerances=[math.fsum([a,b,spatial]) for a,b in zip(temporal,temporal[1:])],
        policy='Separate component tolerances; combined uncertainty acceptance remains unchanged')

def _lawson(initial,h,equation,geometry,target,steps,check):
    """Exact-linear-flow RK4, separate from deployed anchors and ETDRK4."""
    from .numerics import diffusion_symbol,product
    dt=h/steps; u=initial.clone();axes=tuple(range(2,u.ndim))
    symbol=diffusion_symbol(initial,equation,geometry,target)
    full=torch.exp(dt*symbol);half=torch.exp(dt*.5*symbol)
    def flow(x,t):
        return torch.fft.ifftn(torch.fft.fftn(x,dim=axes)*(half if t==dt/2 else full),dim=axes).real
    def nonlinear(x):return equation.reaction_rate*(x-product(x,x,geometry,target))
    with torch.no_grad():
        for i in range(steps):
            if i%8==0:check()
            a=nonlinear(u);b=nonlinear(flow(u+dt*a/2,dt/2))
            c=nonlinear(flow(u,dt/2)+dt*b/2)
            d=nonlinear(flow(u,dt)+dt*flow(c,dt/2))
            u=flow(u,dt)+dt/6*(flow(a,dt)+2*flow(b+c,dt/2)+d)
    check();return u

def _scipy_crosscheck(initial,h,equation,geometry,target,tolerance,check):
    """DOP853 + independently constructed NumPy/scipy-resample RHS.

    Independent integration and FFT/product construction; shares only the
    equation specification. Explicit adaptation may exhaust the bounded budget.
    """
    from scipy.integrate import solve_ivp
    from scipy.signal import resample
    u=initial.detach().cpu().numpy();shape=u.shape;axes=tuple(range(2,u.ndim))
    symbol=np.zeros(geometry.grid,np.float64)
    for axis,(n,L) in enumerate(zip(geometry.grid,geometry.lengths)):
        k=np.fft.fftfreq(n)*n
        v=-4*(n/L)**2*np.sin(np.pi*k/n)**2 if target=='fd_same_grid' else -(2*np.pi*k/L)**2
        s=[1]*geometry.ndim;s[axis]=n;symbol+=float(equation.kappa)*v.reshape(s)
    def rhs(t,flat):
        check();v=flat.reshape(shape)
        diffusion=np.fft.ifftn(np.fft.fftn(v,axes=axes)*symbol,axes=axes).real
        if target=='fd_same_grid':square=v*v
        else:
            fine=v
            for axis,n in zip(axes,geometry.grid):fine=resample(fine,2*n+1,axis=axis)
            square=fine*fine
            for axis,n in zip(axes,geometry.grid):square=resample(square,n,axis=axis)
        return (diffusion+float(equation.reaction_rate)*(v-square)).reshape(-1)
    if h==0:return initial.clone(),{'method':'scipy-DOP853','nfev':0,'success':True}
    solution=solve_ivp(rhs,(0,h),u.reshape(-1),method='DOP853',rtol=max(2.3e-14,tolerance*.05),atol=max(1e-15,tolerance*.01),t_eval=[h])
    if not solution.success or solution.y.shape[1]!=1:raise RuntimeError('Independent DOP853 reference failed')
    return torch.from_numpy(solution.y[:,-1].reshape(shape).copy()),{'method':'scipy-DOP853','nfev':solution.nfev,'success':True}

def temporal_reference(initial,h,equation,geometry,target,options=None,budget=None):
    options=options or {};start=time.monotonic();history=[];last=initial.clone()
    if initial.dtype!=torch.float64 or initial.device.type!='cpu':raise ValueError('Teachers require CPU FP64 actual deployed input')
    tolerance=float(options.get('tolerance',1e-10));cap=int(options.get('max_steps',options.get('max_substeps',2048)))
    steps=max(int(options.get('initial_steps',options.get('substeps',8))),math.ceil(h*float(equation.reaction_rate)),1)
    seconds=float(options.get('max_seconds_per_query',120));floor=float(options.get('roundoff_floor',2e-14))
    if not math.isfinite(h) or h<0 or not math.isfinite(tolerance) or tolerance<=0 or cap<1:raise ValueError('Invalid teacher contract')
    def check():
        if budget is not None:budget.check()
        if time.monotonic()-start>seconds:raise TimeoutError('Per-query teacher runtime cap')
    observation_grid=options.get('observation_grid')
    def observed(value):
        if observation_grid is None:return value
        from .numerics import resample
        return resample(value,tuple(observation_grid))
    accepted=False;reason='REFINEMENT_CAP';previous=None;previous_difference=None;unc={'rms':None,'maximum':None};cross=None
    try:
        while steps<=cap:
            check();last=_lawson(initial,h,equation,geometry,target,steps,check)
            finite=bool(torch.isfinite(last).all());diff=_norms(observed(last-previous)) if previous is not None else {'rms':None,'maximum':None}
            converged=previous_difference is not None and all(diff[k] is not None and (max(diff[k],previous_difference[k])<=floor or diff[k]<=.5*previous_difference[k]) for k in diff)
            history.append(dict(steps=steps,difference=diff,working_grid_difference=_norms(last-previous) if previous is not None else None,finite=finite,contracting=converged))
            if not finite:reason='NONFINITE_TEACHER';break
            if previous is not None:unc={k:max(float(diff[k]),floor) for k in diff}
            if converged and all(v<=tolerance for v in unc.values()):accepted=True;reason='TIME_REFINED';break
            if previous is not None:previous_difference=diff
            previous=last;steps*=2
        if accepted and options.get('crosscheck',True):
            independent,info=_scipy_crosscheck(initial,h,equation,geometry,target,tolerance,check)
            discrepancy=_norms(observed(independent-last));cross={**info,'difference':discrepancy,'working_grid_difference':_norms(independent-last)}
            accepted=all(discrepancy[k] is not None and discrepancy[k]<=tolerance for k in discrepancy)
            unc={k:max(unc[k],discrepancy[k]) if discrepancy[k] is not None else None for k in unc}
            reason='TIME_AND_INDEPENDENT_SOLVER_ACCEPTED' if accepted else 'CROSS_SOLVER_DISAGREEMENT'
    except (TimeoutError,RuntimeError,FloatingPointError) as exc:
        accepted=False;reason=f'{type(exc).__name__}: {exc}'
    return last,dict(accepted=accepted,status='ACCEPTED' if accepted else 'UNRESOLVED_REFERENCE',reason=reason,
        uncertainty_rms=unc['rms'],uncertainty_maximum=unc['maximum'],temporal_refinements=history,cross_solver=cross,
        cross_solver_required=bool(options.get('crosscheck',True)),elapsed_seconds=time.monotonic()-start,
        roundoff_floor=floor,uncertainty_is_certificate=False,solver='Lawson-RK4 with exact diffusion',
        uncertainty_norm='normalized sampled RMS and maximum on observation grid; RMS conservatively dominates physical interpolant L2',
        observation_grid=list(observation_grid or geometry.grid),working_grid=list(geometry.grid))

def generate_reference(u0,h,equation,geometry,target,options=None,budget=None):
    from .numerics import resample
    from tdn.numerics.types import Geometry
    if target not in TARGETS:raise ValueError('Unknown equation target')
    options=options or {};initial=u0.detach().cpu().double()
    if target!='galerkin_projected_fine':return temporal_reference(initial,h,equation,geometry,target,options,budget)
    start=time.monotonic();hist=[];previous=None;answer=initial.clone();unc={'rms':None,'maximum':None};accepted=False
    factors=list(options.get('spatial_factors',[2,4]));tol=float(options.get('tolerance',1e-10));max_grid=int(options.get('max_grid',512))
    allocation=projected_error_budget(options);attempts=[]
    if not math.isfinite(tol) or tol<=0:raise ValueError('Invalid projected reference total tolerance')
    for index,factor in enumerate(factors):
        if factor<2 or any(n*factor>max_grid for n in geometry.grid):
            attempts.extend(dict(factor=f,attempted=False,status='GRID_CAP_OR_INVALID_FACTOR',
                grid=[n*f for n in geometry.grid]) for f in factors[index:]);break
        # Lift the actual FP32 coarse field, not a fresh sampling of its parent.
        fine_grid=tuple(n*factor for n in geometry.grid);fine=resample(initial,fine_grid)
        opt=dict(options,observation_grid=list(geometry.grid),max_seconds_per_query=max(.001,float(options.get('max_seconds_per_query',120))-(time.monotonic()-start)))
        if allocation is not None:opt['tolerance']=allocation['temporal_tolerances'][index]
        ref,meta=temporal_reference(fine,h,equation,Geometry(fine_grid,geometry.lengths),'galerkin_same_grid',opt,budget)
        answer=resample(ref,geometry.grid);difference=_norms(answer-previous) if previous is not None else {'rms':None,'maximum':None}
        hist.append(dict(factor=factor,grid=list(fine_grid),temporal=meta,projected_difference=difference))
        attempts.append(dict(factor=factor,grid=list(fine_grid),attempted=True,status=meta['status'],
            temporal_tolerance=float(opt.get('tolerance',tol)),temporal_uncertainty_rms=meta['uncertainty_rms'],
            temporal_uncertainty_maximum=meta['uncertainty_maximum'],projected_difference=difference))
        if previous is not None and hist[-2]['temporal']['accepted'] and meta['accepted']:
            unc={k:None if difference[k] is None else difference[k]+meta['uncertainty_'+k]+hist[-2]['temporal']['uncertainty_'+k] for k in difference}
            accepted=all(v is not None and v<=tol for v in unc.values())
            if allocation is not None:
                temporal_budget_satisfied=all(
                    m['uncertainty_'+k] is not None and m['uncertainty_'+k]<=allocation['temporal_tolerances'][j]
                    for j,m in ((index,meta),(index-1,hist[-2]['temporal'])) for k in difference)
                spatial_budget_satisfied=all(v is not None and v<=allocation['spatial_tolerance'] for v in difference.values())
                attempts[-1].update(temporal_pair_budget_satisfied=temporal_budget_satisfied,
                    spatial_budget_satisfied=spatial_budget_satisfied)
                accepted=accepted and temporal_budget_satisfied and spatial_budget_satisfied
            attempts[-1]['combined_uncertainty']=dict(unc);attempts[-1]['combined_accepted']=accepted
            if accepted:
                attempts.extend(dict(factor=f,grid=[n*f for n in geometry.grid],attempted=False,
                    status='NOT_NEEDED_AFTER_ACCEPTED_PAIR') for f in factors[index+1:]);break
        previous=answer
    return answer,dict(accepted=accepted,status='ACCEPTED' if accepted else 'UNRESOLVED_REFERENCE',
        reason='SPACE_AND_TIME_REFINED' if accepted else 'SPATIAL_OR_TEMPORAL_REFINEMENT_UNRESOLVED',
        uncertainty_rms=unc['rms'],uncertainty_maximum=unc['maximum'],spatial_refinements=hist,
        component_error_budget=allocation,resolution_attempts=attempts,requested_spatial_factors=factors,
        total_uncertainty_tolerance=tol,combined_acceptance='projected spatial difference + both temporal estimates <= total tolerance',
        elapsed_seconds=time.monotonic()-start,uncertainty_is_certificate=False,
        initial_lifting='Fourier interpolation of actual deployed FP32 coarse input; zero additional hidden coefficients',
        continuum_claim=False)

def _write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.partial');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def _file_hash(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def source_supervision(u0,record,options,budget=None):
    """Extra teacher information for explicitly named source-supervised arms."""
    from .trajectory import anchor_path,actual_source
    if record['target_kind']=='galerkin_projected_fine':
        return {},{'accepted':False,'reason':'Projected closure forcing requires a distinct fine-state source contract','elapsed_seconds':0.}
    start=time.monotonic();eq,geom=equation_geometry(record);times=[record['horizon']*x for x in options.get('source_fractions',[.25,.5,.75])]
    values=[];history=[]
    for t in times:
        target,ref=generate_reference(u0,t,eq,geom,record['target_kind'],options,budget)
        path=anchor_path(u0.double(),t,eq,geom,record['target_kind'],name=record.get('anchor_name','df'),reaction_substeps=8,create_graph=False)
        forcing=actual_source(target,path['state'],path['derivative'],eq,geom,record['target_kind'])['source']
        values.append(forcing.detach().cpu());history.append(dict(time=t,reference=ref))
    accepted=all(x['reference']['accepted'] for x in history) and all(bool(torch.isfinite(v).all()) for v in values)
    return {'source_times':np.asarray(times,np.float64),'source_targets':torch.stack(values).numpy()},dict(accepted=accepted,
        elapsed_seconds=time.monotonic()-start,references=history,source_contract='Exact centered defect forcing against actual teacher mean; same finite equation',
        anchor=record.get('anchor_name','df'),reaction_substeps=record.get('anchor_reaction_substeps',8),
        uncertainty_scope='Accepted endpoint teachers; forcing error may amplify teacher/path errors and is not certified by endpoint acceptance',
        extra_supervision=True,endpoint_only_competitor_has_less_teacher_information=True)


def prepare_dataset(protocol,path,split,parents=None,unit=None,budget=None):
    from .numerics import anchor
    root=Path(path);root.mkdir(parents=True,exist_ok=True);unit=unit or {}
    parents=parent_specs(protocol,split) if parents is None else parents
    if unit.get('parent_ids') is not None:parents=[p for p in parents if p['parent_id'] in unit['parent_ids']]
    validate_parent_splits(parents)
    if any(p['split']!=split for p in parents):raise ValueError('Preparation cannot mix split identities')
    grids=unit.get('grids',protocol.get('grids',[protocol.get('train_grid',8)]))
    if 'grid' in unit:grids=[unit['grid']]
    targets=unit.get('targets',protocol.get('targets',list(TARGETS)))
    if 'target_kind' in unit:targets=[unit['target_kind']]
    horizons=unit.get('horizons',protocol.get(split+'_horizons',protocol.get('horizons',[.01,.06])))
    options=dict(protocol.get('reference',{}))
    if unit.get('reference_max_seconds_per_query') is not None:
        options['max_seconds_per_query']=min(options.get('max_seconds_per_query',120),unit['reference_max_seconds_per_query'])
    variants=unit.get('phase_variants',protocol.get('phase_variants',['original']))
    interventions=unit.get('intervention_variants',['original'])
    if any(v!='original' for v in interventions):
        from .interventions import validate_held_out_values
        validate_held_out_values(parent_specs(protocol,'train')+parent_specs(protocol,'validation'),interventions)
    panel=unit.get('panel','primary');descriptive_only=bool(unit.get('descriptive_only',False))
    identity=dict(schema='tdn.response-data/v1',protocol_sha256=digest(protocol),split=split,parents=parents,grids=grids,targets=targets,horizons=horizons,phase_variants=variants,intervention_variants=interventions,panel=panel,descriptive_only=descriptive_only,reference_options=options)
    manifest_path=root/'dataset.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest['identity']!=identity:raise ValueError('Prepared dataset identity changed')
        for r in manifest['records']:
            if _file_hash(root/r['array_file'])!=r['array_file_sha256']:raise ValueError('Prepared array hash mismatch')
    else:manifest=dict(identity=identity,records=[])
    done={r['query_id'] for r in manifest['records']}
    def transformed(p,variant):
        if variant=='original' and panel!='physical_interventions':return dict(p,intervention_variant='original')
        from .interventions import transform_parent
        return transform_parent(p,variant)
    descendants=[phase_variant(transformed(p,i),v) for p in parents for i in interventions for v in variants]
    for declared_parent in descendants:
      for grid in grids:
        shape=[grid]*declared_parent['dimension'] if isinstance(grid,int) else list(grid)
        parent=grid_relative_parent(declared_parent,min(shape)) if panel=='grid_relative_stress' else declared_parent
        sampled=sample_parent(parent,shape);u0=sampled.float()
        for target in targets:
          for h in horizons:
            if budget is not None:budget.check()
            record=dict(parent_id=parent['parent_id'],split=split,regime=parent['regime'],tiny_regime=parent.get('tiny_regime'),shape_regime=parent['regime'],target_kind=target,grid=shape,
                lengths=parent['lengths'],phase_variant=parent['phase_variant'],phase_scope=parent['phase_scope'],descendant_sha256=parent['descendant_sha256'],horizon=float(h),kappa=parent['kappa'],reaction_rate=parent['reaction_rate'],
                panel=panel,descriptive_only=descriptive_only,frequency_scope=parent.get('frequency_scope',parent['shape_scope']),
                intervention_variant=parent.get('intervention_variant','original'),intervention_metadata=parent.get('intervention'),
                input_modes=parent['terms'],mode_scale=parent.get('mode_scale',1),independence_scope='phase/grid/horizon descendants remain within original parent',
                parent_sha256=parent['parent_sha256'],input_sha256=array_digest(u0),sampled_fp64_sha256=array_digest(sampled),
                input_contract='actual FP32 input cast to FP64 for teacher; continuous parent samples separately stored',
                target_contract='projected fine Galerkin from zero-hidden-mode lift' if target=='galerkin_projected_fine' else 'same finite equation',
                boundary='periodic',fourier_normalization='forward/interpolant Nyquist splitting',inference_dtype='float32',teacher_dtype='float64',
                anchor_name='df',anchor_reaction_substeps=8,physical_anchor_scope='FP64 implementation of declared same 8-substep Galerkin reaction anchor; finite subflow error separately estimated')
            qid=digest(record)[:24];record['query_id']=qid
            if qid in done:continue
            eq,geom=equation_geometry(record);start=time.monotonic()
            answer,reference=generate_reference(u0,float(h),eq,geom,target,options,budget)
            record['reference']=reference
            if parent['phase_variant']=='phase_twin':
                original=next(p for p in parents if p['parent_id']==parent['parent_id'])
                original=transformed(original,parent.get('intervention_variant','original'))
                if panel=='grid_relative_stress':original=grid_relative_parent(original,min(shape))
                base64=sample_parent(original,shape);base32=base64.float()
                axes=tuple(range(2,u0.ndim))
                record['phase_matching']=dict(sampled_fp64_mean_difference=float((sampled.mean()-base64.mean()).abs()),
                    actual_fp32_mean_difference=float((u0.double().mean()-base32.double().mean()).abs()),
                    sampled_fp64_max_power_difference=float((torch.fft.fftn(sampled,dim=axes,norm='forward').abs().square()-torch.fft.fftn(base64,dim=axes,norm='forward').abs().square()).abs().max()),
                    actual_fp32_max_power_difference=float((torch.fft.fftn(u0.double(),dim=axes,norm='forward').abs().square()-torch.fft.fftn(base32.double(),dim=axes,norm='forward').abs().square()).abs().max()),
                    interpretation='Continuous modal amplitudes exactly paired; sampled/quantized power agreement measured, not assumed')
            with torch.no_grad():
                a32=anchor(u0,float(h),eq,geom,target);a64=anchor(u0.double(),float(h),eq,geom,target)
                refined_anchor=anchor(u0.double(),float(h),eq,geom,target,reaction_substeps=32)
            deployment=answer-a32.double();physical=answer-a64
            rounding=(answer.float().double()-answer);input_quantization=u0.double()-sampled
            record['precision']=dict(anchor_reaction_refinement_8_to_32=_norms(refined_anchor-a64),anchor_difference=_norms(a32.double()-a64),input_quantization=_norms(input_quantization),
                teacher_output_rounding=_norms(rounding),deployment_defect=_norms(deployment),physical_defect=_norms(physical),
                naive_fp32_subtraction_error=_norms((answer.float()-a32).double()-deployment),
                assembly_rounding_floor=_norms((a32+deployment.float()).double()-answer))
            record['precision'].update(physical_norm_components=dict(
                deployment_defect=_physical_norms(deployment,geom,target),physical_defect=_physical_norms(physical,geom,target),
                teacher_output_rounding=_physical_norms(rounding,geom,target),
                assembly_rounding_floor=_physical_norms((a32+deployment.float()).double()-answer,geom,target)),
                target_absolute_maximum=float(answer.abs().max()) if bool(torch.isfinite(answer).all()) else None,
                state_scale=max(1.,float(answer.abs().max())) if bool(torch.isfinite(answer).all()) else None,
                output_quantization_scope='Coordinatewise nearest FP32 output; exact nodal representability optimum for FD; benchmark, not global physical-L2 optimum for Galerkin')
            record['endpoint_preparation_seconds']=time.monotonic()-start;record['array_file']=f'query-{qid}.npz'
            np.savez_compressed(root/record['array_file'],u0=u0.numpy(),sampled_fp64=sampled.numpy(),target=answer.numpy(),
                anchor_fp32=a32.numpy(),anchor_fp64=a64.numpy(),deployment_label=deployment.numpy(),physical_label=physical.numpy())
            if unit.get('source_supervision',protocol.get('source_supervision',False)):
                source_arrays,source_meta=source_supervision(u0,record,options,budget)
                record['source_supervision']=source_meta
                # Repack safe numeric arrays only; physical teacher cost is explicit.
                with np.load(root/record['array_file'],allow_pickle=False) as stored:base_arrays={k:stored[k].copy() for k in stored.files}
                np.savez_compressed(root/record['array_file'],**base_arrays,**source_arrays)
            record['elapsed_seconds']=time.monotonic()-start
            record['array_file_sha256']=_file_hash(root/record['array_file']);record['target_sha256']=array_digest(answer)
            manifest['records'].append(record);_write(manifest_path,manifest)
    manifest['requested_queries']=len(parents)*len(interventions)*len(variants)*len(grids)*len(targets)*len(horizons)
    manifest['accepted_references']=sum(r['reference']['accepted'] for r in manifest['records'])
    manifest['unresolved_references']=len(manifest['records'])-manifest['accepted_references'];_write(manifest_path,manifest)
    return manifest

def load_samples(path,device='cpu'):
    root=Path(path)
    if not (root/'dataset.json').exists():
        found=sorted(root.glob('*/dataset.json'))
        if not found:raise FileNotFoundError(f'No Response dataset at {root}')
        return [s for f in found for s in load_samples(f.parent,device)]
    manifest=json.loads((root/'dataset.json').read_text());samples=[]
    if manifest['identity']['schema']!='tdn.response-data/v1':raise ValueError('Unknown dataset schema')
    for r in manifest['records']:
        name=Path(r['array_file'])
        if name.is_absolute() or '..' in name.parts or (root/name).is_symlink():raise ValueError('Unsafe dataset array path')
        if _file_hash(root/name)!=r['array_file_sha256']:raise ValueError('Dataset array bytes changed')
        with np.load(root/name,allow_pickle=False) as arrays:
            sample={k:torch.from_numpy(arrays[k].copy()).to(device) for k in arrays.files}
        if sample['u0'].dtype!=torch.float32 or sample['target'].dtype!=torch.float64 or sample['anchor_fp32'].dtype!=torch.float32:raise ValueError('Dataset precision contract changed')
        if array_digest(sample['u0'])!=r['input_sha256'] or array_digest(sample['target'])!=r['target_sha256']:raise ValueError('Dataset tensor hash changed')
        sample.update(record=r,reference_accepted=bool(r['reference']['accepted']));samples.append(sample)
    return samples

def run_prepare(ctx):
    unit=ctx.unit;split=unit.get('split','train');manifest=prepare_dataset(ctx.protocol,ctx.path,split,unit=unit,budget=ctx.budget)
    metrics=dict(requested_queries=manifest['requested_queries'],accepted_references=manifest['accepted_references'],
        unresolved_references=manifest['unresolved_references'],parents=len(manifest['identity']['parents']))
    _write(Path(ctx.path)/'reference_summary.json',metrics)
    _write(Path(ctx.path)/'reference_history.json',{'rows':[{k:r[k] for k in ('parent_id','query_id','target_kind','grid','horizon','reference','precision','elapsed_seconds')} for r in manifest['records']]})
    if hasattr(ctx,'record'):
        from .core import check
        ctx.record('reference/preparation',unit.get('mechanism_ids',['M2']),metrics=metrics,
            checks=[check('reference_coverage',manifest['unresolved_references'],0,category='math')],config=unit,evidence=['dataset.json'])
    return metrics
