"""Focused numerical attribution and complete-call cost controls.

The panel uses frozen validation parents, never accepted-reference replacement.
Expensive interaction/path controls are single-step diagnostics; inexpensive
methods receive equivalent one/two/four/eight-step opportunities. No two-parent
panel can establish a population advantage or a neural-method contribution.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import numpy as np
import torch
from tdn.runtime.metadata import write_json
from . import numerics as n, kernels as k, trajectory as tr
from .core import check, clean
from .references import load_samples, equation_geometry
from .metrics import endpoint_metrics


def control_specs():
    result=[]
    for method in ('df','rf','etdrk4','shared_gl4','lawson4'):
        for steps in (1,2,4,8):
            result.append(dict(id=f'{method}/steps-{steps}',method=method,steps=steps,
                reaction_substeps=8,precision='float32',ownership='Analytic control',
                work_scope='Complete advancing steps; no cached reference or hindsight schedule'))
    for method in ('sequential_gl4','gl8','gl16','nested8','nested12','leading_cubic','filtered_leading_cubic','linear_path16','linear_path32'):
        result.append(dict(id=f'{method}/steps-1',method=method,steps=1,reaction_substeps=8,
            precision='float32',ownership='Analytic control',
            work_scope='Complete single step including anchor, products and every quadrature/path derivative'))
    return result


def apply_control(u,h,eq,geom,target,spec):
    method=spec['method'];steps=spec['steps'];dt=h/steps;state=u
    for _ in range(steps):
        if method=='lawson4':state=n.lawson_step(state,dt,eq,geom,target)
        elif method in ('nested8','nested12'):
            state=n.anchor(state,dt,eq,geom,target,'quadratic_cubic',reaction_substeps=spec['reaction_substeps'],cubic_nodes=int(method[6:]))
        elif method in ('leading_cubic','filtered_leading_cubic'):
            state=n.anchor(state,dt,eq,geom,target,'shared_gl4',reaction_substeps=spec['reaction_substeps'])+k.leading_cubic_defect(
                state,dt,eq,geom,target,filtered=method=='filtered_leading_cubic')
        elif method in ('linear_path16','linear_path32'):
            # JVP needs a differentiable time, but no parameter/input training
            # graph is retained. All other methods use the same no_grad context.
            base=n.anchor(state,dt,eq,geom,target,'df',reaction_substeps=spec['reaction_substeps'])
            state=base+tr.homogeneous_residual_response(state,dt,eq,geom,target,name='df',nodes=int(method[11:]),
                reaction_substeps=spec['reaction_substeps'],create_graph=False)
        else:state=n.anchor(state,dt,eq,geom,target,method,reaction_substeps=spec['reaction_substeps'])
    return state


def _load_declared_samples(ctx):
    """Match immutable requested parents/grid/horizons; no truth-based choice."""
    unit=ctx.unit;wanted=unit.get('parent_ids',[])
    if not 1<=len(wanted)<=2 or len(set(wanted))!=len(wanted):
        raise ValueError('Numerical controls require one or two frozen distinct parents')
    grid=unit.get('grid',ctx.protocol.get('train_grid'))
    grid=[grid]*ctx.protocol.get('dimension',2) if isinstance(grid,int) else list(grid)
    horizons=unit.get('horizons',[ctx.protocol['validation_horizons'][0]])
    if len(horizons)!=1:raise ValueError('Focused numerical panel requires one frozen horizon per unit')
    expected=[(parent,tuple(grid),float(horizons[0]),unit['target_kind']) for parent in wanted]
    found={}
    for name in unit.get('data_stages',list(ctx.prerequisites)):
        if name not in ctx.prerequisites:raise ValueError(f'Missing numerical-control reference prerequisite {name}')
        for sample in load_samples(ctx.prerequisites[name],ctx.device):
            r=sample['record']
            if r.get('phase_variant','original')!='original':continue
            key=(r['parent_id'],tuple(r['grid']),float(r['horizon']),r['target_kind'])
            if key not in expected:continue
            if r['split']!='validation':raise ValueError('Numerical design panel must use declared validation data')
            if key in found:raise ValueError('Duplicated numerical-control query in prerequisite datasets')
            found[key]=sample
    return expected,found


def _norm(x,geom,target):
    if x is None or not bool(torch.isfinite(x).all()):return dict(rms=None,maximum=None)
    return dict(rms=float(n.inner(x.double(),x.double(),geom,target).mean().sqrt()),maximum=float(x.double().abs().max()))


def run(ctx):
    started=time.perf_counter();specs=control_specs();rows=[];timing=[];comparisons=[];arrays={};missing=[]
    expected,samples=_load_declared_samples(ctx)
    repeats=ctx.protocol.get('cost',{}).get('repeats',2 if ctx.protocol['profile']=='smoke' else 20)
    warmup=ctx.protocol.get('cost',{}).get('warmup',1 if ctx.protocol['profile']=='smoke' else 3)
    if type(repeats) is not int or repeats<1 or type(warmup) is not int or warmup<0:
        raise ValueError('Positive timing rounds and nonnegative warmup required')
    gpu=torch.device(ctx.device).type=='cuda'
    def sync():
        if gpu:torch.cuda.synchronize(ctx.device)
    def save(complete=False):
        write_json(ctx.path/'numerical_controls.json',clean(dict(schema='tdn.response-numerical-controls/v1',
            complete=complete,requested_queries=[dict(parent_id=p,grid=list(g),horizon=h,target_kind=t) for p,g,h,t in expected],
            methods=specs,rows=rows,comparisons=comparisons,missing_queries=missing,timing_samples=timing,
            provenance=dict(device=ctx.device,source_tree_bound_by_engine=True,
                inference_dtype='float32',attribution_dtype='float64',reference_input='actual stored FP32 input cast to FP64',
                fp64_values_outside_timing=True,physical_output_arrays='numerical_predictions.npz',
                norm='target-specific physical RMS plus sampled nodal maximum',
                scope='Frozen designed validation panel; no training, model selection or population claim'))))
    for query_index,key in enumerate(expected):
        ctx.budget.check()
        if key not in samples:
            missing.append(dict(parent_id=key[0],grid=list(key[1]),horizon=key[2],target_kind=key[3],status='NOT_MEASURED',
                                reason='Declared reference query absent; no easier parent substituted'))
            ctx.record(f'numerical-controls/missing/{query_index}',['M1'],metrics=missing[-1],
                checks=[check('declared-query-available',None,True,'eq',category='math',reason=missing[-1]['reason'])])
            save();continue
        sample=samples[key];r=sample['record'];eq,geom=equation_geometry(r);u=sample['u0'];h=r['horizon'];target=r['target_kind']
        if u.dtype!=torch.float32:raise ValueError('Timing requires the actual deployed FP32 input')
        predictions={};row_by_id={}
        for spec in specs:
            ctx.budget.check();outputs={};errors={};first={}
            for precision in ('float32','float64'):
                initial=u if precision=='float32' else u.double()
                sync();before=time.perf_counter()
                try:
                    with torch.no_grad():out=apply_control(initial,h,eq,geom,target,spec)
                    sync()
                    if not bool(torch.isfinite(out).all()):raise FloatingPointError('Nonfinite numerical prediction')
                    outputs[precision]=out.detach().cpu()
                except (RuntimeError,ValueError,FloatingPointError) as exc:
                    sync();outputs[precision]=None;errors[precision]=f'{type(exc).__name__}: {exc}'
                first[precision]=time.perf_counter()-before
            prediction=outputs['float32'];high=outputs['float64']
            teacher=sample['target'].detach().cpu();base=sample['anchor_fp32'].detach().cpu();base64=sample['anchor_fp64'].detach().cpu()
            metrics=endpoint_metrics(prediction,teacher,anchor=base,u0=u.detach().cpu(),record=r,
                                     tolerance=ctx.protocol.get('evaluation',{}).get('rms_target',2e-5))
            physical=endpoint_metrics(high,teacher,anchor=base64,u0=u.detach().cpu(),record=r,
                                     tolerance=ctx.protocol.get('evaluation',{}).get('rms_target',2e-5))
            array_id=f'query{query_index}_{spec["id"].replace("/","_").replace("-","_")}'
            for precision,out in outputs.items():
                if out is not None:arrays[array_id+'_'+precision]=out.numpy()
            roundoff=_norm(None if high is None or prediction is None else prediction.double()-high,geom,target)
            row=dict(query_index=query_index,query_id=r['query_id'],parent_id=r['parent_id'],method=spec['id'],
                deployed=metrics,physical=physical,rounding_difference=roundoff,failures=errors,
                first_invocation_seconds=first,first_invocation_scope='First call in this warm process, not fresh startup',
                physical_arrays_prefix=array_id,cost=None,reference_accepted=sample['reference_accepted'],
                reaction_substeps=spec['reaction_substeps'],numerical_order_claim='No uniform/stiff-order claim')
            rows.append(row);row_by_id[spec['id']]=row;predictions[spec['id']]=outputs
            save()
        # Equivalent complete prediction calls, random paired order and no_grad.
        by_id={spec['id']:spec for spec in specs};eligible=[spec['id'] for spec in specs if predictions[spec['id']]['float32'] is not None]
        for _ in range(warmup):
            for name in eligible:
                ctx.budget.check()
                with torch.no_grad():apply_control(u,h,eq,geom,target,by_id[name])
                sync()
        for round_index in range(repeats):
            seed=int(hashlib.sha256(f'{r["query_id"]}:numerical-control:{round_index}'.encode()).hexdigest()[:16],16)
            order=list(np.random.default_rng(seed).permutation(eligible))
            for position,name in enumerate(order):
                ctx.budget.check();sync()
                if gpu:torch.cuda.reset_peak_memory_stats(ctx.device)
                before=time.perf_counter();failure=None
                try:
                    with torch.no_grad():out=apply_control(u,h,eq,geom,target,by_id[name])
                    sync();seconds=time.perf_counter()-before
                    parity=_norm(out.detach().cpu().double()-predictions[name]['float32'].double(),geom,target)
                except (RuntimeError,ValueError,FloatingPointError) as exc:
                    sync();seconds=None;parity=dict(rms=None,maximum=None);failure=f'{type(exc).__name__}: {exc}'
                timing.append(dict(query_id=r['query_id'],method=name,round=round_index,order=position,order_seed=seed,
                    seconds=seconds,failure=failure,parity_with_first_call=parity,
                    allocated_peak_bytes=torch.cuda.max_memory_allocated(ctx.device) if gpu else None,
                    reserved_peak_bytes=torch.cuda.max_memory_reserved(ctx.device) if gpu else None,
                    device=ctx.device,synchronization='torch.cuda.synchronize before/after' if gpu else 'CPU synchronous perf_counter',
                    context='no_grad for all; path control explicitly re-enables JVP internally',
                    scope='Complete resident-input call; no transfers, no compilation; all reference/path work charged'))
            save()
        for name,row in row_by_id.items():
            observed=[x for x in timing if x['query_id']==r['query_id'] and x['method']==name]
            times=[x['seconds'] for x in observed if x['seconds'] is not None]
            row['cost']=dict(median_seconds=statistics.median(times) if times else None,
                min_seconds=min(times) if times else None,max_seconds=max(times) if times else None,
                repeats_requested=repeats,repeats_completed=len(times),warmup=warmup,failures=sum(x['failure'] is not None for x in observed),
                allocated_peak_bytes=max((x['allocated_peak_bytes'] for x in observed if x['allocated_peak_bytes'] is not None),default=None),
                reserved_peak_bytes=max((x['reserved_peak_bytes'] for x in observed if x['reserved_peak_bytes'] is not None),default=None),
                unavailable_reason=None if times else 'No valid complete-call timing',
                tails='No p99 from this small repeated panel',memory_scope='Per-call process CUDA allocator peak, constant reference inputs resident')
            ctx.record(f'numerical-controls/{query_index}/{name}',['M1','M5'],metrics=dict(
                rms=row['deployed']['rms'],maximum=row['deployed']['maximum'],physical_rms=row['physical']['rms'],
                reference_accepted=row['reference_accepted'],cost=row['cost'],failures=row['failures']),
                config=by_id[name]|{'query_id':r['query_id'],'parent_id':r['parent_id']},
                checks=[check('finite-deployed-prediction',row['deployed']['prediction_finite'],True,'eq',category='correctness'),
                    check('reference-accepted',row['reference_accepted'],True,'eq',category='math',
                          applicable=bool(row['reference_accepted']),reason='Unresolved teachers cannot establish an accuracy gain')],
                evidence=['numerical_controls.json','numerical_predictions.npz'])
        pairs=[('sequential_gl4/steps-1','shared_gl4/steps-1','same_rule_parity'),
            ('gl8/steps-1','gl16/steps-1','quadrature_refinement'),
            ('nested8/steps-1','nested12/steps-1','nested_quadrature_refinement'),
            ('linear_path16/steps-1','linear_path32/steps-1','full_path_quadrature_refinement')]
        for left,right,kind in pairs:
            a=predictions[left]['float64'];b=predictions[right]['float64']
            disagreement=_norm(None if a is None or b is None else a-b,geom,target)
            comparison=dict(query_id=r['query_id'],left=left,right=right,kind=kind,fp64_disagreement=disagreement,
                reference_accepted=sample['reference_accepted'],reference_uncertainty=r['reference'],
                meaning='Observed refinement difference is an estimate, not a certificate or full nonlinear error bound')
            comparisons.append(comparison)
            if kind=='same_rule_parity':
                ctx.record(f'numerical-controls/{query_index}/shared-parity',['M1'],metrics=comparison,
                    checks=[check('same-rule-fp64-parity',disagreement['maximum'],2e-11,category='correctness')])
        np.savez_compressed(ctx.path/'numerical_predictions.npz',**arrays)
        save()
    save(complete=True)
    auxiliary = run_auxiliary_panels(ctx,samples) if ctx.unit.get('auxiliary_panels',bool(ctx.protocol.get('parents'))) else dict(status='NOT_REQUESTED')
    summary=dict(auxiliary=auxiliary,requested_queries=len(expected),attempted_queries=len(samples),missing_queries=len(missing),
        methods_per_query=len(specs),rows=len(rows),timing_samples=len(timing),
        accepted_reference_queries=sum(s['reference_accepted'] for s in samples.values()),
        elapsed_seconds=time.perf_counter()-started,
        scope='Focused designed validation numerical controls; no universal ranking or independent-field significance')
    write_json(ctx.path/'numerical_control_summary.json',summary)
    return summary


def error_decomposition(prediction, u_discrete, u_galerkin, u_projected, geom, *, norm_target='galerkin_same_grid'):
    """Signed vector identity, with every cross term in a single declared norm.

    p-Pu_f = (p-u_D)+(u_D-u_G)+(u_G-Pu_f). The middle term
    includes both FD-versus-spectral diffusion and nodal-versus-Galerkin
    products. Its norm is not a separately additive error budget.
    """
    fields=(prediction,u_discrete,u_galerkin,u_projected)
    if len({tuple(x.shape) for x in fields})!=1 or any(not bool(torch.isfinite(x).all()) for x in fields):
        raise ValueError('Error decomposition requires matching finite fields')
    fields=tuple(x.double() for x in fields);p,d,g,f=fields
    terms=dict(discrete_solver_error=p-d,spatial_discretization=d-g,unresolved_scale_closure=g-f)
    total=p-f;assembled=sum(terms.values());names=list(terms)
    energies={name:float(n.inner(value,value,geom,norm_target).mean()) for name,value in terms.items()}
    cross=[]
    for i,left in enumerate(names):
        for right in names[i+1:]:
            ip=float(n.inner(terms[left],terms[right],geom,norm_target).mean())
            denominator=math.sqrt(max(0.,energies[left]*energies[right]))
            cross.append(dict(left=left,right=right,inner_product=ip,signed_twice_inner_product=2*ip,
                cosine=ip/denominator if denominator>0 else None))
    actual=float(n.inner(total,total,geom,norm_target).mean())
    square_sum=sum(energies.values());cross_sum=sum(x['signed_twice_inner_product'] for x in cross)
    return dict(terms=terms,total=total,metrics=dict(norm_target=norm_target,component_squared_norms=energies,
        component_norms={key:math.sqrt(max(0.,value)) for key,value in energies.items()},cross_terms=cross,
        total_squared_norm=actual,sum_component_squared_norms=square_sum,sum_signed_cross_terms=cross_sum,
        reconstructed_squared_norm=square_sum+cross_sum,squared_identity_discrepancy=abs(actual-square_sum-cross_sum),
        vector_reconstruction_maximum=float((total-assembled).abs().max()),
        interpretation='Signed vector decomposition; component norms must not be added or interpreted as independent errors'))


def run_auxiliary_panels(ctx,samples):
    """Bounded CPU counterreferences outside all prediction latency timers.

    Reuse the first declared original parent. Its broadband/grid-relative
    interventions are paired designed workloads, not new independent parents.
    Only the FD unit computes the three-target decomposition to avoid charging
    the identical counterteachers again in each target-specific unit.
    """
    from .fields import parent_specs,spectral_stress_parent,sample_parent,array_digest,digest
    from .references import generate_reference
    if not samples:return dict(status='NOT_MEASURED',reason='No declared original query available')
    declared_first = ctx.unit.get('parent_ids', [None])[0]
    eligible = [sample for sample in samples.values() if declared_first is None or sample['record']['parent_id'] == declared_first]
    if not eligible:
        return dict(status='NOT_MEASURED',reason='First frozen auxiliary parent is absent; no replacement chosen')
    first=eligible[0];record=first['record'];parent_id=record['parent_id']
    parents={p['parent_id']:p for p in parent_specs(ctx.protocol,'validation')}
    if parent_id not in parents:raise ValueError('Auxiliary stress parent is absent from frozen validation declarations')
    original=parents[parent_id];grid=list(record['grid']);N=min(grid)
    if N<8:return dict(status='NOT_MEASURED',reason='Declared spectral stress requires grid >=8')
    options=dict(ctx.protocol.get('reference',{}));options['max_seconds_per_query']=min(120.,float(options.get('max_seconds_per_query',120.)))
    options['max_grid']=min(128,int(options.get('max_grid',128)))
    options['max_steps']=min(2048,int(options.get('max_steps',2048)))
    options['crosscheck']=True
    options['spatial_factors']=[2,4]
    stress_rows=[];stress_metadata=[];vectors={};decomposition_rows=[];references={}
    # Countertargets share precisely the saved FP32 observation; no re-sampling.
    if ctx.unit['target_kind']=='fd_same_grid':
        initial=first['u0'].detach().cpu();eq,geom=equation_geometry(record);h=record['horizon'];teacher_seconds={}
        for target in ('fd_same_grid','galerkin_same_grid','galerkin_projected_fine'):
            ctx.budget.check();start=time.perf_counter()
            if target=='fd_same_grid':
                value=first['target'].detach().cpu();meta=record['reference'];origin='reused sealed prerequisite teacher'
            else:
                value,meta=generate_reference(initial,h,eq,geom,target,options,ctx.budget);origin='new CPU counterreference outside latency'
            references[target]=dict(meta=meta,origin=origin,input_sha256=array_digest(initial))
            teacher_seconds[target]=time.perf_counter()-start
            vectors['reference_'+target]=value.numpy()
        values={key:torch.from_numpy(vectors['reference_'+key]) for key in references}
        valid=all(v['meta']['accepted'] for v in references.values())
        for method in ('df','shared_gl4'):
            prediction=n.anchor(initial.double(),h,eq,geom,'fd_same_grid',method)
            if all(bool(torch.isfinite(x).all()) for x in (prediction,*values.values())):
                for norm_target in ('fd_same_grid','galerkin_same_grid'):
                    d=error_decomposition(prediction,values['fd_same_grid'],values['galerkin_same_grid'],values['galerkin_projected_fine'],geom,norm_target=norm_target)
                    uncertainties = {name: value['meta'].get('uncertainty_rms') for name,value in references.items()}
                    eps_d, eps_g, eps_f = (uncertainties[name] for name in ('fd_same_grid','galerkin_same_grid','galerkin_projected_fine'))
                    component_unc = dict(discrete_solver_error=eps_d,
                        spatial_discretization=None if eps_d is None or eps_g is None else eps_d+eps_g,
                        unresolved_scale_closure=None if eps_g is None or eps_f is None else eps_g+eps_f)
                    sensitivity = {}
                    for component, uncertainty in component_unc.items():
                        magnitude = d['metrics']['component_norms'][component]
                        sensitivity[component] = dict(estimated_uncertainty=uncertainty,
                            lower=None if uncertainty is None else max(0.,magnitude-uncertainty),
                            upper=None if uncertainty is None else magnitude+uncertainty,
                            exceeds_ten_times_estimated_uncertainty=None if uncertainty is None else magnitude>10*uncertainty)
                    for cross in d['metrics']['cross_terms']:
                        left,right = cross['left'],cross['right']; ea,eb=component_unc[left],component_unc[right]
                        cross['estimated_twice_inner_product_sensitivity'] = None if ea is None or eb is None else 2*(
                            d['metrics']['component_norms'][left]*eb+d['metrics']['component_norms'][right]*ea+ea*eb)
                    row=dict(method=method,norm_target=norm_target,reference_accepted=valid,
                        status='MEASURED' if valid else 'UNRESOLVED_REFERENCE',metrics=d['metrics'],component_uncertainty_sensitivity=sensitivity,
                        reference_uncertainty={key:v['meta'] for key,v in references.items()},
                        primary_total_uncertainty=references['galerkin_projected_fine']['meta'],
                        uncertainty_scope='Shared uncertain reference vectors, not independent random errors; total target uses only projected-fine uncertainty')
                    decomposition_rows.append(row)
                    for name,value in d['terms'].items():vectors[method+'_'+name]=value.numpy()
                    vectors[method+'_prediction']=prediction.numpy()
                    ctx.record(f'decomposition/{method}/{norm_target}',['M1','M2'],metrics=d['metrics']|{'references_accepted':valid},
                        checks=[check('vector-reconstruction',d['metrics']['vector_reconstruction_maximum'],1e-12,category='correctness'),
                            check('signed-squared-norm-identity',d['metrics']['squared_identity_discrepancy'],1e-12,category='correctness'),
                            check('all-targets-resolved',valid,True,'eq',category='math',applicable=valid,
                                reason='Unaccepted counterteachers leave physical attribution unresolved')],evidence=['decomposition.json','numerical_diagnostic_vectors.npz'])
            else:
                decomposition_rows.append(dict(method=method,status='UNRESOLVED_REFERENCE',reason='Nonfinite teacher/prediction retained; no decomposition inferred'))
        write_json(ctx.path/'decomposition.json',clean(dict(schema='tdn.response-error-decomposition/v1',rows=decomposition_rows,
            parent_id=parent_id,input_sha256=array_digest(initial),grid=grid,horizon=h,kappa=record['kappa'],reaction_rate=record['reaction_rate'],
            input_contract='Same actual saved FP32 observation cast to FP64 for all finite equations and known zero-hidden fine lift',
            teachers=references,reference_seconds=teacher_seconds,reference_options=options,
            physical_scope='FD/nodal versus same-grid spectral/Galerkin versus projected refined Galerkin; includes both product and diffusion discretization',
            prediction_precision='float64',inference_cost_includes_reference_generation=False,
            offline_counterreference_cost_is_charged=True)))
    if vectors:
        np.savez_compressed(ctx.path/'numerical_diagnostic_vectors.npz',**vectors)
    for kind in ('broadband','grid_relative'):
        ctx.budget.check();parent=spectral_stress_parent(original,N,kind);initial=sample_parent(parent,grid).float()
        r=dict(record,parent_id=parent['parent_id'],cluster_parent_id=parent_id,regime=kind,shape_regime=kind,
            input_sha256=array_digest(initial),phase_variant='original',parent_sha256=parent['parent_sha256'])
        r['query_id']=digest([r['parent_id'],r['target_kind'],grid,r['horizon'],r['input_sha256']])[:24]
        eq,geom=equation_geometry(r);start=time.perf_counter()
        truth,meta=generate_reference(initial,r['horizon'],eq,geom,r['target_kind'],options,ctx.budget)
        teacher_seconds=time.perf_counter()-start;r['reference']=meta
        stress_metadata.append(dict(parent=parent,record=r,teacher_seconds=teacher_seconds,teacher_options=options,
            independent_parent_id=parent_id,scope='Paired field intervention; not an additional independent observation'))
        vectors[kind+'_initial_fp32']=initial.numpy();vectors[kind+'_teacher_fp64']=truth.numpy()
        for spec in control_specs():
            if spec['method'] not in ('df','shared_gl4','etdrk4'):continue
            ctx.budget.check();start=time.perf_counter();failure=None
            try:
                with torch.no_grad():prediction=apply_control(initial,r['horizon'],eq,geom,r['target_kind'],spec)
            except (RuntimeError,ValueError,FloatingPointError) as exc:prediction=None;failure=f'{type(exc).__name__}: {exc}'
            seconds=time.perf_counter()-start
            metrics=endpoint_metrics(prediction,truth,u0=initial,record=r,tolerance=ctx.protocol.get('evaluation',{}).get('rms_target',2e-5))
            row=dict(kind=kind,method=spec['id'],cluster_parent_id=parent_id,metrics=metrics,failure=failure,
                cpu_execution_seconds=seconds,timing_scope='Single CPU diagnostic call, not paired GPU deployment timing',teacher_seconds=teacher_seconds)
            stress_rows.append(row)
            if prediction is not None:vectors[kind+'_'+spec['id'].replace('/','_')]=prediction.numpy()
            ctx.record(f'spectral-stress/{kind}/{spec["id"]}',['M1'],metrics=metrics|{'cpu_diagnostic_seconds':seconds},
                config=spec|{'cluster_parent_id':parent_id,'frequency_scope':parent['frequency_scope']},
                checks=[check('finite-prediction',metrics['prediction_finite'],True,'eq',category='correctness'),
                    check('reference-resolved',meta['accepted'],True,'eq',category='math',applicable=meta['accepted'],
                          reason='Unaccepted stress teachers cannot establish accuracy benefit')],evidence=['spectral_stress.json'])
            np.savez_compressed(ctx.path/'numerical_diagnostic_vectors.npz',**vectors)
            write_json(ctx.path/'spectral_stress.json',clean(dict(schema='tdn.response-spectral-stress/v1',rows=stress_rows,fields=stress_metadata,
                scope='Two paired spectral interventions, not neural evaluation or resolution transfer',complete=False)))
    np.savez_compressed(ctx.path/'numerical_diagnostic_vectors.npz',**vectors)
    write_json(ctx.path/'spectral_stress.json',clean(dict(schema='tdn.response-spectral-stress/v1',rows=stress_rows,fields=stress_metadata,
        scope='Two paired spectral interventions, not neural evaluation or resolution transfer',complete=True)))
    return dict(status='COMPLETED',stress_rows=len(stress_rows),stress_fields=2,independent_original_parents=1,
        decomposition_rows=len(decomposition_rows),counterreference_targets=len(references),
        offline_reference_seconds=sum(f['teacher_seconds'] for f in stress_metadata)+sum(
            v['meta'].get('elapsed_seconds',0) for v in references.values() if v['origin'].startswith('new')))
