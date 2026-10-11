"""Frozen checkpoint evaluation; predictors never receive target or teacher handles."""
from __future__ import annotations
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from .references import load_samples,equation_geometry,generate_reference,_write,_file_hash
from .fields import digest,array_digest
from .metrics import endpoint_metrics,summarize_endpoints

PUBLIC_INPUT_KEYS=('parent_id','query_id','target_kind','grid','lengths','horizon','kappa','reaction_rate','input_sha256')

def inference_input(sample):
    """A deliberately narrow inference view: no teacher, labels or future data."""
    return {'u0':sample['u0'],'record':{k:sample['record'][k] for k in PUBLIC_INPUT_KEYS if k in sample['record']}}


def load_stage_samples(ctx,validation_only=False):
    names=ctx.unit.get('data_stages',list(ctx.prerequisites));samples=[];seen=set()
    for name in names:
        path=ctx.prerequisites.get(name)
        if path is None or not (Path(path)/'dataset.json').exists():continue
        for s in load_samples(path,ctx.device):
            r=s['record']
            if r['query_id'] in seen or r['target_kind']!=ctx.unit.get('target_kind',r['target_kind']):continue
            if validation_only and r['split']!='validation':continue
            if not validation_only and r['split'] not in ('development_test','development','confirmation','test'):continue
            seen.add(r['query_id']);samples.append(s)
    return samples


def _selection(ctx):
    stage=ctx.unit.get('selection_stage');roots=[Path(ctx.prerequisites[stage])] if stage in ctx.prerequisites else []
    roots += [Path(p) for p in ctx.prerequisites.values() if Path(p) not in roots]
    frozen=ctx.protocol.get('freeze') or {}
    if ctx.protocol.get('profile')=='confirmation':
        path=Path(frozen['path'])
        if _file_hash(path)!=frozen['sha256']:raise ValueError('Frozen external model manifest changed')
        content=json.loads(path.read_text())
        return {'selected_models':content.get('models',content.get('selected_models',[]))},path
    for root in roots:
        for name in ('selection.json','selected_models.json','freeze.json'):
            p=root/name
            if p.exists():
                content=json.loads(p.read_text())
                if 'selected_models' in content:return content,p
    return {'selected_models':[]},None


def load_selected_methods(ctx,target,selection_filter=None):
    from .training import load_model,predict
    selection,source=_selection(ctx);methods={}
    for item in selection.get('selected_models',[]):
        if item.get('target_kind')!=target:continue
        declared_method=item.get('method_id',f"{item['family']}::{item.get('variant_id','base')}::n{item.get('data_count',ctx.protocol.get('parent_counts',{}).get('train',0))}")
        if selection_filter is not None and (declared_method,item.get('seed',0)) not in selection_filter:continue
        path=item.get('trial_directory',item.get('path',item.get('checkpoint_dir')))
        if path is None and item.get('stage') in ctx.prerequisites:path=ctx.prerequisites[item['stage']]
        if path is None and item.get('stage_id') in ctx.prerequisites:path=ctx.prerequisites[item['stage_id']]
        if path is None:raise ValueError('Selected model has no explicit checkpoint location')
        path=Path(path)
        if not path.is_absolute():path=source.parent/path
        checkpoint=path/'selected.pt' if path.is_dir() else path
        if item.get('checkpoint_sha256') and _file_hash(checkpoint)!=item['checkpoint_sha256']:raise ValueError('Selected checkpoint bytes changed')
        model,metadata=load_model(path,ctx.device);model.eval()
        family=item.get('family',metadata.get('family','unknown'));seed=item.get('seed',metadata.get('seed',0))
        method_id=item.get('method_id',f"{family}::{item.get('variant_id','base')}::n{item.get('data_count',ctx.protocol.get('parent_counts',{}).get('train',0))}")
        name=f'{method_id}/seed-{seed}'
        if name in methods:raise ValueError('Duplicate selected family/seed')
        def call(sample,model=model):return predict(model,inference_input(sample))
        call.model=model
        call.metadata=dict(family=family,method_id=method_id,variant_id=item.get('variant_id','base'),data_count=item.get('data_count'),seed=seed,checkpoint_path=str(path),selection_sha256=_file_hash(source),
            information_contract='current input, requested horizon, known coefficients/geometry; no reference access')
        methods[name]=call
    return methods


def classical_methods(record,schedules=(1,2,4,8)):
    from .numerics import anchor,lawson_step
    methods={}
    for family in ('df','shared_gl4','quadratic_cubic','etdrk4','lawson4'):
      for count in schedules:
        if type(count) is not int or count<1:raise ValueError('Positive schedule count required')
        def call(sample,family=family,count=count):
            r=sample['record'];eq,geom=equation_geometry(r);u=sample['u0'];h=r['horizon']/count
            for _ in range(count):
                u=lawson_step(u,h,eq,geom,r['target_kind']) if family=='lawson4' else anchor(u,h,eq,geom,r['target_kind'],name=family)
            return u
        call.metadata=dict(family='analytic_'+family,method_id='analytic_'+family,seed=None,steps=count,learned=False,
            information_contract='same current input and known coefficients; no reference access',reaction_substeps=8,cubic_nodes=8 if family=='quadratic_cubic' else None)
        methods[f'analytic_{family}/steps-{count}']=call
    return methods


def _execute(method,inp,count):
    u=inp['u0'];record=dict(inp['record'],horizon=inp['record']['horizon']/count)
    for _ in range(count):u=method({'u0':u,'record':record})
    return u


def _result(method,inp,count):
    start=time.perf_counter();reason=None
    try:
        with torch.no_grad():p=_execute(method,inp,count)
        if p.device.type=='cuda':torch.cuda.synchronize(p.device)
    except (RuntimeError,ValueError,FloatingPointError) as exc:p=None;reason=f'{type(exc).__name__}: {exc}'
    return p,time.perf_counter()-start,reason


def _role(record,count=1):
    return f"{record['target_kind']}/grid-{'x'.join(map(str,record['grid']))}/h-{record['horizon']:.12g}/steps-{count}" + (f"/phase-{record['phase_variant']}" if record.get('phase_variant','original')!='original' else '') + (f"/intervention-{record['intervention_variant']}" if record.get('intervention_variant','original')!='original' else '')


def run_evaluate(ctx):
    samples=load_stage_samples(ctx)
    if not samples:raise ValueError('Evaluation requires declared held-out parent data')
    if any(s['record']['split'] in ('train','validation') for s in samples):raise ValueError('Evaluation cannot select or train on labels')
    target=ctx.unit['target_kind'];methods=load_selected_methods(ctx,target)
    neural_count=len(methods);methods.update(classical_methods(samples[0]['record'],[1]))
    schedules=ctx.protocol.get('evaluation',{}).get('frozen_schedules',[1,2,4,8]);rows=[];spatial_examples=[]
    seeds=ctx.protocol.get('seeds',[0]);selection,selection_path=_selection(ctx)
    declared_refs=[ctx.protocol['units'][name] for name in ctx.unit.get('data_stages',[]) if name in ctx.protocol.get('units',{})]
    intended_parents=sorted({p for unit in declared_refs for p in unit.get('parent_ids',[])}) or sorted({s['record']['parent_id'] for s in samples})
    roles=set()
    for unit in declared_refs:
        grid=unit.get('grid',samples[0]['record']['grid']);grid=[grid]*ctx.protocol.get('dimension',2) if isinstance(grid,int) else list(grid)
        for horizon in unit.get('horizons',[]):
            for count in schedules:
                for variant in unit.get('phase_variants',ctx.protocol.get('phase_variants',['original'])):
                    for intervention in unit.get('intervention_variants',['original']):roles.add(_role({'target_kind':target,'grid':grid,'horizon':horizon,'phase_variant':variant,'intervention_variant':intervention},count))
    roles=sorted(roles or {_role(s['record'],n) for s in samples for n in schedules})
    trials=ctx.protocol.get('trials',{});trial_list=list(trials.values()) if isinstance(trials,dict) else trials
    intended_neural={x.get('method_id',f"{x['family']}::{x.get('variant_id','base')}::n{x.get('data_count',ctx.protocol.get('parent_counts',{}).get('train',0))}") for x in trial_list if x.get('phase') in ('main','data_efficiency') and x.get('target_kind')==target}
    if ctx.protocol.get('profile')=='confirmation':intended_neural={x.get('method_id',x['family']) for x in selection.get('selected_models',[]) if x.get('target_kind')==target}
    if not intended_neural:intended_neural={x.get('method_id',x['family']) for x in selection.get('selected_models',[]) if x.get('target_kind')==target}
    expected_seed_map={}
    for trial in trial_list:
        if trial.get('target_kind')!=target or trial.get('phase') not in ('main','data_efficiency'):continue
        method_id=trial.get('method_id',f"{trial['family']}::{trial.get('variant_id','base')}::n{trial.get('data_count',ctx.protocol.get('parent_counts',{}).get('train',0))}")
        expected_seed_map.setdefault(method_id,set()).add(trial['seed'])
    for item in selection.get('selected_models',[]):
        if item.get('target_kind')==target:expected_seed_map.setdefault(item.get('method_id',item['family']),set()).add(item['seed'])
    expected_seed_map={k:sorted(v) for k,v in expected_seed_map.items()}
    for name,method in methods.items():
        if method.metadata.get('seed') is None:expected_seed_map[method.metadata['method_id']]=list(seeds)
    missing=[]
    for method_id in sorted(intended_neural):
        for seed in expected_seed_map.get(method_id,seeds):
            if f'{method_id}/seed-{seed}' not in methods:missing.append({'family':method_id.split('::')[0],'method_id':method_id,'seed':seed,'reason':'No selected checkpoint; remains in intended denominator'})
    registry=dict(schema='tdn.response-query-registry/v1',parents=intended_parents,panel=ctx.unit.get('panel','primary'),descriptive_only=ctx.unit.get('descriptive_only',False),
        query_roles=roles,query_ids=roles,seeds=seeds,
        norm='nodal_rms' if target=='fd_same_grid' else 'trigonometric_interpolant_rms',
        methods=sorted(intended_neural|{getattr(f,'metadata',{}).get('method_id',name) for name,f in methods.items()}),
        missing_models=missing,expected_seed_map=expected_seed_map,grid_horizon_targets=[{'target_kind':target,'grid':unit.get('grid'),'horizons':unit.get('horizons')} for unit in declared_refs],
        target_kind=target,selection_sha256=_file_hash(selection_path) if selection_path else None,
        checkpoints={name:getattr(f,'metadata',{}) for name,f in methods.items()},
        primary_comparisons=ctx.protocol.get('evaluation',{}).get('comparisons',[]),frozen_before_evaluation=True,
        reference_eligible_queries=sum(s['reference_accepted'] for s in samples),attempted_queries=len(samples),
        grid_horizon_menu='Parent-relative query_roles; original globally unique input query_id retained separately')
    _write(Path(ctx.path)/'query_registry.json',registry)
    for sample in samples:
        r=sample['record'];inp=inference_input(sample)
        for name,method in methods.items():
          metadata=method.metadata
          for count in schedules:
            ctx.budget.check();prediction,seconds,error=_result(method,inp,count)
            result=endpoint_metrics(prediction,sample['target'],anchor=sample['anchor_fp32'],u0=sample['u0'],record=r,
                tolerance=ctx.protocol.get('evaluation',{}).get('rms_target',2e-5))
            result.update(method=metadata['method_id'],family=metadata['family'],model_id=name,steps=count,
                query_role=_role(r,count),global_input_query_id=r['query_id'],query_id=digest([r['query_id'],count])[:24],
                execution_seconds=seconds,execution_seconds_scope='diagnostic call including Python overhead; not randomized deployment timing',failure=error,
                shared_reference=True,seed=metadata.get('seed'),checkpoint_path=metadata.get('checkpoint_path'))
            if sample is samples[0] and count==1:
                filename=f"spatial-{digest([name,r['query_id']])[:20]}.npz"
                shape=tuple(sample['u0'].shape)
                out=prediction.detach().cpu().numpy() if prediction is not None else np.full(shape,np.nan)
                truth=sample['target'].detach().cpu().numpy()
                np.savez_compressed(Path(ctx.path)/filename,input=sample['u0'].detach().cpu().numpy(),reference=truth,prediction=out,error=out-truth)
                spatial_examples.append({'parent_id':r['parent_id'],'query_id':r['query_id'],'method':metadata['method_id'],'seed':metadata.get('seed'),
                    **{key:r.get(key) for key in ('target_kind','grid','panel','horizon','kappa','reaction_rate','phase_variant','intervention_variant','descriptive_only')},
                    'arrays':filename,'sha256':_file_hash(Path(ctx.path)/filename),'reference_accepted':r['reference']['accepted'],'prediction_finite':result['prediction_finite'],
                    'selection':'first declared query, single complete call; no result-informed example selection'})
                _write(Path(ctx.path)/'spatial_examples.json',{'rows':spatial_examples})
            if metadata.get('seed') is None:
                for seed in seeds:rows.append(result|{'seed':seed,'measurement_reused_across_seeds':True,'measurement_id':digest([r['query_id'],count,name])})
            else:rows.append(result|{'measurement_reused_across_seeds':False})
        for item in missing:
            for count in schedules:
                absent=endpoint_metrics(None,sample['target'],record=r)
                rows.append(absent|dict(method=item['method_id'],family=item['family'],model_id=f"{item['method_id']}/seed-{item['seed']}",seed=item['seed'],steps=count,
                    query_role=_role(r,count),global_input_query_id=r['query_id'],query_id=digest([r['query_id'],count])[:24],status='NOT_MEASURED',
                    unavailable_reason=item['reason'],norm=registry['norm']))
        _write(Path(ctx.path)/'endpoint_metrics.json',{'rows':rows,'summary':summarize_endpoints(rows)})
    declared_parent=next((p['parent_id'] for p in ctx.protocol.get('parents',[]) if p['split']==samples[0]['record']['split']),samples[0]['record']['parent_id'])
    representative=not ctx.unit.get('descriptive_only',False) and any(s['record']['parent_id']==declared_parent for s in samples)
    ctx._diagnostic_samples=samples;ctx._diagnostic_teacher_cache={};ctx._diagnostic_teacher_costs=[]
    if representative:
        selected_samples=[s for s in samples if s['record']['parent_id']==declared_parent]
        rollout=run_rollout_panel(ctx,selected_samples,methods)
        sensitivity=run_sensitivity_panel(ctx,selected_samples,methods)
        dense=run_dense_panel(ctx,selected_samples,methods)
        from .diagnostics import run as run_diagnostics
        diagnostics=run_diagnostics(ctx,selected_samples,methods)
        from .inspection import run as run_inspection
        inspection=run_inspection(ctx,selected_samples,methods)
        from .quadrature_audit import run as run_quadrature_audit
        run_quadrature_audit(ctx,selected_samples,methods)
    else:
        rollout=[];sensitivity=[];dense=[];diagnostics={};inspection={}
        for filename in ('rollout_metrics.json','mean_paths.json','sensitivity_metrics.json','dense_query_metrics.json','symmetry_metrics.json','parameter_metrics.json','source_diagnostics.json','quadrature_refinement.json'):
            _write(Path(ctx.path)/filename,dict(rows=[],status='NOT_SELECTED_FOR_REPRESENTATIVE_PANEL',declared_parent_id=declared_parent,
                reason='Auxiliary diagnostics restricted to first declared whole-cohort parent per target/grid; stress/intervention endpoints reported separately'))
    _write(Path(ctx.path)/'diagnostic_teacher_costs.json',dict(rows=ctx._diagnostic_teacher_costs,
        total_cpu_acquisition_seconds=sum(row['cpu_acquisition_seconds'] for row in ctx._diagnostic_teacher_costs),
        scope='CPU diagnostic teacher acquisition charged inside this stage allocation, outside inference timers; native GPU allocation can include this CPU work. Main reference bank remains CPU-only stages.'))
    from .core import check
    summary=summarize_endpoints(rows)
    ctx.record('evaluation/endpoints',ctx.unit.get('mechanism_ids',['M6']),metrics=summary|{'selected_neural_methods':neural_count},
        checks=[check('finite-predictions-valid-teachers',summary['prediction_failures_with_valid_reference'],0,category='correctness'),
                check('selected-neural-coverage',neural_count,1,'ge',category='gap')],evidence=['endpoint_metrics.json','query_registry.json','rollout_metrics.json','sensitivity_metrics.json','dense_query_metrics.json'])
    return summary|{'rollout_rows':len(rollout),'sensitivity_rows':len(sensitivity),'dense_query_rows':len(dense),'additional_diagnostics':diagnostics,'source_inspection':inspection}


def _representatives(samples,count):
    result=[];seen=set()
    for s in samples:
        if s['record']['parent_id'] not in seen:result.append(s);seen.add(s['record']['parent_id'])
        if len(result)>=count:break
    return result


def diagnostic_teacher(ctx,u,record,horizon,purpose):
    """Reuse exact accepted input/physics queries; charge fresh CPU acquisition."""
    key=(array_digest(u),record['target_kind'],tuple(record['grid']),tuple(record['lengths']),float(record['kappa']),float(record['reaction_rate']),float(horizon))
    cache=getattr(ctx,'_diagnostic_teacher_cache',{})
    if key in cache:return cache[key]
    for sample in getattr(ctx,'_diagnostic_samples',[]):
        r=sample['record']
        if (r['reference']['accepted'] and r['target_kind']==record['target_kind'] and r['grid']==record['grid'] and r['lengths']==record['lengths']
            and r['kappa']==record['kappa'] and r['reaction_rate']==record['reaction_rate'] and r['horizon']==horizon
            and torch.equal(sample['u0'].to(u.device),u)):
            answer=(sample['target'].detach().cpu(),r['reference']);cache[key]=answer;ctx._diagnostic_teacher_cache=cache
            if hasattr(ctx,'_diagnostic_teacher_costs'):ctx._diagnostic_teacher_costs.append(dict(parent_id=record['parent_id'],horizon=horizon,purpose=purpose,
                cpu_acquisition_seconds=0.,status='REUSED_ACCEPTED_CPU_REFERENCE',source_query_id=r['query_id']))
            return answer
    eq,geom=equation_geometry(record);start=time.perf_counter()
    answer=generate_reference(u,horizon,eq,geom,record['target_kind'],ctx.protocol.get('reference'),ctx.budget)
    elapsed=time.perf_counter()-start;cache[key]=answer;ctx._diagnostic_teacher_cache=cache
    if hasattr(ctx,'_diagnostic_teacher_costs'):ctx._diagnostic_teacher_costs.append(dict(parent_id=record['parent_id'],horizon=horizon,purpose=purpose,
        cpu_acquisition_seconds=elapsed,status='FRESH_BOUNDED_CPU_REFERENCE',accepted=answer[1]['accepted'],input_sha256=key[0]))
    return answer


def run_rollout_panel(ctx,samples,methods):
    """Autonomous increasing-horizon rollout; each intermediate teacher audited."""
    settings=ctx.protocol.get('evaluation',{});steps=settings.get('rollout_steps',[1,2,4,8,16,32]);rows=[];mean_paths=[]
    for sample in _representatives(samples,settings.get('rollout_parents',1)):
        record=sample['record'];eq,geom=equation_geometry(record);dt=min(ctx.protocol.get('train_horizons',[.02]));truth={}
        for n in steps:
            ctx.budget.check();h=dt*n
            value,ref=diagnostic_teacher(ctx,sample['u0'],record,h,'rollout')
            truth[n]=(value.to(ctx.device),ref)
        for name,method in methods.items():
            u=sample['u0'];first_failure=None;previous_mean=float(u.double().mean());previous_rhs=None
            for n in range(1,max(steps)+1):
                ctx.budget.check();inp=inference_input({'u0':u,'record':record|{'horizon':dt}})
                u,seconds,failure=_result(method,inp,1)
                if u is None or not bool(torch.isfinite(u).all()):first_failure=dt*n;break
                from .numerics import inner
                centered=u.double()-u.double().mean();variance=float(inner(centered,centered,geom,record['target_kind']).mean())
                mean=float(u.double().mean());mean_rhs=record['reaction_rate']*(mean*(1-mean)-variance)
                mean_paths.append(dict(parent_id=record['parent_id'],method=method.metadata['method_id'],seed=method.metadata.get('seed'),
                    time=n*dt,macrosteps=n,prediction_mean=mean,prediction_physical_variance=variance,
                    mean_ode_trapezoid_defect=None if previous_rhs is None or record['target_kind']=='galerkin_projected_fine' else (mean-previous_mean)/dt-.5*(mean_rhs+previous_rhs),
                    mean_ode_scope='finite difference/trapezoid diagnostic, not exact identity; projected closure target excluded',
                    reference_mean=float(truth[n][0].mean()) if n in truth else None,reference_accepted=truth[n][1]['accepted'] if n in truth else None))
                previous_mean=mean;previous_rhs=mean_rhs
                if n in steps:
                    ref,meta=truth[n];r=record|{'horizon':dt*n,'reference':meta}
                    metrics=endpoint_metrics(u,ref,u0=sample['u0'],record=r)
                    rows.append(metrics|dict(method=method.metadata['method_id'],model_id=name,seed=method.metadata.get('seed'),macrosteps=n,
                        macrostep=dt,first_failure_time=None,scope='increasing physical horizon, autonomous re-encoding each step'))
            if first_failure is not None:rows.append(dict(parent_id=record['parent_id'],method=method.metadata['method_id'],model_id=name,
                status='FAILED',first_failure_time=first_failure,failure=failure or 'NONFINITE_PREDICTION'))
            _write(Path(ctx.path)/'rollout_metrics.json',{'rows':rows,'representative_parent_count':settings.get('rollout_parents',1),
                'scope':'preregistered first parent(s); all queried horizons retained; not whole-cohort uncertainty'})
    _write(Path(ctx.path)/'mean_paths.json',{'rows':mean_paths,'missing_reference_times':'No interpolation or imputation; teachers only at declared queried horizons'})
    return rows


def run_sensitivity_panel(ctx,samples,methods):
    from .numerics import inner
    settings=ctx.protocol.get('evaluation',{});rows=[]
    for sample in _representatives(samples,settings.get('sensitivity_parents',1)):
        r=sample['record'];u=sample['u0'];eq,geom=equation_geometry(r)
        norm='nodal_rms' if r['target_kind']=='fd_same_grid' else 'trigonometric_interpolant_rms'
        def rms(value):
            value=value.double()
            return inner(value,value,geom,r['target_kind']).mean().clamp_min(0).sqrt()
        x=torch.arange(u.shape[-1],dtype=u.dtype,device=u.device)/u.shape[-1]
        shape=[1]*u.ndim;shape[-1]=u.shape[-1]
        directions={'mean':torch.ones_like(u),'first_mode':torch.cos(2*torch.pi*x).reshape(shape).expand_as(u)}
        for direction_name,direction in directions.items():
            direction=direction/rms(direction).to(direction);references={}
            for epsilon in settings.get('sensitivity_epsilons',[1e-3,5e-4]):
                ctx.budget.check();states=[];metas=[]
                for sign in (-1,1):
                    state,ref=diagnostic_teacher(ctx,u+sign*epsilon*direction,r,r['horizon'],'input_sensitivity')
                    states.append(state.to(ctx.device));metas.append(ref)
                tangent=(states[1]-states[0])/(2*epsilon);references[epsilon]=tangent
                for name,method in methods.items():
                    out=[];reason=None
                    for sign in (-1,1):
                        pred,_,failure=_result(method,inference_input({'u0':u+sign*epsilon*direction,'record':r}),1);out.append(pred);reason=reason or failure
                    finite=all(p is not None and bool(torch.isfinite(p).all()) for p in out)
                    difference=(out[1].double()-out[0].double())/(2*epsilon)-tangent if finite else None
                    error=float(rms(difference)) if finite else None
                    rows.append(dict(parent_id=r['parent_id'],target_kind=r['target_kind'],method=method.metadata['method_id'],model_id=name,
                        direction=direction_name,epsilon=epsilon,tangent_error_rms=error,tangent_error=error,reference_accepted=all(m['accepted'] for m in metas),
                        norm=norm,direction_normalization_norm=norm,direction_rms=float(rms(direction)),
                        nodal_tangent_error_rms=float(difference.square().mean().sqrt()) if finite else None,
                        reference_uncertainty_amplification=sum((m.get('uncertainty_rms') or 0) for m in metas)/(2*epsilon) if all(m.get('uncertainty_rms') is not None for m in metas) else None,
                        status='COMPLETED' if finite else 'FAILED',failure=reason,linear_response_established=False))
            eps=sorted(references)
            if len(eps)>=2:
                change=references[eps[0]]-references[eps[1]]
                variation=float(rms(change));scale=float(rms(references[eps[0]]))
                for row in rows:
                    if row['parent_id']==r['parent_id'] and row['direction']==direction_name:
                        row['teacher_tangent_halving_change_rms']=variation
                        row['nodal_teacher_tangent_halving_change_rms']=float(change.square().mean().sqrt())
                        row['teacher_tangent_rms']=scale
                        row['linear_response_established']=bool(row['reference_accepted'] and variation<=max(1e-5,.01*scale))
            _write(Path(ctx.path)/'sensitivity_metrics.json',{'rows':rows,'scope':'central differences of actual FP32 perturbed inputs; direction normalization, tangent errors and halving use the declared target norm; two amplitudes audit linear response; separate from timing'})
    return rows


def run_dense_panel(ctx,samples,methods):
    """Requested-time ordering/call parity with one state, no teacher in decode."""
    sample=samples[0];r=sample['record'];eq,geom=equation_geometry(r);rows=[]
    times=sorted(set([.25*r['horizon'],.5*r['horizon'],r['horizon']]))
    teachers={}
    for h in times:
        value,meta=diagnostic_teacher(ctx,sample['u0'],r,h,'dense_query');teachers[h]=(value.to(ctx.device),meta)
    for name,method in methods.items():
        predictions={}
        for h in reversed(times):
            pred,elapsed,failure=_result(method,inference_input({'u0':sample['u0'],'record':r|{'horizon':h}}),1);predictions[h]=pred
            target,meta=teachers[h];row=endpoint_metrics(pred,target,record=r|{'horizon':h,'reference':meta})
            row.update(method=method.metadata['method_id'],model_id=name,execution_seconds=elapsed,failure=failure,
                encoding_reuse=False,reuse_status='NOT_MEASURED; independent complete calls',query_order='descending')
            rows.append(row)
        for h in times:
            pred,_,_=_result(method,inference_input({'u0':sample['u0'],'record':r|{'horizon':h}}),1)
            delta=float((pred-predictions[h]).abs().max()) if pred is not None and predictions[h] is not None else None
            for row in rows:
                if row['model_id']==name and row['horizon']==h:row['query_order_max_difference']=delta
    _write(Path(ctx.path)/'dense_query_metrics.json',{'rows':rows,'scope':'independent requested-time accuracy and order; reusable encoding costs require separate measured API'})
    return rows
