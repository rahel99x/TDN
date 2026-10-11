"""Complete-call randomized paired latency, separate transfers and honest costs."""
from __future__ import annotations
import math
from pathlib import Path
import random
import statistics
import time
import torch
from .references import _write


def scheduled_methods(methods,schedules):
    """Give every learned endpoint operator the identical autonomous menu."""
    from .evaluation import _execute
    result={}
    for name,method in methods.items():
        for count in schedules:
            def call(sample,method=method,count=count):return _execute(method,sample,count)
            call.metadata=dict(method.metadata,steps=count)
            if hasattr(method,'model'):call.model=method.model
            result[f'{name}/steps-{count}']=call
    return result


def model_resources(method):
    """Static bytes and parameter counts, distinct from live allocator peaks."""
    model=getattr(method,'model',None);checkpoint=method.metadata.get('checkpoint_path')
    out=dict(total_parameters=sum(p.numel() for p in model.parameters()) if model is not None else 0,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad) if model is not None else 0,
        parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()) if model is not None else 0,
        buffer_bytes=sum(p.numel()*p.element_size() for p in model.buffers()) if model is not None else 0,
        checkpoint_bytes=None,artifact_directory_bytes=None,checkpoint_path=checkpoint)
    if checkpoint:
        path=Path(checkpoint);file=path/'selected.pt' if path.is_dir() else path
        out['checkpoint_bytes']=file.stat().st_size if file.is_file() else None
        root=path if path.is_dir() else path.parent
        out['artifact_directory_bytes']=sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())
    else:out.update(checkpoint_bytes=0,artifact_directory_bytes=0)
    return out


def offline_accounting(ctx):
    """Charge cumulative trial work once and every measured chunk setup once.

    This is measured scientific wall work. Slurm allocation/setup, unavailable
    attempts, monetary cost and energy are explicitly outside this ledger.
    """
    import json
    rows=[];trials={};missing=[];common=0.;families={}
    for stage,value in ctx.prerequisites.items():
        path=Path(value);unit=ctx.protocol.get('units',{}).get(stage,{})
        if (path/'trial.json').exists():
            r=json.loads((path/'trial.json').read_text());trial_id=r['trial_id'];family=r['family']
            total=r.get('cost',{}).get('total_seconds');known=total is not None and math.isfinite(total) and total>=0
            slot=trials.setdefault(trial_id,dict(family=family,cumulative=None,stage_extra=0.,sources=[]))
            if known:slot['cumulative']=max(slot['cumulative'] or 0.,total)
            else:missing.append(dict(stage=stage,field='trial.cost.total_seconds'))
            extra=(r.get('continuation_noop_seconds',0.) if r.get('continuation_noop') else
                   r.get('stage_setup_seconds',0.)+sum(r.get('postprocessing_cost',{}).values()))
            slot['stage_extra']+=extra;slot['sources'].append(str(path/'trial.json'))
            rows.append(dict(stage=stage,kind='training_chunk',trial_id=trial_id,family=family,
                cumulative_training_seconds=total,stage_extra_seconds=extra,source=str(path/'trial.json'),add_cumulative_chunks=False))
        elif unit.get('kind') in ('fields','references','select','gate','compression'):
            summary=path/'summary.json'
            if not summary.exists():missing.append(dict(stage=stage,field='summary.elapsed_seconds'));continue
            r=json.loads(summary.read_text());elapsed=r.get('elapsed_seconds')
            if elapsed is None or not math.isfinite(elapsed) or elapsed<0:missing.append(dict(stage=stage,field='summary.elapsed_seconds'));continue
            common+=elapsed;rows.append(dict(stage=stage,kind=unit.get('kind'),elapsed_seconds=elapsed,source=str(summary)))
    for trial_id,r in trials.items():
        if r['cumulative'] is not None:families[r['family']]=families.get(r['family'],0.)+r['cumulative']+r['stage_extra']
    return dict(rows=rows,unique_trials=trials,family_training_tuning_seconds=families,shared_preparation_selection_seconds=common,
        accounted_total_seconds=common+sum(families.values()),missing=missing,
        scope='Measured scientific wall work only; each cumulative trial once plus each chunk setup/postprocessing. Shared offline work fully charged to each learned family for per-family estimates.',
        unmeasured=['scheduler allocation and process startup','unretained failed attempts','energy','monetary rates'],
        complete_campaign_cost_available=False,compilation_seconds=0.,compilation_scope='Declared eager implementation; no compilation arm executed')


def accuracy_cost_frontiers(rows,tolerances,offline=None):
    """Frozen-tolerance, reference-informed descriptive representative frontier.

    A posthoc best schedule is not a deployable policy. Do not extrapolate the
    representative-field qualification to unseen parent coverage.
    """
    from .metrics import accuracy_status
    groups={};qualification=[];frontiers=[];amortization=[]
    for row in rows:groups.setdefault((row['parent_id'],row['query_id'],row['target_kind'],tuple(row['grid'])),[]).append(row)
    for key,group in groups.items():
        parent,query,target,grid=key
        for tolerance in tolerances:
            qualified=[]
            for r in group:
                status=accuracy_status(r.get('rms'),r.get('error_max'),dict(accepted=r.get('reference_accepted',False),
                    uncertainty_rms=r.get('uncertainty_rms'),uncertainty_maximum=r.get('uncertainty_maximum')),tolerance)
                measured=r.get('median_seconds') is not None and r.get('timing_status')=='COMPLETED' and r.get('measured_rounds')==r.get('requested_rounds')
                row=dict(parent_id=parent,query_id=query,target_kind=target,grid=list(grid),tolerance=tolerance,method=r['method'],
                    family=r['family'],model_id=r['model_id'],seed=r.get('seed'),steps=r.get('steps'),rms=r.get('rms'),error_max=r.get('error_max'),norm=r.get('norm'),
                    median_seconds=r.get('median_seconds'),accuracy_qualification=status,complete_timing=measured)
                qualification.append(row)
                if status=='QUALIFIED' and measured:qualified.append(row)
            for category in ('all','analytic','learned'):
                candidates=[r for r in qualified if category=='all' or (r['family'].startswith('analytic_'))==(category=='analytic')]
                best=min(candidates,key=lambda r:(r['median_seconds'],r['model_id'])) if candidates else None
                frontiers.append(dict(parent_id=parent,query_id=query,target_kind=target,grid=list(grid),tolerance=tolerance,category=category,
                    attempted_operating_points=sum(category=='all' or (r['family'].startswith('analytic_'))==(category=='analytic') for r in group),
                    qualified_operating_points=len(candidates),status='QUALIFIED_REPRESENTATIVE' if best else 'NO_QUALIFIED_TIMED_CANDIDATE',selected=best,
                    selection_scope='reference-informed posthoc frontier; not a deployed schedule policy',full_cohort_coverage_established=False))
            baselines=[r for r in qualified if r['family'].startswith('analytic_')]
            baseline=min(baselines,key=lambda r:r['median_seconds']) if baselines else None
            for candidate in (r for r in qualified if not r['family'].startswith('analytic_')):
                family_cost=(offline or {}).get('family_training_tuning_seconds',{}).get(candidate['family'])
                numerator=None if family_cost is None or (offline or {}).get('missing') else family_cost+offline['shared_preparation_selection_seconds']
                estimate=break_even(numerator,0.,candidate['median_seconds'],baseline['median_seconds'] if baseline else None,accuracy_equivalent=baseline is not None)
                amortization.append(dict(parent_id=parent,query_id=query,target_kind=target,grid=list(grid),tolerance=tolerance,
                    candidate=candidate['model_id'],comparator=baseline['model_id'] if baseline else None,
                    candidate_accounted_offline_seconds=numerator,analytic_training_offline_seconds=0.,
                    analytic_zero_scope='Fixed numerical formula requires no data fitting; common setup and first-call runtime measured separately',
                    conditional_representative_break_even=estimate,complete_campaign_break_even_queries=None,
                    complete_campaign_break_even_reason='Unmeasured allocation/startup or unretained attempts; full-cohort matched-accuracy coverage and frozen deployment policy not established',
                    deployment_scope='resident-device repeated identical declared workload; transfers reported separately'))
    return dict(qualification_rows=qualification,rows=frontiers,amortization_rows=amortization,tolerances=list(tolerances),
        scope='Representative validation timing fields, paired actual accuracy. Different targets/grids/horizons remain separate; no universal leaderboard or full-cohort solver claim.')


def break_even(offline_new,offline_base,call_new,call_base,*,accuracy_equivalent):
    values=(offline_new,offline_base,call_new,call_base)
    if any(v is None or not math.isfinite(v) or v<0 for v in values):
        return {'queries':None,'status':'NOT_MEASURED','reason':'Required cost field absent, nonfinite or negative'}
    if not accuracy_equivalent:return {'queries':None,'status':'INELIGIBLE','reason':'Matched accuracy and coverage not established'}
    if call_base<=call_new:return {'queries':None,'status':'NO_BREAK_EVEN','reason':'No positive per-call cost margin'}
    if offline_new<=offline_base:return {'queries':0.,'status':'ALREADY_AMORTIZED','reason':'Nonpositive incremental offline work with a positive per-call margin'}
    return {'queries':max(0.,(offline_new-offline_base)/(call_base-call_new)),'status':'ESTIMATED','reason':None}


def paired_timing(methods,*,device='cpu',repeats=20,warmup=3,seed=0,budget=None):
    """Methods are zero-argument complete prediction calls; no teacher handles."""
    if not methods or repeats<1 or warmup<0:raise ValueError('Methods and positive repeats required')
    gpu=torch.device(device).type=='cuda';sync=lambda:torch.cuda.synchronize(device) if gpu else None
    def check():
        if budget is not None:budget.check()
    first={};raw=[];warm={k:[] for k in methods};failures={};memory={}
    def call_timed(name):
        check();sync();start=time.perf_counter();answer=methods[name]();sync();elapsed=time.perf_counter()-start
        if not bool(torch.isfinite(answer).all()):raise FloatingPointError('Nonfinite inference output')
        return elapsed
    with torch.no_grad():
        for name in methods:
            try:
                first[name]=call_timed(name)
                for _ in range(warmup):call_timed(name)
            except (RuntimeError,ValueError,FloatingPointError) as e:failures[name]=f'{type(e).__name__}: {e}';first[name]=None
        rng=random.Random(seed)
        for round_id in range(repeats):
            order=list(methods);rng.shuffle(order)
            for position,name in enumerate(order):
                if name in failures:continue
                try:
                    elapsed=call_timed(name);warm[name].append(elapsed)
                    raw.append(dict(round=round_id,position=position,method=name,seconds=elapsed))
                except (RuntimeError,ValueError,FloatingPointError) as e:failures[name]=f'{type(e).__name__}: {e}'
        # Memory probes are separate from latency rounds and cannot rank a
        # group maximum as a per-model measurement.
        for name in methods:
            if name in failures:continue
            if gpu:
                sync();torch.cuda.reset_peak_memory_stats(device);base_alloc=torch.cuda.memory_allocated(device)
                base_reserved=torch.cuda.memory_reserved(device);methods[name]();sync()
                memory[name]=dict(allocated_baseline_bytes=base_alloc,reserved_baseline_bytes=base_reserved,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                    incremental_peak_allocated_bytes=max(0,torch.cuda.max_memory_allocated(device)-base_alloc),
                    incremental_peak_reserved_bytes=max(0,torch.cuda.max_memory_reserved(device)-base_reserved),
                    scope='Common resident-process baseline includes other loaded selected models; incremental peak and isolated fresh-process probe are separate')
            else:memory[name]=dict(peak_allocated_bytes=None,peak_reserved_bytes=None,reason='CUDA allocator metrics not applicable to CPU')
    summaries={name:dict(first_invocation_warm_process_seconds=first.get(name),warm_median_seconds=statistics.median(v) if v else None,
        warm_min_seconds=min(v) if v else None,warm_max_seconds=max(v) if v else None,measured_rounds=len(v),
        requested_rounds=repeats,status='FAILED' if name in failures else 'COMPLETED',failure=failures.get(name),
        fresh_process_seconds=None,fresh_process_unavailable_reason='Not measured in a new interpreter',
        energy_joules=None,monetary_cost=None,unavailable_cost_reason='No energy/rate accounting supplied',memory=memory.get(name)) for name,v in warm.items()}
    return dict(schema='tdn.response-timing/v1',raw_rounds=raw,summaries=summaries,warmup=warmup,seed=seed,
        synchronization='torch.cuda.synchronize before and after every measured call' if gpu else 'synchronous CPU calls',
        call_contract='complete anchor + features + encoder + decoder + assembly; resident input/output; torch.no_grad',
        inference_graphs=False,p99=None,p99_reason='Dedicated independent tail study not performed',fresh_process_status='NOT_MEASURED')


def transfer_timing(u,device,repeats=3):
    if torch.device(device).type!='cuda':return {'status':'NOT_APPLICABLE','host_to_device_seconds':None,'device_to_host_seconds':None}
    src=u.detach().cpu();up=[];down=[]
    for _ in range(repeats):
        torch.cuda.synchronize(device);start=time.perf_counter();dst=src.to(device);torch.cuda.synchronize(device);up.append(time.perf_counter()-start)
        start=time.perf_counter();out=dst.cpu();torch.cuda.synchronize(device);down.append(time.perf_counter()-start)
    return dict(status='MEASURED',host_to_device_raw=up,device_to_host_raw=down,host_to_device_seconds=statistics.median(up),
        device_to_host_seconds=statistics.median(down),bytes=src.numel()*src.element_size(),pinned=False)


def fresh_process_probe(method,inp,path,device='cpu',timeout=120,expected=None):
    """True new interpreter startup, checkpoint load and first prediction.

    Inputs contain only the present field and public physics. Probe files live
    in the cost stage, never mutate a sealed training directory.
    """
    import json,subprocess,sys,numpy as np
    from .evaluation import inference_input
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path/'input.npz',u0=inp['u0'].detach().cpu().numpy())
    record=inference_input(inp)['record'];meta=method.metadata
    _write(path/'request.json',{'record':record,'method':meta,'device':str(device)})
    start=time.perf_counter()
    try:
        result=subprocess.run([sys.executable,'-m','tdn.analysis.response.cost','fresh',str(path.resolve())],
            capture_output=True,text=True,timeout=timeout,check=False)
        elapsed=time.perf_counter()-start
        (path/'child.stdout').write_text(result.stdout);(path/'child.stderr').write_text(result.stderr)
        if result.returncode!=0:return {'status':'FAILED','elapsed_seconds':elapsed,'exit_code':result.returncode,'reason':'Fresh process failed; child logs retained'}
        result=json.loads((path/'result.json').read_text());result['parent_observed_total_seconds']=elapsed
        if expected is not None:
            with np.load(path/'output.npz',allow_pickle=False) as arrays:actual=torch.from_numpy(arrays['prediction'].copy())
            finite=bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
            result['reload_maximum_difference']=float((actual.double()-expected.detach().cpu().double()).abs().max()) if finite else None
            result['reload_parity_status']='MEASURED' if finite else 'UNAVAILABLE_NONFINITE_OUTPUT'
            result['reload_exact_equal']=bool(torch.equal(actual,expected.detach().cpu()))
        return result
    except subprocess.TimeoutExpired:
        return {'status':'CENSORED','elapsed_seconds':time.perf_counter()-start,'reason':'Fresh process timeout; no success latency inferred'}


def _fresh_child(path):
    import json,resource,numpy as np
    from .evaluation import classical_methods
    from .training import load_model,predict
    path=Path(path);request=json.loads((path/'request.json').read_text());device=request['device'];m=request['method']
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sync=lambda:torch.cuda.synchronize(device) if torch.device(device).type=='cuda' else None
    gpu=torch.device(device).type=='cuda'
    if gpu:torch.cuda.reset_peak_memory_stats(device)
    start=time.perf_counter()
    with np.load(path/'input.npz',allow_pickle=False) as arrays:u=torch.from_numpy(arrays['u0'].copy()).to(device)
    inp={'u0':u,'record':request['record']}
    if m.get('checkpoint_path'):
        from .evaluation import _execute
        model,_=load_model(m['checkpoint_path'],device);call=lambda:_execute(lambda sample:predict(model,sample),inp,m.get('steps',1))
    else:
        methods=classical_methods(inp['record'],[m['steps']]);key=f"{m['family']}/steps-{m['steps']}";call=lambda:methods[key](inp)
    sync();loaded=time.perf_counter()-start;start=time.perf_counter()
    with torch.no_grad():out=call()
    sync();inference=time.perf_counter()-start
    np.savez_compressed(path/'output.npz',prediction=out.detach().cpu().numpy())
    _write(path/'result.json',dict(status='COMPLETED' if bool(torch.isfinite(out).all()) else 'FAILED',
        data_device_and_model_load_seconds=loaded,first_prediction_seconds=inference,
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        isolated_cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if gpu else None,
        isolated_cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if gpu else None,
        memory_scope='Fresh process: input, this selected model or analytic rule, inference intermediates and output; external device usage not included',
        torch_cuda_runtime=torch.version.cuda,device=device,tf32=False,
        scope='Parent wall includes interpreter/imports, input load/transfer, checkpoint load, first prediction and serialization; internal timers are overlapping components'))


def _run_performance(ctx,sample,selected,query_teachers):
    record=sample['record'];u=sample['u0'].to(ctx.device);target=sample['target'].to(ctx.device);grid=list(u.shape[2:])
    settings=ctx.protocol.get('cost',{});performance=[];performance_rows=[]
    _write(Path(ctx.path)/'performance.json',dict(rows=[],panels=[],status='NO_COMPLETED_REPRESENTATIVE_WORKLOAD'))
    _write(Path(ctx.path)/'profiling.json',dict(rows=[],scope='Separate instrumented calls; not deployment latency'))
    from .performance import measure_model_workloads
    from .references import equation_geometry
    eq,geom=equation_geometry(record);done=set()
    ordered=sorted(selected.items(),key=lambda item:(item[1].metadata.get('variant_id')!='primary',item[1].metadata.get('seed',0),item[0]))
    for name,method in ordered:
        if not hasattr(method,'model') or method.metadata['family'] in done:continue
        ctx.budget.check();done.add(method.metadata['family'])
        result=measure_model_workloads(method.model,u,record['horizon'],eq,geom,record['target_kind'],
            repeats=settings.get('reuse_repeats',settings.get('repeats',20)),warmup=settings.get('warmup',1),query_counts=tuple(settings.get('query_counts',[1,2,4,8,16])),
            batch_sizes=tuple(settings.get('batch_sizes',[1,4,16])),target_teacher=target,budget=ctx.budget,seed=ctx.protocol.get('root_seed',0),
            query_teachers=query_teachers)
        performance.append({'model_id':name,'family':method.metadata['family'],'method':method.metadata['method_id'],'target_kind':record['target_kind'],
            'parent_id':record['parent_id'],'grid':grid,'checkpoint_scope':'primary variant if available, then minimum declared seed and method ID per family; no performance-based selection',**result})
        for kind,key in [('batch','batch_panels'),('queries','reuse_panels')]:
            for perf in result.get(key,[]):
                for arm,summary in perf['timing']['summaries'].items():
                    error=perf.get('final_endpoint_error') or {}
                    query_accuracy=[row for row in perf.get('query_accuracy',[]) if row['arm']==arm]
                    complete_accuracy=len(query_accuracy)==perf.get('query_count') and bool(query_accuracy)
                    rms_values=[row['error_rms'] for row in query_accuracy if row.get('error_rms') is not None]
                    maximum_values=[row['error_max'] for row in query_accuracy if row.get('error_max') is not None]
                    performance_rows.append(dict(method=method.metadata['method_id'],family=method.metadata['family'],model_id=name,
                        workload=kind,arm=arm,grid=grid,target_kind=record['target_kind'],parent_id=record['parent_id'],
                        batch_size=perf.get('batch_size',1),query_count=perf.get('query_count',1),
                        median_seconds=summary.get('warm_median_seconds'),error_rms=error.get('rms') if arm.startswith('model') or arm in ('batched','sequential') else None,
                        error_max=error.get('maximum') if arm.startswith('model') or arm in ('batched','sequential') else None,
                        error_scope='Final endpoint for batch throughput; dense queries use separate per-query accepted-reference metrics',
                        all_queries_qualified=perf.get('all_queries_qualified',{}).get(arm),
                        accuracy_query_count=len(query_accuracy),expected_accuracy_query_count=perf.get('query_count') if kind=='queries' else 1,
                        accepted_reference_query_count=sum(row.get('reference_accepted',False) for row in query_accuracy),
                        worst_query_error_rms=max(rms_values) if complete_accuracy and len(rms_values)==len(query_accuracy) else None,
                        worst_query_error_max=max(maximum_values) if complete_accuracy and len(maximum_values)==len(query_accuracy) else None,
                        query_accuracy=query_accuracy,
                        peak_allocated_bytes=(summary.get('memory') or {}).get('peak_allocated_bytes'),
                        reference_accepted=record['reference']['accepted'],scope=perf.get('intermediate_query_accuracy','replicated-input throughput; one independent field')))
        _write(Path(ctx.path)/'performance.json',{'rows':performance_rows,'panels':performance})
        _write(Path(ctx.path)/'profiling.json',{'rows':[dict(model_id=p['model_id'],target_kind=p['target_kind'],grid=p['grid'],**op) for p in performance for op in p.get('profiler',{}).get('rows',[])],
            'scope':'Separate instrumented calls; not deployment latency'})
    return performance,performance_rows


def run_cost(ctx):
    from .evaluation import load_stage_samples,load_selected_methods,inference_input,classical_methods
    from .metrics import endpoint_metrics
    samples=[s for s in load_stage_samples(ctx,validation_only=True) if s['record'].get('phase_variant','original')=='original']
    if not samples:raise ValueError('Cost panel requires a declared validation field')
    settings=ctx.protocol.get('cost',{});groups={};workload=ctx.unit.get('cost_workload','all')
    if workload not in ('all','warm','dense'):raise ValueError('Unknown frozen cost workload')
    specs=ctx.unit.get('cost_models')
    selection_filter=None if specs is None else {(s['method_id'],s['seed']) for s in specs}
    offline=offline_accounting(ctx);_write(Path(ctx.path)/'offline_costs.json',offline)
    if workload=='warm':
        _write(Path(ctx.path)/'performance.json',dict(rows=[],panels=[],status='SEPARATE_DENSE_WORKLOAD',scope='Query/batch work is assigned to the declared dense cost stage'))
        _write(Path(ctx.path)/'profiling.json',dict(rows=[],status='SEPARATE_DENSE_WORKLOAD'))
    resources={}
    for sample in samples:
        r=sample['record'];key=(r['parent_id'],r.get('phase_variant','original'),tuple(r['grid']))
        groups.setdefault(key,[]).append(sample)
    representative=[max(group,key=lambda s:s['record']['horizon']) for group in groups.values()][:settings.get('representative_queries',1)]
    panels=[];endpoint=[];flat=[];performance=[];performance_rows=[];coverage=[]
    for panel_index,sample in enumerate(representative):
        ctx.budget.check();record=dict(sample['record']);u=sample['u0'].to(ctx.device);n=ctx.unit.get('grid',record['grid'])
        grid=[n]*(u.ndim-2) if isinstance(n,int) else list(n)
        if list(u.shape[2:])!=grid:
            raise ValueError('Cost grid requires matching precomputed CPU reference input; no teacher generation in GPU allocation')
        if ctx.unit.get('warm_horizon') is not None and not math.isclose(record['horizon'],ctx.unit['warm_horizon'],rel_tol=0.,abs_tol=1e-12):
            raise ValueError('Representative reference bank is missing the declared complete-call endpoint')
        target=sample['target'].to(ctx.device);selected=load_selected_methods(ctx,record['target_kind'],selection_filter=selection_filter)
        for spec in specs or []:
            name=f"{spec['method_id']}/seed-{spec['seed']}";method=selected.get(name)
            if method is None:continue
            architecture=spec.get('architecture_config')
            if architecture is not None:
                from .protocol import digest as protocol_digest
                if protocol_digest(architecture)!=spec['architecture_sha256']:raise ValueError('Frozen timing architecture descriptor hash changed')
                actual=getattr(getattr(method,'model',None),'config',{})
                if method.metadata['family']!=architecture['family'] or any(actual.get(k)!=v for k,v in architecture['model'].items()):
                    raise ValueError('Selected checkpoint architecture differs from its frozen timing representative')
            method.metadata['timing_architecture_sha256']=spec.get('architecture_sha256')
        if workload=='dense':
            missing=[f"{method_id}/seed-{seed}" for method_id,seed in (selection_filter or []) if f"{method_id}/seed-{seed}" not in selected]
            _write(Path(ctx.path)/'timing_coverage.json',dict(rows=[dict(workload='dense',declared_models=specs,measured_models=sorted(selected),
                missing_selected_models=missing,representative_only=True,phase_variant='original')]))
            _write(Path(ctx.path)/'model_resources.json',dict(rows=[dict(model_id=name,method=method.metadata['method_id'],family=method.metadata['family'],**model_resources(method)) for name,method in selected.items()]))
            performance,performance_rows=_run_performance(ctx,sample,selected,groups[(record['parent_id'],'original',tuple(record['grid']))])
            rounds=[summary['measured_rounds'] for model in performance for key in ('batch_panels','reuse_panels') for panel in model.get(key,[]) for summary in panel['timing']['summaries'].values()]
            from .core import check
            _write(Path(ctx.path)/'timing.json',dict(rows=[],panels=[],raw_rounds=[],status='SEPARATE_DENSE_WORKLOAD',fresh_process_status='NOT_APPLICABLE',
                scope='Dense-query and batch timings retained in performance.json; no duplicated complete-call or cold-start sweep'))
            _write(Path(ctx.path)/'endpoint_metrics.json',dict(rows=[],scope='Per-query accepted-teacher accuracy retained in performance.json'))
            ctx.record('cost/dense',ctx.unit.get('mechanism_ids',['M5']),metrics=dict(models=len(selected),panels=len(performance_rows),missing_models=len(missing)),
                checks=[check('paired-rounds',min(rounds) if rounds else None,settings.get('reuse_repeats',settings.get('repeats',20)),'eq',category='correctness'),
                        check('representative-checkpoint-coverage',len(missing),0,'eq',category='gap')],evidence=['timing.json','timing_coverage.json','performance.json'])
            return dict(workload='dense',models=len(selected),missing_models=missing,fresh_process='NOT_APPLICABLE')
        schedules=ctx.protocol.get('evaluation',{}).get('frozen_schedules',[1,2,4,8])
        predictors=scheduled_methods(selected,schedules)
        predictors.update(classical_methods(record,schedules))
        for name,method in predictors.items():
            resources[name]=dict(method=method.metadata['method_id'],family=method.metadata['family'],steps=method.metadata.get('steps'),**model_resources(method))
        _write(Path(ctx.path)/'model_resources.json',{'rows':[dict(model_id=name,**row) for name,row in resources.items()],
            'scope':'Static per-model counts and checkpoint/artifact size; allocator peaks measured separately per method in timing.json'})
        inp=inference_input({'u0':u,'record':record});calls={name:(lambda f=f:f(inp)) for name,f in predictors.items()}
        timing=paired_timing(calls,device=ctx.device,repeats=settings.get('repeats',20),warmup=settings.get('warmup',3),seed=ctx.protocol.get('root_seed',0),budget=ctx.budget)
        declarations=ctx.protocol.get('trials',{});declarations=list(declarations.values()) if isinstance(declarations,dict) else declarations
        expected=({f"{method_id}/seed-{seed}/steps-{count}" for method_id,seed in selection_filter for count in schedules} if selection_filter is not None else
            {f"{trial.get('method_id',trial['family']+'::'+trial.get('variant_id','base')+'::n'+str(trial.get('data_count',0)))}/seed-{trial['seed']}/steps-{count}"
            for trial in declarations if trial.get('target_kind')==record['target_kind'] and trial.get('phase') in ('main','data_efficiency') for count in schedules})
        expected.update(name for name in predictors if name.startswith('analytic_'))
        coverage.append(dict(parent_id=record['parent_id'],query_id=record['query_id'],expected_methods=sorted(expected or set(predictors)),
            measured_methods=sorted(predictors),missing_selected_methods=sorted(expected-set(predictors)),
            failed_methods={name:row['failure'] for name,row in timing['summaries'].items() if row['status']!='COMPLETED'},
            excluded_for_performance_selection=False,representative_only=True,phase_variant='original',architecture_members=ctx.unit.get('cost_aliases',[]),
            alias_timing_status='NOT_MEASURED; identical declared architecture does not establish identical weight-sensitive execution or accuracy',
            scope='Frozen architecture representatives, same 1/2/4/8 schedules. Other checkpoints retain their own accuracy but receive no imputed timing/frontier; missing representatives remain visible'))
        _write(Path(ctx.path)/'timing_coverage.json',{'rows':coverage})
        timing.update(parent_id=record['parent_id'],query_id=record['query_id'],grid=grid,target_kind=record['target_kind'],batch_size=u.shape[0],transfers=transfer_timing(u,ctx.device))
        panels.append(timing)
        with torch.no_grad():
            for name,call in calls.items():
                try:p=call()
                except (RuntimeError,ValueError,FloatingPointError):p=None
                method=predictors[name];meta=method.metadata;metric=endpoint_metrics(p,target,record=record)|{'method':meta['method_id'],'family':meta['family'],'model_id':name}
                endpoint.append(metric);summary=timing['summaries'][name]
                measured=dict(method=meta['method_id'],family=meta['family'],model_id=name,seed=meta.get('seed'),parent_id=record['parent_id'],query_id=record['query_id'],
                    target_kind=record['target_kind'],phase_variant=record.get('phase_variant','original'),grid=grid,batch_size=u.shape[0],median_seconds=summary['warm_median_seconds'],error_rms=metric['error_rms'],error_max=metric['error_max'],
                    horizon=record['horizon'],steps=meta.get('steps',1),rms=metric['rms'],norm=metric['norm'],uncertainty_rms=metric['uncertainty_rms'],uncertainty_maximum=metric['uncertainty_maximum'],
                    timing_status=summary['status'],measured_rounds=summary['measured_rounds'],requested_rounds=summary['requested_rounds'],
                    reference_accepted=record['reference']['accepted'],accuracy_qualification=metric['accuracy_qualification'],
                    first_call_seconds=summary['first_invocation_warm_process_seconds'],peak_allocated_bytes=(summary.get('memory') or {}).get('peak_allocated_bytes'),
                    peak_reserved_bytes=(summary.get('memory') or {}).get('peak_reserved_bytes'),raw_panel_index=panel_index,
                    fresh_process_seconds=None,energy_joules=None,monetary_cost=None,
                    unavailable_cost_reason='Energy/rate accounting not supplied; never inferred from CPU or GPU utilization')
                if panel_index==0 and settings.get('fresh_process',True) and meta.get('steps',1)==1:
                    ctx.budget.check()
                    from .fields import digest
                    probe=fresh_process_probe(method,inp,Path(ctx.path)/('fresh-'+digest(name)[:12]),ctx.device,
                        timeout=settings.get('fresh_process_timeout',120),expected=p)
                    measured['fresh_process']=probe
                    measured['fresh_process_seconds']=probe.get('parent_observed_total_seconds') if probe['status']=='COMPLETED' else None
                flat.append(measured)
                for raw in timing['raw_rounds']:
                    if raw['method']==name:raw.update(error_rms=metric['error_rms'],error_max=metric['error_max'],accuracy_qualification=metric['accuracy_qualification'])
        _write(Path(ctx.path)/'timing.json',{'rows':flat,'panels':panels,'raw_rounds':[r|{'panel':i} for i,p in enumerate(panels) for r in p['raw_rounds']],
            'fresh_process_status':'MEASURED' if settings.get('fresh_process',True) else 'NOT_MEASURED','offline_costs_not_imputed':True})
        frontier=accuracy_cost_frontiers(flat,ctx.protocol.get('evaluation',{}).get('tolerances',[1e-6,2e-6,1e-5,2e-5,1e-4,1e-3]),offline)
        _write(Path(ctx.path)/'accuracy_cost_frontiers.json',frontier)
        _write(Path(ctx.path)/'amortization.json',{'rows':frontier['amortization_rows'],'scope':frontier['scope']})
        if panel_index==0 and workload=='all':
            performance,performance_rows=_run_performance(ctx,sample,selected,groups[(record['parent_id'],record.get('phase_variant','original'),tuple(record['grid']))])
    _write(Path(ctx.path)/'endpoint_metrics.json',{'rows':endpoint})
    from .core import check
    ctx.record('cost/paired',ctx.unit.get('mechanism_ids',['M5']),metrics={'panels':len(panels),'methods':len(predictors)},
        checks=[check('paired-rounds',min(s['measured_rounds'] for p in panels for s in p['summaries'].values()),settings.get('repeats',20),'eq',category='correctness')],evidence=['timing.json','endpoint_metrics.json','performance.json'])
    return {'panels':len(panels),'accuracy_rows':len(endpoint),'fresh_process':'MEASURED' if settings.get('fresh_process',True) else 'NOT_MEASURED'}


if __name__=='__main__':
    import sys
    if len(sys.argv)!=3 or sys.argv[1]!='fresh':raise SystemExit('Only internal fresh <project-probe-dir> is supported')
    _fresh_child(sys.argv[2])
