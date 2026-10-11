"""Amortized-query and batching experiments, with separate profiling probes.

Reuse is available to classical methods too. The comparison includes a cached
initial Fourier transform for Strang and a fixed-mesh Lawson trajectory with
cubic Hermite dense output. Dense output is an approximation and its off-mesh
error/refinement are reported; matching a neural query count alone never makes
two methods accuracy-equivalent.
"""
from __future__ import annotations
import math
import time
import torch
from . import numerics as n
from .contracts import Physics


def _version(tensor):
    try:
        return tensor._version
    except RuntimeError:
        return None


class CachedDF:
    """Exact deployed DF arithmetic with reusable initial FFT and symbol.

    The cache belongs to one initial state, geometry, target and physics. It
    must be rebuilt after an accepted rollout step; this is not a trajectory
    solver that can reuse the old initial state's transform indefinitely.
    """
    def __init__(self, u, eq, geom, target, *, reaction_substeps=8):
        n.validate(u, 0., geom, target)
        self.original, self.version, self.u = u, _version(u), u.clone()
        self.eq, self.geom, self.target = eq, geom, target
        self.physics = [(x, _version(x), x.clone()) for x in (eq.kappa, eq.reaction_rate) if torch.is_tensor(x)]
        self.reaction_substeps = reaction_substeps
        self.spectrum = torch.fft.fftn(self.u, dim=n.axes(u))
        self.symbol = n.diffusion_symbol(self.u, eq, geom, target)

    def _check(self):
        if self.version != _version(self.original) or (self.version is None and not torch.equal(self.original, self.u)):
            raise ValueError('Stale classical cache: initial state changed')
        if any(version != _version(x) or (version is None and not torch.equal(x, snapshot)) for x, version, snapshot in self.physics):
            raise ValueError('Stale classical cache: physics changed')

    def decode(self, h):
        self._check()
        step = n.validate(self.u, h, self.geom, self.target)
        propagator = torch.exp(step*self.symbol/2)
        half = torch.fft.ifftn(self.spectrum*propagator, dim=n.axes(self.u)).real
        reacted = n.reaction(half, step, self.eq, self.geom, self.target, substeps=self.reaction_substeps)
        return torch.fft.ifftn(torch.fft.fftn(reacted, dim=n.axes(self.u))*propagator, dim=n.axes(self.u)).real


class FixedMeshDenseOutput:
    """Fourth-order Lawson mesh with endpoint-RHS cubic Hermite interpolation.

    This is a simple explicit numerical comparator, not an authoritative
    library implementation or a uniformly stiff-order claim. The mesh count
    is fixed before observing the query error. Interpolation can overshoot and
    no output clipping or positivity certificate is applied.
    """
    def __init__(self, u, horizon, eq, geom, target, *, steps=32):
        if type(steps) is not int or steps < 1:
            raise ValueError('Dense output needs a positive fixed mesh count')
        H = torch.as_tensor(horizon, dtype=u.dtype, device=u.device)
        if H.ndim != 0 or not bool(torch.isfinite(H)) or not bool(H > 0):
            raise ValueError('Dense output horizon must be a finite positive scalar')
        n.validate(u, H, geom, target)
        self.original, self.version, self.u = u, _version(u), u.clone()
        self.H, self.steps, self.geom, self.eq, self.target = H, steps, geom, eq, target
        self.physics = [(x, _version(x), x.clone()) for x in (eq.kappa, eq.reaction_rate) if torch.is_tensor(x)]
        self.dt = H/steps
        states, slopes = [self.u], [n.rhs(self.u, eq, geom, target)]
        for _ in range(steps):
            value = n.lawson_step(states[-1], self.dt, eq, geom, target)
            states.append(value); slopes.append(n.rhs(value, eq, geom, target))
        self.states, self.slopes = torch.stack(states), torch.stack(slopes)

    def _single(self, h):
        if self.version != _version(self.original) or (self.version is None and not torch.equal(self.u, self.original)):
            raise ValueError('Stale dense trajectory: initial state changed')
        if any(version != _version(x) or (version is None and not torch.equal(x, snapshot)) for x, version, snapshot in self.physics):
            raise ValueError('Stale dense trajectory: physics changed')
        step = n.validate(self.u, h, self.geom, self.target)
        if bool((step > self.H).any()):
            raise ValueError('Dense query exceeds the encoded time interval')
        time = step.expand(self.u.shape[0], 1, *([1]*self.geom.ndim)).flatten()
        index = torch.floor(time/self.dt).long().clamp(max=self.steps-1)
        fraction = ((time-index*self.dt)/self.dt).reshape(self.u.shape[0], 1, *([1]*self.geom.ndim))
        batch = torch.arange(self.u.shape[0], device=self.u.device)
        left, right = self.states[index, batch], self.states[index+1, batch]
        fleft, fright = self.slopes[index, batch], self.slopes[index+1, batch]
        t = fraction
        return (2*t**3-3*t**2+1)*left+(t**3-2*t**2+t)*self.dt*fleft+(-2*t**3+3*t**2)*right+(t**3-t**2)*self.dt*fright

    def decode(self, horizons):
        value = torch.as_tensor(horizons, dtype=self.u.dtype, device=self.u.device)
        if value.ndim == 2:
            if value.shape[0] != self.u.shape[0] or value.shape[1] < 1:
                raise ValueError('Dense query table must have shape [B,Q]')
            return torch.stack([self._single(value[:, j]) for j in range(value.shape[1])], dim=1)
        if value.ndim > 1 or (value.ndim == 1 and value.numel() != self.u.shape[0]):
            raise ValueError('Dense horizon must be scalar, [B] or [B,Q]')
        return self._single(value)


def profile_prediction(call, *, device, budget=None):
    """Operator attribution from a separate instrumented invocation.

    These times are profiler observations, never deployment latency. Primitive
    multiplication counts do not identify physical quadratic products versus
    learned pointwise algebra; allocation bytes are gross events, not a peak.
    """
    if budget is not None:
        budget.check()
    cuda = torch.device(device).type == 'cuda'
    activities = [torch.profiler.ProfilerActivity.CPU]
    if cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    start = time.perf_counter()
    try:
        with torch.no_grad(), torch.profiler.profile(activities=activities,
                record_shapes=True, profile_memory=True) as profile:
            value = call()
            if cuda:
                torch.cuda.synchronize(device)
        rows = []
        for event in profile.key_averages():
            key = event.key
            if '_fft' in key:
                category = 'transform'
            elif 'conv' in key or 'mm' in key or 'linear' in key:
                category = 'learned_linear_and_convolution'
            elif any(token in key for token in ('empty', 'clone', 'copy_', 'to', 'contiguous')):
                category = 'allocation_or_transfer'
            elif any(token in key for token in ('synchronize', '_local_scalar_dense', 'item')):
                category = 'synchronization_or_host_scalar'
            elif key.startswith('aten::'):
                category = 'pointwise_and_other_aten'
            else:
                category = 'runtime_other'
            rows.append(dict(operator=key, category=category, calls=event.count,
                self_cpu_seconds=event.self_cpu_time_total/1e6,
                self_device_seconds=getattr(event, 'self_device_time_total', 0.)/1e6 if cuda else None,
                self_cpu_memory_bytes=event.self_cpu_memory_usage,
                self_device_memory_bytes=getattr(event, 'self_device_memory_usage', None) if cuda else None))
        return dict(status='MEASURED_SEPARATE_INSTRUMENTED_CALL', elapsed_wall_seconds=time.perf_counter()-start,
            finite=bool(torch.isfinite(value).all()), rows=rows,
            deployment_latency=False, physical_product_count=None,
            limitations='Self-event attribution only; no exact transform FLOPs, physical-product isolation, peak RSS, or joules inferred')
    except (RuntimeError, ValueError) as exc:
        return dict(status='UNAVAILABLE', reason=f'{type(exc).__name__}: {exc}', rows=[], deployment_latency=False)


def _repeat_physics(eq, count, original_batch):
    def repeat(value):
        if not torch.is_tensor(value) or value.numel() == 1:
            return value
        if value.numel() != original_batch:
            raise ValueError('Physics must be scalar or one coefficient per input field')
        return value.reshape(original_batch)[:1].expand(count).clone()
    return Physics(repeat(eq.kappa), repeat(eq.reaction_rate))


def _differences(left, right):
    if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
        return dict(maximum=None, rms=None, finite=False,
                    unavailable_reason='Nonfinite output; absent error is not a zero')
    difference = (left.double()-right.double())
    return dict(maximum=float(difference.abs().max()), rms=float(difference.square().mean().sqrt()),
                finite=True)


def measure_model_workloads(model, u, h, eq, geom, target, *, repeats=2, warmup=1,
        query_counts=(1, 2, 4, 8, 16), batch_sizes=(1, 4, 16), target_teacher=None,
        query_teachers=None, budget=None, seed=0, profile=True, dense_steps=32):
    """Bounded detailed cost panel for one model and one resident input state.

    Batches replicate this field to measure throughput and launch overhead;
    they do not create independent accuracy observations. Query timing includes
    encoding for the shared-encode arms; the decode-only arm is separately
    labelled with its excluded preparation. Missing accuracy remains missing.
    """
    from .cost import paired_timing
    if u.shape[0] != 1:
        raise ValueError('Cost workload panel requires one identified representative field')
    if any(type(q) is not int or q < 1 for q in query_counts) or any(type(b) is not int or b < 1 for b in batch_sizes):
        raise ValueError('Query and batch counts must be positive integers')
    if not math.isfinite(float(h)) or float(h) <= 0:
        raise ValueError('Cost workload horizon must be positive')
    result = dict(schema='tdn.response-workloads/v1', model=model.architecture_metadata(),
        target=target, horizon=float(h), query_counts=list(query_counts), batch_sizes=list(batch_sizes),
        reuse_panels=[], batch_panels=[], raw_rounds=[], status='COMPLETED',
        interpretation='Timing study only; no matched-accuracy solver advantage is inferred',
        requested_repeats=repeats, warmup=warmup, dense_steps=dense_steps)
    def check():
        if budget is not None:
            budget.check()
    try:
        with torch.no_grad():
            check()
            encoded = model.encode(u, eq, geom, target)
            classical = CachedDF(u, eq, geom, target, reaction_substeps=model.reaction_substeps)
            # Preparation is done once for the decode-only timing arm but all
            # end-to-end shared arms below reconstruct their own cache.
            dense = FixedMeshDenseOutput(u, h, eq, geom, target, steps=dense_steps)
            finer = FixedMeshDenseOutput(u, h, eq, geom, target, steps=2*dense_steps)
            for count in query_counts:
                check()
                # Include off-mesh points and an exact final endpoint. This
                # actually tests interpolation instead of only reading nodes.
                # Canonical FP64 requested times match CPU reference-bank
                # declarations exactly; deployed operators cast their own
                # scalar inputs according to the actual precision contract.
                times = torch.tensor([float(h)*(j/count)**1.3 for j in range(1,count+1)], device=u.device, dtype=torch.float64)
                table = times[None]
                repeated = lambda: torch.stack([model(u, t, eq, geom, target) for t in times], dim=1)
                shared = lambda: model.decode(model.encode(u, eq, geom, target), table)
                df_repeated = lambda: torch.stack([n.df(u, t, eq, geom, target,
                    reaction_substeps=model.reaction_substeps) for t in times], dim=1)
                def df_shared():
                    cache = CachedDF(u, eq, geom, target, reaction_substeps=model.reaction_substeps)
                    return torch.stack([cache.decode(t) for t in times], dim=1)
                def classical_independent():
                    values = []
                    for t in times:
                        # Matching maximum nominal mesh width, not matched
                        # observed accuracy or a tuned adaptive tolerance.
                        steps = max(1, int(math.ceil(dense_steps*float(t)/float(h))))
                        values.append(n.integrate(u, t, eq, geom, target, substeps=steps, method='lawson'))
                    return torch.stack(values, dim=1)
                calls = dict(model_repeated_full=repeated, model_shared_encoding=shared,
                    model_decode_only=lambda:model.decode(encoded,table),
                    df_repeated_full=df_repeated, df_shared_initial_fft=df_shared,
                    df_decode_only=lambda:torch.stack([classical.decode(t) for t in times],dim=1),
                    classical_independent_lawson=classical_independent,
                    classical_shared_dense=lambda:FixedMeshDenseOutput(u,h,eq,geom,target,steps=dense_steps).decode(table),
                    classical_dense_decode_only=lambda:dense.decode(table))
                timing = paired_timing(calls, device=u.device, repeats=repeats,
                    warmup=warmup, seed=seed+count, budget=budget)
                check()
                a, b = repeated(), shared()
                df_a, df_b = df_repeated(), df_shared()
                interp, refined = dense.decode(table), finer.decode(table)
                panel = dict(query_count=count, times=times.cpu().tolist(), timing=timing,
                    reuse_parity=_differences(a,b), df_reuse_parity=_differences(df_a,df_b),
                    dense_refinement=_differences(interp,refined),
                    dense_refinement_scope='Observed 2x fixed-mesh difference, not a rigorous uncertainty certificate',
                    intermediate_query_accuracy='NOT_MEASURED against accepted reference teachers',
                    final_endpoint_error=_differences(a[:,-1],target_teacher.to(u.device)) if target_teacher is not None else None,
                    decode_only_excludes='Initial state encoding or full dense-trajectory preparation',
                    shared_encoding_includes='A newly built initial encoding per measured multi-query call')
                accuracy=[]
                if query_teachers is not None:
                    from .metrics import endpoint_metrics
                    values=dict(model_repeated_full=a,model_shared_encoding=b,
                        df_repeated_full=df_a,df_shared_initial_fft=df_b,
                        classical_shared_dense=interp,classical_independent_lawson=classical_independent())
                    for index, requested in enumerate(times.cpu().tolist()):
                        check()
                        matches=[sample for sample in query_teachers
                            if sample['record']['target_kind']==target
                            and abs(float(sample['record']['horizon'])-requested)<=1e-12
                            and tuple(sample['u0'].shape)==tuple(u.shape)
                            and torch.equal(sample['u0'].to(u.device),u)]
                        if len(matches)!=1:
                            accuracy.extend(dict(arm=name,time=requested,status='NOT_MEASURED',
                                accuracy_qualification='NOT_MEASURED',reference_accepted=False,
                                error_rms=None,error_max=None,
                                unavailable_reason='Missing or ambiguous same-input CPU reference-bank query') for name in values)
                            continue
                        sample=matches[0]
                        for name, prediction in values.items():
                            metrics=endpoint_metrics(prediction[:,index],sample['target'].to(u.device),
                                record=sample['record'],u0=u)
                            accuracy.append(dict(arm=name,time=requested,**metrics))
                    panel['intermediate_query_accuracy']='PRECOMPUTED_CPU_REFERENCE_BANK; missing and unresolved queries explicit'
                    panel['all_queries_qualified']={name:all(row['accuracy_qualification']=='QUALIFIED'
                        for row in accuracy if row['arm']==name) and sum(row['arm']==name for row in accuracy)==count for name in values}
                else:
                    panel['all_queries_qualified']={}
                panel['query_accuracy']=accuracy
                tolerance=32*torch.finfo(u.dtype).eps*max(1.,float(a.abs().max()))
                panel['reuse_parity_tolerance']=tolerance
                panel['reuse_parity_passed']=(panel['reuse_parity']['finite'] and panel['reuse_parity']['maximum']<=tolerance)
                panel['reuse_parity_scope']='Dtype-scaled arithmetic equivalence only; not matched physical accuracy'
                full = timing['summaries']['model_repeated_full']['warm_median_seconds']
                cached = timing['summaries']['model_shared_encoding']['warm_median_seconds']
                panel['observed_reuse_speedup'] = full/cached if full is not None and cached is not None and cached>0 else None
                complete=all(timing['summaries'][name]['status']=='COMPLETED' and timing['summaries'][name]['measured_rounds']==repeats
                    for name in ('model_repeated_full','model_shared_encoding'))
                panel['observed_reuse_helps'] = bool(complete and panel['reuse_parity_passed'] and full is not None and cached is not None and full>cached)
                result['reuse_panels'].append(panel)
                result['raw_rounds'].extend([dict(panel='queries',query_count=count,**r) for r in timing['raw_rounds']])
            for batch in batch_sizes:
                check()
                value = u.expand(batch,*u.shape[1:]).clone()
                physics = _repeat_physics(eq,batch,u.shape[0])
                calls = dict(batched=lambda:model(value,h,physics,geom,target),
                    sequential=lambda:torch.cat([model(value[i:i+1],h,eq,geom,target) for i in range(batch)]))
                timing = paired_timing(calls,device=u.device,repeats=repeats,warmup=warmup,seed=seed+100+batch,budget=budget)
                check(); pred=calls['batched'](); serial=calls['sequential']()
                seconds=timing['summaries']['batched']['warm_median_seconds']
                panel=dict(batch_size=batch,timing=timing,parity=_differences(pred,serial),
                    fields_per_second=batch/seconds if seconds is not None and seconds>0 else None,
                    source_fields=1,replicated_inputs=True,independent_accuracy_sample_count=1,
                    final_endpoint_error=_differences(pred[:1],target_teacher.to(u.device)) if target_teacher is not None else None)
                result['batch_panels'].append(panel)
                result['raw_rounds'].extend([dict(panel='batch',batch_size=batch,**r) for r in timing['raw_rounds']])
            result['profiler'] = profile_prediction(lambda:model(u,h,eq,geom,target),device=u.device,budget=budget) if profile else dict(status='NOT_REQUESTED',rows=[])
    except TimeoutError as exc:
        result['status']='PARTIAL_BUDGET';result['interruption']=str(exc)
        result.setdefault('profiler',dict(status='NOT_RUN_BUDGET',rows=[]))
    eligible=[p['query_count'] for p in result['reuse_panels'] if p['observed_reuse_helps'] and p['reuse_parity']['finite']]
    result['first_observed_helpful_query_count']=min(eligible) if eligible else None
    result['break_even_interpretation']='Observed discrete counts only, without confidence interval; not a universal break-even or cross-method accuracy-cost advantage. Per-arm accuracy uses separately accepted query teachers when supplied.'
    return result
