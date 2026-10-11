"""Read-only finite-time decoder quadrature refinement on frozen checkpoints.

This measures sensitivity of a learned model's own integral. It is neither a
quadrature certificate nor evidence that the represented nonlinear source is
correct. Refinement/FP64 diagnostic work is excluded from deployment timings
and charged explicitly to this panel.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import time

import torch
from tdn.research.protocol import file_digest, digest
from tdn.runtime.metadata import write_json
from . import numerics as n
from .references import equation_geometry
from .metrics import endpoint_metrics
from .training import _state_hash

SCHEMA = 'tdn.response-quadrature-refinement/v1'


def _norms(value, geom, target):
    if not bool(torch.isfinite(value).all()):
        return dict(finite=False,physical_rms=None,nodal_rms=None,sampled_maximum=None,
                    mean=None,centered_physical_rms=None)
    x=value.double();v=n.center(x)
    return dict(finite=True,physical_rms=float(n.inner(x,x,geom,target).clamp_min(0).mean().sqrt()),
        nodal_rms=float(x.square().mean().sqrt()),sampled_maximum=float(x.abs().max()),
        mean=float(n.mean(x).mean()),centered_physical_rms=float(n.inner(v,v,geom,target).clamp_min(0).mean().sqrt()))


def _sync(device):
    if torch.device(device).type=='cuda':torch.cuda.synchronize(device)


def _declared_queries(ctx,samples):
    """First declared parent per target/grid; never substitute an available one."""
    groups={}
    for name in ctx.unit.get('data_stages',[]):
        unit=ctx.protocol.get('units',{}).get(name,{})
        target=unit.get('target_kind',ctx.unit.get('target_kind'))
        grid=unit.get('grid');parents=unit.get('parent_ids',[]);horizons=unit.get('horizons',[])
        if not target or grid is None or not parents or not horizons:continue
        grid=tuple([grid]*ctx.protocol.get('dimension',2)) if isinstance(grid,int) else tuple(grid)
        key=(target,grid)
        if key not in groups:
            groups[key]=dict(target_kind=target,grid=list(grid),parent_id=parents[0],horizon=max(horizons),selection='protocol declaration')
        else:groups[key]['horizon']=max(groups[key]['horizon'],max(horizons))
    if not groups:
        # Only useful for callers without a workflow protocol. The distinction
        # from a frozen protocol declaration is retained in every query row.
        for sample in samples:
            r=sample['record'];key=(r['target_kind'],tuple(r['grid']))
            if key not in groups:
                groups[key]={k:r[k] for k in ('target_kind','grid','parent_id','horizon')}
                groups[key]['selection']='first supplied parent; no protocol query declaration available'
            groups[key]['horizon']=max(groups[key]['horizon'],r['horizon'])
    result=[]
    for key,query in sorted(groups.items()):
        matches=[s for s in samples if s['record']['target_kind']==key[0] and tuple(s['record']['grid'])==key[1]
                 and s['record']['parent_id']==query['parent_id'] and abs(s['record']['horizon']-query['horizon'])<=1e-12
                 and s['record'].get('phase_variant','original')=='original'
                 and s['record'].get('intervention_variant','original')=='original']
        result.append((query,matches[0] if len(matches)==1 else None,
                       None if len(matches)==1 else 'Declared original parent/query is missing or ambiguous; no replacement'))
    return result


def _representatives(methods):
    eligible=[(name,m) for name,m in methods.items() if getattr(getattr(m,'model',None),'family',None)=='coupled_source']
    eligible.sort(key=lambda pair:(getattr(pair[1],'metadata',{}).get('variant_id') not in ('primary','base'),
                   getattr(pair[1],'metadata',{}).get('seed',0) or 0,pair[0]))
    seen={};selected=[];aliases=[]
    for name,method in eligible:
        config=dict(method.model.config)
        config['family']=method.model.family
        key=digest(config)
        if key in seen:
            aliases.append(dict(model_id=name,representative_model_id=seen[key],configuration_sha256=key,
                status='NOT_MEASURED',reason='Same inference configuration; different learned weights are not assumed to have the representative result'))
        else:
            seen[key]=name;selected.append((name,method,key))
    return selected,aliases


def run(ctx,samples,methods):
    """Write a bounded, incremental selected-model quadrature diagnostic."""
    root=Path(ctx.path);root.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();rows=[];rounding=[]
    settings=ctx.protocol.get('response',{}).get('quadrature_audit',{})
    seconds=float(settings.get('seconds',60.))
    if not math.isfinite(seconds) or not 0<=seconds<=60:
        raise ValueError('Quadrature audit has a declared 0..60 second diagnostic cap')
    selected,aliases=_representatives(methods);queries=_declared_queries(ctx,samples)
    source_files={f.name:file_digest(f) for f in [Path(__file__),*[Path(__file__).with_name(name) for name in ('models.py','features.py','numerics.py','kernels.py','trajectory.py')]]}
    def check():
        ctx.budget.check()
        if time.perf_counter()-start>=seconds:raise TimeoutError('Bounded quadrature audit allowance exhausted')
    def save():
        result=dict(schema=SCHEMA,rows=rows,arithmetic_comparisons=rounding,aliases=aliases,
            selected_models=[name for name,_,_ in selected],declared_queries=[q for q,_,_ in queries],
            source_files_sha256=source_files,elapsed_seconds=time.perf_counter()-start,allowance_seconds=seconds,
            cost_scope='All copying, encoding, quadrature, synchronization, metrics and I/O are diagnostic campaign cost; excluded from deployment measurements',
            precision_scope='FP64 arithmetic applied to the identical learned numeric weights and exact values of deployed input; FP64 changes execution arithmetic, not training or selection',
            interpretation='Observed quadrature differences, not error bounds or certificates; trained-source representation error and reference uncertainty remain separate',
            selection='First declared original parent per target/grid at maximum declared horizon; primary first then one checkpoint per distinct inference configuration; no outcome-informed substitutions',
            status='NOT_MEASURED' if not selected or not queries else 'PARTIAL' if any(r['status']!='COMPLETED' for r in rows) else 'COMPLETED')
        write_json(root/'quadrature_refinement.json',result)
        return result
    if not selected or not queries:
        rows.append(dict(status='NOT_MEASURED',reason='No selected coupled-source checkpoint' if not selected else 'No declared parent/query'))
        return save()
    for name,method,configuration in selected:
        original=method.model;metadata=getattr(method,'metadata',{})
        original_state=_state_hash(original.state_dict());original_training=original.training
        checkpoint=Path(metadata['checkpoint_path']) if metadata.get('checkpoint_path') else None
        if checkpoint is not None and checkpoint.is_dir():checkpoint=checkpoint/'selected.pt'
        checkpoint_sha256=file_digest(checkpoint) if checkpoint is not None and checkpoint.is_file() else None
        for query,sample,missing in queries:
            plan=[]
            for pair in [(original.nodes,original.mean_nodes),(8,4),(4,8),(8,8),(16,16),(32,32)]:
                if pair not in plan:plan.append(pair)
            common=dict(model_id=name,method=metadata.get('method_id',name),method_id=metadata.get('method_id',name),seed=metadata.get('seed'),variant_id=metadata.get('variant_id'),
                checkpoint_path=metadata.get('checkpoint_path'),checkpoint_sha256=checkpoint_sha256,selection_sha256=metadata.get('selection_sha256'),
                parameter_state_sha256=original_state,configuration_sha256=configuration,
                source_decoder=original.source_decoder,source_representation=original.source_representation,
                mean_parameterization=original.mean_parameterization,source_feedback=original.source_feedback,
                query_id=None if sample is None else sample['record'].get('query_id'),
                input_sha256=None if sample is None else sample['record'].get('input_sha256'),**query)
            completed=[];clone=None;encoded=None;base_prediction=None;deploy_prediction=None;preparation_status=None
            if sample is not None:
                try:
                    check();eq,geom=equation_geometry(sample['record'])
                    # Never change original mode, dtype, nodes, cached encodings
                    # or parameter versions. Native and FP64 copies are separate.
                    prep_start=time.perf_counter();deployment=deepcopy(original).eval()
                    dtype=next(deployment.parameters()).dtype
                    with torch.no_grad():
                        deploy_prediction=deployment(sample['u0'].to(dtype=dtype),query['horizon'],eq,geom,query['target_kind'])
                    _sync(deploy_prediction.device)
                    rounding.append(dict(**common,status='DEPLOYED_MEASURED_FP64_BASE_PENDING',deployed_dtype=str(deploy_prediction.dtype),
                        source_nodes=original.nodes,mean_nodes=original.mean_nodes,elapsed_seconds=time.perf_counter()-prep_start))
                    clone=deepcopy(original).double().eval()
                    original_mean_path=clone.mean_path
                    def bounded_mean_path(encoded,s):
                        check();return original_mean_path(encoded,s)
                    clone.mean_path=bounded_mean_path
                    with torch.no_grad():encoded=clone.encode(sample['u0'].double(),eq,geom,query['target_kind'])
                except (TimeoutError,InterruptedError) as exc:missing=str(exc);preparation_status='CENSORED'
                except (ValueError,RuntimeError,FloatingPointError) as exc:missing=f'{type(exc).__name__}: {exc}';preparation_status='FAILED'
            for source_nodes,mean_nodes in plan:
                iteration=time.perf_counter()
                row=dict(**common,source_nodes=source_nodes,mean_nodes=mean_nodes,dtype='torch.float64',
                    variant=f'source-{source_nodes}/mean-{mean_nodes}',difference_rms_vs_finest=None,difference_maximum_vs_finest=None,
                    baseline_source_nodes=original.nodes,baseline_mean_nodes=original.mean_nodes,
                    source_nodes_changed=source_nodes!=original.nodes,mean_nodes_changed=mean_nodes!=original.mean_nodes)
                try:
                    if sample is None:row.update(status='NOT_MEASURED',reason=missing)
                    elif clone is None or encoded is None:
                        row.update(status=preparation_status or 'FAILED',reason=missing)
                    else:
                        check();clone.nodes=source_nodes;clone.mean_nodes=mean_nodes
                        _sync(encoded.u.device)
                        with torch.no_grad():prediction=clone.decode(encoded,query['horizon'])
                        _sync(prediction.device)
                        if not bool(torch.isfinite(prediction).all()):raise FloatingPointError('Nonfinite refined model prediction')
                        if base_prediction is None:base_prediction=prediction
                        row.update(status='COMPLETED',difference_from_fp64_baseline=_norms(prediction-base_prediction,geom,query['target_kind']),
                            mean_prediction=float(n.mean(prediction).mean()),
                            reference_accepted=bool(sample['record'].get('reference',{}).get('accepted',False)))
                        if row['reference_accepted'] and sample.get('target') is not None:
                            row['reference_error']=endpoint_metrics(prediction,sample['target'],u0=sample['u0'],record=sample['record'])
                        else:row.update(reference_error=None,reference_error_status='NOT_MEASURED: accepted reference unavailable')
                        completed.append((row,prediction))
                        if (source_nodes,mean_nodes)==plan[0] and deploy_prediction is not None:
                            rounding[-1].update(status='COMPLETED',fp64_baseline_dtype=str(prediction.dtype),
                                deployed_minus_fp64_baseline=_norms(deploy_prediction.double()-prediction,geom,query['target_kind']))
                            rounding[-1]['difference_rms']=rounding[-1]['deployed_minus_fp64_baseline']['physical_rms']
                            rounding[-1]['difference_maximum']=rounding[-1]['deployed_minus_fp64_baseline']['sampled_maximum']
                except (TimeoutError,InterruptedError) as exc:row.update(status='CENSORED',reason=str(exc))
                except (ValueError,RuntimeError,FloatingPointError) as exc:row.update(status='FAILED',reason=f'{type(exc).__name__}: {exc}')
                row['elapsed_seconds']=time.perf_counter()-iteration
                rows.append(row);save()
            joint=[(row,p) for row,p in completed if row['source_nodes']==row['mean_nodes']]
            if joint:
                finest,field=max(joint,key=lambda pair:pair[0]['source_nodes'])
                for row,prediction in completed:
                    row['finest_completed_joint_nodes']=finest['source_nodes']
                    row['difference_from_finest_completed_joint_rule']=_norms(prediction-field,geom,query['target_kind'])
                    row['difference_rms_vs_finest']=row['difference_from_finest_completed_joint_rule']['physical_rms']
                    row['difference_maximum_vs_finest']=row['difference_from_finest_completed_joint_rule']['sampled_maximum']
                    row['comparison_scope']='Finite highest completed joint rule, not truth and not a certified quadrature error'
            if _state_hash(original.state_dict())!=original_state or original.training!=original_training:
                raise ValueError('Quadrature audit mutated the selected original model')
            save()
    return save()
