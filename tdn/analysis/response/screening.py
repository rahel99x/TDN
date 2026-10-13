"""Metadata-only teacher and output-representation screening for new protocols.

This is an experimental-design screen, not a replacement scientific gate.
Selection uses only predeclared candidate order, accepted reference metadata and
precision diagnostics. No learned result, checkpoint or prediction array is read.
All requested candidate queries and every rejection remain in the artifact.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from .references import TARGETS, _write

SCHEMA = 'tdn.response-screening/v1'
FIT_REGIMES = ('low_mean','high_mean','large_amplitude','strong_diffusion','strong_reaction','phase_twin')


def _finite_nonnegative(value):
    if isinstance(value,bool):return None
    try:value=float(value)
    except (TypeError,ValueError):return None
    return value if math.isfinite(value) and value>=0 else None


def _positive(value,name):
    result=_finite_nonnegative(value)
    if result is None or result<=0:raise ValueError(f'{name} must be positive and finite')
    return result


def _signature(row):
    grid=row['grid']
    return (tuple(grid),float(row['horizon']),row.get('phase_variant','original'),
        row.get('intervention_variant','original'))


def _menu(protocol,config,split):
    explicit=config.get('query_menu',{}).get(split,{})
    dimension=protocol.get('dimension',2)
    grids=explicit.get('grids',[protocol.get('train_grid',protocol.get('grids',[8])[0])])
    normalized=[list([g]*dimension if isinstance(g,int) else g) for g in grids]
    horizons=explicit.get('horizons',protocol.get(split+'_horizons',protocol.get('horizons')))
    if not horizons or not normalized:raise ValueError('Screening requires a complete requested horizon/grid menu')
    if any(not math.isfinite(float(h)) or float(h)<=0 for h in horizons):
        raise ValueError('Screening fit horizons must be positive and finite; null controls remain separate')
    if any(len(g)!=dimension or any(type(n) is not int or n<2 for n in g) for g in normalized):
        raise ValueError('Invalid screening grids')
    phases=explicit.get('phase_variants',protocol.get('phase_variants',['original']))
    interventions=explicit.get('intervention_variants',['original'])
    if not phases or not interventions:raise ValueError('Screening requested query menu cannot be empty')
    rows=[dict(grid=g,horizon=float(h),phase_variant=v,intervention_variant=i)
        for g in normalized for h in horizons for v in phases for i in interventions]
    if len({_signature(r) for r in rows})!=len(rows):raise ValueError('Duplicate screening query specifications')
    return rows


def _component(record,name):
    precision=record.get('precision',{})
    value=precision.get('physical_norm_components',{}).get(name)
    # The FD physical and nodal norms coincide. For Galerkin the nearest
    # coordinatewise FP32 output remains only a physical-norm benchmark.
    if value is None and record.get('target_kind')=='fd_same_grid':value=precision.get(name)
    return {key:_finite_nonnegative((value or {}).get(key)) for key in ('rms','maximum')}


def _floor(records,minimum):
    value=minimum;components=dict(fixed_minimum=minimum,reference_uncertainty=0.,anchor_roundoff=0.,half_fp32_epsilon_at_state_scale=0.)
    for r in records:
        if not r.get('reference',{}).get('accepted'):continue
        ref=r['reference'];precision=r.get('precision',{})
        uncertainty=max(_finite_nonnegative(ref.get('uncertainty_rms')) or 0.,
            _finite_nonnegative(ref.get('uncertainty_maximum')) or 0.)
        roundoff=_finite_nonnegative(precision.get('anchor_difference',{}).get('maximum')) or 0.
        state=_finite_nonnegative(precision.get('state_scale'))
        if state is None:state=max(1.,_finite_nonnegative(precision.get('target_absolute_maximum')) or 1.)
        resolution=.5*torch.finfo(torch.float32).eps*max(1.,state)
        components['reference_uncertainty']=max(components['reference_uncertainty'],uncertainty)
        components['anchor_roundoff']=max(components['anchor_roundoff'],roundoff)
        components['half_fp32_epsilon_at_state_scale']=max(components['half_fp32_epsilon_at_state_scale'],resolution)
        value=max(value,uncertainty,roundoff,resolution)
    return value,components


def _query_screen(record,requested,floor,margin,reduction,tolerance):
    base=dict(requested,query_id=None,recorded=False,eligible=False,unresolved_reasons=[])
    if record is None:
        base['unresolved_reasons']=['MISSING_QUERY_RECORD'];return base
    ref=record.get('reference',{});precision=record.get('precision',{})
    deployment=_component(record,'deployment_defect');physical=_component(record,'physical_defect')
    quantization=_component(record,'teacher_output_rounding')
    assembly=_component(record,'assembly_rounding_floor')
    uncertainty={k:_finite_nonnegative(ref.get('uncertainty_'+k)) for k in ('rms','maximum')}
    base.update(query_id=record.get('query_id'),recorded=True,reference_accepted=bool(ref.get('accepted',False)),
        reference_status=ref.get('status','NOT_RECORDED'),reference_reason=ref.get('reason'),
        reference_metadata=ref,
        reference_uncertainty=uncertainty,normalization_floor=floor,
        precision_components=dict(deployment_defect=deployment,physical_defect=physical,
            nearest_fp32_output=quantization,actual_fp32_assembly=assembly,
            anchor_roundoff=precision.get('anchor_difference'),
            input_quantization=precision.get('input_quantization'),
            subtraction_roundoff=precision.get('naive_fp32_subtraction_error')),
        reference_component_error_budget=ref.get('component_error_budget'),
        reference_resolution_attempts=ref.get('resolution_attempts',ref.get('temporal_refinements',[])),
        output_quantization_scope=('Exact coordinatewise nodal FP32 representability optimum against recorded teacher'
            if record['target_kind']=='fd_same_grid' else
            'Coordinatewise nearest FP32 benchmark; not a rigorous global physical-L2 optimum'),
        physical_norm='nodal_rms' if record['target_kind']=='fd_same_grid' else 'trigonometric_interpolant_rms')
    reasons=base['unresolved_reasons']
    if not base['reference_accepted']:reasons.append('UNRESOLVED_REFERENCE')
    numbers=[deployment['rms'],physical['rms'],quantization['rms'],quantization['maximum'],
        assembly['rms'],assembly['maximum'],uncertainty['rms'],uncertainty['maximum']]
    if any(v is None for v in numbers):
        reasons.append('MISSING_OR_NONFINITE_PRECISION_METADATA');return base
    signal=max(0.,deployment['rms']-uncertainty['rms']);resolution=quantization['rms']+uncertainty['rms']
    nearest_mse=quantization['rms']**2;initial_mse=deployment['rms']**2
    base.update(conservative_signal_rms=signal,conservative_representation_rms=resolution,
        teacher_defect_above_floor=deployment['rms']>floor,
        rms_headroom_ratio=signal/resolution if resolution>0 else None,
        headroom_ratio_status='FINITE' if resolution>0 else 'ZERO_RECORDED_QUANTIZATION_AND_UNCERTAINTY',
        nearest_fp32_mse_reduction_benchmark=initial_mse/nearest_mse if nearest_mse>0 else None,
        mse_reduction_benchmark_status='FINITE' if nearest_mse>0 else 'ZERO_RECORDED_QUANTIZATION',
        required_mse_reduction=reduction,minimum_rms_margin=margin,
        output_maximum_with_uncertainty=quantization['maximum']+uncertainty['maximum'])
    if deployment['rms']<=floor:reasons.append('NORMALIZATION_FLOOR_LIMITED')
    if signal<margin*resolution:reasons.append('INSUFFICIENT_CONSERVATIVE_RMS_HEADROOM')
    if initial_mse<reduction*nearest_mse:reasons.append('INSUFFICIENT_FP32_MSE_REDUCTION_BENCHMARK')
    if quantization['maximum']+uncertainty['maximum']>tolerance:reasons.append('FP32_OUTPUT_MAXIMUM_EXCEEDS_TARGET')
    base['eligible']=not reasons
    return base


def screen_candidates(protocol,manifests,*,budget=None):
    """Evaluate every declared candidate, then select the first eligible one."""
    config=protocol.get('screening',{});declared=config.get('candidate_parents')
    if not isinstance(declared,dict) or not declared:raise ValueError('Explicit ordered screening candidates required')
    margin=_positive(config.get('minimum_rms_margin',20.),'Minimum RMS margin')
    reduction=_positive(config.get('required_mse_reduction',100.),'Required MSE reduction')
    minimum=_positive(config.get('minimum_floor',1e-8),'Minimum normalization floor')
    tolerance=_positive(config.get('accuracy_tolerance',protocol.get('tolerance',2e-5)),'Accuracy tolerance')
    required_regimes=config.get('required_regimes',list(FIT_REGIMES))
    if not isinstance(required_regimes,list) or not required_regimes or len(set(required_regimes))!=len(required_regimes):
        raise ValueError('Screening requires a unique explicit regime scope')
    records=[r for manifest in manifests for r in manifest.get('records',[]) if r.get('panel','primary')=='primary']
    parent_specs={p['parent_id']:p for p in protocol.get('parents',[])}
    requested_targets=protocol.get('targets',list(declared))
    if set(declared)!=set(requested_targets) or any(t not in TARGETS for t in declared):
        raise ValueError('Screening candidate targets must match the complete protocol target scope')
    result=dict(schema=SCHEMA,eligible=True,status='RESOLVED_SCREEN',selected_parents={},target_groups={},
        candidates=[],parent_scopes=[],
        policy=dict(minimum_rms_margin=margin,required_mse_reduction=reduction,accuracy_tolerance=tolerance,
            minimum_floor=minimum,selection='First eligible predeclared ordered candidate; all candidates retained',
            scope='Experimental-design headroom screen, not a new scientific passing threshold',
            headroom='(deployment defect RMS - reference RMS uncertainty) >= margin * (nearest FP32 output RMS + reference RMS uncertainty)',
            floor='Training-only max(fixed minimum, reference uncertainty, observed anchor roundoff, half FP32 epsilon at state scale)',
            quantization='FD nodal optimum against recorded teacher; Galerkin coordinatewise benchmark, not physical-L2 optimum',
            independence='Continuous field parent; horizons, phases, grids and seeds remain paired descendants',
            data_access='JSON metadata only; no checkpoint or prediction-array loading'),
        denominators=dict(requested_candidates=0,recorded_candidates=0,eligible_candidates=0,
            requested_candidate_queries=0,recorded_candidate_queries=0,accepted_reference_queries=0,
            unresolved_reference_queries=0,missing_query_records=0,duplicate_query_records=0,
            unexpected_query_records=0,selected_requested_queries=0,selected_recorded_queries=0))
    counts=result['denominators']
    for target,splits in declared.items():
        if set(splits)!={'train','validation'}:raise ValueError('Screening requires both train and validation candidates')
        if set(splits['train'])!=set(required_regimes) or set(splits['validation'])!=set(required_regimes):
            raise ValueError('Screening train/validation regimes must match the complete declared regime scope')
        train_records=[r for r in records if r.get('target_kind')==target and r.get('split')=='train' and
            r.get('parent_id') in {p for ids in splits['train'].values() for p in ids}]
        floor,floor_components=_floor(train_records,minimum)
        group=dict(eligible=True,unresolved_reasons=[],normalization_floor=floor,floor_components=floor_components,
            requested_counts={},selected_counts={})
        result['target_groups'][target]=group;result['selected_parents'][target]={'train':{},'validation':{}}
        for split,regimes in splits.items():
            menu=_menu(protocol,config,split);group['requested_counts'][split]=len(regimes)
            used=set()
            for regime,parents in regimes.items():
                if not isinstance(parents,list) or not parents or len(parents)>2 or len(set(parents))!=len(parents):
                    raise ValueError('Each regime needs one or two distinct ordered candidate parents')
                if used.intersection(parents):raise ValueError('A screening parent cannot represent multiple regimes')
                used.update(parents)
                for position,parent in enumerate(parents):
                    if budget is not None:budget.check()
                    counts['requested_candidates']+=1;counts['requested_candidate_queries']+=len(menu)
                    matching=[r for r in records if r.get('target_kind')==target and r.get('split')==split and r.get('parent_id')==parent]
                    counts['recorded_candidates']+=bool(matching)
                    by_query={}
                    for r in matching:by_query.setdefault(_signature(r),[]).append(r)
                    candidate=dict(target_kind=target,split=split,regime=regime,parent_id=parent,candidate_order=position,
                        eligible=True,selection_status='INELIGIBLE',unresolved_reasons=[],requested_queries=len(menu),
                        recorded_queries=len(matching),queries=[],unexpected_queries=[])
                    parent_spec=parent_specs.get(parent)
                    if parent_spec is None or parent_spec.get('split')!=split or parent_spec.get('tiny_regime')!=regime:
                        candidate['unresolved_reasons'].append('PARENT_SCOPE_MISMATCH')
                    if any(r.get('tiny_regime')!=regime for r in matching):
                        candidate['unresolved_reasons'].append('QUERY_REGIME_SCOPE_MISMATCH')
                    result['parent_scopes'].append(dict(parent_id=parent,target_kind=target,split=split,regime=regime,
                        parent_sha256=(parent_spec or {}).get('parent_sha256'),
                        independent_unit='Continuous field parent; repeated query descendants are not independent fields'))
                    requested_signatures={_signature(q) for q in menu}
                    for request in menu:
                        matches=by_query.get(_signature(request),[])
                        query=_query_screen(matches[0] if matches else None,request,floor,margin,reduction,tolerance)
                        query['all_matching_query_ids']=[r.get('query_id') for r in matches]
                        if len(matches)>1:
                            query['eligible']=False;query['unresolved_reasons'].append('DUPLICATE_QUERY_RECORD')
                            query['duplicate_records']=matches
                            counts['duplicate_query_records']+=len(matches)-1
                        if not matches:counts['missing_query_records']+=1
                        candidate['queries'].append(query)
                        counts['recorded_candidate_queries']+=len(matches)
                        counts['accepted_reference_queries']+=sum(bool(r.get('reference',{}).get('accepted')) for r in matches)
                        counts['unresolved_reference_queries']+=sum(not r.get('reference',{}).get('accepted',False) for r in matches)
                        if not query['eligible']:candidate['unresolved_reasons'].extend(query['unresolved_reasons'])
                    for signature,matches in by_query.items():
                        if signature in requested_signatures:continue
                        candidate['unexpected_queries'].extend(matches);counts['unexpected_query_records']+=len(matches)
                        counts['recorded_candidate_queries']+=len(matches)
                        counts['accepted_reference_queries']+=sum(bool(r.get('reference',{}).get('accepted')) for r in matches)
                        counts['unresolved_reference_queries']+=sum(not r.get('reference',{}).get('accepted',False) for r in matches)
                        candidate['unresolved_reasons'].append('UNEXPECTED_QUERY_RECORD')
                    candidate['unresolved_reasons']=sorted(set(candidate['unresolved_reasons']))
                    candidate['eligible']=not candidate['unresolved_reasons']
                    if candidate['eligible']:
                        counts['eligible_candidates']+=1
                        if regime not in result['selected_parents'][target][split]:
                            result['selected_parents'][target][split][regime]=parent;candidate['selection_status']='SELECTED'
                            counts['selected_requested_queries']+=len(menu);counts['selected_recorded_queries']+=len(matching)
                        else:candidate['selection_status']='ELIGIBLE_NOT_SELECTED'
                    result['candidates'].append(candidate)
                if regime not in result['selected_parents'][target][split]:
                    group['eligible']=False;group['unresolved_reasons'].append(f'{split}/{regime}:NO_ELIGIBLE_CANDIDATE')
            group['selected_counts'][split]=len(result['selected_parents'][target][split])
        result['eligible']=result['eligible'] and group['eligible']
    result['status']='RESOLVED_SCREEN' if result['eligible'] else 'UNRESOLVED_SCREEN'
    return result


def run_screen(ctx):
    stages=ctx.unit.get('data_stages',ctx.unit.get('dependencies',[]));manifests=[]
    for stage in stages:
        if stage not in ctx.prerequisites:raise ValueError('Missing screening reference prerequisite')
        path=Path(ctx.prerequisites[stage])/'dataset.json'
        if not path.is_file():raise ValueError('Screening prerequisite lacks requested dataset metadata')
        manifests.append(json.loads(path.read_text()))
    result=screen_candidates(ctx.protocol,manifests,budget=ctx.budget)
    _write(Path(ctx.path)/'screening.json',result)
    _write(Path(ctx.path)/'screening_summary.json',dict(schema=SCHEMA,status=result['status'],eligible=result['eligible'],
        denominators=result['denominators'],target_groups=result['target_groups'],selected_parents=result['selected_parents']))
    if hasattr(ctx,'record'):
        from .core import check
        for target,group in result['target_groups'].items():
            ctx.record('screening/'+target,['M2','M3'],metrics=dict(target_kind=target,eligible=group['eligible'],
                normalization_floor=group['normalization_floor'],selected_counts=group['selected_counts'],
                requested_counts=group['requested_counts'],unresolved_reasons=group['unresolved_reasons']),
                checks=[check('all_regimes_resolvable',int(group['eligible']),1,'eq',category='math')],
                config=ctx.unit,evidence=['screening.json','screening_summary.json'])
    return dict(screening_status=result['status'],eligible=result['eligible'],denominators=result['denominators'],
        target_groups=result['target_groups'],selected_parents=result['selected_parents'])


# The default response dispatcher loads kind modules and calls run(ctx).
run=run_screen
