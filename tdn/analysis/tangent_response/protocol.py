"""Prospectively frozen work and gates; phase plans never authorize expansion."""
from __future__ import annotations

import copy
import math
import re
from .contracts import SCHEMA, TARGETS, digest, namespaced_seed, validate_target

PROFILES = ('cpu-smoke', 'pilot', 'mechanism', 'compression', 'fit-pilot', 'development')
PHASES = ('mechanism', 'compression', 'fit_prepare', 'calibration', 'learning', 'utility', 'all')
REGIMES = ('low_mean', 'mid_mean', 'high_mean', 'large_amplitude', 'localized', 'mixed', 'overlapping', 'high_frequency')
KINDS = {'audit', 'prepare', 'references', 'mechanism', 'actions', 'compression', 'fit_prepare', 'fit_screen',
         'fit_labels', 'calibration', 'training', 'evaluation', 'performance', 'decision', 'report'}


def _parents(profile, cohort):
    cpu = profile == 'cpu-smoke'
    counts = (dict(train=2, validation=2, diagnostic=2) if cpu else
              dict(train=64, validation=16, diagnostic=24) if profile == 'development' else
              dict(train=8, validation=4, diagnostic=8) if cohort == 'fit' else
              dict(train=16, validation=8, diagnostic=8))
    result = []
    for split, count in counts.items():
        for slot in range(count):
            # Ordered candidates are frozen before reference/headroom screening.
            for candidate in range(1 if cpu or cohort == 'mechanism' else 2):
                label = f'{profile}/{cohort}/{split}/{slot}/{candidate}'
                result.append(dict(parent_id='tr-parent-'+digest(label)[:20], split=split,
                    seed=namespaced_seed('parent', label), slot=slot, candidate=candidate,
                    regime=REGIMES[slot % len(REGIMES)], cohort=cohort,
                    primary=cohort == 'mechanism' or candidate == 0))
    return counts, result


def training_workload(p, arm):
    """Same actual bank/layout/gradients in calibration and committed training."""
    model = dict(rank=p['training']['rank'], width=p['training']['width'], depth=p['training']['depth'],
                 source_nodes=p['training']['source_nodes'], mean_nodes=p['training']['mean_nodes'])
    return digest(dict(target=p['target'], geometry=p['fit_geometry'],
        parent_declarations=p['fit_parents'], horizon_menu=p['horizons'], arm=arm, model=model,
        controls={key:value for key,value in p['training'].items() if key not in ('updates', 'seeds')},
        action_supervision=arm == 'learned_response', endpoint_target='oracle_tangent',
        reference=p['reference'], telemetry=p['worker_telemetry']))


def query_plan(p, cohort='mechanism'):
    """All paired descendants are declared before numerical measurement."""
    table=p['parents'] if cohort=='mechanism' else p['fit_parents']
    cpu=p['profile']=='cpu-smoke';rows=[]
    base_grids=p['geometry']['grids'] if cohort=='mechanism' else [p['fit_geometry']['grid']]
    for parent in table:
        grids=list(base_grids)
        if not cpu and cohort=='mechanism' and parent['slot'] in p['geometry']['refinement_parent_slots']:
            grids+=p['geometry']['paired_refinement_grids']
        if not cpu and cohort=='fit' and parent['split']=='diagnostic' and parent['slot'] in p['evaluation']['refinement_parent_slots']:
            grids+=p['geometry']['paired_refinement_grids']
        for grid in grids:
            for h in p['horizons'][parent['split']]:
                spec=dict(parent_id=parent['parent_id'],split=parent['split'],slot=parent['slot'],candidate=parent['candidate'],
                    regime=parent['regime'],cohort=cohort,grid=list(grid),lengths=list(p['geometry']['lengths']),
                    horizon=h,target_kind=p['target'],kappa=.01,reaction_rate=8.,panel='primary',
                    base_grid=list(base_grids[0]),extrapolation=h>.12,independent_unit=parent['parent_id'])
                spec['query_id']='tr-query-'+digest(spec)[:24];rows.append(spec)
        intervention_slots=(p['mechanism']['intervention_parent_slots'] if cohort=='mechanism' else p['evaluation']['intervention_parent_slots'])
        if parent['split']=='diagnostic' and parent['slot'] in intervention_slots:
            # A fixed field under crossed physics; no new independent parents.
            numbers=[(.1,.1),(1.,1.)] if cpu else [(d,r) for d in (.1,1.,5.) for r in (.1,1.,3.)]
            for d,r in numbers:
                spec=dict(parent_id=parent['parent_id'],split=parent['split'],slot=parent['slot'],candidate=parent['candidate'],
                    regime=parent['regime'],cohort=cohort,grid=list(base_grids[0]),base_grid=list(base_grids[0]),
                    lengths=list(p['geometry']['lengths']),horizon=.12,target_kind=p['target'],panel='physics_intervention',
                    diffusion_number=d,reaction_number=r,coefficient_rule='active_generator_at_base_grid',
                    independent_unit=parent['parent_id'],extrapolation=False)
                spec['query_id']='tr-query-'+digest(spec)[:24];rows.append(spec)
            for phase_shift in ([.125] if cpu else [.125,.25]):
                spec=dict(parent_id=parent['parent_id'],split=parent['split'],slot=parent['slot'],candidate=parent['candidate'],
                    regime=parent['regime'],cohort=cohort,grid=list(base_grids[0]),base_grid=list(base_grids[0]),
                    lengths=list(p['geometry']['lengths']),horizon=.12,target_kind=p['target'],panel='phase_intervention',
                    phase_shift=phase_shift,kappa=.01,reaction_rate=8.,independent_unit=parent['parent_id'],extrapolation=False)
                spec['query_id']='tr-query-'+digest(spec)[:24];rows.append(spec)
    return rows


def _calibration_measurement(calibration, arm, expected):
    if not isinstance(calibration, dict):
        raise ValueError('Learning allocations require actual native calibration')
    data = calibration.get('data', calibration)
    rows = data.get('measurements', data.get('trial_measurements', {}))
    row = rows.get(arm) if isinstance(rows, dict) else None
    if not isinstance(row, dict):
        raise ValueError('Native calibration lacks the declared arm measurement')
    native = data.get('native', data.get('execution_mode') == 'desktop-slurm')
    if (not native or data.get('device') != 'cuda' or
        row.get('completed_updates', row.get('updates', 0)) < 64 or
        row.get('workload_sha256', row.get('workload_hash')) != expected):
        raise ValueError('Calibration hardware, committed work or exact workload differs')
    rate = row.get('seconds_per_update')
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
        raise ValueError('Calibration rate must be a positive measured duration')
    return row


def build_protocol(profile='pilot', target='fd_same_grid', *, phase=None, calibration=None,
                   prerequisites=None, prerequisite=None, chunk_seconds=1500,
                   worker_telemetry=('cpu', 'memory', 'gpu'), sampling_seconds=5, run_id=None,
                   _validate=True):
    validate_target(target)
    if profile not in PROFILES:
        raise ValueError('Unsupported tangent-response profile')
    cpu = profile == 'cpu-smoke'
    phase = phase or ('all' if cpu else 'mechanism')
    if phase not in PHASES or phase == 'all' and not cpu:
        raise ValueError('Only the CPU implementation smoke can exercise all stages without promotion')
    if type(chunk_seconds) is not int or not 1 <= chunk_seconds <= 1800:
        raise ValueError('Science chunks must be between1and1800seconds')
    if not isinstance(sampling_seconds, (int, float)) or isinstance(sampling_seconds, bool) or not 1 <= sampling_seconds <= 300:
        raise ValueError('Telemetry sampling period must be declared between1and300seconds')
    if isinstance(worker_telemetry, str):
        worker_telemetry = tuple(x for x in worker_telemetry.split(',') if x)
    if len(set(worker_telemetry)) != len(worker_telemetry) or set(worker_telemetry)-{'cpu', 'memory', 'gpu'}:
        raise ValueError('Worker telemetry selects cpu,memory,gpu or an empty disabled list')
    mc, parents = _parents(profile, 'mechanism')
    fc, fit_parents = _parents(profile, 'fit')
    horizons = ({'train':[.03], 'validation':[.045], 'diagnostic':[.075]} if cpu else
                {'train':[.01,.03,.06,.12], 'validation':[.02,.045,.09], 'diagnostic':[.015,.075,.15]})
    p = dict(schema=SCHEMA, program='tangent_response', profile=profile, phase=phase, target=target,
        run_id=run_id, execution_scope='CPU_IMPLEMENTATION_SMOKE' if cpu else 'NATIVE_RESEARCH',
        scientific_scope='Gated development; implementation, mechanism, compression, learning and utility are separate claims',
        precision=dict(input='float32', teacher='float64', path='float64', model='float32', tf32=False,
            subtraction='float64', deployment='float32', arithmetic_bridge='separately_computed_and_charged'),
        geometry=dict(grids=[[5,5],[6,6]] if cpu else [[16,16]], lengths=[1.,1.],
            paired_refinement_grids=[] if cpu else [[32,32],[64,64]] if profile=='development' else [[32,32]], refinement_parent_slots=[0],
            resolution_scope='Fixed continuous field and fixed physics on paired grids'),
        fit_geometry=dict(grid=[6,6] if cpu else [16,16], lengths=[1.,1.]),
        horizons=horizons, parent_counts=mc, parents=parents, fit_parent_counts=fc, fit_parents=fit_parents,
        anchor=dict(name='df', reaction_substeps=8, smooth_dtype='float64', fixed_work=True),
        reference=dict(tolerance=1e-10, temporal_budget_fraction=.25, cross_solver_budget_fraction=.25,
            path_budget_fraction=.25, response_budget_fraction=.25,
            initial_steps=8 if cpu else 32, max_steps=256 if cpu else 4096,
            seconds_per_query=30 if cpu else 180, independent_method='numpy_scipy_DOP853',
            dense_verification_max_degrees=64, acceptance='sum_of_estimates_within_total_budget'),
        mechanism=dict(initial_steps=8 if cpu else 32, max_steps=256 if cpu else 4096,
            source_nodes=4 if cpu else 8, max_source_nodes=64,
            tolerance=1e-10, seconds_per_query=30 if cpu else 240,
            controls=['anchor','matched_mean','spatial_tangent','nonlinear_1','nonlinear_2','full_error_oracle','gl4','classical'],
            minimum_defect_to_uncertainty=10., bounded_inputs=True,
            intervention_diffusion_numbers=[.1,1.,5.], intervention_reaction_numbers=[.1,1.,3.],
            intervention_reference_horizon=.12, intervention_parent_slots=[0] if cpu else [0,1],
            intervention_scope='Paired designed descendants; not additional independent parents'),
        compression=dict(ranks=[2] if cpu else [2,4,8], source_fractions=[0.,.5] if cpu else [0.,.25,.5,.75],
            probes_per_type=1 if cpu else 2, independent_probe_types=['fourier_band','signed_random'],
            source_nodes=2 if cpu else 8, initial_steps=8 if cpu else 32,
            max_steps=256 if cpu else 4096, tolerance=1e-10,
            nontrivial_action_threshold=1e-9, minimum_action_to_uncertainty=10.,
            rcond=1e-10, ridge=0., training_only_bases=True, temporal_channels=2 if cpu else 4,t_ref=.12,
            objects=['reusable_operator_action','task_residual_action']),
        training=dict(updates=2 if cpu else 1000 if profile=='development' else 300,
            seeds=[namespaced_seed('model',profile,j) for j in range(3 if profile=='development' else 1)],
            arms=['learned_response','residual_fno'], rank=2 if cpu else 4, width=4 if cpu else 16,
            depth=1 if cpu else 2, source_nodes=2 if cpu else 8, mean_nodes=2 if cpu else 8,
            temporal_channels=2 if cpu else 4,t_ref=.12,modes=4,ndim=2,
            lr=.001, eps=1e-12, warmup_updates=1 if cpu else 5, head_warmup_updates=1 if cpu else 5,
            minimum_lr_ratio=.05, action_fraction=.5, endpoint_target='oracle_tangent',
            objective='equal_parent_physical_mse', peak_penalty=0., validation_tiebreak='sampled_maximum',
            validation_every=1 if cpu else 40, public_curve_every=1 if cpu else 40, live_every=1 if cpu else 40,
            assembly='fp64_bridge', full_parent=True, normalization='frozen_train_complete_gain',
            calibration_updates=2 if cpu else 64, checkpoint_selection='validation_only',
            supervision_scope='Response action and oracle endpoint labels; residual-FNO endpoint labels only'),
        evaluation=dict(rollout_steps=[2,4,8], shifts=['reaction','diffusion','mean','amplitude','phase','frequency','horizon','resolution'],
            intermediate_teachers=True, extrapolation_horizon=.15, sampled_maximum=True,
            auxiliary_parents=1 if cpu else 2,refinement_parent_slots=[0],
            intervention_parent_slots=[0] if cpu else [0,1],sensitivity_epsilons=[1e-4,1e-5],
            maximum_oversampling=4,source_nodes=2 if cpu else 8),
        performance=dict(tolerances=[1e-4,2e-5,1e-5], primary_tolerance=2e-5,
            repeats=2 if cpu else 5, warmup=1, batch_sizes=[1], selection='validation_only',
            classical_schedules=[1,2] if cpu else [1,2,4,8],adaptive_tolerances=[1e-8] if cpu else [1e-8,1e-10],
            fresh_process_repeats=1 if cpu else 2,fresh_process_timeout=60,
            fresh_process_query='first declared primary diagnostic query; separate startup/checkpoint/input scope',
            include_path_residual=True, include_arithmetic_bridge=True, full_process='separate_measured_scope'),
        gates=dict(A=dict(rms_ratio=1.5, parent_improvement_fraction=.75, maximum_regression=2.,
                         uncertainty_fraction_of_improvement=.1, minimum_signal_uncertainty_ratio=10.),
                   B=dict(rank_cap=8, action_relative_error=.1, probe_fraction=.9, worst_action_relative_error=.3,
                         retained_oracle_gain=.8),
                   C1=dict(relative_oracle_correction_error=.05, query_fraction=.9),
                   C2=dict(retained_oracle_gain=.8), D=dict(primary_tolerance=2e-5, cost_ratio=1.)),
        worker_telemetry=dict(enabled=list(worker_telemetry), sampling_seconds=sampling_seconds,
            scope='Actual allocated worker process/descendants and assigned devices', missing='NA',
            observer_failure='distinct_warning_unless_measurement_required_for_claim'),
        prerequisites=copy.deepcopy(prerequisites or ({'prerequisite':prerequisite} if prerequisite else {})),
        calibration=copy.deepcopy(calibration), chunk_seconds=chunk_seconds,
        downstream=dict(compression='UNAVAILABLE_UNTIL_A_PASS', calibration='UNAVAILABLE_UNTIL_B_AND_FIT_DATA_PASS',
                        learning='UNAVAILABLE_UNTIL_SOURCE_BOUND_NATIVE_CALIBRATION', utility='UNAVAILABLE_UNTIL_C_PASS'),
        unsupported_targets=dict(galerkin_projected_fine='Hidden-state/lift and projected forcing contract absent'),
        decisions_are_prospective=True, confirmation=False, stages=[], units={}, trials={})
    p['queries']=query_plan(p,'mechanism')
    p['fit_queries']=query_plan(p,'fit')
    _build_units(p)
    # The explicit suite is assembled from this namespace's declared cases.
    try:
        from .gpu_manifest import GPU_TEST_CASES, GPU_TEST_FILES,test_source_hashes
        p.update(gpu_test_cases=copy.deepcopy(list(GPU_TEST_CASES)), gpu_test_files=list(GPU_TEST_FILES),
                 gpu_test_source_sha256=test_source_hashes())
    except ModuleNotFoundError:
        p.update(gpu_test_cases=[],gpu_test_files=[])
    if _validate:validate_protocol(p, require_binding=False)
    return p


def _build_units(p):
    cpu = p['profile']=='cpu-smoke'
    all_phases = p['phase']=='all'
    units=p['units'];stages=p['stages']
    def add(name,kind,deps=(),seconds=240,gpu=False,**fields):
        seconds=max(1,min(1800,int(seconds)))
        wall=seconds+180
        units[name]=dict(id=name,kind=kind,dependencies=list(deps),device='cpu' if cpu or not gpu else 'cuda',
            cpus=4 if gpu else 8,mem_gib=32 if gpu else 48,seconds=seconds,
            walltime=f'{wall//3600:02}:{wall%3600//60:02}:{wall%60:02}',**fields)
        stages.append(name);return name
    audit=add('audit','audit',seconds=600,gpu=not cpu)
    prepared=None;mechanism_units=[]
    if p['phase']=='mechanism' or all_phases:
        prepared=add('prepare','prepare',[audit],cohort='mechanism')
        for index,query in enumerate(p['queries']):
            ref=add(f'references-{index:03}','references',[prepared],seconds=600,parent_ids=[query['parent_id']],query_ids=[query['query_id']],cohort='mechanism')
            mechanism_units.append(add(f'mechanism-{index:03}','mechanism',[ref],seconds=1500,
                parent_ids=[query['parent_id']],query_ids=[query['query_id']],cohort='mechanism'))
        if not all_phases:
            add('decision','decision',mechanism_units,gate='A')
    if p['phase']=='compression' or all_phases:
        actions=[]
        for index,query in enumerate(p['queries']):
            actions.append(add(f'actions-{index:03}','actions',mechanism_units[index:index+1] or [audit],
                seconds=1800,parent_ids=[query['parent_id']],query_ids=[query['query_id']],cohort='mechanism'))
        comp=add('compression','compression',actions,seconds=1800,cohort='mechanism')
        if not all_phases:add('decision','decision',[comp],gate='B')
    fit_label_units=[];screen=None
    if p['phase']=='fit_prepare' or all_phases:
        fprep=add('fit-prepare','fit_prepare',[audit],cohort='fit')
        for index,query in enumerate(p['fit_queries']):
            ref=add(f'fit-references-{index:03}','references',[fprep],seconds=600,parent_ids=[query['parent_id']],query_ids=[query['query_id']],cohort='fit')
            fit_label_units.append(add(f'fit-labels-{index:03}','fit_labels',[ref],seconds=1500,
                parent_ids=[query['parent_id']],query_ids=[query['query_id']],cohort='fit'))
        screen=add('fit-screen','fit_screen',fit_label_units,cohort='fit')
        if not all_phases:add('decision','decision',[screen],gate='C0')
    cal_units=[]
    if p['phase']=='calibration' or all_phases:
        for arm in p['training']['arms']:
            cal_units.append(add('calibration-'+arm,'calibration',[screen] if screen else [audit],seconds=1800,gpu=True,
                arm=arm,updates=p['training']['calibration_updates'],workload_sha256=training_workload(p,arm)))
        if not all_phases:add('decision','decision',cal_units,gate='CALIBRATION')
    training_units=[]
    if p['phase']=='learning' or all_phases:
        for arm in p['training']['arms']:
            if cpu:
                allowance=1200;chunks=1;measurement=None
            elif p['calibration'] is None:
                allowance=None;chunks=0;measurement=None
            else:
                measurement=_calibration_measurement(p['calibration'],arm,training_workload(p,arm))
                allowance=max(180,math.ceil(2*measurement['seconds_per_update']*p['training']['updates']+120))
                chunks=math.ceil(allowance/p['chunk_seconds'])
            for seed in p['training']['seeds']:
                trial='trial-'+digest({'arm':arm,'seed':seed,'target':p['target']})[:16]
                p['trials'][trial]=dict(id=trial,arm=arm,seed=seed,updates=p['training']['updates'],
                    workload_sha256=training_workload(p,arm),total_seconds=allowance,
                    allocation_status='UNAVAILABLE_UNTIL_NATIVE_CALIBRATION' if allowance is None else 'FROZEN',
                    measurement=measurement,model_config=dict(family=arm,rank=p['training']['rank'],width=p['training']['width'],depth=p['training']['depth']))
                prior=None
                for chunk in range(chunks):
                    prior=add(f'{trial}-{chunk:03}','training',[prior] if prior else ([screen]+cal_units if screen else [audit]),
                        seconds=min(p['chunk_seconds'],allowance-chunk*p['chunk_seconds']),gpu=True,
                        trial_id=trial,arm=arm,seed=seed,updates=p['training']['updates'],chunk_index=chunk,
                        chunk_count=chunks,previous_chunk=prior,cohort='fit')
                if prior:training_units.append(prior)
        evaluation=add('evaluation','evaluation',training_units or [audit],seconds=1800,gpu=True,cohort='fit')
        if not all_phases:add('decision','decision',[evaluation],gate='C')
    if p['phase']=='utility' or all_phases:
        perf=add('performance','performance',[evaluation] if all_phases else [audit],seconds=1800,gpu=True,cohort='fit')
        if not all_phases:add('decision','decision',[perf],gate='D')
    if all_phases:
        add('decision','decision',[s for s in stages if s!='audit'],gate='ALL',implementation_only=True)
    add('report','report',list(stages),seconds=900)


def ancestors(p,stage):
    if stage not in p['units']:raise ValueError('Unknown tangent-response unit')
    found=set()
    def visit(name):
        for parent in p['units'][name]['dependencies']:
            if parent not in found:found.add(parent);visit(parent)
    visit(stage)
    return tuple(name for name in p['stages'] if name in found)


def validate_protocol(p, *, require_binding=True, canonical=True):
    """Validate current execution plans or a frozen historical read-only plan.

    Historical callers still prove the stored protocol/unit digests and artifact
    bytes. They skip rebuilding today's defaults, which is required for every
    fresh execution, binding and recovery path.
    """
    if type(canonical) is not bool or type(require_binding) is not bool:
        raise ValueError('Protocol validation modes must be explicit booleans')
    if not canonical and require_binding:
        raise ValueError('Historical protocol validation cannot authorize execution or binding')
    if not isinstance(p,dict) or p.get('schema')!=SCHEMA or p.get('program')!='tangent_response':
        raise ValueError('Invalid tangent-response protocol')
    validate_target(p.get('target'))
    if p.get('profile') not in PROFILES or p.get('phase') not in PHASES or p.get('confirmation') is not False:
        raise ValueError('Invalid tangent-response scope')
    if p['phase']=='all' and p['profile']!='cpu-smoke':
        raise ValueError('All-stage bypass is CPU implementation verification only')
    if p.get('execution_scope')!=('CPU_IMPLEMENTATION_SMOKE' if p['profile']=='cpu-smoke' else 'NATIVE_RESEARCH'):
        raise ValueError('Frozen execution scope differs from its declared profile')
    if canonical:
        expected=build_protocol(p['profile'],p['target'],phase=p['phase'],calibration=p.get('calibration'),
            prerequisites=p.get('prerequisites'),chunk_seconds=p.get('chunk_seconds'),
            worker_telemetry=p.get('worker_telemetry',{}).get('enabled',[]),
            sampling_seconds=p.get('worker_telemetry',{}).get('sampling_seconds'),run_id=p.get('run_id'),_validate=False)
        if p!=expected:
            raise ValueError('Frozen protocol, query identities, gates or unit plan changed')
    if not isinstance(p.get('stages'),list) or not isinstance(p.get('units'),dict) or not p['stages']:
        raise ValueError('Frozen protocol requires a bounded declared unit roster')
    seen=set()
    for name in p['stages']:
        if not isinstance(name,str) or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,127}',name) or name not in p['units']:
            raise ValueError('Frozen protocol has an unsafe or missing unit identity')
        row=p['units'][name];deps=row.get('dependencies') if isinstance(row,dict) else None
        if not isinstance(deps,list) or not all(isinstance(d,str) for d in deps):
            raise ValueError('Frozen unit dependencies must be explicit identities')
        if name in seen or row['id']!=name or row['kind'] not in KINDS or len(deps)!=len(set(deps)) or not set(deps)<=seen:
            raise ValueError('Tangent-response units must have immutable ordered dependencies')
        if row['device'] not in ('cpu','cuda') or p['profile']=='cpu-smoke' and row['device']!='cpu':
            raise ValueError('CPU smoke cannot establish or silently request CUDA')
        if not isinstance(row.get('walltime'),str) or not re.fullmatch('[0-9]{2}:[0-5][0-9]:[0-5][0-9]',row['walltime']):
            raise ValueError('Frozen unit walltime must retain its exact bounded format')
        h,m,s=map(int,row['walltime'].split(':'))
        if (type(row.get('seconds')) is not int or type(row.get('cpus')) is not int or type(row.get('mem_gib')) is not int
                or not 0<row['seconds']<=1800 or not row['seconds']+120<=h*3600+m*60+s<=2100
                or not 1<=row['cpus']<=8 or not 1<=row['mem_gib']<=48 or row['device']=='cuda' and row['cpus']!=4):
            raise ValueError('Tangent-response resource envelope changed')
        seen.add(name)
    if set(p['units'])!=seen or p['units']['report']['kind']!='report':
        raise ValueError('Complete declared unit roster and failure-aware report are required')
    if set(p['units']['report']['dependencies'])!=seen-{'report'}:
        raise ValueError('Frozen failure-aware report must observe every declared unit')
    if require_binding and p['execution_scope']=='NATIVE_RESEARCH':
        if p['phase']!='mechanism' and not p.get('prerequisites'):
            raise ValueError('Downstream native work requires literal sealed prerequisite evidence')
        if p['profile']=='development' and not p.get('prerequisites'):
            raise ValueError('Development requires a literal eligible pilot decision')
        if p['phase']=='learning' and (not p.get('calibration') or any(t['total_seconds'] is None for t in p['trials'].values())):
            raise ValueError('Learning submission requires actual source-bound native calibration')
        if not p.get('gpu_test_cases') or not p.get('gpu_test_files'):
            raise ValueError('Native readiness must enumerate this program\'s exact CUDA cases')
    ids=[row['parent_id'] for row in p['parents']+p['fit_parents']]
    if len(ids)!=len(set(ids)):
        raise ValueError('Independent parent declarations overlap across cohorts/splits')
    return p
