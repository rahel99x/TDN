#!/usr/bin/env python3
"""Bounded Response development; controllers only schedule and inspect metadata."""
from __future__ import annotations
import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location('response_fedora_helpers', ROOT / 'scripts/fedora_workflow.py')
fw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fw)
pw, cw, rw = fw.pw, fw.cw, fw.rw
from tdn.runtime.desktop_slurm import load_profile, validate_profile, verify_allocation as runtime_allocation

MANIFEST = 'response-workflow.json'
POINTER = '.fedora-response-latest.json'
PROFILES = ('smoke', 'pilot', 'development', 'confirmation', 'diagnostic-smoke', 'diagnostic')
SCHEDULER_MEMORY_CAP_MIB = 110000
ARCHIVE_PART_BYTES = 28 * 2**20
ACCOUNTING_FIELDS = ('JobIDRaw', 'State', 'ExitCode', 'ElapsedRaw', 'TotalCPU', 'AllocTRES', 'MaxRSS', 'CPUTimeRAW')
TERMINAL_JOB_STATES = frozenset(('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
    'PREEMPTED', 'NODE_FAIL', 'BOOT_FAIL', 'DEADLINE', 'REVOKED', 'SPECIAL_EXIT'))


def scientific_protocol(value):
    from tdn.analysis.response.protocol import build_protocol
    return value['scientific_protocol'] if isinstance(value,dict) else build_protocol(value)


def units(workflow):
    return scientific_protocol(workflow)['units']


def physical_stages(workflow):
    from tdn.analysis.response.protocol import plan_units
    return tuple(row['id'] for row in plan_units(scientific_protocol(workflow)))


def ancestors(workflow, stage):
    rows = units(workflow)
    if stage not in rows:
        raise ValueError('Unknown response unit')
    found = set()
    def visit(name):
        for parent in rows[name]['dependencies']:
            if parent not in found:
                found.add(parent)
                visit(parent)
    visit(stage)
    return tuple(name for name in physical_stages(workflow) if name in found)


def declaration(profile):
    protocol = scientific_protocol(profile)
    return {'version': 1, 'benchmark_suite': 'response', 'profile': protocol['profile'],
        'scientific_protocol': protocol, 'execution_mode': 'desktop-slurm',
        'evidence_scope': protocol['scientific_scope'],
        'scheduler_memory_cap_mib': SCHEDULER_MEMORY_CAP_MIB,
        'scheduling': 'afterok science DAG; afterany single-node resource serialization; report afterany all',
        'scope': 'No pending-job cap, automatic budget expansion, or automatic retries'}


def is_gpu_stage(workflow, stage):
    return units(workflow)[stage]['device'] == 'cuda'


def resource_for(workflow, stage):
    if stage not in physical_stages(workflow):
        raise ValueError('Unknown response unit')
    return workflow['resources'][stage]


def validate_resources(protocol):
    rows = protocol['units']
    for stage, row in rows.items():
        if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]*', stage):
            raise ValueError('Unsafe response unit identifier')
        if type(row['cpus']) is not int or not 1 <= row['cpus'] <= 8:
            raise ValueError('Response requires at most eight physical CPU cores')
        if row['device'] == 'cuda' and row['cpus'] != 4:
            raise ValueError('Response GPU units require four CPUs and one GPU')
        if type(row['mem_gib']) is not int or not 1 <= row['mem_gib'] <= 48:
            raise ValueError('Response memory must fit 48 GiB per unit and 110000 MiB node')
        if not 180 <= wall_seconds(row['walltime']) <= 2100:
            raise ValueError('Response allocations must remain within 35 minutes')
        if not 0 < row['seconds'] <= min(1800, wall_seconds(row['walltime']) - 120):
            raise ValueError('Scientific budget must leave bounded reporting/signal headroom')
    return {name: {key: row[key] for key in ('cpus', 'mem_gib', 'walltime')} for name,row in rows.items()}


def validate(workflow):
    if workflow.get('schema_version') != 1 or workflow.get('kind') != 'desktop-slurm-response':
        raise ValueError('Not a supported response workflow')
    run_id = pw.identifier(workflow['run_id'])
    base = cw.inside(workflow['run_dir'])
    if base != cw.inside(ROOT/'runs'/run_id) or cw.inside(workflow['root']) != ROOT.resolve():
        raise ValueError('Response workflow layout changed')
    profile = workflow['profile']
    from tdn.analysis.response.protocol import validate_protocol
    validate_protocol(scientific_protocol(workflow),require_binding=False)
    if scientific_protocol(workflow)['profile'] != profile: raise ValueError('Workflow scientific profile differs')
    if profile not in PROFILES or workflow.get('execution_mode') != 'desktop-slurm':
        raise ValueError('Unknown response profile or execution mode')
    if cw.inside(workflow['protocol_path']) != base/'protocol.json':
        raise ValueError('Frozen protocol must remain within coordinator')
    if workflow['protocol_sha256'] != pw.canonical_hash(declaration(workflow)):
        raise ValueError('Response scientific declaration changed')
    site = validate_profile(workflow['slurm_profile'], root=ROOT)
    if workflow['slurm_profile_sha256'] != pw.canonical_hash(site) or cw.inside(workflow['slurm_profile_path']) != base/'slurm-profile.json':
        raise ValueError('Frozen Slurm profile differs')
    if workflow.get('torch_version') != site['torch_version']:
        raise ValueError('Torch distribution differs from Slurm profile')
    resources = validate_resources(scientific_protocol(workflow))
    expected = {stage: {**res, 'partition': site['gpu_partition' if is_gpu_stage(workflow,stage) else 'cpu_partition']}
                for stage,res in resources.items()}
    if workflow.get('resources') != expected:
        raise ValueError('Allocation differs from frozen resource budgets')
    for key in ('source_sha256','source_tree_sha256'):
        if not pw.SHA.fullmatch(workflow.get(key,'')):
            raise ValueError('Missing execution source fingerprint')
    recovery = workflow.get('recovery')
    if recovery:
        if set(recovery) != {'manifest_path','sha256','stage_paths','resume_paths'} or not pw.SHA.fullmatch(recovery['sha256']):
            raise ValueError('Invalid frozen recovery descriptor')
        if (not set(recovery['stage_paths']).isdisjoint(recovery['resume_paths']) or
                not (set(recovery['stage_paths']) | set(recovery['resume_paths'])) <= set(units(workflow)) - {'report'}):
            raise ValueError('Invalid completed/resumable unit mapping')
    return workflow


def workflow_path(value):
    if value == 'latest':
        pointer = os.environ.get('TDN_RESPONSE_POINTER', POINTER)
        if pointer not in (POINTER, '.fedora-response-diagnostic-latest.json'):
            raise ValueError('Unknown Response latest pointer')
        value = cw.read_json(ROOT/'runs'/pointer)['run_id']
    return cw.inside(ROOT/'runs'/pw.identifier(value)/MANIFEST)


def recovery_bridge(workflow):
    descriptor = workflow.get('recovery')
    if not descriptor:
        return None
    path = cw.inside(descriptor['manifest_path'])
    if path != Path(workflow['run_dir'])/'recovery.json' or cw.digest(path) != descriptor['sha256']:
        raise ValueError('Recovery bridge changed')
    bridge = cw.read_json(path)
    if any(bridge[key] != descriptor[key] for key in ('stage_paths','resume_paths')):
        raise ValueError('Recovery paths differ from sealed bridge')
    return bridge


def stage_path(workflow, stage):
    resource_for(workflow,stage)
    bridge = recovery_bridge(workflow)
    return cw.inside(bridge['stage_paths'][stage] if bridge and stage in bridge['stage_paths'] else Path(workflow['run_dir'])/stage)


def pending_stages(workflow):
    return tuple(stage for stage in physical_stages(workflow) if stage not in workflow.get('recovery',{}).get('stage_paths',{}))


def load(path, *, verify=True):
    path = cw.inside(path)
    value = validate(cw.read_json(path))
    if path != Path(value['run_dir'])/MANIFEST:
        raise ValueError('Response manifest is outside its coordinator')
    if verify:
        if cw.read_json(value['protocol_path']) != declaration(value):
            raise ValueError('Frozen protocol changed')
        if load_profile(value['slurm_profile_path'],root=ROOT) != value['slurm_profile']:
            raise ValueError('Frozen Slurm profile changed')
        if cw.source_hash() != value['source_sha256']:
            raise ValueError('Source changed; preserve run and submit a new workflow')
    return value


def prepare(args):
    from tdn.analysis.response.protocol import build_protocol
    site = load_profile(root=ROOT)
    run_id = pw.identifier(args.run_id or 'fedora-response-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    base = cw.inside(ROOT/'runs'/run_id)
    if base.exists(): raise ValueError('Run directory exists; preserve it and select a fresh run ID')
    profile = ('diagnostic' if getattr(args,'diagnostic',False) else 'diagnostic-smoke' if getattr(args,'diagnostic_smoke',False)
        else 'confirmation' if getattr(args,'confirmation',False) else 'development' if getattr(args,'development',False)
        else 'pilot' if getattr(args,'pilot',None) else 'smoke')
    if profile.startswith('diagnostic') and (getattr(args,'pilot',None) or getattr(args,'freeze',None)):
        raise ValueError('Diagnostics are independent of the historical pilot/development/confirmation bindings')
    overrides={}
    for item in getattr(args,'trial_seconds',[]) or []:
        family,sep,seconds=item.partition('=')
        if not sep or not family or family in overrides: raise ValueError('Use unique --trial-seconds FAMILY=SECONDS declarations')
        overrides[family]=int(seconds)
    update_overrides={}
    for item in getattr(args,'updates',[]) or []:
        family,sep,count=item.partition('=')
        if not sep or not family or family in update_overrides: raise ValueError('Use unique --updates FAMILY=COUNT declarations')
        update_overrides[family]=int(count)
    calibration=read_calibration(args.calibration) if getattr(args,'calibration',None) else None
    selected=pilot_selection(args.pilot) if profile=='development' and isinstance(getattr(args,'pilot',None),str) else None
    frozen=read_freeze(args.freeze) if getattr(args,'freeze',None) else None
    protocol=build_protocol(profile,trial_seconds=overrides,chunk_seconds=getattr(args,'chunk_seconds',1500),
        selected=selected,freeze=frozen,calibration=calibration,updates=update_overrides,
        targets=getattr(args,'targets',None))
    return prepare_from_protocol(protocol,run_id,site=site)


def prepare_from_protocol(protocol,run_id,*,site=None):
    site=load_profile(root=ROOT) if site is None else site
    base=cw.inside(ROOT/'runs'/pw.identifier(run_id))
    if base.exists(): raise ValueError('Run directory exists; preserve the original')
    resources=validate_resources(protocol)
    workflow={'schema_version':1,'kind':'desktop-slurm-response','profile':protocol['profile'],'run_id':run_id,
        'root':str(ROOT.resolve()),'run_dir':str(base),'created_at':cw.now(),'execution_mode':'desktop-slurm',
        'scientific_protocol':protocol,'protocol_path':str(base/'protocol.json'),
        'slurm_profile':site,'slurm_profile_path':str(base/'slurm-profile.json'),
        'slurm_profile_sha256':pw.canonical_hash(site),'torch_version':site['torch_version'],
        'source_sha256':cw.source_hash(),'source_tree_sha256':cw.source_hash(metadata=True),
        'resources':{stage:{**row,'partition':site['gpu_partition' if protocol['units'][stage]['device']=='cuda' else 'cpu_partition']} for stage,row in resources.items()}}
    workflow['protocol_sha256']=pw.canonical_hash(declaration(workflow))
    return validate(workflow)


def read_calibration(run):
    origin=load(workflow_path(run))
    if origin['profile']=='diagnostic-smoke':
        return read_diagnostic_calibration(origin)
    if origin['profile']!='smoke': raise ValueError('Calibration must be a completed native Response smoke')
    measurements={}; sources={}
    for name,unit in units(origin).items():
        if unit['kind']!='train' or unit.get('phase')!='factorial': continue
        verified=verify_stage(origin,name)
        path=stage_path(origin,name); row=cw.read_json(path/'trial.json')
        execution=cw.read_json(path/'execution.json')
        if execution.get('execution_mode')!='desktop-slurm' or execution.get('device')!='cuda':
            raise ValueError('CPU verification cannot calibrate a native training allowance')
        check_gpu_tests(path.parent/"reporter-tests"/name/"tests.xml")
        if row.get('status')!='COMPLETED_UPDATES' or row.get('updates',0)<64:
            raise ValueError('Calibration requires at least64 completed updates per model/target')
        cost=row.get('total_cost_including_postprocessing_seconds',row.get('cost',{}).get('total_seconds'))
        if not isinstance(cost,(float,int)) or not math.isfinite(cost) or cost<=0: raise ValueError('Measured calibration cost unavailable')
        key=row['family']+'/'+row['target_kind']; rate=cost/row['updates']
        measurements[key]=max(measurements.get(key,0),rate)
        sources[name]={'path':str(path),'trial_sha256':cw.digest(path/'trial.json'),'workflow_seal_sha256':cw.digest(path/'workflow-seal.json')}
    expected={f+'/'+t for f in ('direct_fno','residual_fno','temporal','coupled_source') for t in scientific_protocol(origin)['targets']}
    if set(measurements)!=expected: raise ValueError('Native calibration lacks required family/target workloads')
    return {'execution_mode':'desktop-slurm','verified_native_gpu':True,'run_id':origin['run_id'],
        'workflow_path':str(Path(origin['run_dir'])/MANIFEST),'workflow_sha256':cw.digest(Path(origin['run_dir'])/MANIFEST),
        'source_tree_sha256':origin['source_tree_sha256'],'seconds_per_update':measurements,'sources':sources,
        'train_grid':scientific_protocol(origin)['train_grid'],
        'estimator':'conservative observed total64-update walltime/update; includes recorded validation and checkpoint work; forecast =2*updates*rate*declared capacity/rank/objective/cohort operation factors +120seconds; not a confidence guarantee'}


def read_diagnostic_calibration(origin):
    """Bind every exact architecture/objective/batch workload to real CUDA work."""
    protocol=scientific_protocol(origin); measurements={}; sources={}
    for trial_id, declared in protocol['trials'].items():
        stages=[name for name,unit in units(origin).items() if unit.get('trial_id')==trial_id]
        stage=max(stages,key=lambda name:units(origin)[name]['chunk_index'])
        verify_stage(origin,stage)
        path=stage_path(origin,stage); row=cw.read_json(path/'trial.json'); execution=cw.read_json(path/'execution.json')
        if execution.get('execution_mode')!='desktop-slurm' or execution.get('device')!='cuda':
            raise ValueError('Diagnostic calibration needs real native CUDA execution')
        check_gpu_tests(junit_path(origin,stage),protocol=protocol)
        if row.get('status')!='COMPLETED_UPDATES' or row.get('updates',0)<64:
            raise ValueError('Every diagnostic architecture needs at least64 completed native updates')
        if any(row.get('accepted_'+s+'_queries')!=row.get('requested_'+s+'_queries') for s in ('training','validation')):
            raise ValueError('Calibration with unresolved teacher queries cannot measure the complete declared workload')
        expected_counts={'training':6*len(protocol['train_horizons'])*len(protocol['phase_variants']),
                         'validation':6*len(protocol['validation_horizons'])*len(protocol['phase_variants'])}
        if any(row.get('accepted_'+split+'_queries')!=count for split,count in expected_counts.items()):
            raise ValueError('Calibration differs from its declared six-parent query layout')
        correction=row.get('correction_diagnostics',{})
        if (correction.get('status')!='MEASURED' or correction.get('queries')!=row.get('accepted_training_queries')
                or correction.get('requested_queries')!=row.get('accepted_training_queries')):
            raise ValueError('Calibration requires complete requested correction measurements')
        if row.get('inference_cost',{}).get('status')!='MEASURED':
            raise ValueError('Calibration requires the measured selected-checkpoint inference panel')
        seconds=row.get('total_cost_including_postprocessing_seconds',row.get('cost',{}).get('total_seconds'))
        if not isinstance(seconds,(float,int)) or not math.isfinite(seconds) or seconds<=0:
            raise ValueError('Diagnostic measured complete workload time is unavailable')
        key=declared['calibration_key']
        if key in measurements: raise ValueError('Diagnostic smoke must measure each variant/target once')
        measurements[key]={'seconds_per_update':seconds/row['updates'],'updates':row['updates'],
            'workload_sha256':declared['calibration_workload_sha256'],'seconds':seconds,
            'accepted_training_queries':row['accepted_training_queries'],'accepted_validation_queries':row['accepted_validation_queries']}
        sources[key]={'stage':stage,'trial_sha256':cw.digest(path/'trial.json'),
            'execution_sha256':cw.digest(path/'execution.json'),'workflow_seal_sha256':cw.digest(path/'workflow-seal.json')}
    if not measurements: raise ValueError('Diagnostic calibration contains no measured architectures')
    return {'execution_mode':'desktop-slurm','verified_native_gpu':True,'run_id':origin['run_id'],
        'workflow_path':str(Path(origin['run_dir'])/MANIFEST),'workflow_sha256':cw.digest(Path(origin['run_dir'])/MANIFEST),
        'source_tree_sha256':origin['source_tree_sha256'],'variant_measurements':measurements,'sources':sources,
        'train_grid':protocol['train_grid'],'estimator':'2*declared updates*observed exact-variant seconds/update +120 seconds; individual full batches use a declared six-parent work ratio; not measured future runtime'}


def gate_results(workflow):
    if workflow['profile']!='pilot': raise ValueError('Only the full trainability pilot can authorize development; smoke cannot')
    verify_stage(workflow,'gate'); verify_stage(workflow,'optimizer-selection')
    gate=cw.read_json(stage_path(workflow,'gate')/'gate.json')
    selection=cw.read_json(stage_path(workflow,'optimizer-selection')/'selection.json')
    expected={t+'/'+f for t in scientific_protocol(workflow)['targets'] for f in ('direct_fno','residual_fno','temporal')}
    if {row['target_kind']+'/'+row['family'] for row in gate.get('groups',[])}!=expected or set(selection.get('selected_settings',{}))!=expected:
        raise ValueError('Pilot gate has incomplete family/target coverage')
    passed=gate.get('gate_passed') is True and all(r.get('passed') is True for r in gate['groups']) and all(selection['selected_settings'][k] for k in expected)
    return {**gate,'decision':'CONTINUE' if passed else 'STOP_OR_UNRESOLVED',
        'eligible_for_development':passed,'selected_settings':selection['selected_settings']}


def pilot_selection(run):
    origin=load(workflow_path(run)); result=gate_results(origin)
    if not result['eligible_for_development']: raise ValueError('Pilot trainability/competitor gate is not eligible; preserve negative results')
    models=[]
    for key,choices in result['selected_settings'].items():
        target,family=key.split('/',1)
        for choice in choices: models.append({'target_kind':target,'family':family,'setting':choice['setting']})
    for target in scientific_protocol(origin)['targets']:
        choice=result['selected_settings'][target+'/residual_fno'][0]
        models.append({'target_kind':target,'family':'coupled_source','setting':choice['setting'],
            'scope':'StageB conditional on StageA trainability; not claimed trained by the pilot'})
    return {'pilot_run':origin['run_id'],'workflow_path':str(Path(origin['run_dir'])/MANIFEST),
        'workflow_sha256':cw.digest(Path(origin['run_dir'])/MANIFEST),'source_tree_sha256':origin['source_tree_sha256'],
        'gate_sha256':cw.digest(stage_path(origin,'gate')/'gate.json'),'gate_decision':'CONTINUE',
        'selected_models':models,'selected_protocol_sha256':pw.canonical_hash(models)}


def read_freeze(path):
    path=cw.inside(path); frozen=cw.read_json(path)
    if frozen.get('schema')!='tdn.response-freeze/v1' or frozen.get('selection_split')!='validation':
        raise ValueError('Confirmation needs a Response validation-selected freeze')
    if frozen.get('source_tree_sha256')!=cw.source_hash(metadata=True): raise ValueError('Freeze scientific source differs from this checkout')
    for row in frozen.get('models',[]):
        checkpoint=cw.inside(path.parent/row['checkpoint'])
        if not checkpoint.is_relative_to(path.parent) or cw.digest(checkpoint)!=row['checkpoint_sha256']:
            raise ValueError('Frozen checkpoint bytes/path differ')
    if not frozen.get('models') or not frozen.get('estimands'):raise ValueError('Freeze has no concrete candidate/comparators and estimands')
    return {'path':str(path),'sha256':cw.digest(path),'source_tree_sha256':frozen['source_tree_sha256'],
        'protocol_sha256':frozen['protocol_sha256'],'models':frozen['models'],'estimands':frozen['estimands'],
        'confirmation_root_seed':frozen['confirmation_root_seed'],'confirmation_parent_count':frozen['confirmation_parent_count'],
        'target_contracts':frozen['target_contracts'],'evaluation':frozen['evaluation']}


def verify_external_bindings(workflow):
    protocol=scientific_protocol(workflow)
    calibration=protocol['allowance'].get('calibrated_forecast')
    if calibration:
        if calibration['source_tree_sha256']!=workflow['source_tree_sha256']:raise ValueError('Calibration scientific source differs')
        if read_calibration(calibration['run_id'])!=calibration:raise ValueError('Calibration measurements changed')
    selected=protocol.get('selection_source')
    if selected and pilot_selection(selected['pilot_run'])!=selected:raise ValueError('Pilot gate/selected protocol changed')
    frozen=protocol.get('freeze')
    if frozen and read_freeze(frozen['path'])!=frozen:raise ValueError('Confirmation freeze changed')



def freeze_results(workflow, output_dir):
    """Create an immutable derivative selection, never alter completed science."""
    if workflow['profile']!='development': raise ValueError('Freeze requires completed selected development, not smoke/pilot')
    verify_external_bindings(workflow)
    for name in ('selection','aggregate'): verify_stage(workflow,name)
    selected_path=stage_path(workflow,'selection')/'selection.json'
    aggregate_path=stage_path(workflow,'aggregate')/'aggregate.json'
    selected=cw.read_json(selected_path);aggregate=cw.read_json(aggregate_path)
    if aggregate.get('freeze_eligible') is not True:
        raise ValueError('Development claim/coverage gate does not permit confirmation; preserve negative or unresolved evidence')
    records=selected.get('selected_models',[])
    if not records or selected.get('missing'):raise ValueError('Selection is incomplete')
    output=cw.inside(output_dir)
    if output.exists() or output.parent!=ROOT/'outputs':
        raise ValueError('Freeze must use a fresh outputs/<freeze-id> derivative directory')
    source_tree=workflow['source_tree_sha256']
    if source_tree!=cw.source_hash(metadata=True):raise ValueError('Scientific source changed before freeze')
    checked=[]
    for row in records:
        stage=row.get('stage')
        if stage not in units(workflow) or units(workflow)[stage]['kind']!='train':
            raise ValueError('Selected model lacks a declared training unit')
        verify_stage(workflow,stage)
        directory=cw.inside(row['trial_directory'])
        if directory!=stage_path(workflow,stage):raise ValueError('Selected trial directory differs from its sealed stage')
        checkpoint=directory/'selected.pt'
        if cw.digest(checkpoint)!=row['checkpoint_sha256']:raise ValueError('Selected checkpoint changed')
        checked.append((row,checkpoint))
    estimands=aggregate.get('estimands') or scientific_protocol(workflow)['evaluation']
    output.mkdir(parents=True,exist_ok=False)
    frozen_records=[]
    for row,checkpoint in checked:
        relative=Path('checkpoints')/pw.canonical_hash([row['family'],row.get('variant_id'),row.get('data_count'),row['target_kind'],row['seed']])/'selected.pt'
        destination=output/relative;destination.parent.mkdir(parents=True,exist_ok=False)
        shutil.copyfile(checkpoint,destination)
        if cw.digest(destination)!=row['checkpoint_sha256']:raise ValueError('Checkpoint copy verification failed; preserve incomplete derivative')
        destination.chmod(0o444)
        frozen_records.append({**row,'origin_trial_directory':row['trial_directory'],
            'trial_directory':str(destination.parent),'checkpoint':relative.as_posix()})
    frozen={'schema':'tdn.response-freeze/v1','created_at':cw.now(),'source_tree_sha256':source_tree,
        'source_sha256':workflow['source_sha256'],'protocol_sha256':pw.canonical_hash(scientific_protocol(workflow)),
        'origin_workflow':str(Path(workflow['run_dir'])/MANIFEST),'origin_workflow_sha256':cw.digest(Path(workflow['run_dir'])/MANIFEST),
        'software':cw.read_json(Path(workflow['run_dir'])/'state/submission-software.json'),
        'selection_split':'validation','selection_sha256':cw.digest(selected_path),'aggregate_sha256':cw.digest(aggregate_path),
        'models':frozen_records,'estimands':estimands,'confirmation_parent_count':128,
        'confirmation_root_seed':int(pw.canonical_hash([workflow['run_id'],cw.now(),'fresh-confirmation'])[:15],16),
        'evaluation':scientific_protocol(workflow)['evaluation'],
        'normalization':'Stored in each state-dict checkpoint; frozen training statistics only',
        'target_contracts':scientific_protocol(workflow)['targets'],'precision':scientific_protocol(workflow)['precision'],
        'deployment_policy':{'schedules':scientific_protocol(workflow)['evaluation']['frozen_schedules'],
            'cache':'same-state only; physical macrosteps refresh encodings','timing':scientific_protocol(workflow)['cost'],
            'fallback':'No trained acceptance/fallback policy; any oracle schedule remains descriptive'},
        'source_science_mutated':False,'confirmation_truth_read':False}
    cw.atomic_json(output/'freeze.json',frozen);(output/'freeze.json').chmod(0o444)
    verified=read_freeze(output/'freeze.json')
    print(json.dumps({'freeze':str(output/'freeze.json'),'sha256':verified['sha256'],
        'models':len(frozen_records),'confirmation_submitted':False},indent=2))
    return verified


def comment(workflow,stage):
    return f"tdn-response:{workflow['run_id']}:{stage}:{workflow['slurm_profile_sha256']}"


def scheduler_args(workflow, stage, dependency=None):
    base,site,res = Path(workflow['run_dir']),workflow['slurm_profile'],resource_for(workflow,stage)
    args = ['sbatch','--parsable',f"--partition={res['partition']}",'--nodes=1','--ntasks=1',
        f'--job-name=tdn-response-{stage}',f'--comment={comment(workflow,stage)}',
        f"--cpus-per-task={res['cpus']}",f"--mem={res['mem_gib']}G",f"--time={res['walltime']}",
        '--signal=USR1@120','--export=ALL',f'--chdir={ROOT}','--open-mode=append','--kill-on-invalid-dep=yes',
        f"--output={base/'logs'/(stage+'-%j.out')}",f"--error={base/'logs'/(stage+'-%j.err')}",
        f"--gres={site['gpu_gres'] if is_gpu_stage(workflow,stage) else 'none'}"]
    if site.get('account'):
        args.append(f"--account={site['account']}")
    if dependency:
        if not re.fullmatch(r'(?:afterok|afterany):[1-9][0-9]*(?::[1-9][0-9]*)*(?:,(?:afterok|afterany):[1-9][0-9]*(?::[1-9][0-9]*)*)*',dependency):
            raise ValueError('Invalid scheduler dependency')
        if stage=='report' and not re.fullmatch(r'afterany:[1-9][0-9]*(?::[1-9][0-9]*)*',dependency):
            raise ValueError('Report must run after any scientific outcome')
        args.append('--dependency='+dependency)
    return args+[str(ROOT/'scripts/response_worker.sh')]


def job_dependency(workflow, stage, jobs):
    """Science failure blocks only descendants; resource serialization waits for any outcome."""
    if not jobs:
        return None
    if stage=='report':
        return 'afterany:'+':'.join(row['job_id'] for row in jobs)
    if units(workflow)[stage]['kind']=='diagnostic_metrics':
        return 'afterany:'+':'.join(row['job_id'] for row in jobs)
    by_stage = {row['stage']:row['job_id'] for row in jobs}
    required = [by_stage[name] for name in units(workflow)[stage]['dependencies'] if name in by_stage]
    # Physical eight-core desktop: resource serialization is independent of scientific success.
    previous = jobs[-1]['job_id']
    groups=[]
    if required:
        groups.append('afterok:'+':'.join(required))
    if previous and previous not in required:
        groups.append('afterany:'+previous)
    return ','.join(groups) or None


def check_capacity(workflow):
    policies={}
    for partition in dict.fromkeys(row['partition'] for row in workflow['resources'].values()):
        fields=dict(re.findall(r'(\w+)=([^\s]+)',cw.command(['scontrol','show','partition',partition,'-o']).stdout))
        if fields.get('PartitionName')!=partition or fields.get('State')!='UP':
            raise ValueError('Configured partition is not UP')
        resources=[row for row in workflow['resources'].values() if row['partition']==partition]
        limit=fields.get('MaxTime','UNLIMITED')
        if limit not in ('UNLIMITED','INFINITE') and wall_seconds(limit)<max(wall_seconds(row['walltime']) for row in resources):
            raise ValueError('Partition walltime is below response request')
        candidates=[]
        for line in cw.command(['sinfo','-N','-h','-p',partition,'-o','%N|%c|%m|%G']).stdout.strip().splitlines():
            values=line.strip().split('|')
            if len(values)!=4 or not values[1].isdigit() or not values[2].isdigit():
                raise ValueError('Cannot verify node capacity')
            candidates.append({'node':values[0],'cpus':int(values[1]),'mem_mib':int(values[2]),'gres':values[3]})
        for stage,res in workflow['resources'].items():
            if res['partition']==partition and not any(node['cpus']>=res['cpus'] and node['mem_mib']>=res['mem_gib']*1024 and
                    (not is_gpu_stage(workflow,stage) or re.search(r'(?:^|,)gpu:(?:[^,:]+:)?[1-9][0-9]*(?:\([^)]*\))?(?:,|$)',node['gres'])) for node in candidates):
                raise ValueError(f'{stage}: request exceeds configured CPU/memory/GPU capacity')
        policies[partition]={'partition':fields,'nodes':candidates}
    return policies


def clean_environment():
    return {key:value for key,value in os.environ.items() if not key.startswith(('SBATCH_','SRUN_','RAVEN_','TDN_PREMIX_','TDN_CONSISTENCY_','TDN_AGENDA_','TDN_ROADMAP_','TDN_FRONTIER_','TDN_PORTFOLIO_','TDN_ADVANCE_','TDN_RESOLVE_','TDN_RESPONSE_'))
        and not key.startswith('TDN_REQUIRE_')
        and key not in ('CARC_ACCOUNT','TORCH_CUDA_ARCH_LIST','TDN_TOWER_DIR')}


def submission_environment(workflow,stage):
    env=clean_environment()
    env.update(TDN_EXECUTION_MODE='desktop-slurm',TDN_PROJECT_ROOT=str(ROOT),TDN_REPO_ROOT=str(ROOT),
        TDN_SLURM_CONFIG=workflow['slurm_profile_path'],TDN_RESPONSE_WORKFLOW=str(Path(workflow['run_dir'])/MANIFEST),
        TDN_RESPONSE_STAGE=stage,TDN_RESPONSE_DEVICE=units(workflow)[stage]['device'],
        TDN_RESPONSE_CPUS=str(resource_for(workflow,stage)['cpus']),
        TDN_FEDORA_GPU_GRES=workflow['slurm_profile']['gpu_gres'],TORCH_VERSION=workflow['torch_version'])
    return env


def submit(workflow, *, plan_only=False, bridge=None):
    print(f"Fedora response: {workflow['run_id']} ({workflow['profile']})\nRun: {workflow['run_dir']}")
    print(scientific_protocol(workflow)['scientific_scope'] + '. Serial desktop resources, no pending-job cap. Report afterany.')
    pending=pending_stages(workflow)
    reserved=sum(wall_seconds(resource_for(workflow,stage)['walltime']) for stage in pending)
    gpu_reserved=sum(wall_seconds(resource_for(workflow,stage)['walltime']) for stage in pending if is_gpu_stage(workflow,stage))
    numerical=sum(units(workflow)[stage]['seconds'] for stage in pending)
    print(f'Budget ceilings: {len(pending)} jobs, {reserved/3600:.2f} allocation-hours, '
          f'{gpu_reserved/3600:.2f} GPU allocation-hours, {numerical/3600:.2f} numerical hours. '
          'These are configured limits, not measured costs.')
    protocol=scientific_protocol(workflow)
    calibrated=protocol.get('allowance',{}).get('calibrated_forecast')
    print('Native calibration: '+json.dumps({'status':'MEASURED' if calibrated else 'NOT_AVAILABLE',
        'run_id':calibrated.get('run_id') if calibrated else None,
        'seconds_per_update':calibrated.get('seconds_per_update') if calibrated else None},sort_keys=True))
    for trial in protocol.get('trials',{}).values():
        print(f"Trial {trial['id']}: {trial['target_kind']}/{trial['family']}/{trial['variant_id']} "
              f"phase={trial['phase']} seed={trial['seed']} parents={trial['data_count']} "
              f"updates={trial['updates']} science_allowance_seconds={trial['total_seconds']} "
              f"estimated_operation_factor={trial.get('forecast_operation_factor',1):.3g}")
    for stage in pending_stages(workflow):
        print(shlex.join(scheduler_args(workflow,stage)))
    if plan_only:
        print('PLAN ONLY: no writes, scheduler calls or numerical work.')
        return workflow
    from tdn.analysis.response.protocol import validate_protocol
    validate_protocol(scientific_protocol(workflow))
    if not scientific_protocol(workflow).get('gpu_test_cases'): raise ValueError('Response native CUDA case manifest is empty')
    verify_external_bindings(workflow)
    fw.controller_policy()
    policies=check_capacity(workflow)
    with rw.acquire_venv_lock():
        software=rw.software_report(workflow)
    with cw.controller_lock():
        base=Path(workflow['run_dir']);base.mkdir(parents=True,exist_ok=False);(base/'logs').mkdir()
        for path,value in ((base/MANIFEST,workflow),(base/'protocol.json',declaration(workflow)),(base/'slurm-profile.json',workflow['slurm_profile'])):
            cw.atomic_json(path,value);path.chmod(0o444)
        if bridge:
            cw.atomic_json(base/'recovery.json',bridge);(base/'recovery.json').chmod(0o444)
            verify_recovery_bridge(workflow)
        cw.atomic_json(base/'state/slurm-policy.json',policies)
        cw.atomic_json(base/'state/submission-software.json',software)
        jobs=[];cw.atomic_json(base/'jobs.json',jobs)
        pointer='.fedora-response-diagnostic-latest.json' if workflow['profile'].startswith('diagnostic') else POINTER
        cw.atomic_json(ROOT/'runs'/pointer,{'run_id':workflow['run_id']})
        def queue(stage):
            load(base/MANIFEST)
            dependency=job_dependency(workflow,stage,jobs)
            response=cw.command(scheduler_args(workflow,stage,dependency),env=submission_environment(workflow,stage))
            match=re.fullmatch(r'([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?',response.stdout.strip())
            if not match or any(row['job_id']==match.group(1) for row in jobs):
                raise ValueError('Ambiguous or duplicate sbatch ID; inspect Slurm before retrying')
            jobs.append({'stage':stage,'job_id':match.group(1),'dependency':dependency,'submitted_at':cw.now()})
            cw.atomic_json(base/'jobs.json',jobs)
            print(f"TDN_RESPONSE_{stage.upper().replace('-','_')}_JOB_ID={match.group(1)}",flush=True)
        for stage in pending_stages(workflow):
            try:
                queue(stage)
            except BaseException as error:
                failure={'status':'FAILED','stage':stage,'error':str(error),'submitted_jobs':jobs,'updated_at':cw.now()}
                if stage!='report' and not isinstance(error,(KeyboardInterrupt,SystemExit)):
                    try: queue('report')
                    except Exception as report_error: failure['aggregation_submission_error']=str(report_error)
                cw.atomic_json(base/'state/submission.json',failure)
                raise
        cw.atomic_json(base/'state/submission.json',{'status':'COMPLETED','updated_at':cw.now()})
    return workflow


def stage_cli():
    spec=importlib.util.spec_from_file_location('response_stage_cli',ROOT/'scripts/response.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def verify_stage(workflow,stage):
    checked=stage_cli().verify_execution(stage_path(workflow,stage),scientific_protocol(workflow),source_tree_sha256=workflow['source_tree_sha256'])
    execution=checked['execution']
    expected={'stage':stage,'profile':workflow['profile'],'execution_mode':'desktop-slurm',
        'device':units(workflow)[stage]['device'],'slurm_profile_sha256':workflow['slurm_profile_sha256'],
        'workflow_protocol_sha256':workflow['protocol_sha256']}
    if any(execution.get(key)!=value for key,value in expected.items()):
        raise ValueError(f'{stage}: execution differs from frozen workflow')
    return checked


def verify_software(workflow,software):
    if cw.read_json(Path(workflow['run_dir'])/'state/submission-software.json')!=software:
        raise ValueError('Software differs from frozen submission')


def verify_prerequisites(workflow,software,stage):
    hashes={}
    verify_recovery_bridge(workflow)
    for prior in ancestors(workflow,stage):
        verify_stage(workflow,prior)
        hashes[prior]=cw.digest(stage_path(workflow,prior)/'workflow-seal.json')
    return hashes


def verify_allocation(workflow,stage):
    site,res=workflow['slurm_profile'],resource_for(workflow,stage)
    if load_profile(root=ROOT)!=site:
        raise ValueError('Active Slurm profile differs')
    allocation=runtime_allocation('cuda' if is_gpu_stage(workflow,stage) else 'cpu',profile=site,root=ROOT)
    job,fields=allocation['job_id'],allocation['job']
    expected={'JobId':job,'JobName':f'tdn-response-{stage}','Comment':comment(workflow,stage),
        'JobState':'RUNNING','Partition':res['partition'],'WorkDir':str(ROOT)}
    if any(fields.get(key)!=value for key,value in expected.items()) or fields.get('UserId','').split('(')[0]!=site['user']:
        raise ValueError('Allocation is not this running response unit')
    tres=dict(item.split('=',1) for item in fields.get('AllocTRES','').split(',') if '=' in item)
    if tres.get('cpu')!=str(res['cpus']) or memory_mib(tres.get('mem',''))!=res['mem_gib']*1024:
        raise ValueError('Allocated CPU/memory differs from frozen budget')
    if wall_seconds(fields.get('TimeLimit',''))!=wall_seconds(res['walltime']):
        raise ValueError('Allocated walltime differs from frozen budget')
    step=dict(item.split('=',1) for item in allocation['step'].get('TRES','').split(',') if '=' in item)
    if step.get('cpu')!=str(res['cpus']):
        raise ValueError('Task CPU count differs from unit budget')
    if not is_gpu_stage(workflow,stage) and any(
            key.startswith('gres/gpu') and value not in ('0','') for fields in (tres,step) for key,value in fields.items()):
        raise ValueError('CPU-only response unit unexpectedly allocated a GPU')
    recorded=[row for row in cw.read_json(Path(workflow['run_dir'])/'jobs.json') if row.get('stage')==stage]
    if len(recorded)!=1 or recorded[0].get('job_id')!=job:
        raise ValueError('Allocation differs from exact recorded submission')


def junit_path(workflow,stage):
    return cw.inside(Path(workflow['run_dir'])/'reporter-tests'/stage/'tests.xml')


def worker_commands(workflow,stage):
    python=str(cw.inside(ROOT/'.venv/bin')/'python');base=Path(workflow['run_dir']);commands=[]
    gpu=is_gpu_stage(workflow,stage)
    if stage=='report':
        commands.append(('accounting',[python,str(ROOT/'scripts/response_workflow.py'),'snapshot-accounting','--workflow',str(base/MANIFEST)]))
    if stage=='audit' or gpu:
        tests=[ROOT/relative for relative in scientific_protocol(workflow).get('gpu_test_files',['tests/test_response_gpu.py'])] if gpu else sorted((ROOT/'tests').glob('test_response*.py'))+[ROOT/'tests/test_desktop_slurm_runtime.py']
        if not tests or any(not path.is_file() for path in tests):
            raise ValueError('Response mandatory correctness tests are missing')
        if gpu:
            commands.append(('preflight',[python,str(ROOT/'scripts/fedora_gpu_preflight.py'),'--output',str(base/f'{stage}-gpu-preflight.json')]))
        commands.append(('gpu-tests' if gpu else 'tests',[python,'-m','pytest','-q','-m','gpu' if gpu else 'not gpu',*map(str,tests),
            '--basetemp',str(base/f'{stage}-pytest-work'),'-o',f"cache_dir={base/(stage+'-pytest-cache')}",'--junitxml',str(junit_path(workflow,stage))]))
        if gpu:
            commands.append(('check-gpu-tests',[python,str(ROOT/'scripts/response_workflow.py'),'check-gpu-tests','--junit',str(junit_path(workflow,stage)), '--protocol',workflow['protocol_path']]))
    command=[python,str(ROOT/'scripts/response.py'),'--stage',stage,'--profile',workflow['profile'],'--protocol',workflow['protocol_path'],'--run-dir',str(base/stage),'--device','cuda' if gpu else 'cpu']
    for prior in ancestors(workflow,stage):
        command += ['--prerequisite-dir',f'{prior}={stage_path(workflow,prior)}']
    bridge=recovery_bridge(workflow)
    if bridge and stage in bridge['resume_paths']:
        command+=['--resume-from',bridge['resume_paths'][stage]]
    commands.append(('experiment',command))
    return commands


def workflow_environment(workflow,stage):
    return {'TDN_RESPONSE_PROTOCOL_SHA256':workflow['protocol_sha256'],'TDN_RESPONSE_WORKFLOW':str(Path(workflow['run_dir'])/MANIFEST),
        'TDN_RESPONSE_STAGE':stage,'TDN_RESPONSE_CPUS':str(resource_for(workflow,stage)['cpus'])}


def worker(workflow,stage):
    if stage not in pending_stages(workflow):
        raise ValueError('Unknown or inherited response unit cannot run')
    native_ctx=None; native_runtime=None; native_started=None
    previous_raven=os.environ.get('TDN_RESPONSE_RAVEN_DIR')
    backend=SimpleNamespace(**{name:globals()[name] for name in ('verify_allocation','load','begin_report','worker_commands',
        'verify_stage','prepare_test_environment','workflow_environment','finish_report','verify_software')},
        gpu_junit_path=lambda workflow:junit_path(workflow,stage),workflow_filename=MANIFEST,workflow_label='response',
        stage_variable='TDN_RESPONSE_STAGE',prerequisite_stages=tuple(name for name in physical_stages(workflow) if name not in ('audit','report')),
        verify_prerequisites=lambda workflow,software:verify_prerequisites(workflow,software,stage))
    if workflow['profile'].startswith('diagnostic'):
        def begin_native_report(bound,unit):
            nonlocal native_ctx,native_started
            import time
            from tdn.analysis.response.diagnostic_metrics import start_attempt
            from tdn.runtime.metadata import software_metadata
            # The shared worker verifies the real allocation before this callback.
            native_ctx=SimpleNamespace(path=Path(bound['run_dir'])/unit,stage=unit,
                protocol=scientific_protocol(bound),unit=units(bound)[unit],device=units(bound)[unit]['device'])
            native_started=time.monotonic()
            start_attempt(native_ctx,job_id=os.environ['SLURM_JOB_ID'],
                sourcehash=software_metadata()['source_tree_sha256'],worker_managed=True)
            return begin_report(bound,unit)
        def finish_native_report(bound,unit,report,**kwargs):
            nonlocal native_runtime
            native_runtime=kwargs.get('runtime_seconds')
            return finish_report(bound,unit,report,**kwargs)
        backend.begin_report=begin_native_report
        backend.finish_report=finish_native_report
    try:
        return pw.worker(workflow,stage,backend=backend)
    except BaseException as error:
        path=pw.state_path(workflow,stage)
        if not path.exists():
            cw.atomic_json(path,{'status':'FAILED','stage':'startup','exit_code':2,'error':str(error),
                'source_sha256':workflow['source_sha256'],'protocol_sha256':workflow['protocol_sha256'],
                'updated_at':cw.now(),'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'slurm_step_id':os.environ.get('SLURM_STEP_ID')})
        raise
    finally:
        if native_ctx is not None:
            try:
                import time
                from tdn.analysis.response.diagnostic_metrics import finish_attempt
                state=cw.read_json(pw.state_path(workflow,stage))
                body_path=native_ctx.path/'summary.json'
                body=cw.read_json(body_path) if body_path.is_file() else {}
                if state.get('status') not in ('COMPLETED','FAILED','INTERRUPTED'):
                    raise ValueError('Native Raven finalization requires the recorded terminal worker state')
                finish_attempt(native_ctx,{**body,'status':state['status'],
                    'science_body_status':body.get('status','NOT_STARTED'),
                    'science_body_elapsed_seconds':body.get('elapsed_seconds'),
                    'native_worker_finalized':True,'native_worker_exit_code':state.get('exit_code'),
                    'native_worker_runtime_seconds':native_runtime if native_runtime is not None else time.monotonic()-native_started,
                    'native_worker_error':state.get('error')})
            except Exception as raven_error:
                terminal=cw.read_json(pw.state_path(workflow,stage))
                print(f'Native Raven finalization failed: {raven_error}',file=sys.stderr)
                if terminal.get('status')=='COMPLETED':
                    cw.atomic_json(pw.state_path(workflow,stage),{**terminal,'status':'FAILED','exit_code':2,
                        'error':'Native Raven finalization failed: '+str(raven_error),'updated_at':cw.now()})
                    raise ValueError('Native Raven finalization failed') from raven_error
            finally:
                if previous_raven is None:os.environ.pop('TDN_RESPONSE_RAVEN_DIR',None)
                else:os.environ['TDN_RESPONSE_RAVEN_DIR']=previous_raven


def require_terminal_origin(workflow):
    jobs=cw.read_json(Path(workflow['run_dir'])/'jobs.json')
    if not jobs or len({row['stage'] for row in jobs})!=len(jobs) or len({row['job_id'] for row in jobs})!=len(jobs):
        raise ValueError('Recovery requires unique origin jobs')
    states={}
    for row in jobs:
        if row['stage'] not in units(workflow) or not pw.JOB.fullmatch(row['job_id']):
            raise ValueError('Invalid origin allocation identity')
        state=cw.scheduler_state(row['job_id'])
        if state not in TERMINAL_JOB_STATES:
            raise ValueError(f"Recovery origin job {row['job_id']} is {state}; all jobs must be terminal")
        states[row['job_id']]=state
    return states


def verify_recovery_bridge(workflow):
    bridge=recovery_bridge(workflow)
    if not bridge:
        return {}
    if bridge['source_sha256']!=workflow['source_sha256'] or bridge['protocol_sha256']!=workflow['protocol_sha256']:
        raise ValueError('Recovery source or protocol differs')
    for stage,path in bridge['stage_paths'].items():
        checked=stage_cli().verify_execution(path,scientific_protocol(workflow),source_tree_sha256=workflow['source_tree_sha256'])
        if cw.digest(Path(path)/'workflow-seal.json')!=bridge['seals'][stage]:
            raise ValueError('Completed origin unit changed')
        if checked['execution']['stage']!=stage:
            raise ValueError('Inherited unit scope differs')
    for stage,path in bridge['resume_paths'].items():
        if stage_cli().resume_fingerprint(path)!=bridge['resume_fingerprints'][stage]:
            raise ValueError('Interrupted origin artifacts changed')
    return bridge['stage_paths']


def recover(args):
    origin=load(workflow_path(args.run))
    run_id=args.run_id or 'fedora-response-recovery-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    workflow=prepare_from_protocol(scientific_protocol(origin),run_id)
    fw.controller_policy();states=require_terminal_origin(origin)
    if any(workflow[key]!=origin[key] for key in ('source_sha256','source_tree_sha256','protocol_sha256','slurm_profile_sha256')):
        raise ValueError('Recovery requires identical source, protocol and hardware profile')
    with rw.acquire_venv_lock():
        software=rw.software_report(workflow)
    if software!=cw.read_json(Path(origin['run_dir'])/'state/submission-software.json'):
        raise ValueError('Recovery software differs')
    completed,resume,seals,fingerprints={},{},{},{}
    for stage in physical_stages(origin):
        if units(origin)[stage]['kind'] in ('report','diagnostic_metrics'): continue
        path=stage_path(origin,stage)
        if not path.exists(): continue
        if (path/'workflow-seal.json').exists():
            verify_stage(origin,stage);completed[stage]=str(path);seals[stage]=cw.digest(path/'workflow-seal.json')
        elif (units(origin)[stage]['kind'] in ('train','references') and (path/'execution.json').is_file()
                and not any((path/name).exists() for name in ('COMPLETED','science_manifest.json'))):
            record=cw.read_json(path/'stage.json')
            if record.get('status') in ('FAILED','INTERRUPTED','INCOMPLETE') and set(ancestors(origin,stage))<=set(completed):
                resume[stage]=str(path);fingerprints[stage]=stage_cli().resume_fingerprint(path)
    for stage in completed:
        if not set(ancestors(origin,stage))<=set(completed):
            raise ValueError('Cannot reuse a completed unit whose dependencies are not sealed')
    previous=recovery_bridge(origin)
    roots=set(previous.get('origin_roots',[])) if previous else set()
    roots.add(origin['run_dir'])
    bridge={'schema':'tdn.response-recovery/v1','origin_workflow_path':str(Path(origin['run_dir'])/MANIFEST),
        'origin_roots':sorted(roots),'source_sha256':workflow['source_sha256'],'protocol_sha256':workflow['protocol_sha256'],
        'software':software,'stage_paths':completed,'resume_paths':resume,'seals':seals,'resume_fingerprints':fingerprints,
        'origin_job_states':states,'cost_scope':'Origin attempts retained; resumed work charged to new allocations'}
    serialized=json.dumps(bridge,indent=2,sort_keys=True,allow_nan=False)+'\n'
    workflow['recovery']={'manifest_path':str(Path(workflow['run_dir'])/'recovery.json'),'sha256':hashlib.sha256(serialized.encode()).hexdigest(),
        'stage_paths':completed,'resume_paths':resume}
    validate(workflow)
    return submit(workflow,plan_only=args.plan,bridge=bridge)


def paths(workflow):
    base=Path(workflow['run_dir'])
    for file in sorted((base/'report').rglob('*')) if (base/'report').exists() else ():
        if file.is_file() and file.suffix in ('.json','.jsonl','.csv','.md','.txt','.pdf','.png','.html','.gz'):
            name=re.sub('[^A-Z0-9]','_',str(file.relative_to(base/'report')).upper())
            print(f'TDN_RESPONSE_{name}={file}')
    for row in cw.read_json(base/'jobs.json'):
        for report in cw.tower_reports_for_job(workflow,row['job_id']):
            print(f"{row['stage']} job {row['job_id']}\nTDN_TOWER_DIR={report}\nTDN_TOWER_METRICS={report/'metrics.jsonl'}")


def status(workflow, *, json_output=False):
    if json_output:
        base=Path(workflow['run_dir']);jobs={row['stage']:row for row in cw.read_json(base/'jobs.json')}
        result={'run_id':workflow['run_id'],'profile':workflow['profile'],'units':{},'failed_units':[],
            'terminal':True,'completed':True,'report_complete':False}
        for stage in physical_stages(workflow):
            inherited=stage in workflow.get('recovery',{}).get('stage_paths',{})
            job=jobs.get(stage);state='COMPLETED' if inherited else 'NOT_SUBMITTED'
            if job:
                try:state=cw.scheduler_state(job['job_id']).split()[0].rstrip('+')
                except (OSError,ValueError):state='UNKNOWN'
                if state not in TERMINAL_JOB_STATES:result['terminal']=False
            worker_path=pw.state_path(workflow,stage)
            worker=cw.read_json(worker_path) if worker_path.is_file() else {}
            path=stage_path(workflow,stage);summary_path=path/'summary.json'
            summary=cw.read_json(summary_path) if summary_path.is_file() else {}
            has_seal=(path/'workflow-seal.json').is_file() and (path/'COMPLETED').is_file()
            complete=state=='COMPLETED' and (inherited or worker.get('status')=='COMPLETED') and has_seal
            result['completed'] &= complete
            if state in TERMINAL_JOB_STATES-{'COMPLETED'} or state=='NOT_SUBMITTED' or (state=='COMPLETED' and not complete):
                result['failed_units'].append(stage)
            result['units'][stage]={'job_id':job['job_id'] if job else None,'scheduler_state':state,
                'worker_status':worker.get('status','INHERITED' if inherited else 'UNKNOWN'),
                'artifact_validation':'SEALED_UNVERIFIED' if has_seal else 'MISSING',
                'scientific_outcome':summary.get('scientific_outcome','NA')}
            if stage=='report':result['report_complete']=complete
        print(json.dumps(result,sort_keys=True,allow_nan=False))
        return result
    base=Path(workflow['run_dir']);jobs={row['stage']:row for row in cw.read_json(base/'jobs.json')}
    print(f"Fedora response {workflow['run_id']} ({workflow['profile']})\nRun: {base}")
    for stage in physical_stages(workflow):
        if stage in workflow.get('recovery',{}).get('stage_paths',{}):
            print(f'{stage}: INHERITED completed evidence')
        elif stage in jobs:
            try: state=cw.scheduler_state(jobs[stage]['job_id'])
            except (OSError,ValueError): state='UNKNOWN (scheduler unavailable)'
            print(f"{stage} job {jobs[stage]['job_id']}: {state}")
        else: print(f'{stage}: NOT_SUBMITTED')
        file=stage_path(workflow,stage)/'summary.json'
        if file.exists():
            summary=cw.read_json(file);print(f"  computational: {summary.get('status','UNKNOWN')}; scientific: {summary.get('scientific_outcome','NA')}")
            if summary.get('status')=='COMPLETED':
                try: verify_stage(workflow,stage);print('  sealed artifacts: VERIFIED')
                except (OSError,ValueError,KeyError,TypeError,ImportError) as error: print(f'  artifact verification: INVALID ({error})')
        state_path=pw.state_path(workflow,stage)
        if state_path.exists() and (record:=cw.read_json(state_path)).get('error'):
            print('  worker error: '+record['error'])


def list_workflows():
    """Inventory coordinator metadata without importing scientific dependencies."""
    for path in sorted((ROOT/'runs').glob('*/'+MANIFEST)):
        try:
            if path.is_symlink():
                raise ValueError('Coordinator manifest is a symlink')
            value=cw.read_json(cw.inside(path))
            if value.get('kind')!='desktop-slurm-response':
                raise ValueError('Unexpected coordinator identity')
            print(f"{value['run_id']}\t{value['profile']}\t{path.parent}")
        except (OSError,ValueError,KeyError,TypeError) as error:
            print(f'{path.parent.name}\tINVALID\t{error}')


def logs(workflow,lines):
    if not 1<=lines<=100000: raise ValueError('Choose 1–100000 lines')
    for path in sorted((Path(workflow['run_dir'])/'logs').glob('*')):
        if cw.inside(path).is_file():
            print('\n'+str(path))
            with path.open(errors='replace') as stream: print(''.join(deque(stream,maxlen=lines)),end='')


def validate_results(workflow):
    result={'validation':'response_sealed_artifacts','units':{}};failed=False
    for stage in physical_stages(workflow):
        try:
            if not stage_path(workflow,stage).is_dir():
                raise FileNotFoundError('Unit has not produced artifacts')
            verify_stage(workflow,stage)
            result['units'][stage]={'validation':'VERIFIED','scientific_outcome':cw.read_json(stage_path(workflow,stage)/'summary.json').get('scientific_outcome','NA')}
        except (OSError,ValueError,KeyError,TypeError,ImportError) as error:
            result['units'][stage]={'validation':'INVALID' if stage_path(workflow,stage).exists() else 'MISSING','scientific_outcome':'NA','error':str(error)};failed=True
    print(json.dumps(result,indent=2))
    if failed: raise ValueError('Some response units are missing or invalid; report retains NA evidence')
    return result


def parser():
    result=argparse.ArgumentParser(description=__doc__);commands=result.add_subparsers(dest='command',required=True)
    commands.add_parser('list',help='List Response only, using coordinator metadata')
    for name in ('plan','run'):
        item=commands.add_parser(name);item.add_argument('--run-id')
        group=item.add_mutually_exclusive_group()
        group.add_argument('--smoke',action='store_true');group.add_argument('--development',action='store_true');group.add_argument('--confirmation',action='store_true')
        group.add_argument('--diagnostic-smoke',action='store_true');group.add_argument('--diagnostic',action='store_true')
        item.add_argument('--pilot',nargs='?',const=True,default=None,help='Pilot profile, or explicit pilot ID with --development')
        item.add_argument('--calibration',help='Completed native smoke ID with64-update workload calibration')
        item.add_argument('--freeze',type=Path);item.add_argument('--trial-seconds',action='append',default=[],metavar='FAMILY=SECONDS')
        item.add_argument('--chunk-seconds',type=int,default=1500)
        item.add_argument('--updates',action='append',default=[],metavar='FAMILY=COUNT')
        item.add_argument('--targets',nargs='+',choices=('fd_same_grid','galerkin_same_grid','galerkin_projected_fine'))
    item=commands.add_parser('recover');item.add_argument('run');item.add_argument('--run-id');item.add_argument('--plan',action='store_true')
    for name in ('status','logs','paths','validate','collect','gate','decision'):
        item=commands.add_parser(name);item.add_argument('run')
        if name=='logs':item.add_argument('--lines',type=int,default=200)
        if name=='collect':item.add_argument('--no-accounting',action='store_true')
        if name=='status':item.add_argument('--json',action='store_true',dest='json_output')
    item=commands.add_parser('freeze');item.add_argument('run');item.add_argument('--output-dir',type=Path,required=True)
    item=commands.add_parser('worker');item.add_argument('--workflow',type=Path,required=True);item.add_argument('--stage',required=True)
    item=commands.add_parser('check-gpu-tests');item.add_argument('--junit',type=Path,required=True);item.add_argument('--protocol',type=Path)
    item=commands.add_parser('snapshot-accounting');item.add_argument('--workflow',type=Path,required=True)
    return result


def main(argv=None):
    args=parser().parse_args(argv)
    try:
        if args.command=='list': list_workflows()
        elif args.command in ('plan','run'): submit(prepare(args),plan_only=args.command=='plan')
        elif args.command=='recover': recover(args)
        elif args.command=='freeze': freeze_results(load(workflow_path(args.run)),args.output_dir)
        elif args.command=='worker': return worker(load(args.workflow,verify=False),args.stage)
        elif args.command=='check-gpu-tests':
            protocol=cw.read_json(cw.inside(args.protocol)) if args.protocol else None
            protocol=protocol.get('scientific_protocol',protocol) if protocol else None
            print(json.dumps(check_gpu_tests(args.junit,protocol=protocol),sort_keys=True))
        elif args.command=='snapshot-accounting': scheduler_accounting(load(args.workflow,verify=False))
        else:
            workflow=load(workflow_path(args.run),verify=False)
            if args.command=='logs': logs(workflow,args.lines)
            elif args.command=='collect': collect(workflow,accounting=not args.no_accounting)
            elif args.command=='validate': validate_results(workflow)
            elif args.command=='status': status(workflow,json_output=args.json_output)
            elif args.command=='decision':
                if not workflow['profile'].startswith('diagnostic'):
                    raise ValueError('Decision command requires the separate diagnostic workflow')
                verify_stage(workflow,'diagnostic-metrics')
                print(json.dumps(cw.read_json(stage_path(workflow,'diagnostic-metrics')/'diagnostic_decision.json'),indent=2))
            elif args.command=='gate':
                result=gate_results(workflow);print(json.dumps(result,indent=2))
                if not result['eligible_for_development']:return 3
            else: globals()[args.command](workflow)
    except rw.WorkerFailure as error:
        print(f'TDN response: {error}',file=sys.stderr);return error.exit_code
    except ImportError as error:
        print(f'TDN response: scientific dependency unavailable ({error}). '
              'Use bash scripts/fedora_response.sh so the project .venv is selected; '
              'run its setup command if that venv is missing.',file=sys.stderr)
        return 2
    except (ValueError,OSError,KeyError,TypeError) as error:
        print(f'TDN response: {error}',file=sys.stderr);return 2
    return 0
def wall_seconds(value):
    if not isinstance(value, str) or not re.fullmatch(r"(?:[0-9]+-)?[0-9]{2}:[0-9]{2}:[0-9]{2}", value):
        raise ValueError("Invalid bounded allocation walltime")
    parts = value.split("-")
    hours, minutes, seconds = map(int, parts[-1].split(":"))
    if minutes >= 60 or seconds >= 60:
        raise ValueError("Invalid allocation minutes/seconds")
    return (int(parts[0]) * 86400 if len(parts) == 2 else 0) + hours * 3600 + minutes * 60 + seconds

def execution_software():
    """Read execution metadata from this checkout's installed Python venv."""
    python = str(cw.inside(ROOT / ".venv" / "bin") / "python")
    return json.loads(cw.command([python, "-c",
        "import json; from tdn.runtime.metadata import software_metadata; print(json.dumps(software_metadata()))"]).stdout)

def memory_mib(value):
    match = re.fullmatch(r"([0-9]+)([KMGT]?)", value)
    if not match:
        raise ValueError("Cannot verify actual allocated host memory")
    amount = int(match.group(1))
    return amount * {"": 1, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024**2}[match.group(2)]

def check_gpu_tests(path, *, protocol=None):
    """Require every declared response CUDA case; no skips or filtered suites."""
    from collections import Counter
    from xml.etree import ElementTree
    from tdn.analysis.response.protocol import build_protocol
    GPU_TEST_CASES = build_protocol("smoke")["gpu_test_cases"]
    path = cw.inside(path)
    pw.validate_gpu_junit(path)
    tree = ElementTree.parse(path).getroot()
    cases = tree.findall(".//testcase")
    identities = [(case.get("classname"), case.get("name")) for case in cases]
    expected = [("tests.test_response_gpu", name) for name in GPU_TEST_CASES]
    if protocol is not None and protocol['profile'].startswith('diagnostic'):
        from tdn.analysis.response.protocol import validate_protocol
        from tdn.analysis.response.diagnostic_protocol import DIAGNOSTIC_GPU_CASES
        validate_protocol(protocol,require_binding=False)
        expected += [("tests.test_response_diagnostic_gpu",name) for name in DIAGNOSTIC_GPU_CASES]
        GPU_TEST_CASES = [*GPU_TEST_CASES,*DIAGNOSTIC_GPU_CASES]
    if not expected or Counter(identities) != Counter(expected) or len(set(identities)) != len(identities):
        raise ValueError("Response GPU readiness requires every unique mandatory declared CUDA case; filtered suites are invalid")
    if any(case.find(tag) is not None for case in cases for tag in ("failure", "error", "skipped")):
        raise ValueError("All mandatory response GPU tests must pass without failures, errors or skips")
    suites = [tree] if tree.tag == "testsuite" else tree.findall(".//testsuite")
    if any(int(suite.get("tests", -1)) != len(suite.findall("testcase")) for suite in suites):
        raise ValueError("Response GPU JUnit declared counts do not match the recorded cases")
    if tree.tag == "testsuites" and tree.get("tests") is not None and int(tree.get("tests")) != len(cases):
        raise ValueError("Response GPU JUnit aggregate count differs")
    return {"validation": "mandatory_response_gpu_suite", "test_cases": len(cases), "case_names": list(GPU_TEST_CASES)}

def begin_report(workflow, stage):
    base, site, resource = Path(workflow["run_dir"]), workflow["slurm_profile"], resource_for(workflow, stage)
    job = os.environ["SLURM_JOB_ID"]
    resources = {"partition": resource["partition"], "nodes": 1, "cpus": resource["cpus"],
                 "gpus": int(is_gpu_stage(workflow, stage)), "mem_bytes": resource["mem_gib"] * 1024**3,
                 "time_seconds": wall_seconds(resource["walltime"])}
    if site.get("account"):
        resources["account"] = site["account"]
    if is_gpu_stage(workflow, stage):
        resources["gpu_type"] = site["expected_gpu_name"]
    return cw.reporting_api().begin_report(base / stage, name=f"TDN/response/{stage}", script="scripts/response_worker.sh",
        parameters={"source_sha256": workflow["source_sha256"], "protocol_sha256": workflow["protocol_sha256"],
                    "slurm_profile_sha256": workflow["slurm_profile_sha256"], "stage": stage,
                    "profile": workflow["profile"], "execution_mode": "desktop-slurm"},
        resources=resources, job_id=job, report_parent=base / "tower",
        logs=[{"id": f"scheduler.{kind}", "path": str(base / "logs" / f"{stage}-{job}.{suffix}"),
               "label": f"Slurm {kind}", "group": "Scheduler"} for kind, suffix in (("stdout", "out"), ("stderr", "err"))],
        metadata={"workflow_id": workflow["run_id"], "phase": stage, "execution_scope": "one bounded Fedora Slurm allocation"})

def finish_report(workflow, stage, report, *, state, runtime_seconds, exit_code, software=None, error=None):
    from tdn.tower_analytics import publish_outputs
    base = Path(workflow["run_dir"])
    if cw.inside(report).parent != base / "tower":
        raise ValueError("Tower report is not this workflow's sidecar")
    if cw.read_json(report / "run.json").get("job_id") != os.environ["SLURM_JOB_ID"]:
        raise ValueError("Tower report belongs to a different scheduler job")
    sources = [path for path in (base / stage, base / "reporter-tests" / stage) if path.is_dir()]
    results = publish_outputs(report, sources)
    cw.reporting_api().finish_report(report, state=state, runtime_seconds=runtime_seconds, exit_code=exit_code, results=results,
        metadata={"workflow_id": workflow["run_id"], "phase": stage,
                  "execution_scope": "allocated worker elapsed time; excludes pending time",
                  **({"verified_software_sha256": pw.canonical_hash(software)} if software else {}),
                  **({"error": str(error)} if error is not None else {})})

def prepare_test_environment(env, name):
    for key in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
        env.pop(key, None)
    for key in tuple(env):
        if key.startswith(("TDN_REQUIRE_", "TDN_ADVANCE_", "TDN_RESOLVE_", "TDN_RESPONSE_", "TDN_PORTFOLIO_", "TDN_FRONTIER_", "TDN_ROADMAP_", "TDN_AGENDA_", "TDN_PREMIX_", "TDN_CONSISTENCY_")):
            env.pop(key)
    if name == "gpu-tests":
        env["TDN_REQUIRE_RESPONSE_GPU_TESTS"] = "1"
    if name == "tests":
        for key in ("TDN_EXECUTION_MODE", "TDN_SLURM_CONFIG"):
            env.pop(key, None)

def scheduler_accounting(workflow, *, enabled=True, runner=None):
    """Snapshot exact submitted jobs; unavailable accounting never invents cost."""
    base = cw.inside(workflow["run_dir"])
    jobs = cw.read_json(base / "jobs.json")
    if (not isinstance(jobs, list) or any(not isinstance(row, dict) or row.get("stage") not in physical_stages(workflow)
            or not pw.JOB.fullmatch(row.get("job_id", "")) for row in jobs)
            or len({row["job_id"] for row in jobs}) != len(jobs)
            or len({row["stage"] for row in jobs}) != len(jobs)):
        raise ValueError("Accounting requires unique exact workflow stage/job identities")
    by_job = {row["job_id"]: row["stage"] for row in jobs}
    command = ["sacct", "--noheader", "--parsable2", "--jobs", ",".join(by_job),
               "--format=" + ",".join(ACCOUNTING_FIELDS)]
    result = {"schema": "tdn.scheduler-accounting/v1", "workflow_id": workflow["run_id"],
        "collected_at": cw.now(), "requested_job_ids": list(by_job), "records": [],
        "status": "UNAVAILABLE", "command": command if by_job and enabled else None,
        "scope": "Slurm accounting for recorded workflow allocations and their exact task steps; allocation and step rows are never summed",
        "cost_scope": "Reported resource/time usage, distinct from scientific experiment timers; no monetary rate is assumed",
        "monetary_cost": None, "all_allocations_terminal": None,
        "inherited_stage_paths": workflow.get("recovery", {}).get("stage_paths", {}),
        "inherited_cost_scope": "Origin allocation costs remain in the preserved origin workflow; never added to new recovery jobs",
        "snapshot_note": "Snapshot values may lag Slurm; a running reporting allocation is partial, never a completed cost.",
        "units": {"ElapsedRaw": "seconds", "CPUTimeRAW": "allocated CPU-seconds reported by Slurm",
            "TotalCPU": "Slurm CPU-time string", "MaxRSS": "Slurm memory string; an empty allocation value is unknown, not zero"}}
    if not enabled:
        result.update(status="DISABLED", reason="Collection accounting was explicitly disabled with --no-accounting")
    elif not by_job:
        result["reason"] = "No submitted scheduler jobs are recorded for this workflow"
    else:
        try:
            response = (runner or subprocess.run)(command, cwd=ROOT, check=False, text=True,
                capture_output=True, timeout=30)
            if response.returncode:
                raise ValueError(f"sacct exited {response.returncode}: {response.stderr.strip()[:2000]}")
            if len(response.stdout.encode()) > 1 << 20:
                raise ValueError("Accounting output exceeds its 1 MiB snapshot limit")
            records, seen = [], set()
            for line in response.stdout.splitlines():
                if not line.strip():
                    continue
                fields = line.strip().split("|")
                if len(fields) == len(ACCOUNTING_FIELDS) + 1 and fields[-1] == "":
                    fields.pop()
                if len(fields) != len(ACCOUNTING_FIELDS):
                    raise ValueError("Accounting returned malformed columns; no partial resource totals were inferred")
                row = dict(zip(ACCOUNTING_FIELDS, fields))
                identity = row["JobIDRaw"]
                allocation = identity.split(".", 1)[0]
                if allocation not in by_job or not re.fullmatch(r"[1-9][0-9]*(?:\.[A-Za-z0-9_-]+)?", identity):
                    raise ValueError("Accounting returned a record outside the frozen workflow jobs")
                if identity in seen:
                    raise ValueError("Accounting returned duplicate job or step identities")
                seen.add(identity)
                for key in ("ElapsedRaw", "CPUTimeRAW"):
                    if row[key] and not row[key].isdigit():
                        raise ValueError(f"Accounting returned an invalid {key} value")
                row.update(workflow_stage=by_job[allocation], logical_stage=units(workflow)[by_job[allocation]]['kind'],
                    allocation_role='unit',
                    allocation_job_id=allocation,
                    record_kind="allocation" if identity == allocation else "step",
                    terminal_state=row["State"].split(" ", 1)[0].rstrip("+") in TERMINAL_JOB_STATES,
                    elapsed_seconds=int(row["ElapsedRaw"]) if row["ElapsedRaw"] else None,
                    allocated_cpu_seconds=int(row["CPUTimeRAW"]) if row["CPUTimeRAW"] else None,
                    missing_fields=[key for key in ACCOUNTING_FIELDS if not row[key]])
                records.append(row)
            allocations = {row["allocation_job_id"] for row in records if row["record_kind"] == "allocation"}
            missing = [job for job in by_job if job not in allocations]
            result.update(records=records, missing_allocation_job_ids=missing,
                all_allocations_terminal=not missing and all(row["terminal_state"] for row in records
                    if row["record_kind"] == "allocation"),
                nonterminal_allocation_job_ids=[row["allocation_job_id"] for row in records
                    if row["record_kind"] == "allocation" and not row["terminal_state"]],
                status=("RECORDED" if not missing and all(row["terminal_state"] for row in records if row["record_kind"]=="allocation") else "PENDING_ACCOUNTING" if not missing else "PARTIAL" if records else "UNAVAILABLE"),
                reason="All requested allocation rows returned; fields may still be unavailable" if not missing
                       else "Some requested allocations are unavailable in sacct; absent fields remain unknown")
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            result.update(status="UNAVAILABLE", reason=f"{type(error).__name__}: {error}", records=[])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = base / "state" / "scheduler-accounting.json"
    cw.atomic_json(base / "state" / f"scheduler-accounting-{stamp}.json", result)
    cw.atomic_json(path, result)
    print(f"Scheduler accounting: {result['status']} — {path}")
    if result["status"] != "RECORDED":
        print("Accounting note: " + result["reason"])
    return result

def collect(workflow, *, part_bytes=ARCHIVE_PART_BYTES, accounting=True):
    """Keep one complete archive and provide verified upload-sized parts if needed."""
    if type(part_bytes) is not int or not 0 < part_bytes <= ARCHIVE_PART_BYTES:
        raise ValueError("Archive part size must be between 1 byte and 28 MiB")
    base = cw.inside(workflow["run_dir"])
    scheduler_accounting(workflow, enabled=accounting)
    destination = cw.inside(ROOT / "runs" / (workflow["run_id"] + "-review-" +
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".tar.gz"))
    members = []
    roots = {base}
    bridge = recovery_bridge(workflow)
    if bridge:
        verify_recovery_bridge(workflow)
        roots.update(cw.inside(path).parent for path in bridge["stage_paths"].values())
        roots.add(cw.inside(bridge["origin_workflow_path"]).parent)
        roots.update(cw.inside(path) for path in bridge.get("origin_roots", []))
    for source in sorted(roots):
        if source.parent != ROOT / "runs":
            raise ValueError("Review archive inherited evidence must remain in project run directories")
        for path in sorted(source.rglob("*")):
            relative = path.relative_to(source)
            if any(part == "__pycache__" or part.endswith(("pytest-work", "pytest-cache")) for part in relative.parts):
                continue
            if path.is_symlink():
                raise ValueError(f"Review archive refuses symlink: {path}")
            if cw.inside(path).is_file():
                members.append((path, str(Path(source.name) / relative)))
    with destination.open("xb") as stream:
        with tarfile.open(fileobj=stream, mode="w:gz", dereference=False) as archive:
            for path, relative in members:
                archive.add(path, arcname=relative, recursive=False)
    index = {"schema_version": 1, "archive": destination.name, "bytes": destination.stat().st_size,
             "sha256": cw.digest(destination), "member_count": len(members), "parts": [],
             "included_workflow_directories": sorted(path.name for path in roots),
             "reassemble": "Concatenate parts in their listed order; verify the complete SHA-256 before extracting."}
    if destination.stat().st_size > part_bytes:
        with destination.open("rb") as stream:
            ordinal = 1
            while block := stream.read(part_bytes):
                part = destination.with_name(destination.name + f".part{ordinal:03d}")
                with part.open("xb") as output:
                    output.write(block)
                index["parts"].append({"path": part.name, "bytes": len(block), "sha256": hashlib.sha256(block).hexdigest()})
                ordinal += 1
    index_path = destination.with_name(destination.name + ".index.json")
    cw.atomic_json(index_path, index)
    print(f"Review archive: {destination}\nReview index: {index_path}")
    print("Includes all stages, checkpoints, references, metrics, logs and failures; excludes pytest temporary directories.")
    if index["parts"]:
        print(f"Archive exceeds upload limit; upload the index and all {len(index['parts'])} parts (each at most 28 MiB).")
        for part in index["parts"]:
            print(base.parent / part["path"])
    return destination

if __name__ == '__main__':
    raise SystemExit(main())
