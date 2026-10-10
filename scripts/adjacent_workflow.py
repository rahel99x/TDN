#!/usr/bin/env python3
"""Bounded adjacent interaction study; controllers only schedule and inspect metadata."""
from __future__ import annotations
import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location('adjacent_fedora_helpers', ROOT / 'scripts/fedora_workflow.py')
fw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fw)
pw, cw, rw = fw.pw, fw.cw, fw.rw
from tdn.runtime.desktop_slurm import load_profile, validate_profile, verify_allocation as runtime_allocation

MANIFEST = 'adjacent-workflow.json'
POINTER = '.fedora-adjacent-latest.json'
RESOLUTION_POINTER = '.fedora-adjacent-resolution-latest.json'
from tdn.analysis.adjacent.protocol import PROFILES, HISTORICAL_PROFILES, RESOLUTION_PROFILES
SCHEDULER_MEMORY_CAP_MIB = 110000
ARCHIVE_PART_BYTES = 28 * 2**20
ACCOUNTING_FIELDS = ('JobIDRaw', 'State', 'ExitCode', 'ElapsedRaw', 'TotalCPU', 'AllocTRES', 'MaxRSS', 'CPUTimeRAW')
TERMINAL_JOB_STATES = frozenset(('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
    'PREEMPTED', 'NODE_FAIL', 'BOOT_FAIL', 'DEADLINE', 'REVOKED', 'SPECIAL_EXIT'))


def scientific_protocol(profile):
    from tdn.analysis.adjacent.protocol import build_protocol
    return build_protocol(profile)


def units(workflow):
    return scientific_protocol(workflow['profile'])['units']


def physical_stages(workflow):
    from tdn.analysis.adjacent.protocol import plan_units
    return tuple(row['id'] for row in plan_units(scientific_protocol(workflow['profile'])))


def ancestors(workflow, stage):
    rows = units(workflow)
    if stage not in rows:
        raise ValueError('Unknown adjacent unit')
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
    return {'version': 1, 'benchmark_suite': 'adjacent', 'profile': profile,
        'scientific_protocol': protocol, 'execution_mode': 'desktop-slurm',
        'scheduler_memory_cap_mib': SCHEDULER_MEMORY_CAP_MIB,
        'scheduling': 'afterok science DAG; afterany single-node resource serialization; report afterany all',
        'scope': 'No pending-job cap, automatic budget expansion, or automatic retries'}


def is_gpu_stage(workflow, stage):
    return units(workflow)[stage]['device'] == 'cuda'


def resource_for(workflow, stage):
    if stage not in physical_stages(workflow):
        raise ValueError('Unknown adjacent unit')
    return workflow['resources'][stage]


def validate_resources(protocol):
    rows = protocol['units']
    for stage, row in rows.items():
        if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]*', stage):
            raise ValueError('Unsafe adjacent unit identifier')
        if type(row['cpus']) is not int or not 1 <= row['cpus'] <= 8:
            raise ValueError('Adjacent requires at most eight physical CPU cores')
        if row['device'] == 'cuda' and row['cpus'] != 4:
            raise ValueError('Adjacent GPU units require four CPUs and one GPU')
        if type(row['mem_gib']) is not int or not 1 <= row['mem_gib'] <= 48:
            raise ValueError('Adjacent memory must fit 48 GiB per unit and 110000 MiB node')
        if not 180 <= wall_seconds(row['walltime']) <= 2700:
            raise ValueError('Adjacent allocations must remain within 45 minutes')
        if not 0 < row['seconds'] <= min(1800, wall_seconds(row['walltime']) - 120):
            raise ValueError('Scientific budget must leave bounded reporting/signal headroom')
    return {name: {key: row[key] for key in ('cpus', 'mem_gib', 'walltime')} for name,row in rows.items()}


def validate(workflow):
    if workflow.get('schema_version') != 1 or workflow.get('kind') != 'desktop-slurm-adjacent':
        raise ValueError('Not a supported adjacent workflow')
    run_id = pw.identifier(workflow['run_id'])
    base = cw.inside(workflow['run_dir'])
    if base != cw.inside(ROOT/'runs'/run_id) or cw.inside(workflow['root']) != ROOT.resolve():
        raise ValueError('Adjacent workflow layout changed')
    profile = workflow['profile']
    if profile not in PROFILES or workflow.get('execution_mode') != 'desktop-slurm':
        raise ValueError('Unknown adjacent profile or execution mode')
    if cw.inside(workflow['protocol_path']) != base/'protocol.json':
        raise ValueError('Frozen protocol must remain within coordinator')
    if workflow['protocol_sha256'] != pw.canonical_hash(declaration(profile)):
        raise ValueError('Adjacent scientific declaration changed')
    site = validate_profile(workflow['slurm_profile'], root=ROOT)
    if workflow['slurm_profile_sha256'] != pw.canonical_hash(site) or cw.inside(workflow['slurm_profile_path']) != base/'slurm-profile.json':
        raise ValueError('Frozen Slurm profile differs')
    if workflow.get('torch_version') != site['torch_version']:
        raise ValueError('Torch distribution differs from Slurm profile')
    resources = validate_resources(scientific_protocol(profile))
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


def workflow_path(value, *, resolution=False):
    if value == 'latest':
        value = cw.read_json(ROOT/'runs'/(RESOLUTION_POINTER if resolution else POINTER))['run_id']
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
        raise ValueError('Adjacent manifest is outside its coordinator')
    if verify:
        if cw.read_json(value['protocol_path']) != declaration(value['profile']):
            raise ValueError('Frozen protocol changed')
        if load_profile(value['slurm_profile_path'],root=ROOT) != value['slurm_profile']:
            raise ValueError('Frozen Slurm profile changed')
        if cw.source_hash() != value['source_sha256']:
            raise ValueError('Source changed; preserve run and submit a new workflow')
    return value


def prepare(args):
    site = load_profile(root=ROOT)
    resolution = getattr(args, 'resolution', False)
    selected_profile = getattr(args, 'selected_profile', None)
    if resolution and getattr(args, 'development', False):
        raise ValueError('Resolution studies use --smoke or --full; all profiles are development evidence')
    grid = getattr(args, 'grid', None)
    if grid is not None and (not resolution or args.smoke):
        raise ValueError('--grid requires a full resolution study; resolution smoke covers both grids')
    profile = selected_profile or (('resolution-smoke' if args.smoke else 'resolution'+str(grid) if grid else 'resolution-full') if resolution
        else 'smoke' if args.smoke else 'development' if args.development else 'full')
    prefix = 'fedora-adjacent-resolution-' if profile in RESOLUTION_PROFILES else 'fedora-adjacent-'
    run_id = pw.identifier(args.run_id or prefix+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    base = cw.inside(ROOT/'runs'/run_id)
    if base.exists():
        raise ValueError('Run directory exists; preserve it and select a fresh run ID')
    defer_jobs=getattr(args,'after_job',[])
    if any(not re.fullmatch(r'[1-9][0-9]*',v) for v in defer_jobs):
        raise ValueError('--after-job requires ordinary positive Slurm job IDs')
    resources = validate_resources(scientific_protocol(profile))
    return validate({'schema_version':1,'kind':'desktop-slurm-adjacent','profile':profile,'run_id':run_id,
        'root':str(ROOT.resolve()),'run_dir':str(base),'created_at':cw.now(),'execution_mode':'desktop-slurm','defer_job_ids':defer_jobs,
        'protocol_path':str(base/'protocol.json'),'protocol_sha256':pw.canonical_hash(declaration(profile)),
        'slurm_profile':site,'slurm_profile_path':str(base/'slurm-profile.json'),
        'slurm_profile_sha256':pw.canonical_hash(site),'torch_version':site['torch_version'],
        'source_sha256':cw.source_hash(),'source_tree_sha256':cw.source_hash(metadata=True),
        'resources':{stage:{**row,'partition':site['gpu_partition' if scientific_protocol(profile)['units'][stage]['device']=='cuda' else 'cpu_partition']}
                     for stage,row in resources.items()}})


def comment(workflow,stage):
    return f"tdn-adjacent:{workflow['run_id']}:{stage}:{workflow['slurm_profile_sha256']}"


def scheduler_args(workflow, stage, dependency=None):
    base,site,res = Path(workflow['run_dir']),workflow['slurm_profile'],resource_for(workflow,stage)
    args = ['sbatch','--parsable',f"--partition={res['partition']}",'--nodes=1','--ntasks=1',
        f'--job-name=tdn-adjacent-{stage}',f'--comment={comment(workflow,stage)}',
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
    return args+[str(ROOT/'scripts/adjacent_worker.sh')]


def job_dependency(workflow, stage, jobs):
    """Science failure blocks only descendants; resource serialization waits for any outcome."""
    if not jobs:
        waiting=workflow.get('defer_job_ids',[])
        return 'afterany:'+':'.join(waiting) if waiting else None
    if stage=='report':
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
            raise ValueError('Partition walltime is below adjacent request')
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
    return {key:value for key,value in os.environ.items() if not key.startswith(('SBATCH_','SRUN_','TDN_PREMIX_','TDN_CONSISTENCY_','TDN_AGENDA_','TDN_ROADMAP_','TDN_FRONTIER_','TDN_PORTFOLIO_','TDN_ADJACENT_'))
        and key not in ('CARC_ACCOUNT','TORCH_CUDA_ARCH_LIST','TDN_REQUIRE_GPU_TESTS','TDN_TOWER_DIR')}


def submission_environment(workflow,stage):
    env=clean_environment()
    env.update(TDN_EXECUTION_MODE='desktop-slurm',TDN_PROJECT_ROOT=str(ROOT),TDN_REPO_ROOT=str(ROOT),
        TDN_SLURM_CONFIG=workflow['slurm_profile_path'],TDN_ADJACENT_WORKFLOW=str(Path(workflow['run_dir'])/MANIFEST),
        TDN_ADJACENT_STAGE=stage,TDN_ADJACENT_DEVICE=units(workflow)[stage]['device'],
        TDN_ADJACENT_CPUS=str(resource_for(workflow,stage)['cpus']),
        TDN_FEDORA_GPU_GRES=workflow['slurm_profile']['gpu_gres'],TORCH_VERSION=workflow['torch_version'])
    return env


def submit(workflow, *, plan_only=False, bridge=None):
    print(f"Fedora adjacent: {workflow['run_id']} ({workflow['profile']})\nRun: {workflow['run_dir']}")
    print('Separate adjacent-study DAG; serial desktop resources, no pending-job cap. Final report afterany.')
    if workflow['profile'] in RESOLUTION_PROFILES:
        p = scientific_protocol(workflow['profile']); budget = p['adjacent_budget']
        allocation_ceiling = sum(wall_seconds(u['walltime']) for u in p['units'].values())
        print(f"Bounded development: grids={p['resolution']['grids']}; {len(p['units'])} jobs; summed science CEILING={budget['summed_science_ceiling_seconds']}s; summed allocation CEILING={allocation_ceiling}s; protected exploration={budget['protected_exploration_seconds']}s.")
        print('Ceilings are safety bounds, not runtime predictions or billed cost.')
    for stage in pending_stages(workflow):
        print(shlex.join(scheduler_args(workflow,stage)))
    if plan_only:
        print('PLAN ONLY: no writes, scheduler calls or numerical work.')
        return workflow
    fw.controller_policy()
    # Do not compete with the user's main portfolio. Resolve actual active
    # jobs at submission, then freeze the explicit afterany boundary.
    queue=cw.command(['squeue','--noheader','--user',workflow['slurm_profile']['user'],'--format=%i|%j']).stdout
    main_jobs=[]
    for line in queue.splitlines():
        job,sep,name=line.strip().partition('|')
        if sep and (name.startswith('tdn-portfolio-') or
                (workflow['profile'] in RESOLUTION_PROFILES and name.startswith('tdn-adjacent-'))):
            if not re.fullmatch(r'[1-9][0-9]*',job):
                raise ValueError('Cannot resolve main-campaign allocation; use explicit ordinary job IDs')
            main_jobs.append(job)
    workflow['defer_job_ids']=sorted(set(workflow.get('defer_job_ids',[])+main_jobs),key=int)
    if workflow['defer_job_ids']:
        print('Deferring adjacent allocations until main jobs finish: '+','.join(workflow['defer_job_ids']))
    policies=check_capacity(workflow)
    with rw.acquire_venv_lock():
        software=rw.software_report(workflow)
    with cw.controller_lock():
        base=Path(workflow['run_dir']);base.mkdir(parents=True,exist_ok=False);(base/'logs').mkdir()
        for path,value in ((base/MANIFEST,workflow),(base/'protocol.json',declaration(workflow['profile'])),(base/'slurm-profile.json',workflow['slurm_profile'])):
            cw.atomic_json(path,value);path.chmod(0o444)
        if bridge:
            cw.atomic_json(base/'recovery.json',bridge);(base/'recovery.json').chmod(0o444)
            verify_recovery_bridge(workflow)
        cw.atomic_json(base/'state/slurm-policy.json',policies)
        cw.atomic_json(base/'state/submission-software.json',software)
        jobs=[];cw.atomic_json(base/'jobs.json',jobs)
        pointer = RESOLUTION_POINTER if workflow['profile'] in RESOLUTION_PROFILES else POINTER
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
            print(f"TDN_ADJACENT_{stage.upper().replace('-','_')}_JOB_ID={match.group(1)}",flush=True)
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
    spec=importlib.util.spec_from_file_location('adjacent_stage_cli',ROOT/'scripts/adjacent.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def verify_stage(workflow,stage):
    checked=stage_cli().verify_execution(stage_path(workflow,stage),scientific_protocol(workflow['profile']),source_tree_sha256=workflow['source_tree_sha256'])
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
    expected={'JobId':job,'JobName':f'tdn-adjacent-{stage}','Comment':comment(workflow,stage),
        'JobState':'RUNNING','Partition':res['partition'],'WorkDir':str(ROOT)}
    if any(fields.get(key)!=value for key,value in expected.items()) or fields.get('UserId','').split('(')[0]!=site['user']:
        raise ValueError('Allocation is not this running adjacent unit')
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
        raise ValueError('CPU-only adjacent unit unexpectedly allocated a GPU')
    recorded=[row for row in cw.read_json(Path(workflow['run_dir'])/'jobs.json') if row.get('stage')==stage]
    if len(recorded)!=1 or recorded[0].get('job_id')!=job:
        raise ValueError('Allocation differs from exact recorded submission')


def junit_path(workflow,stage):
    return cw.inside(Path(workflow['run_dir'])/'reporter-tests'/stage/'tests.xml')


def worker_commands(workflow,stage):
    python=str(cw.inside(ROOT/'.venv/bin')/'python');base=Path(workflow['run_dir']);commands=[]
    gpu=is_gpu_stage(workflow,stage)
    if stage=='report':
        commands.append(('accounting',[python,str(ROOT/'scripts/adjacent_workflow.py'),'snapshot-accounting','--workflow',str(base/MANIFEST)]))
    if stage=='audit' or gpu:
        tests=[ROOT/'tests/test_adjacent_gpu.py',ROOT/'tests/test_portfolio_gpu.py'] if gpu else sorted((ROOT/'tests').glob('test_adjacent*.py'))+[ROOT/'tests/test_desktop_slurm_runtime.py']
        if gpu and workflow['profile'] in RESOLUTION_PROFILES:
            tests.extend(ROOT/'tests'/name for name in ('test_adjacent_resolution_gpu.py', 'test_adjacent_resolution_exploration_gpu.py'))
        if not tests or any(not path.is_file() for path in tests):
            raise ValueError('Adjacent mandatory correctness tests are missing')
        if gpu:
            commands.append(('preflight',[python,str(ROOT/'scripts/fedora_gpu_preflight.py'),'--output',str(base/f'{stage}-gpu-preflight.json')]))
        commands.append(('gpu-tests' if gpu else 'tests',[python,'-m','pytest','-q','-m','gpu' if gpu else 'not gpu',*map(str,tests),
            '--basetemp',str(base/f'{stage}-pytest-work'),'-o',f"cache_dir={base/(stage+'-pytest-cache')}",'--junitxml',str(junit_path(workflow,stage))]))
        if gpu:
            commands.append(('check-gpu-tests',[python,str(ROOT/'scripts/adjacent_workflow.py'),'check-gpu-tests','--junit',str(junit_path(workflow,stage)),'--profile',workflow['profile']]))
    command=[python,str(ROOT/'scripts/adjacent.py'),'--stage',stage,'--profile',workflow['profile'],'--run-dir',str(base/stage),'--device','cuda' if gpu else 'cpu']
    for prior in ancestors(workflow,stage):
        command += ['--prerequisite-dir',f'{prior}={stage_path(workflow,prior)}']
    bridge=recovery_bridge(workflow)
    if bridge and stage in bridge['resume_paths']:
        command+=['--resume-from',bridge['resume_paths'][stage]]
    commands.append(('experiment',command))
    return commands


def workflow_environment(workflow,stage):
    return {'TDN_ADJACENT_PROTOCOL_SHA256':workflow['protocol_sha256'],'TDN_ADJACENT_WORKFLOW':str(Path(workflow['run_dir'])/MANIFEST),
        'TDN_ADJACENT_STAGE':stage,'TDN_ADJACENT_CPUS':str(resource_for(workflow,stage)['cpus'])}


def worker(workflow,stage):
    if stage not in pending_stages(workflow):
        raise ValueError('Unknown or inherited adjacent unit cannot run')
    backend=SimpleNamespace(**{name:globals()[name] for name in ('verify_allocation','load','begin_report','worker_commands',
        'verify_stage','prepare_test_environment','workflow_environment','finish_report','verify_software')},
        gpu_junit_path=lambda workflow:junit_path(workflow,stage),workflow_filename=MANIFEST,workflow_label='adjacent',
        stage_variable='TDN_ADJACENT_STAGE',prerequisite_stages=tuple(name for name in physical_stages(workflow) if name not in ('audit','report')),
        verify_prerequisites=lambda workflow,software:verify_prerequisites(workflow,software,stage))
    try:
        return pw.worker(workflow,stage,backend=backend)
    except BaseException as error:
        path=pw.state_path(workflow,stage)
        if not path.exists():
            cw.atomic_json(path,{'status':'FAILED','stage':'startup','exit_code':2,'error':str(error),
                'source_sha256':workflow['source_sha256'],'protocol_sha256':workflow['protocol_sha256'],
                'updated_at':cw.now(),'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'slurm_step_id':os.environ.get('SLURM_STEP_ID')})
        raise


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
        checked=stage_cli().verify_execution(path,scientific_protocol(workflow['profile']),source_tree_sha256=workflow['source_tree_sha256'])
        if cw.digest(Path(path)/'workflow-seal.json')!=bridge['seals'][stage]:
            raise ValueError('Completed origin unit changed')
        if checked['execution']['stage']!=stage:
            raise ValueError('Inherited unit scope differs')
    for stage,path in bridge['resume_paths'].items():
        if stage_cli().resume_fingerprint(path)!=bridge['resume_fingerprints'][stage]:
            raise ValueError('Interrupted origin artifacts changed')
    return bridge['stage_paths']


def recover(args):
    origin=load(workflow_path(args.run, resolution=getattr(args,'resolution',False)))
    if (origin['profile'] in RESOLUTION_PROFILES) != getattr(args,'resolution',False):
        raise ValueError('Use the matching adjacent or adjacent-resolution launcher to recover this workflow')
    options=SimpleNamespace(run_id=args.run_id,after_job=[],smoke=origin['profile']=='smoke',development=origin['profile']=='development',
        selected_profile=origin['profile'],resolution=origin['profile'] in RESOLUTION_PROFILES)
    workflow=prepare(options)
    fw.controller_policy();states=require_terminal_origin(origin)
    if any(workflow[key]!=origin[key] for key in ('source_sha256','source_tree_sha256','protocol_sha256','slurm_profile_sha256')):
        raise ValueError('Recovery requires identical source, protocol and hardware profile')
    with rw.acquire_venv_lock():
        software=rw.software_report(workflow)
    if software!=cw.read_json(Path(origin['run_dir'])/'state/submission-software.json'):
        raise ValueError('Recovery software differs')
    completed,resume,seals,fingerprints={},{},{},{}
    for stage in physical_stages(origin):
        if stage=='report': continue
        path=stage_path(origin,stage)
        if not path.exists(): continue
        if (path/'workflow-seal.json').exists():
            verify_stage(origin,stage);completed[stage]=str(path);seals[stage]=cw.digest(path/'workflow-seal.json')
        elif ((units(origin)[stage].get('resumable',False) or units(origin)[stage]['kind'] in ('train','confirm')) and (path/'execution.json').is_file()
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
    bridge={'schema':'tdn.adjacent-recovery/v1','origin_workflow_path':str(Path(origin['run_dir'])/MANIFEST),
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
            print(f'TDN_ADJACENT_{name}={file}')
    for row in cw.read_json(base/'jobs.json'):
        for report in cw.tower_reports_for_job(workflow,row['job_id']):
            print(f"{row['stage']} job {row['job_id']}\nTDN_TOWER_DIR={report}\nTDN_TOWER_METRICS={report/'metrics.jsonl'}")


def status(workflow):
    base=Path(workflow['run_dir']);jobs={row['stage']:row for row in cw.read_json(base/'jobs.json')}
    print(f"Fedora adjacent {workflow['run_id']} ({workflow['profile']})\nRun: {base}")
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
                except (OSError,ValueError,KeyError,TypeError) as error: print(f'  artifact verification: INVALID ({error})')
        state_path=pw.state_path(workflow,stage)
        if state_path.exists() and (record:=cw.read_json(state_path)).get('error'):
            print('  worker error: '+record['error'])


def logs(workflow,lines):
    if not 1<=lines<=100000: raise ValueError('Choose 1–100000 lines')
    for path in sorted((Path(workflow['run_dir'])/'logs').glob('*')):
        if cw.inside(path).is_file():
            print('\n'+str(path))
            with path.open(errors='replace') as stream: print(''.join(deque(stream,maxlen=lines)),end='')


def validate_results(workflow):
    result={'validation':'adjacent_sealed_artifacts','units':{}};failed=False
    for stage in physical_stages(workflow):
        try:
            if not stage_path(workflow,stage).is_dir():
                raise FileNotFoundError('Unit has not produced artifacts')
            verify_stage(workflow,stage)
            result['units'][stage]={'validation':'VERIFIED','scientific_outcome':cw.read_json(stage_path(workflow,stage)/'summary.json').get('scientific_outcome','NA')}
        except (OSError,ValueError,KeyError,TypeError) as error:
            result['units'][stage]={'validation':'INVALID' if stage_path(workflow,stage).exists() else 'MISSING','scientific_outcome':'NA','error':str(error)};failed=True
    print(json.dumps(result,indent=2))
    if failed: raise ValueError('Some adjacent units are missing or invalid; report retains NA evidence')
    return result


def parser():
    result=argparse.ArgumentParser(description=__doc__);result.add_argument('--resolution',action='store_true',help='Use the isolated 64/128 study and its own latest pointer');commands=result.add_subparsers(dest='command',required=True)
    for name in ('plan','run'):
        item=commands.add_parser(name);item.add_argument('--run-id');item.add_argument('--after-job',action='append',default=[]);group=item.add_mutually_exclusive_group()
        for profile in HISTORICAL_PROFILES: group.add_argument('--'+profile,action='store_true')
        item.add_argument('--grid',type=int,choices=(64,128),help='Resolution full only: run one grid')
    item=commands.add_parser('recover');item.add_argument('run',nargs='?',default='latest');item.add_argument('--run-id');item.add_argument('--plan',action='store_true')
    for name in ('status','logs','paths','validate','collect'):
        item=commands.add_parser(name);item.add_argument('run',nargs='?',default='latest')
        if name=='logs': item.add_argument('--lines',type=int,default=200)
        if name=='collect': item.add_argument('--no-accounting',action='store_true')
    item=commands.add_parser('worker');item.add_argument('--workflow',type=Path,required=True);item.add_argument('--stage',required=True)
    item=commands.add_parser('check-gpu-tests');item.add_argument('--junit',type=Path,required=True);item.add_argument('--profile',choices=PROFILES,default='full')
    item=commands.add_parser('snapshot-accounting');item.add_argument('--workflow',type=Path,required=True)
    return result


def main(argv=None):
    args=parser().parse_args(argv)
    try:
        if args.command in ('plan','run'): submit(prepare(args),plan_only=args.command=='plan')
        elif args.command=='recover': recover(args)
        elif args.command=='worker': return worker(load(args.workflow,verify=False),args.stage)
        elif args.command=='check-gpu-tests': print(json.dumps(check_gpu_tests(args.junit,profile=args.profile),sort_keys=True))
        elif args.command=='snapshot-accounting': scheduler_accounting(load(args.workflow,verify=False))
        else:
            workflow=load(workflow_path(args.run,resolution=args.resolution),verify=False)
            if (workflow['profile'] in RESOLUTION_PROFILES) != args.resolution:
                raise ValueError('Use the matching adjacent or adjacent-resolution launcher for this workflow')
            if args.command=='logs': logs(workflow,args.lines)
            elif args.command=='collect': collect(workflow,accounting=not args.no_accounting)
            elif args.command=='validate': validate_results(workflow)
            else: globals()[args.command](workflow)
    except rw.WorkerFailure as error:
        print(f'TDN adjacent: {error}',file=sys.stderr);return error.exit_code
    except (ValueError,OSError,KeyError,TypeError) as error:
        print(f'TDN adjacent: {error}',file=sys.stderr);return 2
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

def check_gpu_tests(path, *, profile="full"):
    """Require every declared adjacent CUDA case; no skips or filtered suites."""
    from collections import Counter
    from xml.etree import ElementTree
    from tdn.analysis.adjacent.protocol import build_protocol
    GPU_TEST_CASES = build_protocol(profile)["gpu_test_cases"]
    path = cw.inside(path)
    pw.validate_gpu_junit(path)
    tree = ElementTree.parse(path).getroot()
    cases = tree.findall(".//testcase")
    identities = [(case.get("classname"), case.get("name")) for case in cases]
    expected = [("tests.test_portfolio_gpu" if name.startswith('test_portfolio_') else "tests.test_adjacent_resolution_exploration_gpu" if name.startswith("test_resolution_cuda_exploration") else "tests.test_adjacent_resolution_gpu" if name.startswith("test_resolution_") else "tests.test_adjacent_gpu", name) for name in GPU_TEST_CASES]
    if not expected or Counter(identities) != Counter(expected) or len(set(identities)) != len(identities):
        raise ValueError("Adjacent GPU readiness requires every unique mandatory declared CUDA case; filtered suites are invalid")
    if any(case.find(tag) is not None for case in cases for tag in ("failure", "error", "skipped")):
        raise ValueError("All mandatory adjacent GPU tests must pass without failures, errors or skips")
    suites = [tree] if tree.tag == "testsuite" else tree.findall(".//testsuite")
    if any(int(suite.get("tests", -1)) != len(suite.findall("testcase")) for suite in suites):
        raise ValueError("Adjacent GPU JUnit declared counts do not match the recorded cases")
    if tree.tag == "testsuites" and tree.get("tests") is not None and int(tree.get("tests")) != len(cases):
        raise ValueError("Adjacent GPU JUnit aggregate count differs")
    return {"validation": "mandatory_adjacent_gpu_suite", "test_cases": len(cases), "case_names": list(GPU_TEST_CASES)}

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
    return cw.reporting_api().begin_report(base / stage, name=f"TDN/adjacent/{stage}", script="scripts/adjacent_worker.sh",
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
    from tdn.adjacent_reporting import publish_outputs as publish_adjacent
    publish_adjacent(report, sources)
    cw.reporting_api().finish_report(report, state=state, runtime_seconds=runtime_seconds, exit_code=exit_code, results=results,
        metadata={"workflow_id": workflow["run_id"], "phase": stage,
                  "execution_scope": "allocated worker elapsed time; excludes pending time",
                  **({"verified_software_sha256": pw.canonical_hash(software)} if software else {}),
                  **({"error": str(error)} if error is not None else {})})

def prepare_test_environment(env, name):
    for key in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
        env.pop(key, None)
    for key in tuple(env):
        if key.startswith(("TDN_ADJACENT_", "TDN_PORTFOLIO_", "TDN_FRONTIER_", "TDN_ROADMAP_", "TDN_AGENDA_", "TDN_PREMIX_", "TDN_CONSISTENCY_")):
            env.pop(key)
    if name == "gpu-tests":
        env["TDN_REQUIRE_ADJACENT_GPU_TESTS"] = "1"
        env["TDN_REQUIRE_PORTFOLIO_GPU_TESTS"] = "1"
    if name == "tests":
        for key in ("TDN_EXECUTION_MODE", "TDN_SLURM_CONFIG"):
            env.pop(key, None)
        # CPU unit tests own their synthetic runtime fixtures. Removing the
        # desktop mode but retaining real Slurm IDs would instead make local
        # path checks infer a CARC allocation. This is only the child test
        # environment: the allocated worker and every GPU test retain their
        # real scheduler identity and all native verification requirements.
        for key in tuple(env):
            if key.startswith("SLURM_"):
                env.pop(key)

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
                status="RECORDED" if not missing else "PARTIAL" if records else "UNAVAILABLE",
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
