#!/usr/bin/env python3
"""Gated tangent-response native workflows, using the verified allocated worker."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import fedora_workflow as fw
from scripts import response_workflow as shared
from scripts.response_diagnostic_archive import confined, read_json, controller_lock, identifier
from tdn.runtime.desktop_slurm import load_profile, validate_profile, verify_allocation as runtime_allocation

cw,pw,rw=fw.cw,fw.pw,fw.rw
MANIFEST='tangent-response-workflow.json'
CAMPAIGN='tangent-response-campaign.json'
POINTER='.fedora-tangent-response-latest.json'
PHASES=('mechanism','compression','fit_prepare','calibration','learning','utility')
TERMINAL=shared.TERMINAL_JOB_STATES


def digest(value):
    return pw.canonical_hash(value)


def read(path,root=ROOT):
    return read_json(root,path)[0]


def write(path,value):
    cw.atomic_json(path,value)


def fresh(prefix):
    return identifier(prefix+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8])


class NativeWorkflow:
    """Root-bound adapter; old experiment globals/defaults are never changed."""
    def __init__(self,root=ROOT,*,command=None,builder=None,science_verifier=None):
        self.root=Path(root).resolve()
        self.command=command or cw.command
        self.builder=builder
        self.science_verifier=science_verifier

    def path(self,path):return confined(self.root,path)
    def read(self,path):return read_json(self.root,path)[0]
    def protocol(self,w):return w['scientific_protocol']
    def units(self,w):return self.protocol(w)['units']
    def stages(self,w):return tuple(self.protocol(w)['stages'])
    def state(self,w,stage):return self.path(Path(w['run_dir'])/'state'/f'{stage}.json')

    def stage_path(self,w,stage):
        if stage not in self.units(w):raise ValueError('Unknown tangent-response unit')
        return self.path(w.get('recovery',{}).get('stage_paths',{}).get(stage, str(Path(w['run_dir'])/stage)))

    def workflow_path(self,run):
        if run=='latest':run=self.read(self.root/'runs'/POINTER)['run_id']
        return self.path(self.root/'runs'/identifier(run)/MANIFEST)

    def validate(self,w,*,historical=False):
        if w.get('schema_version')!=1 or w.get('kind')!='desktop-slurm-tangent-response':raise ValueError('Not a tangent-response workflow')
        base=self.path(self.root/'runs'/identifier(w['run_id']))
        if w.get('run_dir')!=str(base) or w.get('root')!=str(self.root) or w.get('execution_mode')!='desktop-slurm':raise ValueError('Native workflow root/identity differs')
        p=self.protocol(w)
        importlib.import_module('tdn.analysis.tangent_response.protocol').validate_protocol(p,require_binding=False,canonical=not historical)
        if p.get('profile')!=w['profile'] or set(p['stages'])!=set(p['units']) or len(p['stages'])!=len(set(p['stages'])):raise ValueError('Scientific unit roster differs')
        prior=set()
        for stage in p['stages']:
            u=p['units'][stage]
            if u.get('id')!=stage or not set(u['dependencies'])<=prior or len(u['dependencies'])!=len(set(u['dependencies'])):raise ValueError('Protocol must declare a unique topological DAG')
            prior.add(stage)
        if w['protocol_sha256']!=digest(p):raise ValueError('Frozen protocol changed')
        site=validate_profile(w['slurm_profile'],root=self.root)
        if w['slurm_profile_sha256']!=digest(site):raise ValueError('Frozen native profile changed')
        if self.resources(p,site,ready=False)!=w['resources']:raise ValueError('Frozen scheduler resources differ from declared science')
        return w

    def load(self,path,*,verify=True):
        path=self.path(path);w=self.validate(self.read(path),historical=not verify)
        if path!=Path(w['run_dir'])/MANIFEST:raise ValueError('Manifest has a different root')
        if verify:
            if self.read(w['protocol_path'])!=self.protocol(w):raise ValueError('Scientific protocol bytes changed')
            if load_profile(root=self.root)!=w['slurm_profile']:raise ValueError('Native profile changed')
            if cw.source_hash()!=w['source_sha256']:raise ValueError('Source changed; preserve this run and use a fresh protocol')
            recovery=w.get('recovery')
            if recovery:
                if cw.digest(recovery['origin_workflow'])!=recovery['origin_workflow_sha256']:raise ValueError('Recovery origin identity changed')
                for stage,checksum in recovery['seals'].items():
                    if stage not in self.units(w) or cw.digest(self.stage_path(w,stage)/'workflow-seal.json')!=checksum:raise ValueError('Inherited recovery seal changed')
        return w

    def resources(self,p,site,*,ready=True):
        result={}
        for name,u in p['units'].items():
            identifier(name)
            if u.get('seconds') is None or u.get('walltime') is None:
                if ready:raise ValueError('Downstream allocations unavailable without valid measured calibration')
                result[name]=None;continue
            seconds=shared.wall_seconds(u['walltime'])
            if (u['device'] not in ('cpu','cuda') or type(u['cpus']) is not int or not 1<=u['cpus']<=8
                    or u['device']=='cuda' and u['cpus']!=4 or type(u['mem_gib']) is not int or not 1<=u['mem_gib']<=48
                    or not 180<=seconds<=2100 or not 0<u['seconds']<=min(1800,seconds-120)):
                raise ValueError('Unit exceeds native CPU/host/GPU/35-minute envelope')
            result[name]={k:u[k] for k in ('cpus','mem_gib','walltime')}
            result[name]['partition']=site['gpu_partition' if u['device']=='cuda' else 'cpu_partition']
        return result

    def binding(self,run,*,required_gate=None):
        campaign=self.path(self.root/'runs'/identifier(run)/CAMPAIGN)
        if campaign.is_file():
            receipt=self.read(self.read(campaign)['receipt_path'])
            if receipt.get('status')!='COMPLETED' or 'learning' not in receipt['run_ids']:raise ValueError('Prerequisite campaign has no completed literal learning decision')
            run=receipt['run_ids']['learning']
        w=self.load(self.workflow_path(run));status=self.status(w)
        if not status['terminal'] or not status['completed']:raise ValueError('Upstream native workflow is not terminal and complete')
        self.validate_results(w)
        decision=self.decision(w) if any(u['kind']=='decision' for u in self.units(w).values()) else dict(gate=None,decision='NOT_APPLICABLE',eligible=False)
        if required_gate and (decision.get('gate')!=required_gate or decision.get('decision')!='PASS' or decision.get('eligible') is not True):
            raise ValueError('A literal passing native upstream decision is required')
        return dict(run_id=w['run_id'],phase=w['phase'],profile=w['profile'],workflow_path=str(self.workflow_path(run)),
            workflow_sha256=cw.digest(self.workflow_path(run)),protocol_sha256=w['protocol_sha256'],source_sha256=w['source_sha256'],
            source_tree_sha256=w['source_tree_sha256'],slurm_profile_sha256=w['slurm_profile_sha256'],
            software_sha256=cw.digest(Path(w['run_dir'])/'state/submission-software.json'),
            stage_paths={s:str(self.stage_path(w,s)) for s in self.stages(w)},
            seals={s:cw.digest(self.stage_path(w,s)/'workflow-seal.json') for s in self.stages(w)},decision=decision)

    def read_calibration(self,run):
        bound=self.binding(run,required_gate='CALIBRATION');w=self.load(self.workflow_path(run));values={}
        if w['phase']!='calibration':raise ValueError('Calibration identity must name the actual fit-layout calibration phase')
        for stage,u in self.units(w).items():
            if u['kind']!='calibration':continue
            path=self.stage_path(w,stage);value=self.read(path/'calibration.json')
            if (u['device']!='cuda' or value.get('native') is not True or value.get('device')!='cuda' or value.get('completed_updates',value.get('updates',0))<64
                    or value.get('status') not in ('MEASURED','COMPLETED_UPDATES','COMPLETED')
                    or value.get('execution_mode','desktop-slurm')!='desktop-slurm'):
                raise ValueError('Calibration requires 64 actual complete CUDA updates for every exact workload')
            if value.get('workload_sha256')!=u['workload_sha256'] or not isinstance(value.get('seconds_per_update'),(int,float)) or isinstance(value.get('seconds_per_update'),bool) or not math.isfinite(value['seconds_per_update']) or value['seconds_per_update']<=0:
                raise ValueError('Calibration must retain the exact declared workload and positive measured update duration')
            if u['arm'] in values:raise ValueError('Repeated learned calibration arm')
            self.check_gpu_tests(self.junit(w,stage),self.protocol(w));values[u['arm']]=value
        if set(values)!=set(self.protocol(w)['training']['arms']):raise ValueError('Native calibration lacks a declared learned workload')
        return dict(binding=bound,data=dict(native=True,device='cuda',execution_mode='desktop-slurm',measurements=values),verified_native_gpu=True,execution_mode='desktop-slurm',
                    source_tree_sha256=w['source_tree_sha256'],slurm_profile_sha256=w['slurm_profile_sha256'])

    def prepare(self,args):
        builder=self.builder
        if builder is None:builder=importlib.import_module('tdn.analysis.tangent_response.protocol').build_protocol
        site=load_profile(root=self.root)
        run=identifier(args.run_id or fresh('tangent-response-'+args.phase))
        base=self.path(self.root/'runs'/run)
        if base.exists():raise ValueError('Run exists; preserve it and use a fresh identifier')
        bindings={}
        for item in getattr(args,'prerequisite',[]) or []:
            phase,sep,name=item.partition('=')
            bound=self.binding(name if sep else phase)
            key=phase if sep else bound['phase']
            if key in bindings:raise ValueError('Duplicate prerequisite phase')
            bindings[key]=bound
        needed={'compression':('mechanism','A'),'fit_prepare':('compression','B'),'calibration':('compression','B'),
                'learning':('compression','B'),'utility':('learning','C')}.get(args.phase)
        if needed and args.command!='plan':
            b=bindings.get(needed[0]);d=(b or {}).get('decision',{})
            if d.get('gate')!=needed[1] or d.get('decision')!='PASS' or d.get('eligible') is not True:raise ValueError('Required phase gate does not authorize this native expansion')
        if args.phase in ('calibration','learning') and args.command!='plan':
            d=bindings.get('fit_prepare',{}).get('decision',{})
            if d.get('gate')!='C0' or d.get('decision')!='PASS' or d.get('eligible') is not True:raise ValueError('Passing fresh fit screen is required before learned calibration or training')
        if args.profile=='development':
            b=bindings.get('pilot') or bindings.get('learning');d=(b or {}).get('decision',{})
            if args.command!='plan' and (d.get('gate')!='C' or d.get('decision')!='PASS' or b.get('profile') not in ('pilot','fit-pilot')):raise ValueError('Development requires a literal prerequisite pilot decision')
        calibration=self.read_calibration(args.calibration) if getattr(args,'calibration',None) else None
        if args.phase=='learning' and args.command!='plan' and calibration is None:raise ValueError('Learning allocations need exact native workload calibration')
        enabled=() if args.no_worker_telemetry else tuple(args.worker_telemetry.split(','))
        p=builder(profile=args.profile,target=args.target,phase=args.phase,prerequisites=bindings,calibration=calibration,
                  worker_telemetry=enabled,sampling_seconds=args.sampling_seconds,run_id=run)
        w=dict(schema_version=1,kind='desktop-slurm-tangent-response',execution_mode='desktop-slurm',run_id=run,
            profile=args.profile,phase=args.phase,root=str(self.root),run_dir=str(base),created_at=cw.now(),
            scientific_protocol=p,protocol_path=str(base/'protocol.json'),protocol_sha256=digest(p),
            slurm_profile=site,slurm_profile_path=str(base/'slurm-profile.json'),slurm_profile_sha256=digest(site),
            torch_version=site['torch_version'],source_sha256=cw.source_hash(),source_tree_sha256=cw.source_hash(metadata=True),
            resources=self.resources(p,site,ready=args.command!='plan'))
        if getattr(args,'campaign_run_id',None):w['campaign_run_id']=identifier(args.campaign_run_id)
        return self.validate(w)

    def comment(self,w,stage):return f"tdn-tangent-response:{w['run_id']}:{stage}:{w['slurm_profile_sha256']}"

    def scheduler_args(self,w,stage,dependency=None):
        r=w['resources'][stage];u=self.units(w)[stage];base=Path(w['run_dir'])
        if r is None:raise ValueError('Uncalibrated downstream scheduling is unavailable')
        args=['sbatch','--parsable',f"--partition={r['partition']}",'--nodes=1','--ntasks=1',
              '--job-name=tdn-tangent-response-'+stage,'--comment='+self.comment(w,stage),
              f"--cpus-per-task={r['cpus']}",f"--mem={r['mem_gib']}G",'--time='+r['walltime'],
              '--signal=USR1@120','--export=ALL','--chdir='+str(self.root),'--open-mode=append','--kill-on-invalid-dep=yes',
              '--output='+str(base/'logs'/f'{stage}-%j.out'),'--error='+str(base/'logs'/f'{stage}-%j.err'),
              '--gres='+ (w['slurm_profile']['gpu_gres'] if u['device']=='cuda' else 'none')]
        if w['slurm_profile'].get('account'):args+=['--account='+w['slurm_profile']['account']]
        if dependency:args+=['--dependency='+dependency]
        return args+[str(self.root/'scripts/fedora_tangent_response.sh'),'_worker-batch']

    def dependency(self,w,stage,jobs):
        if self.units(w)[stage]['kind']=='report' and jobs:
            return 'afterany:'+':'.join(row['job_id'] for row in jobs)
        observers=self.units(w)[stage]['kind'] in ('report','decision')
        parents={row['stage']:row['job_id'] for row in jobs}
        ids=[parents[d] for d in self.units(w)[stage]['dependencies'] if d in parents]
        groups=[]
        if ids:groups.append(('afterany' if observers else 'afterok')+':'+':'.join(ids))
        if jobs and jobs[-1]['job_id'] not in ids:groups.append('afterany:'+jobs[-1]['job_id'])
        return ','.join(groups) or None

    def environment(self,w,stage):
        env={k:v for k,v in shared.clean_environment().items() if not k.startswith('TDN_TANGENT_')}
        env.update(TDN_EXECUTION_MODE='desktop-slurm',TDN_PROJECT_ROOT=str(self.root),TDN_REPO_ROOT=str(self.root),
            TDN_SLURM_CONFIG=w['slurm_profile_path'],TDN_TANGENT_WORKFLOW=str(Path(w['run_dir'])/MANIFEST),
            TDN_TANGENT_STAGE=stage,TDN_TANGENT_DEVICE=self.units(w)[stage]['device'],
            TDN_TANGENT_CPUS=str(w['resources'][stage]['cpus']),TDN_FEDORA_GPU_GRES=w['slurm_profile']['gpu_gres'])
        return env

    def submit(self,w,*,plan=False):
        print(f"Tangent response {w['run_id']} ({w['profile']}/{w['phase']})")
        for stage in self.stages(w):
            if w['resources'][stage] is None:print(stage+': UNAVAILABLE until exact workload calibration')
            else:print(shlex.join(self.scheduler_args(w,stage)))
        if w['phase']=='mechanism':print('Downstream B/C/D allocations: UNAVAILABLE until sealed passing decisions and required native calibration')
        if plan:print('PLAN ONLY: no writes, scheduler calls, or numerical work');return w
        self.validate(w)
        importlib.import_module('tdn.analysis.tangent_response.protocol').validate_protocol(self.protocol(w),require_binding=True)
        if cw.source_hash()!=w['source_sha256'] or cw.source_hash(metadata=True)!=w['source_tree_sha256']:raise ValueError('Source changed before submission; preserve the plan and use a fresh identity')
        fw.controller_policy();self.resources(self.protocol(w),w['slurm_profile'])
        policies=shared.check_capacity(w)
        with rw.acquire_venv_lock():software=rw.software_report(w)
        with controller_lock(self.root):
            base=Path(w['run_dir']);base.mkdir(parents=True,exist_ok=False);(base/'logs').mkdir()
            for file,value in ((MANIFEST,w),('protocol.json',self.protocol(w)),('slurm-profile.json',w['slurm_profile'])):
                write(base/file,value);(base/file).chmod(0o444)
            write(base/'state/submission-software.json',software);write(base/'state/slurm-policy.json',policies)
            jobs=[];write(base/'jobs.json',jobs);write(self.root/'runs'/POINTER,{'run_id':w['run_id']})
            def queue(stage):
                self.load(base/MANIFEST)
                dependency=self.dependency(w,stage,jobs)
                intent=base/'state/submission-intents'/f'{stage}.json'
                if intent.exists():raise ValueError('Submission intent exists; reconcile ambiguity before any recovery')
                write(intent,dict(stage=stage,status='SUBMITTING',created_at=cw.now(),comment=self.comment(w,stage),dependency=dependency))
                response=self.command(self.scheduler_args(w,stage,dependency),env=self.environment(w,stage))
                match=re.fullmatch(r'([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?',response.stdout.strip())
                if not match or any(row['job_id']==match.group(1) for row in jobs):raise ValueError('Ambiguous native submission; inspect retained intent, never retry automatically')
                jobs.append(dict(stage=stage,job_id=match.group(1),dependency=dependency,submitted_at=cw.now()))
                write(base/'jobs.json',jobs);write(intent,dict(stage=stage,status='SUBMITTED',job_id=match.group(1),comment=self.comment(w,stage)))
                print(stage+' job '+match.group(1),flush=True)
            for stage in self.stages(w):
                if stage in w.get('recovery',{}).get('stage_paths',{}):continue
                try:queue(stage)
                except BaseException as error:
                    if stage!='report':
                        try:queue('report')
                        except Exception as report_error:print('Final report submission failed: '+str(report_error),file=sys.stderr)
                    write(base/'state/submission.json',dict(status='FAILED',error=str(error),submitted_jobs=jobs));raise
            write(base/'state/submission.json',dict(status='COMPLETED',updated_at=cw.now()))
        return w


    def verify_science(self,w,stage,*,historical=False):
        verifier=self.science_verifier
        if verifier is None:verifier=importlib.import_module('tdn.analysis.tangent_response.engine').verify_science
        path=self.stage_path(w,stage)
        options={'sourcehash':w['source_tree_sha256']}
        if historical:options['historical']=True
        result=verifier(path,self.protocol(w),**options)
        execution=self.read(path/'execution.json')
        expected=dict(stage=stage,profile=w['profile'],device=self.units(w)[stage]['device'],execution_mode='desktop-slurm')
        if any(execution.get(k)!=v for k,v in expected.items()):raise ValueError('Science execution identity differs from its allocated unit')
        return result

    def verify_stage(self,w,stage,*,historical=False):
        self.verify_science(w,stage,historical=historical)
        path=self.stage_path(w,stage);seal=self.read(path/'workflow-seal.json')
        if (seal.get('schema')!='tdn.tangent-response-workflow-seal/v1' or seal.get('stage')!=stage
                or seal.get('source_tree_sha256')!=w['source_tree_sha256'] or seal.get('protocol_sha256')!=w['protocol_sha256']
                or seal.get('slurm_profile_sha256')!=w['slurm_profile_sha256']
                or seal.get('science_manifest_sha256')!=cw.digest(path/'science_manifest.json')
                or seal.get('status')!='COMPLETED'):
            raise ValueError('Terminal native workflow seal differs')
        worker=self.read(self.path(self.root/seal['worker_state_path']))
        if (worker.get('status')!='COMPLETED' or worker.get('source_sha256')!=w['source_sha256']
                or worker.get('protocol_sha256')!=w['protocol_sha256'] or worker.get('slurm_job_id')!=seal.get('slurm_job_id')):
            raise ValueError('Sealed worker state differs from the frozen native execution')
        for relative,checksum in seal['artifacts'].items():
            candidate=self.path(self.root/relative)
            if not candidate.is_file() or candidate.is_symlink() or cw.digest(candidate)!=checksum:raise ValueError('Native worker artifact changed: '+relative)
        return seal

    def validate_results(self,w,*,historical=False):
        result={'run_id':w['run_id'],'validation':'tangent-response-native-seals','units':{}}
        for stage in self.stages(w):
            try:self.verify_stage(w,stage,historical=historical);result['units'][stage]={'validation':'VERIFIED'}
            except (OSError,ValueError,KeyError,TypeError,ImportError) as error:
                result['units'][stage]={'validation':'INVALID' if self.stage_path(w,stage).exists() else 'MISSING','error':str(error)}
        if any(row['validation']!='VERIFIED' for row in result['units'].values()):
            raise ValueError(json.dumps(result,sort_keys=True))
        return result

    def decision(self,w,*,historical=False):
        stages=[s for s,u in self.units(w).items() if u['kind']=='decision']
        if len(stages)!=1:raise ValueError('Phase needs exactly one sealed scientific decision')
        self.verify_stage(w,stages[0],historical=historical);value=self.read(self.stage_path(w,stages[0])/'decision.json')
        if value.get('decision') not in ('PASS','STOP','UNRESOLVED','IMPLEMENTATION_ONLY') or type(value.get('eligible')) is not bool:raise ValueError('Invalid prospective gate decision')
        if value['decision']=='PASS' and value['eligible'] is not True:raise ValueError('Passing decision lacks literal eligibility')
        return value

    def status(self,w):
        jobs={r['stage']:r for r in self.read(Path(w['run_dir'])/'jobs.json')}
        result=dict(run_id=w['run_id'],profile=w['profile'],phase=w['phase'],terminal=True,completed=True,failed_units=[],report_complete=False,units={})
        for stage in self.stages(w):
            inherited=stage in w.get('recovery',{}).get('stage_paths',{})
            job=jobs.get(stage);state='COMPLETED' if inherited else 'NOT_SUBMITTED'
            if job:
                try:state=cw.scheduler_state(job['job_id']).split()[0].rstrip('+')
                except (OSError,ValueError):state='UNKNOWN'
                if state not in TERMINAL:result['terminal']=False
            file=self.state(w,stage);worker=self.read(file) if file.is_file() else {}
            path=self.stage_path(w,stage);has_seal=(path/'workflow-seal.json').is_file() and (path/'COMPLETED').is_file()
            complete=state=='COMPLETED' and (inherited or worker.get('status')=='COMPLETED') and has_seal
            result['completed'] &= complete
            if state in TERMINAL-{'COMPLETED'} or state=='NOT_SUBMITTED' or state=='COMPLETED' and not complete:result['failed_units'].append(stage)
            result['units'][stage]=dict(job_id=job['job_id'] if job else None,scheduler_state=state,
                worker_status=worker.get('status','INHERITED' if inherited else 'UNKNOWN'),artifact_validation='SEALED_UNVERIFIED' if has_seal else 'MISSING')
            if self.units(w)[stage]['kind']=='report':result['report_complete']=complete
        return result

    def ancestors(self,w,stage):
        found=set()
        def visit(name):
            for parent in self.units(w)[name]['dependencies']:
                if parent not in found:found.add(parent);visit(parent)
        visit(stage)
        return tuple(s for s in self.stages(w) if s in found)

    def verify_software(self,w,software):
        if self.read(Path(w['run_dir'])/'state/submission-software.json')!=software:raise ValueError('Exact native software changed')

    def verify_prerequisites(self,w,software,stage):
        hashes={}
        for parent in self.ancestors(w,stage):
            self.verify_stage(w,parent)
            hashes[parent]=cw.digest(self.stage_path(w,parent)/'workflow-seal.json')
        for phase,b in self.protocol(w).get('prerequisites',{}).items():
            old=self.load(b['workflow_path'])
            software_path=Path(old['run_dir'])/'state/submission-software.json'
            if (old['source_sha256']!=w['source_sha256'] or old['slurm_profile_sha256']!=w['slurm_profile_sha256']
                    or b['workflow_sha256']!=cw.digest(b['workflow_path'])
                    or b['software_sha256']!=cw.digest(software_path) or self.read(software_path)!=software):
                raise ValueError('Upstream source/profile/software binding differs')
            for parent,checksum in b['seals'].items():
                self.verify_stage(old,parent)
                if cw.digest(self.stage_path(old,parent)/'workflow-seal.json')!=checksum:raise ValueError('Upstream native seal changed')
                hashes[phase+'/'+parent]=checksum
        return hashes

    def terminal_jobs(self,w):
        jobs=self.read(Path(w['run_dir'])/'jobs.json')
        # State-only success is insufficient for archival/recovery. Use exact
        # ownership plus the live inventory, including unregistered submissions.
        from scripts.tangent_response_archive import TangentScheduler
        if not isinstance(jobs,list) or len(jobs)>20000 or len({r['stage'] for r in jobs})!=len(jobs) or len({r['job_id'] for r in jobs})!=len(jobs):raise ValueError('Origin has ambiguous allocation inventory')
        for path in (Path(w['run_dir'])/'state/submission-intents').glob('*.json'):
            intent=self.read(path)
            if intent.get('status')!='SUBMITTED' or not any(r['stage']==intent.get('stage') and r['job_id']==intent.get('job_id') for r in jobs):
                raise ValueError('Unresolved submission intent requires explicit scheduler investigation; never rerun ambiguous work')
        expected=[]
        for row in jobs:
            if row['stage'] not in self.units(w) or not pw.JOB.fullmatch(row['job_id']):raise ValueError('Unsafe origin job identity')
            expected.append({**row,'expected_name':'tdn-tangent-response-'+row['stage'],'expected_comment':self.comment(w,row['stage'])})
        observed=TangentScheduler().observe(expected,user=w['slurm_profile']['user'],root=self.root,run_ids=[w['run_id']])
        for job in expected:
            row=observed.get(job['job_id'],{});state=row.get('state','UNKNOWN').split()[0].rstrip('+')
            if state not in TERMINAL:raise ValueError('Recovery requires all actual submitted jobs terminal')
            if (row.get('user')!=w['slurm_profile']['user'] or row.get('workdir')!=str(self.root)
                    or row.get('job_name')!=job['expected_name'] or row.get('comment') and row['comment']!=job['expected_comment']
                    or w['slurm_profile'].get('account') is not None and row.get('account')!=w['slurm_profile']['account']):
                raise ValueError('Recovery scheduler ownership differs')
        return observed

    def recover(self,run,new_id,*,plan=False):
        origin=self.load(self.workflow_path(run));self.terminal_jobs(origin)
        old_software=self.read(Path(origin['run_dir'])/'state/submission-software.json')
        with rw.acquire_venv_lock():
            if rw.software_report(origin)!=old_software:raise ValueError('Exact recovery software changed')
        completed,resume,seals={},{},{}
        for stage,u in self.units(origin).items():
            if u['kind'] in ('report','decision'):continue
            path=self.stage_path(origin,stage)
            if (path/'workflow-seal.json').is_file():
                self.verify_stage(origin,stage);completed[stage]=str(path);seals[stage]=cw.digest(path/'workflow-seal.json')
            elif u['kind'] in ('training','calibration') and (path/'journal.pt').is_file():
                if (path/'COMPLETED').exists():raise ValueError('A completed unit without valid native seal cannot be resumed')
                execution=self.read(path/'execution.json')
                expected=dict(source_tree_sha256=origin['source_tree_sha256'],protocol_sha256=origin['protocol_sha256'],stage=stage,profile=origin['profile'],device=u['device'],execution_mode='desktop-slurm')
                if any(execution.get(k)!=v for k,v in expected.items()):raise ValueError('Interrupted journal has a different frozen execution identity')
                resume[stage]=str(path)
        base=self.path(self.root/'runs'/identifier(new_id))
        if base.exists():raise ValueError('Recovery destination already exists')
        w={**origin,'run_id':new_id,'run_dir':str(base),'created_at':cw.now(),'protocol_path':str(base/'protocol.json'),
           'slurm_profile_path':str(base/'slurm-profile.json'),
           'recovery':dict(origin_run=origin['run_id'],origin_workflow=str(self.workflow_path(run)),
                          origin_workflow_sha256=cw.digest(self.workflow_path(run)),stage_paths=completed,resume_paths=resume,seals=seals)}
        self.submit(w,plan=plan)
        return w

    def verify_allocation(self,w,stage):
        if load_profile(root=self.root)!=w['slurm_profile']:raise ValueError('Current native profile differs')
        mode=self.units(w)[stage]['device'];allocated=runtime_allocation(mode,profile=w['slurm_profile'],root=self.root)
        job,fields=allocated['job_id'],allocated['job'];res=w['resources'][stage]
        expected=dict(JobId=job,JobName='tdn-tangent-response-'+stage,Comment=self.comment(w,stage),JobState='RUNNING',WorkDir=str(self.root),Partition=res['partition'])
        if any(fields.get(k)!=v for k,v in expected.items()):raise ValueError('Real allocation is not this tangent-response unit')
        recorded=[r for r in self.read(Path(w['run_dir'])/'jobs.json') if r['stage']==stage]
        if len(recorded)!=1 or recorded[0]['job_id']!=job:raise ValueError('Actual job ID differs from the exact submission receipt')
        tres=dict(item.split('=',1) for item in fields.get('AllocTRES','').split(',') if '=' in item)
        if (tres.get('cpu')!=str(res['cpus']) or shared.memory_mib(tres.get('mem',''))!=res['mem_gib']*1024
                or shared.wall_seconds(fields.get('TimeLimit',''))!=shared.wall_seconds(res['walltime'])):
            raise ValueError('Actual allocation budget differs from frozen resources')
        return allocated

    def junit(self,w,stage):return Path(w['run_dir'])/'reporter-tests'/stage/'tests.xml'

    def check_gpu_tests(self,path,p):
        from xml.etree import ElementTree
        tree=ElementTree.parse(path).getroot();cases=tree.findall('.//testcase')
        expected=[]
        for case in p.get('gpu_test_cases',[]):
            if isinstance(case,dict):expected.append((case['classname'],case['name']))
            elif isinstance(case,(list,tuple)) and len(case)==2:expected.append(tuple(case))
            elif isinstance(case,str) and '::' in case:
                module,name=case.split('::',1);expected.append((module.replace('/','.').removesuffix('.py'),name))
            elif isinstance(case,str) and len(p.get('gpu_test_files',[]))==1:
                expected.append((p['gpu_test_files'][0].replace('/','.').removesuffix('.py'),case))
            else:raise ValueError('CUDA case manifest needs exact file/class and testcase identities')
        actual=[(c.get('classname'),c.get('name')) for c in cases]
        if not expected or Counter(actual)!=Counter(expected) or len(set(actual))!=len(actual):raise ValueError('Mandatory new-namespace CUDA roster incomplete or duplicated')
        if any(c.find(tag) is not None for c in cases for tag in ('failure','error','skipped')):raise ValueError('Every required actual CUDA case must pass without skips')
        return dict(validation='mandatory_tangent_response_cuda',test_cases=len(actual))

    def worker_commands(self,w,stage):
        base=Path(w['run_dir']);python=str(self.root/'.venv/bin/python');u=self.units(w)[stage];gpu=u['device']=='cuda';commands=[]
        if gpu:
            files=self.protocol(w).get('gpu_test_files',[])
            if not files:raise ValueError('Required native CUDA manifest is missing')
            commands.append(('preflight',[python,str(self.root/'scripts/fedora_gpu_preflight.py'),'--output',str(base/f'{stage}-gpu-preflight.json')]))
        elif u['kind']=='audit':files=self.protocol(w).get('cpu_test_files') or [p.relative_to(self.root).as_posix() for p in sorted((self.root/'tests').glob('test_tangent_response*.py'))]
        else:files=[]
        if files:
            if any(not self.path(self.root/name).is_file() for name in files):raise ValueError('Mandatory correctness test file missing')
            commands.append(('gpu-tests' if gpu else 'tests',[python,'-m','pytest','-q','-m','gpu' if gpu else 'not gpu',
                *[str(self.path(self.root/name)) for name in files],'--basetemp',str(base/(stage+'-pytest-work')),
                '-o','cache_dir='+str(base/(stage+'-pytest-cache')),'--junitxml',str(self.junit(w,stage))]))
            if gpu:commands.append(('check-gpu-tests',[python,str(self.root/'scripts/tangent_response_workflow.py'),'check-gpu-tests','--protocol',w['protocol_path'],'--junit',str(self.junit(w,stage))]))
        args=[python,str(self.root/'scripts/tangent_response.py'),'--stage',stage,'--protocol',w['protocol_path'],
              '--run-dir',str(base/stage),'--device',u['device']]
        for parent in self.ancestors(w,stage):args+=['--prerequisite-dir',parent+'='+str(self.stage_path(w,parent))]
        for phase,b in self.protocol(w).get('prerequisites',{}).items():
            for parent,path in b['stage_paths'].items():args+=['--prerequisite-dir',phase+'/'+parent+'='+path]
        resume=w.get('recovery',{}).get('resume_paths',{}).get(stage)
        if resume:args+=['--resume-from',resume]
        commands.append(('experiment',args));return commands

    def worker(self,w,stage):
        if stage not in self.units(w) or stage in w.get('recovery',{}).get('stage_paths',{}):raise ValueError('Unknown or inherited unit cannot execute')
        report_module=importlib.import_module('tdn.analysis.tangent_response.report')
        telemetry=importlib.import_module('tdn.analysis.tangent_response.telemetry')
        ctx=SimpleNamespace(path=Path(w['run_dir'])/stage,stage=stage,protocol=self.protocol(w),unit=self.units(w)[stage],device=self.units(w)[stage]['device'])
        started=time.monotonic();attempt=None;observer=None;runtime=None
        def begin(bound,unit):
            nonlocal attempt,observer
            attempt=report_module.start_attempt(ctx,job_id=os.environ['SLURM_JOB_ID'],sourcehash=w['source_tree_sha256'],worker_managed=True)
            observer=telemetry.start_recording(attempt,options=self.protocol(w).get('worker_telemetry',{}),pid=os.getpid(),environ=os.environ)
            return attempt
        def finish(bound,unit,report,**kwargs):
            nonlocal runtime
            runtime=kwargs['runtime_seconds']
        def prepare_tests(env,name):
            for key in tuple(env):
                if key.startswith(('TDN_TANGENT_','RAVEN_')) or key=='TDN_TOWER_DIR':env.pop(key,None)
            env.pop('PYTEST_ADDOPTS',None)
            if name=='gpu-tests':env['TDN_REQUIRE_TANGENT_RESPONSE_GPU_TESTS']='1'
        backend=SimpleNamespace(verify_allocation=self.verify_allocation,load=self.load,begin_report=begin,
            finish_report=finish,worker_commands=self.worker_commands,verify_stage=self.verify_science,
            verify_software=self.verify_software,verify_prerequisites=lambda bound,software:self.verify_prerequisites(bound,software,stage),
            prepare_test_environment=prepare_tests,gpu_junit_path=lambda bound:self.junit(bound,stage),workflow_filename=MANIFEST,
            workflow_label='tangent-response',stage_variable='TDN_TANGENT_STAGE',
            prerequisite_stages=tuple(s for s,u in self.units(w).items() if u['kind'] not in ('audit','decision','report')),
            workflow_environment=lambda bound,unit:{'TDN_TANGENT_WORKFLOW':str(Path(bound['run_dir'])/MANIFEST),
                'TDN_TANGENT_PROTOCOL_SHA256':bound['protocol_sha256'],'TDN_TANGENT_STAGE':unit})
        try:return pw.worker(w,stage,backend=backend)
        finally:
            state=self.read(self.state(w,stage)) if self.state(w,stage).is_file() else dict(status='FAILED',exit_code=2,error='Worker startup failed')
            if observer is not None:
                try:coverage=telemetry.stop_recording(observer,state=state['status'])
                except Exception as error:coverage=dict(status='unavailable',error=str(error),index=None)
                state['worker_telemetry']=coverage;write(self.state(w,stage),state)
            if attempt is not None:
                try:
                    body=self.read(ctx.path/'summary.json') if (ctx.path/'summary.json').is_file() else {}
                    report_module.finish_attempt(ctx,{**body,'status':state['status'],'science_body_status':body.get('status','NOT_STARTED'),
                        'native_worker_finalized':True,'native_worker_runtime_seconds':runtime or time.monotonic()-started,
                        'native_worker_exit_code':state.get('exit_code'),'native_worker_error':state.get('error'),
                        'worker_telemetry':state.get('worker_telemetry')})
                    if state['status']=='COMPLETED':
                        artifacts={}
                        owned=[self.state(w,stage),Path(w['run_dir'])/'state'/f'{stage}-software.json',
                               Path(w['run_dir'])/'state/submission-software.json',*(p for p in Path(attempt).rglob('*') if p.is_file())]
                        if self.units(w)[stage]['device']=='cuda':owned.extend([self.junit(w,stage),Path(w['run_dir'])/f'{stage}-gpu-preflight.json'])
                        for path in owned:
                            safe=self.path(path)
                            if safe.is_symlink():raise ValueError('Native reporting artifact symlink is invalid')
                            artifacts[safe.relative_to(self.root).as_posix()]=cw.digest(safe)
                        write(ctx.path/'workflow-seal.json',dict(schema='tdn.tangent-response-workflow-seal/v1',stage=stage,status='COMPLETED',
                            source_tree_sha256=w['source_tree_sha256'],protocol_sha256=w['protocol_sha256'],
                            slurm_profile_sha256=w['slurm_profile_sha256'],slurm_job_id=state['slurm_job_id'],
                            worker_state_path=self.state(w,stage).relative_to(self.root).as_posix(),
                            science_manifest_sha256=cw.digest(ctx.path/'science_manifest.json'),artifacts=artifacts))
                except BaseException as error:
                    state.update(status='FAILED',exit_code=2,error='Terminal native reporting/seal failed: '+str(error));write(self.state(w,stage),state)
                    raise rw.WorkerFailure(state['error'],2) from error

def process_identity():
    import socket
    pid=os.getpid()
    # Linux start ticks distinguish a dead process from a reused PID.
    fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,host=socket.gethostname(),start_ticks=fields[19])


def coordinator_dead(record):
    import socket
    if record.get('host')!=socket.gethostname():raise ValueError('Coordinator host differs; stopped-process evidence unavailable')
    pid=record.get('pid')
    if type(pid) is not int or pid<=0:raise ValueError('Coordinator process identity missing')
    try:current=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError:return True
    return current!=record.get('start_ticks')


class CampaignCoordinator:
    def __init__(self,native,args,*,sleep=time.sleep,bridge=None):
        self.native,self.args,self.sleep=native,args,sleep
        self.bridge=bridge
        self.ctx=None
        if bridge is None and os.environ.get('RAVEN_LAUNCH_PLAN'):
            self.bridge=importlib.import_module('scripts.tangent_response_raven')
        if self.bridge is not None:self.ctx=self.bridge.context(os.environ.get('TDN_TANGENT_CAPTURED_SCRIPT'))

    def receipt_path(self,run):return self.native.path(self.native.root/'runs/.tangent-response-campaigns'/identifier(run)/'receipt.json')

    def event(self,phase,status,detail=''):
        now=cw.now();self.receipt.update(phase=phase,status=status,updated_at=now)
        self.receipt['events'].append(dict(time=now,phase=phase,status=status,detail=detail))
        write(self.path,self.receipt)
        write(self.native.root/'runs'/self.receipt['run_id']/'campaign-receipt.json',self.receipt)
        if self.ctx is not None:self.bridge.publish(self.ctx,self.path)

    def wait(self,w):
        while True:
            status=self.native.status(w)
            path=self.path.parent/(w['run_id']+'-status.json');write(path,status)
            if self.ctx is not None:self.bridge.publish(self.ctx,self.path,status_file=path)
            print(f"Tangent response: {w['run_id']} {'terminal' if status['terminal'] else 'pending'}",flush=True)
            if status['terminal']:return status
            self.sleep(30)

    def compact(self,run):
        result=subprocess.run([str(self.native.root/'.venv/bin/python'),str(self.native.root/'scripts/compact_review.py'),
                               'pack',run,'--mode','review'],cwd=self.native.root,check=False)
        if result.returncode:raise ValueError('Exact review export verification failed')

    def start(self,*,continuing=False):
        run=identifier(self.args.run_id or fresh('tangent-response-'+self.args.profile))
        if len(run)>80:raise ValueError('Campaign IDs must leave room for unique phase suffixes (at most80 characters)')
        self.path=self.receipt_path(run);parent=self.native.path(self.native.root/'runs'/run)
        if continuing:
            self.receipt=self.native.read(self.path)
            if self.receipt['status'] not in ('FAILED','INTERRUPTED'):raise ValueError('Continue a stopped/reconciled coordinator, never a live one')
            if not coordinator_dead(self.receipt['coordinator']):raise ValueError('Original coordinator is still alive')
            if self.args.profile!=self.receipt['profile'] or self.args.target!=self.receipt['target']:raise ValueError('Continuation profile/target differs from the original campaign')
            from scripts.tangent_response_archive import TangentArchive
            TangentArchive(self.native.root).outer(self.receipt,[])
            for child in self.receipt['run_ids'].values():self.native.terminal_jobs(self.native.load(self.native.workflow_path(child)))
            if self.receipt['source_sha256']!=cw.source_hash():raise ValueError('Exact coordinator continuation source differs')
            frozen=self.receipt.get('campaign_options')
            if not isinstance(frozen,dict):raise ValueError('Original campaign option snapshot is missing')
            if self.args.prerequisite and self.args.prerequisite!=frozen['prerequisite']:raise ValueError('Continuation prerequisite identities differ')
            for key,value in frozen.items():setattr(self.args,key,value)
            self.receipt.setdefault('coordinator_attempts',[]).append(self.receipt['coordinator'])
            self.receipt['coordinator']=process_identity()
            if self.ctx is not None:
                self.receipt.setdefault('raven_launch_history',[]).append(self.receipt.get('raven_launch'))
                self.receipt['raven_launch']=self.bridge.association(self.ctx)
        else:
            if parent.exists() or self.path.exists():raise ValueError('Campaign exists; preserve it and select a fresh identifier')
            parent.mkdir(parents=True,exist_ok=False);self.path.parent.mkdir(parents=True,exist_ok=False)
            self.receipt=dict(schema='tdn.tangent-response-campaign/v1',run_id=run,project_root=str(self.native.root),created_at=cw.now(),
                profile=self.args.profile,target=self.args.target,status='RUNNING',phase='CREATED',run_ids={},gates={},outcomes={},events=[],
                source_sha256=cw.source_hash(),coordinator=process_identity(),raven_launch=self.bridge.association(self.ctx) if self.ctx else None,
                campaign_options={key:getattr(self.args,key) for key in ('prerequisite','worker_telemetry','no_worker_telemetry','sampling_seconds')},
                requested_phases=list(PHASES),not_submitted={},automatic_retry=False,poll_interval_seconds=30)
            write(parent/CAMPAIGN,dict(schema='tdn.tangent-response-campaign-pointer/v1',run_id=run,receipt_path=str(self.path)))
            write(self.native.root/'runs'/POINTER,dict(run_id=run))
        write(self.path,self.receipt)
        write(parent/'campaign-receipt.json',self.receipt)
        if self.ctx is not None:self.bridge.initialize(self.ctx,self.path)
        print('Campaign: '+run+'\nReceipt: '+str(self.path),flush=True)
        return run

    def execute(self,*,continuing=False):
        run=self.start(continuing=continuing)
        replacements={}
        for item in getattr(self.args,'replacement',[]) or []:
            phase,sep,child=item.partition('=')
            if not sep or phase not in PHASES or phase in replacements:raise ValueError('Use unique --replacement PHASE=EXACT_RECOVERY_ID')
            replacements[phase]=identifier(child)
        prior={}
        # A development pilot prerequisite is an independent gate binding.
        initial=list(self.args.prerequisite or [])
        try:
            for phase in PHASES:
                if phase in replacements:
                    replacement=self.native.load(self.native.workflow_path(replacements[phase]))
                    original=self.native.load(self.native.workflow_path(self.receipt['run_ids'][phase]))
                    if replacement['protocol_sha256']!=original['protocol_sha256'] or replacement.get('recovery',{}).get('origin_run')!=original['run_id']:
                        raise ValueError('Explicit replacement must be exact recovery of this literal phase')
                    self.receipt.setdefault('replaced_run_ids',{}).setdefault(phase,[]).append(original['run_id'])
                    self.receipt['run_ids'][phase]=replacement['run_id']
                child=self.receipt['run_ids'].get(phase)
                self.event(phase,'RUNNING')
                if child:
                    w=self.native.load(self.native.workflow_path(child))
                    status=self.wait(w)
                else:
                    child=fresh(run+'-'+phase)
                    self.receipt['run_ids'][phase]=child
                    self.event(phase,'RUNNING','Fresh child identity recorded before submission')
                    kwargs=vars(self.args).copy();kwargs.update(command='run',phase=phase,run_id=child,campaign_run_id=run,
                        prerequisite=initial+[key+'='+value for key,value in prior.items()],calibration=prior.get('calibration'))
                    w=self.native.prepare(SimpleNamespace(**kwargs))
                    try:self.native.submit(w)
                    except BaseException:
                        # A partial submission's afterany report still owns its
                        # actual allocation. Wait for recorded jobs before the
                        # coordinator becomes terminal; never retry submission.
                        if (Path(w['run_dir'])/'jobs.json').is_file():
                            self.wait(w)
                            try:self.compact(child)
                            except (OSError,ValueError) as export_error:print('Partial evidence export: '+str(export_error),file=sys.stderr)
                        raise
                    status=self.wait(w)
                self.receipt['outcomes'][phase]=dict(terminal=status['terminal'],completed=status['completed'],failed_units=status['failed_units'])
                write(self.path,self.receipt)
                try:self.native.validate_results(w)
                except ValueError:
                    self.compact(child);raise ValueError('Phase infrastructure failed; preserve child and recover explicitly before coordinator continuation')
                if not status['completed']:
                    self.compact(child);raise ValueError('Phase jobs did not complete; no downstream expansion')
                prior[phase]=child
                if any(u['kind']=='decision' for u in self.native.units(w).values()):
                    decision=self.native.decision(w);self.receipt['gates'][phase]=decision;write(self.path,self.receipt)
                    if decision['decision']!='PASS' or decision['eligible'] is not True:
                        remaining=PHASES[PHASES.index(phase)+1:]
                        self.receipt['not_submitted']={p:'STOPPED_BY_'+decision['decision'] for p in remaining}
                        self.receipt['decision']=decision['decision'];self.event(phase,'COMPLETED','Negative or unresolved science; report verified and downstream not submitted')
                        self.compact(run);return self.receipt
            self.receipt['decision']=self.receipt['gates']['utility']['decision']
            self.event('FINISHED','COMPLETED','All declared eligible phases completed; no confirmation authorized')
            self.compact(run);return self.receipt
        except BaseException as error:
            self.event(self.receipt.get('phase','UNKNOWN'),'FAILED',str(error))
            raise


def parser():
    result=argparse.ArgumentParser(description=__doc__);commands=result.add_subparsers(dest='command',required=True)
    def options(item):
        item.add_argument('--profile',choices=('pilot','mechanism','compression','fit-pilot','development'),default='pilot')
        item.add_argument('--target',choices=('fd_same_grid','galerkin_same_grid'),default='fd_same_grid')
        item.add_argument('--run-id');item.add_argument('--prerequisite',action='append',default=[])
        item.add_argument('--calibration');item.add_argument('--worker-telemetry',default='cpu,memory,gpu')
        item.add_argument('--no-worker-telemetry',action='store_true');item.add_argument('--sampling-seconds',type=float,default=5)
    for name in ('plan','run'):
        item=commands.add_parser(name);options(item);item.add_argument('--phase',choices=PHASES,default='mechanism')
    item=commands.add_parser('campaign');options(item)
    item=commands.add_parser('continue');options(item);item.add_argument('--replacement',action='append',default=[])
    for name in ('status','validate','decision','paths','compact','reconcile'):
        item=commands.add_parser(name);item.add_argument('run')
        if name=='status':item.add_argument('--json',dest='json_output',action='store_true')
        if name=='compact':item.add_argument('--mode',choices=('review','replay'),default='review')
        if name=='reconcile':item.add_argument('--plan',action='store_true')
    item=commands.add_parser('recover');item.add_argument('run');item.add_argument('--run-id',required=True);item.add_argument('--plan',action='store_true')
    item=commands.add_parser('worker');item.add_argument('--workflow',required=True);item.add_argument('--stage',required=True)
    item=commands.add_parser('check-gpu-tests');item.add_argument('--protocol',required=True);item.add_argument('--junit',required=True)
    return result


def campaign_receipt(native,run):
    if run=='latest':run=native.read(native.root/'runs'/POINTER)['run_id']
    marker=native.path(native.root/'runs'/identifier(run)/CAMPAIGN)
    if not marker.is_file():return None,None
    value=native.read(marker)
    path=native.path(value['receipt_path']);receipt=native.read(path)
    if receipt.get('schema')!='tdn.tangent-response-campaign/v1' or receipt['run_id']!=run or receipt['project_root']!=str(native.root):raise ValueError('Campaign receipt identity differs')
    return path,receipt


def main(argv=None):
    args=parser().parse_args(argv);native=NativeWorkflow()
    try:
        if args.command in ('plan','run'):native.submit(native.prepare(args),plan=args.command=='plan')
        elif args.command in ('campaign','continue'):CampaignCoordinator(native,args).execute(continuing=args.command=='continue')
        elif args.command=='worker':return native.worker(native.load(args.workflow),args.stage)
        elif args.command=='check-gpu-tests':print(json.dumps(native.check_gpu_tests(args.junit,native.read(args.protocol)),indent=2))
        else:
            path,receipt=campaign_receipt(native,args.run)
            if receipt is not None:
                children={phase:native.load(native.workflow_path(run),verify=False) for phase,run in receipt['run_ids'].items() if native.workflow_path(run).exists()}
                if args.command=='status':
                    statuses={phase:native.status(w) for phase,w in children.items()}
                    missing={phase:run for phase,run in receipt['run_ids'].items() if phase not in children}
                    value=dict(run_id=receipt['run_id'],status=receipt['status'],terminal=receipt['status'] in ('COMPLETED','FAILED','INTERRUPTED') and all(s['terminal'] for s in statuses.values()),
                        completed=receipt['status']=='COMPLETED' and not missing and all(s['completed'] for s in statuses.values()),decision=receipt.get('decision'),
                        absent_declared_children=missing,
                        phases=statuses,not_submitted=receipt.get('not_submitted',{}),receipt_path=str(path))
                    print(json.dumps(value,indent=2))
                elif args.command=='validate':
                    if len(children)!=len(receipt['run_ids']):raise ValueError('Campaign has missing declared child artifacts')
                    print(json.dumps({p:native.validate_results(w,historical=True) for p,w in children.items()},indent=2))
                elif args.command=='decision':
                    gates={}
                    for phase,saved in receipt['gates'].items():
                        if phase not in children:raise ValueError('Campaign decision has a missing declared child')
                        gates[phase]=native.decision(children[phase],historical=True)
                        if gates[phase]!=saved:raise ValueError('Campaign receipt decision differs from its exact sealed child')
                    print(json.dumps(dict(decision=receipt.get('decision','UNRESOLVED'),gates=gates,not_submitted=receipt.get('not_submitted',{})),indent=2))
                elif args.command=='reconcile':
                    if not coordinator_dead(receipt['coordinator']):raise ValueError('Original coordinator is alive; do not reconcile it')
                    proofs={p:native.terminal_jobs(w) for p,w in children.items()}
                    from scripts.tangent_response_archive import TangentArchive
                    outer=TangentArchive(native.root).outer(receipt,[])
                    value=dict(coordinator_dead=True,all_submitted_jobs_terminal=True,children=list(children),native_proof_hash=digest(proofs))
                    value['raven_coordinator']=outer
                    if not args.plan:
                        receipt.setdefault('reconciliation',[]).append({**value,'checked_at':cw.now(),'previous_status':receipt['status']})
                        receipt['status']='INTERRUPTED';write(path,receipt)
                        write(native.root/'runs'/receipt['run_id']/'campaign-receipt.json',receipt)
                    print(json.dumps(value,indent=2))
                elif args.command=='paths':print(str(path)+'\n'+'\n'.join(str(Path(w['run_dir'])/'raven') for w in children.values()))
                elif args.command=='recover':
                    if receipt['status'] not in ('FAILED','INTERRUPTED') or not coordinator_dead(receipt['coordinator']):raise ValueError('Campaign recovery needs a terminal stopped coordinator')
                    from scripts.tangent_response_archive import TangentArchive
                    TangentArchive(native.root).outer(receipt,[])
                    for w in children.values():native.terminal_jobs(w)
                    failed=[(phase,w) for phase,w in children.items() if not native.status(w)['completed']]
                    if len(failed)!=1:raise ValueError('Campaign recovery requires exactly one failed child; inspect exact child identities')
                    phase,w=failed[0];native.recover(w['run_id'],args.run_id,plan=args.plan)
                    print('After the recovery verifies, explicitly continue '+receipt['run_id']+' with --replacement '+phase+'='+args.run_id)
                elif args.command=='compact':
                    return subprocess.call([str(native.root/'.venv/bin/python'),str(native.root/'scripts/compact_review.py'),'pack',receipt['run_id'],'--mode','full' if args.mode=='replay' else 'review'],cwd=native.root)
            else:
                w=native.load(native.workflow_path(args.run),verify=False)
                if args.command=='status':print(json.dumps(native.status(w),indent=2))
                elif args.command=='validate':print(json.dumps(native.validate_results(w,historical=True),indent=2))
                elif args.command=='decision':print(json.dumps(native.decision(w,historical=True),indent=2))
                elif args.command=='recover':native.recover(args.run,args.run_id,plan=args.plan)
                elif args.command=='paths':print(str(Path(w['run_dir'])/'raven'))
                elif args.command=='compact':return subprocess.call([str(native.root/'.venv/bin/python'),str(native.root/'scripts/compact_review.py'),'pack',w['run_id'],'--mode','full' if args.mode=='replay' else 'review'],cwd=native.root)
                else:raise ValueError('Reconcile requires an exact campaign identity')
    except rw.WorkerFailure as error:
        print('Tangent response: '+str(error),file=sys.stderr);return error.exit_code
    except (OSError,ValueError,KeyError,TypeError,ImportError) as error:
        print('Tangent response: '+str(error),file=sys.stderr);return 2
    return 0


if __name__=='__main__':raise SystemExit(main())
