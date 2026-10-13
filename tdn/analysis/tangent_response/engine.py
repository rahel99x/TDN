"""Sealed, bounded units for the staged tangent-response research program."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from pathlib import Path
import time
from types import SimpleNamespace
import torch

from tdn.runtime.metadata import software_metadata, write_json
from tdn.runtime.precision import reference_precision
from .contracts import clean, digest
from .protocol import ancestors, validate_protocol, training_workload
from . import data, gates

SCHEMA='tdn.tangent-response-science/v1'
ROOT=Path(__file__).resolve().parents[3]


def file_hash(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def inside(path,root=ROOT):
    path=Path(path)
    if path.is_symlink() or not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('Tangent artifacts must remain under the project without symlinks')
    for part in [path,*path.parents]:
        if part==Path(root):break
        if part.is_symlink():raise ValueError('Tangent paths cannot traverse symlinks')
    return path


class Budget:
    def __init__(self,seconds):self.start=time.monotonic();self.seconds=float(seconds)
    def check(self):
        if time.monotonic()-self.start>=self.seconds:raise TimeoutError('Declared science unit allowance exhausted')


def save_private(path,value):
    """Own-run tensor and primitive payload, loaded only with weights_only=True."""
    path=Path(path)
    if path.exists() or path.with_suffix(path.suffix+'.partial').exists():raise ValueError('Preserve existing private artifact')
    def convert(obj):
        if torch.is_tensor(obj):return obj.detach().cpu()
        if isinstance(obj,dict):return {str(k):convert(v) for k,v in obj.items()}
        if isinstance(obj,(list,tuple)):return [convert(v) for v in obj]
        if obj is None or isinstance(obj,(str,bool,int,float)):return obj
        raise TypeError('Private payloads accept tensors and primitives only: '+type(obj).__name__)
    tmp=path.with_suffix(path.suffix+'.partial');torch.save(convert(value),tmp);tmp.replace(path)


def load_private(path,device='cpu'):
    return torch.load(inside(path),weights_only=True,map_location=device)


def verify_science(path,p=None,*,sourcehash=None,historical=False):
    if type(historical) is not bool:raise ValueError('Historical verification requires an explicit boolean')
    if historical and (not isinstance(sourcehash,str) or not re.fullmatch('[0-9a-f]{64}',sourcehash)):
        raise ValueError('Historical science verification requires the explicit sealed source hash')
    path=inside(path);manifest=read(path/'science_manifest.json')
    if p is None:p=read(path/'protocol.json')
    validate_protocol(p,require_binding=False,canonical=not historical)
    marker=path/'COMPLETED'
    if (manifest.get('schema')!=SCHEMA or not marker.is_file() or marker.stat().st_size>128
            or marker.read_text().strip()!=file_hash(path/'science_manifest.json')
            or manifest.get('protocol_sha256')!=digest(p) or manifest.get('profile')!=p['profile']
            or manifest.get('stage') not in p['units'] or manifest.get('kind')!=p['units'][manifest['stage']]['kind']
            or manifest.get('unit_sha256')!=digest(p['units'][manifest['stage']])):
        raise ValueError('Tangent science seal or registered unit differs')
    if sourcehash is not None and manifest.get('source_tree_sha256')!=sourcehash:raise ValueError('Executable source identity changed')
    artifacts=manifest.get('artifacts',{})
    if not {'summary.json','protocol.json','execution.json'}<=set(artifacts):raise ValueError('Science seal lacks required artifacts')
    for name,checksum in artifacts.items():
        rel=Path(name)
        if rel.is_absolute() or '..' in rel.parts or not rel.parts:raise ValueError('Artifact path leaves its stage')
        artifact=inside(path/rel,path)
        if not artifact.is_file() or file_hash(artifact)!=checksum:raise ValueError('Sealed science artifact changed: '+name)
    if read(path/'protocol.json')!=p:raise ValueError('Exact sealed protocol differs')
    execution=read(path/'execution.json')
    if any(execution.get(key)!=manifest.get(key) for key in ('stage','profile','source_tree_sha256','protocol_sha256')):
        raise ValueError('Execution identity differs from science seal')
    return manifest


def verify_prerequisites(ctx):
    p=ctx.protocol;required=set(ancestors(p,ctx.stage))
    observer=ctx.unit['kind'] in ('decision','report')
    internal={key for key in ctx.prereqs if '/' not in key}
    if internal-required:raise ValueError('Unregistered internal prerequisite')
    if required-internal and not observer:raise ValueError('Missing declared ancestor: '+','.join(sorted(required-internal)))
    ctx.evidence_errors={};ctx.validated_prereqs={}
    for key,path in ctx.prereqs.items():
        try:
            if '/' not in key:
                verify_science(path,p,sourcehash=ctx.sourcehash)
                if read(Path(path)/'science_manifest.json')['stage']!=key:raise ValueError('Ancestor unit identity differs')
            else:
                phase,stage=key.split('/',1);binding=p['prerequisites'].get(phase)
                if not binding or binding['stage_paths'].get(stage)!=str(Path(path)):
                    raise ValueError('External evidence must match its exact bound stage path')
                parent_p=read(Path(path)/'protocol.json');verify_science(path,parent_p,sourcehash=ctx.sourcehash)
                if digest(parent_p)!=binding['protocol_sha256'] or file_hash(binding['workflow_path'])!=binding['workflow_sha256']:
                    raise ValueError('External workflow/protocol binding changed')
                seal=Path(path)/'workflow-seal.json'
                if file_hash(seal)!=binding['seals'].get(stage):raise ValueError('Native ancestor worker seal changed')
            ctx.validated_prereqs[key]=path
        except (OSError,ValueError,KeyError,TypeError) as exc:
            if not observer:raise
            ctx.evidence_errors[key]=str(exc)
    for key in required-internal:ctx.evidence_errors[key]='Missing registered prerequisite'
    if p['execution_scope']=='NATIVE_RESEARCH':
        required_gates={'compression':['A'],'fit_prepare':['B'],'calibration':['B','C0'],'learning':['B','C0'],'utility':['C']}.get(p['phase'],[])
        decisions=[b.get('decision',{}) for b in p['prerequisites'].values()]
        for gate in required_gates:
            matching=[d for d in decisions if d.get('gate')==gate]
            if not matching:raise ValueError('Missing literal prerequisite gate '+gate)
            gates.require_promotion(matching[0],gate)
        if p['phase']=='learning':
            calibration=p.get('calibration',{})
            gates.require_promotion(calibration.get('binding',{}).get('decision',{}),'CALIBRATION')
            if calibration.get('source_tree_sha256')!=ctx.sourcehash:raise ValueError('Calibration source changed')
        if p['profile']=='development':
            if not any(b.get('profile')!='development' and b.get('decision',{}).get('gate')=='C' for b in p['prerequisites'].values()):
                raise ValueError('Development requires eligible literal pilot learning evidence')


def _paths(ctx,kind):
    result=[]
    for key,path in ctx.validated_prereqs.items():
        mp=read(Path(path)/'science_manifest.json')
        if mp['kind']==kind:result.append((key,Path(path)))
    return result


def _samples(ctx,kind,cohort=None):
    result=[];seen=set()
    for _,path in _paths(ctx,kind):
        if not (path/'samples.json').is_file():continue
        for sample in data.load_samples(path,device=ctx.device):
            record=sample['record']
            if cohort is not None and record.get('cohort')!=cohort:continue
            if record['query_id'] in seen:continue
            seen.add(record['query_id']);result.append(sample)
    return result


def _selected(ctx):
    screens=_paths(ctx,'fit_screen')
    if len(screens)!=1:raise ValueError('Exactly one literal screened fit bank is required')
    return data.load_samples(screens[0][1],device=ctx.device)


def mechanism_config(p):
    m=p['mechanism']
    return {**p['reference'],**m,'response_initial_steps':m['initial_steps'],'response_max_steps':m['max_steps'],
            'response_tolerance':m['tolerance']*p['reference']['response_budget_fraction'],
            'path_tolerance':m['tolerance']*p['reference']['path_budget_fraction'],
            'total_uncertainty_budget':p['reference']['tolerance'],
            'reaction_substeps':p['anchor']['reaction_substeps']}


def _attach_oracles(sample,result):
    row=result['rows'][0];private=result['private_tensors'][0]
    sample['oracle_metadata']={'accepted':row['primary_status']=='RESOLVED','status':row['primary_status'],
        'uncertainty_rms':row.get('controls',{}).get('U',{}).get('uncertainty_rms'),
        'uncertainty_maximum':row.get('controls',{}).get('U',{}).get('uncertainty_maximum'),
        'matched_mean_uncertainty_rms':row.get('controls',{}).get('G',{}).get('uncertainty_rms')}
    for name,field in [('U','oracle_endpoint64'),('G','mean_endpoint64')]:
        if name in private['endpoints']:sample[field]=private['endpoints'][name]
    if 'U' in private['corrections']:sample['oracle_correction64']=private['corrections']['U']
    if 'G' in private['corrections']:sample['mean_correction64']=private['corrections']['G']
    return sample


def _training_config(ctx,arm,updates,trial=None):
    t=ctx.protocol['training'];model={'family':arm,**{k:t[k] for k in ('rank','width','depth','source_nodes','mean_nodes','assembly','temporal_channels','t_ref','modes','ndim')},
        'anchor':ctx.protocol['anchor']['name'],'reaction_substeps':ctx.protocol['anchor']['reaction_substeps']}
    return {**t,'family':arm,'id':trial['id'] if trial else 'calibration-'+arm,
        'seed':trial['seed'] if trial else t['seeds'][0], 'updates':updates,'model_config':model,
        'source_tree_sha256':ctx.sourcehash,'total_seconds':trial['total_seconds'] if trial else ctx.unit['seconds']-60,
        'lr_schedule':{'kind':'cosine','minimum_ratio':t['minimum_lr_ratio'],'warmup_updates':min(t['warmup_updates'],updates-1)}}


def _trials(ctx):
    latest={}
    for key,path in _paths(ctx,'training'):
        if (path/'trial.json').is_file():
            result=read(path/'trial.json');trial=result['trial_id']
            if result.get('family') not in ctx.protocol['training']['arms'] or result.get('seed') not in ctx.protocol['training']['seeds']:continue
            index=read(path/'protocol.json')['units'][key.split('/')[-1]].get('chunk_index',0)
            if trial not in latest or index>latest[trial][0]:latest[trial]=(index,path,result)
    return [{'path':str(path),'result':value} for _,path,value in latest.values()]


def _compression_factors(ctx):
    paths=_paths(ctx,'compression')
    if not paths:return None,[]
    path=paths[0][1];return read(path/'compression_results.json'),load_private(path/'factors.pt',ctx.device)


def _decision(ctx,gate):
    p=ctx.protocol
    mechanism_rows=[row for _,path in _paths(ctx,'mechanism') for row in read(path/'mechanism_results.json')['rows']]
    if gate=='A':return gates.mechanism_decision(mechanism_rows,p)
    if gate=='B':return gates.compression_decision(read(_paths(ctx,'compression')[0][1]/'compression_results.json'),p)
    if gate=='C0':return gates.data_decision(read(_paths(ctx,'fit_screen')[0][1]/'fit_screen.json'),p)
    if gate=='CALIBRATION':
        rows={read(path/'calibration.json')['arm']:read(path/'calibration.json') for _,path in _paths(ctx,'calibration')}
        native=bool(rows) and all(r.get('native') is True and r.get('device')=='cuda' for r in rows.values())
        return gates.calibration_decision({'native':native,'device':'cuda' if native else 'cpu','source_tree_sha256':ctx.sourcehash,'measurements':rows},p)
    if gate=='C':
        return gates.learning_decision([r['result'] for r in _trials(ctx)],read(_paths(ctx,'evaluation')[0][1]/'evaluation_results.json'),p)
    if gate=='D':return gates.utility_decision(read(_paths(ctx,'performance')[0][1]/'performance_results.json'),p)
    results={'A':_decision(ctx,'A'),'B':_decision(ctx,'B'),'C0':_decision(ctx,'C0'),
             'CALIBRATION':_decision(ctx,'CALIBRATION'),'C':_decision(ctx,'C'),'D':_decision(ctx,'D')}
    return {'schema':'tdn.tangent-response-decision/v1','gate':'ALL','decision':'IMPLEMENTATION_ONLY','status':'IMPLEMENTATION_ONLY',
            'stage':'all','eligible':False,'eligible_next':False,'profile':p['profile'],'underlying_decisions':results,
            'reason':'CPU software rehearsal cannot promote native numerical work'}


def dispatch(ctx):
    p,u=ctx.protocol,ctx.unit;kind=u['kind'];path=ctx.path
    if kind=='audit':
        result={'status':'VERIFIED_CPU_IMPLEMENTATION' if ctx.device=='cpu' else 'ALLOCATED_CUDA_BODY',
                'software':software_metadata(),'target':p['target'],'matrix_free':True,'native_readiness':False}
        write_json(path/'audit.json',clean(result));return result
    if kind in ('prepare','fit_prepare'):
        cohort='mechanism' if kind=='prepare' else 'fit';bank=data.parent_bank(p,cohort)
        write_json(path/'data_manifest.json',bank);write_json(path/'query_registry.json',{'queries':p['queries' if cohort=='mechanism' else 'fit_queries']})
        return {'parents':len(bank['parents']),'query_count':len(p['queries' if cohort=='mechanism' else 'fit_queries'])}
    if kind=='references':
        prepared=_paths(ctx,'prepare' if u['cohort']=='mechanism' else 'fit_prepare')
        bank=read(prepared[0][1]/'data_manifest.json');menu=p['queries' if u['cohort']=='mechanism' else 'fit_queries']
        samples=[data.add_reference(data.make_query(p,bank,q,device=ctx.device),p,budget=ctx.budget) for q in menu if q['query_id'] in u['query_ids']]
        data.save_samples(path,samples);write_json(path/'reference_summary.json',{'rows':[s['record'] for s in samples]})
        return {'requested_queries':len(u['query_ids']),'accepted_queries':sum(s['reference']['accepted'] for s in samples)}
    if kind in ('mechanism','fit_labels'):
        from .oracle import run_mechanism
        samples=[s for s in _samples(ctx,'references',u['cohort']) if s['record']['query_id'] in u['query_ids']]
        result=run_mechanism(samples,mechanism_config(p),budget=ctx.budget)
        write_json(path/'mechanism_results.json',clean({k:v for k,v in result.items() if k!='private_tensors'}))
        samples=[_attach_oracles(s,result) for s in samples]
        if kind=='fit_labels':
            from .compression import collect_experiment_actions
            action_result=collect_experiment_actions(samples,{**p['compression'],'reaction_substeps':p['anchor']['reaction_substeps']},budget=ctx.budget)
            write_json(path/'action_label_receipt.json',clean(action_result))
            for sample in samples:
                sample['action_labels']=[{'probe_id':r['probe_id'],'source':r['source'],'target_action':r['tangent'],
                    's':r['source_time'],'h':r['horizon'],'accepted':r['status']=='RESOLVED','uncertainty':r.get('uncertainty'),
                    'role':r['probe_role']} for r in action_result['_actions'] if r.get('query_id')==sample['record']['query_id'] and r['probe_role']=='fit' and r.get('source') is not None and r.get('tangent') is not None]
        data.save_samples(path,samples)
        return {'requested_queries':len(samples),'resolved_primary_queries':sum(r['primary_status']=='RESOLVED' for r in result['rows'])}
    if kind=='actions':
        from .compression import collect_experiment_actions
        samples=[s for s in _samples(ctx,'mechanism','mechanism') if s['record']['query_id'] in u['query_ids']]
        result=collect_experiment_actions(samples,{**p['compression'],'reaction_substeps':p['anchor']['reaction_substeps']},budget=ctx.budget)
        save_private(path/'actions.pt',result['_actions']);write_json(path/'action_receipt.json',clean(result));data.save_samples(path,samples)
        return {'requested_actions':len(result['_actions']),'resolved_actions':sum(r['status']=='RESOLVED' for r in result['_actions'])}
    if kind=='compression':
        from .compression import run_experiment
        action_paths=_paths(ctx,'actions');records=[];checks=[];receipts=[]
        for _,parent in action_paths:
            records+=load_private(parent/'actions.pt',ctx.device);receipt=read(parent/'action_receipt.json');checks+=receipt['endpoint_quadrature_checks'];receipts.append({**receipt,'source_artifact_sha256':file_hash(parent/'actions.pt')})
        samples=_samples(ctx,'actions','mechanism')
        result=run_experiment(samples,{**p['compression'],'reaction_substeps':p['anchor']['reaction_substeps']},
            precomputed_actions=records,precomputed_endpoint_checks=checks,acquisition_receipts=receipts,budget=ctx.budget)
        sample_index={s['record']['query_id']:s for s in samples}
        for row in result['endpoint_rows']:
            s=sample_index[row['query_id']];unc=s.get('oracle_metadata',{}).get('uncertainty_rms')
            row['teacher_path_uncertainty']=unc
            row['combined_uncertainty']=row.get('uncertainty',0)+unc if unc is not None and row.get('uncertainty') is not None else None
        factors=[{'identity':list(identity),'context':list(identity[0]),'method':identity[1],
                  'parent_id':identity[2] if len(identity)>2 else None,'factors':factor.to_payload()} for identity,factor in result['_factors'].items()]
        save_private(path/'factors.pt',factors);write_json(path/'compression_results.json',clean(result))
        return {'actions':len(result['measurement_rows']),'fits':len(result['method_rows']),'requested_endpoints':len(result['endpoint_rows'])}
    if kind=='fit_screen':
        samples,screen=data.screening(_samples(ctx,'fit_labels','fit'),p);data.save_samples(path,samples)
        write_json(path/'fit_screen.json',screen);return screen
    if kind in ('calibration','training'):
        from .training import fit,_data_hash
        samples=[s for s in _selected(ctx) if s['record']['split'] in ('train','validation')]
        trial=p['trials'][u['trial_id']] if kind=='training' else None
        config=_training_config(ctx,u['arm'],u['updates'],trial)
        if kind=='training' and p['execution_scope']=='NATIVE_RESEARCH':
            measured=p['calibration']['data']['measurements'][u['arm']]
            if measured.get('data_sha256')!=_data_hash(samples):
                raise ValueError('Actual screened labels/query layout changed since native calibration')
        resume=ctx.resume_from or (ctx.prereqs.get(u.get('previous_chunk')) if u.get('previous_chunk') else None)
        result=None
        if u.get('previous_chunk') and resume is not None:
            from .continuation import reuse_completed_trial
            result=reuse_completed_trial(resume,path,config,samples,device=ctx.device,sourcehash=ctx.sourcehash)
        if result is None:
            result=fit(samples,config,device=ctx.device,output_path=path,resume_from=resume,chunk_seconds=max(1,u['seconds']-45),budget=ctx.budget)
        result.setdefault('seed',config['seed']);result.setdefault('trial_id',config['id'])
        result['requested_train_query_ids']=[s['record']['query_id'] for s in samples if s['record']['split']=='train']
        write_json(path/'trial.json',clean(result));write_json(path/'training_results.json',{'trials':[clean(result)]})
        if kind=='calibration':
            completed=result['updates'];seconds=result.get('cost',{}).get('total_seconds')
            calibration={'arm':u['arm'],'status':result['status'],'native':ctx.device=='cuda','device':ctx.device,
                'execution_mode':os.environ.get('TDN_EXECUTION_MODE','local_cpu'),'completed_updates':completed,
                'workload_sha256':u['workload_sha256'],'seconds_per_update':seconds/completed if completed and seconds else None,
                'source_tree_sha256':ctx.sourcehash,'requested_updates':u['updates'],'fit_bank_query_ids':[s['record']['query_id'] for s in samples],
                'data_sha256':_data_hash(samples),'query_plan_sha256':result.get('query_plan_sha256'),
                'rate_scope':'Complete fit including preparation, forward/backward, validation, journals and declared conditional controls per committed update'}
            write_json(path/'calibration.json',calibration)
        return {'training_status':result['status'],'completed_updates':result['updates'],'requested_updates':result['requested_updates']}
    if kind in ('evaluation','performance'):
        from .training import load_model
        from .inference import make_methods
        from .stages_evaluation import run_evaluation,run_utility
        samples=_selected(ctx);trials=_trials(ctx);models={};loaded=[]
        for trial in sorted(trials,key=lambda t:(t['result']['seed'],t['result']['family'])):
            selected=Path(trial['path'])/'selected.pt'
            if selected.is_file():
                model,metadata=load_model(selected,ctx.device)
                models.setdefault(trial['result']['family'],model)
                loaded.append((trial,model,metadata,file_hash(selected)))
        compression,factors=_compression_factors(ctx)
        methods=make_methods(samples,p,compression,models,private_factors=factors,budget=ctx.budget)
        for trial,model,metadata,checksum in loaded:
            family=trial['result']['family'];seed=trial['result']['seed']
            named=family if models.get(family) is model else family+'-seed-'+str(seed)
            callback=methods[family] if named==family else make_methods(samples,p,compression,{family:model},private_factors=factors,budget=ctx.budget)[family]
            methods[named]={'call':callback,'family':family,'seed':seed,'checkpoint_sha256':checksum,
                            'model':model,'deployable':True,'schedule':'validation-selected'}
        return (run_evaluation if kind=='evaluation' else run_utility)(ctx,samples,methods,trials)
    if kind=='decision':
        try:result=_decision(ctx,u['gate'])
        except (OSError,IndexError,KeyError,ValueError,TypeError) as exc:
            result=gates._decision(u['gate'],p['phase'],'UNRESOLVED',p,evidence={'missing_or_invalid_evidence':ctx.evidence_errors},reason=str(exc))
        if ctx.evidence_errors:
            result=gates._decision(u['gate'],p['phase'],'UNRESOLVED',p,evidence={'observed_decision':result,'missing_or_invalid_evidence':ctx.evidence_errors},reason='Requested evidence incomplete or invalid')
        write_json(path/'decision.json',clean(result));return {'scientific_outcome':result['decision'],'gate':u['gate'],'eligible':result['eligible']}
    if kind=='report':
        from .report import run
        return run(ctx)
    raise ValueError('Unimplemented registered unit kind '+kind)


def execute(stage,p,path,*,device='cpu',prereqs=None,resume_from=None):
    reference_precision();validate_protocol(p)
    if stage not in p['units']:raise ValueError('Unknown registered stage')
    path=inside(path);unit=p['units'][stage]
    if device!=unit['device']:raise ValueError('Requested device differs from frozen unit')
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA required; CPU fallback is forbidden')
    if p['execution_scope']=='NATIVE_RESEARCH':
        if os.environ.get('TDN_EXECUTION_MODE')!='desktop-slurm' or not os.environ.get('SLURM_JOB_ID') or not os.environ.get('TDN_TANGENT_WORKFLOW'):
            raise RuntimeError('Native science requires the verified desktop Slurm allocated worker')
    if path.exists() and (path/'execution.json').exists():raise ValueError('Run stage exists; preserve it and use explicit recovery')
    path.mkdir(parents=True,exist_ok=True);started=time.perf_counter();sourcehash=software_metadata()['source_tree_sha256']
    ctx=SimpleNamespace(stage=stage,protocol=p,path=path,unit=unit,device=device,prereqs=prereqs or {},
        prerequisites=prereqs or {},sourcehash=sourcehash,budget=Budget(unit['seconds']),resume_from=resume_from)
    if resume_from is not None:
        ctx.resume_from=inside(resume_from)
        journal=ctx.resume_from/'journal.pt' if ctx.resume_from.is_dir() else ctx.resume_from
        journal=inside(journal)
        saved=load_private(journal)
        if journal.parent==path or saved.get('update',0)>=unit.get('updates',0):
            raise ValueError('Resume requires an exact unfinished journal from a preserved prior attempt')
    verify_prerequisites(ctx)
    from . import report,telemetry
    if p['execution_scope']=='NATIVE_RESEARCH':
        from scripts.tangent_response_workflow import NativeWorkflow
        native=NativeWorkflow(ROOT);workflow=native.load(os.environ['TDN_TANGENT_WORKFLOW'])
        if native.protocol(workflow)!=p or workflow['source_tree_sha256']!=sourcehash:
            raise ValueError('Allocated workflow differs from the exact science contract')
        native.verify_allocation(workflow,stage)
        if device=='cuda':
            proof=native.check_gpu_tests(native.junit(workflow,stage),p)
            ctx.native_readiness={'verified':True,'source_tree_sha256':sourcehash,'job_id':os.environ['SLURM_JOB_ID'],
                'mandatory_cuda_cases':proof['test_cases'],'junit_sha256':file_hash(native.junit(workflow,stage))}
        native_attempt=read(report.attempt_path(ctx)/'run.json')
        if (native_attempt.get('state')!='RUNNING' or native_attempt.get('job_id')!=os.environ['SLURM_JOB_ID']
                or native_attempt.get('metadata',{}).get('native_worker_managed') is not True):
            raise ValueError('Native science must reuse its allocated worker-owned Raven attempt')
    execution={'stage':stage,'profile':p['profile'],'device':device,'execution_mode':os.environ.get('TDN_EXECUTION_MODE','local_cpu'),
        'source_tree_sha256':sourcehash,'protocol_sha256':digest(p),'software':software_metadata(),
        'requested_prerequisites':{k:str(v) for k,v in ctx.prereqs.items()},'resume_from':str(ctx.resume_from) if ctx.resume_from else None,
        'native_readiness':getattr(ctx,'native_readiness',None)}
    write_json(path/'protocol.json',p);write_json(path/'execution.json',execution)
    report.start_attempt(ctx,job_id=os.environ.get('SLURM_JOB_ID'),sourcehash=sourcehash,worker_managed=False)
    observer=None
    if p['execution_scope']=='CPU_IMPLEMENTATION_SMOKE':observer=telemetry.start_recording(report.attempt_path(ctx),options=p['worker_telemetry'])
    try:
        details=dispatch(ctx)
        verify_prerequisites(ctx)
        if software_metadata()['source_tree_sha256']!=sourcehash:raise ValueError('Source changed while scientific unit ran; preserve incomplete attempt')
        summary={'status':'COMPLETED','stage':stage,'kind':unit['kind'],'profile':p['profile'],'elapsed_seconds':time.perf_counter()-started,
            'execution_scope':p['execution_scope'],'scientific_outcome':details.get('scientific_outcome'), 'details':clean(details)}
        write_json(path/'summary.json',summary)
        artifacts={f.relative_to(path).as_posix():file_hash(f) for f in sorted(path.rglob('*')) if f.is_file() and f.name not in ('science_manifest.json','COMPLETED')}
        manifest={'schema':SCHEMA,'stage':stage,'kind':unit['kind'],'profile':p['profile'],'protocol_sha256':digest(p),
            'unit_sha256':digest(unit),'source_tree_sha256':sourcehash,'artifacts':artifacts,
            'prerequisite_manifest_sha256':{key:file_hash(Path(value)/'science_manifest.json') for key,value in ctx.validated_prereqs.items()},
            'unresolved_prerequisites':ctx.evidence_errors}
        write_json(path/'science_manifest.json',manifest);(path/'COMPLETED').write_text(file_hash(path/'science_manifest.json')+'\n')
        verify_science(path,p,sourcehash=sourcehash);return summary
    except Exception as exc:
        summary={'status':'FAILED','stage':stage,'kind':unit['kind'],'profile':p['profile'],'elapsed_seconds':time.perf_counter()-started,
                 'error':{'type':type(exc).__name__,'message':str(exc)},'scientific_outcome':'UNRESOLVED'}
        write_json(path/'summary.json',summary);raise
    finally:
        if observer is not None:telemetry.stop_recording(observer,state=summary['status'])
        report.finish_attempt(ctx,summary)
