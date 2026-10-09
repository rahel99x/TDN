#!/usr/bin/env python3
"""Run one provenance-bound unit of the TDN three-path research portfolio."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
SCHEMA='tdn.portfolio/v1'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def file_digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): result.update(block)
    return result.hexdigest()


def ancestors(protocol,stage):
    if stage not in protocol['units']: raise ValueError('Unknown portfolio unit')
    found=set()
    def visit(unit):
        for parent in protocol['units'][unit]['dependencies']:
            if parent not in found: found.add(parent);visit(parent)
    visit(stage)
    return tuple(name for name in protocol['units'] if name in found)


def parse_prerequisites(values,protocol,stage):
    result={}
    for value in values:
        name,separator,path=value.partition('=')
        if not separator or not path or name in result or name not in protocol['units']:
            raise ValueError('Prerequisites require unique declared UNIT=PATH entries')
        result[name]=Path(path)
    if set(result)!=set(ancestors(protocol,stage)):
        raise ValueError('Every ancestor unit is required exactly once')
    return result


def compatible_software(value):
    keys=('python','executable','venv','torch','numpy','scipy','torch_cuda_runtime','git_commit','source_tree_sha256')
    return {key:value.get(key) for key in keys}


def verify_execution(path,protocol,*,source_tree_sha256=None):
    from tdn.analysis.portfolio.engine import verify_science
    path=Path(path)
    science=verify_science(protocol,path,source_tree_sha256=source_tree_sha256)
    seal=json.loads((path/'workflow-seal.json').read_text())
    names=('execution.json','protocol.json','stage.json','science_manifest.json')
    if seal.get('schema')!=SCHEMA or seal.get('protocol_sha256')!=digest(protocol) or set(seal.get('files',{}))!=set(names):
        raise ValueError('Portfolio execution seal has changed scope/inventory')
    for name in names:
        file=path/name
        if file.is_symlink() or not file.is_file() or file_digest(file)!=seal['files'][name]:
            raise ValueError('Portfolio sealed execution evidence changed: '+name)
    execution=json.loads((path/'execution.json').read_text());record=json.loads((path/'stage.json').read_text())
    if record.get('status')!='COMPLETED' or execution.get('protocol_sha256')!=digest(protocol):
        raise ValueError('Portfolio unit did not complete under its frozen protocol')
    if execution.get('stage')!=science['stage'] or execution.get('software',{}).get('source_tree_sha256')!=science['source_tree_sha256']:
        raise ValueError('Portfolio execution differs from scientific identity')
    if source_tree_sha256 is not None and execution['software']['source_tree_sha256']!=source_tree_sha256:
        raise ValueError('Portfolio execution source differs')
    return {'science':science,'execution':execution,'seal':seal}


def seal_execution(path,protocol):
    from tdn.runtime.metadata import write_json
    path=Path(path)
    write_json(path/'workflow-seal.json',{'schema':SCHEMA,'protocol_sha256':digest(protocol),
        'files':{name:file_digest(path/name) for name in ('execution.json','protocol.json','stage.json','science_manifest.json')}})


def resume_fingerprint(path):
    """Content identity of every retained interrupted artifact; symlinks are refused."""
    from tdn.runtime.storage import contained_path
    path=contained_path(path);inventory={}
    for file in sorted(path.rglob('*')):
        if file.is_symlink(): raise ValueError('Resume refuses symlink artifacts')
        if file.is_file(): inventory[file.relative_to(path).as_posix()]=file_digest(file)
    return digest(inventory)


def check_resume(path,*,stage,protocol,software,mode,device,lineage,site=None):
    path=Path(path)
    if any((path/name).exists() for name in ('workflow-seal.json','COMPLETED','science_manifest.json')):
        raise ValueError('Completed units cannot be resumed or overwritten')
    record=json.loads((path/'stage.json').read_text());execution=json.loads((path/'execution.json').read_text())
    if record.get('status') not in ('FAILED','INTERRUPTED','INCOMPLETE'):
        raise ValueError('Resume requires an explicitly interrupted or failed terminal attempt')
    if protocol['units'][stage]['kind'] not in ('train','confirm'):
        raise ValueError('Only journaled train/confirm units may resume')
    expected={'stage':stage,'profile':protocol['profile'],'protocol_sha256':digest(protocol),'execution_mode':mode,'device':device,'prerequisites':lineage}
    if any(execution.get(key)!=value for key,value in expected.items()) or compatible_software(execution['software'])!=compatible_software(software):
        raise ValueError('Resume scope, software, or prerequisite lineage differs')
    if site is not None and execution.get('slurm_profile_sha256')!=digest(site):
        raise ValueError('Resume belongs to another Slurm profile')
    return execution


def failed_stage_diagnostic(path, error):
    """Retain operational elapsed work without treating unsealed science as evidence."""
    path = Path(path)
    result = {"run_dir": str(path), "verification": "INVALID" if path.exists() else "MISSING",
              "error": str(error), "evidence_status": "UNVERIFIED_DIAGNOSTIC",
              "cost_scope": "Operational CLI elapsed time; includes failed work, is not scheduler accounting"}
    stage_file = path / "stage.json"
    try:
        if stage_file.is_symlink() or stage_file.stat().st_size > 2**20:
            raise ValueError("Unsafe or oversized operational record")
        record = json.loads(stage_file.read_text())
        elapsed = record.get("elapsed_seconds")
        if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool) and math.isfinite(elapsed) and elapsed >= 0:
            result["elapsed_seconds"] = elapsed
        for key in ("status", "slurm_job_id", "slurm_step_id"):
            if isinstance(record.get(key), str):
                result[key] = record[key]
        result["operational_source"] = {"path": str(stage_file), "sha256": file_digest(stage_file)}
    except (OSError, ValueError, TypeError):
        pass
    return result


def verify_native_binding(stage, profile, device):
    """A native stage must be the recorded worker, including its complete CUDA suite."""
    import importlib.util
    workflow_path = os.environ.get("TDN_PORTFOLIO_WORKFLOW")
    if not workflow_path or os.environ.get("TDN_PORTFOLIO_STAGE") != stage:
        raise ValueError("Native portfolio CLI requires its frozen workflow worker binding")
    spec = importlib.util.spec_from_file_location("portfolio_native_binding", ROOT / "scripts/portfolio_workflow.py")
    controller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)
    workflow = controller.load(Path(workflow_path))
    if workflow["profile"] != profile or controller.units(workflow)[stage]["device"] != device:
        raise ValueError("Native CLI scope differs from the frozen workflow")
    if os.environ.get("TDN_PORTFOLIO_PROTOCOL_SHA256") != workflow["protocol_sha256"]:
        raise ValueError("Native CLI protocol binding differs")
    controller.verify_allocation(workflow, stage)
    state = controller.cw.read_json(controller.pw.state_path(workflow, stage))
    if (state.get("status") != "RUNNING" or state.get("stage") != "experiment" or
            state.get("slurm_job_id") != os.environ.get("SLURM_JOB_ID") or
            state.get("slurm_step_id") != os.environ.get("SLURM_STEP_ID")):
        raise ValueError("Native CLI must execute inside its current recorded experiment task")
    if device == "cuda":
        controller.check_gpu_tests(controller.junit_path(workflow, stage))
    return workflow


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True);parser.add_argument('--profile',choices=('smoke','development','full'),default='full')
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--prerequisite-dir',action='append',default=[])
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu');parser.add_argument('--local-root',type=Path)
    parser.add_argument('--resume-from',type=Path)
    args=parser.parse_args(argv)
    run_dir=report=record=None;owned=False;exit_code=1;started=time.monotonic();previous_tower=os.environ.get('TDN_TOWER_DIR')
    try:
        from tdn.runtime.storage import CARC_ROOT,configure_storage,contained_path
        from tdn.runtime.preflight import execution_mode,verify_runtime
        if args.local_root is not None:
            if args.local_root.resolve()!=ROOT or ROOT==CARC_ROOT or any(os.environ.get(key) for key in ('SLURM_JOB_ID','SLURM_STEP_ID','TDN_EXECUTION_MODE')):
                raise ValueError('Local portfolio requires this CPU checkout without scheduler overrides')
            if args.device!='cpu' or args.profile=='full':
                raise ValueError('Local portfolio supports CPU smoke/development; full requires Fedora Slurm')
        mode=execution_mode()
        if mode not in ('local-cpu','desktop-slurm') or (mode=='local-cpu' and (args.local_root is None or args.profile=='full' or args.device!='cpu')):
            raise ValueError('Portfolio requires allocated Fedora or explicit local CPU development')
        from tdn.analysis.portfolio.protocol import build_protocol,validate_protocol
        protocol=build_protocol(args.profile);validate_protocol(protocol)
        if args.stage not in protocol['units']: raise ValueError('Unknown portfolio unit')
        if mode=='desktop-slurm' and args.device!=protocol['units'][args.stage]['device']:
            raise ValueError('Allocated unit must use its frozen device')
        configure_storage();verify_runtime(args.device,'portfolio-'+args.stage)
        if mode == 'desktop-slurm': verify_native_binding(args.stage,args.profile,args.device)
        from tdn.analysis.portfolio.engine import run_stage,verify_science
        from tdn.runtime.metadata import software_metadata,write_json
        from tdn.runtime.precision import reference_precision
        from tdn.runtime.signal_handling import StopRequest
        import torch
        torch.set_num_threads(1)
        try: torch.set_num_interop_threads(1)
        except RuntimeError: pass
        reference_precision();software=software_metadata();site=None
        if mode=='desktop-slurm':
            from tdn.runtime.desktop_slurm import load_profile
            site=load_profile()
        candidate=contained_path(args.run_dir)
        if candidate.exists(): raise ValueError('Preserve prior results; select a fresh stage directory')
        paths={key:contained_path(value) for key,value in parse_prerequisites(args.prerequisite_dir,protocol,args.stage).items()}
        lineage={};verified_paths={}
        for name in ancestors(protocol,args.stage):
            path=paths[name]
            if candidate==path or candidate.is_relative_to(path) or path.is_relative_to(candidate):
                raise ValueError('Stage and prerequisite directories must remain separate')
            try:
                checked=verify_execution(path,protocol,source_tree_sha256=software['source_tree_sha256']);execution=checked['execution']
                if execution.get('stage')!=name or execution.get('execution_mode')!=mode or compatible_software(execution['software'])!=compatible_software(software):
                    raise ValueError('Prerequisite source/software/execution mode differs')
                if site is not None and execution.get('slurm_profile_sha256')!=digest(site):
                    raise ValueError('Prerequisite belongs to a different Slurm profile')
                expected_prior={key:lineage[key] for key in ancestors(protocol,name)}
                if execution.get('prerequisites')!=expected_prior:
                    raise ValueError('Prerequisites belong to different frozen lineages')
                lineage[name]={'run_dir':str(path),'workflow_seal_sha256':file_digest(path/'workflow-seal.json')}
                verified_paths[name]=path
            except (ValueError,OSError,KeyError,TypeError) as error:
                if args.stage!='report': raise
                lineage[name]=failed_stage_diagnostic(path,error)
        origin=None;origin_hash=None
        if args.resume_from:
            origin=contained_path(args.resume_from)
            if candidate==origin or candidate.is_relative_to(origin) or origin.is_relative_to(candidate):
                raise ValueError('Resume target must be separate from the immutable origin')
            check_resume(origin,stage=args.stage,protocol=protocol,software=software,mode=mode,device=args.device,lineage=lineage,site=site)
            origin_hash=resume_fingerprint(origin)
            shutil.copytree(origin,candidate,symlinks=False)
            if resume_fingerprint(origin)!=origin_hash or resume_fingerprint(candidate)!=origin_hash:
                raise ValueError('Interrupted origin changed while copying')
            attempt=candidate/'prior_attempts'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            attempt.mkdir(parents=True)
            for name in ('execution.json','stage.json','summary.json','science_manifest.json'):
                if (candidate/name).exists(): shutil.copy2(candidate/name,attempt/name)
        else: candidate.mkdir(parents=True,exist_ok=False)
        run_dir=candidate
        execution={'stage':args.stage,'profile':args.profile,'device':args.device,'execution_mode':mode,
            'command':[sys.executable,str(Path(__file__).resolve()),*(sys.argv[1:] if argv is None else argv)],
            'software':software,'protocol_sha256':digest(protocol),'prerequisites':lineage,
            'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'slurm_step_id':os.environ.get('SLURM_STEP_ID')}
        if origin: execution['resume']={'path':str(origin),'sha256':origin_hash}
        if os.environ.get('TDN_PORTFOLIO_PROTOCOL_SHA256'): execution['workflow_protocol_sha256']=os.environ['TDN_PORTFOLIO_PROTOCOL_SHA256']
        if site is not None: execution['slurm_profile_sha256']=digest(site)
        write_json(run_dir/'execution.json',execution);write_json(run_dir/'protocol.json',protocol)
        record={**execution,'status':'RUNNING','actually_ran':False,'benchmark_suite':'portfolio'}
        write_json(run_dir/'stage.json',record)
        from tdn.reporting import attach_report,emit
        report,owned=attach_report(run_dir,name='TDN/portfolio/'+args.stage,script='scripts/portfolio.py',
            parameters={'device':args.device,'stage':args.stage,'profile':args.profile,'execution_mode':mode,
                        'protocol_sha256':digest(protocol),'source_tree_sha256':software['source_tree_sha256']})
        os.environ['TDN_TOWER_DIR']=str(report);record['actually_ran']=True;write_json(run_dir/'stage.json',record)
        emit({'stage_started':1},phase='portfolio/'+args.stage)
        with StopRequest() as stop:
            result=run_stage(protocol,args.stage,run_dir,prerequisites={name:path for name,path in verified_paths.items() if name in protocol["units"][args.stage]["dependencies"]},device=args.device,stop=stop,resume=origin is not None,
                **({'stage_failures':{name:details for name,details in lineage.items() if name not in verified_paths}} if args.stage=='report' else {}))
            if stop.requested: raise InterruptedError('Stop requested before sealing evidence')
        if result.get('status')!='COMPLETED':
            if result.get('status') in ('INCOMPLETE','INTERRUPTED'): raise InterruptedError('Unit stopped; partial evidence retained')
            raise RuntimeError('Unit failed; partial evidence retained')
        if software_metadata()['source_tree_sha256']!=software['source_tree_sha256'] or digest(json.loads((run_dir/'protocol.json').read_text()))!=digest(protocol):
            raise RuntimeError('Execution source or protocol changed during unit')
        if site is not None and digest(load_profile())!=digest(site): raise RuntimeError('Slurm profile changed during unit')
        for name,path in verified_paths.items():
            verify_execution(path,protocol,source_tree_sha256=software['source_tree_sha256'])
            if file_digest(path/'workflow-seal.json')!=lineage[name]['workflow_seal_sha256']: raise RuntimeError('Prerequisite changed during unit')
        if origin and resume_fingerprint(origin)!=origin_hash: raise RuntimeError('Interrupted origin changed during resume')
        verify_science(protocol,run_dir,source_tree_sha256=software['source_tree_sha256'])
        record.update(status='COMPLETED',elapsed_seconds=time.monotonic()-started);write_json(run_dir/'stage.json',record)
        seal_execution(run_dir,protocol);emit({'stage_completed':1},phase='portfolio/'+args.stage)
        print(json.dumps({'status':'COMPLETED','stage':args.stage,'summary':str(run_dir/'summary.json'),'scientific_outcome':result.get('scientific_outcome'),'tower':str(report)},indent=2));exit_code=0
    except (Exception,KeyboardInterrupt) as error:
        interrupted=isinstance(error,(InterruptedError,TimeoutError,KeyboardInterrupt));exit_code=130 if isinstance(error,KeyboardInterrupt) else 75 if interrupted else 1
        if record is not None:
            record.update(status='INTERRUPTED' if interrupted else 'FAILED',error=f'{type(error).__name__}: {error}',elapsed_seconds=time.monotonic()-started)
            write_json(run_dir/'stage.json',record)
        print(f'TDN portfolio: {type(error).__name__}: {error}',file=sys.stderr)
    finally:
        try:
            if owned:
                from tdn.cli import finalize_tower_report
                from tdn.portfolio_reporting import publish_outputs
                publish_outputs(report,[run_dir])
                if not finalize_tower_report(report,source_dirs=[run_dir],state=record['status'],started=started,exit_code=exit_code,
                    metadata={'device':args.device,'stage':args.stage,'benchmark_suite':'portfolio','actually_ran':record['actually_ran']}): exit_code=exit_code or 1
        except Exception as reporting_error:
            print(f'TDN portfolio reporting: {type(reporting_error).__name__}: {reporting_error}',file=sys.stderr)
            exit_code=exit_code or 1
        finally:
            if previous_tower is None: os.environ.pop('TDN_TOWER_DIR',None)
            else: os.environ['TDN_TOWER_DIR']=previous_tower
    return exit_code


from datetime import datetime,timezone
if __name__=='__main__':
    raise SystemExit(main())
