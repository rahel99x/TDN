#!/usr/bin/env python3
"""Execute every reduced diagnostic unit on CPU, retaining failures and reports."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--targets',nargs='+',choices=('fd_same_grid','galerkin_same_grid','galerkin_projected_fine'))
    args=parser.parse_args(argv)
    if any(os.environ.get(k) for k in ('SLURM_JOB_ID','SLURM_STEP_ID','TDN_EXECUTION_MODE','TDN_PROJECT_ROOT')):
        parser.error('CPU smoke needs a separate checkout without native scheduler overrides')
    from tdn.runtime.storage import configure_storage,contained_path
    from tdn.runtime.metadata import write_json,software_metadata
    from tdn.analysis.response.protocol import build_protocol,ancestors,digest
    configure_storage()
    base=contained_path(args.run_dir)
    if base.exists():parser.error('Preserve prior results; select a fresh run directory')
    base.mkdir(parents=True,exist_ok=False)
    (base/'logs').mkdir();(base/'state').mkdir()
    protocol=build_protocol('diagnostic-smoke',local_smoke=True,targets=args.targets)
    write_json(base/'protocol.json',protocol)
    receipt=dict(schema='tdn.response-diagnostic-cpu/v1',created_at=datetime.now(timezone.utc).isoformat(),
        execution_mode='local-cpu',device='cpu',native_gpu_verified=False,source=software_metadata(),
        protocol_sha256=digest(protocol),units={},status='RUNNING',run_dir=str(base))
    write_json(base/'cpu-workflow.json',receipt)
    failed=[]
    for stage in protocol['stages']:
        unit=protocol['units'][stage]
        observer=unit['kind'] in ('report','diagnostic_metrics')
        if not observer and any(receipt['units'].get(p,{}).get('status')!='COMPLETED' for p in unit['dependencies']):
            receipt['units'][stage]={'status':'BLOCKED_DEPENDENCY','requested':True,'device':'cpu'}
            failed.append(stage);write_json(base/'cpu-workflow.json',receipt)
            continue
        command=[sys.executable,str(ROOT/'scripts/response.py'),'--stage',stage,'--profile','diagnostic-smoke',
            '--run-dir',str(base/stage),'--protocol',str(base/'protocol.json'),'--device','cpu','--local-root',str(ROOT)]
        for parent in ancestors(protocol,stage):command+=['--prerequisite-dir',f'{parent}={base/parent}']
        out,err=base/'logs'/(stage+'.out'),base/'logs'/(stage+'.err')
        print(f'CPU diagnostic {stage} ({unit["kind"]})',flush=True)
        with out.open('w') as stdout,err.open('w') as stderr:
            result=subprocess.run(command,cwd=ROOT,stdout=stdout,stderr=stderr,check=False)
        state='COMPLETED' if result.returncode==0 else 'FAILED'
        receipt['units'][stage]={'status':state,'exit_code':result.returncode,'device':'cpu',
            'stdout':out.relative_to(base).as_posix(),'stderr':err.relative_to(base).as_posix()}
        if result.returncode:
            failed.append(stage)
            print(err.read_text()[-6000:],file=sys.stderr,flush=True)
        write_json(base/'cpu-workflow.json',receipt)
    receipt.update(status='COMPLETED' if not failed else 'FAILED',failed_units=failed,
        completed_units=sum(row['status']=='COMPLETED' for row in receipt['units'].values()),
        requested_units=len(protocol['units']),native_calibration_eligible=False,
        report=str(base/'report'/'index.html'),raven_project=str(base/'raven'))
    write_json(base/'cpu-workflow.json',receipt)
    print(json.dumps(receipt,indent=2,allow_nan=False))
    return 1 if failed else 0


if __name__=='__main__':raise SystemExit(main())
