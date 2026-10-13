#!/usr/bin/env python3
"""Execute one frozen unit or the explicitly nonpromoting CPU implementation smoke."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tdn.analysis.tangent_response.protocol import build_protocol,ancestors
from tdn.analysis.tangent_response.engine import execute,verify_science,inside
from tdn.runtime.metadata import write_json


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage');parser.add_argument('--protocol',type=Path);parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--prerequisite-dir',action='append',default=[]);parser.add_argument('--resume-from',type=Path)
    parser.add_argument('--profile',choices=('cpu-smoke',));parser.add_argument('--run-id')
    parser.add_argument('--target',choices=('fd_same_grid','galerkin_same_grid'),default='fd_same_grid')
    parser.add_argument('--worker-telemetry',default='cpu,memory');parser.add_argument('--sampling-seconds',type=float,default=5)
    parser.add_argument('--no-worker-telemetry',action='store_true')
    args=parser.parse_args(argv)
    if args.profile:
        if args.stage or args.protocol or args.device!='cpu' or args.prerequisite_dir or args.resume_from:
            parser.error('CPU smoke requires its own fresh complete protocol')
        if not args.run_id or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,127}',args.run_id):parser.error('Provide a safe fresh --run-id')
        base=inside(ROOT/'runs'/args.run_id)
        if base.exists():raise ValueError('Run directory exists; preserve it and select a fresh run ID')
        base.mkdir(parents=True)
        p=build_protocol('cpu-smoke',args.target,run_id=args.run_id,
            worker_telemetry=() if args.no_worker_telemetry else args.worker_telemetry,
            sampling_seconds=args.sampling_seconds)
        write_json(base/'protocol.json',p)
        outcomes={};completed={}
        for stage in p['stages']:
            unit=p['units'][stage];requested=ancestors(p,stage)
            missing=set(requested)-set(completed)
            if missing and unit['kind']!='report':
                outcomes[stage]={'status':'NOT_RUN_MISSING_PREREQUISITE','missing':sorted(missing)}
                print(stage+': NOT_RUN_MISSING_PREREQUISITE',flush=True);continue
            prereqs={name:completed[name] for name in requested if name in completed}
            try:
                summary=execute(stage,p,base/stage,device='cpu',prereqs=prereqs)
                completed[stage]=str(base/stage);outcomes[stage]=summary
                print(stage+': '+summary['status'],flush=True)
            except Exception as exc:
                outcomes[stage]={'status':'FAILED','error':str(exc)}
                print(stage+': FAILED '+str(exc),flush=True)
            write_json(base/'local_status.json',{'execution_scope':'CPU_IMPLEMENTATION_SMOKE','units':outcomes,
                'native_readiness':False,'scientific_promotion':False})
        failed=[name for name,value in outcomes.items() if value['status']!='COMPLETED']
        print(json.dumps({'run_dir':str(base),'requested_units':len(p['stages']),'completed_units':len(completed),
            'incomplete_units':failed,'decision':'IMPLEMENTATION_ONLY','native_readiness':False}))
        return 1 if failed else 0
    if not args.stage or not args.protocol or not args.run_dir:parser.error('A stage requires --stage, --protocol and --run-dir')
    p=json.loads(inside(args.protocol).read_text());prereqs={}
    for item in args.prerequisite_dir:
        name,sep,value=item.partition('=')
        if not sep or not name or name in prereqs:parser.error('Prerequisite uses unique UNIT=PATH or PHASE/UNIT=PATH')
        prereqs[name]=str(inside(value))
    result=execute(args.stage,p,args.run_dir,device=args.device,prereqs=prereqs,resume_from=args.resume_from)
    print(json.dumps(result,allow_nan=False));return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except (OSError,ValueError,RuntimeError) as exc:print('Tangent response: '+str(exc),file=sys.stderr);raise SystemExit(2)
