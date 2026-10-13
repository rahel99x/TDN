#!/usr/bin/env python3
"""Trusted fresh-process deployment benchmark; no teacher/action labels accepted."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True,type=Path)
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--device',choices=('cpu','cuda'),required=True)
    parser.add_argument('--source-sha256',required=True)
    args=parser.parse_args(argv)
    import torch
    from tdn.runtime.metadata import software_metadata
    from tdn.runtime.precision import reference_precision
    from tdn.analysis.tangent_response.contracts import equation_geometry,physical_metrics
    from tdn.analysis.tangent_response.engine import inside
    from tdn.analysis.tangent_response.training import load_model
    reference_precision()
    if software_metadata()['source_tree_sha256']!=args.source_sha256:raise ValueError('Fresh-process executable source changed')
    if args.device=='cuda' and (not torch.cuda.is_available() or not os.environ.get('SLURM_JOB_ID') or os.environ.get('TDN_EXECUTION_MODE')!='desktop-slurm'):
        raise RuntimeError('CUDA benchmark requires the actual native allocation; CPU fallback is forbidden')
    payload=torch.load(inside(args.input),weights_only=True,map_location='cpu')
    if set(payload)!={'u0','record'} or set(payload['record'])-{'query_id','parent_id','grid','lengths','horizon','target_kind','kappa','reaction_rate'}:
        raise ValueError('Fresh-process inference accepts initial state and known context only')
    record=payload['record'];u0=payload['u0']
    if u0.dtype!=torch.float32 or not bool(torch.isfinite(u0).all()):raise ValueError('Actual finite FP32 inference input required')
    eq,geom=equation_geometry(record)
    model,metadata=load_model(inside(args.checkpoint),args.device)
    if metadata.get('identity',{}).get('source')!=args.source_sha256:raise ValueError('Checkpoint source differs from inference')
    if metadata.get('checkpoint_role')!='validation-selected deployment':raise ValueError('Deployment requires the frozen validation-selected checkpoint')
    u0=u0.to(args.device)
    if args.device=='cuda':torch.cuda.synchronize()
    started=time.perf_counter()
    with torch.no_grad():output=model(u0,record['horizon'],eq,geom,record['target_kind'])
    if args.device=='cuda':torch.cuda.synchronize()
    if output.dtype!=torch.float32 or not bool(torch.isfinite(output).all()):raise ValueError('Deployment output must be finite FP32')
    print(json.dumps({'status':'COMPLETED','device':args.device,'query_id':record['query_id'],
        'complete_inference_seconds':time.perf_counter()-started,'output_norm':physical_metrics(output,geom,record['target_kind']),
        'source_tree_sha256':args.source_sha256,'scope':'One fresh process; outer benchmark includes startup, checkpoint/input loading and serialization'}))

if __name__=='__main__':
    try:main()
    except (OSError,ValueError,RuntimeError) as exc:print(str(exc),file=sys.stderr);raise SystemExit(2)
