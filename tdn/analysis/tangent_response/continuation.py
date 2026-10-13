"""Freeze completed optimizer work in later, already allocated chunk units."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from tdn.runtime.metadata import software_metadata
from .contracts import clean
from . import training

ROOT=Path(__file__).resolve().parents[3]
COPY_ARTIFACTS=(
    'journal.pt','initial.pt','selected.pt','final.pt','criterion-diagnostic.pt',
    'action-stage.pt','learning_curves.jsonl','query_plan.json','fixed_controls.json','attempt.json',
)
REQUIRED_ARTIFACTS=set(COPY_ARTIFACTS)-{'action-stage.pt'}
SOFTWARE_FIELDS=('python','torch','numpy','scipy','torch_cuda_runtime')


def _inside(path):
    path=Path(path)
    if not path.is_absolute():path=ROOT/path
    if not path.resolve().is_relative_to(ROOT.resolve()):raise ValueError('Continuation artifacts must remain within the project')
    for item in (path,*path.parents):
        if item==ROOT:break
        if item.is_symlink():raise ValueError('Continuation paths cannot traverse symlinks')
    return path


def _hash(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def _read(path,limit=16*2**20):
    if not path.is_file() or path.stat().st_size>limit:raise ValueError('Continuation metadata missing or exceeds its bound')
    return json.loads(path.read_text())


def _origin_seal(origin,sourcehash):
    manifest=_read(origin/'science_manifest.json')
    checksum=_hash(origin/'science_manifest.json')
    marker=origin/'COMPLETED'
    if (manifest.get('schema')!='tdn.tangent-response-science/v1' or manifest.get('kind')!='training'
            or manifest.get('source_tree_sha256')!=sourcehash or not marker.is_file() or marker.stat().st_size>128
            or marker.read_text().strip()!=checksum):raise ValueError('Completed continuation requires the exact sealed source ancestor')
    artifacts=manifest.get('artifacts')
    required=REQUIRED_ARTIFACTS|{'trial.json','training_results.json','protocol.json','execution.json','summary.json'}
    if not isinstance(artifacts,dict) or not required<=set(artifacts):raise ValueError('Completed continuation lacks required sealed artifacts')
    for relative,expected in artifacts.items():
        rel=Path(relative)
        if rel.is_absolute() or '..' in rel.parts or not rel.parts:raise ValueError('Sealed continuation artifact leaves its stage')
        path=_inside(origin/rel)
        if not path.is_file() or _hash(path)!=expected:raise ValueError('Sealed continuation artifact changed: '+relative)
    protocol=_read(origin/'protocol.json');execution=_read(origin/'execution.json')
    stage=manifest.get('stage')
    if (protocol.get('profile')!=manifest.get('profile') or training.digest(protocol)!=manifest.get('protocol_sha256')
            or stage not in protocol.get('units',{}) or protocol['units'][stage].get('kind')!='training'
            or training.digest(protocol['units'][stage])!=manifest.get('unit_sha256')
            or any(execution.get(k)!=manifest.get(k) for k in ('stage','profile','protocol_sha256','source_tree_sha256'))):
        raise ValueError('Completed continuation ancestor execution identity differs')
    return manifest,checksum


def reuse_completed_trial(origin_path,new_path,config,samples,device,sourcehash):
    """Reuse exact generated completed work, without model/optimizer execution.

    The native controller verifies the full workflow ancestor seal before this
    call. This helper independently checks the science bytes and journal
    identity. A partial journal returns None for normal exact fitting/resume.
    The caller writes returned trial/training metadata into its current unit;
    checkpoint and observation bytes are copied exclusively into fresh paths.
    """
    origin,destination=_inside(origin_path),_inside(new_path)
    if origin==destination or origin.is_relative_to(destination) or destination.is_relative_to(origin):raise ValueError('Continuation requires distinct nonoverlapping stage roots')
    journal_path=_inside(origin/'journal.pt')
    if not journal_path.is_file():return None
    saved=torch.load(journal_path,weights_only=True,map_location='cpu')
    requested=config.get('updates',300);completed=saved.get('update')
    if type(requested) is not int or requested<1 or type(completed) is not int or completed<0:raise ValueError('Continuation committed-work counters invalid')
    if completed<requested:return None
    if completed!=requested:raise ValueError('Completed journal exceeds the exact declared work')
    cfg=json.loads(json.dumps(clean(config),allow_nan=False));runtime=software_metadata()
    identity=dict(config=cfg,data=training._data_hash(samples),source=cfg.get('source_tree_sha256',runtime.get('source_tree_sha256')),
        software={key:runtime.get(key) for key in SOFTWARE_FIELDS},device=str(device))
    identity=json.loads(json.dumps(clean(identity),allow_nan=False))
    if identity['source']!=sourcehash or saved.get('identity')!=identity:
        raise ValueError('Completed continuation data/config/source/software/device identity changed')
    manifest,checksum=_origin_seal(origin,sourcehash)
    trial=_read(origin/'trial.json');training_results=_read(origin/'training_results.json')
    ledger,committed=trial.get('cost',{}),saved.get('cost',{})
    costs_match=(isinstance(ledger,dict) and isinstance(committed,dict) and set(ledger)==set(committed)
        and all(ledger[key]==committed[key] for key in ledger if key!='checkpoint_seconds'))
    # The final journal is serialized before its own write duration is charged
    # to the public trial ledger. Preserve that honest post-write charge.
    if 'checkpoint_seconds' in ledger:
        old,new=committed.get('checkpoint_seconds'),ledger['checkpoint_seconds']
        costs_match &= (isinstance(old,(int,float)) and not isinstance(old,bool) and isinstance(new,(int,float))
            and not isinstance(new,bool) and math.isfinite(old) and math.isfinite(new) and 0<=old<=new)
    if (trial.get('updates')!=completed or trial.get('requested_updates')!=requested
            or trial.get('trial_id')!=cfg.get('id',cfg.get('family',cfg.get('arm','learned_response')))
            or trial.get('family')!=cfg.get('family',cfg.get('arm','learned_response'))
            or trial.get('seed')!=cfg.get('seed',0) or training_results.get('trials')!=[trial]
            or trial.get('journal_sha256')!=_hash(journal_path)
            or trial.get('selected_checkpoint_sha256')!=_hash(origin/'selected.pt')
            or trial.get('work')!=saved.get('work') or not costs_match):
        raise ValueError('Completed trial metadata differs from its exact generated journal')
    files=[name for name in COPY_ARTIFACTS if name in manifest['artifacts']]
    targets=[destination/name for name in [*files,'trial.json','training_results.json','continuation_reuse.json']]
    if any(path.exists() or path.is_symlink() for path in targets):raise ValueError('Preserve existing continuation artifacts; no overwrites')
    destination.mkdir(parents=True,exist_ok=True)
    copied={}
    for name in files:
        source=origin/name;target=_inside(destination/name)
        with source.open('rb') as source_stream,target.open('xb') as output:
            while block:=source_stream.read(1024*1024):output.write(block)
            output.flush();os.fsync(output.fileno())
        copied[name]=_hash(target)
        if copied[name]!=manifest['artifacts'][name]:raise ValueError('Continuation source changed while copying; preserve partial output')
    receipt=dict(schema='tdn.tangent-response-completed-continuation/v1',continuation_noop=True,new_committed_updates=0,
        origin_path=origin.relative_to(ROOT).as_posix(),origin_science_manifest_sha256=checksum,
        origin_trial_sha256=manifest['artifacts']['trial.json'],source_tree_sha256=sourcehash,
        committed_updates_preserved=completed,artifacts=copied,
        cost_scope='Original cumulative optimizer ledger unchanged; current body/worker/allocation costs are observed separately')
    raw=(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    if len(raw)>2*2**20:raise ValueError('Continuation receipt exceeds its bound')
    with (destination/'continuation_reuse.json').open('xb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    result=copy.deepcopy(trial)
    result.update(continuation_noop=True,new_committed_updates=0,origin_science_manifest_sha256=checksum,
        continuation_origin=receipt['origin_path'],continuation_reuse=receipt)
    return result
