#!/usr/bin/env python3
"""Preserve a stopped tangent campaign after actual terminal scheduler checks."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import response_diagnostic_archive as transport
from scripts.tangent_response_workflow import MANIFEST, CAMPAIGN, POINTER, coordinator_dead
from tdn.runtime.desktop_slurm import validate_profile

SCHEMA='tdn.tangent-response-archive/v1'
MAX_JOBS=20000


class TangentScheduler(transport.NativeScheduler):
    """Reuse bounded retained accounting, adding this program's live census."""
    def observe(self,jobs,*,user,root,run_ids):
        text=self.command(['squeue','--noheader','--user',user,'--format=%.40i|%.512k|%.256j|%.4096Z'])
        prefixes=tuple('tdn-tangent-response:'+name+':' for name in run_ids)
        for line in text.splitlines():
            fields=line.strip().split('|')
            if len(fields)!=4:raise ValueError('Unparseable live scheduler inventory')
            job,comment,name,workdir=(value.strip() for value in fields)
            if (job in {r['job_id'] for r in jobs} or prefixes and comment.startswith(prefixes)
                    or not comment and workdir==str(root) and name.startswith('tdn-tangent-response-')):
                raise ValueError('Campaign has live submitted or ambiguously owned tangent work')
        return super().observe(jobs,user=user,root=root,run_ids=[])


class TangentArchive:
    def __init__(self,root=ROOT,*,scheduler=None,site=None):
        self.root=Path(root).resolve();self.scheduler=scheduler or TangentScheduler();self.site=site
        self.inventory=self.root/'runs/.tangent-response-campaigns'

    def read(self,path,metadata=None,**kwargs):
        value,mark=transport.read_json(self.root,path,**kwargs)
        if metadata is not None:metadata.append(mark)
        return value

    def select(self,campaign=None):
        inventory=transport.confined(self.root,self.inventory,exists=True)
        if campaign:return transport.confined(self.root,inventory/transport.identifier(campaign)/'receipt.json',exists=True)
        paths=list(inventory.glob('*/receipt.json'))
        if not paths or len(paths)>1024:raise ValueError('Campaign inventory is absent or exceeds bounded discovery')
        rows=[]
        for path in paths:
            try:created=datetime.fromisoformat(self.read(path)['created_at'])
            except (ValueError,KeyError,TypeError):raise ValueError('Campaign creation timestamp invalid; select an exact campaign') from None
            if created.tzinfo is None:raise ValueError('Campaign creation timestamp needs a timezone; select an exact campaign')
            rows.append((created,path.parent.name,path))
        return max(rows)[2]

    def outer(self,receipt,metadata):
        value=receipt.get('raven_launch')
        if value is None:return None
        if not isinstance(value,dict) or value.get('project_root')!=str(self.root):raise ValueError('Raven association differs from this root')
        token=transport.identifier(value.get('launch_token'));name=transport.identifier(value.get('coordinator_run_id'))
        if name!='launch-'+token+'-a1' or value.get('coordinator_run_dir')!='runs/'+name:raise ValueError('Raven exact coordinator identity differs')
        base=transport.confined(self.root,'runs/'+name,exists=True)
        run=self.read(base/'run.json',metadata);summary=self.read(base/'summary.json',metadata)
        automation=run.get('metadata',{}).get('automation',{});plan=automation.get('plan',{})
        if (run.get('schema')!='raven.run/v1' or run.get('run_id')!=name or run.get('job_id') is not None or summary.get('schema')!='raven.summary/v1'
                or summary.get('id')!=name or run.get('state') not in transport.RECEIPT_TERMINAL or summary.get('state')!=run.get('state')
                or automation.get('execution')!='local' or automation.get('project_root')!=str(self.root)
                or plan.get('token')!=token or plan.get('project_root')!=str(self.root)
                or plan.get('script_relative')!='scripts/tangent_response_campaign.sh'
                or run.get('provenance',{}).get('script')!='scripts/tangent_response_campaign.sh'):
            raise ValueError('Exact associated Raven coordinator must have matching terminal records')
        return dict(action='PRESERVED',run_id=name,run_dir=base.relative_to(self.root).as_posix(),state=run['state'])

    def inspect(self,campaign=None):
        site=validate_profile(self.site,root=self.root) if self.site is not None else transport.native_policy(self.root)
        metadata=[];path=self.select(campaign);receipt=self.read(path,metadata)
        parent=transport.identifier(path.parent.name)
        if (receipt.get('schema')!='tdn.tangent-response-campaign/v1' or receipt.get('run_id')!=parent
                or receipt.get('project_root')!=str(self.root) or receipt.get('status') not in transport.RECEIPT_TERMINAL):
            raise ValueError('Exact campaign receipt must be terminal')
        if not coordinator_dead(receipt.get('coordinator',{})):raise ValueError('Campaign coordinator has not stopped')
        declared=receipt.get('run_ids');replaced=receipt.get('replaced_run_ids',{})
        if not isinstance(declared,dict) or not isinstance(replaced,dict):raise ValueError('Campaign phase inventory missing')
        names=[transport.identifier(name) for name in declared.values()]
        for rows in replaced.values():
            if not isinstance(rows,list):raise ValueError('Recovery origin inventory invalid')
            names.extend(transport.identifier(name) for name in rows)
        if len(names)>128 or len(names)!=len(set(names)):raise ValueError('Campaign roots are duplicate or exceed bounded inventory')
        moves=[];jobs=[];absent=[]
        for name in names:
            base=transport.confined(self.root,'runs/'+name)
            if not base.exists():absent.append(name);continue
            w=self.read(base/MANIFEST,metadata)
            if (w.get('kind')!='desktop-slurm-tangent-response' or w.get('execution_mode')!='desktop-slurm'
                    or w.get('run_id')!=name or w.get('run_dir')!=str(base) or w.get('root')!=str(self.root)
                    or w.get('campaign_run_id')!=parent):raise ValueError('Native root does not belong to this exact campaign')
            frozen=validate_profile(w.get('slurm_profile'),root=self.root)
            if frozen['user']!=site['user']:raise ValueError('Native owner differs')
            profilehash=w.get('slurm_profile_sha256')
            if not isinstance(profilehash,str) or not re.fullmatch('[0-9a-f]{64}',profilehash):raise ValueError('Frozen profile fingerprint missing')
            units=w.get('scientific_protocol',{}).get('units');recorded=self.read(base/'jobs.json',metadata)
            if not isinstance(units,dict) or not isinstance(recorded,list) or len(recorded)>MAX_JOBS:raise ValueError('Native job inventory invalid')
            seen=set()
            for row in recorded:
                if (not isinstance(row,dict) or row.get('stage') not in units or row['stage'] in seen
                        or not isinstance(row.get('job_id'),str) or not transport.JOB.fullmatch(row['job_id'])):
                    raise ValueError('Repeated or invalid exact native allocation')
                seen.add(row['stage']);jobs.append(dict(job_id=row['job_id'],stage=row['stage'],run_id=name,
                    expected_name='tdn-tangent-response-'+row['stage'],expected_comment=f"tdn-tangent-response:{name}:{row['stage']}:{profilehash}",expected_account=frozen.get('account')))
            moves.append(dict(kind='native_run',original='runs/'+name,archived='native/'+name))
        if len(jobs)>MAX_JOBS or len({j['job_id'] for j in jobs})!=len(jobs):raise ValueError('Duplicate or excessive native allocations')
        outer=self.outer(receipt,metadata)
        history=[self.outer({**receipt,'raven_launch':old},metadata) for old in receipt.get('raven_launch_history',[]) if old is not None]
        observed=self.scheduler.observe(jobs,user=site['user'],root=self.root,run_ids=names)
        proof=[]
        for job in jobs:
            row=observed.get(job['job_id'],{});state=row.get('state','UNKNOWN').split()[0].rstrip('+')
            if state not in transport.TERMINAL:raise ValueError('Every actual submitted job must be terminal')
            if (row.get('user')!=site['user'] or row.get('workdir')!=str(self.root) or row.get('job_name')!=job['expected_name']
                    or row.get('comment') and row['comment']!=job['expected_comment']
                    or job['expected_account'] is not None and row.get('account')!=job['expected_account']):raise ValueError('Native scheduler ownership differs')
            proof.append({**row,'run_id':job['run_id'],'stage':job['stage'],'state':state,
                          'comment_coverage':'VERIFIED' if row.get('comment') else 'NOT_RETAINED_BY_SCHEDULER'})
        parentroot=transport.confined(self.root,'runs/'+parent)
        if parentroot.exists():
            marker=self.read(parentroot/CAMPAIGN,metadata)
            if marker.get('run_id')!=parent or marker.get('receipt_path')!=str(path):raise ValueError('Parent campaign marker differs')
            moves.append(dict(kind='campaign_root',original='runs/'+parent,archived='campaign-root/'+parent))
        moves.append(dict(kind='campaign_receipt_directory',original=path.parent.relative_to(self.root).as_posix(),archived='campaign/'+parent))
        pointer=transport.confined(self.root,'runs/'+POINTER)
        if pointer.exists() and self.read(pointer,metadata,limit=65536).get('run_id') in [parent,*names]:
            moves.append(dict(kind='tangent_latest_pointer',original='runs/'+POINTER,archived='pointers/'+POINTER))
        for move in moves:
            item=transport.confined(self.root,move['original'],exists=True).stat();move['source_identity']=[item.st_dev,item.st_ino]
        return dict(schema=SCHEMA,campaign_run_id=parent,coordinator_status=receipt['status'],run_ids=names,absent_run_ids=absent,
            moves=moves,inspected_metadata=metadata,submitted_jobs=len(jobs),scheduler_proof=proof,
            scheduler_state_counts=dict(Counter(p['state'] for p in proof)),raven_coordinator=outer,raven_coordinator_history=history,
            native_source_fingerprints='PRESERVED_WITHOUT_CURRENT_SOURCE_COMPARISON',
            historical_path_contract='Frozen absolute paths unchanged; archived runs are evidence, not resumable coordinators')

    def archive(self,campaign=None,*,plan=False):
        with transport.controller_lock(self.root):
            selected=self.inspect(campaign)
            if plan:return {**{k:v for k,v in selected.items() if k!='scheduler_proof'},'action':'PLAN_ONLY','archive_destination':'runs/archived-tangent-response/<fresh-id>'}
            if self.inspect(selected['campaign_run_id'])!=selected:raise ValueError('Campaign evidence changed before archive')
            for expected in selected['inspected_metadata']:
                if transport.read_json(self.root,expected['path'])[1]!=expected:raise ValueError('Campaign metadata changed during scheduler check')
            parent=transport.confined(self.root,'runs/archived-tangent-response');parent.mkdir(exist_ok=True)
            dest=parent/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:12]);dest.mkdir(mode=0o700)
            if any(transport.confined(self.root,m['original']).stat().st_dev!=dest.stat().st_dev for m in selected['moves']):raise ValueError('Archive requires one project filesystem')
            checksum=transport._write(dest/'scheduler-proof.json',selected.pop('scheduler_proof'))
            manifest={**selected,'status':'MOVING','action':'ARCHIVE','archive_directory':dest.relative_to(self.root).as_posix(),
                      'completed_moves':[],'scheduler_proof':dict(path='scheduler-proof.json',sha256=checksum)}
            journal=dest/'archive-manifest.json';transport._write(journal,manifest,limit=128*2**10);moved=[]
            try:
                for move in selected['moves']:
                    left=transport.confined(self.root,move['original'],exists=True);right=transport.confined(self.root,dest/move['archived'])
                    right.parent.mkdir(parents=True,exist_ok=True)
                    transport.rename_noreplace(self.root,left,right,move['source_identity']);moved.append((left,right,move['source_identity']))
                    manifest['completed_moves'].append(move['original']);transport._write(journal,manifest,limit=128*2**10)
            except BaseException:
                restored=[];retained=[]
                for left,right,identity in reversed(moved):
                    try:transport.rename_noreplace(self.root,right,left,identity);restored.append(left.relative_to(self.root).as_posix())
                    except (OSError,ValueError):retained.append(right.relative_to(self.root).as_posix())
                manifest.update(status='INTERRUPTED_PRESERVED',restored_paths=restored,retained_archive_paths=retained)
                transport._write(journal,manifest,limit=128*2**10);raise
            manifest['status']='ARCHIVED';transport._write(journal,manifest,limit=128*2**10);return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--campaign');parser.add_argument('--plan',action='store_true');args=parser.parse_args(argv)
    try:print(json.dumps(TangentArchive().archive(args.campaign,plan=args.plan),indent=2));return 0
    except (OSError,ValueError,KeyError,TypeError) as error:print('Tangent archive: '+str(error),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
