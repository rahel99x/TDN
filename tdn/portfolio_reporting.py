"""Bounded stdlib-only Tower sidecars for the portfolio; no Tower changes.

Exact canonical JSON records remain in paged CSV cells with source pointers.
A valid manifest/marker and matching consumed-file hash permit sealed-source
labels. Failed, partial, or corrupt artifacts retain raw content with NA display
verdicts; publication never silently promotes them into confirmed science.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import stat

from tdn.reporting import register_log

MAX_SOURCE_BYTES = 512 << 20
PAGE_ROWS = 128
PAGE_BYTES = 256 << 10
MAX_TOTAL_BYTES = 1 << 30
FILES = ('rows.jsonl','summary.json','catalog.json','learning_curves.jsonl','endpoint_rows.json',
         'comparisons.json','claims.json','prototype_rows.json','prototype_summary.json','scaling_rows.json',
         'diagnostic_rows.json','profile_rows.json','diagnostics.json','validation_rows.json','freeze.json',
         'training_plan.json','analysis.json','timing_rounds.json','coverage.json')
COLUMNS = ('source_path','source_sha256','source_pointer','source_line','evidence_status',
           'validation_error','experiment_id','family','track','status','raw_verdict',
           'verdict','score_1_100','record_json')


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _directory(path):
    path=Path(os.path.abspath(path))
    for item in (path,*path.parents):
        if not stat.S_ISDIR(item.lstat().st_mode):
            raise ValueError('Portfolio report directories must not contain symlinks')
    return path


def _read(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size>MAX_SOURCE_BYTES:
        raise ValueError('Portfolio source must be a bounded regular file')
    with path.open('rb') as stream:
        before=os.fstat(stream.fileno());raw=stream.read(MAX_SOURCE_BYTES+1);after=os.fstat(stream.fileno())
    signature=lambda info:(info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns)
    if signature(before)!=signature(after) or signature(after)!=signature(path.lstat()) or len(raw)!=before.st_size:
        raise ValueError('Portfolio source changed while reading')
    return raw


def _decode(raw):
    def bad(value):raise ValueError('Nonfinite JSON constant '+value)
    return json.loads(raw,parse_constant=bad)


def _seal(root,path,sha):
    try:
        manifest=_decode(_read(root/'science_manifest.json'))
        marker=_read(root/'COMPLETED').decode().strip()
        if marker!=_digest(manifest):raise ValueError('Stage completion marker differs from manifest')
        if manifest.get('schema')!='tdn.portfolio-science/v1':raise ValueError('Unexpected science manifest schema')
        artifacts=manifest.get('artifacts',{})
        entry=artifacts.get(path.relative_to(root).as_posix())
        expected=entry.get('sha256') if isinstance(entry,dict) else entry
        if expected!=sha:raise ValueError('Consumed source bytes differ from sealed hash')
        return 'VERIFIED_MANIFEST_BOUND_SOURCE',''
    except (OSError,ValueError,TypeError,KeyError) as exc:
        return 'UNVERIFIED_DIAGNOSTIC',str(exc)


def _records(path,value):
    if path.suffix=='.jsonl':
        return [(f'/{i}',i+1,row) for i,row in enumerate(value)]
    if isinstance(value,list):return [(f'/{i}',None,row) for i,row in enumerate(value)]
    if isinstance(value,dict):
        lists=[(key,items) for key,items in value.items() if isinstance(items,list)]
        if lists:return [(f'/{key}/{i}',None,row) for key,items in lists for i,row in enumerate(items)]
    return [('',None,value)]


def _cell(value):
    if isinstance(value,str) and value.lstrip().startswith(('=','+','-','@')):return "'"+value
    return value


def _csv(rows):
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,COLUMNS,lineterminator='\n');writer.writeheader()
    for row in rows:writer.writerow({k:_cell(row.get(k,'')) for k in COLUMNS})
    return stream.getvalue().encode()


def publish_outputs(report_dir,source_dirs):
    report_dir=_directory(report_dir);output=report_dir/'outputs';output.mkdir(exist_ok=True);_directory(output)
    table_dir=output/'portfolio';table_dir.mkdir(exist_ok=True);_directory(table_dir)
    pages=[];omissions=[];counts={};total=0;sealed=0;unverified=0
    for rawroot in sorted(set(map(str,source_dirs))):
        root=_directory(rawroot)
        if root==report_dir or report_dir in root.parents:raise ValueError('Scientific source cannot be Tower report directory')
        for filename in FILES:
            path=root/filename
            if not path.exists():continue
            try:
                raw=_read(path);total+=len(raw)
                if total>MAX_TOTAL_BYTES:raise ValueError('Portfolio aggregate reporting read budget exceeded')
                sha=hashlib.sha256(raw).hexdigest();evidence,error=_seal(root,path,sha)
                value=[_decode(line) for line in raw.splitlines() if line.strip()] if path.suffix=='.jsonl' else _decode(raw)
                records=[]
                for pointer,line,record in _records(path,value):
                    fields=record if isinstance(record,dict) else {}
                    assessment=fields.get('assessment',{})
                    raw_verdict=assessment.get('verdict',fields.get('verdict','NA'))
                    row=dict(source_path=str(path),source_sha256=sha,source_pointer=pointer,source_line=line,
                        evidence_status=evidence,validation_error=error,experiment_id=fields.get('experiment_id'),
                        family=fields.get('family'),track=fields.get('track'),status=fields.get('status'),raw_verdict=raw_verdict,
                        verdict=raw_verdict if not error else 'NA',score_1_100=assessment.get('score_1_100',1) if not error else 1,
                        record_json=json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False))
                    records.append(row);sealed+=not bool(error);unverified+=bool(error)
                name=root.name+'--'+filename.replace('.','-')
                counts[name]=len(records);pending=[]
                def flush():
                    if not pending:return
                    rawpage=_csv(pending);destination=table_dir/f'{len(pages)+1:06d}.csv'
                    if destination.is_symlink():raise ValueError('Symlink report output refused')
                    destination.write_bytes(rawpage)
                    pages.append({'table':name,'path':str(destination.relative_to(report_dir)),
                                  'rows':len(pending),'bytes':len(rawpage),'sha256':hashlib.sha256(rawpage).hexdigest()})
                    pending.clear()
                for row in records:
                    if pending and (len(pending)>=PAGE_ROWS or len(_csv([*pending,row]))>PAGE_BYTES):flush()
                    if len(_csv([row]))>PAGE_BYTES:
                        # Exact large JSON is linked as its own source-backed artifact, never truncated.
                        blob=table_dir/f'record-{hashlib.sha256(row["record_json"].encode()).hexdigest()}.json'
                        if blob.is_symlink():raise ValueError('Symlink report output refused')
                        blob.write_text(row['record_json']+'\n')
                        row['record_json']=json.dumps({'external_exact_record':str(blob.relative_to(report_dir)),
                            'sha256':hashlib.sha256(blob.read_bytes()).hexdigest()})
                    pending.append(row)
                flush()
            except (OSError,ValueError,TypeError,json.JSONDecodeError) as exc:
                omissions.append({'source':str(path),'reason':str(exc)})
    manifest={'schema':'tdn.portfolio-tables/v1','pages':pages,'row_counts':counts,
              'sealed_source_records':sealed,'unverified_diagnostic_records':unverified,
              'omissions':omissions,'reporting_complete':not omissions,
              'scope':'Source-bound projections only. Stage completion, mathematical checks and scientific advantage are separate. Unverified display scores are NA; original values retained.'}
    index=output/'portfolio-tables.json'
    if index.is_symlink():raise ValueError('Symlink report output refused')
    index.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    # Re-publication updates pages without accumulating duplicate Tower log registrations.
    known=json.loads((report_dir/'logs.json').read_text()).get('logs',[])
    if not any(item.get('id')=='portfolio-analytics' for item in known):
        register_log(report_dir,'portfolio-analytics',str(index),label='TDN portfolio: every source-linked record',
                     group='research',description='Paged complete experiment/checkpoint/loss/endpoint/prototype projections; NA remains explicit.')
    return manifest
