from pathlib import Path
import gzip
import hashlib
import importlib.util
import json
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
INPUT = ROOT / '.runtime/portfolio-upload-20261009'
spec=importlib.util.spec_from_file_location('compact',ROOT/'scripts/compact_review.py')
compact=importlib.util.module_from_spec(spec);spec.loader.exec_module(compact)
idx=json.loads((INPUT/'index.json').read_bytes())
manifest=json.loads((INPUT/'manifest.json').read_bytes())
prefix='runs/'+manifest['primary_run']+'/'
files={}
for row in manifest['files']:
    if not row['included'] or not row['path'].startswith(prefix):continue
    rel=row['path'][len(prefix):]
    parts=Path(rel).parts
    if parts[0]=='tower':continue
    if parts[0]=='report' and len(parts)>1 and parts[1]=='tables':continue
    if rel in ('report/figures/chart-data.json.gz','report/figures/learning-range-data.json.gz'):continue
    if 'completed-groups' in parts:continue
    if parts[0].startswith('confirm-') and not parts[0].startswith('confirm-prepare-') and Path(rel).name in ('endpoint_rows.json','timing_rounds.json'):continue
    files[rel]=row
needed={r['sha256'] for r in files.values()}
allrows={r['path']:r for r in manifest['files']}
(BASE/'objects').mkdir(exist_ok=True)
print('Selected',len(files),'paths',len(needed),'unique payloads',sum(r['bytes'] for r in files.values()),'raw bytes',flush=True)
reader=compact.ChainReader(INPUT,idx['parts'])
seen=set();start=time.monotonic();last=start;count=0
try:
    with tarfile.open(fileobj=reader,mode='r|xz') as archive:
        for member in archive:
            if member.name==compact.MANIFEST:continue
            row=allrows[member.name];sha=row['sha256']
            if sha not in needed or sha in seen or not member.isfile():continue
            dest=BASE/'objects'/(sha+'.gz')
            check=hashlib.sha256()
            with archive.extractfile(member) as inp,dest.open('xb') as out:
                with gzip.GzipFile(fileobj=out,mode='wb',compresslevel=1,mtime=0) as enc:
                    while block:=inp.read(1024*1024):
                        check.update(block);enc.write(block);count+=len(block)
            if check.hexdigest()!=sha:raise ValueError('Hash mismatch '+member.name)
            seen.add(sha)
            if time.monotonic()-last>8:
                print(len(seen),'/',len(needed),'objects;',round(count/2**20),'MiB; elapsed',round(time.monotonic()-start,1),'s',flush=True);last=time.monotonic()
finally:reader.close()
assert seen==needed,(needed-seen)
(BASE/'evidence-index.json').write_text(json.dumps({'source_archive_sha256':idx['sha256'],'run_id':manifest['primary_run'],'scope':'Canonical exact bytes in gzip storage; report/table/Tower copies and duplicate confirm group rows excluded from working set; original archive retained intact','files':files},indent=2)+'\n')
print('READY',len(files),'paths;',round(sum(p.stat().st_size for p in (BASE/'objects').iterdir())/2**20,2),'MiB stored',flush=True)
