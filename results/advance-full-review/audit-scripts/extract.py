"""Select exact canonical files from an already fully verified compact archive."""
from pathlib import Path, PurePosixPath
import gzip, hashlib, importlib.util, json, tarfile, time
ROOT=Path('/workspace/TDN'); BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('compact',ROOT/'scripts/compact_review.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
upload=ROOT/'.runtime/advance-full-native-upload';meta=json.loads((upload/'index.json').read_bytes());manifest=json.loads((upload/'manifest.json').read_bytes())
def wanted(row):
    if not row['included']:return False
    p=PurePosixPath(row['path']); parts=p.parts[2:];stage=parts[0]
    if stage=='tower':return False
    if len(parts)==1 or stage in ('logs','state','reporter-tests'):return True
    if len(parts)==2:return p.name not in ('rows.json','rows.jsonl','review.csv','review.md')
    if stage=='report' and p.name in ('manifest.json','chart-data.json.gz','index.html'):return True
    if stage=='prepare' or stage.startswith('confirm-prepare-') or stage=='scaling-prepare':return p.suffix=='.json'
    return False
selected={str(PurePosixPath(r['path']).relative_to('runs/'+manifest['primary_run'])):r for r in manifest['files'] if wanted(r)}
by_hash={r['sha256']:r for r in selected.values()};objects=BASE/'objects';objects.mkdir(exist_ok=False)
lookup={r['path']:r for r in manifest['files']}
reader=mod.ChainReader(upload,meta['parts']);start=time.monotonic();done=set();names=set()
with tarfile.open(fileobj=reader,mode='r|gz') as tar:
    while member:=tar.next():
        name=mod.safe_name(member.name)
        if name==mod.MANIFEST:continue
        names.add(name)
        # Lookup original hashes; verified deduplication links need no payload read.
        if member.isfile():
            original=lookup[name]
            h=original['sha256']
            if h in by_hash and h not in done:
                stream=tar.extractfile(member);actual=hashlib.sha256();count=0
                with gzip.GzipFile(filename=str(objects/(h+'.gz')),mode='wb',compresslevel=6,mtime=0) as out:
                    while block:=stream.read(1024*1024):actual.update(block);out.write(block);count+=len(block)
                assert actual.hexdigest()==h and count==original['bytes'],name
                done.add(h)
reader.close()
assert done==set(by_hash),(len(done),len(by_hash))
index={'schema':'tdn.exact-canonical-evidence/v1','run_id':manifest['primary_run'],'archive_sha256':meta['sha256'],'full_transport_verified_files':sum(r['included'] for r in manifest['files']),'scope':'Exact canonical subset for analysis; full original compressed upload preserved; no numeric values altered.','files':{name:{'sha256':r['sha256'],'bytes':r['bytes'],'original_path':r['path']} for name,r in selected.items()}}
(BASE/'evidence-index.json').write_text(json.dumps(index,indent=2)+'\n')
print({'selected_files':len(selected),'unique_objects':len(done),'source_bytes':sum(r['bytes'] for r in selected.values()),'compressed_bytes':sum(p.stat().st_size for p in objects.iterdir()),'elapsed':time.monotonic()-start},flush=True)
