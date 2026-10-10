"""Read-only provenance and teacher-metadata review of exact retained evidence."""
import collections
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from review_io import load, raw, names

BASE = Path(__file__).resolve().parent
def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
protocol = load('protocol.json')['scientific_protocol']
manifest = json.loads((BASE.parent/'portfolio-upload-20261009/manifest.json').read_bytes())
prefix = 'runs/fedora-portfolio-20261009T183307014599Z/'
files = {x['path'][len(prefix):]: x for x in manifest['files'] if x['path'].startswith(prefix)}
result = {'scope': 'Metadata, retained bytes and declared omitted-file hashes; not replay or full omitted-payload verification', 'protocol_sha256': digest(protocol), 'stages': [], 'issues': []}
for stage in protocol['units']:
    m = load(stage+'/science_manifest.json'); summary=load(stage+'/summary.json')
    checks = {'protocol': m['protocol_sha256']==digest(protocol), 'unit': m['unit_sha256']==digest(protocol['units'][stage]), 'marker':raw(stage+'/COMPLETED').decode().strip()==digest(m), 'stored_protocol':load(stage+'/protocol.json')==protocol, 'status':summary['status']=='COMPLETED'}
    checks['lineage'] = all(hashlib.sha256(raw(prior+'/science_manifest.json')).hexdigest()==h for prior,h in m['lineage'].items())
    checks['prerequisites'] = all(hashlib.sha256(raw(prior+'/science_manifest.json')).hexdigest()==h for prior,h in m['prerequisites'].items())
    retained=omitted=0
    for name,h in m['artifacts'].items():
        f=files.get(stage+'/'+name)
        if f is None or f['sha256']!=h: result['issues'].append([stage,name,'missing or unequal declared hash'])
        elif f['included']: retained+=1
        else: omitted+=1
    rows=load(stage+'/rows.json')['rows']
    checks['ledgers']=rows==[json.loads(x) for x in raw(stage+'/rows.jsonl').splitlines() if x.strip()]
    checks['count']=len(rows)==summary['experiment_count']==len({x['experiment_id'] for x in rows})
    result['stages'].append({'stage':stage,'checks':checks,'retained_artifacts':retained,'omitted_artifacts':omitted,'source_tree_sha256':m['source_tree_sha256']})
    for k,v in checks.items():
        if not v:result['issues'].append([stage,k])
parents=[];refs=[]
for name in names():
    if name.endswith('/data_manifest.json'):
        d=load(name); stage=name.split('/')[0]
        for record in d['records']:
            p=load(stage+'/'+record['metadata']);parents.append(p)
            for r in p['references'].values():refs.append(dict(r,split=p['split'],regime=p['regime'],parent=p['parent_id']))
groups={}
for split in ['train','validation','confirmation']:
    for track in ['discrete','continuum']:
        rs=[r for r in refs if r['split']==split and r['track']==track]
        groups[split+'/'+track]={'references':len(rs),'parents':len({r['parent'] for r in rs}),'accepted':sum(r['accepted'] for r in rs),'max_uncertainty_rms':max(r['uncertainty_rms'] for r in rs),'max_uncertainty_max':max(r['uncertainty_max_bound'] for r in rs),'grids':sorted({r['grid'] for r in rs}),'horizons':sorted({r['horizon'] for r in rs})}
result['references']=groups
result['parent_counts']=dict(collections.Counter(p['split'] for p in parents))
result['split_id_overlap']={a+'/'+b:sorted({p['field_cluster'] for p in parents if p['split']==a}&{p['field_cluster'] for p in parents if p['split']==b}) for a,b in [('train','validation'),('train','confirmation'),('validation','confirmation')]}
result['tests']=[]
for n in names():
    if n.endswith('/tests.xml'):
        root=ET.fromstring(raw(n));cases=list(root.iter('testcase'))
        result['tests'].append({'path':n,'cases':len(cases),'failures':len(list(root.iter('failure'))),'errors':len(list(root.iter('error'))),'skipped':len(list(root.iter('skipped')))})
(BASE/'provenance-summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['stages','tests']},indent=2))
print('stages',len(result['stages']),'test batches',len(result['tests']),'test instances',sum(t['cases'] for t in result['tests']))
