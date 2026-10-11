from pathlib import Path
import json,hashlib,collections
import review_io as io
BASE=Path(__file__).resolve().parent
manifest=json.loads((BASE.parent/'advance-full-native-upload/manifest.json').read_bytes())
export_index=json.loads((BASE.parent/'advance-full-native-upload/index.json').read_bytes())
files={r['path']:r for r in manifest['files']}
workflow=io.load('advance-workflow.json');jobs=io.load('jobs.json');prefix='runs/'+workflow['run_id']+'/'
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def original(name):return files[prefix+name]
def checkhash(name,h):assert original(name)['sha256']==h,(name,'manifest hash mismatch')
result={'run_id':workflow['run_id'],'archive_sha256':export_index['sha256'],'parts':len(export_index['parts']),'retained_files':export_index['statistics']['included_files'],'omitted_files':export_index['statistics']['omitted_files'],'source_tree_sha256':workflow['source_tree_sha256'],'stages':{},'mismatches':[]}
for row in jobs:
 stage=row['stage'];science=io.load(stage+'/science_manifest.json');summary=io.load(stage+'/summary.json');protocol=io.load(stage+'/protocol.json');seal=io.load(stage+'/workflow-seal.json')
 assert io.raw(stage+'/COMPLETED').decode().strip()==digest(science)
 assert seal['protocol_sha256']==science['protocol_sha256']==summary['protocol_sha256']==digest(protocol)
 assert science['unit_sha256']==digest(protocol['units'][stage])
 assert science['source_tree_sha256']==summary['source_tree_sha256']==workflow['source_tree_sha256']
 for name,h in seal['files'].items():checkhash(stage+'/'+name,h)
 counted=collections.Counter()
 for name,h in science['artifacts'].items():
  checkhash(stage+'/'+name,h);counted['retained_and_transport_verified' if original(stage+'/'+name)['included'] else 'declared_omission_unavailable']+=1
 for field in ('prerequisites','lineage'):
  for name,h in science[field].items():checkhash(name+'/science_manifest.json',h)
 result['stages'][stage]={'job_id':row['job_id'],'status':summary['status'],'correctness_failures':summary['correctness_failures'],'git_commit':summary['software']['git_commit'],'device':summary['device'],'scientific_artifact_references':dict(counted),'science_elapsed_seconds':summary['elapsed_seconds'],'verdict_counts':summary['verdict_counts']}
result['scope']='All included archive bytes were streamed and hashed first. This verifies consistency of stage/protocol/source/prerequisite seals and all original science artifact hashes with the verified export manifest. Omitted checkpoint/array/figure bytes cannot be rehashed or replayed from this review subset.'
(BASE/'provenance-summary.json').write_text(json.dumps(result,indent=2)+'\n')
print({'stages':len(result['stages']),'statuses':dict(collections.Counter(v['status'] for v in result['stages'].values())),'correctness_failures':sum(v['correctness_failures'] for v in result['stages'].values()),'science_references':dict(sum((collections.Counter(v['scientific_artifact_references']) for v in result['stages'].values()),collections.Counter())),'mismatches':result['mismatches']})
