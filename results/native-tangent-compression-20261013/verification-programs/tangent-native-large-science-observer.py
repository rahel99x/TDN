"""Read-only re-encoding audit; never executes native scientific stages."""
import collections
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import zipfile

from tdn.analysis.response import diagnostic_metrics as shared
from tdn.analysis.response import diagnostic_report as portable
from tdn.analysis.tangent_response import report,report_artifacts as artifacts
from tdn.research.protocol import digest
from tdn.runtime.metadata import write_json

native=Path('.runtime/native-compression-review-20261013-3e51c6f1/restored/runs/tangent-response-pilot-20261013T085856-8a44d33a-compression-20261013T094550-3e51c6f1')
science=native/'compression'
manifest,manifest_sha,_=shared.read_metadata(science/'science_manifest.json',root=science)
assert (science/'COMPLETED').read_text().strip()==manifest_sha
protocol,_,_=shared.read_metadata(native/'protocol.json',root=native)
observer=Path('.runtime/onboarding/tangent-native-bulk-observer-20261013-b')
observer.mkdir()
project=observer/'raven';shared._project_files(project,report.CONFIG)
output=project/'runs/native-bulk-encoding-observer/outputs';output.mkdir(parents=True)
started=time.monotonic()
payload=artifacts.ScientificPayload(science/'compression_results.json',science,manifest['artifacts']['compression_results.json'])
decoded=payload.json()
copies={};sources=[];tables=collections.defaultdict(list);decisions=[]
for name,raw,value in [('compression_results.json',payload,decoded),('decision.json',None,None)]:
    source_stage='compression';source_manifest_sha=manifest_sha
    if raw is None:
        source_stage='decision';decision_science=native/source_stage
        decision_manifest,source_manifest_sha,_=shared.read_metadata(decision_science/'science_manifest.json',root=decision_science)
        assert (decision_science/'COMPLETED').read_text().strip()==source_manifest_sha
        value,checksum,raw=shared.read_metadata(decision_science/name,root=decision_science,expected=decision_manifest['artifacts'][name])
    else:checksum=raw.sha256
    pointer='sources/'+checksum+'.json';copies[pointer]=raw
    source=dict(stage=source_stage,name=name,path=pointer,sha256=checksum,bytes=len(raw),science_manifest_sha256=source_manifest_sha,
        source_tree_sha256=manifest['source_tree_sha256'])
    sources.append(source)
    base=dict(stage=source_stage,source_path=pointer,source_sha256=checksum,source_file=name)
    tables['stage_metadata'].append(dict(report._flat(value),**base,source_record='/'))
    for table,row in report._tables(value,base):tables[table].append(row)
    if name=='decision.json':decisions.append(dict(value,**base))
del decoded
data=dict(schema=report.SCHEMA,profile=protocol['profile'],protocol_sha256=digest(protocol),source_tree_sha256=manifest['source_tree_sha256'],
    independent_unit='Original native parent field; this observer adds no independent scientific measurements',
    native_gpu_verification='NOT_INFERRED_FROM_REPORTING',sources=sources,decisions=decisions,tables=dict(tables),
    omissions=[],lineage_projects=[],incomplete_attempts=[],lineage_costs={},
    summary=dict(original_compression_science_completed=1,original_native_compression_worker_failed=1,
        compiled_records=sum(len(rows) for name,rows in tables.items() if name!='stage_metadata')),
    observer_scope='Fresh posthoc lossless encoding audit. Original native SCI COMPLETED and worker FAILED remain unchanged; gate B STOP remains unchanged.')
index,published=artifacts.publish(output,data,copies,report.CONFIG,'native-bulk-encoding-observer')
overview=output.parent;(overview/'logs').mkdir()
write_json(overview/'run.json',dict(schema='raven.run/v1',project_id=report.CONFIG['project_id'],run_id=overview.name,
    experiment_id='tangent-response/encoding-audit',attempt=1,name='TDN/native bulk artifact encoding audit',state='COMPLETED',
    paths=dict(metrics='metrics.jsonl',summary='summary.json',log_index='logs.json',outputs='outputs',logs='logs',contract='outputs/contract.json'),
    metadata=dict(scope=data['observer_scope'],source_tree_sha256=data['source_tree_sha256'],native_gpu_verification='NOT_INFERRED_FROM_REPORTING')))
write_json(overview/'summary.json',dict(schema='raven.summary/v1',project_id=report.CONFIG['project_id'],id=overview.name,
    name='TDN/native bulk artifact encoding audit',experiment_id='tangent-response/encoding-audit',attempt=1,state='COMPLETED',exit_code=0,
    results=data['summary'],metadata=dict(scope=data['observer_scope'])))
write_json(overview/'logs.json',dict(schema='raven.logs/v1',run_id=overview.name,logs=[dict(id='audit',path='logs/audit.log',label='Encoding audit',group='Research')]))
(overview/'logs/audit.log').write_text(data['observer_scope']+'\n')
(overview/'metrics.jsonl').write_text(json.dumps(shared.metric_record(data['summary'],phase='artifact_encoding'))+'\n')
write_json(output/'contract.json',dict(version=1,outputs=[dict(path='outputs/tangent_research.json',format='json',required=True,min_bytes=1,
    sha256=hashlib.sha256((output/'tangent_research.json').read_bytes()).hexdigest())]))
source=next(row for row in artifacts.read_section(output,index,'sources') if row['name']=='compression_results.json')
whole=hashlib.sha256();size=0
for part in artifacts.source_bytes(output,source):whole.update(part);size+=len(part)
assert size==len(payload) and whole.hexdigest()==payload.sha256
table_json=sum(chunk['json']['bytes'] for table in index['tables'].values() for chunk in table['chunks'])
table_csv=sum(chunk['csv']['bytes'] for table in index['tables'].values() for chunk in table['chunks'])
uncompressed=sum(file.stat().st_size for file in project.rglob('*') if file.is_file())
print(json.dumps(dict(phase='published',source_bytes=size,table_json_bytes=table_json,table_csv_bytes=table_csv,
    uncompressed_project_bytes=uncompressed,artifact_projects=len(index['artifact_projects']),elapsed_seconds=time.monotonic()-started)),flush=True)
bundle=observer/'native-bulk-encoding-raven.zip'
receipt=portable.portable_export(project,bundle)
moved=observer/'relocated';moved.mkdir()
with zipfile.ZipFile(bundle) as archive:archive.extractall(moved)
sys.path.insert(0,'/workspace/Raven')
from raven import projects,reporting,metrics
from raven import artifacts as native_artifacts
counts=collections.Counter()
for root in [moved]+[moved/'runs'/overview.name/'outputs'/page['path'] for page in index['artifact_projects']]:
    discovery=projects.discover_project(str(root));assert not discovery['warnings'];assert len(discovery['runs'])<=255
    for record in discovery['runs']:
        run=projects.validate_inventory(record['inventory']);attempt=root/'runs'/run['run_id'];assert run.get('job_id') is None
        reporting._summary(json.loads((attempt/'summary.json').read_text()))
        for line in (attempt/'metrics.jsonl').read_text().splitlines():metrics._record(json.loads(line));counts['metric_records']+=1
        result=native_artifacts.validate_contract(json.loads((attempt/'outputs/contract.json').read_text()),str(attempt))
        assert result['valid'],result;counts['attempts']+=1;counts['contracts']+=1
    counts['projects']+=1
relocated_output=moved/'runs'/overview.name/'outputs'
relocated_index=json.loads((relocated_output/'tangent_research.json').read_text())
source=next(row for row in artifacts.read_section(relocated_output,relocated_index,'sources') if row['name']=='compression_results.json')
whole=hashlib.sha256()
for part in artifacts.source_bytes(relocated_output,source):whole.update(part)
assert whole.hexdigest()==payload.sha256
result=dict(schema='tdn.tangent-response-native-large-artifact-verification/v1',source_sha256=payload.sha256,source_bytes=len(payload),
    original_science_manifest_sha256=manifest_sha,source_tree_sha256=manifest['source_tree_sha256'],protocol_sha256=digest(protocol),
    compiled_records=data['summary']['compiled_records'],table_json_bytes=table_json,table_csv_bytes=table_csv,
    uncompressed_project_bytes=uncompressed,projected_with_whole_original_export_bytes=uncompressed+335870246,
    unchanged_transport_bound_bytes=2<<30,portable_sha256=receipt['sha256'],portable_bytes=receipt['bytes'],portable_members=receipt['included_files'],
    default_raven_budget_bytes=8<<20,counts=dict(counts),relocated_whole_source_sha256=whole.hexdigest(),
    elapsed_seconds=time.monotonic()-started,issues=[],original_gate_b=published['decisions'][0]['decision'],
    scope=data['observer_scope'])
assert result['projected_with_whole_original_export_bytes']<result['unchanged_transport_bound_bytes']
write_json(Path('.runtime/onboarding/tangent-native-large-science-verification.json'),result)
print(json.dumps(result),flush=True)
