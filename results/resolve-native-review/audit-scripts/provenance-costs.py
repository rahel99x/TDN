"""Independent read-only audit of the verified native Resolve review bundle.

No binary checkpoint or array is loaded. Declared omitted artifact hashes are
compared as provenance declarations only, never claimed as measured bytes.
Run from the repository with .venv/bin/python .runtime/resolve-native-analysis/provenance-costs.py.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib,json,math,re,subprocess
from pathlib import Path
from xml.etree import ElementTree as ET
import review_io as io

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
UPLOAD=ROOT/'.runtime/resolve-native-upload'
def sha(data):return hashlib.sha256(data).hexdigest()
def digest(value):return sha(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
def duration(value):
    if not value:return None
    value=str(value);days=0
    if '-' in value:
        days,value=value.split('-',1);days=int(days)
    parts=[float(v) for v in value.split(':')]
    return days*86400+sum(v*60**i for i,v in enumerate(reversed(parts)))
def mem_bytes(value):
    if not value:return None
    m=re.fullmatch(r'([0-9.]+)([KMGT]?)',value)
    if not m:raise ValueError(value)
    return float(m[1])*{'':1,'K':2**10,'M':2**20,'G':2**30,'T':2**40}[m[2]]
def iso(epoch):return datetime.fromtimestamp(epoch,timezone.utc).isoformat()
def mismatch(condition,label):
    if not condition:failures.append(label)
def ancestors(name):
    result=set()
    def walk(n):
        for x in units[n]['dependencies']:
            if x not in result:result.add(x);walk(x)
    walk(name);return result

failures=[]
index=json.loads((UPLOAD/'index.json').read_bytes())
export_manifest=json.loads((UPLOAD/'manifest.json').read_bytes())
exports={r['path']:r for r in export_manifest['files']}
workflow=io.load('resolve-workflow.json');declaration=io.load('protocol.json')
protocol=declaration['scientific_protocol'];units=protocol['units'];jobs=io.load('jobs.json')
by_stage={r['stage']:r for r in jobs};by_job={r['job_id']:r for r in jobs}
protocol_hash=digest(protocol);run_id=workflow['run_id'];prefix=f'runs/{run_id}/'
mismatch(sha((UPLOAD/'manifest.json').read_bytes())==index['manifest_sha256'],'uploaded optional manifest digest')
mismatch(digest(declaration)==workflow['protocol_sha256'],'coordinator declaration identity')
mismatch(digest(io.load('slurm-profile.json'))==workflow['slurm_profile_sha256'],'frozen hardware profile')
mismatch(protocol['schema']=='tdn.resolve/v1' and protocol['scientific_scope']=='DEVELOPMENT_ONLY_NO_CONFIRMATORY_CLAIMS' and protocol['confirmation']['enabled'] is False,'development-only scope')
mismatch(len(units)==28 and set(units)==set(by_stage) and len(by_job)==28,'exact unique 28 units/jobs')
mismatch(io.load('state/submission.json')['status']=='COMPLETED','submission completion')
# Verify original upload parts and concatenated archive independently.
archive=hashlib.sha256();part_details=[]
for part in index['parts']:
    p=UPLOAD/part['path'];h=hashlib.sha256();count=0
    with p.open('rb') as stream:
        while block:=stream.read(2**20):archive.update(block);h.update(block);count+=len(block)
    good=count==part['bytes'] and h.hexdigest()==part['sha256'];mismatch(good,'upload part '+p.name)
    part_details.append(dict(path=p.name,bytes=count,sha256=h.hexdigest(),verified=good))
mismatch(archive.hexdigest()==index['sha256'] and sum(p['bytes'] for p in part_details)==index['bytes'],'whole original archive')
# Verify the exact canonical subset's compressed storage hashes (all original bytes).
for name,e in io.index()['files'].items():
    raw=io.raw(name);mismatch(len(raw)==e['bytes'] and sha(raw)==e['sha256'],'canonical content '+name)
    record=exports[e['original_path']]
    mismatch(record['included'] and record['sha256']==e['sha256'] and record['bytes']==e['bytes'],'canonical/export identity '+name)
# Capture source identity before any reporting-only follow-up fix is applied.
h=hashlib.sha256()
for directory in ('tdn','reference','scripts','configs','tests'):
    for p in sorted((ROOT/directory).rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:h.update(p.relative_to(ROOT).as_posix().encode());h.update(p.read_bytes())
current_source=h.hexdigest()
for name in ('requirements.txt','pyproject.toml'):h.update(name.encode());h.update((ROOT/name).read_bytes())
current_controller_source=h.hexdigest()
current_commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip()
mismatch(current_source==workflow['source_tree_sha256'],'reviewed checkout scientific source before reporting fix')
mismatch(current_controller_source==workflow['source_sha256'],'reviewed checkout controller source before reporting fix')

manifests={n:io.load(n+'/science_manifest.json') for n in units}
seals={n:io.load(n+'/workflow-seal.json') for n in units}
scientific_refs=Counter();scientific_ref_bytes=Counter();unit_rows=[];gpu_cases=[];tests=[]
reference_software=None
submission_software=io.load('state/submission-software.json')
for order,job in enumerate(jobs):
    name=job['stage'];u=units[name];m=manifests[name];seal=seals[name]
    summary=io.load(name+'/summary.json');execution=io.load(name+'/execution.json');stage=io.load(name+'/stage.json')
    state=io.load('state/'+name+'.json');software=execution['software']
    mismatch(sha(io.raw('state/'+name+'-software.json'))==state['software_sha256'],'worker software hash '+name)
    mismatch(io.load('state/'+name+'-software.json')==submission_software,'software change '+name)
    mismatch(m['schema']=='tdn.resolve-science/v1' and m['stage']==name and m['kind']==u['kind'] and m['unit_sha256']==digest(u),'unit identity '+name)
    mismatch(io.raw(name+'/COMPLETED').decode().strip()==digest(m),'COMPLETED '+name)
    mismatch(m['protocol_sha256']==protocol_hash and io.load(name+'/protocol.json')==protocol,'stage protocol '+name)
    mismatch(m['source_tree_sha256']==workflow['source_tree_sha256'] and software['source_tree_sha256']==workflow['source_tree_sha256'],'stage source '+name)
    mismatch(software['git_commit']==current_commit and software['tracked_or_untracked_changes'] is False,'native git state '+name)
    mismatch(software['venv'] and software['executable']=='/home/rahel/TDN/.venv/bin/python','native venv '+name)
    if reference_software is None:reference_software=m['software']
    mismatch(m['software']==reference_software and summary['software']==reference_software,'scientific software '+name)
    expected_priors={p:sha(io.raw(p+'/science_manifest.json')) for p in u['dependencies']}
    expected_lineage={p:sha(io.raw(p+'/science_manifest.json')) for p in ancestors(name)}
    expected_execution_prior={p:dict(run_dir=workflow['run_dir']+'/'+p,workflow_seal_sha256=sha(io.raw(p+'/workflow-seal.json'))) for p in ancestors(name)}
    mismatch(m['prerequisites']==expected_priors and summary['prerequisites']==expected_priors,'immediate science prerequisites '+name)
    mismatch(m['lineage']==expected_lineage,'complete science lineage '+name)
    mismatch(execution['prerequisites']==expected_execution_prior,'wrapper ancestor lineage '+name)
    mismatch(summary['status']==stage['status']==state['status']=='COMPLETED' and state['exit_code']==0,'computational completion '+name)
    mismatch(summary['stage']==name and summary['device']==execution['device']==u['device']==m['device'],'device/stage scope '+name)
    mismatch(execution['slurm_job_id']==job['job_id']==state['slurm_job_id'] and execution['slurm_step_id']==state['slurm_step_id']=='0','exact allocation/step '+name)
    mismatch(execution['execution_mode']=='desktop-slurm' and execution['slurm_profile_sha256']==workflow['slurm_profile_sha256'] and execution['workflow_protocol_sha256']==workflow['protocol_sha256'],'native workflow binding '+name)
    expected_seal_files={'execution.json','protocol.json','stage.json','science_manifest.json'}
    mismatch(seal['schema']=='tdn.resolve/v1' and seal['protocol_sha256']==protocol_hash and set(seal['files'])==expected_seal_files,'execution seal inventory '+name)
    for file,expected in seal['files'].items():mismatch(sha(io.raw(name+'/'+file))==expected,'execution file hash '+name+'/'+file)
    for file,expected in m['artifacts'].items():
        exp=exports.get(prefix+name+'/'+file)
        mismatch(exp is not None and exp['sha256']==expected,'science artifact '+name+'/'+file)
        if exp:
            scope='included_verified_transport' if exp['included'] else 'omitted_declared_hash_only'
            scientific_refs[scope]+=1;scientific_ref_bytes[scope]+=exp['bytes']
    res=workflow['resources'][name]
    mismatch(res['cpus']==u['cpus'] and res['mem_gib']==u['mem_gib'] and res['walltime']==u['walltime'],'frozen resources '+name)
    mismatch(res['cpus']<=8 and res['mem_gib']<=48 and duration(res['walltime'])<=2100 and u['seconds']<=1800,'resource envelope '+name)
    expected_dependency=None
    if order:
        if name=='report':expected_dependency='afterany:'+':'.join(j['job_id'] for j in jobs[:order])
        else:
            previous=jobs[order-1]['job_id'];required=[by_stage[p]['job_id'] for p in u['dependencies']]
            groups=['afterok:'+':'.join(required)] if required else []
            if previous not in required:groups+=['afterany:'+previous]
            expected_dependency=','.join(groups) or None
    mismatch(job['dependency']==expected_dependency,'DAG versus resource-only serial dependency '+name)
    record=dict(stage=name,kind=u['kind'],job_id=job['job_id'],device=u['device'],cpus=res['cpus'],mem_gib=res['mem_gib'],
        walltime_limit_seconds=duration(res['walltime']),scientific_budget_seconds=u['seconds'],
        scientific_elapsed_seconds=summary['elapsed_seconds'],cli_elapsed_seconds=stage['elapsed_seconds'],
        experiment_count=summary['experiment_count'],correctness_failures=summary['correctness_failures'],
        verdict_counts=summary['verdict_counts'],memory=summary['memory'])
    xmlname='reporter-tests/'+name+'/tests.xml'
    if xmlname in io.names():
        tree=ET.fromstring(io.raw(xmlname));cases=tree.findall('.//testcase')
        outcomes=Counter('error' if c.find('error') is not None else 'failure' if c.find('failure') is not None else 'skipped' if c.find('skipped') is not None else 'passed' for c in cases)
        t=dict(stage=name,case_count=len(cases),outcomes=dict(outcomes),summed_case_seconds=sum(float(c.get('time',0)) for c in cases),suite_seconds=sum(float(s.get('time',0)) for s in tree.findall('.//testsuite')))
        mismatch(outcomes=={'passed':len(cases)},'JUnit all pass '+name)
        if u['device']=='cuda':
            expected=Counter(('tests.test_resolve_gpu',n) for n in protocol['gpu_test_cases'])
            observed=Counter((c.get('classname'),c.get('name')) for c in cases)
            mismatch(observed==expected and len(cases)==20,'mandatory unique GPU identities '+name)
            gpu_cases.extend(c.get('name') for c in cases)
        tests.append(t)
    else:mismatch(u['device']!='cuda' and name!='audit','missing required JUnit '+name)
    if u['device']=='cuda':
        pre=io.load(name+'-gpu-preflight.json')
        mismatch(pre['slurm_job_id']==job['job_id'] and pre['slurm_step_id']=='0' and pre['gpu_name']=='NVIDIA GeForce RTX 4090' and pre['cuda_visible_devices']=='0','native GPU preflight '+name)
        mismatch(pre['memory_policy']==dict(soft_cap_bytes=18*2**30,soft_fraction=.75,hard_fraction=.9),'VRAM budget '+name)
        mismatch(record['memory']['peak_allocated_bytes']<=record['memory']['soft_budget_bytes'] and record['memory']['peak_sampled_device_used_bytes']<=.9*pre['total_bytes'],'sampled memory boundary '+name)
    unit_rows.append(record)

# Exact allocation rows only for totals. Step rows supply sampled MaxRSS only.
accounting=io.load('state/scheduler-accounting.json');records=accounting['records']
alloc=[r for r in records if r['record_kind']=='allocation']
mismatch(accounting['status']=='RECORDED' and accounting['all_allocations_terminal'] and not accounting['missing_allocation_job_ids'],'terminal scheduler snapshot')
mismatch(len(alloc)==28 and {r['JobIDRaw'] for r in alloc}==set(by_job),'exact accounting jobs')
by_alloc={r['JobIDRaw']:r for r in alloc};by_unit={r['stage']:r for r in unit_rows}
for stage,row in by_unit.items():
    a=by_alloc[row['job_id']];tres=dict(x.split('=',1) for x in a['AllocTRES'].split(','))
    mismatch(a['State']=='COMPLETED' and a['ExitCode']=='0:0','scheduler completion '+stage)
    mismatch(int(tres['cpu'])==row['cpus'] and mem_bytes(tres['mem'])==row['mem_gib']*2**30 and int(tres.get('gres/gpu','0'))==int(row['device']=='cuda'),'actual scheduler resources '+stage)
    elapsed=int(a['ElapsedRaw']);cpu=duration(a['TotalCPU']);allocated_cpu=int(a['CPUTimeRAW'])
    mismatch(allocated_cpu==elapsed*row['cpus'],'allocated CPU accounting '+stage)
    rss=[mem_bytes(r['MaxRSS']) for r in records if r['allocation_job_id']==row['job_id'] and r['MaxRSS']]
    row.update(allocation_elapsed_seconds=elapsed,total_cpu_seconds=cpu,allocated_cpu_seconds=allocated_cpu,
        cpu_allocation_utilization_proxy=cpu/allocated_cpu if allocated_cpu else None,
        equivalent_busy_cpu_cores=cpu/elapsed if elapsed else None,
        max_reported_step_rss_bytes=max(rss) if rss else None,
        allocation_minus_science_seconds=elapsed-row['scientific_elapsed_seconds'])

# Tower is a projection, independently trace each stage report to canonical hashes.
tower_rows=[]
for name in io.names():
    if not name.startswith('tower/') or not name.endswith('/run.json'):continue
    base=name.rsplit('/',1)[0];run=io.load(name);terminal=io.load(base+'/summary.json');projection=io.load(base+'/outputs/frontier-results.json')
    stage=run['metadata']['phase'];r=by_unit[stage];res=workflow['resources'][stage]
    mismatch(run['job_id']==terminal['job_id']==r['job_id'] and run['state']==terminal['state']=='COMPLETED' and terminal['exit_code']==0,'Tower job terminal '+stage)
    mismatch(run['metadata']['workflow_id']==run_id and run['parameters']['source_sha256']==workflow['source_sha256'] and run['parameters']['protocol_sha256']==workflow['protocol_sha256'],'Tower coordinator binding '+stage)
    mismatch(run['resources']['cpus']==res['cpus'] and run['resources']['mem_bytes']==res['mem_gib']*2**30 and run['resources']['gpus']==int(r['device']=='cuda'),'Tower frozen resource identity '+stage)
    mismatch(projection['reporting_complete'] and projection['reporting_omission_count']==0 and terminal['results']['reporting_omission_count']==0,'Tower complete projection '+stage)
    mismatch(projection==terminal['results']['frontier'],'Tower embedded projection '+stage)
    mismatch(len(projection['sources'])==1,'Tower canonical source multiplicity '+stage)
    source=projection['sources'][0]
    mismatch(source['stage']==stage and source['program']=='resolve' and source['seal_status']=='VERIFIED_CANONICAL' and source['source_sha256']==exports[prefix+stage+'/rows.json']['sha256'],'Tower canonical row hash '+stage)
    mismatch(source['canonical_rows']==r['experiment_count']==projection['table_rows']['experiments'],'Tower experiment denominator '+stage)
    mismatch(source['scientific_scope']=='DEVELOPMENT_ONLY_NO_CONFIRMATORY_CLAIMS','Tower development scope '+stage)
    start=float(run['start']);end=float(run['end']);submitted=datetime.fromisoformat(by_stage[stage]['submitted_at']).timestamp()
    out=dict(stage=stage,job_id=r['job_id'],report_dir=base,start=start,end=end,start_utc=iso(start),end_utc=iso(end),
        worker_span_seconds=end-start,runtime_seconds=terminal['runtime_seconds'],submission_to_worker_start_seconds=start-submitted,
        table_rows=projection['table_rows'],reporting_complete=projection['reporting_complete'],figure_count=projection['figure_count'],
        observed_read_bytes=projection['observed_read_bytes'],observed_output_bytes=projection['observed_output_bytes'])
    tower_rows.append(out);r['worker_span_seconds']=end-start;r['tower_read_bytes']=out['observed_read_bytes'];r['tower_output_bytes']=out['observed_output_bytes']
mismatch(len(tower_rows)==28 and {r['stage'] for r in tower_rows}==set(units),'one Tower terminal per unit')
tower_rows.sort(key=lambda x:x['start']);gaps=[dict(before=a['stage'],after=b['stage'],seconds=b['start']-a['end']) for a,b in zip(tower_rows,tower_rows[1:])]
summary_by_kind={}
for kind in sorted({r['kind'] for r in unit_rows}):
    group=[r for r in unit_rows if r['kind']==kind]
    summary_by_kind[kind]=dict(jobs=len(group),gpu_jobs=sum(r['device']=='cuda' for r in group),
        **{k:sum(r[k] for r in group) for k in ('allocation_elapsed_seconds','scientific_elapsed_seconds','cli_elapsed_seconds','worker_span_seconds','total_cpu_seconds','allocated_cpu_seconds','experiment_count','correctness_failures','tower_read_bytes','tower_output_bytes')},
        max_reported_step_rss_bytes=max(r['max_reported_step_rss_bytes'] for r in group))
    summary_by_kind[kind]['cpu_allocation_utilization_proxy']=summary_by_kind[kind]['total_cpu_seconds']/summary_by_kind[kind]['allocated_cpu_seconds']
report_figures=[n for n in manifests['report']['artifacts'] if n.endswith(('.png','.pdf'))]
result=dict(schema='tdn.resolve-native-independent-provenance-cost-review/v1',run_id=run_id,
    verification_failures=failures,transport=dict(parts=part_details,archive_bytes=index['bytes'],archive_sha256=archive.hexdigest(),
        optional_uploaded_manifest_sha256=sha((UPLOAD/'manifest.json').read_bytes()),expected_manifest_sha256=index['manifest_sha256'],
        transport_verified_file_count=io.index()['full_transport_verified_files'],canonical_exact_files_rehashed=len(io.names()),
        export_statistics=index['statistics'],export_and_verification_seconds=index['export_and_verification_seconds']),
    provenance=dict(scientific_protocol_sha256=protocol_hash,coordinator_protocol_sha256=workflow['protocol_sha256'],
        native_commit=reference_software['git_commit'],native_source_tree_sha256=workflow['source_tree_sha256'],native_controller_source_sha256=workflow['source_sha256'],
        reviewed_checkout_commit_before_reporting_fix=current_commit,reviewed_checkout_source_tree_sha256_before_reporting_fix=current_source,
        reviewed_checkout_controller_source_sha256_before_reporting_fix=current_controller_source,
        native_scientific_software=reference_software,completed_units=len(unit_rows),science_artifact_references=dict(scientific_refs),science_artifact_reference_bytes=dict(scientific_ref_bytes),
        scope='Included hashes trace to independently verified transport. Omitted arrays, checkpoints and rendered figures have matching declarations only; they are not loaded, byte-verified or replayed here.'),
    tests=dict(suites=tests,gpu_allocations=len([r for r in tests if r['stage']!='audit']),gpu_case_executions=len(gpu_cases),distinct_gpu_cases=len(set(gpu_cases)),cpu_case_count=next(r['case_count'] for r in tests if r['stage']=='audit')),
    totals=dict(allocation_elapsed_seconds=sum(r['allocation_elapsed_seconds'] for r in unit_rows),
        gpu_allocation_seconds=sum(r['allocation_elapsed_seconds'] for r in unit_rows if r['device']=='cuda'),
        allocated_cpu_seconds=sum(r['allocated_cpu_seconds'] for r in unit_rows),total_cpu_seconds=sum(r['total_cpu_seconds'] for r in unit_rows),
        science_elapsed_seconds=sum(r['scientific_elapsed_seconds'] for r in unit_rows),cli_elapsed_seconds=sum(r['cli_elapsed_seconds'] for r in unit_rows),
        worker_spans_seconds=sum(r['worker_span_seconds'] for r in unit_rows),observed_campaign_span_seconds=tower_rows[-1]['end']-tower_rows[0]['start'],
        launch_and_between_worker_gaps_seconds=sum(r['seconds'] for r in gaps),
        max_reported_step_rss_bytes=max(r['max_reported_step_rss_bytes'] for r in unit_rows),
        peak_torch_allocated_bytes=max(r['memory'].get('peak_allocated_bytes',0) for r in unit_rows),
        peak_torch_reserved_bytes=max(r['memory'].get('peak_reserved_bytes',0) for r in unit_rows),
        peak_sampled_device_used_bytes=max(r['memory'].get('peak_sampled_device_used_bytes',0) for r in unit_rows),
        allocation_records=len(alloc),all_accounting_records=len(records),failure_allocations=sum(a['State']!='COMPLETED' for a in alloc),
        experiment_rows=sum(r['experiment_count'] for r in unit_rows),correctness_failures=sum(r['correctness_failures'] for r in unit_rows),
        max_science_budget_seconds=max(u['seconds'] for u in units.values()),max_allocation_limit_seconds=max(duration(u['walltime']) for u in units.values()),
        configured_allocation_ceiling_seconds=sum(duration(u['walltime']) for u in units.values()),
        configured_gpu_allocation_ceiling_seconds=sum(duration(u['walltime']) for u in units.values() if u['device']=='cuda'),
        energy_joules=None,monetary_cost=None,gpu_utilization=None),
    by_kind=summary_by_kind,units=unit_rows,tower=dict(reports=tower_rows,gaps=gaps,
        all_projection_complete=all(r['reporting_complete'] for r in tower_rows),
        experiment_rows=sum(r['table_rows']['experiments'] for r in tower_rows),
        check_rows=sum(r['table_rows']['checks'] for r in tower_rows),
        learning_observations=sum(r['table_rows']['learning_curves'] for r in tower_rows),
        figure_presentation_defect=dict(generated_figure_count=io.load('report/summary.json')['details']['figures'],
            declared_png_pdf_files=len(report_figures),native_tower_registered_figure_count=next(r['figure_count'] for r in tower_rows if r['stage']=='report'),
            paths=report_figures,root_cause='Canonical Resolve figures are direct children of report/, but the publisher only recognizes files whose parent directory is named figures.',
            implication='Scientific tables project fully. Native Tower graphical figure index is empty; raw artifacts and browser atlas remain separately retained.')),
    interpretation_limits=[
        'This campaign is development only; completed jobs and GOOD required-check scores are not scientific confirmation.',
        'Canonical subset verification and full transport verification cannot substantiate omitted binary checkpoint/array/figure bytes.',
        'TotalCPU / CPUTimeRAW measures allocated host CPU utilization proxy; it is not GPU utilization.',
        'MaxRSS is the maximum sampled reported step RSS; allocation rows omit it and subprocess peaks need not be additive or fully simultaneous.',
        'Tower first/last timestamps and adjacent worker gaps are operational proxies, not scheduler Start/End/Eligible queue accounting.',
        'Submission-to-worker-start includes planned dependencies and startup, so it cannot be interpreted as resource queue pressure.',
        'Sampled total GPU device use includes display/other processes; CUDA allocator maxima are the per-process observations.',
        'No energy meter or monetary rate is available. No GPU utilization or dollar/energy total can be inferred from memory or CPU accounting.',
        'These costs include preparation, diagnostic, fitting, optimization, readiness and reporting allocations, but exclude earlier campaigns and external unrelated jobs.',
    ])
(BASE/'provenance-costs.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:result[k] for k in ('verification_failures','totals','by_kind')},indent=2))
