import collections,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,'/workspace/Raven')
from raven import artifacts,projects
root=Path('.runtime/onboarding/tangent-native-bulk-observer-20261013-b/relocated')
output=root/'runs/native-bulk-encoding-observer/outputs';index=json.loads((output/'tangent_research.json').read_text())
counts=collections.Counter();checks=collections.Counter();statuses=collections.Counter()
for project in [root]+[output/page['path'] for page in index['artifact_projects']]:
    for record in projects.discover_project(str(project))['runs']:
        attempt=project/'runs'/record['inventory']['run_id']
        result=artifacts.validate_contract(json.loads((attempt/'outputs/contract.json').read_text()),str(attempt))
        assert result['status']=='valid' and result['valid']
        for entry in result['outputs']:
            counts['artifact_outputs']+=1;statuses[entry['status']]+=1
            for check in entry['checks']:
                assert check['status']=='pass',check
                checks[check['name']]+=1
receipt=Path('.runtime/onboarding/tangent-native-large-science-verification.json');p=json.loads(receipt.read_text())
p['counts'].update(counts);p['default_native_check_counts']=dict(checks);p['default_native_output_status_counts']=dict(statuses)
p['trusted_raven_commit']='c34f0892d7bddc685c5af10489ecb114fd6adab6'
native=Path('.runtime/native-compression-review-20261013-3e51c6f1/restored/runs/tangent-response-pilot-20261013T085856-8a44d33a-compression-20261013T094550-3e51c6f1')
state=native/'state/compression.json';worker=json.loads(state.read_text());run=native/'raven/runs'/(native.name+'-compression')/'run.json';inventory=json.loads(run.read_text())
p['original_worker_status']=worker['status'];p['original_worker_job_id']=worker['slurm_job_id'];p['original_worker_state_sha256']=hashlib.sha256(state.read_bytes()).hexdigest()
p['original_raven_inventory_state']=inventory['state'];p['original_raven_inventory_sha256']=hashlib.sha256(run.read_bytes()).hexdigest()
assert p['original_worker_status']=='FAILED' and p['original_worker_job_id']=='2891' and p['original_raven_inventory_state']=='RUNNING'
receipt.write_text(json.dumps(p,indent=2)+'\n')
print(json.dumps({'artifact_outputs':counts['artifact_outputs'],'checks':dict(checks),'statuses':dict(statuses),'worker_status':p['original_worker_status'],'gate':p['original_gate_b']}))
