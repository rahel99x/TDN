import json
from pathlib import Path
from tdn.analysis.response.diagnostic_metrics import stable_id
receipt=Path('.runtime/onboarding/tangent-native-large-science-verification.json')
p=json.loads(receipt.read_text())
name='tangent-response-pilot-20261013T085856-8a44d33a-compression-20261013T094550-3e51c6f1'
old='native-bulk-encoding-observer'
a=len(stable_id(name+'-data-0000'))-len(stable_id(old+'-data-0000'))
# Two CSV/JSON source_path values per compiled record; small catalog and section
# rows also grow by the campaign name. Reserve another8MiB for headers/metadata.
extra=2*p['compiled_records']*max(a,0)+(8<<20)
p['native_campaign_long_name_additional_bytes_upper']=extra
p['projected_native_campaign_with_original_export_bytes_upper']=p['projected_with_whole_original_export_bytes']+extra
p['native_campaign_naming_scope']='Conservative rebased-citation projection for the actual long native campaign name; raw native source bytes remain verified, unchanged. Adds8MiB for headers, catalog and page metadata.'
assert p['projected_native_campaign_with_original_export_bytes_upper']<p['unchanged_transport_bound_bytes']
receipt.write_text(json.dumps(p,indent=2)+'\n')
print(json.dumps({'extra_bytes_upper':extra,'native_bytes_upper':p['projected_native_campaign_with_original_export_bytes_upper'],'remaining_bytes':p['unchanged_transport_bound_bytes']-p['projected_native_campaign_with_original_export_bytes_upper']}))
