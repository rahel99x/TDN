"""Publish the already computed balanced primary-endpoint summaries."""
import csv,json
from pathlib import Path
base=Path(__file__).resolve().parent
root=base.parents[1]
data=json.loads((base/'statistics-results.json').read_text())['primary_method_summary']
out=root/'results/advance-full-review';out.mkdir(parents=True,exist_ok=True)
labels={'two_basis':'Ours (fitted two-coefficient model)','quad2_conditioned':'Ours (state-conditioned model)','commutator_raw':'Ours (frozen numerical control)','commutator_cubic':'Ours (frozen numerical control)','quad2_full':'Analytic control','quad4_full':'Analytic control','analytic_quad_cubic':'Analytic control','df':'Theirs (classical control)','etdrk4':'Theirs (classical control)','fno_legacy':'Theirs (local hybrid FNO control)','fno_zero':'Theirs (local hybrid FNO control)','fno_scaled':'Theirs (local hybrid FNO control)','fno_mean':'Theirs (local hybrid FNO control)'}
columns=['track','family','method_group','train_count','final_time','schedule','independent_fields','training_seeds','endpoint_rows','rms_geometric_mean','rms_median','rms_p90_observed','rms_max_observed','sampled_spatial_max_error_worst_endpoint','mean_bias_mse_fraction_mean','endpoint_pass_count','endpoint_pass_fraction','joint_field_pass_count','joint_field_pass_fraction','rms_target','max_target','median_warm_complete_call_seconds','physical_interval_violation_endpoint_count','energy_increase_above_1e_minus12_endpoint_count','rms_below_1e_minus7_endpoint_count']
with (out/'all-model-metrics.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
 for r in data:
  w.writerow(dict(track=r['track'],family=r['family'],method_group=labels[r['family']],train_count=r['train_count'],final_time=r['final_time'],schedule='0.04+0.08' if r['final_time']==.12 else '0.08+0.16',independent_fields=r['fields'],training_seeds=3 if r['train_count'] else 0,endpoint_rows=r['rows'],rms_geometric_mean=r['gmean_rms'],rms_median=r['median_rms'],rms_p90_observed=r['p90_rms'],rms_max_observed=r['max_rms'],sampled_spatial_max_error_worst_endpoint=r['max_error'],mean_bias_mse_fraction_mean=r['mean_bias_mse_fraction'],endpoint_pass_count=r['eligible_rows'],endpoint_pass_fraction=r['eligible_rows']/r['rows'],joint_field_pass_count=r['all_queries_pass_fields'],joint_field_pass_fraction=r['all_queries_pass_fields']/r['fields'],rms_target=2e-5,max_target=2e-5,median_warm_complete_call_seconds=r['median_cost'],physical_interval_violation_endpoint_count=r['physical_violation_rows'],energy_increase_above_1e_minus12_endpoint_count=r['energy_increase_rows'],rms_below_1e_minus7_endpoint_count=r['below_1e7_rows']))
with (out/'all-model-metrics.csv').open() as f:
 check=list(csv.DictReader(f))
assert len(check)==len(data)==60
assert len({r['family'] for r in check})==13
for original,published in zip(data,check):
 assert float(published['rms_geometric_mean'])==original['gmean_rms']
 assert int(published['endpoint_rows'])==original['rows']
print('Published and round-trip verified 60 rows, 13 families.')
