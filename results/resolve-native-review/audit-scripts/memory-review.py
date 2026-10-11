"""Independent review of the verified native RX evidence, without new science."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,math,statistics
import numpy as np
import review_io as io
BASE=Path(__file__).resolve().parent
raw=io.load('memory/memory_results.json'); summary=io.load('memory/summary.json')
original=io.load('memory/memory_summary.json'); selection=io.load('memory/memory_selection.json')
execution=io.load('memory/execution.json'); stages=io.load('memory/stage.json')
ledger=io.load('memory/rows.jsonl') if False else list(io.lines('memory/rows.jsonl'))
accounting=io.load('state/scheduler-accounting.json')
rows=raw['rows']; N=raw['settings']['rollout_steps']; methods=sorted({r['method']for r in rows})
all_index={(r['scenario'],r['parent_id'],r['noise_replicate'],r['method'],r['rollout_step']):r for r in rows}
assert len(all_index)==len(rows)
endpoints=[r for r in rows if r['rollout_step']==N]
end_index={(r['scenario'],r['parent_id'],r['noise_replicate'],r['method']):r for r in endpoints}
results=[]; errors=[]
original_index={(r['scenario'],r['method']):r['metrics'] for r in original}
rng=np.random.default_rng(96873301)
for scenario in sorted({r['scenario'] for r in rows}):
 for method in methods:
  selected=[r for r in endpoints if r['scenario']==scenario and r['method']==method]
  trajectory=[r for r in rows if r['scenario']==scenario and r['method']==method]
  ratios=[r['error_rms']/end_index[(scenario,r['parent_id'],r['noise_replicate'],'coarse_rk4')]['error_rms']for r in selected]
  field_logs=defaultdict(list)
  for r,ratio in zip(selected,ratios):field_logs[r['parent_id']].append(math.log(ratio))
  logs=np.array([statistics.mean(v)for p,v in sorted(field_logs.items())])
  boot=np.exp(rng.choice(logs,(5000,len(logs)),replace=True).mean(1))
  ratio_ci=np.quantile(boot,[.025,.975]).tolist()
  by_parent=defaultdict(list)
  for r in trajectory:by_parent[r['parent_id']].append(r)
  record=dict(scenario=scenario,method=method,independent_parents=len(field_logs),endpoint_queries=len(selected),
   native_median_paired_ratio=float(np.median(ratios)),parent_clustered_geometric_ratio=float(np.exp(logs.mean())),
   descriptive_95_parent_bootstrap=ratio_ci,parents_improved=int(sum(v<0 for v in logs)),parents_worse=int(sum(v>0 for v in logs)),
   endpoint_median_rms=statistics.median(r['error_rms']for r in selected),endpoint_worst_max=max(r['error_max']for r in selected),
   endpoint_median_mean_error=statistics.median(r['mean_error']for r in selected),
   rollout_worst_rms=max(r['error_rms']for r in trajectory),rollout_worst_max=max(r['error_max']for r in trajectory),
   rollout_min_state=min(r['minimum_state']for r in trajectory),rollout_max_state=max(r['maximum_state']for r in trajectory),
   interval_violation=max(r['observed_interval_violation']for r in trajectory),
   violated_parent_count=sum(any(r['observed_interval_violation']>0 for r in rr)for rr in by_parent.values()),
   violated_trajectory_count=len({(r['parent_id'],r['noise_replicate'])for r in trajectory if r['observed_interval_violation']>0}),
   nonfinite_rows=sum(not r['finite']for r in trajectory),
   endpoint_complete_seconds_median=statistics.median(r['complete_seconds']for r in selected),
   endpoint_inference_seconds_median=statistics.median(r['cumulative_inference_seconds']for r in selected),
   startup_seconds_median=statistics.median(r['startup_seconds']for r in selected))
  for native,derived in [('endpoint_ratio_to_coarse','native_median_paired_ratio'),('endpoint_rms_median','endpoint_median_rms'),
        ('endpoint_max_worst','endpoint_worst_max'),('endpoint_mean_error_median','endpoint_median_mean_error'),
        ('rollout_rms_worst','rollout_worst_rms'),('rollout_max_worst','rollout_worst_max'),
        ('observed_interval_violation','interval_violation'),('complete_seconds_median','endpoint_complete_seconds_median')]:
   if not math.isclose(original_index[(scenario,method)][native],record[derived],rel_tol=1e-13,abs_tol=1e-18):errors.append([scenario,method,native])
  results.append(record)
# Validate paired zero-gain state identity, including noisy histories.
zero_gain_identity=all(r['state']==all_index[(r['scenario'],r['parent_id'],r['noise_replicate'],'coarse_rk4',r['rollout_step'])]['state']for r in rows if r['method']=='validated_damped_secant')
# Fourier physical maximum diagnostics are sampled, and the reference stays interior on those points.
modes=np.arange(-raw['settings']['cutoff'],raw['settings']['cutoff']+1)
decode=np.exp(1j*np.outer(np.arange(257)*2*np.pi/257,modes))
ref_min,ref_max=math.inf,-math.inf
for observation in raw['startup_observations']:
 coefficients=np.array(observation['targets']);coefficients=coefficients[...,0]+1j*coefficients[...,1]
 physical=(coefficients@decode.T).real
 ref_min=min(ref_min,float(physical.min()));ref_max=max(ref_max,float(physical.max()))
# Explicit derivative-noise interpretation uses the effective input-coordinate map, not normalized network weights.
models={}
for method,model in raw['models'].items():
 effective=np.array(model['weights'])/np.array(model['scale'])[:,None]
 memory_block=effective[-len(modes):]
 models[method]=dict(ridge=model['ridge'],coefficient_count=int(effective.size),feature_normalizers=len(model['scale']),
  secant_diagonal=memory_block.diagonal().tolist(),secant_singular_values=np.linalg.svd(memory_block,compute_uv=False).tolist(),
  secant_matrix=memory_block.tolist())
refs=[v for v in raw['teacher_records']if v['split']=='development_test']
cost={**raw['offline_costs'],'fit_and_selection_seconds':raw['fit_and_selection_seconds'],
 'science_stage_seconds':summary['elapsed_seconds'],'worker_experiment_seconds':stages['elapsed_seconds'],
 'accounting_rows':[r for r in accounting['records']if r['allocation_job_id']==execution['slurm_job_id']],
 'observed_sum_inference_seconds':sum(r['inference_seconds']for r in rows),
 'cost_warning':'Per-method timing is descriptive fixed-order CPU measurement, not a randomized paired deployment benchmark. Repeated startup/teacher costs are not additive.'}
reference_ratios=[r['reference_uncertainty']/r['error_rms']for r in rows if r['error_rms']>0]
source_names=['memory/memory_results.json','memory/memory_summary.json','memory/memory_selection.json','memory/summary.json',
'memory/stage.json','memory/execution.json','memory/rows.jsonl','state/scheduler-accounting.json']
review=dict(schema='tdn.resolve-native-memory-review/v1',scope='Native full CPU development result; not the earlier cloud probe or untouched confirmation',
 source_tree_sha256=summary['source_tree_sha256'],git_commit=summary['software']['git_commit'],slurm_job_id=execution['slurm_job_id'],
 source_files={name:dict(sha256=io.index()['files'][name]['sha256'],bytes=io.index()['files'][name]['bytes'])for name in source_names},
 settings=raw['settings'],parent_ids=raw['parent_ids'],row_count=len(rows),endpoint_queries=len(endpoints),independent_test_parents=len(raw['parent_ids']['development_test']),
 scenario_count=len({r['scenario']for r in rows}),train_rows=len(raw['training_rows']),summary_reconstruction_mismatches=errors,
 split_disjoint=all(set(raw['parent_ids'][a]).isdisjoint(raw['parent_ids'][b])for a,b in [('train','validation'),('train','development_test'),('validation','development_test')]),
 selected_damping=raw['selected_damping'],selected_zero_gain_exactly_coarse_state=zero_gain_identity,
 selection_sha256_matches=hashlib.sha256(io.raw('memory/memory_selection.json')).hexdigest()==raw['selection_sha256'],
 validation_candidates=raw['selection'],models=models,reference=dict(trajectories=len(refs),all_below_threshold=all(v['reference_uncertainty']<=raw['settings']['reference_tolerance']for v in refs),
  uncertainty_min=min(v['reference_uncertainty']for v in refs),uncertainty_max=max(v['reference_uncertainty']for v in refs),
  maximum_uncertainty_over_measured_rms=max(reference_ratios),sampled_reference_minimum=ref_min,sampled_reference_maximum=ref_max,
  crosscheck='DOP853 tightening plus independent 512-step RK4 endpoint; finite-mode temporal estimate, no continuum certificate'),
 checks=Counter(c['verdict']for r in ledger for c in r['checks']),stage_verdicts=summary['verdict_counts'],all_rows_finite=all(r['finite']for r in rows),
 costs=cost,results=results,statistical_scope='Per-parent geometric mean across two noise realizations, then 5000 parent-resampled bootstrap draws. Descriptive unadjusted 95% intervals; development only; exact duplicate clean realizations do not add independent fields.')
(BASE/'memory-review.json').write_text(json.dumps(review,indent=2,allow_nan=False)+'\n')
lines=['# Native Resolve memory review','',
'**The protected memory path does not justify promotion.** The validation-selected deterministic answer is zero memory; the noise-aware fitted model has worse median paired RMS than coarse RK4 in every reported scenario. Clean-only fitting has a real clean-history benefit, but amplifies small observation errors and fails a sampled boundedness check in the shifted, noisier case.','',
'## Scope and verification','',
f"Native job {execution['slurm_job_id']} used Python {summary['software']['python']} in the project venv on CPU. Source `{summary['source_tree_sha256']}` / commit `{summary['software']['git_commit']}`. The result is independent of the earlier cloud-sized probe: phase seed base {raw['settings']['seed']}, 48 train parents, 16 validation parents, 16 development-test parents. All splits are disjoint. Full means coverage, not confirmation.", '',
'The 30,720 rows are 16 fields × two noise draws × three noise levels × two reaction rates × five methods × 32 steps. The zero-noise draws repeat identical states. Independent evidence remains 16 test fields, not 30,720 samples. The six scopes reuse those fields and are paired.', '',
'Each method receives two noisy coarse observations at t=0.08 and t=0.10, then predicts without new measurements for 32 steps of h=0.02, to t=0.74. The fine/coarse Galerkin cutoffs are 8/2 (17/5 real degrees of freedom), on a periodic one-dimensional domain. κ=0.02 and r=0.7 or 2.1 are tested. Thus hr=0.014/0.042 and the predicted interval spans 0.448/1.344 reaction times: this is not a long-time or stiff spatial stress guarantee.', '',
'Noise σ is per independent real Fourier coordinate, not physical pointwise RMS. At cutoff 2 its root-mean-square over noise and physical space is 3σ. The two history perturbations are independent; methods and reaction scenarios share the same standardized draws. Noisy histories are not refreshed from future truth.', '',
f"All 30 native summary cells were independently reconstructed with {len(errors)} mismatches. The selection-file SHA-256 matches. Zero-gain trajectories are bitwise identical to coarse RK4. All {len(refs)} reference trajectories pass the declared threshold; uncertainty estimates range {review['reference']['uncertainty_min']:.3g}–{review['reference']['uncertainty_max']:.3g}, versus 1e-9 allowed. The largest uncertainty/RMS ratio in any recorded point is {max(reference_ratios):.3g}. References use a tighter DOP853 solve plus independent 512-step RK4 endpoint; this addresses temporal error in the finite ODE, not continuum closure certification.", '',
'## Endpoint comparisons','',
'Values below are the native median of paired candidate/coarse-RK4 RMS ratios across parent/noise realizations; **lower is better, 1 is a tie**. They are not ratios of unpaired median errors. The JSON also provides independent-parent geometric ratios and descriptive bootstrap intervals.', '',
'| Reaction | Coordinate noise σ | Plain secant | Clean fitted | Noise-aware fitted |',
'|---|---:|---:|---:|---:|']
lookup={(x['scenario'],x['method']):x for x in results}
for scenario in sorted({r['scenario']for r in rows}):
 rate,noise=scenario.split('/'); vals=[lookup[(scenario,m)]['native_median_paired_ratio']for m in ['plain_secant','clean_fitted_memory','noise_aware_fitted_memory']]
 lines.append(f"| {rate} | {noise.split('-')[1]} | {vals[0]:.3f} | {vals[1]:.3f} | {vals[2]:.3f} |")
lines+=['',
'Clean fitted memory reduces paired RMS by 87.2% at the base rate and 76.4% after tripling reaction, but not under noise. At σ=1e-4 its RMS is 18.8–29.1× worse than coarse; at σ=1e-3 it is 39.2–45.6× worse. Noise-aware fitting reduces much of this amplification, yet still loses to coarse by 1.195–12.057× in native median paired RMS. Its improvement over a broken noisy-history model is not a solver advantage.', '',
'This is not uniform worsening on every parent. At base rate and σ=1e-3, the parent-clustered geometric noise-aware/coarse ratio is 1.109, with a descriptive unadjusted 95% bootstrap interval [0.950, 1.297]; six of 16 parents improve. At 3r and σ=1e-3 the ratio is 1.331 [1.116, 1.587], with four of 16 improving. These small development samples do not establish a benefit or a universal theorem of failure.', '',
'Worst sampled spatial error agrees with the clean-history benefit: clean fit reaches 1.08e-4/4.19e-4 at r/3r, versus coarse 1.51e-3/3.83e-3. At σ=1e-3 it instead reaches 0.383/0.284, while coarse remains at 0.00875/0.00622. Noise-aware worst errors are 0.00993/0.00774 there. Mean errors likewise expose damage: at the base rate and σ=1e-3, clean fit has median mean error 0.0352 versus coarse 0.000693.', '',
'## Boundedness and what the score misses','',
'All states remain finite and all reference checks pass. That does **not** establish numerical stability. The required scoring checks currently test finiteness, but interval violations are only reported metrics, so zero correctness failures can coexist with severe overshoot.', '',
'| Scenario | Method | Parents violating [0,1] / 16 | Noise trajectories / 32 | Largest sampled violation |',
'|---|---|---:|---:|---:|']
for r in results:
 if r['violated_parent_count']:lines.append(f"| {r['scenario']} | {r['method']} | {r['violated_parent_count']} | {r['violated_trajectory_count']} | {r['interval_violation']:.6f} |")
lines += ['',f"The scored projected references remain within [{ref_min:.6f}, {ref_max:.6f}] on the same 257 sampling points. These are observed physical-interval violations, not a general maximum-principle theorem for finite Galerkin truncations; unsampled maxima may be larger.",'',
'## Mathematical interpretation','',
'With S_h the coarse RK4 map, plain memory advances x[n+1] = S_h(x[n]) + x[n] − S_h(x[n−1]). In the locally neutral limit S_h≈I this becomes x[n+1]=2x[n]−x[n−1], with a repeated unit characteristic root. Startup noise propagates as η[n]=(n+1)η[0]−nη[−1]; its standard deviation grows as σ√((n+1)²+n²). At n=32 the factor is 45.97. This simple limit explains why a method with plausible clean closure can amplify history noise dramatically; it is a mechanistic approximation, not a proof for the nonlinear problem.', '',
'Fixed damping α changes the neutral-limit roots to 1 and α. The validation objective is the average log of whole-trajectory RMS relative to coarse across equal-weight noise scenarios. Every tested nonzero gain loses: even the best nonzero option, α=0.25 and decay=4, has log ratio 0.1723 (about 1.188×). The selected α=0 has exactly coarse states. This selection concerns a mixed clean/noisy workload; it does not say that memory is useless when observations are certified clean.', '',
'Effective clean-fit secant coefficients have diagonal 0.9844–0.9879, singular values 0.9821–0.9925: they remain close to persistent secant memory. The noise-aware fit instead learns a negative secant block, diagonal −0.5002 to −0.4314. A secant-only, small-step Gaussian calculation predicts −1/2: the noisy derivative feature contains (η[0]−η[−1])/h, while its supervised correction target contains −η[0]/h. The covariance ratio is −σ²/(2σ²). The full model also uses state and squared-state features, so this is an explanatory limit, not a derivation of every fitted coefficient.', '',
'This reveals an important learning problem: noise augmentation teaches a one-step denoising response and then reuses it autonomously as a dynamical closure. New measurement noise does not arrive after startup, while the regression was trained on noisy teacher snapshots at every example. Both fits are linear-in-coefficients regressions over nonlinear state/secant features, with 80 coefficients plus 16 feature normalizers; increasing network size would not repair this information/target mismatch by itself. The noisy fit also has a different validation objective from the historical clean recipe, so their differences cannot be attributed solely to augmentation.', '',
'## Costs','',
f"The numerical stage took {summary['elapsed_seconds']:.3f}s; the experiment worker took {stages['elapsed_seconds']:.3f}s. Unique offline teacher/reference work was {cost['unique_teacher_and_reference_seconds']:.3f}s, startup solves {cost['unique_startup_acquisition_seconds']:.3f}s, and fitting/selection {cost['all_fit_and_selection_seconds']:.3f}s (mostly the damping grid: {cost['fit_and_selection_seconds']['validated_damped_secant']:.3f}s). Repeated per-method rows reuse startup/reference costs and must not be summed as independent solves.", '',
'At base rate with clean inputs, observed median complete 32-step prediction cost including startup is 1.209 ms for coarse, 2.071 ms for plain secant, 2.093 ms for selected zero-gain damping, and approximately 2.75 ms for either fitted method. Zero-gain damping still computes the unused secant, so it reproduces coarse dynamics with avoidable overhead. A future implementation should dispatch α=0 directly to coarse. These fixed-order CPU timing observations are descriptive, not randomized deployment speed estimates; no accuracy-qualified cost claim or monetary claim follows.', '',
'## Decision','',
'1. Stop promotion of the present noise-aware affine closure and stop treating noisy secant persistence as a deployment candidate. Preserve the useful negative evidence.',
'2. Keep the clean-only fit as a narrow control when clean causal state history is genuinely available, and require it to beat simple lightly damped secant on an explicitly clean validation workload before adding capacity.',
'3. Any next protected prototype should estimate state/noise jointly: test a deterministic two-observation filter or causal observer, then add learned closure only if the filter leaves a resolved residual. Supply known noise level or estimate uncertainty causally; gate memory when it is below its observation-noise floor. Compare against direct coarse and a deterministic observer at equal information and complete cost.',
'4. Use fresh development parents for this revision, train on autonomous rollouts or a correctly modeled observation process, and keep clean/noisy/rate-shift objectives separate. Advance only with at least 20% error reduction over the best validation-selected deterministic control, no reference-resolved maximum/mean regression, no new interval violations, and a plausible complete-cost margin. Require new locked confirmation before a comparative claim.',
'5. Make boundedness, failure rate, mean drift and long-horizon growth explicit required acceptance checks. Verify over longer physical horizons and independent noise structures, not just more steps at a single small h.', '',
'No new experiments or source changes were made in this review. All native outcomes, including failed utility checks and overshoots, remain intact.']
alloc=[r for r in cost['accounting_rows']if r['record_kind']=='allocation']
if alloc:lines.insert(lines.index('## Costs')+2,f"Scheduler allocation: {alloc[0]['elapsed_seconds']} s, {alloc[0]['AllocTRES']}, state {alloc[0]['State']}; do not sum allocation and step rows.")
(BASE/'memory-review.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'summary_reconstruction_mismatches':errors,'zero_gain_exact':zero_gain_identity,'references':review['reference'],'violations':[r for r in results if r['violated_parent_count']],'allocation':alloc},indent=2))
