"""Independent read-only reconstruction of native Resolve kernel evidence."""
from pathlib import Path
from collections import Counter, defaultdict
import json, math, statistics as st
import xml.etree.ElementTree as ET
import numpy as np
import review_io as io
BASE=Path(__file__).resolve().parent
stages=sorted(n.rsplit('/',1)[0] for n in io.names() if n.startswith('kernels-') and n.endswith('/kernel_rows.json'))
rows=[]; comparisons=[]; representations=[];profiles=[]; references=[]; summaries=[];timings=[];mismatches=[]
for stage in stages:
 d=io.load(stage+'/kernel_rows.json'); p=io.load(stage+'/profiling.json'); s=io.load(stage+'/summary.json')
 for name,target in [('rows',rows),('comparisons',comparisons),('representation_comparisons',representations),('references',references)]:
  target.extend(dict(stage=stage,**r) for r in d[name])
 profiles.extend(dict(stage=stage,**r) for r in p['profiles'])
 timings.extend(dict(stage=stage,**r) for r in p['paired_timing'])
 summaries.append(s)
key=lambda r:(r['stage'],r['parent_id'],r['schedule_steps'],r['method'])
lookup={key(r):r for r in rows}
for c in comparisons+representations:
 left=lookup[(c['stage'],c['parent_id'],c['schedule_steps'],c['comparator'])]
 right=lookup[(c['stage'],c['parent_id'],c['schedule_steps'],c['candidate'])]
 rat=[a/b for a,b in zip(left['raw_seconds'],right['raw_seconds'])]
 if rat != c['paired_round_speed_ratios']:mismatches.append(('ratios',c))
 eligible=all(r['reference_accepted'] and r['finite'] and r['upper_rms']<=2e-5 and r['upper_max']<=2e-5 for r in (left,right))
 if eligible != c['both_accuracy_eligible']:mismatches.append(('eligible',c))
 if c['accuracy_qualified_speed_ratio']!=(st.median(rat) if eligible else None):mismatches.append(('qualified',c))
for t in timings:
 for method, measured in t['methods'].items():
  raw=[round_['samples_seconds'][method] for round_ in t['rounds']]
  if raw!=measured['samples_seconds']:mismatches.append(('rawtimingrounds',t['parent_id'],method))
  row=lookup[(t['stage'],t['parent_id'],t['schedule_steps'],method)]
  if raw!=row['raw_seconds']:mismatches.append(('rowsrawtiming',t['parent_id'],method))
 if any(len(round_['order'])!=len(set(round_['order'])) or set(round_['order'])!=set(t['methods']) for round_ in t['rounds']):
  mismatches.append(('roundroster',t['parent_id']))
for r in rows:
 if st.median(r['raw_seconds']) != r['cost_seconds']:mismatches.append(('median',r))
 if len(r['raw_seconds'])!=20:mismatches.append(('count',r))
 if r['accuracy_status']!=('ELIGIBLE' if r['reference_accepted'] and r['finite'] and r['upper_rms']<=2e-5 and r['upper_max']<=2e-5 else 'ACCURACY_INFEASIBLE'):mismatches.append(('rowstatus',r))

def rng(x):return dict(min=min(x),median=st.median(x),max=max(x)) if x else None

def describe_comparison(cs):
 ratios=[st.median(c['paired_round_speed_ratios']) for c in cs]
 rounds=[x for c in cs for x in c['paired_round_speed_ratios']]
 eligible=[c for c in cs if c['both_accuracy_eligible']]
 return dict(conditions=len(cs),paired_rounds=len(rounds),same_schedule_eligible_conditions=len(eligible),
  median_speed_ratio=rng(ratios),geomean_condition_median_speed_ratio=math.exp(st.mean(map(math.log,ratios))),
  timing_rounds_candidate_faster=sum(x>1 for x in rounds),timing_rounds_over_20_percent=sum(x>=1.2 for x in rounds),
  conditions_candidate_faster=sum(x>1 for x in ratios),conditions_over_20_percent=sum(x>=1.2 for x in ratios),
  eligible_conditions_over_20_percent=sum(st.median(c['paired_round_speed_ratios'])>=1.2 for c in eligible),
  observed_round_ratio=rng(rounds),maximum_parity_error=max((c.get('maximum_parity_error',0) for c in cs),default=None))

groups=defaultdict(list)
for c in comparisons:groups[c['candidate']].append(c)
execution={k:describe_comparison(v) for k,v in groups.items()}
detail=defaultdict(list)
for c in comparisons:
 r=lookup[(c['stage'],c['parent_id'],c['schedule_steps'],c['candidate'])]
 detail[(r['track'],r['grid'],r['regime'],r['schedule_steps'],c['candidate'])].append(c)
execution_detail=[dict(track=k[0],grid=k[1],regime=k[2],schedule_steps=k[3],method=k[4],**describe_comparison(v)) for k,v in sorted(detail.items())]
rep_groups=defaultdict(list)
for c in representations:rep_groups[c['comparator']].append(c)
representation={k:{**describe_comparison(v),'control_over_candidate_rms_ratio':rng([c['control_over_candidate_rms_ratio'] for c in v]),
 'conditions_candidate_lower_rms':sum(c['control_over_candidate_rms_ratio']>1 for c in v)} for k,v in rep_groups.items()}
accuracy=[]
for scope in sorted({(r['track'],r['grid'],r['regime'],r['schedule_steps']) for r in rows}):
 rs=[r for r in rows if (r['track'],r['grid'],r['regime'],r['schedule_steps'])==scope]
 for method in sorted({r['method'] for r in rs}):
  ms=[r for r in rs if r['method']==method]
  accuracy.append(dict(track=scope[0],grid=scope[1],regime=scope[2],schedule_steps=scope[3],method=method,
      phase_count=len(ms),eligible=sum(r['accuracy_status']=='ELIGIBLE' for r in ms),
      rms=rng([r['error_rms'] for r in ms]),maximum=rng([r['error_max'] for r in ms]),
      latency_ms=rng([1000*r['cost_seconds'] for r in ms]),
      first_warm_ratio=rng([r['first_invocation_seconds']/r['cost_seconds'] for r in ms])))
frontiers=[]
for stage,parent in sorted({(r['stage'],r['parent_id']) for r in rows}):
 rs=[r for r in rows if r['stage']==stage and r['parent_id']==parent]
 good=[r for r in rs if r['accuracy_status']=='ELIGIBLE']
 best=min(good,key=lambda r:r['cost_seconds']) if good else None
 by_method={name:min([r for r in good if r['method']==name],key=lambda r:r['cost_seconds'],default=None) for name in {r['method'] for r in rs}}
 frontiers.append(dict(stage=stage,parent_id=parent,track=rs[0]['track'],grid=rs[0]['grid'],regime=rs[0]['regime'],
  best_method=best['method'] if best else None,best_schedule_steps=best['schedule_steps'] if best else None,
  best_milliseconds=best['cost_seconds']*1000 if best else None,
  choices={name:dict(schedule_steps=r['schedule_steps'],milliseconds=r['cost_seconds']*1000,rms=r['error_rms'],maximum=r['error_max']) if r else None for name,r in by_method.items()},
  scope='Exploratory reference-informed post-hoc 1/2-step menu; not deployment policy or validation-selected frontier'))
profile_summary=[]
for p in profiles:
 counts=defaultdict(int)
 for op in p['operators']:counts[op['operator']]+=op['calls']
 profile_summary.append(dict(stage=p['stage'],method=p['method'],**p['invocation_counts'],
     local_scalar_reads=counts['aten::_local_scalar_dense'],cudaStreamSynchronize=counts['cudaStreamSynchronize'],
     cudaMemcpyAsync=counts['cudaMemcpyAsync'],cudaLaunchKernel=counts['cudaLaunchKernel'],cuLaunchKernel=counts['cuLaunchKernel'],
     profile_seconds=p['profile_seconds'],warmup_seconds=p['warmup_seconds'],
     aten_self_cpu_microseconds=sum(r['self_cpu_microseconds'] for r in p['operators'] if r['operator'].startswith('aten::')),
     fft_dispatch_self_device_microseconds=sum(r['self_device_microseconds'] or 0 for r in p['operators'] if r['operator'].startswith('aten::_fft_')),
     scope='Single low-regime phase, one step, profiled separately; invocation counts are not arithmetic, operation categories overlap'))
accounting=dict(stage_elapsed_seconds=sum(s['elapsed_seconds'] for s in summaries),
 reference_seconds=sum(r['teacher_seconds'] for r in references),
 profiler_seconds=sum(p['profile_seconds'] for p in profiles),profiler_warmups_seconds=sum(p['warmup_seconds'] for p in profiles),
 measured_calls_seconds=sum(r['cost_seconds']*0+sum(r['raw_seconds']) for r in rows),
 first_call_seconds=sum(r['first_invocation_seconds'] for r in rows),
 warmup_seconds=sum(sum(m['warmup_seconds'] for m in t['methods'].values()) for t in timings),
 memory_probe_seconds=sum(t['memory_probe_seconds'] for t in timings),
 allocator_peak_allocated_bytes=max(s['memory']['peak_allocated_bytes'] for s in summaries),
 allocator_peak_reserved_bytes=max(s['memory']['peak_reserved_bytes'] for s in summaries),
 physical_peak_sampled_device_used_bytes=max(s['memory']['peak_sampled_device_used_bytes'] for s in summaries),
 max_incremental_memory_probe_bytes=max(r['memory']['incremental_peak_allocated_bytes'] for r in rows),
 min_soft_budget_bytes=min(s['memory']['soft_budget_bytes'] for s in summaries),
 monetary_cost=None,energy=None)
accounting['component_sum_seconds']=sum(accounting[k] for k in ['reference_seconds','profiler_seconds','profiler_warmups_seconds','measured_calls_seconds','first_call_seconds','warmup_seconds','memory_probe_seconds'])
junit=[]
for stage in stages:
 xml=ET.fromstring(io.raw('reporter-tests/'+stage+'/tests.xml'))
 cases=list(xml.iter('testcase'))
 junit.append(dict(stage=stage,cases=len(cases),failed=sum(c.find('failure') is not None or c.find('error') is not None for c in cases),skipped=sum(c.find('skipped') is not None for c in cases)))
symbols=[]
for n in (64,128):
 modes=[(n//2-3,1),(n//2-2,1)]
 for track in ('discrete','continuum'):
  values=[.12*.004*sum(4*n*n*math.sin(math.pi*k/n)**2 if track=='discrete' else 4*math.pi**2*k*k for k in mode) for mode in modes]
  symbols.append(dict(grid=n,track=track,modes=modes,h_diffusion_rate=values,h_reaction_rate=.36))
result=dict(schema='tdn.resolve-kernel-independent-review/v1',source_git=summaries[0]['software']['git_commit'],
 archive_verification_scope='Canonical bytes supplied by root verified extraction; independent reconstruction of all timing ratios and eligibility checks',
 stages=stages,denominators=dict(endpoint_rows=len(rows),methods=8,conditions=32,conditions_per_track_grid=8,
  declared_field_rows=len(references),phase_seed_clusters=4,phase_clusters_per_regime=2,
  interpretation='Low fields reuse physical modes across grids; high_pair modes depend on grid. Tracks/schedules reuse fields; 16 declared reference cases are not 16 independent phase draws. Two phase draws per regime vary translations of a fixed two-mode continuous shape; they do not provide new spectral/mean/amplitude/physics diversity.',
  execution_comparisons=len(comparisons),representation_comparisons=len(representations),timed_calls=sum(len(r['raw_seconds']) for r in rows),
  first_calls=len(rows),warmup_calls=len(rows),memory_probe_calls=len(rows),profile_calls=len(profiles),profile_warmup_calls=len(profiles)),
 validation=dict(reconstruction_mismatches=len(mismatches),reference_accepted=sum(r['accepted'] for r in references),
   max_reference_uncertainty_rms=max(r['uncertainty_rms'] for r in references),max_reference_uncertainty_max=max(r['uncertainty_max_bound'] for r in references),
   all_references_refined_steps=sorted({tuple(r['temporal_steps']) for r in references}),
   finite_rows=sum(r['finite'] for r in rows),eligible_rows=sum(r['accuracy_status']=='ELIGIBLE' for r in rows),
   interval_violation_rows=sum(bool(r['physical_interval_violations']) for r in rows),energy_increase_rows=sum(r['energy_increase']>1e-12 for r in rows),
   stage_correctness_failures=sum(s['correctness_failures'] for s in summaries)),
 native_cuda_tests=junit,actual_stiffness_symbols=symbols,execution=execution,execution_detail=execution_detail,representation=representation,accuracy=accuracy,
 posthoc_frontiers=frontiers,profiles=profile_summary,accounting=accounting,
 stage_summaries=[dict(stage=s['stage'],elapsed_seconds=s['elapsed_seconds'],memory=s['memory'],verdict_counts=s['verdict_counts']) for s in summaries],
 cold_scope='First invocation after coefficient caches cleared, but process/libraries/CUDA plans previously exercised by profiling; not genuine cold-start latency',
 target_scope='Same-grid FD/nodal or spectral Galerkin equation. No spatial refinement in this kernel panel; continuum label is equation track, not continuum convergence.',
 uncertainty_scope='Reference numerical differences and paired timing observations; no population or service-tail confidence intervals')
(BASE/'kernels-review.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:result[k] for k in ['denominators','validation','execution','representation','accounting']},indent=2))
print('GL4 details')
for r in execution_detail:
 if r['method']=='quad4_batched': print(r['track'],r['grid'],r['regime'],r['schedule_steps'],r['same_schedule_eligible_conditions'],r['median_speed_ratio'],r['observed_round_ratio'],r['timing_rounds_candidate_faster'])
print('posthoc best',Counter((r['track'],r['grid'],r['regime'],r['best_method'],r['best_schedule_steps']) for r in frontiers))
print('profile summary')
for p in profile_summary:
 if p['method'] in ['df','quad2_full','quad4_full','quad4_batched','quad3_midpoint_elided']:
  print({k:p[k] for k in ['stage','method','fft_dispatch','allocations','local_scalar_reads','cudaStreamSynchronize','cudaLaunchKernel','cuLaunchKernel','profile_seconds']})

# Human-readable synthesis generated from the same independently rebuilt tables.
def f(x,d=3):return f'{x:.{d}g}'
md=['# Native Resolve kernel review','',
'**Batched GL4 is a demonstrated execution improvement over the project’s sequential GL4 implementation. The cheaper GL3 representation does not establish an additional cost advantage on this RTX 4090.** This result strengthens the analytic control; it does not establish a learned-model advantage, superiority over published FNO, or an optimal numerical solver.','',
f'Source `{result["source_git"]}`. Four native CUDA stages (jobs 870–873), discrete and Galerkin tracks at 64² and 128². All stages completed with zero correctness failures. The four mandatory suites contain {sum(v["cases"] for v in junit)} CUDA case executions, {sum(v["failed"] for v in junit)} failures and {sum(v["skipped"] for v in junit)} skips.','',
'## Evidence and denominators','',
'- 8 methods × 2 schedules × 4 field cases per stage × 4 stages = **256 endpoint rows** and 5,120 warm complete-call observations. There are 96 same-rule execution comparisons and 96 GL3 representation comparisons; each uses 20 randomized paired rounds.',
'- The 16 reference cases are repeated uses of **four independent phase-seed clusters**, two low-mode and two high-pair. Tracks and schedules reuse those phases. Low-mode physical fields are shared across grids; high-pair frequencies change with grid size. Do not treat the 32 conditions, 256 endpoints or timing repetitions as independent fields.',
'- The four draws vary phase only: amplitudes, mean, variance, physics and two-mode support are fixed within each regime. Each two-mode pair has independent wavevectors, so its phase choices describe translations of one continuous field shape. Galerkin translation-equivariant error norms are therefore nearly redundant; discrete sub-grid phases may change aliasing but do not add new amplitude or spectral regimes. This is a narrow execution panel, not broad statistical generalization.',
'- All 16 references passed actual 128/256/512-step refinement, with maximum empirical RMS uncertainty '+f(result['validation']['max_reference_uncertainty_rms'])+' and maximum uncertainty '+f(result['validation']['max_reference_uncertainty_max'])+'. These are same-grid finite-equation teachers, not spatially converged continuum truth.',
'- All 256 endpoints are finite, with no sampled interval violation or endpoint energy increase. **198/256 meet both RMS and maximum targets of 2e−5.** Mathematical correctness and practical feasibility remain separate.',
'- Independently reconstructed all recorded speed ratios, eligibility decisions, medians and raw randomized-round alignments: **zero mismatches**.','',
'## Same-rule execution result','',
'Ratio = sequential control time / batched candidate time; higher is better. Ranges below are the eight condition-specific median paired ratios, not confidence intervals.','',
'| Track | Grid | GL4 speed ratio range | Accuracy-qualified conditions |',
'|---|---:|---:|---:|']
for track in ('discrete','continuum'):
 for n in (64,128):
  cs=[c for c in comparisons if c['candidate']=='quad4_batched' and c['track']==track and c['grid']==n]
  z=describe_comparison(cs)
  md.append(f'| {track} | {n}² | {z["median_speed_ratio"]["min"]:.3f}–{z["median_speed_ratio"]["max"]:.3f}× | {z["same_schedule_eligible_conditions"]}/8 |')
md += ['',
'GL4 was faster in **640/640 paired rounds**; 639/640 exceeded the declared 1.2× ratio. Every condition median exceeded 1.2×. Maximum endpoint disagreement between the two executions was **1.788×10−7**, well below the 3×10−6 parity threshold. This supports adopting the vectorized implementation as a frozen analytic baseline. It does not imply bitwise identity: easy cases at approximately 5×10−8 RMS are near the FP32 error floor.',
'',
'GL2 batching also clears the speed threshold in all 32 condition medians (1.233–1.568×), but only 20 conditions pass accuracy. Midpoint-elided GL3 improves over sequential GL3 by 1.346–1.817×, with 24 eligible conditions. These are execution improvements, not proof that GL2 or GL3 solves the harder workload accurately.','',
'## Representation, accuracy and practical choice','',
'| Method | Eligible endpoints / 32 | Meaning |',
'|---|---:|---|']
for method in ('df','quad2_batched','quad3_midpoint_elided','quad4_batched','analytic_quad_cubic'):
 count=sum(r['accuracy_status']=='ELIGIBLE' for r in rows if r['method']==method)
 meaning={'df':'Cheapest method when its accuracy suffices','quad2_batched':'Two active quadrature nodes','quad3_midpoint_elided':'Two active nodes plus an implicit zero-response midpoint','quad4_batched':'Four active nodes with shared transforms','analytic_quad_cubic':'Sequential GL4 quadratic plus nested cubic'}[method]
 md.append(f'| {method} | {count}/32 | {meaning} |')
md += ['',
'GL3 and batched GL4 have essentially the same runtime here: GL4/GL3 condition-median speed ratios range **0.981–1.042×**; none reaches the 1.2× threshold. GL3 has lower RMS on only 8/32 conditions, predominantly at the easy FP32 floor; GL4 can have **18.23× lower RMS**. GL3’s coverage is worse, so its smaller node count does not justify replacing batched GL4 on this hardware.',
'',
'At 64² high-pair, one GL3 step fails the maximum-error target on both tracks, while one GL4 step passes. Two GL3 steps pass but take about twice the time of one GL4 step. At discrete 128² high-pair, GL3 fails even in two steps; GL4 needs two steps and passes. At Galerkin 128² high-pair, even DF passes, and its prediction is identical in the reported one-step errors to the more expensive corrections. This is an easy absolute-tolerance workload despite its large dimensionless diffusion.',
'',
'An explicitly exploratory post-hoc 1/2-step menu selects DF in 10/16 field/track/grid cells (all eight low-mode cells and both Galerkin-128 high-pair cells), and batched GL4 in the other six. No GL3 winner emerges. These selections use test references and are **not a deployable policy or a validation-selected frontier**. DF receives the same ordinary step loop; the panel did not implement adjacent-heat fusion for two-step DF, so it cannot establish an optimized lower cost bound against all classical implementations.',
'',
'Nested cubic is much slower and generally offers no useful extra error reduction on the high-pair fields at these settings. Its 30/32 coverage equals GL4. That finding is local to this two-mode panel; it is not a general rejection of cubic structure. GL3’s 4.66–7.68× advantage over the expensive nested control must not conceal its lack of advantage over batched GL4.','',
'## What the math and profiler explain','',
'For centered state v, define G(s)=E(h−s)P[(E(s)v)²]. The correction is −r h q(h) Σᵢ wᵢ q(sᵢ)[G(sᵢ)−G(h/2)]. GL3’s midpoint term is exactly zero, leaving nodes 1/2±√15/10 with weights 5/18 each; the implicit midpoint has weight 4/9. The complete rule remains normalized and degree-five polynomial-exact. This is not a fifth-order full solver claim.',
'',
'Batching shares the centered-state FFT, evaluates node transports/products together, and combines the weighted transported spectra before the final inverse FFT. The same operator, precision, product, cutoff and physical formula are retained. It is an implementation change, not learned physics.',
'',
'| One-step profile | Sequential GL4 | Batched GL4 | DF base |',
'|---|---:|---:|---:|',
'| Discrete FFT dispatch calls | 24 | 8 | 4 |',
'| Galerkin FFT dispatch calls | 276 | 212 | 196 |',
'| Discrete `_local_scalar_dense` calls | 20 | 10 | 12 |',
'| Galerkin `_local_scalar_dense` calls | 43 | 13 | 15 |',
'| Discrete allocation-related operator calls | 106 | 53 | 28 |',
'| Galerkin allocation-related operator calls | 1028 | 788 | 700 |',
'',
'Counts are measured PyTorch invocations, not FLOPs or numbers of transformed scalar fields. Batched node transforms contain more fields per invocation. Allocation operators overlap; `scalar_reads` in the original profile adds wrapper `item` and backend `_local_scalar_dense`, so this table uses the latter rather than double-counting scalar reads. Copy-category counts also include wrappers and no-op conversion opportunities and must not be described as PCIe traffic.',
'',
'The Galerkin backbone performs four projected RK4 reaction subdivisions, hence sixteen dealiased quadratic RHS evaluations before the correction. Its 196 FFT dispatch calls remain in the 212-call optimized GL4 path. The batch optimization removes much of the *additional* correction dispatch and synchronization overhead; changing from four to two active nodes then does little to end-to-end latency. This makes projected reaction/backbone overhead and static metadata validation promising profiling targets, while preserving matched numerical accuracy and comparable optimization opportunities for controls.',
'',
'## Grid and stiffness interpretation','',
'| Grid | High-pair modes | Discrete hλ range | Galerkin hλ range |',
'|---|---|---:|---:|']
for n in (64,128):
 a=next(s for s in symbols if s['grid']==n and s['track']=='discrete');b=next(s for s in symbols if s['grid']==n and s['track']=='continuum')
 md.append(f'| {n}² | {a["modes"]} | {min(a["h_diffusion_rate"]):.3f}–{max(a["h_diffusion_rate"]):.3f} | {min(b["h_diffusion_rate"]):.3f}–{max(b["h_diffusion_rate"]):.3f} |')
md += ['',
'At h=.12, κ=.004 and r=3, hr=.36 on every kernel field. The diffusion symbols are the actual FD symbol 4κN²Σsin²(πk/N) and Galerkin κ(2π)²Σk². Two-step schedules halve these per-step numbers. The high-pair workload changes its physical frequencies with N, so its 64→128 comparison does not isolate resolution scaling or continuum generalization. Very strong damping moves the important interaction toward the start of the step; interior Gauss nodes may miss that narrow layer. The N128 Galerkin errors are smaller partly because the absolute nonlinear effect is smaller, not because generalization improved. A fixed 2e−5 tolerance can make the stiffest case operationally easier.',
'',
'## Runtime, memory and scope','',
f'The four experiment bodies consumed **{accounting["stage_elapsed_seconds"]:.2f} s**: {accounting["reference_seconds"]:.2f} s reference generation, {accounting["measured_calls_seconds"]:.2f} s warm timed calls, {accounting["first_call_seconds"]:.2f} s first invocations, {accounting["warmup_seconds"]:.2f} s measurement warmups, {accounting["memory_probe_seconds"]:.2f} s memory probes, and {accounting["profiler_seconds"]+accounting["profiler_warmups_seconds"]:.2f} s profiling and its warmups. The remaining approximately {accounting["stage_elapsed_seconds"]-accounting["component_sum_seconds"]:.2f} s covers other metric/report/orchestration work. These are experiment-body components; allocated Slurm time additionally includes preflight, required CUDA tests and launch overhead.',
'',
f'Maximum stage PyTorch allocated peak was **{accounting["allocator_peak_allocated_bytes"]/2**20:.3f} MiB**, reserved peak **{accounting["allocator_peak_reserved_bytes"]/2**20:.0f} MiB**, and maximum incremental isolated probe peak **{accounting["max_incremental_memory_probe_bytes"]/2**20:.3f} MiB**. Maximum sampled physical-device use was {accounting["physical_peak_sampled_device_used_bytes"]/2**30:.3f} GiB and includes desktop/other resident allocations. The minimum permitted soft budget was {accounting["min_soft_budget_bytes"]/2**30:.3f} GiB. No memory-pressure or allocation-limit failure occurred; more VRAM is not the next bottleneck.',
'',
'Timed complete calls start from resident device state and include the full scheduled solver, validation and cache use, but exclude caller-side input transfer, model construction and metric transfer. They are not complete service startup/transport costs. First-call values follow cache clearing but not a cold process: the profiler already exercised libraries, CUDA plans and the device. Twenty warm rounds support observed latency variation, not p95/p99 service guarantees. Profiling occurs separately and includes startup overhead (especially the first profile); do not rank methods using profiler durations. No power, energy, monetary rate, compiler optimization or cross-device throughput result was measured.',
'',
'## Next decisive experiment','',
'1. Preserve sequential GL4 as an exact attribution control and promote batched GL4 to the strong analytic baseline for subsequent learned comparisons. Expand parity tests to long repeated rollouts, non-square grids, and harder hλ/hr ranges before broader deployment.',
'2. Freeze a modest fresh panel with many independent phase/shape parents, independent diffusion/reaction sweeps, and both fixed-physical-frequency resolution tests and separately labeled grid-relative stress cases. Retain same-grid and spatially refined teacher tracks separately.',
'3. Test batched GL4/GL8 and a boundary-layer-aware or exponentially weighted interaction rule. At large hλ the missing early-time interaction is a more concrete mechanism than reducing GL4 to two active nodes. Keep a reference-informed oracle only as a diagnostic; any deployable rule must use state/physics features and independent validation.',
'4. Profile and optimize the Galerkin projected reaction/backbone and avoid repeated host scalar validation of immutable metadata, without changing the numerical equation. Give DF/nested/learned competitors the same caching, batching and fusion opportunities. Measure accurate one-, two- and several-step calls before any deployment-policy claim.',
'5. Deprioritize GL3 as a standalone RTX4090 latency claim: fewer nodes did not avoid enough launch/backbone work. Revisit only for larger spatial batches or different hardware where arithmetic or memory traffic, rather than host/dispatch overhead, becomes limiting.','',
'All tables derive from `kernels-review.json`; `kernels-review.py` reconstructs them from verified canonical artifacts using `review_io.py`. No source or historical scientific artifact was edited.']
(BASE/'kernels-review.md').write_text('\n'.join(md)+'\n')
