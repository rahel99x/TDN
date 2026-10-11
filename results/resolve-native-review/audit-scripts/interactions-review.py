"""Read-only audit of exact verified R2 evidence; no new science is executed."""
from pathlib import Path
import collections
import json
import math
import statistics

import review_io as io

BASE = Path(__file__).resolve().parent
STAGES = sorted(name.rsplit('/', 1)[0] for name in io.names() if name.endswith('/interaction_cases.json'))
METHODS = ['df', 'etdrk4', 'gl2', 'gl4', 'gl2_raw_cubic', 'gl2_filtered_cubic', 'gl4_nested_cubic']
CASES, ROWS, SUMMARIES = [], [], []
for stage in STAGES:
    CASES += [dict(c, stage=stage) for c in io.load(stage + '/interaction_cases.json')['cases']]
    ROWS += [dict(r, stage=stage) for r in io.load(stage + '/interaction_rows.json')['rows']]
    SUMMARIES.append(dict(io.load(stage + '/summary.json'), stage=stage))
C = {(c['stage'], c['case_id']): c for c in CASES}
R = {(r['stage'], r['case_id'], r['method']): r for r in ROWS}
assert len(C) == len(CASES) and len(R) == len(ROWS)
assert all(len([r for r in ROWS if r['stage'] == c['stage'] and r['case_id'] == c['case_id']]) == 7 for c in CASES)
TARGET = io.load(STAGES[0] + '/protocol.json')['primary_target']


def stats(xs):
    xs = [float(x) for x in xs if x is not None and math.isfinite(x)]
    if not xs:
        return dict(count=0, minimum=None, median=None, maximum=None, geometric_mean=None)
    return dict(count=len(xs), minimum=min(xs), median=statistics.median(xs), maximum=max(xs),
                geometric_mean=math.exp(statistics.mean(map(math.log, xs))) if min(xs) > 0 else None)


def compare(row, control='gl4', norm='rms'):
    peer = R[row['stage'], row['case_id'], control]
    a = peer['same_grid_metrics']['error_' + norm]
    b = row['same_grid_metrics']['error_' + norm]
    u = row['same_grid_metrics']['reference_uncertainty_' + ('max' if norm == 'max' else 'rms')]
    valid = row['same_grid_metrics']['reference_accepted'] and peer['same_grid_metrics']['reference_accepted']
    result = 'unavailable' if not valid else 'win' if a - u > b + u else 'loss' if b - u > a + u else 'unresolved'
    return dict(control_over_candidate=a / b if b > 0 else None, outcome=result)


def method_table(rows):
    result = {}
    for method in METHODS:
        selected = [r for r in rows if r['method'] == method]
        result[method] = dict(n=len(selected),
            rms=stats(r['same_grid_metrics']['error_rms'] for r in selected),
            maximum_error=stats(r['same_grid_metrics']['error_max'] for r in selected),
            mean_error=stats(r['same_grid_metrics']['mean_error'] for r in selected),
            upper_joint_target_passes=sum(r['same_grid_metrics']['upper_rms'] <= TARGET and r['same_grid_metrics']['upper_max'] <= TARGET for r in selected),
            upper_rms_target_passes=sum(r['same_grid_metrics']['upper_rms'] <= TARGET for r in selected),
            vs_gl4_rms_counts=dict(collections.Counter(compare(r)['outcome'] for r in selected)),
            vs_gl4_max_counts=dict(collections.Counter(compare(r,norm='max')['outcome'] for r in selected)),
            vs_gl4_rms_ratio=stats(compare(r)['control_over_candidate'] for r in selected),
            interval_violations=sum(r['same_grid_metrics']['physical_interval_violations'] for r in selected),
            energy_increases_over_1e_12=sum(r['same_grid_metrics']['energy_increase'] > 1e-12 for r in selected))
    return result


alignment_rows = []
for case in CASES:
    g = case['response_geometry']
    base_norm = R[case['stage'], case['case_id'], 'df']['same_grid_metrics']['error_rms']
    row = {k:case[k] for k in ('stage','case_id','track','grid','diffusion_number','reaction_number')}
    row.update(regime=case['field']['regime'], phase=case['field']['phase'],
               filtered_vs_nested_cosine=g['normalized_gram'][3][4],
               raw_vs_nested_cosine=g['normalized_gram'][2][4])
    for q, index, method in [('q2',0,'gl2'), ('q4',1,'gl4')]:
        error = R[case['stage'],case['case_id'],method]['same_grid_metrics']['error_rms']
        for basis, bi in [('filtered_c3',3),('nested_c3',4)]:
            cos_base = g['pair_oracles'][q + ':' + basis]['residual_basis_cosines'][1]
            row['after_' + q + '_' + basis + '_cosine'] = (cos_base * base_norm - g['normalized_gram'][index][bi] * g['basis_rms'][q]) / error
            row['after_' + q + '_defect_over_uncertainty'] = error / R[case['stage'],case['case_id'],method]['same_grid_metrics']['reference_uncertainty_rms']
            row[basis + '_rms'] = g['basis_rms'][basis]
    assert all(abs(v) <= 1.00001 for k,v in row.items() if k.endswith('_cosine'))
    alignment_rows.append(row)

unresolved = []
for case in CASES:
    if case['all_references_accepted']:
        continue
    fine = case['references']['projected_fine']
    unresolved.append(dict(stage=case['stage'],case_id=case['case_id'],regime=case['field']['regime'],phase=case['field']['phase'],
        diffusion_number=case['diffusion_number'],reaction_number=case['reaction_number'],
        accepted_by_target={k:case['references'][k]['accepted'] for k in ('same_grid','galerkin_grid','projected_fine')},
        fine_uncertainty_rms=fine['uncertainty_rms'],fine_uncertainty_max=fine['uncertainty_max_bound'],
        projected_spatial_difference_max=fine['projected_spatial_difference_max'],
        both_fine_temporal_accepted=all(t['accepted'] for t in fine['temporal_refinements'])))

# Decomposition is summarized once per case, not once per method.
decomposition = {}
for track in ('discrete','continuum'):
    for regime in ('overlapping_triads','aligned_harmonics','grid_relative'):
        selected = [c for c in CASES if c['track']==track and c['field']['regime']==regime and c['all_references_accepted']]
        output = dict(cases=len(selected), unresolved_excluded=sum(c['track']==track and c['field']['regime']==regime and not c['all_references_accepted'] for c in CASES))
        for component, reference in [('discretization','same_grid'),('closure','projected_fine')]:
            values=[]; separated=0
            for case in selected:
                val=R[case['stage'],case['case_id'],'df']['error_decomposition']['component_rms'][component]
                uncertainty=case['references']['galerkin_grid']['uncertainty_rms']+case['references'][reference]['uncertainty_rms']
                values.append(val); separated += val > uncertainty
            output[component]=dict(rms=stats(values),larger_than_sum_of_estimated_uncertainties=separated)
        decomposition[track+'/'+regime]=output

stiffness_tables={}
for dn in (.1,1.,5.):
    for rn in (.1,1.,3.):
        cs=[c for c in CASES if c['diffusion_number']==dn and c['reaction_number']==rn]
        ar=[r for r in alignment_rows if r['diffusion_number']==dn and r['reaction_number']==rn]
        stiffness_tables[f'{dn:g}/{rn:g}']=dict(cases=len(cs),methods=method_table([r for r in ROWS if r['diffusion_number']==dn and r['reaction_number']==rn]),
            quadratic_q2_minus_q4=stats(c['quadratic_quadrature_differences_rms']['GL2_minus_GL4'] for c in cs),
            quadratic_q4_minus_q8=stats(c['quadratic_quadrature_differences_rms']['GL4_minus_GL8'] for c in cs),
            quadratic_q8_minus_q16=stats(c['quadratic_quadrature_differences_rms']['GL8_minus_GL16'] for c in cs),
            q4_q8_difference_over_gl4_total_error=stats(c['quadratic_quadrature_differences_rms']['GL4_minus_GL8']/R[c['stage'],c['case_id'],'gl4']['same_grid_metrics']['error_rms'] for c in cs),
            raw_over_nested_cubic_norm=stats(c['response_geometry']['basis_rms']['raw_c3']/c['response_geometry']['basis_rms']['nested_c3'] for c in cs),
            filtered_over_nested_cubic_norm=stats(c['response_geometry']['basis_rms']['filtered_c3']/c['response_geometry']['basis_rms']['nested_c3'] for c in cs),
            alignment={k:stats(r[k]for r in ar) for k in ar[0] if k.endswith('_cosine')})

geometry={}
for regime in ('overlapping_triads','aligned_harmonics','grid_relative'):
    cs=[c for c in CASES if c['field']['regime']==regime]
    ps=[c['response_geometry']['pair_oracles']['q2:filtered_c3'] for c in cs]
    geometry[regime]=dict(cases=len(cs),q2_filtered_cosine=stats(c['response_geometry']['normalized_gram'][0][3]for c in cs),
        normalized_pair_condition=stats(p['condition_number'] for p in ps),
        filtered_over_quadratic_norm=stats(c['response_geometry']['basis_rms']['filtered_c3']/c['response_geometry']['basis_rms']['q2'] for c in cs),
        teacher_informed_oracle_outside_old_gain_box=sum(any(not .25<=a<=1.75 for a in p['oracle_coefficients'])for p in ps),
        oracle_g2=stats(p['oracle_coefficients'][0]for p in ps),oracle_g3=stats(p['oracle_coefficients'][1]for p in ps))

counterexamples=[]
for method in ('gl2_filtered_cubic','gl2_raw_cubic','gl4_nested_cubic','etdrk4'):
    selected=sorted([r for r in ROWS if r['method']==method],key=lambda r:compare(r)['control_over_candidate'])[:4]
    for row in selected:
        c=C[row['stage'],row['case_id']]
        counterexamples.append(dict(method=method,stage=row['stage'],case_id=row['case_id'],phase=c['field']['phase'],
            diffusion_number=row['diffusion_number'],reaction_number=row['reaction_number'],
            candidate_rms=row['same_grid_metrics']['error_rms'],candidate_max=row['same_grid_metrics']['error_max'],
            control_gl4_rms=R[row['stage'],row['case_id'],'gl4']['same_grid_metrics']['error_rms'],
            candidate_over_gl4_rms=1/compare(row)['control_over_candidate'],reference_uncertainty_rms=row['same_grid_metrics']['reference_uncertainty_rms'],
            all_target_references_accepted=c['all_references_accepted'],quadratic_quadrature=c['quadratic_quadrature_differences_rms']))

fields={c['field']['field_id']:c['field'] for c in CASES}
ps=[c['response_geometry']['pair_oracles']['q2:filtered_c3'] for c in CASES]
result=dict(schema='tdn.resolve-interactions-review/v1',analysis_type='read_only_posthoc_review',
    source_artifacts={s+'/'+f:io.index()['files'][s+'/'+f]['sha256'] for s in STAGES for f in ('interaction_rows.json','interaction_cases.json','summary.json','protocol.json')},
    counts=dict(stages=len(STAGES),completed_stages=sum(s['status']=='COMPLETED' for s in SUMMARIES),cases=len(CASES),endpoints=len(ROWS),
        finite_endpoints=sum(r['same_grid_metrics']['finite']for r in ROWS),distinct_continuous_fields=len(fields),unique_sampled_field_arrays=len({c['field_state_sha256']for c in CASES}),
        independently_sampled_field_count=None,independent_sampling_claim=False,field_templates=3,phase_variants_per_template=2,
        same_grid_targets_accepted=sum(c['references']['same_grid']['accepted']for c in CASES),galerkin_targets_accepted=sum(c['references']['galerkin_grid']['accepted']for c in CASES),
        projected_fine_targets_accepted=sum(c['references']['projected_fine']['accepted']for c in CASES),
        full_reference_triples_accepted=sum(c['all_references_accepted']for c in CASES),
        correctness_failures=sum(s['correctness_failures']for s in SUMMARIES),component_errors=sum(bool(c['component_errors'])for c in CASES),
        same_grid_interval_violations=sum(r['same_grid_metrics']['physical_interval_violations']for r in ROWS),
        same_grid_energy_increase_over_1e_12=sum(r['same_grid_metrics']['energy_increase']>1e-12 for r in ROWS)),
    costs=dict(stage_seconds=sum(s['elapsed_seconds']for s in SUMMARIES),reference_seconds=sum(c['references']['reference_seconds']for c in CASES),
        scope='CPU FP64 diagnostic numerical work; shared components and single calls do not establish deployment cost'),
    field_declarations=fields,
    success_semantics=dict(ratio='control RMS / candidate RMS; greater than 1 favors candidate',
        win='control RMS minus estimated uncertainty exceeds candidate RMS plus estimated uncertainty',
        uncertainty_is_certificate=False,statistical_inference='none; repeated deterministic interventions, grids and tracks are not independent fields',
        joint_target=TARGET,norm='sampled nodal RMS/max; not Nyquist-weighted continuous L2'),
    overall_methods=method_table(ROWS),
    by_track_regime={track+'/'+regime:method_table([r for r in ROWS if r['track']==track and C[r['stage'],r['case_id']]['field']['regime']==regime])for track in ('discrete','continuum')for regime in ('overlapping_triads','aligned_harmonics','grid_relative')},
    stiffness_tables=stiffness_tables,error_decomposition=decomposition,
    maximum_error_identity_residual=max(abs(r['error_decomposition']['squared_identity_residual'])for r in ROWS),
    maximum_vector_identity_residual=max(r['error_decomposition']['vector_identity_max']for r in ROWS),
    unresolved_references=unresolved,identifiability=geometry,
    negative_alignment_counts={k:sum(r[k]<0 for r in alignment_rows)for k in alignment_rows[0]if k.endswith('_cosine')},
    alignment_reconstruction='cos(C,R-Q) = [cos(C,R)*RMS(R) - cos(C,Q)*RMS(Q)]/RMS(R-Q); R=teacher-DF, independently computed from logged residual cosines, Gram and endpoint norms',
    alignment_rows=alignment_rows,
    oracle_limits=dict(teacher_informed=True,deployable=False,negative_cubic_gain=sum(p['oracle_coefficients'][1]<0 for p in ps),
        outside_old_gain_box=sum(any(not .25<=a<=1.75 for a in p['oracle_coefficients'])for p in ps),
        explanation='Unbounded per-case least squares can compensate quadrature errors and small basis norms; coefficients are not identified physical laws or deployable fits'),
    alignment_sensitivity={q:dict(defect_over_uncertainty=stats(r['after_'+q+'_defect_over_uncertainty'] for r in alignment_rows),
        eligible=sum(r['after_'+q+'_defect_over_uncertainty']>10 and r['filtered_c3_rms']>1e-12 for r in alignment_rows),
        negative_after_filter=sum(r['after_'+q+'_defect_over_uncertainty']>10 and r['filtered_c3_rms']>1e-12 and r['after_'+q+'_filtered_c3_cosine']<0 for r in alignment_rows),
        negative_beyond_estimated_uncertainty=sum(r['after_'+q+'_filtered_c3_cosine'] < -1/r['after_'+q+'_defect_over_uncertainty'] for r in alignment_rows),
        minimum_filtered_basis_rms=min(r['filtered_c3_rms'] for r in alignment_rows),
        floor=1e-12,scope='Reported uncertainty is estimated, not certified; basis floor is stricter than 128*FP64epsilon') for q in ('q2','q4')},
    counterexamples=counterexamples,
    nested_nonwins=[dict(stage=r['stage'],case_id=r['case_id'],**compare(r))for r in ROWS if r['method']=='gl4_nested_cubic'and compare(r)['outcome']!='win'])
(BASE/'interactions-review.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:result[k]for k in ('counts','costs','negative_alignment_counts')},indent=2))
