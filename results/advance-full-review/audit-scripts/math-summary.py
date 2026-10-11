import json, math, statistics as st
from collections import Counter, defaultdict
from review_io import load,BASE

def med(x): return st.median(x)
def rang(x): return [min(x),med(x),max(x)]
rows=load('diagnose/diagnostic_rows.json')['rows']
key=lambda r:(r['track'],r['regime'],r['horizon'])
by={f:{key(r):r for r in rows if r['family']==f} for f in sorted({r['family'] for r in rows})}
out={'diagnostic_counts':{f:len(rs) for f,rs in by.items()},'diagnostic_methods':{}}
for f,rs in by.items():
 ratio=[by['quad2_full'][k]['error_rms']/r['error_rms'] for k,r in rs.items()]
 out['diagnostic_methods'][f]={'gl2_over_method_rms_min_median_max':rang(ratio),'rms_range':rang([r['error_rms'] for r in rs.values()]),'joint_2e5_pass':sum(max(r['upper_rms'],r['upper_max'])<=2e-5 for r in rs.values()),'strictly_better_than_GL2':sum(v>1 for v in ratio),'worse_than_GL2':[{**{k:r[k] for k in ['track','regime','horizon','error_rms']},'ratio':by['quad2_full'][key(r)]['error_rms']/r['error_rms']} for r in rs.values() if by['quad2_full'][key(r)]['error_rms']/r['error_rms']<1], 'energy_increases':sum(r['energy_increase']>0 for r in rs.values()),'interval_violations':sum(r['physical_interval_violations'] for r in rs.values())}
projection=load('diagnose/residual_projections.json')['rows']
out['projection']={'n':len(projection),'rank_counts':dict(Counter(r['rank'] for r in projection)),'condition_range':rang([r['condition_number'] for r in projection]),'gain_quadratic_range':rang([r['oracle_coefficients'][0] for r in projection]),'gain_cubic_range':rang([r['oracle_coefficients'][1] for r in projection]),'one_over_two_basis_error':rang([r['one_basis_oracle_error_rms']/r['oracle_error_rms'] for r in projection]),'unrepresented_fraction':rang([r['relative_unrepresented_norm'] for r in projection])}
orders=defaultdict(list)
for r in load('diagnose/order_rows.json')['rows']: orders[(r['track'],r['regime'])].append(r)
out['small_step_order']=[{'track':k[0],'regime':k[1],'orders':[math.log(b['raw_remainder_rms']/a['raw_remainder_rms'])/math.log(b['horizon']/a['horizon']) for a,b in zip(sorted(rs,key=lambda v:v['horizon']),sorted(rs,key=lambda v:v['horizon'])[1:])], 'h4_coeff':rang([r['remainder_over_h4'] for r in rs])} for k,rs in orders.items()]
com=load('diagnose/composition_rows.json')['rows']
out['composition']={f:{'defect_rms':rang([r['composition_defect_rms'] for r in com if r['family']==f]),'directional_amplification':rang([r['physical_directional_amplification'] for r in com if r['family']==f])} for f in by}
out['audit']=load('audit/audit_rows.json')['rows']
out['prototype']={}
for p in ['XH','XR','XT']:
 r=load('explore-'+p+'/prototype_rows.json')
 out['prototype'][p]=[{'id':v['experiment_id'],'metrics':v['metrics'],'checks':[{k:c[k] for k in ['check_id','category','measured','target','verdict','reason']} for c in v['checks']]} for v in r]
 out['prototype'][p+'_check_counts']={f'{cat}/{verdict}':n for (cat,verdict),n in Counter((c['category'],c['verdict']) for v in r for c in v['checks']).items()}
(BASE/'math-summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['prototype','audit']},indent=2))

# Post-hoc diagnostic-design audit: enumerate Fourier supports, without rerunning science.
from itertools import product
modes_by_regime={"low":[(1,0),(0,1)],"high_pair":[(5,1),(6,1)],"broadband":[(1,0),(1,2),(2,-1),(6,1)],"near_nyquist":[(6,0),(0,6)]}
supports={}
for name,modes in modes_by_regime.items():
    signed=modes+[(-a,-b) for a,b in modes]
    sets=[{tuple(sum(v[j] for v in vs)%16 for j in (0,1)) for vs in product(signed,repeat=order)} for order in (2,3)]
    supports[name]={"quadratic_support_count":len(sets[0]),"cubic_support_count":len(sets[1]),"intersection":sorted(sets[0]&sets[1])}
repro={}
from pathlib import Path
for proto,fname,cols in [("XH","XH_history.json",["error_rms","error_max","mean_error"]),("XR","XR_resolvent.json",["error_rms","error_max","mean_error"]),("XT","XT_tangent.json",["tangent_fd_relative_error","semigroup_relative_error"])]:
    old=json.loads((Path(".runtime/advance-exploration-full-development")/proto/fname).read_text())
    new=load("explore-"+proto+"/"+fname)
    repro[proto]={"same_parents":old["parent_ids"]==new["parent_ids"],"same_settings":old["settings"]==new["settings"],"old_rows":len(old["rows"]),"native_rows":len(new["rows"]),"checked_columns":cols,"maximum_numerical_difference":max(abs(a[k]-b[k]) for a,b in zip(old["rows"],new["rows"]) for k in cols)}
(BASE/"math-design-audit.json").write_text(json.dumps({"fourier_supports_modulo16":supports,"previous_development_reproducibility":repro},indent=2)+"\n")
