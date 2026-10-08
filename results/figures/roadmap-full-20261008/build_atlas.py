#!/usr/bin/env python3
"""Render every archived roadmap experiment, check and JUnit occurrence; execute no archive code."""
import os, argparse
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-root', type=Path, required=True, help='Extracted full-run root; read only')
parser.add_argument('--review-root', type=Path, required=True, help='Directory containing independently derived review JSONs and archive index')
parser.add_argument('--output-dir', type=Path, required=True)
args=parser.parse_args()
HERE=args.review_root.resolve()
OUT=args.output_dir.resolve(); OUT.mkdir(parents=True, exist_ok=True)
os.environ['MPLCONFIGDIR']=str(OUT/'.matplotlib')
os.environ['XDG_CACHE_HOME']=str(OUT/'.cache'); Path(os.environ['XDG_CACHE_HOME']).mkdir(exist_ok=True)
os.environ['TMPDIR']=str(OUT/'.tmp'); Path(os.environ['TMPDIR']).mkdir(exist_ok=True)
import csv, gzip, hashlib, json, math, textwrap, zipfile
from collections import Counter
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

ROOT=args.run_root.resolve()
RUN=ROOT.name
STAGES=['audit','headroom','prepare','train','confirm_prepare','confirm','policy','transfer','scaling','report']
ECOLS=256; CCOLS=768
CODE={'GOOD':0,'BAD':1,'NA':2}; COLORS=['#168779','#cf654f','#adbbc7']
INK='#173044'; MUTED='#557083'; BG='#f8fafb'; LINE='#dce5eb'; BLUE='#256d9d'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':INK,'text.color':INK,'xtick.color':MUTED,'ytick.color':MUTED,'axes.edgecolor':LINE,'axes.titleweight':'bold','savefig.facecolor':BG,'figure.facecolor':BG,'pdf.fonttype':42})
VCMAP=ListedColormap(COLORS); VCMAP.set_bad('#ffffff')
SCMAP=plt.get_cmap('cividis').copy(); SCMAP.set_bad('#ffffff')

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def compact(v): return json.dumps(v,separators=(',',':'),ensure_ascii=False)
def raster(values,width):
 a=np.full(math.ceil(len(values)/width)*width,np.nan,dtype=np.float32);a[:len(values)]=values
 return a.reshape(-1,width)
def panel_title(fig,x,y,title,subtitle=None):
 fig.text(x,y,title,fontsize=19,fontweight='bold',va='top')
 if subtitle: fig.text(x,y-.010,subtitle,fontsize=10.7,color=MUTED,va='top',linespacing=1.4)
def stage_name(s): return s.replace('_',' ')

source=[]; stages={}; exp_counts=Counter(); check_counts=Counter(); categories={}
# Gzip makes all 619,697 point identities and underlying check operands available without a giant loose CSV.
ef=gzip.open(OUT/'experiment-cell-index.csv.gz','wt',newline='',encoding='utf8')
cf=gzip.open(OUT/'check-cell-index.csv.gz','wt',newline='',encoding='utf8')
ew=csv.writer(ef);cw=csv.writer(cf)
ew.writerow(['stage','stage_experiment_index_0','cell_row_0','cell_column_0','experiment_id','verdict','score_1_100','evidence_coverage','mechanism_ids','combination_ids','row_sha256','source_line_1'])
cw.writerow(['stage','stage_check_index_0','cell_row_0','cell_column_0','stage_experiment_index_0','experiment_id','check_index_0','check_id','category','verdict','score_1_100','required','applicable','measured_json','target_json','relation','units','reason','experiment_row_sha256'])
for stage in STAGES:
 p=ROOT/stage/'rows.jsonl'; vals=[];scores=[];checks=[];ec=Counter();cc=Counter(); ids=set(); summary=json.load(open(ROOT/stage/'summary.json'))
 for i,line in enumerate(p.open()):
  r=json.loads(line);key=r['experiment_id']; assert key not in ids,(stage,key);ids.add(key)
  a=r['assessment'];v=a['verdict'];sc=a['score_1_100'];assert v in CODE and 1<=sc<=100
  vals.append(CODE[v]);scores.append(sc);ec[v]+=1
  ew.writerow([stage,i,i//ECOLS,i%ECOLS,key,v,sc,a['evidence_coverage'],';'.join(r['mechanism_ids']),';'.join(r['combination_ids']),r['row_sha256'],i+1])
  for j,c in enumerate(r['checks']):
   k=len(checks);v=c['verdict'];checks.append(CODE[v]);cc[v]+=1;categories.setdefault(c['category'],Counter())[v]+=1
   cw.writerow([stage,k,k//CCOLS,k%CCOLS,i,key,j,c['check_id'],c['category'],v,c['score_1_100'],c['required'],c['applicable'],compact(c['measured']),compact(c['target']),c['relation'],c['units'],c['reason'],r['row_sha256']])
 assert len(vals)==summary['experiment_count']; assert dict(ec)==summary['verdict_counts']
 source.append({'stage':stage,'path':str(p.relative_to(ROOT)),'sha256':sha(p),'experiments':len(vals),'checks':len(checks)})
 stages[stage]={'verdict':raster(vals,ECOLS),'score':raster(scores,ECOLS),'checks':raster(checks,CCOLS),'n':len(vals),'nc':len(checks),'ec':ec,'cc':cc,'seconds':summary['elapsed_seconds']}
 exp_counts.update(ec);check_counts.update(cc)
ef.close();cf.close()
assert sum(x['n'] for x in stages.values())==74150
assert sum(x['nc'] for x in stages.values())==619697

junit=[];jc=Counter()
with (OUT/'junit-cell-index.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['stage','case_index_0','cell_row_0','cell_column_0','class','test_name','status','seconds','source'])
 for stage in STAGES:
  p=ROOT/'reporter-tests'/stage/'tests.xml'
  if not p.exists():continue
  cases=[]
  for i,c in enumerate(ET.parse(p).iter('testcase')):
   status='FAIL' if c.find('failure') is not None else 'ERROR' if c.find('error') is not None else 'SKIP' if c.find('skipped') is not None else 'PASS'
   cases.append({'status':status,'seconds':float(c.get('time','0'))});jc[status]+=1
   w.writerow([stage,i,i//150,i%150,c.get('classname'),c.get('name'),status,c.get('time'),str(p.relative_to(ROOT))])
  junit.append((stage,cases))
aggregate=json.load(open(HERE/'aggregate-review.json'))
neural=json.load(open(HERE/'neural-review.json'))
fixed=json.load(open(HERE/'neural-fixed-horizon-review.json'))
policy=json.load(open(HERE/'policy-review.json'))
mechanisms=aggregate['mechanisms_and_combinations']
assert len(mechanisms)==29
with (OUT/'mechanism-recorded-summary.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['id','name','recorded_verdict','recorded_score_1_100','evidence_coverage','linked_experiment_count'])
 for m in mechanisms:w.writerow([m['id'],m['name'],m['verdict'],m['score_1_100'],m['evidence_coverage'],m['experiment_count']])

fig=plt.figure(figsize=(36,46),dpi=90)
# Full-width report header.
fig.text(.035,.975,'TDN  /  THE COMPLETE EXPERIMENT ATLAS',fontsize=44,fontweight='bold',va='top')
fig.text(.035,.957,'Native Fedora · Ryzen 7800X3D · RTX 4090 24 GB · full bounded roadmap · M00–M23 + C0–C4',fontsize=21,va='top',color=MUTED)
fig.text(.035,.942,RUN+'  |  10 completed stages  |  3 paired training seeds',fontsize=13,color=MUTED,va='top')
# KPI boxes.
kpis=[('74,150','experiment records','74,149 science + 1 report inventory'),('619,697','individual check records','Every occurrence is mapped below'),(f"{jc['PASS']:,} / {sum(jc.values()):,}",'software tests passed',f"{jc['SKIP']} skip · {jc['FAIL']+jc['ERROR']} failures/errors"),('24','independent confirmation fields','12 continuum fields; 6 policy fields')]
for i,(big,label,note) in enumerate(kpis):
 x=.035+i*.239
 fig.add_artist(Rectangle((x,.890),.226,.040,transform=fig.transFigure,facecolor='#eaf1f5',edgecolor='none',zorder=0))
 fig.text(x+.009,.924,big,fontsize=32,fontweight='bold',va='top')
 fig.text(x+.009,.906,label,fontsize=14,fontweight='bold',va='top')
 fig.text(x+.009,.895,note,fontsize=10,color=MUTED,va='top')

# Mechanism summary table.
panel_title(fig,.035,.878,'A  Recorded mechanism / combination evidence','Scores reflect required check weight met, not model quality or proof.\nOne mechanism can link to many repeated experiment records.')
ax=fig.add_axes([.035,.675,.303,.180]);ax.set_xlim(0,1);ax.set_ylim(29.7,-1.7);ax.axis('off')
ax.text(.0,-1,'ID',fontsize=10,fontweight='bold');ax.text(.105,-1,'DECLARED MECHANISM',fontsize=10,fontweight='bold');ax.text(.785,-1,'SCORE',fontsize=10,fontweight='bold');ax.text(.91,-1,'KNOWN',fontsize=10,fontweight='bold')
for i,m in enumerate(mechanisms):
 if i%2==0:ax.add_patch(Rectangle((0,i-.47),1,.94,facecolor='#eef3f6',edgecolor='none'))
 ax.text(.006,i,m['id'],fontsize=11,fontweight='bold',va='center',color=COLORS[CODE[m['verdict']]])
 name=m['name'];name=name if len(name)<51 else name[:48]+'…'
 ax.text(.105,i,name,fontsize=9.5,va='center')
 ax.barh(i,m['score_1_100']/100*.12,left=.77,height=.62,color=BLUE,alpha=.3)
 ax.text(.82,i,str(m['score_1_100']),fontsize=11,fontweight='bold',va='center',ha='center')
 ax.text(.94,i,f"{m['evidence_coverage']:.0%}",fontsize=10,va='center',ha='center')
fig.text(.035,.669,'Recorded aggregate verdicts: '+', '.join(f'{n} {k}' for k,n in Counter(x['verdict'] for x in mechanisms).items())+'.  Names and exact counts: companion CSV.',fontsize=10,color=MUTED)

# Accuracy by every family; unconditioned endpoint success, no biased equal-feasibility filtering.
panel_title(fig,.371,.878,'B  Joint accuracy across all endpoint schedules','Cells = fraction meeting BOTH RMS and maximum error targets.\nDiscrete and continuum targets differ; repeated endpoints are paired.')
families=list(neural['training']['by_family'])+[x for x in neural['by_family_track'] if x not in neural['training']['by_family']]
targets=neural['targets'];arr=np.array([[neural['by_family_track'][f][track]['passes'][t]/neural['by_family_track'][f][track]['rows'] for track in ['discrete','continuum'] for t in targets] for f in families])
ax=fig.add_axes([.435,.681,.183,.168]);im=ax.imshow(arr,cmap='Blues',vmin=0,vmax=1,aspect='auto',interpolation='nearest')
ax.set_yticks(range(len(families)),families,fontsize=9.4);ax.tick_params(axis='y',length=0)
ax.set_xticks(range(6),['2e−4','2e−5','2e−6']*2,fontsize=10);ax.xaxis.tick_top()
for i in range(len(families)):
 for j in range(6):ax.text(j,i,f'{arr[i,j]*100:.0f}',ha='center',va='center',fontsize=9,color='white' if arr[i,j]>.58 else INK)
ax.axvline(2.5,color=BG,lw=4)
fig.text(.458,.860,'DISCRETE',fontsize=11,fontweight='bold');fig.text(.548,.860,'CONTINUUM',fontsize=11,fontweight='bold')
# Latency labels alongside the same family order.
lax=fig.add_axes([.626,.681,.036,.168]);lax.set_ylim(len(families)-.5,-.5);lax.set_xlim(0,1);lax.axis('off')
for i,f in enumerate(families):lax.text(.5,i,f"{neural['by_family_track'][f]['discrete']['median_latency_ms']:.2f}",ha='center',va='center',fontsize=9.6)
fig.text(.621,.860,'ms¹',fontsize=11,fontweight='bold')
fig.text(.371,.669,'¹ Median discrete latency across measured schedules (different work per schedule); descriptive, not an accuracy-matched ranking.',fontsize=9.5,color=MUTED)

# Run time and compact policy outcomes.
panel_title(fig,.698,.878,'C  Work, verification and deployment','Times show science vs. complete Slurm allocation (excludes queue).\nPolicy costs include rejected attempts and fallback.')
rax=fig.add_axes([.757,.772,.190,.077]);yt=np.arange(10);times=np.array([stages[s]['seconds']/60 for s in STAGES])
accounting=json.load(open(ROOT/'state/scheduler-accounting.json'))
allocation_by_stage={r['workflow_stage']:r for r in accounting['records'] if r['record_kind']=='allocation'}
assert set(allocation_by_stage)==set(STAGES)
assert all(r['State']=='COMPLETED' and r['ExitCode']=='0:0' for r in allocation_by_stage.values())
allocation_seconds=[allocation_by_stage[s]['elapsed_seconds'] for s in STAGES]
rax.barh(yt,np.array(allocation_seconds)/60,color='#d1dfe8',height=.82,label='Allocation')
rax.barh(yt,times,color=[BLUE if s in ['train','confirm','policy','scaling'] else '#79a4bc' for s in STAGES],height=.40,label='Science')
rax.set_yticks(yt,[stage_name(s) for s in STAGES],fontsize=9.4);rax.invert_yaxis();rax.set_xlim(0,34);rax.set_xticks([0,10,20],['0','10','20 min'],fontsize=9);rax.grid(axis='x',color=LINE);rax.set_axisbelow(True)
for y,t,a in zip(yt,times,allocation_seconds):rax.text(a/60+.25,y,f'{t:.1f} / {a/60:.1f}',fontsize=8.5,va='center')
fig.text(.698,.759,f"Science {sum(times):.1f} min  /  allocated 78.0 min  /  GPU allocated 62.4 min",fontsize=11,fontweight='bold')
fig.text(.698,.747,'DEPLOYED POLICY · 984 endpoints · 6 independent fields',fontsize=11,fontweight='bold')
fig.text(.698,.735,'900 jointly accurate  /  84 false accepts\n412 accurate fallbacks  /  no formal coverage guarantee\n12.99× aggregate classical cost; accuracy is not matched',fontsize=13,linespacing=1.55,va='top')
fig.text(.698,.704,'CHECK PURPOSE',fontsize=10.2,fontweight='bold')
for x, label, color in zip([.804,.869,.933], ['GOOD','BAD','NA'], COLORS): fig.text(x,.704,label,fontsize=10.2,fontweight='bold',ha='right',color=color)
for i,cat in enumerate(['correctness','math','gap','utility']):
 yy=.695-i*.007
 fig.text(.698,yy,cat,fontsize=10.3)
 for x,v in zip([.804,.869,.933],['GOOD','BAD','NA']): fig.text(x,yy,f'{categories.get(cat,{}).get(v,0):,}',fontsize=10.3,ha='right')

# Corrected comparison / interpretation banner.
fig.add_artist(Rectangle((.035,.624),.928,.033,transform=fig.transFigure,facecolor='#fff0d9',edgecolor='none'))
fig.text(.045,.650,'READ THE COLORS CORRECTLY',fontsize=13,fontweight='bold',va='top')
fig.text(.045,.640,'GOOD = declared checks met   •   BAD = at least one required check failed   •   NA = evidence incomplete   •   scores are evidence attainment, not probabilities.',fontsize=13,va='top')
fig.text(.045,.630,'The original accuracy-matched frontiers mixed horizons in 1,412 / 22,032 rows. Raw scores below are preserved; the companion PDF adds corrected comparisons.',fontsize=11.5,va='top')

# Complete raster. Every record appears exactly once in each applicable channel.
panel_title(fig,.035,.612,'D  EVERY EXPERIMENT: verdict + score','Left-to-right, then next row. 256 columns per stage. White = unused padding.')
panel_title(fig,.526,.612,'E  EVERY INDIVIDUAL CHECK: verdict','Left-to-right, then next row. 768 columns per stage. No sampling or aggregation.')
# Shared stage block positions, with a minimum height so small stages remain legible.
start=.583; available=.459
weights=np.array([max(48,stages[s]['nc']/768) for s in STAGES],float);weights/=weights.sum()
positions={}; y=start
for stage,wgt in zip(STAGES,weights):
 h=available*wgt;d=stages[stage]; labelh=.0048;datah=h-labelh-.002
 top=y;bottom=y-h;positions[stage]={'top':top,'bottom':bottom,'data_top':top-labelh,'data_bottom':bottom+.002}
 fig.text(.035,top-.0008,f"{stage.upper().replace('_',' ')}    {d['n']:,} records",fontsize=10.3,fontweight='bold',va='top')
 fig.text(.526,top-.0008,f"{stage.upper().replace('_',' ')}    {d['nc']:,} checks",fontsize=10.3,fontweight='bold',va='top')
 for x,w,key,cmap,norm in [(.035,.222,'verdict',VCMAP,Normalize(-.5,2.5)),(.276,.222,'score',SCMAP,Normalize(1,100)),(.526,.437,'checks',VCMAP,Normalize(-.5,2.5))]:
  a=fig.add_axes([x,bottom+.002,w,datah]);a.imshow(d[key],cmap=cmap,norm=norm,aspect='auto',interpolation='nearest',resample=False);a.set_xticks([]);a.set_yticks([])
  for sp in a.spines.values():sp.set_visible(False)
 y=bottom
fig.text(.035,.590,'VERDICT',fontsize=11,fontweight='bold');fig.text(.276,.590,'RECORDED SCORE  1–100',fontsize=11,fontweight='bold')
# Color legend and rigorous count summaries.
legendy=.110
for i,(name,col) in enumerate(zip(['GOOD','BAD','NA'],COLORS)):
 x=.035+i*.093
 fig.add_artist(Rectangle((x,legendy),.007,.004,transform=fig.transFigure,facecolor=col,edgecolor='none'))
 fig.text(x+.010,legendy+.001,name,fontsize=11,va='center')
cax=fig.add_axes([.351,.110,.146,.004]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(1,100),cmap=SCMAP),cax=cax,orientation='horizontal');cb.set_ticks([1,25,50,75,100]);cb.ax.tick_params(labelsize=9,length=2)
fig.text(.526,.112,'ALL CHECKS   '+ '   /   '.join(f'{k} {check_counts[k]:,}' for k in ['GOOD','BAD','NA']),fontsize=11,fontweight='bold')
fig.text(.035,.094,'ALL EXPERIMENTS   '+ '   /   '.join(f'{k} {exp_counts[k]:,}' for k in ['GOOD','BAD','NA']),fontsize=11,fontweight='bold')
fig.text(.526,.096,'BAD can reflect cost alone: 394 transfer records fail utility checks, not trajectory correctness.',fontsize=10.7,color=MUTED)

# Every JUnit occurrence, indexed separately.
panel_title(fig,.035,.080,'F  EVERY SOFTWARE TEST OCCURRENCE','641 passed, 1 optional Tower-checkout skip; four repetitions of 48 GPU tests remain separate occurrences.')
x=.035
for stage,cases in junit:
 w=.335 if stage=='audit' else .135
 a=fig.add_axes([x,.043,w,.020]);v=[0 if c['status']=='PASS' else 2 if c['status']=='SKIP' else 1 for c in cases]
 a.imshow(raster(v,150),cmap=VCMAP,norm=Normalize(-.5,2.5),interpolation='nearest',aspect='auto',resample=False);a.axis('off')
 fig.text(x,.039,f'{stage}: {len(cases)}',fontsize=10,fontweight='bold')
 x+=w+.015
fig.text(.035,.023,'TRACEABILITY  Each cell maps to its exact experiment/check/test identity, score, operands and source in the companion compressed CSV indexes.',fontsize=12,fontweight='bold')
fig.text(.035,.014,'Stage-local coordinates are zero-based. Colored cells are observations, not independent samples. A numerical pass is not a mathematical proof; these internal FNO controls are not an official-paper reproduction.',fontsize=10.5,color=MUTED)
fig.text(.035,.006,'Source: integrity-checked uploaded full-run archive. Read-only visualization; no models retrained, no source observations modified. See atlas-manifest.json for source hashes and coverage audit.',fontsize=10.3,color=MUTED)

# High-resolution PNG / compact review preview. Directly re-render each resolution, no resampling of scientific data.
fig.savefig(OUT/'TDN-complete-experiment-atlas.png',dpi=320,facecolor=BG)
fig.savefig(OUT/'TDN-complete-experiment-atlas-preview.png',dpi=65,facecolor=BG)
with PdfPages(OUT/'TDN-complete-experiment-atlas.pdf',metadata={'Title':'TDN Complete Experiment Atlas','Author':'TDN results review','Subject':'Exhaustive 74150 experiment / 619697 check atlas; recorded scores and corrected comparison diagnostics'}) as pdf:
 pdf.savefig(fig,dpi=200)
 # Second page: scientifically valid comparisons, all cells separately represent exact scope.
 for tolerance,target_key in zip([2e-4,2e-5,2e-6],targets):
  f=plt.figure(figsize=(20,14),dpi=100)
  f.text(.045,.955,f'HORIZON-MATCHED COMPARISONS  /  {tolerance:g}',fontsize=28,fontweight='bold')
  f.text(.045,.925,'Offline reanalysis of existing timings: same target track, same final time, same parent / grid / paired seed.',fontsize=15,color=MUTED)
  f.text(.045,.900,'Median comparator latency ÷ candidate latency. Above 1 = candidate faster. Post-hoc feasible schedule selection; excludes offline training cost.',fontsize=12)
  fs=list(neural['training']['by_family']);combos=[(t,h,c) for t in ['discrete','continuum'] for h in [.27,.81] for c in ['cheap_fno','deep_fno','best_classical']]
  vals=np.full((len(fs),len(combos)),np.nan);eligible=np.zeros_like(vals,int)
  for r in fixed['summaries']:
   if r['family'] not in fs or r['target']!=target_key:continue
   for c,comp in r['controls'].items():
    col=(r['track'],r['horizon'],c)
    if col in combos:
     j=combos.index(col);i=fs.index(r['family']);v=comp['median_control_over_candidate'];vals[i,j]=v if v is not None else np.nan;eligible[i,j]=comp['eligible']
  ax=f.add_axes([.195,.32,.75,.535]);cmap=plt.get_cmap('RdYlBu').copy();cmap.set_bad('#e5eaef')
  from matplotlib.colors import TwoSlopeNorm
  show=np.log2(vals);ax.imshow(show,cmap=cmap,norm=TwoSlopeNorm(vmin=-4,vcenter=0,vmax=4),interpolation='nearest',aspect='auto')
  ax.set_yticks(range(len(fs)),fs,fontsize=11);ax.set_xticks(range(len(combos)),[c.replace('_',' ') for t,h,c in combos],rotation=30,ha='right',fontsize=10)
  for i in range(len(fs)):
   for j in range(len(combos)):
    txt='NA' if not np.isfinite(vals[i,j]) else f'{vals[i,j]:.2f}×\nn={eligible[i,j]}'
    ax.text(j,i,txt,ha='center',va='center',fontsize=9,color='white' if np.isfinite(show[i,j]) and abs(show[i,j])>2.2 else INK)
  for j in [2.5,5.5,8.5]:ax.axvline(j,color=BG,lw=6)
  for i,(track,horizon) in enumerate([(t,h) for t in ['discrete','continuum'] for h in [.27,.81]]):
   f.text(.195+(.75/4)*(i+.5),.865,f'{track.upper()} · T={horizon}',ha='center',fontsize=12,fontweight='bold')
  f.text(.045,.225,'Scope and limits',fontsize=18,fontweight='bold')
  notes=[
  f'Both RMS and maximum error must be ≤ {tolerance:g}. n is eligible paired measurement cells, not independent fields. Discrete: 24 independent fields; continuum: 12.',
  'The 3 paired training seeds, 2 grids and 2 coefficient pairs reuse fields. Timing differences of only a few percent are not demonstrated robust speedups.',
  'Cheap FNO selected initialization in all 3 seeds. Direct FNO has no feasible matched comparator cells here. These bounded internal baselines do not establish a win over the FNO paper.',
  'Original frontier scores are retained on page 1 for faithful traceability. 1,412 original rows mixed final times; this page reconstructs both horizons separately.',
  'Policy: 84 continuum false accepts remain failures. The reported formal-policy break-even timing uses identical fallback work and is confounded by first-call timing; it is not shown as a win.',
  'No panel is a theorem, population guarantee or deployment certificate. Exact source observations and all missing / failed checks remain visible in the lookup files.'
  ]
  for i,note in enumerate(notes):f.text(.045,.198-i*.029,note,fontsize=10.4,color=MUTED)
  pdf.savefig(f,dpi=180)
  f.savefig(OUT/f'TDN-horizon-matched-{tolerance:g}-preview.png',dpi=90,facecolor=BG)
  plt.close(f)
plt.close(fig)

# Verify the *rendered* lossless PNG at every scientific cell center, independently
# of the CSV traversal. Every small check cell has multiple pixels in both directions.
Image.MAX_IMAGE_PIXELS=200_000_000
png=np.asarray(Image.open(OUT/'TDN-complete-experiment-atlas.png'))[:,:,:3]
pixel_audit={}; H,W=png.shape[:2]
for key,x,width,cmap,norm in [('verdict',.035,.222,VCMAP,Normalize(-.5,2.5)),('score',.276,.222,SCMAP,Normalize(1,100)),('checks',.526,.437,VCMAP,Normalize(-.5,2.5))]:
 count=0; mismatches=0
 for stage in STAGES:
  a=stages[stage][key];rr,cc=np.where(np.isfinite(a));v=a[rr,cc]
  pos=positions[stage]; bottom=pos['data_bottom'];height=pos['data_top']-bottom
  px=np.floor((x+(cc+.5)/a.shape[1]*width)*W).astype(int)
  py=np.floor((1-(bottom+(a.shape[0]-rr-.5)/a.shape[0]*height))*H).astype(int)
  expected=cmap(norm(v),bytes=True)[:,:3]
  observed=png[py,px]
  # One 8-bit level accommodates documented renderer rounding, not color classes.
  delta=np.max(np.abs(expected.astype(int)-observed.astype(int)),axis=1)
  bad=int(np.count_nonzero(delta>1));mismatches+=bad;count+=len(v)
 assert mismatches==0,(key,mismatches,count)
 pixel_audit[key]={'verified_cell_centers':count,'mismatched_colors':mismatches,'max_allowed_8bit_rounding':1}
del png

# Independent one-to-one coverage audit by rereading emitted index files.
lookup_audit={}
for stem,width,expected in [('experiment',ECOLS,74150),('check',CCOLS,619697)]:
 counts=Counter();positions_seen=set();n=0
 with gzip.open(OUT/f'{stem}-cell-index.csv.gz','rt',newline='') as f:
  for r in csv.DictReader(f):
   idx=int(r[f'stage_{stem}_index_0']); row=int(r['cell_row_0']);col=int(r['cell_column_0']);key=(r['stage'],row,col)
   assert idx==row*width+col and 0<=col<width and key not in positions_seen
   positions_seen.add(key);counts[r['stage']]+=1;n+=1
 assert n==expected
 lookup_audit[stem]={'records':n,'unique_stage_cell_coordinates':len(positions_seen),'columns':width,'stage_counts':dict(counts),'zero_omissions':True,'zero_duplicate_cells':True}
manifest={'schema':'tdn.exhaustive-atlas/v1','run':RUN,'source_archive_index_sha256':sha(HERE/(RUN+'-review-20261008T044131314641Z.tar.gz.index.json')),'review_input_sha256':{name:sha(HERE/name) for name in ['aggregate-review.json','neural-review.json','neural-fixed-horizon-review.json','policy-review.json']},'read_only_review':True,'source_record_count':74150,'science_record_count':74149,'report_inventory_record_count':1,'check_occurrences':619697,'junit_occurrences':sum(jc.values()),'junit_status':dict(jc),'allocation_seconds_by_stage':dict(zip(STAGES,allocation_seconds)),'total_allocation_seconds':sum(allocation_seconds),'gpu_allocation_seconds':sum(r['elapsed_seconds'] for r in allocation_by_stage.values() if 'gres/gpu=' in r['AllocTRES']),'allocated_cpu_seconds':sum(r['allocated_cpu_seconds'] for r in allocation_by_stage.values()),'scheduler_accounting_sha256':sha(ROOT/'state/scheduler-accounting.json'),'experiment_verdicts':dict(exp_counts),'check_verdicts':dict(check_counts),'check_categories':{k:dict(v) for k,v in categories.items()},'source_ledgers':source,'coverage_audit':lookup_audit,'rendered_png_cell_audit':pixel_audit,'rendering':{'png_pixels':[11520,14720],'raster_mapping':'one logged record per logical color cell, nearest-neighbor; stage-local left-to-right row-major; unfilled cells white','experiment_columns':ECOLS,'check_columns':CCOLS,'junit_columns':150,'stage_figure_positions':positions,'channel_figure_positions':{'verdict':{'left':.035,'width':.222},'score':{'left':.276,'width':.222},'checks':{'left':.526,'width':.437}},'score_meaning':'recorded weighted required-check attainment, 1..100; not accuracy/probability/proof','raw_scores_preserved':True,'horizon_comparisons':'PDF pages 2–4 reconstructed with separate horizons 0.27 and 0.81; not substituted into recorded scores'},'limitations':['Observations are paired/repeated, not independent sample counts.','Scientific BAD is distinct from execution failure.','No official FNO-paper reproduction.','No retraining or scientific reruns.','All original scores retained, including known mixed-horizon original frontier flaw.']}
(OUT/'atlas-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(OUT/'README.txt').write_text('''TDN COMPLETE EXPERIMENT ATLAS\n\nOpen TDN-complete-experiment-atlas.pdf and zoom. Page 1 maps EVERY archived experiment, check and software-test occurrence. Pages 2–4 show corrected horizon-matched comparisons at all three tolerances. The lossless PNG is 11520 x 14720 pixels. The preview is for navigation; use the full PNG/PDF for cells.\n\nCOVERAGE\n74,150 experiment rows = 74,149 scientific records + 1 report-inventory record.\n619,697 logged individual checks, including required / inapplicable / NA checks.\n642 native software-test occurrences = 641 passed + 1 skipped.\nEach color cell is a logged occurrence, NOT a new independent sample.\n\nFIND A CELL\nEach stage is separately indexed from zero. Read left to right then downward.\nExperiment verdict and score maps: 256 columns.\nIndividual check maps: 768 columns.\nJUnit software-test strips: 150 columns.\nIndex = row * number_of_columns + column.\nWhite cells are padding and do not refer to a test.\n\nUnzip the CSV .gz files (gzip-compatible, e.g. 7-Zip), then filter by stage and ID or coordinates in a spreadsheet. The experiment lookup links to source ledger line and SHA256. The check lookup includes identity, measured value, target, comparison, required/applicable flags and reason. JUnit CSV includes exact parameterized test identities.\n\nHOW TO READ SCORES\nGOOD: declared checks met; BAD: at least one required failure; NA: incomplete evidence. The 1–100 score is required-check weight met, with no credit for NA; not model quality, probability or mathematical proof. Mechanism rows overlap and cannot be summed as independent evidence.\n\nKNOWN SCIENTIFIC LIMITS\nThe original frontier rows sometimes mix final times. Page 1 preserves their recorded scores; pages 2–4 reconstruct comparisons with T=0.27 and T=0.81 kept separate. It uses existing measurements and post-hoc feasible schedule selection, not a new experiment or deployable policy. Formal fallback amortization is not displayed as a win because identical fallback work has a first-call timing confound. Internal FNO baselines are not an official paper reproduction.\n\nPROVENANCE\nSource is the uploaded full Fedora/RTX4090 run, with immutable scientific ledgers preserved. atlas-manifest.json records source hashes and an independent one-to-one CSV coordinate coverage audit. Source archive Python / pickle code was not executed. build_atlas.py is the review-side plotting script, requiring numpy, matplotlib and Pillow. No observations were dropped or resampled.\n\nREGENERATE\npython build_atlas.py --run-root /path/to/extracted/fedora-roadmap-20261008T022526329292Z --review-root /path/to/review-json-directory --output-dir /path/to/new-figures\nThe review directory must contain aggregate-review.json, neural-review.json, neural-fixed-horizon-review.json, policy-review.json and the uploaded archive index. Its independently derived summaries are additional inputs to the plot, not newly generated science.\n\nALLOCATIONS\nSlurm allocation elapsed seconds (jobs 197–206): 28,118,114,536,148,1757,1066,390,382,140. Total 4,679 s; GPU allocations 3,741 s. Allocation bars include stage worker/export overhead but exclude queue time. No invented monetary cost.\nThe skipped software test requires an optional unchanged Tower checkout. Separate read-only native Tower review validated all 45,277 exported pages.\n''')
# Keep the regeneration script with the artifacts, without copying data.
import shutil
if Path(__file__).resolve() != (OUT/'build_atlas.py').resolve():
 shutil.copy2(__file__,OUT/'build_atlas.py')
files={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in OUT.iterdir() if p.is_file() and p.name!='artifact-sha256.json' and p.suffix!='.zip'}
(OUT/'artifact-sha256.json').write_text(json.dumps(files,indent=2)+'\n')
with zipfile.ZipFile(OUT/'TDN-complete-experiment-atlas.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(OUT.iterdir()):
  if p.is_file() and p.suffix!='.zip':z.write(p,p.name)
print(json.dumps({'out':str(OUT),'experiments':dict(exp_counts),'checks':dict(check_counts),'junit':dict(jc),'files':{p.name:p.stat().st_size for p in OUT.iterdir() if p.is_file()}},indent=2))
