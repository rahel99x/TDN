#!/usr/bin/env python3
"""Rebuild the native Resolve overview from its small, fully traced data snapshot.

Run with --refresh-data in the original review workspace to re-extract measurements;
run without arguments beside plotted-data.json to render without the source archive.
Observed condition/seed ranges are descriptive, not confidence intervals.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent

def refresh_data():
    base = HERE.parents[2] / '.runtime' / 'resolve-native-analysis'
    sys.path.insert(0, str(base))
    import review_io as io
    evidence = io.index()['files']
    sources = {}
    def review(name):
        p = base / name
        raw = p.read_bytes()
        sources[name] = {'path': str(p.relative_to(HERE.parents[2])), 'sha256': hashlib.sha256(raw).hexdigest()}
        return json.loads(raw)
    fit = review('fit-review.json')
    fairness = review('fairness-review.json')
    memory = review('memory-review.json')
    review('kernels-review.json')
    for name in [*(v['source'] for v in fit.values()), *fairness['sources'], *memory['source_files']]:
        sources[name] = evidence[name]
    data = {'schema': 'tdn.resolve-review-figure/v1', 'sources': sources,
            'title': 'TDN Resolve: what the native run established',
            'evidence_status': 'Development evidence; no untouched confirmation cohort.',
            'fit': [], 'kernels': [], 'fairness': [], 'memory': []}
    fitmethods = ['fixed_unit', 'two_gain_direct', 'quad4_full', 'analytic_quad_cubic']
    for track in ['discrete', 'continuum']:
        for method in fitmethods:
            r = fit[track]['models'][method]
            data['fit'].append({'track': track, 'method': method, 'rms': r['mean_rms'],
                'fields': r['fields'], 'queries': r['queries'],
                'trace': f'fit-review.json/{track}/models/{method}/mean_rms'})
    for track in ['discrete', 'continuum']:
        for n in [64, 128]:
            name = f'kernels-{track}-n{n}/kernel_rows.json'
            sources[name] = evidence[name]
            rows = [r for r in io.load(name)['comparisons'] if r['candidate'] == 'quad4_batched']
            vals = [r['median_paired_speed_ratio'] for r in rows]
            assert len(vals) == 8
            data['kernels'].append({'track': track, 'grid': n, 'median': statistics.median(vals),
                'minimum': min(vals), 'maximum': max(vals), 'conditions': len(vals),
                'accuracy_qualified_conditions': sum(r['both_accuracy_eligible'] for r in rows),
                'max_parity_error': max(r['maximum_parity_error'] for r in rows),
                'paired_rounds_per_condition': [len(r['paired_round_speed_ratios']) for r in rows],
                'condition_measurements': [{k:r[k] for k in ['parent_id', 'schedule_steps',
                    'median_paired_speed_ratio', 'both_accuracy_eligible']} for r in rows],
                'trace': name + '/comparisons[candidate=quad4_batched]'})
    for r in fairness['model_summary']:
        data['fairness'].append({'track': r['track'], 'method': r['family'],
            'rms': r['validation']['field_mean_rms'], 'training_seeds': r['seeds'],
            'trace': f"fairness-review.json/model_summary[track={r['track']},family={r['family']}]/validation/field_mean_rms"})
    for r in memory['results']:
        data['memory'].append({k:r[k] for k in ['scenario', 'method', 'independent_parents',
            'endpoint_queries', 'native_median_paired_ratio'] } | {
            'trace': f"memory-review.json/results[scenario={r['scenario']},method={r['method']}]/native_median_paired_ratio"})
    data['scopes'] = {
        'fit': 'Each track: 32 validation fields, 4 horizons per field (128 queries); equal-field mean RMS. Fits use development training data. Point estimates; no interval shown.',
        'kernels': 'Each track/grid: 2 development regimes × 2 phase fields × 2 schedules = 8 conditions; 20 randomized paired warm timing rounds per condition. Point: median condition speedup; whisker: observed condition min–max, not CI. Same mathematical GL4 rule, complete-call cost. Reference track here is same-grid, temporally refined; no continuum spatial certification.',
        'fairness': 'Each track: the same 32 validation fields × 4 horizons, 3 training seeds. Validation-selected family checkpoints; point: median seed field-mean RMS; whisker: seed min–max, not CI. Frozen analytic controls repeat the same outputs across seeds. Adapted split FNO implementations, not published-paper benchmark claims.',
        'memory': '16 independent development test parents per scenario, 2 noise draws each (32 endpoint queries), 32 autonomous steps. Points: median paired endpoint RMS / coarse RK4 RMS. Fields reused across noise levels and reaction multipliers. No uncertainty interval shown; connecting lines only guide the eye. Reaction multipliers 1 and 3 correspond to rates 0.7 and 2.1.',
    }
    (HERE/'plotted-data.json').write_text(json.dumps(data, indent=2) + '\n')


def render():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter
    import numpy as np
    d=json.loads((HERE/'plotted-data.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':12, 'axes.labelsize':12,
        'axes.titlesize':13, 'axes.titleweight':'bold', 'axes.linewidth':.6, 'xtick.labelsize':11,
        'ytick.labelsize':11, 'savefig.facecolor':'white', 'pdf.fonttype':42,
        'lines.linewidth':.8, 'axes.spines.top':False, 'axes.spines.right':False})
    teal='#087F8C'; navy='#355E8D'; purple='#77589C'; gray='#657381'; orange='#CF6D32'; red='#B44056'
    fig=plt.figure(figsize=(18,14), facecolor='white')
    fig.text(.05,.967,d['title'],ha='left',va='top',fontsize=24,weight='bold',color='#182C3A')
    fig.text(.05,.930,'RTX 4090 native run  •  Four different research questions  •  Development evidence only',
             fontsize=13,color='#51616B')
    topgrid=fig.add_gridspec(1,2,left=.17,right=.96,bottom=.61,top=.82,wspace=.55)
    bottomgrid=fig.add_gridspec(1,2,left=.17,right=.96,bottom=.205,top=.405,wspace=.55)
    a=topgrid[0,0].subgridspec(1,2,wspace=.18)
    c=bottomgrid[0,0].subgridspec(1,2,wspace=.18)
    dd=bottomgrid[0,1].subgridspec(1,2,wspace=.19)
    def frame(ax):
        ax.grid(axis='x',color='#E5EAED',lw=.6,zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(length=3,width=.6)
        ax.spines['left'].set_color('#A8B2B8');ax.spines['bottom'].set_color('#A8B2B8')
    def head(x,y,title,subtitle):
        fig.text(x,y,title,fontsize=17,weight='bold',color='#203442')
        fig.text(x,y-.025,subtitle,fontsize=11.5,color='#51616B')
    head(.05,.885,'A  Calibration versus a stronger formula','Lower RMS is better · same fit cohort; controls kept separate')
    afamilies=[('fixed_unit','Analytic · fixed parent',gray,'s'),('two_gain_direct','Ours · direct two gains',teal,'^'),
               ('quad4_full','Analytic · GL4',navy,'s'),('analytic_quad_cubic','Analytic · quadratic + cubic',purple,'s')]
    for ti,track in enumerate(['discrete','continuum']):
        ax=fig.add_subplot(a[0,ti]);frame(ax)
        for i,(m,label,color,mark) in enumerate(afamilies):
            val=next(r['rms'] for r in d['fit'] if r['track']==track and r['method']==m)*1e6
            ax.plot(val,i,marker=mark,color=color,ms=7,ls='none')
            ax.annotate(f'{val:.2f}',(val,i),xytext=(7,0),textcoords='offset points',va='center',fontsize=10,color=color)
        ax.set(yticks=range(4),yticklabels=[x[1] for x in afamilies] if ti==0 else ['']*4,
               ylim=(3.55,-.55),xlim=(0,15),xticks=[0,5,10,15],xlabel='RMS × 10⁻⁶')
        ax.set_title('Discrete' if track=='discrete' else 'Continuum target',pad=13)
    fig.text(.05,.545,'32 validation fields × 4 horizons per track; equal-field mean RMS.\nSmall fitted gain versus the parent; GL4 and nested analytic terms remain stronger.',
             fontsize=10.5,linespacing=1.6,color='#51616B')
    head(.58,.885,'B  GL4 execution: batching the same rule','Higher speedup is better · complete warm calls, same output formula')
    ax=fig.add_subplot(topgrid[0,1]);frame(ax)
    for i,r in enumerate(d['kernels']):
        color=teal if r['track']=='discrete' else navy
        med,lo,hi=r['median'],r['minimum'],r['maximum']
        ax.errorbar(med,i,xerr=[[med-lo],[hi-med]],fmt='^',color=color,ms=8,lw=1,capsize=4,zorder=4)
        pts=[x['median_paired_speed_ratio'] for x in r['condition_measurements']]
        ax.scatter(pts,[i+.10]*len(pts),s=9,marker='^',color=color,alpha=.35,zorder=3)
        ax.text(2.59,i,f'{lo:.2f}–{hi:.2f}×',ha='right',va='center',fontsize=10,color=color)
    ax.axvline(1,color=gray,ls='--',lw=.8)
    ax.set(yticks=range(4),yticklabels=['Discrete · 64²','Discrete · 128²','Same-grid Galerkin · 64²','Same-grid Galerkin · 128²'],
           ylim=(3.55,-.55),xlim=(.92,2.63),xticks=[1,1.25,1.5,1.75,2,2.25],xlabel='Sequential GL4 time / Ours · batched GL4 time')
    fig.text(.58,.545,'8 conditions per row; 20 paired rounds per condition. Whiskers: observed min–max, not CI.\n30/32 conditions meet both accuracy targets; maximum parity difference ≤ 1.79 × 10⁻⁷.',
             fontsize=10.5,linespacing=1.6,color='#51616B')
    head(.05,.480,'C  Adapted FNO fairness checks','Lower RMS is better · validation-selected checkpoints, equal data')
    cfamilies=[('fno_scaled','Theirs · scaled split FNO',orange,'o'),('fno_mean','Theirs · FNO + mean head',red,'o'),
               ('two_basis','Ours · learned two basis',teal,'^'),('df','Analytic · DF baseline',gray,'s'),
               ('quad2_full','Analytic · GL2',purple,'s'),('quad4_full','Analytic · GL4',navy,'s')]
    for ti,track in enumerate(['discrete','continuum']):
        ax=fig.add_subplot(c[0,ti]);frame(ax)
        for i,(m,label,color,mark) in enumerate(cfamilies):
            vals=next(r['rms'] for r in d['fairness'] if r['track']==track and r['method']==m)
            med,lo,hi=[vals[k]*1e6 for k in ['median','minimum','maximum']]
            ax.errorbar(med,i,xerr=[[med-lo],[hi-med]],fmt=mark,color=color,ms=6,lw=.8,capsize=3)
        ax.set(yticks=range(6),yticklabels=[x[1] for x in cfamilies] if ti==0 else ['']*6,
               ylim=(5.55,-.55),xlim=(0,44),xticks=[0,10,20,30,40],xlabel='RMS × 10⁻⁶')
        ax.set_title('Discrete' if track=='discrete' else 'Continuum target',pad=13)
    fig.text(.05,.099,'32 validation fields × 4 horizons; 3 training seeds. Whiskers: seed range, not CI.\nAnalytic controls repeat identical outputs. These are adapted implementations,\nnot a reproduced published-FNO benchmark or untouched confirmation.',
             fontsize=10.5,linespacing=1.55,color='#51616B')
    head(.58,.480,'D  Memory: clean gains do not survive noise','Lower ratio is better · below 1 beats the coarse RK4 control')
    mfamilies=[('clean_fitted_memory','Ours · clean-fit memory',purple,'^'),
               ('noise_aware_fitted_memory','Ours · noise-aware memory',teal,'^'),
               ('plain_secant','Analytic · plain secant',gray,'s')]
    noise=['0','0.0001','0.001']
    for ti,mult in enumerate([1,3]):
        ax=fig.add_subplot(dd[0,ti]);frame(ax)
        ax.axhline(1,color='#A4ABB1',ls='--',lw=.8,zorder=2)
        for method,label,color,marker in mfamilies:
            vals=[next(r['native_median_paired_ratio'] for r in d['memory'] if r['scenario']==f'reaction-{mult}/noise-{n}' and r['method']==method) for n in noise]
            ax.plot(range(3),vals,marker=marker,color=color,ms=6,lw=.85,label=label)
        ax.set(yscale='log',ylim=(.075,100),xticks=[0,1,2],xticklabels=['0','10⁻⁴','10⁻³'],
               xlim=(-.16,2.16),xlabel='Noise standard deviation')
        ax.yaxis.set_major_locator(FixedLocator([.1,1,10,100]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:g}×'))
        ax.grid(axis='y',color='#E5EAED',lw=.6)
        ax.set_title(f'Reaction rate = {0.7*mult:.1f}',pad=13)
        if ti==0:ax.set_ylabel('Median paired endpoint RMS / coarse RMS')
        else:ax.tick_params(labelleft=False)
        if ti==1:ax.text(2.05,1.16,'1×',color=gray,fontsize=9,ha='right')
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='center',bbox_to_anchor=(.769,.127),ncol=1,frameon=False,fontsize=10,handlelength=1.8,labelspacing=.45)
    fig.text(.58,.058,'16 independent parents × 2 noise draws; 32-step autonomous rollout.\nCoarse RK4 and validation-selected zero damping both equal 1×. No interval shown.',
             fontsize=10,linespacing=1.5,color='#51616B')
    fig.text(.05,.018,'No universal leaderboard: panels use different tasks, cohorts and estimands. Raw values, denominators and SHA-256 source traces are in plotted-data.json.',
             fontsize=10,color='#51616B')
    for ext in ['png','pdf']:
        fig.savefig(HERE/f'resolve-native-overview.{ext}',dpi=300)
    plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--refresh-data',action='store_true');a=p.parse_args()
    if a.refresh_data:refresh_data()
    render()
