"""Plot retained R7 measurements only; no teacher acquisition or model execution."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((source/'metrics.json').read_text())
    frontier = json.loads((source/'work_precision.json').read_text())
    contexts = sorted({r['context_id'] for r in frontier})
    intervals = json.loads((source/'protocol.json').read_text())['intervals']
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for ax, context in zip(axes.flat, contexts):
        by_method = {r['method']:r for r in frontier if r['context_id']==context}
        rows = [r for r in metrics if r['context_id']==context and r['role']=='withheld']
        for prefix, color, label in [('ordered_', '#4676a8', 'Ordered 1/2/4/8'), ('lawson_', '#267b55', 'Lawson 2/4/8')]:
            values = sorted((r for m,r in by_method.items() if m.startswith(prefix)), key=lambda r:int(r['method'].split('_')[-1]))
            ax.plot([r['median_seconds']*1000 for r in values], [r['pooled_source_rms'] for r in values],
                    '-o', color=color, lw=1, ms=4, label=label)
        learned=by_method['fixed_feature_fitted']
        ax.plot(learned['median_seconds']*1000, learned['pooled_source_rms'], '^', color='#b94336', ms=8, label='Fitted response')
        local=[r['rms'] for r in rows if r['method']=='projection']
        ax.axhline(np.sqrt(np.mean(np.square(local))), color='#89659a', lw=1, ls='--', label='Offline projection error')
        refs=[r['reference_uncertainty_rms'] for r in rows if r['method']=='fixed_feature_fitted']
        uncertainty=np.sqrt(np.mean(np.square(refs)))
        ax.axhspan(1e-16, uncertainty, color='#aaaaaa', alpha=.18, label='Reference uncertainty estimate')
        ax.set_xscale('log');ax.set_yscale('log')
        ax.set_xlim(3.5,80);ax.set_ylim(1e-15,1e-5)
        index=int(context.rsplit('-',1)[-1]);s,t=intervals[index]
        ax.set_title(f's={s:.2f}, t={t:.2f}; lag={t-s:.2f}', fontsize=10)
        ax.set_xlabel('Complete four-source bundle, median ms')
        ax.set_ylabel('Pooled source RMS error')
        ax.grid(True,alpha=.18)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=9)
    fig.suptitle('Native R7: one diagnostic development parent, six contexts\nFive paired rounds; fresh setup per source; no utility threshold',fontsize=13)
    fig.savefig(output/'work-precision.png',dpi=170)
    fig.savefig(output/'work-precision.svg')
    plt.close(fig)

    methods=['ordered_2','fixed_feature_fitted','projection','ordered_8','lawson_2']
    labels=['Ordered 2','Global fit','Local projection','Ordered 8','Lawson 2']
    colors=['#89a6c5','#b94336','#89659a','#4676a8','#267b55']
    families=['dense_random','low_frequency','high_frequency','constant']
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for method,label,color in zip(methods,labels,colors):
        values=[]
        for family in families:
            rows=[r['rms'] for r in metrics if r['split']=='diagnostic' and r['role']=='withheld' and r['method']==method and r['family']==family]
            values.append(np.sqrt(np.mean(np.square(rows))))
        axes[0].plot(range(4),values,'-^' if method=='fixed_feature_fitted' else '-o',label=label,color=color,lw=1,ms=5)
    axes[0].set_xticks(range(4),['Random','Low frequency','High frequency','Constant'])
    axes[0].set_yscale('log');axes[0].set_ylabel('RMS pooled over six contexts')
    axes[0].set_title('Withheld-source families');axes[0].grid(True,alpha=.18)
    for method,label,color in [('fixed_feature_fitted','Global fit','#b94336'),('projection','Local projection','#89659a')]:
        values=[]
        for cid in contexts:
            values.append(sum(r['rms']**2 for r in metrics if r['context_id']==cid and r['role']=='withheld' and r['method']==method))
        values=np.array(values)/sum(values)*100
        offset=-.18 if method=='fixed_feature_fitted' else .18
        axes[1].bar(np.arange(6)+offset,values,width=.36,label=label,color=color)
    axes[1].set_xticks(range(6),[f'{s:.2f}→{t:.2f}' for s,t in intervals],rotation=25)
    axes[1].set_ylabel('Share of diagnostic withheld squared error (%)')
    axes[1].set_title('The longest interval dominates the pooled loss')
    axes[1].grid(True,axis='y',alpha=.18)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=5)
    fig.suptitle('R7 diagnostic accuracy: paired descendants of one inspected parent',fontsize=13)
    fig.savefig(output/'source-family-and-loss.png',dpi=170)
    fig.savefig(output/'source-family-and-loss.svg')
    plt.close(fig)
    for name in ('work-precision.svg', 'source-family-and-loss.svg'):
        path = output/name
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study');parser.add_argument('output')
    args=parser.parse_args();main(args.study,args.output)
