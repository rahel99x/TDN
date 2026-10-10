"""Separate source-bound atlas for the adjacent interaction study.

The engine supplies verified prerequisites. This module selects no checkpoints,
loads no model pickle and never treats completion as a scientific result. Raw
records, including failed and unresolved experiments, remain in chart data.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import html
import json
import math
import os
from pathlib import Path
import textwrap

from tdn.analysis.frontier.core import check, clean
from tdn.analysis.frontier.report import _learning_ranges
from tdn.analysis.portfolio.report import _read_source, _rows, _pointer

SCHEMA = 'tdn.adjacent-analytics/v1'
TABLES = ('experiments', 'checks', 'diagnostics', 'learning', 'evaluation', 'temporal',
          'exploration', 'roughness', 'parameters', 'profiles', 'claims', 'stage_status', 'trajectories', 'patterns')
SCOPE = ('Adjacent study only; no promotion into the main architecture. Computational completion, '
         'mathematical checks and scientific utility are separate outcomes. Independent parents are '
         'the statistical units; paired grids, phases, schedules and seeds are repeated observations. '
         'Observed-range bands are not confidence intervals. Missing evidence is NA.')
ROLE_GUIDE = ('Ours: triangles; Theirs: circles; analytic controls: squares. Thin line styles '
              'and colors distinguish methods. Local FNO controls are not paper reproductions.')


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _write(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')


def _gzip(path, value):
    with gzip.GzipFile(filename=str(path), mode='wb', mtime=0) as stream:
        stream.write(json.dumps(clean(value), separators=(',', ':'), allow_nan=False).encode())


def role(family, row=None):
    row = row or {}
    declared = str(row.get('role', row.get('method_role', row.get('ownership', ''))))
    if declared.startswith('Ours'): return 'Ours'
    if declared.startswith('Theirs'): return 'Theirs'
    if declared.lower().startswith(('analytic', 'oracle')): return 'Analytic control'
    name = str(family).lower()
    if any(word in name for word in ('oracle', 'fixed', 'quadratic', 'quad_cubic', 'cached_analytic', 'gauss', 'recombination', 'projection')):
        return 'Analytic control'
    if name in ('df', 'rf', 'etdrk4', 'backbone', 'dop853') or any(word in name for word in ('fno', 'wavelet', 'dense_output', 'classical')):
        return 'Theirs'
    if name.startswith(('quad2', 'quad4')) and not any(word in name for word in ('fit', 'condition', 'node', 'joint', 'linear', 'amplitude')):
        return 'Analytic control'
    return 'Ours'


def style(family, row=None):
    category = role(family, row)
    palettes = {'Ours': ('#0072B2', '#17A8AB', '#414CA1', '#7665A6'),
                'Theirs': ('#D55E00', '#B77700', '#BC5F92', '#666666'),
                'Analytic control': ('#008F70', '#739D42', '#8C6835', '#425A39')}
    code = int(hashlib.sha256(str(family).encode()).hexdigest()[:8], 16)
    return {'color': palettes[category][code % 4], 'marker': {'Ours': '^', 'Theirs': 'o', 'Analytic control': 's'}[category],
            'linestyle': ('-', '--', ':', '-.')[(code // 4) % 4], 'linewidth': .55}


def family(row):
    return str(row.get('family', row.get('method', row.get('basis', row.get('model_id', 'unlabeled')))))


def label(row):
    return f'{role(family(row), row)}: {family(row)}' + (f' [{row["track"]}]' if row.get('track') else '')


def flatten(row):
    """A derived plotting view; exact input remains untouched in the raw table."""
    result = {}
    for name in ('config', 'effective_config', 'metrics', 'cost', 'costs', 'timing', 'errors'):
        if isinstance(row.get(name), dict): result.update(row[name])
    result.update(row)
    aliases = {
        'rms': ('rms_error', 'residual_rms', 'endpoint_rms', 'upper_rms', 'error_rms'),
        'max_error': ('error_max', 'residual_max', 'endpoint_max', 'upper_max'),
        'seconds': ('median_seconds', 'cost_seconds', 'latency_seconds', 'measured_complete_seconds', 'complete_seconds', 'elapsed_seconds', 'total_seconds'),
        'n': ('grid_size', 'resolution', 'N'), 'h': ('step', 'timestep', 'step_size', 'h_max'),
        'T': ('final_time', 'horizon'), 'time': ('t', 'query_time'),
        'amplitude': ('epsilon',), 'alpha': ('roughness_alpha',),
        'loss': ('train_loss', 'training_loss'), 'validation_loss': ('val_loss',),
        'query_count': ('queries', 'Q'), 'gain': ('coefficient', 'optimal_gain', 'a_star'),
        'effective_rank': ('rank',), 'integrated_rms': ('integrated_error',),
        'coefficient_abs_error': ('absolute_coefficient_error',),
        'parameters': ('parameter_count',), 'quadrature_error': ('quadrature_error_rms',),
    }
    for name, candidates in aliases.items():
        if name not in result:
            result[name] = next((result[key] for key in candidates if key in result), None)
    if result.get('n') is None and isinstance(result.get('grid'), (list, tuple)) and result['grid']:
        result['n'] = result['grid'][0]
    if isinstance(result.get('oracle'), dict):
        for key in ('effective_rank', 'condition_number', 'coefficient_refinement_sensitivity_l2', 'coefficient_uncertainty_radius_l2'):
            if key in result['oracle']: result[key] = result['oracle'][key]
    interval = result.get('acceptable_gain_interval')
    if isinstance(interval, list) and len(interval) == 2:
        result['gain_interval_low'], result['gain_interval_high'] = interval
    for name, target in (('coefficient_reference', 'target'), ('coefficient_predicted', 'predicted')):
        if isinstance(result.get(name), dict):
            result[target + '_real'] = result[name].get('real')
            result[target + '_imag'] = result[name].get('imag')
    if finite(result.get('shared_error_rms')) and finite(result.get('separate_error_rms')):
        result['shared_response_penalty'] = result['shared_error_rms'] - result['separate_error_rms']
    if result.get('feature_distance') is None and isinstance(result.get('feature_differences'), (list, dict)):
        differences = list(result['feature_differences'].values()) if isinstance(result['feature_differences'], dict) else result['feature_differences']
        if isinstance(result['feature_differences'], dict) and finite(result['feature_differences'].get('original')):
            result['feature_distance'] = abs(result['feature_differences']['original'])
        elif differences and all(finite(v) for v in differences): result['feature_distance'] = max(abs(v) for v in differences)
    return result


def _source(row, unit, path, sha, pointer, index=None):
    return {**row, '_source': {'unit': unit, 'path': str(path), 'sha256': sha,
        'pointer': pointer, 'line': index + 1 if index is not None and path.suffix == '.jsonl' else None,
        'evidence_status': 'VERIFIED_PREREQUISITE'}}


def _table(filename, unit):
    if filename == 'rows.jsonl': return 'experiments'
    if 'learning' in filename or 'loss' in filename: return 'learning'
    if 'catalog' in filename or 'parameter' in filename or 'response' in filename: return 'parameters'
    if 'claim' in filename or 'primary-comparison' in filename or 'confirmation-plan' in filename: return 'claims'
    if 'profile' in filename or 'timing' in filename: return 'profiles'
    if unit == 'roughness' or filename.startswith('roughness'): return 'roughness'
    if unit == 'D07' or filename.startswith('temporal'): return 'temporal'
    if unit.startswith('explore-'): return 'exploration'
    if unit.startswith(('pilot', 'evaluate')) or any(word in filename for word in ('endpoint', 'validation', 'evaluation', 'comparison')): return 'evaluation'
    return 'diagnostics'


def collect(ctx):
    data = {name: [] for name in TABLES}
    data.update(schema=SCHEMA, profile=ctx.protocol.get('profile'), protocol=ctx.protocol,
                device=str(ctx.device), sources=[], auxiliary=[], omissions=[])
    units = ctx.protocol.get('units', {})
    for unit in sorted(set(units) | set(ctx.prerequisites)):
        if unit == ctx.stage: continue
        base = ctx.prerequisites.get(unit)
        failure = getattr(ctx, 'stage_failures', {}).get(unit, {})
        status = {'unit': unit, 'verified': bool(base), 'status': 'VERIFIED' if base else failure.get('status', 'NA'),
                  'planned_seconds': units.get(unit, {}).get('seconds'), 'elapsed_seconds': failure.get('elapsed_seconds'),
                  'scientific_outcome': 'NA', 'math_outcome': 'NA', 'monetary_cost': None,
                  'energy_joules': None, 'error': failure.get('error')}
        data['stage_status'].append(status)
        if not base: continue
        base = Path(base)
        for path in sorted([*base.glob('*.json'), *base.glob('*.jsonl')]):
            if path.name in ('science_manifest.json', 'execution.json', 'workflow-seal.json', 'stage.json', 'protocol.json', 'rows.json'):
                continue
            ctx.budget.check()
            value, sha, size = _read_source(path)
            provenance = {'unit': unit, 'path': str(path), 'sha256': sha, 'bytes': size}
            data['sources'].append(provenance)
            if path.name == 'summary.json':
                status.update(elapsed_seconds=value.get('elapsed_seconds'), scientific_outcome=value.get('scientific_outcome', 'NA'),
                              computational_status=value.get('status'), math_outcome=value.get('math_outcome', 'NA'))
            rows = _rows(value)
            table = _table(path.name, unit)
            if rows:
                pointer = _pointer(value)
                data[table].extend(_source(row, unit, path, sha, f'{pointer}/{i}', i) for i, row in enumerate(rows))
            else:
                data['auxiliary'].append({**provenance, 'value': value})
                # A single measurement dictionary is a valid record, while pure
                # metadata remains auxiliary. Do not manufacture numeric values.
                if isinstance(value, dict) and any(k in value for k in ('metrics', 'method', 'family', 'break_even_queries', 'diagnostic')):
                    data[table].append(_source(value, unit, path, sha, ''))
            # Preserve named tables independently instead of discarding siblings
            # when an artifact contains more than one list of observations.
            if isinstance(value, dict):
                first = _pointer(value).lstrip('/')
                for key, items in value.items():
                    if key == first or not isinstance(items, list) or not items or not all(isinstance(r, dict) for r in items): continue
                    subtable = _table(key + '.json', unit)
                    data[subtable].extend(_source(row, unit, path, sha, f'/{key}/{i}') for i, row in enumerate(items))
    for row in list(data['parameters']):
        for i, probe in enumerate(row.get('response_probes', [])):
            report = probe.get('report', {})
            response = report.get('response', {})
            gains = response.get('gain', [])
            if gains and isinstance(gains[0], list): gains = gains[0]
            for channel, gain in enumerate(gains):
                data['parameters'].append({'family':row.get('family'), 'role':row.get('role'), 'track':row.get('track'), 'model_id':row.get('model_id'),
                    'h':probe.get('horizon'), 'gain':gain, 'channel':channel, 'split':probe.get('split'),
                    '_source':{**row['_source'], 'pointer':row['_source']['pointer']+f'/response_probes/{i}/report/response/gain'}})
    for row in data['experiments']:
        for i, item in enumerate(row.get('checks', [])):
            data['checks'].append({**item, 'experiment_id': row.get('experiment_id'),
                'stage': row.get('stage'), '_source': {**row['_source'], 'pointer': row['_source']['pointer'] + f'/checks/{i}'}})
    data['trajectories'] = trajectory_rows(data['diagnostics'])
    data['patterns'] = pattern_rows(data['diagnostics'], data['sources'])
    return data


def trajectory_rows(rows):
    result = []
    for row in rows:
        if row.get('diagnostic') != 'D06' or row.get('category') != 'trajectory': continue
        times = row.get('times', [])
        actual, reference = row.get('autonomous_observables', {}), row.get('reference_observables', {})
        for i, time in enumerate(times):
            item = {key: row.get(key) for key in ('method','method_role','parent_id','track','schedule_name','T','initial_perturbation')}
            item.update(time=time, series_id=row.get('case_id'), diagnostic='D06', _source={**row['_source'], 'derived_from':[f"{row['_source']['pointer']}/times/{i}", f"{row['_source']['pointer']}/trajectory_error_rms/{i}"]})
            for source, target in (('trajectory_error_rms','rms'), ('trajectory_error_max','max_error')):
                values = row.get(source, []); item[target] = values[i] if i < len(values) else None
            for observable in ('mean','variance'):
                a, b = actual.get(observable, []), reference.get(observable, [])
                item[observable] = a[i] if i < len(a) else None
                item[observable+'_error'] = abs(a[i]-b[i]) if i < min(len(a),len(b)) and finite(a[i]) and finite(b[i]) else None
            result.append(item)
    return result


def pattern_rows(rows, sources):
    """Predeclared representative high-pair D01 patterns, exact NPZ references.

    Raw source NPZs retain all fields/methods. The chart stores selected arrays
    explicitly, including their original complex spectra via real/imag values.
    """
    import numpy as np
    candidates = [r for r in rows if r.get('diagnostic')=='D01' and r.get('array_prefix') and r.get('regime')=='high_pair']
    selected_methods = ('df','quad2_fixed','scalar_bounded_oracle','interaction_channel_bounded_oracle','quadratic_cubic_bounded_oracle','analytic_quad_cubic')
    chosen = [r for r in candidates if r.get('method') in selected_methods]
    if not chosen: chosen = candidates[:6]
    output=[]; cache={}
    for row in chosen:
        root=Path(row['_source']['path']).parent
        paths=list(root.glob('*.npz'))
        for path in paths:
            if path not in cache:
                if path.is_symlink() or path.stat().st_size > 512 << 20: raise ValueError('Unbounded or symlink pattern source')
                before=path.stat();raw=path.read_bytes();after=path.stat()
                if (before.st_ino,before.st_mtime_ns,before.st_size)!=(after.st_ino,after.st_mtime_ns,after.st_size): raise ValueError('Pattern source changed while reading')
                import io
                with np.load(io.BytesIO(raw),allow_pickle=False) as bundle:
                    cache[path]=({key:bundle[key] for key in bundle.files},hashlib.sha256(raw).hexdigest())
                sources.append({'unit':row['_source']['unit'],'path':str(path),'sha256':cache[path][1],'bytes':len(raw)})
            arrays,sha=cache[path];key=row['array_prefix']+'__residual'
            if key not in arrays: continue
            residual=np.asarray(arrays[key]).squeeze();spectral_key=row['array_prefix']+'__residual_spectrum'
            if residual.ndim != 2: continue
            spectrum=arrays.get(spectral_key)
            output.append({'method':row['method'],'method_role':row.get('method_role'),'parent_id':row.get('parent_id'),'track':row['track'],
                'residual':residual.tolist(),'spectrum_real':np.asarray(spectrum).squeeze().real.tolist() if spectrum is not None else None,
                'spectrum_imag':np.asarray(spectrum).squeeze().imag.tolist() if spectrum is not None else None,
                '_source':{'unit':row['_source']['unit'],'path':str(path),'sha256':sha,'npz_keys':[key,spectral_key],'row_source':row['_source']}})
            break
    return output


def temporal_view(rows):
    values=[flatten(r) for r in rows]
    controls={}
    for row in values:
        if row.get('method') in ('direct_gl4','uncached_pairs'):
            controls[(row.get('task'),row.get('track'),row.get('workload'),row.get('query_count'))]=row
    for row in values:
        control=controls.get((row.get('task'),row.get('track'),row.get('workload'),row.get('query_count')))
        if control and finite(row.get('seconds')) and row['seconds']>0 and finite(control.get('seconds')):
            row['measured_speedup_over_direct']=control['seconds']/row['seconds']
            row['_derived_comparator_source']=control.get('_source')
    return values


def roughness_view(rows):
    result=[]
    for source in rows:
        row=flatten(source); rough=row.get('roughness',row.get('metrics',{}))
        if not isinstance(rough,dict): rough={}
        row['interaction_strength']=rough.get('nodal_quadratic_interaction_rms', row.get('interaction_strength'))
        for estimate in rough.get('roughness_estimates',[]):
            result.append({**row,'empirical_hurst':estimate.get('empirical_H'),'axis':estimate.get('axis')})
        if not rough.get('roughness_estimates'): result.append(row)
    return result


def _cell(value):
    if isinstance(value, (dict, list)): value = json.dumps(clean(value), sort_keys=True, separators=(',', ':'))
    if isinstance(value, str) and value.lstrip().startswith(('=', '+', '-', '@')): return "'" + value
    return value


def write_tables(data, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    inventory = []
    for table in TABLES:
        rows = data.get(table, [])
        columns = sorted({key for row in rows for key in row}) or ['status', 'reason']
        path = output / (table + '.csv')
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, columns, lineterminator='\n'); writer.writeheader()
            for row in rows: writer.writerow({key: _cell(row.get(key)) for key in columns})
        inventory.append({'table': table, 'path': path.name, 'rows': len(rows), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    return inventory


def _blank(ax, reason='NA: no verified measurements for this panel'):
    ax.text(.5, .5, textwrap.fill(reason, 75), ha='center', va='center', transform=ax.transAxes, color='#657383')
    ax.set_xticks([]); ax.set_yticks([])


def _scatter(ax, rows, x, y, *, connect=False, log=True):
    groups = defaultdict(list); missing = 0
    for original in rows:
        row = flatten(original)
        if not finite(row.get(x)):
            missing += 1; continue
        if not finite(row.get(y)):
            missing += 1
            if not connect: continue
        # Connecting a different field, track, grid or schedule implies a curve
        # that was not measured. These identities are never pooled into lines.
        key = (family(row), row.get('track'), row.get('parent_id', row.get('field_cluster')),
               row.get('model_id'), row.get('schedule_id', row.get('schedule_name')), row.get('seed'),
               row.get('series_id'), row.get('workload'), row.get('task'),
               row.get('n') if x != 'n' else None, row.get('h') if x != 'h' else None,
               row.get('T') if x not in ('T', 'time') else None)
        groups[key].append(row)
    if not groups: _blank(ax); return
    valid = [r for rows_ in groups.values() for r in rows_ if finite(r.get(y))]
    if not valid: _blank(ax); return
    labeled = set()
    for key, values in sorted(groups.items(), key=lambda item: str(item[0])):
        values.sort(key=lambda r: r[x]); observed = [r for r in values if finite(r.get(y))]
        if not observed: continue
        sty = style(key[0], values[0]); name = label(values[0])
        display = name if name not in labeled else None; labeled.add(name)
        if connect:
            # Duplicate abscissae are separate paired observations, not one
            # oscillating path through unrelated experiments.
            if len({r[x] for r in values}) != len(values):
                ax.scatter([r[x] for r in observed], [r[y] for r in observed], color=sty['color'], marker=sty['marker'], s=14, alpha=.6, label=display)
            else:
                ax.plot([r[x] for r in values], [r[y] if finite(r.get(y)) else float('nan') for r in values], **sty, markersize=3, alpha=.8, label=display)
        else:
            ax.scatter([r[x] for r in values], [r[y] for r in values], color=sty['color'], marker=sty['marker'], s=15, alpha=.6, label=display, rasterized=True)
    if log and all(r[x] > 0 for r in valid): ax.set_xscale('log')
    if log and all(r[y] > 0 for r in valid): ax.set_yscale('log')
    if x.startswith('target_') and y.startswith('predicted_'):
        lo=min(min(r[x],r[y]) for r in valid);hi=max(max(r[x],r[y]) for r in valid)
        ax.plot([lo,hi],[lo,hi],color='#999999',linewidth=.5,linestyle=':',label='Exact agreement')
    ax.set_xlabel(x); ax.set_ylabel(y); ax.grid(alpha=.15)
    ax.text(.99, 1.015, f'{len(valid)} observed rows; {missing} unavailable', ha='right', fontsize=6, transform=ax.transAxes)


def _categorical(ax, rows, metric):
    valid = [flatten(row) for row in rows if finite(flatten(row).get(metric))]
    if not valid: _blank(ax); return
    methods = sorted({family(row) for row in valid})
    tracks=sorted({str(r.get('track','unspecified')) for r in valid})
    for name in methods:
        for track_index,track in enumerate(tracks):
            selected=[r for r in valid if family(r)==name and str(r.get('track','unspecified'))==track]
            if not selected:continue
            sty=style(name,selected[0]);offset=(track_index-(len(tracks)-1)/2)*.22
            ax.scatter([methods.index(name)+offset]*len(selected),[r[metric] for r in selected],
                       marker=sty['marker'],color=sty['color'],s=16,alpha=.6,label=label(selected[0]))
    ax.set_xticks(range(len(methods)), methods, rotation=35, ha='right', fontsize=7)
    ax.set_ylabel(metric + ' (lower is better)'); ax.grid(axis='y', alpha=.15)
    if all(r[metric] > 0 for r in valid): ax.set_yscale('log')


def range_plot(ax, records):
    """Continuous low/high observed envelopes; zero-count windows create gaps."""
    groups = defaultdict(list)
    for row in records:
        key = (row['family'], row.get('track'), row.get('phase'), row.get('train_count'))
        groups[key].append(row)
    found = False
    for key, values in sorted(groups.items(), key=lambda item: str(item[0])):
        values.sort(key=lambda r: r['window_index']); sty = style(key[0]); run = []
        def draw(segment):
            if not segment: return
            bounds = [(max(r['window_left'], r['family_x_min']), min(r['window_right'], r['family_x_max'])) for r in segment]
            xs = [bounds[0][0], *[(lo + hi) / 2 for lo, hi in bounds], bounds[-1][1]]
            lows = [segment[0]['y_min'], *[r['y_min'] for r in segment], segment[-1]['y_min']]
            highs = [segment[0]['y_max'], *[r['y_max'] for r in segment], segment[-1]['y_max']]
            ax.fill_between(xs, lows, highs, color=sty['color'], alpha=.10, linewidth=0)
            for boundary in (lows, highs): ax.plot(xs, boundary, color=sty['color'], linestyle=sty['linestyle'], linewidth=.55, alpha=.95)
        for row in values:
            if not row['observation_count']:
                draw(run); run = []; continue
            found = True
            if row['distinct_x_count'] <= 1:
                draw(run); run = []
                ax.vlines(row['x_min'], row['y_min'], row['y_max'], color=sty['color'], linewidth=.55)
                ax.plot([row['x_min']], [row['y_min']], linestyle='None', marker=sty['marker'], color=sty['color'], markersize=3)
            else:
                if run and row['window_index'] != run[-1]['window_index'] + 1: draw(run); run = []
                run.append(row)
        draw(run)
        if any(r['observation_count'] for r in values):
            ax.plot([], [], **sty, markersize=3, label=label({'family': key[0]}) + f' [{key[1]}, {key[2]}]')
    if not found: _blank(ax)
    ax.set_xlabel('Optimizer update'); ax.set_ylabel('Observed loss range (lower is better)'); ax.grid(alpha=.15)


def _layout(fig, title, guide, scope):
    handles = {}
    for ax in fig.axes:
        for handle, name in zip(*ax.get_legend_handles_labels()): handles.setdefault(name, handle)
        if ax.get_legend() is not None: ax.get_legend().remove()
    width, height = fig.get_size_inches(); wrap = max(80, int(width * 13))
    footer = '\n'.join(textwrap.fill(prefix + text, wrap, break_long_words=False, break_on_hyphens=False)
        for prefix, text in (('READ: ', guide), ('METHODS: ', ROLE_GUIDE), ('SCOPE: ', scope)))
    footer_height = (len(footer.splitlines()) * 10 + 22) / 72
    columns = 3; legend_height = (math.ceil(len(handles) / columns) * 14 + 18) / 72 if handles else 0
    total = height + footer_height + legend_height; fig.set_size_inches(width, total)
    fig.text(.025, .10 / total, footer, ha='left', va='bottom', fontsize=7, linespacing=1.25)
    if handles:
        fig.legend(list(handles.values()), list(handles), loc='lower center', bbox_to_anchor=(.5, (footer_height + .04) / total), ncol=columns, fontsize=7, frameon=False)
    fig.suptitle(title, fontsize=13); fig.tight_layout(rect=(.025, (footer_height + legend_height) / total, .985, .945))


def _diagnostic(data, name):
    return [r for r in data['diagnostics'] + data['experiments'] if r.get('diagnostic', r.get('program', r.get('stage', r.get('_source', {}).get('unit')))) == name]


def _architecture(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    boxes = [(0.07, .75, 'Full state u = m + v\nphysical parameters, h'), (.36, .75, 'Analytic backbone\nS_h(u)'),
             (.07, .38, 'Split v into v_L, v_H\nretain signed phases'), (.39, .38, 'Fixed-background channels\nC_LL, C_LH, C_HH'),
             (.75, .75, 'Global features\nfitted or learned gains'), (.75, .38, 'Σ g_c C_c\nthen output P_K'),
             (.73, .07, 'S_h(u) + P_K Σ g_c C_c\nnew state or same-state query')]
    for x, y, text in boxes:
        ax.text(x, y, text, ha='center', va='center', fontsize=9, bbox={'boxstyle': 'round,pad=.5', 'fc': '#f0f5f9', 'ec': '#698597', 'lw': .7})
    for a, b in (((.07,.65),(.07,.48)),((.17,.38),(.25,.38)),((.54,.38),(.61,.38)),((.75,.65),(.75,.48)),((.75,.28),(.75,.16)),((.40,.66),(.68,.13)),((.17,.75),(.26,.75))):
        ax.annotate('', xy=b, xytext=a, arrowprops={'arrowstyle':'->','lw':.65,'color':'#4d6674'})


def build_figures(data, output, budget=None):
    # Also protect direct helper use outside the workflow storage bootstrap.
    root=Path(__file__).resolve().parents[3]
    for variable, relative in (('MPLCONFIGDIR','.runtime/matplotlib'), ('XDG_CACHE_HOME','.cache')):
        location=Path(os.environ.setdefault(variable,str(root/relative)))
        location.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    _gzip(output / 'chart-data.json.gz', data)
    panels = []; ranges = []
    with PdfPages(output / 'adjacent-atlas.pdf') as pdf:
        def page(title, draw, guide, scope=SCOPE):
            if budget is not None: budget.check()
            fig, ax = plt.subplots(figsize=(13, 5.4)); draw(ax); _layout(fig, title, guide, scope)
            name = f'{len(panels)+1:02d}-' + ''.join(c.lower() if c.isalnum() else '-' for c in title).strip('-') + '.png'
            fig.savefig(output / name, dpi=180); pdf.savefig(fig); plt.close(fig)
            panels.append({'title': title, 'path': name, 'reading_guide': guide, 'scope': scope,
                           'sha256': hashlib.sha256((output / name).read_bytes()).hexdigest()})
        page('Architecture and information flow', _architecture,
             'Arrows denote data flow, not better/worse. Input-origin channels are not a third spatial dimension. Equal unit gains reproduce normalized quadratic correction.',
             'Fixed physical background and operator shared by every channel. Learning receives features; the physical branch still receives the full field. Promotion into main TDN requires a separate decision.')
        def stages(ax):
            rows = data['stage_status']; values = [r.get('elapsed_seconds') for r in rows]
            if not any(finite(v) for v in values): _blank(ax); return
            for i, (row, value) in enumerate(zip(rows, values)):
                if finite(value): ax.bar(i, value, color='#268F84' if row['verified'] else '#AF7770')
                else: ax.text(i, 0, 'NA', rotation=90, fontsize=6)
            ax.set_xticks(range(len(rows)), [r['unit'] for r in rows], rotation=55, ha='right', fontsize=7)
            ax.set_ylabel('Recorded stage seconds (lower for equal work)')
        page('Stage completion and observed compute', stages,
             'Y lower is less stage time for equal completed work. A short failed stage is not a scientific win. Missing measurements are NA; walltime ceilings are not consumed costs.')
        for pattern in data.get('patterns', []):
            import numpy as np
            def spatial(ax, pattern=pattern):
                arr=np.asarray(pattern['residual']);image=ax.imshow(arr,cmap='coolwarm',vmin=-max(abs(arr.min()),abs(arr.max()),1e-30),vmax=max(abs(arr.min()),abs(arr.max()),1e-30),origin='lower')
                ax.figure.colorbar(image,ax=ax,label='Signed residual; absolute distance from zero is worse')
                ax.set_xlabel('Grid index');ax.set_ylabel('Grid index')
            page('D01 Spatial residual '+label(pattern),spatial,'White/zero is better; larger absolute red or blue residual is worse. Spatial indices have no favorable direction. This is the predeclared representative high-pair parent, not a pooled comparison.')
            def spectral(ax, pattern=pattern):
                if pattern['spectrum_real'] is None: _blank(ax); return
                z=np.asarray(pattern['spectrum_real'])+1j*np.asarray(pattern['spectrum_imag'])
                image=ax.imshow(np.log10(np.maximum(np.abs(np.fft.fftshift(z)),1e-16)),origin='lower',cmap='viridis')
                ax.figure.colorbar(image,ax=ax,label='log10 coefficient magnitude (lower is better)')
                ax.set_xlabel('Shifted Fourier index');ax.set_ylabel('Shifted Fourier index')
            page('D01 Spectral residual '+label(pattern),spectral,'Lower color values mean smaller residual Fourier magnitude; a 1e-16 display floor is explicit. Signed complex coefficients remain in raw chart data and NPZ sources.')
        def outcomes(ax):
            groups = defaultdict(Counter)
            for row in data['checks']: groups[row.get('category', 'unspecified')][row.get('verdict', 'NA')] += 1
            if not groups: _blank(ax); return
            bottom = [0] * len(groups)
            for verdict, color in (('GOOD','#239B86'),('BAD','#C16F63'),('NA','#B5BDC4')):
                values = [group[verdict] for group in groups.values()]
                ax.bar(range(len(groups)), values, bottom=bottom, color=color, label=verdict)
                bottom = [a+b for a,b in zip(bottom,values)]
            ax.set_xticks(range(len(groups)), list(groups)); ax.set_ylabel('Check records (counts are not effect size)')
        page('Mathematics gap and utility outcomes', outcomes,
             'GOOD meets a declared check; BAD fails it; NA is unresolved. More passing checks do not prove a theorem, improve statistical precision or establish a model advantage.')
        specs = [
            ('D01 Scalar and channel correction floors', 'D01', 'seconds', 'rms', False, 'Lower-left is favorable only for matched targets. Oracle projections are reference-informed floors and cannot be deployed.'),
            ('D01 Residual RMS and maximum error', 'D01', 'rms', 'max_error', False, 'Lower on each axis is better. A small RMS does not imply small worst-cell error; compare the same parent, equation, h, nodes and cutoff.'),
            ('D01 Coefficient sensitivity', 'D01', 'condition_number', 'coefficient_refinement_sensitivity_l2', False, 'Lower Y means fitted coefficients change less under reference refinement. Ill-conditioning can make gains unidentifiable even with a small residual.'),
            ('D01 Basis conditioning and effective rank', 'D01', 'effective_rank', 'condition_number', False, 'Larger rank expands identifiable response directions. Lower condition number is better numerical conditioning; rank alone is not solver quality.'),
            ('D02 Feature collisions and response ambiguity', 'D02', 'feature_distance', 'shared_response_penalty', False, 'A small X with materially positive Y supports insufficient conditioning information only beyond reference uncertainty and flat-response sensitivity.'),
            ('D02 Acceptable gain intervals', 'D02', 'gain_interval_low', 'gain_interval_high', False, 'Intervals are acceptable objective ranges, not confidence intervals. Separated optima alone are insufficient; interval incompatibility and excess error matter.'),
            ('D03 Grid refinement at fixed workload', 'D03', 'n', 'rms', True, 'Y lower is better; X finer grid costs more. Follow only paired parents with fixed physical bandwidth. Added scales change the workload.'),
            ('D03 Timestep refinement', 'D03', 'h', 'rms', True, 'Y lower is better; X smaller step generally costs more. Amplitude, grid and endpoint must stay fixed to interpret temporal order.'),
            ('D03 Duration and complete solve error', 'D03', 'T', 'rms', True, 'Y lower is better at the same T. A later endpoint near equilibrium may hide earlier errors. Every exact step sequence remains in raw data.'),
            ('D03 Accuracy cost observations', 'D03', 'seconds', 'rms', False, 'Lower-left is favorable within the same track, field, grid, T and target. Unqualified latency is not time to an accurate solution.'),
            ('D04 Signed interaction coefficient error', 'D04', 'target_real', 'predicted_real', False, 'Agreement with the diagonal is better; sign matters. Neither larger X nor larger Y alone is favorable. Near-zero targets require absolute error.'),
            ('D04 Signed imaginary interaction coefficient', 'D04', 'target_imag', 'predicted_imag', False, 'Agreement with the diagonal is better; imaginary signs carry phase. Larger magnitude alone is neither better nor worse.'),
            ('D04 Phase and coefficient magnitude error', 'D04', 'coefficient_abs_error', 'phase_error', False, 'Lower on both axes is better. Phase at vanishing amplitude is unresolved; nodal and dealiased products are different target equations.'),
            ('D05 Fluctuation amplitude order', 'D05', 'amplitude', 'rms', True, 'Y lower is better; X varies perturbation amplitude at fixed h. Slopes are meaningful only above uncertainty and roundoff; this is not temporal order.'),
            ('D05 Quadrature convergence', 'D05', 'nodes', 'quadrature_error', True, 'Y lower is better; X more nodes adds work. Refining a quadratic integral does not introduce a missing cubic spatial direction.'),
            ('D05 Cubic spatial direction', 'D05', 'amplitude', 'cubic_perpendicular_fraction', False, 'Y closer to one means a larger cubic component orthogonal to the quadratic direction. That is new spatial information, not automatic accuracy or speed utility.'),
            ('D06 Autonomous trajectory RMS', 'D06', 'time', 'rms', True, 'Y lower is better throughout the trajectory. Own-output rollouts and teacher-forced one-step measurements are distinct; final equilibrium is insufficient.'),
            ('D06 Worst cell trajectory error', 'D06', 'time', 'max_error', True, 'Lower Y is better. Maximum error can regress despite lower RMS; zero-time exactness is retained.'),
            ('D06 Variance trajectory error', 'D06', 'time', 'variance_error', True, 'Lower absolute variance error is better at each matched time. Diffusive smoothing can erase initial roughness and hide transient defects.'),
            ('D06 Mean variance and physical range', 'D06', 'time', 'mean_error', True, 'Smaller absolute mean error is better. Logistic reaction does not conserve mean. Variance, range violations and late absolute error remain raw metrics.'),
            ('D06 Threshold crossing times', 'D06', 'reference_threshold_crossing_time', 'threshold_crossing_time', False, 'Agreement with the diagonal is better. Neither early nor late crossing is automatically accurate. Missing crossings remain unresolved.'),
            ('D06 Integrated error and stability', 'D06', 'T', 'integrated_rms', False, 'Y lower is better over the same physical interval. Stable and accurate, stable but inaccurate, and unstable cases require separate denominators.'),
            ('D08 Competitor favorable regimes', 'D08', 'seconds', 'rms', False, 'Lower-left is favorable within a matched regime. Smooth, weak-interaction and small-step workloads stay visible; short undertraining cannot establish neural inferiority.'),
        ]
        for title, diagnostic, x, y, connect, guide in specs:
            rows = data['trajectories'] if diagnostic == 'D06' and x == 'time' else _diagnostic(data, diagnostic)
            if title == 'D01 Scalar and channel correction floors':
                page(title, lambda ax: _categorical(ax, rows, 'rms'), 'Y lower is a smaller correction-space error floor. X lists methods, not a ranking. Oracle fits use references and are not deployable. Values below reference uncertainty are unresolved. Compare identical field/operator/h/nodes/cutoff; offset groups distinguish tracks.')
            else:
                page(title, lambda ax, rows=rows, x=x, y=y, connect=connect: _scatter(ax,rows,x,y,connect=connect), guide)
        for title, table, x, y, guide in (
            ('D07 Actual multi query costs','temporal','query_count','seconds','Lower Y is better for the same accurate query workload. Compare measured totals with build plus Q times query models; caching must be equally available.'),
            ('D07 Encoding refresh and break even','temporal','query_count','measured_speedup_over_direct','Y greater than one favors the named reusable representation. No positive query saving means no finite break-even. Refresh and operator preparation are charged.'),
            ('Roughness approximation and empirical scaling','roughness','alpha','empirical_hurst','X is a generator label, not a neural-network dimension. Empirical H is descriptive over the recorded scale range; finite trigonometric fields are smooth.'),
            ('Roughness accuracy and nonlinear interaction','roughness','alpha','rms','Lower Y is better at fixed mean, amplitude, bandwidth, grid and horizon. Alpha=1 is a finite-bandwidth stress endpoint, not the convergent ridge theorem.'),
            ('Roughness spectrum and runtime','roughness','interaction_strength','seconds','Lower Y means cheaper complete work; X interaction strength has no universal favorable direction. Matched spectral power does not match signed phase interactions.'),
            ('Pilot accuracy cost comparisons','evaluation','seconds','rms','Lower-left is favorable only at matched accuracy definitions and physical tasks. These are development pilots unless a fresh locked confirmation is explicitly supplied.'),
            ('Pilot gain responses','parameters','h','gain','X is step size; neither larger nor smaller gain is inherently better. Inspect response identifiability and sensitivity before attributing physical meaning.'),
            ('Pilot model size and error','evaluation','parameters','rms','Lower-left is compact and accurate; parameter count is not measured compute or a fairness definition. Fixed and initialized selections remain visible.'),
            ('Peak measured memory','profiles','n','peak_reserved_bytes','Lower Y uses less GPU memory for matched batch, precision and work. Host RAM is not VRAM; missing GPU measurements remain NA.'),
            ('Exploratory alternatives','exploration','seconds','rms','Lower-left is favorable for matched targets. A successful mathematical prototype is not a complete solver or evidence of useful amortization.'),
        ):
            rows = temporal_view(data[table]) if table == 'temporal' else roughness_view(data[table]) if table == 'roughness' else data[table]
            page(title, lambda ax, rows=rows, x=x, y=y: _scatter(ax, rows, x, y), guide)
        def stability(ax):
            rows=[r for r in _diagnostic(data,'D06') if r.get('category')=='rollout_summary']
            counts=Counter(r.get('classification','unresolved') for r in rows)
            if not counts: _blank(ax); return
            ax.bar(range(len(counts)),list(counts.values()),color=['#279D89' if 'accurate_stable'==k else '#BE7768' if 'unstable'==k else '#9DA9B2' for k in counts])
            ax.set_xticks(range(len(counts)),list(counts),rotation=20,ha='right');ax.set_ylabel('Paired rollout records; not independent parents')
        page('D06 Stability classifications and denominators',stability,'Accurate/stable is favorable; stable/inaccurate and unstable remain distinct. Counts refer to the declared paired rollouts, not independent samples or confidence bounds.')
        def initialization(ax):
            counts=Counter(r.get('selection_status') for r in data['parameters'] if r.get('selection_status'))
            if not counts: _blank(ax); return
            ax.bar(range(len(counts)),list(counts.values()),color='#587FA3');ax.set_xticks(range(len(counts)),list(counts),rotation=20,ha='right');ax.set_ylabel('Selected model instances')
        page('Pilot checkpoint and baseline adequacy',initialization,'A fitted selection shows that this validation objective improved over initialization; it does not establish optimization convergence. Initialization-selected or failed competitors leave broad inferiority claims unresolved.')
        def build_costs(ax):
            rows=[]
            for auxiliary in data['auxiliary']:
                if Path(auxiliary['path']).name != 'temporal_build_costs.json': continue
                value=auxiliary['value']
                if not isinstance(value,dict):continue
                rows.extend(value.get('pde',[]))
            if not rows: _blank(ax); return
            for j,key in enumerate(('encoding_seconds','classical_build_seconds','query_reconstruction_seconds','validation_seconds','reference_generation_seconds')):
                vals=[r.get(key) for r in rows]
                for i,value in enumerate(vals):
                    if finite(value):ax.bar(i+j*.14,value,width=.13,label=key if i==0 else None,alpha=.8)
            ax.set_xticks([i+.28 for i in range(len(rows))],[r.get('track','') for r in rows]);ax.set_ylabel('Measured seconds (lower for equal work)');ax.set_yscale('log')
        page('D07 Encoding reference and reconstruction costs',build_costs,'Lower seconds is cheaper for equal work. Reference generation is offline cost, while encoding, validation and reconstruction may be paid at runtime. Compilation and energy stay NA when unavailable.')
        curves = []
        for source in data['learning']:
            row = flatten(source); row.setdefault('family', family(row))
            row.setdefault('track', 'unspecified'); row.setdefault('phase','training'); row.setdefault('train_count',None)
            if finite(row.get('update')) and finite(row.get('loss')): curves.append(row)
        ranges = _learning_ranges(curves, 'update', 'loss', max_windows=20) if curves else []
        page('Learning loss observed ranges', lambda ax: range_plot(ax, ranges),
             'Lower loss is a closer fit to the stated training objective. Translucent bands span observed minima and maxima in adjacent windows; they are continuous observed ranges, not confidence intervals. Empty windows leave visible gaps.')
        validation_curves=[]
        for source in data['learning']:
            row=flatten(source);row.setdefault('family',family(row));row.setdefault('track','unspecified');row.setdefault('phase','training');row.setdefault('train_count',None)
            if finite(row.get('update')) and finite(row.get('validation_loss')):validation_curves.append(row)
        validation_ranges=_learning_ranges(validation_curves,'update','validation_loss',max_windows=20) if validation_curves else []
        page('Validation loss observed ranges',lambda ax:range_plot(ax,validation_ranges),'Lower validation loss is better on the same objective. Observed low/high ranges are not confidence intervals; unmeasured validation windows remain gaps. Selection does not replace fresh confirmation.')
        def economics(ax):
            known = [(r['unit'], r['elapsed_seconds']) for r in data['stage_status'] if finite(r.get('elapsed_seconds'))]
            text = f"Recorded scientific-stage time: {sum(v for _,v in known):.3f} s across {len(known)} measured units.\n"
            text += ('This is a partial ledger unless operational startup, preprocessing, references, tuning, failures, recovery, reporting and allocations are also supplied.\n\n'
                     'Energy: NA without measurements. Monetary cost: NA without an actual rate and accounting model.\n\n'
                     'Comparative claims require same-task accuracy-qualified full solver cost, including estimation, rejection and fallback where used.')
            ax.axis('off'); ax.text(.03,.9,textwrap.fill(text,100,replace_whitespace=False),va='top',fontsize=11,transform=ax.transAxes)
        page('Complete cost ledger and limitations', economics,
             'Smaller complete cost is better for the same accuracy and deployment volume. Missing charges are not zero; ceilings and operation counts are not resource bills.')
    _gzip(output / 'learning-range-data.json.gz', {'schema':'tdn.adjacent-ranges/v1','records':ranges,'validation_records':validation_ranges,'meaning':'Observed ranges, not confidence intervals; empty windows remain gaps.'})
    manifest = {'schema':'tdn.adjacent-figures/v1','panels':panels,'pdf':'adjacent-atlas.pdf','profile':data['profile'],
        'counts':{name:len(data[name]) for name in TABLES},'sources':data['sources'],'stage_status':data['stage_status'],
        'chart_data':'chart-data.json.gz','chart_data_sha256':hashlib.sha256((output/'chart-data.json.gz').read_bytes()).hexdigest(),
        'learning_range_data':'learning-range-data.json.gz','raw_observations_discarded':False,'scope':SCOPE}
    _write(output/'manifest.json',manifest)
    body = ''.join(f'<section><h2>{html.escape(p["title"])}</h2><p>{html.escape(p["reading_guide"])}</p><img loading="lazy" src="{p["path"]}" alt="{html.escape(p["title"])}"><p>{html.escape(p["scope"])}</p></section>' for p in panels)
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>TDN adjacent interaction atlas</title><style>body{font:16px system-ui;margin:2em;background:#f3f6f8}section{background:white;padding:1em;margin:1em 0}img{max-width:100%}</style><h1>TDN adjacent interaction study</h1><p>'+html.escape(SCOPE)+'</p><p><a href="adjacent-atlas.pdf">PDF atlas</a> · <a href="chart-data.json.gz">Raw chart data</a> · <a href="learning-range-data.json.gz">Observed ranges</a> · <a href="manifest.json">Source hashes</a></p>'+body)
    return manifest


def run(ctx):
    data = collect(ctx); tables = write_tables(data, ctx.path/'tables')
    figures = build_figures(data, ctx.path/'figures', ctx.budget)
    result = {'schema':SCHEMA,'verified_experiments':len(data['experiments']),'panels':len(figures['panels']),
        'tables':tables,'stage_status':data['stage_status'],'complete_planned_evidence':all(r['verified'] for r in data['stage_status']),
        'scientific_outcome':'DESCRIPTIVE_VERIFIED_EVIDENCE' if data['experiments'] else 'NA','scope':SCOPE,
        'monetary_cost':None,'energy_joules':None,'promotion_to_main_authorized':False}
    _write(ctx.path/'analysis.json',result)
    mechanisms = list(ctx.protocol.get('mechanisms',{}))
    if mechanisms:
        ctx.record('report/source-linked-atlas',[mechanisms[0]],metrics={'panels':len(figures['panels']),
            'verified_experiments':len(data['experiments']),'scientific_outcome':'REPORT_INVENTORY_ONLY'},
            config={'scope':'Inventory and visualization do not establish a scientific advantage'}, checks=[
                check('source-linked-observations',all('_source' in r for r in data['experiments']),True,'eq',category='correctness'),
                check('complete-planned-evidence',result['complete_planned_evidence'],True,'eq',category='gap'),
                check('report-is-not-proof',None,None,category='math')])
    return result
