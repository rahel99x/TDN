"""Source-bound 64²/128² atlas with explicitly display-only smooth drawings.

Only the engine's verified prerequisites contribute observations. Canonical row
files are counted once; copied catalogs and embedded summary rows are metadata.
Every raw source remains available, including failures and incomplete evidence.
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

import numpy as np

from tdn.analysis.frontier.core import check, clean
from tdn.analysis.portfolio.report import _read_source, _rows, _pointer
from .report import finite, role, style, family
from .resolution_plotting import smooth_curve, observed_windows, draw_curve, spatial_image

SCHEMA = 'tdn.adjacent-resolution-analytics/v1'
TABLES = ('experiments', 'checks', 'diagnostics', 'evaluation', 'partial_evaluation',
          'learning', 'catalog', 'validation', 'responses', 'timing', 'claims',
          'exploration', 'stage_status', 'spatial_observations', 'patterns')
CANONICAL = {'rows.jsonl':'experiments', 'resolution_diagnostic_rows.json':'diagnostics',
    'evaluation-rows.json':'evaluation', 'loss-curves.json':'learning', 'catalog.json':'catalog',
    'validation-rows.json':'validation', 'parameter-responses.json':'responses',
    'paired-timings.json':'timing', 'primary-comparisons.json':'claims', 'descriptive-comparisons.json':'claims',
    'prototype_rows.json':'exploration', 'resolution_exploration_rows.json':'exploration',
    'spatial-observations.json':'spatial_observations'}
SCOPE = ('Separate resolution study. Same-grid FD and same-grid dealiased Galerkin are distinct targets; '
    'Galerkin is not automatically continuum truth. Fields are statistical units; grids, schedules, '
    'horizons and seeds are paired observations. Completion, mathematical checks and scientific utility '
    'are distinct. Local FNO controls are not paper reproductions. Missing measurements remain NA.')
DISPLAY = ('Thin curves use shape-preserving PCHIP display interpolation with original observations visible; '
    'missing intervals remain gaps. Translucent bands span observed minima/maxima, not confidence intervals. '
    'Bicubic spatial images are display only; raw arrays, errors, extrema, selection and statistics are unchanged.')
ROLES = 'Ours: triangles; Theirs: circles; analytic controls: squares. Colors identify methods, not quality.'


def _write(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False)+'\n')


def _gzip(path, value):
    with gzip.GzipFile(filename=str(path), mode='wb', mtime=0) as stream:
        stream.write(json.dumps(clean(value), separators=(',', ':'), allow_nan=False).encode())


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): digest.update(block)
    return digest.hexdigest()


def view(row):
    """Explicit semantic aliases; physical RMS/maximum never become errors."""
    result = {}
    for key in ('config', 'metrics'):
        if isinstance(row.get(key), dict): result.update(row[key])
    result.update(row)
    aliases = {'seconds': ('cost_seconds', 'median_seconds', 'latency_seconds', 'complete_seconds'),
        'N': ('evaluation_grid', 'grid'), 'T': ('final_time', 'horizon'),
        'error_rms': ('rms_error',), 'error_max': ('max_error',),
        'uncertainty': ('reference_uncertainty', 'reference_uncertainty_rms', 'uncertainty_rms'),
        'time': ('t', 'query_time', 'T'), 'h': ('h_max', 'timestep')}
    for key, candidates in aliases.items():
        if key not in result:
            result[key] = next((result[name] for name in candidates if name in result), None)
    if isinstance(result.get('N'), (list, tuple)): result['N'] = result['N'][0]
    if isinstance(result.get('uncertainty'), dict):
        result['uncertainty'] = result['uncertainty'].get('rms')
    if isinstance(result.get('oracle'), dict):
        for key in ('effective_rank', 'condition_number', 'coefficient_refinement_sensitivity_l2'):
            result[key] = result['oracle'].get(key)
    if result.get('category') == 'endpoint' and finite(result.get('T')) and finite(result.get('steps')) and result['steps'] > 0:
        result['h'] = result['T']/result['steps']
    if result.get('category') == 'spatial_refinement':
        result['refinement_grid'] = max(result.get('fine_grids', [result.get('N')]))
        result['spatial_difference_rms'] = result.get('refinement_difference_rms')
    return result


def _with_source(row, source, pointer, unit_metadata):
    out = dict(row)
    context = {key: unit_metadata[key] for key in ('grid', 'track', 'split') if key in unit_metadata and key not in out}
    out.update(context)
    out['_unit_context'] = context
    out['_source'] = {**source, 'pointer': pointer, 'evidence_status':'VERIFIED_PREREQUISITE'}
    return out


def collect(ctx):
    data = {name: [] for name in TABLES}
    data.update(schema=SCHEMA, profile=ctx.protocol.get('profile'), protocol=ctx.protocol,
        device=str(ctx.device), sources=[], auxiliary=[], omissions=[], scope=SCOPE, display_policy=DISPLAY,
        statistical_intervals={'status':'NA_NOT_COMPUTED',
            'reason':'The resolution aggregate reports descriptive paired-cohort summaries; it does not compute parent-cluster statistical intervals.'},
        interval_estimand='Statistical intervals: NA — no parent-cluster intervals are computed by this resolution aggregate. Reported medians and observed ranges do not estimate inferential uncertainty.')
    units = ctx.protocol.get('units', {})
    aggregate_units = [name for name, unit in units.items() if unit.get('kind') == 'resolution_aggregate'
                       and name in ctx.prerequisites and (Path(ctx.prerequisites[name])/'evaluation-rows.json').is_file()]
    if len(aggregate_units) > 1: raise ValueError('Ambiguous canonical aggregate')
    data['evaluation_inventory'] = 'verified aggregate' if aggregate_units else 'verified shards; whole-cohort aggregate absent'
    for name in sorted(set(units) | set(ctx.prerequisites)):
        if name == ctx.stage: continue
        unit = units.get(name, {})
        failure = getattr(ctx, 'stage_failures', {}).get(name, {})
        base = ctx.prerequisites.get(name)
        status = dict(unit=name, kind=unit.get('kind'), grid=unit.get('grid'), track=unit.get('track'),
            computational_status='VERIFIED' if base else failure.get('status', 'NA'),
            math_outcome='NA', scientific_outcome='NA', verified=bool(base), planned_seconds=unit.get('seconds'),
            elapsed_seconds=failure.get('elapsed_seconds'), error=failure.get('error'), monetary_cost=None, energy_joules=None)
        data['stage_status'].append(status)
        if base is None: continue
        base = Path(base)
        for path in sorted([*base.glob('*.json'), *base.glob('*.jsonl')]):
            ctx.budget.check()
            value, sha, size = _read_source(path)
            source = dict(unit=name, path=str(path), sha256=sha, bytes=size)
            data['sources'].append(source)
            table = CANONICAL.get(path.name)
            if path.name == 'summary.json' and isinstance(value, dict):
                status.update(computational_status=value.get('status', 'VERIFIED'),
                    elapsed_seconds=value.get('elapsed_seconds'), math_outcome=value.get('math_outcome', 'NA'),
                    scientific_outcome=value.get('scientific_outcome', 'NA'))
            # Freeze copies are provenance, not additional trained model instances.
            if table == 'catalog' and unit.get('kind') == 'resolution_freeze': table = None
            if table == 'evaluation' and aggregate_units and name not in aggregate_units: table = 'partial_evaluation'
            if table is None:
                data['auxiliary'].append({**source, 'value':value})
                continue
            records = _rows(value)
            if isinstance(value, dict) and not records:
                # Structured timing and claims containers remain exact metadata
                # when they have no declared canonical row collection.
                data['auxiliary'].append({**source, 'value':value})
            for index, row in enumerate(records):
                item = _with_source(row, source, f'{_pointer(value)}/{index}', unit)
                if table=='spatial_observations' and isinstance(value,dict):
                    if value.get('array_sha256') is not None:item.setdefault('array_sha256',value['array_sha256'])
                    parent=value.get('source_parent',{})
                    if isinstance(parent,dict):
                        for key in ('domain','regime','split'): 
                            if key in parent:item.setdefault(key,parent[key])
                data[table].append(item)
                if table == 'experiments':
                    for j, test in enumerate(row.get('checks', [])):
                        data['checks'].append(_with_source({**test, 'experiment_id':row.get('experiment_id')},
                            source, f'{_pointer(value)}/{index}/checks/{j}', unit))
    catalog={r['model_id']:r for r in data['catalog'] if r.get('model_id')}
    for row in data['responses']:
        model=catalog.get(row.get('model_id'),{})
        context={key:model[key] for key in ('family','role','track','seed') if key not in row and key in model}
        row.update(context);row['_model_context']=context
    _collect_patterns(data, ctx)
    return data


def _array_reference(row):
    arrays = row.get('arrays') if isinstance(row.get('arrays'), dict) else {}
    file = row.get('array_file', row.get('arrays_file', arrays.get('file')))
    keys = row.get('array_keys', arrays.get('keys', {}))
    return file, keys if isinstance(keys, dict) else {}


def _collect_patterns(data, ctx):
    """At most one declared representative per grid/track; never error-selected."""
    candidates = defaultdict(list)
    # Aggregate rows retain paths relative to their producing shard. Read its
    # already-verified canonical row, never reinterpret that relative path as
    # belonging to the aggregate directory.
    endpoint_sources=data['partial_evaluation'] if data['evaluation_inventory']=='verified aggregate' else data['evaluation']
    for raw in data['spatial_observations'] + endpoint_sources + data['diagnostics']:
        row = view(raw); file, keys = _array_reference(row)
        if file and keys:
            candidates[(str(row.get('track')), str(row.get('N')))].append(row)
    priority = {'channel_neural':0, 'channel_global':1, 'quad2_fixed':2}
    for group, rows in sorted(candidates.items()):
        rows.sort(key=lambda r:(priority.get(family(r), 3), str(r.get('parent_id')), str(r.get('case_id')), str(r['_source'])))
        row = rows[0]; file, keys = _array_reference(row)
        base = Path(ctx.prerequisites[row['_source']['unit']]).resolve()
        path = base / file
        if Path(file).is_absolute() or path.is_symlink() or not path.resolve().is_relative_to(base):
            raise ValueError('Spatial array reference escapes its verified prerequisite')
        if not path.is_file() or path.stat().st_size > 32 << 20:
            data['omissions'].append({'reason':'missing or oversized spatial array', 'row_source':row['_source'], 'file':file})
            continue
        ctx.budget.check()
        sha = _sha(path)
        if row.get('array_sha256') is not None and row['array_sha256'] != sha:
            raise ValueError('Referenced spatial array differs from its recorded digest')
        source = {'unit':row['_source']['unit'], 'path':str(path), 'sha256':sha, 'bytes':path.stat().st_size}
        data['sources'].append(source)
        stored, array_shapes = {}, {}
        with np.load(path, allow_pickle=False) as bank:
            for logical, key in keys.items():
                if not isinstance(key, str) or key not in bank:
                    data['omissions'].append({'reason':'missing referenced NPZ key', 'key':key, 'source':source})
                    continue
                array = bank[key]
                original_shape=list(array.shape)
                # Only leading singleton sample/channel axes may be removed.
                # Multiple times, channels or batch members require an explicit
                # selection and cannot silently become one spatial image.
                while array.ndim > 2 and array.shape[0] == 1: array=array[0]
                if array.size > 512*512 or array.ndim not in (1, 2) or not np.issubdtype(array.dtype, np.number): continue
                array_shapes[logical]={'original_shape':original_shape,'display_shape':list(array.shape)}
                if np.iscomplexobj(array):
                    stored[logical] = {'real':array.real.tolist(), 'imag':array.imag.tolist(), 'shape':list(array.shape)}
                else: stored[logical] = array.tolist()
        if _sha(path) != sha: raise ValueError('Spatial source changed while reading')
        data['patterns'].append({'track':row.get('track'), 'N':row.get('N'), 'parent_id':row.get('parent_id'),
            'method':family(row), 'role':role(family(row), row), 'split':row.get('split'), 'regime':row.get('regime'),
            'domain':row.get('domain', [1., 1.]), 'row_source':row['_source'], '_source':source,
            'selection':'first available declared parent/case; family priority channel_neural, channel_global, quad2_fixed, then other; not error-selected',
            'array_keys':keys, 'array_shapes':array_shapes, 'arrays':stored})


def write_tables(data, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for table in TABLES:
        rows = data[table]
        fields = sorted({key for row in rows for key in row})
        path = output/f'{table}.csv'
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields or ['status'], lineterminator='\n')
            writer.writeheader()
            for row in rows:
                values = {}
                for key, val in row.items():
                    text = json.dumps(clean(val), separators=(',', ':'), allow_nan=False) if isinstance(val, (dict, list)) else '' if val is None else str(val)
                    values[key] = "'"+text if text.startswith(('=', '+', '-', '@', '\t', '\r')) else text
                writer.writerow(values)
        artifacts.append({'table':table, 'path':path.name, 'rows':len(rows), 'sha256':_sha(path)})
    return artifacts


def _blank(ax, reason='NA — no applicable verified measurement'):
    ax.text(.5, .5, reason, ha='center', va='center', transform=ax.transAxes, fontsize=9, wrap=True)
    ax.set_xticks([]); ax.set_yticks([])


def _legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles)); ax.legend(unique.values(), unique.keys(), fontsize=5.2, framealpha=.85)


def _scatter(ax, rows, x, y, *, xlabel=None, ylabel=None, logx=False, logy=True):
    groups = defaultdict(list)
    for raw in rows:
        row = view(raw)
        if finite(row.get(x)) and finite(row.get(y)) and (not logx or row[x] > 0) and (not logy or row[y] > 0):
            groups[(family(row), role(family(row), row))].append(row)
    if not groups: _blank(ax); return
    for (method, owner), group in sorted(groups.items()):
        s = style(method, {'role':owner})
        ax.scatter([r[x] for r in group], [r[y] for r in group], marker=s['marker'], s=14,
            facecolors='none', edgecolors=s['color'], linewidths=.5, label=f'{owner}: {method}')
    if logx: ax.set_xscale('log')
    if logy: ax.set_yscale('log')
    ax.set_xlabel(xlabel or x); ax.set_ylabel(ylabel or y); ax.grid(alpha=.15, linewidth=.4); _legend(ax)


def _curve_panel(ax, rows, x, y, display_rows, *, xlabel=None, ylabel=None, positive=True):
    groups = defaultdict(list)
    for raw in rows:
        row = view(raw)
        if finite(row.get(x)):
            # Distinct physical fields, schedules and model instances are never
            # connected into one apparent function.
            varying={x}
            if x in ('time','h'):varying.add('T')
            if x=='h':varying.add('steps')
            if x=='alpha':varying.add('parent_id')
            identity = tuple(str(row.get(key)) for key in ('family', 'method', 'track', 'N', 'parent_id', 'field_cluster',
                'model_id', 'seed', 'trial_id', 'category', 'compression', 'output_modes', 'nodes', 'T', 'steps') if key not in varying)
            groups[identity].append(row)
    drawn = 0
    for identity, group in groups.items():
        if len({r[x] for r in group}) != len(group):
            # Distinct but incompletely identified cases: raw points are honest.
            _scatter(ax, group, x, y, logy=positive); continue
        curve = smooth_curve([r[x] for r in group], [r.get(y) for r in group], positive=positive)
        if not curve['segments']: continue
        s = style(family(group[0]), group[0])
        draw_curve(ax, curve, style=s, label=f'{role(family(group[0]), group[0])}: {family(group[0])}')
        display_rows.append({'panel_metric':y, 'abscissa':x, 'identity':identity, 'sources':[r['_source'] for r in group if '_source' in r], 'curve':curve})
        drawn += 1
    if not drawn and not ax.has_data(): _blank(ax); return
    ax.set_xlabel(xlabel or x); ax.set_ylabel(ylabel or y)
    if positive and ax.has_data(): ax.set_yscale('log')
    ax.grid(alpha=.15, linewidth=.4); _legend(ax)


def _facet(rows, track, grid):
    return [view(r) for r in rows if str(view(r).get('track')) == str(track) and str(view(r).get('N')) == str(grid)]


def _text(ax, lines):
    ax.axis('off')
    ax.text(.01, .98, '\n\n'.join(textwrap.fill(str(line), 125) for line in lines), va='top', fontsize=9, transform=ax.transAxes)


def _learning(ax, rows, metric, displays):
    groups = defaultdict(list)
    for raw in rows:
        row = view(raw)
        groups[(family(row), row.get('seed'), row.get('trial_id'))].append(row)
    drawn = False
    for identity, group in sorted(groups.items(), key=lambda kv:str(kv[0])):
        windows = observed_windows(group, 'update', metric, maximum_windows=35)
        curve = smooth_curve([w['x'] for w in windows], [w['median'] for w in windows],
            lower=[w['lower'] for w in windows], upper=[w['upper'] for w in windows], positive=True)
        if not curve['segments']: continue
        method = identity[0]; first = group[0]
        draw_curve(ax, curve, style=style(method, first), label=f'{role(method, first)}: {method}', observed_range=True)
        original=[r for r in group if finite(r.get('update')) and finite(r.get(metric)) and r[metric]>0]
        s=style(method,first)
        ax.scatter([r['update'] for r in original],[r[metric] for r in original],marker=s['marker'],
            s=5,facecolors='none',edgecolors=s['color'],alpha=.22,linewidths=.35)
        displays.append({'panel_metric':metric, 'series':identity, 'windows':windows, 'curve':curve})
        drawn = True
    if not drawn: _blank(ax); return
    ax.set_yscale('log'); ax.set_xlabel('Optimizer update'); ax.set_ylabel(metric+' (lower is better)')
    ax.grid(alpha=.15, linewidth=.4); _legend(ax)


def _response_rows(records):
    """Extract only explicit response gains, never interpret neural weights as gains."""
    output = []
    for raw in records:
        root = view(raw)
        probes = root.get('response_probes', [root])
        for probe in probes:
            if not isinstance(probe, dict): continue
            report = probe.get('report', probe)
            response = report.get('response', report) if isinstance(report, dict) else {}
            gain = response.get('gain') if isinstance(response, dict) else None
            if not isinstance(gain, list): continue
            arrays = np.asarray(gain)
            if not np.issubdtype(arrays.dtype, np.number): continue
            for index, value in enumerate(arrays.reshape(-1)):
                output.append({**root, 'channel_index':index, 'gain':float(value), 'h':probe.get('horizon', probe.get('h'))})
    return output


def _trajectory_rows(records):
    rows = []
    for raw in records:
        row = view(raw)
        times = row.get('times', row.get('trajectory_times', []))
        values = row.get('trajectory_error_rms', row.get('errors_rms', []))
        if isinstance(times, list) and isinstance(values, list) and len(times) == len(values):
            rows.extend({**row, 'time':t, 'error_rms':e} for t, e in zip(times, values))
        elif row.get('category') == 'rollout': rows.append(row)
    return rows


def _pattern_array(pattern, *keys):
    for key in keys:
        value = pattern.get('arrays', {}).get(key)
        if isinstance(value, list):
            array = np.asarray(value, dtype=float)
            if array.ndim == 2: return array
    return None


def build_figures(data, output, budget=None, *, _test_dpi=None, _test_page_limit=None):
    """Production output is 400 DPI; private test options avoid heavy fixtures."""
    root = Path(__file__).resolve().parents[3]
    os.environ.setdefault('MPLCONFIGDIR', str(root/'.runtime/matplotlib'))
    os.environ.setdefault('XDG_CACHE_HOME', str(root/'.cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    dpi = 400 if _test_dpi is None else _test_dpi
    panels, displays = [], []
    omissions=list(data['omissions'])
    _gzip(output/'chart-data.json.gz', data)
    resolution = data['protocol'].get('resolution', {})
    grids = resolution.get('grids', [64, 128]); tracks = resolution.get('tracks', ['discrete', 'continuum'])
    pairs = [(track, grid) for track in tracks for grid in grids]
    if len(pairs) > 4: raise ValueError('Bounded resolution atlas supports at most four track/grid facets')
    rows = data['evaluation']; diagnostics = data['diagnostics']
    single_rows=[r for r in rows if r.get('batch_size',1)==1]
    plt.rcParams.update({'font.size':8, 'axes.linewidth':.5, 'lines.linewidth':.55, 'pdf.fonttype':42})
    with PdfPages(output/'resolution-atlas.pdf') as pdf:
        def page(title, draw, guide, shape=(1, 1)):
            if _test_page_limit is not None and len(panels) >= _test_page_limit: return
            if len(panels) >= 28: raise ValueError('Resolution atlas exceeded its bounded panel budget')
            if budget: budget.check()
            fig, axes = plt.subplots(*shape, figsize=(12, 8), squeeze=False)
            try:
                draw(axes.ravel())
                fig.suptitle(title, fontsize=13, y=.98)
                fig.text(.02, .02, '\n'.join(textwrap.wrap(guide+' '+ROLES, 170)), fontsize=6.8, va='bottom')
                fig.tight_layout(rect=(.01, .105, .99, .955))
                name=f'{len(panels)+1:02d}-'+''.join(c.lower() if c.isalnum() else '-' for c in title).strip('-')+'.png'
                fig.savefig(output/name, dpi=dpi); pdf.savefig(fig)
                panels.append({'title':title, 'path':name, 'sha256':_sha(output/name), 'dpi':dpi,
                    'reading_guide':guide, 'scope':SCOPE, 'display_policy':DISPLAY})
            finally: plt.close(fig)

        def facets(axes, records, callback):
            for ax, (track, grid) in zip(axes, pairs):
                callback(ax, _facet(records, track, grid)); ax.set_title(f'{track}; {grid}²')
            for ax in axes[len(pairs):]: ax.set_visible(False)

        page('Architecture information and evidence boundaries', lambda ax:_text(ax[0], [
            SCOPE, 'Physical state → fixed-background low/high decomposition → signed LL/LH/HH interactions → gains → output projection → physical backbone plus correction. Channel labels describe origins, not additional spatial dimensions.',
            'Conditioners consume their declared physical features; FNO alternatives consume the declared fields/parameters. Learned gains, analytic normalization, node counts, projection and extra work require separate attribution.',
            f"Evaluation inventory: {data['evaluation_inventory']}. Profiles: {data['profile']}. Raw sources: {len(data['sources'])}.", DISPLAY,
            'Higher accuracy-qualified speedup is better; lower error/cost/memory is better at the same task. Gains, learned weights, roughness and numerical rank have no universal better direction.']),
            'This diagram describes information flow, not measured superiority. Analytic structure and learned adaptation require separate controlled comparisons.')
        def coverage(axes):
            counts = Counter(r['computational_status'] for r in data['stage_status'])
            axes[0].bar(list(counts), list(counts.values()), color='#647C90'); axes[0].set_ylabel('Units (completion is not scientific success)'); axes[0].tick_params(axis='x', rotation=25)
            known=[r for r in data['stage_status'] if finite(r.get('elapsed_seconds'))]
            if known:
                bykind=defaultdict(float)
                for r in known: bykind[r.get('kind') or 'unspecified']+=r['elapsed_seconds']
                axes[1].barh(list(bykind), list(bykind.values()), color='#647C90'); axes[1].set_xlabel('Measured stage seconds; lower for equivalent work')
            else: _blank(axes[1], 'NA — no measured stage durations')
        page('Coverage failures and stage compute', coverage, 'Failures and missing units remain in the denominator. Summed science-stage time is a partial ledger; scheduler startup, recovered work and reporting may be additional.', (1, 2))
        page('Complete cost accounting and limitations', lambda ax:_text(ax[0], [
            f"Measured scientific stage durations: {sum(r['elapsed_seconds'] for r in data['stage_status'] if finite(r.get('elapsed_seconds'))):.3f} s; {sum(finite(r.get('elapsed_seconds')) for r in data['stage_status'])}/{len(data['stage_status'])} units measured.",
            'Training catalogs and all tuning trials are retained separately. Reference generation, failed/interrupted jobs, device transfer, preprocessing and deployment checks must be charged where incurred. An unrecorded charge is NA, not zero.',
            'Warm and cold timings are separate. Repeated schedules and model calls sharing timing_id are paired, not independent timing studies. Allocation-wide peak memory cannot be assigned to individual model calls.',
            'Monetary cost and energy: NA without actual rates/meters. No deployment payback claim is made without a measured accuracy-qualified runtime margin and complete offline ledger.']), 'Lower complete cost is better for an identical workload and accuracy requirement. Budgets are ceilings, not bills.')
        for track, grid in pairs:
            selected=_facet(single_rows, track, grid)
            regimes=sorted({str(r.get('regime', r.get('generator', 'unspecified'))) for r in selected})
            shown=regimes[:8] or ['unspecified']
            def frontier(axes, selected=selected, shown=shown):
                for ax, regime in zip(axes, shown):
                    subset=[r for r in selected if str(r.get('regime', r.get('generator', 'unspecified')))==regime]
                    _scatter(ax, subset, 'seconds', 'error_rms', xlabel='Complete endpoint seconds ↓', ylabel='Endpoint RMS error ↓', logx=True)
                    ax.set_title(regime)
                for ax in axes[len(shown):]: ax.set_visible(False)
            page(f'Accuracy cost {track} {grid} square', frontier, 'Lower-left is better within a regime and fixed task. Points retain all schedules, horizons, parents and failures in raw data; no cross-regime or post-hoc universal winner is inferred. Reference acceptance and maximum-error qualification must also hold.', (2, 4))
            if len(regimes)>8: omissions.append({'reason':'additional regime facets retained in raw data', 'track':track, 'grid':grid, 'regimes':regimes[8:]})
        for track, grid in pairs:
            selected=_facet(data['learning'], track, grid)
            def learning(axes, selected=selected):
                for ax, metric in zip(axes, ('train_loss','validation_loss')): _learning(ax, selected, metric, displays)
            page(f'Learning observations {track} {grid} square', learning, 'Lower is better only on the stated objective. Each seed/trial remains its own curve. Continuous translucent min/max windows are observed ranges, not confidence intervals; original raw updates remain in chart data and window records. Missing or nonpositive log observations leave gaps.', (1, 2))
        page('Quadrature and compression attribution', lambda ax:facets(ax, diagnostics,
            lambda a,r:_scatter(a, r, 'nodes', 'error_rms', xlabel='Quadrature nodes (more work →)', ylabel='RMS error ↓')),
            'Fewer nodes are cheaper only after actual timing; lower error is better. Output cutoff and compression placement remain separate row attributes. Additional quadrature cannot remove a compression floor.', (2, 2))
        def floors(axes):
            facets(axes[:4],diagnostics,lambda a,r:_scatter(a,r,'output_modes','output_compression_rms_floor',xlabel='Output cutoff',ylabel='L2/RMS projection floor ↓'))
            facets(axes[4:],diagnostics,lambda a,r:_scatter(a,r,'effective_rank','error_rms',xlabel='Reference-resolved numerical rank',ylabel='Oracle RMS error ↓'))
        page('Compression floors and resolved representation rank',floors,
            'A projection floor is an RMS/L2 bound, not a maximum-error minimax bound. Reference-resolved numerical rank differs from algebraic rank and has no universally better direction. Oracle coefficients are representation diagnostics, not deployable learned methods.', (2, 4))
        def orders(axes):
            facets(axes[:4],diagnostics,lambda a,r:_curve_panel(a,r,'h','error_rms',displays,xlabel='Maximum timestep',ylabel='RMS error ↓'))
            parity=[]
            for row in diagnostics:
                if row.get('category')!='amplitude_parity':continue
                for metric in ('even_defect_rms','odd_defect_rms','even_after_quadratic_rms','odd_after_cubic_rms'):
                    parity.append({**row,'method':metric,'error_rms':row.get(metric)})
            facets(axes[4:],parity,lambda a,r:_curve_panel(a,r,'epsilon','error_rms',displays,xlabel='Fluctuation amplitude ε',ylabel='Parity defect RMS ↓'))
        page('Timestep off-grid and amplitude convergence', orders,
            'Smaller timestep usually costs more. Curves join only identified physical cases; no order theorem follows from appearance. Parity removes measured constant and linear backbone defects; cancellation and reference floors can obscure fitted orders.', (2, 4))
        page('Autonomous rollout and stability', lambda ax:facets(ax, _trajectory_rows(diagnostics),
            lambda a,r:_curve_panel(a, r, 'time', 'error_rms', displays, xlabel='Physical time', ylabel='Rollout RMS error ↓')),
            'Lower error and fewer failures are better at the same horizon. Curves are display interpolation between measured times; absent rollout references remain gaps. Endpoint-only agreement does not establish stability.', (2, 2))
        page('Reference uncertainty and observable error', lambda ax:facets(ax, rows+diagnostics,
            lambda a,r:_scatter(a,r,'uncertainty','error_rms',xlabel='Estimated RMS reference uncertainty ↓',ylabel='Observed RMS error ↓',logx=True)),
            'Ratios near the uncertainty floor are unresolved. Agreement under actual refinement is an uncertainty estimate, not a certified error bound. Same-grid time uncertainty and continuum spatial uncertainty must stay separate.', (2, 2))
        responses=_response_rows(data['responses']+data['catalog'])
        def information(axes):
            facets(axes[:4],responses,lambda a,r:_scatter(a,r,'channel_index','gain',xlabel='Declared gain index',ylabel='Effective gain (no best direction)',logy=False))
            collisions=[]
            for row in diagnostics:
                if row.get('category')=='feature_collision':
                    collisions.append({**row,'overlap':int(row['acceptable_intervals_overlap']) if isinstance(row.get('acceptable_intervals_overlap'),bool) else None})
            facets(axes[4:8],collisions,lambda a,r:_scatter(a,r,'feature_distance','overlap',xlabel='Feature distance (no best direction)',ylabel='Acceptable scalar intervals overlap (1/0)',logy=False))
            phases=[]
            for row in diagnostics:
                if row.get('category')!='signed_coefficients':continue
                target=row.get('desired_coefficient')
                channels=row.get('channel_coefficients')
                if isinstance(target,list) and len(target)==2 and isinstance(channels,list):
                    combined=np.sum(np.asarray(channels),axis=0)
                    for j,name in enumerate(('real','imag')):
                        phases.append({**row,'method':'recombined_channels_'+name,'desired':target[j],'predicted':float(combined[j])})
            facets(axes[8:],phases,lambda a,r:_scatter(a,r,'desired','predicted',xlabel='Desired signed Fourier coefficient',ylabel='Predicted (closer to y=x is better)',logy=False))
            for ax in axes[8:]:
                if ax.has_data():
                    limits=[*ax.get_xlim(),*ax.get_ylim()];ax.plot([min(limits),max(limits)],[min(limits),max(limits)],color='#777777',linewidth=.4,linestyle=':')
        page('Learned responses feature ambiguity and signed interactions',information,
            'Explicit gains are not neural weights. Scalar feature collisions concern fixed GL2 gain only, not all neural responses. Signed coefficient closeness to y=x is better; unresolved amplitudes/phases remain NA. Different gains can yield the same rounded endpoint.', (3, 4))
        def runtime(axes):
            facets(axes[:4],single_rows,lambda a,r:_scatter(a,r,'seconds','cold_seconds',xlabel='Warm complete seconds ↓',ylabel='First invocation seconds ↓',logx=True))
            # A batch's complete timing is shared by all constituent field rows.
            # Plot it once per model/timing group, while retaining all field errors.
            seen=set();throughput=[]
            for row in rows:
                key=(row.get('timing_id'),row.get('model_id'))
                if row.get('batch_size',1)>1 and key not in seen:throughput.append(row);seen.add(key)
            facets(axes[4:8],throughput,lambda a,r:_scatter(a,r,'batch_size','throughput_fields_per_second',xlabel='Distinct fields per batch',ylabel='Fields/second ↑ (accuracy still required)'))
            memory=[]
            for row in data['timing']:
                peak=row.get('peak_memory',row.get('memory',{}))
                if not isinstance(peak,dict):peak={}
                allocated=row.get('peak_allocated_bytes',peak.get('peak_allocated_bytes',peak.get('allocated_bytes')))
                reserved=row.get('peak_reserved_bytes',peak.get('peak_reserved_bytes',peak.get('reserved_bytes')))
                memory.append({**row,'method':'paired_timing_group','role':'Analytic control','allocated':allocated,'reserved':reserved})
            facets(axes[8:],memory,lambda a,r:_scatter(a,r,'allocated','reserved',xlabel='Group peak allocated bytes ↓',ylabel='Group peak reserved bytes ↓'))
        page('Warm cold throughput and memory accounting',runtime,
            'Lower latency/memory and higher accuracy-qualified throughput are better. Group memory peaks are not per-model costs; square symbols here identify an accounting control. Raw batch timings are total batch seconds; per-field error rows share timing_id. Warmup/first invocation remain separate.', (3, 4))
        def spectra(axes):
            for ax, pair in zip(axes, pairs):
                pats=[p for p in data['patterns'] if (str(p.get('track')),str(p.get('N')))==tuple(map(str,pair))]
                plotted=False
                for pattern in pats:
                    residual=_pattern_array(pattern,'residual','error')
                    if residual is None: continue
                    spectrum=np.abs(np.fft.fft2(residual)/residual.size)**2
                    ky,kx=np.meshgrid(np.fft.fftfreq(residual.shape[0])*residual.shape[0],np.fft.fftfreq(residual.shape[1])*residual.shape[1],indexing='ij')
                    shells=np.floor(np.sqrt(kx*kx+ky*ky)).astype(int)
                    energy=np.bincount(shells.ravel(),weights=spectrum.ravel())
                    curve=smooth_curve(list(range(len(energy))),energy.tolist(),positive=True)
                    draw_curve(ax,curve,style=style(pattern['method'],pattern),label=f"{pattern['role']}: {pattern['method']}")
                    displays.append({'panel_metric':'nodal_residual_spectral_shell_energy','source':pattern['_source'],'curve':curve,
                        'meaning':'Sum |FFT(error)/node_count|² over integer-radius frequency shells; raw residual, no interpolated field used.'})
                    plotted=True
                if plotted: ax.set_yscale('log');ax.set_xlabel('Integer mode-radius shell');ax.set_ylabel('Error spectral energy ↓');_legend(ax)
                else:_blank(ax)
                ax.set_title(f'{pair[0]}; {pair[1]}²')
            for ax in axes[len(pairs):]:ax.set_visible(False)
        page('Spectral error of declared representative fields', spectra, 'Lower energy is better per mode at the same task. Spectra use original nodal error arrays; bicubic display is never transformed. One predeclared representative cannot establish population generalization.', (2, 2))
        def refinement(axes):
            facets(axes[:4],[r for r in diagnostics if r.get('category')=='spatial_refinement'],lambda a,r:_scatter(a,r,'refinement_grid','spatial_difference_rms',xlabel='Actually refined maximum grid',ylabel='2N vs 4N projected RMS difference ↓'))
            facets(axes[4:],[r for r in diagnostics if r.get('regime')=='roughness' and r.get('category')=='endpoint'],lambda a,r:_curve_panel(a,r,'alpha','error_rms',displays,xlabel='Finite spectral envelope α',ylabel='Endpoint RMS error ↓'))
        page('Actual spatial refinement and finite roughness stress',refinement,
            'Refinement agreement is an estimate, not a certificate; same-grid Galerkin is not continuum truth. Roughness α varies a finite smooth field envelope; α=1 is a finite-band stress, not an infinite fractal theorem. Variants share one parent cluster.', (2, 4))
        page('Cross-grid transfer on paired physical fields', lambda ax:facets(ax,[r for r in rows if r.get('transfer')],
            lambda a,r:_scatter(a,r,'train_grid','error_rms',xlabel='Training grid',ylabel='Transferred RMS error ↓')),
            'Lower error is better on the same field and final time. Fixed physical bandwidth refinement differs from resolution-relative stress. Reused parents across grids are paired observations, not new independent samples.', (2, 2))
        def claims(axes):
            ax=axes[0];ax.axis('off')
            if not data['claims']:_blank(ax,'NA — no verified canonical comparison rows.');return
            def number(value):return f'{value:.3g}' if finite(value) else 'NA'
            body=[]
            for row in data['claims'][:24]:
                model=str(row.get('model_id',row.get('comparison',row.get('contrast','unspecified'))))
                model=model.replace('discrete/','').replace('continuum/','')
                body.append([f"{row.get('track','NA')}/{row.get('grid','NA')}",model,
                    f"T={number(row.get('final_time'))}; s={row.get('steps','NA')}; b={row.get('batch_size','NA')}",
                    f"{row.get('valid_parents','NA')}/{row.get('independent_parent_denominator','NA')}",
                    number(row.get('median_error_rms')),number(row.get('median_cost_seconds')),row.get('scientific_verdict','NA')])
            table=ax.table(cellText=body,colLabels=['Track/grid','Model identity','Task','Valid/parents','Median RMS ↓','Median seconds ↓','Verdict'],
                loc='upper center',colWidths=[.12,.29,.20,.10,.105,.105,.07],cellLoc='left')
            table.auto_set_font_size(False);table.set_fontsize(6);table.scale(1,1.5)
            ax.text(.01,.04,f"Showing the first {len(body)}/{len(data['claims'])} source-ordered rows; all rows remain in claims.csv and raw chart data.\n"+data['interval_estimand'],
                transform=ax.transAxes,fontsize=7,wrap=True)
        page('Descriptive comparisons and unresolved statistical claims',claims,'These are descriptive cohort summaries. Parent-cluster statistical intervals are NA, not computed. Repeated schedules/seeds do not increase the independent-field denominator. All exact comparison rows remain in claims.csv and chart data.')
        def exploration(axes):
            facets(axes[:4],data['exploration'],lambda a,r:_scatter(a,r,'seconds','error_rms',xlabel='Complete seconds ↓',ylabel='RMS error ↓',logx=True))
            facets(axes[4:],data['exploration'],lambda a,r:_curve_panel(a,r,'query_count','seconds',displays,xlabel='Same-state requested queries',ylabel='Measured build + all queries seconds ↓'))
        page('Protected exploration accuracy cost and temporal reuse',exploration,
            'Faster but less accurate may fail. Complete reuse cost includes encoding/build, transfers and queries. Endpoint accuracy does not establish every intermediate query; Q>1 all-query utility remains NA without its own references. Resolved negative cases remain negative.', (2, 4))
        def checks(axes):
            counts=Counter((r.get('category'),r.get('verdict')) for r in data['checks'])
            if not counts:_blank(axes[0]);return
            labels=[f'{a}/{b}' for a,b in sorted(counts,key=str)]
            axes[0].barh(labels,[counts[k] for k in sorted(counts,key=str)],color='#647C90')
            axes[0].set_xlabel('Logged checks; NA remains in the denominator')
        page('Mathematical checks versus scientific outcomes',checks,'More completed tests is not a better model. GOOD/BAD/NA and 1–100 scores express the predeclared evidence checks, not theorem proofs or a universal scientific ranking.')
        for track, grid in pairs:
            patterns=[p for p in data['patterns'] if str(p.get('track'))==str(track) and str(p.get('N'))==str(grid)]
            def spatial(axes,patterns=patterns):
                if not patterns:
                    for ax in axes:_blank(ax)
                    return
                pattern=patterns[0]
                state_arrays=[_pattern_array(pattern,*keys) for keys in (('input','initial'),('prediction',),('reference',))]
                state_finite=[a[np.isfinite(a)] for a in state_arrays if a is not None and np.isfinite(a).any()]
                state_limits=(min(float(a.min()) for a in state_finite),max(float(a.max()) for a in state_finite)) if state_finite else None
                for col,(label,keys) in enumerate((('Input',('input','initial')),('Prediction',('prediction',)),('Reference',('reference',)),('Error',('residual','error')))):
                    array=_pattern_array(pattern,*keys)
                    if array is None or not np.isfinite(array).any():
                        _blank(axes[col]);_blank(axes[col+4]);continue
                    limit=float(np.nanmax(np.abs(array))) if array.size else 0.
                    vmin,vmax=(-max(limit,1e-30),max(limit,1e-30)) if label=='Error' else state_limits
                    for offset,display in ((0,'raw'),(4,'bicubic')):
                        # Solver tensors are indexed [x,y]; image rows are y.
                        image,metadata=spatial_image(axes[col+offset],array.T,display=display,domain=pattern['domain'],vmin=vmin,vmax=vmax,cmap='coolwarm' if label=='Error' else 'viridis')
                        metadata['source_to_display_axes']='transpose scalar [x,y] field into image [row=y,column=x]'
                        axes[col+offset].set_xlabel('x');axes[col+offset].set_ylabel('y')
                        axes[col+offset].set_title(f'{label}: {display} ({array.shape[0]}² samples)',fontsize=8)
                        axes[col+offset].figure.colorbar(image,ax=axes[col+offset],shrink=.65,pad=.02)
                        displays.append({'panel_metric':label,'source':pattern['_source'],'row_source':pattern['row_source'],'display':metadata})
            method_label=f" — {patterns[0]['role']}: {patterns[0]['method']}" if patterns else ''
            selection_label=f" Selected parent: {patterns[0]['parent_id']}." if patterns else ''
            page(f'Raw and smooth spatial display {track} {grid} square{method_label}',spatial,
                'Top: raw nearest samples. Bottom: bicubic display of identical arrays. Input/prediction/reference share a color scale; states have no preferred numerical direction. Signed error closest to zero is better; red and blue indicate sign, not good or bad. Raw and smooth limits match. Interpolation can overshoot; colors clip, metrics do not change. Representative selection and source hashes are recorded.'+selection_label,(2,4))
    _gzip(output/'smoothed-display-data.json.gz',{'schema':'tdn.adjacent-display/v1','policy':DISPLAY,'records':displays})
    artifacts={name:_sha(output/name) for name in ('resolution-atlas.pdf','chart-data.json.gz','smoothed-display-data.json.gz')}
    manifest={'schema':SCHEMA,'profile':data['profile'],'panels':panels,'sources':data['sources'],
        'artifacts':artifacts,'counts':{key:len(data[key]) for key in TABLES},'scope':SCOPE,
        'display_policy':DISPLAY,'metrics_changed_by_smoothing':False,'test_preview':_test_dpi is not None,
        'raw_observations_discarded':False,'omissions':omissions}
    _write(output/'manifest.json',manifest)
    sections=''.join(f'<section><h2>{html.escape(p["title"])}</h2><p>{html.escape(p["reading_guide"])}</p><img loading="lazy" src="{p["path"]}" alt="{html.escape(p["title"])}"></section>' for p in panels)
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>TDN resolution atlas</title><style>body{font:16px system-ui;margin:2em;background:#f4f7f9}section{background:white;padding:1em;margin:1em 0}img{max-width:100%}</style><h1>TDN 64²/128² resolution study</h1><p>'+html.escape(SCOPE)+'</p><p>'+html.escape(DISPLAY)+'</p><p><a href="resolution-atlas.pdf">Vector PDF</a> · <a href="chart-data.json.gz">Exact raw chart data</a> · <a href="smoothed-display-data.json.gz">Display interpolation data</a> · <a href="manifest.json">Source manifest</a></p>'+sections)
    return manifest


def run(ctx):
    data=collect(ctx)
    tables=write_tables(data,ctx.path/'tables')
    figures=build_figures(data,ctx.path/'figures',ctx.budget)
    result={'schema':SCHEMA,'status':'COMPLETED','scientific_outcome':'DESCRIPTIVE_VERIFIED_EVIDENCE',
        'math_outcome':'NA_REPORT_IS_NOT_PROOF','panels':len(figures['panels']),'tables':tables,
        'complete_planned_evidence':all(row['verified'] for row in data['stage_status']),
        'evaluation_rows':len(data['evaluation']),'stage_status':data['stage_status'],
        'scope':SCOPE,'display_policy':DISPLAY,'metrics_changed_by_smoothing':False,
        'monetary_cost':None,'energy_joules':None,'promotion_to_main_authorized':False}
    _write(ctx.path/'analysis.json',result)
    mechanisms=list(ctx.protocol.get('mechanisms',{}))
    if mechanisms:
        ctx.record('resolution-report/source-linked-atlas',[mechanisms[0]],
            metrics={'panels':len(figures['panels']),'evaluation_rows':len(data['evaluation']),'scientific_outcome':'REPORT_INVENTORY_ONLY'},
            config={'scope':SCOPE,'display_only_smoothing':True},checks=[
                check('source-linked-observations',all('_source' in row for row in data['evaluation']),True,'eq',category='correctness'),
                check('complete-planned-evidence',result['complete_planned_evidence'],True,'eq',category='gap'),
                check('report-is-not-proof',None,None,category='math')])
    return result
