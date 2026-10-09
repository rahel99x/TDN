"""Source-linked, partial-safe analytics for the A/B/C research portfolio.

No checkpoint is opened and no selection is performed here. Only prerequisites
verified by the engine contribute scientific measurements. Failed/missing units
remain visible as NA; descriptive post-hoc frontiers are never deployment rules.
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
import statistics
import textwrap

from tdn.analysis.frontier.report import _learning_ranges
from tdn.analysis.frontier.core import check, clean

SCHEMA = 'tdn.portfolio-analytics/v1'
TABLES = ('experiments', 'checks', 'catalog', 'learning_curves', 'validation',
          'endpoints', 'partial_endpoints', 'matched_configuration', 'posthoc_frontiers',
          'locked_frontiers', 'paired_field_summaries', 'scaling', 'prototypes',
          'temporal_queries', 'closure', 'selection', 'diagnostics', 'stage_status',
          'learned_parameters', 'amortization', 'temporal_amortization', 'timing_rounds', 'profiling', 'claims_accuracy', 'claims_cost', 'response_probes')
FILENAMES = {
    'rows.jsonl': 'experiments', 'catalog.json': 'catalog',
    'learning_curves.jsonl': 'learning_curves', 'validation_rows.json': 'validation',
    'endpoint_rows.json': 'endpoints', 'scaling_rows.json': 'scaling',
    'prototype_rows.json': 'prototypes', 'temporal_query_rows.json': 'temporal_queries',
    'closure_rows.json': 'closure', 'selection_rows.json': 'selection',
    'diagnostic_rows.json': 'diagnostics', 'diagnostics.json': 'diagnostics',
    'profile_rows.json': 'profiling', 'temporal_amortization.json': 'temporal_amortization', 'timing_rounds.json': 'timing_rounds',
}
METHOD_NOTES = ('Ours uses triangles; Theirs uses circles; analytic controls use squares. '
    'Local FNO implementations are not published-paper reproductions. Colors identify methods, not quality.')
SCOPE_NOTES = ('Only sealed prerequisites contribute science. Missing evidence is NA. '
    'Independent fields, not repeated schedules/grids/seeds, are statistical units. '
    'Observed learning ranges are not confidence intervals. No universal leaderboard.')
MAX_SOURCE_BYTES = 512 << 20


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _write(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')


def _gzip(path, value):
    with gzip.GzipFile(filename=str(path), mode='wb', mtime=0) as stream:
        stream.write(json.dumps(clean(value), separators=(',', ':'), allow_nan=False).encode())


def _rows(value):
    if isinstance(value, list):
        return [v for v in value if isinstance(v, dict)]
    if isinstance(value, dict):
        for key in ('rows', 'records', 'experiments', 'groups'):
            if isinstance(value.get(key), list):
                return [v for v in value[key] if isinstance(v, dict)]
    return []


def _pointer(value):
    if isinstance(value, dict):
        return next(('/' + key for key in ('rows', 'records', 'experiments', 'groups')
                     if isinstance(value.get(key), list)), '')
    return ''


def _read_source(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError(f'Not a regular bounded source: {path}')
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError(f'Source changed while reading: {path}')
    def bad(value):
        raise ValueError(f'Nonfinite JSON constant: {value}')
    value = ([json.loads(line, parse_constant=bad) for line in raw.decode().splitlines() if line.strip()]
             if path.suffix == '.jsonl' else json.loads(raw, parse_constant=bad))
    return value, hashlib.sha256(raw).hexdigest(), len(raw)


def _with_source(row, unit, path, sha, pointer, index):
    return {**row, '_source': {'unit': unit, 'path': str(path), 'sha256': sha,
             'pointer': f'{pointer}/{index}', 'line': index+1 if path.suffix == '.jsonl' else None,
             'evidence_status': 'VERIFIED_PREREQUISITE'}}


def flatten(row):
    """Convenience view only; exact nested original remains in chart data."""
    result = {}
    for key in ('config', 'effective_config', 'metrics', 'cost', 'costs', 'timing'):
        if isinstance(row.get(key), dict):
            result.update(row[key])
    if isinstance(result.get('cost'), dict):
        result.update(result['cost'])
    result.update(row)
    result.setdefault("horizon", result.get("final_time"))
    result.setdefault("median_seconds", result.get("cost_seconds"))
    if isinstance(row.get('errors'), list):
        for metric in ('upper_rms', 'upper_max'):
            errors = [e.get(metric) for e in row['errors']]
            result.setdefault(metric, max(errors) if errors and all(finite(e) for e in errors) else None)
    if isinstance(row.get('raw_timing'), dict):
        timing = row['raw_timing'].get('methods', {}).get(row.get('model_id'), {})
        for key in ('peak_allocated_bytes', 'peak_reserved_bytes'):
            result.setdefault(key, timing.get(key))
    return result


def _unit_kind(protocol, unit):
    entry = protocol.get('units', {}).get(unit, {})
    return entry.get('kind', unit)


def collect(ctx):
    data = {name: [] for name in TABLES}
    data.update(schema=SCHEMA, profile=ctx.protocol.get('profile', 'unknown'),
                device=str(ctx.device), protocol=ctx.protocol, sources=[], omissions=[])
    units = ctx.protocol.get('units', {})
    aggregate_available = any(_unit_kind(ctx.protocol, unit) == 'aggregate' for unit in ctx.prerequisites)
    freeze_available = any(_unit_kind(ctx.protocol, unit) == 'freeze' for unit in ctx.prerequisites)
    for unit in sorted(set(units) | set(ctx.prerequisites)):
        if unit == ctx.stage:
            continue
        base = ctx.prerequisites.get(unit)
        failure = getattr(ctx, 'stage_failures', {}).get(unit, {})
        status = {'unit': unit, 'kind': _unit_kind(ctx.protocol, unit),
                  'path': units.get(unit, {}).get('path'), 'planned_seconds': units.get(unit, {}).get('seconds'),
                  'status': 'VERIFIED' if base else failure.get('status', 'NA'),
                  'verified': bool(base), 'error': failure.get('error'),
                  'elapsed_seconds': failure.get('elapsed_seconds'), 'monetary_cost': None,
                  'operational_time_verified': failure.get('operational_time_verified', False),
                  'operational_diagnostic': dict(failure)}
        data['stage_status'].append(status)
        if not base:
            continue
        base = Path(base)
        if (base / 'summary.json').is_file():
            summary, sha, size = _read_source(base / 'summary.json')
            status.update(elapsed_seconds=summary.get('elapsed_seconds'),
                          scientific_outcome=summary.get('scientific_outcome'),
                          computational_status=summary.get('status'))
            data['sources'].append({'unit': unit, 'path': str(base / 'summary.json'), 'sha256': sha, 'bytes': size})
        for filename, table in FILENAMES.items():
            path = base / filename
            if not path.exists():
                continue
            ctx.budget.check()
            value, sha, size = _read_source(path)
            data['sources'].append({'unit': unit, 'path': str(path), 'sha256': sha, 'bytes': size})
            if table == 'catalog' and freeze_available and _unit_kind(ctx.protocol, unit) != 'freeze':
                continue  # Frozen ledger is authoritative; do not count train ledgers twice.
            selected = table
            if table == 'endpoints' and _unit_kind(ctx.protocol, unit) != 'aggregate':
                selected = 'partial_endpoints'
            if selected == 'partial_endpoints' and aggregate_available:
                continue  # Raw partition sources are listed; aggregate preserves every canonical row.
            data[selected].extend(_with_source(row, unit, path, sha, _pointer(value), index)
                                  for index, row in enumerate(_rows(value)))
        claims_path = base / 'claims.json'
        if claims_path.is_file():
            claims, sha, size = _read_source(claims_path)
            data['sources'].append({'unit': unit, 'path': str(claims_path), 'sha256': sha, 'bytes': size})
            for key in ('accuracy', 'cost'):
                data['claims_' + key].extend(_with_source(row, unit, claims_path, sha, '/' + key, index)
                    for index, row in enumerate(claims.get(key, [])))
            data.setdefault('claims_metadata', []).append({key: value for key, value in claims.items()
                                                         if key not in ('accuracy', 'cost')})
        path = base / 'comparisons.json'
        if path.exists():
            value, sha, size = _read_source(path)
            data['sources'].append({'unit': unit, 'path': str(path), 'sha256': sha, 'bytes': size})
            for key in ('matched_configuration', 'posthoc_frontiers', 'locked_frontiers', 'paired_field_summaries'):
                data[key].extend(_with_source(row, unit, path, sha, '/' + key, index)
                                 for index, row in enumerate(value.get(key, [])))
        # Small summaries and freeze decisions are retained in full for interpretation.
        for filename in ('prototype_summary.json', 'freeze.json', 'validation_frontier.json',
                         'training_plan.json', 'profiling.json', 'architecture.json', 'coverage.json',
                         'temporal_encoding.json', 'closure_fit.json', 'selection_fit.json',
                         'selection_phase_audit.json', 'selection_states.json'):
            path = base / filename
            if path.is_file():
                value, sha, size = _read_source(path)
                data.setdefault('auxiliary', []).append({'unit': unit, 'filename': filename, 'value': value,
                                                         '_source': {'path': str(path), 'sha256': sha}})
                data['sources'].append({'unit': unit, 'path': str(path), 'sha256': sha, 'bytes': size})
    for row in data['experiments']:
        data['checks'].extend({'experiment_id': row.get('experiment_id'), 'stage': row.get('stage'), **item,
            '_source': {**row['_source'], 'pointer': row['_source']['pointer'] + f'/checks/{i}'}}
            for i, item in enumerate(row.get('checks', [])))
    for row in data['catalog']:
        record = {'model_id': row.get('model_id'), 'family': row.get('family'), 'track': row.get('track'),
                  'seed': row.get('seed'), 'train_count': row.get('train_count'),
                  'parameter_report': row.get('parameter_report'), '_source': row['_source']}
        data['learned_parameters'].append(record)
        for index, probe in enumerate(row.get('response_probes', [])):
            data['response_probes'].append({**probe, 'model_id': row.get('model_id'),
                'family': row.get('family'), 'track': row.get('track'), 'seed': row.get('seed'),
                'train_count': row.get('train_count'), '_source': {**row['_source'],
                    'pointer': row['_source']['pointer'] + f'/response_probes/{index}'}})
    data['amortization'] = amortization(data)
    return data


def role(family, row=None):
    row = row or {}
    declared = row.get('role', row.get('ownership'))
    prototype_roles = {
        'compact_temporal_encoding': 'Ours', 'classical_dense_output': 'Theirs',
        'exact_cached_pairs': 'Analytic control', 'uncached_pairs': 'Analytic control',
        'gauss_two': 'Analytic control', 'gauss_four': 'Analytic control', 'fft_gauss_four': 'Analytic control',
        'instantaneous_fit': 'Analytic control', 'memory_persistence': 'Analytic control',
        'memory_fitted': 'Ours', 'full_state_instantaneous_oracle': 'Analytic control',
        'full_pairs': 'Analytic control', 'deterministic_l1_tail': 'Analytic control',
        'fitted_rank_rule': 'Ours'}
    if declared is None:
        declared = prototype_roles.get(family)
    if declared is None:
        try:
            from .models import MODEL_SPECS
            declared = MODEL_SPECS.get(family, {}).get('role')
        except ImportError:
            pass
    if str(declared).startswith('Ours'):
        return 'Ours'
    if declared in ('Ours', 'Theirs', 'Analytic control'):
        return declared
    name = str(family).lower()
    if name.startswith(('fno', 'direct_fno', 'df', 'etdrk', 'classical')):
        return 'Theirs'
    if any(token in name for token in ('analytic', 'normalized_frozen', 'gauss', 'quadrature')):
        return 'Analytic control'
    return 'Ours'


def style(family, row=None):
    category = role(family, row)
    fixed = {'rank1': '#0072B2', 'rank1_frozen': '#56B4E9', 'fno_small': '#E69F00',
             'fno_standard': '#D55E00', 'direct_fno': '#CC79A7', 'df': '#777777', 'rf': '#5555AA', 'etdrk4': '#222222'}
    palettes = {'Ours': ['#0072B2', '#17BECF', '#445DA8', '#4878CF', '#1594AA'],
                'Theirs': ['#E69F00', '#D55E00', '#CC79A7', '#777777', '#222222'],
                'Analytic control': ['#009E73', '#8C6D31', '#5F8F35', '#638777']}
    digest = int(hashlib.sha256(str(family).encode()).hexdigest()[:8], 16)
    index = digest
    try:
        from .models import MODEL_SPECS
        roster = [name for name in MODEL_SPECS if role(name) == category]
        if family in roster:
            index = roster.index(family)
    except ImportError:
        pass
    return {'color': fixed.get(family, palettes[category][index % len(palettes[category])]),
            'marker': {'Ours': '^', 'Theirs': 'o', 'Analytic control': 's'}[category],
            'linestyle': ['-', '--', ':', '-.'][(index // len(palettes[category])) % 4], 'linewidth': .55}


def label(row):
    family = row.get('family', row.get('method', row.get('model_id', 'unlabeled')))
    return f'{role(family, row)}: {family}'


def _cell(value):
    if isinstance(value, (dict, list)):
        value = json.dumps(clean(value), sort_keys=True, separators=(',', ':'))
    if isinstance(value, str) and value.lstrip().startswith(('=', '+', '-', '@')):
        return "'" + value
    return value


def write_tables(data, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    inventory = []
    for name in TABLES:
        rows = data.get(name, [])
        columns = sorted({key for row in rows for key in row}) or ['status', 'reason']
        with (output / f'{name}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, columns); writer.writeheader()
            for row in rows:
                writer.writerow({key: _cell(row.get(key)) for key in columns})
        inventory.append({'name': name, 'path': f'{name}.csv', 'rows': len(rows),
                          'sha256': hashlib.sha256((output / f'{name}.csv').read_bytes()).hexdigest()})
    return inventory


def amortization(data):
    """Measured time economics, never made-up monetary rates or amortization."""
    rows = []
    training = defaultdict(float)
    for row in data.get('catalog', []):
        cost = row.get('total_training_seconds', row.get('training_seconds'))
        if finite(cost):
            training[row.get('model_id')] += cost
    for source in data.get('locked_frontiers', []):
        row = flatten(source)
        candidate = row.get('candidate_seconds', row.get('model_seconds'))
        comparator = row.get('comparator_seconds', row.get('baseline_seconds'))
        offline = training.get(row.get('model_id'))
        margin = comparator-candidate if finite(candidate) and finite(comparator) else None
        rows.append({'model_id': row.get('model_id'), 'track': row.get('track'), 'horizon': row.get('horizon'),
            'offline_training_seconds': offline, 'inference_saving_seconds': margin,
            'training_only_break_even_queries': math.ceil(offline/margin) if offline is not None and margin and margin > 0 else None,
            'status': 'MEASURED_TRAINING_ONLY' if offline is not None and margin and margin > 0 else 'NA',
            'monetary_cost': None, 'reason': 'Training-only optimistic bound; full reference/tuning/preprocessing and failures remain stage costs; no price supplied.',
            '_source': source.get('_source')})
    return rows


def _blank(ax, reason='NA: no verified measurements for this panel'):
    ax.text(.5, .5, textwrap.fill(reason, 62), ha='center', va='center', transform=ax.transAxes, color='#6b7380')
    ax.set_xticks([]); ax.set_yticks([])


def range_plot(ax, records, x, y):
    """Same continuous range semantics as frontier, using portfolio identities."""
    groups = defaultdict(list)
    for row in records:
        groups[row['family']].append(row)
    measured = []
    for family, values in sorted(groups.items()):
        values.sort(key=lambda row: row['window_index'])
        sty = style(family)
        def draw(run):
            if not run:
                return
            bounds = [(max(r['window_left'], r['family_x_min']), min(r['window_right'], r['family_x_max'])) for r in run]
            xs = [bounds[0][0], *[(lo+hi)/2 for lo, hi in bounds], bounds[-1][1]]
            lower = [run[0]['y_min'], *[r['y_min'] for r in run], run[-1]['y_min']]
            upper = [run[0]['y_max'], *[r['y_max'] for r in run], run[-1]['y_max']]
            ax.fill_between(xs, lower, upper, color=sty['color'], alpha=.10, linewidth=0)
            for boundary in (lower, upper):
                ax.plot(xs, boundary, color=sty['color'], linestyle=sty['linestyle'], linewidth=.6, alpha=.95)
        run = []
        for row in values:
            if not row['observation_count']:
                draw(run); run = []; continue
            measured.append(row)
            if row['distinct_x_count'] > 1:
                if run and row['window_index'] != run[-1]['window_index']+1:
                    draw(run); run = []
                run.append(row)
            else:
                draw(run); run = []
                ax.vlines(row['x_min'], row['y_min'], row['y_max'], color=sty['color'], linewidth=.6)
                ax.plot([row['x_min']]*2, [row['y_min'], row['y_max']], linestyle='None', marker=sty['marker'],
                        color=sty['color'], markersize=3)
        draw(run)
        if any(r['observation_count'] for r in values):
            ax.plot([], [], **sty, markersize=4, label=label({'family': family}))
    if not measured:
        _blank(ax)
    else:
        if all(r['y_min'] > 0 for r in measured):
            ax.set_yscale('log')
        ax.legend(fontsize=5, ncol=2)
    ax.set_xlabel(x); ax.set_ylabel(y + ' (lower is better)'); ax.grid(alpha=.15)


def _scatter(ax, rows, x, y):
    groups = defaultdict(list); omitted = 0
    for item in rows:
        row = flatten(item)
        if finite(row.get(x)) and finite(row.get(y)):
            groups[row.get('family', row.get('method', row.get('model_id', 'unlabeled')))].append(row)
        else:
            omitted += 1
    if not groups:
        _blank(ax); return
    valid = [row for values in groups.values() for row in values]
    for family, values in sorted(groups.items()):
        sty = style(family, values[0])
        ax.scatter([r[x] for r in values], [r[y] for r in values], color=sty['color'], marker=sty['marker'],
                   s=19 if sty['marker'] == '^' else 10, alpha=.6, label=label(values[0]), rasterized=True)
    if all(r[x] > 0 for r in valid):
        ax.set_xscale('log')
    if all(r[y] > 0 for r in valid):
        ax.set_yscale('log')
    ax.set_xlabel(x); ax.set_ylabel(y); ax.grid(alpha=.15); ax.legend(fontsize=5, ncol=2)
    ax.text(.99, -.22, f'{len(valid)} observations; {omitted} unavailable. Repeated fields are paired.',
            ha='right', va='top', fontsize=5, transform=ax.transAxes)


def build_figures(data, output, budget=None):
    project = Path(__file__).resolve().parents[3]
    for name, suffix in (('MPLCONFIGDIR', 'matplotlib'), ('XDG_CACHE_HOME', 'xdg')):
        current = Path(os.environ.get(name, '/')).resolve()
        if project not in current.parents:
            current = project / 'cache' / 'portfolio-figures' / suffix
            os.environ[name] = str(current)
        current.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    _gzip(output / 'chart-data.json.gz', data)
    panels = []; ranges = []
    with PdfPages(output / 'portfolio-atlas.pdf') as pdf:
        def save(title, fig, guide, scope):
            if budget: budget.check()
            index = len(panels)+1
            name = f'{index:02d}-' + ''.join(c if c.isalnum() else '-' for c in title.lower()).strip('-') + '.png'
            footer = '\n'.join(textwrap.fill(prefix+text, 145) for prefix, text in (
                ('HOW TO READ: ', guide), ('METHODS: ', METHOD_NOTES), ('SCOPE: ', scope)))
            width, height = fig.get_size_inches(); extra = .22*len(footer.splitlines())+.35
            fig.set_size_inches(width, height+extra)
            fig.text(.025, .1/(height+extra), footer, fontsize=7, va='bottom', linespacing=1.3)
            fig.suptitle(title, fontsize=12)
            fig.tight_layout(rect=(.015, extra/(height+extra), .99, .94))
            fig.savefig(output/name, dpi=170); pdf.savefig(fig, dpi=170); plt.close(fig)
            panels.append({'id': index, 'title': title, 'path': name, 'reading_guide': guide, 'scope': scope,
                           'sha256': hashlib.sha256((output/name).read_bytes()).hexdigest()})
        def single(title, draw, guide, scope=SCOPE_NOTES):
            fig, ax = plt.subplots(figsize=(13, 6)); draw(ax); save(title, fig, guide, scope)
        def facets(title, values, keys, draw, guide, scope=SCOPE_NOTES):
            groups = defaultdict(list)
            for item in values:
                row = flatten(item)
                groups[tuple(str(row.get(key, 'NA')) for key in keys)].append(row)
            items = sorted(groups.items()) or [(tuple('NA' for _ in keys), [])]
            # Fixed six facets per page keeps text readable regardless of campaign size.
            for offset in range(0, len(items), 6):
                subset = items[offset:offset+6]
                columns = min(3, len(subset)); rows = math.ceil(len(subset)/columns)
                fig, axes = plt.subplots(rows, columns, figsize=(max(13, columns*5), rows*3.8+1), squeeze=False)
                for ax, (key, members) in zip(axes.flat, subset):
                    draw(ax, members); ax.set_title(' | '.join(f'{k}={v}' for k,v in zip(keys,key)), fontsize=7)
                for ax in list(axes.flat)[len(subset):]: ax.axis('off')
                suffix = f' ({offset//6+1})' if len(items)>6 else ''
                save(title+suffix, fig, guide, scope)

        def architecture(ax):
            ax.axis('off')
            boxes = [(.02,.65,.26,.20,'Full spatial field u\nPhysical parameters, requested h'),
                     (.37,.78,.28,.14,'Analytic DF base step\nHalf diffusion → reaction → half diffusion'),
                     (.36,.50,.29,.15,'Nonlinear physical interactions\nBEFORE output compression'),
                     (.02,.17,.27,.24,'Learned optional conditioner\nFeatures → nodes / weights / gain\nAblations isolate every choice'),
                     (.73,.48,.24,.21,'Base + correction\nNext field / rollout')]
            for x,y,w,h,text in boxes:
                ax.add_patch(plt.Rectangle((x,y),w,h,facecolor='#edf4f8',edgecolor='#436784',linewidth=.7))
                ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9)
            for start,end in (((.28,.76),(.37,.85)),((.28,.72),(.36,.59)),((.29,.3),(.42,.50)),((.65,.85),(.85,.69)),((.65,.58),(.73,.58))):
                ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','lw':.7})
            ax.text(.5,.06,'A: normalized controls and attribution     B: grounded changes     C: independent exploratory formulations',ha='center',fontsize=9)
        single('Architecture and information flow', architecture,
               'Arrows show information and computation, not an accuracy ranking.',
               'Schematic for the interaction-correction family. Competitors and high-risk kernels/closures have different computational graphs; exact configurations and parameter reports remain in chart data.')
        def statuses(ax):
            rows=data['stage_status']
            if not rows: _blank(ax); return
            colors=['#218c83' if r['verified'] else '#9ba6b5' for r in rows]
            ys=list(range(len(rows)))
            ax.barh(ys,[r.get('elapsed_seconds') or 0 for r in rows],color=colors,height=.6)
            ax.set_yticks(ys,[r['unit'] for r in rows],fontsize=6)
            for i,r in enumerate(rows): ax.text(0,i,' '+r['status'],va='center',fontsize=5)
            ax.set_xlabel('Measured stage elapsed seconds (lower for the same complete work)')
        single('Stage completion and complete campaign cost', statuses,
               'Shorter measured time is cheaper only for the same completed work. NA/failed stages are not zero-cost successes.',
               'Measured numerical stage time is separate from scheduler allocation and accounting. Failed/interrupted attempt costs may be unavailable; no monetary rate is assumed.')
        def verdicts(ax):
            rows=data['experiments']
            if not rows: _blank(ax); return
            counts=Counter((r.get('stage','unknown'),r.get('assessment',{}).get('verdict','NA')) for r in rows)
            stages=sorted({key[0] for key in counts}); bottom=[0]*len(stages)
            for verdict,color in [('GOOD','#218c83'),('BAD','#d56352'),('NA','#9ba6b5')]:
                values=[counts[(s,verdict)] for s in stages]
                ax.bar(stages,values,bottom=bottom,label=verdict,color=color)
                bottom=[a+b for a,b in zip(bottom,values)]
            ax.tick_params(axis='x',labelrotation=55); ax.legend(); ax.set_ylabel('Experiment records, not independent replications')
        single('Every experiment outcome including failures',verdicts,
               'GOOD means declared checks met; BAD means failed checks; NA means unresolved. Taller bars are not better models.')
        facets('Diagnosing the dominant error source',data['diagnostics'],('regime','diagnostic'),
               lambda ax,rows:_scatter(ax,[{**r,'family':r.get('component',r.get('diagnostic','diagnostic')),'role':'Analytic control'} for r in rows],'reference_uncertainty','error_rms'),
               'Error is favorable when smaller, but agreement below reference uncertainty is unresolved. Compare the explicitly named component and regime.',
               'Discrete dynamics and continuum compensation are separate targets. Temporal, quadrature, compression and reference-refinement probes are diagnostic, not independent final-solver comparisons.')
        def profile(ax,rows):
            valid=[r for r in rows if finite(r.get('seconds'))]
            if not valid:_blank(ax);return
            for i,row in enumerate(valid):
                sty=style(row.get('family','unlabeled'),row)
                ax.barh(i,row['seconds'],color=sty['color'],height=.6,hatch='' if role(row.get('family'))=='Ours' else '//')
            ax.set_yticks(range(len(valid)),[str(r.get('component','complete')) for r in valid],fontsize=6)
            ax.set_xlabel('Measured component seconds (lower for identical work)')
        facets('Runtime component profiling',data['profiling'],('regime','family'),profile,
               'Less time is favorable for the same operation and accuracy. Profiling overhead is excluded from deployment comparisons.',
               'Transforms, transport, products, features, synchronization and allocation are measured only when instrumented; a missing category is NA, never zero. Nested profiler events must not be summed as independent costs.')
        def claim_table(ax, values):
            if not values:
                _blank(ax, 'NA: primary claims require a complete verified aggregate; job completion alone cannot satisfy them.')
                return
            ax.axis('off'); cells=[]
            for row in values:
                ratio=row.get('geometric_control_over_candidate_rms', row.get('geometric_control_over_candidate_cost'))
                coverage=row.get('coverage_difference', row.get('candidate_joint_field_coverage'))
                cells.append([row.get('control_family'), row.get('verdict','NA'),
                    f'{ratio:.3g}' if finite(ratio) else 'NA',
                    f"{row['lower_ratio']:.3g}" if finite(row.get('lower_ratio')) else 'NA',
                    str(row.get('independent_fields','NA')), str(row.get('denominator','NA')),
                    f'{coverage:.3g}' if finite(coverage) else 'NA'])
            table=ax.table(cellText=cells,colLabels=['Control','Verdict','Ratio','Lower CI','Fields','Rows','Coverage*'],
                           cellLoc='left',loc='center',colWidths=[.28,.1,.12,.12,.09,.09,.14])
            table.auto_set_font_size(False);table.set_fontsize(6);table.scale(1,1.45)
            for i,row in enumerate(values,1):
                table[i,1].set_facecolor({'GOOD':'#b5ded5','BAD':'#efc7be','NA':'#e0e4eb'}.get(row.get('verdict'),'#e0e4eb'))
        for title,table_name in [('Primary learned attribution and neural accuracy gates','claims_accuracy'),
                                 ('Primary accuracy qualified solver cost gates','claims_cost')]:
            facets(title,data[table_name],('track','horizon','train_count'),claim_table,
                   'GOOD means the declared bounded gates met; BAD means insufficient effect/coverage or failures; NA means unresolved. Control/candidate ratios above one favor Ours. More repeated rows are not more independent fields.',
                   'Primary contrasts were declared before confirmation. Lower CI is a descriptive field/seed interval, unadjusted for multiplicity. Coverage* is candidate-minus-control pass fraction for accuracy, candidate all-query field coverage for cost. Smoke/development verdicts remain development diagnostics, not confirmation. Exact thresholds, unresolved pairs and full interval metadata are in claims tables.')
        for title,x,y in [('Training loss observed ranges','update','train_loss'),
                          ('Validation loss observed ranges','update','validation_loss'),
                          ('Validation error versus training compute','elapsed_seconds','validation_rms'),
                          ('Validation error versus examples seen','examples_seen','validation_rms')]:
            records=_learning_ranges(data['learning_curves'],x,y,max_windows=20)
            ranges.extend({'panel':title,**r} for r in records)
            facets(title,records,('track','phase','train_count'),lambda ax,rows,x=x,y=y:range_plot(ax,rows,x,y),
                   'Lower loss/error is better for the same objective. Left is less training work, not necessarily a better selected model.',
                   'Continuous translucent observed min/max ranges in shared x-windows; not confidence intervals. Straight boundary connections interpolate summaries, not observations. Empty windows and isolated checkpoints break bands; tuning configurations remain development evidence.')
        facets('Selected models and data efficiency',data['catalog'],('track',),
               lambda ax,rows:_scatter(ax,rows,'train_count','validation_objective'),
               'Lower-left is favorable: fewer independent training fields and lower held-out objective. Objective definitions must match.',
               'Validation-selected checkpoints, including initialization-selected and failed trials. Selection is not untouched confirmation; every record remains in catalog.csv.')
        facets('Matched schedule accuracy and inference cost',data['endpoints'],('track','horizon','schedule_id'),
               lambda ax,rows:_scatter(ax,rows,'median_seconds','upper_rms'),
               'Lower-left is favorable: less complete endpoint time and smaller RMS plus reference uncertainty.',
               'Exact schedule, final time and target are separated. Points repeat independent fields across methods, grids and seeds. Maximum-error requirements are retained separately in raw rows.')
        facets('Partial confirmation evidence',data['partial_endpoints'],('track','horizon'),
               lambda ax,rows:_scatter(ax,rows,'median_seconds','upper_rms'),
               'Lower-left is favorable within a matched workload; incomplete cohorts cannot establish confirmation.',
               'Only sealed available partitions, shown when the full aggregate is absent. Missing partitions remain NA; no complete-cohort claims or best-model selection are made.')
        for title,table in [('Validation locked accuracy cost frontier','locked_frontiers'),
                            ('Posthoc descriptive accuracy cost frontier','posthoc_frontiers')]:
            facets(title,data[table],('track','horizon'),lambda ax,rows,table=table:_scatter(ax,rows,'median_seconds','upper_rms' if table=='locked_frontiers' else 'rms_target'),
                   'Lower-left means less inference time and error on the identical target. Missing columns produce NA.',
                   'Locked choices were made on validation; post-hoc minima inspect confirmation truth and are descriptive only. Exact tolerances, infeasible candidates and schedule identities remain in the table.')
        def interval(ax,rows):
            values=[]
            for item in rows:
                if finite(item.get('geometric_control_over_candidate_rms')):
                    ci = item.get('error_log_ratio_interval', {})
                    item.update(geometric_ratio=item['geometric_control_over_candidate_rms'],
                        family=item.get('candidate_family'),
                        comparison=f"{item.get('candidate_family')} vs {item.get('control_family')}",
                        ci_low=math.exp(ci['lower']) if finite(ci.get('lower')) else None,
                        ci_high=math.exp(ci['upper']) if finite(ci.get('upper')) else None)
            for row in rows:
                point=row.get('geometric_ratio',row.get('effect',row.get('ratio')))
                lo=row.get('ci_low',row.get('lower'));hi=row.get('ci_high',row.get('upper'))
                if finite(point): values.append((row,point,lo,hi))
            if not values: _blank(ax,'NA: no declared independent-field interval schema available; exact summaries remain in CSV');return
            for i,(row,point,lo,hi) in enumerate(values):
                sty=style(row.get('family',row.get('candidate','unknown')),row)
                ax.scatter([point],[i],marker=sty['marker'],color=sty['color'],s=20)
                if finite(lo) and finite(hi): ax.hlines(i,lo,hi,color=sty['color'],linewidth=.6)
            ax.axvline(1,color='#777777',linewidth=.5);ax.set_yticks(range(len(values)),[str(v[0].get('comparison',v[0].get('model_id','pair'))) for v in values],fontsize=5)
            ax.set_xlabel('Declared ratio and independent-field interval (see ratio definition)')
        primary = data['protocol'].get('selection', {}).get('primary_family', 'quad2_conditioned')
        displayed_controls = set(data['protocol'].get('hypothesis_thresholds', {}).get('learned_comparators', [])) | {'quad2_fixed', 'quad4_full', 'analytic_quad_cubic', 'df', 'etdrk4', 'fno_small', 'fno_standard'}
        primary_intervals = [r for r in data['paired_field_summaries'] if
            (r.get('candidate_family') == primary and r.get('control_family') in displayed_controls) or
            (r.get('control_family') == primary and r.get('candidate_family') in displayed_controls)]
        facets('Primary paired field statistical uncertainty',primary_intervals,('track','horizon','regime','train_count'),interval,
               'Control/candidate RMS ratio: above one favors the first listed candidate, below one favors its control. Field-bootstrap intervals are separate from observed training ranges.',
               'Plot shows the declared primary family against normalization, learned-attribution and main solver controls for legibility. Every other measured pair, denominator and interval is preserved in paired_field_summaries.csv; no selection by observed effect.')
        for title,x,y in [('Generalization and failure map','upper_max','upper_rms'),
                          ('Spectral error decomposition','spectral_low_rms','spectral_high_rms'),
                          ('Mean and spatial error','mean_error','centered_rms')]:
            facets(title,data['endpoints'],('track','regime','horizon'),lambda ax,rows,x=x,y=y:_scatter(ax,rows,x,y),
                   'Lower-left is better: both error components are smaller on matched targets. Failures/unaccepted references stay in raw data.')
        intermediate=[]
        for row in data['endpoints']:
            for index,item in enumerate(row.get('intermediates',[]) or []):
                if isinstance(item,dict): intermediate.append({**row,**item,'_source':row.get('_source'),'intermediate_index':index})
        facets('Rollout intermediate accuracy and stability',intermediate,('track','regime'),
               lambda ax,rows:_scatter(ax,rows,'time','upper_rms'),
               'Lower error is better at the same requested time. Right is a longer horizon, not necessarily greater quality.',
               'Each intermediate truth must have an accepted reference. No interpolated teacher is invented; absent trajectory measurements remain NA.')
        facets('Accuracy qualified scaling',data['scaling'],('track','batch_size'),
               lambda ax,rows:_scatter(ax,rows,'median_seconds','upper_rms'),
               'Lower-left is faster and more accurate. Large-grid timings without qualified error are not speed wins.')
        facets('Cold versus warm latency',data['endpoints'],('track','horizon'),
               lambda ax,rows:_scatter(ax,rows,'cold_seconds','median_seconds'),
               'Lower-left is less first-invocation and warmed runtime. Both include the declared complete operation.')
        facets('Peak memory and warmed cost',data['scaling'],('track','batch_size'),
               lambda ax,rows:_scatter(ax,rows,'peak_allocated_bytes','median_seconds'),
               'Lower-left is less observed memory and time for matched accuracy/workload. CPU process memory is not GPU VRAM.')
        response_display=[]
        for row in data['response_probes']:
            response=row.get('response') or {}
            pending=[response.get('gain')];values=[]
            while pending:
                value=pending.pop(0)
                if isinstance(value,list):pending[:0]=value
                elif finite(value):values.append(value)
            for component,value in enumerate(values):
                response_display.append({**row,'gain':value,'gain_component':component})
        facets('Learned effective gain response',response_display,('track','family','split'),
               lambda ax,rows:_scatter(ax,rows,'horizon','gain'),
               'Gain has no universal higher-is-better direction. Compare variation across times and states with the matched fitted/analytic controls.',
               'First declared training/validation state probes use no reference truth. Band gains remain separate observations, not an averaged gain; raw response features, nodes and weights remain source-linked. Similar effective gains do not prove parameter identifiability.')
        def parameters(ax):
            rows=data['learned_parameters']
            if not rows:_blank(ax);return
            text=[]
            for row in rows[:18]:
                text.append(str(row.get('model_id'))+'\n'+textwrap.shorten(json.dumps(row.get('parameter_report'),sort_keys=True),width=135))
            ax.axis('off');ax.text(.01,.98,'\n\n'.join(text),va='top',fontsize=6,transform=ax.transAxes)
            if len(rows)>18:ax.text(.01,.01,f'All {len(rows)} exact parameter reports in learned_parameters.csv; first 18 shown here.',fontsize=7)
        single('Learned parameters and identifiability evidence',parameters,
               'Parameters are not inherently better when larger or smaller. Compare effective responses and matched ablations.',
               'Exact learned node/weight/feature responses are retained. Several parameterizations can realize the same effective correction; no identifiability proof follows from a successful fit.')
        def prototypes(ax):
            rows=data['prototypes']
            if not rows:_blank(ax);return
            counts=Counter((r.get('prototype_id','unknown'),r.get('status','NA')) for r in rows)
            labels=[f'{a}: {b}' for a,b in sorted(counts)]
            ax.barh(labels,[counts[k] for k in sorted(counts)],color=['#9ba6b5' if 'NA' in k else '#407ba1' for k in labels])
            ax.set_xlabel('Prototype records (not independent scientific successes)')
        single('Protected high risk exploration including negative outcomes',prototypes,
               'Counts show coverage; larger counts do not mean greater promise. Every status and invalidation check remains visible.',
               'New to TDN is not novelty in the literature. Numerical prototypes are development diagnostics, not neural/FNO or GPU superiority claims.')
        closure_display = []
        for row in data['closure']:
            for method, error in row.get('errors', {}).items():
                closure_display.append({**row,'method':method,'error':error,
                    'target_norm':math.sqrt(sum(v*v for v in row['target']))})
        for title,table,x,y in [('Amortized temporal query workload','temporal_queries','median_seconds','relative_rms'),
                                ('Coarse closure and unresolved information','closure','target_norm','error'),
                                ('Selecting work before expensive computation','selection','median_seconds','relative_error')]:
            facets(title,closure_display if table=='closure' else data[table],('split',),lambda ax,rows,x=x,y=y:_scatter(ax,rows,x,y),
                   'Smaller error is better; less complete work is better at equal accuracy. Feature/selection costs belong in complete time.')
        facets('Repeated query build and refresh amortization',data['temporal_amortization'],(),
               lambda ax,rows:_scatter(ax,rows,'query_count','total_seconds'),
               'Lower total time is better for the same number of requested horizons. Right means more queries.',
               'Measured build plus query count times median query cost; a scenario calculation, not an observed batched execution. Reuse requires fixed state/operator/horizon range; state refresh costs remain included.')
        def economics(ax):
            rows=data['amortization']
            valid=[r for r in rows if finite(r.get('training_only_break_even_queries'))]
            if not valid:_blank(ax,'NA: no positive measured matched-accuracy margin with known offline cost. No invented amortization or monetary price.');return
            ax.bar(range(len(valid)),[r['training_only_break_even_queries'] for r in valid],color='#0072B2')
            ax.set_yscale('log');ax.set_xticks(range(len(valid)),[r.get('model_id','unknown') for r in valid],rotation=60,fontsize=5)
            ax.set_ylabel('Training-only optimistic break-even query count (lower is better)')
        single('Amortization and deployment economics',economics,
               'Fewer break-even queries is better if both solvers satisfy the same accuracy. No positive margin means no amortization.',
               'Training-only lower bound when computable; reference generation, tuning, failures, verification, rejection and fallback can increase cost. Money remains NA without an explicit rate.')
    _gzip(output/'learning-range-data.json.gz',{'schema':'tdn.portfolio-ranges/v1','records':ranges,
          'meaning':'Observed low/high range, not a confidence interval; straight connections between adjacent dense window centers.'})
    manifest={'schema':'tdn.portfolio-figures/v1','panels':panels,'profile':data['profile'],
              'pdf':'portfolio-atlas.pdf','chart_data':'chart-data.json.gz',
              'chart_data_sha256':hashlib.sha256((output/'chart-data.json.gz').read_bytes()).hexdigest(),
              'learning_range_data':'learning-range-data.json.gz','stage_status':data['stage_status'],
              'counts':{k:len(data[k]) for k in TABLES},'sources':data['sources'],
              'raw_observations_discarded':False,'scope':SCOPE_NOTES}
    _write(output/'manifest.json',manifest)
    body=''.join(f'<section><h2>{html.escape(p["title"])}</h2><p>{html.escape(p["reading_guide"])}</p><img loading="lazy" src="{p["path"]}" alt="{html.escape(p["title"])}"><p>{html.escape(p["scope"])}</p></section>' for p in panels)
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>TDN research portfolio</title><style>body{font:16px system-ui;margin:2em;background:#f5f7fa}section{background:white;padding:1em;margin:1em 0}img{max-width:100%}</style><h1>TDN A/B/C research portfolio</h1><p>'+html.escape(SCOPE_NOTES)+'</p><p><a href="portfolio-atlas.pdf">PDF atlas</a> · <a href="chart-data.json.gz">Every chart source</a> · <a href="learning-range-data.json.gz">Exact observed ranges</a> · <a href="manifest.json">Provenance</a></p>'+body)
    return manifest


def run(ctx):
    data=collect(ctx)
    tables=write_tables(data,ctx.path/'tables')
    manifest=build_figures(data,ctx.path/'figures',ctx.budget)
    result={'schema':SCHEMA,'verified_experiments':len(data['experiments']),
            'stage_status':data['stage_status'],'panels':len(manifest['panels']),'tables':tables,
            'complete_planned_evidence':all(row['verified'] for row in data['stage_status']),
            'scientific_outcome':'DESCRIPTIVE_VERIFIED_EVIDENCE' if data['experiments'] else 'NA',
            'scope':SCOPE_NOTES,'monetary_cost':None}
    _write(ctx.path/'analysis.json',result)
    mids=list(ctx.protocol.get('mechanisms',{}))
    if mids:
        ctx.record('report/source-linked-atlas',[mids[0]],metrics={'panels':len(manifest['panels']),
            'verified_experiments':len(data['experiments']),'scientific_outcome':'REPORT_INVENTORY_ONLY'},
            config={'scope':'Inventory does not establish any architecture claim'},checks=[
                check('source-linked-data',all('_source' in r for r in data['experiments']),True,'eq',category='correctness'),
                check('planned-evidence-complete',result['complete_planned_evidence'],True,'eq',category='gap'),
                check('report-not-a-mathematical-proof',None,None,category='math')])
    return result
