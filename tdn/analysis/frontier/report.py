"""Complete, source-linked analytics for the five-gate experiment.

Figures are descriptive. They retain initialization, infeasible and missing
outcomes; no plot pools different spatial targets or final simulation times.
The compressed chart source preserves every observation and every check.
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
import xml.etree.ElementTree as ET

STAGES = ('audit', 'screen', 'prepare', 'train', 'confirm_prepare', 'confirm', 'scaling', 'policy')
VERDICTS = ('GOOD', 'BAD', 'NA')
COLORS = {'GOOD': '#218c83', 'BAD': '#d56352', 'NA': '#9ba6b5'}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _read(path):
    def reject(value):
        raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def _rows(value):
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ('rows', 'records', 'comparisons', 'frontiers'):
            if isinstance(value.get(key), list):
                return value[key]
    return []


def _write(path, value):
    from tdn.runtime.metadata import write_json
    write_json(path, value)


def _gzip_json(path, value):
    with Path(path).open('wb') as stream:
        with gzip.GzipFile(fileobj=stream, mode='wb', mtime=0) as zipped:
            zipped.write(json.dumps(value, separators=(',', ':'), allow_nan=False).encode())


def collect(ctx, verifier=None):
    """Only verified sources may contribute affirmative scientific analytics."""
    if verifier is None:
        from .engine import verify_science
        verifier = verify_science
    from .core import validate_row
    from tdn.research.protocol import digest
    from tdn.runtime.metadata import software_metadata
    output = {'schema': 'tdn.frontier-chart-data/v1', 'profile': ctx.protocol['profile'],
              'stage_status': {}, 'rows': [], 'learning_curves': [], 'catalog': [],
              'confirmation': [], 'comparisons': [], 'scaling': [], 'policy': [],
              'policy_summaries': [], 'scaling_comparisons': [],
              'sources': [], 'costs': [], 'protocol': ctx.protocol}
    current_source = software_metadata()['source_tree_sha256']
    execution_path = ctx.path / 'execution.json'
    lineage = _read(execution_path).get('prerequisites', {}) if execution_path.exists() else {}
    names = {'learning_curves.jsonl': 'learning_curves', 'catalog.json': 'catalog',
             'confirmation_rows.json': 'confirmation', 'comparisons.json': 'comparisons',
             'scaling_rows.json': 'scaling', 'policy_rows.json': 'policy',
             'policy_summary.json': 'policy_summaries', 'scaling_comparisons.json': 'scaling_comparisons'}
    prior_hashes = {}
    for stage in STAGES:
        ctx.budget.check()
        path = ctx.prerequisites.get(stage)
        lengths = {key: len(value) for key, value in output.items() if isinstance(value, list)}
        previous_keys = set(output)
        try:
            if path is None:
                raise ValueError('Prerequisite is absent')
            path = Path(path)
            if lineage.get(stage, {}).get('verification') in ('INVALID', 'MISSING'):
                raise ValueError('Worker rejected prerequisite lineage')
            manifest = verifier(ctx.protocol, path)
            if manifest.get('stage') != stage or manifest.get('source_tree_sha256') != current_source:
                raise ValueError('Prerequisite stage or executable source differs')
            preceding = STAGES[:STAGES.index(stage)]
            if set(prior_hashes) != set(preceding) or manifest.get('prerequisites', {}) != prior_hashes:
                raise ValueError('Prerequisite manifest chain differs from the supplied preceding stages')
            if (path / 'stage.json').exists() and _read(path / 'stage.json').get('status') != 'COMPLETED':
                raise ValueError('Wrapper did not complete')
            if (path / 'execution.json').exists():
                seal = _read(path / 'workflow-seal.json')
                expected_files = ('execution.json', 'protocol.json', 'stage.json', 'science_manifest.json')
                if (seal.get('schema') != 'tdn.frontier/v1' or seal.get('schema_version') != 1
                        or seal.get('protocol_sha256') != digest(ctx.protocol)
                        or set(seal.get('files', {})) != set(expected_files)):
                    raise ValueError('Wrapper protocol differs')
                for name in expected_files:
                    if seal.get('files', {}).get(name) != hashlib.sha256((path / name).read_bytes()).hexdigest():
                        raise ValueError('Wrapper source files are unsealed or changed')
                execution = _read(path / 'execution.json')
                record = _read(path / 'stage.json')
                expected = {'stage': stage, 'profile': ctx.protocol['profile'], 'protocol_sha256': digest(ctx.protocol)}
                if any(execution.get(k) != v or record.get(k) != v for k,v in expected.items()):
                    raise ValueError('Wrapper stage/profile/protocol differs from scientific identity')
                if execution.get('software', {}).get('source_tree_sha256') != current_source:
                    raise ValueError('Wrapper executable source differs')
            raw = _read(path / 'rows.json')
            rows = [validate_row(row) for row in raw['rows']]
            output['rows'].extend(rows)
            summary = _read(path / 'summary.json')
            output['stage_status'][stage] = {'status': 'VERIFIED', 'experiments': len(rows),
                'scientific_outcome': summary.get('scientific_outcome'),
                'manifest_sha256': hashlib.sha256((path / 'science_manifest.json').read_bytes()).hexdigest()}
            output['costs'].append({'stage': stage, 'numerical_seconds': summary.get('elapsed_seconds'),
                'budget_seconds': ctx.protocol.get('budgets', {}).get(stage, {}).get('seconds'),
                'allocation_seconds': None, 'monetary_cost': None,
                'scope': 'Numerical stage time; allocation accounting and tariffs not available here'})
            for name, key in names.items():
                source = path / name
                if not source.exists():
                    continue
                if name.endswith('.jsonl'):
                    values = [json.loads(line) for line in source.read_text().splitlines() if line]
                else:
                    document = _read(source)
                    values = _rows(document)
                    if key == 'catalog':
                        values += document.get('tuning_records', [])
                    if key == 'comparisons':
                        output['comparison_document'] = document
                output[key].extend(values)
                output['sources'].append({'stage': stage, 'path': str(source),
                    'sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'rows': len(values)})
            if stage == 'policy' and (path/'gate.json').is_file():
                output['policy_gate'] = _read(path/'gate.json')
            output['sources'].append({'stage': stage, 'path': str(path / 'rows.json'),
                'sha256': hashlib.sha256((path / 'rows.json').read_bytes()).hexdigest(), 'rows': len(rows)})
            prior_hashes[stage] = hashlib.sha256((path / 'science_manifest.json').read_bytes()).hexdigest()
        except (OSError, KeyError, ValueError, TypeError) as error:
            for key, length in lengths.items():
                del output[key][length:]
            for key in set(output) - previous_keys:
                del output[key]
            output['stage_status'][stage] = {'status': 'MISSING_OR_INVALID', 'experiments': 0, 'reason': str(error)}
    accounting = ctx.path.parent / 'state' / 'scheduler-accounting.json'
    if accounting.is_file():
        snapshot = _read(accounting)
        _write(ctx.path / 'accounting-snapshot.json', snapshot)
        output['scheduler_accounting'] = snapshot
        output['sources'].append({'path': str(ctx.path / 'accounting-snapshot.json'),
            'sha256': hashlib.sha256((ctx.path / 'accounting-snapshot.json').read_bytes()).hexdigest(),
            'scope': 'Sealed snapshot consumed at report creation; collect may later refresh workflow accounting'})
        allocations = {r['workflow_stage']: r for r in snapshot.get('records', [])
                       if r.get('record_kind') == 'allocation' and r.get('terminal_state') is True}
        for row in output['costs']:
            allocation = allocations.get(row['stage'], {})
            row['allocation_seconds'] = allocation.get('elapsed_seconds')
            row['allocated_cpu_seconds'] = allocation.get('allocated_cpu_seconds')
            row['accounting_collected_at'] = snapshot.get('collected_at')
    tests=[]; test_sources=[]
    for stage in (*STAGES,'report'):
        path=ctx.path.parent/'reporter-tests'/stage/'tests.xml'
        if not path.is_file(): continue
        try:
            if path.is_symlink() or path.stat().st_size > 16 << 20:
                raise ValueError('Native JUnit must be a bounded regular file')
            raw=path.read_bytes()
            if b'<!DOCTYPE' in raw.upper(): raise ValueError('JUnit document declarations are refused')
            tree=ET.fromstring(raw)
            occurrences=[]
            for index,case in enumerate(tree.iter('testcase')):
                result='FAILED' if case.find('failure') is not None or case.find('error') is not None else 'SKIPPED' if case.find('skipped') is not None else 'PASSED'
                seconds=case.get('time'); seconds=float(seconds) if seconds else None
                occurrences.append({'stage':stage,'occurrence':index,'name':case.get('name'),
                    'classname':case.get('classname'),'status':result,'seconds':seconds if finite(seconds) else None})
            tests.extend(occurrences)
            test_sources.append({'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'stage':stage,'status':'READ_COMPLETE'})
        except (OSError, ValueError, ET.ParseError) as error:
            test_sources.append({'path':str(path),'stage':stage,'status':'INVALID_OR_INCOMPLETE','reason':str(error)})
    output['software_tests']=tests
    output['software_test_sources']=test_sources
    _write(ctx.path/'junit-snapshot.json',{'occurrences':tests,'sources':test_sources,
        'scope':'Every native reporter-test occurrence; repeated GPU cases across allocations are separate executions, never independent scientific observations'})
    return output


def aggregate(data):
    from .core import check, score_checks
    results = []
    declared = data['protocol'].get('mechanisms', data['protocol'].get('gates', {}))
    for identity in ('G1', 'G2', 'G3', 'G4', 'G5'):
        spec = declared.get(identity, {})
        related = [row for row in data['rows'] if identity in row.get('mechanism_ids', row.get('gate_ids', []))]
        checks = [dict(item, check_id=f"{row['stage']}/{row['experiment_id']}/{item['check_id']}")
                  for row in related for item in row['checks']]
        observed = sorted({row['stage'] for row in related})
        missing = [s for s in spec.get('stages', []) if s != 'report' and s not in observed]
        for stage in missing:
            checks.append(check(f'coverage/{identity}/{stage}', None, True, 'eq', category='gap',
                                reason='No verified executable evidence for this gate stage'))
        if not related:
            checks.append(check('no-mathematical-evidence', None, None, category='math'))
        results.append({'id': identity, 'name': spec.get('name', identity), **score_checks(checks),
            'experiment_count': len(related), 'expected_stages': spec.get('stages', []),
            'observed_stages': observed, 'missing_stages': missing,
            'failed_check_ids': [v['check_id'] for v in checks if v['verdict'] == 'BAD'],
            'na_check_ids': [v['check_id'] for v in checks if v['verdict'] == 'NA'],
            'gap_assessment': score_checks([v for v in checks if v['category'] == 'gap']),
            'math_assessment': score_checks([v for v in checks if v['category'] == 'math'])})
    return results


def _key(row, names):
    return tuple(json.dumps(row.get(k), sort_keys=True) for k in names)


def _family(row):
    return str(row.get('family', row.get('candidate_family', row.get('model_id', 'unlabeled'))))


def _blank(ax, reason='NA — no verified measurements for this panel'):
    ax.text(.5, .5, reason, ha='center', va='center', transform=ax.transAxes, wrap=True, color='#58677b')
    ax.set_xticks([]); ax.set_yticks([])


def _label(ax, x, y):
    ax.set_xlabel(x); ax.set_ylabel(y); ax.grid(alpha=.16)


def _lines(ax, curves, x, y):
    groups = defaultdict(list)
    for row in curves:
        if finite(row.get(x)) and finite(row.get(y)):
            groups[_key(row, ('model_id', 'family', 'track', 'seed', 'train_count', 'phase'))].append(row)
    colors = {}
    for key, rows in sorted(groups.items()):
        rows.sort(key=lambda r: r[x])
        family = _family(rows[0]); label = f"{family}/{rows[0].get('track', '?')}"
        colors.setdefault(label, f'C{len(colors) % 10}')
        ax.plot([r[x] for r in rows], [r[y] for r in rows], alpha=.45, lw=.7,
                color=colors[label], label=label if sum(1 for v in ax.lines if v.get_label() == label) == 0 else None)
    if not groups:
        _blank(ax)
    else:
        if all(r.get(y, 0) > 0 for values in groups.values() for r in values):
            ax.set_yscale('log')
        ax.legend(fontsize=6, ncol=2)
    _label(ax, x, y)


def _bars(ax, values, xlabel='', ylabel='', colors=None):
    if not values:
        _blank(ax); return
    labels, heights = zip(*values)
    ax.bar(range(len(labels)), heights, color=colors or '#407ba1')
    ax.set_xticks(range(len(labels)), labels, rotation=70, ha='right', fontsize=6)
    _label(ax, xlabel, ylabel)


def _facets(plt, rows, keys, draw, title, *, maximum_columns=3):
    groups = defaultdict(list)
    for row in rows:
        groups[_key(row, keys)].append(row)
    groups = groups or {tuple('NA' for _ in keys): []}
    columns = min(maximum_columns, len(groups)); count = math.ceil(len(groups) / columns)
    fig, axes = plt.subplots(count, columns, figsize=(6 * columns, 4 * count + 1), squeeze=False)
    for ax, (key, values) in zip(axes.flat, sorted(groups.items())):
        draw(ax, values)
        ax.set_title(' | '.join(f'{name}={value}' for name, value in zip(keys, key)), fontsize=8)
    for ax in list(axes.flat)[len(groups):]:
        ax.axis('off')
    fig.suptitle(title, fontsize=13)
    return fig


def _scatter(ax, rows, x, y, *, log=True):
    groups = defaultdict(list)
    missing = 0
    for row in rows:
        if finite(row.get(x)) and finite(row.get(y)) and (not log or row[x] > 0 and row[y] > 0):
            groups[_family(row)].append(row)
        else:
            missing += 1
    for name, values in sorted(groups.items()):
        ax.scatter([r[x] for r in values], [r[y] for r in values], s=9, alpha=.4, label=name, rasterized=True)
    if not groups:
        _blank(ax)
    else:
        if log:
            ax.set_xscale('log'); ax.set_yscale('log')
        ax.legend(fontsize=6, ncol=2)
    ax.text(.99, -.28, f'All {sum(map(len, groups.values()))} finite points; unavailable/nonpositive {missing}',
            va='top', ha='right', transform=ax.transAxes, fontsize=6)
    _label(ax, x, y)


def build_figures(data, output, budget=None):
    """Write all plot sources, every-cell lookups, standalone PNG/PDF and HTML."""
    # The allocated workflow configures storage already. Direct API callers
    # must also avoid Matplotlib/fontconfig falling back to a system temp/cache.
    project = Path(__file__).resolve().parents[3]
    for variable, suffix in (('MPLCONFIGDIR', 'matplotlib'), ('XDG_CACHE_HOME', 'xdg')):
        current = Path(os.environ.get(variable, '/')).expanduser().resolve()
        if project not in current.parents:
            current = project / 'cache' / 'frontier-figures' / suffix
            os.environ[variable] = str(current)
        current.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import ListedColormap
    import numpy as np
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    sources = output / 'chart-data.json.gz'; _gzip_json(sources, data)
    rows, curves = data['rows'], data['learning_curves']
    catalog, endpoints = data['catalog'], data['confirmation']
    comparisons, scaling, policy = data['comparisons'], data['scaling'], data['policy']
    check_index = [{'cell': index, 'stage': row['stage'], 'experiment_id': row['experiment_id'],
                    'check_id': item['check_id'], 'category': item['category'], 'verdict': item['verdict'],
                    'measured': item.get('measured'), 'target': item.get('target')}
                   for index, (row, item) in enumerate((r, c) for r in rows for c in r['checks'])]
    _gzip_json(output / 'check-cell-index.json.gz', check_index)
    panels = []
    pdf_path = output / 'frontier-atlas.pdf'
    with PdfPages(pdf_path, metadata={'Title': 'TDN five-gate frontier: complete measured analytics',
             'Subject': 'Finite evidence; matched final times and spatial targets; unavailable results remain NA'}) as pdf:
        def save(title, fig, note):
            if budget: budget.check()
            ordinal = len(panels) + 1
            name = f'{ordinal:02d}-' + ''.join(c if c.isalnum() else '-' for c in title.lower()).strip('-') + '.png'
            fig.text(.012, .006, note, ha='left', va='bottom', fontsize=6, wrap=True)
            fig.tight_layout(rect=(0, .035, 1, .95))
            fig.savefig(output / name, dpi=170)
            pdf.savefig(fig, dpi=170)
            panels.append({'id': ordinal, 'title': title, 'path': name, 'note': note,
                           'sha256': hashlib.sha256((output / name).read_bytes()).hexdigest()})
            plt.close(fig)
        def single(title, draw, note):
            fig, ax = plt.subplots(figsize=(12, 6)); ax.set_title(title); draw(ax); save(title, fig, note)

        def costs(ax):
            v = [r for r in data['costs'] if finite(r.get('numerical_seconds'))]
            if not v: _blank(ax); return
            x = np.arange(len(v)); ax.bar(x-.24, [r['numerical_seconds'] for r in v], .24, label='measured stage seconds')
            ax.bar(x, [r.get('budget_seconds') or 0 for r in v], .24, label='frozen compute ceiling')
            allocation = [(i,r['allocation_seconds']) for i,r in enumerate(v) if finite(r.get('allocation_seconds'))]
            if allocation: ax.bar([i+.24 for i,_ in allocation], [value for _,value in allocation], .24, label='terminal allocation seconds')
            ax.set_xticks(x, [r['stage'] for r in v], rotation=30); ax.legend(); _label(ax, 'stage', 'seconds')
        single('Stage compute and frozen budgets', costs,
               'Terminal allocation rows only, never summed with batch/step rows. Running report job is partial and excluded. Exact accounting snapshot sealed; queue time and tariffs remain NA.')
        def verdicts(ax):
            groups = sorted({r['stage'] for r in rows}); bottom = np.zeros(len(groups))
            for verdict in VERDICTS:
                heights = [sum(r['stage'] == stage and r['assessment']['verdict'] == verdict for r in rows) for stage in groups]
                ax.bar(groups, heights, bottom=bottom, color=COLORS[verdict], label=verdict); bottom += heights
            if not groups: _blank(ax)
            ax.legend(); _label(ax, 'stage', 'all experiment records')
        single('Complete experiment verdicts', verdicts, 'A required cost failure can make an otherwise accurate experiment BAD. Scores measure evidence attainment, not model quality.')
        def checks(ax):
            categories = sorted({r['category'] for r in check_index}); bottom = np.zeros(len(categories))
            for verdict in VERDICTS:
                heights = [sum(r['category'] == category and r['verdict'] == verdict for r in check_index) for category in categories]
                ax.bar(categories, heights, bottom=bottom, color=COLORS[verdict], label=verdict); bottom += heights
            if not categories: _blank(ax)
            ax.legend(); _label(ax, 'check category', 'all individual checks')
        single('Correctness math gap and utility checks', checks, 'Numerical agreement is finite evidence, never a general proof. Missing evidence is retained as NA.')
        def check_map(ax):
            if not check_index: _blank(ax); return
            width = min(512, len(check_index)); height = math.ceil(len(check_index) / width)
            ax.figure.set_size_inches(12, max(6, height / 120 + 2))
            cells = np.full(width * height, 3, dtype=np.uint8)
            cells[:len(check_index)] = [VERDICTS.index(r['verdict']) for r in check_index]
            ax.imshow(cells.reshape(height, width), cmap=ListedColormap([COLORS[v] for v in VERDICTS]+['white']), vmin=0, vmax=3, interpolation='nearest', aspect='auto')
            _label(ax, f'row-major column (width {width})', 'row')
        single('Every individual check', check_map,
               f'{len(check_index)} cells; no downsampling. Teal GOOD, coral BAD, gray NA, white padding. Exact identities and values in check-cell-index.json.gz.')
        for title, x, y in (('Training loss by update', 'update', 'train_loss'),
                             ('Validation loss by update', 'update', 'validation_loss'),
                             ('Validation error versus training time', 'elapsed_seconds', 'validation_rms'),
                             ('Validation error versus examples seen', 'examples_seen', 'validation_rms')):
            figure = _facets(plt, curves, ('track',), lambda ax, vals: _lines(ax, vals, x, y), title)
            save(title, figure, 'Every trial/phase/seed/sample count is a separate curve; gaps do not invent validation measurements. All points retained in chart-data.json.gz.')
        def parameter_draw(ax):
            records = {}
            for r in catalog + curves:
                value = r.get('parameters', r.get('parameter_count'))
                if finite(value): records[f"{_family(r)} / {r.get('track','?')} / p={value}"] = value
            _bars(ax, sorted(records.items()), 'model/trial', 'stored model parameters')
        single('Model parameter counts', parameter_draw, 'Identical parameter counts collapse only for display by family/track; every model ID is retained in chart data. Parameter count is not FLOPs or runtime.')
        def selections(ax):
            counts = Counter(f"{_family(r)}/{r.get('selected_state', r.get('selection', r.get('outcome', r.get('status', 'NA'))))}" for r in catalog)
            _bars(ax, sorted(counts.items()), 'family / selection outcome', 'trials')
        single('Baseline adequacy and checkpoint selections', selections,
               'Initialization selection is not evidence of learned improvement. A weak trained control does not establish superiority over a published method.')
        figure = _facets(plt, curves, ('track',), lambda ax, vals: _scatter(ax, vals, 'train_count', 'validation_rms'), 'Data efficiency validation observations')
        save('Data efficiency validation observations', figure, 'All observed updates shown; this is descriptive, not an independent sample count. Training and confirmation remain disjoint.')
        for norm in ('rms', 'max'):
            figure = _facets(plt, endpoints, ('track', 'horizon', 'grid'), lambda ax, vals, norm=norm: _scatter(ax, vals, 'median_seconds', 'upper_'+norm), f'Work precision {norm} error with reference uncertainty')
            save(f'Work precision {norm} error with reference uncertainty', figure,
                 'All endpoint schedules retained; each panel has one spatial target and final time. Reference-informed frontier selection is post hoc, not a deployable policy.')
        def comparison_draw(ax, vals):
            groups = defaultdict(list); ineligible = Counter()
            for r in vals:
                value = r.get('control_over_candidate_speed_ratio')
                if r.get('eligibility') == 'ELIGIBLE' and finite(value) and value > 0:
                    groups[str(r.get('candidate_family', r.get('family', r.get('model_id', '?')))) + ' vs ' + str(r.get('control_family', r.get('comparator', '?')))].append(value)
                else: ineligible[str(r.get('eligibility', 'NA'))] += 1
            for i, (name, values) in enumerate(sorted(groups.items())):
                ax.scatter([i]*len(values), values, s=9, alpha=.4, label=name, rasterized=True)
            if groups:
                ax.set_xticks(range(len(groups)), sorted(groups), rotation=60, ha='right', fontsize=6); ax.set_yscale('log'); ax.axhline(1, color='black', ls='--', lw=.7)
            else: _blank(ax)
            ax.text(.01, .99, 'Ineligible retained: ' + str(dict(ineligible)), transform=ax.transAxes, va='top', fontsize=6)
            _label(ax, 'paired candidate / control', 'control / candidate time (>1 favors candidate)')
        figure = _facets(plt, comparisons, ('track', 'final_time', 'rms_target', 'max_target'), comparison_draw, 'Matched target paired speed distributions')
        save('Matched target paired speed distributions', figure,
             'Different final times and tolerances are never pooled. Repeated schedules/grids/seeds are paired evidence; use field-clustered inference in comparisons.json.')
        def stress(ax, vals):
            expanded = []
            for r in vals:
                targets = r.get('target_results', [])
                if isinstance(targets, dict): targets = [dict(v, target=k) if isinstance(v, dict) else {'target': k, 'passed': v} for k,v in targets.items()]
                for target in targets:
                    passed = target.get('passed', target.get('feasible', target.get('joint_pass')))
                    expanded.append((_family(r), f"{r.get('regime', '?')} / {target.get('rms_target', target.get('target', '?'))}", passed))
            groups = defaultdict(list)
            for family,label,value in expanded: groups[(family,label)].append(value)
            if not groups: _blank(ax, 'NA — target-specific feasibility records unavailable'); return
            families=sorted({k[0] for k in groups}); labels=sorted({k[1] for k in groups})
            values=np.full((len(families),len(labels)),np.nan)
            for i,family in enumerate(families):
                for j,label in enumerate(labels):
                    measured=groups.get((family,label),[])
                    if measured: values[i,j]=sum(v is True for v in measured)/len(measured)
            palette=plt.get_cmap('viridis').copy(); palette.set_bad('#9ba6b5')
            ax.imshow(values,aspect='auto',cmap=palette,vmin=0,vmax=1,interpolation='nearest')
            ax.set_yticks(range(len(families)),families,fontsize=6)
            ax.set_xticks(range(len(labels)),labels,rotation=70,ha='right',fontsize=5)
            _label(ax,'regime / target; purple=0, yellow=1, gray=no cell','family (observed joint pass fraction)')
        figure = _facets(plt, endpoints, ('track', 'horizon', 'schedule_kind'), stress, 'Stress regime and tolerance coverage')
        save('Stress regime and tolerance coverage', figure, 'Normal, best and worst cases use frozen field regimes, not retrospective exclusion. Cells with no target evidence remain NA.')
        for title, y in (('Mean error across spatial resolution', 'mean_error'), ('Centered error across spatial resolution', 'centered_rms'), ('High frequency error across spatial resolution', 'spectral_high_rms')):
            grid_rows = [{**r, 'grid_size': (r['grid'][0] if isinstance(r.get('grid'), list) else r.get('grid'))} for r in endpoints]
            figure = _facets(plt, grid_rows, ('track', 'horizon'), lambda ax, vals, y=y: _scatter(ax, vals, 'grid_size', y), title)
            save(title, figure, 'Each point retains field, grid, schedule and seed. Spatial and temporal targets are separate; zeros/nonfinite values are counted as unavailable on log axes.')
        for title, y in (('Scaling latency and throughput', 'median_seconds'), ('Scaling peak GPU memory', 'peak_reserved_bytes'), ('Scaling accepted reference accuracy', 'upper_rms')):
            normalized = []
            for row in scaling:
                r = {**row, **(row.get('metrics', {}) if isinstance(row.get('metrics'), dict) else {})}
                grid = r.get('grid'); r['cells'] = math.prod(grid) if isinstance(grid, list) else grid**2 if finite(grid) else None
                cost = r.get('cost', {})
                if isinstance(cost, dict):
                    for key in ('median_seconds', 'peak_reserved_bytes'): r.setdefault(key, cost.get(key))
                r.setdefault('median_seconds', r.get('cost_seconds'))
                r.setdefault('horizon', r.get('final_time'))
                errors = r.get('errors', [])
                r['upper_rms'] = max((e['upper_rms'] for e in errors), default=None) if (
                    errors and all(e.get('reference_accepted') and finite(e.get('upper_rms')) for e in errors)) else None
                normalized.append(r)
            figure = _facets(plt, normalized, ('track', 'horizon', 'batch_size'), lambda ax, vals, y=y: _scatter(ax, vals, 'cells', y), title)
            save(title, figure, 'Accuracy is the worst accepted-reference field in each distinct batch; missing references are NA. GPU memory is a shared paired-workload peak, not attributable to a family or shared host RAM.')
        def policy_cost(ax, subset):
            components = ('proposal_seconds', 'estimator_seconds', 'fallback_seconds', 'attributed_seconds', 'cost_seconds')
            values = []
            for component in components:
                samples = [r[component] for r in subset if finite(r.get(component))]
                if samples: values.append((component, statistics.mean(samples)))
            _bars(ax, values, 'recorded component', 'mean seconds')
        figure = _facets(plt, policy, ('track','final_time','mode'), policy_cost, 'Deployment complete cost components')
        save('Deployment complete cost components', figure,
               'First three components sum to attributed_seconds for a representative run; cost_seconds is independent complete warmed median. Do not sum these totals together. Missing or blocked deployment remains NA.')
        def decisions(ax, subset):
            counts = Counter(('ACCEPT' if r.get('accepted') is True else 'FALLBACK' if r.get('fallback') is True else 'NA') + (' / false acceptance' if r.get('false_accept') is True else '') for r in subset)
            _bars(ax, sorted(counts.items()), 'decision / measured false acceptance', 'endpoints')
        figure = _facets(plt, policy, ('track','final_time','mode'), decisions, 'Policy coverage and false acceptance')
        save('Policy coverage and false acceptance', figure,
               'Decisions use calibration and deployable features only. Confirmation truth audits decisions afterward; finite empirical coverage is not a distribution-free guarantee.')
        def uncertainty(ax):
            if not endpoints: _blank(ax); return
            bytrack = defaultdict(list)
            for r in endpoints:
                if finite(r.get('upper_rms')) and finite(r.get('error_rms')): bytrack[str(r.get('track'))].append(max(0.,r['upper_rms']-r['error_rms']))
            _bars(ax, [(k, statistics.median(v)) for k,v in sorted(bytrack.items())], 'spatial target', 'median RMS reference uncertainty')
        single('Reference uncertainty and unresolved evidence', uncertainty,
               'A tighter solver error beneath reference uncertainty is unresolved. Spatial and time refinement are not interchangeable.')
        figure = _facets(plt, endpoints, ('track', 'horizon'),
            lambda ax, vals: _scatter(ax, vals, 'median_seconds', 'cold_seconds'), 'First invocation versus randomized warmed timing')
        save('First invocation versus randomized warmed timing', figure,
             'First observed invocation is not guaranteed process-cold startup. Speed comparisons use randomized warmed measurements; first-call savings do not establish amortization.')
        summaries = data.get('comparison_document', {}).get('paired_field_summaries', [])
        for metric in ('error_ratio', 'speed_ratio'):
            def intervals(ax, vals, metric=metric):
                usable = [r for r in vals if finite(r.get(metric))]
                if not usable: _blank(ax); return
                for i,r in enumerate(usable):
                    ax.plot(r[metric], i, 'o', color='#407ba1')
                    ci = r.get(metric+'_ci95')
                    if isinstance(ci,list) and len(ci)==2 and all(finite(v) and v>0 for v in ci):
                        ax.plot(ci, [i,i], color='#407ba1')
                labels = [f"{r.get('competitor')} / n={r.get('train_count')} / fields={r.get('independent_fields')}" for r in usable]
                ax.set_yticks(range(len(usable)), labels, fontsize=6); ax.set_xscale('log'); ax.axvline(1,color='black',ls='--',lw=.7)
                _label(ax, metric+' competitor / rank one', 'paired comparison')
            figure = _facets(plt, summaries, ('track','final_time','grid'), intervals, 'Independent field intervals '+metric)
            save('Independent field intervals '+metric, figure,
                 'Descriptive field-cluster bootstrap; repeated seeds and physics do not increase independent field count. Missing intervals remain absent, never zero width.')
        def gate_draw(ax):
            values = data.get('gate_summary', [])
            _bars(ax, [(r['id'],r['score_1_100']) for r in values], 'research gate',
                  'required evidence attained (1–100)', colors=[COLORS[r['verdict']] for r in values] if values else None)
            ax.set_ylim(0,105)
        single('Five gate evidence and missing coverage', gate_draw,
               'Teal GOOD, coral BAD, gray NA. Scores are predeclared check-weight attainment, never scientific quality, effect size, or a proof.')
        summaries = data.get('policy_summaries', [])
        def risk_draw(ax, subset):
            pairs=[]
            for r in subset:
                risk=r.get('risk',{})
                for key in ('conditional_function_risk','one_sided_binomial_upper','target_risk'):
                    if finite(risk.get(key)): pairs.append((key,risk[key]))
            _bars(ax,pairs,'risk statistic','probability'); ax.set_ylim(0,1)
        figure = _facets(plt,summaries,('track','mode'),risk_draw,'Independent function risk and available evidence')
        save('Independent function risk and available evidence',figure,
             'Binomial upper bound is conditional on iid comparable fields and a frozen policy. No accepted functions means NA; repeated query rows are not independent trials.')
        def amortization_draw(ax, subset):
            values=[]; unavailable=[]
            for r in subset:
                a=r.get('amortization',{}); value=a.get('break_even_queries')
                if finite(value): values.append((str(r.get('mode')),value))
                else: unavailable.append(str(a.get('status','NA')))
            _bars(ax,values,'policy mode','break-even queries')
            ax.text(.01,.99,'Unestablished: '+', '.join(unavailable),transform=ax.transAxes,va='top',fontsize=7)
        figure = _facets(plt,summaries,('track',),amortization_draw,'Supported amortization including offline costs')
        save('Supported amortization including offline costs',figure,
             'Requires matched accuracy, accepted learned queries and positive saving uncertainty bound. All-classical routing or startup timing differences cannot earn neural amortization.')
        def software_tests(ax):
            values=data.get('software_tests',[])
            if not values: _blank(ax,'NA — no native per-stage JUnit artifacts were supplied'); return
            groups=sorted({r['stage'] for r in values}); bottom=np.zeros(len(groups))
            for status,color in (('PASSED',COLORS['GOOD']),('FAILED',COLORS['BAD']),('SKIPPED',COLORS['NA'])):
                counts=[sum(r['stage']==stage and r['status']==status for r in values) for stage in groups]
                ax.bar(groups,counts,bottom=bottom,label=status,color=color); bottom+=counts
            ax.legend();_label(ax,'native allocation stage','software test executions')
        single('Every native software test execution',software_tests,
               'Repeated cases in different allocations remain separate executions. Skipped tests never count as passed GPU readiness; software correctness is not scientific superiority.')
    manifest = {'schema': 'tdn.frontier-figures/v1', 'profile': data['profile'], 'panels': panels,
        'pdf': 'frontier-atlas.pdf', 'chart_data': 'chart-data.json.gz', 'check_index': 'check-cell-index.json.gz',
        'experiment_count': len(rows), 'check_count': len(check_index), 'loss_observations': len(curves),
        'confirmation_endpoints': len(endpoints), 'comparisons': len(comparisons), 'downsampled': False,
        'software_test_executions':len(data.get('software_tests',[])),
        'unavailable_panels_are_na': True, 'sources': data['sources'], 'stage_status': data['stage_status']}
    # A compact overview is an entry point, not a replacement for complete pages.
    fig, axes = plt.subplots(2,3,figsize=(18,11))
    for ax, panel in zip(axes.flat, [panels[i] for i in (0,1,5,11,13,21)]):
        ax.imshow(plt.imread(output / panel['path'])); ax.axis('off'); ax.set_title(panel['title'],fontsize=9)
    fig.suptitle(f"TDN frontier / {data['profile']} / {len(rows)} experiments / {len(check_index)} checks",fontsize=16)
    fig.tight_layout(); fig.savefig(output / 'frontier-overview.png',dpi=200); plt.close(fig)
    _write(output / 'manifest.json', manifest)
    body = ''.join(f'<section id="panel-{p["id"]}"><h2>{html.escape(p["title"])}</h2><p>{html.escape(p["note"])}</p><a href="{p["path"]}"><img loading="lazy" src="{p["path"]}" alt="{html.escape(p["title"])}"></a></section>' for p in panels)
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>TDN frontier analytical atlas</title><style>body{font:16px system-ui;margin:2em;max-width:1600px;background:#f4f6f9;color:#172437}img{max-width:100%;height:auto}section{background:white;padding:1em;margin:2em 0}a{color:#175c8f}</style><h1>TDN five-gate analytical atlas</h1><p>Profile: '+html.escape(data['profile'])+'. Computational completion does not establish scientific superiority. All verified observations retained; unavailable evidence is NA.</p><p><a href="frontier-atlas.pdf">Complete PDF</a> · <a href="chart-data.json.gz">Every source observation</a> · <a href="check-cell-index.json.gz">Every check lookup</a> · <a href="manifest.json">Provenance and coverage</a></p>'+body)
    return manifest


def run(ctx):
    from .core import check
    data = collect(ctx)
    gates = aggregate(data)
    data['gate_summary'] = gates
    _write(ctx.path / 'gate_summary.json', {'schema': 'tdn.frontier-gate-summary/v1', 'records': gates,
        'stage_status': data['stage_status'], 'scope': 'Required evidence attainment; not model quality or proof'})
    figures = build_figures(data, ctx.path / 'figures', ctx.budget)
    analysis = {'schema': 'tdn.frontier-analysis/v1', 'stage_status': data['stage_status'],
        'verified_experiments': len(data['rows']), 'gate_verdicts': {r['id']: r['verdict'] for r in gates},
        'figures': figures, 'scientific_claim': 'Finite declared comparisons only; no automatic paper-level or population guarantee',
        'all_stage_evidence_verified': all(v['status'] == 'VERIFIED' for v in data['stage_status'].values())}
    _write(ctx.path / 'analysis.json', analysis)
    with (ctx.path / 'gate_summary.csv').open('w', newline='') as handle:
        fields = ('id','name','verdict','score_1_100','evidence_coverage','experiment_count','missing_stages')
        writer = csv.DictWriter(handle, fields); writer.writeheader()
        for r in gates: writer.writerow({k: r[k] for k in fields})
    ctx.record('report/complete-source-and-figure-inventory', ['G1'], metrics={
        'verified_experiments': len(data['rows']), 'checks': figures['check_count'], 'panels': len(figures['panels'])},
        config={'scope': 'report inventory only; scientific conclusions are gate_summary.json'},
        checks=[check('every-prerequisite-verified', analysis['all_stage_evidence_verified'], True, 'eq', category='gap'),
                check('all-five-gates-visible', len(gates), 5, 'eq', category='correctness'),
                check('analytical-panel-minimum', len(figures['panels']), 12, 'ge', category='correctness'),
                check('report-is-not-mathematical-proof', None, None, category='math')])
    print(f"TDN frontier report: {len(data['rows'])} verified experiments; {figures['check_count']} checks; {len(figures['panels'])} analytical panels. Gates: {analysis['gate_verdicts']}", flush=True)
    return {'verified_experiments': len(data['rows']), 'gate_verdicts': analysis['gate_verdicts'],
            'all_stage_evidence_verified': analysis['all_stage_evidence_verified'], 'panels': len(figures['panels'])}
