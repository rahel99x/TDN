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
import textwrap
import xml.etree.ElementTree as ET

STAGES = ('audit', 'screen', 'prepare', 'train', 'confirm_prepare', 'confirm', 'scaling', 'policy')
VERDICTS = ('GOOD', 'BAD', 'NA')
COLORS = {'GOOD': '#218c83', 'BAD': '#d56352', 'NA': '#9ba6b5'}


# Display labels and reading directions never modify scientific family IDs.
METHOD_LABELS = {
    "rank1": "Ours: learned rank-one",
    "rank1_frozen": "Ours: frozen rank-one control",
    "rank1_postcompression": "Ours: input-compression ablation",
    "analytic_quad": "Analytic control: quadratic defect",
    "analytic_quad_cubic": "Analytic control: quadratic + cubic defect",
    "fno_small": "Theirs: small hybrid FNO (local)",
    "fno_standard": "Theirs: standard hybrid FNO (local)",
    "direct_fno": "Theirs: direct FNO (local)",
    "df": "Theirs: diffusion-first Strang",
    "etdrk4": "Theirs: ETDRK4",
    "strongest_classical": "Theirs: strongest classical",
}

METHOD_CONVENTION = (
    "Ours = proposed rank-one family and its controls. Theirs = established-method "
    "comparators implemented here; local FNO variants are not published-paper "
    "reproductions. Analytic control = untrained attribution reference, not an "
    "external FNO result or an ownership/novelty claim."
)
MARKER_CONVENTION = (
    "Blue solid line / up triangle = Ours, learned rank-one; light-blue dotted / down triangle = frozen; "
    "cyan dashed / left triangle = ablation. Other methods use thinner patterned lines / circles. "
    "Method colors are consistent; ratios use candidate styling. Hatched bars are controls. "
    "Heatmap colors show accuracy; partial-diagnostic colors show evidence status."
)
METHOD_STYLES = {
    'rank1': {'color': '#0072B2', 'marker': '^', 'linestyle': '-', 'linewidth': 1.8},
    'rank1_frozen': {'color': '#56B4E9', 'marker': 'v', 'linestyle': ':', 'linewidth': 1.1},
    'rank1_postcompression': {'color': '#17BECF', 'marker': '<', 'linestyle': '--', 'linewidth': 1.1},
    'fno_small': {'color': '#E69F00', 'marker': 'o', 'linestyle': '--', 'linewidth': .9},
    'fno_standard': {'color': '#D55E00', 'marker': 'o', 'linestyle': '-.', 'linewidth': .9},
    'direct_fno': {'color': '#CC79A7', 'marker': 'o', 'linestyle': ':', 'linewidth': .9},
    'analytic_quad': {'color': '#009E73', 'marker': 'o', 'linestyle': '--', 'linewidth': .9},
    'analytic_quad_cubic': {'color': '#8C6D31', 'marker': 'o', 'linestyle': '-.', 'linewidth': .9},
    'df': {'color': '#777777', 'marker': 'o', 'linestyle': ':', 'linewidth': .9},
    'etdrk4': {'color': '#222222', 'marker': 'o', 'linestyle': '--', 'linewidth': .9},
    'strongest_classical': {'color': '#222222', 'marker': 'o', 'linestyle': '--', 'linewidth': .9},
}


def method_label(family: str) -> str:
    """Return a readable provenance label without modifying scientific IDs."""
    name = str(family)
    if name in METHOD_LABELS:
        return METHOD_LABELS[name]
    # Endpoint frontier candidates carry a full identity instead of a family.
    # Retain track/data/seed rather than merging distinct measured candidates.
    parts = name.split('/')
    if (len(parts) == 4 and parts[0] in ('discrete', 'continuum')
            and parts[1] in METHOD_LABELS and parts[2].startswith('n')
            and parts[2][1:].isdigit() and parts[3].startswith('seed')
            and parts[3][4:].isdigit()):
        return f'{METHOD_LABELS[parts[1]]} / {parts[0]} / {parts[2]} / {parts[3]}'
    return name


def _method_id(row):
    """Resolve raw family IDs, including final and tuning model identities."""
    name = str(row.get('family', row.get('candidate_family', row.get('model_id', 'unlabeled'))))
    parts = name.split('/')
    if len(parts) >= 2 and parts[0] in ('discrete', 'continuum') and parts[1] in METHOD_STYLES:
        return parts[1]
    return name


def _method_style(row):
    return METHOD_STYLES.get(_method_id(row),
        {'color': '#777777', 'marker': 'o', 'linestyle': '--', 'linewidth': .9})


def _point_style(row):
    style = _method_style(row)
    ours = _method_id(row) == 'rank1'
    return {'color': style['color'], 'marker': style['marker'],
            's': 22 if ours else 18 if style['marker'] in ('v', '<') else 9,
            'alpha': .8 if ours else .4, 'zorder': 4 if ours else 2}


PANEL_GUIDANCE = {
    "Stage compute and frozen budgets": (
        "Y: lower measured/allocation seconds means less time for the same completed work. "
        "A budget is a ceiling, not consumed time; a fast failed stage is not a win. "
        "X lists stages, not a quality ranking."
    ),
    "Physical confirmation coverage": (
        "Y: more sealed endpoints means more of the planned coverage, not greater accuracy. "
        "Full planned coverage is the goal; teal is verified, gray is unavailable. "
        "Repeated endpoints are not independent fields."
    ),
    "Unmerged partial confirmation diagnostics": (
        "Y: lower RMS error is more accurate for matched cases. X groups physical parts. "
        "Blue is sealed partial evidence; gray is unsealed forensic evidence. "
        "Neither establishes a complete comparison; an empty panel is NA, not zero error."
    ),
    "Complete experiment verdicts": (
        "Teal GOOD means declared checks attained; coral BAD means failed requirements; "
        "gray NA means unresolved or unavailable. Y counts records, so taller bars do not "
        "mean better methods. Compare verdicts within the declared experiment scope."
    ),
    "Correctness math gap and utility checks": (
        "Teal GOOD is favorable check evidence; coral BAD is unfavorable; gray NA is unresolved. "
        "Y counts checks, not effect size or independent replications. "
        "More checks or passing numerical examples do not prove a mathematical theorem."
    ),
    "Every individual check": (
        "Color is the result: teal GOOD, coral BAD, gray NA; white is padding. "
        "X/Y locate checks in row-major order and have no better/worse direction. "
        "NA is not a pass; the lookup file identifies every cell."
    ),
    "Training loss by update": (
        "Y: lower training loss means a closer fit to that training objective. "
        "X: farther right means more optimizer updates, not inherently better quality. "
        "Compare like objectives; training loss alone does not establish generalization."
    ),
    "Validation loss by update": (
        "Y: lower validation loss is better on the same held-out objective. "
        "X: fewer updates to the same loss is preferable; more updates alone is not a win. "
        "Validation selection does not substitute for fresh confirmation."
    ),
    "Validation error versus training time": (
        "Lower-left is favorable: X less training time, Y lower validation RMS error. "
        "Compare the same target and validation cohort; a longer run may reach lower error. "
        "These are training costs, not inference latency."
    ),
    "Validation error versus examples seen": (
        "Lower-left is favorable: X fewer training-example presentations, Y lower validation RMS. "
        "Examples seen includes repeated exposure, not just independent fields. "
        "Compare like targets; this alone does not establish a sample-complexity law."
    ),
    "Model parameter counts": (
        "Y: lower counts mean a smaller stored model, not necessarily faster or more accurate. "
        "Capacity and accuracy can trade off; parameter count is not FLOPs or measured runtime. "
        "X lists methods/trials and has no favorable direction."
    ),
    "Baseline adequacy and checkpoint selections": (
        "Y counts selection outcomes, not method quality. A selected trained checkpoint can "
        "show improvement over its initialization, but does not establish convergence. "
        "An initialization selection is not evidence of successful learning."
    ),
    "Data efficiency validation observations": (
        "Lower-left is favorable: X fewer training fields, Y lower validation RMS error. "
        "Compare like targets and training budgets; updates from one run are repeated "
        "observations, not additional independent fields."
    ),
    "Work precision rms error with reference uncertainty": (
        "Lower-left is favorable: X less warmed runtime, Y lower RMS error plus reference uncertainty. "
        "Both axes are logarithmic; compare within the same track, grid and final time. "
        "Speed is useful only when the required accuracy is met."
    ),
    "Work precision max error with reference uncertainty": (
        "Lower-left is favorable: X less warmed runtime, Y lower maximum error plus reference uncertainty. "
        "Both axes are logarithmic; compare within the same track, grid and final time. "
        "Meeting RMS alone does not establish the maximum-error target."
    ),
    "Matched target paired speed distributions": (
        "Y = control time / candidate time: above 1 favors the candidate, below 1 favors "
        "the control, and 1 is a tie. Higher is better for the named candidate. "
        "The axis is logarithmic; ineligible cases are not speed wins."
    ),
    "Stress regime and tolerance coverage": (
        "Color: higher joint pass fraction is better; yellow = 1, purple = 0. "
        "Gray is missing evidence, not a favorable result. X/Y identify regimes/targets "
        "and methods; their positions are not quality rankings."
    ),
    "Mean error across spatial resolution": (
        "Y: lower absolute mean error is better. X: larger grid means finer spatial resolution, "
        "not automatically better efficiency. Compare methods at the same grid/target; "
        "both axes are logarithmic and exact zeros cannot appear."
    ),
    "Centered error across spatial resolution": (
        "Y: lower centered RMS error is better for spatial variation after removing the mean. "
        "X: larger grid means finer resolution, not automatically better efficiency. "
        "Both axes are logarithmic; this panel does not measure mean accuracy."
    ),
    "High frequency error across spatial resolution": (
        "Y: lower high-frequency RMS error is better on the declared spectral band. "
        "X: larger grid means finer resolution, not a quality score. "
        "Compare matched grids/targets; both axes are logarithmic."
    ),
    "Scaling latency and throughput": (
        "Y: lower seconds is faster; at fixed batch size this means higher throughput. "
        "X: more cells means a larger problem, not a worse result. Both axes are logarithmic. "
        "Compare matched accuracy, grid, batch size, target and final time."
    ),
    "Scaling peak GPU memory": (
        "Y: lower reserved bytes is less GPU memory for the same workload. "
        "This is a shared paired-workload peak, not an isolated method footprint. "
        "X is problem size; both axes are logarithmic. Host RAM is not GPU VRAM."
    ),
    "Scaling accepted reference accuracy": (
        "Y: lower worst-field RMS plus accepted-reference uncertainty is better. "
        "X: more cells means a larger problem; compare the same workload. "
        "Both axes are logarithmic. Missing accepted references remain NA, not accurate."
    ),
    "Deployment complete cost components": (
        "Y: lower complete runtime is better at matched accuracy. X lists different cost scopes. "
        "Proposal/estimator/fallback components sum to attributed time; do not add that total "
        "to complete-call time. Blocked deployment is NA, not free computation."
    ),
    "Policy coverage and false acceptance": (
        "Fewer false acceptances is better. More accepted queries is useful only with "
        "verified accuracy and cost savings; fallback can be the correct decision. "
        "Y counts decisions, not independent fields or a risk guarantee."
    ),
    "Reference uncertainty and unresolved evidence": (
        "Y: lower uncertainty permits finer accuracy distinctions, but does not prove the "
        "reference is exact. X separates spatial targets, not a ranking. "
        "Differences beneath uncertainty are unresolved; missing evidence is not zero."
    ),
    "First invocation versus randomized warmed timing": (
        "Lower-left means lower time in both scopes: X warmed median, Y first invocation. "
        "Both axes are logarithmic; compare identical workloads and accuracy. "
        "First invocation is not guaranteed process-cold startup or an amortization result."
    ),
    "Independent field intervals error_ratio": (
        "X = competitor RMS / Ours learned rank-one RMS: right of 1 favors Ours; left "
        "of 1 favors the named competitor. Higher is better for Ours; 1 is a tie. "
        "Logarithmic X; an interval crossing 1 leaves the error direction unresolved."
    ),
    "Independent field intervals speed_ratio": (
        "X = competitor time / Ours learned rank-one time: right of 1 favors Ours; left "
        "of 1 favors the named competitor. Higher is faster for Ours; 1 is a tie. "
        "Logarithmic X; a speed ratio alone does not establish matched-accuracy utility."
    ),
    "Five gate evidence and missing coverage": (
        "Y: higher scores mean more declared evidence attained, not better model quality. "
        "Teal GOOD, coral BAD and gray NA retain the gate verdict; a high NA score is "
        "not success. X lists research gates, not a performance ranking."
    ),
    "Independent function risk and available evidence": (
        "Y: lower empirical risk and a lower valid upper bound are better. The target is "
        "a requirement, not an observed result; compare the bound with that target. "
        "NA or no accepted fields means unestablished risk, not zero risk."
    ),
    "Supported amortization including offline costs": (
        "Y: fewer break-even queries is better when matched accuracy and positive savings "
        "are established. X lists policy modes. NA means amortization was not established; "
        "it must not be read as zero required queries."
    ),
    "Every native software test execution": (
        "Teal PASSED is favorable readiness evidence; coral FAILED needs attention; gray "
        "SKIPPED provides no pass evidence. Y counts executions, not model quality. "
        "Repeated passes across allocations do not add independent scientific evidence."
    ),
}


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


def _verified_recovery(ctx):
    """Engine validation is repeated before mixed-source evidence is credited."""
    if not getattr(ctx, 'recovery', None):
        return {}
    from .engine import recovery_context_for_report
    bridge = recovery_context_for_report(ctx)
    return {**bridge, 'verified_stage_sources': dict(getattr(ctx, 'recovery_stage_sources', {}))}


def _expected_source(stage, path, current_source, recovery):
    inherited = recovery.get('stage_paths', {})
    if stage not in inherited:
        return current_source
    if Path(inherited[stage]).resolve() != Path(path).resolve():
        raise ValueError('Inherited stage differs from the verified recovery bridge')
    return recovery.get('verified_stage_sources', {}).get(stage, recovery['origin_source_tree_sha256'])


def _physical_stages(ctx, root):
    from .partition import plan_shards
    parts = [part['shard_id'] for part in plan_shards(ctx.protocol)]
    # A historical unpartitioned workflow must not acquire imaginary allocations.
    manifest = root / 'frontier-workflow.json'
    workflow = _read(manifest) if manifest.is_file() else {}
    inherited = (getattr(ctx, 'recovery', None) or {}).get('stage_paths', {})
    inherited_here = root.resolve() == ctx.path.parent.resolve() and any(name in inherited for name in parts)
    partitioned = workflow.get('execution_version') == 2 or inherited_here or any((root / name).is_dir() for name in parts)
    return (*STAGES[:5], *(parts if partitioned else ()), *STAGES[5:], 'report')


def _verify_part_wrapper(path, protocol, source, partition):
    """An available native wrapper must bind the exact physical part."""
    from tdn.research.protocol import digest
    execution_path = path / 'execution.json'
    if not execution_path.is_file():
        if protocol['profile'] == 'full':
            raise ValueError('Native full confirmation part has no execution wrapper')
        return
    execution = _read(execution_path); stage = _read(path / 'stage.json')
    seal = _read(path / 'workflow-seal.json')
    names = ('execution.json', 'protocol.json', 'stage.json', 'science_manifest.json')
    if (seal.get('schema') != 'tdn.frontier/v1' or seal.get('schema_version') != 1
            or seal.get('protocol_sha256') != digest(protocol) or set(seal.get('files', {})) != set(names)):
        raise ValueError('Confirmation part wrapper seal differs')
    for name in names:
        if seal['files'][name] != hashlib.sha256((path / name).read_bytes()).hexdigest():
            raise ValueError('Confirmation part wrapper was modified')
    expected = {'stage': 'confirm', 'profile': protocol['profile'], 'protocol_sha256': digest(protocol)}
    if (any(execution.get(key) != value or stage.get(key) != value for key, value in expected.items())
            or stage.get('status') != 'COMPLETED'
            or execution.get('confirmation_partition') != partition
            or execution.get('physical_stage') != partition['shard_id']
            or execution.get('software', {}).get('source_tree_sha256') != source):
        raise ValueError('Confirmation part wrapper identity, partition or source differs')


def _collect_parts(ctx, output, verifier, current_source, recovery=None):
    """Parts are visible diagnostics, never a substitute for merged gate evidence."""
    from .partition import plan_shards
    from .core import validate_row
    parts = plan_shards(ctx.protocol)
    recovery = recovery or {}
    if not any(name.startswith('confirm-part-') for name in _physical_stages(ctx, ctx.path.parent)):
        return
    merged = output['stage_status'].get('confirm', {}).get('status') == 'VERIFIED'
    output['confirmation_parts'] = []
    output['partial_confirmation'] = []
    output['partial_rows'] = []
    output['unsealed_confirmation'] = []
    for part in parts:
        name = part['shard_id']; path = Path(recovery.get('stage_paths', {}).get(name, ctx.path.parent / name))
        record = {'physical_stage': name, 'logical_stage': 'confirm', 'parent_ids': part['parent_ids'],
                  'status': 'MISSING_OR_INVALID', 'scientific_credit': False, 'rows': 0,
                  'scope': 'Execution partition; complete confirmation requires the verified aggregate'}
        try:
            manifest = verifier(ctx.protocol, path, confirmation_partition=part)
            scope = _read(path / 'confirmation_scope.json')
            if (manifest.get('stage') != 'confirm' or manifest.get('source_tree_sha256') != current_source
                    or scope.get('mode') != 'partition' or scope.get('partition') != part
                    or scope.get('status') != 'COMPLETED'):
                raise ValueError('Confirmation part identity, source or completion differs')
            expected_prior = {stage: output['stage_status'][stage]['manifest_sha256'] for stage in STAGES[:5]}
            if manifest.get('prerequisites') != expected_prior:
                raise ValueError('Confirmation part lineage differs from the verified prerequisites')
            _verify_part_wrapper(path, ctx.protocol, current_source, part)
            rows = [validate_row(row) for row in _read(path / 'rows.json')['rows']]
            values = _rows(_read(path / 'confirmation_rows.json'))
            if any(row.get('parent_id') not in part['parent_ids'] for row in values):
                raise ValueError('Confirmation endpoint falls outside its execution partition')
            record.update(status='VERIFIED_PARTITION', rows=len(rows), endpoints=len(values),
                          manifest_sha256=hashlib.sha256((path / 'science_manifest.json').read_bytes()).hexdigest(),
                          represented_by_aggregate=merged)
            if not merged:
                output['partial_rows'].extend({'physical_stage': name, 'row': row,
                    'scientific_credit': False} for row in rows)
                output['partial_confirmation'].extend({**row, 'physical_stage': name,
                    'evidence_scope': 'PARTIAL_COVERAGE_ONLY'} for row in values)
            output['sources'].append({'stage': name, 'path': str(path / 'confirmation_rows.json'),
                'sha256': hashlib.sha256((path / 'confirmation_rows.json').read_bytes()).hexdigest(),
                'rows': len(values), 'scope': 'Partition source; canonical merged confirmation is counted once'})
        except (OSError, KeyError, ValueError, TypeError) as error:
            record['reason'] = str(error)
            # Atomic complete groups survive a timeout. They are forensic
            # diagnostics only and are never fed to gate scoring or a resume.
            checkpoints = [] if path.is_symlink() or (path / 'complete-groups').is_symlink() else sorted((path / 'complete-groups').glob('*.json'))
            for checkpoint in checkpoints:
                try:
                    if checkpoint.is_symlink() or checkpoint.stat().st_size > 64 << 20:
                        raise ValueError('Forensic checkpoint must be a bounded regular file')
                    group = _read(checkpoint)
                    if group.get('schema') != 'tdn.frontier-confirmation-group/v1':
                        raise ValueError('Unknown forensic group schema')
                    values = group.get('rows', [])
                    if not isinstance(values, list) or any(row.get('parent_id') not in part['parent_ids'] for row in values):
                        raise ValueError('Forensic group parent differs from partition')
                    output['unsealed_confirmation'].extend({**row, 'physical_stage': name,
                        'evidence_scope': 'UNSEALED_FORENSIC_ONLY', 'scientific_credit': False} for row in values)
                    output['sources'].append({'stage': name, 'path': str(checkpoint),
                        'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                        'rows': len(values), 'scope': 'Unsealed forensic checkpoint; no scientific credit or resumption'})
                except (OSError, ValueError, TypeError, AttributeError) as partial_error:
                    record.setdefault('forensic_errors', []).append(str(partial_error))
        output['confirmation_parts'].append(record)
    output['confirmation_partition_coverage'] = {
        'expected_parts': len(parts),
        'verified_parts': sum(row['status'] == 'VERIFIED_PARTITION' for row in output['confirmation_parts']),
        'aggregate_verified': merged,
        'partial_endpoints': len(output['partial_confirmation']),
        'scientific_outcome': 'MERGED_VERIFIED' if merged else 'NA_INCOMPLETE_CONFIRMATION'}


def _collect_execution_telemetry(ctx, output, recovery):
    """Retain every allocation/test occurrence, separating inherited sunk costs."""
    roots = [(ctx.path.parent, 'current_workflow')]
    if recovery:
        origin = Path(recovery['origin_run_dir'])
        if origin.resolve() == ctx.path.parent.resolve():
            raise ValueError('Recovery accounting requires a distinct origin coordinator')
        inherited_roots = [origin, *(Path(path).parent for path in recovery.get('stage_paths', {}).values())]
        seen_roots = {ctx.path.parent.resolve()}
        for inherited in inherited_roots:
            if inherited.resolve() not in seen_roots:
                roots.append((inherited, 'inherited_sunk')); seen_roots.add(inherited.resolve())
        output['recovery'] = {key: recovery[key] for key in
            ('origin_run_dir', 'origin_source_tree_sha256', 'stage_paths')}
    tests = []; test_sources = []; snapshots = []
    costs = {(row['source_run_dir'], row['stage']): row for row in output['costs']}
    for root, cost_scope in roots:
        stage_names = _physical_stages(ctx, root)
        accounting = root / 'state' / 'scheduler-accounting.json'
        try:
            snapshot = _read(accounting) if accounting.is_file() and not accounting.is_symlink() else {'records': [], 'status': 'UNAVAILABLE'}
        except (OSError, ValueError, TypeError) as error:
            snapshot = {'records': [], 'status': 'INVALID', 'reason': str(error)}
        snapshot = {**snapshot, 'source_run_dir': str(root), 'cost_scope': cost_scope}
        snapshots.append(snapshot)
        allocations = defaultdict(list)
        for row in snapshot.get('records', []):
            if row.get('record_kind') == 'allocation':
                allocations[row.get('workflow_stage')].append(row)
        for stage in stage_names:
            identity = (str(root), stage)
            summary_path = root / stage / 'summary.json'
            summary = {}
            if summary_path.is_file() and not summary_path.is_symlink() and not summary_path.parent.is_symlink():
                try:
                    summary = _read(summary_path)
                except (OSError, ValueError, TypeError):
                    summary = {'status': 'UNREADABLE'}
            records = allocations.get(stage, [])
            if identity not in costs and not summary and not records:
                continue
            cost = costs.setdefault(identity, {'stage': stage, 'logical_stage': 'confirm' if stage.startswith('confirm-part-') else stage,
                'source_run_dir': str(root), 'cost_scope': cost_scope,
                'numerical_seconds': summary.get('elapsed_seconds'),
                'numerical_status': summary.get('status', 'UNAVAILABLE'),
                'budget_seconds': ctx.protocol.get('budgets', {}).get('confirm' if stage.startswith('confirm-part-') else stage, {}).get('seconds'),
                'monetary_cost': None, 'allocation_seconds': None,
                'scope': 'Descriptive execution cost only; failed or inherited runs do not add scientific evidence'})
            terminal = records[0] if len(records) == 1 and records[0].get('terminal_state') is True else {}
            cost.update(allocation_seconds=terminal.get('elapsed_seconds'),
                allocated_cpu_seconds=terminal.get('allocated_cpu_seconds'),
                allocation_job_id=terminal.get('allocation_job_id'),
                accounting_collected_at=snapshot.get('collected_at'),
                accounting_status='DUPLICATE_ALLOCATION_INVALID' if len(records) > 1 else 'TERMINAL' if terminal else 'UNAVAILABLE_OR_RUNNING')
        for stage in stage_names:
            path = root / 'reporter-tests' / stage / 'tests.xml'
            if not path.is_file():
                continue
            try:
                if path.is_symlink() or path.stat().st_size > 16 << 20:
                    raise ValueError('Native JUnit must be a bounded regular file')
                raw = path.read_bytes()
                if b'<!DOCTYPE' in raw.upper():
                    raise ValueError('JUnit document declarations are refused')
                tree = ET.fromstring(raw)
                occurrences = []
                for index, case in enumerate(tree.iter('testcase')):
                    result = 'FAILED' if case.find('failure') is not None or case.find('error') is not None else 'SKIPPED' if case.find('skipped') is not None else 'PASSED'
                    seconds = case.get('time'); seconds = float(seconds) if seconds else None
                    occurrences.append({'stage': stage, 'source_run_dir': str(root), 'cost_scope': cost_scope,
                        'occurrence': index, 'name': case.get('name'), 'classname': case.get('classname'),
                        'status': result, 'seconds': seconds if finite(seconds) else None})
                tests.extend(occurrences)
                test_sources.append({'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
                    'stage': stage, 'cost_scope': cost_scope, 'status': 'READ_COMPLETE'})
            except (OSError, ValueError, ET.ParseError) as error:
                test_sources.append({'path': str(path), 'stage': stage, 'status': 'INVALID_OR_INCOMPLETE', 'reason': str(error)})
    output['costs'] = list(costs.values())
    output['scheduler_accounting'] = snapshots[0]
    output['scheduler_accounting_sources'] = snapshots
    output['software_tests'] = tests
    output['software_test_sources'] = test_sources
    _write(ctx.path / 'accounting-snapshot.json', {'sources': snapshots,
        'scope': 'Allocation rows only contribute totals. Origin sunk costs are separate from current allocations; no monetary tariff is assumed.'})
    _write(ctx.path / 'junit-snapshot.json', {'occurrences': tests, 'sources': test_sources,
        'scope': 'Every native reporter-test occurrence, including inherited runs and physical confirmation parts; never independent scientific observations'})
    for name in ('accounting-snapshot.json', 'junit-snapshot.json'):
        output.setdefault('sources', []).append({'path': str(ctx.path / name),
            'sha256': hashlib.sha256((ctx.path / name).read_bytes()).hexdigest(),
            'scope': 'Exact report-creation telemetry snapshot; inherited and current costs remain separate'})


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
    recovery = _verified_recovery(ctx)
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
            expected_source = _expected_source(stage, path, current_source, recovery)
            if manifest.get('stage') != stage or manifest.get('source_tree_sha256') != expected_source:
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
                if execution.get('software', {}).get('source_tree_sha256') != expected_source:
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
                'cost_scope': 'inherited_sunk' if stage in recovery.get('stage_paths', {}) else 'current_workflow',
                'source_run_dir': str(path.parent),
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
    _collect_parts(ctx, output, verifier, current_source, recovery)
    _collect_execution_telemetry(ctx, output, recovery)
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
    return method_label(row.get('family', row.get('candidate_family', row.get('model_id', 'unlabeled'))))


def _blank(ax, reason='NA — no verified measurements for this panel'):
    ax.text(.5, .5, reason, ha='center', va='center', transform=ax.transAxes, wrap=True, color='#58677b')
    ax.set_xticks([]); ax.set_yticks([])


def _label(ax, x, y):
    labels = {
        'update': 'Optimizer updates (training work)',
        'train_loss': 'Training objective (lower within a trial)',
        'validation_loss': 'Validation objective (lower within a trial)',
        'elapsed_seconds': 'Training seconds (left = less compute)',
        'examples_seen': 'Examples processed (left = less training work)',
        'train_count': 'Distinct training fields (left = less data)',
        'validation_rms': 'Validation RMS error (lower is better)',
        'median_seconds': 'Warm latency, seconds (lower is better)',
        'cold_seconds': 'First invocation, seconds (lower is better)',
        'upper_rms': 'RMS error + reference uncertainty (lower is better)',
        'upper_max': 'Maximum error + reference uncertainty (lower is better)',
        'mean_error': 'Mean error (lower is better)',
        'centered_rms': 'Centered RMS error (lower is better)',
        'spectral_high_rms': 'High-frequency RMS error (lower is better)',
        'grid_size': 'Grid width (larger = finer resolution)',
        'cells': 'Spatial cells per field (larger = bigger workload)',
        'peak_reserved_bytes': 'Shared workload GPU bytes (lower footprint)',
    }
    ax.set_xlabel(labels.get(x, x)); ax.set_ylabel(labels.get(y, y)); ax.grid(alpha=.16)


def _lines(ax, curves, x, y):
    groups = defaultdict(list)
    for row in curves:
        if finite(row.get(x)) and finite(row.get(y)):
            groups[_key(row, ('model_id', 'family', 'track', 'seed', 'train_count', 'phase'))].append(row)
    for key, rows in sorted(groups.items()):
        rows.sort(key=lambda r: r[x])
        family = _family(rows[0]); label = f"{family}/{rows[0].get('track', '?')}"
        ours = _method_id(rows[0]) == 'rank1'
        ax.plot([r[x] for r in rows], [r[y] for r in rows], **_method_style(rows[0]),
                alpha=.85 if ours else .5, zorder=4 if ours else 2,
                markersize=4 if ours else 2.5, markevery=max(1, len(rows) // 8),
                label=label if sum(1 for v in ax.lines if v.get_label() == label) == 0 else None)
    if not groups:
        _blank(ax)
    else:
        if all(r.get(y, 0) > 0 for values in groups.values() for r in values):
            ax.set_yscale('log')
        ax.legend(fontsize=6, ncol=2)
    _label(ax, x, y)


def _bars(ax, values, xlabel='', ylabel='', colors=None, method_rows=None):
    if not values:
        _blank(ax); return
    labels, heights = zip(*values)
    if method_rows is not None:
        colors = [_method_style(row)['color'] for row in method_rows]
    bars = ax.bar(range(len(labels)), heights, color=colors or '#407ba1')
    if method_rows is not None:
        for bar, row in zip(bars, method_rows):
            ours = _method_id(row) == 'rank1'
            bar.set_hatch('' if ours else '//')
            bar.set_edgecolor(_method_style(row)['color'] if ours else '#333333')
            bar.set_linewidth(1.3 if ours else .4)
    ax.set_xticks(range(len(labels)), labels, rotation=70, ha='right', fontsize=6)
    _label(ax, xlabel, ylabel)


def _facets(plt, rows, keys, draw, title, *, maximum_columns=3, row_height=4):
    groups = defaultdict(list)
    for row in rows:
        groups[_key(row, keys)].append(row)
    groups = groups or {tuple('NA' for _ in keys): []}
    columns = min(maximum_columns, len(groups)); count = math.ceil(len(groups) / columns)
    fig, axes = plt.subplots(count, columns, figsize=(6 * columns, row_height * count + 1), squeeze=False)
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
        ax.scatter([r[x] for r in values], [r[y] for r in values],
                   **_point_style(values[0]), label=name, rasterized=True)
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
            guidance = PANEL_GUIDANCE[title]
            # A fixed physical footer remains readable on both short and tall
            # faceted pages. Expand the canvas instead of covering data/labels.
            width, height = fig.get_size_inches()
            wrap_width = max(70, int(width * 14))
            footer = '\n'.join(textwrap.fill(prefix + value, width=wrap_width,
                break_long_words=False, break_on_hyphens=False) for prefix, value in (
                    ('HOW TO READ: ', guidance), ('METHODS: ', METHOD_CONVENTION),
                    ('MARKERS: ', MARKER_CONVENTION), ('SCOPE: ', note)))
            footer_height = (len(footer.splitlines()) * 11 + 24) / 72
            fig.set_size_inches(width, height + footer_height)
            fig.text(.025, .14 / (height + footer_height), footer,
                     ha='left', va='bottom', fontsize=8, linespacing=1.35,
                     color='#172437', family='DejaVu Sans')
            fig.tight_layout(rect=(.01, footer_height / (height + footer_height), .99, .95))
            fig.savefig(output / name, dpi=170)
            pdf.savefig(fig, dpi=170)
            panels.append({'id': ordinal, 'title': title, 'path': name, 'note': note,
                           'reading_guide': guidance,
                           'sha256': hashlib.sha256((output / name).read_bytes()).hexdigest()})
            plt.close(fig)
        def single(title, draw, note):
            fig, ax = plt.subplots(figsize=(12, 6)); ax.set_title(title); draw(ax); save(title, fig, note)

        def costs(ax):
            v = [r for r in data['costs'] if finite(r.get('numerical_seconds')) or finite(r.get('allocation_seconds'))]
            if not v: _blank(ax); return
            x = np.arange(len(v))
            numerical = [(i, r['numerical_seconds']) for i, r in enumerate(v) if finite(r.get('numerical_seconds'))]
            if numerical: ax.bar([i-.24 for i,_ in numerical], [value for _,value in numerical], .24, label='measured stage seconds')
            ax.bar(x, [r.get('budget_seconds') or 0 for r in v], .24, label='frozen compute ceiling')
            allocation = [(i,r['allocation_seconds']) for i,r in enumerate(v) if finite(r.get('allocation_seconds'))]
            if allocation: ax.bar([i+.24 for i,_ in allocation], [value for _,value in allocation], .24, label='terminal allocation seconds')
            ax.set_xticks(x, [('sunk/' if r.get('cost_scope') == 'inherited_sunk' else '') + r['stage'] for r in v], rotation=60, ha='right'); ax.legend(); _label(ax, 'stage', 'seconds')
        single('Stage compute and frozen budgets', costs,
               'Each physical allocation appears once; batch/step records are never added. Sunk/origin costs are separate from current work. Failed stage costs remain visible; running report allocation and tariffs remain NA.')
        def partition_coverage(ax):
            parts = data.get('confirmation_parts', [])
            if not parts: _blank(ax, 'NA: no partitioned confirmation allocations in this run'); return
            x = np.arange(len(parts))
            ax.bar(x, [row.get('endpoints', 0) for row in parts], color=[
                COLORS['GOOD'] if row['status'] == 'VERIFIED_PARTITION' else COLORS['NA'] for row in parts])
            ax.set_xticks(x, [row['physical_stage'] for row in parts], rotation=30)
            _label(ax, 'physical confirmation allocation', 'sealed endpoint observations')
            for i, row in enumerate(parts):
                if row['status'] != 'VERIFIED_PARTITION': ax.text(i, 0, 'NA', ha='center', va='bottom')
        single('Physical confirmation coverage', partition_coverage,
               'Partial partition observations are descriptive only. Missing or failed allocations remain NA; G3 requires the verified complete aggregate. Merged scientific rows count once.')
        def partial_errors(ax):
            values = data.get('partial_confirmation', []) + data.get('unsealed_confirmation', [])
            usable = [row for row in values if finite(row.get('error_rms'))]
            if not usable: _blank(ax, 'NA: no unmerged sealed partial endpoint errors'); return
            groups = sorted({row.get('physical_stage') for row in usable})
            for i, group in enumerate(groups):
                for scope, color in (('PARTIAL_COVERAGE_ONLY', '#407ba1'), ('UNSEALED_FORENSIC_ONLY', COLORS['NA'])):
                    by_method = defaultdict(list)
                    for row in usable:
                        if row.get('physical_stage') == group and row.get('evidence_scope') == scope:
                            by_method[_family(row)].append(row)
                    for vals in by_method.values():
                        ax.scatter([i] * len(vals), [row['error_rms'] for row in vals],
                                   **{**_point_style(vals[0]), 'color': color})
            ax.set_xticks(range(len(groups)), groups, rotation=30)
            _label(ax, 'physical part (all cases; descriptive only)', 'endpoint RMS error')
        single('Unmerged partial confirmation diagnostics', partial_errors,
               'Blue: sealed partial coverage. Gray: unsealed forensic groups (no scientific credit or resumption). All observations remain in chart-data.json.gz. This is not a model ranking, speedup estimate, or complete gate comparison.')
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
            records = {}; method_rows = {}
            for r in catalog + curves:
                value = r.get('parameters', r.get('parameter_count'))
                if finite(value):
                    label = f"{_family(r)} / {r.get('track','?')} / p={value}"
                    records[label] = value; method_rows[label] = r
            _bars(ax, sorted(records.items()), 'model/trial', 'stored model parameters',
                  method_rows=[method_rows[label] for label in sorted(records)])
        single('Model parameter counts', parameter_draw, 'Identical parameter counts collapse only for display by family/track; every model ID is retained in chart data. Parameter count is not FLOPs or runtime.')
        def selections(ax):
            counts = Counter(); method_rows = {}
            for r in catalog:
                label = f"{_family(r)}/{r.get('selected_state', r.get('selection', r.get('outcome', r.get('status', 'NA'))))}"
                counts[label] += 1; method_rows[label] = r
            _bars(ax, sorted(counts.items()), 'family / selection outcome', 'trials',
                  method_rows=[method_rows[label] for label in sorted(counts)])
        single('Baseline adequacy and checkpoint selections', selections,
               'Initialization selection is not evidence of learned improvement. A weak trained control does not establish superiority over a published method.')
        figure = _facets(plt, curves, ('track',), lambda ax, vals: _scatter(ax, vals, 'train_count', 'validation_rms'), 'Data efficiency validation observations')
        save('Data efficiency validation observations', figure, 'All observed updates shown; this is descriptive, not an independent sample count. Training and confirmation remain disjoint.')
        for norm in ('rms', 'max'):
            figure = _facets(plt, endpoints, ('track', 'horizon', 'grid'), lambda ax, vals, norm=norm: _scatter(ax, vals, 'median_seconds', 'upper_'+norm), f'Work precision {norm} error with reference uncertainty')
            save(f'Work precision {norm} error with reference uncertainty', figure,
                 'All endpoint schedules retained; each panel has one spatial target and final time. Reference-informed frontier selection is post hoc, not a deployable policy.')
        def comparison_draw(ax, vals):
            groups = defaultdict(list); styles = {}; ineligible = Counter()
            for r in vals:
                value = r.get('control_over_candidate_speed_ratio')
                if r.get('eligibility') == 'ELIGIBLE' and finite(value) and value > 0:
                    candidate = r.get('candidate_family', r.get('family', r.get('model_id', '?')))
                    name = method_label(candidate) + ' vs ' + method_label(r.get('control_family', r.get('comparator', '?')))
                    groups[name].append(value)
                    styles[name] = _point_style({'candidate_family': candidate})
                else: ineligible[str(r.get('eligibility', 'NA'))] += 1
            for i, (name, values) in enumerate(sorted(groups.items())):
                ax.scatter([i]*len(values), values, **styles[name], label=name, rasterized=True)
            if groups:
                ax.set_xticks(range(len(groups)), sorted(groups), rotation=90, ha='center', fontsize=6); ax.set_yscale('log'); ax.axhline(1, color='black', ls='--', lw=.7)
            else: _blank(ax)
            ax.text(.01, .99, 'Ineligible retained: ' + str(dict(ineligible)), transform=ax.transAxes, va='top', fontsize=6)
            _label(ax, 'paired candidate / control', 'control / candidate time (>1 favors candidate)')
        figure = _facets(plt, comparisons, ('track', 'final_time', 'rms_target', 'max_target'), comparison_draw, 'Matched target paired speed distributions', row_height=8)
        save('Matched target paired speed distributions', figure,
             'Different final times and tolerances are never pooled. Repeated schedules/grids/seeds are paired evidence; use field-clustered inference in comparisons.json.')
        def stress(ax, vals):
            expanded = []; method_rows = {}
            for r in vals:
                method_rows[_family(r)] = r
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
            from matplotlib.patches import Rectangle
            for i, (family, tick) in enumerate(zip(families, ax.get_yticklabels())):
                row = method_rows[family]
                tick.set_color(_method_style(row)['color'])
                if _method_id(row) == 'rank1':
                    tick.set_fontweight('bold')
                    ax.add_patch(Rectangle((-.5, i-.5), len(labels), 1, fill=False,
                        edgecolor=METHOD_STYLES['rank1']['color'], linewidth=1.5))
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
                    # These estimates always compare the named control with
                    # the learned rank-one candidate, not standalone controls.
                    ax.plot(r[metric], i, '^', color=METHOD_STYLES['rank1']['color'])
                    ci = r.get(metric+'_ci95')
                    if isinstance(ci,list) and len(ci)==2 and all(finite(v) and v>0 for v in ci):
                        ax.plot(ci, [i,i], color=METHOD_STYLES['rank1']['color'], linewidth=1.8)
                labels = [f"{method_label(r.get('competitor'))} / n={r.get('train_count')} / fields={r.get('independent_fields')}" for r in usable]
                ax.set_yticks(range(len(usable)), labels, fontsize=6); ax.set_xscale('log'); ax.axvline(1,color='black',ls='--',lw=.7)
                quantity = 'RMS' if metric == 'error_ratio' else 'time'
                _label(ax, f'Competitor / Ours {quantity} (>1 favors Ours)', 'paired comparison')
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
            label = lambda r: ('sunk/' if r.get('cost_scope') == 'inherited_sunk' else '') + r['stage']
            groups=sorted({label(r) for r in values}); bottom=np.zeros(len(groups))
            for status,color in (('PASSED',COLORS['GOOD']),('FAILED',COLORS['BAD']),('SKIPPED',COLORS['NA'])):
                counts=[sum(label(r)==stage and r['status']==status for r in values) for stage in groups]
                ax.bar(groups,counts,bottom=bottom,label=status,color=color); bottom+=counts
            ax.tick_params(axis='x', labelrotation=60)
            ax.legend();_label(ax,'native allocation stage (sunk/origin shown separately)','software test executions')
        single('Every native software test execution',software_tests,
               'Repeated cases in different allocations remain separate executions. Skipped tests never count as passed GPU readiness; software correctness is not scientific superiority.')
    manifest = {'schema': 'tdn.frontier-figures/v1', 'profile': data['profile'], 'panels': panels,
        'pdf': 'frontier-atlas.pdf', 'chart_data': 'chart-data.json.gz', 'check_index': 'check-cell-index.json.gz',
        'experiment_count': len(rows), 'check_count': len(check_index), 'loss_observations': len(curves),
        'confirmation_endpoints': len(endpoints), 'comparisons': len(comparisons), 'downsampled': False,
        'software_test_executions':len(data.get('software_tests',[])),
        'confirmation_partition_coverage':data.get('confirmation_partition_coverage'),
        'unavailable_panels_are_na': True, 'sources': data['sources'], 'stage_status': data['stage_status'],
        'method_convention': METHOD_CONVENTION, 'method_labels': METHOD_LABELS,
        'marker_convention': MARKER_CONVENTION, 'method_styles': METHOD_STYLES}
    # A compact overview is an entry point, not a replacement for complete pages.
    fig, axes = plt.subplots(2,3,figsize=(18,11))
    for ax, panel in zip(axes.flat, [panels[i] for i in (0,1,5,11,13,21)]):
        ax.imshow(plt.imread(output / panel['path'])); ax.axis('off'); ax.set_title(panel['title'],fontsize=9)
    fig.suptitle(f"TDN frontier / {data['profile']} / {len(rows)} experiments / {len(check_index)} checks",fontsize=16)
    fig.tight_layout(); fig.savefig(output / 'frontier-overview.png',dpi=200); plt.close(fig)
    _write(output / 'manifest.json', manifest)
    body = ''.join(f'<section id="panel-{p["id"]}"><h2>{html.escape(p["title"])}</h2><p><strong>How to read:</strong> {html.escape(p["reading_guide"])}</p><p>{html.escape(p["note"])}</p><a href="{p["path"]}"><img loading="lazy" src="{p["path"]}" alt="{html.escape(p["title"])}"></a></section>' for p in panels)
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
