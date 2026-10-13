"""Prospective campaign identities and resource limits, without native claims."""
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from tdn.analysis.tangent_response.contracts import digest
from tdn.analysis.tangent_response.protocol import (
    ancestors, build_protocol, query_plan, training_workload, validate_protocol,
)


ROOT = Path(__file__).resolve().parents[1]


def measured_calibration(*, seconds=8.):
    """Parser fixture only: it does not establish an actual native measurement."""
    p = build_protocol('pilot', phase='learning')
    return dict(data=dict(native=True, device='cuda', execution_mode='desktop-slurm',
        measurements={arm: dict(completed_updates=64, seconds_per_update=seconds,
            workload_sha256=training_workload(p, arm)) for arm in p['training']['arms']}))


def test_cpu_whole_dag_is_bounded_and_cannot_be_confirmation():
    p = build_protocol('cpu-smoke')
    assert validate_protocol(p) is p
    assert p['phase'] == 'all' and p['confirmation'] is False
    assert p['execution_scope'] == 'CPU_IMPLEMENTATION_SMOKE'
    assert {u['device'] for u in p['units'].values()} == {'cpu'}
    assert set(p['units']['report']['dependencies']) == set(p['units']) - {'report'}
    assert len(p['training']['seeds']) == 1 and p['training']['updates'] == 2
    assert p['training']['head_warmup_updates'] < p['training']['updates']
    assert p['reference']['tolerance'] == p['mechanism']['tolerance'] == 1e-10
    assert p['precision']['input'] == 'float32' and p['precision']['path'] == 'float64'
    assert p['precision']['tf32'] is False
    for stage, unit in p['units'].items():
        assert set(unit['dependencies']) <= set(ancestors(p, stage))
        assert 0 < unit['cpus'] <= 8 and 0 < unit['mem_gib'] <= 48
        assert 0 < unit['seconds'] <= 1800
        h, m, s = map(int, unit['walltime'].split(':'))
        assert unit['seconds'] + 120 <= 3600*h + 60*m + s <= 2100


@pytest.mark.parametrize('target', ['fd_same_grid', 'galerkin_same_grid'])
def test_protocol_survives_json_roundtrip_with_complete_query_menu(target):
    p = build_protocol('cpu-smoke', target, run_id='serialization-check')
    restored = json.loads(json.dumps(p, allow_nan=False))
    assert validate_protocol(restored) == p
    for cohort, key in [('mechanism', 'queries'), ('fit', 'fit_queries')]:
        assert restored[key] == query_plan(restored, cohort)
        ids = [q['query_id'] for q in restored[key]]
        assert len(ids) == len(set(ids))
        for q in restored[key]:
            without_id = {k: v for k, v in q.items() if k != 'query_id'}
            assert q['query_id'] == 'tr-query-' + digest(without_id)[:24]
            assert q['independent_unit'] == q['parent_id']


def test_fresh_namespaces_parent_disjointness_and_paired_descendants():
    smoke = build_protocol('cpu-smoke')
    pilot = build_protocol('pilot')
    development = build_protocol('development')
    all_sets = []
    for p in (smoke, pilot, development):
        for key in ('parents', 'fit_parents'):
            ids = {row['parent_id'] for row in p[key]}
            seeds = {row['seed'] for row in p[key]}
            assert len(ids) == len(seeds) == len(p[key])
            all_sets.append(ids)
            split_ids = [{r['parent_id'] for r in p[key] if r['split'] == s}
                         for s in ('train', 'validation', 'diagnostic')]
            assert all(a.isdisjoint(b) for i, a in enumerate(split_ids) for b in split_ids[i+1:])
    assert all(a.isdisjoint(b) for i, a in enumerate(all_sets) for b in all_sets[i+1:])
    paired = [q for q in pilot['queries'] if q['split'] == 'diagnostic' and q['slot'] == 0]
    assert {tuple(q['grid']) for q in paired} >= {(16, 16), (32, 32)}
    assert {'primary', 'phase_intervention', 'physics_intervention'} <= {q['panel'] for q in paired}
    assert len({q['parent_id'] for q in paired}) == 1
    assert all(q['extrapolation'] == (q['horizon'] > .12) for q in paired)


@pytest.mark.parametrize('kwargs', [
    dict(target='galerkin_projected_fine'), dict(target='ambiguous'),
    dict(profile='confirmation'), dict(profile='pilot', phase='all'),
    dict(chunk_seconds=0), dict(chunk_seconds=1801), dict(chunk_seconds=True),
    dict(sampling_seconds=.5), dict(sampling_seconds=True),
    dict(worker_telemetry=('gpu', 'gpu')), dict(worker_telemetry=('unknown',)),
])
def test_invalid_target_scope_or_resource_controls_rejected(kwargs):
    with pytest.raises(ValueError):
        build_protocol(**kwargs)


@pytest.mark.parametrize('mutation', [
    'query_horizon', 'query_removed', 'fit_target', 'parent_seed', 'precision',
    'unit_budget', 'unit_zero_cpu', 'report_dependencies', 'gate_threshold',
])
def test_serialized_frozen_plan_cannot_be_silently_changed(mutation):
    p = build_protocol('cpu-smoke')
    if mutation == 'query_horizon': p['queries'][0]['horizon'] += .001
    elif mutation == 'query_removed': p['queries'].pop()
    elif mutation == 'fit_target': p['fit_queries'][0]['target_kind'] = 'galerkin_projected_fine'
    elif mutation == 'parent_seed': p['parents'][0]['seed'] += 1
    elif mutation == 'precision': p['precision']['tf32'] = True
    elif mutation == 'unit_budget': p['units']['audit']['seconds'] += 1
    elif mutation == 'unit_zero_cpu': p['units']['audit']['cpus'] = 0
    elif mutation == 'report_dependencies': p['units']['report']['dependencies'] = []
    elif mutation == 'gate_threshold': p['gates']['A']['rms_ratio'] = 1.01
    with pytest.raises(ValueError, match='Frozen|immutable|envelope'):
        validate_protocol(p)


def test_uncalibrated_native_learning_has_no_allocations_or_automatic_expansion():
    p = build_protocol('pilot', phase='learning')
    assert not any(u['kind'] == 'training' for u in p['units'].values())
    assert all(t['total_seconds'] is None for t in p['trials'].values())
    assert all(t['allocation_status'] == 'UNAVAILABLE_UNTIL_NATIVE_CALIBRATION' for t in p['trials'].values())
    with pytest.raises(ValueError): validate_protocol(p)
    p = build_protocol('development', phase='mechanism')
    with pytest.raises(ValueError, match='Development'): validate_protocol(p)


def test_measured_allowance_has_exact_work_and_continuation_chunks():
    p = build_protocol('pilot', phase='learning', calibration=measured_calibration(),
                       prerequisites={'fit_prepare': {'parser_fixture': True}}, chunk_seconds=1500)
    validate_protocol(p)
    for trial in p['trials'].values():
        chunks = [u for u in p['units'].values() if u.get('trial_id') == trial['id']]
        assert trial['updates'] == 300 and trial['total_seconds'] == math.ceil(2*8*300+120)
        assert sum(c['seconds'] for c in chunks) == trial['total_seconds']
        assert len(chunks) == 4 and [c['chunk_index'] for c in chunks] == list(range(4))
        assert all(c['device'] == 'cuda' and c['cpus'] == 4 for c in chunks)
        assert chunks[0]['previous_chunk'] is None
        for first, second in zip(chunks, chunks[1:]):
            assert second['previous_chunk'] == first['id']
            assert second['dependencies'] == [first['id']]


@pytest.mark.parametrize('mutation', ['cpu', 'not_native', 'missing_arm', 'short_work', 'workload', 'zero', 'nan', 'boolean'])
def test_native_calibration_requires_all_matching_actual_committed_work(mutation):
    calibration = measured_calibration()
    data = calibration['data']; row = data['measurements']['learned_response']
    if mutation == 'cpu': data['device'] = 'cpu'
    elif mutation == 'not_native': data['native'] = False
    elif mutation == 'missing_arm': data['measurements'].pop('residual_fno')
    elif mutation == 'short_work': row['completed_updates'] = 63
    elif mutation == 'workload': row['workload_sha256'] = '0'*64
    elif mutation == 'zero': row['seconds_per_update'] = 0
    elif mutation == 'nan': row['seconds_per_update'] = float('nan')
    elif mutation == 'boolean': row['seconds_per_update'] = True
    with pytest.raises(ValueError):
        build_protocol('pilot', phase='learning', calibration=calibration)


def test_gpu_manifest_is_exact_source_bound_and_matches_collected_inventory():
    from tdn.analysis.tangent_response.gpu_manifest import (
        GPU_TEST_CASES, GPU_TEST_FILES, GPU_TEST_SOURCE_FILES, test_source_hashes,
    )
    p = build_protocol('cpu-smoke')
    assert p['gpu_test_cases'] == list(GPU_TEST_CASES)
    assert p['gpu_test_files'] == list(GPU_TEST_FILES)
    assert len({(r['classname'], r['name']) for r in GPU_TEST_CASES}) == len(GPU_TEST_CASES)
    hashes = test_source_hashes()
    assert set(hashes) == set(GPU_TEST_SOURCE_FILES) and all(len(s) == 64 for s in hashes.values())
    assert p['gpu_test_source_sha256'] == hashes
    result = subprocess.run([sys.executable, '-m', 'pytest', '--collect-only', '-q', '-m', 'gpu', *GPU_TEST_FILES],
                            cwd=ROOT, text=True, capture_output=True, check=True)
    collected = set()
    for line in result.stdout.splitlines():
        if '::test_' in line:
            module, name = line.strip().split('::', 1)
            collected.add((module.replace('/', '.').removesuffix('.py'), name))
    assert collected == {(r['classname'], r['name']) for r in GPU_TEST_CASES}
    altered = deepcopy(p); altered['gpu_test_cases'].pop()
    with pytest.raises(ValueError, match='Frozen'): validate_protocol(altered)
    altered = deepcopy(p); altered['gpu_test_cases'][0]['name'] = 'undeclared_case'
    with pytest.raises(ValueError, match='Frozen'): validate_protocol(altered)
    # An in-memory edit must not mutate the canonical module-level inventory.
    p['gpu_test_cases'][0]['name'] = 'in_memory_changed_case'
    with pytest.raises(ValueError, match='Frozen'): validate_protocol(p)
    assert GPU_TEST_CASES[0]['name'] != 'in_memory_changed_case'
    p = build_protocol('cpu-smoke')
    altered = deepcopy(p); altered['gpu_test_source_sha256'][GPU_TEST_FILES[0]] = '0'*64
    with pytest.raises(ValueError, match='Frozen'): validate_protocol(altered)
