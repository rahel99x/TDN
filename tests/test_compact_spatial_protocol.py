"""Freeze genuinely paired stress inputs and keep interpretation boundaries."""
from copy import deepcopy
import math

import pytest
import torch
import yaml
from pathlib import Path

from tdn.analysis.compact_spatial.bank import cases, plan, state
from tdn.analysis.compact_spatial.protocol import (
    DEFAULTS, METHODS, TABLE_IDS, digest, make_protocol, validate_config, validate_rows,
)
from tdn.analysis.work_precision.experiment import cases as old_cases, state as old_state


@pytest.mark.parametrize("key,value", [
    ("protocol_version", True), ("suite", "work-precision"), ("max_seconds", 1201),
    ("reference_tolerance", 1e-7), ("reference_attempts", 3), ("intraop_threads", 2),
    ("interop_threads", 2), ("warmup_rollouts", 0), ("timing_repeats", 4),
    ("nsteps", [1.0, 2, 4, 8, 16, 32]), ("scaling_nsteps", [True, 8]),
    ("error_targets", [2e-3]), ("methods", list(reversed(METHODS))),
])
def test_config_rejects_silent_tuning_and_boolean_numeric_aliases(key, value):
    raw = deepcopy(DEFAULTS)
    raw[key] = value
    with pytest.raises(ValueError):
        validate_config(raw)


def test_checked_in_config_matches_versioned_defaults():
    root = Path(__file__).resolve().parents[1]
    assert yaml.safe_load((root / 'configs/compact-spatial.yaml').read_text()) == DEFAULTS
    for raw in ({}, {**DEFAULTS, "adaptive": True}, []):
        with pytest.raises(ValueError):
            validate_config(raw)


def test_complete_plan_counts_roles_and_rotation_are_frozen():
    config = validate_config(DEFAULTS)
    p = plan(config)
    assert {name: len(ids) for name, ids in p['expected_ids'].items()} == {
        'candidate_rows': 2760, 'frontier_rows': 4640, 'reference_rows': 58, 'parity_rows': 624}
    assert len([s for s in p['cases'] if s['panel'] == 'accuracy']) == 40
    assert len([s for s in p['cases'] if s['case_role'] == 'new_diagnostic']) == 16
    assert len([s for s in p['cases'] if s['panel'] == 'scaling']) == 18
    for ids in p['expected_ids'].values():
        assert len(ids) == len(set(ids))
    for order in p['timing_orders']:
        for variants in [order['preparation_order'], order['warmup_order'], *order['measured_orders']]:
            assert len(variants) == len(METHODS) and set(variants) == set(METHODS)
        assert len(order['measured_orders']) == 5
        assert len({tuple(v) for v in order['measured_orders']}) == 5
    protocol = make_protocol(config, p, {'source_tree_sha256': 'test'}, ['test'])
    assert protocol['case_plan_sha256'] == digest(p['cases'])
    assert not protocol['training_performed'] and protocol['device'] == 'cpu'
    assert len(protocol['hypotheses']) == 5


def test_smoke_is_fixed_subset_and_does_not_mutate_full_protocol():
    raw = deepcopy(DEFAULTS)
    config = validate_config(raw, smoke=True)
    full, smoke = plan(validate_config(raw)), plan(config)
    assert config['max_seconds'] == 240 and len(smoke['cases']) == 4
    assert raw == DEFAULTS and 'smoke' not in raw
    assert {s['case_id'] for s in smoke['cases']} <= {s['case_id'] for s in full['cases']}
    for spec in smoke['cases']:
        assert spec == next(s for s in full['cases'] if s['case_id'] == spec['case_id'])


def test_review_inputs_remain_identical_and_are_never_relabeled_fresh():
    bank = cases()
    for old in old_cases():
        spec = next(s for s in bank if s['case_id'] == 'review/' + old['case_id'])
        assert spec['case_role'] in {'historical_control', 'reused_review'}
        assert torch.equal(state(spec), old_state(old))
        for key in ('grid', 'lengths', 'kappa', 'reaction_rate', 'final_time'):
            assert spec[key] == old[key]


def test_fields_are_bounded_amplitude_pairs_and_batch_prefixes():
    bank = cases()
    for spec in bank:
        u = state(spec)
        assert u.shape == (spec['batch_size'], 1, *spec['grid'])
        assert torch.isfinite(u).all() and ((u >= 0) & (u <= 1)).all()
        assert spec['cells'] == math.prod(spec['grid'])
        if spec['case_role'] == 'new_diagnostic' and spec['amplitude'] == .28:
            small = next(s for s in bank if s['case_id'] == spec['case_id'].replace('a0.28', 'a0.04'))
            torch.testing.assert_close((u - spec['mean']) / .28,
                                       (state(small) - small['mean']) / .04, rtol=1e-13, atol=3e-15)
        if spec['panel'] == 'scaling' and spec['batch_size'] > 1:
            single = next(s for s in bank if s['case_id'] == spec['case_id'].rsplit('/b', 1)[0] + '/b1')
            assert torch.equal(u[:1], state(single))
            assert not torch.equal(u[0], u[1])


def test_scaling_keeps_physics_and_continuum_inputs_fixed_across_grids():
    bank = [s for s in cases() if s['panel'] == 'scaling' and s['batch_size'] == 4]
    assert {(s['kappa'], s['reaction_rate'], s['final_time']) for s in bank} == {(.0005, 2., .1)}
    for dim in (1, 2):
        sequence = [s for s in bank if len(s['grid']) == dim]
        for a, b in zip(sequence[:-1], sequence[1:]):
            strides = [big // small for small, big in zip(a['grid'], b['grid'])]
            sliced = state(b)[(slice(None), slice(None), *[slice(None, None, n) for n in strides])]
            torch.testing.assert_close(state(a), sliced, rtol=1e-13, atol=2e-15)


def test_high_high_pair_really_has_missing_low_output_interaction():
    spec = next(s for s in cases() if s['case_id'] == 'new/1d/cutoff_pair/a0.28')
    v = state(spec) - spec['mean']
    spectrum = torch.fft.fft(v).abs()
    assert spectrum[..., 1:9].max() < 1e-12
    assert torch.fft.fft(v.square()).abs()[..., 1].item() > .1


@pytest.mark.parametrize('damage', ['duplicate', 'missing_table', 'missing_orders', 'nonfinite'])
def test_bad_declarations_are_rejected(damage):
    config = validate_config(DEFAULTS, smoke=True)
    p = plan(config)
    if damage == 'duplicate':
        p['cases'].append(p['cases'][0])
    elif damage == 'missing_table':
        p['expected_ids'].pop('reference_rows')
    elif damage == 'missing_orders':
        p.pop('timing_orders')
    else:
        p['cases'][0]['kappa'] = float('nan')
    with pytest.raises(ValueError):
        make_protocol(config, p, {}, [])
