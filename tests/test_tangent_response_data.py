"""Fresh continuous fields, prospective materialization and private label seals."""
from copy import deepcopy
import hashlib
import json

import pytest
import torch

from tdn.analysis.tangent_response import data
from tdn.analysis.tangent_response.contracts import PrecisionContract, digest
from tdn.analysis.tangent_response.protocol import build_protocol


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def bank_and_sample(*, target='fd_same_grid', cohort='mechanism'):
    p = build_protocol('cpu-smoke', target)
    bank = data.parent_bank(p, cohort)
    spec = p['queries' if cohort == 'mechanism' else 'fit_queries'][0]
    return p, bank, data.make_query(p, bank, spec)


def labeled_sample(sample):
    """Known tensors for serialization/screening; these are not teacher evidence."""
    sample = deepcopy(sample); u = sample['u0']
    a64 = u.double(); a32 = u.clone(); teacher = a64 + 1e-4
    sample.update(teacher64=teacher, target=teacher, smooth_anchor64=a64, anchor_fp64=a64,
                  deployed_anchor32=a32, anchor_fp32=a32,
                  **PrecisionContract().labels(teacher, a64, a32))
    sample['reference'] = {'accepted': True, 'uncertainty_rms': 1e-11, 'fixture': 'serialization_only'}
    sample['oracle_metadata'] = {'accepted': True, 'fixture': 'serialization_only'}
    sample['action_labels'] = [dict(s=.01, h=.03, accepted=True, z=u.double()-.5,
                                    output64=2*(u.double()-.5), label_role='test_fixture')]
    return sample


def test_continuous_parent_resolution_preserves_field_not_nodal_interpolation():
    p = build_protocol('pilot'); bank = data.parent_bank(p)
    parent = bank['parents'][0]; field = parent['field_spec']
    coarse = data.materialize_parent(parent, field, (16, 16))
    fine = data.materialize_parent(parent, field, (32, 32))
    torch.testing.assert_close(coarse, fine[..., ::2, ::2], atol=0, rtol=0)
    assert coarse.dtype == fine.dtype == torch.float32
    assert float(fine.min()) >= field['mean']-field['amplitude']-1e-7
    assert float(fine.max()) <= field['mean']+field['amplitude']+1e-7
    assert 0 < fine.min() < fine.max() < 1
    assert parent['shape_sha256'] == digest({key: field[key] for key in ('modes', 'weights', 'phases')})


@pytest.mark.parametrize('size', [5, 6])
def test_phase_intervention_changes_signed_field_preserving_modal_power(size):
    p, bank, _ = bank_and_sample(); parent = bank['parents'][0]; field = parent['field_spec']
    baseline = data.materialize_parent(parent, field, (size, size))
    phase = data.materialize_parent(parent, field, (size, size), phase_shift=.125)
    assert float((phase-baseline).abs().max()) > .001
    power = lambda x: torch.fft.fftn(x.double(), dim=(-2, -1), norm='forward').abs().square()
    torch.testing.assert_close(power(phase), power(baseline), atol=2e-9, rtol=1e-5)
    torch.testing.assert_close(phase.mean(), baseline.mean(), atol=1e-7, rtol=0)


def test_fresh_bank_is_reproducible_and_rejects_prior_shape_reuse():
    p, bank, _ = bank_and_sample()
    assert data.parent_bank(p) == bank
    assert bank['protocol_sha256'] == digest(p)
    with pytest.raises(ValueError, match='reused'):
        data.parent_bank(p, prior_shape_hashes=[bank['parents'][0]['shape_sha256']])
    fit = data.parent_bank(p, 'fit')
    assert {r['shape_sha256'] for r in bank['parents']}.isdisjoint({r['shape_sha256'] for r in fit['parents']})


@pytest.mark.parametrize('target', ['fd_same_grid', 'galerkin_same_grid'])
def test_declared_physics_intervention_keeps_field_and_matches_active_generator(target):
    p, bank, _ = bank_and_sample(target=target)
    query = next(q for q in p['queries'] if q['panel'] == 'physics_intervention')
    sample = data.make_query(p, bank, query); record = sample['record']
    assert record['role'] == 'physics_intervention' and record['intervention'] is True
    assert record['primary_case'] is False
    assert record['independent_unit'] == record['parent_id']
    assert record['kappa']*.12*record['active_generator_magnitude_at_unit_kappa'] == pytest.approx(query['diffusion_number'])
    assert record['reaction_rate']*.12 == pytest.approx(query['reaction_number'])
    primary = next(q for q in p['queries'] if q['parent_id'] == query['parent_id'] and q['panel'] == 'primary' and q['grid'] == query['grid'])
    torch.testing.assert_close(sample['u0'], data.make_query(p, bank, primary)['u0'], atol=0, rtol=0)


@pytest.mark.parametrize('mutation', ['horizon', 'projected_fine', 'new_grid', 'bank_protocol'])
def test_materialization_rejects_unregistered_query_or_stale_bank(mutation):
    p, bank, _ = bank_and_sample(); query = deepcopy(p['queries'][0])
    if mutation == 'horizon': query['horizon'] += .001
    elif mutation == 'projected_fine': query['target_kind'] = 'galerkin_projected_fine'
    elif mutation == 'new_grid': query['grid'] = [7, 7]
    else: bank['protocol_sha256'] = '0'*64
    with pytest.raises(ValueError, match='frozen|prospective'):
        data.make_query(p, bank, query)


def test_private_labels_roundtrip_dtypes_aliases_and_action_opportunities(tmp_path):
    _, _, sample = bank_and_sample(); sample = labeled_sample(sample)
    data.save_samples(tmp_path/'labels', [sample])
    restored = data.load_samples(tmp_path/'labels')[0]
    assert restored['record'] == sample['record']
    for key, value in sample.items():
        if torch.is_tensor(value):
            assert restored[key].dtype == value.dtype
            torch.testing.assert_close(restored[key], value, atol=0, rtol=0)
    assert restored['target'] is restored['teacher64']
    assert restored['anchor_fp64'] is restored['smooth_anchor64']
    assert restored['anchor_fp32'] is restored['deployed_anchor32']
    torch.testing.assert_close(restored['action_labels'][0]['output64'], sample['action_labels'][0]['output64'], atol=0, rtol=0)
    torch.testing.assert_close(restored['deployment_defect'], restored['physical_defect']+restored['arithmetic_bridge'], atol=0, rtol=0)
    metadata = json.loads((tmp_path/'labels/samples.json').read_text())
    assert metadata['numerical_payload_sha256'] == hashlib.sha256((tmp_path/'labels/samples.npz').read_bytes()).hexdigest()
    assert 'never model inference' in metadata['label_scope']


def test_completed_private_samples_are_preserved_on_second_save(tmp_path):
    _, _, sample = bank_and_sample(); path = tmp_path/'labels'
    data.save_samples(path, [labeled_sample(sample)])
    before = {name: (path/name).read_bytes() for name in ('samples.json', 'samples.npz')}
    changed = labeled_sample(sample); changed['u0'] = changed['u0'] + .01
    with pytest.raises((ValueError, FileExistsError), match='Preserve|exist|fresh'):
        data.save_samples(path, [changed])
    assert before == {name: (path/name).read_bytes() for name in before}


def test_private_numerical_payload_corruption_and_symlinks_rejected(tmp_path):
    _, _, sample = bank_and_sample(); path = tmp_path/'labels'
    data.save_samples(path, [labeled_sample(sample)])
    linked = tmp_path/'linked'; linked.symlink_to(path, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'): data.load_samples(linked)
    (path/'samples.npz').write_bytes((path/'samples.npz').read_bytes()+b'corruption')
    with pytest.raises(ValueError, match='identity'): data.load_samples(path)


def screened_fixture():
    p = build_protocol('cpu-smoke'); bank = data.parent_bank(p, 'fit')
    samples = [labeled_sample(data.make_query(p, bank, q)) for q in p['fit_queries']]
    return p, samples


def test_screening_requires_complete_parent_menu_and_retains_failure_rows():
    p, samples = screened_fixture(); selected, report = data.screening(samples, p)
    assert report['status'] == 'ACCEPTED' and len(selected) == len(samples)
    assert report['accepted_parent_slots'] == sum(p['fit_parent_counts'].values())
    missing = samples[1:]; selected, report = data.screening(missing, p)
    assert report['status'] == 'UNRESOLVED'
    failed = next(row for row in report['candidates'] if row['parent_id'] == samples[0]['record']['parent_id'])
    assert failed['eligible'] is False and 'MISSING_REQUESTED_QUERY' in failed['reasons']
    assert len(report['candidates']) == len(p['fit_parents'])


@pytest.mark.parametrize('failure', ['reference', 'oracle', 'uncertainty', 'precision'])
def test_screening_censors_missing_accuracy_and_precision_headroom(failure):
    p, samples = screened_fixture(); item = samples[0]
    if failure == 'reference': item['reference']['accepted'] = False
    elif failure == 'oracle': item['oracle_metadata']['accepted'] = False
    elif failure == 'uncertainty': item['reference']['uncertainty_rms'] = 1.
    else: item['deployed_anchor32'] = torch.full_like(item['u0'], 100.)
    selected, report = data.screening(samples, p)
    assert report['status'] == 'UNRESOLVED'
    assert item['record']['parent_id'] not in report['selected_parent_ids']
    assert len(report['candidates']) == len(p['fit_parents'])


def test_screening_rejects_duplicate_query_as_complete_parent_evidence():
    p, samples = screened_fixture()
    selected, report = data.screening(samples+[deepcopy(samples[0])], p)
    assert report['status'] == 'UNRESOLVED'
    assert samples[0]['record']['parent_id'] not in report['selected_parent_ids']


def test_screening_selects_first_eligible_prospective_candidate_without_loss():
    p = build_protocol('pilot', phase='fit_prepare'); bank = data.parent_bank(p, 'fit')
    samples = [labeled_sample(data.make_query(p, bank, q)) for q in p['fit_queries']]
    first = p['fit_parents'][0]; second = p['fit_parents'][1]
    assert first['slot'] == second['slot'] and [first['candidate'], second['candidate']] == [0, 1]
    next(s for s in samples if s['record']['parent_id'] == first['parent_id'])['reference']['accepted'] = False
    selected, report = data.screening(samples, p)
    assert report['status'] == 'ACCEPTED'
    assert second['parent_id'] in report['selected_parent_ids']
    assert first['parent_id'] not in report['selected_parent_ids']
    assert len(report['candidates']) == len(p['fit_parents'])
    assert len(report['selected_parent_ids']) == sum(p['fit_parent_counts'].values())
    assert all(s['record']['parent_id'] in report['selected_parent_ids'] for s in selected)
