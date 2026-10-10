"""Replay correctness, information boundaries and immutable-source visualization."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tdn.analysis.adjacent.fields import make_parent, sample_field
from tdn.analysis.adjacent.models import make_model
from tdn.analysis.adjacent import prediction_view as view
from tdn.numerics import Equation, Geometry


@pytest.fixture(scope="module", autouse=True)
def bounded_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def example():
    parent = make_parent(90531, .5, levels=2, rms=.04)
    initial = sample_field(parent, 8).numpy()
    eq, geom = Equation(.004, 3.), Geometry((8, 8), (1., 1.))
    torch.manual_seed(83017)
    model = make_model("channel_neural", "discrete", dict(modes=2, split_modes=1)).float()
    with torch.no_grad():
        model.conditioner[-1].weight.fill_(.03)
        model.conditioner[-1].bias.copy_(torch.tensor([.02, -.01, .04]))
        truth = model(torch.from_numpy(initial).float(), .06, eq, geom).double().numpy() + 1e-6
    return parent, initial, truth, model, eq, geom


def test_prediction_view_exact_channel_and_parameter_mapping(example):
    _, initial, truth, model, eq, geom = example
    info, arrays = view.compute_prediction_data(model, initial, truth, .06, eq, geom)
    reconstructed = arrays['baseline'] + np.sum(arrays['channels'] * arrays['gains'][..., None, None], axis=1, keepdims=True)
    np.testing.assert_allclose(reconstructed, arrays['prediction'], atol=6e-8, rtol=0)
    np.testing.assert_allclose(arrays['error'], arrays['prediction'].astype('float64') - truth, atol=0, rtol=0)
    first = arrays['features'] @ arrays['parameter__conditioner_0_weight'].T + arrays['parameter__conditioner_0_bias']
    hidden = first / (1 + np.exp(-first))
    logits = hidden @ arrays['parameter__conditioner_2_weight'].T + arrays['parameter__conditioner_2_bias']
    np.testing.assert_allclose(1+.75*np.tanh(logits), arrays['gains'], atol=1e-7, rtol=0)
    assert info['network']['trainable_parameters'] == 59
    assert info['network']['exact_gain_reconstruction']
    assert info['physical']['grid'] == [8, 8]
    assert info['error_rms'] == pytest.approx(1e-6, rel=1e-9)


def test_prediction_view_reference_cannot_influence_forward(example):
    _, initial, truth, model, eq, geom = example
    first, a = view.compute_prediction_data(model, initial, truth, .06, eq, geom)
    second, b = view.compute_prediction_data(model, initial, truth+.01, .06, eq, geom)
    for key in ('prediction','channels','gains','features','baseline','correction'):
        np.testing.assert_array_equal(a[key], b[key])
    assert first['error_rms'] != second['error_rms']
    assert not first['reference_used_for_inference']


def test_prediction_view_gain_sensitivity_is_not_network_weight_sensitivity(example):
    _, initial, truth, model, eq, geom = example
    info, arrays = view.compute_prediction_data(model, initial, truth, .06, eq, geom)
    for channel in range(3):
        for index, delta in enumerate(arrays['gain_perturbation_deltas']):
            np.testing.assert_allclose(arrays['gain_perturbation_changes'][channel,index],
                delta * arrays['gain_derivative'][:,channel:channel+1].astype('float64'), atol=7e-17, rtol=1e-7)
    assert info['sensitivity']['not_input_or_network_weight_derivative']
    with pytest.raises(ValueError, match='perturbation'):
        view.compute_prediction_data(model, initial, truth, .06, eq, geom, gain_delta=1.)


def test_prediction_view_detects_field_identity_mismatch(example):
    parent, initial, truth, _, _, _ = example
    row = dict(parent=parent.to_dict(), parent_index=0, grid=8, state_key='state', key='truth')
    bank = {'binding': {'parents': [parent.to_dict()]}}
    stored = {'state': initial.copy(), 'truth': truth.copy()}
    recovered, a, b = view._validate_bank_arrays(bank, stored, row)
    assert recovered.identity_sha256 == parent.identity_sha256
    np.testing.assert_array_equal(a, initial)
    np.testing.assert_array_equal(b, truth)
    stored['state'][0,0,0,0] += .001
    with pytest.raises(ValueError, match='continuous parent'):
        view._validate_bank_arrays(bank, stored, row)


def test_prediction_view_output_guard_before_source_access(tmp_path, monkeypatch):
    monkeypatch.setattr(view, 'ROOT', tmp_path)
    source = tmp_path/'source'; source.mkdir()
    with pytest.raises(ValueError, match='new output'):
        view.render_prediction_view(source, source/'graphics')
    with pytest.raises(ValueError, match='inside this project'):
        view._contained(tmp_path.parent/'outside')
    target = tmp_path/'target'; target.mkdir()
    linked = tmp_path/'link'; linked.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match='symlinks'):
        view._contained(linked/'new')
    with pytest.raises(ValueError, match='400'):
        view.render_prediction_view(source, tmp_path/'new', dpi=72)


def test_prediction_view_renders_actual_cells_at_small_test_dpi(example, tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.axes import Axes
    _, initial, truth, model, eq, geom = example
    info, arrays = view.compute_prediction_data(model, initial, truth, .06, eq, geom)
    info.update(selected_model={'family':'channel_neural','selection_status':'TEST_FIXTURE_ONLY',
                'selected_update':0,'checkpoint_sha256':'0'*64}, parent_id='unit-test-field')
    calls = []
    original = Axes.imshow
    def tracked(self, value, *args, **kwargs):
        calls.append((np.asarray(value).shape, kwargs.get('interpolation')))
        return original(self, value, *args, **kwargs)
    monkeypatch.setattr(Axes, 'imshow', tracked)
    paths = view._images(info, arrays, tmp_path, dpi=40)
    assert len(paths) == 6 and all((tmp_path/p).is_file() for p in paths)
    assert all(interpolation == 'nearest' for _, interpolation in calls)
    assert sum(shape == (8,8) for shape, _ in calls) >= 20


def test_prediction_view_actual_sealed_replay_if_available(tmp_path, monkeypatch):
    source = view.ROOT/'runs/adjacent-local-review-v2'
    if not source.exists():
        pytest.skip('Archived local development run is not distributed as a test fixture')
    bundle = view.load_bundle(source, family='channel_neural', track='discrete', parent_index=0)
    before = deepcopy(bundle['provenance']['source_artifacts'])
    info, _ = view.compute_prediction_data(bundle['model'], bundle['initial'], bundle['reference_state'],
        bundle['reference']['horizon'], bundle['equation'], bundle['geometry'])
    assert info['error_rms'] == pytest.approx(bundle['original_evaluation']['error_rms'], abs=1e-12)
    assert info['error_max'] == pytest.approx(bundle['original_evaluation']['error_max'], abs=1e-12)
    assert bundle['spec']['selection_status'] == 'FITTED_CHECKPOINT'
    assert bundle['spec']['selected_update'] == 2
    assert all(view.file_digest(view.ROOT/path) == expected for path,expected in before.items())
    # Exercise provenance and complete output sealing; plotting has a separate
    # actual-cell render test, so this avoids duplicating high-resolution work.
    monkeypatch.setattr(view, '_images', lambda *args, **kwargs: [])
    output = tmp_path / 'sealed-replay'
    result = view.render_prediction_view(source, output)
    manifest = json.loads((output/'manifest.json').read_text())
    assert result['replay']['status'] == 'SAME_CPU_METRIC_REPLAY'
    assert manifest['source_artifacts'] == before
    assert all(view.file_digest(output/path) == expected
               for path, expected in manifest['files'].items())
    assert all(view.file_digest(view.ROOT/path) == expected
               for path, expected in manifest['renderer']['files'].items())


def test_prediction_view_near_bound_fitted_gains_clip_probes(example):
    _, initial, truth, _, eq, geom = example
    model = make_model('channel_affine', 'discrete', dict(modes=2, split_modes=1)).float()
    with torch.no_grad():
        model.affine_coefficients[0].copy_(torch.tensor([-20., 20., 0.]))
    info, arrays = view.compute_prediction_data(model, initial, truth, .06, eq, geom)
    actual = arrays['gain_perturbation_actual_deltas']
    assert actual.shape == (3, 9)
    assert actual[0, 0] == 0 and actual[1, -1] == 0
    assert actual[2, 0] == pytest.approx(-.1) and actual[2, -1] == pytest.approx(.1)
    perturbed = arrays['gains'].reshape(3,1) + actual
    assert np.all(perturbed >= .25) and np.all(perturbed <= 1.75)
    for channel in range(3):
        for index, delta in enumerate(actual[channel]):
            np.testing.assert_allclose(arrays['gain_perturbation_changes'][channel,index],
                delta * arrays['gain_derivative'][:,channel:channel+1].astype('float64'), atol=7e-17, rtol=1e-7)
    assert info['sensitivity']['curves'][0]['clipped_to_admissible_range']
    assert info['sensitivity']['curves'][1]['clipped_to_admissible_range']
    assert not info['sensitivity']['curves'][2]['clipped_to_admissible_range']
