import numpy as np
import pytest

from tdn.analysis.response.compression import (apply_response, fit_decoder_optimized,
    fit_pod, reconstruct, response_metric, whiten_channels)
from tdn.analysis.response.compression import select_source_cohort


def test_response_weighting_preserves_low_energy_high_impact_direction():
    # Constant and alternating Fourier modes are orthogonal. Alternating source
    # has 100x source energy but its heat response is almost extinguished.
    sources = np.array([[[1., 1., 1., 1.]], [[10., -10., 10., -10.]]])
    weights = np.array([[[1., .01, .001, .01]], [[1., .01, .001, .01]]])
    plain = fit_pod(sources, 1, split="train", spatial_shape=(4,))
    weighted = fit_pod(sources, 1, split="train", spatial_shape=(4,), metric=response_metric(weights))
    truth = apply_response(sources, weights, (4,))
    plain_error = np.linalg.norm(apply_response(reconstruct(sources, plain), weights, (4,))-truth)
    weighted_error = np.linalg.norm(apply_response(reconstruct(sources, weighted), weights, (4,))-truth)
    assert weighted_error < .02*plain_error
    assert np.linalg.norm(reconstruct(sources, weighted)-sources) > np.linalg.norm(reconstruct(sources, plain)-sources)


def test_basis_cannot_fit_or_optimize_on_validation_or_test():
    source = np.random.default_rng(5).normal(size=(3, 4, 8))
    for split in ("validation", "development_test", "confirmation"):
        with pytest.raises(ValueError, match="train"):
            fit_pod(source, 2, split=split, spatial_shape=(8,))
    model = fit_pod(source, 2, split="train", spatial_shape=(8,))
    with pytest.raises(ValueError, match="train"):
        fit_decoder_optimized(source, np.ones_like(source), model, split="validation")


def test_equal_full_rank_reconstruction_and_two_dimensional_decoder():
    rng = np.random.default_rng(10)
    source = rng.normal(size=(3, 4, 16))
    model = fit_pod(source, 12, split="train", spatial_shape=(4, 4))
    np.testing.assert_allclose(reconstruct(source, model), source, atol=2e-14)
    weights = np.ones((3, 4, 4, 4))*.25
    np.testing.assert_allclose(apply_response(source, weights, (4, 4)), source.mean(1), atol=1e-14)


def test_metric_cannot_break_hermitian_reality():
    source = np.ones((2, 3, 8))
    source[:, :, 1] = 2
    metric = np.ones(8)
    metric[1] = 7
    with pytest.raises(ValueError, match="conjugate"):
        fit_pod(source, 2, split="train", spatial_shape=(8,), metric=metric)


def test_decoder_training_records_measured_loss_and_preserves_rank():
    source = np.random.default_rng(5).normal(size=(3, 4, 8))
    weights = np.ones_like(source)*.25
    model = fit_pod(source, 2, split="train", spatial_shape=(8,))
    fitted = fit_decoder_optimized(source, weights, model, split="train", updates=5, learning_rate=.001)
    assert len(fitted["loss_history"]) == 5
    assert np.linalg.matrix_rank(fitted["encoder"]@fitted["decoder"], tol=1e-8) == 2
    assert fitted["training_sha256"] == model["training_sha256"]
    assert fitted["fit_seconds"] > 0


def test_held_out_data_never_change_training_basis():
    source = np.random.default_rng(51).normal(size=(3, 4, 8))
    model = fit_pod(source, 2, split="train", spatial_shape=(8,))
    before = model["decoder"].copy()
    reconstruct(np.ones((2, 5, 8))*1e6, model)
    np.testing.assert_array_equal(model["decoder"], before)


def test_regularized_whitening_is_train_only_and_preserves_observable_prediction():
    rng = np.random.default_rng(19)
    train = rng.normal(size=(4, 8, 16))*np.geomspace(.001, 1., 16)
    test = rng.normal(size=(2, 4, 16))
    model = fit_pod(train, 4, split="train", spatial_shape=(16,))
    whitened = whiten_channels(train, model, split="train")
    np.testing.assert_allclose(reconstruct(test, whitened), reconstruct(test, model), atol=1e-13)
    channels=(train@whitened["encoder"]).reshape(-1,4)
    np.testing.assert_allclose(np.cov(channels.T), np.eye(4), atol=1e-4)
    with pytest.raises(ValueError,match="training"):
        whiten_channels(test, model, split="development_test")


def test_bounded_source_menu_uses_only_inputs_not_teacher_or_loss():
    samples=[dict(record=dict(parent_id=f"{split}-{p:03d}", split=split, query_id=f"{split}-{p}-{h}-{phase}",
        horizon=h, phase_variant=phase), target=object())
        for split in ("train","validation","development_test") for p in range(40)
        for h in (.02,.04,.06,.08) for phase in ("original","phase_twin")]
    selected,parents,horizons,omitted=select_source_cohort(samples)
    assert {k:len(v) for k,v in parents.items()}=={"train":32,"validation":16,"development_test":16}
    assert len(selected)==128
    assert horizons=={"train":.06,"validation":.04,"development_test":.08}
    assert all(sum(s["record"]["parent_id"]==p for s in selected)==2 for menu in parents.values() for p in menu)
    # Neither returned order nor identities depend on incidental artifact order.
    reversed_selected,*_=select_source_cohort(samples[::-1])
    assert [s["record"]["query_id"] for s in selected]==[s["record"]["query_id"] for s in reversed_selected]
    assert len(omitted)+len(selected)==len(samples)


def test_bounded_compression_menu_rejects_parent_split_leakage():
    samples=[dict(record=dict(parent_id="same",split=split,query_id=split,horizon=.04)) for split in ("train","validation")]
    with pytest.raises(ValueError,match="cross"):
        select_source_cohort(samples)
