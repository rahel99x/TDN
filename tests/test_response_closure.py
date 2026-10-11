import inspect

import numpy as np
import pytest

from tdn.analysis.response.closure import (advance, coefficients, coordinates, forecast,
    initialize_observer, lift, observation_noise, onset_diagnostics, product, project,
    rates, rhs, select_observer, shadow_rhs)
from tdn.analysis.response.closure import (_controlled_path, _teacher,
    signed_interaction, temporal_closure_attribution)


class Budget:
    def check(self):
        pass


def field():
    a = np.zeros(9, complex)
    a[4] = .4
    a[3] = a[5] = .03
    return a


def test_unit_domain_physical_diffusion_symbol_and_growth_allowed_rates():
    a = field()
    derivative = rhs(a, .02, 0.)
    assert derivative[5] == pytest.approx(-.02*(2*np.pi)**2*.03)
    growth = rates(12, .08, .0005, 2.1)
    assert growth[12+7] > 0
    assert growth[12+12] < 0


def test_exact_projected_convolution_keeps_high_to_low_interactions():
    a = np.zeros(21, complex)
    a[10] = .4
    a[19] = a[1] = .04
    a[18] = a[2] = .03
    coarse_product = product(project(a, 2), project(a, 2))
    fine_then_project = project(product(a, a), 2)
    assert abs(fine_then_project[3]-coarse_product[3]) > 1e-3
    assert fine_then_project[3] == pytest.approx(2*.04*.03)


def test_shadow_is_driven_from_zero_hidden_modes_and_full_feedback_matches_rhs():
    a = np.zeros(9, complex)
    a[4] = .4
    a[0] = a[8] = .05
    fine = lift(a, 10)
    derivative = shadow_rhs(fine, 4, 0., 3.)
    assert abs(derivative[10+8]) > 0
    np.testing.assert_allclose(shadow_rhs(fine, 4, .001, 3., full_feedback=True), rhs(fine, .001, 3.))


def test_reaction_only_closure_is_second_order_but_hidden_initial_term_is_first_order():
    measured = onset_diagnostics(Budget())
    assert min(measured["observed_orders"]) > 1.98
    assert max(measured["observed_orders"]) < 2.1
    assert measured["reaction_only"][0]["coefficient_relative_error"] < .004
    assert measured["initially_hidden"][0]["first_order_relative_error"] < .01
    assert measured["initially_hidden"][0]["first_order_relative_error"] < measured["initially_hidden"][-1]["first_order_relative_error"]


def test_observer_interface_is_causal_and_zero_gain_avoids_propagation(monkeypatch):
    previous, current = field(), field()*.99
    assert not any("truth" in p or "future" in p for p in inspect.signature(forecast).parameters)
    baseline = initialize_observer(previous, current, h=.01, kappa=.001, reaction=1., kind="current_only")
    import tdn.analysis.response.closure as module
    monkeypatch.setattr(module, "advance", lambda *a, **k: (_ for _ in ()).throw(AssertionError("unnecessary propagation")))
    zero = initialize_observer(previous, current, h=.01, kappa=.001, reaction=1., kind="deterministic", gain=0.)
    np.testing.assert_array_equal(baseline, zero)


def test_observer_assimilates_once_then_rolls_out_without_new_noise():
    previous, current = field(), field()*.99
    initial = initialize_observer(previous, current, h=.01, kappa=.001, reaction=1., kind="deterministic", gain=.5)
    expected = []
    for _ in range(4):
        initial = advance(initial, .01, .001, 1.)
        expected.append(initial)
    prediction, cost = forecast(previous, current, h=.01, kappa=.001, reaction=1., steps=4,
                                method="deterministic_observer", gain=.5)
    np.testing.assert_allclose(prediction, expected)
    assert cost["startup_seconds"] >= 0 and cost["total_seconds"] > 0


def test_noise_is_real_independent_and_has_declared_physical_rms():
    a = field()
    values = [observation_noise(a, .001, seed)-a for seed in range(1000)]
    measured = np.sqrt(np.mean([np.sum(abs(v)**2) for v in values]))
    assert measured == pytest.approx(.001, rel=.025)
    for noise in values[:5]:
        np.testing.assert_allclose(noise, noise[::-1].conj())
    np.testing.assert_array_equal(observation_noise(a, 0., 5), a)
    assert not np.array_equal(values[0], values[1])


def test_model_selection_rejects_training_or_held_out_parents():
    item = dict(previous=field(), current=field(), truth=np.stack([field()]), split="development_test")
    with pytest.raises(ValueError, match="validation"):
        select_observer([item], kind="deterministic_observer", candidates=[dict(gain=0.)], h=.01,
                        kappa=.001, reaction=1., steps=1)


def test_clean_zero_noise_coordinates_roundtrip():
    a = field()
    a[2] = .01+.03j
    a[6] = a[2].conjugate()
    np.testing.assert_array_equal(coefficients(coordinates(a)), a)


def test_temporal_refinement_reduces_actual_fixed_equation_error():
    a = field()
    times = [.2, .4]
    truth, reference = _teacher(a, times, .001, 3., Budget())
    coarse, _ = _controlled_path(a, times, .001, 3., Budget(), substeps=1)
    refined, _ = _controlled_path(a, times, .001, 3., Budget(), substeps=4)
    assert reference["accepted"]
    assert np.linalg.norm(refined[-1]-truth[-1]) < np.linalg.norm(coarse[-1]-truth[-1])/100


def test_signed_factorial_attribution_retains_destructive_interactions():
    truth = field()
    direction = np.zeros_like(truth); direction[len(truth)//2] = 1.
    measured = signed_interaction(truth+direction, truth+.2*direction,
        truth+.3*direction, truth+2*direction, truth)
    assert measured["vector_identity_maximum"] < 1e-15
    assert measured["squared_identity_discrepancy"] < 1e-14
    assert measured["combined_beats_strongest_component"] is False
    assert measured["combined_minus_strongest_component_rms"] == pytest.approx(1.8)
    assert any(row["twice_inner_product"] < 0 for row in measured["signed_cross_terms"])


def test_same_state_temporal_closure_controls_use_equal_information_and_signed_errors():
    a = np.zeros(5, complex); a[2] = .4; a[0] = a[4] = .075
    times = [.1, .2]
    truth, reference = _teacher(lift(a, 12), times, 0., 3., Budget())
    measured, raw = temporal_closure_attribution(a, times, truth, reference,
        kappa=0., reaction=3., shadow_cutoff=6, rate_shift=.2, budget=Budget())
    assert measured["decomposition"]["omitted_scale_closure"]["rms"] > 1e-5
    assert measured["decomposition"]["vector_identity_maximum"] < 1e-15
    assert len(measured["rows"]) == 8
    assert measured["all_temporal_references_accepted"]
    calls = [r["cost"]["nfev"] for r in measured["rows"] if r["component"] == "closure_only"]
    assert calls == [measured["rows"][0]["cost"]["nfev"]]*3
    # Perturbing evaluation truth changes diagnostics, never the autonomous
    # component predictions or predeclared time mesh.
    again, raw_again = temporal_closure_attribution(a, times, truth+1e-4, reference,
        kappa=0., reaction=3., shadow_cutoff=6, rate_shift=.2, budget=Budget())
    for key in raw:
        if key != "truth":
            np.testing.assert_array_equal(raw[key], raw_again[key])
    assert measured["rows"][0]["endpoint_rms"] != again["rows"][0]["endpoint_rms"]
