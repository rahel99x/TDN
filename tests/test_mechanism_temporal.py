"""Independent numerical checks for the training-free temporal audit."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy.integrate import quad

from tdn.analysis.mechanisms.temporal import (
    FIT_HORIZONS, GRID, TEST_HORIZONS, VALIDATION_HORIZONS, VARIANTS,
    basis, eigenvalue, finite_amplitude_scaling, fit_coefficients, interaction_coefficients,
    interaction_ode, phi_negative, plan, run,
)


@pytest.mark.parametrize("order", [3, 4])
@pytest.mark.parametrize("dtype,tolerance", [(np.float64, 3e-12), (np.float32, 4e-6)])
def test_phi_agrees_with_independent_adaptive_quadrature(order, dtype, tolerance):
    x = np.array([0., 1e-10, 1e-4, .5, 1.999, 2., 2.001, 12., 100., 1000.])
    # Change variables to exponential coordinate for a well-resolved boundary
    # layer at large x; this reference does not use the implementation series.
    def reference(value):
        if value == 0:
            return 1 / math.factorial(order)
        integral, error = quad(lambda s: np.exp(-s) * (1 - s / value)**(order - 1),
                               0, min(value, 50.), epsabs=1e-13, epsrel=1e-13)
        assert error < 1e-11
        return integral / value / math.factorial(order - 1)
    expected = np.array([reference(value) for value in x])
    measured = phi_negative(x, order, dtype)
    assert measured.dtype == dtype
    np.testing.assert_allclose(measured, expected, rtol=tolerance, atol=1e-15)


def test_phi_large_decay_remains_finite_without_power_overflow():
    for dtype in (np.float32, np.float64):
        for order in (3, 4):
            x = np.array([1e6, 1e20, 1e30], dtype=dtype)
            value = phi_negative(x, order, dtype)
            assert np.isfinite(value).all()
            assert (value > 0).all()
            asymptotic = 1 / math.factorial(order - 1) - 1 / (math.factorial(order - 2) * x.astype(np.float64))
            np.testing.assert_allclose(value * x, asymptotic, rtol=3e-7)


@pytest.mark.parametrize("x,order", [([-1.], 3), ([np.nan], 4), ([np.inf], 3), ([1.], 0), ([1.], True)])
def test_phi_rejects_out_of_domain_inputs(x, order):
    with pytest.raises(ValueError):
        phi_negative(x, order)


@pytest.mark.parametrize("pair", [(1, 3), (2, 2), (3, 5), (1, -1), (3, -1)])
def test_quadratic_response_agrees_with_independent_coefficient_ode(pair):
    parameters = dict(p=pair[0], q=pair[1], kappa=.02, rate=6., base=.3)
    h = [.02, .1, .4]
    exact, _ = interaction_coefficients(h, **parameters)
    expected = [interaction_ode(value, **parameters) for value in h]
    np.testing.assert_allclose(exact, expected, atol=3e-11, rtol=2e-10)


@pytest.mark.parametrize("pair", [(1, 3), (2, 2), (1, -1), (3, -1)])
def test_quadratic_coefficient_is_actual_small_amplitude_pde_and_split_limit(pair):
    # Full nonlinear PDE references exercise three amplitudes. At mode2,
    # (3,-1) must isolate its mixed response from the self harmonic of mode1.
    # DC requires a distinct projection normalization. Either mistake would
    # leave a quadratic residual instead of the quartic residual checked here.
    p, q = pair
    metrics = finite_amplitude_scaling(p, q, lambda: None)
    assert metrics["polarized_mixed_coefficient"] == (abs(p) != abs(q))
    for name in ("coupled", "split"):
        assert metrics[f"{name}_epsilon_2_coefficient"] == pytest.approx(metrics[f"{name}_coefficient"], abs=8e-6)
        for index in (0, 1):
            assert metrics[f"{name}_residual_order_{index}"] == pytest.approx(4., abs=.02)


def test_dc_transport_dictionary_records_known_nonidentifiability_and_prediction_errors():
    parameters = dict(p=1, q=-1, kappa=.002, rate=1.)
    exact, split = interaction_coefficients(FIT_HORIZONS, **parameters)
    _, fit_info = fit_coefficients(FIT_HORIZONS, exact - split, "transported", **parameters)
    assert fit_info["rank"] == 1
    rows = run({"smoke": True}, lambda: None)["rows"]
    dc_transport = [row for row in rows if row["case_id"] == "r0-p1-q-1-transported"]
    assert len(dc_transport) == 1
    assert dc_transport[0]["outcome"] == "EXPECTED_LIMITATION"
    metrics = dc_transport[0]["metrics"]
    assert metrics["nominal_coefficient_count"] == 2
    assert metrics["effective_coefficient_count"] == 1
    assert not metrics["two_coefficients_identifiable"]
    assert metrics["test_max_absolute_error"] > 0
    assert "test_0_prediction" in metrics
    assert "not a matched-two-effective-coefficient comparison" in dc_transport[0]["note"]


def test_unexpected_rank_deficiency_is_not_reclassified_as_known_limitation(monkeypatch):
    import tdn.analysis.mechanisms.temporal as temporal
    original = temporal.fit_coefficients

    def unexpected_rank(*args, **kwargs):
        coefficients, info = original(*args, **kwargs)
        if args[2] == "scalar":
            info["rank"] = 1
        return coefficients, info

    monkeypatch.setattr(temporal, "fit_coefficients", unexpected_rank)
    rows = run({"smoke": True}, lambda: None)["rows"]
    scalar_rows = [row for row in rows if row["variant"] == "scalar"]
    assert scalar_rows
    assert all(row["outcome"] == "INCONCLUSIVE" for row in scalar_rows)


def test_same_output_frequency_does_not_imply_same_nonlinear_time_response():
    h = [.03, .1, .3]
    parameters = dict(kappa=.02, rate=6.)
    a, _ = interaction_coefficients(h, p=1, q=3, **parameters)
    b, _ = interaction_coefficients(h, p=2, q=2, **parameters)
    assert eigenvalue(1 + 3) == eigenvalue(2 + 2)
    assert np.max(np.abs(a - b)) > .001
    # Different response shapes, not merely a fixed amplitude rescaling.
    assert np.ptp(a / b) > .01


def test_fit_validation_test_horizons_are_disjoint():
    sets = [set(values) for values in (FIT_HORIZONS, VALIDATION_HORIZONS, TEST_HORIZONS)]
    assert all(not left.intersection(right) for i, left in enumerate(sets) for right in sets[i + 1:])
    assert max(TEST_HORIZONS) > max(FIT_HORIZONS)


@pytest.mark.parametrize("variant", VARIANTS)
def test_fit_recovers_two_coefficients_and_heldout_targets_cannot_change_fit(variant):
    parameters = dict(p=1, q=3, kappa=.002, rate=2.)
    expected = np.array([.5, -.125])
    fitting = np.array(FIT_HORIZONS)
    targets = basis(fitting, variant, **parameters) @ expected
    original, info = fit_coefficients(fitting, targets, variant, **parameters)
    # Scoring labels are external to the fit API; perturbing them changes
    # scoring while the coefficients and predictions remain bit-identical.
    prediction = basis(TEST_HORIZONS, variant, **parameters) @ original
    test_targets = prediction.copy()
    error_before = np.max(np.abs(prediction - test_targets))
    test_targets += 1
    repeated, repeated_info = fit_coefficients(fitting, targets, variant, **parameters)
    assert np.array_equal(original, repeated)
    assert info == repeated_info
    assert np.max(np.abs(prediction - test_targets)) > error_before
    np.testing.assert_allclose(original, expected, rtol=1e-10, atol=1e-11)
    assert info["rank"] == 2


@pytest.mark.parametrize("h,target", [([.1, .1, .2], [1., 2., 3.]), ([0., .1, .2], [1., 2., 3.]),
                                     ([.1, .2, .3], [1., np.inf, 3.]), ([.1, .2], [1., 2.])])
def test_oracle_rejects_invalid_fitting_observations(h, target):
    with pytest.raises(ValueError):
        fit_coefficients(h, target, "scalar", p=1, q=3, kappa=.002, rate=2.)


def test_budget_callback_stops_panel_instead_of_returning_false_completion():
    def expired():
        raise TimeoutError("declared panel budget exhausted")
    with pytest.raises(TimeoutError, match="declared panel budget exhausted"):
        run({"smoke": True}, expired)


def test_smoke_preserves_negative_findings_and_strict_json_evidence():
    report = run({"smoke": True}, lambda: None)
    assert report["status"] == "COMPLETED"
    assert report["summary"]["correctness_failures"] == 0
    assert report["summary"]["trained_models"] == 0
    assert len({row["case_id"] for row in report["rows"]}) == len(report["rows"])
    assert [row["case_id"] for row in report["rows"]] == plan({"smoke": True})["expected_case_ids"]
    assert any(row["outcome"] == "EXPECTED_LIMITATION" for row in report["rows"])
    results = {row["case_id"]: row for row in report["rows"]}
    assert results["r1-p1-q3-output_phi"]["metrics"]["test_max_absolute_error"] > results["r1-p1-q3-scalar"]["metrics"]["test_max_absolute_error"]
    json.dumps(report, allow_nan=False)
