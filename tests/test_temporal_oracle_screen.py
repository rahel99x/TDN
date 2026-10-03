"""Temporal screening must identify fitting information and preserve hold-outs."""
from __future__ import annotations

import json

import numpy as np
import pytest

from tdn.analysis.convergence import (_psi_numpy, temporal_oracle_fits,
                                       temporal_screen_horizons)


def _known_rate_targets(horizons, rates=(2., 25.)):
    amplitudes = np.array([[1., 2.], [3., 4.]])
    h = np.asarray(horizons)
    return h[:, None]**3 * (_psi_numpy(h[:, None] * np.asarray(rates)) @ amplitudes)


def test_expanded_fitting_grid_preserves_original_independent_holdouts():
    original = [.04, .08, .16, .32]
    fitting, heldout = temporal_screen_horizons(original, modes=4)
    assert len(fitting) == 10 > 4
    assert set(original) <= set(fitting)
    assert heldout == pytest.approx([.06, .12, .24, .4])
    assert not np.isclose(np.asarray(fitting)[:, None], heldout, rtol=1e-12, atol=0).any()
    assert fitting == sorted(fitting)
    assert temporal_screen_horizons(list(reversed(original)), modes=4) == (fitting, heldout)


def test_grid_expansion_stays_overdetermined_for_larger_bounded_dictionary():
    fitting, heldout = temporal_screen_horizons([.04, .08, .16], modes=16)
    assert len(fitting) > 16
    assert not np.isclose(np.asarray(fitting)[:, None], heldout, rtol=1e-12, atol=0).any()


def test_overdetermined_known_rates_require_real_optimization_and_improve_holdout():
    fitting = np.linspace(.02, 1.2, 16)
    heldout = np.array([.075, .215, .455, .775, 1.35])
    result = temporal_oracle_fits(fitting, _known_rate_targets(fitting), heldout,
                                  _known_rate_targets(heldout), fixed_rates=(.5, 8.))
    learned = result["models"]["learned_rate_oracle"]
    assert learned["rate_fit_structurally_identifiable"]
    assert learned["rate_fit_identifiable"]
    assert learned["informative_rate_fit"]
    assert learned["optimizer_success"]
    assert 1 < learned["rate_function_evaluations"] <= 40
    assert learned["rate_residual_evaluations"] > learned["rate_function_evaluations"]
    assert learned["rate_residual_evaluations"] <= learned["rate_residual_evaluation_upper_bound"]
    assert learned["rates_reference_time_units"] == pytest.approx([2., 25.], rel=1e-5)
    assert max(learned["heldout_absolute_rms"]) < 1e-9
    assert max(learned["heldout_absolute_rms"]) < .01 * max(result["models"]["fixed_rate"]["heldout_absolute_rms"])
    assert learned["final_normalized_fit_cost"] < learned["initial_normalized_fit_cost"]


def test_rate_optimization_does_not_use_heldout_labels():
    fitting = np.linspace(.02, 1.2, 16)
    heldout = np.array([.075, .215, .455, .775, 1.35])
    targets = _known_rate_targets(heldout)
    original = temporal_oracle_fits(fitting, _known_rate_targets(fitting), heldout,
                                    targets, fixed_rates=(.5, 8.))
    changed = temporal_oracle_fits(fitting, _known_rate_targets(fitting), heldout,
                                   targets + .05, fixed_rates=(.5, 8.))
    for key in ("rates_reference_time_units", "rate_function_evaluations", "final_normalized_fit_cost"):
        assert original["models"]["learned_rate_oracle"][key] == changed["models"]["learned_rate_oracle"][key]
    assert original["models"]["learned_rate_oracle"]["heldout_absolute_rms"] != changed["models"]["learned_rate_oracle"]["heldout_absolute_rms"]


@pytest.mark.parametrize("fitting", [[.04, .08, .16], [.04, .08, .16, .32]])
def test_saturated_or_underdetermined_amplitudes_skip_rate_optimization(fitting):
    heldout = np.array([.06, .12, .24, .4])
    h = np.array(fitting)
    targets = (h**3 * (1 - 2 * h))[:, None]
    heldout_targets = (heldout**3 * (1 - 2 * heldout))[:, None]
    result = temporal_oracle_fits(h, targets, heldout, heldout_targets)
    learned = result["models"]["learned_rate_oracle"]
    assert not learned["amplitude_fit_overdetermined"]
    assert not learned["rate_fit_structurally_identifiable"]
    assert not learned["informative_rate_fit"]
    assert not learned["optimizer_success"]
    assert learned["rate_function_evaluations"] == 0
    assert learned["rates_reference_time_units"] == [.1, 1., 10., 100.]
    assert learned["heldout_absolute_rms"] == result["models"]["fixed_rate"]["heldout_absolute_rms"]


def test_zero_response_does_not_claim_informative_rate_search():
    fitting = np.linspace(.02, 1.2, 12)
    heldout = [.075, .215, .455]
    result = temporal_oracle_fits(fitting, np.zeros((12, 2)), heldout, np.zeros((3, 2)), fixed_rates=(.5, 8.))
    learned = result["models"]["learned_rate_oracle"]
    assert not learned["informative_rate_fit"]
    assert not learned["rate_fit_identifiable"]
    assert learned["rate_function_evaluations"] == 1


@pytest.mark.parametrize("fitting,heldout", [
    ([.04, .04, .16], [.06, .12]),
    ([.04, .08, .16], [.06, .06]),
    ([.04, .08, .16], [.08, .12]),
    ([.04, .08, float("nan")], [.06, .12]),
    ([.04, .08, .16], [.06, float("inf")]),
    ([.04, .08, .16], []),
])
def test_oracle_rejects_duplicate_nonfinite_or_overlapping_horizons(fitting, heldout):
    with pytest.raises(ValueError):
        temporal_oracle_fits(fitting, np.ones((len(fitting), 1)), heldout, np.ones((len(heldout), 1)))


@pytest.mark.parametrize("rates", [[float("nan")], [float("inf")], [-.1], [10001.], [], [1.] * 17, [.1, .1]])
def test_rate_dictionary_has_explicit_finite_mode_and_rate_bounds(rates):
    h = np.array([.04, .08, .16, .32, .64])
    heldout = [.06, .12]
    with pytest.raises(ValueError):
        temporal_oracle_fits(h, np.ones((len(h), 1)), heldout, np.ones((len(heldout), 1)), fixed_rates=rates)


@pytest.mark.parametrize("modes", [0, 17, 1.5, True, float("inf"), float("nan")])
def test_expansion_rejects_invalid_mode_count(modes):
    with pytest.raises(ValueError):
        temporal_screen_horizons([.04, .08, .16], modes=modes)


@pytest.mark.parametrize("targets,heldout_targets", [
    (np.ones((4, 2)), np.ones((2, 3))),
    (np.full((4, 1), np.nan), np.ones((2, 1))),
    (np.ones((4, 1)), np.full((2, 1), np.inf)),
])
def test_oracle_rejects_misaligned_or_nonfinite_teacher_targets(targets, heldout_targets):
    with pytest.raises(ValueError):
        temporal_oracle_fits([.04, .08, .16, .32], targets, [.06, .12], heldout_targets)


def test_boundary_rate_dictionary_produces_finite_strict_json_report():
    fitting = np.linspace(.02, .9, 10)
    heldout = np.array([.025, .115, .255])
    result = temporal_oracle_fits(fitting, (fitting**3)[:, None], heldout,
                                  (heldout**3)[:, None], fixed_rates=(0., 1e4))
    learned = result["models"]["learned_rate_oracle"]
    assert all(1e-4 <= rate <= 1e4 for rate in learned["rates_reference_time_units"])
    assert learned["rate_residual_evaluations"] <= learned["rate_residual_evaluation_upper_bound"]
    assert not learned["informative_rate_fit"]
    json.dumps(result, allow_nan=False)
