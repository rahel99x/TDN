"""Scientific-report safeguards, discrete sensitivity, and independent controls."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tdn.analysis.convergence import fit_order, temporal_oracle_fits, tier_a_audit
from tdn.analysis.influence import derivative_audit, same_observation_audit
from tdn.analysis.profiling import measure
from tdn.analysis.statistics import (break_even_uses, failure_upper_bound,
                                     paired_parent_comparison, parent_summary)
from tdn.analysis.workflow import _aggregate, _rollout, step_schedule
from tdn.numerics import Equation, Geometry


def test_order_fit_requires_three_noise_resolved_points():
    h = np.array([.2, .1, .05, .025])
    result = fit_order(h, h**3, noise_floor=1e-8)
    assert result["valid"]
    assert result["slope"] == pytest.approx(3.)
    assert result["raw_errors"] == pytest.approx(h**3)
    floor = fit_order(h, h**3, noise_floor=2e-4)
    assert not floor["valid"]
    assert sum(floor["fit_mask"]) == 1
    with pytest.raises(ValueError):
        fit_order(h, h**3, min_points=2)


def test_noncommuting_exact_local_and_global_orders():
    result = tier_a_audit()
    assert result["passed"]
    assert result["commutator_norm"] > 0
    assert result["fits"]["split_global"]["slope"] == pytest.approx(2., abs=.03)
    assert result["fits"]["anchored_global"]["slope"] == pytest.approx(3., abs=.06)
    assert result["limits"]["commuting"]["passed"]


def test_parent_failure_denominator_and_exact_zero_failure_bound():
    expected = 1 - .05**(1 / 16)
    assert failure_upper_bound(0, 16) == pytest.approx(expected)
    assert expected > .15  # Sixteen successes cannot certify a one-percent rate.
    summary = parent_summary([.1, .2, None, float("nan")], bootstrap_samples=50)
    assert summary["independent_parents"] == 4
    assert summary["failed_parents"] == 2
    assert summary["all_parent_mean"] is None
    assert summary["successful_parent_mean"] == pytest.approx(.15)
    assert summary["failure_probability_upper_bound"] > .5
    assert failure_upper_bound(4, 4) == 1.
    with pytest.raises(ValueError):
        failure_upper_bound(0, 0)


def test_paired_statistics_preserve_failures_and_training_amortization():
    failed = paired_parent_comparison([1., float("nan")], [.9, .8], bootstrap_samples=20)
    assert not failed["available"]
    measured = paired_parent_comparison([1., 2.], [.8, 1.8], bootstrap_samples=20)
    assert measured["candidate_minus_baseline_mean"] == pytest.approx(-.2)
    assert measured["paired_parent_bootstrap_interval"] == pytest.approx([-.2, -.2])
    assert break_even_uses(10., 90., 5., 3., 2.) == 105
    assert break_even_uses(10., 90., 5., 2., 3.) is None


def test_multiple_anchors_and_horizons_do_not_inflate_parent_count():
    rows = [{"parent_id": parent, "method": "split", "error": .01, "failed": False,
             "performance": {"wall_seconds_median": .1}}
            for parent in ("p1", "p2") for _ in range(6)]
    rows[-1]["failed"] = True
    config = {"seed": 0, "validation": {"bootstrap_samples": 20},
              "horizons": {"mixed_factors": [.5, 1., 1.5]}}
    summary = _aggregate(rows, config)
    assert summary["independent_parents"] == 2
    assert summary["failed_parents"] == 1
    assert summary["all_parent_mean"] is None


def test_exact_final_step_and_unequal_steps():
    steps = step_schedule(1., .3)
    assert steps == pytest.approx([.3, .3, .3, .1])
    mixed = step_schedule(1., .2, (.5, 1., 1.5))
    assert sum(mixed) == pytest.approx(1.)
    assert mixed[:3] == pytest.approx([.1, .2, .3])
    with pytest.raises(ValueError):
        step_schedule(1., .2, (0.,))


def test_hidden_feature_pair_and_matrix_free_derivatives():
    geometry, equation = Geometry((8,), (1.,)), Equation(.01, 2.)
    u = (.45 + .08 * torch.sin(2 * math.pi * torch.arange(8, dtype=torch.float64) / 8))[None, None]
    information = same_observation_audit(u, equation, geometry, .1, substeps=16)
    assert information["feature_equality_exact"]
    assert information["feature_difference_max"] == 0.
    assert information["deterministic_local_defect_worst_case_error_lower_bound"] >= 0.
    assert not information["locality_certified"]
    derivatives = derivative_audit(u, equation, geometry, .1, substeps=16)
    assert derivatives["passed"]
    assert len(derivatives["maps"]["splitting_defect"]["finite_difference_sweep"]) >= 3


def test_temporal_fit_reports_heldout_oracle_condition_and_cancellation():
    horizons = np.array([.04, .08, .16, .32])
    heldout = np.array([.06, .12, .24, .4])
    # Exactly polynomial response, fitting oracle amplitudes rather than state
    # features. A strong control must recover it on unseen horizons.
    defects = (horizons**3 * (1 - 2 * horizons))[:, None]
    heldout_defects = (heldout**3 * (1 - 2 * heldout))[:, None]
    result = temporal_oracle_fits(horizons, defects, heldout, heldout_defects)
    assert "not deployable" in result["scope"]
    assert max(result["models"]["polynomial"]["heldout_absolute_rms"]) < 1e-14
    for method in result["models"].values():
        assert len(method["heldout_absolute_rms"]) == len(heldout)
        assert math.isfinite(method["design_condition_number"])
        assert method["cancellation_ratio_max"] >= 0.


def test_measurement_exposes_first_use_warmup_and_cpu_memory_scope():
    calls = []
    output, report = measure(lambda: calls.append(1) or torch.ones(2), warmup=1, repeats=2)
    assert len(calls) == 4
    assert len(report["wall_seconds_raw"]) == 2
    assert report["cuda_event_seconds_raw"] is None
    assert report["peak_allocated_bytes"] is None
    assert report["host_process_peak_rss_bytes"] > 0
    assert output.shape == (2,)


def test_rollout_failure_does_not_clip_or_hide_physical_state():
    geometry, equation = Geometry((8,), (1.,)), Equation(.01, 2.)
    state = torch.full((1, 1, 8), .45, dtype=torch.float64)
    final, cost = _rollout(state, [.03, .02], equation, geometry, "richardson_split")
    assert cost["split_calls"] == 6
    assert cost["simulated_time"] == pytest.approx(.05)
    assert cost["fallback_steps"] == 0
    assert torch.isfinite(final).all()
    with pytest.raises(ValueError, match="Unsupported"):
        _rollout(state, [.01], equation, geometry, "invented_method")
