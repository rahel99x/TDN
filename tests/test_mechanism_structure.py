from __future__ import annotations

import json
import math

import numpy as np
import pytest

from tdn.analysis.mechanisms.structure import (
    additive_axis_projection, compatible_projection, filtered_square,
    heun_step, plan, rank_projection, run, scalar_controller,
)


def test_squared_harmonic_alias_and_filtered_physical_target():
    x = np.arange(16) / 16
    high = np.cos(2 * np.pi * 6 * x + 0.17)
    # cos(6x)^2 has a physical k=12 component, which aliases to k=4 here.
    assert np.sqrt(np.mean((high ** 2 - 0.5) ** 2)) == pytest.approx(math.sqrt(1 / 8))
    np.testing.assert_allclose(filtered_square(high), 0.5, atol=2e-15)
    low = np.cos(2 * np.pi * 3 * x + 0.17)
    np.testing.assert_allclose(filtered_square(low), 0.5 + 0.5 * np.cos(2 * np.pi * 6 * x + 0.34), atol=3e-15)


def test_factorized_linear_classes_are_distinct():
    k = np.arange(-3, 4, dtype=float)
    product = np.outer(k, k)
    # Every axis-wise mean vanishes, so the best additive approximation is 0.
    np.testing.assert_allclose(additive_axis_projection(product), 0, atol=1e-15)
    np.testing.assert_allclose(rank_projection(product, 1), product, atol=5e-15)
    diagonal = np.eye(7)
    # Eckart--Young: six singular values of 1 remain after rank-one projection.
    assert np.linalg.norm(diagonal - rank_projection(diagonal, 1)) == pytest.approx(math.sqrt(6))


def test_projection_removes_a_gradient_and_keeps_solenoidal_shear():
    n = 12
    x, y = np.meshgrid(np.arange(n) / n, np.arange(n) / n, indexing="ij")
    potential = np.sin(2 * np.pi * x) * np.cos(4 * np.pi * y)
    gradient = np.stack((n * (np.roll(potential, -1, axis=0) - potential),
                         n * (np.roll(potential, -1, axis=1) - potential)))
    # A velocity in x depending only on y (and conversely) has zero divergence.
    shear = np.stack((np.cos(2 * np.pi * y), np.sin(2 * np.pi * x)))
    np.testing.assert_allclose(compatible_projection(gradient + shear), shear, atol=8e-15)
    constant = np.ones((2, n, n))
    np.testing.assert_allclose(compatible_projection(constant), constant, atol=1e-15)


def test_step_doubling_approaches_known_local_error_without_bias_detection():
    ratios = []
    errors = []
    for h in (0.04, 0.02, 0.01):
        coarse = heun_step(1, h, 2)
        fine = heun_step(heun_step(1, h / 2, 2), h / 2, 2)
        error = abs(fine - math.exp(-2 * h))
        errors.append(error)
        ratios.append(abs(fine - coarse) / (3 * error))
    assert abs(ratios[-1] - 1) < abs(ratios[0] - 1)
    assert 2.9 < math.log2(errors[-2] / errors[-1]) < 3.1


def test_controller_counts_rejections_and_lands_on_requested_endpoint():
    calls = []
    got = scalar_controller(0.002, lambda: calls.append(True))
    assert got["rejected_steps"] > 0
    assert got["attempted_steps"] == got["accepted_steps"] + got["rejected_steps"] == len(calls)
    assert got["rhs_evaluations"] == 6 * got["attempted_steps"]
    assert got["final_time"] == 1.0
    assert got["absolute_final_error"] < 0.002
    assert got["maximum_accepted_estimate_ratio"] <= 1
    tighter = scalar_controller(0.0002, lambda: None)
    assert tighter["absolute_final_error"] < got["absolute_final_error"]
    assert tighter["rhs_evaluations"] > got["rhs_evaluations"]


def test_budget_cancellation_is_not_swallowed():
    class BudgetExceeded(RuntimeError):
        pass
    def stop():
        raise BudgetExceeded("stop")
    with pytest.raises(BudgetExceeded):
        run({"smoke": True}, stop)
    with pytest.raises(BudgetExceeded):
        scalar_controller(0.002, stop)


@pytest.fixture(scope="module")
def report():
    return run({"smoke": True, "tolerance": 0.002}, lambda: None)


def test_report_contains_explicit_wrong_equation_negative_controls(report):
    rows = {row["case_id"]: row for row in report["rows"]}
    for name in ("identity", "diffusion_only"):
        row = rows[f"composition-0.12-{name}"]
        assert row["outcome"] == "EXPECTED_LIMITATION"
        assert row["metrics"]["composition_rms"] < 1e-13
        assert row["metrics"]["true_rms_error"] > 0.002
        assert row["metrics"]["pde_residual_rms"] > 0.002
    wrong = rows["estimator-shared-generator-bias"]
    assert wrong["metrics"]["tolerance_met_by_estimator"]
    assert not wrong["metrics"]["true_tolerance_met"]


def test_report_preserves_source_balance_and_all_mechanisms(report):
    assert report["panel"] == "structure" and report["status"] == "COMPLETED"
    assert len({row["case_id"] for row in report["rows"]}) == len(report["rows"])
    assert all(row["outcome"] != "FAIL" for row in report["rows"])
    assert {row["mechanism"] for row in report["rows"]} == {
        "alias_control", "spectral_factorization", "composition_and_pde",
        "equation_specific_structure", "adaptive_stepping"}
    source = next(row for row in report["rows"] if row["case_id"] == "balance-reaction-diffusion")
    assert source["metrics"]["reaction_mean_gain"] > 0.1
    assert source["metrics"]["integrated_source_error"] < 1e-13
    assert [row["case_id"] for row in report["rows"]] == plan({"smoke": True})["expected_case_ids"]
    json.dumps(report, allow_nan=False)


def test_frozen_plan_includes_full_cases_without_solving(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("plan must not run a numerical probe")
    monkeypatch.setattr(np.linalg, "svd", forbidden)
    full = plan({"smoke": False})
    smoke = plan({"smoke": True})
    assert len(full["expected_case_ids"]) == 38
    assert len(smoke["expected_case_ids"]) == 32
    assert set(smoke["expected_case_ids"]) < set(full["expected_case_ids"])
    assert full["composition_and_pde"]["reference_counts"] == [32, 64, 128]
    json.dumps(full, allow_nan=False)


@pytest.mark.parametrize("tolerance", [0, -1, math.inf, math.nan])
def test_invalid_tolerance_rejected(tolerance):
    with pytest.raises(ValueError):
        run({"tolerance": tolerance}, lambda: None)
