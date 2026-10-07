import pytest
import torch

from tdn.analysis.agenda.policy import (ESTIMATORS, accepted, fit_envelope, refined_schedule,
                                        policy_schedule, _defect_proxy)
from tdn.numerics import Equation, Geometry
from tdn.numerics.subflows import reaction_step


def test_zero_agreement_cannot_approve_known_large_error():
    row = {"estimates": {"step_doubling": {"rms": 0., "max": 0.}}, "upper_rms": .03, "upper_max": .07}
    envelope = fit_envelope([row], "step_doubling", safety_factor=2.)
    assert not accepted({"rms": 0., "max": 0.}, envelope, .0002, .0002)


def test_acceptance_requires_both_independent_norms():
    envelope = {"rms": 2., "max": 4.}
    assert accepted({"rms": .00001, "max": .00001}, envelope, .0002, .0002)
    assert not accepted({"rms": .00001, "max": .001}, envelope, .0002, .0002)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_calibration_envelope_covers_every_observed_upper_error(estimator):
    rows = [{"estimates": {estimator: {"rms": x, "max": 3 * x}}, "upper_rms": 4 * x, "upper_max": 8 * x} for x in (.001, .002)]
    envelope = fit_envelope(rows, estimator, safety_factor=2.)
    assert envelope["rms"] >= 8
    assert envelope["max"] >= 16 / 3


def test_fixed_policy_schedules_and_refinements_preserve_interval():
    for h in (.27, .81):
        for step in (.09, .045, .0225, .01125):
            schedule = policy_schedule(h, step)
            assert sum(schedule) == pytest.approx(h)
            assert max(schedule) <= step * (1 + 1e-12)
            assert sum(refined_schedule(schedule, 2)) == pytest.approx(h)


def test_actual_defect_proxy_vanishes_for_exact_homogeneous_reaction_flow():
    class Exact(torch.nn.Module):
        def forward(self, u, h, equation, geometry):
            return reaction_step(u, h, equation)
    u = torch.full((1, 1, 8, 8), .4, dtype=torch.float64)
    with torch.no_grad():
        estimate = _defect_proxy(Exact(), u, [.03, .07], Equation(.03, 2.), Geometry((8, 8), (1., 1.)))
    assert max(estimate.values()) < 2e-14


def test_actual_defect_proxy_detects_a_consistently_wrong_identity_solver():
    class Wrong(torch.nn.Module):
        def forward(self, u, h, equation, geometry):
            return u + h * 0
    u = torch.full((1, 1, 8, 8), .4, dtype=torch.float64)
    estimate = _defect_proxy(Wrong(), u, [.03, .07], Equation(0., 2.), Geometry((8, 8), (1., 1.)))
    assert estimate["rms"] == pytest.approx(.048)
    assert estimate["max"] == pytest.approx(.048)
