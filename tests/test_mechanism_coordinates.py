"""Independent equations and retained counterexamples for coordinate probes."""
import json
import math

import pytest
import torch

from tdn.analysis.mechanisms.coordinates import (
    VARIANTS, capacity_coordinate, controlled_step, global_commuting_gate, plan, run,
)
from tdn.numerics.splitting import split_step
from tdn.numerics.types import Equation, Geometry
from tdn.research.reaction import capacity_update


def test_capacity_inverse_matches_closed_form_for_both_directions_and_endpoints():
    for base, target in ((0., .2), (.2, .1), (.2, .8), (1., .8), (1e-8, .01), (.999, .99)):
        coordinate, reachable = capacity_coordinate(base, target)
        assert reachable
        capacity = 1 - base if target > base else base
        expected = (target - base) * capacity / (capacity - abs(target - base))
        assert coordinate == pytest.approx(expected)
        reconstructed = capacity_update(torch.tensor(base, dtype=torch.float64),
                                        torch.tensor(coordinate, dtype=torch.float64))
        assert float(reconstructed) == pytest.approx(target, abs=2e-16)
    assert capacity_coordinate(.2, 1.) == (0., False)
    assert capacity_coordinate(.2, 0.) == (0., False)
    assert capacity_coordinate(.2, -.01) == (0., False)
    assert capacity_coordinate(0., 0.) == (0., True)


def test_global_gate_is_smooth_equivariant_and_sees_beyond_local_stencil():
    geometry, equation = Geometry((16,), (1.,)), Equation(.03, 2.)
    state = torch.full((1, 1, 16), .25, dtype=torch.float64)
    state[..., 5] = .75
    actual = global_commuting_gate(state, equation, geometry)
    # Two affected edges, each with squared difference .25, averaged over 16 cells.
    scaled = .03 * 2 * 16**2 * (.5 / 16)
    assert torch.allclose(actual, torch.full_like(actual, scaled / (1 + scaled)))
    assert float(actual[..., 0]) > 0
    for shift in (1, 3, 8):
        assert torch.equal(torch.roll(actual, shift, -1),
                           global_commuting_gate(torch.roll(state, shift, -1), equation, geometry))
    state.requires_grad_(True)
    assert torch.autograd.gradcheck(lambda u: global_commuting_gate(u, equation, geometry), (state,))
    for value in (0., .27, 1.):
        assert torch.count_nonzero(global_commuting_gate(torch.full_like(state, value), equation, geometry)) == 0
    for inactive in (Equation(0., 2.), Equation(.03, 0.)):
        assert torch.count_nonzero(global_commuting_gate(state, inactive, geometry)) == 0


@pytest.mark.parametrize("variant", VARIANTS)
def test_zero_coefficient_derivative_has_correct_coordinate_units(variant):
    from tdn.numerics.subflows import diffusion_step, reaction_step
    from tdn.research.common import commuting_gate
    state = torch.tensor([[[.2, .7]]], dtype=torch.float64)
    geometry, equation, h = Geometry((2,), (1.,)), Equation(.05, 2.), .13
    coefficient = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    result = controlled_step(state, h, equation, geometry, variant, (coefficient, 0.))
    derivative = torch.autograd.grad(result[0, 0, 0], coefficient)[0]
    base = split_step(state, h, equation, geometry)
    assert torch.equal(result, base)
    gate = (commuting_gate if variant.endswith("local") else global_commuting_gate)(state, equation, geometry)
    amplitude = float(gate[0, 0, 0]) * (-math.expm1(-2 * h))**3
    if variant.startswith("clock"):
        expected = amplitude * float(base[0, 0, 0] * (1 - base[0, 0, 0]))
    elif variant.startswith("pre"):
        middle = float(diffusion_step(reaction_step(state, h / 2, equation), h, equation, geometry)[0, 0, 0])
        exp_minus_q = math.exp(-h)
        expected = amplitude * exp_minus_q / (middle + (1 - middle) * exp_minus_q)**2
    else:
        expected = amplitude
    assert float(derivative) == pytest.approx(expected, rel=1e-12)


def test_run_honors_budget_before_work_and_does_not_swallow_interrupt():
    class StopAudit(Exception):
        pass
    calls = 0
    def budget():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise StopAudit
    with pytest.raises(StopAudit):
        run({"smoke": True}, budget)
    assert calls == 3


@pytest.fixture(scope="module")
def smoke_report():
    return run({"smoke": True, "tolerance": .002}, lambda: None)


def test_smoke_reference_endpoints_preserve_known_obstruction_and_oracle_context(smoke_report):
    rows = smoke_report["rows"]
    cubic = {row["variant"]: row["metrics"] for row in rows if row["test"] == "analytic_boundary_cubic"}
    # Independent coupled two-site Taylor coefficient minus the Strang coefficient.
    for metrics in cubic.values():
        assert metrics["required_defect_cubic"] == pytest.approx(49 / 750, abs=1e-13)
    assert cubic["clock_local"]["prescribed_correction_cubic"] == 0
    assert cubic["clock_global"]["prescribed_correction_cubic"] == 0
    assert cubic["post_capacity_global"]["prescribed_correction_cubic"] > 0
    assert cubic["pre_capacity_global"]["prescribed_correction_cubic"] == pytest.approx(
        cubic["post_capacity_global"]["prescribed_correction_cubic"])
    remote = [row for row in rows if row["test"] == "locally_identical_remote_bump"]
    assert remote
    for row in remote:
        metrics = row["metrics"]
        assert metrics["reference_accepted"]
        assert metrics["initial_radius_two_patch_max_difference"] == 0
        assert metrics["local_gate"] == 0
        assert metrics["global_gate"] > 0
        assert metrics["physical_middle_difference"] > 0
        assert metrics["resolved_nonzero_required_clock"]
        assert metrics["required_clock_coordinate"] > metrics["required_clock_uncertainty_bound"]
        assert metrics["required_post_capacity_coordinate"] != 0
        assert metrics["required_pre_capacity_coordinate"] != 0
        if row["variant"].endswith("local"):
            assert metrics["receiver_error"] == pytest.approx(abs(metrics["required_receiver_defect"]))


def test_smoke_plan_matches_every_observation_and_serializes_without_nonfinite_values(smoke_report):
    protocol = plan({"smoke": True})
    ids = [row["case_id"] for row in smoke_report["rows"]]
    assert ids == protocol["expected_case_ids"]
    assert len(ids) == len(set(ids))
    json.dumps(smoke_report, allow_nan=False)
    assert smoke_report["summary"]["correctness_failed"] == 0
    assert smoke_report["summary"]["reference_rows_rejected"] == 0
    assert smoke_report["summary"]["trained_models"] == 0
    expected_limitations = [row for row in smoke_report["rows"] if row["outcome"] == "EXPECTED_LIMITATION"]
    assert expected_limitations
    cubic_capacity = [row for row in smoke_report["rows"]
                     if row["test"] == "directional_capacity_scaling" and row["inputs"]["capacity_power"] == 3]
    for row in cubic_capacity:
        assert row["metrics"]["correction_retention"] == pytest.approx(2 / 3, abs=5e-9)
    assert any(row["variant"] == "additive" and not row["metrics"]["output_in_bounds"]
               for row in smoke_report["rows"] if row["test"] == "signed_near_bound_response")


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_invalid_tolerance_is_rejected(value):
    with pytest.raises(ValueError, match="tolerance"):
        run({"tolerance": value}, lambda: None)
