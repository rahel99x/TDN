"""Physical invariants and retained counterexamples for research clock models."""
from __future__ import annotations

import math
from unittest.mock import patch

import pytest
import torch

from tdn.numerics.splitting import split_step
from tdn.numerics.subflows import reaction_step
from tdn.numerics.types import Equation, Geometry
from tdn.research.reaction import (
    AdditiveClockTDN,
    HybridReactionTDN,
    PolynomialClockTDN,
    ReactionClockTDN,
    capacity_update,
    interior_gate,
    reaction_clock_step,
)


FAMILIES = (ReactionClockTDN, AdditiveClockTDN, PolynomialClockTDN, HybridReactionTDN)


def _nonzero_heads(model):
    with torch.no_grad():
        model.encoder.head.bias.copy_(torch.linspace(-0.17, 0.31,
            model.encoder.head.bias.numel(), dtype=model.encoder.head.bias.dtype))
    return model


def _state(dtype=torch.float64):
    return torch.tensor([[[[0.0, .16, .24, .7], [.98, .8, .44, .3],
                           [.08, .5, .2, 1.0], [.9, .18, .73, .41]]]], dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_signed_clock_bounds_endpoints_and_zero_shift_identity(dtype):
    state = torch.tensor([0, 2**-20, .01, .3, .7, 1 - 2**-20, 1], dtype=dtype)
    clocks = torch.tensor([-1000, -20, -2, 0, .2, 20, 1000], dtype=dtype)
    answer = reaction_clock_step(state[None], clocks[:, None])
    assert torch.isfinite(answer).all()
    assert ((answer >= 0) & (answer <= 1)).all()
    assert torch.equal(answer[:, 0], torch.zeros_like(clocks))
    assert torch.equal(answer[:, -1], torch.ones_like(clocks))
    for h in (0, .01, .1, 3, 1000):
        clock = torch.tensor(2 * h, dtype=dtype)
        actual = reaction_clock_step(state, clock + torch.zeros_like(state))
        expected = reaction_step(state, h, Equation(.1, 2))
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_heads_exactly_reproduce_split_and_support_batch_h(family, dtype):
    model = family().to(dtype=dtype)
    geometry, equation = Geometry((4, 4), (1, 1)), Equation(.03, 6)
    state = torch.cat([_state(dtype), 1 - _state(dtype)], dim=0)
    horizons = torch.tensor([.01, .4], dtype=dtype)
    assert torch.equal(model(state, horizons, equation, geometry),
                       split_step(state, horizons, equation, geometry))
    for index in range(2):
        assert torch.equal(model(state[index:index + 1], horizons[index], equation, geometry),
                           model(state, horizons, equation, geometry)[index:index + 1])


@pytest.mark.parametrize("family", FAMILIES)
def test_commuting_limits_hold_with_trained_nonzero_heads(family):
    model = _nonzero_heads(family().double())
    geometry, state = Geometry((4, 4), (1, 1)), _state()
    for equation in (Equation(0, 2), Equation(.03, 0)):
        assert torch.equal(model(state, .2, equation, geometry),
                           split_step(state, .2, equation, geometry))
    for value in (0.0, .27, 1.0):
        constant = torch.full_like(state, value)
        equation = Equation(.03, 2)
        assert torch.equal(model(constant, .2, equation, geometry),
                           split_step(constant, .2, equation, geometry))


@pytest.mark.parametrize("family", [ReactionClockTDN, HybridReactionTDN])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_learned_maps_bound_actual_diffused_fields(family, dtype):
    model = _nonzero_heads(family().to(dtype=dtype))
    geometry, equation = Geometry((4, 4), (1, 1)), Equation(.03, 6)
    for h in (.001, .1, .4, 10):
        result = model(_state(dtype), h, equation, geometry)
        assert torch.isfinite(result).all()
        assert ((result >= 0) & (result <= 1)).all()


@pytest.mark.parametrize("family", FAMILIES)
def test_zero_initialized_heads_receive_live_gradients(family):
    model = family().double()
    # Include a transition region so the hybrid's boundary heads participate.
    state = _state().clone()
    state[0, 0, 0, 0] = .025
    answer = model(state, .2, Equation(.03, 6), Geometry((4, 4), (1, 1)))
    answer.square().sum().backward()
    gradient = model.encoder.head.bias.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert (gradient.abs() > 1e-12).all()


@pytest.mark.parametrize("family", FAMILIES)
def test_each_complete_forward_uses_one_fft_pair(family):
    model = _nonzero_heads(family().double())
    with patch("torch.fft.fftn", wraps=torch.fft.fftn) as forward, \
         patch("torch.fft.ifftn", wraps=torch.fft.ifftn) as inverse:
        model(_state(), .2, Equation(.03, 6), Geometry((4, 4), (1, 1)))
    assert forward.call_count == inverse.call_count == 1


def test_signed_flow_first_second_derivatives_and_jvp():
    state = torch.tensor([.1, .4, .9], dtype=torch.float64, requires_grad=True)
    clock = torch.tensor([-2.0, 0.0, 1.3], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(reaction_clock_step, (state, clock))
    assert torch.autograd.gradgradcheck(reaction_clock_step, (state, clock))
    result, tangent = torch.func.jvp(reaction_clock_step, (state, clock),
                                    (torch.zeros_like(state), torch.ones_like(clock)))
    assert torch.allclose(tangent, result * (1 - result), atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize("family", [ReactionClockTDN, HybridReactionTDN])
def test_complete_model_time_and_state_gradgrad_include_physical_flows(family):
    model = _nonzero_heads(family(ndim=1).double())
    geometry, equation = Geometry((3,), (1,)), Equation(.03, 2)
    # Strictly inside both gate transitions; no non-C2 gate/sign switch is
    # mistaken for a twice-differentiable point in this local derivative audit.
    state = torch.tensor([[[.02, .55, .975]]], dtype=torch.float64, requires_grad=True)
    horizon = torch.tensor(.17, dtype=torch.float64, requires_grad=True)
    function = lambda u, h: model(u, h, equation, geometry)
    assert torch.autograd.gradcheck(function, (state, horizon))
    assert torch.autograd.gradgradcheck(function, (state, horizon))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_bounded_clock_recovers_equilibrium_but_additive_control_need_not(dtype):
    geometry, equation = Geometry((4, 4), (1, 1)), Equation(.03, 2)
    clock = _nonzero_heads(ReactionClockTDN().to(dtype=dtype))
    additive = AdditiveClockTDN().to(dtype=dtype)
    additive.load_state_dict(clock.state_dict())
    state = _state(dtype)
    assert torch.equal(clock(state, 100, equation, geometry), torch.ones_like(state))
    assert (additive(state, 100, equation, geometry) > 1).any()


def _time_coefficient(function, degree):
    h = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    value = function(h)
    for _ in range(degree):
        value = torch.autograd.grad(value, h, create_graph=True)[0]
    return value.detach() / math.factorial(degree)


def test_clock_boundary_cubic_obstruction_is_retained_not_masked():
    # Independently derive the coupled ODE's cubic term on a two-site grid.
    model = _nonzero_heads(ReactionClockTDN(ndim=1).double())
    geometry, equation = Geometry((2,), (1,)), Equation(.05, 2)
    state = torch.tensor([[[0.0, .7]]], dtype=torch.float64)
    diffusion = torch.tensor([[-.4, .4], [.4, -.4]], dtype=torch.float64)
    u = state.flatten()
    field = diffusion @ u + 2 * u * (1 - u)
    jacobian = diffusion + torch.diag(2 * (1 - 2 * u))
    true_cubic = ((jacobian @ jacobian @ field - 4 * field.square()) / 6)[0]
    base = lambda h: split_step(state, h, equation, geometry)[0, 0, 0]
    correction = lambda h: model(state, h, equation, geometry)[0, 0, 0] - base(h)
    leading_defect = true_cubic - _time_coefficient(base, 3)
    assert float(leading_defect) == pytest.approx(.06533333333333333, abs=1e-12)
    assert abs(float(_time_coefficient(correction, 3))) < 1e-12
    assert abs(float(_time_coefficient(correction, 4))) > 1e-6
    # In the interior the same model does have the declared cubic coefficient.
    interior = torch.tensor([[[.35, .7]]], dtype=torch.float64)
    amplitude = model.amplitudes(interior, equation, geometry)[0, 0, 0].detach()
    interior_correction = lambda h: (model(interior, h, equation, geometry)
                                     - split_step(interior, h, equation, geometry))[0, 0, 0]
    assert float(_time_coefficient(interior_correction, 3)) == pytest.approx(
        float(.35 * .65 * amplitude * 2**3), rel=1e-10, abs=1e-12)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_capacity_bounds_identity_and_live_zero_gradient(dtype):
    state = torch.tensor([0, 2**-20, .01, .2, .8, 1 - 2**-20, 1], dtype=dtype)
    increment = torch.tensor([-1e20, -1, -.01, 0, .01, 1, 1e20], dtype=dtype)
    answer = capacity_update(state[None], increment[:, None])
    assert torch.isfinite(answer).all()
    assert ((answer >= 0) & (answer <= 1)).all()
    assert torch.equal(capacity_update(state, torch.zeros_like(state)), state)
    interior = state[1:-1]
    zero = torch.zeros_like(interior, requires_grad=True)
    gradient = torch.autograd.grad(capacity_update(interior, zero).sum(), zero)[0]
    assert torch.allclose(gradient, torch.ones_like(gradient),
                          atol=2 * torch.finfo(dtype).eps, rtol=0)


@pytest.mark.parametrize("coefficient", [.06533333333333333, -.2])
def test_capacity_branch_recovers_boundary_cubic_when_capacity_is_linear(coefficient):
    geometry, equation = Geometry((2,), (1,)), Equation(.05, 2)
    state = torch.tensor([[[0.0, .7]]], dtype=torch.float64)
    ratios, errors = [], []
    for h in (.04, .02, .01, .005):
        base = split_step(state, h, equation, geometry)[0, 0, 0]
        increment = torch.tensor(coefficient * h**3, dtype=torch.float64)
        actual = capacity_update(base, increment) - base
        ratios.append(float(actual / h**3))
        errors.append(abs(float(actual - increment)))
    assert abs(ratios[-1] - coefficient) < abs(ratios[0] - coefficient)
    assert ratios[-1] == pytest.approx(coefficient, rel=5e-5)
    assert math.log(errors[-2] / errors[-1], 2) > 4.8


def test_capacity_cubic_capacity_counterexample_remains_explicit():
    # Capacity=h^3 changes the leading coefficient: desired zero becomes h^3/2.
    for h in (.1, .05, .025):
        value = torch.tensor(h**3, dtype=torch.float64)
        assert float(capacity_update(value, -value) / h**3) == pytest.approx(.5)


def test_fixed_interior_gate_is_convex_and_has_declared_support():
    state = torch.tensor([0, .025, .05, .2, .95, .975, 1], dtype=torch.float64,
                         requires_grad=True)
    gate = interior_gate(state)
    assert torch.allclose(gate, torch.tensor([0, .5, 1, 1, 1, .5, 0], dtype=state.dtype))
    derivative = torch.autograd.grad(gate.sum(), state)[0]
    assert torch.equal(derivative[[0, 2, 3, 4, 6]], torch.zeros(5, dtype=state.dtype))


@pytest.mark.parametrize("kwargs", [{"amplitude_cap": 0}, {"t_ref": float("nan")},
                                    {"U_ref": -1}, {"epsilon": .5}])
def test_invalid_fixed_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        HybridReactionTDN(**kwargs)
