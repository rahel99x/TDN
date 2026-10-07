"""Structural constraints, complete physical mean control and derivatives."""
from __future__ import annotations

import io
import json

import pytest
import torch

from tdn.numerics import Equation, Geometry
from tdn.numerics.operators import laplacian
from tdn.numerics.splitting import split_step
from tdn.research import premix_neural
from tdn.research.consistency_neural import (
    CONSTRAINED_FAMILIES, FAMILIES, build_model, calibrate_mean,
    commutator_gate, discrete_logistic_commutator, mean_shape_update,
)
from tdn.research.reaction import capacity_update


def _state(*, dtype=torch.float64, grid=(8, 10), batch=2):
    generator = torch.Generator().manual_seed(407)
    return .12 + .74 * torch.rand(batch, 1, *grid, dtype=dtype, generator=generator)


def _arbitrary_parameters(model):
    generator = torch.Generator().manual_seed(719)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(.25 * torch.randn(parameter.shape, generator=generator,
                                               dtype=parameter.dtype, device=parameter.device))
    return model


def _fixture(grid=(8, 10)):
    return Equation(.014, 3.1), Geometry(grid, (1.3, .9))


def test_commutator_matches_independent_operator_derivative_and_sign():
    u = _state()
    equation, geometry = _fixture()
    reaction = equation.reaction_rate * u * (1 - u)
    direct = equation.reaction_rate * (1 - 2 * u) * equation.kappa * laplacian(u, geometry)
    direct = direct - equation.kappa * laplacian(reaction, geometry)
    actual = discrete_logistic_commutator(u, equation, geometry)
    torch.testing.assert_close(actual, direct, atol=2e-14, rtol=3e-13)
    assert (actual >= 0).all() and (actual > 0).any()
    # Periodicity provides a second independent global identity.
    energy = torch.zeros_like(u[:, :, :1, :1])
    for axis, dx in enumerate(geometry.dx, start=2):
        energy += 2 * equation.kappa * equation.reaction_rate * (torch.roll(u, -1, axis) - u).square().mean((-2, -1), keepdim=True) / dx**2
    torch.testing.assert_close(actual.mean((-2, -1), keepdim=True), energy, atol=2e-14, rtol=2e-14)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_commutator_is_quadratic_in_amplitude_and_soft_gate_has_quadratic_limit(dtype):
    pattern = _state(dtype=dtype) - .5
    equation, geometry = _fixture()
    low = discrete_logistic_commutator(.5 + .04 * pattern, equation, geometry)
    high = discrete_logistic_commutator(.5 + .08 * pattern, equation, geometry)
    tolerance = 2e-5 if dtype == torch.float32 else 3e-13
    torch.testing.assert_close(high, 4 * low, rtol=tolerance, atol=tolerance * float(low.max()))
    small = commutator_gate(.5 + .002 * pattern, equation, geometry, t_ref=.2)
    twice = commutator_gate(.5 + .004 * pattern, equation, geometry, t_ref=.2)
    torch.testing.assert_close(twice, 4 * small, rtol=4e-4 if dtype == torch.float32 else 1e-9,
                               atol=1e-12 if dtype == torch.float32 else 2e-16)


def test_gate_is_dimensionless_under_consistent_space_and_time_unit_changes():
    u = _state()
    equation, geometry = _fixture()
    original = commutator_gate(u, equation, geometry, t_ref=.2, U_ref=.7)
    scaled_space = Geometry(geometry.grid, tuple(5 * x for x in geometry.lengths))
    scaled_equation = Equation(25 * equation.kappa, equation.reaction_rate)
    torch.testing.assert_close(commutator_gate(u, scaled_equation, scaled_space, t_ref=.2, U_ref=.7),
                               original, atol=2e-16, rtol=3e-15)
    scaled_time = Equation(equation.kappa / 7, equation.reaction_rate / 7)
    torch.testing.assert_close(commutator_gate(u, scaled_time, geometry, t_ref=1.4, U_ref=.7),
                               original, atol=2e-16, rtol=3e-15)


@pytest.mark.parametrize("family", ("premix", "fno", "precompress"))
def test_current_controls_reuse_original_factory_exactly(family):
    torch.manual_seed(114)
    current = premix_neural.build_model(family, width=2, modes=2, t_ref=.2).double()
    torch.manual_seed(114)
    experiment = build_model(family, width=2, modes=2, t_ref=.2).double()
    assert type(experiment) is type(current)
    assert current.architecture_metadata() == experiment.architecture_metadata()
    assert current.state_dict().keys() == experiment.state_dict().keys()
    for name, value in current.state_dict().items():
        assert torch.equal(value, experiment.state_dict()[name])


@pytest.mark.parametrize("family", CONSTRAINED_FAMILIES)
def test_gated_backbone_preserves_current_parameter_initialization(family):
    base_family = family.split("_")[0]
    torch.manual_seed(552)
    original = build_model(base_family, width=2, modes=2)
    torch.manual_seed(552)
    constrained = build_model(family, width=2, modes=2)
    for name, value in original.state_dict().items():
        assert torch.equal(value, constrained.backbone.state_dict()[name])
    base_count = sum(parameter.numel() for parameter in original.parameters())
    count = sum(parameter.numel() for parameter in constrained.parameters())
    assert count == base_count + (3 if family.endswith("_moment") else 0)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_zero_heads_preserve_full_state_strang_initialization(family, dtype):
    model = build_model(family, width=2, modes=2, t_ref=.2).to(dtype=dtype)
    u = _state(dtype=dtype)
    equation, geometry = _fixture()
    expected = split_step(u, .06, equation, geometry)
    assert torch.equal(model(u, .06, equation, geometry), expected)


@pytest.mark.parametrize("family", CONSTRAINED_FAMILIES)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("limit", ("constant", "zero_reaction", "zero_diffusion", "zero_time"))
def test_arbitrary_learned_parameters_preserve_exact_commuting_limits(family, dtype, limit):
    model = _arbitrary_parameters(build_model(family, width=2, modes=2, t_ref=.2).to(dtype=dtype))
    u = _state(dtype=dtype)
    equation, geometry = _fixture()
    h = .06
    if limit == "constant":
        u = torch.cat((torch.full_like(u[:1], .13), torch.full_like(u[:1], .82)))
    elif limit == "zero_reaction":
        equation = Equation(equation.kappa, 0.)
    elif limit == "zero_diffusion":
        equation = Equation(0., equation.reaction_rate)
    else:
        h = 0.
    terms = model.correction_components(u, h, equation, geometry)
    assert torch.count_nonzero(terms["spatial_increment"]) == 0
    if family.endswith("_moment"):
        assert torch.count_nonzero(terms["mean_increment"]) == 0
    expected = split_step(u, h, equation, geometry)
    assert torch.equal(model(u, h, equation, geometry), expected)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_mean_shape_map_removes_dc_created_by_signed_capacity(dtype):
    base = torch.tensor([.04, .17, .51, .9], dtype=dtype).reshape(1, 1, 2, 2)
    spatial = torch.tensor([-.6, .2, -.1, .5], dtype=dtype).reshape_as(base)
    assert abs(float(spatial.mean())) < 2 * torch.finfo(dtype).eps
    candidate = capacity_update(base, spatial)
    assert abs(float(candidate.mean() - base.mean())) > .01
    for scalar in (0., .2, -.15):
        actual = mean_shape_update(base, scalar, spatial)
        target = capacity_update(base.mean(), torch.as_tensor(scalar, dtype=dtype))
        torch.testing.assert_close(actual.mean(), target, atol=3 * torch.finfo(dtype).eps, rtol=0.)
        assert ((actual >= 0) & (actual <= 1)).all()
        assert float((actual - base).abs().max()) > .01


@pytest.mark.parametrize("target", (0., .001, .35, .8, .999, 1.))
def test_calibration_handles_extreme_targets_and_saturated_states(target):
    base = torch.tensor([0., .1, .25, 1.], dtype=torch.float64).reshape(1, 1, 2, 2)
    state = torch.cat((base, torch.zeros_like(base), torch.ones_like(base)))
    actual = calibrate_mean(state, torch.tensor([target] * 3, dtype=torch.float64))
    assert torch.isfinite(actual).all()
    assert ((actual >= 0) & (actual <= 1)).all()
    torch.testing.assert_close(actual.mean((-2, -1)).flatten(), torch.full((3,), target, dtype=state.dtype),
                               atol=2e-16, rtol=0.)


@pytest.mark.parametrize("family", ("premix_moment", "fno_moment"))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_complete_moment_model_has_controlled_physical_mean_and_nonzero_mean_correction(family, dtype):
    model = _arbitrary_parameters(build_model(family, width=2, modes=2, t_ref=.2).to(dtype=dtype))
    with torch.no_grad():
        model.mean_head.weight.zero_()
        model.mean_head.bias.fill_(.6)
    state = _state(dtype=dtype)
    equation, geometry = _fixture()
    components = model.correction_components(state, .1, equation, geometry)
    actual = model(state, .1, equation, geometry)
    torch.testing.assert_close(actual.mean((-2, -1), keepdim=True), components["target_mean"],
                               atol=4 * torch.finfo(dtype).eps, rtol=0.)
    base_mean = components["base"].mean((-2, -1), keepdim=True)
    assert (components["target_mean"] > base_mean).all()
    assert ((actual >= 0) & (actual <= 1)).all()
    with torch.no_grad():
        model.mean_head.bias.zero_()
    torch.testing.assert_close(model(state, .1, equation, geometry).mean((-2, -1), keepdim=True),
                               base_mean, atol=4 * torch.finfo(dtype).eps, rtol=0.)


@pytest.mark.parametrize("family", CONSTRAINED_FAMILIES)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_full_map_gradients_translation_and_parent_independence(family, dtype):
    model = _arbitrary_parameters(build_model(family, width=2, modes=2, t_ref=.2).to(dtype=dtype))
    state = _state(dtype=dtype).requires_grad_()
    horizon = torch.tensor([.05, .1], dtype=dtype, requires_grad=True)
    equation, geometry = _fixture()
    actual = model(state, horizon, equation, geometry)
    assert ((actual >= 0) & (actual <= 1)).all()
    actual.square().mean().backward()
    assert torch.isfinite(state.grad).all() and state.grad.abs().sum() > 0
    assert torch.isfinite(horizon.grad).all() and (horizon.grad.abs() > 0).all()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    with torch.no_grad():
        separate = torch.cat([model(state[i:i + 1], horizon[i], equation, geometry) for i in range(2)])
        shifted = model(torch.roll(state, (2, 3), (-2, -1)), horizon, equation, geometry)
    tolerance = 2e-6 if dtype == torch.float32 else 3e-14
    torch.testing.assert_close(actual, separate, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(shifted, torch.roll(actual, (2, 3), (-2, -1)), atol=tolerance, rtol=tolerance)
    json.dumps(model.architecture_metadata(), allow_nan=False)


@pytest.mark.parametrize("family", CONSTRAINED_FAMILIES)
def test_state_time_and_mean_calibration_derivatives_match_finite_differences(family):
    model = _arbitrary_parameters(build_model(family, width=1, modes=1, t_ref=.2).double())
    with torch.no_grad():
        model.head.bias.fill_(.05)
        if family.endswith("_moment"):
            model.mean_head.weight.zero_()
            model.mean_head.bias.fill_(.8)
    state = _state(grid=(4, 4), batch=1).requires_grad_()
    horizon = torch.tensor(.07, dtype=torch.float64, requires_grad=True)
    equation, geometry = _fixture((4, 4))
    assert torch.autograd.gradcheck(lambda u, h: model(u, h, equation, geometry),
                                   (state, horizon), eps=1e-6, atol=3e-7, rtol=3e-5)


@pytest.mark.parametrize("family", CONSTRAINED_FAMILIES)
def test_optimizer_updates_and_checkpoint_roundtrip_preserve_normalization(family):
    mean, std = torch.linspace(-.1, .1, 12), torch.linspace(.8, 1.2, 12)
    model = build_model(family, width=2, modes=2, normalization=(mean, std), t_ref=.2)
    state = _state(dtype=torch.float32)
    equation, geometry = _fixture()
    base = split_step(state, .1, equation, geometry)
    target = capacity_update(base, .008 * torch.ones_like(base))
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    losses = []
    for _ in range(3):
        optimizer.zero_grad()
        loss = (model(state, .1, equation, geometry) - target).square().mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]
    assert torch.equal(model.feature_mean, mean) and torch.equal(model.feature_std, std)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = build_model(family, width=2, modes=2, t_ref=.2)
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    with torch.no_grad():
        assert torch.equal(model(state, .1, equation, geometry), restored(state, .1, equation, geometry))


@pytest.mark.parametrize("kwargs", ({"family": "unknown"}, {"family": "premix_gated", "modes": 0},
                                    {"family": "fno_moment", "width": True},
                                    {"family": "fno_gated", "t_ref": 0},
                                    {"family": "premix_moment", "normalization": [1]}))
def test_invalid_families_scales_and_shapes_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        build_model(**kwargs)


def test_invalid_mean_and_spatial_shapes_are_rejected():
    state = _state()
    with pytest.raises(ValueError, match="one scalar per parent"):
        calibrate_mean(state, torch.ones(2, 2))
    with pytest.raises(ValueError, match="match the physical base"):
        mean_shape_update(state, 0., torch.ones(1, 1, 2, 2))
