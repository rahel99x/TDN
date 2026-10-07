"""Independent compression-order, physical-map and derivative checks."""
from __future__ import annotations

import io
import json
import math
from unittest.mock import patch

import pytest
import torch

from tdn.features.local import extract_features
from tdn.numerics import Equation, Geometry
from tdn.numerics.splitting import split_step
from tdn.research.premix_neural import (
    FAMILIES, PremixSolver, SymmetricSpectralMixer, build_model, project_modes,
)


def _state(grid=(12, 12), *, dtype=torch.float64, batch=1):
    generator = torch.Generator().manual_seed(927)
    return .15 + .7 * torch.rand(batch, 1, *grid, dtype=dtype, generator=generator)


def _learned(model):
    with torch.no_grad():
        model.head.weight.fill_(.015)
        model.head.bias.fill_(.005)
        if hasattr(model, "local_head"):
            model.local_head.weight.fill_(.009)
    return model


@pytest.mark.parametrize("grid", ((7, 9), (8, 10), (12, 12)))
@pytest.mark.parametrize("cutoff", (1, 3, 12))
def test_projection_matches_full_complex_dft_and_is_orthogonal(grid, cutoff):
    value = _state(grid, batch=2)
    nx, ny = grid
    transform = torch.fft.fft2(value, norm="ortho")
    # Independent full-complex frequency loops, not the production rFFT mask.
    for i in range(nx):
        for j in range(ny):
            if min(i, nx - i) > cutoff or min(j, ny - j) > cutoff:
                transform[..., i, j] = 0
    expected = torch.fft.ifft2(transform, norm="ortho").real
    actual = project_modes(value, cutoff)
    assert torch.allclose(actual, expected, atol=1e-14, rtol=1e-14)
    assert torch.allclose(project_modes(actual, cutoff), actual, atol=1e-14, rtol=1e-14)
    assert torch.allclose(actual.mean((-2, -1)), value.mean((-2, -1)), atol=1e-14, rtol=0)
    residual = value - actual
    assert abs(float((actual * residual).sum())) < 3e-13


def test_learned_spectral_mixer_matches_independent_scalar_frequency_product():
    torch.manual_seed(912)
    layer = SymmetricSpectralMixer(2, 2).double()
    value = torch.randn(2, 2, 9, 10, dtype=torch.float64)
    spectrum = torch.fft.rfft2(value, norm="ortho")
    expected = torch.zeros_like(spectrum)
    for i in range(9):
        signed = i if i <= 4 else i - 9
        for j in range(6):
            if abs(signed) <= 2 and j <= 2:
                for output in range(2):
                    for input_channel in range(2):
                        weight = complex(float(layer.weight_real[input_channel, output, signed + 2, j].detach()),
                                         float(layer.weight_imag[input_channel, output, signed + 2, j].detach()))
                        expected[:, output, i, j] += spectrum[:, input_channel, i, j] * weight
    expected = torch.fft.irfft2(expected, s=(9, 10), norm="ortho")
    assert torch.allclose(layer(value), expected, atol=2e-14, rtol=2e-14)
    weights = layer.weight_imag.detach().clone()
    layer.float().double()
    assert torch.equal(layer.weight_imag, weights.float().double())


def _single_channel_product_model():
    model = PremixSolver(width=1, modes=4).double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.lift.weight[0, 0, 0, 0] = 1
        model.factor_a.weight.fill_(1)
        model.factor_b.weight.fill_(1)
        model.spectral.weight_real.fill_(1)
        model.head.weight.fill_(1)
    return model


def test_actual_learned_gate_retains_high_high_to_low_before_compression():
    model = _single_channel_product_model()
    x = torch.arange(32, dtype=torch.float64) / 32
    field = (torch.cos(2 * math.pi * 9 * x) + torch.cos(2 * math.pi * 10 * x))[:, None].expand(32, 32)
    features = torch.zeros(1, 13, 32, 32, dtype=torch.float64)
    features[0, 0] = field
    result = model._network(features)
    # cos(9x)*cos(10x) contributes cos(x); their sum squared has DC one.
    expected = (1 + torch.cos(2 * math.pi * x))[None, None, :, None].expand_as(result)
    assert torch.allclose(result, expected, atol=3e-14, rtol=3e-14)
    compressed_first = features.clone()
    compressed_first[:, :1] = project_modes(features[:, :1], 4)
    assert float(model._network(compressed_first).detach().abs().max()) < 2e-14


def test_input_ablation_filters_state_before_reaction_features_but_never_base():
    model = build_model("precompress", width=2, modes=2).double()
    state = _state()
    equation, geometry = Equation(.003, 2.), Geometry((12, 12), (1., 1.))
    observed = {}

    def record_features(value, *args, **kwargs):
        observed["encoder"] = value.detach().clone()
        return extract_features(value, *args, **kwargs)

    def record_base(value, *args, **kwargs):
        observed["base"] = value.detach().clone()
        return split_step(value, *args, **kwargs)

    with patch("tdn.research.neural_baselines.extract_features", record_features), \
         patch("tdn.research.neural_baselines.split_step", record_base):
        actual = model(state, .1, equation, geometry)
    assert torch.equal(observed["base"], state)
    projected = project_modes(state, 2)
    assert torch.equal(observed["encoder"], projected)
    assert not torch.allclose(projected, state)
    assert torch.equal(actual, split_step(state, .1, equation, geometry))
    reaction_after = equation.reaction_rate * projected * (1 - projected)
    reaction_before = project_modes(equation.reaction_rate * state * (1 - state), 2)
    assert not torch.allclose(reaction_after, reaction_before)


def test_order_pair_has_identical_parameters_and_checkpoint_compatibility():
    torch.manual_seed(89)
    first = build_model("premix", width=3, modes=2)
    torch.manual_seed(89)
    second = build_model("precompress", width=3, modes=2)
    assert dict(first.named_parameters()).keys() == dict(second.named_parameters()).keys()
    for name, parameter in first.named_parameters():
        assert torch.equal(parameter, dict(second.named_parameters())[name])
    second.load_state_dict(first.state_dict(), strict=True)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_full_solver_initialization_zero_time_batching_and_finite_gradients(family, dtype):
    torch.manual_seed(714)
    model = build_model(family, width=2, modes=2, t_ref=.2).to(dtype=dtype)
    state = _state(dtype=dtype, batch=2)
    equation, geometry = Equation(.003, 2.), Geometry((12, 12), (1., 1.))
    expected = split_step(state, .07, equation, geometry)
    assert torch.equal(model(state, .07, equation, geometry), expected)
    _learned(model)
    zero = model(state, 0., equation, geometry)
    assert torch.allclose(zero, state, atol=8 * torch.finfo(dtype).eps, rtol=0)
    state.requires_grad_()
    horizon = torch.tensor([.04, .09], dtype=dtype, requires_grad=True)
    result = model(state, horizon, equation, geometry)
    result.square().mean().backward()
    assert torch.isfinite(state.grad).all() and state.grad.abs().sum() > 0
    assert torch.isfinite(horizon.grad).all() and (horizon.grad.abs() > 0).all()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    assert torch.isfinite(result).all()
    assert ((result >= 0) & (result <= 1)).all()
    with torch.no_grad():
        separate = torch.cat([model(state[i:i + 1], horizon[i], equation, geometry) for i in range(2)])
    assert torch.allclose(result, separate, atol=12 * torch.finfo(dtype).eps, rtol=1e-5)
    json.dumps(model.architecture_metadata(), allow_nan=False)


@pytest.mark.parametrize("family", ("premix", "premix_local", "precompress"))
@pytest.mark.parametrize("grid", ((7, 9), (8, 10), (12, 12)))
def test_periodic_translation_equivariance_including_even_nyquist(family, grid):
    torch.manual_seed(457)
    model = _learned(build_model(family, width=2, modes=4).double())
    state, equation = _state(grid), Equation(.003, 2.)
    geometry = Geometry(grid, (1., 1.))
    shift = (2, 3)
    expected = torch.roll(model(state, .13, equation, geometry), shift, (-2, -1))
    actual = model(torch.roll(state, shift, (-2, -1)), .13, equation, geometry)
    assert torch.allclose(actual, expected, atol=3e-14, rtol=3e-14)


def test_local_bypass_is_the_explicit_high_frequency_complement():
    base = _single_channel_product_model()
    local = PremixSolver(width=1, modes=4, local=True).double()
    local.load_state_dict(base.state_dict(), strict=False)
    with torch.no_grad():
        local.local_head.weight.fill_(1)
    generator = torch.Generator().manual_seed(174)
    features = torch.randn(1, 13, 32, 32, generator=generator, dtype=torch.float64)
    low = base._network(features)
    assert torch.allclose(project_modes(low, 4), low, atol=2e-14, rtol=2e-14)
    difference = local._network(features) - low
    assert float(difference.detach().abs().max()) > .1
    assert float(project_modes(difference, 4).detach().abs().max()) < 3e-14


@pytest.mark.parametrize("family", ("premix", "premix_local", "precompress"))
def test_state_and_time_autograd_matches_independent_finite_differences(family):
    torch.manual_seed(19)
    model = _learned(build_model(family, width=1, modes=1, t_ref=.2).double())
    state = _state((4, 4)).requires_grad_()
    horizon = torch.tensor(.07, dtype=torch.float64, requires_grad=True)
    geometry, equation = Geometry((4, 4), (1., 1.)), Equation(.003, 2.)
    assert torch.autograd.gradcheck(lambda u, h: model(u, h, equation, geometry),
                                    (state, horizon), eps=1e-6, atol=2e-7, rtol=2e-5)


@pytest.mark.parametrize("family", FAMILIES)
def test_serialization_training_normalization_and_optimizer_update(family):
    mean = torch.linspace(-.1, .1, 12)
    std = torch.linspace(.8, 1.2, 12)
    torch.manual_seed(51)
    model = build_model(family, width=2, modes=2, normalization=(mean, std), t_ref=.2)
    state, equation, geometry = _state(dtype=torch.float32), Equation(.003, 2.), Geometry((12, 12), (1., 1.))
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    target = split_step(state, .05, equation, geometry)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    for _ in range(2):
        optimizer.zero_grad()
        loss = (model(state, .1, equation, geometry) - target).square().mean()
        loss.backward()
        optimizer.step()
    assert any(not torch.equal(initial[name], value) for name, value in model.named_parameters())
    assert torch.equal(model.feature_mean, mean) and torch.equal(model.feature_std, std)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    copy = build_model(family, width=2, modes=2, t_ref=.2)
    copy.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    with torch.no_grad():
        assert torch.equal(model(state, .1, equation, geometry), copy(state, .1, equation, geometry))


@pytest.mark.parametrize("kwargs", ({"family": "unknown"}, {"family": "premix", "modes": 0},
                                    {"family": "premix", "modes": True},
                                    {"family": "premix", "width": 0},
                                    {"family": "premix", "normalization": [1]},
                                    {"family": "premix", "t_ref": 0}))
def test_invalid_constructor_requests_are_not_silently_changed(kwargs):
    with pytest.raises(ValueError):
        build_model(**kwargs)
