"""Independent spatial, derivative and learning audits for neural controls."""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest
import torch

from tdn.numerics.splitting import split_step
from tdn.numerics.operators import rhs
from tdn.numerics.types import Equation, Geometry
from tdn.research.neural_baselines import (
    FourierNeuralOperator,
    PeriodicUNet,
    ResidualCNN,
    SpectralConv2d,
    capacity_update,
)


MODELS = (ResidualCNN, PeriodicUNet, FourierNeuralOperator)


def _state(grid=8, *, dtype=torch.float64, batch=1):
    generator = torch.Generator().manual_seed(47)
    return .1 + .8 * torch.rand(batch, 1, grid, grid, dtype=dtype, generator=generator)


def _learned(model):
    with torch.no_grad():
        model.head.weight.copy_(torch.linspace(-.1, .13, model.head.weight.numel(),
                                              dtype=model.head.weight.dtype).reshape_as(model.head.weight))
        model.head.bias.fill_(.03)
    return model


@pytest.mark.parametrize("model_type", MODELS)
@pytest.mark.parametrize("mode", ("direct", "hybrid"))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_zero_head_initialization_and_learned_zero_horizon_identity(model_type, mode, dtype):
    model = model_type(mode=mode, width=3).to(dtype=dtype)
    state, equation, geometry = _state(dtype=dtype), Equation(.03, 2.), Geometry((8, 8), (1, 1))
    expected = state if mode == "direct" else split_step(state, .17, equation, geometry)
    assert torch.equal(model(state, .17, equation, geometry), expected)
    _learned(model)
    zero_result = model(state, 0., equation, geometry)
    # The existing physical diffusion FFT roundtrip has finite roundoff at
    # h=0; retain its live time derivative instead of masking a zero horizon.
    assert torch.allclose(zero_result, state, atol=4 * torch.finfo(dtype).eps, rtol=0)
    metadata = model.architecture_metadata()
    assert metadata["track"] == mode
    assert metadata["physical_split_calls_per_step"] == int(mode == "hybrid")
    json.dumps(metadata, allow_nan=False)


@pytest.mark.parametrize("model_type", MODELS)
@pytest.mark.parametrize("mode", ("direct", "hybrid"))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_complete_learned_map_has_finite_live_state_time_and_parameter_gradients(model_type, mode, dtype):
    model = _learned(model_type(mode=mode, width=3).to(dtype=dtype))
    state = _state(dtype=dtype, batch=2).requires_grad_()
    horizon = torch.tensor([.09, .17], dtype=dtype, requires_grad=True)
    equation, geometry = Equation(.03, 2.), Geometry((8, 8), (1, 1))
    answer = model(state, horizon, equation, geometry)
    answer.square().mean().backward()
    assert state.grad is not None and torch.isfinite(state.grad).all()
    assert horizon.grad is not None and torch.isfinite(horizon.grad).all()
    assert (horizon.grad.abs() > 0).all()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    assert torch.isfinite(answer).all()
    assert ((answer >= 0) & (answer <= 1)).all()
    with torch.no_grad():
        separate = torch.cat([model(state[i:i + 1], horizon[i], equation, geometry) for i in range(2)])
    assert torch.allclose(answer, separate, atol=16 * torch.finfo(dtype).eps, rtol=1e-6)


@pytest.mark.parametrize("model_type", MODELS)
def test_spatial_backbones_have_reach_beyond_the_local_feature_stencil(model_type):
    torch.manual_seed(13)
    model = _learned(model_type(width=3).double())
    state = _state(grid=16).requires_grad_()
    result = model(state, .3, Equation(.03, 2), Geometry((16, 16), (1, 1)))
    gradient = torch.autograd.grad(result[0, 0, 0, 0], state)[0]
    # A pointwise MLP on the shared radius-one feature stencil has zero
    # dependence here. These backbones genuinely aggregate spatial context.
    assert abs(float(gradient[0, 0, 0, 4])) > 1e-12


@pytest.mark.parametrize("model_type", (ResidualCNN, FourierNeuralOperator))
def test_arbitrary_periodic_translation_equivariance(model_type):
    torch.manual_seed(17)
    model = _learned(model_type(width=3).double())
    state, geometry, equation = _state(), Geometry((8, 8), (1, 1)), Equation(.03, 2)
    shifts = (1, 3)
    actual = model(torch.roll(state, shifts, (-2, -1)), .17, equation, geometry)
    expected = torch.roll(model(state, .17, equation, geometry), shifts, (-2, -1))
    assert torch.allclose(actual, expected, atol=3e-14, rtol=3e-14)


@pytest.mark.parametrize("grid", (4, 6, 8, 32))
def test_unet_even_grids_and_declared_pooling_translation_phase(grid):
    model = _learned(PeriodicUNet(width=2).double())
    state, geometry, equation = _state(grid), Geometry((grid, grid), (1, 1)), Equation(.03, 2)
    answer = model(state, .17, equation, geometry)
    assert answer.shape == state.shape and torch.isfinite(answer).all()
    if grid % 4 == 0:
        translated = model(torch.roll(state, (4, 4), (-2, -1)), .17, equation, geometry)
        assert torch.allclose(translated, torch.roll(answer, (4, 4), (-2, -1)),
                              atol=3e-14, rtol=3e-14)


def test_periodic_cnn_convolution_wraps_opposite_edge():
    model = ResidualCNN(width=1).double()
    convolution = model.blocks[0].first
    with torch.no_grad():
        convolution.weight.zero_()
        convolution.bias.zero_()
        convolution.weight[0, 0, 0, 1] = 1.
    impulse = torch.zeros(1, 1, 8, 8, dtype=torch.float64)
    impulse[0, 0, 7, 2] = 1.
    answer = convolution(impulse)
    assert answer[0, 0, 0, 2] == 1.
    assert answer.sum() == 1.


@pytest.mark.parametrize("grid", ((4, 4), (5, 6), (8, 8), (32, 32)))
def test_fourier_banks_are_nonoverlapping_and_match_independent_transform(grid):
    nx, ny = grid
    layer = SpectralConv2d(1, 1, modes=4).double()
    with torch.no_grad():
        layer.positive_real.fill_(2.)
        layer.positive_imag.fill_(.3)
        layer.negative_real.fill_(-.7)
        layer.negative_imag.fill_(.2)
    positive, negative, y_modes = layer.active_modes(grid)
    assert set(range(positive)).isdisjoint(range(nx - negative, nx))
    generator = torch.Generator().manual_seed(19)
    state = torch.rand(1, 1, nx, ny, generator=generator, dtype=torch.float64)
    spectrum = torch.fft.rfft2(state, norm="ortho")
    expected_spectrum = torch.zeros_like(spectrum)
    # Independent scalar multiplication, deliberately without the layer's
    # einsum or weight slicing implementation.
    for i in range(nx):
        for j in range(ny // 2 + 1):
            if j < y_modes and i < positive:
                expected_spectrum[:, :, i, j] = spectrum[:, :, i, j] * complex(2., .3)
            elif j < y_modes and i >= nx - negative:
                expected_spectrum[:, :, i, j] = spectrum[:, :, i, j] * complex(-.7, .2)
    expected = torch.fft.irfft2(expected_spectrum, s=grid, norm="ortho")
    assert torch.allclose(layer(state), expected, atol=2e-14, rtol=2e-14)
    assert layer.positive_imag.dtype == torch.float64
    assert layer.negative_imag.dtype == torch.float64


def test_fourier_weights_retained_through_float64_conversion_and_both_banks_learn():
    layer = SpectralConv2d(2, 3, modes=4)
    imaginary = layer.positive_imag.detach().clone()
    layer.double()
    assert torch.equal(layer.positive_imag, imaginary.double())
    state = torch.rand(2, 2, 8, 8, dtype=torch.float64, requires_grad=True)
    layer(state).square().mean().backward()
    for parameter in layer.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert state.grad is not None and torch.isfinite(state.grad).all()


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_capacity_endpoints_zero_denominator_and_boundary_movement(dtype):
    state = torch.tensor([0., 1., 0., 1., .25, .75], dtype=dtype, requires_grad=True)
    increments = torch.tensor([0., 0., .4, -.4, 1e20, -1e20], dtype=dtype,
                              requires_grad=True)
    result = capacity_update(state, increments)
    assert torch.isfinite(result).all()
    assert ((result >= 0) & (result <= 1)).all()
    assert result[0] == 0 and result[1] == 1
    assert result[2] > 0 and result[3] < 1
    result.square().sum().backward()
    assert torch.isfinite(state.grad).all() and torch.isfinite(increments.grad).all()
    interior = torch.tensor([.2, .8], dtype=dtype)
    zero = torch.zeros_like(interior, requires_grad=True)
    derivative = torch.autograd.grad(capacity_update(interior, zero).sum(), zero)[0]
    assert torch.allclose(derivative, torch.ones_like(derivative),
                          atol=2 * torch.finfo(dtype).eps, rtol=0)


@pytest.mark.parametrize("model_type", MODELS)
def test_direct_map_can_supply_diffusion_to_an_initially_zero_cell(model_type):
    model = model_type(mode="direct", width=2).double()
    with torch.no_grad():
        model.head.bias.fill_(1.)
    state = torch.zeros(1, 1, 8, 8, dtype=torch.float64)
    assert (model(state, .1, Equation(.03, 2), Geometry((8, 8), (1, 1))) > 0).all()


@pytest.mark.parametrize("model_type", MODELS)
@pytest.mark.parametrize("mode", ("direct", "hybrid"))
def test_zero_head_then_body_parameters_actually_update(model_type, mode):
    torch.manual_seed(23)
    model = model_type(mode=mode, width=3).double()
    state, equation, geometry = _state(), Equation(.03, 2.), Geometry((8, 8), (1, 1))
    with torch.no_grad():
        baseline = model(state, .3, equation, geometry)
        target = baseline + .01 * (.5 - baseline)
    originals = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(state, .3, equation, geometry) - target).square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    assert not torch.equal(model.head.weight, originals["head.weight"])
    assert not torch.equal(model.lift.weight, originals["lift.weight"])


@pytest.mark.parametrize("mode", ("direct", "hybrid"))
def test_physical_work_is_declared_and_real(mode):
    model = ResidualCNN(mode=mode, width=2).double()
    with patch("tdn.research.neural_baselines.split_step", wraps=split_step) as physical:
        model(_state(), .1, Equation(.03, 2), Geometry((8, 8), (1, 1)))
    assert physical.call_count == int(mode == "hybrid")


@pytest.mark.parametrize("model_type", MODELS)
def test_hybrid_zero_horizon_keeps_the_complete_physical_time_derivative(model_type):
    model = _learned(model_type(mode="hybrid", width=2).double())
    state, equation, geometry = _state(), Equation(.03, 2.), Geometry((8, 8), (1, 1))
    horizon = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    weights = torch.linspace(-1., 1., state.numel(), dtype=state.dtype).reshape_as(state)
    derivative = torch.autograd.grad((model(state, horizon, equation, geometry) * weights).sum(),
                                     horizon)[0]
    expected = (rhs(state, equation, geometry) * weights).sum()
    assert torch.allclose(derivative, expected, atol=3e-13, rtol=3e-13)


@pytest.mark.parametrize("model_type", MODELS)
@pytest.mark.parametrize("mode", ("direct", "hybrid"))
def test_learned_complete_state_and_horizon_derivatives_match_finite_differences(model_type, mode):
    model = model_type(mode=mode, width=1).double()
    with torch.no_grad():
        model.head.weight.fill_(.02)
        model.head.bias.fill_(.2)  # Stay away from the capacity map's sign kink.
    state = _state(grid=4).requires_grad_()
    horizon = torch.tensor(.13, dtype=torch.float64, requires_grad=True)
    equation, geometry = Equation(.03, 2.), Geometry((4, 4), (1, 1))
    assert torch.autograd.gradcheck(lambda u, h: model(u, h, equation, geometry),
                                   (state, horizon), atol=2e-6, rtol=2e-4)


@pytest.mark.parametrize("model_type", MODELS)
def test_normalization_is_validated_copied_and_used_without_detaching_state(model_type):
    model = _learned(model_type(width=2).double())
    mean, std = torch.linspace(-.2, .2, 12), torch.linspace(.8, 1.2, 12)
    model.set_normalization(mean, std)
    mean.zero_()
    assert model.feature_mean[0] != 0
    state = _state().requires_grad_()
    answer = model(state, .2, Equation(.03, 2), Geometry((8, 8), (1, 1)))
    gradient = torch.autograd.grad(answer.square().sum(), state)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    for bad_mean, bad_std in ((torch.zeros(11), torch.ones(12)),
                              (torch.zeros(12), torch.zeros(12)),
                              (torch.full((12,), float("nan")), torch.ones(12))):
        with pytest.raises(ValueError):
            model.set_normalization(bad_mean, bad_std)


@pytest.mark.parametrize("model_type", MODELS)
@pytest.mark.parametrize("kwargs", ({"mode": "unknown"}, {"ndim": 1}, {"width": 0},
                                    {"t_ref": 0.}, {"U_ref": float("nan")}))
def test_invalid_architecture_choices_rejected(model_type, kwargs):
    with pytest.raises(ValueError):
        model_type(**kwargs)


def test_unet_invalid_small_or_odd_grid_rejected():
    model = PeriodicUNet(width=2)
    for grid in (2, 5):
        with pytest.raises(ValueError, match="even grid"):
            model(_state(grid, dtype=torch.float32), .1, Equation(.03, 2),
                  Geometry((grid, grid), (1, 1)))
