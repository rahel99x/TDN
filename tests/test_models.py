from __future__ import annotations

import copy
import inspect

import mpmath as mp
import numpy as np
import pytest
from scipy.integrate import quad
import torch

from reference.temporal_core import mode_response, psi3
from tdn.features.local import (SUPPORT_RADIUS, extract_features, feature_chunk,
                                feature_count, feature_names)
from tdn.models import build_model
from tdn.models.solver import corrected_step
from tdn.numerics import Equation, Geometry, split_step


def _state(geometry, *, batch=2, requires_grad=False):
    torch.manual_seed(13)
    return (0.2 + 0.6 * torch.rand(batch, 1, *geometry.grid, dtype=torch.float64)).requires_grad_(requires_grad)


def _active(model):
    with torch.no_grad():
        torch.manual_seed(9)
        model.amplitude.weight.normal_(0, 0.04)
        model.amplitude.bias.normal_(0, 0.04)
    return model


def test_independent_temporal_quadrature():
    for rate in (0.0, 0.01, 1.2, 100.0):
        for h in (0.0, 0.02, 0.5, 2.0):
            a = torch.tensor([[[0.37]]], dtype=torch.float64)
            got = mode_response(a, torch.tensor(rate, dtype=torch.float64), torch.tensor(h, dtype=torch.float64)).item()
            oracle, _ = quad(lambda s: 0.37 * np.exp(-rate * (h - s)) * s**2,
                             0.0, h, epsabs=1e-14, epsrel=1e-13)
            assert got == pytest.approx(oracle, abs=1e-13, rel=3e-12)


def test_psi_cut_derivatives_and_large_scaling():
    with mp.workdps(80):
        oracle = lambda x: (x*x - 2*x + 2 - 2*mp.exp(-x)) / x**3
        for point in (0.5 - 1e-8, 0.5, 0.5 + 1e-8):
            x = torch.tensor(point, dtype=torch.float64, requires_grad=True)
            value = psi3(x)
            for order in range(4):
                reference = float(mp.diff(oracle, mp.mpf(point), order))
                assert value.item() == pytest.approx(reference, abs=2e-11, rel=2e-9)
                if order < 3:
                    value, = torch.autograd.grad(value, x, create_graph=True)
    x = torch.tensor(1e6, dtype=torch.float64)
    assert (x * psi3(x)).item() == pytest.approx(1.0, rel=3e-6)


@pytest.mark.parametrize("ndim", (1, 2, 3))
def test_streamed_feature_parity_and_state_gradients(ndim):
    geometry = Geometry(tuple(4 + d for d in range(ndim)), tuple(1.5 + d for d in range(ndim)))
    equation = Equation(kappa=0.02, reaction_rate=1.3)
    state = _state(geometry, requires_grad=True)
    full = extract_features(state, equation, geometry, t_ref=0.8, U_ref=2.0)
    streamed = torch.cat([feature_chunk(state, equation, geometry, left, min(left + 7, state.numel()),
                                        t_ref=0.8, U_ref=2.0)
                          for left in range(0, state.numel(), 7)])
    assert full.shape == (2, *geometry.grid, feature_count(ndim))
    assert len(feature_names(ndim)) == feature_count(ndim)
    assert torch.equal(full.reshape(-1, full.shape[-1]), streamed)
    full_grad, = torch.autograd.grad(full.square().sum(), state, retain_graph=True)
    chunk_grad, = torch.autograd.grad(streamed.square().sum(), state)
    assert torch.allclose(full_grad, chunk_grad, atol=1e-12, rtol=1e-12)


def test_radius_one_features_and_fixed_metadata():
    geometry = Geometry((12,), (2.0,))
    equation = Equation(kappa=0.01, reaction_rate=1.2)
    first = _state(geometry, batch=1)
    second = first.clone()
    second[..., 7] += 0.2
    feature_a = extract_features(first, equation, geometry)
    feature_b = extract_features(second, equation, geometry)
    assert SUPPORT_RADIUS == 1
    assert torch.equal(feature_a[:, 0], feature_b[:, 0])
    assert "h" not in inspect.signature(extract_features).parameters
    assert torch.equal(feature_a[..., -2], torch.full_like(feature_a[..., -2], equation.reaction_rate))
    assert torch.equal(feature_a[..., -1], torch.full_like(feature_a[..., -1], 1.0 / 12))


@pytest.mark.parametrize("family", ("temporal_mlp", "fixed_rate", "polynomial", "rational", "generic_mlp", "taylor"))
def test_all_baseline_zero_heads_and_anchored_order(family):
    model = build_model(12, {"family": family, "width": 8, "modes": 3}, t_ref=0.7, U_ref=2.0).double()
    inputs = torch.randn(5, 12, dtype=torch.float64)
    for h in (torch.tensor(0.3, dtype=torch.float64), torch.full((5, 1, 1), 0.3, dtype=torch.float64)):
        output = model(inputs, h)
        assert output.shape == (5, 1)
        assert output.dtype == torch.float64
        assert torch.count_nonzero(output) == 0
    _active(model)
    h = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    derivative = model(inputs, h).sum()
    assert derivative.item() == 0.0
    for _ in range(2):
        derivative, = torch.autograd.grad(derivative, h, create_graph=True)
        assert derivative.item() == 0.0
    third, = torch.autograd.grad(derivative, h)
    assert torch.isfinite(third)


def test_primary_h_independence_and_physical_jets():
    model = _active(build_model(8, {"family": "temporal_mlp", "width": 9}, t_ref=0.4, U_ref=2.3).double())
    features = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
    assert "h" not in inspect.signature(model.encode).parameters
    encoded = model.encode(features)
    assert encoded.amplitudes.shape == (4, 1, 4)
    assert encoded.rates.shape == (4, 1, 4)
    for value in (0.0, 0.03, 0.8):
        h = torch.tensor(value, dtype=torch.float64, requires_grad=True)
        jets = model.jets(encoded, h)
        assert torch.equal(model.decode(encoded, h), model(features, h))
        derivative = jets[0].sum()
        for order in range(1, 4):
            derivative, = torch.autograd.grad(derivative, h, create_graph=True)
            assert torch.allclose(derivative, jets[order].sum(), atol=2e-11, rtol=3e-10)
    state_gradient, = torch.autograd.grad(model(features, torch.tensor(0.2)).sum(), features)
    assert torch.count_nonzero(state_gradient) > 0


def test_exact_leading_anchor_physical_units_and_missing_audit():
    model = build_model(6, {"family": "temporal_mlp", "width": 8,
                            "anchored_leading_defect": True}, t_ref=0.2, U_ref=3.0).double()
    features = torch.randn(3, 6, dtype=torch.float64)
    with pytest.raises(ValueError, match="independently audited"):
        model.encode(features)
    coefficient = torch.tensor([[0.2], [-0.7], [0.9]], dtype=torch.float64)
    encoded = model.encode(features, coefficient)
    assert torch.allclose(encoded.amplitudes.sum(-1), 3 * coefficient * 0.2**3 / 3.0)
    jets = model.jets(encoded, torch.tensor(0.0, dtype=torch.float64))
    assert all(torch.count_nonzero(value) == 0 for value in jets[:3])
    assert torch.allclose(jets[3], 6 * coefficient, atol=1e-14, rtol=1e-14)


def test_rate_gradients_activate_after_amplitudes():
    model = build_model(8, {"family": "temporal_mlp", "width": 7}).double()
    features = torch.randn(4, 8, dtype=torch.float64)
    h = torch.tensor(0.2, dtype=torch.float64)
    model(features, h).sum().backward()
    assert torch.count_nonzero(model.rate.weight.grad) == 0
    assert torch.count_nonzero(model.amplitude.weight.grad) > 0
    model.zero_grad(set_to_none=True)
    _active(model)
    model(features, h).sum().backward()
    assert torch.count_nonzero(model.rate.weight.grad) > 0
    assert torch.count_nonzero(model.body[0].weight.grad) > 0


def test_bf16_network_keeps_decoder_fp32_and_gradients():
    model = _active(build_model(8, {"family": "temporal_mlp", "width": 8}))
    features = torch.randn(8, 8)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        encoded = model.encode(features)
        output = model.decode(encoded, torch.tensor(0.2))
    assert encoded.amplitudes.dtype == torch.float32
    assert encoded.rates.dtype == torch.float32
    assert output.dtype == torch.float32
    assert torch.all(encoded.rates >= 0)
    output.square().sum().backward()
    assert torch.isfinite(model.rate.weight.grad).all()


def test_fixed_dictionary_is_not_trainable():
    model = build_model(8, {"family": "fixed_rate", "modes": 3, "fixed_rates": [0.0, 0.3, 5.0]})
    assert model.rate is None
    assert "fixed_rates" in dict(model.named_buffers())
    assert "fixed_rates" not in dict(model.named_parameters())
    assert torch.equal(model.encode(torch.randn(2, 8)).rates[0, 0], torch.tensor([0.0, 0.3, 5.0]))


def test_rational_denominator_is_finite_for_positive_horizons():
    model = _active(build_model(8, {"family": "rational", "width": 9}).double())
    features = torch.randn(3, 8, dtype=torch.float64)
    for value in (0.0, 1e-8, 0.5, 10.0):
        result = model(features, torch.tensor(value, dtype=torch.float64))
        assert torch.isfinite(result).all()


def test_zero_correction_reproduces_base_exactly():
    geometry = Geometry((6, 5), (1.0, 1.5))
    equation = Equation(kappa=0.01, reaction_rate=1.1)
    state = _state(geometry)
    model = build_model(feature_count(geometry.ndim), {"family": "temporal_mlp", "width": 8}).double()
    with torch.no_grad():
        base = split_step(state, 0.02, equation, geometry)
        result = corrected_step(state, 0.02, equation, geometry, model, chunk_size=7)
    assert torch.equal(result, base)


def test_chunk_checkpoint_forward_parameter_state_and_h_gradients():
    geometry = Geometry((5, 6), (1.0, 1.2))
    equation = Equation(kappa=0.01, reaction_rate=1.1)
    model = _active(build_model(feature_count(geometry.ndim), {"family": "temporal_mlp", "width": 7}).double())
    models = [copy.deepcopy(model) for _ in range(3)]
    outputs, gradients = [], []
    for current, chunks, recompute in zip(models, (10000, 7, 7), (False, False, True)):
        state = _state(geometry, requires_grad=True)
        step = torch.tensor([0.015, 0.02], dtype=torch.float64, requires_grad=True)
        output = corrected_step(state, step, equation, geometry, current,
                                chunk_size=chunks, checkpoint_chunks=recompute)
        output.square().sum().backward()
        outputs.append(output.detach())
        gradients.append((state.grad, step.grad, [p.grad for p in current.parameters()]))
    for output, (state_grad, step_grad, parameter_grads) in zip(outputs[1:], gradients[1:]):
        assert torch.allclose(output, outputs[0], atol=1e-14, rtol=1e-13)
        assert torch.allclose(state_grad, gradients[0][0], atol=2e-12, rtol=2e-12)
        assert torch.allclose(step_grad, gradients[0][1], atol=2e-12, rtol=2e-12)
        for got, reference in zip(parameter_grads, gradients[0][2]):
            assert torch.allclose(got, reference, atol=2e-12, rtol=2e-12)


def test_rollout_recomputes_features_at_changed_state():
    geometry = Geometry((8,), (1.0,))
    equation = Equation(kappa=0.01, reaction_rate=1.5)
    model = _active(build_model(feature_count(1), {"family": "temporal_mlp", "width": 7}).double())
    state = _state(geometry, batch=1)
    first = corrected_step(state, 0.05, equation, geometry, model)
    second = corrected_step(first, 0.05, equation, geometry, model)
    stale_features = extract_features(state, equation, geometry).reshape(-1, feature_count(1))
    stale = split_step(first, 0.05, equation, geometry) + model(stale_features, torch.tensor(0.05)).reshape_as(first)
    assert not torch.allclose(second, stale, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("configuration", ({"family": "kan"}, {"family": "temporal_kan"},
                                            {"family": "unknown"},
                                            {"family": "temporal_mlp", "h_independent_encoder": False}))
def test_unsupported_or_invalid_architectures_fail_closed(configuration):
    with pytest.raises(ValueError):
        build_model(8, configuration)
