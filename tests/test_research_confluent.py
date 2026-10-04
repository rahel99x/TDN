"""Numerical and physical-limit audits for the experimental shared decoders."""
from __future__ import annotations

import inspect
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import hyp1f1
import torch

from reference.temporal_core import psi3
from tdn.numerics import Equation, Geometry, split_step
from tdn.research.confluent import ConfluentTDN, FixedDecayTDN, confluent_basis


def _state(dtype=torch.float64, batch=2, ndim=2):
    generator = torch.Generator().manual_seed(318)
    return 0.2 + 0.6 * torch.rand(batch, 1, *([4] * ndim), generator=generator, dtype=dtype)


def _activate(model):
    with torch.no_grad():
        model.encoder.head.weight.fill_(0.04)
        model.encoder.head.bias.copy_(torch.tensor([0.3, -0.1, 0.2, -0.4],
                                                  dtype=model.encoder.head.bias.dtype))
    return model


@pytest.mark.parametrize("rho", [0.0, 0.03, 3.0, 8.0, 40.0])
def test_independent_confluent_quadrature(rho):
    for tau in (0.0, 0.02, 0.2, 1.0):
        actual = confluent_basis(torch.tensor(rho, dtype=torch.float64),
                                 torch.tensor(tau, dtype=torch.float64), orders=5)
        for k, value in enumerate(actual):
            reference = quad(lambda s: math.exp(-rho * (tau - s)) * (tau - s)**k * s*s,
                             0, tau, epsabs=1e-14, epsrel=1e-13)[0] / math.factorial(k)
            assert value.item() == pytest.approx(reference, rel=3e-12, abs=1e-15)


@pytest.mark.parametrize("dtype,rtol", [(torch.float32, 1.5e-6), (torch.float64, 2e-12)])
def test_broad_rate_range_and_safe_inactive_branches(dtype, rtol):
    z = torch.cat((torch.zeros(1, dtype=dtype), torch.logspace(-7, 4, 201, dtype=dtype)))
    values = confluent_basis(z, torch.ones_like(z), orders=5)
    exact = np.stack([2.0 / math.factorial(k + 3) * hyp1f1(k + 1, k + 4, -z.double().numpy())
                      for k in range(5)], axis=-1)
    np.testing.assert_allclose(values.double().numpy(), exact, rtol=rtol, atol=0.0)
    for rate in (0.0, 1e4, 1e20):
        rho = torch.tensor(rate, dtype=dtype, requires_grad=True)
        tau = torch.tensor(0.0, dtype=dtype, requires_grad=True)
        output = confluent_basis(rho, tau).sum()
        gradients = torch.autograd.grad(output, (rho, tau))
        assert output == 0
        assert all(torch.isfinite(gradient) for gradient in gradients)


@pytest.mark.parametrize("point", [0.0, 0.02, 0.4, 7.999999, 8.0, 8.000001, 40.0, 80.0, 100.0])
def test_analytic_time_and_rate_derivatives(point):
    rho = torch.tensor(point, dtype=torch.float64, requires_grad=True)
    tau = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    values = confluent_basis(rho, tau, orders=5)
    for k in range(4):
        drho, dtau = torch.autograd.grad(values[k], (rho, tau), retain_graph=True)
        forcing = tau.square() if k == 0 else values[k - 1]
        torch.testing.assert_close(dtau, forcing - rho * values[k], rtol=2e-10, atol=2e-13)
        torch.testing.assert_close(drho, -(k + 1) * values[k + 1], rtol=2e-10, atol=2e-13)


def test_origin_cubic_and_nonzero_time_gradgrad():
    rho = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    tau = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    value = confluent_basis(rho, tau).sum()
    for _ in range(3):
        assert value == 0
        value, = torch.autograd.grad(value, tau, create_graph=True)
    torch.testing.assert_close(value, torch.tensor(2.0, dtype=torch.float64))
    tau = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda r, t: confluent_basis(r, t), (rho, tau))
    assert torch.autograd.gradgradcheck(lambda r, t: confluent_basis(r, t), (rho, tau))


def test_direct_fp32_confluence_avoids_rate_subtraction():
    tau = torch.logspace(-2, 0, 41, dtype=torch.float32)
    gap = 1e-4
    subtractive = tau.pow(3) * (psi3(3 * tau) - psi3((3 + gap) * tau)) / gap
    direct = confluent_basis(torch.tensor(3.0), tau)[..., 1]
    exact = (tau.double().pow(4).numpy() / 12.0
             * hyp1f1(2, 5, -3 * tau.double().numpy()))
    direct_error = np.max(np.abs(direct.double().numpy() / exact - 1))
    subtractive_error = np.max(np.abs(subtractive.double().numpy() / exact - 1))
    assert direct_error < 1e-6
    assert subtractive_error > 100 * direct_error


def test_encoder_autocast_does_not_lower_basis_precision():
    rho = torch.tensor(3.0)
    tau = torch.tensor([0.0, 0.1, 0.6])
    expected = confluent_basis(rho, tau)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = confluent_basis(rho, tau)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("family", [ConfluentTDN, FixedDecayTDN])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_heads_batched_horizons_and_live_state_gradients(family, dtype):
    equation, geometry = Equation(0.03, 2.0), Geometry((4, 4), (1.0, 1.0))
    state = _state(dtype)
    model = family(width=5, t_ref=0.8, U_ref=1.4).to(dtype)
    horizons = torch.tensor([0.02, 0.3], dtype=dtype)
    assert torch.equal(model(state, horizons, equation, geometry),
                       split_step(state, horizons, equation, geometry))
    _activate(model)
    actual = model(state, horizons, equation, geometry)
    separate = torch.cat([model(state[i:i + 1], horizons[i], equation, geometry)
                          for i in range(2)])
    torch.testing.assert_close(actual, separate)
    shaped = model(state, horizons.reshape(2, 1, 1, 1), equation, geometry)
    torch.testing.assert_close(actual, shaped, rtol=0, atol=0)
    assert "h" not in inspect.signature(model.encode).parameters
    state.requires_grad_()
    correction = model(state, horizons, equation, geometry) - split_step(state, horizons, equation, geometry)
    gradient, = torch.autograd.grad(correction.sum(), state)
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


@pytest.mark.parametrize("family", [ConfluentTDN, FixedDecayTDN])
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_exact_commuting_limits_with_active_heads(family, ndim):
    geometry = Geometry(tuple([4] * ndim), tuple([1.0] * ndim))
    state = _state(ndim=ndim)
    model = _activate(family(width=4, ndim=ndim).double())
    for equation, values in ((Equation(0.0, 2.0), state),
                             (Equation(0.03, 0.0), state),
                             (Equation(0.03, 2.0), torch.full_like(state, 0.37))):
        assert torch.count_nonzero(model.encode(values, equation, geometry)) == 0
        assert torch.equal(model(values, 0.3, equation, geometry),
                           split_step(values, 0.3, equation, geometry))


def test_shared_rate_and_amplitude_gradients_activate():
    equation, geometry = Equation(0.03, 2.0), Geometry((4, 4), (1.0, 1.0))
    state = _state()
    model = ConfluentTDN(width=4, rho_init=3.0).double()
    torch.testing.assert_close(model.rho, torch.tensor(3.0, dtype=torch.float64), rtol=1e-7, atol=1e-7)
    model(state, 0.3, equation, geometry).sum().backward()
    assert torch.count_nonzero(model.encoder.head.weight.grad) > 0
    assert model.raw_rho.grad == 0
    model.zero_grad(set_to_none=True)
    _activate(model)
    model(state, 0.3, equation, geometry).sum().backward()
    assert model.raw_rho.numel() == 1
    assert torch.isfinite(model.raw_rho.grad)
    assert model.raw_rho.grad != 0
    assert torch.count_nonzero(next(model.encoder.body.parameters()).grad) > 0


@pytest.mark.parametrize("family", [ConfluentTDN, FixedDecayTDN])
def test_physical_damping_control_and_normalization_buffers(family):
    equation = Equation(0.03, 2.0)
    undamped = family(decay_multiplier=0.0).double()
    damped = family(decay_multiplier=2.0).double()
    tau = torch.tensor([0.0, 0.2, 1.0], dtype=torch.float64)
    expected = undamped.temporal_basis(tau, equation) * torch.exp(-4.0 * tau).unsqueeze(-1)
    torch.testing.assert_close(damped.temporal_basis(tau, equation), expected)
    mean, std = torch.arange(12, dtype=torch.float64), torch.ones(12, dtype=torch.float64) * 2
    damped.set_normalization(mean, std)
    torch.testing.assert_close(damped.encoder.feature_mean, mean)
    torch.testing.assert_close(damped.encoder.feature_std, std)


@pytest.mark.parametrize("kwargs", [{"t_ref": 0}, {"U_ref": float("nan")},
                                    {"decay_multiplier": -1}, {"rho_init": 0}])
def test_invalid_model_parameters_fail(kwargs):
    with pytest.raises(ValueError):
        ConfluentTDN(**kwargs)
