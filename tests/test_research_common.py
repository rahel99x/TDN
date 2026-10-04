"""Training-only feature scaling and exact physical limits of research heads."""
import inspect

import pytest
import torch

from tdn.features.local import extract_features
from tdn.numerics import Equation, Geometry
from tdn.research.common import LocalEncoder, commuting_gate, fit_feature_normalization


def test_training_feature_normalization_matches_explicit_population():
    geometry = Geometry((5, 6), (1., 2.))
    equation = Equation(.03, 6.)
    generator = torch.Generator().manual_seed(876)
    states = [torch.rand(1, 1, 5, 6, generator=generator, dtype=torch.float64) for _ in range(3)]
    mean, std = fit_feature_normalization((u, equation, geometry) for u in states)
    whole = torch.cat([extract_features(u, equation, geometry).reshape(-1, 12) for u in states])
    expected = whole.std(0, correction=0)
    expected = torch.where(expected > 1e-6, expected, torch.ones_like(expected))
    torch.testing.assert_close(mean, whole.mean(0), rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(std, expected, rtol=1e-13, atol=1e-13)
    with pytest.raises(ValueError, match="training parents"):
        fit_feature_normalization([])


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_encoder_zero_head_has_live_gradient_and_no_horizon(ndim):
    geometry = Geometry((4,) * ndim, (1.,) * ndim)
    u = torch.full((2, 1, *geometry.grid), .35)
    encoder = LocalEncoder(4, ndim=ndim)
    out = encoder(u, Equation(.01, 2.), geometry)
    assert out.shape == (2, 4, *geometry.grid)
    assert torch.count_nonzero(out) == 0
    out.sum().backward()
    assert encoder.head.weight.grad.abs().sum() > 0
    assert "h" not in inspect.signature(encoder.forward).parameters


def test_radius_two_gate_includes_diagonal_information_and_exact_limits():
    geometry = Geometry((7, 7), (1., 1.))
    u = torch.full((1, 1, 7, 7), .5, dtype=torch.float64)
    equation = Equation(.01, 2.)
    assert torch.count_nonzero(commuting_gate(u, equation, geometry)) == 0
    u[0, 0, 4, 4] = .7
    gate = commuting_gate(u, equation, geometry)
    assert 0 < gate[0, 0, 3, 3] < 1
    assert torch.all((gate >= 0) & (gate < 1))
    for inactive in (Equation(0., 2.), Equation(.01, 0.)):
        assert torch.count_nonzero(commuting_gate(u, inactive, geometry)) == 0
    torch.testing.assert_close(commuting_gate(torch.roll(u, 2, 2), equation, geometry),
                               torch.roll(gate, 2, 2))
    u.requires_grad_()
    assert torch.autograd.gradcheck(lambda x: commuting_gate(x, equation, geometry), (u,))


def test_normalization_rejects_invalid_scales_and_persists():
    model = LocalEncoder(2)
    with pytest.raises(ValueError, match="positive"):
        model.set_normalization(torch.zeros(12), torch.zeros(12))
    with pytest.raises(ValueError, match="per feature"):
        model.set_normalization(torch.zeros(1), torch.ones(1))
    with pytest.raises(ValueError, match="positive"):
        model.set_normalization(torch.zeros(12), torch.full((12,), 1e-100, dtype=torch.float64))
    with pytest.raises(ValueError, match="finite"):
        model.set_normalization(torch.full((12,), 1e100, dtype=torch.float64), torch.ones(12))
    model.set_normalization(torch.arange(12.), torch.full((12,), 2.))
    restored = LocalEncoder(2)
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored.feature_mean, torch.arange(12.))
    torch.testing.assert_close(restored.feature_std, torch.full((12,), 2.))
