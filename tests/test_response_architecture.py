"""New diagnostic architectures: numerical contracts, never pilot success."""
import pytest
import torch

from tdn.analysis.response.contracts import Physics
from tdn.analysis.response.models import create_model, DIAGNOSTIC_FAMILIES, FAMILIES
from tdn.analysis.response.temporal_basis import endpoint_basis, response_kernel, correction_gains
from tdn.analysis.response.features import correction_gate
from tdn.analysis.response import numerics as n
from tdn.numerics.types import Geometry

TARGETS = ('fd_same_grid', 'galerkin_same_grid', 'galerkin_projected_fine')
ENDPOINTS = ('direct_fno', 'residual_fno', 'temporal', *DIAGNOSTIC_FAMILIES)


@pytest.fixture(autouse=True)
def deterministic_threads():
    old = torch.get_num_threads(); torch.set_num_threads(1); torch.manual_seed(81713)
    yield
    torch.set_num_threads(old)


def field(size=7, batch=2, ndim=2, dtype=torch.float64, device='cpu'):
    x = torch.arange(size, dtype=dtype, device=device)/size
    u = .31+.07*torch.cos(2*torch.pi*x)
    if ndim == 2:
        u = u[:, None]+.04*torch.sin(4*torch.pi*x[None, :]+.3)
    geom = Geometry((size,)*ndim, (1.,)*ndim)
    return u.reshape(1, 1, *geom.grid).expand(batch, 1, *geom.grid).clone(), geom


def model(family, **options):
    config = dict(family=family, width=4, depth=1, modes=2, rank=4,
                  quadrature_nodes=2, mean_nodes=2, reaction_substeps=2)
    config.update(options)
    return create_model(config).double()


def activate(m):
    with torch.no_grad():
        for name, parameter in m.named_parameters():
            if 'head' in name or 'amplitude' in name:
                parameter.add_(.08*torch.randn_like(parameter))


@pytest.mark.parametrize('order', (2, 3))
def test_generalized_response_kernel_independent_quadrature_and_zero_gradient(order):
    # Independent high-order Gauss integration checks cancellation and inverse
    # forms at zero, around the series switch, and across rapid time scales.
    import numpy as np
    nodes, weights = np.polynomial.legendre.leggauss(96)
    s = torch.tensor((nodes+1)/2, dtype=torch.float64)
    w = torch.tensor(weights/2, dtype=torch.float64)
    x = torch.tensor([0., 1e-9, .4999, .5001, 3., 30.], dtype=torch.float64, requires_grad=True)
    expected = (torch.exp(-x[:, None]*(1-s))*s.pow(order-1)*w).sum(1)
    actual = response_kernel(x, order)
    torch.testing.assert_close(actual, expected, rtol=3e-12, atol=3e-14)
    derivative = torch.autograd.grad(actual.sum(), x)[0]
    torch.testing.assert_close(derivative[0], torch.tensor(-1/(order*(order+1)), dtype=x.dtype), atol=1e-15, rtol=1e-14)
    assert torch.isfinite(derivative).all()


@pytest.mark.parametrize('kind', ('mixed', 'response'))
@pytest.mark.parametrize('rank', (4, 8))
@pytest.mark.parametrize('order', (2, 3))
def test_time_dictionary_has_declared_onset_finite_gradients_and_localized_peak(kind, rank, order):
    tau = torch.tensor([1e-6, 2e-6, .02, .2, 1., 3., 100.], dtype=torch.float64).reshape(-1, 1, 1).requires_grad_()
    basis = endpoint_basis(tau, order=order, rank=rank, kind=kind)
    torch.testing.assert_close(basis[1]/basis[0], torch.full_like(basis[0], 2.**order), rtol=2e-4, atol=1e-7)
    assert torch.isfinite(torch.autograd.grad(basis.sum(), tau)[0]).all()
    assert torch.equal(endpoint_basis(tau[:1]*0, order=order, rank=rank, kind=kind), torch.zeros_like(basis[:1]))
    if kind == 'mixed':
        assert basis[2:-1, -1].max() > basis[2, -1] and basis[2:-1, -1].max() > basis[-1, -1]


@pytest.mark.parametrize('family', DIAGNOSTIC_FAMILIES)
@pytest.mark.parametrize('target', TARGETS)
@pytest.mark.parametrize('size', (6, 7))
@pytest.mark.parametrize('ndim', (1, 2))
def test_target_models_zero_heads_active_queries_and_translation(family, target, size, ndim):
    u, geom = field(size, ndim=ndim); eq = Physics(.013, 2.4)
    m = model(family, ndim=ndim)
    base = n.anchor(u, .03, eq, geom, target, reaction_substeps=2)
    torch.testing.assert_close(m(u, .03, eq, geom, target), base, atol=0, rtol=0)
    activate(m)
    enc = m.encode(u, eq, geom, target)
    times = torch.tensor([[.03, .01], [.04, .02]], dtype=u.dtype)
    many = m.decode(enc, times)
    torch.testing.assert_close(m.decode(enc, times.flip(1)), many.flip(1), atol=0, rtol=0)
    single = torch.cat([m(u[i:i+1], times[i, 0], eq, geom, target) for i in range(2)])
    torch.testing.assert_close(many[:, 0], single, atol=2e-12, rtol=2e-11)
    shifted = m(torch.roll(u, 2, -1), times[:, 0], eq, geom, target)
    torch.testing.assert_close(shifted, torch.roll(many[:, 0], 2, -1), atol=2e-12, rtol=2e-11)
    torch.testing.assert_close(m(u, 0., eq, geom, target), u, atol=0, rtol=0)
    assert many.shape == (2, 2, 1, *geom.grid) and not many.is_complex()


@pytest.mark.parametrize('family', DIAGNOSTIC_FAMILIES)
@pytest.mark.parametrize('kind', ('mixed', 'response'))
def test_new_model_physics_and_time_derivatives_and_live_head_trunk(family, kind):
    u, geom = field(batch=1); k = torch.tensor(.013, dtype=u.dtype, requires_grad=True)
    r = torch.tensor(2.4, dtype=u.dtype, requires_grad=True); h = torch.tensor(.04, dtype=u.dtype, requires_grad=True)
    m = model(family, temporal_basis=kind, rank=8)
    target = n.anchor(u, h.detach(), Physics(k.detach(), r.detach()), geom, 'fd_same_grid', reaction_substeps=2)+.03+.02*torch.randn_like(u)
    opt = torch.optim.SGD(m.parameters(), lr=3.)
    (m(u, h, Physics(k, r), geom, 'fd_same_grid')-target).square().mean().backward()
    assert m.mean_head.weight.grad.norm() > 0 and m.spatial_head.weight.grad.norm() > 0
    opt.step(); opt.zero_grad()
    (m(u, h, Physics(k, r), geom, 'fd_same_grid')-target).square().mean().backward()
    assert sum(float(p.grad.abs().sum()) for p in m.encoder.parameters() if p.grad is not None) > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    probe = torch.randn_like(u)
    fun = lambda kv, rv, hv: (m(u, hv, Physics(kv, rv), geom, 'galerkin_projected_fine')*probe).sum()
    grads = torch.autograd.grad(fun(k, r, h), (k, r, h))
    values = [k.detach(), r.detach(), h.detach()]
    for i, grad in enumerate(grads):
        plus, minus = list(values), list(values)
        plus[i], minus[i] = values[i]+1e-6, values[i]-1e-6
        torch.testing.assert_close(grad, (fun(*plus)-fun(*minus))/(2e-6), rtol=2e-5, atol=2e-8)


@pytest.mark.parametrize('family', ENDPOINTS)
def test_frozen_gain_scaling_is_exact_affine_rescaling_and_preserves_fd_nulls(family):
    u, geom = field(batch=1); eq = Physics(.013, 2.4)
    ordinary = model(family); activate(ordinary)
    frozen = model(family, gain_normalization='frozen', normalization=dict(mean_gain=.03, centered_gain=.07))
    frozen.load_state_dict(ordinary.state_dict(), strict=False)
    with torch.no_grad():
        if family == 'temporal':
            # One shared output feeds both channels: equal gains provide the
            # exact same parameter rescaling for this historical architecture.
            frozen.centered_gain.copy_(frozen.mean_gain)
            frozen.temporal_core.amplitude.weight.mul_(.03); frozen.temporal_core.amplitude.bias.mul_(.03)
        else:
            frozen.mean_head.weight.mul_(.03); frozen.mean_head.bias.mul_(.03)
            frozen.spatial_head.weight.mul_(.07)
    torch.testing.assert_close(frozen(u, .04, eq, geom, 'fd_same_grid'), ordinary(u, .04, eq, geom, 'fd_same_grid'), atol=2e-13, rtol=2e-12)
    if family != 'direct_fno':
        for physical in (Physics(0., 2.4), Physics(.013, 0.)):
            torch.testing.assert_close(frozen(u, .03, physical, geom, 'fd_same_grid'), n.anchor(u, .03, physical, geom, 'fd_same_grid', reaction_substeps=2), atol=0, rtol=0)
    torch.testing.assert_close(frozen(u, 0., eq, geom, 'fd_same_grid'), u, atol=0, rtol=0)


@pytest.mark.parametrize('family', ENDPOINTS)
def test_endpoint_remainder_and_fp64_assembly_separate_arithmetic_from_physical_null(family):
    u, geom = field(batch=1, dtype=torch.float32); u.fill_(.4); eq = Physics(0., 2.4)
    low = model(family, numerical_remainder=True).float()
    high = model(family, numerical_remainder=True, assembly='fp64').float()
    with torch.no_grad():
        low.remainder_head.bias.fill_(1e-9)
    high.load_state_dict(low.state_dict())
    a, b = low(u, .03, eq, geom, 'fd_same_grid'), high(u, .03, eq, geom, 'fd_same_grid')
    assert a.dtype == torch.float32 and b.dtype == torch.float64 and (b-a.double()).abs().max() > 0
    diagnostics = high.correction_diagnostics(u, .03, eq, geom, 'fd_same_grid')
    torch.testing.assert_close(diagnostics['output'], b, atol=0, rtol=0)
    assert diagnostics['numerical_remainder'].abs().max() > 0 and diagnostics['assembly_difference'].abs().max() > 0
    assert all(not value.requires_grad for value in diagnostics.values())
    torch.testing.assert_close(high(u, 0., eq, geom, 'fd_same_grid'), u.double(), atol=0, rtol=0)


@pytest.mark.parametrize('family', DIAGNOSTIC_FAMILIES)
@pytest.mark.parametrize('change', ('state', 'parameter', 'gain', 'physics'))
def test_new_family_encoding_rejects_stale_components(family, change):
    u, geom = field(batch=1); eq = Physics(torch.tensor(.013), torch.tensor(2.4))
    m = model(family, gain_normalization='frozen', normalization=dict(mean_gain=.04, centered_gain=.05))
    encoded = m.encode(u, eq, geom, 'fd_same_grid')
    with torch.no_grad():
        if change == 'state': u.add_(.01)
        elif change == 'parameter': next(m.parameters()).add_(.01)
        elif change == 'gain': m.mean_gain.mul_(2)
        else: eq.kappa.add_(.01)
    with pytest.raises(ValueError, match='Stale'): m.decode(encoded, .03)


def test_gain_helper_uses_inputs_only_and_rejects_invalid_model_gains():
    u, geom = field(); eq = Physics(torch.tensor([.01, .02]), torch.tensor([2., 3.]))
    for family in ENDPOINTS:
        gains = correction_gains(u, torch.tensor([.03, .04]), eq, geom, 'fd_same_grid', family=family)
        assert set(gains) == {'mean_gain', 'centered_gain'}
        assert all(value.shape == (2, 1, 1, 1) and torch.isfinite(value).all() and (value > 0).all() for value in gains.values())
    for value in (0., -1., float('nan'), float('inf'), 1e-100, 1e100):
        with pytest.raises(ValueError, match='positive finite'):
            model('spatial_temporal', gain_normalization='frozen', normalization=dict(mean_gain=value, centered_gain=1.))


def test_historical_registry_and_state_defaults_remain_distinct_and_new_affine_fit_refuses():
    assert FAMILIES == ('direct_fno', 'residual_fno', 'temporal', 'coupled_source', 'linear_response')
    historical = model('temporal')
    assert not any('gain' in key for key in historical.state_dict())
    u, geom = field()
    for family in DIAGNOSTIC_FAMILIES:
        with pytest.raises(ValueError, match='frozen'):
            model(family).affine_features(u, .03, Physics(.013, 2.4), geom, 'fd_same_grid')


@pytest.mark.parametrize('family', DIAGNOSTIC_FAMILIES)
@pytest.mark.parametrize('target', TARGETS)
def test_actual_endpoint_correction_has_target_order_and_independent_mean_channel(family, target):
    u, geom = field(batch=1); eq = Physics(.013, 2.4)
    m = model(family)
    with torch.no_grad(): m.mean_head.bias[0] = 1.
    low = m.correction_diagnostics(u, .0001, eq, geom, target)
    high = m.correction_diagnostics(u, .0002, eq, geom, target)
    order = 2 if target == 'galerkin_projected_fine' else 3
    torch.testing.assert_close(high['mean_correction']/low['mean_correction'],
                               torch.full_like(low['mean_correction'], 2.**order), atol=1e-5, rtol=1e-5)
    assert torch.equal(low['centered_correction'], torch.zeros_like(u))


@pytest.mark.parametrize('family', ('direct_fno', 'residual_fno', 'temporal'))
def test_default_fp32_output_preserves_historical_arithmetic(family):
    u, geom = field(batch=1, dtype=torch.float32); eq = Physics(.013, 2.4)
    m = model(family).float(); activate(m); encoded = m.encode(u, eq, geom, 'fd_same_grid')
    t = n.validate(u, .04, geom, 'fd_same_grid')
    if family == 'temporal':
        raw = m.temporal_core.decode(encoded.temporal, t.reshape(1, 1, 1, 1)).movedim(-1, 1)
        correction = m.normalization.mean_scale*n.mean(raw)+m.normalization.centered_scale*n.center(raw)
        expected = m._anchor(encoded, t)+correction*correction_gate(encoded.stats, encoded.target)
    else:
        z = m.encoder(torch.cat((encoded.features, (t/m.t_ref).expand_as(u)/(1+(t/m.t_ref).expand_as(u))), 1))
        spatial = n.center(m.spatial_head(z))
        avg = m.mean_head(z.mean((2, 3))).reshape(1, 1, 1, 1)
        if family == 'direct_fno':
            expected = u+(t/m.t_ref)*m.normalization.full_scale*(spatial+avg)
        else:
            scale = (t/m.t_ref)*m._envelope(t, encoded)*correction_gate(encoded.stats, encoded.target)
            expected = m._anchor(encoded, t)+scale*(m.normalization.centered_scale*spatial+m.normalization.mean_scale*avg)
    torch.testing.assert_close(m.decode(encoded, .04), expected, atol=0, rtol=0)


@pytest.mark.parametrize('family', ('direct_fno', 'residual_fno'))
def test_fp64_affine_control_reencodes_same_fp32_state_and_preserves_zero_time(family):
    # FP64 final states are cast back to the network dtype before subsequent
    # macrosteps. The affine design must use that same cast and deployed base.
    u, geom = field(batch=1); eq = Physics(.013, 2.4)
    m = model(family, frozen_encoder=True, assembly='fp64').float()
    base, basis = m.affine_features(u, .03, eq, geom, 'fd_same_grid')
    assert base.dtype == basis.dtype == torch.float64
    torch.testing.assert_close(base, m(u, .03, eq, geom, 'fd_same_grid'), atol=0, rtol=0)
    zero, features = m.affine_features(u, 0., eq, geom, 'fd_same_grid')
    torch.testing.assert_close(zero, u, atol=0, rtol=0)
    assert not features.count_nonzero()
