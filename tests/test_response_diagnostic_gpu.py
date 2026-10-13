"""Additional allocated CUDA contracts; historical 54-case suite is unchanged."""
from copy import deepcopy
import os
import pytest
import torch

from tdn.analysis.response.contracts import Physics
from tdn.analysis.response.models import DIAGNOSTIC_FAMILIES
from tdn.analysis.response.temporal_basis import endpoint_basis
from tdn.analysis.response import numerics as n
from test_response_architecture import field, model, activate, TARGETS, ENDPOINTS

pytestmark = pytest.mark.gpu


@pytest.fixture(scope='module', autouse=True)
def limited_threads():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture(autouse=True)
def allocated_cuda():
    if not torch.cuda.is_available():
        if any(os.environ.get(flag) == '1' for flag in ('TDN_REQUIRE_RESPONSE_GPU_TESTS', 'TDN_REQUIRE_RESPONSE_DIAGNOSTIC_GPU_TESTS')):
            pytest.fail('Required diagnostic suite has no actual CUDA device')
        pytest.skip('No actual CUDA; not diagnostic GPU validation')
    from tdn.runtime.preflight import verify_runtime
    from tdn.runtime.precision import reference_precision
    verify_runtime('cuda', 'response-diagnostic-gpu-tests')
    reference_precision()
    torch.manual_seed(81713); torch.cuda.manual_seed_all(81713)


@pytest.mark.parametrize('target', TARGETS)
@pytest.mark.parametrize('family', DIAGNOSTIC_FAMILIES)
@pytest.mark.parametrize('size', (6, 7))
def test_response_diagnostic_cuda_odd_even_parity(size, family, target):
    u, geom = field(size, batch=1); eq = Physics(.013, 2.4)
    cpu = model(family); activate(cpu); gpu = deepcopy(cpu).cuda()
    expected, actual = cpu(u, .04, eq, geom, target), gpu(u.cuda(), .04, eq, geom, target)
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-11, rtol=2e-9)
    shifted = gpu(torch.roll(u.cuda(), (1, -2), (-2, -1)), .04, eq, geom, target)
    torch.testing.assert_close(shifted, torch.roll(actual, (1, -2), (-2, -1)), atol=2e-11, rtol=2e-9)


@pytest.mark.parametrize('target', TARGETS)
@pytest.mark.parametrize('family', DIAGNOSTIC_FAMILIES)
def test_response_diagnostic_cuda_head_trunk_gradients(family, target):
    u, geom = field(batch=1, device='cuda'); eq = Physics(.013, 2.4)
    m = model(family).cuda()
    teacher = n.anchor(u, .04, eq, geom, target, reaction_substeps=2)+.03+.02*torch.randn_like(u)
    opt = torch.optim.SGD(m.parameters(), lr=3.)
    (m(u, .04, eq, geom, target)-teacher).square().mean().backward()
    assert m.mean_head.weight.grad.norm() > 0 and m.spatial_head.weight.grad.norm() > 0
    opt.step(); opt.zero_grad()
    (m(u, .04, eq, geom, target)-teacher).square().mean().backward()
    assert sum(p.grad.abs().sum() for p in m.encoder.parameters() if p.grad is not None) > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())


@pytest.mark.parametrize('target', TARGETS)
@pytest.mark.parametrize('family', DIAGNOSTIC_FAMILIES)
def test_response_diagnostic_cuda_physics_time_and_query_parity(family, target):
    u, geom = field(device='cuda'); u.requires_grad_()
    k = torch.tensor([.01, .02], dtype=u.dtype, device=u.device, requires_grad=True)
    r = torch.tensor([2., 3.], dtype=u.dtype, device=u.device, requires_grad=True)
    h = torch.tensor([.03, .04], dtype=u.dtype, device=u.device, requires_grad=True)
    m = model(family, rank=8, temporal_basis='response').cuda(); activate(m)
    encoded = m.encode(u, Physics(k, r), geom, target)
    y = m.decode(encoded, h)
    grads = torch.autograd.grad(y.square().sum(), (u, k, r, h))
    assert all(torch.isfinite(g).all() and g.norm() > 0 for g in grads)
    times = torch.stack((h, h*.5), 1)
    many = m.decode(encoded, times)
    torch.testing.assert_close(many[:, 0], y, atol=0, rtol=0)
    torch.testing.assert_close(m.decode(encoded, times.flip(1)), many.flip(1), atol=0, rtol=0)
    parts = torch.cat([m(u[i:i+1], h[i], Physics(k[i], r[i]), geom, target) for i in range(2)])
    torch.testing.assert_close(parts, y, atol=2e-11, rtol=2e-9)


@pytest.mark.parametrize('family', ENDPOINTS)
def test_response_diagnostic_cuda_remainder_fp64_assembly(family):
    u, geom = field(batch=1, dtype=torch.float32, device='cuda'); u.fill_(.4); eq = Physics(0., 2.4)
    low = model(family, numerical_remainder=True).float().cuda()
    high = model(family, numerical_remainder=True, assembly='fp64').float().cuda()
    with torch.no_grad(): low.remainder_head.bias.fill_(1e-9)
    high.load_state_dict(low.state_dict())
    a, b = low(u, .03, eq, geom, 'fd_same_grid'), high(u, .03, eq, geom, 'fd_same_grid')
    assert a.dtype == torch.float32 and b.dtype == torch.float64 and (b-a.double()).abs().max() > 0
    terms = high.correction_diagnostics(u, .03, eq, geom, 'fd_same_grid')
    torch.testing.assert_close(terms['output'], b, rtol=0, atol=0)
    assert terms['assembly_difference'].abs().max() > 0 and terms['numerical_remainder'].abs().max() > 0
    torch.testing.assert_close(high(u, 0., eq, geom, 'fd_same_grid'), u.double(), atol=0, rtol=0)


@pytest.mark.parametrize('family', ENDPOINTS)
def test_response_diagnostic_cuda_frozen_gains_and_physical_nulls(family):
    u, geom = field(batch=1, device='cuda'); eq = Physics(.013, 2.4)
    m = model(family, gain_normalization='frozen', normalization=dict(mean_gain=.03, centered_gain=.05)).cuda(); activate(m)
    assert m.mean_gain > 0 and m.centered_gain > 0
    torch.testing.assert_close(m(u, 0., eq, geom, 'fd_same_grid'), u, atol=0, rtol=0)
    if family != 'direct_fno':
        for physical in (Physics(0., 2.4), Physics(.013, 0.)):
            torch.testing.assert_close(m(u, .03, physical, geom, 'fd_same_grid'), n.anchor(u, .03, physical, geom, 'fd_same_grid', reaction_substeps=2), atol=0, rtol=0)
        constant = torch.full_like(u, .4)
        torch.testing.assert_close(m(constant, .03, eq, geom, 'fd_same_grid'), n.anchor(constant, .03, eq, geom, 'fd_same_grid', reaction_substeps=2), atol=0, rtol=0)


@pytest.mark.parametrize('kind', ('mixed', 'response'))
@pytest.mark.parametrize('rank', (4, 8))
def test_response_diagnostic_cuda_target_order_dictionary(kind, rank):
    t = torch.tensor([1e-6, 2e-6, .5, 3.], device='cuda', dtype=torch.float64).reshape(-1, 1, 1, 1).requires_grad_()
    for order in (2, 3):
        y = endpoint_basis(t, order=order, rank=rank, kind=kind)
        torch.testing.assert_close(y.cpu(), endpoint_basis(t.cpu(), order=order, rank=rank, kind=kind), atol=2e-14, rtol=2e-11)
        torch.testing.assert_close(y[1]/y[0], torch.full_like(y[0], 2.**order), atol=1e-7, rtol=2e-4)
        assert torch.isfinite(torch.autograd.grad(y.sum(), t)[0]).all()


@pytest.mark.parametrize('family', DIAGNOSTIC_FAMILIES)
@pytest.mark.parametrize('dtype_name', ('float32', 'float64'))
def test_response_diagnostic_cuda_inference_cache_and_serialized_gains(family, dtype_name):
    dtype = getattr(torch, dtype_name)
    u, geom = field(batch=1, dtype=dtype, device='cuda'); eq = Physics(.013, 2.4)
    m = model(family, gain_normalization='frozen', normalization=dict(mean_gain=.03, centered_gain=.05)).to(device='cuda', dtype=dtype)
    restored = deepcopy(m); restored.load_state_dict(m.state_dict())
    with torch.inference_mode():
        state = u.clone(); encoded = m.encode(state, eq, geom, 'fd_same_grid')
        torch.testing.assert_close(m.decode(encoded, .03), restored(state, .03, eq, geom, 'fd_same_grid'), atol=0, rtol=0)
        state.add_(.01)
        with pytest.raises(ValueError, match='Stale'): m.decode(encoded, .03)


def test_response_diagnostic_cuda_actual_allocation():
    assert os.environ.get('TDN_EXECUTION_MODE') == 'desktop-slurm'
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_STEP_ID')
    assert torch.ones(1, device='cuda').device.type == 'cuda'
    assert torch.cuda.get_device_properties(0).total_memory > 0
