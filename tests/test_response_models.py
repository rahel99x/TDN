"""Executable architecture contracts, distinct from scientific success gates."""
import copy
import io
import pytest
import torch
from tdn.numerics.types import Geometry
from tdn.analysis.response.contracts import Physics
from tdn.analysis.response.models import create_model, FAMILIES
from tdn.analysis.response.features import build_features, FrozenNormalization
from tdn.analysis.response import numerics as n


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(71109)
    yield
    torch.set_num_threads(old)


def case(size=7, batch=2, ndim=2):
    geom = Geometry((size,)*ndim, (1.,)*ndim)
    x = torch.arange(size, dtype=torch.float64)/size
    field = .32+.09*torch.cos(2*torch.pi*x)
    if ndim == 2:
        field = field[:, None]+.04*torch.sin(4*torch.pi*x[None, :]+.3)
    u = field.reshape(1, 1, *geom.grid).expand(batch, 1, *geom.grid).clone()
    return u, Physics(.013, 2.4), geom


def model(family='coupled_source', **kwargs):
    return create_model(dict(family=family, width=4, depth=1, modes=2, rank=2,
        quadrature_nodes=2, mean_nodes=2, reaction_substeps=2, **kwargs)).double()


def activate(m):
    with torch.no_grad():
        for name, p in m.named_parameters():
            if 'head' in name or 'amplitude' in name:
                p.add_(.04*torch.randn_like(p))


@pytest.mark.parametrize('family', FAMILIES)
@pytest.mark.parametrize('size', [6, 7])
def test_response_real_odd_even_shapes_and_step(family, size):
    u, eq, geom = case(size)
    m = model(family)
    y, diagnostic = m.step(u, torch.tensor([.02, .03]), eq, geom, 'fd_same_grid')
    assert y.shape == u.shape and y.dtype == torch.float64 and not y.is_complex()
    assert torch.isfinite(y).all() and diagnostic['finite']


@pytest.mark.parametrize('family', ['residual_fno', 'temporal', 'coupled_source'])
@pytest.mark.parametrize('target', ['fd_same_grid', 'galerkin_same_grid'])
def test_response_zero_heads_retain_declared_anchor(family, target):
    u, eq, geom = case(batch=1)
    m = model(family)
    expected = n.anchor(u, .04, eq, geom, target, reaction_substeps=2)
    torch.testing.assert_close(m(u, .04, eq, geom, target), expected, rtol=2e-13, atol=2e-14)


@pytest.mark.parametrize('target',['fd_same_grid','galerkin_same_grid'])
def test_response_zero_coupled_heads_preserve_literal_fp32_anchor(target):
    u,eq,geom=case(batch=1);u=u.float()
    m=model().float()
    torch.testing.assert_close(m(u,.06,eq,geom,target),n.anchor(u,.06,eq,geom,target,reaction_substeps=2),rtol=0,atol=0)


def test_response_stage_a_alias_is_not_a_new_architecture():
    u, eq, geom = case()
    a, b = model('endpoint_tdn'), model('residual_fno')
    b.load_state_dict(a.state_dict())
    activate(a); b.load_state_dict(a.state_dict())
    torch.testing.assert_close(a(u, .03, eq, geom, 'fd_same_grid'), b(u, .03, eq, geom, 'fd_same_grid'), rtol=0, atol=0)
    assert a.architecture_metadata()['alias_of'] == 'residual_fno'


@pytest.mark.parametrize('family', ['direct_fno', 'residual_fno', 'temporal', 'coupled_source'])
def test_response_query_order_and_batch_parity(family):
    u, eq, geom = case()
    m = model(family); activate(m)
    encoded = m.encode(u, eq, geom, 'fd_same_grid')
    times = torch.tensor([[.015, .035], [.025, .045]], dtype=u.dtype)
    a = m.decode(encoded, times)
    b = m.decode(encoded, times.flip(1))
    torch.testing.assert_close(a, b.flip(1), rtol=0, atol=0)
    assert a.shape == (2, 2, 1, 7, 7)
    separate = torch.cat([m(u[i:i+1], times[i, 0], eq, geom, 'fd_same_grid') for i in range(2)])
    torch.testing.assert_close(a[:, 0], separate, rtol=2e-12, atol=2e-13)
    with pytest.raises(ValueError, match='Horizons'):
        m.decode(encoded, torch.ones(3))


@pytest.mark.parametrize('change', ['input', 'parameter', 'normalization', 'physics'])
def test_response_cache_rejects_stale_components(change):
    u, _, geom = case()
    eq = Physics(torch.tensor(.01), torch.tensor(2.))
    m = model()
    encoded = m.encode(u, eq, geom, 'fd_same_grid')
    with torch.no_grad():
        if change == 'input': u.add_(.01)
        elif change == 'parameter': next(m.parameters()).add_(.01)
        elif change == 'normalization': m.normalization.mean_scale.mul_(2)
        else: eq.kappa.add_(.01)
    with pytest.raises(ValueError, match='Stale'):
        m.decode(encoded, .03)


def test_response_inference_tensors_have_value_checked_cache():
    u, eq, geom = case(batch=1)
    m = model()
    with torch.inference_mode():
        state = u.clone()
        encoded = m.encode(state, eq, geom, 'fd_same_grid')
        assert torch.isfinite(m.decode(encoded,.03)).all()
        state.add_(.01)
        with pytest.raises(ValueError,match='Stale'):
            m.decode(encoded,.03)


def test_response_serialized_normalizers_and_rollout_refresh():
    u, eq, geom = case(batch=1)
    m = model(normalization=dict(mean_scale=.001, centered_scale=.003, full_mean=.31, full_scale=.08, floor=1e-8))
    activate(m)
    f = io.BytesIO(); torch.save(m.state_dict(), f); f.seek(0)
    restored = model(); restored.load_state_dict(torch.load(f, weights_only=True))
    assert restored.normalization.metadata() == m.normalization.metadata()
    expected = m(m(u, .01, eq, geom, 'fd_same_grid'), .02, eq, geom, 'fd_same_grid')
    torch.testing.assert_close(restored.rollout(u, [.01, .02], eq, geom, 'fd_same_grid'), expected, rtol=0, atol=0)


@pytest.mark.parametrize('family', ['direct_fno', 'residual_fno', 'temporal', 'coupled_source'])
def test_response_translation_equivariance_with_active_heads(family):
    u, eq, geom = case()
    m = model(family); activate(m)
    y = m(u, .03, eq, geom, 'fd_same_grid')
    rolled = m(torch.roll(u, (2, -1), (-2, -1)), .03, eq, geom, 'fd_same_grid')
    torch.testing.assert_close(rolled, torch.roll(y, (2, -1), (-2, -1)), rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize('family', ['residual_fno', 'temporal', 'coupled_source'])
@pytest.mark.parametrize('null', ['zero_h', 'zero_kappa', 'zero_r', 'constant'])
def test_response_exact_fd_nulls(family, null):
    u, eq, geom = case()
    m = model(family); activate(m)
    h = .04
    if null == 'zero_h': h = 0.
    elif null == 'zero_kappa': eq = Physics(0., eq.reaction_rate)
    elif null == 'zero_r': eq = Physics(eq.kappa, 0.)
    else: u.fill_(.3)
    torch.testing.assert_close(m(u, h, eq, geom, 'fd_same_grid'),
        n.anchor(u, h, eq, geom, 'fd_same_grid', reaction_substeps=2), rtol=1e-12, atol=1e-13)


def test_response_does_not_force_false_galerkin_or_closure_kappa_null():
    u, _, geom = case(batch=1)
    eq = Physics(0., 2.)
    m = model(); activate(m)
    for target in ('galerkin_same_grid', 'galerkin_projected_fine'):
        assert not torch.allclose(m(u, .1, eq, geom, target), n.anchor(u, .1, eq, geom, target, reaction_substeps=2), atol=1e-9, rtol=1e-9)


def test_response_constant_galerkin_reaction_remainder_can_be_learned():
    u, eq, geom = case(batch=1); u.fill_(.3)
    m = model()
    with torch.no_grad():m.mean_head.bias[2]=.4
    # Same-grid Galerkin uses finite RK4 reaction: constant physical dynamics
    # are logistic, but its deployed numerical subflow can still need repair.
    predicted = m(u,.08,eq,geom,'galerkin_same_grid')
    base = n.anchor(u,.08,eq,geom,'galerkin_same_grid',reaction_substeps=2)
    assert n.mean(predicted-base).abs().max()>1e-7
    torch.testing.assert_close(m(u,.08,eq,geom,'fd_same_grid'),n.anchor(u,.08,eq,geom,'fd_same_grid'),rtol=1e-13,atol=1e-13)


@pytest.mark.parametrize('family', ['residual_fno', 'direct_fno', 'temporal', 'coupled_source'])
def test_response_gradient_lifecycle(family):
    u, eq, geom = case(batch=1)
    m = model(family)
    target = n.anchor(u, .05, eq, geom, 'fd_same_grid', reaction_substeps=2)+.013+.009*torch.randn_like(u)
    optimizer = torch.optim.SGD(m.parameters(), lr=10.)
    first = m(u, .05, eq, geom, 'fd_same_grid')
    ((first-target)**2).mean().backward()
    head = [(name, p) for name, p in m.named_parameters() if 'head' in name or 'amplitude' in name]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for _, p in head)
    assert sum(float(p.grad.abs().sum()) for _, p in head) > 0
    if family == 'coupled_source':
        assert m.source_head.weight.grad.abs().sum() > 0
        assert m.mean_head.weight.grad.abs().sum() > 0
    optimizer.step(); optimizer.zero_grad()
    ((m(u, .05, eq, geom, 'fd_same_grid')-target)**2).mean().backward()
    trunk = [p for name, p in m.named_parameters() if ('encoder' in name or 'body' in name or 'source_fields' in name)]
    assert sum(float(p.grad.abs().sum()) for p in trunk if p.grad is not None) > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())


@pytest.mark.parametrize('family', ['direct_fno', 'residual_fno', 'temporal', 'coupled_source'])
def test_response_actual_reference_has_a_loss_reducing_fp64_step(family):
    u, eq, geom = case(batch=1)
    m = model(family)
    target = n.integrate(u,.06,eq,geom,'fd_same_grid',substeps=64).detach()
    initial = copy.deepcopy(m.state_dict())
    loss = (m(u,.06,eq,geom,'fd_same_grid')-target).square().mean()
    loss.backward()
    gradients = {name:p.grad.clone() for name,p in m.named_parameters() if p.grad is not None}
    norm = torch.stack([g.square().sum() for g in gradients.values()]).sum().sqrt()
    assert torch.isfinite(norm) and norm>0
    losses=[]
    # This diagnostic checks a descent direction, not a tuned optimizer recipe
    # or evidence of held-out benefit. Every attempted rate remains explicit.
    for rate in (1e-3,1e-4,1e-5,1e-6,1e-7,1e-8):
        m.load_state_dict(initial)
        with torch.no_grad():
            for name,p in m.named_parameters():
                if name in gradients:p.add_(-rate*gradients[name]/norm)
            losses.append(float((m(u,.06,eq,geom,'fd_same_grid')-target).square().mean()))
    assert min(losses)<float(loss.detach())


def test_response_tensor_physics_horizon_gradients_match_finite_difference():
    u, _, geom = case(batch=1)
    m = model(); activate(m)
    k = torch.tensor(.013, dtype=u.dtype, requires_grad=True)
    r = torch.tensor(2.4, dtype=u.dtype, requires_grad=True)
    h = torch.tensor(.03, dtype=u.dtype, requires_grad=True)
    probe = torch.randn_like(u)
    fun = lambda kv, rv, hv: (m(u, hv, Physics(kv, rv), geom, 'fd_same_grid')*probe).sum()
    grads = torch.autograd.grad(fun(k, r, h), (k, r, h))
    values = [k.detach(), r.detach(), h.detach()]
    for i, grad in enumerate(grads):
        plus, minus = list(values), list(values)
        plus[i] = plus[i]+1e-6; minus[i] = minus[i]-1e-6
        fd = (fun(*plus)-fun(*minus))/(2e-6)
        torch.testing.assert_close(grad, fd, rtol=3e-5, atol=2e-7)


def test_response_power_only_features_are_phase_blind():
    u, eq, geom = case(batch=1)
    x = torch.arange(geom.grid[0], dtype=u.dtype)/geom.grid[0]
    # Independent phase change of one of two collinear modes changes nonlinear
    # moments; both fields nevertheless have the same exact power spectrum.
    u = (.3+.07*torch.cos(2*torch.pi*x)+.03*torch.cos(4*torch.pi*x)).reshape(1,1,-1,1).expand_as(u).clone()
    v = (.3+.07*torch.cos(2*torch.pi*x)+.03*torch.cos(4*torch.pi*x+.9)).reshape(1,1,-1,1).expand_as(u).clone()
    norm = FrozenNormalization().double()
    a, _ = build_features(u, eq, geom, 'fd_same_grid', norm, mode='power')
    b, _ = build_features(v, eq, geom, 'fd_same_grid', norm, mode='power')
    torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-12)
    a, _ = build_features(u, eq, geom, 'fd_same_grid', norm, mode='phase')
    b, _ = build_features(v, eq, geom, 'fd_same_grid', norm, mode='phase')
    assert not torch.allclose(a, b)


@pytest.mark.parametrize('family', ['direct_fno', 'residual_fno'])
def test_response_frozen_encoder_affine_design_is_exact(family):
    u, eq, geom = case()
    m = model(family, frozen_encoder=True)
    base, basis = m.affine_features(u, .03, eq, geom, 'fd_same_grid')
    coefficients = torch.randn(basis.shape[0], dtype=u.dtype)
    expected = base+torch.einsum('k,kbchw->bchw', coefficients, basis)
    m.set_affine_coefficients(coefficients)
    torch.testing.assert_close(m(u, .03, eq, geom, 'fd_same_grid'), expected, rtol=2e-12, atol=2e-13)
    with pytest.raises(ValueError, match='frozen'):
        model(family).affine_features(u, .03, eq, geom, 'fd_same_grid')


def test_response_conditional_clock_contract_and_non_df_parent():
    u, eq, geom = case(batch=1)
    m = model(mean_parameterization='clock')
    encoded = m.encode(u, eq, geom, 'fd_same_grid')
    pred = m.mean_path(encoded, .04)
    assert ((pred >= 0) & (pred <= 1)).all()
    assert not torch.allclose(pred, n.mean(n.anchor(u, .04, eq, geom, 'fd_same_grid')), atol=1e-8, rtol=1e-8)
    for endpoint in (0., 1.):
        constant = torch.full_like(u, endpoint)
        enc = m.encode(constant, eq, geom, 'fd_same_grid')
        torch.testing.assert_close(m.mean_path(enc, .04), n.mean(constant), rtol=0, atol=0)
        torch.testing.assert_close(m.decode(enc, .04), constant, rtol=1e-14, atol=1e-14)
    with pytest.raises(ValueError, match='bounded nodal'):
        m(u, .04, eq, geom, 'galerkin_same_grid')
    with pytest.raises(ValueError, match='bounded nodal'):
        m(u+2, .04, eq, geom, 'fd_same_grid')


@pytest.mark.parametrize('rank', [2, 4, 8])
@pytest.mark.parametrize('basis', ['rational', 'exponential', 'polynomial'])
def test_response_rank_time_basis_and_one_dimensional_pilot(rank, basis):
    u, eq, geom = case(batch=1, ndim=1)
    config = dict(family='coupled_source', width=4, depth=1, modes=2, rank=rank, ndim=1,
        source_time_basis=basis, quadrature_nodes=2, mean_nodes=2)
    m = create_model(config).double(); activate(m)
    encoded = m.encode(u, eq, geom, 'fd_same_grid')
    source = m.source_terms(encoded, .03)
    assert source.shape == u.shape and n.mean(source).abs().max() < 1e-14
    assert torch.isfinite(m.decode(encoded, .03)).all()


def test_response_source_and_mean_quadrature_refinement():
    u, eq, geom = case(size=5, batch=1)
    models = [create_model(dict(family='coupled_source', width=4, depth=1, modes=2, rank=2,
        quadrature_nodes=count, mean_nodes=count)).double() for count in (2,4,8)]
    activate(models[0])
    for m in models[1:]: m.load_state_dict(models[0].state_dict())
    y = [m(u, .12, eq, geom, 'fd_same_grid') for m in models]
    assert (y[1]-y[2]).norm() < (y[0]-y[2]).norm()


def test_response_frozen_background_changes_propagation_without_changing_source():
    u, eq, geom = case(batch=1)
    full = model(); activate(full)
    frozen = model(source_feedback='frozen_background'); frozen.load_state_dict(full.state_dict())
    ef = full.encode(u, eq, geom, 'fd_same_grid')
    el = frozen.encode(u, eq, geom, 'fd_same_grid')
    torch.testing.assert_close(full.source_terms(ef,.07), frozen.source_terms(el,.07), rtol=0, atol=0)
    torch.testing.assert_close(full.mean_path(ef,.07), frozen.mean_path(el,.07), rtol=0, atol=0)
    assert (full.decode(ef,.12)-frozen.decode(el,.12)).abs().max()>1e-12


def test_response_coupled_is_not_an_affine_readout():
    u, eq, geom = case()
    with pytest.raises(ValueError,match='frozen direct/residual'):
        model(frozen_encoder=True).affine_features(u,.03,eq,geom,'fd_same_grid')


def test_response_competitor_auxiliary_source_supervision_and_inference_isolation():
    u, eq, geom = case(batch=1)
    m = model('residual_fno', auxiliary_source_head=True)
    encoded = m.encode(u, eq, geom, 'fd_same_grid')
    source = m.source_terms(encoded, .03)
    label = .01*(u-n.mean(u))
    (source-label).square().mean().backward()
    assert m.auxiliary_head.weight.grad.norm()>0 and torch.isfinite(m.auxiliary_head.weight.grad).all()
    before = m(u,.03,eq,geom,'fd_same_grid')
    with torch.no_grad():m.auxiliary_head.weight.add_(1.)
    after = m(u,.03,eq,geom,'fd_same_grid')
    torch.testing.assert_close(before,after,rtol=0,atol=0)


def test_response_invalid_scale_target_and_horizon_fail_explicitly():
    u, eq, geom = case()
    with pytest.raises(ValueError): model(normalization={'centered_scale': -1})
    with pytest.raises(ValueError): model()(u, .02, eq, geom, 'continuum')
    with pytest.raises(ValueError): model()(u, -.02, eq, geom, 'fd_same_grid')
    with pytest.raises(ValueError): model()(u, float('nan'), eq, geom, 'fd_same_grid')

@pytest.mark.parametrize('target', ['fd_same_grid','galerkin_same_grid','galerkin_projected_fine'])
@pytest.mark.parametrize('size', [6,7])
@pytest.mark.parametrize('representation', ['learned_basis','fixed_physical','direct_interpolation'])
def test_response_cached_source_decoder_parity(target,size,representation):
    u,eq,geom=case(size,batch=1)
    a=model(source_representation=representation);activate(a)
    b=model(source_representation=representation,source_decoder='cached_spectral');b.load_state_dict(a.state_dict())
    h=torch.tensor([[0.,.03,.067]],dtype=u.dtype)
    y=a.decode(a.encode(u,eq,geom,target),h)
    z=b.decode(b.encode(u,eq,geom,target),h)
    torch.testing.assert_close(z,y,rtol=3e-12,atol=3e-13)
    torch.testing.assert_close(z[:,0],u,rtol=0,atol=0)
    enc=b.encode(u,eq,geom,target);components=b.source_components(enc,.035)
    rebuilt=n.center((components['fields']*components['effective_coefficients']).sum(1,keepdim=True))
    torch.testing.assert_close(rebuilt,b.source_terms(enc,.035),rtol=0,atol=0)
    shifted=b(torch.roll(u,(1,-2),(-2,-1)),.03,eq,geom,target)
    torch.testing.assert_close(shifted,torch.roll(z[:,1],(1,-2),(-2,-1)),rtol=3e-12,atol=3e-13)


@pytest.mark.parametrize('target',['fd_same_grid','galerkin_same_grid','galerkin_projected_fine'])
def test_response_cached_decoder_parameter_and_physics_gradients(target):
    u,_,geom=case(batch=1);u.requires_grad_()
    k=torch.tensor(.013,dtype=u.dtype,requires_grad=True)
    r=torch.tensor(2.4,dtype=u.dtype,requires_grad=True)
    h=torch.tensor(.04,dtype=u.dtype,requires_grad=True)
    eq=Physics(k,r)
    a=model(learned_time_rates=True);activate(a)
    b=model(learned_time_rates=True,source_decoder='cached_spectral');b.load_state_dict(a.state_dict())
    values=[];gradients=[]
    for m in [a,b]:
        y=m(u,h,eq,geom,target);values.append(y)
        gradients.append(torch.autograd.grad(y.square().sum(),(u,k,r,h,*m.parameters()),allow_unused=True))
    torch.testing.assert_close(values[0],values[1],rtol=2e-12,atol=2e-13)
    for x,y in zip(*gradients):
        assert (x is None)==(y is None)
        if x is not None:
            assert torch.isfinite(x).all() and torch.isfinite(y).all()
            torch.testing.assert_close(x,y,rtol=2e-9,atol=2e-11)
    assert b.source_rate_logits.grad is None  # autograd.grad does not accumulate


def test_response_cached_source_has_one_inverse_transform_and_no_source_fft(monkeypatch):
    u,eq,geom=case(batch=1);m=model(source_decoder='cached_spectral');activate(m)
    enc=m.encode(u,eq,geom,'fd_same_grid')
    counts={'fft':0,'ifft':0};original_fft=torch.fft.fftn;original_ifft=torch.fft.ifftn
    def fft(*a,**kw):counts['fft']+=1;return original_fft(*a,**kw)
    def ifft(*a,**kw):counts['ifft']+=1;return original_ifft(*a,**kw)
    monkeypatch.setattr(torch.fft,'fftn',fft);monkeypatch.setattr(torch.fft,'ifftn',ifft)
    h=n.scalar(.03,u)
    result=m._spectral_response(enc,h,lambda s:enc.stats['mean'])
    assert counts=={'fft':0,'ifft':1} and result.shape==u.shape
    # This count excludes declared anchor/mean-path transforms, deliberately.
    with torch.no_grad():m.source_head.bias.add_(.01)
    with pytest.raises(ValueError,match='Stale'):m.decode(enc,.03)


def test_response_interpolation_is_c1_convex_and_learned_heads_live():
    u,eq,geom=case(batch=1)
    m=create_model(dict(family='coupled_source',width=4,depth=1,modes=2,rank=4,
        source_decoder='cached_spectral',source_representation='direct_interpolation',quadrature_nodes=2,mean_nodes=2)).double()
    enc=m.encode(u,eq,geom,'fd_same_grid')
    # The internal knot x=1/3 is s=t_ref/2. Basis derivatives tend to zero
    # from both sides; second derivatives need not agree (documented C1).
    derivatives=[]
    for t in [.05-1e-8,.05,.05+1e-8]:
        h=torch.tensor(t,dtype=u.dtype,requires_grad=True)
        basis=m._time_basis(enc,h)
        torch.testing.assert_close(basis.sum(1),torch.ones_like(basis[:,0]),rtol=0,atol=2e-15)
        assert (basis>=0).all()
        derivatives.append(torch.autograd.grad(basis[:,1].sum(),h)[0])
    assert max(abs(x.item()) for x in derivatives)<1e-4
    target=n.anchor(u,.05,eq,geom,'fd_same_grid')+.01*n.center(u)
    (m(u,.05,eq,geom,'fd_same_grid')-target).square().mean().backward()
    assert m.interpolation_head.weight.grad.norm()>0
    optimizer=torch.optim.SGD(m.parameters(),lr=100.)
    optimizer.step();optimizer.zero_grad()
    (m(u,.05,eq,geom,'fd_same_grid')-target).square().mean().backward()
    assert sum(p.grad.norm() for p in m.encoder.parameters() if p.grad is not None)>0


def test_response_numerical_remainder_separate_from_physical_null_and_fp64_assembly():
    u,eq,geom=case(batch=1);u=torch.full_like(u,.4).float()
    ordinary=model(numerical_remainder=True).float()
    high=model(numerical_remainder=True,assembly='fp64').float()
    with torch.no_grad():ordinary.remainder_head.bias.fill_(1e-10)
    high.load_state_dict(ordinary.state_dict())
    for physical in [Physics(0.,2.4),Physics(.013,0.)]:
        low=ordinary(u,.03,physical,geom,'fd_same_grid')
        y=high(u,.03,physical,geom,'fd_same_grid')
        baseline=n.anchor(u,.03,physical,geom,'fd_same_grid',reaction_substeps=2)
        assert y.dtype==torch.float64 and low.dtype==torch.float32
        assert torch.max(torch.abs(y-baseline.double()))>0
        assert torch.equal(low,baseline)  # same sub-ULP correction disappears
        torch.testing.assert_close(high(u,0.,physical,geom,'fd_same_grid'),u.double(),rtol=0,atol=0)
    assert 'arithmetic/subflow' in high.architecture_metadata()['numerical_remainder_scope']


def test_response_fixed_physical_bands_are_grid_consistent_and_partition_centered_field():
    from tdn.analysis.response.features import physical_band_fields,FEATURE_CHANNELS
    u,eq,geom=case(size=9,batch=1)
    fine_geom=Geometry((19,19),(1.,1.))
    fine=n.resample(u,fine_geom.grid)
    bands=physical_band_fields(n.center(u),geom)
    refined=physical_band_fields(n.center(fine),fine_geom)
    torch.testing.assert_close(sum(bands),n.center(u),rtol=2e-13,atol=2e-14)
    for coarse,high in zip(bands,refined):
        torch.testing.assert_close(n.resample(coarse,fine_geom.grid),high,rtol=2e-12,atol=2e-14)
    features,_=build_features(u,eq,geom,'fd_same_grid',FrozenNormalization())
    assert features.shape[1]==FEATURE_CHANNELS==24
    torch.testing.assert_close(features[:,14:15],u.amin((-2,-1),keepdim=True).expand_as(u))
    torch.testing.assert_close(features[:,15:16],u.amax((-2,-1),keepdim=True).expand_as(u))


def test_response_fp64_final_addition_preserves_small_learned_mean_and_restarts():
    u,eq,geom=case(batch=1);u=u.float()
    a=model().float();b=model(assembly='fp64').float()
    with torch.no_grad():a.mean_head.bias.fill_(1e-8)
    b.load_state_dict(a.state_dict())
    low=a(u,.03,eq,geom,'fd_same_grid');high=b(u,.03,eq,geom,'fd_same_grid')
    assert (high-low.double()).abs().max()>0
    output=b.rollout(u,[.03,.03],eq,geom,'fd_same_grid')
    assert output.dtype==torch.float64 and torch.isfinite(output).all()
    torch.testing.assert_close(b(high,0.,eq,geom,'fd_same_grid'),high,rtol=0,atol=0)
