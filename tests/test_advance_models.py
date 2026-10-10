"""Independent coefficient checks and controls for the bounded advance roster."""
from copy import deepcopy

import pytest
import torch
from scipy.integrate import solve_ivp

from tdn.analysis.advance.models import (FAMILIES, MODEL_SPECS, PHYSICAL_FAMILIES,
    TRAINABLE_FAMILIES, ZERO_HEAD_FAMILIES, make_model)
from tdn.analysis.advance.numerics import cubic_commutator_defect, deployable_defect_scale, diffusion_generator
from tdn.analysis.frontier import numerics
from tdn.analysis.portfolio.models import make_model as portfolio_model
from tdn.analysis.roadmap.numerics import cubic_df_defect
from tdn.numerics import Equation, Geometry


@pytest.fixture(scope="module", autouse=True)
def bounded_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def state(n=8, *, device="cpu", dtype=torch.float64):
    x = torch.arange(n, device=device, dtype=dtype) / n
    return (.43 + .12 * torch.cos(2 * torch.pi * x[:, None] + .2)
            + .06 * torch.sin(4 * torch.pi * x[None, :] - .4))[None, None]


def model(family, track="discrete"):
    torch.manual_seed(730021)
    return make_model(family, track, dict(width=3, depth=2, modes=2,
        reaction_substeps=2, cubic_nodes=2)).double()


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("track", numerics.TRACKS)
def test_advance_roster_identity_finiteness_and_metadata(family, track):
    value, u, eq, geo = model(family, track), state(), Equation(.003, 2.), Geometry((8, 8), (1., 1.))
    actual = value(u, .03, eq, geo)
    assert actual.shape == u.shape and actual.dtype == u.dtype and torch.isfinite(actual).all()
    torch.testing.assert_close(value(u, 0., eq, geo), u, atol=0, rtol=0)
    info = value.architecture_metadata()
    assert value.family == info["family"] == family and info["track"] == track
    assert MODEL_SPECS[family]["trainable"] == any(p.requires_grad for p in value.parameters())
    assert not info["stability_certificate"] and not info["published_FNO_reproduction"]


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("track", numerics.TRACKS)
def test_advance_batch_precision_and_state_dict_round_trip(family, track):
    value, eq, geo = model(family, track), Equation(.003, 2.), Geometry((8, 8), (1., 1.))
    u = torch.cat([state(), .89 * state()])
    h = torch.tensor([.03, .05], dtype=u.dtype)
    expected = torch.cat([value(u[i:i+1], h[i], eq, geo) for i in range(2)])
    torch.testing.assert_close(value(u, h, eq, geo), expected, atol=3e-13, rtol=2e-11)
    copy = model(family, track)
    copy.load_state_dict(value.state_dict(), strict=True)
    torch.testing.assert_close(copy(u, h, eq, geo), expected, atol=3e-13, rtol=2e-11)
    torch.testing.assert_close(deepcopy(value).float()(u.float(), h.float(), eq, geo).double(),
                               expected, atol=8e-7, rtol=5e-6)


@pytest.mark.parametrize("family", PHYSICAL_FAMILIES)
@pytest.mark.parametrize("track", numerics.TRACKS)
def test_advance_physical_nulls_survive_nonzero_weights(family, track):
    value = model(family, track)
    with torch.no_grad():
        for p in value.parameters():
            if p.requires_grad:
                p.copy_(torch.randn_like(p) * .2)
    geo = Geometry((8, 8), (1., 1.))
    for u, eq, h in [(state(), Equation(0., 2.), .03), (state(), Equation(.003, 0.), .03),
                      (state(), Equation(.003, 2.), 0.), (state() * 0 + .7, Equation(.003, 2.), .03)]:
        correction = value.correction_components(u, h, eq, geo)["increment"]
        torch.testing.assert_close(correction, torch.zeros_like(correction), atol=0, rtol=0)


@pytest.mark.parametrize("track", numerics.TRACKS)
@pytest.mark.parametrize("transport", ("none", "symmetric_heat"))
def test_advance_cubic_homogeneity_and_time_coefficient(track, transport):
    u, eq, geo = state(), Equation(.003, 2.), Geometry((8, 8), (1., 1.))
    c = u.mean()
    call = lambda value, h: cubic_commutator_defect(value, h, eq, geo, track=track, transport=transport)
    torch.testing.assert_close(call(c + 2 * (u-c), .01), 8 * call(u, .01), atol=2e-19, rtol=1e-10)
    torch.testing.assert_close(call(2*c-u, .01), -call(u, .01), atol=2e-19, rtol=1e-10)
    ratio = float(call(u, .004).norm() / call(u, .002).norm())
    assert 7.8 < ratio < 8.2
    if transport == "none":
        torch.testing.assert_close(call(u, .004), 8 * call(u, .002), atol=0, rtol=2e-15)


@pytest.mark.parametrize("track", numerics.TRACKS)
def test_advance_cubic_matches_small_step_nested_volterra_coefficient(track):
    u, eq, geo = state(6), Equation(.003, 2.), Geometry((6, 6), (1., 1.))
    relative = []
    for h in (.02, .01, .005):
        leading = cubic_commutator_defect(u, h, eq, geo, track=track)
        full = cubic_df_defect(u, h, eq, geo, nodes=8, target=track)
        relative.append(float((full-leading).norm()/leading.norm()))
    assert relative[-1] < .002
    assert relative[-1] < .6 * relative[-2] < .4 * relative[0]


@pytest.mark.parametrize("track", numerics.TRACKS)
def test_advance_cubic_independent_dop853_full_equation_odd_defect(track):
    """Extract cubic amplitude parity from the actual coupled-vs-DF defect.

    This integrates the independent finite RHS, rather than just comparing two
    versions of the proposed correction. Linear perturbations commute with the
    scalar background, and odd amplitude subtraction removes the quadratic term.
    """
    u, eq, geo = state(6), Equation(.003, 2.), Geometry((6, 6), (1., 1.))
    relative = []
    for h in (.02, .01):
        differences = []
        for sign in (1, -1):
            initial = .43 + sign * (u-.43)
            def rhs(_, flat):
                return numerics.rhs(torch.from_numpy(flat.copy()).reshape_as(initial), eq, geo, track).numpy().ravel()
            solution = solve_ivp(rhs, [0, h], initial.numpy().ravel(), method="DOP853", rtol=2e-13, atol=2e-14)
            assert solution.success
            target = torch.from_numpy(solution.y[:, -1]).reshape_as(initial)
            differences.append(target-numerics.df_step(initial, h, eq, geo, track, reaction_substeps=8))
        actual = (differences[0]-differences[1])/2
        leading = cubic_commutator_defect(u, h, eq, geo, track=track)
        relative.append(float((actual-leading).norm()/leading.norm()))
    assert relative[-1] < .004 and relative[-1] < .65 * relative[0]


def test_advance_projected_cubic_requires_nonassociative_products():
    n = 8
    x = torch.arange(n, dtype=torch.float64)/n
    u = (.43 + .12*torch.cos(6*torch.pi*x[:, None]+.4) + .07*torch.cos(4*torch.pi*x[None, :]))[None, None]
    eq, geo, h = Equation(.004, 3.), Geometry((n, n), (1., 1.)), .01
    actual = cubic_commutator_defect(u, h, eq, geo, track="continuum")
    v = u-u.mean()
    apply = lambda q: diffusion_generator(q, eq, geo, "continuum")
    # The unprojected pointwise identity does NOT apply to the Galerkin algebra.
    wrong = eq.reaction_rate**2*h**3/6*(-v.square()*apply(v)+2*v*apply(v.square())-apply(v.pow(3)))
    assert float((actual-wrong).norm()) > .1*float(actual.norm())


@pytest.mark.parametrize("track", numerics.TRACKS)
def test_advance_two_basis_unit_control_gains_and_heat_are_explicit(track):
    u, eq, geo = state(), Equation(.02, 3.), Geometry((8, 8), (1., 1.))
    fit, fixed, raw = (model(family, track) for family in ("two_basis", "commutator_cubic", "commutator_raw"))
    torch.testing.assert_close(fit(u, .04, eq, geo), fixed(u, .04, eq, geo), atol=0, rtol=0)
    assert float((fixed(u, .04, eq, geo)-raw(u, .04, eq, geo)).norm()) > 1e-12
    with torch.no_grad(): fit.gain_logits.copy_(torch.tensor([.4, -.3]))
    parts = fit.correction_components(u, .04, eq, geo)
    gains = fit.gains()
    torch.testing.assert_close(parts["increment"], gains[0]*parts["quadratic"]+gains[1]*parts["cubic"], atol=0, rtol=0)
    assert bool(((gains > .25) & (gains < 1.75)).all())
    assert fit.parameter_report()["effective_gains"] == gains.detach().tolist()


@pytest.mark.parametrize("track", numerics.TRACKS)
@pytest.mark.parametrize("family", ZERO_HEAD_FAMILIES)
def test_advance_zero_heads_preserve_df_and_activate_trunk_after_update(track, family):
    value, u, eq, geo = model(family, track), state(), Equation(.01, 3.), Geometry((8, 8), (1., 1.))
    baseline = numerics.df_step(u, .08, eq, geo, track, reaction_substeps=2)
    torch.testing.assert_close(value(u, .08, eq, geo), baseline, atol=0, rtol=0)
    target = baseline + .01*u
    loss = (value(u, .08, eq, geo)-target).square().mean()
    loss.backward()
    assert value.head[-1].weight.grad.norm() > 0
    assert value.lift.weight.grad is not None and value.lift.weight.grad.count_nonzero() == 0
    with torch.no_grad():
        heads = [value.head[-1]] + ([value.mean_head[-1]] if family == "fno_mean" else [])
        for head in heads:
            for p in head.parameters():
                assert p.grad is not None and p.grad.norm() > 0
                p.add_(-.01*p.grad/p.grad.norm())
    value.zero_grad(set_to_none=True)
    (value(u, .08, eq, geo)-target).square().mean().backward()
    assert torch.isfinite(value.lift.weight.grad).all() and value.lift.weight.grad.norm() > 0


@pytest.mark.parametrize("track", numerics.TRACKS)
def test_advance_fno_mean_has_no_duplicate_mean_and_known_coefficients(track):
    value, u, eq, geo = model("fno_mean", track), state(), Equation(.01, 3.), Geometry((8, 8), (1., 1.))
    assert value.head[-1].bias is None
    with torch.no_grad():
        value.head[-1].weight.fill_(.4)
        value.mean_head[-1].bias.fill_(.2)
    parts = value.correction_components(u, .06, eq, geo)
    torch.testing.assert_close(parts["centered_increment"].mean((-2,-1)), torch.zeros((1,1), dtype=u.dtype), atol=2e-20, rtol=0)
    torch.testing.assert_close(parts["mean_increment"], .2*parts["output_scale"], atol=0, rtol=0)
    torch.testing.assert_close(parts["increment"], parts["centered_increment"]+parts["mean_increment"], atol=2e-20, rtol=2e-14)


@pytest.mark.parametrize("track", numerics.TRACKS)
def test_advance_physical_scale_respects_amplitude_time_and_lengths(track):
    u, eq = state(), Equation(.01, 3.)
    geo = Geometry((8, 8), (1., 1.))
    scale = deployable_defect_scale(u, .04, eq, geo, track=track)
    torch.testing.assert_close(deployable_defect_scale(u, .08, eq, geo, track=track), 8*scale, atol=0, rtol=2e-15)
    torch.testing.assert_close(deployable_defect_scale(u.mean()+2*(u-u.mean()), .04, eq, geo, track=track), 4*scale, atol=1e-20, rtol=2e-14)
    assert deployable_defect_scale(u, .04, eq, Geometry((8,8),(2.,2.)), track=track) < scale


@pytest.mark.parametrize("family", TRAINABLE_FAMILIES)
@pytest.mark.parametrize("track", numerics.TRACKS)
def test_advance_activated_input_time_and_parameter_gradients(family, track):
    value = model(family, track)
    with torch.no_grad():
        for p in value.parameters():
            if p.requires_grad: p.copy_(torch.randn_like(p)*.1)
    u = state().requires_grad_()
    h = torch.tensor(.04, dtype=u.dtype, requires_grad=True)
    loss = value(u, h, Equation(.01,3.), Geometry((8,8),(1.,1.))).square().mean()
    gradients = torch.autograd.grad(loss, (u, h, *(p for p in value.parameters() if p.requires_grad)))
    assert all(torch.isfinite(g).all() and g.abs().max() > 0 for g in gradients)


@pytest.mark.parametrize("track", numerics.TRACKS)
def test_advance_legacy_fno_is_exact_historical_control(track):
    config = dict(width=3, depth=2, modes=2, reaction_substeps=2)
    torch.manual_seed(701)
    legacy = make_model("fno_legacy", track, config).double()
    torch.manual_seed(701)
    previous = portfolio_model("fno_small", track, config).double()
    u, eq, geo = state(), Equation(.003,2.), Geometry((8,8),(1.,1.))
    torch.testing.assert_close(legacy(u,.04,eq,geo), previous(u,.04,eq,geo), atol=0,rtol=0)


def test_advance_invalid_inputs_rejected():
    for args in [("unknown",), ("df","unknown"), ("df","discrete",{"surprise":1})]:
        with pytest.raises(ValueError): make_model(*args)
    with pytest.raises(ValueError):
        cubic_commutator_defect(state(), .01, Equation(.01,2.), Geometry((8,8),(1.,1.)), transport="magic")
