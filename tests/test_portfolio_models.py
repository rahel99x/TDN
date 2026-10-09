"""Attribution controls, physical invariants and closed-form fit regressions."""
from copy import deepcopy
import math

import pytest
import torch

from tdn.analysis.portfolio.models import (FAMILIES, MODEL_SPECS, PHYSICAL_FAMILIES,
                                         TRAINABLE_FAMILIES, make_model)
from tdn.analysis.frontier.models import make_model as frontier_model
from tdn.analysis.frontier import numerics
from tdn.analysis.roadmap.numerics import lowpass, quadratic_df_defect, quadrature
from tdn.numerics import Equation, Geometry


@pytest.fixture(scope="module", autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def state(n=8, device="cpu", dtype=torch.float64):
    x = torch.arange(n, dtype=dtype, device=device) / n
    return (.43 + .08 * torch.cos(2 * torch.pi * x[:, None] + .2)
            + .03 * torch.sin(4 * torch.pi * x[None, :] - .4))[None, None]


def model(family, track="discrete", **config):
    torch.manual_seed(5301701)
    return make_model(family, track, {"width": 3, "depth": 2, "modes": 2,
        "reaction_substeps": 2, "quad_nodes": 4, "cubic_nodes": 2, **config}).double()


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("track", numerics.TRACKS)
def test_portfolio_all_families_valid_track_identity_and_metadata(family, track):
    value = model(family, track)
    u, eq, geo = state(), Equation(.003, 2.), Geometry((8, 8), (1., 1.))
    actual = value(u, .03, eq, geo)
    assert actual.shape == u.shape and actual.dtype == u.dtype and torch.isfinite(actual).all()
    torch.testing.assert_close(value(u, 0., eq, geo), u, rtol=0, atol=0)
    info = value.architecture_metadata()
    assert info["trainable"] == any(p.requires_grad for p in value.parameters())
    assert info["track"] == track and not info["published_FNO_reproduction"]
    assert not info["clipping"] and not info["stability_certificate"]


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("track", numerics.TRACKS)
def test_portfolio_operational_precision_and_batched_steps(family, track):
    value = model(family, track)
    u, eq, geo = torch.cat((state(), state() * .87)), Equation(.003, 2.), Geometry((8, 8), (1., 1.))
    h = torch.tensor([.03, .05], dtype=u.dtype)
    actual = value(u, h, eq, geo)
    separate = torch.cat([value(u[i:i+1], h[i], eq, geo) for i in range(2)])
    torch.testing.assert_close(actual, separate, atol=2e-13, rtol=2e-11)
    torch.testing.assert_close(deepcopy(value).float()(u.float(), h.float(), eq, geo).double(), actual,
                               atol=6e-7, rtol=4e-6)


@pytest.mark.parametrize("family", PHYSICAL_FAMILIES)
@pytest.mark.parametrize("track", numerics.TRACKS)
def test_portfolio_physical_nulls_survive_arbitrary_parameters(family, track):
    value = model(family, track)
    with torch.no_grad():
        for p in value.parameters():
            p.copy_(torch.randn_like(p) * .3)
        if family == "quad2_linear": value.linear_coefficients.fill_(.6)
    geo = Geometry((8, 8), (1., 1.))
    for u, eq, h in [(state(), Equation(.003, 2.), 0.), (state(), Equation(0., 2.), .04),
                      (state(), Equation(.003, 0.), .04), (torch.full_like(state(), .7), Equation(.003, 2.), .04)]:
        increment = value.correction_components(u, h, eq, geo)["increment"]
        torch.testing.assert_close(increment, torch.zeros_like(increment), atol=2e-15, rtol=0)


@pytest.mark.parametrize("family", TRAINABLE_FAMILIES)
@pytest.mark.parametrize("track", numerics.TRACKS)
def test_portfolio_input_time_parameter_gradients(family, track):
    value = model(family, track)
    with torch.no_grad():
        for p in value.parameters():
            if p.requires_grad: p.copy_(torch.randn_like(p) * .1)
    u = state().requires_grad_()
    h = torch.tensor(.03, dtype=u.dtype, requires_grad=True)
    params = tuple(p for p in value.parameters() if p.requires_grad)
    grads = torch.autograd.grad(value(u, h, Equation(.003, 2.), Geometry((8, 8), (1., 1.))).square().mean(),
                                (u, h, *params))
    assert all(torch.isfinite(g).all() for g in grads)
    assert grads[0].abs().max() > 0 and grads[1].abs() > 0
    assert all(g.abs().max() > 0 for g in grads[2:])


@pytest.mark.parametrize("track", numerics.TRACKS)
def test_portfolio_normalization_and_historical_control_are_separate(track):
    u, eq, geo = state(), Equation(.003, 2.), Geometry((8, 8), (1., 1.))
    historical = model("historical_half", track)
    torch.manual_seed(5301701)
    original = frontier_model("rank1_frozen", track, dict(width=3, depth=2, modes=2,
        reaction_substeps=2, quad_nodes=4, cubic_nodes=2)).double()
    torch.testing.assert_close(historical(u, .04, eq, geo), original(u, .04, eq, geo), atol=0, rtol=0)
    fixed = model("quad2_fixed", track)
    increment = fixed.correction_components(u, .04, eq, geo)["increment"]
    expected = lowpass(quadratic_df_defect(u, .04, eq, geo, target=track, nodes=2), 2)
    torch.testing.assert_close(increment, expected, atol=2e-16, rtol=1e-11)
    old = historical.correction_components(u, .04, eq, geo)["increment"]
    torch.testing.assert_close(increment, 2 * old, atol=3e-12, rtol=2e-6)
    assert fixed.parameter_report()["weight_sum"] == 1
    assert historical.architecture_metadata()["quadrature_normalization"].endswith("0.5")


@pytest.mark.parametrize("nodes", [2, 4])
def test_portfolio_quadrature_moments_and_symmetric_learned_nodes(nodes):
    value = model(f"quad{nodes}_fixed")
    x, w = value.nodes_weights(state())
    for degree in range(2 * nodes):
        assert float((w * x.pow(degree)).sum()) == pytest.approx(1 / (degree + 1), abs=3e-15)
    learned = model(f"quad{nodes}_conditioned")
    with torch.no_grad(): learned.node_logits.copy_(torch.linspace(-30., 30., len(learned.node_logits)))
    x, w = learned.nodes_weights(state())
    torch.testing.assert_close(x + x.flip(0), torch.ones_like(x), atol=0, rtol=0)
    assert bool(((x > 0) & (x < 1)).all()) and float(w.sum()) == pytest.approx(1)


@pytest.mark.parametrize("track", numerics.TRACKS)
def test_portfolio_all_normalized_two_node_initializations_match(track):
    names = ("quad2_fixed", "quad2_amplitude", "quad2_nodes", "quad2_joint", "quad2_linear",
             "quad2_conditioned", "residual_quad2", "conditioned_rich", "band_gain")
    u, eq, geo = state(), Equation(.003, 2.), Geometry((8, 8), (1., 1.))
    expected = model(names[0], track)(u, .05, eq, geo)
    for name in names[1:]:
        torch.testing.assert_close(model(name, track)(u, .05, eq, geo), expected, atol=2e-16, rtol=1e-14)


@pytest.mark.parametrize("track", numerics.TRACKS)
def test_portfolio_output_cutoff_and_input_ablation_preserve_fair_base(track):
    n = 16
    x = torch.arange(n, dtype=torch.float64) / n
    u = (.45 + .08 * torch.cos(10 * torch.pi * x[:, None]) + .06 * torch.cos(12 * torch.pi * x[:, None])).expand(n, n)[None, None]
    eq, geo = Equation(.004, 2.), Geometry((n, n), (1., 1.))
    fixed, full, compressed = [model(name, track).correction_components(u, .04, eq, geo)
                               for name in ("quad2_fixed", "quad2_full", "quad2_input")]
    torch.testing.assert_close(fixed["base"], full["base"], atol=0, rtol=0)
    torch.testing.assert_close(fixed["base"], compressed["base"], atol=0, rtol=0)
    torch.testing.assert_close(fixed["increment"], lowpass(full["increment"], 2), atol=0, rtol=0)
    assert torch.fft.fft2(fixed["increment"])[..., 1, 0].abs().max() > 1e-8
    assert compressed["increment"].abs().max() < 1e-14


@pytest.mark.parametrize("track", numerics.TRACKS)
def test_portfolio_physical_quadratic_amplitude_and_small_time_order(track):
    value, u, eq, geo = model("quad2_fixed", track), state(), Equation(.003, 2.), Geometry((8, 8), (1., 1.))
    mean = u.mean()
    increment = lambda a, h: value.correction_components(mean + a * (u - mean), h, eq, geo)["increment"]
    torch.testing.assert_close(increment(2., .03), 4 * increment(1., .03), atol=3e-15, rtol=2e-8)
    ratio = float(increment(1., .004).norm() / increment(1., .002).norm())
    assert 7.5 < ratio < 8.5


@pytest.mark.parametrize("family", ["quad2_amplitude", "quad2_linear"])
def test_portfolio_training_only_matched_fit_recovers_known_gain(family):
    value = model(family)
    oracle = model(family)
    eq, geo, samples = Equation(.013, 2.), Geometry((8, 8), (1., 1.)), []
    coefficients = torch.tensor([.12] if family == "quad2_amplitude" else [.12, -.08, .04, -.02], dtype=torch.float64)
    if family == "quad2_linear": oracle.linear_coefficients.copy_(coefficients)
    else: oracle.amplitude_logit.fill_(math.atanh(.12 / .75))
    for i in range(12):
        u = state() + (.15 + .2 * (i % 4) - .43)
        h = .03 + .07 * (i // 4)
        varying_eq = Equation((.003, .012, .035)[i % 3], (.7, 1.3, 2.7)[(i * 2) % 3])
        target = oracle(u, h, varying_eq, geo)
        samples.append((u, h, varying_eq, geo, target))
    fitted = value.fit_least_squares(samples, ridge=0)
    assert fitted["identifiable"]
    assert fitted["fitted_weighted_sse"] < fitted["initial_weighted_sse"] * 1e-10
    for u, h, eq, geo, target in samples:
        torch.testing.assert_close(value(u, h, eq, geo), target, atol=3e-12, rtol=2e-11)
    if family == "quad2_linear":
        torch.testing.assert_close(value.linear_coefficients, coefficients, atol=1e-6, rtol=1e-4)
    else:
        assert fitted["fitted_amplitude"] == pytest.approx(1.12, abs=1e-9)


def test_portfolio_amplitude_fit_reports_constraint_and_linear_nonidentifiability():
    u, eq, geo = state(), Equation(.01, 2.), Geometry((8, 8), (1., 1.))
    value = model("quad2_amplitude")
    base, basis = value.linear_design(u, .05, eq, geo)
    fit = value.fit_least_squares([(u, .05, eq, geo, base + 3 * basis)], ridge=0)
    assert fit["amplitude_clamped"] and fit["unconstrained_amplitude"] > 3.9
    assert fit["fitted_amplitude"] < 1.75
    linear = model("quad2_linear")
    zero = torch.ones_like(u) * .4
    info = linear.fit_least_squares([(zero, .05, eq, geo, linear(zero, .05, eq, geo))])
    assert info["rank"] == 0 and not info["identifiable"]
    assert torch.isfinite(linear.linear_coefficients).all()


def test_portfolio_peak_loss_is_matched_and_changes_optimal_amplitude():
    mse, peak = model("quad2_amplitude"), model("quad2_amplitude")
    u, eq, geo = state(), Equation(.02, 2.), Geometry((8, 8), (1., 1.))
    baseline, basis = mse.linear_design(u, .12, eq, geo)
    target = baseline + .1 * basis
    index = int(basis.abs().flatten().argmax())
    target.flatten()[index] += .6 * basis.flatten()[index]
    samples = [(u, .12, eq, geo, target)]
    mse_fit = mse.fit_least_squares(samples, ridge=0, peak_weight=0)
    peak_fit = peak.fit_least_squares(samples, ridge=0, peak_weight=.5)
    def objective(value):
        error = (value(u, .12, eq, geo) - target).square()
        return float(error.mean() + .5 * error.max())
    assert objective(peak) < .95 * objective(mse)
    assert peak_fit["fitted_training_objective"] == pytest.approx(objective(peak), rel=1e-9)
    assert peak_fit["optimizer_success"] and mse_fit["optimizer_success"]
    assert peak_fit["peak_weight"] == .5 and peak_fit["effective_gain_bounds"] == [.25, 1.75]


def test_portfolio_affine_control_gain_is_bounded_and_fit_interrupts_cleanly():
    value = model("quad2_linear")
    u, eq, geo = state(), Equation(.01, 2.), Geometry((8, 8), (1., 1.))
    for sign in [-1, 1]:
        value.linear_coefficients.fill_(sign * 100.)
        gain = value.response(u, .03, eq, geo)["gain"]
        assert bool(((gain >= .25) & (gain <= 1.75)).all())
    initial = deepcopy(value.state_dict())
    class Interrupted:
        def check(self): raise TimeoutError("bounded fitting interrupted")
    with pytest.raises(TimeoutError, match="bounded fitting"):
        value.fit_least_squares([(u, .03, eq, geo, u)], budget=Interrupted())
    for key, tensor in value.state_dict().items():
        torch.testing.assert_close(tensor, initial[key], atol=0, rtol=0)


def test_portfolio_parameter_redundancy_reported_through_effective_response():
    first, second = model("quad2_conditioned"), model("quad2_conditioned")
    with torch.no_grad():
        second.amplitude_logit.fill_(math.atanh(.2 / .75))
        second.conditioner[-1].bias.fill_(-math.atanh(.2 / .75))
    u, eq, geo = state(), Equation(.003, 2.), Geometry((8, 8), (1., 1.))
    torch.testing.assert_close(first(u, .04, eq, geo), second(u, .04, eq, geo), atol=1e-15, rtol=1e-14)
    report = second.parameter_report(u, .04, eq, geo)
    assert report["global_amplitude"] != 1 and "compensate" in report["identifiability"]
    assert report["response"]["gain"][0][0] == pytest.approx(1)


@pytest.mark.parametrize("gain", [.251, .6, 1.4, 1.749])
def test_portfolio_conditioner_and_scalar_fit_have_same_constant_gain_range(gain):
    scalar, conditioned = model("quad2_amplitude"), model("quad2_conditioned")
    logit = math.atanh((gain - 1) / .75)
    with torch.no_grad():
        scalar.amplitude_logit.fill_(logit)
        conditioned.conditioner[-1].bias.fill_(logit)
    u, eq, geo = torch.cat((state(), .8 * state())), Equation(.01, 2.), Geometry((8, 8), (1., 1.))
    h = torch.tensor([.03, .08], dtype=torch.float64)
    torch.testing.assert_close(scalar(u, h, eq, geo), conditioned(u, h, eq, geo), atol=2e-16, rtol=1e-14)
    torch.testing.assert_close(conditioned.response(u, h, eq, geo)["gain"],
                               torch.full((2, 1), gain, dtype=u.dtype), atol=3e-16, rtol=0)


@pytest.mark.parametrize("track", numerics.TRACKS)
def test_portfolio_learned_rule_gradient_matches_finite_differences(track):
    value = model("quad2_joint", track, modes=1)
    u, h = state(n=4).requires_grad_(), torch.tensor(.04, dtype=torch.float64, requires_grad=True)
    eq, geo = Equation(.01, 2.), Geometry((4, 4), (1., 1.))
    def evaluate(initial, step, nodes, amplitude):
        return torch.func.functional_call(value, {"node_logits": nodes, "amplitude_logit": amplitude},
                                          (initial, step, eq, geo))
    assert torch.autograd.gradcheck(evaluate, (u, h, value.node_logits, value.amplitude_logit),
                                   eps=1e-6, atol=3e-6, rtol=2e-4)


@pytest.mark.parametrize("family", ["quad2_conditioned", "conditioned_rich", "band_gain"])
def test_portfolio_state_conditioning_and_band_gains_are_translation_equivariant(family):
    value = model(family)
    with torch.no_grad():
        for parameter in value.parameters(): parameter.copy_(torch.randn_like(parameter) * .3)
    eq, geo, u = Equation(.01, 2.), Geometry((8, 8), (2., 1.)), state()
    actual = value(torch.roll(u, (2, -3), (-2, -1)), .04, eq, geo)
    expected = torch.roll(value(u, .04, eq, geo), (2, -3), (-2, -1))
    torch.testing.assert_close(actual, expected, atol=2e-13, rtol=2e-12)


@pytest.mark.parametrize("track", numerics.TRACKS)
def test_portfolio_reaction_first_has_correct_generator_and_exact_subflows(track):
    value, u, geo = model("rf", track), state(), Geometry((8, 8), (1., 1.))
    eq, h = Equation(.003, 2.), 1e-6
    torch.testing.assert_close((value(u, h, eq, geo) - u) / h, numerics.rhs(u, eq, geo, track), atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(value(u, .04, Equation(.003, 0.), geo),
                               numerics.heat_step(u, .04, Equation(.003, 0.), geo, track), atol=0, rtol=0)


def test_portfolio_fno_keeps_full_local_path_and_declares_no_reproduction():
    for name in ("fno_small", "fno_standard", "direct_fno"):
        value = model(name)
        assert all(block.local is not None for block in value.delegate.blocks)
        assert value.architecture_metadata()["fno_full_grid_local_path"]
        assert not value.architecture_metadata()["published_FNO_reproduction"]


def test_portfolio_rejects_unknown_configuration_and_negative_time():
    with pytest.raises(ValueError, match="Unknown"):
        make_model("quad2_fixed", config={"unused": 1})
    with pytest.raises(ValueError, match="nonnegative"):
        model("quad2_fixed")(state(), -.01, Equation(.003, 2.), Geometry((8, 8), (1., 1.)))
