"""Physical channel mathematics and exact attribution controls."""
from copy import deepcopy
import math

import pytest
import torch

from tdn.analysis.adjacent.models import (CHANNEL_FAMILIES, gain_bounds, interaction_channels,
    fixed_background_quadratic, make_model)
from tdn.analysis.frontier import numerics
from tdn.analysis.portfolio.models import make_model as portfolio_model
from tdn.analysis.roadmap.numerics import lowpass, quadratic_df_defect
from tdn.numerics import Equation, Geometry


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def state(n=12, dtype=torch.float64, device="cpu"):
    x = torch.arange(n, dtype=dtype, device=device) / n
    return (.43 + .07 * torch.cos(2 * torch.pi * (x[:, None] + x[None, :]) + .4)
            + .04 * torch.cos(6 * torch.pi * x[:, None] - .3)
            + .025 * torch.sin(8 * torch.pi * x[None, :] + .2))[None, None]


def model(family, track="discrete", **config):
    return make_model(family, track, dict(modes=3, split_modes=1, quad_nodes=2,
                                        reaction_substeps=2, **config)).double()


@pytest.mark.parametrize("track", numerics.TRACKS)
@pytest.mark.parametrize("nodes", [2, 4])
@pytest.mark.parametrize("output_modes", [None, 2])
def test_adjacent_polarization_matches_shared_and_original(track, nodes, output_modes):
    u, eq, geo = state(), Equation(.012, 2.3), Geometry((12, 12), (1., 1.))
    reference = interaction_channels(u, .07, eq, geo, track=track, nodes=nodes,
                                     split_modes=1, output_modes=output_modes, implementation="reference")
    shared = interaction_channels(u, .07, eq, geo, track=track, nodes=nodes,
                                  split_modes=1, output_modes=output_modes)
    torch.testing.assert_close(shared, reference, atol=3e-18, rtol=2e-10)
    expected = quadratic_df_defect(u, .07, eq, geo, nodes=nodes, target=track)
    if output_modes is not None:
        expected = lowpass(expected, output_modes)
    torch.testing.assert_close(shared.sum(1, keepdim=True), expected, atol=3e-18, rtol=2e-10)
    assert shared.shape == (1, 3, 12, 12)


@pytest.mark.parametrize("track", numerics.TRACKS)
def test_adjacent_fixed_background_not_recentered_and_cross_symmetry(track):
    u, eq, geo = state(), Equation(.012, 2.3), Geometry((12, 12), (1., 1.))
    m = u.mean((-2, -1), keepdim=True)
    v = u - m
    a, b = lowpass(v, 1), v - lowpass(v, 1)
    q = lambda z: fixed_background_quadratic(z, m, .08, eq, geo, track=track)
    cross = q(a + b) - q(a) - q(b)
    swapped = q(b + a) - q(b) - q(a)
    torch.testing.assert_close(cross, swapped, atol=2e-20, rtol=1e-12)
    # Fixed background is an explicit argument; adding a constant direction
    # must not silently replace it with the direction's mean.
    altered = fixed_background_quadratic(v, m + .2, .08, eq, geo, track=track)
    assert not torch.allclose(q(v), altered, atol=1e-12, rtol=1e-4)
    torch.testing.assert_close(q(-v), q(v), atol=0, rtol=0)
    torch.testing.assert_close(q(2 * v), 4 * q(v), atol=0, rtol=0)


@pytest.mark.parametrize("family", CHANNEL_FAMILIES)
@pytest.mark.parametrize("track", numerics.TRACKS)
def test_adjacent_gain_one_initialization_matches_baseline(family, track):
    u, eq, geo = state(), Equation(.012, 2.3), Geometry((12, 12), (1., 1.))
    instance = model(family, track)
    base = portfolio_model("quad2_fixed", track, dict(modes=3, reaction_substeps=2)).double()
    torch.testing.assert_close(instance(u, .06, eq, geo), base(u, .06, eq, geo), atol=3e-16, rtol=1e-14)
    torch.testing.assert_close(instance.response(u, .06, eq, geo)["gain"], torch.ones(1, 3, dtype=u.dtype), atol=0, rtol=0)
    metadata = instance.architecture_metadata()
    assert metadata["track"] == track and metadata["reference_or_future_inputs"] is False
    assert not metadata["stability_certificate"] and not metadata["compute_matching_claim"]


@pytest.mark.parametrize("family", CHANNEL_FAMILIES)
@pytest.mark.parametrize("track", numerics.TRACKS)
def test_adjacent_arbitrary_gain_physical_nulls(family, track):
    value = model(family, track)
    with torch.no_grad():
        for p in value.parameters():
            p.copy_(torch.randn_like(p) * .2)
        if hasattr(value, "gain_logits"):
            value.gain_logits.copy_(torch.tensor([1., -2., .7]))
        if hasattr(value, "affine_coefficients"):
            value.affine_coefficients.fill_(.5)
    geo = Geometry((12, 12), (1., 1.))
    for u, eq, h in [(state(), Equation(.012, 2.3), 0.), (state(), Equation(0., 2.3), .07),
                     (state(), Equation(.012, 0.), .07), (torch.full_like(state(), .7), Equation(.012, 2.3), .07)]:
        increment = value.correction_components(u, h, eq, geo)["increment"]
        torch.testing.assert_close(increment, torch.zeros_like(increment), atol=0, rtol=0)
        torch.testing.assert_close(value(u, 0., eq, geo), u, atol=0, rtol=0)


@pytest.mark.parametrize("track", numerics.TRACKS)
def test_adjacent_signed_cross_and_aliasing_semantics(track):
    n = 16
    x = torch.arange(n, dtype=torch.float64) / n
    # The cross term changes sign under one input's phase reversal; LL/HH
    # stay unchanged. Raw signed Fourier coefficients are retained.
    lo = (.07 * torch.cos(2 * torch.pi * x[:, None] + .4)).expand(n, n)
    hi = (.05 * torch.cos(6 * torch.pi * x[:, None] - .2)).expand(n, n)
    eq, geo = Equation(.014, 2.), Geometry((n, n), (1., 1.))
    a, b = [interaction_channels((.4 + lo + sign * hi)[None, None], .08, eq, geo,
                                track=track, split_modes=1) for sign in (1, -1)]
    torch.testing.assert_close(a[:, 0], b[:, 0], atol=2e-18, rtol=3e-10)
    torch.testing.assert_close(a[:, 2], b[:, 2], atol=2e-18, rtol=3e-10)
    torch.testing.assert_close(a[:, 1], -b[:, 1], atol=2e-18, rtol=3e-10)
    coefficient = torch.fft.fft2(a, norm="forward")[0, 1, 2, 0]
    assert coefficient.abs() > 1e-8 and abs(float(coefficient.imag)) > 1e-9


def test_adjacent_nodal_and_galerkin_products_are_distinct():
    n = 12
    x = torch.arange(n, dtype=torch.float64) / n
    u = (.4 + .08 * torch.cos(10 * torch.pi * x[:, None])).expand(n, n)[None, None]
    eq, geo = Equation(.008, 2.), Geometry((n, n), (1., 1.))
    discrete, continuum = [interaction_channels(u, .1, eq, geo, track=t, split_modes=1) for t in numerics.TRACKS]
    # A nodal 5+5 product aliases to mode -2; Galerkin does not retain it.
    a, b = [torch.fft.fft2(c[:, 2:3], norm="forward")[0, 0, 2, 0].abs() for c in (discrete, continuum)]
    assert a > 1e-6 and b < 1e-14


@pytest.mark.parametrize("track", numerics.TRACKS)
def test_adjacent_translation_batch_and_operational_precision(track):
    u = torch.cat((state(), state() * .8))
    h, eq, geo = torch.tensor([.03, .07], dtype=u.dtype), Equation(.012, 2.3), Geometry((12, 12), (1., 1.))
    value = model("channel_neural", track)
    with torch.no_grad():
        value.conditioner[-1].bias.copy_(torch.tensor([.3, -.2, .7]))
    batched = value(u, h, eq, geo)
    separate = torch.cat([value(u[i:i+1], h[i], eq, geo) for i in range(2)])
    torch.testing.assert_close(batched, separate, atol=2e-15, rtol=1e-12)
    torch.testing.assert_close(value(torch.roll(u, (2, -1), (-2, -1)), h, eq, geo),
                               torch.roll(batched, (2, -1), (-2, -1)), atol=2e-15, rtol=1e-12)
    torch.testing.assert_close(deepcopy(value).float()(u.float(), h.float(), eq, geo).double(), batched,
                               atol=7e-7, rtol=5e-6)


@pytest.mark.parametrize("family", ("channel_neural", "feature_capacity"))
@pytest.mark.parametrize("track", numerics.TRACKS)
def test_adjacent_nonzero_parameter_state_and_time_gradients(family, track):
    value = model(family, track)
    with torch.no_grad():
        for p in value.parameters():
            p.copy_(torch.randn_like(p) * .2)
    u = state().requires_grad_()
    h = torch.tensor(.05, dtype=u.dtype, requires_grad=True)
    eq, geo = Equation(.012, 2.3), Geometry((12, 12), (1., 1.))
    output = value(u, h, eq, geo).square().mean()
    gradients = torch.autograd.grad(output, (u, h, *value.parameters()))
    assert all(torch.isfinite(g).all() and g.abs().max() > 0 for g in gradients)
    eps = 1e-5
    finite = (value(u, h.detach() + eps, eq, geo).square().mean() - value(u, h.detach() - eps, eq, geo).square().mean()) / (2 * eps)
    torch.testing.assert_close(gradients[1], finite, atol=1e-9, rtol=2e-7)


@pytest.mark.parametrize("track", numerics.TRACKS)
def test_adjacent_each_channel_small_step_order_and_cutoff(track):
    u, eq, geo = state(), Equation(.012, 2.3), Geometry((12, 12), (1., 1.))
    one = interaction_channels(u, .001, eq, geo, track=track, split_modes=1)
    two = interaction_channels(u, .002, eq, geo, track=track, split_modes=1)
    ratios = two.flatten(2).norm(dim=2) / one.flatten(2).norm(dim=2)
    assert bool(((ratios > 7.5) & (ratios < 8.5)).all())
    compressed = interaction_channels(u, .04, eq, geo, track=track, split_modes=1, output_modes=2)
    torch.testing.assert_close(lowpass(compressed, 2), compressed, atol=2e-18, rtol=2e-10)


@pytest.mark.parametrize("family", ("channel_global", "channel_affine"))
def test_adjacent_training_only_fit_recovers_small_channel_rule(family):
    candidate, truth = model(family), model(family)
    with torch.no_grad():
        if family == "channel_global":
            truth.gain_logits.copy_(torch.tensor([.12, -.07, .09]))
        else:
            truth.affine_coefficients.copy_(torch.tensor([[.08, -.06, .04], [.03, .02, -.03],
                                                         [-.02, .01, .04], [.01, -.01, .02]]))
    geo, samples = Geometry((12, 12), (1., 1.)), []
    for index in range(16):
        u = state() + (.1 * (index % 4) - .15)
        h, eq = .025 + .035 * (index // 4), Equation(.004 + .005 * (index % 3), 1. + .5 * (index % 5))
        samples.append((u, h, eq, geo, truth(u, h, eq, geo)))
    report = candidate.fit_least_squares(samples, ridge=1e-10, max_iterations=180)
    assert report["fitted_weighted_sse"] < report["initial_weighted_sse"] * 2e-5
    assert report["matched_objective"] and not report["reference_inputs_during_inference"]
    assert report["rank"] == report["columns"]
    assert candidate.parameter_report()["fitted_buffer_coefficients"] in (3, 12)


def test_adjacent_bounds_and_existing_models_preserved():
    assert gain_bounds("quad2_conditioned") == (.25, 1.75)
    assert gain_bounds("band_gain") == (.5, 1.5)
    assert gain_bounds("residual_quad2") == (.75, 1.25)
    assert gain_bounds("historical_half") == (.5, .5)
    assert gain_bounds("fno_standard") is None
    value = make_model("quad2_fixed", config=dict(modes=3, split_modes=1, output_compression=False))
    assert value.family == "quad2_fixed" and value.spec["compression"] == "output"
    with pytest.raises(ValueError):
        interaction_channels(state(), .03, Equation(.01, 2.), Geometry((12, 12), (1., 1.)), implementation="wrong")
