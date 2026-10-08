"""Mathematical and comparator-integrity regressions for the focused program."""
import math
import pytest
import torch

from tdn.analysis.frontier.models import FAMILIES, TRAINABLE_FAMILIES, RANK_FAMILIES, make_model
from tdn.analysis.frontier import numerics
from tdn.numerics import Equation, Geometry


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def state(n=8, dtype=torch.float64, device="cpu"):
    x = torch.arange(n, dtype=dtype, device=device) / n
    return (.43 + .08 * torch.cos(2 * torch.pi * x[:, None] + .2)
            + .03 * torch.sin(4 * torch.pi * x[None, :] - .4))[None, None]


def model(family, track, **kw):
    torch.manual_seed(170031)
    return make_model(family, track, {"width": 3, "depth": 2, "modes": 2, "quad_nodes": 2,
                                    "cubic_nodes": 2, "reaction_substeps": 2, **kw}).double()


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("track", numerics.TRACKS)
def test_frontier_every_family_shape_identity_and_metadata(track, family):
    value = model(family, track)
    u, eq, geo = state(), Equation(.003, 2.), Geometry((8, 8), (1., 1.))
    actual = value(u, .03, eq, geo)
    assert actual.shape == u.shape and actual.dtype == u.dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(value(u, 0., eq, geo), u, atol=0., rtol=0.)
    meta = value.architecture_metadata()
    assert meta["track"] == track and not meta["published_FNO_reproduction"]
    assert not meta["clipping"] and not meta["continuum_truth_claim"]
    assert bool(meta["trainable_parameters"]) == (family in TRAINABLE_FAMILIES)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("track", numerics.TRACKS)
def test_frontier_fp32_matches_fp64_at_operational_precision(track, family):
    from copy import deepcopy
    reference = model(family, track)
    operational = deepcopy(reference).float()
    eq, geo, u = Equation(.003, 2.), Geometry((8, 8), (1., 1.)), state()
    torch.testing.assert_close(operational(u.float(), .03, eq, geo).double(),
                               reference(u, .03, eq, geo), atol=4e-7, rtol=3e-6)


@pytest.mark.parametrize("family", RANK_FAMILIES + ("analytic_quad", "analytic_quad_cubic"))
@pytest.mark.parametrize("track", numerics.TRACKS)
def test_frontier_correction_physical_nulls_after_arbitrary_weights(track, family):
    value = model(family, track)
    with torch.no_grad():
        for p in value.parameters():
            p.copy_(torch.randn_like(p) * .4)
    geo = Geometry((8, 8), (1., 1.))
    for u, eq, h in [(state(), Equation(.003, 2.), 0.),
                      (state(), Equation(0., 2.), .04),
                      (state(), Equation(.003, 0.), .04),
                      (torch.full_like(state(), .7), Equation(.003, 2.), .04)]:
        parts = value.correction_components(u, h, eq, geo)
        torch.testing.assert_close(parts["increment"], torch.zeros_like(u), atol=2e-15, rtol=0.)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("track", numerics.TRACKS)
def test_frontier_translation_equivariance(track, family):
    value = model(family, track)
    eq, geo, u = Equation(.003, 2.), Geometry((8, 8), (1., 1.)), state()
    actual = value(torch.roll(u, (2, -3), (-2, -1)), .04, eq, geo)
    expected = torch.roll(value(u, .04, eq, geo), (2, -3), (-2, -1))
    torch.testing.assert_close(actual, expected, atol=2e-13, rtol=2e-12)


@pytest.mark.parametrize("family", TRAINABLE_FAMILIES)
@pytest.mark.parametrize("track", numerics.TRACKS)
def test_frontier_input_time_and_parameter_gradients(track, family):
    value = model(family, track)
    u = state().requires_grad_()
    step = torch.tensor(.03, dtype=torch.float64, requires_grad=True)
    out = value(u, step, Equation(.003, 2.), Geometry((8, 8), (1., 1.)))
    grads = torch.autograd.grad(out.square().mean(), (u, step, *value.parameters()), allow_unused=True)
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert grads[0].abs().max() > 0 and grads[1].abs() > 0
    assert any(g.abs().max() > 0 for g in grads[2:])
    if family.startswith("fno") or family == "direct_fno":
        # No zero-initialized head blocking the competitors' first update.
        assert all(g.abs().max() > 0 for g in grads[2:])


@pytest.mark.parametrize("track", numerics.TRACKS)
def test_frontier_rank_input_and_time_derivatives_match_finite_differences(track):
    value = model("rank1", track, modes=1)
    eq, geo = Equation(.003, 2.), Geometry((4, 4), (1., 1.))
    u = state(n=4).requires_grad_()
    h = torch.tensor(.03, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x, t: value(x, t, eq, geo), (u, h),
                                   eps=1e-6, atol=3e-6, rtol=2e-4)


@pytest.mark.parametrize("track", numerics.TRACKS)
def test_frontier_rank_frozen_and_precompression_are_identical_initializations(track):
    trained, frozen, post = (model(name, track) for name in ("rank1", "rank1_frozen", "rank1_postcompression"))
    for p, q, r in zip(trained.parameters(), frozen.parameters(), post.parameters()):
        torch.testing.assert_close(p, q, atol=0., rtol=0.)
        torch.testing.assert_close(p, r, atol=0., rtol=0.)
    assert all(not p.requires_grad for p in frozen.parameters())
    assert sum(p.numel() for p in trained.parameters()) == 43
    u, eq, geo = state(), Equation(.003, 2.), Geometry((8, 8), (1., 1.))
    torch.testing.assert_close(trained(u, .03, eq, geo), frozen(u, .03, eq, geo), atol=0., rtol=0.)
    torch.testing.assert_close(trained.correction_components(u, .03, eq, geo)["base"],
                               post.correction_components(u, .03, eq, geo)["base"], atol=0., rtol=0.)


@pytest.mark.parametrize("track", numerics.TRACKS)
def test_frontier_high_pair_difference_retained_before_compression(track):
    n = 16
    x = torch.arange(n, dtype=torch.float64) / n
    u = (.45 + .08 * torch.cos(10 * torch.pi * x[:, None])
         + .06 * torch.cos(12 * torch.pi * x[:, None])).expand(n, n)[None, None]
    eq, geo = Equation(.004, 2.), Geometry((n, n), (1., 1.))
    before = model("rank1", track).correction_components(u, .04, eq, geo)["increment"]
    after = model("rank1_postcompression", track).correction_components(u, .04, eq, geo)["increment"]
    assert torch.fft.fft2(before)[..., 1, 0].abs().max() > 1e-8
    assert after.abs().max() < 1e-14
    flipped = (.45 + .08 * torch.cos(10 * torch.pi * x[:, None])
               - .06 * torch.cos(12 * torch.pi * x[:, None])).expand(n, n)[None, None]
    negative = model("rank1", track).correction_components(flipped, .04, eq, geo)["increment"]
    torch.testing.assert_close(torch.fft.fft2(negative)[..., 1, 0], -torch.fft.fft2(before)[..., 1, 0], atol=1e-13, rtol=1e-9)


@pytest.mark.parametrize("track", numerics.TRACKS)
def test_frontier_rank_quadratic_amplitude_and_cubic_time_order(track):
    value = model("rank1", track)
    eq, geo = Equation(.003, 2.), Geometry((8, 8), (1., 1.))
    u = state()
    center = u.mean()
    delta = lambda a, h: value.correction_components(center + a * (u - center), h, eq, geo)["increment"]
    torch.testing.assert_close(delta(2., .03), 4 * delta(1., .03), atol=3e-15, rtol=2e-8)
    torch.testing.assert_close(delta(-1., .03), delta(1., .03), atol=3e-15, rtol=2e-8)
    ratio = float((delta(1., .004).norm() / delta(1., .002).norm()).detach())
    assert 7.5 < ratio < 8.5


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("track", numerics.TRACKS)
def test_frontier_batched_h_matches_individual_calls(track, family):
    value = model(family, track)
    u = torch.cat((state(), .9 * state()))
    eq, geo = Equation(.003, 2.), Geometry((8, 8), (1., 1.))
    h = torch.tensor([.02, .04], dtype=torch.float64)
    actual = value(u, h, eq, geo)
    expected = torch.cat([value(u[i:i+1], h[i], eq, geo) for i in range(2)])
    torch.testing.assert_close(actual, expected, atol=2e-13, rtol=2e-11)


@pytest.mark.parametrize("track", numerics.TRACKS)
def test_frontier_etdrk4_order_and_exact_diffusion(track):
    eq, geo, initial = Equation(.003, 2.), Geometry((8, 8), (1., 1.)), state()
    def solve(count):
        u = initial.clone()
        for _ in range(count):
            u = numerics.etdrk4_step(u, .2 / count, eq, geo, track)
        return u
    truth = solve(64)
    errors = [float((solve(n) - truth).norm()) for n in (1, 2, 4)]
    assert errors[0] / errors[1] > 10 and errors[1] / errors[2] > 10
    no_reaction = Equation(eq.kappa, 0.)
    torch.testing.assert_close(numerics.etdrk4_step(initial, .2, no_reaction, geo, track),
                               numerics.heat_step(initial, .2, no_reaction, geo, track), atol=0., rtol=0.)


def test_frontier_discrete_etdrk4_matches_independently_existing_control():
    from tdn.research.interaction import etdrk4_step
    eq, geo, u = Equation(.003, 2.), Geometry((8, 8), (1., 1.)), state()
    torch.testing.assert_close(numerics.etdrk4_step(u, .07, eq, geo), etdrk4_step(u, .07, eq, geo), atol=2e-15, rtol=2e-14)


def test_frontier_continuum_product_rejects_aliased_harmonic():
    n = 16
    x = torch.arange(n, dtype=torch.float64) / n
    u = torch.cos(12 * torch.pi * x[:, None]).expand(n, n)[None, None]
    product = numerics.product(u, u, "continuum")
    torch.testing.assert_close(product, torch.full_like(product, .5), atol=3e-15, rtol=0.)
    assert float((numerics.product(u, u, "discrete") - product).norm()) > 1.


def test_frontier_continuum_core_has_galerkin_generator_not_sampled_logistic():
    n = 16
    x = torch.arange(n, dtype=torch.float64) / n
    u = (.45 + .2 * torch.cos(12 * torch.pi * x[:, None])).expand(n, n)[None, None]
    eq, geo, h = Equation(0., 2.), Geometry((n, n), (1., 1.)), 1e-5
    generator = (numerics.df_step(u, h, eq, geo, "continuum") - u) / h
    torch.testing.assert_close(generator, numerics.reaction_rhs(u, eq, "continuum"), atol=3e-6, rtol=2e-5)
    assert (generator - numerics.reaction_rhs(u, eq, "discrete")).abs().max() > .03


@pytest.mark.parametrize("track", numerics.TRACKS)
def test_frontier_analytic_amplitude_hierarchy_matches_independent_teacher(track):
    from tdn.analysis.frontier.data import lawson_reference
    n = 8
    x = torch.arange(n, dtype=torch.float64) / n
    direction = (torch.cos(2 * torch.pi * x[:, None]) + .3 * torch.cos(4 * torch.pi * x[None, :])
                 + .13 * torch.sin(6 * torch.pi * x[:, None]))[None, None]
    eq, geo = Equation(.02, 1.3), Geometry((n, n), (1., 1.))
    # Resolve reaction integration and quadrature independently so this tests
    # the spatially projected coefficient hierarchy, not a numerical error floor.
    quadratic = model("analytic_quad", track, quad_nodes=8, reaction_substeps=64)
    cubic = model("analytic_quad_cubic", track, quad_nodes=8, cubic_nodes=8, reaction_substeps=64)
    errors = []
    for amplitude in (.1, .05, .025):
        initial = .4 + amplitude * direction
        teacher = lawson_reference(initial, .15, eq, geo, 256, track)
        errors.append([float((teacher - solver(initial, .15, eq, geo)).abs().max())
                       for solver in (quadratic, cubic)])
    assert min(math.log2(errors[i][0] / errors[i + 1][0]) for i in (0, 1)) > 2.9
    assert min(math.log2(errors[i][1] / errors[i + 1][1]) for i in (0, 1)) > 3.8
    assert errors[0][1] < errors[0][0] / 10


@pytest.mark.parametrize("track", numerics.TRACKS)
def test_frontier_operator_cache_keys_gradients_and_cold_reset(track):
    value = model("etdrk4", track)
    u, eq, geo = state(), Equation(.003, 2.), Geometry((8, 8), (1., 1.))
    first = value(u, .03, eq, geo)
    second = value(u + .01, .03, eq, geo)
    assert value.cache_metadata["coefficient_preparations"] == 1 and value.cache_metadata["cache_hits"] == 1
    independent = model("etdrk4", track)(u + .01, .03, eq, geo)
    torch.testing.assert_close(second, independent, atol=0., rtol=0.)
    value(u, .04, eq, geo)
    value(u, .03, Equation(.005, 2.), geo)
    value(u, .03, eq, Geometry((8, 8), (2., 1.)))
    assert value.cache_metadata["coefficient_preparations"] == 4
    h = torch.tensor(.03, dtype=u.dtype, requires_grad=True)
    derivative = torch.autograd.grad(value(u, h, eq, geo).square().mean(), h)[0]
    assert torch.isfinite(derivative) and derivative.abs() > 0
    assert value.cache_metadata["gradient_or_batch_bypasses"] == 1
    value.float()
    assert value.cache_metadata["cache_entries"] == 0
    assert torch.isfinite(first).all()


@pytest.mark.parametrize("track", numerics.TRACKS)
def test_frontier_no_unadvertised_output_clipping(track):
    value = model("direct_fno", track)
    with torch.no_grad():
        value.head[-1].bias.fill_(100.)
    result = value(state(), .1, Equation(.003, 2.), Geometry((8, 8), (1., 1.)))
    assert result.max() > 1.


@pytest.mark.parametrize("h", [-1., float("nan"), float("inf")])
def test_frontier_rejects_invalid_horizons(h):
    with pytest.raises(ValueError, match="nonnegative"):
        model("rank1", "discrete")(state(), h, Equation(.003, 2.), Geometry((8, 8), (1., 1.)))


def test_frontier_rejects_unknown_architecture_configuration():
    with pytest.raises(ValueError, match="Unknown"):
        make_model("rank1", "discrete", {"silent_unused_knob": 9})
