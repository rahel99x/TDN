"""Question-bearing checks for the new physically factored trained arms."""
import math

import pytest
import torch

from tdn.numerics import Equation, Geometry, rhs
from tdn.research.agenda_neural import physical_base
from tdn.analysis.roadmap.models import FAMILIES, FiniteStepRank, build_model, fractional_feature


def field(n=8, dtype=torch.float64):
    x = torch.arange(n, dtype=dtype) / n
    return (.45 + .09 * torch.cos(2 * torch.pi * x[:, None]) + .04 * torch.sin(4 * torch.pi * x[None, :]))[None, None]


@pytest.mark.parametrize("family", FAMILIES)
def test_roadmap_model_arbitrary_weights_physical_correction_nulls(family):
    torch.manual_seed(12)
    model = build_model(family, {"width": 3, "modes": 1}).double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-.4, .4)
    u, geometry = field(), Geometry((8, 8), (1., 1.))
    cases = [(u, 0., Equation(.003, 2.)), (u, .1, Equation(0., 2.)),
             (u, .1, Equation(.003, 0.)), (torch.full_like(u, .4), .1, Equation(.003, 2.))]
    if model.direct:
        cases = cases[:1]
    for state, h, equation in cases:
        components = model.correction_components(state, h, equation, geometry)
        assert float(components["increment"].detach().abs().max()) < 2.e-14
        actual = model(state, h, equation, geometry)
        assert torch.allclose(actual, components["base"], atol=2.e-14, rtol=0)


@pytest.mark.parametrize("family", FAMILIES)
def test_roadmap_model_generator_and_live_parameter_gradients(family):
    torch.manual_seed(2)
    model = build_model(family, {"width": 3, "modes": 1}).double()
    u, geometry, equation = field(), Geometry((8, 8), (1., 1.)), Equation(.004, 2.)
    h = torch.zeros((), dtype=u.dtype)
    _, derivative = torch.func.jvp(lambda time: model(u, time, equation, geometry), (h,), (torch.ones_like(h),))
    if not model.direct:
        assert torch.allclose(derivative, rhs(u, equation, geometry), atol=2.e-12, rtol=2.e-11)
    if list(model.parameters()):
        value = model(u, .12, equation, geometry)
        loss = ((value - .53) * (u - .3)).square().mean()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads) > 0


@pytest.mark.parametrize("rank", [1, 2])
def test_symmetric_rank_response_is_quadratic_and_third_order(rank):
    model = FiniteStepRank(rank).double()
    with torch.no_grad():
        model.amplitudes.fill_(.4)
    u, geometry, equation = field(), Geometry((8, 8), (1., 1.)), Equation(.004, 2.)
    v = u - u.mean()
    a = model(.45 + .5 * v, .04, equation, geometry)
    b = model(.45 + .25 * v, .04, equation, geometry)
    assert torch.allclose(a, 4 * b, atol=1.e-14, rtol=1.e-8)
    values = [float(model(u, h, equation, geometry).detach().square().mean().sqrt()) for h in (.002, .001)]
    assert 2.85 < math.log2(values[0] / values[1]) < 3.15
    assert values[0] > 0, "Rank one must not collapse to the identically-zero midpoint defect"


def test_fractional_feature_uses_physical_spacing_not_training_grid():
    outputs = []
    for n in (16, 32):
        x = torch.arange(n, dtype=torch.float64) / n
        state = torch.cos(2 * torch.pi * x)[:, None].expand(n, n)[None, None]
        geometry = Geometry((n, n), (2., 2.))
        result = fractional_feature(state, geometry, torch.tensor(1.3), length_scale=2.)
        expected = (2 * math.pi)**1.3 * state
        assert torch.allclose(result, expected, atol=2.e-6, rtol=2.e-6)
        outputs.append(result)
    assert torch.allclose(outputs[0], outputs[1][..., ::2, ::2], atol=2.e-12, rtol=2.e-12)
    constant = torch.ones((1, 1, 16, 16), dtype=torch.float64)
    assert torch.count_nonzero(fractional_feature(constant, Geometry((16, 16), (1., 1.)), .8)) == 0


def test_pure_compression_control_has_no_full_source_bypass():
    n = 16
    x = torch.arange(n, dtype=torch.float64) / n
    u = (.4 + .15 * torch.cos(10 * torch.pi * x[:, None]) * torch.ones_like(x)[None, :])[None, None]
    eq, geom = Equation(.001, 2.), Geometry((n, n), (1., 1.))
    before = build_model("source", {"width": 2, "modes": 1}).double()
    after = build_model("source_postcompression", {"width": 2, "modes": 1}).double()
    with torch.no_grad():
        for parameter in before.parameters():
            parameter.fill_(.2)
    after.load_state_dict(before.state_dict())
    source_response = before.correction_components(u, .1, eq, geom)["increment"]
    compressed_response = after.correction_components(u, .1, eq, geom)["increment"]
    assert float(source_response.detach().abs().max()) > 1.e-8
    assert float(compressed_response.detach().abs().max()) < 1.e-25


def test_trust_uses_active_physical_frequency_and_preserves_low_output_mean():
    model = build_model("source_trust", {"width": 2, "modes": 1}).double()
    eq = Equation(.004, 2.)
    envelopes = []
    for n in (16, 32):
        u = field(n)
        envelopes.append(float(model.trust_envelope(u, torch.tensor(.1), eq, Geometry((n, n), (1., 1.)))))
    assert abs(envelopes[0] - envelopes[1]) < .01
    assert 0 < envelopes[0] < 2


def test_c1_and_c2_have_real_combined_response_and_distinct_envelope():
    eq, geom, u = Equation(.01, 4.), Geometry((8, 8), (1., 1.)), field()
    c1 = build_model("c1_rank2", {"width": 2, "modes": 1, "trust_horizon": .05}).double()
    c2 = build_model("c2_rank2", {"width": 2, "modes": 1, "trust_horizon": .05}).double()
    c2.load_state_dict(c1.state_dict())
    first = c1.correction_components(u, .2, eq, geom)["increment"]
    second = c2.correction_components(u, .2, eq, geom)["increment"]
    assert float(first.detach().abs().max()) > 0
    assert float(second.detach().abs().max()) < float(first.detach().abs().max())
    assert c2.architecture_metadata()["combination_ids"] == ["C1", "C2"]


def test_c1_rank_ablations_share_identical_source_initialization():
    models = []
    for rank in (0, 1, 2):
        torch.manual_seed(511)
        models.append(build_model(f"c1_rank{rank}", {"width": 3, "modes": 1}))
    for key, value in models[0].state_dict().items():
        assert torch.equal(value, models[1].state_dict()[key])
        assert torch.equal(value, models[2].state_dict()[key])


def test_deep_fno_reports_real_nearest_parameter_count_without_compute_claim():
    source = build_model("source", {"width": 12, "modes": 4})
    deep = build_model("deep_fno", {"width": 12, "modes": 4})
    metadata = deep.architecture_metadata()
    assert metadata["parameter_matching_target"] == sum(p.numel() for p in source.parameters())
    assert metadata["parameter_matching_relative_error"] < .02
    assert metadata["spectral_depth"] == 4 and not metadata["compute_matching_claim"]
