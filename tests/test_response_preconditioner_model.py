"""Structural tests of the complete fixed-feature map, not science gates."""
import pytest
import torch

from tdn.analysis.response import numerics as n
from tdn.analysis.response.contracts import Physics
from tdn.analysis.response_preconditioner.model import FixedFeatureResponse
from tdn.analysis.response_preconditioner.numerics import ordered_response
from tdn.analysis.roadmap.numerics import quadrature
from tdn.analysis.tangent_response.tangent import AnchorPath
from tdn.numerics.types import Geometry


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def case():
    x = torch.arange(5, dtype=torch.float64) * (2 * torch.pi / 5)
    u = (.35 + .07 * torch.sin(x))[None, None]
    z = (torch.cos(x) + .2 * torch.sin(2 * x))[None, None]
    return u, z, Physics(.02, 1.3), Geometry((5,), (1.,))


def nonzero_model():
    model = FixedFeatureResponse()
    model.set_readout(torch.linspace(-.12, .17, 40, dtype=torch.float64))
    return model


def test_zero_readout_matches_independent_classical_corrected_baseline():
    u, _, eq, geom = case()
    model = FixedFeatureResponse()
    path = AnchorPath(u, eq, geom, "fd_same_grid", create_graph=True)
    h = .07
    expected = path(h)
    for x, weight in zip(*quadrature(3)):
        s = h * x
        expected = expected - h * weight * ordered_response(
            path, path.evaluate(s)["residual"], s, h, eq, geom)
    actual = model(u, h, eq, geom)
    torch.testing.assert_close(actual, expected, atol=2e-15, rtol=2e-15)
    assert model.readout.count_nonzero() == 0
    assert model.architecture_metadata()["correction_stages"] == 1


def test_action_and_endpoint_design_are_exact_affine_maps():
    u, z, eq, geom = case()
    model = nonzero_model()
    for base, design, actual in (
        (*model.action_design(u, z, .01, .06, eq, geom),
         model.response_action(u, z, .01, .06, eq, geom)),
        (*model.endpoint_design(u, .06, eq, geom), model(u, .06, eq, geom)),
    ):
        assert design.shape == (40, *u.shape)
        torch.testing.assert_close(actual, base + torch.einsum("p,p...->...", model.readout, design),
                                   atol=2e-15, rtol=2e-15)
    base, design = model.endpoint_design(u, .06, eq, geom)
    loss = model(u, .06, eq, geom).square().sum()
    gradient, = torch.autograd.grad(loss, model.readout)
    expected = 2 * (design * (base + torch.einsum("p,p...->...", model.readout, design))).flatten(1).sum(1)
    torch.testing.assert_close(gradient, expected, atol=1e-17, rtol=1e-11)


def test_response_is_linear_in_unseen_sources_with_nonzero_readout():
    u, z, eq, geom = case()
    model = nonzero_model()
    other = torch.roll(z, 1, -1) + .17
    action = lambda source: model.response_action(u, source, .01, .09, eq, geom)
    torch.testing.assert_close(action(1.7*z - .9*other), 1.7*action(z) - .9*action(other),
                               atol=2e-15, rtol=3e-14)
    torch.testing.assert_close(action(torch.zeros_like(z)), torch.zeros_like(z), atol=0, rtol=0)


def test_complete_input_jvp_matches_central_difference():
    u, direction, eq, geom = case()
    model = nonzero_model()
    step = lambda state: model(state, .09, eq, geom)
    _, jvp = torch.autograd.functional.jvp(step, u, direction)
    eps = 2e-5
    numerical = (step(u + eps*direction) - step(u - eps*direction)) / (2*eps)
    torch.testing.assert_close(jvp, numerical, atol=2e-9, rtol=2e-8)
    # The actual step Jacobian includes input/path dependence and is not merely
    # the fixed-anchor response action on the same direction.
    internal = model.response_action(u, direction, 0., .09, eq, geom)
    assert (jvp - internal).abs().max() > 1e-9


def test_complete_tensor_physics_derivative_matches_difference():
    u, _, _, geom = case()
    model = nonzero_model()
    r = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
    objective = lambda rate: model(u, .08, Physics(.02, rate), geom).square().sum()
    derivative, = torch.autograd.grad(objective(r), r)
    eps = 1e-5
    numerical = (objective(r.detach()+eps) - objective(r.detach()-eps)) / (2*eps)
    torch.testing.assert_close(derivative, numerical, atol=2e-10, rtol=2e-8)


def test_input_change_recomputes_factors_and_path():
    u, z, eq, geom = case()
    model = nonzero_model()
    first = model.response_action(u, z, .01, .1, eq, geom)
    u.add_(.08)
    second = model.response_action(u, z, .01, .1, eq, geom)
    independent = nonzero_model().response_action(u.clone(), z, .01, .1, eq, geom)
    assert (first-second).abs().max() > 1e-5
    torch.testing.assert_close(second, independent, atol=0, rtol=0)


def test_zero_time_identity_and_complete_right_derivative():
    u, _, eq, geom = case()
    model = nonzero_model()
    h = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    weights = torch.linspace(.2, .8, u.numel(), dtype=u.dtype).reshape_as(u)
    actual = model(u, h, eq, geom)
    torch.testing.assert_close(actual, u, atol=0, rtol=0)
    derivative, = torch.autograd.grad((actual*weights).sum(), h)
    torch.testing.assert_close(derivative, (n.rhs(u, eq, geom, "fd_same_grid")*weights).sum(),
                               atol=2e-14, rtol=2e-13)


@pytest.mark.parametrize("null", ("heat", "reaction", "constant"))
def test_full_step_exact_limits_to_float64_roundoff(null):
    u, _, eq, geom = case()
    if null == "heat":
        eq = Physics(eq.kappa, 0.)
    elif null == "reaction":
        eq = Physics(0., eq.reaction_rate)
    else:
        u = torch.full_like(u, .35)
    model = nonzero_model()
    expected = n.anchor(u, .1, eq, geom, "fd_same_grid", name="df")
    torch.testing.assert_close(model(u, .1, eq, geom), expected, atol=2e-14, rtol=2e-14)


def test_batch_matches_separate_parents_and_translation():
    u, z, eq, geom = case()
    model = nonzero_model()
    inputs = torch.cat((u, u + .1))
    sources = torch.cat((z, torch.roll(z, 1, -1)))
    result = model.response_action(inputs, sources, .01, .07, eq, geom)
    separate = torch.cat([model.response_action(inputs[i:i+1], sources[i:i+1], .01, .07, eq, geom)
                          for i in range(2)])
    torch.testing.assert_close(result, separate, atol=2e-15, rtol=2e-14)
    shifted = model(torch.roll(u, 2, -1), .07, eq, geom)
    torch.testing.assert_close(shifted, torch.roll(model(u, .07, eq, geom), 2, -1), atol=2e-15, rtol=2e-14)


def test_batched_tensor_physics_and_times_match_separate_evaluations():
    u, z, _, geom = case()
    model = nonzero_model()
    inputs = torch.cat((u, u + .1))
    sources = torch.cat((z, torch.roll(z, 1, -1)))
    kappas = torch.tensor([.02, .008], dtype=torch.float64)
    rates = torch.tensor([1.3, 2.1], dtype=torch.float64)
    starts = torch.tensor([.01, .02], dtype=torch.float64)
    stops = torch.tensor([.06, .09], dtype=torch.float64)
    batched = model.response_action(inputs, sources, starts, stops, Physics(kappas, rates), geom)
    independent = torch.cat([model.response_action(inputs[i:i+1], sources[i:i+1], starts[i], stops[i],
                                Physics(kappas[i], rates[i]), geom) for i in range(2)])
    torch.testing.assert_close(batched, independent, atol=2e-15, rtol=2e-14)


@pytest.mark.parametrize("options", ({"panels": 3}, {"potential_nodes": 8},
                                    {"time_reference": 0.}, {"output_precision": "half"}))
def test_invalid_work_and_precision_configuration_rejected_early(options):
    with pytest.raises(ValueError):
        FixedFeatureResponse(**options)


def test_work_and_explicit_output_precision():
    u, _, eq, geom = case()
    work = {}
    model = FixedFeatureResponse(output_precision="input")
    output = model(u.float(), .07, eq, geom, work=work)
    assert output.dtype == torch.float32
    assert work["heat_actions"] == 12
    assert work["fft_transforms"] == 24
    assert work["laplacian_actions"] == 21
    assert work["source_quadrature_nodes"] == 3
    assert work["path_anchor_evaluations"] > 0
    assert work["path_derivative_evaluations"] > 0
    assert work["fixed_feature_encodings"] == 1
    assert work["temporal_feature_calls"] == 3
    baseline, design = model.endpoint_design(u.float(), .07, eq, geom)
    assert baseline.dtype == design.dtype == torch.float64


@pytest.mark.parametrize("bad", ("target", "nan", "reverse", "shape", "readout", "model_precision"))
def test_invalid_inputs_are_rejected(bad):
    u, z, eq, geom = case()
    model = FixedFeatureResponse()
    with pytest.raises(ValueError):
        if bad == "target":
            model(u, .1, eq, geom, target="galerkin_same_grid")
        elif bad == "nan":
            model(u*float("nan"), .1, eq, geom)
        elif bad == "reverse":
            model.response_action(u, z, .2, .1, eq, geom)
        elif bad == "shape":
            model.response_action(u, z[:, :, :-1], 0., .1, eq, geom)
        elif bad == "readout":
            model.set_readout([float("nan")]*40)
        else:
            model.float()(u, .1, eq, geom)
