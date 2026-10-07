"""Physical limits and representational changes of the audit architectures."""
from __future__ import annotations

import io
import json
from copy import deepcopy

import pytest
import torch

from tdn.numerics import Equation, Geometry
from tdn.numerics.operators import rhs
from tdn.numerics.subflows import reaction_step
from tdn.research.agenda_neural import (
    FAMILIES, ORIENTATIONS, MeanVarianceClosure, build_model, centered_state,
    parameter_matched_width, physical_base, physical_moments,
)
from tdn.research.consistency_neural import build_model as old_model


def fixture(grid=(8, 10), dtype=torch.float64, batch=2):
    x = torch.arange(grid[0], dtype=dtype) * 2 * torch.pi / grid[0]
    y = torch.arange(grid[1], dtype=dtype) * 2 * torch.pi / grid[1]
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    pattern = .08 * torch.cos(2 * xx + .2) + .025 * torch.sin(3 * yy - .3)
    u = torch.stack((.3 + pattern, .7 - .6 * pattern))[:batch, None]
    return u, Equation(.013, 3.2), Geometry(grid, (1.2, .8))


def learned(family, dtype=torch.float64, orientation="reaction-first", width=2, modes=2):
    torch.manual_seed(914023)
    model = build_model(family, width=width, modes=modes, t_ref=.2,
                        base_orientation=orientation).to(dtype=dtype)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(.2 * torch.randn_like(parameter))
    return model


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_zero_amplitudes_are_the_same_physical_initialization(family, orientation):
    u, equation, geometry = fixture()
    model = build_model(family, width=2, modes=2, t_ref=.2,
                        base_orientation=orientation).double()
    actual = model(u, .1, equation, geometry)
    expected = physical_base(u, .1, equation, geometry, orientation)
    assert torch.equal(actual, expected)
    assert torch.equal(model.step(u, .1, equation, geometry), expected)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("orientation", ORIENTATIONS)
@pytest.mark.parametrize("limit", ("constant", "reaction", "diffusion", "time"))
def test_arbitrary_weights_preserve_exact_physical_correction_limits(family, orientation, limit):
    model = learned(family, orientation=orientation)
    u, equation, geometry = fixture()
    h = .08
    if limit == "constant":
        u = torch.cat((torch.full_like(u[:1], .17), torch.full_like(u[:1], .83)))
    elif limit == "reaction":
        equation = Equation(equation.kappa, 0.)
    elif limit == "diffusion":
        equation = Equation(0., equation.reaction_rate)
    else:
        h = 0.
    components = model.correction_components(u, h, equation, geometry)
    assert torch.count_nonzero(components["spatial_increment"]) == 0
    if "mean_increment" in components:
        assert torch.count_nonzero(components["mean_increment"]) == 0
    assert torch.equal(model(u, h, equation, geometry),
                       physical_base(u, h, equation, geometry, orientation))


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_complete_maps_are_bounded_differentiable_parent_independent_and_equivariant(family, dtype):
    model = learned(family, dtype)
    u, equation, geometry = fixture(dtype=dtype)
    u.requires_grad_()
    h = torch.tensor([.07, .11], dtype=dtype, requires_grad=True)
    actual = model(u, h, equation, geometry)
    assert ((actual >= 0) & (actual <= 1)).all()
    gradients = torch.autograd.grad(actual.square().mean(), (u, h, *model.parameters()))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert gradients[0].abs().sum() > 0 and (gradients[1].abs() > 0).all()
    with torch.no_grad():
        independent = torch.cat([model(u[index:index + 1], h[index], equation, geometry)
                                 for index in range(2)])
        translated = model(torch.roll(u, (2, 3), (-2, -1)), h, equation, geometry)
    tolerance = 3e-6 if dtype == torch.float32 else 4e-13
    torch.testing.assert_close(actual, independent, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(translated, torch.roll(actual, (2, 3), (-2, -1)),
                               atol=tolerance, rtol=tolerance)
    if "endpoint" in family or "closure" in family:
        target = model.correction_components(u, h, equation, geometry)["target_mean"]
        torch.testing.assert_close(actual.mean((-2, -1), keepdim=True), target,
                                   atol=5 * torch.finfo(dtype).eps, rtol=0.)
    json.dumps(model.architecture_metadata(), allow_nan=False)


@pytest.mark.parametrize("family,old_family", (("local_gate", "premix_gated"),
                                              ("local_endpoint", "premix_moment")))
def test_local_controls_reuse_historical_map_including_nonzero_weights(family, old_family):
    torch.manual_seed(1703)
    historical = old_model(old_family, width=2, modes=2, t_ref=.2).double()
    torch.manual_seed(1703)
    current = build_model(family, width=2, modes=2, t_ref=.2).double()
    current.backbone.load_state_dict(historical.state_dict())
    with torch.no_grad():
        for parameter in historical.parameters():
            parameter.copy_(.2 * torch.randn_like(parameter))
    current.backbone.load_state_dict(historical.state_dict())
    u, equation, geometry = fixture()
    assert torch.equal(current(u, .1, equation, geometry), historical(u, .1, equation, geometry))


def test_premix_precompression_controls_have_identical_parameters_and_explicit_source_bypass():
    torch.manual_seed(50)
    premix = build_model("source", width=3, modes=2)
    torch.manual_seed(50)
    precompress = build_model("precompress_source", width=3, modes=2)
    assert premix.state_dict().keys() == precompress.state_dict().keys()
    assert all(torch.equal(value, precompress.state_dict()[name])
               for name, value in premix.state_dict().items())
    metadata = precompress.architecture_metadata()
    assert metadata["correction_encoder_state"] == "P_K(u)"
    assert metadata["full_state_source_bypass"] is True
    assert "full-state physical commutator source" in metadata["information_bypasses"]


def test_source_filter_reaches_flat_receiver_where_post_gate_cannot():
    geometry = Geometry((12, 12), (1., 1.))
    equation = Equation(.01, 2.)
    u = torch.full((1, 1, 12, 12), .4, dtype=torch.float64)
    u[..., 2:4, 2:4] += .08
    source = build_model("source", width=1, modes=1, t_ref=.2).double()
    local = build_model("local_gate", width=1, modes=1, t_ref=.2).double()
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.zero_()
        source.backbone.lift.bias.fill_(1.)
        source.backbone.spectral.weight_real[0, 0, 1, 0] = 1.
        source.backbone.spectral.weight_real[0, 0, 0, 0] = 1.
        source.backbone.spectral.weight_real[0, 0, 2, 0] = 1.
        source.backbone.head.weight.fill_(1.)
        local.backbone.head.bias.fill_(1.)
    local_increment = local.correction_components(u, .1, equation, geometry)["spatial_increment"]
    transported = source.correction_components(u, .1, equation, geometry)["spatial_increment"]
    assert local_increment[..., 8, 8].item() == 0.
    assert abs(transported[..., 8, 8].item()) > 1e-6
    assert transported[..., 8, 8].item() != pytest.approx(transported[..., 6, 8].item(), abs=1e-8)
    output = source(u, .1, equation, geometry)
    base = physical_base(u, .1, equation, geometry)
    assert abs((output - base)[..., 8, 8].item()) > 1e-6


def test_source_quadratic_direction_escapes_local_gate_plus_constant_span():
    geometry = Geometry((12, 12), (1., 1.))
    equation = Equation(.013, 3.2)
    x = torch.arange(12, dtype=torch.float64) * 2 * torch.pi / 12
    xx, yy = torch.meshgrid(x, x, indexing="ij")
    v = .06 * torch.cos(xx) + .03 * torch.sin(3 * xx + yy + .4)
    model = build_model("source", width=1, modes=2, t_ref=.2).double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.backbone.lift.bias.fill_(1.)
        model.backbone.head.weight.fill_(1.)
        model.backbone.spectral.weight_real[0, 0, 4, 0] = 1.
        model.backbone.spectral.weight_real[0, 0, 0, 0] = 1.
    first = (.4 + .01 * v)[None, None]
    second = (.4 + .02 * v)[None, None]
    a = model.correction_components(first, .09, equation, geometry)
    b = model.correction_components(second, .09, equation, geometry)
    proposed = a["spatial_increment"]
    torch.testing.assert_close(b["spatial_increment"], 4 * proposed, rtol=3e-10, atol=2e-15)
    design = torch.stack((a["source"].flatten(), torch.ones(proposed.numel(), dtype=proposed.dtype)), 1)
    coefficients = torch.linalg.lstsq(design, proposed.flatten()).solution
    residual = proposed.flatten() - design @ coefficients
    assert residual.norm() / proposed.norm() > .5
    actual = model(first, .09, equation, geometry) - a["base"]
    torch.testing.assert_close(actual, proposed, rtol=4e-6, atol=2e-15)


def test_pair_kernel_input_factor_swap_is_symmetric_and_signed():
    model = learned("pair_rank4")
    swapped = deepcopy(model)
    rank = model.kernel.rank
    with torch.no_grad():
        final = swapped.kernel.input_factors[-1]
        final.weight.copy_(torch.cat((final.weight[rank:].clone(), final.weight[:rank].clone())))
        final.bias.copy_(torch.cat((final.bias[rank:].clone(), final.bias[:rank].clone())))
    u, equation, geometry = fixture()
    original = model.correction_components(u, .1, equation, geometry)["spatial_increment"]
    exchanged = swapped.correction_components(u, .1, equation, geometry)["spatial_increment"]
    torch.testing.assert_close(exchanged, original, rtol=3e-12, atol=2e-15)
    with torch.no_grad():
        swapped.kernel.amplitude.neg_()
    opposite = swapped.correction_components(u, .1, equation, geometry)["spatial_increment"]
    torch.testing.assert_close(opposite, -original, rtol=3e-12, atol=2e-15)


@pytest.mark.parametrize("family", ("pair_rank2", "pair_rank4", "pair_rank8"))
def test_pair_branch_is_quadratic_not_quartic_and_complete_map_preserves_leading_pattern(family):
    model = learned(family)
    u, equation, geometry = fixture(batch=1)
    v = centered_state(u)
    first = .4 + .01 * v
    second = .4 + .02 * v
    a = model.correction_components(first, .09, equation, geometry)["spatial_increment"]
    b = model.correction_components(second, .09, equation, geometry)["spatial_increment"]
    assert a.abs().max() > 1e-12
    torch.testing.assert_close(b, 4 * a, rtol=2e-10, atol=2e-15)
    da = model(first, .09, equation, geometry) - physical_base(first, .09, equation, geometry)
    db = model(second, .09, equation, geometry) - physical_base(second, .09, equation, geometry)
    # The capacity map has derivative one at zero, hence transports the signed
    # quadratic pattern. Its next amplitude term is fourth order, not its first.
    torch.testing.assert_close(da, a, rtol=1e-5, atol=2e-15)
    torch.testing.assert_close(db, 4 * da, rtol=4e-5, atol=3e-15)
    assert (a > 0).any() or (a < 0).any()


def test_pair_phase_dependence_is_not_replaced_by_a_spectrum_only_summary():
    model = learned("pair_rank4")
    geometry = Geometry((16, 16), (1., 1.))
    equation = Equation(.01, 2.)
    x = 2 * torch.pi * torch.arange(16, dtype=torch.float64) / 16
    xx, yy = torch.meshgrid(x, x, indexing="ij")
    a = (.4 + .02 * torch.cos(xx) + .02 * torch.cos(2 * xx + yy))[None, None]
    b = (.4 + .02 * torch.cos(xx) + .02 * torch.cos(2 * xx + yy + .7))[None, None]
    torch.testing.assert_close(torch.fft.rfft2(a - a.mean()).abs(),
                               torch.fft.rfft2(b - b.mean()).abs(), atol=2e-14, rtol=2e-12)
    first = model.correction_components(a, .1, equation, geometry)["spatial_increment"]
    second = model.correction_components(b, .1, equation, geometry)["spatial_increment"]
    assert (first - second).square().mean().sqrt() > 1e-7
    assert first.is_floating_point() and torch.isfinite(first).all()


def test_temporal_control_has_actual_nonlinearity_before_the_capacity_map():
    u, equation, geometry = fixture(batch=1)
    nonlinear = learned("local_gate_time")
    polynomial = learned("local_gate")
    h = torch.tensor(.08, dtype=torch.float64, requires_grad=True)

    def third_derivative(model):
        proposed = model.correction_components(u, h, equation, geometry)["spatial_increment"]
        value = (proposed / h**3).sum()
        for _ in range(3):
            value = torch.autograd.grad(value, h, create_graph=True)[0]
        return value

    assert abs(third_derivative(polynomial).item()) < 2e-7
    assert abs(third_derivative(nonlinear).item()) > 1e-5


def test_actual_mask_and_parameter_matching_are_declared_without_padding_parameters():
    premix = build_model("source", width=16, modes=4)
    one = build_model("fno_source", width=16, modes=4)
    matched = build_model("fno_source_matched", width=16, modes=4)
    depth = build_model("fno_source_depth4", width=16, modes=4)
    anchor = build_model("fno_anchor", width=16, modes=4)
    assert premix.backbone.spectral.weight_real.shape[-2:] == one.blocks[0].spectral.weight_real.shape[-2:]
    assert matched.width == parameter_matched_width(16, 4, depth=4) == 8
    target = sum(parameter.numel() for parameter in premix.parameters())
    assert matched.architecture_metadata()["parameter_matching_target"] == target
    assert matched.architecture_metadata()["parameter_matching_relative_error"] < .02
    assert len(one.blocks) == 1 and len(depth.blocks) == len(matched.blocks) == 4
    assert "historical banks" in anchor.architecture_metadata()["fourier_support"]
    assert depth.architecture_metadata()["parameters"] > premix.architecture_metadata()["parameters"]


def test_actual_inclusive_mask_keeps_the_boundary_mode_missing_from_the_old_anchor():
    matched = build_model("fno_source", width=1, modes=2).double()
    anchor = build_model("fno_anchor", width=1, modes=2).double()
    for model in (matched, anchor):
        with torch.no_grad():
            for parameter in model.blocks[0].spectral.parameters():
                parameter.zero_()
    with torch.no_grad():
        matched.blocks[0].spectral.weight_real[0, 0, 2, 2] = 1.
        anchor.blocks[0].spectral.positive_real.fill_(1.)
        anchor.blocks[0].spectral.negative_real.fill_(1.)
    y = torch.arange(12, dtype=torch.float64) * 2 * torch.pi / 12
    boundary = torch.cos(2 * y)[None, None, None].expand(1, 1, 12, 12)
    retained = matched.blocks[0].spectral(boundary)
    removed = anchor.blocks[0].spectral(boundary)
    torch.testing.assert_close(retained, boundary, rtol=0., atol=2e-15)
    assert removed.abs().max() < 2e-15


def test_instantaneous_mean_and_variance_identities_match_full_rhs_autograd():
    u, equation, geometry = fixture()
    u.requires_grad_()
    moments = physical_moments(u, equation, geometry)
    drift = rhs(u, equation, geometry)
    actual_mean = drift.mean((-2, -1), keepdim=True)
    actual_variance = 2 * ((u - u.mean((-2, -1), keepdim=True)) * drift).mean((-2, -1), keepdim=True)
    torch.testing.assert_close(moments["mean_derivative"], actual_mean, rtol=3e-13, atol=2e-15)
    torch.testing.assert_close(moments["variance_derivative"], actual_variance, rtol=3e-13, atol=2e-15)
    assert moments["diffusion_energy"].max() < 0


def test_closure_is_bounded_dynamical_and_has_cubic_learned_mean_remainder():
    u, equation, geometry = fixture(batch=1)
    model = learned("source_closure")
    closure = model.mean_closure
    steps = (.002, .004)
    corrections = []
    for h in steps:
        values = closure(u, torch.tensor(h, dtype=u.dtype), equation, geometry, .2)
        assert (values["closure_mean"] >= 0).all() and (values["closure_mean"] <= 1).all()
        bound = values["closure_mean"] * (1 - values["closure_mean"])
        assert (values["closure_variance"] >= 0).all() and (values["closure_variance"] <= bound).all()
        corrections.append(values["mean_increment"].item())
    assert abs(corrections[0]) > 1e-12
    assert corrections[1] / corrections[0] == pytest.approx(8., rel=.06)
    # At infinitesimal time the known mean law uses the actual initial variance.
    h = torch.tensor(1e-6, dtype=u.dtype)
    values = closure(u, h, equation, geometry, .2)
    rate = (values["closure_mean"] - values["mean"]) / h
    torch.testing.assert_close(rate, values["mean_derivative"], atol=2e-5, rtol=2e-5)


def test_equal_mean_variance_different_spectra_produce_different_closure_dynamics():
    geometry = Geometry((16, 16), (1., 1.))
    equation = Equation(.013, 3.2)
    x = 2 * torch.pi * torch.arange(16, dtype=torch.float64) / 16
    xx, _ = torch.meshgrid(x, x, indexing="ij")
    low = (.4 + .08 * torch.cos(xx))[None, None]
    high = (.4 + .08 * torch.cos(5 * xx))[None, None]
    model = learned("source_closure")
    a = model.mean_closure(low, torch.tensor(.1, dtype=low.dtype), equation, geometry, .2)
    b = model.mean_closure(high, torch.tensor(.1, dtype=low.dtype), equation, geometry, .2)
    torch.testing.assert_close(a["mean"], b["mean"], rtol=0., atol=1e-15)
    torch.testing.assert_close(a["variance"], b["variance"], rtol=0., atol=1e-15)
    assert abs((a["diffusion_energy"] - b["diffusion_energy"]).item()) > .01
    assert abs((a["closure_mean"] - b["closure_mean"]).item()) > 1e-4


def test_equal_mean_variance_and_energy_different_phase_changes_the_third_moment_closure():
    geometry = Geometry((16, 16), (1., 1.))
    equation = Equation(.013, 3.2)
    x = 2 * torch.pi * torch.arange(16, dtype=torch.float64) / 16
    xx, _ = torch.meshgrid(x, x, indexing="ij")
    a = (.4 + .07 * torch.cos(xx) + .035 * torch.cos(2 * xx))[None, None]
    b = (.4 + .07 * torch.cos(xx) + .035 * torch.cos(2 * xx + torch.pi / 2))[None, None]
    model = learned("source_closure")
    first = model.mean_closure(a, torch.tensor(.1, dtype=a.dtype), equation, geometry, .2)
    second = model.mean_closure(b, torch.tensor(.1, dtype=b.dtype), equation, geometry, .2)
    for key in ("mean", "variance", "diffusion_energy"):
        torch.testing.assert_close(first[key], second[key], rtol=2e-12, atol=2e-15)
    assert abs((first["third_centered_moment"] - second["third_centered_moment"]).item()) > 1e-4
    assert abs((first["closure_mean"] - second["closure_mean"]).item()) > 1e-6


def test_binary_moment_fraction_is_an_explicit_closure_limitation():
    geometry = Geometry((8, 8), (1., 1.))
    equation = Equation(.013, 3.2)
    state = torch.zeros(1, 1, 8, 8, dtype=torch.float64)
    state[..., :4, :] = 1.
    model = learned("source_closure")
    values = model.mean_closure(state, torch.tensor(.1, dtype=state.dtype), equation, geometry, .2)
    assert values["variance_derivative"].item() < 0
    assert values["closure_mean"].item() == state.mean().item()
    assert values["mean_increment"].item() == 0.
    assert "binary-field" in model.architecture_metadata()["closure_moment_coordinate_limitation"]


@pytest.mark.parametrize("family", ("source_time", "source_closure", "pair_rank4"))
def test_complete_new_maps_state_and_time_derivatives_match_finite_differences(family):
    model = learned(family, width=1, modes=1)
    u, equation, geometry = fixture(grid=(4, 4), batch=1)
    u.requires_grad_()
    h = torch.tensor(.07, dtype=u.dtype, requires_grad=True)
    assert torch.autograd.gradcheck(lambda state, time: model(state, time, equation, geometry),
                                   (u, h), eps=1e-6, atol=5e-7, rtol=4e-5)


def test_diffusion_first_base_has_correct_strong_diffusion_homogenized_limit():
    u, _, geometry = fixture()
    equation = Equation(1e4, 3.2)
    exact_limit = reaction_step(u.mean((-2, -1), keepdim=True), .2, equation).expand_as(u)
    diffusion_first = physical_base(u, .2, equation, geometry, "diffusion-first")
    reaction_first = physical_base(u, .2, equation, geometry, "reaction-first")
    torch.testing.assert_close(diffusion_first, exact_limit, rtol=0., atol=5e-15)
    assert (reaction_first - exact_limit).abs().max() > 2e-4


@pytest.mark.parametrize("family", ("source", "source_time", "source_closure", "precompress_source",
                                    "fno_source", "pair_rank4"))
def test_optimizer_moves_zero_heads_and_checkpoint_roundtrip_preserves_all_buffers(family):
    torch.manual_seed(2703)
    model = build_model(family, width=2, modes=2, t_ref=.2).double()
    mean = torch.linspace(-.1, .1, 12, dtype=torch.float64)
    std = torch.linspace(.8, 1.2, 12, dtype=torch.float64)
    model.set_normalization(mean, std)
    u, equation, geometry = fixture()
    target = physical_base(u, .1, equation, geometry) + .001 * torch.cos(u * 13)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    losses = []
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(u, .1, equation, geometry) - target).square().mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    assert any(not torch.equal(value, initial[name]) for name, value in model.named_parameters())
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = build_model(family, width=2, modes=2, t_ref=.2).double()
    restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    assert torch.equal(model(u, .1, equation, geometry), restored(u, .1, equation, geometry))
    assert torch.equal(restored.feature_mean, mean) and torch.equal(restored.feature_std, std)


@pytest.mark.parametrize("kwargs", ({"family": "missing"}, {"family": "source", "width": True},
                                    {"family": "source", "modes": 0}, {"family": "source", "t_ref": 0},
                                    {"family": "source", "base_orientation": "unknown"},
                                    {"family": "pair_rank4", "rank": 2}, {"family": "source", "rank": 2},
                                    {"family": "fno_source", "depth": 4}))
def test_invalid_family_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        build_model(**kwargs)
