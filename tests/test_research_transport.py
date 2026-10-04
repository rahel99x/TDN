"""Independent mathematical and differentiation checks for the transport step.

These bounded CPU checks do not measure a trained advantage or GPU speed.
"""
from __future__ import annotations

import math
from unittest.mock import patch

import pytest
import torch

from tdn.numerics import (Equation, Geometry, laplacian, refined_reference,
                         split_step, weighted_norm)
from tdn.research.transport import (LeadingDefectStep, TransportTDN,
                                    commutator_fields, e3_anchor_step)


def _case(grid=(4, 4), dtype=torch.float64, seed=731):
    geometry = Geometry(grid, tuple(1.0 + .2 * d for d in range(len(grid))))
    u = .15 + .7 * torch.rand((1, 1, *grid), dtype=dtype,
                              generator=torch.Generator().manual_seed(seed))
    return u, Equation(.025, 3.0), geometry


def _nonzero_model(ndim=2, dtype=torch.float64):
    model = TransportTDN(width=5, ndim=ndim).to(dtype=dtype)
    # Do not depend on global test ordering for initial parameters.
    generator = torch.Generator().manual_seed(992)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name != "raw_damping_rates":
                parameter.copy_(.15 * torch.randn(parameter.shape, dtype=dtype,
                                                  generator=generator))
    return model


@pytest.mark.parametrize("grid", [(2,), (3, 4), (2, 3, 2)])
def test_commutators_agree_with_independent_vector_field_jvps(grid):
    u, equation, geometry = _case(grid)
    a = lambda state: equation.kappa * laplacian(state, geometry)
    b = lambda state: equation.reaction_rate * state * (1 - state)

    def bracket(first, second, state):
        return (torch.func.jvp(first, (state,), (second(state),))[1]
                - torch.func.jvp(second, (state,), (first(state),))[1])

    c = lambda state: bracket(a, b, state)
    fields = commutator_fields(u, equation, geometry)
    assert torch.allclose(fields.C, c(u), atol=4e-15, rtol=2e-13)
    assert torch.allclose(fields.ZA, bracket(a, c, u), atol=2e-14, rtol=2e-12)
    assert torch.allclose(fields.ZB, bracket(b, c, u), atol=2e-14, rtol=2e-12)


@pytest.mark.parametrize("kappa,rate", [(.01, 2.), (.03, 6.), (.1, 12.)])
def test_exact_anchor_sign_and_local_orders_against_refined_coupled_teacher(kappa, rate):
    u, _, geometry = _case()
    equation = Equation(kappa, rate)
    model = TransportTDN(width=4).double()
    fields = commutator_fields(u, equation, geometry)
    horizons = [.01, .005, .0025]
    errors = {"split": [], "post_anchor": [], "transport": []}
    for h in horizons:
        teacher = refined_reference(u, h, equation, geometry, substeps=16,
                                    tolerance=.002, noise_floor=1e-12)
        assert teacher.accepted, teacher.reason
        base = split_step(u, h, equation, geometry)
        for key, prediction in (("split", base),
                                ("post_anchor", e3_anchor_step(u, h, equation, geometry)),
                                ("transport", model(u, h, equation, geometry))):
            error = float(weighted_norm(prediction.detach() - teacher.state))
            assert error > 10 * teacher.uncertainty
            errors[key].append(error)
    e3_error = weighted_norm((teacher.state - base) / horizons[-1]**3 - fields.e3)
    assert float(e3_error / weighted_norm(fields.e3)) < .08
    for key, values in errors.items():
        order = math.log2(values[-2] / values[-1])
        if key == "split":
            assert 2.8 < order < 3.2
        else:
            assert 3.6 < order < 4.3
    # A default zero head must actually apply its declared exact anchor.
    assert not torch.equal(model(u, horizons[0], equation, geometry),
                           split_step(u, horizons[0], equation, geometry))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("limit", ["constant", "zero_reaction", "zero_diffusion"])
def test_exact_commuting_limits_even_with_nonzero_heads(dtype, limit):
    u, equation, geometry = _case(dtype=dtype)
    if limit == "constant":
        u = torch.full_like(u, .4)
    elif limit == "zero_reaction":
        equation = Equation(equation.kappa, 0.)
    else:
        equation = Equation(0., equation.reaction_rate)
    fields = commutator_fields(u, equation, geometry)
    assert all(torch.equal(value, torch.zeros_like(u))
               for value in (fields.C, fields.ZA, fields.ZB, fields.e3))
    base = split_step(u, .13, equation, geometry)
    assert torch.equal(_nonzero_model(dtype=dtype)(u, .13, equation, geometry), base)
    assert torch.equal(LeadingDefectStep()(u, .13, equation, geometry), base)


def test_one_fft_pair_and_explicit_stage_diagnostics():
    u, equation, geometry = _case()
    model = _nonzero_model()
    with patch("torch.fft.fftn", wraps=torch.fft.fftn) as fft, \
            patch("torch.fft.ifftn", wraps=torch.fft.ifftn) as ifft:
        output, stages = model.audit_step(u, .015, equation, geometry)
    assert fft.call_count == 1
    assert ifft.call_count == 1
    assert set(stages) == {"reaction_first", "injected", "diffused",
                           "reaction_final", "output"}
    assert stages["output"] is output
    assert all(value.requires_grad for name, value in stages.items()
               if name != "reaction_first")
    assert torch.equal(output, model(u, .015, equation, geometry))
    with patch("torch.fft.fftn", wraps=torch.fft.fftn) as fft, \
            patch("torch.fft.ifftn", wraps=torch.fft.ifftn) as ifft:
        LeadingDefectStep()(u, .015, equation, geometry)
    assert fft.call_count == ifft.call_count == 1


def test_stage_audit_exposes_invalid_injection_without_clipping():
    u, _, geometry = _case()
    model = TransportTDN(width=4, damping_rate=.01).double()
    output, stages = model.audit_step(u, .6, Equation(.1, 12.), geometry)
    assert bool(((stages["injected"] < 0) | (stages["injected"] > 1)).any())
    assert torch.equal(output, model(u, .6, Equation(.1, 12.), geometry))


def test_state_and_time_gradcheck_gradgradcheck_and_parameter_gradients():
    u, equation, geometry = _case((3,))
    u.requires_grad_()
    h = torch.tensor(.025, dtype=u.dtype, requires_grad=True)
    model = _nonzero_model(ndim=1)
    function = lambda state, step: model(state, step, equation, geometry)
    assert torch.autograd.gradcheck(function, (u, h), atol=1e-6, rtol=1e-4)
    assert torch.autograd.gradgradcheck(function, (u, h), atol=2e-6, rtol=2e-4)
    # Both composed physical subflows and the live commutator/encoder path
    # participate in the time/state derivatives, not the old additive jets.
    _, tangent = torch.func.jvp(lambda step: function(u, step), (h,),
                                (torch.ones_like(h),))
    epsilon = 1e-5
    finite_difference = (function(u, h + epsilon) - function(u, h - epsilon)) / (2 * epsilon)
    assert torch.allclose(tangent, finite_difference, atol=1e-9, rtol=1e-7)
    prediction = function(u, h)
    prediction.square().sum().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad), name


def test_batched_horizon_and_fp32_parity():
    initial, equation, geometry = _case()
    u = torch.cat([initial, .8 * initial + .05, .7 * initial + .1])
    h = torch.tensor([0., .02, .05], dtype=u.dtype)
    model64 = _nonzero_model()
    batch = model64(u, h, equation, geometry)
    serial = torch.cat([model64(u[j:j + 1], h[j], equation, geometry)
                        for j in range(len(h))])
    assert torch.allclose(batch, serial, atol=2e-15, rtol=2e-15)
    assert torch.allclose(batch[0], u[0], atol=3e-15, rtol=0)
    model32 = _nonzero_model(dtype=torch.float32)
    model32.load_state_dict(model64.state_dict())
    result32 = model32(u.float(), h.float(), equation, geometry)
    assert float(weighted_norm((batch - result32.double()).detach())) < 2e-7


@pytest.mark.parametrize("keyword,value", [("t_ref", 0.), ("U_ref", float("nan")),
                                            ("coefficient_bound", -1.),
                                            ("damping_rate", float("inf"))])
def test_invalid_fixed_scales_rejected(keyword, value):
    with pytest.raises(ValueError, match=keyword):
        TransportTDN(**{keyword: value})
