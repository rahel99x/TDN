"""Independent CPU checks of the input-computed quadratic interaction branch.

The coefficient reference evolves variations of the coupled finite-dimensional
ODE and its split flows. It does not reuse the implementation's anchored
integral, reaction weights, Fourier diffusion, or phi functions.
"""
from __future__ import annotations

import math
from unittest.mock import patch

import mpmath as mp
import numpy as np
import pytest
import torch
from scipy.integrate import quad, solve_ivp
from scipy.linalg import expm

from tdn.numerics.splitting import split_step
from tdn.numerics.subflows import diffusion_step
from tdn.numerics.types import Equation, Geometry
from tdn.research.interaction import (
    etdrk2_step,
    etdrk4_step,
    interaction_cubic_coefficient,
    interaction_defect,
    interaction_step,
    interaction_work,
    output_phi_defect,
    phi,
    reaction_jacobian,
    reaction_weight,
    scalar_defect,
)


def _laplacian(n: int, length: float = 1.) -> np.ndarray:
    identity = np.eye(n)
    return (np.roll(identity, 1, axis=0) - 2 * identity
            + np.roll(identity, -1, axis=0)) * (n / length)**2


def _mixed_state(n=16, base=.4, amplitude=.08, pair=(1, 3)):
    x = torch.arange(n, dtype=torch.float64) * (2 * torch.pi / n)
    return (base + amplitude * (torch.cos(pair[0] * x)
                               + .6 * torch.sin(pair[1] * x)))[None, None]


def _solve(fun, state, h, *, rtol=3e-13, atol=3e-15):
    result = solve_ivp(fun, (0., h), state, method="DOP853", rtol=rtol, atol=atol)
    assert result.success, result.message
    assert np.isfinite(result.y[:, -1]).all()
    return result.y[:, -1]


def _quadratic_ode_defect(u, h, equation, geometry):
    """Coefficient of epsilon**2 for u=c+epsilon*v, minus the same split."""
    values = u.detach().numpy().reshape(-1)
    n = len(values)
    rate = equation.reaction_rate
    diffusion = equation.kappa * _laplacian(n, geometry.lengths[0])
    background = values.mean()
    initial = np.r_[background, values - background, np.zeros(n)]

    def field(matrix):
        def rhs(_t, state):
            c, first, second = state[0], state[1:1 + n], state[1 + n:]
            sensitivity = rate * (1 - 2 * c)
            return np.r_[rate * c * (1 - c),
                         matrix @ first + sensitivity * first,
                         matrix @ second + sensitivity * second - rate * first**2]
        return rhs

    coupled = _solve(field(diffusion), initial, h)
    reaction = field(np.zeros_like(diffusion))
    split = _solve(reaction, initial, h / 2)
    propagator = expm(h * diffusion)
    split[1:1 + n] = propagator @ split[1:1 + n]
    split[1 + n:] = propagator @ split[1 + n:]
    split = _solve(reaction, split, h / 2)
    return coupled[1 + n:] - split[1 + n:]


@pytest.mark.parametrize("pair", [(1, 3), (2, 2), (1, -1), (3, -1)])
@pytest.mark.parametrize("base", [.15, .5, .85])
def test_runtime_branch_matches_independent_coupled_and_split_variation_odes(pair, base):
    geometry, equation = Geometry((16,), (1.,)), Equation(.012, 3.)
    state = _mixed_state(base=base, amplitude=.035, pair=pair)
    expected = _quadratic_ode_defect(state, .13, equation, geometry)
    actual = interaction_defect(state, .13, equation, geometry, nodes=5)
    np.testing.assert_allclose(actual.numpy().reshape(-1), expected,
                               rtol=3e-6, atol=3e-11)
    assert np.linalg.norm(expected) > 1e-8


def test_amplitude_scaling_is_quadratic_without_an_extra_variation_gate():
    geometry, equation = Geometry((16,), (1.,)), Equation(.02, 4.)
    base = .375  # Exactly represented so halving amplitudes preserves the mean.
    state = _mixed_state(base=base)
    first = interaction_defect(state, .19, equation, geometry, nodes=5)
    second = interaction_defect(base + (state - base) / 2, .19, equation, geometry, nodes=5)
    assert first.abs().max() > 1e-5
    torch.testing.assert_close(second * 4, first, rtol=2e-12, atol=2e-17)


def test_cubic_anchor_matches_small_time_ode_and_runtime_defect():
    geometry, equation = Geometry((16,), (1.,)), Equation(.02, 4.)
    state = _mixed_state()
    coefficient = interaction_cubic_coefficient(state, equation, geometry)
    errors, sizes = [], []
    for h in (.004, .002, .001):
        defect = interaction_defect(state, h, equation, geometry, nodes=5)
        errors.append(float(torch.linalg.vector_norm(defect / h**3 - coefficient)))
        sizes.append(float(torch.linalg.vector_norm(defect)))
    assert errors[-1] < errors[0] / 3.5
    assert math.log2(sizes[-2] / sizes[-1]) == pytest.approx(3., abs=.05)
    # The independent variation ODE fixes the sign and leading coefficient.
    ode = _quadratic_ode_defect(state, .004, equation, geometry)
    np.testing.assert_allclose(ode / .004**3, coefficient.numpy().reshape(-1),
                               rtol=.2, atol=.015)


@pytest.mark.parametrize("nodes,horizons,minimum_order", [(3, (.05, .025), 6.), (5, (.1, .05), 9.)])
def test_quadrature_refinement_resolves_the_independent_defect(nodes, horizons, minimum_order):
    geometry, equation = Geometry((16,), (1.,)), Equation(.04, 6.)
    state = _mixed_state(pair=(3, 5), amplitude=.05)
    errors = []
    for h in horizons:
        expected = _quadratic_ode_defect(state, h, equation, geometry)
        actual = interaction_defect(state, h, equation, geometry, nodes=nodes)
        errors.append(np.linalg.norm(actual.numpy().reshape(-1) - expected))
    assert errors[0] > 1e-12  # The order check must be above reference noise.
    # These finite steps approach the fixed-grid orders seven and eleven;
    # they do not assert a mesh-uniform or stiff quadrature order.
    assert math.log2(errors[0] / errors[1]) > minimum_order


def test_five_nodes_improve_quadrature_on_a_resolved_finite_step():
    geometry, equation = Geometry((16,), (1.,)), Equation(.04, 6.)
    state = _mixed_state(pair=(3, 5), amplitude=.05)
    expected = _quadratic_ode_defect(state, .2, equation, geometry)
    errors = [np.linalg.norm(interaction_defect(state, .2, equation, geometry, nodes=n)
                             .numpy().reshape(-1) - expected) for n in (3, 5)]
    assert errors[1] < errors[0] / 50


@pytest.mark.parametrize("dtype,tolerance", [(torch.float32, 4e-6), (torch.float64, 3e-13)])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_phi_agrees_with_independent_integral_at_zero_small_and_large_arguments(dtype, tolerance, order):
    values = [-1000., -100., -10., -1., -1e-4, -1e-12, 0., 1e-12, 1e-4, 1., 10.]
    z = torch.tensor(values, dtype=dtype)
    expected = [quad(lambda t: math.exp((1 - t) * value) * t**(order - 1),
                     0., 1., epsabs=2e-13, epsrel=2e-13)[0] / math.factorial(order - 1)
                for value in z.double().tolist()]
    actual = phi(z, order=order)
    assert actual.dtype == dtype
    np.testing.assert_allclose(actual.numpy(), expected, rtol=tolerance, atol=1e-15)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_phi_has_the_correct_derivative_at_zero(order):
    z = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    result = phi(z, order=order)
    derivative = torch.autograd.grad(result, z)[0]
    assert float(result.detach()) == pytest.approx(1 / math.factorial(order), abs=1e-15)
    assert float(derivative) == pytest.approx(1 / math.factorial(order + 1), abs=1e-15)


@pytest.mark.parametrize("dtype,tolerance", [(torch.float32, 3e-6), (torch.float64, 3e-13)])
@pytest.mark.parametrize("h,rate", [(0., 6.), (.2, 0.), (1e-8, 1e-10), (.3, 6.), (3., 6.)])
def test_analytic_jacobian_and_half_to_full_weight_match_independent_quadrature(dtype, tolerance, h, rate):
    c = torch.tensor([0., 2**-20, .2, .7, 1 - 2**-20, 1.], dtype=dtype)

    def jacobian(base, time):
        # Positive-exponent form is independent of the implementation's
        # stable negative-exponent denominators, at representable horizons.
        exponential = math.exp(rate * time)
        return exponential / (1 + base * math.expm1(rate * time))**2

    expected_j = [jacobian(base, h) for base in c.double().tolist()]
    expected_w = [quad(lambda s: jacobian(base, s), h / 2, h,
                       epsabs=1e-12, epsrel=2e-13)[0] for base in c.double().tolist()]
    np.testing.assert_allclose(reaction_jacobian(c, h, rate).numpy(), expected_j,
                               rtol=tolerance, atol=1e-16)
    np.testing.assert_allclose(reaction_weight(c, h, rate).numpy(), expected_w,
                               rtol=tolerance, atol=1e-20)


@pytest.mark.parametrize("dtype,base,h,tolerance", [(torch.float64, 1e-200, 750., 4e-13),
                                                 (torch.float32, 1e-25, 110., 2e-5)])
def test_reaction_sensitivities_survive_exponential_underflow_when_result_is_representable(dtype, base, h, tolerance):
    c = torch.tensor(base, dtype=dtype)
    with mp.workdps(90):
        base_mp = mp.mpf(float(c))

        def denominator(t):
            return base_mp + (1 - base_mp) * mp.exp(-t)

        jacobian = mp.exp(-h) / denominator(h)**2
        # W2 at twice the horizon has an underflowing half-time exponential.
        weight = mp.exp(-h) * (-mp.expm1(-h)) / (denominator(h) * denominator(2 * h))
    actual_j = reaction_jacobian(c, h, 1.)
    actual_w = reaction_weight(c, 2 * h, 1.)
    assert torch.isfinite(actual_j) and actual_j > 0
    assert torch.isfinite(actual_w) and actual_w > 0
    assert float(actual_j) == pytest.approx(float(jacobian), rel=tolerance)
    assert float(actual_w) == pytest.approx(float(weight), rel=tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("coordinate", ["additive", "capacity", "projected"])
def test_zero_limits_reproduce_split_and_zero_time_preserves_state(dtype, coordinate):
    geometry = Geometry((16,), (1.,))
    state = _mixed_state().to(dtype)
    for equation in (Equation(0., 4.), Equation(.02, 0.), Equation(.02, 4.)):
        assert torch.equal(interaction_defect(state, 0., equation, geometry), torch.zeros_like(state))
        torch.testing.assert_close(interaction_step(state, 0., equation, geometry, coordinate=coordinate),
                                   state, rtol=0., atol=4 * torch.finfo(dtype).eps)
    for equation in (Equation(0., 4.), Equation(.02, 0.)):
        assert torch.equal(interaction_defect(state, .2, equation, geometry), torch.zeros_like(state))
        assert torch.equal(interaction_step(state, .2, equation, geometry, coordinate=coordinate),
                           split_step(state, .2, equation, geometry))
    equation = Equation(.02, 4.)
    for level in (0., .375, 1.):
        uniform = torch.full_like(state, level)
        assert torch.equal(interaction_defect(uniform, .2, equation, geometry), torch.zeros_like(uniform))
        assert torch.equal(interaction_step(uniform, .2, equation, geometry, coordinate=coordinate),
                           split_step(uniform, .2, equation, geometry))


@pytest.mark.parametrize("coordinate", ["capacity", "projected"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_bounded_coordinates_remain_finite_near_both_boundaries(dtype, coordinate):
    geometry, equation = Geometry((8,), (1.,)), Equation(.03, 6.)
    state = torch.tensor([0., 2**-22, .01, .3, .7, .99, 1 - 2**-22, 1.], dtype=dtype)[None, None]
    for h in (1e-4, .1, .4, 2.):
        result = interaction_step(state, h, equation, geometry, coordinate=coordinate)
        assert result.dtype == dtype and torch.isfinite(result).all()
        assert ((result >= 0) & (result <= 1)).all()


def test_fp32_anchored_defect_preserves_small_time_signal_against_fp64():
    geometry, equation = Geometry((16,), (1.,)), Equation(.02, 4.)
    state32 = _mixed_state(base=.375).float()
    for h in (.01, .001):
        actual = interaction_defect(state32, h, equation, geometry, nodes=5).double()
        expected = interaction_defect(state32.double(), h, equation, geometry, nodes=5)
        assert expected.abs().max() > 1e-11
        assert float(torch.linalg.vector_norm(actual - expected) /
                     torch.linalg.vector_norm(expected)) < .002


@pytest.mark.parametrize("grid", [(8,), (4, 6), (3, 4, 4)])
def test_batches_and_spatial_dimensions_match_individual_evaluations(grid):
    geometry, equation = Geometry(grid, (1.,) * len(grid)), Equation(.03, 3.)
    count = math.prod(grid)
    state = torch.linspace(.03, .94, 3 * count, dtype=torch.float64).reshape(3, 1, *grid)
    h = torch.tensor([0., .025, .2], dtype=state.dtype)
    for method in (interaction_defect, interaction_step, etdrk2_step, etdrk4_step):
        actual = method(state, h, equation, geometry)
        assert actual.shape == state.shape
        for index in range(3):
            expected = method(state[index:index + 1], h[index], equation, geometry)
            torch.testing.assert_close(actual[index:index + 1], expected, rtol=2e-12, atol=2e-14)


def test_mean_controls_decompose_the_actual_additive_state_correction():
    geometry, equation = Geometry((16,), (1.,)), Equation(.02, 4.)
    state = torch.cat([_mixed_state(base=.2), _mixed_state(base=.7)], dim=0)
    h = torch.tensor([.1, .2], dtype=state.dtype)
    baseline = split_step(state, h, equation, geometry)
    full = interaction_step(state, h, equation, geometry, coordinate="additive", mean_mode="full") - baseline
    zero_mean = interaction_step(state, h, equation, geometry, coordinate="additive", mean_mode="zero_mean") - baseline
    mean_only = interaction_step(state, h, equation, geometry, coordinate="additive", mean_mode="mean_only") - baseline
    torch.testing.assert_close(full, zero_mean + mean_only, rtol=2e-11, atol=2e-16)
    torch.testing.assert_close(zero_mean.mean(dim=-1), torch.zeros_like(zero_mean[..., 0]),
                               rtol=0., atol=2e-16)
    torch.testing.assert_close(mean_only, full.mean(dim=-1, keepdim=True).expand_as(full),
                               rtol=2e-11, atol=2e-16)
    assert full.mean(dim=-1).abs().min() > 1e-7


@pytest.mark.parametrize("nodes", [3, 5])
@pytest.mark.parametrize("include_split", [False, True])
def test_reported_work_matches_every_actual_fft_and_accumulates_rollouts(nodes, include_split):
    geometry, equation, state = Geometry((16,), (1.,)), Equation(.02, 4.), _mixed_state()
    method = interaction_step if include_split else interaction_defect
    work = {}
    with patch("torch.fft.fftn", wraps=torch.fft.fftn) as forward, \
         patch("torch.fft.ifftn", wraps=torch.fft.ifftn) as inverse:
        method(state, .2, equation, geometry, nodes=nodes, work=work)
    assert work["fft_forward"] == forward.call_count == nodes + 2 + int(include_split)
    assert work["fft_inverse"] == inverse.call_count == 3 * nodes + 2 + int(include_split)
    assert work["fft_total"] == forward.call_count + inverse.call_count
    assert work["quadrature_evaluations"] == nodes
    assert work["reaction_evaluations"] == 2 * int(include_split)
    assert work == interaction_work(nodes=nodes, include_split=include_split)
    first = dict(work)
    method(state, .2, equation, geometry, nodes=nodes, work=work)
    assert work == {key: 2 * value for key, value in first.items()}
    method(state, 0., equation, geometry, nodes=nodes, work=work)
    assert work == {key: 2 * value for key, value in first.items()}


@pytest.mark.parametrize("method,order", [(etdrk2_step, 2), (etdrk4_step, 4)])
def test_etd_rollout_convergence_against_independent_coupled_ode(method, order):
    geometry, equation = Geometry((12,), (1.,)), Equation(.015, 4.)
    state = _mixed_state(n=12, amplitude=.12)
    diffusion = equation.kappa * _laplacian(12)
    truth = _solve(lambda _t, u: diffusion @ u + equation.reaction_rate * u * (1 - u),
                   state.numpy().reshape(-1), .4)
    errors = []
    for count in (4, 8, 16):
        actual = state.clone()
        for _ in range(count):
            actual = method(actual, .4 / count, equation, geometry)
        errors.append(np.linalg.norm(actual.numpy().reshape(-1) - truth))
    assert errors[-1] < errors[0]
    assert math.log2(errors[-2] / errors[-1]) == pytest.approx(order, abs=.35)


@pytest.mark.parametrize("method", [etdrk2_step, etdrk4_step])
def test_etd_diffusion_limit_and_zero_time(method):
    geometry, equation, state = Geometry((16,), (1.,)), Equation(.02, 0.), _mixed_state()
    torch.testing.assert_close(method(state, .3, equation, geometry),
                               diffusion_step(state, .3, equation, geometry), rtol=3e-14, atol=3e-14)
    torch.testing.assert_close(method(state, 0., equation, geometry), state, rtol=0., atol=3e-14)


def test_scalar_and_output_phi_controls_share_the_cubic_anchor():
    geometry, equation, state = Geometry((16,), (1.,)), Equation(.02, 4.), _mixed_state()
    coefficient = interaction_cubic_coefficient(state, equation, geometry)
    h = .001
    torch.testing.assert_close(scalar_defect(state, h, equation, geometry),
                               h**3 * coefficient, rtol=2e-14, atol=1e-19)
    actual = output_phi_defect(state, h, equation, geometry)
    assert float(torch.linalg.vector_norm(actual / h**3 - coefficient) /
                 torch.linalg.vector_norm(coefficient)) < .02


@pytest.mark.parametrize("method", [interaction_defect, interaction_step, etdrk2_step, etdrk4_step])
@pytest.mark.parametrize("h", [-.1, float("nan"), float("inf")])
def test_numerical_methods_reject_invalid_horizons(method, h):
    with pytest.raises(ValueError):
        method(_mixed_state(), h, Equation(.02, 4.), Geometry((16,), (1.,)))


@pytest.mark.parametrize("nodes", [0, 1, 4, 7, True, 3.0])
def test_interaction_rejects_unsupported_quadrature(nodes):
    with pytest.raises(ValueError):
        interaction_defect(_mixed_state(), .1, Equation(.02, 4.), Geometry((16,), (1.,)), nodes=nodes)
    with pytest.raises(ValueError):
        interaction_work(nodes=nodes)


@pytest.mark.parametrize("kwargs", [{"coordinate": "clock"}, {"mean_mode": "conserved"}])
def test_step_rejects_unknown_ablation(kwargs):
    with pytest.raises(ValueError):
        interaction_step(_mixed_state(), .1, Equation(.02, 4.), Geometry((16,), (1.,)), **kwargs)


@pytest.mark.parametrize("order", [0, 4, True, 1.0])
def test_phi_rejects_unsupported_order(order):
    with pytest.raises(ValueError):
        phi(torch.tensor(-1., dtype=torch.float64), order=order)


@pytest.mark.parametrize("method", [interaction_defect, interaction_step, etdrk2_step, etdrk4_step])
def test_numerical_methods_reject_mismatched_geometry_and_batch_h(method):
    with pytest.raises(ValueError):
        method(_mixed_state(), .1, Equation(.02, 4.), Geometry((8,), (1.,)))
    with pytest.raises(ValueError):
        method(_mixed_state(), torch.tensor([.1, .2]), Equation(.02, 4.), Geometry((16,), (1.,)))
