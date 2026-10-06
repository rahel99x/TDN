"""Independent Parseval checks and prepared-control parity on actual CPU work."""
from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest
import torch
from scipy.linalg import expm

from tdn.numerics.splitting import split_step
from tdn.numerics.types import Equation, Geometry
from tdn.research.interaction import (
    etdrk2_step, etdrk4_step, interaction_cubic_coefficient,
    interaction_defect, interaction_step,
)
from tdn.research.work_precision import (
    VARIANTS, _cache_tensors, prepare_step, spectral_mean_defect, spectral_mean_step,
)


def _state(grid=(16,), dtype=torch.float64, batch=3):
    count = math.prod(grid)
    x = torch.arange(count, dtype=dtype).reshape(1, 1, *grid) * (2 * torch.pi / count)
    pattern = .055 * torch.cos(x) + .035 * torch.sin(3 * x)
    bases = torch.linspace(.22, .7, batch, dtype=dtype).reshape(batch, 1, *([1] * len(grid)))
    return bases + pattern


def _direct(variant, u, h, equation, geometry, work=None):
    if variant == "strang":
        return split_step(u, h, equation, geometry)
    if variant == "etdrk2":
        return etdrk2_step(u, h, equation, geometry, work=work)
    if variant == "etdrk4":
        return etdrk4_step(u, h, equation, geometry, work=work)
    if variant == "gl3_mean_spectral":
        return spectral_mean_step(u, h, equation, geometry, work=work)
    return interaction_step(u, h, equation, geometry, nodes=5 if variant == "gl5" else 3,
                            mean_mode="mean_only" if variant == "gl3_mean_full" else "full", work=work)


@pytest.mark.parametrize("grid", [(16,), (6, 8)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("nodes", [3, 5])
def test_spectral_mean_matches_full_raw_defect_for_batches_and_batched_times(grid, dtype, nodes):
    geometry, equation = Geometry(grid, (1.,) * len(grid)), Equation(.017, 3.5)
    state = _state(grid, dtype)
    h = torch.tensor([0., .025, .13], dtype=dtype)
    mean = spectral_mean_defect(state, h, equation, geometry, nodes=nodes)
    full = interaction_defect(state, h, equation, geometry, nodes=nodes)
    expected = full.mean(dim=tuple(range(2, state.ndim)), keepdim=True)
    assert mean.shape == (3, 1, *([1] * len(grid)))
    assert mean.dtype == dtype
    tolerance = 3e-6 if dtype == torch.float32 else 3e-12
    torch.testing.assert_close(mean, expected, rtol=tolerance, atol=2e-11 if dtype == torch.float32 else 2e-17)
    for index in range(3):
        one = spectral_mean_defect(state[index:index + 1], h[index], equation, geometry, nodes=nodes)
        torch.testing.assert_close(mean[index:index + 1], one, rtol=tolerance, atol=2e-17)
    result = spectral_mean_step(state, h, equation, geometry, nodes=nodes)
    direct = interaction_step(state, h, equation, geometry, nodes=nodes, mean_mode="mean_only")
    torch.testing.assert_close(result, direct, rtol=2e-6 if dtype == torch.float32 else 2e-13,
                               atol=2e-7 if dtype == torch.float32 else 2e-16)
    assert torch.equal(result[:1], state[:1])


@pytest.mark.parametrize("grid", [(7,), (4, 5)])
@pytest.mark.parametrize("nodes", [3, 5])
def test_mean_normalization_matches_independent_dense_diffusion_energy(grid, nodes):
    """No FFT, discrete spectrum, implementation GL table or reaction helper."""
    geometry, equation = Geometry(grid, (1.3,) * len(grid)), Equation(.027, 2.7)
    state = _state(grid, batch=1)
    u = state.numpy().reshape(-1)
    c, v, h = u.mean(), u - u.mean(), .16
    matrix = np.zeros((u.size, u.size))
    for axis, (n, dx) in enumerate(zip(grid, geometry.dx)):
        identity = np.eye(n)
        one = (np.roll(identity, 1, axis=0) - 2 * identity + np.roll(identity, -1, axis=0)) / dx**2
        factor = np.array([[1.]])
        for index, dimension in enumerate(grid):
            factor = np.kron(factor, one if index == axis else np.eye(dimension))
        matrix += equation.kappa * factor

    def energy_difference(s):
        evolved = expm(s * matrix) @ v
        return np.mean(evolved**2) - np.mean(v**2)

    def jacobian(s):
        q = math.exp(-equation.reaction_rate * s)
        return q / (c + (1 - c) * q)**2

    locations, weights = np.polynomial.legendre.leggauss(nodes)
    integral = sum(h * weight / 2 * jacobian(h * (1 + x) / 2)
                   * energy_difference(h * (1 + x) / 2) for x, weight in zip(locations, weights))
    qhalf, qfull = math.exp(-equation.reaction_rate * h / 2), math.exp(-equation.reaction_rate * h)
    weight = (qhalf - qfull) / (equation.reaction_rate * (c + (1 - c) * qhalf) * (c + (1 - c) * qfull))
    expected = -equation.reaction_rate * jacobian(h) * (integral - weight * energy_difference(h))
    actual = float(spectral_mean_defect(state, h, equation, geometry, nodes=nodes))
    assert abs(expected) > 1e-7
    assert actual == pytest.approx(expected, rel=3e-12, abs=3e-18)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("nodes", [3, 5])
def test_exact_zero_limits_and_uniform_parents(dtype, nodes):
    geometry = Geometry((16,), (1.,))
    state = _state(dtype=dtype)
    for equation, h in [(Equation(.02, 4.), 0.), (Equation(0., 4.), .2), (Equation(.02, 0.), .2)]:
        work = {}
        defect = spectral_mean_defect(state, h, equation, geometry, nodes=nodes, work=work)
        assert torch.equal(defect, torch.zeros_like(defect))
        assert work["fft_total"] == 0
        actual = spectral_mean_step(state, h, equation, geometry, nodes=nodes)
        expected = state if h == 0 else split_step(state, h, equation, geometry)
        assert torch.equal(actual, expected)
    equation = Equation(.02, 4.)
    for base in (0., .1, .375, 1.):
        uniform = torch.full_like(state, base)
        work = {}
        defect = spectral_mean_defect(uniform, .2, equation, geometry, nodes=nodes, work=work)
        assert torch.equal(defect, torch.zeros_like(defect))
        assert work["fft_total"] == 0
        assert torch.equal(spectral_mean_step(uniform, .2, equation, geometry, nodes=nodes),
                           split_step(uniform, .2, equation, geometry))


def test_uniform_endpoints_do_not_contaminate_mixed_batches_at_large_reaction_time():
    geometry, equation = Geometry((16,), (1.,)), Equation(.02, 100.)
    state = torch.cat([torch.zeros_like(_state(batch=1)), _state(batch=1),
                       torch.ones_like(_state(batch=1))])
    actual = spectral_mean_defect(state, 20., equation, geometry)
    assert torch.isfinite(actual).all()
    assert float(actual[0]) == float(actual[-1]) == 0.


@pytest.mark.parametrize("nodes", [3, 5])
def test_fp32_small_time_mean_retains_cubic_signal_against_fp64(nodes):
    geometry, equation = Geometry((16,), (1.,)), Equation(.02, 4.)
    state = _state(dtype=torch.float32, batch=1)
    for h in (.01, .001, 1e-5, 1e-8):
        actual = spectral_mean_defect(state, h, equation, geometry, nodes=nodes).double()
        expected = spectral_mean_defect(state.double(), h, equation, geometry, nodes=nodes)
        assert expected.abs().max() > 1e-27
        assert float((actual - expected).abs().max() / expected.abs().max()) < 2e-5


def test_small_time_mean_matches_the_quadratic_cubic_anchor():
    geometry, equation = Geometry((16,), (1.,)), Equation(.02, 4.)
    state = _state(batch=1)
    coefficient = interaction_cubic_coefficient(state, equation, geometry).mean(-1, keepdim=True)
    errors = []
    for h in (.004, .002, .001):
        value = spectral_mean_defect(state, h, equation, geometry, nodes=5) / h**3
        errors.append(float((value - coefficient).abs().max()))
    assert errors[-1] < errors[0] / 3.5


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("grid", [(16,), (6, 8)])
def test_prepared_controls_match_direct_rollouts_and_counter_totals(variant, dtype, grid):
    geometry, equation = Geometry(grid, (1.,) * len(grid)), Equation(.014, 2.4)
    direct = _state(grid, dtype)
    prepared = direct.clone()
    stepper = prepare_step(prepared, .017, equation, geometry, variant)
    prepared_work, direct_work = {}, {}
    for _ in range(4):
        direct = _direct(variant, direct, .017, equation, geometry, direct_work)
        prepared = stepper(prepared, work=prepared_work)
    torch.testing.assert_close(prepared, direct, rtol=0., atol=0.)
    expected = dict(strang=2, etdrk2=5, etdrk4=9, gl3=18, gl5=26,
                    gl3_mean_full=18, gl3_mean_spectral=3)[variant]
    assert prepared_work["fft_total"] == 4 * expected
    if variant != "strang":
        assert prepared_work == direct_work


@pytest.mark.parametrize("variant", VARIANTS)
def test_prepared_cache_is_state_independent_and_no_phi_or_spectrum_is_rebuilt(variant):
    geometry, equation = Geometry((16,), (1.,)), Equation(.023, 3.)
    example = _state(batch=1)
    horizon = torch.tensor(.04, dtype=example.dtype)
    stepper = prepare_step(example, horizon, equation, geometry, variant)
    tensors = list(_cache_tensors(stepper._coefficients))
    snapshots = [value.clone() for value in tensors]
    changed = _state(batch=4).flip(-1)
    expected = _direct(variant, changed, .04, equation, geometry)
    # Mutating caller-owned inputs must not alter frozen operators or retain
    # the example mean. Batch size is deliberately allowed to change.
    example.fill_(.92)
    horizon.fill_(.25)
    with patch("tdn.research.work_precision.phi", side_effect=AssertionError("phi rebuilt")), \
         patch("tdn.research.work_precision.diffusion_eigenvalues", side_effect=AssertionError("spectrum rebuilt")):
        actual = stepper(changed)
    torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
    for value, snapshot in zip(tensors, snapshots):
        assert torch.equal(value, snapshot)
    assert stepper.metadata["h"] == .04
    assert stepper.metadata["cached_tensor_bytes"] > 0
    assert stepper.metadata["setup_fft_total"] == 0
    assert stepper.metadata["state_dependent_cache"] is False


@pytest.mark.parametrize("variant", VARIANTS)
def test_counter_dictionary_reports_actual_fft_calls_across_repeated_steps(variant):
    geometry, equation = Geometry((16,), (1.,)), Equation(.017, 3.)
    state = _state()
    with patch("torch.fft.fftn", wraps=torch.fft.fftn) as forward, \
         patch("torch.fft.ifftn", wraps=torch.fft.ifftn) as inverse:
        stepper = prepare_step(state, .025, equation, geometry, variant)
        assert forward.call_count == inverse.call_count == 0
        work = {"external": 7}
        for _ in range(3):
            state = stepper(state, work)
        assert work["fft_forward"] == forward.call_count
        assert work["fft_inverse"] == inverse.call_count
        assert work["fft_total"] == forward.call_count + inverse.call_count
        assert work["external"] == 7


@pytest.mark.parametrize("nodes", [3, 5])
def test_spectral_defect_has_one_actual_forward_and_zero_inverse(nodes):
    geometry, equation = Geometry((16,), (1.,)), Equation(.017, 3.)
    state, work = _state(), {}
    with patch("torch.fft.fftn", wraps=torch.fft.fftn) as forward, \
         patch("torch.fft.ifftn", wraps=torch.fft.ifftn) as inverse:
        spectral_mean_defect(state, .025, equation, geometry, nodes=nodes, work=work)
    assert work["fft_forward"] == forward.call_count == 1
    assert work["fft_inverse"] == inverse.call_count == 0


def test_failed_step_counts_only_completed_operations_and_bad_input_does_no_work():
    geometry, equation = Geometry((16,), (1.,)), Equation(.02, 3.)
    state = _state()
    stepper = prepare_step(state, .025, equation, geometry, "gl3_mean_spectral")
    work = {}
    with patch("torch.fft.ifftn", side_effect=RuntimeError("deliberate transform failure")):
        with pytest.raises(RuntimeError, match="deliberate"):
            stepper(state, work)
    assert work["fft_forward"] == work["fft_total"] == 1
    assert work["fft_inverse"] == 0
    assert work["reaction_evaluations"] == 1
    before = dict(work)
    with pytest.raises(ValueError, match="nonfinite"):
        stepper(state * float("nan"), work)
    assert work == before


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("equation,h", [(Equation(.02, 3.), 0.), (Equation(0., 3.), .03), (Equation(.02, 0.), .03)])
def test_prepared_exact_limits_match_their_direct_controls(variant, equation, h):
    geometry, state = Geometry((16,), (1.,)), _state()
    work = {}
    actual = prepare_step(state, h, equation, geometry, variant)(state, work)
    expected = state if h == 0 else _direct(variant, state, h, equation, geometry)
    torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
    if h == 0 or equation.kappa == 0:
        assert work["fft_total"] == 0


def test_preparation_and_repeated_calls_reject_invalid_contracts():
    geometry, equation, state = Geometry((16,), (1.,)), Equation(.02, 3.), _state()
    for h in (-.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            prepare_step(state, h, equation, geometry, "gl3")
    with pytest.raises(ValueError, match="scalar"):
        prepare_step(state, torch.tensor([.1, .1, .1]), equation, geometry, "gl3")
    with pytest.raises(ValueError, match="frozen"):
        prepare_step(state, torch.tensor(.1, requires_grad=True), equation, geometry, "gl3")
    with pytest.raises(ValueError, match="variant"):
        prepare_step(state, .1, equation, geometry, "unknown")
    with pytest.raises(TypeError, match="FP32 or FP64"):
        prepare_step(state.half(), .1, equation, geometry, "gl3")
    stepper = prepare_step(state, .1, equation, geometry, "gl3")
    for invalid in (state.float(), state[..., :-1], state[:, :, None], state * 0 + 1.1):
        with pytest.raises(ValueError):
            stepper(invalid)
    with pytest.raises(ValueError, match="dtype and device"):
        stepper(torch.empty(state.shape, dtype=state.dtype, device="meta"))
    for nodes in (True, 4, 3.):
        with pytest.raises(ValueError, match="nodes"):
            spectral_mean_defect(state, .1, equation, geometry, nodes=nodes)
