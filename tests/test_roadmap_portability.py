"""Numerical properties of the explicitly declared portability targets."""
import math

import numpy as np
import pytest
import torch

from tdn.analysis.roadmap.portability import (
    FieldSpec, burgers_rhs, burgers_split, c4_logistic_step, coupled_rhs,
    coupled_split, diffusion_symbol, factor_bank, fractional_feature,
    fractional_filter, geometry_corrected_step, geometry_operator,
    geometry_split, independent_reference, logistic_split, periodic_logistic_rhs,
    product, quadratic_defect, resample, richardson_step, sample_field, transport,
    wavevectors, weighted_filter,
)


@pytest.fixture(autouse=True)
def modest_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_same_continuous_field_resampling_and_independent_clusters():
    specs = factor_bank(3)
    assert len({s.seed for s in specs}) == 3
    base = specs[0]
    torch.testing.assert_close(resample(sample_field(base, 12), 24), sample_field(base, 24), atol=2e-15, rtol=0)
    normal = sample_field(base, 24)
    rough = sample_field(next(s for s in specs if s.variant == "roughness_variance_matched"), 24)
    torch.testing.assert_close(normal.var(unbiased=False), rough.var(unbiased=False), atol=2e-17, rtol=0)
    for spec in specs:
        assert len(spec.changed_factors) <= 1 or spec.variant == "amplitude_roughness"
    assert not torch.allclose(normal, sample_field(next(s for s in specs if s.cluster == 1), 24))


@pytest.mark.parametrize("dimension", [1, 2])
def test_dealiased_product_exposes_folded_low_output(dimension):
    n = 12
    x = torch.arange(n, dtype=torch.float64) / n
    if dimension == 2:
        x = x[:, None].expand(n, n)
    u = torch.cos(2 * math.pi * 4 * x)
    clean = product(u, u)
    nodal = product(u, u, dealiased=False)
    torch.testing.assert_close(clean, torch.full_like(u, 0.5), atol=3e-15, rtol=0)
    assert (nodal - clean).abs().max() > 0.49


def test_fractional_physical_units_and_constant_null():
    outputs = []
    for n in (16, 32):
        x = torch.arange(n, dtype=torch.float64) / n
        u = torch.cos(2 * math.pi * 3 * x)
        outputs.append(fractional_filter(u, 0.07))
        torch.testing.assert_close(fractional_feature(u), (6 * math.pi) ** 1.5 * u, atol=3e-12, rtol=2e-13)
        assert fractional_feature(torch.ones(n, dtype=torch.float64)).abs().max() == 0
    torch.testing.assert_close(outputs[0], outputs[1][::2], atol=2e-15, rtol=0)


def test_anisotropic_tensor_rotation_and_translation_covariance():
    n = 16
    u = sample_field(FieldSpec(0, 8743, "rotation"), n)
    tensor = torch.tensor([[0.02, 0.006], [0.006, 0.008]], dtype=torch.float64)
    rotation = torch.tensor([[0., 1.], [-1., 0.]], dtype=torch.float64)
    symbol = diffusion_symbol(n, tensor)
    rotated_symbol = diffusion_symbol(n, rotation.T @ tensor @ rotation)
    rotate = lambda z: z.T[:, (-torch.arange(n)) % n]
    actual, _ = c4_logistic_step(u, 0.08, symbol, 2)
    moved, _ = c4_logistic_step(rotate(u), 0.08, rotated_symbol, 2)
    torch.testing.assert_close(moved, rotate(actual), atol=3e-14, rtol=0)
    shifted, _ = c4_logistic_step(torch.roll(u, 3, 0), 0.08, symbol, 2)
    torch.testing.assert_close(shifted, torch.roll(actual, 3, 0), atol=3e-14, rtol=0)


@pytest.mark.parametrize("null", ["time", "reaction", "diffusion", "constant"])
def test_finite_time_source_physical_nulls(null):
    u = sample_field(FieldSpec(0, 1237, "null"), 12)
    symbol = diffusion_symbol(12, [[0.02, 0], [0, 0.01]])
    h, r = 0.05, 3.0
    if null == "time":
        h = 0
    elif null == "reaction":
        r = 0
    elif null == "diffusion":
        symbol.zero_()
    else:
        u.fill_(0.4)
    assert quadratic_defect(u, h, symbol, r).abs().max() < 2e-30


def test_quadratic_defect_reduces_small_amplitude_teacher_error():
    n = 12
    symbol = diffusion_symbol(n, [[0.03, 0.006], [0.006, 0.01]])
    u = sample_field(FieldSpec(0, 1237, "small", amplitude=0.005), n)
    h = 0.06
    ref, info = independent_reference(u.numpy(), h, periodic_logistic_rhs(symbol.numpy(), 2))
    base = logistic_split(u, h, symbol, 2)
    corrected, _ = c4_logistic_step(u, h, symbol, 2, fractional_length=0)
    error = torch.tensor(ref) - base
    improved = torch.tensor(ref) - corrected
    assert info["uncertainty_max"] < 1e-10
    assert improved.norm() < 0.2 * error.norm()


@pytest.mark.parametrize("equation", ["burgers", "coupled"])
def test_independent_coupled_teacher_and_split_refinement(equation):
    n = 16
    x = torch.arange(n, dtype=torch.float64) / n
    if equation == "burgers":
        initial = 0.2 + 0.15 * torch.sin(2 * math.pi * x)
        rhs, split = burgers_rhs(), burgers_split
    else:
        initial = torch.stack((0.75 + 0.04 * torch.cos(2 * math.pi * x), 0.2 + 0.03 * torch.sin(4 * math.pi * x)))
        rhs, split = coupled_rhs(), coupled_split
    ref, uncertainty = independent_reference(initial.numpy(), 0.1, rhs)
    single = split(initial, 0.1)
    half = split(split(initial, 0.05), 0.05)
    corrected, fallback = richardson_step(initial, 0.1, split)
    error = lambda z: np.linalg.norm(z.numpy() - ref)
    assert uncertainty["uncertainty_max"] < 1e-10
    assert error(half) < 0.4 * error(single)
    assert error(corrected) < 0.1 * error(single)
    assert not fallback


def test_periodic_burgers_mass_and_energy_and_coupled_feed_balance():
    n = 24
    x = torch.arange(n, dtype=torch.float64) / n
    u = 0.2 + 0.12 * torch.sin(2 * math.pi * x)
    out = burgers_split(u, 0.1)
    assert abs(float(out.mean() - u.mean())) < 1e-14
    assert out.square().mean() < u.square().mean()
    state = torch.stack((0.75 + 0.04 * torch.cos(2 * math.pi * x), 0.2 + 0.03 * torch.sin(4 * math.pi * x)))
    derivative = coupled_rhs()(state.numpy())
    balance = 0.04 * (1 - state[0].mean()) - 0.1 * state[1].mean()
    assert abs(float(derivative.sum(axis=0).mean()) - float(balance)) < 1e-14


def test_linear_advection_diffusion_has_exact_mode_response():
    n = 24
    x = torch.arange(n, dtype=torch.float64) / n
    k = wavevectors(n, 1)[0]
    state = torch.cos(2 * math.pi * 3 * x)
    h, nu, advection = 0.07, 0.01, 0.3
    actual = transport(state, h, -nu * k * k - 1j * advection * k)
    expected = math.exp(-nu * (6 * math.pi) ** 2 * h) * torch.cos(6 * math.pi * (x - advection * h))
    torch.testing.assert_close(actual, expected, atol=3e-15, rtol=0)


def test_nonuniform_geometry_weighted_measure_and_quadrature_convergence():
    errors = []
    for n in (17, 33):
        nodes, weights, a = geometry_operator(n)
        np.testing.assert_allclose(weights[:, None] * a, a.T * weights[None, :], atol=3e-16, rtol=0)
        assert np.min(a - np.diag(np.diag(a))) >= 0
        assert np.linalg.eigvalsh(np.sqrt(weights)[:, None] * a / np.sqrt(weights)[None, :]).max() < 0
        errors.append(abs(weights @ (nodes[1:-1] * (1 - nodes[1:-1])) - 1 / 6))
    assert errors[1] < 0.27 * errors[0]


def test_nonperiodic_heat_remeshing_converges_to_analytic_solution():
    errors = []
    for n in (17, 33):
        nodes, _, matrix = geometry_operator(n, diffusivity=0.04, variable=False)
        initial = torch.tensor(np.sin(math.pi * nodes[1:-1]), dtype=torch.float64)
        out = geometry_split(initial, 0.2, torch.tensor(matrix), reaction=0)
        expected = math.exp(-0.04 * math.pi ** 2 * 0.2) * initial
        errors.append(float((out - expected).abs().max()))
    assert errors[1] < 0.3 * errors[0]


def test_geometry_teacher_corrector_and_boundary_domain():
    nodes, weights, matrix = geometry_operator(17)
    initial = torch.tensor(0.45 * np.sin(math.pi * nodes[1:-1]))
    a, w = torch.tensor(matrix), torch.tensor(weights)
    reference, _ = independent_reference(initial.numpy(), 0.1, lambda u: matrix @ u + 2 * u * (1 - u))
    base = geometry_split(initial, 0.1, a)
    corrected, rejected = geometry_corrected_step(initial, 0.1, a, w, length=0)
    assert np.max(np.abs(corrected.numpy() - reference)) < 0.1 * np.max(np.abs(base.numpy() - reference))
    assert not rejected and corrected.min() > 0 and corrected.max() < 1
    delta = torch.sin(initial)
    torch.testing.assert_close(weighted_filter(delta, a, w, length=0), delta, atol=3e-15, rtol=0)


def test_teacher_free_guard_rejects_invalid_correction():
    u = torch.linspace(0.1, 0.4, 12, dtype=torch.float64)
    result, rejected = richardson_step(u, 0.1, lambda x, h: x + h * x.square(),
                                       guard=lambda candidate, old: False)
    torch.testing.assert_close(result, u + 0.1 * u.square())
    assert rejected


def test_reject_invalid_geometry_and_indefinite_diffusion():
    with pytest.raises(ValueError, match="positive semidefinite"):
        diffusion_symbol(12, [[0.01, 0.03], [0.03, 0.01]])
    with pytest.raises(ValueError):
        geometry_operator(12, warp=1.1)
