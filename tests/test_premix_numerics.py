"""Independent formula, representation, invariance and work audits for premixing."""
from __future__ import annotations

import math
from unittest.mock import patch

import pytest
import torch

from tdn.numerics.splitting import split_step
from tdn.numerics.types import Equation, Geometry
from tdn.research.compact_spatial import prepare_compact_step
from tdn.research.interaction import interaction_defect
from tdn.research.premix import (
    VARIANTS, _cached_tensors, _transported_difference, prepare_premix_step,
)
from tdn.research.work_precision import spectral_mean_defect


def state(grid=(24,), dtype=torch.float64, batch=3):
    axes = [2 * torch.pi * torch.arange(n, dtype=dtype) / n for n in grid]
    mesh = torch.meshgrid(*axes, indexing="ij")
    pattern = sum(.025 * torch.cos(x) + .015 * torch.sin(3 * x) + .011 * torch.cos((n // 2) * x)
                  for x, n in zip(mesh, grid))
    means = torch.linspace(.22, .76, batch, dtype=dtype).reshape(batch, 1, *([1] * len(grid)))
    return means + pattern


def project(value, modes):
    grid = value.shape[2:]
    modes = (modes,) * len(grid) if isinstance(modes, int) else modes
    mask = torch.ones(grid, dtype=torch.bool, device=value.device)
    for axis, (n, k) in enumerate(zip(grid, modes)):
        frequency = torch.fft.fftfreq(n, d=1 / n, device=value.device)
        shape = [1] * len(grid)
        shape[axis] = n
        mask = mask & (frequency.abs() <= k).reshape(shape)
    return torch.fft.ifftn(torch.fft.fftn(value, dim=tuple(range(2, value.ndim))) * mask,
                           dim=tuple(range(2, value.ndim))).real


@pytest.mark.parametrize("grid", [(15,), (16,), (7, 10)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("variant", VARIANTS)
def test_full_retention_matches_independent_unprepared_gl3(grid, dtype, variant):
    u = state(grid, dtype)
    equation, geometry = Equation(.017, 2.8), Geometry(grid, (1.2,) * len(grid))
    prepared = prepare_premix_step(u, .07, equation, geometry, variant, modes=1000, chunk_size=3)
    expected = interaction_defect(u, .07, equation, geometry)
    torch.testing.assert_close(prepared.defect(u), expected,
                               rtol=3e-5 if dtype == torch.float32 else 8e-12,
                               atol=3e-10 if dtype == torch.float32 else 3e-18)
    full = prepare_compact_step(u, .07, equation, geometry, "gl3_fused")
    a, b = u, u
    for _ in range(3):
        a, b = prepared(a), full(b)
    torch.testing.assert_close(a, b, rtol=2e-6 if dtype == torch.float32 else 2e-13,
                               atol=2e-7 if dtype == torch.float32 else 4e-16)


@pytest.mark.parametrize("grid,modes", [((23,), 0), ((24,), 2), ((9, 12), (2, 3))])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("variant", ["output_gl3", "selected_output_gl3"])
def test_output_selection_equals_projection_of_full_correction(grid, modes, dtype, variant):
    u = state(grid, dtype)
    equation, geometry = Equation(.024, 3.1), Geometry(grid, (1.1,) * len(grid))
    expected = project(interaction_defect(u, .08, equation, geometry), modes)
    prepared = prepare_premix_step(u, .08, equation, geometry, variant, modes=modes, chunk_size=2)
    torch.testing.assert_close(prepared.defect(u), expected,
                               rtol=3e-5 if dtype == torch.float32 else 8e-12,
                               atol=2e-10 if dtype == torch.float32 else 4e-18)
    assert prepared.metadata["cutoff_semantics"].startswith("axis-wise absolute Fourier output")


@pytest.mark.parametrize("grid", [(32,), (32, 20)])
def test_high_high_to_low_interaction_survives_output_compression(grid):
    x = 2 * torch.pi * torch.arange(grid[0], dtype=torch.float64) / grid[0]
    wave = .035 * torch.cos(9 * x) + .024 * torch.cos(10 * x)
    wave = wave if len(grid) == 1 else wave[:, None].expand(grid)
    u = (.4 + wave).reshape(1, 1, *grid)
    equation, geometry = Equation(.003, 3.), Geometry(grid, (1.,) * len(grid))
    full = interaction_defect(u, .1, equation, geometry)
    premix = prepare_premix_step(u, .1, equation, geometry, modes=2).defect(u)
    compact = prepare_compact_step(u, .1, equation, geometry, modes=2).defect(u)
    axes = tuple(range(2, u.ndim))
    index = (0, 0, 1) + (0,) * (len(grid) - 1)
    full_mode, new_mode, old_mode = [torch.fft.fftn(value, dim=axes)[index] for value in (full, premix, compact)]
    assert full_mode.abs() > 1e-5
    torch.testing.assert_close(new_mode, full_mode, rtol=2e-12, atol=1e-17)
    assert old_mode.abs() < 1e-15


def test_original_grid_cyclic_alias_is_preserved():
    n = 16
    x = 2 * torch.pi * torch.arange(n, dtype=torch.float64) / n
    u = (.43 + .03 * torch.cos(7 * x)).reshape(1, 1, n)
    equation, geometry = Equation(.014, 2.7), Geometry((n,), (1.,))
    raw = prepare_premix_step(u, .09, equation, geometry, modes=2).defect(u)
    expected = project(interaction_defect(u, .09, equation, geometry), 2)
    torch.testing.assert_close(raw, expected, rtol=1e-11, atol=1e-18)
    # 7+7=14=-2 cyclically. A continuum/dealiased or low-input computation
    # would incorrectly erase this retained output interaction.
    assert torch.fft.fftn(raw, dim=(-1,))[..., 2].abs().min() > 1e-6


@pytest.mark.parametrize("grid", [(24,), (9, 12)])
def test_zero_output_mode_is_full_spectrum_mean_for_each_parent(grid):
    u = state(grid)
    equation, geometry = Equation(.024, 2.4), Geometry(grid, (1.,) * len(grid))
    raw = prepare_premix_step(u, .06, equation, geometry, modes=0).defect(u)
    expected = spectral_mean_defect(u, .06, equation, geometry)
    torch.testing.assert_close(raw, expected.expand_as(u), rtol=2e-11, atol=2e-18)


@pytest.mark.parametrize("variant", VARIANTS)
def test_translations_parent_permutations_batch_partition_and_chunk_choices(variant):
    u = state((9, 12), batch=4)
    equation, geometry = Equation(.02, 3.2), Geometry((9, 12), (1.2, .8))
    prepared = prepare_premix_step(u[:1], .09, equation, geometry, variant, modes=(2, 3), chunk_size=2)
    raw = prepared.defect(u)
    shifted = prepared.defect(torch.roll(u, (2, -5), (-2, -1)))
    torch.testing.assert_close(shifted, torch.roll(raw, (2, -5), (-2, -1)), rtol=1e-10, atol=3e-18)
    permutation = torch.tensor([3, 1, 0, 2])
    torch.testing.assert_close(prepared(u[permutation]), prepared(u)[permutation], rtol=0., atol=0.)
    individual = torch.cat([prepared(parent[None]) for parent in u])
    torch.testing.assert_close(prepared(u), individual, rtol=0., atol=0.)
    for chunk in (1, 8, 100):
        other = prepare_premix_step(u, .09, equation, geometry, variant, modes=(2, 3), chunk_size=chunk)
        torch.testing.assert_close(other.defect(u), raw, rtol=2e-12, atol=2e-18)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exact_limits_and_mixed_uniform_members(variant, dtype):
    u = state(dtype=dtype)
    geometry = Geometry((24,), (1.,))
    for h, equation in [(0., Equation(.01, 2.)), (.08, Equation(0., 2.)), (.08, Equation(.01, 0.))]:
        prepared = prepare_premix_step(u, h, equation, geometry, variant)
        work = {}
        assert torch.equal(prepared.defect(u, work), u * 0)
        assert work["fft_total"] == work["pair_product_cells"] == 0
        expected = u if h == 0 else split_step(u, h, equation, geometry)
        torch.testing.assert_close(prepared(u), expected, rtol=0., atol=0.)
    mixed = torch.cat((u[:1] * 0, u[1:2], u[:1] * 0 + 1))
    prepared = prepare_premix_step(mixed, 20., Equation(.02, 100.), geometry, variant)
    raw = prepared.defect(mixed)
    assert torch.isfinite(raw).all()
    assert torch.equal(raw[0], raw[0] * 0)
    assert torch.equal(raw[2], raw[2] * 0)
    work = {}
    assert torch.equal(prepared.defect(torch.ones_like(u), work), u * 0)
    assert work["fft_total"] == work["pair_product_cells"] == 0


def test_exponential_kernel_is_stable_at_small_time_and_extreme_stiffness():
    a = torch.tensor([0., -1e10, -3., -3., -3.0000000001], dtype=torch.float64)
    b = torch.tensor([-1e10, 0., -4., -3., -3.], dtype=torch.float64)
    result = _transported_difference(a, b, torch.tensor(.1), torch.tensor(.1))
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result[:2], torch.tensor([1., -1.], dtype=torch.float64), rtol=0., atol=0.)
    assert result[3] == 0
    s, h = torch.tensor(1e-12, dtype=torch.float64), torch.tensor(2e-12, dtype=torch.float64)
    result = _transported_difference(a[2:], b[2:], s, h)
    torch.testing.assert_close(result / s, a[2:] - b[2:], rtol=1e-10, atol=0.)


def test_fp32_pair_reduction_preserves_small_time_correction():
    u = state(dtype=torch.float32, batch=1)
    geometry, equation = Geometry((24,), (1.,)), Equation(.013, 2.4)
    for h in (1e-3, 1e-5):
        native = prepare_premix_step(u, h, equation, geometry, modes=3).defect(u)
        reference = prepare_premix_step(u.double(), float(torch.tensor(h, dtype=u.dtype)), equation, geometry, modes=3).defect(u.double())
        assert native.abs().max() > 0
        torch.testing.assert_close(native.double(), reference, rtol=2e-3, atol=float(reference.abs().max()) * 2e-5)


@pytest.mark.parametrize("variant", VARIANTS)
def test_preparation_caches_no_state_and_never_builds_pair_table(variant):
    u = state(batch=1)
    h = torch.tensor(.09, dtype=u.dtype)
    equation, geometry = Equation(.014, 3.2), Geometry((24,), (1.,))
    prepared = prepare_premix_step(u, h, equation, geometry, variant, modes=3)
    tensors = list(_cached_tensors(tuple(vars(prepared).values())))
    snapshots = [x.clone() for x in tensors]
    expected = prepared(state(batch=4))
    u.fill_(.99)
    h.fill_(.7)
    with patch("tdn.research.premix.diffusion_eigenvalues", side_effect=AssertionError("spectrum rebuilt")), \
         patch("tdn.research.premix.phi", side_effect=AssertionError("phi rebuilt")):
        torch.testing.assert_close(prepared(state(batch=4)), expected, rtol=0., atol=0.)
    assert prepared.metadata["h"] == .09
    assert prepared.metadata["pair_kernel_cache"] is False
    assert prepared.metadata["state_dependent_cache"] is False
    assert prepared.metadata["cached_tensor_bytes"] > 0
    for tensor, snapshot in zip(tensors, snapshots):
        assert torch.equal(tensor, snapshot)
    if variant == "selected_output_gl3":
        assert max(tensor.numel() for tensor in tensors) <= u.shape[-1]


@pytest.mark.parametrize("variant", VARIANTS)
def test_work_counters_match_actual_transforms_and_pair_operations(variant):
    u = state((9, 12), batch=5)
    equation, geometry = Equation(.013, 2.9), Geometry((9, 12), (1., 1.))
    observed = {"forward": [], "inverse": []}
    def record(function, direction):
        def wrapped(value, *args, **kwargs):
            cells = math.prod(value.shape[axis] for axis in kwargs["dim"])
            observed[direction].append((value.numel() // cells, value.numel()))
            return function(value, *args, **kwargs)
        return wrapped
    with patch("torch.fft.fftn", side_effect=record(torch.fft.fftn, "forward")), \
         patch("torch.fft.ifftn", side_effect=record(torch.fft.ifftn, "inverse")):
        prepared = prepare_premix_step(u, .025, equation, geometry, variant, modes=(2, 3), chunk_size=3)
        assert observed == {"forward": [], "inverse": []}
        work = {"external": 7}
        for _ in range(2):
            u = prepared(u, work)
    assert work["external"] == 7
    for direction in observed:
        assert work[f"fft_{direction}"] == len(observed[direction])
        assert work[f"fft_{direction}_fields"] == sum(item[0] for item in observed[direction])
    assert work["fft_total"] == sum(len(values) for values in observed.values())
    assert work["fft_total_fields"] == sum(item[0] for values in observed.values() for item in values)
    assert work["fft_transformed_cells"] == sum(item[1] for values in observed.values() for item in values)
    if variant == "selected_output_gl3":
        assert work["fft_total"] == 8
        assert work["pair_product_cells"] == 2 * 5 * (9 * 12) * (5 * 7)
        assert work["pair_kernel_values"] == 2 * 4 * (9 * 12) * (5 * 7)
        assert work["pair_chunks"] == 2 * math.ceil(35 / 3)
    elif variant == "gl3_fused_chunked":
        assert work["fft_total"] == 2 * 2 * 7
        assert work["fft_total_fields"] == 2 * 5 * 17
        assert work["parent_chunks"] == 4
    else:
        assert work["output_projection_cells"] == 2 * u.numel()
        assert work["fft_total"] == 14


def test_work_keeps_successful_prefix_on_failure():
    u = state()
    prepared = prepare_premix_step(u, .04, Equation(.01, 2.), Geometry((24,), (1.,)))
    work = {}
    with patch("torch.fft.ifftn", side_effect=RuntimeError("transform failed")):
        with pytest.raises(RuntimeError, match="transform failed"):
            prepared.defect(u, work)
    assert work["fft_forward"] == 1
    assert work["fft_inverse"] == 0
    assert work["pair_product_cells"] == 3 * 24 * 9


@pytest.mark.parametrize("variant", VARIANTS)
def test_state_gradients_match_full_output_projection(variant):
    u = state((9,), batch=1).requires_grad_()
    equation, geometry = Equation(.013, 2.8), Geometry((9,), (1.,))
    prepared = prepare_premix_step(u, .1, equation, geometry, variant, modes=2, chunk_size=2)
    raw = prepared.defect(u)
    expected = interaction_defect(u, .1, equation, geometry)
    if variant != "gl3_fused_chunked":
        expected = project(expected, 2)
    weights = torch.linspace(-.7, .9, u.numel(), dtype=u.dtype).reshape_as(u)
    actual_gradient, = torch.autograd.grad((raw * weights).sum(), u)
    expected_gradient, = torch.autograd.grad((expected * weights).sum(), u)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-9, atol=3e-17)


@pytest.mark.parametrize("chunk_size", [0, -1, True, 1.5])
def test_invalid_chunk_size_rejected(chunk_size):
    with pytest.raises(ValueError, match="chunk_size"):
        prepare_premix_step(state(), .1, Equation(.01, 2.), Geometry((24,), (1.,)), chunk_size=chunk_size)


@pytest.mark.parametrize("modes", [-1, True, 2.5, (1, 2), [2]])
def test_invalid_cutoffs_rejected(modes):
    with pytest.raises(ValueError):
        prepare_premix_step(state(), .1, Equation(.01, 2.), Geometry((24,), (1.,)), modes=modes)


def test_bad_horizon_variant_dtype_geometry_and_state_are_rejected():
    u = state()
    equation, geometry = Equation(.01, 2.), Geometry((24,), (1.,))
    for h in (-.1, float("nan"), torch.tensor([.1, .1, .1]), torch.tensor(.1, requires_grad=True)):
        with pytest.raises(ValueError):
            prepare_premix_step(u, h, equation, geometry)
    prepared = prepare_premix_step(u, .1, equation, geometry)
    for bad in (u.float(), u[..., :-1], u * 0 + 1.1, u * float("nan")):
        with pytest.raises(ValueError):
            prepared(bad)
    with pytest.raises(ValueError):
        prepare_premix_step(u, .1, equation, geometry, "unknown")
