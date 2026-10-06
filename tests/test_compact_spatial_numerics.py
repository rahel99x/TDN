"""Parity, structural invariants and actual-operation checks for compression."""
from __future__ import annotations

import math
from unittest.mock import patch

import pytest
import torch

from tdn.numerics.splitting import split_step
from tdn.numerics.types import Equation, Geometry
from tdn.research.compact_spatial import (
    VARIANTS, _tensors, compact_defect, prepare_compact_step,
)
from tdn.research.interaction import interaction_defect
from tdn.research.work_precision import prepare_step, spectral_mean_defect


def _state(grid=(32,), dtype=torch.float64, batch=3, high=False):
    axes = [2 * torch.pi * torch.arange(n, dtype=dtype) / n for n in grid]
    mesh = torch.meshgrid(*axes, indexing="ij")
    pattern = sum(.04 * torch.cos(x) + .025 * torch.sin(2 * x) for x in mesh)
    if high:
        pattern = pattern + .013 * torch.cos((grid[0] // 2) * mesh[0])
        if len(grid) > 1:
            pattern = pattern + .011 * torch.sin(3 * mesh[0] + 4 * mesh[1])
    means = torch.linspace(.24, .73, batch, dtype=dtype).reshape(batch, 1, *([1] * len(grid)))
    return means + pattern


@pytest.mark.parametrize("grid", [(17,), (24,), (7, 10)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fused_raw_defect_and_repeated_steps_match_original_gl3(grid, dtype):
    geometry, equation = Geometry(grid, (1.3,) * len(grid)), Equation(.017, 2.8)
    u = _state(grid, dtype, high=True)
    fused = prepare_compact_step(u, .065, equation, geometry, "gl3_fused")
    raw = fused.defect(u)
    original = interaction_defect(u, .065, equation, geometry)
    torch.testing.assert_close(raw, original, rtol=2e-5 if dtype == torch.float32 else 2e-12,
                               atol=2e-10 if dtype == torch.float32 else 2e-18)
    baseline = prepare_step(u, .065, equation, geometry, "gl3")
    a, b = u.clone(), u.clone()
    for _ in range(4):
        a, b = fused(a), baseline(b)
    torch.testing.assert_close(a, b, rtol=2e-6 if dtype == torch.float32 else 2e-13,
                               atol=2e-7 if dtype == torch.float32 else 3e-16)


@pytest.mark.parametrize("grid", [(16,), (17,), (7, 10)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("variant", ["compact_gl3", "compact_gl3_no_mean"])
def test_full_band_compact_correction_includes_nyquist_and_matches_original(grid, dtype, variant):
    geometry, equation = Geometry(grid, (1.,) * len(grid)), Equation(.023, 3.)
    u = _state(grid, dtype, high=True)
    compact = prepare_compact_step(u, .09, equation, geometry, variant, modes=1000)
    expected = interaction_defect(u, .09, equation, geometry)
    if variant.endswith("no_mean"):
        expected = expected - expected.mean(dim=tuple(range(2, u.ndim)), keepdim=True)
    torch.testing.assert_close(compact.defect(u), expected,
                               rtol=2e-5 if dtype == torch.float32 else 5e-12,
                               atol=3e-10 if dtype == torch.float32 else 5e-18)
    assert compact.metadata["compact_grid"] == list(grid)


@pytest.mark.parametrize("grid", [(32,), (24, 32)])
def test_bandlimited_input_compact_keeps_all_generated_quadratic_modes(grid):
    geometry, equation = Geometry(grid, (1.,) * len(grid)), Equation(.019, 3.1)
    u = _state(grid)
    result = compact_defect(u, .1, equation, geometry, modes=2)
    expected = interaction_defect(u, .1, equation, geometry)
    torch.testing.assert_close(result, expected, rtol=2e-11, atol=2e-17)
    # Input cutoff 2 generates output frequencies up to 4; simply cropping
    # the output to the input cutoff would incorrectly discard this energy.
    if len(grid) == 1:
        assert float(torch.fft.fftn(result, dim=(-1,))[..., 4].abs().max()) > 1e-6


@pytest.mark.parametrize("grid", [(32,), (18, 23)])
def test_exact_mean_and_zero_mean_ablation_are_separated_per_parent(grid):
    geometry, equation = Geometry(grid, (1.,) * len(grid)), Equation(.017, 2.4)
    u = _state(grid, high=True)
    full = compact_defect(u, .08, equation, geometry, modes=2)
    spatial = compact_defect(u, .08, equation, geometry, variant="compact_gl3_no_mean", modes=2)
    mean = spectral_mean_defect(u, .08, equation, geometry)
    axes = tuple(range(2, u.ndim))
    torch.testing.assert_close(spatial.mean(dim=axes, keepdim=True), torch.zeros_like(mean), rtol=0., atol=2e-20)
    torch.testing.assert_close(full - spatial, mean.expand_as(full), rtol=1e-14, atol=2e-20)
    torch.testing.assert_close(full.mean(dim=axes, keepdim=True), mean, rtol=2e-14, atol=2e-20)
    zero = compact_defect(u, .08, equation, geometry, modes=0)
    torch.testing.assert_close(zero, mean.expand_as(zero), rtol=0., atol=0.)
    zero_spatial = compact_defect(u, .08, equation, geometry, variant="compact_gl3_no_mean", modes=0)
    assert torch.equal(zero_spatial, torch.zeros_like(zero_spatial))


@pytest.mark.parametrize("variant", VARIANTS)
def test_translation_equivariance_native_batch_and_parent_permutation(variant):
    geometry, equation = Geometry((18, 23), (1.2, .9)), Equation(.015, 4.)
    u = _state(geometry.grid, high=True)
    step = prepare_compact_step(u[:1], .065, equation, geometry, variant, modes=(2, 3))
    answer = step(u)
    shifted = step(torch.roll(u, shifts=(5, -7), dims=(-2, -1)))
    torch.testing.assert_close(shifted, torch.roll(answer, shifts=(5, -7), dims=(-2, -1)),
                               rtol=0., atol=16 * torch.finfo(u.dtype).eps)
    raw = step.defect(u)
    shifted_raw = step.defect(torch.roll(u, shifts=(5, -7), dims=(-2, -1)))
    torch.testing.assert_close(shifted_raw, torch.roll(raw, shifts=(5, -7), dims=(-2, -1)),
                               rtol=2e-11, atol=2e-18)
    singles = torch.cat([step(parent[None]) for parent in u])
    torch.testing.assert_close(answer, singles, rtol=0., atol=0.)
    permutation = torch.tensor([2, 0, 1])
    torch.testing.assert_close(step(u[permutation]), answer[permutation], rtol=0., atol=0.)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exact_limits_and_uniform_batch_members(variant, dtype):
    geometry = Geometry((24,), (1.,))
    u = _state(geometry.grid, dtype)
    for equation, h in [(Equation(.02, 4.), 0.), (Equation(0., 4.), .2), (Equation(.02, 0.), .2)]:
        step = prepare_compact_step(u, h, equation, geometry, variant, modes=2)
        work = {}
        assert torch.equal(step.defect(u, work), torch.zeros_like(u))
        assert work["fft_total"] == work["fft_total_fields"] == 0
        expected = u if h == 0 else split_step(u, h, equation, geometry)
        torch.testing.assert_close(step(u), expected, rtol=0., atol=0.)
    equation = Equation(.02, 4.)
    for value in (0., .375, 1.):
        uniform = torch.full_like(u, value)
        step = prepare_compact_step(uniform, .2, equation, geometry, variant, modes=2)
        work = {}
        assert torch.equal(step.defect(uniform, work), torch.zeros_like(uniform))
        assert work["fft_total"] == 0
    # Endpoints at large r*h have singular intermediate logistic derivatives,
    # but they must not contaminate nonuniform parents in the same batch.
    mixed = torch.cat((torch.zeros_like(u[:1]), u[1:2], torch.ones_like(u[:1])))
    step = prepare_compact_step(mixed, 20., Equation(.02, 100.), geometry, variant, modes=2)
    raw = step.defect(mixed)
    assert torch.isfinite(raw).all()
    assert torch.equal(raw[0], torch.zeros_like(raw[0]))
    assert torch.equal(raw[-1], torch.zeros_like(raw[-1]))


@pytest.mark.parametrize("variant", VARIANTS)
def test_preparation_owns_no_state_and_reuses_native_and_reduction_coefficients(variant):
    geometry, equation = Geometry((32,), (1.,)), Equation(.019, 3.)
    example = _state(batch=1)
    h = torch.tensor(.08, dtype=example.dtype)
    step = prepare_compact_step(example, h, equation, geometry, variant, modes=2)
    values = list(_tensors((step._p, step._spatial, step._indices, step._mask, step._flat_indices)))
    snapshots = [value.clone() for value in values]
    changed = _state(batch=4, high=True).flip(-1)
    expected = prepare_compact_step(changed, .08, equation, geometry, variant, modes=2)(changed)
    example.fill_(.91)
    h.fill_(.4)
    with patch("tdn.research.compact_spatial.diffusion_eigenvalues", side_effect=AssertionError("spectrum rebuilt")), \
         patch("tdn.research.compact_spatial.phi", side_effect=AssertionError("phi rebuilt")):
        actual = step(changed)
    torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
    for value, snapshot in zip(values, snapshots):
        assert torch.equal(value, snapshot)
    assert step.metadata["h"] == .08
    assert step.metadata["state_dependent_cache"] is False
    assert step.metadata["setup_fft_total"] == 0
    assert step.metadata["cached_tensor_bytes"] > 0


@pytest.mark.parametrize("variant", VARIANTS)
def test_work_records_actual_fft_calls_field_counts_and_cells(variant):
    geometry, equation = Geometry((40, 32), (1., 1.)), Equation(.013, 2.7)
    u = _state(geometry.grid, batch=3, high=True)
    observed = {"forward": [], "inverse": []}
    forward, inverse = torch.fft.fftn, torch.fft.ifftn

    def recording(function, direction):
        def wrapped(value, *args, **kwargs):
            dims = kwargs["dim"]
            size = math.prod(value.shape[axis] for axis in dims)
            observed[direction].append((value.numel() // size, value.numel()))
            return function(value, *args, **kwargs)
        return wrapped

    with patch("torch.fft.fftn", side_effect=recording(forward, "forward")), \
         patch("torch.fft.ifftn", side_effect=recording(inverse, "inverse")):
        step = prepare_compact_step(u, .02, equation, geometry, variant, modes=2)
        assert observed == {"forward": [], "inverse": []}
        work = {"external": 17}
        for _ in range(2):
            u = step(u, work)
    assert work["external"] == 17
    for direction in observed:
        assert work[f"fft_{direction}"] == len(observed[direction])
        assert work[f"fft_{direction}_fields"] == sum(item[0] for item in observed[direction])
    assert work["fft_total"] == sum(len(records) for records in observed.values())
    assert work["fft_total_fields"] == sum(item[0] for records in observed.values() for item in records)
    assert work["fft_transformed_cells"] == sum(item[1] for records in observed.values() for item in records)
    if variant == "gl3_fused":
        assert work["fft_total"] == 14  # two steps, seven API calls each
        assert work["fft_total_fields"] == 2 * 17 * 3
    else:
        assert work["fft_total"] == 16
        assert work["fft_transformed_cells"] == 2 * 3 * (4 * 40 * 32 + 14 * 9 * 9)


def test_failed_fft_is_not_counted_but_prior_operations_are_retained():
    geometry, equation = Geometry((32,), (1.,)), Equation(.017, 3.)
    u = _state()
    step = prepare_compact_step(u, .08, equation, geometry)
    work = {}
    with patch("torch.fft.ifftn", side_effect=RuntimeError("injected transform failure")):
        with pytest.raises(RuntimeError, match="injected transform failure"):
            step(u, work)
    assert work["fft_forward"] == 1
    assert work["fft_inverse"] == 0
    assert work["fft_total_fields"] == u.shape[0]
    assert work["reaction_evaluations"] == 1


@pytest.mark.parametrize("modes", [-1, True, 2.5, (1, 2), [2]])
def test_invalid_modes_rejected(modes):
    u = _state()
    with pytest.raises(ValueError):
        prepare_compact_step(u, .01, Equation(.01, 2.), Geometry((32,), (1.,)), modes=modes)


def test_invalid_horizon_dtype_geometry_and_state_rejected():
    geometry, equation = Geometry((32,), (1.,)), Equation(.017, 3.)
    u = _state()
    for h in (-.1, float("nan"), torch.tensor([.1, .1, .1]), torch.tensor(.1, requires_grad=True)):
        with pytest.raises(ValueError):
            prepare_compact_step(u, h, equation, geometry)
    step = prepare_compact_step(u, .01, equation, geometry)
    for bad in (u.float(), torch.zeros((3, 1, 31), dtype=u.dtype), u * 0 + 1.1, u * float("nan")):
        with pytest.raises(ValueError):
            step(bad)
    with pytest.raises(ValueError):
        prepare_compact_step(u, .01, equation, geometry, "unknown")
