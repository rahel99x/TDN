"""Dense discrete-system oracle; no production spectrum, FFT, or GL tables."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.linalg import expm
import torch

from tdn.numerics.types import Equation, Geometry
from tdn.research.compact_spatial import prepare_compact_step


def _matrix(grid, lengths, kappa):
    """Construct the physical central-difference matrix directly."""
    matrix = np.zeros((math.prod(grid), math.prod(grid)))
    for axis, (n, length) in enumerate(zip(grid, lengths)):
        identity = np.eye(n)
        derivative = (np.roll(identity, 1, axis=0) - 2 * identity
                      + np.roll(identity, -1, axis=0)) / (length / n)**2
        factor = np.ones((1, 1))
        for index, cells in enumerate(grid):
            factor = np.kron(factor, derivative if index == axis else np.eye(cells))
        matrix += kappa * factor
    return matrix


def _projection(grid, cutoff):
    """Dense DFT projector on the ORIGINAL grid, including cyclic Nyquist."""
    result = np.ones((1, 1), dtype=np.complex128)
    for cells, keep in zip(grid, cutoff):
        indices = np.arange(cells)
        signed = np.where(indices <= (cells - 1) // 2, indices, indices - cells)
        transform = np.exp(-2j * np.pi * np.outer(indices, indices) / cells)
        retained = np.abs(signed) <= min(keep, cells // 2)
        axis = transform.conj().T @ np.diag(retained.astype(float)) @ transform / cells
        result = np.kron(result, axis)
    assert np.abs(result.imag).max() < 2e-14
    return result.real


def _dense_gl(v, mean, matrix, h, rate):
    """Full-grid quadratic variation with independent Gaussian quadrature."""
    locations, weights = np.polynomial.legendre.leggauss(3)

    def jacobian(s):
        q = math.exp(-rate * s)
        return q / (mean + (1 - mean) * q)**2

    integral = np.zeros_like(v)
    for x, weight in zip(locations, weights):
        s = h * (1 + x) / 2
        semigroup = expm(s * matrix)
        variation = (semigroup @ v)**2 - semigroup @ (v**2)
        integral += h * weight / 2 * jacobian(s) * (expm((h - s) * matrix) @ variation)
    full = expm(h * matrix)
    endpoint = (full @ v)**2 - full @ (v**2)
    qhalf, qfull = math.exp(-rate * h / 2), math.exp(-rate * h)
    denominator_half = mean + (1 - mean) * qhalf
    denominator_full = mean + (1 - mean) * qfull
    reaction_weight = (qhalf - qfull) / (rate * denominator_half * denominator_full)
    return -rate * jacobian(h) * (integral - reaction_weight * endpoint)


@pytest.mark.parametrize("grid,lengths,cutoff", [
    ((13,), (1.3,), (2,)),       # smaller odd compact grid and FFT normalization
    ((8,), (1.7,), (4,)),        # full cyclic band, including the Nyquist mode
    ((6, 11), (1.1, .8), (2, 2)),  # one cyclic axis, one alias-free compact axis
])
def test_compact_defect_matches_dense_fine_grid_oracle(grid, lengths, cutoff):
    equation, geometry = Equation(.012, 2.3), Geometry(grid, lengths)
    coordinate = np.indices(grid)
    # Deliberately include high-frequency and cross-axis content; no random
    # seed or solver helper determines the expected field.
    pattern = np.cos(2 * np.pi * (coordinate[0] / grid[0]) + .37)
    pattern += .6 * np.cos(2 * np.pi * (grid[0] // 2) * coordinate[0] / grid[0] + .21)
    if len(grid) == 2:
        pattern += .7 * np.sin(2 * np.pi * (2 * coordinate[0] / grid[0]
                                           + 3 * coordinate[1] / grid[1]) - .19)
    u = .43 + .08 * pattern
    mean, h = float(u.mean()), .17
    v = u.reshape(-1) - mean
    matrix = _matrix(grid, lengths, equation.kappa)
    retained = _projection(grid, cutoff) @ v
    exact_low = _dense_gl(retained, mean, matrix, h, equation.reaction_rate)
    exact_full = _dense_gl(v, mean, matrix, h, equation.reaction_rate)
    expected_spatial = exact_low - exact_low.mean()
    field = torch.tensor(u.reshape(1, 1, *grid), dtype=torch.float64)
    for variant in ("compact_gl3", "compact_gl3_no_mean"):
        expected = expected_spatial + (exact_full.mean() if variant == "compact_gl3" else 0.)
        prepared = prepare_compact_step(field, h, equation, geometry, variant, modes=cutoff)
        actual = prepared.defect(field).numpy().reshape(-1)
        assert np.max(np.abs(expected)) > 1e-8
        np.testing.assert_allclose(actual, expected, rtol=3e-10, atol=3e-16)


def test_discarded_high_pair_has_a_resolved_low_spatial_interaction():
    """Mean repair cannot recover discarded high-high-to-low mode coupling."""
    cells, cutoff = 32, 4
    geometry, equation = Geometry((cells,), (1.,)), Equation(.003, 3.1)
    x = torch.arange(cells, dtype=torch.float64) / cells
    field = (.42 + .10 * torch.cos(2 * torch.pi * 5 * x + .13)
             + .08 * torch.cos(2 * torch.pi * 6 * x - .41)).reshape(1, 1, cells)
    compact = prepare_compact_step(field, .16, equation, geometry, modes=cutoff).defect(field)
    full = prepare_compact_step(field, .16, equation, geometry, "gl3_fused").defect(field)
    # Use a direct dot-product mode projection, not FFT output or the
    # implementation's frequency indices, to observe the low difference mode.
    basis = torch.exp(-2j * torch.pi * x)
    full_mode_one = torch.sum(full.flatten() * basis) / cells
    compact_mode_one = torch.sum(compact.flatten() * basis) / cells
    assert float(torch.abs(full_mode_one)) > 1e-7
    assert float(torch.abs(compact_mode_one)) < 1e-15
    torch.testing.assert_close(compact.mean(), full.mean(), rtol=3e-11, atol=2e-16)
    assert float((full - compact).square().mean().sqrt()) > 1e-7
