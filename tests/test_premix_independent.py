"""Premixing checks against a dense physical-space oracle, without FFT solvers.

The oracle assembles the periodic finite-difference generator directly and
uses SciPy matrix exponentials and independently generated GL nodes. It never
imports the implementation's spectrum, quadrature, reaction or projection
helpers. This validates the complete correction, not only pair convolution.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
import torch
from scipy.linalg import expm
from scipy.special import roots_legendre

from tdn.numerics.types import Equation, Geometry
from tdn.research.premix import prepare_premix_step


def dense_generator(grid, lengths, kappa):
    total = math.prod(grid)
    matrix = np.zeros((total, total))
    for position in itertools.product(*(range(n) for n in grid)):
        row = np.ravel_multi_index(position, grid)
        for axis, (n, length) in enumerate(zip(grid, lengths)):
            coefficient = kappa / (length / n) ** 2
            matrix[row, row] -= 2 * coefficient
            for direction in (-1, 1):
                neighbor = list(position)
                neighbor[axis] = (neighbor[axis] + direction) % n
                matrix[row, np.ravel_multi_index(tuple(neighbor), grid)] += coefficient
    return matrix


def dense_projection(grid, cutoff):
    """Explicit Fourier synthesis matrix; no numpy/torch FFT or solver helper."""
    positions = np.array(list(itertools.product(*(range(n) for n in grid))))
    frequencies = list(itertools.product(*(range(n) for n in grid)))
    retained = [frequency for frequency in frequencies
                if all(min(k, n - k) <= cutoff for k, n in zip(frequency, grid))]
    basis = np.exp(2j * np.pi * (positions / np.array(grid)) @ np.array(retained).T)
    return (basis @ basis.conj().T).real / math.prod(grid)


def dense_defect(state, h, equation, geometry, cutoff):
    generator = dense_generator(geometry.grid, geometry.lengths, equation.kappa)
    v = state.reshape(state.shape[0], -1).numpy()
    mean = v.mean(axis=1, keepdims=True)
    v = v - mean
    rate = equation.reaction_rate
    def jacobian(time):
        q = np.exp(-rate * time)
        return q / (mean + (1 - mean) * q) ** 2
    def variation(time):
        evolution = expm(time * generator)
        return (v @ evolution.T) ** 2 - v ** 2 @ evolution.T
    result = np.zeros_like(v)
    locations, weights = roots_legendre(3)
    for location, weight in zip(locations, weights):
        time = h * (1 + location) / 2
        result += h * weight / 2 * jacobian(time) * (variation(time) @ expm((h - time) * generator).T)
    qhalf, qfull = np.exp(-rate * h / 2), np.exp(-rate * h)
    whalf = (qhalf - qfull) / (rate * (mean + (1 - mean) * qhalf) * (mean + (1 - mean) * qfull))
    result -= whalf * variation(h)
    result *= -rate * jacobian(h)
    return torch.from_numpy(result @ dense_projection(geometry.grid, cutoff).T).reshape_as(state)


@pytest.mark.parametrize("grid,lengths", [((9,), (1.3,)), ((5, 6), (.8, 1.4))])
@pytest.mark.parametrize("h", [.003, .075, .4])
@pytest.mark.parametrize("variant", ["output_gl3", "selected_output_gl3"])
def test_premix_matches_independent_dense_physical_quadrature(grid, lengths, h, variant):
    generator = torch.Generator().manual_seed(1797)
    state = .23 + .46 * torch.rand(3, 1, *grid, generator=generator, dtype=torch.float64)
    equation, geometry = Equation(.019, 3.7), Geometry(grid, lengths)
    expected = dense_defect(state, h, equation, geometry, 2)
    actual = prepare_premix_step(state, h, equation, geometry, variant, modes=2, chunk_size=3).defect(state)
    torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-17)


@pytest.mark.parametrize("variant", ["output_gl3", "selected_output_gl3"])
def test_nonzero_phase_and_mixed_parent_means_survive_dense_projection(variant):
    grid = (32,)
    x = torch.arange(32, dtype=torch.float64) * (2 * torch.pi / 32)
    # The output mode 1 phase depends on both independent input phases;
    # replacing products by powers, magnitudes, or a global batch mean fails.
    first = .31 + .04 * torch.cos(9 * x + .37) + .03 * torch.sin(10 * x - .81)
    second = .72 + .02 * torch.cos(9 * x - .19) + .05 * torch.sin(10 * x + .28)
    state = torch.stack((first, second))[:, None]
    equation, geometry = Equation(.004, 4.), Geometry(grid, (1.2,))
    expected = dense_defect(state, .13, equation, geometry, 2)
    actual = prepare_premix_step(state, .13, equation, geometry, variant, modes=2).defect(state)
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=5e-18)
    assert float((actual - actual.mean(dim=-1, keepdim=True)).abs().max()) > 1e-6
