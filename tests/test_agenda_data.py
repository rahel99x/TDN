"""Independent evidence for paired fields and the separate continuum target."""
import math

import pytest
import torch

from tdn.analysis.agenda.data import continuous_field, continuum_ifrk4, _horizons
from tdn.analysis.agenda.protocol import build_protocol
from tdn.numerics import Equation


def test_same_continuous_parent_exactly_agrees_at_shared_grid_nodes():
    parent = next(p for p in build_protocol("full")["parents"] if p["split"] == "confirmation" and p["spectrum"] == "rough")
    a, b, c = [continuous_field(parent, n) for n in (32, 64, 128)]
    torch.testing.assert_close(a, b[..., ::2, ::2], rtol=0, atol=0)
    torch.testing.assert_close(a, c[..., ::4, ::4], rtol=0, atol=0)
    assert float(a.mean()) == pytest.approx(parent["mean"], abs=2e-15)
    assert float((a - a.mean()).square().mean()) == pytest.approx(parent["variance"], abs=2e-15)


def test_diffusion_reaction_crossing_changes_physics_without_changing_field():
    parents = [p for p in build_protocol("full")["parents"] if p["split"] == "confirmation" and p["controlled_field_index"] == 2]
    assert len({(p["kappa"], p["reaction_rate"]) for p in parents}) == 4
    fields = [continuous_field(p, 32) for p in parents]
    assert all(torch.equal(fields[0], field) for field in fields[1:])


def test_phase_and_spectrum_controls_keep_same_mean_and_variance():
    parents = [p for p in build_protocol("full")["parents"] if p["split"] == "confirmation" and p["physics_index"] == 0 and p["controlled_field_index"] in (0, 1, 2, 3, 5)]
    fields = [continuous_field(p, 32) for p in parents]
    for state in fields:
        assert float(state.mean()) == pytest.approx(.5, abs=2e-15)
        assert float((state - state.mean()).square().mean()) == pytest.approx(.004, abs=2e-15)
    assert not torch.equal(fields[1], fields[-1])


def test_lawson_teacher_exactly_resolves_linear_spectral_diffusion():
    n, h, kappa, mode = 32, .37, .007, 3
    x = torch.arange(n, dtype=torch.float64) / n
    wave = torch.cos(2 * math.pi * mode * x)[:, None].expand(n, n)[None, None]
    u = .5 + .1 * wave
    answer = continuum_ifrk4(u, h, Equation(kappa, 0.), 7)
    exact = .5 + .1 * math.exp(-4 * math.pi**2 * mode**2 * kappa * h) * wave
    torch.testing.assert_close(answer, exact, rtol=1e-12, atol=2e-14)


def test_lawson_teacher_refines_toward_independent_homogeneous_logistic_flow():
    c, rate, h = .35, 3., .8
    exact = c / (c + (1 - c) * math.exp(-rate * h))
    u = torch.full((1, 1, 8, 8), c, dtype=torch.float64)
    errors = [float((continuum_ifrk4(u, h, Equation(.03, rate), n) - exact).abs().max()) for n in (4, 8, 16)]
    assert errors[0] > errors[1] > errors[2]
    assert 8 < errors[1] / errors[2] < 30


def test_floating_step_sums_do_not_create_fake_distinct_teacher_horizons():
    assert _horizons(build_protocol("smoke"), "confirmation") == [.27, .81]
