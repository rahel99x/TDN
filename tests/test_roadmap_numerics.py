"""Independent coefficient/target/order checks for roadmap numerical upgrades."""
import math

import pytest
import torch

from tdn.numerics import Equation, Geometry
from tdn.numerics.reference import reference_step
from tdn.numerics.subflows import diffusion_step, reaction_step
from tdn.analysis.agenda.physics import fourier_resample as scipy_resample, moment_derivatives
from tdn.analysis.roadmap.numerics import (
    background_response, cubic_df_defect, dealiased_product, diffusion_first_step,
    dynamic_moment_step, embedded_rk4, finite_amplitude_step, fourier_resample,
    match_mean_bounded, order_preserving_rk4, physical_diagnostics,
    quadratic_df_defect, signed_source, source_transport,
)


def field(n=16, amplitude=.1):
    x = torch.arange(n, dtype=torch.float64) / n
    v = torch.cos(2 * torch.pi * x) + .3 * torch.cos(4 * torch.pi * x) + .13 * torch.sin(6 * torch.pi * x)
    return (.4 + amplitude * v).reshape(1, 1, n)


@pytest.mark.parametrize("kind", ["constant", "kappa", "reaction", "time"])
def test_physical_coefficients_have_exact_nulls(kind):
    u = torch.full_like(field(), .4) if kind == "constant" else field()
    eq = Equation(0 if kind == "kappa" else .02, 0 if kind == "reaction" else 1.3)
    g = Geometry((16,), (1.,))
    for function in (quadratic_df_defect, cubic_df_defect):
        result = function(u, 0 if kind == "time" else .15, eq, g)
        assert result.abs().max() < 1e-15


def test_quadratic_midpoint_is_zero_but_two_nodes_retain_response():
    u, eq, g = field(), Equation(.02, 1.3), Geometry((16,), (1.,))
    assert quadratic_df_defect(u, .15, eq, g, nodes=1).abs().max() == 0
    assert quadratic_df_defect(u, .15, eq, g, nodes=2).abs().max() > 1e-6


def test_coefficient_amplitude_hierarchy_matches_independent_coupled_teacher():
    eq, g = Equation(.02, 1.3), Geometry((16,), (1.,))
    errors_q, errors_c = [], []
    for amp in (.1, .05, .025):
        u = field(amplitude=amp)
        teacher = reference_step(u, .15, eq, g, 256)
        base = diffusion_first_step(u, .15, eq, g)
        q = quadratic_df_defect(u, .15, eq, g, nodes=10)
        cubic = cubic_df_defect(u, .15, eq, g, nodes=10)
        errors_q.append(float((teacher - base - q).abs().max()))
        errors_c.append(float((teacher - base - q - cubic).abs().max()))
    # O(epsilon^3) becomes O(epsilon^4); quadrature and teacher independently converge.
    assert min(math.log2(errors_q[i] / errors_q[i + 1]) for i in range(2)) > 2.9
    assert min(math.log2(errors_c[i] / errors_c[i + 1]) for i in range(2)) > 3.8
    assert errors_c[0] < errors_q[0] / 10


def test_quadratic_batched_times_backgrounds_and_learned_nodes_are_differentiable():
    u = torch.cat([field(), field() + .2]).requires_grad_(True)
    h = torch.tensor([.07, .12], dtype=torch.float64, requires_grad=True)
    nodes = torch.tensor([.2, .8], dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([.4, -.2], dtype=torch.float64, requires_grad=True)
    answer = quadratic_df_defect(u, h, Equation(.02, 1.3), Geometry((16,), (1.,)),
                                 node_positions=nodes, node_weights=weights)
    grads = torch.autograd.grad(answer.square().sum(), (u, h, nodes, weights))
    assert all(torch.isfinite(x).all() and x.abs().max() > 0 for x in grads)
    for i in range(2):
        single = quadratic_df_defect(u[i:i+1], h[i], Equation(.02, 1.3), Geometry((16,), (1.,)),
                                     node_positions=nodes, node_weights=weights)
        torch.testing.assert_close(single, answer[i:i+1])


@pytest.mark.parametrize("grid,target", [((8,), (13,)), ((9,), (6,)), ((8, 10), (13, 17)), ((9, 11), (6, 8))])
def test_differentiable_resampling_matches_independent_scipy_nyquist(grid, target):
    generator = torch.Generator().manual_seed(4102)
    u = torch.randn((2, 1, *grid), generator=generator, dtype=torch.float64)
    torch.testing.assert_close(fourier_resample(u, target), scipy_resample(u, target), rtol=1e-12, atol=1e-12)


def test_dealias_product_does_not_fold_high_square_and_supports_autograd():
    x = torch.arange(12, dtype=torch.float64) / 12
    u = torch.cos(10 * torch.pi * x).reshape(1, 1, 12)
    torch.testing.assert_close(dealiased_product(u, u), torch.full_like(u, .5), rtol=1e-12, atol=1e-12)
    assert (u.square() - .5).abs().max() > .4
    a = (u * .3).requires_grad_()
    assert torch.autograd.gradcheck(lambda z: dealiased_product(z, z), (a,), fast_mode=True)


def test_pure_precompression_preserves_high_high_low_without_bypass():
    x = torch.arange(32, dtype=torch.float64) / 32
    u = (.4 + .1 * torch.cos(10 * torch.pi * x) + .1 * torch.cos(12 * torch.pi * x)).reshape(1, 1, 32)
    eq, g = Equation(.02, 1.3), Geometry((32,), (1.,))
    a = source_transport(u, .02, eq, g, cutoff=2, placement="before")
    b = source_transport(u, .02, eq, g, cutoff=2, placement="after")
    assert a.abs().max() > .001 and b.abs().max() < 1e-20
    scaled = source_transport(.4 + 2 * (u - .4), .02, eq, g, cutoff=2, placement="before")
    torch.testing.assert_close(scaled, 4 * a, rtol=1e-12, atol=1e-12)


def test_moment_initial_derivatives_and_binary_inward_flux():
    u, eq, g = field(), Equation(.02, 1.3), Geometry((16,), (1.,))
    exact = moment_derivatives(u, eq, g)
    dt = 1e-7
    result = dynamic_moment_step(u, dt, eq, g, steps=1)
    torch.testing.assert_close((result["mean"] - exact["mean"]) / dt, exact["mean_derivative"], rtol=1e-5, atol=1e-7)
    torch.testing.assert_close((result["variance"] - exact["variance"]) / dt, exact["variance_derivative"], rtol=1e-5, atol=1e-7)
    binary = torch.zeros_like(u)
    binary[..., :8] = 1
    result = dynamic_moment_step(binary, .05, eq, g, steps=32)
    assert 0 <= result["normalized_variance"] < .999
    assert 0 <= result["variance"] <= result["mean"] * (1 - result["mean"])


def test_mean_replacement_removes_cubic_zero_mode_exactly_once():
    u, eq, g = field(amplitude=.2), Equation(.02, 1.3), Geometry((16,), (1.,))
    answer = finite_amplitude_step(u, .15, eq, g, include_mean=True, include_cubic=True, nodes=6)
    wanted = dynamic_moment_step(u, .15, eq, g)["mean"]
    torch.testing.assert_close(answer.mean((2,), keepdim=True), wanted, rtol=0, atol=1e-14)
    assert answer.min() >= 0 and answer.max() <= 1
    wild = torch.tensor([[[-10., 0., 3., 7.]]], dtype=torch.float64)
    bounded, scale = match_mean_bounded(wild, torch.tensor([[[.3]]], dtype=torch.float64))
    assert bounded.min() >= -1e-15 and bounded.max() <= 1
    assert abs(float(bounded.mean()) - .3) < 1e-15 and scale < 1


def test_h5_residual_retains_rk4_order_and_embedded_estimator():
    u, eq, g = field(), Equation(.001, 1.3), Geometry((16,), (1.,))
    truth = reference_step(u, .4, eq, g, 512)
    residual = lambda state, equation, geometry: 3 * signed_source(state, equation, geometry)
    errors = []
    for count in (4, 8, 16):
        answer = u
        for _ in range(count):
            answer = order_preserving_rk4(answer, .4 / count, eq, g, residual=residual)
        errors.append(float((answer - truth).square().mean().sqrt()))
    assert min(math.log2(errors[i] / errors[i + 1]) for i in range(2)) > 3.7
    fine, estimate = embedded_rk4(u, .05, eq, g, residual=residual)
    target = reference_step(u, .05, eq, g, 64)
    ratio = float((fine - target).norm() / estimate.norm())
    assert .5 < ratio < 2


def test_exact_df_monotonicity_concavity_and_expansive_growth_are_separate():
    u, eq, g = field(), Equation(.02, 1.3), Geometry((16,), (1.,))
    result = physical_diagnostics(diffusion_first_step, u, torch.roll(u, 3, -1), .15, eq, g)
    assert result["monotonicity_violation"] < 1e-13
    assert result["concavity_violation"] < 1e-13
    assert result["growth_ratio"] <= result["growth_envelope"]
    nearzero = torch.full_like(u, .001)
    expanded = physical_diagnostics(diffusion_first_step, nearzero, nearzero * 2, .15, eq, g)
    assert expanded["growth_ratio"] > 1


@pytest.mark.parametrize("h", [-.1, float("nan"), float("inf")])
def test_invalid_response_horizons_are_rejected(h):
    with pytest.raises(ValueError):
        quadratic_df_defect(field(), h, Equation(.02, 1.3), Geometry((16,), (1.,)))

@pytest.mark.parametrize("eq", [Equation(0, 1.3), Equation(.02, 0)])
def test_combined_mean_and_cubic_preserve_commuting_subflows(eq):
    u, g = field(amplitude=.2), Geometry((16,), (1.,))
    answer = finite_amplitude_step(u, .2, eq, g, include_mean=True, include_cubic=True)
    expected = diffusion_first_step(u, .2, eq, g)
    torch.testing.assert_close(answer, expected, rtol=0, atol=2e-15)

@pytest.mark.parametrize("constant", [0., 1.])
def test_constant_endpoints_do_not_form_infinity_times_zero_at_long_h(constant):
    u = torch.full((1, 1, 8), constant, dtype=torch.float32)
    for function in (quadratic_df_defect, cubic_df_defect):
        value = function(u, 100., Equation(.1, 1.), Geometry((8,), (1.,)), nodes=2)
        assert torch.isfinite(value).all() and value.abs().max() == 0


def test_physical_diagnostics_detect_bounded_but_nonmonotone_map():
    u, eq, g = field(), Equation(.02, 1.3), Geometry((16,), (1.,))
    bad_map = lambda state, h, equation, geometry: 1 - state
    result = physical_diagnostics(bad_map, u, torch.roll(u, 3, -1), .15, eq, g)
    assert result["bound_violation"] == 0
    assert result["monotonicity_violation"] > .05


def test_equal_moments_do_not_determine_future_moments():
    x = torch.arange(16, dtype=torch.float64) / 16
    u = torch.stack([.4 + .15 * torch.cos(2 * torch.pi * mode * x) for mode in (1, 4)]).unsqueeze(1)
    eq, g = Equation(.02, 1.3), Geometry((16,), (1.,))
    initial = moment_derivatives(u, eq, g)
    torch.testing.assert_close(initial["mean"][0], initial["mean"][1], rtol=0, atol=1e-14)
    torch.testing.assert_close(initial["variance"][0], initial["variance"][1], rtol=0, atol=1e-14)
    assert abs(float(initial["diffusion_energy"][0] - initial["diffusion_energy"][1])) > .01
    predicted = dynamic_moment_step(u, .15, eq, g, steps=32)
    truth = reference_step(u, .15, eq, g, 128)
    assert abs(float(predicted["mean"][0] - predicted["mean"][1])) > 1e-4
    assert abs(float(truth[0].mean() - truth[1].mean())) > 1e-4


def test_unresolved_teacher_cannot_credit_c3_gaps_or_coefficient_math(monkeypatch, tmp_path):
    from tdn.analysis.roadmap import numerical_audit
    from tdn.analysis.roadmap.core import Context
    from tdn.analysis.roadmap.protocol import build_protocol
    class Budget:
        def check(self):
            pass
    original = numerical_audit._teacher
    def unresolved(*args, **kwargs):
        value, metadata = original(*args, **kwargs)
        return value, {**metadata, "teacher_refinement_converged": False}
    monkeypatch.setattr(numerical_audit, "_teacher", unresolved)
    ctx = Context(build_protocol("smoke"), "audit", tmp_path, {}, "cpu", Budget())
    numerical_audit.run(ctx)
    c3 = [row for row in ctx.rows if "C3" in row["combination_ids"]]
    assert len(c3) == 20
    for row in c3:
        assert any(c["check_id"] == "reference_refinement" and c["verdict"] == "BAD" for c in row["checks"])
        assert all(c["verdict"] == "NA" for c in row["checks"] if c["category"] == "gap")
    coefficients = [c for row in ctx.rows for c in row["checks"]
                    if c["check_id"] in {"independent_quadratic_coefficient", "independent_cubic_coefficient"}]
    assert len(coefficients) == 2 and all(c["verdict"] == "NA" for c in coefficients)
