"""Executable, costed numerical experiments for M02/04/05/12/20/21/23 and C3.

Every statement is a finite diagnostic on disclosed fields. In particular,
passed numerical identities are not elevated to mathematical proofs and a
feasible closure is not labeled an exact future moment law.
"""
from __future__ import annotations

import math
from dataclasses import asdict

import torch

from tdn.numerics import Equation, Geometry
from tdn.numerics.operators import rhs
from tdn.numerics.reference import choose_substeps, reference_step
from tdn.numerics.subflows import diffusion_step, reaction_step
from tdn.analysis.agenda.physics import moment_derivatives
from .numerics import (
    background_response, cubic_df_defect, dealiased_product, diffusion_first_step,
    dynamic_moment_step, embedded_rk4, finite_amplitude_step, fourier_resample,
    lowpass, match_mean_bounded, order_preserving_rk4, physical_diagnostics,
    quadratic_df_defect, signed_source, source_transport,
)


def _max(u):
    return float(u.detach().abs().max())


def _rms(u):
    return float(u.detach().double().square().mean().sqrt())


def _field(n=16, ndim=1, amplitude=.15, mean=.4):
    x = torch.arange(n, dtype=torch.float64) / n
    meshes = torch.meshgrid(*([x] * ndim), indexing="ij")
    v = torch.cos(2 * torch.pi * meshes[0]) + .3 * torch.cos(4 * torch.pi * meshes[0]) + .13 * torch.sin(6 * torch.pi * meshes[-1])
    if ndim == 2:
        v = v + .21 * torch.sin(2 * torch.pi * (meshes[0] + meshes[1]))
    v = v / v.abs().max()
    return (mean + amplitude * v).reshape(1, 1, *([n] * ndim))


def _teacher(ctx, u, h, eq, geo):
    n = max(16, choose_substeps(h, eq, geo))
    values = [reference_step(u, h, eq, geo, n * k, check=ctx.budget.check) for k in (1, 2, 4)]
    d0, d1 = _max(values[1] - values[0]), _max(values[2] - values[1])
    floor = 128 * torch.finfo(u.dtype).eps
    return values[-1], {"teacher_substeps": [n, 2 * n, 4 * n],
                        "teacher_uncertainty_max_estimate": max(d1, floor),
                        "teacher_refinement_converged": d1 <= max(d0 / 4, floor),
                        "teacher_uncertainty_is_certificate": False}


def _record(ctx, identifier, ids, metrics, checks, *, config=None, combinations=()):
    metrics.setdefault("parameters", metrics.get("parameter_count", 0))
    ctx.record(identifier, ids, combination_ids=combinations, metrics=metrics, checks=checks,
               config=config or {}, evidence={"kind": "finite_numerical_audit", "formal_proof": False,
                                               "device": "cpu", "dtype": "float64"})


def run(ctx):
    from .core import check

    profile = ctx.protocol.get("profile", "smoke")
    full = profile == "full"
    eq, geo = Equation(.02, 1.3), Geometry((16,), (1.,))
    u, h = _field(), .15
    v, c = u - u.mean(), float(u.mean())
    rows = 0
    ctx.budget.check()

    # M02: independent amplitude extraction and quadrature convergence.
    for ndim, background, kappa, horizon in ([(1, .4, .02, .15), (1, .1, .2, .025), (2, .7, .005, .05)] if full else [(1, .4, .02, .15)]):
        g = Geometry((16,) * ndim, (1.,) * ndim)
        e = Equation(kappa, 1.3)
        direction = (_field(16, ndim, 1., background) - background)
        epsilon = .025
        states = torch.cat([background + scale * direction for scale in (epsilon, -epsilon, epsilon / 2, -epsilon / 2)])
        truth, teacher = _teacher(ctx, states, horizon, e, g)
        split = diffusion_first_step(states, horizon, e, g)
        residual = truth - split
        coarse = (residual[0:1] + residual[1:2]) / (2 * epsilon**2)
        fine = (residual[2:3] + residual[3:4]) / (2 * (epsilon / 2)**2)
        oracle = (4 * fine - coarse) / 3
        answer, cost = ctx.measure(lambda: quadratic_df_defect(background + direction, horizon, e, g, nodes=12), repeats=1)
        node2 = quadratic_df_defect(background + direction, horizon, e, g, nodes=2)
        node6 = quadratic_df_defect(background + direction, horizon, e, g, nodes=6)
        midpoint = quadratic_df_defect(background + direction, horizon, e, g, nodes=1)
        uncertainty = 8 * teacher["teacher_uncertainty_max_estimate"] / (epsilon / 2)**2
        discrepancy = _max(answer - oracle)
        small_answer = split[2:3] + (epsilon / 2)**2 * answer
        checks = [check("independent_quadratic_coefficient", discrepancy, max(2e-7, 8 * uncertainty), category="math", applicable=teacher["teacher_refinement_converged"]),
                  check("coefficient_signal_resolved", _max(oracle), 10 * uncertainty, relation="ge", category="correctness", applicable=teacher["teacher_refinement_converged"]),
                  check("small_amplitude_defect_reduced", _max(small_answer - truth[2:3]), _max(split[2:3] - truth[2:3]), category="gap", applicable=teacher["teacher_refinement_converged"]),
                  check("teacher_refinement", teacher["teacher_refinement_converged"], True, relation="eq", category="correctness"),
                  check("midpoint_defect_is_zero", _max(midpoint), 1e-13, category="math"),
                  check("quadrature_refinement", _max(node6 - answer), max(_max(node2 - answer), 1e-13), category="math")]
        _record(ctx, f"m02/quadratic/{ndim}d/c{background}/k{kappa}", ["M02"],
                {**cost, **teacher, "coefficient_error_max": discrepancy, "coefficient_uncertainty_estimate": uncertainty,
                 "node2_error_max": _max(node2 - answer), "node6_error_max": _max(node6 - answer),
                 "quadrature_nodes": 12, "parameter_count": 0}, checks,
                config={"equation": asdict(e), "geometry": asdict(g), "h": horizon, "background": background,
                        "amplitude_levels": [epsilon, epsilon / 2], "target": "FD_nodal"})
        rows += 1
    nulls = []
    for name, state, time, equation in [("constant", torch.full_like(u, .4), h, eq), ("zero_diffusion", u, h, Equation(0, 1.3)),
                                         ("zero_reaction", u, h, Equation(.02, 0)), ("zero_time", u, 0., eq)]:
        q = quadratic_df_defect(state, time, equation, geo, nodes=4)
        third = cubic_df_defect(state, time, equation, geo, nodes=4)
        nulls.extend([check(f"quadratic_{name}", _max(q), 1e-13, category="correctness"),
                      check(f"cubic_{name}", _max(third), 1e-13, category="correctness")])
    _record(ctx, "m02-m23/physical-nullspace", ["M02", "M23"], {"parameter_count": 0}, nulls)
    rows += 1

    # M04: high-high -> retained low/zero modes, with identical output band and
    # no hidden full-source bypass in the compressed-input arm.
    n = 32
    g = Geometry((n,), (1.,))
    x = torch.arange(n, dtype=torch.float64) / n
    high = (.4 + .1 * torch.cos(10 * torch.pi * x) + .1 * torch.cos(12 * torch.pi * x)).reshape(1, 1, n)
    before, cost_before = ctx.measure(lambda: source_transport(high, .02, eq, g, cutoff=2, placement="before"), repeats=1)
    after, cost_after = ctx.measure(lambda: source_transport(high, .02, eq, g, cutoff=2, placement="after"), repeats=1)
    scaled = source_transport(.4 + 2 * (high - .4), .02, eq, g, cutoff=2, placement="before")
    negative = source_transport(high, .02, eq, g, cutoff=2, placement="before", weight=-1.)
    _record(ctx, "m04/pure-compression-high-high-low", ["M04"],
            {"before_source_rms": _rms(before), "after_source_rms": _rms(after), "before_cost": cost_before,
             "after_cost": cost_after, "quadratic_amplitude_ratio": _rms(scaled) / _rms(before),
             "bypass": "none_in_either_source_arm", "parameter_count": 0},
            [check("retained_high_high_response", _rms(before), 1e-4, relation="ge", category="gap"),
             check("compressed_input_loses_response", _rms(after), 1e-12, category="math"),
             check("quadratic_amplitude", _max(scaled - 4 * before), 1e-12, category="math"),
             check("signed_transport", _max(negative + before), 1e-13, category="correctness")],
            config={"input_modes": [5, 6], "output_cutoff": 2, "grid": [32], "identical_DF_base": True})
    rows += 1

    # M05: independent trigonometric product target and differentiable padding.
    for n in ([12, 16, 24] if full else [12]):
        x = torch.arange(n, dtype=torch.float64) / n
        mode = n // 2 - 1
        modefield = torch.cos(2 * torch.pi * mode * x).reshape(1, 1, n)
        padded, cost = ctx.measure(lambda: dealiased_product(modefield, modefield), repeats=1)
        expected = torch.full_like(modefield, .5)
        nodal = modefield.square()
        probe = modefield.clone().requires_grad_(True)
        value = dealiased_product(probe, probe).square().mean()
        gradient = torch.autograd.grad(value, probe)[0]
        eps = 1e-5
        direction = torch.sin(2 * torch.pi * x).reshape_as(probe) + .2
        fd = (dealiased_product(modefield + eps * direction, modefield + eps * direction).square().mean()
              - dealiased_product(modefield - eps * direction, modefield - eps * direction).square().mean()) / (2 * eps)
        _record(ctx, f"m05/quadratic-dealias/grid{n}", ["M05"],
                {**cost, "retained_product_error_max": _max(padded - expected), "nodal_alias_error_max": _max(nodal - expected),
                 "gradient_direction_error": abs(float((gradient * direction).sum() - fd)), "parameter_count": 0},
                [check("dealiased_retained_target", _max(padded - expected), 1e-12, category="math"),
                 check("alias_removed", _max(padded - expected), _max(nodal - expected) / 100, category="gap"),
                 check("differentiable_padding", abs(float((gradient * direction).sum() - fd)), 1e-7, category="correctness")],
                config={"grid": [n], "mode": mode, "padding": 3 * n // 2 + 1,
                        "target": "Fourier_Galerkin_quadratic_RHS_not_exact_logistic_subflow"})
        rows += 1

    # M12: rates agree with exact moment identities as dt tends to zero; the
    # binary z=1 boundary has inward diffusion flux rather than artificial lock.
    moments = moment_derivatives(u, eq, geo)
    tiny_h = 1e-7
    evolved = dynamic_moment_step(u, tiny_h, eq, geo, steps=1)
    mean_rate_error = _max((evolved["mean"] - moments["mean"]) / tiny_h - moments["mean_derivative"])
    var_rate_error = _max((evolved["variance"] - moments["variance"]) / tiny_h - moments["variance_derivative"])
    binary = torch.zeros_like(u)
    binary[..., :8] = 1
    binary_moments = dynamic_moment_step(binary, .05, eq, geo, steps=32)
    closure, cost = ctx.measure(lambda: dynamic_moment_step(u, h, eq, geo, steps=32), repeats=1)
    truth, teacher = _teacher(ctx, u, h, eq, geo)
    true_m = truth.mean()
    true_v = (truth - true_m).square().mean()
    _record(ctx, "m12/dynamic-feasible-closure", ["M12"],
            {**cost, **teacher, "initial_mean_rate_error": mean_rate_error, "initial_variance_rate_error": var_rate_error,
             "predicted_mean_error": _max(closure["mean"] - true_m), "predicted_variance_error": _max(closure["variance"] - true_v),
             "binary_final_normalized_variance": float(binary_moments["normalized_variance"]), "parameter_count": 0},
            [check("instantaneous_mean_identity", mean_rate_error, 2e-7, category="math"),
             check("instantaneous_variance_identity", var_rate_error, 2e-7, category="math"),
             check("binary_boundary_releases", float(binary_moments["normalized_variance"]), .999, category="gap"),
             check("feasible_variance", float(closure["normalized_variance"]), 1., category="correctness"),
             check("future_moment_law_exact", None, True, category="math", applicable=False,
                   reason="Finite closure is approximate; only instantaneous identities are exact")],
            config={"moment_steps": 32, "closure": closure["closure"]})
    rows += 1

    # Equal initial (mean,variance) does not identify diffusion energy or
    # future moments. The dynamic shape closure is required to distinguish it.
    x = torch.arange(16, dtype=torch.float64) / 16
    matched = torch.stack([.4 + .15 * torch.cos(2 * torch.pi * mode * x) for mode in (1, 4)]).unsqueeze(1)
    matched_exact = moment_derivatives(matched, eq, geo)
    matched_truth, teacher = _teacher(ctx, matched, h, eq, geo)
    matched_closure, cost = ctx.measure(lambda: dynamic_moment_step(matched, h, eq, geo, steps=32), repeats=1)
    actual_gap = abs(float(matched_truth[0].mean() - matched_truth[1].mean()))
    predicted_gap = abs(float(matched_closure["mean"][0] - matched_closure["mean"][1]))
    _record(ctx, "m12/equal-moments-different-spectra", ["M12"],
            {**cost, **teacher, "actual_future_mean_gap": actual_gap, "predicted_future_mean_gap": predicted_gap,
             "initial_means": matched_exact["mean"].flatten().tolist(),
             "initial_variances": matched_exact["variance"].flatten().tolist(),
             "initial_diffusion_energies": matched_exact["diffusion_energy"].flatten().tolist(), "parameters": 0},
            [check("initial_mean_matched", _max(matched_exact["mean"][0] - matched_exact["mean"][1]), 1e-14, category="correctness"),
             check("initial_variance_matched", _max(matched_exact["variance"][0] - matched_exact["variance"][1]), 1e-14, category="correctness"),
             check("future_moments_need_spectral_information", actual_gap, 100 * teacher["teacher_uncertainty_max_estimate"], relation="ge", category="math", applicable=teacher["teacher_refinement_converged"]),
             check("closure_retains_spectral_information", predicted_gap, actual_gap / 4, relation="ge", category="gap", applicable=teacher["teacher_refinement_converged"])],
            config={"grid": [16], "modes": [1, 4], "h": h, "initial_mean": .4, "amplitude": .15})
    rows += 1

    # M20: learned bounded residual keeps fixed-grid order four, and the
    # separately charged embedded error is measured against an independent flow.
    mild = Equation(.001, 1.3)
    horizon = .4
    oracle, teacher = _teacher(ctx, u, horizon, mild, geo)
    residual = lambda state, equation, geometry: 3 * signed_source(state, equation, geometry)
    errors = []
    work = []
    for count in (4, 8, 16):
        def solve():
            answer = u
            for _ in range(count):
                answer = order_preserving_rk4(answer, horizon / count, mild, geo, residual=residual)
            return answer
        answer, cost = ctx.measure(solve, repeats=1)
        errors.append(_rms(answer - oracle))
        work.append(cost)
    orders = [math.log2(errors[i] / errors[i + 1]) if errors[i + 1] > 0 else None for i in range(2)]
    embedded, cost = ctx.measure(lambda: embedded_rk4(u, .05, mild, geo, residual=residual), repeats=1)
    fine, estimate = embedded
    embedded_truth, embedded_teacher = _teacher(ctx, u, .05, mild, geo)
    _record(ctx, "m20/order-constrained-residual-and-embedded", ["M20"],
            {"errors_rms": errors, "observed_orders": orders, "costs": work, "embedded_cost": cost,
             "embedded_error_estimate_rms": _rms(estimate), "embedded_actual_error_rms": _rms(fine - embedded_truth),
             "parameter_count": 0, "residual_here": "fixed_coefficient_neural_training_arm_is_separate", **teacher},
            [check("fixed_grid_global_order", min(x for x in orders if x is not None), 3.7, relation="ge", category="math", applicable=teacher["teacher_refinement_converged"]),
             check("embedded_asymptotic_effectivity", _rms(fine - embedded_truth), max(3 * _rms(estimate), embedded_teacher["teacher_uncertainty_max_estimate"]), category="gap", applicable=embedded_teacher["teacher_refinement_converged"]),
             check("unconditional_stiff_stability", None, True, applicable=False, category="math",
                   reason="Explicit RK4 has a bounded stability region; no A-stability claim")],
            config={"counts": [4, 8, 16], "horizon": horizon, "residual_order": 5, "embedded_field_evaluations": 12})
    rows += 1

    # M21: controls and actual corrected maps both receive the same tests.
    other = torch.roll(u, 3, -1) * .9 + .03
    for name, stepper in [("DF", diffusion_first_step),
                          ("quadratic", lambda state, time, equation, geometry: finite_amplitude_step(state, time, equation, geometry, nodes=8)),
                          ("cubic", lambda state, time, equation, geometry: finite_amplitude_step(state, time, equation, geometry, include_cubic=True, nodes=6))]:
        diagnostics, cost = ctx.measure(lambda: physical_diagnostics(stepper, u, other, h, eq, geo), repeats=1)
        _record(ctx, f"m21/physical-map/{name}", ["M21"], {**diagnostics, **cost, "parameter_count": 0},
                [check("monotonicity", diagnostics["monotonicity_violation"], 2e-12, category="math"),
                 check("concavity", diagnostics["concavity_violation"], 2e-12, category="math"),
                 check("physical_growth", diagnostics["growth_ratio"], diagnostics["growth_envelope"] + 1e-12, category="math"),
                 check("invariant_interval", diagnostics["bound_violation"], 1e-12, category="math"),
                 check("global_certificate", None, True, applicable=False, category="math", reason="Sampled inequalities are not global bounds")],
                config={"model": name, "h": h, "equation": asdict(eq)})
        rows += 1

    if full:
        for rate, kappa, horizon in [(1.3, .002, .5), (1.3, .2, .2), (8., .02, .5)]:
            stressed_eq = Equation(kappa, rate)
            stressed = _field(amplitude=.28, mean=.4)
            for name, stepper in [("DF", diffusion_first_step),
                                  ("quadratic_cubic", lambda state, time, equation, geometry: finite_amplitude_step(state, time, equation, geometry, include_cubic=True, nodes=8))]:
                diag, cost = ctx.measure(lambda: physical_diagnostics(stepper, stressed, torch.roll(stressed, 5, -1), horizon, stressed_eq, geo), repeats=1)
                _record(ctx, f"m21/stiffness/{rate}-{kappa}-{horizon}/{name}", ["M21"], {**diag, **cost, "parameters": 0},
                        [check("monotonicity", diag["monotonicity_violation"], 2e-12, category="math"),
                         check("concavity", diag["concavity_violation"], 2e-12, category="math"),
                         check("physical_growth", diag["growth_ratio"], diag["growth_envelope"] + 1e-12, category="math"),
                         check("interval_at_stiff_step", diag["bound_violation"], 1e-12, category="gap")],
                        config={"h": horizon, "kappa": kappa, "reaction_rate": rate, "model": name})
                rows += 1

    # M23: independent odd-amplitude extraction isolates cubic defect rather
    # than merely measuring the full nonlinear endpoint.
    direction = (u - c) / .15
    epsilon = .04
    states = torch.cat([c + scale * direction for scale in (epsilon, -epsilon, epsilon / 2, -epsilon / 2)])
    truth_batch, teacher = _teacher(ctx, states, h, eq, geo)
    residual_batch = truth_batch - diffusion_first_step(states, h, eq, geo)
    coarse = (residual_batch[0:1] - residual_batch[1:2]) / (2 * epsilon**3)
    fine = (residual_batch[2:3] - residual_batch[3:4]) / (2 * (epsilon / 2)**3)
    oracle = (4 * fine - coarse) / 3
    coefficient, cost = ctx.measure(lambda: cubic_df_defect(c + direction, h, eq, geo, nodes=10), repeats=1)
    uncertainty = 8 * teacher["teacher_uncertainty_max_estimate"] / (epsilon / 2)**3
    lower = cubic_df_defect(c + direction, h, eq, geo, nodes=5)
    _record(ctx, "m23/independent-cubic-coefficient", ["M23"],
            {**cost, **teacher, "coefficient_error_max": _max(coefficient - oracle),
             "coefficient_uncertainty_estimate": uncertainty, "quadrature_difference_max": _max(coefficient - lower), "parameter_count": 0},
            [check("reference_refinement", teacher["teacher_refinement_converged"], True, relation="eq", category="correctness"),
             check("independent_cubic_coefficient", _max(coefficient - oracle), max(3e-7, 8 * uncertainty), category="math", applicable=teacher["teacher_refinement_converged"]),
             check("cubic_signal_resolved", _max(oracle), 10 * uncertainty, relation="ge", category="correctness", applicable=teacher["teacher_refinement_converged"]),
             check("causal_quadrature_converged", _max(coefficient - lower), 1e-7, category="correctness")],
            config={"h": h, "amplitudes": [epsilon, epsilon / 2], "nodes": 10, "causal_domain": "0<=tau<=s<=h"})
    rows += 1

    # Actual C3 singles and combinations, including the zero-mean branch and
    # single endpoint mean replacement that prevents zero-mode double-counting.
    amplitudes = [.04, .12, .25] if full else [.06, .18]
    for amp in amplitudes:
        state = _field(amplitude=amp)
        target, teacher = _teacher(ctx, state, h, eq, geo)
        base = diffusion_first_step(state, h, eq, geo)
        axes = (2,)
        target_mean = target.mean(axes, keepdim=True)
        base_error = _rms(base - target)
        base_mean_error = _max(base.mean(axes, keepdim=True) - target_mean)
        base_spatial_error = _rms((base - base.mean(axes, keepdim=True)) - (target - target_mean))
        arms = {
            "DF": (False, False, False), "quadratic": (False, True, False),
            "cubic_only": (False, False, True), "quadratic_cubic": (False, True, True),
            "mean_only": (True, False, False), "mean_quadratic": (True, True, False),
            "mean_quadratic_cubic": (True, True, True),
        }
        for name, (mean, quad, cubic) in arms.items():
            answer, cost = ctx.measure(lambda: finite_amplitude_step(state, h, eq, geo,
                include_mean=mean, include_quadratic=quad, include_cubic=cubic, nodes=8, moment_steps=32), repeats=1)
            predicted_mean = answer.mean(axes, keepdim=True)
            rms_error = _rms(answer - target)
            mean_error = _max(predicted_mean - target_mean)
            spatial_error = _rms((answer - predicted_mean) - (target - target_mean))
            checks = [check("reference_refinement", teacher["teacher_refinement_converged"], True, relation="eq", category="correctness"),
                      check("rms_not_worse_than_DF", rms_error, base_error + teacher["teacher_uncertainty_max_estimate"], category="gap", applicable=teacher["teacher_refinement_converged"]),
                      check("mean_not_worse_than_DF", mean_error, base_mean_error + teacher["teacher_uncertainty_max_estimate"], category="gap", applicable=teacher["teacher_refinement_converged"]),
                      check("spatial_not_worse_than_DF", spatial_error, base_spatial_error + teacher["teacher_uncertainty_max_estimate"], category="gap", applicable=teacher["teacher_refinement_converged"]),
                      check("bounded_output", max(float((-answer).clamp_min(0).max()), float((answer - 1).clamp_min(0).max())), 1e-12, category="math")]
            if mean:
                wanted = dynamic_moment_step(state, h, eq, geo, steps=32)["mean"]
                checks.append(check("zero_mode_replaced_once", _max(predicted_mean - wanted), 2e-14, category="correctness"))
            ids = ["M04"] + (["M12"] if mean else []) + (["M02"] if quad else []) + (["M23"] if cubic else [])
            _record(ctx, f"c3/amplitude{amp}/{name}", ids, {**cost, **teacher,
                    "rms_error": rms_error, "max_error": _max(answer - target), "mean_error": mean_error,
                    "centered_spatial_rms_error": spatial_error, "base_rms_error": base_error,
                    "base_mean_error": base_mean_error, "base_centered_spatial_rms_error": base_spatial_error,
                    "parameter_count": 0}, checks,
                    config={"arm": name, "amplitude": amp, "h": h, "nodes": 8, "mean_steps": 32,
                            "include_mean": mean, "include_quadratic": quad, "include_cubic": cubic}, combinations=["C3"])
            rows += 1
        for placement in ("source_before_compression", "input_before_source"):
            def projected_combination():
                source_state = state if placement == "source_before_compression" else lowpass(state, 2)
                q = quadratic_df_defect(source_state, h, eq, geo, nodes=8)
                d = cubic_df_defect(source_state, h, eq, geo, nodes=8)
                response = lowpass(q + d, 2)
                mean_target = dynamic_moment_step(state, h, eq, geo, steps=32)["mean"]
                return match_mean_bounded(base + response, mean_target)[0]
            projected, cost = ctx.measure(projected_combination, repeats=1)
            wanted = dynamic_moment_step(state, h, eq, geo, steps=32)["mean"]
            _record(ctx, f"c3/amplitude{amp}/{placement}", ["M04", "M12", "M23"],
                    {**cost, **teacher, "rms_error": _rms(projected - target),
                     "max_error": _max(projected - target), "base_rms_error": base_error,
                     "mean_error": _max(projected.mean(axes, keepdim=True) - target_mean),
                     "parameter_count": 0},
                    [check("reference_refinement", teacher["teacher_refinement_converged"], True, relation="eq", category="correctness"),
                     check("rms_not_worse_than_DF", _rms(projected - target), base_error + teacher["teacher_uncertainty_max_estimate"], category="gap", applicable=teacher["teacher_refinement_converged"]),
                     check("zero_mode_replaced_once", _max(projected.mean(axes, keepdim=True) - wanted), 2e-14, category="correctness")],
                    config={"arm": placement, "amplitude": amp, "h": h,
                            "output_cutoff": 2, "full_source_bypass": False, "identical_DF_base": True}, combinations=["C3"])
            rows += 1
        def spatial_solve():
            q = quadratic_df_defect(state, h, eq, geo, nodes=8)
            d = cubic_df_defect(state, h, eq, geo, nodes=8)
            return base + q + d - (q + d).mean(axes, keepdim=True)
        spatial_only, spatial_cost = ctx.measure(spatial_solve, repeats=1)
        _record(ctx, f"c3/amplitude{amp}/zero_mean_spatial", ["M04", "M23"],
                {**spatial_cost, **teacher, "rms_error": _rms(spatial_only - target), "max_error": _max(spatial_only - target), "parameter_count": 0},
                [check("reference_refinement", teacher["teacher_refinement_converged"], True, relation="eq", category="correctness"),
                 check("no_spatial_zero_mode", _max(spatial_only.mean(axes, keepdim=True) - base.mean(axes, keepdim=True)), 1e-13, category="correctness"),
                 check("rms_not_worse_than_DF", _rms(spatial_only - target), base_error + teacher["teacher_uncertainty_max_estimate"], category="gap", applicable=teacher["teacher_refinement_converged"])],
                config={"arm": "zero_mean_spatial", "amplitude": amp, "h": h}, combinations=["C3"])
        rows += 1
    return {"numerical_audit_rows": rows, "mechanisms": ["M02", "M04", "M05", "M12", "M20", "M21", "M23"],
            "combinations": ["C3"], "formal_proof": False, "training": "none_in_this_panel_neural_arms_are_separate"}
