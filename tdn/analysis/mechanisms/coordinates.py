"""Training-free coordinate and spatial-information audits.

The six controlled maps below are probes, not deployable learned models. All
use the same prescribed two coefficients and saturating time basis. Observed
errors are evidence about those probes, never optimized model performance.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Callable

import torch
from torch import Tensor

from tdn.numerics.operators import broadcast_h, check_shape
from tdn.numerics.reference import choose_substeps, refined_reference
from tdn.numerics.splitting import split_step
from tdn.numerics.subflows import diffusion_step, reaction_step
from tdn.numerics.types import Equation, Geometry
from tdn.research.common import commuting_gate
from tdn.research.reaction import capacity_update, reaction_clock_step


VARIANTS = (
    "clock_local", "post_capacity_local", "clock_global",
    "post_capacity_global", "pre_capacity_global", "additive_global",
)


def global_commuting_gate(u: Tensor, equation: Equation, geometry: Geometry) -> Tensor:
    """Smooth global mean edge variation; zero for every uniform input.

    This deliberately simple information probe retains the existing physical
    prefactor. It is not asserted to be resolution independent or optimal.
    Subtracting neighbours, rather than a rounded mean, makes the uniform
    zero exact. No min/max, clipping or data-dependent hard threshold is used.
    """
    check_shape(u, geometry)
    if equation.kappa == 0 or equation.reaction_rate == 0:
        return u * 0
    variation = torch.zeros_like(u)
    for axis in range(2, u.ndim):
        variation = variation + (torch.roll(u, 1, axis) - u).square()
    variation = variation.mean(dim=tuple(range(2, u.ndim)), keepdim=True)
    factor = equation.kappa * equation.reaction_rate * sum(dx**-2 for dx in geometry.dx)
    scaled = factor * variation
    return (scaled / (1 + scaled)).expand_as(u)


def controlled_step(u: Tensor, h: float | Tensor, equation: Equation,
                    geometry: Geometry, variant: str,
                    coefficients: tuple[float | Tensor, float | Tensor] = (.7, -.2)) -> Tensor:
    """Panel A forward probe with fixed h-independent coefficient fields."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown coordinate probe: {variant}")
    check_shape(u, geometry)
    step = broadcast_h(h, u)
    middle = diffusion_step(reaction_step(u, step / 2, equation), step, equation, geometry)
    gate = (commuting_gate(u, equation, geometry) if variant.endswith("_local")
            else global_commuting_gate(u, equation, geometry))
    z = -torch.expm1(-equation.reaction_rate * step)
    increment = gate * (coefficients[0] * z**3 + coefficients[1] * z**4)
    if variant.startswith("clock_"):
        return reaction_clock_step(middle, equation.reaction_rate * step / 2 + increment)
    if variant == "pre_capacity_global":
        return reaction_step(capacity_update(middle, increment), step / 2, equation)
    base = reaction_step(middle, step / 2, equation)
    if variant.startswith("post_capacity_"):
        return capacity_update(base, increment)
    return base + increment


def capacity_coordinate(base: float, target: float) -> tuple[float, bool]:
    """Invert the capacity map for an interior target, without clipping it."""
    delta = target - base
    if delta == 0:
        return 0., True
    capacity = 1 - base if delta > 0 else base
    if not (0 <= base <= 1 and 0 <= target <= 1) or abs(delta) >= capacity:
        return 0., False
    return delta * capacity / (capacity - abs(delta)), True


def _row(case_id, mechanism, test, variant, kind, outcome, metrics, inputs, note):
    for key, value in metrics.items():
        if not isinstance(value, (str, bool, int, float)):
            raise TypeError(f"Metric {key} is not scalar")
        if isinstance(value, (float, int)) and not math.isfinite(value):
            raise ValueError(f"Nonfinite coordinate metric: {key}")
    return dict(case_id=case_id, mechanism=mechanism, test=test, variant=variant,
                kind=kind, outcome=outcome, metrics=metrics, inputs=inputs, note=note)


def _teacher(u, h, equation, geometry, tolerance, check_budget):
    count = max(4, choose_substeps(h, equation, geometry))
    for _ in range(5):
        check_budget()
        result = refined_reference(u, h, equation, geometry, count, tolerance=tolerance)
        if result.accepted:
            break
        count *= 2
    check_budget()
    return result


def _reference_metrics(result):
    return {
        "reference_accepted": result.accepted,
        "reference_uncertainty_rms": result.uncertainty,
        "reference_defect_rms": result.defect_norm,
        "reference_n": result.refinement_substeps[0],
        "reference_2n": result.refinement_substeps[1],
        "reference_4n": result.refinement_substeps[2],
        "reference_difference_n_2n": result.refinement_differences[0],
        "reference_difference_2n_4n": result.refinement_differences[1],
        "reference_order_resolved": result.observed_order is not None,
        "reference_observed_order": result.observed_order if result.observed_order is not None else 0.,
        "reference_reason": result.reason,
    }


def _coefficient(function, order):
    step = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    value = function(step)
    for _ in range(order):
        value = torch.autograd.grad(value, step, create_graph=True)[0]
    return float(value.detach()) / math.factorial(order)


def _exact_limits(rows, check_budget):
    geometry = Geometry((8,), (1.,))
    pattern = torch.tensor([[[0., .2, .8, 1., .3, .6, .4, .1]]], dtype=torch.float64)
    cases = [("zero_h", pattern, 0., Equation(.03, 2.)),
             ("zero_diffusion", pattern, .2, Equation(0., 2.)),
             ("zero_reaction", pattern, .2, Equation(.03, 0.))]
    cases += [(f"uniform_{index}", torch.full_like(pattern, value), .2, Equation(.03, 2.))
              for index, value in enumerate((0., .27, 1.))]
    for name, state, h, equation in cases:
        check_budget()
        base = split_step(state, h, equation, geometry)
        for variant in VARIANTS:
            answer = controlled_step(state, h, equation, geometry, variant)
            error = float((answer - base).abs().max())
            rows.append(_row(f"limits/{name}/{variant}", "coordinates_gate", "commuting_limits",
                variant, "correctness", "PASS" if error <= 8 * torch.finfo(state.dtype).eps else "FAIL",
                {"max_difference_from_split": error},
                {"h": h, "kappa": equation.kappa, "reaction_rate": equation.reaction_rate},
                "Nonzero prescribed heads must retain the physical split in exact commuting limits."))
    for variant in VARIANTS:
        check_budget()
        coefficients = torch.zeros(2, dtype=torch.float64, requires_grad=True)
        answer = controlled_step(pattern, .2, Equation(.03, 2.), geometry, variant,
                                 (coefficients[0], coefficients[1]))
        derivative = torch.autograd.grad(answer.sum(), coefficients)[0]
        difference = float((answer - split_step(pattern, .2, Equation(.03, 2.), geometry)).detach().abs().max())
        live = bool(torch.isfinite(derivative).all() and (derivative.abs() > 1e-12).all())
        rows.append(_row(f"zero_heads/{variant}", "coordinates_gate", "zero_head_identity_and_gradient",
            variant, "correctness", "PASS" if live and difference <= 2e-15 else "FAIL",
            {"max_difference_from_split": difference, "cubic_head_derivative": float(derivative[0]),
             "quartic_head_derivative": float(derivative[1]), "both_gradients_live": live},
            {"h": .2, "kappa": .03, "reaction_rate": 2.},
            "The zero-head gradient is tested on this nonuniform interior-after-diffusion state only."))
    for coefficient in (-8., 8.):
        for h in (.01, .2, 3.):
            check_budget()
            for variant in VARIANTS:
                answer = controlled_step(pattern, h, Equation(.03, 2.), geometry,
                                         variant, (coefficient, coefficient))
                bounded = bool(torch.isfinite(answer).all() and ((answer >= 0) & (answer <= 1)).all())
                negative_control = variant == "additive_global"
                rows.append(_row(f"forward_bounds/{coefficient}/{h}/{variant}", "bounded_coordinates",
                    "actual_diffused_field_bounds", variant,
                    "negative_control" if negative_control else "correctness",
                    "OBSERVED" if negative_control else ("PASS" if bounded else "FAIL"),
                    {"minimum": float(answer.min()), "maximum": float(answer.max()), "output_in_bounds": bounded},
                    {"h": h, "coefficients": [coefficient, coefficient], "kappa": .03, "reaction_rate": 2.},
                    "Actual complete physical-plus-correction map; no clipping masks FFT or output excursions. Additive is an explicit unbounded control."))


def _capacity_audits(rows, check_budget):
    for dtype in (torch.float64, torch.float32):
        check_budget()
        values = torch.tensor([0., 2**-20, .2, .5, .8, 1 - 2**-20, 1.], dtype=dtype)
        increments = torch.tensor([-1e12, -.1, -1e-9, 0., 1e-9, .1, 1e12], dtype=dtype)
        result = capacity_update(values[None], increments[:, None])
        zero = torch.zeros_like(values, requires_grad=True)
        derivative = torch.autograd.grad(capacity_update(values, zero).sum(), zero)[0]
        bounded = bool(torch.isfinite(result).all() and ((result >= 0) & (result <= 1)).all())
        rows.append(_row(f"capacity/bounds/{dtype}", "bounded_coordinates", "signed_bounds_and_live_gradient",
            "capacity", "correctness", "PASS" if bounded and torch.allclose(derivative[1:-1], torch.ones(5, dtype=dtype)) else "FAIL",
            {"minimum": float(result.min()), "maximum": float(result.max()),
             "minimum_interior_zero_increment_derivative": float(derivative[1:-1].min()),
             "maximum_interior_zero_increment_derivative": float(derivative[1:-1].max())},
            {"dtype": str(dtype), "states": values.tolist(), "increments": increments.tolist()},
            "Interior zero-increment gradients are live; no derivative claim is made at endpoint sign transitions."))
    for state in (0., 2**-20, .5, 1 - 2**-20, 1.):
        for sign in (-1, 1):
            check_budget()
            initial = torch.tensor(state, dtype=torch.float64)
            increment = torch.tensor(sign * .001, dtype=torch.float64)
            for variant in ("clock", "capacity", "additive"):
                if variant == "clock":
                    result = reaction_clock_step(initial, increment)
                elif variant == "capacity":
                    result = capacity_update(initial, increment)
                else:
                    result = initial + increment
                target = state + float(increment)
                limitation = variant == "clock" and state in (0., 1.)
                rows.append(_row(f"reachable/{state}/{sign}/{variant}", "bounded_coordinates", "signed_near_bound_response",
                    variant, "representation", "EXPECTED_LIMITATION" if limitation else "OBSERVED",
                    {"output": float(result), "actual_state_change": float(result - initial),
                     "state_target_error": abs(float(result) - target),
                     "target_in_bounds": 0 <= target <= 1, "output_in_bounds": 0 <= float(result) <= 1,
                     "clock_linear_sensitivity": state * (1 - state)},
                    {"state": state, "raw_coordinate": float(increment)},
                    "The common raw coordinate is a clock shift for clock and a state increment otherwise; it is not an equal-output calibration."))
    for power in (0, 1, 3):
        for sign in (-1, 1):
            for h in (.1, .05, .025, .0125):
                check_budget()
                capacity = .4 * h**power
                base = torch.tensor(1 - capacity if sign > 0 else capacity, dtype=torch.float64)
                increment = torch.tensor(sign * .2 * h**3, dtype=torch.float64, requires_grad=True)
                answer = capacity_update(base, increment)
                derivative = torch.autograd.grad(answer, increment)[0]
                ratio = float(((answer - base) / increment).detach())
                expected = capacity / (capacity + abs(float(increment.detach())))
                rows.append(_row(f"capacity/scaling/{power}/{sign}/{h}", "bounded_coordinates", "directional_capacity_scaling",
                    "capacity", "representation", "EXPECTED_LIMITATION" if power == 3 else "OBSERVED",
                    {"correction_retention": ratio, "analytic_retention": expected,
                     "coordinate_derivative": float(derivative), "directional_capacity": capacity,
                     "coefficient_error": abs(float(((answer - base) / h**3).detach()) - sign * .2)},
                    {"h": h, "capacity_power": power, "sign": sign, "cubic_coefficient": .2},
                    "Capacity of cubic order changes the leading coefficient; constant/linear capacity recovers it only asymptotically."))
    curvature = {}
    for label, direction in (("left", -1.), ("right", 1.)):
        increment = torch.tensor(direction * 1e-7, dtype=torch.float64, requires_grad=True)
        value = capacity_update(torch.tensor(.23, dtype=torch.float64), increment)
        first = torch.autograd.grad(value, increment, create_graph=True)[0]
        second = torch.autograd.grad(first, increment)[0]
        curvature[f"{label}_first_derivative"] = float(first.detach())
        curvature[f"{label}_second_derivative"] = float(second)
    rows.append(_row("capacity/curvature", "bounded_coordinates", "one_sided_curvature", "capacity",
        "representation", "EXPECTED_LIMITATION", curvature, {"state": .23, "epsilon": 1e-7},
        "One-sided second derivatives differ: this map is generally C1, not C2, at zero correction."))


def _two_site(rows, tolerance, smoke, check_budget):
    geometry = Geometry((2,), (1.,))
    state = torch.tensor([[[0., .7]]], dtype=torch.float64)
    equation = Equation(.05, 2.)
    diffusion = torch.tensor([[-.4, .4], [.4, -.4]], dtype=torch.float64)
    vector = state.flatten()
    field = diffusion @ vector + 2 * vector * (1 - vector)
    jacobian = diffusion + torch.diag(2 * (1 - 2 * vector))
    exact_cubic = float(((jacobian @ jacobian @ field - 4 * field.square()) / 6)[0])
    base_cubic = _coefficient(lambda h: split_step(state, h, equation, geometry)[0, 0, 0], 3)
    for variant in VARIANTS:
        check_budget()
        cubic = _coefficient(lambda h: (controlled_step(state, h, equation, geometry, variant)
                             - split_step(state, h, equation, geometry))[0, 0, 0], 3)
        rows.append(_row(f"two_site/cubic/{variant}", "bounded_coordinates", "analytic_boundary_cubic",
            variant, "representation", "EXPECTED_LIMITATION" if variant.startswith("clock_") else "OBSERVED",
            {"required_defect_cubic": exact_cubic - base_cubic, "prescribed_correction_cubic": cubic,
             "cubic_coefficient_gap": abs(cubic - (exact_cubic - base_cubic))},
            {"state": [0., .7], "kappa": .05, "reaction_rate": 2., "coefficients": [.7, -.2]},
            "The coupled cubic comes from an independent two-site ODE Jacobian; heads are fixed, not fitted to cancel it."))
    equations = (equation,) if smoke else tuple(Equation(k, r) for k in (.025, .05) for r in (1., 2.))
    for equation in equations:
        for h in (.04, .02, .01, .005):
            reference = _teacher(state, h, equation, geometry, tolerance, check_budget)
            base = split_step(state, h, equation, geometry)
            for variant in VARIANTS:
                answer = controlled_step(state, h, equation, geometry, variant)
                metrics = _reference_metrics(reference)
                metrics.update(error_rms=float((answer - reference.state).square().mean().sqrt()),
                               error_receiver=float((answer - reference.state)[0, 0, 0]),
                               required_receiver_defect=float((reference.state - base)[0, 0, 0]),
                               correction_over_h3=float((answer - base)[0, 0, 0] / h**3),
                               output_in_bounds=bool(((answer >= 0) & (answer <= 1)).all()))
                rows.append(_row(f"two_site/reference/{equation.kappa}/{equation.reaction_rate}/{h}/{variant}",
                    "bounded_coordinates", "coupled_three_level_reference", variant, "scientific",
                    "OBSERVED" if reference.accepted else "INCONCLUSIVE", metrics,
                    {"h": h, "kappa": equation.kappa, "reaction_rate": equation.reaction_rate},
                    "Fixed coefficients are shared across variants; observed error improvements are not trained performance."))
            pre = controlled_step(state, h, equation, geometry, "pre_capacity_global")
            post = controlled_step(state, h, equation, geometry, "post_capacity_global")
            rows.append(_row(f"placement/{equation.kappa}/{equation.reaction_rate}/{h}", "bounded_coordinates",
                "pre_post_placement", "pre_vs_post_capacity", "representation", "OBSERVED",
                {"max_difference": float((pre - post).abs().max()),
                 "difference_over_h4": float((pre - post).abs().max() / h**4)},
                {"h": h, "kappa": equation.kappa, "reaction_rate": equation.reaction_rate},
                "At this smooth fixed-grid small-h setting, placement changes higher-order terms, not the available leading state direction."))


def _remote_context(rows, tolerance, smoke, check_budget):
    n = 8 if smoke else 16
    geometry = Geometry((n,), (1.,))
    constant = torch.full((1, 1, n), .25, dtype=torch.float64)
    equations = (Equation(.03, 2.),) if smoke else tuple(Equation(k, r) for k in (.03, .09) for r in (1., 3.))
    for distance in ((3,) if smoke else (3, 5)):
        bump = constant.clone()
        bump[..., distance] += .5
        for equation in equations:
            local = commuting_gate(bump, equation, geometry)[0, 0, 0]
            global_gate = global_commuting_gate(bump, equation, geometry)[0, 0, 0]
            for h in ((.1, .3) if smoke else (.1, .3, .6)):
                reference = _teacher(bump, h, equation, geometry, tolerance, check_budget)
                # The comparison field is uniform and has an analytic exact reaction solution.
                uniform_exact = reaction_step(constant, h, equation)
                uniform_base = split_step(constant, h, equation, geometry)
                middle = diffusion_step(reaction_step(bump, h / 2, equation), h, equation, geometry)
                uniform_middle = reaction_step(constant, h / 2, equation)
                base = reaction_step(middle, h / 2, equation)
                receiver, target = float(base[0, 0, 0]), float(reference.state[0, 0, 0])
                # sqrt(n) converts the reference RMS difference into a conservative cell bound.
                uncertainty = math.sqrt(n) * reference.uncertainty
                clock = math.log(target / (1 - target)) - math.log(receiver / (1 - receiver))
                interval_min = max(torch.finfo(torch.float64).tiny, target - uncertainty)
                interval_max = min(1 - torch.finfo(torch.float64).eps, target + uncertainty)
                clock_uncertainty = uncertainty / min(interval_min * (1 - interval_min), interval_max * (1 - interval_max))
                post_coordinate, post_reachable = capacity_coordinate(receiver, target)
                target_middle = float(reaction_clock_step(torch.tensor(target, dtype=torch.float64),
                                      torch.tensor(-equation.reaction_rate * h / 2, dtype=torch.float64)))
                pre_coordinate, pre_reachable = capacity_coordinate(float(middle[0, 0, 0]), target_middle)
                for variant in VARIANTS:
                    answer = controlled_step(bump, h, equation, geometry, variant)
                    metrics = _reference_metrics(reference)
                    metrics.update(
                        local_gate=float(local), global_gate=float(global_gate),
                        initial_radius_two_patch_max_difference=float((bump - constant)[..., [n - 2, n - 1, 0, 1, 2]].abs().max()),
                        physical_middle_difference=float((middle - uniform_middle)[0, 0, 0]),
                        uniform_exact_minus_split=float((uniform_exact - uniform_base)[0, 0, 0]),
                        required_clock_coordinate=clock, required_clock_uncertainty_bound=clock_uncertainty,
                        required_post_capacity_coordinate=post_coordinate,
                        required_pre_capacity_coordinate=pre_coordinate,
                        post_coordinate_reachable=post_reachable, pre_coordinate_reachable=pre_reachable,
                        receiver_reference_uncertainty_bound=uncertainty,
                        required_receiver_defect=target - receiver,
                        receiver_error=abs(float(answer[0, 0, 0]) - target),
                        resolved_nonzero_required_clock=abs(clock) > 5 * clock_uncertainty,
                    )
                    rows.append(_row(f"remote/{distance}/{equation.kappa}/{equation.reaction_rate}/{h}/{variant}",
                        "spatial_context_gate", "locally_identical_remote_bump", variant, "representation",
                        "OBSERVED" if reference.accepted else "INCONCLUSIVE", metrics,
                        {"grid": n, "receiver": 0, "bump_distance_cells": distance, "h": h,
                         "kappa": equation.kappa, "reaction_rate": equation.reaction_rate},
                        "The diffused physical intermediates already differ. Required oracle coordinates are computed relative to each physical base; a raw target difference is not a local-encoder lower bound. No oracle coordinates are fed to the probes."))


def plan(config: dict) -> dict:
    """Enumerate fixed cases without evaluating a model or a reference."""
    smoke = bool(config.get("smoke", False))
    ids = [f"limits/{name}/{variant}" for name in
           ("zero_h", "zero_diffusion", "zero_reaction", "uniform_0", "uniform_1", "uniform_2")
           for variant in VARIANTS]
    ids += [f"zero_heads/{variant}" for variant in VARIANTS]
    ids += [f"forward_bounds/{coefficient}/{h}/{variant}" for coefficient in (-8., 8.)
            for h in (.01, .2, 3.) for variant in VARIANTS]
    ids += [f"capacity/bounds/{dtype}" for dtype in (torch.float64, torch.float32)]
    ids += [f"reachable/{state}/{sign}/{variant}" for state in (0., 2**-20, .5, 1 - 2**-20, 1.)
            for sign in (-1, 1) for variant in ("clock", "capacity", "additive")]
    ids += [f"capacity/scaling/{power}/{sign}/{h}" for power in (0, 1, 3)
            for sign in (-1, 1) for h in (.1, .05, .025, .0125)]
    ids += ["capacity/curvature"]
    ids += [f"two_site/cubic/{variant}" for variant in VARIANTS]
    two_site_equations = ((.05, 2.),) if smoke else tuple((k, r) for k in (.025, .05) for r in (1., 2.))
    for kappa, rate in two_site_equations:
        for h in (.04, .02, .01, .005):
            ids += [f"two_site/reference/{kappa}/{rate}/{h}/{variant}" for variant in VARIANTS]
            ids += [f"placement/{kappa}/{rate}/{h}"]
    remote_equations = ((.03, 2.),) if smoke else tuple((k, r) for k in (.03, .09) for r in (1., 3.))
    distances = (3,) if smoke else (3, 5)
    remote_horizons = (.1, .3) if smoke else (.1, .3, .6)
    ids += [f"remote/{distance}/{kappa}/{rate}/{h}/{variant}" for distance in distances
            for kappa, rate in remote_equations for h in remote_horizons for variant in VARIANTS]
    return {
        "panel": "coordinates", "expected_case_ids": ids,
        "variants": list(VARIANTS), "coefficients": [.7, -.2],
        "two_site": {"grid": 2, "state": [0., .7], "equations": list(two_site_equations),
                     "horizons": [.04, .02, .01, .005]},
        "remote_context": {"grid": 8 if smoke else 16, "base_state": .25, "bump": .5,
                           "distances": list(distances), "equations": list(remote_equations),
                           "horizons": list(remote_horizons), "receiver": 0},
        "reference": {"levels": [1, 2, 4], "maximum_attempts": 5,
                      "initial_count": "max(4, choose_substeps(h,equation,geometry))",
                      "tolerance": float(config.get("tolerance", .002)), "error_fraction": .05},
        "capacity_scaling": {"powers": [0, 1, 3], "capacity_coefficient": .4,
                             "correction_coefficient": .2, "horizons": [.1, .05, .025, .0125]},
        "forward_bounds": {"coefficients": [-8., 8.], "horizons": [.01, .2, 3.]},
    }


def run(config: dict, check_budget: Callable[[], None]) -> dict:
    """Run bounded deterministic FP64/FP32 probes with no optimizer or GPU."""
    check_budget()
    tolerance = float(config.get("tolerance", .002))
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Coordinate audit tolerance must be finite and positive")
    rows = []
    smoke = bool(config.get("smoke", False))
    _exact_limits(rows, check_budget)
    _capacity_audits(rows, check_budget)
    _two_site(rows, tolerance, smoke, check_budget)
    _remote_context(rows, tolerance, smoke, check_budget)
    check_budget()
    counts = Counter(row["outcome"] for row in rows)
    return {
        "panel": "coordinates", "status": "COMPLETED", "rows": rows,
        "summary": {"row_count": len(rows), "outcome_counts": dict(counts),
                    "correctness_failed": sum(row["kind"] == "correctness" and row["outcome"] == "FAIL" for row in rows),
                    "variants": list(VARIANTS), "trained_models": 0,
                    "distinct_reference_cases": len({row["case_id"].rsplit("/", 1)[0] for row in rows
                        if "reference_accepted" in row["metrics"]}),
                    "reference_rows_rejected": sum(row["metrics"].get("reference_accepted") is False for row in rows)},
        "limitations": [
            "Controlled coefficient probes and oracle-coordinate observations are not learned-model validation or evidence of superiority.",
            "Capacity updates preserve the interval conditional on an admissible input; bounds do not establish stability or uniform third-order defect cancellation.",
            "The global gate is a smooth information probe with mesh-dependent scaling, not a resolution-independent selected architecture.",
            "Remote-bump comparisons include the different diffused physical intermediates; only required correction coordinates are compared.",
            "Teacher uncertainty is a conservative refinement diagnostic, not a rigorous exact-solution error bound; unresolved cases remain inconclusive.",
            "No GPU timing, training, physical boundary condition change, or learned spatial encoder is included.",
        ],
    }
