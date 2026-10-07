"""Bounded, training-free tests of the research audit's response hypotheses.

Oracle least-squares fits expose representation limits; none is a deployable
solver or a claim of learned superiority. Correctness checks and scientific
observations are separate, and every declared case remains in the report.
"""
from __future__ import annotations

from collections import Counter
import itertools
import math
from pathlib import Path
import time

import numpy as np
import torch

from tdn.numerics import Equation, Geometry
from tdn.numerics.operators import laplacian, rhs
from tdn.numerics.reference import choose_substeps, reference_step
from tdn.numerics.splitting import split_step
from tdn.numerics.subflows import diffusion_step, reaction_step
from tdn.research.consistency_neural import discrete_logistic_commutator
from tdn.research.interaction import etdrk4_step
from tdn.research.reaction import capacity_update
from tdn.runtime.metadata import write_json

from .physics import (apply_pair_kernel, coupled_quadratic_response, dealiased_square,
                      discrete_eigenvalues, diffusion_first_step, moment_derivatives,
                      quadratic_pair_kernel, split_quadratic_response,
                      symmetric_quadratic_reference)


from .structural_spec import SCHEMA, specification


def _rms(value):
    return float(value.double().square().mean().sqrt())


def _direction(name, geometry):
    n = geometry.grid[0]
    x = torch.arange(n, dtype=torch.float64) / n
    if name == "uniform":
        value = torch.ones_like(x)
    elif name == "single_low":
        value = torch.cos(2 * math.pi * x + .31)
    elif name == "single_high":
        value = torch.cos(2 * math.pi * (n // 3) * x - .52)
    elif name.startswith("mixed_phase"):
        phase = .17 if name.endswith("0") else 1.27
        value = torch.cos(2 * math.pi * 2 * x + phase) + .7 * torch.cos(2 * math.pi * 5 * x - .61)
    elif name in ("remote_bump", "remote_bumps"):
        distance = torch.minimum((x - .2).abs(), 1 - (x - .2).abs())
        value = torch.where(distance < .08, torch.cos(math.pi * distance / .16).square(), torch.zeros_like(x))
        if name == "remote_bumps":
            second = torch.minimum((x - .7).abs(), 1 - (x - .7).abs())
            value = value - .6 * torch.where(second < .10, torch.cos(math.pi * second / .20).square(), torch.zeros_like(x))
        value = value - value.mean()
    elif name == "nyquist_pair":
        value = torch.cos(2 * math.pi * (n // 2 - 1) * x + .3) + .7 * torch.sin(2 * math.pi * (n // 2 - 2) * x - .4)
    else:
        raise ValueError("Unknown structural pattern")
    return (value / value.abs().max().clamp_min(1e-15)).reshape(1, 1, n)


def _fit_span(target, columns):
    if not columns:
        return dict(rms=_rms(target), maximum=float(target.abs().max()), rank=0, coefficients=[])
    a = torch.stack([x.flatten() for x in columns], dim=1).double()
    scales = torch.linalg.vector_norm(a, dim=0)
    keep = scales > 1e-15
    if not bool(keep.any()):
        return dict(rms=_rms(target), maximum=float(target.abs().max()), rank=0, coefficients=[0.] * len(columns))
    normalized = a[:, keep] / scales[keep]
    solution = torch.linalg.lstsq(normalized, target.flatten().double(), rcond=1e-11, driver="gelsd")
    residual = target.flatten() - normalized @ solution.solution
    coefficients = torch.zeros(len(columns), dtype=torch.float64)
    coefficients[keep] = solution.solution / scales[keep]
    return dict(rms=_rms(residual), maximum=float(residual.abs().max()),
                rank=int(solution.rank), coefficients=coefficients.tolist())


def _resolved_teacher(state, h, equation, geometry, check):
    n = max(16, choose_substeps(h, equation, geometry))
    answers = [reference_step(state, h, equation, geometry, count, check=check)
               for count in (n, 2 * n, 4 * n)]
    uncertainty = max(_rms(answers[2] - answers[1]), 64 * torch.finfo(torch.float64).eps)
    return answers[-1], uncertainty


def _capacity_inverse(base, target):
    delta = target - base
    capacity = torch.where(delta >= 0, 1 - base, base)
    fraction = delta.abs() / capacity.clamp_min(1e-30)
    if bool((fraction >= 1).any()):
        raise ValueError("Teacher target exceeds finite signed-capacity reachability")
    return delta / (1 - fraction)


def run(protocol, path, stop=None):
    """Write every structural/literature case, witnesses and bounded summary."""
    spec = protocol.get("structural", specification(protocol.get("profile", "smoke")))
    if spec != specification(spec.get("profile", "smoke")):
        raise ValueError("Structural specification differs from its immutable profile")
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    rows, witnesses = [], {}
    geometry = Geometry((spec["grid"],), (1.,))
    expected = set(spec["case_ids"])

    def check():
        if stop is not None and (getattr(stop, "requested", False) or getattr(stop, "signal_number", None)):
            raise InterruptedError("Structural stop requested; preserve this attempt")
        if time.monotonic() - start >= spec["max_seconds"]:
            raise TimeoutError("Structural numerical budget exhausted")

    def append(identity, panel, question, *, passed=None, **values):
        check()
        required = identity in spec["required_case_ids"]
        outcome = ("PASS" if passed else "FAILED") if required else "OBSERVED"
        questions = {"gate_span": ["Q2", "Q3"], "temporal_capacity": ["Q1", "Q2", "Q6"], "strong_diffusion_orientation": ["Q2"], "moment_closure": ["Q2", "Q3"], "separable_phase_kernel": ["Q2", "Q3"], "aliasing_discretization": ["Q4"], "cubic_finite_step": ["Q1"], "stiff_input_output_scales": ["Q2"], "defect_estimator_acceptance": ["Q5"], "unequal_unseen_steps": ["Q6"]}[question]
        record_type = "check" if required else "rank" if panel == "kernel_rank" else "case"
        rows.append(dict(case_id=identity, record_type=record_type, panel=panel,
                         question_id=question, question_ids=questions,
                         literature_question_ids=questions, required=required,
                         outcome=outcome, status=outcome, **values))
        # Preserve partial evidence even if a later case exhausts its budget.
        write_json(directory / "cases.json", dict(schema=SCHEMA, rows=rows))
        write_json(directory / "rows.json", dict(schema="tdn.agenda-rows/v1", stage="structure", rows=rows))

    rank_rows = []
    try:
        for index, (c, h, kappa, rate) in enumerate(spec["response_settings"]):
            equation = Equation(kappa, rate)
            for pattern in spec["patterns"]:
                check()
                v = _direction(pattern, geometry)
                response = coupled_quadratic_response(v, c, h, equation, geometry,
                                                     quadrature_nodes=spec["quadrature_nodes"])
                refined = coupled_quadratic_response(v, c, h, equation, geometry,
                                                    quadrature_nodes=spec["uncertainty_nodes"])
                independent = symmetric_quadratic_reference(v, c, h, equation, geometry,
                                                            epsilon=spec["symmetric_epsilon"], check=check)
                error = _rms(response - independent["response"])
                uncertainty = independent["uncertainty_rms"] + _rms(response - refined)
                prefix = f"response/{index}/{pattern}"
                append(prefix + "/teacher", "quadratic", "gate_span",
                       passed=independent["accepted"] and error <= uncertainty + 2e-8,
                       background=c, horizon=h, kappa=kappa, reaction_rate=rate,
                       teacher_error_rms=error, quadratic_error=error, uncertainty_rms=uncertainty,
                       epsilon=spec["symmetric_epsilon"], uncertainty_after_scaling=uncertainty,
                       epsilon_levels=independent["epsilon_levels"],
                       refinement_counts=independent["refinement_counts"],
                       amplitude_uncertainty=independent["amplitude_uncertainty_rms"],
                       uncertainty_after_amplitude_squared_division=True)
                defect = response - split_quadratic_response(v, c, h, equation, geometry)
                source = discrete_logistic_commutator(v, equation, geometry)
                L = lambda x: equation.kappa * laplacian(x, geometry)
                filtered = [source] + [diffusion_step(source, h * fraction, equation, geometry)
                                       for fraction in (.25, .5, 1.)]
                signed = filtered + [L(source), equation.reaction_rate * L(v * L(v)),
                                     equation.reaction_rate * L(v).square(),
                                     equation.reaction_rate * v * L(L(v))]
                spans = dict(local_gate=_fit_span(defect, [source]),
                             local_gate_and_constant=_fit_span(defect, [source, torch.ones_like(source)]),
                             filtered_source=_fit_span(defect, filtered),
                             signed_source_dictionary=_fit_span(defect, signed))
                for fit in spans.values():
                    fit["relative_rms"] = fit["rms"] / max(_rms(defect), 1e-15)
                    fit["above_reference_uncertainty"] = fit["rms"] > spec["span_uncertainty_multiplier"] * uncertainty
                receivers = source == 0
                receiver_metrics = dict(receiver_cells=int(receivers.sum()),
                                        receiver_defect_rms=_rms(defect[receivers]) if bool(receivers.any()) else None,
                                        receiver_source_rms=_rms(source[receivers]) if bool(receivers.any()) else None)
                key = f"quadratic_{index}_{pattern}"
                witnesses[key + "_defect"] = defect.numpy()
                witnesses[key + "_source"] = source.numpy()
                append(prefix + "/span", "gate_span", "gate_span", background=c, horizon=h,
                       kappa=kappa, reaction_rate=rate, pattern=pattern, defect_rms=_rms(defect),
                       uncertainty_rms=uncertainty, oracle_spans=spans, witness_prefix=key,
                       residual_rms=spans["local_gate_and_constant"]["rms"],
                       relative_error=spans["local_gate_and_constant"]["relative_rms"],
                       **receiver_metrics,
                       interpretation="best unconstrained scalar/dictionary fit; representation diagnostic, no learned result")

        for index, (c, amplitude, kappa, rate, h) in enumerate(spec["strong_diffusion_settings"]):
            check()
            equation = Equation(kappa, rate)
            v = _direction("single_low", geometry)
            state = c + amplitude * v
            truth, uncertainty = _resolved_teacher(state, h, equation, geometry, check)
            reaction_first = split_step(state, h, equation, geometry)
            diffusion_first = diffusion_first_step(state, h, equation, geometry)
            etdrk4 = etdrk4_step(state, h, equation, geometry)
            homogenized = reaction_step(state.mean(-1, keepdim=True).expand_as(state), h, equation)
            wrong_limit = reaction_step(reaction_step(state, h / 2, equation).mean(-1, keepdim=True).expand_as(state), h / 2, equation)
            lam = discrete_eigenvalues(geometry, state, equation)
            gap = float(-lam[1])
            append(f"strong_diffusion/{index}", "strong_diffusion", "strong_diffusion_orientation",
                   background=c, amplitude=amplitude, kappa=kappa, reaction_rate=rate, horizon=h,
                   spectral_gap_h=gap * h, largest_eigenvalue_h=float(lam.abs().max()) * h,
                   reference_uncertainty=uncertainty,
                   reaction_first_rms=_rms(reaction_first - truth), diffusion_first_rms=_rms(diffusion_first - truth),
                   etdrk4_rms=_rms(etdrk4 - truth),
                   coupled_to_homogenized_rms=_rms(truth - homogenized),
                   reaction_first_to_its_limit_rms=_rms(reaction_first - wrong_limit),
                   reaction_first_limit_bias_rms=_rms(wrong_limit - homogenized),
                   diffusion_first_to_homogenized_rms=_rms(diffusion_first - homogenized),
                   interpretation="global homogenization is governed by spectral gap, not largest eigenvalue")

        for index, (c, h, kappa, rate) in enumerate(spec["rank_settings"]):
            equation = Equation(kappa, rate)
            v = _direction("mixed_phase1", geometry)
            exact = quadratic_pair_kernel(c, h, equation, geometry, v, quadrature_nodes=128,
                                          orientation="reaction_first")
            check_kernel = quadratic_pair_kernel(c, h, equation, geometry, v, quadrature_nodes=64,
                                                orientation="reaction_first")
            response = apply_pair_kernel(v, exact)
            for rank in spec["ranks"]:
                check()
                approximate = quadratic_pair_kernel(c, h, equation, geometry, v,
                                                    quadrature_nodes=rank, orientation="reaction_first")
                field = apply_pair_kernel(v, approximate)
                relative = _rms(approximate - exact) / max(_rms(exact), 1e-15)
                info = dict(response_rank=rank, rank=rank, defect_separable_terms=rank + 2,
                            kernel_relative_rms=relative, field_relative_rms=_rms(field - response) / max(_rms(response), 1e-15),
                            kernel_uncertainty_rms=_rms(exact - check_kernel),
                            promising=relative <= spec["rank_relative_tolerance"], rank_gate_passed=relative <= spec["rank_relative_tolerance"],
                            rank_residual=relative, relative_error=relative,
                            background=c, horizon=h, kappa=kappa, reaction_rate=rate,
                            transforms=dict(shared_input_forward=1, input_inverse=rank,
                                            product_forward=rank, output_inverse=1,
                                            analytic_split_extra_transforms=4),
                            phase_and_pair_symmetry_error=float((approximate - approximate.T).abs().max()),
                            interpretation="quadrature-factor oracle; not fitted/trained kernel, no neural accuracy claim")
                rank_rows.append(info)
                append(f"kernel_rank/{index}/{rank}", "kernel_rank", "separable_phase_kernel", **info)

        for index, (pattern, kappa, rate) in enumerate(spec["temporal_settings"]):
            equation = Equation(kappa, rate)
            state = .4 + .15 * _direction(pattern, geometry)
            train = spec["temporal_train_steps"]
            unseen = spec["temporal_unseen_steps"]
            all_steps = train + unseen
            bases, targets, increments, uncertainties = [], [], [], []
            for h in all_steps:
                target, uncertainty = _resolved_teacher(state, h, equation, geometry, check)
                base = split_step(state, h, equation, geometry)
                bases.append(base); targets.append(target); uncertainties.append(uncertainty)
                increments.append(_capacity_inverse(base, target).flatten() / h**3)
            lam = discrete_eigenvalues(geometry, state, equation)
            fast, slow = float(lam.abs().max()), float(-lam[1])
            polynomial = lambda h: [1., h, h * h]
            nonlinear = lambda h: [1., math.exp(-fast * h), math.exp(-(slow + rate) * h)]
            fits = {}
            for name, basis in (("quadratic_raw", polynomial), ("three_feature_exponential", nonlinear)):
                matrix = torch.tensor([basis(h) for h in train], dtype=torch.float64)
                weights = torch.linalg.lstsq(matrix, torch.stack(increments[:len(train)]), driver="gelsd").solution
                errors, normalized = [], []
                for position, h in enumerate(unseen, start=len(train)):
                    proposal = h**3 * (torch.tensor(basis(h), dtype=torch.float64) @ weights).reshape_as(state)
                    result = capacity_update(bases[position], proposal)
                    errors.append(_rms(result - targets[position]))
                    normalized.append(errors[-1] / max(_rms(targets[position] - bases[position]), 1e-15))
                fits[name] = dict(unseen_rms=errors, unseen_relative_to_defect=normalized,
                                  fitted_parameters_per_cell=3, output_map="actual signed capacity_update")
            append(f"temporal_capacity/{index}", "temporal_capacity", "temporal_capacity",
                   pattern=pattern, kappa=kappa, reaction_rate=rate, train_steps=train,
                   genuinely_unseen_steps=unseen, reference_uncertainties=uncertainties, fits=fits,
                   interpretation="per-cell representation oracle; actual bounded map retained; neither decoder is a trained model")

        x = torch.arange(geometry.grid[0], dtype=torch.float64) / geometry.grid[0]
        cosine = lambda k, phase=0.: torch.cos(2 * math.pi * k * x + phase).reshape(1, 1, -1)
        spectral_states = [.4 + .1 * cosine(1), .4 + .1 * cosine(5)]
        phase_states = [.4 + .07 * (cosine(1) + cosine(2, phase)) for phase in (0., math.pi / 2)]
        equation = Equation(.003, 2.)
        for label, states in (("spectrum", spectral_states), ("phase", phase_states)):
            terms = [moment_derivatives(u, equation, geometry) for u in states]
            identities = []
            endpoints = []
            for u, values in zip(states, terms):
                direct = rhs(u, equation, geometry)
                identities.extend([abs(float(direct.mean() - values["mean_derivative"])),
                                   abs(float((2 * (u - u.mean()) * direct).mean() - values["variance_derivative"]))])
                endpoints.append(_resolved_teacher(u, .2, equation, geometry, check)[0])
            error = max(identities)
            append(f"moment/equal_moments_{label}", "moments", "moment_closure", passed=error <= 2e-12,
                   identity_maximum_error=error,
                   equal_initial_mean_error=abs(float(states[0].mean() - states[1].mean())),
                   equal_initial_variance_error=abs(float(terms[0]["variance"] - terms[1]["variance"])),
                   initial_moments=[{name: float(value) for name, value in item.items()} for item in terms],
                   endpoint_mean_difference=abs(float(endpoints[0].mean() - endpoints[1].mean())),
                   endpoint_spatial_rms_difference=_rms((endpoints[0] - endpoints[0].mean()) - (endpoints[1] - endpoints[1].mean())),
                   interpretation="equal initial mean/variance do not determine future mean; phase retains information beyond energy")

        v = _direction("nyquist_pair", geometry)
        nodal, dealiased = v.square(), dealiased_square(v, geometry)
        append("alias/nodal_vs_dealiased", "aliasing", "aliasing_discretization",
               difference_rms=_rms(nodal - dealiased), difference_maximum=float((nodal - dealiased).abs().max()),
               nodal_mean=float(nodal.mean()), dealiased_mean=float(dealiased.mean()),
               interpretation="declared cyclic nodal product and dealiased Galerkin product solve different discrete equations")
        discrete = discrete_eigenvalues(geometry, v)
        continuous = -(2 * math.pi * torch.fft.fftfreq(geometry.grid[0], dtype=torch.float64))**2
        continuous = continuous * geometry.grid[0]**2
        append("alias/discrete_vs_continuum_eigenvalues", "aliasing", "aliasing_discretization",
               low_mode_relative_difference=float((discrete[1] - continuous[1]).abs() / continuous[1].abs()),
               high_mode_relative_difference=float((discrete[geometry.grid[0] // 2 - 1] - continuous[geometry.grid[0] // 2 - 1]).abs() / continuous[geometry.grid[0] // 2 - 1].abs()),
               interpretation="time refinement cannot repair changing spatial multipliers or product convention")

        for label, pattern in (("low", "single_low"), ("high", "nyquist_pair")):
            v = _direction(pattern, geometry)
            equation = Equation(.003, 2.)
            horizons = [.005, .01, .02, .04, .08, .16, .32]
            defects = [coupled_quadratic_response(v, .4, h, equation, geometry) - split_quadratic_response(v, .4, h, equation, geometry) for h in horizons]
            magnitudes = [_rms(d) for d in defects]
            orders = [math.log(b / a, 2) if min(a, b) > 1e-15 else None for a, b in zip(magnitudes, magnitudes[1:])]
            append(f"cubic/finite_step_{label}", "finite_step", "cubic_finite_step", horizons=horizons,
                   quadratic_defect_rms=magnitudes, observed_local_orders=orders,
                   scaled_cubic_defect_rms=[m / h**3 for m, h in zip(magnitudes, horizons)],
                   high_mode_stiffness=[float(discrete_eigenvalues(geometry, v, equation).abs().max()) * h for h in horizons],
                   interpretation="small-step cubic order and finite-step stiff response are distinct properties")

        equation = Equation(.02, 4.)
        matrix = quadratic_pair_kernel(.4, .24, equation, geometry, v, quadrature_nodes=128, orientation="reaction_first")
        uncertainty_matrix = quadratic_pair_kernel(.4, .24, equation, geometry, v, quadrature_nodes=64, orientation="reaction_first")
        pair_lam = discrete_eigenvalues(geometry, v, equation)
        # Both pairs sum to output mode zero, yet have distinct input damping.
        pairs = [(1, geometry.grid[0] - 1), (geometry.grid[0] // 3, geometry.grid[0] - geometry.grid[0] // 3)]
        append("stiff/pair_same_output", "stiff_scales", "stiff_input_output_scales",
               pairs=[list(p) for p in pairs], output_mode=0,
               input_output_eigenvalues=[[float(pair_lam[k]), float(pair_lam[l]), float(pair_lam[(k + l) % geometry.grid[0]])] for k, l in pairs],
               kernel_uncertainty_maximum=float((matrix - uncertainty_matrix).abs().max()),
               defect_pair_coefficients=[float(matrix[k, l]) for k, l in pairs],
               coefficient_difference=abs(float(matrix[pairs[0]] - matrix[pairs[1]])),
               interpretation="output-frequency-only response cannot distinguish these input mode-pair scales")
        pairs = [(1, 1), (1, geometry.grid[0] - 1)]
        append("stiff/pair_same_inputs", "stiff_scales", "stiff_input_output_scales",
               pairs=[list(p) for p in pairs], output_modes=[2, 0],
               input_output_eigenvalues=[[float(pair_lam[k]), float(pair_lam[l]), float(pair_lam[(k + l) % geometry.grid[0]])] for k, l in pairs],
               kernel_uncertainty_maximum=float((matrix - uncertainty_matrix).abs().max()),
               defect_pair_coefficients=[float(matrix[k, l]) for k, l in pairs],
               coefficient_difference=abs(float(matrix[pairs[0]] - matrix[pairs[1]])),
               interpretation="input-frequency-only response cannot distinguish these output mode scales")

        state = .4 + .15 * _direction("mixed_phase1", geometry)
        equation = Equation(.003, 2.)
        tolerance = 2e-5
        estimators, errors = [], []
        for h in (.03, .06, .12, .24):
            one = split_step(state, h, equation, geometry)
            two = split_step(split_step(state, h / 2, equation, geometry), h / 2, equation, geometry)
            truth, uncertainty = _resolved_teacher(state, h, equation, geometry, check)
            estimators.append(_rms(two - one) / 3)
            errors.append((_rms(two - truth), uncertainty))
        append("estimator/false_acceptance", "acceptance", "defect_estimator_acceptance",
               tolerance=tolerance, horizons=[.03, .06, .12, .24], estimator=estimators,
               true_errors=[x[0] for x in errors], reference_uncertainties=[x[1] for x in errors],
               false_acceptance_count=sum(e <= tolerance and actual - uncertainty > tolerance for e, (actual, uncertainty) in zip(estimators, errors)),
               undecidable_count=sum(abs(actual - tolerance) <= uncertainty for actual, uncertainty in errors),
               interpretation="diagnostic estimator only; composition agreement is not independent validation; trained deployment policy separately calibrated")
        schedules = [[.03, .07, .02, .12], [.06, .06, .06, .06], [.11, .04, .09]]
        truth, uncertainty = _resolved_teacher(state, .24, equation, geometry, check)
        answers = []
        for schedule in schedules:
            answer = state
            for h in schedule:
                answer = split_step(answer, h, equation, geometry)
            answers.append(answer)
        append("variable_step/unequal_composition", "variable_steps", "unequal_unseen_steps",
               schedules=schedules, common_total_time=.24, reference_uncertainty=uncertainty,
               endpoint_true_rms=[_rms(u - truth) for u in answers],
               pairwise_composition_rms=[_rms(a - b) for a, b in itertools.combinations(answers, 2)],
               interpretation="endpoint truth and composition disagreement reported separately; agreeing biased predictions can remain inaccurate")
        reported = [r["case_id"] for r in rows]
        if set(reported) != expected or len(reported) != len(expected):
            raise RuntimeError("Structural coverage differs from frozen case IDs")
        failed = [r["case_id"] for r in rows if r["required"] and r["outcome"] != "PASS"]
        status = "FAILED" if failed else "COMPLETED"
        error = None
    except (Exception, KeyboardInterrupt) as exc:
        status = "INCOMPLETE" if isinstance(exc, (InterruptedError, TimeoutError, KeyboardInterrupt)) else "FAILED"
        failed = [r["case_id"] for r in rows if r["required"] and r["outcome"] != "PASS"]
        error = f"{type(exc).__name__}: {exc}"
    np.savez_compressed(directory / "fields.npz", **witnesses)
    rank_summary = {str(rank): dict(cases=sum(x["response_rank"] == rank for x in rank_rows),
                                  all_promising=sum(x["response_rank"] == rank for x in rank_rows) == len(spec["rank_settings"])
                                  and all(x["promising"] for x in rank_rows if x["response_rank"] == rank),
                                  maximum_relative_kernel_error=max((x["kernel_relative_rms"] for x in rank_rows if x["response_rank"] == rank), default=None))
                    for rank in spec["ranks"]}
    # Even a stopped first case has a canonical empty table.
    write_json(directory / "rows.json", dict(schema="tdn.agenda-rows/v1", stage="structure", rows=rows))
    write_json(directory / "cases.json", dict(schema=SCHEMA, rows=rows))
    summary = dict(schema=SCHEMA, benchmark_suite="agenda", stage="structure", status=status,
                   expected_cases=len(expected), reported_cases=len(rows),
                   required_failures=failed, correctness_failures=len(failed),
                   unreported_case_ids=sorted(expected - {x["case_id"] for x in rows}),
                   outcome_counts=dict(Counter(x["outcome"] for x in rows)),
                   elapsed_seconds=time.monotonic() - start, numerical_budget_seconds=spec["max_seconds"],
                   device="cpu", training_attempted=False, question_ids=spec["question_ids"],
                   rank_summary=rank_summary, error=error,
                   rank_promising=status == "COMPLETED" and any(v["all_promising"] and v["cases"] > 0 for v in rank_summary.values()),
                   selected_rank=min((int(k) for k, v in rank_summary.items() if status == "COMPLETED" and v["all_promising"] and v["cases"] > 0), default=None),
                   rank_gate_convention="oracle Gauss-Legendre response R terms plus two analytic Strang terms; learned commutator-prefactored rank R expands to up to 3R terms",
                   rank_gate_limitation="development eligibility only; passing the response quadrature oracle does not prove learned filter/MLP representability or solver advantage",
                   interpretation="oracle representation diagnostics; no learned/FNO/classical solver advantage established")
    write_json(directory / "summary.json", summary)
    (directory / "summary.txt").write_text(f"TDN bounded structural audit: {status}\nCases: {len(rows)}/{len(expected)}; required failures: {len(failed)}\nNo neural training. Rank and dictionary oracle diagnostics do not establish solver superiority.\n")
    return summary
