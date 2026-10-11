"""Paired finite-step interaction diagnostics, with separated target errors.

The leading cubic defect is a fixed-grid small-step expansion. This panel
measures its failure modes; it does not infer a stiff-order theorem from a
finite sweep. Every physical-parameter intervention retains the same field.
Three independently refined targets separate numerical temporal error,
FD-versus-Galerkin discretization error and unresolved-scale closure error.
"""
from __future__ import annotations

import itertools
import math
import time

import numpy as np
import torch

from tdn.analysis.advance.diagnostics import residual_projection
from tdn.analysis.advance.metrics import endpoint_metrics
from tdn.analysis.advance.numerics import cubic_commutator_defect
from tdn.analysis.agenda.physics import fourier_resample
from tdn.analysis.frontier.core import check, clean
from tdn.analysis.frontier.data import _temporal_teacher
from tdn.analysis.frontier import numerics as physical
from tdn.analysis.roadmap.numerics import quadratic_df_defect, cubic_df_defect
from tdn.numerics import Equation, Geometry
from tdn.research.experiment import state_digest
from tdn.research.protocol import digest
from tdn.runtime.metadata import write_json


REGIMES = ("overlapping_triads", "aligned_harmonics", "grid_relative")
METHODS = ("df", "etdrk4", "gl2", "gl4", "gl2_raw_cubic",
           "gl2_filtered_cubic", "gl4_nested_cubic")


def field_declaration(n, regime, phase, *, seed=85210000, mean=.4, amplitude=.04):
    """Freeze a trigonometric polynomial, including its continuous RMS scale.

    The first two families use fixed physical frequencies. The third explicitly
    changes its frequencies with the *declared coarse grid*, then retains those
    frequencies on every reference refinement. The harmonic family challenges
    identifiability but is never presumed to yield collinear response vectors.
    """
    if type(n) is not int or n < 8 or n % 2 or regime not in REGIMES:
        raise ValueError("Interaction fields require an even grid >= 8 and a declared regime")
    if not all(math.isfinite(float(v)) for v in (phase, mean, amplitude)) or not 0 < mean < 1 or amplitude <= 0:
        raise ValueError("Finite interior mean, positive amplitude and finite phase required")
    k = n // 2 - 1
    modes = {
        "overlapping_triads": [(1, 0), (0, 1), (1, 1), (2, 1)],
        "aligned_harmonics": [(1, 0), (2, 0), (3, 0), (1, 1)],
        "grid_relative": [(k, 0), (k - 1, 1), (1, -1), (1, 0)],
    }[regime]
    # An intentionally coherent harmonic family, with a phase intervention
    # that is not equivalent to a common spatial translation.
    rng = np.random.default_rng(int(seed) + REGIMES.index(regime))
    phases = [0.] * len(modes) if regime == "aligned_harmonics" else rng.uniform(0, 2 * math.pi, len(modes))
    weights = [1., .8, .6, .25] if regime == "aligned_harmonics" else [1., .8, .65, .45]
    terms = [dict(mode=list(mode), weight=weight, phase=float(base + phase * (j + 1)**2))
             for j, (mode, weight, base) in enumerate(zip(modes, weights, phases))]
    normalization = math.sqrt(sum(w * w for w in weights) / 2)
    if amplitude * sum(abs(w) for w in weights) / normalization >= min(mean, 1 - mean):
        raise ValueError("Declared continuous field may leave [0,1]; reduce amplitude")
    declaration = dict(regime=regime, terms=terms, mean=float(mean), amplitude=float(amplitude),
                       normalization=normalization, phase=float(phase), seed=int(seed),
                       declared_coarse_grid=n if regime == "grid_relative" else None,
                       frequency_scope="grid_relative_stress" if regime == "grid_relative" else "fixed_physical_modes",
                       identifiability_scope="measured Gram matrix; no a priori independence or collinearity claim")
    declaration["field_id"] = "resolve-interaction-" + digest(declaration)[:20]
    return declaration


def field_state(declaration, n):
    """Evaluate the *same* continuous field on a requested refinement."""
    if type(n) is not int or n < 8 or n % 2:
        raise ValueError("An even field grid >= 8 is required")
    if any(max(abs(k) for k in term["mode"]) >= n / 2 for term in declaration["terms"]):
        raise ValueError("Field modes must remain strictly below Nyquist")
    axis = torch.arange(n, dtype=torch.float64) / n
    x, y = torch.meshgrid(axis, axis, indexing="ij")
    value = sum(term["weight"] * torch.cos(2 * math.pi * (term["mode"][0] * x + term["mode"][1] * y) + term["phase"])
                for term in declaration["terms"])
    return (declaration["mean"] + declaration["amplitude"] * value / declaration["normalization"])[None, None]


def active_rate(u, geometry, track):
    """State-weighted decay rate for unit diffusivity, with physical units."""
    v = u - u.mean((-2, -1), keepdim=True)
    symbol = physical.diffusion_symbol(u, Equation(1., 0.), geometry, track)
    av = physical.apply_multiplier(v, symbol)
    return float(-(v * av).mean() / v.square().mean())


def physical_parameters(u, h, geometry, track, diffusion_number, reaction_number):
    """Independently set h*lambda_active and h*r without altering the state."""
    if not math.isfinite(h) or h <= 0 or any(not math.isfinite(float(v)) or v < 0
                                           for v in (diffusion_number, reaction_number)):
        raise ValueError("Positive horizon and finite nonnegative stiffness numbers required")
    rate = active_rate(u, geometry, track)
    if rate <= 0 or not math.isfinite(rate):
        raise ValueError("A nonconstant field with positive active decay rate is required")
    equation = Equation(float(diffusion_number) / (h * rate), float(reaction_number) / h)
    return equation, dict(diffusion_number=float(diffusion_number), reaction_number=float(reaction_number),
                         active_rate_unit_kappa=rate, kappa=equation.kappa, reaction_rate=equation.reaction_rate,
                         parameter_intervention="independent diffusivity and reaction; identical continuous field within each sweep",
                         diffusion_normalization_scope="active rate of the declared grid and track; matched dimensionless stress does not imply identical kappa across grids or tracks")


def _rms(value):
    return float(value.square().mean().sqrt())


def error_decomposition(prediction, same_grid, galerkin_grid, projected_fine):
    r"""Exact vector identity, including cross terms rather than adding norms.

    p-y_f = (p-y_d) + (y_d-y_g) + (y_g-y_f). Error cancellation is retained;
    the last two terms are reference differences, not independent errors.
    """
    if any(v.shape != prediction.shape for v in (same_grid, galerkin_grid, projected_fine)):
        raise ValueError("Error decomposition requires matching fields")
    components = dict(temporal=prediction - same_grid, discretization=same_grid - galerkin_grid,
                      closure=galerkin_grid - projected_fine)
    total = prediction - projected_fine
    norm2 = {key: float(value.square().mean()) for key, value in components.items()}
    cross = {f"{a}:{b}": float((components[a] * components[b]).mean())
             for a, b in itertools.combinations(components, 2)}
    reconstructed = sum(norm2.values()) + 2 * sum(cross.values())
    total2 = float(total.square().mean())
    return dict(component_rms={key: math.sqrt(value) for key, value in norm2.items()},
                component_signed_mean={key: float(value.mean()) for key, value in components.items()},
                component_max={key: float(value.abs().max()) for key, value in components.items()},
                pair_inner_products=cross, total_rms=math.sqrt(total2),
                reconstructed_total_squared_rms=reconstructed, squared_identity_residual=total2 - reconstructed,
                vector_identity_max=float((total - sum(components.values())).abs().max()),
                semantics="Temporal = method minus converged specified N-grid equation; discretization = specified N-grid minus N-Galerkin; closure = N-Galerkin minus projected refined Galerkin. Cross terms can cancel.")


def response_geometry(residual, components):
    """Signed alignment and Gram conditioning, including zero/duplicate bases."""
    names = list(components)
    values = list(components.values())
    matrix = torch.stack([value.flatten().double() for value in values], 1)
    gram = matrix.T @ matrix / matrix.shape[0]
    norms = torch.linalg.vector_norm(matrix, dim=0)
    nonzero = norms > 1e-30
    normalized = matrix / norms.clamp_min(1e-30)
    correlation = normalized.T @ normalized
    for i in range(len(names)):
        for j in range(len(names)):
            if not (nonzero[i] and nonzero[j]):
                correlation[i, j] = float("nan")
    pairs = {}
    for a, b in itertools.combinations(names, 2):
        pairs[f"{a}:{b}"] = residual_projection(residual, [components[a], components[b]])
    return clean(dict(basis_names=names, gram_mean_inner_products=gram.tolist(), normalized_gram=correlation.tolist(),
                      basis_rms={key: _rms(value) for key, value in components.items()}, pair_oracles=pairs,
                      projection_scope="same-field teacher-informed diagnostic, not a fitted deployable model"))


def refined_targets(declaration, n, h, equation, track, options, budget):
    """Actual three-level temporal and two-level spatial teacher refinements."""
    geometry = Geometry((n, n), (1., 1.))
    initial = field_state(declaration, n)
    started = time.perf_counter()
    same, same_meta = _temporal_teacher(initial, h, equation, geometry, track, options, budget)
    if track == "continuum":
        galerkin, galerkin_meta = same, same_meta
    else:
        galerkin, galerkin_meta = _temporal_teacher(initial, h, equation, geometry, "continuum", options, budget)
    factors = options["spatial_factors"]
    if len(factors) != 2 or factors[0] < 2 or factors[1] != 2 * factors[0]:
        raise ValueError("Two doubled fine grids, starting at least at 2N, are required")
    fine_values, fine_meta, projected_differences = [], [], []
    for factor in factors:
        budget.check()
        fine_n = factor * n
        value, meta = _temporal_teacher(field_state(declaration, fine_n), h, equation,
                                        Geometry((fine_n, fine_n), (1., 1.)), "continuum",
                                        {**options, "tolerance": options["tolerance"] / 4}, budget)
        fine_values.append(fourier_resample(value, (n, n)))
        projected_differences.append(fourier_resample(meta["_difference"], (n, n)))
        fine_meta.append(meta)
    spatial = fine_values[1] - fine_values[0]
    floor = max(options["roundoff_floor"], 128 * torch.finfo(torch.float64).eps)
    uncertain_rms = _rms(spatial) + sum(max(_rms(d), floor) for d in projected_differences)
    uncertain_max = float(spatial.abs().max()) + sum(max(float(d.abs().max()), floor) for d in projected_differences)
    continuum_meta = dict(accepted=all(v["accepted"] for v in fine_meta) and max(uncertain_rms, uncertain_max) <= options["tolerance"],
                          uncertainty_rms=uncertain_rms, uncertainty_max_bound=uncertain_max,
                          spatial_reference_grids=[factor * n for factor in factors],
                          projected_spatial_difference_rms=_rms(spatial), projected_spatial_difference_max=float(spatial.abs().max()),
                          temporal_refinements=[{k: v for k, v in row.items() if k != "_difference"} for row in fine_meta],
                          uncertainty_combination="projected spatial difference plus both projected temporal differences and roundoff floors")
    metadata = dict(same_grid={k: v for k, v in same_meta.items() if k != "_difference"},
                    galerkin_grid={k: v for k, v in galerkin_meta.items() if k != "_difference"},
                    projected_fine=continuum_meta, reference_seconds=time.perf_counter() - started,
                    reference_method="independently implemented CPU FP64 Lawson RK4 with actual refinement",
                    uncertainty_is_certificate=False)
    return dict(same_grid=same, galerkin_grid=galerkin, projected_fine=fine_values[-1]), clean(metadata)


def predictions(u, h, equation, geometry, track, *, nested_nodes=6, reaction_substeps=16, budget=None, errors=None):
    """Analytic methods at identical endpoint opportunity and full output."""
    check_budget = budget.check if budget is not None else lambda: None
    calls = {
        "base": lambda: physical.df_step(u, h, equation, geometry, track, reaction_substeps=reaction_substeps),
        "q2": lambda: quadratic_df_defect(u, h, equation, geometry, nodes=2, target=track),
        "q4": lambda: quadratic_df_defect(u, h, equation, geometry, nodes=4, target=track),
        "q8": lambda: quadratic_df_defect(u, h, equation, geometry, nodes=8, target=track),
        "q16": lambda: quadratic_df_defect(u, h, equation, geometry, nodes=16, target=track),
        "raw_c3": lambda: cubic_commutator_defect(u, h, equation, geometry, track=track, transport="none"),
        "filtered_c3": lambda: cubic_commutator_defect(u, h, equation, geometry, track=track, transport="symmetric_heat"),
        "nested_c3": lambda: cubic_df_defect(u, h, equation, geometry, nodes=nested_nodes, target=track),
        "etdrk4": lambda: physical.etdrk4_step(u, h, equation, geometry, track),
    }
    values, timings = {}, {}
    for name, call in calls.items():
        check_budget()
        started = time.perf_counter()
        try:
            values[name] = call()
        except (RuntimeError, FloatingPointError, OverflowError) as error:
            if errors is None:
                raise
            errors[name] = f"{type(error).__name__}: {error}"
            values[name] = torch.full_like(u, float("nan"))
        timings[name] = time.perf_counter() - started
    result = dict(df=values["base"], etdrk4=values["etdrk4"], gl2=values["base"] + values["q2"],
                  gl4=values["base"] + values["q4"],
                  gl2_raw_cubic=values["base"] + values["q2"] + values["raw_c3"],
                  gl2_filtered_cubic=values["base"] + values["q2"] + values["filtered_c3"],
                  gl4_nested_cubic=values["base"] + values["q4"] + values["nested_c3"])
    return result, values, timings


def _options(config):
    ref = config.get("reference", {})
    result = dict(substeps=int(ref.get("substeps", 4)), max_substeps=int(ref.get("max_substeps", 128)),
                  tolerance=float(ref.get("tolerance", 1e-8)), roundoff_floor=float(ref.get("roundoff_floor", 1e-12)),
                  spatial_factors=list(ref.get("spatial_factors", [2, 4])))
    if result["substeps"] < 1 or result["max_substeps"] < 4 * result["substeps"] or result["tolerance"] <= 0:
        raise ValueError("Reference budget must support actual three-level refinement")
    return result


def _measured_metrics(answer, target, initial, equation, geometry, track, reference):
    """A failed teacher yields missing accuracy evidence, never a fake target."""
    if not bool(torch.isfinite(target).all()):
        return dict(finite=bool(torch.isfinite(answer).all()), reference_accepted=False,
                    reference_status="NONFINITE_REFERENCE", error_rms=None, error_max=None,
                    signed_mean_error=None, spectral_bands=[],
                    reason="Reference failed; no finite target was substituted")
    return endpoint_metrics(answer, target, initial, equation, geometry, track, reference=reference)


def _comparison(control, candidate, reference):
    """Raw ratios and uncertainty-separated observations, never a certificate."""
    a, b = control.get("error_rms"), candidate.get("error_rms")
    valid = bool(reference["accepted"] and a is not None and b is not None)
    uncertainty = reference.get("uncertainty_rms")
    return dict(control_over_candidate_rms=a / b if a is not None and b is not None and b > 0 else None,
                higher_ratio_is_better=True,
                uncertainty_separated_improvement=(a - uncertainty > b + uncertainty if valid else None),
                status="MEASURED_DEVELOPMENT" if valid else "NA_UNRESOLVED_REFERENCE_OR_METHOD",
                uncertainty_is_certificate=False)


def run(ctx):
    """Run one bounded track/grid development panel and retain every result."""
    config = ctx.protocol["resolve"]["interactions"]
    unit = ctx.unit
    track, n = unit["track"], int(unit["grid"])
    if track not in ("discrete", "continuum"):
        raise ValueError("Interaction unit must declare its equation track")
    h = float(config.get("horizon", .12))
    regimes = [unit["regime"]] if "regime" in unit else config.get("regimes", list(REGIMES))
    phases = [unit["phase"]] if "phase" in unit else config.get("phases", [0., .7])
    diffusion_numbers = config.get("diffusion_numbers", [.1, 1., 5.])
    reaction_numbers = config.get("reaction_numbers", [.1, 1., 3.])
    options = _options(config)
    geometry = Geometry((n, n), (1., 1.))
    rows, cases = [], []
    for regime, phase in itertools.product(regimes, phases):
        declaration = field_declaration(n, regime, float(phase), seed=int(config.get("seed", 85210000)),
                                        mean=float(config.get("mean", .4)), amplitude=float(config.get("amplitude", .04)))
        u = field_state(declaration, n)
        for diffusion_number, reaction_number in itertools.product(diffusion_numbers, reaction_numbers):
            ctx.budget.check()
            equation, parameters = physical_parameters(u, h, geometry, track, diffusion_number, reaction_number)
            case_id = f"{declaration['field_id']}/d{diffusion_number:g}/r{reaction_number:g}"
            targets, references = refined_targets(declaration, n, h, equation, track, options, ctx.budget)
            component_errors = {}
            with torch.no_grad():
                outputs, components, timings = predictions(u, h, equation, geometry, track,
                    nested_nodes=int(config.get("nested_nodes", 6)), reaction_substeps=int(config.get("reaction_substeps", 16)),
                    budget=ctx.budget, errors=component_errors)
            direction = (response_geometry(targets["same_grid"] - components["base"],
                        {name: components[name] for name in ("q2", "q4", "raw_c3", "filtered_c3", "nested_c3")})
                        if all(bool(torch.isfinite(v).all()) for v in (targets["same_grid"],
                            *(components[name] for name in ("base", "q2", "q4", "raw_c3", "filtered_c3", "nested_c3"))))
                        else dict(status="NA_NONFINITE_REFERENCE_OR_COMPONENT"))
            case = dict(case_id=case_id, field=declaration, field_state_sha256=state_digest(u), track=track, grid=n,
                        horizon=h, **parameters, references=references, response_geometry=direction,
                        quadratic_quadrature_differences_rms={"GL2_minus_GL4": _rms(components["q2"] - components["q4"]),
                            "GL4_minus_GL8": _rms(components["q4"] - components["q8"]),
                            "GL8_minus_GL16": _rms(components["q8"] - components["q16"])},
                        component_generation_seconds=timings,
                        component_errors=component_errors,
                        timing_scope="single CPU diagnostic calls with shared bases; not paired deployment latency",
                        all_references_accepted=all(references[key]["accepted"] for key in targets))
            cases.append(case)
            df_metrics = _measured_metrics(outputs["df"], targets["same_grid"], u, equation, geometry, track,
                                            references["same_grid"])
            gl4_metrics = _measured_metrics(outputs["gl4"], targets["same_grid"], u, equation, geometry, track,
                                             references["same_grid"])
            for method, answer in outputs.items():
                metrics = _measured_metrics(answer, targets["same_grid"], u, equation, geometry, track,
                                             references["same_grid"])
                fine_metrics = _measured_metrics(answer, targets["projected_fine"], u, equation, geometry, "continuum",
                                                  references["projected_fine"])
                decomposition = (error_decomposition(answer, targets["same_grid"], targets["galerkin_grid"], targets["projected_fine"])
                    if all(bool(torch.isfinite(v).all()) for v in (answer, *targets.values())) else
                    dict(component_rms=None, pair_inner_products=None, total_rms=None,
                         vector_identity_max=None, squared_identity_residual=None,
                         status="NA_NONFINITE_REFERENCE_OR_METHOD"))
                row = dict(case_id=case_id, field_id=declaration["field_id"], method=method,
                           ownership="Ours numerical" if "raw_cubic" in method or "filtered_cubic" in method else "Theirs" if method in ("df", "etdrk4") else "Analytic control",
                           track=track, grid=n, horizon=h, **parameters,
                           same_grid_metrics=metrics, projected_fine_metrics=fine_metrics,
                           comparisons_to={"df": _comparison(df_metrics, metrics, references["same_grid"]),
                                           "gl4": _comparison(gl4_metrics, metrics, references["same_grid"])},
                           error_decomposition=decomposition, all_references_accepted=case["all_references_accepted"],
                           error_norm_scope="RMS and maxima on sampled nodal values; not a Nyquist-weighted continuous physical L2 norm",
                           reaction_substeps=int(config.get("reaction_substeps", 16)),
                           nested_quadrature_nodes=int(config.get("nested_nodes", 6)), output_compression=False,
                           endpoint_schedule_steps=1, hypothesis="R2", scope="paired development diagnostic; no held-out confirmation or stiff-order claim")
                row["component_failures_in_case"] = component_errors
                rows.append(clean(row))
                accepted = references["same_grid"]["accepted"]
                improvement = (df_metrics["error_rms"] - references["same_grid"]["uncertainty_rms"] >=
                               metrics["error_rms"] + references["same_grid"]["uncertainty_rms"]
                               if method != "df" and accepted and metrics["finite"] and df_metrics["finite"] else None)
                # Accuracy versus DF is a descriptive gap check, not the
                # advance criterion of beating the strongest analytic control.
                ctx.record(f"{case_id}/{method}", ["R2"], metrics=row, config=config,
                    checks=[check("finite-endpoint", metrics["finite"], True, "eq", category="gap",
                                  reason="Nonfinite method outcomes are scientific failures on the declared stress case"),
                            check("error-vector-identity", decomposition["vector_identity_max"], 2e-14, category="math"),
                            check("same-grid-reference-accepted", True if accepted else None, True, "eq", category="math",
                                  reason="Unresolved teacher remains unavailable mathematical evidence"),
                            check("separated-target-references", True if case["all_references_accepted"] else None, True, "eq", category="math",
                                  reason="A target decomposition needs all three accepted references"),
                            check("uncertainty-separated-rms-improvement-over-DF", improvement, True, "eq", category="gap",
                                  reason="Development diagnostic versus DF only; unresolved reference comparisons remain NA"),
                            check("stiff-order-or-deployment-advantage", None, None, category="utility",
                                  reason="No theorem or comparable deployment timing is inferred from this panel")])
            # Completed paired cases survive interruption, including failures
            # to meet reference tolerances. Original fields are not altered.
            write_json(ctx.path / "interaction_rows.json", clean(dict(rows=rows)))
            write_json(ctx.path / "interaction_cases.json", clean(dict(cases=cases)))
    summary = dict(cases=len(cases), endpoint_rows=len(rows), methods=list(METHODS),
                   accepted_cases=sum(case["all_references_accepted"] for case in cases),
                   unresolved_cases=sum(not case["all_references_accepted"] for case in cases),
                   distinct_fields=len({case["field"]["field_id"] for case in cases}),
                   independent_sampling_claim=False,
                   field_sampling_scope="deterministic phase interventions within declared spectral templates; not an IID population sample",
                   parameter_pairs_are_independent_fields=False, no_confirmation_access=True,
                   finite_step_scope="separated temporal/discretization/closure development evidence; no stiff-order claim",
                   fixed_physical_and_grid_relative_panels_are_distinct=True,
                   uncertainty_is_certificate=False)
    write_json(ctx.path / "interaction_summary.json", summary)
    return summary
