"""Small fixed field bank with independent references and complete inference costs.

All cases and variants are declared before computation. Cases are deterministic
diagnostics, not independent statistical replications. There is no fitting,
selection, gain search, teacher access in a method, or generating-pair input to
the interaction branch. The mean ablations retain and report logistic DC drift.
"""
from __future__ import annotations

from collections import Counter
import itertools
import math
import time

import torch

from tdn.numerics.reference import choose_substeps, refined_reference
from tdn.numerics.splitting import split_step
from tdn.numerics.types import Equation, Geometry
from tdn.research.interaction import (interaction_step, interaction_defect,
    etdrk2_step, etdrk4_step, scalar_defect, output_phi_defect)
from tdn.runtime.metadata import write_json
from .protocol import PANELS, validate_rows

INTERACTIONS = tuple(f"gl{nodes}_{coordinate}" for nodes in (3, 5)
                     for coordinate in ("additive", "capacity", "projected")) + (
                         "gl3_zero_mean", "gl3_mean_only")
VARIANTS = ("strang", "etdrk2", "etdrk4", "scalar_cubic", "output_phi") + INTERACTIONS
WORK_KEYS = ("fft_forward", "fft_inverse", "quadrature_evaluations", "reaction_evaluations",
             "reaction_jacobian_evaluations", "reaction_weight_evaluations", "nonlinear_evaluations",
             "laplacian_evaluations")
LIMITS = ("zero_h", "zero_diffusion", "zero_reaction", "uniform_0", "uniform_mid", "uniform_1")


def _spec(case_id, *, grid=(16,), kappa=.03, rate=2., h=.12, mean=.5,
          pattern="mixed", amplitude=.25, steps=1):
    return dict(case_id=case_id, grid=list(grid), lengths=[1.] * len(grid), kappa=kappa,
                reaction_rate=rate, h=h, mean=mean, pattern=pattern, amplitude=amplitude,
                steps=steps, final_time=h * steps)


def cases():
    """Rates/time/grid form a full 2x2x2x2 crossing; means/patterns are declared."""
    bank = []
    for index, (n, kappa, rate, h) in enumerate(itertools.product(
            (16, 32), (.003, .03), (.5, 4.), (.03, .2))):
        bank.append(_spec(f"cross/{index:02d}", grid=(n,), kappa=kappa, rate=rate,
                          h=h, mean=.5, pattern="mixed", amplitude=.14))
    for pattern in ("signed_1_minus3", "signed_2_minus5"):
        bank.append(_spec(f"signed/{pattern}", pattern=pattern, amplitude=.18))
    bank.append(_spec("historical/phi_counterexample", kappa=.02, rate=6., h=.4,
                      mean=.3, amplitude=.14, pattern="signed_3_minus1"))
    for index, mean in enumerate((.2, .5, .8)):
        bank.append(_spec(f"means/{index}", mean=mean, amplitude=.18))
    for pattern in ("near_lower", "near_upper", "remote_3", "remote_5"):
        bank.append(_spec(f"boundary/{pattern}", pattern=pattern))
    for index, mean in enumerate((.3, .7)):
        bank.append(_spec(f"two_d/{index}", grid=(8, 8), mean=mean,
                          pattern="oblique_mixed", amplitude=.25))
    rollouts = [_spec("rollout/mixed", steps=5, h=.06, grid=(32,)),
                _spec("rollout/near_lower", steps=5, h=.06, pattern="near_lower"),
                _spec("rollout/remote", steps=5, h=.12, pattern="remote_5"),
                _spec("rollout/two_d", steps=5, h=.06, grid=(8, 8), pattern="oblique_mixed")]
    return bank, rollouts


def plan(config):
    bank, rollouts = cases()
    if config["smoke"]:
        smoke_ids = {"cross/00", "cross/15", "boundary/near_lower", "two_d/0"}
        bank = [case for case in bank if case["case_id"] in smoke_ids]
        rollouts = [case for case in rollouts if case["case_id"] == "rollout/mixed"]
    controls = [f"limits/{limit}/{variant}" for limit in LIMITS for variant in INTERACTIONS]
    controls += [f"precision/gl{nodes}/{index}" for nodes in (3, 5) for index in range(3)]
    result = {"controls": {"expected_case_ids": controls,
                           "exact_limits": list(LIMITS), "precision_h": [1e-5, .003, .2]}}
    for name, specs in (("one_step", bank), ("rollout", rollouts)):
        result[name] = {"expected_case_ids": [f"{case['case_id']}/{variant}"
                        for case in specs for variant in ("reference", *VARIANTS)],
                        "cases": specs, "variants": list(VARIANTS), "dtype": "float64",
                        "reference": "same-grid coupled RK4; three levels; at most three refinements"}
    return result


def state(spec, *, dtype=torch.float64):
    grid = tuple(spec["grid"])
    coordinates = torch.meshgrid(*(torch.arange(n, dtype=dtype) / n for n in grid), indexing="ij")
    x = coordinates[0]
    pattern = spec["pattern"]
    if pattern.startswith("remote_"):
        field = torch.full(grid, .4, dtype=dtype)
        field[int(pattern.rsplit("_", 1)[1])] = .7
    elif pattern in ("near_lower", "near_upper"):
        field = 1e-6 + .025 * (1 + torch.sin(2 * math.pi * x)) ** 2
        if pattern == "near_upper":
            field = 1 - field
    else:
        if pattern == "signed_1_minus3":
            mode = .65 * torch.cos(2 * math.pi * x) + .35 * torch.sin(-6 * math.pi * x + .37)
        elif pattern == "signed_2_minus5":
            mode = .55 * torch.cos(4 * math.pi * x + .2) + .45 * torch.sin(-10 * math.pi * x)
        elif pattern == "signed_3_minus1":
            mode = .5 * torch.cos(6 * math.pi * x) + .5 * torch.sin(-2 * math.pi * x)
        elif pattern == "oblique_mixed":
            y = coordinates[1]
            mode = .5 * torch.cos(2 * math.pi * (x + y)) + .3 * torch.sin(2 * math.pi * (2 * x - y))
            mode = mode + .2 * torch.cos(2 * math.pi * y + .4)
        else:
            mode = (.5 * torch.cos(2 * math.pi * x) + .3 * torch.sin(-4 * math.pi * x + .1)
                    + .2 * torch.cos(10 * math.pi * x + .4))
        field = spec["mean"] + spec["amplitude"] * mode
    return field.reshape(1, 1, *grid)


def _row(case_id, panel, variant, kind, outcome, metrics, inputs, note, *, test=None):
    # JSON cannot preserve NaN/Inf. Nulls explicitly accompany failure reasons.
    metrics = {key: (None if isinstance(value, float) and not math.isfinite(value) else value)
               for key, value in metrics.items()}
    return dict(case_id=case_id, panel=panel, mechanism="field_interaction_response",
                test=test or panel, variant=variant, kind=kind, outcome=outcome,
                metrics=metrics, inputs=inputs, note=note)


def method_step(u, h, equation, geometry, variant, *, work):
    if variant == "strang":
        for key, count in (("fft_forward", int(equation.kappa != 0)),
                           ("fft_inverse", int(equation.kappa != 0)),
                           ("reaction_evaluations", 2 * int(equation.reaction_rate != 0))):
            work[key] = work.get(key, 0) + count
        return split_step(u, h, equation, geometry, differentiable=False)
    if variant in ("etdrk2", "etdrk4"):
        return {"etdrk2": etdrk2_step, "etdrk4": etdrk4_step}[variant](u, h, equation, geometry, work=work)
    if variant in ("scalar_cubic", "output_phi"):
        base = method_step(u, h, equation, geometry, "strang", work=work)
        function = scalar_defect if variant == "scalar_cubic" else output_phi_defect
        return base + function(u, h, equation, geometry, work=work)
    nodes, coordinate = variant.split("_", 1)
    mean_mode = coordinate if coordinate in ("zero_mean", "mean_only") else "full"
    return interaction_step(u, h, equation, geometry, nodes=int(nodes[2:]),
                            coordinate="additive" if mean_mode != "full" else coordinate,
                            mean_mode=mean_mode, work=work)


def _teacher(u, spec, equation, geometry, config, check):
    count = max(4, choose_substeps(spec["final_time"], equation, geometry))
    rhs_calls = 0
    started = time.perf_counter()
    for attempt in range(config["reference_attempts"]):
        check()
        reference = refined_reference(u, spec["final_time"], equation, geometry, count,
                                      tolerance=config["reference_tolerance"])
        rhs_calls += 28 * count
        check()
        if reference.accepted:
            break
        count *= 2
    return reference, {"reference_accepted": reference.accepted,
        "uncertainty_l2": reference.uncertainty, "split_defect_l2": reference.defect_norm,
        "reference_n": reference.refinement_substeps[0],
        "reference_2n": reference.refinement_substeps[1], "reference_4n": reference.refinement_substeps[2],
        "difference_n_2n": reference.refinement_differences[0],
        "difference_2n_4n": reference.refinement_differences[1],
        "observed_order": reference.observed_order, "reference_reason": reference.reason,
        "attempts": attempt + 1, "rhs_evaluations": rhs_calls,
        "reference_seconds": time.perf_counter() - started}


def _controls(rows, check):
    geometry = Geometry((16,), (1.,))
    original = state(_spec("control"))
    for limit in LIMITS:
        u = original
        h, kappa, rate = (.0 if limit == "zero_h" else .2), .03, 2.
        if limit == "zero_diffusion":
            kappa = 0.
        if limit == "zero_reaction":
            rate = 0.
        if limit.startswith("uniform_"):
            u = torch.full_like(u, {"uniform_0": 0., "uniform_mid": .27, "uniform_1": 1.}[limit])
        equation = Equation(kappa, rate)
        base = split_step(u, h, equation, geometry)
        for variant in INTERACTIONS:
            check()
            answer = method_step(u, h, equation, geometry, variant, work={})
            error = float((answer - base).abs().max())
            rows.append(_row(f"limits/{limit}/{variant}", "controls", variant, "correctness",
                "PASS" if error <= 2e-14 else "FAIL", {"max_difference_from_split": error},
                {"limit": limit, "h": h, "kappa": kappa, "reaction_rate": rate},
                "Field-derived correction vanishes in the declared exact split limit.", test="exact_limits"))
    for nodes in (3, 5):
        for index, h in enumerate((1e-5, .003, .2)):
            check()
            equation = Equation(.03, 2.)
            fp64 = interaction_defect(original, h, equation, geometry, nodes=nodes)
            fp32 = interaction_defect(original.float(), h, equation, geometry, nodes=nodes).double()
            error = float((fp64 - fp32).square().mean().sqrt())
            amplitude = float(fp64.square().mean().sqrt())
            rows.append(_row(f"precision/gl{nodes}/{index}", "controls", f"gl{nodes}", "representation",
                "OBSERVED", {"fp32_fp64_l2": error, "fp64_defect_l2": amplitude,
                             "relative_defect_error": error / amplitude if amplitude else None},
                {"h": h, "grid": [16], "kappa": .03, "reaction_rate": 2.},
                "Complete anchored defect evaluation, including small-step cancellation; no speed claim.",
                test="precision_cancellation"))


def _case(rows, spec, panel, config, check):
    u = state(spec)
    geometry = Geometry(tuple(spec["grid"]), tuple(spec["lengths"]))
    equation = Equation(spec["kappa"], spec["reaction_rate"])
    reference, reference_metrics = _teacher(u, spec, equation, geometry, config, check)
    inputs = {**spec, "initial_actual_mean": float(u.mean()), "device": "cpu", "dtype": "float64"}
    rows.append(_row(f"{spec['case_id']}/reference", panel, "reference", "scientific",
        "OBSERVED" if reference.accepted else "INCONCLUSIVE", reference_metrics, inputs,
        "Independent coupled central-difference RK4 at n,2n,4n on this same grid. Full last difference is uncertainty."))
    split_error = None
    for variant in VARIANTS:
        check()
        work = {}
        value = u.clone()
        step_times = []
        completed_steps = 0
        failure = None
        try:
            for _ in range(spec["steps"]):
                check()
                started = time.perf_counter()
                try:
                    value = method_step(value, spec["h"], equation, geometry, variant, work=work)
                finally:
                    step_times.append(time.perf_counter() - started)
                completed_steps += 1
                if not bool(torch.isfinite(value).all()) or not bool(((value >= -2e-14) & (value <= 1 + 2e-14)).all()):
                    break
            finite = bool(torch.isfinite(value).all())
            bounded = finite and bool(((value >= -2e-14) & (value <= 1 + 2e-14)).all())
            if completed_steps == spec["steps"]:
                error = value - reference.state
                l2 = float(error.square().mean().sqrt())
                maximum = float(error.abs().max())
                mean_error = float(error.mean())
            else:
                l2 = maximum = mean_error = math.inf
        except (ScreenIncomplete, KeyboardInterrupt, InterruptedError):
            raise
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            finite, bounded, l2, maximum, mean_error = False, False, math.inf, math.inf, math.inf
        if variant == "strang":
            split_error = l2
        resolved = bool(reference.accepted and finite and bounded and completed_steps == spec["steps"] and math.isfinite(split_error)
                        and abs(split_error - l2) > 2 * reference.uncertainty)
        metrics = {"error_l2": l2, "error_max": maximum, "mean_error": mean_error,
            "output_mean": float(value.mean()) if not failure else None,
            "minimum": float(value.min()) if not failure else None,
            "maximum": float(value.max()) if not failure else None,
            "finite": finite, "in_bounds": bounded, "reference_accepted": reference.accepted,
            "error_ratio_to_strang": l2 / split_error if split_error and math.isfinite(split_error) else None,
            "comparison_resolved": resolved, "improves_strang_resolved": bool(resolved and l2 < split_error),
            "first_step_seconds": step_times[0] if step_times else None,
            "complete_rollout_seconds": sum(step_times), "completed_steps": completed_steps,
            "trajectory_completed": completed_steps == spec["steps"] and failure is None,
            "work_counts_complete": failure is None,
            **{f"work_{key}": work.get(key, 0) for key in WORK_KEYS}}
        outcome = ("FAIL" if failure else "INCONCLUSIVE" if not reference.accepted else
                   "EXPECTED_LIMITATION" if not finite or not bounded else "OBSERVED")
        note = ("Full step/rollout, uncached coefficients, one sequential timing sample; no tolerance certificate. "
                "L2 is volume-normalized. Mean error is signed; logistic mean is not conserved. "
                "Projection is explicitly a clipping diagnostic; capacity is a distinct bounded directional map.")
        if failure:
            note += f" Method failed and was retained: {failure}. Failed-call elapsed time is retained; partial failed-call counters may be unavailable."
        elif completed_steps < spec["steps"]:
            note += " Rollout stopped at its first invalid state; final-horizon errors are null and all attempted work is retained."
        rows.append(_row(f"{spec['case_id']}/{variant}", panel, variant, "scientific", outcome,
                         metrics, inputs, note))


class ScreenIncomplete(RuntimeError):
    """Safe-boundary stop or exhausted numerical budget."""


def _publish(protocol, run_dir, rows, status, errors, elapsed):
    plans = protocol["plans"]
    reported = {row["case_id"] for row in rows}
    expected = [case for name in PANELS for case in plans[name]["expected_case_ids"]]
    unreported = [case for case in expected if case not in reported]
    if status == "COMPLETED" and unreported:
        status = "INCOMPLETE"
    inconclusive = any(row["outcome"] == "INCONCLUSIVE" for row in rows)
    scientific = "INCONCLUSIVE" if status != "COMPLETED" or inconclusive else "OBSERVED_MIXED"
    canonical = {"schema": "tdn.interaction-screen/v1", "benchmark_suite": "interaction-screen",
        "status": status, "computational_status": status, "scientific_outcome": scientific,
        "device": "cpu", "training_attempted": False, "training_performed": False,
        "reference_scope": "FP64 coupled three-level RK4 on the same semidiscrete grid; full final refinement difference",
        "timing_scope": "One sequential sample of complete uncached CPU method steps; rollout sums all method calls; excludes reference/report costs",
        "work_scope": "Actual successful-call counters accumulated across steps; split charges each active FFT pair and exact reaction; validation is timed; failed-call partial counters are unavailable when work_counts_complete is false",
        "elapsed_seconds": elapsed, "numerical_budget_seconds": protocol["config"]["max_seconds"],
        "panel_states": {name: ("COMPLETED" if all(case in reported for case in plans[name]["expected_case_ids"])
                                 else "INCOMPLETE") for name in PANELS},
        "rows": rows, "errors": errors, "unreported_case_ids": unreported,
        "correctness_failures": [r["case_id"] for r in rows if r["kind"] == "correctness" and r["outcome"] == "FAIL"]}
    summary = {key: value for key, value in canonical.items() if key != "rows"}
    comparisons = [r for r in rows if "improves_strang_resolved" in r["metrics"] and r["variant"] != "strang"]
    summary.update(expected_cases=len(expected), reported_cases=len(rows),
        scalar_metrics=sum(len(row["metrics"]) for row in rows),
        outcome_counts=dict(Counter(row["outcome"] for row in rows)),
        kind_counts=dict(Counter(row["kind"] for row in rows)),
        resolved_improvements=sum(r["metrics"]["improves_strang_resolved"] for r in comparisons),
        resolved_regressions=sum(r["metrics"]["comparison_resolved"] and not r["metrics"]["improves_strang_resolved"] for r in comparisons),
        scope=protocol["scope"], interpretation=protocol["interpretation"])
    write_json(run_dir / "interaction-screen.json", canonical)
    write_json(run_dir / "summary.json", summary)
    (run_dir / "summary.txt").write_text(
        f"TDN field-interaction CPU screen: computational {status}; scientific {scientific}\n"
        f"Reported cases: {len(rows)}/{len(expected)}; scalar metrics: {summary['scalar_metrics']}\n"
        f"Numerical wall time: {elapsed:.3f}s / {protocol['config']['max_seconds']}s\n"
        f"Outcomes: {summary['outcome_counts']}\n"
        "Every variant and negative result is retained. No training, GPU work, or fitted selection.\n"
        "Same-grid references and single CPU timings do not establish continuum accuracy or cost-to-tolerance superiority.\n")
    return summary


def run(protocol, run_dir, *, stop=None, progress=None, clock=time.monotonic):
    started = clock()
    rows, errors = [], []

    def check():
        if stop is not None and stop.requested:
            raise ScreenIncomplete("Stop requested; preserve this attempt and use a fresh run directory")
        if clock() - started >= protocol["config"]["max_seconds"]:
            raise ScreenIncomplete("Frozen numerical time budget exhausted")

    status = "COMPLETED"
    try:
        with torch.no_grad():
            _controls(rows, check)
            for name in ("one_step", "rollout"):
                for spec in protocol["plans"][name]["cases"]:
                    check()
                    _case(rows, spec, name, protocol["config"], check)
                    validate_rows(rows, protocol["plans"], complete=False)
                    _publish(protocol, run_dir, rows, "RUNNING", errors, clock() - started)
                    if progress:
                        progress(name, {"status": "RUNNING", "rows": rows})
        check()
        validate_rows(rows, protocol["plans"], complete=True)
        if any(row["outcome"] == "FAIL" for row in rows):
            status = "FAILED"
    except (Exception, KeyboardInterrupt) as error:
        status = "INCOMPLETE" if isinstance(error, (ScreenIncomplete, KeyboardInterrupt, InterruptedError)) else "FAILED"
        errors.append(f"{type(error).__name__}: {error}")
    return _publish(protocol, run_dir, rows, status, errors, clock() - started)
