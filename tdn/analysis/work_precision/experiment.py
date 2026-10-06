"""Prepared fixed-horizon CPU candidates and explicitly post-hoc frontiers.

All fields, steps, timing orders and parity checks are frozen before execution.
Timing includes per-step state validation and stop checks, but excludes cloning,
reference construction, endpoint errors, parity checks and report publication.
"""
from __future__ import annotations

from collections import Counter
import math
import json
import os
import statistics
import time

import torch

from tdn.analysis.interactions.experiment import cases as historical_cases
from tdn.analysis.interactions.experiment import state as historical_state
from tdn.numerics.reference import choose_substeps, refined_reference
from tdn.numerics.types import Equation, Geometry
from tdn.research.interaction import interaction_defect
from tdn.research.work_precision import prepare_step, spectral_mean_defect
from tdn.runtime.metadata import write_json
from .protocol import (METHODS, NSTEPS, TARGETS, NORMS, TABLE_IDS, candidate_id,
                       frontier_id, validate_rows)


def cases():
    """20 fresh partial stress conditions and four exact historical controls."""
    bank = []
    for regime, (grid, kappa, rate, final_time) in enumerate(((24, .0025, .8, .24),
                                                            (40, .018, 3.2, .45))):
        for mean in (.32, .68):
            for amplitude in (.04, .16, .30):
                bank.append(dict(case_id=f"fresh/1d/{len(bank):02d}", case_role="fresh",
                    grid=[grid], lengths=[1.], kappa=kappa, reaction_rate=rate,
                    final_time=final_time, mean=mean, amplitude=amplitude,
                    pattern="fresh_phased_1d", phase=.23 + .19 * regime,
                    design_regime=regime, historical_case_id=None))
    for regime, (grid, kappa, rate, final_time) in enumerate((((10, 12), .004, 1.3, .18),
                                                            ((12, 12), .024, 5., .36))):
        for mean in (.25, .75):
            for amplitude in (.05, .22):
                index = len(bank) - 12
                bank.append(dict(case_id=f"fresh/2d/{index:02d}", case_role="fresh",
                    grid=list(grid), lengths=[1., 1.], kappa=kappa, reaction_rate=rate,
                    final_time=final_time, mean=mean, amplitude=amplitude,
                    pattern="fresh_phased_2d", phase=.31 + .17 * regime,
                    design_regime=regime, historical_case_id=None))
    old_steps, old_rollouts = historical_cases()
    for old_id, new_id in (("two_d/0", "two_d_low"), ("two_d/1", "two_d_high"),
                           ("rollout/two_d", "two_d_rollout"),
                           ("historical/phi_counterexample", "phi_counterexample")):
        old = next(item for item in old_steps + old_rollouts if item["case_id"] == old_id)
        bank.append({**old, "case_id": f"historical/{new_id}", "case_role": "historical_control",
                     "historical_case_id": old_id, "design_regime": "historical"})
    return bank


def state(spec, *, dtype=torch.float64):
    if spec["case_role"] == "historical_control":
        return historical_state(spec, dtype=dtype)
    coordinates = torch.meshgrid(*(torch.arange(n, dtype=dtype) / n for n in spec["grid"]), indexing="ij")
    x, phase = coordinates[0], spec["phase"]
    if len(coordinates) == 1:
        mode = (.43 * torch.cos(4 * math.pi * x + phase)
                + .34 * torch.sin(-6 * math.pi * x + .71)
                + .23 * torch.cos(14 * math.pi * x - .29))
    else:
        y = coordinates[1]
        mode = (.41 * torch.cos(2 * math.pi * (2 * x + y) + phase)
                + .36 * torch.sin(2 * math.pi * (x - 2 * y) - .43)
                + .23 * torch.cos(2 * math.pi * (3 * x - y) + .61))
    return (spec["mean"] + spec["amplitude"] * mode).reshape(1, 1, *spec["grid"])


def _identity_id(case_id, dtype, nodes):
    return f"{case_id}/identity/{dtype}/gl{nodes}"


def _rollout_parity_id(case_id, nsteps):
    return f"{case_id}/rollout_mean/n{nsteps}"


def plan(config):
    bank = cases()
    if config["smoke"]:
        selected = {"fresh/1d/00", "fresh/2d/07", "historical/phi_counterexample", "historical/two_d_rollout"}
        bank = [case for case in bank if case["case_id"] in selected]
    expected = {name: [] for name in TABLE_IDS}
    for spec in bank:
        case = spec["case_id"]
        expected["reference_rows"].append(f"{case}/reference")
        expected["candidate_rows"].extend(candidate_id(case, variant, n) for n in NSTEPS for variant in METHODS)
        expected["frontier_rows"].extend(frontier_id(case, variant, norm, tol)
            for variant in METHODS for norm in NORMS for tol in TARGETS)
        expected["parity_rows"].extend(_identity_id(case, dtype, nodes)
            for dtype in ("float64", "float32") for nodes in (3, 5))
        expected["parity_rows"].extend(_rollout_parity_id(case, n) for n in NSTEPS)
    orders = []
    for case_index, spec in enumerate(bank):
        for step_index, nsteps in enumerate(NSTEPS):
            base = (case_index + step_index) % len(METHODS)
            rounds = []
            for repeat in range(config["timing_repeats"]):
                shift = (base + repeat + 1) % len(METHODS)
                rounds.append(list(METHODS[shift:] + METHODS[:shift]))
            orders.append({"case_id": spec["case_id"], "nsteps": nsteps,
                "preparation_order": list(METHODS[base:] + METHODS[:base]),
                "warmup_order": list(METHODS[base:] + METHODS[:base]), "measured_orders": rounds})
    return {"cases": bank, "expected_ids": expected, "timing_orders": orders,
            "parity": {"identity_nsteps": 4, "identity_nodes": [3, 5],
                       "dtype_absolute_tolerances": {"float64": 2e-12, "float32": 3e-7},
                       "rollout_absolute_tolerance": 2e-11}}


class ScreenIncomplete(RuntimeError):
    """Stopped at a safe boundary; this attempt is retained and never sealed."""


def _finite(value):
    return value if math.isfinite(value) else None


def _teacher(u, spec, equation, geometry, config, check):
    count = max(4, choose_substeps(spec["final_time"], equation, geometry))
    started, rhs_calls = time.perf_counter(), 0
    cells = math.prod(spec["grid"])
    # refined_reference accepts <= 0.05*tolerance. Divide by sqrt(N) so
    # BOTH uncertainty estimates are stricter than the fixed 1e-8 budget.
    tolerance = config["reference_tolerance"] / math.sqrt(cells)
    for attempt in range(config["reference_attempts"]):
        check()
        reference = refined_reference(u, spec["final_time"], equation, geometry, count,
                                      tolerance=tolerance)
        rhs_calls += 28 * count
        check()
        if reference.accepted:
            break
        count *= 2
    uncertainty_max = math.sqrt(cells) * reference.uncertainty
    accepted = bool(reference.accepted and reference.uncertainty < min(TARGETS)
                    and uncertainty_max < min(TARGETS))
    return reference, {"reference_id": f"{spec['case_id']}/reference", "case_id": spec["case_id"],
        "case_role": spec["case_role"], "reference_accepted": accepted,
        "uncertainty_rms": _finite(reference.uncertainty),
        "uncertainty_max_estimate": _finite(uncertainty_max),
        "reference_n": reference.refinement_substeps[0], "reference_2n": reference.refinement_substeps[1],
        "reference_4n": reference.refinement_substeps[2],
        "difference_n_2n": _finite(reference.refinement_differences[0]),
        "difference_2n_4n": _finite(reference.refinement_differences[1]),
        "observed_order": _finite(reference.observed_order) if reference.observed_order is not None else None,
        "reference_reason": reference.reason, "attempts": attempt + 1,
        "rhs_evaluations": rhs_calls, "reference_seconds": time.perf_counter() - started}


def _identity_parity(spec, u, equation, geometry, protocol, rows, check):
    declaration = protocol["plans"]["parity"]
    h = spec["final_time"] / declaration["identity_nsteps"]
    for dtype_name, dtype in (("float64", torch.float64), ("float32", torch.float32)):
        for nodes in declaration["identity_nodes"]:
            check()
            field = u.to(dtype)
            full = interaction_defect(field, h, equation, geometry, nodes=nodes)
            full_mean = full.mean(dim=tuple(range(2, full.ndim)), keepdim=True)
            cheap = spectral_mean_defect(field, h, equation, geometry, nodes=nodes)
            difference = full_mean.double() - cheap.double()
            maximum = float(difference.abs().max())
            allowed = declaration["dtype_absolute_tolerances"][dtype_name]
            passed = math.isfinite(maximum) and maximum <= allowed
            rows.append({"parity_id": _identity_id(spec["case_id"], dtype_name, nodes),
                "case_id": spec["case_id"], "check": "identity", "dtype": dtype_name,
                "nodes": nodes, "nsteps": declaration["identity_nsteps"],
                "max_difference": _finite(maximum),
                "rms_difference": _finite(float(difference.square().mean().sqrt())),
                "allowed_absolute_difference": allowed, "passed": passed,
                "status": "PASS" if passed else "FAIL", "failure_reason": None if passed else "spectral/full raw mean mismatch"})


def _rollout(prepared, u, nsteps, check, repeat_index, method_order_index):
    """Return partial work and the exception to rethrow after storing its row."""
    value = u.clone()
    work, completed_steps, error = {}, 0, None
    finite, bounded = True, True
    started = time.perf_counter()
    try:
        for _ in range(nsteps):
            check()
            value = prepared(value, work=work)
            completed_steps += 1
            finite = bool(torch.isfinite(value).all())
            allowance = 64 * torch.finfo(value.dtype).eps
            bounded = finite and bool(((value >= -allowance) & (value <= 1 + allowance)).all())
            if not finite or not bounded:
                break
    except (Exception, KeyboardInterrupt) as exc:
        error = exc
    seconds = time.perf_counter() - started
    complete = completed_steps == nsteps and error is None
    status = ("INCOMPLETE" if isinstance(error, (ScreenIncomplete, KeyboardInterrupt, InterruptedError))
              else "FAILED" if error is not None else "INVALID" if not finite or not bounded
              else "VALID" if complete else "INCOMPLETE")
    reason = (f"{type(error).__name__}: {error}" if error is not None else
              "nonfinite_state" if not finite else "out_of_bounds_state" if not bounded else None)
    sample = {"repeat_index": repeat_index, "method_order_index": method_order_index,
              "seconds": seconds, "status": status, "completed_steps": completed_steps,
              "trajectory_completed": complete, "finite": finite, "in_bounds": bounded,
              "work": work, "work_counts_complete": error is None, "failure_reason": reason}
    return sample, value, error


def _new_candidate(spec, variant, nsteps, refrow):
    return {"candidate_id": candidate_id(spec["case_id"], variant, nsteps),
        "case_id": spec["case_id"], "case_role": spec["case_role"], "variant": variant,
        "nsteps": nsteps, "h": spec["final_time"] / nsteps, "final_time": spec["final_time"],
        "grid": list(spec["grid"]), "dtype": "float64", "status": "INCOMPLETE",
        "trajectory_completed": False, "completed_steps": 0, "finite": True, "in_bounds": True,
        "error_rms": None, "error_max": None, "error_mean": None,
        "error_upper_rms": None, "error_upper_max_estimate": None,
        "reference_accepted": refrow["reference_accepted"],
        "reference_uncertainty_rms": refrow["uncertainty_rms"],
        "reference_uncertainty_max_estimate": refrow["uncertainty_max_estimate"],
        "preparation_seconds": None, "prepared_median_seconds": None,
        "prepared_min_seconds": None, "prepared_max_seconds": None,
        "setup_inclusive_median_seconds": None, "timing_repeats": [], "warmup": None,
        "work_per_rollout": {}, "cache_metadata": {}, "failure_reason": None}


def _refresh_candidate(row, value, reference, repeats):
    samples = ([row["warmup"]] if row["warmup"] is not None else []) + row["timing_repeats"]
    if not samples:
        return
    statuses = {sample["status"] for sample in samples}
    complete_sampling = row["warmup"] is not None and len(row["timing_repeats"]) == repeats
    row["completed_steps"] = min(sample["completed_steps"] for sample in samples)
    row["trajectory_completed"] = all(sample["trajectory_completed"] for sample in samples)
    row["finite"] = all(sample["finite"] for sample in samples)
    row["in_bounds"] = all(sample["in_bounds"] for sample in samples)
    row["status"] = ("FAILED" if "FAILED" in statuses else "INCOMPLETE" if "INCOMPLETE" in statuses
                     else "INVALID" if "INVALID" in statuses else "VALID" if complete_sampling else "INCOMPLETE")
    row["failure_reason"] = next((s["failure_reason"] for s in samples if s["failure_reason"]), None)
    durations = [sample["seconds"] for sample in row["timing_repeats"] if sample["status"] == "VALID"]
    # Partial/invalid timing samples remain raw and never define eligible costs.
    if complete_sampling and len(durations) == repeats and row["status"] == "VALID":
        row["prepared_median_seconds"] = statistics.median(durations)
        row["prepared_min_seconds"] = min(durations)
        row["prepared_max_seconds"] = max(durations)
        row["setup_inclusive_median_seconds"] = row["preparation_seconds"] + row["prepared_median_seconds"]
        row["work_per_rollout"] = dict(row["timing_repeats"][0]["work"])
    for field in ("error_rms", "error_max", "error_mean", "error_upper_rms", "error_upper_max_estimate"):
        row[field] = None
    if row["trajectory_completed"] and row["finite"]:
        error = value - reference.state
        row["error_rms"] = _finite(float(error.square().mean().sqrt()))
        row["error_max"] = _finite(float(error.abs().max()))
        row["error_mean"] = _finite(float(error.mean()))
        if row["reference_uncertainty_rms"] is not None and row["error_rms"] is not None:
            row["error_upper_rms"] = row["error_rms"] + row["reference_uncertainty_rms"]
        if row["reference_uncertainty_max_estimate"] is not None and row["error_max"] is not None:
            row["error_upper_max_estimate"] = row["error_max"] + row["reference_uncertainty_max_estimate"]


def _run_case(protocol, spec, tables, check):
    config = protocol["config"]
    u = state(spec)
    geometry, equation = Geometry(tuple(spec["grid"]), tuple(spec["lengths"])), Equation(spec["kappa"], spec["reaction_rate"])
    reference, refrow = _teacher(u, spec, equation, geometry, config, check)
    tables["reference_rows"].append(refrow)
    _identity_parity(spec, u, equation, geometry, protocol, tables["parity_rows"], check)
    for nsteps in NSTEPS:
        order = next(item for item in protocol["plans"]["timing_orders"]
                     if item["case_id"] == spec["case_id"] and item["nsteps"] == nsteps)
        prepared, rows, endpoints = {}, {}, {}
        for variant in order["preparation_order"]:
            check()
            row = _new_candidate(spec, variant, nsteps, refrow)
            rows[variant] = row
            tables["candidate_rows"].append(row)
            started = time.perf_counter()
            try:
                prepared[variant] = prepare_step(u, row["h"], equation, geometry, variant)
            except (Exception, KeyboardInterrupt) as error:
                row["status"] = "INCOMPLETE" if isinstance(error, (ScreenIncomplete, KeyboardInterrupt, InterruptedError)) else "FAILED"
                row["failure_reason"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                row["preparation_seconds"] = time.perf_counter() - started
            row["cache_metadata"] = prepared[variant].metadata
        # Every method finishes its complete warmup before ANY measured rollout.
        for repeat, variants in [(-1, order["warmup_order"]), *enumerate(order["measured_orders"])]:
            for position, variant in enumerate(variants):
                row = rows[variant]
                sample, value, error = _rollout(prepared[variant], u, nsteps, check, repeat, position)
                if repeat == -1:
                    row["warmup"] = sample
                else:
                    row["timing_repeats"].append(sample)
                _refresh_candidate(row, value, reference, config["timing_repeats"])
                endpoints[variant] = value
                if error is not None:
                    raise error
        pair = [rows[variant] for variant in ("gl3_mean_full", "gl3_mean_spectral")]
        available = all(row["status"] == "VALID" for row in pair)
        difference = endpoints["gl3_mean_full"] - endpoints["gl3_mean_spectral"] if available else None
        maximum = float(difference.abs().max()) if available else None
        allowed = protocol["plans"]["parity"]["rollout_absolute_tolerance"]
        passed = maximum <= allowed if available else None
        tables["parity_rows"].append({"parity_id": _rollout_parity_id(spec["case_id"], nsteps),
            "case_id": spec["case_id"], "check": "rollout_mean", "dtype": "float64", "nodes": 3,
            "nsteps": nsteps, "max_difference": maximum,
            "rms_difference": float(difference.square().mean().sqrt()) if available else None,
            "allowed_absolute_difference": allowed, "passed": passed,
            "status": "PASS" if passed else "FAIL" if available else "UNAVAILABLE",
            "failure_reason": None if passed else "spectral/full rollout mismatch" if available else "invalid or incomplete trajectory"})


def select_frontiers(protocol, candidate_rows, reference_rows):
    """Reference-informed post-hoc selection, never a runtime step controller."""
    references = {row["case_id"]: row for row in reference_rows}
    result = []
    for spec in protocol["plans"]["cases"]:
        case_id = spec["case_id"]
        reference = references.get(case_id)
        for variant in METHODS:
            candidates = [row for row in candidate_rows if row["case_id"] == case_id and row["variant"] == variant]
            coverage_complete = len(candidates) == len(NSTEPS) and all(row["status"] in {"VALID", "INVALID"} for row in candidates)
            for norm in NORMS:
                error_key = "error_rms" if norm == "rms" else "error_max"
                upper_key = "error_upper_rms" if norm == "rms" else "error_upper_max_estimate"
                uncertainty_key = "uncertainty_rms" if norm == "rms" else "uncertainty_max_estimate"
                for tolerance in TARGETS:
                    resolved = bool(reference and reference["reference_accepted"] and coverage_complete)
                    feasible = [row for row in candidates if row["status"] == "VALID" and row["reference_accepted"]
                        and row[upper_key] is not None and row[upper_key] <= tolerance
                        and row["prepared_median_seconds"] is not None] if resolved else []
                    fastest = min(feasible, key=lambda row: (row["prepared_median_seconds"], row["nsteps"])) if feasible else None
                    setup_fastest = min(feasible, key=lambda row: (row["setup_inclusive_median_seconds"], row["nsteps"])) if feasible else None
                    result.append({"frontier_id": frontier_id(case_id, variant, norm, tolerance),
                        "case_id": case_id, "case_role": spec["case_role"], "variant": variant,
                        "norm": norm, "tolerance": tolerance,
                        "status": "FEASIBLE" if fastest else "NO_FEASIBLE_CANDIDATE" if resolved else "INCONCLUSIVE",
                        "selected_candidate_id": fastest["candidate_id"] if fastest else None,
                        "selected_nsteps": fastest["nsteps"] if fastest else None,
                        "selected_error": fastest[error_key] if fastest else None,
                        "reference_uncertainty": reference[uncertainty_key] if reference else None,
                        "adjusted_error": fastest[upper_key] if fastest else None,
                        "prepared_seconds": fastest["prepared_median_seconds"] if fastest else None,
                        "setup_inclusive_seconds": fastest["setup_inclusive_median_seconds"] if fastest else None,
                        "setup_selected_candidate_id": setup_fastest["candidate_id"] if setup_fastest else None,
                        "setup_selected_nsteps": setup_fastest["nsteps"] if setup_fastest else None,
                        "setup_selected_seconds": setup_fastest["setup_inclusive_median_seconds"] if setup_fastest else None,
                        "setup_selected_error": setup_fastest[error_key] if setup_fastest else None,
                        "setup_selected_adjusted_error": setup_fastest[upper_key] if setup_fastest else None,
                        "feasible_candidate_count": len(feasible), "selection_scope": "posthoc_declared_grid"})
    return result


def write_canonical(path, payload):
    """Atomically retain all raw repeats within the bounded JSON export size."""
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w") as handle:
        json.dump(payload, handle, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _publish(protocol, run_dir, tables, status, errors, elapsed):
    tables["frontier_rows"] = select_frontiers(protocol, tables["candidate_rows"], tables["reference_rows"])
    missing = {name: [key for key in protocol["plans"]["expected_ids"][name]
                     if key not in {row[identity] for row in tables[name]}]
               for name, identity in TABLE_IDS.items()}
    if status == "COMPLETED" and any(missing.values()):
        status = "INCOMPLETE"
    correctness_failures = [row["parity_id"] for row in tables["parity_rows"] if row["status"] == "FAIL"]
    if correctness_failures and status == "COMPLETED":
        status = "FAILED"
    scientific = ("INCONCLUSIVE" if status != "COMPLETED" or
                  any(not row["reference_accepted"] for row in tables["reference_rows"]) else "OBSERVED_MIXED")
    canonical = {"schema": "tdn.work-precision/v1", "benchmark_suite": "work-precision",
        "status": status, "computational_status": status, "scientific_outcome": scientific,
        "device": "cpu", "dtype": "float64", "training_attempted": False, "training_performed": False,
        "reference_scope": "Same-grid coupled FP64 RK4 n/2n/4n; full last RMS difference and rounding allowance; sqrt(N)*RMS maximum estimate is not a certificate",
        "timing_scope": "Prepared full CPU rollouts: all-method warmup once, then five rotated/interleaved repeats; validation and stop checks timed; setup separately timed",
        "setup_inclusive_scope": "One separately timed preparation sample plus median warmed rollout; excludes cache-metadata reporting, reference, warmup, parity, cloning, endpoint errors, reports and post-hoc selection; not a physical cold-cache measurement",
        "selection_scope": "posthoc_declared_grid; reference-informed, not a deployable adaptive policy",
        "work_scope": "Per-rollout actual method counters, including attempted invalid work; failed-call partial counters may be incomplete; no reference or setup work hidden in rollout counts",
        "elapsed_seconds": elapsed, "numerical_budget_seconds": protocol["config"]["max_seconds"],
        "case_plan_sha256": protocol["case_plan_sha256"], "errors": errors,
        "unreported_ids": missing, "correctness_failures": correctness_failures, **tables}
    summary = {key: value for key, value in canonical.items() if key not in TABLE_IDS}
    for name in TABLE_IDS:
        label = name.removesuffix("_rows")
        summary[f"expected_{label}s"] = len(protocol["plans"]["expected_ids"][name])
        summary[f"reported_{label}s"] = len(tables[name])
    summary["expected_parities"] = summary.pop("expected_paritys")
    summary["reported_parities"] = summary.pop("reported_paritys")
    summary.update(candidate_status_counts=dict(Counter(row["status"] for row in tables["candidate_rows"])),
        frontier_status_counts=dict(Counter(row["status"] for row in tables["frontier_rows"])),
        scope=protocol["scope"], interpretation=protocol["interpretation"])
    write_canonical(run_dir / "work-precision.json", canonical)
    write_json(run_dir / "summary.json", summary)
    (run_dir / "summary.txt").write_text(
        f"TDN prepared CPU work–precision: computational {status}; scientific {scientific}\n"
        f"Candidates: {summary['reported_candidates']}/{summary['expected_candidates']}; "
        f"frontiers: {summary['reported_frontiers']}/{summary['expected_frontiers']}\n"
        f"Numerical wall time: {elapsed:.3f}s / {protocol['config']['max_seconds']}s\n"
        f"Candidates: {summary['candidate_status_counts']}; frontiers: {summary['frontier_status_counts']}\n"
        "RMS and maximum targets are separate; failed and infeasible candidates remain visible.\n"
        "Frontiers are reference-informed post-hoc selections on a fixed step grid, not a deployable adaptive policy.\n"
        "Maximum reference uncertainty is a conservative estimate, not a certificate. No neural/FNO or GPU benefit is established.\n")
    return summary


def run(protocol, run_dir, *, stop=None, progress=None, clock=time.monotonic):
    started = clock()
    tables, errors = {name: [] for name in TABLE_IDS}, []

    def check():
        if stop is not None and stop.requested:
            raise ScreenIncomplete("Stop requested; preserve this attempt and use a fresh run directory")
        if clock() - started >= protocol["config"]["max_seconds"]:
            raise ScreenIncomplete("Frozen numerical time budget exhausted")

    status = "COMPLETED"
    try:
        with torch.no_grad():
            for spec in protocol["plans"]["cases"]:
                check()
                _run_case(protocol, spec, tables, check)
                tables["frontier_rows"] = select_frontiers(protocol, tables["candidate_rows"], tables["reference_rows"])
                validate_rows(tables, protocol["plans"], complete=False)
                _publish(protocol, run_dir, tables, "RUNNING", errors, clock() - started)
                if progress:
                    progress(spec["case_id"], tables)
        check()
        validate_rows(tables, protocol["plans"], complete=True)
    except (Exception, KeyboardInterrupt) as error:
        status = "INCOMPLETE" if isinstance(error, (ScreenIncomplete, KeyboardInterrupt, InterruptedError)) else "FAILED"
        errors.append(f"{type(error).__name__}: {error}")
    return _publish(protocol, run_dir, tables, status, errors, clock() - started)
