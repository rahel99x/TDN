"""Bounded CPU comparisons of compact spatial defects and classical controls.

The same declared batch is timed by every method. References are constructed
independently per member; eligibility uses the worst *member's* adjusted error,
so a good member cannot conceal a poor one. All timing samples and failures are
retained. Selection is post hoc on a frozen step grid, never a step controller.
"""
from __future__ import annotations

from collections import Counter
import math
import statistics
import time
from types import SimpleNamespace

import torch

from tdn.analysis.work_precision.experiment import (
    ScreenIncomplete, _finite, _rollout, write_canonical,
)
from tdn.numerics.reference import choose_substeps, refined_reference
from tdn.numerics.types import Equation, Geometry
from tdn.research.interaction import interaction_defect
from tdn.research.compact_spatial import prepare_compact_step
from tdn.research.work_precision import prepare_step as prepare_control_step
from tdn.runtime.metadata import write_json
from .bank import cases, state, plan
from .protocol import METHODS, TARGETS, NORMS, TABLE_IDS, candidate_id, frontier_id, validate_rows


def prepare_step(u, h, equation, geometry, variant):
    """Dispatch without modifying either numerical implementation's globals."""
    if variant == "gl3_fused":
        return prepare_compact_step(u, h, equation, geometry, variant="gl3_fused")
    compact = {"compact_gl3_m2": ("compact_gl3", 2),
               "compact_gl3_m4": ("compact_gl3", 4),
               "compact_gl3_m8": ("compact_gl3", 8),
               "compact_gl3_m4_no_mean": ("compact_gl3_no_mean", 4)}
    if variant in compact:
        family, modes = compact[variant]
        return prepare_compact_step(u, h, equation, geometry, variant=family, modes=modes)
    return prepare_control_step(u, h, equation, geometry, variant)


def _context(spec):
    return {key: spec[key] for key in ("case_id", "case_role", "panel", "batch_size", "cells")}


def _new_reference(spec):
    return {**_context(spec), "reference_id": f"{spec['case_id']}/reference",
        "status": "INCOMPLETE", "reference_accepted": False,
        "uncertainty_rms": None, "uncertainty_max_estimate": None,
        "reference_seconds": None, "member_references": [
            {"member_index": index, "status": "NOT_STARTED", "reference_accepted": False,
             "uncertainty_rms": None, "uncertainty_max_estimate": None,
             "refinement_attempts": [], "attempts": 0, "rhs_evaluations": 0,
             "rhs_evaluations_complete": False, "reference_reason": "reference_not_started"}
            for index in range(spec["batch_size"])],
        "completed_members": 0, "accepted_members": 0,
        "reference_reason": "reference_not_completed"}


def _teacher(u, spec, equation, geometry, config, check, row=None):
    """Preserve each member's completed refinement attempts, even on a stop."""
    row = _new_reference(spec) if row is None else row
    started = time.perf_counter()
    states = []
    tolerance = config["reference_tolerance"] / math.sqrt(spec["cells"])
    try:
        for member_index in range(u.shape[0]):
            check()
            record = row["member_references"][member_index]
            record.update(status="INCOMPLETE", reference_reason="reference_not_completed")
            count = max(4, choose_substeps(spec["final_time"], equation, geometry))
            for attempt in range(config["reference_attempts"]):
                check()
                reference = refined_reference(u[member_index:member_index + 1],
                    spec["final_time"], equation, geometry, count,
                    tolerance=tolerance, check=check)
                record["rhs_evaluations"] += 28 * count
                uncertainty_max = math.sqrt(spec["cells"]) * reference.uncertainty
                detail = {"attempt": attempt + 1, "reference_n": reference.refinement_substeps[0],
                    "reference_2n": reference.refinement_substeps[1],
                    "reference_4n": reference.refinement_substeps[2],
                    "difference_n_2n": _finite(reference.refinement_differences[0]),
                    "difference_2n_4n": _finite(reference.refinement_differences[1]),
                    "observed_order": _finite(reference.observed_order) if reference.observed_order is not None else None,
                    "uncertainty_rms": _finite(reference.uncertainty),
                    "uncertainty_max_estimate": _finite(uncertainty_max),
                    "reference_accepted": bool(reference.accepted and reference.uncertainty < min(TARGETS)
                                               and uncertainty_max < min(TARGETS)),
                    "reference_reason": reference.reason}
                record["refinement_attempts"].append(detail)
                record.update({key: value for key, value in detail.items() if key != "attempt"})
                record["attempts"] = attempt + 1
                # Finish bookkeeping before the next safe-boundary stop check.
                # Otherwise a final accepted member could coexist with a
                # rejected/incomplete aggregate after an interrupt.
                if detail["reference_accepted"]:
                    break
                count *= 2
            states.append(reference.state)
            record["status"] = "ACCEPTED" if record["reference_accepted"] else "UNRESOLVED"
            record["rhs_evaluations_complete"] = True
            row["completed_members"] += 1
            row["accepted_members"] += int(record["reference_accepted"])
        row["reference_accepted"] = row["accepted_members"] == spec["batch_size"]
        row["status"] = "ACCEPTED" if row["reference_accepted"] else "UNRESOLVED"
        row["reference_reason"] = "accepted" if row["reference_accepted"] else "one_or_more_members_unresolved"
        for key in ("uncertainty_rms", "uncertainty_max_estimate"):
            values = [item[key] for item in row["member_references"]]
            row[key] = max(values) if all(value is not None for value in values) else None
        return SimpleNamespace(state=torch.cat(states, dim=0)), row
    finally:
        row["reference_seconds"] = time.perf_counter() - started


def _identity_parity(spec, u, equation, geometry, protocol, rows, check):
    declaration = protocol["plans"]["parity"]
    h = spec["final_time"] / declaration["identity_nsteps"]
    for dtype_name, dtype in (("float64", torch.float64), ("float32", torch.float32)):
        check()
        field = u.to(dtype)
        full = interaction_defect(field, h, equation, geometry, nodes=3)
        fused = prepare_step(field, h, equation, geometry, "gl3_fused").defect(field)
        difference = full.double() - fused.double()
        maximum = float(difference.abs().max())
        allowed = declaration["dtype_absolute_tolerances"][dtype_name]
        passed = math.isfinite(maximum) and maximum <= allowed
        rows.append({**_context(spec),
            "parity_id": f"{spec['case_id']}/identity/{dtype_name}/gl3_fused",
            "check": "identity", "variant": "gl3_fused", "dtype": dtype_name,
            "nodes": 3, "nsteps": declaration["identity_nsteps"],
            "max_difference": _finite(maximum),
            "rms_difference": _finite(float(difference.square().mean().sqrt())),
            "allowed_absolute_difference": allowed, "passed": passed,
            "status": "PASS" if passed else "FAIL",
            "failure_reason": None if passed else "fused/full raw defect mismatch"})
    # Approximation error against GL3's input-derived defect is a representation
    # diagnostic, not a correctness identity and not a PDE endpoint error.
    check()
    full = interaction_defect(u, h, equation, geometry, nodes=3)
    axes = tuple(range(1, u.ndim))
    full_rms = float(full.square().mean(dim=axes).sqrt().max())
    full_max = float(full.abs().flatten(1).max(dim=1).values.max())
    threshold = declaration.get("relative_defect_resolution_factor", 64) * torch.finfo(u.dtype).eps * max(
        1., float(u.square().mean(dim=axes).sqrt().max()))
    for variant in METHODS:
        if not variant.startswith("compact_"):
            continue
        check()
        approximate = prepare_step(u, h, equation, geometry, variant).defect(u)
        difference = approximate - full
        mean = difference.mean(dim=axes, keepdim=True)
        rms = float(difference.square().mean(dim=axes).sqrt().max())
        rows.append({**_context(spec), "parity_id": f"{spec['case_id']}/approximation/{variant}",
            "check": "approximation", "variant": variant, "dtype": "float64",
            "nodes": 3, "nsteps": declaration["identity_nsteps"],
            "max_difference": _finite(float(difference.abs().max())), "rms_difference": _finite(rms),
            "reference_defect_rms": _finite(full_rms), "reference_defect_max": _finite(full_max),
            "relative_rms_when_resolved": _finite(rms / full_rms) if full_rms > threshold else None,
            "relative_resolution_threshold": threshold,
            "mean_difference_abs": _finite(float(mean.abs().max())),
            "spatial_rms_difference": _finite(float((difference - mean).square().mean(dim=axes).sqrt().max())),
            "allowed_absolute_difference": None, "passed": None, "status": "OBSERVED",
            "failure_reason": None})


def _new_candidate(spec, variant, nsteps, refrow):
    return {**_context(spec), "candidate_id": candidate_id(spec["case_id"], variant, nsteps),
        "variant": variant, "nsteps": nsteps, "h": spec["final_time"] / nsteps,
        "final_time": spec["final_time"], "grid": list(spec["grid"]),
        "dtype": "float64", "status": "INCOMPLETE", "trajectory_completed": False,
        "completed_steps": 0, "finite": True, "in_bounds": True,
        "error_rms": None, "error_max": None, "error_mean": None,
        "error_mean_abs": None, "error_spatial_rms": None, "error_spatial_max": None,
        "error_rms_worst_member": None, "error_max_worst_member": None,
        "error_rms_worst_member_index": None, "error_max_worst_member_index": None,
        "error_upper_rms": None, "error_upper_max_estimate": None,
        "member_errors": [], "reference_accepted": refrow["reference_accepted"],
        "reference_uncertainty_rms": refrow["uncertainty_rms"],
        "reference_uncertainty_max_estimate": refrow["uncertainty_max_estimate"],
        "preparation_seconds": None, "prepared_median_seconds": None,
        "prepared_min_seconds": None, "prepared_max_seconds": None,
        "setup_inclusive_median_seconds": None, "throughput_members_per_second": None,
        "timing_repeats": [], "warmup": None, "work_per_rollout": {},
        "cache_metadata": {}, "failure_reason": None}


def _expand_control_work(sample, spec):
    """Old controls transform one full-grid field per member at each call."""
    work = sample["work"]
    if "fft_total_fields" not in work:
        forward = work.get("fft_forward", 0) * spec["batch_size"]
        inverse = work.get("fft_inverse", 0) * spec["batch_size"]
        work.update(fft_forward_fields=forward, fft_inverse_fields=inverse,
                    fft_total_fields=forward + inverse,
                    fft_transformed_cells=(forward + inverse) * spec["cells"])


def _refresh_candidate(row, value, reference, refrow, repeats):
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
    if complete_sampling and len(durations) == repeats and row["status"] == "VALID":
        row["prepared_median_seconds"] = statistics.median(durations)
        row["prepared_min_seconds"] = min(durations)
        row["prepared_max_seconds"] = max(durations)
        row["setup_inclusive_median_seconds"] = row["preparation_seconds"] + row["prepared_median_seconds"]
        row["throughput_members_per_second"] = row["batch_size"] / row["prepared_median_seconds"]
        row["work_per_rollout"] = dict(row["timing_repeats"][0]["work"])
    fields = ("error_rms", "error_max", "error_mean", "error_mean_abs", "error_spatial_rms", "error_spatial_max",
              "error_rms_worst_member", "error_max_worst_member", "error_upper_rms", "error_upper_max_estimate",
              "error_rms_worst_member_index", "error_max_worst_member_index")
    for field in fields:
        row[field] = None
    row["member_errors"] = []
    if not row["trajectory_completed"] or not row["finite"]:
        return
    for index, member_reference in enumerate(refrow["member_references"]):
        error = value[index:index + 1] - reference.state[index:index + 1]
        mean = error.mean()
        spatial = error - mean
        rms, maximum = _finite(float(error.square().mean().sqrt())), _finite(float(error.abs().max()))
        uncertainty_rms, uncertainty_max = member_reference["uncertainty_rms"], member_reference["uncertainty_max_estimate"]
        row["member_errors"].append({"member_index": index, "error_rms": rms, "error_max": maximum,
            "error_mean": _finite(float(mean)), "error_spatial_rms": _finite(float(spatial.square().mean().sqrt())),
            "error_spatial_max": _finite(float(spatial.abs().max())),
            "error_upper_rms": rms + uncertainty_rms if rms is not None and uncertainty_rms is not None else None,
            "error_upper_max_estimate": maximum + uncertainty_max if maximum is not None and uncertainty_max is not None else None})
    members = row["member_errors"]
    for key in ("error_rms", "error_max", "error_spatial_rms", "error_spatial_max", "error_upper_rms", "error_upper_max_estimate"):
        values = [member[key] for member in members]
        row[key] = max(values) if all(value is not None for value in values) else None
    means = [member["error_mean"] for member in members]
    if all(value is not None for value in means):
        row["error_mean"] = max(means, key=abs)
        row["error_mean_abs"] = abs(row["error_mean"])
    # These scalars name the aggregation explicitly; indices are separate.
    row["error_rms_worst_member"] = row["error_rms"]
    row["error_max_worst_member"] = row["error_max"]
    for norm in ("rms", "max"):
        key = f"error_{norm}"
        row[f"{key}_worst_member_index"] = max(range(len(members)), key=lambda i: members[i][key]) if row[key] is not None else None


def _run_case(protocol, spec, tables, check):
    config = protocol["config"]
    u = state(spec)
    if u.shape[0] != spec["batch_size"]:
        raise ValueError("Declared batch size does not match generated state")
    geometry = Geometry(tuple(spec["grid"]), tuple(spec["lengths"]))
    equation = Equation(spec["kappa"], spec["reaction_rate"])
    refrow = _new_reference(spec)
    tables["reference_rows"].append(refrow)
    reference, refrow = _teacher(u, spec, equation, geometry, config, check, refrow)
    _identity_parity(spec, u, equation, geometry, protocol, tables["parity_rows"], check)
    for nsteps in spec["nsteps"]:
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
        for repeat, variants in [(-1, order["warmup_order"]), *enumerate(order["measured_orders"])]:
            for position, variant in enumerate(variants):
                row = rows[variant]
                sample, value, error = _rollout(prepared[variant], u, nsteps, check, repeat, position)
                _expand_control_work(sample, spec)
                row["warmup"] = sample if repeat == -1 else row["warmup"]
                if repeat != -1:
                    row["timing_repeats"].append(sample)
                _refresh_candidate(row, value, reference, refrow, config["timing_repeats"])
                endpoints[variant] = value
                if error is not None:
                    raise error
        available = all(rows[variant]["status"] == "VALID" for variant in ("gl3", "gl3_fused"))
        difference = endpoints["gl3"] - endpoints["gl3_fused"] if available else None
        maximum = float(difference.abs().max()) if available else None
        allowed = protocol["plans"]["parity"]["rollout_absolute_tolerance"]
        passed = maximum <= allowed if available else None
        tables["parity_rows"].append({**_context(spec),
            "parity_id": f"{spec['case_id']}/rollout_fused/n{nsteps}",
            "check": "rollout_fused", "variant": "gl3_fused", "dtype": "float64", "nodes": 3,
            "nsteps": nsteps, "max_difference": maximum,
            "rms_difference": float(difference.square().mean().sqrt()) if available else None,
            "allowed_absolute_difference": allowed, "passed": passed,
            "status": "PASS" if passed else "FAIL" if available else "UNAVAILABLE",
            "failure_reason": None if passed else "fused/full rollout mismatch" if available else "invalid or incomplete trajectory"})


def select_frontiers(protocol, candidate_rows, reference_rows):
    """Select independently for warmed/setup cost using worst-member errors."""
    references = {row["case_id"]: row for row in reference_rows}
    result = []
    for spec in protocol["plans"]["cases"]:
        case_id = spec["case_id"]
        reference = references.get(case_id)
        for variant in METHODS:
            candidates = [row for row in candidate_rows if row["case_id"] == case_id and row["variant"] == variant]
            coverage_complete = (len(candidates) == len(spec["nsteps"])
                and {row["nsteps"] for row in candidates} == set(spec["nsteps"])
                and all(row["status"] in {"VALID", "INVALID"} for row in candidates))
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
                    result.append({**_context(spec), "frontier_id": frontier_id(case_id, variant, norm, tolerance),
                        "variant": variant, "norm": norm, "tolerance": tolerance,
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
                        "throughput_members_per_second": fastest.get("throughput_members_per_second") if fastest else None,
                        "feasible_candidate_count": len(feasible), "selection_scope": "posthoc_declared_grid_worst_member"})
    return result


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
    canonical = {"schema": "tdn.compact-spatial/v1", "benchmark_suite": "compact-spatial",
        "status": status, "computational_status": status, "scientific_outcome": scientific,
        "device": "cpu", "dtype": "float64", "training_attempted": False, "training_performed": False,
        "reference_scope": "Independent same-grid coupled FP64 RK4 n/2n/4n per batch member; refinement and rounding uncertainty estimates are not certificates or continuum error estimates",
        "timing_scope": "Prepared complete native-batch CPU rollouts; one all-method warmup followed by five rotated/interleaved repeats; state validation and budget/stop checks are timed",
        "setup_inclusive_scope": "One preparation sample plus median warmed rollout; excludes metadata, reference, warmup, parity, cloning, endpoint errors and reports; not a cold-cache measurement",
        "selection_scope": "Post-hoc declared step grid, independent warmed/setup selections; worst per-member (error + uncertainty), with every member reference accepted; not an adaptive policy",
        "work_scope": "Actual method operations over entire native batch; failed-call counters can be partial; coefficient-cache bytes do not measure allocator or process peak memory",
        "approximation_scope": "Raw compact defect versus input-derived full GL3 defect; not a reference PDE error or a learned approximation",
        "elapsed_seconds": elapsed, "numerical_budget_seconds": protocol["config"]["max_seconds"],
        "case_plan_sha256": protocol["case_plan_sha256"], "errors": errors,
        "unreported_ids": missing, "correctness_failures": correctness_failures, **tables}
    summary = {key: value for key, value in canonical.items() if key not in TABLE_IDS}
    for name in TABLE_IDS:
        label = name.removesuffix("_rows")
        plural = "parities" if label == "parity" else label + "s"
        summary[f"expected_{plural}"] = len(protocol["plans"]["expected_ids"][name])
        summary[f"reported_{plural}"] = len(tables[name])
    summary.update(candidate_status_counts=dict(Counter(row["status"] for row in tables["candidate_rows"])),
        frontier_status_counts=dict(Counter(row["status"] for row in tables["frontier_rows"])),
        parity_status_counts=dict(Counter(row["status"] for row in tables["parity_rows"])),
        scope=protocol["scope"], interpretation=protocol["interpretation"])
    write_canonical(run_dir / "compact-spatial.json", canonical)
    write_json(run_dir / "summary.json", summary)
    (run_dir / "summary.txt").write_text(
        f"TDN compact spatial CPU experiment: computational {status}; scientific {scientific}\n"
        f"Candidates: {summary['reported_candidates']}/{summary['expected_candidates']}; "
        f"frontiers: {summary['reported_frontiers']}/{summary['expected_frontiers']}\n"
        f"Numerical wall time: {elapsed:.3f}s / {protocol['config']['max_seconds']}s\n"
        f"Candidates: {summary['candidate_status_counts']}; frontiers: {summary['frontier_status_counts']}\n"
        "Native batches use independent member references and worst-member adjusted RMS/maximum targets.\n"
        "Accuracy and scaling panels are separate; failed, invalid and infeasible candidates remain visible.\n"
        "Frontiers are post-hoc selections, not an adaptive policy. No neural/FNO or GPU benefit is established.\n")
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
