"""Frozen CPU mechanism panels for nonlinear mixing before compression.

The numerical and learned-model tracks are separate experiments. This module
does no training. Output projection is explicitly a full-cost representation
control; direct cyclic convolution is tested only on declared small cases.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from tdn.analysis.compact_spatial.experiment import (
    _expand_control_work, _new_candidate, _new_reference, _refresh_candidate,
    prepare_step as prepare_old_step,
)
from tdn.analysis.work_precision.experiment import ScreenIncomplete, _finite, _rollout, write_canonical
from tdn.analysis.work_precision.protocol import (
    TABLE_IDS, candidate_id, digest, frontier_id, validate_rows as validate_work_rows,
)
from tdn.numerics.reference import choose_substeps, refined_reference
from tdn.numerics.types import Equation, Geometry
from tdn.research.interaction import interaction_defect
from tdn.research.premix import prepare_premix_step
from tdn.runtime.metadata import write_json


ARTIFACTS = ("protocol.json", "config.json", "metrics.json", "summary.json", "summary.txt")
METHODS = ("strang", "etdrk4", "gl3", "gl3_fused", "compact_gl3_m4", "compact_gl3_m8",
           "output_gl3_m4", "output_gl3_m8", "gl3_fused_chunked")
TARGETS = (2e-3, 2e-4, 2e-5, 2e-6, 2e-7, 2e-8)
NORMS = ("rms", "max")
INTERPRETATION = [
    "Regimes are declared expectations, not labels selected from measured wins: favorable has a smooth resolved spectrum (also small amplitude in the accuracy panel); typical is broadband; adverse contains cutoff pairs, Nyquist content, localized structure, or steep fronts.",
    "This partial stress design couples factors; it is not a full factorial or an independent statistical sample. Resolutions and prefix batch members are paired.",
    "Full-input output projection preserves high-high-to-low interactions only for retained outputs; it still computes the full fused GL3 defect and is not an inexpensive algorithm.",
    "Selected-output cyclic convolution retains all original-grid source pairs at O(batch*cells*retained_outputs) cost. Original-grid aliases are preserved; this is not continuum dealiasing.",
    "Every numerical solver uses the same FP64 state, discrete PDE, horizon, native batch and frozen step grid. State checks and safe-boundary budget checks are timed.",
    "References are independently refined coupled RK4 per member. Accepted identical prefix-member references can be reused by a state-and-equation digest; reuse is explicit and never counted as fresh evidence.",
    "Reference uncertainties are full last-refinement differences plus rounding allowances; sqrt(cells) supplies a conservative maximum-norm estimate, not a certificate or continuum error.",
    "A batch meets a target only if every member meets error plus its own accepted uncertainty. RMS and maximum targets are separate.",
    "One all-method warmup precedes five rotated full-rollout measurements. Preparation has one sample. Setup-inclusive cost is preparation plus warmed median, not a cold-cache measurement.",
    "Post-hoc matched-target frontiers are reference-informed choices on a fixed schedule grid, not an adaptive deployment policy. Invalid, infeasible and unmeasured candidates remain visible.",
    "Approximation differences against GL3 are observations; required projection, direct-convolution and batching identities can fail correctness. Coefficient bytes are not peak memory.",
    "The numerical track establishes no learned TDN/FNO ranking or GPU benefit. The separately bounded neural track supplies that evidence.",
]


def _spec(grid, pattern, panel, batch_size=1):
    dim, size = len(grid), max(grid)
    parameters = {
        "smooth": ("favorable", .4, .04, .0005, 1.3, .16),
        "broadband": ("typical", .45, .16, .00025, 3.2, .3),
        "cutoff_pair": ("adverse", .4, .28, .00005, 3.2, .6),
        "near_nyquist": ("adverse", .5, .24, 2 / size**2, 4., .4),
        "localized": ("adverse", .38, .36, .00008, 6., .5),
        "front": ("adverse", .5, .45, .00005, 8., .7),
    }
    regime, mean, amplitude, kappa, rate, horizon = parameters[pattern]
    if panel == "scaling":
        # Stronger reaction, longer horizon and tighter targets than the prior
        # compact experiment. Diffusion is fixed across grids within dimension.
        mean, amplitude, kappa, rate, horizon = .4, .28, .000005 if dim == 1 else .000025, 6., .5
        regime = "favorable" if pattern == "smooth" else "adverse"
    case_id = f"{panel}/{dim}d/{'x'.join(map(str, grid))}/{pattern}/b{batch_size}"
    methods = list(METHODS)
    if panel == "accuracy" and pattern == "cutoff_pair" and grid in ([128], [32, 32]):
        methods.append("selected_output_gl3_m4")
    return dict(case_id=case_id, case_role="declared_mechanism_stress", panel=panel,
                regime=regime, pattern=pattern, grid=list(grid), lengths=[1.] * dim,
                cells=math.prod(grid), batch_size=batch_size, mean=mean, amplitude=amplitude,
                kappa=kappa, reaction_rate=rate, final_time=horizon, phase=.193,
                nsteps=[1, 2, 4, 8, 16, 32] if panel == "accuracy" else [2, 8],
                methods=methods, near_nyquist_diffusion_scaled=pattern == "near_nyquist",
                pairing="Phase-indexed batch prefixes; same pattern and physical parameters across grids except explicitly grid-scaled Nyquist diffusion")


def _cases(profile, panel):
    if panel == "accuracy":
        bank = [_spec(grid, pattern, panel) for grid in ([128], [512], [32, 32], [64, 64])
                for pattern in ("smooth", "broadband", "cutoff_pair", "near_nyquist", "localized", "front")]
        if profile == "smoke":
            wanted = {("128", "smooth"), ("128", "cutoff_pair"), ("32x32", "broadband"), ("32x32", "front")}
            bank = [s for s in bank if ("x".join(map(str, s["grid"])), s["pattern"]) in wanted]
            for spec in bank:
                spec["nsteps"] = [1, 4]
        return bank
    bank = [_spec(grid, pattern, panel, batch) for grid in ([128], [512], [2048], [32, 32], [64, 64], [128, 128])
            for pattern in ("smooth", "cutoff_pair") for batch in (1, 4, 16)]
    if profile == "smoke":
        bank = [s for s in bank if s["grid"] == [32, 32] and
                ((s["pattern"] == "smooth" and s["batch_size"] in (1, 4)) or
                 (s["pattern"] == "cutoff_pair" and s["batch_size"] == 4))]
    return bank


def _parity_keys(spec):
    case = spec["case_id"]
    variants = ["output_gl3_m4", "output_gl3_m8", "gl3_fused_chunked"]
    if "selected_output_gl3_m4" in spec["methods"]:
        variants.append("selected_output_gl3_m4")
    return ([f"{case}/identity/{dtype}/{variant}" for dtype in ("float64", "float32") for variant in variants]
            + [f"{case}/approximation/{variant}" for variant in
               ("compact_gl3_m4", "compact_gl3_m8", "output_gl3_m4", "output_gl3_m8")]
            + [f"{case}/rollout/{variant}/n{n}" for n in spec["nsteps"] for variant in
               ("gl3_fused", "gl3_fused_chunked")])


def build_protocol(profile="full", panel="accuracy"):
    """Build the entire immutable case, method, timing and identity plan."""
    if profile not in {"smoke", "full"} or panel not in {"accuracy", "scaling"}:
        raise ValueError("Use profile smoke/full and panel accuracy/scaling")
    cases, expected, orders = _cases(profile, panel), {key: [] for key in TABLE_IDS}, []
    for index, spec in enumerate(cases):
        case, methods = spec["case_id"], spec["methods"]
        expected["reference_rows"].append(f"{case}/reference")
        expected["candidate_rows"].extend(candidate_id(case, method, n) for n in spec["nsteps"] for method in methods)
        expected["frontier_rows"].extend(frontier_id(case, method, norm, target) for method in methods for norm in NORMS for target in TARGETS)
        expected["parity_rows"].extend(_parity_keys(spec))
        for step_index, n in enumerate(spec["nsteps"]):
            rotation = (index + step_index) % len(methods)
            order = methods[rotation:] + methods[:rotation]
            measured = []
            for repeat in range(5):
                shift = (rotation + repeat + 1) % len(methods)
                measured.append(methods[shift:] + methods[:shift])
            orders.append(dict(case_id=case, nsteps=n, preparation_order=order,
                               warmup_order=order, measured_orders=measured))
    config = dict(protocol_version=1, suite="premix-mechanisms", profile=profile, panel=panel,
                  max_seconds=240 if profile == "smoke" else 1200, reference_tolerance=1e-9,
                  reference_attempts=7, intraop_threads=1, interop_threads=1, timing_repeats=5,
                  error_targets=list(TARGETS), warmup_rollouts=1, parent_chunk_size=4, pair_chunk_size=8)
    plans = dict(cases=cases, expected_ids=expected, timing_orders=orders,
                 parity=dict(identity_nsteps=4, dtype_absolute_tolerances={"float64": 2e-12, "float32": 3e-7},
                             rollout_absolute_tolerance=2e-11))
    return dict(version=1, benchmark_suite="premix-mechanisms", profile=profile, panel=panel,
                device="cpu", dtype="float64", config=config, plans=plans,
                case_plan_sha256=digest(cases), training_attempted=False, training_performed=False,
                scope="Bounded full-input nonlinear-mixing accuracy and scaling panels for the periodic logistic reaction-diffusion system",
                interpretation=list(INTERPRETATION))


def state(spec, *, dtype=torch.float64):
    """Periodic smooth, broadband, high-high, Nyquist, localized and front fields."""
    coordinates = torch.meshgrid(*(torch.arange(n, dtype=dtype) / n for n in spec["grid"]), indexing="ij")
    x, tau = coordinates[0], 2 * math.pi
    y = coordinates[1] if len(coordinates) == 2 else torch.zeros_like(x)
    members = []
    for index in range(spec["batch_size"]):
        phase, p = spec["phase"] + .113 * index, spec["pattern"]
        if p == "smooth":
            wave = .55 * torch.cos(tau * (2*x + y) + phase) + .45 * torch.sin(tau * (x - 3*y) - .71)
        elif p == "broadband":
            wave = sum(w * torch.cos(tau * (kx*x + ky*y) + phase*kx)
                       for w, kx, ky in ((.32, 1, 2), (.27, 4, -3), (.23, 9, 5), (.18, 13, -11)))
        elif p == "cutoff_pair":
            wave = .5 * torch.cos(tau * (9*x + 9*y) + phase) + .5 * torch.sin(tau * (10*x + 10*y) - .37)
        elif p == "near_nyquist":
            nx, ny = spec["grid"][0], spec["grid"][-1]
            wave = .5 * torch.cos(tau * ((nx//2-1)*x + (ny//2-1)*y) + phase)
            wave = wave + .5 * torch.sin(tau * ((nx//2-2)*x + (ny//2-2)*y) - .37)
        elif p == "localized":
            distance = torch.sin(math.pi * (x - .37 - .02 * index)).square()
            if len(coordinates) == 2:
                distance = distance + torch.sin(math.pi * (y - .61)).square()
            wave = 2 * torch.exp(-24 * distance) - 1
        elif p == "front":
            wave = torch.tanh(12 * torch.sin(tau * (x + y) + phase))
        else:
            raise ValueError(f"Unknown pattern {p}")
        mean = .32 + .36 * ((index * 7) % 16) / 15 if spec["panel"] == "scaling" else spec["mean"]
        members.append(mean + spec["amplitude"] * wave)
    return torch.stack(members).unsqueeze(1)


def prepare_step(u, h, equation, geometry, variant):
    if variant.startswith("output_gl3_m"):
        return prepare_premix_step(u, h, equation, geometry, "output_gl3", modes=int(variant.rsplit("m", 1)[1]))
    if variant == "selected_output_gl3_m4":
        return prepare_premix_step(u, h, equation, geometry, "selected_output_gl3", modes=4, chunk_size=8)
    if variant == "gl3_fused_chunked":
        return prepare_premix_step(u, h, equation, geometry, variant, chunk_size=4)
    return prepare_old_step(u, h, equation, geometry, variant)


def _context(spec):
    return {key: spec[key] for key in ("case_id", "case_role", "panel", "batch_size", "cells", "regime", "pattern")}


def _reference_key(member, spec):
    metadata = {key: spec[key] for key in ("grid", "lengths", "kappa", "reaction_rate", "final_time")}
    return hashlib.sha256(digest(metadata).encode() + member.contiguous().numpy().tobytes()).hexdigest()


def _teacher(u, spec, equation, geometry, config, check, row, cache):
    """Accepted-only prefix reuse; each first-use member has its own refinement."""
    started, states = time.perf_counter(), []
    try:
        for index in range(spec["batch_size"]):
            check()
            member, record = u[index:index+1], row["member_references"][index]
            key = _reference_key(member, spec)
            record["reference_state_sha256"] = key
            if key in cache:
                reference_state, source = cache[key]
                record.update(deepcopy(source), member_index=index, cache_reused=True,
                              rhs_evaluations_this_case=0,
                              reference_source=source["reference_source"])
                states.append(reference_state)
            else:
                count = max(4, choose_substeps(spec["final_time"], equation, geometry))
                record.update(status="INCOMPLETE", cache_reused=False,
                              reference_source=f"{spec['case_id']}/reference/member-{index}")
                for attempt in range(config["reference_attempts"]):
                    check()
                    reference = refined_reference(member, spec["final_time"], equation, geometry, count,
                        tolerance=config["reference_tolerance"] / math.sqrt(spec["cells"]), check=check)
                    record["rhs_evaluations"] += 28 * count
                    maximum = math.sqrt(spec["cells"]) * reference.uncertainty
                    detail = dict(attempt=attempt+1, reference_n=reference.refinement_substeps[0],
                        reference_2n=reference.refinement_substeps[1], reference_4n=reference.refinement_substeps[2],
                        difference_n_2n=_finite(reference.refinement_differences[0]),
                        difference_2n_4n=_finite(reference.refinement_differences[1]),
                        observed_order=_finite(reference.observed_order) if reference.observed_order is not None else None,
                        uncertainty_rms=_finite(reference.uncertainty), uncertainty_max_estimate=_finite(maximum),
                        reference_accepted=bool(reference.accepted and maximum < min(TARGETS)),
                        reference_reason=reference.reason)
                    record["refinement_attempts"].append(detail)
                    record.update({k: v for k, v in detail.items() if k != "attempt"}, attempts=attempt+1)
                    if detail["reference_accepted"]:
                        break
                    count *= 2
                states.append(reference.state)
                record.update(status="ACCEPTED" if record["reference_accepted"] else "UNRESOLVED",
                              rhs_evaluations_complete=True, rhs_evaluations_this_case=record["rhs_evaluations"])
                if record["reference_accepted"]:
                    cache[key] = (reference.state, deepcopy(record))
            row["completed_members"] += 1
            row["accepted_members"] += int(record["reference_accepted"])
        row.update(reference_accepted=row["accepted_members"] == spec["batch_size"],
                   cached_members=sum(m.get("cache_reused", False) for m in row["member_references"]))
        row["status"] = "ACCEPTED" if row["reference_accepted"] else "UNRESOLVED"
        row["reference_reason"] = "accepted" if row["reference_accepted"] else "one_or_more_members_unresolved"
        for field in ("uncertainty_rms", "uncertainty_max_estimate"):
            values = [r[field] for r in row["member_references"]]
            row[field] = max(values) if all(v is not None for v in values) else None
        return SimpleNamespace(state=torch.cat(states)), row
    finally:
        row["reference_seconds"] = time.perf_counter() - started
        row["rhs_evaluations_this_case"] = sum(r.get("rhs_evaluations_this_case", r["rhs_evaluations"])
                                                 for r in row["member_references"])


def _project(field, modes):
    axes = tuple(range(2, field.ndim))
    mask = torch.ones(field.shape[2:], dtype=torch.bool, device=field.device)
    for axis, n in enumerate(field.shape[2:]):
        indices = torch.arange(n, device=field.device)
        frequencies = torch.where(indices <= (n-1)//2, indices, indices-n)
        shape = [1] * len(axes)
        shape[axis] = n
        mask = mask & (frequencies.reshape(shape).abs() <= modes)
    return torch.fft.ifftn(torch.fft.fftn(field, dim=axes) * mask, dim=axes).real


def _parity(spec, u, equation, geometry, protocol, rows, check):
    h = spec["final_time"] / 4
    for dtype_name, dtype in (("float64", torch.float64), ("float32", torch.float32)):
        check()
        field = u.to(dtype)
        full = interaction_defect(field, h, equation, geometry, nodes=3)
        for variant in ("output_gl3_m4", "output_gl3_m8", "gl3_fused_chunked", "selected_output_gl3_m4"):
            if variant not in spec["methods"]:
                continue
            check()
            expected = full if variant == "gl3_fused_chunked" else _project(full, 8 if variant.endswith("m8") else 4)
            measured = prepare_step(field, h, equation, geometry, variant).defect(field)
            difference = measured.double() - expected.double()
            allowed = protocol["plans"]["parity"]["dtype_absolute_tolerances"][dtype_name]
            maximum = _finite(float(difference.abs().max()))
            passed = maximum is not None and maximum <= allowed
            rows.append({**_context(spec), "parity_id": f"{spec['case_id']}/identity/{dtype_name}/{variant}",
                "check": "identity", "variant": variant, "dtype": dtype_name, "nsteps": 4,
                "status": "PASS" if passed else "FAIL", "passed": passed,
                "max_difference": maximum, "rms_difference": _finite(float(difference.square().mean().sqrt())),
                "allowed_absolute_difference": allowed,
                "failure_reason": None if passed else "independent full-GL3 projection/batching identity mismatch"})
    full = interaction_defect(u, h, equation, geometry, nodes=3)
    axes = tuple(range(1, u.ndim))
    full_rms = float(full.square().mean(dim=axes).sqrt().max())
    for variant in ("compact_gl3_m4", "compact_gl3_m8", "output_gl3_m4", "output_gl3_m8"):
        check()
        difference = prepare_step(u, h, equation, geometry, variant).defect(u) - full
        rms = float(difference.square().mean(dim=axes).sqrt().max())
        rows.append({**_context(spec), "parity_id": f"{spec['case_id']}/approximation/{variant}",
            "check": "approximation", "variant": variant, "dtype": "float64", "nsteps": 4,
            "status": "OBSERVED", "passed": None, "max_difference": _finite(float(difference.abs().max())),
            "rms_difference": _finite(rms), "reference_defect_rms": _finite(full_rms),
            "relative_rms_when_resolved": _finite(rms/full_rms) if full_rms > 64*torch.finfo(u.dtype).eps else None,
            "retained_low_output_rms_difference": _finite(float(_project(difference, 8 if variant.endswith("m8") else 4).square().mean(dim=axes).sqrt().max())),
            "allowed_absolute_difference": None, "failure_reason": None})


def _run_case(protocol, spec, tables, check, cache):
    u, config = state(spec), protocol["config"]
    geometry = Geometry(tuple(spec["grid"]), tuple(spec["lengths"]))
    equation = Equation(spec["kappa"], spec["reaction_rate"])
    refrow = {**_new_reference(spec), **_context(spec)}
    tables["reference_rows"].append(refrow)
    reference, refrow = _teacher(u, spec, equation, geometry, config, check, refrow, cache)
    _parity(spec, u, equation, geometry, protocol, tables["parity_rows"], check)
    for n in spec["nsteps"]:
        order = next(o for o in protocol["plans"]["timing_orders"] if o["case_id"] == spec["case_id"] and o["nsteps"] == n)
        prepared, candidates, endpoints = {}, {}, {}
        for variant in order["preparation_order"]:
            check()
            row = {**_new_candidate(spec, variant, n, refrow), **_context(spec)}
            candidates[variant] = row
            tables["candidate_rows"].append(row)
            started = time.perf_counter()
            try:
                prepared[variant] = prepare_step(u, row["h"], equation, geometry, variant)
            except (Exception, KeyboardInterrupt) as error:
                row.update(status="INCOMPLETE" if isinstance(error, (ScreenIncomplete, KeyboardInterrupt, InterruptedError)) else "FAILED",
                           failure_reason=f"{type(error).__name__}: {error}")
                raise
            finally:
                row["preparation_seconds"] = time.perf_counter() - started
            row["cache_metadata"] = prepared[variant].metadata
        for repeat, variants in [(-1, order["warmup_order"]), *enumerate(order["measured_orders"])]:
            for position, variant in enumerate(variants):
                sample, value, error = _rollout(prepared[variant], u, n, check, repeat, position)
                _expand_control_work(sample, spec)
                row = candidates[variant]
                if repeat == -1:
                    row["warmup"] = sample
                else:
                    row["timing_repeats"].append(sample)
                _refresh_candidate(row, value, reference, refrow, config["timing_repeats"])
                endpoints[variant] = value
                if error is not None:
                    raise error
        for variant in ("gl3_fused", "gl3_fused_chunked"):
            available = all(candidates[v]["status"] == "VALID" for v in ("gl3", variant))
            difference = endpoints[variant] - endpoints["gl3"] if available else None
            maximum = float(difference.abs().max()) if available else None
            allowed = protocol["plans"]["parity"]["rollout_absolute_tolerance"]
            passed = maximum <= allowed if available else None
            tables["parity_rows"].append({**_context(spec), "parity_id": f"{spec['case_id']}/rollout/{variant}/n{n}",
                "check": "rollout", "variant": variant, "dtype": "float64", "nsteps": n,
                "status": "PASS" if passed else "FAIL" if available else "UNAVAILABLE", "passed": passed,
                "max_difference": maximum, "rms_difference": float(difference.square().mean().sqrt()) if available else None,
                "allowed_absolute_difference": allowed, "failure_reason": None if passed else "invalid trajectory or full/fused rollout mismatch"})


def select_frontiers(protocol, candidates, references):
    references = {r["case_id"]: r for r in references}
    result = []
    for spec in protocol["plans"]["cases"]:
        ref = references.get(spec["case_id"])
        for variant in spec["methods"]:
            rows = [r for r in candidates if r["case_id"] == spec["case_id"] and r["variant"] == variant]
            covered = len(rows) == len(spec["nsteps"]) and {r["nsteps"] for r in rows} == set(spec["nsteps"]) and all(r["status"] in {"VALID", "INVALID"} for r in rows)
            resolved = bool(ref and ref["reference_accepted"] and covered)
            for norm in NORMS:
                error_key = f"error_{norm}"
                upper = "error_upper_rms" if norm == "rms" else "error_upper_max_estimate"
                uncertainty = "uncertainty_rms" if norm == "rms" else "uncertainty_max_estimate"
                for target in protocol["config"]["error_targets"]:
                    feasible = [r for r in rows if r["status"] == "VALID" and r["reference_accepted"] and r[upper] is not None and r[upper] <= target and r["prepared_median_seconds"] is not None] if resolved else []
                    fastest = min(feasible, key=lambda r: (r["prepared_median_seconds"], r["nsteps"])) if feasible else None
                    setup = min(feasible, key=lambda r: (r["setup_inclusive_median_seconds"], r["nsteps"])) if feasible else None
                    result.append({**_context(spec), "frontier_id": frontier_id(spec["case_id"], variant, norm, target),
                        "variant": variant, "norm": norm, "tolerance": target,
                        "status": "FEASIBLE" if fastest else "NO_FEASIBLE_CANDIDATE" if resolved else "INCONCLUSIVE",
                        "selected_candidate_id": fastest["candidate_id"] if fastest else None,
                        "selected_nsteps": fastest["nsteps"] if fastest else None,
                        "selected_error": fastest[error_key] if fastest else None,
                        "adjusted_error": fastest[upper] if fastest else None,
                        "reference_uncertainty": ref[uncertainty] if ref else None,
                        "prepared_seconds": fastest["prepared_median_seconds"] if fastest else None,
                        "setup_inclusive_seconds": fastest["setup_inclusive_median_seconds"] if fastest else None,
                        "setup_selected_candidate_id": setup["candidate_id"] if setup else None,
                        "setup_selected_nsteps": setup["nsteps"] if setup else None,
                        "setup_selected_error": setup[error_key] if setup else None,
                        "setup_selected_adjusted_error": setup[upper] if setup else None,
                        "setup_selected_seconds": setup["setup_inclusive_median_seconds"] if setup else None,
                        "feasible_candidate_count": len(feasible), "selection_scope": "posthoc_declared_grid_worst_member"})
    return result


def validate_tables(protocol, tables, *, complete):
    validate_work_rows(tables, protocol["plans"], complete=complete)
    specs = {s["case_id"]: s for s in protocol["plans"]["cases"]}
    for name, rows in tables.items():
        for row in rows:
            spec = specs[row["case_id"]]
            if any(row.get(k) != v for k, v in _context(spec).items()):
                raise ValueError("Row context disagrees with frozen case plan")
            if name == "candidate_rows":
                if row["variant"] not in spec["methods"] or row["nsteps"] not in spec["nsteps"]:
                    raise ValueError("Undeclared candidate method/schedule")
                members = row["member_errors"]
                if row["trajectory_completed"] and row["finite"]:
                    if [r["member_index"] for r in members] != list(range(spec["batch_size"])):
                        raise ValueError("Every completed batch member needs endpoint errors")
                    for key in ("error_rms", "error_max", "error_upper_rms", "error_upper_max_estimate"):
                        values = [r[key] for r in members]
                        expected = max(values) if all(v is not None for v in values) else None
                        if row[key] != expected:
                            raise ValueError("Batch feasibility must retain each member's adjusted error")
                elif members:
                    raise ValueError("Partial trajectories cannot report endpoint errors")
            if name == "reference_rows":
                accepted = len(row["member_references"]) == spec["batch_size"] and all(r["reference_accepted"] for r in row["member_references"])
                if row["reference_accepted"] != accepted:
                    raise ValueError("Reference acceptance must include every member")
            if name == "parity_rows" and row["check"] == "approximation" and (row["status"] != "OBSERVED" or row["passed"] is not None):
                raise ValueError("Approximation error is an observation, not correctness")


def _publish(protocol, run_dir, tables, status, errors, elapsed):
    tables["frontier_rows"] = select_frontiers(protocol, tables["candidate_rows"], tables["reference_rows"])
    observed = {name: {r[key] for r in tables[name]} for name, key in TABLE_IDS.items()}
    missing = {name: [i for i in protocol["plans"]["expected_ids"][name] if i not in observed[name]]
               for name in TABLE_IDS}
    if status == "COMPLETED" and any(missing.values()):
        status = "INCOMPLETE"
    failures = [r["parity_id"] for r in tables["parity_rows"] if r["status"] == "FAIL"]
    if failures and status == "COMPLETED":
        status = "FAILED"
    scientific = "OBSERVED_MIXED" if status == "COMPLETED" and all(r["reference_accepted"] for r in tables["reference_rows"]) else "INCONCLUSIVE"
    canonical = dict(schema="tdn.premix-mechanisms/v1", benchmark_suite="premix-mechanisms",
        profile=protocol["profile"], panel=protocol["panel"], status=status, computational_status=status,
        scientific_outcome=scientific, device="cpu", dtype="float64", training_attempted=False, training_performed=False,
        config_sha256=digest(protocol["config"]), case_plan_sha256=protocol["case_plan_sha256"],
        elapsed_seconds=elapsed, numerical_budget_seconds=protocol["config"]["max_seconds"],
        errors=errors, unreported_ids=missing, correctness_failures=failures,
        actual_runtime={"torch": str(torch.__version__), "intraop_threads": torch.get_num_threads(),
                        "interop_threads": torch.get_num_interop_threads()},
        selection_scope="Reference-informed post-hoc step grid; worst member error plus own uncertainty; separate RMS/maximum targets",
        scope=protocol["scope"], interpretation=protocol["interpretation"], **tables)
    summary = {k: v for k, v in canonical.items() if k not in TABLE_IDS}
    summary["coverage"] = {name: dict(expected=len(protocol["plans"]["expected_ids"][name]), reported=len(rows)) for name, rows in tables.items()}
    for name in TABLE_IDS:
        summary[name.removesuffix("_rows") + "_status_counts"] = dict(Counter(r["status"] for r in tables[name]))
    summary["accepted_reference_members"] = sum(r["accepted_members"] for r in tables["reference_rows"])
    summary["cached_reference_members"] = sum(r.get("cached_members", 0) for r in tables["reference_rows"])
    summary["regime_case_counts"] = dict(Counter(s["regime"] for s in protocol["plans"]["cases"]))
    write_canonical(run_dir / "metrics.json", canonical)
    write_json(run_dir / "summary.json", summary)
    (run_dir / "summary.txt").write_text(
        f"TDN premix {protocol['panel']} ({protocol['profile']}): computational {status}; scientific {scientific}\n"
        f"Candidates: {len(tables['candidate_rows'])}/{len(protocol['plans']['expected_ids']['candidate_rows'])}; "
        f"references: {summary['accepted_reference_members']} accepted member instances ({summary['cached_reference_members']} cached)\n"
        f"Numerical wall time: {elapsed:.3f}s / {protocol['config']['max_seconds']}s\n"
        f"Frontiers: {summary['frontier_status_counts']}; correctness failures: {len(failures)}\n"
        "Favorable/typical/adverse labels are declared before measurement. All failures remain visible.\n"
        "Native-batch RMS/maximum targets require every accepted member; frontiers are post-hoc, not adaptive.\n"
        "Output projection computes full interactions. The separate neural track is needed for learned TDN/FNO evidence.\n")
    return summary


def run(protocol, run_dir, *, stop=None, progress=None, clock=time.monotonic):
    """Retain partial reports on stop/budget failure; caller owns completion seal."""
    declared = build_protocol(protocol["profile"], protocol["panel"])
    for field in declared:
        if digest(protocol[field]) != digest(declared[field]):
            raise ValueError(f"Changed frozen premix mechanism {field}; declare a new protocol")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if any((run_dir/name).exists() for name in ("metrics.json", "summary.json", "COMPLETED", "manifest.json")):
        raise ValueError("Preserve prior premix results; use a fresh run directory")
    for name, payload in (("protocol.json", protocol), ("config.json", protocol["config"])):
        path = run_dir / name
        if path.exists():
            if digest(json.loads(path.read_text())) != digest(payload):
                raise ValueError("Preserve contradictory existing protocol/configuration; use a fresh run directory")
        else:
            write_json(path, payload)
    started, tables, errors, cache = clock(), {name: [] for name in TABLE_IDS}, [], {}

    def check():
        if stop is not None and stop.requested:
            raise ScreenIncomplete("Stop requested; preserve this attempt and use a fresh run")
        if clock() - started >= protocol["config"]["max_seconds"]:
            raise ScreenIncomplete("Frozen numerical time budget exhausted")

    status = "COMPLETED"
    try:
        with torch.no_grad():
            for spec in protocol["plans"]["cases"]:
                check()
                _run_case(protocol, spec, tables, check, cache)
                tables["frontier_rows"] = select_frontiers(protocol, tables["candidate_rows"], tables["reference_rows"])
                validate_tables(protocol, tables, complete=False)
                _publish(protocol, run_dir, tables, "RUNNING", errors, clock()-started)
                if progress:
                    progress(spec["case_id"], tables)
        check()
        validate_tables(protocol, tables, complete=True)
    except (Exception, KeyboardInterrupt) as error:
        status = "INCOMPLETE" if isinstance(error, (ScreenIncomplete, KeyboardInterrupt, InterruptedError)) else "FAILED"
        errors.append(f"{type(error).__name__}: {error}")
    return _publish(protocol, run_dir, tables, status, errors, clock()-started)
