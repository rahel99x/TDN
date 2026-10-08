"""Target-matched comparisons and paired, reproducible latency measurement.

These are descriptive measurements of project-adapted implementations. A
post-hoc accurate schedule frontier is not an operational error controller.
"""
from __future__ import annotations

from collections import defaultdict
import math
import random
import statistics
import time


def final_time(row):
    """A declared final time is mandatory; never infer it from schedule names."""
    value = row.get("final_time", row.get("horizon"))
    schedule = row.get("schedule")
    if value is None and schedule is not None:
        value = math.fsum(float(h) for h in schedule)
    if value is None or not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError("A positive finite final_time or explicit schedule is required")
    if schedule is not None:
        if not schedule or any(not math.isfinite(float(h)) or float(h) <= 0 for h in schedule):
            raise ValueError("A schedule must contain positive finite time steps")
        if not math.isclose(float(value), math.fsum(schedule), rel_tol=1e-11, abs_tol=1e-14):
            raise ValueError("Declared final_time differs from the explicit schedule")
    # Same convention as immutable reference keys; decimal summation noise
    # must not split 0.1+0.2 and 0.3 into different physical endpoints.
    return float(format(float(value), ".12g"))


def matched_key(row, rms_target, max_target):
    for value in (rms_target, max_target):
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError("Positive finite RMS and maximum targets are required")
    parent = row.get("parent_id", row.get("parent"))
    grid = row.get("grid", row.get("n"))
    track = row.get("track")
    if parent is None or grid is None or track is None:
        raise ValueError("Comparisons require parent, grid, and spatial target")
    grid = tuple(grid) if isinstance(grid, (list, tuple)) else grid
    return (str(parent), grid, str(track), final_time(row), float(rms_target), float(max_target))


def endpoint_eligibility(row, rms_target, max_target):
    """Failures and missing references remain distinct from inaccurate results."""
    if row.get("numerical_failure") or row.get("status") in {"FAILED", "NUMERICAL_FAILURE"}:
        return "NUMERICAL_FAILURE"
    if row.get("reference_accepted") is not True:
        return "REFERENCE_UNACCEPTED"
    keys = ("upper_rms", "upper_max", "cost_seconds")
    if any(row.get(k) is None for k in keys):
        return "MISSING_METRIC"
    if any(not math.isfinite(float(row[k])) or float(row[k]) < 0 for k in keys):
        return "NONFINITE_OR_NEGATIVE_METRIC"
    if float(row["cost_seconds"]) <= 0:
        return "INVALID_COST"
    if row["upper_rms"] > rms_target or row["upper_max"] > max_target:
        return "ACCURACY_INFEASIBLE"
    return "ELIGIBLE"


def paired_frontiers(rows, *, target_pairs, comparators=None):
    """Return all method/control pairs, including infeasible or missing arms.

    ``comparators`` maps output labels to model IDs or lists of IDs. If omitted,
    each observed model is a comparator. Each cell retains the entire observed
    model roster, so an absent model is NA rather than removed from coverage.
    Aggregating seed instances into comparator families requires explicit IDs;
    callers must not choose the best training seed against a fixed candidate.
    """
    rows = list(rows)
    model_ids = sorted({str(r["model_id"]) for r in rows})
    comparators = comparators or {m: [m] for m in model_ids}
    comparators = {k: [v] if isinstance(v, str) else list(v) for k, v in comparators.items()}
    cells = defaultdict(list)
    for row in rows:
        for rt, mt in target_pairs:
            cells[matched_key(row, rt, mt)].append(row)
    result = []
    for (parent, grid, track, horizon, rt, mt), members in cells.items():
        clusters = {str(r["field_cluster"]) for r in members if r.get("field_cluster") is not None}
        if len(clusters) != 1:
            raise ValueError("Each parent cell needs one declared independent field cluster")
        by_model = {m: [r for r in members if str(r["model_id"]) == m] for m in model_ids}
        for model_id, candidates in by_model.items():
            good = [r for r in candidates if endpoint_eligibility(r, rt, mt) == "ELIGIBLE"]
            best = min(good, key=lambda r: r["cost_seconds"]) if good else None
            for label, ids in comparators.items():
                if ids == [model_id]:
                    continue
                controls = [r for r in members if str(r["model_id"]) in ids]
                eligible = [r for r in controls if endpoint_eligibility(r, rt, mt) == "ELIGIBLE"]
                control = min(eligible, key=lambda r: r["cost_seconds"]) if eligible else None
                status = ("ELIGIBLE" if best is not None and control is not None else
                          "BOTH_INFEASIBLE" if best is None and control is None else
                          "CANDIDATE_INFEASIBLE" if best is None else "CONTROL_INFEASIBLE")
                result.append({"model_id": model_id, "comparator": label, "parent_id": parent,
                    "field_cluster": next(iter(clusters)), "grid": grid, "track": track,
                    "final_time": horizon, "rms_target": rt, "max_target": mt,
                    "candidate_feasible": best is not None, "control_feasible": control is not None,
                    "candidate_schedule": best.get("schedule_id") if best else None,
                    "control_schedule": control.get("schedule_id") if control else None,
                    "control_model_id": control["model_id"] if control else None,
                    "candidate_cost_seconds": best["cost_seconds"] if best else None,
                    "control_cost_seconds": control["cost_seconds"] if control else None,
                    "control_over_candidate_speed_ratio": control["cost_seconds"] / best["cost_seconds"] if best and control else None,
                    "candidate_statuses": sorted({endpoint_eligibility(r, rt, mt) for r in candidates}) or ["MISSING_ARM"],
                    "control_statuses": sorted({endpoint_eligibility(r, rt, mt) for r in controls}) or ["MISSING_ARM"],
                    "eligibility": status, "seed": best.get("seed") if best else candidates[0].get("seed") if candidates else None,
                    "scope": "post-hoc reference-informed frozen-schedule frontier; not a deployable policy",
                    "paper_reproduction": False})
    return result


def measure_paired(calls, *, device="cpu", repeats=5, warmup=1, seed=0, budget=None):
    """Separately time first invocation, warmup, and interleaved warm rounds.

    Calls should be deterministic, restart from the same input, and avoid
    persistent mutation. Outputs of the last invocation are returned. The
    ``cold_seconds`` field is explicitly the first invocation of this timer,
    not a guarantee that the process, device or OS cache was cold.
    """
    if not calls or any(not callable(f) for f in calls.values()):
        raise ValueError("At least one named callable is required")
    if type(repeats) is not int or repeats < 2 or type(warmup) is not int or warmup < 1:
        raise ValueError("Paired timing requires at least two rounds and one warmup")
    import torch
    gpu = torch.device(device).type == "cuda"
    def check():
        if budget is not None:
            budget.check()
    def sync():
        if gpu:
            torch.cuda.synchronize(device)
    def invoke(name):
        check(); sync()
        start = time.perf_counter()
        answer = calls[name]()
        sync()
        elapsed = time.perf_counter() - start
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError("Timer did not produce a positive finite duration")
        if gpu and budget is not None and hasattr(budget, "observe"):
            budget.observe(force=True)
        check()
        return answer, elapsed
    rng = random.Random(seed)
    names = list(calls)
    answers, cold, samples = {}, {}, {name: [] for name in names}
    order = names.copy(); rng.shuffle(order)
    cold_order = order.copy()
    for name in order:
        answers[name], cold[name] = invoke(name)
    warmup_rounds = []
    for index in range(warmup):
        order = names.copy(); rng.shuffle(order)
        values = {}
        for name in order:
            answers[name], values[name] = invoke(name)
        warmup_rounds.append({"round": index, "order": order, "samples_seconds": values})
    if gpu:
        torch.cuda.reset_peak_memory_stats(device)
    rounds = []
    for index in range(repeats):
        order = names.copy(); rng.shuffle(order)
        values = {}
        for name in order:
            answers[name], values[name] = invoke(name)
            samples[name].append(values[name])
        rounds.append({"round": index, "order": order, "samples_seconds": values})
    methods = {}
    for name, values in samples.items():
        ordered = sorted(values)
        methods[name] = {"median_seconds": statistics.median(values), "cold_seconds": cold[name],
            "samples_seconds": values, "raw_seconds": values, "min_seconds": min(values),
            "max_seconds": max(values), "p95_seconds": ordered[max(0, math.ceil(.95 * len(values)) - 1)],
            "total_seconds": sum(values), "warmup_seconds": sum(r["samples_seconds"][name] for r in warmup_rounds),
            "repeats": repeats, "warmup": warmup}
    return answers, {"methods": methods, "rounds": rounds, "cold_order": cold_order,
        "cold_samples": cold, "warmup_rounds": warmup_rounds, "repeats": repeats,
        "warmup": warmup, "seed": seed, "device": str(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if gpu else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if gpu else None,
        "memory_scope": "whole paired group, not per-model attribution",
        "timing_scope": "warm randomized interleaved complete calls; first invocation and warmup charged separately",
        "cold_scope": "first invocation in this timing group; no cold-process/cache guarantee"}


def field_cluster_interval(rows, *, value_key="difference", repeats=1000, seed=73421):
    """Descriptive crossed field/seed CI; repeated queries are not new fields."""
    from tdn.analysis.roadmap.statistics import paired_cluster_bootstrap
    return paired_cluster_bootstrap(rows, value_key=value_key, repeats=repeats, random_seed=seed)


def supported_amortization(offline_components, deployed_mean, classical_mean, *,
                           matched_accuracy, accepted_neural_queries, saving_lower):
    """Do not credit all-classical routing or timing noise as neural benefit."""
    from tdn.analysis.roadmap.statistics import amortization_summary
    result = amortization_summary(offline_components, deployed_mean, classical_mean,
                                  matched_accuracy=matched_accuracy)
    if accepted_neural_queries <= 0:
        reason = "NO_ACCEPTED_NEURAL_QUERIES"
    elif saving_lower is None or not math.isfinite(float(saving_lower)) or saving_lower <= 0:
        reason = "SAVING_UNCERTAINTY_INCLUDES_ZERO"
    else:
        reason = None
    if reason is not None:
        result.update(status=reason, break_even_queries=None, fractional_break_even_queries=None)
    result.update(accepted_neural_queries=int(accepted_neural_queries),
                  saving_lower_seconds=saving_lower, statistical_scope="descriptive field-cluster evidence, not future guarantee")
    return result


def run_audit(ctx):
    """Execute small real regression probes before funding new comparisons."""
    from tdn.analysis.roadmap.core import check
    from tdn.analysis.roadmap.portability import independent_reference
    from tdn.runtime.metadata import write_json
    import numpy as np
    common = {"parent_id": "probe", "field_cluster": "field-0", "grid": 8,
              "track": "discrete", "reference_accepted": True, "upper_rms": 1e-8,
              "upper_max": 1e-8, "seed": 0, "schedule_id": "probe"}
    rows = [{**common, "model_id": m, "final_time": t, "cost_seconds": cost}
            for m, t, cost in (("candidate", .1, 2.), ("control", .1, 1.),
                               ("candidate", .3, .01), ("control", .3, 1.))]
    compared = paired_frontiers(rows, target_pairs=[(1e-4, 1e-4)], comparators={"control": ["control"]})
    ratio = next(r["control_over_candidate_speed_ratio"] for r in compared
                 if r["model_id"] == "candidate" and r["final_time"] == .1)
    ctx.record("measurement/fixed-final-time", ["G1"], metrics={"short_horizon_speed_ratio": ratio,
        "distinct_final_times": sorted({r["final_time"] for r in compared})},
        checks=[check("no_cross_horizon_win", ratio, .5, "eq", category="math"),
                check("all_fixed_horizon_cells_retained", len(compared), 2, "eq", category="gap")])
    calls = {name: (lambda value=value: np.sin(np.arange(256, dtype=float) + value).sum())
             for name, value in (("a", 0.), ("b", 1.))}
    _, timing = measure_paired(calls, repeats=3, warmup=1, seed=14, budget=ctx.budget)
    write_json(ctx.path / "paired-timing-probe.json", timing)
    ctx.record("measurement/paired-timing", ["G1"], metrics={"timing": timing},
        checks=[check("warm_samples_per_method", min(len(v["samples_seconds"]) for v in timing["methods"].values()), 3, "eq", category="math"),
                check("first_invocations_separate", len(timing["cold_samples"]), 2, "eq", category="gap")])
    target, info = independent_reference(np.array([.2]), .3, lambda x: 2*x*(1-x), budget=ctx.budget)
    truth = .2*np.exp(.6)/(1+.2*np.expm1(.6))
    error = abs(float(target[0]) - truth)
    ctx.record("measurement/teacher-refinement", ["G1"], metrics={"reference": info, "exact_error": error},
        checks=[check("refined_max_step", info["max_steps"][1], info["max_steps"][0], "lt", category="math"),
                check("independent_method_crosscheck", info["methods"][-1], "RK45", "eq", category="gap"),
                check("roundoff_floor_retained", info["uncertainty_max"], 0., "gt", category="correctness"),
                check("exact_scalar_solution", error, 1e-10, category="correctness")])
    example = [{"field_cluster": f, "seed": s, "difference": 1.+f} for f in range(3) for s in range(2)]
    original = field_cluster_interval(example, repeats=100)
    replicated = field_cluster_interval(example*7, repeats=100)
    ctx.record("measurement/independent-units", ["G1"], metrics={"original": original, "repeated": replicated},
        checks=[check("repeats_do_not_increase_fields", replicated["independent_fields"], 3, "eq", category="math"),
                check("repeated_query_ci_unchanged", replicated["lower"], original["lower"], "eq", category="gap")])
    return {"status": "COMPLETED", "audit_probes": 4, "paper_reproduction": False}
