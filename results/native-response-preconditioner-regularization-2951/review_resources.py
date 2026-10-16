#!/usr/bin/env python3
"""Review retained P1R costs, publication and process-tree observations.

This is read-only analysis of an exported run, with no numerical model calls.
Use --run EXTRACTED_RUN --output cost-telemetry-review.json.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def numbers(rows, key):
    return [float(row[key]) for row in rows if isinstance(row.get(key), (float, int))
            and not isinstance(row[key], bool) and math.isfinite(row[key])]


def stats(values):
    return dict(count=len(values), minimum=min(values) if values else None,
                median=statistics.median(values) if values else None,
                maximum=max(values) if values else None)


def guard_windows(status):
    points = {(row["command"], row["point"]): datetime.fromisoformat(row["checked_at"]).timestamp()
              for row in status["identity_checks"]}
    return {stage: dict(start=points[(stage, "before")], stop=points[(stage, "after")])
            for stage in ("verify", "study")}


def telemetry(run, prefix, windows):
    index = read(run / prefix / "index.json")
    rows = read_lines(run / prefix / "cpu-memory.jsonl")
    source = next(s for s in index["sources"] if s["family"] == "cpu-memory")
    coverage = source["coverage"]
    assert len(rows) == coverage["samples"]
    assert len({row["sequence"] for row in rows}) == len(rows)
    assert all(a["timestamp"] < b["timestamp"] for a, b in zip(rows, rows[1:]))
    rss = numbers(rows, "rss_bytes")
    result = dict(index_status=source["status"], coverage=coverage,
                  sample_status_counts=dict(Counter(row["status"] for row in rows)),
                  reasons=dict(Counter(row.get("reason", "") for row in rows if row.get("reason"))),
                  rss_bytes=stats(rss), sampled_peak_rss_gib=max(rss) / 2**30,
                  sampled_peak_rss_mib=max(rss) / 2**20,
                  process_count=stats(numbers(rows, "process_count")),
                  cpu_utilization_percent=stats(numbers(rows, "cpu_utilization_percent")),
                  unavailable_cpu_utilization_samples=len(rows)-len(numbers(rows, "cpu_utilization_percent")),
                  current_process_tree_cpu_seconds=stats(numbers(rows, "cpu_seconds")),
                  cpu_seconds_are_total_job_cpu=False,
                  rss_is_method_peak_or_allocation_peak=False,
                  observations_are_gpu_evidence=False,
                  interpretation="Snapshots exclude departed/remote processes; RSS may count shared pages repeatedly. CPU percent, when present, uses one logical CPU as 100%.",
                  stage_guard_windows={})
    for name, window in windows.items():
        subset = [row for row in rows if window["start"] <= row["timestamp"] <= window["stop"]]
        values = numbers(subset, "rss_bytes")
        result["stage_guard_windows"][name] = dict(samples=len(subset),
            sampled_peak_rss_gib=max(values) / 2**30 if values else None,
            cpu_utilization_percent=stats(numbers(subset, "cpu_utilization_percent")))
    return result


def review(run):
    status = read(run / "outputs/native/status.json")
    raven = read(run / "run.json")
    outer = read(run / "summary.json")
    study = read(run / "outputs/study/summary.json")
    costs = read(run / "outputs/study/costs.json")
    selection = read(run / "outputs/study/selection.json")
    assert status["status"] == outer["state"] == study["status"] == "COMPLETED"
    assert status["exit_code"] == outer["exit_code"] == 0
    assert costs["fresh_reference_solves"] == costs["new_timing_bundles"] == 0
    candidates = [c for cell in selection["cells"] for c in cell["candidates"]]
    candidate_sum = math.fsum(c["elapsed_seconds"] for c in candidates)
    cell_sum = math.fsum(c["audit_candidate_seconds"] for c in costs["cells"])
    assert math.isclose(candidate_sum, costs["fitting_seconds"], rel_tol=1e-14, abs_tol=1e-14)
    assert math.isclose(cell_sum, costs["fitting_seconds"], rel_tol=1e-14, abs_tol=1e-14)
    assert 0 <= costs["sensitivity_seconds"] <= costs["fitting_seconds"]
    series = read(run / "outputs/study/regularization_metrics.json")
    published = read_lines(run / "metrics.jsonl")
    recorded = {(row["phase"], row["step"]): row for row in published}
    assert len(recorded) == len(published)
    for row in series:
        original = recorded[("regularization/"+row["phase_suffix"], row["step"])]
        assert original["metrics"] == row["values"]
    windows = guard_windows(status)
    result = dict(schema="tdn.native_regularization.cost_telemetry_review.v1", job_id=str(raven["job_id"]),
        run_id=raven["run_id"], analysis_scope="Retained JSON/JSONL only; no fits, solves or new action timings",
        lifecycle=dict(native_status=status["status"], raven_state=outer["state"], exit_code=status["exit_code"],
            scheduler_outcome=raven["metadata"]["automation"]["scheduler_outcome"],
            startup_job_state=status["allocation"]["job"]["JobState"],
            final_scheduler_accounting_available=False),
        wall_scopes=dict(raven_runtime_seconds=outer["runtime_seconds"], native_coordinator_seconds=status["elapsed_seconds"],
            study_internal_seconds=study["elapsed_seconds"],
            commands={row["command"]: row["elapsed_seconds"] for row in status["commands"]},
            scope="Nested wall-clock scopes; not quantities to add. Command duration includes process startup/teardown beyond internal stage work."),
        cost_accounting=dict(retained_costs=costs, candidate_elapsed_sum=candidate_sum,
            cell_candidate_elapsed_sum=cell_sum, fitting_less_sensitivity_seconds=costs["fitting_seconds"]-costs["sensitivity_seconds"],
            fitting_less_sensitivity_scope="Remainder of measured candidate time after its measured sensitivity subset; not a separate solver benchmark.",
            separate_solver_and_validation_durations_available=False,
            reused_reference_seconds_charged_to_new_audit=False,
            per_cell_original_reference_demands_are_disjoint=False,
            components_are_additive_to_wall_time=False,
            scope="Fitting includes validation and sensitivity. Historical reference/design durations are reused costs, not new work. Overlapping original training prefixes must not be summed."),
        publication=dict(research_metrics_published=status["research_metrics_published"],
            expected_compact_research_rows=len(series), reproduced_compact_research_rows=len(series),
            total_raven_metric_rows=len(published), stage_rows=len(published)-len(series),
            all_research_values_match=True),
        native_worker=telemetry(run, Path("outputs/native/telemetry"), windows),
        raven_outer_supervisor=telemetry(run, Path("telemetry"), windows),
        stage_guard_windows=windows,
        scope_warning="Raven's outer process tree did not observe the Slurm worker: its zero CPU and small constant RSS are supervisor measurements. Use native worker observations for workload graphs, with partial CPU intervals retained.")
    inputs = ["run.json", "summary.json", "metrics.jsonl", "outputs/native/status.json",
              "outputs/study/summary.json", "outputs/study/costs.json", "outputs/study/selection.json",
              "outputs/study/regularization_metrics.json", "telemetry/index.json", "telemetry/cpu-memory.jsonl",
              "outputs/native/telemetry/index.json", "outputs/native/telemetry/cpu-memory.jsonl"]
    result["input_sha256"] = {name: hashlib.sha256((run/name).read_bytes()).hexdigest() for name in inputs}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = review(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(args.output)


if __name__ == "__main__":
    main()
