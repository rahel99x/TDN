"""Execute declared panels, retaining incomplete work without sealing it."""
from __future__ import annotations

from collections import Counter
import time

from .protocol import PANELS, validate_panel
from tdn.runtime.metadata import write_json


class AuditIncomplete(RuntimeError):
    pass


def run(protocol, run_dir, modules, *, stop=None, progress=None, clock=time.monotonic):
    start = clock()
    budget = protocol["config"]["max_seconds"]
    rows, panel_states, errors = [], {}, []

    def check_budget():
        if stop is not None and stop.requested:
            raise AuditIncomplete("Stop requested; preserve this attempt and start a fresh run")
        if clock() - start >= budget:
            raise AuditIncomplete(f"Numerical budget of {budget} seconds exhausted")

    abort = False
    for name in PANELS:
        returned = None
        try:
            if abort:
                raise AuditIncomplete("Not run after an earlier interrupted or failed panel")
            check_budget()
            result = modules[name].run(protocol["config"], check_budget)
            validate_panel(result, name, protocol["plans"][name])
            returned = result
            check_budget()
        except (Exception, KeyboardInterrupt) as error:
            abort = True
            incomplete = isinstance(error, (AuditIncomplete, KeyboardInterrupt, InterruptedError))
            message = f"{type(error).__name__}: {error}"
            errors.append({"panel": name, "error": message})
            result = {"panel": name, "status": "INCOMPLETE" if incomplete else "FAILED",
                      "rows": returned["rows"] if returned is not None else [], "error": message,
                      "unreported_case_ids": [] if returned is not None else protocol["plans"][name]["expected_case_ids"]}
        write_json(run_dir / f"{name}.json", result)
        panel_states[name] = result["status"]
        rows.extend({**row, "panel": name} for row in result["rows"])
        if progress is not None:
            progress(name, result)

    correctness_failures = [f"{row['panel']}/{row['case_id']}" for row in rows
                            if row["kind"] == "correctness" and row["outcome"] == "FAIL"]
    audit_failures = [f"{row['panel']}/{row['case_id']}" for row in rows if row["outcome"] == "FAIL"]
    inconclusive = sum(row["outcome"] == "INCONCLUSIVE" for row in rows)
    status = ("FAILED" if audit_failures or "FAILED" in panel_states.values() else
              "INCOMPLETE" if inconclusive or any(s != "COMPLETED" for s in panel_states.values())
              else "COMPLETED")
    elapsed = clock() - start
    canonical = {"schema": "tdn.mechanism-audit/v1", "benchmark_suite": "mechanism-audit",
                 "status": status, "device": "cpu", "training_attempted": False,
                 "elapsed_seconds": elapsed, "numerical_budget_seconds": budget,
                 "panel_states": panel_states, "rows": rows, "errors": errors,
                 "correctness_failures": correctness_failures, "audit_failures": audit_failures}
    expected = sum(len(protocol["plans"][name]["expected_case_ids"]) for name in PANELS)
    summary = {key: value for key, value in canonical.items() if key != "rows"}
    summary.update(expected_cases=expected, reported_cases=len(rows),
                   outcome_counts=dict(Counter(row["outcome"] for row in rows)),
                   kind_counts=dict(Counter(row["kind"] for row in rows)),
                   scope=protocol["scope"], interpretation=protocol["interpretation"])
    write_json(run_dir / "mechanism-audit.json", canonical)
    write_json(run_dir / "summary.json", summary)
    text = (f"TDN training-free mechanism audit: {status}\n"
            f"Reported cases: {len(rows)}/{expected}; correctness failures: {len(correctness_failures)}\n"
            f"Numerical wall time: {elapsed:.2f}s / {budget}s budget\n"
            f"Outcomes: {summary['outcome_counts']}\n"
            "No neural models trained; oracle fits are representation diagnostics.\n"
            "Inspect panel reports and Tower mechanisms.csv before selecting architecture changes.\n")
    (run_dir / "summary.txt").write_text(text)
    return summary
