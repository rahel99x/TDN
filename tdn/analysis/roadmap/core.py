"""Measured experiment rows, explicit missing evidence and reproducible scores.

The score is evidence attainment, not a probability or an accuracy estimate.
Numerical tests can support or refute a claim on their declared cases; they do
not prove a theorem. An NA row has score 1 and zero evidence coverage.
"""
from __future__ import annotations

from collections import Counter
import csv
import json
import math
import os
from pathlib import Path
import statistics
import time

from tdn.research.protocol import digest

WEIGHTS = {"correctness": 2, "math": 2, "gap": 3, "utility": 3}
RELATIONS = {"le": lambda a, b: a <= b, "ge": lambda a, b: a >= b,
             "eq": lambda a, b: a == b, "lt": lambda a, b: a < b,
             "gt": lambda a, b: a > b}


def _finite(value):
    return not isinstance(value, float) or math.isfinite(value)


def clean(value):
    """JSON never represents infinity as an established bound."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return clean(value.item())
    return value


def check(check_id, measured, target, relation="le", *, category="gap", units="",
          required=True, applicable=True, reason="", evidence_kind="numerical_test"):
    measured = measured.item() if hasattr(measured, "item") else measured
    target = target.item() if hasattr(target, "item") else target
    if not isinstance(check_id, str) or not check_id or category not in WEIGHTS:
        raise ValueError("A check needs a stable ID and a declared category")
    if relation not in RELATIONS or type(required) is not bool or type(applicable) is not bool:
        raise ValueError("Invalid check relation or applicability")
    raw_finite = _finite(measured) and _finite(target)
    if not applicable or measured is None or target is None:
        verdict = "NA"
        reason = reason or "No applicable measured evidence or valid target"
    elif not raw_finite:
        verdict, reason = "BAD", reason or "Nonfinite measured value or target"
    else:
        try:
            verdict = "GOOD" if bool(RELATIONS[relation](measured, target)) else "BAD"
        except (TypeError, ValueError) as error:
            raise ValueError("Measured and target values cannot be compared") from error
    return dict(check_id=check_id, category=category, measured=clean(measured), target=clean(target),
                relation=relation, units=units, required=required, applicable=applicable,
                verdict=verdict, score_1_100=100 if verdict == "GOOD" else 1,
                weight=WEIGHTS[category], reason=reason, evidence_kind=evidence_kind,
                nonfinite=not raw_finite, proof_status="NOT_A_PROOF")


def score_checks(checks):
    required = [v for v in checks if v["required"]]
    # Inapplicable required checks remain visible in the denominator. They
    # cannot disappear to improve a score or manufacture complete evidence.
    total = sum(v["weight"] for v in required)
    passed = sum(v["weight"] for v in required if v["verdict"] == "GOOD")
    observed = sum(v["weight"] for v in required if v["verdict"] != "NA")
    if any(v["verdict"] == "BAD" for v in required):
        verdict = "BAD"
    elif not required or any(v["verdict"] == "NA" for v in required):
        verdict = "NA"
    else:
        verdict = "GOOD"
    return dict(verdict=verdict, score_1_100=1 + round(99 * passed / total) if total else 1,
                evidence_coverage=observed / total if total else 0.,
                required_checks=len(required), passed_checks=sum(v["verdict"] == "GOOD" for v in required),
                failed_checks=sum(v["verdict"] == "BAD" for v in required),
                na_checks=sum(v["verdict"] == "NA" for v in required),
                score_meaning="fraction of predeclared required check weight met, mapped to 1..100; NA earns no credit",
                proof_status="NOT_A_PROOF")


def _category(checks, category):
    return score_checks([v for v in checks if v["category"] == category])


def validate_row(row):
    if row.get("schema") != "tdn.roadmap-experiment/v1" or not row.get("experiment_id"):
        raise ValueError("Invalid experiment row identity")
    seen = set()
    for item in row["checks"]:
        if item["check_id"] in seen:
            raise ValueError("Duplicate check ID conceals an experimental requirement")
        seen.add(item["check_id"])
        rebuilt = check(item["check_id"], item["measured"], item["target"], item["relation"],
            category=item["category"], units=item["units"], required=item["required"],
            applicable=item["applicable"], reason=item["reason"], evidence_kind=item["evidence_kind"])
        if item.get("nonfinite"):
            # The JSON retains an explicit failure marker instead of NaN/Inf.
            rebuilt.update(nonfinite=True, verdict="BAD" if item["applicable"] else "NA", score_1_100=1)
        if rebuilt != item:
            raise ValueError(f"Check was not reproducibly scored: {item['check_id']}")
    if row["assessment"] != score_checks(row["checks"]):
        raise ValueError("Experiment score differs from its required evidence")
    if sorted(row.get("required_check_ids", [])) != sorted(v["check_id"] for v in row["checks"] if v["required"]):
        raise ValueError("Experiment omits or adds a declared required check")
    if not {"math", "gap"} <= {v["category"] for v in row["checks"] if v["required"]}:
        raise ValueError("Every experiment must state mathematical and gap evidence, including explicit NA")
    if row["gap_assessment"] != _category(row["checks"], "gap"):
        raise ValueError("Gap score differs from logged measurements")
    if row["math_assessment"] != _category(row["checks"], "math"):
        raise ValueError("Math score differs from logged measurements")
    if row.get("row_sha256") != digest({k: v for k, v in row.items() if k != "row_sha256"}):
        raise ValueError("Experiment row hash differs")
    json.dumps(row, allow_nan=False)
    return row


class Context:
    def __init__(self, protocol, stage, path, prerequisites, device, budget):
        self.protocol, self.stage = protocol, stage
        self.path, self.run_dir = Path(path), Path(path).parent
        self.prerequisites = {k: Path(v) for k, v in prerequisites.items()}
        self.device, self.budget = device, budget
        self.rows, self.identities = [], set()
        self._last = time.monotonic()
        self.path.mkdir(parents=True, exist_ok=True)

    def measure(self, call, repeats=None, warmup=0):
        import torch
        repeats = self.protocol.get("timing_repeats", 1) if repeats is None else repeats
        if type(repeats) is not int or repeats < 1 or warmup < 0:
            raise ValueError("Timing needs positive repeats and nonnegative warmup")
        gpu = torch.device(self.device).type == "cuda"
        sync = lambda: torch.cuda.synchronize(self.device) if gpu else None
        for _ in range(warmup):
            self.budget.check(); call(); sync()
        if gpu:
            if hasattr(self.budget, "observe"):
                self.budget.observe(force=True)
            torch.cuda.reset_peak_memory_stats(self.device)
        values = []
        for _ in range(repeats):
            self.budget.check(); sync(); start = time.perf_counter()
            answer = call(); sync(); values.append(time.perf_counter() - start)
            if gpu and hasattr(self.budget, "observe"):
                self.budget.observe(force=True)
            self.budget.check()
        ordered = sorted(values)
        return answer, dict(elapsed_seconds=sum(values), median_seconds=statistics.median(values),
            total_seconds=sum(values), min_seconds=min(values), max_seconds=max(values),
            p95_seconds=ordered[max(0, math.ceil(.95 * len(values)) - 1)],
            repeats=repeats, warmup=warmup, samples_seconds=values,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device) if gpu else None,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device) if gpu else None,
            memory_scope="CUDA allocator whole-process peak during measured call" if gpu else "CPU peak not isolated by this timer")

    def record(self, experiment_id, mechanism_ids, *, combination_ids=(), metrics=None,
               checks=None, config=None, evidence=None, status="COMPLETED"):
        if experiment_id in self.identities:
            raise ValueError(f"Repeated experiment identity {experiment_id}")
        mids, cids = list(mechanism_ids), list(combination_ids)
        if not mids or any(v not in self.protocol["mechanisms"] for v in mids):
            raise ValueError("Every row must map to declared mechanisms")
        if any(v not in self.protocol["combinations"] for v in cids):
            raise ValueError("Undeclared combination")
        checks = list(checks or [check("no-measured-evidence", None, None, category="gap")])
        for category in ("math", "gap"):
            if not any(v["category"] == category and v["required"] for v in checks):
                checks.append(check(f"unmeasured-{category}", None, None, category=category,
                    reason=f"This experiment does not establish {category} evidence; retained as NA"))
        now = time.monotonic()
        metrics = clean(metrics or {})
        metrics.setdefault("wall_seconds", now - self._last)
        metrics.setdefault("parameters", None)
        cost = metrics.get("cost") if isinstance(metrics.get("cost"), dict) else {}
        for name in ("median_seconds", "peak_allocated_bytes", "peak_reserved_bytes"):
            metrics.setdefault(name, cost.get(name))
        row = dict(schema="tdn.roadmap-experiment/v1", experiment_id=str(experiment_id), stage=self.stage,
            mechanism_ids=mids, combination_ids=cids, status=status, device=self.device,
            profile=self.protocol["profile"], protocol_sha256=digest(self.protocol),
            effective_config=clean(config or {}), metrics=metrics, checks=clean(checks),
            required_check_ids=[v["check_id"] for v in checks if v["required"]],
            evidence=clean(evidence or []), assessment=score_checks(checks),
            gap_assessment=_category(checks, "gap"), math_assessment=_category(checks, "math"),
            scope="finite declared experiment; numerical support is not a general proof")
        row["row_sha256"] = digest(row)
        validate_row(row)
        with (self.path / "rows.jsonl").open("a") as handle:
            handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            handle.flush()
        self.rows.append(row); self.identities.add(experiment_id); self._last = now
        count = len(self.rows)
        if count <= 3 or count % self.protocol.get("console_every", 100) == 0:
            print(f"TDN roadmap {self.stage}: {count} experiments; latest {experiment_id}: "
                  f"{row['assessment']['verdict']} {row['assessment']['score_1_100']}/100; "
                  f"evidence {row['assessment']['evidence_coverage']:.0%}", flush=True)
            from tdn.reporting import emit
            emit({"experiment_rows": count, "latest_score_1_100": row["assessment"]["score_1_100"],
                  "latest_evidence_coverage": row["assessment"]["evidence_coverage"],
                  "latest_wall_seconds": metrics["wall_seconds"]}, phase="roadmap/" + self.stage,
                 step=count)
        return row


def write_reviews(path, rows):
    """One CSV line per experiment and a short complete verdict inventory."""
    path = Path(path)
    columns = ["experiment_id", "stage", "mechanisms", "combinations", "status", "verdict", "score_1_100",
        "evidence_coverage", "gap_verdict", "math_verdict", "parameters", "median_seconds", "wall_seconds",
        "peak_allocated_bytes", "failed_checks", "na_checks", "effective_config", "metrics", "row_sha256"]
    with (path / "review.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader()
        for row in rows:
            a, m = row["assessment"], row["metrics"]
            writer.writerow(dict(experiment_id=row["experiment_id"], stage=row["stage"],
                mechanisms=" ".join(row["mechanism_ids"]), combinations=" ".join(row["combination_ids"]),
                status=row["status"], verdict=a["verdict"], score_1_100=a["score_1_100"],
                evidence_coverage=a["evidence_coverage"], gap_verdict=row["gap_assessment"]["verdict"],
                math_verdict=row["math_assessment"]["verdict"], parameters=m.get("parameters"),
                median_seconds=m.get("median_seconds"), wall_seconds=m.get("wall_seconds"),
                peak_allocated_bytes=m.get("peak_allocated_bytes"),
                failed_checks="; ".join(v["check_id"] for v in row["checks"] if v["verdict"] == "BAD"),
                na_checks="; ".join(v["check_id"] for v in row["checks"] if v["verdict"] == "NA"),
                effective_config=json.dumps(row["effective_config"], separators=(",", ":")),
                metrics=json.dumps(m, separators=(",", ":")), row_sha256=row["row_sha256"]))
    # The full campaign has tens of thousands of rows. Preserve the complete
    # JSON projection without whitespace inflation alongside the streamed log.
    temporary = path / "rows.json.partial"
    with temporary.open("w") as handle:
        json.dump({"schema": "tdn.roadmap-rows/v1", "rows": rows}, handle,
                  separators=(",", ":"), allow_nan=False)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path / "rows.json")
    counts = Counter(r["assessment"]["verdict"] for r in rows)
    (path / "review.md").write_text("# Bounded roadmap experiments\n\n"
        f"Experiments: {len(rows)}. GOOD: {counts['GOOD']}; BAD: {counts['BAD']}; NA: {counts['NA']}.\n\n"
        "Every experiment, effective parameter, cost, error and failed/missing check is in `review.csv`; "
        "the full reproducible check operands and row hashes are in `rows.jsonl`. "
        "Scores measure evidence attainment. An NA score of 1 means no credited evidence, not measured poor accuracy. "
        "Numerical checks do not prove a general theorem.\n")
    return dict(counts)
