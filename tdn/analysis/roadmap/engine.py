"""Stage execution, immutable scientific evidence and complete ID aggregation."""
from __future__ import annotations

from collections import Counter
import csv
import json
import os
from pathlib import Path
import time

from tdn.analysis.premix.neural import _RunBudget
from tdn.research.protocol import digest, file_digest
from tdn.runtime.metadata import software_metadata, write_json
from .core import Context, check, clean, score_checks, validate_row, write_reviews
from .protocol import STAGES, validate_protocol

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = "science_manifest.json"
EXCLUDED = {MANIFEST, "stage.json", "workflow-seal.json", "COMPLETED"}


def _inside(value):
    path = Path(value)
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Roadmap artifacts must stay in this project without symlink indirection")
    return path.resolve()


def _seal(protocol, path, stage, prerequisites, source):
    files = {}
    for file in sorted(path.rglob("*")):
        if file.is_symlink():
            raise ValueError("Scientific stage contains a symlink")
        if file.is_file() and file.name not in EXCLUDED and not file.name.endswith(".partial"):
            files[file.relative_to(path).as_posix()] = file_digest(file)
    manifest = dict(schema="tdn.roadmap-science-manifest/v1", stage=stage,
        protocol_sha256=digest(protocol), source_tree_sha256=source,
        prerequisites=prerequisites, artifacts=files)
    write_json(path / MANIFEST, manifest)
    (path / "COMPLETED").write_text(digest(manifest) + "\n")
    return manifest


def verify_science(protocol, path):
    path = _inside(path)
    manifest = json.loads((path / MANIFEST).read_text())
    if manifest.get("schema") != "tdn.roadmap-science-manifest/v1" or manifest.get("protocol_sha256") != digest(protocol):
        raise ValueError("Roadmap prerequisite protocol or manifest differs")
    if (path / "COMPLETED").read_text().strip() != digest(manifest):
        raise ValueError("Roadmap completion marker differs from its science seal")
    for required in ("summary.json", "protocol.json", "rows.jsonl", "rows.json", "review.csv", "review.md", "summary.txt"):
        if required not in manifest.get("artifacts", {}):
            raise ValueError("Roadmap prerequisite has incomplete scientific evidence")
    for name, expected in manifest["artifacts"].items():
        relative = Path(name)
        target = path / relative
        if (relative.is_absolute() or ".." in relative.parts or target.is_symlink()
                or not target.resolve().is_relative_to(path.resolve())
                or any(parent.is_symlink() for parent in target.parents if parent != path and parent.is_relative_to(path))):
            raise ValueError("Unsafe scientific artifact path")
        if file_digest(target) != expected:
            raise ValueError(f"Scientific artifact changed: {name}")
    actual = {file.relative_to(path).as_posix() for file in path.rglob("*")
              if file.is_file() and file.name not in EXCLUDED and not file.name.endswith(".partial")}
    if actual != set(manifest["artifacts"]):
        raise ValueError("Unlisted or missing scientific artifacts")
    summary = json.loads((path / "summary.json").read_text())
    if summary.get("status") != "COMPLETED" or summary.get("stage") != manifest.get("stage"):
        raise ValueError("Only complete scientific execution can satisfy a prerequisite")
    rows = [validate_row(json.loads(line)) for line in (path / "rows.jsonl").read_text().splitlines() if line]
    if json.loads((path / "rows.json").read_text()).get("rows") != rows:
        raise ValueError("Compact and streamed experiment ledgers differ")
    if len(rows) != summary.get("experiment_count") or len({r["experiment_id"] for r in rows}) != len(rows):
        raise ValueError("Experiment ledger completeness differs from summary")
    if any(r["protocol_sha256"] != digest(protocol) or r["stage"] != manifest["stage"]
           or not set(r["mechanism_ids"]) <= set(protocol["mechanisms"])
           or not set(r["combination_ids"]) <= set(protocol["combinations"]) for r in rows):
        raise ValueError("Experiment row belongs to another protocol")
    if manifest["stage"] == "policy":
        from .policy import validate_policy_artifacts
        validate_policy_artifacts(path)
    return manifest


def _aggregate(ctx):
    all_rows, stage_status, hashes = [], {}, {}
    execution = json.loads((ctx.path / "execution.json").read_text()) if (ctx.path / "execution.json").exists() else {}
    lineage = execution.get("prerequisites", {})
    current_source = software_metadata()["source_tree_sha256"]
    for stage in STAGES[:-1]:
        path = ctx.prerequisites.get(stage)
        try:
            if not path:
                raise FileNotFoundError("Stage path is missing")
            if lineage.get(stage, {}).get("verification") in ("INVALID", "MISSING"):
                raise ValueError("Worker rejected the prerequisite execution provenance: " + lineage[stage].get("error", ""))
            if (path / "stage.json").exists() and json.loads((path / "stage.json").read_text()).get("status") != "COMPLETED":
                raise ValueError("Worker did not finish sealing successful execution")
            manifest = verify_science(ctx.protocol, path)
            if manifest["stage"] != stage or manifest["source_tree_sha256"] != current_source:
                raise ValueError("Aggregate prerequisite stage or executable source differs")
            # CLI runs have a second seal binding source/software/task evidence.
            # Direct development-engine tests are marked separately.
            if (path / "execution.json").exists():
                wrapper = json.loads((path / "workflow-seal.json").read_text())
                if wrapper.get("protocol_sha256") != digest(ctx.protocol):
                    raise ValueError("Execution seal protocol differs")
                for filename in ("execution.json", "protocol.json", "stage.json", MANIFEST):
                    if wrapper.get("files", {}).get(filename) != file_digest(path / filename):
                        raise ValueError("Execution seal is incomplete or changed")
            rows = [validate_row(json.loads(line)) for line in (path / "rows.jsonl").read_text().splitlines() if line]
            all_rows.extend(rows)
            stage_status[stage] = {"status": "VERIFIED", "experiments": len(rows)}
            hashes[stage] = file_digest(path / MANIFEST)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            stage_status[stage] = {"status": "MISSING_OR_INVALID", "experiments": 0, "reason": str(error)}
    result = []
    for kind, declared in (("mechanism", ctx.protocol["mechanisms"]), ("combination", ctx.protocol["combinations"])):
        for identity, spec in declared.items():
            related = [row for row in all_rows if identity in row[kind + "_ids"]]
            checks = [dict(item, check_id=f"{row['stage']}/{row['experiment_id']}/{item['check_id']}")
                      for row in related for item in row["checks"]]
            observed_stages = {row["stage"] for row in related}
            for expected_stage in spec["stages"]:
                if expected_stage == "report":
                    continue
                # Existing computational stage alone is not executable evidence
                # for an ID; it needs at least one specific experiment row.
                if expected_stage not in observed_stages:
                    checks.append(check(f"coverage/{identity}/{expected_stage}", None, True, "eq", category="gap",
                        reason="Declared experiment stage has no verified evidence for this ID"))
            if not related:
                checks.append(check(f"coverage/{identity}/math", None, None, category="math", reason="No executed mathematical evidence"))
            assessment = score_checks(checks)
            record = dict(id=identity, kind=kind, name=spec["name"], expected_stages=spec["stages"],
                observed_stages=sorted(observed_stages), experiment_count=len(related),
                **assessment, gap_assessment=score_checks([v for v in checks if v["category"] == "gap"]),
                math_assessment=score_checks([v for v in checks if v["category"] == "math"]),
                experiment_verdicts=dict(Counter(row["assessment"]["verdict"] for row in related)),
                missing_stages=[s for s in spec["stages"] if s != "report" and s not in observed_stages],
                failed_check_ids=[v["check_id"] for v in checks if v["verdict"] == "BAD"],
                na_check_ids=[v["check_id"] for v in checks if v["verdict"] == "NA"],
                gaps=spec.get("gaps", []), mathematical_significance="numerical evidence on declared cases; general proof not established")
            result.append(record)
    write_json(ctx.path / "mechanism_summary.json", {"schema": "tdn.roadmap-coverage/v1", "records": result,
        "stage_status": stage_status, "verified_manifest_sha256": hashes, "expected_mechanisms": 24,
        "expected_combinations": 5, "verified_experiment_count": len(all_rows),
        "scope": "Scores are evidence attainment; unavailable/failed work is retained"})
    with (ctx.path / "experiment_summary.csv").open("w", newline="") as handle:
        names = ("id", "kind", "name", "verdict", "score_1_100", "evidence_coverage", "experiment_count",
                 "gap_verdict", "math_verdict", "missing_stages", "failed_checks", "na_checks")
        writer = csv.DictWriter(handle, fieldnames=names); writer.writeheader()
        for r in result:
            writer.writerow({**{key: r[key] for key in names if key in r},
                "gap_verdict": r["gap_assessment"]["verdict"], "math_verdict": r["math_assessment"]["verdict"],
                "missing_stages": "; ".join(r["missing_stages"]),
                "failed_checks": "; ".join(r["failed_check_ids"]), "na_checks": "; ".join(r["na_check_ids"])})
    # Full experiment evidence remains source-linked rather than duplicated as
    # millions of console lines or conflated with aggregate pseudo-experiments.
    write_json(ctx.path / "experiment_index.json", {"stages": stage_status,
        "sources": {s: str(p / "review.csv") for s, p in ctx.prerequisites.items()},
        "manifest_sha256": hashes, "review_rows": len(all_rows)})
    ctx.record("inventory/all-29-declarations-accounted", ["M22"],
        metrics={"parameters": 0, "verified_experiments": len(all_rows), "declared_ids": len(result),
                 "missing_or_invalid_stages": sum(v["status"] != "VERIFIED" for v in stage_status.values())},
        config={"scope": "report integrity only; mechanism conclusions are in mechanism_summary.json and experiment_summary.csv"},
        checks=[check("complete-mechanism-combination-inventory", len(result), 29, "eq", category="correctness"),
                check("all-source-stages-verified", sum(v["status"] != "VERIFIED" for v in stage_status.values()), 0, "eq", category="gap"),
                check("infrastructure-is-not-mathematical-model-evidence", None, None, category="math")])
    return dict(mechanism_count=24, combination_count=5, verified_experiments=len(all_rows),
                stages=stage_status, mechanism_verdicts=dict(Counter(r["verdict"] for r in result)),
                all_stage_evidence_verified=all(v["status"] == "VERIFIED" for v in stage_status.values()))


def run_stage(protocol, stage, path, *, prerequisites=None, device="cpu", stop=None):
    validate_protocol(protocol)
    if stage not in STAGES or device not in ("cpu", "cuda"):
        raise ValueError("Invalid roadmap stage or device")
    if protocol["profile"] == "full" and stage in ("confirm_prepare", "confirm", "policy", "scaling"):
        if os.environ.get("TDN_EXECUTION_MODE") != "desktop-slurm":
            raise ValueError("Full fresh confirmation requires the native allocated Fedora workflow")
        from tdn.runtime.preflight import verify_runtime
        verify_runtime(device, "roadmap-" + stage)
    path = _inside(path)
    path.mkdir(parents=True, exist_ok=True)
    if any((path / name).exists() for name in ("summary.json", "rows.jsonl", MANIFEST, "COMPLETED")):
        raise FileExistsError("Preserve the previous roadmap stage; use a fresh run directory")
    if (path / "protocol.json").exists() and json.loads((path / "protocol.json").read_text()) != protocol:
        raise ValueError("Worker and engine protocols differ")
    write_json(path / "protocol.json", protocol)
    (path / "rows.jsonl").touch()
    prerequisites = {k: _inside(v) for k, v in (prerequisites or {}).items()}
    software = software_metadata()
    source = software["source_tree_sha256"]
    prior_hashes = {}
    if stage != "report":
        for label in STAGES[:STAGES.index(stage)]:
            if label not in prerequisites:
                raise ValueError(f"Missing required stage {label}")
            manifest = verify_science(protocol, prerequisites[label])
            if manifest["stage"] != label or manifest["source_tree_sha256"] != source:
                raise ValueError("Prerequisite stage/source differs")
            prior_hashes[label] = file_digest(prerequisites[label] / MANIFEST)
    start = time.monotonic()
    budget = _RunBudget(protocol["budgets"][stage]["seconds"], stop, device)
    ctx = Context(protocol, stage, path, prerequisites, device, budget)
    try:
        if stage == "audit":
            from .numerical_audit import run
            extras = run(ctx)
            ctx.record("provenance/scoring-and-complete-declaration", ["M22"],
                metrics={"parameters": 0, "mechanisms": len(protocol["mechanisms"]), "combinations": len(protocol["combinations"])},
                config={"software": software, "source_tree_sha256": source, "scoring": protocol["scoring"]},
                checks=[check("all-24-mechanisms-declared", len(protocol["mechanisms"]), 24, "eq", category="gap"),
                    check("all-five-combinations-declared", len(protocol["combinations"]), 5, "eq", category="gap"),
                    check("no-numerical-proof-claim", protocol["scoring"]["mathematical_proof_claim"], False, "eq", category="correctness")])
        elif stage == "headroom":
            from .classical import run_headroom
            extras = run_headroom(ctx)
        elif stage in ("prepare", "confirm_prepare"):
            from .data import prepare
            extras = prepare(ctx, confirmation=stage == "confirm_prepare")
        elif stage in ("train", "confirm"):
            from . import neural
            extras = getattr(neural, stage)(ctx)
        elif stage == "policy":
            from .policy import run
            extras = run(ctx)
        elif stage == "transfer":
            from .portability import run
            extras = run(ctx)
        elif stage == "scaling":
            from .classical import run_scaling
            extras = run_scaling(ctx)
        else:
            extras = _aggregate(ctx)
        correctness_failures = sum(v["required"] and v["category"] == "correctness" and v["verdict"] == "BAD"
                                   for row in ctx.rows for v in row["checks"])
        if stage == "audit" and correctness_failures:
            raise RuntimeError(f"{correctness_failures} required structural correctness checks failed")
        budget.check()
        if software_metadata()["source_tree_sha256"] != source:
            raise ValueError("Executable source changed during the stage")
        for label, expected in prior_hashes.items():
            verify_science(protocol, prerequisites[label])
            if file_digest(prerequisites[label] / MANIFEST) != expected:
                raise ValueError("Prerequisite changed during execution")
        verdicts = write_reviews(path, ctx.rows)
        summary = dict(schema=protocol["schema"], stage=stage, profile=protocol["profile"], status="COMPLETED",
            device=device, source_tree_sha256=source, protocol_sha256=digest(protocol),
            elapsed_seconds=time.monotonic() - start, experiment_count=len(ctx.rows), verdict_counts=verdicts,
            correctness_failures=correctness_failures,
            scientific_outcome="BOUNDED_MEASURED_EVIDENCE; SEE GOOD/BAD/NA CHECKS", memory=budget.memory,
            details=clean(extras or {}))
        write_json(path / "summary.json", summary)
        text = (f"TDN roadmap {stage}: {summary['status']} | {len(ctx.rows)} experiments | "
            f"GOOD {verdicts.get('GOOD',0)} BAD {verdicts.get('BAD',0)} NA {verdicts.get('NA',0)} | "
            f"{summary['elapsed_seconds']:.2f}s / {protocol['budgets'][stage]['seconds']}s\n"
            f"Review: {path / 'review.csv'}\nChecks: {path / 'rows.jsonl'}\n"
            "Scores are evidence attainment; numerical checks are not general proofs.\n")
        if stage == "report":
            records = json.loads((path / "mechanism_summary.json").read_text())["records"]
            text += "\nID   Result  Score  Coverage  Experiments  Missing stages\n"
            for row in records:
                text += (f"{row['id']:<4} {row['verdict']:<6} {row['score_1_100']:>5} "
                         f"{row['evidence_coverage']:>8.0%} {row['experiment_count']:>12} "
                         f" {','.join(row['missing_stages']) or '-'}\n")
        (path / "summary.txt").write_text(text)
        _seal(protocol, path, stage, prior_hashes, source)
        return summary
    except BaseException as error:
        verdicts = write_reviews(path, ctx.rows)
        status = "INTERRUPTED" if isinstance(error, (TimeoutError, InterruptedError, KeyboardInterrupt)) else "FAILED"
        write_json(path / "summary.json", dict(schema=protocol["schema"], stage=stage, status=status,
            experiment_count=len(ctx.rows), elapsed_seconds=time.monotonic() - start, verdict_counts=verdicts,
            source_tree_sha256=source, error=f"{type(error).__name__}: {error}", scientific_outcome="PARTIAL_EVIDENCE_RETAINED"))
        (path / "summary.txt").write_text(f"TDN roadmap {stage}: {status}; {error}\nPartial experiments: {len(ctx.rows)}\n")
        raise
