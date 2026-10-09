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
    if (path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve())
            or any(parent.is_symlink() for parent in path.absolute().parents if parent.is_relative_to(ROOT))):
        raise ValueError("Frontier artifacts must stay in this project without symlink indirection")
    return path.resolve()


def _seal(protocol, path, stage, prerequisites, source, confirmation_partition=None):
    files = {}
    for file in sorted(path.rglob("*")):
        if file.is_symlink():
            raise ValueError("Scientific stage contains a symlink")
        if file.is_file() and file.relative_to(path).as_posix() not in EXCLUDED and not file.name.endswith(".partial"):
            files[file.relative_to(path).as_posix()] = file_digest(file)
    manifest = dict(schema="tdn.frontier-science-manifest/v1", stage=stage,
        protocol_sha256=digest(protocol), source_tree_sha256=source,
        prerequisites=prerequisites, artifacts=files)
    if confirmation_partition is not None:
        manifest["confirmation_partition"] = confirmation_partition
    write_json(path / MANIFEST, manifest)
    (path / "COMPLETED").write_text(digest(manifest) + "\n")
    return manifest


def verify_science(protocol, path, *, confirmation_partition=None):
    validate_protocol(protocol)
    path = _inside(path)
    manifest = json.loads((path / MANIFEST).read_text())
    if manifest.get("schema") != "tdn.frontier-science-manifest/v1" or manifest.get("protocol_sha256") != digest(protocol):
        raise ValueError("Frontier prerequisite protocol or manifest differs")
    if manifest.get("stage") not in STAGES:
        raise ValueError("Frontier manifest names an undeclared stage")
    if manifest.get("confirmation_partition") != confirmation_partition:
        raise ValueError("A confirmation partition cannot satisfy a full-stage prerequisite")
    if confirmation_partition is not None:
        from .partition import validate_partition
        validate_partition(protocol, confirmation_partition)
        if manifest["stage"] != "confirm":
            raise ValueError("Only confirmation may declare an execution partition")
    if (path / "COMPLETED").read_text().strip() != digest(manifest):
        raise ValueError("Frontier completion marker differs from its science seal")
    for required in ("summary.json", "protocol.json", "rows.jsonl", "rows.json", "review.csv", "review.md", "summary.txt"):
        if required not in manifest.get("artifacts", {}):
            raise ValueError("Frontier prerequisite has incomplete scientific evidence")
    for name, expected in manifest["artifacts"].items():
        relative = Path(name)
        target = path / relative
        if (relative.is_absolute() or ".." in relative.parts or target.is_symlink()
                or not target.resolve().is_relative_to(path.resolve())
                or any(parent.is_symlink() for parent in target.parents if parent != path and parent.is_relative_to(path))):
            raise ValueError("Unsafe scientific artifact path")
        if file_digest(target) != expected:
            raise ValueError(f"Scientific artifact changed: {name}")
    if any(file.is_symlink() for file in path.rglob("*")):
        raise ValueError("Scientific stage contains a symlink")
    actual = {file.relative_to(path).as_posix() for file in path.rglob("*")
              if file.is_file() and file.relative_to(path).as_posix() not in EXCLUDED and not file.name.endswith(".partial")}
    if actual != set(manifest["artifacts"]):
        raise ValueError("Unlisted or missing scientific artifacts")
    summary = json.loads((path / "summary.json").read_text())
    if (summary.get("status") != "COMPLETED" or summary.get("stage") != manifest.get("stage")
            or summary.get("schema") != protocol["schema"]
            or summary.get("protocol_sha256") != digest(protocol)
            or summary.get("source_tree_sha256") != manifest.get("source_tree_sha256")):
        raise ValueError("Only complete scientific execution can satisfy a prerequisite")
    if json.loads((path / "protocol.json").read_text()) != protocol:
        raise ValueError("Stored scientific protocol differs from its declaration")
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
    if manifest["stage"] == "confirm" and (confirmation_partition is not None or "confirmation_scope.json" in manifest["artifacts"]):
        from .confirmation import validate_confirmation_coverage
        validate_confirmation_coverage(path, protocol, confirmation_partition)
        scope = json.loads((path / "confirmation_scope.json").read_text())
        binding = scope.get("binding", {})
        if (binding.get("source_tree_sha256") != manifest["source_tree_sha256"]
                or binding.get("prerequisites") != manifest.get("prerequisites")
                or scope.get("measurement_device") != summary.get("device")):
            raise ValueError("Confirmation scope source, prerequisites or device differs from its scientific seal")
    return manifest


def recovery_context_for_report(ctx):
    """Return a freshly verified, narrowly authorized source transition."""
    if not getattr(ctx, "recovery", None):
        return {}
    from .recovery import verify_recovery
    site = None
    if os.environ.get("TDN_EXECUTION_MODE") == "desktop-slurm":
        from tdn.runtime.desktop_slurm import load_profile
        site = load_profile()
    paths = verify_recovery(ctx.recovery, ctx.protocol, software_metadata(), site)
    for label, path in paths.items():
        if label in ctx.prerequisites and Path(ctx.prerequisites[label]).resolve() != path.resolve():
            raise ValueError("Recovery source authorization belongs to another prerequisite path")
    return ctx.recovery


def _aggregate(ctx):
    from .report import run
    return run(ctx)


def run_stage(protocol, stage, path, *, prerequisites=None, device="cpu", stop=None,
              confirmation_partition=None, confirmation_shards=None, recovery=None):
    validate_protocol(protocol)
    if stage not in STAGES or device not in ("cpu", "cuda"):
        raise ValueError("Invalid frontier stage or device")
    if confirmation_partition is not None or confirmation_shards is not None:
        if stage != "confirm" or (confirmation_partition is not None and confirmation_shards is not None):
            raise ValueError("Confirmation execution must be one partition or one complete merge")
    if confirmation_partition is not None:
        from .partition import validate_partition
        confirmation_partition = validate_partition(protocol, confirmation_partition)
    if protocol["profile"] == "full":
        if os.environ.get("TDN_EXECUTION_MODE") != "desktop-slurm":
            raise ValueError("Full fresh confirmation requires the native allocated Fedora workflow")
        if device != ("cuda" if stage in protocol["gpu_stages"] else "cpu"):
            raise ValueError("Full frontier stages require their frozen CPU/GPU device")
        from tdn.runtime.preflight import verify_runtime
        verify_runtime(device, "frontier-" + stage)
    path = _inside(path)
    path.mkdir(parents=True, exist_ok=True)
    if any((path / name).exists() for name in ("summary.json", "rows.jsonl", MANIFEST, "COMPLETED")):
        raise FileExistsError("Preserve the previous frontier stage; use a fresh run directory")
    if (path / "protocol.json").exists() and json.loads((path / "protocol.json").read_text()) != protocol:
        raise ValueError("Worker and engine protocols differ")
    write_json(path / "protocol.json", protocol)
    (path / "rows.jsonl").touch()
    prerequisites = {k: _inside(v) for k, v in (prerequisites or {}).items()}
    required_prior = set(STAGES[:STAGES.index(stage)])
    if stage != "report" and set(prerequisites) != required_prior:
        raise ValueError("Missing or unexpected required stage prerequisites")
    if stage == "report" and not set(prerequisites) <= required_prior:
        raise ValueError("Report prerequisites include an undeclared or future stage")
    for prior in prerequisites.values():
        if path == prior or path.is_relative_to(prior) or prior.is_relative_to(path):
            raise ValueError("Stage and prerequisite science directories must be separate")
    software = software_metadata()
    source = software["source_tree_sha256"]
    inherited = {}
    if recovery is not None:
        from .recovery import verify_recovery, allowed_stage_sources
        site = None
        if os.environ.get("TDN_EXECUTION_MODE") == "desktop-slurm":
            from tdn.runtime.desktop_slurm import load_profile
            site = load_profile()
        inherited_paths = verify_recovery(recovery, protocol, software, site)
        if stage in inherited_paths:
            raise ValueError("Recovery must reuse completed stages, not rerun them")
        for label, prior in inherited_paths.items():
            if label in prerequisites and prerequisites[label] != prior.resolve():
                raise ValueError("Recovery does not authorize this prerequisite path")
        inherited = allowed_stage_sources(recovery)
    prior_hashes = {}
    if stage != "report":
        for label in STAGES[:STAGES.index(stage)]:
            if label not in prerequisites:
                raise ValueError(f"Missing required stage {label}")
            manifest = verify_science(protocol, prerequisites[label])
            expected_source = inherited.get(label, {}).get("source_tree_sha256", source)
            if manifest["stage"] != label or manifest["source_tree_sha256"] != expected_source:
                raise ValueError("Prerequisite stage/source differs")
            if manifest.get("prerequisites") != prior_hashes:
                raise ValueError("Prerequisites belong to different frontier lineages")
            prior_hashes[label] = file_digest(prerequisites[label] / MANIFEST)
    start = time.monotonic()
    budget = _RunBudget(protocol["budgets"][stage]["seconds"], stop, device)
    ctx = Context(protocol, stage, path, prerequisites, device, budget)
    ctx.recovery = recovery
    ctx.recovery_stage_paths = {key: Path(value["path"]) for key, value in inherited.items()}
    ctx.recovery_stage_sources = {key: value["source_tree_sha256"] for key, value in inherited.items()}
    ctx.confirmation_partition = confirmation_partition
    ctx.confirmation_shards = confirmation_shards
    try:
        if stage == "audit":
            from .measurement import run_audit
            extras = run_audit(ctx)
        elif stage == "screen":
            from .screening import run
            extras = run(ctx)
        elif stage in ("prepare", "confirm_prepare"):
            from .data import prepare
            extras = prepare(ctx, confirmation=stage == "confirm_prepare")
        elif stage in ("train", "confirm"):
            from . import neural
            extras = (neural.confirm(ctx, partition=confirmation_partition, shard_dirs=confirmation_shards)
                      if stage == "confirm" else neural.train(ctx))
        elif stage == "scaling":
            from .scaling import run
            extras = run(ctx)
        elif stage == "policy":
            from .policy import run
            extras = run(ctx)
        else:
            extras = _aggregate(ctx)
        correctness_failures = sum(v["required"] and v["category"] == "correctness" and v["verdict"] == "BAD"
                                   for row in ctx.rows for v in row["checks"])
        if stage == "audit" and correctness_failures:
            raise RuntimeError(f"{correctness_failures} required structural correctness checks failed")
        budget.check()
        if software_metadata()["source_tree_sha256"] != source:
            raise ValueError("Executable source changed during the stage")
        if json.loads((path / "protocol.json").read_text()) != protocol:
            raise ValueError("Scientific protocol changed during the stage")
        if recovery is not None:
            recovery_context_for_report(ctx)
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
        text = (f"TDN frontier {stage}: {summary['status']} | {len(ctx.rows)} experiments | "
            f"GOOD {verdicts.get('GOOD',0)} BAD {verdicts.get('BAD',0)} NA {verdicts.get('NA',0)} | "
            f"{summary['elapsed_seconds']:.2f}s / {protocol['budgets'][stage]['seconds']}s\n"
            f"Review: {path / 'review.csv'}\nChecks: {path / 'rows.jsonl'}\n"
            "Scores are evidence attainment; numerical checks are not general proofs.\n")
        if stage == "report" and (path / "gate_summary.json").exists():
            text += "\nGate assessments: " + str(path / "gate_summary.json") + "\n"
        (path / "summary.txt").write_text(text)
        _seal(protocol, path, stage, prior_hashes, source, confirmation_partition)
        return summary
    except BaseException as error:
        verdicts = write_reviews(path, ctx.rows)
        status = "INTERRUPTED" if isinstance(error, (TimeoutError, InterruptedError, KeyboardInterrupt)) else "FAILED"
        write_json(path / "summary.json", dict(schema=protocol["schema"], stage=stage, status=status,
            experiment_count=len(ctx.rows), elapsed_seconds=time.monotonic() - start, verdict_counts=verdicts,
            source_tree_sha256=source, error=f"{type(error).__name__}: {error}", scientific_outcome="PARTIAL_EVIDENCE_RETAINED"))
        (path / "summary.txt").write_text(f"TDN frontier {stage}: {status}; {error}\nPartial experiments: {len(ctx.rows)}\n")
        raise
