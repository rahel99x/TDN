"""Bounded resolve execution and strict immutable science manifests."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

from tdn.analysis.premix.neural import _RunBudget
from tdn.research.protocol import file_digest
from tdn.runtime.metadata import software_metadata, write_json
from .core import Context, clean, validate_row, write_reviews
from .protocol import digest, validate_protocol

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = "science_manifest.json"
EXCLUDED = {MANIFEST, "stage.json", "execution.json", "workflow-seal.json", "COMPLETED"}
REQUIRED = {"summary.json", "protocol.json", "rows.jsonl", "rows.json", "review.csv", "review.md", "summary.txt"}


def software_identity(software=None):
    software = software_metadata() if software is None else software
    return {key: software.get(key) for key in ("python", "executable", "venv", "torch", "numpy", "scipy",
            "torch_cuda_runtime", "git_commit", "source_tree_sha256")}


def _inside(value):
    path = Path(value)
    if (path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()) or
        any(p.is_symlink() for p in path.absolute().parents if p.is_relative_to(ROOT))):
        raise ValueError("Resolve artifacts must stay inside this project without symlinks")
    return path.resolve()


def inventory(path):
    result = {}
    for file in sorted(path.rglob("*")):
        if file.is_symlink():
            raise ValueError("Resolve science contains a symlink")
        relative = file.relative_to(path).as_posix()
        if file.is_file() and relative not in EXCLUDED:
            result[relative] = file_digest(file)
    return result


def verify_science(protocol, path, *, source_tree_sha256=None):
    validate_protocol(protocol)
    path = _inside(path)
    if (path / MANIFEST).is_symlink() or (path / "COMPLETED").is_symlink():
        raise ValueError("Unsafe resolve manifest or marker")
    manifest = json.loads((path / MANIFEST).read_text())
    stage = manifest.get("stage")
    if (manifest.get("schema") != "tdn.resolve-science/v1" or stage not in protocol["units"] or
        manifest.get("protocol_sha256") != digest(protocol)):
        raise ValueError("Resolve manifest protocol or unit differs")
    unit = protocol["units"][stage]
    if manifest.get("kind") != unit["kind"] or manifest.get("unit_sha256") != digest(unit):
        raise ValueError("Resolve unit scope differs")
    if source_tree_sha256 is not None and manifest.get("source_tree_sha256") != source_tree_sha256:
        raise ValueError("Resolve source differs from expected source")
    if manifest.get("software", {}).get("source_tree_sha256") != manifest.get("source_tree_sha256"):
        raise ValueError("Resolve software identity differs from source")
    if (path / "COMPLETED").read_text().strip() != digest(manifest):
        raise ValueError("Resolve completion marker differs")
    if not REQUIRED <= set(manifest.get("artifacts", {})):
        raise ValueError("Resolve prerequisite has incomplete evidence")
    for name in manifest["artifacts"]:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name:
            raise ValueError("Unsafe resolve artifact path")
    if inventory(path) != manifest["artifacts"]:
        raise ValueError("Resolve scientific artifact inventory or bytes changed")
    if json.loads((path / "protocol.json").read_text()) != protocol:
        raise ValueError("Stored resolve declaration differs")
    summary = json.loads((path / "summary.json").read_text())
    if (summary.get("status") != "COMPLETED" or summary.get("schema") != protocol["schema"] or
        summary.get("stage") != stage or summary.get("protocol_sha256") != digest(protocol) or
        summary.get("source_tree_sha256") != manifest.get("source_tree_sha256") or
        summary.get("device") != manifest.get("device")):
        raise ValueError("Only completed correctly bound science satisfies a prerequisite")
    expected_prior = set(unit["dependencies"])
    actual_prior = set(manifest.get("prerequisites", {}))
    if not (actual_prior <= expected_prior if unit["kind"] == "report" else actual_prior == expected_prior):
        raise ValueError("Resolve prerequisite scope differs")
    rows = [validate_row(json.loads(line)) for line in (path / "rows.jsonl").read_text().splitlines() if line]
    if json.loads((path / "rows.json").read_text()).get("rows") != rows:
        raise ValueError("Resolve streamed and compact ledgers differ")
    if len(rows) != summary.get("experiment_count") or len({r["experiment_id"] for r in rows}) != len(rows):
        raise ValueError("Resolve ledger count or identity differs")
    if any(r["protocol_sha256"] != digest(protocol) or r["stage"] != stage or
           not set(r["mechanism_ids"]) <= set(protocol["mechanisms"]) or r["combination_ids"] for r in rows):
        raise ValueError("Resolve row belongs to another scope")
    return manifest


def _dispatch(ctx):
    kind = ctx.unit["kind"]
    if kind == "audit":
        from tdn.analysis.advance.diagnostics import audit
        return audit(ctx)
    if kind == "prepare":
        from tdn.analysis.advance.data import prepare
        return prepare(ctx)
    import importlib
    module = importlib.import_module("tdn.analysis.resolve." + kind)
    return module.run(ctx)


def run_stage(protocol, stage, path, *, prerequisites=None, device="cpu", stop=None,
              resume=False, stage_failures=None):
    validate_protocol(protocol)
    if stage not in protocol["units"] or device not in ("cpu", "cuda"):
        raise ValueError("Invalid resolve stage or device")
    unit = protocol["units"][stage]
    if protocol["profile"] == "full":
        if os.environ.get("TDN_EXECUTION_MODE") != "desktop-slurm" or device != unit["device"]:
            raise ValueError("Full resolve requires its native allocated Fedora device")
        from tdn.runtime.preflight import verify_runtime
        verify_runtime(device, "resolve-" + stage)
    if resume and unit["kind"] not in ("fit", "fairness"):
        raise ValueError("Only journaled fit and fairness units support interrupted resumption")
    path = _inside(path); path.mkdir(parents=True, exist_ok=True)
    if (path / "COMPLETED").exists() or (path / MANIFEST).exists():
        raise FileExistsError("Preserve completed resolve evidence; use a fresh run")
    if not resume and any((path / name).exists() for name in ("summary.json", "rows.jsonl")):
        raise FileExistsError("Preserve the previous attempt; use a fresh run or verified resumption")
    if (path / "protocol.json").exists() and json.loads((path / "protocol.json").read_text()) != protocol:
        raise ValueError("Worker and engine protocol differ")
    prior = {key: _inside(value) for key, value in (prerequisites or {}).items()}
    required = set(unit["dependencies"])
    if not (set(prior) <= required if unit["kind"] == "report" else set(prior) == required):
        raise ValueError("Missing or unexpected resolve prerequisites")
    software = software_metadata(); source = software["source_tree_sha256"]
    stable_software = software_identity(software)
    hashes = {}; lineage = {}
    for key, base in prior.items():
        if path == base or path.is_relative_to(base) or base.is_relative_to(path):
            raise ValueError("Resolve stages must have separate directories")
        sealed = verify_science(protocol, base, source_tree_sha256=source)
        if sealed["stage"] != key:
            raise ValueError("Resolve prerequisite unit differs")
        if sealed.get("software") != stable_software:
            raise ValueError("Resolve prerequisite software differs")
        hashes[key] = file_digest(base / MANIFEST)
    # Cross-check every ancestor identity visible among immediate prerequisites.
    for key, base in prior.items():
        sealed = json.loads((base / MANIFEST).read_text())
        inherited = {**sealed.get("lineage", {}), key: hashes[key]}
        for name, sha in inherited.items():
            if name in lineage and lineage[name] != sha:
                raise ValueError("Resolve prerequisites mix frozen lineages")
            lineage[name] = sha
    if resume:
        prior_summary = json.loads((path / "summary.json").read_text())
        if (prior_summary.get("stage") != stage or prior_summary.get("protocol_sha256") != digest(protocol) or
            prior_summary.get("source_tree_sha256") != source or prior_summary.get("software") != stable_software or
            prior_summary.get("prerequisites") != hashes or prior_summary.get("device") != device or
            prior_summary.get("status") not in ("FAILED", "INTERRUPTED")):
            raise ValueError("Interrupted scientific attempt has incompatible identity")
    write_json(path / "protocol.json", protocol)
    (path / "rows.jsonl").touch()
    start = time.monotonic(); budget = _RunBudget(unit["seconds"], stop, device)
    ctx = Context(protocol, stage, path, prior, device, budget, resume=resume, stage_failures=stage_failures)
    try:
        extras = _dispatch(ctx)
        correctness = sum(c["required"] and c["category"] in ("correctness", "math") and c["verdict"] == "BAD"
                          for row in ctx.rows for c in row["checks"])
        if unit["kind"] == "audit" and correctness:
            raise RuntimeError(f"{correctness} required resolve mathematical checks failed")
        budget.check()
        if software_metadata()["source_tree_sha256"] != source:
            raise ValueError("Executable source changed during resolve execution")
        for key, base in prior.items():
            verify_science(protocol, base, source_tree_sha256=source)
            if file_digest(base / MANIFEST) != hashes[key]:
                raise ValueError("Resolve prerequisite changed during execution")
        verdicts = write_reviews(path, ctx.rows)
        summary = dict(schema=protocol["schema"], stage=stage, kind=unit["kind"], profile=protocol["profile"],
            status="COMPLETED", device=device, source_tree_sha256=source, protocol_sha256=digest(protocol),
            software=stable_software, prerequisites=hashes,
            elapsed_seconds=time.monotonic()-start, experiment_count=len(ctx.rows), verdict_counts=verdicts,
            correctness_failures=correctness, memory=budget.memory,
            scientific_outcome="DEVELOPMENT_EVIDENCE_ONLY; SEE GOOD/BAD/NA CHECKS", details=clean(extras or {}))
        write_json(path / "summary.json", summary)
        (path / "summary.txt").write_text(f"TDN resolve {stage}: COMPLETED; {len(ctx.rows)} rows; {summary['elapsed_seconds']:.2f}s / {unit['seconds']}s\n"
            f"Verdicts: {verdicts}\nScores measure declared check attainment, not proof or a universal ranking.\n")
        manifest = dict(schema="tdn.resolve-science/v1", stage=stage, kind=unit["kind"], device=device,
            protocol_sha256=digest(protocol), unit_sha256=digest(unit), source_tree_sha256=source,
            prerequisites=hashes, lineage=lineage, software=stable_software, artifacts=inventory(path))
        write_json(path / MANIFEST, manifest)
        (path / "COMPLETED").write_text(digest(manifest)+"\n")
        return summary
    except BaseException as error:
        verdicts = write_reviews(path, ctx.rows)
        status = "INTERRUPTED" if isinstance(error, (TimeoutError, InterruptedError, KeyboardInterrupt)) else "FAILED"
        write_json(path / "summary.json", dict(schema=protocol["schema"], stage=stage, status=status,
            protocol_sha256=digest(protocol), source_tree_sha256=source, device=device,
            software=stable_software, prerequisites=hashes,
            experiment_count=len(ctx.rows), elapsed_seconds=time.monotonic()-start, verdict_counts=verdicts,
            error=f"{type(error).__name__}: {error}", scientific_outcome="PARTIAL_EVIDENCE_RETAINED"))
        (path / "summary.txt").write_text(f"TDN resolve {stage}: {status}; {error}\nPartial rows: {len(ctx.rows)}\n")
        raise
