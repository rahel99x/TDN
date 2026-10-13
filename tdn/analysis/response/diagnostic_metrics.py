"""Source-bound diagnostic metrics and dependency-free Raven reporting.

Raven consumes standard project files; it is never imported by production code.
The exported project lives beside the scientific units, so its live records do
not mutate a completed science seal. Only selected research metadata is copied:
no checkpoint, pickle or numerical replay array is loaded here.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import time

from tdn.analysis.frontier.core import check, clean
from tdn.research.protocol import digest
from tdn.runtime.metadata import write_json

SCHEMA = "tdn.response-diagnostic-metrics/v1"
MAX_METADATA_BYTES = 256 << 20
METADATA_FILES = {
    "training_results.json", "learning_curves.jsonl", "query_plan.json", "attempt.json",
    "screening.json", "fixed_features.json", "fixed_encoder.json", "reference_summary.json",
    "query_registry.json", "checkpoint_reload.json", "summary.json",
    "diagnostic_decision.json", "screening_summary.json", "readiness.json",
    "inference_cost.json",
}
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_STATES = {"CREATED", "SUBMITTED", "PENDING", "RUNNING", "COMPLETED", "FAILED",
           "INTERRUPTED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "UNKNOWN"}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def relative_path(value):
    """Use Raven's exact confined POSIX path vocabulary for every pointer."""
    value = str(value)
    if (not value or len(value) > 4096 or not value.isprintable() or value.startswith("/")
        or any(c in value for c in "\\*?[]")
        or any(part in ("", ".", "..") for part in value.split("/"))):
        raise ValueError("Diagnostic paths must be exact confined relative POSIX paths")
    return value


def _inside(root, value):
    root, path = Path(root).absolute(), Path(value).absolute()
    if ".." in root.parts or ".." in path.parts:
        raise ValueError("Diagnostic artifacts cannot traverse parent directories")
    if not path.is_relative_to(root):
        raise ValueError("Diagnostic artifact is outside its declared root")
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError("Diagnostic artifacts cannot traverse symlinks")
    return path


def read_metadata(path, *, root=None, expected=None):
    """Stable regular-file JSON/JSONL reads, with an optional sealed checksum."""
    path = Path(path)
    _inside(path.parent if root is None else root, path)
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_METADATA_BYTES:
            raise ValueError("Diagnostic metadata must be a bounded regular file")
        raw = stream.read(MAX_METADATA_BYTES + 1)
        after = os.fstat(stream.fileno())
    signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if (len(raw) != before.st_size or signature(before) != signature(after)
            or signature(after) != signature(path.stat(follow_symlinks=False))):
        raise ValueError("Diagnostic metadata changed while being read")
    checksum = hashlib.sha256(raw).hexdigest()
    if expected is not None and checksum != expected:
        raise ValueError("Diagnostic source differs from its science seal")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate diagnostic JSON key")
            result[key] = value
        return result
    def reject(value):
        raise ValueError("Nonfinite diagnostic JSON constant: " + value)
    parse = lambda s: json.loads(s, object_pairs_hook=unique, parse_constant=reject)
    value = [parse(line) for line in raw.splitlines() if line] if path.suffix == ".jsonl" else parse(raw)
    # JSON's exponent overflow is not caught by parse_constant.
    json.dumps(value, allow_nan=False)
    return value, checksum, raw


def stable_id(value):
    value = re.sub(r"[^A-Za-z0-9_.-]", "-", str(value)).strip(".-_")
    if not value or not value[0].isalnum():
        raise ValueError("Diagnostic attempt identity is empty")
    if len(value) > 128:
        value = value[:107] + "-" + hashlib.sha256(value.encode()).hexdigest()[:20]
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError("Invalid diagnostic attempt identity")
    return value


def raven_root(ctx):
    return Path(ctx.path).parent / "raven"


def raven_project(ctx):
    """Page exact attempts before Raven's 256-attempt discovery ceiling."""
    root = raven_root(ctx)
    stages = list(ctx.protocol.get("stages", ctx.protocol["units"]))
    if len(stages) <= 255:
        return root
    if ctx.stage not in stages:
        raise ValueError("Raven attempt belongs to an undeclared diagnostic stage")
    return root / f"page-{stages.index(ctx.stage)//255:03d}"


def attempt_path(ctx):
    root = Path(ctx.path).parent
    return raven_project(ctx) / "runs" / stable_id(root.name + "-" + ctx.stage)


def _reporting_config(ctx):
    return getattr(ctx, "_raven_config", {})


def _project_catalog(ctx):
    root = raven_root(ctx)
    stages = list(ctx.protocol.get("stages", ctx.protocol["units"]))
    rows = []
    for index, stage in enumerate(stages):
        page = f"page-{index//255:03d}" if len(stages) > 255 else None
        unit = ctx.protocol["units"][stage]
        attempt = stable_id(Path(ctx.path).parent.name + "-" + stage)
        rows.append(dict(stage=stage, kind=unit["kind"], trial_id=unit.get("trial_id"),
            page=page, attempt=attempt,
            inventory_path=(page+"/" if page else "")+"runs/"+attempt+"/run.json"))
    write_json(root/"project_catalog.json", dict(schema=_reporting_config(ctx).get("catalog_schema", "tdn.response-raven-catalog/v1"),
        protocol_sha256=digest(ctx.protocol), source_tree_sha256=getattr(ctx,"_raven_sourcehash",None) or ctx.protocol.get("source_tree_sha256"),
        path_base="Raven overview project root; page directories are independent standard projects",
        declared_attempts=len(rows), page_capacity=255, attempts=rows,
        guidance="Select the overview project for complete research metrics; register the named page project for an exact native job attempt"))


def _project_files(root, config=None):
    root = Path(root)
    (root / ".raven" / "contracts").mkdir(parents=True, exist_ok=True)
    (root / "runs").mkdir(exist_ok=True)
    _inside(root, root / ".raven" / "contracts")
    settings = dict(logs={"manifest_file": "logs.json"}, research={"metrics_file": "metrics.jsonl",
        "contract": ".raven/contracts/outputs.v1.json", "interval": 5})
    write_json(root / ".raven" / "config.json", settings)
    # Each attempt has its own exact output contract; shared empty declarations
    # cannot falsely mark an absent scientific output as present.
    write_json(root / ".raven" / "contracts" / "outputs.v1.json",
               {"version": 1, "outputs": []})
    (root / "README.md").write_text(
        "# Response diagnostic Research project\n\n"
        "Register this directory with `:project /absolute/path/to/raven`, then "
        "select an exact job or use `:run select ATTEMPT_ID`. Each native attempt "
        "uses its actual Slurm job ID; the portable aggregate analysis has no "
        "scheduler identity.\n\n"
        "Large campaigns keep native attempts in `page-000/`, `page-001/`, and "
        "later standalone projects, each below Raven's discovery ceiling. "
        "`project_catalog.json` lists every declared attempt and its page. "
        "Register the overview root for complete research tables and register "
        "a page root to bind an exact native job.\n\n"
        "`outputs/` retains relevant research tables, metric definitions and "
        "source checksums. Missing measurements remain missing. Learning curves "
        "are observed histories, not confidence intervals. This diagnostic "
        "does not revise the historical pilot or authorize development.\n",
        encoding="utf-8")
    if config and config.get("readme"):
        (root / "README.md").write_text(config["readme"], encoding="utf-8")


def start_attempt(ctx, job_id=None, sourcehash=None, *, worker_managed=False, config=None):
    """Publish identity after allocation checks, before the worker's preflight.

    A native worker owns its terminal state. Its numerical CLI reuses this
    attempt without resetting the start, progress or attached logs.
    """
    if config is not None:
        ctx._raven_config = dict(config)
    config = _reporting_config(ctx)
    project, path = raven_project(ctx), attempt_path(ctx)
    sourcehash = sourcehash or ctx.protocol.get("source_tree_sha256")
    if sourcehash is not None and (not isinstance(sourcehash,str) or not _HASH.fullmatch(sourcehash)):
        raise ValueError("Source identity must be an actual SHA256")
    if job_id is not None and (not isinstance(job_id, str) or not job_id
                              or not job_id.isprintable() or len(job_id) > 128):
        raise ValueError("Actual scheduler job identity must be printable text")
    if worker_managed and job_id is None:
        raise ValueError("A native worker-managed attempt requires its actual scheduler job identity")
    ctx._raven_sourcehash = sourcehash
    _project_files(raven_root(ctx), config)
    _project_files(project, config)
    _project_catalog(ctx)
    path.mkdir(parents=True, exist_ok=True)
    _inside(project, path)
    for name in ("outputs", "logs", "reports"):
        (path / name).mkdir(exist_ok=True)
    identity = path.name
    unit = ctx.protocol["units"][ctx.stage]
    existing = None
    if (path / "run.json").exists():
        existing, _, _ = read_metadata(path / "run.json", root=project)
        if existing.get("state") == "COMPLETED":
            raise FileExistsError("Preserve completed Raven attempts")
        if existing.get("metadata", {}).get("protocol_sha256") != digest(ctx.protocol):
            raise ValueError("Raven attempt protocol changed")
        if existing.get("job_id") != job_id:
            raise ValueError("A changed scheduler identity requires a fresh attempt")
        metadata = existing.get("metadata", {})
        previous_source = metadata.get("source_tree_sha256")
        if previous_source is not None and sourcehash is not None and previous_source != sourcehash:
            raise ValueError("Raven attempt source identity changed")
        if metadata.get("native_worker_managed"):
            if existing.get("state") != "RUNNING":
                raise FileExistsError("Preserve finalized native worker attempts")
            if previous_source is None and sourcehash is not None:
                metadata["source_tree_sha256"] = sourcehash
            if not worker_managed and not metadata.get("science_body_started"):
                metadata["science_body_started"] = True
                metadata["science_body_start"] = time.time()
                write_json(path / "run.json", existing)
                emit_metric(ctx, {"science_body_started":1}, phase="lifecycle")
            elif metadata.get("source_tree_sha256") != previous_source:
                write_json(path / "run.json", existing)
            os.environ["TDN_RESPONSE_RAVEN_DIR"] = str(path)
            return path
        if worker_managed:
            raise ValueError("A worker-managed attempt must start before the numerical CLI")
    started = existing.get("start") if existing else time.time()
    trial = ctx.protocol.get("trials", {}).get(unit.get("trial_id"), {})
    run = dict(schema="raven.run/v1", project_id=config.get("project_id", "TDN-response-diagnostic"), run_id=identity,
        experiment_id=config.get("experiment_prefix", "response-diagnostic")+"/"+ctx.stage, attempt=1,
        name=config.get("name_prefix", "TDN/response-diagnostic")+"/"+unit["kind"], state="RUNNING", job_id=job_id,
        start=started, paths=dict(metrics="metrics.jsonl", summary="summary.json", log_index="logs.json",
            outputs="outputs", logs="logs", contract="outputs/contract.json"),
        parameters={k:trial[k] for k in config.get("parameter_keys", ("family", "target_kind", "seed", "variant_id", "phase")) if k in trial},
        metadata=dict(stage=ctx.stage, profile=ctx.protocol["profile"], source_tree_sha256=sourcehash,
            protocol_sha256=digest(ctx.protocol), device=str(ctx.device),
            metric_scope=config.get("metric_scope", "Relevant registered Response diagnostic research only"),
            native_worker_managed=bool(worker_managed), science_body_started=not worker_managed,
            device_scope="Requested execution device; reporting does not verify allocation or CUDA",
            source_identity_scope="Supplied source provenance; reporting does not verify executable readiness",
            native_gpu_verification="NOT_INFERRED_FROM_REPORTING"))
    write_json(path / "run.json", run)
    write_json(path / "logs.json", dict(schema="raven.logs/v1", run_id=identity, job_id=job_id,
        logs=[dict(id="research", path="logs/research.log", label="Research progress", group="Research")])
        if job_id is not None else dict(schema="raven.logs/v1", run_id=identity,
        logs=[dict(id="research", path="logs/research.log", label="Research progress", group="Research")]))
    (path / "metrics.jsonl").touch()
    (path / "logs" / "research.log").touch()
    write_json(path / "outputs" / "contract.json", {"version":1, "outputs":[]})
    os.environ["TDN_RESPONSE_RAVEN_DIR"] = str(path)
    emit_metric(ctx, {"execution_started":1}, phase="lifecycle")
    return path


def metric_record(metrics, *, phase="research", step=None, progress=None, t=None):
    """Missing values are omitted, never converted to a successful zero."""
    if not isinstance(metrics, dict) or len(metrics) > 64:
        raise ValueError("Raven metrics require a mapping of at most 64 names")
    values = {}
    for key, value in metrics.items():
        if not isinstance(key, str) or not 1 <= len(key) <= 96 or not key.isprintable():
            raise ValueError("Invalid Raven metric name")
        if value is None:
            continue
        if not finite(value):
            raise ValueError("Raven metric values must be finite numbers, not booleans")
        values[key] = value
    timestamp = time.time() if t is None else t
    if not finite(timestamp) or timestamp < 0:
        raise ValueError("Metric time must be a nonnegative epoch")
    if not isinstance(phase, str) or len(phase) > 160 or phase and not phase.isprintable():
        raise ValueError("Invalid Raven metric phase")
    row = dict(t=timestamp, metrics=values, phase=phase)
    if step is not None:
        if type(step) is not int or not 0 <= step <= (1 << 63)-1:
            raise ValueError("Metric step must be a nonnegative integer")
        row["step"] = step
    if progress is not None:
        if not isinstance(progress, dict) or set(progress) - {"completed", "total", "unit"}:
            raise ValueError("Invalid Raven progress fields")
        completed, total = progress.get("completed"), progress.get("total")
        unit = progress.get("unit", "")
        if (not finite(completed) or not finite(total) or not 0 <= completed <= total or total <= 0
            or not isinstance(unit, str) or len(unit) > 64 or unit and not unit.isprintable()):
            raise ValueError("Raven progress needs 0 <= completed <= total and total > 0")
        row["progress"] = dict(completed=completed, total=total, unit=unit)
    return row


def _append_metric(path, row):
    path = Path(path)
    _inside(path, path / "metrics.jsonl")
    raw = json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
    if len(raw.encode()) > 65536:
        raise ValueError("Raven metric record exceeds its line budget")
    with (path / "metrics.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(raw)
        stream.flush()
    _inside(path, path / "logs" / "research.log")
    with (path / "logs" / "research.log").open("a", encoding="utf-8") as stream:
        stream.write(f"{row['t']:.6f} {row['phase']} step={row.get('step','NA')} "
                     + json.dumps(row["metrics"], sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def emit_metric(ctx, metrics, **kwargs):
    path = attempt_path(ctx)
    if not (path / "run.json").exists():
        return None
    row = metric_record(metrics, **kwargs)
    _append_metric(path, row)
    return row


def emit_live(metrics, **kwargs):
    """Training hook: absent opt-in context leaves legacy execution untouched."""
    value = os.environ.get("TDN_RESPONSE_RAVEN_DIR")
    if not value:
        return None
    path = Path(value)
    if not (path / "run.json").is_file():
        raise ValueError("Raven reporting context has no published exact identity")
    row = metric_record(metrics, **kwargs)
    _append_metric(path, row)
    return row


def register_log(ctx, source, *, label="Scientific log", group="Research"):
    """Copy one explicitly supplied project-owned log; do not scan directories."""
    path, source = attempt_path(ctx), Path(source)
    _inside(Path(ctx.path).parent, source)
    if source.is_symlink() or not source.is_file():
        raise ValueError("Diagnostic log source must be a project-owned regular file")
    entry_id = "log-" + hashlib.sha256(source.name.encode()).hexdigest()[:16]
    destination = path / "logs" / (entry_id + ".log")
    _inside(path, destination)
    shutil.copyfile(source, destination)
    index, _, _ = read_metadata(path / "logs.json", root=path)
    entries = [entry for entry in index["logs"] if entry["id"] != entry_id]
    if len(entries) >= 256:
        raise ValueError("Raven log attachment budget exceeded; use an explicit catalog")
    entries.append(dict(id=entry_id, path=destination.relative_to(path).as_posix(), label=label, group=group))
    index["logs"] = entries
    write_json(path / "logs.json", index)
    return destination


def _copy_metadata(source, destination, names):
    destination.mkdir(parents=True, exist_ok=True)
    entries = []
    for name in sorted(names):
        file = Path(source) / name
        if not file.exists():
            continue
        _, checksum, raw = read_metadata(file, root=source)
        target = destination / relative_path(name)
        _inside(destination, target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        entries.append(dict(path=target.name, source_name=name, sha256=checksum, bytes=len(raw)))
    return entries


def finish_attempt(ctx, summary, *, metadata_files=None):
    """Record science separately from the final native worker lifecycle.

    Unflagged numerical-CLI completion leaves a worker-managed attempt RUNNING.
    Only ``native_worker_finalized`` supplies the terminal worker state, elapsed
    time and exit code after preflight, tests and final source checks.
    """
    path = attempt_path(ctx)
    if not (path / "run.json").exists():
        return None
    run, _, _ = read_metadata(path / "run.json", root=path)
    observed_state = summary.get("status", "UNKNOWN")
    if observed_state not in _STATES:
        observed_state = "UNKNOWN"
    managed = bool(run.get("metadata", {}).get("native_worker_managed"))
    finalized = managed and summary.get("native_worker_finalized") is True
    deferred = managed and not finalized
    state = "RUNNING" if deferred else observed_state
    now = time.time()
    run["state"] = state
    if deferred:
        run.pop("end", None)
    else:
        run["end"] = now
    science = None
    science_summary = Path(ctx.path) / "summary.json"
    if science_summary.is_file():
        science, _, _ = read_metadata(science_summary, root=ctx.path)
    if science is not None:
        body_state, body_seconds = science.get("status", "UNKNOWN"), science.get("elapsed_seconds")
    elif finalized:
        body_state = summary.get("science_body_status", "NOT_STARTED")
        body_seconds = summary.get("science_body_elapsed_seconds")
    elif run.get("metadata", {}).get("science_body_started"):
        body_state, body_seconds = observed_state, summary.get("elapsed_seconds")
    else:
        body_state, body_seconds = "NOT_STARTED", None
    if body_seconds is not None and (not finite(body_seconds) or body_seconds < 0):
        raise ValueError("Science body elapsed time must be nonnegative and finite")
    runtime = (summary.get("native_worker_runtime_seconds") if finalized else
               None if deferred else summary.get("elapsed_seconds"))
    exit_code = (summary.get("native_worker_exit_code") if finalized else
                 None if deferred else 0 if state == "COMPLETED" else None)
    if runtime is not None and (not finite(runtime) or runtime < 0):
        raise ValueError("Worker elapsed time must be nonnegative and finite")
    if exit_code is not None and type(exit_code) is not int:
        raise ValueError("Worker exit code must be an integer")
    run["metadata"]["native_worker_finalized"] = bool(finalized)
    source_dir = path / "outputs" / "science"
    entries = _copy_metadata(ctx.path, source_dir,
        (METADATA_FILES if metadata_files is None else set(metadata_files)) | {"summary.json", "rows.jsonl"})
    # Hashes are provenance tokens; portable pointers refer to copied bytes.
    manifest = Path(ctx.path) / "science_manifest.json"
    if manifest.exists():
        _, _, _ = read_metadata(manifest, root=ctx.path)
        entries += _copy_metadata(ctx.path, source_dir, {"science_manifest.json"})
    write_json(path / "outputs" / "source_index.json", dict(schema="tdn.response-raven-sources/v1",
        sources=[dict(entry, path="outputs/science/"+entry["path"]) for entry in entries],
        scope="Selected metadata only; binary replay is not included"))
    native = dict(schema="raven.summary/v1", id=run["run_id"], name=run["name"], state=state,
        project_id=run["project_id"], experiment_id=run["experiment_id"], attempt=run["attempt"],
        job_id=run.get("job_id"), start=run["start"],
        runtime_seconds=runtime, exit_code=exit_code,
        parameters=run.get("parameters", {}),
        results=dict(experiments=(science or summary).get("experiment_count"),
            scientific_outcome=(science or summary).get("scientific_outcome"), details=(science or summary).get("details"),
            science_body_status=body_state, science_body_elapsed_seconds=body_seconds,
            science_cli_status=observed_state if deferred else None,
            native_worker_error=summary.get("native_worker_error") if finalized else None),
        metadata=dict(run.get("metadata", {}), runtime_scope=(
            "Native worker from startup through preflight, tests, science and final checks; excludes pending allocation time"
            if managed else "Numerical science stage body, not allocation accounting"),
            science_body_runtime_scope="Numerical science stage body only; distinct from native worker elapsed time",
            source_index="outputs/source_index.json", replay_scope="Research metadata; no binary replay"))
    if not deferred:
        native["end"] = now
    write_json(path / "summary.json", clean(native))
    write_json(path / "run.json", run)
    lifecycle = {"stage_body_seconds":body_seconds}
    if deferred:
        lifecycle["science_body_completed"] = int(body_state == "COMPLETED")
    else:
        lifecycle["execution_completed"] = int(state == "COMPLETED")
        if finalized:
            lifecycle["native_worker_seconds"] = runtime
    emit_metric(ctx, lifecycle, phase="lifecycle")
    # These are the launcher's exact documented files, never a directory scan
    # or a guessed scheduler identity. Copying preserves the portable view.
    logs = Path(ctx.path).parent/"logs"
    for suffix in ("out","err"):
        source = logs/(ctx.stage+("-"+run["job_id"] if run.get("job_id") else "")+"."+suffix)
        if source.is_file():
            owner="Scheduler" if run.get("job_id") else "Worker"
            register_log(ctx,source,label=owner+(" stdout" if suffix == "out" else " stderr"),group="Execution")
    os.environ.pop("TDN_RESPONSE_RAVEN_DIR", None)
    return path


def sealed_sources(ctx):
    """Collect selected tables from exact completed ancestors and nothing else."""
    sources, tables = [], []
    for stage, base in sorted(ctx.prerequisites.items()):
        base = Path(base)
        manifest, manifest_hash, _ = read_metadata(base / "science_manifest.json", root=base)
        marker = _inside(base,base/"COMPLETED")
        if not marker.is_file() or marker.stat().st_size > 128:
            raise ValueError("Diagnostic science completion marker must be a bounded regular file")
        if (manifest.get("schema") != "tdn.response-science/v1" or manifest.get("stage") != stage
            or manifest.get("protocol_sha256") != digest(ctx.protocol)
            or manifest.get("kind") != ctx.protocol["units"][stage]["kind"]
            or (ctx.protocol.get("source_tree_sha256") is not None
                and manifest.get("source_tree_sha256") != ctx.protocol["source_tree_sha256"])
            or marker.read_text().strip() != digest(manifest)):
            raise ValueError("Diagnostic metrics require the exact completed science seal")
        for name, expected in sorted(manifest.get("artifacts", {}).items()):
            relative_path(name)
            if name not in METADATA_FILES:
                continue
            payload, checksum, raw = read_metadata(base / name, root=base, expected=expected)
            pointer = "sources/" + relative_path(stage) + "/" + relative_path(name)
            sources.append(dict(source_id=stage+"/"+name, path=pointer, sha256=checksum,
                bytes=len(raw), science_manifest_sha256=manifest_hash,
                source_tree_sha256=manifest.get("source_tree_sha256")))
            tables.append(dict(stage=stage, name=Path(name).name, source_path=pointer,
                sha256=checksum, payload=payload, raw=raw))
    return sources, tables


def _ratio(numerator, denominator):
    return numerator / denominator if finite(numerator) and finite(denominator) and denominator > 0 else None


def write_compact_json(path, value):
    """Atomically stream compiled metadata without redundant pretty-print bytes."""
    import uuid
    path = Path(path)
    temporary = path.with_name(path.name+".tmp-"+uuid.uuid4().hex)
    try:
        with temporary.open("w",encoding="utf-8") as stream:
            json.dump(value,stream,sort_keys=True,separators=(",",":"),allow_nan=False)
            stream.write("\n")
        if temporary.stat().st_size > MAX_METADATA_BYTES:
            raise ValueError("Compiled diagnostic metadata exceeds its bounded file size")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _compact_fit(fit, *, source_path, source_sha256, source_record):
    """Keep scalar initial diagnostics; exact endpoint rows remain in the source."""
    return {key:value for key,value in fit.items() if key != "rows"} | dict(
        endpoint_rows_count=len(fit.get("rows",[])),source_path=source_path,
        source_sha256=source_sha256,source_record=source_record,
        export_scope="Aggregate initialization metrics; complete query rows are in the exact canonical source JSON")


def compact_control(payload):
    """Remove duplicated dense controls, preserving their complete source bytes."""
    vectors=("coefficients","column_scales","singular_values","raw_singular_values")
    dense={"rows","fp64_design_rows","gram",*vectors}
    summary={key:value for key,value in payload.items() if key not in dense}
    summary.update(endpoint_rows_count=len(payload.get("rows",[])),
        fp64_design_rows_count=len(payload.get("fp64_design_rows",[])),
        gram_shape=[len(payload.get("gram",[])),len(payload.get("gram",[[]])[0]) if payload.get("gram") else 0],
        vector_sizes={key:len(payload[key]) for key in vectors if isinstance(payload.get(key),list)},
        export_scope="Scalar fit/conditioning metadata and matrix shapes; complete rows, vectors and Gram values remain in the exact canonical source JSON")
    return summary


def trial_metrics(trial, *, source_path, source_sha256, registered=None):
    """Separate absolute error, common anchor gain and initialization progress."""
    spec = registered or {}
    initial = trial.get("initial_train") or {}
    initial_validation = trial.get("initial_validation") or {}
    fits = {key:trial.get(key) or {} for key in ("selected_train", "final_train", "attainable_train", "criterion_train", "selected_validation")}
    selected = fits["selected_train"]
    result = dict(trial_id=trial.get("trial_id", spec.get("id")),
        family=trial.get("family", spec.get("family")), target_kind=trial.get("target_kind", spec.get("target_kind")),
        variant_id=trial.get("variant_id", spec.get("variant_id")), seed=trial.get("seed", spec.get("seed")),
        phase=trial.get("phase", spec.get("phase")), regime=trial.get("regime", spec.get("regime")),
        status=trial.get("status", "MISSING"), requested_updates=trial.get("requested_updates", spec.get("updates")),
        updates=trial.get("updates"), selected_initialization=trial.get("selected_initialization"),
        requested_training_queries=trial.get("requested_training_queries"),
        accepted_training_queries=trial.get("accepted_training_queries"),
        requested_validation_queries=trial.get("requested_validation_queries"),
        accepted_validation_queries=trial.get("accepted_validation_queries"),
        parameters=trial.get("total_parameters"), trainable_parameters=trial.get("trainable_parameters"),
        examples_seen=trial.get("examples_seen"), work=trial.get("work"), attempts=trial.get("attempts", []),
        learning_curve_export=trial.get("learning_curve_export"),optimization_history_summary=trial.get("optimization_history_summary"),
        query_plan_source_path=(str(Path(source_path).parent/"query_plan.json") if (trial.get("learning_curve_export") or {}).get("query_plan_file") else None),
        query_plan_source_sha256=(trial.get("learning_curve_export") or {}).get("query_plan_file_sha256"),
        inference_cost=trial.get("inference_cost",{}),
        training_controls=trial.get("training_controls"), cost=trial.get("cost"),
        cuda_peak_allocated_bytes=trial.get("cuda_peak_allocated_bytes"),
        cuda_peak_reserved_bytes=trial.get("cuda_peak_reserved_bytes"),
        source_path=source_path, source_sha256=source_sha256,
        source_trial_index=trial.get("_source_trial_index",0),
        objective_contract=trial.get("normalization", {}).get("objective_contract"),
        objective_mode=trial.get("normalization", {}).get("objective_mode"),
        normalization=trial.get("normalization"),
        fits={name:{k:fit.get(k) for k in ("mse", "rms", "maximum", "mean_absolute", "centered_rms", "objective", "physical_criterion")}
              for name,fit in fits.items()},
        initial=_compact_fit(initial,source_path=source_path,source_sha256=source_sha256,source_record=f"/trials/{trial.get('_source_trial_index',0)}/initial_train"),
        initial_validation=_compact_fit(initial_validation,source_path=source_path,source_sha256=source_sha256,source_record=f"/trials/{trial.get('_source_trial_index',0)}/initial_validation"),
        initialization_mse_reduction=_ratio(initial.get("mse"), selected.get("mse")),
        diagnostic_initialization_mse_reduction=_ratio(initial.get("mse"), fits["attainable_train"].get("mse")),
        validation_initialization_mse_reduction=_ratio(initial_validation.get("mse"),fits["selected_validation"].get("mse")),
        failure=trial.get("failure"), reference_coverage_rows=trial.get("reference_coverage_rows", []),
        correction_diagnostics=trial.get("correction_diagnostics", {}))
    initial_rows = {row.get("query_id"):row for row in initial.get("rows", [])}
    validation_initial_rows = {row.get("query_id"):row for row in initial_validation.get("rows", [])}
    queries = []
    for name, fit in fits.items():
        for index, row in enumerate(fit.get("rows", [])):
            record = dict(row)
            rms = row.get("rms")
            anchor_rms = row.get("target_defect_rms")
            record.update(trial_id=result["trial_id"], fit=name,
                anchor_mse=(anchor_rms**2 if finite(anchor_rms) else None),
                anchor_mse_reduction=_ratio(anchor_rms**2 if finite(anchor_rms) else None, row.get("mse")),
                anchor_rms_reduction=_ratio(anchor_rms, rms),
                initialization_mse_reduction=_ratio((validation_initial_rows if name == "selected_validation" else initial_rows).get(row.get("query_id"), {}).get("mse"), row.get("mse")),
                source_path=source_path, source_sha256=source_sha256,
                source_record=f"/trials/{trial.get('_source_trial_index',0)}/{name}/rows/{index}")
            queries.append(record)
    return clean(result), clean(queries)


def _parent_mse(rows):
    groups = {}
    for row in rows:
        if not finite(row.get("mse")) or row.get("parent_id") is None:
            return None
        groups.setdefault(row["parent_id"], []).append(row["mse"])
    return sum(sum(v)/len(v) for v in groups.values())/len(groups) if groups else None


def assess_fit(trial, name, *, regimes, rms_target, max_target, mse_ratio=.01,
               expected_queries_per_regime=None):
    """Keep every requested query and one common checkpoint per fit separate."""
    split = "validation" if name == "selected_validation" else "train"
    fit = trial.get(name) or {}
    coverage = [r for r in trial.get("reference_coverage_rows", []) if r.get("split") == split]
    endpoints = fit.get("rows", [])
    initial = trial.get("initial_validation" if split == "validation" else "initial_train") or {}
    result = []
    for regime in regimes:
        is_individual = trial.get("phase") in ("tiny_individual", "diagnostic_individual")
        belongs = lambda r:(r.get("tiny_regime") or r.get("regime")) == regime or is_individual and trial.get("regime") == regime
        expected = [r for r in coverage if belongs(r)]
        observed = [r for r in endpoints if belongs(r)]
        initial_rows = [r for r in initial.get("rows", []) if belongs(r)]
        declared_count = expected_queries_per_regime
        if declared_count is None and is_individual:
            declared_count = trial.get("requested_validation_queries" if split == "validation" else "requested_training_queries")
        requested = max(len(expected), declared_count or 0)
        expected_ids = [r.get("query_id") for r in expected]
        observed_ids = [r.get("query_id") for r in observed]
        identities_complete = (bool(expected_ids) and None not in expected_ids and None not in observed_ids
            and len(expected_ids) == len(set(expected_ids)) and len(observed_ids) == len(set(observed_ids))
            and set(expected_ids) == set(observed_ids))
        reference_complete = (requested > 0 and len(expected) == requested and len(observed) == requested
            and identities_complete and all(r.get("reference_accepted") is True for r in expected+observed))
        finite_errors = bool(observed) and all(finite(r.get("conservative_rms"))
                                              and finite(r.get("conservative_maximum")) for r in observed)
        absolute = reference_complete and finite_errors and all(r["conservative_rms"] <= rms_target
                                                               and r["conservative_maximum"] <= max_target for r in observed)
        floor_count = sum(r.get("floor_limited") is True for r in observed)
        headroom_observed = bool(observed) and all(type(r.get("floor_limited")) is bool for r in observed)
        initial_mse, achieved_mse = _parent_mse(initial_rows), _parent_mse(observed)
        initial_identities = {r.get("query_id") for r in initial_rows} == set(observed_ids) and len(initial_rows) == len(observed)
        improvement_observed = (initial_identities and finite(initial_mse) and initial_mse > 0
                                and finite(achieved_mse))
        reduction_pass = improvement_observed and achieved_mse <= initial_mse*mse_ratio
        finished = trial.get("status") == "COMPLETED_UPDATES" and trial.get("updates") == trial.get("requested_updates")
        unresolved = not (reference_complete and finite_errors and headroom_observed and improvement_observed and finished) or floor_count > 0
        failed = str(trial.get("status", "")).startswith("FAILED")
        status = "FAILED" if failed else "UNRESOLVED" if unresolved else "PASS" if absolute and reduction_pass else "FAILED"
        absolute_status = "UNRESOLVED" if not reference_complete or not finite_errors else "PASS" if absolute else "FAILED"
        result.append(dict(trial_id=trial.get("trial_id"), fit=name, split=split, regime=regime,
            status=status, absolute_accuracy_status=absolute_status, passed=status == "PASS",
            requested_queries=requested, recorded_reference_queries=len(expected), recorded_metric_queries=len(observed),
            reference_complete=reference_complete, query_identities_complete=bool(identities_complete),
            unresolved_reference_queries=sum(r.get("reference_accepted") is not True for r in expected),
            floor_queries=floor_count, floor_classification="FLOOR_LIMITED" if floor_count else "NON_FLOOR" if headroom_observed else "UNRESOLVED",
            initial_mse=initial_mse, achieved_mse=achieved_mse,
            initialization_mse_reduction=_ratio(initial_mse, achieved_mse), mse_reduction_passed=bool(reduction_pass),
            physical_accuracy_passed=bool(absolute), completed_requested_updates=finished,
            conservative_rms=max((r["conservative_rms"] for r in observed if finite(r.get("conservative_rms"))),default=None),
            conservative_maximum=max((r["conservative_maximum"] for r in observed if finite(r.get("conservative_maximum"))),default=None),
            rms_target=rms_target, max_target=max_target, required_mse_ratio=mse_ratio,
            checkpoint_contract="One entire-bank checkpoint for this named fit; regimes never choose independent iterates"))
    return clean(result)


def prospective_assessments(protocol, raw_trials):
    """A bounded readiness diagnosis never overrides the historical gate."""
    criteria = protocol.get("training", {})
    regimes = list(criteria.get("tiny_regimes", protocol.get("screening", {}).get("required_regimes", [])))
    rms_target, max_target = criteria.get("target_rms", 2e-5), criteria.get("target_max", 2e-5)
    ratio, required = criteria.get("tiny_squared_error_ratio", .01), criteria.get("tiny_required_regimes", 5)
    variants = protocol.get("diagnostic", {})
    primary = set(variants.get("primary_variants", []))
    required_seeds = variants.get("full_primary_seeds", 3)
    phase_count = len(protocol.get("phase_variants", []))
    raw = {trial["trial_id"]:trial for trial in raw_trials}
    cells, groups = [], {}
    for ident, spec in protocol.get("trials", {}).items():
        trial = {**spec,"trial_id":ident,"status":"MISSING",**raw.get(ident,{})}
        trial_regimes = [spec.get("regime")] if spec.get("phase") in ("tiny_individual", "diagnostic_individual") else regimes
        key = (spec.get("target_kind"), spec.get("family"), spec.get("variant_id"))
        group = groups.setdefault(key, dict(target_kind=key[0], family=key[1], variant_id=key[2], primary=key[2] in primary,
            registered_trial_ids=[], seeds={}, required_regimes=required, declared_regimes=regimes,
            required_primary_seeds=required_seeds))
        group["registered_trial_ids"].append(ident)
        seed = group["seeds"].setdefault(str(spec.get("seed")), dict(seed=spec.get("seed"), individual={}, joint={}, validation=[]))
        for name in ("selected_train", "final_train", "attainable_train", "criterion_train", "selected_validation"):
            count = len(protocol.get("validation_horizons" if name == "selected_validation" else "train_horizons", []))*phase_count or None
            fit_cells = assess_fit(trial,name,regimes=trial_regimes,rms_target=rms_target,max_target=max_target,
                                   mse_ratio=ratio,expected_queries_per_regime=count)
            for cell in fit_cells:
                cell.update(target_kind=key[0],family=key[1],variant_id=key[2],seed=spec.get("seed"),phase=spec.get("phase"))
                cells.append(cell)
                if name == "attainable_train":
                    bucket = "individual" if spec.get("phase") in ("tiny_individual", "diagnostic_individual") else "joint"
                    if cell["regime"] in seed[bucket]:
                        raise ValueError("Diagnostic group has duplicate predeclared regime fits")
                    seed[bucket][cell["regime"]] = cell
                elif name == "selected_validation":
                    seed["validation"].append(cell)
    outcomes = []
    for group in groups.values():
        registered_complete = all(ident in raw and raw[ident].get("status") == "COMPLETED_UPDATES"
            and raw[ident].get("updates") == raw[ident].get("requested_updates") for ident in group["registered_trial_ids"])
        opportunities_complete = len(group["seeds"]) >= required_seeds and registered_complete
        seed_rows = []
        for seed in group["seeds"].values():
            seed_result = dict(seed=seed["seed"],individual_passes=sum(v["passed"] for v in seed["individual"].values()),
                joint_passes=sum(v["passed"] for v in seed["joint"].values()),
                individual_regimes_recorded=len(seed["individual"]),joint_regimes_recorded=len(seed["joint"]),
                validation_absolute_passes=sum(v["absolute_accuracy_status"] == "PASS" for v in seed["validation"]),
                validation_cells=len(seed["validation"]))
            complete = set(seed["individual"]) == set(regimes) and set(seed["joint"]) == set(regimes) and len(regimes) == 6
            covered = bool(seed["validation"]) and all(v["reference_complete"] for v in
                [*seed["individual"].values(),*seed["joint"].values(),*seed["validation"]])
            headroom = all(v["floor_classification"] == "NON_FLOOR" for v in
                [*seed["individual"].values(),*seed["joint"].values()])
            seed_result.update(all_registered_reference_queries_covered=covered,all_regimes_have_recorded_headroom=headroom)
            opportunities_complete = opportunities_complete and complete and covered and headroom
            seed_result["prospective_fit_criterion_passed"] = complete and seed_result["individual_passes"] >= required and seed_result["joint_passes"] >= required
            seed_rows.append(seed_result)
        passed = opportunities_complete and all(s["prospective_fit_criterion_passed"] for s in seed_rows)
        outcome = "READY_FOR_FRESH_PILOT_REVIEW" if passed else "EXPLORATORY_ONLY" if not group["primary"] else "INSUFFICIENT_OPPORTUNITY" if not opportunities_complete else "FAILED_OR_UNRESOLVED_FITS"
        outcomes.append({k:v for k,v in group.items() if k != "seeds"} | dict(seed_outcomes=seed_rows,
            all_registered_trials_complete=registered_complete,
            fit_status=outcome, prospective_fresh_pilot_readiness=passed, eligible_for_development=False,
            validation_contract="Validation-selected absolute error is reported separately; no training diagnostic selects deployment"))
    return dict(cells=cells,variant_outcomes=outcomes,unsupported_cells=protocol.get("unsupported_cells", []),
        criteria=dict(required_regimes=required,declared_regimes=regimes,primary_seeds=required_seeds,
            rms=rms_target,maximum=max_target,mse_ratio=ratio),
        historical_gate=variants.get("historical_pilot_decision", "UNCHANGED"),eligible_for_development=False,
        interpretation="Prospective diagnostic fit criteria; all methods, missing opportunities and unresolved queries retained")


def accounting_sources(ctx):
    """Read explicit allocation snapshots; never sum steps with allocations."""
    campaign = Path(ctx.path).parent
    roots = [campaign]
    bridge = campaign/"recovery.json"
    if bridge.is_file():
        recovery, _, _ = read_metadata(bridge,root=campaign)
        from .engine import ROOT
        for value in recovery.get("origin_roots", []):
            roots.append(_inside(ROOT, value))
    sources, tables, snapshots, allocations = [], [], [], []
    missing, jobs = [], set()
    for root in dict.fromkeys(roots):
        file = root/"state"/"scheduler-accounting.json"
        if not file.is_file():
            missing.append(root.name)
            continue
        payload, checksum, raw = read_metadata(file,root=root)
        if payload.get("schema") != "tdn.scheduler-accounting/v1":
            raise ValueError("Unexpected diagnostic scheduler accounting schema")
        pointer = "sources/accounting/"+relative_path(root.name)+"/scheduler-accounting.json"
        sources.append(dict(source_id="accounting/"+root.name,path=pointer,sha256=checksum,bytes=len(raw),
                            evidence_scope="Operational accounting, outside immutable numerical science"))
        tables.append(dict(stage="accounting",name="scheduler-accounting.json",source_path=pointer,
                           sha256=checksum,payload=payload,raw=raw))
        requested = set(payload.get("requested_job_ids", []))
        snapshots.append(dict(run_id=root.name,source_path=pointer,status=payload.get("status"),
            collected_at=payload.get("collected_at"),all_allocations_terminal=payload.get("all_allocations_terminal"),
            requested_job_ids=sorted(requested)))
        for row in payload.get("records", []):
            if row.get("record_kind") != "allocation":
                continue
            ident = row.get("allocation_job_id")
            if ident not in requested:
                raise ValueError("Diagnostic accounting contains an undeclared allocation")
            if ident in jobs:
                raise ValueError("Diagnostic accounting repeats an allocation across snapshots")
            jobs.add(ident)
            allocations.append(dict(row,origin_run=root.name,source_path=pointer,source_sha256=checksum))
    complete = bool(snapshots) and not missing and all(s["status"] == "RECORDED" and s["all_allocations_terminal"] is True for s in snapshots)
    result = dict(status="COMPLETE_SNAPSHOT" if complete else "PARTIAL_OR_NOT_AVAILABLE",snapshots=snapshots,
        missing_runs=missing,allocation_rows=allocations,
        observed_allocation_seconds=sum(r["elapsed_seconds"] for r in allocations if finite(r.get("elapsed_seconds"))) if allocations else None,
        observed_allocated_cpu_seconds=sum(r["allocated_cpu_seconds"] for r in allocations if finite(r.get("allocated_cpu_seconds"))) if allocations else None,
        scope="Slurm allocation records only; task steps excluded. Running reporting job and missing fields remain partial",
        monetary_cost=None,energy_cost=None)
    return result,sources,tables


def collect(ctx):
    sources, tables = sealed_sources(ctx)
    declared = ctx.protocol.get("trials", {})
    latest, histories, screening, fixed_controls, stage_costs, query_plans = {}, [], [], [], [], {}
    for table in tables:
        payload = table["payload"]
        if table["name"] == "training_results.json":
            for index, trial in enumerate(payload.get("trials", [])):
                ident = trial.get("trial_id")
                if ident not in declared:
                    raise ValueError("Diagnostic source contains an unregistered trial")
                prior = latest.get(ident)
                opportunity = (trial.get("updates", 0), len(trial.get("attempts", [])))
                if prior is None or opportunity > prior[0]:
                    latest[ident] = (opportunity, dict(trial, _source_trial_index=index), table)
                elif opportunity == prior[0] and trial != {k:v for k,v in prior[1].items() if k != "_source_trial_index"}:
                    raise ValueError("Equal-opportunity duplicate trials disagree")
        elif table["name"] == "query_plan.json":
            query_plans[table["stage"]] = table
        elif table["name"] == "learning_curves.jsonl":
            histories.extend(dict(row, source_path=table["source_path"], source_sha256=table["sha256"],
                source_record=f"/lines/{index+1}", stage=table["stage"]) for index,row in enumerate(payload))
        elif table["name"] == "screening.json":
            screening.append(dict(stage=table["stage"], source_path=table["source_path"],
                source_sha256=table["sha256"], payload=payload))
        elif table["name"] in ("fixed_features.json", "fixed_encoder.json"):
            fixed_controls.append(dict(stage=table["stage"], source_path=table["source_path"],
                source_sha256=table["sha256"], control=table["name"], payload=compact_control(payload)))
        elif table["name"] == "summary.json":
            stage_costs.append(dict(stage=table["stage"],status=payload.get("status"),kind=payload.get("kind"),
                device=payload.get("device"),elapsed_seconds=payload.get("elapsed_seconds"),memory=payload.get("memory"),
                source_tree_sha256=payload.get("source_tree_sha256"),source_path=table["source_path"],
                source_sha256=table["sha256"],scope="Sealed science-stage body timer; each stage once"))
    rows, queries = [], []
    for ident, spec in declared.items():
        if ident not in latest:
            rows.append(dict(trial_id=ident, family=spec.get("family"), target_kind=spec.get("target_kind"),
                variant_id=spec.get("variant_id"), seed=spec.get("seed"), phase=spec.get("phase"),
                regime=spec.get("regime"), status="MISSING", requested_updates=spec.get("updates"),
                updates=None, source_path=None, source_sha256=None,
                reason="No sealed registered trial result; retained in the denominator"))
            continue
        _, trial, table = latest[ident]
        row, endpoint_rows = trial_metrics(trial, source_path=table["source_path"],
                                          source_sha256=table["sha256"], registered=spec)
        rows.append(row); queries.extend(endpoint_rows)
    # Chunk histories can repeat observations. Preserve an identical observation
    # once, retaining every original source citation rather than double-counting.
    curves = {}
    for row in histories:
        key = (row.get("trial_id"), row.get("update"), row.get("observation_kind", "scheduled"))
        stripped = {k:v for k,v in row.items() if k not in ("source_path", "source_sha256", "source_record", "stage")}
        citation = {k:row[k] for k in ("source_path", "source_sha256", "source_record", "stage")}
        if key in curves:
            if curves[key][0] != stripped:
                raise ValueError("Duplicated logical learning observation disagrees")
            curves[key][1].append(citation)
        else:
            curves[key] = (stripped, [citation])
    learning = []
    for row,citations in curves.values():
        links=[]
        checksum=row.get("query_plan_sha256")
        if checksum is not None:
            for citation in citations:
                table=query_plans.get(citation["stage"])
                plan=table["payload"].get("plans",{}).get(checksum) if table else None
                if plan is None or digest(plan) != checksum:
                    raise ValueError("Public learning curve query plan is missing or has a different canonical digest")
                links.append(dict(source_path=table["source_path"],source_sha256=table["sha256"],
                    source_record="/plans/"+checksum,query_plan_sha256=checksum))
        learning.append(dict(row,citations=citations,query_plan_sources=links))
    raw_trials = [value[1] for value in latest.values()]
    assessments = prospective_assessments(ctx.protocol,raw_trials)
    costs, operational_sources, operational_tables = accounting_sources(ctx)
    sources.extend(operational_sources)
    tables.extend(operational_tables)
    failures = []
    for stage, value in getattr(ctx,"stage_failures",{}).items():
        if stage not in {row["stage"] for row in stage_costs}:
            failures.append(dict(stage=stage,details=value,
                scope="Unsealed worker observation; its cost scope is retained and never silently equated to the science timer"))
    source_hashes = {s.get("source_tree_sha256") for s in sources if s.get("source_tree_sha256")}
    if len(source_hashes) > 1:
        raise ValueError("Diagnostic scientific sources mix executable source identities")
    source_hash = next(iter(source_hashes),None) or getattr(ctx,"_raven_sourcehash",None)
    summary = dict(registered_trials=len(declared), observed_trials=len(latest),
        completed_trials=sum(row["status"] == "COMPLETED_UPDATES" for row in rows),
        missing_trials=sum(row["status"] == "MISSING" for row in rows),
        time_censored_trials=sum(row["status"] in ("TIME_CENSORED", "CONTINUATION_REQUIRED") for row in rows),
        failed_trials=sum(str(row["status"]).startswith("FAILED") for row in rows),
        no_headroom_trials=sum(row["status"] in ("NO_HEADROOM", "SCREENING_UNRESOLVED", "UNRESOLVED_SCREEN", "INELIGIBLE_SCREEN") for row in rows),
        initialization_selected=sum(row.get("selected_initialization") is True for row in rows),
        achieved_updates=sum(row["updates"] for row in rows if finite(row.get("updates"))),
        requested_updates=sum(row["requested_updates"] for row in rows if finite(row.get("requested_updates"))),
        endpoint_metric_rows=len(queries),
        complete_reference_coverage_trials=sum(
            row.get("requested_training_queries") is not None
            and row.get("requested_validation_queries") is not None
            and row.get("requested_training_queries") == row.get("accepted_training_queries")
            and row.get("requested_validation_queries") == row.get("accepted_validation_queries") for row in rows))
    return clean(dict(schema=SCHEMA, profile=ctx.protocol["profile"], protocol_sha256=digest(ctx.protocol),source_tree_sha256=source_hash,
        source_path_base="Directory containing diagnostic_metrics.json; all citations are relative",
        independent_unit="Initial field parent; phases, horizons and seeds are paired descendants",
        no_development_authorization=True,
        historical_gate_policy="Original pilot gate and outcomes remain unchanged",
        sources=sources, trials=rows, endpoint_queries=queries, learning_curves=learning,
        screening=screening, fixed_controls=fixed_controls, summary=summary,assessments=assessments,
        costs=dict(stage_bodies=stage_costs,unsealed_attempts=failures,scheduler=costs,
            observed_sealed_stage_body_seconds=sum(r["elapsed_seconds"] for r in stage_costs if finite(r.get("elapsed_seconds"))),
            missing_science_stage_bodies=[stage for stage in ctx.protocol["units"] if stage not in {r["stage"] for r in stage_costs}],
            stage_cost_status="OBSERVED_COMPLETED_ANCESTORS_ONLY; CURRENT_REPORTING_BODY_EXCLUDED",
            scope="Science-stage, trial-cumulative and allocation timers are separate and never added together"))), tables


def write_csv(path, rows, columns=None):
    rows = list(rows)
    columns = columns or sorted({key for row in rows for key in row}) or ["status"]
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            converted = {}
            for key in columns:
                value = row.get(key)
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
                if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                    value = "'" + value
                converted[key] = value
            writer.writerow(converted)


def run(ctx):
    data, tables = collect(ctx)
    for table in tables:
        destination = Path(ctx.path) / table["source_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        _inside(ctx.path, destination)
        destination.write_bytes(table["raw"])
    write_compact_json(Path(ctx.path)/"diagnostic_metrics.json", data)
    write_csv(Path(ctx.path)/"trial_metrics.csv", data["trials"])
    write_csv(Path(ctx.path)/"endpoint_queries.csv", data["endpoint_queries"])
    write_csv(Path(ctx.path)/"learning_curves.csv", data["learning_curves"])
    write_csv(Path(ctx.path)/"fit_assessments.csv", data["assessments"]["cells"])
    write_csv(Path(ctx.path)/"variant_outcomes.csv", data["assessments"]["variant_outcomes"])
    summary = data["summary"]
    decision = dict(schema="tdn.response-diagnostic-decision/v1",
        decision="DIAGNOSTIC_REVIEW_REQUIRED", eligible_for_development=False,
        historical_pilot_gate="UNCHANGED", summary=summary,assessments=data["assessments"],
        reason="Diagnostic implementation and fitting observations do not override the historical gate or constitute fresh confirmation")
    write_json(Path(ctx.path)/"diagnostic_decision.json", decision)
    mechanism = ctx.unit.get("mechanism", "M3")
    if mechanism not in ctx.protocol["mechanisms"]:
        mechanism = "M3" if "M3" in ctx.protocol["mechanisms"] else next(iter(ctx.protocol["mechanisms"]))
    ctx.record("diagnostic/registered-metrics", [mechanism], metrics=summary,
        checks=[check("registered_trial_denominator", len(data["trials"]), len(ctx.protocol.get("trials", {})), "eq", category="correctness"),
                check("requested_updates_complete", summary["achieved_updates"], summary["requested_updates"], "eq", category="gap"),
                check("reference_coverage", summary["complete_reference_coverage_trials"], summary["registered_trials"], "eq", category="gap"),
                check("generalization_not_established", None, None, category="math",
                      reason="Trainability and numerical diagnostics are not a theorem or fresh generalization evidence")],
        evidence=["diagnostic_metrics.json", "diagnostic_decision.json", "trial_metrics.csv", "endpoint_queries.csv", "learning_curves.csv"])
    emit_metric(ctx, {key:value for key,value in summary.items() if finite(value)}, phase="diagnostic-summary")
    return dict(summary=summary, decision=decision["decision"], eligible_for_development=False)
