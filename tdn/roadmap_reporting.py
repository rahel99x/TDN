"""Complete paged, source-linked roadmap views for the unmodified Tower reader.

Only project-owned report sidecars are written. Scientific inputs remain
immutable, binary checkpoints are never loaded, and an invalid seal cannot
contribute an affirmative score. Canonical rows remain authoritative.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import time
import uuid

from tdn.premix_reporting import _cell as _legacy_cell, _directory, _relative
from tdn.reporting import atomic_json, read_json, register_log
from tdn.research.protocol import digest as canonical_digest

SCHEMA = "tdn.roadmap-rows/v1"
STAGES = ("audit", "headroom", "prepare", "train", "confirm_prepare", "confirm", "policy", "transfer", "scaling", "report")
PAGE_ROWS = 128
PAGE_BYTES = 256 << 10
MAX_SOURCE_BYTES = 1 << 30
MAX_READ_BYTES = 3 << 30
MAX_FILES = 8192
MAX_PAGES = 131072
MAX_OUTPUT_BYTES = 4 << 30
MAX_CELL_BYTES = 8192
MAX_OMISSIONS = 128
PREFIX = ["source_path", "source_record", "source_line", "source_sha256", "row_sha256", "evidence_status", "validation_error"]
COLUMNS = {
    "experiments": PREFIX + ["experiment_id", "stage", "mechanisms", "combinations", "status", "device", "profile",
        "verdict", "score_1_100", "raw_verdict", "raw_score_1_100", "gap_verdict", "gap_score_1_100",
        "math_verdict", "math_score_1_100", "evidence_coverage", "parameters", "median_seconds", "wall_seconds",
        "training_seconds", "peak_allocated_bytes", "peak_reserved_bytes", "family", "seed", "grid", "batch_size",
        "learning_rate", "config_record", "metrics_record", "checks_record", "evidence_record"],
    "checks": PREFIX + ["experiment_id", "stage", "experiment_record", "check_id", "category", "measured", "target",
        "relation", "units", "required", "applicable", "verdict", "score_1_100", "raw_verdict", "weight",
        "reason", "reason_record", "evidence_kind", "nonfinite", "proof_status"],
    "values": PREFIX + ["experiment_id", "stage", "field", "value", "value_type", "value_record", "storage"],
    "mechanisms": PREFIX + ["id", "kind", "name", "verdict", "score_1_100", "raw_verdict", "raw_score_1_100",
        "evidence_coverage", "experiment_count", "gap_verdict", "math_verdict", "expected_stages", "observed_stages",
        "missing_stages", "failed_checks", "na_checks", "failed_checks_record", "na_checks_record", "gaps_record", "mathematical_significance"],
    "artifacts": ["source_path", "bytes", "kind", "sha256", "hash_status", "manifest_path"],
}
_ASCII_CONTROLS = re.compile(r"[\x00-\x1f]")


def _cell(value):
    """Preserve existing CSV escaping without rescanning safe strings in Python."""
    if not isinstance(value, str) or _ASCII_CONTROLS.search(value):
        return _legacy_cell(value)
    return "'" + value if value.lstrip().startswith(("=", "+", "-", "@")) else value


def _signature(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _decode(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    def constant(value):
        raise ValueError("Nonfinite JSON value")
    return json.loads(raw, object_pairs_hook=unique, parse_constant=constant)


def _escaped(value):
    return str(value).replace("~", "~0").replace("/", "~1")


class Budget:
    def __init__(self):
        self.read_bytes = self.output_bytes = self.page_count = self.omission_count = 0
        self.omissions, self.snapshots, self.hashes = [], {}, {}

    def omit(self, source, reason):
        self.omission_count += 1
        if len(self.omissions) < MAX_OMISSIONS:
            self.omissions.append({"source": str(source), "reason": str(reason)})

    def read(self, path, limit=MAX_SOURCE_BYTES):
        path = Path(path)
        _directory(path.parent)
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
                raise ValueError("Source is not a regular bounded file")
            if self.read_bytes + before.st_size > MAX_READ_BYTES:
                raise ValueError("Roadmap aggregate source-read budget exceeded")
            raw = stream.read(limit + 1)
            after = os.fstat(stream.fileno())
        if len(raw) != before.st_size or _signature(before) != _signature(after) or _signature(after) != _signature(path.lstat()):
            raise ValueError("Source changed while reading")
        self.read_bytes += len(raw)
        self.snapshots[path] = _signature(after)
        self.hashes[path] = hashlib.sha256(raw).hexdigest()
        return raw


def is_roadmap_root(root):
    """Recognize only the declared family without importing numerical code."""
    root = Path(root)
    for candidate in (root, *(root / stage for stage in STAGES)):
        for name, token in (("rows.json", b'"tdn.roadmap-rows/v1"'), ("protocol.json", b'"tdn.roadmap/v1"'),
                            ("rows.jsonl", b'"tdn.roadmap-experiment/v1"')):
            path = candidate / name
            try:
                if path.is_symlink() or not path.is_file():
                    continue
                with path.open("rb") as stream:
                    if token in stream.read(4096):
                        return True
            except OSError:
                continue
    return False


def _write(path, raw, budget):
    if budget.output_bytes + len(raw) > MAX_OUTPUT_BYTES:
        raise ValueError("Roadmap report-output budget exceeded; canonical inputs remain unchanged")
    if path.exists() and not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("Roadmap report output must be a regular file")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("xb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    budget.output_bytes += len(raw)


def _csv_bytes(columns, rows, header=True):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, columns, lineterminator="\n")
    if header:
        writer.writeheader()
    for row in rows:
        writer.writerow({key: _cell(row.get(key, "")) for key in columns})
    return buffer.getvalue().encode()


def _pages(output, name, rows, columns, budget, pages, *, jsonl=False):
    """Page complete records by both row count and encoded bytes, never slices."""
    header = b"" if jsonl else _csv_bytes(columns, [])
    pending, count, ordinal, start = [header], 0, 0, 0
    total = 0
    def flush():
        nonlocal pending, count, ordinal, start
        if not count:
            return
        if budget.page_count >= MAX_PAGES:
            raise ValueError("Roadmap page-count ceiling exceeded")
        ordinal += 1
        path = output / f"{name}-{ordinal:05d}.{'jsonl' if jsonl else 'csv'}"
        raw = b"".join(pending)
        _write(path, raw, budget)
        pages.append({"path": "outputs/roadmap/" + path.name, "table": name, "rows": count,
            "start_record": start, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "format": "text" if jsonl else "csv"})
        budget.page_count += 1
        start += count; pending, count = [header], 0
    for row in rows:
        raw = (json.dumps(row, separators=(",", ":"), allow_nan=False).encode() + b"\n") if jsonl else _csv_bytes(columns, [row], header=False)
        if len(raw) + len(header) > PAGE_BYTES or (jsonl and len(raw) > 65536):
            raise ValueError(f"One complete {name} record exceeds Tower's bounded reader; exact source retained")
        if count and (count >= PAGE_ROWS or sum(map(len, pending)) + len(raw) > PAGE_BYTES):
            flush()
        pending.append(raw); count += 1; total += 1
    flush()
    return total


def _leaves(value, pointer="", depth=0):
    if depth > 24:
        yield pointer, value, "source_pointer"
    elif isinstance(value, dict):
        if not value:
            yield pointer, value, "inline"
        for key, item in value.items():
            yield from _leaves(item, pointer + "/" + _escaped(key), depth + 1)
    elif isinstance(value, list):
        if not value:
            yield pointer, value, "inline"
        for index, item in enumerate(value):
            yield from _leaves(item, pointer + "/" + str(index), depth + 1)
    else:
        yield pointer, value, "inline"


def _manifest_files(manifest):
    files = manifest.get("files", manifest.get("artifacts", {}))
    if isinstance(files, list):
        if any(not isinstance(item, dict) or not isinstance(item.get("path"), str) for item in files):
            raise ValueError("Invalid scientific artifact inventory")
        files = {item["path"]: item for item in files}
    if not isinstance(files, dict):
        raise ValueError("Science manifest has no artifact inventory")
    result = {name: item.get("sha256") if isinstance(item, dict) else item for name, item in files.items()}
    if any(not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts
           or not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for name, value in result.items()):
        raise ValueError("Science manifest contains unsafe paths or malformed hashes")
    return result


def _source(root, report, budget, delegated):
    documents, contents = {}, {}
    for name in ("rows.json", "rows.jsonl", "summary.json", "protocol.json", "science_manifest.json", "COMPLETED",
                 "workflow-seal.json", "stage.json", "execution.json", "mechanism_summary.json"):
        path = root / name
        if not path.exists() and not path.is_symlink():
            continue
        try:
            raw = budget.read(path)
            contents[name] = raw
            if name not in ("rows.jsonl", "COMPLETED"):
                decoded = _decode(raw)
                if not isinstance(decoded, dict):
                    raise ValueError("Canonical roadmap JSON document must be an object")
                documents[name] = decoded
        except (OSError, ValueError, TypeError, RecursionError) as error:
            budget.omit(_relative(path, report), error)
    rows_document = documents.get("rows.json")
    stage = documents.get("summary.json", {}).get("stage", root.name)
    rows = []
    name = "rows.json"
    if isinstance(rows_document, dict) and rows_document.get("schema") == SCHEMA and isinstance(rows_document.get("rows"), list):
        rows = [(row, f"/rows/{index}", "") for index, row in enumerate(rows_document["rows"])]
    elif "rows.json" in contents:
        budget.omit(root / "rows.json", "Invalid roadmap canonical rows schema")
    elif "rows.jsonl" in contents:
        name = "rows.jsonl"
        for index, line in enumerate(contents[name].splitlines()):
            try:
                rows.append((_decode(line), "", index + 1))
            except (ValueError, TypeError, RecursionError) as error:
                budget.omit(f"{root / name}:line {index + 1}", error)
    else:
        budget.omit(root, "Roadmap source has no readable canonical experiments")
    status, reason = "UNSEALED", "Partial or interrupted source; scores are not credited"
    files = {}
    try:
        manifest = documents.get("science_manifest.json", {})
        files = _manifest_files(manifest)
        if "COMPLETED" in contents:
            if (manifest.get("schema") != "tdn.roadmap-science-manifest/v1" or stage not in STAGES
                    or manifest.get("stage") != stage or documents.get("protocol.json", {}).get("schema") != "tdn.roadmap/v1"):
                raise ValueError("Science manifest family, stage or protocol schema differs")
            if canonical_digest(manifest) != contents["COMPLETED"].decode().strip():
                raise ValueError("COMPLETED marker differs from science manifest")
            if manifest.get("protocol_sha256") != canonical_digest(documents.get("protocol.json")):
                raise ValueError("Science manifest protocol differs from canonical protocol")
            for required in ("rows.json", "summary.json", "protocol.json"):
                if required not in contents or files.get(required) != hashlib.sha256(contents[required]).hexdigest():
                    raise ValueError(f"{required} differs from or is absent from the science seal")
            for filename in ("rows.jsonl", "mechanism_summary.json"):
                if filename in contents and files.get(filename) != hashlib.sha256(contents[filename]).hexdigest():
                    raise ValueError(f"{filename} differs from or is absent from the science seal")
            if "rows.jsonl" in contents and [_decode(line) for line in contents["rows.jsonl"].splitlines() if line] != [r for r, _, _ in rows]:
                raise ValueError("Canonical compact and streamed experiment ledgers differ")
            if documents.get("summary.json", {}).get("status") != "COMPLETED":
                raise ValueError("Completed marker conflicts with science status")
            if "experiment_count" in documents["summary.json"] and documents["summary.json"]["experiment_count"] != len(rows):
                raise ValueError("Declared experiment count differs from canonical rows")
            if "stage.json" in documents and documents["stage.json"].get("status") != "COMPLETED":
                raise ValueError("Wrapper execution did not complete")
            if "execution.json" in documents:
                seal = documents.get("workflow-seal.json", {})
                if seal.get("schema_version") != 1 or seal.get("protocol_sha256") != manifest.get("protocol_sha256"):
                    raise ValueError("Wrapper protocol seal differs from scientific lineage")
                if documents["execution.json"].get("software", {}).get("source_tree_sha256") != manifest.get("source_tree_sha256"):
                    raise ValueError("Wrapper source differs from scientific lineage")
                for filename in ("execution.json", "protocol.json", "stage.json", "science_manifest.json"):
                    if filename not in contents or seal.get("files", {}).get(filename) != hashlib.sha256(contents[filename]).hexdigest():
                        raise ValueError("Wrapper provenance is unsealed or changed")
            status, reason = "VERIFIED_CANONICAL", "Projected canonical files match science and available wrapper seals; binaries are not verified here"
        else:
            budget.omit(root, reason)
    except (KeyError, ValueError, TypeError, UnicodeError) as error:
        status, reason = "INVALID", str(error)
        budget.omit(root, reason)
    for filename, raw in contents.items():
        path = root / filename
        delegated[path] = {"sha256": hashlib.sha256(raw).hexdigest(), "signature": budget.snapshots[path],
                           "schema": documents.get(filename, {}).get("schema") if isinstance(documents.get(filename), dict) else None}
    source = _relative(root / name, report)
    identities = [row.get("experiment_id") for row, _, _ in rows if isinstance(row, dict)]
    if (len(identities) != len(rows) or any(not isinstance(item, str) or not item for item in identities)
            or len(set(item for item in identities if isinstance(item, str))) != len(identities)):
        status, reason = "INVALID", "Canonical experiment identities are malformed or duplicated"
        budget.omit(source, reason)
    return rows, {"source_dir": _relative(root, report), "source_path": source, "source_sha256": budget.hashes.get(root / name),
        "stage": stage, "seal_status": status, "reason": reason, "canonical_rows": len(rows),
        "protocol_sha256": canonical_digest(documents.get("protocol.json")),
        "scientific_outcome": documents.get("summary.json", {}).get("scientific_outcome")}, files, documents


def _projection(rows, source, budget, now):
    from tdn.analysis.roadmap.core import validate_row
    experiments, checks, values, metric_rows = [], [], [], []
    for index, (row, pointer, line) in enumerate(rows):
        valid, reason = source["seal_status"] == "VERIFIED_CANONICAL", ""
        try:
            validate_row(row)
            if row.get("protocol_sha256") != source["protocol_sha256"] or row.get("stage") != source["stage"]:
                raise ValueError("Experiment stage or protocol differs from its source")
        except (ValueError, KeyError, TypeError, RecursionError) as error:
            valid, reason = False, str(error)
            budget.omit(source["source_path"] + "#" + pointer, "Invalid reproducible score: " + reason)
        if not isinstance(row, dict):
            continue
        common = dict(source_path=source["source_path"], source_record=pointer, source_line=line,
            source_sha256=source["source_sha256"], row_sha256=row.get("row_sha256"),
            evidence_status=source["seal_status"] if not reason else "INVALID_ROW",
            validation_error=reason or (source["reason"] if not valid else ""))
        assessment = row.get("assessment") if isinstance(row.get("assessment"), dict) else {}
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        config = row.get("effective_config") if isinstance(row.get("effective_config"), dict) else {}
        experiment = {**common, "experiment_id": row.get("experiment_id"), "stage": row.get("stage", source["stage"]),
            "mechanisms": row.get("mechanism_ids", []), "combinations": row.get("combination_ids", []),
            "status": row.get("status"), "device": row.get("device"), "profile": row.get("profile"),
            "raw_verdict": assessment.get("verdict"), "raw_score_1_100": assessment.get("score_1_100"),
            "verdict": assessment.get("verdict") if valid else "NA", "score_1_100": assessment.get("score_1_100") if valid else 1,
            "evidence_coverage": assessment.get("evidence_coverage") if valid else 0,
            **{key: metrics.get(key) for key in ("parameters", "median_seconds", "wall_seconds", "training_seconds", "peak_allocated_bytes", "peak_reserved_bytes")},
            **{key: config.get(key) for key in ("family", "seed", "grid", "batch_size", "learning_rate")},
            "config_record": pointer + "/effective_config", "metrics_record": pointer + "/metrics",
            "checks_record": pointer + "/checks", "evidence_record": pointer + "/evidence"}
        for category in ("gap", "math"):
            a = row.get(category + "_assessment") if isinstance(row.get(category + "_assessment"), dict) else {}
            experiment[category + "_verdict"] = a.get("verdict") if valid else "NA"
            experiment[category + "_score_1_100"] = a.get("score_1_100") if valid else 1
        experiments.append(experiment)
        for number, check in enumerate(row.get("checks", []) if isinstance(row.get("checks"), list) else []):
            if not isinstance(check, dict):
                continue
            item = {**common, **{key: check.get(key) for key in COLUMNS["checks"] if key not in PREFIX},
                "source_record": pointer + f"/checks/{number}", "experiment_record": pointer,
                "experiment_id": row.get("experiment_id"), "stage": row.get("stage", source["stage"]),
                "raw_verdict": check.get("verdict"), "verdict": check.get("verdict") if valid else "NA",
                "score_1_100": check.get("score_1_100") if valid else 1,
                "reason_record": pointer + f"/checks/{number}/reason", "proof_status": "NOT_A_PROOF"}
            if len(_cell(item.get("reason")).encode()) > MAX_CELL_BYTES:
                item["reason"] = "See exact reason_record in canonical source"
            checks.append(item)
        for group in ("metrics", "effective_config", "evidence"):
            for field, value, storage in _leaves(row.get(group), "/" + group):
                if len(_cell(value).encode()) > MAX_CELL_BYTES:
                    storage = "source_pointer"
                values.append({**common, "source_record": pointer + field, "experiment_id": row.get("experiment_id"),
                    "stage": row.get("stage", source["stage"]), "field": field, "value": value if storage == "inline" else "",
                    "value_type": type(value).__name__, "value_record": pointer + field, "storage": storage})
        named = {key: experiment[key] for key in ("score_1_100", "gap_score_1_100", "math_score_1_100", "evidence_coverage",
                 "parameters", "median_seconds", "wall_seconds", "training_seconds", "peak_allocated_bytes", "peak_reserved_bytes")
                 if isinstance(experiment.get(key), (int, float)) and not isinstance(experiment.get(key), bool)
                 and math.isfinite(experiment[key])}
        metric_rows.append({"t": now, "phase": "roadmap/" + str(experiment["stage"]) + "/canonical", "step": index,
            "metrics": named, "source_path": common["source_path"], "source_record": pointer, "source_line": line,
            "source_sha256": common["source_sha256"], "row_sha256": common["row_sha256"],
            "evidence_status": common["evidence_status"], "timestamp_scope": "report publication; measured cost remains in metrics"})
    return experiments, checks, values, metric_rows


def publish_roadmap_outputs(report_dir, source_dirs, *, delegated_sources=None):
    report = _directory(report_dir)
    roots, seen = [], set()
    for supplied in source_dirs:
        root = _directory(supplied)
        if root == report or report in root.parents:
            raise ValueError("Roadmap science cannot be a report directory or its child")
        for candidate in (root, *(root / stage for stage in STAGES)):
            if candidate in seen or not candidate.is_dir():
                continue
            seen.add(candidate)
            if any((candidate / name).exists() for name in ("rows.json", "rows.jsonl")) and is_roadmap_root(candidate):
                roots.append(_directory(candidate))
    if not roots:
        return None
    budget, delegated = Budget(), delegated_sources if delegated_sources is not None else {}
    tables = {name: [] for name in COLUMNS}
    metric_rows, sources, pages = [], [], []
    for root in roots:
        rows, source, declared, documents = _source(root, report, budget, delegated)
        sources.append(source)
        projected = _projection(rows, source, budget, time.time())
        for name, items in zip(("experiments", "checks", "values"), projected[:3]):
            tables[name].extend(items)
        metric_rows.extend(projected[3])
        for index, record in enumerate(documents.get("mechanism_summary.json", {}).get("records", [])):
            if not isinstance(record, dict):
                budget.omit(root / "mechanism_summary.json", "Invalid mechanism summary record")
                continue
            valid = source["seal_status"] == "VERIFIED_CANONICAL"
            pointer = f"/records/{index}"
            tables["mechanisms"].append({"source_path": _relative(root / "mechanism_summary.json", report),
                "source_record": pointer, "source_sha256": budget.hashes.get(root / "mechanism_summary.json"),
                "evidence_status": source["seal_status"], "validation_error": source["reason"] if not valid else "",
                **{key: record.get(key) for key in ("id", "kind", "name", "experiment_count", "expected_stages", "observed_stages", "missing_stages", "mathematical_significance")},
                "raw_verdict": record.get("verdict"), "raw_score_1_100": record.get("score_1_100"),
                "verdict": record.get("verdict") if valid else "NA", "score_1_100": record.get("score_1_100") if valid else 1,
                "evidence_coverage": record.get("evidence_coverage") if valid else 0,
                "gap_verdict": record.get("gap_assessment", {}).get("verdict") if valid else "NA",
                "math_verdict": record.get("math_assessment", {}).get("verdict") if valid else "NA",
                "failed_checks": record.get("failed_checks"), "na_checks": record.get("na_checks"),
                "failed_checks_record": pointer + ("/failed_check_ids" if "failed_check_ids" in record else "/failed_checks"),
                "na_checks_record": pointer + ("/na_check_ids" if "na_check_ids" in record else "/na_checks"),
                "gaps_record": pointer + "/gaps"})
        for path in sorted(root.rglob("*")):
            if path.is_dir() and not path.is_symlink():
                continue
            relative = _relative(path, report)
            if len(tables["artifacts"]) >= MAX_FILES:
                budget.omit(relative, "Roadmap inventory ceiling exceeded; collection archive remains complete")
                break
            if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
                budget.omit(relative, "Scientific symlinks and nonregular files are refused")
                continue
            name = str(path.relative_to(root))
            digest = budget.hashes.get(path, declared.get(name))
            tables["artifacts"].append(dict(source_path=relative, bytes=path.stat().st_size, kind=path.suffix,
                sha256=digest, hash_status="computed_projection" if path in budget.hashes else "declared_unverified" if digest else "not_read",
                manifest_path=_relative(root / "science_manifest.json", report) if name in declared else ""))
    output = report / "outputs/roadmap"
    output.mkdir(parents=True, exist_ok=True)
    _directory(output)
    counts = {}
    try:
        for name, items in tables.items():
            counts[name] = _pages(output, name, items, COLUMNS[name], budget, pages)
        counts["metrics"] = _pages(output, "metrics", metric_rows, [], budget, pages, jsonl=True)
    except (OSError, ValueError) as error:
        budget.omit(output, error)
    counts = {name: sum(page["rows"] for page in pages if page["table"] == name) for name in (*tables, "metrics")}
    catalogs, contracts = [], []
    for index in range(0, len(pages), 128):
        batch = pages[index:index + 128]
        catalog = output / f"pages-{index // 128 + 1:04d}.json"
        atomic_json(catalog, {"schema": "tdn.roadmap-pages/v1", "pages": batch})
        catalogs.append({"path": "outputs/roadmap/" + catalog.name, "pages": len(batch), "sha256": hashlib.sha256(catalog.read_bytes()).hexdigest()})
        contract = output / f"contract-{index // 128 + 1:04d}.json"
        atomic_json(contract, {"version": 1, "outputs": [{"path": p["path"], "required": True, "format": p["format"],
            "min_bytes": p["bytes"], "max_bytes": p["bytes"], **({"rows": p["rows"], "columns": COLUMNS[p["table"]]} if p["format"] == "csv" else {})}
            for p in batch]})
        contracts.append({"path": "outputs/roadmap/" + contract.name, "outputs": len(batch)})
    expected = {key: len(value) for key, value in tables.items()}
    expected["metrics"] = len(metric_rows)
    complete = budget.omission_count == 0 and counts == expected
    index = {"schema": "tdn.roadmap.tower-tables/v1", "columns": COLUMNS, "expected_rows": expected,
        "published_rows": counts, "page_count": len(pages), "page_catalogs": catalogs, "output_contracts": contracts,
        "page_limits": {"rows": PAGE_ROWS, "bytes": PAGE_BYTES, "metric_line_bytes": 65536},
        "publication_reserves": {"canonical_file_bytes": MAX_SOURCE_BYTES, "aggregate_source_bytes": MAX_READ_BYTES,
            "aggregate_page_bytes": MAX_OUTPUT_BYTES, "pages": MAX_PAGES,
            "scope": "Explicit full-profile reserves; complete record paging retains the unchanged native reader limits"},
        "sources": sources, "source_record": "RFC6901 JSON pointer in source_path; source_line is one-based only for incomplete JSONL sources",
        "metrics": "Paged JSONL records use Tower's canonical t/phase/step/metrics format; additional source fields preserve exact lineage",
        "metric_root_stream": "Live bounded progress remains in report metrics.jsonl; complete per-experiment points are paged and explicitly indexed here",
        "checkpoint_hashes": "Binary file hashes remain declared_unverified unless separately computed; binaries are never deserialized",
        "scientific_scope": "Scores report required-evidence attainment; mathematical tests are finite numerical evidence, never a proof",
        "reporting_complete": complete, "omission_count": budget.omission_count, "omissions": budget.omissions,
        "omissions_list_truncated": budget.omission_count > len(budget.omissions)}
    atomic_json(report / "outputs/roadmap-tables.json", index)
    results = {"schema": "tdn.roadmap.tower-results/v1", "reporting_complete": complete,
        "reporting_omission_count": budget.omission_count, "table_rows": counts, "page_count": len(pages),
        "sources": sources, "application_completion_is_scientific_success": False, "science_sources_mutated": False,
        "observed_read_bytes": budget.read_bytes, "observed_output_bytes": budget.output_bytes,
        "verdicts": dict(Counter(row["verdict"] for row in tables["experiments"])),
        "index": "outputs/roadmap-tables.json", "claim_scope": "Numerical evidence and scores retain gap/math/cost scope; no superiority inferred"}
    atomic_json(report / "outputs/roadmap-results.json", results)
    registered = {entry.get("path") for entry in read_json(report / "logs.json").get("logs", [])}
    for name in ("roadmap-tables.json", "roadmap-results.json"):
        location = "outputs/" + name
        if location not in registered:
            register_log(report, "analytics." + name.replace(".json", ""), location, group="Roadmap review",
                         label="Roadmap complete page index" if name == "roadmap-tables.json" else "Roadmap scores and evidence integrity")
    # Register first pages for convenient entry, keeping Tower's 256-log cap.
    # Every remaining page is reachable through exact catalog/contract paths.
    for kind in counts:
        page = next((p for p in pages if p["table"] == kind), None)
        if page and page["path"] not in registered:
            register_log(report, "analytics.roadmap." + kind, page["path"], group="Roadmap review",
                         label="Roadmap " + kind + " (first page; complete catalog in index)")
    return results
