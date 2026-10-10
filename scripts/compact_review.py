#!/usr/bin/env python3
"""Exact-byte, deduplicated, streaming review packages. Python stdlib only.

Review is an explicit subset, never a replacement for the original experiment.
No model objects are loaded, source files rewritten, or science claims verified.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import lzma
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tarfile
import time
import uuid
import zlib

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "tdn.compact-review/v1"
MANIFEST = "REVIEW-MANIFEST.json"
BLOCK = 1024 * 1024
SHA = re.compile(r"[0-9a-f]{64}\Z")
REPLAY = {".pt", ".pth", ".npz", ".npy", ".safetensors"}
RENDERED = {".png", ".pdf", ".svg", ".jpg", ".jpeg", ".webp"}


class Progress:
    """Bounded-frequency byte progress, including activity inside large files."""
    def __init__(self, phase, total, *, output_bytes=None, interval=5.0):
        self.phase, self.total, self.output_bytes = phase, total, output_bytes
        self.interval, self.done, self.name = interval, 0, ""
        self.started = time.monotonic()
        self.last = self.started
        self.emit(force=True)

    def add(self, count, name=""):
        self.done += count
        self.name = name
        self.emit()

    def emit(self, *, force=False):
        now = time.monotonic()
        if not force and now - self.last < self.interval:
            return
        self.last = now
        elapsed = now - self.started
        percent = 100 * self.done / self.total if self.total else 100
        rate = self.done / BLOCK / elapsed if elapsed else 0
        output = f"; compressed output {self.output_bytes() / BLOCK:.1f} MiB" if self.output_bytes else ""
        name = f"; file={self.name}" if self.name else ""
        print(f"{self.phase}: {percent:.1f}% ({self.done / BLOCK:.1f}/{self.total / BLOCK:.1f} MiB); "
              f"{elapsed:.1f}s; {rate:.1f} MiB/s{output}{name}", flush=True)

    def finish(self):
        self.emit(force=True)


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, indent=2, allow_nan=False) + "\n").encode()


def inside(value):
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    # Reject symlink components, including symlinks which point back inside.
    path = Path(os.path.abspath(path))
    path.relative_to(ROOT.resolve())
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError(f"Symlinks are not supported: {parent}")
        if parent == ROOT.resolve():
            break
    return path


def safe_name(value):
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("Unsafe archive path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(p in (".", "..") for p in value.split("/")) or str(path) != value:
        raise ValueError(f"Unsafe archive path: {value}")
    return value


def read_json(path):
    return json.loads(inside(path).read_bytes())


def fingerprint(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Only regular files are supported: {path}")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_mode)


def digest(path, *, progress=None):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(BLOCK):
            result.update(block)
            if progress:
                progress.add(len(block), str(path.relative_to(ROOT)))
    return result.hexdigest()


def resolve_run(value):
    if value == "latest":
        value = read_json(ROOT / "runs/.fedora-portfolio-latest.json")["run_id"]
    candidate = Path(value)
    if not candidate.is_absolute() and len(candidate.parts) == 1:
        candidate = ROOT / "runs" / candidate
    candidate = inside(candidate)
    if candidate.parent != ROOT.resolve() / "runs" or not candidate.is_dir():
        raise ValueError("Select an existing original runs/<run-id> directory, not a collected archive")
    return candidate


def run_roots(value):
    """Read frozen recovery identities without importing today's science code."""
    initial = resolve_run(value)
    pending, found = [initial], set()
    while pending:
        base = pending.pop()
        if base in found:
            continue
        found.add(base)
        workflows = [base / name for name in ("portfolio-workflow.json", "advance-workflow.json")
                     if (base / name).exists()]
        if not workflows:
            continue
        if len(workflows) != 1:
            raise ValueError("Ambiguous workflow manifests in one run directory")
        workflow = workflows[0]
        descriptor = read_json(workflow).get("recovery")
        if not descriptor:
            continue
        bridge_path = inside(descriptor["manifest_path"])
        if bridge_path != base / "recovery.json" or digest(bridge_path) != descriptor["sha256"]:
            raise ValueError("Recovery bridge identity/hash differs; preserve the evidence and investigate")
        bridge = read_json(bridge_path)
        for key in ("stage_paths", "resume_paths"):
            if bridge.get(key) != descriptor.get(key):
                raise ValueError("Recovery bridge paths differ from the frozen descriptor")
        origins = list(bridge.get("origin_roots", []))
        origins.append(str(Path(bridge["origin_workflow_path"]).parent))
        for key in ("stage_paths", "resume_paths"):
            origins.extend(str(Path(p).parent) for p in bridge.get(key, {}).values())
        for origin in origins:
            pending.append(resolve_run(origin))
    return initial, sorted(found)


def excluded_cache(name):
    return name in {"__pycache__", ".pytest_cache"} or name.endswith(("pytest-work", "pytest-cache"))


def enumerate_files(roots):
    files, skipped = [], []
    def fail_walk(error):
        raise error
    for root in roots:
        for base, directories, names in os.walk(root, followlinks=False, onerror=fail_walk):
            directories.sort()
            for name in list(directories):
                path = Path(base) / name
                if excluded_cache(name):
                    directories.remove(name)
                    skipped.append(str(path.relative_to(ROOT)))
                elif path.is_symlink():
                    raise ValueError(f"Symlink directory: {path}")
            for name in sorted(names):
                path = Path(base) / name
                fingerprint(path)
                files.append(path)
    return sorted(files), skipped


def role(path):
    if path.suffix.lower() in RENDERED:
        return "rendered_figure"
    if path.suffix.lower() in REPLAY:
        return "replay_array_or_checkpoint"
    return "evidence_or_other"


def inventory(value, mode="review", *, hash_files=True, show_progress=False):
    initial, roots = run_roots(value)
    paths, skipped = enumerate_files(roots)
    # Related tables/catalogs share the XZ dictionary even across distant stages.
    # Choose aliases in this same order so hardlink targets always come first.
    paths.sort(key=lambda path: (path.name, str(path.relative_to(ROOT))))
    progress = Progress("Hashing source", sum(p.stat().st_size for p in paths)) if show_progress and hash_files else None
    rows, stamps, duplicates = [], {}, {}
    for path in paths:
        name = str(path.relative_to(ROOT))
        stamp = fingerprint(path)
        sha = digest(path, progress=progress) if hash_files else None
        if stamp != fingerprint(path):
            raise ValueError(f"Source changed while hashing: {path}")
        stamps[name] = stamp
        category = role(path)
        include = mode == "full" or category == "evidence_or_other"
        row = {"path": name, "bytes": stamp[2], "sha256": sha,
               "mode": stat.S_IMODE(stamp[5]), "mtime_ns": stamp[3],
               "included": include, "role": category,
               "reason": "exact bytes retained" if include else "optional replay/rendered payload; retained in original run"}
        # Hard links only between identical bytes AND modes; ordinary tar readers
        # can restore them too. Our restore makes independent files.
        key = (sha, row["bytes"], row["mode"])
        if include and hash_files:
            if key in duplicates:
                row["duplicate_of"] = duplicates[key]
            else:
                duplicates[key] = name
        rows.append(row)
    if progress:
        progress.finish()
    if not rows:
        raise ValueError("Run contains no regular files")
    manifest = {"schema": SCHEMA, "mode": mode, "primary_run": initial.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(ROOT.resolve()), "run_roots": [p.name for p in roots],
        "exporter_sha256": digest(Path(__file__)), "files": rows,
        "excluded_cache_directories": skipped,
        "scope": "Transport integrity only; original science/software/source seals remain unchanged. "
                 "Review omits listed replay arrays/checkpoints and rendered figures; it is not a complete replay bundle. "
                 "Full retains all regular files outside declared disposable cache directories. "
                 "No values, precision, line endings, failures or learning-curve samples are altered. "
                 "Scheduler accounting is the saved snapshot, not a new sacct query."}
    return manifest, stamps, roots


def summarize(manifest):
    rows = manifest["files"]
    totals = Counter()
    for row in rows:
        totals["source_bytes"] += row["bytes"]
        totals["included_bytes" if row["included"] else "omitted_bytes"] += row["bytes"]
        totals["included_files" if row["included"] else "omitted_files"] += 1
        if row.get("duplicate_of"):
            totals["deduplicated_bytes"] += row["bytes"]
        totals["bytes_" + row["role"]] += row["bytes"]
    totals["unique_payload_bytes"] = totals["included_bytes"] - totals["deduplicated_bytes"]
    return dict(totals)


class PartWriter:
    """Split compressed bytes directly; never keep an additional full archive."""
    def __init__(self, directory, archive, limit):
        self.directory, self.archive, self.limit = directory, archive, limit
        self.stream = None
        self.parts, self.total = [], 0
        self.whole_hash = hashlib.sha256()

    def _close_part(self):
        if self.stream is not None:
            self.stream.close()
            self.parts.append({"path": self.path.name, "bytes": self.count, "sha256": self.part_hash.hexdigest()})
            self.stream = None

    def write(self, value):
        size = len(value)
        self.whole_hash.update(value)
        self.total += size
        view = memoryview(value)
        while view:
            if self.stream is None:
                self.path = self.directory / f"{self.archive}.part{len(self.parts)+1:03d}"
                self.stream = self.path.open("xb")
                self.part_hash, self.count = hashlib.sha256(), 0
            block = view[:self.limit-self.count]
            self.stream.write(block)
            self.part_hash.update(block)
            self.count += len(block)
            view = view[len(block):]
            if self.count == self.limit:
                self._close_part()
        return size

    def flush(self):
        if self.stream is not None:
            self.stream.flush()

    def finish(self):
        self._close_part()
        if len(self.parts) == 1:
            (self.directory / self.parts[0]["path"]).rename(self.directory / self.archive)
            self.parts[0]["path"] = self.archive


class HashReader:
    def __init__(self, stream, *, progress=None, name=""):
        self.stream, self.sha = stream, hashlib.sha256()
        self.progress, self.name = progress, name

    def read(self, size=-1):
        data = self.stream.read(size)
        self.sha.update(data)
        if self.progress:
            self.progress.add(len(data), self.name)
        return data


class CompressionWriter:
    """Flush only on success: Ctrl-C must not finish compressing a doomed part."""
    def __init__(self, sink, codec, level):
        self.sink = sink
        self.compressor = (lzma.LZMACompressor(preset=level) if codec == "xz" else
                           zlib.compressobj(level=level, wbits=31))

    def write(self, data):
        self.sink.write(self.compressor.compress(data))
        return len(data)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.sink.write(self.compressor.flush())
        # Dropping the compressor after an exception does not flush pending data.
        self.compressor = None


class ChainReader:
    def __init__(self, directory, parts):
        self.paths = iter(directory / row["path"] for row in parts)
        self.stream = None

    def read(self, size):
        result = bytearray()
        while len(result) < size:
            if self.stream is None:
                path = next(self.paths, None)
                if path is None:
                    break
                self.stream = path.open("rb")
            block = self.stream.read(size-len(result))
            result.extend(block)
            if not block:
                self.stream.close()
                self.stream = None
        return bytes(result)

    def close(self):
        if self.stream is not None:
            self.stream.close()


def check_parts(index_path, *, show_progress=False):
    index_path = inside(index_path)
    index = read_json(index_path)
    if index.get("schema") != SCHEMA or not index.get("parts"):
        raise ValueError("Not a compact review index")
    progress = Progress("Verifying compressed parts", index["bytes"]) if show_progress else None
    whole, total, names = hashlib.sha256(), 0, set()
    for row in index["parts"]:
        name = safe_name(row["path"])
        if "/" in name or name in names:
            raise ValueError("Duplicate or nonlocal part name")
        names.add(name)
        path = inside(index_path.parent / name)
        if fingerprint(path)[2] != row["bytes"] or not SHA.fullmatch(row["sha256"]):
            raise ValueError(f"Part size/hash declaration invalid: {name}")
        sha = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(BLOCK):
                sha.update(block)
                whole.update(block)
                total += len(block)
                if progress:
                    progress.add(len(block), name)
        if sha.hexdigest() != row["sha256"]:
            raise ValueError(f"Part checksum mismatch: {name}")
    if total != index["bytes"] or whole.hexdigest() != index["sha256"]:
        raise ValueError("Complete archive checksum/size mismatch")
    if progress:
        progress.finish()
    return index


def inspect_archive(index_path, *, restore=None, show_progress=False):
    """Verify every exact included member; never use tar.extract(all)."""
    index_path = inside(index_path)
    index = check_parts(index_path, show_progress=show_progress)
    codec = index.get("compression", "xz")  # Read all pre-update XZ indices.
    if codec not in ("gzip", "xz"):
        raise ValueError("Unsupported archive compression")
    reader = ChainReader(index_path.parent, index["parts"])
    try:
        with tarfile.open(fileobj=reader, mode="r|gz" if codec == "gzip" else "r|xz") as archive:
            first = archive.next()
            if first is None or first.name != MANIFEST or not first.isfile() or first.size > 64 * BLOCK:
                raise ValueError("Missing or oversized review manifest")
            raw = archive.extractfile(first).read()
            if hashlib.sha256(raw).hexdigest() != index["manifest_sha256"]:
                raise ValueError("Manifest checksum mismatch")
            manifest = json.loads(raw)
            if manifest.get("schema") != SCHEMA:
                raise ValueError("Unsupported manifest schema")
            expected = {}
            for row in manifest["files"]:
                name = safe_name(row["path"])
                if not name.startswith("runs/") or name in expected:
                    raise ValueError("Duplicate or invalid manifest member")
                if type(row["bytes"]) is not int or row["bytes"] < 0 or not SHA.fullmatch(row["sha256"]):
                    raise ValueError("Invalid file size/hash")
                expected[name] = row
            progress = Progress("Restoring exact files" if restore else "Verifying exact files",
                sum(r["bytes"] for r in expected.values() if r["included"] and not r.get("duplicate_of"))) if show_progress else None
            seen = set()
            # archive.next() above caches the first entry; iteration would emit it twice.
            while member := archive.next():
                name = safe_name(member.name)
                row = expected.get(name)
                if name in seen or row is None or not row["included"]:
                    raise ValueError(f"Unexpected or duplicate archive member: {name}")
                destination = restore / name if restore else None
                if member.islnk():
                    target = safe_name(member.linkname)
                    source_row = expected.get(target)
                    if (target not in seen or target != row.get("duplicate_of") or source_row is None
                            or any(row[k] != source_row[k] for k in ("sha256", "bytes", "mode"))):
                        raise ValueError("Invalid deduplication link")
                    if destination:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with (restore / target).open("rb") as source, destination.open("xb") as output:
                            shutil.copyfileobj(source, output, BLOCK)
                elif member.isfile():
                    if member.size != row["bytes"] or row.get("duplicate_of"):
                        raise ValueError(f"Unexpected file size/type: {name}")
                    stream = archive.extractfile(member)
                    sha = hashlib.sha256()
                    output = None
                    try:
                        if destination:
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            output = destination.open("xb")
                        while block := stream.read(BLOCK):
                            sha.update(block)
                            if output:
                                output.write(block)
                            if progress:
                                progress.add(len(block), name)
                    finally:
                        if output:
                            output.close()
                    if sha.hexdigest() != row["sha256"]:
                        raise ValueError(f"File checksum mismatch: {name}")
                else:
                    raise ValueError("Only regular files and verified internal deduplication links are permitted")
                if destination:
                    destination.chmod(row["mode"] & 0o777)  # never restore setuid bits
                    os.utime(destination, ns=(row["mtime_ns"], row["mtime_ns"]))
                seen.add(name)
            if seen != {name for name, row in expected.items() if row["included"]}:
                raise ValueError("Archive is missing declared files")
            if progress:
                progress.finish()
            return manifest, index
    finally:
        reader.close()


def assert_stable(manifest, stamps, roots):
    # Re-read bridge hashes/closure too: an updated bridge must not silently
    # introduce an ancestor which was absent from the original inventory.
    if run_roots(str(ROOT / "runs" / manifest["primary_run"]))[1] != roots:
        raise ValueError("Recovery ancestry changed during export")
    paths, skipped = enumerate_files(roots)
    if {str(p.relative_to(ROOT)) for p in paths} != set(stamps) or skipped != manifest["excluded_cache_directories"]:
        raise ValueError("Source file set changed during export; wait for the jobs to finish")
    for path in paths:
        if fingerprint(path) != stamps[str(path.relative_to(ROOT))]:
            raise ValueError(f"Source changed during export: {path}")


def fresh_output(value, default):
    destination = inside(value or default)
    if destination.exists():
        raise ValueError(f"Output exists; select a fresh directory: {destination}")
    return destination


def pack(args):
    started = time.monotonic()
    # Preserve explicit legacy --preset commands, but never select slow XZ by
    # default for multi-gigabyte evidence. Conflicts fail before source I/O.
    codec = args.compression or ("xz" if args.preset is not None else "gzip")
    if (codec == "gzip" and args.preset is not None) or (codec == "xz" and args.gzip_level is not None):
        raise ValueError("Use --gzip-level with gzip, or --preset with xz; do not combine them")
    level = (args.preset if args.preset is not None else 1) if codec == "xz" else (args.gzip_level or 1)
    print("Inventory and SHA-256 of every source file (original files are read-only)...", flush=True)
    manifest, stamps, roots = inventory(args.run, args.mode, show_progress=True)
    destination = fresh_output(args.output, ROOT / "runs/review-exports" /
        (manifest["primary_run"] + "-" + args.mode + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")))
    if any(destination.is_relative_to(root) for root in roots):
        raise ValueError("Export must be outside every source run")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(destination.name + ".partial-" + uuid.uuid4().hex)
    staging.mkdir()
    writer = None
    try:
        manifest_data = json_bytes(manifest)
        if len(manifest_data) > 64 * BLOCK:
            raise ValueError("Inventory exceeds the 64 MiB verifier limit; export smaller independent runs")
        archive_name = manifest["primary_run"] + "-" + args.mode + (".tar.gz" if codec == "gzip" else ".tar.xz")
        writer = PartWriter(staging, archive_name, args.part_mib * BLOCK)
        unique_bytes = summarize(manifest)["unique_payload_bytes"]
        print(f"Compressing {unique_bytes:,} unique exact bytes using {codec.upper()}-{level} (one CPU thread)...", flush=True)
        if codec == "xz" and level >= 6:
            print("High XZ levels can be slow on multi-GB runs; gzip-1 is the fast default, XZ-1 gives smaller uploads.", flush=True)
        progress = Progress("Compression input", unique_bytes, output_bytes=lambda: writer.total)
        with CompressionWriter(writer, codec, level) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                entry = tarfile.TarInfo(MANIFEST)
                entry.size, entry.mode = len(manifest_data), 0o644
                archive.addfile(entry, io.BytesIO(manifest_data))
                # Inventory order keeps each hardlink target before its aliases.
                for row in manifest["files"]:
                    if not row["included"]:
                        continue
                    path = ROOT / row["path"]
                    if fingerprint(path) != stamps[row["path"]]:
                        raise ValueError(f"Source changed during export: {path}")
                    entry = tarfile.TarInfo(row["path"])
                    entry.mode, entry.mtime = row["mode"], row["mtime_ns"] // 10**9
                    if row.get("duplicate_of"):
                        entry.type, entry.linkname = tarfile.LNKTYPE, row["duplicate_of"]
                        archive.addfile(entry)
                    else:
                        entry.size = row["bytes"]
                        with path.open("rb") as source:
                            reader = HashReader(source, progress=progress, name=row["path"])
                            archive.addfile(entry, reader)
                        if reader.sha.hexdigest() != row["sha256"]:
                            raise ValueError(f"Source bytes changed during export: {path}")
        progress.finish()
        writer.finish()
        assert_stable(manifest, stamps, roots)
        index = {"schema": SCHEMA, "archive": archive_name, "bytes": writer.total,
            "sha256": writer.whole_hash.hexdigest(), "manifest_sha256": hashlib.sha256(manifest_data).hexdigest(),
            "parts": writer.parts, "mode": args.mode, "compression": codec,
            "xz_preset": level if codec == "xz" else None, "gzip_level": level if codec == "gzip" else None,
            "statistics": summarize(manifest), "included_workflow_directories": manifest["run_roots"],
            "reassemble": "Concatenate parts in the listed order; verify SHA-256. Or use compact_review.sh unpack.",
            "scope": manifest["scope"]}
        (staging / "index.json").write_bytes(json_bytes(index))
        print("Verifying parts, archive, and every retained file...", flush=True)
        inspect_archive(staging / "index.json", show_progress=True)
        assert_stable(manifest, stamps, roots)
        index["export_and_verification_seconds"] = time.monotonic() - started
        (staging / "index.json").write_bytes(json_bytes(index))
        (staging / "manifest.json").write_bytes(manifest_data)
        (staging / "README.txt").write_text(
            "Upload index.json and every archive/part named in it. manifest.json and this README are optional.\n"
            "The original run is unchanged. Do not remove it. The review package is an explicit subset.\n"
            "Verify: bash scripts/compact_review.sh verify <this-directory>/index.json\n"
            "Restore: bash scripts/compact_review.sh unpack <this-directory>/index.json --output runs/review-restored\n"
            "No concatenated full archive is needed by these commands.\n")
        # Exclusive ownership: never replace a concurrently created destination.
        if destination.exists():
            raise ValueError("Output appeared while exporting; refusing to overwrite")
        staging.rename(destination)
        totals = index["statistics"]
        print(f"Source: {totals['source_bytes']:,} bytes; retained: {totals['included_bytes']:,}; "
              f"upload: {index['bytes']:,} bytes in {len(index['parts'])} file(s).")
        print(f"Verified upload index: {destination / 'index.json'}")
        for row in index["parts"]:
            print(destination / row["path"])
        print("Upload the index and ALL listed files. Original runs are unchanged.")
        return index
    finally:
        try:
            if writer is not None:
                writer._close_part()
        finally:
            if staging.exists():
                shutil.rmtree(staging)  # Only this invocation's private output, never source evidence.


def unpack(args):
    index_path = inside(args.index)
    destination = fresh_output(args.output, ROOT / "runs" / ("review-restored-" + uuid.uuid4().hex[:12]))
    if index_path.is_relative_to(destination):
        raise ValueError("Restore cannot contain its input bundle")
    # Verify everything before publishing a restore directory.
    inspect_archive(index_path, show_progress=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(destination.name + ".partial-" + uuid.uuid4().hex)
    staging.mkdir()
    try:
        manifest, _ = inspect_archive(index_path, restore=staging, show_progress=True)
        (staging / MANIFEST).write_bytes(json_bytes(manifest))
        if destination.exists():
            raise ValueError("Restore destination appeared; refusing to overwrite")
        staging.rename(destination)
        print(f"Verified exact-byte restore: {destination}\nMode: {manifest['mode']}; original absolute paths in metadata remain unchanged.")
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("plan", "pack"):
        command = commands.add_parser(name)
        command.add_argument("run", nargs="?", default="latest", help="original runs/<id> directory or portfolio latest")
        command.add_argument("--mode", choices=("review", "full"), default="review")
        if name == "pack":
            command.add_argument("--output", type=Path, help="fresh project-contained directory")
            command.add_argument("--part-mib", type=int, choices=range(1, 29), default=24, metavar="1..28")
            command.add_argument("--compression", choices=("gzip", "xz"), help="default gzip: fast and dependency-free; xz: smaller but slower")
            command.add_argument("--gzip-level", type=int, choices=range(1, 10), default=None, metavar="1..9", help="gzip level, default 1")
            command.add_argument("--preset", type=int, choices=range(10), default=None, metavar="0..9",
                                 help="select XZ at this level; --compression xz defaults to 1; 9 is very slow on large runs")
    for name in ("verify", "unpack"):
        command = commands.add_parser(name)
        command.add_argument("index", type=Path)
        if name == "unpack":
            command.add_argument("--output", type=Path)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "plan":
            manifest, _, _ = inventory(args.run, args.mode, hash_files=False)
            print(json.dumps({"mode": args.mode, "run_roots": manifest["run_roots"],
                "statistics_before_hash_deduplication": summarize(manifest),
                "note": "Read-only estimate; compressed size is unknown until pack finishes."}, indent=2))
        elif args.command == "pack":
            pack(args)
        elif args.command == "verify":
            manifest, index = inspect_archive(args.index, show_progress=True)
            print(f"VERIFIED: {index['bytes']:,} compressed bytes; {sum(r['included'] for r in manifest['files'])} exact files; mode {manifest['mode']}. Transport integrity only.")
        elif args.command == "unpack":
            unpack(args)
        return 0
    except KeyboardInterrupt:
        print("TDN compact review: cancelled; original runs are unchanged. No incomplete bundle was published. "
              "Retry without --preset 9 to use fast gzip-1.", file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError, lzma.LZMAError, EOFError) as exc:
        print(f"TDN compact review: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
