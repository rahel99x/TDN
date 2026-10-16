#!/usr/bin/env python3
"""Verify the exact P1 upload and extract numeric evidence; execute no contents.

The retained ZIP is the byte-for-byte concatenation of the three uploaded
parts. Their individual hashes are checked over those same byte ranges.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
from zipfile import ZipFile

RUN = "launch-0e144975eb4b433492fe55d75d209ef2-a1"
ARCHIVE_SHA = "caaacb862df0e272763340b69f523e66884c40e32013cd1ec6f0ff46eb89b6fb"
INDEX_SHA = "43e44377ffb99f0ac4c056132b551e17736b61f35af26352b1650ec1692480b3"
CHUNK = 1024**2


def require(ok, message):
    if not ok:
        raise ValueError(message)


def extract(archive, index, output):
    root = Path(__file__).resolve().parents[2]
    output = output.resolve()
    require(output.is_relative_to(root) and output != root and not output.exists(),
            "Extraction requires a fresh directory inside this checkout")
    require(not archive.is_symlink() and archive.is_file(), "Archive must be a regular file")
    require(not index.is_symlink() and index.is_file(), "Index must be a regular file")
    require(index.stat().st_size < 16*CHUNK, "Index exceeds receipt budget")
    raw = index.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == INDEX_SHA, "Index differs from reviewed upload")
    inventory = json.loads(raw)
    require(inventory["run_id"] == RUN and inventory["archive_sha256"] == ARCHIVE_SHA,
            "Unexpected receipt identity")
    require(archive.stat().st_size == inventory["archive_bytes"], "Archive size mismatch")
    whole = hashlib.sha256()
    with archive.open("rb") as stream:
        for part in inventory["parts"]:
            remaining, digest = part["bytes"], hashlib.sha256()
            while remaining:
                block = stream.read(min(CHUNK, remaining))
                require(bool(block), "Truncated archive part")
                remaining -= len(block)
                digest.update(block)
                whole.update(block)
            require(digest.hexdigest() == part["sha256"], "Part hash mismatch: " + part["path"])
        require(not stream.read(1), "Trailing archive bytes")
    require(whole.hexdigest() == ARCHIVE_SHA, "Archive hash mismatch")
    expected = {RUN + "/" + row["path"]: row for row in inventory["files"]}
    require(len(expected) == len(inventory["files"]) == inventory["file_count"] == 7757,
            "File roster mismatch")
    records = []
    with ZipFile(archive) as bundle:
        entries = bundle.infolist()
        require(len(entries) == len(expected), "Unexpected ZIP member count")
        require(sum(row.file_size for row in entries) == inventory["uncompressed_bytes"]
                == 320826795, "Uncompressed byte count mismatch")
        names = set()
        for row in entries:
            path = PurePosixPath(row.filename)
            require(not path.is_absolute() and ".." not in path.parts
                    and "\\" not in row.filename and "\x00" not in row.filename
                    and path.as_posix() == row.filename and path.parts[0] == RUN
                    and row.filename not in names and row.filename in expected
                    and not row.is_dir() and not row.flag_bits & 1
                    and stat.S_IFMT(row.external_attr >> 16) in (0, stat.S_IFREG),
                    "Unsupported ZIP entry: " + row.filename)
            require(row.file_size == expected[row.filename]["bytes"], "ZIP entry size mismatch")
            names.add(row.filename)
        require(names == expected.keys(), "Incomplete ZIP file roster")
        output.mkdir(parents=True, exist_ok=False)
        for row in entries:
            target = output.joinpath(*PurePosixPath(row.filename).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            digest, size = hashlib.sha256(), 0
            with bundle.open(row) as src, target.open("xb") as dst:
                while block := src.read(CHUNK):
                    size += len(block)
                    require(size <= row.file_size, "ZIP entry exceeded declared size")
                    digest.update(block)
                    dst.write(block)
            require(size == row.file_size and digest.hexdigest() == expected[row.filename]["sha256"],
                    "File hash mismatch: " + row.filename)
            records.append(dict(path=row.filename, bytes=size, sha256=digest.hexdigest()))
    receipt = dict(status="PASS", archive_sha256=ARCHIVE_SHA, index_sha256=INDEX_SHA,
                   parts_verified=len(inventory["parts"]), files_verified=len(records),
                   bytes_verified=sum(row["bytes"] for row in records), uploaded_code_executed=False)
    (output / "extraction-manifest.json").write_text(
        json.dumps(dict(**receipt, files=records), indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(extract(args.archive, args.index, args.output), sort_keys=True))
