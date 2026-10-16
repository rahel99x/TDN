"""Verify and safely extract the exact P1R review upload; execute no uploaded code."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import stat
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARCHIVE_SHA256 = "c1b847269855135459e865d655aba316a3e0e21d77afcdf0763d29af2c47d262"
RUN_ID = "launch-630d054468ab4bb7915d33cc9482ebf2-a1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Fresh extraction directory inside this repository")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    require(output != ROOT and output.is_relative_to(ROOT), "Keep extraction inside this checkout")
    require(not output.exists(), "Preserve existing extraction; choose a fresh output")
    archive = HERE / "native-review.zip"
    index = json.loads((HERE / "upload-index.json").read_text())
    require(index["run_id"] == RUN_ID and index["archive_sha256"] == ARCHIVE_SHA256,
            "Unexpected upload identity")
    require(archive.stat().st_size == index["archive_bytes"] == 53991337, "Archive length mismatch")
    h = hashlib.sha256()
    with archive.open("rb") as source:
        for part in index["parts"]:
            chunk = source.read(part["bytes"])
            require(len(chunk) == part["bytes"] and hashlib.sha256(chunk).hexdigest() == part["sha256"],
                    "Original upload part mismatch")
            h.update(chunk)
        require(not source.read(1), "Unexpected archive suffix")
    require(h.hexdigest() == ARCHIVE_SHA256, "Archive hash mismatch")
    rows = {row["path"]: row for row in index["files"]}
    require(len(rows) == len(index["files"]) == index["file_count"] == 6228, "Inventory count mismatch")
    output.mkdir(parents=True, exist_ok=False)
    seen = set()
    total = 0
    with zipfile.ZipFile(archive) as bundle:
        require(len(bundle.infolist()) == len(rows), "ZIP count mismatch")
        for info in bundle.infolist():
            path = PurePosixPath(info.filename)
            require(not path.is_absolute() and ".." not in path.parts and "\\" not in info.filename
                    and len(path.parts) > 1 and path.parts[0] == RUN_ID, "Unsafe ZIP path")
            require(not info.is_dir() and not info.flag_bits & 1
                    and stat.S_IFMT(info.external_attr >> 16) in (0, stat.S_IFREG), "Non-regular ZIP member")
            relative = PurePosixPath(*path.parts[1:]).as_posix()
            require(relative not in seen and relative in rows, "Duplicate or unexpected ZIP member")
            seen.add(relative)
            row = rows[relative]
            require(info.file_size == row["bytes"] and info.file_size <= 128 << 20, "Member size mismatch")
            total += info.file_size
            require(total <= 256 << 20, "Extraction budget exceeded")
            target = output / path
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            count = 0
            with bundle.open(info) as source, target.open("xb") as destination:
                while chunk := source.read(1 << 20):
                    digest.update(chunk)
                    count += len(chunk)
                    destination.write(chunk)
            require(count == row["bytes"] and digest.hexdigest() == row["sha256"], "Member hash mismatch")
    require(total == index["uncompressed_bytes"] and seen == set(rows), "Incomplete extraction")
    result = dict(status="PASS", parts=len(index["parts"]), files=len(rows),
                  archive_sha256=ARCHIVE_SHA256, uncompressed_bytes=total,
                  extraction_root=str(output / RUN_ID), uploaded_instructions_executed=False)
    (output / "transport-check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
