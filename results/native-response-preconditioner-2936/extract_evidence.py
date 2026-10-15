"""Reconstruct only the reviewed regular files; never extract test symlinks."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(here.parents[1]) or output == here.parents[1]:
        raise ValueError("Extraction must remain inside this checkout")
    manifest = json.loads((here / "extraction-manifest.json").read_text())
    archive = here / "launch-ba0ccf6660874b2f826b1d7fd9e25886-a1.zip"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest["archive_sha256"]:
        raise ValueError("Archive checksum mismatch")
    payloads = {}
    with zipfile.ZipFile(archive) as bundle:
        if len(bundle.namelist()) != len(set(bundle.namelist())):
            raise ValueError("Duplicate archive entries")
        for row in manifest["retained"]:
            path = PurePosixPath(row["path"])
            if path.is_absolute() or ".." in path.parts or "\\" in str(path):
                raise ValueError("Unsafe retained path")
            info = bundle.getinfo(manifest["prefix"] + row["path"])
            if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("Retained file is not a regular payload")
            raw = bundle.read(info)
            if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
                raise ValueError("Retained payload checksum mismatch")
            payloads[path] = raw
    output.mkdir(parents=True, exist_ok=False)
    for path, raw in payloads.items():
        target = output / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    print(f"Verified and reconstructed {len(payloads)} regular evidence files: {output}")


if __name__ == "__main__":
    main()
