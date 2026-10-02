"""Small, deterministic provenance and durable publication helpers."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    """Publish in the same directory; never stage checkpoint files in /tmp."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".pending")
    with staging.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    staging.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def source_provenance() -> dict:
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                            capture_output=True, text=True, check=False)
    source_paths = [*(root / "tdn").rglob("*.py"), *(root / "reference").rglob("*.py")]
    source_paths.extend(p for p in (root / "requirements.txt", root / "pyproject.toml") if p.exists())
    files = {str(p.relative_to(root)): file_hash(p) for p in sorted(source_paths)}
    return {"commit": commit.stdout.strip() or "unversioned",
            "python_source_hash": canonical_hash(files), "files": files}
