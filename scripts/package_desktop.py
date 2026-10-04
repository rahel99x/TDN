#!/usr/bin/env python3
"""Build a source-only Windows ZIP, excluding local environments and run data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_ROOTS = {".git", ".venv", ".runtime", ".cache", ".local", ".pytest_cache",
                  "build", "dist", "runs", "logs", "data", "datasets", "reports"}


def excluded(path: Path) -> bool:
    parts = path.parts
    return (parts[0] in EXCLUDED_ROOTS or "__pycache__" in parts or
            any(part.endswith(".egg-info") for part in parts) or
            path.suffix in (".pyc", ".pyo", ".zip", ".gz", ".whl") or
            path.name in ("user.env.sh", ".env") or path.name.startswith(".env.") or
            (len(parts) > 1 and parts[0] == ".tower" and parts[1] in ("passports", "state")) or
            path.as_posix().startswith("requirements/environment-") or
            (len(parts) > 1 and parts[0] == "results" and
             (parts[1] in ("generated", "test-work") or parts[1].startswith(("dev-", "analysis-agent-")))))


def source_files(root: Path) -> list[Path]:
    root = root.resolve()
    if (root / ".git").exists() and shutil.which("git"):
        result = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                                cwd=root, capture_output=True, check=True)
        candidates = [Path(name.decode("utf-8")) for name in result.stdout.split(b"\0") if name]
    else:
        candidates = [path.relative_to(root) for path in root.rglob("*") if path.is_file()]
    selected = []
    for relative in sorted(set(candidates)):
        if excluded(relative):
            continue
        source = root / relative
        if source.is_symlink() or not source.resolve().is_relative_to(root):
            raise ValueError(f"Archive source is a symlink or escapes the project: {relative}")
        if source.is_file():
            selected.append(relative)
    return selected


def build_archive(root: Path, output: Path) -> dict:
    root = root.resolve()
    output = output.resolve()
    if not output.is_relative_to(root):
        raise ValueError("Archive output must remain inside the project")
    files = source_files(root)
    required = ("pyproject.toml", "requirements.txt", "tdn/data/dataset.py",
                "configs/desktop-smoke.yaml", "scripts/windows/Setup.ps1",
                "scripts/windows/Run.ps1", "WINDOWS_START_HERE.md", "AGENTS.md")
    present = {relative.as_posix() for relative in files}
    missing = set(required) - present
    if missing:
        raise ValueError(f"Incomplete desktop archive; missing {sorted(missing)}")
    commit = None
    if (root / ".git").exists() and shutil.which("git"):
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                                text=True, capture_output=True, check=True).stdout.strip()
    manifest = {"format": "source-only Windows desktop repository", "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_commit": commit, "hardware_target": "RTX 4090, 24 GB dedicated VRAM, 128 GB host RAM",
                "venv_included": False, "files": {relative.as_posix(): hashlib.sha256((root / relative).read_bytes()).hexdigest()
                                                   for relative in files}}
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_name(output.name + ".pending")
    with zipfile.ZipFile(pending, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in files:
            archive.write(root / relative, "TDN/" + relative.as_posix())
        archive.writestr("TDN/PACKAGING_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(pending) as archive:
        broken = archive.testzip()
        if broken:
            raise ValueError(f"ZIP integrity check failed: {broken}")
    pending.replace(output)
    return {"path": str(output), "bytes": output.stat().st_size, "source_files": len(files),
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "TDN-Windows-RTX4090.zip")
    args = parser.parse_args()
    print(json.dumps(build_archive(ROOT, args.output), indent=2))
