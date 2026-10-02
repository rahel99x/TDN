from __future__ import annotations
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

def software_metadata() -> dict:
    import numpy, scipy, torch
    root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True)
    digest = hashlib.sha256()
    for directory in ("tdn", "reference", "scripts", "configs", "tests"):
        for f in sorted((root / directory).rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts:
                digest.update(str(f.relative_to(root)).encode()); digest.update(f.read_bytes())
    return {"python": platform.python_version(), "executable": sys.executable,
            "venv": sys.prefix != sys.base_prefix, "torch": torch.__version__, "numpy": numpy.__version__,
            "scipy": scipy.__version__, "torch_cuda_runtime": torch.version.cuda,
            "git_commit": proc.stdout.strip() if proc.returncode == 0 else None,
            "tracked_or_untracked_changes": bool(dirty.stdout.strip()), "source_tree_sha256": digest.hexdigest(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "cuda_available": torch.cuda.is_available()}

def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)
