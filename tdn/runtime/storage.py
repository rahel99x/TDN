"""All project outputs, temporary files and caches stay inside the project root."""
from __future__ import annotations
import os
from pathlib import Path

CARC_ROOT = Path("/home1/aadaniel/projects/TDN")

def project_root() -> Path:
    checkout = Path(__file__).resolve().parents[2]
    root = Path(os.environ.get("TDN_PROJECT_ROOT", checkout)).resolve()
    if root != checkout: raise ValueError("TDN_PROJECT_ROOT must be this repository, including for local CPU validation")
    if os.environ.get("SLURM_JOB_ID") and root != CARC_ROOT:
        raise ValueError(f"CARC runtime root must be {CARC_ROOT}")
    return root

def contained_path(path: str | Path, root: Path | None = None) -> Path:
    root = (root or project_root()).resolve()
    result = Path(path).expanduser().resolve()
    if not result.is_relative_to(root): raise ValueError(f"Path escapes project root: {result}")
    return result

def configure_storage() -> Path:
    root = project_root()
    paths = {"TMPDIR": ".runtime/tmp", "TMP": ".runtime/tmp", "TEMP": ".runtime/tmp",
             "PIP_CACHE_DIR": ".runtime/cache/pip", "XDG_CACHE_HOME": ".runtime/cache",
             "TORCH_HOME": ".runtime/cache/torch", "TORCHINDUCTOR_CACHE_DIR": ".runtime/cache/inductor",
             "TRITON_CACHE_DIR": ".runtime/cache/triton", "TORCH_EXTENSIONS_DIR": ".runtime/cache/extensions",
             "CUDA_CACHE_PATH": ".runtime/cache/cuda", "MPLCONFIGDIR": ".runtime/cache/matplotlib"}
    for name, relative in paths.items():
        target = contained_path(root / relative, root)
        target.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(target)
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return root
