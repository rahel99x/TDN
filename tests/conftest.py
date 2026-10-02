"""Test artifacts remain under the checkout, including pytest's temporary files."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for name,child in {"TMPDIR":"tmp","TMP":"tmp","TEMP":"tmp","MPLCONFIGDIR":"cache/matplotlib","TORCHINDUCTOR_CACHE_DIR":"cache/inductor","TRITON_CACHE_DIR":"cache/triton"}.items():
    target=(ROOT/".runtime"/child).resolve()
    if not target.is_relative_to(ROOT):raise ValueError("Test temporary/cache path escapes the checkout")
    target.mkdir(parents=True,exist_ok=True);os.environ[name]=str(target)
os.environ["MPLBACKEND"]="Agg"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

def pytest_configure(config):
    if os.environ.get("TDN_REQUIRE_GPU_TESTS") == "1":
        import torch
        import pytest
        if os.environ.get("TDN_EXECUTION_MODE") == "desktop":
            if not torch.cuda.is_available():
                raise pytest.UsageError("Desktop GPU readiness requires real CUDA; a skipped suite is not GPU readiness")
            from tdn.runtime.preflight import verify_runtime
            try:
                verify_runtime("cuda", "gpu-tests")
            except (ValueError, RuntimeError) as error:
                raise pytest.UsageError(f"Desktop GPU readiness preflight failed: {error}") from error
        elif not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID"):
            raise pytest.UsageError("Allocated GPU tests require CUDA in a real Slurm task; a skipped suite is not GPU readiness")
    if not config.option.basetemp:
        config.option.basetemp=str(ROOT/".runtime"/"pytest")
