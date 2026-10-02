"""Read-only runtime checks; no scheduler submissions or hardware changes."""
from __future__ import annotations
import getpass
import os
from pathlib import Path
import subprocess
import sys
from .storage import CARC_ROOT, project_root


def execution_mode() -> str:
    """Desktop is explicit and cannot relax checks for a real CARC task."""
    root = project_root()
    requested = os.environ.get("TDN_EXECUTION_MODE", "")
    if requested not in ("", "desktop"):
        raise ValueError("TDN_EXECUTION_MODE must be unset or desktop")
    slurm = any(os.environ.get(name) for name in ("SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_JOB_ACCOUNT"))
    if requested == "desktop":
        if root == CARC_ROOT or slurm:
            raise ValueError("Desktop execution cannot bypass CARC or an active Slurm environment")
        return "desktop"
    return "carc" if root == CARC_ROOT or slurm else "local-cpu"


def desktop_cuda_device() -> int:
    """Select one local NVIDIA device without changing task-visible GPUs."""
    if execution_mode() != "desktop":
        raise ValueError("Local CUDA device selection requires explicit desktop execution")
    raw = os.environ.get("TDN_DESKTOP_CUDA_DEVICE", "0")
    if not raw.isdecimal():
        raise ValueError("TDN_DESKTOP_CUDA_DEVICE must be a nonnegative integer")
    index = int(raw)
    import torch
    if not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use CPU or install a compatible NVIDIA PyTorch wheel")
    if index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device index {index} is not visible")
    torch.cuda.set_device(index)
    return index


def verify_runtime(device: str, stage: str):
    root=project_root()
    mode=execution_mode()
    if Path(sys.prefix).resolve() != (root/".venv").resolve() or sys.prefix == sys.base_prefix:
        raise ValueError("Use the Python venv at the project root/.venv")
    base = Path(sys.base_prefix).resolve()
    if (any(name in str(base).lower() for name in ("conda", "miniforge", "mambaforge"))
            or (base / "conda-meta").is_dir() or os.environ.get("CONDA_PREFIX")):
        raise ValueError("Use a standalone Python venv, outside an active Conda environment")
    if device not in ("cpu", "cuda"):
        raise ValueError("Select device cpu or cuda")
    is_carc=root == CARC_ROOT
    if is_carc:
        if getpass.getuser() != "aadaniel": raise ValueError("CARC user must be aadaniel")
        if stage != "validate-config":
            if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID"):
                raise ValueError("CARC compute stages must execute through srun in an allocation")
            if os.environ.get("SLURM_JOB_ACCOUNT") != "anakano_81":
                raise ValueError("Allocated Slurm charging account must be anakano_81")
    if device == "cuda":
        if mode == "desktop":
            desktop_cuda_device()
            return
        if not is_carc or not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID"):
            raise ValueError("CUDA stages require an actual CARC Slurm task at the configured root")
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("Require exactly one task-visible CUDA GPU")
        prop=torch.cuda.get_device_properties(0)
        if "A100" not in prop.name or (prop.major,prop.minor)!=(8,0) or not 35*2**30 <= prop.total_memory <=45*2**30:
            raise ValueError("Require a full A100 40 GB-class device")
        # Read the actual allocation; visibility of a partition is not authorization.
        proc=subprocess.run(["scontrol","show","job",os.environ["SLURM_JOB_ID"],"-o"],text=True,capture_output=True)
        if proc.returncode: raise ValueError("Cannot verify the active Slurm GPU allocation")
        fields=dict(token.split("=",1) for token in proc.stdout.split() if "=" in token)
        if fields.get("Account") != "anakano_81" or fields.get("Partition") != "gpu" or fields.get("JobState") != "RUNNING":
            raise ValueError("Actual allocation does not match account anakano_81 and running gpu partition")
        if not fields.get("UserId","").startswith("aadaniel("):
            raise ValueError("Actual allocation is not owned by aadaniel")
