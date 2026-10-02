"""Read-only runtime checks; no scheduler submissions or hardware changes."""
from __future__ import annotations
import getpass
import os
from pathlib import Path
import subprocess
import sys
from .storage import CARC_ROOT, project_root

def verify_runtime(device: str, stage: str):
    root=project_root()
    if Path(sys.prefix).resolve() != (root/".venv").resolve() or sys.prefix == sys.base_prefix:
        raise ValueError("Use the Python venv at the project root/.venv")
    if "conda" in str(sys.base_prefix).lower() or os.environ.get("CONDA_PREFIX"):
        raise ValueError("Use a standalone Python venv, outside an active Conda environment")
    is_carc=root == CARC_ROOT
    if is_carc:
        if getpass.getuser() != "aadaniel": raise ValueError("CARC user must be aadaniel")
        if stage != "validate-config":
            if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID"):
                raise ValueError("CARC compute stages must execute through srun in an allocation")
            if os.environ.get("SLURM_JOB_ACCOUNT") != "anakano_81":
                raise ValueError("Allocated Slurm charging account must be anakano_81")
    if device == "cuda":
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
