#!/usr/bin/env python3
"""Allocated A100 40 GB checks. This script never requests an allocation."""
from __future__ import annotations

import argparse
import json
import os
import platform
import pwd
import subprocess
from pathlib import Path


def validate_gpu(torch) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; GPU work must not start")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one task-visible GPU is required")
    props = torch.cuda.get_device_properties(0)
    total = props.total_memory
    if "A100" not in props.name or (props.major, props.minor) != (8, 0):
        raise RuntimeError(f"Expected A100 compute capability 8.0, found {props.name}")
    if "MIG" in props.name.upper() or not 35 * 1024**3 <= total <= 45 * 1024**3:
        raise RuntimeError("Expected a full A100 40 GB-class allocation")
    free, _ = torch.cuda.mem_get_info()
    budget = int(min(30 * 1024**3, 0.8 * total))
    if total - free >= 0.9 * total:
        raise RuntimeError("Device-used memory already exceeds the 90% warning threshold")
    return {"gpu_name": props.name, "compute_capability": [props.major, props.minor],
            "total_bytes": total, "free_bytes_at_start": free,
            "device_used_bytes_at_start": total - free,
            "soft_reserved_budget_bytes": budget}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID"):
        parser.exit(2, "Preflight requires an allocated Slurm job step\n")
    if os.environ.get("SLURM_JOB_ACCOUNT") != "anakano_81":
        parser.exit(2, "Preflight requires charging account anakano_81\n")
    root = Path(__file__).resolve().parents[1]
    if root != Path("/home1/aadaniel/projects/TDN") or pwd.getpwuid(os.getuid()).pw_name != "aadaniel":
        parser.exit(2, "GPU preflight requires the specified CARC root and identity\n")
    import sys
    if Path(sys.prefix).resolve() != root / ".venv" or sys.prefix == sys.base_prefix:
        parser.exit(2, "GPU preflight requires the project Python venv\n")
    output = args.output.resolve()
    if not output.is_relative_to(root):
        parser.exit(2, "Preflight output leaves the project\n")
    import torch

    try:
        data = validate_gpu(torch)
    except RuntimeError as exc:
        parser.exit(2, str(exc) + "\n")
    torch.cuda.reset_peak_memory_stats()
    x = torch.arange(1024, device="cuda:0", dtype=torch.float32)
    if not torch.isfinite((x * x).sum()).item():
        parser.exit(2, "Allocated GPU arithmetic check failed\n")
    torch.cuda.synchronize()
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True, check=False)
        smi_text = smi.stdout.strip() if smi.returncode == 0 else "unavailable"
    except OSError:
        smi_text = "unavailable"
    data.update({"category": "newly_measured_result", "python": platform.python_version(),
                 "torch": torch.__version__, "torch_cuda_runtime": torch.version.cuda,
                 "bfloat16_supported": torch.cuda.is_bf16_supported(),
                 "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                 "slurm_job_id": os.environ["SLURM_JOB_ID"],
                 "slurm_step_id": os.environ["SLURM_STEP_ID"],
                 "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                 "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                 "nvidia_smi": smi_text,
                 "nvidia_smi_note": "May list physical GPUs beyond the task-visible mapping."})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
