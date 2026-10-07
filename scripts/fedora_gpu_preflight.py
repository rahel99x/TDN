#!/usr/bin/env python3
"""Measure the configured desktop GPU inside its verified Slurm task."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        from tdn.runtime.storage import configure_storage, contained_path
        from tdn.runtime.preflight import execution_mode, verify_runtime
        from tdn.runtime.desktop_slurm import desktop_memory_policy, load_profile
        from tdn.runtime.metadata import write_json
        from tdn.runtime.precision import reference_precision

        configure_storage()
        if execution_mode() != "desktop-slurm":
            raise ValueError("Fedora GPU preflight requires explicit desktop-slurm execution")
        verify_runtime("cuda", "premix-preflight")
        output = contained_path(args.output)
        if output.exists():
            raise ValueError("Preserve existing preflight evidence; choose a fresh workflow")
        import torch
        reference_precision()
        profile = load_profile()
        policy = desktop_memory_policy()
        props = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        budget = int(min(policy["soft_cap_bytes"], policy["soft_fraction"] * total,
                         policy["soft_fraction"] * free))
        torch.cuda.reset_peak_memory_stats()
        x = torch.arange(1024, device="cuda", dtype=torch.float32)
        if not torch.isfinite((x * x).sum()).item():
            raise RuntimeError("Allocated desktop GPU arithmetic check failed")
        torch.cuda.synchronize()
        try:
            smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                  "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
            smi_text = smi.stdout.strip() if smi.returncode == 0 else "unavailable"
        except (OSError, subprocess.TimeoutExpired):
            smi_text = "unavailable"
        data = {"category": "newly_measured_result", "execution_mode": "desktop-slurm",
                "gpu_name": props.name, "compute_capability": [props.major, props.minor],
                "total_bytes": total, "free_bytes_at_start": free,
                "device_used_bytes_at_start": total - free, "soft_reserved_budget_bytes": budget,
                "memory_policy": policy, "host_memory_is_gpu_vram": False,
                "expected_gpu_name": profile["expected_gpu_name"],
                "python": platform.python_version(), "torch": torch.__version__,
                "torch_cuda_runtime": torch.version.cuda,
                "bfloat16_supported": torch.cuda.is_bf16_supported(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "slurm_job_id": os.environ["SLURM_JOB_ID"], "slurm_step_id": os.environ["SLURM_STEP_ID"],
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "nvidia_smi": smi_text,
                "nvidia_smi_note": "May list physical GPUs beyond task-visible mapping."}
        write_json(output, data)
        print(json.dumps(data, indent=2))
        return 0
    except (ValueError, RuntimeError, OSError) as error:
        print(f"TDN Fedora preflight: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
