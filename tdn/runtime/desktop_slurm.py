"""Explicit desktop Slurm policy, with read-only scheduler verification.

Profile loading is deliberately standard-library-only so login/setup scripts do
not need a working torch installation. Mocked scheduler records exercise policy;
only the real allocated worker can establish actual CPU or GPU readiness.
"""
from __future__ import annotations

import getpass
import json
import math
import os
from pathlib import Path
import re
import subprocess

from .storage import CARC_ROOT, contained_path, project_root


def _root(root: Path | None) -> Path:
    value = (root or project_root()).resolve()
    if value == CARC_ROOT:
        raise ValueError("Desktop Slurm execution cannot bypass the CARC root policy")
    return value


def profile_path(path: str | Path | None = None, *, root: Path | None = None) -> Path:
    root = _root(root)
    selected = Path(path or os.environ.get("TDN_SLURM_CONFIG", ".tdn/fedora-slurm.json"))
    return contained_path(selected if selected.is_absolute() else root / selected, root)


def validate_profile(profile: dict, *, root: Path | None = None) -> dict:
    """Validate the explicit local machine configuration; never discover defaults."""
    root = _root(root)
    required = {"schema_version", "kind", "root", "user", "cpu_partition", "gpu_partition",
                "account", "gpu_gres", "expected_gpu_name", "gpu_vram_gib", "torch_version",
                "torch_wheel_index"}
    if not isinstance(profile, dict) or not required <= profile.keys() or profile.keys() - required - {"python"}:
        raise ValueError("Desktop Slurm profile has missing or unknown fields")
    if type(profile["schema_version"]) is not int or profile["schema_version"] != 1 or profile["kind"] != "desktop-slurm":
        raise ValueError("Unsupported desktop Slurm profile schema or kind")
    if not isinstance(profile["root"], str) or not Path(profile["root"]).is_absolute() or Path(profile["root"]).resolve() != root:
        raise ValueError("Desktop Slurm profile root must match this checkout")
    if profile["user"] != getpass.getuser():
        raise ValueError("Desktop Slurm profile user must match the current user")
    for key in ("cpu_partition", "gpu_partition", "user"):
        if not isinstance(profile[key], str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", profile[key]):
            raise ValueError(f"Invalid desktop Slurm profile {key}")
    if profile["account"] is not None and (not isinstance(profile["account"], str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", profile["account"])):
        raise ValueError("Desktop Slurm account must be a scheduler account token or null")
    if not isinstance(profile["gpu_gres"], str) or not re.fullmatch(r"gpu:(?:[A-Za-z0-9_.-]+:)?1", profile["gpu_gres"]):
        raise ValueError("Desktop Slurm gpu_gres must request exactly one GPU")
    name = profile["expected_gpu_name"]
    if not isinstance(name, str) or not name.strip() or any(ord(char) < 32 for char in name):
        raise ValueError("Desktop Slurm expected_gpu_name must be a nonempty GPU name substring")
    vram = profile["gpu_vram_gib"]
    if isinstance(vram, bool) or not isinstance(vram, (int, float)) or not math.isfinite(vram) or not 1 <= vram <= 256:
        raise ValueError("Desktop Slurm gpu_vram_gib must be finite positive dedicated VRAM")
    version = profile["torch_version"]
    index = profile["torch_wheel_index"]
    if not isinstance(version, str) or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+\+cu[0-9]+", version):
        raise ValueError("Desktop Slurm torch_version must be an exact CUDA torch release including +cuNNN")
    if not isinstance(index, str) or not re.fullmatch(r"https://download\.pytorch\.org/whl/cu[0-9]+", index):
        raise ValueError("Desktop Slurm requires an official HTTPS CUDA wheel index")
    if version.split("+", 1)[1] != index.rsplit("/", 1)[1]:
        raise ValueError("Desktop Slurm torch version and wheel index CUDA builds disagree")
    if "python" in profile and (not isinstance(profile["python"], str) or not Path(profile["python"]).is_absolute()):
        raise ValueError("Desktop Slurm optional python must name an absolute executable")
    return dict(profile)


def load_profile(path: str | Path | None = None, *, root: Path | None = None) -> dict:
    root = _root(root)
    source = profile_path(path, root=root)
    if source.stat().st_size > 65536:
        raise ValueError("Desktop Slurm profile exceeds the 64 KiB limit")

    def unique_fields(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate desktop Slurm profile field: {key}")
            result[key] = value
        return result

    return validate_profile(json.loads(source.read_text(), object_pairs_hook=unique_fields), root=root)


def desktop_memory_policy() -> dict:
    """Dedicated VRAM only; host/shared RAM never increases this budget."""
    return {"soft_cap_bytes": 18 * 2**30, "soft_fraction": .75, "hard_fraction": .90}


def _fields(line: str) -> dict[str, str]:
    # scontrol -o does not reliably quote paths with spaces. Split at field names,
    # not whitespace, so WorkDir=/home/user/my project remains an exact path.
    markers = list(re.finditer(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=", line))
    result = {}
    for index, marker in enumerate(markers):
        key = marker.group(1)
        if key in result:
            raise ValueError(f"Ambiguous repeated scontrol field: {key}")
        stop = markers[index + 1].start() if index + 1 < len(markers) else len(line)
        result[key] = line[marker.end():stop].strip()
    return result


def _show(kind: str, identifier: str) -> dict[str, str]:
    try:
        proc = subprocess.run(["scontrol", "show", kind, identifier, "-o"],
                              text=True, capture_output=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"Cannot verify the active desktop Slurm {kind}") from exc
    if proc.returncode:
        raise ValueError(f"Cannot verify the active desktop Slurm {kind}: {proc.stderr.strip()}")
    return _fields(proc.stdout)


def _one_gpu(fields: dict[str, str]) -> bool:
    """Accept Slurm TRES and GRES forms without counting typed TRES twice."""
    counts = []
    for key in ("AllocTRES", "TRES"):
        generic, typed = [], []
        for item in fields.get(key, "").split(","):
            match = re.fullmatch(r"gres/gpu(?::[^=,]+)?=([0-9]+)", item)
            if match:
                (generic if item.startswith("gres/gpu=") else typed).append(int(match.group(1)))
        if generic:
            counts.extend(generic)
        if typed:
            counts.append(sum(typed))
    for key in ("TresPerNode", "Gres"):
        gpu_items = []
        for item in fields.get(key, "").split(","):
            match = re.fullmatch(r"(?:gres/)?gpu:(?:[A-Za-z0-9_.-]+:)?([0-9]+)(?:\([^)]*\))?", item)
            if match:
                gpu_items.append(int(match.group(1)))
        if gpu_items:
            counts.append(sum(gpu_items))
    return bool(counts) and all(count == 1 for count in counts)


def verify_allocation(device: str, *, profile: dict | None = None, root: Path | None = None) -> dict:
    """Verify this process's running job and srun step using scheduler records."""
    root = _root(root)
    profile = load_profile(root=root) if profile is None else validate_profile(profile, root=root)
    if device not in ("cpu", "cuda"):
        raise ValueError("Desktop Slurm device must be cpu or cuda")
    job_id = os.environ.get("SLURM_JOB_ID", "")
    step_id = os.environ.get("SLURM_STEP_ID", "")
    if not job_id.isdecimal() or not step_id.isdecimal():
        raise ValueError("Desktop Slurm compute must execute through a real numeric srun task step")
    job = _show("job", job_id)
    step = _show("step", f"{job_id}.{step_id}")
    partition = profile["gpu_partition" if device == "cuda" else "cpu_partition"]
    owner = f"{profile['user']}({os.getuid()})"
    if job.get("JobId") != job_id or job.get("JobState") != "RUNNING" or job.get("UserId") != owner:
        raise ValueError("Actual desktop Slurm job is not this user's running allocation")
    if job.get("Partition") != partition:
        raise ValueError("Actual desktop Slurm job partition differs from the profile")
    if profile["account"] is not None and job.get("Account") != profile["account"]:
        raise ValueError("Actual desktop Slurm job account differs from the profile")
    workdir = job.get("WorkDir", "")
    if not Path(workdir).is_absolute() or Path(workdir).resolve() != root:
        raise ValueError("Actual desktop Slurm job WorkDir must be this checkout")
    # Slurm versions print step ownership either as the numeric UID or as the
    # same name(uid) form used by job records. Both must identify this process.
    if (step.get("StepId") != f"{job_id}.{step_id}" or step.get("State") != "RUNNING"
            or step.get("UserId") not in {owner, str(os.getuid())} or step.get("Partition") != partition):
        raise ValueError("Actual desktop Slurm srun step does not match this running task")
    if any(fields.get(key, "1") != "1" for fields, key in ((job, "NumNodes"), (step, "Nodes"), (step, "Tasks"))):
        raise ValueError("Desktop Slurm workflow requires one node and one task")
    if device == "cuda" and (not _one_gpu(job) or not _one_gpu(step)):
        raise ValueError("Actual desktop Slurm job and srun step must each allocate exactly one GPU")
    return {"execution_mode": "desktop-slurm", "device": device, "job_id": job_id,
            "step_id": step_id, "job": job, "step": step}


def verify_cuda_device(profile: dict | None = None) -> dict:
    """Inspect the actual task-visible device; this does not replace allocation checks."""
    profile = load_profile() if profile is None else validate_profile(profile)
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Desktop Slurm requires exactly one task-visible CUDA GPU")
    expected_version = profile["torch_version"]
    actual_version = str(torch.__version__)
    if actual_version != expected_version:
        raise ValueError("Installed torch version differs from the desktop Slurm profile")
    expected_runtime = profile["torch_wheel_index"].rsplit("cu", 1)[1]
    if not torch.version.cuda or torch.version.cuda.replace(".", "") != expected_runtime:
        raise ValueError("Installed torch CUDA runtime differs from the desktop Slurm wheel index")
    torch.cuda.set_device(0)
    prop = torch.cuda.get_device_properties(0)
    if profile["expected_gpu_name"].casefold() not in prop.name.casefold():
        raise ValueError("Task-visible GPU name differs from the desktop Slurm profile")
    expected = profile["gpu_vram_gib"]
    if abs(prop.total_memory / 2**30 - expected) > max(1., expected / 12.):
        raise ValueError("Task-visible dedicated GPU VRAM differs from the desktop Slurm profile")
    free, total = torch.cuda.mem_get_info(0)
    if not 0 < free <= total or abs(total - prop.total_memory) > 2**20:
        raise ValueError("Cannot verify consistent task-visible CUDA memory")
    policy = desktop_memory_policy()
    if total - free >= policy["hard_fraction"] * total:
        raise ValueError("Desktop GPU is already above the hard dedicated-memory use limit")
    budget = int(min(policy["soft_cap_bytes"], policy["soft_fraction"] * total,
                     policy["soft_fraction"] * free))
    return {"execution_mode": "desktop-slurm", "device_index": 0, "gpu_name": prop.name,
            "compute_capability": [prop.major, prop.minor], "total_bytes": int(total),
            "free_bytes_at_start": int(free), "device_used_bytes_at_start": int(total - free),
            "soft_reserved_budget_bytes": budget, "memory_policy": policy,
            "torch": actual_version, "torch_cuda_runtime": torch.version.cuda}
