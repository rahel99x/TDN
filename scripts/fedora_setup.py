#!/usr/bin/env python3
"""Read-only discovery and project-contained bootstrap for Fedora Slurm desktops."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import errno
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CARC_ROOT = Path("/home1/aadaniel/projects/TDN")
PROFILE = ".tdn/fedora-slurm.json"
TORCH_VERSION = "2.10.0+cu126"
TORCH_INDEX = "https://download.pytorch.org/whl/cu126"
# Conservative full-runtime Linux driver baselines, rather than the less
# restrictive CUDA minor-version compatibility floor.
CUDA_DRIVER_MINIMUM = {"cu126": (560, 28, 3), "cu128": (570, 26), "cu130": (580, 65, 6)}


def inside(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Path escapes this project: {path}")
    return path


def command(argv: list[str], *, timeout: int = 30) -> dict:
    try:
        process = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": argv, "returncode": None, "stdout": "", "stderr": str(exc)}
    return {"command": argv, "returncode": process.returncode,
            "stdout": process.stdout.strip(), "stderr": process.stderr.strip()}


def standalone() -> None:
    if ROOT.resolve() == CARC_ROOT.resolve():
        raise ValueError("Fedora desktop setup cannot bypass the CARC project policy")
    if platform.system() != "Linux":
        raise ValueError("This launcher requires a Linux desktop with Slurm")
    if os.environ.get("CONDA_PREFIX") or os.environ.get("CONDA_SHLVL", "0") not in ("", "0"):
        raise ValueError("Use standalone Python outside an active Conda environment")
    if any(os.environ.get(key) for key in ("SLURM_JOB_ID", "SLURM_STEP_ID")):
        raise ValueError("Run desktop setup from your normal shell, outside a Slurm allocation")


def profile_module():
    # Keep doctor usable on Python 3.14 and before project dependencies exist.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tdn.runtime import desktop_slurm
    return desktop_slurm


def partitions(text: str) -> list[dict]:
    found: dict[str, dict] = {}
    for line in text.splitlines():
        columns = line.strip().split("|", 2)
        if len(columns) != 3:
            continue
        name, state, gres = columns
        name = name.strip()
        default = name.endswith("*")
        name = name.rstrip("*")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            continue
        row = found.setdefault(name, {"name": name, "default": False, "available": False, "gres": []})
        row["default"] |= default
        row["available"] |= state.strip().lower() == "up"
        if gres.strip() not in row["gres"]:
            row["gres"].append(gres.strip())
    return list(found.values())


def has_gpu(row: dict) -> bool:
    return any(re.search(r"(?:^|,)gpu(?::[^,:()]+)?:[1-9][0-9]*(?:\(|,|$)", item)
               for item in row["gres"])


def choose_partition(rows: list[dict], requested: str | None, *, gpu: bool) -> str:
    if requested:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", requested):
            raise ValueError("Partition names must be a single Slurm partition")
        if rows:
            matching = [row for row in rows if row["name"] == requested and row["available"]]
            if not matching:
                raise ValueError(f"Partition {requested!r} is absent or not UP")
            if gpu and not has_gpu(matching[0]):
                raise ValueError(f"Partition {requested!r} does not advertise GPU GRES")
        return requested
    candidates = [row for row in rows if row["available"] and (not gpu or has_gpu(row))]
    defaults = [row for row in candidates if row["default"]]
    if len(defaults) == 1:
        return defaults[0]["name"]
    if len(candidates) == 1:
        return candidates[0]["name"]
    label = "gpu" if gpu else "cpu"
    raise ValueError(f"Cannot choose an unambiguous {label} partition; pass --{label}-partition explicitly")


def gpu_rows(observation: dict) -> list[dict]:
    if observation["returncode"] != 0:
        return []
    rows = []
    for fields in csv.reader(observation["stdout"].splitlines(), skipinitialspace=True):
        if len(fields) != 4:
            continue
        index, name, driver, memory = [field.strip() for field in fields]
        try:
            rows.append({"index": int(index), "name": name, "driver_version": driver,
                         "dedicated_memory_mib": float(memory)})
        except ValueError:
            continue
    return rows


def observe_gpu() -> dict:
    return command(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total",
                    "--format=csv,noheader,nounits"])


def wheel_check(version: str, index: str) -> str:
    match = re.fullmatch(r"https://download\.pytorch\.org/whl/(cu[0-9]+)", index)
    if not match or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+\+cu[0-9]+", version):
        raise ValueError("Choose an exact torch CUDA release including +cuNNN and an official HTTPS CUDA wheel index")
    build = match.group(1)
    if "+" in version and version.split("+", 1)[1] != build:
        raise ValueError("The torch CUDA suffix must match its wheel index")
    if build not in CUDA_DRIVER_MINIMUM:
        raise ValueError(f"No audited driver baseline for {build}; supported indexes: {', '.join(CUDA_DRIVER_MINIMUM)}")
    return build


def check_driver(profile: dict, observation: dict) -> dict:
    build = wheel_check(profile["torch_version"], profile["torch_wheel_index"])
    matching = [row for row in gpu_rows(observation)
                if profile["expected_gpu_name"].lower() in row["name"].lower()]
    if not matching:
        raise ValueError("nvidia-smi could not observe the configured GPU; run doctor and resolve the driver/GPU selection")
    minimum = CUDA_DRIVER_MINIMUM[build]
    for row in matching:
        try:
            installed = tuple(int(part) for part in row["driver_version"].split("."))
        except ValueError as exc:
            raise ValueError("nvidia-smi reported an unrecognized driver version") from exc
        if installed < minimum:
            raise ValueError(f"Driver {row['driver_version']} is below the conservative {build} baseline "
                             f"{'.'.join(map(str, minimum))}; no software or drivers were changed")
        if row["dedicated_memory_mib"] < 1024 * profile["gpu_vram_gib"] * 0.95:
            raise ValueError("Observed dedicated GPU memory is below the configured GPU capacity")
    return {"observed_gpus": matching, "cuda_build": build,
            "minimum_linux_driver": ".".join(map(str, minimum)),
            "scope": "Driver inventory only; CUDA execution is checked inside the allocated GPU job"}


def inspect_python(value: str) -> dict:
    executable = shutil.which(value)
    if not executable:
        raise ValueError(f"Python executable not found: {value}; pass --python python3.13 (or Python 3.11/3.12)")
    query = """import json, pathlib, struct, sys
print(json.dumps(dict(executable=sys.executable, base_executable=sys._base_executable,
version=list(sys.version_info[:3]), bits=struct.calcsize('P')*8,
is_venv=sys.prefix != sys.base_prefix,
is_conda=(pathlib.Path(sys.base_prefix)/'conda-meta').exists())))"""
    observed = command([executable, "-I", "-B", "-c", query])
    try:
        result = json.loads(observed["stdout"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot inspect Python interpreter {value}: {observed['stderr']}") from exc
    if (observed["returncode"] != 0 or result.get("version", [])[:2] not in ([3, 11], [3, 12], [3, 13])
            or result.get("bits") != 64 or result.get("is_venv") or result.get("is_conda")):
        raise ValueError("Select standalone 64-bit Python 3.11, 3.12 or 3.13, outside Conda and outside a venv")
    return result


def write_json(path: Path, payload: dict) -> None:
    path = inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = inside(path.with_name(path.name + ".partial"))
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def tower_config(profile: dict) -> dict:
    result = {"user": profile["user"], "profiles": {"desktop-slurm": {
        "gpu_sampling": False, "weather": False, "budget": False,
        "intervals": {"jobs": 10, "starts": 60, "live": 30, "nodes": 60,
                      "partitions": 120, "finished": 120, "share": 300,
                      "account": 120, "details": 60, "trace": 120}}},
        "logs": {"manifest_file": "logs.json"},
        "research": {"metrics_file": "metrics.jsonl",
                     "contract": str(ROOT / ".tower/contracts/outputs.v1.json"),
                     "planning_file": str(ROOT / "reports/planning.json"), "interval": 5}}
    if profile["account"] is not None:
        result["account"] = profile["account"]
    return result


def doctor(args) -> int:
    """No writes, package imports, CUDA work, scheduler changes or installation."""
    sinfo = command(["sinfo", "--noheader", "--format=%P|%a|%G"])
    smi = observe_gpu()
    interpreters = {}
    for executable in ("python3.13", "python3.12", "python3.11", "python3"):
        if shutil.which(executable):
            try:
                interpreters[executable] = inspect_python(executable)
            except ValueError as exc:
                interpreters[executable] = {"usable": False, "reason": str(exc)}
    observations = {"scope": "Read-only desktop inventory; no numerical or GPU readiness claim",
        "platform": platform.platform(), "root": str(ROOT), "user": getpass.getuser(),
        "python": platform.python_version(), "candidate_interpreters": interpreters,
        "slurm_version": command(["scontrol", "--version"]), "partitions": partitions(sinfo["stdout"]),
        "sinfo": sinfo, "nvidia_smi": smi, "gpus": gpu_rows(smi),
        "associations": command(["sacctmgr", "--noheader", "--parsable2", "show", "association",
                                 "where", f"user={getpass.getuser()}", "format=Cluster,Account,Partition,DefaultQOS"]),
        "default_cuda_wheel": {"version": TORCH_VERSION, "index": TORCH_INDEX,
            "conservative_minimum_linux_driver": ".".join(map(str, CUDA_DRIVER_MINIMUM["cu126"]))},
        "note": "Shared/system RAM is not GPU VRAM. Account defaults to unset; no CARC billing is inherited."}
    print(json.dumps(observations, indent=2, allow_nan=False))
    return 0


def configure(args) -> int:
    standalone()
    path = inside(PROFILE)
    tower_path = inside(".tower/fedora-slurm.json")
    if (path.exists() or tower_path.exists()) and not args.replace:
        raise ValueError("A desktop profile or Tower configuration already exists; inspect it and use --replace explicitly")
    sinfo = command(["sinfo", "--noheader", "--format=%P|%a|%G"])
    rows = partitions(sinfo["stdout"]) if sinfo["returncode"] == 0 else []
    selected = args.python
    if selected is None:
        selected = next((name for name in ("python3.13", "python3.12", "python3.11") if shutil.which(name)), "python3")
    interpreter = inspect_python(selected)
    profile = {"schema_version": 1, "kind": "desktop-slurm", "root": str(ROOT.resolve()),
        "user": getpass.getuser(), "cpu_partition": choose_partition(rows, args.cpu_partition, gpu=False),
        "gpu_partition": choose_partition(rows, args.gpu_partition, gpu=True), "account": args.account,
        "gpu_gres": args.gpu_gres, "expected_gpu_name": args.expected_gpu_name,
        "gpu_vram_gib": args.gpu_vram_gib, "torch_version": args.torch_version,
        "torch_wheel_index": args.torch_wheel_index,
        "python": str(Path(interpreter["executable"]).resolve())}
    wheel_check(profile["torch_version"], profile["torch_wheel_index"])
    profile_module().validate_profile(profile, root=ROOT)
    with acquire_lock():
        write_json(path, profile)
        write_json(tower_path, tower_config(profile))
    print(json.dumps(profile, indent=2))
    print(f"Desktop profile: {path}\nTower configuration: {tower_path}")
    return 0


def install_environment() -> dict[str, str]:
    env = os.environ.copy()
    for name in tuple(env):
        if name.startswith("PIP_") or name in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"):
            env.pop(name)
    for name, relative in {"TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp", "PIP_CACHE_DIR": "cache/pip",
        "XDG_CACHE_HOME": "cache", "TORCH_HOME": "cache/torch", "TORCHINDUCTOR_CACHE_DIR": "cache/inductor",
        "TRITON_CACHE_DIR": "cache/triton", "TORCH_EXTENSIONS_DIR": "cache/extensions",
        "CUDA_CACHE_PATH": "cache/cuda", "MPLCONFIGDIR": "cache/matplotlib",
        "PYTHONPYCACHEPREFIX": "cache/pycache"}.items():
        path = inside(ROOT / ".runtime" / relative)
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    env.update(PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_NO_INPUT="1",
               PYTHONNOUSERSITE="1", MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
    return env


def acquire_lock():
    path = inside(".cache/carc-phase.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        if exc.errno in (errno.EACCES, errno.EAGAIN):
            raise ValueError("A workflow is using this project's venv; setup made no venv changes") from exc
        raise
    return handle


def execute(argv: list[str], env: dict) -> None:
    print("Running: " + subprocess.list2cmdline(argv), flush=True)
    subprocess.run(argv, cwd=ROOT, env=env, check=True)


def verify_existing_venv(prefix: Path, interpreter: dict, env: dict) -> None:
    if not (prefix / "pyvenv.cfg").is_file() or not (prefix / "bin/python").is_file():
        raise ValueError("Existing .venv is not a Python venv; it was preserved")
    query = "import json,sys; print(json.dumps(dict(prefix=sys.prefix, base=sys.base_prefix, executable=sys._base_executable, version=list(sys.version_info[:3]))))"
    output = subprocess.run([str(prefix / "bin/python"), "-I", "-B", "-c", query],
                            cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    report = json.loads(output.stdout)
    if (Path(report["prefix"]).resolve() != prefix or report["prefix"] == report["base"]
            or Path(report["executable"]).resolve() != Path(interpreter["base_executable"]).resolve()
            or report["version"] != interpreter["version"]):
        raise ValueError("Existing .venv uses an incompatible base interpreter; it was preserved")


def setup(args) -> int:
    standalone()
    profile = profile_module().load_profile(root=ROOT)
    interpreter = inspect_python(args.python or profile.get("python", "python3"))
    if args.python and profile.get("python") and Path(interpreter["executable"]).resolve() != Path(profile["python"]).resolve():
        raise ValueError("--python differs from the configured interpreter; configure --replace first")
    driver = check_driver(profile, observe_gpu())
    # The same file is held shared by every allocated worker. Hold it exclusively
    # across creation, pip operations and metadata publication.
    with acquire_lock():
        env = install_environment()
        prefix = inside(".venv")
        if prefix.exists():
            verify_existing_venv(prefix, interpreter, env)
        else:
            execute([interpreter["executable"], "-m", "venv", str(prefix)], env)
        for name in ("bin", "lib", "lib64", "include"):
            inside(prefix / name)
        python = str(prefix / "bin/python")
        execute([python, "-m", "pip", "install", "setuptools==80.9.0", "wheel==0.45.1"], env)
        execute([python, "-m", "pip", "install", f"torch=={profile['torch_version']}",
                 "--index-url", profile["torch_wheel_index"]], env)
        execute([python, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")], env)
        execute([python, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e", str(ROOT)], env)
        execute([python, "-m", "pip", "check"], env)
        query = """import importlib.metadata as m,json,platform,sys,torch
print(json.dumps(dict(python=platform.python_version(), python_executable=sys.executable,
venv=sys.prefix, torch_cuda_runtime=torch.version.cuda,
packages={k:m.version(k) for k in ('torch','numpy','scipy','PyYAML','pytest','mpmath','matplotlib','setuptools','wheel')})))"""
        installed = subprocess.run([python, "-c", query], cwd=ROOT, env=env, check=True,
                                   capture_output=True, text=True)
        report = json.loads(installed.stdout)
        build = wheel_check(profile["torch_version"], profile["torch_wheel_index"])
        expected = profile["torch_version"]
        expected_cuda = build[2:-1] + "." + build[-1]
        if report["packages"]["torch"] != expected or report["torch_cuda_runtime"] != expected_cuda:
            raise ValueError("Installed torch does not match the exact selected CUDA wheel; setup did not publish a success record")
        report.update(status="COMPLETED", scope="Installed desktop Slurm software; numerical/GPU readiness is checked in allocations",
            kind="desktop-slurm", configured_profile=profile, driver_inventory=driver,
            torch_wheel_index=profile["torch_wheel_index"], completed_at=datetime.now(timezone.utc).isoformat(),
            requirements_sha256=hashlib.sha256((ROOT / "requirements.txt").read_bytes()).hexdigest())
        freeze = subprocess.run([python, "-m", "pip", "freeze"], cwd=ROOT, env=env,
                                capture_output=True, text=True, check=True)
        freeze_path = inside(".tdn/fedora-environment-freeze.txt")
        freeze_path.parent.mkdir(parents=True, exist_ok=True)
        freeze_path.write_text(freeze.stdout, encoding="utf-8")
        write_json(inside(".tdn/fedora-environment.json"), report)
    print(f"Project venv ready: {prefix}\nSoftware record: {ROOT / '.tdn/fedora-environment.json'}")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="read-only Slurm, Python and NVIDIA inventory")
    configuration = commands.add_parser("configure", help="write this checkout's explicit desktop Slurm profile")
    configuration.add_argument("--cpu-partition")
    configuration.add_argument("--gpu-partition")
    configuration.add_argument("--account", default=None)
    configuration.add_argument("--gpu-gres", default="gpu:1")
    configuration.add_argument("--expected-gpu-name", default="RTX 4090")
    configuration.add_argument("--gpu-vram-gib", type=float, default=24.0)
    configuration.add_argument("--torch-version", default=TORCH_VERSION)
    configuration.add_argument("--torch-wheel-index", default=TORCH_INDEX)
    configuration.add_argument("--python")
    configuration.add_argument("--replace", action="store_true")
    commands.add_parser("setup", help="install the standalone project venv after checking the driver").add_argument("--python")
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        return {"doctor": doctor, "configure": configure, "setup": setup}[args.command](args)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"TDN Fedora: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
