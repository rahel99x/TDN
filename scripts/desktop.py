#!/usr/bin/env python3
"""Queue-free desktop orchestration; bootstrap uses only the standard library."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]


def inside(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Path escapes the project: {path}")
    return path


def prepare_environment(cuda_device: int = 0) -> None:
    if ROOT == Path("/home1/aadaniel/projects/TDN").resolve():
        raise ValueError("Desktop execution cannot bypass allocations at the fixed CARC project root")
    if any(os.environ.get(key) for key in ("SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_JOB_ACCOUNT")):
        raise ValueError("Desktop execution cannot run inside a Slurm allocation")
    if os.environ.get("CONDA_PREFIX") or (Path(sys.base_prefix) / "conda-meta").exists():
        raise ValueError("Use standalone Python outside an active Conda environment")
    for name in ("PIP_TARGET", "PIP_PREFIX", "PIP_USER", "PIP_LOG", "PIP_BUILD_TRACKER",
                 "PIP_EXTRA_INDEX_URL", "PYTHONUSERBASE", "PYTHONPATH"):
        os.environ.pop(name, None)
    os.environ["PIP_CONFIG_FILE"] = os.devnull
    os.environ["TDN_EXECUTION_MODE"] = "desktop"
    os.environ["TDN_PROJECT_ROOT"] = str(ROOT)
    os.environ["TDN_DESKTOP_CUDA_DEVICE"] = str(cuda_device)
    for name, child in {
        "TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp",
        "PIP_CACHE_DIR": "cache/pip", "XDG_CACHE_HOME": "cache",
        "TORCH_HOME": "cache/torch", "TORCHINDUCTOR_CACHE_DIR": "cache/inductor",
        "TRITON_CACHE_DIR": "cache/triton", "TORCH_EXTENSIONS_DIR": "cache/extensions",
        "CUDA_CACHE_PATH": "cache/cuda", "MPLCONFIGDIR": "cache/matplotlib",
        "PYTHONPYCACHEPREFIX": "cache/pycache",
    }.items():
        target = inside(ROOT / ".runtime" / child)
        target.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(target)
    os.environ.update(PYTHONNOUSERSITE="1", MPLBACKEND="Agg", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    tempfile.tempdir = os.environ["TEMP"]
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"


def write_json(path: Path, payload: dict) -> None:
    path = inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def validate_venv() -> None:
    prefix = inside(ROOT / ".venv")
    if Path(sys.prefix).resolve() != prefix or sys.prefix == sys.base_prefix:
        raise ValueError("Run with this project's .venv interpreter; complete Setup.ps1 first")
    if os.environ.get("CONDA_PREFIX") or (Path(sys.base_prefix) / "conda-meta").exists():
        raise ValueError("Use standalone Python, outside an active Conda environment")
    if sys.version_info[:2] not in ((3, 11), (3, 12)):
        raise ValueError("The desktop setup supports standalone Python 3.11 or 3.12")


def interpreter(args) -> int:
    if os.environ.get("CONDA_PREFIX") or sys.prefix != sys.base_prefix or (Path(sys.base_prefix) / "conda-meta").exists():
        raise ValueError("Use standalone Python outside an active Conda environment")
    if sys.version_info[:2] not in ((3, 11), (3, 12)) or sys.maxsize <= 2**32:
        raise ValueError("Use standalone 64-bit Python 3.11 or 3.12")
    print(sys.executable)
    return 0


def verify_venv(args) -> int:
    validate_venv()
    observed = subprocess.check_output([args.base_python, str(ROOT / "scripts" / "desktop.py"), "interpreter"],
                                       text=True).strip()
    if Path(observed).resolve() != Path(sys._base_executable).resolve():
        raise ValueError("Existing venv uses a different base interpreter; preserve it and resolve manually")
    return 0


def environment_report(args) -> int:
    validate_venv()
    import torch
    selection = re.fullmatch(r"https://download\.pytorch\.org/whl/(cpu|cu[0-9]+)", args.torch_index)
    if not selection:
        raise ValueError("Use an official HTTPS PyTorch CPU/CUDA wheel index")
    build = selection.group(1)
    expected_cuda = None if build == "cpu" else build[2:-1] + "." + build[-1]
    if torch.version.cuda != expected_cuda:
        raise ValueError(f"Installed torch CUDA runtime {torch.version.cuda!r} does not match selected wheel index {build}")
    report = {
        "scope": "Installed desktop software; numerical/GPU readiness checks have not run",
        "python": platform.python_version(), "python_executable": sys.executable, "venv": sys.prefix,
        "torch_wheel_index": args.torch_index, "torch_cuda_runtime": torch.version.cuda,
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "numpy", "scipy", "PyYAML", "pytest", "mpmath", "matplotlib", "setuptools", "wheel")},
    }
    write_json(ROOT / ".runtime" / "windows-environment.json", report)
    print(json.dumps(report, indent=2))
    return 0


def gpu_check(args) -> int:
    validate_venv()
    from tdn.runtime.preflight import verify_runtime
    verify_runtime("cuda", "desktop-setup")
    import torch
    observed = {}
    for dtype in (torch.float32, torch.float64):
        values = torch.arange(16, dtype=dtype, device="cuda")
        result = values.square().sum()
        if not torch.isfinite(result) or float(result.cpu()) != 1240.0:
            raise ValueError(f"Basic CUDA arithmetic failed for {dtype}")
        observed[str(dtype)] = float(result.cpu())
    torch.cuda.synchronize()
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    report = {"scope": "Measured basic CUDA access/arithmetic; full numerical GPU tests are still required",
              "passed": True, "cuda_device": index, "gpu": properties.name,
              "dedicated_vram_bytes": properties.total_memory, "torch": torch.__version__,
              "torch_cuda_runtime": torch.version.cuda, "arithmetic_checks": observed}
    write_json(ROOT / ".runtime" / "setup-gpu-preflight.json", report)
    print(json.dumps(report, indent=2))
    return 0


def doctor(args) -> int:
    """Observe hardware; driver probing does not require PyTorch or a venv."""
    report = {
        "scope": "Observed desktop hardware/software; no numerical or GPU validation claim",
        "platform": platform.platform(), "python": platform.python_version(),
        "python_executable": sys.executable, "project_root": str(ROOT),
        "requested_cuda_device": args.cuda_device,
        "gpu_memory_policy": "Dedicated VRAM only; shared/system RAM does not extend the CUDA budget",
    }
    smi = shutil.which("nvidia-smi")
    if smi:
        observed = subprocess.run(
            [smi, "--query-gpu=index,name,driver_version,memory.total,memory.free", "--format=csv,noheader,nounits"],
            text=True, capture_output=True, timeout=30,
        )
        report["nvidia_smi"] = {"returncode": observed.returncode,
                                "csv_columns": "index,name,driver_version,total_MiB,free_MiB",
                                "stdout": observed.stdout.strip(), "stderr": observed.stderr.strip()}
    else:
        report["nvidia_smi"] = {"status": "UNAVAILABLE", "note": "CPU mode remains available"}
    try:
        report["installed_torch_version"] = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        report["installed_torch_version"] = None
    output = inside(args.output)
    write_json(output, report)
    print(json.dumps(report, indent=2))
    print(f"Doctor report: {output}")
    return 0


def execute(command: list[str], log_path: Path) -> int:
    """Stream output to the terminal and a project-contained UTF-8 transcript."""
    log_path = inside(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("Running: " + subprocess.list2cmdline(command), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\nCOMMAND: " + subprocess.list2cmdline(command) + "\n")
        log.flush()
        with subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace") as proc:
            assert proc.stdout is not None
            try:
                for line in proc.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
            except KeyboardInterrupt:
                # The foreground console also delivers Ctrl+C to the child on Windows.
                # Give its safe-boundary checkpoint handler time to finish.
                print("Interrupt requested; waiting for the current process to stop safely...", flush=True)
                try:
                    remaining, _ = proc.communicate(timeout=60)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    remaining, _ = proc.communicate()
                if remaining:
                    print(remaining, end="", flush=True)
                    log.write(remaining)
                # Even if the child happened to finish successfully, Ctrl+C stops
                # the serial pipeline rather than starting another stage.
                return 75 if proc.returncode == 75 else 130
            return proc.wait()


def run(args) -> int:
    validate_venv()
    from tdn.config import config_hash, load_config
    from tdn.runtime.metadata import software_metadata
    config_path = inside(args.config)
    config = load_config(config_path)
    if config["purpose"] != "development" or config["runtime"]["confirmatory_authorized"]:
        raise ValueError("Desktop launchers currently support development runs only")
    if not 1 <= args.max_steps <= 1000 or config["training"]["max_steps"] > args.max_steps:
        raise ValueError("Config optimizer steps exceed the explicit 1-1000 step desktop budget")
    if len(config["problem"]["grid"]) > 2 or any(n > 64 for n in config["problem"]["grid"]):
        raise ValueError("The initial desktop pipeline is limited to 1-D/2-D grids up to 64 per axis")
    if config["precision"]["compile_mode"] != "eager" or config["precision"]["network_autocast"] != "none":
        raise ValueError("Desktop runs currently use measured eager FP32; compiled/BF16 modes are unvalidated")
    if args.resume and not args.run_id:
        raise ValueError("--resume requires the original --run-id")
    run_id = args.run_id or ("desktop-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("Use a simple run identifier containing letters, numbers, dot, dash or underscore")
    base = inside(ROOT / "runs" / run_id)
    manifest_path = inside(base / "pipeline.json")
    config_digest = config_hash(config)
    software = software_metadata()
    source_digest = software["source_tree_sha256"]
    software_snapshot = {key: software[key] for key in ("python", "torch", "numpy", "scipy", "torch_cuda_runtime")}
    software_snapshot["platform"] = platform.platform()
    hardware = {"device": "cpu"}
    if args.device == "cuda" and not args.dry_run:
        # Fail before the CPU audit/data work if the requested GPU cannot run.
        from tdn.runtime.preflight import verify_runtime
        verify_runtime("cuda", "desktop-run")
        import torch
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        hardware = {"device": "cuda", "name": properties.name, "dedicated_vram_bytes": properties.total_memory,
                    "compute_capability": [properties.major, properties.minor],
                    "uuid": str(getattr(properties, "uuid", "unavailable"))}
    stages = ["cpu-tests", "audit", "generate"]
    if args.device == "cuda":
        stages.append("gpu-tests")
    stages += ["calibrate", "train", "evaluate", "benchmark"]
    dataset = inside(base / "dataset")
    audit_report = inside(base / "audit" / "audit.json")
    calibration_report = inside(base / "calibrate" / "calibration.json")
    best = inside(base / "train" / "checkpoints" / "best.pt")
    commands = {}
    for stage in stages:
        stage_dir = inside(base / stage)
        if stage == "cpu-tests":
            command = [sys.executable, "-m", "pytest", "-q", "-m", "not gpu", "--basetemp", str(stage_dir / "pytest-work")]
        elif stage == "gpu-tests":
            command = [sys.executable, "-m", "pytest", "-q", "tests/test_desktop_gpu.py", "-m", "gpu", "--basetemp", str(stage_dir / "pytest-work")]
        else:
            # Numerical audits and immutable teacher data stay on CPU in both modes.
            device = "cpu" if stage in ("audit", "generate") else args.device
            command = [sys.executable, "-u", "-m", "tdn.cli", stage, "--config", str(config_path),
                       "--run-dir", str(stage_dir), "--dataset", str(dataset), "--device", device,
                       "--gate-report", str(audit_report), "--calibration-report", str(calibration_report)]
            if stage in ("evaluate", "benchmark"):
                command += ["--resume", str(best)]
        commands[stage] = command
    print(f"Run identifier: {run_id}\nRun directory: {base}\nSequential stages: {' -> '.join(stages)}")
    print(f"Device: {args.device}; configured optimizer steps: {config['training']['max_steps']}; budget: {args.max_steps}")
    if args.dry_run:
        if base.exists() and not args.resume:
            raise ValueError("Run directory exists; choose a fresh identifier or --resume")
        for command in commands.values():
            print(subprocess.list2cmdline(command))
        print("Dry run: no stages executed and no run directory created")
        return 0
    if args.resume:
        if not manifest_path.is_file():
            raise ValueError("Resume requires the original pipeline.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for field, value in (("config_hash", config_digest), ("source_tree_sha256", source_digest),
                             ("device", args.device), ("cuda_device", args.cuda_device),
                             ("software", software_snapshot), ("hardware", hardware), ("max_steps_budget", args.max_steps)):
            if manifest.get(field) != value:
                raise ValueError(f"Resume {field} differs from the original run; preserve results and start a fresh run")
    else:
        if base.exists():
            raise ValueError("Run directory already exists; preserve it and choose a fresh identifier")
        base.mkdir(parents=True)
        manifest = {"scope": "Newly measured desktop development pipeline", "run_id": run_id,
                    "config": str(config_path), "config_hash": config_digest, "source_tree_sha256": source_digest,
                    "device": args.device, "cuda_device": args.cuda_device, "max_steps_budget": args.max_steps,
                    "software": software_snapshot, "hardware": hardware,
                    "stages": {}, "gpu_checks": "PENDING" if args.device == "cuda" else "UNRUN_CPU_MODE"}
    manifest["status"] = "RUNNING"
    manifest.pop("error", None)
    write_json(manifest_path, manifest)
    for stage in stages:
        stage_dir = inside(base / stage)
        previous = manifest["stages"].get(stage, {})
        def fail_before_stage(message: str) -> None:
            manifest["status"] = "FAILED"
            manifest["error"] = message
            manifest["stages"][stage] = {"status": "FAILED", "error": message,
                                          "actually_ran": False, "previous_attempt": previous or None}
            write_json(manifest_path, manifest)
            raise ValueError(message)
        if args.resume and previous.get("status") == "COMPLETED":
            if stage not in ("cpu-tests", "gpu-tests") and not (stage_dir / "COMPLETED").is_file():
                fail_before_stage(f"Completed {stage} stage marker is missing; preserve the inconsistent run")
            print(f"Already completed: {stage}")
            continue
        command = commands[stage].copy()
        if stage == "train" and args.resume:
            last = inside(stage_dir / "checkpoints" / "last.pt")
            if last.is_file():
                command += ["--resume", str(last)]
            elif previous.get("status") in ("PAUSED_NEEDS_RESUME", "RUNNING", "INTERRUPTED"):
                fail_before_stage("Interrupted training has no last.pt checkpoint; preserve results and use a fresh run")
        if stage in ("evaluate", "benchmark") and not best.is_file():
            fail_before_stage("Training produced no feasible validation-selected best.pt; evaluation cannot proceed")
        if stage == "gpu-tests":
            if not inside(ROOT / "tests" / "test_desktop_gpu.py").is_file():
                fail_before_stage("Desktop GPU test suite is missing; CUDA readiness cannot be claimed")
            os.environ["TDN_REQUIRE_GPU_TESTS"] = "1"
        else:
            os.environ.pop("TDN_REQUIRE_GPU_TESTS", None)
        stage_dir.mkdir(parents=True, exist_ok=True)
        entry = {"status": "RUNNING", "command": command, "actually_ran": True}
        manifest["stages"][stage] = entry
        write_json(manifest_path, manifest)
        started = time.monotonic()
        try:
            exit_code = execute(command, inside(base / "logs" / f"{stage}.log"))
        except (OSError, subprocess.SubprocessError) as error:
            fail_before_stage(f"Could not execute {stage}: {error}")
        entry.update(exit_code=exit_code, elapsed_seconds=time.monotonic() - started,
                     status="COMPLETED" if exit_code == 0 else "PAUSED_NEEDS_RESUME" if exit_code == 75 else "INTERRUPTED" if exit_code == 130 else "FAILED")
        if stage in ("cpu-tests", "gpu-tests"):
            write_json(stage_dir / "stage.json", entry)
        if stage == "gpu-tests":
            manifest["gpu_checks"] = entry["status"]
        manifest["status"] = entry["status"] if exit_code else "RUNNING"
        write_json(manifest_path, manifest)
        if exit_code:
            print(f"Stopped at {stage} (exit {exit_code}); see {base / 'logs' / (stage + '.log')}", file=sys.stderr)
            if exit_code == 75:
                print(f"Resume with the same config/device and --run-id {run_id} --resume", file=sys.stderr)
            return exit_code
    manifest["status"] = "COMPLETED"
    write_json(manifest_path, manifest)
    print(f"Desktop pipeline completed. Results: {base}")
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="command", required=True)
    subs.add_parser("interpreter", help="Internal standalone Python discovery")
    v = subs.add_parser("verify-venv", help="Internal existing-venv compatibility check")
    v.add_argument("--base-python", required=True)
    e = subs.add_parser("environment", help="Record installed software without running numerical kernels")
    e.add_argument("--torch-index", required=True)
    g = subs.add_parser("gpu-check", help="Verify basic real CUDA access after installing an explicit CUDA wheel")
    g.add_argument("--cuda-device", type=int, default=0)
    d = subs.add_parser("doctor", help="Observe NVIDIA driver information before installing a CUDA wheel")
    d.add_argument("--output", default=".runtime/desktop-doctor.json")
    d.add_argument("--cuda-device", type=int, default=0)
    r = subs.add_parser("run", help="Run the serial desktop development pipeline")
    r.add_argument("--config", default="configs/desktop-smoke.yaml")
    r.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    r.add_argument("--cuda-device", type=int, default=0)
    r.add_argument("--max-steps", type=int, default=12)
    r.add_argument("--run-id")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--resume", action="store_true")
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        cuda_device = getattr(args, "cuda_device", 0)
        if cuda_device < 0:
            raise ValueError("CUDA device index must be nonnegative")
        prepare_environment(cuda_device)
        commands = {"interpreter": interpreter, "verify-venv": verify_venv,
                    "environment": environment_report, "gpu-check": gpu_check, "doctor": doctor, "run": run}
        return commands[args.command](args)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Desktop setup/run error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
