#!/usr/bin/env python3
"""Standard-library controller for allocated CARC workflows and CPU screening.

Login commands inspect files and scheduler metadata only. Numerical work,
dependency installation and environment validation belong to allocated workers.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CARC_ROOT = Path("/home1/aadaniel/projects/TDN")
USER = "aadaniel"
ACCOUNT = "anakano_81"
CPU_STAGES = ("setup", "cpu-tests", "audit", "generate")
GPU_STAGES = ("gpu-tests", "calibrate", "train", "evaluate", "benchmark")
LIGHT_STAGES = ("setup", "light-tests", "light-screen")
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}
PROFILES = {
    "light": {"cpu": (2, 8, "00:15:00")},
    "smoke": {"cpu": (4, 16, "01:00:00"), "gpu": (4, 16, "00:45:00")},
    "pilot": {"cpu": (8, 32, "02:00:00"), "gpu": (8, 64, "04:00:00")},
}


def now():
    return datetime.now(timezone.utc).isoformat()


def inside(value):
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Path leaves the project: {value}")
    return path


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def source_hash(*, metadata=False):
    result = hashlib.sha256()
    for name in ("tdn", "reference", "scripts", "configs", "tests"):
        for path in sorted((ROOT / name).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                inside(path)
                # Match existing stage metadata exactly. The controller's larger
                # immutable fingerprint also includes dependency definitions.
                result.update(path.relative_to(ROOT).as_posix().encode())
                result.update(path.read_bytes())
    if not metadata:
        for name in ("requirements.txt", "pyproject.toml"):
            path = ROOT / name
            result.update(name.encode())
            result.update(path.read_bytes())
    return result.hexdigest()


def read_json(path):
    return json.loads(inside(path).read_text())


def atomic_json(path, value):
    path = inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = inside(path.with_name(path.name + ".pending"))
    with pending.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def command(args, *, env=None, required=True):
    result = subprocess.run([str(item) for item in args], cwd=ROOT, env=env,
                            capture_output=True, text=True, check=False)
    if required and result.returncode:
        raise ValueError(f"{args[0]} failed: {result.stderr.strip() or result.stdout.strip()}")
    return result


def controller_lock():
    """Serialize short controller mutations across this shared checkout."""
    import fcntl

    path = inside(ROOT / ".cache" / "carc-controller.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise ValueError("Another CARC controller is submitting or managing jobs; retry after it finishes") from None
    except OSError:
        handle.close()
        raise
    return handle


def actual_policy(*, worker=False):
    if ROOT.resolve() != CARC_ROOT.resolve() or getpass.getuser() != USER:
        raise ValueError(f"Actual CARC actions require {USER} at {CARC_ROOT}")
    if os.environ.get("CARC_ACCOUNT", ACCOUNT) != ACCOUNT:
        raise ValueError(f"Charging account must be {ACCOUNT}")
    if any(os.environ.get(name) for name in ("TDN_LOCAL_TEST_ROOT", "TDN_EXECUTION_MODE", "TDN_PROJECT_ROOT")):
        raise ValueError("Clear local/desktop/root overrides before CARC actions")
    if os.environ.get("CONDA_PREFIX") or os.environ.get("CONDA_SHLVL", "0") != "0":
        raise ValueError("Deactivate Conda before using the project Python venv")
    allocation = any(os.environ.get(name) for name in ("SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_JOB_ACCOUNT"))
    if worker:
        if not all(os.environ.get(name) for name in ("SLURM_JOB_ID", "SLURM_STEP_ID")) or os.environ.get("SLURM_JOB_ACCOUNT") != ACCOUNT:
            raise ValueError("Worker requires a real allocated srun task charged to anakano_81")
        if not re.fullmatch(r"[1-9][0-9]*", os.environ["SLURM_JOB_ID"]) or not re.fullmatch(r"[0-9]+", os.environ["SLURM_STEP_ID"]):
            raise ValueError("Worker requires numeric allocation and srun step identifiers")
    elif allocation:
        raise ValueError("Submit and manage workflows from the login shell, outside an allocation")


def workflow_path(identifier):
    if identifier == "latest":
        identifier = read_json(ROOT / "runs" / ".carc-latest.json")["run_id"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", identifier):
        raise ValueError("Use a simple workflow identifier")
    return inside(ROOT / "runs" / identifier / "workflow.json")


def load_workflow(path, *, verify=True):
    path = inside(path)
    data = read_json(path)
    base = inside(data["run_dir"])
    if path != base / "workflow.json" or inside(data["root"]) != ROOT.resolve():
        raise ValueError("Workflow root or manifest layout differs from this checkout")
    if inside(data["config_path"]) != base / "config.yaml":
        raise ValueError("Workflow config must use the immutable run-local copy")
    validate_light_workflow(data)
    if verify:
        if digest(data["config_path"]) != data["config_sha256"]:
            raise ValueError("Workflow configuration changed; start a fresh workflow")
        if source_hash() != data["source_sha256"]:
            raise ValueError("Execution source changed; start a fresh workflow")
    return data


def jobs_for(workflow):
    path = inside(Path(workflow["run_dir"]) / "jobs.json")
    return read_json(path) if path.exists() else {"cpu": [], "gpu": []}


def phase_stages(workflow, phase):
    if workflow["action"] == "light":
        return LIGHT_STAGES if phase == "cpu" else ()
    if phase == "cpu" and workflow["action"] == "setup":
        return ("setup",)
    return CPU_STAGES if phase == "cpu" else GPU_STAGES


def workflow_phases(workflow):
    return ("cpu",) if workflow["action"] in ("setup", "light") else ("cpu", "gpu")


def validate_light_workflow(workflow):
    """Keep the short CPU screen separate from training and GPU workflows."""
    if workflow["action"] != "light" and workflow["profile"] != "light":
        return
    if workflow["action"] != "light" or workflow["profile"] != "light":
        raise ValueError("Light profile requires the CPU-only light action")
    resources = workflow["resources"]
    expected = {"cpus": 2, "mem_gib": 8, "walltime": "00:15:00"}
    if set(resources) != {"cpu"} or any(resources["cpu"].get(key) != value for key, value in expected.items()):
        raise ValueError("Light workflow requires only 2 CPUs, 8 GiB and 15 minutes")
    if workflow["setup"] != "never" or workflow["pilot_budget"] != 12:
        raise ValueError("Light workflow requires setup=never and its fixed scope")
    if inside(workflow["original_config"]) != inside(ROOT / "configs" / "carc-light.yaml"):
        raise ValueError("Light workflow uses only configs/carc-light.yaml")


def stage_record(workflow, phase, stage):
    if stage not in phase_stages(workflow, phase):
        raise ValueError(f"Stage {stage} does not belong to phase {phase}")
    return inside(Path(workflow["run_dir"]) / "state" / phase / f"{stage}.json")


def software_record(workflow):
    return inside(Path(workflow["run_dir"]) / "state" / "software.json")


def venv_report(workflow):
    # Normal Linux venvs symlink their interpreter to standalone base Python.
    # Validate the venv directory here and sys.prefix inside that interpreter.
    python = inside(ROOT / ".venv" / "bin") / "python"
    if not python.is_file() or not inside(ROOT / ".venv" / "pyvenv.cfg").is_file():
        raise ValueError("Project Python venv is missing")
    # Allocated workers and explicit resume checks inspect this metadata.
    # No Torch import or CUDA operation is needed to parse the config.
    program = r'''
import hashlib, importlib.metadata as m, json, pathlib, platform, re, sys
root=pathlib.Path(sys.argv[1]).resolve()
if pathlib.Path(sys.prefix).resolve()!=root/'.venv' or sys.prefix==sys.base_prefix:
    raise SystemExit('Expected the project Python venv')
if (pathlib.Path(sys.base_prefix)/'conda-meta').exists():
    raise SystemExit('Use standalone Python, not Conda')
if not (3,11)<=sys.version_info[:2]<=(3,13):
    raise SystemExit('Use tested standalone Python 3.11-3.13')
if platform.python_version()!=sys.argv[4]:
    raise SystemExit('Existing venv interpreter differs from the selected Python module')
expected={'torch':sys.argv[3], 'setuptools':'80.9.0', 'wheel':'0.45.1'}
for line in (root/'requirements.txt').read_text().splitlines():
    if line.strip() and not line.lstrip().startswith('#'):
        name, version=line.strip().split('=='); expected[name]=version
versions={name:m.version(name) for name in expected}
if versions!=expected: raise SystemExit('Installed package versions differ from the workflow pins')
from tdn.config import load_config, config_hash
config=load_config(sys.argv[2])
print(json.dumps({'python':platform.python_version(), 'prefix':str(pathlib.Path(sys.prefix).resolve()),
                 'base_prefix':str(pathlib.Path(sys.base_prefix).resolve()), 'packages':versions,
                 'config_hash':config_hash(config)}))
'''
    result = command([python, "-c", program, ROOT, workflow["config_path"], workflow["torch_version"], platform.python_version()])
    return json.loads(result.stdout)


def venv_ready(workflow, phase):
    report = venv_report(workflow)
    baseline = software_record(workflow)
    if baseline.exists():
        if read_json(baseline) != report:
            raise ValueError("Workflow software environment changed; start a fresh workflow")
    elif phase == "cpu":
        atomic_json(baseline, report)
    else:
        raise ValueError("CPU software baseline is missing")
    return report


def completed(workflow, phase, stage, *, recover=True):
    record = stage_record(workflow, phase, stage)
    baseline = software_record(workflow)
    if not baseline.exists():
        return False
    expected = {"source_sha256": workflow["source_sha256"], "config_sha256": workflow["config_sha256"],
                "software_sha256": digest(baseline)}
    recorded = False
    if record.exists():
        data = read_json(record)
        if data.get("status") == "COMPLETED" and all(data.get(k) == v for k, v in expected.items()):
            recorded = True
            if stage in ("setup", "cpu-tests", "gpu-tests", "light-tests"):
                return data.get("exit_code") == 0
            # CLI marker and measured report are retained proof of completion.
    if stage in ("setup", "cpu-tests", "gpu-tests", "light-tests"):
        return False
    run = inside(Path(workflow["run_dir"]) / stage)
    marker, status = run / "COMPLETED", run / "stage.json"
    if not marker.is_file() or not status.is_file():
        return False
    status = read_json(status)
    baseline_data = read_json(baseline)
    valid = status.get("status") == "COMPLETED" and status.get("actually_ran") is True
    valid = valid and status.get("stage") == stage and status.get("config_hash") == baseline_data["config_hash"]
    valid = valid and marker.read_text().strip() == baseline_data["config_hash"]
    valid = valid and status.get("software", {}).get("source_tree_sha256") == workflow["source_tree_sha256"]
    observed_software = status.get("software", {})
    valid = valid and observed_software.get("python") == baseline_data.get("python")
    for name in ("torch", "numpy", "scipy"):
        if name in baseline_data.get("packages", {}):
            valid = valid and observed_software.get(name) == baseline_data["packages"][name]
    valid = valid and status.get("device") == ("cpu" if phase == "cpu" else "cuda")
    if valid and not recorded and recover:
        mark(workflow, phase, stage, "COMPLETED", 0, recovered=True)
    return valid and (recorded or recover)


def mark(workflow, phase, stage, status, exit_code, *, recovered=False):
    if status == "COMPLETED" and exit_code != 0:
        raise ValueError("A completed stage must have exit code zero")
    if status == "PAUSED_NEEDS_RESUME" and exit_code != 75:
        raise ValueError("A paused stage must have exit code 75")
    if stage == "setup" and status == "COMPLETED":
        venv_ready(workflow, phase)
    path = stage_record(workflow, phase, stage)
    baseline = software_record(workflow)
    entry = {"stage": stage, "phase": phase, "status": status, "exit_code": exit_code,
             "updated_at": now(), "source_sha256": workflow["source_sha256"],
             "config_sha256": workflow["config_sha256"],
             "software_sha256": digest(baseline) if baseline.exists() else None,
             "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "recovered_cli_proof": recovered}
    atomic_json(path, entry)
    final = stage == phase_stages(workflow, phase)[-1] and status == "COMPLETED"
    summary = {"phase": phase, "stage": stage, "status": "COMPLETED" if final else "RUNNING" if status == "COMPLETED" else status,
               "exit_code": exit_code, "updated_at": now()}
    atomic_json(Path(workflow["run_dir"]) / "state" / f"{phase}.json", summary)
    if os.environ.get("TDN_TOWER_DIR") and not recovered:
        emit = reporting_api().emit
        stages = phase_stages(workflow, phase)
        position = stages.index(stage) + (status == "COMPLETED")
        emit({"stage_exit_code": exit_code}, phase=f"workflow.{phase}",
             step=position, completed=position,
             total=len(stages), unit="stages")


def reporting_api():
    # Allocated setup may run before the editable package or Torch is installed.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tdn import reporting
    return reporting


def begin_workflow_report(workflow, phase, *, research=False, software=None):
    """Create one sidecar attempt after the caller verifies scheduler ownership."""
    api = reporting_api()
    base = inside(workflow["run_dir"])
    job = os.environ["SLURM_JOB_ID"]
    if not re.fullmatch(r"[1-9][0-9]*", job):
        raise ValueError("Tower allocation reports require the actual numeric job ID")
    resource = workflow["resources"][phase]
    hours, minutes, seconds = map(int, resource["walltime"].split(":"))
    resources = {"account": ACCOUNT, "partition": resource["partition"],
                 "nodes": 1, "cpus": resource["cpus"], "gpus": int(phase == "gpu"),
                 "mem_bytes": resource["mem_gib"] * 1024 ** 3,
                 "time_seconds": hours * 3600 + minutes * 60 + seconds}
    if phase == "gpu":
        resources["gpu_type"] = "a100"
    parameters = {"source_sha256": workflow["source_sha256"],
                  "config_sha256": workflow["config_sha256"], "phase": phase,
                  "execution_mode": "carc"}
    if software is not None:
        parameters["software_sha256"] = hashlib.sha256(json.dumps(
            software, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    elif software_record(workflow).is_file():
        parameters["software_sha256"] = digest(software_record(workflow))
    if research:
        parameters["smoke"] = workflow["smoke"]
        parameters["benchmark_suite"] = workflow.get("benchmark_suite", "architecture")
        if workflow.get("source_manifest_sha256"):
            parameters["input_manifest_sha256"] = workflow["source_manifest_sha256"]
    else:
        parameters.update(profile=workflow["profile"], action=workflow["action"])
        manifest = base / "dataset" / "manifest.json"
        if manifest.is_file():
            parameters["input_manifest_sha256"] = digest(manifest)
    kind = "research" if research else "carc"
    label = (parameters["benchmark_suite"] if research and parameters["benchmark_suite"] in ("neural-benchmarks", "neural-replication", "mechanism-audit", "interaction-screen", "work-precision")
             else parameters.get("profile", "smoke" if parameters.get("smoke") else "default"))
    return api.begin_report(base, name=f"TDN/{kind}/{label}/{phase}",
                            script=f"scripts/{kind}_{phase}.sbatch", parameters=parameters,
                            resources=resources, job_id=job, report_parent=base / "tower",
                            logs=[{"id": "scheduler.stdout", "path": str(base / "logs" / f"{phase}-{job}.out"),
                                   "label": "Slurm stdout", "group": "Scheduler"},
                                  {"id": "scheduler.stderr", "path": str(base / "logs" / f"{phase}-{job}.err"),
                                   "label": "Slurm stderr", "group": "Scheduler"}],
                            metadata={"workflow_id": workflow["run_id"], "phase": phase,
                                      "resource_source": "verified single-node single-task workflow request",
                                      "execution_scope": "one actual allocated workflow phase"})


def finish_workflow_report(workflow, phase, report_dir, *, state, runtime_seconds, exit_code,
                           research=False, error=None, software=None):
    api = reporting_api()
    from tdn.tower_analytics import publish_outputs
    base = inside(workflow["run_dir"])
    report_dir = inside(report_dir)
    if report_dir.parent != base / "tower":
        raise ValueError("Tower report is not a sidecar of this workflow")
    inventory = read_json(report_dir / "run.json")
    if inventory.get("job_id") != os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Tower report belongs to a different scheduler job")
    # The bounded adapter excludes Tower sidecars and pytest/cache directories.
    results = publish_outputs(report_dir, [base])
    verified_software = ({"verified_software_sha256": hashlib.sha256(json.dumps(
        software, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()}
        if software is not None else {})
    api.finish_report(report_dir, state=state, runtime_seconds=runtime_seconds,
                      exit_code=exit_code, results=results,
                      observed_parameters=({"software_sha256": verified_software["verified_software_sha256"]}
                                           if verified_software else None),
                      metadata={"workflow_id": workflow["run_id"], "phase": phase,
                                "execution_scope": "allocated worker elapsed time; excludes pending time",
                                **verified_software,
                                **({"error": str(error)} if error is not None else {})})


def worker_report_begin(workflow, phase):
    actual_policy(worker=True)
    if phase not in workflow_phases(workflow):
        raise ValueError("Unknown workflow phase")
    allocation = ownership(workflow, phase, os.environ["SLURM_JOB_ID"])
    if allocation is None or allocation.get("JobState") != "RUNNING":
        raise ValueError("Tower reporting requires this workflow's running allocation")
    path = Path(workflow["run_dir"]) / "state" / f"tower-{phase}-{os.environ['SLURM_JOB_ID']}.json"
    if path.exists():
        raise ValueError("This allocation already has a Tower attempt; preserve its evidence")
    started = time.monotonic()
    report = begin_workflow_report(workflow, phase)
    atomic_json(path, {"report_dir": str(report), "started_monotonic": started})
    return report


def worker_report_finish(workflow, phase, exit_code, *, interrupted=False):
    path = Path(workflow["run_dir"]) / "state" / f"tower-{phase}-{os.environ['SLURM_JOB_ID']}.json"
    evidence = read_json(path)
    report = inside(os.environ["TDN_TOWER_DIR"])
    if report != inside(evidence["report_dir"]):
        raise ValueError("Tower report does not match this allocated worker attempt")
    state = "INTERRUPTED" if interrupted or exit_code == 75 else "FAILED" if exit_code else "COMPLETED"
    finish_workflow_report(workflow, phase, report, state=state,
                           runtime_seconds=max(0., time.monotonic() - evidence["started_monotonic"]),
                           exit_code=exit_code)


def tower_reports_for_job(workflow, job):
    """Bounded read-only discovery of exact job-bound coordinator sidecars."""
    from itertools import islice
    parent = Path(workflow["run_dir"]) / "tower"
    if parent.is_symlink() or not parent.is_dir():
        return []
    candidates = list(islice(parent.iterdir(), 259))
    if len(candidates) > 258:  # 256 attempts plus the lock and attempt index.
        return []
    result = []
    for path in sorted(candidates):
        if not path.name.startswith("tdn-") or path.is_symlink() or not path.is_dir():
            continue
        inventory = path / "run.json"
        try:
            if inventory.is_symlink() or not inventory.is_file() or inventory.stat().st_size > 1024 * 1024:
                continue
            entry = read_json(inventory)
            if (entry.get("schema") == "tower.run/v1" and entry.get("job_id") == job
                    and entry.get("run_id") == path.name):
                result.append(inside(path))
        except (ValueError, OSError, TypeError):
            continue
    return result


def worker_verify(workflow, phase):
    actual_policy(worker=True)
    validate_light_workflow(workflow)
    if phase not in workflow_phases(workflow):
        raise ValueError("This CPU-only workflow has no GPU phase")
    # The first CPU task can begin before sbatch's returned ID is published.
    # Its live scheduler comment, identity and running state bind it to this
    # manifest without depending on that publication race.
    allocation = ownership(workflow, phase, os.environ.get("SLURM_JOB_ID", ""))
    if allocation is None or allocation.get("JobState") != "RUNNING":
        raise ValueError("Worker allocation is not this workflow's running phase")
    if software_record(workflow).exists():
        venv_ready(workflow, phase)
    if phase == "gpu":
        if workflow["action"] == "setup":
            raise ValueError("Setup-only workflows have no GPU phase")
        if not all(completed(workflow, "cpu", stage) for stage in CPU_STAGES):
            raise ValueError("GPU work requires the completed matching CPU prerequisites")
        if not software_record(workflow).exists():
            raise ValueError("CPU software baseline is missing")


def live_check(workflow, phase):
    resource = workflow["resources"][phase]
    args = [sys.executable, ROOT / "scripts" / "carc_check.py", "--partition", resource["partition"],
            "--walltime", resource["walltime"], "--cpus", resource["cpus"], "--mem-gib", resource["mem_gib"]]
    if phase == "gpu":
        args += ["--gpu"]
    report = command(args)
    return json.loads(report.stdout)


def scheduler_args(workflow, phase, dependency=None):
    if phase not in workflow_phases(workflow):
        raise ValueError("This CPU-only workflow has no GPU phase")
    r = workflow["resources"][phase]
    base = inside(workflow["run_dir"])
    args = ["sbatch", "--parsable", f"--account={ACCOUNT}", f"--partition={r['partition']}",
            "--nodes=1", "--ntasks=1", f"--job-name=tdn-carc-{phase}",
            f"--comment=tdn:{workflow['run_id']}:{phase}", f"--cpus-per-task={r['cpus']}",
            f"--mem={r['mem_gib']}G", f"--time={r['walltime']}", "--signal=USR1@180", "--export=ALL",
            f"--output={base / 'logs' / (phase + '-%j.out')}", f"--error={base / 'logs' / (phase + '-%j.err')}"]
    if phase == "gpu":
        args += ["--gpus-per-task=a100:1", "--constraint=a100-40gb", "--kill-on-invalid-dep=yes"]
    if dependency:
        if not re.fullmatch(r"afterok:[1-9][0-9]*", dependency):
            raise ValueError("Dependency must contain an actual numeric scheduler job ID")
        args += [f"--dependency={dependency}"]
    args += [str(ROOT / "scripts" / f"carc_{phase}.sbatch")]
    return args


def submission_environment(workflow, phase, resume="none"):
    env = os.environ.copy()
    env.update({"CARC_ACCOUNT": ACCOUNT, "TDN_REPO_ROOT": str(ROOT), "TDN_WORKFLOW": str(Path(workflow["run_dir"]) / "workflow.json"),
                "TDN_WORKFLOW_ROOT": workflow["run_dir"], "TDN_WORKFLOW_PHASE": phase,
                "TDN_WORKFLOW_ACTION": workflow["action"], "TDN_CONFIG": workflow["config_path"],
                "TDN_DATASET": str(Path(workflow["run_dir"]) / "dataset"), "TDN_SETUP_MODE": workflow["setup"],
                "TDN_PILOT_BUDGET": str(workflow["pilot_budget"]), "TDN_RESUME": resume,
                "TORCH_VERSION": workflow["torch_version"], "TORCH_WHEEL_INDEX": workflow["torch_wheel_index"]})
    return env


def submit_phase(workflow, phase, dependency=None, resume="none"):
    result = command(scheduler_args(workflow, phase, dependency), env=submission_environment(workflow, phase, resume))
    text = result.stdout.strip()
    match = re.fullmatch(r"([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?", text)
    if not match:
        raise ValueError("sbatch returned an ambiguous job ID; inspect the scheduler before retrying")
    job = match.group(1)
    jobs = jobs_for(workflow)
    jobs[phase].append({"job_id": job, "submitted_at": now(), "dependency": dependency, "resume": resume})
    atomic_json(Path(workflow["run_dir"]) / "jobs.json", jobs)
    print(f"TDN_{phase.upper()}_JOB_ID={job}")
    return job


def prepare(args, *, previous=None):
    prior_light = previous is not None and previous["action"] == "light"
    light = args.command == "light" or prior_light or args.profile == "light"
    if light and args.command == "setup":
        raise ValueError("Use the regular setup command separately; light requires an existing venv")
    if light and args.profile not in (None, "light"):
        raise ValueError("Light screening cannot change its fixed CPU-only profile")
    profile = "light" if light else args.profile or (previous["profile"] if previous else "smoke")
    budget = args.pilot_budget if args.pilot_budget is not None else previous["pilot_budget"] if previous else 12 if profile in ("smoke", "light") or args.command == "setup" else None
    if budget is None or not 1 <= budget <= 1000:
        raise ValueError("Pilot profile requires an explicit --pilot-budget of 1-1000 optimizer steps")
    action = "light" if light else "setup" if args.command == "setup" else "start"
    default_config = {"light": "carc-light.yaml", "smoke": "carc-smoke.yaml", "pilot": "pilot.yaml"}[profile]
    config = inside(args.config or (previous["original_config"] if previous else ROOT / "configs" / default_config))
    if not config.is_file():
        raise ValueError(f"Configuration is missing: {config}")
    torch_version = os.environ.get("TORCH_VERSION", "2.10.0+cu126")
    wheel_index = os.environ.get("TORCH_WHEEL_INDEX", "https://download.pytorch.org/whl/cu126")
    release = re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+\+(cu[0-9]+)", torch_version)
    index = re.fullmatch(r"https://download\.pytorch\.org/whl/(cu[0-9]+)", wheel_index)
    if not release or not index or release.group(1) != index.group(1):
        raise ValueError("Choose an exact CUDA Torch local version matching the official CUDA wheel index")
    identifier = args.run_id or "carc-" + (action + "-" if action in ("setup", "light") else "") + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    base = workflow_path(identifier).parent
    if base.exists():
        raise ValueError("Workflow directory exists; preserve it and choose a fresh --run-id")
    resources = {}
    for phase, (cpus, mem, wall) in PROFILES[profile].items():
        resources[phase] = {"partition": os.environ.get("TDN_CPU_PARTITION", "main") if phase == "cpu" else "gpu",
                            "cpus": cpus, "mem_gib": mem, "walltime": wall}
    workflow = {"schema_version": 1, "run_id": identifier, "root": str(ROOT.resolve()), "run_dir": str(base),
            "created_at": now(), "action": action, "profile": profile, "pilot_budget": budget,
            "setup": args.setup or (previous["setup"] if previous else "never" if light else "auto"), "original_config": str(config),
            "config_path": str(base / "config.yaml"), "config_sha256": digest(config),
            "source_sha256": source_hash(), "source_tree_sha256": source_hash(metadata=True),
            "torch_version": torch_version, "torch_wheel_index": wheel_index,
            "resources": resources}
    validate_light_workflow(workflow)
    return workflow


def preview(workflow, *, resume=False):
    scope = "CPU-only bounded screen; no training or GPU" if workflow["action"] == "light" else f"budget={workflow['pilot_budget']}"
    print(f"Workflow: {workflow['run_id']}  profile={workflow['profile']}  {scope}")
    print(f"Config: {workflow['original_config']}\nRun: {workflow['run_dir']}")
    phases = ("gpu",) if resume else workflow_phases(workflow)
    for phase in phases:
        print(f"{phase}: {' -> '.join(phase_stages(workflow, phase))}")
        print(shlex.join(scheduler_args(workflow, phase)))
        if phase == "gpu" and not resume:
            print("Dependency: afterok:<actual CPU job ID>; invalid dependencies are cancelled by Slurm.")
    print("DRY RUN: no directories, installation, compute, live checks or jobs. Add --submit explicitly.")


def start(args, *, previous=None):
    workflow = prepare(args, previous=previous)
    if not args.submit:
        preview(workflow)
        return
    actual_policy()
    # Check every proposed allocation before submitting the first job.
    phases = workflow_phases(workflow)
    reports = {phase: live_check(workflow, phase) for phase in phases}
    base = inside(workflow["run_dir"])
    base.mkdir(parents=True, exist_ok=False)
    inside(base / "logs").mkdir()
    config = inside(workflow["config_path"])
    with config.open("xb") as target:
        target.write(inside(workflow["original_config"]).read_bytes())
        target.flush(); os.fsync(target.fileno())
    if digest(config) != workflow["config_sha256"] or source_hash() != workflow["source_sha256"]:
        raise ValueError("Configuration or source changed during submission; no jobs submitted")
    atomic_json(base / "workflow.json", workflow)
    # Never edit either input after submission; workers certify their hashes.
    config.chmod(0o444)
    (base / "workflow.json").chmod(0o444)
    atomic_json(base / "jobs.json", {"cpu": [], "gpu": []})
    for phase, report in reports.items():
        atomic_json(base / "state" / f"{phase}-slurm-policy.json", report)
    atomic_json(ROOT / "runs" / ".carc-latest.json", {"run_id": workflow["run_id"]})
    print(f"TDN_RUN_ID={workflow['run_id']}")
    cpu = submit_phase(workflow, "cpu")
    if "gpu" in phases:
        submit_phase(workflow, "gpu", f"afterok:{cpu}")
    print(f"Monitor: bash scripts/carc.sh status {workflow['run_id']}")


def scheduler_state(job):
    pending = command(["squeue", "-h", "-j", job, "-o", "%T"], required=False)
    state = pending.stdout.strip().splitlines()
    if state:
        return state[0].strip().upper()
    history = command(["sacct", "-nP", "-j", job, "--format=JobIDRaw,State%30,ExitCode"], required=False)
    for line in history.stdout.splitlines():
        cells = line.split("|")
        if len(cells) >= 3 and cells[0] == job:
            return cells[1].split()[0].rstrip("+").upper()
    return "UNKNOWN"


def ownership(workflow, phase, job):
    result = command(["scontrol", "show", "job", job, "-o"], required=False)
    if result.returncode:
        state = scheduler_state(job)
        if state in TERMINAL:
            return None
        raise ValueError(f"Cannot verify scheduler ownership of job {job}")
    fields = dict(re.findall(r"(\w+)=([^\s]+)", result.stdout))
    expected = {"JobId": job, "JobName": f"tdn-carc-{phase}", "Account": ACCOUNT,
                "Comment": f"tdn:{workflow['run_id']}:{phase}"}
    if any(fields.get(key) != value for key, value in expected.items()) or fields.get("UserId", "").split("(")[0] != USER:
        raise ValueError(f"Job {job} does not have this workflow's exact scheduler ownership")
    return fields


def cancel(workflow, *, submit=False):
    if submit:
        actual_policy()
    for phase, entries in jobs_for(workflow).items():
        for entry in entries:
            job = entry["job_id"]
            if not re.fullmatch(r"[1-9][0-9]*", job):
                raise ValueError("Stored scheduler job ID is invalid")
            if not submit:
                print(f"DRY RUN: verify ownership, then scancel --state=PENDING {job}")
                continue
            fields = ownership(workflow, phase, job)
            if fields is not None and fields.get("JobState") == "PENDING":
                command(["scancel", "--state=PENDING", job])
                print(f"Cancelled owned pending {phase} job {job}")
            else:
                print(f"Preserved {phase} job {job}: completed or active")


def status(workflow):
    print(f"Workflow {workflow['run_id']} ({workflow['profile']})\nRun: {workflow['run_dir']}")
    for phase, entries in jobs_for(workflow).items():
        if phase not in workflow_phases(workflow):
            continue
        if not entries:
            print(f"{phase}: no job submitted")
        current_scheduler = "UNKNOWN"
        for entry in entries:
            job = entry["job_id"]
            current_scheduler = scheduler_state(job)
            print(f"{phase} job {job}: {current_scheduler}")
            for report in tower_reports_for_job(workflow, job):
                print(f"  TDN_TOWER_DIR={report}")
        for stage in phase_stages(workflow, phase):
            path = stage_record(workflow, phase, stage)
            entry = read_json(path) if path.exists() else {}
            recorded_status = entry.get("status", "NOT_STARTED")
            if recorded_status == "RUNNING" and current_scheduler in TERMINAL and current_scheduler != "COMPLETED":
                print(f"  {stage}: INTERRUPTED (scheduler {current_scheduler}; saved record RUNNING)")
            else:
                print(f"  {stage}: {recorded_status}")
            if entry.get("status") in ("FAILED", "PAUSED_NEEDS_RESUME"):
                print(f"  First stopping stage: {stage}, exit {entry.get('exit_code')}")
                break


def logs(workflow, count):
    first_failure = None
    for phase in workflow_phases(workflow):
        for stage in phase_stages(workflow, phase):
            path = stage_record(workflow, phase, stage)
            if path.exists() and read_json(path).get("status") in ("FAILED", "PAUSED_NEEDS_RESUME"):
                first_failure = stage
                break
        if first_failure:
            break
    if first_failure:
        print(f"First stopping stage: {first_failure}")
    for phase, entries in jobs_for(workflow).items():
        for entry in entries:
            for suffix in ("out", "err"):
                path = inside(Path(workflow["run_dir"]) / "logs" / f"{phase}-{entry['job_id']}.{suffix}")
                print(f"\n{path}")
                if path.is_file():
                    print("\n".join(path.read_text(errors="replace").splitlines()[-count:]))
                else:
                    print("Not written yet")


def resume(args, workflow):
    if workflow["action"] == "light":
        raise ValueError("Light screening has no training checkpoint; use restart for a fresh CPU-only run")
    if any(getattr(args, name) is not None for name in ("run_id", "profile", "config", "pilot_budget", "setup")):
        raise ValueError("Resume preserves the original run, profile, config, budget and setup mode")
    if workflow["action"] == "setup":
        raise ValueError("A setup-only workflow cannot resume GPU training")
    if args.submit:
        actual_policy()
    if not all(completed(workflow, "cpu", stage, recover=args.submit) for stage in CPU_STAGES):
        raise ValueError("Resume requires every verified CPU prerequisite")
    if not all(completed(workflow, "gpu", stage, recover=args.submit) for stage in ("gpu-tests", "calibrate")):
        raise ValueError("Resume requires verified GPU tests and calibration")
    latest = inside(Path(workflow["run_dir"]) / "train" / "checkpoints" / "latest.json")
    if not latest.is_file():
        raise ValueError("Resume requires a certified paused training checkpoint")
    certificate = read_json(latest)
    checkpoint = inside(latest.parent / certificate["path"])
    if checkpoint.parent != latest.parent or certificate.get("status") != "PAUSED_NEEDS_RESUME" or digest(checkpoint) != certificate.get("sha256"):
        raise ValueError("Resume requires an intact PAUSED_NEEDS_RESUME checkpoint")
    if not args.submit:
        preview(workflow, resume=True)
        return
    for entries in jobs_for(workflow).values():
        for entry in entries:
            if scheduler_state(entry["job_id"]) not in TERMINAL:
                raise ValueError("Resume blocked: a previous job is active or its terminal state is unknown")
    # Validate software without writing or changing the baseline on login.
    if venv_report(workflow) != read_json(software_record(workflow)):
        raise ValueError("Resume software environment changed")
    report = live_check(workflow, "gpu")
    atomic_json(Path(workflow["run_dir"]) / "state" / "gpu-resume-slurm-policy.json", report)
    submit_phase(workflow, "gpu", resume=str(latest))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("start", "light", "setup", "restart", "resume"):
        item = commands.add_parser(name)
        if name in ("restart", "resume"):
            item.add_argument("identifier", nargs="?", default="latest")
        item.add_argument("--submit", action="store_true")
        item.add_argument("--dry-run", action="store_true")
        item.add_argument("--run-id")
        item.add_argument("--profile", choices=tuple(PROFILES))
        item.add_argument("--config")
        item.add_argument("--pilot-budget", type=int)
        item.add_argument("--setup", choices=("auto", "always", "never"))
    for name in ("status", "logs", "cancel"):
        item = commands.add_parser(name)
        item.add_argument("identifier", nargs="?", default="latest")
        if name == "logs":
            item.add_argument("--lines", type=int, default=60)
        if name == "cancel":
            item.add_argument("--submit", action="store_true")
            item.add_argument("--dry-run", action="store_true")
    for name in ("worker-verify", "worker-mark", "worker-venv-ready", "worker-stage-complete",
                 "worker-report-begin", "worker-report-finish"):
        item = commands.add_parser(name)
        item.add_argument("--workflow", "--workflow-path", required=True)
        item.add_argument("--phase", choices=("cpu", "gpu"), required=True)
        if name in ("worker-mark", "worker-stage-complete"):
            item.add_argument("--stage", required=True)
        if name == "worker-mark":
            item.add_argument("--status", choices=("RUNNING", "COMPLETED", "FAILED", "PAUSED_NEEDS_RESUME"), required=True)
            item.add_argument("--exit-code", type=int, required=True)
        if name == "worker-report-finish":
            item.add_argument("--exit-code", type=int, required=True)
            item.add_argument("--interrupted", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if getattr(args, "dry_run", False):
        args.submit = False
    lock = None
    try:
        if getattr(args, "submit", False):
            actual_policy()
            lock = controller_lock()
        if args.command.startswith("worker-"):
            # Finish reporting even if a source edit caused the worker failure.
            workflow = load_workflow(args.workflow, verify=args.command != "worker-report-finish")
            actual_policy(worker=True)
            if args.command == "worker-verify":
                worker_verify(workflow, args.phase)
            elif args.command == "worker-mark":
                mark(workflow, args.phase, args.stage, args.status, args.exit_code)
            elif args.command == "worker-venv-ready":
                venv_ready(workflow, args.phase)
            elif args.command == "worker-report-begin":
                print(worker_report_begin(workflow, args.phase))
            elif args.command == "worker-report-finish":
                worker_report_finish(workflow, args.phase, args.exit_code, interrupted=args.interrupted)
            else:
                return 0 if completed(workflow, args.phase, args.stage) else 1
        elif args.command in ("start", "light", "setup"):
            start(args)
        else:
            # Status/logs/cancel remain useful even after code or config changes.
            workflow = load_workflow(workflow_path(args.identifier), verify=args.command == "resume")
            if args.command == "status":
                status(workflow)
            elif args.command == "logs":
                if args.lines < 1:
                    raise ValueError("--lines must be positive")
                logs(workflow, args.lines)
            elif args.command == "cancel":
                cancel(workflow, submit=args.submit)
            elif args.command == "restart":
                if args.submit:
                    actual_policy()
                    # Refuse to change a shared venv while an old workflow runs.
                    for entries in jobs_for(workflow).values():
                        for entry in entries:
                            state = scheduler_state(entry["job_id"])
                            if state not in TERMINAL and state != "PENDING":
                                raise ValueError("Restart blocked: old job active or terminal state unknown")
                prepare(args, previous=workflow)
                if args.submit:
                    cancel(workflow, submit=True)
                start(args, previous=workflow)
            else:
                resume(args, workflow)
        return 0
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"TDN CARC: {error}", file=sys.stderr)
        return 2
    finally:
        if lock is not None:
            lock.close()


if __name__ == "__main__":
    sys.exit(main())
