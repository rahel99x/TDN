#!/usr/bin/env python3
"""Bounded premix DAG: three CPU stages followed by one 30-minute A100 stage.

Planning, submission and inspection import only the standard library. Scientific
work belongs to the allocated project venv; no scheduler is called by ``plan``.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tarfile
import time

_spec = importlib.util.spec_from_file_location(
    "premix_research_helpers", Path(__file__).with_name("research_workflow.py"))
rw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rw)
cw = rw.cw
ROOT = cw.ROOT
STAGES = ("accuracy", "scaling", "prepare", "neural")
CPU_STAGES = STAGES[:3]
RESOURCE = {"cpus": 4, "mem_gib": 16, "walltime": "00:30:00"}
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
JOB = re.compile(r"[1-9][0-9]*\Z")
SHA = re.compile(r"[a-f0-9]{64}\Z")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def identifier(value):
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError("Use a simple premix run identifier")
    return value


def workflow_path(value):
    if value == "latest":
        value = cw.read_json(ROOT / "runs" / ".premix-latest.json")["run_id"]
    return cw.inside(ROOT / "runs" / identifier(value) / "premix-workflow.json")


def declaration(profile):
    return {"version": 1, "benchmark_suite": "premix", "profile": profile,
            "stages": list(STAGES), "dependencies": {"neural": list(CPU_STAGES)},
            "device": {stage: "cuda" if stage == "neural" else "cpu" for stage in STAGES},
            "resources_per_stage": RESOURCE,
            "scope": "bounded development experiment; no automatic expansion or queue cap"}


def validate(workflow):
    if workflow.get("schema_version") != 1 or workflow.get("kind") != "bounded-premix":
        raise ValueError("Not a supported premix workflow")
    run_id = identifier(workflow["run_id"])
    base = cw.inside(workflow["run_dir"])
    if base != cw.inside(ROOT / "runs" / run_id) or cw.inside(workflow["root"]) != ROOT.resolve():
        raise ValueError("Premix workflow root/layout changed")
    if workflow.get("profile") not in ("smoke", "full"):
        raise ValueError("Unknown premix profile")
    if cw.inside(workflow["protocol_path"]) != base / "protocol.json":
        raise ValueError("Premix requires the immutable run-local protocol")
    if workflow.get("protocol_sha256") != canonical_hash(declaration(workflow["profile"])):
        raise ValueError("Premix protocol declaration changed")
    if workflow.get("account") != cw.ACCOUNT or workflow.get("user") != cw.USER:
        raise ValueError("Premix user/account differs from CARC policy")
    if set(workflow.get("resources", {})) != set(STAGES):
        raise ValueError("Premix requires exactly four bounded stages")
    for stage in STAGES:
        resource = workflow["resources"][stage]
        if any(resource.get(key) != value for key, value in RESOURCE.items()):
            raise ValueError("Each premix allocation is capped at 4 CPUs, 16 GiB and 30 minutes")
        if not isinstance(resource.get("partition"), str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", resource["partition"]):
            raise ValueError("Invalid partition")
        if stage == "neural" and resource["partition"] != "gpu":
            raise ValueError("Neural training requires the GPU partition")
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+\+cu[0-9]+", workflow.get("torch_version", "")):
        raise ValueError("Premix CARC requires an exact CUDA Torch version")
    for key in ("source_sha256", "source_tree_sha256"):
        if not SHA.fullmatch(workflow.get(key, "")):
            raise ValueError("Missing premix source fingerprint")
    return workflow


def load(path, *, verify=True):
    path = cw.inside(path)
    workflow = validate(cw.read_json(path))
    if path != Path(workflow["run_dir"]) / "premix-workflow.json":
        raise ValueError("Premix manifest is outside its run directory")
    if verify:
        if cw.read_json(workflow["protocol_path"]) != declaration(workflow["profile"]):
            raise ValueError("Frozen premix protocol changed; use a fresh run")
        if cw.source_hash() != workflow["source_sha256"]:
            raise ValueError("Execution source changed; use a fresh premix run")
    return workflow


def prepare(args):
    run_id = identifier(args.run_id or "carc-premix-" +
                        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    base = cw.inside(ROOT / "runs" / run_id)
    if base.exists():
        raise ValueError("Run directory exists; preserve it and choose a fresh --run-id")
    profile = "smoke" if args.smoke else "full"
    return validate({"schema_version": 1, "kind": "bounded-premix", "run_id": run_id,
        "root": str(ROOT.resolve()), "run_dir": str(base), "created_at": cw.now(),
        "profile": profile, "account": cw.ACCOUNT, "user": cw.USER,
        "protocol_path": str(base / "protocol.json"),
        "protocol_sha256": canonical_hash(declaration(profile)),
        "source_sha256": cw.source_hash(), "source_tree_sha256": cw.source_hash(metadata=True),
        "torch_version": os.environ.get("TORCH_VERSION", "2.10.0+cu126"),
        "resources": {stage: {**RESOURCE, "partition": "gpu" if stage == "neural" else
                              os.environ.get("TDN_CPU_PARTITION", "main")} for stage in STAGES}})


def scheduler_args(workflow, stage, dependency=None):
    if stage not in STAGES:
        raise ValueError("Unknown premix stage")
    resource = workflow["resources"][stage]
    base = Path(workflow["run_dir"])
    args = ["sbatch", "--parsable", f"--account={cw.ACCOUNT}",
        f"--partition={resource['partition']}", "--nodes=1", "--ntasks=1",
        f"--job-name=tdn-premix-{stage}", f"--comment=tdn-premix:{workflow['run_id']}:{stage}",
        f"--cpus-per-task={resource['cpus']}", f"--mem={resource['mem_gib']}G",
        f"--time={resource['walltime']}", "--signal=USR1@120", "--export=ALL", "--open-mode=append",
        f"--output={base / 'logs' / (stage + '-%j.out')}",
        f"--error={base / 'logs' / (stage + '-%j.err')}"]
    if stage == "neural":
        args += ["--gpus-per-task=a100:1", "--constraint=a100-40gb", "--kill-on-invalid-dep=yes"]
    if dependency is not None:
        if stage != "neural" or not re.fullmatch(r"afterok:[1-9][0-9]*:[1-9][0-9]*:[1-9][0-9]*", dependency):
            raise ValueError("Neural dependency must name all three actual CPU job IDs")
        if len(set(dependency.split(":")[1:])) != 3:
            raise ValueError("CPU prerequisite job IDs must be distinct")
        args += [f"--dependency={dependency}"]
    return args + [str(ROOT / "scripts" / "premix_worker.sh")]


def start(args):
    workflow = prepare(args)
    print(f"Premix workflow: {workflow['run_id']} ({workflow['profile']})\nRun: {workflow['run_dir']}")
    print("Three CPU jobs run independently; one A100 job requires all three to succeed.")
    print("Each allocation: 4 CPUs, 16 GiB, 30 minutes. One A100 maximum; no queue cap or automatic expansion.")
    for stage in STAGES:
        print(shlex.join(scheduler_args(workflow, stage)))
    print("Neural dependency: afterok:<accuracy job>:<scaling job>:<prepare job>")
    if args.command == "plan":
        print("PLAN ONLY: no writes, scheduler calls or numerical work.")
        return workflow
    cw.actual_policy()
    # Check all resource types before submitting any part of the DAG.
    policies = {}
    for stage in ("accuracy", "neural"):
        phase = "gpu" if stage == "neural" else "cpu"
        policies[phase] = cw.live_check({"resources": {phase: workflow["resources"][stage]}}, phase)
    with cw.controller_lock():
        base = cw.inside(workflow["run_dir"])
        base.mkdir(parents=True, exist_ok=False)
        (base / "logs").mkdir()
        cw.atomic_json(workflow["protocol_path"], declaration(workflow["profile"]))
        path = base / "premix-workflow.json"
        cw.atomic_json(path, workflow)
        path.chmod(0o444)
        Path(workflow["protocol_path"]).chmod(0o444)
        cw.atomic_json(base / "state" / "slurm-policy.json", policies)
        cw.atomic_json(base / "jobs.json", [])
        cw.atomic_json(ROOT / "runs" / ".premix-latest.json", {"run_id": workflow["run_id"]})
        jobs = []
        for stage in STAGES:
            dependency = "afterok:" + ":".join(row["job_id"] for row in jobs) if stage == "neural" else None
            try:
                load(path)
                env = os.environ.copy()
                env.update(CARC_ACCOUNT=cw.ACCOUNT, TDN_REPO_ROOT=str(ROOT),
                           TDN_PREMIX_WORKFLOW=str(path), TDN_PREMIX_STAGE=stage,
                           TORCH_VERSION=workflow["torch_version"])
                result = cw.command(scheduler_args(workflow, stage, dependency), env=env)
                match = re.fullmatch(r"([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?", result.stdout.strip())
                if not match:
                    raise ValueError("sbatch returned an ambiguous job ID; inspect Slurm before retrying")
                job_id = match.group(1)
                if any(row["job_id"] == job_id for row in jobs):
                    raise ValueError("sbatch returned a duplicate job ID; inspect Slurm before retrying")
                jobs.append({"stage": stage, "job_id": job_id, "dependency": dependency,
                             "submitted_at": cw.now()})
                cw.atomic_json(base / "jobs.json", jobs)
                print(f"TDN_PREMIX_{stage.upper()}_JOB_ID={job_id}", flush=True)
            except BaseException as exc:
                cw.atomic_json(base / "state" / "submission.json", {"status": "FAILED", "stage": stage,
                    "error": str(exc), "submitted_jobs": jobs, "updated_at": cw.now(),
                    "note": "Existing submissions are preserved; no automatic retry or cancellation."})
                raise
        cw.atomic_json(base / "state" / "submission.json", {"status": "COMPLETED", "updated_at": cw.now()})
    print(f"Monitor: bash scripts/carc_premix.sh status {workflow['run_id']}")
    return workflow


def state_path(workflow, stage):
    return cw.inside(Path(workflow["run_dir"]) / "state" / f"{stage}.json")


def verify_stage(workflow, stage, *, execution_mode="carc"):
    """Hash every sealed artifact; Slurm success alone is not a prerequisite."""
    base = cw.inside(Path(workflow["run_dir"]) / stage)
    manifest_path, marker = base / "manifest.json", base / "COMPLETED"
    if manifest_path.is_symlink() or marker.is_symlink() or not manifest_path.is_file() or not marker.is_file():
        raise ValueError(f"{stage}: completed artifact seal is missing or unsafe")
    if marker.read_text().strip() != cw.digest(manifest_path):
        raise ValueError(f"{stage}: completion marker differs from manifest")
    manifest = cw.read_json(manifest_path)
    expected = {"version": 1, "benchmark_suite": "premix", "stage": stage,
                "profile": workflow["profile"], "source_tree_sha256": workflow["source_tree_sha256"]}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{stage}: sealed artifact provenance differs")
    required = {"summary.json", "protocol.json", "execution.json"}
    if stage == "prepare":
        required.update(("dataset.pt", "dataset_manifest.json"))
    files = manifest.get("files")
    if not isinstance(files, dict) or not required <= files.keys() or len(files) > 4096:
        raise ValueError(f"{stage}: incomplete or oversized artifact manifest")
    for relative, expected_hash in files.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts or not SHA.fullmatch(str(expected_hash)):
            raise ValueError(f"{stage}: unsafe artifact entry")
        path = base / relative_path
        if path.is_symlink() or not cw.inside(path).is_relative_to(base) or not path.is_file():
            raise ValueError(f"{stage}: missing or unsafe artifact {relative}")
        if cw.digest(path) != expected_hash:
            raise ValueError(f"{stage}: artifact changed: {relative}")
    protocol = cw.read_json(base / "protocol.json")
    if manifest.get("protocol_sha256") != canonical_hash(protocol):
        raise ValueError(f"{stage}: protocol fingerprint differs")
    summary = cw.read_json(base / "summary.json")
    execution = cw.read_json(base / "execution.json")
    if summary.get("status") != "COMPLETED":
        raise ValueError(f"{stage}: scientific engine did not complete")
    for key, value in {"stage": stage, "profile": workflow["profile"],
                       "device": "cuda" if stage == "neural" else "cpu", "execution_mode": execution_mode}.items():
        if execution.get(key) != value:
            raise ValueError(f"{stage}: execution {key} differs")
    return manifest


def verify_prerequisites(workflow, software, *, execution_mode="carc"):
    hashes = {}
    for stage in CPU_STAGES:
        record = cw.read_json(state_path(workflow, stage))
        expected = {"status": "COMPLETED", "exit_code": 0,
                    "source_sha256": workflow["source_sha256"],
                    "protocol_sha256": workflow["protocol_sha256"]}
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError(f"{stage}: CPU worker prerequisite did not complete")
        software_path = Path(workflow["run_dir"]) / "state" / f"{stage}-software.json"
        if (record.get("software_sha256") != cw.digest(software_path) or
                cw.read_json(software_path) != software):
            raise ValueError(f"{stage}: prerequisite software differs")
        verify_stage(workflow, stage, execution_mode=execution_mode)
        hashes[stage] = cw.digest(Path(workflow["run_dir"]) / stage / "manifest.json")
    return hashes


def verify_allocation(workflow, stage):
    cw.actual_policy(worker=True)
    if stage not in STAGES:
        raise ValueError("Unknown premix stage")
    job = os.environ["SLURM_JOB_ID"]
    result = cw.command(["scontrol", "show", "job", job, "-o"])
    fields = dict(re.findall(r"(\w+)=([^\s]+)", result.stdout))
    expected = {"JobId": job, "JobName": f"tdn-premix-{stage}", "Account": cw.ACCOUNT,
                "Comment": f"tdn-premix:{workflow['run_id']}:{stage}", "JobState": "RUNNING",
                "Partition": workflow["resources"][stage]["partition"]}
    if any(fields.get(key) != value for key, value in expected.items()) or fields.get("UserId", "").split("(")[0] != cw.USER:
        raise ValueError("Allocation is not this premix workflow's running stage")


def worker_commands(workflow, stage):
    python = str(cw.inside(ROOT / ".venv" / "bin") / "python")
    base = Path(workflow["run_dir"])
    commands = []
    if stage in ("prepare", "neural"):
        tests = ([ROOT / "tests" / "test_premix_gpu.py"] if stage == "neural" else
                 sorted((ROOT / "tests").glob("test_premix_*.py")))
        if not tests or any(not path.is_file() for path in tests):
            raise ValueError("Premix correctness tests are missing")
        if stage == "neural":
            commands.append(("preflight", [python, str(ROOT / "scripts" / "gpu_preflight.py"),
                                           "--output", str(base / "gpu-preflight.json")]))
        commands.append(("gpu-tests" if stage == "neural" else "tests",
            [python, "-m", "pytest", "-q", "-m", "gpu" if stage == "neural" else "not gpu", *map(str, tests),
             "--basetemp", str(base / f"{stage}-pytest-work"), "-o", f"cache_dir={base / (stage + '-pytest-cache')}",
             "--junitxml", str(base / f"{stage}-tests.xml")]))
    args = [python, str(ROOT / "scripts" / "premix.py"), "--stage", stage,
            "--profile", workflow["profile"], "--run-dir", str(base / stage),
            "--device", "cuda" if stage == "neural" else "cpu"]
    if stage == "neural":
        args += ["--dataset-dir", str(base / "prepare")]
    commands.append(("experiment", args))
    return commands


def begin_report(workflow, stage):
    base, resource = Path(workflow["run_dir"]), workflow["resources"][stage]
    job = os.environ["SLURM_JOB_ID"]
    # The workflow root has its own protocol.json. Bind the report to this
    # stage's science directory, with tower/ alongside it, never inside it.
    return cw.reporting_api().begin_report(base / stage, name=f"TDN/premix/{stage}", script="scripts/premix_worker.sh",
        parameters={"source_sha256": workflow["source_sha256"], "protocol_sha256": workflow["protocol_sha256"],
                    "stage": stage, "profile": workflow["profile"], "execution_mode": "carc"},
        resources={"account": cw.ACCOUNT, "partition": resource["partition"], "nodes": 1,
                   "cpus": 4, "gpus": int(stage == "neural"), "mem_bytes": 16 * 1024**3,
                   "time_seconds": 1800, **({"gpu_type": "a100"} if stage == "neural" else {})},
        job_id=job, report_parent=base / "tower",
        logs=[{"id": f"scheduler.{kind}", "path": str(base / "logs" / f"{stage}-{job}.{suffix}"),
               "label": f"Slurm {kind}", "group": "Scheduler"} for kind, suffix in (("stdout", "out"), ("stderr", "err"))],
        metadata={"workflow_id": workflow["run_id"], "phase": stage,
                  "execution_scope": "one bounded premix allocation"})


def validate_gpu_junit(path):
    from xml.etree import ElementTree
    tree = ElementTree.parse(path).getroot()
    suites = [tree] if tree.tag == "testsuite" else tree.findall(".//testsuite")
    if (sum(int(suite.get("tests", 0)) for suite in suites) <= 0 or
            any(int(suite.get(key, 0)) for suite in suites for key in ("skipped", "failures", "errors"))):
        raise ValueError("Actual premix GPU tests must execute and pass with no skips")


def finish_report(workflow, stage, report, *, state, runtime_seconds, exit_code, software=None, error=None):
    # Stages can finish concurrently. Export only this stage's evidence instead
    # of scanning another stage while it is writing its canonical artifacts.
    from tdn.tower_analytics import publish_outputs
    base = Path(workflow["run_dir"])
    if cw.inside(report).parent != base / "tower":
        raise ValueError("Tower report is not this workflow's sidecar")
    if cw.read_json(report / "run.json").get("job_id") != os.environ["SLURM_JOB_ID"]:
        raise ValueError("Tower report belongs to a different scheduler job")
    sources = [base / stage] if (base / stage).is_dir() else []
    results = publish_outputs(report, sources)
    cw.reporting_api().finish_report(report, state=state, runtime_seconds=runtime_seconds,
        exit_code=exit_code, results=results,
        metadata={"workflow_id": workflow["run_id"], "phase": stage,
                  "execution_scope": "allocated worker elapsed time; excludes pending time",
                  **({"verified_software_sha256": canonical_hash(software)} if software else {}),
                  **({"error": str(error)} if error is not None else {})})


def worker(workflow, stage, *, backend=None):
    # Policy adapters share interruption, locking, reporting and sealing. The
    # default remains CARC; desktop Slurm supplies its own explicit policy.
    if backend is None:
        from types import SimpleNamespace
        backend = SimpleNamespace(**{name: globals()[name] for name in (
            "verify_allocation", "load", "begin_report", "worker_commands",
            "verify_stage", "verify_prerequisites")})
    label = getattr(backend, "workflow_label", "premix")
    stage_variable = getattr(backend, "stage_variable", "TDN_PREMIX_STAGE")
    manifest_name = getattr(backend, "workflow_filename", "premix-workflow.json")
    prerequisite_required = stage in getattr(backend, "prerequisite_stages", ("neural",))
    report_finalizer = getattr(backend, "finish_report", finish_report)
    backend.verify_allocation(workflow, stage)
    base = Path(workflow["run_dir"])
    if state_path(workflow, stage).exists() or (base / stage).exists():
        raise ValueError(f"{label.capitalize()} stage already started; preserve it and submit a fresh workflow")
    record = {"status": "RUNNING", "exit_code": 0, "stage": "startup", "updated_at": cw.now(),
              "source_sha256": workflow["source_sha256"], "protocol_sha256": workflow["protocol_sha256"],
              "slurm_job_id": os.environ["SLURM_JOB_ID"], "slurm_step_id": os.environ["SLURM_STEP_ID"]}
    cw.atomic_json(state_path(workflow, stage), record)
    started = time.monotonic()
    stopped_at = None
    child = None
    report = None
    software = None
    lock = None
    previous_report = os.environ.pop("TDN_TOWER_DIR", None)
    code = 2

    def stop(signum, frame):
        nonlocal stopped_at
        if stopped_at is None:
            stopped_at = time.monotonic()
            if child is not None and child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGUSR1)
                except ProcessLookupError:
                    pass

    handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT)}
    try:
        report = backend.begin_report(workflow, stage)
        os.environ["TDN_TOWER_DIR"] = str(report)
        print(f"{stage_variable}={stage}\nTDN_TOWER_DIR={report}\nTDN_TOWER_METRICS={report / 'metrics.jsonl'}", flush=True)
        backend.load(base / manifest_name)
        lock = rw.acquire_venv_lock()
        software = rw.software_report(workflow)
        if hasattr(backend, "verify_software"):
            backend.verify_software(workflow, software)
        software_path = base / "state" / f"{stage}-software.json"
        cw.atomic_json(software_path, software)
        record["software_sha256"] = cw.digest(software_path)
        if prerequisite_required:
            record["prerequisite_manifest_sha256"] = backend.verify_prerequisites(workflow, software)
        for name, command in backend.worker_commands(workflow, stage):
            if stopped_at is not None:
                raise InterruptedError("Stop requested; no successor started")
            backend.load(base / manifest_name)
            if rw.software_report(workflow) != software:
                raise ValueError("Software changed during premix execution")
            record.update(stage=name, updated_at=cw.now())
            cw.atomic_json(state_path(workflow, stage), record)
            env = os.environ.copy()
            if hasattr(backend, "workflow_environment"):
                env.update(backend.workflow_environment(workflow, stage))
            else:
                env.update(TDN_PREMIX_PROTOCOL_SHA256=workflow["protocol_sha256"],
                           TDN_PREMIX_WORKFLOW=str(base / manifest_name))
            if name in ("tests", "gpu-tests"):
                # Tests own their fixtures and must not inherit a production
                # reporting stream or workflow binding. Real allocation/device
                # metadata remains available to allocated CUDA parity tests.
                for key in ("TDN_TOWER_DIR", "TDN_PREMIX_WORKFLOW", "TDN_PREMIX_STAGE", "TDN_PREMIX_PROTOCOL_SHA256"):
                    env.pop(key, None)
                if hasattr(backend, "prepare_test_environment"):
                    backend.prepare_test_environment(env, name)
            if name == "gpu-tests":
                env["TDN_REQUIRE_GPU_TESTS"] = "1"
            print(f"TDN {label} {stage}: starting {name}", flush=True)
            child = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True)
            if stopped_at is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGUSR1)
            while True:
                try:
                    code = child.wait(timeout=1)
                    break
                except subprocess.TimeoutExpired:
                    if stopped_at is not None and time.monotonic() - stopped_at >= 30:
                        try:
                            os.killpg(child.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
            child = None
            if stopped_at is not None or code == 75:
                raise InterruptedError(f"{label.capitalize()} interrupted; partial evidence preserved")
            if code:
                raise rw.WorkerFailure(f"{label.capitalize()} {stage}/{name} failed with exit code {code}",
                                       128 - code if code < 0 else code)
            if name == "gpu-tests":
                junit_path = backend.gpu_junit_path(workflow) if hasattr(backend, "gpu_junit_path") else base / "neural-tests.xml"
                validate_gpu_junit(junit_path)
        backend.load(base / manifest_name)
        if rw.software_report(workflow) != software:
            raise ValueError("Software changed during premix execution")
        if prerequisite_required and backend.verify_prerequisites(workflow, software) != record["prerequisite_manifest_sha256"]:
            raise ValueError("CPU prerequisite seal changed during neural execution")
        backend.verify_stage(workflow, stage)
        report_finalizer(workflow, stage, report, state="COMPLETED",
            runtime_seconds=time.monotonic() - started, exit_code=0, software=software)
        record.update(status="COMPLETED", exit_code=0, updated_at=cw.now())
        cw.atomic_json(state_path(workflow, stage), record)
        return 0
    except BaseException as exc:
        interrupted = stopped_at is not None or isinstance(exc, (InterruptedError, KeyboardInterrupt)) or code == 75
        failure_code = 75 if interrupted else exc.exit_code if isinstance(exc, rw.WorkerFailure) else 2
        record.update(status="INTERRUPTED" if interrupted else "FAILED", exit_code=failure_code,
                      error=str(exc), updated_at=cw.now())
        cw.atomic_json(state_path(workflow, stage), record)
        if report is not None:
            try:
                report_finalizer(workflow, stage, report, state=record["status"],
                    runtime_seconds=time.monotonic() - started, exit_code=failure_code,
                    error=exc, software=software)
            except Exception as report_error:
                print(f"{label.capitalize()} reporting failed: {report_error}; original failure preserved", file=sys.stderr)
        raise rw.WorkerFailure(str(exc), failure_code) from exc
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        if lock is not None:
            lock.close()
        if previous_report is None:
            os.environ.pop("TDN_TOWER_DIR", None)
        else:
            os.environ["TDN_TOWER_DIR"] = previous_report


def paths(workflow):
    for row in cw.read_json(Path(workflow["run_dir"]) / "jobs.json"):
        if row.get("stage") not in STAGES or not JOB.fullmatch(row.get("job_id", "")):
            raise ValueError("Invalid stored premix scheduler job")
        for report in cw.tower_reports_for_job(workflow, row["job_id"]):
            print(f"{row['stage']} job {row['job_id']}\nTDN_TOWER_DIR={report}\nTDN_TOWER_METRICS={report / 'metrics.jsonl'}")


def status(workflow):
    base = Path(workflow["run_dir"])
    print(f"Premix workflow {workflow['run_id']} ({workflow['profile']})\nRun: {base}")
    jobs = {row["stage"]: row for row in cw.read_json(base / "jobs.json")}
    for stage in STAGES:
        if stage in jobs:
            job = jobs[stage]["job_id"]
            if not JOB.fullmatch(job):
                raise ValueError("Invalid stored job ID")
            try:
                scheduler = cw.scheduler_state(job)
            except (FileNotFoundError, ValueError):
                scheduler = "UNKNOWN (scheduler unavailable)"
            print(f"{stage} job {job}: {scheduler}")
        else:
            print(f"{stage}: NOT_SUBMITTED")
        path = state_path(workflow, stage)
        record = cw.read_json(path) if path.exists() else {}
        print(f"  {record.get('stage', 'worker')}: {record.get('status', 'NOT_STARTED')}")
        if record.get("error"):
            print(f"  error: {record['error']}")
        if record.get("status") == "COMPLETED":
            verify_stage(workflow, stage)
            print("  sealed artifacts: VERIFIED")
    submission_path = base / "state" / "submission.json"
    if submission_path.exists() and (submission := cw.read_json(submission_path)).get("status") == "FAILED":
        print(f"Submission failed at {submission['stage']}: {submission['error']}")
    paths(workflow)


def logs(workflow, lines):
    if not 1 <= lines <= 100000:
        raise ValueError("Choose 1-100000 log lines")
    for path in sorted((Path(workflow["run_dir"]) / "logs").glob("*")):
        path = cw.inside(path)
        if path.is_file():
            print(f"\n{path}")
            with path.open(errors="replace") as stream:
                print("".join(deque(stream, maxlen=lines)), end="")


def collect(workflow):
    base = Path(workflow["run_dir"])
    destination = cw.inside(ROOT / "runs" / (workflow["run_id"] + "-review-" +
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".tar.gz"))
    members = []
    for path in sorted(base.rglob("*")):
        relative = path.relative_to(base)
        if any(part == "__pycache__" or part.endswith(("pytest-work", "pytest-cache")) for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"Review archive refuses symlink: {path}")
        if cw.inside(path).is_file():
            members.append((path, str(Path(base.name) / relative)))
    with destination.open("xb") as stream:
        with tarfile.open(fileobj=stream, mode="w:gz", dereference=False) as archive:
            for path, relative in members:
                archive.add(path, arcname=relative, recursive=False)
    print(f"Review archive: {destination}")
    print("Includes all stages, checkpoints, references, metrics, logs and failures; excludes pytest temporary directories.")
    return destination


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("plan", "run"):
        item = commands.add_parser(name)
        item.add_argument("--run-id")
        item.add_argument("--smoke", action="store_true", help="Tiny integration profile; not a scientific comparison")
    for name in ("status", "logs", "collect", "paths"):
        item = commands.add_parser(name)
        item.add_argument("run", nargs="?", default="latest")
        if name == "logs":
            item.add_argument("--lines", type=int, default=200)
    item = commands.add_parser("worker", help=argparse.SUPPRESS)
    item.add_argument("--workflow", type=Path, required=True)
    item.add_argument("--stage", choices=STAGES, required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command in ("plan", "run"):
            start(args)
        elif args.command == "worker":
            # Start failure reporting before immutable-source verification.
            return worker(load(args.workflow, verify=False), args.stage)
        else:
            workflow = load(workflow_path(args.run), verify=False)
            if args.command == "logs":
                logs(workflow, args.lines)
            else:
                globals()[args.command](workflow)
    except rw.WorkerFailure as exc:
        print(f"TDN premix: {exc}", file=sys.stderr)
        return exc.exit_code
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"TDN premix: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
