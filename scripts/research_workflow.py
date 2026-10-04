#!/usr/bin/env python3
"""Read-only login planning and bounded allocated research experiments.

This controller imports only the standard library and the existing standard-
library CARC policy helpers. Torch, YAML and numerical work run in the venv in
an actual allocation. A preview never creates directories or calls Slurm.
"""
from __future__ import annotations

import argparse
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

_spec = importlib.util.spec_from_file_location(
    "tdn_research_carc_policy", Path(__file__).with_name("carc_workflow.py"))
cw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cw)
ROOT = cw.ROOT
ACCOUNT = cw.ACCOUNT
ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
JOB_PATTERN = re.compile(r"[1-9][0-9]*\Z")
RESOURCE = {"cpus": 4, "mem_gib": 16, "walltime": "00:30:00"}


def identifier(value):
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value):
        raise ValueError("Use a simple research workflow identifier")
    return value


def workflow_path(value):
    if value == "latest":
        value = cw.read_json(ROOT / "runs" / ".research-latest.json")["run_id"]
    return cw.inside(ROOT / "runs" / identifier(value) / "research-workflow.json")


def validate(workflow):
    if workflow.get("schema_version") != 1 or workflow.get("kind") != "bounded-research":
        raise ValueError("Not a supported research workflow")
    run_id = identifier(workflow["run_id"])
    base = cw.inside(workflow["run_dir"])
    if base != cw.inside(ROOT / "runs" / run_id) or cw.inside(workflow["root"]) != ROOT.resolve():
        raise ValueError("Research workflow root/layout changed")
    if cw.inside(workflow["config_path"]) != base / "config.yaml":
        raise ValueError("Use the immutable run-local research config")
    phase = workflow["phase"]
    if phase not in ("cpu", "gpu") or set(workflow["resources"]) != {phase}:
        raise ValueError("A research workflow has exactly one CPU or GPU phase")
    resource = workflow["resources"][phase]
    if any(resource.get(key) != value for key, value in RESOURCE.items()):
        raise ValueError("Research allocations are capped at 4 CPUs, 16 GiB and 30 minutes")
    if not isinstance(resource.get("partition"), str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", resource["partition"]):
        raise ValueError("Invalid partition")
    if phase == "gpu" and resource["partition"] != "gpu":
        raise ValueError("Research GPU work requires the GPU partition")
    if not isinstance(workflow.get("smoke"), bool):
        raise ValueError("Malformed smoke flag")
    if phase == "cpu" and workflow.get("source_workflow") is not None:
        raise ValueError("CPU research cannot consume a benchmark predecessor")
    if phase == "gpu" and not workflow.get("source_workflow"):
        raise ValueError("GPU benchmark requires a completed CPU workflow")
    return workflow


def load(path, *, verify=True):
    path = cw.inside(path)
    workflow = validate(cw.read_json(path))
    if path != cw.inside(workflow["run_dir"]) / "research-workflow.json":
        raise ValueError("Research manifest is outside its run directory")
    if verify:
        if cw.digest(workflow["config_path"]) != workflow["config_sha256"]:
            raise ValueError("Research configuration changed; start a fresh run")
        if cw.source_hash() != workflow["source_sha256"]:
            raise ValueError("Research execution source changed; start a fresh run")
    return workflow


def state_path(workflow):
    return cw.inside(Path(workflow["run_dir"]) / "state" / "phase.json")


def software_path(workflow):
    return cw.inside(Path(workflow["run_dir"]) / "state" / "software.json")


def verify_experiment(workflow, *, require_headroom=False):
    """Verify the engine's sealed artifacts without importing numerical code."""
    base = cw.inside(Path(workflow["run_dir"]) / "experiment")
    manifest_path = cw.inside(base / "manifest.json")
    marker = cw.inside(base / "COMPLETED")
    if not manifest_path.is_file() or not marker.is_file():
        raise ValueError("Research experiment has no completed artifact manifest")
    manifest = cw.read_json(manifest_path)
    if marker.read_text().strip() != cw.digest(manifest_path):
        raise ValueError("Research completion marker does not match its artifact manifest")
    if manifest.get("source_tree_sha256") != workflow["source_tree_sha256"]:
        raise ValueError("Research artifact source fingerprint differs")
    if manifest.get("config_file_sha256") != workflow["config_sha256"]:
        raise ValueError("Research artifact configuration fingerprint differs")
    files = manifest.get("files", {})
    required = {"protocol.json", "dataset.pt", "summary.json"}
    if not isinstance(files, dict) or not required.issubset(files):
        raise ValueError("Research artifact manifest is incomplete")
    for relative, expected in files.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("Artifact manifest contains an escaping path")
        path = cw.inside(base / relative_path)
        if not path.is_relative_to(base) or not path.is_file() or path.is_symlink():
            raise ValueError(f"Research artifact is missing or unsafe: {relative}")
        if cw.digest(path) != expected:
            raise ValueError(f"Research artifact changed: {relative}")
    protocol_digest = hashlib.sha256(json.dumps(cw.read_json(base / "protocol.json"),
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if manifest.get("protocol_sha256") != protocol_digest:
        raise ValueError("Research protocol fingerprint differs")
    summary = cw.read_json(base / "summary.json")
    if summary.get("status") != "COMPLETED" or summary.get("device") != ("cpu" if workflow["phase"] == "cpu" else "cuda"):
        raise ValueError("Research experiment did not complete")
    if require_headroom and summary.get("headroom", {}).get("passed") is not True:
        raise ValueError("CPU research found no eligible numerical headroom; no A100 job is submitted")
    return manifest


def completed_source(path):
    workflow = load(path)
    if workflow["phase"] != "cpu":
        raise ValueError("Benchmark source must be a CPU research workflow")
    record = cw.read_json(state_path(workflow))
    expected = {"status": "COMPLETED", "exit_code": 0, "stage": "experiment",
                "source_sha256": workflow["source_sha256"],
                "config_sha256": workflow["config_sha256"],
                "software_sha256": cw.digest(software_path(workflow))}
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("CPU tests and experiment must have completed with matching fingerprints")
    verify_experiment(workflow, require_headroom=True)
    return workflow


def prepare(args):
    phase = "gpu" if args.command == "benchmark" else "cpu"
    source = completed_source(workflow_path(args.source)) if phase == "gpu" else None
    original_config = cw.inside(source["config_path"] if source else args.config or ROOT / "configs" / "research.yaml")
    if not original_config.is_file():
        raise ValueError(f"Research configuration missing: {original_config}")
    run_id = identifier(args.run_id or "carc-research-" + ("gpu-" if phase == "gpu" else "") +
                        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    base = cw.inside(ROOT / "runs" / run_id)
    if base.exists():
        raise ValueError("Research run directory exists; preserve it and choose a fresh --run-id")
    torch_version = os.environ.get("TORCH_VERSION", "2.10.0+cu126")
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+\+cu[0-9]+", torch_version):
        raise ValueError("CARC research requires an exact CUDA Torch version")
    workflow = {"schema_version": 1, "kind": "bounded-research", "run_id": run_id,
                "root": str(ROOT.resolve()), "run_dir": str(base), "created_at": cw.now(),
                "phase": phase, "smoke": source["smoke"] if source else args.smoke,
                "original_config": str(original_config), "config_path": str(base / "config.yaml"),
                "config_sha256": cw.digest(original_config), "source_sha256": cw.source_hash(),
                "source_tree_sha256": cw.source_hash(metadata=True), "torch_version": torch_version,
                "source_workflow": str(workflow_path(args.source)) if source else None,
                "source_manifest_sha256": cw.digest(Path(source["run_dir"]) / "experiment" / "manifest.json") if source else None,
                "resources": {phase: {**RESOURCE, "partition": "gpu" if phase == "gpu" else os.environ.get("TDN_CPU_PARTITION", "main")}}}
    if source and (workflow["config_sha256"] != source["config_sha256"] or torch_version != source["torch_version"]):
        raise ValueError("Benchmark configuration and software selection must match the CPU source")
    return validate(workflow)


def scheduler_args(workflow):
    phase = workflow["phase"]
    resource = workflow["resources"][phase]
    base = cw.inside(workflow["run_dir"])
    args = ["sbatch", "--parsable", f"--account={ACCOUNT}", f"--partition={resource['partition']}",
            "--nodes=1", "--ntasks=1", f"--job-name=tdn-research-{phase}",
            f"--comment=tdn-research:{workflow['run_id']}:{phase}",
            f"--cpus-per-task={resource['cpus']}", f"--mem={resource['mem_gib']}G",
            f"--time={resource['walltime']}", "--signal=TERM@120", "--export=ALL",
            f"--output={base / 'logs' / (phase + '-%j.out')}",
            f"--error={base / 'logs' / (phase + '-%j.err')}"]
    if phase == "gpu":
        args += ["--gpus-per-task=a100:1", "--constraint=a100-40gb"]
    return args + [str(ROOT / "scripts" / f"research_{phase}.sbatch")]


def start(args):
    workflow = prepare(args)
    print(f"Research workflow: {workflow['run_id']} ({workflow['phase']})")
    print(f"Run: {workflow['run_dir']}\nConfig: {workflow['original_config']}")
    print("Scope: bounded development hypotheses; no confirmatory campaign or automatic GPU successor.")
    print(shlex.join(scheduler_args(workflow)))
    if not args.submit or args.dry_run:
        print("DRY RUN: no writes, numerical work, scheduler calls or jobs. Add --submit explicitly.")
        return
    cw.actual_policy()
    policy = cw.live_check(workflow, workflow["phase"])
    base = cw.inside(workflow["run_dir"])
    base.mkdir(parents=True, exist_ok=False)
    (base / "logs").mkdir()
    with Path(workflow["config_path"]).open("xb") as stream:
        stream.write(Path(workflow["original_config"]).read_bytes())
    if cw.digest(workflow["config_path"]) != workflow["config_sha256"] or cw.source_hash() != workflow["source_sha256"]:
        raise ValueError("Configuration or source changed during submission; no job submitted")
    if workflow["phase"] == "gpu":
        completed_source(workflow["source_workflow"])
    cw.atomic_json(base / "research-workflow.json", workflow)
    Path(workflow["config_path"]).chmod(0o444)
    (base / "research-workflow.json").chmod(0o444)
    cw.atomic_json(base / "state" / "slurm-policy.json", policy)
    cw.atomic_json(base / "jobs.json", [])
    cw.atomic_json(ROOT / "runs" / ".research-latest.json", {"run_id": workflow["run_id"]})
    env = os.environ.copy()
    env.update({"CARC_ACCOUNT": ACCOUNT, "TDN_REPO_ROOT": str(ROOT),
                "TDN_RESEARCH_WORKFLOW": str(base / "research-workflow.json"),
                "TDN_RESEARCH_PHASE": workflow["phase"], "TORCH_VERSION": workflow["torch_version"]})
    result = cw.command(scheduler_args(workflow), env=env)
    match = re.fullmatch(r"([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?", result.stdout.strip())
    if not match:
        raise ValueError("sbatch returned an ambiguous job ID; inspect the scheduler before retrying")
    job = match.group(1)
    cw.atomic_json(base / "jobs.json", [{"job_id": job, "phase": workflow["phase"], "submitted_at": cw.now()}])
    print(f"TDN_RESEARCH_RUN_ID={workflow['run_id']}\nTDN_RESEARCH_JOB_ID={job}")
    print(f"Monitor: bash scripts/carc_research.sh status {workflow['run_id']}")


def software_report(workflow):
    """Run package metadata inspection inside the allocated project venv."""
    python = cw.inside(ROOT / ".venv" / "bin") / "python"
    if not python.is_file() or not cw.inside(ROOT / ".venv" / "pyvenv.cfg").is_file():
        raise ValueError("Existing project Python venv is missing; run allocated setup separately")
    program = r'''
import hashlib, importlib.metadata as m, json, pathlib, platform, sys
root=pathlib.Path(sys.argv[1]).resolve()
if pathlib.Path(sys.prefix).resolve()!=root/'.venv' or sys.prefix==sys.base_prefix:
    raise SystemExit('Expected project Python venv')
if (pathlib.Path(sys.base_prefix)/'conda-meta').exists():
    raise SystemExit('Conda base Python is forbidden')
if not (3,11)<=sys.version_info[:2]<=(3,13):
    raise SystemExit('Use tested standalone Python 3.11-3.13')
expected={'torch':sys.argv[2], 'setuptools':'80.9.0', 'wheel':'0.45.1'}
for line in (root/'requirements.txt').read_text().splitlines():
    if line.strip() and not line.lstrip().startswith('#'):
        name,version=line.strip().split('=='); expected[name]=version
versions={name:m.version(name) for name in expected}
if versions!=expected: raise SystemExit('Existing venv differs from pinned dependencies; run allocated setup separately')
print(json.dumps({'python':platform.python_version(), 'prefix':str(pathlib.Path(sys.prefix).resolve()),
                  'base_prefix':str(pathlib.Path(sys.base_prefix).resolve()), 'packages':versions}))
'''
    return json.loads(cw.command([python, "-c", program, ROOT, workflow["torch_version"]]).stdout)


def verify_worker(workflow, phase):
    cw.actual_policy(worker=True)
    if phase != workflow["phase"]:
        raise ValueError("Allocated phase differs from workflow")
    job = os.environ["SLURM_JOB_ID"]
    result = cw.command(["scontrol", "show", "job", job, "-o"])
    fields = dict(re.findall(r"(\w+)=([^\s]+)", result.stdout))
    expected = {"JobId": job, "JobName": f"tdn-research-{phase}", "Account": ACCOUNT,
                "Comment": f"tdn-research:{workflow['run_id']}:{phase}", "JobState": "RUNNING"}
    if any(fields.get(key) != value for key, value in expected.items()) or fields.get("UserId", "").split("(")[0] != cw.USER:
        raise ValueError("Allocation is not this research workflow's running job")
    report = software_report(workflow)
    if phase == "gpu":
        source = completed_source(workflow["source_workflow"])
        if cw.digest(Path(source["run_dir"]) / "experiment" / "manifest.json") != workflow["source_manifest_sha256"]:
            raise ValueError("CPU research artifacts changed since benchmark submission")
        if cw.read_json(software_path(source)) != report:
            raise ValueError("Benchmark software differs from the completed CPU experiment")
    return report


def worker_commands(workflow):
    python = str(cw.inside(ROOT / ".venv" / "bin") / "python")
    base = cw.inside(workflow["run_dir"])
    engine = [python, str(ROOT / "scripts" / "research.py")]
    common = ["--config", workflow["config_path"], "--run-dir", str(base / "experiment")]
    if workflow["phase"] == "cpu":
        tests = sorted((ROOT / "tests").glob("test_research_*.py"))
        if not tests:
            raise ValueError("Focused research tests are missing")
        return [("tests", [python, "-m", "pytest", "-q", "-m", "not gpu", *map(str, tests),
                           "--basetemp", str(base / "pytest-work"), "-o", f"cache_dir={base / 'pytest-cache'}"]),
                ("experiment", engine + ["run", *common, "--device", "cpu"] + (["--smoke"] if workflow["smoke"] else []))]
    source = load(workflow["source_workflow"])
    return [("preflight", [python, str(ROOT / "scripts" / "gpu_preflight.py"), "--output", str(base / "gpu-preflight.json")]),
            ("gpu-tests", [python, "-m", "pytest", "-q", "-m", "gpu", str(ROOT / "tests" / "test_research_gpu.py"),
                           "--basetemp", str(base / "pytest-work"), "-o", f"cache_dir={base / 'pytest-cache'}",
                           "--junitxml", str(base / "gpu-tests.xml")]),
            ("benchmark", engine + ["benchmark", *common, "--device", "cuda", "--source-run", str(Path(source["run_dir"]) / "experiment")])]


def worker(workflow, phase):
    software = verify_worker(workflow, phase)
    base = cw.inside(workflow["run_dir"])
    if state_path(workflow).exists() or (base / "experiment").exists():
        raise ValueError("Research worker was already started; preserve it and use a fresh workflow")
    cw.atomic_json(software_path(workflow), software)
    record = {"status": "RUNNING", "exit_code": 0, "stage": "verify", "updated_at": cw.now(),
              "slurm_job_id": os.environ["SLURM_JOB_ID"], "slurm_step_id": os.environ["SLURM_STEP_ID"],
              "source_sha256": workflow["source_sha256"], "config_sha256": workflow["config_sha256"],
              "software_sha256": cw.digest(software_path(workflow))}
    cw.atomic_json(state_path(workflow), record)
    child = None
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    prior = {item: signal.signal(item, stop) for item in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1)}
    code = 1
    try:
        for stage, args in worker_commands(workflow):
            if stopped:
                raise InterruptedError("Stop requested; no successor stage started")
            load(base / "research-workflow.json")
            if software_report(workflow) != software:
                raise ValueError("Software changed while research was running")
            record.update(stage=stage, updated_at=cw.now())
            cw.atomic_json(state_path(workflow), record)
            print(f"TDN research: starting {stage} in job {record['slurm_job_id']} step {record['slurm_step_id']}", flush=True)
            env = os.environ.copy()
            if stage == "gpu-tests":
                env["TDN_REQUIRE_GPU_TESTS"] = "1"
            if stopped:
                raise InterruptedError("Stop requested; no successor stage started")
            child = subprocess.Popen(args, cwd=ROOT, env=env, start_new_session=True)
            if stopped and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
            while True:
                try:
                    code = child.wait(timeout=5)
                    break
                except subprocess.TimeoutExpired:
                    if stopped:
                        os.killpg(child.pid, signal.SIGKILL)
            child = None
            if stopped:
                raise InterruptedError("Allocated research was interrupted; outputs preserved")
            if code:
                raise ValueError(f"Research {stage} failed with exit code {code}")
            if stage == "gpu-tests":
                from xml.etree import ElementTree
                suites = ElementTree.parse(base / "gpu-tests.xml").getroot().findall(".//testsuite")
                if (sum(int(suite.get("tests", 0)) for suite in suites) <= 0 or
                        any(int(suite.get(key, 0)) for suite in suites for key in ("skipped", "failures", "errors"))):
                    raise ValueError("Actual research GPU tests must execute and pass with no skips")
            print(f"TDN research: completed {stage}", flush=True)
        load(base / "research-workflow.json")
        if software_report(workflow) != software:
            raise ValueError("Software changed while research was running")
        verify_experiment(workflow)
        record.update(status="COMPLETED", exit_code=0, updated_at=cw.now())
        cw.atomic_json(state_path(workflow), record)
        return 0
    except BaseException as exc:
        record.update(status="INTERRUPTED" if stopped else "FAILED", exit_code=143 if stopped else max(1, code),
                      error=str(exc), updated_at=cw.now())
        cw.atomic_json(state_path(workflow), record)
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        for item, handler in prior.items():
            signal.signal(item, handler)


def status(workflow):
    base = cw.inside(workflow["run_dir"])
    print(f"Research workflow {workflow['run_id']} ({workflow['phase']})\nRun: {base}")
    for entry in cw.read_json(base / "jobs.json"):
        job = entry["job_id"]
        if not JOB_PATTERN.fullmatch(job):
            raise ValueError("Invalid stored scheduler job ID")
        try:
            state = cw.scheduler_state(job)
        except FileNotFoundError:
            state = "UNKNOWN (scheduler unavailable in this checkout)"
        print(f"{workflow['phase']} job {job}: {state}")
    record = cw.read_json(state_path(workflow)) if state_path(workflow).exists() else {}
    print(f"  {record.get('stage', 'worker')}: {record.get('status', 'NOT_STARTED')}")
    summary_path = base / "experiment" / "summary.json"
    if summary_path.is_file():
        report = cw.read_json(summary_path)
        print(f"  experiment: {report.get('status', 'UNKNOWN')}")
        if workflow["phase"] == "cpu":
            print(f"  numerical headroom: {report.get('headroom', {}).get('passed', False)}")


def logs(workflow, lines):
    from collections import deque
    for path in sorted(cw.inside(Path(workflow["run_dir"]) / "logs").glob("*")):
        path = cw.inside(path)
        if path.is_file():
            print(f"\n{path}")
            with path.open(errors="replace") as stream:
                print("".join(deque(stream, maxlen=lines)), end="")


def collect(workflow):
    """Collect full bounded artifacts, including failures, without modifying them."""
    base = cw.inside(workflow["run_dir"])
    destination = cw.inside(ROOT / "runs" / (workflow["run_id"] + "-review-" +
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".tar.gz"))
    members = []
    roots = [base]
    if workflow["phase"] == "gpu":
        source = load(workflow["source_workflow"], verify=False)
        roots.append(cw.inside(source["run_dir"]))
    excluded = {"pytest-work", "pytest-cache", "__pycache__"}
    for run_root in roots:
        for path in sorted(run_root.rglob("*")):
            if any(part in excluded for part in path.relative_to(run_root).parts):
                continue
            if path.is_symlink():
                raise ValueError(f"Review archive refuses symlink: {path}")
            path = cw.inside(path)
            if path.is_file():
                members.append((path, str(Path(run_root.name) / path.relative_to(run_root))))
    with destination.open("xb") as stream:
        with tarfile.open(fileobj=stream, mode="w:gz", dereference=False) as archive:
            for path, name in members:
                archive.add(path, arcname=name, recursive=False)
    print(f"Review archive: {destination}")
    print("Includes protocol, data, checkpoints, metrics, logs and failures; excludes pytest temporary directories.")
    return destination


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("start", "benchmark"):
        item = commands.add_parser(name)
        if name == "benchmark":
            item.add_argument("source", help="Completed CPU research run ID (or latest)")
        else:
            item.add_argument("--config", type=Path)
            item.add_argument("--smoke", action="store_true", help="Tiny integration test; not a scientific comparison")
        item.add_argument("--run-id")
        item.add_argument("--submit", action="store_true")
        item.add_argument("--dry-run", action="store_true")
    for name in ("status", "logs", "collect"):
        item = commands.add_parser(name)
        item.add_argument("run", nargs="?", default="latest")
        if name == "logs":
            item.add_argument("--lines", type=int, default=200)
    item = commands.add_parser("worker", help=argparse.SUPPRESS)
    item.add_argument("--workflow", type=Path, required=True)
    item.add_argument("--phase", choices=("cpu", "gpu"), required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    lock = None
    try:
        if args.command in ("start", "benchmark"):
            if args.submit and not args.dry_run:
                cw.actual_policy()
                lock = cw.controller_lock()
            start(args)
        elif args.command == "worker":
            return worker(load(args.workflow), args.phase)
        else:
            workflow = load(workflow_path(args.run), verify=False)
            if args.command == "status":
                status(workflow)
            elif args.command == "logs":
                if not 1 <= args.lines <= 100000:
                    raise ValueError("--lines must be between 1 and 100000")
                logs(workflow, args.lines)
            else:
                collect(workflow)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"TDN research: {exc}", file=sys.stderr)
        return 2
    finally:
        if lock is not None:
            lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
