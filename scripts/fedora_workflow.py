#!/usr/bin/env python3
"""Explicit, bounded Fedora Slurm adapter for the four-stage premix experiment.

The controller uses the standard library only. Fedora ownership, partitions and
GPU requests come from a validated project-local profile; CARC stays separate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import os
from pathlib import Path
import re
import shlex
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("fedora_premix_helpers", ROOT / "scripts" / "premix_workflow.py")
pw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pw)
cw, rw = pw.cw, pw.rw
STAGES, CPU_STAGES, RESOURCE = pw.STAGES, pw.CPU_STAGES, pw.RESOURCE
from tdn.runtime.desktop_slurm import load_profile, validate_profile, verify_allocation as runtime_allocation


def declaration(profile):
    return {**pw.declaration(profile), "execution_mode": "desktop-slurm",
            "dependencies": {"scaling": ["accuracy"], "prepare": ["scaling"], "neural": list(CPU_STAGES)},
            "cpu_schedule": "serial stages on an eight-core desktop; no pending-job cap"}


def validate(workflow):
    if workflow.get("schema_version") != 1 or workflow.get("kind") != "desktop-slurm-premix":
        raise ValueError("Not a supported Fedora premix workflow")
    run_id = pw.identifier(workflow["run_id"])
    base = cw.inside(workflow["run_dir"])
    if base != cw.inside(ROOT / "runs" / run_id) or cw.inside(workflow["root"]) != ROOT.resolve():
        raise ValueError("Fedora workflow root/layout changed")
    if workflow.get("profile") not in ("smoke", "full") or workflow.get("execution_mode") != "desktop-slurm":
        raise ValueError("Unknown Fedora workflow profile or execution mode")
    if cw.inside(workflow["protocol_path"]) != base / "protocol.json":
        raise ValueError("Frozen protocol must remain within this workflow")
    if workflow.get("protocol_sha256") != pw.canonical_hash(declaration(workflow["profile"])):
        raise ValueError("Fedora protocol declaration changed")
    site = validate_profile(workflow["slurm_profile"], root=ROOT)
    if workflow.get("slurm_profile_sha256") != pw.canonical_hash(site):
        raise ValueError("Frozen Slurm profile fingerprint differs")
    if cw.inside(workflow["slurm_profile_path"]) != base / "slurm-profile.json":
        raise ValueError("Frozen Slurm profile must remain within this workflow")
    if workflow.get("torch_version") != site["torch_version"]:
        raise ValueError("Workflow Torch release differs from its Slurm profile")
    if set(workflow.get("resources", {})) != set(STAGES):
        raise ValueError("Fedora premix requires exactly four bounded stages")
    for stage in STAGES:
        expected = {**RESOURCE, "partition": site["gpu_partition" if stage == "neural" else "cpu_partition"]}
        if workflow["resources"][stage] != expected:
            raise ValueError("Each Fedora allocation requires the declared partition, 4 CPUs, 16 GiB and 30 minutes")
    for key in ("source_sha256", "source_tree_sha256"):
        if not pw.SHA.fullmatch(workflow.get(key, "")):
            raise ValueError("Missing execution source fingerprint")
    return workflow


def workflow_path(value):
    if value == "latest":
        value = cw.read_json(ROOT / "runs" / ".fedora-premix-latest.json")["run_id"]
    return cw.inside(ROOT / "runs" / pw.identifier(value) / "premix-workflow.json")


def load(path, *, verify=True):
    path = cw.inside(path)
    workflow = validate(cw.read_json(path))
    if path != Path(workflow["run_dir"]) / "premix-workflow.json":
        raise ValueError("Workflow manifest is outside its run directory")
    if verify:
        if cw.read_json(workflow["protocol_path"]) != declaration(workflow["profile"]):
            raise ValueError("Frozen scientific declaration changed; submit a fresh workflow")
        if load_profile(workflow["slurm_profile_path"], root=ROOT) != workflow["slurm_profile"]:
            raise ValueError("Frozen Slurm profile changed; submit a fresh workflow")
        if cw.source_hash() != workflow["source_sha256"]:
            raise ValueError("Execution source changed; submit a fresh workflow")
    return workflow


def prepare(args):
    site = load_profile(root=ROOT)
    run_id = pw.identifier(args.run_id or "fedora-premix-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    base = cw.inside(ROOT / "runs" / run_id)
    if base.exists():
        raise ValueError("Run directory exists; preserve it and choose a fresh --run-id")
    profile = "smoke" if args.smoke else "full"
    return validate({"schema_version": 1, "kind": "desktop-slurm-premix", "run_id": run_id,
        "root": str(ROOT.resolve()), "run_dir": str(base), "created_at": cw.now(),
        "execution_mode": "desktop-slurm", "profile": profile,
        "protocol_path": str(base / "protocol.json"), "protocol_sha256": pw.canonical_hash(declaration(profile)),
        "slurm_profile": site, "slurm_profile_path": str(base / "slurm-profile.json"),
        "slurm_profile_sha256": pw.canonical_hash(site), "torch_version": site["torch_version"],
        "source_sha256": cw.source_hash(), "source_tree_sha256": cw.source_hash(metadata=True),
        "resources": {stage: {**RESOURCE, "partition": site["gpu_partition" if stage == "neural" else "cpu_partition"]}
                      for stage in STAGES}})


def comment(workflow, stage):
    return f"tdn-fedora:{workflow['run_id']}:{stage}:{workflow['slurm_profile_sha256']}"


def scheduler_args(workflow, stage, dependency=None):
    if stage not in STAGES:
        raise ValueError("Unknown premix stage")
    base, site = Path(workflow["run_dir"]), workflow["slurm_profile"]
    args = ["sbatch", "--parsable", f"--partition={workflow['resources'][stage]['partition']}",
        "--nodes=1", "--ntasks=1", f"--job-name=tdn-fedora-{stage}", f"--comment={comment(workflow, stage)}",
        "--cpus-per-task=4", "--mem=16G", "--time=00:30:00", "--signal=USR1@120", "--export=ALL",
        f"--chdir={ROOT}", "--open-mode=append", "--kill-on-invalid-dep=yes",
        f"--output={base / 'logs' / (stage + '-%j.out')}", f"--error={base / 'logs' / (stage + '-%j.err')}",
        f"--gres={site['gpu_gres'] if stage == 'neural' else 'none'}"]
    if site.get("account"):
        args.append(f"--account={site['account']}")
    if dependency is not None:
        ids = dependency.split(":")
        count = 3 if stage == "neural" else 1
        if stage == "accuracy" or ids[0] != "afterok" or len(ids) != count + 1 or any(
                not pw.JOB.fullmatch(item) for item in ids[1:]) or len(set(ids[1:])) != count:
            raise ValueError("Invalid fixed workflow dependency")
        args.append(f"--dependency={dependency}")
    return args + [str(ROOT / "scripts" / "fedora_slurm_worker.sh")]


def controller_policy():
    if ROOT.resolve() == cw.CARC_ROOT:
        raise ValueError("Use the separate CARC workflow at the CARC project root")
    if any(os.environ.get(key) for key in ("SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_JOB_ACCOUNT")):
        raise ValueError("Submit Fedora workflows outside an existing Slurm allocation")
    if os.environ.get("TDN_EXECUTION_MODE", "desktop-slurm") != "desktop-slurm" or os.environ.get("TDN_LOCAL_TEST_ROOT"):
        raise ValueError("Clear local CPU or native-desktop overrides before Fedora Slurm submission")
    if os.environ.get("CONDA_PREFIX") or os.environ.get("CONDA_SHLVL", "0") != "0":
        raise ValueError("Deactivate Conda; use the standalone project Python venv")
    if os.environ.get("TDN_PROJECT_ROOT", str(ROOT)) != str(ROOT):
        raise ValueError("TDN_PROJECT_ROOT differs from this checkout")


def check_partitions(workflow):
    """Read-only live checks before any part of the DAG is submitted."""
    checks = {}
    for partition in dict.fromkeys(row["partition"] for row in workflow["resources"].values()):
        result = cw.command(["scontrol", "show", "partition", partition, "-o"])
        fields = dict(re.findall(r"(\w+)=([^\s]+)", result.stdout))
        if fields.get("PartitionName") != partition or fields.get("State") != "UP":
            raise ValueError(f"Configured partition {partition} is missing or not UP")
        limit = fields.get("MaxTime", "UNLIMITED")
        if limit not in ("UNLIMITED", "INFINITE"):
            parts = limit.split("-")
            clock = parts[-1].split(":")
            if not all(piece.isdigit() for piece in clock) or len(clock) != 3:
                raise ValueError(f"Cannot verify partition time limit: {limit}")
            seconds = (int(parts[0]) * 86400 if len(parts) == 2 else 0) + sum(
                int(number) * factor for number, factor in zip(clock, (3600, 60, 1)))
            if seconds < 1800:
                raise ValueError(f"Partition {partition} allows less than the declared 30 minutes")
        checks[partition] = fields
    return checks


def start(args):
    workflow = prepare(args)
    print(f"Fedora premix workflow: {workflow['run_id']} ({workflow['profile']})\nRun: {workflow['run_dir']}")
    print("CPU stages run sequentially; one desktop GPU job requires all three CPU stages to succeed.")
    print("Each allocation: 4 CPUs, 16 GiB, 30 minutes. No pending-job cap or automatic resubmission.")
    for stage in STAGES:
        print(shlex.join(scheduler_args(workflow, stage)))
    print("Dependencies: scaling after accuracy; prepare after scaling; neural after all three CPU jobs.")
    if args.command == "plan":
        print("PLAN ONLY: no writes, scheduler calls or numerical work.")
        return workflow
    controller_policy()
    policies = check_partitions(workflow)
    # Metadata inspection only; no Torch import or numerical work on the controller.
    with rw.acquire_venv_lock():
        software = rw.software_report(workflow)
    with cw.controller_lock():
        base = Path(workflow["run_dir"])
        base.mkdir(parents=True, exist_ok=False)
        (base / "logs").mkdir()
        path = base / "premix-workflow.json"
        for output, data in ((path, workflow), (Path(workflow["protocol_path"]), declaration(workflow["profile"])),
                             (Path(workflow["slurm_profile_path"]), workflow["slurm_profile"])):
            cw.atomic_json(output, data)
            output.chmod(0o444)
        cw.atomic_json(base / "state" / "slurm-policy.json", policies)
        cw.atomic_json(base / "state" / "submission-software.json", software)
        jobs = []
        cw.atomic_json(base / "jobs.json", jobs)
        cw.atomic_json(ROOT / "runs" / ".fedora-premix-latest.json", {"run_id": workflow["run_id"]})
        for stage in STAGES:
            dependency = ("afterok:" + ":".join(row["job_id"] for row in (jobs if stage == "neural" else jobs[-1:]))) if jobs else None
            try:
                load(path)
                # Do not inherit CARC/other-project scheduler defaults. Explicit
                # flags and the frozen local profile define this allocation.
                env = {key: value for key, value in os.environ.items()
                       if not key.startswith(("SBATCH_", "SRUN_")) and key not in (
                           "CARC_ACCOUNT", "TORCH_CUDA_ARCH_LIST")}
                env.update(TDN_EXECUTION_MODE="desktop-slurm", TDN_PROJECT_ROOT=str(ROOT), TDN_REPO_ROOT=str(ROOT),
                           TDN_SLURM_CONFIG=workflow["slurm_profile_path"], TDN_PREMIX_WORKFLOW=str(path),
                           TDN_PREMIX_STAGE=stage, TDN_FEDORA_GPU_GRES=workflow["slurm_profile"]["gpu_gres"],
                           TORCH_VERSION=workflow["torch_version"])
                result = cw.command(scheduler_args(workflow, stage, dependency), env=env)
                match = re.fullmatch(r"([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?", result.stdout.strip())
                if not match or any(row["job_id"] == match.group(1) for row in jobs):
                    raise ValueError("sbatch returned an ambiguous or duplicate ID; inspect Slurm before retrying")
                jobs.append({"stage": stage, "job_id": match.group(1), "dependency": dependency, "submitted_at": cw.now()})
                cw.atomic_json(base / "jobs.json", jobs)
                print(f"TDN_PREMIX_{stage.upper()}_JOB_ID={match.group(1)}", flush=True)
            except BaseException as exc:
                cw.atomic_json(base / "state" / "submission.json", {"status": "FAILED", "stage": stage,
                    "error": str(exc), "submitted_jobs": jobs, "updated_at": cw.now(),
                    "note": "Existing submissions preserved; no automatic retry or cancellation."})
                raise
        cw.atomic_json(base / "state" / "submission.json", {"status": "COMPLETED", "updated_at": cw.now()})
    print(f"Monitor: bash scripts/fedora_slurm.sh status {workflow['run_id']}")
    return workflow


def verify_stage(workflow, stage):
    manifest = pw.verify_stage(workflow, stage, execution_mode="desktop-slurm")
    execution = cw.read_json(Path(workflow["run_dir"]) / stage / "execution.json")
    if execution.get("slurm_profile_sha256") != workflow["slurm_profile_sha256"]:
        raise ValueError(f"{stage}: Slurm execution profile differs from the frozen workflow")
    return manifest


def verify_prerequisites(workflow, software):
    hashes = pw.verify_prerequisites(workflow, software, execution_mode="desktop-slurm")
    for stage in CPU_STAGES:
        verify_stage(workflow, stage)
    return hashes


def verify_allocation(workflow, stage):
    if stage not in STAGES:
        raise ValueError("Unknown premix stage")
    site = workflow["slurm_profile"]
    if load_profile(root=ROOT) != site:
        raise ValueError("Active Slurm profile differs from this frozen workflow")
    runtime_allocation("cuda" if stage == "neural" else "cpu", profile=site, root=ROOT)
    job = os.environ["SLURM_JOB_ID"]
    result = cw.command(["scontrol", "show", "job", job, "-o"])
    # Split at the next key so paths containing spaces remain intact.
    fields = dict(re.findall(r"(?:^|\s)(\w+)=(.*?)(?=\s\w+=|$)", result.stdout.strip()))
    expected = {"JobId": job, "JobName": f"tdn-fedora-{stage}", "Comment": comment(workflow, stage),
                "JobState": "RUNNING", "Partition": workflow["resources"][stage]["partition"], "WorkDir": str(ROOT)}
    if any(fields.get(key) != value for key, value in expected.items()) or fields.get("UserId", "").split("(")[0] != site["user"]:
        raise ValueError("Allocation is not this Fedora workflow's running stage")
    jobs = cw.read_json(Path(workflow["run_dir"]) / "jobs.json")
    recorded = [row for row in jobs if row.get("stage") == stage]
    # Slurm can start the allocation before sbatch returns its ID. The exact
    # scheduler owner/name/comment/root/profile checks above establish identity
    # during that short interval; a recorded mismatch is always an error.
    if recorded and (len(recorded) != 1 or recorded[0].get("job_id") != job):
        raise ValueError("Allocation differs from the workflow submission record")


def worker_commands(workflow, stage):
    commands = pw.worker_commands(workflow, stage)
    for name, command in commands:
        if name == "preflight":
            command[1] = str(ROOT / "scripts" / "fedora_gpu_preflight.py")
        elif name == "tests":
            command[command.index("--basetemp"):command.index("--basetemp")] = [
                str(path) for path in sorted((ROOT / "tests").glob("test_fedora_*.py"))] + [
                str(ROOT / "tests" / "test_desktop_slurm_runtime.py")]
    return commands


def begin_report(workflow, stage):
    base, site = Path(workflow["run_dir"]), workflow["slurm_profile"]
    job = os.environ["SLURM_JOB_ID"]
    resources = {"partition": workflow["resources"][stage]["partition"], "nodes": 1, "cpus": 4,
                 "gpus": int(stage == "neural"), "mem_bytes": 16 * 1024**3, "time_seconds": 1800}
    if site.get("account"):
        resources["account"] = site["account"]
    if stage == "neural":
        resources["gpu_type"] = site["expected_gpu_name"]
    return cw.reporting_api().begin_report(base / stage, name=f"TDN/premix/{stage}", script="scripts/fedora_slurm_worker.sh",
        parameters={"source_sha256": workflow["source_sha256"], "protocol_sha256": workflow["protocol_sha256"],
                    "slurm_profile_sha256": workflow["slurm_profile_sha256"], "stage": stage,
                    "profile": workflow["profile"], "execution_mode": "desktop-slurm"},
        resources=resources, job_id=job, report_parent=base / "tower",
        logs=[{"id": f"scheduler.{kind}", "path": str(base / "logs" / f"{stage}-{job}.{suffix}"),
               "label": f"Slurm {kind}", "group": "Scheduler"} for kind, suffix in (("stdout", "out"), ("stderr", "err"))],
        metadata={"workflow_id": workflow["run_id"], "phase": stage, "execution_scope": "one bounded Fedora Slurm allocation"})


def worker(workflow, stage):
    backend = SimpleNamespace(**{name: globals()[name] for name in ("verify_allocation", "load", "begin_report",
        "worker_commands", "verify_stage", "verify_prerequisites", "prepare_test_environment")})
    return pw.worker(workflow, stage, backend=backend)


def prepare_test_environment(env, name):
    if name == "tests":
        for key in ("TDN_EXECUTION_MODE", "TDN_SLURM_CONFIG"):
            env.pop(key, None)


def status(workflow):
    base = Path(workflow["run_dir"])
    print(f"Fedora premix workflow {workflow['run_id']} ({workflow['profile']})\nRun: {base}")
    jobs = {row["stage"]: row for row in cw.read_json(base / "jobs.json")}
    for stage in STAGES:
        if stage in jobs:
            job = jobs[stage]["job_id"]
            if not pw.JOB.fullmatch(job):
                raise ValueError("Invalid stored job ID")
            try:
                state = cw.scheduler_state(job)
            except (FileNotFoundError, ValueError):
                state = "UNKNOWN (scheduler unavailable)"
            print(f"{stage} job {job}: {state}")
        else:
            print(f"{stage}: NOT_SUBMITTED")
        path = pw.state_path(workflow, stage)
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
    pw.paths(workflow)


def parser():
    result = pw.parser()
    result.description = __doc__
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command in ("plan", "run"):
            start(args)
        elif args.command == "worker":
            return worker(load(args.workflow, verify=False), args.stage)
        else:
            workflow = load(workflow_path(args.run), verify=False)
            if args.command == "status":
                status(workflow)
            elif args.command == "logs":
                pw.logs(workflow, args.lines)
            else:
                getattr(pw, args.command)(workflow)
    except rw.WorkerFailure as exc:
        print(f"TDN Fedora: {exc}", file=sys.stderr)
        return exc.exit_code
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"TDN Fedora: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
