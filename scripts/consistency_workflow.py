#!/usr/bin/env python3
"""Bounded Fedora consistency experiment: audit, prepare, then allocated CUDA.

Controllers perform standard-library metadata and scheduler work only. The new
run pointer keeps historical premix runs intact; science lives in the venv task.
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
_spec = importlib.util.spec_from_file_location("consistency_fedora_helpers", ROOT / "scripts" / "fedora_workflow.py")
fw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fw)
pw, cw, rw = fw.pw, fw.cw, fw.rw
STAGES = ("audit", "prepare", "neural")
CPU_STAGES = STAGES[:2]
RESOURCE = {"cpus": 4, "mem_gib": 16, "walltime": "00:30:00"}
MANIFEST = "consistency-workflow.json"
POINTER = ".fedora-consistency-latest.json"
from tdn.runtime.desktop_slurm import load_profile, validate_profile, verify_allocation as runtime_allocation


def scientific_protocol(profile):
    # The protocol package is dependency-free for plan and inspection.
    from tdn.analysis.consistency.protocol import build_protocol
    return build_protocol(profile)

def declaration(profile):
    return {"version": 1, "benchmark_suite": "consistency", "profile": profile,
            "scientific_protocol": scientific_protocol(profile), "stages": list(STAGES),
            "execution_mode": "desktop-slurm", "dependencies": {"prepare": ["audit"], "neural": list(CPU_STAGES)},
            "device": {stage: "cuda" if stage == "neural" else "cpu" for stage in STAGES},
            "resources_per_stage": RESOURCE,
            "scope": "bounded development experiment; no automatic expansion or pending-job cap"}


def validate(workflow):
    if workflow.get("schema_version") != 1 or workflow.get("kind") != "desktop-slurm-consistency":
        raise ValueError("Not a supported Fedora consistency workflow")
    run_id = pw.identifier(workflow["run_id"])
    base = cw.inside(workflow["run_dir"])
    if base != cw.inside(ROOT / "runs" / run_id) or cw.inside(workflow["root"]) != ROOT.resolve():
        raise ValueError("Consistency workflow root/layout changed")
    if workflow.get("profile") not in ("smoke", "full") or workflow.get("execution_mode") != "desktop-slurm":
        raise ValueError("Unknown consistency profile or execution mode")
    if cw.inside(workflow["protocol_path"]) != base / "protocol.json":
        raise ValueError("Frozen protocol must remain within this workflow")
    if workflow.get("protocol_sha256") != pw.canonical_hash(declaration(workflow["profile"])):
        raise ValueError("Consistency protocol declaration changed")
    site = validate_profile(workflow["slurm_profile"], root=ROOT)
    if workflow.get("slurm_profile_sha256") != pw.canonical_hash(site):
        raise ValueError("Frozen Slurm profile fingerprint differs")
    if cw.inside(workflow["slurm_profile_path"]) != base / "slurm-profile.json":
        raise ValueError("Frozen Slurm profile must remain within this workflow")
    if workflow.get("torch_version") != site["torch_version"]:
        raise ValueError("Workflow Torch release differs from its Slurm profile")
    if set(workflow.get("resources", {})) != set(STAGES):
        raise ValueError("Consistency requires exactly three bounded stages")
    for stage in STAGES:
        expected = {**RESOURCE, "partition": site["gpu_partition" if stage == "neural" else "cpu_partition"]}
        if workflow["resources"][stage] != expected:
            raise ValueError("Each consistency allocation requires the declared partition, 4 CPUs, 16 GiB and 30 minutes")
    for key in ("source_sha256", "source_tree_sha256"):
        if not pw.SHA.fullmatch(workflow.get(key, "")):
            raise ValueError("Missing execution source fingerprint")
    return workflow


def workflow_path(value):
    if value == "latest":
        value = cw.read_json(ROOT / "runs" / POINTER)["run_id"]
    return cw.inside(ROOT / "runs" / pw.identifier(value) / MANIFEST)


def load(path, *, verify=True):
    path = cw.inside(path)
    workflow = validate(cw.read_json(path))
    if path != Path(workflow["run_dir"]) / MANIFEST:
        raise ValueError("Consistency manifest is outside its run directory")
    if verify:
        if cw.read_json(workflow["protocol_path"]) != declaration(workflow["profile"]):
            raise ValueError("Frozen consistency declaration changed; submit a fresh workflow")
        if load_profile(workflow["slurm_profile_path"], root=ROOT) != workflow["slurm_profile"]:
            raise ValueError("Frozen Slurm profile changed; submit a fresh workflow")
        if cw.source_hash() != workflow["source_sha256"]:
            raise ValueError("Execution source changed; submit a fresh workflow")
    return workflow


def prepare(args):
    site = load_profile(root=ROOT)
    run_id = pw.identifier(args.run_id or "fedora-consistency-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    base = cw.inside(ROOT / "runs" / run_id)
    if base.exists():
        raise ValueError("Run directory exists; preserve it and choose a fresh --run-id")
    profile = "smoke" if args.smoke else "full"
    return validate({"schema_version": 1, "kind": "desktop-slurm-consistency", "run_id": run_id,
        "root": str(ROOT.resolve()), "run_dir": str(base), "created_at": cw.now(),
        "execution_mode": "desktop-slurm", "profile": profile,
        "protocol_path": str(base / "protocol.json"), "protocol_sha256": pw.canonical_hash(declaration(profile)),
        "slurm_profile": site, "slurm_profile_path": str(base / "slurm-profile.json"),
        "slurm_profile_sha256": pw.canonical_hash(site), "torch_version": site["torch_version"],
        "source_sha256": cw.source_hash(), "source_tree_sha256": cw.source_hash(metadata=True),
        "resources": {stage: {**RESOURCE, "partition": site["gpu_partition" if stage == "neural" else "cpu_partition"]}
                      for stage in STAGES}})


def comment(workflow, stage):
    return f"tdn-consistency:{workflow['run_id']}:{stage}:{workflow['slurm_profile_sha256']}"


def scheduler_args(workflow, stage, dependency=None):
    if stage not in STAGES:
        raise ValueError("Unknown consistency stage")
    base, site = Path(workflow["run_dir"]), workflow["slurm_profile"]
    args = ["sbatch", "--parsable", f"--partition={workflow['resources'][stage]['partition']}",
        "--nodes=1", "--ntasks=1", f"--job-name=tdn-consistency-{stage}", f"--comment={comment(workflow, stage)}",
        "--cpus-per-task=4", "--mem=16G", "--time=00:30:00", "--signal=USR1@120", "--export=ALL",
        f"--chdir={ROOT}", "--open-mode=append", "--kill-on-invalid-dep=yes",
        f"--output={base / 'logs' / (stage + '-%j.out')}", f"--error={base / 'logs' / (stage + '-%j.err')}",
        f"--gres={site['gpu_gres'] if stage == 'neural' else 'none'}"]
    if site.get("account"):
        args.append(f"--account={site['account']}")
    if dependency is not None:
        ids = dependency.split(":")
        count = 2 if stage == "neural" else 1
        if stage == "audit" or ids[0] != "afterok" or len(ids) != count + 1 or any(
                not pw.JOB.fullmatch(item) for item in ids[1:]) or len(set(ids[1:])) != count:
            raise ValueError("Invalid fixed consistency dependency")
        args.append(f"--dependency={dependency}")
    return args + [str(ROOT / "scripts" / "consistency_worker.sh")]


def start(args):
    workflow = prepare(args)
    print(f"Fedora consistency workflow: {workflow['run_id']} ({workflow['profile']})\nRun: {workflow['run_dir']}")
    print("Stages run sequentially: audit -> prepare -> neural. The GPU requires both sealed CPU prerequisites.")
    print("Each allocation: 4 CPUs, 16 GiB, 30 minutes. No pending-job cap or automatic resubmission.")
    for stage in STAGES:
        print(shlex.join(scheduler_args(workflow, stage)))
    if args.command == "plan":
        print("PLAN ONLY: no writes, scheduler calls or numerical work.")
        return workflow
    fw.controller_policy()
    policies = fw.check_partitions(workflow)
    with rw.acquire_venv_lock():
        software = rw.software_report(workflow)
    with cw.controller_lock():
        base = Path(workflow["run_dir"])
        base.mkdir(parents=True, exist_ok=False)
        (base / "logs").mkdir()
        path = base / MANIFEST
        for output, data in ((path, workflow), (Path(workflow["protocol_path"]), declaration(workflow["profile"])),
                             (Path(workflow["slurm_profile_path"]), workflow["slurm_profile"])):
            cw.atomic_json(output, data)
            output.chmod(0o444)
        cw.atomic_json(base / "state" / "slurm-policy.json", policies)
        cw.atomic_json(base / "state" / "submission-software.json", software)
        jobs = []
        cw.atomic_json(base / "jobs.json", jobs)
        cw.atomic_json(ROOT / "runs" / POINTER, {"run_id": workflow["run_id"]})
        for stage in STAGES:
            dependency = ("afterok:" + ":".join(row["job_id"] for row in (jobs if stage == "neural" else jobs[-1:]))) if jobs else None
            try:
                load(path)
                env = {key: value for key, value in os.environ.items()
                       if not key.startswith(("SBATCH_", "SRUN_")) and key not in (
                           "CARC_ACCOUNT", "TORCH_CUDA_ARCH_LIST", "TDN_PREMIX_WORKFLOW", "TDN_PREMIX_STAGE",
                           "TDN_PREMIX_PROTOCOL_SHA256", "TDN_CONSISTENCY_WORKFLOW", "TDN_CONSISTENCY_STAGE",
                           "TDN_CONSISTENCY_PROTOCOL_SHA256")}
                env.update(TDN_EXECUTION_MODE="desktop-slurm", TDN_PROJECT_ROOT=str(ROOT), TDN_REPO_ROOT=str(ROOT),
                           TDN_SLURM_CONFIG=workflow["slurm_profile_path"], TDN_CONSISTENCY_WORKFLOW=str(path),
                           TDN_CONSISTENCY_STAGE=stage, TDN_FEDORA_GPU_GRES=workflow["slurm_profile"]["gpu_gres"],
                           TORCH_VERSION=workflow["torch_version"])
                result = cw.command(scheduler_args(workflow, stage, dependency), env=env)
                match = re.fullmatch(r"([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?", result.stdout.strip())
                if not match or any(row["job_id"] == match.group(1) for row in jobs):
                    raise ValueError("sbatch returned an ambiguous or duplicate ID; inspect Slurm before retrying")
                jobs.append({"stage": stage, "job_id": match.group(1), "dependency": dependency, "submitted_at": cw.now()})
                cw.atomic_json(base / "jobs.json", jobs)
                print(f"TDN_CONSISTENCY_{stage.upper()}_JOB_ID={match.group(1)}", flush=True)
            except BaseException as exc:
                cw.atomic_json(base / "state" / "submission.json", {"status": "FAILED", "stage": stage,
                    "error": str(exc), "submitted_jobs": jobs, "updated_at": cw.now(),
                    "note": "Existing submissions preserved; no automatic retry or cancellation."})
                raise
        cw.atomic_json(base / "state" / "submission.json", {"status": "COMPLETED", "updated_at": cw.now()})
    print(f"Monitor: bash scripts/fedora_consistency.sh status {workflow['run_id']}")
    return workflow


def verify_stage(workflow, stage):
    from tdn.analysis.consistency.artifacts import verify_stage as verify_artifacts
    base = cw.inside(Path(workflow["run_dir"]) / stage)
    manifest = verify_artifacts(base, stage=stage, profile=workflow["profile"],
                                source_tree_sha256=workflow["source_tree_sha256"])
    if cw.read_json(base / "protocol.json") != scientific_protocol(workflow["profile"]):
        raise ValueError(f"{stage}: scientific protocol differs from the frozen workflow")
    execution = cw.read_json(base / "execution.json")
    expected = {"stage": stage, "profile": workflow["profile"], "execution_mode": "desktop-slurm",
                "device": "cuda" if stage == "neural" else "cpu", "slurm_profile_sha256": workflow["slurm_profile_sha256"],
                "workflow_protocol_sha256": workflow["protocol_sha256"]}
    if any(execution.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{stage}: execution provenance differs from the frozen workflow")
    if cw.read_json(base / "summary.json").get("status") != "COMPLETED":
        raise ValueError(f"{stage}: experiment did not complete")
    if stage == "audit" and cw.read_json(base / "summary.json").get("correctness_failures") != 0:
        raise ValueError("Structural audit correctness failures forbid successor stages")
    return manifest


def verify_prerequisites(workflow, software, stage="neural"):
    required = ("audit",) if stage == "prepare" else CPU_STAGES
    hashes = {}
    for prerequisite in required:
        record = cw.read_json(pw.state_path(workflow, prerequisite))
        expected = {"status": "COMPLETED", "exit_code": 0, "source_sha256": workflow["source_sha256"],
                    "protocol_sha256": workflow["protocol_sha256"]}
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError(f"{prerequisite}: CPU worker prerequisite did not complete")
        software_path = Path(workflow["run_dir"]) / "state" / f"{prerequisite}-software.json"
        if (record.get("software_sha256") != cw.digest(software_path) or cw.read_json(software_path) != software):
            raise ValueError(f"{prerequisite}: prerequisite software differs")
        verify_stage(workflow, prerequisite)
        hashes[prerequisite] = cw.digest(Path(workflow["run_dir"]) / prerequisite / "manifest.json")
    return hashes


def verify_software(workflow, software):
    if cw.read_json(Path(workflow["run_dir"]) / "state" / "submission-software.json") != software:
        raise ValueError("Software differs from the frozen submission environment")


def verify_allocation(workflow, stage):
    if stage not in STAGES:
        raise ValueError("Unknown consistency stage")
    site = workflow["slurm_profile"]
    if load_profile(root=ROOT) != site:
        raise ValueError("Active Slurm profile differs from this frozen workflow")
    allocation = runtime_allocation("cuda" if stage == "neural" else "cpu", profile=site, root=ROOT)
    job, fields = allocation["job_id"], allocation["job"]
    expected = {"JobId": job, "JobName": f"tdn-consistency-{stage}", "Comment": comment(workflow, stage),
                "JobState": "RUNNING", "Partition": workflow["resources"][stage]["partition"], "WorkDir": str(ROOT)}
    if any(fields.get(key) != value for key, value in expected.items()) or fields.get("UserId", "").split("(")[0] != site["user"]:
        raise ValueError("Allocation is not this consistency workflow's running stage")
    jobs = cw.read_json(Path(workflow["run_dir"]) / "jobs.json")
    recorded = [row for row in jobs if row.get("stage") == stage]
    if recorded and (len(recorded) != 1 or recorded[0].get("job_id") != job):
        raise ValueError("Allocation differs from the workflow submission record")


def junit_path(workflow, stage):
    return cw.inside(Path(workflow["run_dir"]) / "reporter-tests" / stage / "tests.xml")


def gpu_junit_path(workflow):
    return junit_path(workflow, "neural")


def worker_commands(workflow, stage):
    python = str(cw.inside(ROOT / ".venv" / "bin") / "python")
    base = Path(workflow["run_dir"])
    commands = []
    if stage in ("audit", "neural"):
        tests = ([ROOT / "tests" / "test_consistency_gpu.py"] if stage == "neural" else
                 sorted((ROOT / "tests").glob("test_consistency*.py")) +
                 [ROOT / "tests" / "test_desktop_slurm_runtime.py", ROOT / "tests" / "test_fedora_workflow.py"])
        if not tests or any(not path.is_file() for path in tests):
            raise ValueError("Consistency correctness tests are missing")
        if stage == "neural":
            commands.append(("preflight", [python, str(ROOT / "scripts" / "fedora_gpu_preflight.py"),
                                           "--output", str(base / "gpu-preflight.json")]))
        commands.append(("gpu-tests" if stage == "neural" else "tests",
            [python, "-m", "pytest", "-q", "-m", "gpu" if stage == "neural" else "not gpu", *map(str, tests),
             "--basetemp", str(base / f"{stage}-pytest-work"), "-o", f"cache_dir={base / (stage + '-pytest-cache')}",
             "--junitxml", str(junit_path(workflow, stage))]))
    args = [python, str(ROOT / "scripts" / "consistency.py"), "--stage", stage,
            "--profile", workflow["profile"], "--run-dir", str(base / stage),
            "--device", "cuda" if stage == "neural" else "cpu"]
    if stage in ("prepare", "neural"):
        args += ["--audit-dir", str(base / "audit")]
    if stage == "neural":
        args += ["--dataset-dir", str(base / "prepare")]
    commands.append(("experiment", args))
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
    return cw.reporting_api().begin_report(base / stage, name=f"TDN/consistency/{stage}", script="scripts/consistency_worker.sh",
        parameters={"source_sha256": workflow["source_sha256"], "protocol_sha256": workflow["protocol_sha256"],
                    "slurm_profile_sha256": workflow["slurm_profile_sha256"], "stage": stage,
                    "profile": workflow["profile"], "execution_mode": "desktop-slurm"},
        resources=resources, job_id=job, report_parent=base / "tower",
        logs=[{"id": f"scheduler.{kind}", "path": str(base / "logs" / f"{stage}-{job}.{suffix}"),
               "label": f"Slurm {kind}", "group": "Scheduler"} for kind, suffix in (("stdout", "out"), ("stderr", "err"))],
        metadata={"workflow_id": workflow["run_id"], "phase": stage, "execution_scope": "one bounded Fedora Slurm allocation"})


def finish_report(workflow, stage, report, *, state, runtime_seconds, exit_code, software=None, error=None):
    from tdn.tower_analytics import publish_outputs
    base = Path(workflow["run_dir"])
    if cw.inside(report).parent != base / "tower":
        raise ValueError("Tower report is not this workflow's sidecar")
    if cw.read_json(report / "run.json").get("job_id") != os.environ["SLURM_JOB_ID"]:
        raise ValueError("Tower report belongs to a different scheduler job")
    sources = [path for path in (base / stage, base / "reporter-tests" / stage) if path.is_dir()]
    results = publish_outputs(report, sources)
    cw.reporting_api().finish_report(report, state=state, runtime_seconds=runtime_seconds, exit_code=exit_code, results=results,
        metadata={"workflow_id": workflow["run_id"], "phase": stage,
                  "execution_scope": "allocated worker elapsed time; excludes pending time",
                  **({"verified_software_sha256": pw.canonical_hash(software)} if software else {}),
                  **({"error": str(error)} if error is not None else {})})


def workflow_environment(workflow, stage):
    return {"TDN_CONSISTENCY_PROTOCOL_SHA256": workflow["protocol_sha256"],
            "TDN_CONSISTENCY_WORKFLOW": str(Path(workflow["run_dir"]) / MANIFEST), "TDN_CONSISTENCY_STAGE": stage}


def prepare_test_environment(env, name):
    for key in ("TDN_CONSISTENCY_PROTOCOL_SHA256", "TDN_CONSISTENCY_WORKFLOW", "TDN_CONSISTENCY_STAGE"):
        env.pop(key, None)
    if name == "tests":
        for key in ("TDN_EXECUTION_MODE", "TDN_SLURM_CONFIG"):
            env.pop(key, None)


def worker(workflow, stage):
    backend = SimpleNamespace(**{name: globals()[name] for name in ("verify_allocation", "load", "begin_report",
        "worker_commands", "verify_stage", "prepare_test_environment", "workflow_environment", "finish_report", "gpu_junit_path", "verify_software")},
        workflow_filename=MANIFEST, workflow_label="consistency", stage_variable="TDN_CONSISTENCY_STAGE",
        prerequisite_stages=("prepare", "neural"),
        verify_prerequisites=lambda workflow, software: verify_prerequisites(workflow, software, stage))
    return pw.worker(workflow, stage, backend=backend)


def paths(workflow):
    for row in cw.read_json(Path(workflow["run_dir"]) / "jobs.json"):
        if row.get("stage") not in STAGES or not pw.JOB.fullmatch(row.get("job_id", "")):
            raise ValueError("Invalid stored consistency scheduler job")
        for report in cw.tower_reports_for_job(workflow, row["job_id"]):
            print(f"{row['stage']} job {row['job_id']}\nTDN_TOWER_DIR={report}\nTDN_TOWER_METRICS={report / 'metrics.jsonl'}")


def status(workflow):
    base = Path(workflow["run_dir"])
    print(f"Fedora consistency workflow {workflow['run_id']} ({workflow['profile']})\nRun: {base}")
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
    paths(workflow)


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
            return worker(load(args.workflow, verify=False), args.stage)
        else:
            workflow = load(workflow_path(args.run), verify=False)
            if args.command == "logs":
                pw.logs(workflow, args.lines)
            elif args.command == "collect":
                pw.collect(workflow)
            else:
                globals()[args.command](workflow)
    except rw.WorkerFailure as exc:
        print(f"TDN consistency: {exc}", file=sys.stderr)
        return exc.exit_code
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"TDN consistency: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
