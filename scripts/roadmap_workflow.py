#!/usr/bin/env python3
"""Ten-stage bounded Fedora research roadmap; controllers do metadata work only."""
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
import subprocess
import sys
import tarfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("roadmap_fedora_helpers", ROOT / "scripts" / "fedora_workflow.py")
fw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fw)
pw, cw, rw = fw.pw, fw.cw, fw.rw
from tdn.runtime.desktop_slurm import load_profile, validate_profile, verify_allocation as runtime_allocation

STAGES = ("audit", "headroom", "prepare", "train", "confirm_prepare", "confirm", "policy", "transfer", "scaling", "report")
GPU_STAGES = ("train", "confirm", "policy", "scaling")
CPU_STAGES = tuple(stage for stage in STAGES if stage not in GPU_STAGES)
MANIFEST = "roadmap-workflow.json"
POINTER = ".fedora-roadmap-latest.json"
PROFILES = ("smoke", "development", "full")
SCHEDULER_MEMORY_CAP_MIB = 110000
ARCHIVE_PART_BYTES = 28 * 2**20


def scientific_protocol(profile):
    from tdn.analysis.roadmap.protocol import build_protocol
    return build_protocol(profile)


def stage_resources(profile):
    """Scheduler limits are part of the immutable scientific declaration."""
    protocol = scientific_protocol(profile)
    return {stage: {key: protocol["budgets"][stage][key] for key in ("cpus", "mem_gib", "walltime")} for stage in STAGES}


def declaration(profile):
    return {"version": 1, "benchmark_suite": "roadmap", "profile": profile,
            "scientific_protocol": scientific_protocol(profile), "stages": list(STAGES),
            "execution_mode": "desktop-slurm",
            "dependencies": {stage: list(STAGES[:index]) for index, stage in enumerate(STAGES)},
            "device": {stage: "cpu" if stage in CPU_STAGES else "cuda" for stage in STAGES},
            "resources_per_stage": stage_resources(profile),
            "scheduler_memory_cap_mib": SCHEDULER_MEMORY_CAP_MIB,
            "scope": "bounded preregistered roadmap; no automatic expansion or pending-job cap",
            "report_dependencies": "afterany across all scientific stages",
            "scientific_bad_is_scheduler_failure": False}


def wall_seconds(value):
    if not isinstance(value, str) or not re.fullmatch(r"(?:[0-9]+-)?[0-9]{2}:[0-9]{2}:[0-9]{2}", value):
        raise ValueError("Invalid bounded allocation walltime")
    parts = value.split("-")
    hours, minutes, seconds = map(int, parts[-1].split(":"))
    if minutes >= 60 or seconds >= 60:
        raise ValueError("Invalid allocation minutes/seconds")
    return (int(parts[0]) * 86400 if len(parts) == 2 else 0) + hours * 3600 + minutes * 60 + seconds


def validate_resources(resources):
    if not isinstance(resources, dict) or set(resources) != set(STAGES):
        raise ValueError("Roadmap requires exactly ten bounded stages")
    for stage, resource in resources.items():
        if not isinstance(resource, dict) or set(resource) != {"cpus", "mem_gib", "walltime"}:
            raise ValueError(f"{stage}: invalid allocation resource declaration")
        if type(resource["cpus"]) is not int or not 1 <= resource["cpus"] <= 8:
            raise ValueError("Roadmap CPU requests must fit the eight physical desktop cores")
        if stage in GPU_STAGES and resource["cpus"] != 4:
            raise ValueError("Roadmap GPU stages require exactly four allocated CPUs")
        if type(resource["mem_gib"]) is not int or resource["mem_gib"] not in (32, 48) or resource["mem_gib"] * 1024 > SCHEDULER_MEMORY_CAP_MIB:
            raise ValueError("Roadmap memory request exceeds Slurm's 110000 MiB host limit")
        if not 180 <= wall_seconds(resource["walltime"]) <= 2700:
            raise ValueError("Roadmap allocations must stay within the declared 45-minute hard maximum")
    return resources


def validate(workflow):
    if workflow.get("schema_version") != 1 or workflow.get("kind") != "desktop-slurm-roadmap":
        raise ValueError("Not a supported Fedora roadmap workflow")
    run_id = pw.identifier(workflow["run_id"])
    base = cw.inside(workflow["run_dir"])
    if base != cw.inside(ROOT / "runs" / run_id) or cw.inside(workflow["root"]) != ROOT.resolve():
        raise ValueError("Roadmap workflow root/layout changed")
    profile = workflow.get("profile")
    if profile not in PROFILES or workflow.get("execution_mode") != "desktop-slurm":
        raise ValueError("Unknown roadmap profile or execution mode")
    if cw.inside(workflow["protocol_path"]) != base / "protocol.json":
        raise ValueError("Frozen protocol must remain within this workflow")
    if workflow.get("protocol_sha256") != pw.canonical_hash(declaration(profile)):
        raise ValueError("Roadmap protocol declaration changed")
    site = validate_profile(workflow["slurm_profile"], root=ROOT)
    if workflow.get("slurm_profile_sha256") != pw.canonical_hash(site):
        raise ValueError("Frozen Slurm profile fingerprint differs")
    if cw.inside(workflow["slurm_profile_path"]) != base / "slurm-profile.json":
        raise ValueError("Frozen Slurm profile must remain within this workflow")
    if workflow.get("torch_version") != site["torch_version"]:
        raise ValueError("Workflow Torch release differs from its Slurm profile")
    expected_resources = validate_resources(stage_resources(profile))
    if set(workflow.get("resources", {})) != set(STAGES):
        raise ValueError("Roadmap requires exactly ten bounded stages")
    for stage in STAGES:
        expected = {**expected_resources[stage], "partition": site["cpu_partition" if stage in CPU_STAGES else "gpu_partition"]}
        if workflow["resources"][stage] != expected:
            raise ValueError(f"{stage}: allocation differs from the frozen desktop budget")
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
        raise ValueError("Roadmap manifest is outside its run directory")
    if verify:
        if cw.read_json(workflow["protocol_path"]) != declaration(workflow["profile"]):
            raise ValueError("Frozen roadmap declaration changed; submit a fresh workflow")
        if load_profile(workflow["slurm_profile_path"], root=ROOT) != workflow["slurm_profile"]:
            raise ValueError("Frozen Slurm profile changed; submit a fresh workflow")
        if cw.source_hash() != workflow["source_sha256"]:
            raise ValueError("Execution source changed; submit a fresh workflow")
    return workflow


def prepare(args):
    site = load_profile(root=ROOT)
    run_id = pw.identifier(args.run_id or "fedora-roadmap-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    base = cw.inside(ROOT / "runs" / run_id)
    if base.exists():
        raise ValueError("Run directory exists; preserve it and choose a fresh --run-id")
    profile = "smoke" if args.smoke else "development" if args.development else "full"
    resources = validate_resources(stage_resources(profile))
    return validate({"schema_version": 1, "kind": "desktop-slurm-roadmap", "run_id": run_id,
        "root": str(ROOT.resolve()), "run_dir": str(base), "created_at": cw.now(),
        "execution_mode": "desktop-slurm", "profile": profile,
        "protocol_path": str(base / "protocol.json"), "protocol_sha256": pw.canonical_hash(declaration(profile)),
        "slurm_profile": site, "slurm_profile_path": str(base / "slurm-profile.json"),
        "slurm_profile_sha256": pw.canonical_hash(site), "torch_version": site["torch_version"],
        "source_sha256": cw.source_hash(), "source_tree_sha256": cw.source_hash(metadata=True),
        "resources": {stage: {**resources[stage], "partition": site["cpu_partition" if stage in CPU_STAGES else "gpu_partition"]}
                      for stage in STAGES}})


def comment(workflow, stage):
    return f"tdn-roadmap:{workflow['run_id']}:{stage}:{workflow['slurm_profile_sha256']}"


def scheduler_args(workflow, stage, dependency=None):
    if stage not in STAGES:
        raise ValueError("Unknown roadmap stage")
    base, site, resource = Path(workflow["run_dir"]), workflow["slurm_profile"], workflow["resources"][stage]
    args = ["sbatch", "--parsable", f"--partition={resource['partition']}", "--nodes=1", "--ntasks=1",
        f"--job-name=tdn-roadmap-{stage}", f"--comment={comment(workflow, stage)}",
        f"--cpus-per-task={resource['cpus']}", f"--mem={resource['mem_gib']}G", f"--time={resource['walltime']}",
        "--signal=USR1@120", "--export=ALL", f"--chdir={ROOT}", "--open-mode=append", "--kill-on-invalid-dep=yes",
        f"--output={base / 'logs' / (stage + '-%j.out')}", f"--error={base / 'logs' / (stage + '-%j.err')}",
        f"--gres={site['gpu_gres'] if stage in GPU_STAGES else 'none'}"]
    if site.get("account"):
        args.append(f"--account={site['account']}")
    if dependency is not None:
        ids = dependency.split(":")
        count = STAGES.index(stage)
        valid_count = 1 <= len(ids) - 1 <= count if stage == "report" else len(ids) == count + 1
        if not count or ids[0] != ("afterany" if stage == "report" else "afterok") or not valid_count or any(
                not pw.JOB.fullmatch(item) for item in ids[1:]) or len(set(ids[1:])) != len(ids) - 1:
            raise ValueError("Invalid fixed roadmap dependency")
        args.append(f"--dependency={dependency}")
    return args + [str(ROOT / "scripts" / "roadmap_worker.sh")]


def check_capacity(workflow):
    """Check configured scheduler capacity, rather than current free resources."""
    policies = {}
    for partition in dict.fromkeys(row["partition"] for row in workflow["resources"].values()):
        response = cw.command(["scontrol", "show", "partition", partition, "-o"])
        fields = dict(re.findall(r"(\w+)=([^\s]+)", response.stdout))
        if fields.get("PartitionName") != partition or fields.get("State") != "UP":
            raise ValueError(f"Configured partition {partition} is missing or not UP")
        resources = [row for row in workflow["resources"].values() if row["partition"] == partition]
        limit = fields.get("MaxTime", "UNLIMITED")
        if limit not in ("UNLIMITED", "INFINITE") and wall_seconds(limit) < max(wall_seconds(row["walltime"]) for row in resources):
            raise ValueError(f"Partition {partition} walltime is below the declared stage budget")
        nodes = cw.command(["sinfo", "-N", "-h", "-p", partition, "-o", "%N|%c|%m|%G"])
        candidates = []
        for line in nodes.stdout.strip().splitlines():
            values = line.strip().split("|")
            if len(values) != 4 or not values[1].isdigit() or not values[2].isdigit():
                raise ValueError("Cannot verify configured Slurm node CPU/memory capacity")
            candidates.append({"node": values[0], "cpus": int(values[1]), "mem_mib": int(values[2]), "gres": values[3]})
        if not candidates:
            raise ValueError(f"Partition {partition} has no configured nodes")
        for stage, resource in workflow["resources"].items():
            if resource["partition"] != partition:
                continue
            eligible = [node for node in candidates if node["cpus"] >= resource["cpus"] and
                        node["mem_mib"] >= resource["mem_gib"] * 1024 and
                        (stage in CPU_STAGES or re.search(r"(?:^|,)gpu:(?:[^,:]+:)?[1-9][0-9]*(?:\([^)]*\))?(?:,|$)", node["gres"]))]
            if not eligible:
                raise ValueError(f"{stage}: declared CPU/memory/GPU resources exceed this partition's node capacity")
        policies[partition] = {"partition": fields, "nodes": candidates}
    return policies


def clean_environment():
    return {key: value for key, value in os.environ.items()
            if not key.startswith(("SBATCH_", "SRUN_", "TDN_PREMIX_", "TDN_CONSISTENCY_", "TDN_AGENDA_", "TDN_ROADMAP_")) and
            key not in ("CARC_ACCOUNT", "TORCH_CUDA_ARCH_LIST", "TDN_REQUIRE_GPU_TESTS", "TDN_TOWER_DIR")}


def submission_environment(workflow, stage):
    env = clean_environment()
    env.update(TDN_EXECUTION_MODE="desktop-slurm", TDN_PROJECT_ROOT=str(ROOT), TDN_REPO_ROOT=str(ROOT),
        TDN_SLURM_CONFIG=workflow["slurm_profile_path"], TDN_ROADMAP_WORKFLOW=str(Path(workflow["run_dir"]) / MANIFEST),
        TDN_ROADMAP_STAGE=stage, TDN_FEDORA_GPU_GRES=workflow["slurm_profile"]["gpu_gres"],
        TDN_ROADMAP_CPUS=str(workflow["resources"][stage]["cpus"]), TORCH_VERSION=workflow["torch_version"])
    return env


def start(args):
    workflow = prepare(args)
    print(f"Fedora roadmap workflow: {workflow['run_id']} ({workflow['profile']})\nRun: {workflow['run_dir']}")
    print("Stages run sequentially: " + " -> ".join(STAGES))
    print("One desktop GPU per learned stage; prior science is sealed. Final report runs after any scientific outcome. No pending-job cap or automatic resubmission.")
    for stage in STAGES:
        print(shlex.join(scheduler_args(workflow, stage)))
    if args.command == "plan":
        print("PLAN ONLY: no writes, scheduler calls or numerical work.")
        return workflow
    fw.controller_policy()
    policies = check_capacity(workflow)
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
            dependency = (("afterany:" if stage == "report" else "afterok:") + ":".join(row["job_id"] for row in jobs)) if jobs else None
            try:
                load(path)
                env = submission_environment(workflow, stage)
                result = cw.command(scheduler_args(workflow, stage, dependency), env=env)
                match = re.fullmatch(r"([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?", result.stdout.strip())
                if not match or any(row["job_id"] == match.group(1) for row in jobs):
                    raise ValueError("sbatch returned an ambiguous or duplicate ID; inspect Slurm before retrying")
                jobs.append({"stage": stage, "job_id": match.group(1), "dependency": dependency, "submitted_at": cw.now()})
                cw.atomic_json(base / "jobs.json", jobs)
                print(f"TDN_ROADMAP_{stage.upper()}_JOB_ID={match.group(1)}", flush=True)
            except BaseException as exc:
                failure = {"status": "FAILED", "stage": stage,
                    "error": str(exc), "submitted_jobs": jobs, "updated_at": cw.now(),
                    "note": "Existing submissions preserved; no scientific retry or cancellation."}
                # A distinct aggregation job is useful even if scientific
                # submission fails. It reports missing stages as NA; it does
                # not resubmit, replace, or reinterpret the failed job.
                if stage != "report" and not isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    try:
                        load(path)
                        report_dependency = "afterany:" + ":".join(row["job_id"] for row in jobs) if jobs else None
                        result = cw.command(scheduler_args(workflow, "report", report_dependency),
                                            env=submission_environment(workflow, "report"))
                        match = re.fullmatch(r"([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?", result.stdout.strip())
                        if not match or any(row["job_id"] == match.group(1) for row in jobs):
                            raise ValueError("Aggregation submission returned an ambiguous or duplicate ID")
                        jobs.append({"stage": "report", "job_id": match.group(1), "dependency": report_dependency,
                                     "submitted_at": cw.now(), "partial_submission_aggregation": True})
                        cw.atomic_json(base / "jobs.json", jobs)
                        failure["aggregation_job_id"] = match.group(1)
                        print(f"TDN_ROADMAP_REPORT_JOB_ID={match.group(1)} (partial submission aggregation)", flush=True)
                    except Exception as report_error:
                        failure["aggregation_submission_error"] = str(report_error)
                cw.atomic_json(base / "state" / "submission.json", failure)
                raise
        cw.atomic_json(base / "state" / "submission.json", {"status": "COMPLETED", "updated_at": cw.now()})
    print(f"Monitor: bash scripts/fedora_roadmap.sh status {workflow['run_id']}")
    return workflow


def stage_cli():
    spec = importlib.util.spec_from_file_location("roadmap_stage_cli", ROOT / "scripts" / "roadmap.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_stage(workflow, stage):
    base = cw.inside(Path(workflow["run_dir"]) / stage)
    verified = stage_cli().verify_execution(base, scientific_protocol(workflow["profile"]),
        source_tree_sha256=workflow["source_tree_sha256"])
    execution = verified["execution"]
    expected = {"stage": stage, "profile": workflow["profile"], "execution_mode": "desktop-slurm",
                "device": "cpu" if stage in CPU_STAGES else "cuda", "slurm_profile_sha256": workflow["slurm_profile_sha256"],
                "workflow_protocol_sha256": workflow["protocol_sha256"]}
    if any(execution.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{stage}: execution provenance differs from the frozen workflow")
    summary = cw.read_json(base / "summary.json")
    if summary.get("status") != "COMPLETED":
        raise ValueError(f"{stage}: experiment did not complete")
    if stage == "audit" and summary.get("correctness_failures", 0) != 0:
        raise ValueError("Structural correctness failures forbid successor stages")
    # A scientific BAD/NA is a completed outcome, not a scheduler failure.
    return verified


def verify_prerequisites(workflow, software, stage):
    hashes = {}
    for prerequisite in STAGES[:STAGES.index(stage)]:
        record = cw.read_json(pw.state_path(workflow, prerequisite))
        expected = {"status": "COMPLETED", "exit_code": 0, "source_sha256": workflow["source_sha256"],
                    "protocol_sha256": workflow["protocol_sha256"]}
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError(f"{prerequisite}: worker prerequisite did not complete")
        software_path = Path(workflow["run_dir"]) / "state" / f"{prerequisite}-software.json"
        if record.get("software_sha256") != cw.digest(software_path) or cw.read_json(software_path) != software:
            raise ValueError(f"{prerequisite}: prerequisite software differs")
        verify_stage(workflow, prerequisite)
        hashes[prerequisite] = cw.digest(Path(workflow["run_dir"]) / prerequisite / "workflow-seal.json")
    return hashes


def verify_software(workflow, software):
    if cw.read_json(Path(workflow["run_dir"]) / "state" / "submission-software.json") != software:
        raise ValueError("Software differs from the frozen submission environment")


def memory_mib(value):
    match = re.fullmatch(r"([0-9]+)([KMGT]?)", value)
    if not match:
        raise ValueError("Cannot verify actual allocated host memory")
    amount = int(match.group(1))
    return amount * {"": 1, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024**2}[match.group(2)]


def verify_allocation(workflow, stage):
    if stage not in STAGES:
        raise ValueError("Unknown roadmap stage")
    site, resource = workflow["slurm_profile"], workflow["resources"][stage]
    if load_profile(root=ROOT) != site:
        raise ValueError("Active Slurm profile differs from this frozen workflow")
    allocation = runtime_allocation("cpu" if stage in CPU_STAGES else "cuda", profile=site, root=ROOT)
    job, fields = allocation["job_id"], allocation["job"]
    expected = {"JobId": job, "JobName": f"tdn-roadmap-{stage}", "Comment": comment(workflow, stage),
                "JobState": "RUNNING", "Partition": resource["partition"], "WorkDir": str(ROOT)}
    if any(fields.get(key) != value for key, value in expected.items()) or fields.get("UserId", "").split("(")[0] != site["user"]:
        raise ValueError("Allocation is not this roadmap workflow's running stage")
    tres = dict(item.split("=", 1) for item in fields.get("AllocTRES", "").split(",") if "=" in item)
    if tres.get("cpu") != str(resource["cpus"]) or memory_mib(tres.get("mem", "")) != resource["mem_gib"] * 1024:
        raise ValueError("Actual allocated CPU/host-memory resources differ from the frozen roadmap budget")
    if fields.get("TimeLimit") is None or wall_seconds(fields["TimeLimit"]) != wall_seconds(resource["walltime"]):
        raise ValueError("Actual allocation walltime differs from the frozen roadmap budget")
    step_tres = dict(item.split("=", 1) for item in allocation["step"].get("TRES", "").split(",") if "=" in item)
    if step_tres.get("cpu") != str(resource["cpus"]):
        raise ValueError("Actual task CPU count differs from this stage budget")
    jobs = cw.read_json(Path(workflow["run_dir"]) / "jobs.json")
    recorded = [row for row in jobs if row.get("stage") == stage]
    if recorded and (len(recorded) != 1 or recorded[0].get("job_id") != job):
        raise ValueError("Allocation differs from the workflow submission record")


def junit_path(workflow, stage):
    return cw.inside(Path(workflow["run_dir"]) / "reporter-tests" / stage / "tests.xml")


def check_gpu_tests(path):
    """Require every declared roadmap CUDA case; no skips or filtered suites."""
    from collections import Counter
    from xml.etree import ElementTree
    from tdn.analysis.roadmap.protocol import GPU_TEST_CASES
    path = cw.inside(path)
    pw.validate_gpu_junit(path)
    tree = ElementTree.parse(path).getroot()
    cases = tree.findall(".//testcase")
    identities = [(case.get("classname"), case.get("name")) for case in cases]
    expected = [("tests.test_roadmap_gpu", name) for name in GPU_TEST_CASES]
    if not expected or Counter(identities) != Counter(expected) or len(set(identities)) != len(identities):
        raise ValueError("Roadmap GPU readiness requires every unique mandatory declared CUDA case; filtered suites are invalid")
    if any(case.find(tag) is not None for case in cases for tag in ("failure", "error", "skipped")):
        raise ValueError("All mandatory roadmap GPU tests must pass without failures, errors or skips")
    suites = [tree] if tree.tag == "testsuite" else tree.findall(".//testsuite")
    if any(int(suite.get("tests", -1)) != len(suite.findall("testcase")) for suite in suites):
        raise ValueError("Roadmap GPU JUnit declared counts do not match the recorded cases")
    if tree.tag == "testsuites" and tree.get("tests") is not None and int(tree.get("tests")) != len(cases):
        raise ValueError("Roadmap GPU JUnit aggregate count differs")
    return {"validation": "mandatory_roadmap_gpu_suite", "test_cases": len(cases), "case_names": list(GPU_TEST_CASES)}


def worker_commands(workflow, stage):
    python = str(cw.inside(ROOT / ".venv" / "bin") / "python")
    base = Path(workflow["run_dir"])
    commands = []
    if stage == "audit" or stage in GPU_STAGES:
        tests = ([ROOT / "tests" / "test_roadmap_gpu.py"] if stage in GPU_STAGES else
                 sorted((ROOT / "tests").glob("test_roadmap*.py")) +
                 [ROOT / "tests" / "test_desktop_slurm_runtime.py", ROOT / "tests" / "test_fedora_workflow.py"])
        if not tests or any(not path.is_file() for path in tests):
            raise ValueError("Roadmap correctness tests are missing")
        if stage in GPU_STAGES:
            commands.append(("preflight", [python, str(ROOT / "scripts" / "fedora_gpu_preflight.py"),
                                           "--output", str(base / f"{stage}-gpu-preflight.json")]))
        commands.append(("gpu-tests" if stage in GPU_STAGES else "tests",
            [python, "-m", "pytest", "-q", "-m", "gpu" if stage in GPU_STAGES else "not gpu", *map(str, tests),
             "--basetemp", str(base / f"{stage}-pytest-work"), "-o", f"cache_dir={base / (stage + '-pytest-cache')}",
             "--junitxml", str(junit_path(workflow, stage))]))
        if stage in GPU_STAGES:
            commands.append(("check-gpu-tests", [python, str(ROOT / "scripts" / "roadmap_workflow.py"),
                "check-gpu-tests", "--junit", str(junit_path(workflow, stage))]))
    args = [python, str(ROOT / "scripts" / "roadmap.py"), "--stage", stage,
            "--profile", workflow["profile"], "--run-dir", str(base / stage),
            "--device", "cpu" if stage in CPU_STAGES else "cuda"]
    for prior in STAGES[:STAGES.index(stage)]:
        args += ["--prerequisite-dir", f"{prior}={base / prior}"]
    commands.append(("experiment", args))
    return commands


def begin_report(workflow, stage):
    base, site, resource = Path(workflow["run_dir"]), workflow["slurm_profile"], workflow["resources"][stage]
    job = os.environ["SLURM_JOB_ID"]
    resources = {"partition": resource["partition"], "nodes": 1, "cpus": resource["cpus"],
                 "gpus": int(stage in GPU_STAGES), "mem_bytes": resource["mem_gib"] * 1024**3,
                 "time_seconds": wall_seconds(resource["walltime"])}
    if site.get("account"):
        resources["account"] = site["account"]
    if stage in GPU_STAGES:
        resources["gpu_type"] = site["expected_gpu_name"]
    return cw.reporting_api().begin_report(base / stage, name=f"TDN/roadmap/{stage}", script="scripts/roadmap_worker.sh",
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
    return {"TDN_ROADMAP_PROTOCOL_SHA256": workflow["protocol_sha256"],
            "TDN_ROADMAP_WORKFLOW": str(Path(workflow["run_dir"]) / MANIFEST), "TDN_ROADMAP_STAGE": stage,
            "TDN_ROADMAP_CPUS": str(workflow["resources"][stage]["cpus"])}


def prepare_test_environment(env, name):
    for key in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
        env.pop(key, None)
    for key in tuple(env):
        if key.startswith(("TDN_ROADMAP_", "TDN_AGENDA_", "TDN_PREMIX_", "TDN_CONSISTENCY_")):
            env.pop(key)
    if name == "tests":
        for key in ("TDN_EXECUTION_MODE", "TDN_SLURM_CONFIG"):
            env.pop(key, None)


def worker(workflow, stage):
    backend = SimpleNamespace(**{name: globals()[name] for name in ("verify_allocation", "load", "begin_report",
        "worker_commands", "verify_stage", "prepare_test_environment", "workflow_environment", "finish_report", "verify_software")},
        gpu_junit_path=lambda workflow: junit_path(workflow, stage), workflow_filename=MANIFEST,
        workflow_label="roadmap", stage_variable="TDN_ROADMAP_STAGE", prerequisite_stages=STAGES[1:-1],
        verify_prerequisites=lambda workflow, software: verify_prerequisites(workflow, software, stage))
    try:
        return pw.worker(workflow, stage, backend=backend)
    except BaseException as error:
        # Allocation verification happens before the shared worker opens its
        # running-state record. Preserve those failures for final aggregation.
        path = pw.state_path(workflow, stage)
        if not path.exists():
            cw.atomic_json(path, {"status": "FAILED", "stage": "startup", "exit_code": 2,
                "error": str(error), "source_sha256": workflow["source_sha256"],
                "protocol_sha256": workflow["protocol_sha256"], "updated_at": cw.now(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_step_id": os.environ.get("SLURM_STEP_ID")})
        raise


def paths(workflow):
    base = Path(workflow["run_dir"])
    for filename in ("summary.txt", "mechanism_summary.json", "experiment_summary.csv", "experiment_index.json",
                     "review.csv", "review.md", "summary.json", "rows.jsonl"):
        path = base / "report" / filename
        if path.is_file():
            print(f"TDN_ROADMAP_{filename.replace('.', '_').upper()}={path}")
    for row in cw.read_json(Path(workflow["run_dir"]) / "jobs.json"):
        if row.get("stage") not in STAGES or not pw.JOB.fullmatch(row.get("job_id", "")):
            raise ValueError("Invalid stored roadmap scheduler job")
        for report in cw.tower_reports_for_job(workflow, row["job_id"]):
            print(f"{row['stage']} job {row['job_id']}\nTDN_TOWER_DIR={report}\nTDN_TOWER_METRICS={report / 'metrics.jsonl'}")


def status(workflow):
    base = Path(workflow["run_dir"])
    print(f"Fedora roadmap workflow {workflow['run_id']} ({workflow['profile']})\nRun: {base}")
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
            summary = cw.read_json(base / stage / "summary.json")
            print("  sealed artifacts: VERIFIED")
            if summary.get("scientific_outcome"):
                print(f"  scientific outcome: {summary['scientific_outcome']}")
    submission_path = base / "state" / "submission.json"
    if submission_path.exists() and (submission := cw.read_json(submission_path)).get("status") == "FAILED":
        print(f"Submission failed at {submission['stage']}: {submission['error']}")
    paths(workflow)


def validate_results(workflow):
    """Validate seals without mistaking a scientific BAD for artifact failure."""
    output = {"workflow": workflow["run_id"], "validation": "roadmap_science_and_execution_seals", "stages": {}}
    invalid = False
    for stage in STAGES:
        directory = Path(workflow["run_dir"]) / stage
        if not directory.exists():
            output["stages"][stage] = {"validation": "MISSING", "scientific_outcome": "NA"}
            invalid = True
            continue
        try:
            verify_stage(workflow, stage)
            summary = cw.read_json(directory / "summary.json")
            output["stages"][stage] = {"validation": "VERIFIED", "scientific_outcome": summary.get("scientific_outcome", "NA")}
        except (OSError, ValueError, KeyError, TypeError) as error:
            output["stages"][stage] = {"validation": "INVALID", "error": str(error), "scientific_outcome": "NA"}
            invalid = True
    print(json.dumps(output, indent=2))
    if invalid:
        raise ValueError("Some roadmap stages are missing or invalid; final aggregation preserves their NA evidence")
    return output


ACCOUNTING_FIELDS = ("JobIDRaw", "State", "ExitCode", "ElapsedRaw", "TotalCPU", "AllocTRES", "MaxRSS", "CPUTimeRAW")


def scheduler_accounting(workflow, *, enabled=True, runner=None):
    """Snapshot exact submitted jobs; unavailable accounting never invents cost."""
    base = cw.inside(workflow["run_dir"])
    jobs = cw.read_json(base / "jobs.json")
    if (not isinstance(jobs, list) or any(not isinstance(row, dict) or row.get("stage") not in STAGES
            or not pw.JOB.fullmatch(row.get("job_id", "")) for row in jobs)
            or len({row["job_id"] for row in jobs}) != len(jobs)
            or len({row["stage"] for row in jobs}) != len(jobs)):
        raise ValueError("Accounting requires unique exact workflow stage/job identities")
    by_job = {row["job_id"]: row["stage"] for row in jobs}
    command = ["sacct", "--noheader", "--parsable2", "--jobs", ",".join(by_job),
               "--format=" + ",".join(ACCOUNTING_FIELDS)]
    result = {"schema": "tdn.scheduler-accounting/v1", "workflow_id": workflow["run_id"],
        "collected_at": cw.now(), "requested_job_ids": list(by_job), "records": [],
        "status": "UNAVAILABLE", "command": command if by_job and enabled else None,
        "scope": "Slurm accounting for recorded workflow allocations and their exact task steps; allocation and step rows are never summed",
        "cost_scope": "Reported resource/time usage, distinct from scientific experiment timers; no monetary rate is assumed",
        "monetary_cost": None, "units": {"ElapsedRaw": "seconds", "CPUTimeRAW": "allocated CPU-seconds reported by Slurm",
            "TotalCPU": "Slurm CPU-time string", "MaxRSS": "Slurm memory string; an empty allocation value is unknown, not zero"}}
    if not enabled:
        result.update(status="DISABLED", reason="Collection accounting was explicitly disabled with --no-accounting")
    elif not by_job:
        result["reason"] = "No submitted scheduler jobs are recorded for this workflow"
    else:
        try:
            response = (runner or subprocess.run)(command, cwd=ROOT, check=False, text=True,
                capture_output=True, timeout=30)
            if response.returncode:
                raise ValueError(f"sacct exited {response.returncode}: {response.stderr.strip()[:2000]}")
            if len(response.stdout.encode()) > 1 << 20:
                raise ValueError("Accounting output exceeds its 1 MiB snapshot limit")
            records, seen = [], set()
            for line in response.stdout.splitlines():
                if not line.strip():
                    continue
                fields = line.strip().split("|")
                if len(fields) == len(ACCOUNTING_FIELDS) + 1 and fields[-1] == "":
                    fields.pop()
                if len(fields) != len(ACCOUNTING_FIELDS):
                    raise ValueError("Accounting returned malformed columns; no partial resource totals were inferred")
                row = dict(zip(ACCOUNTING_FIELDS, fields))
                identity = row["JobIDRaw"]
                allocation = identity.split(".", 1)[0]
                if allocation not in by_job or not re.fullmatch(r"[1-9][0-9]*(?:\.[A-Za-z0-9_-]+)?", identity):
                    raise ValueError("Accounting returned a record outside the frozen workflow jobs")
                if identity in seen:
                    raise ValueError("Accounting returned duplicate job or step identities")
                seen.add(identity)
                for key in ("ElapsedRaw", "CPUTimeRAW"):
                    if row[key] and not row[key].isdigit():
                        raise ValueError(f"Accounting returned an invalid {key} value")
                row.update(workflow_stage=by_job[allocation], allocation_job_id=allocation,
                    record_kind="allocation" if identity == allocation else "step",
                    elapsed_seconds=int(row["ElapsedRaw"]) if row["ElapsedRaw"] else None,
                    allocated_cpu_seconds=int(row["CPUTimeRAW"]) if row["CPUTimeRAW"] else None,
                    missing_fields=[key for key in ACCOUNTING_FIELDS if not row[key]])
                records.append(row)
            allocations = {row["allocation_job_id"] for row in records if row["record_kind"] == "allocation"}
            missing = [job for job in by_job if job not in allocations]
            result.update(records=records, missing_allocation_job_ids=missing,
                status="RECORDED" if not missing else "PARTIAL" if records else "UNAVAILABLE",
                reason="All requested allocation rows returned; fields may still be unavailable" if not missing
                       else "Some requested allocations are unavailable in sacct; absent fields remain unknown")
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            result.update(status="UNAVAILABLE", reason=f"{type(error).__name__}: {error}", records=[])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = base / "state" / "scheduler-accounting.json"
    cw.atomic_json(base / "state" / f"scheduler-accounting-{stamp}.json", result)
    cw.atomic_json(path, result)
    print(f"Scheduler accounting: {result['status']} — {path}")
    if result["status"] != "RECORDED":
        print("Accounting note: " + result["reason"])
    return result


def collect(workflow, *, part_bytes=ARCHIVE_PART_BYTES, accounting=True):
    """Keep one complete archive and provide verified upload-sized parts if needed."""
    if type(part_bytes) is not int or not 0 < part_bytes <= ARCHIVE_PART_BYTES:
        raise ValueError("Archive part size must be between 1 byte and 28 MiB")
    base = cw.inside(workflow["run_dir"])
    scheduler_accounting(workflow, enabled=accounting)
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
    index = {"schema_version": 1, "archive": destination.name, "bytes": destination.stat().st_size,
             "sha256": cw.digest(destination), "member_count": len(members), "parts": [],
             "reassemble": "Concatenate parts in their listed order; verify the complete SHA-256 before extracting."}
    if destination.stat().st_size > part_bytes:
        with destination.open("rb") as stream:
            ordinal = 1
            while block := stream.read(part_bytes):
                part = destination.with_name(destination.name + f".part{ordinal:03d}")
                with part.open("xb") as output:
                    output.write(block)
                index["parts"].append({"path": part.name, "bytes": len(block), "sha256": hashlib.sha256(block).hexdigest()})
                ordinal += 1
    index_path = destination.with_name(destination.name + ".index.json")
    cw.atomic_json(index_path, index)
    print(f"Review archive: {destination}\nReview index: {index_path}")
    print("Includes all stages, checkpoints, references, metrics, logs and failures; excludes pytest temporary directories.")
    if index["parts"]:
        print(f"Archive exceeds upload limit; upload the index and all {len(index['parts'])} parts (each at most 28 MiB).")
        for part in index["parts"]:
            print(base.parent / part["path"])
    return destination


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("plan", "run"):
        item = commands.add_parser(name)
        item.add_argument("--run-id")
        profile = item.add_mutually_exclusive_group()
        profile.add_argument("--smoke", action="store_true", help="Tiny integration profile; not scientific confirmation")
        profile.add_argument("--development", action="store_true", help="Bounded development cohort, no fresh confirmation")
        profile.add_argument("--full", action="store_true", help="Complete bounded native experiment (the default)")
    for name in ("status", "logs", "collect", "paths", "validate"):
        item = commands.add_parser(name)
        item.add_argument("run", nargs="?", default="latest")
        if name == "logs":
            item.add_argument("--lines", type=int, default=200)
        elif name == "collect":
            item.add_argument("--no-accounting", action="store_true", help="Skip optional exact-job sacct metadata collection")
    item = commands.add_parser("worker", help=argparse.SUPPRESS)
    item.add_argument("--workflow", type=Path, required=True)
    item.add_argument("--stage", choices=STAGES, required=True)
    item = commands.add_parser("check-gpu-tests", help=argparse.SUPPRESS)
    item.add_argument("--junit", type=Path, required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command in ("plan", "run"):
            start(args)
        elif args.command == "worker":
            return worker(load(args.workflow, verify=False), args.stage)
        elif args.command == "check-gpu-tests":
            print(json.dumps(check_gpu_tests(args.junit), sort_keys=True))
        else:
            workflow = load(workflow_path(args.run), verify=False)
            if args.command == "logs":
                pw.logs(workflow, args.lines)
            elif args.command == "validate":
                validate_results(workflow)
            elif args.command == "collect":
                collect(workflow, accounting=not args.no_accounting)
            else:
                globals()[args.command](workflow)
    except rw.WorkerFailure as exc:
        print(f"TDN roadmap: {exc}", file=sys.stderr)
        return exc.exit_code
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"TDN roadmap: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
