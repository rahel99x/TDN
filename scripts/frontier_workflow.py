#!/usr/bin/env python3
"""Nine-stage bounded Fedora research frontier; controllers do metadata work only."""
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
_spec = importlib.util.spec_from_file_location("frontier_fedora_helpers", ROOT / "scripts" / "fedora_workflow.py")
fw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fw)
pw, cw, rw = fw.pw, fw.cw, fw.rw
from tdn.runtime.desktop_slurm import load_profile, validate_profile, verify_allocation as runtime_allocation

STAGES = ("audit", "screen", "prepare", "train", "confirm_prepare", "confirm", "scaling", "policy", "report")
GPU_STAGES = ("train", "confirm", "scaling", "policy")
CPU_STAGES = tuple(stage for stage in STAGES if stage not in GPU_STAGES)
MANIFEST = "frontier-workflow.json"
POINTER = ".fedora-frontier-latest.json"
PROFILES = ("smoke", "development", "full")
SCHEDULER_MEMORY_CAP_MIB = 110000
ARCHIVE_PART_BYTES = 28 * 2**20


def scientific_protocol(profile):
    from tdn.analysis.frontier.protocol import build_protocol
    return build_protocol(profile)


def stage_resources(profile):
    """Scheduler limits are part of the immutable scientific declaration."""
    protocol = scientific_protocol(profile)
    return {stage: {key: protocol["budgets"][stage][key] for key in ("cpus", "mem_gib", "walltime")} for stage in STAGES}


def declaration(profile, version=2):
    value = {"version": version, "benchmark_suite": "frontier", "profile": profile,
            "scientific_protocol": scientific_protocol(profile), "stages": list(STAGES),
            "execution_mode": "desktop-slurm",
            "dependencies": {stage: list(STAGES[:index]) for index, stage in enumerate(STAGES)},
            "device": {stage: "cpu" if stage in CPU_STAGES else "cuda" for stage in STAGES},
            "resources_per_stage": stage_resources(profile),
            "scheduler_memory_cap_mib": SCHEDULER_MEMORY_CAP_MIB,
            "scope": "bounded preregistered frontier; no automatic expansion or pending-job cap",
            "report_dependencies": "afterany across all scientific stages",
            "scientific_bad_is_scheduler_failure": False}
    if version == 2:
        shards = confirmation_shards(profile)
        value.update(confirmation_shards=shards,
            physical_stages=list(STAGES[:5]) + [row["shard_id"] for row in shards] + list(STAGES[5:]),
            confirmation_execution="Whole parent groups; fixed all-model paired rounds per shard; sealed exact-cohort aggregation",
            physical_dependencies="Sequential afterok; final report afterany all newly submitted jobs")
    elif version != 1:
        raise ValueError("Unsupported frontier execution declaration")
    return value


def confirmation_shards(profile):
    from tdn.analysis.frontier.partition import plan_shards
    return plan_shards(scientific_protocol(profile))


def execution_version(workflow):
    return workflow.get("execution_version", 1)


def physical_stages(workflow):
    if execution_version(workflow) == 1:
        return STAGES
    return STAGES[:5] + tuple(row["shard_id"] for row in confirmation_shards(workflow["profile"])) + STAGES[5:]


def logical_stage(stage):
    return "confirm" if re.fullmatch(r"confirm-part-[0-9]{3}", stage) else stage


def is_gpu_stage(stage):
    return logical_stage(stage) in GPU_STAGES


def resource_for(workflow, stage):
    if stage not in physical_stages(workflow):
        raise ValueError("Unknown frontier physical stage")
    return workflow["resources"][logical_stage(stage)]


def recovery_bridge(workflow):
    recovery = workflow.get("recovery")
    if recovery is None:
        return None
    path = cw.inside(recovery["manifest_path"])
    if path != Path(workflow["run_dir"]) / "recovery.json" or cw.digest(path) != recovery["sha256"]:
        raise ValueError("Frozen frontier recovery bridge changed")
    bridge = cw.read_json(path)
    if bridge.get("stage_paths") != recovery.get("stage_paths"):
        raise ValueError("Recovery stage mapping differs from its frozen bridge")
    return bridge


def stage_path(workflow, stage):
    if stage not in physical_stages(workflow):
        raise ValueError("Unknown frontier stage path")
    bridge = recovery_bridge(workflow)
    if bridge and stage in bridge["stage_paths"]:
        return cw.inside(bridge["stage_paths"][stage])
    return cw.inside(Path(workflow["run_dir"]) / stage)


def pending_stages(workflow):
    inherited = set(workflow.get("recovery", {}).get("stage_paths", {}))
    return tuple(stage for stage in physical_stages(workflow) if stage not in inherited)


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
        raise ValueError("Frontier requires exactly nine bounded stages")
    for stage, resource in resources.items():
        if not isinstance(resource, dict) or set(resource) != {"cpus", "mem_gib", "walltime"}:
            raise ValueError(f"{stage}: invalid allocation resource declaration")
        if type(resource["cpus"]) is not int or not 4 <= resource["cpus"] <= 8:
            raise ValueError("Frontier CPU requests must fit the eight physical desktop cores")
        if stage in GPU_STAGES and resource["cpus"] != 4:
            raise ValueError("Frontier GPU stages require exactly four allocated CPUs")
        if type(resource["mem_gib"]) is not int or resource["mem_gib"] not in (32, 48) or resource["mem_gib"] * 1024 > SCHEDULER_MEMORY_CAP_MIB:
            raise ValueError("Frontier memory request exceeds Slurm's 110000 MiB host limit")
        if not 180 <= wall_seconds(resource["walltime"]) <= 2700:
            raise ValueError("Frontier allocations must stay within the declared 45-minute hard maximum")
    return resources


def validate(workflow):
    if workflow.get("schema_version") != 1 or workflow.get("kind") != "desktop-slurm-frontier":
        raise ValueError("Not a supported Fedora frontier workflow")
    run_id = pw.identifier(workflow["run_id"])
    base = cw.inside(workflow["run_dir"])
    if base != cw.inside(ROOT / "runs" / run_id) or cw.inside(workflow["root"]) != ROOT.resolve():
        raise ValueError("Frontier workflow root/layout changed")
    profile = workflow.get("profile")
    if profile not in PROFILES or workflow.get("execution_mode") != "desktop-slurm":
        raise ValueError("Unknown frontier profile or execution mode")
    if cw.inside(workflow["protocol_path"]) != base / "protocol.json":
        raise ValueError("Frozen protocol must remain within this workflow")
    if workflow.get("protocol_sha256") != pw.canonical_hash(declaration(profile, execution_version(workflow))):
        raise ValueError("Frontier protocol declaration changed")
    if execution_version(workflow) not in (1, 2):
        raise ValueError("Unsupported frontier execution version")
    recovery = workflow.get("recovery")
    if recovery is not None:
        if execution_version(workflow) != 2 or set(recovery) != {"manifest_path", "sha256", "stage_paths"}:
            raise ValueError("Invalid frontier recovery declaration")
        if not pw.SHA.fullmatch(recovery.get("sha256", "")) or not isinstance(recovery["stage_paths"], dict):
            raise ValueError("Invalid frontier recovery fingerprint or stage map")
        allowed = set(STAGES[:5]) | {stage for stage in physical_stages(workflow) if stage.startswith("confirm-part-")}
        if not set(STAGES[:5]) <= set(recovery["stage_paths"]) <= allowed:
            raise ValueError("Recovery must reuse the five sealed prerequisites and only complete confirmation parts")
    site = validate_profile(workflow["slurm_profile"], root=ROOT)
    if workflow.get("slurm_profile_sha256") != pw.canonical_hash(site):
        raise ValueError("Frozen Slurm profile fingerprint differs")
    if cw.inside(workflow["slurm_profile_path"]) != base / "slurm-profile.json":
        raise ValueError("Frozen Slurm profile must remain within this workflow")
    if workflow.get("torch_version") != site["torch_version"]:
        raise ValueError("Workflow Torch release differs from its Slurm profile")
    expected_resources = validate_resources(stage_resources(profile))
    if set(workflow.get("resources", {})) != set(STAGES):
        raise ValueError("Frontier requires exactly nine bounded stages")
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
        raise ValueError("Frontier manifest is outside its run directory")
    if verify:
        if cw.read_json(workflow["protocol_path"]) != declaration(workflow["profile"], execution_version(workflow)):
            raise ValueError("Frozen frontier declaration changed; submit a fresh workflow")
        if load_profile(workflow["slurm_profile_path"], root=ROOT) != workflow["slurm_profile"]:
            raise ValueError("Frozen Slurm profile changed; submit a fresh workflow")
        if cw.source_hash() != workflow["source_sha256"]:
            raise ValueError("Execution source changed; submit a fresh workflow")
    return workflow


def prepare(args):
    site = load_profile(root=ROOT)
    run_id = pw.identifier(args.run_id or "fedora-frontier-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    base = cw.inside(ROOT / "runs" / run_id)
    if base.exists():
        raise ValueError("Run directory exists; preserve it and choose a fresh --run-id")
    profile = "smoke" if args.smoke else "development" if args.development else "full"
    resources = validate_resources(stage_resources(profile))
    return validate({"schema_version": 1, "execution_version": 2, "kind": "desktop-slurm-frontier", "run_id": run_id,
        "root": str(ROOT.resolve()), "run_dir": str(base), "created_at": cw.now(),
        "execution_mode": "desktop-slurm", "profile": profile,
        "protocol_path": str(base / "protocol.json"), "protocol_sha256": pw.canonical_hash(declaration(profile)),
        "slurm_profile": site, "slurm_profile_path": str(base / "slurm-profile.json"),
        "slurm_profile_sha256": pw.canonical_hash(site), "torch_version": site["torch_version"],
        "source_sha256": cw.source_hash(), "source_tree_sha256": cw.source_hash(metadata=True),
        "resources": {stage: {**resources[stage], "partition": site["cpu_partition" if stage in CPU_STAGES else "gpu_partition"]}
                      for stage in STAGES}})


def comment(workflow, stage):
    return f"tdn-frontier:{workflow['run_id']}:{stage}:{workflow['slurm_profile_sha256']}"


def scheduler_args(workflow, stage, dependency=None):
    base, site, resource = Path(workflow["run_dir"]), workflow["slurm_profile"], resource_for(workflow, stage)
    args = ["sbatch", "--parsable", f"--partition={resource['partition']}", "--nodes=1", "--ntasks=1",
        f"--job-name=tdn-frontier-{stage}", f"--comment={comment(workflow, stage)}",
        f"--cpus-per-task={resource['cpus']}", f"--mem={resource['mem_gib']}G", f"--time={resource['walltime']}",
        "--signal=USR1@120", "--export=ALL", f"--chdir={ROOT}", "--open-mode=append", "--kill-on-invalid-dep=yes",
        f"--output={base / 'logs' / (stage + '-%j.out')}", f"--error={base / 'logs' / (stage + '-%j.err')}",
        f"--gres={site['gpu_gres'] if is_gpu_stage(stage) else 'none'}"]
    if site.get("account"):
        args.append(f"--account={site['account']}")
    if dependency is not None:
        ids = dependency.split(":")
        count = pending_stages(workflow).index(stage)
        valid_count = 1 <= len(ids) - 1 <= count if stage == "report" else len(ids) == count + 1
        if not count or ids[0] != ("afterany" if stage == "report" else "afterok") or not valid_count or any(
                not pw.JOB.fullmatch(item) for item in ids[1:]) or len(set(ids[1:])) != len(ids) - 1:
            raise ValueError("Invalid fixed frontier dependency")
        args.append(f"--dependency={dependency}")
    return args + [str(ROOT / "scripts" / "frontier_worker.sh")]


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
            if not key.startswith(("SBATCH_", "SRUN_", "TDN_PREMIX_", "TDN_CONSISTENCY_", "TDN_AGENDA_", "TDN_ROADMAP_", "TDN_FRONTIER_")) and
            key not in ("CARC_ACCOUNT", "TORCH_CUDA_ARCH_LIST", "TDN_REQUIRE_GPU_TESTS", "TDN_TOWER_DIR")}


def submission_environment(workflow, stage):
    env = clean_environment()
    env.update(TDN_EXECUTION_MODE="desktop-slurm", TDN_PROJECT_ROOT=str(ROOT), TDN_REPO_ROOT=str(ROOT),
        TDN_SLURM_CONFIG=workflow["slurm_profile_path"], TDN_FRONTIER_WORKFLOW=str(Path(workflow["run_dir"]) / MANIFEST),
        TDN_FRONTIER_STAGE=stage, TDN_FEDORA_GPU_GRES=workflow["slurm_profile"]["gpu_gres"],
        TDN_FRONTIER_CPUS=str(resource_for(workflow, stage)["cpus"]), TORCH_VERSION=workflow["torch_version"])
    return env


def start(args):
    workflow = prepare(args)
    return submit(workflow, plan_only=args.command == "plan")


def submit(workflow, *, plan_only=False, bridge=None, software=None):
    print(f"Fedora frontier workflow: {workflow['run_id']} ({workflow['profile']})\nRun: {workflow['run_dir']}")
    print("Stages run sequentially: " + " -> ".join(pending_stages(workflow)))
    print("One desktop GPU per learned stage; prior science is sealed. Final report runs after any scientific outcome. No pending-job cap or automatic resubmission.")
    for stage in pending_stages(workflow):
        print(shlex.join(scheduler_args(workflow, stage)))
    if plan_only:
        print("PLAN ONLY: no writes, scheduler calls or numerical work.")
        return workflow
    fw.controller_policy()
    policies = check_capacity(workflow)
    with rw.acquire_venv_lock():
        actual_software = rw.software_report(workflow)
        if software is not None and software != actual_software:
            raise ValueError("Software changed while preparing recovery")
        software = actual_software
    with cw.controller_lock():
        base = Path(workflow["run_dir"])
        base.mkdir(parents=True, exist_ok=False)
        (base / "logs").mkdir()
        path = base / MANIFEST
        for output, data in ((path, workflow), (Path(workflow["protocol_path"]), declaration(workflow["profile"], execution_version(workflow))),
                             (Path(workflow["slurm_profile_path"]), workflow["slurm_profile"])):
            cw.atomic_json(output, data)
            output.chmod(0o444)
        if bridge is not None:
            cw.atomic_json(base / "recovery.json", bridge)
            (base / "recovery.json").chmod(0o444)
            recovery_bridge(workflow)
        cw.atomic_json(base / "state" / "slurm-policy.json", policies)
        cw.atomic_json(base / "state" / "submission-software.json", software)
        jobs = []
        cw.atomic_json(base / "jobs.json", jobs)
        cw.atomic_json(ROOT / "runs" / POINTER, {"run_id": workflow["run_id"]})
        for stage in pending_stages(workflow):
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
                print(f"TDN_FRONTIER_{stage.upper().replace(chr(45), chr(95))}_JOB_ID={match.group(1)}", flush=True)
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
                        print(f"TDN_FRONTIER_REPORT_JOB_ID={match.group(1)} (partial submission aggregation)", flush=True)
                    except Exception as report_error:
                        failure["aggregation_submission_error"] = str(report_error)
                cw.atomic_json(base / "state" / "submission.json", failure)
                raise
        cw.atomic_json(base / "state" / "submission.json", {"status": "COMPLETED", "updated_at": cw.now()})
    print(f"Monitor: bash scripts/fedora_frontier.sh status {workflow['run_id']}")
    return workflow


def execution_software():
    """Read execution metadata from this checkout's installed Python venv."""
    python = str(cw.inside(ROOT / ".venv" / "bin") / "python")
    return json.loads(cw.command([python, "-c",
        "import json; from tdn.runtime.metadata import software_metadata; print(json.dumps(software_metadata()))"]).stdout)


def require_terminal_origin(workflow):
    """Recovery must not overlap a live or unverified scheduler attempt."""
    jobs = cw.read_json(Path(workflow["run_dir"]) / "jobs.json")
    if (not isinstance(jobs, list) or not jobs or any(not isinstance(row, dict)
            or row.get("stage") not in physical_stages(workflow)
            or not pw.JOB.fullmatch(row.get("job_id", "")) for row in jobs)
            or len({row["job_id"] for row in jobs}) != len(jobs)
            or len({row["stage"] for row in jobs}) != len(jobs)):
        raise ValueError("Recovery requires unique recorded origin scheduler jobs")
    states = {}
    for row in jobs:
        state = cw.scheduler_state(row["job_id"])
        if state not in TERMINAL_JOB_STATES:
            raise ValueError(f"Recovery origin {row['stage']} job {row['job_id']} is {state}; "
                             "all origin jobs must have verified terminal states before recovery")
        states[row["job_id"]] = state
    return states


def recover(args):
    """Start a fresh, finite coordinator referencing verified immutable evidence."""
    origin_path = workflow_path(args.run)
    origin = load(origin_path, verify=False)
    options = SimpleNamespace(run_id=args.run_id, smoke=origin["profile"] == "smoke",
        development=origin["profile"] == "development")
    workflow = prepare(options)
    fw.controller_policy()
    require_terminal_origin(origin)
    from tdn.analysis.frontier.recovery import create_recovery
    with rw.acquire_venv_lock():
        software = rw.software_report(workflow)
        bridge = create_recovery(origin_path, scientific_protocol(workflow["profile"]),
            current_software=execution_software(), site=workflow["slurm_profile"], root=ROOT)
    serialized = json.dumps(bridge, indent=2, sort_keys=True, allow_nan=False) + "\n"
    workflow["recovery"] = {"manifest_path": str(Path(workflow["run_dir"]) / "recovery.json"),
        "sha256": hashlib.sha256(serialized.encode()).hexdigest(), "stage_paths": bridge["stage_paths"]}
    validate(workflow)
    print(f"Recovery origin: {origin['run_id']}; frozen inputs/checkpoints reused without retraining.")
    print("Unsealed or incomplete confirmation measurements are excluded; only complete compatible parts can be reused.")
    return submit(workflow, plan_only=args.plan, bridge=bridge, software=software)


def stage_cli():
    spec = importlib.util.spec_from_file_location("frontier_stage_cli", ROOT / "scripts" / "frontier.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_recovery_bridge(workflow, *, software=None):
    bridge = recovery_bridge(workflow)
    if bridge is None:
        return {}
    from tdn.analysis.frontier.recovery import verify_recovery
    return verify_recovery(bridge, scientific_protocol(workflow["profile"]),
        current_software=software or bridge["target_software"], site=workflow["slurm_profile"], root=ROOT)


def verify_stage(workflow, stage):
    base = stage_path(workflow, stage)
    inherited = stage in workflow.get("recovery", {}).get("stage_paths", {})
    if inherited:
        mapping = verify_recovery_bridge(workflow)
        if Path(mapping[stage]) != base:
            raise ValueError("Inherited stage differs from the verified recovery bridge")
        execution = cw.read_json(base / "execution.json")
        source_sha = execution["software"]["source_tree_sha256"]
        workflow_sha = execution["workflow_protocol_sha256"]
    else:
        source_sha, workflow_sha = workflow["source_tree_sha256"], workflow["protocol_sha256"]
    descriptor = (next(row for row in confirmation_shards(workflow["profile"]) if row["shard_id"] == stage)
                  if logical_stage(stage) != stage else None)
    verified = stage_cli().verify_execution(base, scientific_protocol(workflow["profile"]),
        source_tree_sha256=source_sha, confirmation_partition=descriptor)
    execution = verified["execution"]
    expected = {"stage": logical_stage(stage), "profile": workflow["profile"], "execution_mode": "desktop-slurm",
                "device": "cuda" if is_gpu_stage(stage) else "cpu", "slurm_profile_sha256": workflow["slurm_profile_sha256"],
                "workflow_protocol_sha256": workflow_sha}
    if any(execution.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{stage}: execution provenance differs from the frozen workflow")
    if logical_stage(stage) != stage:
        if execution.get("confirmation_partition") != descriptor:
            raise ValueError("Confirmation part execution differs from its exact frozen parent group")
    elif stage == "confirm" and execution.get("confirmation_partition") is not None:
        raise ValueError("A partial confirmation cannot stand in for a complete confirmation stage")
    summary = cw.read_json(base / "summary.json")
    if summary.get("status") != "COMPLETED":
        raise ValueError(f"{stage}: experiment did not complete")
    if stage == "audit" and summary.get("correctness_failures", 0) != 0:
        raise ValueError("Structural correctness failures forbid successor stages")
    return verified


def verify_prerequisites(workflow, software, stage):
    hashes = {}
    inherited = verify_recovery_bridge(workflow, software=execution_software()) if workflow.get("recovery") else {}
    for prerequisite in STAGES[:STAGES.index(logical_stage(stage))]:
        if prerequisite not in inherited:
            record = cw.read_json(pw.state_path(workflow, prerequisite))
            expected = {"status": "COMPLETED", "exit_code": 0, "source_sha256": workflow["source_sha256"],
                        "protocol_sha256": workflow["protocol_sha256"]}
            if any(record.get(key) != value for key, value in expected.items()):
                raise ValueError(f"{prerequisite}: worker prerequisite did not complete")
            software_path = Path(workflow["run_dir"]) / "state" / f"{prerequisite}-software.json"
            if record.get("software_sha256") != cw.digest(software_path) or cw.read_json(software_path) != software:
                raise ValueError(f"{prerequisite}: prerequisite software differs")
        verify_stage(workflow, prerequisite)
        hashes[prerequisite] = cw.digest(stage_path(workflow, prerequisite) / "workflow-seal.json")
    if stage == "confirm" and execution_version(workflow) == 2:
        for descriptor in confirmation_shards(workflow["profile"]):
            part = descriptor["shard_id"]
            verify_stage(workflow, part)
            if part not in inherited:
                record = cw.read_json(pw.state_path(workflow, part))
                expected = {"status": "COMPLETED", "exit_code": 0, "source_sha256": workflow["source_sha256"],
                            "protocol_sha256": workflow["protocol_sha256"]}
                if any(record.get(key) != value for key, value in expected.items()):
                    raise ValueError(f"{part}: confirmation worker did not complete under this workflow")
                software_path = Path(workflow["run_dir"]) / "state" / f"{part}-software.json"
                if record.get("software_sha256") != cw.digest(software_path) or cw.read_json(software_path) != software:
                    raise ValueError(f"{part}: confirmation software differs")
            hashes[part] = cw.digest(stage_path(workflow, part) / "workflow-seal.json")
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
    site, resource = workflow["slurm_profile"], resource_for(workflow, stage)
    if load_profile(root=ROOT) != site:
        raise ValueError("Active Slurm profile differs from this frozen workflow")
    allocation = runtime_allocation("cuda" if is_gpu_stage(stage) else "cpu", profile=site, root=ROOT)
    job, fields = allocation["job_id"], allocation["job"]
    expected = {"JobId": job, "JobName": f"tdn-frontier-{stage}", "Comment": comment(workflow, stage),
                "JobState": "RUNNING", "Partition": resource["partition"], "WorkDir": str(ROOT)}
    if any(fields.get(key) != value for key, value in expected.items()) or fields.get("UserId", "").split("(")[0] != site["user"]:
        raise ValueError("Allocation is not this frontier workflow's running stage")
    tres = dict(item.split("=", 1) for item in fields.get("AllocTRES", "").split(",") if "=" in item)
    if tres.get("cpu") != str(resource["cpus"]) or memory_mib(tres.get("mem", "")) != resource["mem_gib"] * 1024:
        raise ValueError("Actual allocated CPU/host-memory resources differ from the frozen frontier budget")
    if fields.get("TimeLimit") is None or wall_seconds(fields["TimeLimit"]) != wall_seconds(resource["walltime"]):
        raise ValueError("Actual allocation walltime differs from the frozen frontier budget")
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
    """Require every declared frontier CUDA case; no skips or filtered suites."""
    from collections import Counter
    from xml.etree import ElementTree
    from tdn.analysis.frontier.protocol import GPU_TEST_CASES
    path = cw.inside(path)
    pw.validate_gpu_junit(path)
    tree = ElementTree.parse(path).getroot()
    cases = tree.findall(".//testcase")
    identities = [(case.get("classname"), case.get("name")) for case in cases]
    expected = [("tests.test_frontier_gpu", name) for name in GPU_TEST_CASES]
    if not expected or Counter(identities) != Counter(expected) or len(set(identities)) != len(identities):
        raise ValueError("Frontier GPU readiness requires every unique mandatory declared CUDA case; filtered suites are invalid")
    if any(case.find(tag) is not None for case in cases for tag in ("failure", "error", "skipped")):
        raise ValueError("All mandatory frontier GPU tests must pass without failures, errors or skips")
    suites = [tree] if tree.tag == "testsuite" else tree.findall(".//testsuite")
    if any(int(suite.get("tests", -1)) != len(suite.findall("testcase")) for suite in suites):
        raise ValueError("Frontier GPU JUnit declared counts do not match the recorded cases")
    if tree.tag == "testsuites" and tree.get("tests") is not None and int(tree.get("tests")) != len(cases):
        raise ValueError("Frontier GPU JUnit aggregate count differs")
    return {"validation": "mandatory_frontier_gpu_suite", "test_cases": len(cases), "case_names": list(GPU_TEST_CASES)}


def worker_commands(workflow, stage):
    python = str(cw.inside(ROOT / ".venv" / "bin") / "python")
    base = Path(workflow["run_dir"])
    commands = []
    if stage == "report":
        commands.append(("accounting", [python, str(ROOT / "scripts" / "frontier_workflow.py"),
            "snapshot-accounting", "--workflow", str(base / MANIFEST)]))
    if stage == "audit" or is_gpu_stage(stage):
        tests = ([ROOT / "tests" / "test_frontier_gpu.py"] if is_gpu_stage(stage) else
                 sorted((ROOT / "tests").glob("test_frontier*.py")) +
                 [ROOT / "tests" / "test_desktop_slurm_runtime.py", ROOT / "tests" / "test_fedora_workflow.py"])
        if not tests or any(not path.is_file() for path in tests):
            raise ValueError("Frontier correctness tests are missing")
        if is_gpu_stage(stage):
            commands.append(("preflight", [python, str(ROOT / "scripts" / "fedora_gpu_preflight.py"),
                                           "--output", str(base / f"{stage}-gpu-preflight.json")]))
        commands.append(("gpu-tests" if is_gpu_stage(stage) else "tests",
            [python, "-m", "pytest", "-q", "-m", "gpu" if is_gpu_stage(stage) else "not gpu", *map(str, tests),
             "--basetemp", str(base / f"{stage}-pytest-work"), "-o", f"cache_dir={base / (stage + '-pytest-cache')}",
             "--junitxml", str(junit_path(workflow, stage))]))
        if is_gpu_stage(stage):
            commands.append(("check-gpu-tests", [python, str(ROOT / "scripts" / "frontier_workflow.py"),
                "check-gpu-tests", "--junit", str(junit_path(workflow, stage))]))
    args = [python, str(ROOT / "scripts" / "frontier.py"), "--stage", logical_stage(stage),
            "--profile", workflow["profile"], "--run-dir", str(base / stage),
            "--device", "cuda" if is_gpu_stage(stage) else "cpu"]
    for prior in STAGES[:STAGES.index(logical_stage(stage))]:
        args += ["--prerequisite-dir", f"{prior}={stage_path(workflow, prior)}"]
    if logical_stage(stage) != stage:
        args += ["--confirm-shard", stage]
    elif stage == "confirm" and execution_version(workflow) == 2:
        for descriptor in confirmation_shards(workflow["profile"]):
            part = descriptor["shard_id"]
            args += ["--confirm-shard-dir", f"{part}={stage_path(workflow, part)}"]
    if workflow.get("recovery"):
        args += ["--recovery-manifest", workflow["recovery"]["manifest_path"]]
    commands.append(("experiment", args))
    return commands


def begin_report(workflow, stage):
    base, site, resource = Path(workflow["run_dir"]), workflow["slurm_profile"], resource_for(workflow, stage)
    job = os.environ["SLURM_JOB_ID"]
    resources = {"partition": resource["partition"], "nodes": 1, "cpus": resource["cpus"],
                 "gpus": int(is_gpu_stage(stage)), "mem_bytes": resource["mem_gib"] * 1024**3,
                 "time_seconds": wall_seconds(resource["walltime"])}
    if site.get("account"):
        resources["account"] = site["account"]
    if is_gpu_stage(stage):
        resources["gpu_type"] = site["expected_gpu_name"]
    return cw.reporting_api().begin_report(base / stage, name=f"TDN/frontier/{stage}", script="scripts/frontier_worker.sh",
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
    return {"TDN_FRONTIER_PROTOCOL_SHA256": workflow["protocol_sha256"],
            "TDN_FRONTIER_WORKFLOW": str(Path(workflow["run_dir"]) / MANIFEST), "TDN_FRONTIER_STAGE": stage,
            "TDN_FRONTIER_CPUS": str(resource_for(workflow, stage)["cpus"])}


def prepare_test_environment(env, name):
    for key in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
        env.pop(key, None)
    for key in tuple(env):
        if key.startswith(("TDN_FRONTIER_", "TDN_ROADMAP_", "TDN_AGENDA_", "TDN_PREMIX_", "TDN_CONSISTENCY_")):
            env.pop(key)
    if name == "tests":
        for key in ("TDN_EXECUTION_MODE", "TDN_SLURM_CONFIG"):
            env.pop(key, None)


def worker(workflow, stage):
    if stage not in pending_stages(workflow):
        raise ValueError("Unknown or inherited frontier stage cannot run in a new allocation")
    backend = SimpleNamespace(**{name: globals()[name] for name in ("verify_allocation", "load", "begin_report",
        "worker_commands", "verify_stage", "prepare_test_environment", "workflow_environment", "finish_report", "verify_software")},
        gpu_junit_path=lambda workflow: junit_path(workflow, stage), workflow_filename=MANIFEST,
        workflow_label="frontier", stage_variable="TDN_FRONTIER_STAGE", prerequisite_stages=physical_stages(workflow)[1:-1],
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
    if workflow.get("recovery"):
        print(f"TDN_FRONTIER_RECOVERY_MANIFEST={workflow['recovery']['manifest_path']}")
        for stage, path in workflow["recovery"]["stage_paths"].items():
            print(f"TDN_FRONTIER_INHERITED_{stage.upper().replace('-', '_')}={path}")
    for filename in ("summary.txt", "mechanism_summary.json", "experiment_summary.csv", "experiment_index.json",
                     "review.csv", "review.md", "summary.json", "rows.jsonl", "analysis.json",
                     "figures/frontier-atlas.pdf", "figures/frontier-overview.png", "figures/index.html",
                     "figures/chart-data.json.gz", "figures/manifest.json"):
        path = base / "report" / filename
        if path.is_file():
            variable = re.sub(r"[^A-Z0-9_]", "_", filename.upper())
            print(f"TDN_FRONTIER_{variable}={path}")
    for row in cw.read_json(Path(workflow["run_dir"]) / "jobs.json"):
        if row.get("stage") not in physical_stages(workflow) or not pw.JOB.fullmatch(row.get("job_id", "")):
            raise ValueError("Invalid stored frontier scheduler job")
        for report in cw.tower_reports_for_job(workflow, row["job_id"]):
            print(f"{row['stage']} job {row['job_id']}\nTDN_TOWER_DIR={report}\nTDN_TOWER_METRICS={report / 'metrics.jsonl'}")


def status(workflow):
    base = Path(workflow["run_dir"])
    print(f"Fedora frontier workflow {workflow['run_id']} ({workflow['profile']})\nRun: {base}")
    jobs = {row["stage"]: row for row in cw.read_json(base / "jobs.json")}
    for stage in physical_stages(workflow):
        if stage in workflow.get("recovery", {}).get("stage_paths", {}):
            print(f"{stage}: INHERITED (sealed origin {stage_path(workflow, stage)})")
            try:
                verify_stage(workflow, stage)
                print("  sealed artifacts: VERIFIED; inherited cost is not charged as a new job")
            except (OSError, ValueError, KeyError, TypeError) as error:
                print(f"  sealed artifacts: INVALID ({error})")
            continue
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
            try:
                verify_stage(workflow, stage)
                summary = cw.read_json(base / stage / "summary.json")
                print("  sealed artifacts: VERIFIED")
                if summary.get("scientific_outcome"):
                    print(f"  scientific outcome: {summary['scientific_outcome']}")
            except (OSError, ValueError, KeyError, TypeError) as error:
                print(f"  sealed artifacts: INVALID ({error})")
    submission_path = base / "state" / "submission.json"
    if submission_path.exists() and (submission := cw.read_json(submission_path)).get("status") == "FAILED":
        print(f"Submission failed at {submission['stage']}: {submission['error']}")
    paths(workflow)


def validate_results(workflow):
    """Validate seals without mistaking a scientific BAD for artifact failure."""
    output = {"workflow": workflow["run_id"], "validation": "frontier_science_and_execution_seals", "stages": {}}
    invalid = False
    for stage in physical_stages(workflow):
        directory = stage_path(workflow, stage)
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
        raise ValueError("Some frontier stages are missing or invalid; final aggregation preserves their NA evidence")
    return output


ACCOUNTING_FIELDS = ("JobIDRaw", "State", "ExitCode", "ElapsedRaw", "TotalCPU", "AllocTRES", "MaxRSS", "CPUTimeRAW")
TERMINAL_JOB_STATES = frozenset(("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
    "PREEMPTED", "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "REVOKED", "SPECIAL_EXIT"))


def scheduler_accounting(workflow, *, enabled=True, runner=None):
    """Snapshot exact submitted jobs; unavailable accounting never invents cost."""
    base = cw.inside(workflow["run_dir"])
    jobs = cw.read_json(base / "jobs.json")
    if (not isinstance(jobs, list) or any(not isinstance(row, dict) or row.get("stage") not in physical_stages(workflow)
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
        "monetary_cost": None, "all_allocations_terminal": None,
        "inherited_stage_paths": workflow.get("recovery", {}).get("stage_paths", {}),
        "inherited_cost_scope": "Origin allocation costs remain in the preserved origin workflow; never added to new recovery jobs",
        "snapshot_note": "Snapshot values may lag Slurm; a running reporting allocation is partial, never a completed cost.",
        "units": {"ElapsedRaw": "seconds", "CPUTimeRAW": "allocated CPU-seconds reported by Slurm",
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
                row.update(workflow_stage=by_job[allocation], logical_stage=logical_stage(by_job[allocation]),
                    allocation_role=("confirmation_part" if logical_stage(by_job[allocation]) != by_job[allocation]
                        else "confirmation_aggregate" if by_job[allocation] == "confirm" and execution_version(workflow) == 2 else "stage"),
                    allocation_job_id=allocation,
                    record_kind="allocation" if identity == allocation else "step",
                    terminal_state=row["State"].split(" ", 1)[0].rstrip("+") in TERMINAL_JOB_STATES,
                    elapsed_seconds=int(row["ElapsedRaw"]) if row["ElapsedRaw"] else None,
                    allocated_cpu_seconds=int(row["CPUTimeRAW"]) if row["CPUTimeRAW"] else None,
                    missing_fields=[key for key in ACCOUNTING_FIELDS if not row[key]])
                records.append(row)
            allocations = {row["allocation_job_id"] for row in records if row["record_kind"] == "allocation"}
            missing = [job for job in by_job if job not in allocations]
            result.update(records=records, missing_allocation_job_ids=missing,
                all_allocations_terminal=not missing and all(row["terminal_state"] for row in records
                    if row["record_kind"] == "allocation"),
                nonterminal_allocation_job_ids=[row["allocation_job_id"] for row in records
                    if row["record_kind"] == "allocation" and not row["terminal_state"]],
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
    roots = {base}
    bridge = recovery_bridge(workflow)
    if bridge:
        verify_recovery_bridge(workflow)
        roots.update(cw.inside(path).parent for path in bridge["stage_paths"].values())
        roots.add(cw.inside(bridge["origin_workflow_path"]).parent)
        roots.update(cw.inside(record["path"]).parent for record in bridge.get("ancestry_workflows", []))
    for source in sorted(roots):
        if source.parent != ROOT / "runs":
            raise ValueError("Review archive inherited evidence must remain in project run directories")
        for path in sorted(source.rglob("*")):
            relative = path.relative_to(source)
            if any(part == "__pycache__" or part.endswith(("pytest-work", "pytest-cache")) for part in relative.parts):
                continue
            if path.is_symlink():
                raise ValueError(f"Review archive refuses symlink: {path}")
            if cw.inside(path).is_file():
                members.append((path, str(Path(source.name) / relative)))
    with destination.open("xb") as stream:
        with tarfile.open(fileobj=stream, mode="w:gz", dereference=False) as archive:
            for path, relative in members:
                archive.add(path, arcname=relative, recursive=False)
    index = {"schema_version": 1, "archive": destination.name, "bytes": destination.stat().st_size,
             "sha256": cw.digest(destination), "member_count": len(members), "parts": [],
             "included_workflow_directories": sorted(path.name for path in roots),
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
    item = commands.add_parser("recover", help="Reuse sealed prerequisites and complete parts in a fresh bounded workflow")
    item.add_argument("run", nargs="?", default="latest")
    item.add_argument("--run-id")
    item.add_argument("--plan", action="store_true", help="Validate compatibility and show recovery jobs without submitting")
    for name in ("status", "logs", "collect", "paths", "validate"):
        item = commands.add_parser(name)
        item.add_argument("run", nargs="?", default="latest")
        if name == "logs":
            item.add_argument("--lines", type=int, default=200)
        elif name == "collect":
            item.add_argument("--no-accounting", action="store_true", help="Skip optional exact-job sacct metadata collection")
    item = commands.add_parser("worker", help=argparse.SUPPRESS)
    item.add_argument("--workflow", type=Path, required=True)
    item.add_argument("--stage", required=True)
    item = commands.add_parser("check-gpu-tests", help=argparse.SUPPRESS)
    item.add_argument("--junit", type=Path, required=True)
    item = commands.add_parser("snapshot-accounting", help=argparse.SUPPRESS)
    item.add_argument("--workflow", type=Path, required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command in ("plan", "run"):
            start(args)
        elif args.command == "recover":
            recover(args)
        elif args.command == "worker":
            return worker(load(args.workflow, verify=False), args.stage)
        elif args.command == "check-gpu-tests":
            print(json.dumps(check_gpu_tests(args.junit), sort_keys=True))
        elif args.command == "snapshot-accounting":
            scheduler_accounting(load(args.workflow))
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
        print(f"TDN frontier: {exc}", file=sys.stderr)
        return exc.exit_code
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"TDN frontier: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
