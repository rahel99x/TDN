"""Narrow immutable bridge for interrupted frontier confirmation.

This is an orchestration compatibility exception, not a scientific migration.
Origin artifacts are retained in place. Only the pinned release, or exactly the
current execution source, can supply the first five stages. No checkpoint is
unpickled here; ordinary model loading occurs only in the allocated worker.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
ORIGIN_COMMIT = "aa5760cd7f0ecc0a734c1d3f9090e48d2f22cb51"
ORIGIN_SOURCE = "7999ec430eebede273c14614359c08fa1a7e564c4a0ec9412e12d54337adfcb2"
ORIGIN_CONTROLLER_SOURCE = "4c308247d3aad4af3957efe8501734c3f1514532aa5eb46ef0f5c521814842c5"
INHERITED_STAGES = ("audit", "screen", "prepare", "train", "confirm_prepare")
STAGES = INHERITED_STAGES + ("confirm", "scaling", "policy", "report")
GPU_STAGES = ("train", "confirm", "scaling", "policy")
SOFTWARE_KEYS = ("python", "executable", "venv", "torch", "numpy", "scipy", "torch_cuda_runtime")
TARGET_KEYS = SOFTWARE_KEYS + ("git_commit", "source_tree_sha256")
# These modules may change orchestration/reporting only. The exact target source
# is still bound in the recovery manifest, and checked before/after execution.
ORCHESTRATION_FILES = frozenset(("tdn/analysis/frontier/engine.py", "tdn/analysis/frontier/report.py",
    "tdn/frontier_reporting.py"))
NEW_ORCHESTRATION_FILES = frozenset(("tdn/analysis/frontier/recovery.py", "tdn/analysis/frontier/partition.py",
    "tdn/analysis/frontier/confirmation.py"))
NEURAL = "tdn/analysis/frontier/neural.py"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate recovery evidence JSON field")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite recovery JSON")))


def _inside(path, root):
    path, root = Path(path), Path(root).resolve()
    if not path.is_absolute():
        path = root / path
    if (not path.resolve().is_relative_to(root) or path.is_symlink() or any(
            parent.is_symlink() for parent in path.parents if parent.is_relative_to(root))):
        raise ValueError("Recovery evidence must stay inside this project without symlinks")
    return path.resolve()


def software_identity(value, *, target=False):
    keys = TARGET_KEYS if target else SOFTWARE_KEYS
    if any(key not in value for key in keys) or value.get("venv") is not True:
        raise ValueError("Recovery requires complete project-venv software identity")
    if any(not isinstance(value[key], str) or not value[key] for key in keys if key not in ("venv", "torch_cuda_runtime")):
        raise ValueError("Recovery software identity is incomplete")
    return {key: value[key] for key in keys}


def source_tree_hash(root):
    result = hashlib.sha256()
    for directory in ("tdn", "reference", "scripts", "configs", "tests"):
        for path in sorted((Path(root) / directory).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                result.update(path.relative_to(root).as_posix().encode())
                result.update(path.read_bytes())
    return result.hexdigest()


@lru_cache(maxsize=4)
def _baseline(root):
    """Read pinned Git blobs as bytes, never import or execute prior source."""
    root = Path(root)
    names = subprocess.run(["git", "ls-tree", "-rz", "--name-only", ORIGIN_COMMIT,
        "tdn", "reference", "scripts", "configs", "tests"], cwd=root, capture_output=True, check=True).stdout
    files = {}
    for name in filter(None, names.split(b"\0")):
        files[name.decode()] = subprocess.run(["git", "show", ORIGIN_COMMIT + ":" + name.decode()],
            cwd=root, capture_output=True, check=True).stdout
    hasher = hashlib.sha256()
    for directory in ("tdn", "reference", "scripts", "configs", "tests"):
        for name in sorted(key for key in files if key.startswith(directory + "/")):
            hasher.update(name.encode()); hasher.update(files[name])
    if hasher.hexdigest() != ORIGIN_SOURCE:
        raise ValueError("Pinned recovery release source inventory differs")
    for name in ("requirements.txt", "pyproject.toml"):
        files[name] = subprocess.run(["git", "show", ORIGIN_COMMIT + ":" + name], cwd=root,
            capture_output=True, check=True).stdout
    return files


def semantic_fingerprint(root=ROOT):
    """Reject any scientific implementation drift across the compatibility bridge."""
    root, fingerprints = Path(root), {}
    try:
        baseline = _baseline(str(root.resolve()))
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("Pinned recovery release is unavailable in local Git history; fetch that revision before recovery") from error
    expected = {name for name in baseline if name.startswith(("tdn/", "reference/"))}
    actual = {path.relative_to(root).as_posix() for directory in ("tdn", "reference")
              for path in (root / directory).rglob("*") if path.is_file() and "__pycache__" not in path.parts}
    if actual - expected - NEW_ORCHESTRATION_FILES or expected - actual:
        raise ValueError("Recovery scientific source inventory changed")
    guarded = (expected - ORCHESTRATION_FILES) | {name for name in ("requirements.txt", "pyproject.toml") if name in baseline}
    for name in sorted(guarded):
        path = _inside(root / name, root)
        before, after = baseline[name], path.read_bytes()
        if name == NEURAL:
            marker = b"\ndef confirm("
            if marker not in before or marker not in after:
                raise ValueError("Recovery neural training boundary changed")
            before, after = before.split(marker, 1)[0], after.split(marker, 1)[0]
        if before != after:
            raise ValueError(f"Recovery scientific implementation changed: {name}")
        fingerprints[name] = hashlib.sha256(after).hexdigest()
    return digest(fingerprints)


def _old_declaration(protocol):
    return {"version": 1, "benchmark_suite": "frontier", "profile": protocol["profile"],
        "scientific_protocol": protocol, "stages": list(STAGES), "execution_mode": "desktop-slurm",
        "dependencies": {stage: list(STAGES[:index]) for index, stage in enumerate(STAGES)},
        "device": {stage: "cuda" if stage in GPU_STAGES else "cpu" for stage in STAGES},
        "resources_per_stage": {stage: {key: protocol["budgets"][stage][key]
            for key in ("cpus", "mem_gib", "walltime")} for stage in STAGES},
        "scheduler_memory_cap_mib": 110000,
        "scope": "bounded preregistered frontier; no automatic expansion or pending-job cap",
        "report_dependencies": "afterany across all scientific stages", "scientific_bad_is_scheduler_failure": False}


def _execution(path, protocol, *, partition=None):
    # Import trusted current code lazily, preserving torch-free controller imports.
    spec = importlib.util.spec_from_file_location("_frontier_recovery_stage_cli", ROOT / "scripts/frontier.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.verify_execution(path, protocol, **({"confirmation_partition": partition} if partition is not None else {}))


def _origin(origin_path, protocol, site, root):
    origin_path = _inside(origin_path, root)
    if origin_path.is_dir():
        if (origin_path / "frontier-workflow.json").exists():
            origin_path /= "frontier-workflow.json"
        else:
            if site is not None or protocol["profile"] == "full":
                raise ValueError("Only explicit local CPU smoke/development may recover without a scheduler manifest")
            return origin_path, None, None, "local-cpu"
    workflow = _json(origin_path)
    base = _inside(workflow.get("run_dir", ""), root)
    if (origin_path != base / "frontier-workflow.json" or workflow.get("kind") != "desktop-slurm-frontier"
            or workflow.get("schema_version") != 1 or workflow.get("execution_mode") != "desktop-slurm"
            or _inside(workflow.get("root", ""), root) != Path(root).resolve()
            or base != Path(root).resolve() / "runs" / workflow.get("run_id", "")
            or workflow.get("profile") != protocol["profile"]):
        raise ValueError("Recovery origin workflow root/layout/profile differs")
    if site is None:
        raise ValueError("Native recovery requires the unchanged Fedora Slurm profile")
    from tdn.runtime.desktop_slurm import validate_profile
    validate_profile(site, root=Path(root))
    if (workflow.get("slurm_profile") != site or workflow.get("slurm_profile_sha256") != digest(site)
            or _inside(workflow.get("slurm_profile_path", ""), root) != base / "slurm-profile.json"
            or _json(base / "slurm-profile.json") != site or workflow.get("torch_version") != site["torch_version"]):
        raise ValueError("Recovery origin Slurm profile changed")
    if _inside(workflow.get("protocol_path", ""), root) != base / "protocol.json":
        raise ValueError("Recovery origin declaration path changed")
    declaration = _json(base / "protocol.json")
    if workflow.get("execution_version", 1) == 2:
        spec = importlib.util.spec_from_file_location("_frontier_recovery_workflow", ROOT / "scripts/frontier_workflow.py")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        module.validate(workflow)
        expected_declaration = module.declaration(protocol["profile"], 2)
    else:
        expected_declaration = _old_declaration(protocol)
    if declaration != expected_declaration or workflow.get("protocol_sha256") != digest(declaration):
        raise ValueError("Recovery origin workflow declaration changed")
    resources = {stage: {**declaration["resources_per_stage"][stage],
        "partition": site["gpu_partition" if stage in GPU_STAGES else "cpu_partition"]} for stage in STAGES}
    if workflow.get("resources") != resources:
        raise ValueError("Recovery origin resources changed")
    return base, origin_path, workflow, "desktop-slurm"


def _validate_banks(protocol, paths):
    """Hash every checkpoint; load numeric NPZ with allow_pickle=False only."""
    from .data import load_split, verify_frozen_training
    ctx = SimpleNamespace(protocol=protocol, stage="confirm", prerequisites=paths)
    frozen = verify_frozen_training(ctx)
    counts = {}
    for split in ("train", "validation", "calibration", "confirmation", "scaling", "policy"):
        bank = load_split(ctx, split)
        refs = [row for parent in bank for row in parent["references"].values()]
        counts[split] = {"parents": len(bank), "accepted_references": sum(row["accepted"] for row in refs),
                         "unresolved_references": sum(not row["accepted"] for row in refs)}
    return {"frozen_training": frozen, "banks": counts, "checkpoint_deserialization": False}


def _reusable_parts(base, workflow, prior_bridge, protocol, paths, current, site, root, mode):
    from .partition import plan_shards
    from .data import verify_frozen_training
    result = {}
    lineage = {stage: {"run_dir": str(path), "workflow_seal_sha256": file_digest(path / "workflow-seal.json")}
               for stage, path in paths.items()}
    science_lineage = {stage: file_digest(path / "science_manifest.json") for stage, path in paths.items()}
    for partition in plan_shards(protocol):
        stage = partition["shard_id"]
        inherited = prior_bridge is not None and stage in prior_bridge["stage_paths"]
        path = _inside(prior_bridge["stage_paths"][stage] if inherited else base / stage, root)
        if not (path / "COMPLETED").exists():
            if inherited:
                raise ValueError("Previously completed confirmation partition disappeared")
            continue
        verified = _execution(path, protocol, partition=partition)
        science, execution = verified["science"], verified["execution"]
        if (science.get("source_tree_sha256") != current["source_tree_sha256"]
                or software_identity(execution.get("software", {}), target=True) != current
                or science.get("stage") != "confirm" or execution.get("stage") != "confirm"
                or execution.get("execution_mode") != mode or execution.get("profile") != protocol["profile"]
                or execution.get("device") != ("cuda" if mode == "desktop-slurm" else "cpu")
                or execution.get("prerequisites") != lineage or science.get("prerequisites") != science_lineage):
            raise ValueError("Completed confirmation partition source/software/device/lineage differs")
        if mode == "desktop-slurm" and (execution.get("slurm_profile_sha256") != digest(site)
                or not re.fullmatch(r"[0-9]+", str(execution.get("software", {}).get("slurm_job_id", "")))):
            raise ValueError("Completed confirmation partition Slurm profile/job differs")
        from .neural import validate_confirmation_coverage
        validate_confirmation_coverage(path, protocol, partition=partition)
        scope = _json(path / "confirmation_scope.json")
        frozen = verify_frozen_training(SimpleNamespace(protocol=protocol, prerequisites=paths))
        binding = {**frozen, "source_tree_sha256": current["source_tree_sha256"], "prerequisites": science_lineage}
        if (scope.get("binding") != binding or scope.get("partition") != partition
                or scope.get("measurement_device") != execution["device"]):
            raise ValueError("Completed confirmation partition frozen models or parent assignment differs")
        result[stage] = {"run_dir": str(path), "science_manifest_sha256": file_digest(path / "science_manifest.json"),
            "workflow_seal_sha256": file_digest(path / "workflow-seal.json"), "artifacts": science["artifacts"],
            "source_tree_sha256": current["source_tree_sha256"], "partition": partition, "confirmation_scope_sha256": file_digest(path / "confirmation_scope.json")}
    return result


def create_recovery(origin_workflow_path, protocol, current_software, site, *, root=ROOT, _trail=()):
    """Build a JSON-only bridge; caller writes it into a NEW coordinator directory."""
    from .protocol import validate_protocol
    validate_protocol(protocol)
    root = Path(root).resolve()
    current = software_identity(current_software, target=True)
    if source_tree_hash(root) != current["source_tree_sha256"]:
        raise ValueError("Recovery target source differs from current checkout")
    base, origin_file, workflow, mode = _origin(origin_workflow_path, protocol, site, root)
    identity = str(origin_file or base)
    if identity in _trail or len(_trail) >= 16:
        raise ValueError("Recovery origin lineage contains a cycle or exceeds its bounded depth")
    _trail = (*_trail, identity)
    prior_bridge, local_recovery_origin = None, None
    descriptor = workflow.get("recovery") if workflow is not None else None
    if workflow is None and (base / "recovery.json").exists():
        if mode != "local-cpu" or site is not None or protocol["profile"] == "full":
            raise ValueError("Local recovery inheritance requires explicit CPU smoke/development")
        if any((base / stage).exists() for stage in INHERITED_STAGES):
            raise ValueError("Local recovery coordinator ambiguously contains inherited stage directories")
        bridge_path = _inside(base / "recovery.json", root)
        local_recovery_origin = {"path": str(bridge_path), "sha256": file_digest(bridge_path)}
        descriptor = {"manifest_path": str(bridge_path), "sha256": local_recovery_origin["sha256"]}
    if descriptor is not None:
        bridge_path = _inside(descriptor.get("manifest_path", ""), root)
        if bridge_path != base / "recovery.json" or file_digest(bridge_path) != descriptor.get("sha256"):
            raise ValueError("Prior recovery manifest changed")
        prior_bridge = _json(bridge_path)
        if workflow is not None and descriptor.get("stage_paths") != prior_bridge.get("stage_paths"):
            raise ValueError("Prior recovery stage mapping changed")
        inherited = verify_recovery(prior_bridge, protocol, current_software, site, root=root, _trail=_trail)
        paths = {stage: inherited[stage] for stage in INHERITED_STAGES}
    else:
        paths = {stage: _inside(base / stage, root) for stage in INHERITED_STAGES}
    stages, execution_lineage, science_lineage, origin_software = {}, {}, {}, None
    for stage, path in paths.items():
        verified = _execution(path, protocol)
        science, execution = verified["science"], verified["execution"]
        software = software_identity(execution.get("software", {}), target=True)
        if origin_software is None:
            origin_software = software
        if software != origin_software or software_identity(software) != software_identity(current):
            raise ValueError("Recovery cannot change origin or target Python/Torch/software environment")
        expected_device = "cuda" if mode == "desktop-slurm" and stage in GPU_STAGES else "cpu"
        if (execution.get("stage") != stage or execution.get("profile") != protocol["profile"]
                or execution.get("device") != expected_device or execution.get("execution_mode") != mode
                or science.get("stage") != stage or science.get("source_tree_sha256") != software["source_tree_sha256"]):
            raise ValueError("Recovery origin stage/device/source differs")
        if (execution.get("prerequisites") != execution_lineage or science.get("prerequisites") != science_lineage):
            raise ValueError("Recovery origin stages belong to different lineages")
        if mode == "desktop-slurm":
            if (execution.get("slurm_profile_sha256") != digest(site)
                    or (prior_bridge is None and execution.get("workflow_protocol_sha256") != workflow["protocol_sha256"])
                    or not re.fullmatch(r"[0-9]+", str(execution.get("software", {}).get("slurm_job_id", "")))):
                raise ValueError("Recovery origin native execution profile/job differs")
        stages[stage] = {"run_dir": str(path), "science_manifest_sha256": file_digest(path / "science_manifest.json"),
            "workflow_seal_sha256": file_digest(path / "workflow-seal.json"), "artifacts": science["artifacts"],
            "source_tree_sha256": software["source_tree_sha256"]}
        execution_lineage[stage] = {"run_dir": str(path), "workflow_seal_sha256": stages[stage]["workflow_seal_sha256"]}
        science_lineage[stage] = stages[stage]["science_manifest_sha256"]
    source = origin_software["source_tree_sha256"]
    if source == current["source_tree_sha256"]:
        if origin_software != current:
            raise ValueError("Same-source recovery cannot change revision/software identity")
        semantics = None
    elif source == ORIGIN_SOURCE and (origin_software["git_commit"] == ORIGIN_COMMIT or (
            mode == "local-cpu" and origin_software["git_commit"] == "758cbf5a35998aa443c6dd7e38266bab4ff536b6")):
        semantics = semantic_fingerprint(root)
    else:
        raise ValueError("Recovery origin is neither the pinned compatible release nor this exact source")
    if workflow is not None and workflow.get("source_tree_sha256") != (current["source_tree_sha256"] if prior_bridge is not None else source):
        raise ValueError("Recovery origin coordinator and stage source differ")
    if workflow is not None and prior_bridge is None and source == ORIGIN_SOURCE and workflow.get("source_sha256") != ORIGIN_CONTROLLER_SOURCE:
        raise ValueError("Recovery origin controller/dependency fingerprint differs from the pinned release")
    validated_banks = _validate_banks(protocol, paths)
    reusable = _reusable_parts(base, workflow, prior_bridge, protocol, paths, current, site, root, mode)
    paths.update({key: Path(value["run_dir"]) for key, value in reusable.items()})
    stages.update(reusable)
    result = {"schema": "tdn.frontier-recovery/v1", "profile": protocol["profile"], "execution_mode": mode,
        "protocol_sha256": digest(protocol), "origin_root": str(root), "origin_run_dir": str(base),
        "origin_workflow_path": str(origin_file) if origin_file else None,
        "origin_workflow_sha256": file_digest(origin_file) if origin_file else None,
        "ancestry_workflows": (prior_bridge.get("ancestry_workflows", []) if prior_bridge else []) +
            ([{"path": str(origin_file), "sha256": file_digest(origin_file)}] if origin_file else []),
        "origin_declaration_sha256": file_digest(base / "protocol.json") if origin_file else None,
        "origin_source_tree_sha256": source, "origin_software": origin_software, "target_software": current,
        "slurm_profile_sha256": digest(site) if site is not None else None,
        "semantic_fingerprint": semantics, "stage_paths": {key: str(path) for key, path in paths.items()},
        "stages": stages, "validated_banks": validated_banks,
        "scope": "immutable first-five-stage reuse and exact-current-source completed partitions; original partial confirmation excluded"}
    if local_recovery_origin is not None:
        result["origin_recovery_manifest"] = local_recovery_origin
    return result


def verify_recovery(bridge, protocol, current_software, site, *, root=ROOT, _trail=()):
    """Reverify all origin evidence and the narrow bridge; never trust its claims."""
    if not isinstance(bridge, dict) or bridge.get("schema") != "tdn.frontier-recovery/v1":
        raise ValueError("Unsupported frontier recovery bridge")
    origin = bridge.get("origin_workflow_path") or bridge.get("origin_run_dir")
    regenerated = create_recovery(origin, protocol, current_software, site, root=root, _trail=_trail)
    if regenerated != bridge:
        raise ValueError("Recovery bridge or origin evidence changed")
    return {stage: Path(path) for stage, path in bridge["stage_paths"].items()}


def allowed_stage_sources(bridge):
    """Use only AFTER verify_recovery, for exact inherited paths and stages."""
    return {stage: {"path": path, "source_tree_sha256": bridge["origin_source_tree_sha256"]
                   if stage in INHERITED_STAGES else bridge["target_software"]["source_tree_sha256"]}
            for stage, path in bridge["stage_paths"].items()}
