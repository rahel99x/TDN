"""Recovery compatibility and immutable provenance; fixtures are unit evidence only."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from tdn.analysis.frontier import recovery
from tdn.analysis.frontier.protocol import build_protocol
from tdn.research.protocol import digest, file_digest
from tdn.runtime.metadata import write_json


def software(source=recovery.ORIGIN_SOURCE, commit=recovery.ORIGIN_COMMIT):
    return dict(python="3.13.13", executable="/project/.venv/bin/python", venv=True,
        torch="2.10.0+cu126", numpy="2.2.6", scipy="1.16", torch_cuda_runtime="12.6",
        source_tree_sha256=source, git_commit=commit)


@pytest.fixture
def origin(tmp_path, monkeypatch):
    """Use actual science/wrapper seals; only expensive bank checks are isolated."""
    from tdn.analysis.frontier.engine import _seal
    spec = recovery.importlib.util.spec_from_file_location("_recovery_test_cli", recovery.ROOT / "scripts/frontier.py")
    cli = recovery.importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    protocol = build_protocol("smoke")
    base = tmp_path / "local-origin"
    paths, science_lineage, execution_lineage = {}, {}, {}
    for stage in recovery.INHERITED_STAGES:
        path = base / stage; path.mkdir(parents=True)
        execution = dict(stage=stage, profile="smoke", device="cpu", execution_mode="local-cpu",
            software=software(), protocol_sha256=digest(protocol), prerequisites=deepcopy(execution_lineage))
        write_json(path / "execution.json", execution)
        write_json(path / "protocol.json", protocol)
        write_json(path / "summary.json", dict(schema=protocol["schema"], stage=stage, status="COMPLETED",
            protocol_sha256=digest(protocol), source_tree_sha256=recovery.ORIGIN_SOURCE,
            experiment_count=0, verdict_counts={}))
        write_json(path / "rows.json", {"rows": []})
        (path / "rows.jsonl").write_text("")
        (path / "review.csv").write_text("unit fixture only\n")
        (path / "review.md").write_text("unit fixture only\n")
        (path / "summary.txt").write_text("unit fixture only\n")
        write_json(path / "stage.json", {**execution, "status": "COMPLETED"})
        _seal(protocol, path, stage, deepcopy(science_lineage), recovery.ORIGIN_SOURCE)
        cli.seal_execution(path, protocol)
        science_lineage[stage] = file_digest(path / "science_manifest.json")
        execution_lineage[stage] = {"run_dir": str(path), "workflow_seal_sha256": file_digest(path / "workflow-seal.json")}
        paths[stage] = path
    current = software("a" * 64, "b" * 40)
    monkeypatch.setattr(recovery, "source_tree_hash", lambda root: current["source_tree_sha256"])
    monkeypatch.setattr(recovery, "semantic_fingerprint", lambda root: "verified-unit-semantic-fingerprint")
    bank_calls = []
    def validate_banks(protocol, paths):
        bank_calls.append(tuple(paths))
        return {"unit_bank_checks_only": True, "checkpoint_deserialization": False}
    monkeypatch.setattr(recovery, "_validate_banks", validate_banks)
    return base, protocol, current, paths, bank_calls


def test_recovery_reuses_exact_first_five_paths_and_never_partial_confirmation(origin):
    base, protocol, current, paths, bank_calls = origin
    (base / "confirm").mkdir()
    (base / "confirm" / "summary.json").write_text('{"status":"INTERRUPTED"}')
    bridge = recovery.create_recovery(base, protocol, current, None)
    assert recovery.verify_recovery(bridge, protocol, current, None) == paths
    assert set(bridge["stage_paths"]) == set(recovery.INHERITED_STAGES)
    assert "confirm" not in bridge["stage_paths"]
    assert bridge["origin_software"] == software()
    assert bridge["target_software"] == current
    assert bridge["semantic_fingerprint"] == "verified-unit-semantic-fingerprint"
    assert len(bank_calls) == 2
    for stage in paths:
        assert recovery.allowed_stage_sources(bridge)[stage] == dict(path=str(paths[stage]), source_tree_sha256=recovery.ORIGIN_SOURCE)


@pytest.mark.parametrize("name", ["summary.json", "execution.json", "workflow-seal.json", "science_manifest.json", "COMPLETED", "rows.jsonl"])
def test_recovery_rejects_any_modified_origin_evidence(origin, name):
    base, protocol, current, paths, _ = origin
    bridge = recovery.create_recovery(base, protocol, current, None)
    (paths["train"] / name).write_text("{}")
    with pytest.raises((ValueError, KeyError)):
        recovery.verify_recovery(bridge, protocol, current, None)


@pytest.mark.parametrize("key,value", [("python", "3.14"), ("torch", "2.11"), ("numpy", "1"),
    ("scipy", "2"), ("torch_cuda_runtime", "other"), ("executable", "/elsewhere/python"), ("venv", False)])
def test_recovery_rejects_environment_changes(origin, key, value):
    base, protocol, current, _, _ = origin
    with pytest.raises(ValueError, match="software|venv"):
        recovery.create_recovery(base, protocol, {**current, key: value}, None)


@pytest.mark.parametrize("change", ["target", "path", "software", "manifest", "extra"])
def test_recovery_rejects_manifest_claim_tampering(origin, change):
    base, protocol, current, _, _ = origin
    bridge = recovery.create_recovery(base, protocol, current, None)
    if change == "target": bridge["target_software"]["source_tree_sha256"] = "c" * 64
    elif change == "path": bridge["stage_paths"]["train"] = str(base / "confirm")
    elif change == "software": bridge["origin_software"]["torch"] = "wrong"
    elif change == "manifest": bridge["stages"]["train"]["science_manifest_sha256"] = "c" * 64
    else: bridge["extra_permission"] = "change teacher"
    with pytest.raises(ValueError, match="bridge or origin"):
        recovery.verify_recovery(bridge, protocol, current, None)


def test_recovery_requires_all_five_stages(origin):
    base, protocol, current, paths, _ = origin
    (paths["confirm_prepare"] / "COMPLETED").unlink()
    with pytest.raises(OSError):
        recovery.create_recovery(base, protocol, current, None)


def test_full_recovery_cannot_disguise_itself_as_local_cpu(origin):
    base, _, current, _, _ = origin
    with pytest.raises(ValueError, match="local CPU"):
        recovery.create_recovery(base, build_protocol("full"), current, None)


def test_recovery_checks_actual_target_source_before_reading_origins(origin, monkeypatch):
    base, protocol, current, _, bank_calls = origin
    monkeypatch.setattr(recovery, "source_tree_hash", lambda root: "d" * 64)
    with pytest.raises(ValueError, match="target source"):
        recovery.create_recovery(base, protocol, current, None)
    assert bank_calls == []


def test_recovery_rejects_symlinked_origin(origin):
    base, protocol, current, _, _ = origin
    shortcut = base.parent / "shortcut"; shortcut.symlink_to(base, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        recovery.create_recovery(shortcut, protocol, current, None)


def test_json_rejects_duplicate_or_nonfinite_claims(tmp_path):
    path = tmp_path / "claim.json"
    for value in ('{"source":1,"source":2}', '{"source":NaN}'):
        path.write_text(value)
        with pytest.raises(ValueError):
            recovery._json(path)


@pytest.fixture
def semantic_tree(tmp_path, monkeypatch):
    baseline = {"tdn/analysis/frontier/models.py": b"physical model\n",
        recovery.NEURAL: b"training and measurement formulas\n\ndef confirm(ctx):\n    old()\n",
        "tdn/analysis/frontier/engine.py": b"old orchestration\n",
        "reference/core.py": b"reference equation\n"}
    for name, content in baseline.items():
        path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(content)
    monkeypatch.setattr(recovery, "_baseline", lambda root: baseline)
    return tmp_path


def test_semantic_guard_allows_only_confirmation_suffix_and_orchestration(semantic_tree):
    root = semantic_tree
    before = recovery.semantic_fingerprint(root)
    (root / recovery.NEURAL).write_bytes(b"training and measurement formulas\n\ndef confirm(ctx):\n    parts()\n")
    (root / "tdn/analysis/frontier/engine.py").write_text("new orchestration\n")
    (root / "tdn/analysis/frontier/recovery.py").write_text("new recovery\n")
    assert recovery.semantic_fingerprint(root) == before


@pytest.mark.parametrize("name", ["tdn/analysis/frontier/models.py", "reference/core.py", recovery.NEURAL])
def test_semantic_guard_rejects_changed_physics_training_or_timing(semantic_tree, name):
    path = semantic_tree / name
    path.write_bytes(b"changed\n" + path.read_bytes())
    with pytest.raises(ValueError, match="scientific implementation"):
        recovery.semantic_fingerprint(semantic_tree)


def test_semantic_guard_rejects_added_unreviewed_science_module(semantic_tree):
    (semantic_tree / "tdn/analysis/frontier/new_teacher.py").write_text("new teacher\n")
    with pytest.raises(ValueError, match="inventory"):
        recovery.semantic_fingerprint(semantic_tree)


def test_recovery_module_import_does_not_import_torch():
    import subprocess, sys
    source = "import sys; import tdn.analysis.frontier.recovery; assert 'torch' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", source], cwd=recovery.ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_native_origin_checks_fixed_scheduler_declaration_and_device(origin, monkeypatch):
    from tdn.analysis.frontier.engine import _seal
    from tdn.runtime import desktop_slurm
    base, protocol, current, paths, _ = origin
    root = base.parent
    native = root / "runs" / "native-origin"; native.parent.mkdir()
    base.rename(native)
    site = {"torch_version": "2.10.0+cu126", "cpu_partition": "local", "gpu_partition": "local"}
    monkeypatch.setattr(desktop_slurm, "validate_profile", lambda profile, root: profile)
    declaration = recovery._old_declaration(protocol)
    write_json(native / "protocol.json", declaration)
    write_json(native / "slurm-profile.json", site)
    workflow = dict(schema_version=1, kind="desktop-slurm-frontier", run_id=native.name,
        root=str(root), run_dir=str(native), execution_mode="desktop-slurm", profile="smoke",
        protocol_path=str(native / "protocol.json"), protocol_sha256=digest(declaration),
        slurm_profile=site, slurm_profile_path=str(native / "slurm-profile.json"),
        slurm_profile_sha256=digest(site), torch_version=site["torch_version"],
        source_tree_sha256=recovery.ORIGIN_SOURCE, source_sha256=recovery.ORIGIN_CONTROLLER_SOURCE,
        resources={stage: {**declaration["resources_per_stage"][stage], "partition": "local"} for stage in recovery.STAGES})
    write_json(native / "frontier-workflow.json", workflow)
    science_lineage, execution_lineage = {}, {}
    for stage in recovery.INHERITED_STAGES:
        path = native / stage
        execution = recovery._json(path / "execution.json")
        execution.update(execution_mode="desktop-slurm", device="cuda" if stage == "train" else "cpu",
            prerequisites=deepcopy(execution_lineage), slurm_profile_sha256=digest(site),
            workflow_protocol_sha256=digest(declaration))
        execution["software"]["slurm_job_id"] = "450"
        write_json(path / "execution.json", execution)
        write_json(path / "stage.json", {**execution, "status": "COMPLETED"})
        _seal(protocol, path, stage, deepcopy(science_lineage), recovery.ORIGIN_SOURCE)
        files = ("execution.json", "protocol.json", "stage.json", "science_manifest.json")
        write_json(path / "workflow-seal.json", dict(schema="tdn.frontier/v1", schema_version=1,
            protocol_sha256=digest(protocol), files={name: file_digest(path / name) for name in files}))
        science_lineage[stage] = file_digest(path / "science_manifest.json")
        execution_lineage[stage] = {"run_dir": str(path), "workflow_seal_sha256": file_digest(path / "workflow-seal.json")}
    bridge = recovery.create_recovery(native / "frontier-workflow.json", protocol, current, site, root=root)
    assert bridge["execution_mode"] == "desktop-slurm"
    assert bridge["ancestry_workflows"] == [{"path": str(native / "frontier-workflow.json"),
                                            "sha256": file_digest(native / "frontier-workflow.json")}]
    assert recovery.verify_recovery(bridge, protocol, current, site, root=root)["train"] == native / "train"
    workflow["resources"]["train"]["mem_gib"] = 100
    write_json(native / "frontier-workflow.json", workflow)
    with pytest.raises(ValueError, match="resources"):
        recovery.verify_recovery(bridge, protocol, current, site, root=root)


@pytest.fixture
def reusable_part(origin, monkeypatch):
    from tdn.analysis.frontier import data, neural
    from tdn.analysis.frontier.partition import plan_shards
    base, protocol, current, paths, _ = origin
    part = plan_shards(protocol)[0]
    path = base / part["shard_id"]; path.mkdir()
    (path / "COMPLETED").write_text("unit completed marker")
    (path / "science_manifest.json").write_text("unit manifest")
    (path / "workflow-seal.json").write_text("unit seal")
    lineage = {stage: {"run_dir": str(prior), "workflow_seal_sha256": file_digest(prior / "workflow-seal.json")}
               for stage, prior in paths.items()}
    sciences = {stage: file_digest(prior / "science_manifest.json") for stage, prior in paths.items()}
    execution = dict(software=current, stage="confirm", profile="smoke", device="cpu", execution_mode="local-cpu", prerequisites=lineage)
    science = dict(stage="confirm", source_tree_sha256=current["source_tree_sha256"], prerequisites=sciences, artifacts={})
    frozen = dict(catalog_sha256="a", freeze_sha256="b", checkpoint_hashes={"unit": "c"}, selection_split="validation")
    write_json(path / "confirmation_scope.json", dict(partition=part, measurement_device="cpu",
        binding={**frozen, "source_tree_sha256": current["source_tree_sha256"], "prerequisites": sciences}))
    monkeypatch.setattr(recovery, "_execution", lambda path, protocol, partition=None: {"science": science, "execution": execution})
    monkeypatch.setattr(data, "verify_frozen_training", lambda ctx: frozen)
    monkeypatch.setattr(neural, "validate_confirmation_coverage", lambda *args, **kwargs: None)
    return base, protocol, current, paths, part, path, execution, science


def test_completed_part_reuse_keeps_exact_partition_and_measurement_revision(reusable_part):
    base, protocol, current, paths, part, path, execution, science = reusable_part
    result = recovery._reusable_parts(base, None, None, protocol, paths, current, None, recovery.ROOT, "local-cpu")
    assert set(result) == {part["shard_id"]}
    assert result[part["shard_id"]]["run_dir"] == str(path)
    assert result[part["shard_id"]]["source_tree_sha256"] == current["source_tree_sha256"]
    assert result[part["shard_id"]]["partition"] == part


@pytest.mark.parametrize("changed", ["source", "software", "device", "lineage", "scope"])
def test_completed_part_reuse_rejects_changed_measurement_identity(reusable_part, changed):
    base, protocol, current, paths, part, path, execution, science = reusable_part
    if changed == "source": science["source_tree_sha256"] = "c" * 64
    elif changed == "software": execution["software"] = {**current, "torch": "different"}
    elif changed == "device": execution["device"] = "cuda"
    elif changed == "lineage": execution["prerequisites"] = {}
    else:
        scope = recovery._json(path / "confirmation_scope.json")
        scope["binding"]["checkpoint_hashes"] = {}
        write_json(path / "confirmation_scope.json", scope)
    with pytest.raises(ValueError, match="partition"):
        recovery._reusable_parts(base, None, None, protocol, paths, current, None, recovery.ROOT, "local-cpu")


def test_partial_part_is_never_reused(reusable_part):
    base, protocol, current, paths, _, path, _, _ = reusable_part
    (path / "COMPLETED").unlink()
    assert recovery._reusable_parts(base, None, None, protocol, paths, current, None, recovery.ROOT, "local-cpu") == {}


def test_previous_completed_part_cannot_disappear(reusable_part):
    base, protocol, current, paths, part, path, _, _ = reusable_part
    (path / "COMPLETED").unlink()
    with pytest.raises(ValueError, match="disappeared"):
        recovery._reusable_parts(base, None, {"stage_paths": {part["shard_id"]: str(path)}},
            protocol, paths, current, None, recovery.ROOT, "local-cpu")


def test_repeated_recovery_preserves_original_stage_paths_and_rechecks_prior_bridge(origin, monkeypatch):
    base, protocol, current, paths, _ = origin
    initial = recovery.create_recovery(base, protocol, current, None)
    coordinator = base.parent / "second-coordinator"; coordinator.mkdir()
    write_json(coordinator / "recovery.json", initial)
    write_json(coordinator / "protocol.json", {"unit_coordinator": True})
    workflow = dict(recovery={"manifest_path": str(coordinator / "recovery.json"),
        "sha256": file_digest(coordinator / "recovery.json"), "stage_paths": initial["stage_paths"]},
        source_tree_sha256=current["source_tree_sha256"])
    origin_file = coordinator / "frontier-workflow.json"; write_json(origin_file, workflow)
    real_origin = recovery._origin
    def declared_origin(path, protocol, site, root):
        # Unit-isolate scheduler declaration validation, exercised separately.
        if Path(path) == origin_file:
            return coordinator, origin_file, workflow, "local-cpu"
        return real_origin(path, protocol, site, root)
    monkeypatch.setattr(recovery, "_origin", declared_origin)
    repeated = recovery.create_recovery(origin_file, protocol, current, None)
    assert recovery.verify_recovery(repeated, protocol, current, None) == paths
    assert repeated["origin_workflow_path"] == str(origin_file)
    assert repeated["origin_source_tree_sha256"] == recovery.ORIGIN_SOURCE
    assert repeated["stage_paths"] == initial["stage_paths"]
    assert repeated["ancestry_workflows"] == [{"path": str(origin_file), "sha256": file_digest(origin_file)}]
    with (coordinator / "recovery.json").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="Prior recovery manifest"):
        recovery.verify_recovery(repeated, protocol, current, None)


def test_cyclic_recovery_ancestry_is_rejected(origin):
    base, protocol, current, _, _ = origin
    with pytest.raises(ValueError, match="cycle"):
        recovery.create_recovery(base, protocol, current, None, _trail=(str(base),))


def test_local_precommit_is_allowed_only_for_the_exact_pinned_source(origin, monkeypatch):
    base, protocol, current, _, _ = origin
    real_execution = recovery._execution
    def precommit(path, protocol, **kwargs):
        result = real_execution(path, protocol, **kwargs)
        result["execution"]["software"]["git_commit"] = "758cbf5a35998aa443c6dd7e38266bab4ff536b6"
        return result
    monkeypatch.setattr(recovery, "_execution", precommit)
    bridge = recovery.create_recovery(base, protocol, current, None)
    assert bridge["semantic_fingerprint"] == "verified-unit-semantic-fingerprint"


def test_local_repeated_recovery_verifies_prior_bridge_without_copying_stages(origin):
    base, protocol, current, paths, _ = origin
    initial = recovery.create_recovery(base, protocol, current, None)
    coordinator = base.parent / "local-recovery-coordinator"; coordinator.mkdir()
    write_json(coordinator / "recovery.json", initial)
    repeated = recovery.create_recovery(coordinator, protocol, current, None)
    assert recovery.verify_recovery(repeated, protocol, current, None) == paths
    assert repeated["origin_run_dir"] == str(coordinator)
    assert repeated["origin_workflow_path"] is None
    assert repeated["origin_recovery_manifest"] == {"path": str(coordinator / "recovery.json"),
        "sha256": file_digest(coordinator / "recovery.json")}
    assert repeated["stage_paths"] == initial["stage_paths"]
    assert all(not (coordinator / stage).exists() for stage in recovery.INHERITED_STAGES)
    with (coordinator / "recovery.json").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="bridge or origin"):
        recovery.verify_recovery(repeated, protocol, current, None)


def test_local_repeated_recovery_collects_only_verified_completed_parts(origin, monkeypatch):
    base, protocol, current, paths, _ = origin
    initial = recovery.create_recovery(base, protocol, current, None)
    coordinator = base.parent / "local-with-completed-part"; coordinator.mkdir()
    write_json(coordinator / "recovery.json", initial)
    real_parts = recovery._reusable_parts
    calls = []
    # Isolate the independently tested strict partition verifier. This checks
    # local inheritance passes actual original paths, software and prior bridge.
    def verified_parts(base, workflow, prior_bridge, protocol, paths, software, site, root, mode):
        if base != coordinator:
            return real_parts(base, workflow, prior_bridge, protocol, paths, software, site, root, mode)
        calls.append((paths.copy(), software.copy(), prior_bridge, mode))
        return {"confirm-part-000": {"run_dir": str(coordinator / "confirm-part-000"),
            "source_tree_sha256": software["source_tree_sha256"], "unit_verified_partition": True}}
    monkeypatch.setattr(recovery, "_reusable_parts", verified_parts)
    repeated = recovery.create_recovery(coordinator, protocol, current, None)
    assert repeated["stage_paths"]["confirm-part-000"] == str(coordinator / "confirm-part-000")
    assert recovery.allowed_stage_sources(repeated)["confirm-part-000"]["source_tree_sha256"] == current["source_tree_sha256"]
    assert calls == [(paths, current, initial, "local-cpu")]
    assert recovery.verify_recovery(repeated, protocol, current, None)["confirm-part-000"] == coordinator / "confirm-part-000"


@pytest.mark.parametrize("change", ["target_software", "source_evidence", "ambiguous_directory", "symlink"])
def test_local_repeated_recovery_cannot_bypass_original_guards(origin, change):
    base, protocol, current, paths, _ = origin
    initial = recovery.create_recovery(base, protocol, current, None)
    coordinator = base.parent / "local-invalid-recovery"; coordinator.mkdir()
    bridge_file = coordinator / "recovery.json"
    write_json(bridge_file, initial)
    if change == "target_software":
        initial["target_software"]["source_tree_sha256"] = "e" * 64
        write_json(bridge_file, initial)
    elif change == "source_evidence":
        (paths["prepare"] / "summary.json").write_text("{}")
    elif change == "ambiguous_directory":
        (coordinator / "train").mkdir()
    else:
        bridge_file.rename(coordinator / "other.json")
        bridge_file.symlink_to(coordinator / "other.json")
    with pytest.raises(ValueError):
        recovery.create_recovery(coordinator, protocol, current, None)
