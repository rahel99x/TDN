"""Scientific provenance, gate failure semantics and dispatch, with unit evidence.

The dispatch stubs below produce explicitly labeled unit records, not numerical
results. They exercise the real ledger, source guard, seals and chained lineage.
"""
import json
import sys
from types import ModuleType

import pytest

from tdn.analysis.frontier import engine
from tdn.analysis.frontier.core import check
from tdn.analysis.frontier.protocol import STAGES, build_protocol
from tdn.research.protocol import digest, file_digest
from tdn.runtime.metadata import write_json

SOURCE = "9" * 64


@pytest.fixture
def handlers(monkeypatch):
    import tdn.analysis.frontier as package
    monkeypatch.setattr(engine, "software_metadata", lambda: {"source_tree_sha256": SOURCE})
    calls = []

    def record(ctx, *, confirmation=None):
        calls.append((ctx.stage, confirmation, tuple(ctx.prerequisites)))
        blocked = ctx.stage == "policy"
        ctx.record("unit-dispatch-" + ctx.stage, ["G5" if blocked else "G1"],
            checks=[check("unit-math", 1, 1, "eq", category="math"),
                    check("unit-gap", None if blocked else 1, 1, "eq", category="gap")],
            metrics={"unit_test_evidence_only": True}, status="BLOCKED" if blocked else "COMPLETED")
        write_json(ctx.path / "unit-payload.json", {"stage": ctx.stage, "confirmation": confirmation})
        return {"dispatch_evidence_only": True, "blocked": blocked}

    modules = {}
    for name in ("measurement", "screening", "data", "neural", "scaling", "policy", "report"):
        module = ModuleType("tdn.analysis.frontier." + name)
        for entry in ("run", "run_audit", "prepare", "train", "confirm"):
            setattr(module, entry, record)
        if name == "policy":
            module.validate_policy_artifacts = lambda path: json.loads((path / "unit-payload.json").read_text())
        monkeypatch.setitem(sys.modules, module.__name__, module)
        monkeypatch.setattr(package, name, module, raising=False)
        modules[name] = module
    return modules, calls


def audit(tmp_path, handlers):
    protocol = build_protocol("smoke")
    directory = tmp_path / "audit"
    summary = engine.run_stage(protocol, "audit", directory)
    return protocol, directory, summary


def rehash(directory):
    manifest = json.loads((directory / engine.MANIFEST).read_text())
    manifest["artifacts"] = {name: file_digest(directory / name) for name in manifest["artifacts"]}
    write_json(directory / engine.MANIFEST, manifest)
    (directory / "COMPLETED").write_text(digest(manifest) + "\n")


def test_all_nine_stages_preserve_lineage_and_blocked_policy_evidence(tmp_path, handlers):
    _, calls = handlers
    protocol = build_protocol("smoke")
    prior, hashes = {}, {}
    for stage in STAGES:
        path = tmp_path / stage
        result = engine.run_stage(protocol, stage, path, prerequisites=prior)
        assert result["status"] == "COMPLETED"
        seal = engine.verify_science(protocol, path)
        assert seal["source_tree_sha256"] == SOURCE
        if stage != "report":
            assert seal["prerequisites"] == hashes
        rows = json.loads((path / "rows.json").read_text())["rows"]
        if stage == "policy":
            assert rows[0]["status"] == "BLOCKED"
            assert rows[0]["assessment"]["verdict"] == "NA"
            assert result["details"]["blocked"] is True
        hashes[stage] = file_digest(path / engine.MANIFEST)
        prior[stage] = path
    assert [stage for stage, _, _ in calls] == list(STAGES)
    assert dict((stage, value) for stage, value, _ in calls)["prepare"] is False
    assert dict((stage, value) for stage, value, _ in calls)["confirm_prepare"] is True
    assert calls[-1][2] == STAGES[:-1]


@pytest.mark.parametrize("category,blocks", [("correctness", True), ("math", False), ("gap", False), ("utility", False)])
def test_failed_structure_blocks_but_negative_science_is_a_completed_outcome(tmp_path, handlers, category, blocks):
    modules, _ = handlers
    def bad(ctx):
        ctx.record("unit-required-failure", ["G1"], checks=[check("required-" + category, 2, 1, category=category)])
    modules["measurement"].run_audit = bad
    path = tmp_path / "audit"
    if blocks:
        with pytest.raises(RuntimeError, match="structural correctness"):
            engine.run_stage(build_protocol("smoke"), "audit", path)
        assert not (path / "COMPLETED").exists()
        assert not (path / engine.MANIFEST).exists()
    else:
        result = engine.run_stage(build_protocol("smoke"), "audit", path)
        assert result["status"] == "COMPLETED" and result["correctness_failures"] == 0
        engine.verify_science(build_protocol("smoke"), path)
    row = json.loads((path / "rows.jsonl").read_text().splitlines()[0])
    assert row["assessment"]["verdict"] == "BAD"


@pytest.mark.parametrize("artifact", ["review.csv", "rows.jsonl", "COMPLETED", "unit-payload.json"])
def test_post_completion_mutation_is_rejected(tmp_path, handlers, artifact):
    protocol, path, _ = audit(tmp_path, handlers)
    with (path / artifact).open("a") as stream:
        stream.write("changed")
    with pytest.raises(ValueError):
        engine.verify_science(protocol, path)


@pytest.mark.parametrize("name", ["new-experiment.json", "nested/stage.json", "nested/COMPLETED"])
def test_unlisted_files_cannot_hide_under_wrapper_filenames(tmp_path, handlers, name):
    protocol, path, _ = audit(tmp_path, handlers)
    file = path / name
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text("{}")
    with pytest.raises(ValueError, match="Unlisted"):
        engine.verify_science(protocol, path)


@pytest.mark.parametrize("key,value", [("source_tree_sha256", "0" * 64), ("protocol_sha256", "0" * 64), ("schema", "tdn.roadmap/v1")])
def test_rehashed_summary_still_must_match_declared_provenance(tmp_path, handlers, key, value):
    protocol, path, _ = audit(tmp_path, handlers)
    summary = json.loads((path / "summary.json").read_text())
    summary[key] = value
    write_json(path / "summary.json", summary)
    rehash(path)
    with pytest.raises(ValueError, match="complete scientific execution"):
        engine.verify_science(protocol, path)


def test_rehashed_protocol_cannot_substitute_another_declaration(tmp_path, handlers):
    protocol, path, _ = audit(tmp_path, handlers)
    altered = {**protocol, "profile": "development"}
    write_json(path / "protocol.json", altered)
    rehash(path)
    with pytest.raises(ValueError, match="Stored scientific protocol"):
        engine.verify_science(protocol, path)


def test_mixed_prerequisite_lineages_are_rejected_even_with_same_source(tmp_path, handlers):
    protocol, original, _ = audit(tmp_path, handlers)
    screen = tmp_path / "screen"
    engine.run_stage(protocol, "screen", screen, prerequisites={"audit": original})
    alternate = tmp_path / "alternate-audit"
    engine.run_stage(protocol, "audit", alternate)
    assert file_digest(alternate / engine.MANIFEST) != file_digest(original / engine.MANIFEST)
    with pytest.raises(ValueError, match="different frontier lineages"):
        engine.run_stage(protocol, "prepare", tmp_path / "prepare", prerequisites={"audit": alternate, "screen": screen})


def test_source_or_protocol_drift_mid_execution_retains_unsealed_failure(tmp_path, handlers, monkeypatch):
    modules, _ = handlers
    def mutate(ctx):
        ctx.record("before-drift", ["G1"], checks=[check("measured", 1, 1, "eq", category="math")])
        monkeypatch.setattr(engine, "software_metadata", lambda: {"source_tree_sha256": "0" * 64})
    modules["measurement"].run_audit = mutate
    path = tmp_path / "audit"
    with pytest.raises(ValueError, match="source changed"):
        engine.run_stage(build_protocol("smoke"), "audit", path)
    assert not (path / "COMPLETED").exists()
    assert json.loads((path / "summary.json").read_text())["status"] == "FAILED"
    assert len((path / "rows.jsonl").read_text().splitlines()) == 1


def test_protocol_drift_mid_execution_cannot_be_sealed(tmp_path, handlers):
    modules, _ = handlers
    modules["measurement"].run_audit = lambda ctx: write_json(ctx.path / "protocol.json", {})
    path = tmp_path / "audit"
    with pytest.raises(ValueError, match="protocol changed"):
        engine.run_stage(build_protocol("smoke"), "audit", path)
    assert not (path / engine.MANIFEST).exists()


def test_interruption_keeps_partial_rows_and_never_writes_success(tmp_path, handlers):
    modules, _ = handlers
    def stop(ctx):
        ctx.record("before-stop", ["G1"], checks=[check("measured", 1, 1, "eq", category="math")])
        raise TimeoutError("bounded unit interruption")
    modules["measurement"].run_audit = stop
    path = tmp_path / "audit"
    with pytest.raises(TimeoutError):
        engine.run_stage(build_protocol("smoke"), "audit", path)
    result = json.loads((path / "summary.json").read_text())
    assert result["status"] == "INTERRUPTED" and result["experiment_count"] == 1
    assert not (path / engine.MANIFEST).exists()
    assert not (path / "COMPLETED").exists()


@pytest.mark.parametrize("stage", STAGES)
def test_direct_full_engine_calls_cannot_bypass_native_allocation(tmp_path, monkeypatch, stage):
    monkeypatch.delenv("TDN_EXECUTION_MODE", raising=False)
    with pytest.raises(ValueError, match="native allocated Fedora"):
        engine.run_stage(build_protocol("full"), stage, tmp_path / stage)
    assert not (tmp_path / stage).exists()


def test_direct_full_gpu_stage_cannot_silently_use_cpu(tmp_path, monkeypatch):
    monkeypatch.setenv("TDN_EXECUTION_MODE", "desktop-slurm")
    with pytest.raises(ValueError, match="frozen CPU/GPU device"):
        engine.run_stage(build_protocol("full"), "train", tmp_path / "train", device="cpu")


def test_successful_run_is_never_overwritten(tmp_path, handlers):
    protocol, path, _ = audit(tmp_path, handlers)
    with pytest.raises(FileExistsError, match="Preserve"):
        engine.run_stage(protocol, "audit", path)


def test_linked_artifact_and_ancestor_indirection_are_rejected(tmp_path, handlers):
    protocol, path, _ = audit(tmp_path, handlers)
    (path / "new-link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        engine.verify_science(protocol, path)
    link = tmp_path / "stage-link"
    link.symlink_to(path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink indirection"):
        engine.run_stage(protocol, "audit", link / "new-stage")
