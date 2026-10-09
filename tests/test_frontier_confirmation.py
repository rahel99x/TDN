"""Partition/reduction tests use explicitly manufactured timing observations.

They validate coverage and provenance, never constitute native GPU evidence.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tdn.analysis.frontier import confirmation, neural
from tdn.analysis.frontier.core import Context, check
from tdn.analysis.frontier.partition import get_partition, plan_shards, validate_partition
from tdn.analysis.frontier.protocol import build_protocol
from tdn.research.protocol import digest
from tdn.runtime.metadata import write_json


class Budget:
    def check(self):
        pass


def read(path):
    return json.loads(Path(path).read_text())


@pytest.mark.parametrize("profile,count,size", [("smoke", 2, 1), ("development", 4, 2), ("full", 6, 4)])
def test_partition_plan_only_moves_whole_parents(profile, count, size):
    protocol = build_protocol(profile)
    before = digest(protocol)
    parts = plan_shards(protocol)
    assert len(parts) == count and all(len(p["parent_ids"]) == size for p in parts)
    assert [x for p in parts for x in p["parent_ids"]] == [p["parent_id"] for p in protocol["parents"] if p["split"] == "confirmation"]
    assert len({x for p in parts for x in p["parent_ids"]}) == count * size
    assert before == digest(protocol)
    assert get_partition(protocol, parts[-1]["shard_id"]) == parts[-1]


def test_partition_planner_is_standard_library_only():
    code = "from tdn.analysis.frontier.partition import plan_shards; import sys; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).resolve().parents[1])


@pytest.mark.parametrize("change", ["id", "parents", "order", "extra"])
def test_partition_identity_cannot_be_reselected(change):
    protocol = build_protocol("full")
    part = plan_shards(protocol)[0]
    if change == "id":
        part["shard_id"] = "../../confirm"
    elif change == "parents":
        part["parent_ids"][0] = plan_shards(protocol)[1]["parent_ids"][0]
    elif change == "order":
        part["parent_ids"].reverse()
    else:
        part["models"] = ["rank1"]
    with pytest.raises(ValueError, match="deterministic"):
        validate_partition(protocol, part)


def test_full_partition_inventory_preserves_21888_endpoints_and_all_80_models():
    protocol = build_protocol("full")
    groups, endpoints = confirmation._expected(protocol, [p["parent_id"] for p in protocol["parents"] if p["split"] == "confirmation"])
    assert len(endpoints) == 21888 and len(groups) == 1152
    shard_keys = set()
    for part in plan_shards(protocol):
        paired, rows = confirmation._expected(protocol, part["parent_ids"])
        assert len(rows) == 3648 and len(paired) == 192
        assert len({key[-1] for key in rows}) == 80
        assert not shard_keys.intersection(rows)
        shard_keys.update(rows)
        assert all(len(g["model_ids"]) == (40 if g["specification"]["kind"] == "primary" else 4) for g in paired.values())
    assert shard_keys == set(endpoints)


def fabricated_timing(timing_id, expected, records, protocol):
    parent, spec = expected["parent"], expected["specification"]
    names = [name for name in expected["model_ids"] if records[name]["checkpoint_validated"]]
    seed = int(digest([parent["parent_id"], expected["grid"], expected["track"], spec["schedule_id"]])[:12], 16)
    value = {"timing_id": timing_id, "parent_id": parent["parent_id"], "grid": expected["grid"],
        "track": expected["track"], "schedule": spec["schedule"], "final_time": round(sum(spec["schedule"]), 12),
        "requested_outputs": "endpoint_only", "schedule_kind": spec["kind"], "methods": {}, "rounds": []}
    if not names:
        return value | {"status": "NO_VALIDATED_CHECKPOINTS"}
    rng = random.Random(seed)
    order = names.copy(); rng.shuffle(order)
    value.update(cold_order=order, cold_samples={name: .1 for name in names}, warmup_rounds=[],
                 repeats=protocol["timing_repeats"], warmup=protocol["timing_warmup"], seed=seed, device="cpu")
    for collection, count in (("warmup_rounds", protocol["timing_warmup"]), ("rounds", protocol["timing_repeats"])):
        for i in range(count):
            order = names.copy(); rng.shuffle(order)
            value[collection].append({"round": i, "order": order, "samples_seconds": {name: .01 * (i + 1) for name in names}})
    for name in names:
        samples = [r["samples_seconds"][name] for r in value["rounds"]]
        value["methods"][name] = dict(median_seconds=statistics.median(samples), cold_seconds=.1,
            samples_seconds=samples, raw_seconds=samples, min_seconds=min(samples), max_seconds=max(samples),
            p95_seconds=max(samples), total_seconds=sum(samples), warmup_seconds=.01,
            repeats=protocol["timing_repeats"], warmup=protocol["timing_warmup"])
    return value


@pytest.fixture
def specimen(tmp_path, monkeypatch):
    protocol = build_protocol("smoke")
    protocol.update(models=["rank1", "df"], tracks=["discrete"], grids=[8])
    catalog = {"records": [{**r, "checkpoint_sha256": digest(r), "checkpoint_validated": True} for r in neural.expected_model_specs(protocol)],
               "selected_policy_models": {"discrete": "discrete/rank1/n2/seed3600011"}}
    train = tmp_path / "train"; train.mkdir()
    write_json(train / "catalog.json", catalog)
    binding = {"catalog_sha256": confirmation.file_digest(train / "catalog.json"), "freeze_sha256": "f" * 64,
        "source_tree_sha256": "s" * 64, "prerequisites": {"train": "t" * 64}, "selection_split": "validation",
        "checkpoint_hashes": {r["model_id"]: r["checkpoint_sha256"] for r in catalog["records"]}}
    monkeypatch.setattr(confirmation, "confirmation_binding", lambda ctx: copy.deepcopy(binding))
    def make(partition=None, path=None, unavailable=False):
        path = path or tmp_path / (partition["shard_id"] if partition else "full")
        ctx = Context(protocol, "confirm", path, {"train": train}, "cpu", Budget())
        parents = [p for p in protocol["parents"] if p["split"] == "confirmation" and (not partition or p["parent_id"] in partition["parent_ids"])]
        local_catalog = copy.deepcopy(catalog)
        if unavailable:
            local_catalog["records"][0]["checkpoint_validated"] = False
        records = {r["model_id"]: r for r in local_catalog["records"]}
        identity = confirmation.scope(ctx, local_catalog, parents, partition)
        groups, endpoints = confirmation._expected(protocol, [p["parent_id"] for p in parents])
        timings = {name: fabricated_timing(name, expected, records, protocol) for name, expected in groups.items()}
        rows = []
        for key, (timing_id, spec) in endpoints.items():
            parent_id, n, track, schedule_id, model_id = key
            item = groups[timing_id]["specification"]
            row = {**spec, "parent_id": parent_id, "field_cluster": groups[timing_id]["parent"]["field_cluster"],
                "grid": n, "schedule_id": schedule_id, "schedule": item["schedule"], "schedule_kind": item["kind"],
                "requested_outputs": "endpoint_only", "diagnostic_intermediates": item["diagnostic_intermediates"],
                "final_time": round(sum(item["schedule"]), 12), "horizon": round(sum(item["schedule"]), 12),
                "status": "COMPLETED", "finite": True, "reference_accepted": True,
                "error_rms": 1.e-7, "error_max": 2.e-7, "upper_rms": 1.e-6, "upper_max": 2.e-6,
                "target_results": {f"{v:g}": True for v in protocol["targets"]}}
            if records[model_id]["checkpoint_validated"]:
                cost = timings[timing_id]["methods"][model_id]
                middle = [{"time": round(sum(item["schedule"][:i]), 12)} for i in range(1, len(item["schedule"]))] if item["diagnostic_intermediates"] else []
                row.update(timing_id=timing_id, timing=cost, cost_seconds=cost["median_seconds"],
                    median_seconds=cost["median_seconds"], cold_seconds=cost["cold_seconds"],
                    intermediates=middle, intermediate_reference_count=len(middle))
            else:
                row.update(status="UNAVAILABLE_CHECKPOINT", cost_seconds=None)
            rows.append(row)
            ctx.record(f"confirmation/{model_id}/{timing_id}", ["G3"], metrics=row,
                checks=[check("fixture", None, 1, category="math"), check("fixture-gap", None, 1)])
        write_json(path / "frozen_model_ledger.json", {"models": local_catalog["records"], "catalog_sha256": binding["catalog_sha256"], "protocol_sha256": digest(protocol)})
        write_json(path / "confirmation_plan.json", neural.confirmation_plan(protocol))
        confirmation.finish(ctx, local_catalog, rows, list(timings.values()), identity,
                            model_count=len(records), parent_count=len(parents))
        return ctx, rows, list(timings.values())
    return SimpleNamespace(protocol=protocol, catalog=catalog, binding=binding, train=train, make=make)


def test_exact_full_coverage_accepts_complete_measurements(specimen):
    ctx, rows, timings = specimen.make()
    result = neural.validate_confirmation_coverage(ctx.path, specimen.protocol)
    assert result["endpoint_rows"] == len(rows) and result["timing_groups"] == len(timings)


def test_partition_cannot_be_used_as_complete_confirmation(specimen):
    part = plan_shards(specimen.protocol)[0]
    ctx, _, _ = specimen.make(part)
    assert neural.validate_confirmation_coverage(ctx.path, specimen.protocol, part)["mode"] == "partition"
    assert read(ctx.path / "gate.json")["status"] == "PARTITION_ONLY"
    assert read(ctx.path / "comparisons.json")["frontiers"] == []
    with pytest.raises(ValueError, match="partition"):
        neural.validate_confirmation_coverage(ctx.path, specimen.protocol)


@pytest.mark.parametrize("corruption", ["missing_row", "duplicate_row", "extra_parent", "missing_group", "duplicate_group",
    "missing_method", "round_count", "round_order", "round_sample", "median", "cold", "model_hash", "model_seed",
    "intermediate", "cost", "schedule", "scope_count", "scope_incomplete"])
def test_incomplete_or_changed_measurements_rejected(specimen, corruption):
    ctx, _, _ = specimen.make()
    rows = read(ctx.path / "confirmation_rows.json")
    timings = read(ctx.path / "timing_rounds.json")
    scope = read(ctx.path / "confirmation_scope.json")
    ledger = read(ctx.path / "frozen_model_ledger.json")
    if corruption == "missing_row": rows["rows"].pop()
    elif corruption == "duplicate_row": rows["rows"].append(rows["rows"][0])
    elif corruption == "extra_parent": rows["rows"][0]["parent_id"] = "not-declared"
    elif corruption == "missing_group": timings["groups"].pop()
    elif corruption == "duplicate_group": timings["groups"].append(timings["groups"][0])
    elif corruption == "missing_method": timings["groups"][0]["methods"].pop(next(iter(timings["groups"][0]["methods"])))
    elif corruption == "round_count": timings["groups"][0]["rounds"].pop()
    elif corruption == "round_order": timings["groups"][0]["rounds"][0]["order"].reverse()
    elif corruption == "round_sample": timings["groups"][0]["rounds"][0]["samples_seconds"].pop(next(iter(timings["groups"][0]["methods"])))
    elif corruption == "median": next(iter(timings["groups"][0]["methods"].values()))["median_seconds"] = 1.e-30
    elif corruption == "cold": timings["groups"][0]["cold_order"].pop()
    elif corruption == "model_hash": ledger["models"][0]["checkpoint_sha256"] = "changed"
    elif corruption == "model_seed": rows["rows"][0]["seed"] = 42
    elif corruption == "intermediate": rows["rows"][0]["intermediates"] = []
    elif corruption == "cost": rows["rows"][0]["cost_seconds"] = 1.e-20
    elif corruption == "schedule": rows["rows"][0]["schedule"] = [.1]
    elif corruption == "scope_count": scope["endpoint_rows"] -= 1
    else: scope["status"] = "RUNNING"
    for name, value in (("confirmation_rows.json", rows), ("timing_rounds.json", timings), ("confirmation_scope.json", scope), ("frozen_model_ledger.json", ledger)):
        write_json(ctx.path / name, value)
    with pytest.raises(ValueError):
        neural.validate_confirmation_coverage(ctx.path, specimen.protocol)


def test_failed_checkpoint_arms_remain_counted_without_timing(specimen):
    ctx, rows, _ = specimen.make(unavailable=True)
    assert any(r["status"] == "UNAVAILABLE_CHECKPOINT" for r in rows)
    assert neural.validate_confirmation_coverage(ctx.path, specimen.protocol)["endpoint_rows"] == len(rows)


def test_whole_cohort_merge_preserves_every_raw_observation_and_original_score(specimen, monkeypatch, tmp_path):
    from tdn.analysis.frontier import engine
    parts = plan_shards(specimen.protocol)
    originals = [specimen.make(part) for part in parts]
    calls = []
    def verify(protocol, path, **kwargs):
        calls.append(Path(path).name)
        return {"stage": "confirm", "source_tree_sha256": specimen.binding["source_tree_sha256"],
                "prerequisites": specimen.binding["prerequisites"]}
    monkeypatch.setattr(engine, "verify_science", verify)
    for context, _, _ in originals:
        write_json(context.path / "science_manifest.json", verify(specimen.protocol, context.path))
    calls.clear()
    ctx = Context(specimen.protocol, "confirm", tmp_path / "merged", {"train": specimen.train}, "cpu", Budget())
    result = neural.confirm(ctx, shard_dirs={part["shard_id"]: original.path for part, (original, _, _) in zip(parts, originals)})
    expected_rows = [r for _, rows, _ in originals for r in rows]
    expected_timing = [r for _, _, groups in originals for r in groups]
    assert read(ctx.path / "confirmation_rows.json")["rows"] == expected_rows
    assert read(ctx.path / "timing_rounds.json")["groups"] == expected_timing
    assert ctx.rows == [r for context, _, _ in originals for r in context.rows]
    comparison = read(ctx.path / "comparisons.json")
    assert comparison["frontiers"] == neural.matched_comparisons(expected_rows, specimen.protocol)
    assert comparison["paired_field_summaries"] == neural.paired_field_summary(expected_rows, bootstrap=specimen.protocol["bootstrap_replicates"])
    assert read(ctx.path / "gate.json") == neural.confirmation_gate(specimen.catalog, expected_rows, comparison["frontiers"], specimen.protocol)
    assert result["independent_fields"] == 2 and result["confirmation_scope"] == "merged"
    assert calls == [part["shard_id"] for part in parts]


@pytest.mark.parametrize("fault", ["missing", "duplicate_path", "foreign_source", "foreign_binding", "unsealed", "swapped_parts", "changed_ledger"])
def test_merge_rejects_missing_mixed_unsealed_or_swapped_parts(specimen, monkeypatch, tmp_path, fault):
    from tdn.analysis.frontier import engine
    parts = plan_shards(specimen.protocol)
    originals = [specimen.make(part)[0] for part in parts]
    mapping = {p["shard_id"]: c.path for p, c in zip(parts, originals)}
    manifest = {"stage": "confirm", "source_tree_sha256": specimen.binding["source_tree_sha256"], "prerequisites": specimen.binding["prerequisites"]}
    for c in originals:
        write_json(c.path / "science_manifest.json", manifest)
    def verify(protocol, path, **kwargs):
        if fault == "unsealed":
            raise ValueError("unsealed partition")
        return manifest | ({"source_tree_sha256": "other"} if fault == "foreign_source" else {})
    monkeypatch.setattr(engine, "verify_science", verify)
    if fault == "missing": mapping.pop(parts[0]["shard_id"])
    elif fault == "duplicate_path": mapping[parts[1]["shard_id"]] = originals[0].path
    elif fault == "swapped_parts": mapping = {parts[0]["shard_id"]: originals[1].path, parts[1]["shard_id"]: originals[0].path}
    elif fault == "foreign_binding":
        value = read(originals[0].path / "confirmation_scope.json"); value["binding"]["freeze_sha256"] = "other"
        write_json(originals[0].path / "confirmation_scope.json", value)
    elif fault == "changed_ledger":
        (originals[0].path / "rows.jsonl").write_text("")
    ctx = Context(specimen.protocol, "confirm", tmp_path / "reject", {"train": specimen.train}, "cpu", Budget())
    with pytest.raises(ValueError):
        neural.confirm(ctx, shard_dirs=mapping)
    assert not (ctx.path / "gate.json").exists() and not ctx.rows


def test_complete_group_checkpoint_is_atomic_forensic_evidence(tmp_path):
    timing = {"timing_id": "parent/N8/discrete/schedule-0", "rounds": [{"order": ["rank1", "df"]}]}
    rows = [{"model_id": "rank1"}, {"model_id": "df"}]
    confirmation.write_completed_group(tmp_path, timing, rows)
    files = list((tmp_path / "complete-groups").glob("*.json"))
    assert len(files) == 1 and not list(tmp_path.rglob("*.partial"))
    value = read(files[0])
    assert value["timing"] == timing and value["rows"] == rows
    assert "not a sealed scientific stage" in value["scope"]


@pytest.mark.parametrize("changed", [None, "source", "prerequisites", "device"])
def test_science_seal_rejects_rehashed_confirmation_binding_drift(specimen, monkeypatch, changed):
    from tdn.analysis.frontier import engine
    from tdn.analysis.frontier.core import write_reviews
    ctx, _, _ = specimen.make()
    # This small manufactured inventory deliberately uses a unit protocol.
    # All artifact sealing, scope checks and row validation remain real.
    monkeypatch.setattr(engine, "validate_protocol", lambda p: p)
    identity = read(ctx.path / "confirmation_scope.json")
    if changed == "source": identity["binding"]["source_tree_sha256"] = "other-source"
    elif changed == "prerequisites": identity["binding"]["prerequisites"] = {}
    elif changed == "device": identity["measurement_device"] = "cuda"
    write_json(ctx.path / "confirmation_scope.json", identity)
    write_json(ctx.path / "protocol.json", ctx.protocol)
    write_json(ctx.path / "summary.json", {"schema": ctx.protocol["schema"], "stage": "confirm", "status": "COMPLETED",
        "protocol_sha256": digest(ctx.protocol), "source_tree_sha256": specimen.binding["source_tree_sha256"],
        "experiment_count": len(ctx.rows), "device": "cpu"})
    (ctx.path / "summary.txt").write_text("Manufactured test evidence only\n")
    write_reviews(ctx.path, ctx.rows)
    engine._seal(ctx.protocol, ctx.path, "confirm", specimen.binding["prerequisites"], specimen.binding["source_tree_sha256"])
    if changed:
        with pytest.raises(ValueError, match="scope source, prerequisites or device"):
            engine.verify_science(ctx.protocol, ctx.path)
    else:
        assert engine.verify_science(ctx.protocol, ctx.path)["stage"] == "confirm"


def test_training_and_model_prefix_unchanged_from_timeout_run():
    path = Path(neural.__file__)
    prefix = path.read_text().split("def confirm(", 1)[0]
    assert hashlib.sha256(prefix.encode()).hexdigest() == "d0e3ad7fe39f35a75cd9057fc1bd8a9dc56a6ce1e32ecab0f01f7544b80a7d87"


@pytest.fixture
def trained_fixture(tmp_path, monkeypatch):
    """Tiny actual model execution with manufactured labels, not a teacher study."""
    import torch
    from tdn.analysis.frontier import data
    from tdn.analysis.frontier.models import make_model
    from tdn.research.experiment import horizon_key
    protocol = build_protocol("smoke")
    protocol.update(models=["rank1", "df"], tracks=["discrete"], grids=[8])
    horizons = sorted(set(protocol["train_horizons"] + protocol["validation_horizons"] +
        [round(sum(s[:i]), 12) for s in protocol["confirm_schedules"] for i in range(1, len(s) + 1)]))
    banks = {}
    base = make_model("df", "discrete", protocol["model_config"])
    for split in ("train", "validation", "confirmation"):
        banks[split] = []
        for declaration in (p for p in protocol["parents"] if p["split"] == split):
            u = data.field_state(declaration, 8)
            eq, geom = neural.eq_geom(declaration, 8)
            parent = {**declaration, "states": {"8": u}, "references": {}}
            for horizon in horizons:
                parent["references"][f"discrete:8:{horizon_key(horizon)}"] = {
                    "state": base(u, horizon, eq, geom), "accepted": True,
                    "uncertainty_rms": 1.e-10, "uncertainty_max_bound": 1.e-10}
            banks[split].append(parent)
    accesses = []
    def load(ctx, split):
        accesses.append(split)
        return banks[split]
    monkeypatch.setattr(data, "load_split", load)
    trained = Context(protocol, "train", tmp_path / "train", {}, "cpu", Budget())
    neural.train(trained)
    return SimpleNamespace(protocol=protocol, trained=trained, banks=banks, accesses=accesses)


def test_real_partition_calls_preserve_predictions_and_never_retrain(trained_fixture, tmp_path):
    setup = trained_fixture
    monolith = Context(setup.protocol, "confirm", tmp_path / "monolithic", {"train": setup.trained.path}, "cpu", Budget())
    neural.confirm(monolith)
    neural.validate_confirmation_coverage(monolith.path, setup.protocol)
    def scientific(rows):
        return {(r["parent_id"], r["model_id"], r["schedule_id"]):
            {k: r[k] for k in ("error_rms", "error_max", "upper_rms", "upper_max", "target_results", "intermediates")}
            for r in rows}
    all_rows = []
    setup.accesses.clear()
    for part in plan_shards(setup.protocol):
        ctx = Context(setup.protocol, "confirm", tmp_path / part["shard_id"], {"train": setup.trained.path}, "cpu", Budget())
        neural.confirm(ctx, partition=part)
        covered = neural.validate_confirmation_coverage(ctx.path, setup.protocol, part)
        assert covered["parents"] == 1
        rows = read(ctx.path / "confirmation_rows.json")["rows"]
        assert {r["parent_id"] for r in rows} == set(part["parent_ids"])
        assert len(list((ctx.path / "complete-groups").glob("*.json"))) == covered["timing_groups"]
        all_rows.extend(rows)
    assert scientific(all_rows) == scientific(read(monolith.path / "confirmation_rows.json")["rows"])
    assert setup.accesses == ["confirmation", "confirmation"]


def test_real_timeout_retains_complete_group_and_cannot_validate(trained_fixture, tmp_path):
    setup = trained_fixture
    path = tmp_path / "interrupted"
    class InterruptAfterGroup:
        def check(self):
            if list((path / "complete-groups").glob("*.json")):
                raise TimeoutError("bounded test interruption after one complete paired group")
    ctx = Context(setup.protocol, "confirm", path, {"train": setup.trained.path}, "cpu", InterruptAfterGroup())
    part = plan_shards(setup.protocol)[0]
    with pytest.raises(TimeoutError, match="complete paired group"):
        neural.confirm(ctx, partition=part)
    groups = list((path / "complete-groups").glob("*.json"))
    assert len(groups) == 1
    saved = read(groups[0])
    assert {r["family"] for r in saved["rows"]} == {"rank1", "df"}
    assert len(saved["timing"]["rounds"]) == setup.protocol["timing_repeats"]
    assert len(ctx.rows) == 2 and read(path / "confirmation_scope.json")["status"] == "RUNNING"
    assert not (path / "gate.json").exists()
    with pytest.raises(ValueError, match="incomplete"):
        neural.validate_confirmation_coverage(path, setup.protocol, part)
