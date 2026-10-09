"""Confirmation execution scopes, exact coverage and whole-cohort aggregation.

Partitioning never splits a randomized timing group or averages a shard score.
Only sealed, complete partitions can contribute to the canonical confirmation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import random
import statistics

from tdn.research.protocol import digest, file_digest
from tdn.runtime.metadata import software_metadata, write_json
from .partition import plan_shards, validate_partition

SCOPE_SCHEMA = "tdn.frontier-confirmation-scope/v1"


def _read(path):
    return json.loads(Path(path).read_text())


def confirmation_binding(ctx):
    from .data import verify_frozen_training
    frozen = verify_frozen_training(ctx)
    return {**frozen, "source_tree_sha256": software_metadata()["source_tree_sha256"],
            "prerequisites": {name: file_digest(Path(path) / "science_manifest.json")
                for name, path in ctx.prerequisites.items()
                if (Path(path) / "science_manifest.json").is_file()}}


def scope(ctx, catalog, parents, partition=None, *, mode=None):
    return {"schema": SCOPE_SCHEMA, "status": "RUNNING", "mode": mode or ("partition" if partition else "full"),
        "partition": partition, "protocol_sha256": digest(ctx.protocol),
        "parent_ids": [p["parent_id"] for p in parents],
        "model_ids": [r["model_id"] for r in catalog["records"]],
        "binding": confirmation_binding(ctx), "measurement_device": ctx.device,
        "timing_scope": "all validated model/seed/data-fraction arms remain in each randomized paired group",
        "selection": "execution partition only; no scientific setting or confirmation parent removed"}


def _expected(protocol, parent_ids):
    from .neural import confirmation_plan, expected_model_specs
    specs = expected_model_specs(protocol)
    groups, endpoints = {}, {}
    for parent in protocol["parents"]:
        if parent["parent_id"] not in parent_ids:
            continue
        for n in protocol["grids"]:
            for track in protocol["tracks"]:
                for item in confirmation_plan(protocol)["schedules"]:
                    if not item["families"]:
                        continue
                    timing_id = f"{parent['parent_id']}/N{n}/{track}/{item['schedule_id']}"
                    members = [r for r in specs if r["track"] == track and r["family"] in item["families"]]
                    groups[timing_id] = dict(parent=parent, grid=n, track=track, specification=item,
                        model_ids=[r["model_id"] for r in members])
                    for spec in members:
                        endpoints[(parent["parent_id"], n, track, item["schedule_id"], spec["model_id"])] = (timing_id, spec)
    return groups, endpoints


def _positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _validate_timing(group, expected, records, protocol):
    item, parent = expected["specification"], expected["parent"]
    if any(group.get(key) != value for key, value in {
        "parent_id": parent["parent_id"], "grid": expected["grid"], "track": expected["track"],
        "schedule": item["schedule"], "final_time": round(sum(item["schedule"]), 12),
        "requested_outputs": "endpoint_only", "schedule_kind": item["kind"]}.items()):
        raise ValueError("Confirmation timing group identity or schedule differs")
    names = [name for name in expected["model_ids"] if records[name]["checkpoint_validated"]]
    if set(group.get("methods", {})) != set(names):
        raise ValueError("Confirmation timing group omits or adds a validated model")
    if not names:
        if group.get("status") != "NO_VALIDATED_CHECKPOINTS" or group.get("rounds") != []:
            raise ValueError("Unavailable checkpoint timing cannot represent measured work")
        return
    repeats, warmup = protocol["timing_repeats"], protocol["timing_warmup"]
    seed = int(digest([parent["parent_id"], expected["grid"], expected["track"], item["schedule_id"]])[:12], 16)
    if (group.get("repeats") != repeats or group.get("warmup") != warmup or group.get("seed") != seed
            or len(group.get("rounds", [])) != repeats or len(group.get("warmup_rounds", [])) != warmup):
        raise ValueError("Confirmation timing repeats, warmup or seed changed")
    rng = random.Random(seed)
    expected_order = names.copy(); rng.shuffle(expected_order)
    if group.get("cold_order") != expected_order or set(group.get("cold_samples", {})) != set(names):
        raise ValueError("Confirmation first-invocation order or coverage differs")
    if any(not _positive(v) for v in group["cold_samples"].values()):
        raise ValueError("Confirmation first-invocation timing must be positive and finite")
    for rounds in (group["warmup_rounds"], group["rounds"]):
        for index, observed in enumerate(rounds):
            order = names.copy(); rng.shuffle(order)
            if (observed.get("round") != index or observed.get("order") != order
                    or set(observed.get("samples_seconds", {})) != set(names)
                    or any(not _positive(v) for v in observed["samples_seconds"].values())):
                raise ValueError("Confirmation raw timing round is incomplete or not the declared paired order")
    for name in names:
        values = [r["samples_seconds"][name] for r in group["rounds"]]
        ordered = sorted(values)
        expected_cost = dict(median_seconds=statistics.median(values), cold_seconds=group["cold_samples"][name],
            samples_seconds=values, raw_seconds=values, min_seconds=min(values), max_seconds=max(values),
            p95_seconds=ordered[max(0, math.ceil(.95 * len(values)) - 1)], total_seconds=sum(values),
            warmup_seconds=sum(r["samples_seconds"][name] for r in group["warmup_rounds"]), repeats=repeats, warmup=warmup)
        if group["methods"][name] != expected_cost:
            raise ValueError("Confirmation timing summary differs from retained raw rounds")


def validate_confirmation_coverage(path, protocol, partition=None):
    """Verify a complete exact endpoint inventory, including failed model arms.

This supplements the stage byte seal. A partition can only validate when its
explicit descriptor is supplied; it can never satisfy a full-stage prerequisite.
"""
    from .neural import confirmation_plan, expected_model_specs
    path = Path(path)
    partition = validate_partition(protocol, partition) if partition is not None else None
    parents = partition["parent_ids"] if partition else [p["parent_id"] for p in protocol["parents"] if p["split"] == "confirmation"]
    identity = _read(path / "confirmation_scope.json")
    if (identity.get("schema") != SCOPE_SCHEMA or identity.get("status") != "COMPLETED"
            or identity.get("protocol_sha256") != digest(protocol) or identity.get("partition") != partition
            or identity.get("parent_ids") != parents
            or identity.get("mode") not in (("partition",) if partition else ("full", "merged"))):
        raise ValueError("Confirmation scope is incomplete or a partition is being used as full evidence")
    specs = expected_model_specs(protocol)
    if identity.get("model_ids") != [s["model_id"] for s in specs]:
        raise ValueError("Confirmation declared model coverage differs")
    ledger = _read(path / "frozen_model_ledger.json")
    records = {r["model_id"]: r for r in ledger["models"]}
    binding = identity.get("binding", {})
    if (len(records) != len(ledger["models"]) or set(records) != {r["model_id"] for r in specs}
            or ledger.get("protocol_sha256") != digest(protocol)
            or ledger.get("catalog_sha256") != binding.get("catalog_sha256")
            or {k: v["checkpoint_sha256"] for k, v in records.items()} != binding.get("checkpoint_hashes")):
        raise ValueError("Confirmation frozen model binding differs")
    for spec in specs:
        if any(records[spec["model_id"]].get(key) != value for key, value in spec.items()):
            raise ValueError("Confirmation model identity differs from the frozen scientific inventory")
    if _read(path / "confirmation_plan.json") != confirmation_plan(protocol):
        raise ValueError("Confirmation schedules differ from the immutable scientific plan")
    groups, expected_rows = _expected(protocol, parents)
    rows = _read(path / "confirmation_rows.json")["rows"]
    timings = _read(path / "timing_rounds.json")["groups"]
    observed_groups = {r["timing_id"]: r for r in timings}
    if len(observed_groups) != len(timings) or set(observed_groups) != set(groups):
        raise ValueError("Confirmation has duplicated, missing or unexpected paired timing groups")
    for key, group in observed_groups.items():
        _validate_timing(group, groups[key], records, protocol)
    seen = set()
    for row in rows:
        key = tuple(row.get(k) for k in ("parent_id", "grid", "track", "schedule_id", "model_id"))
        if key not in expected_rows or key in seen:
            raise ValueError("Confirmation has duplicate or unexpected model endpoints")
        seen.add(key)
        timing_id, spec = expected_rows[key]
        group = groups[timing_id]; item = group["specification"]
        if (any(row.get(k) != v for k, v in spec.items())
                or row.get("field_cluster") != group["parent"]["field_cluster"]
                or row.get("schedule") != item["schedule"]
                or row.get("schedule_kind") != item["kind"]
                or row.get("requested_outputs") != "endpoint_only"
                or row.get("diagnostic_intermediates") != item["diagnostic_intermediates"]
                or row.get("final_time") != round(sum(item["schedule"]), 12)
                or row.get("horizon") != row["final_time"]):
            raise ValueError("Confirmation endpoint identity or declared output differs")
        if not records[row["model_id"]]["checkpoint_validated"]:
            if row.get("status") != "UNAVAILABLE_CHECKPOINT" or row.get("cost_seconds") is not None:
                raise ValueError("Unvalidated checkpoint cannot be reported as measured confirmation")
            continue
        cost = observed_groups[timing_id]["methods"][row["model_id"]]
        if (row.get("timing_id") != timing_id or row.get("timing") != cost
                or any(row.get(k) != cost[k] for k in ("median_seconds", "cold_seconds"))
                or row.get("cost_seconds") != cost["median_seconds"]
                or row.get("status") not in ("COMPLETED", "NUMERICAL_FAILURE")):
            raise ValueError("Confirmation endpoint cost differs from its original paired timing")
        middle = row.get("intermediates", [])
        times = [round(sum(item["schedule"][:i]), 12) for i in range(1, len(item["schedule"]))]
        required_times = times if item["diagnostic_intermediates"] and row.get("finite") else []
        if ([r.get("time") for r in middle] != required_times
                or row.get("intermediate_reference_count") != len(required_times)):
            raise ValueError("Confirmation omitted a prescribed intermediate teacher audit")
    if seen != set(expected_rows):
        raise ValueError("Confirmation endpoint inventory is incomplete")
    if identity.get("endpoint_rows") != len(rows) or identity.get("timing_groups") != len(timings):
        raise ValueError("Confirmation completion counts differ from its exact coverage")
    return {"endpoint_rows": len(rows), "timing_groups": len(timings), "parents": len(parents),
            "mode": identity["mode"], "partition": partition, "binding": binding}


def write_completed_group(path, timing, rows):
    """Atomic forensic checkpoint only; interrupted jobs are never resumed here."""
    path = Path(path) / "complete-groups" / (digest(timing["timing_id"]) + ".json")
    write_json(path, {"schema": "tdn.frontier-confirmation-group/v1", "timing_id": timing["timing_id"],
                      "timing": timing, "rows": rows,
                      "scope": "complete paired group; not a sealed scientific stage or a resumable checkpoint"})


def finish(ctx, catalog, rows, timings, identity, *, model_count, parent_count):
    from .neural import matched_comparisons, paired_field_summary, confirmation_gate
    path = Path(ctx.path)
    write_json(path / "confirmation_rows.json", {"rows": rows,
        "independence_unit": "field_cluster; seeds, grids, schedules and physics are paired"})
    write_json(path / "timing_rounds.json", {"groups": timings,
        "cold_definition": "first observed invocation, not guaranteed process-cold startup"})
    if identity["mode"] == "partition":
        write_json(path / "gate.json", {"schema": "tdn.frontier-confirmation-part/v1", "status": "PARTITION_ONLY",
            "tracks": {}, "scope": "No cohort decision is computed before every declared partition completes"})
        write_json(path / "comparisons.json", {"status": "PARTITION_ONLY", "frontiers": [],
            "paired_field_summaries": [], "scope": "Whole-cohort comparisons are deferred to exact complete merge"})
    else:
        comparison = matched_comparisons(rows, ctx.protocol)
        write_json(path / "comparisons.json", {
            "scope": "reference-informed posthoc frontiers, not deployable adaptive decisions",
            "horizon_matching": "parent/grid/track/final_time/RMS target/maximum target", "frontiers": comparison,
            "paired_field_summaries": paired_field_summary(rows, bootstrap=int(ctx.protocol.get("bootstrap_replicates", 1000))),
            "paper_reproduction": False, "independence_unit": "field_cluster"})
        write_json(path / "gate.json", confirmation_gate(catalog, rows, comparison, ctx.protocol))
    identity.update(status="COMPLETED", endpoint_rows=len(rows), timing_groups=len(timings))
    write_json(path / "confirmation_scope.json", identity)
    partial = path / "confirmation_rows.partial.jsonl"
    if partial.exists():
        partial.unlink()
    return {"models": model_count, "declared_models": len(catalog["records"]), "endpoint_rows": len(rows),
        "independent_fields": parent_count, "failed_or_missing_endpoints": sum(r["status"] != "COMPLETED" for r in rows),
        "confirmation_scope": identity["mode"], "partition": identity["partition"],
        "scope": "frozen project controls; descriptive cluster uncertainty; no FNO paper superiority claim"}


def merge(ctx, shard_dirs):
    """Read only sealed exact partitions and recompute the original estimands."""
    from .core import validate_row
    from .engine import verify_science
    from .neural import confirmation_plan
    expected = plan_shards(ctx.protocol)
    if not isinstance(shard_dirs, dict) or set(shard_dirs) != {p["shard_id"] for p in expected}:
        raise ValueError("Confirmation merge requires every exact declared partition once")
    if len({Path(v).resolve() for v in shard_dirs.values()}) != len(expected):
        raise ValueError("Confirmation partition directories must be distinct")
    binding = confirmation_binding(ctx)
    catalog = _read(Path(ctx.prerequisites["train"]) / "catalog.json")
    parents = [p for p in ctx.protocol["parents"] if p["split"] == "confirmation"]
    identity = scope(ctx, catalog, parents, mode="merged")
    rows, timings, ledger_rows, provenance = [], [], [], []
    for part in expected:
        ctx.budget.check()
        base = Path(shard_dirs[part["shard_id"]])
        manifest = verify_science(ctx.protocol, base, confirmation_partition=part)
        if manifest.get("stage") != "confirm" or manifest.get("source_tree_sha256") != binding["source_tree_sha256"]:
            raise ValueError("Confirmation partition source or stage differs")
        checked = validate_confirmation_coverage(base, ctx.protocol, part)
        part_scope = _read(base / "confirmation_scope.json")
        if (checked["binding"] != binding or part_scope.get("measurement_device") != ctx.device
                or manifest.get("prerequisites") != binding["prerequisites"]):
            raise ValueError("Confirmation partitions belong to different source, device, models or prerequisite lineages")
        members = _read(base / "confirmation_rows.json")["rows"]
        logs = [validate_row(json.loads(line)) for line in (base / "rows.jsonl").read_text().splitlines() if line]
        expected_logs = {f"confirmation/{r['model_id']}/{r['parent_id']}/N{r['grid']}/{r['track']}/{r['schedule_id']}": r for r in members}
        if (len(logs) != len(members) or len({r["experiment_id"] for r in logs}) != len(logs)
                or {r["experiment_id"] for r in logs} != set(expected_logs)):
            raise ValueError("Confirmation experiment ledger differs from the exact endpoint inventory")
        for row in logs:
            if (row["stage"] != "confirm" or row["protocol_sha256"] != digest(ctx.protocol)
                    or row["device"] != part_scope["measurement_device"]
                    or any(row["metrics"].get(k) != v for k, v in expected_logs[row["experiment_id"]].items())):
                raise ValueError("Confirmation experiment metrics differ from retained original observations")
        rows.extend(members); timings.extend(_read(base / "timing_rounds.json")["groups"]); ledger_rows.extend(logs)
        provenance.append({**part, "science_manifest_sha256": file_digest(base / "science_manifest.json"),
            "confirmation_scope_sha256": file_digest(base / "confirmation_scope.json"),
            "measurement_device": part_scope["measurement_device"]})
    path = Path(ctx.path)
    if ctx.rows or ctx.identities:
        raise ValueError("Confirmation aggregate must begin with an empty experiment ledger")
    # Original per-experiment CUDA device, wall cost, check results and hashes
    # remain untouched. Aggregation overhead belongs to the new stage summary.
    with (path / "rows.jsonl").open("a") as handle:
        for row in ledger_rows:
            handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    ctx.rows.extend(ledger_rows); ctx.identities.update(row["experiment_id"] for row in ledger_rows)
    write_json(path / "frozen_model_ledger.json", {"models": catalog["records"],
        "catalog_sha256": binding["catalog_sha256"], "protocol_sha256": digest(ctx.protocol)})
    write_json(path / "confirmation_plan.json", confirmation_plan(ctx.protocol))
    identity["partitions"] = provenance
    identity["aggregation_scope"] = "No model calls; recomputed whole-cohort comparisons from unchanged paired endpoint and timing observations"
    outcome = finish(ctx, catalog, rows, timings, identity,
        model_count=sum(r["checkpoint_validated"] for r in catalog["records"]), parent_count=len(parents))
    validate_confirmation_coverage(path, ctx.protocol)
    return outcome
