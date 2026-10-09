"""Deterministic execution partitions; no scientific settings or imports change.

Each timing group still contains every declared model, seed and data fraction.
Only independent confirmation parents move into separate bounded allocations.
This module deliberately needs only the Python standard library.
"""
from __future__ import annotations


def plan_shards(protocol):
    profile = protocol.get("profile")
    counts = {"smoke": 1, "development": 2, "full": 4}
    if profile not in counts:
        raise ValueError("Unknown confirmation partition profile")
    parents = [p["parent_id"] for p in protocol["parents"] if p["split"] == "confirmation"]
    if not parents or len(parents) != len(set(parents)):
        raise ValueError("Confirmation partition requires unique declared parents")
    size = counts[profile]
    return [{"shard_id": f"confirm-part-{i // size:03d}", "parent_ids": parents[i:i + size]}
            for i in range(0, len(parents), size)]


def validate_partition(protocol, partition):
    if not isinstance(partition, dict) or partition not in plan_shards(protocol):
        raise ValueError("Confirmation partition differs from the deterministic execution plan")
    return {"shard_id": partition["shard_id"], "parent_ids": list(partition["parent_ids"])}


def get_partition(protocol, shard_id):
    matches = [part for part in plan_shards(protocol) if part["shard_id"] == shard_id]
    if len(matches) != 1:
        raise ValueError("Unknown confirmation partition identity")
    return matches[0]
