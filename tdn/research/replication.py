"""Bounded paired training-seed replication with frozen diagnostic evaluation.

Training states, accepted references and normalization are shared by every
replicate. All checkpoints are frozen before the first diagnostic block is
evaluated. The same independent parents are reused across training seeds, so
repeated timings never become additional independent scientific samples.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import time

import torch

from tdn.numerics import Geometry
from tdn.runtime.metadata import write_json
from .experiment import (Budget, evaluate, generate_data, state_digest,
                         train_family, training_normalization)
from .protocol import assert_parent_disjoint, digest, file_digest, verify_artifacts
from .replication_protocol import SUITE, trained_checkpoint_pairs


def _block_dataset(dataset: dict, block: int) -> dict:
    """Retain split isolation while exposing exactly one diagnostic block."""
    parents = [parent for parent in dataset["parents"]
               if parent["split"] != "diagnostic" or parent["diagnostic_block"] == block]
    assert_parent_disjoint(parents)
    return {**dataset, "parents": parents}


def _validate_dataset(dataset: dict, protocol: dict) -> None:
    if dataset.get("protocol_sha256") != digest(protocol):
        raise ValueError("Dataset does not belong to the immutable replication protocol")
    parents = dataset.get("parents", [])
    assert_parent_disjoint(parents)
    declared = protocol["parents"]
    if len(parents) != len(declared) or dataset.get("geometry") != protocol["config"]["grid"]:
        raise ValueError("Replication dataset parent/grid plan differs from the protocol")
    for parent, expected in zip(parents, declared):
        if {key: parent.get(key) for key in expected} != expected:
            raise ValueError("Replication dataset parent identities differ from the protocol")
        if parent["split"] != "diagnostic" and "diagnostic_block" in parent:
            raise ValueError("Only diagnostic parents may belong to a diagnostic block")
    hashes = [state_digest(parent["initial"]) for parent in parents]
    if len(hashes) != len(set(hashes)) or hashes != [parent["initial_state_sha256"] for parent in parents]:
        raise ValueError("Replication initial-state hashes are duplicated or corrupted")


def _training_plan(protocol: dict) -> list[dict]:
    return [{**replicate, "family": family, "status": "NOT_STARTED", "steps": 0,
             "selected_step": None, "benchmark_role": "tdn" if family == "reaction_clock" else "neural_baseline"}
            for replicate in protocol["replicates"] for family in protocol["config"]["families"]]


def _write_outcome(protocol: dict, run_dir: Path, records: list[dict], blocks: list[dict], *,
                   status: str, device: str, start: float, training_attempted: bool,
                   error: BaseException | None = None, source_run: Path | None = None) -> dict:
    from .replication_summary import summarize_replication
    detail = summarize_replication(protocol, records, blocks)
    detail.update(status=status, device=device, smoke=protocol["smoke"], benchmark_suite=SUITE)
    completed = sum(block.get("status") == "COMPLETED" for block in blocks)
    detail.update(completed_block_result_count=completed, partial_block_result_count=len(blocks) - completed)
    summary = {"status": status, "device": device, "smoke": protocol["smoke"],
               "benchmark_suite": SUITE, "scope": protocol["scope"],
               "elapsed_seconds": time.monotonic() - start,
               "training_attempted": training_attempted,
               "actual_neural_training": device == "cpu" and any(record.get("steps", 0) > 0 for record in records),
               "selected_checkpoint_count": sum(record["status"] == "COMPLETED" for record in records),
               "trained_checkpoint_count": sum(record["status"] == "COMPLETED" and
                   type(record.get("selected_step")) is int and record["selected_step"] > 0 and
                   record.get("selected_parameters_changed") is True for record in records),
               "replicate_count": len(protocol["replicates"]),
               "diagnostic_parent_count": sum(parent["split"] == "diagnostic" for parent in protocol["parents"]),
               "completed_block_count": completed, "partial_block_count": len(blocks) - completed,
               "expected_block_result_count": detail["expected_block_result_count"],
               "observed_block_result_count": detail["observed_block_result_count"],
               "endpoint_rows": detail["endpoint_rows"], "comparison_rows": detail["comparison_rows"],
               "replication_file": "replication.json"}
    summary["training" if device == "cpu" else "source_training"] = records
    if source_run is not None:
        summary["source_run"] = str(source_run)
        summary["scope"] = "Allocated GPU timing of frozen paired CPU checkpoints; no additional training or confirmation"
    if error is not None:
        summary["error"] = f"{type(error).__name__}: {error}"
    for key in ("actual_neural_training", "training_attempted", "selected_checkpoint_count", "trained_checkpoint_count"):
        detail[key] = summary[key]
    write_json(run_dir / "replication.json", detail)
    write_json(run_dir / "summary.json", summary)
    return summary


def _evaluate_blocks(protocol: dict, dataset: dict, models_by_seed: dict, records: list[dict],
                     run_dir: Path, budget: Budget, device: str, blocks: list[dict]) -> None:
    for replicate in protocol["replicates"]:
        replicate_id = replicate["replicate_id"]
        training = [record for record in records if record["replicate_id"] == replicate_id]
        for block in range(protocol["diagnostic_replicates"]):
            budget.check()
            context = {**replicate, "diagnostic_block": block}
            block_dir = run_dir / "replicates" / replicate_id / "blocks" / f"block-{block}"
            block_dir.mkdir(parents=True, exist_ok=False)
            try:
                summary = evaluate(models_by_seed[replicate_id], _block_dataset(dataset, block), protocol,
                                   block_dir, budget, device, training_records=training, replication=context)
            except BaseException as error:
                # Already persisted parents are actual evidence even when the
                # current block does not finish. Missing rows remain missing.
                partial = {**context, "status": "PARTIAL", "error": f"{type(error).__name__}: {error}"}
                for name in ("frontier", "heldout"):
                    path = block_dir / f"{name}.json"
                    partial[name] = json.loads(path.read_text())["rows"] if path.is_file() else []
                blocks.append(partial)
                write_json(block_dir / "summary.json", {"status": "INTERRUPTED" if isinstance(error, (InterruptedError, KeyboardInterrupt, TimeoutError)) else "FAILED",
                           "device": device, "benchmark_suite": SUITE, "replication": context,
                           "error": partial["error"], "partial": True})
                raise
            summary.update(status="COMPLETED", device=device, benchmark_suite=SUITE)
            write_json(block_dir / "summary.json", summary)
            frontier = json.loads((block_dir / "frontier.json").read_text())["rows"]
            heldout = json.loads((block_dir / "heldout.json").read_text())["rows"]
            neural = json.loads((block_dir / "neural-comparisons.json").read_text())
            blocks.append({**context, "status": "COMPLETED", "frontier": frontier, "heldout": heldout,
                           "neural": neural, "summary": summary})


def run(protocol: dict, run_dir: Path, *, stop=None) -> dict:
    """Train all fifteen candidates before inspecting any fresh diagnostic."""
    start = time.monotonic()
    budget = Budget(protocol["config"]["max_seconds"], stop)
    run_dir = Path(run_dir)
    records = _training_plan(protocol)
    blocks, models_by_seed = [], {}
    training_attempted = False
    try:
        dataset = generate_data(protocol, run_dir, budget)
        _validate_dataset(dataset, protocol)
        geometry = Geometry(tuple(protocol["config"]["grid"]), (1., 1.))
        normalization, parent_ids = training_normalization(dataset, geometry)
        write_json(run_dir / "normalization.json", {"mean": normalization[0].tolist(),
                   "std": normalization[1].tolist(), "parent_ids": parent_ids, "split": "train",
                   "sharing": "one fit reused across every family and training seed"})
        index = 0
        for replicate in protocol["replicates"]:
            models = {}
            replicate_dir = run_dir / "replicates" / replicate["replicate_id"]
            for family in protocol["config"]["families"]:
                budget.check()
                training_attempted = True
                try:
                    model, record = train_family(family, dataset, protocol, replicate_dir, normalization, budget,
                        initialization_seed=replicate["training_seed"],
                        sample_schedule_seed=replicate["sample_schedule_seed"], replicate_id=replicate["replicate_id"])
                except BaseException:
                    record_path = replicate_dir / "training" / f"{family}.json"
                    if record_path.is_file():
                        records[index] = json.loads(record_path.read_text())
                    raise
                records[index] = record
                index += 1
                if model is not None:
                    models[family] = model
            models_by_seed[replicate["replicate_id"]] = models
        # No diagnostic evaluation is allowed above this point.
        _evaluate_blocks(protocol, dataset, models_by_seed, records, run_dir, budget, "cpu", blocks)
    except BaseException as error:
        status = "INTERRUPTED" if isinstance(error, (InterruptedError, KeyboardInterrupt, TimeoutError)) else "FAILED"
        _write_outcome(protocol, run_dir, records, blocks, status=status, device="cpu", start=start,
                       training_attempted=training_attempted, error=error)
        raise
    return _write_outcome(protocol, run_dir, records, blocks, status="COMPLETED", device="cpu", start=start,
                          training_attempted=training_attempted)


def _frozen_models(protocol: dict, source_run: Path, records: list[dict], device: str) -> dict:
    from . import build_research_model
    expected = _training_plan(protocol)
    identities = [(row.get("replicate_id"), row.get("training_seed"), row.get("sample_schedule_seed"), row.get("family"))
                  for row in records]
    if identities != [(row["replicate_id"], row["training_seed"], row["sample_schedule_seed"], row["family"])
                      for row in expected]:
        raise ValueError("Frozen training records differ from the paired seed/family plan")
    normalization = json.loads((source_run / "normalization.json").read_text())
    mean, std = torch.tensor(normalization["mean"], dtype=torch.float32), torch.tensor(normalization["std"], dtype=torch.float32)
    expected_train_ids = [parent["parent_id"] for parent in protocol["parents"] if parent["split"] == "train"]
    if normalization["split"] != "train" or normalization["parent_ids"] != expected_train_ids:
        raise ValueError("Frozen normalization differs from the training-only fit plan")
    models = {replicate["replicate_id"]: {} for replicate in protocol["replicates"]}
    for record in records:
        if record["status"] != "COMPLETED":
            continue
        family, replicate_id = record["family"], record["replicate_id"]
        path = source_run / "replicates" / replicate_id / "checkpoints" / f"{family}.pt"
        if record.get("checkpoint_sha256") != file_digest(path):
            raise ValueError("Frozen checkpoint hash differs from its training record")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        required = {"family": family, "protocol_sha256": digest(protocol), "selected_step": record["selected_step"],
                    "replicate_id": replicate_id, "training_seed": record["training_seed"],
                    "sample_schedule_seed": record["sample_schedule_seed"]}
        if any(checkpoint.get(key) != value for key, value in required.items()):
            raise ValueError("Frozen checkpoint family/protocol/paired seed mismatch")
        model = build_research_model(family, ndim=2, width=protocol["config"]["width"]).to(dtype=torch.float32)
        model.set_normalization(mean, std)
        expected_buffers = {name: tensor.clone() for name, tensor in model.state_dict().items()
                            if name.endswith(("feature_mean", "feature_std"))}
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        if any(not torch.equal(model.state_dict()[name], value) for name, value in expected_buffers.items()):
            raise ValueError("Frozen checkpoint normalization differs from the shared training-only fit")
        models[replicate_id][family] = model.to(device=device).eval()
    return models


def benchmark(protocol: dict, source_run: Path, run_dir: Path, *, stop=None) -> dict:
    """Measure exact frozen CPU checkpoints on CUDA, with no training fallback."""
    start = time.monotonic()
    budget = Budget(min(600, protocol["config"]["max_seconds"]), stop)
    source_run, run_dir = Path(source_run), Path(run_dir)
    records, blocks = [], []
    try:
        verify_artifacts(source_run)
        if digest(json.loads((source_run / "protocol.json").read_text())) != digest(protocol):
            raise ValueError("Frozen CPU source protocol differs from the GPU replication protocol")
        source_summary = json.loads((source_run / "summary.json").read_text())
        if source_summary.get("status") != "COMPLETED" or source_summary.get("device") != "cpu" or source_summary.get("benchmark_suite") != SUITE:
            raise ValueError("Replication benchmark requires a completed CPU replication source")
        records = source_summary["training"]
        if not trained_checkpoint_pairs(records):
            raise ValueError("Replication GPU benchmark requires a trained clock and baseline checkpoint from the same seed")
        dataset = torch.load(source_run / "dataset.pt", map_location="cpu", weights_only=True)
        _validate_dataset(dataset, protocol)
        shutil.copyfile(source_run / "dataset.pt", run_dir / "dataset.pt")
        shutil.copyfile(source_run / "manifest.json", run_dir / "source_manifest.json")
        shutil.copyfile(source_run / "normalization.json", run_dir / "normalization.json")
        shutil.copyfile(source_run / "references.json", run_dir / "references.json")
        models = _frozen_models(protocol, source_run, records, "cuda")
        _evaluate_blocks(protocol, dataset, models, records, run_dir, budget, "cuda", blocks)
    except BaseException as error:
        status = "INTERRUPTED" if isinstance(error, (InterruptedError, KeyboardInterrupt, TimeoutError)) else "FAILED"
        _write_outcome(protocol, run_dir, records, blocks, status=status, device="cuda", start=start,
                       training_attempted=False, source_run=source_run, error=error)
        raise
    return _write_outcome(protocol, run_dir, records, blocks, status="COMPLETED", device="cuda", start=start,
                          training_attempted=False, source_run=source_run)
