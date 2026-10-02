"""Complete, same-directory atomic checkpoints at committed step boundaries."""
from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch

from tdn.data.provenance import atomic_json, file_hash, fsync_directory


def capture_rng() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("CUDA checkpoint cannot exactly resume on CPU")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def _atomic_torch_save(path: Path, payload: dict, previous: Path | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    with pending.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    # A failed serialization leaves the published current checkpoint intact.
    if previous is not None and path.exists():
        backup_pending = previous.with_name(previous.name + ".pending")
        if backup_pending.exists():
            backup_pending.unlink()
        try:
            os.link(path, backup_pending)
        except OSError:
            # Some Windows filesystems do not support hard links. Preserve the
            # previous published checkpoint using a flushed local copy.
            with path.open("rb") as source, backup_pending.open("wb") as target:
                shutil.copyfileobj(source, target)
                target.flush()
                os.fsync(target.fileno())
        backup_pending.replace(previous)
    pending.replace(path)
    fsync_directory(path.parent)


def save_checkpoint(run_dir: Path, payload: dict, *, best: bool = False) -> Path:
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last = checkpoint_dir / "last.pt"
    previous = checkpoint_dir / "last.previous.pt"
    _atomic_torch_save(last, payload, previous)
    if best:
        _atomic_torch_save(checkpoint_dir / "best.pt", payload)
    latest = {"schema_version": 1, "path": last.name, "sha256": file_hash(last),
              "global_step": payload["global_step"], "status": payload["status"],
              "previous": {"path": previous.name, "sha256": file_hash(previous)}
              if previous.exists() else None}
    best_path = checkpoint_dir / "best.pt"
    latest["best"] = {"path": best_path.name, "sha256": file_hash(best_path)} if best_path.exists() else None
    atomic_json(checkpoint_dir / "latest.json", latest)
    return last


def load_checkpoint(path: Path | str, map_location: str | torch.device = "cpu") -> dict:
    path = Path(path)
    if path.is_dir():
        if (path / "checkpoints").is_dir():
            path = path / "checkpoints"
        path = path / "latest.json"
    if path.suffix == ".json":
        latest = json.loads(path.read_text())
        checkpoint = path.parent / latest["path"]
        if file_hash(checkpoint) != latest["sha256"]:
            raise ValueError("Checkpoint checksum does not match latest.json")
        path = checkpoint
    elif (path.parent / "latest.json").exists():
        latest = json.loads((path.parent / "latest.json").read_text())
        candidates = [latest, latest.get("previous"), latest.get("best")]
        entry = next((item for item in candidates if item and item["path"] == path.name), None)
        if entry is not None:
            actual = file_hash(path)
            # If interruption follows last's replacement but precedes latest's
            # replacement, the backup is the checkpoint certified by old latest.
            recoverable_backup = path.name == "last.previous.pt" and actual == latest["sha256"]
            if actual != entry["sha256"] and not recoverable_backup:
                raise ValueError("Checkpoint checksum does not match latest.json")
    # Loading is restricted to checkpoints produced by this project, never a URL.
    payload = torch.load(path, map_location=map_location, weights_only=False)
    required = {"model_state", "optimizer_state", "scheduler_state", "rng_state",
                "normalization", "global_step", "sampler", "config_hash", "dataset_hash",
                "split_hash", "source_hash", "config", "status"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"Incomplete checkpoint: {sorted(missing)}")
    return payload
