"""Read-only, checksum-validated paired full-domain samples."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np

from .provenance import canonical_hash, file_hash


SPLIT_NAMES = ("train", "validation", "diagnostic", "confirmatory")


def make_parent_split(config: dict) -> dict[str, list[dict]]:
    """Each parent seed has exactly one split, independent of derived windows."""
    result = {}
    problem = config["problem"]
    for split in SPLIT_NAMES:
        count = int(config.get("data", {}).get(f"{split}_count", 0))
        parents = []
        for index in range(count):
            token = f"tdn-parent-v1:{int(config.get('seed', 0))}:{split}:{index}"
            seed = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "little")
            rng = np.random.default_rng(seed)
            kappa_range = problem.get("kappa_range", [problem["kappa"]] * 2)
            rate_range = problem.get("reaction_rate_range", [problem["reaction_rate"]] * 2)
            parent = {"parent_id": canonical_hash({"token": token, "problem": problem})[:24],
                      "split": split, "seed": seed,
                      "kappa": float(rng.uniform(*kappa_range)),
                      "reaction_rate": float(rng.uniform(*rate_range))}
            parents.append(parent)
        result[split] = parents
    seeds = [parent["seed"] for parents in result.values() for parent in parents]
    ids = [parent["parent_id"] for parents in result.values() for parent in parents]
    if len(seeds) != len(set(seeds)) or len(ids) != len(set(ids)):
        raise RuntimeError("Parent split collision")
    return result


class DatasetStore:
    def __init__(self, path: Path | str, verify: bool = True):
        self.path = Path(path)
        if self.path.is_file():
            self.path = self.path.parent
        if not (self.path / "COMPLETE.json").exists():
            raise ValueError("Dataset has no atomic completion marker")
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        marker = json.loads((self.path / "COMPLETE.json").read_text())
        self.manifest_hash = canonical_hash(self.manifest)
        self.split_hash = self.manifest["split_hash"]
        if marker.get("manifest_hash") != self.manifest_hash:
            raise ValueError("Dataset completion marker does not match manifest")
        if canonical_hash(self.manifest["parents"]) != self.split_hash:
            raise ValueError("Parent split hash mismatch")
        all_ids = [p["parent_id"] for parents in self.manifest["parents"].values() for p in parents]
        all_seeds = [p["seed"] for parents in self.manifest["parents"].values() for p in parents]
        if len(all_ids) != len(set(all_ids)) or len(all_seeds) != len(set(all_seeds)):
            raise ValueError("Dataset contains overlapping parent splits")
        self._arrays = {}
        if verify:
            for group in self.manifest["groups"].values():
                for entry in group["arrays"].values():
                    if file_hash(self.path / entry["path"]) != entry["sha256"]:
                        raise ValueError(f"Dataset checksum mismatch: {entry['path']}")

    def count(self, split: str) -> int:
        return len(self.manifest.get("groups", {}).get(split, {}).get("samples", []))

    def arrays(self, split: str) -> dict[str, np.ndarray]:
        if split == "confirmatory":
            raise ValueError("Confirmatory parents are sealed; pilot may not read them")
        if split not in self.manifest["groups"]:
            raise ValueError(f"Dataset split unavailable: {split}")
        if split not in self._arrays:
            self._arrays[split] = {
                name: np.load(self.path / entry["path"], mmap_mode="r", allow_pickle=False)
                for name, entry in self.manifest["groups"][split]["arrays"].items()}
        return self._arrays[split]

    def sample(self, split: str, index: int) -> dict:
        arrays = self.arrays(split)
        metadata = self.manifest["groups"][split]["samples"][index]
        parent = self.manifest["parents"][split][metadata["parent_index"]]
        return {"state": np.array(arrays["states"][index], copy=True)[None, None],
                "state_fp64": np.array(arrays["states_fp64"][index], copy=True)[None, None],
                "parameters": {"kappa": parent["kappa"], "reaction_rate": parent["reaction_rate"]},
                "horizons": self.manifest["horizons"],
                "defects": np.array(arrays["defects"][index], copy=True)[:, None],
                "split": np.array(arrays["split"][index], copy=True)[:, None],
                "teacher": np.array(arrays["teacher"][index], copy=True)[:, None],
                "final_teacher": np.array(arrays["final_teacher"][index], copy=True)[None, None],
                "rollout_teachers": np.array(arrays["rollout_teachers"][index], copy=True)[:, :, None],
                "uncertainty": np.array(arrays["uncertainty"][index], copy=True),
                "parent_id": parent["parent_id"], "metadata": metadata}

    def iter_samples(self, split: str) -> Iterator[dict]:
        for index in range(self.count(split)):
            yield self.sample(split, index)
