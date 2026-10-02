"""Deterministic full-domain FP64 teachers and separately stored defect labels."""
from __future__ import annotations

from dataclasses import asdict
import math
import os
from pathlib import Path

import numpy as np
import torch

from tdn.numerics import (Equation, Geometry, REFERENCE_METHOD, SPLIT_METHOD,
                          choose_substeps, refined_reference, split_step, weighted_norm)
from .dataset import DatasetStore, make_parent_split
from .provenance import atomic_json, canonical_hash, file_hash, source_provenance


def _generation_configuration(config: dict) -> dict:
    return {"seed": config["seed"], "problem": config["problem"],
            "horizons": config["horizons"], "data": config["data"],
            "teacher": config["teacher"], "input_precision": config["precision"]["state"],
            "validation_tolerance": config["validation"]["tolerance"],
            "rollout_windows": config["data"].get("rollout_windows", sorted(
                {1, int(config["training"]["rollout_windows"])}))}


def _initial_state(seed: int, geometry: Geometry) -> torch.Tensor:
    """Smooth bounded independent ICs, with no teacher or future-state inputs."""
    rng = np.random.default_rng(seed)
    coordinates = np.meshgrid(*[np.arange(n) / n for n in geometry.grid], indexing="ij")
    wave = np.zeros(geometry.grid, dtype=np.float64)
    for mode in range(1, 4):
        for coordinate in coordinates:
            wave += rng.uniform(0.2, 0.7) * np.sin(2 * np.pi * mode * coordinate + rng.uniform(0, 2 * np.pi))
    state = 0.5 + 0.3 * np.tanh(wave)
    return torch.from_numpy(state.copy())[None, None]


def _accepted_reference(state, h, equation, geometry, config):
    teacher = config["teacher"]
    count = max(int(teacher["base_substeps"]), choose_substeps(float(h), equation, geometry))
    tolerance = float(config["validation"]["tolerance"])
    retries = []
    for _ in range(5):
        result = refined_reference(state, float(h), equation, geometry, count,
                                   error_fraction=float(teacher["error_fraction"]),
                                   # The teacher independently checks its 5% defect budget.
                                   tolerance=tolerance, noise_floor=float(teacher["noise_floor"]))
        tolerance_ok = result.uncertainty <= float(teacher["tolerance_fraction"]) * tolerance
        if result.accepted and tolerance_ok:
            metadata = asdict(result)
            del metadata["state"]
            metadata["refinement_attempts"] = retries
            return result.state, metadata
        retries.append({"coarse_substeps": count, "reason": result.reason,
                        "uncertainty": result.uncertainty, "defect_norm": result.defect_norm})
        count *= 2
    raise ValueError(f"Teacher rejected after bounded refinement: {retries[-1]}")


def generate_dataset(config: dict, destination: Path) -> Path:
    from tdn.runtime.storage import contained_path
    destination = contained_path(destination)
    generation_config = _generation_configuration(config)
    provenance = source_provenance()
    teacher_sources = {name: digest for name, digest in provenance["files"].items()
                       if name.startswith(("tdn/numerics/", "tdn/data/"))
                       or name in ("requirements.txt", "pyproject.toml")}
    teacher_code_hash = canonical_hash(teacher_sources)
    generation_hash = canonical_hash({"configuration": generation_config, "teacher_code_hash": teacher_code_hash})
    if destination.exists() and any(destination.iterdir()):
        if (destination / "COMPLETE.json").exists():
            existing = DatasetStore(destination)
            if existing.manifest["generation_hash"] == generation_hash:
                return destination / "manifest.json"
            raise ValueError("Existing immutable dataset has a different generation configuration")
        raise ValueError("Dataset directory contains incomplete or unrelated work; choose a new destination")
    destination.mkdir(parents=True, exist_ok=True)
    geometry = Geometry(tuple(config["problem"]["grid"]), tuple(config["problem"]["lengths"]))
    parents = make_parent_split(config)
    horizons = list(map(float, config["horizons"]["values"]))
    windows = generation_config["rollout_windows"]
    if any(window not in (1, 2, 4) for window in windows):
        raise ValueError("Only audited short rollout windows 1/2/4 are supported")
    anchors = int(config["data"].get("anchors_per_parent", 1))
    if anchors < 1:
        raise ValueError("At least one anchor per parent is required")
    manifest = {"schema_version": 1, "generation_hash": generation_hash,
                "generation_configuration": generation_config, "problem": config["problem"],
                "source_provenance": provenance, "teacher_code_hash": teacher_code_hash, "parents": parents,
                "split_hash": canonical_hash(parents), "horizons": horizons,
                "rollout_windows": windows, "rollout_time": config["horizons"]["rollout_time"],
                "teacher_method": REFERENCE_METHOD, "split_method": SPLIT_METHOD,
                "teacher_precision": "float64", "label_subtraction_precision": "float64",
                "state_precision": config["precision"]["state"], "stored_defect_precision": "float32",
                "ic_generator": "bounded_smooth_periodic_fourier_tanh_v1",
                "confirmatory": {"status": "SEALED", "generated": False,
                                 "note": "Separate frozen confirmatory protocol required before access"},
                "norm": {"cell_volume": math.prod(geometry.lengths) / math.prod(geometry.grid),
                         "species_scale": 1.0, "mask": "all scalar cells"}, "groups": {}}
    for split in ("train", "validation", "diagnostic"):
        split_parents = parents[split]
        if not split_parents:
            continue
        count = len(split_parents) * anchors
        base_shape = (count, *geometry.grid)
        paired_shape = (count, len(horizons), *geometry.grid)
        specs = {"states": (base_shape, "float32"), "states_fp64": (base_shape, "float64"),
                 "defects": (paired_shape, "float32"), "defects_fp64": (paired_shape, "float64"),
                 "split": (paired_shape, "float64"), "teacher": (paired_shape, "float64"),
                 "uncertainty": ((count, len(horizons)), "float64"),
                 "rollout_teachers": ((count, len(horizons), len(windows), *geometry.grid), "float64"),
                 "final_teacher": (base_shape, "float64")}
        if max(math.prod(shape) * np.dtype(dtype).itemsize for shape, dtype in specs.values()) > 512 * 1024**2:
            raise ValueError("Requested contiguous shard exceeds 512 MiB; use a smaller staged dataset")
        arrays = {}
        paths = {}
        for name, (shape, dtype) in specs.items():
            path = destination / f"{split}_{name}.npy"
            paths[name] = path
            arrays[name] = np.lib.format.open_memmap(path.with_suffix(".npy.pending"), mode="w+",
                                                    dtype=dtype, shape=shape)
        samples = []
        for parent_index, parent in enumerate(split_parents):
            equation = Equation(parent["kappa"], parent["reaction_rate"])
            initial = _initial_state(parent["seed"], geometry)
            for anchor_index in range(anchors):
                row = parent_index * anchors + anchor_index
                anchor_time = anchor_index * max(horizons)
                state = initial
                if anchor_time:
                    state, _ = _accepted_reference(initial, anchor_time, equation, geometry, config)
                input32 = state.float()
                input_quantization_norm = float(weighted_norm(input32.double() - state, geometry))
                arrays["states"][row] = input32.numpy()[0, 0]
                arrays["states_fp64"][row] = state.numpy()[0, 0]
                reference_records = []
                for horizon_index, horizon in enumerate(horizons):
                    teacher, reference = _accepted_reference(state, horizon, equation, geometry, config)
                    base = split_step(state, horizon, equation, geometry, differentiable=False)
                    defect = teacher - base  # Mandatory FP64 subtraction before storing FP32.
                    quantized_teacher, quantized_reference = _accepted_reference(
                        input32.double(), horizon, equation, geometry, config)
                    quantized_base = split_step(input32.double(), horizon, equation, geometry, differentiable=False)
                    quantization_defect_error = float(weighted_norm(
                        quantized_teacher - quantized_base - defect, geometry))
                    arrays["split"][row, horizon_index] = base.numpy()[0, 0]
                    arrays["teacher"][row, horizon_index] = teacher.numpy()[0, 0]
                    arrays["defects_fp64"][row, horizon_index] = defect.numpy()[0, 0]
                    arrays["defects"][row, horizon_index] = defect.float().numpy()[0, 0]
                    arrays["uncertainty"][row, horizon_index] = reference["uncertainty"]
                    reference["horizon"] = horizon
                    reference["input_quantization_defect_error"] = quantization_defect_error
                    reference["input_quantization_defect_error_fraction"] = quantization_defect_error / max(
                        reference["defect_norm"], float(config["teacher"]["noise_floor"]))
                    reference["quantized_input_reference"] = quantized_reference
                    reference["rollout_references"] = {}
                    for window_index, window in enumerate(windows):
                        if window == 1:
                            final, rollout_reference = teacher, reference.copy()
                            rollout_reference.pop("rollout_references", None)
                        else:
                            final, rollout_reference = _accepted_reference(
                                state, window * horizon, equation, geometry, config)
                        arrays["rollout_teachers"][row, horizon_index, window_index] = final.numpy()[0, 0]
                        reference["rollout_references"][str(window)] = rollout_reference
                    reference_records.append(reference)
                final_teacher, final_record = _accepted_reference(
                    state, float(config["horizons"]["rollout_time"]), equation, geometry, config)
                arrays["final_teacher"][row] = final_teacher.numpy()[0, 0]
                samples.append({"parent_index": parent_index, "anchor_index": anchor_index,
                                "anchor_time": anchor_time, "parent_id": parent["parent_id"],
                                "input_quantization_state_norm": input_quantization_norm,
                                "references": reference_records, "final_reference": final_record})
        entries = {}
        for name, array in arrays.items():
            array.flush()
            path = paths[name]
            array._mmap.close()
            pending = path.with_suffix(".npy.pending")
            # Windows' CRT commit requires a writable file handle.
            with pending.open("r+b") as stream:
                os.fsync(stream.fileno())
            pending.replace(path)
            entries[name] = {"path": path.name, "sha256": file_hash(path),
                             "shape": list(specs[name][0]), "dtype": specs[name][1],
                             "bytes": path.stat().st_size}
        manifest["groups"][split] = {"arrays": entries, "samples": samples,
                                     "writer_count": 1, "format": "contiguous_numpy_memmap"}
    atomic_json(destination / "manifest.json", manifest)
    atomic_json(destination / "COMPLETE.json", {"schema_version": 1,
                "manifest_hash": canonical_hash(manifest), "status": "COMPLETE"})
    DatasetStore(destination)  # Fresh reader checks every completed shard.
    return destination / "manifest.json"
