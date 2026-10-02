"""Small deterministic reference training, with parent-balanced losses.

No worker or prefetch advances the authoritative sampler. A committed cursor
counts complete parent anchors only after their optimizer update has finished.
"""
from __future__ import annotations

from contextlib import nullcontext
import math
import json
import os
import platform
from pathlib import Path
import random
import time

import numpy as np
import torch

from tdn.data import DatasetStore
from tdn.data.provenance import atomic_json, canonical_hash, source_provenance
from tdn.features.local import feature_count, feature_names, iter_feature_chunks
from tdn.models import build_model
from tdn.numerics import Equation, Geometry, validate_state
from tdn.runtime.precision import reference_precision
from tdn.runtime.signal_handling import StopRequest
from tdn.models.solver import corrected_step
from .checkpoint import capture_rng, load_checkpoint, restore_rng, save_checkpoint
from .model import NormalizedModel


class _MemoryBudget:
    def __init__(self, config: dict, device: str):
        self.device = device
        self.enabled = str(device).startswith("cuda")
        self.samples = []
        self.hard_fraction = config["runtime"]["hard_memory_fraction"]
        if self.enabled:
            free, total = torch.cuda.mem_get_info(device)
            self.total = total
            self.initial_free = free
            self.soft = int(min(config["runtime"]["soft_vram_gib"] * 2**30,
                                config["runtime"]["soft_vram_fraction"] * total,
                                config["runtime"]["soft_vram_fraction"] * free))
            torch.cuda.reset_peak_memory_stats(device)

    def observe(self, step: int) -> dict | None:
        if not self.enabled:
            return None
        free, total = torch.cuda.mem_get_info(self.device)
        observation = {"step": step, "allocated_bytes": torch.cuda.memory_allocated(self.device),
                       "reserved_bytes": torch.cuda.memory_reserved(self.device),
                       "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
                       "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
                       "device_used_bytes": total - free}
        self.samples.append(observation)
        if observation["peak_reserved_bytes"] > self.soft or observation["device_used_bytes"] >= self.hard_fraction * total:
            return observation
        return None

    def report(self):
        if not self.enabled:
            return {"scope": "CPU run; no allocated GPU memory measurement"}
        return {"scope": ("desktop CUDA device" if os.environ.get("TDN_EXECUTION_MODE") == "desktop" else "allocated CUDA task") + "; device-used values sampled at safe boundaries",
                "initial_free_bytes": self.initial_free, "total_bytes": self.total,
                "soft_budget_bytes": self.soft, "hard_device_used_fraction": self.hard_fraction,
                "observations": self.samples}


def _geometry(config: dict) -> Geometry:
    return Geometry(tuple(config["problem"]["grid"]), tuple(config["problem"]["lengths"]))


def _equation(sample: dict) -> Equation:
    return Equation(**sample["parameters"])


def _state(sample: dict, config: dict, device: str):
    high_precision = config["precision"]["state"] == "float64"
    return torch.as_tensor(sample["state_fp64" if high_precision else "state"], device=device)


def _amp(config: dict, device: str):
    if config["precision"]["network_autocast"] == "bfloat16":
        if not str(device).startswith("cuda"):
            raise ValueError("BF16 candidate requires an allocated CUDA device")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _normalization(config: dict, data: DatasetStore, geometry: Geometry) -> dict:
    count = feature_count(geometry.ndim)
    total = torch.zeros(count, dtype=torch.float64)
    squared = torch.zeros(count, dtype=torch.float64)
    points = 0
    label_squared = 0.0
    label_cases = 0
    cell_volume = math.prod(geometry.lengths) / math.prod(geometry.grid)
    for sample in data.iter_samples("train"):
        state = _state(sample, config, "cpu")
        for _, _, features in iter_feature_chunks(
                state, _equation(sample), geometry,
                chunk_size=config["training"]["microbatch_cells"],
                t_ref=config["problem"]["t_ref"], U_ref=config["problem"]["U_ref"]):
            value = features.double()
            total += value.sum(0)
            squared += value.square().sum(0)
            points += len(value)
        labels = sample["defects"].astype(np.float64)
        label_squared += float(np.sum(labels * labels) * cell_volume)
        label_cases += len(sample["horizons"])
    if points == 0:
        raise ValueError("Training split is empty")
    mean = total / points
    std = torch.sqrt(torch.clamp(squared / points - mean.square(), min=0.0))
    # Constants are centered but not divided by tiny cancellation errors.
    std = torch.where(std > 1e-6, std, torch.ones_like(std))
    noise_floor = float(config["teacher"]["noise_floor"])
    return {"mean": mean.tolist(), "std": std.tolist(),
            "defect_scale": max(math.sqrt(label_squared / label_cases), noise_floor),
            "fit_split": "train", "split_hash": data.split_hash,
            "feature_names": list(feature_names(geometry.ndim)),
            "feature_version": 1, "valid_training_points": points,
            "norm": "sum(cell_volume * scalar_error**2); parent/horizon arithmetic mean"}


def _parent_index(seed: int, cursor: int, count: int) -> int:
    epoch, offset = divmod(cursor, count)
    return int(np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(count)[offset])


def _regression_backward(model: NormalizedModel, sample: dict, config: dict,
                         geometry: Geometry, device: str, scale: float, weight: float) -> float:
    state = _state(sample, config, device)
    equation = _equation(sample)
    targets = torch.as_tensor(sample["defects"], device=device, dtype=torch.float32).reshape(
        len(sample["horizons"]), -1, 1)
    cell_volume = math.prod(geometry.lengths) / math.prod(geometry.grid)
    factor = weight * cell_volume / (len(sample["horizons"]) * scale**2)
    measured = 0.0
    for start, stop, features in iter_feature_chunks(
            state, equation, geometry, chunk_size=config["training"]["microbatch_cells"],
            t_ref=model.t_ref, U_ref=model.U_ref):
        # Reuse the h-independent parameter tensor for all requested horizons.
        with _amp(config, device):
            encoded = model.encode(features) if hasattr(model.base, "encode") else None
        loss = torch.zeros((), device=device)
        for horizon_index, horizon in enumerate(sample["horizons"]):
            h = torch.tensor(horizon, device=device, dtype=torch.float32)
            if encoded is None:
                with _amp(config, device):
                    prediction = model(features, h)
            else:
                prediction = model.decode(encoded, h)
            loss = loss + factor * (prediction.float() - targets[horizon_index, start:stop]).square().sum()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite one-step loss")
        loss.backward()
        measured += float(loss.detach().cpu())
    return measured


def _rollout_backward(model: NormalizedModel, sample: dict, config: dict,
                      geometry: Geometry, device: str, scale: float, weight: float) -> float:
    windows = int(config["training"]["rollout_windows"])
    if windows == 1 or config["training"]["rollout_weight"] == 0:
        return 0.0
    available = config.get("data", {}).get("rollout_windows", [1, windows])
    if windows not in available:
        raise ValueError("Rollout teacher window is not present in dataset")
    target_index = available.index(windows)
    cell_volume = math.prod(geometry.lengths) / math.prod(geometry.grid)
    factor = weight * float(config["training"]["rollout_weight"]) * cell_volume / (
        len(sample["horizons"]) * scale**2)
    measured = 0.0
    for horizon_index, horizon in enumerate(sample["horizons"]):
        state = _state(sample, config, device)
        h = torch.tensor(horizon, device=device, dtype=state.dtype)
        for _ in range(windows):
            # Full-domain FFT physics and features remain live in this graph.
            with _amp(config, device):
                state = corrected_step(state, h, _equation(sample), geometry, model,
                                       chunk_size=config["model"]["chunk_size"],
                                       checkpoint_chunks=config["precision"]["checkpoint_chunks"])
            validate_state(state)
        target = torch.as_tensor(sample["rollout_teachers"][horizon_index, target_index],
                                 device=device, dtype=state.dtype)[None]
        loss = factor * (state - target).square().sum()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite short-rollout loss")
        loss.backward()
        measured += float(loss.detach().cpu())
    return measured


@torch.no_grad()
def _validate(model: NormalizedModel, config: dict, data: DatasetStore,
              geometry: Geometry, device: str) -> dict:
    errors, failures = [], []
    final_time = float(config["horizons"]["rollout_time"])
    h_base = max(config["horizons"]["values"])
    cell_volume = math.prod(geometry.lengths) / math.prod(geometry.grid)
    for sample in data.iter_samples("validation"):
        try:
            state = _state(sample, config, device)
            elapsed = 0.0
            while elapsed < final_time - 1e-14:
                step = min(h_base, final_time - elapsed)
                with _amp(config, device):
                    state = corrected_step(state, torch.tensor(step, device=device, dtype=state.dtype),
                                           _equation(sample), geometry, model,
                                           chunk_size=config["model"]["chunk_size"])
                validate_state(state)
                elapsed += step
            target = torch.as_tensor(sample["final_teacher"], device=device, dtype=torch.float64)
            error = float(torch.sqrt((state.double() - target).square().sum() * cell_volume).cpu())
            if not math.isfinite(error):
                raise FloatingPointError("Nonfinite validation error")
            errors.append(error)
        except (ValueError, FloatingPointError, RuntimeError) as exc:
            failures.append({"parent_id": sample["parent_id"], "reason": str(exc)})
    tolerance = float(config["validation"]["tolerance"])
    feasible = bool(errors) and not failures and max(errors) <= tolerance
    return {"metric": "full_rollout_error_at_fixed_physical_time", "final_time": final_time,
            "mean_error": float(np.mean(errors)) if errors else None,
            "max_error": max(errors) if errors else None, "failures": failures,
            "tolerance": tolerance, "feasible": feasible}


def train(config: dict, dataset_dir: Path, run_dir: Path, device: str = "cpu",
          resume: Path | None = None, max_steps: int | None = None) -> dict:
    from tdn.runtime.storage import contained_path
    dataset_dir = contained_path(dataset_dir)
    run_dir = contained_path(run_dir)
    if resume is not None:
        resume = contained_path(resume)
    with DatasetStore(dataset_dir) as data:
        return _train_with_data(config, data, run_dir, device, resume, max_steps)


def _train_with_data(config: dict, data: DatasetStore, run_dir: Path, device: str,
                     resume: Path | None, max_steps: int | None) -> dict:
    if data.manifest["problem"] != config["problem"]:
        raise ValueError("Dataset physical problem differs from training configuration")
    if data.manifest["horizons"] != config["horizons"]["values"]:
        raise ValueError("Dataset horizon list differs from training configuration")
    if int(config.get("data", {}).get("workers", 0)) != 0:
        raise ValueError("Reference exact-replay sampler requires data.workers=0")
    optimized = config["precision"]["compile_mode"] != "eager" or config["precision"]["network_autocast"] != "none"
    if optimized:
        report_path = os.environ.get("TDN_CALIBRATION_REPORT")
        if not report_path or not str(device).startswith("cuda"):
            raise ValueError("Optimized training requires TDN_CALIBRATION_REPORT from allocated GPU parity")
        report = json.loads(Path(report_path).read_text())
        if report.get("config_hash") != canonical_hash(config) or report.get("passed") is not True:
            raise ValueError("GPU calibration is missing, failed, or has a different configuration hash")
    if config["precision"]["tf32"]:
        raise ValueError("TF32 must first pass the separate optimization audit")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    configuration_hash = canonical_hash(config)
    provenance = source_provenance()
    software_environment = {"python": platform.python_version(), "torch": torch.__version__,
                            "numpy": np.__version__, "cuda_runtime": torch.version.cuda,
                            "device": str(device), "intraop_threads": torch.get_num_threads(),
                            "interop_threads": torch.get_num_interop_threads()}
    geometry = _geometry(config)
    seed = int(config.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    reference_precision()
    torch.use_deterministic_algorithms(True)
    memory = _MemoryBudget(config, device)
    normalization = _normalization(config, data, geometry)
    model = NormalizedModel(build_model(feature_count(geometry.ndim), config["model"],
                                       t_ref=config["problem"]["t_ref"],
                                       U_ref=config["problem"]["U_ref"]),
                            normalization["mean"], normalization["std"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"],
                                 weight_decay=config["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    global_step, cursor, best_metric = 0, 0, math.inf
    history = []
    if resume is not None:
        payload = load_checkpoint(resume, map_location=device)
        checks = {"config_hash": configuration_hash, "dataset_hash": data.manifest_hash,
                  "split_hash": data.split_hash, "source_hash": provenance["python_source_hash"]}
        for key, expected in checks.items():
            if payload[key] != expected:
                raise ValueError(f"Resume {key} mismatch")
        if payload.get("software_environment") != software_environment:
            raise ValueError("Resume software/device/thread environment mismatch")
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
        normalization = payload["normalization"]
        global_step, cursor = payload["global_step"], payload["sampler"]["committed_cursor"]
        best_metric = payload["best_metric"]
        history = payload["history"]
        restore_rng(payload["rng_state"])
    if config["precision"]["compile_mode"] != "eager":
        model.compile_kernels(config["precision"]["compile_mode"])
    target_steps = int(config["training"]["max_steps"])
    if max_steps is not None:
        target_steps = min(target_steps, int(max_steps))
    if target_steps < global_step:
        raise ValueError("Requested step budget precedes checkpoint step")
    batch_parents = int(config["training"]["batch_parents"])
    if batch_parents < 1 or data.count("train") < 1:
        raise ValueError("Positive parent batch and nonempty training split required")
    atomic_json(run_dir / "training_config.json", config)
    atomic_json(run_dir / "normalization.json", normalization)
    last_saved = time.monotonic()
    latest_validation = None
    status = "RUNNING"

    def commit(*, best: bool = False):
        payload = {"schema_version": 1, "config": config, "config_hash": configuration_hash,
                   "source_hash": provenance["python_source_hash"], "source_provenance": provenance,
                   "software_environment": software_environment,
                   "dataset_hash": data.manifest_hash, "split_hash": data.split_hash,
                   "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                   "scheduler_state": scheduler.state_dict(), "scaler_state": None,
                   "rng_state": capture_rng(), "normalization": normalization,
                   "global_step": global_step, "sampler": {"committed_cursor": cursor,
                   "seed": seed, "workers": 0, "prefetch": 0, "kind": "epoch_parent_permutation"},
                   "training_phase": f"{config['training']['rollout_windows']}_window",
                   "precision_policy": config["precision"], "feature_version": 1,
                   "anchor_version": None, "knots": None, "history": history,
                   "best_metric": best_metric, "validation": latest_validation, "status": status}
        payload["memory"] = memory.report()
        return save_checkpoint(run_dir, payload, best=best)

    with StopRequest() as stop:
        while global_step < target_steps:
            optimizer.zero_grad(set_to_none=True)
            loss = 0.0
            for parent_offset in range(batch_parents):
                index = _parent_index(seed, cursor + parent_offset, data.count("train"))
                sample = data.sample("train", index)
                loss += _regression_backward(model, sample, config, geometry, device,
                                             normalization["defect_scale"], 1.0 / batch_parents)
                loss += _rollout_backward(model, sample, config, geometry, device,
                                          normalization["defect_scale"], 1.0 / batch_parents)
            for parameter in model.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError("Nonfinite gradient; previous checkpoint is authoritative")
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"])
            optimizer.step()
            scheduler.step()
            global_step += 1
            cursor += batch_parents
            history.append({"step": global_step, "loss": loss})
            memory_failure = memory.observe(global_step) if global_step == 1 else None
            best = False
            validation_due = global_step % config["training"]["validation_every_steps"] == 0
            if validation_due or global_step == config["training"]["max_steps"]:
                model.eval()
                latest_validation = _validate(model, config, data, geometry, device)
                model.train()
                if latest_validation["feasible"] and latest_validation["mean_error"] < best_metric:
                    best_metric = latest_validation["mean_error"]
                    best = True
            cadence_due = global_step % config["training"]["checkpoint_every_steps"] == 0
            time_due = time.monotonic() - last_saved >= config["training"]["checkpoint_every_seconds"]
            if global_step != 1 and (validation_due or cadence_due or time_due or global_step == target_steps):
                memory_failure = memory.observe(global_step)
            if memory_failure is not None:
                status = "FAILED_MEMORY_BUDGET"
                commit()
                atomic_json(run_dir / "training_result.json", {"status": status, "global_step": global_step,
                            "memory": memory.report(), "failure": memory_failure})
                raise RuntimeError("GPU memory budget exceeded; resume only after a recorded configuration correction")
            if stop.requested:
                status = "PAUSED_NEEDS_RESUME"
            elif global_step == target_steps:
                status = "COMPLETE" if global_step == config["training"]["max_steps"] else "PAUSED_BUDGET"
            if cadence_due or time_due or best or status != "RUNNING":
                commit(best=best)
                last_saved = time.monotonic()
            if stop.requested:
                atomic_json(run_dir / "training_result.json", {"status": status, "global_step": global_step,
                            "exit_code": 75, "signal": stop.signal_number})
                raise SystemExit(75)
    if status == "RUNNING":
        status = "COMPLETE" if global_step == config["training"]["max_steps"] else "PAUSED_BUDGET"
    if target_steps == global_step and not (run_dir / "checkpoints" / "last.pt").exists():
        status = "COMPLETE" if global_step == config["training"]["max_steps"] else "PAUSED_BUDGET"
        commit()
    result = {"schema_version": 1, "status": status, "global_step": global_step,
              "committed_cursor": cursor, "validation": latest_validation,
              "config_hash": configuration_hash, "dataset_hash": data.manifest_hash,
              "split_hash": data.split_hash, "source_hash": provenance["python_source_hash"],
              "last_checkpoint": str(run_dir / "checkpoints" / "last.pt"),
              "best_checkpoint": str(run_dir / "checkpoints" / "best.pt")
              if (run_dir / "checkpoints" / "best.pt").exists() else None,
              "history": history, "resume_guarantee": "bitwise CPU deterministic reference path on the same software/device/thread environment",
              "workers": 0, "prefetch": 0, "memory": memory.report()}
    atomic_json(run_dir / "training_result.json", result)
    return result
