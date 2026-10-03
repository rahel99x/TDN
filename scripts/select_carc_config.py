#!/usr/bin/env python3
"""Repeat the bounded CPU smoke comparison using validation-only selection."""
from __future__ import annotations

import argparse
import copy
import json
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tdn.config import config_hash, load_config, validate_config
from tdn.data.provenance import atomic_json, file_hash, source_provenance
from tdn.runtime.preflight import execution_mode, verify_runtime
from tdn.runtime.storage import configure_storage, contained_path

CANDIDATES = ((32, .001), (16, .001), (32, .003), (16, .003))
REPETITIONS = 3
SELECTION_RULE = ("validation feasibility; lowest best mean full-rollout validation error; "
                  "median CPU training wall seconds breaks exact ties")


def validate_scope(config: dict) -> dict:
    """This utility permits one prespecified tiny comparison, never a pilot."""
    config = validate_config(config)
    requirements = {
        "purpose": "development",
        "seed": 0,
        "problem.family": "logistic_rd",
        "problem.grid": [8, 8],
        "problem.lengths": [1.0, 1.0],
        "problem.kappa": .01,
        "problem.reaction_rate": 2.0,
        "problem.kappa_range": [.01, .01],
        "problem.reaction_rate_range": [2.0, 2.0],
        "problem.t_ref": 1.0,
        "problem.U_ref": 1.0,
        "horizons.values": [.04, .08, .16, .32],
        "horizons.rollout_time": .64,
        "horizons.evaluation_steps": [.04, .08, .16, .32],
        "horizons.mixed_factors": [.5, 1.0, 1.5],
        "data.train_count": 4,
        "data.validation_count": 2,
        "data.diagnostic_count": 2,
        "data.confirmatory_count": 0,
        "data.anchors_per_parent": 1,
        "data.workers": 0,
        "data.rollout_windows": [1, 2, 4],
        "teacher.base_substeps": 16,
        "teacher.error_fraction": .05,
        "teacher.tolerance_fraction": .1,
        "teacher.noise_floor": 1e-10,
        "model.family": "temporal_mlp",
        "model.modes": 4,
        "model.fixed_rates": [.1, 1.0, 10.0, 100.0],
        "model.anchored": False,
        "training.max_steps": 12,
        "training.batch_parents": 1,
        "training.weight_decay": 1e-5,
        "training.grad_clip": 1.0,
        "training.rollout_windows": 1,
        "training.rollout_weight": .1,
        "training.checkpoint_every_steps": 5,
        "training.checkpoint_every_seconds": 600,
        "training.validation_every_steps": 5,
        "precision.state": "float32",
        "precision.teacher": "float64",
        "precision.network_autocast": "none",
        "precision.compile_mode": "eager",
        "precision.tf32": False,
        "precision.checkpoint_chunks": False,
        "validation.tolerance": .002,
        "validation.require_headroom": False,
        "validation.tolerance_manifest": None,
        "validation.reference_fraction": .1,
        "validation.bootstrap_samples": 200,
        "runtime.intraop_threads": 1,
        "runtime.interop_threads": 1,
        "runtime.soft_vram_gib": 30,
        "runtime.soft_vram_fraction": .8,
        "runtime.hard_memory_fraction": .9,
        "runtime.confirmatory_authorized": False,
    }
    for field, expected in requirements.items():
        value = config
        for key in field.split("."):
            value = value[key]
        if value != expected:
            raise ValueError(f"Bounded config selection requires {field}={expected!r}")
    return config


def make_plan(config: dict) -> dict:
    base = validate_scope(config)
    candidates = {}
    for width, learning_rate in CANDIDATES:
        label = f"width{width}-lr{learning_rate}"
        candidate = copy.deepcopy(base)
        candidate["experiment"] = "tdn_carc_smoke"
        candidate["model"].update(width=width, chunk_size=64)
        candidate["training"].update(learning_rate=learning_rate, microbatch_cells=64)
        candidate = validate_config(candidate)
        candidates[label] = {"config": candidate, "config_hash": config_hash(candidate)}
    return {"candidate_order": list(candidates), "candidates": candidates,
            "repetitions": REPETITIONS, "steps_per_trial": 12,
            "selection_rule": SELECTION_RULE,
            "diagnostic_parents_used_for_selection": False,
            "confirmatory_parents_opened": False}


def select_winner(candidates: dict) -> str | None:
    """Use only checkpoint validation; diagnostics have no selection input."""
    feasible = {}
    for label, candidate in candidates.items():
        trials = candidate["trials"]
        if len(trials) != REPETITIONS:
            raise ValueError("Selection requires all three recorded repetitions")
        for trial in trials:
            validation = trial.get("best_validation")
            if (trial.get("status") != "COMPLETE" or trial.get("steps") != 12
                    or not validation or not validation.get("feasible")):
                break
        else:
            feasible[label] = candidate
    if not feasible:
        return None
    return min(feasible, key=lambda label: (
        feasible[label]["trials"][0]["best_validation"]["mean_error"],
        feasible[label]["median_wall_seconds"]))


def cpu_name() -> str:
    path = Path("/proc/cpuinfo")
    if path.is_file():
        try:
            for line in path.read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine() or "unknown CPU"


def prepare_run(run_dir: Path) -> tuple[Path, dict, dict]:
    """Refuse policy/scope/storage failures before creating run artifacts."""
    root = configure_storage()
    verify_runtime("cpu", "config-selection")
    config_path = contained_path(root / "configs/smoke.yaml")
    config = load_config(config_path)
    plan = make_plan(config)
    destination = contained_path(run_dir)
    if destination.exists():
        raise ValueError("Config-selection run directory already exists; choose a fresh --run-dir")
    provenance = source_provenance()
    plan.update({"category": "proposed_configuration", "device": "cpu",
                 "execution_mode": execution_mode(), "source_provenance": provenance,
                 "base_config_path": str(config_path),
                 "base_config_sha256": file_hash(config_path),
                 "selector_sha256": file_hash(Path(__file__))})
    destination.mkdir(parents=True, exist_ok=False)
    atomic_json(destination / "plan.json", plan)
    return destination, config, plan


def verify_unchanged(plan: dict) -> None:
    if source_provenance()["python_source_hash"] != plan["source_provenance"]["python_source_hash"]:
        raise RuntimeError("Scientific Python source changed during config selection; results are incomplete")
    if file_hash(Path(plan["base_config_path"])) != plan["base_config_sha256"]:
        raise RuntimeError("Base smoke config changed during config selection; results are incomplete")
    if file_hash(Path(__file__)) != plan["selector_sha256"]:
        raise RuntimeError("Selector changed during config selection; results are incomplete")


def measure(run_dir: Path) -> dict:
    # Set project-local storage and require allocation/venv before heavy imports.
    destination, base_config, plan = prepare_run(run_dir)
    import numpy as np
    import torch
    import yaml
    from tdn.data import DatasetStore, generate_dataset
    from tdn.train import load_checkpoint, train

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    candidates = copy.deepcopy(plan["candidates"])
    for label, candidate in candidates.items():
        path = destination / f"{label}.yaml"
        path.write_text(yaml.safe_dump(candidate["config"], sort_keys=False))
        candidate.update(config_path=str(path), trials=[])
    verify_unchanged(plan)
    generation_start = time.perf_counter()
    generate_dataset(base_config, destination / "dataset")
    generation_seconds = time.perf_counter() - generation_start
    with DatasetStore(destination / "dataset") as dataset:
        dataset_hash = dataset.manifest_hash
        generation_hash = dataset.manifest["generation_hash"]
        split_hash = dataset.split_hash
    for repetition in range(REPETITIONS):
        order = plan["candidate_order"]
        order = order[repetition:] + order[:repetition]
        for label in order:
            verify_unchanged(plan)
            candidate = candidates[label]
            trial_dir = destination / label / f"repetition-{repetition + 1}"
            start = time.perf_counter()
            result = train(candidate["config"], destination / "dataset", trial_dir,
                           device="cpu", max_steps=12)
            wall_seconds = time.perf_counter() - start
            best_path = Path(result["best_checkpoint"]) if result["best_checkpoint"] else None
            best = load_checkpoint(best_path) if best_path else None
            trial = {"repetition": repetition + 1, "actual_order": order,
                     "wall_seconds": wall_seconds, "status": result["status"],
                     "steps": result["global_step"],
                     "best_checkpoint_step": best["global_step"] if best else None,
                     "best_validation": best["validation"] if best else None,
                     "last_validation": result["validation"],
                     "last_training_loss": result["history"][-1]["loss"],
                     "result_path": str(trial_dir / "training_result.json"),
                     "result_sha256": file_hash(trial_dir / "training_result.json"),
                     "best_checkpoint_sha256": file_hash(best_path) if best_path else None}
            candidate["trials"].append(trial)
            atomic_json(destination / "measurements.json", candidates)
            print(json.dumps({"candidate": label, "repetition": repetition + 1,
                              "wall_seconds": wall_seconds,
                              "best_validation": trial["best_validation"]}), flush=True)
            verify_unchanged(plan)
    for candidate in candidates.values():
        candidate["median_wall_seconds"] = statistics.median(
            trial["wall_seconds"] for trial in candidate["trials"])
        candidate["validation_reproducible"] = all(
            trial["best_validation"] == candidate["trials"][0]["best_validation"]
            for trial in candidate["trials"])
    winner = select_winner(candidates)
    summary = {"category": "newly_measured_result", "status": "COMPLETED",
               "actually_ran": True, "command": [sys.executable, str(Path(__file__).resolve()),
                                                  "--run-dir", str(destination)],
               "execution_mode": plan["execution_mode"], "device": "cpu",
               "python": platform.python_version(), "torch": torch.__version__,
               "numpy": np.__version__, "cuda_available": torch.cuda.is_available(),
               "cpu_model": cpu_name(), "thread_counts": {"intraop": 1, "interop": 1},
               "source_provenance": plan["source_provenance"],
               "selector_sha256": plan["selector_sha256"],
               "generation_seconds": generation_seconds, "dataset_hash": dataset_hash,
               "generation_hash": generation_hash, "split_hash": split_hash,
               "diagnostic_parents_used_for_selection": False,
               "confirmatory_parents_opened": False,
               "selection_rule": SELECTION_RULE, "candidates": candidates, "winner": winner,
               "claim": "Bounded CPU validation selection only; A100 optimum, speedup and confirmatory claims are unmeasured."}
    if winner is not None:
        selected_path = destination / "selected-config.yaml"
        selected_path.write_text(yaml.safe_dump(candidates[winner]["config"], sort_keys=False))
        summary["selected_config"] = str(selected_path)
        summary["selected_config_hash"] = candidates[winner]["config_hash"]
    verify_unchanged(plan)
    atomic_json(destination / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Fresh directory inside the project; existing runs are preserved")
    args = parser.parse_args(argv)
    try:
        result = measure(args.run_dir)
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Config selection stopped: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "winner": result["winner"],
                      "summary": str(Path(args.run_dir).resolve() / "summary.json")}), flush=True)
    return 0 if result["winner"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
