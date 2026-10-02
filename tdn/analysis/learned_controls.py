"""Bounded, matched-budget CPU screening of every implemented neural control.

This is a development runnability/accuracy-cost diagnostic. It does not open
confirmatory parents, submit jobs, approve a GPU campaign, or establish a new
method's efficiency from tiny CPU timings. Failures and infeasible policies are
retained rather than dropped from the comparison.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import time

import torch

from tdn.data import DatasetStore
from tdn.data.provenance import atomic_json, canonical_hash, source_provenance
from tdn.runtime.storage import configure_storage, contained_path
from tdn.train import load_checkpoint, train
from .workflow import evaluate

FAMILIES = ("taylor", "polynomial", "rational", "fixed_rate", "generic_mlp", "temporal_mlp")
MAX_CPU_SCREEN_STEPS = 64
MAX_CPU_SCREEN_CELLS = 4096
MAX_CPU_SCREEN_PARENTS = 32


def _screen_contract(config: dict, device: str) -> None:
    if str(device) != "cpu":
        raise ValueError("Matched controls are a CPU development screen; GPU campaigns require a separately audited budget")
    if config.get("purpose") != "development" or config.get("runtime", {}).get("confirmatory_authorized", False):
        raise ValueError("Matched-control screen is restricted to development with no confirmatory authorization")
    steps = config["training"]["max_steps"]
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= MAX_CPU_SCREEN_STEPS:
        raise ValueError(f"CPU screen requires a prespecified 1..{MAX_CPU_SCREEN_STEPS} optimizer-step budget per family")
    grid = config["problem"]["grid"]
    if len(grid) not in (1, 2) or math.prod(grid) > MAX_CPU_SCREEN_CELLS:
        raise ValueError("CPU screen is restricted to tiny 1-D/2-D grids with at most 4096 cells")
    count = sum(config["data"][split] for split in ("train_count", "validation_count", "diagnostic_count"))
    if count > MAX_CPU_SCREEN_PARENTS:
        raise ValueError("CPU screen allows at most 32 development parents; freeze a separate budget for larger comparisons")
    precision = config["precision"]
    if precision["network_autocast"] != "none" or precision["compile_mode"] != "eager" or precision["tf32"]:
        raise ValueError("CPU screen uses matched eager FP32/FP64 precision without unaudited GPU optimization")
    if config["model"].get("anchored", False) or config["model"].get("anchored_leading_defect", False):
        raise ValueError("Matched PDE controls require unanchored models until an exact discrete e3 audit exists")


def _trained_head_count(payload: dict) -> int:
    return sum(int(torch.count_nonzero(value)) for key, value in payload["model_state"].items()
               if key in ("base.amplitude.weight", "base.amplitude.bias"))


def compare_learned_controls(config: dict, dataset_dir: Path, run_dir: Path,
                             device: str = "cpu") -> dict:
    """Train all six controls sequentially with only model.family changed.

    Parent splits, features, normalization recipe, state/network precision,
    width, seeds, horizons, optimizer budget and evaluation candidates are
    identical. Intrinsic output parameter counts differ by architecture and
    are explicitly recorded. A checkpoint is selected by feasible validation
    only; latest checkpoints without such evidence remain failure diagnostics.
    """
    _screen_contract(config, device)
    configure_storage()
    dataset_dir, run_dir = contained_path(dataset_dir), contained_path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError("Comparison output directory must be empty to preserve existing runs")
    data = DatasetStore(dataset_dir, verify=True)
    if data.manifest["problem"] != config["problem"]:
        raise ValueError("Comparison physical problem differs from immutable dataset")
    if data.manifest["horizons"] != config["horizons"]["values"]:
        raise ValueError("Comparison horizons differ from immutable dataset")
    actual_parents = sum(data.count(split) for split in ("train", "validation", "diagnostic"))
    if actual_parents > MAX_CPU_SCREEN_PARENTS:
        raise ValueError("Immutable dataset exceeds the 32-parent CPU screen budget")
    source = source_provenance()
    design = copy.deepcopy(config)
    design["model"]["family"] = "MATCHED_CONTROL_FAMILY"
    summary = {
        "schema_version": 1, "stage": "compare_learned_controls",
        "evidence_category": "newly measured result", "status": "RUNNING",
        "scope": "bounded CPU development diagnostics; sealed confirmatory parents remain unopened",
        "device": "cpu", "families": list(FAMILIES),
        "matched_design_hash": canonical_hash(design), "source_provenance": source,
        "dataset_hash": data.manifest_hash, "split_hash": data.split_hash,
        "optimizer_steps_per_family": config["training"]["max_steps"],
        "tuning_policy": "only model.family changes; no per-family retuning or diagnostic-parent checkpoint selection",
        "selection_policy": "feasible validation best; otherwise last checkpoint only for failure diagnosis",
        "precision_policy": copy.deepcopy(config["precision"]),
        "feature_policy": "identical legal radius-one features and training-parent-only normalization",
        "normalization_hash": None, "results": [], "accuracy_time_frontier": [],
        "accuracy_memory_scope": "CPU lifetime peak RSS is available in individual evaluations; no CUDA memory measurement",
        "unsupported_optional": ["KAN", "exact nonlinear PDE anchor", "GPU control campaign"],
        "claim": "Tiny CPU control training and measured development frontiers; no confirmed efficiency or GPU result",
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(run_dir / "comparison_config.json", config)
    summary_path = run_dir / "learned_controls.json"
    atomic_json(summary_path, summary)

    for family in FAMILIES:
        if source_provenance()["python_source_hash"] != source["python_source_hash"]:
            summary["status"] = "STOPPED_SOURCE_CHANGED"
            for pending in FAMILIES[len(summary["results"]):]:
                summary["results"].append({"family": pending, "status": "NOT_RUN_SOURCE_CHANGED",
                                           "failure": "Package source changed after comparison started"})
            atomic_json(summary_path, summary)
            return summary
        current = copy.deepcopy(config)
        current["model"]["family"] = family
        family_dir = run_dir / family
        family_dir.mkdir()
        atomic_json(family_dir / "config.json", current)
        record = {"family": family, "status": "RUNNING", "config_hash": canonical_hash(current),
                  "run_dir": str(family_dir), "training": None, "evaluation": None,
                  "failure": None, "checkpoint_selection": None}
        summary["results"].append(record)
        atomic_json(summary_path, summary)
        phase = "training"
        started = time.perf_counter()
        try:
            training = train(current, dataset_dir, family_dir / "training", device="cpu")
            record["training_wall_seconds"] = time.perf_counter() - started
            record["training"] = training
            if training["status"] != "COMPLETE" or training["global_step"] != current["training"]["max_steps"]:
                raise RuntimeError(f"Prespecified optimizer budget did not complete: {training['status']}")
            if training["source_hash"] != source["python_source_hash"]:
                raise RuntimeError("Training source hash differs from the frozen comparison")
            if training["dataset_hash"] != data.manifest_hash or training["split_hash"] != data.split_hash:
                raise RuntimeError("Training parent/dataset provenance differs from the matched design")
            best = training["best_checkpoint"]
            selected = Path(best) if best is not None else Path(training["last_checkpoint"])
            payload = load_checkpoint(selected)
            feasible_best = best is not None and bool((payload.get("validation") or {}).get("feasible"))
            if best is not None and not feasible_best:
                raise RuntimeError("Best checkpoint lacks feasible validation evidence")
            record["checkpoint_selection"] = "validation_feasible_best" if feasible_best else "last_failure_diagnostic_only"
            record["selected_checkpoint"] = str(selected)
            record["trained_head_nonzero_count"] = _trained_head_count(payload)
            record["parameter_count"] = sum(value.numel() for key, value in payload["model_state"].items()
                                             if key.startswith("base.") and key != "base.fixed_rates")
            if record["trained_head_nonzero_count"] == 0:
                raise RuntimeError("Learned amplitude head remained zero after the optimizer budget")
            normalization_hash = canonical_hash(payload["normalization"])
            record["normalization_hash"] = normalization_hash
            if summary["normalization_hash"] is None:
                summary["normalization_hash"] = normalization_hash
            elif summary["normalization_hash"] != normalization_hash:
                raise RuntimeError("Training feature/label normalization differs across controls")
            phase = "evaluation"
            evaluation_start = time.perf_counter()
            evaluation = evaluate(current, dataset_dir, selected, family_dir / "evaluation", device="cpu")
            record["evaluation_wall_seconds"] = time.perf_counter() - evaluation_start
            # The generic evaluator assumes its caller supplied a previously
            # selected model. Keep that provenance factual for an explicitly
            # retained infeasible/latest diagnostic checkpoint as well.
            if evaluation.get("checkpoint") is not None:
                evaluation["checkpoint"]["selection"] = record["checkpoint_selection"]
                evaluation["checkpoint"]["validation_checkpoint_feasible"] = feasible_best
                atomic_json(family_dir / "evaluation" / "evaluation.json", evaluation)
            learned = evaluation["summaries"]["learned"]
            policy = evaluation["selected_policies"]["learned"]
            record["evaluation"] = {"path": str(family_dir / "evaluation" / "evaluation.json"),
                                     "summary": learned, "selected_policy": policy,
                                     "classical_comparisons": evaluation["parent_paired_runtime_comparisons"]}
            feasible = feasible_best and policy["validation_feasible"] and learned["failed_parents"] == 0
            record["status"] = "COMPLETE_FEASIBLE" if feasible else "COMPLETE_INFEASIBLE"
            summary["accuracy_time_frontier"].append({
                "family": family, "status": record["status"],
                "validation_checkpoint_feasible": feasible_best,
                "candidate_h": policy["macrostep"],
                "validation_policy_feasible": policy["validation_feasible"],
                "diagnostic_failed_parents": learned["failed_parents"],
                "diagnostic_worst_weighted_error": learned["worst_measured_error"],
                "diagnostic_median_wall_seconds": learned["median_wall_seconds"],
                "training_wall_seconds": record["training_wall_seconds"],
                "parameter_count": record["parameter_count"],
                "claim_scope": "development diagnostics; single timing repeat per parent/sequence",
            })
        except SystemExit as error:
            # Preserve Slurm safe-boundary pause evidence and the job's exit
            # code. The wrapper never resubmits remaining families implicitly.
            record.update(status="PAUSED_NEEDS_RESUME" if error.code == 75 else "INTERRUPTED",
                          failure={"phase": phase, "type": type(error).__name__, "message": str(error)})
            atomic_json(family_dir / "control_result.json", record)
            summary["status"] = record["status"]
            for pending in FAMILIES[len(summary["results"]):]:
                summary["results"].append({"family": pending, "status": "NOT_RUN_INTERRUPTED"})
            atomic_json(summary_path, summary)
            raise
        except Exception as error:
            record.update(status="FAILED", failure={"phase": phase, "type": type(error).__name__, "message": str(error)})
        atomic_json(family_dir / "control_result.json", record)
        atomic_json(summary_path, summary)
    if source_provenance()["python_source_hash"] != source["python_source_hash"]:
        summary["status"] = "STOPPED_SOURCE_CHANGED"
    else:
        summary["status"] = "COMPLETE" if all(row["status"].startswith("COMPLETE_") for row in summary["results"]) else "COMPLETE_WITH_FAILURES"
    summary["failed_families"] = [row["family"] for row in summary["results"] if row["status"] == "FAILED"]
    summary["tolerance_infeasible_families"] = [row["family"] for row in summary["results"] if row["status"] == "COMPLETE_INFEASIBLE"]
    atomic_json(summary_path, summary)
    return summary
