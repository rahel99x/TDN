"""Prespecified, bounded development comparisons; importable without torch."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

VERSION = 1
FAMILIES = ("confluent_decay", "fixed_decay", "fixed_decay_r", "fixed_undamped", "reaction_clock",
            "reaction_additive", "transport", "temporal_mlp")
NEURAL_BASELINES = ("generic_mlp", "residual_cnn", "unet", "fno",
                    "residual_cnn_split", "unet_split", "fno_split")
NEURAL_FAMILIES = (*FAMILIES, *NEURAL_BASELINES)
CLASSICAL = ("split", "richardson_split", "adaptive_split", "coupled_rk4", "e3_anchor")
REGIMES = (("baseline", .01, 2.), ("intermediate", .03, 6.), ("stiff", .1, 12.))
STATE_CLASSES = ("mixed_frequency", "bounded_random", "boundary")
TOLERANCE = .002
DEFAULT = {
    "protocol_version": VERSION, "grid": [32, 32], "max_steps": 96,
    "width": 16, "learning_rate": .003, "validation_every": 8,
    "max_seconds": 1200, "max_finest_substeps": 4096,
    "families": list(FAMILIES), "tolerance": TOLERANCE,
    "train_horizons": [.02, .08, .16], "heldout_horizons": [.03, .11],
    "rollout_time": .32, "benchmark_steps": [.04, .08, .16, .32],
    "rollout_weight": .5, "timing_repeats": 3,
}


def benchmark_suite(config: dict) -> str:
    return "neural-benchmarks" if config.get("families") == list(NEURAL_FAMILIES) else "architecture"


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def validate_config(config: dict, *, smoke: bool = False) -> dict:
    if not isinstance(config, dict) or set(config) != set(DEFAULT):
        raise ValueError("Research configuration must contain exactly the documented protocol fields")
    checked = json.loads(json.dumps(config, allow_nan=False))
    for key in ("protocol_version", "tolerance", "train_horizons", "heldout_horizons",
                "rollout_time", "benchmark_steps", "rollout_weight"):
        if checked[key] != DEFAULT[key]:
            raise ValueError(f"Research protocol fixes {key}; do not weaken the comparison")
    if checked["grid"] != [32, 32]:
        raise ValueError("Full research uses grid32x32; select --smoke for the integration fixture")
    for key, upper in (("max_steps", 96), ("width", 32), ("validation_every", 16),
                       ("max_seconds", 1200), ("max_finest_substeps", 4096), ("timing_repeats", 5)):
        if type(checked[key]) is not int or not 1 <= checked[key] <= upper:
            raise ValueError(f"{key} must be an integer in [1,{upper}]")
    rate = checked["learning_rate"]
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or not 0 < rate <= .01:
        raise ValueError("learning_rate must lie in (0,.01]")
    if checked["families"] not in (list(FAMILIES), [*FAMILIES, "reaction_hybrid"], list(NEURAL_FAMILIES)):
        raise ValueError("Retain every architecture and its paired controls")
    if smoke:
        checked.update(grid=[8, 8], max_steps=min(2, checked["max_steps"]),
                       validation_every=1, timing_repeats=1, max_seconds=min(180, checked["max_seconds"]))
    return checked


def parent_plan(*, smoke: bool = False) -> list[dict]:
    """A parent means one initial state AND physical regime; no seed is reused."""
    result = []
    for split, offset in (("train", 31000), ("validation", 32000), ("diagnostic", 33000)):
        for i, (regime, kappa, rate) in enumerate(REGIMES):
            for j, state_class in enumerate(STATE_CLASSES):
                if smoke and i != j:
                    continue
                seed = offset + i * len(STATE_CLASSES) + j + 1
                result.append({"parent_id": f"{split}-{regime}-{state_class}-{seed}",
                               "split": split, "seed": seed, "state_class": state_class,
                               "regime": regime, "kappa": kappa, "reaction_rate": rate})
    assert_parent_disjoint(result)
    return result


def assert_parent_disjoint(parents: list[dict]) -> None:
    identities = [parent["parent_id"] for parent in parents]
    seeds = [parent["seed"] for parent in parents]
    if len(set(identities)) != len(identities) or len(set(seeds)) != len(seeds):
        raise ValueError("Parent identities and initial-state seeds must be globally disjoint")
    if {parent["split"] for parent in parents} != {"train", "validation", "diagnostic"}:
        raise ValueError("Training, validation and independent diagnostic parents are required")


def make_protocol(config: dict, *, smoke: bool, software: dict, command: list[str]) -> dict:
    return {"version": VERSION, "config": config, "smoke": smoke,
            "benchmark_suite": benchmark_suite(config),
            "parents": parent_plan(smoke=smoke), "software": software, "command": command,
            "state_precision": "float32", "teacher_precision": "float64", "tf32": False,
            "normalization": "training initial states only; shared across all learned families",
            "rate_sharing": "one learned shared rate across all training regimes; no per-parent rate oracle",
            "training_loss": "physical one-step MSE + 0.5*two-step MSE, divided by tolerance squared",
            "checkpoint_selection": "minimum validation loss only, including initialization; diagnostics after freeze",
            "training_budget": {"optimizer": "Adam", "learning_rate": config["learning_rate"],
                                "maximum_updates_per_family": config["max_steps"],
                                "batch": "one identical scheduled training parent/horizon per update",
                                "initialization_seed": 74001, "sample_schedule_seed": 74002,
                                "matched": "data, updates, objective, validation schedule, horizons and device",
                                "unmatched": "parameter count, FLOPs and training walltime; recorded per family",
                                "scope": "small representative adaptations; no converged or published-SOTA claim"},
            "reference": {"method": "coupled RK4 n,2n,4n", "error_fraction": .05,
                          "tolerance_fraction": .1, "noise_floor": 1e-10},
            "headroom_rule": "both split and Richardson error minus reference uncertainty >0.002 at h=.32,T=.32",
            "cost_rule": "whole FP32 rollout, state checks included, warmup1, all declared step sizes, no fallback",
            "scope": "bounded development hypotheses; neither G0-G6 authorization nor confirmatory evidence"}


def readable_summary(summary: dict) -> str:
    """Report measured development outcomes without hiding numerical failures."""
    device = "GPU" if summary["device"] == "cuda" else "CPU"
    lines = [f"TDN {device} development research: {summary['status']}"]
    if summary["status"] == "SKIPPED":
        lines.extend([summary["reason"], "No GPU numerical work was performed."])
        return "\n".join(lines) + "\n"
    if summary.get("smoke"):
        lines.append("Smoke integration fixture; not the full development comparison.")
    training = summary.get("training")
    if training is None:
        lines.append("Frozen CPU checkpoints benchmarked; no additional neural training.")
    else:
        completed = sum(row["status"] == "COMPLETED" for row in training)
        lines.append(f"Completed neural training: {completed}/{len(training)} families.")
        for row in training:
            selected = row.get("selected_step")
            selection = ("selected initialization (step 0)" if selected == 0 else
                         f"selected step {selected}" if selected is not None else "no selected checkpoint")
            suffix = f"; {row['error']}" if row.get("error") else ""
            lines.append(f"  {row['family']}: {row['status']}; {selection}{suffix}")
    headroom = summary["headroom"]
    count = len(headroom["case_ids"])
    total = summary["diagnostic_parent_count"]
    lines.append(f"Classical headroom: {'PASS' if headroom['passed'] else 'NO HEADROOM'} ({count}/{total} diagnostic parents).")
    comparisons = summary["comparisons"]
    eligible = sum(row["eligible"] for row in comparisons)
    faster = sum(row["eligible"] and row["twenty_percent_faster"] for row in comparisons)
    lines.append(f"Eligible matched-accuracy comparisons: {eligible}/{len(comparisons)} family-parent pairs.")
    lines.append(f"At least 20% faster than the best tested classical control: {faster}/{eligible} eligible comparisons.")
    lines.append(f"Invalid timed trajectories retained: {summary['failed_trajectories']}.")
    lines.extend([
        "Hypotheses: H1 confluent_decay vs fixed_decay_r; H2 transport vs e3_anchor and classical controls;",
        "            H3 reaction_clock vs reaction_additive.",
        "Inspect heldout.json for independent one-/two-step errors; frontier.json for full rollout errors and timings.",
        "This development comparison is not confirmatory evidence or authorization for longer training.",
    ])
    return "\n".join(lines) + "\n"


def verify_artifacts(run_dir: Path, *, source_tree_sha256: str | None = None) -> dict:
    """Verify an immutable completed run using only stdlib; safe on login nodes."""
    run_dir = Path(run_dir)
    if run_dir.is_symlink():
        raise ValueError("Research run directories cannot be symlinks")
    manifest_path = run_dir / "manifest.json"
    if (run_dir / "COMPLETED").read_text().strip() != file_digest(manifest_path):
        raise ValueError("Completed research manifest digest mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != VERSION or not isinstance(manifest.get("files"), dict):
        raise ValueError("Unsupported or incomplete research artifact manifest")
    required = {"protocol.json", "dataset.pt", "summary.json"}
    if not required <= set(manifest["files"]):
        raise ValueError("Research manifest omits required artifacts")
    for name, expected in manifest["files"].items():
        path = run_dir / name
        if Path(name).is_absolute() or ".." in Path(name).parts or path.is_symlink():
            raise ValueError("Unsafe research artifact path")
        if not path.resolve().is_relative_to(run_dir.resolve()) or file_digest(path) != expected:
            raise ValueError(f"Research artifact digest mismatch: {name}")
    protocol = json.loads((run_dir / "protocol.json").read_text())
    if digest(protocol) != manifest["protocol_sha256"]:
        raise ValueError("Research protocol digest mismatch")
    if source_tree_sha256 is not None and manifest["source_tree_sha256"] != source_tree_sha256:
        raise ValueError("Research source changed; generate a new CPU run before benchmarking")
    return manifest
