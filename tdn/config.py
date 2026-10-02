"""Strict versioned schema. Development tolerances are proposals, never claims."""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path
import yaml

SCHEMA = {
    "problem": {"family", "grid", "lengths", "kappa", "reaction_rate", "kappa_range", "reaction_rate_range", "t_ref", "U_ref", "periodic"},
    "horizons": {"values", "rollout_time", "evaluation_steps", "mixed_factors"},
    "data": {"train_count", "validation_count", "diagnostic_count", "confirmatory_count", "anchors_per_parent", "workers", "rollout_windows"},
    "teacher": {"base_substeps", "error_fraction", "tolerance_fraction", "noise_floor"},
    "model": {"family", "width", "modes", "fixed_rates", "chunk_size", "anchored"},
    "training": {"max_steps", "learning_rate", "weight_decay", "grad_clip", "batch_parents", "microbatch_cells", "rollout_windows", "rollout_weight", "checkpoint_every_steps", "checkpoint_every_seconds", "validation_every_steps"},
    "precision": {"state", "teacher", "network_autocast", "compile_mode", "tf32", "checkpoint_chunks"},
    "validation": {"tolerance", "tolerance_provenance", "tolerance_manifest", "require_headroom", "reference_fraction", "bootstrap_samples"},
    "runtime": {"intraop_threads", "interop_threads", "soft_vram_gib", "soft_vram_fraction", "hard_memory_fraction", "confirmatory_authorized"},
}
TOP = {"schema_version", "experiment", "seed", "purpose", *SCHEMA}

def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def _number(x, name, *, positive=False, integer=False):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError(f"{name} must be a finite number")
    if positive and x <= 0:
        raise ValueError(f"{name} must be positive")
    if integer and (not isinstance(x, int) or x < 0):
        raise ValueError(f"{name} must be a nonnegative integer")

def validate_config(config: dict) -> dict:
    c = copy.deepcopy(config)
    if not isinstance(c, dict) or set(c) != TOP:
        raise ValueError(f"Top-level fields must be exactly {sorted(TOP)}")
    for section, fields in SCHEMA.items():
        if not isinstance(c[section], dict) or set(c[section]) != fields:
            raise ValueError(f"{section}: missing or unknown fields; expected {sorted(fields)}")
    if c["schema_version"] != 1:
        raise ValueError("Unsupported schema_version")
    if not isinstance(c["experiment"], str) or not c["experiment"].replace("_", "").replace("-", "").isalnum():
        raise ValueError("experiment must be a safe nonempty identifier")
    _number(c["seed"], "seed", integer=True)
    if c["purpose"] not in ("development", "confirmatory"):
        raise ValueError("purpose must be development or confirmatory")
    p = c["problem"]
    if p["family"] != "logistic_rd" or p["periodic"] is not True:
        raise ValueError("This audited initial pipeline supports periodic logistic_rd only")
    if len(p["grid"]) not in (1, 2, 3) or len(p["lengths"]) != len(p["grid"]):
        raise ValueError("grid and lengths must have matching dimensions, 1 through 3")
    for n in p["grid"]:
        _number(n, "grid", positive=True, integer=True)
        if n < 4: raise ValueError("grid dimensions must be >=4")
    for n in p["lengths"]: _number(n, "lengths", positive=True)
    for name in ("kappa", "reaction_rate", "t_ref", "U_ref"): _number(p[name], name, positive=True)
    for name in ("kappa_range", "reaction_rate_range"):
        if len(p[name]) != 2: raise ValueError(f"{name} requires two endpoints")
        for x in p[name]: _number(x, name, positive=True)
        if p[name][0] > p[name][1]: raise ValueError(f"{name} is reversed")
    for name in ("values", "evaluation_steps", "mixed_factors"):
        values = c["horizons"][name]
        if not isinstance(values, list) or not values: raise ValueError(f"{name} cannot be empty")
        for x in values: _number(x, name, positive=True)
    if len(set(c["horizons"]["values"])) != len(c["horizons"]["values"]): raise ValueError("Duplicate horizons")
    _number(c["horizons"]["rollout_time"], "rollout_time", positive=True)
    for name, value in c["data"].items():
        if name != "rollout_windows": _number(value, name, integer=True)
    if c["data"]["rollout_windows"] != [1,2,4]: raise ValueError("Store audited full-domain rollout windows [1,2,4]")
    if min(c["data"][k] for k in ("train_count", "validation_count", "diagnostic_count")) < 1: raise ValueError("Each development parent split needs >=1 parent")
    if c["data"]["anchors_per_parent"] != 1 or c["data"]["workers"] != 0:
        raise ValueError("Audited exact-resume loader uses one anchor and synchronous workers=0")
    if c["data"]["confirmatory_count"] != 0:
        raise ValueError("Create a separately reviewed immutable confirmatory design after pilot gates; this development generator does not open sealed parents")
    for name, value in c["teacher"].items(): _number(value, name, positive=True, integer=name == "base_substeps")
    if not 0 < c["teacher"]["error_fraction"] <= 0.05: raise ValueError("Teacher uncertainty fraction must be <=0.05")
    if not 0 < c["teacher"]["tolerance_fraction"] <= .1: raise ValueError("Teacher tolerance fraction must be <=0.1")
    if c["model"]["family"] not in ("temporal_mlp", "fixed_rate", "polynomial", "rational", "generic_mlp", "taylor"): raise ValueError("Unsupported or unaudited model family")
    for name in ("width", "modes", "chunk_size"): _number(c["model"][name], name, positive=True, integer=True)
    if c["model"]["anchored"] is not False: raise ValueError("PDE leading coefficient is not yet independently exact; use the audited Tier-A anchor tests")
    if len(c["model"]["fixed_rates"]) != c["model"]["modes"]: raise ValueError("One dictionary rate is required per mode")
    for x in c["model"]["fixed_rates"]: _number(x, "fixed rate", positive=True)
    for name, value in c["training"].items():
        _number(value, name, positive=name not in ("weight_decay", "rollout_weight"), integer=name in ("max_steps", "batch_parents", "microbatch_cells", "rollout_windows", "checkpoint_every_steps", "validation_every_steps"))
        if value < 0: raise ValueError(f"{name} cannot be negative")
    if c["training"]["max_steps"] > 5000: raise ValueError("Prespecified per-run optimizer budget must be <=5000")
    if c["training"]["rollout_windows"] not in (1, 2, 4): raise ValueError("Rollout curriculum supports 1, 2, or 4 full-domain windows")
    precision = c["precision"]
    if precision["teacher"] != "float64" or precision["state"] not in ("float32", "float64"): raise ValueError("Keep teacher FP64 and physical state FP32/64")
    if precision["network_autocast"] not in ("none", "bfloat16"): raise ValueError("Only audited BF16 candidate supported")
    if precision["compile_mode"] not in ("eager", "default", "max-autotune-no-cudagraphs"): raise ValueError("Unsupported compile mode")
    if precision["tf32"] is not False: raise ValueError("TF32 is disabled until a separate parity study")
    for x in (c["validation"]["tolerance"], c["validation"]["reference_fraction"]): _number(x, "validation tolerance/fraction", positive=True)
    if c["validation"]["reference_fraction"] > .1: raise ValueError("Reference fraction must be <=0.1")
    _number(c["validation"]["bootstrap_samples"], "bootstrap_samples", positive=True, integer=True)
    r = c["runtime"]
    for name, value in r.items():
        if name != "confirmatory_authorized": _number(value, name, positive=True, integer=name in ("intraop_threads","interop_threads"))
    if r["soft_vram_gib"] > 30 or r["soft_vram_fraction"] > 0.8 or r["hard_memory_fraction"] > 0.9: raise ValueError("Memory limits exceed protocol caps")
    for section, names in {"precision": ("tf32", "checkpoint_chunks"), "runtime": ("confirmatory_authorized",), "validation": ("require_headroom",)}.items():
        for name in names:
            if not isinstance(c[section][name], bool): raise ValueError(f"{section}.{name} must be boolean")
    if c["purpose"] == "confirmatory":
        raise ValueError("Confirmation is blocked pending a reviewed physical tolerance, parameter design, untouched parent manifest, and measured G0–G6; development configs cannot be relabeled as confirmation")
    return c

def load_config(path: str | Path) -> dict:
    return validate_config(yaml.safe_load(Path(path).read_text()))
