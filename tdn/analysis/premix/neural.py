"""Frozen, bounded learned comparison of interaction-before-compression models.

The independent teachers solve the *coupled PDE*, not a GL3 correction. Every
validation-selected checkpoint is frozen before diagnostics are evaluated. The
FNO is a representative repository implementation with the same physical split
and output constraints; this experiment does not reproduce the FNO paper.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
import time

import torch

from tdn.numerics import Equation, Geometry, choose_substeps, refined_reference
from tdn.numerics.invariants import validate_state
from tdn.research.common import fit_feature_normalization
from tdn.research.experiment import Budget, atomic_torch_save, horizon_key, state_digest
from tdn.research.protocol import assert_parent_disjoint, digest, file_digest
from tdn.runtime.metadata import write_json
from tdn.runtime.precision import reference_precision
from tdn.reporting import emit


SCHEMA = "tdn.premix-neural/v1"
FAMILIES = ("premix", "premix_local", "precompress", "fno", "cnn")
CLASSICAL = ("strang", "etdrk4", "gl3_fused")
REGIMES = (
    ("smooth", "favorable", "in_distribution", .003, 2.),
    ("mixed", "typical", "in_distribution", .003, 2.),
    ("inband_pair", "typical", "in_distribution", .003, 2.),
    ("high_pair", "adverse", "heldout_spectrum", .003, 2.),
    ("near_nyquist", "adverse", "heldout_spectrum", .003, 2.),
    ("rough", "adverse", "in_distribution", .003, 2.),
    ("bounds_low", "adverse", "heldout_amplitude_mean", .003, 2.),
    ("bounds_high", "adverse", "heldout_amplitude_mean", .003, 2.),
    ("reaction_stiff", "adverse", "heldout_reaction_rate", .003, 12.),
    ("diffusion_stiff", "adverse", "heldout_diffusion", .02, 2.),
    ("long_rollout", "adverse", "heldout_horizon", .003, 2.),
    ("grid_transfer", "adverse", "heldout_resolution", .003, 2.),
)


def build_protocol(profile: str = "smoke") -> dict:
    """Return the exact protocol; changing it requires a new version, not tuning."""
    if profile not in ("smoke", "full"):
        raise ValueError("Neural premix profile must be smoke or full")
    smoke = profile == "smoke"
    grid = 8 if smoke else 32
    parents = []
    train_styles = ("smooth", "mixed", "inband_pair", "rough")
    for split, count, offset in (("train", 4 if smoke else 48, 810000),
                                  ("validation", 4 if smoke else 24, 820000)):
        for index in range(count):
            regime = train_styles[index % 4]
            category = "favorable" if regime == "smooth" else "adverse" if regime == "rough" else "typical"
            parents.append({"parent_id": f"{split}-{offset + index}", "seed": offset + index,
                "split": split, "regime": regime, "category": category,
                "distribution": "development", "grid": [grid, grid],
                "kappa": (.002, .003, .004)[(index // 4) % 3],
                "reaction_rate": (1., 2., 3.)[(index // 12) % 3],
                "horizons": [.08, .16, .32] if split == "train" else [.12, .24]})
    for index, (regime, category, distribution, kappa, rate) in enumerate(REGIMES):
        for replicate in range(1 if smoke else 4):
            seed = 830000 + 100 * index + replicate
            size = 2 * grid if regime == "grid_transfer" else grid
            horizon = .96 if regime == "long_rollout" else .24
            parents.append({"parent_id": f"diagnostic-{regime}-{seed}", "seed": seed,
                "split": "diagnostic", "regime": regime, "category": category,
                "distribution": distribution, "grid": [size, size], "kappa": kappa,
                "reaction_rate": rate, "horizons": [horizon]})
    assert_parent_disjoint(parents)
    return {"schema": SCHEMA, "version": 1, "profile": profile, "parents": parents,
        "families": list(FAMILIES), "classical": list(CLASSICAL),
        "seeds": [840011] if smoke else [840011, 840021, 840031],
        "width": 8 if smoke else 16, "modes": 1 if smoke else 4,
        "t_ref": .2, "U_ref": 1., "updates": 2 if smoke else 300,
        "validation_every": 1 if smoke else 50, "learning_rate": .001,
        "train_horizons": [.08, .16], "validation_horizons": [.12],
        "loss_scale": 2e-4, "two_step_loss_weight": .5,
        "step_counts": [1, 2] if smoke else [1, 2, 4],
        "long_step_counts": [4, 8] if smoke else [4, 8, 16],
        "targets": [2e-3, 2e-4, 2e-5, 2e-6],
        "reference_tolerance": 2e-7, "reference_attempts": 5,
        "max_finest_substeps": 32768, "prepare_seconds": 1200,
        "run_seconds": 1200, "timing_repeats": 1 if smoke else 3,
        "timing_warmup": 1, "state_precision": "float32", "teacher_precision": "float64",
        "timing_order": "deterministic per-parent method shuffle; consecutive repeats within a method",
        "resolution_design": "independent heldout 64-square parents, not paired discretizations of the same initial field",
        "tf32": False,
        "normalization": "full training initial states only; shared across every family and seed",
        "checkpoint_selection": "minimum validation loss, including initialization; all frozen before diagnostics",
        "matching": "same data, loss, update count, sample schedule, validation schedule, split core, precision and device",
        "unmatched": "parameter counts, architecture FLOPs and training walltime; recorded without capacity-matching claims",
        "reference": "same-grid independent coupled RK4 n/2n/4n; conservative estimate, not a certificate",
        "frontier": "post-hoc reference-informed selection on a fixed step grid; not an adaptive policy",
        "categories": "a priori mechanistic favorable/typical/adverse labels, not observed rankings or universal worst cases",
        "scope": "bounded periodic logistic reaction-diffusion comparison; no new PDE/boundary-condition or FNO-paper/SOTA claim"}


def validate_protocol(protocol: dict) -> dict:
    if not isinstance(protocol, dict) or protocol.get("profile") not in ("smoke", "full"):
        raise ValueError("Malformed neural premix protocol")
    expected = build_protocol(protocol["profile"])
    if digest(protocol) != digest(expected):
        raise ValueError("Neural premix protocol differs from the immutable declared plan")
    return protocol


def _table(path: Path, rows: list[dict]) -> None:
    write_json(path, {"schema": SCHEMA, "rows": rows})


def _begin(path: Path, protocol: dict) -> None:
    validate_protocol(protocol)
    path.mkdir(parents=True, exist_ok=True)
    if any((path / name).exists() for name in ("summary.json", "dataset.pt", "training.json", "COMPLETED")):
        raise FileExistsError("Preserve existing premix artifacts; use a fresh run directory")
    existing = path / "protocol.json"
    if existing.exists() and digest(json.loads(existing.read_text())) != digest(protocol):
        raise ValueError("Existing protocol differs from the requested immutable experiment")
    write_json(existing, protocol)


def _manifest(path: Path, names: list[str], protocol: dict, filename: str) -> None:
    write_json(path / filename, {"schema": SCHEMA, "protocol_sha256": digest(protocol),
        "artifacts": {name: file_digest(path / name) for name in names}})


def _verify_manifest(path: Path, protocol: dict, filename: str, expected: set[str] | None = None) -> dict:
    manifest = json.loads((path / filename).read_text())
    if manifest.get("protocol_sha256") != digest(protocol):
        raise ValueError("Artifact manifest belongs to a different protocol")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or (expected is not None and set(artifacts) != expected):
        raise ValueError("Artifact manifest file plan differs from the expected files")
    for name, fingerprint in artifacts.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not (path / relative).resolve().is_relative_to(path.resolve()):
            raise ValueError("Artifact manifest escapes the run directory")
        if file_digest(path / relative) != fingerprint:
            raise ValueError(f"Artifact integrity mismatch: {name}")
    return manifest


def _initial(parent: dict) -> torch.Tensor:
    generator = torch.Generator().manual_seed(parent["seed"])
    x, y = torch.meshgrid(*(torch.arange(n, dtype=torch.float64) / n for n in parent["grid"]), indexing="ij")
    phase = torch.rand(4, generator=generator, dtype=torch.float64) * (2 * math.pi)
    regime, n = parent["regime"], parent["grid"][0]
    if regime == "smooth":
        value = .5 + .015 * torch.sin(2 * math.pi * x + phase[0]) + .01 * torch.cos(2 * math.pi * y + phase[1])
    elif regime in ("inband_pair", "high_pair", "near_nyquist"):
        first = 1 if regime == "inband_pair" else max(2, n // 3 - 1) if regime == "high_pair" else n // 2 - 2
        value = .5 + .12 * torch.sin(2 * math.pi * first * (x + y) + phase[0]) + .09 * torch.cos(2 * math.pi * (first + 1) * (x + y) + phase[1])
    elif regime == "rough":
        value = .1 + .8 * torch.rand(parent["grid"], generator=generator, dtype=torch.float64)
    elif regime.startswith("bounds_"):
        wave = (.5 + .25 * torch.sin(2 * math.pi * x + phase[0]) + .25 * torch.cos(2 * math.pi * y + phase[1]))
        value = .005 + .07 * wave
        if regime == "bounds_high":
            value = 1 - value
    else:
        value = (.5 + .16 * torch.sin(2 * math.pi * x + phase[0]) + .1 * torch.cos(4 * math.pi * y + phase[1])
                 + .07 * torch.sin(6 * math.pi * (x + y) + phase[2]))
    value = value[None, None]
    validate_state(value)
    return value


def _summary(path, protocol, *, stage, status, device, start, error=None, **values):
    result = {"schema": SCHEMA, "status": status, "stage": stage, "profile": protocol["profile"],
        "device": device, "elapsed_seconds": time.monotonic() - start, "scope": protocol["scope"],
        "scientific_outcome": "DESCRIPTIVE_BOUNDED_COMPARISON" if status == "COMPLETED" and stage == "neural" else "NOT_ESTABLISHED",
        **values}
    if error is not None:
        result["error"] = f"{type(error).__name__}: {error}"
    write_json(path / "summary.json", result)
    return result


def _status(error):
    return "INTERRUPTED" if isinstance(error, (InterruptedError, KeyboardInterrupt, TimeoutError)) else "FAILED"


class _RunBudget(Budget):
    """Observe CUDA memory at safe boundaries without claiming kernel peaks."""

    def __init__(self, seconds, stop, device):
        super().__init__(seconds, stop)
        self.device = device
        self.memory = {"enabled": device == "cuda", "observations": 0,
                       "scope": "CUDA allocator peaks and physical device use sampled at safe boundaries"}
        self.last_observation = 0.
        if device == "cuda":
            free, total = torch.cuda.mem_get_info(device)
            self.memory.update(initial_free_bytes=free, total_bytes=total,
                               soft_budget_bytes=int(min(30 * 2**30, .8 * total, .8 * free)),
                               hard_device_used_fraction=.9, peak_allocated_bytes=0,
                               peak_reserved_bytes=0, peak_sampled_device_used_bytes=0)
            torch.cuda.reset_peak_memory_stats(device)

    def observe(self, *, force=False):
        if self.device != "cuda" or (not force and time.monotonic() - self.last_observation < .1):
            return
        self.last_observation = time.monotonic()
        free, total = torch.cuda.mem_get_info(self.device)
        self.memory["observations"] += 1
        for name, value in (("peak_allocated_bytes", torch.cuda.max_memory_allocated(self.device)),
                            ("peak_reserved_bytes", torch.cuda.max_memory_reserved(self.device)),
                            ("peak_sampled_device_used_bytes", total - free)):
            self.memory[name] = max(self.memory[name], value)
        if self.memory["peak_reserved_bytes"] > self.memory["soft_budget_bytes"] or total - free >= .9 * total:
            raise MemoryError("CUDA memory budget exceeded; preserve partial artifacts and use a fresh bounded run")

    def check(self):
        super().check()
        self.observe()


def prepare(protocol: dict, run_dir: Path, *, stop=None) -> dict:
    """CPU-only teacher generation, with all accepted references sealed for reuse."""
    path, start = Path(run_dir), time.monotonic()
    _begin(path, protocol)
    budget = Budget(protocol["prepare_seconds"], stop)
    dataset = {"schema": SCHEMA, "protocol_sha256": digest(protocol), "parents": []}
    records, seen = [], set()
    try:
        for declared in protocol["parents"]:
            budget.check()
            state = _initial(declared)
            fingerprint = state_digest(state)
            if fingerprint in seen:
                raise ValueError("Duplicate actual initial fields across supposedly independent parents")
            seen.add(fingerprint)
            parent = {**declared, "initial": state, "initial_state_sha256": fingerprint, "references": {}}
            equation, geometry = Equation(declared["kappa"], declared["reaction_rate"]), Geometry(tuple(declared["grid"]), (1., 1.))
            for horizon in declared["horizons"]:
                count = max(8, choose_substeps(horizon, equation, geometry))
                attempts = []
                reference = None
                for _ in range(protocol["reference_attempts"]):
                    budget.check()
                    if count * 4 > protocol["max_finest_substeps"]:
                        break
                    reference = refined_reference(state, horizon, equation, geometry, count,
                        tolerance=protocol["reference_tolerance"] / math.sqrt(math.prod(declared["grid"])), check=budget.check)
                    attempts.append({"substeps": list(reference.refinement_substeps),
                        "differences": [x if math.isfinite(x) else None for x in reference.refinement_differences],
                        "reason": reference.reason, "accepted": reference.accepted})
                    if reference.accepted:
                        break
                    count *= 2
                accepted = reference is not None and reference.accepted
                record = {key: declared[key] for key in ("parent_id", "split", "regime", "category", "grid")}
                record.update(horizon=horizon, accepted=accepted, attempts=attempts,
                    uncertainty_rms=reference.uncertainty if reference is not None and math.isfinite(reference.uncertainty) else None,
                    uncertainty_max_bound=reference.uncertainty * math.sqrt(state.numel()) if reference is not None and math.isfinite(reference.uncertainty) else None,
                    reference_state_sha256=state_digest(reference.state) if accepted else None)
                records.append(record)
                _table(path / "references.json", records)
                if not accepted:
                    raise FloatingPointError(f"Independent reference acceptance failed for {declared['parent_id']} at {horizon}; training withheld")
                parent["references"][horizon_key(horizon)] = {**record, "state": reference.state.cpu()}
            dataset["parents"].append(parent)
            atomic_torch_save(dataset, path / "dataset.partial.pt")
            emit({"accepted_references": len(records), "completed_parents": len(dataset["parents"])},
                 phase="premix-references", step=len(dataset["parents"]), completed=len(dataset["parents"]),
                 total=len(protocol["parents"]), unit="parents")
        train = [parent for parent in dataset["parents"] if parent["split"] == "train"]
        mean, std = fit_feature_normalization(((parent["initial"], Equation(parent["kappa"], parent["reaction_rate"]),
            Geometry(tuple(parent["grid"]), (1., 1.))) for parent in train), t_ref=protocol["t_ref"], U_ref=protocol["U_ref"])
        write_json(path / "normalization.json", {"mean": mean.tolist(), "std": std.tolist(), "split": "train",
            "parent_ids": [parent["parent_id"] for parent in train], "protocol_sha256": digest(protocol),
            "scope": protocol["normalization"]})
        atomic_torch_save(dataset, path / "dataset.pt")
        (path / "dataset.partial.pt").unlink()
        summary = _summary(path, protocol, stage="dataset", status="COMPLETED", device="cpu", start=start,
            counts={"parents": len(dataset["parents"]), "references": len(records), "accepted_references": len(records)})
        _manifest(path, ["protocol.json", "dataset.pt", "normalization.json", "references.json", "summary.json"], protocol, "dataset_manifest.json")
        return summary
    except BaseException as error:
        _summary(path, protocol, stage="dataset", status=_status(error), device="cpu", start=start, error=error,
            counts={"completed_parents": len(dataset["parents"]), "references": len(records)})
        raise


def _load_dataset(protocol: dict, path: Path) -> tuple[dict, tuple]:
    _verify_manifest(path, protocol, "dataset_manifest.json", {"protocol.json", "dataset.pt", "normalization.json", "references.json", "summary.json"})
    if digest(json.loads((path / "protocol.json").read_text())) != digest(protocol):
        raise ValueError("Dataset protocol mismatch")
    source_summary = json.loads((path / "summary.json").read_text())
    if source_summary.get("status") != "COMPLETED" or source_summary.get("stage") != "dataset":
        raise ValueError("Neural training requires a completed, accepted dataset")
    dataset = torch.load(path / "dataset.pt", map_location="cpu", weights_only=True)
    if dataset.get("protocol_sha256") != digest(protocol) or len(dataset.get("parents", [])) != len(protocol["parents"]):
        raise ValueError("Dataset identity or parent count mismatch")
    assert_parent_disjoint(dataset["parents"])
    seen = set()
    for parent, declared in zip(dataset["parents"], protocol["parents"]):
        if {key: parent.get(key) for key in declared} != declared:
            raise ValueError("Dataset parent plan mismatch")
        state = parent["initial"]
        if state.dtype != torch.float64 or list(state.shape) != [1, 1, *declared["grid"]]:
            raise ValueError("Dataset field shape or precision mismatch")
        fingerprint = state_digest(state)
        if fingerprint != parent["initial_state_sha256"] or fingerprint in seen:
            raise ValueError("Dataset initial field identity mismatch or duplicate")
        seen.add(fingerprint)
        validate_state(state)
        if set(parent["references"]) != {horizon_key(h) for h in declared["horizons"]}:
            raise ValueError("Dataset reference horizons mismatch")
        for reference in parent["references"].values():
            if reference["accepted"] is not True or reference["state"].dtype != torch.float64 or reference["state"].shape != state.shape:
                raise ValueError("Dataset reference acceptance/shape/precision mismatch")
            if state_digest(reference["state"]) != reference["reference_state_sha256"]:
                raise ValueError("Dataset reference state digest mismatch")
    normalization = json.loads((path / "normalization.json").read_text())
    expected_ids = [parent["parent_id"] for parent in protocol["parents"] if parent["split"] == "train"]
    if normalization.get("parent_ids") != expected_ids or normalization.get("split") != "train" or normalization.get("protocol_sha256") != digest(protocol):
        raise ValueError("Normalization is not the shared training-only fit")
    return dataset, (torch.tensor(normalization["mean"], dtype=torch.float32), torch.tensor(normalization["std"], dtype=torch.float32))


def _sync(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _safe(model, state, h, equation, geometry):
    value = model(state, h, equation, geometry)
    try:
        validate_state(value)
    except ValueError as error:
        raise FloatingPointError(f"Neural trajectory inadmissible: {error}") from error
    return value


def _loss(model, parent, h, protocol, device):
    equation, geometry = Equation(parent["kappa"], parent["reaction_rate"]), Geometry(tuple(parent["grid"]), (1., 1.))
    initial = parent["initial"].to(device=device, dtype=torch.float32)
    first = _safe(model, initial, h, equation, geometry)
    second = _safe(model, first, h, equation, geometry)
    target1 = parent["references"][horizon_key(h)]["state"].to(device=device, dtype=torch.float32)
    target2 = parent["references"][horizon_key(2 * h)]["state"].to(device=device, dtype=torch.float32)
    loss = ((first - target1).square().mean() + protocol["two_step_loss_weight"] * (second - target2).square().mean()) / protocol["loss_scale"]**2
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite physical PDE endpoint loss")
    return loss


def _validation(model, validation, protocol, device, budget):
    model.eval()
    values = []
    with torch.no_grad():
        for parent in validation:
            for horizon in protocol["validation_horizons"]:
                budget.check()
                values.append(float(_loss(model, parent, horizon, protocol, device)))
    return statistics.fmean(values)


def _state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _train(family, seed, dataset, normalization, protocol, path, device, budget):
    from tdn.research.premix_neural import build_model
    _sync(device)
    start = time.monotonic()
    torch.manual_seed(seed)
    model = build_model(family, width=protocol["width"], modes=protocol["modes"], normalization=normalization,
                        t_ref=protocol["t_ref"], U_ref=protocol["U_ref"]).to(device=device, dtype=torch.float32)
    initial = {name: value.detach().cpu().clone() for name, value in model.named_parameters()}
    train = [parent for parent in dataset["parents"] if parent["split"] == "train"]
    validation = [parent for parent in dataset["parents"] if parent["split"] == "validation"]
    pairs = [(parent, h) for parent in train for h in protocol["train_horizons"]]
    generator = torch.Generator().manual_seed(seed + 1)
    schedule, queue = [], []
    for _ in range(protocol["updates"]):
        if not queue:
            queue = torch.randperm(len(pairs), generator=generator).tolist()
        schedule.append(queue.pop())
    record = {"family": family, "seed": seed, "status": "RUNNING", "steps": 0, "selected_step": None,
        "selected_parameters_changed": False, "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_parent_ids": [parent["parent_id"] for parent in train],
        "validation_parent_ids": [parent["parent_id"] for parent in validation],
        "diagnostics_seen_during_training": False, "sample_schedule_sha256": digest(schedule),
        "history": [], "architecture": model.architecture_metadata() if hasattr(model, "architecture_metadata") else {"family": family}}
    checkpoint = path / "checkpoints" / f"seed-{seed}-{family}.pt"
    record_path = path / "training" / f"seed-{seed}-{family}.json"
    optimizer = torch.optim.Adam(model.parameters(), lr=protocol["learning_rate"])
    best, last_loss = math.inf, None
    _sync(device)
    record["initialization_seconds"] = time.monotonic() - start
    try:
        for step in range(protocol["updates"] + 1):
            budget.check()
            if step:
                parent, horizon = pairs[schedule[step - 1]]
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = _loss(model, parent, horizon, protocol, device)
                loss.backward()
                if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()):
                    raise FloatingPointError("Nonfinite parameter gradient")
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
                optimizer.step()
                budget.check()
                record["steps"], last_loss = step, float(loss.detach())
            if step % protocol["validation_every"] == 0 or step == protocol["updates"]:
                try:
                    score = _validation(model, validation, protocol, device, budget)
                except FloatingPointError:
                    score = math.inf
                record["history"].append({"step": step, "validation_loss": score if math.isfinite(score) else None, "training_loss": last_loss})
                if math.isfinite(score) and score < best:
                    best = score
                    record["selected_step"] = step
                    atomic_torch_save({"family": family, "seed": seed, "protocol_sha256": digest(protocol),
                        "selected_step": step, "validation_loss": best, "state_dict": _state(model)}, checkpoint)
                write_json(record_path, record)
                emit({"optimizer_steps": step, "finite_validation": int(math.isfinite(score)),
                      **({"training_loss": last_loss} if last_loss is not None else {}),
                      **({"validation_loss": score} if math.isfinite(score) else {})},
                     phase=f"premix-train-{seed}-{family}", step=step,
                     completed=step, total=protocol["updates"], unit="updates")
        if record["selected_step"] is None:
            raise FloatingPointError("No finite admissible validation checkpoint")
        if not any(not torch.equal(initial[name], value.detach().cpu()) for name, value in model.named_parameters()):
            raise FloatingPointError("Optimizer completed without changing any trainable parameter")
        selected = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(selected["state_dict"], strict=True)
        model.eval()
        record.update(status="COMPLETED", checkpoint_sha256=file_digest(checkpoint),
            selected_parameters_changed=any(not torch.equal(initial[name], value.detach().cpu()) for name, value in model.named_parameters()),
            best_validation_loss=best)
        record["selection"] = "TRAINED_CHECKPOINT" if record["selected_step"] > 0 and record["selected_parameters_changed"] else "SELECTED_INITIALIZATION"
    except FloatingPointError as error:
        record.update(status="NUMERICAL_FAILURE", error=str(error), selection="NO_ELIGIBLE_CHECKPOINT")
        model = None
    except BaseException as error:
        _sync(device)
        record.update(status=_status(error), error=f"{type(error).__name__}: {error}", training_seconds=time.monotonic() - start)
        write_json(record_path, record)
        raise
    _sync(device)
    record["training_seconds"] = time.monotonic() - start
    write_json(record_path, record)
    return model, record


def _errors(value, reference):
    difference = value.detach().double().cpu() - reference["state"]
    mean = difference.mean()
    rms = float(difference.square().mean().sqrt())
    maximum = float(difference.abs().max())
    return {"rms": rms, "max_error": maximum, "mean_error": abs(float(mean)),
        "spatial_rms": float((difference - mean).square().mean().sqrt()),
        "upper_rms": rms + reference["uncertainty_rms"], "upper_max": maximum + reference["uncertainty_max_bound"]}


def _frontiers(rows: list[dict], protocol: dict) -> list[dict]:
    groups = {}
    identity = ("parent_id", "regime", "category", "distribution", "grid", "seed", "family")
    for row in rows:
        key = (row["parent_id"], row["seed"], row["family"])
        groups.setdefault(key, []).append(row)
    result = []
    for candidates in groups.values():
        for norm in ("rms", "max"):
            for target in protocol["targets"]:
                eligible = [row for row in candidates if row["status"] == "COMPLETED" and row.get("memory_ok", True)
                            and row[f"upper_{norm}"] <= target]
                selected = min(eligible, key=lambda row: row["timing"]["wall_seconds_median"]) if eligible else None
                result.append({**{key: candidates[0][key] for key in identity}, "norm": norm, "target": target,
                    "status": "FEASIBLE" if selected else "NO_FEASIBLE_CANDIDATE", "selected_steps": selected["steps"] if selected else None,
                    "seconds": selected["timing"]["wall_seconds_median"] if selected else None,
                    "error_upper": selected[f"upper_{norm}"] if selected else None})
    return result


def _evaluate(protocol, dataset, models, records, path, device, budget, candidates):
    from tdn.analysis.profiling import measure
    from tdn.research.work_precision import prepare_step
    from tdn.research.compact_spatial import prepare_compact_step
    diagnostics = [parent for parent in dataset["parents"] if parent["split"] == "diagnostic"]
    methods = [(None, family, None) for family in CLASSICAL]
    methods.extend((seed, family, models.get((seed, family))) for seed in protocol["seeds"] for family in FAMILIES)
    record_map = {(record["seed"], record["family"]): record for record in records}
    with torch.no_grad():
        for index, parent in enumerate(diagnostics, 1):
            equation, geometry = Equation(parent["kappa"], parent["reaction_rate"]), Geometry(tuple(parent["grid"]), (1., 1.))
            initial = parent["initial"].to(device=device, dtype=torch.float32)
            horizon = parent["horizons"][0]
            reference = parent["references"][horizon_key(horizon)]
            counts = protocol["long_step_counts"] if parent["regime"] == "long_rollout" else protocol["step_counts"]
            order = torch.randperm(len(methods), generator=torch.Generator().manual_seed(parent["seed"] + 99)).tolist()
            for timing_order, method_index in enumerate(order):
                seed, family, model = methods[method_index]
                for steps in counts:
                    budget.check()
                    row = {key: parent[key] for key in ("parent_id", "regime", "category", "distribution", "grid")}
                    row.update(seed=seed, family=family, steps=steps, horizon=horizon, device=device,
                        state_precision="float32", status="RUNNING", admissible=False,
                        timing_order=timing_order,
                        uncertainty_rms=reference["uncertainty_rms"], uncertainty_max_bound=reference["uncertainty_max_bound"])
                    if seed is not None:
                        row["checkpoint_selection"] = record_map[seed, family]["selection"]
                        if model is None:
                            row.update(status="TRAINING_FAILED", error=record_map[seed, family].get("error"))
                            candidates.append(row)
                            continue
                    try:
                        _sync(device)
                        setup_start = time.perf_counter()
                        if family in CLASSICAL:
                            prepared = (prepare_compact_step(initial, horizon / steps, equation, geometry, "gl3_fused") if family == "gl3_fused"
                                        else prepare_step(initial, horizon / steps, equation, geometry, family))
                        _sync(device)
                        row["setup_seconds"] = time.perf_counter() - setup_start
                        def operation():
                            state = initial.clone()
                            for _ in range(steps):
                                budget.check()
                                if seed is None:
                                    state = prepared(state)
                                    try:
                                        validate_state(state)
                                    except ValueError as error:
                                        raise FloatingPointError(f"Classical output inadmissible: {error}") from error
                                else:
                                    state = _safe(model, state, horizon / steps, equation, geometry)
                            return state
                        value, timing = measure(operation, device=device, warmup=protocol["timing_warmup"], repeats=protocol["timing_repeats"])
                        row.update(status="COMPLETED", admissible=True, timing=timing, **_errors(value, reference),
                                   memory_ok=timing.get("soft_budget_passed", True) and not timing.get("hard_device_memory_warning", False))
                    except FloatingPointError as error:
                        row.update(status="INVALID_TRAJECTORY", error=str(error))
                    except BaseException as error:
                        row.update(status=_status(error), error=f"{type(error).__name__}: {error}",
                                   timing_complete=False)
                        candidates.append(row)
                        raise
                    candidates.append(row)
            _table(path / "candidates.json", candidates)
            _table(path / "frontiers.json", _frontiers(candidates, protocol))
            emit({"candidate_rows": len(candidates), "diagnostic_parents": index}, phase="premix-diagnostics",
                 step=index, completed=index, total=len(diagnostics), unit="parents")


def run(protocol: dict, dataset_dir: Path, run_dir: Path, *, device: str = "cpu", stop=None) -> dict:
    """Actual bounded training and independent diagnosis on the requested device."""
    validate_protocol(protocol)
    if device not in ("cpu", "cuda"):
        raise ValueError("Neural device must be explicitly cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; CPU fallback is forbidden")
    if device == "cuda":
        from tdn.runtime.preflight import verify_runtime
        verify_runtime("cuda", "premix-neural")
    path, source, start = Path(run_dir), Path(dataset_dir), time.monotonic()
    if path.resolve() == source.resolve():
        raise ValueError("Dataset and neural artifacts require separate directories")
    _begin(path, protocol)
    budget = _RunBudget(protocol["run_seconds"], stop, device)
    records, candidates, models = [], [], {}
    reference_precision()
    try:
        dataset, normalization = _load_dataset(protocol, source)
        dataset_hash = file_digest(source / "dataset.pt")
        write_json(path / "dataset_source.json", {"path": str(source.resolve()), "dataset_sha256": dataset_hash,
                   "manifest_sha256": file_digest(source / "dataset_manifest.json"), "protocol_sha256": digest(protocol)})
        for seed in protocol["seeds"]:
            for family in FAMILIES:
                budget.check()
                model, record = _train(family, seed, dataset, normalization, protocol, path, device, budget)
                records.append(record)
                models[seed, family] = model
                budget.observe(force=True)
                write_json(path / "memory.json", budget.memory)
                _table(path / "training.json", records)
        # Freeze and verify *every* candidate before exposing any diagnostic.
        frozen = {f"checkpoints/seed-{record['seed']}-{record['family']}.pt": record["checkpoint_sha256"]
                  for record in records if record["status"] == "COMPLETED"}
        for name, fingerprint in frozen.items():
            if file_digest(path / name) != fingerprint:
                raise ValueError("Checkpoint changed before diagnostic freeze")
        write_json(path / "checkpoint_freeze.json", {"protocol_sha256": digest(protocol), "dataset_sha256": dataset_hash,
            "selection_split": "validation", "all_training_completed_before_diagnostics": True, "artifacts": frozen})
        _evaluate(protocol, dataset, models, records, path, device, budget, candidates)
        budget.observe(force=True)
        write_json(path / "memory.json", budget.memory)
        for name, fingerprint in frozen.items():
            if file_digest(path / name) != fingerprint:
                raise ValueError("Frozen checkpoint changed during diagnostics")
        summary = _summary(path, protocol, stage="neural", status="COMPLETED", device=device, start=start,
            counts={"training_records": len(records), "trained_checkpoints": sum(row.get("selection") == "TRAINED_CHECKPOINT" for row in records),
                    "selected_initializations": sum(row.get("selection") == "SELECTED_INITIALIZATION" for row in records),
                    "numerical_training_failures": sum(row["status"] == "NUMERICAL_FAILURE" for row in records),
                    "candidates": len(candidates), "completed_candidates": sum(row["status"] == "COMPLETED" for row in candidates),
                    "diagnostic_parents": sum(parent["split"] == "diagnostic" for parent in protocol["parents"])},
            actual_training_device=device, actual_neural_training=any(row["steps"] > 0 for row in records),
            pairing="training seeds reuse the same independent diagnostic parents; repeated parents are not independent evidence",
            resolution_design=protocol["resolution_design"],
            preparation_seconds=json.loads((source / "summary.json").read_text())["elapsed_seconds"],
            setup_timing_scope="classical coefficient preparation only; learned model construction/training belongs to training_seconds; steady-state timing excludes preparation for every family",
            frontier_scope=protocol["frontier"], fnopaper_reproduction=False)
        _manifest(path, ["protocol.json", "dataset_source.json", "training.json", "candidates.json", "frontiers.json",
            "checkpoint_freeze.json", "summary.json", "memory.json", *frozen], protocol, "neural_manifest.json")
        return summary
    except BaseException as error:
        # A failed/interrupted family writes its per-family training record;
        # preserve it in the aggregate even when its call did not return.
        known = {(row["seed"], row["family"]) for row in records}
        for record_path in sorted((path / "training").glob("*.json")):
            record = json.loads(record_path.read_text())
            if (record["seed"], record["family"]) not in known:
                records.append(record)
        _table(path / "training.json", records)
        write_json(path / "memory.json", budget.memory)
        _table(path / "candidates.json", candidates)
        _table(path / "frontiers.json", _frontiers(candidates, protocol))
        _summary(path, protocol, stage="neural", status=_status(error), device=device, start=start, error=error,
                 counts={"training_records": len(records), "candidate_rows": len(candidates)})
        raise
