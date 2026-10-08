"""Independent FP64 teachers and immutable, split-disjoint continuous fields.

The teacher is Lawson RK4, independently implemented from evaluated ETDRK4
and split methods. Acceptance uses actual timestep refinement and, for a
continuum estimate, actual spatial refinement. Differences are uncertainty
estimates, never certificates. Unresolved teachers are retained and cannot
silently become supervised labels.
"""
from __future__ import annotations

import json
import io
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from tdn.analysis.agenda.data import continuous_field
from tdn.analysis.agenda.physics import dealiased_square, fourier_resample
from tdn.analysis.roadmap.core import check, clean
from tdn.numerics import Equation, Geometry
from tdn.research.experiment import horizon_key, state_digest
from tdn.research.protocol import digest, file_digest
from tdn.runtime.metadata import write_json

SPLITS = ("development", "train", "validation", "calibration", "confirmation", "scaling", "policy")
CONFIRMATION_STAGES = {"confirm_prepare", "confirm", "scaling", "policy", "report"}
TARGETS = {"discrete": "central_difference_laplacian_and_nodal_logistic",
           "continuum": "projected_spatially_refined_dealiased_spectral_Galerkin"}


def _safe_file(base, name):
    base, relative = Path(base), Path(name)
    if not name or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe sealed artifact path")
    path = base / relative
    if not path.is_file() or base.is_symlink() or any(
            (base / Path(*relative.parts[:i])).is_symlink() for i in range(1, len(relative.parts) + 1)):
        raise ValueError("Missing or unsafe sealed artifact")
    if not path.resolve().is_relative_to(base.resolve()):
        raise ValueError("Artifact escaped its stage")
    return path


def validate_cohorts(protocol):
    """Repeated physics may share a field within a split, never across splits."""
    identities, owners, seeds, field_values = set(), {}, {}, {}
    for parent in protocol["parents"]:
        if parent["parent_id"] in identities or parent["split"] not in SPLITS:
            raise ValueError("Unique parent IDs and supported splits required")
        identities.add(parent["parent_id"])
        for key in ("field_cluster", "continuous_field_id"):
            identity = (key, parent[key])
            if identity in owners and owners[identity] != parent["split"]:
                raise ValueError("Continuous fields must be split-disjoint")
            owners[identity] = parent["split"]
        seed = int(parent.get("field_seed", parent.get("seed")))
        if seed in seeds and seeds[seed] != parent["split"]:
            raise ValueError("Field seeds must be split-disjoint")
        seeds[seed] = parent["split"]
        if "field_terms" in parent:
            coefficients = digest({k: parent[k] for k in ("field_terms", "mean", "variance")})
            if coefficients in field_values and field_values[coefficients] != parent["split"]:
                raise ValueError("Identical continuous coefficients must be split-disjoint")
            field_values[coefficients] = parent["split"]
    training_physics = {(p["kappa"], p["reaction_rate"]) for p in protocol["parents"] if p["split"] == "train"}
    if any(p.get("distribution") == "coefficient_shift" and (p["kappa"], p["reaction_rate"]) in training_physics
           for p in protocol["parents"]):
        raise ValueError("Coefficient-shift labels require physics absent from training")
    return {split: len({p["field_cluster"] for p in protocol["parents"] if p["split"] == split})
            for split in SPLITS}


def field_state(parent, n):
    """Evaluate frozen continuous coefficients, without clipping or grid noise."""
    declaration = dict(parent)
    declaration.setdefault("seed", int(parent.get("field_seed", 0)))
    return continuous_field(declaration, n)


def geometry(parent, n):
    return Geometry((int(n), int(n)), tuple(parent.get("lengths", (1., 1.))))


def reference_horizons(protocol, split):
    if split in ("train", "validation"):
        times = protocol[f"{split}_horizons"]
    elif split == "development":
        times = protocol.get("screening", protocol.get("screen", {})).get("horizons", [.08])
    elif split == "scaling":
        times = [protocol["scaling"]["horizon"]]
    elif split in ("calibration", "policy"):
        times = [protocol["policies"]["horizon"]]
    elif split == "confirmation":
        times = [math.fsum(schedule[:index]) for schedule in protocol["confirm_schedules"]
                 for index in range(1, len(schedule) + 1)]
    else:
        times = [sum(s) for s in protocol["confirm_schedules"]]
    return sorted({round(float(h), 12) for h in times})


def _norms(value):
    return {"rms": float(value.square().mean().sqrt()), "max": float(value.abs().max())}


def lawson_reference(initial, horizon, equation, geom, count, track, *, check_budget=None):
    """Independent exact-linear-flow RK4; no neural/evaluated solver imports."""
    if track not in TARGETS or initial.dtype != torch.float64 or initial.device.type != "cpu":
        raise ValueError("Teacher requires declared target and CPU FP64")
    if type(count) is not int or count < 1 or not math.isfinite(horizon) or horizon < 0:
        raise ValueError("Finite nonnegative horizon and positive step count required")
    symbol = torch.zeros(geom.grid, dtype=torch.float64)
    for axis, (n, length) in enumerate(zip(geom.grid, geom.lengths)):
        k = torch.fft.fftfreq(n, dtype=torch.float64) * n
        shape = [1] * geom.ndim
        shape[axis] = n
        wave = -4 * (n / length)**2 * torch.sin(math.pi * k / n).square() if track == "discrete" else -(2 * math.pi * k / length).square()
        symbol += equation.kappa * wave.reshape(shape)
    dt = horizon / count
    full, half = torch.exp(dt * symbol), torch.exp(dt * symbol / 2)
    def flow(u, multiplier):
        return torch.fft.ifftn(torch.fft.fftn(u, dim=(-2, -1)) * multiplier, dim=(-2, -1)).real
    def nonlinear(u):
        square = u.square() if track == "discrete" else dealiased_square(u, geom)
        return equation.reaction_rate * (u - square)
    answer = initial.clone()
    with torch.no_grad():
        for index in range(count):
            if check_budget and index % 8 == 0:
                check_budget()
            k1 = nonlinear(answer)
            k2 = nonlinear(flow(answer + dt * k1 / 2, half))
            k3 = nonlinear(flow(answer, half) + dt * k2 / 2)
            k4 = nonlinear(flow(answer, full) + dt * flow(k3, half))
            answer = flow(answer, full) + dt / 6 * (flow(k1, full) + 2 * flow(k2 + k3, half) + k4)
    if check_budget:
        check_budget()
    return answer


def _teacher_options(protocol):
    options = protocol.get("reference", {})
    return dict(tolerance=float(options.get("tolerance", protocol.get("reference_tolerance", 1e-8))),
        substeps=int(options.get("substeps", protocol.get("reference_substeps", 8))),
        max_substeps=int(options.get("max_substeps", protocol.get("reference_max_substeps", protocol.get("max_finest_substeps", 512)))),
        roundoff_floor=float(protocol.get("reference_roundoff_floor", 1e-12)),
        spatial_factors=options.get("spatial_factors", protocol.get("reference_spatial_factors", [2, 4])))


def _temporal_teacher(initial, h, equation, geom, track, options, budget):
    base = max(options["substeps"], math.ceil(2 * h * equation.reaction_rate), 1)
    tolerance = options["tolerance"]
    attempts, last = [], None
    while 4 * base <= options["max_substeps"]:
        counts = [base, 2 * base, 4 * base]
        states = [lawson_reference(initial, h, equation, geom, count, track,
                    check_budget=budget.check) for count in counts]
        finite = all(bool(torch.isfinite(value).all()) for value in states)
        floor = max(options.get("roundoff_floor", 0.), 128 * torch.finfo(torch.float64).eps * max(1., float(states[-1].abs().max())))
        diffs = [_norms(states[i + 1] - states[i]) for i in range(2)] if finite else [dict(rms=math.inf, max=math.inf)] * 2
        converged = finite and all(max(diffs[0][key], diffs[1][key]) <= floor or
            diffs[1][key] <= diffs[0][key] / 4 for key in ("rms", "max"))
        unc = {key: max(diffs[-1][key], floor) for key in ("rms", "max")}
        accepted = converged and all(v <= tolerance for v in unc.values())
        attempts.append(dict(counts=counts, differences=clean(diffs), rounding_allowance=floor,
                             finite=finite, converged=converged, accepted=accepted))
        last = (states[-1], dict(accepted=accepted, uncertainty_rms=unc["rms"],
             uncertainty_max_bound=unc["max"], temporal_refinements=attempts,
             refinement_substeps=counts, temporal_accepted=accepted,
             _difference=states[-1] - states[-2]))
        if accepted or not finite:
            break
        base *= 2
    if last is None:
        return initial.clone(), dict(accepted=False, temporal_accepted=False, uncertainty_rms=None,
            uncertainty_max_bound=None, temporal_refinements=[], refinement_substeps=[],
            reason="reference_budget_has_no_three_level_trial", _difference=torch.zeros_like(initial))
    return last


def generate_reference(parent, n, h, track, protocol, budget, *, _cache=None):
    """Return one safe teacher; unresolved estimates are explicit, never labels."""
    started = time.perf_counter()
    equation, geom = Equation(parent["kappa"], parent["reaction_rate"]), geometry(parent, n)
    options = _teacher_options(protocol)
    if options["tolerance"] <= 0 or options["substeps"] < 1:
        raise ValueError("Teacher tolerance/count must be positive")
    if track == "discrete":
        state, meta = _temporal_teacher(field_state(parent, n), h, equation, geom, track, options, budget)
        meta.pop("_difference")
    elif track == "continuum":
        factors = options["spatial_factors"]
        if len(factors) != 2 or any(type(f) is not int or f < 1 for f in factors) or factors[1] != 2 * factors[0]:
            raise ValueError("Continuum requires two genuinely doubled spatial grids")
        values, temporal, cache_hits = [], [], 0
        # The final uncertainty adds two temporal estimates and a spatial
        # difference. Allocate temporal tolerances within the total budget;
        # individually accepting two estimates at the full budget would cause
        # avoidable unusable labels when their sum exceeds it.
        temporal_options = {**options, "tolerance": options["tolerance"] / 4}
        cache = {} if _cache is None else _cache
        for factor in factors:
            size = factor * n
            cache_key = (size, float(h), track)
            if cache_key not in cache:
                cache[cache_key] = _temporal_teacher(field_state(parent, size), h, equation,
                    geometry(parent, size), track, temporal_options, budget)
            else:
                cache_hits += 1
            value, record = cache[cache_key]
            values.append(value); temporal.append(record)
        state = fourier_resample(values[-1], (n, n))
        spatial = fourier_resample(values[-1], (n, n)) - fourier_resample(values[0], (n, n))
        # Both temporal uncertainties contribute to the spatial comparison.
        # Max bounds use a conservative sqrt(numel) RMS relation after projection.
        temporal_projected = [_norms(fourier_resample(record["_difference"], (n, n))) for record in temporal]
        floor = max(options["roundoff_floor"], 128 * torch.finfo(torch.float64).eps)
        spatial_error = _norms(spatial)
        uncertainties = {norm: spatial_error[norm] + sum(max(row[norm], floor) for row in temporal_projected)
                         for norm in ("rms", "max")}
        accepted = all(t["accepted"] for t in temporal) and all(v <= options["tolerance"] for v in uncertainties.values())
        meta = dict(accepted=accepted, temporal_accepted=all(t["accepted"] for t in temporal),
            uncertainty_rms=uncertainties["rms"], uncertainty_max_bound=uncertainties["max"],
            spatial_reference_grids=[n * f for f in factors], spatial_refinement=spatial_error,
            temporal_refinements=[{k: v for k, v in t.items() if k != "_difference"} for t in temporal],
            shared_temporal_teacher_cache_hits=cache_hits,
            uncertainty_combination="projected spatial difference plus BOTH projected temporal differences and rounding floors")
    else:
        raise ValueError("Unknown reference track")
    finite = bool(torch.isfinite(state).all())
    meta["accepted"] = bool(meta["accepted"] and finite)
    return dict(parent_id=parent["parent_id"], field_cluster=parent["field_cluster"], track=track,
        grid=int(n), horizon=float(h), state=state, state_sha256=state_digest(state),
        method="independent_FP64_Lawson_RK4_actual_three_step_refinement",
        target_convention=TARGETS[track], uncertainty_is_certificate=False, **meta,
        finite=finite, teacher_seconds=time.perf_counter() - started)


def _parent_grids(protocol, split):
    if split in ("train", "validation"):
        return [protocol["train_grid"]]
    if split == "development":
        return protocol.get("screening", protocol.get("screen", {})).get("grids", protocol["grids"])
    if split == "scaling":
        return protocol["scaling"]["grids"]
    if split == "policy":
        n = protocol["policies"]["grid"]
        return [n, 2 * n]
    return protocol["grids"]


def generate_parent(declared, protocol, budget):
    parent = {**declared, "states": {}, "references": {}, "state_digests": {}}
    cache = {}
    for n in _parent_grids(protocol, declared["split"]):
        state = field_state(declared, n)
        parent["states"][str(n)] = state
        parent["state_digests"][str(n)] = state_digest(state)
        for h in reference_horizons(protocol, declared["split"]):
            for track in protocol.get("tracks", list(TARGETS)):
                budget.check()
                parent["references"][f"{track}:{n}:{horizon_key(h)}"] = generate_reference(declared, n, h, track, protocol, budget, _cache=cache)
    return parent


def _write_parent(path, index, parent):
    """NPZ contains numeric arrays only; JSON carries all descriptions."""
    arrays, metadata = {}, {k: v for k, v in parent.items() if k not in ("states", "references")}
    metadata["states"], metadata["references"] = {}, {}
    for key, state in parent["states"].items():
        name = f"state_{len(arrays)}"; arrays[name] = state.numpy(); metadata["states"][key] = name
    for key, row in parent["references"].items():
        name = f"reference_{len(arrays)}"; arrays[name] = row["state"].numpy()
        metadata["references"][key] = {**{k: v for k, v in row.items() if k != "state"}, "array": name}
    stem = f"parent-{index:05d}"
    with (path / f"{stem}.npz").open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    write_json(path / f"{stem}.json", clean(metadata))
    return {"parent_id": parent["parent_id"], "split": parent["split"], "metadata": stem + ".json", "arrays": stem + ".npz"}


def _worker_parent(declared, protocol, deadline):
    """Byte-only IPC avoids tensor sharing files outside the project root."""
    from tdn.research.experiment import Budget
    torch.set_num_threads(1)
    parent = generate_parent(declared, protocol, Budget(max(.001, deadline - time.monotonic())))
    buffer = io.BytesIO()
    torch.save(parent, buffer)
    return buffer.getvalue()


def _prepared_parents(ctx, declarations):
    """Parallelism cannot exceed actual allocated CPUs; non-Slurm defaults one."""
    requested = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    workers = min(8, max(1, requested), len(declarations) or 1)
    if workers == 1:
        for parent in declarations:
            yield generate_parent(parent, ctx.protocol, ctx.budget)
        return
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    # Budget has a documented monotonic deadline. Preserve the stage limit in
    # every process instead of giving every parent a fresh walltime budget.
    deadline = ctx.budget.deadline
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(_worker_parent, parent, ctx.protocol, deadline) for parent in declarations]
        for future in futures:
            ctx.budget.check()
            yield torch.load(io.BytesIO(future.result()), weights_only=True, map_location="cpu")


def verify_frozen_training(ctx):
    base = Path(ctx.prerequisites["train"])
    catalog_path, freeze_path = _safe_file(base, "catalog.json"), _safe_file(base, "freeze.json")
    catalog, freeze = json.loads(catalog_path.read_text()), json.loads(freeze_path.read_text())
    if any(value.get("protocol_sha256") != digest(ctx.protocol) for value in (catalog, freeze)) or freeze.get("catalog_sha256") != file_digest(catalog_path):
        raise ValueError("Frozen training protocol or catalog changed")
    if catalog.get("selection_split") != "validation" or catalog.get("frozen_before_confirmation") is not True:
        raise ValueError("Only validation-selected models may precede confirmation")
    if catalog.get("training_plan_sha256") != file_digest(_safe_file(base, "training_plan.json")):
        raise ValueError("Frozen training plan changed")
    records = catalog.get("records", [])
    identities = [r["model_id"] for r in records]
    if not records or len(set(identities)) != len(identities) or set(freeze.get("checkpoint_hashes", {})) != set(identities):
        raise ValueError("Frozen checkpoint coverage incomplete or duplicated")
    from .neural import expected_model_specs
    key = lambda r: (r["family"], r.get("seed"), r["track"], r["train_count"])
    specs = expected_model_specs(ctx.protocol)
    if len(records) != len(specs) or {key(r) for r in records} != {key(r) for r in specs} or set(identities) != {r["model_id"] for r in specs}:
        raise ValueError("Frozen training plan does not cover every declared model")
    for row in records:
        actual = file_digest(_safe_file(base, row["checkpoint"]))
        if actual != row["checkpoint_sha256"] or actual != freeze["checkpoint_hashes"][row["model_id"]]:
            raise ValueError("Frozen checkpoint bytes changed")
    return dict(catalog_sha256=file_digest(catalog_path), freeze_sha256=file_digest(freeze_path),
                checkpoint_hashes=freeze["checkpoint_hashes"], selection_split="validation")


def prepare(ctx, confirmation=False, *, splits=None):
    counts = validate_cohorts(ctx.protocol)
    frozen = verify_frozen_training(ctx) if confirmation else None
    splits = list(splits or (["confirmation", "scaling", "policy"] if confirmation else ["train", "validation", "calibration"]))
    if any(s in ("confirmation", "scaling", "policy") for s in splits) and not confirmation:
        raise ValueError("Fresh confirmation requires frozen training")
    path = Path(ctx.path); path.mkdir(parents=True, exist_ok=True)
    if (path / "data_manifest.json").exists() or list(path.glob("parent-*.json")):
        raise ValueError("Refusing to overwrite an existing data bank")
    records, artifacts, accepted, unresolved = [], {}, 0, 0
    declared = [p for p in ctx.protocol["parents"] if p["split"] in splits]
    for index, parent in enumerate(_prepared_parents(ctx, declared)):
        declaration = declared[index]
        ctx.budget.check()
        record = _write_parent(path, index, parent); records.append(record)
        for name in (record["metadata"], record["arrays"]):
            artifacts[name] = file_digest(path / name)
        for key, ref in parent["references"].items():
            accepted += ref["accepted"]; unresolved += not ref["accepted"]
            gate_ids = ["G2", "G3"] if confirmation else ["G2"]
            if declaration["split"] == "scaling":
                gate_ids.append("G4")
            ctx.record(f"teacher/{declaration['parent_id']}/{key}", gate_ids,
                metrics={k: v for k, v in ref.items() if k != "state"},
                config={"split": declaration["split"], "track": ref["track"], "grid": ref["grid"], "horizon": ref["horizon"]},
                checks=[check("actual-refinement-accepted", ref["accepted"], True, "eq", category="math"),
                    check("usable-independent-teacher", ref["accepted"], True, "eq", category="gap")])
    manifest = dict(schema="tdn.frontier-data/v1", protocol_sha256=digest(ctx.protocol), stage=ctx.stage,
        splits=splits, records=records, artifacts=artifacts, independent_clusters={s: counts[s] for s in splits},
        accepted_references=accepted, unresolved_references=unresolved, frozen_training=frozen,
        uncertainty_is_certificate=False, storage="numeric_NPZ_allow_pickle_false_and_JSON")
    write_json(path / "data_manifest.json", manifest)
    return dict(parents={s: sum(p["split"] == s for p in declared) for s in splits},
        accepted_references=accepted, unresolved_references=unresolved, training_labels_ready=unresolved == 0,
        data_manifest_sha256=file_digest(path / "data_manifest.json"))


def load_split(ctx, split):
    if split not in SPLITS:
        raise ValueError("Unsupported split")
    if split in ("confirmation", "scaling", "policy") and ctx.stage not in CONFIRMATION_STAGES:
        raise ValueError("Confirmation access forbidden before model freeze")
    validate_cohorts(ctx.protocol)
    stage = "confirm_prepare" if split in ("confirmation", "scaling", "policy") else "screen" if split == "development" else "prepare"
    base = Path(ctx.path) if ctx.stage == stage else Path(ctx.prerequisites[stage])
    manifest = json.loads(_safe_file(base, "data_manifest.json").read_text())
    if manifest.get("schema") != "tdn.frontier-data/v1" or manifest.get("protocol_sha256") != digest(ctx.protocol) or manifest.get("stage") != stage or split not in manifest["splits"]:
        raise ValueError("Data manifest protocol/stage/split changed")
    expected = [p for p in ctx.protocol["parents"] if p["split"] == split]
    records = [r for r in manifest["records"] if r["split"] == split]
    if [r["parent_id"] for r in records] != [r["parent_id"] for r in expected]:
        raise ValueError("Data bank has incomplete or duplicate parents")
    required = {r[name] for r in manifest["records"] for name in ("metadata", "arrays")}
    if set(manifest["artifacts"]) != required:
        raise ValueError("Data bank artifact inventory differs")
    for name, fingerprint in manifest["artifacts"].items():
        if file_digest(_safe_file(base, name)) != fingerprint:
            raise ValueError("Data bank bytes changed")
    if split in ("confirmation", "scaling", "policy") and manifest["frozen_training"] != verify_frozen_training(ctx):
        raise ValueError("Confirmation was generated for different frozen models")
    parents = []
    for record, declaration in zip(records, expected):
        parent = json.loads(_safe_file(base, record["metadata"]).read_text())
        if any(parent.get(k) != v for k, v in declaration.items()):
            raise ValueError("Declared parent metadata changed")
        with np.load(_safe_file(base, record["arrays"]), allow_pickle=False) as arrays:
            parent["states"] = {k: torch.from_numpy(arrays[v].copy()) for k, v in parent["states"].items()}
            for row in parent["references"].values():
                row["state"] = torch.from_numpy(arrays[row.pop("array")].copy())
        grids = _parent_grids(ctx.protocol, split)
        keys = {f"{track}:{n}:{horizon_key(h)}" for n in grids for h in reference_horizons(ctx.protocol, split) for track in ctx.protocol["tracks"]}
        if set(parent["states"]) != {str(n) for n in grids} or set(parent["references"]) != keys:
            raise ValueError("Data bank is missing declared grids or teacher horizons")
        for n, state in parent["states"].items():
            if state.dtype != torch.float64 or not bool(torch.isfinite(state).all()) or state_digest(state) != state_digest(field_state(declaration, int(n))):
                raise ValueError("Initial continuous field changed")
        for row in parent["references"].values():
            if row["state"].dtype != torch.float64 or state_digest(row["state"]) != row["state_sha256"]:
                raise ValueError("Reference state changed")
            if row["accepted"] and (not bool(torch.isfinite(row["state"]).all()) or any(
                row[k] is None or not math.isfinite(row[k]) or row[k] < 0 for k in ("uncertainty_rms", "uncertainty_max_bound"))):
                raise ValueError("Accepted reference has invalid uncertainty")
            if split in ("train", "validation") and not row["accepted"]:
                raise ValueError("Unresolved reference cannot be used as a training/selection label")
        parents.append(parent)
    return parents
