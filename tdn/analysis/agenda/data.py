"""Independent, separately sealed physical-field cohorts for the agenda.

A parent is a continuous Fourier field, not a grid random-number array.
Training never loads the calibration or confirmation bank. Continuum teachers
use a dealiased spectral equation; they are explicitly a different target from
the original nodal central-difference equation.
"""
from __future__ import annotations

import json
import io
import math
import os
from pathlib import Path

import torch

from tdn.numerics import Equation, Geometry, choose_substeps, refined_reference
from tdn.numerics.invariants import validate_state
from tdn.research.common import fit_feature_normalization
from tdn.research.experiment import atomic_torch_save, horizon_key, state_digest
from tdn.research.protocol import digest, file_digest
from tdn.runtime.metadata import write_json


SPLITS = ("train", "validation", "calibration", "confirmation")


def _field_terms(parent):
    if "field_terms" in parent:
        return parent["field_terms"]
    generator = torch.Generator().manual_seed(parent["seed"])
    spectrum = parent.get("spectrum", parent.get("regime", "mixed"))
    pairs = {"smooth": [(1, 0), (0, 2), (1, 1)],
             "low": [(1, 0), (0, 2), (1, 1)],
             "mixed": [(1, 0), (0, 3), (5, 2), (3, -4)],
             "high_pair": [(10, 9), (11, 9), (0, 2)],
             "near_nyquist": [(14, 1), (15, 1), (1, 0)],
             "rough": [(1, 0), (3, 2), (7, -4), (11, 9), (14, 1), (5, -13)]}
    modes = pairs.get(spectrum, pairs["mixed"])
    # At fixed total variance, this factor changes the relative harmonic
    # amplitudes independently of phases and diffusion/reaction coefficients.
    ratio = float(parent.get("amplitude", parent.get("amplitude_ratio", .6)))
    phase_factor = float(parent.get("phase", parent.get("phase_offset", 0.)))
    return [{"mode": list(mode), "weight": 1. if i == 0 else ratio / math.sqrt(i + 1),
             "phase": float(2 * math.pi * torch.rand((), generator=generator)) + phase_factor * (i + 1)}
            for i, mode in enumerate(modes)]


def continuous_field(parent, grid):
    """Sample one immutable trigonometric polynomial at any declared grid."""
    grid = tuple(grid) if not isinstance(grid, int) else (grid, grid)
    x, y = torch.meshgrid(*(torch.arange(n, dtype=torch.float64) / n for n in grid), indexing="ij")
    terms = _field_terms(parent)
    variance = float(parent.get("variance", .0025))
    mean = float(parent.get("mean", .5))
    norm = math.sqrt(sum(float(term["weight"])**2 for term in terms) / 2)
    wave = sum(float(term["weight"]) * torch.cos(2 * math.pi *
                   (term["mode"][0] * x + term["mode"][1] * y) + float(term["phase"])) for term in terms)
    state = (mean + math.sqrt(variance) * wave / norm)[None, None]
    validate_state(state)
    return state


def _continuum_multiplier(state, kappa, dt):
    n, m = state.shape[-2:]
    k = torch.fft.fftfreq(n, device=state.device, dtype=state.dtype) * n
    l = torch.fft.fftfreq(m, device=state.device, dtype=state.dtype) * m
    return torch.exp(-4 * math.pi**2 * kappa * dt * (k[:, None].square() + l[None, :].square()))


def continuum_ifrk4(state, horizon, equation, substeps, *, check=None):
    """Independent Lawson RK4 with exact spectral diffusion and 3/2 products.

    The method may undergo stiff order reduction: acceptance is based on the
    actual three-level refinement, never its formal fourth-order label.
    """
    from .physics import dealiased_square
    geometry = Geometry(tuple(state.shape[2:]), (1., 1.))
    dt = float(horizon) / substeps
    full, half = _continuum_multiplier(state, equation.kappa, dt), _continuum_multiplier(state, equation.kappa, dt / 2)
    def flow(u, multiplier):
        return torch.fft.ifftn(torch.fft.fftn(u, dim=(-2, -1)) * multiplier, dim=(-2, -1)).real
    def nonlinear(u):
        return equation.reaction_rate * (u - dealiased_square(u, geometry))
    u = state
    with torch.no_grad():
        for index in range(substeps):
            if check and index % 8 == 0:
                check()
            k1 = nonlinear(u)
            a = flow(u + dt * k1 / 2, half)
            k2 = nonlinear(a)
            b = flow(u, half) + dt * k2 / 2
            k3 = nonlinear(b)
            c = flow(u, full) + dt * flow(k3, half)
            k4 = nonlinear(c)
            u = flow(u, full) + dt / 6 * (flow(k1, full) + 2 * flow(k2 + k3, half) + k4)
        if check:
            check()
    return u


def _continuum_reference(parent, horizon, protocol, budget):
    from .physics import fourier_resample
    largest = max(n if isinstance(n, int) else n[0] for n in protocol["grids"])
    resolutions = protocol.get("continuum_reference_grids", [2 * largest, 4 * largest])
    # Every field is sampled independently from its continuous coefficients;
    # no interpolation of a coarse field can erase high input modes.
    states, records, temporal_difference = [], [], None
    for n in resolutions:
        u = continuous_field(parent, n)
        eq = Equation(parent["kappa"], parent["reaction_rate"])
        count = max(8, math.ceil(4 * horizon * max(eq.reaction_rate, 1)))
        accepted = False
        for _ in range(protocol.get("reference_attempts", 5)):
            budget.check()
            if 4 * count > protocol.get("max_finest_substeps", 32768):
                break
            values = [continuum_ifrk4(u, horizon, eq, steps, check=budget.check) for steps in (count, 2 * count, 4 * count)]
            diffs = [float((values[i + 1] - values[i]).square().mean().sqrt()) for i in range(2)]
            floor = 128 * torch.finfo(torch.float64).eps
            uncertainty = max(diffs[-1], floor)
            accepted = all(bool(torch.isfinite(value).all()) for value in values) and diffs[-1] <= max(diffs[0] / 4, floor)
            accepted = accepted and uncertainty <= protocol["reference_tolerance"]
            if accepted:
                break
            count *= 2
        if not accepted:
            raise FloatingPointError(f"Continuum temporal reference unresolved for {parent['parent_id']} n={n}, h={horizon}")
        states.append(values[-1])
        temporal_difference = values[-1] - values[-2]
        records.append({"grid": [n, n], "counts": [count, 2 * count, 4 * count], "differences": diffs,
                        "uncertainty_rms": uncertainty, "accepted": accepted})
    spatial_difference = states[0] - fourier_resample(states[-1], states[0].shape[2:])
    spatial_rms, spatial_max = float(spatial_difference.square().mean().sqrt()), float(spatial_difference.abs().max())
    temporal_max = records[-1]["uncertainty_rms"] * math.sqrt(states[-1].numel())
    # Store uncertainty even if it exceeds a target. The target then becomes
    # inconclusive rather than quietly declaring a continuum success.
    return states[-1], {"accepted": True, "temporal_accepted": True,
        "spatial_resolved_at_reference_tolerance": spatial_rms <= protocol["reference_tolerance"],
        "reference_acceptance_scope": "temporal refinement accepted; spatial uncertainty retained per projected target",
        "method": "independent_Lawson_RK4_3level_and_2grid_dealiased_spectral",
        "temporal_refinements": records, "spatial_resolution_difference_rms": spatial_rms,
        "spatial_resolution_difference_max": spatial_max,
        "uncertainty_rms": records[-1]["uncertainty_rms"] + spatial_rms,
        "uncertainty_max_bound": temporal_max + spatial_rms * states[0].shape[-1], "uncertainty_is_certificate": False,
        "_spatial_difference": spatial_difference, "_temporal_difference": temporal_difference}


def _horizons(protocol, split):
    if split in ("train", "validation"):
        hs = protocol[f"{split}_horizons"]
        return sorted(set(hs + [2 * h for h in hs]))
    return sorted({round(sum(schedule), 12) for schedule in protocol.get("confirm_schedules", protocol.get("confirmation_schedules", []))})


def _generate_parent(declared, protocol, budget):
    references = []
    grids = protocol["grids"]
    grids = [n if isinstance(n, int) else n[0] for n in grids]
    budget.check()
    split = declared["split"]
    parent = {**declared, "field_terms": _field_terms(declared), "states": {}, "references": {}}
    sizes = [protocol["train_grid"]] if split in ("train", "validation") else grids
    for n in sizes:
        state = continuous_field(parent, n)
        parent["states"][str(n)] = state
        terms = parent["field_terms"]
        parent.setdefault("initial_grid_metadata", {})[str(n)] = {
            "initial_modes_above_or_at_nyquist": [term["mode"] for term in terms if any(abs(k) >= n / 2 for k in term["mode"])],
            "initial_aliasing_present": any(any(abs(k) >= n / 2 for k in term["mode"]) for term in terms),
            "actual_mean": float(state.mean()), "actual_variance": float((state - state.mean()).square().mean()),
            "continuous_mean": parent["mean"], "continuous_variance": parent["variance"]}
        parent.setdefault("state_digests", {})[str(n)] = state_digest(state)
        geometry, equation = Geometry((n, n), (1., 1.)), Equation(parent["kappa"], parent["reaction_rate"])
        for h in _horizons(protocol, split):
            count = max(8, choose_substeps(h, equation, geometry))
            reference = None
            for _ in range(protocol.get("reference_attempts", 5)):
                budget.check()
                if 4 * count > protocol["max_finest_substeps"]:
                    break
                reference = refined_reference(state, h, equation, geometry, count,
                    tolerance=20 * protocol["reference_tolerance"] / n, check=budget.check)
                if reference.accepted:
                    break
                count *= 2
            row = {"parent_id": parent["parent_id"], "split": split, "grid": [n, n], "horizon": h,
                   "track": "discrete", "accepted": bool(reference and reference.accepted)}
            references.append(row)
            if not row["accepted"]:
                raise FloatingPointError(f"Discrete teacher unresolved for {parent['parent_id']} n={n} h={h}")
            row.update(uncertainty_rms=reference.uncertainty,
                uncertainty_max_bound=reference.uncertainty * n,
                refinement_substeps=list(reference.refinement_substeps), reason=reference.reason,
                state_sha256=state_digest(reference.state))
            parent["references"][f"discrete:{n}:{horizon_key(h)}"] = {**row, "state": reference.state}
    if parent["parent_id"] in protocol.get("continuum_parent_ids", []):
        from .physics import fourier_resample
        for h in _horizons(protocol, split):
            finest, record = _continuum_reference(parent, h, protocol, budget)
            spatial_difference, temporal_difference = record.pop("_spatial_difference"), record.pop("_temporal_difference")
            for n in sizes:
                teacher = fourier_resample(finest, (n, n))
                spatial = fourier_resample(spatial_difference, (n, n))
                temporal = fourier_resample(temporal_difference, (n, n))
                floor = 128 * torch.finfo(torch.float64).eps
                row = {"parent_id": parent["parent_id"], "split": split, "grid": [n, n], "horizon": h,
                       "track": "continuum", **record, "state_sha256": state_digest(teacher)}
                row.update(uncertainty_rms=float(spatial.square().mean().sqrt()) + max(float(temporal.square().mean().sqrt()), floor),
                           uncertainty_max_bound=float(spatial.abs().max()) + max(float(temporal.abs().max()), floor),
                           projected_spatial_uncertainty_rms=float(spatial.square().mean().sqrt()),
                           projected_spatial_uncertainty_max=float(spatial.abs().max()),
                           uncertainty_combination="sum of projected temporal and spatial refinement differences")
                references.append(row)
                parent["references"][f"continuum:{n}:{horizon_key(h)}"] = {**row, "state": teacher}
    return parent, references


def _worker_parent(declared, protocol, deadline):
    from tdn.research.experiment import Budget
    import time
    torch.set_num_threads(1)
    parent, references = _generate_parent(declared, protocol, Budget(max(.001, deadline - time.monotonic())))
    # Return bytes, avoiding torch multiprocessing's shared-storage temporary
    # files. Each worker remains CPU-only and writes no output paths.
    buffer = io.BytesIO()
    torch.save(parent, buffer)
    return buffer.getvalue(), references


def prepare_banks(protocol, path, budget):
    """Generate sealed banks with up to eight allocated CPU worker processes."""
    path = Path(path)
    banks, references, identities = {split: [] for split in SPLITS}, [], set()
    for parent in protocol["parents"]:
        if parent["split"] not in SPLITS or parent["parent_id"] in identities:
            raise ValueError("Cohorts must be parent-disjoint and have supported split names")
        identities.add(parent["parent_id"])
    workers = min(8, int(os.environ.get("TDN_AGENDA_CPUS", "1")), len(protocol["parents"]))
    if workers < 1:
        raise ValueError("Allocated teacher CPU count must be positive")
    completed = {}
    def retain(index, parent, rows):
        completed[index] = parent
        references.extend(rows)
        for split in SPLITS:
            banks[split] = [completed[i] for i in sorted(completed) if completed[i]["split"] == split]
            if banks[split]:
                atomic_torch_save({"protocol_sha256": digest(protocol), "split": split, "parents": banks[split]}, path / f"{split}.partial.pt")
        write_json(path / "references.json", {"rows": references, "completed_parents": len(completed), "worker_processes": workers})
    if workers == 1:
        for index, declared in enumerate(protocol["parents"]):
            parent, rows = _generate_parent(declared, protocol, budget)
            retain(index, parent, rows)
    else:
        from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
        import multiprocessing
        executor = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
        futures = {executor.submit(_worker_parent, declared, protocol, budget.deadline): index
                   for index, declared in enumerate(protocol["parents"])}
        try:
            while futures:
                budget.check()
                done, _ = wait(futures, timeout=.2, return_when=FIRST_COMPLETED)
                for future in done:
                    index = futures.pop(future)
                    payload, rows = future.result()
                    retain(index, torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True), rows)
        except BaseException:
            # SIGUSR1/timeout/failure preserve all completed parents and stop
            # child CPU use promptly; no automatic resubmission is performed.
            for process in tuple((executor._processes or {}).values()):
                process.terminate()
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
    training = banks["train"]
    mean, std = fit_feature_normalization(((p["states"][str(protocol["train_grid"])],
                    Equation(p["kappa"], p["reaction_rate"]), Geometry((protocol["train_grid"],) * 2, (1., 1.))) for p in training),
                    t_ref=protocol["t_ref"], U_ref=protocol["U_ref"])
    write_json(path / "normalization.json", {"mean": mean.tolist(), "std": std.tolist(), "split": "train",
                  "parent_ids": [p["parent_id"] for p in training], "protocol_sha256": digest(protocol)})
    for split in SPLITS:
        atomic_torch_save({"protocol_sha256": digest(protocol), "split": split, "parents": banks[split]}, path / f"{split}.pt")
        partial = path / f"{split}.partial.pt"
        if partial.exists():
            partial.unlink()
    names = [f"{split}.pt" for split in SPLITS] + ["normalization.json", "references.json"]
    write_json(path / "data_manifest.json", {"protocol_sha256": digest(protocol), "artifacts": {name: file_digest(path / name) for name in names},
        "counts": {"parents": {split: len(banks[split]) for split in SPLITS}, "accepted_references": len(references)}})
    return {"parents": {split: len(banks[split]) for split in SPLITS}, "accepted_references": len(references),
            "continuum_parent_ids": protocol.get("continuum_parent_ids", []),
            "reference_tracks": {track: sum(row["track"] == track for row in references) for track in ("discrete", "continuum")}}


def load_bank(protocol, path, split):
    """Read exactly one bank after checking every sealed dataset fingerprint."""
    path = Path(path)
    if split not in SPLITS:
        raise ValueError("Unsupported cohort")
    manifest = json.loads((path / "data_manifest.json").read_text())
    if manifest["protocol_sha256"] != digest(protocol):
        raise ValueError("Data protocol mismatch")
    for name, fingerprint in manifest["artifacts"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or (path / name).is_symlink() or file_digest(path / name) != fingerprint:
            raise ValueError(f"Dataset artifact mismatch: {name}")
    bank = torch.load(path / f"{split}.pt", map_location="cpu", weights_only=True)
    expected = [p["parent_id"] for p in protocol["parents"] if p["split"] == split]
    if bank["protocol_sha256"] != digest(protocol) or bank["split"] != split or [p["parent_id"] for p in bank["parents"]] != expected:
        raise ValueError("Bank parent identities mismatch")
    return bank["parents"]


def load_normalization(protocol, path):
    value = json.loads((Path(path) / "normalization.json").read_text())
    expected = [p["parent_id"] for p in protocol["parents"] if p["split"] == "train"]
    if value["protocol_sha256"] != digest(protocol) or value["split"] != "train" or value["parent_ids"] != expected:
        raise ValueError("Normalization must be fitted exclusively on the training cohort")
    return torch.tensor(value["mean"], dtype=torch.float32), torch.tensor(value["std"], dtype=torch.float32)
