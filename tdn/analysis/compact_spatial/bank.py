"""Deterministic paired fields and complete pre-execution row/timing plans."""
from __future__ import annotations

import math
import torch

from tdn.analysis.work_precision.experiment import cases as review_cases, state as review_state
from .protocol import (METHODS, COMPACT_METHODS, NSTEPS, SCALING_NSTEPS, TARGETS,
                       NORMS, TABLE_IDS, candidate_id, frontier_id)


def cases():
    bank = []
    for old in review_cases():
        bank.append({**old, "case_id": "review/" + old["case_id"],
                     "case_role": "historical_control" if old["case_role"] == "historical_control" else "reused_review",
                     "panel": "accuracy", "batch_size": 1, "cells": math.prod(old["grid"]),
                     "nsteps": list(NSTEPS), "source_spec": old})
    patterns = ("smooth", "broadband", "cutoff_pair", "near_nyquist")
    for dim, grid in ((1, [64]), (2, [24, 28])):
        for pattern_index, pattern in enumerate(patterns):
            for amplitude in (.04, .28):
                bank.append({
                    "case_id": f"new/{dim}d/{pattern}/a{amplitude:.2f}",
                    "case_role": "new_diagnostic", "panel": "accuracy",
                    "grid": grid, "lengths": [1.] * dim, "cells": math.prod(grid), "batch_size": 1,
                    "nsteps": list(NSTEPS), "mean": .4 if pattern_index % 2 == 0 else .6,
                    "amplitude": amplitude, "phase": .193 + .137 * pattern_index,
                    "pattern": pattern, "kappa": .0025 if pattern_index < 2 else .0005,
                    "reaction_rate": 1.3 if pattern_index < 2 else 3.2,
                    "final_time": .16 if pattern_index < 2 else .24,
                    "pairing": "Within-pattern amplitude pairs share all parameters and phase",
                })
    for grid in ([64], [256], [1024], [16, 16], [32, 32], [64, 64]):
        for batch_size in (1, 4, 16):
            bank.append({
                "case_id": f"scaling/{len(grid)}d/" + "x".join(map(str, grid)) + f"/b{batch_size}",
                "case_role": "scaling_control", "panel": "scaling", "grid": grid,
                "lengths": [1.] * len(grid), "cells": math.prod(grid), "batch_size": batch_size,
                "nsteps": list(SCALING_NSTEPS), "mean": .25, "amplitude": .18,
                "phase": .217, "pattern": "scaling", "kappa": .0005,
                "reaction_rate": 2., "final_time": .1,
                "pairing": "Distinct deterministic members; smaller batches are prefixes; physical parameters and continuum pattern fixed across grids",
            })
    return bank


def _pattern(spec, coordinates, phase):
    x, dim = coordinates[0], len(coordinates)
    p = spec["pattern"]
    tau = 2 * math.pi
    if dim == 1:
        if p in ("smooth", "scaling"):
            return (.43 * torch.cos(tau * 2 * x + phase)
                    + .34 * torch.sin(tau * 3 * x - .71)
                    + .23 * torch.cos(tau * 5 * x + .29))
        if p == "broadband":
            return sum(weight * torch.cos(tau * mode * x + phase * mode)
                       for weight, mode in ((.32, 1), (.27, 4), (.23, 9), (.18, 13)))
        frequencies = (9, 10) if p == "cutoff_pair" else (spec["grid"][0] // 2 - 1, spec["grid"][0] // 2 - 2)
        return .5 * torch.cos(tau * frequencies[0] * x + phase) + .5 * torch.sin(tau * frequencies[1] * x - .37)
    y = coordinates[1]
    if p in ("smooth", "scaling"):
        return (.43 * torch.cos(tau * (2 * x + y) + phase)
                + .34 * torch.sin(tau * (x - 3 * y) - .71)
                + .23 * torch.cos(tau * (5 * x - 2 * y) + .29))
    if p == "broadband":
        return sum(weight * torch.cos(tau * (kx * x + ky * y) + phase * (kx + ky))
                   for weight, kx, ky in ((.32, 1, 2), (.27, 4, -3), (.23, 9, 5), (.18, 7, -11)))
    if p == "cutoff_pair":
        return .5 * torch.cos(tau * (9 * x + 9 * y) + phase) + .5 * torch.sin(tau * (10 * x + 10 * y) - .37)
    nx, ny = spec["grid"]
    return (.5 * torch.cos(tau * ((nx // 2 - 1) * x + (ny // 2 - 1) * y) + phase)
            + .5 * torch.sin(tau * ((nx // 2 - 2) * x + (ny // 2 - 2) * y) - .37))


def state(spec, *, dtype=torch.float64):
    if "source_spec" in spec:
        return review_state(spec["source_spec"], dtype=dtype)
    coordinates = torch.meshgrid(*(torch.arange(n, dtype=dtype) / n for n in spec["grid"]), indexing="ij")
    members = []
    for member in range(spec["batch_size"]):
        mean = .25 + .5 * ((member * 7) % 16) / 15 if spec["panel"] == "scaling" else spec["mean"]
        phase = spec["phase"] + .113 * member
        members.append(mean + spec["amplitude"] * _pattern(spec, coordinates, phase))
    return torch.stack(members).unsqueeze(1)


def plan(config):
    bank = cases()
    if config["smoke"]:
        selected = {"review/historical/two_d_rollout", "new/1d/cutoff_pair/a0.28",
                    "new/2d/smooth/a0.04", "scaling/2d/16x16/b4"}
        bank = [spec for spec in bank if spec["case_id"] in selected]
    expected = {name: [] for name in TABLE_IDS}
    orders = []
    for case_index, spec in enumerate(bank):
        case_id = spec["case_id"]
        expected["reference_rows"].append(f"{case_id}/reference")
        expected["candidate_rows"].extend(candidate_id(case_id, variant, n) for n in spec["nsteps"] for variant in METHODS)
        expected["frontier_rows"].extend(frontier_id(case_id, variant, norm, tol)
            for variant in METHODS for norm in NORMS for tol in TARGETS)
        expected["parity_rows"].extend(f"{case_id}/identity/{dtype}/gl3_fused" for dtype in ("float64", "float32"))
        expected["parity_rows"].extend(f"{case_id}/approximation/{variant}" for variant in COMPACT_METHODS)
        expected["parity_rows"].extend(f"{case_id}/rollout_fused/n{n}" for n in spec["nsteps"])
        for step_index, nsteps in enumerate(spec["nsteps"]):
            base = (case_index + step_index) % len(METHODS)
            rotations = [(base + repeat + 1) % len(METHODS) for repeat in range(config["timing_repeats"])]
            orders.append({"case_id": case_id, "nsteps": nsteps,
                "preparation_order": list(METHODS[base:] + METHODS[:base]),
                "warmup_order": list(METHODS[base:] + METHODS[:base]),
                "measured_orders": [list(METHODS[shift:] + METHODS[:shift]) for shift in rotations]})
    return {"cases": bank, "expected_ids": expected, "timing_orders": orders,
            "parity": {"identity_nsteps": 4, "relative_defect_resolution_factor": 64,
                       "dtype_absolute_tolerances": {"float64": 2e-12, "float32": 3e-7},
                       "rollout_absolute_tolerance": 2e-11}}
