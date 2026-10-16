"""Reproducible discrete parents and source prefixes for frozen scaling P1.

Generation is label-free. Independent RNG streams keep field/physics identity
unchanged by source counts, fitting cohorts, timing or reference acquisition.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from tdn.analysis.response.contracts import Physics
from tdn.numerics.types import Geometry

from .scaling_protocol import FAMILIES, SPLITS


def _rng(seed, stream):
    return np.random.default_rng(np.random.SeedSequence([seed, stream]))


def parent_identity(u, eq, geom):
    """Hash actual physical input, without names that could hide duplicates."""
    values = u.detach().cpu().numpy()
    if values.dtype != np.float32 or values.shape != (1, 1, 5, 5) or not np.isfinite(values).all():
        raise ValueError("Parent identity requires the actual finite FP32 5x5 field")
    field = np.ascontiguousarray(values, dtype="<f4").tobytes(order="C")
    metadata = json.dumps(dict(dtype="little_endian_float32", shape=list(values.shape),
        grid=list(geom.grid), lengths=list(geom.lengths), kappa=float(eq.kappa),
        reaction_rate=float(eq.reaction_rate), target="fd_same_grid"),
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return dict(field_sha256=hashlib.sha256(field).hexdigest(),
                identity_sha256=hashlib.sha256(metadata+b"\n"+field).hexdigest())


def _morphology(seed, family):
    rng = _rng(seed, 101)
    x, y = np.meshgrid(np.arange(5)/5, np.arange(5)/5, indexing="ij")
    phases = rng.uniform(0., 2*np.pi, 4)
    parameters = dict(phases=phases.tolist())
    if family == "smooth_spectral":
        shape = (np.cos(2*np.pi*x+phases[0]) + .7*np.sin(2*np.pi*y+phases[1])
                 + .3*np.cos(2*np.pi*(x+y)+phases[2]))
    elif family == "localized_periodic":
        concentration = float(rng.uniform(1., 2.5))
        shape = np.exp(concentration*(np.cos(2*np.pi*x+phases[0])
                                      + np.cos(2*np.pi*y+phases[1])-2.))
        parameters["concentration"] = concentration
    elif family == "periodic_transition":
        sharpness = float(rng.uniform(1.5, 3.))
        shape = np.tanh(sharpness*(np.cos(2*np.pi*x+phases[0])
                                   + .35*np.sin(2*np.pi*y+phases[1])))
        parameters["sharpness"] = sharpness
    elif family == "multiscale_discrete":
        shape = (np.cos(2*np.pi*(x+y)+phases[0])
                 + .7*np.cos(4*np.pi*x+phases[1])
                 + .5*np.sin(4*np.pi*y+phases[2])
                 + .3*np.cos(2*np.pi*(2*x-y)+phases[3]))
    else:
        raise ValueError("Unknown frozen parent morphology")
    shape -= shape.mean()
    scale = float(np.max(np.abs(shape)))
    if not np.isfinite(scale) or scale <= 0:
        raise ArithmeticError("Degenerate parent morphology")
    return shape/scale, parameters


def parents(config, split):
    if split not in SPLITS:
        raise ValueError("Unknown parent split")
    if config["grid"] != [5, 5] or config["lengths"] != [1., 1.]:
        raise ValueError("P1 parents support the frozen 5x5 periodic unit grid only")
    if config["parent_families"] != list(FAMILIES):
        raise ValueError("P1 parent family order is frozen")
    ranges = config["parent_ranges"]
    for index, seed in enumerate(config["parent_seeds"][split]):
        family = FAMILIES[index % len(FAMILIES)]
        shape, parameters = _morphology(seed, family)
        state_rng, physics_rng = _rng(seed, 102), _rng(seed, 103)
        mean = float(state_rng.uniform(*ranges["mean"]))
        amplitude = float(state_rng.uniform(*ranges["amplitude"]))
        values = np.asarray(mean+amplitude*shape, dtype=np.float32)
        if not np.isfinite(values).all() or np.any(values <= 0) or np.any(values >= 1):
            raise ArithmeticError("Generated parent violates the declared logistic state range")
        u = torch.from_numpy(values.copy())[None, None]
        eq = Physics(float(physics_rng.uniform(*ranges["kappa"])),
                     float(physics_rng.uniform(*ranges["reaction_rate"])))
        geom = Geometry((5, 5), (1., 1.))
        yield dict(parent_id=f"{split}-{index:03d}-{seed}", split=split, seed=seed,
            parent_index=index, index=index, cohort_order=index, parent_family=family,
            u=u, eq=eq, geom=geom, morphology_parameters=parameters,
            requested_mean=mean, requested_amplitude=amplitude,
            actual_mean=float(values.astype(np.float64).mean()),
            actual_variance=float(values.astype(np.float64).var()),
            state_seed=[seed, 102], morphology_seed=[seed, 101], physics_seed=[seed, 103],
            **parent_identity(u, eq, geom))


def source_bank(parent_seed, context_index, role, config):
    if role not in ("calibration", "withheld"):
        raise ValueError("Unknown source role")
    if (type(parent_seed) is not int or parent_seed < 0
            or type(context_index) is not int or not 0 <= context_index < len(config["intervals"])):
        raise ValueError("Source seed/context must be an available nonnegative integer")
    if config["grid"] != [5, 5]:
        raise ValueError("P1 sources require the frozen 5x5 grid")
    rng = np.random.default_rng(np.random.SeedSequence(
        [parent_seed, context_index, 701 if role == "calibration" else 702]))
    families = config[role+"_families"]
    if len(families) != config[role+"_sources"]:
        raise ValueError("Source family count differs from the planned source count")
    x, y = np.meshgrid(np.arange(5)/5, np.arange(5)/5, indexing="ij")
    result = []
    for index, family in enumerate(families):
        if family == "dense_random":
            z = rng.normal(size=(5, 5))
        elif family in ("low_frequency", "high_frequency"):
            k = 1 if family == "low_frequency" else 2
            z = np.cos(2*np.pi*k*(x+y)+rng.uniform(0., 2*np.pi))
        elif family == "constant":
            z = np.ones((5, 5))
        else:
            raise ValueError("Unknown frozen source family")
        z /= np.sqrt(np.mean(z*z))
        result.append(dict(source_id=f"{role}-{index:02d}", source_index=index,
            role=role, family=family, z=torch.from_numpy(z.copy())[None, None]))
    return result
