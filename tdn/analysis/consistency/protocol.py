"""Immutable plan for the fresh, bounded structural-consistency comparison."""
from __future__ import annotations

from tdn.research.protocol import assert_parent_disjoint, digest

SCHEMA = "tdn.consistency-neural/v1"
FAMILIES = ("premix", "fno", "premix_gated", "fno_gated", "premix_moment", "fno_moment", "precompress")
CLASSICAL = ("strang", "etdrk4", "gl3_fused")
DEVELOPMENT_REGIMES = ("smooth", "mixed", "inband_pair", "rough", "bounds_low", "bounds_high")
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


def expected_case_ids() -> list[str]:
    """Declare every required structural check before running the audit."""
    result = []
    for precision in ("float32", "float64"):
        for grid in ("16", "9x10"):
            result.extend(f"helper/{precision}/{grid}/{case}" for case in
                          ("commutator_identity", "commutator_sign", "periodic_telescoping"))
        result.extend(f"helper/{precision}/{case}" for case in
                      ("small_amplitude_commutator_order", "small_amplitude_gate_order"))
        result.extend(f"helper/{precision}/calibrate_mean/{index}" for index in range(6))
        result.extend(f"helper/{precision}/mean_shape_target/{index}" for index in range(5))
        result.append(f"helper/{precision}/mean_shape_zero_identity")
        for family in FAMILIES:
            result.extend(f"model/{family}/{precision}/{case}" for case in
                          ("constant", "zero_reaction", "zero_diffusion", "zero_horizon", "finite_gradients", "bounded_output"))
            if family.endswith("_moment"):
                result.append(f"model/{family}/{precision}/final_mean_target")
    return result


def expected_required_case_ids() -> list[str]:
    controls = {f"model/{family}/{precision}/{case}" for family in ("premix", "fno", "precompress")
                for precision in ("float32", "float64")
                for case in ("constant", "zero_reaction", "zero_diffusion")}
    return [identity for identity in expected_case_ids() if identity not in controls]


def build_protocol(profile: str = "smoke") -> dict:
    if profile not in ("smoke", "full"):
        raise ValueError("Consistency profile must be smoke or full")
    smoke, parents = profile == "smoke", []
    grid = 8 if smoke else 32
    train_horizons, validation_horizons = [.04, .08, .16, .24], [.06, .12, .20]
    for split, count, offset, horizons in (("train", 6 if smoke else 48, 910000, train_horizons),
                                           ("validation", 6 if smoke else 24, 920000, validation_horizons)):
        for index in range(count):
            regime = DEVELOPMENT_REGIMES[index % len(DEVELOPMENT_REGIMES)]
            cycle = index // len(DEVELOPMENT_REGIMES)
            if regime == "bounds_low":
                mean, amplitude = (.025, .05, .08, .10)[cycle % 4], (.012, .025, .035, .045)[cycle % 4]
            elif regime == "bounds_high":
                mean, amplitude = 1 - (.025, .05, .08, .10)[cycle % 4], (.012, .025, .035, .045)[cycle % 4]
            else:
                mean = (.15, .35, .65, .85)[cycle % 4]
                amplitude = min((.025, .10, .25)[cycle % 3], .8 * min(mean, 1 - mean))
            kappa = (.001, .003, .009)[(index + cycle) % 3]
            rate = (.5, 2., 6.)[(index // 2 + cycle) % 3]
            teachers = sorted(set(horizons + [2 * horizon for horizon in horizons]))
            parents.append({"parent_id": f"{split}-{offset + index}", "seed": offset + index,
                "split": split, "regime": regime,
                "category": "favorable" if regime == "smooth" else "typical" if regime in ("mixed", "inband_pair") else "adverse",
                "distribution": "development", "grid": [grid, grid], "mean": mean, "amplitude": amplitude,
                "kappa": kappa, "reaction_rate": rate, "horizons": teachers,
                "dimensionless": [{"horizon": h, "reaction_h": rate * h,
                                    "diffusion_h_over_dx2": kappa * h * grid**2} for h in horizons]})
    for index, (regime, category, distribution, kappa, rate) in enumerate(REGIMES):
        for replicate in range(1 if smoke else 4):
            seed = 930000 + 100 * index + replicate
            size = 2 * grid if regime == "grid_transfer" else grid
            horizon = .96 if regime == "long_rollout" else .24
            parents.append({"parent_id": f"diagnostic-{regime}-{seed}", "seed": seed,
                "split": "diagnostic", "regime": regime, "category": category, "distribution": distribution,
                "grid": [size, size], "kappa": kappa, "reaction_rate": rate, "horizons": [horizon],
                "dimensionless": [{"horizon": horizon, "reaction_h": rate * horizon,
                                    "diffusion_h_over_dx2": kappa * horizon * size**2}]})
    assert_parent_disjoint(parents)
    return {"schema": SCHEMA, "version": 1, "profile": profile, "parents": parents,
        "audit_case_ids": expected_case_ids(),
        "audit_required_case_ids": expected_required_case_ids(),
        "families": list(FAMILIES), "classical": list(CLASSICAL),
        "seeds": [940011] if smoke else [940011, 940021, 940031],
        "width": 8 if smoke else 16, "modes": 1 if smoke else 4,
        "t_ref": .2, "U_ref": 1., "updates": 2 if smoke else 300,
        "validation_every": 1 if smoke else 50, "learning_rate": .001,
        "train_horizons": train_horizons, "validation_horizons": validation_horizons,
        "loss_scale": 2e-4, "two_step_loss_weight": .5,
        "step_counts": [1, 2] if smoke else [1, 2, 4],
        "long_step_counts": [4, 8] if smoke else [4, 8, 16],
        "targets": [2e-3, 2e-4, 2e-5, 2e-6],
        "reference_tolerance": 2e-7, "reference_attempts": 5,
        "max_finest_substeps": 32768, "audit_seconds": 1200, "prepare_seconds": 1200, "run_seconds": 1200,
        "timing_repeats": 1 if smoke else 3, "timing_warmup": 1,
        "state_precision": "float32", "teacher_precision": "float64", "tf32": False,
        "normalization": "full training initial states only; shared across every family and seed",
        "checkpoint_selection": "minimum validation loss including initialization; all attempts frozen before fresh diagnostics",
        "matching": "same data, loss, update count, sample schedule, validation schedule, split core, precision and device",
        "unmatched": "stored parameter capacity, architecture FLOPs and training walltime; recorded without capacity-matching claims",
        "reference": "same-grid independent coupled RK4 n/2n/4n; conservative estimate, not a certificate",
        "frontier": "post-hoc reference-informed selection on a fixed step grid; not an adaptive policy",
        "categories": "a priori mechanistic labels, not universal best or worst cases",
        "timing_order": "deterministic per-parent shuffle; consecutive repeats within a method",
        "resolution_design": "fresh independent heldout 64-square parents, not paired discretizations of the same field",
        "historical_parents": "old observed parents are excluded; any future optional historical evaluation is regression evidence only",
        "mean_law": "m'=r*(m-m^2-Var[u]); no mean conservation assumption",
        "hypotheses": ["commutator gate prevents learned drift in exact splitting limits",
                       "separate bounded mean/shape correction improves error decomposition",
                       "premix versus FNO benefit is tested within each constraint treatment"],
        "scope": "fresh bounded periodic logistic reaction-diffusion ablation; no FNO-paper reproduction or superiority claim"}


def validate_protocol(protocol: dict) -> dict:
    if not isinstance(protocol, dict) or protocol.get("profile") not in ("smoke", "full"):
        raise ValueError("Malformed consistency protocol")
    if digest(protocol) != digest(build_protocol(protocol["profile"])):
        raise ValueError("Consistency protocol differs from the immutable declared plan")
    return protocol


def initial_state(parent: dict):
    """Deterministic fresh FP64 field; torch is imported only when constructing data."""
    import math
    import torch
    from tdn.numerics.invariants import validate_state
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
    if parent["split"] in ("train", "validation"):
        fluctuation = value - value.mean()
        scale = fluctuation.abs().max()
        if not bool(scale > 0):
            raise ValueError("Development initializer unexpectedly produced a constant field")
        value = parent["mean"] + parent["amplitude"] * fluctuation / scale
    validate_state(value)
    return value
