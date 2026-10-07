"""Frozen all-mechanism scope, resource ceilings and independent cohorts."""
from __future__ import annotations

import json
from pathlib import Path

from tdn.research.protocol import digest

SCHEMA = "tdn.roadmap/v1"
STAGES = ("audit", "headroom", "prepare", "train", "confirm_prepare", "confirm", "policy", "transfer", "scaling", "report")
GPU_STAGES = ("train", "confirm", "policy", "scaling")
MODEL_IDS = ("source", "rank1", "rank2", "source_postcompression", "source_trust", "source_consistency",
             "source_df_loss", "source_df_loss_shared", "source_multiband", "cheap_fno", "deep_fno", "direct_fno",
             "c1_rank0", "c1_rank1", "c1_rank2", "c2_rank2", "source_embedded")
GPU_TEST_CASES = tuple(f"test_roadmap_cuda_model_{kind}[{family}]" for kind in ("limits", "gradient") for family in MODEL_IDS) + (
    "test_roadmap_cuda_numerical_parity[quadratic]", "test_roadmap_cuda_numerical_parity[cubic]",
    "test_roadmap_cuda_numerical_parity[dealias]", "test_roadmap_cuda_numerical_parity[moments]",
    "test_roadmap_cuda_fractional_physical_scale", "test_roadmap_cuda_allocation_visible")
MECHANISM_STAGES = {
    "M00": ["headroom", "confirm"], "M01": ["headroom", "policy"], "M02": ["audit", "confirm"],
    "M03": ["train", "confirm"], "M04": ["audit", "train", "confirm"], "M05": ["audit", "headroom", "transfer"],
    "M06": ["train", "confirm", "policy"], "M07": ["policy"], "M08": ["policy"], "M09": ["policy"],
    "M10": ["train", "confirm", "policy"], "M11": ["train", "confirm"], "M12": ["audit"],
    "M13": ["train", "confirm", "transfer"], "M14": ["transfer"], "M15": ["scaling", "policy"],
    "M16": ["headroom", "train", "confirm"], "M17": ["prepare", "confirm_prepare", "transfer"],
    "M18": ["prepare", "confirm_prepare", "confirm", "policy"], "M19": ["transfer"], "M20": ["audit", "train", "confirm"],
    "M21": ["audit", "confirm", "transfer"], "M22": ["audit", "report"], "M23": ["audit"],
}
COMBINATION_STAGES = {"C0": ["headroom"], "C1": ["train", "confirm"], "C2": ["policy"], "C3": ["audit"], "C4": ["transfer"]}


def _registry():
    # The dated design is part of the immutable declaration. A changed design
    # changes the protocol hash and cannot be grafted onto an existing run.
    path = Path(__file__).resolve().parents[3] / "results" / "RESEARCH_ROADMAP_20261007.json"
    roadmap = json.loads(path.read_text())
    return ({m["id"]: {"name": m["name"], "gaps": m["shortcoming_ids"], "priority": m["priority"],
                      "stages": MECHANISM_STAGES[m["id"]], "required_work": m["required_work"]}
             for m in roadmap["mechanisms"]},
            {c["id"]: {**c, "stages": COMBINATION_STAGES[c["id"]]} for c in roadmap["combinations"]})


def _parents(profile, n):
    import math
    import random
    offset = {"smoke": 2100000, "development": 2300000, "full": 2500000}[profile]
    full, smoke = profile == "full", profile == "smoke"
    counts = {"train": 24 if full else 2 if smoke else 8,
              "validation": 12 if full else 2 if smoke else 4,
              "calibration": 24 if full else 2 if smoke else 6,
              "confirmation": 24 if full else 2 if smoke else 6}
    styles = ("low", "mixed", "high_pair", "rough", "near_nyquist", "boundary_mean")
    high = max(2, n // 2 - 2)
    modes = {"low": [(1, 0), (0, 1)], "mixed": [(1, 0), (2, 1), (1, -2)],
             "high_pair": [(high - 1, 1), (high, 1)],
             "rough": [(1, 0), (2, 1), (high, -1)],
             "near_nyquist": [(high, 0), (0, high)], "boundary_mean": [(1, 0), (0, 2)]}
    parents = []
    for split_index, (split, count) in enumerate(counts.items()):
        for index in range(count):
            seed = offset + 10000 * split_index + index
            rng = random.Random(seed)
            style = styles[index % len(styles)]
            terms = [{"mode": list(mode), "phase": rng.uniform(0, 2 * math.pi),
                      "weight": 1 / math.sqrt(j + 1)} for j, mode in enumerate(modes[style])]
            physics = [(.002, 2.)]
            if full and split == "confirmation":
                physics = [(.002, 2.), (.02, 8.)]
            elif full:
                physics = [((.001, .004, .015)[index % 3], (.5, 2., 6.)[(index // 3) % 3])]
            for pi, (kappa, reaction) in enumerate(physics):
                parents.append(dict(parent_id=f"{split}-{seed}-p{pi}", seed=seed, split=split,
                    continuous_field_id=f"field-{seed}", field_cluster=f"cluster-{seed}",
                    field_terms=terms, grid=[n, n], mean=.18 if style == "boundary_mean" else (.35, .5, .65)[index % 3],
                    variance=.0016 if style == "boundary_mean" else .004,
                    spectrum=style, regime=style, physics_index=pi, kappa=kappa, reaction_rate=reaction,
                    distribution="independent_frozen_confirmation" if split == "confirmation" else split))
    return parents


def build_protocol(profile="smoke"):
    if profile not in ("smoke", "development", "full"):
        raise ValueError("Roadmap profile must be smoke, development or full")
    full, smoke = profile == "full", profile == "smoke"
    n = 32 if full else 8 if smoke else 16
    mechanisms, combinations = _registry()
    parents = _parents(profile, n)
    limits = {"audit": (600, 8, 48, 20), "headroom": (1800, 8, 48, 45),
              "prepare": (1800, 8, 48, 45), "train": (1800, 4, 32, 45),
              "confirm_prepare": (1800, 8, 48, 45), "confirm": (1800, 4, 48, 45),
              "policy": (1200, 4, 32, 30), "transfer": (1800, 8, 48, 45),
              "scaling": (1200, 4, 32, 30), "report": (600, 4, 32, 20)}
    budgets = {stage: dict(seconds=seconds if full else min(seconds, 240 if smoke else 900),
        cpus=cpu, mem_gib=mem, walltime=f"00:{minutes:02}:00") for stage, (seconds, cpu, mem, minutes) in limits.items()}
    schedules = [[.03, .07, .17], [.17, .07, .03], [.025, .055, .075, .115], [.045] * 6, [.03, .07, .17] * 3]
    if smoke:
        schedules = [[.02, .04], [.04, .02]]
    # Select complete six-style blocks, not alternating seeds: seed parity is
    # correlated with the six-style cycle and would exclude rough, mixed and
    # boundary-mean fields from every full continuum diagnostic.
    confirmation_origin = min(p["seed"] for p in parents if p["split"] == "confirmation")
    continuum_ids = [p["parent_id"] for p in parents if p["split"] in ("calibration", "confirmation")
                     and (p["split"] == "calibration" or not full
                          or ((p["seed"] - confirmation_origin) // 6) % 2 == 0)]
    return dict(schema=SCHEMA, version=1, profile=profile, stages=list(STAGES), gpu_stages=list(GPU_STAGES),
        gpu_test_cases=list(GPU_TEST_CASES), mechanisms=mechanisms, combinations=combinations,
        budgets=budgets, parents=parents, train_grid=n, grids=[n, 2 * n],
        train_horizons=[.02, .04, .08, .12, .16, .20, .24] if not smoke else [.02, .04, .06],
        validation_horizons=[.03, .06, .09, .12, .18, .20] if not smoke else [.03, .06],
        confirm_schedules=schedules, seeds=[2600011, 2600021, 2600031] if full else [2600011],
        models=list(MODEL_IDS), width=12 if full else 4 if smoke else 8, modes=4 if full else 1 if smoke else 2,
        t_ref=.2, U_ref=1., reference_tolerance=2e-8 if full else 2e-6,
        reference_attempts=5, max_finest_substeps=131072,
        continuum_reference_grids=[4 * n, 8 * n], continuum_parent_ids=continuum_ids,
        targets=[2e-4, 2e-5, 2e-6], timing_repeats=3 if full else 1, timing_warmup=1,
        data=dict(teacher_workers=8 if full else 1, independent_unit="continuous_field_cluster",
                  preserve_same_fields_across_grids=True, confirmation_after_checkpoint_freeze=True),
        training=dict(updates=300 if full else 2 if smoke else 40,
            tuning_updates=120 if full else 1 if smoke else 10,
            learning_rates=[.0003, .001] if not smoke else [.001], batches=[1, 4] if not smoke else [1],
            weight_decay=0., validation_every=20 if full else 1 if smoke else 10,
            clip_grad_norm=1., base_harm_weight=.25, peak_weight=.2, centered_weight=.2,
            consistency_weight=.1, generator_weight=.05, intermediate_weight=.2,
            loss_floor=1e-6, orientation="diffusion_first", preserve_initialization=True),
        policies=dict(alpha=.01, model_families=["source", "rank2", "cheap_fno", "c2_rank2"], model_seed_limit=1,
            max_calibration_parents=24 if full else 2 if smoke else 6,
            max_confirmation_clusters=6 if full else 2 if smoke else 4,
            grids=[n], horizons=[round(sum(schedules[0]), 12)], tracks=["discrete", "continuum"] if full else ["continuum"],
            rms_target=2e-4, max_target=2e-4, max_attempts=3,
            attempted_step_sizes=[.09, .045, .0225] if not smoke else [.06, .03],
            fallback_max_refinements=5 if full else 3, max_refinements=5 if full else 3,
            estimators=["step_doubling", "interior_residual"], empirical_safety_factor=2.),
        headroom=dict(clusters=12 if full else 2 if smoke else 4, grids=[n, 2 * n, 4 * n],
            horizons=[.08, .24] if full else [.04], steps=[1, 2, 4, 8] if full else [1, 2], target=2e-4),
        scaling=dict(grids=[32, 64, 128, 256, 512, 1024] if full else [8, 16] if smoke else [16, 32, 64],
            batches=[1, 4, 16] if full else [1, 2], repeats=5 if full else 1, maximum_cases=18 if full else 4),
        portability=dict(grids=[16, 32] if full else [12, 24], clusters=6 if full else 2,
            horizon=.04, teacher_tolerance=2e-7, steps=4, timing_repeats=3 if full else 1),
        scoring=dict(weights={"correctness": 2, "math": 2, "gap": 3, "utility": 3},
            labels=["GOOD", "BAD", "NA"], formula="1+round(99*passed_required_weight/declared_required_weight)",
            na_score=1, na_meaning="no credited evidence, not a measured performance result",
            mathematical_proof_claim=False, minimum_mechanism_categories=["math", "gap"]),
        hardware=dict(root="/home/rahel/TDN", python="3.13.13", physical_cpu_cores=8,
            scheduler_cpus=16, scheduler_memory_mib=110000, physical_host_ram_gib=128,
            gpu="RTX4090", gpu_count=1, dedicated_vram_gib=24),
        gpu_memory=dict(soft_cap_gib=18, soft_fraction=.75, hard_device_used_fraction=.9),
        console_every=100, scope="all M00–M23 and feasible C0–C4, bounded experimental evidence; no paper reproduction or theorem proof",
        scientific_failure_blocks_other_mechanisms=False, automatic_budget_expansion=False, pending_job_cap=None)


def validate_protocol(protocol):
    if not isinstance(protocol, dict) or protocol.get("schema") != SCHEMA:
        raise ValueError("Invalid roadmap protocol")
    expected = build_protocol(protocol.get("profile"))
    if digest(expected) != digest(protocol):
        raise ValueError("Roadmap protocol differs from its frozen declaration")
    clusters = {}
    for parent in protocol["parents"]:
        prior = clusters.setdefault(parent["field_cluster"], parent["split"])
        if prior != parent["split"]:
            raise ValueError("Independent field cluster crosses dataset splits")
    if len({p["parent_id"] for p in protocol["parents"]}) != len(protocol["parents"]):
        raise ValueError("Duplicate parent identity")
    for stage, b in protocol["budgets"].items():
        if b["mem_gib"] > 48 or b["mem_gib"] * 1024 > 110000 or b["cpus"] > (4 if stage in GPU_STAGES else 8):
            raise ValueError("Declared resource budget exceeds desktop limits")
    return protocol
