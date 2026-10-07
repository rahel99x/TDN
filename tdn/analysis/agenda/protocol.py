"""Standard-library declarations frozen before the research agenda executes."""
from __future__ import annotations

from itertools import product

from tdn.research.protocol import digest
from .structural_spec import specification

SCHEMA = "tdn.research-agenda/v1"
STAGES = ("structure", "prepare", "controls", "optimize", "compression", "kernel", "confirm", "policy")
GPU_STAGES = STAGES[2:]
PROFILES = ("smoke", "development", "full")
QUESTION_STAGES = {
    "Q1": ["structure", "controls", "confirm"],
    "Q2": ["structure", "kernel", "confirm"],
    "Q3": ["structure", "kernel", "confirm"],
    "Q4": ["structure", "prepare", "compression", "confirm"],
    "Q5": ["structure", "policy"],
    "Q6": ["structure", "confirm", "policy"],
    "Q7": ["policy"],
}
FULL_BUDGETS = {
    "structure": (600, 8, 48, "00:20:00"),
    "prepare": (1800, 8, 48, "00:45:00"),
    "controls": (1200, 4, 32, "00:30:00"),
    "optimize": (1800, 4, 32, "00:45:00"),
    "compression": (1200, 4, 32, "00:30:00"),
    "kernel": (1200, 4, 32, "00:30:00"),
    "confirm": (1800, 4, 48, "00:45:00"),
    "policy": (1200, 4, 32, "00:30:00"),
}


def _parents(profile, grid):
    full, smoke = profile == "full", profile == "smoke"
    offset = {"full": 1100000, "development": 1300000, "smoke": 1500000}[profile]
    result = []
    counts = {"train": 48 if full else 4 if smoke else 12,
              "validation": 24 if full else 4 if smoke else 6,
              "calibration": 16 if full else 2 if smoke else 4}
    spectra = ("low", "mixed", "high_pair", "rough", "near_nyquist", "mixed")
    for split_index, (split, count) in enumerate(counts.items()):
        for index in range(count):
            seed = offset + 10000 * split_index + index
            result.append(dict(parent_id=f"{split}-{seed}", continuous_field_id=f"field-{seed}",
                seed=seed, split=split, grid=[grid, grid], mean=(.35, .5, .65)[index % 3],
                variance=(.0016, .004, .0081)[(index // 3) % 3],
                amplitude=(.35, .75)[(index // 2) % 2], spectrum=spectra[index % len(spectra)],
                phase=0., kappa=(.001, .006, .015)[(index // 2) % 3],
                reaction_rate=(.5, 2., 6.)[(index // 3) % 3],
                regime=spectra[index % len(spectra)], distribution="development" if split != "calibration" else "policy_calibration"))
    # Eight controlled fields: each shape/mean/variance change is explicit.
    # Each is crossed with diffusion and reaction; repeated fields are paired
    # cases, not independent samples. All grids sample the same Fourier field.
    field_profiles = [
        ("low", .5, .004, .35, 0., 1),
        ("rough", .5, .004, .35, 0., 1),
        ("high_pair", .5, .004, .35, 0., 1),
        ("near_nyquist", .5, .004, .35, 0., 1),
        ("rough", .5, .004, .75, 0., 1),
        ("rough", .5, .004, .35, .7, 1),
        ("rough", .5, .0121, .35, 0., 1),
        ("rough", .35, .004, .35, 0., 1),
    ]
    field_count = 8 if full else 2 if smoke else 4
    physics = list(product((.001, .03), (.5, 8.))) if full else [(.003, 2.), (.03, 8.)]
    for field_index, (spectrum, mean, variance, amplitude, phase, seed_group) in enumerate(field_profiles[:field_count]):
        seed = offset + 30000 + seed_group
        for physics_index, (kappa, rate) in enumerate(physics):
            identity = f"confirmation-{offset}-field-{field_index}-physics-{physics_index}"
            result.append(dict(parent_id=identity, continuous_field_id=f"fresh-field-{offset}-{field_index}",
                seed=seed, split="confirmation", grid=[grid, grid], mean=mean,
                variance=variance, amplitude=amplitude, spectrum=spectrum, phase=phase,
                kappa=kappa, reaction_rate=rate, regime=spectrum,
                distribution="fresh_controlled_factorial", field_cluster=seed_group,
                controlled_field_index=field_index, physics_index=physics_index))
    return result


def build_protocol(profile="smoke"):
    if profile not in PROFILES:
        raise ValueError("Agenda profile must be smoke, development or full")
    full, smoke = profile == "full", profile == "smoke"
    grid, width, modes = (32, 16, 4) if full else (8, 4, 1) if smoke else (16, 8, 2)
    seeds = [960011, 960021, 960031] if full else [960011]
    learning_rates = [.0001, .0003, .001] if not smoke else [.0003, .001]
    batches = [1, 4, 8] if not smoke else [1, 2]
    trial_groups = {stage: [] for stage in ("controls", "optimize", "compression", "kernel")}

    def trial(stage, family, seed, updates, learning_rate=.001, batch_size=1, orientation="reaction-first"):
        identifier = f"{stage}-{family}-{seed}-{orientation}-lr{learning_rate:g}-b{batch_size}"
        trial_groups[stage].append(dict(trial_id=identifier, family=family, seed=seed,
            updates=updates, learning_rate=learning_rate, batch_size=batch_size,
            width=width, modes=modes, base_orientation=orientation, orientation=orientation))

    controls = ("local_gate", "local_gate_time", "local_endpoint", "source", "source_time",
                "source_endpoint", "source_closure", "source_time_closure")
    for family in controls:
        trial("controls", family, seeds[0], 120 if full else 2 if smoke else 30)
    for family in ("local_gate", "source", "source_time"):
        trial("controls", family, seeds[0], 120 if full else 2 if smoke else 30, orientation="diffusion-first")
    for family, rate, batch in product(("local_gate", "source", "source_time", "source_closure"), learning_rates, batches):
        trial("optimize", family, seeds[0], 120 if full else 2 if smoke else 20, rate, batch)
    for family, seed in product(("source", "precompress_source", "fno_source", "fno_source_matched", "fno_source_depth4", "fno_anchor"), seeds):
        trial("compression", family, seed, 300 if full else 3 if smoke else 60)
    for family in ("pair_rank2", "pair_rank4", "pair_rank8", "pair_rank4_closure"):
        trial("kernel", family, seeds[0], 180 if full else 2 if smoke else 40)
    budgets = {stage: dict(seconds=values[0], cpus=values[1], mem_gib=values[2], walltime=values[3])
               for stage, values in FULL_BUDGETS.items()}
    if not full:
        for stage in STAGES:
            budgets[stage]["seconds"] = 120 if smoke else 600 if stage in ("prepare", "optimize", "confirm") else 300
    parents = _parents(profile, grid)
    return dict(schema=SCHEMA, version=1, profile=profile, stages=list(STAGES),
        question_stages=QUESTION_STAGES, structural=specification(profile), budgets=budgets,
        hardware=dict(user="rahel", root="/home/rahel/TDN", physical_cpu_cores=8,
            scheduler_cpus=16, physical_host_ram_gib=128, scheduler_memory_mib=110000,
            gpu_name="RTX 4090", dedicated_gpu_vram_gib=24, gpu_count=1),
        gpu_memory=dict(soft_cap_gib=18, soft_fraction=.75, hard_device_used_fraction=.9),
        parents=parents, grids=[32, 64, 128] if full else [8, 16] if smoke else [16, 32, 64],
        train_grid=grid, train_horizons=[.04, .08, .16, .24], validation_horizons=[.06, .12, .20],
        confirm_schedules=[[.03, .07, .17], [.17, .07, .03], [.025, .055, .075, .115], [.045] * 6, [.03, .07, .17] * 3],
        reference_tolerance=2e-8 if full else 2e-6, max_finest_substeps=131072,
        continuum_reference_grids=[256, 512] if full else [32, 64] if smoke else [128, 256],
        continuum_parent_ids=[p["parent_id"] for p in parents if p["split"] == "confirmation" and p["physics_index"] == 0],
        target_conventions=["same_grid_discrete", "continuum_dealiased"],
        product_convention="same-grid nodal product; continuum track separately uses padded dealiased spectral product",
        seeds=seeds, width=width, modes=modes, t_ref=.2, U_ref=1.,
        validation_every=20 if full else 1 if smoke else 10,
        loss_scale=.02, two_step_loss_weight=1., stage_trials=trial_groups,
        classical=["strang_reaction_first", "strang_diffusion_first", "etdrk4", "gl3_fused", "spectral_mean"],
        targets=[.002, .0002, .00002, .000002], norms=["rms", "max"],
        timing_repeats=3 if full else 1, timing_warmup=1,
        rank_relative_tolerance=.05,
        base_orientation_policy="validation_only_source_control_minimum",
        hyperparameter_policy="optimization_selected_source_for_matched_stages",
        maximum_confirmation_variants=12,
        confirmation_selection=[dict(stage="compression", family=family, seed=seed)
            for family, seed in product(("source", "precompress_source", "fno_source_matched"), seeds)]
            + [dict(stage="controls", family=family, seed=seeds[0]) for family in ("source_time", "source_closure")]
            + [dict(stage="kernel", family="best_eligible_pair", seed=seeds[0])],
        policy=dict(estimators=["step_doubling", "splitting_defect", "validated_envelope"],
            calibration_split="calibration", evaluation_split="confirmation",
            primary_target=.0002, require_both_norms=True, safety_factor=2.,
            fallback="etdrk4", max_attempts=8, minimum_step=.005,
            attempted_step_sizes=[.09, .045] if smoke else [.09, .045, .0225, .01125],
            false_acceptance_definition="accepted while independent error plus estimated reference uncertainty exceeds either target"),
        success_criteria=dict(primary_target=.0002, norms=["rms", "max"],
            minimum_overall_coverage=.9, minimum_each_regime_coverage=.75,
            maximum_spatial_regression_ratio=1.1, spatial_absolute_floor=.000001,
            minimum_median_speedup_over_classical=1.1,
            classical_coverage_must_be_retained=True, maximum_false_acceptance_rate=.01,
            operational_proposal=True, statistical_significance_claim=False),
        scope="bounded research agenda; no published FNO reproduction or novelty/superiority claim",
        fresh_cohort_policy="freeze checkpoint catalogs and selection before opening confirmation bank; CPU development profiles use disjoint seeds")


def validate_protocol(protocol):
    if not isinstance(protocol, dict) or protocol.get("profile") not in PROFILES:
        raise ValueError("Invalid agenda protocol")
    if protocol != build_protocol(protocol["profile"]):
        raise ValueError("Agenda protocol differs from its immutable declared plan")
    if any(value["mem_gib"] * 1024 > protocol["hardware"]["scheduler_memory_mib"]
           for value in protocol["budgets"].values()):
        raise ValueError("Stage host memory exceeds the Slurm cap")
    return protocol


__all__ = ["SCHEMA", "STAGES", "GPU_STAGES", "PROFILES", "QUESTION_STAGES", "build_protocol", "validate_protocol", "digest"]
