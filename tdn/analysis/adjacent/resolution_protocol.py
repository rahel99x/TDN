"""Frozen 64²/128² follow-up, independent of historical adjacent declarations.

Bank teachers solve the specified same-grid equation. Selected spatial
refinement is a separate diagnostic; it never silently changes training truth.
"""
from __future__ import annotations

import math

PROFILES = ("resolution-smoke", "resolution-full", "resolution64", "resolution128")
NATIVE_FULL_PROFILES = frozenset(PROFILES[1:])
FITTED_GROUPS = {
    "ours": ["channel_global", "channel_affine", "channel_neural", "quad2_conditioned", "band_gain"],
    "theirs": ["fno_small", "fno_standard"],
}
FROZEN_FAMILIES = ["quad2_fixed", "quad4_fixed", "quad2_full", "quad4_full",
                   "channel_fixed", "analytic_quad_cubic", "df", "etdrk4"]


def build_resolution_protocol(profile="resolution-smoke"):
    from .protocol import SCHEMA, HYPOTHESES, _unit, gpu_test_cases
    if profile not in PROFILES:
        raise ValueError("Unknown adjacent resolution profile")
    smoke = profile == "resolution-smoke"
    grids = [64] if profile == "resolution64" else [128] if profile == "resolution128" else [64, 128]
    tracks = ["discrete", "continuum"]
    splits = {"train": 1, "validation": 1, "evaluation": 2} if smoke else {"train": 16, "validation": 8, "evaluation": 12}
    shard_size = 1 if smoke else 4
    seeds = [871001] if smoke else [871001, 871011]
    family_groups = {key: list(value) for key, value in FITTED_GROUPS.items()}
    resolution = dict(
        grids=grids, tracks=tracks, splits=splits, parent_shard_size=shard_size,
        seed=86000000 if smoke else 87000000, seeds=seeds,
        paired_grids=len(grids)==2, parent_identity_excludes_grid=True,
        cross_grid_transfer=len(grids)==2, transfer_direction="64-to-128 only" if len(grids)==2 else None,
        target_by_track={"discrete": "same-grid periodic FD diffusion and nodal logistic reaction",
                         "continuum": "same-grid dealiased Fourier Galerkin; not continuum truth"},
        horizons=[.04] if smoke else [.04, .12], training_horizons=[.04] if smoke else [.04, .12],
        evaluation_horizons=[.04] if smoke else [.04, .12],
        endpoint_steps=[1, 2] if smoke else [1, 2, 4, 8],
        additional_classical_steps=[] if smoke else [16, 32],
        kappa=.004, reaction_rate=3., mean=.43, rms=.05,
        physical_cutoff=8, split_cutoff=4, cutoff_semantics="fixed physical integer mode cutoff across paired grids",
        regimes=["low", "high_pair", "mixed", "localized", "phase_cancellation", "roughness", "nearly_constant"],
        alpha_levels=[.3, .7, 1.] if smoke else [i/10 for i in range(1, 11)],
        changed_bandwidth_regime="resolution_relative_highpair; not fixed-field refinement",
        families=[family for group in family_groups.values() for family in group],
        family_groups=family_groups, frozen_families=list(FROZEN_FAMILIES),
        model_config=dict(modes=8, split_modes=4, quad_nodes=2, cubic_nodes=4, hidden=8, reaction_substeps=4),
        model_configs={"analytic_quad_cubic": dict(quad_nodes=4, cubic_nodes=4)},
        training=dict(updates=2 if smoke else 128, seeds=seeds,
            learning_rates=[.001] if smoke else [.001, .0003], rates=[.001] if smoke else [.001, .0003],
            trial_seconds=15 if smoke else 20, validation_every=1 if smoke else 8,
            peak_weight=.1, fit_iterations=4 if smoke else 60,
            optimizer_controls=[dict(name="clipped", loss_scale=1., clip_grad_norm=1.)] if smoke else [
                dict(name="clipped", loss_scale=1., clip_grad_norm=1.),
                dict(name="scaled_unclipped", loss_scale=.01, clip_grad_norm=None)],
            fairness="paired data and declared trial menus; actual measured training compute is reported, not asserted equal"),
        reference=dict(tolerance=1e-10, substeps=8, max_substeps=256, precision="float64",
            roundoff_floor=1e-12, target="same_grid", spatial_case_limit=1 if smoke else 2,
            spatial_indices=[0] if smoke else [0, 5], spatial_factors=[2, 4], max_spatial_grid=512,
            uncertainty="actual time refinement; selected spatial refinement separately reported, neither is a certificate"),
        timing=dict(repeats=2 if smoke else 7, warmup=1, random_seed=873091,
                    batches=[1] if smoke else [1, 4], precision="float32", tf32=False),
        rms_target=2e-5, max_target=2e-5,
        primary_eligibility="joint RMS and maximum upper errors at the declared targets",
        tolerances=[2e-5] if smoke else [1e-4, 2e-5, 1e-6],
        tolerance_sweep_scope="separate descriptive accuracy-cost frontiers; not additional primary claims",
        diagnostics=dict(programs=["D01", "D02", "D03", "D04", "D05", "D06", "D07", "D08"],
            selected_parent_indices=[0] if smoke else [0, 1, 5],
            autonomous_horizons=[.04, .08] if smoke else [.04, .12, .24, .48],
            physical_and_resolution_relative_cutoff_slices=True,
            no_unobserved_case_imputation=True),
        exploration=dict(parent_limit=1 if smoke else 4, query_counts=[1, 4, 8], repeats=2,
            temporal_degree=6, selector_budget_fraction=.1, selector_rms_ratio_limit=1.05,
            required_speedup=1.10, relative_error_floor_multiple=5., temporal_rtol=1e-7,
            temporal_atol=1e-9, maximum_wall_fraction=.90,
            prototypes={"HR-E01": "selective cubic computation", "HR-E02": "amortized temporal follow-up"},
            scope="shared development parents; no held-out superiority claim"),
        source_scope="new resolution study; historical adjacent/portfolio outcomes and budgets unchanged")
    p = dict(schema=SCHEMA, version=1, profile=profile, study="adjacent-resolution",
        purpose="Controlled paired 64-square and 128-square complete-solver study",
        original_objective="Finite-time physical splitting correction at matched accuracy and complete cost",
        main_campaign_unchanged=True, automatic_budget_expansion=False,
        tracks=tracks, seed=resolution["seed"], resolution=resolution,
        mechanisms={key: dict(name=value, question=value) for key, value in HYPOTHESES.items()}, combinations={},
        timing_repeats=resolution["timing"]["repeats"],
        confirmation=dict(independent_unit="parent field; grids, schedules, horizons and seeds remain paired",
            maximum_parents=splits["evaluation"], confidence_level=.95, primary_comparisons=3*len(tracks)*len(grids),
            desired_log_halfwidth=math.log(1.15), primary_effect_ratio=1.10, practical_speedup=1.20,
            coverage_regression=.05, primary_claims=["channel_neural vs channel_global", "channel_neural vs quad2_conditioned", "channel_neural vs band_gain"],
            multiple_comparisons="Bonferroni across three contrasts, separate equation tracks and resolution claims; repeated fields remain paired",
            planning="fixed bounded 12-parent evaluation; precision and missing-cell gates can leave all claims NA",
            insufficient_precision="NA; never silently expand", inspected_confirmation_becomes_development=True),
        deployment=dict(automatic=False, reason="Require complete-cost margin and independently adequate evidence"),
        scientific_scope="All resolution profiles are bounded development studies, including full/64/128. Twelve evaluation parents and 128-update ceilings do not authorize confirmatory superiority; fresh sealed fields support only explicitly bounded development evidence.")
    p["mechanisms"].update({key: dict(name=value, question=value) for key, value in
        resolution["exploration"]["prototypes"].items()})
    units = {}

    def add(name, kind, prior, seconds, device="cpu", **extra):
        units[name] = _unit(name, kind, prior, seconds, device=device, **extra)

    add("audit", "audit", [], 180)
    bank_units = {}

    def banks(split, prerequisites):
        for grid in grids:
            for track in tracks:
                ids = []
                shards = ([list(range(offset, splits[split], 3)) for offset in range(3)]
                          if split == "evaluation" and not smoke else
                          [list(range(first, min(first+shard_size, splits[split])))
                           for first in range(0, splits[split], shard_size)])
                for part, indices in enumerate(shards):
                    name = f"prepare-{split}-n{grid}-{track}-p{part:03d}"
                    add(name, "resolution_prepare", prerequisites, 300 if smoke else 900,
                        grid=grid, track=track, split=split, part=part,
                        field_indices=indices,
                        resumable=True)
                    ids.append(name)
                bank_units[(grid, track, split)] = ids

    banks("train", ["audit"])
    banks("validation", ["audit"])
    train_units = []
    for grid in grids:
        for track in tracks:
            train_bank = bank_units[(grid, track, "train")]
            data = train_bank + bank_units[(grid, track, "validation")]
            add(f"diagnostic-n{grid}-{track}", "resolution_diagnostic", ["audit", *train_bank],
                300 if smoke else 900, grid=grid, track=track, bank_units=train_bank,
                diagnostic_ids=resolution["diagnostics"]["programs"])
            for group, families in family_groups.items():
                name = f"train-n{grid}-{track}-{group}"
                add(name, "resolution_train", ["audit", *data], 120 if smoke else 900,
                    device="cuda", grid=grid, track=track, group=group, families=families,
                    seeds=seeds, bank_units=data, resumable=True)
                train_units.append(name)
            add(f"explore-n{grid}-{track}", "resolution_explore", ["audit", *train_bank],
                60 if smoke else 450, device="cuda", grid=grid, track=track,
                prototype_ids=["HR-E01", "HR-E02"], bank_units=train_bank)
    add("freeze", "resolution_freeze", train_units, 180 if smoke else 300,
        train_units=train_units)
    banks("evaluation", ["audit", "freeze"])
    evaluation_units = []
    for grid in grids:
        for track in tracks:
            for bank in bank_units[(grid, track, "evaluation")]:
                part = units[bank]["part"]
                name = f"evaluate-n{grid}-{track}-p{part:03d}"
                add(name, "resolution_evaluate", ["freeze", bank], 300 if smoke else 900,
                    device="cuda", grid=grid, track=track, part=part, split="evaluation",
                    field_indices=units[bank]["field_indices"], bank_units=[bank],
                    freeze_unit="freeze", resumable=True)
                evaluation_units.append(name)
    add("aggregate", "resolution_aggregate", ["freeze", *evaluation_units], 180 if smoke else 300,
        evaluation_units=evaluation_units)
    add("report", "report", list(units), 240 if smoke else 600)
    p["units"] = units
    p["stages"] = list(units)
    p["gpu_stages"] = [name for name, unit in units.items() if unit["device"] == "cuda"]
    p["budgets"] = {name: {key: unit[key] for key in ("seconds", "cpus", "mem_gib", "walltime")}
                    for name, unit in units.items()}
    p["gpu_test_cases"] = gpu_test_cases() + [
        f"{name}[{grid}-{track}]"
        for name in ("test_resolution_cuda_large_grid_contract", "test_resolution_cuda_fno_local_path",
                     "test_resolution_cuda_precision", "test_resolution_cuda_exploration")
        for grid in (64, 128) for track in tracks]
    architecture = sum(u["seconds"] for u in units.values() if u["kind"] == "resolution_train")
    exploration = sum(u["seconds"] for u in units.values() if u["kind"] == "resolution_explore")
    p["adjacent_budget"] = dict(discretionary_seconds=architecture+exploration,
        architecture_seconds=architecture, protected_exploration_seconds=exploration,
        exploration_fraction=.20, shared_work_separately_enumerated=True,
        transfer_from_main_campaign=0, monetary_cost=None, energy_joules=None,
        summed_science_ceiling_seconds=sum(u["seconds"] for u in units.values()),
        training_reference_generation="Separate CPU parent shards; not hidden inside GPU training budgets")
    return p


def model_config_for(protocol, family):
    """Frozen per-family parameters; GL4 quadratic+cubic is never silently GL2."""
    cfg = protocol["resolution"]
    result = dict(cfg["model_config"])
    result.update(cfg.get("model_configs", {}).get(family, {}))
    if family == "analytic_quad_cubic":
        result.update(quad_nodes=4, cubic_nodes=4)
    return result
