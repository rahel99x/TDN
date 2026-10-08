"""Frozen five-gate research questions, cohorts and desktop resource ceilings."""
from __future__ import annotations

import math
import random
import copy

from tdn.research.protocol import digest

SCHEMA = "tdn.frontier/v1"
STAGES = ("audit", "screen", "prepare", "train", "confirm_prepare", "confirm", "scaling", "policy", "report")
GPU_STAGES = ("train", "confirm", "scaling", "policy")
MODEL_IDS = ("rank1", "rank1_postcompression", "rank1_frozen", "analytic_quad", "analytic_quad_cubic",
             "fno_small", "fno_standard", "direct_fno", "df", "etdrk4")
TRAINABLE = ("rank1", "rank1_postcompression", "fno_small", "fno_standard", "direct_fno")
TRACKS = ("discrete", "continuum")
# Kept synchronized with the exact native suite; the allocated launcher rejects
# skips, missing parameterizations and duplicate cases.
GPU_TEST_CASES = tuple(f"test_frontier_cuda_model_limits[{track}-{family}]" for track in TRACKS for family in MODEL_IDS) + tuple(
    f"test_frontier_cuda_model_gradient[{track}-{family}]" for track in TRACKS for family in TRAINABLE) + tuple(
    f"test_frontier_cuda_numerical_parity[{track}-{kind}]" for track in TRACKS for kind in ("heat", "product", "quadratic", "etdrk4")) + (
    "test_frontier_cuda_allocation_visible",) + tuple(
    f"test_frontier_cuda_policy_proposal[{track}]" for track in TRACKS) + tuple(
    f"test_frontier_cuda_policy_fallback[{track}]" for track in TRACKS) + tuple(
    f"test_frontier_cuda_scaling_distinct_fields[{track}]" for track in TRACKS)
GATES = {
    "G1": dict(name="Valid comparisons and independently refined references", stages=["audit"],
               question="Do comparisons represent identical requested outputs and real measured work?"),
    "G2": dict(name="Spatial consistency and classical headroom", stages=["screen", "prepare"],
               question="Where is spatial error resolved and where does classical refinement require work?"),
    "G3": dict(name="Compact learned interaction versus honest neural controls", stages=["train", "confirm_prepare", "confirm"],
               question="Does learning improve the physical core at competitive error, data and compute cost?"),
    "G4": dict(name="Accuracy-checked workload scaling", stages=["scaling"],
               question="Does a larger-workload throughput crossover survive independent accuracy checks?"),
    "G5": dict(name="Conditional deployment with full cost", stages=["policy"],
               question="Does available solver cost margin pay for estimation, rejected work and fallback?"),
}


def _parents(profile, n):
    offsets = {"smoke": 3100000, "development": 3300000, "full": 3500000}
    full, smoke = profile == "full", profile == "smoke"
    counts = dict(development=12 if full else 2 if smoke else 6,
                  train=32 if full else 2 if smoke else 8,
                  validation=16 if full else 2 if smoke else 4,
                  calibration=16 if full else 2 if smoke else 4,
                  confirmation=24 if full else 2 if smoke else 8,
                  scaling=4 if full else 2, policy=12 if full else 2 if smoke else 4)
    styles = ("low", "mixed", "high_pair", "rough", "near_nyquist", "boundary_mean", "reaction_stiff", "diffusion_stiff")
    high = max(2, n // 2 - 2)
    modes = {"low": [(1, 0), (0, 1)], "mixed": [(1, 0), (2, 1), (1, -2)],
             "high_pair": [(high - 1, 1), (high, 1)], "rough": [(1, 0), (2, 1), (high, -1)],
             "near_nyquist": [(high, 0), (0, high)], "boundary_mean": [(1, 0), (0, 2)],
             "reaction_stiff": [(1, 0), (2, 1)], "diffusion_stiff": [(1, 0), (high, 1)]}
    rows = []
    for si, (split, count) in enumerate(counts.items()):
        for index in range(count):
            seed = offsets[profile] + si * 10000 + index
            rng = random.Random(seed)
            style = styles[index % len(styles)]
            kappa = .04 if style == "diffusion_stiff" else (.001, .004, .012)[(index // 8) % 3]
            rate = 10. if style == "reaction_stiff" else (1., 3., 6.)[(index // 8) % 3]
            # A shift label must correspond to genuinely unseen coefficients,
            # not merely a fresh phase with training physics repeated.
            if full and split == "confirmation" and index >= 16:
                kappa *= 1.5
                rate *= 1.5
            rows.append(dict(parent_id=f"{split}-{seed}", seed=seed, field_seed=seed, split=split,
                field_cluster=f"field-{seed}", continuous_field_id=f"field-{seed}",
                field_terms=[dict(mode=list(mode), phase=rng.uniform(0, 2 * math.pi), weight=1 / math.sqrt(j + 1))
                             for j, mode in enumerate(modes[style])],
                grid=[n, n], mean=.12 if style == "boundary_mean" else (.35, .5, .65)[index % 3],
                variance=.0016 if style == "boundary_mean" else .004,
                spectrum=style, regime=style, category="favorable" if style == "low" else "typical" if style == "mixed" else "stress",
                distribution="coefficient_shift" if split == "confirmation" and index >= 16 else "declared_distribution",
                kappa=kappa, reaction_rate=rate, lengths=[1., 1.]))
    return rows


def build_protocol(profile="smoke"):
    if profile not in ("smoke", "development", "full"):
        raise ValueError("Frontier profile must be smoke, development or full")
    full, smoke = profile == "full", profile == "smoke"
    n = 32 if full else 8 if smoke else 16
    schedules = [[.02, .04, .06], [.06, .04, .02], [.03] * 4, [.04, .08, .12], [.12, .08, .04]]
    if smoke:
        schedules = [[.02, .04], [.04, .02], [.04, .08]]
    limits = {"audit": (300, 4, 32, 20), "screen": (1800, 8, 48, 45),
              "prepare": (1800, 8, 48, 45), "train": (1800, 4, 32, 45),
              "confirm_prepare": (1800, 8, 48, 45), "confirm": (1800, 4, 48, 45),
              "scaling": (1200, 4, 48, 30), "policy": (1200, 4, 32, 30), "report": (600, 4, 32, 20)}
    budgets = {s: dict(seconds=seconds if full else min(seconds, 240 if smoke else 900), cpus=cpus,
                       mem_gib=mem, walltime=f"00:{minutes:02}:00")
               for s, (seconds, cpus, mem, minutes) in limits.items()}
    return dict(schema=SCHEMA, version=1, profile=profile, stages=list(STAGES), gpu_stages=list(GPU_STAGES),
        gpu_test_cases=list(GPU_TEST_CASES), mechanisms=copy.deepcopy(GATES), combinations={}, budgets=budgets,
        parents=_parents(profile, n), tracks=list(TRACKS), models=list(MODEL_IDS), trainable_models=list(TRAINABLE),
        train_grid=n, grids=[n, 2 * n], seeds=[3600011, 3600021, 3600031] if full else [3600011],
        train_horizons=[.02, .04, .08, .12] if not smoke else [.02, .04, .06],
        validation_horizons=[.03, .06, .10] if not smoke else [.03, .06],
        confirm_schedules=schedules, evaluation_horizons=sorted({round(sum(s), 12) for s in schedules}),
        classical_frontier_steps=[1, 2, 4, 8],
        reference_tolerance=2e-8 if full else 2e-6, reference_substeps=8,
        reference_max_substeps=16384, reference_spatial_factors=[2, 4], reference_roundoff_floor=1e-12,
        reference_attempts=6, max_finest_substeps=16384,
        model_config=dict(width=16 if full else 4 if smoke else 8, modes=4 if full else 1 if smoke else 2,
                          depth=2, t_ref=.2, reaction_substeps=4),
        model_configs=dict(fno_standard=dict(width=32 if full else 6 if smoke else 16, depth=4,
                                            modes=8 if full else 2 if smoke else 4),
                           direct_fno=dict(width=32 if full else 6 if smoke else 16, depth=4,
                                           modes=8 if full else 2 if smoke else 4)),
        training=dict(updates=200 if full else 2 if smoke else 30, tuning_updates=40 if full else 1 if smoke else 8,
            learning_rates=[.001, .0003] if not smoke else [.001], batch_size=4 if full else 1,
            subset_sizes=[8, 32] if full else [2] if smoke else [4, 8], validation_every=50 if full else 1 if smoke else 10,
            clip_grad_norm=1., weight_decay=0., loss_floor=1e-6, peak_weight=.1,
            baseline_min_relative_improvement=.05, initialization_control=True,
            candidate_selection="validation only; lowest joint normalized RMS/maximum error"),
        targets=[2e-4, 2e-5, 2e-6], primary_target=2e-5, timing_repeats=5 if full else 2 if smoke else 3,
        timing_warmup=2 if full else 1, timing_seed=3700011, bootstrap_replicates=1000 if full else 100,
        practical_speedup=1.2, confidence_level=.95,
        screening=dict(grids=[n, 2 * n, 4 * n], horizons=[.08, .24] if full else [.04],
                       steps=[1, 2, 4, 8] if full else [1, 2], tracks=list(TRACKS), target=2e-5),
        scaling=dict(grids=[64, 128, 256] if full else [8, 16] if smoke else [32, 64],
                     batches=[1, 4] if full else [1, 2], horizon=.06, steps=2, repeats=5 if full else 2,
                     families=["rank1", "fno_small", "fno_standard", "df", "etdrk4"],
                     reference_required_for_speed_claim=True, maximum_cases=6 if full else 4),
        policies=dict(alpha=.01, empirical_safety_factor=2., target=2e-5,
                      model_family="rank1", horizon=round(sum(schedules[0]), 12), grid=n,
                      max_refinements=5 if full else 3, estimated_cost_margin_required=True,
                      selection="fixed family; seed and largest training subset fixed before confirmation",
                      modes=["temporal", "temporal_spatial"], calibration_split="calibration"),
        scoring=dict(mathematical_proof_claim=False, score_meaning="required-check evidence attainment, not model quality"),
        console_every=100, state_precision="float32", teacher_precision="float64", tf32=False,
        independent_unit="continuous_field_cluster", confirmation_after_checkpoint_freeze=True,
        frontier="reference-informed fixed-time post-hoc schedule selection; not a deployable policy",
        claims=dict(neural="learned contribution and matched accuracy/cost or data efficiency against competitive local FNO",
                    solver="complete measured work versus optimized target-consistent classical controls",
                    published_fno_reproduction=False, universal_worst_case=False),
        scientific_failure_blocks_all_research=False, automatic_budget_expansion=False)


def validate_protocol(protocol):
    if not isinstance(protocol, dict) or protocol.get("schema") != SCHEMA:
        raise ValueError("Invalid frontier protocol")
    if digest(protocol) != digest(build_protocol(protocol.get("profile"))):
        raise ValueError("Frontier protocol differs from its frozen declaration")
    if not protocol["gpu_test_cases"] or len(protocol["gpu_test_cases"]) != len(set(protocol["gpu_test_cases"])):
        raise ValueError("The complete native GPU suite must be nonempty and unique")
    clusters, identities = {}, set()
    for row in protocol["parents"]:
        if row["parent_id"] in identities:
            raise ValueError("Duplicate parent identity")
        identities.add(row["parent_id"])
        if clusters.setdefault(row["field_cluster"], row["split"]) != row["split"]:
            raise ValueError("Independent field crosses dataset splits")
    for stage, budget in protocol["budgets"].items():
        if budget["cpus"] > (4 if stage in GPU_STAGES else 8) or budget["mem_gib"] > 48 or budget["mem_gib"] * 1024 > 110000:
            raise ValueError("Frontier allocation exceeds the desktop resource envelope")
    return protocol
