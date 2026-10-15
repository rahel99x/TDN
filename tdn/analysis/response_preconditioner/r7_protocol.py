"""Prospective bounded R7 development contract; freeze before acquiring labels."""
from __future__ import annotations

from .workflow import VERIFY_TESTS as IMPLEMENTATION_TESTS

SCHEMA = "tdn.response_preconditioner.r7.v1"
VERIFY_TESTS = (*IMPLEMENTATION_TESTS, *(
    f"tests/test_response_preconditioner_r7_{name}.py"
    for name in ("actions", "references", "fitting", "workflow")), *(
    "tests/test_response_preconditioner_r7_launch.py::"+name for name in (
        "test_shell_contract_and_non_native_rejection",
        "test_cli_study_rejects_standalone_and_gpu_before_science",
        "test_recipe_has_only_r7_science_and_cpu_memory",
        "test_r7_rejects_resource_expansion",
        "test_prepare_separate_frozen_protocol_and_modes",
        "test_recursive_artifacts_preserve_binary_and_reject_symlinks",
        "test_stage_preserves_failed_outputs_and_rejects_promotion",
        "test_metrics_keep_task_families_and_unresolved_rows",
        "test_preflight_failure_never_starts_study_and_is_preserved")))


def protocol(command="study"):
    if command not in ("verify", "study"):
        raise ValueError("R7 supports verify and study only")
    return dict(
        schema=SCHEMA, command=command, device="cpu", target="fd_same_grid",
        decision="BOUNDED_DEVELOPMENT", scientific_promotion=False,
        cohort="four previously inspected development parents; not fresh confirmation",
        parent_seeds={"train": [2026101501, 2026101502],
                      "validation": [2026101503], "diagnostic": [2026101504]},
        parent_generator="implementation.v2 parents with source_directions_per_parent=2; preserve physics RNG draws",
        source_directions_per_parent=2, grid=[5, 5], lengths=[1., 1.],
        intervals=[[0., .04], [0., .08], [.02, .04], [.02, .06], [.04, .06], [.04, .08]],
        anchor="original smooth df AnchorPath(u0), evaluated at absolute time; never restarted at s",
        calibration_sources=4, withheld_sources=4,
        source_seed_rule="SeedSequence([parent_seed, context_index, 701]) for calibration; 702 for withheld",
        calibration_families=["dense_random"]*4,
        withheld_families=["dense_random", "low_frequency", "high_frequency", "constant"],
        source_definition="independent normal dense; cos(2pi*(x+y)+phase), cos(4pi*(x+y)+phase), ones; each unit physical RMS",
        source_family_selection="fixed before all labels; directions are paired descendants, not independent parents",
        model_precision="FP64 from actual FP32 initial states", mass="uniform cell mass / domain volume; RMS^2 = mean squares",
        ridge_menu=[1e-10, 1e-6], zero_candidate=True,
        training="action-only equal parent mass, equal context/source/cell mass within parent; train calibration only",
        selection="minimum physical action MSE on validation withheld directions; initialization first breaks exact ties",
        local_oracles="five coefficients shared across calibration sources per context; SVD projection and both positive ridges; no local ridge selection",
        local_context_weight=1/12,
        local_penalty="lambda_c=lambda/(pi_q*||eta||^2); pi_q=1/12 training-equivalent context weight, independent fits relax W coupling",
        projection_rank="eps64 * max(rows, columns) * largest singular value; discarded singular values reported, sampled projection only",
        reference=dict(tolerance=1e-10, initial_steps=4, max_steps=256,
                       max_seconds_per_query=3., independent_tolerance=1e-12,
                       independent_max_rhs_calls=4096),
        reference_count=192, max_reference_queries=192,
        reference_query_definition="one source/context query contains refined U and G plus independent DOP853 U and G",
        reference_caps="per query: at most 2*(4+8+16+32+64+128+256)=1016 Lawson steps and 8192 DOP853 RHS calls, shared 3s deadline",
        reference_resolution_multiple=10.,
        sensitivity=dict(seed=2026101507, random_trials=3,
                         uncertainty="aggregate weighted L2 ball from per-source estimated maximum uncertainty repeated over cells; empirical scenario, not a certificate",
                         perturbations=["leading_gain_direction", "three_fixed_seed_random", "independent_minus_refined"],
                         fit_scope="calibration or training labels only; withheld truth never used in sensitivity maps"),
        ordered_panels=[1, 2, 4, 8], lawson_steps=[2, 4, 8],
        timing=dict(rounds=5, warmup=1, seed=2026101508, threads=1,
                    scope="diagnostic contexts only; four withheld sources per complete bundle; fresh path/setup for every source; no cross-source cache",
                    cold="first measured bundle after accuracy acquisition; process already warm, not process startup",
                    order="seeded randomized paired order each round; raw observations and min/median/max, no confidence claim",
                    memory="actual native worker process-tree sampled RSS; not method-specific peak memory",
                    max_complete_source_calls=1344),
        budget=dict(study_seconds=900., native_cpus=4, native_memory_gib=8, native_wall_minutes=20,
                    native_only=True, automatic_expansion=False),
        endpoint_scope="no new endpoint/trajectory labels or claims; native 2936 endpoint benefit remains unresolved",
        deferred=["full operator audit", "BCH coefficients pending independent derivation", "ETDRK4/Radau/BDF",
                  "learned encoder", "nonlinear correction", "rollout", "GPU", "fresh confirmation"],
        utility_gate="NOT_DEFINED: report work-precision curves, no superiority or matched-tolerance speedup claim",
        verify_tests=list(VERIFY_TESTS), historical_gate="B STOP unchanged")
