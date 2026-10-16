"""Frozen retained-P1 regularization audit. No new labels or architecture."""
from __future__ import annotations

import copy
import json

from .scaling_protocol import VERIFY_TESTS as SCALING_VERIFY_TESTS

SCHEMA = "tdn.response_preconditioner.regularization.v1"
SELECTION_SCHEMA = "tdn.response_preconditioner.regularization.selection.v1"
CELLS = ((8, 8), (32, 2), (32, 8))
COEFFICIENT_RIDGES = (1e-6, 1e-10, 1e-12, 1e-14, 1e-16, 1e-18)
ACTION_ENERGIES = (1e-6, 1e-4, 1e-2, 1., 1e2)
CONTROLS = ("ordered_2", "ordered_8", "lawson_2", "lawson_4", "lawson_8")
VERIFY_TESTS = (*SCALING_VERIFY_TESTS, *(
    f"tests/test_response_preconditioner_regularization_{name}.py"
    for name in ("fitting", "evidence", "sensitivity", "analysis", "workflow")),
    *("tests/test_response_preconditioner_regularization_launch.py::test_" + name for name in (
        "shell_contract_and_non_native_rejection", "cli_study_rejects_standalone_and_gpu_before_science",
        "recipe_has_only_regularization_science_and_cpu_memory", "regularization_rejects_resource_expansion",
        "prepare_separate_frozen_protocol_and_modes", "prepared_input_identity_must_remain_exact",
        "post_study_identity_and_metrics_precede_optional_packing",
        "recursive_artifacts_preserve_binary_and_reject_symlinks",
        "stage_preserves_failed_outputs_and_rejects_promotion", "metrics_publish_named_finite_research_rows",
        "preflight_failure_never_starts_study_and_is_preserved", "export_terminal_snapshot_and_refusals",
        "full_inventory_uses_explicit_metadata_budget")))


def protocol(command="study"):
    if command not in ("verify", "study"):
        raise ValueError("Regularization supports verify and study only")
    return dict(
        schema=SCHEMA, phase="P1R", command=command, device="cpu",
        decision="BOUNDED_RETAINED_DATA_DEVELOPMENT", scientific_promotion=False,
        target="fd_same_grid", grid=[5, 5], model="unchanged five branches/eight features/40 FP64 weights",
        input_run="launch-0e144975eb4b433492fe55d75d209ef2-a1", input_job="2944",
        input_source="7d0c6201ad3ec399568492077066bd359c59829e",
        input_archive_sha256="caaacb862df0e272763340b69f523e66884c40e32013cd1ec6f0ff46eb89b6fb",
        input_index_sha256="43e44377ffb99f0ac4c056132b551e17736b61f35af26352b1650ec1692480b3",
        input_scope="exact reviewed P1 archive; completed numerical study despite original wrapper packing failure",
        data_status="previously inspected development; not fresh confirmation",
        cells=[dict(training_parent_count=n, calibration_source_count=k) for n, k in CELLS],
        training="original sorted parent and source prefixes; calibration only; parent-balanced physical MSE",
        coefficient_ridges=list(COEFFICIENT_RIDGES), action_energies=list(ACTION_ENERGIES),
        zero_candidate=True, projection_diagnostic=True,
        objectives=dict(
            coefficient_ridge="||sqrt(W)(A w-b)||^2 + lambda ||w||^2",
            action_energy="||sqrt(W)(A w-b)||^2 + alpha ||sqrt(W) A w||^2",
            action_energy_factor="training-only SVD L=diag(sigma) V^T; L^T L=A^T W A, without forming Gram normal equations",
            prior="w0=0 in both families: shrink the correction toward the existing ordered-two backbone",
            interpretation="action energy is uniform shrinkage of the full-rank unregularized action, not a derived BCH/Magnus or PDE law"),
        normalization="positive equal scalar cell masses; normalize to total one and equal mass per training parent",
        projection="unregularized training SVD with eps64*max(shape)*sigma_max cutoff; offline diagnostic, never selected",
        selection="independently per cell/family: smallest validation physical MSE among zero and numerically eligible positive penalties; zero first then listed strengths break exact ties",
        numerical_eligibility=dict(optimality_rtol=1e-8, absolute_objective_slack=1e-28,
            rule="finite full-rank augmented solution, original-coordinate objective gap and baseline nonregression; finite validation prediction and affine/five-branch parity",
            parity_factor=128., original_design_full_rank_required_for="action_energy only"),
        sensitivity=dict(scope="training estimated-reference label ball mapped to validation predictions only",
            method="thin SVD operator gain and actual retained refined-minus-independent U label perturbation",
            selection_gate=False, certificate=False,
            rank_rule="coefficient ridge uses all singular values; projection uses declared diagnostic TSVD cutoff"),
        freeze="seal all six family/cell selections before this audit reads any diagnostic or training-withheld numeric arrays",
        evaluation="all 1152 original withheld queries; six selected, three original, three projection models and five original classical controls",
        original_baselines="saved P1 coefficients and classical predictions; old timings are historical only, not newly measured",
        expected_fit_cells=3, expected_candidates=39, expected_positive_fits=33,
        expected_projection_fits=3, expected_selected_models=6,
        expected_input_queries=2688, expected_withheld_queries=1152,
        expected_methods=17, expected_metrics=19584,
        fresh_reference_solves=0, new_timing_bundles=0,
        aggregation="equal parent weight, exact per-action mean/spatial MSE decomposition; failed/missing slots retained",
        budget=dict(study_seconds=600., native_cpus=4, native_memory_gib=8,
                    native_wall_minutes=20, native_only=True, automatic_expansion=False),
        historical_gate="B STOP unchanged", matched_accuracy_speedup_established=False,
        deferred=["new labels", "larger cohorts", "derived analytic prior", "full-operator reference audit",
                  "new branches", "encoder", "rollout", "GPU", "fresh confirmation"],
        verify_tests=list(VERIFY_TESTS))


def validate_config(config, *, allow_fixture=False):
    canonical = protocol(config.get("command"))
    if config.get("_implementation_fixture") is True and allow_fixture:
        # Synthetic retained-array fixtures cannot change objectives or eligibility.
        keys = {"_implementation_fixture", "cells", "expected_fit_cells", "expected_candidates",
                "expected_positive_fits", "expected_projection_fits", "expected_selected_models",
                "expected_input_queries", "expected_withheld_queries", "expected_methods", "expected_metrics"}
        if set(config) != set(canonical) | {"_implementation_fixture"}:
            raise ValueError("Unexpected private fixture fields")
        if any(config[k] != canonical[k] for k in set(canonical)-keys):
            raise ValueError("Private fixture changed frozen scientific semantics")
        if not config["cells"] or len(config["cells"]) > len(CELLS):
            raise ValueError("Fixture cells exceed production scope")
        for cell in config["cells"]:
            if set(cell) != {"training_parent_count", "calibration_source_count"} or any(
                    type(x) is not int or x <= 0 for x in cell.values()):
                raise ValueError("Invalid fixture cell")
            if cell["training_parent_count"] > 32 or cell["calibration_source_count"] > 8:
                raise ValueError("Fixture cannot expand the bank")
        if any(type(config[k]) is not int or not 0 < config[k] <= canonical[k]
               for k in keys-{"_implementation_fixture", "cells"}):
            raise ValueError("Fixture denominators must be bounded positive integers")
    elif json.dumps(config, sort_keys=True, allow_nan=False) != json.dumps(canonical, sort_keys=True, allow_nan=False):
        raise ValueError("Changed frozen regularization protocol; no automatic expansion")
    count = len(config["cells"])
    expected = dict(expected_fit_cells=count, expected_candidates=13*count,
                    expected_positive_fits=11*count, expected_projection_fits=count,
                    expected_selected_models=2*count, expected_methods=4*count+len(CONTROLS))
    expected["expected_metrics"] = config["expected_withheld_queries"] * expected["expected_methods"]
    if any(config[k] != v for k, v in expected.items()):
        raise ValueError("Regularization work denominators disagree")
    return expected


def _fixture_config():
    """Synthetic arrays only; never a subset that fits the retained P1 labels."""
    config = copy.deepcopy(protocol())
    config.update(_implementation_fixture=True,
        cells=[dict(training_parent_count=2, calibration_source_count=2)],
        expected_fit_cells=1, expected_candidates=13, expected_positive_fits=11,
        expected_projection_fits=1, expected_selected_models=2,
        expected_input_queries=32, expected_withheld_queries=16,
        expected_methods=9, expected_metrics=144)
    validate_config(config, allow_fixture=True)
    return config
