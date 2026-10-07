"""Stdlib-only immutable declaration of structural cases."""
from __future__ import annotations

import itertools

SCHEMA = "tdn.agenda-structural/v1"
QUESTION_IDS = ("gate_span", "temporal_capacity", "strong_diffusion_orientation",
                "moment_closure", "cubic_finite_step", "stiff_input_output_scales",
                "separable_phase_kernel", "aliasing_discretization",
                "defect_estimator_acceptance", "unequal_unseen_steps")
PATTERNS = ("uniform", "single_low", "single_high", "mixed_phase0", "mixed_phase1",
            "remote_bump", "remote_bumps", "nyquist_pair")


def specification(profile="smoke"):
    """Freeze deterministic cases/settings before any responses are inspected."""
    if profile not in ("smoke", "development", "full"):
        raise ValueError("Structural profile must be smoke, development or full")
    smoke = profile == "smoke"
    development = profile == "development"
    patterns = PATTERNS
    response_settings = [( .35, .12, .003, 2.)] if smoke else [
        (.25, .06, .003, 2.), (.65, .12, .003, 2.),
        (.25, .12, .02, 2.), (.65, .24, .003, 6.)]
    if development:
        response_settings = response_settings[::2]
    stiff = list(itertools.product((.25,) if smoke or development else (.25, .65),
                                  (.10,) if smoke else (.02, .15),
                                  (.001, .1) if smoke or development else (.001, .01, .1, 1.),
                                  (2.,) if smoke else (.5, 4.),
                                  (.12,) if smoke else (.03, .3)))
    rank_settings = [( .35, .12, .003, 2.)] if smoke else list(itertools.product(
        (.25, .65), (.03, .12, .3), (.001, .02), (.5, 6.)))
    if development:
        rank_settings = [(.35, .12, .003, 2.), (.35, .24, .02, 6.)]
    temporal = [("single_low", .003, 2.)] if smoke else list(itertools.product(
        ("single_low", "mixed_phase1", "nyquist_pair"), (.003, .02), (2., 6.)))
    ids = []
    for index, _ in enumerate(response_settings):
        for pattern in patterns:
            ids.extend((f"response/{index}/{pattern}/teacher", f"response/{index}/{pattern}/span"))
    ids.extend(f"strong_diffusion/{index}" for index in range(len(stiff)))
    for index in range(len(rank_settings)):
        ids.extend(f"kernel_rank/{index}/{rank}" for rank in (2, 4, 8))
    ids.extend(f"temporal_capacity/{index}" for index in range(len(temporal)))
    ids.extend(("moment/equal_moments_spectrum", "moment/equal_moments_phase",
                "alias/nodal_vs_dealiased", "alias/discrete_vs_continuum_eigenvalues",
                "cubic/finite_step_low", "cubic/finite_step_high",
                "stiff/pair_same_output", "stiff/pair_same_inputs", "estimator/false_acceptance",
                "variable_step/unequal_composition"))
    return dict(schema=SCHEMA, profile=profile, grid=16 if smoke or development else 24,
                patterns=list(patterns), response_settings=[list(x) for x in response_settings],
                strong_diffusion_settings=[list(x) for x in stiff],
                rank_settings=[list(x) for x in rank_settings],
                temporal_settings=[list(x) for x in temporal], ranks=[2, 4, 8],
                quadrature_nodes=64, uncertainty_nodes=128,
                symmetric_epsilon=.04, epsilon_levels=[.04, .02, .01], reference_tolerance=2e-6,
                rank_relative_tolerance=.05, span_uncertainty_multiplier=5.,
                temporal_train_steps=[.02, .04, .08, .16],
                temporal_unseen_steps=[.03, .06, .12, .24],
                case_ids=ids, question_ids=list(QUESTION_IDS), literature_question_ids=["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"],
                required_case_ids=[x for x in ids if x.endswith("/teacher")]
                + ["moment/equal_moments_spectrum", "moment/equal_moments_phase"],
                max_seconds=120 if smoke else 600, rank_interpretation="R quadrature response terms plus two analytic reaction-first split terms",
                product_convention="nodal real product; cyclic mode-pair convolution; separate continuum diagnostic uses 3/2+1 padded product")
