"""Frozen, input-only descendants for descriptive physical generalization.

The primary statistical unit remains the original independent parent.  These
interventions do not add independent fields or enter checkpoint selection.
Teachers must be acquired in the declared CPU reference stages.  The function
below only transforms a continuous input specification; it never reads a
prediction, reference, error, or selected checkpoint.
"""
from __future__ import annotations

import copy
import math

from .fields import digest


VARIANTS = (
    "original", "reaction_interpolation", "reaction_extrapolation",
    "diffusion_interpolation", "diffusion_extrapolation", "amplitude_half",
    "amplitude_near_boundary", "mean_pair_low", "mean_pair_high",
)
GROUPS = {
    "reaction": VARIANTS[:3],
    "diffusion": ("original", "diffusion_interpolation", "diffusion_extrapolation"),
    "amplitude": ("original", "amplitude_half", "amplitude_near_boundary"),
    "mean": ("mean_pair_low", "mean_pair_high"),
}
PARAMETER_VALUES = {
    "reaction_interpolation": ("reaction_rate", 1.5, (.5, 3., 8., 25., 60.)),
    "reaction_extrapolation": ("reaction_rate", 90., (.5, 3., 8., 25., 60.)),
    "diffusion_interpolation": ("kappa", .001, (.0003, .003, .03)),
    "diffusion_extrapolation": ("kappa", .06, (.0003, .003, .03)),
}


def _amplitude(parent):
    return sum(abs(float(term["amplitude"])) for term in parent["terms"])


def _set_amplitude(parent, target):
    current = _amplitude(parent)
    if current == 0:
        return 1., "Constant input has no centered shape to amplify; retained as an exact control"
    factor = target/current
    for term in parent["terms"]:
        term["amplitude"] = float(term["amplitude"])*factor
    return factor, None


def transform_parent(parent, variant):
    """Return a bounded continuous-field descendant, preserving parent identity.

    Mean-pair members share exactly the same centered field.  If the original
    centered amplitude exceeds .12, both members use the same common scaling;
    the isolated mean comparison is between the two members, not either member
    and the original.  ``amplitude_near_boundary`` changes the analytic L1
    envelope, not a claimed sampled extremum.  No clipping is performed.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown frozen physical intervention: {variant}")
    if parent.get("intervention_variant", "original") != "original":
        raise ValueError("Interventions must start from the original parent, not be composed")
    result = copy.deepcopy(parent)
    mean = float(parent["mean"])
    amplitude = _amplitude(parent)
    if not math.isfinite(mean) or not 0 <= mean <= 1 or not math.isfinite(amplitude):
        raise ValueError("Physical interventions require a finite bounded parent")
    if amplitude > min(mean, 1-mean)+1e-12:
        raise ValueError("Input parent lacks an analytic [0,1] bound; clipping is forbidden")
    for term in parent["terms"]:
        if not math.isfinite(float(term["phase"])):
            raise ValueError("Parent phases must be finite")
    for parameter in ("kappa", "reaction_rate"):
        if not math.isfinite(float(parent[parameter])) or parent[parameter] < 0:
            raise ValueError("Physical coefficients must be finite and nonnegative")
    metadata = dict(variant=variant, changed_quantity="none", original_mean=mean,
        original_amplitude_l1=amplitude, amplitude_scale=1.,
        cluster_parent_id=parent["parent_id"], descriptive_only=True,
        selection_scope="Frozen input intervention; excluded from training, validation selection and primary inference",
        comparison_scope="Paired descendant of the same parent; not an independent observation")
    if variant in PARAMETER_VALUES:
        parameter, value, support = PARAMETER_VALUES[variant]
        result[parameter] = value
        metadata.update(changed_quantity=parameter, original_value=float(parent[parameter]), value=value,
            declared_training_parameter_support=list(support),
            generalization_kind="interpolation" if min(support)<value<max(support) else "extrapolation",
            support_scope="Relative to the frozen main generator menu; actual acquired training support is reported separately",
            paired_invariants="Initial continuous field, other physical coefficient, geometry and requested horizon")
    elif variant in ("amplitude_half", "amplitude_near_boundary"):
        target = .5*amplitude if variant == "amplitude_half" else .94*min(mean, 1-mean)
        factor, note = _set_amplitude(result, target)
        metadata.update(changed_quantity="centered amplitude", amplitude_scale=factor,
            paired_invariants="Initial mean, phase, mode geometry, physical coefficients and horizon",
            amplitude_contract="Coefficient L1 envelope; observed extrema are measured separately", note=note)
    elif variant in ("mean_pair_low", "mean_pair_high"):
        target = min(amplitude, .12)
        factor, note = _set_amplitude(result, target)
        result["mean"] = .15 if variant == "mean_pair_low" else .85
        metadata.update(changed_quantity="mean within a matched pair", amplitude_scale=factor,
            comparison_partner="mean_pair_high" if variant == "mean_pair_low" else "mean_pair_low",
            paired_invariants="Identical centered field, phase, mode geometry, physical coefficients and horizon",
            original_comparison_caveat="If the common amplitude cap activates, comparison to original also changes amplitude",
            note=note)
    result["amplitude_bound"] = _amplitude(result)
    metadata.update(mean=float(result["mean"]), amplitude_l1=result["amplitude_bound"],
        guaranteed_lower=float(result["mean"])-result["amplitude_bound"],
        guaranteed_upper=float(result["mean"])+result["amplitude_bound"],
        reaction_rate=float(result["reaction_rate"]), kappa=float(result["kappa"]))
    result.update(intervention_variant=variant, intervention=metadata,
        panel="physical_interventions", descriptive_only=True,
        independence_scope="Input intervention descendant; unchanged original parent ID")
    # Preserve the sealed parent identity/hash.  Bind the modified observation
    # separately, exactly as for phase and grid-relative descendants.
    result["descendant_sha256"] = digest({key: value for key, value in result.items()
        if key != "descendant_sha256"})
    return result


def validate_held_out_values(parents, variants=VARIANTS):
    """Reject accidental parameter-value leakage into fit/selection cohorts.

    This is an exact-value holdout, distinct from being outside the convex hull.
    It must be checked before acquiring intervention references, not used to
    choose replacement values after inspecting an experiment.
    """
    fitting = [p for p in parents if p.get("split") in ("train", "validation")]
    for variant in variants:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown frozen physical intervention: {variant}")
        if variant not in PARAMETER_VALUES:
            continue
        parameter, value, _ = PARAMETER_VALUES[variant]
        if any(math.isclose(float(p[parameter]), value, rel_tol=1e-12, abs_tol=1e-15) for p in fitting):
            raise ValueError(f"Intervention {variant} value entered training/validation; do not relabel it held out")
    return dict(status="VERIFIED", fitting_parents=len(fitting),
        scope="Exact parameter values absent from training and validation; no population-independence claim")
