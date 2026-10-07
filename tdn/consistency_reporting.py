"""Tower projections for the declared consistency experiment, never model claims."""
from __future__ import annotations

from tdn.premix_reporting import COLUMNS, publish_premix_outputs
from tdn.analysis.consistency.comparison import DIAGNOSTIC_FIELDS

SCHEMAS = {"tdn.consistency-neural/v1", "tdn.consistency-comparisons/v1"}
EXTRA_COLUMNS = {
    "premix_candidates": ["checkpoint_selection", "signed_mean_error", "base_rms", "base_max_error",
        "base_mean_error", "base_spatial_rms", "rms_vs_base", "max_vs_base", "mean_vs_base", "spatial_vs_base"],
    "premix_checks": ["check_id", "case", "required", "maximum_absolute_error", "absolute_tolerance",
        "max_base_difference", "target_mean", "output_min", "output_max", "observed_orders_record",
        "missing_parameter_gradients_record", "nonfinite_parameter_gradients_record", "time_gradients_record", "error", "tolerance",
        "target_mean_record", "final_mean_record", "mean_law_error", "diffusion_mean_error",
        "commutator_mean_error", "reaction_rate", "mean_law_identity", "mean_state_record",
        "variance_record", "physical_mean_derivative_record"],
    "premix_comparisons": ["comparison_kind", *[side + "_" + field for side in ("ours", "baseline")
        for field in (*DIAGNOSTIC_FIELDS, "rms_regresses_base", "diagnostic_issues_record")]],
    "premix_groups": ["comparison_kind"],
}
CONSISTENCY_COLUMNS = {name.replace("premix", "consistency", 1): [*columns,
    *[key for key in EXTRA_COLUMNS.get(name, ()) if key not in columns]] for name, columns in COLUMNS.items()}


def publish_consistency_outputs(report_dir, source_dirs, *, delegated_sources=None):
    return publish_premix_outputs(report_dir, source_dirs, delegated_sources=delegated_sources,
        _suite="consistency", _schemas=SCHEMAS, _extra_columns=EXTRA_COLUMNS,
        _neural_schema="tdn.consistency-neural/v1", _comparison_schema="tdn.consistency-comparisons/v1")


__all__ = ["publish_consistency_outputs", "CONSISTENCY_COLUMNS"]
