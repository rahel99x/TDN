"""Matched local-input order-three model families for the pilot controls."""
from __future__ import annotations

import torch

from .temporal import (GenericMLPCorrection, PolynomialCorrection,
                       RationalCorrection, TaylorCorrection, TemporalMLP)


def build_model(feature_count: int, model_config: dict, *, t_ref: float = 1.0,
                U_ref: float = 1.0):
    config = dict(model_config)
    family = config.get("family", "temporal_mlp")
    family = {"temporal_parameter_mlp": "temporal_mlp",
              "time_conditioned": "generic_mlp"}.get(family, family)
    if family in ("kan", "temporal_kan", "generic_kan"):
        raise ValueError("KAN is an optional gated branch; it is not implemented or validated")
    if int(config.get("channels", 1)) != 1:
        raise ValueError("implemented problem/model supports one scalar channel")
    if config.get("h_independent_encoder", True) is False and family == "temporal_mlp":
        raise ValueError("primary temporal encoder must be independent of queried h")
    if config.get("shared_rates_across_species", True) is False:
        raise ValueError("species-specific rates are not implemented in the scalar pilot")
    if config.get("rate_parameterization", "softplus") != "softplus":
        raise ValueError("only FP32/FP64 softplus positivity is implemented")
    if config.get("amplitude_initialization", "zero") != "zero":
        raise ValueError("only zero free-amplitude initialization is implemented")
    common = dict(width=int(config.get("width", 64)),
                  hidden_layers=int(config.get("hidden_layers", 2)),
                  t_ref=t_ref, U_ref=U_ref)
    anchored = bool(config.get("anchored_leading_defect", config.get("anchored", False)))
    if family in ("temporal_mlp", "fixed_rate"):
        modes = int(config.get("modes", 4))
        rates = None
        if family == "fixed_rate":
            rates = config.get("fixed_rates", torch.logspace(-1, 2, modes).tolist())
        return TemporalMLP(feature_count, modes=modes, anchored=anchored,
                           fixed_rates=rates, **common)
    if anchored:
        raise ValueError("exact-leading anchor is implemented only for temporal units")
    if family == "polynomial":
        return PolynomialCorrection(feature_count, degree=int(config.get("polynomial_degree", 3)), **common)
    if family == "rational":
        return RationalCorrection(feature_count,
                                  degree=int(config.get("polynomial_degree", 2)),
                                  denominator_degree=int(config.get("rational_degree", 2)), **common)
    if family == "generic_mlp":
        return GenericMLPCorrection(feature_count, **common)
    if family == "taylor":
        return TaylorCorrection(feature_count, **common)
    raise ValueError(f"unsupported model family: {family}")


__all__ = ["build_model", "TemporalMLP", "PolynomialCorrection",
           "RationalCorrection", "GenericMLPCorrection", "TaylorCorrection"]
