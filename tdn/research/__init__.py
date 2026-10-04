"""Experimental full-state architectures and matched controls.

The existing production-pilot model/config/checkpoint interfaces are unchanged.
Use scripts/research.py or the allocated research workflow for these models.
"""
from __future__ import annotations

RESEARCH_FAMILIES = ("confluent_decay", "fixed_decay_r", "fixed_decay", "fixed_undamped",
                     "reaction_clock", "reaction_additive", "transport", "temporal_mlp")
OPTIONAL_FAMILIES = ("reaction_hybrid", "reaction_polynomial")
CLASSICAL_FAMILIES = ("split", "richardson_split", "e3_anchor")


def build_research_model(family: str, *, ndim: int = 2, width: int = 16,
                         t_ref: float = 1., U_ref: float = 1.):
    """Construct a declared research model, never silently substitute a family.

    The original temporal baseline retains its actual width 32/two-layer body.
    Other learned families use the supplied small width and one hidden layer.
    The initial experiment fits one shared confluent rate over all train regimes;
    this is stricter than the historical per-parent teacher-informed oracle.
    """
    # Keep protocol/config inspection on login nodes free of scientific imports.
    from .adapters import ExistingTemporalTDN, RichardsonStep, SplitStep
    from .confluent import ConfluentTDN, FixedDecayTDN
    from .reaction import (AdditiveClockTDN, HybridReactionTDN,
                           PolynomialClockTDN, ReactionClockTDN)
    from .transport import LeadingDefectStep, TransportTDN
    common = dict(ndim=ndim, width=width, t_ref=t_ref, U_ref=U_ref)
    if family == "confluent_decay":
        return ConfluentTDN(**common)
    if family in ("fixed_decay", "fixed_undamped", "fixed_decay_r"):
        multiplier = {"fixed_decay": 2., "fixed_undamped": 0., "fixed_decay_r": 1.}[family]
        return FixedDecayTDN(decay_multiplier=multiplier, **common)
    if family == "reaction_clock":
        return ReactionClockTDN(**common)
    if family == "reaction_additive":
        return AdditiveClockTDN(**common)
    if family == "reaction_polynomial":
        return PolynomialClockTDN(**common)
    if family == "reaction_hybrid":
        return HybridReactionTDN(**common)
    if family == "transport":
        return TransportTDN(**common)
    if family == "temporal_mlp":
        return ExistingTemporalTDN(ndim=ndim, t_ref=t_ref, U_ref=U_ref)
    if family == "split":
        return SplitStep()
    if family == "richardson_split":
        return RichardsonStep()
    if family == "e3_anchor":
        return LeadingDefectStep()
    raise ValueError(f"Unsupported research model family: {family}")


__all__ = ["build_research_model", "RESEARCH_FAMILIES", "OPTIONAL_FAMILIES",
           "CLASSICAL_FAMILIES"]
