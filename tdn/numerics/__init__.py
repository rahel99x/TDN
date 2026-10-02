"""Scalar periodic logistic RD prototype and exact tiny linear controls."""
from .types import Equation, Geometry
from .operators import laplacian, rhs, weighted_norm
from .subflows import diffusion_step, reaction_step
from .splitting import SPLIT_METHOD, split_step
from .reference import (REFERENCE_METHOD, ReferenceResult, choose_substeps,
                        reference_step, refined_reference)
from .invariants import validate_state

__all__ = ["Equation", "Geometry", "laplacian", "rhs", "weighted_norm",
           "diffusion_step", "reaction_step", "SPLIT_METHOD", "split_step",
           "REFERENCE_METHOD", "ReferenceResult", "choose_substeps",
           "reference_step", "refined_reference", "validate_state"]
