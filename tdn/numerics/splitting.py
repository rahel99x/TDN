"""Audited symmetric R(h/2) D(h) R(h/2), rightmost action first."""
from __future__ import annotations

import torch
from torch import Tensor

from .operators import broadcast_h, check_shape
from .subflows import diffusion_step, reaction_step
from .types import Equation, Geometry


SPLIT_METHOD = "exact_logistic_half_exact_discrete_periodic_fft_full_logistic_half"


def split_step(u: Tensor, h: float | Tensor, equation: Equation, geometry: Geometry,
               *, differentiable: bool = True) -> Tensor:
    check_shape(u, geometry)
    step = broadcast_h(h, u)

    def apply() -> Tensor:
        first = reaction_step(u, step / 2, equation)
        second = diffusion_step(first, step, equation, geometry)
        return reaction_step(second, step / 2, equation)

    if differentiable:
        return apply()
    with torch.no_grad():
        return apply()
