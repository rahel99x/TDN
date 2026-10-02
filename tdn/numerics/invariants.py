"""Fail explicitly on scalar logistic inadmissibility; never repair by clipping."""
from __future__ import annotations

import math

import torch
from torch import Tensor


def validate_state(u: Tensor, tolerance: float | None = None) -> None:
    """Boundary/validation check, intentionally outside compiled tensor kernels.

    Permit only an absolute 64*machine-epsilon rounding allowance by default.
    This allowance is not a physical projection or a fitted error tolerance.
    """
    if not u.is_floating_point():
        raise TypeError("physical states must be real floating-point tensors")
    atol = 64 * torch.finfo(u.dtype).eps if tolerance is None else tolerance
    if not math.isfinite(atol) or atol < 0:
        raise ValueError("state admissibility tolerance must be finite and nonnegative")
    if not bool(torch.isfinite(u).all()):
        raise ValueError("nonfinite physical state")
    if bool((u < -atol).any()) or bool((u > 1 + atol).any()):
        raise ValueError("scalar logistic state outside the admissible interval [0,1]")
