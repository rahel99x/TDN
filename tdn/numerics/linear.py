"""Tier A exact noncommuting matrix controls; no spatial teacher ambiguity."""
from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor


def _check(A: Tensor, B: Tensor) -> None:
    if A.ndim != 2 or A.shape != B.shape or A.shape[0] != A.shape[1]:
        raise ValueError("A and B must be same-shaped square matrices")
    if A.dtype != B.dtype or A.device != B.device or not A.is_floating_point():
        raise ValueError("A and B must share a real floating dtype and device")


def control_matrices(*, dtype: torch.dtype = torch.float64,
                     device: str | torch.device = "cpu") -> tuple[Tensor, Tensor, Tensor]:
    A = torch.tensor([[-0.3, 0.8], [-0.1, -0.2]], dtype=dtype, device=device)
    B = torch.tensor([[-1.0, 0.1], [0.4, -0.8]], dtype=dtype, device=device)
    u0 = torch.tensor([0.7, -0.4], dtype=dtype, device=device)
    return A, B, u0


def exact_matrix(A: Tensor, B: Tensor, h: float | Tensor) -> Tensor:
    _check(A, B)
    return torch.matrix_exp(torch.as_tensor(h, dtype=A.dtype, device=A.device) * (A + B))


def split_matrix(A: Tensor, B: Tensor, h: float | Tensor) -> Tensor:
    _check(A, B)
    step = torch.as_tensor(h, dtype=A.dtype, device=A.device)
    half = torch.matrix_exp(step * A / 2)
    return half @ torch.matrix_exp(step * B) @ half


def exact_flow(u: Tensor, h: float | Tensor, A: Tensor, B: Tensor) -> Tensor:
    return torch.einsum("ij,...j->...i", exact_matrix(A, B, h), u)


def symmetric_split(u: Tensor, h: float | Tensor, A: Tensor, B: Tensor) -> Tensor:
    return torch.einsum("ij,...j->...i", split_matrix(A, B, h), u)


def leading_matrix(A: Tensor, B: Tensor) -> Tensor:
    """Coefficient e3 in exp(h(A+B))-exp(hA/2)exp(hB)exp(hA/2).

    Multiply power-series matrices through degree three directly. This avoids
    any convention-dependent continuum commutator sign formula.
    """
    _check(A, B)
    coefficient = torch.zeros_like(A)
    for p in range(4):
        for q in range(4 - p):
            r = 3 - p - q
            coefficient = coefficient + (
                torch.linalg.matrix_power(A / 2, p) / math.factorial(p)
                @ (torch.linalg.matrix_power(B, q) / math.factorial(q))
                @ (torch.linalg.matrix_power(A / 2, r) / math.factorial(r))
            )
    return torch.linalg.matrix_power(A + B, 3) / 6 - coefficient


def rollout_convergence(A: Tensor, B: Tensor, u0: Tensor, *, total_time: float = 1.0,
                        counts: Iterable[int] = (8, 16, 32, 64), anchored: bool = False) -> dict:
    """Raw fixed-final-time errors and adjacent observed orders, no fitted claim."""
    counts = tuple(counts)
    if not math.isfinite(total_time) or total_time <= 0:
        raise ValueError("total_time must be finite and positive")
    if not counts or any(type(n) is not int or n < 1 for n in counts):
        raise ValueError("rollout counts must be positive integers")
    if any(b <= a for a, b in zip(counts, counts[1:])):
        raise ValueError("rollout counts must strictly increase")
    truth = exact_flow(u0, total_time, A, B)
    e3 = leading_matrix(A, B)
    errors = []
    for n in counts:
        h = total_time / n
        matrix = split_matrix(A, B, h)
        if anchored:
            matrix = matrix + h**3 * e3
        got = torch.linalg.matrix_power(matrix, n) @ u0
        errors.append(float(torch.linalg.vector_norm(got - truth)))
    orders = [math.log(a / b) / math.log(n2 / n1) if a > 0 and b > 0 else None
              for a, b, n1, n2 in zip(errors, errors[1:], counts, counts[1:])]
    return {"counts": list(counts), "horizons": [total_time / n for n in counts],
            "errors": errors, "observed_orders": orders, "anchored": anchored,
            "total_time": total_time, "norm": "euclidean_vector_l2"}
