"""Chunked physical split-plus-defect map with live differentiable features."""
from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from tdn.features.local import feature_chunk
from tdn.numerics import split_step


def _point_steps(h: Tensor | float, u: Tensor, geometry, start: int, stop: int) -> Tensor:
    dtype = torch.float64 if u.dtype == torch.float64 else torch.float32
    step = torch.as_tensor(h, device=u.device, dtype=dtype)
    if step.ndim == 0:
        return step
    if step.numel() != u.shape[0]:
        raise ValueError("corrected_step accepts one shared h or one h per parent")
    cells = math.prod(geometry.grid)
    parents = torch.arange(start, stop, device=u.device) // cells
    return step.reshape(-1)[parents, None, None]


def corrected_step(u: Tensor, h: Tensor | float, equation, geometry, model, *,
                   chunk_size: int = 32768, checkpoint_chunks: bool = False) -> Tensor:
    """Advance [B,1,*grid] using one local correction evaluation per cell.

    Features are computed from the current initial state on every call.
    Chunked gathers bound encoder/feature working storage during inference.
    Non-reentrant checkpointing recomputes both features and the encoder during
    rollout training; state, amplitude, rate, base-map and time gradients remain
    in the graph. The exact periodic FFT base is a global operation, regardless
    of this correction's radius-one observation support.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    t_ref, U_ref = float(model.t_ref), float(model.U_ref)
    physical = u if u.dtype == torch.float64 else u.float()
    base = split_step(physical, h, equation, geometry)
    count = physical.numel()
    gradients = torch.is_grad_enabled()
    chunks: list[Tensor] = []
    correction = None if gradients else torch.empty((count, 1), dtype=base.dtype, device=base.device)
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        step = _point_steps(h, physical, geometry, start, stop)

        # Bind bounds per closure: checkpoint backward may run after this loop.
        def apply_chunk(state: Tensor, requested_h: Tensor, left=start, right=stop) -> Tensor:
            features = feature_chunk(state, equation, geometry, left, right,
                                     t_ref=t_ref, U_ref=U_ref)
            result = model(features, requested_h)
            if tuple(result.shape) != (right - left, 1):
                raise ValueError("scalar correction must have shape [points, 1]")
            return result if result.dtype == torch.float64 else result.float()

        if checkpoint_chunks and gradients:
            value = checkpoint(apply_chunk, physical, step, use_reentrant=False)
        else:
            value = apply_chunk(physical, step)
        if gradients:
            chunks.append(value)
        else:
            correction[start:stop] = value
    if gradients:
        correction = torch.cat(chunks, dim=0)
    correction = correction.reshape(physical.shape[0], *geometry.grid).unsqueeze(1)
    return base + correction.to(dtype=base.dtype)
