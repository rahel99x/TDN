"""Target-aware endpoint dictionaries and input-only correction gain estimates.

No function divides by a queried horizon or a physical gate. The gain estimates
describe the declared time/gate basis before its physical output scale; they
are not a Jacobian norm of a learned encoder or a bound on prediction error.
"""
from __future__ import annotations

import math
import torch

from . import numerics as n
from .features import FrozenNormalization, build_features, correction_gate


def target_time_order(target, requested="target"):
    from .contracts import validate_target
    validate_target(target)
    if requested == "target":
        return 2 if target == "galerkin_projected_fine" else 3
    if requested == "legacy":
        return 3
    if type(requested) is int and requested in (2, 3):
        return requested
    raise ValueError("Endpoint time order must be target, legacy, 2 or 3")


def response_kernel(x, order):
    """Integral_0^1 exp(-x*(1-s))*s**(order-1) ds, for order 2/3.

    Its series and inverse expressions have finite inactive branches, including
    at x=0. Preserve FP64 audits and promote lower-precision inputs to FP32.
    """
    if order not in (2, 3):
        raise ValueError("Response kernel order must be 2 or 3")
    x = torch.as_tensor(x)
    if not x.is_floating_point() or not bool(torch.isfinite(x).all()) or bool((x < 0).any()):
        raise ValueError("Response kernels require finite nonnegative floating rates")
    z = x if x.dtype == torch.float64 else x.float()
    small = z <= .5
    s = torch.where(small, z, torch.zeros_like(z))
    value = torch.full_like(s, math.factorial(order-1)*(-1.)**18/math.factorial(18+order))
    for j in range(17, -1, -1):
        value = value*s+math.factorial(order-1)*(-1.)**j/math.factorial(j+order)
    inv = z.clamp(min=.5).reciprocal()
    if order == 2:
        regular = inv*(1+torch.expm1(-z.clamp(min=.5))*inv)
    else:
        regular = inv*(1-2*inv-2*torch.expm1(-z.clamp(min=.5))*inv.square())
    return torch.where(small, value, regular)


def endpoint_basis(tau, *, order, rank=4, kind="mixed"):
    """[B,J,1,...] signed-coefficient dictionary with explicit O(h**order).

    Mixed dictionaries include a constant channel; they do not force every
    learned response to decay. Localized channels use a bounded time coordinate
    and smooth Gaussian windows. All coefficients are free to have either sign.
    """
    if rank not in (4, 8) or kind not in ("mixed", "response") or order not in (2, 3):
        raise ValueError("Endpoint dictionary requires rank 4/8, mixed/response, and order 2/3")
    tau = torch.as_tensor(tau)
    if not tau.is_floating_point() or not bool(torch.isfinite(tau).all()) or bool((tau < 0).any()):
        raise ValueError("Endpoint dictionary requires finite nonnegative time")
    if kind == "response":
        rates = torch.logspace(-1, 2, rank, dtype=tau.dtype, device=tau.device)
        return torch.cat([tau.pow(order)*response_kernel(tau*rate, order) for rate in rates], dim=1)
    # Inverse form avoids tau**order overflow at long, finite horizons.
    coordinate = tau/(1+tau)
    envelope = coordinate.pow(order)/(coordinate.pow(order)+(1-coordinate).pow(order))
    if rank == 4:
        channels = (torch.ones_like(tau), torch.exp(-tau), (1+tau).reciprocal(),
                    torch.exp(-((coordinate-.5)/.25).square()))
    else:
        channels = (torch.ones_like(tau), torch.exp(-.1*tau), torch.exp(-tau), torch.exp(-10*tau),
                    (1+tau).reciprocal(), (1+10*tau).reciprocal().square(),
                    torch.exp(-((coordinate-.25)/.2).square()),
                    torch.exp(-((coordinate-.75)/.2).square()))
    return torch.cat([envelope*channel for channel in channels], dim=1)


def correction_gains(u, h, eq, geom, target, *, family, config=None):
    """Input-only nominal gains, to be fitted using accepted TRAIN queries only.

    Each output has shape [B,1,1,...]. Fit a parent-weighted RMS over these
    values and store positive finite ``mean_gain`` and ``centered_gain`` in
    model normalization. Coupled-source centered gain is a nominal envelope,
    not the state-dependent learned propagator/field gain. No labels enter.
    """
    config = dict(config or {})
    family = {"endpoint": "residual_fno", "endpoint_tdn": "residual_fno", "original_tdn": "temporal"}.get(family, family)
    step = n.validate(u, h, geom, target)
    t_ref = float(config.get("t_ref", .1))
    if not math.isfinite(t_ref) or t_ref <= 0:
        raise ValueError("Gain reference time must be positive and finite")
    tau = (step/t_ref).expand(u.shape[0], 1, *([1]*geom.ndim))
    norm = FrozenNormalization(config.get("normalization")).to(device=u.device, dtype=u.dtype)
    _, stats = build_features(u, eq, geom, target, norm, t_ref=t_ref, mode=config.get("feature_mode", "phase"))
    gate = correction_gate(stats, target)
    if family in ("adaptive_temporal", "spatial_temporal"):
        basis = endpoint_basis(tau, order=target_time_order(target, config.get("time_order", "target")),
                               rank=int(config.get("rank", 4)), kind=config.get("temporal_basis", "mixed"))
        gain = gate*basis.square().mean(1, keepdim=True).sqrt()
    elif family == "temporal":
        rank = int(config.get("rank", 4))
        rates = torch.logspace(-1, 2, rank, dtype=u.dtype, device=u.device)
        basis = torch.cat([tau.pow(3)*response_kernel(tau*rate, 3) for rate in rates], 1)
        gain = gate*basis.square().mean(1, keepdim=True).sqrt()
    elif family == "direct_fno":
        gain = tau
    elif family in ("residual_fno", "coupled_source"):
        order = int(config.get("source_order", 1 if target == "galerkin_projected_fine" else 2))
        if order not in (0, 1, 2):
            raise ValueError("Source onset order must be 0, 1 or 2")
        envelope = torch.ones_like(tau) if order == 0 else tau.pow(order)/(1+tau.pow(order))
        gain = gate*tau*envelope
    else:
        raise ValueError("Gain fitting is only defined for learned Response families")
    return {"mean_gain": gain, "centered_gain": gain}
