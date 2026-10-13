"""Finite-equation, independent-parent and precision contracts for this study."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import torch

from tdn.numerics.types import Geometry
from tdn.analysis.response.contracts import Physics
from tdn.analysis.response.numerics import inner, mean, center, resample

TARGETS = ('fd_same_grid', 'galerkin_same_grid')
SCHEMA = 'tdn.tangent-response/v1'


def validate_target(target):
    if target == 'galerkin_projected_fine':
        raise ValueError('Projected-fine tangent dynamics are unsupported: a hidden-state/lift and projected forcing contract is required')
    if target not in TARGETS:
        raise ValueError('An explicit same-grid FD or Galerkin target is required')
    return target


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def namespaced_seed(namespace, *parts):
    return int(digest({'program': 'tangent_response', 'namespace': namespace, 'parts': parts})[:15], 16) % (2**31-1)


def equation_geometry(record):
    validate_target(record['target_kind'])
    return Physics(record['kappa'], record['reaction_rate']), Geometry(tuple(record['grid']), tuple(record['lengths']))


@dataclass(frozen=True)
class PrecisionContract:
    input_dtype: str = 'float32'
    teacher_dtype: str = 'float64'
    path_dtype: str = 'float64'
    deployment_dtype: str = 'float32'

    def labels(self, teacher64, smooth_anchor64, deployed_anchor32):
        if teacher64.dtype != torch.float64 or smooth_anchor64.dtype != torch.float64 or deployed_anchor32.dtype != torch.float32:
            raise ValueError('Actual FP32 input/deployment and FP64 teacher/path arithmetic are required')
        if teacher64.shape != smooth_anchor64.shape or teacher64.shape != deployed_anchor32.shape:
            raise ValueError('Precision bridge fields must have the same geometry')
        physical = teacher64-smooth_anchor64
        bridge = smooth_anchor64-deployed_anchor32.double()
        return dict(physical_defect=physical, arithmetic_bridge=bridge,
                    deployment_defect=teacher64-deployed_anchor32.double())


def physical_metrics(error, geom, target, *, maximum_oversampling=2):
    """Per-field physical metrics; the maximum is explicitly a sampled maximum."""
    validate_target(target)
    if error.shape[0] != 1:
        raise ValueError('Research rows identify one independent parent/query at a time')
    e = error.double()
    m = mean(e)
    mse = float(inner(e, e, geom, target).item())
    centered_mse = float(inner(center(e), center(e), geom, target).item())
    sampled = resample(e, tuple(max(n, int(maximum_oversampling*n)) for n in geom.grid)) if target == 'galerkin_same_grid' else e
    return dict(mse=mse, rms=math.sqrt(max(0., mse)), mean_error=float(m.item()),
                mean_absolute=float(m.abs().item()), mean_squared=float(m.square().item()),
                centered_mse=centered_mse, centered_rms=math.sqrt(max(0., centered_mse)),
                maximum=float(sampled.abs().max().item()), maximum_kind='sampled_physical_interpolant' if target == 'galerkin_same_grid' else 'sampled_nodal',
                maximum_grid=list(sampled.shape[2:]), physical_norm='normalized_nodal_L2' if target == 'fd_same_grid' else 'normalized_interpolant_L2_with_Nyquist_mass')


def clean(value):
    if torch.is_tensor(value):
        return clean(value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items() if not str(k).startswith('_')}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
