"""Explicit equation, target and precision contracts for the Response study."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import torch
from tdn.numerics.types import Geometry

Target = Literal['fd_same_grid', 'galerkin_same_grid', 'galerkin_projected_fine']
TARGETS = ('fd_same_grid', 'galerkin_same_grid', 'galerkin_projected_fine')


def validate_target(target: str) -> str:
    if target not in TARGETS:
        raise ValueError(f'An explicit Response target in {TARGETS} is required, got {target!r}')
    return target


@dataclass(frozen=True)
class Physics:
    """Known coefficients; tensor values retain their complete autograd graph.

    Values may be scalars or one scalar per independent field. Shape matching
    happens when an operator receives a field, never by implicit spatial
    broadcasting. Validation reads values, but never replaces them by floats.
    """
    kappa: float | torch.Tensor
    reaction_rate: float | torch.Tensor

    def __post_init__(self):
        for name in ('kappa', 'reaction_rate'):
            value = torch.as_tensor(getattr(self, name))
            if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
                raise ValueError(f'{name} must be finite and nonnegative')


@dataclass(frozen=True)
class TargetContract:
    kind: Target
    grid: tuple[int, ...]
    lengths: tuple[float, ...]
    fine_grid: tuple[int, ...] | None = None

    def __post_init__(self):
        validate_target(self.kind)
        Geometry(self.grid, self.lengths)
        if self.kind == 'galerkin_projected_fine':
            if self.fine_grid is None or len(self.fine_grid) != len(self.grid) or any(f <= n for f, n in zip(self.fine_grid, self.grid)):
                raise ValueError('Projected-fine targets require an explicitly finer working grid')
        elif self.fine_grid is not None:
            raise ValueError('A same-grid target cannot silently carry a fine-grid teacher')

    @property
    def operator_scope(self):
        return ('Fine Galerkin evolution then Fourier projection; coarse RHS is not this dynamics'
                if self.kind == 'galerkin_projected_fine' else 'Declared finite equation on the supplied geometry')


@dataclass(frozen=True)
class PrecisionContract:
    """Separate mathematical anchor defects from deployed rounding defects."""
    input_dtype: str = 'float32'
    teacher_dtype: str = 'float64'
    anchor_dtype: str = 'float64'
    kind: str = 'physical_defect'

    def __post_init__(self):
        if self.kind not in ('physical_defect', 'deployment_defect'):
            raise ValueError('Precision label must be physical_defect or deployment_defect')
        if self.input_dtype not in ('float32', 'float64') or self.teacher_dtype != 'float64':
            raise ValueError('Declare FP32/FP64 inputs and FP64 teachers')
        expected = 'float64' if self.kind == 'physical_defect' else 'float32'
        if self.anchor_dtype != expected:
            raise ValueError(f'{self.kind} requires {expected} anchor arithmetic')

    def labels(self, teacher, physical_anchor, deployed_anchor):
        if teacher.dtype != torch.float64 or physical_anchor.dtype != torch.float64:
            raise ValueError('Teacher and physical anchor must retain FP64')
        if deployed_anchor.dtype != torch.float32:
            raise ValueError('Deployment anchor must be actually evaluated in FP32')
        return {'physical_defect': teacher - physical_anchor,
                'deployment_defect': teacher - deployed_anchor.to(torch.float64),
                'anchor_rounding': physical_anchor - deployed_anchor.to(torch.float64)}
