"""Declared semidiscrete scalar reaction–diffusion problem.

Physical states use [batch, 1, *grid]. Geometry lengths have physical length
units, kappa has length**2 / time units, and reaction_rate has 1 / time units.
No advection or multicomponent HALO operator is implied by this prototype.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Geometry:
    grid: tuple[int, ...]
    lengths: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "grid", tuple(self.grid))
        object.__setattr__(self, "lengths", tuple(float(x) for x in self.lengths))
        if not self.grid or len(self.grid) != len(self.lengths):
            raise ValueError("grid and lengths must have the same positive dimension")
        if any(type(n) is not int or n < 2 for n in self.grid):
            raise ValueError("periodic grid counts must be integers >= 2")
        if any(not math.isfinite(x) or x <= 0 for x in self.lengths):
            raise ValueError("domain lengths must be finite and positive")

    @property
    def ndim(self) -> int:
        return len(self.grid)

    @property
    def dx(self) -> tuple[float, ...]:
        return tuple(length / n for n, length in zip(self.grid, self.lengths))

    @property
    def cell_volume(self) -> float:
        return math.prod(self.dx)

    @property
    def domain_volume(self) -> float:
        return math.prod(self.lengths)


@dataclass(frozen=True)
class Equation:
    kappa: float
    reaction_rate: float

    def __post_init__(self) -> None:
        for name in ("kappa", "reaction_rate"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
