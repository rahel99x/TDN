"""Full-state adapters for established learned and classical controls."""
from __future__ import annotations

import torch
from torch import nn

from tdn.features.local import feature_count
from tdn.models.solver import corrected_step
from tdn.models.temporal import TemporalMLP
from tdn.numerics import split_step
from tdn.train.model import NormalizedModel
from .common import validate_normalization


class ExistingTemporalTDN(nn.Module):
    """Unmodified width-32/two-layer/four-mode baseline under the same state API."""

    def __init__(self, *, ndim: int = 2, t_ref: float = 1., U_ref: float = 1.):
        super().__init__()
        count = feature_count(ndim)
        self.t_ref, self.U_ref = float(t_ref), float(U_ref)
        base = TemporalMLP(count, modes=4, width=32, hidden_layers=2,
                           t_ref=t_ref, U_ref=U_ref)
        self.model = NormalizedModel(base, torch.zeros(count), torch.ones(count))

    def set_normalization(self, mean, std):
        mean, std = validate_normalization(mean, std, self.model.feature_mean.numel(),
                                           dtype=self.model.feature_mean.dtype,
                                           device=self.model.feature_mean.device)
        with torch.no_grad():
            self.model.feature_mean.copy_(mean)
            self.model.feature_std.copy_(std)
        return self

    def forward(self, u, h, equation, geometry):
        return corrected_step(u, h, equation, geometry, self.model)


class SplitStep(nn.Module):
    def forward(self, u, h, equation, geometry):
        return split_step(u, h, equation, geometry)


class RichardsonStep(nn.Module):
    """Three physical Strang calls; do not clip extrapolated values."""

    def forward(self, u, h, equation, geometry):
        half = h / 2
        fine = split_step(split_step(u, half, equation, geometry), half, equation, geometry)
        return (4 * fine - split_step(u, h, equation, geometry)) / 3
