"""Order-three corrections with fixed physical reference scales.

Only neural layers may run under BF16 autocast. Rate positivity, temporal
functions, modal reduction, anchors and physical scaling use FP32 or FP64.
The optional leading anchor consumes an independently audited physical e3;
this module does not substitute a continuum/PDE approximation for that audit.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from reference.temporal_core import (EncodedModes, anchored_amplitudes,
                                      temporal_defect, temporal_jets)


def _body(features: int, width: int, hidden_layers: int) -> nn.Sequential:
    if min(features, width, hidden_layers) <= 0:
        raise ValueError("feature count, width and hidden layer count must be positive")
    layers: list[nn.Module] = []
    for _ in range(hidden_layers):
        layers.extend((nn.Linear(features, width), nn.SiLU()))
        features = width
    return nn.Sequential(*layers)


def _zero_head(head: nn.Linear) -> None:
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)


def _audit_dtype(value: Tensor) -> Tensor:
    return value if value.dtype == torch.float64 else value.float()


def _mode_step(h: Tensor | float, values: Tensor, t_ref: float) -> Tensor:
    step = torch.as_tensor(h, device=values.device, dtype=values.dtype) / t_ref
    prefix = values.shape[:-2]
    if step.ndim and tuple(step.shape) == tuple(prefix):
        step = step.unsqueeze(-1).unsqueeze(-1)
    elif step.ndim and tuple(step.shape) == (*prefix, 1):
        step = step.unsqueeze(-1)
    return step


class CorrectionModel(nn.Module):
    def __init__(self, *, t_ref: float, U_ref: float) -> None:
        super().__init__()
        if not all(math.isfinite(v) and v > 0 for v in (t_ref, U_ref)):
            raise ValueError("reference scales must be finite and positive, and independent of h")
        self.t_ref, self.U_ref = float(t_ref), float(U_ref)


class TemporalMLP(CorrectionModel):
    """h-independent local encoder followed by analytic frozen-rate response."""
    family = "temporal_mlp"
    is_h_independent = True
    description = "Proposed order-anchored temporal units with local MLP parameters"

    def __init__(self, features: int, *, channels: int = 1, modes: int = 4,
                 width: int = 64, hidden_layers: int = 2, anchored: bool = False,
                 fixed_rates: list[float] | None = None,
                 t_ref: float = 1.0, U_ref: float = 1.0) -> None:
        super().__init__(t_ref=t_ref, U_ref=U_ref)
        if channels != 1:
            raise ValueError("the implemented RD pilot supports one scalar channel")
        if modes <= 0 or (anchored and modes < 2):
            raise ValueError("positive modes required; learned anchor needs at least two")
        self.channels, self.modes, self.anchored = channels, modes, anchored
        self.body = _body(features, width, hidden_layers)
        count = modes - 1 if anchored else modes
        self.amplitude = nn.Linear(width, channels * count)
        _zero_head(self.amplitude)
        if fixed_rates is None:
            self.rate = nn.Linear(width, modes)
            nn.init.zeros_(self.rate.weight)
            desired = torch.logspace(-1, 2, modes)
            with torch.no_grad():
                self.rate.bias.copy_(desired + torch.log(-torch.expm1(-desired)))
            self.register_buffer("fixed_rates", None)
        else:
            rates = torch.tensor(fixed_rates, dtype=torch.float32)
            if rates.shape != (modes,) or not torch.all(torch.isfinite(rates) & (rates >= 0)):
                raise ValueError("fixed rate dictionary must have M finite nonnegative entries")
            self.rate = None
            self.register_buffer("fixed_rates", rates)
            self.family = "fixed_rate"
            self.description = "Fixed dimensionless rate dictionary with nonlinear amplitude regression"

    def encode(self, features: Tensor, leading_defect: Tensor | None = None) -> EncodedModes:
        """Encode once per initial state; this API intentionally accepts no h.

        leading_defect has physical units U/time³ and shape [..., channels].
        Feature normalization must be learned from training parents only.
        """
        hidden = self.body(features)
        raw_a = _audit_dtype(self.amplitude(hidden))
        count = self.modes - 1 if self.anchored else self.modes
        amplitudes = raw_a.reshape(*features.shape[:-1], self.channels, count)
        if self.anchored:
            if leading_defect is None:
                raise ValueError("anchored model requires independently audited physical e3")
            coefficient = leading_defect.to(device=raw_a.device, dtype=raw_a.dtype)
            coefficient = coefficient * (self.t_ref**3 / self.U_ref)
            if tuple(coefficient.shape) != tuple(amplitudes.shape[:-1]):
                raise ValueError("physical e3 must have shape [..., channels]")
            amplitudes = anchored_amplitudes(amplitudes, coefficient)
        elif leading_defect is not None:
            raise ValueError("physical e3 supplied to an unanchored model")
        if self.rate is not None:
            rates = F.softplus(_audit_dtype(self.rate(hidden))).unsqueeze(-2)
        else:
            rates = self.fixed_rates.to(device=raw_a.device, dtype=raw_a.dtype)
            rates = rates.expand(*features.shape[:-1], 1, self.modes)
        return EncodedModes(amplitudes, rates)

    def decode(self, encoded: EncodedModes, h: Tensor | float) -> Tensor:
        tau = _mode_step(h, encoded.amplitudes, self.t_ref)
        return self.U_ref * temporal_defect(encoded.amplitudes, encoded.rates, tau)

    def jets(self, encoded: EncodedModes, h: Tensor | float) -> tuple[Tensor, ...]:
        tau = _mode_step(h, encoded.amplitudes, self.t_ref)
        return tuple(value * (self.U_ref / self.t_ref**order)
                     for order, value in enumerate(temporal_jets(encoded.amplitudes, encoded.rates, tau)))

    def forward(self, features: Tensor, h: Tensor | float,
                leading_defect: Tensor | None = None) -> Tensor:
        return self.decode(self.encode(features, leading_defect), h)


@dataclass(frozen=True)
class PolynomialCoefficients:
    numerator: Tensor
    denominator: Tensor | None = None


class PolynomialCorrection(CorrectionModel):
    """h³ times an h-polynomial, with coefficients from the same local inputs."""
    family = "polynomial"
    is_h_independent = True
    description = "Order-three polynomial temporal correction with local learned coefficients"

    def __init__(self, features: int, *, degree: int = 3, width: int = 64,
                 hidden_layers: int = 2, t_ref: float = 1.0, U_ref: float = 1.0) -> None:
        super().__init__(t_ref=t_ref, U_ref=U_ref)
        if degree < 0:
            raise ValueError("polynomial degree must be nonnegative")
        self.degree = degree
        self.body = _body(features, width, hidden_layers)
        self.amplitude = nn.Linear(width, degree + 1)
        _zero_head(self.amplitude)

    def encode(self, features: Tensor, leading_defect: Tensor | None = None) -> PolynomialCoefficients:
        if leading_defect is not None:
            raise ValueError("polynomial baseline has no exact-leading-anchor implementation")
        raw = _audit_dtype(self.amplitude(self.body(features)))
        return PolynomialCoefficients(raw.unsqueeze(-2))

    def decode(self, encoded: PolynomialCoefficients, h: Tensor | float) -> Tensor:
        coefficients = encoded.numerator
        tau = _mode_step(h, coefficients, self.t_ref)
        value = coefficients[..., -1:]
        for degree in reversed(range(self.degree)):
            value = value * tau + coefficients[..., degree:degree + 1]
        return (self.U_ref * tau.pow(3) * value).squeeze(-1)

    def forward(self, features: Tensor, h: Tensor | float) -> Tensor:
        return self.decode(self.encode(features), h)


class RationalCorrection(PolynomialCorrection):
    """Positive-denominator rational control on the declared nonnegative horizon."""
    family = "rational"
    description = "Order-three rational control with a positive denominator for h >= 0"

    def __init__(self, features: int, *, denominator_degree: int = 2, **kwargs) -> None:
        super().__init__(features, **kwargs)
        if denominator_degree < 1:
            raise ValueError("rational denominator degree must be positive")
        self.denominator_degree = denominator_degree
        width = self.amplitude.in_features
        self.denominator = nn.Linear(width, denominator_degree)
        nn.init.zeros_(self.denominator.weight)
        nn.init.constant_(self.denominator.bias, math.log(math.expm1(0.1)))

    def encode(self, features: Tensor, leading_defect: Tensor | None = None) -> PolynomialCoefficients:
        if leading_defect is not None:
            raise ValueError("rational baseline has no exact-leading-anchor implementation")
        hidden = self.body(features)
        numerator = _audit_dtype(self.amplitude(hidden)).unsqueeze(-2)
        denominator = F.softplus(_audit_dtype(self.denominator(hidden))).unsqueeze(-2)
        return PolynomialCoefficients(numerator, denominator)

    def decode(self, encoded: PolynomialCoefficients, h: Tensor | float) -> Tensor:
        numerator = super().decode(encoded, h)
        denominator = encoded.denominator
        if denominator is None:
            raise ValueError("rational encoder must supply its denominator")
        tau = _mode_step(h, denominator, self.t_ref)
        value = denominator[..., -1:]
        for degree in reversed(range(self.denominator_degree - 1)):
            value = value * tau + denominator[..., degree:degree + 1]
        positive = 1.0 + tau * value
        return numerator / positive.squeeze(-1)


class GenericMLPCorrection(CorrectionModel):
    """h³*MLP(features,h/t_ref) control; frozen temporal jets do not apply."""
    family = "generic_mlp"
    is_h_independent = False
    description = "h-conditioned order-three MLP baseline, not a frozen-rate temporal model"

    def __init__(self, features: int, *, width: int = 64, hidden_layers: int = 2,
                 t_ref: float = 1.0, U_ref: float = 1.0) -> None:
        super().__init__(t_ref=t_ref, U_ref=U_ref)
        self.body = _body(features + 1, width, hidden_layers)
        self.amplitude = nn.Linear(width, 1)
        _zero_head(self.amplitude)

    def forward(self, features: Tensor, h: Tensor | float) -> Tensor:
        dtype = torch.float64 if features.dtype == torch.float64 else torch.float32
        step = torch.as_tensor(h, device=features.device, dtype=dtype) / self.t_ref
        if step.ndim >= 2 and step.shape[-2:] == (1, 1):
            step = step.squeeze(-1)
        if step.ndim and step.shape == features.shape[:-1]:
            step = step.unsqueeze(-1)
        tau = torch.ones_like(features[..., :1], dtype=dtype) * step
        network_input = torch.cat((features, tau.to(features.dtype)), dim=-1)
        value = _audit_dtype(self.amplitude(self.body(network_input)))
        return self.U_ref * tau.pow(3) * value


class TaylorCorrection(PolynomialCorrection):
    """Leading h³ coefficient adaptation inspired by Taylor/hypersolver controls.

    This is an explicitly labelled local learned-defect adaptation, not a
    reproduction of a published hypersolver or an exact Taylor derivative.
    """
    family = "taylor"
    description = "Local learned h³ coefficient; Taylor/hypersolver-inspired adaptation"

    def __init__(self, features: int, **kwargs) -> None:
        super().__init__(features, degree=0, **kwargs)
