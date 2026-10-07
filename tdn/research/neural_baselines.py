"""Representative spatial neural backbones under two declared solver tracks.

These are small, independently trained adaptations of established residual
CNN, U-Net and Fourier-operator designs, not reproductions of published
pretrained models. All receive the same twelve physical features and queried
horizon channel. The direct track learns a state increment; the hybrid track
learns a cubic correction to one physical Strang step. Both use the same
explicit signed-capacity output map, with no output clipping.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tdn.features.local import extract_features, feature_count
from tdn.numerics.operators import broadcast_h, check_shape
from tdn.numerics.splitting import split_step

from .common import validate_normalization
from .reaction import capacity_update


def _periodic_conv(inputs: int, outputs: int) -> nn.Conv2d:
    return nn.Conv2d(inputs, outputs, 3, padding=1, padding_mode="circular")


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.first = _periodic_conv(channels, channels)
        self.second = _periodic_conv(channels, channels)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.second(F.silu(self.first(F.silu(x))))


class _SpatialSolver(nn.Module):
    backbone_name = "abstract"

    def __init__(self, *, mode: str, ndim: int, width: int, t_ref: float,
                 U_ref: float):
        super().__init__()
        if mode not in ("direct", "hybrid"):
            raise ValueError("Neural baseline mode must be 'direct' or 'hybrid'")
        if ndim != 2:
            raise ValueError("Spatial neural baselines currently require ndim=2")
        if type(width) is not int or width <= 0:
            raise ValueError("Neural baseline width must be a positive integer")
        if not all(math.isfinite(float(x)) and float(x) > 0 for x in (t_ref, U_ref)):
            raise ValueError("Reference scales must be finite and positive")
        self.mode, self.width = mode, width
        self.t_ref, self.U_ref = float(t_ref), float(U_ref)
        count = feature_count(ndim)
        self.register_buffer("feature_mean", torch.zeros(count))
        self.register_buffer("feature_std", torch.ones(count))

    def set_normalization(self, mean, std):
        mean, std = validate_normalization(mean, std, self.feature_mean.numel(),
                                           dtype=self.feature_mean.dtype,
                                           device=self.feature_mean.device)
        with torch.no_grad():
            self.feature_mean.copy_(mean)
            self.feature_std.copy_(std)
        return self

    def _initialize_head(self) -> None:
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _network(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def _feature_state(self, u: Tensor) -> Tensor:
        """State used by the correction encoder; the physical base stays full.

        Existing backbones use the complete state. Experimental compression
        controls can override this hook without filtering their Strang base.
        """
        return u

    def forward(self, u: Tensor, h: float | Tensor, equation, geometry) -> Tensor:
        if geometry.ndim != 2:
            raise ValueError("Spatial neural baselines require two-dimensional geometry")
        check_shape(u, geometry)
        step = broadcast_h(h, u)
        features = extract_features(self._feature_state(u), equation, geometry,
                                    t_ref=self.t_ref, U_ref=self.U_ref)
        features = ((features - self.feature_mean.to(features.dtype)) /
                    self.feature_std.to(features.dtype)).movedim(-1, 1)
        horizon = (step / self.t_ref).expand(u.shape[0], 1, *geometry.grid)
        x = torch.cat((features, horizon), dim=1).to(self.head.weight.dtype)
        raw = self._network(x).to(u.dtype)
        tau = step / self.t_ref
        if self.mode == "hybrid":
            base = split_step(u, step, equation, geometry)
            increment = self.U_ref * tau.pow(3) * raw
        else:
            base = u
            increment = self.U_ref * tau * raw
        return capacity_update(base, increment)

    def architecture_metadata(self) -> dict:
        return {
            "backbone": self.backbone_name,
            "track": self.mode,
            "width": self.width,
            "input_features": 12,
            "horizon_channel": "h/t_ref",
            "normalization": "shared training-only physical feature statistics",
            "physical_split_calls_per_step": 1 if self.mode == "hybrid" else 0,
            "base": "strang_split" if self.mode == "hybrid" else "input_state",
            "increment": "U_ref*(h/t_ref)^3*network" if self.mode == "hybrid"
                         else "U_ref*(h/t_ref)*network",
            "output_form": "base + m*delta/(m+abs(delta)); directional capacity m",
            "output_adaptation": "bounded signed residual; no output clipping",
            "bound_condition": "base in [0,1] and finite increments; physical FFT roundoff is not clipped",
            "initialization": "zero output head",
            "reproduction_claim": "representative adapted backbone, not pretrained SOTA",
        }


class ResidualCNN(_SpatialSolver):
    """Periodic residual CNN with three two-convolution residual blocks."""

    backbone_name = "periodic_residual_cnn"

    def __init__(self, *, mode: str = "direct", ndim: int = 2, width: int = 16,
                 t_ref: float = 1., U_ref: float = 1.):
        super().__init__(mode=mode, ndim=ndim, width=width, t_ref=t_ref, U_ref=U_ref)
        self.lift = nn.Conv2d(13, width, 1)
        self.blocks = nn.Sequential(*(_ResidualBlock(width) for _ in range(3)))
        self.head = nn.Conv2d(width, 1, 1)
        self._initialize_head()

    def _network(self, x: Tensor) -> Tensor:
        return self.head(F.silu(self.blocks(self.lift(x))))

    def architecture_metadata(self) -> dict:
        return {**super().architecture_metadata(), "residual_blocks": 3,
                "spatial_convolutions_per_block": 2, "padding": "periodic"}


class PeriodicUNet(_SpatialSolver):
    """Two-level U-Net with periodic convolutions and average-pooling skips.

    Strided pooling chooses a phase: exact translation equivariance holds for
    shifts divisible by four on grids divisible by four, not arbitrary cell
    shifts. Even grids >=4 are supported; odd intermediate pooled dimensions
    use an unpadded partial average cell and exact-size nearest upsampling.
    """

    backbone_name = "periodic_two_level_unet"

    def __init__(self, *, mode: str = "direct", ndim: int = 2, width: int = 16,
                 t_ref: float = 1., U_ref: float = 1.):
        super().__init__(mode=mode, ndim=ndim, width=width, t_ref=t_ref, U_ref=U_ref)
        self.lift = nn.Conv2d(13, width, 1)
        self.encoder0 = _ResidualBlock(width)
        self.down1 = nn.Conv2d(width, 2 * width, 1)
        self.encoder1 = _ResidualBlock(2 * width)
        self.down2 = nn.Conv2d(2 * width, 4 * width, 1)
        self.bottleneck = _ResidualBlock(4 * width)
        self.up1 = nn.Conv2d(6 * width, 2 * width, 1)
        self.decoder1 = _ResidualBlock(2 * width)
        self.up0 = nn.Conv2d(3 * width, width, 1)
        self.decoder0 = _ResidualBlock(width)
        self.head = nn.Conv2d(width, 1, 1)
        self._initialize_head()

    def _network(self, x: Tensor) -> Tensor:
        if any(n < 4 or n % 2 for n in x.shape[-2:]):
            raise ValueError("Periodic U-Net requires even grid dimensions >=4")
        skip0 = self.encoder0(self.lift(x))
        skip1 = self.encoder1(self.down1(F.avg_pool2d(skip0, 2)))
        center = self.bottleneck(self.down2(F.avg_pool2d(skip1, 2, ceil_mode=True)))
        up1 = F.interpolate(center, size=skip1.shape[-2:], mode="nearest")
        up1 = self.decoder1(self.up1(torch.cat((up1, skip1), dim=1)))
        up0 = F.interpolate(up1, size=skip0.shape[-2:], mode="nearest")
        return self.head(F.silu(self.decoder0(self.up0(torch.cat((up0, skip0), dim=1)))))

    def architecture_metadata(self) -> dict:
        return {**super().architecture_metadata(), "levels": 2,
                "level_widths": [self.width, 2 * self.width, 4 * self.width],
                "padding": "periodic", "downsampling": "average_pool_2",
                "upsampling": "nearest_exact_skip_size",
                "translation_phase": "stride four on grids divisible by four"}


class SpectralConv2d(nn.Module):
    """Two nonoverlapping signed-x Fourier banks with real/imag parameters.

    Real storage makes .to(float64) preserve both complex components. At small
    resolutions each bank is explicitly clipped; inactive slices are not
    represented as measured active capacity. On the default 8/32 grids with
    four modes, both complete signed-x banks are used.
    """

    def __init__(self, inputs: int, outputs: int, modes: int = 4):
        super().__init__()
        if any(type(x) is not int or x <= 0 for x in (inputs, outputs, modes)):
            raise ValueError("Spectral channel counts and modes must be positive integers")
        self.inputs, self.outputs, self.modes = inputs, outputs, modes
        scale = 1 / math.sqrt(inputs * outputs)
        shape = (inputs, outputs, modes, modes)
        self.positive_real = nn.Parameter(scale * torch.randn(shape))
        self.positive_imag = nn.Parameter(scale * torch.randn(shape))
        self.negative_real = nn.Parameter(scale * torch.randn(shape))
        self.negative_imag = nn.Parameter(scale * torch.randn(shape))

    def active_modes(self, grid: tuple[int, int]) -> tuple[int, int, int]:
        nx, ny = grid
        return (min(self.modes, (nx + 1) // 2), min(self.modes, nx // 2),
                min(self.modes, ny // 2 + 1))

    def forward(self, x: Tensor) -> Tensor:
        nx, ny = x.shape[-2:]
        positive, negative, y_modes = self.active_modes((nx, ny))
        spectrum = torch.fft.rfft2(x, norm="ortho")
        transformed = spectrum.new_zeros(x.shape[0], self.outputs, nx, ny // 2 + 1)
        w_positive = torch.complex(self.positive_real, self.positive_imag)
        w_negative = torch.complex(self.negative_real, self.negative_imag)
        transformed[:, :, :positive, :y_modes] = torch.einsum(
            "bixy,ioxy->boxy", spectrum[:, :, :positive, :y_modes],
            w_positive[:, :, :positive, :y_modes])
        if negative:
            transformed[:, :, -negative:, :y_modes] = torch.einsum(
                "bixy,ioxy->boxy", spectrum[:, :, -negative:, :y_modes],
                w_negative[:, :, -negative:, :y_modes])
        return torch.fft.irfft2(transformed, s=(nx, ny), norm="ortho")


class _FourierBlock(nn.Module):
    def __init__(self, width: int, modes: int):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes)
        self.local = nn.Conv2d(width, width, 1)

    def forward(self, x: Tensor) -> Tensor:
        return F.silu(self.spectral(x) + self.local(x))


class FourierNeuralOperator(_SpatialSolver):
    """Four spectral-plus-pointwise layers on the periodic domain."""

    backbone_name = "fourier_neural_operator"

    def __init__(self, *, mode: str = "direct", ndim: int = 2, width: int = 16,
                 t_ref: float = 1., U_ref: float = 1., modes: int = 4):
        super().__init__(mode=mode, ndim=ndim, width=width, t_ref=t_ref, U_ref=U_ref)
        if type(modes) is not int or modes <= 0:
            raise ValueError("Fourier mode count must be a positive integer")
        self.modes = modes
        self.lift = nn.Conv2d(13, width, 1)
        self.blocks = nn.Sequential(*(_FourierBlock(width, modes) for _ in range(4)))
        self.project = nn.Conv2d(width, width, 1)
        self.head = nn.Conv2d(width, 1, 1)
        self._initialize_head()

    def _network(self, x: Tensor) -> Tensor:
        return self.head(F.silu(self.project(self.blocks(self.lift(x)))))

    def architecture_metadata(self) -> dict:
        return {**super().architecture_metadata(), "spectral_layers": 4,
                "requested_modes_per_bank": self.modes,
                "spectral_storage": "separate real and imaginary parameters",
                "signed_x_modes": "positive=min(modes,ceil(nx/2)); negative=min(modes,floor(nx/2))",
                "y_modes": "min(modes,floor(ny/2)+1)",
                "small_grid_behavior": "clip nonoverlapping banks; unused slices inactive",
                "fft_normalization": "ortho", "padding": "none; periodic Fourier transform"}


__all__ = ["ResidualCNN", "PeriodicUNet", "FourierNeuralOperator", "SpectralConv2d"]
