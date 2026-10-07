"""Learned spatial corrections that mix before discarding Fourier modes.

These are bounded research candidates, not a reproduction or improvement claim
for the FNO paper. All five families use the existing full-state Strang base,
physical features, cubic horizon factor and signed-capacity output map.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .neural_baselines import _SpatialSolver, FourierNeuralOperator, ResidualCNN


FAMILIES = ("premix", "premix_local", "precompress", "fno", "cnn")


def _cutoff(value: int) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("Fourier cutoff must be a positive integer")
    return value


def _signed_frequencies(size: int, device) -> Tensor:
    indices = torch.arange(size, device=device)
    # The even-grid Nyquist frequency has exactly one representation.
    return torch.where(indices <= (size - 1) // 2, indices, indices - size)


def project_modes(state: Tensor, modes: int) -> Tensor:
    """Real orthogonal box projection with inclusive symmetric cutoff K.

    Products are formed on the original periodic collocation grid. This does
    not add padding/dealiasing or assert continuum-product accuracy.
    """
    _cutoff(modes)
    if state.ndim != 4 or not state.is_floating_point():
        raise ValueError("Fourier projection expects floating [batch,channels,x,y]")
    nx, ny = state.shape[-2:]
    x = _signed_frequencies(nx, state.device)
    y = torch.arange(ny // 2 + 1, device=state.device)
    mask = (x.abs()[:, None] <= modes) & (y[None, :] <= modes)
    spectrum = torch.fft.rfft2(state, norm="ortho")
    return torch.fft.irfft2(spectrum * mask, s=(nx, ny), norm="ortho")


class SymmetricSpectralMixer(nn.Module):
    """Learned channel mixing on both signed-x banks of a symmetric box.

    Real/imaginary parameter storage preserves both parts under .double().
    ``irfft2`` applies the real-field constraint on self-conjugate lines. These
    lines therefore contain redundant parameters, as in the adapted FNO bank;
    the recorded parameter count is stored capacity, not independent DOFs.
    """

    def __init__(self, width: int, modes: int):
        super().__init__()
        if type(width) is not int or width < 1:
            raise ValueError("Spectral width must be a positive integer")
        self.width, self.modes = width, _cutoff(modes)
        shape = (width, width, 2 * modes + 1, modes + 1)
        scale = 1. / math.sqrt(width)
        self.weight_real = nn.Parameter(scale * torch.randn(shape))
        self.weight_imag = nn.Parameter(scale * torch.randn(shape))

    def forward(self, state: Tensor) -> Tensor:
        nx, ny = state.shape[-2:]
        frequencies = _signed_frequencies(nx, state.device)
        indices = torch.nonzero(frequencies.abs() <= self.modes, as_tuple=False).flatten()
        y_count = min(self.modes + 1, ny // 2 + 1)
        spectrum = torch.fft.rfft2(state, norm="ortho")
        weights = torch.complex(self.weight_real, self.weight_imag)
        weights = weights[:, :, frequencies[indices] + self.modes, :y_count]
        result = spectrum.new_zeros(state.shape[0], self.width, nx, ny // 2 + 1)
        result[:, :, indices, :y_count] = torch.einsum(
            "bixy,ioxy->boxy", spectrum[:, :, indices, :y_count], weights)
        return torch.fft.irfft2(result, s=(nx, ny), norm="ortho")


class PremixSolver(_SpatialSolver):
    """Pointwise quadratic interactions followed by learned output filtering.

    z = lift(features, h); q = z + A(z) B(z) / sqrt(width).
    The main raw correction is head(S_K(q)). The optional local branch adds a
    separately learned head(q - P_K(q)), explicitly retaining high output modes.
    Filtering the *input state* is the paired compression-order ablation.
    """

    backbone_name = "quadratic_premix_spectral_correction"

    def __init__(self, *, width: int = 16, modes: int = 4,
                 input_first: bool = False, local: bool = False,
                 t_ref: float = 1., U_ref: float = 1.):
        super().__init__(mode="hybrid", ndim=2, width=width, t_ref=t_ref, U_ref=U_ref)
        if type(input_first) is not bool or type(local) is not bool:
            raise ValueError("Compression ablation switches must be boolean")
        if input_first and local:
            raise ValueError("Input-first plus local is outside the declared family plan")
        self.modes = _cutoff(modes)
        self.input_first, self.local = input_first, local
        self.lift = nn.Conv2d(13, width, 1)
        self.factor_a = nn.Conv2d(width, width, 1)
        self.factor_b = nn.Conv2d(width, width, 1)
        self.spectral = SymmetricSpectralMixer(width, modes)
        self.head = nn.Conv2d(width, 1, 1)
        self._initialize_head()
        if local:
            # A bias would duplicate the low-band head's DC parameter.
            self.local_head = nn.Conv2d(width, 1, 1, bias=False)
            nn.init.zeros_(self.local_head.weight)

    def _feature_state(self, u: Tensor) -> Tensor:
        # Never filter the full-state Strang base. Projection can overshoot the
        # physical interval; these encoder inputs are deliberately not clipped.
        return project_modes(u, self.modes) if self.input_first else u

    def interactions(self, features: Tensor) -> Tensor:
        latent = self.lift(features)
        return latent + self.factor_a(latent) * self.factor_b(latent) / math.sqrt(self.width)

    def _network(self, features: Tensor) -> Tensor:
        mixed = self.interactions(features)
        raw = self.head(self.spectral(mixed))
        if self.local:
            raw = raw + self.local_head(mixed - project_modes(mixed, self.modes))
        return raw

    def architecture_metadata(self) -> dict:
        return {
            **super().architecture_metadata(),
            "family": "precompress" if self.input_first else "premix_local" if self.local else "premix",
            "cutoff": self.modes,
            "fourier_support": "inclusive |kx|<=K and |ky|<=K, naturally clipped to the grid",
            "requested_fourier_locations_per_channel_pair": (2 * self.modes + 1) * (self.modes + 1),
            "spectral_layers": 1,
            "correction_encoder_state": "P_K(u)" if self.input_first else "full u",
            "physical_base_state": "full u for every family",
            "feature_extraction_order": "after input projection" if self.input_first else "before output projection",
            "interaction": "z + A(z)*B(z)/sqrt(width), z=pointwise_lift(features,h)",
            "main_raw_correction": "pointwise_head(S_K(interaction)); symmetric learned Fourier box",
            "local_output_bypass": "separate_head(q-P_K(q))" if self.local else None,
            "raw_output_bandlimited": not self.local,
            "physical_output_bandlimited": False,
            "product_convention": "native collocation-grid product; no padding or dealiasing",
            "normalization": "same statistics fitted on full training fields for every family",
            "compression_order_pair": "premix/precompress have identical learned parameter names and shapes",
            "spectral_parameter_count": "stored real/imag values; redundant self-conjugate-line degrees retained",
            "claim_scope": "experimental inductive bias; nonlinear mixing is also available in FNO",
            "capacity_comparison": "only premix/precompress match learned capacity; FNO and CNN have different parameters and FLOPs",
        }


class _FNOControl(FourierNeuralOperator):
    def architecture_metadata(self) -> dict:
        return {
            **super().architecture_metadata(),
            "family": "fno",
            "requested_fourier_locations_per_channel_pair_per_layer": 2 * self.modes * self.modes,
            "fourier_support": "kx=0..K-1 and -K..-1; ky=0..K-1, clipped to nonoverlapping banks",
            "capacity_comparison": "same width and training budget; Fourier support, parameters and FLOPs differ from premix",
            "interaction_scope": "four full-grid local branches and nonlinear activations already permit nonlinear interactions",
        }


def build_model(family: str, *, width: int = 16, modes: int = 4,
                normalization=None, t_ref: float = 1., U_ref: float = 1.) -> nn.Module:
    """Construct one declared hybrid candidate; never substitute a family."""
    _cutoff(modes)
    common = dict(width=width, t_ref=t_ref, U_ref=U_ref)
    if family in ("premix", "premix_local", "precompress"):
        model = PremixSolver(modes=modes, input_first=family == "precompress",
                             local=family == "premix_local", **common)
    elif family == "fno":
        model = _FNOControl(mode="hybrid", modes=modes, **common)
    elif family == "cnn":
        model = ResidualCNN(mode="hybrid", **common)
    else:
        raise ValueError(f"Unsupported premix neural family: {family}")
    if normalization is not None:
        if not isinstance(normalization, (tuple, list)) or len(normalization) != 2:
            raise ValueError("Normalization must be a (mean, standard deviation) pair")
        model.set_normalization(*normalization)
    return model


__all__ = ["FAMILIES", "PremixSolver", "SymmetricSpectralMixer", "build_model", "project_modes"]
