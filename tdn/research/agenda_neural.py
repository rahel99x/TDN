"""Physically vanishing sources and mode-pair kernels for the research agenda.

These are experimental adaptations, not published-model reproductions. Sources
enter before zero-preserving spatial transport. The pair branch is already
quadratic and is never multiplied by the local quadratic gate. All products
use the native periodic collocation grid, including its circular aliasing.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tdn.features.local import extract_features
from tdn.numerics.operators import broadcast_h, check_shape
from tdn.numerics.splitting import split_step
from tdn.numerics.subflows import diffusion_eigenvalues, diffusion_step, reaction_step

from .common import validate_normalization
from .consistency_neural import (
    ConsistencySolver, commutator_gate, discrete_logistic_commutator,
    mean_shape_update,
)
from .neural_baselines import SpectralConv2d
from .premix_neural import PremixSolver, SymmetricSpectralMixer, project_modes
from .reaction import capacity_update, reaction_clock_step


FAMILIES = (
    "local_gate", "local_gate_time", "local_endpoint", "source", "source_time",
    "source_endpoint", "source_closure", "source_time_closure",
    "precompress_source", "fno_source", "fno_source_depth4",
    "fno_source_matched", "fno_anchor", "pair_rank2", "pair_rank4",
    "pair_rank8", "pair_rank4_closure",
)
CONSTRAINED_FAMILIES = FAMILIES
ORIENTATIONS = ("reaction-first", "diffusion-first")


def spatial_mean(value: Tensor) -> Tensor:
    return value.mean(tuple(range(2, value.ndim)), keepdim=True)


def centered_state(u: Tensor) -> Tensor:
    """Mean centering with an exactly zero path for a constant floating field."""
    difference = u - u[..., :1, :1]
    return difference - spatial_mean(difference)


def physical_base(u: Tensor, h: float | Tensor, equation, geometry,
                  orientation: str = "reaction-first") -> Tensor:
    if orientation not in ORIENTATIONS:
        raise ValueError("Unknown Strang orientation")
    if orientation == "reaction-first":
        return split_step(u, h, equation, geometry)
    step = broadcast_h(h, u)
    # A real view of the final complex FFT has a different reduction order
    # from a newly materialized candidate. Contiguous storage keeps the zero
    # mean-head calibration exactly identical to its physical initialization.
    return diffusion_step(reaction_step(diffusion_step(u, step / 2, equation, geometry),
                                        step, equation), step / 2, equation, geometry).contiguous()


def _laplace_channels(value: Tensor, geometry) -> Tensor:
    answer = torch.zeros_like(value)
    for axis, dx in enumerate(geometry.dx, start=2):
        answer = answer + (torch.roll(value, -1, axis) - 2 * value +
                           torch.roll(value, 1, axis)) / dx**2
    return answer


def physical_moments(u: Tensor, equation, geometry) -> dict[str, Tensor]:
    """Exact instantaneous semidiscrete moment identities, not a closure."""
    v = centered_state(u)
    c = spatial_mean(u)
    variance = spatial_mean(v.square())
    third = spatial_mean(v.pow(3))
    energy = equation.kappa * spatial_mean(v * _laplace_channels(v, geometry))
    dc = equation.reaction_rate * (c - c.square() - variance)
    dv = (2 * energy + 2 * equation.reaction_rate * (1 - 2 * c) * variance -
          2 * equation.reaction_rate * third)
    return dict(mean=c, variance=variance, third_centered_moment=third,
                diffusion_energy=energy, mean_derivative=dc, variance_derivative=dv)


class NonlinearTimeDecoder(nn.Module):
    """A genuine SiLU temporal decoder, independently ablated from topology."""
    def __init__(self, outputs: int, hidden: int = 8):
        super().__init__()
        self.first = nn.Linear(4, hidden)
        self.second = nn.Linear(hidden, outputs)

    def forward(self, u, step, equation, geometry, t_ref):
        horizon = step.expand(u.shape[0], 1, 1, 1).flatten(1)
        scale = equation.kappa * sum(dx**-2 for dx in geometry.dx)
        inputs = torch.cat((horizon / t_ref, horizon * equation.reaction_rate,
                            torch.log1p(horizon * scale), spatial_mean(u).flatten(1)), 1)
        return (1 + torch.tanh(self.second(F.silu(self.first(inputs)))))[:, :, None, None]


class _ZeroFourierBlock(nn.Module):
    def __init__(self, width: int, modes: int, *, old_mask: bool = False):
        super().__init__()
        self.spectral = (SpectralConv2d(width, width, modes) if old_mask else
                         SymmetricSpectralMixer(width, modes))
        self.local = nn.Conv2d(width, width, 1, bias=False)

    def forward(self, source):
        return F.silu(self.spectral(source) + self.local(source))


def _source_parameter_count(width: int, modes: int) -> int:
    locations = (2 * modes + 1) * (modes + 1)
    return 14 * width + 2 * (width * width + width) + 2 * width * width * locations + width


def _fno_parameter_count(width: int, modes: int, depth: int) -> int:
    locations = (2 * modes + 1) * (modes + 1)
    return 14 * width + depth * (2 * width * width * locations + width * width) + width * width + width


def parameter_matched_width(width: int, modes: int, *, depth: int = 1) -> int:
    """Nearest total stored parameter count, with a deterministic lower tie."""
    if any(type(value) is not int or value < 1 for value in (width, modes, depth)):
        raise ValueError("Matching width, modes and depth must be positive integers")
    target = _source_parameter_count(width, modes)
    return min(range(1, 4 * width + 1),
               key=lambda candidate: (abs(_fno_parameter_count(candidate, modes, depth) - target), candidate))


class MeanVarianceClosure(nn.Module):
    """Bounded dynamical moment closure with an elapsed-time variance remainder.

    V=z*c*(1-c), c'=r*c*(1-c)*(1-z). For resolved variance and
    nondegenerate 0<z<1, its frozen initial z rate matches the exact
    instantaneous variance identity. Saturated binary moments and unresolved
    variance are explicit limitations of the regularized fraction coordinate.
    The learned bounded coefficient
    changes that rate by elapsed_time*r*diffusion_energy/variance; therefore
    its integrated mean effect starts cubically in time. Paired integration
    subtracts the zero-learned endpoint, preserving the physical initialization.
    This finite-substep frozen-moment closure is an approximation, not an exact
    evolution law or a positivity theorem for unconstrained moment integrators.
    """
    def __init__(self, hidden: int = 8, substeps: int = 4):
        super().__init__()
        self.body = nn.Linear(6, hidden)
        self.head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.substeps = substeps

    def forward(self, u, step, equation, geometry, t_ref):
        terms = physical_moments(u, equation, geometry)
        c, variance, energy, third = (terms[key] for key in
                                      ("mean", "variance", "diffusion_energy", "third_centered_moment"))
        tiny = torch.finfo(u.dtype).eps
        cap = c * (1 - c)
        safe_cap = torch.where(cap > 0, cap, torch.ones_like(cap))
        z = variance / safe_cap
        # Roundoff can make the Bhatia-Davis fraction a few ulps above one;
        # only the derived moment fraction, never the physical state, is bounded.
        z = z.clamp(0., 1.)
        cap_derivative = (1 - 2 * c) * terms["mean_derivative"]
        dz = (terms["variance_derivative"] - z * cap_derivative) / safe_cap
        denominator = z * (1 - z)
        rate = torch.where(denominator > tiny, dz / denominator.clamp_min(tiny), torch.zeros_like(dz))
        energy_rate = torch.where(variance > tiny, 2 * energy / variance.clamp_min(tiny), torch.zeros_like(energy))
        inputs = torch.cat((c, variance, t_ref * energy, third,
                            step.expand_as(c) * equation.reaction_rate,
                            step.expand_as(c) * energy_rate), 1).flatten(1)
        coefficient = torch.tanh(self.head(F.silu(self.body(inputs.to(self.head.weight.dtype)))))
        coefficient = coefficient.to(u.dtype)[:, :, None, None]
        ds = step / self.substeps

        def integrate(amplitude):
            mean, fraction = c, z
            for index in range(self.substeps):
                elapsed = (index + .5) * ds
                current_rate = rate + elapsed * equation.reaction_rate * energy_rate * amplitude
                middle_fraction = reaction_clock_step(fraction, ds * current_rate / 2)
                mean = reaction_clock_step(mean, ds * equation.reaction_rate * (1 - middle_fraction))
                fraction = reaction_clock_step(fraction, ds * current_rate)
            return mean, fraction * mean * (1 - mean)

        baseline_mean, baseline_variance = integrate(torch.zeros_like(coefficient))
        mean, evolved_variance = integrate(coefficient)
        return dict(mean_increment=mean - baseline_mean, closure_mean=mean,
                    closure_variance=evolved_variance, closure_baseline_mean=baseline_mean,
                    closure_baseline_variance=baseline_variance,
                    closure_coefficient=coefficient, **terms)


class EigenPairKernel(nn.Module):
    """Symmetric separable real-even filters acting on phase-bearing fields.

    Each rank forms L(Av*Bv)-Av*L(Bv)-Bv*L(Av), then applies an output
    filter conditioned on h*lambda_m, hr and mean. Symmetry follows from the
    symmetric product; real-even factors preserve phase and real output.
    Native-grid products imply circular k+l=m, including wrapped output modes.
    The discrete commutator factor expands each nominal rank to at most three
    separable terms; the recorded nominal rank is not the tensor's strict rank.
    """
    def __init__(self, rank: int, hidden: int = 8):
        super().__init__()
        if rank not in (2, 4, 8):
            raise ValueError("Declared pair ranks are 2, 4 and 8")
        self.rank = rank
        self.input_factors = nn.Sequential(nn.Linear(3, hidden), nn.SiLU(), nn.Linear(hidden, 2 * rank))
        self.output_factors = nn.Sequential(nn.Linear(3, hidden), nn.SiLU(), nn.Linear(hidden, rank))
        self.amplitude = nn.Parameter(torch.zeros(rank))

    def forward(self, u, step, equation, geometry, t_ref, U_ref):
        v = centered_state(u) / U_ref
        lam = equation.kappa * diffusion_eigenvalues(geometry, u)[..., :geometry.grid[1] // 2 + 1]
        horizon = step.expand(u.shape[0], 1, 1, 1)
        stiffness = -horizon * lam[None, None]
        c = spatial_mean(u)
        conditions = torch.stack((torch.log1p(stiffness[:, 0]),
                                  (horizon * equation.reaction_rate)[:, 0].expand_as(stiffness[:, 0]),
                                  c[:, 0].expand_as(stiffness[:, 0])), -1)
        conditions = conditions.to(self.amplitude.dtype)
        factors = 1 + torch.tanh(self.input_factors(conditions))
        factors = factors.movedim(-1, 1).to(u.dtype)
        output = self.output_factors(conditions).movedim(-1, 1).to(u.dtype)
        spectrum = torch.fft.rfft2(v, norm="ortho")
        a = torch.fft.irfft2(spectrum * factors[:, :self.rank], s=geometry.grid, norm="ortho")
        b = torch.fft.irfft2(spectrum * factors[:, self.rank:], s=geometry.grid, norm="ortho")
        product = a * b
        commutator = equation.kappa * (_laplace_channels(product, geometry) -
                                       a * _laplace_channels(b, geometry) -
                                       b * _laplace_channels(a, geometry))
        transformed = torch.fft.rfft2(commutator, norm="ortho")
        raw = torch.fft.irfft2(transformed * output, s=geometry.grid, norm="ortho")
        raw = (raw * self.amplitude.to(u.dtype)[None, :, None, None]).sum(1, keepdim=True) / math.sqrt(self.rank)
        increment = U_ref * (step / t_ref).pow(3) * t_ref**2 * equation.reaction_rate * raw
        return dict(spatial_increment=increment, raw_spatial=raw, pair_sources=commutator,
                    input_factors=factors, output_factors=output)


class AgendaSolver(nn.Module):
    def __init__(self, family: str, *, width: int = 16, modes: int = 4,
                 t_ref: float = 1., U_ref: float = 1.,
                 base_orientation: str = "reaction-first", depth: int | None = None,
                 rank: int | None = None):
        super().__init__()
        if family not in FAMILIES:
            raise ValueError(f"Unknown agenda family: {family}")
        if any(type(value) is not int or value <= 0 for value in (width, modes)):
            raise ValueError("Width and modes must be positive integers")
        if base_orientation not in ORIENTATIONS:
            raise ValueError("Unknown Strang orientation")
        if not all(math.isfinite(float(value)) and float(value) > 0 for value in (t_ref, U_ref)):
            raise ValueError("Reference scales must be finite and positive")
        self.family, self.requested_width, self.modes = family, width, modes
        self.t_ref, self.U_ref = float(t_ref), float(U_ref)
        self.base_orientation = base_orientation
        self.register_buffer("feature_mean", torch.zeros(12))
        self.register_buffer("feature_std", torch.ones(12))
        self.local = family.startswith("local_")
        self.pair = family.startswith("pair_")
        self.endpoint = family.endswith("endpoint")
        self.closure = family.endswith("closure")
        self.temporal = "time" in family
        self.input_first = family == "precompress_source"
        self.fno = family.startswith("fno_")
        self.depth = 4 if family in ("fno_source_depth4", "fno_source_matched", "fno_anchor") else 1
        if depth is not None and (type(depth) is not int or depth != self.depth):
            raise ValueError("Depth is fixed by the family; use the explicit depth-control family")
        self.width = parameter_matched_width(width, modes, depth=self.depth) if family == "fno_source_matched" else width
        if self.local:
            self.backbone = ConsistencySolver("premix_moment" if self.endpoint else "premix_gated",
                                              width=width, modes=modes, t_ref=t_ref, U_ref=U_ref)
        elif self.pair:
            self.rank = int(family.split("rank", 1)[1].split("_", 1)[0])
            if rank is not None and (type(rank) is not int or rank != self.rank):
                raise ValueError("Pair rank must agree with the declared family")
            self.kernel = EigenPairKernel(self.rank)
        elif self.fno:
            self.lift = nn.Conv2d(13, self.width, 1)
            self.blocks = nn.Sequential(*(_ZeroFourierBlock(self.width, modes, old_mask=family == "fno_anchor")
                                          for _ in range(self.depth)))
            self.project = nn.Conv2d(self.width, self.width, 1, bias=False)
            self.head = nn.Conv2d(self.width, 1, 1, bias=False)
            nn.init.zeros_(self.head.weight)
        else:
            self.backbone = PremixSolver(width=width, modes=modes, input_first=self.input_first,
                                         t_ref=t_ref, U_ref=U_ref)
            # Every operation following the source must map zero to zero.
            self.backbone.head = nn.Conv2d(width, 1, 1, bias=False)
            nn.init.zeros_(self.backbone.head.weight)
        if not self.pair and rank is not None:
            raise ValueError("Rank is only valid for an explicit pair-kernel family")
        if self.temporal:
            self.time_decoder = NonlinearTimeDecoder(1 if self.local else self.width)
        if self.endpoint and not self.local:
            self.mean_head = nn.Conv2d(self.width, 1, 1, bias=False)
            nn.init.zeros_(self.mean_head.weight)
        if self.closure:
            self.mean_closure = MeanVarianceClosure()

    def set_normalization(self, mean, std):
        mean, std = validate_normalization(mean, std, 12, dtype=self.feature_mean.dtype,
                                           device=self.feature_mean.device)
        with torch.no_grad():
            self.feature_mean.copy_(mean)
            self.feature_std.copy_(std)
        if hasattr(self, "backbone"):
            self.backbone.set_normalization(mean, std)
        return self

    def _features(self, u, step, equation, geometry):
        state = project_modes(u, self.modes) if self.input_first else u
        features = extract_features(state, equation, geometry, t_ref=self.t_ref, U_ref=self.U_ref)
        features = ((features - self.feature_mean.to(features.dtype)) /
                    self.feature_std.to(features.dtype)).movedim(-1, 1)
        horizon = (step / self.t_ref).expand(u.shape[0], 1, *geometry.grid)
        dtype = self.lift.weight.dtype if self.fno else self.backbone.head.weight.dtype
        return torch.cat((features, horizon), 1).to(dtype)

    def correction_components(self, u, h, equation, geometry):
        check_shape(u, geometry)
        if geometry.ndim != 2:
            raise ValueError("Agenda neural models require two-dimensional geometry")
        step = broadcast_h(h, u)
        base = physical_base(u, step, equation, geometry, self.base_orientation)
        if self.local:
            latent = self.backbone._latent(self._features(u, step, equation, geometry))
            raw = self.backbone.head(latent).to(u.dtype)
            gate = commutator_gate(u, equation, geometry, t_ref=self.t_ref, U_ref=self.U_ref)
            factor = self.U_ref * (step / self.t_ref).pow(3)
            result = dict(base=base, gate=gate, raw_spatial=raw,
                          spatial_increment=factor * gate * raw)
            if self.endpoint:
                raw_mean = self.backbone.mean_head(spatial_mean(latent)).to(u.dtype)
                mean_increment = factor * spatial_mean(gate) * raw_mean
                result.update(mean_increment=mean_increment,
                              target_mean=capacity_update(spatial_mean(base), mean_increment))
            if self.temporal:
                modulation = self.time_decoder(u, step, equation, geometry, self.t_ref).to(u.dtype)
                result["spatial_increment"] = result["spatial_increment"] * modulation
                result["time_modulation"] = modulation
        elif self.pair:
            result = dict(base=base, **self.kernel(u, step, equation, geometry, self.t_ref, self.U_ref))
        else:
            features = self._features(u, step, equation, geometry)
            amplitudes = self.lift(features) if self.fno else self.backbone.interactions(features)
            source = self.t_ref**2 * discrete_logistic_commutator(u, equation, geometry) / self.U_ref
            weighted_source = source.to(amplitudes.dtype) * amplitudes
            if self.temporal:
                modulation = self.time_decoder(u, step, equation, geometry, self.t_ref)
                weighted_source = weighted_source * modulation
            latent = (F.silu(self.project(self.blocks(weighted_source))) if self.fno else
                      self.backbone.spectral(weighted_source))
            head = self.head if self.fno else self.backbone.head
            raw = head(latent).to(u.dtype)
            factor = self.U_ref * (step / self.t_ref).pow(3)
            result = dict(base=base, source=source, weighted_source=weighted_source,
                          raw_spatial=raw, spatial_increment=factor * raw)
            if self.temporal:
                result["time_modulation"] = modulation.to(u.dtype)
            if self.endpoint:
                mean_increment = factor * self.mean_head(spatial_mean(latent)).to(u.dtype)
                result.update(mean_increment=mean_increment,
                              target_mean=capacity_update(spatial_mean(base), mean_increment))
        if self.closure:
            closure = self.mean_closure(u, step, equation, geometry, self.t_ref)
            result.update(closure)
            result["target_mean"] = capacity_update(spatial_mean(base), closure["mean_increment"])
        return result

    def forward(self, u, h, equation, geometry):
        result = self.correction_components(u, h, equation, geometry)
        if self.endpoint or self.closure:
            return mean_shape_update(result["base"], result["mean_increment"], result["spatial_increment"])
        return capacity_update(result["base"], result["spatial_increment"])

    step = forward

    def architecture_metadata(self):
        count = sum(parameter.numel() for parameter in self.parameters())
        locations = (2 * self.modes + 1) * (self.modes + 1)
        metadata = dict(
            family=self.family, track="hybrid", parameters=count,
            requested_width=self.requested_width, width=self.width, cutoff=self.modes,
            base_orientation=self.base_orientation,
            physical_base_state="full unprojected input; no encoder compression",
            physical_split_calls_per_step=1,
            physical_base_transform_count=2 if self.base_orientation == "reaction-first" else 4,
            exact_correction_limits=["constant input", "reaction_rate=0", "kappa=0", "h=0"],
            initialization="zero spatial amplitude/output head and zero mean head; same physical base",
            physical_output="bounded signed-capacity map, optional final mean calibration; no state clipping",
            physical_output_bandlimited=False,
            raw_output_bandlimited=not self.fno and not self.pair,
            product_convention="native periodic collocation-grid products; circular mode addition, no padding/dealiasing",
            normalization="same full-training-state statistics for all arms",
            nonlinear_time_decoder=self.temporal,
            time_decoder="SiLU MLP(h/t_ref,hr,log1p(h*kappa*sum(dx^-2)),mean), independent ablation" if self.temporal else None,
            spectral_layers=self.depth,
            fourier_support="inclusive |kx|<=K and |ky|<=K, naturally clipped to grid",
            requested_fourier_locations_per_channel_pair=locations,
            correction_encoder_state="P_K(u)" if self.input_first else "full u",
            physical_source_state="full u even in precompression control",
            information_bypasses=["full-state Strang base", "full-state physical commutator source"] if not self.pair else
                                 ["full-state Strang base", "full-state centered mode-pair branch"],
            full_state_source_bypass=self.input_first,
            gate_placement="source before zero-preserving nonlocal transport" if not self.local else "historical local post gate",
            output_mean_treatment="dynamical frozen-moment variance closure" if self.closure else
                                  "existing learned endpoint mean head" if self.endpoint else "capacity-map generated mean",
            novelty_claim=False,
        )
        if self.local:
            metadata.update(backbone="historical consistency premix", correction_transform_count=2,
                            temporal_unmodulated_raw_degree_at_most=2,
                            information_bypasses=["full-state Strang base", "full-state local output gate"])
        elif self.pair:
            metadata.update(backbone="symmetric eigenvalue-conditioned separable mode-pair kernel",
                            nominal_pair_rank=self.rank, algebraic_separable_term_bound=3 * self.rank,
                            model_rank_convention="R symmetric input-factor pairs times discrete commutator; at most 3R separable terms",
                            offline_rank_scope="R-node quadratic-response quadrature plus two exact Strang terms is an exploration gate, not a representability guarantee for this parameterized kernel",
                            pair_inputs="h*lambda_k,h*lambda_l,h*lambda_m,hr,mean; actual discrete diffusion eigenvalues",
                            pair_symmetry="A_k B_l + A_l B_k via same-grid field product",
                            phase="complex input phase retained; real-even learned scalar filters",
                            pair_zero_limits="centered fields and r*(L(AB)-A LB-B LA); no quadratic output gate",
                            correction_forward_transform_count=1 + self.rank,
                            correction_inverse_transform_count=3 * self.rank,
                            correction_transform_count=1 + 4 * self.rank,
                            local_laplacian_applications=3 * self.rank,
                            spectral_layers=None,
                            fourier_support="all input/output discrete modes; learned even filters",
                            requested_fourier_locations_per_channel_pair=None)
        else:
            metadata.update(backbone="zero-preserving FNO source transport" if self.fno else "premix source transport",
                            source="t_ref^2*C/U_ref; C=r*sum(kappa/dx_d^2)*(neighbor-u)^2",
                            increment="U_ref*(h/t_ref)^3*head(K(source*amplitudes))",
                            correction_transform_count=2 * self.depth,
                            operations_after_source="bias-free spectral/local maps, zero-preserving SiLU, bias-free head",
                            high_frequency_local_branch=self.fno)
            if self.input_first:
                metadata["correction_transform_count"] += 2
            if self.family == "fno_anchor":
                metadata.update(fourier_support="historical banks kx=0..K-1,-K..-1; ky=0..K-1",
                                requested_fourier_locations_per_channel_pair=2 * self.modes**2,
                                matching="larger historical four-layer anchor; not support matched")
            elif self.family == "fno_source_matched":
                target = _source_parameter_count(self.requested_width, self.modes)
                metadata.update(parameter_matching_target=target,
                                parameter_matching_relative_error=abs(count - target) / target,
                                parameter_matching="nearest stored count over integer widths 1..4*requested_width; no padding parameters")
        if self.closure:
            metadata.update(closure_substeps=self.mean_closure.substeps,
                            closure_mean_law="c'=r*(c-c^2-V), V=z*c*(1-c)",
                            closure_variance_law="initial exact V'=2<E(v,Lv)>+2r(1-2c)V-2r<E(v^3)> for resolved nondegenerate moment fraction; frozen rate thereafter",
                            closure_moment_coordinate_limitation="z=0/1 or unresolved variance uses regularized zero fraction rate; binary-field diffusion variance evolution is not represented",
                            closure_learned_remainder="elapsed_time*r*(2 diffusion_energy/variance)*bounded scalar coefficient",
                            closure_endpoint="paired learned-minus-zero-coefficient mean; bounded target relative to base",
                            closure_scope="bounded approximate dynamical closure, not exact future variance or mean")
        return metadata


def build_model(family: str, *, width: int = 16, modes: int = 4,
                normalization=None, t_ref: float = 1., U_ref: float = 1.,
                base_orientation: str = "reaction-first", depth: int | None = None,
                rank: int | None = None):
    model = AgendaSolver(family, width=width, modes=modes, t_ref=t_ref, U_ref=U_ref,
                         base_orientation=base_orientation, depth=depth, rank=rank)
    if normalization is not None:
        if not isinstance(normalization, (tuple, list)) or len(normalization) != 2:
            raise ValueError("Normalization must be a (mean, standard deviation) pair")
        model.set_normalization(*normalization)
    return model


__all__ = ["FAMILIES", "CONSTRAINED_FAMILIES", "ORIENTATIONS", "AgendaSolver", "build_model",
           "physical_base", "physical_moments", "centered_state", "EigenPairKernel",
           "MeanVarianceClosure", "NonlinearTimeDecoder", "parameter_matched_width"]
