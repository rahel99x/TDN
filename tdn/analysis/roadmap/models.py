"""Finite-step, physically factored experimental neural corrections.

Hybrid source and rank arms share the full-state diffusion-first (DF) base.
The explicit order-control arm uses stability-subdivided RK4; direct FNO is a
neural-only residual map with no physical subflow. These are project controls,
not reproductions of a published FNO benchmark. The rank convention counts
symmetric quadrature pairs, not tensor rank.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from tdn.numerics.operators import broadcast_h, check_shape, laplacian
from tdn.numerics.subflows import diffusion_step
from tdn.research.agenda_neural import centered_state, physical_base
from tdn.research.premix_neural import SymmetricSpectralMixer, project_modes
from tdn.research.reaction import capacity_update


FAMILIES = ("source", "rank1", "rank2", "source_postcompression", "source_trust",
            "source_consistency", "source_df_loss", "source_df_loss_shared", "source_multiband", "cheap_fno",
            "deep_fno", "direct_fno", "c1_rank0", "c1_rank1", "c1_rank2", "c2_rank2", "source_embedded", "df_base", "df_quad2")
MECHANISMS = {"source": ["M04"], "rank1": ["M03"], "rank2": ["M03"],
    "source_postcompression": ["M04"], "source_trust": ["M06"],
    "source_consistency": ["M10"], "source_df_loss": ["M11"],
    "source_multiband": ["M13"], "cheap_fno": ["M16"], "deep_fno": ["M16"],
    "c1_rank1": ["M02", "M03", "M04", "M11"],
    "c1_rank2": ["M02", "M03", "M04", "M11"],
    "df_base": ["M00", "M16"], "df_quad2": ["M02", "M16"]}
MECHANISMS["source_embedded"] = ["M20"]
MECHANISMS["source_df_loss_shared"] = ["M11"]
MECHANISMS["c2_rank2"] = ["M02", "M03", "M04", "M06", "M10", "M11"]
MECHANISMS["direct_fno"] = ["M16"]
MECHANISMS["c1_rank0"] = ["M02", "M03", "M04", "M11"]


def physical_frequencies(u, geometry):
    """Squared angular frequency in inverse physical length squared."""
    result = torch.zeros(geometry.grid, dtype=u.dtype, device=u.device)
    for axis, (n, dx) in enumerate(zip(geometry.grid, geometry.dx)):
        xi = 2 * torch.pi * torch.fft.fftfreq(n, d=dx, device=u.device, dtype=u.dtype)
        shape = [1] * geometry.ndim
        shape[axis] = n
        result = result + xi.square().reshape(shape)
    return result


def fractional_feature(u, geometry, beta, *, length_scale=1.):
    """Dimensionless L^beta (-Delta)^(beta/2)u with a zero constant mode.

    Actual geometry spacing defines the symbol; no learned grid spacing is
    substituted. The declared fixed physical length makes fractional units
    comparable as beta changes.
    """
    if not math.isfinite(float(length_scale)) or length_scale <= 0:
        raise ValueError("Fractional feature length must be positive")
    xi2 = physical_frequencies(u, geometry)
    safe = torch.where(xi2 > 0, xi2, torch.ones_like(xi2))
    symbol = torch.where(xi2 > 0, (length_scale**2 * safe).pow(beta / 2), torch.zeros_like(xi2))
    axes = tuple(range(2, u.ndim))
    return torch.fft.ifftn(torch.fft.fftn(centered_state(u), dim=axes) * symbol, dim=axes).real


def _source_dictionary(u, equation, geometry):
    """Signed degree-two/three source channels with units state/time^3.

    Squared neighbor differences compute the exact nodal commutator without
    subtractive cancellation on constants. Subsequent sources retain signs.
    """
    c = torch.zeros_like(u)
    for axis, dx in enumerate(geometry.dx, start=2):
        c = c + equation.kappa * equation.reaction_rate / dx**2 * (
            (torch.roll(u, 1, axis) - u).square() + (torch.roll(u, -1, axis) - u).square())
    return torch.cat((equation.kappa * laplacian(c, geometry),
                      equation.reaction_rate * (1 - 2 * u) * c), dim=1)


def _active_rate(u, equation, geometry):
    v = centered_state(u)
    variance = v.square().mean((-2, -1), keepdim=True)
    energy = -(v * (equation.kappa * laplacian(v, geometry))).mean((-2, -1), keepdim=True)
    return torch.where(variance > 0, energy / variance.clamp_min(torch.finfo(u.dtype).tiny),
                       torch.zeros_like(variance))


class FiniteStepRank(nn.Module):
    """R learned symmetric pairs of exact zero-subtracted physical factors.

    A lone midpoint contributes identically zero. Each learned direction has
    two nonmidpoint nodes symmetric about 1/2, preserving O(h^3) at fixed
    operator as well as exact h/r/kappa/constant nulls. The coefficients may
    have either sign; this is a learned remainder, not positive quadrature.
    """
    def __init__(self, rank):
        super().__init__()
        if rank not in (1, 2):
            raise ValueError("The bounded learned rank plan is 1 or 2")
        self.rank = rank
        self.node_logits = nn.Parameter(torch.linspace(-.6, .6, rank))
        self.amplitudes = nn.Parameter(torch.zeros(rank))
        self.conditions = nn.Sequential(nn.Linear(3, 8), nn.SiLU(), nn.Linear(8, rank))

    def forward(self, u, h, equation, geometry):
        from .numerics import quadratic_df_defect
        step = broadcast_h(h, u)
        c = u.mean((-2, -1), keepdim=True)
        rate = _active_rate(u, equation, geometry)
        conditions = torch.cat((c, torch.log1p(step.expand_as(c) * equation.reaction_rate),
                                torch.log1p(step.expand_as(c) * rate)), 1).flatten(1)
        gains = 1 + .5 * torch.tanh(self.conditions(conditions.to(self.amplitudes.dtype)))
        positions = .05 + .4 * torch.sigmoid(self.node_logits)
        nodes, weights = [], []
        for index in range(self.rank):
            amplitude = (torch.tanh(self.amplitudes[index]) * gains[:, index]).to(u.dtype)
            weight = amplitude[:, None, None, None] / (2 * math.sqrt(self.rank))
            nodes.extend((positions[index], 1 - positions[index]))
            weights.extend((weight, weight))
        # One shared midpoint response across ranks is algebraically identical
        # to separate calls and saves four transforms for rank two.
        return quadratic_df_defect(u, step, equation, geometry,
            node_positions=torch.stack(nodes).to(u.dtype), node_weights=torch.stack(weights))


class _SourceBlock(nn.Module):
    def __init__(self, width, modes, local):
        super().__init__()
        self.spectral = SymmetricSpectralMixer(width, modes)
        self.local = nn.Conv2d(width, width, 1, bias=False) if local else None

    def forward(self, u):
        value = self.spectral(u)
        return F.silu(value + self.local(u)) if self.local is not None else value


class RoadmapSolver(nn.Module):
    def __init__(self, family, *, width=8, modes=3, trust_horizon=.15,
                 trust_stiffness=8., fractional_length=1., t_ref=1.):
        super().__init__()
        if family not in FAMILIES or width < 1 or modes < 1:
            raise ValueError("Unknown roadmap family or invalid architecture size")
        if trust_horizon <= 0 or trust_stiffness <= 0:
            raise ValueError("Trust support must be positive")
        self.family, self.width, self.modes = family, int(width), int(modes)
        self.requested_width = int(width)
        locations = (2 * self.modes + 1) * (self.modes + 1)
        self.parameter_target = 2 * locations * self.requested_width**2 + 3 * self.requested_width
        if family == "deep_fno":
            self.width = min(range(1, 2 * self.requested_width + 1),
                key=lambda candidate: (abs(4 * (2 * locations + 1) * candidate**2 + 3 * candidate - self.parameter_target), candidate))
        self.trust_horizon, self.trust_stiffness = float(trust_horizon), float(trust_stiffness)
        self.fractional_length = float(fractional_length)
        if not math.isfinite(float(t_ref)) or t_ref <= 0:
            raise ValueError("The physical reference time must be positive")
        self.t_ref = float(t_ref)
        self.direct = family == "direct_fno"
        self.rank = int(family[-1]) if family.startswith(("rank", "c1_rank", "c2_rank")) else 0
        self.is_c1 = family.startswith(("c1_", "c2_"))
        self.has_source = (self.rank == 0 and not family.startswith("df_") or self.is_c1) and not self.direct
        self.depth = 4 if family in ("deep_fno", "direct_fno") else 1
        self.local_fno = family in ("cheap_fno", "deep_fno", "direct_fno")
        if self.has_source:
            self.multiband = family == "source_multiband"
            self.lift = nn.Conv2d(4 if self.multiband else 2, self.width, 1, bias=False)
            self.blocks = nn.Sequential(*(_SourceBlock(self.width, self.modes, self.local_fno)
                                          for _ in range(self.depth)))
            self.head = nn.Conv2d(self.width, 1, 1, bias=False)
            nn.init.zeros_(self.head.weight)
            if self.multiband:
                self.beta_logit = nn.Parameter(torch.zeros(()))
        if self.direct:
            self.lift = nn.Conv2d(4, self.width, 1)
            self.blocks = nn.Sequential(*(_SourceBlock(self.width, self.modes, True) for _ in range(self.depth)))
            self.head = nn.Conv2d(self.width, 1, 1)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        if self.rank:
            # Construct the common source first so a paired random seed gives
            # C1 ranks 0/1/2 exactly identical source initialization.
            self.pair = FiniteStepRank(self.rank)
        self.register_buffer("trust_scale", torch.tensor(1.))

    def source_increment(self, u, step, equation, geometry):
        # Input compression applies to ALL learned inputs and source factors.
        # The physical DF base remains common to both pure ablation arms.
        state = project_modes(u, self.modes) if self.family == "source_postcompression" else u
        source = _source_dictionary(state, equation, geometry)
        if self.multiband:
            beta = .25 + 1.75 * torch.sigmoid(self.beta_logit)
            frac = fractional_feature(state, geometry, beta, length_scale=self.fractional_length)
            low = project_modes(frac, self.modes)
            # Source multiplier preserves constant/r/kappa nulls, while the
            # two factors expose low and high physical derivative bands.
            scale = source.abs().mean(1, keepdim=True)
            source = torch.cat((source, scale * torch.tanh(low), scale * torch.tanh(frac - low)), 1)
        # Source channels have inverse-time-cubed units; nonlinear feature
        # maps consume dimensionless values in the declared physical time unit.
        raw = self.head(self.blocks(self.lift((source * self.t_ref**3).to(self.head.weight.dtype)))).to(u.dtype)
        increment = (step / self.t_ref).pow(5) * torch.tanh(raw) if self.family == "source_embedded" else (step / self.t_ref).pow(3) * raw
        if self.family == "source_trust":
            increment = increment * self.trust_envelope(u, step, equation, geometry)
        return increment

    def trust_envelope(self, u, step, equation, geometry):
        rate = _active_rate(u, equation, geometry)
        mean = u.mean((-2, -1), keepdim=True)
        z = torch.exp(-step * equation.reaction_rate)
        denominator = mean + (1 - mean) * z
        q = (z / denominator.clamp_min(torch.finfo(u.dtype).tiny)) / denominator.clamp_min(torch.finfo(u.dtype).tiny)
        support_h = F.relu(step / (self.trust_horizon * self.trust_scale) - 1)
        support_stiffness = step * rate / (self.trust_stiffness * self.trust_scale)
        return q / (1 + support_h.square() + support_stiffness.pow(3))

    def correction_components(self, u, h, equation, geometry):
        check_shape(u, geometry)
        if geometry.ndim != 2:
            raise ValueError("Roadmap trained arms require a two-dimensional periodic grid")
        step = broadcast_h(h, u)
        if self.direct:
            base = u
        elif self.family == "source_embedded":
            from .numerics import rk4_step
            # Explicit-RK stability work is charged, not hidden as an exact
            # stiff integrator. The integer subdivision is fixed during each
            # differentiable call; generator/order claims concern h -> 0.
            rate_bound = 4 * equation.kappa * sum(dx**-2 for dx in geometry.dx) + equation.reaction_rate
            count = max(1, math.ceil(float(step.detach().max()) * rate_bound))
            base = u
            for _ in range(count):
                base = rk4_step(base, step / count, equation, geometry)
        else:
            base = physical_base(u, step, equation, geometry, "diffusion-first")
        increment = torch.zeros_like(u)
        if self.direct:
            # A neural-only residual operator: no physical subflow or source.
            # It has only the exact zero-time identity by construction.
            ones = torch.ones_like(u)
            features = torch.cat((u, step * ones, equation.kappa * ones,
                                  equation.reaction_rate * ones), 1)
            increment = step * self.head(self.blocks(self.lift(features.to(self.head.weight.dtype)))).to(u.dtype)
        if self.has_source:
            increment = increment + self.source_increment(u, step, equation, geometry)
        if self.rank:
            increment = increment + self.pair(u, step, equation, geometry)
        if self.is_c1 or self.family == "df_quad2":
            from .numerics import quadratic_df_defect
            increment = increment + quadratic_df_defect(u, step, equation, geometry, nodes=2)
        if self.family == "c2_rank2":
            increment = increment * self.trust_envelope(u, step, equation, geometry)
        return {"base": base, "increment": increment}

    def forward(self, u, h, equation, geometry):
        values = self.correction_components(u, h, equation, geometry)
        return self.apply_increment(values)

    def apply_increment(self, values):
        return capacity_update(values["base"], values["increment"])

    def architecture_metadata(self):
        return {"family": self.family, "mechanism_ids": MECHANISMS[self.family],
            "combination_ids": ["C1", "C2"] if self.family == "c2_rank2" else ["C1"] if self.is_c1 else [], "parameters": sum(p.numel() for p in self.parameters()),
            "base_orientation": "none-neural-only" if self.direct else "classical-rk4" if self.family == "source_embedded" else "diffusion-first", "width": self.width, "requested_width": self.requested_width, "modes": self.modes,
            "track": "neural_only" if self.direct else "hybrid_physical_correction",
            "physical_correction_nulls_declared": not self.direct,
            "spectral_depth": self.depth if self.has_source or self.direct else 0,
            "nominal_learned_rank": self.rank, "physical_quadrature_nodes": 2 * self.rank,
            "expanded_separable_term_bound": 2 * self.rank + 1 if self.rank else 0,
            "rank_convention": "symmetric nonmidpoint node pairs; not tensor or Tucker rank",
            "physical_base_transform_count": 0 if self.family == "source_embedded" or self.direct else 4, "source_transform_count": 2 * self.depth if self.has_source or self.direct else 0,
            "rank_transform_count": 4 + 8 * self.rank if self.rank else 0,
            "rk4_rhs_calls": "4*max(1,ceil(h*(4*kappa*sum(dx^-2)+r))); charged in whole-step timing)" if self.family == "source_embedded" else 0,
            "source_projection_transform_count": 2 if self.family == "source_postcompression" else 0,
            "fractional_feature_transform_count": 4 if self.family == "source_multiband" else 0,
            "parameter_matching_claim": "nearest integer width to source stored parameter count" if self.family == "deep_fno" else False,
            "parameter_matching_target": self.parameter_target if self.family == "deep_fno" else None,
            "parameter_matching_relative_error": abs(sum(p.numel() for p in self.parameters()) - self.parameter_target) / self.parameter_target if self.family == "deep_fno" else None,
            "compute_matching_claim": False,
            "learned_input_bypass": False, "physical_base_uses_full_input": not self.direct,
            "fractional_symbol": "(L^2 sum_j (2*pi*k_j/domain_length_j)^2)^(beta/2); DC=0",
            "reference_time": self.t_ref, "source_input_units": "t_ref^3 times state/time^3 signed physical dictionary",
            "embedded_residual": "(h/t_ref)^5*tanh(dimensionless_source_network); capacity-map modification O(h^10) at interior states" if self.family == "source_embedded" else None,
            "trust_calibration_split": "validation", "trust_scale": float(self.trust_scale),
            "output_map": "stability-subdivided RK4 plus capacity-mapped h^5 residual; no unconditional stiff stability claim" if self.family == "source_embedded" else "signed capacity; invariant interval is not an accuracy/stability proof",
            "novelty_claim": False, "published_FNO_reproduction": False}


def build_model(family, config=None):
    config = config or {}
    allowed = ("width", "modes", "trust_horizon", "trust_stiffness", "fractional_length", "t_ref")
    return RoadmapSolver(family, **{k: config[k] for k in allowed if k in config})
