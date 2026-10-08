"""Focused rank-one interaction hypothesis and competitive declared controls.

This is an experimental architecture study, not a published FNO reproduction.
All hybrid models share a track-correct full-state diffusion-first core. The
learned rank branch forms signed physical products before output compression;
its pure ablation compresses the input first and has no correction bypass.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from tdn.numerics.operators import broadcast_h, check_shape
from tdn.analysis.roadmap.numerics import quadratic_df_defect, cubic_df_defect, lowpass
from tdn.research.premix_neural import SymmetricSpectralMixer
from . import numerics

FAMILIES = ("rank1", "rank1_postcompression", "rank1_frozen", "analytic_quad",
            "analytic_quad_cubic", "fno_small", "fno_standard", "direct_fno", "df", "etdrk4")
TRAINABLE_FAMILIES = ("rank1", "rank1_postcompression", "fno_small", "fno_standard", "direct_fno")
RANK_FAMILIES = ("rank1", "rank1_postcompression", "rank1_frozen")


class PhaseRankOne(nn.Module):
    """One signed symmetric node pair, with 43 stored real parameters.

    Heat transport retains complex Fourier phase before real quadratic
    multiplication. The zero-subtracted pair vanishes for constants, zero
    diffusion and zero reaction independently of arbitrary learned weights.
    The pair is quadratic in centered amplitude at fixed mean/active rate.
    """
    def __init__(self, modes, *, initial_amplitude=.5):
        super().__init__()
        self.modes = modes
        if not math.isfinite(initial_amplitude) or not -2 < initial_amplitude < 2:
            raise ValueError("Rank initialization amplitude must be between -2 and 2")
        position = .5 - 1 / (2 * math.sqrt(3))
        scaled = (position - .05) / .4
        self.node_logit = nn.Parameter(torch.tensor(math.log(scaled / (1 - scaled))))
        self.amplitude = nn.Parameter(torch.tensor(math.atanh(initial_amplitude / 2)))
        self.conditioner = nn.Sequential(nn.Linear(3, 8), nn.SiLU(), nn.Linear(8, 1))
        nn.init.zeros_(self.conditioner[-1].weight)
        nn.init.zeros_(self.conditioner[-1].bias)

    def forward(self, u, step, equation, geometry, track):
        axes = tuple(range(2, u.ndim))
        center = u.mean(axes, keepdim=True)
        v = u - center
        variance = v.square().mean(axes, keepdim=True)
        energy = -(v * numerics.apply_multiplier(v, numerics.diffusion_symbol(u, equation, geometry, track))).mean(axes, keepdim=True)
        active = torch.where(variance > 0, energy / variance.clamp_min(torch.finfo(u.dtype).tiny), torch.zeros_like(variance)).clamp_min(0)
        features = torch.cat((center, torch.log1p(step * equation.reaction_rate).expand_as(center),
                              torch.log1p(step * active)), 1).flatten(1)
        gain = 1 + .5 * torch.tanh(self.conditioner(features.to(self.amplitude.dtype))).to(u.dtype)
        position = .05 + .4 * torch.sigmoid(self.node_logit)
        weight = (2 * torch.tanh(self.amplitude) * gain).reshape(u.shape[0], 1, 1, 1) / 2
        increment = quadratic_df_defect(u, step, equation, geometry, target=track,
            node_positions=torch.stack((position, 1 - position)).to(u.dtype),
            node_weights=torch.stack((weight, weight)))
        # Compression is on the output only. Complex phase products and all
        # high-high-to-low interactions have already been formed above.
        increment = lowpass(increment, self.modes)
        constant = (u == u[..., :1, :1]).flatten(2).all(-1)[..., None, None]
        return torch.where(constant | (step == 0), torch.zeros_like(increment), increment)


class FNOBlock(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.spectral = SymmetricSpectralMixer(width, modes)
        # This full-grid local path must not be removed to manufacture an FNO
        # high-mode weakness. Nonlinearity can create retained-mode interactions.
        self.local = nn.Conv2d(width, width, 1)

    def forward(self, value):
        return F.gelu(self.spectral(value) + self.local(value))


class FrontierModel(nn.Module):
    def __init__(self, family, track, *, width=None, modes=4, depth=None, t_ref=.1,
                 reaction_substeps=4, quad_nodes=4, cubic_nodes=4,
                 initial_amplitude=.5, cache_entries=32):
        super().__init__()
        if family not in FAMILIES or track not in numerics.TRACKS:
            raise ValueError("Unknown frontier model family or target track")
        self.family, self.track = family, track
        self.width = (12 if family == "fno_small" else 24) if width is None else width
        self.depth = (2 if family == "fno_small" else 4) if depth is None else depth
        if any(type(x) is not int or x < 1 for x in (self.width, self.depth, modes, reaction_substeps, quad_nodes, cubic_nodes)):
            raise ValueError("Model dimensions, modes, quadrature and subdivisions must be positive integers")
        if not math.isfinite(t_ref) or t_ref <= 0:
            raise ValueError("Reference time must be finite and positive")
        self.modes, self.t_ref = modes, float(t_ref)
        self.reaction_substeps, self.quad_nodes, self.cubic_nodes = reaction_substeps, quad_nodes, cubic_nodes
        self.cache = numerics.CoefficientCache(cache_entries)
        if family in RANK_FAMILIES:
            self.pair = PhaseRankOne(modes, initial_amplitude=initial_amplitude)
        if family in ("fno_small", "fno_standard", "direct_fno"):
            self.lift = nn.Conv2d(8, self.width, 1)
            self.blocks = nn.Sequential(*(FNOBlock(self.width, modes) for _ in range(self.depth)))
            self.head = nn.Sequential(nn.Conv2d(self.width, self.width, 1), nn.GELU(), nn.Conv2d(self.width, 1, 1))
            # A nonzero head makes every layer trainable at initialization.
            # Its small scale is declared, not selected using confirmation.
            nn.init.normal_(self.head[-1].weight, std=.02)
            nn.init.zeros_(self.head[-1].bias)
        if family == "rank1_frozen":
            self.requires_grad_(False)

    def _apply(self, fn, recurse=True):
        # nn.Module.to() does not transform our unregistered coefficient cache.
        # Clearing avoids retaining old device allocations or stale precision.
        self.clear_cache()
        return super()._apply(fn, recurse=recurse)

    def clear_cache(self):
        self.cache.clear()

    @property
    def cache_metadata(self):
        return self.cache.metadata

    def _features(self, u, step, equation, geometry):
        ones = torch.ones_like(u)
        centered = u - u.mean((-2, -1), keepdim=True)
        lx, ly = geometry.lengths
        return torch.cat((u, centered, ones * step / self.t_ref,
            ones * equation.reaction_rate * self.t_ref,
            ones * equation.kappa * self.t_ref / lx**2,
            ones * equation.kappa * self.t_ref / ly**2,
            ones / geometry.grid[0], ones / geometry.grid[1]), 1)

    def correction_components(self, u, h, equation, geometry):
        check_shape(u, geometry)
        if geometry.ndim != 2:
            raise ValueError("Frontier architecture comparisons require two-dimensional periodic fields")
        step = numerics.validate(u, h, geometry, self.track)
        if self.family == "etdrk4":
            return {"base": numerics.etdrk4_step(u, step, equation, geometry, self.track, cache=self.cache),
                    "increment": torch.zeros_like(u)}
        base = u if self.family == "direct_fno" else numerics.df_step(u, step, equation, geometry, self.track,
            reaction_substeps=self.reaction_substeps, cache=self.cache)
        increment = torch.zeros_like(u)
        if self.family in RANK_FAMILIES:
            state = lowpass(u, self.modes) if self.family == "rank1_postcompression" else u
            increment = self.pair(state, step, equation, geometry, self.track)
        elif self.family.startswith("analytic_quad"):
            increment = quadratic_df_defect(u, step, equation, geometry, nodes=self.quad_nodes, target=self.track)
            if self.family == "analytic_quad_cubic":
                increment = increment + cubic_df_defect(u, step, equation, geometry, nodes=self.cubic_nodes, target=self.track)
        elif self.family in ("fno_small", "fno_standard", "direct_fno"):
            features = self._features(u, step, equation, geometry).to(self.lift.weight.dtype)
            raw = self.head(self.blocks(self.lift(features))).to(u.dtype)
            exponent = 1 if self.family == "direct_fno" else 3
            increment = (step / self.t_ref).pow(exponent) * raw
        return {"base": base, "increment": increment}

    def forward(self, u, h, equation, geometry):
        parts = self.correction_components(u, h, equation, geometry)
        value = parts["base"] + parts["increment"]
        # Exact identity; no clipping, invariant enforcement or concealed
        # repair of failed outputs. Downstream metrics retain those failures.
        return torch.where(broadcast_h(h, u) == 0, u, value)

    def architecture_metadata(self):
        rank = self.family in RANK_FAMILIES
        fno = self.family in ("fno_small", "fno_standard", "direct_fno")
        return dict(family=self.family, track=self.track,
            parameters=sum(p.numel() for p in self.parameters()),
            trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
            width=self.width if fno else None, depth=self.depth if fno else None, modes=self.modes,
            physical_core="none" if self.family == "direct_fno" else "etdrk4" if self.family == "etdrk4" else "diffusion_first_strang",
            diffusion_symbol="periodic_fd" if self.track == "discrete" else "spectral",
            reaction_target="nodal_logistic" if self.track == "discrete" else "quadratic_dealiased_galerkin",
            reaction_flow="exact_logistic" if self.track == "discrete" else "projected_RK4_not_exact_logistic",
            reaction_substeps_minimum=self.reaction_substeps,
            nominal_learned_rank=1 if rank else 0, expanded_separable_term_bound=3 if rank else 0,
            rank_convention="symmetric physical node pair, not Tucker or matrix rank",
            correction_input="compressed" if self.family == "rank1_postcompression" else "full_state",
            physical_core_uses_full_state=self.family != "direct_fno",
            source_before_output_compression=rank and self.family != "rank1_postcompression",
            learned_input_bypass=False, fno_full_grid_local_path=fno,
            exact_correction_nulls=["constant", "reaction=0", "diffusion=0", "time=0"] if rank or self.family.startswith("analytic_quad") else ["time=0"],
            invariant_interval_guarantee=False, clipping=False,
            continuum_truth_claim=False, published_FNO_reproduction=False,
            precision_policy="input FP32 or FP64; no TF32 enablement", t_ref=self.t_ref,
            cache_policy="state-independent frozen scalar-time coefficients; separate charged cold/warm timing",
            parameter_matching_claim=False, compute_matching_claim=False)


def make_model(family, track="discrete", config=None):
    config = {} if config is None else dict(config)
    allowed = {"width", "modes", "depth", "t_ref", "reaction_substeps", "quad_nodes", "cubic_nodes", "initial_amplitude", "cache_entries"}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"Unknown frontier architecture settings: {sorted(unknown)}")
    return FrontierModel(family, track, **config)
