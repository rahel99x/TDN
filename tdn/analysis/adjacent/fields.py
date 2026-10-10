"""Finite, alias-free parent fields for the adjacent interaction study.

The fractional labels refer to an ideal infinite ridge, never to a finite
trigonometric polynomial or the dimension of a neural network. All grid samples
come from the same stored Fourier coefficients. Bounds are certified before
sampling with a conservative triangle bound; neither clipping nor per-grid
normalization is permitted.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math

import numpy as np
import torch

ALPHAS = tuple(i / 10 for i in range(1, 11))


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class FieldParent:
    seed: int
    alpha: float
    generator: str
    levels: int
    phase_mode: str
    orientation: tuple[int, int]
    mean: float
    rms: float
    bounds: tuple[float, float]
    modes: tuple[tuple[int, int], ...]
    amplitudes: tuple[float, ...]
    phases: tuple[float, ...]
    shell_indices: tuple[int, ...]
    split: str
    cluster_id: str
    parent_id: str
    modes_per_level: int
    translation: tuple[float, float] = (0., 0.)
    workload: str = "fixed_physical_bandwidth"

    @property
    def hurst(self):
        return 1. - self.alpha

    @property
    def dimension_label(self):
        if self.alpha == 1.:
            return "D=3.0 label: finite-bandwidth H=0 stress, no convergent-series claim"
        if self.generator == "ridge" and self.phase_mode == "structured":
            return f"D={2+self.alpha:.1f} ideal canonical ridge; finite sample is smooth"
        return f"nominal D={2+self.alpha:.1f}; finite sample smooth, dimension theorem not transferred"

    @property
    def continuous_range_bound(self):
        radius = math.fsum(abs(a) for a in self.amplitudes)
        return (self.mean - radius, self.mean + radius)

    @property
    def identity_sha256(self):
        return _hash(asdict(self))

    def to_dict(self):
        # Round trip through JSON supplies a completely JSON-native structure.
        result = json.loads(json.dumps(asdict(self), allow_nan=False))
        result.update(hurst=self.hurst, dimension_label=self.dimension_label,
                      continuous_range_bound=list(self.continuous_range_bound),
                      identity_sha256=self.identity_sha256,
                      finite_field_dimension=2,
                      statistical_unit="cluster_id, never phase/grid/roughness variants",
                      normalization="continuous Fourier RMS; no clipping or grid renormalization",
                      physical_frequency_units="integer cycles per domain length")
        return result

    @classmethod
    def from_dict(cls, value):
        data = {k: value[k] for k in cls.__dataclass_fields__ if k in value}
        for key in ("orientation", "bounds", "amplitudes", "phases", "shell_indices", "translation"):
            if key in data:
                data[key] = tuple(data[key])
        data["modes"] = tuple(tuple(m) for m in data["modes"])
        parent = cls(**data)
        _validate(parent)
        if value.get("identity_sha256", parent.identity_sha256) != parent.identity_sha256:
            raise ValueError("Field parent identity hash mismatch")
        return parent


def _validate(parent):
    if not 0 < parent.alpha <= 1 or not math.isfinite(parent.alpha):
        raise ValueError("alpha must be in (0,1]")
    if parent.generator not in ("ridge", "multiscale_2d"):
        raise ValueError("Unknown generator")
    if parent.phase_mode not in ("random", "structured"):
        raise ValueError("Unknown phase mode")
    if parent.levels < 1 or parent.levels > 12:
        raise ValueError("levels must be between 1 and 12")
    if not math.isfinite(parent.rms) or parent.rms < 0 or not math.isfinite(parent.mean):
        raise ValueError("mean and nonnegative RMS must be finite")
    if len(parent.bounds) != 2 or not all(math.isfinite(x) for x in parent.bounds) or parent.bounds[0] >= parent.bounds[1]:
        raise ValueError("Finite ordered physical bounds required")
    size = len(parent.modes)
    if not size or any(len(v) != size for v in (parent.amplitudes, parent.phases, parent.shell_indices)):
        raise ValueError("Mode, amplitude, phase and shell inventories must match")
    canonical = []
    for mode in parent.modes:
        if len(mode) != 2 or any(int(k) != k for k in mode) or mode == (0, 0):
            raise ValueError("Nonzero integer two-dimensional modes required")
        canonical.append(mode if mode[0] > 0 or (mode[0] == 0 and mode[1] > 0) else tuple(-k for k in mode))
    if len(set(canonical)) != size:
        raise ValueError("Duplicate or conjugate mode would invalidate Fourier RMS")
    if not all(math.isfinite(x) for x in parent.amplitudes + parent.phases + parent.translation):
        raise ValueError("Finite Fourier coefficients and translation required")
    actual_rms = math.sqrt(math.fsum(a*a/2 for a in parent.amplitudes))
    if not math.isclose(actual_rms, parent.rms, rel_tol=2e-14, abs_tol=2e-15):
        raise ValueError("Stored coefficients do not have declared continuous RMS")
    low, high = parent.continuous_range_bound
    if low < parent.bounds[0] - 2e-14 or high > parent.bounds[1] + 2e-14:
        raise ValueError("Mean, RMS and certified physical range cannot all be matched; lower RMS explicitly, never clip")


def _modes_at_level(seed, level, count):
    lower, upper = 2**level, 2**(level+1)
    # A random direction set spans both spatial axes. The random stream is
    # level-local, so increasing bandwidth does not reshuffle existing modes.
    rng = np.random.default_rng(np.random.SeedSequence([seed, level, 7001]))
    if level <= 3:
        pool = [(i, j) for i in range(upper) for j in range(-upper+1, upper)
                if (i > 0 or j > 0) and lower**2 <= i*i+j*j < upper**2]
        indices = rng.permutation(len(pool))[:min(count, len(pool))]
        modes = [pool[int(i)] for i in indices]
    else:
        # Bounded O(mode count) storage even for high physical bandwidth. No
        # all-mode/all-pair tensor is needed to generate a sparse parent.
        modes = []
        while len(modes) < count:
            i, j = int(rng.integers(upper)), int(rng.integers(-upper+1, upper))
            if (i > 0 or j > 0) and lower**2 <= i*i+j*j < upper**2 and (i,j) not in modes:
                modes.append((i,j))
    if np.linalg.matrix_rank(np.asarray(modes, dtype=float)) != 2:
        # This is only reachable with an invalid mode-count request or a very
        # unlikely collinear random selection. A deterministic replacement
        # guarantees genuinely two-dimensional support without retries.
        modes[-1] = (0, lower) if any(k[0] for k in modes[:-1]) else (lower, 0)
    return modes


def make_parent(seed, alpha, *, generator="multiscale_2d", levels=2,
                phase_mode="random", orientation=(1, 0), mean=.43, rms=.07,
                bounds=(0., 1.), parent_id=None, split="development", modes_per_level=4):
    """Create coefficients once; use :func:`sample_field` on every paired grid.

    ``ridge/structured`` is the canonical Weierstrass approximation. Random
    phases are paired controls at identical power, not automatic extensions of
    its dimension theorem. Distinct roughness/phase/grid variants with the same
    seed retain one cluster; structured ridges are deterministic mathematical
    checks and must not be counted as independent random examples.
    """
    if isinstance(seed, bool) or int(seed) != seed or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(levels, bool) or int(levels) != levels or not 1 <= levels <= 12:
        raise ValueError("levels must be between 1 and 12")
    if not math.isfinite(float(alpha)) or not 0 < float(alpha) <= 1:
        raise ValueError("alpha must be in (0,1]")
    if generator not in ("ridge", "multiscale_2d") or phase_mode not in ("random", "structured"):
        raise ValueError("Unsupported generator or phase mode")
    orientation = tuple(orientation)
    if len(orientation) != 2 or not any(orientation) or any(int(k) != k for k in orientation):
        raise ValueError("Ridge orientation must be a nonzero integer pair")
    if not isinstance(modes_per_level, int) or not 2 <= modes_per_level <= 16:
        raise ValueError("modes_per_level must be 2..16")
    modes, raw, phases, shells = [], [], [], []
    for j in range(int(levels)):
        selected = [tuple(int(k)*2**j for k in orientation)] if generator == "ridge" else _modes_at_level(int(seed), j, modes_per_level)
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), j, 9001]))
        for k in selected:
            modes.append(k)
            # Equal total energy per octave before H-dependent decay. Shell
            # multiplicity therefore does not alter the intended H control.
            raw.append(2**(-(1-float(alpha))*j)/math.sqrt(len(selected)))
            phases.append(float(rng.uniform(0, 2*math.pi)) if phase_mode == "random" else 0.)
            shells.append(j)
    factor = float(rms)/math.sqrt(math.fsum(a*a/2 for a in raw))
    cluster = f"{split}/seed-{int(seed)}"
    if generator == "ridge" and phase_mode == "structured":
        cluster = f"{split}/canonical-ridge-{orientation[0]}-{orientation[1]}"
    identity = dict(seed=int(seed), alpha=float(alpha), generator=generator, levels=int(levels),
                    phase_mode=phase_mode, orientation=orientation, mean=float(mean), rms=float(rms),
                    split=str(split), modes_per_level=modes_per_level)
    parent = FieldParent(int(seed), float(alpha), generator, int(levels), phase_mode,
                         orientation, float(mean), float(rms), tuple(float(x) for x in bounds),
                         tuple(modes), tuple(a*factor for a in raw), tuple(phases), tuple(shells),
                         str(split), cluster, parent_id or f"{cluster}/{_hash(identity)[:16]}", modes_per_level)
    _validate(parent)
    return parent


def phase_variant(parent, phase_mode):
    """A paired phase construction with unchanged power and statistical unit."""
    candidate = make_parent(parent.seed, parent.alpha, generator=parent.generator,
                            levels=parent.levels, phase_mode=phase_mode,
                            orientation=parent.orientation, mean=parent.mean, rms=parent.rms,
                            bounds=parent.bounds, split=parent.split,
                            modes_per_level=parent.modes_per_level)
    return replace(candidate, cluster_id=parent.cluster_id,
                   parent_id=f"{parent.parent_id}/phase-{phase_mode}")


def translate_parent(parent, shift):
    """Translate by fractions of the physical domain; never a new independent field."""
    shift = tuple(float(s) for s in shift)
    if len(shift) != 2 or not all(math.isfinite(s) for s in shift):
        raise ValueError("Two finite fractional-domain translations required")
    phases = tuple(p + 2*math.pi*sum(k*s for k,s in zip(mode,shift)) for mode,p in zip(parent.modes,parent.phases))
    candidate = replace(parent, phases=phases,
                        translation=tuple(a+b for a,b in zip(parent.translation,shift)),
                        parent_id=f"{parent.parent_id}/translation-{_hash(shift)[:8]}")
    _validate(candidate)
    return candidate


def with_bandwidth(parent, levels):
    """Add/remove physical scales at fixed RMS, explicitly changing the workload."""
    candidate = make_parent(parent.seed, parent.alpha, generator=parent.generator,
                            levels=levels, phase_mode=parent.phase_mode,
                            orientation=parent.orientation, mean=parent.mean, rms=parent.rms,
                            bounds=parent.bounds, split=parent.split,
                            modes_per_level=parent.modes_per_level)
    candidate = translate_parent(candidate, parent.translation) if any(parent.translation) else candidate
    return replace(candidate, cluster_id=parent.cluster_id,
                   parent_id=f"{parent.parent_id}/bandwidth-{levels}",
                   workload="changed_physical_bandwidth_at_fixed_total_RMS")


def sample_field(parent, grid, *, domain=(1., 1.), dtype=torch.float64, device="cpu"):
    if isinstance(parent, dict):
        parent = FieldParent.from_dict(parent)
    _validate(parent)
    grid = (grid, grid) if isinstance(grid, int) else tuple(grid)
    if len(grid) != 2 or any(isinstance(n, bool) or int(n) != n or n < 4 for n in grid):
        raise ValueError("Two integer grid sizes >= 4 required")
    if len(domain) != 2 or any(not math.isfinite(float(L)) or L <= 0 for L in domain):
        raise ValueError("Two positive finite domain lengths required")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("Use FP32 model inputs or FP64 teacher inputs")
    if any(2*abs(mode[axis]) >= grid[axis] for mode in parent.modes for axis in range(2)):
        raise ValueError("Grid must strictly resolve every physical mode below Nyquist; refinement must not alias the parent")
    x = torch.arange(grid[0], dtype=dtype, device=device)/grid[0]
    y = torch.arange(grid[1], dtype=dtype, device=device)/grid[1]
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    value = torch.full(grid, parent.mean, dtype=dtype, device=device)
    for (kx, ky), amplitude, phase in zip(parent.modes, parent.amplitudes, parent.phases):
        value = value + amplitude*torch.cos(2*math.pi*(kx*xx+ky*yy)+phase)
    return value[None, None]


def roughness_metrics(value, domain=(1., 1.)):
    """Measured finite-band diagnostics, not a fractal-dimension estimator.

    Spectral moments use physical angular wavenumbers. Increment fits require
    three measured separations and retain all fit points; finite bandwidth can
    invalidate the asymptotic power law even when a regression has small error.
    The quadratic interaction proxy uses the nodal product, explicitly labeled.
    """
    u = value.detach().to(dtype=torch.float64, device="cpu").squeeze()
    if u.ndim != 2 or not bool(torch.isfinite(u).all()):
        raise ValueError("Metrics require one finite two-dimensional field")
    if len(domain) != 2 or any(not math.isfinite(float(L)) or L <= 0 for L in domain):
        raise ValueError("Two positive finite domain lengths required")
    centered = u-u.mean()
    spectrum = torch.fft.fftn(centered, norm="forward")
    power = spectrum.abs().square()
    kx = torch.fft.fftfreq(u.shape[0], d=float(domain[0])/u.shape[0], dtype=torch.float64)
    ky = torch.fft.fftfreq(u.shape[1], d=float(domain[1])/u.shape[1], dtype=torch.float64)
    radial = torch.sqrt(kx[:,None].square()+ky[None,:].square())
    angular2 = (2*math.pi*radial).square()
    variance = float(centered.square().mean())
    shells = []
    min_frequency = min(1/float(L) for L in domain)
    max_frequency = float(radial.max())
    j = 0
    while min_frequency*2**j <= max_frequency:
        lower, upper = min_frequency*2**j, min_frequency*2**(j+1)
        mask = (radial >= lower*(1-1e-12)) & (radial < upper*(1-1e-12))
        shells.append(dict(lower_cycles_per_length=lower, upper_cycles_per_length=upper,
                           energy=float(power[mask].sum())))
        j += 1
    increments, estimates = [], []
    for axis, n in enumerate(u.shape):
        points = []
        offset = 1
        while offset <= n//8:
            sf = float((u-torch.roll(u, offset, dims=axis)).square().mean())
            record = dict(axis=axis, grid_offset=offset,
                          separation=float(domain[axis])*offset/n, mean_squared_increment=sf)
            increments.append(record)
            if sf > 1e-28:
                points.append(record)
            offset *= 2
        if len(points) >= 3:
            x = np.log([p["separation"] for p in points])
            y = np.log([p["mean_squared_increment"] for p in points])
            slope, intercept = np.polyfit(x, y, 1)
            estimates.append(dict(axis=axis, empirical_H=float(slope/2), points=len(points),
                                  log_residual_rms=float(np.sqrt(np.mean((y-slope*x-intercept)**2)))))
        else:
            estimates.append(dict(axis=axis, empirical_H=None, points=len(points),
                                  reason="fewer than three nonzero increment scales"))
    return dict(mean=float(u.mean()), rms=math.sqrt(variance), minimum=float(u.min()), maximum=float(u.max()),
                variance=variance, spectral_variance=float(power.sum()),
                gradient_energy=float((power*angular2).sum()),
                laplacian_energy=float((power*angular2.square()).sum()),
                mean_squared_angular_wavenumber=float((power*angular2).sum())/variance if variance > 0 else None,
                fourth_moment=float(centered.pow(4).mean()),
                nodal_quadratic_interaction_rms=float(centered.square().square().mean().sqrt()),
                interaction_proxy_equation="nodal v^2, not dealiased Galerkin and not a solver defect",
                shell_energies=shells, increments=increments, roughness_estimates=estimates,
                dimension_inference="NA: finite trigonometric samples are smooth")
