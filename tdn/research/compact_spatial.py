"""Prepared, training-free compression of the quadratic interaction defect.

``gl3_fused`` preserves the full GL3 formula and batches its quadrature FFTs.
``compact_gl3`` retains input frequencies |k_i| <= K_i, evaluates their
quadratic interactions on M_i=min(N_i,4*K_i+1) cells, and injects the zero-mean
result into the original grid. It adds the full-spectrum Parseval GL3 mean.
``compact_gl3_no_mean`` omits that mean, isolating the spatial component.

The compact grid is an alias-safe evaluation device, NOT a coarsened PDE:
every multiplier uses the original discrete diffusion eigenvalue. Whenever
M_i=N_i the original finite-grid cyclic product aliases are retained. For a
smaller M_i, all retained input products fit in its frequency range. High-mode
input interactions are deliberately discarded, including high-high-to-low
coupling. No reference or full spatial GL3 correction is computed first.

FFT counters distinguish API calls, transformed fields (including parents),
and transformed cells. Batching reduces dispatch, not equivalent FFT work.
Complete steps still have full-grid FFTs and O(N log N) cost. Preparation is
state independent; these are forward numerical controls, not time-autograd
or positivity-preserving solvers. Outputs are never clipped.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from tdn.numerics.subflows import diffusion_eigenvalues
from tdn.numerics.types import Equation, Geometry
from tdn.research.interaction import Work, _GL, _center, _counts, _finish, _validate, phi
from tdn.research.work_precision import _Coefficients, _ReactionConstants


VARIANTS = ("gl3_fused", "compact_gl3", "compact_gl3_no_mean")


def _modes(modes: int | tuple[int, ...], geometry: Geometry) -> tuple[int, ...]:
    values = (modes,) * geometry.ndim if type(modes) is int else modes
    if not isinstance(values, tuple) or len(values) != geometry.ndim:
        raise ValueError("modes must be a nonnegative integer or one integer per axis")
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("mode cutoffs must be nonnegative integers")
    return tuple(min(value, n // 2) for value, n in zip(values, geometry.grid))


def _select(value: Tensor, indices: tuple[Tensor, ...]) -> Tensor:
    for axis, index in enumerate(indices, start=value.ndim - len(indices)):
        value = value.index_select(axis, index)
    return value


class _SpatialCoefficients:
    # Reuse the analytically stable weight formula, with native-dtype prepared
    # constants. The separately computed Parseval mean uses FP64 arithmetic.
    weight = _Coefficients.weight

    def __init__(self, step: Tensor, spectrum: Tensor, rate: float):
        self.rate = rate
        self.half = _ReactionConstants.at(step / 2, rate)
        self.full = _ReactionConstants.at(step, rate)
        self.weight_factor = step / 2 * phi(-rate * step / 2)
        locations, weights = _GL[3]
        times = torch.stack([step * ((1 + x) / 2) for x in locations] + [step])
        shape = (4, 1, 1) + (1,) * spectrum.ndim
        times = times.reshape(shape)
        self.reactions = _ReactionConstants.at(times, rate)
        self.increments = torch.expm1(times * spectrum)
        self.transport = torch.exp((step - times) * spectrum)
        self.weights = step * torch.tensor((*weights, 0.), dtype=step.dtype, device=step.device).reshape(shape) / 2


def _tensors(value: Any):
    if isinstance(value, Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _tensors(child)
    elif isinstance(value, (_Coefficients, _ReactionConstants, _SpatialCoefficients)):
        for child in vars(value).values():
            yield from _tensors(child)


class _Transforms:
    def __init__(self, ndim: int, counts: dict[str, int]):
        self.axes = tuple(range(-ndim, 0))
        self.counts = counts

    def _record(self, value: Tensor, direction: str) -> None:
        cells = math.prod(value.shape[axis] for axis in self.axes)
        fields = value.numel() // cells
        key = f"fft_{direction}"
        self.counts[key] += 1
        self.counts[f"{key}_fields"] += fields
        self.counts["fft_total_fields"] += fields
        self.counts["fft_transformed_cells"] += value.numel()

    def fft(self, value: Tensor) -> Tensor:
        answer = torch.fft.fftn(value, dim=self.axes)
        self._record(value, "forward")
        return answer

    def inverse(self, value: Tensor) -> Tensor:
        answer = torch.fft.ifftn(value, dim=self.axes).real
        self._record(value, "inverse")
        return answer


class PreparedCompactStep:
    """Freeze scalar time/grid/dtype/device; allow changing batch size/state."""

    def __init__(self, example: Tensor, h: float | Tensor, equation: Equation,
                 geometry: Geometry, variant: str = "compact_gl3", *,
                 modes: int | tuple[int, ...] = 4):
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if not isinstance(example, Tensor):
            raise TypeError("example must be a Tensor")
        step = _validate(example, h, equation, geometry)
        if step.ndim != 0 or step.requires_grad:
            raise ValueError("prepared h must be a frozen scalar without time gradients")
        self.variant = variant
        self._geometry, self._equation = geometry, equation
        self._dtype, self._device = example.dtype, example.device
        self.modes = _modes(modes, geometry)
        if variant == "gl3_fused":
            self.modes = tuple(n // 2 for n in geometry.grid)
        self.compact_grid = (geometry.grid if variant == "gl3_fused" else
                             tuple(min(n, 4 * k + 1) for n, k in zip(geometry.grid, self.modes)))
        self._p = _Coefficients(example, step.detach().clone(), equation, geometry,
                                "gl3_mean_spectral" if variant == "compact_gl3" else "strang")
        self._indices: tuple[Tensor, ...] = ()
        self._flat_indices: Tensor | None = None
        self._mask: Tensor | None = None
        self._spatial: _SpatialCoefficients | None = None
        if self._p.zero_time or not equation.kappa or not equation.reaction_rate:
            return
        if variant == "gl3_fused":
            spectrum = equation.kappa * diffusion_eigenvalues(geometry, example)
            self._spatial = _SpatialCoefficients(self._p.step, spectrum, equation.reaction_rate)
            return
        indices, mask = [], torch.ones(self.compact_grid, dtype=torch.bool, device=example.device)
        for axis, (n, m, k) in enumerate(zip(geometry.grid, self.compact_grid, self.modes)):
            frequency = torch.arange(m, device=example.device)
            frequency = torch.where(frequency <= (m - 1) // 2, frequency, frequency - m)
            indices.append(frequency.remainder(n))
            shape = [1] * geometry.ndim
            shape[axis] = m
            mask = mask & (frequency.abs() <= k).reshape(shape)
        self._indices = tuple(indices)
        self._mask = mask
        mesh = torch.meshgrid(*indices, indexing="ij")
        flat = torch.zeros(self.compact_grid, dtype=torch.int64, device=example.device)
        for axis, index in enumerate(mesh):
            flat = flat + index * math.prod(geometry.grid[axis + 1:])
        self._flat_indices = flat.flatten()
        spectrum = equation.kappa * _select(diffusion_eigenvalues(geometry, example), self._indices)
        self._spatial = _SpatialCoefficients(self._p.step, spectrum, equation.reaction_rate)

    @property
    def metadata(self) -> dict[str, Any]:
        tensors = list(_tensors((self._p, self._spatial, self._indices,
                                  self._flat_indices, self._mask)))
        storages = {item.untyped_storage().data_ptr(): item.untyped_storage().nbytes() for item in tensors}
        return {"variant": self.variant, "h": float(self._p.step), "dtype": str(self._dtype),
                "device": str(self._device), "grid": list(self._geometry.grid),
                "modes": list(self.modes), "compact_grid": list(self.compact_grid),
                "cutoff_semantics": "axis-wise absolute Fourier input frequency",
                "full_spectrum_mean": self.variant != "compact_gl3_no_mean",
                "mean_evaluation": ("parseval_gl3" if self.variant == "compact_gl3" else
                                    "full_spatial_gl3" if self.variant == "gl3_fused" else "omitted"),
                "original_discrete_diffusion_symbol": True,
                "cached_tensor_bytes": sum(storages.values()),
                "cache_tensor_count": len({id(item) for item in tensors}),
                "state_dependent_cache": False, "validation_per_call": True,
                "setup_fft_total": 0,
                "work_fields_include_batch": True,
                "spectral_reduction_dtype": "torch.float64" if self.variant == "compact_gl3" else None}

    def _check(self, u: Tensor) -> None:
        if not isinstance(u, Tensor):
            raise TypeError("state must be a Tensor")
        if u.dtype != self._dtype or u.device != self._device:
            raise ValueError("state dtype and device must match the prepared example")
        _validate(u, self._p.step, self._equation, self._geometry)

    def _split(self, u: Tensor, transform: _Transforms, counts: dict[str, int]) -> Tensor:
        p = self._p

        def reaction(value):
            if not p.rate:
                return value + p.step * 0
            denominator = value + (1 - value) * p.split_q
            safe = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
            answer = value / safe
            counts["reaction_evaluations"] += 1
            return answer

        first = reaction(u)
        second = transform.inverse(transform.fft(first) * p.E) if p.kappa else first + p.step * 0
        return reaction(second)

    def _mean(self, vhat: Tensor, c: Tensor, uniform: Tensor, counts: dict[str, int]) -> Tensor:
        p = self._p
        cells = math.prod(self._geometry.grid)
        power = (vhat.real.double().square() + vhat.imag.double().square()) / cells**2
        safe_c = torch.where(uniform, torch.full_like(c, .5), c).double()
        counts["nonlinear_evaluations"] += 1
        integral = torch.zeros_like(safe_c)
        for weight, reaction, increment, _ in p.quadrature:
            variation = (power * increment).sum(dim=p.axes, keepdim=True)
            integral = integral + weight * reaction.jacobian(safe_c, p.rate) * variation
            counts["quadrature_evaluations"] += 1
            counts["reaction_jacobian_evaluations"] += 1
        integral = integral - p.weight(safe_c) * (power * p.endpoint_increment).sum(dim=p.axes, keepdim=True)
        answer = -p.rate * p.full.jacobian(safe_c, p.rate) * integral
        counts["reaction_weight_evaluations"] += 1
        counts["reaction_jacobian_evaluations"] += 1
        return torch.where(uniform, torch.zeros_like(answer), answer).to(vhat.real.dtype)

    def _defect(self, u: Tensor, transform: _Transforms, counts: dict[str, int]) -> Tensor:
        p = self._p
        c, v, uniform = _center(u)
        if p.zero_time or not p.kappa or not p.rate or bool(uniform.all()):
            return u * 0
        vhat = transform.fft(v)
        mean = self._mean(vhat, c, uniform, counts) if self.variant == "compact_gl3" else 0.
        if self.variant != "gl3_fused":
            # Crop normalized Fourier coefficients. M/N is essential because
            # PyTorch's inverse divides by the compact cell count.
            ratio = math.prod(self.compact_grid) / math.prod(self._geometry.grid)
            vhat = _select(vhat, self._indices) * self._mask * ratio
            v = transform.inverse(vhat)
        square_hat = transform.fft(v.square())
        counts["nonlinear_evaluations"] += 1
        spatial = self._spatial
        # [node, component, parent, channel, *grid], with FFTs over spatial
        # axes only. Nodes and parents never contaminate each other's means.
        hats = torch.stack((vhat, square_hat), dim=0).unsqueeze(0)
        evolved = transform.inverse(spatial.increments.unsqueeze(1) * hats)
        dv, dsquare = evolved[:, 0], evolved[:, 1]
        variation = 2 * v.unsqueeze(0) * dv + dv.square() - dsquare
        counts["nonlinear_evaluations"] += 4
        variations_hat = transform.fft(variation)
        safe_c = torch.where(uniform, torch.full_like(c, .5), c)
        jacobians = spatial.reactions.jacobian(safe_c.unsqueeze(0), p.rate)
        integral_hat = (spatial.weights * jacobians * spatial.transport * variations_hat).sum(dim=0)
        integral_hat = integral_hat - spatial.weight(safe_c) * variations_hat[-1]
        defect_hat = -p.rate * jacobians[-1] * integral_hat
        counts["quadrature_evaluations"] += 3
        counts["reaction_jacobian_evaluations"] += 4
        counts["reaction_weight_evaluations"] += 1
        if self.variant != "gl3_fused":
            defect_hat = defect_hat.flatten(start_dim=2).clone()
            defect_hat[..., 0] = 0  # isolate spatial branch before synthesis
            full_hat = torch.zeros((u.shape[0], 1, math.prod(self._geometry.grid)),
                                   dtype=defect_hat.dtype, device=u.device)
            full_hat.index_copy_(2, self._flat_indices, defect_hat / ratio)
            defect_hat = full_hat.reshape(u.shape)
        answer = transform.inverse(defect_hat) + mean
        return torch.where(uniform, torch.zeros_like(answer), answer)

    def _call(self, u: Tensor, work: Work | None, include_split: bool) -> Tensor:
        self._check(u)
        counts = _counts()
        counts.update(fft_forward_fields=0, fft_inverse_fields=0,
                      fft_total_fields=0, fft_transformed_cells=0)
        transform = _Transforms(self._geometry.ndim, counts)
        try:
            if self._p.zero_time:
                return u + self._p.step * 0 if include_split else u * 0
            base = self._split(u, transform, counts) if include_split else 0.
            return base + self._defect(u, transform, counts)
        finally:
            _finish(work, counts)

    def __call__(self, u: Tensor, work: Work | None = None) -> Tensor:
        return self._call(u, work, True)

    def defect(self, u: Tensor, work: Work | None = None) -> Tensor:
        """Full-grid raw correction, before addition to the Strang base."""
        return self._call(u, work, False)


def prepare_compact_step(example: Tensor, h: float | Tensor, equation: Equation,
                         geometry: Geometry, variant: str = "compact_gl3", *,
                         modes: int | tuple[int, ...] = 4) -> PreparedCompactStep:
    return PreparedCompactStep(example, h, equation, geometry, variant, modes=modes)


def compact_defect(u: Tensor, h: float | Tensor, equation: Equation,
                   geometry: Geometry, *, variant: str = "compact_gl3",
                   modes: int | tuple[int, ...] = 4, work: Work | None = None) -> Tensor:
    """Convenience raw correction; preparation is included in each call."""
    return prepare_compact_step(u, h, equation, geometry, variant, modes=modes).defect(u, work)
