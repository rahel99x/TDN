"""Full-input nonlinear mixing followed by a fixed Fourier output projection.

``output_gl3`` computes the full fused GL3 quadratic interaction, then retains
only output modes |m_i| <= K_i. This is a representation control, not a cheap
solver. ``selected_output_gl3`` computes exactly those coefficients by cyclic
convolution over *all* original input modes. It retains high-high-to-low
coupling and original-grid aliases, but costs O(B*N*M) pair products for B
parents, N grid cells and M retained output modes. Its chunked workspace is
O(B*N*C) without autograd; backpropagation retains intermediate graphs.
Fourier synthesis and the Strang base remain full-grid operations.
``gl3_fused_chunked`` changes only parent batching of full fused GL3.

The direct path uses FP64 pair/reduction and coefficient arithmetic even for
FP32 input; the input FFT and final synthesis retain the input dtype. Stable
exponential differences avoid small-time subtraction and positive-exponent
overflow. Four kernel evaluations per output/input pair happen inside every
measured call, not in an unreported preparation cache. Counters expose that
work separately from FFT dispatch/fields/cells. No clipping, neural training,
continuum dealiasing, or higher-order accuracy guarantee is implied.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from tdn.numerics.subflows import diffusion_eigenvalues
from tdn.numerics.types import Equation, Geometry
from tdn.research.compact_spatial import (
    PreparedCompactStep, _SpatialCoefficients, _Transforms, _modes,
)
from tdn.research.interaction import Work, _GL, _center, _counts, _finish, _validate, phi
from tdn.research.work_precision import _Coefficients, _ReactionConstants


VARIANTS = ("output_gl3", "selected_output_gl3", "gl3_fused_chunked")


class _TemporalCoefficients:
    weight = _Coefficients.weight

    def __init__(self, step: Tensor, rate: float):
        self.rate = rate
        self.half = _ReactionConstants.at(step / 2, rate)
        self.full = _ReactionConstants.at(step, rate)
        self.weight_factor = step / 2 * phi(-rate * step / 2)
        locations, weights = _GL[3]
        self.nodes = tuple((step * ((1 + x) / 2), step * weight / 2,
                            _ReactionConstants.at(step * ((1 + x) / 2), rate))
                           for x, weight in zip(locations, weights))


def _cached_tensors(value: Any):
    if isinstance(value, Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _cached_tensors(child)
    elif isinstance(value, (_Coefficients, _SpatialCoefficients, _ReactionConstants,
                            _TemporalCoefficients)):
        for child in vars(value).values():
            yield from _cached_tensors(child)


def _transported_difference(a: Tensor, b: Tensor, s: Tensor, h: Tensor) -> Tensor:
    """exp((h-s)b) * (exp(s*a)-exp(s*b)), with a,b <= 0.

    Keeping all exponent arguments nonpositive is important when the two
    eigenvalues differ greatly. exp(h*b)*expm1((a-b)*s) can form 0*infinity.
    """
    difference = a - b
    return (torch.exp(torch.maximum(a, b) * s + b * (h - s))
            * (-torch.expm1(-difference.abs() * s)) * difference.sign())


class PreparedPremixStep:
    """Freeze h/grid/dtype/device; cache no input states or convolution pairs."""

    _check = PreparedCompactStep._check
    _split = PreparedCompactStep._split

    def __init__(self, example: Tensor, h: float | Tensor, equation: Equation,
                 geometry: Geometry, variant: str = "selected_output_gl3", *,
                 modes: int | tuple[int, ...] = 4, chunk_size: int = 8):
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        if not isinstance(example, Tensor):
            raise TypeError("example must be a Tensor")
        step = _validate(example, h, equation, geometry)
        if step.ndim != 0 or step.requires_grad:
            raise ValueError("prepared h must be a frozen scalar without time gradients")
        self.variant = variant
        self.chunk_size = chunk_size
        self._geometry, self._equation = geometry, equation
        self._dtype, self._device = example.dtype, example.device
        self.modes = _modes(modes, geometry)
        if variant == "gl3_fused_chunked":
            self.modes = tuple(n // 2 for n in geometry.grid)
        self._p = _Coefficients(example, step.detach().clone(), equation, geometry, "strang")
        self._spatial = self._temporal = self._spectrum = None
        self._coordinates = self._output_coordinates = self._output_indices = self._mask = None
        self._cells = math.prod(geometry.grid)
        self._retained = math.prod(min(n, 2 * k + 1) for n, k in zip(geometry.grid, self.modes))
        if self._p.zero_time or not equation.kappa or not equation.reaction_rate:
            return
        spectrum = equation.kappa * diffusion_eigenvalues(geometry, example)
        if variant != "selected_output_gl3":
            self._spatial = _SpatialCoefficients(self._p.step, spectrum, equation.reaction_rate)
        if variant == "gl3_fused_chunked":
            return
        # Original FFT indexing includes each even-grid Nyquist mode exactly
        # once. The rectangular mask is conjugate symmetric in every axis.
        coordinates = torch.meshgrid(*(torch.arange(n, device=example.device)
                                       for n in geometry.grid), indexing="ij")
        mask = torch.ones(geometry.grid, dtype=torch.bool, device=example.device)
        for coordinate, n, k in zip(coordinates, geometry.grid, self.modes):
            signed = torch.where(coordinate <= (n - 1) // 2, coordinate, coordinate - n)
            mask = mask & (signed.abs() <= k)
        self._mask = mask
        self._output_indices = mask.flatten().nonzero().flatten()
        if variant == "selected_output_gl3":
            self._coordinates = tuple(coordinate.flatten() for coordinate in coordinates)
            self._output_coordinates = tuple(coordinate.index_select(0, self._output_indices)
                                             for coordinate in self._coordinates)
            self._spectrum = spectrum.flatten().double()
            self._temporal = _TemporalCoefficients(self._p.step.double(), equation.reaction_rate)

    @property
    def metadata(self) -> dict[str, Any]:
        tensors = list(_cached_tensors((self._p, self._spatial, self._temporal, self._spectrum,
                                       self._coordinates, self._output_coordinates,
                                       self._output_indices, self._mask)))
        storage = {item.untyped_storage().data_ptr(): item.untyped_storage().nbytes() for item in tensors}
        direct = self.variant == "selected_output_gl3"
        return {"variant": self.variant, "h": float(self._p.step), "dtype": str(self._dtype),
                "device": str(self._device), "grid": list(self._geometry.grid),
                "modes": list(self.modes), "retained_output_modes": self._retained,
                "cutoff_semantics": "axis-wise absolute Fourier output frequency after full-input mixing",
                "full_input_spectrum": True, "full_spectrum_mean": True,
                "original_discrete_diffusion_symbol": True, "cyclic_original_grid_products": True,
                "chunk_size": self.chunk_size,
                "chunk_axis": "output_modes" if direct else "parents" if self.variant.endswith("chunked") else None,
                "cached_tensor_bytes": sum(storage.values()),
                "cache_tensor_count": len({id(item) for item in tensors}),
                "state_dependent_cache": False, "pair_kernel_cache": False,
                "setup_fft_total": 0, "work_fields_include_batch": True,
                "validation_per_call": True, "time_autograd_supported": False,
                "pair_reduction_dtype": "torch.float64" if direct else None,
                "pair_product_cells_per_parent": self._cells * self._retained if direct else 0,
                "pair_kernel_values_per_call": 4 * self._cells * self._retained if direct else 0,
                "workspace_scope": "forward evaluation without retained autograd graph; not measured peak memory",
                "workspace_scaling": "O(batch*cells*min(chunk_size,retained_output_modes))" if direct else
                                     "O(min(batch,chunk_size)*cells)" if self.variant.endswith("chunked") else "O(batch*cells)"}

    def _full_defect(self, u: Tensor, transform: _Transforms, counts: dict[str, int]) -> Tensor:
        p, spatial = self._p, self._spatial
        c, v, uniform = _center(u)
        if p.zero_time or not p.kappa or not p.rate or bool(uniform.all()):
            return u * 0
        vhat = transform.fft(v)
        square_hat = transform.fft(v.square())
        counts["nonlinear_evaluations"] += 1
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
        if self.variant == "output_gl3":
            defect_hat = defect_hat * self._mask
            counts["output_projection_cells"] += defect_hat.numel()
        answer = transform.inverse(defect_hat)
        return torch.where(uniform, torch.zeros_like(answer), answer)

    def _selected_defect(self, u: Tensor, transform: _Transforms, counts: dict[str, int]) -> Tensor:
        p, temporal = self._p, self._temporal
        c, v, uniform = _center(u)
        if p.zero_time or not p.kappa or not p.rate or bool(uniform.all()):
            return u * 0
        native_hat = transform.fft(v)
        # With the default FFT normalization, a cyclic product coefficient is
        # (1/N) sum_k V[k] V[m-k]. Division occurs only after the reduction.
        vhat = native_hat.flatten(start_dim=2).to(torch.complex128)
        safe_c = torch.where(uniform, torch.full_like(c, .5), c).reshape(u.shape[0], 1, 1).double()
        jacobians = [reaction.jacobian(safe_c, p.rate) for _, _, reaction in temporal.nodes]
        final_jacobian = temporal.full.jacobian(safe_c, p.rate)
        weight = temporal.weight(safe_c)
        counts["reaction_jacobian_evaluations"] += 4
        counts["reaction_weight_evaluations"] += 1
        values = []
        h = temporal.full.time
        for start in range(0, self._retained, self.chunk_size):
            stop = min(start + self.chunk_size, self._retained)
            pair_indices = torch.zeros((stop - start, self._cells), dtype=torch.int64, device=u.device)
            for axis, n in enumerate(self._geometry.grid):
                shifted = (self._output_coordinates[axis][start:stop, None]
                           - self._coordinates[axis][None, :]).remainder(n)
                pair_indices = pair_indices + shifted * math.prod(self._geometry.grid[axis + 1:])
            products = vhat.unsqueeze(2) * vhat[:, :, pair_indices]
            counts["pair_product_cells"] += products.numel()
            counts["pair_chunks"] += 1
            counts["nonlinear_evaluations"] += 1
            a = self._spectrum[None, :] + self._spectrum[pair_indices]
            b = self._spectrum[self._output_indices[start:stop]][:, None]
            integral = torch.zeros(products.shape[:-1], dtype=products.dtype, device=u.device)
            for (s, quadrature_weight, _), jacobian in zip(temporal.nodes, jacobians):
                kernel = _transported_difference(a, b, s, h)
                counts["pair_kernel_values"] += kernel.numel()
                transported = (products * kernel).sum(dim=-1) / self._cells
                integral = integral + quadrature_weight * jacobian * transported
            endpoint = _transported_difference(a, b, h, h)
            counts["pair_kernel_values"] += endpoint.numel()
            integral = integral - weight * (products * endpoint).sum(dim=-1) / self._cells
            values.append(-p.rate * final_jacobian * integral)
        counts["quadrature_evaluations"] += 3
        selected = torch.cat(values, dim=-1).to(native_hat.dtype)
        full_hat = torch.zeros_like(native_hat).flatten(start_dim=2)
        full_hat = full_hat.index_copy(2, self._output_indices, selected).reshape_as(native_hat)
        answer = transform.inverse(full_hat)
        return torch.where(uniform, torch.zeros_like(answer), answer)

    def _evaluate(self, u: Tensor, transform: _Transforms, counts: dict[str, int], include_split: bool) -> Tensor:
        base = self._split(u, transform, counts) if include_split else 0.
        defect = self._selected_defect(u, transform, counts) if self.variant == "selected_output_gl3" else self._full_defect(u, transform, counts)
        return base + defect

    def _call(self, u: Tensor, work: Work | None, include_split: bool) -> Tensor:
        self._check(u)
        counts = _counts()
        counts.update(fft_forward_fields=0, fft_inverse_fields=0, fft_total_fields=0,
                      fft_transformed_cells=0, pair_product_cells=0, pair_kernel_values=0,
                      pair_chunks=0, parent_chunks=0, output_projection_cells=0)
        transform = _Transforms(self._geometry.ndim, counts)
        try:
            if self._p.zero_time:
                return u + self._p.step * 0 if include_split else u * 0
            if self.variant == "gl3_fused_chunked":
                values = []
                for start in range(0, u.shape[0], self.chunk_size):
                    values.append(self._evaluate(u[start:start + self.chunk_size], transform, counts, include_split))
                    counts["parent_chunks"] += 1
                return torch.cat(values, dim=0)
            return self._evaluate(u, transform, counts, include_split)
        finally:
            _finish(work, counts)

    def __call__(self, u: Tensor, work: Work | None = None) -> Tensor:
        return self._call(u, work, True)

    def defect(self, u: Tensor, work: Work | None = None) -> Tensor:
        """Raw full-grid correction, before addition to the Strang base."""
        return self._call(u, work, False)


def prepare_premix_step(example: Tensor, h: float | Tensor, equation: Equation,
                        geometry: Geometry, variant: str = "selected_output_gl3", *,
                        modes: int | tuple[int, ...] = 4, chunk_size: int = 8) -> PreparedPremixStep:
    return PreparedPremixStep(example, h, equation, geometry, variant, modes=modes, chunk_size=chunk_size)


def premix_defect(u: Tensor, h: float | Tensor, equation: Equation, geometry: Geometry, *,
                  variant: str = "selected_output_gl3", modes: int | tuple[int, ...] = 4,
                  chunk_size: int = 8, work: Work | None = None) -> Tensor:
    """Convenience raw correction; preparation is included in each call."""
    return prepare_premix_step(u, h, equation, geometry, variant,
                               modes=modes, chunk_size=chunk_size).defect(u, work)
