"""Bounded M14/M17/M19 transfer experiments and the actual C4 solver.

Targets are explicitly *semidiscrete*: spectral Galerkin periodic equations or
weighted finite-volume Dirichlet equations.  SciPy integrates the coupled RHS
independently of the split torch implementations.  Time refinement estimates
uncertainty; it is not a continuum certificate.  These controls do not claim a
paper reproduction, trained cross-PDE superiority, or a global stability proof.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable
import math

import numpy as np
import torch
from scipy.integrate import solve_ivp


def resample(u: torch.Tensor, n: int) -> torch.Tensor:
    """Fourier resampling with the ambiguous even-grid Nyquist modes removed.

    All retained modes satisfy |k_j| < min(old_n, new_n)/2.  The explicit
    Nyquist convention is shared by the target Galerkin generator and controls.
    States are scalar 1D or 2D arrays, not batched/channel-first arrays.
    """
    if u.ndim not in (1, 2) or len(set(u.shape)) != 1 or n < 4:
        raise ValueError("Expected a square 1D/2D scalar field and n >= 4")
    old = u.shape[0]
    if old == n:
        return u.clone()
    coeff = torch.fft.fftn(u, norm="forward")
    out = torch.zeros((n,) * u.ndim, dtype=coeff.dtype, device=u.device)
    k = torch.fft.fftfreq(old, d=1 / old, device=u.device).to(torch.int64)
    selected = k[torch.abs(k) < min(old, n) / 2]
    src = torch.remainder(selected, old)
    dst = torch.remainder(selected, n)
    if u.ndim == 1:
        out[dst] = coeff[src]
    else:
        out[dst[:, None], dst[None, :]] = coeff[src[:, None], src[None, :]]
    return torch.fft.ifftn(out, norm="forward").real


def product(*fields: torch.Tensor, dealiased: bool = True) -> torch.Tensor:
    """Project a quadratic/cubic product after enough zero padding.

    Padding 2N is used for both degrees, with Nyquist inputs excluded.  This
    evaluates the finite-band Galerkin product, not the infinite-band exact
    logistic flow.  The latter remains a deliberately separate split control.
    """
    if not fields or len(fields) > 3 or any(x.shape != fields[0].shape for x in fields):
        raise ValueError("One to three identically shaped fields are required")
    if not dealiased:
        return torch.stack(fields).prod(dim=0)
    n = fields[0].shape[0]
    padded = [resample(x, 2 * n) for x in fields]
    return resample(torch.stack(padded).prod(dim=0), n)


def wavevectors(n: int, dimension: int, *, device=None, dtype=torch.float64):
    k = 2 * math.pi * torch.fft.fftfreq(n, d=1 / n, device=device, dtype=dtype)
    return torch.meshgrid(*([k] * dimension), indexing="ij")


def diffusion_symbol(n: int, tensor, *, device=None, dtype=torch.float64):
    tensor = torch.as_tensor(tensor, device=device, dtype=dtype)
    if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
        raise ValueError("Diffusion tensor must be square")
    if not torch.allclose(tensor, tensor.T) or torch.linalg.eigvalsh(tensor).min() < 0:
        raise ValueError("Diffusion tensor must be symmetric positive semidefinite")
    waves = wavevectors(n, tensor.shape[0], device=device, dtype=dtype)
    return -sum(tensor[i, j] * waves[i] * waves[j]
                for i in range(len(waves)) for j in range(len(waves)))


def transport(u: torch.Tensor, h: float, symbol: torch.Tensor) -> torch.Tensor:
    return torch.fft.ifftn(torch.fft.fftn(u) * torch.exp(h * symbol)).real


def logistic(u: torch.Tensor, h: float, reaction: float) -> torch.Tensor:
    decay = math.exp(-reaction * h)
    return u / (u + (1 - u) * decay)


def fractional_filter(u: torch.Tensor, length: float, order: float = 1.5) -> torch.Tensor:
    """Physical-frequency regularizer (1 + (ell |xi|)^beta)^-1.

    ell is a physical length on the unit torus, never a grid-index scale.
    Constants are preserved; the zero-mean derivative feature is exposed below.
    """
    if length < 0 or not 0 < order <= 2:
        raise ValueError("Require ell >= 0 and 0 < beta <= 2")
    waves = wavevectors(u.shape[0], u.ndim, device=u.device, dtype=u.dtype)
    radius = sum(k.square() for k in waves).sqrt()
    multiplier = 1 / (1 + (length * radius).pow(order))
    return torch.fft.ifftn(torch.fft.fftn(u) * multiplier).real


def fractional_feature(u: torch.Tensor, order: float = 1.5) -> torch.Tensor:
    waves = wavevectors(u.shape[0], u.ndim, device=u.device, dtype=u.dtype)
    multiplier = sum(k.square() for k in waves).pow(order / 2)
    return torch.fft.ifftn(torch.fft.fftn(u) * multiplier).real


def _rk4(u, h, rhs):
    a = rhs(u)
    b = rhs(u + h * a / 2)
    c = rhs(u + h * b / 2)
    d = rhs(u + h * c)
    return u + h * (a + 2 * b + 2 * c + d) / 6


def logistic_split(u, h, symbol, reaction, *, dealiased=True):
    """Diffusion-first split for the declared spectral Galerkin target.

    The dealiased arm integrates the Galerkin reaction by four RK4 substeps;
    exact nodal logistic is a different (aliasing) control.  Stage cost includes
    these evaluations.  No claim that padding the exact logistic map suffices.
    """
    v = transport(u, h / 2, symbol)
    if dealiased:
        rhs = lambda x: reaction * (x - product(x, x))
        for _ in range(4):
            v = _rk4(v, h / 4, rhs)
    else:
        v = logistic(v, h, reaction)
    return transport(v, h / 2, symbol)


def quadratic_defect(u, h, symbol, reaction, *, dealiased=True):
    """Two-node finite-h exact quadratic DF defect about the logistic mean.

    Quadrature approximates only its scalar-time integral.  The zero-subtracted
    integrand enforces h=0, reaction=0 and diffusion=0 limits.  Unlike a midpoint
    rule, two Gauss nodes can represent a nonzero DF defect.  This expansion is
    quadratic in deviations, not exact at finite amplitude.
    """
    if h == 0 or reaction == 0 or not bool(torch.any(symbol != 0)):
        return torch.zeros_like(u)
    c = u.mean()
    v = u - c
    z = math.exp(-reaction * h)
    qh = z / (c + (1 - c) * z).square()
    midpoint = transport(product(transport(v, h / 2, symbol),
                                 transport(v, h / 2, symbol), dealiased=dealiased),
                         h / 2, symbol)
    value = torch.zeros_like(u)
    for node in (-1 / math.sqrt(3), 1 / math.sqrt(3)):
        s = h * (1 + node) / 2
        zs = math.exp(-reaction * s)
        qs = zs / (c + (1 - c) * zs).square()
        dv = transport(v, s, symbol)
        integrand = transport(product(dv, dv, dealiased=dealiased), h - s, symbol)
        value = value + (h / 2) * qs * (integrand - midpoint)
    return -reaction * qh * value


def c4_logistic_step(u, h, symbol, reaction, *, dealiased=True,
                     fractional_length=0.015, order=1.5, guard=True):
    base = logistic_split(u, h, symbol, reaction, dealiased=dealiased)
    correction = quadratic_defect(u, h, symbol, reaction, dealiased=dealiased)
    candidate = base + fractional_filter(correction, fractional_length, order)
    valid = bool(torch.isfinite(candidate).all() and candidate.min() >= 0 and candidate.max() <= 1)
    return (candidate if valid or not guard else base), (not valid and guard)


@dataclass(frozen=True)
class FieldSpec:
    cluster: int
    seed: int
    variant: str
    amplitude: float = 0.055
    frequency_scale: int = 1
    kappa: float = 0.012
    reaction: float = 2.0
    horizon_scale: float = 1.0
    rotated: bool = False
    changed_factors: tuple[str, ...] = ()


def factor_bank(clusters=2, seed=962701):
    """Independent random phase clusters with explicit paired factor changes."""
    if clusters < 1:
        raise ValueError("At least one independent cluster is required")
    specs = []
    variants = {
        "normal": {},
        "amplitude": {"amplitude": 0.09},
        "roughness_variance_matched": {"frequency_scale": 2},
        "diffusion_extrapolation": {"kappa": 0.05},
        "reaction_extrapolation": {"reaction": 6.0},
        "longer_horizon": {"horizon_scale": 3.0},
        "rotation": {"rotated": True},
        "amplitude_roughness": {"amplitude": 0.09, "frequency_scale": 2},
    }
    for cluster in range(clusters):
        for variant, edits in variants.items():
            specs.append(FieldSpec(cluster, seed + 104729 * cluster, variant,
                                   changed_factors=tuple(edits), **edits))
    return specs


def sample_field(spec: FieldSpec, n: int, *, device=None):
    """Evaluate the exact same continuous trigonometric field on each grid."""
    phase = np.random.default_rng(spec.seed).uniform(0, 2 * math.pi, 3)
    x = torch.arange(n, dtype=torch.float64, device=device) / n
    x, y = torch.meshgrid(x, x, indexing="ij")
    if spec.rotated:
        x, y = y, -x
    modes = ((1, 0, 1.0), (0, 1, 0.7), (1, -1, 0.4))
    result = torch.zeros_like(x)
    norm = math.sqrt(sum(a * a for _, _, a in modes) / 2)
    for p, (kx, ky, amplitude) in zip(phase, modes):
        result += amplitude * torch.cos(2 * math.pi * spec.frequency_scale * (kx * x + ky * y) + p)
    return 0.45 + spec.amplitude * result / norm


def _numpy_resample(u, n):
    """Independent NumPy implementation for the coupled reference RHS."""
    old = u.shape[0]
    if old == n:
        return u.copy()
    coeff = np.fft.fftn(u, norm="forward")
    out = np.zeros((n,) * u.ndim, dtype=np.complex128)
    modes = np.fft.fftfreq(old, d=1 / old).astype(int)
    selected = modes[np.abs(modes) < min(old, n) / 2]
    src, dst = selected % old, selected % n
    if u.ndim == 1:
        out[dst] = coeff[src]
    else:
        out[np.ix_(dst, dst)] = coeff[np.ix_(src, src)]
    return np.fft.ifftn(out, norm="forward").real


def _numpy_product(*fields):
    n = fields[0].shape[0]
    return _numpy_resample(np.prod([_numpy_resample(x, 2 * n) for x in fields], axis=0), n)


def independent_reference(initial, horizon, rhs, *, tolerance=2e-9, budget=None):
    """Two independent adaptive coupled solves, with finer time tolerance.

    Return the tight solution and a conservative observed difference estimate.
    The reference solves are independent of the split implementation but share
    the deliberately specified spatial generator.  This is no spatial bound.
    """
    initial = np.asarray(initial, dtype=np.float64)
    shape = initial.shape
    calls = 0
    def fun(t, flat):
        nonlocal calls
        calls += 1
        if budget is not None and calls % 32 == 0:
            budget.check()
        return np.asarray(rhs(flat.reshape(shape))).ravel()
    outputs, evaluations = [], []
    for factor in (1.0, 0.1):
        solve = solve_ivp(fun, (0, horizon), initial.ravel(), method="DOP853",
                         rtol=tolerance * factor, atol=tolerance * factor * 0.05,
                         max_step=max(horizon / 8, 1e-12))
        if not solve.success or not np.isfinite(solve.y[:, -1]).all():
            raise RuntimeError(f"Portability teacher failed: {solve.message}")
        outputs.append(solve.y[:, -1].reshape(shape))
        evaluations.append(solve.nfev)
    discrepancy = outputs[1] - outputs[0]
    return outputs[1], {
        "uncertainty_rms": float(2 * np.sqrt(np.mean(discrepancy ** 2))),
        "uncertainty_max": float(2 * np.max(np.abs(discrepancy))),
        "rhs_evaluations": sum(evaluations), "method": "independent_numpy_DOP853",
        "uncertainty_kind": "observed_time_refinement_estimate_not_certificate",
        "spatial_target": "same_generator_semidiscrete",
    }


def periodic_logistic_rhs(symbol, reaction):
    symbol = np.asarray(symbol)
    return lambda u: np.fft.ifftn(symbol * np.fft.fftn(u)).real + reaction * (u - _numpy_product(u, u))


def coupled_rhs(diffusivities=(0.015, 0.006), feed=0.04, kill=0.06):
    def rhs(u):
        n = u.shape[-1]
        k = 2 * math.pi * np.fft.fftfreq(n, d=1 / n)
        nonlinear = _numpy_product(u[0], u[1], u[1])
        a = np.fft.ifft(-diffusivities[0] * k * k * np.fft.fft(u[0])).real - nonlinear + feed * (1 - u[0])
        b = np.fft.ifft(-diffusivities[1] * k * k * np.fft.fft(u[1])).real + nonlinear - (feed + kill) * u[1]
        return np.stack((a, b))
    return rhs


def coupled_split(u, h, diffusivities=(0.015, 0.006), feed=0.04, kill=0.06, *, dealiased=True):
    n = u.shape[-1]
    k = wavevectors(n, 1, device=u.device, dtype=u.dtype)[0]
    v = torch.stack([transport(u[j], h / 2, -d * k * k) for j, d in enumerate(diffusivities)])
    def reaction(x):
        nonlinear = product(x[0], x[1], x[1], dealiased=dealiased)
        return torch.stack((-nonlinear + feed * (1 - x[0]), nonlinear - (feed + kill) * x[1]))
    for _ in range(4):
        v = _rk4(v, h / 4, reaction)
    return torch.stack([transport(v[j], h / 2, -d * k * k) for j, d in enumerate(diffusivities)])


def burgers_rhs(viscosity=0.02, advection=0.3):
    def rhs(u):
        n = u.size
        k = 2 * math.pi * np.fft.fftfreq(n, d=1 / n)
        linear = -viscosity * k * k - 1j * advection * k
        return np.fft.ifft(linear * np.fft.fft(u) - 0.5j * k * np.fft.fft(_numpy_product(u, u))).real
    return rhs


def burgers_split(u, h, viscosity=0.02, advection=0.3, *, dealiased=True):
    k = wavevectors(u.numel(), 1, device=u.device, dtype=u.dtype)[0]
    symbol = -viscosity * k * k - 1j * advection * k
    v = transport(u, h / 2, symbol)
    def nonlinear(x):
        return torch.fft.ifft(-0.5j * k * torch.fft.fft(product(x, x, dealiased=dealiased))).real
    for _ in range(4):
        v = _rk4(v, h / 4, nonlinear)
    return transport(v, h / 2, symbol)


def richardson_step(u, h, split: Callable, *, fractional_length=0.0, guard=None):
    """Measured extra-work control, not a free or learned correction.

    Richardson cancellation assumes smooth fixed-grid splitting error.  A
    fractional filter changes that leading cancellation and is an explicit
    ablation.  A guard may revert to the base without consulting a reference.
    """
    base = split(u, h)
    fine = split(split(u, h / 2), h / 2)
    delta = (4 / 3) * (fine - base)
    if u.ndim == 2:  # Species x 1D is filtered species-wise.
        delta = torch.stack([fractional_filter(x, fractional_length) for x in delta])
    else:
        delta = fractional_filter(delta, fractional_length)
    candidate = base + delta
    accepted = bool(torch.isfinite(candidate).all()) and (guard(candidate, u) if guard else True)
    return (candidate if accepted else base), not accepted


def geometry_operator(n, *, warp=0.25, diffusivity=0.02, variable=True):
    """Nonuniform conservative FV generator, homogeneous Dirichlet endpoints.

    M A is symmetric negative definite; A is Metzler.  The physical inner
    product uses dual-cell volumes M, not unweighted Euclidean geometry.
    """
    if n < 5 or abs(warp) >= 1 or diffusivity < 0:
        raise ValueError("Require n>=5, |warp|<1, diffusivity>=0")
    uniform = np.linspace(0, 1, n)
    nodes = uniform + warp * np.sin(2 * math.pi * uniform) / (2 * math.pi)
    spacing = np.diff(nodes)
    weights = (spacing[:-1] + spacing[1:]) / 2
    middle = (nodes[:-1] + nodes[1:]) / 2
    conductance = diffusivity * (1 + 0.3 * np.cos(2 * math.pi * middle) if variable else np.ones_like(middle)) / spacing
    matrix = np.diag(-(conductance[:-1] + conductance[1:]) / weights)
    matrix += np.diag(conductance[1:-1] / weights[:-1], 1)
    matrix += np.diag(conductance[1:-1] / weights[1:], -1)
    return nodes, weights, matrix


def geometry_split(u, h, matrix, reaction=2.0):
    heat = torch.matrix_exp(h / 2 * matrix)
    return heat @ logistic(heat @ u, h, reaction)


def weighted_filter(delta, matrix, weights, length=0.015, order=1.5):
    sqrtm = weights.sqrt()
    symmetric = sqrtm[:, None] * matrix / sqrtm[None, :]
    values, vectors = torch.linalg.eigh(symmetric)
    # Matrix eigenvalues include diffusivity.  Here the legacy argument name
    # ``length`` denotes a physical TIME scale tau; tau*(-lambda) is
    # dimensionless.  These eigenvalues are not Cartesian frequencies.
    multiplier = 1 / (1 + (length * (-values).clamp_min(0)).pow(order / 2))
    return (vectors @ (multiplier * (vectors.T @ (sqrtm * delta)))) / sqrtm


def geometry_corrected_step(u, h, matrix, weights, reaction=2.0, *, length=0.015):
    base = geometry_split(u, h, matrix, reaction)
    half = geometry_split(geometry_split(u, h / 2, matrix, reaction), h / 2, matrix, reaction)
    candidate = base + weighted_filter((4 / 3) * (half - base), matrix, weights, length)
    accepted = bool(torch.isfinite(candidate).all() and candidate.min() >= 0 and candidate.max() <= 1)
    return (candidate if accepted else base), not accepted


def _error(state, reference):
    difference = state.detach().cpu().numpy() - reference
    return float(np.sqrt(np.mean(difference * difference))), float(np.max(np.abs(difference)))


def _rollout(initial, horizon, steps, step):
    state = initial.clone()
    fallbacks = 0
    for _ in range(steps):
        result = step(state, horizon / steps)
        if isinstance(result, tuple):
            state, fallback = result
            fallbacks += int(fallback)
        else:
            state = result
    return state, fallbacks


def _artifact(ctx, name, initial, reference, outputs):
    target = Path(ctx.path) / (name.replace("/", "-") + ".npz")
    target.parent.mkdir(parents=True, exist_ok=True)
    values = {"initial": initial.detach().cpu().numpy(), "reference": reference}
    values.update({key: value.detach().cpu().numpy() for key, value in outputs.items()})
    np.savez_compressed(target, **values)
    return str(target.relative_to(ctx.path))


def run(ctx):
    """Run the bounded factor, anisotropy, geometry and cross-PDE ladder."""
    from .core import check
    settings = ctx.protocol.get("portability", {})
    grids = [int(x) for x in settings.get("grids", [12, 24])]
    clusters = int(settings.get("clusters", 2))
    horizon = float(settings.get("horizon", 0.04))
    tolerance = float(settings.get("teacher_tolerance", 2e-7))
    steps = int(settings.get("steps", 4))
    repeats = int(settings.get("timing_repeats", 1))
    cohort_seed = int(settings.get("seed", 962701 + {"smoke": 0, "development": 10000000, "full": 20000000}[ctx.protocol["profile"]]))
    if min(grids) < 10 or len(set(grids)) < 2 or steps < 1 or horizon <= 0 or tolerance <= 0:
        raise ValueError("Portability needs two distinct grids >=10 and positive budgets")
    device = torch.device(ctx.device)
    records = 0
    field_rows = []
    for spec in factor_bank(clusters, cohort_seed):
        ctx.budget.check()
        for n in grids:
            ctx.budget.check()
            initial = sample_field(spec, n, device=device)
            tensor = torch.tensor([[spec.kappa, 0.2 * spec.kappa],
                                   [0.2 * spec.kappa, 0.45 * spec.kappa]], dtype=torch.float64, device=device)
            if spec.rotated:
                rotation = torch.tensor([[0., 1.], [-1., 0.]], dtype=torch.float64, device=device)
                tensor = rotation.T @ tensor @ rotation
            symbol = diffusion_symbol(n, tensor, device=device)
            end = horizon * spec.horizon_scale
            reference, uncertainty = independent_reference(initial.cpu().numpy(), end,
                periodic_logistic_rhs(symbol.cpu().numpy(), spec.reaction), tolerance=tolerance / 10, budget=ctx.budget)
            accepted = uncertainty["uncertainty_max"] <= tolerance
            outputs = {}
            base_error = None
            base_cost = None
            arms = {
                "base_galerkin": lambda u, h: logistic_split(u, h, symbol, spec.reaction),
                "nodal_unfiltered": lambda u, h: c4_logistic_step(u, h, symbol, spec.reaction,
                    dealiased=False, fractional_length=0, guard=False),
                "dealiased_unfiltered": lambda u, h: c4_logistic_step(u, h, symbol, spec.reaction,
                    fractional_length=0, guard=False),
                "dealiased_fractional": lambda u, h: c4_logistic_step(u, h, symbol, spec.reaction, guard=False),
                "grid_index_fractional": lambda u, h: c4_logistic_step(u, h, symbol, spec.reaction,
                    fractional_length=0.015 * min(grids) / n, guard=False),
                "c4_guarded": lambda u, h: c4_logistic_step(u, h, symbol, spec.reaction),
            }
            for arm, function in arms.items():
                ctx.budget.check()
                (state, fallback), cost = ctx.measure(lambda: _rollout(initial, end, steps, function), repeats=repeats)
                outputs[arm] = state
                rms, maximum = _error(state, reference)
                if base_error is None:
                    base_error = (rms, maximum)
                    base_cost = cost["median_seconds"]
                checks = [
                    check("reference_time_refinement", uncertainty["uncertainty_max"], tolerance, category="correctness"),
                    check("finite_output", bool(torch.isfinite(state).all()), True, "eq", category="correctness"),
                    check("invariant_interval_min", float(state.min()), -1e-10, "ge", category="math"),
                    check("invariant_interval_max", float(state.max()), 1 + 1e-10, category="math"),
                    check("same_step_max_no_harm", maximum, base_error[1] + 2 * tolerance,
                          applicable=accepted and arm != "base_galerkin", category="gap"),
                    check("accuracy_max", maximum + uncertainty["uncertainty_max"], 2e-4,
                          applicable=accepted, category="gap"),
                    check("complete_same_step_cost", cost["median_seconds"], base_cost,
                          applicable=arm != "base_galerkin", category="utility",
                          reason="Same-step cost only; additional accuracy may permit a different step count in a later work-precision gate"),
                ]
                config = {**asdict(spec), "grid": n, "steps": steps, "horizon": end,
                          "equation": "anisotropic_periodic_logistic_Galerkin", "diffusion_tensor": tensor.cpu().tolist(),
                          "arm": arm, "fractional_length": (0.015 * min(grids) / n if arm == "grid_index_fractional"
                              else 0.015 if "fractional" in arm or arm == "c4_guarded" else 0),
                          "fractional_order": 1.5, "trainable_parameters": 0,
                          "target": "semidiscrete_dealiased_spectral", "independent_unit": f"cluster-{spec.cluster}"}
                experiment = f"transfer/cluster-{spec.cluster}/{spec.variant}/n{n}/{arm}"
                ctx.record(experiment, ["M14", "M17", "M19", "M05", "M13", "M21"],
                    combination_ids=["C4"] if arm == "c4_guarded" else [],
                    metrics={**cost, **uncertainty, "parameters": 0, "error_rms": rms, "error_max": maximum,
                             "reference_accepted": accepted, "fallback_steps": fallback, "base_error_max": base_error[1],
                             "base_median_seconds": base_cost,
                             "cost_ratio_vs_base": cost["median_seconds"] / max(base_cost, 1e-15)},
                    checks=checks, config=config,
                    evidence={"scope": "numerical_single_equation_transfer_not_neural_superiority", "reference_independent_integrator": True})
                records += 1
            artifact = _artifact(ctx, f"cluster-{spec.cluster}-{spec.variant}-n{n}", initial, reference, outputs)
            field_rows.append({"cluster": spec.cluster, "variant": spec.variant, "grid": n, "artifact": artifact})
    # Functional field pairing and independent-cluster bookkeeping are measured
    # explicitly.  Paired resolution/factor rows never multiply the sample count.
    specs = factor_bank(clusters, cohort_seed)
    a = sample_field(specs[0], grids[0])
    b = sample_field(specs[0], grids[1])
    pairing = float((resample(a, grids[1]) - b).abs().max())
    rough = sample_field(next(x for x in specs if x.variant == "roughness_variance_matched"), grids[0])
    variance_error = float(abs(a.var(unbiased=False) - rough.var(unbiased=False)))
    ctx.record("transfer/factor-bank", ["M17"], metrics={"resampling_error": pairing,
        "variance_pairing_error": variance_error, "independent_clusters": clusters, "parameters": 0,
        "paired_rows": len(field_rows)}, checks=[check("same_continuous_field", pairing, 1e-11, category="correctness"),
        check("roughness_variance_matched", variance_error, 1e-11, category="math"),
        check("independent_cluster_seeds", len({x.seed for x in specs}), clusters, "eq", category="correctness")],
        config={"factors": [asdict(x) for x in specs], "grids": grids}, evidence={"artifacts": field_rows})
    records += 1
    records += _geometry_panel(ctx, grids, horizon, tolerance, steps, repeats, check)
    records += _pde_panel(ctx, grids, horizon, tolerance, steps, repeats, check)
    records += _covariance_panel(ctx, grids, horizon, check)
    return {"experiments": records, "independent_field_clusters": clusters,
            "scope": "bounded_numerical_transfer_controls; no cross-PDE trained-neural claim",
            "mechanisms": ["M14", "M17", "M19"], "combinations": ["C4"]}


def _geometry_panel(ctx, grids, horizon, tolerance, steps, repeats, check):
    count = 0
    quadrature_errors, analytic_errors = [], []
    for n in grids:
        ctx.budget.check()
        nodes, weights, matrix = geometry_operator(n + 1)
        initial = torch.tensor(0.45 * np.sin(math.pi * nodes[1:-1]), device=ctx.device, dtype=torch.float64)
        mat = torch.tensor(matrix, device=ctx.device, dtype=torch.float64)
        weight = torch.tensor(weights, device=ctx.device, dtype=torch.float64)
        reference, uncertainty = independent_reference(initial.cpu().numpy(), horizon,
            lambda u: matrix @ u + 2 * u * (1 - u), tolerance=tolerance / 10, budget=ctx.budget)
        accepted = uncertainty["uncertainty_max"] <= tolerance
        outputs = {}
        base_max = None
        base_cost = None
        for arm in ("base", "c4_weighted"):
            function = (lambda u, h: geometry_split(u, h, mat)) if arm == "base" else (
                lambda u, h: geometry_corrected_step(u, h, mat, weight))
            (state, fallback), cost = ctx.measure(lambda: _rollout(initial, horizon, steps, function), repeats=repeats)
            outputs[arm] = state
            rms, maximum = _error(state, reference)
            weighted_rms = float(np.sqrt(np.dot(weights, (state.cpu().numpy() - reference) ** 2) / weights.sum()))
            boundary_error = float(np.abs(np.pad(state.cpu().numpy(), (1, 1))[[0, -1]]).max())
            if base_max is None:
                base_max = maximum
                base_cost = cost["median_seconds"]
            symmetry = float(np.max(np.abs(weights[:, None] * matrix - matrix.T * weights[None, :])))
            offdiag = matrix - np.diag(np.diag(matrix))
            quadrature_error = abs(float(np.dot(weights, nodes[1:-1] * (1 - nodes[1:-1]))) - 1 / 6)
            ctx.record(f"geometry/nonuniform-dirichlet/n{n}/{arm}", ["M14", "M19", "M13", "M21"],
                combination_ids=["C4"] if arm == "c4_weighted" else [], metrics={**cost, **uncertainty,
                    "parameters": 0, "error_rms": rms, "error_weighted_rms": weighted_rms, "error_max": maximum,
                    "reference_accepted": accepted, "fallback_steps": fallback,
                    "base_median_seconds": base_cost,
                    "cost_ratio_vs_base": cost["median_seconds"] / max(base_cost, 1e-15),
                    "weighted_symmetry_error": symmetry, "quadrature_error": quadrature_error},
                checks=[check("weighted_self_adjointness", symmetry, 1e-12, category="math"),
                    check("nonnegative_offdiagonals", float(offdiag.min()), -1e-12, "ge", category="math"),
                    check("teacher_refinement", uncertainty["uncertainty_max"], tolerance, category="correctness"),
                    check("homogeneous_boundary_lifting", boundary_error, 0.0, "eq", category="math",
                          evidence_kind="structural_identity_by_construction",
                          reason="Reconstructed boundary values are fixed zero; boundary degrees of freedom are eliminated"),
                    check("same_step_no_harm", maximum, base_max + 2 * tolerance,
                          applicable=accepted and arm != "base", category="gap"),
                    check("accuracy_max", maximum + uncertainty["uncertainty_max"], 2e-4,
                          applicable=accepted, category="gap"),
                    check("interval_min", float(state.min()), -1e-10, "ge", category="math"),
                    check("interval_max", float(state.max()), 1 + 1e-10, category="math"),
                    check("complete_same_step_cost", cost["median_seconds"], base_cost,
                          applicable=arm != "base", category="utility")],
                config={"equation": "variable_coefficient_logistic_RD", "boundary": "homogeneous_Dirichlet",
                    "geometry": "warped_1d_finite_volume", "grid_intervals": n, "warp": 0.25,
                    "diffusivity": "0.02*(1+0.3*cos(2*pi*x))", "reaction": 2.0, "steps": steps,
                    "fractional_generator_time_scale": 0.015 if arm == "c4_weighted" else 0,
                    "fractional_generator_filter": "1/(1+(tau*(-lambda))**(beta/2)); tau has time units",
                    "horizon": horizon, "target": "weighted_semidiscrete_finite_volume", "trainable_parameters": 0},
                evidence={"physical_measure": "dual_cell_volumes", "no_periodic_Fourier_transplant": True})
            count += 1
        _artifact(ctx, f"geometry-n{n}", initial, reference, outputs)
        quadrature_errors.append(quadrature_error)
        nodes_heat, _, matrix_heat = geometry_operator(n + 1, diffusivity=0.04, variable=False)
        initial_heat = torch.tensor(np.sin(math.pi * nodes_heat[1:-1]), dtype=torch.float64, device=ctx.device)
        actual_heat = geometry_split(initial_heat, horizon, torch.tensor(matrix_heat, dtype=torch.float64, device=ctx.device), reaction=0)
        analytic_heat = math.exp(-0.04 * math.pi ** 2 * horizon) * initial_heat
        analytic_errors.append(float((actual_heat - analytic_heat).abs().max()))
    order_quadrature = math.log(quadrature_errors[0] / quadrature_errors[-1]) / math.log(grids[-1] / grids[0])
    order_heat = math.log(analytic_errors[0] / analytic_errors[-1]) / math.log(grids[-1] / grids[0])
    ctx.record("geometry/independent-remeshing-and-quadrature", ["M14", "M19"],
        metrics={"parameters": 0, "quadrature_errors": quadrature_errors, "analytic_heat_max_errors": analytic_errors,
                 "observed_quadrature_order": order_quadrature, "observed_heat_order": order_heat},
        checks=[check("physical_measure_refinement", order_quadrature, 1.7, "ge", category="math"),
            check("continuous_solution_remeshing", order_heat, 1.5, "ge", category="gap"),
            check("arbitrary_geometry_guarantee", None, None, applicable=False, required=False, category="math",
                  reason="Two resolved warped 1D grids do not establish arbitrary remeshing robustness")],
        config={"grids": grids, "warp": 0.25, "diffusivity": 0.04, "horizon": horizon,
                "boundary": "homogeneous_Dirichlet", "analytic_solution": "exp(-0.04*pi^2*t)*sin(pi*x)",
                "quadrature_integrand": "x*(1-x)", "exact_integral": 1 / 6},
        evidence={"target": "analytic_continuum_heat_solution", "fields_sampled_from_same_continuous_function": True})
    return count + 1


def _pde_panel(ctx, grids, horizon, tolerance, steps, repeats, check):
    count = 0
    for n in grids:
        x = torch.arange(n, dtype=torch.float64, device=ctx.device) / n
        initial_bank = {
            "coupled_gray_scott": torch.stack((0.75 + 0.04 * torch.cos(2 * math.pi * x),
                                               0.2 + 0.03 * torch.sin(4 * math.pi * x))),
            "viscous_burgers_advection": 0.2 + 0.12 * torch.sin(2 * math.pi * x) + 0.04 * torch.cos(4 * math.pi * x),
        }
        for equation, initial in initial_bank.items():
            ctx.budget.check()
            coupled = equation == "coupled_gray_scott"
            rhs = coupled_rhs() if coupled else burgers_rhs()
            split = coupled_split if coupled else burgers_split
            guard = (lambda state, old: bool(state.min() >= 0 and state.max() <= 1.2)) if coupled else (
                lambda state, old: bool(state.square().mean() <= old.square().mean() + 1e-12
                                       and abs(float(state.mean() - old.mean())) <= 1e-11))
            reference, uncertainty = independent_reference(initial.cpu().numpy(), horizon, rhs,
                tolerance=tolerance / 10, budget=ctx.budget)
            accepted = uncertainty["uncertainty_max"] <= tolerance
            base_max = None
            base_cost = None
            outputs = {}
            for arm in ("base_dealiased", "unfiltered_correction", "c4_fractional_guarded"):
                if arm == "base_dealiased":
                    function = split
                else:
                    length = 0.015 if arm == "c4_fractional_guarded" else 0.0
                    function = lambda u, h, length=length: richardson_step(u, h, split, fractional_length=length,
                        guard=guard if arm == "c4_fractional_guarded" else None)
                (state, fallback), cost = ctx.measure(lambda: _rollout(initial, horizon, steps, function), repeats=repeats)
                outputs[arm] = state
                rms, maximum = _error(state, reference)
                if base_max is None:
                    base_max = maximum
                    base_cost = cost["median_seconds"]
                conservation = float(abs(state.mean() - initial.mean())) if not coupled else None
                # Cubic exchange cancels between species exactly; feed/kill
                # means total species mass is not conserved.
                derivative = rhs(initial.cpu().numpy())
                if coupled:
                    exact_rate = 0.04 * (1 - initial[0].mean()) - 0.1 * initial[1].mean()
                    balance_error = abs(float(derivative.sum(axis=0).mean()) - float(exact_rate))
                else:
                    balance_error = abs(float(derivative.mean()))
                ctx.record(f"pde/{equation}/n{n}/{arm}", ["M05", "M13", "M17", "M19", "M21"],
                    combination_ids=["C4"] if arm == "c4_fractional_guarded" else [], metrics={**cost, **uncertainty,
                        "parameters": 0, "error_rms": rms, "error_max": maximum, "fallback_steps": fallback,
                        "reference_accepted": accepted,
                        "base_median_seconds": base_cost,
                        "cost_ratio_vs_base": cost["median_seconds"] / max(base_cost, 1e-15),
                        "mean_conservation_error": conservation, "instantaneous_balance_error": balance_error},
                    checks=[check("teacher_refinement", uncertainty["uncertainty_max"], tolerance, category="correctness"),
                        check("finite_output", bool(torch.isfinite(state).all()), True, "eq", category="correctness"),
                        check("instantaneous_physical_balance", balance_error, 1e-12, category="math"),
                        check("mean_conservation", conservation, 1e-11, applicable=not coupled, category="math",
                              reason="Gray–Scott mean changes through feed and kill" if coupled else "periodic divergence flux"),
                        check("energy_dissipation", float(state.square().mean()), float(initial.square().mean()) + 1e-11,
                              applicable=not coupled, category="math"),
                        check("same_step_no_harm", maximum, base_max + 2 * tolerance,
                              applicable=accepted and arm != "base_dealiased", category="gap"),
                        check("accuracy_max", maximum + uncertainty["uncertainty_max"], 2e-4,
                              applicable=accepted, category="gap"),
                        check("complete_same_step_cost", cost["median_seconds"], base_cost,
                              applicable=arm != "base_dealiased", category="utility")],
                    config={"equation": equation, "grid": n, "steps": steps, "horizon": horizon,
                        "boundary": "periodic", "target": "dealiased_spectral_Galerkin", "trainable_parameters": 0,
                        "parameters": {"diffusivities": [0.015, 0.006], "feed": 0.04, "kill": 0.06} if coupled
                        else {"viscosity": 0.02, "advection": 0.3},
                        "fractional_length": 0.015 if arm == "c4_fractional_guarded" else 0,
                        "extra_work": "three split maps per correction; four reaction RK4 substeps per split"},
                    evidence={"scope": "numerical_transfer_only_not_neural_superiority"})
                count += 1
            _artifact(ctx, f"pde-{equation}-n{n}", initial, reference, outputs)
    return count


def _covariance_panel(ctx, grids, horizon, check):
    n = grids[0]
    field = sample_field(FieldSpec(0, 821047, "covariance"), n, device=ctx.device)
    tensor = torch.tensor([[0.02, 0.004], [0.004, 0.01]], dtype=torch.float64, device=ctx.device)
    rotation = torch.tensor([[0., 1.], [-1., 0.]], dtype=torch.float64, device=ctx.device)
    symbol = diffusion_symbol(n, tensor, device=ctx.device)
    # u(y,-x): transpose then reverse periodic indices of the second coordinate.
    rotate = lambda u: u.T[:, torch.remainder(-torch.arange(n, device=u.device), n)]
    rotated_tensor = rotation.T @ tensor @ rotation
    rotated_symbol = diffusion_symbol(n, rotated_tensor, device=ctx.device)
    result, _ = c4_logistic_step(field, horizon, symbol, 2.0)
    rotated, _ = c4_logistic_step(rotate(field), horizon, rotated_symbol, 2.0)
    covariance = float((rotated - rotate(result)).abs().max())
    shifted, _ = c4_logistic_step(torch.roll(field, (2, 3), (0, 1)), horizon, symbol, 2.0)
    translation = float((shifted - torch.roll(result, (2, 3), (0, 1))).abs().max())
    perturbation = 1e-6 * torch.cos(2 * math.pi * torch.arange(n, device=ctx.device, dtype=torch.float64) / n)[:, None]
    perturbed, _ = c4_logistic_step(field + perturbation, horizon, symbol, 2.0)
    ratio = float((perturbed - result).norm() / perturbation.expand_as(field).norm())
    # Spectral Galerkin is not automatically an invariant Metzler discretization;
    # report the PDE comparison as a sampled diagnostic, not a certified bound.
    ctx.record("transfer/covariance-and-growth", ["M14", "M21"], combination_ids=["C4"],
        metrics={"parameters": 0, "tensor_rotation_error": covariance, "translation_error": translation,
                 "sampled_growth_ratio": ratio, "physical_comparison_envelope": math.exp(2 * horizon)},
        checks=[check("tensor_rotation_covariance", covariance, 2e-11, category="math"),
            check("translation_covariance", translation, 2e-11, category="math"),
            check("sampled_growth_envelope", ratio, math.exp(2 * horizon) + 1e-5, category="math", required=False),
            check("global_stability_proof", None, None, applicable=False, category="math",
                  reason="Sampled directional amplification does not prove a global bound")],
        config={"grid": n, "horizon": horizon, "reaction": 2.0, "perturbation_amplitude": 1e-6},
        evidence={"evidence_kind": "numerical_covariance_and_directional_growth"})
    return 1
