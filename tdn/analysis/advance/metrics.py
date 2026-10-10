"""Traceable diagnostics and inference measurement for the advance campaign.

Metrics do not assert conservation, accept a teacher implicitly, or turn grid
points into independent statistical samples. Evaluation runs on detached FP64
CPU values; its transfers and reductions are outside inference timing.
"""
from __future__ import annotations

import math
import time

import torch

from tdn.analysis.frontier.measurement import measure_paired


PHYSICAL_BAND_EDGES = (2.0, 8.0)  # cycles per physical length, independent of N


def _finite_json(result):
    """Make arithmetic overflow explicit; never serialize NaN or infinity."""
    bad = False
    def visit(value):
        nonlocal bad
        if isinstance(value,float) and not math.isfinite(value):
            bad=True
            return None
        if isinstance(value,dict): return {k:visit(v) for k,v in value.items()}
        if isinstance(value,list): return [visit(v) for v in value]
        return value
    answer=visit(result)
    answer['metrics_finite']=not bad
    answer['metric_status']='NONFINITE_DERIVED_METRICS' if bad else 'MEASURED' if answer['finite'] else 'NONFINITE_PREDICTION'
    return answer


def _rms(value):
    return float(value.square().mean().sqrt())


def physical_inner_product(a, b, geom, track):
    """Domain-average inner product consistent with the actual spatial basis.

    Even-grid real Fourier interpolation splits each Nyquist cosine between
    signed frequencies. Its continuous squared norm is half its nodal squared
    norm per Nyquist axis. Ignoring these mass weights breaks the Galerkin
    gradient-flow identity for fields carrying Nyquist content.
    """
    if a.shape != b.shape or tuple(a.shape[2:]) != tuple(geom.grid):
        raise ValueError("Inner product fields must match the geometry")
    if track == "discrete": return (a*b).mean()
    if track != "continuum": raise ValueError("Explicit spatial target required")
    weights = torch.ones(geom.grid,dtype=a.dtype,device=a.device)
    for axis,n in enumerate(geom.grid):
        if n%2==0:
            shape=[1]*geom.ndim;shape[axis]=n
            mass=torch.ones(n,dtype=a.dtype,device=a.device);mass[n//2]=.5
            weights=weights*mass.reshape(shape)
    axes=tuple(range(2,a.ndim))
    fa=torch.fft.fftn(a,dim=axes,norm="ortho")
    fb=torch.fft.fftn(b,dim=axes,norm="ortho")
    return (fa.conj()*fb*weights).real.mean()


def energy_value(u, eq, geom, track):
    """Differentiable mean energy of the specified semidiscrete gradient flow.

    E = -<u,L u>/2 - r<u²>/2 + r<u B(u,u)>/3, where B is nodal
    multiplication or the dealiased Galerkin product. Brackets use nodal mean
    for FD and the continuous Fourier-interpolant inner product for Galerkin,
    including even-grid Nyquist mass weights. Thus dE/dt = -<rhs,rhs>; nodal
    mean(rhs²) is correct only for FD or absent Nyquist content. This identity
    describes exact semidiscrete dynamics, not a certificate for a learned step.
    """
    from tdn.analysis.frontier.numerics import validate, apply_multiplier, diffusion_symbol, product
    validate(u,0.,geom,track)
    diffusion = apply_multiplier(u,diffusion_symbol(u,eq,geom,track))
    return (-.5*physical_inner_product(u,diffusion,geom,track)
            -.5*eq.reaction_rate*physical_inner_product(u,u,geom,track)
            +(eq.reaction_rate/3.)*physical_inner_product(u,product(u,u,track),geom,track))


def _state_diagnostics(value, geometry, track):
    axes = tuple(range(2, value.ndim))
    means = value.mean(axes, keepdim=True)
    gradient_energy = torch.zeros_like(value)
    if track == "discrete":
        # Forward-edge differences give the exact quadratic form of periodic
        # central-difference diffusion, including its Nyquist contribution.
        for axis, dx in zip(axes, geometry.dx):
            gradient_energy += ((torch.roll(value, -1, axis) - value) / dx).square()
        gradient_squared = float(gradient_energy.mean())
    else:
        transformed = torch.fft.fftn(value, dim=axes, norm="ortho")
        wave2 = torch.zeros(geometry.grid, dtype=value.dtype)
        for i, (n, dx) in enumerate(zip(geometry.grid, geometry.dx)):
            freq = 2 * math.pi * torch.fft.fftfreq(n, d=dx, dtype=value.dtype)
            shape = [1] * len(axes); shape[i] = n
            wave2 += freq.square().reshape(shape)
        mass = torch.ones(geometry.grid,dtype=value.dtype)
        for axis,n in enumerate(geometry.grid):
            if n%2==0:
                one=torch.ones(n,dtype=value.dtype);one[n//2]=.5
                shape=[1]*geometry.ndim;shape[axis]=n
                mass*=one.reshape(shape)
        gradient_squared = float((transformed.abs().square() * wave2 * mass).mean())
    below = int((value < 0).sum()); above = int((value > 1).sum())
    return dict(mean=float(value.mean()), variance=float((value-means).square().mean()),
                l2_energy_density=float(value.square().mean()),
                moment_scope="Sampled grid-point mean/variance/L2; physical interpolant moments are also supplied and can differ at even-grid Nyquist modes",
                physical_variance=float(physical_inner_product(value-means,value-means,geometry,track)),
                physical_l2_energy_density=float(physical_inner_product(value,value,geometry,track)),
                gradient_squared_mean=gradient_squared,
                gradient_scope="FD forward-edge diffusion quadratic form; continuum physical Fourier-interpolant gradient norm including Nyquist mass weights",
                minimum=float(value.min()), maximum=float(value.max()),
                below_zero_count=below, above_one_count=above,
                physical_interval_violations=below+above,
                physical_interval_violation_fraction=(below+above)/value.numel(),
                energy_semantics="L2 signal energy and diffusion quadratic form; neither is declared conserved under logistic reaction")


def endpoint_metrics(pred, target, initial, eq, geom, track, reference=None):
    """Endpoint errors, physical-frequency bands and state diagnostics.

    `reference` uses accepted/uncertainty_rms/uncertainty_max_bound from teacher
    metadata. Optional `time_derivative` is the actual model derivative at this
    endpoint, not a finite endpoint difference. Its absence leaves mean-balance
    residual NA. Absolute units remain authoritative when relative scales vanish.
    """
    if track not in ("discrete", "continuum"):
        raise ValueError("Metrics require an explicit discrete or continuum track")
    if any(not isinstance(x, torch.Tensor) or x.is_complex() for x in (pred, target, initial)):
        raise TypeError("Metrics require real tensor fields")
    if pred.shape != target.shape or pred.shape != initial.shape or pred.ndim != geom.ndim+2 or tuple(pred.shape[2:]) != tuple(geom.grid):
        raise ValueError("Metric fields must have matching batch/channel/grid shapes")
    if not pred.numel(): raise ValueError("Empty metric fields are unsupported")
    if not all(math.isfinite(float(v)) and float(v) >= 0 for v in (eq.kappa, eq.reaction_rate)):
        raise ValueError("Finite nonnegative physical coefficients are required")
    p, y, u = [x.detach().to(device="cpu", dtype=torch.float64) for x in (pred, target, initial)]
    if not bool(torch.isfinite(y).all()) or not bool(torch.isfinite(u).all()):
        raise ValueError("Initial state and reference target must be finite")
    ref = reference or {}
    unc_rms, unc_max = ref.get("uncertainty_rms"), ref.get("uncertainty_max_bound")
    for value in (unc_rms, unc_max):
        if value is not None and (isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0):
            raise ValueError("Reference uncertainty must be nonnegative finite or unavailable")
    unc_rms = float(unc_rms) if unc_rms is not None else None
    unc_max = float(unc_max) if unc_max is not None else None
    result = dict(track=track, finite=bool(torch.isfinite(p).all()),
        reference_accepted=ref.get("accepted") is True,
        reference_uncertainty_rms=unc_rms, reference_uncertainty_max=unc_max,
        reference_status="UNAVAILABLE" if reference is None else "ACCEPTED" if ref.get("accepted") is True else "UNACCEPTED",
        uncertainty_is_certificate=False, grid=list(geom.grid), lengths=list(geom.lengths),
        batch_size=int(p.shape[0]), channels=int(p.shape[1]), spatial_values=int(p.numel()),
        statistical_unit="independent field; grid points/channels/repeated queries are not independent fields",
        mean_balance_residual=None, mean_balance_reason="No actual endpoint time derivative supplied",
        distribution_tail_scope="Spatial extrema within this endpoint only; no population p95/p99 or failure-probability guarantee")
    names = ("error_rms", "error_max", "relative_rms", "upper_rms", "upper_max", "centered_rms",
             "mean_error", "signed_mean_error", "mean_error_rms", "error_decomposition_residual",
             "minimum", "maximum", "physical_interval_violations", "expected_mean_rate")
    if not result["finite"]:
        result.update({name:None for name in names}, spectral_bands=[], prediction=None,
                      target=_state_diagnostics(y,geom,track), initial=_state_diagnostics(u,geom,track),
                      nonfinite_prediction_values=int((~torch.isfinite(p)).sum()))
        return _finite_json(result)
    axes = tuple(range(2, p.ndim)); error = p-y
    mean_error = error.mean(axes, keepdim=True)
    rms, maximum = _rms(error), float(error.abs().max())
    centered, mean_rms = _rms(error-mean_error), _rms(mean_error)
    truth_scale = _rms(y)
    # Unit carrying-capacity logistic fields: a 1e-12 absolute RMS floor is
    # explicit and prevents a near-zero reference being reported as infinity.
    denominator = max(truth_scale, 1e-12)
    result.update(error_rms=rms,error_max=maximum,relative_rms=rms/denominator,
        relative_rms_denominator=denominator,relative_rms_floor=1e-12,
        relative_rms_floor_active=truth_scale < 1e-12,
        relative_rms_scope="RMS error / max(RMS target, 1e-12 carrying-capacity units); absolute errors retained",
        upper_rms=rms+float(unc_rms) if unc_rms is not None else None,
        upper_max=maximum+float(unc_max) if unc_max is not None else None,
        centered_rms=centered, mean_error=float(mean_error.abs().max()),
        signed_mean_error=float(error.mean()), mean_error_rms=mean_rms,
        error_decomposition_residual=rms*rms-centered*centered-mean_rms*mean_rms)
    states = {name:_state_diagnostics(value,geom,track) for name,value in (("prediction",p),("target",y),("initial",u))}
    result.update(states)
    result.update({name:states["prediction"][name] for name in ("minimum","maximum","physical_interval_violations")})
    result["variance_error"] = states["prediction"]["variance"]-states["target"]["variance"]
    result["gradient_squared_error"] = states["prediction"]["gradient_squared_mean"]-states["target"]["gradient_squared_mean"]
    ei,ep,et = [float(energy_value(v,eq,geom,track)) for v in (u,p,y)]
    result.update(energy_initial=ei,energy_prediction=ep,energy_target=et,
        energy_change=ep-ei,energy_target_change=et-ei,energy_increase=max(0.,ep-ei),
        energy_error=ep-et,energy_absolute_error=abs(ep-et),
        energy_semantics="Domain-average Lyapunov energy for specified FD/nodal or dealiased Galerkin equation; exact flow derivative is minus squared RHS norm in the corresponding spatial inner product (Nyquist-aware continuous interpolant for Galerkin). Endpoint increase is a diagnostic, not a universal stability certificate.")
    radial2 = torch.zeros(geom.grid,dtype=p.dtype)
    for axis,(n,dx) in enumerate(zip(geom.grid,geom.dx)):
        freq = torch.fft.fftfreq(n,d=dx,dtype=p.dtype)
        shape = [1]*geom.ndim; shape[axis] = n
        radial2 += freq.square().reshape(shape)
    spectral = torch.fft.fftn(error,dim=axes,norm="ortho").abs().square()
    edges = (0.,*PHYSICAL_BAND_EDGES,None); bands=[]
    for i,(lower,upper) in enumerate(zip(edges,edges[1:])):
        mask = radial2 >= lower*lower
        if upper is not None: mask &= radial2 < upper*upper
        count = int(mask.sum())
        # Divide by all field points to retain an additive Parseval partition.
        contribution = float((spectral*mask).mean().sqrt()) if count else None
        bands.append(dict(label=("low","middle","high")[i],lower_cycles_per_length=lower,
            upper_cycles_per_length=upper,error_rms=contribution,mode_count=count,
            status="MEASURED" if count else "NA_NO_RESOLVED_MODES",
            normalization="RMS contribution on full grid; band squared contributions sum to total squared RMS"))
    result["spectral_bands"] = bands
    from tdn.analysis.frontier.numerics import reaction_rhs
    expected = reaction_rhs(p,eq,track).mean(axes)
    result["expected_mean_rate"] = float(expected.mean())
    result["mean_rate_semantics"] = "Periodic diffusion has zero spatial mean; logistic reaction generally changes it"
    derivative = ref.get("time_derivative")
    if derivative is not None:
        if not isinstance(derivative,torch.Tensor) or derivative.shape != pred.shape or derivative.is_complex():
            raise ValueError("Endpoint time derivative must match prediction shape")
        derivative = derivative.detach().to(device="cpu",dtype=torch.float64)
        if not bool(torch.isfinite(derivative).all()): raise ValueError("Endpoint time derivative must be finite")
        residual = derivative.mean(axes)-expected
        result.update(mean_balance_residual=float(residual.abs().max()),mean_balance_reason="Measured actual endpoint derivative minus logistic mean-rate identity")
    return _finite_json(result)


def measure_paired_metrics(calls, *, device="cpu", repeats=20, warmup=1, seed=0, budget=None, memory_probe=True):
    """Randomized synchronized complete calls plus separately charged memory probes.

    CUDA peaks are reset for each additional warm memory probe. Absolute peaks
    still include all live models/cached outputs in the process; incremental
    peaks subtract that model's pre-probe baseline and are not total deployment
    memory. Probe time is reported separately, never mixed into latency samples.
    """
    gpu = torch.device(device).type == "cuda"
    with torch.no_grad():
        answers, timing = measure_paired(calls,device=device,repeats=repeats,warmup=warmup,seed=seed,budget=budget)
    timing["first_invocation_scope"] = "First call in this group, warm process/libraries; excludes process/device initialization, model loading and transfers outside callable"
    timing["latency_scope"] = "Complete provided callable with synchronized device; safety checks inside callable remain charged"
    probe_total = 0.
    for name, method in timing["methods"].items():
        samples = sorted(method["samples_seconds"]); n=len(samples)
        method["tail_statistics"] = dict(sample_count=n,
            p95_seconds=samples[math.ceil(.95*n)-1] if n>=100 else None,
            p99_seconds=samples[math.ceil(.99*n)-1] if n>=1000 else None,
            minimum_samples_p95=100,minimum_samples_p99=1000,
            observed_max_seconds=max(samples),
            scope="Empirical quantiles only when minimally sampled; no tail-confidence or future-service guarantee. Legacy p95 field is merely an order statistic.")
        memory = dict(status="NA_CPU_ALLOCATOR_PEAK_UNAVAILABLE" if not gpu else "NOT_REQUESTED",
            absolute_peak_allocated_bytes=None,absolute_peak_reserved_bytes=None,
            incremental_peak_allocated_bytes=None,incremental_peak_reserved_bytes=None,
            baseline_allocated_bytes=None,baseline_reserved_bytes=None,probe_seconds=0.,
            scope="Separate warm probe; absolute includes process-resident models/caches/outputs; incremental excludes baseline, not complete deployment footprint")
        if gpu and memory_probe:
            if budget is not None: budget.check()
            torch.cuda.synchronize(device)
            allocated=torch.cuda.memory_allocated(device); reserved=torch.cuda.memory_reserved(device)
            torch.cuda.reset_peak_memory_stats(device); start=time.perf_counter()
            with torch.no_grad(): probe_answer=calls[name]()
            torch.cuda.synchronize(device); elapsed=time.perf_counter()-start
            peak_allocated=torch.cuda.max_memory_allocated(device); peak_reserved=torch.cuda.max_memory_reserved(device)
            del probe_answer
            memory.update(status="MEASURED_SEPARATE_PROBE",baseline_allocated_bytes=allocated,
                baseline_reserved_bytes=reserved,absolute_peak_allocated_bytes=peak_allocated,
                absolute_peak_reserved_bytes=peak_reserved,
                incremental_peak_allocated_bytes=max(0,peak_allocated-allocated),
                incremental_peak_reserved_bytes=max(0,peak_reserved-reserved),probe_seconds=elapsed)
            probe_total+=elapsed
            if budget is not None and hasattr(budget,"observe"): budget.observe(force=True)
        method["memory"] = memory
    timing["memory_probe_seconds"] = probe_total
    timing["energy_joules"] = None
    timing["monetary_cost"] = None
    return answers,timing
