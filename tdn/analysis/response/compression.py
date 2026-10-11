"""Train-only finite-rank source compression through a physical decoder.

This is an oracle-source representation diagnostic: sources and the mean path
come from a charged numerical teacher. It cannot establish inference speedup or
the learnability of those sources. No production tangent matrix is constructed.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from tdn.analysis.frontier.core import check, clean


def _filter(fields, multiplier, spatial_shape):
    shape = fields.shape
    x = fields.reshape(*shape[:-1], *spatial_shape)
    axes = tuple(range(x.ndim-len(spatial_shape), x.ndim))
    result = np.fft.ifftn(np.fft.fftn(x, axes=axes)*multiplier, axes=axes)
    if np.max(np.abs(result.imag), initial=0.) > 1e-8*max(1., np.max(np.abs(result.real), initial=0.)):
        raise ValueError("Metric must preserve real-valued conjugate symmetry")
    return result.real.reshape(shape)


def fit_pod(sources, rank, *, split, spatial_shape, metric=None):
    """Spatial POD shared by all training parents/times; metric is frozen first.

    A positive Fourier multiplier defines ||M^(1/2) F||. Weighted SVD uses the
    same rank and coefficient count as unweighted SVD. Metric weighting is an
    approximation to the full nonseparable response Gram, explicitly measured
    through the exact discrete decoder afterwards.
    """
    if split != "train":
        raise ValueError("Compression bases may be fitted on train parents only")
    source = np.asarray(sources, float)
    features = math.prod(spatial_shape)
    if source.ndim != 3 or source.shape[-1] != features or not np.isfinite(source).all():
        raise ValueError("Sources require finite [parents,times,spatial] arrays")
    if type(rank) is not int or not 1 <= rank <= min(source.shape[0]*source.shape[1], features):
        raise ValueError("Requested rank exceeds available training snapshots or spatial coordinates")
    metric = np.ones(spatial_shape) if metric is None else np.asarray(metric, float)
    if metric.shape != tuple(spatial_shape) or not np.isfinite(metric).all() or np.any(metric <= 0):
        raise ValueError("Compression metric must be strictly positive")
    start = time.perf_counter()
    root = np.sqrt(metric)
    transformed = _filter(source, root, spatial_shape).reshape(-1, features)
    _, singular, right = np.linalg.svd(transformed, full_matrices=False)
    vectors = right[:rank]
    encoder = _filter(vectors, root, spatial_shape).T
    decoder = _filter(vectors, 1/root, spatial_shape)
    return dict(encoder=encoder, decoder=decoder, rank=rank, spatial_shape=list(spatial_shape),
        metric=metric, fit_seconds=time.perf_counter()-start,
        singular_values=singular[:min(32, len(singular))], effective_training_rank=int(np.sum(singular > singular[0]*1e-10)) if singular[0] else 0,
        discarded_metric_energy_fraction=float(np.sum(singular[rank:]**2)/max(np.sum(singular**2), 1e-300)),
        training_sha256=hashlib.sha256(source.tobytes()).hexdigest(), training_split=split,
        parameterization="rank-J shared spatial basis; all time coefficients retained; no temporal-query caching claim")


def reconstruct(sources, model):
    return np.asarray(sources) @ model["encoder"] @ model["decoder"]


def whiten_channels(sources, initial, *, split, ridge=1e-6):
    """Train-only invertible channel reparameterization, not an accuracy gain.

    This control reveals scale/rotation nonidentifiability without increasing
    rank. Regularization makes low-energy channels finite and is reported.
    """
    if split != "train" or initial.get("training_split") != "train":
        raise ValueError("Channel whitening uses training parents only")
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("Whitening needs a positive finite regularizer")
    start = time.perf_counter()
    channels = (np.asarray(sources)@initial["encoder"]).reshape(-1, initial["rank"])
    centered = channels-channels.mean(0)
    covariance = centered.T@centered/max(1, len(centered)-1)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    floor = max(float(eigenvalues.max())*ridge, 1e-30)
    scales = np.sqrt(np.maximum(eigenvalues, 0.)+floor)
    transform = vectors/scales
    inverse = scales[:, None]*vectors.T
    return dict(initial, encoder=initial["encoder"]@transform, decoder=inverse@initial["decoder"],
        fit_seconds=time.perf_counter()-start, whitening_ridge=ridge, training_channel_covariance_eigenvalues=eigenvalues.tolist(),
        whitening_scope="invertible train-only channel coordinates; predictions must remain numerically unchanged")


def apply_response(sources, weights, spatial_shape):
    """Apply the actual Fourier heat/mean response to every temporal snapshot."""
    source = np.asarray(sources)
    shape = tuple(spatial_shape)
    if source.ndim != 3 or source.shape[-1] != math.prod(shape):
        raise ValueError("Source shape differs from decoder geometry")
    fields = source.reshape(*source.shape[:2], *shape)
    axes = tuple(range(2, fields.ndim))
    spectral = np.fft.fftn(fields, axes=axes)
    if np.asarray(weights).shape != fields.shape:
        raise ValueError("Response weights must cover every parent, time, and Fourier mode")
    output = np.fft.ifftn(np.sum(spectral*weights, axis=1), axes=tuple(range(1, fields.ndim-1)))
    return output.real.reshape(source.shape[0], -1)


def fit_decoder_optimized(sources, weights, initial, *, split, updates=30, learning_rate=.003, budget=None):
    """Equal-rank linear encoder/decoder trained through the actual response.

    The objective includes endpoint RMS and maximum error, with a small source
    reconstruction penalty to discourage arbitrary response-null directions.
    No validation/test array is accepted by this training interface.
    """
    if split != "train" or initial.get("training_split") != "train":
        raise ValueError("Decoder optimization must use train parents only")
    start = time.perf_counter()
    shape = tuple(initial["spatial_shape"])
    data = torch.as_tensor(np.asarray(sources), dtype=torch.float64)
    response = torch.as_tensor(np.asarray(weights), dtype=torch.float64)
    encoder = torch.nn.Parameter(torch.tensor(initial["encoder"], dtype=torch.float64))
    decoder = torch.nn.Parameter(torch.tensor(initial["decoder"], dtype=torch.float64))
    optimizer = torch.optim.Adam([encoder, decoder], lr=learning_rate)
    axes = tuple(range(2, 2+len(shape)))
    def decode(value):
        fields = value.reshape(*value.shape[:2], *shape)
        return torch.fft.ifftn((torch.fft.fftn(fields, dim=axes)*response).sum(1), dim=tuple(range(1, 1+len(shape)))).real.flatten(1)
    target = decode(data).detach()
    mass = torch.as_tensor(initial.get("physical_mass", np.ones(shape)), dtype=torch.float64)
    def physical_mse(values):
        fields = values.reshape(-1, *shape)
        transform = torch.fft.fftn(fields, dim=tuple(range(1, 1+len(shape))), norm="forward")
        return (transform.abs().square()*mass).sum(tuple(range(1, 1+len(shape)))).mean()
    source_scale = physical_mse(data).clamp_min(1e-30)
    endpoint_scale = physical_mse(target).clamp_min(1e-30)
    history = []
    for update in range(int(updates)):
        if budget is not None:
            budget.check()
        optimizer.zero_grad()
        reconstruction = data@encoder@decoder
        delta = decode(reconstruction)-target
        loss = physical_mse(delta)/endpoint_scale + .05*delta.abs().amax(1).square().mean()/endpoint_scale + .001*physical_mse(reconstruction-data)/source_scale
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite compression training objective")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([encoder, decoder], 10.)
        optimizer.step()
        history.append(dict(update=update, loss=float(loss.detach())))
    result = dict(initial, encoder=encoder.detach().numpy(), decoder=decoder.detach().numpy(),
        fit_seconds=time.perf_counter()-start, optimization_updates=int(updates), loss_history=history,
        objective="train endpoint physical normalized RMS + .05 nodal maximum-squared + .001 physical source MSE")
    return result


def response_metric(weights, *, mass=None, relative_floor=1e-8):
    """Training-only diagonal surrogate for the response-induced source norm."""
    weights = np.asarray(weights, float)
    metric = np.mean(weights**2, axis=(0, 1))
    if mass is not None:
        metric = metric*np.asarray(mass)
    floor = max(float(metric.max())*relative_floor, 1e-30)
    return np.maximum(metric, floor)


def physical_metrics(value, spatial_shape, mass, target_kind="galerkin_same_grid"):
    """Interpolant L2 with Nyquist masses; maximum on a declared 4x sample grid."""
    from scipy.signal import resample
    shape = tuple(spatial_shape)
    values = np.asarray(value).reshape(-1, *shape)
    axes = tuple(range(1, values.ndim))
    spectrum = np.fft.fftn(values, axes=axes, norm="forward")
    rms = float(np.sqrt(np.mean(np.sum(abs(spectrum)**2*mass, axis=axes))))
    if target_kind == "fd_same_grid":
        return dict(rms=rms, maximum=float(np.max(abs(values))), norm="nodal RMS and nodal maximum of the declared FD equation")
    for axis, n in enumerate(shape, 1):
        values = resample(values, 4*n, axis=axis)
    return dict(rms=rms, maximum=float(np.max(abs(values))),
                norm="continuous interpolant L2; sampled 4x interpolant maximum (not supremum certificate)")


def _trajectory(sample, nodes, budget):
    """Charged DOP853 dense path and exact centered defect forcing.

    The independent NumPy FFT path is checked against the prepared endpoint.
    Endpoint agreement is a cross-check, not a certificate for dense output.
    """
    from scipy.integrate import solve_ivp
    from scipy.signal import resample
    from . import numerics as n
    from .references import equation_geometry
    from .trajectory import anchor_path, actual_source
    record = sample["record"]
    if record["target_kind"] not in ("fd_same_grid", "galerkin_same_grid"):
        raise ValueError("Compression diagnostic requires an explicit same-grid equation; closure is separate")
    eq, geom = equation_geometry(record)
    u0 = sample["u0"].double().cpu()
    h = record["horizon"]
    shape = tuple(geom.grid)
    start = time.perf_counter()
    symbol = n.diffusion_symbol(u0, eq, geom, record["target_kind"]).cpu().numpy().reshape(shape)
    spatial_axes = tuple(range(len(shape)))
    def rhs(t, flat):
        budget.check()
        field = flat.reshape(shape)
        lap = np.fft.ifftn(np.fft.fftn(field)*symbol).real
        if record["target_kind"] == "fd_same_grid":
            square = field*field
        else:
            extended = field
            for axis, count in enumerate(shape):
                extended = resample(extended, 2*count+1, axis=axis)
            square = extended*extended
            for axis, count in enumerate(shape):
                square = resample(square, count, axis=axis)
        return (lap+float(eq.reaction_rate)*(field-square)).reshape(-1)
    solution = solve_ivp(rhs, (0., h), u0.numpy().reshape(-1), method="DOP853", rtol=2e-11, atol=2e-13, dense_output=True)
    if not solution.success:
        raise RuntimeError("Compression source teacher failed")
    teacher_seconds = time.perf_counter()-start
    x, quadrature = np.polynomial.legendre.leggauss(nodes)
    times, quadrature = h*(x+1)/2, h*quadrature/2
    sources, linear_sources, response = [], [], []
    source_tick = time.perf_counter()
    clock_x, clock_w = np.polynomial.legendre.leggauss(max(16, nodes))
    for t, weight in zip(times, quadrature):
        budget.check()
        truth = torch.from_numpy(solution.sol(float(t)).reshape(1, 1, *shape).copy())
        anchor = anchor_path(u0, float(t), eq, geom, record["target_kind"], create_graph=False)
        forcing = actual_source(truth, anchor["state"], anchor["derivative"], eq, geom, record["target_kind"])
        sources.append(forcing["source"].detach().numpy().reshape(-1))
        linear_sources.append(-forcing["centered_residual"].detach().numpy().reshape(-1))
        clock_times = t+(h-t)*(clock_x+1)/2
        mean_values = solution.sol(clock_times).mean(0)
        clock = (h-t)/2*np.dot(clock_w, 1-2*mean_values)
        response.append(weight*np.exp(float(eq.reaction_rate)*clock+(h-t)*symbol))
    endpoint = solution.sol(h).reshape(shape)
    anchor_endpoint = n.anchor(u0, h, eq, geom, record["target_kind"]).numpy().reshape(shape)
    defect = endpoint-anchor_endpoint
    prepared = sample["target"].numpy().reshape(shape)
    mass = n.mass_weights(u0, geom).numpy().reshape(shape) if record["target_kind"] == "galerkin_same_grid" else np.ones(shape)
    return dict(sources=np.asarray(sources), linear_sources=np.asarray(linear_sources), weights=np.asarray(response),
        centered_defect=(defect-defect.mean()).reshape(-1), mean_defect=float(defect.mean()), mass=mass,
        source_seconds=time.perf_counter()-source_tick, teacher_seconds=teacher_seconds,
        prepared_endpoint_discrepancy_rms=float(np.sqrt(np.mean((prepared-endpoint)**2))),
        source_reference_scope="dense DOP853 trajectory cross-checked at endpoint; dense-path refinement uncertainty not certified",
        record=record, nfev=solution.nfev)


def select_source_cohort(samples, cfg=None):
    """Frozen input-only bounded menu; never examines targets or measured loss."""
    cfg = cfg or {}
    caps = cfg.get("parent_caps", {"train":32, "validation":16, "development_test":16, "development":16})
    fixed_horizons = cfg.get("horizons", {"train":.06, "validation":.04, "development_test":.08, "development":.08})
    maximum_descendants = int(cfg.get("max_descendants_per_parent", 2))
    if maximum_descendants < 1:
        raise ValueError("Compression descendant cap must be positive")
    selected, owner = {}, {}
    for sample in sorted(samples, key=lambda s: (s["record"]["split"], s["record"]["parent_id"], s["record"]["query_id"])):
        rec = sample["record"]
        if rec["parent_id"] in owner and owner[rec["parent_id"]] != rec["split"]:
            raise ValueError("A compression parent cannot cross training/validation/test splits")
        owner[rec["parent_id"]] = rec["split"]
        ids = selected.setdefault(rec["split"], [])
        maximum = int(cfg.get("max_parents_per_split", caps.get(rec["split"], 16)))
        if maximum < 1:
            raise ValueError("Compression parent cap must be positive")
        if rec["parent_id"] not in ids and len(ids) < maximum:
            ids.append(rec["parent_id"])
    chosen_horizons = {}
    for split in selected:
        available = {s["record"]["horizon"] for s in samples if s["record"]["split"] == split}
        chosen_horizons[split] = min(available, key=lambda h: (abs(h-fixed_horizons.get(split, .08)), h))
    chosen, skipped, used = [], [], {}
    for sample in sorted(samples, key=lambda s: (s["record"]["parent_id"], s["record"].get("phase_variant", "original"), s["record"]["query_id"])):
        rec = sample["record"]; parent = rec["parent_id"]
        if parent in selected[rec["split"]] and rec["horizon"] == chosen_horizons[rec["split"]] and used.get(parent, 0) < maximum_descendants:
            chosen.append(sample); used[parent] = used.get(parent, 0)+1
        else:
            skipped.append(dict(query_id=rec["query_id"], parent_id=parent, reason="PREDECLARED_BOUNDED_COMPRESSION_SUBSTUDY_MENU"))
    return chosen, selected, chosen_horizons, skipped


def run(ctx):
    from .references import load_samples
    cfg = ctx.protocol.get("response", {}).get("compression", {})
    ranks = list(cfg.get("ranks", [2, 4, 8]))
    nodes = int(cfg.get("time_nodes", 12 if ctx.protocol.get("profile") == "smoke" else 16))
    if nodes < max(ranks):
        raise ValueError("Quadrature node count must cover every declared rank")
    samples = []
    for stage, path in ctx.prerequisites.items():
        if (Path(path)/"dataset.json").exists():
            samples.extend(load_samples(path))
    if not samples:
        raise ValueError("Compression requires verified prepared train and held-out datasets")
    seen = set()
    for sample in samples:
        record = sample["record"]
        if record["query_id"] in seen:
            raise ValueError("Compression prerequisites repeat a query")
        seen.add(record["query_id"])
    groups = {}
    skipped = []
    for sample in samples:
        record = sample["record"]
        if record["target_kind"] not in ("fd_same_grid", "galerkin_same_grid"):
            skipped.append(dict(query_id=record["query_id"], parent_id=record["parent_id"], reason="SEPARATE_CLOSURE_TARGET_NOT_POOLED"))
            continue
        groups.setdefault((record["target_kind"], tuple(record["grid"])), []).append(sample)
    rows, basis_rows, histories, diagnostics = [], [], [], []
    started = time.perf_counter()
    for (target_kind, grid), group in sorted(groups.items()):
        # One preregistered horizon per split, retaining both phase descendants.
        # This is a bounded representation sub-study, not an exhaustive reuse of
        # every main-training query. Selection uses only declared input metadata.
        group, selected, chosen_horizons, menu_skipped = select_source_cohort(group, cfg)
        skipped.extend(menu_skipped)
        requested_queries = len(group)
        for sample in group:
            record = sample["record"]
            if not record["reference"].get("accepted"):
                skipped.append(dict(query_id=record["query_id"], parent_id=record["parent_id"], reason="UNRESOLVED_PREPARED_REFERENCE"))
        group = [s for s in group if s["record"]["reference"].get("accepted")]
        if not any(s["record"]["split"] == "train" for s in group):
            ctx.record(f"compression/{target_kind}/unresolved-training-cohort", ["M4"], metrics=dict(requested_queries=requested_queries),
                checks=[check("accepted_training_source", None, None, category="math", reason="The fixed source cohort has no accepted training teacher; no replacements were drawn"),
                        check("representation_gain", None, None, category="gap")], evidence=["compression_results.json"])
            continue
        trajectories = [_trajectory(sample, nodes, ctx.budget) for sample in group]
        train = [v for v in trajectories if v["record"]["split"] == "train"]
        if not train:
            raise ValueError("No accepted training sources for compression")
        source_train = np.stack([v["sources"] for v in train])
        weight_train = np.stack([v["weights"] for v in train])
        mass = train[0]["mass"]
        metric = response_metric(weight_train, mass=mass)
        for trajectory in trajectories:
            record = trajectory["record"]
            response = apply_response(trajectory["sources"][None], trajectory["weights"][None], grid)[0]
            linear = apply_response(trajectory["linear_sources"][None], trajectory["weights"][None], grid)[0]
            exact = trajectory["centered_defect"]
            diagnostics.append(dict(parent_id=record["parent_id"], query_id=record["query_id"], split=record["split"], target_kind=target_kind,
                full_source_response_rms=physical_metrics(response-exact, grid, mass, target_kind)["rms"],
                residual_only_response_rms=physical_metrics(linear-exact, grid, mass, target_kind)["rms"],
                nonlinear_feedback_response_rms=physical_metrics(response-linear, grid, mass, target_kind)["rms"],
                anchor_centered_defect_rms=physical_metrics(exact, grid, mass, target_kind)["rms"], mean_defect=trajectory["mean_defect"],
                teacher_seconds=trajectory["teacher_seconds"], source_seconds=trajectory["source_seconds"],
                prepared_endpoint_discrepancy_rms=trajectory["prepared_endpoint_discrepancy_rms"],
                reference_scope=trajectory["source_reference_scope"]))
        for rank in ranks:
            ctx.budget.check()
            models = {"unweighted_pod": fit_pod(source_train, rank, split="train", spatial_shape=grid, metric=mass),
                      "response_weighted_pod": fit_pod(source_train, rank, split="train", spatial_shape=grid, metric=metric)}
            for model in models.values():
                model["physical_mass"] = mass
            models["response_whitened_pod"] = whiten_channels(source_train, models["response_weighted_pod"], split="train")
            models["decoder_optimized_pod"] = fit_decoder_optimized(source_train, weight_train, models["response_weighted_pod"],
                split="train", updates=int(cfg.get("optimization_updates", 8 if ctx.protocol.get("profile") == "smoke" else 40)), budget=ctx.budget)
            for name, model in models.items():
                stem = f"basis-{target_kind}-n{'x'.join(map(str, grid))}-j{rank}-{name}"
                np.savez_compressed(ctx.path/(stem+".npz"), encoder=model["encoder"], decoder=model["decoder"], metric=model["metric"])
                gram = model["decoder"]@model["decoder"].T
                basis_rows.append(dict(method=name, rank=rank, target_kind=target_kind, grid=list(grid), basis_file=stem+".npz",
                    fit_seconds=model["fit_seconds"], training_sha256=model["training_sha256"], training_parent_ids=sorted({v["record"]["parent_id"] for v in train}),
                    effective_training_rank=model["effective_training_rank"], gram_eigenvalues=np.linalg.eigvalsh(gram).tolist(),
                    gram_condition=float(np.linalg.cond(gram)), coefficients_per_query=rank*nodes,
                    basis_floats=2*rank*math.prod(grid), identifiability="basis rotations/rescalings are nonidentifiable; endpoint reconstruction is the observable"))
                if model.get("loss_history"):
                    histories.append(dict(method=name, rank=rank, target_kind=target_kind, rows=model["loss_history"]))
                for trajectory in trajectories:
                    ctx.budget.check()
                    record = trajectory["record"]
                    source = trajectory["sources"][None]
                    tick = time.perf_counter()
                    compressed = reconstruct(source, model)
                    answer = apply_response(compressed, trajectory["weights"][None], grid)[0]
                    seconds = time.perf_counter()-tick
                    full = apply_response(source, trajectory["weights"][None], grid)[0]
                    source_error = compressed-source
                    endpoint_error = answer-trajectory["centered_defect"]
                    propagated_error = answer-full
                    source_metrics = physical_metrics(source_error, grid, mass, target_kind)
                    source_size = physical_metrics(source, grid, mass, target_kind)["rms"]
                    endpoint_metrics = physical_metrics(endpoint_error, grid, mass, target_kind)
                    propagated_metrics = physical_metrics(propagated_error, grid, mass, target_kind)
                    row = dict(parent_id=record["parent_id"], query_id=record["query_id"], split=record["split"],
                        method=name, rank=rank, target_kind=target_kind, grid=list(grid), horizon=record["horizon"],
                        source_rms=source_metrics["rms"], source_relative_rms=source_metrics["rms"]/max(source_size, 1e-150),
                        propagated_error_rms=propagated_metrics["rms"], propagated_error_maximum=propagated_metrics["maximum"],
                        endpoint_centered_rms=endpoint_metrics["rms"], endpoint_maximum=endpoint_metrics["maximum"], norm=endpoint_metrics["norm"],
                        endpoint_mean_error=float(endpoint_error.mean()), decoder_seconds=seconds,
                        source_teacher_seconds=trajectory["teacher_seconds"], source_construction_seconds=trajectory["source_seconds"],
                        basis_fit_seconds=model["fit_seconds"], reference_accepted=record["reference"]["accepted"],
                        oracle_information="truth source and truth mean trajectory supplied to every compression method; not a deployed learned solver",
                        role="Ours" if name == "decoder_optimized_pod" else "Analytic control")
                    rows.append(row)
        ctx.record(f"compression/{target_kind}/n{'x'.join(map(str, grid))}", ["M4"],
            metrics=dict(parents={k: len(v) for k, v in selected.items()}, ranks=ranks, time_nodes=nodes,
                measured_queries=len(trajectories), requested_queries=requested_queries, methods=4, chosen_input_horizons=chosen_horizons),
            checks=[check("train_only_basis", True, True, relation="eq", category="math"),
                    check("deployed_inference_advantage", None, None, category="gap", reason="Oracle-source representation diagnostic only")],
            config=dict(target_kind=target_kind, grid=list(grid)), evidence=["compression_results.json", "compression_diagnostics.json"])
    if not groups:
        ctx.record("compression/no-same-grid-source-scope", ["M4"], metrics=dict(omitted_queries=len(skipped)),
            checks=[check("declared_source_identity", None, None, category="math", reason="Projected-fine source needs an extra closure forcing; coarse same-equation identity is not used"),
                    check("compression_advantage", None, None, category="gap", reason="No compatible source cohort")], evidence=["compression_results.json"])
    result = dict(schema="tdn.response-compression/v1", rows=rows, bases=basis_rows, omitted=skipped,
        elapsed_seconds=time.perf_counter()-started, experimental_scope="development oracle-source representation; train-only fitted bases and fixed rank comparisons",
        decoder="full spatial Fourier heat and measured-mean response; diagonal training response metric is an approximation to its full Gram",
        nonlinear_feedback="full exact centered forcing versus centered anchor residual only, evaluated with the same measured mean response",
        limitations=["Dense source path is an independently integrated diagnostic teacher; endpoint is cross-checked, dense-path uncertainty is not a certificate",
            "Compression reduces representation rank; sources are still teacher-acquired, so no solver acceleration follows", "No hyperparameter is selected on development-test fields"])
    (ctx.path/"compression_results.json").write_text(json.dumps(clean(result), indent=2, allow_nan=False)+"\n")
    (ctx.path/"compression_diagnostics.json").write_text(json.dumps(clean(dict(schema="tdn.response-compression-diagnostics/v1", rows=diagnostics)), indent=2, allow_nan=False)+"\n")
    (ctx.path/"compression_learning.json").write_text(json.dumps(clean(dict(schema="tdn.response-compression-learning/v1", trials=histories)), indent=2, allow_nan=False)+"\n")
    return dict(rows=len(rows), bases=len(basis_rows), omitted_queries=len(skipped),
        unresolved_references=sum(v["reason"] == "UNRESOLVED_PREPARED_REFERENCE" for v in skipped), elapsed_seconds=result["elapsed_seconds"])
