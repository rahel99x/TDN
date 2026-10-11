"""Train-only convex attribution of the two physical correction coefficients.

The cached FP32 physical fields are treated as a real affine map in FP64.
Its normalized squared-error objective is convex, including the spatial maximum.
Floating-point supporting-plane bounds diagnose convergence; they are not an
interval-arithmetic certificate or a stability theorem. Native FP32 assembly is
evaluated separately so rounding differences cannot become a learned benefit.
"""
from __future__ import annotations

from collections import Counter
import json
import math
import time

import numpy as np
from scipy.optimize import linprog
import torch

from tdn.analysis.advance.data import load_bank
from tdn.analysis.advance.metrics import endpoint_metrics
from tdn.analysis.advance.models import make_model
from tdn.analysis.frontier.core import check, clean
from tdn.analysis.frontier.neural import _samples, _sync, reference
from tdn.runtime.metadata import write_json


class AffineObjective:
    """Independent-field weighted exact finite-sample convex loss.

    ``basis`` has shape (examples, spatial values, coefficients), and ``offset``
    is baseline minus target, formed after conversion to FP64. Repeated horizons
    or physical variants of one field share that field's total weight.
    """

    def __init__(self, basis, offset, scales, field_ids, peak_weight=.1):
        self.basis = np.asarray(basis, dtype=np.float64)
        self.offset = np.asarray(offset, dtype=np.float64)
        self.scales = np.asarray(scales, dtype=np.float64)
        self.peak_weight = float(peak_weight)
        if (self.basis.ndim != 3 or self.basis.shape[:2] != self.offset.shape
                or min(self.basis.shape) < 1 or self.scales.shape != (len(self.basis),)
                or len(field_ids) != len(self.basis)):
            raise ValueError("Affine objective needs matched nonempty example/field arrays")
        if (not all(np.isfinite(x).all() for x in (self.basis, self.offset, self.scales))
                or np.any(self.scales <= 0) or not math.isfinite(self.peak_weight)
                or self.peak_weight < 0):
            raise ValueError("Affine objective requires finite arrays, positive scales and nonnegative peak weight")
        counts = Counter(field_ids)
        self.field_ids = list(field_ids)
        self.weights = np.asarray([1 / (len(counts) * counts[f]) for f in field_ids])
        self.factors = self.weights / self.scales

    def value_gradient(self, gains):
        gains = np.asarray(gains, dtype=np.float64)
        if gains.shape != (self.basis.shape[-1],) or not np.isfinite(gains).all():
            raise ValueError("Gain vector does not match finite affine coefficients")
        error = self.offset + np.einsum("npk,k->np", self.basis, gains)
        square = error * error
        # Selecting any exactly active maximum gives a valid subgradient. The
        # first index is deterministic; a tied maximum need not be differentiable.
        indices = square.argmax(axis=1)
        n = np.arange(len(indices))
        losses = square.mean(axis=1) + self.peak_weight * square[n, indices]
        gradients = 2 * np.einsum("npk,np->nk", self.basis, error) / error.shape[1]
        gradients += 2 * self.peak_weight * error[n, indices, None] * self.basis[n, indices]
        value = float(np.dot(self.factors, losses))
        gradient = np.einsum("n,nk->k", self.factors, gradients)
        if not math.isfinite(value) or not np.isfinite(gradient).all():
            raise FloatingPointError("Nonfinite affine loss or subgradient")
        return value, gradient

    def gram(self):
        gram = np.einsum("n,npk,npl->kl", self.factors, self.basis, self.basis) / self.basis.shape[1]
        eigenvalues = np.linalg.eigvalsh(gram)
        norms = np.sqrt(np.maximum(np.diag(gram), 0))
        denom = norms[:, None] * norms[None, :]
        correlation = np.divide(gram, denom, out=np.zeros_like(gram), where=denom > 0)
        normalized_eigenvalues = np.linalg.eigvalsh(correlation)
        tolerance = np.finfo(float).eps * max(gram.shape) * max(float(eigenvalues[-1]), 0.)
        rank = int(np.sum(eigenvalues > tolerance))
        return clean(dict(gram=gram.tolist(), eigenvalues=eigenvalues.tolist(), numerical_rank=rank,
            coefficient_count=gram.shape[0], condition_number=float(eigenvalues[-1] / eigenvalues[0])
            if rank == gram.shape[0] else None, normalized_correlation=correlation.tolist(),
            normalized_eigenvalues=normalized_eigenvalues.tolist(),
            zero_basis_columns=np.flatnonzero(norms == 0).tolist(),
            rank_tolerance=tolerance, weighting="Equal independent fields, then repeated queries, divided by fixed loss scale",
            significance="Identifiability on these observed basis fields only; no population guarantee"))


def _box_bounds(bounds, dimension):
    bounds = np.asarray(bounds, dtype=np.float64)
    if bounds.shape == (2,):
        bounds = np.tile(bounds, (dimension, 1))
    if bounds.shape != (dimension, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("Strict finite coefficient bounds required")
    return bounds


def convex_fit(objective, *, free=(0, 1), fixed=(1., 1.), bounds=(.25, 1.75),
               max_iterations=80, atol=1e-8, rtol=1e-6, budget=None, progress=None):
    """Deterministic Kelley cuts with an explicitly reconstructed box lower bound.

    For each cut f(x_i)+g_i.(x-x_i), any nonnegative weights summing to one
    produce an affine minorant. Its minimum over the box is a lower bound.
    LP dual weights merely choose such a mixture; solver success alone is never
    reported as optimality. Rounding is not enclosed and remains explicit.
    """
    dimension = objective.basis.shape[-1]
    fixed = np.asarray(fixed, dtype=np.float64)
    free = tuple(free)
    if fixed.shape != (dimension,) or not np.isfinite(fixed).all() or not free or len(set(free)) != len(free) or any(i not in range(dimension) for i in free):
        raise ValueError("A finite full coefficient vector and distinct free indices are required")
    limits = _box_bounds(bounds, len(free))
    if max_iterations < 1 or not all(math.isfinite(float(x)) and float(x) >= 0 for x in (atol, rtol)):
        raise ValueError("Finite nonnegative tolerances and positive iteration budget required")
    x = np.clip(fixed[list(free)], limits[:, 0], limits[:, 1])
    gradients, intercepts, curves = [], [], []
    # Nonnegativity of this squared-error objective supplies an independent
    # exact symbolic lower bound, useful when floating cuts approach zero.
    best, lower, best_gains = math.inf, 0., fixed.copy()
    start = time.perf_counter(); status = "ITERATION_LIMIT"; max_violation = 0.
    for iteration in range(1, int(max_iterations) + 1):
        if budget is not None:
            budget.check()
        gains = fixed.copy(); gains[list(free)] = x
        value, full_gradient = objective.value_gradient(gains)
        gradient = full_gradient[list(free)]
        if value < best:
            best, best_gains = value, gains.copy()
        gradients.append(gradient); intercepts.append(value - float(gradient @ x))
        matrix = np.asarray(gradients); intercept = np.asarray(intercepts)
        solution = linprog(np.r_[np.zeros(len(free)), 1.],
            A_ub=np.column_stack([matrix, -np.ones(len(matrix))]), b_ub=-intercept,
            bounds=[tuple(v) for v in limits] + [(None, None)], method="highs",
            options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10})
        if not solution.success:
            status = "LP_FAILED"; break
        weights = np.maximum(-np.asarray(solution.ineqlin.marginals), 0.)
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            status = "LP_DUAL_UNAVAILABLE"; break
        weights /= weights.sum()
        slope = weights @ matrix
        candidate_lower = float(weights @ intercept + np.sum(np.where(slope >= 0, slope * limits[:, 0], slope * limits[:, 1])))
        lower = max(lower, candidate_lower)
        violation = float(max(0., np.max(matrix @ solution.x[:-1] - solution.x[-1] + intercept)))
        max_violation = max(max_violation, violation)
        raw_gap = best - lower
        tolerance = float(atol + rtol * max(abs(best), 1.))
        if raw_gap < -max(tolerance, 1e-10 * max(1., abs(best))):
            status = "NUMERICAL_BOUND_INCONSISTENCY"
        elif max(raw_gap, 0.) <= tolerance:
            status = "NUMERICALLY_CONVERGED"
        curves.append(dict(iteration=iteration, objective=value, gains=gains.tolist(), best_objective=best,
            lower_bound=lower, objective_gap=max(0., raw_gap), raw_objective_gap=raw_gap,
            relative_gap=max(0., raw_gap) / max(1., abs(best)), lp_constraint_violation=violation,
            elapsed_seconds=time.perf_counter()-start))
        if progress is not None:
            progress(curves[-1])
        if status != "ITERATION_LIMIT":
            break
        next_x = np.clip(solution.x[:-1], limits[:, 0], limits[:, 1])
        if np.linalg.norm(next_x - x, ord=np.inf) < 1e-14:
            status = "STALLED_WITH_UNRESOLVED_GAP"; break
        x = next_x
    return clean(dict(gains=best_gains.tolist(), objective=best, lower_bound=lower,
        objective_gap=max(0., best-lower), tolerance=atol + rtol * max(1., abs(best)),
        status=status, converged=status == "NUMERICALLY_CONVERGED", iterations=len(curves),
        max_lp_constraint_violation=max_violation, free_indices=list(free), bounds=limits.tolist(),
        elapsed_seconds=time.perf_counter()-start, curves=curves,
        optimality_scope="Floating-point supporting-plane mixture lower bound for this fixed finite-sample real-affine objective; not a rigorous arithmetic certificate",
        proof_status="NOT_A_PROOF"))


def adam_fit(objective, *, bounds=(.25, 1.75), updates=800, learning_rate=.01, budget=None, progress=None):
    """Full-batch Adam/tanh on exactly the same cached affine objective.

    CPU FP64 and identical normalization isolate parametrization/optimization.
    This is a new controlled comparison, not a replay of stochastic FP32 fits.
    No validation loss selects its iterate. Return best training iterate.
    """
    limits = _box_bounds(bounds, objective.basis.shape[-1])
    if updates < 0 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Positive Adam learning rate and nonnegative update count required")
    lower = torch.from_numpy(limits[:, 0].copy()); upper = torch.from_numpy(limits[:, 1].copy())
    if not bool(((lower < 1) & (upper > 1)).all()):
        raise ValueError("Adam/tanh control must initialize at unit gains inside the box")
    logits = torch.nn.Parameter(torch.atanh((2-lower-upper) / (upper-lower)))
    optimizer = torch.optim.Adam([logits], lr=learning_rate)
    basis = torch.from_numpy(objective.basis); offset = torch.from_numpy(objective.offset)
    factors = torch.from_numpy(objective.factors)
    curves = []; start = time.perf_counter(); best = math.inf; best_gains = None; selected = 0
    for update in range(int(updates)+1):
        if budget is not None:
            budget.check()
        gains = (lower+upper)/2 + (upper-lower)/2 * torch.tanh(logits)
        error = offset + torch.einsum("npk,k->np", basis, gains)
        square = error.square()
        loss = torch.sum(factors * (square.mean(1) + objective.peak_weight * square.amax(1)))
        value = float(loss.detach()); physical = gains.detach().tolist()
        if not math.isfinite(value):
            raise FloatingPointError("Nonfinite Adam affine objective")
        if value < best:
            best, best_gains, selected = value, physical, update
        curves.append(dict(update=update, objective=value, gains=physical, elapsed_seconds=time.perf_counter()-start))
        if progress is not None:
            progress(curves[-1])
        if update < updates:
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    return dict(gains=best_gains, objective=best, updates=int(updates), selected_update=selected,
        selection="Minimum training objective only", optimizer="full-batch Adam, no weight decay or clipping",
        learning_rate=learning_rate, dtype="CPU FP64 cached affine objective", bounds=limits.tolist(),
        parameterization="midpoint + half-width * tanh(logits)", elapsed_seconds=time.perf_counter()-start,
        curves=curves, status="BOUNDED_OPTIMIZATION_COMPLETE", optimality_gap=None,
        stochastic_seed_effect="None: fixed unit initialization and deterministic full-batch updates")


def _objective(samples, model, peak_weight, budget):
    fields = []
    with torch.no_grad():
        for sample in samples:
            budget.check()
            parts = model.correction_components(sample["u"], sample["h"], sample["eq"], sample["geom"])
            fields.append({k: parts[k].detach() for k in ("base", "quadratic", "cubic")})
    basis = np.stack([np.stack([v[k].cpu().double().numpy().ravel() for k in ("quadratic", "cubic")], axis=-1) for v in fields])
    offset = np.stack([(v["base"].cpu().double() - sample["target"].cpu().double()).numpy().ravel() for v, sample in zip(fields, samples)])
    return AffineObjective(basis, offset, [s["scale"] for s in samples], [s["field_cluster"] for s in samples], peak_weight), fields


def uncertainty_adjusted_ratio(rows, control, candidate):
    """Ratio of equal-field mean (control error-u)/(candidate error+u).

    This is sensitivity to the supplied empirical teacher uncertainty, not a
    rigorous confidence or error bound. Missing or unaccepted evidence is NA.
    """
    key = lambda row: (row["field_cluster"], row["parent_id"], row["grid"], row["horizon"], row["track"])
    groups = {name: {key(row): row for row in rows if row["method"] == name} for name in (control, candidate)}
    if not groups[control] or groups[control].keys() != groups[candidate].keys():
        return None
    by_field = {}
    for identity, left in groups[control].items():
        right = groups[candidate][identity]
        if any(not row.get("reference_accepted") or not row.get("finite")
               or row.get("reference_uncertainty_rms") is None for row in (left, right)):
            return None
        low = max(0., left["error_rms"] - left["reference_uncertainty_rms"])
        high = right["error_rms"] + right["reference_uncertainty_rms"]
        by_field.setdefault(identity[0], []).append((low, high))
    numerator = float(np.mean([np.mean([row[0] for row in values]) for values in by_field.values()]))
    denominator = float(np.mean([np.mean([row[1] for row in values]) for values in by_field.values()]))
    return numerator / denominator if denominator > 0 else None


def _run(ctx):
    """Run train-only coefficient attribution; validation is reporting only."""
    started = time.perf_counter()
    settings = ctx.protocol["resolve"]["fit"]
    scope = getattr(ctx, "unit", None) or ctx.protocol["units"][ctx.stage]
    track, seed = scope["track"], int(scope.get("seed", 0))
    grid = int(settings.get("grid", ctx.protocol["train_grid"]))
    peak = float(settings.get("peak_weight", .1))
    floor = float(settings.get("normalization_floor", 1e-6))
    bank = load_bank(ctx, "prepare")
    if any(p["split"] not in ("train", "validation") for p in bank):
        raise ValueError("Coefficient fitting may access train/validation banks only")
    parents = {split: [p for p in bank if p["split"] == split] for split in ("train", "validation")}
    if any(not values for values in parents.values()):
        raise ValueError("Coefficient attribution needs separate nonempty train and validation fields")
    if {p["field_cluster"] for p in parents["train"]} & {p["field_cluster"] for p in parents["validation"]}:
        raise ValueError("Training and validation field clusters overlap")
    config = dict(ctx.protocol["model_config"])
    base = make_model("df", track, config).to(device=ctx.device, dtype=torch.float32)
    model = make_model("two_basis", track, config).to(device=ctx.device, dtype=torch.float32)
    samples, objectives, fields = {}, {}, {}
    _sync(ctx.device); tick = time.perf_counter()
    for split in parents:
        horizons = ctx.protocol["train_horizons"] if split == "train" else settings.get("evaluation_horizons", ctx.protocol["validation_horizons"])
        samples[split] = _samples(parents[split], horizons, grid, track, ctx.device, base, floor)
        objectives[split], fields[split] = _objective(samples[split], model, peak, ctx.budget)
    _sync(ctx.device); preparation_seconds = time.perf_counter()-tick
    train = objectives["train"]
    basis_rows = []
    for split, objective in objectives.items():
        for index, sample in enumerate(samples[split]):
            response = objective.basis[index]
            gram = response.T @ response / len(response)
            norms = np.sqrt(np.maximum(np.diag(gram), 0.))
            basis_rows.append(dict(split=split, parent_id=sample["parent_id"], field_cluster=sample["field_cluster"],
                regime=sample["regime"], horizon=sample["h"], grid=grid, track=track,
                quadratic_rms=float(norms[0]), cubic_rms=float(norms[1]),
                quadratic_cubic_cosine=float(gram[0, 1]/(norms[0]*norms[1])) if norms.prod() > 0 else None,
                gram=gram.tolist(), eigenvalues=np.linalg.eigvalsh(gram).tolist(),
                scope="One physical field/query response; repeated queries are not independent samples"))
    write_json(ctx.path / "fit_basis_cases.json", dict(rows=clean(basis_rows)))
    fit_results = {"fixed_unit": dict(gains=[1., 1.], objective=train.value_gradient([1., 1.])[0],
        trainable_parameters=0, status="FIXED_ANALYTIC_PARENT", elapsed_seconds=0.)}
    direct_options = dict(bounds=(float(settings.get("lower", .25)), float(settings.get("upper", 1.75))),
        max_iterations=int(settings.get("direct_max_iterations", 80)),
        atol=float(settings.get("direct_atol", 1e-8)), rtol=float(settings.get("direct_rtol", 1e-6)), budget=ctx.budget)
    # g3=0 tests omission; g3=1 isolates fitting g2 with the same cubic term.
    definitions = [("quadratic_only_direct", (0,), (1., 0.)),
        ("quadratic_gain_direct", (0,), (1., 1.)), ("cubic_gain_direct", (1,), (1., 1.)),
        ("two_gain_direct", (0, 1), (1., 1.))]
    for name, free, fixed in definitions:
        complete = ctx.path / f"{name}.json"
        if complete.exists():
            output = json.loads(complete.read_text())
            if output["free_indices"] != list(free) or not math.isclose(
                    train.value_gradient(output["gains"])[0], output["objective"], rel_tol=1e-12, abs_tol=1e-14):
                raise ValueError("Completed coefficient fit no longer matches the sealed basis/objective")
        else:
            _retain_interrupted_progress(ctx.path, name)
            output = convex_fit(train, free=free, fixed=fixed, **direct_options,
                progress=lambda row, name=name: write_json(ctx.path / f"{name}-progress.json", row))
        output["trainable_parameters"] = len(free); fit_results[name] = output
        if not complete.exists():
            write_json(complete, clean(output))
    complete = ctx.path / "two_gain_adam.json"
    if complete.exists():
        fit_results["two_gain_adam"] = json.loads(complete.read_text())
        if not math.isclose(train.value_gradient(fit_results["two_gain_adam"]["gains"])[0],
                fit_results["two_gain_adam"]["objective"], rel_tol=1e-12, abs_tol=1e-14):
            raise ValueError("Completed Adam fit no longer matches the sealed basis/objective")
    else:
        _retain_interrupted_progress(ctx.path, "two_gain_adam")
        fit_results["two_gain_adam"] = adam_fit(train, bounds=direct_options["bounds"],
            updates=int(settings.get("adam_updates", 800)), learning_rate=float(settings.get("adam_rate", .01)),
            budget=ctx.budget, progress=lambda row: write_json(ctx.path / "two_gain_adam-progress.json", row))
    fit_results["two_gain_adam"]["trainable_parameters"] = 2
    if not complete.exists():
        write_json(complete, clean(fit_results["two_gain_adam"]))
    rows, summaries = [], {}
    by_parent = {p["parent_id"]: p for p in bank}
    controls = {name: make_model(name, track, config).to(device=ctx.device, dtype=torch.float32)
                for name in ("quad2_full", "quad4_full", "analytic_quad_cubic")}
    for name in (*fit_results, *controls):
        _sync(ctx.device); tick = time.perf_counter()
        measured = []
        for sample, physical in zip(samples["validation"], fields["validation"]):
            ctx.budget.check()
            with torch.no_grad():
                if name in fit_results:
                    gains = fit_results[name]["gains"]
                    prediction = physical["base"] + (gains[0]*physical["quadratic"] + gains[1]*physical["cubic"])
                    affine = physical["base"].double() + gains[0]*physical["quadratic"].double() + gains[1]*physical["cubic"].double()
                    assembly_error = float((prediction.double()-affine).abs().max())
                else:
                    prediction = controls[name](sample["u"], sample["h"], sample["eq"], sample["geom"])
                    assembly_error = None
            parent = by_parent[sample["parent_id"]]
            ref = reference(parent, grid, sample["h"], track)
            metrics = endpoint_metrics(prediction, ref["state"], sample["u"], sample["eq"], sample["geom"], track, ref)
            error = prediction.double()-sample["target"].double()
            objective = float((error.square().mean()+peak*error.square().amax())/sample["scale"]) if metrics["finite"] else None
            row = dict(metrics, method=name, track=track, seed=seed, split="validation", grid=grid,
                parent_id=sample["parent_id"], field_cluster=sample["field_cluster"], horizon=sample["h"],
                regime=sample["regime"], kappa=parent["kappa"], reaction_rate=parent["reaction_rate"],
                objective=objective, loss_scale=sample["scale"], fp32_assembly_max_difference=assembly_error)
            rows.append(row); measured.append(row)
        _sync(ctx.device)
        clusters = sorted({r["field_cluster"] for r in measured})
        average = lambda key: float(np.mean([np.mean([r[key] for r in measured if r["field_cluster"] == cluster]) for cluster in clusters])) if all(r[key] is not None for r in measured) else None
        summaries[name] = dict(validation_objective=average("objective"), validation_rms=average("error_rms"),
            validation_max=max(r["error_max"] for r in measured) if all(r["finite"] for r in measured) else None,
            finite=all(r["finite"] for r in measured),
            field_count=len(clusters), queries=len(measured),
            metric_evaluation_seconds=time.perf_counter()-tick,
            inference_latency_seconds=None, cost_scope="Metric loop, transfers and references included; cached-basis arithmetic is not deployment latency",
            real_affine_validation_objective=objectives["validation"].value_gradient(fit_results[name]["gains"])[0] if name in fit_results else None)
    chosen = summaries["two_gain_direct"]
    ratio = summaries["fixed_unit"]["validation_rms"] / max(chosen["validation_rms"], 1e-300)
    strong_ratio = summaries["quad4_full"]["validation_rms"] / max(chosen["validation_rms"], 1e-300) if summaries["quad4_full"]["finite"] else None
    conservative = {name: uncertainty_adjusted_ratio(rows, name, "two_gain_direct") for name in ("fixed_unit", "quad4_full")}
    direct = fit_results["two_gain_direct"]
    result = clean(dict(schema="tdn.resolve-coefficient-fit/v1", track=track, seed=seed, grid=grid,
        selection="Train-only fit; all validation rows reported, no validation checkpoint or family selection",
        objective="Equal field mean of (spatial MSE + peak_weight * maximum squared error) / fixed DF defect scale",
        peak_weight=peak, normalization_floor=floor, target_dtype="FP32 training target; FP64 reference metrics",
        physical_basis_dtype="FP32", optimization_dtype="CPU FP64 real-affine arithmetic",
        parent_ids={split: [p["parent_id"] for p in values] for split, values in parents.items()},
        objective_reference_scope="Fixed cached physical basis; native FP32 replay and teacher uncertainty reported separately",
        elapsed_seconds=time.perf_counter()-started, basis_preparation_seconds=preparation_seconds,
        fit_seconds=sum(x.get("elapsed_seconds", 0.) for x in fit_results.values()),
        train_gram=train.gram(), validation_gram=objectives["validation"].gram(),
        fits=fit_results, validation=summaries,
        direct_vs_adam_training_objective_ratio=fit_results["two_gain_adam"]["objective"]/max(direct["objective"], 1e-300),
        fixed_parent_over_direct_validation_rms_ratio=ratio,
        quad4_over_direct_validation_rms_ratio=strong_ratio,
        estimated_uncertainty_adjusted_validation_ratios=conservative,
        uncertainty_ratio_scope="Ratio of equal-field mean max(control RMS minus estimated uncertainty,0) to equal-field mean candidate RMS plus estimated uncertainty; sensitivity only, not a certificate",
        scientific_scope="Development attribution only; no fresh confirmation, population interval, complete solver cost or FNO claim"))
    write_json(ctx.path / "fit_results.json", result)
    write_json(ctx.path / "fit_endpoints.json", dict(rows=clean(rows)))
    mechanisms = scope.get("mechanism_ids", list(ctx.protocol["mechanisms"])[:1])
    identity = f"fit/{track}/seed-{seed}"
    if identity not in getattr(ctx, "identities", set()):
        ctx.record(identity, mechanisms, metrics={k: v for k, v in result.items() if k != "fits"}, checks=[
        check("convex-solver-status", direct["converged"], True, "eq", category="math",
            reason="An iteration limit, numerical inconsistency or LP failure is not convergence"),
        check("convex-numerical-objective-gap", direct["objective_gap"], direct["tolerance"], category="math",
            reason="Floating-point affine supporting-plane diagnostic, not a rigorous certificate"),
        check("native-replay-finite", all(v["finite"] for v in summaries.values()), True, "eq", category="math"),
        check("heldout-fixed-parent-gain", conservative["fixed_unit"], float(settings.get("min_validation_error_ratio", 1.1)), "ge", category="gap",
            reason="Development RMS ratio adjusted against estimated teacher uncertainty; not a certified error bound"),
        check("heldout-strong-analytic-gain", conservative["quad4_full"], float(settings.get("min_validation_error_ratio", 1.1)), "ge", category="gap",
            reason="Same uncertainty sensitivity against GL4, so a gain over the weaker fixed parent cannot imply strongest-control superiority"),
        check("fresh-confirmed-complete-cost", None, True, "eq", category="utility",
            reason="This coefficient attribution unit does not establish fresh confirmation or deployment cost")])
    return {k: result[k] for k in ("track", "grid", "elapsed_seconds", "fixed_parent_over_direct_validation_rms_ratio")}


def _retain_interrupted_progress(path, name):
    progress = path / f"{name}-progress.json"
    if progress.exists():
        index = 0
        while (path / f"{name}-interrupted-{index:03d}.json").exists():
            index += 1
        saved = json.loads(progress.read_text())
        write_json(path / f"{name}-interrupted-{index:03d}.json", dict(last_completed_progress=saved,
            recovery="Solver restarted deterministically; prior completed fits are reused",
            elapsed_scope="Last saved progress is a lower bound on interrupted optimizer work; stage attempt ledger charges observed attempt time"))


def run(ctx):
    """Recover whole fits, preserving interrupted work and cumulative attempt cost.

    The Resolve engine guards protocol/source/software/prerequisite identity.
    Completed optimizers are reused. An interrupted individual optimizer restarts
    deterministically; its previous progress and observed cost remain visible.
    """
    ledger = ctx.path / "fit_attempt_costs.json"
    attempts = json.loads(ledger.read_text())["attempts"] if ledger.exists() else []
    attempt = dict(attempt=len(attempts), status="RUNNING", elapsed_seconds=None)
    attempts.append(attempt); write_json(ledger, dict(attempts=attempts))
    started = time.perf_counter()
    try:
        result = _run(ctx)
    except BaseException as error:
        attempt.update(status="INTERRUPTED_OR_FAILED", error=f"{type(error).__name__}: {error}")
        raise
    else:
        attempt["status"] = "COMPLETED"
    finally:
        attempt["elapsed_seconds"] = time.perf_counter()-started
        write_json(ledger, dict(attempts=attempts, observed_total_seconds=sum(a["elapsed_seconds"] or 0. for a in attempts),
            scope="All observed stage attempts including reused-fit validation and interrupted optimizer work; killed processes can leave an unmeasured RUNNING attempt"))
    payload = json.loads((ctx.path / "fit_results.json").read_text())
    payload["attempt_costs"] = json.loads(ledger.read_text())
    write_json(ctx.path / "fit_results.json", payload)
    return result
