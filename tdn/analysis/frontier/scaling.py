"""Frozen-model throughput with different fields and independent accuracy."""
from __future__ import annotations

import math
import time

import torch

from tdn.runtime.metadata import write_json
from .core import check
from .measurement import measure_paired
from tdn.research.experiment import state_digest


def select_models(models, protocol):
    """Select declared seeds/data, never the fastest confirmation winner."""
    seed, count = protocol["seeds"][0], max(protocol["training"]["subset_sizes"])
    families = protocol["scaling"]["families"]
    selected = {key: model for key, model in models.items() if model.family in families
            and model.selection_metadata["seed"] in (None, seed)
            and model.selection_metadata["train_count"] in (0, count)}
    identities = [(model.track, model.family) for model in selected.values()]
    if len(identities) != len(set(identities)):
        raise ValueError("Scaling may not select multiple seeds or data fractions for one family/track")
    return selected


def distinct_batch(parents, batch_size, n):
    """Reject synthetic repeated copies even if someone changes their IDs."""
    batch = parents[:batch_size]
    if len(batch) != batch_size or len({p["field_cluster"] for p in batch}) != batch_size:
        raise ValueError("Scaling batches require distinct declared field clusters; no repeated copies")
    if len({(p["kappa"], p["reaction_rate"], tuple(p.get("lengths", (1., 1.)))) for p in batch}) != 1:
        raise ValueError("A batched call requires the same declared equation and geometry")
    states = [p["states"][str(n)] for p in batch]
    if any(tuple(state.shape) != (1, 1, n, n) for state in states):
        raise ValueError("Scaling parents each require one scalar field on the declared grid")
    if len({state_digest(state) for state in states}) != batch_size:
        raise ValueError("Scaling batches contain identical field values despite distinct IDs")
    return batch


def speed_comparison(candidate, comparator):
    """A throughput number alone cannot establish an accuracy-backed gain."""
    eligible_accuracy = candidate["accuracy_passed"] is True and comparator["accuracy_passed"] is True
    costs = [row.get("cost_seconds") for row in (candidate, comparator)]
    valid_cost = all(value is not None and math.isfinite(value) and value > 0 for value in costs)
    eligible = eligible_accuracy and valid_cost
    return dict(candidate=candidate["model_id"], comparator=comparator["model_id"],
        candidate_accuracy=candidate["accuracy_passed"], comparator_accuracy=comparator["accuracy_passed"],
        speed_ratio=costs[1] / costs[0] if eligible else None,
        matched_accuracy=eligible_accuracy, valid_timing=valid_cost,
        status="ELIGIBLE" if eligible else "INVALID_OR_MISSING_TIMING" if eligible_accuracy else "ACCURACY_UNRESOLVED_OR_FAILED",
        throughput_advantage_established=False,
        claim_scope="descriptive paired workload; speed ratio alone is not a robust superiority claim")


def accuracy_gate(errors, target):
    if not errors or any(e.get("reference_accepted") is not True for e in errors):
        return None
    if any(e.get("upper_rms") is None or e.get("upper_max") is None for e in errors):
        return False
    return all(math.isfinite(e["upper_rms"]) and math.isfinite(e["upper_max"])
               and max(e["upper_rms"], e["upper_max"]) <= target for e in errors)


def record_crossover(ctx, candidate, comparator):
    """Score the actual throughput question separately from runnable accuracy.

    The utility criterion is an observed paired-median threshold on this
    workload. It is not a confidence-interval test, an across-workload claim,
    or a reproduction of a literature result. Ineligible accuracy never earns
    throughput credit, regardless of an apparently fast raw implementation.
    """
    threshold = float(ctx.protocol["practical_speedup"])
    if not math.isfinite(threshold) or threshold <= 1:
        raise ValueError("A practical crossover requires a finite speed threshold greater than one")
    fields = ("track", "grid", "batch_size", "final_time", "target", "independent_fields")
    if any(candidate[key] != comparator[key] for key in fields):
        raise ValueError("Scaling crossover compares different requested outputs or fields")
    comparison = speed_comparison(candidate, comparator)
    kind = "neural" if comparator["family"] in ("fno_small", "fno_standard", "direct_fno", "rank1_frozen", "rank1_postcompression") else "classical"
    row = {key: candidate[key] for key in fields}
    row.update(comparison, comparison_kind=kind, candidate_family=candidate["family"],
        comparator_family=comparator["family"], candidate_seed=candidate["seed"],
        comparator_seed=comparator["seed"], candidate_train_count=candidate["train_count"],
        comparator_train_count=comparator["train_count"], candidate_parameters=candidate["parameters"],
        comparator_parameters=comparator["parameters"], candidate_cost_seconds=candidate["cost_seconds"],
        comparator_cost_seconds=comparator["cost_seconds"], candidate_schedule=candidate["schedule"],
        comparator_schedule=comparator["schedule"],
        candidate_selected_state=candidate.get("selected_state"), comparator_selected_state=comparator.get("selected_state"),
        candidate_updates_selected=candidate.get("updates_selected"), comparator_updates_selected=comparator.get("updates_selected"),
        practical_speedup_threshold=threshold, baseline_competitiveness_established=False,
        observed_threshold_passed=comparison["speed_ratio"] >= threshold if comparison["speed_ratio"] is not None else None,
        evidence_scope="observed interleaved median ratio against frozen local controls on one paired workload; no confidence-interval, FNO-adequacy, paper-competitiveness or universal superiority claim")
    accuracy = None if candidate["accuracy_passed"] is None or comparator["accuracy_passed"] is None else comparison["matched_accuracy"]
    timing = None if candidate["cost_seconds"] is None or comparator["cost_seconds"] is None else comparison["valid_timing"]
    ctx.record(f"{row['track']}/{row['grid']}/batch{row['batch_size']}/crossover/{kind}/{row['comparator']}", ["G4"], metrics=row,
        config={"comparison_kind": kind, "candidate": row["candidate"], "comparator": row["comparator"],
                "practical_speedup": threshold, "final_time": row["final_time"], "target": row["target"]},
        checks=[check("identical-requested-output-and-fields", True, True, "eq", category="math",
                      reason="Both rows share final time, spatial track, grid, accuracy target, batch and independent field identities"),
                check("both-models-independently-accurate", accuracy, True, "eq", category="gap"),
                check("positive-finite-paired-timing", timing, True, "eq", category="correctness"),
                check(f"observed-{kind}-throughput-crossover", row["speed_ratio"], threshold, "ge", category="utility", units="ratio",
                      reason=(row["evidence_scope"] if row["speed_ratio"] is not None else
                              "NA: " + row["status"] + "; throughput cannot earn credit without matched accepted accuracy and valid measured timing"),
                      evidence_kind="descriptive_paired_timing_threshold")])
    return row


@torch.no_grad()
def run(ctx):
    from .data import load_split
    from .neural import load_models, rollout, eq_geom, reference, endpoint_errors, expected_model_specs
    models = select_models(load_models(ctx), ctx.protocol)
    declared = [s for s in expected_model_specs(ctx.protocol)
                if s["family"] in ctx.protocol["scaling"]["families"]
                and s["seed"] in (None, ctx.protocol["seeds"][0])
                and s["train_count"] in (0, max(ctx.protocol["training"]["subset_sizes"]))]
    parents = load_split(ctx, "scaling")
    options = ctx.protocol["scaling"]
    rows, comparisons = [], []
    target = ctx.protocol["primary_target"]
    for track in ctx.protocol["tracks"]:
        selected = {name: model for name, model in models.items() if model.track == track}
        specs = {s["model_id"]: s for s in declared if s["track"] == track}
        for n in options["grids"]:
            for batch_size in options["batches"]:
                ctx.budget.check()
                batch = distinct_batch(parents, batch_size, n)
                eq, geom = eq_geom(batch[0], n)
                cpu = torch.cat([p["states"][str(n)] for p in batch], dim=0).float()
                if torch.device(ctx.device).type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                initial = cpu.to(ctx.device)
                if torch.device(ctx.device).type == "cuda":
                    torch.cuda.synchronize()
                transfer = time.perf_counter() - start
                horizon = options["horizon"]
                schedule = [horizon / options["steps"]] * options["steps"]
                calls = {name: lambda model=model: rollout(model, initial, schedule, eq, geom, ctx.budget)[0]
                         for name, model in selected.items()}
                failures, screening_seconds = {}, {}
                for name, call in list(calls.items()):
                    if torch.device(ctx.device).type == "cuda":
                        torch.cuda.synchronize()
                    screen_started = time.perf_counter()
                    try:
                        value = call()
                        if not bool(torch.isfinite(value).all()):
                            raise FloatingPointError("Nonfinite scaling output")
                    except FloatingPointError as error:
                        failures[name] = str(error)
                        del calls[name]
                    finally:
                        if torch.device(ctx.device).type == "cuda":
                            torch.cuda.synchronize()
                        screening_seconds[name] = time.perf_counter() - screen_started
                # Numerical screening above must not secretly prepopulate the
                # cold operator cache. This does not claim a cold process/GPU.
                for name in calls:
                    selected[name].clear_cache()
                outputs, timing = measure_paired(calls, device=ctx.device, repeats=options["repeats"],
                    warmup=ctx.protocol["timing_warmup"], seed=ctx.protocol["timing_seed"] + n + batch_size,
                    budget=ctx.budget) if calls else ({}, {"methods": {}})
                group = []
                for model_id, spec in specs.items():
                    model = selected.get(model_id)
                    errors = []
                    if model_id in outputs:
                        for index, parent in enumerate(batch):
                            ref = reference(parent, n, horizon, track)
                            errors.append(dict(parent_id=parent["parent_id"], field_cluster=parent["field_cluster"],
                                               **endpoint_errors(outputs[model_id][index:index + 1], ref)))
                    cost = timing["methods"].get(model_id, {})
                    passed = accuracy_gate(errors, target) if model_id not in failures else False
                    row = dict(model_id=model_id, family=spec["family"], track=track, grid=n, batch_size=batch_size,
                        final_time=horizon, schedule=schedule, seed=spec["seed"],
                        train_count=spec["train_count"],
                        selected_state=model.selection_metadata.get("selected_state") if model is not None else None,
                        updates_selected=model.selection_metadata.get("updates_selected") if model is not None else None,
                        parameters=sum(p.numel() for p in model.parameters()) if model is not None else None, target=target,
                        independent_fields=[p["field_cluster"] for p in batch], errors=errors,
                        accuracy_passed=passed, reference_accepted=all(e.get("reference_accepted") for e in errors) if errors else False,
                        cost_seconds=cost.get("median_seconds"), per_field_seconds=cost.get("median_seconds", 0.) / batch_size if cost else None,
                        input_transfer_seconds=transfer, cost=cost,
                        numerical_screening_seconds=screening_seconds.get(model_id),
                        total_measured_trial_seconds=(screening_seconds.get(model_id, 0.) + cost.get("cold_seconds", 0.)
                            + cost.get("warmup_seconds", 0.) + cost.get("total_seconds", 0.)) if model is not None else None,
                        peak_allocated_bytes=timing.get("peak_allocated_bytes"),
                        peak_reserved_bytes=timing.get("peak_reserved_bytes"),
                        memory_scope="peak for paired workload; not per-family attributable",
                        cache=model.cache_metadata if model is not None else None,
                        failure=failures.get(model_id) if model is not None else "FROZEN_MODEL_UNAVAILABLE",
                        status="MODEL_UNAVAILABLE" if model is None else "NUMERICAL_FAILURE" if model_id in failures else "COMPLETED",
                        timing_scope="resident-state complete rollout; shared input transfer recorded separately",
                        cold_scope="coefficient cache cleared after numerical screening; process/GPU already initialized",
                        timing_order=timing.get("rounds", []),
                        scope="accuracy-backed project controls; no official FNO reproduction")
                    rows.append(row); group.append(row)
                    write_json(ctx.path / "scaling_rows.json", {"rows": rows})
                    ctx.record(f"{track}/{n}/batch{batch_size}/{model_id}", ["G4"], metrics=row,
                        config={"family": spec["family"], "track": track, "grid": n, "batch_size": batch_size},
                        checks=[check("all-batch-fields-independent", len(set(row["independent_fields"])), batch_size, "eq", category="correctness"),
                                check("independent-accuracy", passed, True, "eq", category="gap"),
                                check("finite-output", model_id not in failures if model is not None else None, True, "eq", category="math")])
                rank = next((r for r in group if r["family"] == "rank1"), None)
                for control in group:
                    if not rank or control["family"] == "rank1":
                        continue
                    comparisons.append(record_crossover(ctx, rank, control))
                    write_json(ctx.path / "scaling_comparisons.json", {"rows": comparisons})
    write_json(ctx.path / "scaling_comparisons.json", {"rows": comparisons})
    details = dict(workload_rows=len(rows), accuracy_passes=sum(r["accuracy_passed"] is True for r in rows),
        accuracy_na=sum(r["accuracy_passed"] is None for r in rows),
        comparison_rows=len(comparisons), observed_crossovers=sum(r["observed_threshold_passed"] is True for r in comparisons),
        crossover_na=sum(r["observed_threshold_passed"] is None for r in comparisons),
        practical_speedup_threshold=ctx.protocol["practical_speedup"],
        throughput_claim_scope="observed median threshold only; no robust confidence-interval or universal superiority claim",
        scope="One declared seed/data subset; independent fields within each batch; workloads remain paired")
    write_json(ctx.path / "gate.json", dict(gate="G4", **details,
        scientific_outcome="MEASURED_ACCURACY_AND_THROUGHPUT; SEE INDIVIDUAL WORKLOADS"))
    return details
