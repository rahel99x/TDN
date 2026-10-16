"""Phase 1 independent-parent scaling, with bounded acquisition and sealed fits.

Only the source-bound native CLI admits the production study. Private reduced
fixtures exercise a declared prefix of the development cohort, not confirmation.
The historical R7 protocol and its model, objective and references are reused
without changing their definitions.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from . import r7_actions as actions
from . import r7_references as references
from . import r7_workflow as r7
from . import workflow as legacy
from .model import FixedFeatureResponse
from .scaling_data import parents, source_bank
from .scaling_protocol import SCHEMA, protocol, validate_config

ROOT = legacy.ROOT


def _csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True, allow_nan=False)
                             if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def _json(path, value):
    legacy._json(path, value)


def _cell_id(parent_count, source_count):
    return f"fit-p{parent_count:03d}-s{source_count:02d}"


def _analysis_row(row):
    result = dict(row)
    if "design" in result:
        result["design"] = result["design"].reshape(40, -1).T
    for key in ("base", "teacher", "matched_mean", "refined", "matched_mean_refined"):
        if key in result:
            result[key] = result[key].reshape(-1)
    return result


class ScalingStudy:
    """One bounded bank; every query and model attempt has a planned identity."""

    def __init__(self, output, config, *, allow_fixture=False):
        validate_config(config, allow_fixture=allow_fixture)
        self.output, self.config = Path(output), config
        self.start = time.monotonic()
        self.deadline = self.start + config["budget"]["study_seconds"]
        self.model = FixedFeatureResponse()
        self.parents, self.contexts, self.records = {}, {}, {}
        self.index, self.work, self.timing, self.timing_records = [], [], [], []
        self.selection = None
        self.fitting_seconds = 0.
        self.attempted = 0
        self.cells = [_cell_id(n, k) for n in config["parent_counts"]
                      for k in config["calibration_counts"]]
        self.cell_specs = {_cell_id(n, k): (n, k) for n in config["parent_counts"]
                           for k in config["calibration_counts"]}
        self.controls = [f"ordered_{n}" for n in config["ordered_panels"]]
        self.controls += [f"lawson_{n}" for n in config["lawson_steps"]]
        self.methods = self.cells + self.controls
        for split in ("train", "validation", "diagnostic"):
            for parent in parents(config, split):
                pid = parent["parent_id"]
                self.parents[pid] = parent
                for ci, (start, stop) in enumerate(config["intervals"]):
                    cid = f"{pid}-context-{ci:02d}"
                    self.contexts[cid] = dict(parent=parent, start=start, stop=stop, index=ci)
                    for role in (("calibration", "withheld") if split == "train" else ("withheld",)):
                        for source in source_bank(parent["seed"], ci, role, config):
                            qid = f"{cid}-{source['source_id']}"
                            self.index.append(dict(query_id=qid, context_id=cid, parent_id=pid,
                                parent_family=parent["parent_family"], split=split, start=start, stop=stop,
                                source_id=source["source_id"], source_index=source["source_index"],
                                role=role, family=source["family"], status="NOT_ACQUIRED", reference=None,
                                array_path=f"queries/{qid}.npz", record_path=f"queries/{qid}.json"))
        if len(self.index) != config["reference_count"] or len(self.index) > config["max_reference_queries"]:
            raise ValueError("Planned query count does not match the frozen reference cap")
        self.lookup = {row["query_id"]: row for row in self.index}
        if len(self.lookup) != len(self.index):
            raise ValueError("Duplicate planned query identities")
        self.metrics = {(row["query_id"], method): self.pending_metric(row, method)
                        for row in self.index if row["role"] == "withheld" for method in self.methods}
        for name in ("queries", "fits", "predictions"):
            (self.output / name).mkdir()
        _json(self.output / "parents.json", [dict(
            **{k: v for k, v in p.items() if k not in ("u", "eq", "geom")},
            kappa=float(p["eq"].kappa), reaction_rate=float(p["eq"].reaction_rate),
            grid=list(p["geom"].grid), lengths=list(p["geom"].lengths))
            for p in self.parents.values()])
        np.savez_compressed(self.output / "parents.npz", **{
            pid: p["u"].detach().numpy() for pid, p in self.parents.items()})
        self.persist()

    def pending_metric(self, row, method):
        identity = {key: row[key] for key in ("query_id", "context_id", "parent_id",
            "parent_family", "split", "source_id", "source_index", "role", "family", "start", "stop")}
        n, k = self.cell_specs.get(method, (None, None))
        membership = "not_applicable"
        if n is not None:
            membership = ("not_training_split" if row["split"] != "train" else
                          "in_cohort" if self.parents[row["parent_id"]]["parent_index"] < n else "outside_cohort")
        return dict(**identity, method=method, status="NOT_EVALUATED", rms=None, maximum=None,
                    mean_error=None, centered_rms=None, reference_accepted=False,
                    training_parent_count=n, calibration_source_count=k,
                    training_membership=membership, scientific_promotion=False)

    def check(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Frozen Phase 1 study deadline exhausted; no automatic expansion")

    def persist(self):
        _json(self.output / "query_index.json", self.index)
        _json(self.output / "references.json", [dict(query_id=row["query_id"], reference=row["reference"])
            for row in self.index if row["reference"] is not None])
        _json(self.output / "denominators.json", dict(
            planned=len(self.index), attempted=self.attempted,
            accepted=sum(r["status"] == "ACCEPTED" for r in self.index),
            unresolved=sum(r["status"] == "UNRESOLVED" for r in self.index),
            failed=sum(r["status"] == "FAILED" for r in self.index),
            missing=sum(r["status"] == "NOT_ACQUIRED" for r in self.index),
            independent_parent_count=len(self.parents),
            parents_by_split={split: sum(p["split"] == split for p in self.parents.values())
                              for split in ("train", "validation", "diagnostic")},
            descendants_are_independent=False))
        expected_timing = len(self.timing_contexts()) * (len(self.controls) + 1) * (
            1 + self.config["timing"]["warmup"] + self.config["timing"]["rounds"])
        values = list(self.metrics.values())
        _json(self.output / "evaluation_denominators.json", dict(
            expected_metrics=len(values), attempted_metrics=sum(r["status"] != "NOT_EVALUATED" for r in values),
            measured_metrics=sum(r["status"] == "MEASURED" for r in values),
            failed_metrics=sum(r["status"] == "FAILED" for r in values),
            unresolved_metrics=sum(r["status"] == "UNRESOLVED" for r in values),
            missing_metrics=sum(r["status"] == "NOT_EVALUATED" for r in values),
            expected_fit_cells=len(self.cells), selected_fit_cells=len(self.selection["cells"]) if self.selection else 0,
            expected_timing_bundles=expected_timing, attempted_timing_bundles=len(self.timing_records),
            failed_timing_bundles=sum(r["status"] != "PASS" for r in self.timing_records),
            missing_timing_bundles=expected_timing-len(self.timing_records)))

    def timing_contexts(self):
        indices = self.config["timing"]["parent_indices"]
        return [cid for cid, ctx in self.contexts.items() if ctx["parent"]["split"] == "diagnostic"
                and ctx["parent"]["parent_index"] in indices]

    def rows(self, *, split=None, role=None):
        return [self.records.get(row["query_id"], row) for row in self.index
                if (split is None or row["split"] == split) and (role is None or row["role"] == role)]

    def acquire(self, cid, role):
        ctx = self.contexts[cid]
        parent = ctx["parent"]
        if (parent["split"] == "diagnostic" or (parent["split"] == "train" and role == "withheld")) and self.selection is None:
            raise RuntimeError("All model selections must seal before diagnostic or training-withheld acquisition")
        sources = source_bank(parent["seed"], ctx["index"], role, self.config)
        try:
            for source in sources:
                qid = f"{cid}-{source['source_id']}"
                if qid in self.records:
                    continue
                if qid not in self.lookup:
                    raise ValueError("Query was not preregistered")
                self.check()
                if self.attempted >= self.config["max_reference_queries"]:
                    raise RuntimeError("Frozen Phase 1 reference-query cap reached")
                row = self.lookup[qid]
                self.attempted += 1
                arrays = dict(initial=parent["u"].numpy(), source=source["z"].numpy())
                legacy._journal(self.output, "query_started", query_id=qid, split=parent["split"], role=role)
                try:
                    began = time.perf_counter()
                    try:
                        result = references.acquire_reference(parent["u"], source["z"], ctx["start"], ctx["stop"],
                            parent["eq"], parent["geom"], self.config, budget=self)
                    finally:
                        row["reference_seconds"] = time.perf_counter()-began
                    row["reference"] = result["reference"]
                    row["status"] = ("ACCEPTED" if result["reference"]["accepted"] and
                                     result["reference"].get("matched_mean", {}).get("accepted", False)
                                     else "UNRESOLVED")
                    for name in ("teacher", "refined", "matched_mean", "matched_mean_refined"):
                        if result.get(name) is not None:
                            arrays[name] = r7._numpy(result[name])
                    if row["status"] == "ACCEPTED" and any(key not in arrays for key in ("teacher", "matched_mean")):
                        raise ValueError("Accepted reference is missing its numeric action")
                    self.check()
                    began = time.perf_counter()
                    try:
                        design = actions.design_context(parent["u"], source["z"], ctx["start"], ctx["stop"],
                            parent["eq"], parent["geom"], model=self.model)
                    finally:
                        row["design_seconds"] = time.perf_counter()-began
                    arrays.update({key: r7._numpy(design[key]) for key in ("base", "branches", "features", "design")})
                    if any(not np.isfinite(array).all() for array in arrays.values()):
                        raise ArithmeticError("Nonfinite acquired reference or design; no row deletion")
                    self.work.append(dict(query_id=qid, method="training_design", work=design["work"]))
                except Exception as error:
                    row.update(status="FAILED", error=f"{type(error).__name__}: {error}")
                    raise
                finally:
                    path = self.output / row["array_path"]
                    np.savez_compressed(path, **arrays)
                    row["array_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                    self.records[qid] = dict(**row, **arrays)
                    _json(self.output / row["record_path"], row)
                    legacy._journal(self.output, "query_finished", query_id=qid, status=row["status"])
        finally:
            self.persist()
        return [self.records[f"{cid}-{s['source_id']}"] for s in sources]

    def freeze_selections(self):
        from .scaling_analysis import fit_nested_models
        if self.selection is not None:
            raise RuntimeError("Model selections already sealed")
        train_ids = [pid for pid, p in self.parents.items() if p["split"] == "train"]

        def on_cell(cell):
            _json(self.output / "fits" / (cell["model_id"] + ".json"), cell)
            legacy._journal(self.output, "fit_cell_finished", model_id=cell["model_id"], status=cell["status"])

        began = time.perf_counter()
        try:
            selection = fit_nested_models(
                [_analysis_row(row) for row in self.rows(split="train", role="calibration")],
                [_analysis_row(row) for row in self.rows(split="validation", role="withheld")],
                training_parent_ids=train_ids, parent_counts=self.config["parent_counts"],
                source_counts=self.config["calibration_counts"], ridge_menu=self.config["ridge_menu"],
                expected_contexts_per_parent=len(self.config["intervals"]),
                expected_validation_parent_ids=[pid for pid, p in self.parents.items() if p["split"] == "validation"],
                expected_withheld_sources=self.config["withheld_sources"],
                budget_check=self.check, on_cell=on_cell)
        finally:
            self.fitting_seconds = time.perf_counter()-began
        _json(self.output / "selection.json", selection)
        if not selection["all_selections_sealed"] or len(selection["cells"]) != len(self.cells):
            raise RuntimeError("Unresolved training/validation data block the complete selection seal")
        self.selection = selection
        self.models = {}
        for cell in selection["cells"]:
            model = FixedFeatureResponse()
            model.set_readout(cell["selected"]["coefficients"])
            self.models[cell["model_id"]] = model
        legacy._journal(self.output, "all_selections_frozen", selection_sha256=legacy._digest(selection),
            model_ids=self.cells, diagnostic_truth_accessed=False, training_withheld_truth_accessed=False)
        self.persist()
        return selection

    def score(self, rows):
        from .scaling_analysis import evaluate_model, evaluate_prediction
        if self.selection is None:
            raise RuntimeError("Scoring requires sealed selections")
        cells = {c["model_id"]: c for c in self.selection["cells"]}
        for row in rows:
            if row["role"] != "withheld":
                raise ValueError("Phase 1 evaluation uses declared withheld sources")
            ctx = self.contexts[row["context_id"]]
            parent = ctx["parent"]
            predictions = {}
            try:
                for method in self.methods:
                    self.check()
                    metric = self.pending_metric(row, method)
                    work = {}
                    began = time.perf_counter()
                    try:
                        if method in cells:
                            value = actions.fresh_action("fixed_feature_fitted", parent["u"], torch.from_numpy(row["source"]),
                                ctx["start"], ctx["stop"], parent["eq"], parent["geom"], model=self.models[method], work=work)
                            coefficients = np.asarray(cells[method]["selected"]["coefficients"])
                            reconstructed = row["base"] + np.einsum("j,j...->...", coefficients, row["design"])
                            discrepancy = float(np.max(np.abs(r7._numpy(value)-reconstructed)))
                            tolerance = 64*np.finfo(np.float64).eps*max(1., float(np.max(np.abs(reconstructed))))
                            metric.update(inference_parity_maximum=discrepancy, inference_parity_tolerance=tolerance)
                            if not np.isfinite(discrepancy) or discrepancy > tolerance:
                                raise ArithmeticError("Fresh inference differs from its sealed affine design")
                        else:
                            value = actions.fresh_action(method, parent["u"], torch.from_numpy(row["source"]),
                                ctx["start"], ctx["stop"], parent["eq"], parent["geom"], work=work)
                        prediction = r7._numpy(value)
                        if not np.isfinite(prediction).all():
                            raise ArithmeticError("Nonfinite fresh prediction")
                        predictions[method] = prediction
                        if method in cells:
                            metric.update(evaluate_model(cells[method], [_analysis_row(row)],
                                predictions={row["query_id"]: prediction.reshape(-1)})[0])
                        else:
                            metric.update(evaluate_prediction(_analysis_row(row), prediction.reshape(-1), method=method))
                        accepted = row["status"] == "ACCEPTED"
                        metric.update(reference_accepted=accepted,
                            reference_uncertainty_rms=row["reference"].get("uncertainty_rms"),
                            reference_uncertainty_maximum=row["reference"].get("uncertainty_maximum"),
                            prediction_mode="fresh_deployable_action", deployable=True, speedup_claim=False)
                        if metric["status"] == "MEASURED":
                            metric.update(r7._correction(row, prediction, self.config))
                            metric["K_minus_P_rms"] = r7._metrics(prediction-row["base"])["rms"]
                    except (ArithmeticError, ValueError, RuntimeError) as error:
                        metric.update(status="FAILED", rms=None, maximum=None, mean_error=None, centered_rms=None,
                                      error=f"{type(error).__name__}: {error}")
                    self.metrics[(row["query_id"], method)] = metric
                    self.work.append(dict(query_id=row["query_id"], method=method, work=work,
                                          status=metric["status"], accuracy_evaluation_seconds=time.perf_counter()-began))
            finally:
                path = self.output / "predictions" / (row["query_id"] + ".npz")
                np.savez_compressed(path, **predictions)
                legacy._journal(self.output, "query_evaluated", query_id=row["query_id"],
                    prediction_path=str(path.relative_to(self.output)), prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    methods=list(predictions))
                # Per-query records survive interruptions without rewriting a
                # multi-megabyte complete table after every individual action.
                _json(path.with_suffix(".json"), [self.metrics[(row["query_id"], method)] for method in self.methods])

    def time_context(self, cid):
        ctx = self.contexts[cid]
        p = ctx["parent"]
        sources = source_bank(p["seed"], ctx["index"], "withheld", self.config)
        model_id = _cell_id(max(self.config["parent_counts"]), max(self.config["calibration_counts"]))
        methods = {}
        for name in [*self.controls, model_id]:
            def bundle(method=name):
                return torch.stack([actions.fresh_action("fixed_feature_fitted" if method == model_id else method,
                    p["u"], src["z"], ctx["start"], ctx["stop"], p["eq"], p["geom"],
                    model=self.models[model_id] if method == model_id else None) for src in sources])
            methods[name] = bundle

        def on_record(row):
            self.timing_records.append(dict(context_id=cid, parent_id=p["parent_id"],
                                           parent_family=p["parent_family"], **row))
            legacy._journal(self.output, "timing_observation", context_id=cid, **row)
            _json(self.output / "timing_records.json", self.timing_records)

        result = actions.measure_paired_actions(methods, rounds=self.config["timing"]["rounds"],
            warmup=self.config["timing"]["warmup"], seed=self.config["timing"]["seed"]+p["parent_index"]*100+ctx["index"],
            budget=self.check, on_record=on_record)
        self.timing.append(dict(context_id=cid, parent_id=p["parent_id"], parent_family=p["parent_family"], **result))
        _json(self.output / "timing.json", self.timing)

    def costs(self):
        """Charge actual shared acquisition once; expose each cell's label demand."""
        def demand(rows):
            return dict(planned=len(rows), attempted=sum("reference_seconds" in r for r in rows),
                        accepted=sum(r["status"] == "ACCEPTED" for r in rows),
                        reference_seconds=sum(r.get("reference_seconds", 0.) for r in rows),
                        design_seconds=sum(r.get("design_seconds", 0.) for r in rows))

        validation = demand([r for r in self.index if r["split"] == "validation"])
        cells = []
        for model_id, (n, k) in self.cell_specs.items():
            training = demand([r for r in self.index if r["split"] == "train" and r["role"] == "calibration"
                and self.parents[r["parent_id"]]["parent_index"] < n and r["source_index"] < k])
            path = self.output / "fits" / (model_id + ".json")
            fitting = json.loads(path.read_text()).get("fitting_seconds") if path.exists() else None
            cells.append(dict(model_id=model_id, training_parent_count=n, calibration_source_count=k,
                training=training, validation=validation, training_reference_seconds=training["reference_seconds"],
                validation_reference_seconds=validation["reference_seconds"], fitting_seconds=fitting))
        return dict(scope="Observed component durations, not additive to wall time; reference/design acquisition is shared once globally. Per-cell demands overlap and must not be summed. Missing costs are not zero-cost successes.",
            global_acquisition=demand(self.index), fitting_seconds=self.fitting_seconds,
            accuracy_evaluation_seconds=sum(r.get("accuracy_evaluation_seconds", 0.) for r in self.work),
            timing_seconds=sum(r["seconds"] for r in self.timing_records), cells=cells)

    def finalize(self):
        from .scaling_analysis import summarize_learning_curves
        self.persist()
        rows = list(self.metrics.values())
        expected = {split: [pid for pid, parent in self.parents.items() if parent["split"] == split]
                    for split in ("train", "validation", "diagnostic")}
        analysis = summarize_learning_curves(rows, expected_parent_ids=expected)
        costs = self.costs()
        _json(self.output / "costs.json", costs)
        cell_costs = {c["model_id"]: c for c in costs["cells"]}
        for group in analysis["groups"]:
            if group["method"] in cell_costs:
                group.update({key: cell_costs[group["method"]][key] for key in (
                    "training_reference_seconds", "validation_reference_seconds", "fitting_seconds")})
        _json(self.output / "learning_curves.json", analysis["groups"])
        _json(self.output / "parent_metrics.json", analysis["parent_rows"])
        _json(self.output / "paired_comparisons.json", {key: value for key, value in analysis.items()
            if key not in ("groups", "parent_rows")})
        for name, values in (("metrics", rows), ("timing", self.timing), ("timing_records", self.timing_records), ("work", self.work)):
            _json(self.output / (name+".json"), values)
        for name, values in (("metrics", rows), ("parent_metrics", analysis["parent_rows"]), ("learning_curves", analysis["groups"])):
            _csv(self.output / (name+".csv"), values)
        precision = []
        for timed in self.timing:
            for method, observations in timed["summary"].items():
                matching = [row for row in rows if row["context_id"] == timed["context_id"] and row["method"] == method]
                resolved = len(matching) == self.config["withheld_sources"] and all(row["status"] == "MEASURED" for row in matching)
                precision.append(dict(context_id=timed["context_id"], parent_id=timed["parent_id"],
                    parent_family=timed["parent_family"], method=method, all_sources_resolved=resolved,
                    pooled_source_rms=float(np.sqrt(np.mean([row["rms"]**2 for row in matching]))) if resolved else None,
                    maximum=max(row["maximum"] for row in matching) if resolved else None,
                    **observations, speedup_claim=False))
        _json(self.output / "work_precision.json", precision)
        _csv(self.output / "work_precision.csv", precision)
        _json(self.output / "decisions.json", dict(status="DESCRIPTIVE_DEVELOPMENT_ONLY",
            automatic_expansion=False, architecture_promotion=False, matched_tolerance_speedup=False,
            fresh_confirmation=False, next_gate="Review independent-parent and source-count curves, failures, uncertainty and costs before separately freezing Phase 2 or a larger cohort",
            limitations=self.config["deferred"]))


def _study(output, config, *, allow_fixture=False):
    legacy._journal(output, "protocol_frozen", protocol_sha256=legacy._digest(config))
    study = ScalingStudy(output, config, allow_fixture=allow_fixture)
    try:
        for cid, ctx in study.contexts.items():
            split = ctx["parent"]["split"]
            if split == "train":
                study.acquire(cid, "calibration")
            elif split == "validation":
                study.acquire(cid, "withheld")
        study.freeze_selections()
        timed = set(study.timing_contexts())
        for cid, ctx in study.contexts.items():
            study.check()
            rows = study.acquire(cid, "withheld")
            study.score(rows)
            if cid in timed:
                study.time_context(cid)
            study.persist()
            legacy._journal(output, "context_complete", context_id=cid)
        study.check()
        return dict(reference_count=study.attempted, metric_rows=len(study.metrics),
            measured_metrics=sum(row["status"] == "MEASURED" for row in study.metrics.values()),
            failed_metrics=sum(row["status"] == "FAILED" for row in study.metrics.values()),
            unresolved_references=sum(row["status"] != "ACCEPTED" for row in study.index),
            selected_models=len(study.selection["cells"]), timing_bundles=len(study.timing_records),
            failed_timing_bundles=sum(row["status"] != "PASS" for row in study.timing_records),
            native_gpu_readiness=False, matched_accuracy_speedup_established=False)
    finally:
        study.finalize()


def execute(command, output=None, *, device="cpu", contract=None):
    if device != "cpu":
        raise ValueError("Phase 1 requires explicit CPU execution")
    config = protocol(command)
    validate_config(config)
    if command == "study":
        if contract is None:
            raise ValueError("Study requires this attempt's source-bound native Raven contract and verification release")
        try:
            from response_preconditioner_scaling_native import require_study_context
        except ImportError as error:
            raise ValueError("Use the native Phase 1 CLI and its source-bound Raven contract") from error
        require_study_context(contract, output)
    output = legacy.create_output("verify", output)
    before = legacy.source_snapshot()
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    started = time.perf_counter()
    summary = dict(schema=SCHEMA, command=command, status="RUNNING", decision="BOUNDED_DEVELOPMENT",
        device="cpu", scientific_promotion=False, native_gpu_readiness=False, output=str(output),
        source_fingerprint=before["fingerprint"], protocol_sha256=legacy._digest(config), historical_gate="B STOP unchanged")
    for name, value in (("protocol", config), ("source", before), ("software", legacy._software()), ("summary", summary)):
        _json(output / (name+".json"), value)
    try:
        summary.update(r7._verify(output, config) if command == "verify" else _study(output, config))
        after = legacy.source_snapshot()
        _json(output / "source_after.json", after)
        if before != after:
            raise RuntimeError("Source changed during Phase 1 execution; cannot seal")
        summary["status"] = "COMPLETED"
    except Exception as error:
        summary.update(status="FAILED", error_type=type(error).__name__, error=str(error))
        (output / "failure.txt").write_text(traceback.format_exc())
    finally:
        summary["elapsed_seconds"] = time.perf_counter()-started
        _json(output / "summary.json", summary)
        legacy._journal(output, "run_finished", status=summary["status"])
        (output / "README.md").write_text(
            "# Phase 1 independent-parent scaling evidence\n\n"
            f"Execution: **{summary['status']}**. Descriptive development; no automatic promotion.\n\n"
            "Read protocol.json and selection.json before interpreting learning_curves.csv and parent_metrics.csv. "
            "Independent physical parents are the sampling units; source vectors and times are paired descendants. "
            "All selected models seal before diagnostic or training-withheld references. Training membership is explicit. "
            "One nested parent cohort/order does not establish population-level scaling or fresh confirmation. "
            "The 5-by-5 morphologies are finite-grid development fields, not continuum-resolved fronts. "
            "query_index.json, denominators.json and evaluation_denominators.json retain missing and failed attempts. "
            "Per-query NPZ/JSON and prediction files preserve partial work; NPZ loads require allow_pickle=False. "
            "Complete fresh-source timings cover only the largest fit on the predeclared diagnostic subset. "
            "CPU/RSS worker telemetry is separate from science; no GPU or endpoint/rollout claim is made. "
            "See docs/RESPONSE_PRECONDITIONER_SCALING_PHASE1.md for definitions, limits and the runbook.\n")
        files = [dict(path=str(path.relative_to(output)), bytes=path.stat().st_size,
                      sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                 for path in sorted(output.rglob("*")) if path.is_file() and "pytest-temp" not in path.parts]
        _json(output / "artifacts.json", dict(schema=SCHEMA, source_fingerprint=before["fingerprint"],
            status=summary["status"], self_excluded="artifacts.json", files=files))
        torch.set_num_threads(old_threads)
    return summary
