"""Bounded R7 identification, with sealed selection and portable all-attempt evidence."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np
import torch

from . import workflow as legacy
from .fitting import fit_parent_balanced_ridge
from .model import FixedFeatureResponse
from .r7_protocol import SCHEMA, VERIFY_TESTS, protocol

ROOT = legacy.ROOT


def _json(path, value):
    legacy._json(path, value)


def _numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _metrics(value):
    x = _numpy(value).astype(np.float64)
    return dict(rms=float(np.sqrt(np.mean(x*x))), maximum=float(np.max(np.abs(x))),
                mean_error=float(np.mean(x)), centered_rms=float(np.std(x)))


def source_bank(parent_seed, context_index, role, config):
    """Sources depend only on the preregistered seed, interval index and role."""
    if role not in ("calibration", "withheld"):
        raise ValueError("Unknown source role")
    rng = np.random.default_rng(np.random.SeedSequence(
        [parent_seed, context_index, 701 if role == "calibration" else 702]))
    families = config[role + "_families"]
    x, y = np.meshgrid(np.arange(5)/5, np.arange(5)/5, indexing="ij")
    result = []
    for i, family in enumerate(families):
        if family == "dense_random":
            z = rng.normal(size=(5, 5))
        elif family in ("low_frequency", "high_frequency"):
            k = 1 if family == "low_frequency" else 2
            z = np.cos(2*np.pi*k*(x+y)+rng.uniform(0, 2*np.pi))
        elif family == "constant":
            z = np.ones((5, 5))
        else:
            raise ValueError("Unknown frozen source family")
        z = z / np.sqrt(np.mean(z*z))
        result.append(dict(source_id=f"{role}-{i:02d}", role=role, family=family,
                           z=torch.from_numpy(z.copy())[None, None]))
    return result


class Study:
    """Each acquired query is persisted before fitting or subsequent acquisition."""
    def __init__(self, output, config):
        self.output, self.config = Path(output), config
        self.start = time.monotonic()
        self.deadline = self.start + config["budget"]["study_seconds"]
        self.model = FixedFeatureResponse()
        self.records, self.metrics, self.work, self.timing = {}, [], [], []
        self.timing_records = []
        self.parents, self.contexts, self.index = {}, {}, []
        self.attempted = 0
        (self.output / "queries").mkdir()
        (self.output / "fits").mkdir()
        (self.output / "sensitivity").mkdir()
        # Parent fields carry no labels. Preserve the old RNG draws for physics.
        for split in ("train", "validation", "diagnostic"):
            for parent in legacy.parents(config, split):
                pid = parent["parent_id"]
                self.parents[pid] = parent
                for ci, (s, t) in enumerate(config["intervals"]):
                    cid = f"{pid}-context-{ci:02d}"
                    self.contexts[cid] = dict(parent=parent, start=s, stop=t, index=ci)
                    for role in ("calibration", "withheld"):
                        for src in source_bank(parent["seed"], ci, role, config):
                            qid = f"{cid}-{src['source_id']}"
                            self.index.append(dict(query_id=qid, context_id=cid, parent_id=pid,
                                split=split, start=s, stop=t, source_id=src["source_id"], role=role,
                                family=src["family"], status="NOT_ACQUIRED", reference=None,
                                array_path=f"queries/{qid}.npz"))
        if len(self.index) > config["max_reference_queries"]:
            raise ValueError("Planned queries exceed the frozen cap")
        self.persist()
        np.savez_compressed(self.output / "parents.npz", **{
            pid: p["u"].numpy() for pid, p in self.parents.items()})
        _json(self.output / "parents.json", [dict(parent_id=pid, split=p["split"],
            seed=p["seed"], kappa=p["eq"].kappa, reaction_rate=p["eq"].reaction_rate,
            grid=list(p["geom"].grid), lengths=list(p["geom"].lengths))
            for pid, p in self.parents.items()])

    def check(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Frozen R7 study deadline exhausted; no automatic expansion")

    def persist(self):
        _json(self.output / "query_index.json", self.index)
        _json(self.output / "references.json", [dict(query_id=r["query_id"], reference=r["reference"])
                                               for r in self.index if r["reference"] is not None])
        _json(self.output / "denominators.json", dict(planned=len(self.index), attempted=self.attempted,
            accepted=sum(r["status"] == "ACCEPTED" for r in self.index),
            unresolved=sum(r["status"] == "UNRESOLVED" for r in self.index),
            failed=sum(r["status"] == "FAILED" for r in self.index),
            missing=sum(r["status"] == "NOT_ACQUIRED" for r in self.index),
            independent_parent_count=len(self.parents), descendants_are_independent=False))
        method_count = len(self.config["ordered_panels"])+len(self.config["lawson_steps"])+2+len(self.config["ridge_menu"])
        diagnostic_contexts = sum(ctx["parent"]["split"] == "diagnostic" for ctx in self.contexts.values())
        expected_timing = diagnostic_contexts*(len(self.config["ordered_panels"])+len(self.config["lawson_steps"])+1)*(1+self.config["timing"]["warmup"]+self.config["timing"]["rounds"])
        _json(self.output / "evaluation_denominators.json", dict(
            expected_metrics=len(self.index)*method_count, attempted_metrics=len(self.metrics),
            measured_metrics=sum(r["status"] == "MEASURED" for r in self.metrics),
            failed_metrics=sum(r["status"] == "FAILED" for r in self.metrics),
            unresolved_metrics=sum(r["status"] == "UNRESOLVED" for r in self.metrics),
            missing_metrics=len(self.index)*method_count-len(self.metrics),
            expected_timing_bundles=expected_timing, attempted_timing_bundles=len(self.timing_records),
            failed_timing_bundles=sum(r["status"] != "PASS" for r in self.timing_records),
            missing_timing_bundles=expected_timing-len(self.timing_records)))

    def timing_record(self, cid, row):
        self.timing_records.append(dict(context_id=cid, **row))
        legacy._journal(self.output, "timing_observation", context_id=cid, **row)
        _json(self.output / "timing_records.json", self.timing_records)
        self.persist()

    def acquire(self, cid, role):
        from .r7_actions import design_context
        from .r7_references import acquire_reference
        ctx = self.contexts[cid]
        p = ctx["parent"]
        sources = source_bank(p["seed"], ctx["index"], role, self.config)
        for src in sources:
            qid = f"{cid}-{src['source_id']}"
            if qid in self.records:
                continue
            self.check()
            if self.attempted >= self.config["max_reference_queries"]:
                raise RuntimeError("Frozen reference-query cap reached")
            row = next(r for r in self.index if r["query_id"] == qid)
            self.attempted += 1
            legacy._journal(self.output, "query_started", query_id=qid, role=role, split=p["split"])
            arrays = dict(initial=p["u"].numpy(), source=src["z"].numpy())
            try:
                record = acquire_reference(p["u"], src["z"], ctx["start"], ctx["stop"],
                                           p["eq"], p["geom"], self.config, budget=self)
                row["reference"] = record["reference"]
                row["status"] = "ACCEPTED" if record["reference"]["accepted"] else "UNRESOLVED"
                for key in ("teacher", "refined", "matched_mean", "matched_mean_refined"):
                    if record.get(key) is not None:
                        arrays[key] = _numpy(record[key])
                design = design_context(p["u"], src["z"], ctx["start"], ctx["stop"],
                                        p["eq"], p["geom"], model=self.model)
                arrays.update({key: _numpy(design[key]) for key in ("base", "branches", "features", "design")})
                self.work.append(dict(query_id=qid, method="training_design", work=design["work"]))
                self.records[qid] = dict(**row, **arrays)
            except Exception as error:
                row.update(status="FAILED", error=f"{type(error).__name__}: {error}")
                self.records[qid] = dict(row)
                raise
            finally:
                np.savez_compressed(self.output / row["array_path"], **arrays)
                row["array_sha256"] = hashlib.sha256((self.output / row["array_path"]).read_bytes()).hexdigest()
                _json(self.output / "work.json", self.work)
                self.persist()
                legacy._journal(self.output, "query_finished", query_id=qid, status=row["status"])
        return [self.records[f"{cid}-{src['source_id']}"] for src in sources]

    def rows(self, *, split=None, role=None, cid=None):
        return [r for r in self.records.values() if (split is None or r["split"] == split)
                and (role is None or r["role"] == role) and (cid is None or r["context_id"] == cid)]


def _matrix(rows, key="design"):
    A = np.concatenate([r[key].reshape(r[key].shape[0], -1).T for r in rows])
    b = np.concatenate([(r["teacher"]-r["base"]).reshape(-1) for r in rows])
    return A, b


def _require_resolved(rows, label):
    if not rows or any(r["status"] != "ACCEPTED" for r in rows):
        raise RuntimeError(f"Unresolved {label} references block fitting/selection; no rows dropped")


def _global_fit(study):
    training = study.rows(split="train", role="calibration")
    validation = study.rows(split="validation", role="withheld")
    _require_resolved(training, "training")
    _require_resolved(validation, "validation")
    A, b = _matrix(training)
    Av, bv = _matrix(validation)
    ids = [r["parent_id"] for r in training for _ in range(r["teacher"].size)]
    candidates = [dict(name="zero_readout", ridge=None, coefficients=[0.]*40, eligible=True,
                       validation_loss=float(np.mean(bv*bv)), diagnostics={"status": "INITIALIZATION"})]
    started = time.perf_counter()
    for ridge in study.config["ridge_menu"]:
        study.check()
        try:
            fit = fit_parent_balanced_ridge(A, b, ids, ridge=ridge)
            c, d = fit["coefficients"].numpy(), fit["diagnostics"]
            candidates.append(dict(name=f"ridge_{ridge:g}", ridge=ridge, coefficients=c.tolist(),
                eligible=bool(d["optimality_verified"] and d["baseline_nonregression_verified"]),
                diagnostics=d, validation_loss=float(np.mean((Av@c-bv)**2))))
        except (ValueError, ArithmeticError, RuntimeError) as error:
            candidates.append(dict(name=f"ridge_{ridge:g}", ridge=ridge, coefficients=None,
                eligible=False, validation_loss=None, error=f"{type(error).__name__}: {error}"))
    selected = min((r for r in candidates if r["eligible"]), key=lambda r: r["validation_loss"])
    selection = dict(selected=selected, candidates=candidates,
        criterion=study.config["selection"], fitting_seconds=time.perf_counter()-started,
        train_query_ids=[r["query_id"] for r in training], validation_query_ids=[r["query_id"] for r in validation],
        diagnostic_truth_accessed=False, training_withheld_truth_accessed=False,
        objective="action-only parent-balanced physical MSE plus original positive ridge")
    _json(study.output / "selection.json", selection)
    study.model.set_readout(selected["coefficients"])
    legacy._journal(study.output, "selection_frozen", selection_sha256=legacy._digest(selection))
    return selection


def _local_fit(study, cid, rows):
    from .r7_fitting import fit_local_projection, fit_local_ridge
    started = time.perf_counter()
    if any(r["status"] != "ACCEPTED" for r in rows):
        payload = dict(context_id=cid, status="UNRESOLVED_CALIBRATION", fits=[
            dict(name=name, ridge=lam, status="UNRESOLVED") for name, lam in
            [("projection", None)]+[(f"local_ridge_{v:g}", v) for v in study.config["ridge_menu"]]])
    else:
        A, b = _matrix(rows, "branches")
        eta = rows[0]["features"].reshape(-1)
        fits = []
        specs = [("projection", None)] + [(f"local_ridge_{lam:g}", lam) for lam in study.config["ridge_menu"]]
        for name, lam in specs:
            study.check()
            try:
                fit = (fit_local_projection(A, b) if lam is None else fit_local_ridge(A, b,
                    global_ridge=lam, context_weight=study.config["local_context_weight"], features=eta))
                fits.append(dict(name=name, ridge=lam, coefficients=fit["coefficients"].tolist(),
                                 diagnostics=fit["diagnostics"], status="SOLVED"))
            except (ValueError, ArithmeticError, RuntimeError) as error:
                fits.append(dict(name=name, ridge=lam, status="FAILED", error=str(error)))
        payload = dict(context_id=cid, status="SEALED", fits=fits,
                       truth_scope="calibration sources only; coefficients never refitted on withheld targets",
                       full_operator_representation_floor=False)
    payload["offline_fitting_seconds"] = time.perf_counter()-started
    _json(study.output / "fits" / f"{cid}.json", payload)
    legacy._journal(study.output, "local_coefficients_sealed", context_id=cid,
                    fit_sha256=legacy._digest(payload))
    return payload


def _correction(row, prediction, config):
    ref = row["reference"]
    result = {}
    for name, base, uncertainty, accepted in (
        ("U_minus_P", row["base"], ref.get("uncertainty_rms"), ref["accepted"]),
        ("U_minus_G", row.get("matched_mean"),
         (ref.get("uncertainty_rms") or 0)+(ref.get("matched_mean", {}).get("uncertainty_rms") or 0),
         ref["accepted"] and ref.get("matched_mean", {}).get("accepted", False))):
        amplitude = None if base is None else _metrics(row["teacher"]-base)["rms"]
        status = ("UNRESOLVED_REFERENCE" if not accepted or amplitude is None or uncertainty is None else
                  "UNRESOLVED_NEAR_NULL" if amplitude <= config["reference_resolution_multiple"]*uncertainty else "RESOLVED")
        result.update({name+"_rms": amplitude, name+"_uncertainty_rms": uncertainty,
                       name+"_status": status, name+"_relative_error":
                       _metrics(prediction-row["teacher"])["rms"]/amplitude if status == "RESOLVED" else None})
    return result


def _score(study, cid, rows, local):
    from .r7_actions import fresh_action
    ctx = study.contexts[cid]
    p = ctx["parent"]
    methods = [f"ordered_{v}" for v in study.config["ordered_panels"]]
    methods += [f"lawson_{v}" for v in study.config["lawson_steps"]] + ["fixed_feature_fitted"]
    local_fits = local["fits"]
    for row in rows:
        identity = {k: row[k] for k in ("query_id", "context_id", "parent_id", "split", "source_id", "role", "family", "start", "stop")}
        predictions = {}
        try:
            for method in methods + [r["name"] for r in local_fits]:
                study.check()
                work, error, pred = {}, None, None
                try:
                    if method in methods:
                        pred = _numpy(fresh_action(method, p["u"], torch.from_numpy(row["source"]),
                            ctx["start"], ctx["stop"], p["eq"], p["geom"], model=study.model, work=work))
                    else:
                        fit = next(r for r in local_fits if r["name"] == method)
                        if fit["status"] == "SOLVED":
                            pred = row["base"] + np.einsum("j,j...->...", fit["coefficients"], row["branches"])
                    if pred is not None and not np.isfinite(pred).all():
                        raise ArithmeticError("Nonfinite action prediction")
                except (ArithmeticError, ValueError, RuntimeError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                study.work.append(dict(query_id=row["query_id"], method=method, work=work, error=error))
                valid = row["status"] == "ACCEPTED" and pred is not None and error is None
                metric = dict(**identity, method=method, reference_accepted=row["status"] == "ACCEPTED",
                    reference_uncertainty_rms=row["reference"].get("uncertainty_rms"),
                    reference_uncertainty_maximum=row["reference"].get("uncertainty_maximum"),
                    status="FAILED" if error else "MEASURED" if valid else "UNRESOLVED",
                    deployable=method in methods, speedup_claim=False,
                    **(_metrics(pred-row["teacher"]) if valid else
                       dict(rms=None, maximum=None, mean_error=None, centered_rms=None)))
                if error:
                    metric["error"] = error
                if pred is not None and error is None:
                    predictions[method] = pred
                if valid:
                    metric.update(_correction(row, pred, study.config))
                    metric["K_minus_P_rms"] = _metrics(pred-row["base"])["rms"]
                study.metrics.append(metric)
        finally:
            # Partial method work survives a deadline or subsequent method failure.
            np.savez_compressed(study.output / "queries" / (row["query_id"]+"-predictions.npz"), **predictions)
            _json(study.output / "metrics.json", study.metrics)
            _json(study.output / "work.json", study.work)
            study.persist()


def _sensitivity(study, cid, cal, withheld, fits):
    from .r7_fitting import ridge_sensitivity
    if any(r["status"] != "ACCEPTED" for r in cal):
        return
    A, _ = _matrix(cal, "branches")
    # Only branch values from withheld sources enter; no withheld targets.
    At = np.concatenate([r["branches"].reshape(5, -1).T for r in withheld])
    eta = cal[0]["features"].reshape(-1)
    unc = np.concatenate([np.full(r["teacher"].size, r["reference"]["uncertainty_maximum"]) for r in cal])
    diff = np.concatenate([(r["teacher"]-r["refined"]).reshape(-1) for r in cal])
    for fit in fits["fits"]:
        if fit["status"] != "SOLVED" or fit["ridge"] is None:
            continue
        lam = fit["ridge"]/(study.config["local_context_weight"]*float(eta@eta))
        result = ridge_sensitivity(A, At, ridge=lam, reference_uncertainty=unc,
            independent_reference_difference=diff, seed=study.config["sensitivity"]["seed"],
            random_trials=study.config["sensitivity"]["random_trials"])
        _save_sensitivity(study, f"{cid}-{fit['name']}", result)


def _save_sensitivity(study, name, result):
    _json(study.output / "sensitivity" / (name+".json"), result["diagnostics"])
    arrays = result.get("arrays", {key: value for key, value in result.items() if isinstance(value, np.ndarray)})
    np.savez_compressed(study.output / "sensitivity" / (name+".npz"), **arrays)


def _global_sensitivity(study, selection):
    from .r7_fitting import ridge_sensitivity
    selected = selection["selected"]
    if selected["ridge"] is None:
        _json(study.output / "sensitivity" / "global.json", dict(status="INITIALIZATION_SELECTED", coefficients_depend_on_labels=False))
        return
    train = study.rows(split="train", role="calibration")
    A, _ = _matrix(train)
    withheld = [r for r in study.rows(role="withheld") if "design" in r]
    At = np.concatenate([r["design"].reshape(40, -1).T for r in withheld])
    unc = np.concatenate([np.full(r["teacher"].size, r["reference"]["uncertainty_maximum"]) for r in train])
    diff = np.concatenate([(r["teacher"]-r["refined"]).reshape(-1) for r in train])
    # Identical grid/source/context counts make these uniform normalized weights
    # exactly the parent-balanced training weights of the selected global solve.
    result = ridge_sensitivity(A, At, ridge=selected["ridge"], reference_uncertainty=unc,
        independent_reference_difference=diff, seed=study.config["sensitivity"]["seed"],
        random_trials=study.config["sensitivity"]["random_trials"])
    result["diagnostics"]["selection_held_fixed"] = True
    _save_sensitivity(study, "global", result)


def _work_precision(study):
    """Join identical-context accuracy and complete bundle times without a gate."""
    rows = []
    for timed in study.timing:
        for method, observations in timed["summary"].items():
            matching = [row for row in study.metrics if row["context_id"] == timed["context_id"]
                        and row["role"] == "withheld" and row["method"] == method]
            resolved = (len(matching) == study.config["withheld_sources"]
                        and all(row["status"] == "MEASURED" for row in matching))
            rows.append(dict(context_id=timed["context_id"], method=method,
                expected_sources=study.config["withheld_sources"], measured_sources=len(matching),
                all_references_and_predictions_resolved=resolved,
                pooled_source_rms=(math.sqrt(sum(r["rms"]**2 for r in matching)/len(matching)) if resolved else None),
                worst_source_rms=max(r["rms"] for r in matching) if resolved else None,
                maximum=max(r["maximum"] for r in matching) if resolved else None,
                **observations, speedup_claim=False))
    _json(study.output / "work_precision.json", rows)
    with (study.output / "work_precision.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader()
        writer.writerows(rows)


def _study(output, config):
    from .r7_actions import fresh_actions, measure_paired_actions
    legacy._journal(output, "protocol_frozen", protocol_sha256=legacy._digest(config))
    study = Study(output, config)
    try:
        for cid, ctx in study.contexts.items():
            split = ctx["parent"]["split"]
            if split == "train":
                study.acquire(cid, "calibration")
            elif split == "validation":
                study.acquire(cid, "calibration")
                study.acquire(cid, "withheld")
        selection = _global_fit(study)
        for cid, ctx in study.contexts.items():
            study.check()
            cal = study.acquire(cid, "calibration")
            local = _local_fit(study, cid, cal)
            withheld = study.acquire(cid, "withheld")
            _score(study, cid, cal+withheld, local)
            _sensitivity(study, cid, cal, withheld, local)
            if ctx["parent"]["split"] == "diagnostic":
                p = ctx["parent"]
                methods = fresh_actions(p["u"], [torch.from_numpy(r["source"]) for r in withheld],
                    ctx["start"], ctx["stop"], p["eq"], p["geom"], model=study.model,
                    lawson_steps=tuple(config["lawson_steps"]))
                measured = measure_paired_actions(methods, rounds=config["timing"]["rounds"],
                    seed=config["timing"]["seed"]+ctx["index"], warmup=config["timing"]["warmup"], budget=study.check,
                    on_record=lambda row: study.timing_record(cid, row))
                study.timing.append(dict(context_id=cid, **measured))
                _json(output / "timing.json", study.timing)
            legacy._journal(output, "context_complete", context_id=cid)
        _global_sensitivity(study, selection)
        _json(output / "decisions.json", dict(status="DESCRIPTIVE_DEVELOPMENT_ONLY",
            full_operator_capacity="UNRESOLVED: finite calibration and withheld directions do not exhaust the operator",
            encoder_promotion=False, superiority=False, matched_tolerance_speedup=False,
            next_gate="review projection/ridge/global source-family residuals and perturbations against reference uncertainty; freeze any subsequent study separately",
            endpoint_benefit="UNRESOLVED_FROM_NATIVE_2936; no R7 endpoint labels acquired",
            limitations=config["deferred"]))
        return dict(reference_count=study.attempted, metrics_count=len(study.metrics),
            failed_metrics=sum(r["status"] == "FAILED" for r in study.metrics),
            failed_timing_bundles=sum(r["status"] != "PASS" for r in study.timing_records),
            unresolved_references=sum(r["status"] != "ACCEPTED" for r in study.index),
            unresolved_matched_mean_references=sum(not r["reference"].get("matched_mean", {}).get("accepted", False)
                                                 for r in study.index if r["reference"]),
            selected_candidate=selection["selected"]["name"], selected_ridge=selection["selected"]["ridge"],
            native_gpu_readiness=False, matched_accuracy_speedup_established=False)
    finally:
        study.persist()
        _work_precision(study)
        _json(output / "metrics.json", study.metrics)
        _json(output / "timing.json", study.timing)
        _json(output / "work.json", study.work)
        fields = sorted({key for row in study.metrics for key in row})
        with (output / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(study.metrics)


def _verify(output, config):
    tests = config["verify_tests"]
    common = [sys.executable, "-m", "pytest", "-q", *tests]
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    collected = subprocess.run(common+["--collect-only"], cwd=ROOT, env=environment, text=True, capture_output=True)
    (output / "collection.log").write_text(collected.stdout+collected.stderr)
    nodes = [line for line in collected.stdout.splitlines() if line.startswith("tests/") and "::" in line]
    if collected.returncode or not nodes or len(nodes) != len(set(nodes)):
        raise RuntimeError("R7 required case collection failed or duplicated cases")
    expected = []
    for node in nodes:
        parts = node.split("::")
        expected.append((parts[0].removesuffix(".py").replace("/", ".")+"".join("."+s for s in parts[1:-1]), parts[-1]))
    _json(output / "expected_cases.json", nodes)
    command = common+["--basetemp="+str(output/"pytest-temp"), "--junitxml="+str(output/"verification.xml")]
    _json(output / "command.json", dict(argv=command, cwd=str(ROOT)))
    with (output / "verification.log").open("w") as handle:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
    cases = list(ET.parse(output / "verification.xml").getroot().iter("testcase"))
    actual = [(r.get("classname"), r.get("name")) for r in cases]
    checks = dict(total=len(cases), expected=len(expected), exact_roster=sorted(actual) == sorted(expected),
        duplicates=len(actual)-len(set(actual)), failed=sum(r.find("failure") is not None for r in cases),
        errors=sum(r.find("error") is not None for r in cases), skipped=sum(r.find("skipped") is not None for r in cases))
    _json(output / "checks.json", checks)
    if result.returncode or not checks["exact_roster"] or any(checks[k] for k in ("duplicates", "failed", "errors", "skipped")):
        raise RuntimeError("R7 CPU verification did not pass its complete exact case roster")
    return dict(checks=checks)


def execute(command, output=None, *, device="cpu"):
    if device != "cpu":
        raise ValueError("R7 requires explicit CPU execution")
    config = protocol(command)
    if command == "study":
        # The CLI additionally verifies Raven capture, source contract and the
        # same attempt's verifier release before reaching this internal guard.
        from tdn.runtime.preflight import verify_runtime
        from tdn.runtime.desktop_slurm import verify_allocation
        verify_runtime("cpu", "response-preconditioner-r7")
        verify_allocation("cpu", root=ROOT)
    output = legacy.create_output("verify", output)
    before = legacy.source_snapshot()
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    started = time.perf_counter()
    summary = dict(schema=SCHEMA, command=command, status="RUNNING", decision="BOUNDED_DEVELOPMENT",
        device="cpu", scientific_promotion=False, native_gpu_readiness=False, output=str(output),
        source_fingerprint=before["fingerprint"], protocol_sha256=legacy._digest(config), historical_gate="B STOP unchanged")
    for name, data in (("protocol", config), ("source", before), ("software", legacy._software()), ("summary", summary)):
        _json(output / (name+".json"), data)
    try:
        summary.update((_verify if command == "verify" else _study)(output, config))
        after = legacy.source_snapshot()
        _json(output / "source_after.json", after)
        if before != after:
            raise RuntimeError("Source changed during R7 execution; cannot seal")
        summary["status"] = "COMPLETED"
    except Exception as error:
        summary.update(status="FAILED", error_type=type(error).__name__, error=str(error))
        (output / "failure.txt").write_text(traceback.format_exc())
    finally:
        summary["elapsed_seconds"] = time.perf_counter()-started
        _json(output / "summary.json", summary)
        legacy._journal(output, "run_finished", status=summary["status"])
        (output / "README.md").write_text(
            "# R7 bounded development evidence\n\n"
            f"Execution: **{summary['status']}**. No scientific promotion.\n\n"
            "Read protocol.json before interpreting metrics. query_index.json and denominators.json retain every planned query, including missing and unresolved cases. "
            "The source bank reuses four inspected parents. Selection uses validation withheld sources; diagnostic local oracles use calibration labels only. "
            "Projection is a sampled-source numerical least-squares result, not a full-operator capacity certificate. Ridge sensitivity uses empirical uncertainty scenarios. "
            "NPZ files contain numeric arrays only; load with allow_pickle=False. Relative artifact paths support export. "
            "Timing includes fresh setup per source; raw paired rounds are retained. Native sampled RSS is worker-level, not method-specific peak memory. "
            "No new endpoint, GPU, rollout, novelty or matched-tolerance superiority claim is supported by completion. "
            "See docs/RESPONSE_PRECONDITIONER_R7_DESIGN.md and docs/RESPONSE_PRECONDITIONER_RUNBOOK.md for definitions and gates.\n")
        files = [dict(path=str(p.relative_to(output)), bytes=p.stat().st_size,
                      sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                 for p in sorted(output.rglob("*")) if p.is_file() and "pytest-temp" not in p.parts]
        _json(output / "artifacts.json", dict(schema=SCHEMA, source_fingerprint=before["fingerprint"],
                                             status=summary["status"], self_excluded="artifacts.json", files=files))
        torch.set_num_threads(old_threads)
    return summary
