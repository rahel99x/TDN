"""Bounded retained-array regularization audit; production execution is native only."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from . import workflow as legacy
from . import r7_workflow as r7
from .regularization_protocol import (SCHEMA, SELECTION_SCHEMA, CONTROLS, protocol, validate_config)
from .regularization_fitting import fit_parent_balanced_tikhonov, fit_parent_balanced_projection

ROOT = legacy.ROOT


def _json(path, value):
    legacy._json(Path(path), value)


def _cell_id(n, k):
    return f"p{n:03d}-s{k:02d}"


def _stack(rows):
    matrices, targets, ids, weights, differences = [], [], [], [], []
    uncertainty = []
    for row in rows:
        a = row["arrays"]
        matrix = np.asarray(a["design"], dtype=np.float64).reshape(40, -1).T
        target = (a["teacher"]-a["base"]).reshape(-1)
        if matrix.shape[0] != target.size or not target.size:
            raise ValueError("Retained design/target dimensions disagree")
        matrices.append(matrix); targets.append(target)
        ids.extend([row["metadata"]["parent_id"]]*target.size)
        weights.extend([1./target.size]*target.size)
        differences.append((a["refined"]-a["teacher"]).reshape(-1))
        uncertainty.extend([row["metadata"]["reference"]["uncertainty_rms"]]*target.size)
    A, b = np.concatenate(matrices), np.concatenate(targets)
    q = np.asarray(weights)
    parent_weights = {pid: float(q[np.asarray(ids) == pid].sum()) for pid in set(ids)}
    omega = np.asarray([v/(len(parent_weights)*parent_weights[pid]) for pid, v in zip(ids, q)])
    if not np.isfinite(A).all() or not np.isfinite(b).all() or not np.isfinite(omega).all():
        raise ValueError("Nonfinite retained training system")
    return dict(A=A, b=b, parent_ids=ids, row_weights=q, omega=omega,
                Aw=A*np.sqrt(omega[:, None]), bw=b*np.sqrt(omega),
                difference=np.concatenate(differences)*np.sqrt(omega),
                uncertainty_radius=float(np.sqrt(np.dot(omega, np.square(uncertainty)))))


def _prediction(arrays, coefficients, parity_factor=128.):
    w = np.asarray(coefficients, dtype=np.float64)
    if w.shape != (40,) or not np.isfinite(w).all():
        raise ValueError("A readout must have forty finite coefficients")
    design = arrays["design"]
    prediction = arrays["base"] + np.einsum("j,j...->...", w, design)
    effective = w.reshape(5, 8) @ arrays["features"]
    streamed = arrays["base"] + np.einsum("j,j...->...", effective, arrays["branches"])
    scale = max(1., float(np.max(np.abs(arrays["base"]) +
                    np.einsum("j,j...->...", np.abs(w), np.abs(design)))))
    discrepancy = float(np.max(np.abs(prediction-streamed)))
    tolerance = parity_factor*np.finfo(np.float64).eps*scale
    if (not np.isfinite(prediction).all() or not np.isfinite(streamed).all()
            or not np.isfinite([scale,discrepancy,tolerance]).all() or discrepancy > tolerance):
        raise ArithmeticError("Retained affine and five-branch predictions disagree or are nonfinite")
    return prediction, discrepancy, tolerance


class RegularizationStudy:
    """Selection reads train/validation only; adapter also enforces the file seal."""

    def __init__(self, output, config, evidence, *, allow_fixture=False, plots=True):
        validate_config(config, allow_fixture=allow_fixture)
        self.output, self.config, self.evidence = Path(output), config, evidence
        self.plots = plots
        self.started = time.perf_counter()
        self.deadline = self.started + config["budget"]["study_seconds"]
        self.index = evidence.queries
        self.parents = {p["parent_id"]: p for p in evidence.parents}
        self.train_ids = sorted(pid for pid, p in self.parents.items() if p["split"] == "train")
        self.specs = {_cell_id(c["training_parent_count"], c["calibration_source_count"]):
                      (c["training_parent_count"], c["calibration_source_count"]) for c in config["cells"]}
        self.methods = {}
        for cid, (n, k) in self.specs.items():
            for kind in ("coefficient_ridge", "action_energy", "original", "projection"):
                self.methods[f"{kind}-{cid}"] = dict(cell_id=cid, kind=kind,
                    training_parent_count=n, calibration_source_count=k)
        self.methods.update({name: dict(kind="classical", cell_id=None,
            training_parent_count=None, calibration_source_count=None) for name in CONTROLS})
        withheld = [q for q in self.index if q["role"] == "withheld"]
        if (len(self.index) != config["expected_input_queries"]
                or len(withheld) != config["expected_withheld_queries"]
                or len(self.methods) != config["expected_methods"]
                or len({q["query_id"] for q in self.index}) != len(self.index)):
            raise ValueError("Input/work roster disagrees with frozen regularization protocol")
        self.metrics = {(q["query_id"], m): self.pending_metric(q, m) for q in withheld for m in self.methods}
        self.cells, self.readouts, self.rows, self.sensitivities = [], {}, {}, []
        self.selection = None
        self.costs = dict(scope="New audit computation only; original label/setup costs are reused, not reacquired",
            component_scope="fitting_seconds is inclusive candidate computation (solve, validation and sensitivity); sensitivity_seconds is its subset. Penalty setup, input reads and evaluation are separate. Components are not additive to wall time.",
            fresh_reference_solves=0, new_timing_bundles=0, fitting_seconds=0.,
            sensitivity_seconds=0., penalty_setup_seconds=0., evaluation_seconds=0., input_read_seconds=0.,
            original_reference_seconds=sum(q.get("reference_seconds", 0.) for q in self.index),
            original_design_seconds=sum(q.get("design_seconds", 0.) for q in self.index),
            reused_reference_pairs=len(self.index), cells=[])
        for name in ("fits", "predictions"):
            (self.output/name).mkdir()
        _json(self.output/"query_index.json", [{k:v for k,v in q.items() if k != "reference"} for q in self.index])
        _json(self.output/"parents.json", list(self.parents.values()))
        _json(self.output/"input_identity.json", evidence.identity)
        _json(self.output/"reference_summary.json", dict(count=len(self.index),
            accepted=sum(q["status"] == "ACCEPTED" for q in self.index),
            source="Original sealed P1 references reused without new solves", certificate=False))
        if any(q["status"] != "ACCEPTED" or not q["reference"].get("accepted") for q in self.index):
            raise ValueError("An unresolved original query cannot silently enter this audit")
        self.persist()

    def check(self):
        if time.perf_counter() >= self.deadline:
            raise TimeoutError("Frozen regularization audit budget exhausted; preserve partial work")

    def pending_metric(self, query, method):
        metadata = self.methods[method]
        n = metadata["training_parent_count"]
        membership = ("not_applicable" if n is None else "not_training_split" if query["split"] != "train"
                      else "in_cohort" if query["parent_id"] in self.train_ids[:n] else "outside_cohort")
        return dict(**{k:query[k] for k in ("query_id", "context_id", "parent_id", "parent_family",
            "split", "role", "family", "start", "stop")}, method=method, **metadata,
            training_membership=membership, status="NOT_EVALUATED", rms=None, mse=None,
            maximum=None, mean_error=None, spatial_centered_rms=None)

    def read(self, q):
        self.check()
        qid = q["query_id"]
        if qid not in self.rows:
            self.rows[qid] = self.input_call(self.evidence.read_query, qid)
            legacy._journal(self.output, "retained_query_read", query_id=qid, split=q["split"], role=q["role"])
        return self.rows[qid]

    def input_call(self, callback, *args):
        started = time.perf_counter()
        try:
            return callback(*args)
        finally:
            self.costs["input_read_seconds"] += time.perf_counter()-started

    def fit(self):
        from .regularization_sensitivity import sensitivity
        train = [q for q in self.index if q["split"] == "train" and q["role"] == "calibration"]
        valid = [q for q in self.index if q["split"] == "validation" and q["role"] == "withheld"]
        for q in train+valid:
            self.read(q)
        validation_rows = [self.rows[q["query_id"]] for q in valid]
        v = _stack(validation_rows)
        original = {c["model_id"]:c for c in self.input_call(self.evidence.read_original_selection)["cells"]}
        for cid, (n,k) in self.specs.items():
            self.check()
            tids = self.train_ids[:n]
            queries = [q for q in train if q["parent_id"] in tids and q["source_index"] < k]
            if len(tids) != n or any(sum(q["parent_id"] == pid for q in queries) == 0 for pid in tids):
                raise ValueError("Missing declared training parent/source prefix")
            rows = [self.rows[q["query_id"]] for q in queries]
            setup_start = time.perf_counter()
            try:
                t = _stack(rows)
                _, singular, vh = np.linalg.svd(t["Aw"], full_matrices=False)
                penalty = singular[:, None]*vh
            finally:
                self.costs["penalty_setup_seconds"] += time.perf_counter()-setup_start
            original_cell = original["fit-"+cid]
            if ([q["query_id"] for q in queries] != original_cell["train_query_ids"]
                    or [q["query_id"] for q in valid] != original_cell["validation_query_ids"]):
                raise ValueError("Audit prefixes differ from original sealed training/validation query lists")
            cell = dict(cell_id=cid, training_parent_count=n, calibration_source_count=k,
                training_parent_ids=tids, train_query_ids=[q["query_id"] for q in queries],
                validation_query_ids=[q["query_id"] for q in valid],
                original_coefficients=original_cell["selected"]["coefficients"],
                original_selection_seal=original_cell["selection_seal"], candidates=[], selected={},
                diagnostic_truth_accessed=False, training_withheld_truth_accessed=False,
                previously_inspected_development=True)
            self.cells.append(cell)
            candidate_specs = [("zero", None)] + [("coefficient_ridge", x) for x in self.config["coefficient_ridges"]]
            candidate_specs += [("action_energy", x) for x in self.config["action_energies"]] + [("projection", None)]
            projection_coefficients = None
            for family, strength in candidate_specs:
                self.check()
                name = "zero_readout" if family == "zero" else "projection" if family == "projection" else f"{family}_{strength:g}"
                c = dict(name=name, family=family, ridge=strength, eligible=False, status="STARTED",
                         validation_mse=None, coefficients=None, reasons=[], diagnostics={})
                cell["candidates"].append(c)
                began = time.perf_counter()
                try:
                    if family == "zero":
                        w = np.zeros(40)
                        c["diagnostics"] = dict(data_loss=float(np.dot(t["bw"],t["bw"])),
                            penalty_loss=0., objective=float(np.dot(t["bw"],t["bw"])), eligible=True)
                    else:
                        fit = (fit_parent_balanced_projection(t["A"],t["b"],t["parent_ids"],row_weights=t["row_weights"])
                            if family == "projection" else fit_parent_balanced_tikhonov(t["A"],t["b"],t["parent_ids"],
                                ridge=strength, penalty=penalty if family == "action_energy" else None,
                                row_weights=t["row_weights"], optimality_rtol=self.config["numerical_eligibility"]["optimality_rtol"]))
                        w = fit["coefficients"].numpy(); c["diagnostics"] = fit["diagnostics"]
                    c["coefficients"] = w.tolist()
                    c["eligible"] = family != "projection" and c["diagnostics"]["eligible"]
                    c["reasons"] = list(c["diagnostics"].get("reasons", []))
                    if family == "action_energy" and c["diagnostics"]["original_design_numerical_rank"] != 40:
                        c["eligible"] = False
                        c["reasons"].append("Action-energy regularization requires full original design rank")
                    # Selection uses the same physical prediction arithmetic as final scoring.
                    val_mse = []
                    max_parity = 0.
                    for row in validation_rows:
                        prediction, delta, _ = _prediction(row["arrays"], w, self.config["numerical_eligibility"]["parity_factor"])
                        loss = float(np.mean((prediction-row["arrays"]["teacher"])**2))
                        if not np.isfinite(loss):
                            raise ArithmeticError("Nonfinite validation physical MSE")
                        val_mse.append((row["metadata"]["parent_id"], loss))
                        max_parity = max(max_parity, delta)
                    parent_loss = {pid:np.mean([e for p,e in val_mse if p==pid]) for pid,_ in val_mse}
                    c["validation_mse"] = float(np.mean(list(parent_loss.values())))
                    if not np.isfinite(c["validation_mse"]):
                        c["validation_mse"] = None
                        raise ArithmeticError("Nonfinite parent-balanced validation MSE")
                    c["validation_parity_maximum"] = max_parity
                    c["status"] = "SOLVED" if family != "projection" else "OFFLINE_DIAGNOSTIC"
                    if family == "projection":
                        projection_coefficients = w
                    if family != "zero":
                        sensitivity_start = time.perf_counter()
                        try:
                            record = sensitivity(t["Aw"],t["bw"],v["Aw"],t["difference"],t["uncertainty_radius"],
                                np.sqrt(c["validation_mse"]),np.linalg.norm(v["bw"]),v["uncertainty_radius"],
                                family=family,ridge=0. if family == "projection" else strength)
                            c["sensitivity"] = record
                            self.sensitivities.append(dict(cell_id=cid,candidate=name,**record))
                        except (ValueError,ArithmeticError,np.linalg.LinAlgError) as error:
                            c["sensitivity"] = dict(status="UNAVAILABLE",reason=str(error),scientific_eligibility_gate=False,error_certificate=False)
                            self.sensitivities.append(dict(cell_id=cid,candidate=name,**c["sensitivity"]))
                        finally:
                            self.costs["sensitivity_seconds"] += time.perf_counter()-sensitivity_start
                except (ValueError,ArithmeticError,RuntimeError,np.linalg.LinAlgError) as error:
                    c.update(status="FAILED",eligible=False,reasons=[f"{type(error).__name__}: {error}"])
                finally:
                    c["elapsed_seconds"] = time.perf_counter()-began
                    self.costs["fitting_seconds"] += c["elapsed_seconds"]
                    _json(self.output/"fits"/(cid+".json"), cell)
                    legacy._journal(self.output,"candidate_finished",cell_id=cid,candidate=name,status=c["status"],eligible=c["eligible"])
            original_fit = next(c for c in cell["candidates"] if c["family"] == "coefficient_ridge" and c["ridge"] == 1e-10)
            reproduction = dict(status="UNAVAILABLE",reason="Original-strength candidate did not solve")
            if original_fit["status"] == "SOLVED":
                old_w = np.asarray(cell["original_coefficients"])
                new_w = np.asarray(original_fit["coefficients"])
                difference = float(np.linalg.norm(v["Aw"] @ (new_w-old_w)))
                scale = max(1.,float(np.linalg.norm(np.abs(v["Aw"]) @ (np.abs(new_w)+np.abs(old_w)))))
                tolerance = self.config["numerical_eligibility"]["parity_factor"]*np.finfo(np.float64).eps*scale
                if not np.isfinite([difference,scale,tolerance]).all() or difference > tolerance:
                    raise ArithmeticError("Original-strength fit fails retained validation-prediction reproduction")
                reproduction = dict(status="PASS",validation_correction_difference_rms=difference,tolerance=tolerance,
                    scope="Arithmetic reproduction of original positive-ridge objective, not a new scientific accuracy gate")
            cell["original_strength_reproduction"] = reproduction
            for family in ("coefficient_ridge","action_energy"):
                eligible = [c for c in cell["candidates"] if c["family"] in ("zero",family) and c["eligible"]]
                if not eligible:
                    raise ArithmeticError("No numerically eligible candidate including zero")
                selected = min(eligible,key=lambda c:c["validation_mse"])
                cell["selected"][family] = selected
                self.readouts[family+"-"+cid] = np.asarray(selected["coefficients"])
            self.readouts["original-"+cid] = np.asarray(cell["original_coefficients"])
            if projection_coefficients is not None:
                self.readouts["projection-"+cid] = projection_coefficients
                equivalence = []
                for c in cell["candidates"]:
                    if c["family"] == "action_energy" and c["status"] == "SOLVED" and c["coefficients"] is not None:
                        prediction_difference = t["Aw"] @ (np.asarray(c["coefficients"])-projection_coefficients/(1+c["ridge"]))
                        difference = float(np.linalg.norm(prediction_difference))
                        equivalence.append(dict(candidate=c["name"],status="MEASURED" if np.isfinite(difference) else "UNAVAILABLE",
                            weighted_prediction_difference=difference if np.isfinite(difference) else None))
                cell["action_energy_projection_equivalence"] = equivalence
            cell["selection_seal"] = legacy._digest(cell)
            _json(self.output/"fits"/(cid+".json"), cell)
            self.costs["cells"].append(dict(cell_id=cid,reused_training_query_count=len(queries),
                original_training_reference_seconds=sum(q.get("reference_seconds",0.) for q in queries),
                audit_candidate_seconds=sum(c["elapsed_seconds"] for c in cell["candidates"])))
            self.persist()
        self.selection = dict(schema=SELECTION_SCHEMA,cells=self.cells,all_selections_sealed=True,
            criterion=self.config["selection"],diagnostic_truth_accessed=False,training_withheld_truth_accessed=False,
            previously_inspected_development=True,scientific_promotion=False)
        path = self.output/"selection.json"
        _json(path,self.selection)
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        legacy._journal(self.output,"all_selections_frozen",selection_file_sha256=checksum,
                        selected_models=self.config["expected_selected_models"])
        self.evidence.release_protected(path,checksum)
        self.persist()

    def evaluate(self):
        from .regularization_analysis import metric
        if self.selection is None:
            raise RuntimeError("All selections must be sealed before withheld evaluation")
        for q in self.index:
            if q["role"] != "withheld":
                continue
            self.check()
            row = self.read(q); a=row["arrays"]
            original = self.input_call(self.evidence.read_predictions, q["query_id"])
            predictions = {}
            try:
                for method,spec in self.methods.items():
                    self.check()
                    began=time.perf_counter()
                    pending=self.metrics[(q["query_id"],method)]
                    try:
                        delta,tolerance=None,None
                        if spec["kind"] == "classical":
                            prediction=original[method]
                        elif spec["kind"] == "original":
                            prediction=original["fit-"+spec["cell_id"]]
                            reconstructed,delta,tolerance=_prediction(a,self.readouts[method])
                            if float(np.max(np.abs(prediction-reconstructed))) > tolerance:
                                raise ArithmeticError("Original saved response differs from retained design replay")
                        elif method in self.readouts:
                            prediction,delta,tolerance=_prediction(a,self.readouts[method])
                        else:
                            raise ArithmeticError("Projection diagnostic fit was unavailable; retain failed slot")
                        reference=q["reference"]
                        values=metric(q,method,prediction,a["teacher"],a["base"],a["matched_mean"],
                            reference["uncertainty_rms"],reference["uncertainty_maximum"],
                            **spec,training_membership=pending["training_membership"],
                            matched_mean_uncertainty_rms=reference["matched_mean"]["uncertainty_rms"],
                            matched_mean_uncertainty_maximum=reference["matched_mean"]["uncertainty_maximum"])
                        values.update(prediction_mode="historical_native_prediction" if spec["kind"] in ("classical","original") else "retained_design_five_branch_replay",
                            inference_parity_maximum=delta,inference_parity_tolerance=tolerance,
                            scientific_promotion=False)
                        self.metrics[(q["query_id"],method)]=values
                        predictions[method]=prediction
                    except (ValueError,ArithmeticError,RuntimeError) as error:
                        pending.update(status="FAILED",reason=f"{type(error).__name__}: {error}")
                    finally:
                        self.costs["evaluation_seconds"] += time.perf_counter()-began
            finally:
                path=self.output/"predictions"/(q["query_id"]+".npz")
                np.savez_compressed(path,**predictions)
                _json(path.with_suffix(".json"),[self.metrics[(q["query_id"],m)] for m in self.methods])
                legacy._journal(self.output,"query_evaluated",query_id=q["query_id"],prediction_path=path.relative_to(self.output).as_posix(),
                                prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        self.persist()

    def denominators(self):
        counts=Counter(row["status"] for row in self.metrics.values())
        candidates=[c for cell in self.cells for c in cell["candidates"]]
        return dict(expected_metrics=self.config["expected_metrics"],attempted_metrics=len(self.metrics)-counts["NOT_EVALUATED"],
            measured_metrics=counts["MEASURED"],failed_metrics=counts["FAILED"],unresolved_metrics=counts["UNRESOLVED"],missing_metrics=counts["NOT_EVALUATED"],
            expected_fit_cells=len(self.specs),attempted_fit_cells=len(self.cells),expected_candidates=self.config["expected_candidates"],
            attempted_candidates=len(candidates),failed_candidates=sum(c["status"]=="FAILED" for c in candidates),
            ineligible_candidates=sum(not c["eligible"] for c in candidates),
            expected_selected_models=self.config["expected_selected_models"],
            selected_models=sum(len(c["selected"]) for c in self.cells),
            expected_input_queries=len(self.index),read_input_queries=len(self.rows),fresh_reference_solves=0,new_timing_bundles=0)

    def persist(self):
        _json(self.output/"denominators.json",self.denominators())
        _json(self.output/"costs.json",self.costs)
        _json(self.output/"sensitivity.json",self.sensitivities)

    def finalize(self):
        from .regularization_analysis import render
        self.persist()
        metrics=list(self.metrics.values())
        _json(self.output/"metrics.json",metrics)
        selection=self.selection or dict(schema=SELECTION_SCHEMA,cells=self.cells,all_selections_sealed=False)
        if self.selection is None:
            _json(self.output/"selection.json",selection)
        render(self.output,metrics,selection,self.costs,plots=self.plots)
        _json(self.output/"decisions.json",dict(status="DESCRIPTIVE_DEVELOPMENT_ONLY",
            scientific_promotion=False,matched_accuracy_speedup_established=False,
            next_gate="Review regularization paths, sensitivity and unchanged classical controls before freezing new work",
            historical_gate="B STOP unchanged",fresh_confirmation=False))


def _study(output,config,*,expected_identity=None,evidence=None,allow_fixture=False,plots=True):
    validate_config(config,allow_fixture=allow_fixture)
    started = time.perf_counter()
    if evidence is not None and not allow_fixture:
        raise ValueError("Injected evidence is private synthetic implementation coverage only")
    if evidence is None:
        from .regularization_evidence import RetainedEvidence
        evidence=RetainedEvidence(ROOT,Path(output)/"inputs",expected_identity=expected_identity)
    legacy._journal(Path(output),"protocol_frozen",protocol_sha256=legacy._digest(config))
    study=None
    primary_error=None
    try:
        study=RegularizationStudy(output,config,evidence,allow_fixture=allow_fixture,plots=plots)
        study.deadline = started + config["budget"]["study_seconds"]
        study.fit();study.evaluate();study.check()
    except BaseException as error:
        primary_error=error
        raise
    finally:
        cleanup_errors=[]
        callbacks=([("reporting",study.finalize)] if study is not None else []) + [("input_close",evidence.close)]
        for stage,callback in callbacks:
            try:
                callback()
            except Exception as error:
                cleanup_errors.append((stage,error))
        if cleanup_errors:
            details=[dict(stage=stage,error_type=type(error).__name__,error=str(error))
                     for stage,error in cleanup_errors]
            try:
                _json(Path(output)/"finalization_failure.json",details)
            except Exception as error:
                cleanup_errors.append(("failure_record",error))
            failure=primary_error if primary_error is not None else cleanup_errors[0][1]
            # Preserve notes on Python 3.10 too; 3.11+ also renders them in tracebacks.
            failure.__notes__=list(getattr(failure,"__notes__",[]))+[
                f"Additional {stage} failure: {type(error).__name__}: {error}"
                for stage,error in cleanup_errors]
            if primary_error is None:
                raise failure
    study.check()  # Include rendering, inventory and retained-input closing in the total budget.
    return dict(**study.denominators(),native_gpu_readiness=False,matched_accuracy_speedup_established=False)


def execute(command,output=None,*,device="cpu",contract=None,expected_identity=None):
    if device != "cpu":
        raise ValueError("Regularization audit requires explicit CPU execution")
    config=protocol(command);validate_config(config)
    if command == "study":
        if contract is None:
            raise ValueError("Study requires same-attempt native Raven verification and a prepared contract")
        try:
            from response_preconditioner_regularization_native import require_study_context
        except ImportError as error:
            raise ValueError("Use the native regularization entry point") from error
        ctx=require_study_context(contract,output)
        if expected_identity is not None and expected_identity != ctx["input_identity"]:
            raise ValueError("CLI input identity differs from native contract")
        expected_identity=ctx["input_identity"]
    output=legacy.create_output("verify",output)
    before=legacy.source_snapshot();old_threads=torch.get_num_threads();torch.set_num_threads(1)
    started=time.perf_counter()
    summary=dict(schema=SCHEMA,command=command,status="RUNNING",device="cpu",decision=config["decision"],
        source_fingerprint=before["fingerprint"],protocol_sha256=legacy._digest(config),output=str(output),
        scientific_promotion=False,native_gpu_readiness=False,fresh_reference_solves=0,new_timing_bundles=0)
    for name,value in (("protocol",config),("source",before),("software",legacy._software()),("summary",summary)):
        _json(output/(name+".json"),value)
    try:
        summary.update(r7._verify(output,config) if command == "verify" else _study(output,config,expected_identity=expected_identity))
        if command == "study" and (summary["failed_metrics"] or summary["unresolved_metrics"] or summary["missing_metrics"]):
            raise RuntimeError("Incomplete regularization metric roster; preserve failed and missing slots")
        after=legacy.source_snapshot();_json(output/"source_after.json",after)
        if before != after:
            raise RuntimeError("Source changed during regularization audit")
        summary["status"]="COMPLETED"
    except Exception as error:
        summary.update(status="FAILED",error_type=type(error).__name__,error=str(error))
        (output/"failure.txt").write_text(traceback.format_exc())
    finally:
        summary["elapsed_seconds"]=time.perf_counter()-started
        _json(output/"summary.json",summary)
        legacy._journal(output,"run_finished",status=summary["status"])
        (output/"README.md").write_text("# Retained-data regularization audit\n\n"
            f"Execution: **{summary['status']}**. This is inspected development evidence.\n\n"
            "Read protocol.json, selection.json, denominators.json, sensitivity.json and learning_curves.csv. "
            "All six validation selections precede this audit's protected array access. Original labels and "
            "classical predictions are reused from native P1; no new references or inference timing are measured. "
            "Projection is diagnostic only. Action energy is a uniform-shrinkage control, not a derived physical law. "
            "Mean-component and spatial-centered MSE are aggregated per action then per parent. "
            "Missing/failed rows remain in denominators. Retain inputs/ and predictions/ for numeric replay. "
            "Sampled worker CPU/RSS is separate from science and supervisor telemetry. "
            "No automatic promotion, GPU readiness, fresh confirmation or speedup follows.\n")
        files=[]
        for path in sorted(output.rglob("*")):
            if path.is_file() and "pytest-temp" not in path.parts and not any(p.startswith(".") for p in path.relative_to(output).parts):
                files.append(dict(path=path.relative_to(output).as_posix(),bytes=path.stat().st_size,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        _json(output/"artifacts.json",dict(schema=SCHEMA,source_fingerprint=before["fingerprint"],status=summary["status"],
            self_excluded="artifacts.json",files=files))
        torch.set_num_threads(old_threads)
    return summary
