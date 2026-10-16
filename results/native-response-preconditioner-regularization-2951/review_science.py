#!/usr/bin/env python3
"""Read-only independent replay of retained P1R fits; no fitting or teachers.

Only NumPy and the Python standard library are imported. Uploaded code is never
imported or executed. Saved coefficients are evaluated as supplied. SVDs inspect
design/penalty spectra and propagate retained label differences; no coefficient
solve, minimizer construction, new reference or scientific candidate occurs.

Usage (one numerical thread is recommended):
  OPENBLAS_NUM_THREADS=1 .venv/bin/python review_science.py \
    --study PATH/outputs/study --output PATH/science-review.json
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


class Review:
    def __init__(self, study):
        self.root = Path(study).resolve(strict=True)
        self.hashes = {}
        self.checks = 0
        self.failures = []
        self.maximum_errors = defaultdict(float)
        self.cache = {}

    def path(self, name):
        path = self.root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Expected regular evidence file: {name}")
        path.resolve(strict=True).relative_to(self.root)
        if path.stat().st_size > 64 * 1024**2:
            raise ValueError(f"Evidence file exceeds 64 MiB: {name}")
        if name not in self.hashes:
            self.hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def read(self, name):
        return json.loads(self.path(name).read_text())

    def check(self, label, condition):
        self.checks += 1
        if not bool(condition):
            self.failures.append(label)

    def close(self, label, actual, expected, *, rtol=1e-8, atol=1e-28):
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        self.check(label, a.shape == b.shape and np.all(np.isfinite(a)) and
                   np.all(np.isfinite(b)) and np.allclose(a, b, rtol=rtol, atol=atol))
        if a.shape == b.shape and a.size:
            group = label.split(":")[-1]
            self.maximum_errors[group] = max(self.maximum_errors[group],
                                             float(np.max(np.abs(a-b))))

    def arrays(self, query_id, index):
        if query_id not in self.cache:
            row = index[query_id]
            name = "inputs/" + row["array_path"]
            path = self.path(name)
            self.check(query_id+":array_sha256", self.hashes[name] == row["array_sha256"])
            with np.load(path, allow_pickle=False) as z:
                arrays = {key: np.asarray(z[key]).copy() for key in z.files}
            self.check(query_id+":finite_arrays", all(np.isfinite(a).all() for a in arrays.values()))
            expected = np.einsum("j...,k->jk...", arrays["branches"], arrays["features"]).reshape(arrays["design"].shape)
            self.close(query_id+":branch_feature_design", arrays["design"], expected,
                       rtol=1e-12, atol=1e-22)
            self.cache[query_id] = arrays
        return self.cache[query_id]

    def stack(self, ids, index):
        parent_queries = Counter(index[q]["parent_id"] for q in ids)
        matrices, targets, masses, differences, uncertainty = [], [], [], [], []
        for q in ids:
            a, row = self.arrays(q, index), index[q]
            target = (a["teacher"] - a["base"]).ravel()
            matrices.append(a["design"].reshape(40, -1).T)
            targets.append(target)
            masses.extend([1. / (len(parent_queries) * parent_queries[row["parent_id"]] * target.size)] * target.size)
            differences.append((a["refined"] - a["teacher"]).ravel())
            uncertainty.extend([row["reference"]["uncertainty_rms"]] * target.size)
        A, b, omega = np.concatenate(matrices), np.concatenate(targets), np.asarray(masses)
        root = np.sqrt(omega)
        self.close("stack:unit_total_mass", omega.sum(), 1., atol=1e-14)
        return dict(A=A, b=b, Aw=A*root[:, None], bw=b*root,
                    difference=np.concatenate(differences)*root,
                    uncertainty=float(np.sqrt(np.dot(omega, np.square(uncertainty)))),
                    parent_count=len(parent_queries), omega=omega)

    def validation(self, w, ids, index):
        by_parent = defaultdict(list)
        parity = 0.
        for q in ids:
            a = self.arrays(q, index)
            prediction = a["base"] + np.einsum("j,j...->...", w, a["design"])
            branches = a["base"] + np.einsum("j,j...->...", w.reshape(5, 8) @ a["features"], a["branches"])
            delta = float(np.max(np.abs(prediction-branches)))
            scale = max(1., float(np.max(np.abs(a["base"]) + np.einsum("j,j...->...", np.abs(w), np.abs(a["design"])))))
            self.check(q+":validation_parity", delta <= 128*np.finfo(float).eps*scale)
            parity = max(parity, delta)
            by_parent[index[q]["parent_id"]].append(float(np.mean((prediction-a["teacher"])**2)))
        parent_mse = {p:float(np.mean(values)) for p, values in by_parent.items()}
        return float(np.mean(list(parent_mse.values()))), parity, parent_mse

    def candidate(self, cell, c, t, v, svd, index):
        cid, name, family = cell["cell_id"], c["name"], c["family"]
        label = cid+"/"+name
        w = np.asarray(c["coefficients"], dtype=float)
        self.check(label+":finite_40_coefficients", w.shape == (40,) and np.isfinite(w).all())
        diag = c["diagnostics"]
        Aw, bw = t["Aw"], t["bw"]
        U, sigma, Vh = svd
        rank_cut = np.finfo(float).eps * max(Aw.shape) * sigma[0]
        rank = int(np.sum(sigma > rank_cut))
        residual = Aw@w-bw
        data_loss = float(residual@residual)
        penalty_loss, mu, gap = 0., None, None
        lam = float(c["ridge"] or 0.)
        B, penalty_matrix = None, None
        if family in ("coefficient_ridge", "action_energy"):
            L = np.asarray(diag["penalty_matrix"], dtype=float)
            prior = np.asarray(diag["prior_coefficients"], dtype=float)
            self.check(label+":zero_prior", np.array_equal(prior, np.zeros(40)))
            if family == "coefficient_ridge":
                self.check(label+":identity_penalty", np.array_equal(L, np.eye(40)))
                df = float(np.sum(sigma**2/(sigma**2+lam)))
            else:
                self.close(label+":action_penalty_gram", L.T@L, Aw.T@Aw, atol=1e-24)
                self.check(label+":full_rank_action_design", rank == 40)
                df = float(rank/(1+lam))
            penalty_matrix = math.sqrt(lam)*L
            pres = penalty_matrix@(w-prior)
            penalty_loss = float(pres@pres)
            B = np.vstack((Aw, penalty_matrix))
            singular_b = np.linalg.svd(B, compute_uv=False)
            scale = np.linalg.norm(B, axis=0)
            singular_scaled = np.linalg.svd(B/scale, compute_uv=False)
            mu = float(singular_b[-1]**2)
            normal = Aw.T@residual + penalty_matrix.T@pres
            gradient = float(2*np.linalg.norm(normal))
            gap = float((gradient/(2*singular_b[-1]))**2)
            self.close(label+":original_augmented_spectrum", singular_b, diag["original_augmented_singular_values"])
            self.close(label+":scaled_augmented_spectrum", singular_scaled, diag["augmented_singular_values"])
            self.close(label+":column_scales", scale, diag["column_scales"])
            self.close(label+":minimum_normal_eigenvalue", mu, diag["minimum_normal_eigenvalue"])
            self.close(label+":recorded_gap_formula", diag["gradient_norm"]**2/(4*diag["minimum_normal_eigenvalue"]), diag["optimality_gap_upper_bound"], atol=1e-42)
            self.check(label+":independent_objective_gap", gap <= diag["optimality_tolerance"])
            self.check(label+":independent_baseline_nonregression", data_loss+penalty_loss <= float(bw@bw)+diag["optimality_tolerance"])
            self.check(label+":eligible_positive_fit", c["eligible"] and diag["eligible"] and c["status"] == "SOLVED")
        elif family == "projection":
            df = float(rank)
            gradient = float(2*np.linalg.norm(Aw.T@residual))
            self.check(label+":ineligible_projection", not c["eligible"] and not diag["eligible"] and not diag["deployable"])
            self.check(label+":projection_residual_orthogonality",
                       np.linalg.norm(U[:, sigma > rank_cut].T@residual) <= 1e-8*np.linalg.norm(residual))
        else:
            df, gradient = 0., None
            self.check(label+":zero_control", family == "zero" and np.array_equal(w, np.zeros(40)) and c["eligible"])
        objective = data_loss+penalty_loss
        self.close(label+":data_loss", data_loss, diag["data_loss"])
        self.close(label+":penalty_loss", penalty_loss, diag["penalty_loss"])
        self.close(label+":objective", objective, diag["objective"])
        if family != "zero":
            self.close(label+":design_spectrum", sigma, diag["original_design_singular_values"])
            self.close(label+":effective_dof", df, diag["generalized_effective_degrees_of_freedom"], atol=1e-12)
            self.close(label+":coefficient_norm", np.linalg.norm(w), diag["coefficient_l2"])
            self.check(label+":design_rank", rank == diag["original_design_numerical_rank"])
        mse, parity, parent_mse = self.validation(w, cell["validation_query_ids"], index)
        self.close(label+":validation_mse", mse, c["validation_mse"])
        self.close(label+":validation_parity_maximum", parity, c["validation_parity_maximum"], atol=2e-16)
        result = dict(name=name, family=family, ridge=c["ridge"], eligible=c["eligible"],
                      data_loss=data_loss, penalty_loss=penalty_loss, objective=objective,
                      penalty_fraction_of_objective=penalty_loss/objective,
                      training_rms=math.sqrt(data_loss), validation_rms=math.sqrt(mse),
                      validation_parent_mse=parent_mse, coefficient_l2=float(np.linalg.norm(w)),
                      effective_degrees_of_freedom=df, recorded_optimality_gap=diag.get("optimality_gap_upper_bound"),
                      replay_optimality_gap=gap, replay_gradient_norm=gradient,
                      gap_diagnostics_are_certificates=False, validation_parity_maximum=parity)
        if family != "zero":
            saved = c["sensitivity"]
            if family == "coefficient_ridge":
                factors = sigma/(sigma*sigma+lam)
            elif family == "action_energy":
                factors = 1./((1+lam)*sigma)
            else:
                factors = np.where(sigma > rank_cut, 1./sigma, 0.)
            thin = (v["Aw"]@Vh.T)*factors
            gain = float(np.linalg.svd(thin, compute_uv=False)[0])
            ball = gain*t["uncertainty"]
            actual = float(np.linalg.norm(thin@(U.T@t["difference"])))
            sensitivity = dict(prediction_operator_gain=gain, estimated_ball_prediction_rms=ball,
                               actual_reference_perturbation_prediction_rms=actual,
                               training_uncertainty_radius=t["uncertainty"],
                               actual_reference_difference_weighted_norm=float(np.linalg.norm(t["difference"])),
                               validation_error_rms=math.sqrt(mse),
                               validation_remainder_rms=float(np.linalg.norm(v["bw"])),
                               validation_uncertainty_rms=v["uncertainty"])
            for prefix, value in (("estimated_ball_prediction", ball), ("actual_reference_perturbation_prediction", actual)):
                for denom in ("validation_error_rms", "validation_remainder_rms", "validation_uncertainty_rms"):
                    sensitivity[prefix+"_to_"+denom] = value/sensitivity[denom]
            for key, value in sensitivity.items():
                self.close(label+":sensitivity_"+key, value, saved[key], atol=1e-24)
            self.check(label+":sensitivity_nongating", saved["error_certificate"] is False and saved["scientific_eligibility_gate"] is False)
            self.check(label+":actual_difference_within_empirical_ball",
                       saved["actual_reference_difference_within_estimated_ball"] ==
                       bool(np.linalg.norm(t["difference"]) <= t["uncertainty"]))
            sensitivity.update(error_certificate=False, scientific_eligibility_gate=False,
                               estimated_ball_to_actual_perturbation=ball/actual)
            result["sensitivity"] = sensitivity
        return result

    def run(self):
        selection = self.read("selection.json")
        index_list = self.read("inputs/query_index.json")
        index = {q["query_id"]:q for q in index_list}
        summary = self.read("summary.json")
        sensitivity_table = {(r["cell_id"], r["candidate"]):r for r in self.read("sensitivity.json")}
        self.check("all_36_sensitivity_records", len(sensitivity_table) == 36)
        parents = self.read("inputs/parents.json")
        training_order = [p["parent_id"] for p in sorted(
            (p for p in parents if p["split"] == "train"), key=lambda p:p["cohort_order"])]
        roster = [("zero", None)] + [("coefficient_ridge", x) for x in (1e-6, 1e-10, 1e-12, 1e-14, 1e-16, 1e-18)]
        roster += [("action_energy", x) for x in (1e-6, 1e-4, 1e-2, 1., 100.)] + [("projection", None)]
        self.check("frozen_cells", [(c["training_parent_count"], c["calibration_source_count"]) for c in selection["cells"]] == [(8, 8), (32, 2), (32, 8)])
        self.check("expected_39_candidates", sum(len(c["candidates"]) for c in selection["cells"]) == 39)
        self.check("six_sealed_choices", selection["all_selections_sealed"] and sum(len(c["selected"]) for c in selection["cells"]) == 6)
        result_cells = []
        for cell in selection["cells"]:
            cid = cell["cell_id"]
            n, k = cell["training_parent_count"], cell["calibration_source_count"]
            expected_training = [q["query_id"] for q in index_list if q["split"] == "train" and q["role"] == "calibration" and q["parent_id"] in training_order[:n] and q["source_index"] < k]
            expected_validation = [q["query_id"] for q in index_list if q["split"] == "validation" and q["role"] == "withheld"]
            self.check(cid+":frozen_candidate_roster", [(c["family"], c["ridge"]) for c in cell["candidates"]] == roster)
            self.check(cid+":exact_training_prefix", cell["train_query_ids"] == expected_training and len(expected_training) == n*k*6)
            self.check(cid+":exact_validation_queries", cell["validation_query_ids"] == expected_validation and len(expected_validation) == 192)
            self.check(cid+":fit_copy", self.read("fits/"+cid+".json") == cell)
            for c in cell["candidates"]:
                if c["family"] != "zero":
                    separate = sensitivity_table[(cid, c["name"])]
                    self.check(cid+":"+c["name"]+":sensitivity_table_copy",
                               {key:value for key, value in separate.items() if key not in ("cell_id", "candidate")} == c["sensitivity"])
            t = self.stack(cell["train_query_ids"], index)
            v = self.stack(cell["validation_query_ids"], index)
            self.check(cid+":training_roles", all(index[q]["split"] == "train" and index[q]["role"] == "calibration" for q in cell["train_query_ids"]))
            self.check(cid+":validation_roles", all(index[q]["split"] == "validation" and index[q]["role"] == "withheld" for q in cell["validation_query_ids"]))
            U, sigma, Vh = np.linalg.svd(t["Aw"], full_matrices=False)
            candidates = [self.candidate(cell, c, t, v, (U, sigma, Vh), index) for c in cell["candidates"]]
            selected = {}
            for family in ("coefficient_ridge", "action_energy"):
                eligible = [c for c in candidates if c["family"] in ("zero", family) and c["eligible"]]
                winner = min(eligible, key=lambda c:c["validation_rms"]**2)
                self.check(cid+":"+family+":validation_selection", winner["name"] == cell["selected"][family]["name"])
                ordered = sorted(eligible, key=lambda c:c["validation_rms"])
                selected[family] = dict(candidate=winner["name"], ridge=winner["ridge"],
                                        validation_rms=winner["validation_rms"],
                                        next_validation_candidate=ordered[1]["name"],
                                        next_candidate_rms=ordered[1]["validation_rms"],
                                        fractional_rms_margin=(ordered[1]["validation_rms"]-winner["validation_rms"])/winner["validation_rms"])
            saved_by_name = {c["name"]:c for c in cell["candidates"]}
            projection = np.asarray(saved_by_name["projection"]["coefficients"])
            equivalence = []
            for record in cell["action_energy_projection_equivalence"]:
                c = saved_by_name[record["candidate"]]
                delta = np.asarray(c["coefficients"])-projection/(1+c["ridge"])
                difference = float(np.linalg.norm(t["Aw"]@delta))
                self.close(cid+":"+c["name"]+":uniform_shrinkage_record", difference, record["weighted_prediction_difference"], atol=1e-25)
                self.check(cid+":"+c["name"]+":uniform_shrinkage_arithmetic", difference <= 1e-12*np.linalg.norm(t["Aw"]@projection))
                equivalence.append(dict(candidate=c["name"], weighted_prediction_difference=difference,
                                        relative_training_correction_difference=difference/np.linalg.norm(t["Aw"]@projection),
                                        coefficient_difference_norm=float(np.linalg.norm(delta))))
            old_delta = np.asarray(saved_by_name["coefficient_ridge_1e-10"]["coefficients"])-np.asarray(cell["original_coefficients"])
            old_rms = float(np.linalg.norm(v["Aw"]@old_delta))
            self.close(cid+":original_strength_reproduction", old_rms, cell["original_strength_reproduction"]["validation_correction_difference_rms"], atol=1e-25)
            self.check(cid+":original_strength_arithmetic_tolerance", old_rms <= cell["original_strength_reproduction"]["tolerance"])
            result_cells.append(dict(cell_id=cid, training_queries=len(cell["train_query_ids"]),
                                     training_parents=cell["training_parent_count"], calibration_sources=cell["calibration_source_count"],
                                     training_rows=t["Aw"].shape[0], design_condition_number=float(sigma[0]/sigma[-1]),
                                     design_singular_values=sigma.tolist(),
                                     selected=selected, candidates=candidates, action_energy_uniform_shrinkage=equivalence,
                                     original_strength_reproduction_validation_rms=old_rms))
        curves = self.read("learning_curves.json")
        diagnostic = {r["method"]:r for r in curves if r["split"] == "diagnostic"}
        comparisons = []
        for cell in result_cells:
            cid = cell["cell_id"]
            original = diagnostic["original-"+cid]["rms"]
            for family in ("coefficient_ridge", "action_energy"):
                row = diagnostic[family+"-"+cid]
                baseline = diagnostic["original-"+cid]
                old_parents = {p["parent_id"]:p for p in baseline["parents"]}
                pairs = [dict(parent_id=p["parent_id"], selected_rms=p["rms"],
                              original_rms=old_parents[p["parent_id"]]["rms"],
                              fractional_rms_reduction=1-p["rms"]/old_parents[p["parent_id"]]["rms"])
                         for p in row["parents"]]
                comparisons.append(dict(cell_id=cid, family=family, selected_candidate=cell["selected"][family]["candidate"],
                                        diagnostic_rms=row["rms"], original_diagnostic_rms=original,
                                        diagnostic_fractional_rms_reduction=1-row["rms"]/original,
                                        ratio_to_ordered8=row["rms"]/diagnostic["ordered_8"]["rms"],
                                        ratio_to_lawson2=row["rms"]/diagnostic["lawson_2"]["rms"],
                                        diagnostic_parent_pairs=pairs,
                                        diagnostic_parents_improved=sum(p["fractional_rms_reduction"] > 0 for p in pairs),
                                        aggregate_source="Retained learning_curves.json; full independent metric replay is separate"))
        return dict(schema="tdn.review.regularization.science.v1", status="PASS" if not self.failures else "FAIL",
                    review_scope="Independent evaluation of all retained candidates, spectra, selection and sensitivity; no fits, teacher solves or uploaded-code execution",
                    reviewer_python_version=sys.version, reviewer_numpy_version=np.__version__,
                    review_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    native_status=summary["status"], candidate_count=39, selections=6,
                    input_arrays_loaded=len(self.cache), checks=self.checks, failures=self.failures,
                    maximum_absolute_replay_differences=dict(self.maximum_errors),
                    cells=result_cells, descriptive_diagnostic_comparisons=comparisons,
                    boundaries=["All source arrays were previously inspected development data.",
                                "Objective gaps and uncertainty balls are ordinary FP64 diagnostics, not certificates.",
                                "Sensitivity varies training labels with retained training/validation designs and penalty fixed; it does not enclose feature/design error, distribution shift or deployment precision error.",
                                "Projection is never selected; action-energy is uniform shrinkage, not a derived physical prior.",
                                "Validation selections use the fixed candidate menu; reported diagnostic comparisons do not make another choice.",
                                "No new inference timing or matched-accuracy speedup was measured.",
                                "The residual on the sampled 40-feature design is not a lower bound on full-operator or new-architecture error."],
                    proposed_next_gate=dict(status="PROPOSAL_ONLY_NOT_IMPLEMENTED_OR_RUN",
                        question="Is remaining error caused mainly by the five-branch source-action span or its shared eight-feature state-to-coefficient map?",
                        design="On the retained 32x8 training cell, freeze 192 context-local five-coefficient source-linear diagnostic fits using each context's eight calibration actions; score the four source-heldout actions per context (768 queries).",
                        controls="Frozen global 40-feature readout, zero/ordered2, and saved classical controls on identical queries. Declare SVD rank cutoff, conditioning, sensitivity and all denominators before fitting.",
                        evaluation_only_oracle="A second, explicitly evaluation-only five-coefficient projection may jointly fit the four withheld source actions per training context. It is never selected, deployed or described as out-of-sample. A fixed truncated-SVD projection gives a residual floor only within that retained numerical span on those sampled actions, not for the full operator or discarded directions.",
                        limits="Context-local fits consume context labels and are diagnostic oracles, not deployed models or new-parent generalization. Do not fit validation/diagnostic withheld truth; those parents lack calibration actions. Training orthogonality does not remove heldout signed cross terms.",
                        interpretation="Stable large oracle gains implicate coefficient-map/identification; small gains with large residual implicate branch span; unstable source-heldout gains implicate source coverage. Any derived BCH/Magnus branch, encoder, new labels or confirmation remains separately frozen future work.",
                        novelty="This attribution audit is established diagnostic methodology, not itself novelty. Novelty would require a distinct mathematically justified response mechanism and independent competitive evidence."),
                    input_file_sha256=self.hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = args.study.resolve(strict=True)
    output = args.output.resolve()
    if output == source or source in output.parents:
        parser.error("Review output must be outside immutable study evidence")
    result = Review(source).run()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps({k:result[k] for k in ("status", "checks", "candidate_count", "selections", "input_arrays_loaded", "failures")}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
