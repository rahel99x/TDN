#!/usr/bin/env python3
"""Independent read-only replay of retained P1 evidence; never fits/solves.

Only standard-library JSON and numerical NPZ data (allow_pickle=False) are read.
No uploaded or production Python is imported. The full declared query/method
roster is required. Descriptive summaries weight independent parents equally;
no resampling, significance, unseen-data or full-operator claims are made.
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

METHODS = [f"fit-p{p:03d}-s{s:02d}" for p in (8, 16, 32) for s in (2, 4, 8)] + [
    "ordered_2", "ordered_8", "lawson_2", "lawson_4", "lawson_8"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rms(a):
    return float(np.sqrt(np.mean(np.square(a))))


def span(values):
    a = np.asarray(values, dtype=np.float64)
    require(bool(a.size) and np.isfinite(a).all(), "Invalid summary values")
    return dict(minimum=float(a.min()), median=float(np.median(a)), maximum=float(a.max()))


class Audit:
    def __init__(self, study):
        self.study = study.resolve()
        self.hashes = {}
        self.deltas = defaultdict(float)
        self.comparisons = Counter()
        self.exact = Counter()

    def path(self, name):
        candidate = self.study / name
        require(not candidate.is_symlink() and candidate.resolve().is_relative_to(self.study), "Unsafe input path")
        require(candidate.is_file(), f"Missing input {name}")
        return candidate

    def read(self, name):
        data = self.path(name).read_bytes()
        self.hashes[name] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    def array(self, name):
        path = self.path(name)
        self.hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as archive:
            values = {k: archive[k] for k in archive.files}
        require(all(v.dtype.kind in "biuf" and np.isfinite(v).all() for v in values.values()), "Nonfinite/non-real NPZ")
        return values

    def close(self, label, got, expected, *, atol=2e-26, rtol=2e-12):
        self.comparisons[label] += 1
        if got is None or expected is None:
            require(got is expected, f"{label}: null mismatch")
            self.exact[label] += 1
            return
        require(math.isfinite(float(got)) and math.isfinite(float(expected)), f"{label}: nonfinite")
        delta = abs(float(got) - float(expected))
        self.deltas[label] = max(self.deltas[label], delta)
        self.exact[label] += int(delta == 0)
        require(delta <= atol + rtol * abs(expected), f"{label}: {got!r} != {expected!r}")


def aggregate(rows):
    """Equal parents, equal queries per parent; spatial and pooled centering kept separate."""
    buckets = defaultdict(list)
    for r in rows:
        buckets[r["parent_id"]].append(r)
    parents = []
    for pid, values in sorted(buckets.items()):
        mse = float(np.mean([r["mse"] for r in values]))
        mean = float(np.mean([r["mean_error"] for r in values]))
        p = dict(parent_id=pid, parent_family=values[0]["parent_family"], queries=len(values),
                 mse=mse, rms=math.sqrt(mse), maximum=max(r["maximum"] for r in values),
                 mean_error=mean, centered_rms=math.sqrt(max(0., mse-mean*mean)),
                 spatial_centered_rms=rms([r["centered_rms"] for r in values]),
                 mean_component_rms=rms([r["mean_error"] for r in values]))
        for k in ("reference_rms", "remainder_rms", "feedback_rms", "reference_uncertainty_rms"):
            p[k] = rms([r[k] for r in values])
        parents.append(p)
    mse = float(np.mean([p["mse"] for p in parents]))
    mean = float(np.mean([p["mean_error"] for p in parents]))
    result = dict(parents=parents, parent_count=len(parents), queries=len(rows), mse=mse, rms=math.sqrt(mse),
                  median_parent_rms=float(np.median([p["rms"] for p in parents])),
                  worst_parent_rms=max(p["rms"] for p in parents), maximum=max(p["maximum"] for p in parents),
                  mean_error=mean, centered_rms=math.sqrt(max(0., mse-mean*mean)),
                  spatial_centered_rms=rms([p["spatial_centered_rms"] for p in parents]),
                  mean_component_rms=rms([p["mean_component_rms"] for p in parents]))
    result["mean_component_fraction_of_mse"] = result["mean_component_rms"]**2/mse
    for k in ("reference_rms", "remainder_rms", "feedback_rms", "reference_uncertainty_rms"):
        result[k] = rms([p[k] for p in parents])
        result["error_to_"+k] = result["rms"]/result[k]
    result["error_over_reference_uncertainty"] = span([r["error_to_uncertainty_rms"] for r in rows])
    result["errors_above_10x_estimated_uncertainty"] = sum(r["rms"] > 10*r["reference_uncertainty_rms"] for r in rows)
    return result


def pairs(before, after):
    old = {r["parent_id"]: r for r in before["parents"]}
    new = {r["parent_id"]: r for r in after["parents"]}
    require(set(old) == set(new), "Pairing changed the evaluation parent bank")
    rows = []
    for pid in sorted(old):
        a, b = old[pid], new[pid]
        rows.append(dict(parent_id=pid, parent_family=a["parent_family"], before_rms=a["rms"], after_rms=b["rms"],
                         rms_difference=b["rms"]-a["rms"], mse_difference=b["mse"]-a["mse"],
                         improvement_ratio=a["rms"]/b["rms"], relative_rms_change=b["rms"]/a["rms"]-1,
                         improved=b["rms"] < a["rms"]))
    return dict(parents=rows, parent_count=len(rows), improved=sum(r["improved"] for r in rows),
                degraded=sum(r["rms_difference"] > 0 for r in rows),
                before_rms=before["rms"], after_rms=after["rms"],
                relative_aggregate_rms_change=after["rms"]/before["rms"]-1,
                parent_relative_rms_change=span([r["relative_rms_change"] for r in rows]),
                median_rms_difference=float(np.median([r["rms_difference"] for r in rows])),
                worst_rms_difference=max(r["rms_difference"] for r in rows))


def review(study):
    audit = Audit(study)
    protocol = audit.read("protocol.json")
    index = audit.read("query_index.json")
    saved = audit.read("metrics.json")
    selection = audit.read("selection.json")
    parents = audit.read("parents.json")
    require(len(parents) == 48 and Counter(p["split"] for p in parents) == {"train":32, "validation":8, "diagnostic":8}, "Parent roster")
    require(len(index) == 2688 and len({q["query_id"] for q in index}) == 2688, "Query roster")
    require(len(saved) == 16128 and len({(m["query_id"],m["method"]) for m in saved}) == 16128, "Metric roster")
    require(all(q["status"] == "ACCEPTED" for q in index), "Unaccepted references")
    require(all(m["status"] == "MEASURED" for m in saved), "Unmeasured metrics")
    require(len(selection["cells"]) == 9 and all(c["status"] == "SELECTED" for c in selection["cells"]), "Selected fit roster")
    queries = {q["query_id"]:q for q in index}
    parent_index = {p["parent_id"]:p for p in parents}
    require(len(parent_index) == 48, "Duplicate parent identity")
    counts = Counter((q["parent_id"], q["role"]) for q in index)
    for parent in parents:
        require(counts[(parent["parent_id"], "withheld")] == 24, "Withheld parent denominator")
        require(counts[(parent["parent_id"], "calibration")] == (48 if parent["split"] == "train" else 0), "Calibration parent denominator")
    cells = {c["model_id"]:c for c in selection["cells"]}
    metric_map = {(m["query_id"],m["method"]):m for m in saved}
    arrays, predictions = {}, {}
    context_features = {}
    reference_summary = []
    branch_parity = []
    replayed = []
    for q in index:
        qid = q["query_id"]
        parent = parent_index[q["parent_id"]]
        require(all(q[k] == parent[k] for k in ("split", "parent_family")), "Query parent metadata")
        require([q["start"], q["stop"]] in protocol["intervals"], "Unplanned interval")
        a = audit.array(q["array_path"])
        require(audit.hashes[q["array_path"]] == q["array_sha256"], f"Array hash {qid}")
        require(audit.read(q["record_path"]) == q, f"Record/index mismatch {qid}")
        require(a["design"].shape == (40,1,1,5,5) and a["teacher"].shape == (1,1,5,5), "Unexpected dimensions")
        arrays[qid] = a
        ref = q["reference"]
        audit.close("source_unit_rms", rms(a["source"]), 1., atol=1e-14, rtol=0)
        context_features.setdefault(q["context_id"], a["features"])
        require(np.array_equal(context_features[q["context_id"]],a["features"]), "Source-dependent coefficients")
        factored = (a["branches"][:,None,...]*a["features"].reshape(1,8,1,1,1,1)).reshape(a["design"].shape)
        audit.close("design_factorization_maximum", np.max(np.abs(factored-a["design"])), 0., atol=1e-16)
        for prefix,tname,rname,r in (("U","teacher","refined",ref),("G","matched_mean","matched_mean_refined",ref["matched_mean"])):
            delta=a[tname]-a[rname]
            audit.close(prefix+"_independent_refined_rms",rms(delta),r["cross_difference"]["rms"])
            audit.close(prefix+"_independent_refined_maximum",np.max(np.abs(delta)),r["cross_difference"]["maximum"])
            audit.close(prefix+"_uncertainty_rms_sum",r["refinement"]["uncertainty_rms"]+r["cross_difference"]["rms"],r["uncertainty_rms"])
        u = float(ref["uncertainty_rms"])
        ug = u + float(ref["matched_mean"]["uncertainty_rms"])
        remainder = rms(a["teacher"]-a["base"])
        feedback = rms(a["teacher"]-a["matched_mean"])
        reference_summary.append(dict(query_id=qid, split=q["split"], role=q["role"],
                                      uncertainty_rms=u, feedback_uncertainty_rms=ug,
                                      remainder_over_uncertainty=remainder/u, feedback_over_uncertainty=feedback/ug))
        if q["role"] != "withheld":
            continue
        pred = audit.array("predictions/"+qid+".npz")
        predictions[qid] = pred
        require(set(pred) == set(METHODS), f"Prediction methods {qid}")
        audit.close("ordered2_equals_base_maximum",np.max(np.abs(pred["ordered_2"]-a["base"])),0.,atol=1e-14)
        for method in METHODS:
            m = metric_map[(qid,method)]
            require(all(m[k] == q[k] for k in ("query_id", "parent_id", "parent_family", "split", "family", "start", "stop", "context_id", "role", "source_index")), "Metric/query identity mismatch")
            membership = (("in_cohort" if q["parent_id"] in cells[method]["training_parent_ids"] else "outside_cohort")
                          if method in cells and q["split"] == "train" else
                          "not_training_split" if method in cells else "not_applicable")
            require(m["training_membership"] == membership, "Incorrect training membership")
            e = pred[method]-a["teacher"]
            mse = float(np.mean(e*e)); error_rms=math.sqrt(mse); mean=float(np.mean(e))
            remainder_status="RESOLVED" if remainder>10*u else "UNRESOLVED_NEAR_NULL"
            feedback_status="RESOLVED" if feedback>10*ug else "UNRESOLVED_NEAR_NULL"
            calculated = dict(mse=mse,rms=error_rms,maximum=float(np.max(np.abs(e))),mean_error=mean,
                              centered_rms=rms(e-mean),reference_rms=rms(a["teacher"]),remainder_rms=remainder,
                              feedback_rms=feedback,reference_uncertainty_rms=u,
                              reference_uncertainty_maximum=ref["uncertainty_maximum"],
                              relative_reference_rms=error_rms/rms(a["teacher"]),
                              relative_remainder_rms=error_rms/remainder if remainder_status=="RESOLVED" else None,
                              relative_feedback_rms=error_rms/feedback if feedback_status=="RESOLVED" else None,
                              error_to_uncertainty_rms=error_rms/u,K_minus_P_rms=rms(pred[method]-a["base"]),
                              U_minus_P_rms=remainder,U_minus_G_rms=feedback,U_minus_P_uncertainty_rms=u,
                              U_minus_G_uncertainty_rms=ug,U_minus_P_relative_error=error_rms/remainder if remainder_status=="RESOLVED" else None,
                              U_minus_G_relative_error=error_rms/feedback if feedback_status=="RESOLVED" else None)
            for key,value in calculated.items():
                audit.close("metric_"+key,value,m[key])
            for key,status in (("remainder_status",remainder_status),("U_minus_P_status",remainder_status),
                               ("feedback_status",feedback_status),("U_minus_G_status",feedback_status)):
                require(m[key]==status,f"Signal resolution mismatch {key}")
            if method in cells:
                c=np.asarray(cells[method]["selected"]["coefficients"])
                reconstructed=a["base"]+np.einsum("j,j...->...",c,a["design"])
                maximum=float(np.max(np.abs(reconstructed-pred[method])))
                tolerance=64*np.finfo(float).eps*max(1.,float(np.max(np.abs(reconstructed))))
                require(maximum<=tolerance,"Affine fresh inference mismatch")
                audit.close("retained_inference_parity",maximum,m["inference_parity_maximum"],atol=1e-16,rtol=0)
                audit.close("retained_inference_tolerance",tolerance,m["inference_parity_tolerance"])
                effective=c.reshape(5,8)@a["features"]
                five=a["base"]+np.einsum("j,j...->...",effective,a["branches"])
                branch_parity.append(float(np.max(np.abs(five-pred[method]))))
                require(branch_parity[-1]<=tolerance,"Five-branch fresh inference mismatch")
            replayed.append({**{k:m[k] for k in ("query_id","parent_id","parent_family","split","family","start","stop","method","training_membership")},**calculated})
    require(len(replayed)==16128,"Replay count")
    # Re-evaluate retained coefficient candidates without optimizing or acquiring labels.
    fit_summaries=[]
    for method,c in sorted(cells.items()):
        candidates=[]
        trainids=c["train_query_ids"]; validids=c["validation_query_ids"]
        require(all(queries[i]["split"]=="train" and queries[i]["role"]=="calibration" for i in trainids),"Train leakage")
        require(all(queries[i]["split"]=="validation" and queries[i]["role"]=="withheld" for i in validids),"Selection leakage")
        require(len(trainids)==c["training_parent_count"]*6*c["calibration_source_count"],"Cell train denominator")
        require(len(validids)==192,"Cell validation denominator")
        A=np.concatenate([arrays[i]["design"].reshape(40,25).T for i in trainids])
        b=np.concatenate([(arrays[i]["teacher"]-arrays[i]["base"]).ravel() for i in trainids])
        Av=np.concatenate([arrays[i]["design"].reshape(40,25).T for i in validids])
        bv=np.concatenate([(arrays[i]["teacher"]-arrays[i]["base"]).ravel() for i in validids])
        require(len(set(Counter(queries[i]["parent_id"] for i in trainids).values()))==1,"Unequal train counts")
        weighted_design = A / math.sqrt(A.shape[0])
        data_singular = np.linalg.svd(weighted_design, compute_uv=False)
        rank_threshold = np.finfo(float).eps * max(A.shape) * data_singular[0]
        data_rank = int(np.count_nonzero(data_singular > rank_threshold))
        data_condition = float(data_singular[0] / data_singular[-1])
        for candidate in c["candidates"]:
            weights=np.asarray(candidate["coefficients"])
            lam=candidate["ridge"] or 0.
            residual=A@weights-b
            trainloss=float(np.mean(residual**2)); penalty=float(lam*np.dot(weights,weights))
            validation=float(np.mean((Av@weights-bv)**2))
            audit.close("candidate_validation_loss",validation,candidate["validation_loss"],rtol=1e-10)
            gradient=2*(A.T@residual/b.size+lam*weights)
            d=candidate["diagnostics"]
            item=dict(name=candidate["name"],eligible=candidate["eligible"],ridge=candidate["ridge"],
                      training_data_mse=trainloss,training_data_rms=math.sqrt(trainloss),coefficient_penalty=penalty,
                      objective=trainloss+penalty,validation_mse=validation,validation_rms=math.sqrt(validation),
                      coefficient_l2=float(np.linalg.norm(weights)),gradient_l2=float(np.linalg.norm(gradient)))
            if lam:
                audit.close("candidate_objective",trainloss+penalty,d["objective"],rtol=1e-10)
                require(data_rank == d["original_design_numerical_rank"], "Original design rank")
                audit.close("candidate_original_condition", data_condition, d["original_design_condition_number"], rtol=1e-9)
                scales = np.hypot(np.linalg.norm(weighted_design, axis=0), math.sqrt(lam))
                augmented = np.concatenate((weighted_design / scales, np.diag(math.sqrt(lam) / scales)))
                singular = np.linalg.svd(augmented, compute_uv=False)
                augmented_condition = float(singular[0] / singular[-1])
                audit.close("candidate_augmented_condition", augmented_condition, d["augmented_condition_number"], rtol=1e-10)
                item.update(ordinary_fp64_gap_diagnostic=float(np.dot(gradient,gradient)/(4*lam)),
                            augmented_condition_number=augmented_condition,
                            original_design_numerical_rank=data_rank,
                            original_design_condition_number=data_condition,
                            penalty_fraction=penalty/(trainloss+penalty))
            candidates.append(item)
        best=min((r for r in candidates if r["eligible"]),key=lambda r:r["validation_mse"])
        require(best["name"]==c["selected"]["name"],"Validation selection does not reproduce")
        fit_summaries.append(dict(method=method,training_parent_count=c["training_parent_count"],
                                  calibration_source_count=c["calibration_source_count"],training_query_count=len(trainids),
                                  validation_query_count=len(validids),selected=best,candidates=candidates))
    groups=[]; parentrows=[]
    buckets=defaultdict(list)
    for r in replayed:
        buckets[(r["method"],r["split"],r["training_membership"])].append(r)
    for (method,split,membership),values in sorted(buckets.items()):
        g=dict(method=method,split=split,training_membership=membership,**aggregate(values))
        g["strata"]={}
        for key in ("family","parent_family"):
            g["strata"][key]={v:aggregate([r for r in values if r[key]==v]) for v in sorted({r[key] for r in values})}
        g["strata"]["interval"]={f"{s:g}:{t:g}":aggregate([r for r in values if (r["start"],r["stop"])==(s,t)])
                                   for s,t in sorted({(r["start"],r["stop"]) for r in values})}
        groups.append(g)
        parentrows.extend(dict(method=method,split=split,training_membership=membership,**p) for p in g["parents"])
    bygroup={(g["method"],g["split"],g["training_membership"]):g for g in groups}
    scalarfields=("mse","rms","maximum","mean_error","centered_rms","reference_rms","remainder_rms","feedback_rms","reference_uncertainty_rms")
    retainedgroups=audit.read("learning_curves.json")
    require(len(retainedgroups)==len(groups)==48,"Learning curve denominator")
    for g in retainedgroups:
        fresh=bygroup[(g["method"],g["split"],g["training_membership"])]
        require(g["complete"] and g["independent_parent_count"]==fresh["parent_count"],"Incomplete group")
        for field in scalarfields+("median_parent_rms","worst_parent_rms"):
            audit.close("group_"+field,fresh[field],g[field])
        for kind,strata in g["strata"].items():
            require(set(strata)==set(fresh["strata"][kind]),"Missing stratum")
            for key,v in strata.items():
                for field in scalarfields:
                    audit.close("stratum_"+field,fresh["strata"][kind][key][field],v[field])
    pm={(p["method"],p["split"],p["training_membership"],p["parent_id"]):p for p in parentrows}
    retainedparents=audit.read("parent_metrics.json")
    require(len(retainedparents)==len(parentrows)==672,"Parent metric denominator")
    for p in retainedparents:
        fresh=pm[(p["method"],p["split"],p["training_membership"],p["parent_id"])]
        require(p["complete"] and p["planned_queries"]==fresh["queries"],"Parent query denominator")
        for field in scalarfields:
            audit.close("parent_"+field,fresh[field],p[field])
    def get(method,split):
        return bygroup[(method,split,"not_training_split" if method.startswith("fit-") else "not_applicable")]
    axis=[]
    for split in ("validation","diagnostic"):
        for s in (2,4,8):
            for p0,p1 in ((8,16),(16,32),(8,32)):
                a,b=f"fit-p{p0:03d}-s{s:02d}",f"fit-p{p1:03d}-s{s:02d}"
                axis.append(dict(split=split,axis="parent_count",fixed_count=s,before_method=a,after_method=b,**pairs(get(a,split),get(b,split))))
        for p in (8,16,32):
            for s0,s1 in ((2,4),(4,8),(2,8)):
                a,b=f"fit-p{p:03d}-s{s0:02d}",f"fit-p{p:03d}-s{s1:02d}"
                axis.append(dict(split=split,axis="source_count",fixed_count=p,before_method=a,after_method=b,**pairs(get(a,split),get(b,split))))
    recordedpairs=audit.read("paired_comparisons.json")
    for key,ax in (("paired_parent_count","parent_count"),("paired_source_count","source_count")):
        for old in recordedpairs[key]:
            fresh=next(r for r in axis if (r["split"],r["axis"],r["before_method"],r["after_method"])==
                       (old["split"],ax,old["before_method"],old["after_method"]))
            audit.close("paired_improved_fraction",fresh["improved"]/fresh["parent_count"],old["improved_parent_fraction"])
            for k in ("median_rms_difference","worst_rms_difference"):
                audit.close("paired_"+k,fresh[k],old[k])
    classical=[]
    for split in ("validation","diagnostic"):
        for method in METHODS[:9]:
            for baseline in METHODS[9:]:
                classical.append(dict(split=split,learned_method=method,baseline=baseline,**pairs(get(baseline,split),get(method,split))))
    equal_query_comparisons = []
    for split in ("validation", "diagnostic"):
        for before, after in (("fit-p008-s04", "fit-p016-s02"),
                              ("fit-p008-s08", "fit-p016-s04"),
                              ("fit-p008-s08", "fit-p032-s02"),
                              ("fit-p016-s04", "fit-p032-s02"),
                              ("fit-p016-s08", "fit-p032-s04")):
            count = len(cells[before]["train_query_ids"])
            require(count == len(cells[after]["train_query_ids"]), "Equal query comparison")
            equal_query_comparisons.append(dict(split=split, training_query_count=count,
                before_method=before, after_method=after, **pairs(get(before, split), get(after, split))))
    for g in groups:
        for strata in g["strata"].values():
            for v in strata.values():
                # Frozen balanced counts permit this additive MSE allocation.
                v["fraction_of_parent_group_mse"] = v["mse"] * v["queries"] / (g["mse"] * g["queries"])
    return dict(schema="tdn.independent_scaling_scientific_review.v1",status="PASS",mode="READ_ONLY_RETAINED_ARRAY_REPLAY",
                scope=dict(query_pairs=2688,withheld_query_pairs=1152,metric_rows=16128,selected_fits=9,candidate_replays=27,
                           parent_metric_rows=672,learning_curve_groups=48,physical_parents=48,
                           source_action_predictions=16128,affine_prediction_replays=10368,
                           fresh_teacher_solves=0,new_fits=0,uploaded_code_executed=False),
                input_hashes=audit.hashes,
                comparison_tolerances=dict(default_relative=2e-12,default_absolute=2e-26,
                                           candidate_relative=1e-10,original_condition_relative=1e-9,affine_maximum="64 eps64 max(1,maxabs(prediction))"),
                comparisons=dict(audit.comparisons),exact_comparisons=dict(audit.exact),maximum_absolute_differences=dict(audit.deltas),
                five_branch_prediction_maximum=max(branch_parity),selected_fits=fit_summaries,
                reference_summary=dict(U_uncertainty_rms=span([r["uncertainty_rms"] for r in reference_summary]),
                                       U_minus_P_over_uncertainty=span([r["remainder_over_uncertainty"] for r in reference_summary]),
                                       U_minus_G_over_combined_uncertainty=span([r["feedback_over_uncertainty"] for r in reference_summary]),
                                       unresolved_remainder=sum(r["remainder_over_uncertainty"]<=10 for r in reference_summary),
                                       unresolved_feedback=sum(r["feedback_over_uncertainty"]<=10 for r in reference_summary),
                                       uncertainties_are_bounds=False),
                learning_curves=groups,parent_metrics=parentrows,paired_scaling_axes=axis,paired_classical_comparisons=classical,
                equal_training_query_comparisons=equal_query_comparisons,
                interpretation_limits=["One nested training order, eight validation and eight diagnostic parents; no population learning law.",
                                       "Each source/context is a paired descendant, not a new physical parent; only two diagnostic parents per morphology.",
                                       "Validation chose ridge per cell. Diagnostic ranking among nine cells is exploratory, not a newly selected deployable model.",
                                       "Development verifier uses prefixes of this seed bank; no untouched confirmation claim.",
                                       "Four normalized withheld directions per context do not establish full-operator accuracy.",
                                       "5x5 same-grid FD, short times, this physics distribution only; no endpoint rollout, continuum or GPU result.",
                                       "Estimated refinement/cross-method uncertainty is not a rigorous error certificate.",
                                       "Recorded centered_rms centers the pooled error; spatial_centered_rms here separately averages per-action spatial variance.",
                                       "Accuracy-only replay cannot establish matched-tolerance speedup; complete native timing and setup costs require their own review."])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    require(not args.output.resolve().is_relative_to(args.study.resolve()),"Write review outside original evidence")
    result=review(args.study)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps(dict(status=result["status"],scope=result["scope"],output=str(args.output))))


if __name__=="__main__":
    main()
