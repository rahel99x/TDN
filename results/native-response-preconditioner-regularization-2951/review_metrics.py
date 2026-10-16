#!/usr/bin/env python3
"""Independent retained-array P1R metric replay; no fitting or reference solves."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ERROR_FIELDS = ("mse", "rms", "maximum", "mean_error", "mean_component_mse", "mean_component_rms",
                "spatial_centered_mse", "spatial_centered_rms", "centered_mse", "centered_rms")
NORMS = ("reference", "remainder", "feedback")
CONTROLS = ("ordered_2", "ordered_8", "lawson_2", "lawson_4", "lawson_8")


def mean(values):
    return math.fsum(float(v)/len(values) for v in values)


def rms(values):
    a = np.asarray(values, dtype=float)
    return math.sqrt(float(np.mean(a*a)))


def pooled_norm(values):
    scale = max(values)
    return scale*math.sqrt(mean([(v/scale)**2 for v in values])) if scale else 0.


def ratios(result, source_rows=None):
    for name in NORMS:
        den, unc = result[name+"_rms"], result[name+"_uncertainty_rms"]
        status = "RESOLVED" if den > 10*unc else "UNRESOLVED_NEAR_NULL"
        if source_rows is not None and not all(r[name+"_resolved"] for r in source_rows):
            status = "UNRESOLVED_MEMBER"
        result[name+"_floor_rms"] = 10*unc
        result[name+"_status"] = status
        result[name+"_resolved"] = status == "RESOLVED"
        result["relative_"+name+"_rms"] = result["rms"]/den if status == "RESOLVED" else None
    result["error_to_uncertainty_rms"] = result["rms"]/result["reference_uncertainty_rms"]
    return result


def physical(prediction, arrays, reference):
    truth, base, matched = arrays["teacher"], arrays["base"], arrays["matched_mean"]
    e = prediction-truth
    m, mse = float(np.mean(e)), float(np.mean(e*e))
    centered = float(np.mean((e-m)**2))
    u, gu = reference["uncertainty_rms"], reference["matched_mean"]["uncertainty_rms"]
    result = dict(mse=mse, rms=math.sqrt(mse), maximum=float(np.max(np.abs(e))), mean_error=m,
        mean_component_mse=m*m, mean_component_rms=abs(m), spatial_centered_mse=centered,
        spatial_centered_rms=math.sqrt(centered), centered_mse=centered, centered_rms=math.sqrt(centered),
        reference_rms=rms(truth), remainder_rms=rms(truth-base), feedback_rms=rms(truth-matched),
        reference_uncertainty_rms=u, reference_uncertainty_maximum=reference["uncertainty_maximum"],
        remainder_uncertainty_rms=u, matched_mean_uncertainty_rms=gu, feedback_uncertainty_rms=u+gu)
    return ratios(result)


def aggregate(rows):
    result = {k: mean([r[k] for r in rows]) for k in
              ("mse", "mean_error", "mean_component_mse", "spatial_centered_mse")}
    result.update(rms=math.sqrt(result["mse"]), maximum=max(r["maximum"] for r in rows),
        mean_component_rms=math.sqrt(result["mean_component_mse"]),
        spatial_centered_rms=math.sqrt(result["spatial_centered_mse"]),
        centered_mse=result["spatial_centered_mse"], centered_rms=math.sqrt(result["spatial_centered_mse"]))
    for name in NORMS:
        for suffix in ("_rms", "_uncertainty_rms"):
            result[name+suffix] = pooled_norm([r[name+suffix] for r in rows])
    return ratios(result, rows)


def by_parent(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["parent_id"]].append(r)
    return {p: aggregate(rr) for p, rr in sorted(grouped.items())}


def csv_write(path, rows):
    fields = sorted({k for r in rows for k,v in r.items() if not isinstance(v,(list,dict))})
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader()
        w.writerows({k:r.get(k) for k in fields} for r in rows)


def run(study):
    failures, checks, input_hashes = [], Counter(), {}
    max_relative = defaultdict(float)
    max_absolute = defaultdict(float)
    def check(ok,label):
        checks["logical_checks"] += 1
        if not ok: failures.append(label)
    def compare(actual,expected,label,scope):
        checks[scope+"_fields_checked"] += 1
        if isinstance(expected,(float,int)) and not isinstance(expected,bool):
            if not math.isfinite(float(actual)):
                failures.append(label+": nonfinite");return
            delta=abs(actual-expected)
            relative=delta/max(abs(actual),abs(expected),np.finfo(float).tiny)
            max_absolute[scope]=max(max_absolute[scope],delta)
            max_relative[scope]=max(max_relative[scope],relative)
            if actual==expected:checks[scope+"_exact"] += 1
            elif not math.isclose(actual,expected,rel_tol=2e-12,abs_tol=1e-300):
                failures.append(label+f": {actual!r} != {expected!r}")
        else:
            if actual==expected:checks[scope+"_exact"] += 1
            else:failures.append(label+": unequal")
    def load(name):
        raw=(study/name).read_bytes();input_hashes[name]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
    queries=load("inputs/query_index.json")
    qmap={q["query_id"]:q for q in queries}
    check(len(qmap)==len(queries)==2688,"unique complete input query roster")
    withheld=[q for q in queries if q["role"]=="withheld"]
    check(len(withheld)==1152,"1152 withheld queries")
    selection=load("selection.json")
    selected,weights={},{}
    cellids=[]
    for cell in selection["cells"]:
        cid=cell["cell_id"];cellids.append(cid)
        for family,c in cell["selected"].items():
            method=family+"-"+cid;weights[method]=np.array(c["coefficients"],dtype=float)
            selected[method]=dict(name=c["name"],ridge=c["ridge"],validation_mse=c["validation_mse"])
        weights["original-"+cid]=np.array(cell["original_coefficients"],dtype=float)
        projection=next(c for c in cell["candidates"] if c["family"]=="projection")
        weights["projection-"+cid]=np.array(projection["coefficients"],dtype=float)
    methods=set(weights)|set(CONTROLS)
    check(len(methods)==17,"17 declared methods")
    recorded=load("metrics.json")
    recorded_bykey={(r["query_id"],r["method"]):r for r in recorded}
    expected_keys={(q["query_id"],m) for q in withheld for m in methods}
    check(len(recorded)==len(recorded_bykey)==19584 and set(recorded_bykey)==expected_keys,"19584 complete method/query slots")
    replayed=[]
    array_files, affine_checks, branch_checks, original_checks=0,0,0,0
    affine_max,branch_max=0.,0.
    for q in withheld:
        qid=q["query_id"]
        with np.load(study/"inputs"/q["array_path"],allow_pickle=False) as raw:
            arrays={k:raw[k] for k in raw.files}
        with np.load(study/"predictions"/(qid+".npz"),allow_pickle=False) as raw:
            predictions={k:raw[k] for k in raw.files}
        with np.load(study/"inputs/predictions"/(qid+".npz"),allow_pickle=False) as raw:
            old={k:raw[k] for k in raw.files}
        array_files += 3
        check(set(predictions)==methods,qid+" all prediction methods present")
        for method in sorted(methods):
            prediction=predictions[method]
            check(prediction.shape==(1,1,5,5) and np.isfinite(prediction).all(),qid+" "+method+" finite physical prediction")
            official=recorded_bykey[(qid,method)]
            check(official["status"]=="MEASURED",qid+" "+method+" measured status")
            if method in weights:
                w=weights[method]
                affine=arrays["base"]+np.einsum("j,j...->...",w,arrays["design"])
                five=arrays["base"]+np.einsum("j,j...->...",w.reshape(5,8)@arrays["features"],arrays["branches"])
                scale=max(1.,float(np.max(np.abs(arrays["base"])+np.einsum("j,j...->...",np.abs(w),np.abs(arrays["design"])))))
                tolerance=128*np.finfo(float).eps*scale
                ad=float(np.max(np.abs(prediction-affine)));bd=float(np.max(np.abs(five-affine)))
                affine_max=max(affine_max,ad);branch_max=max(branch_max,bd)
                check(ad<=tolerance,qid+" "+method+" saved versus affine")
                check(bd<=tolerance,qid+" "+method+" affine versus five branches")
                affine_checks+=1;branch_checks+=1
                compare(official["inference_parity_tolerance"],tolerance,qid+" "+method+" parity tolerance","parity")
                compare(official["inference_parity_maximum"],bd,qid+" "+method+" parity maximum","parity")
            if method in CONTROLS or method.startswith("original-"):
                oldkey="fit-"+method.removeprefix("original-") if method.startswith("original-") else method
                check(np.array_equal(prediction,old[oldkey]),qid+" "+method+" unchanged historical native prediction")
                original_checks+=1
            values=physical(prediction,arrays,q["reference"])
            for key,value in values.items():
                compare(official[key],value,qid+" "+method+" "+key,"query_metrics")
            denom=max(values["mse"],np.finfo(float).tiny)
            check(abs(values["mse"]-values["mean_component_mse"]-values["spatial_centered_mse"])<=64*np.finfo(float).eps*denom,
                  qid+" "+method+" spatial mean decomposition")
            n=official["training_parent_count"]
            membership=("not_applicable" if n is None else "not_training_split" if q["split"]!="train"
                        else "in_cohort" if int(q["parent_id"].split("-")[1])<n else "outside_cohort")
            check(official["training_membership"]==membership,qid+" "+method+" parent membership")
            replayed.append(dict(method=method,query_id=qid,parent_id=q["parent_id"],split=q["split"],
                family=q["family"],parent_family=q["parent_family"],start=q["start"],stop=q["stop"],
                training_membership=membership,**values))
    group_rows=defaultdict(list)
    for row in replayed:
        group_rows[(row["method"],row["split"],row["training_membership"])].append(row)
    official_groups=load("learning_curves.json")
    official_parents=load("parent_metrics.json")
    groups={(g["method"],g["split"],g["training_membership"]):g for g in official_groups}
    parents={(p["method"],p["split"],p["training_membership"],p["parent_id"]):p for p in official_parents}
    check(len(groups)==len(official_groups)==55 and set(groups)==set(group_rows),"55 complete summary groups")
    check(len(parents)==len(official_parents)==816,"816 unique parent summaries")
    summaries,parent_rows=[],[]
    for key,rows in sorted(group_rows.items()):
        method,split,membership=key
        pp=by_parent(rows)
        group_mse=mean([v["mse"] for v in pp.values()])
        for pid,values in pp.items():
            parent=parents[(*key,pid)]
            check(parent["complete"] and parent["planned_queries"]==24 and parent["measured_queries"]==24,
                  str((*key,pid))+" parent complete24")
            for name,value in values.items():compare(parent[name],value,str((*key,pid))+" "+name,"parent_metrics")
            parent_rows.append(dict(method=method,split=split,training_membership=membership,parent_id=pid,
                parent_family=next(r["parent_family"] for r in rows if r["parent_id"]==pid),
                group_mse_contribution_fraction=values["mse"]/(len(pp)*group_mse),**values))
        values=aggregate(list(pp.values()))
        official=groups[key]
        check(official["complete"] and official["independent_parent_count"]==len(pp) and
              official["measured_queries"]==len(rows),str(key)+" complete group accounting")
        for name,value in values.items():compare(official[name],value,str(key)+" "+name,"group_metrics")
        for name,value in (("median_parent_rms",float(np.median([p["rms"] for p in pp.values()]))),
                           ("worst_parent_rms",max(p["rms"] for p in pp.values()))):
            compare(official[name],value,str(key)+" "+name,"group_metrics");values[name]=value
        summary=dict(method=method,split=split,training_membership=membership,
            independent_parent_count=len(pp),query_count=len(rows),mean_error_fraction=values["mean_component_mse"]/values["mse"],
            spatial_error_fraction=values["spatial_centered_mse"]/values["mse"],
            errors_exceeding_ten_reference_uncertainties=sum(r["rms"]>10*r["reference_uncertainty_rms"] for r in rows),**values)
        if method in selected:summary.update(selected[method])
        summaries.append(summary)
    split_lookup={(r["method"],r["split"]):r for r in summaries if r["split"] in ("validation","diagnostic")}
    row_lookup={(r["query_id"],r["method"]):r for r in replayed}
    parent_lookup={(r["parent_id"],r["method"]):r for r in parent_rows}
    pairs=[]
    parent_deletion_sensitivity=[]
    comparisons=[]
    for cid in cellids:
        for family in ("coefficient_ridge","action_energy"):
            method=family+"-"+cid
            for comparator in ["original-"+cid,"projection-"+cid,*CONTROLS]:
                for split in ("validation","diagnostic"):
                    rr=[r for r in replayed if r["method"]==method and r["split"]==split]
                    pids=sorted({r["parent_id"] for r in rr})
                    pr=[parent_lookup[(p,method)]["rms"]/parent_lookup[(p,comparator)]["rms"] for p in pids]
                    own,other=split_lookup[(method,split)],split_lookup[(comparator,split)]
                    pairs.append(dict(method=method,comparator=comparator,split=split,
                        rms_ratio=own["rms"]/other["rms"],rms_reduction_fraction=1-own["rms"]/other["rms"],
                        maximum_ratio=own["maximum"]/other["maximum"],
                        improved_parents=sum(v<1 for v in pr),total_parents=len(pr),
                        parent_rms_ratio_minimum=min(pr),parent_rms_ratio_median=float(np.median(pr)),parent_rms_ratio_maximum=max(pr),
                        improved_queries=sum(r["mse"]<row_lookup[(r["query_id"],comparator)]["mse"] for r in rr),total_queries=len(rr)))
                    if comparator=="original-"+cid:
                        for omitted in pids:
                            retained=[p for p in pids if p!=omitted]
                            reduced_own=math.sqrt(mean([parent_lookup[(p,method)]["mse"] for p in retained]))
                            reduced_other=math.sqrt(mean([parent_lookup[(p,comparator)]["mse"] for p in retained]))
                            parent_deletion_sensitivity.append(dict(method=method,comparator=comparator,split=split,
                                omitted_parent=omitted,remaining_parents=len(retained),
                                unchanged_full_bank_rms_ratio=own["rms"]/other["rms"],
                                omission_summary_rms_ratio=reduced_own/reduced_other,
                                omitted_parent_rms_ratio=parent_lookup[(omitted,method)]["rms"]/parent_lookup[(omitted,comparator)]["rms"],
                                scope="Post hoc aggregation sensitivity only; no refit, no primary exclusion, all eight parents retained in primary tables"))
    for family in ("coefficient_ridge","action_energy","original","projection"):
        for split in ("validation","diagnostic"):
            a,b=split_lookup[(family+"-p032-s02",split)],split_lookup[(family+"-p008-s08",split)]
            c=split_lookup[(family+"-p032-s08",split)]
            comparisons.append(dict(family=family,split=split,equal_label_training_queries=384,
                more_parents_32x2_over_8x8_rms_ratio=a["rms"]/b["rms"],
                more_sources_32x8_over_32x2_rms_ratio=c["rms"]/a["rms"]))
    strata=[]
    for (method,split),summary in sorted(split_lookup.items()):
        rr=[r for r in replayed if r["method"]==method and r["split"]==split]
        for axis,func in (("source",lambda r:r["family"]),("morphology",lambda r:r["parent_family"]),
                          ("interval",lambda r:f"{r['start']:g}:{r['stop']:g}")):
            subsets=defaultdict(list)
            for r in rr:subsets[func(r)].append(r)
            for label,subset in sorted(subsets.items()):
                pp=by_parent(subset);values=aggregate(list(pp.values()))
                strata.append(dict(method=method,split=split,axis=axis,label=label,query_count=len(subset),
                    parent_count=len(pp),mse_contribution_fraction=sum(r["mse"] for r in subset)/sum(r["mse"] for r in rr),
                    mean_component_fraction=values["mean_component_mse"]/values["mse"],**values))
    return dict(schema="tdn.native_p1r.metrics_review.v1",status="PASS" if not failures else "FAIL",
        scope="Independent retained numerical-array replay only; no fitting, reference acquisition, uploaded-code execution or new timing",
        failures=failures,checks=dict(checks),maximum_relative_discrepancy=dict(max_relative),
        maximum_absolute_discrepancy=dict(max_absolute),input_sha256=input_hashes,
        query_count=1152,method_count=17,metric_rows=len(replayed),array_files_read=array_files,
        affine_prediction_checks=affine_checks,five_branch_prediction_checks=branch_checks,
        historical_prediction_exact_checks=original_checks,maximum_affine_prediction_discrepancy=affine_max,
        maximum_five_branch_discrepancy=branch_max,selected=selected,split_summaries=summaries,
        parent_summaries=parent_rows,paired_comparisons=pairs,strata=strata,equal_label_comparisons=comparisons,
        parent_deletion_sensitivity=parent_deletion_sensitivity,
        limitations=["These already-inspected development parents do not establish fresh generalization or a population confidence interval.",
            "Projection is an offline fixed-feature training diagnostic, not a selected deployment competitor or full-operator capacity bound.",
            "Reference uncertainty is an empirical estimate, not an error certificate; near-floor classical error ratios are unresolved.",
            "All timings belong to the earlier P1 run; this audit replay measures no new inference latency or practical speedup.",
            "Stratum shares for morphology, interval and source overlap and cannot be added across axes."])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--study",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    if args.output.resolve().is_relative_to(args.study.resolve()):
        raise ValueError("Write derived review artifacts outside the original study directory")
    result=run(args.study)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    for suffix,key in (("splits","split_summaries"),("parents","parent_summaries"),("strata","strata"),("pairs","paired_comparisons")):
        csv_write(args.output.parent/("metrics-"+suffix+".csv"),result[key])
    print(json.dumps({k:result[k] for k in ("status","metric_rows","checks","failures")}))
    raise SystemExit(0 if result["status"]=="PASS" else 1)


if __name__=="__main__":main()
