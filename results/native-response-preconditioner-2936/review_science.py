#!/usr/bin/env python3
"""Descriptive review only: never trains, solves a teacher, or runs model inference."""
import argparse
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def key(row):
    return tuple(row[k] for k in ("parent_id", "kind", "horizon", "direction"))


def summarize(root, cloud, old_zip):
    stage = root / "outputs/smoke"
    read = lambda name: json.loads((stage / name).read_text())
    metrics, references, selection = read("metrics.json"), read("references.json"), read("selection.json")
    index = read("array_index.json")
    arrays = np.load(stage / "arrays.npz", allow_pickle=False)
    weights = np.asarray(selection["coefficients"])
    fits = read("fitting.json")
    chosen = next(r for r in fits if r["candidate"] == selection["selected_candidate"])
    grouped = {}
    for row in metrics:
        grouped.setdefault(key(row), {})[row["method"]] = row
    refs = {key(r): r["reference"] for r in references}
    endpoint, actions = [], []
    for q, methods in grouped.items():
        if not q[0].startswith("diagnostic"):
            continue
        fitted = methods["fixed_feature_fitted"]
        out = {k: fitted[k] for k in ("parent_id", "horizon", "direction", "kind")}
        out.update(fitted_rms=fitted["rms"], reference_uncertainty_rms=refs[q]["uncertainty_rms"])
        if q[1] == "endpoint":
            baseline = methods["ordered_response_corrected_2"]
            out.update(classical_rms=baseline["rms"], fitted_minus_classical_rms=fitted["rms"]-baseline["rms"],
                       fitted_seconds=fitted["elapsed_seconds"], classical_seconds=baseline["elapsed_seconds"],
                       fitted_over_classical_single_call_time=fitted["elapsed_seconds"]/baseline["elapsed_seconds"],
                       baseline_metrics_equal_cached_zero=all(baseline[k]==methods["zero_readout"][k] for k in ("rms","maximum","mean_error","centered_rms")))
            qi = next(i for i,r in enumerate(index["queries"]) if key(r)==q)
            design = arrays[f"query_{qi:03}_design"].reshape(40,-1)
            out["retained_design_learned_correction_rms"] = rms(weights @ design)
            out["correction_over_reference_uncertainty"] = out["retained_design_learned_correction_rms"]/refs[q]["uncertainty_rms"]
            endpoint.append(out)
        else:
            out.update(classical_2_rms=methods["ordered_response_2"]["rms"],
                       classical_8_rms=methods["ordered_response_8"]["rms"],
                       fitted_over_classical_8_rms=fitted["rms"]/methods["ordered_response_8"]["rms"],
                       initialization_over_fitted_rms=methods["zero_readout"]["rms"]/fitted["rms"],
                       fitted_seconds=fitted["elapsed_seconds"],classical_8_seconds=methods["ordered_response_8"]["elapsed_seconds"],
                       fitted_over_classical_8_single_call_time=fitted["elapsed_seconds"]/methods["ordered_response_8"]["elapsed_seconds"])
            actions.append(out)
    loss = []
    # Equal per-parent mass and equal row counts in this specific retained bank.
    for split in ("train", "validation", "diagnostic"):
        for method in ("zero_readout", "fixed_feature_fitted"):
            rows = [r for r in metrics if r["split"]==split and r["method"]==method]
            parents = sorted({r["parent_id"] for r in rows})
            contributions = {}
            for kind in ("endpoint", "action"):
                contributions[kind] = sum(sum(r["rms"]**2 for r in rows if r["parent_id"]==p and r["kind"]==kind)/sum(r["parent_id"]==p for r in rows) for p in parents)/len(parents)
            total = sum(contributions.values())
            loss.append({"split":split,"method":method,"parent_balanced_mse":total,
                         "endpoint_contribution":contributions["endpoint"],"action_contribution":contributions["action"],
                         "endpoint_fraction":contributions["endpoint"]/total,"action_fraction":contributions["action"]/total})
    comparisons = []
    with zipfile.ZipFile(old_zip) as archive:
        old_arrays = np.load(io.BytesIO(archive.read("outputs/smoke/arrays.npz")),allow_pickle=False)
        old_selection = json.loads(archive.read("outputs/smoke/selection.json"))
        old_fits = json.loads(archive.read("outputs/smoke/fitting.json"))
        old_fit = next(r for r in old_fits if r["candidate"]==old_selection["selected_candidate"])
        comparisons.append(compare_arrays("native_2930_before_repair",arrays,old_arrays,weights,np.asarray(old_selection["coefficients"])))
    cloud_arrays=np.load(cloud/"arrays.npz",allow_pickle=False)
    cloud_selection=json.loads((cloud/"selection.json").read_text())
    comparisons.append(compare_arrays("revised_cloud_cpu",arrays,cloud_arrays,weights,np.asarray(cloud_selection["coefficients"])))
    derivatives = [{"horizon":r["horizon"],"whole_model_jvp_vs_flow":r["whole_model_jvp_vs_flow"],
                    "finite_difference_sweep":r["finite_difference_sweep"],"reference_status":r["derivative_reference_status"]} for r in read("derivatives.json")]
    audits=[{"horizon":r["horizon"],"method":r["method"],"scope":r["scope"],"status":r["audit"]["status"],
             "endpoint_error_bound_infinity":r["audit"]["endpoint_error_bound_infinity"],"certificate_proves":r["audit"]["certificate_proves"]} for r in read("audit.json")]
    status=json.loads((root/"outputs/native/status.json").read_text())
    telemetry=[json.loads(line) for line in (root/"outputs/native/telemetry/cpu-memory.jsonl").read_text().splitlines()]
    return {"schema":"tdn.native_preconditioner_descriptive_science_review.v1",
        "run_id":json.loads((root/"run.json").read_text())["run_id"],"job_id":status["allocation"]["job_id"],"scope":"retained native artifacts only; no new fitting, teacher acquisition or model inference",
        "summary":read("summary.json"),"source_parity":{"source_equals_after":read("source.json")==read("source_after.json"),
            "cloud_source_fingerprint":json.loads((cloud/"summary.json").read_text())["source_fingerprint"]},
        "selected_fit":{k:chosen["diagnostics"][k] for k in ("augmented_condition_number","original_design_numerical_rank","coefficients","objective","data_loss","ridge_loss","gradient_norm","optimality_gap_upper_bound","certificate","original_penalty_preserved")},
        "condition_before_repair":old_fit["diagnostics"]["augmented_condition_number"],
        "selected_validation_loss":chosen["validation_loss"],"initialization_validation_loss":fits[0]["validation_loss"],
        "endpoints":endpoint,"actions":actions,"task_loss_components":loss,"retained_array_comparisons":comparisons,
        "endpoint_work":[r for r in read("work.json") if r["kind"]=="endpoint"],"derivatives":derivatives,"audits":audits,
        "worker_telemetry":{"index_status":status["worker_telemetry"]["status"],"samples":len(telemetry),
            "sample_status_counts":dict(Counter(r["status"] for r in telemetry)),
            "reasons":dict(Counter(r["reason"] for r in telemetry if r["reason"])),
            "observed_rss_peak_bytes":max(r["rss_bytes"] for r in telemetry if r["rss_bytes"] is not None),
            "available_cpu_utilization_samples":sum(r["cpu_utilization_percent"] is not None for r in telemetry),
            "missing_cpu_utilization_samples":sum(r["cpu_utilization_percent"] is None for r in telemetry),
            "scope":"local worker process-tree snapshots; shared RSS can be counted repeatedly; incomplete interval CPU on process churn; not scheduler accounting"},
        "conclusions":{"scientific_promotion":False,"historical_gate":"B STOP unchanged","matched_accuracy_speedup":False,
            "native_numerical_repair_reproduced":True,"classical_eight_panel_wins_all_four_actions":all(r["fitted_over_classical_8_rms"]>1 for r in actions),
            "endpoint_learned_effect_resolved":False,"timing_status":"one call per method/query; descriptive only, no controlled speed inference",
            "next_eligible_work":"freeze bounded R7 identification protocol: nonzero source times, withheld forcing directions, five-coefficient per-state oracle, independently verified classical controls, reference-sized perturbation sensitivity, separate action/endpoint objectives, repeated full-call and amortized costs; no encoder or new numerical campaign before that freeze"}}


def compare_arrays(label, current, other, weights, other_weights):
    common=sorted(set(current.files)&set(other.files))
    deltas={k:float(np.max(np.abs(current[k].astype(float)-other[k].astype(float)))) for k in common}
    predictions=[]
    for name in common:
        if name.endswith("_design"):
            prefix=name[:-7]
            a=current[prefix+"_base"].reshape(-1)+weights@current[name].reshape(40,-1)
            b=other[prefix+"_base"].reshape(-1)+other_weights@other[name].reshape(40,-1)
            predictions.append(rms(a-b))
    groups={}
    for suffix in ("input_fp32","directions","teacher","base","design","matched_mean"):
        names=[k for k in common if k.endswith(suffix)]
        groups[suffix]={"count":len(names),"maximum_absolute_difference":max(deltas[k] for k in names),"all_bit_equal":all(np.array_equal(current[k],other[k]) for k in names)}
    return {"comparison":label,"common_array_count":len(common),"current_only":sorted(set(current.files)-set(other.files)),
        "other_only":sorted(set(other.files)-set(current.files)),"array_groups":groups,
        "coefficient_maximum_absolute_difference":float(np.max(np.abs(weights-other_weights))),
        "maximum_retained_prediction_difference_rms":max(predictions)}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--cloud-smoke",type=Path,required=True)
    parser.add_argument("--old-zip",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    out=summarize(args.run,args.cloud_smoke,args.old_zip)
    inputs=[*sorted((args.run/"outputs/smoke").glob("*.json")),args.run/"outputs/smoke/arrays.npz",
        args.run/"outputs/native/status.json",args.run/"outputs/native/telemetry/cpu-memory.jsonl",
        args.cloud_smoke/"selection.json",args.cloud_smoke/"summary.json",args.cloud_smoke/"arrays.npz",args.old_zip]
    out["input_sha256"]={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    with args.output.open("x") as handle:
        json.dump(out,handle,indent=2,allow_nan=False);handle.write("\n")
    print(args.output)
