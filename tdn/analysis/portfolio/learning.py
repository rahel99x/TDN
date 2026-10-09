"""Bounded attribution training, locked comparisons and exact paired aggregation.

One record is a field/target/seed/data-size/model, never a selected lucky seed.
Failures, initialization selections and unavailable teachers remain visible.
All endpoint schedule menus are shared by classical and learned methods.
"""
from __future__ import annotations
from collections import defaultdict
import copy
import itertools
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import time
import torch
from tdn.analysis.frontier.core import clean, check
from tdn.analysis.frontier.data import _safe_file
from tdn.analysis.frontier.neural import (nested_subset, eq_geom, reference, endpoint_errors,
    rollout, _samples, validation_metrics, TrialNumericalFailure, _sync)
from tdn.analysis.frontier.measurement import measure_paired, endpoint_eligibility
from tdn.research.experiment import atomic_torch_save
from tdn.research.protocol import digest, file_digest
from tdn.runtime.metadata import write_json
from .data import binding, unit, load_bank
from .statistics import field_cluster_interval


def _units(protocol):
    return list(protocol["units"].values()) if isinstance(protocol["units"], dict) else protocol["units"]


def model_specs(protocol, scope=None):
    """Every fitted arm has every declared seed and nested field subset."""
    from .models import MODEL_SPECS
    scope = scope or {}
    families = scope.get("families", protocol["models"])
    seeds = scope.get("seeds", protocol["seeds"])
    tracks = scope.get("tracks", protocol["tracks"])
    sizes = scope.get("subset_sizes", protocol["training"]["subset_sizes"])
    rows = []
    for family in families:
        fitted = MODEL_SPECS[family].get("fit", "gradient" if MODEL_SPECS[family].get("trainable") else "frozen") != "frozen"
        if not fitted and scope and protocol["seeds"][0] not in seeds:
            continue
        for track in tracks:
            for seed, size in itertools.product(seeds, sizes) if fitted else [(None, 0)]:
                rows.append(dict(model_id=f"{track}/{family}/n{size}/seed{seed}", family=family,
                                 track=track, seed=seed, train_count=size))
    return rows


def shared_schedules(protocol):
    """Deduplicate primary and cheap endpoint options without privileging a family."""
    primary = protocol.get("primary_schedules", protocol.get("confirm_schedules", protocol.get("schedules", [[.04]])))
    result = {}
    for index, raw in enumerate(primary):
        schedule = raw.get("schedule") if isinstance(raw, dict) else raw
        schedule = [float(x) for x in schedule]
        key = tuple(round(x,12) for x in schedule)
        result[key] = dict(schedule_id=f"primary-{index:02d}", schedule=schedule,
            final_time=round(math.fsum(schedule),12), primary=True)
    for final in sorted(set(protocol.get("evaluation_horizons", [])) | {x["final_time"] for x in result.values()}):
        for count in protocol.get("endpoint_steps", [1,2,4,8]):
            schedule = [final/count]*count; key = tuple(round(x,12) for x in schedule)
            result.setdefault(key, dict(schedule_id=f"endpoint-{final:.12g}-{count}", schedule=schedule,
                                       final_time=final, primary=False))
    return list(result.values())


def _config(protocol, family):
    return {**protocol.get("model_config", {}), **protocol.get("model_configs", {}).get(family,{})}


def _targets(protocol):
    targets = protocol.get("targets", [protocol.get("primary_target",2e-5)])
    if isinstance(targets, dict): targets = list(targets.values())
    return [(float(t.get("rms", t.get("rms_target"))), float(t.get("max", t.get("max_target"))))
            if isinstance(t,dict) else (float(t),float(t)) for t in targets]


def _record(ctx, identity, metrics, *, hypotheses=("A1",), good=None):
    prior_file=Path(ctx.path)/"rows.jsonl"
    if prior_file.exists():
        prior=[json.loads(line) for line in prior_file.read_text().splitlines() if line]
        matched=[r for r in prior if r["experiment_id"]==identity]
        if matched:
            if len(matched)!=1 or any(matched[0]["metrics"].get(k)!=metrics.get(k) for k in ("model_id","checkpoint_sha256","effective_config")):
                raise ValueError("Recovered experiment ledger identity differs")
            return
    ctx.record(identity, list(hypotheses), metrics=clean(metrics), checks=[
        check("finite-implementation", good, True, "eq", category="math"),
        check("fresh-comparative-benefit", None, True, "eq", category="gap",
              reason="Job completion and model fitting are not evidence of comparative scientific success")])


def _trial(ctx, spec, config, samples, validation, rate, trial_index, *, phase="tuning", control=None, updates=None):
    from .models import make_model, MODEL_SPECS
    settings = ctx.protocol["training"]
    control = control or {"id":"default", "clip_grad_norm":1., "loss_scale":1.}
    stem = digest([spec, config, rate, trial_index, phase, control, updates])
    complete = Path(ctx.path)/"trials"/(stem+".json")
    if complete.exists():
        record = json.loads(complete.read_text())
        if record["binding"] != binding(ctx): raise ValueError("Completed trial recovery identity differs")
        if file_digest(_safe_file(ctx.path,record["checkpoint"])) != record["checkpoint_sha256"]:
            raise ValueError("Completed trial checkpoint changed")
        return record
    torch.manual_seed(spec["seed"] or 0)
    model = make_model(spec["family"], spec["track"], config).to(device=ctx.device,dtype=torch.float32)
    params = [p for p in model.parameters() if p.requires_grad]
    fit = MODEL_SPECS[spec["family"]].get("fit", "gradient" if params else "frozen")
    initial_state = copy.deepcopy(model.state_dict()); selected_state = initial_state
    initial = best = None; selected_update = completed = examples = 0; failure = None
    start = time.perf_counter(); curves=[]; update_seconds=0.
    max_updates = int(updates if updates is not None else settings.get("updates", settings.get("max_updates",40)))
    max_seconds = float(settings.get("trial_seconds", 120))
    initial_path = Path(ctx.path)/"checkpoints"/(stem+".initial.pt")
    atomic_torch_save({"state_dict":initial_state, **spec, "config":config, "binding":binding(ctx)}, initial_path)
    def record_curve(update, loss=None, grad=None, val=None):
        curves.append({**spec,"trial_id":stem,"phase":phase,"update":update,"optimizer_control":control,
            "train_loss":loss,"gradient_norm":grad,"validation_loss":val["objective"] if val else None,
            "validation_rms":val["rms"] if val else None,"validation_max":val["maximum"] if val else None,
            "examples_seen":examples,"elapsed_seconds":time.perf_counter()-start,
            "optimizer_seconds":update_seconds,"learning_rate":rate,"device":str(ctx.device)})
    try:
        initial=best=validation_metrics(model, validation, ctx); record_curve(0,val=best)
        if fit in ("linear_lstsq","amplitude_lstsq"):
            ctx.budget.check()
            fit_info=model.fit_least_squares([(s["u"],s["h"],s["eq"],s["geom"],s["target"],1/s["scale"]) for s in samples],ridge=float(control.get("ridge",1e-8)),
                peak_weight=float(settings.get("peak_weight",.1)),budget=ctx.budget,max_iterations=int(settings.get("fit_iterations",100)))
            measured=validation_metrics(model,validation,ctx); completed=1; examples=len(samples)
            if measured["objective"] < best["objective"]:
                selected_state=copy.deepcopy(model.state_dict()); best=measured; selected_update=1
            record_curve(1,val=measured)
        else:
            fit_info=None
            optimizer=torch.optim.AdamW(params,lr=rate,weight_decay=float(settings.get("weight_decay",0.))) if params else None
            for update in range(1,max_updates+1 if params else 1):
                ctx.budget.check()
                if time.perf_counter()-start >= max_seconds: break
                rng=random.Random((spec["seed"] or 0)+update*104729)
                _sync(ctx.device); tick=time.perf_counter(); optimizer.zero_grad(set_to_none=True)
                losses=[]
                for _ in range(int(settings.get("batch_size",1))):
                    sample=samples[rng.randrange(len(samples))]
                    prediction=model(sample["u"],sample["h"],sample["eq"],sample["geom"])
                    squared=(prediction-sample["target"]).square()
                    loss=(squared.mean()+float(settings.get("peak_weight",.1))*squared.amax())/sample["scale"]
                    if not bool(torch.isfinite(loss)): raise TrialNumericalFailure("NONFINITE_TRAINING_LOSS")
                    (float(control.get("loss_scale",1.))*loss/int(settings.get("batch_size",1))).backward(); losses.append(float(loss.detach())); examples+=1
                clip=control.get("clip_grad_norm",settings.get("clip_grad_norm",1.))
                gradient=torch.nn.utils.clip_grad_norm_(params,float(clip)) if clip is not None else torch.sqrt(sum(p.grad.detach().square().sum() for p in params if p.grad is not None))
                if not bool(torch.isfinite(gradient)): raise TrialNumericalFailure("NONFINITE_GRADIENT")
                optimizer.step(); _sync(ctx.device); update_seconds+=time.perf_counter()-tick; completed=update
                measured=None
                if update%int(settings.get("validation_every",10))==0 or update==max_updates or time.perf_counter()-start>=max_seconds:
                    measured=validation_metrics(model,validation,ctx)
                    if measured["objective"]<best["objective"]:
                        best=measured; selected_state=copy.deepcopy(model.state_dict()); selected_update=update
                record_curve(update,statistics.mean(losses),float(gradient),measured)
    except (TrialNumericalFailure, FloatingPointError) as error:
        failure=str(error); fit_info=None
    except BaseException:
        # An incomplete attempt is forensic cost evidence, never a selected trial.
        attempt_dir=Path(ctx.path)/"interrupted-trials"; attempt_dir.mkdir(exist_ok=True)
        attempt=attempt_dir/(stem+f"-{time.time_ns()}.json")
        write_json(attempt,clean({**spec,"trial_id":stem,"seconds":time.perf_counter()-start,
                  "completed_updates":completed,"curves":curves,"status":"INTERRUPTED_NOT_SELECTED"}))
        raise
    model.load_state_dict(selected_state)
    checkpoint=Path(ctx.path)/"checkpoints"/(stem+".selected.pt")
    atomic_torch_save({"state_dict":selected_state,**spec,"config":config,"binding":binding(ctx)},checkpoint)
    selection=("FROZEN_CONTROL" if fit=="frozen" else "FITTED_CHECKPOINT" if selected_update else "SELECTED_INITIALIZATION")
    report=model.parameter_report() if hasattr(model,"parameter_report") else {}
    responses=[]
    if hasattr(model,"parameter_report"):
        with torch.no_grad():
            for split,probes in (("train",samples[:3]),("validation",validation[:3])):
                for sample in probes:
                    detail=model.parameter_report(sample["u"],sample["h"],sample["eq"],sample["geom"])
                    responses.append({"split":split,"parent_id":sample["parent_id"],"horizon":sample["h"],
                                      "response":detail.get("response"),"reference_truth_used":False})
    record={**spec,"trial_id":stem,"binding":binding(ctx),"effective_config":config,"learning_rate":rate,
        "fit":fit,"phase":phase,"optimizer_control":control,"fit_diagnostics":clean(fit_info),
        "fit_convergence_status":("CONVERGED" if fit_info.get("optimizer_success") else "CONVERGENCE_UNRESOLVED") if isinstance(fit_info,dict) and "optimizer_success" in fit_info else "NOT_APPLICABLE","selection_status":selection,"failure":failure,
        "checkpoint_validated":best is not None,"initial_validation_objective":initial["objective"] if initial else None,
        "validation_objective":best["objective"] if best else None,"updates_completed":completed,
        "updates_selected":selected_update,"updates_requested":max_updates if params else 0,
        "examples_seen":examples,"training_seconds":time.perf_counter()-start,"optimizer_seconds":update_seconds,
        "equal_compute_cap_seconds":max_seconds,"budget_view":"shared per-trial walltime AND update caps; actual work retained",
        "parameters":sum(p.numel() for p in model.parameters())+(4 if spec["family"]=="quad2_linear" else 0),
        "stored_tensor_parameters":sum(p.numel() for p in model.parameters()),
        "fitted_coefficients":(4 if spec["family"]=="quad2_linear" else 1 if spec["family"]=="quad2_amplitude" else sum(p.numel() for p in params)),
        "parameter_count_scope":"stored tensor parameters plus fitted coefficient buffers; fixed quadrature constants excluded",
        "trainable_parameters":sum(p.numel() for p in params),
        "parameter_report":clean(report),"response_probes":clean(responses),"checkpoint":str(checkpoint.relative_to(ctx.path)),
        "checkpoint_sha256":file_digest(checkpoint),"initial_checkpoint":str(initial_path.relative_to(ctx.path)),
        "initial_checkpoint_sha256":file_digest(initial_path),"curves":curves}
    write_json(complete,clean(record)); return record


def _load_model(ctx, row, base):
    from .models import make_model
    path=_safe_file(base,row["checkpoint"])
    if file_digest(path)!=row["checkpoint_sha256"]: raise ValueError("Checkpoint digest changed")
    payload=torch.load(path,map_location="cpu",weights_only=True)
    if payload.get("binding")!=binding(ctx) or payload.get("model_id")!=row["model_id"]:
        raise ValueError("Checkpoint identity changed")
    model=make_model(row["family"],row["track"],row["effective_config"]).to(device=ctx.device,dtype=torch.float32)
    model.load_state_dict(payload["state_dict"],strict=True); model.eval(); return model


def _validation_frontier(ctx, row, validation):
    model=_load_model(ctx,row,ctx.path); rows=[]; n=ctx.protocol["train_grid"]
    batches=defaultdict(list)
    for parent in validation:
        batches[(parent["kappa"],parent["reaction_rate"],tuple(parent.get("lengths",(1.,1.))))].append(parent)
    for parents in batches.values():
        eq,geom=eq_geom(parents[0],n)
        initial=torch.cat([p["states"][str(n)] for p in parents]).to(device=ctx.device,dtype=torch.float32)
        for item in shared_schedules(ctx.protocol):
            with torch.no_grad():
                value,cost=ctx.measure(lambda:rollout(model,initial,item["schedule"],eq,geom,ctx.budget)[0],repeats=2,warmup=1)
            for index,parent in enumerate(parents):
                ref=reference(parent,n,item["final_time"],row["track"])
                rows.append({**{k:row[k] for k in ("model_id","family","track","seed","train_count")},
                    "parent_id":parent["parent_id"],"field_cluster":parent["field_cluster"],"grid":n,**item,
                    **endpoint_errors(value[index:index+1],ref),"cost_seconds":cost["median_seconds"]/len(parents),
                    "raw_timing":cost,"timing_batch_size":len(parents),
                    "timing_scope":"amortized per-field cost in matching-physics distinct-field validation batch; fresh endpoint timing is separately measured"})
    return rows


def train(ctx):
    from .models import make_model, MODEL_SPECS
    bank=load_bank(ctx,"prepare"); training=[p for p in bank if p["split"]=="train"]
    validation=[p for p in bank if p["split"]=="validation"]
    if {p["field_cluster"] for p in training}&{p["field_cluster"] for p in validation}: raise ValueError("Train/validation leakage")
    specs=model_specs(ctx.protocol,unit(ctx)); root=Path(ctx.path); root.mkdir(parents=True,exist_ok=True)
    plan={**binding(ctx),"stage":ctx.stage,"expected_models":specs,"selection_split":"validation",
          "training":ctx.protocol["training"],"shared_schedules":shared_schedules(ctx.protocol)}
    path=root/"training_plan.json"
    if path.exists() and json.loads(path.read_text())!=plan: raise ValueError("Training recovery plan differs")
    write_json(path,plan); records=[]; curves=[]; all_validation=[]
    settings=ctx.protocol["training"]
    for spec in specs:
        ctx.budget.check(); cachefile=root/"completed-models"/(digest(spec)+".json")
        if cachefile.exists():
            saved=json.loads(cachefile.read_text()); selected=saved["record"]
            if saved["binding"]!=binding(ctx): raise ValueError("Model recovery binding differs")
            _load_model(ctx,selected,root)
            records.append(selected); curves.extend(saved["curves"]); all_validation.extend(saved["validation_rows"])
            _record(ctx,"training/"+spec["model_id"],selected,good=selected["checkpoint_validated"])
            continue
        config=_config(ctx.protocol,spec["family"]); base=make_model("df",spec["track"],config).to(device=ctx.device,dtype=torch.float32)
        subset=nested_subset(training,spec["train_count"]) if spec["train_count"] else training
        samples=_samples(subset,ctx.protocol["train_horizons"],ctx.protocol["train_grid"],spec["track"],ctx.device,base,
                         float(settings.get("normalization_floor",1e-6)))
        val_samples=_samples(validation,ctx.protocol["validation_horizons"],ctx.protocol["train_grid"],spec["track"],ctx.device,base,
                         float(settings.get("normalization_floor",1e-6)))
        fit=MODEL_SPECS[spec["family"]].get("fit","gradient" if MODEL_SPECS[spec["family"]].get("trainable") else "frozen")
        rates=settings.get("learning_rates",[1e-3]) if fit=="gradient" else [0.]
        # Direct and hybrid FNO share a declared scale search, never confirmation tuning.
        configs=[config]
        if "fno" in spec["family"]:
            configs=[{**config,"t_ref":float(x)} for x in settings.get("fno_t_refs",[config.get("t_ref",.1)])]
        controls=settings.get("optimizer_controls",[{"id":"default","clip_grad_norm":1.,"loss_scale":1.}]) if "fno" in spec["family"] else [{"id":"default","clip_grad_norm":1.,"loss_scale":1.}]
        if fit in ("linear_lstsq","amplitude_lstsq"):
            controls=[{"id":"ridge-fit","ridge":ridge} for ridge in settings.get("ridge_candidates",[1e-8])]
        trial_specs=list(itertools.product(configs,rates,controls))
        trials=[_trial(ctx,spec,c,samples,val_samples,float(rate),i,control=control,
                      updates=settings.get("tuning_updates",settings.get("updates",40)))
                for i,(c,rate,control) in enumerate(trial_specs)]
        valid=[t for t in trials if t["checkpoint_validated"] and not t["failure"]]
        winner=min(valid,key=lambda t:t["validation_objective"]) if valid else trials[0]
        if fit=="gradient" and valid:
            final=_trial(ctx,spec,winner["effective_config"],samples,val_samples,winner["learning_rate"],len(trials),
                         phase="final",control=winner["optimizer_control"],updates=settings.get("updates",40))
            trials.append(final)
            # Retain a better tuning checkpoint; final training is not required to improve it.
            selected=min([t for t in (winner,final) if t["checkpoint_validated"]],key=lambda t:t["validation_objective"])
        else: selected=winner
        selected={k:v for k,v in selected.items() if k!="curves"}
        selected["trials"]=[{k:v for k,v in t.items() if k!="curves"} for t in trials]
        selected["total_training_seconds"]=sum(t["training_seconds"] for t in trials)
        model_curves=[r for t in trials for r in t["curves"]]
        val_rows=_validation_frontier(ctx,selected,validation) if selected["checkpoint_validated"] else []
        saved={"binding":binding(ctx),"record":selected,"curves":model_curves,"validation_rows":val_rows}
        write_json(cachefile,clean(saved)); records.append(selected); curves.extend(model_curves); all_validation.extend(val_rows)
        _record(ctx,"training/"+spec["model_id"],selected,good=selected["checkpoint_validated"])
    write_json(root/"catalog.json",{"schema":"tdn.portfolio-catalog/v1",**binding(ctx),"stage":ctx.stage,"records":records,
               "selection_split":"validation","training_plan_sha256":file_digest(root/"training_plan.json")})
    write_json(root/"validation_rows.json",{"rows":all_validation})
    with (root/"learning_curves.jsonl").open("w") as handle:
        for row in curves: handle.write(json.dumps(clean(row),allow_nan=False)+"\n")
    return {"models":len(records),"trials":sum(len(r["trials"]) for r in records),"training_seconds":sum(r["total_training_seconds"] for r in records),
            "initialization_selections":sum(r["selection_status"]=="SELECTED_INITIALIZATION" for r in records)}


def validation_selection(rows, targets):
    """Freeze one affordable feasible schedule per model/endpoint/target.

    Every validation field must pass both error bounds; the selected schedule
    transfers unchanged to confirmation grids, without peeking at their truth.
    """
    grouped=defaultdict(list)
    for row in rows: grouped[(row["model_id"],row["track"],row["final_time"])].append(row)
    selected=[]
    for (model_id,track,final),members in sorted(grouped.items()):
        fields={r["field_cluster"] for r in members}
        for rt,mt in targets:
            options=[]
            for schedule in sorted({r["schedule_id"] for r in members}):
                cells=[r for r in members if r["schedule_id"]==schedule]
                feasible={r["field_cluster"] for r in cells if endpoint_eligibility(r,rt,mt)=="ELIGIBLE"}
                if feasible==fields and len(cells)==len(fields):
                    options.append((statistics.mean(r["cost_seconds"] for r in cells),schedule))
            best=min(options) if options else None
            selected.append(dict(model_id=model_id,track=track,final_time=final,rms_target=rt,max_target=mt,
                schedule_id=best[1] if best else None,validation_cost_seconds=best[0] if best else None,
                status="VALIDATION_SELECTED" if best else "NO_FEASIBLE_VALIDATION_SCHEDULE",
                independent_validation_fields=len(fields),selection_split="validation",
                transfer="same schedule to every confirmation grid; no confirmation-dependent selection"))
    return selected


def freeze(ctx):
    records=[]; validation=[]; sources={}; root=Path(ctx.path); root.mkdir(parents=True,exist_ok=True)
    expected=model_specs(ctx.protocol)
    expected_ids={r["model_id"] for r in expected}
    for scope in _units(ctx.protocol):
        if not scope["id"].startswith("train-"): continue
        base=Path(ctx.prerequisites[scope["id"]]); catalog_path=_safe_file(base,"catalog.json")
        catalog=json.loads(catalog_path.read_text())
        if any(catalog.get(k)!=v for k,v in binding(ctx).items()): raise ValueError("Training catalog identity differs")
        if catalog.get("selection_split")!="validation": raise ValueError("Only validation-selected models can freeze")
        planned=model_specs(ctx.protocol,scope)
        ids=[r["model_id"] for r in catalog["records"]]
        if len(ids)!=len(set(ids)) or set(ids)!={r["model_id"] for r in planned}:
            raise ValueError("Training unit model coverage is incomplete or duplicate")
        if file_digest(_safe_file(base,"training_plan.json"))!=catalog["training_plan_sha256"]:
            raise ValueError("Training plan bytes changed")
        sources[scope["id"]]={"catalog_sha256":file_digest(catalog_path),
            "science_manifest_sha256":file_digest(base/"science_manifest.json") if (base/"science_manifest.json").exists() else None}
        for row in catalog["records"]:
            checkpoint=_safe_file(base,row["checkpoint"])
            if file_digest(checkpoint)!=row["checkpoint_sha256"]: raise ValueError("Selected checkpoint changed")
            destination=root/"checkpoints"/(digest(row["model_id"])+".pt"); destination.parent.mkdir(exist_ok=True)
            if destination.exists() and file_digest(destination)!=row["checkpoint_sha256"]:
                raise ValueError("Frozen destination changed")
            if not destination.exists(): shutil.copyfile(checkpoint,destination)
            records.append({**row,"checkpoint":str(destination.relative_to(root)),"training_unit":scope["id"]})
        validation.extend(json.loads(_safe_file(base,"validation_rows.json").read_text())["rows"])
    ids=[r["model_id"] for r in records]
    if len(ids)!=len(set(ids)) or set(ids)!=expected_ids: raise ValueError("Frozen model coverage differs from complete declaration")
    catalog={"schema":"tdn.portfolio-catalog/v1",**binding(ctx),"records":records,
             "selection_split":"validation","frozen_before_confirmation":True,"sources":sources}
    write_json(root/"catalog.json",catalog)
    frontier=validation_selection(validation,_targets(ctx.protocol))
    write_json(root/"validation_frontier.json",{"rows":frontier,"raw_validation_rows":validation,
        "selection_split":"validation","grid_transfer":"same selected schedule to all confirmation grids"})
    identity={**binding(ctx),"catalog_sha256":file_digest(root/"catalog.json"),
        "validation_frontier_sha256":file_digest(root/"validation_frontier.json"),
        "checkpoint_hashes":{r["model_id"]:r["checkpoint_sha256"] for r in records}}
    write_json(root/"freeze.json",identity)
    return {"frozen_models":len(records),"validation_frontier_cells":len(frontier),"fresh_confirmation_accessed":False}


def verify_frozen(ctx):
    root=Path(ctx.path if ctx.stage=="freeze" else ctx.prerequisites["freeze"])
    frozen=json.loads(_safe_file(root,"freeze.json").read_text())
    catalog=json.loads(_safe_file(root,"catalog.json").read_text())
    if any(frozen.get(k)!=v or catalog.get(k)!=v for k,v in binding(ctx).items()): raise ValueError("Frozen source/protocol changed")
    if frozen["catalog_sha256"]!=file_digest(root/"catalog.json") or frozen["validation_frontier_sha256"]!=file_digest(_safe_file(root,"validation_frontier.json")):
        raise ValueError("Frozen selection bytes changed")
    ids=[r["model_id"] for r in catalog["records"]]
    if len(ids)!=len(set(ids)) or set(ids)!={r["model_id"] for r in model_specs(ctx.protocol)}:
        raise ValueError("Frozen roster incomplete or duplicated")
    if set(frozen["checkpoint_hashes"])!=set(ids) or catalog.get("selection_split")!="validation" or catalog.get("frozen_before_confirmation") is not True:
        raise ValueError("Model selection was not frozen before confirmation")
    for row in catalog["records"]:
        if frozen["checkpoint_hashes"][row["model_id"]]!=row["checkpoint_sha256"] or file_digest(_safe_file(root,row["checkpoint"]))!=row["checkpoint_sha256"]:
            raise ValueError("Frozen model bytes changed")
    return catalog,frozen


def _group_key(parent,n,track,schedule):
    return f"{parent}/N{n}/{track}/{schedule}"


def expected_confirmation(protocol, parent_ids, catalog):
    result={}
    for parent in protocol["parents"]:
        if parent["parent_id"] not in parent_ids: continue
        for n,track,item in itertools.product(protocol["grids"],protocol["tracks"],shared_schedules(protocol)):
            group_id=_group_key(parent["parent_id"],n,track,item["schedule_id"])
            result[group_id]={"parent":parent,"grid":n,"track":track,"schedule":item,
                "model_ids":[r["model_id"] for r in catalog["records"] if r["track"]==track]}
    return result


def _validate_group(payload, expected, identity, protocol):
    if payload.get("binding")!=identity: raise ValueError("Confirmation group belongs to different frozen inputs")
    rows=payload["rows"]; ids=[r["model_id"] for r in rows]
    if len(ids)!=len(set(ids)) or set(ids)!=set(expected["model_ids"]): raise ValueError("Confirmation group model coverage incomplete or duplicate")
    item=expected["schedule"]; group=payload["timing"]
    valid={r["model_id"] for r in rows if r["status"]!="UNAVAILABLE_CHECKPOINT"}
    if set(group.get("methods",{}))!=valid: raise ValueError("Paired timing group model coverage differs")
    if valid:
        repeats=int(protocol.get("timing",{}).get("repeats",protocol.get("timing_repeats",3)))
        if len(group["rounds"])!=repeats: raise ValueError("Paired timing round coverage differs")
        for round_ in group["rounds"]:
            if set(round_["order"])!=valid or len(round_["order"])!=len(valid) or set(round_["samples_seconds"])!=valid:
                raise ValueError("Paired round is incomplete or duplicate")
            if any(not math.isfinite(x) or x<=0 for x in round_["samples_seconds"].values()): raise ValueError("Invalid paired timer")
    for row in rows:
        required={"parent_id":expected["parent"]["parent_id"],"field_cluster":expected["parent"]["field_cluster"],
                  "grid":expected["grid"],"track":expected["track"],**item}
        if any(row.get(k)!=v for k,v in required.items()): raise ValueError("Confirmation workload identity differs")
        if row["model_id"] in valid:
            measured=group["methods"][row["model_id"]]
            values=[r["samples_seconds"][row["model_id"]] for r in group["rounds"]]
            if measured["samples_seconds"]!=values or measured["median_seconds"]!=statistics.median(values) or row["cost_seconds"]!=measured["median_seconds"]:
                raise ValueError("Endpoint cost not traceable to raw paired samples")
            times=[round(math.fsum(item["schedule"][:i]),12) for i in range(1,len(item["schedule"]))] if item["primary"] and row["finite"] else []
            if [r["time"] for r in row["intermediates"]]!=times: raise ValueError("Missing declared rollout intermediate audits")
    return True


def confirm(ctx):
    catalog,frozen=verify_frozen(ctx); root=Path(ctx.path); root.mkdir(parents=True,exist_ok=True)
    scope=unit(ctx); prepare_stage=next(k for k in ctx.prerequisites if k.startswith("confirm-prepare-"))
    parents=load_bank(ctx,prepare_stage)
    if {p["parent_id"] for p in parents}!=set(scope["parent_ids"]): raise ValueError("Confirmation partition parent scope differs")
    identity={**binding(ctx),"frozen":frozen,"data_manifest_sha256":file_digest(Path(ctx.prerequisites[prepare_stage])/"data_manifest.json"),
              "stage":ctx.stage,"device":str(ctx.device)}
    expected=expected_confirmation(ctx.protocol,scope["parent_ids"],catalog)
    models={row["model_id"]:_load_model(ctx,row,ctx.prerequisites["freeze"]) for row in catalog["records"] if row["checkpoint_validated"]}
    all_rows=[]; timings=[]; journal_path=root/"confirmation-journal.json"
    journal=json.loads(journal_path.read_text()) if journal_path.exists() else {"binding":identity,"groups":{}}
    if journal["binding"]!=identity: raise ValueError("Confirmation resume source or input changed")
    for parent in parents:
        for n,track,item in itertools.product(ctx.protocol["grids"],ctx.protocol["tracks"],shared_schedules(ctx.protocol)):
            ctx.budget.check(); group_id=_group_key(parent["parent_id"],n,track,item["schedule_id"])
            group_path=root/"completed-groups"/(digest(group_id)+".json")
            if group_id in journal["groups"]:
                if file_digest(_safe_file(root,str(group_path.relative_to(root))))!=journal["groups"][group_id]: raise ValueError("Committed timing group changed")
                payload=json.loads(group_path.read_text()); _validate_group(payload,expected[group_id],identity,ctx.protocol)
                all_rows.extend(payload["rows"]);timings.append(payload["timing"]);continue
            eq,geom=eq_geom(parent,n); initial=parent["states"][str(n)].to(ctx.device,dtype=torch.float32)
            roster=[r for r in catalog["records"] if r["track"]==track]
            calls={r["model_id"]:(lambda m=models[r["model_id"]]:rollout(m,initial,item["schedule"],eq,geom,ctx.budget)[0])
                   for r in roster if r["model_id"] in models}
            options=ctx.protocol.get("timing",{})
            with torch.no_grad():
                answers,timing=measure_paired(calls,device=ctx.device,repeats=int(options.get("repeats",ctx.protocol.get("timing_repeats",3))),
                    warmup=int(options.get("warmup",1)),seed=int(digest(group_id)[:8],16),budget=ctx.budget) if calls else ({},{"methods":{},"rounds":[]})
            timing.update(timing_id=group_id,parent_id=parent["parent_id"],grid=n,track=track,**item)
            ref=reference(parent,n,item["final_time"],track); rows=[]
            for record in roster:
                base={k:record[k] for k in ("model_id","family","track","seed","train_count","parameters")}
                base.update(parent_id=parent["parent_id"],field_cluster=parent["field_cluster"],regime=parent.get("regime"),
                    distribution=parent.get("distribution"),grid=n,**item,timing_id=group_id,
                    reference_uncertainty_rms=ref.get("uncertainty_rms"),reference_uncertainty_max=ref.get("uncertainty_max_bound"),
                    reference_accepted=ref["accepted"],intermediates=[])
                if record["model_id"] not in answers:
                    rows.append({**base,"status":"UNAVAILABLE_CHECKPOINT","finite":False,"cost_seconds":None});continue
                answer=answers[record["model_id"]]; error=endpoint_errors(answer,ref); cost=timing["methods"][record["model_id"]]
                base.update(error,cost_seconds=cost["median_seconds"],cold_seconds=cost["cold_seconds"],timing=cost,
                            status="COMPLETED" if error["finite"] else "NUMERICAL_FAILURE")
                if item["primary"] and error["finite"]:
                    with torch.no_grad():
                        _,middle=rollout(models[record["model_id"]],initial,item["schedule"],eq,geom,ctx.budget,capture_intermediates=True)
                    base["intermediates"]=[{"time":round(math.fsum(item["schedule"][:i+1]),12),
                        **endpoint_errors(value,reference(parent,n,round(math.fsum(item["schedule"][:i+1]),12),track))}
                        for i,value in enumerate(middle[:-1])]
                rows.append(base)
            payload={"binding":identity,"rows":clean(rows),"timing":clean(timing)}
            _validate_group(payload,expected[group_id],identity,ctx.protocol)
            write_json(group_path,payload); journal["groups"][group_id]=file_digest(group_path);write_json(journal_path,journal)
            all_rows.extend(rows);timings.append(timing)
    if set(journal["groups"])!=set(expected): raise ValueError("Confirmation group inventory incomplete")
    write_json(root/"endpoint_rows.json",{"rows":clean(all_rows),"independence_unit":"field_cluster"})
    write_json(root/"timing_rounds.json",{"groups":clean(timings)})
    write_json(root/"coverage.json",{**identity,"parent_ids":scope["parent_ids"],"model_ids":[r["model_id"] for r in catalog["records"]],
              "group_hashes":journal["groups"],"endpoint_rows":len(all_rows),"status":"PARTITION_ONLY"})
    return {"status":"PARTITION_ONLY","parents":len(parents),"endpoint_rows":len(all_rows),"timing_groups":len(timings),
            "cohort_scientific_outcome":"DEFERRED_UNTIL_COMPLETE_AGGREGATE"}


def _pairable(a,b):
    if a["family"]==b["family"]: return False
    # No best-seed selection. Deterministic controls are paired to each fitted
    # instance; fitted methods must have identical seed and field count.
    return a["seed"] is None or b["seed"] is None or (a["seed"],a["train_count"])==(b["seed"],b["train_count"])


def comparison_tables(rows, frozen_selection, protocol):
    cells=defaultdict(list)
    for row in rows: cells[(row["parent_id"],row["grid"],row["track"],row["schedule_id"])].append(row)
    pairs=defaultdict(list); failure_counts=defaultdict(lambda:{"paired_rows":0,"unusable_rows":0})
    for members in cells.values():
        for a,b in itertools.combinations(sorted(members,key=lambda r:r["family"]),2):
            if not _pairable(a,b): continue
            key=(a["family"],b["family"],a["track"],max(a["train_count"],b["train_count"]),a["final_time"],a.get("regime","unspecified"))
            failure_counts[key]["paired_rows"]+=1
            eligible=(a.get("reference_accepted") and b.get("reference_accepted") and a.get("error_rms") is not None
                      and b.get("error_rms") is not None and a.get("cost_seconds") and b.get("cost_seconds"))
            if not eligible: failure_counts[key]["unusable_rows"]+=1;continue
            pairs[key].append({"field_cluster":a["field_cluster"],"seed":a["seed"] if a["seed"] is not None else b["seed"],
                "difference":math.log(max(b["error_rms"],1e-30)/max(a["error_rms"],1e-30)),
                "speed_log_ratio":math.log(b["cost_seconds"]/a["cost_seconds"]),"regime":a.get("regime"),
                "grid":a["grid"],"schedule_id":a["schedule_id"]})
    summaries=[]
    for key,denominators in sorted(failure_counts.items()):
        values=pairs[key]; uncertainty=field_cluster_interval(values,repeats=int(protocol.get("bootstrap_replicates",100)))
        speed=field_cluster_interval([{**r,"difference":r["speed_log_ratio"]} for r in values],
                                      repeats=int(protocol.get("bootstrap_replicates",100)))
        summaries.append(dict(candidate_family=key[0],control_family=key[1],track=key[2],train_count=key[3],final_time=key[4],regime=key[5],
            **denominators,error_log_ratio_interval=uncertainty,speed_log_ratio_interval=speed,
            geometric_control_over_candidate_rms=math.exp(uncertainty["mean"]) if uncertainty.get("mean") is not None else None,
            geometric_control_over_candidate_cost=math.exp(speed["mean"]) if speed.get("mean") is not None else None,
            scope="same schedule/grid/field/target; repeated queries collapsed by field and paired training seed",
            interval_is_observed_range=False,paper_reproduction=False,
            small_regime_sample=uncertainty["independent_fields"]<5,
            analysis_status="exploratory all-pair subgroup comparison; intervals unadjusted for multiplicity"))
    frontiers=[]; locked=[]; groups=defaultdict(list)
    for row in rows: groups[(row["model_id"],row["parent_id"],row["grid"],row["track"],row["final_time"])].append(row)
    selection={(r["model_id"],r["track"],r["final_time"],r["rms_target"],r["max_target"]):r for r in frozen_selection}
    for (_,_,_,track,final),members in groups.items():
        first=members[0]
        for rt,mt in _targets(protocol):
            base={k:first[k] for k in ("model_id","family","seed","train_count","parent_id","field_cluster","grid","track","final_time","regime")}
            base.update(rms_target=rt,max_target=mt)
            feasible=[r for r in members if endpoint_eligibility(r,rt,mt)=="ELIGIBLE"]
            best=min(feasible,key=lambda r:r["cost_seconds"]) if feasible else None
            frontiers.append({**base,"schedule_id":best["schedule_id"] if best else None,
                "cost_seconds":best["cost_seconds"] if best else None,"status":"FEASIBLE" if best else "NO_FEASIBLE_SCHEDULE",
                "scope":"reference-informed posthoc frontier; not deployment selection"})
            decision=selection.get((first["model_id"],track,final,rt,mt))
            chosen=next((r for r in members if decision and r["schedule_id"]==decision["schedule_id"]),None)
            locked.append({**base,"schedule_id":chosen["schedule_id"] if chosen else None,
                "cost_seconds":chosen.get("cost_seconds") if chosen else None,
                "status":endpoint_eligibility(chosen,rt,mt) if chosen else "NO_VALIDATION_SELECTED_SCHEDULE",
                "upper_rms":chosen.get("upper_rms") if chosen else None,"upper_max":chosen.get("upper_max") if chosen else None,
                "scope":"validation-selected schedule fixed before fresh confirmation"})
    return {"matched_configuration":summaries,"paired_field_summaries":summaries,
            "posthoc_frontiers":frontiers,"locked_frontiers":locked,
            "source":"endpoint_rows.json and frozen validation_frontier.json",
            "independence_unit":"field_cluster; paired seeds crossed, repeated schedules/grids averaged within field-seed",
            "paper_reproduction":False}


def aggregate(ctx):
    catalog,frozen=verify_frozen(ctx); rows=[]; timings=[]; seen_parents=set(); source_parts={}
    declared=[u for u in _units(ctx.protocol) if u["id"].startswith("confirm-") and not u["id"].startswith("confirm-prepare-")]
    for part in declared:
        base=Path(ctx.prerequisites[part["id"]]); coverage=json.loads(_safe_file(base,"coverage.json").read_text())
        if any(coverage.get(k)!=v for k,v in binding(ctx).items()) or coverage.get("frozen")!=frozen or coverage.get("stage")!=part["id"]:
            raise ValueError("Confirmation part source/frozen identity differs")
        if coverage["parent_ids"]!=part["parent_ids"] or seen_parents&set(part["parent_ids"]):
            raise ValueError("Confirmation parent partition incomplete or overlaps")
        seen_parents.update(part["parent_ids"])
        expected=expected_confirmation(ctx.protocol,part["parent_ids"],catalog)
        if set(coverage["group_hashes"])!=set(expected): raise ValueError("Confirmation part omits declared groups")
        part_rows=[]; part_timings=[]
        identity={k:coverage[k] for k in ("protocol_sha256","source_tree_sha256","software_sha256","frozen","data_manifest_sha256","stage","device")}
        for group_id,sha in coverage["group_hashes"].items():
            group_path=_safe_file(base,"completed-groups/"+digest(group_id)+".json")
            if file_digest(group_path)!=sha: raise ValueError("Confirmation sealed group bytes changed")
            payload=json.loads(group_path.read_text()); _validate_group(payload,expected[group_id],identity,ctx.protocol)
            part_rows.extend(payload["rows"]);part_timings.append(payload["timing"])
        saved_rows=json.loads(_safe_file(base,"endpoint_rows.json").read_text())["rows"]
        saved_timings=json.loads(_safe_file(base,"timing_rounds.json").read_text())["groups"]
        if sorted(saved_rows,key=lambda r:(r["timing_id"],r["model_id"]))!=sorted(part_rows,key=lambda r:(r["timing_id"],r["model_id"])) or sorted(saved_timings,key=lambda r:r["timing_id"])!=sorted(part_timings,key=lambda r:r["timing_id"]):
            raise ValueError("Confirmation projection differs from committed paired groups")
        if coverage["endpoint_rows"]!=len(part_rows): raise ValueError("Confirmation row count differs")
        rows.extend(part_rows);timings.extend(part_timings)
        source_parts[part["id"]]={"coverage_sha256":file_digest(base/"coverage.json"),
            "science_manifest_sha256":file_digest(base/"science_manifest.json") if (base/"science_manifest.json").exists() else None}
    wanted={p["parent_id"] for p in ctx.protocol["parents"] if p["split"]=="confirmation"}
    if seen_parents!=wanted: raise ValueError("Whole-cohort confirmation parent coverage is incomplete")
    devices={r.get("device") for r in timings}
    if len(devices)>1: raise ValueError("Cannot mix confirmation measurement devices")
    selection=json.loads((Path(ctx.prerequisites["freeze"])/"validation_frontier.json").read_text())["rows"]
    tables=comparison_tables(rows,selection,ctx.protocol)
    write_json(ctx.path/"endpoint_rows.json",{"rows":rows,"independence_unit":"field_cluster"})
    write_json(ctx.path/"timing_rounds.json",{"groups":timings})
    write_json(ctx.path/"comparisons.json",clean(tables))
    claims=primary_claims(rows,tables,ctx.protocol)
    write_json(ctx.path/"claims.json",clean(claims))
    for index,claim in enumerate(claims["accuracy"]+claims["cost"]):
        cost_claim=claim["claim"]=="validation_locked_solver_cost"
        effect=claim.get("geometric_control_over_candidate_cost" if cost_claim else "geometric_control_over_candidate_rms")
        required=claim.get("required_speedup" if cost_claim else "required_ratio")
        no_harm=(claim.get("candidate_joint_field_coverage",0)-claim.get("control_joint_field_coverage",0)
                 if cost_claim and claim.get("candidate_joint_field_coverage") is not None else claim.get("coverage_difference"))
        decidable=claim["verdict"]!="NA"
        enough_fields=claim["interval"]["independent_fields"]>=int(ctx.protocol["hypothesis_thresholds"].get("minimum_fields",5))
        checks=[check("effect-size",effect,required,"ge",category="gap",applicable=decidable),
                check("paired-interval-excludes-no-gain",claim.get("lower_ratio"),1.,"gt",category="gap",applicable=decidable),
                check("joint-coverage-no-harm",no_harm,-float(ctx.protocol["hypothesis_thresholds"].get("maximum_coverage_regression",.05)),"ge",category="gap",applicable=decidable),
                check("field-cluster-inference-ready",True if enough_fields else None,True,"eq",category="math",
                      reason="Insufficient independent fields means unavailable inference, not incorrect mathematics"),
                check("declared-complete-decision",None if not decidable else claim["verdict"]=="GOOD",True,"eq",category="gap")]
        if cost_claim:checks.append(check("joint-field-feasibility",claim.get("candidate_joint_field_coverage"),claim["required_coverage"],"ge",category="gap",applicable=decidable))
        if f"claim/{index:04d}" in getattr(ctx,"identities",set()): continue
        ctx.record(f"claim/{index:04d}",["B4" if cost_claim else "A5" if claim["claim"]=="bounded_neural_accuracy" else "A2"],metrics=clean(claim),checks=checks)

    write_json(ctx.path/"coverage.json",{**binding(ctx),"frozen":frozen,"source_parts":source_parts,
        "parents":sorted(seen_parents),"model_ids":[r["model_id"] for r in catalog["records"]],"status":"VERIFIED_COMPLETE_COHORT"})
    _record(ctx,"aggregate/exact-coverage",{"parents":len(seen_parents),"endpoints":len(rows)},good=True)
    return {"parents":len(seen_parents),"models":len(catalog["records"]),"endpoint_rows":len(rows),
            "timing_groups":len(timings),"scientific_outcome":"DESCRIPTIVE_COMPLETE_COHORT",
            "comparisons":len(tables["matched_configuration"]),"paper_reproduction":False}


def scaling(ctx):
    """Qualified small scaling panel; no extrapolated large-grid speed claims."""
    from tdn.analysis.frontier.data import generate_reference, field_state, geometry
    from tdn.numerics import Equation
    catalog,frozen=verify_frozen(ctx); options=ctx.protocol.get("scaling",{})
    families=options.get("families",["quad2_conditioned","quad2_fixed","fno_small","fno_standard","df","etdrk4"])
    seeds=ctx.protocol["seeds"]; largest=max(ctx.protocol["training"]["subset_sizes"])
    selected=[r for r in catalog["records"] if r["family"] in families and
              (r["seed"] is None or (r["seed"]==seeds[0] and r["train_count"]==largest))]
    parents=[p for p in ctx.protocol["parents"] if p["split"]=="scaling"]
    if not parents:
        write_json(ctx.path/"scaling_rows.json",{"rows":[],"status":"NA","reason":"No frozen fresh scaling parents declared"})
        return {"status":"NA","reason":"No scaling parents declared"}
    models={r["model_id"]:_load_model(ctx,r,ctx.prerequisites["freeze"]) for r in selected if r["checkpoint_validated"]}
    rows=[]; horizon=float(options.get("horizon",.12))
    steps=options.get("steps",1)
    if type(steps) is not int or steps<1: raise ValueError("Scaling requires a positive frozen integer step count")
    schedule=[horizon/steps]*steps
    target=ctx.protocol.get("primary_target",2e-5)
    if isinstance(target,dict):rt=float(target.get("rms",target.get("rms_target")));mt=float(target.get("max",target.get("max_target")))
    else:rt=mt=float(target)
    # Distinct continuous parents are batched, never copies of one field.
    planned=[(n,t,b) for b in sorted(options.get("batch_sizes",options.get("batches",[1])),reverse=True)
             for n,t in itertools.product(options.get("grids",ctx.protocol["grids"]),ctx.protocol["tracks"])]
    # The case cap is frozen in the protocol; omitted cases are explicit NA.
    allowed=int(options.get("maximum_cases",len(planned)))
    for case_index,(n,track,batch) in enumerate(planned):
        if case_index>=allowed:
            rows.append({"grid":n,"track":track,"batch_size":batch,"status":"NA_DECLARED_CASE_CAP"});continue
        if batch>len(parents):
            rows.append({"grid":n,"track":track,"batch_size":batch,"status":"NA_INSUFFICIENT_DISTINCT_FIELDS"});continue
        chunk=parents[:batch]
        physics={(p["kappa"],p["reaction_rate"],tuple(p.get("lengths",[1.,1.]))) for p in chunk}
        if len(physics)!=1:
            rows.append({"grid":n,"track":track,"batch_size":batch,"status":"NA_BATCH_PHYSICS_DIFFER"});continue
        refs=[generate_reference(p,n,horizon,track,ctx.protocol,ctx.budget) for p in chunk]
        u=torch.cat([field_state(p,n) for p in chunk]).to(ctx.device,dtype=torch.float32)
        eq=Equation(chunk[0]["kappa"],chunk[0]["reaction_rate"]);geom=geometry(chunk[0],n)
        calls={r["model_id"]:(lambda m=models[r["model_id"]]:rollout(m,u,schedule,eq,geom,ctx.budget)[0]) for r in selected if r["track"]==track and r["model_id"] in models}
        with torch.no_grad():
            answers,timing=measure_paired(calls,device=ctx.device,repeats=int(ctx.protocol.get("timing",{}).get("repeats",3)),
                                         warmup=1,budget=ctx.budget,seed=n*17+batch)
        for r in selected:
            if r["track"]!=track or r["model_id"] not in answers:continue
            answer=answers[r["model_id"]];cost=timing["methods"][r["model_id"]]
            errors=[endpoint_errors(answer[i:i+1],ref) for i,ref in enumerate(refs)]
            qualified=all(endpoint_eligibility({**e,"cost_seconds":cost["median_seconds"]},rt,mt)=="ELIGIBLE" for e in errors)
            rows.append({"model_id":r["model_id"],"family":r["family"],"track":track,"grid":n,"batch_size":batch,
                "status":"ACCURACY_QUALIFIED" if qualified else "ACCURACY_UNQUALIFIED","errors":errors,
                "rms_target":rt,"max_target":mt,"cost_seconds":cost["median_seconds"],"cold_seconds":cost["cold_seconds"],
                "samples_per_second":batch/cost["median_seconds"],"raw_timing":timing,"final_time":horizon,"schedule":schedule,"step_count":steps,
                "parent_ids":[p["parent_id"] for p in chunk],"teacher_seconds":sum(ref["teacher_seconds"] for ref in refs),
                "precision":"FP32 models/FP64 references","device":str(ctx.device)})
    write_json(ctx.path/"scaling_rows.json",{"rows":clean(rows),"scope":"measured workloads only; first declared seed not selected by confirmation"})
    return {"measured_rows":len(rows),"qualified":sum(r["status"]=="ACCURACY_QUALIFIED" for r in rows)}

from .data import prepare


def primary_claims(rows,tables,protocol):
    """Predeclared limited claims, separate from exploratory all-pair tables.

    Cost success requires both endpoint norms, matched workloads, >=95% joint
    field coverage and a measured 1.2x effect. Intervals are descriptive paired
    field/seed bootstraps; multiplicity is not silently advertised as solved.
    """
    primary=protocol.get("selection",{}).get("primary_family","quad2_conditioned")
    thresholds=protocol.get("hypothesis_thresholds",{})
    target=protocol.get("primary_target",2e-5)
    if isinstance(target,dict):rt=float(target.get("rms",target.get("rms_target")));mt=float(target.get("max",target.get("max_target")))
    else:rt=mt=float(target)
    lookup={ (r["family"],r["track"],r["train_count"],r["seed"],r["parent_id"],r["grid"],r["schedule_id"]):r for r in rows }
    accuracy=[]; learned_controls=thresholds.get("learned_comparators",["quad2_amplitude","quad2_nodes","quad2_joint","quad2_linear"])
    controls=["quad2_fixed",*learned_controls,"fno_small","fno_standard"]
    scopes=sorted({(r["track"],r["train_count"],r["final_time"]) for r in rows if r["family"]==primary})
    for track,count,final in scopes:
        candidates=[r for r in rows if r["family"]==primary and r["track"]==track and r["train_count"]==count and r["final_time"]==final]
        for control in controls:
            observed=[];left_pass=right_pass=unresolved=failed=0;clusters={r["field_cluster"] for r in candidates}
            for left in candidates:
                shared=(left["parent_id"],left["grid"],left["schedule_id"])
                right=lookup.get((control,track,count,left["seed"],*shared)) or lookup.get((control,track,0,None,*shared))
                left_pass+=endpoint_eligibility(left,rt,mt)=="ELIGIBLE"
                right_pass+=bool(right and endpoint_eligibility(right,rt,mt)=="ELIGIBLE")
                if not right or not left.get("reference_accepted") or not right.get("reference_accepted"):
                    unresolved+=1;continue
                if not left.get("finite") or not right.get("finite"):
                    failed+=1;continue
                # Do not claim improvement from a ratio below teacher uncertainty.
                if any(r["error_rms"]<=float(r.get("reference_uncertainty_rms") or 0.) for r in (left,right)):
                    unresolved+=1;continue
                observed.append({"field_cluster":left["field_cluster"],"seed":left["seed"],
                    "difference":math.log(right["error_rms"]/left["error_rms"])})
            interval=field_cluster_interval(observed,repeats=int(protocol.get("bootstrap_replicates",100)),
                minimum_fields=int(thresholds.get("minimum_fields",5)))
            ratio=math.exp(interval["mean"]) if interval.get("mean") is not None else None
            lower=math.exp(interval["lower"]) if interval.get("lower") is not None else None
            coverage=(left_pass-right_pass)/len(candidates) if candidates else None
            if failed:verdict="BAD"
            elif unresolved or lower is None or interval.get("independent_fields")!=len(clusters):verdict="NA"
            else:verdict="GOOD" if ratio>=float(thresholds.get("learned_error_ratio",1.1)) and lower>1 and coverage>=-float(thresholds.get("maximum_coverage_regression",.05)) else "BAD"
            accuracy.append(dict(claim="bounded_neural_accuracy" if "fno" in control else "learned_attribution",candidate_family=primary,
                control_family=control,track=track,train_count=count,final_time=final,regime="ALL_DECLARED_FIELDS",
                verdict=verdict,geometric_control_over_candidate_rms=ratio,lower_ratio=lower,interval=interval,
                candidate_passes=left_pass,control_passes=right_pass,denominator=len(candidates),independent_fields=len(clusters),
                coverage_difference=coverage,unresolved_pairs=unresolved,numerical_failure_pairs=failed,
                required_ratio=float(thresholds.get("learned_error_ratio",1.1)),required_lower_ratio=1.,
                maximum_coverage_regression=float(thresholds.get("maximum_coverage_regression",.05)),
                rms_target=rt,max_target=mt,metric="paired geometric RMS on identical schedules; joint RMS/max coverage retained",
                multiplicity="limited predeclared descriptive contrasts; intervals are unadjusted, no simultaneous familywise guarantee",
                profile_scope="fresh locked desktop cohort" if protocol["profile"]=="full" else "development diagnostic, not scientific confirmation"))
    locked=tables["locked_frontiers"]
    cost=[]
    lookup={ (r["family"],r["track"],r["train_count"],r["seed"],r["parent_id"],r["grid"],r["final_time"],r["rms_target"],r["max_target"]):r for r in locked }
    for track,count,final in scopes:
        candidates=[r for r in locked if r["family"]==primary and r["track"]==track and r["train_count"]==count and r["final_time"]==final and r["rms_target"]==rt and r["max_target"]==mt]
        for control in ["quad2_fixed","df","etdrk4","fno_small","fno_standard"]:
            observations=[];field_left=defaultdict(list);field_right=defaultdict(list);unresolved_pairs=0
            for left in candidates:
                shared=(left["parent_id"],left["grid"],final,rt,mt)
                right=lookup.get((control,track,count,left["seed"],*shared)) or lookup.get((control,track,0,None,*shared))
                unknown={"REFERENCE_UNACCEPTED","MISSING_METRIC","UNAVAILABLE_CHECKPOINT"}
                unresolved_pairs+=left["status"] in unknown or bool(right and right["status"] in unknown)
                lp=left["status"]=="ELIGIBLE";rp=bool(right and right["status"]=="ELIGIBLE")
                field_left[left["field_cluster"]].append(lp);field_right[left["field_cluster"]].append(rp)
                if lp and rp:
                    observations.append({"field_cluster":left["field_cluster"],"seed":left["seed"],
                        "difference":math.log(right["cost_seconds"]/left["cost_seconds"])})
            interval=field_cluster_interval(observations,repeats=int(protocol.get("bootstrap_replicates",100)),
                minimum_fields=int(thresholds.get("minimum_fields",5)))
            coverage=statistics.mean(all(v) for v in field_left.values()) if field_left else None
            control_coverage=statistics.mean(all(v) for v in field_right.values()) if field_right else None
            ratio=math.exp(interval["mean"]) if interval.get("mean") is not None else None
            lower=math.exp(interval["lower"]) if interval.get("lower") is not None else None
            if unresolved_pairs:verdict="NA"
            elif coverage is not None and (coverage<float(thresholds.get("required_feasible_fraction",.95)) or coverage<control_coverage-float(thresholds.get("maximum_coverage_regression",.05))):verdict="BAD"
            elif lower is None or coverage is None:verdict="NA"
            else:verdict="GOOD" if ratio>=float(thresholds.get("practical_speedup",1.2)) and lower>1 else "BAD"
            cost.append(dict(claim="validation_locked_solver_cost",candidate_family=primary,control_family=control,track=track,
                train_count=count,final_time=final,verdict=verdict,geometric_control_over_candidate_cost=ratio,lower_ratio=lower,
                interval=interval,candidate_joint_field_coverage=coverage,control_joint_field_coverage=control_coverage,
                independent_fields=len(field_left),eligible_paired_rows=len(observations),denominator=len(candidates),unresolved_pairs=unresolved_pairs,
                rms_target=rt,max_target=mt,required_speedup=float(thresholds.get("practical_speedup",1.2)),
                required_coverage=float(thresholds.get("required_feasible_fraction",.95)),
                metric="single-field complete-call warm latency at frozen validation schedules",
                coverage_scope="field passes only if every paired seed/grid query passes both error bounds",
                comparison_scope="validation-batch-selected schedules transferred unchanged to single-field confirmation; no estimator/fallback deployed",
                multiplicity="descriptive unadjusted paired intervals; not a simultaneous familywise guarantee"))
    return {"accuracy":accuracy,"cost":cost,"scientific_success_is_not_job_completion":True,
            "statistical_unit":"independent field; repeated grid/schedule/seed observations paired",
            "primary_family":primary,"no_published_fno_superiority_claim":True}
