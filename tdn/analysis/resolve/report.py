"""Small, source-bound Resolve atlas; charts never select scientific methods.

Only prerequisites already verified by the engine are read.  Every consumed
file is matched to its seal and checked again after rendering.  Display bands
are observed ranges, not confidence intervals; raw measurements stay sealed.
"""
from __future__ import annotations

from collections import defaultdict
import gzip
import hashlib
import html
import json
import math
from pathlib import Path

import numpy as np

from tdn.analysis.advance.report import _read
from tdn.analysis.frontier.core import check


SCHEMA = "tdn.resolve-atlas/v1"
FILES = ("summary.json", "rows.jsonl", "fit_results.json", "fit_endpoints.json",
         "interaction_rows.json", "interaction_cases.json", "fairness_results.json",
         "fairness_metrics.json", "learning_curves.json", "kernel_rows.json", "profiling.json",
         "memory_summary.json", "memory_results.json")
GUIDANCE = ("Ours: triangles/solid; Theirs: circles/dashed; analytic controls: squares/dash-dot. "
            "Thin curves and translucent low–high observed bands; bands are not confidence intervals. "
            "Missing evidence is NA, never zero. No universal ranking or published FNO superiority.")


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def role(method):
    if method.startswith("fno") or method in ("df", "etdrk4"):
        return "Theirs"
    if method in ("two_basis", "two_gain_direct", "two_gain_adam", "quadratic_only_direct",
                  "quadratic_gain_direct", "cubic_gain_direct", "noise_aware_fitted_memory", "quad3_sequential") or any(
            part in method for part in ("raw_cubic", "filtered_cubic", "batched", "midpoint_elided")):
        return "Ours"
    if method == "clean_fitted_memory":
        return "Fitted control"
    return "Analytic control"


def style(method):
    import colorsys
    category = role(method)
    # Fixed families need distinguishable saturated colors: hashing into a
    # short categorical palette gave distinct attribution arms the same pastel.
    colors = {"fixed_unit":"#555555", "quadratic_only_direct":"#0072B2",
        "quadratic_gain_direct":"#009E73", "cubic_gain_direct":"#D55E00",
        "two_gain_direct":"#CC3377", "two_gain_adam":"#663399", "two_basis":"#0072B2",
        "df":"#555555", "etdrk4":"#222222", "quad2_full":"#117733", "quad4_full":"#887711",
        "analytic_quad_cubic":"#994F00", "quad3_sequential":"#882255", "quad2_batched":"#4477AA",
        "quad4_batched":"#009988", "quad3_midpoint_elided":"#CC3311",
        "gl2":"#117733", "gl4":"#887711", "gl2_raw_cubic":"#AA3377",
        "gl2_filtered_cubic":"#0072B2", "gl4_nested_cubic":"#994F00",
        "coarse_rk4":"#555555", "plain_secant":"#AA4499", "validated_damped_secant":"#DD6600",
        "clean_fitted_memory":"#117733", "noise_aware_fitted_memory":"#0072B2",
        "fno_scaled":"#D55E00", "fno_mean":"#AA3377"}
    hue = int(hashlib.sha256(method.encode()).hexdigest()[:8], 16) / (2**32)
    color = colors.get(method, colorsys.hsv_to_rgb(hue, .72, .65))
    return dict(color=color, marker={"Ours": "^", "Theirs": "o",
        "Analytic control": "s", "Fitted control": "D"}[category],
        linestyle={"Ours": "-", "Theirs": "--", "Analytic control": "-.", "Fitted control": ":"}[category], linewidth=.65)


def _safe_path(base, name):
    base = Path(base)
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe atlas source path")
    path = base/relative
    if not path.resolve().is_relative_to(base.resolve()) or any(v.is_symlink() for v in (path, *path.parents)):
        raise ValueError("Atlas sources must stay inside their sealed directory without symlinks")
    return path


def _strip(value):
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items() if k not in
            ("curves", "metric_rows", "state", "targets", "previous", "current", "training_rows", "startup_observations")}
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value


def collect(ctx):
    """Collect only explicitly verified prerequisite directories, never siblings."""
    data = dict(schema=SCHEMA, sources=[], stages=[], files={}, omissions=[], profile=ctx.protocol["profile"],
        guidance=GUIDANCE, statistics="Development fields only. Repeated noise draws, seeds, schedules and grids remain paired.",
        projection="Display projection only; raw loss appears once. Source IDs and JSON pointers trace every table.")
    for stage, unit in ctx.protocol["units"].items():
        if stage == ctx.stage:
            continue
        base = ctx.prerequisites.get(stage)
        failure = getattr(ctx, "stage_failures", {}).get(stage, {})
        if not isinstance(failure, dict):
            failure = dict(error=str(failure))
        status = dict(stage=stage, kind=unit["kind"], track=unit.get("track"),
            status="VERIFIED" if base else failure.get("status", "NA"), verified=base is not None,
            elapsed_seconds=failure.get("elapsed_seconds"), error=failure.get("error"), monetary_cost=None)
        data["stages"].append(status)
        if base is None:
            data["omissions"].append(dict(stage=stage, reason="No verified scientific prerequisite; only failure/status metadata shown"))
            continue
        manifest_path = _safe_path(base, "science_manifest.json")
        manifest, sha, size = _read(manifest_path)
        data["sources"].append(dict(source_id=f"{stage}/science_manifest.json", path=str(manifest_path), sha256=sha, bytes=size))
        artifacts = manifest.get("artifacts", {})
        for name in FILES:
            path = _safe_path(base, name)
            if not path.exists():
                continue
            if name not in artifacts:
                raise ValueError(f"Atlas file is absent from its seal: {stage}/{name}")
            value, sha, size = _read(path)
            if sha != artifacts[name]:
                raise ValueError(f"Atlas sealed source changed: {stage}/{name}")
            source_id = f"{stage}/{name}"
            data["sources"].append(dict(source_id=source_id, path=str(path), sha256=sha, bytes=size))
            if name == "summary.json":
                status.update(elapsed_seconds=value.get("elapsed_seconds"), experiments=value.get("experiment_count"),
                    correctness_failures=value.get("correctness_failures"), scientific_outcome=value.get("scientific_outcome"))
            if name == "rows.jsonl":
                # Do not duplicate full module-specific payloads/curves inside
                # the generic experiment ledger projection.
                value = [{k: r.get(k) for k in ("experiment_id", "mechanism_ids", "assessment", "checks", "stage")} for r in value]
            elif name == "memory_results.json":
                value = {k: _strip(value[k]) for k in ("rows", "offline_costs", "selected_damping", "settings", "selection_sha256") if k in value}
            elif name != "learning_curves.json":
                # Fit curves have no other source. All other repeated curve
                # payloads are removed, especially fairness trial dictionaries.
                if name != "fit_results.json":
                    value = _strip(value)
            data["files"][source_id] = dict(stage=stage, kind=unit["kind"], track=unit.get("track"),
                source_id=source_id, json_pointer="", payload=value)
    return data


def verify_sources(data):
    for item in data["sources"]:
        _, sha, size = _read(Path(item["path"]))
        if sha != item["sha256"] or size != item["bytes"]:
            raise ValueError(f"Atlas source changed while rendering: {item['source_id']}")


def observed_ranges(rows, key="train_loss", bins=48):
    """Continuous display envelopes within genuinely contiguous update runs.

    Nulls and absent updates break the band.  Multiple observations at the same
    update contribute their range.  Linear joins connect window centers only
    within each measured run, with first/last update anchors retained.
    """
    updates = defaultdict(list)
    for row in rows:
        if finite(row.get("update")):
            updates[int(row["update"])].append(row.get(key))
    segments, current, previous = [], [], None
    for update in sorted(updates):
        values = [v for v in updates[update] if finite(v)]
        if not values or previous is not None and update != previous+1:
            if current:
                segments.append(current); current = []
        if values:
            current.append((update, min(values), max(values)))
        previous = update
    if current:
        segments.append(current)
    result = []
    for segment in segments:
        width = max(1, math.ceil(len(segment)/bins))
        groups = [segment[i:i+width] for i in range(0, len(segment), width)]
        x, low, high = [], [], []
        for group in groups:
            x.append(float(np.mean([v[0] for v in group])))
            low.append(min(v[1] for v in group)); high.append(max(v[2] for v in group))
        if x[0] != segment[0][0]:
            x.insert(0, float(segment[0][0])); low.insert(0, low[0]); high.insert(0, high[0])
        if x[-1] != segment[-1][0]:
            x.append(float(segment[-1][0])); low.append(low[-1]); high.append(high[-1])
        result.append(dict(x=x, lower=low, upper=high, observed_count=len(segment),
            band_kind="observed minimum–maximum within display windows; not a confidence interval"))
    return result


def _table(data, filename):
    for key, table in data["files"].items():
        if key.endswith("/"+filename):
            yield table


def _records(table, key="rows"):
    value = table["payload"]
    return value if isinstance(value, list) else value.get(key, [])


def _missing(ax, text="NA: no verified measurements"):
    ax.text(.5, .5, text, transform=ax.transAxes, ha="center", va="center", color="#666666")


def _axes(ax, x, y, title):
    ax.set(title=title, xlabel=x, ylabel=y)
    ax.grid(alpha=.16, linewidth=.4)
    ax.tick_params(labelsize=7)


def quantitative_axis(ax, coordinate="y"):
    """Keep quantitative ticks even when a log-like range misses every decade.

    Narrow positive ranges use scientific-formatted linear axes. Wide ranges
    retain log spacing. Zeros and signed values remain representable; this
    changes only presentation, never values, objectives or error thresholds.
    """
    from matplotlib import ticker
    axis = ax.yaxis if coordinate == "y" else ax.xaxis
    get_scale = ax.get_yscale if coordinate == "y" else ax.get_xscale
    set_scale = ax.set_yscale if coordinate == "y" else ax.set_xscale
    interval = ax.dataLim.intervaly if coordinate == "y" else ax.dataLim.intervalx
    lo, hi = map(float, interval)
    if get_scale() not in ("log", "symlog") or not all(map(math.isfinite, (lo, hi))):
        return
    transform = axis.get_transform()
    threshold = getattr(transform, "linthresh", 1e-12)
    narrow = lo > 0 and hi/max(lo, np.finfo(float).tiny) < 10
    central = max(abs(lo), abs(hi)) <= 10*threshold
    if narrow or central:
        set_scale("linear")
        axis.set_major_locator(ticker.MaxNLocator(nbins=5, min_n_ticks=3))
        formatter = ticker.ScalarFormatter(useMathText=True, useOffset=False)
        formatter.set_powerlimits((-3, 3))
        axis.set_major_formatter(formatter)
    elif lo > 0:
        set_scale("log")
        decades = math.log10(hi/lo)
        axis.set_major_locator(ticker.LogLocator(base=10, subs=(1., 2., 5.) if decades < 3 else (1.,), numticks=9))
        axis.set_major_formatter(ticker.FuncFormatter(lambda value, position: f"{value:.3g}"))
    else:
        set_scale("symlog", linthresh=threshold)
        locator = ticker.SymmetricalLogLocator(base=10, linthresh=threshold, subs=(1.,))
        locator.set_params(numticks=9)
        axis.set_major_locator(locator)
        axis.set_major_formatter(ticker.FuncFormatter(lambda value, position: f"{value:.3g}"))


def _save(fig, root, name, caption, figures, sources):
    import matplotlib.pyplot as plt
    for ax in fig.axes:
        quantitative_axis(ax,"x"); quantitative_axis(ax,"y")
    fig.suptitle(caption, fontsize=10, y=.995, wrap=True)
    fig.tight_layout(rect=(0, .018, 1, .958))
    fig.text(.01, .004, "Development evidence · lower error/cost is better · observed bands ≠ confidence intervals", fontsize=6, color="#555555")
    for suffix in ("png", "pdf"):
        fig.savefig(root/f"{name}.{suffix}", dpi=260, bbox_inches="tight")
    plt.close(fig)
    figures.append(dict(name=name, png=f"{name}.png", pdf=f"{name}.pdf", caption=caption,
        source_ids=sorted(set(sources)), guidance=GUIDANCE))


def _fit_figures(data, root, figures):
    import matplotlib.pyplot as plt
    for table in _table(data, "fit_results.json"):
        payload = table["payload"]; fits = payload.get("fits", {})
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
        for method, fit in fits.items():
            s = style(method); label=f"{role(method)}: {method}"
            gains=fit.get("gains", [])
            if len(gains)==2:
                axes[0].plot([0, 1], gains, **s, label=label, markersize=4)
            curves=fit.get("curves", [])
            values=[v for v in curves if finite(v.get("objective", v.get("upper_bound")))]
            if values:
                axes[1].plot([v.get("update", v.get("iteration", i)) for i,v in enumerate(values)],
                    [v.get("objective", v.get("upper_bound")) for v in values], **s, markersize=2)
            val=payload.get("validation", {}).get(method, {}).get("validation_rms")
            if finite(val):
                axes[2].scatter(fit.get("elapsed_seconds", 0.), val, marker=s["marker"], color=s["color"], s=28, label=label)
        axes[0].set_xticks([0,1], ["Quadratic g₂", "Cubic g₃"]);axes[0].axhline(1,color="gray",lw=.5)
        axes[0].legend(fontsize=5.6, loc="best")
        _axes(axes[0], "Coefficient (no universal preferred value)", "Gain (not a quality score)", "Fitted physical coefficients")
        _axes(axes[1], "Optimizer iteration/update (different algorithms)", "Training objective ↓", "Direct versus Adam: raw optimization history")
        _axes(axes[2], "Fitting wall time, seconds ↓ (not solver latency)", "Validation RMS ↓", "Fitting cost and development error")
        if any(line.get_ydata().size for line in axes[1].lines):axes[1].set_yscale("symlog",linthresh=1e-12)
        axes[2].set_yscale("symlog",linthresh=1e-12)
        _save(fig, root, table["stage"]+"-fit", f"{table['stage']}: attribution only; coefficient magnitude has no higher-is-better meaning", figures,[table["source_id"]])


def _interaction_figures(data, root, figures):
    import matplotlib.pyplot as plt
    groups=defaultdict(list)
    for table in _table(data,"interaction_rows.json"):
        for row in _records(table):groups[(row["track"],row["grid"])].append((table["source_id"],row))
    for (track,grid),items in groups.items():
        # Each h*r slice is separate. Fields and methods retain their raw points;
        # no interpolation assumes a stiffness law or pools physical targets.
        rates=sorted({r["reaction_number"] for _,r in items})
        fig,axes=plt.subplots(2,len(rates),figsize=(max(9,4*len(rates)),7.5),squeeze=False)
        for j,rate in enumerate(rates):
            subset=[r for _,r in items if r["reaction_number"]==rate]
            for method in sorted({r["method"] for r in subset}):
                rows=[r for r in subset if r["method"]==method];s=style(method)
                valid=[r for r in rows if finite(r.get("same_grid_metrics",{}).get("error_rms"))
                       and r.get("same_grid_metrics",{}).get("reference_accepted",True)]
                axes[0,j].scatter([r["diffusion_number"] for r in valid], [r["same_grid_metrics"]["error_rms"] for r in valid],
                    marker=s["marker"],color=s["color"],s=15,alpha=.68,label=f"{role(method)}: {method}")
            # Closure/discretization are teacher differences shared by methods.
            unique={r["case_id"]:r for r in subset}
            for component,marker in (("closure","s"),("discretization","D")):
                valid=[r for r in unique.values() if finite(r.get("error_decomposition",{}).get("component_rms",{}).get(component))]
                axes[1,j].scatter([r["diffusion_number"] for r in valid], [r["error_decomposition"]["component_rms"][component] for r in valid],
                    marker=marker,s=18,label=component)
            for i in (0,1):
                axes[i,j].set_xscale("symlog",linthresh=.01);axes[i,j].set_yscale("symlog",linthresh=1e-13)
            unresolved=sum(not r.get("same_grid_metrics",{}).get("reference_accepted",True) or
                not finite(r.get("same_grid_metrics",{}).get("error_rms")) for r in subset)
            if unresolved:axes[0,j].text(.02,.02,f"NA/unaccepted endpoints: {unresolved}",transform=axes[0,j].transAxes,fontsize=6)
            _axes(axes[0,j], "h λ_active (stress coordinate; no preferred direction)", "Same-grid RMS ↓", f"h r = {rate:g}")
            _axes(axes[1,j], "h λ_active (stress coordinate)", "Teacher-difference RMS ↓", "Separated closure and discretization")
            axes[1,j].legend(fontsize=6)
        axes[0,0].legend(fontsize=5,ncol=2)
        _save(fig,root,f"interactions-{track}-n{grid}", f"{track}, {grid}²: independent diffusion/reaction sweeps; raw fields remain visible. Norms do not add: cross terms are in source data.",figures,[sid for sid,_ in items])


def _learning_figures(data,root,figures):
    import matplotlib.pyplot as plt
    groups=defaultdict(list)
    for table in _table(data,"learning_curves.json"):
        for row in _records(table):groups[(row.get("track",table["track"]),row.get("family","unknown"))].append((table["source_id"],row))
    for (track,family),items in groups.items():
        trials=defaultdict(list)
        for sid,row in items:trials[(sid,row.get("trial_id","unknown"))].append(row)
        phases=sorted({r.get("phase","unknown") for _,r in items})
        fig,axes=plt.subplots(2,len(phases),figsize=(max(10,5*len(phases)),7.5),squeeze=False)
        for j,phase in enumerate(phases):
            shown=0
            for (sid,trial),rows in trials.items():
                rows=[r for r in rows if r.get("phase","unknown")==phase]
                if not rows:continue
                method=family; color=plt.get_cmap("tab20")(shown%20);shown+=1
                label=f"{role(method)}: {rows[0].get('optimizer_control','trial')} / seed {rows[0].get('seed','NA')}"
                for band in observed_ranges(rows):
                    x,lo,hi=band["x"],band["lower"],band["upper"]
                    axes[0,j].fill_between(x,lo,hi,color=color,alpha=.12)
                    axes[0,j].plot(x,lo,color=color,lw=.55,linestyle=style(method)["linestyle"],label=label)
                    axes[0,j].plot(x,hi,color=color,lw=.55,linestyle=style(method)["linestyle"])
                # Sparse validation values are separate points: no fabricated
                # continuous line across updates without validation evidence.
                valid=[r for r in rows if finite(r.get("validation_loss"))]
                axes[1,j].scatter([r["update"] for r in valid],[r["validation_loss"] for r in valid],s=12,
                    marker=style(method)["marker"],color=color,alpha=.7)
            for i in (0,1):axes[i,j].set_yscale("symlog",linthresh=1e-10)
            _axes(axes[0,j],"Training update → (work, not quality)","Training loss ↓",f"{phase}: observed windows; not CI")
            _axes(axes[1,j],"Training update → (unmeasured validation stays absent)","Validation objective ↓",f"{phase}: measured checkpoints only")
            if not shown:_missing(axes[0,j])
            else:axes[0,j].legend(fontsize=5)
        _save(fig,root,f"learning-{track}-{family}",f"{role(family)}: {family}, {track}. Continuous translucent display ranges retain raw extrema; no smoothing enters scores.",figures,[sid for sid,_ in items])


def _kernel_figures(data,root,figures):
    import matplotlib.pyplot as plt
    for table in _table(data,"kernel_rows.json"):
        rows=_records(table);grids=sorted({r["grid"] for r in rows})
        if not grids:continue
        fig,axes=plt.subplots(1,len(grids),figsize=(max(8,6*len(grids)),5),squeeze=False)
        for ax,grid in zip(axes[0],grids):
            for method in sorted({r["method"] for r in rows}):
                selected=[r for r in rows if r["method"]==method and r["grid"]==grid and finite(r.get("cost_seconds")) and finite(r.get("error_rms"))]
                s=style(method);ax.scatter([r["cost_seconds"] for r in selected],[r["error_rms"] for r in selected],
                    marker=s["marker"],color=s["color"],s=24,alpha=.7,label=f"{role(method)}: {method}")
            ax.set_xscale("log");ax.set_yscale("symlog",linthresh=1e-12)
            _axes(ax,"Complete-call warm median, seconds ↓","Same-grid RMS ↓",f"{grid}² · lower-left better")
        axes[0,0].legend(fontsize=5.8)
        _save(fig,root,table["stage"]+"-cost",f"{table['stage']}: frozen development schedules; DF is an unfused implementation, not a strongest-classical cost frontier. Accuracy qualification remains in raw records.",figures,[table["source_id"]])
    for table in _table(data,"profiling.json"):
        profiles=table["payload"].get("profiles",[])
        if not profiles:continue
        methods=[p["method"] for p in profiles];x=np.arange(len(methods))
        fig,axes=plt.subplots(1,2,figsize=(14,5))
        for offset,key in enumerate(("fft_dispatch","products_and_scalar_multiplication")):
            values=[p.get("invocation_counts",{}).get(key) for p in profiles]
            valid=[i for i,v in enumerate(values) if finite(v)]
            axes[0].bar(x[valid]+(offset-.5)*.35,[values[i] for i in valid],width=.35,label=key)
        axes[1].bar(x,[p["profile_seconds"] for p in profiles],color="#888888")
        for ax in axes:ax.set_xticks(x,[f"{role(m)}: {m}" for m in methods],rotation=70,ha="right",fontsize=6)
        _axes(axes[0],"Method (same representative field)","Operator dispatch count ↓ (not FLOPs)","Transform/product dispatches")
        _axes(axes[1],"Method (not complete-call latency)","Profiler wall seconds ↓", "Instrumentation overhead: separate from deployment")
        axes[0].legend(fontsize=6)
        _save(fig,root,table["stage"]+"-profile",f"{table['stage']}: profiler counts include scalar products and checks; instrumentation time is not solver latency.",figures,[table["source_id"]])


def _memory_figures(data,root,figures):
    import matplotlib.pyplot as plt
    for table in _table(data,"memory_results.json"):
        rows=table["payload"].get("rows",[]);rates=sorted({r["reaction_multiplier"] for r in rows})
        if not rows:continue
        noises=sorted({r["noise"] for r in rows})
        fig,axes=plt.subplots(len(rates),len(noises),figsize=(14,4*len(rates)),squeeze=False)
        for i,rate in enumerate(rates):
            for j,noise in enumerate(noises):
                ax=axes[i,j]
                for method in sorted({r["method"] for r in rows}):
                    selected=[r for r in rows if r["method"]==method and r["reaction_multiplier"]==rate and r["noise"]==noise]
                    steps=sorted({r["rollout_step"] for r in selected});med=[];lo=[];hi=[]
                    for step in steps:
                        values=[r["error_rms"] for r in selected if r["rollout_step"]==step and finite(r.get("error_rms"))]
                        med.append(float(np.median(values)) if values else np.nan);lo.append(min(values) if values else np.nan);hi.append(max(values) if values else np.nan)
                    s=style(method);ax.fill_between(steps,lo,hi,color=s["color"],alpha=.08)
                    ax.plot(steps,lo,color=s["color"],lw=.4);ax.plot(steps,hi,color=s["color"],lw=.4)
                    ax.plot(steps,med,**s,markersize=2,markevery=max(1,len(steps)//5),label=f"{role(method)}: {method}")
                ax.set_yscale("symlog",linthresh=1e-10)
                _axes(ax,"Autonomous rollout step → (longer horizon)","RMS ↓; range is not CI",f"reaction ×{rate:g}, coordinate noise σ={noise:g}")
        axes[0,0].legend(fontsize=5.4)
        _save(fig,root,"memory-rollouts", "RX: fresh development parents, paired noise. Median and full observed range; no future teacher refresh.",figures,[table["source_id"]])


def _cost_figure(data,root,figures):
    import matplotlib.pyplot as plt
    rows=data["stages"];height=max(4,.19*len(rows)+1.5)
    fig,ax=plt.subplots(figsize=(12,height));y=np.arange(len(rows))
    for i,row in enumerate(rows):
        value=row["elapsed_seconds"]
        if finite(value):ax.barh(i,value,color="#4477AA" if row["verified"] else "#CC6677")
        else:ax.text(0,i,"NA — no elapsed measurement",fontsize=6,va="center",color="#777777")
    ax.set_yticks(y,[f"{r['stage']} [{r['status']}]" for r in rows],fontsize=6);ax.invert_yaxis()
    _axes(ax,"Measured scientific-stage wall seconds ↓; missing ≠ 0", "Stage (failed attempts retained where measured)","Cost inventory, not scheduler billed time")
    _save(fig,root,"stage-costs","Measured wall costs include failed attempts when supplied. Report/queue/accounting, power and dollar costs remain separate or unavailable.",figures,[s["source_id"] for s in data["sources"] if s["source_id"].endswith("/summary.json")])


def run(ctx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":8,"axes.titlesize":9,"axes.labelsize":8,"lines.linewidth":.65,"pdf.fonttype":42})
    ctx.path.mkdir(parents=True,exist_ok=True)
    data=collect(ctx);figures=[]
    for maker in (_fit_figures,_interaction_figures,_learning_figures,_kernel_figures,_memory_figures,_cost_figure):
        ctx.budget.check();maker(data,ctx.path,figures)
    verify_sources(data)
    data["figures"]=figures
    with gzip.open(ctx.path/"chart-data.json.gz","wt",encoding="utf-8",compresslevel=5) as stream:
        json.dump(data,stream,separators=(",",":"),allow_nan=False)
    (ctx.path/"sources.json").write_text(json.dumps(data["sources"],indent=2)+"\n")
    (ctx.path/"figures.json").write_text(json.dumps(figures,indent=2)+"\n")
    esc=html.escape
    stages="".join(f"<tr><td>{esc(r['stage'])}</td><td>{esc(r['status'])}</td><td>{r['elapsed_seconds'] if finite(r['elapsed_seconds']) else 'NA'}</td><td>{esc(str(r.get('error') or ''))}</td></tr>" for r in data["stages"])
    panels="".join(f"<section><h2>{esc(f['name'])}</h2><p>{esc(f['caption'])}</p><a href='{f['png']}'><img src='{f['png']}' alt='{esc(f['caption'])}'></a><p><a href='{f['pdf']}'>Vector PDF</a> · Sources: {esc(', '.join(f['source_ids']))}</p></section>" for f in figures)
    missing="".join(f"<li>{esc(o['stage'])}: {esc(o['reason'])}</li>" for o in data["omissions"])
    (ctx.path/"index.html").write_text("<!doctype html><html><meta charset='utf-8'><title>TDN Resolve analytical atlas</title><style>body{font:15px system-ui;margin:2em auto;max-width:1500px;padding:1em}img{width:100%;height:auto}td,th{padding:.3em;text-align:left;border-bottom:1px solid #ddd}section{margin:3em 0}a{color:#17689a}</style><h1>TDN Resolve: development analytical atlas</h1><p>"+esc(GUIDANCE)+"</p><p>Computational completion, mathematical checks and scientific benefit are distinct. No model is selected by this report.</p><p><a href='chart-data.json.gz'>Compressed exact display data</a> · <a href='sources.json'>Source hashes</a></p><table><tr><th>Stage</th><th>Status</th><th>Measured seconds</th><th>Failure</th></tr>"+stages+"</table><h2>Missing evidence</h2><ul>"+(missing or "<li>All supplied prerequisite stages were verified.</li>")+"</ul>"+panels+"</html>")
    size=sum(p.stat().st_size for p in ctx.path.iterdir() if p.is_file())
    ctx.record("report/source-inventory",["A1"],metrics=dict(consumed_files=len(data["sources"]),
        verified_stages=sum(r["verified"] for r in data["stages"]), missing_stages=len(data["omissions"]),
        figures=len(figures),report_bytes=size),checks=[check("consumed-source-hashes-unchanged",True,True,"eq",category="math"),
        check("report-establishes-scientific-advantage",None,True,"eq",category="gap",reason="A reporting/inventory check cannot establish solver benefit")],
        evidence=["sources.json","chart-data.json.gz","figures.json","index.html"])
    return dict(figures=len(figures),consumed_files=len(data["sources"]),report_bytes=size,
        report="index.html",chart_data="chart-data.json.gz",scientific_outcome="REPORT_ONLY; INSPECT SOURCE CHECKS")
