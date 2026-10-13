"""Portable, source-linked research views for the revised Response diagnostic."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import time
import zipfile

from tdn.analysis.frontier.core import check
from tdn.research.protocol import digest
from tdn.runtime.metadata import write_json
from . import diagnostic_metrics as metrics

SCHEMA = "tdn.response-diagnostic-report/v1"
BUNDLE = "response-diagnostic-raven.zip"
METRIC_DEFINITIONS = {
    "completed_trials": "Count of registered trial records reporting COMPLETED_UPDATES; this is a trial count, not optimizer work.",
    "achieved_updates": "Sum of achieved logical optimizer updates across registered trials; continuation histories are counted once.",
    "requested_updates": "Sum of frozen requested logical optimizer updates across all registered trials, including failed, missing and unresolved work.",
    "learning_curve_export": "Declared public observation cadence and exact query-plan references; public observations are a sampled view of private optimizer history.",
    "optimization_history_summary": "Gradient, clipping and work summaries computed from every committed private optimization row, independently of public observation sampling.",
    "mse": "Mean physical squared endpoint error, parent-weighted; physical interpolant norm for Galerkin targets.",
    "rms": "Physical root mean square endpoint error; parent-mean summary is not sqrt(parent-mean MSE).",
    "maximum": "Largest absolute error on the sampled nodes; not a certified continuum maximum.",
    "mean_error": "Signed spatial mean prediction error.",
    "mean_absolute": "Parent-weighted absolute spatial mean prediction error.",
    "centered_rms": "Physical RMS of error after removing its spatial average.",
    "sampled_rms": "Unweighted sampled nodal RMS; distinct from Galerkin physical RMS.",
    "conservative_rms": "Measured physical RMS plus the reference RMS uncertainty estimate; not an error certificate.",
    "conservative_maximum": "Sampled maximum error plus the reference maximum uncertainty estimate.",
    "objective": "Training loss after fixed mean/centered scaling; compare only within the same objective contract.",
    "anchor_mse_reduction": "Squared teacher-minus-deployed-anchor RMS divided by model MSE, on this same query.",
    "anchor_rms_reduction": "Teacher-minus-deployed-anchor physical RMS divided by model RMS on the same query.",
    "initialization_mse_reduction": "Initial-model physical MSE divided by achieved-model MSE; not a common baseline across families.",
    "floor_limited": "Teacher-minus-deployed-anchor defect is below the frozen normalization safeguard; not a trained-error floor.",
    "rms_headroom_ratio": "Reference-adjusted correction signal relative to a declared representation/precision safeguard.",
    "nearest_fp32_mse_reduction_benchmark": "A declared FP32-output representability benchmark; componentwise nearest rounding minimizes sampled nodal MSE, not Galerkin physical interpolant error. Not guaranteed neural performance.",
    "gradient_norm": "Objective-gradient norm before clipping; depends on objective normalization and parameterization.",
    "optimizer_gradient_norm": "Gradient norm for parameters active in the current optimizer phase, before clipping.",
    "head_gradient_norm": "Gradient norm of parameters classified as final correction heads.",
    "trunk_gradient_norm": "Gradient norm of the remaining feature-producing parameters.",
    "examples_seen": "Cumulative endpoint examples processed by optimizer updates; repeated queries are not independent fields.",
    "logical_updates": "Optimizer updates, distinct from endpoint forward/backward query work.",
    "parameters": "Model scalar parameter count; does not imply matched runtime or representation.",
    "elapsed_seconds": "Observed cumulative trial-body wall time, including explicitly reported diagnostic work.",
    "cuda_peak_allocated_bytes": "PyTorch CUDA allocator peak measured through the trial body before post-fit controls, checkpoint reload and benchmark work; not total device use or the complete stage peak.",
    "cuda_peak_reserved_bytes": "PyTorch CUDA reserved-memory peak measured through the trial body before post-fit work; separate from engine stage-memory observations.",
    "raw_mean": "Magnitude of the raw mean-head output before correction assembly.",
    "raw_centered": "Magnitude of raw centered correction coefficients/features before assembly.",
    "mean_gain": "Effective fixed/physical multiplier applied to the raw mean correction.",
    "centered_gain": "Effective fixed/physical multiplier applied to the centered correction.",
    "assembly_difference": "Difference between final FP32 and FP64 assembly of the same deployed anchor/correction.",
    "sub_ulp_correction_fraction": "Fraction of requested correction values smaller than local FP32 output spacing; diagnostic, not an error bound.",
    "deployment_defect": "FP64 teacher minus actual deployed FP32 anchor.",
    "physical_defect": "FP64 teacher minus the corresponding FP64 anchor.",
    "anchor_roundoff": "Difference between actual FP32 and FP64 anchor results.",
    "input_quantization": "Difference caused by FP32 input quantization; recorded only where measured independently.",
    "subtraction_roundoff": "Finite arithmetic change in residual subtraction; unknown values remain missing.",
    "complete_call_median_seconds": "Median synchronized full selected-model call with resident input on one prospectively chosen accepted validation query.",
    "anchor_median_seconds": "Median paired same-query deployed physical-anchor call; not a matched-tolerance classical frontier.",
    "model_to_anchor_time_ratio": "Selected-model complete-call median divided by paired anchor median; no solver utility claim without matched accuracy.",
    "encoding_seconds": "One synchronized reusable spatial-encoding measurement, when the architecture declares valid same-state reuse.",
    "cached_decode_median_seconds": "Median repeated same-state cached decode; excludes encoding and does not represent autonomous rollout cost.",
}


def _analysis_path(ctx):
    project = metrics.raven_root(ctx)
    return project / "runs" / metrics.stable_id(Path(ctx.path).parent.name + "-analysis")


def _definition_files(path):
    write_json(path/"metric_definitions.json", dict(schema="tdn.response-metric-definitions/v1",
        definitions=METRIC_DEFINITIONS,
        statistical_unit="Independent initial field parent; seeds, phases and horizons remain paired descendants",
        plots="Observed public learning points at the declared cadence, including initialization/final observations; full private histories remain in optimizer journals and whole-history summaries. No confidence intervals or missing-value zeros",
        export_scope="Compiled initials and fitted controls keep scalar summaries and exact source pointers; complete query rows and control matrices remain in canonical copied JSON",
        cost="Science-body, optimizer and scheduler allocation costs are separate scopes; no monetary cost without a rate"))
    lines = ["# Research metric legend", "", "Missing values mean unavailable evidence, never zero or success.", "",
        "Learning plots use the declared public recording cadence. Complete private optimization histories remain in exact-resume journals; all-history gradient, clipping and work summaries are retained independently of plot sampling.", "",
        "Training-update loss and scheduled full-training-bank objective have different sampling scopes. Physical training curves use the scheduled entire training bank; validation uses its separate bank.", "",
        "Compiled initial metrics and fitted controls omit duplicated query rows and dense matrices. Their exact canonical source JSON remains included and hash-cited; scalar summaries, row counts and matrix shapes remain in the overview.", ""]
    for name, meaning in METRIC_DEFINITIONS.items():
        lines.extend([f"- **{name}**: {meaning}"])
    (path/"metric_legend.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def screening_rows(data):
    rows = []
    for source in data.get("screening", []):
        for candidate_index, candidate in enumerate(source["payload"].get("candidates", [])):
            for query_index, query in enumerate(candidate.get("queries", [])):
                row = dict(query, target_kind=candidate.get("target_kind"), split=candidate.get("split"),
                    regime=candidate.get("regime"), parent_id=candidate.get("parent_id"),
                    candidate_eligible=candidate.get("eligible"), selection_status=candidate.get("selection_status"),
                    source_path=source["source_path"], source_sha256=source["source_sha256"],
                    source_record=f"/candidates/{candidate_index}/queries/{query_index}")
                rows.append(row)
    return rows


def correction_rows(data):
    rows = []
    for trial in data.get("trials", []):
        diagnostic = trial.get("correction_diagnostics", {})
        values = diagnostic.get("rows", [])
        if not values:
            rows.append(dict(trial_id=trial["trial_id"], status=diagnostic.get("status", "MISSING"),
                reason=diagnostic.get("reason", "No recorded correction instrumentation"),
                source_path=trial.get("source_path"), source_sha256=trial.get("source_sha256")))
        for index, row in enumerate(values):
            rows.append(dict(row, trial_id=trial["trial_id"], status=diagnostic.get("status"),
                source_path=trial.get("source_path"), source_sha256=trial.get("source_sha256"),
                source_record=f"/trials/{trial.get('source_trial_index',0)}/correction_diagnostics/rows/{index}"))
    return rows


def fixed_control_rows(data):
    """Keep fitted controls and conditioning visible without loading their weights."""
    rows = []
    for source in data.get("fixed_controls", []):
        payload = source["payload"]
        # Each row keeps its full declared JSON; interpretation never changes
        # FP64 design-fit results into actual deployed-model achievements.
        rows.append(dict(stage=source["stage"], control=source["control"],
            source_path=source["source_path"], source_sha256=source["source_sha256"],
            status=payload.get("status"), payload=payload,
            scope="Recorded fitted-feature or frozen-encoder control; FP64 design and deployed errors remain separate"))
    return rows


def inference_rows(data):
    rows = []
    for trial in data["trials"]:
        timing = trial.get("inference_cost") or {}
        rows.append(dict(timing,trial_id=trial["trial_id"],family=trial.get("family"),variant_id=trial.get("variant_id"),
            seed=trial.get("seed"),target_kind=trial.get("target_kind"),phase=trial.get("phase"),
            status=timing.get("status","NOT_MEASURED"),source_path=trial.get("source_path"),source_sha256=trial.get("source_sha256"),
            source_record=f"/trials/{trial.get('source_trial_index',0)}/inference_cost"))
    return rows


def _plots(path, data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .report import style
    curves = {}
    for row in data["learning_curves"]:
        curves.setdefault(row.get("trial_id"), []).append(row)
    figures = []
    for trial in data["trials"]:
        ident = trial["trial_id"]
        history = sorted(curves.get(ident, []), key=lambda row:row.get("update", 0))
        if not history:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.0), constrained_layout=True)
        appearance = style(str(trial.get("family", "unknown")))
        appearance.pop("label", None)
        panels = [(("train_loss", "full_train_loss", "validation_loss"), "Update objective / scheduled full-bank objective"),
                  (("full_train_rms", "validation_rms", "full_train_maximum", "validation_maximum"), "Scheduled full-bank physical endpoint error"),
                  (("gradient_norm", "trunk_gradient_norm", "head_gradient_norm"), "Pre-clipping gradient norm")]
        for axis, (fields, label) in zip(axes, panels):
            any_values = False
            for index, field in enumerate(fields):
                aliases={"full_train_rms":("full_train_rms","train_rms"),
                         "full_train_maximum":("full_train_maximum","train_maximum"),
                         "validation_maximum":("validation_maximum","validation_max")}
                value=lambda row:next((row.get(key) for key in aliases.get(field,(field,)) if metrics.finite(row.get(key))),None)
                observed = [(r.get("update"), value(r)) for r in history
                            if metrics.finite(r.get("update")) and metrics.finite(value(r))]
                if not observed:
                    continue
                any_values = True
                x, y = zip(*observed)
                display = dict(appearance, alpha=max(.4, 1-index*.12))
                display["linestyle"] = ("-", "--", "-.", ":")[index]
                axis.plot(x, y, label=field, **display)
            axis.set_xlabel("Logical optimizer update")
            axis.set_ylabel(label)
            axis.grid(alpha=.15)
            if any_values:
                axis.set_yscale("symlog", linthresh=1e-12)
                axis.legend(fontsize=7)
            else:
                axis.text(.5, .5, "Not measured", ha="center", transform=axis.transAxes)
        axes[1].axhline(2e-5, color="black", lw=.6, ls=":", label="Pilot physical tolerance")
        fig.suptitle(f"{trial.get('target_kind')} | {trial.get('family')} | {trial.get('variant_id')} | seed {trial.get('seed')}\n"
                     f"{trial['status']} — {ident}; histories are paired observations", fontsize=10)
        filename = "learning-" + metrics.stable_id(ident) + ".png"
        fig.savefig(path/filename, dpi=150)
        plt.close(fig)
        figures.append(dict(trial_id=ident, path=filename,
            chart_source="learning_curves.csv", metric_source="diagnostic_metrics.json",
            source_records=[citation for row in history for citation in row.get("citations", [])]))
    measured = [r for r in inference_rows(data) if r["status"] == "MEASURED"
                and metrics.finite(r.get("complete_call_median_seconds")) and metrics.finite(r.get("model_error",{}).get("rms"))]
    targets = sorted({r["target_kind"] for r in measured})
    if targets:
        fig, axes = plt.subplots(1,len(targets),figsize=(5*len(targets),4.0),squeeze=False,constrained_layout=True)
        for axis,target in zip(axes[0],targets):
            families = set()
            for row in (r for r in measured if r["target_kind"] == target):
                appearance = style(str(row["family"]))
                family = row["family"]
                label = appearance.pop("label") if family not in families else None
                families.add(family)
                axis.scatter(row["complete_call_median_seconds"],row["model_error"]["rms"],
                    marker=appearance["marker"],color=appearance["color"],s=12,alpha=.65,label=label)
            axis.set_title(target,fontsize=9)
            axis.set_xlabel("Synchronized complete-call median (seconds)")
            axis.set_ylabel("Physical RMS on recorded validation query")
            axis.set_xscale("log");axis.set_yscale("symlog",linthresh=1e-12)
            axis.grid(alpha=.15);axis.legend(fontsize=6)
        fig.suptitle("Recorded single-query measurements; different query scopes, no matched-tolerance utility frontier",fontsize=9)
        fig.savefig(path/"inference-measurements.png",dpi=150)
        plt.close(fig)
        figures.append(dict(kind="inference_measurements",path="inference-measurements.png",chart_source="inference_costs.csv",
            scope="One prospectively canonical accepted validation query per registered trial; resident-input warmed calls"))
    write_json(path/"figure_index.json", dict(schema="tdn.response-diagnostic-figures/v1", figures=figures,
        missing=[trial["trial_id"] for trial in data["trials"] if trial["trial_id"] not in curves],
        bands="No confidence intervals; every displayed line is an observed trial history"))
    return figures


def _html(path, data, figures):
    summary, rows = data["summary"], data["trials"]
    serializable = []
    for row in rows:
        selected = row.get("fits", {}).get("selected_train", {})
        validation = row.get("fits",{}).get("selected_validation",{})
        serializable.append({key:row.get(key) for key in ("trial_id", "family", "target_kind", "variant_id", "seed", "phase", "status", "updates", "requested_updates", "parameters", "initialization_mse_reduction")} |
            dict(rms=selected.get("rms"), maximum=selected.get("maximum"), mean_absolute=selected.get("mean_absolute"), centered_rms=selected.get("centered_rms"),
                validation_rms=validation.get("rms"),validation_maximum=validation.get("maximum"),
                complete_call_median_seconds=(row.get("inference_cost") or {}).get("complete_call_median_seconds")))
    payload = json.dumps(serializable, allow_nan=False).replace("<", "\\u003c").replace("&", "\\u0026")
    figures_by_trial = {row["trial_id"]:row["path"] for row in figures if "trial_id" in row}
    figure_json = json.dumps(figures_by_trial).replace("<", "\\u003c")
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><title>Response diagnostic research</title>
<style>body{font:15px system-ui;margin:2rem;max-width:1600px;color:#162336}table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:.45rem;border-bottom:1px solid #ccd4dd;text-align:right}td:first-child,th:first-child{text-align:left}th{position:sticky;top:0;background:#eef2f7}input{padding:.6rem;width:26rem}a{color:#1560a4}img{max-width:100%}.notice{background:#eef2f7;padding:1rem}.missing{color:#667}</style>
<h1>Response diagnostic research</h1><p class="notice"><strong>Diagnostic review required; development is not authorized by this run.</strong>
The historical pilot outcome is unchanged. Physical error, improvement over initialization, numerical headroom and computation are separate measurements.
Missing values remain unavailable. Parents are the independent fields; horizons, phases and seeds are paired descendants.</p>
<p>SUMMARY</p><p><a href="trial_metrics.csv">Trial metrics CSV</a> · <a href="endpoint_queries.csv">Every endpoint query CSV</a> ·
<a href="learning_curves.csv">Declared learning observations CSV</a> · <a href="screening_queries.csv">Reference and precision CSV</a> ·
<a href="correction_metrics.csv">Gains and raw corrections CSV</a> · <a href="metric_legend.md">Metric legend</a> ·
<a href="fixed_control_metrics.csv">Fitted controls CSV</a> ·
<a href="fit_assessments.csv">Checkpoint and regime assessments CSV</a> · <a href="variant_outcomes.csv">Variant readiness CSV</a> ·
<a href="stage_costs.csv">Science-stage costs CSV</a> · <a href="allocation_costs.csv">Scheduler allocation snapshot CSV</a> ·
<a href="inference_costs.csv">Paired full-call benchmark CSV</a> ·
COSTFIGURE
<a href="diagnostic_metrics.json">Complete structured metrics</a></p>
<label>Filter family, variant, target or status: <input id="filter" placeholder="e.g. temporal or TIME_CENSORED"></label>
<p id="count"></p><table><thead id="head"></thead><tbody id="body"></tbody></table>
<h2 id="selected"></h2><img id="plot" alt="Selected trial learning curves"><p>Plots show observed histories, not confidence intervals. Normalized objectives from different scaling contracts are not directly comparable.</p>
<script>const rows=PAYLOAD,figures=FIGURES;const columns=['trial_id','family','target_kind','variant_id','seed','phase','status','updates','requested_updates','parameters','rms','maximum','validation_rms','validation_maximum','mean_absolute','centered_rms','initialization_mse_reduction','complete_call_median_seconds'];
const head=document.getElementById('head'),body=document.getElementById('body');const tr=document.createElement('tr');columns.forEach(k=>{const th=document.createElement('th');th.textContent=k;tr.appendChild(th)});head.appendChild(tr);
function render(){body.replaceChildren();const q=document.getElementById('filter').value.toLowerCase();const visible=rows.filter(r=>JSON.stringify(r).toLowerCase().includes(q));document.getElementById('count').textContent=visible.length+' / '+rows.length+' registered trials';visible.forEach(r=>{const tr=document.createElement('tr');columns.forEach(k=>{const td=document.createElement('td');const v=r[k];td.textContent=v===null||v===undefined?'NA':typeof v==='number'&&!Number.isInteger(v)?v.toPrecision(5):v;if(v===null||v===undefined)td.className='missing';tr.appendChild(td)});tr.addEventListener('click',()=>{document.getElementById('selected').textContent=r.trial_id;const p=document.getElementById('plot');if(figures[r.trial_id]){p.src=figures[r.trial_id];p.hidden=false}else{p.removeAttribute('src');p.hidden=true}});body.appendChild(tr)})}document.getElementById('filter').addEventListener('input',render);render();</script></html>"""
    cost_figure = '<a href="inference-measurements.png">Single-query timing/error plot</a> ·' if (path/"inference-measurements.png").is_file() else ""
    page = page.replace("SUMMARY", html.escape(json.dumps(summary, sort_keys=True))).replace("PAYLOAD", payload).replace("FIGURES", figure_json).replace("COSTFIGURE",cost_figure)
    (path/"index.html").write_text(page, encoding="utf-8")


def _contract(path):
    entries = []
    for name in ("index.html", "diagnostic_metrics.json", "trial_metrics.csv", "endpoint_queries.csv",
                 "learning_curves.csv", "screening_queries.csv", "correction_metrics.csv", "fixed_control_metrics.csv", "metric_definitions.json",
                 "fit_assessments.csv", "variant_outcomes.csv", "stage_costs.csv", "allocation_costs.csv", "inference_costs.csv"):
        file = path/name
        entry = dict(path="outputs/"+name, required=True, min_bytes=1,
            max_bytes=max(1, file.stat().st_size), sha256=hashlib.sha256(file.read_bytes()).hexdigest())
        if name.endswith(".csv"):
            entry["format"] = "csv"
        elif name.endswith(".json"):
            entry["format"] = "json"
        else:
            entry["format"] = "text"
        entries.append(entry)
    write_json(path/"contract.json", dict(version=1, outputs=entries))


def portable_export(project, destination, *, exclude_attempts=()):
    """Archive a standard Raven project with exact checksums and relative paths."""
    project, destination = Path(project), Path(destination)
    excluded = {metrics.relative_path(value) for value in exclude_attempts}
    members = []
    total_bytes = 0
    for file in sorted(project.rglob("*")):
        metrics._inside(project, file)
        if file.is_symlink():
            raise ValueError("Portable Raven export refuses symlinks")
        if not file.is_file():
            continue
        relative = metrics.relative_path(file.relative_to(project).as_posix())
        if relative == "portable_manifest.json" or any(relative.startswith(prefix+"/") for prefix in excluded):
            continue
        if file.stat().st_size > metrics.MAX_METADATA_BYTES:
            raise ValueError("Portable research file exceeds its bounded export size")
        hasher = hashlib.sha256()
        size = 0
        with file.open("rb") as stream:
            for chunk in iter(lambda:stream.read(1 << 20), b""):
                size += len(chunk)
                hasher.update(chunk)
        total_bytes += size
        if total_bytes > 2 << 30:
            raise ValueError("Portable research export exceeds its aggregate byte budget")
        members.append(dict(path=relative, bytes=size, sha256=hasher.hexdigest()))
    manifest = dict(schema="tdn.response-raven-portable/v1", files=members,
        excluded_attempts=sorted(excluded), replay_scope="Relevant research metadata, chart data and plots; no checkpoints or replay arrays",
        path_base="Archive root is a standalone Raven project; all inventory and output pointers are relative")
    if destination.exists():
        raise FileExistsError("Preserve completed portable exports")
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for member in members:
            archive.write(project/member["path"], member["path"])
        archive.writestr("portable_manifest.json", json.dumps(manifest, sort_keys=True, allow_nan=False)+"\n")
    receipt = dict(schema="tdn.response-raven-export-receipt/v1", path=destination.name,
        bytes=destination.stat().st_size, sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        included_files=len(members), excluded_attempts=sorted(excluded), replay_scope=manifest["replay_scope"])
    write_json(destination.with_suffix(".manifest.json"), receipt)
    verify_portable(destination)
    return receipt


def verify_portable(path):
    """Verify transport and every inventory pointer without extracting/executing."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Portable archive has duplicate members")
        if len(names) > 131072 or sum(info.file_size for info in archive.infolist()) > 2 << 30:
            raise ValueError("Portable archive exceeds its member or byte budget")
        for name in names:
            metrics.relative_path(name)
            info = archive.getinfo(name)
            if info.file_size > metrics.MAX_METADATA_BYTES:
                raise ValueError("Portable archive member exceeds its bounded file size")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Portable archive contains a symlink")
        manifest = json.loads(archive.read("portable_manifest.json"))
        expected = {member["path"]:member for member in manifest["files"]}
        if set(names) != set(expected) | {"portable_manifest.json"}:
            raise ValueError("Portable archive inventory differs")
        for name, entry in expected.items():
            hasher, size = hashlib.sha256(), 0
            with archive.open(name) as stream:
                for chunk in iter(lambda:stream.read(1 << 20), b""):
                    hasher.update(chunk)
                    size += len(chunk)
            if size != entry["bytes"] or hasher.hexdigest() != entry["sha256"]:
                raise ValueError("Portable archive source hash mismatch")
            if name.endswith("/run.json"):
                run = json.loads(archive.read(name))
                if run["schema"] != "raven.run/v1" or run["run_id"] != Path(name).parent.name:
                    raise ValueError("Portable Raven attempt identity differs")
                parent = Path(name).parent
                for field, pointer in run.get("paths", {}).items():
                    metrics.relative_path(pointer)
                    target = (parent/pointer).as_posix()
                    if field in ("outputs", "logs", "passports"):
                        present = any(n.startswith(target+"/") for n in expected)
                    else:
                        present = target in expected
                    if not present:
                        raise ValueError("Portable Raven declaration points to missing content")
            elif name.endswith("/logs.json"):
                index = json.loads(archive.read(name))
                if index.get("schema") != "raven.logs/v1":
                    raise ValueError("Portable log index has the wrong schema")
                for log in index.get("logs", []):
                    pointer = metrics.relative_path(log["path"])
                    if (Path(name).parent/pointer).as_posix() not in expected:
                        raise ValueError("Portable log index points outside included content")
    return manifest


def run(ctx):
    data, tables = metrics.collect(ctx)
    project, attempt = metrics.raven_root(ctx), _analysis_path(ctx)
    metrics._project_files(project)
    metrics._project_catalog(ctx)
    if attempt.exists():
        raise FileExistsError("Preserve completed diagnostic analysis attempts")
    output = attempt/"outputs"
    output.mkdir(parents=True)
    (attempt/"logs").mkdir()
    for table in tables:
        target = output/table["source_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(table["raw"])
    metrics.write_compact_json(output/"diagnostic_metrics.json", data)
    metrics.write_csv(output/"trial_metrics.csv", data["trials"])
    metrics.write_csv(output/"endpoint_queries.csv", data["endpoint_queries"])
    metrics.write_csv(output/"learning_curves.csv", data["learning_curves"])
    metrics.write_csv(output/"screening_queries.csv", screening_rows(data))
    metrics.write_csv(output/"correction_metrics.csv", correction_rows(data))
    metrics.write_csv(output/"fixed_control_metrics.csv", fixed_control_rows(data),
        columns=["stage", "control", "status", "scope", "source_path", "source_sha256", "payload"])
    metrics.write_csv(output/"fit_assessments.csv",data["assessments"]["cells"])
    metrics.write_csv(output/"variant_outcomes.csv",data["assessments"]["variant_outcomes"])
    metrics.write_csv(output/"stage_costs.csv",data["costs"]["stage_bodies"])
    metrics.write_csv(output/"allocation_costs.csv",data["costs"]["scheduler"]["allocation_rows"])
    metrics.write_csv(output/"inference_costs.csv",inference_rows(data))
    write_json(output/"diagnostic_decision.json",dict(decision="DIAGNOSTIC_REVIEW_REQUIRED",eligible_for_development=False,
        historical_pilot_gate=data["assessments"]["historical_gate"],assessments=data["assessments"],summary=data["summary"]))
    _definition_files(output)
    figures = _plots(output, data)
    _html(output, data, figures)
    _contract(output)
    now = time.time()
    summary = dict(schema="raven.summary/v1", id=attempt.name, name="TDN/response-diagnostic/research-analysis",
        state="COMPLETED", project_id="TDN-response-diagnostic", experiment_id="response-diagnostic-analysis",
        attempt=1, exit_code=0, results=dict(data["summary"], decision="DIAGNOSTIC_REVIEW_REQUIRED",
            eligible_for_development=False, historical_pilot_gate=data["assessments"]["historical_gate"],
            prospective_variant_outcomes=data["assessments"]["variant_outcomes"]),
        metadata=dict(source_tree_sha256=data["source_tree_sha256"], protocol_sha256=digest(ctx.protocol),
            metric_definitions="outputs/metric_definitions.json", scientific_scope="Development diagnostics; no fresh confirmation",
            metric_lineage="outputs/metric_lineage.csv", overview_timestamp_scope="Posthoc analysis publication; step is registered-trial ordinal",
            runtime_scope="Aggregate analysis has no independent scheduler allocation", export_scope="Relevant research only"))
    write_json(attempt/"summary.json", summary)
    write_json(Path(ctx.path)/"diagnostic_summary.json", summary["results"])
    write_json(attempt/"run.json", dict(schema="raven.run/v1", project_id=summary["project_id"], run_id=attempt.name,
        experiment_id=summary["experiment_id"], attempt=1, name=summary["name"], state="COMPLETED",
        paths=dict(metrics="metrics.jsonl", summary="summary.json", log_index="logs.json", outputs="outputs",
            logs="logs", contract="outputs/contract.json"), metadata=summary["metadata"]))
    write_json(attempt/"logs.json", dict(schema="raven.logs/v1", run_id=attempt.name,
        logs=[dict(id="analysis", path="logs/research.log", label="Research analysis summary", group="Research")]))
    (attempt/"logs"/"research.log").write_text(json.dumps(summary["results"], sort_keys=True)+"\n", encoding="utf-8")
    row = metrics.metric_record({k:v for k,v in data["summary"].items() if metrics.finite(v)}, t=now, phase="diagnostic-summary")
    records, lineage = [row], []
    for ordinal, trial in enumerate(data["trials"], 1):
        selected = trial.get("fits", {}).get("selected_train", {})
        validation = trial.get("fits",{}).get("selected_validation",{})
        timing = trial.get("inference_cost") or {}
        phase = "/".join(str(trial.get(key, "NA")) for key in ("target_kind", "family", "variant_id", "seed"))[:160]
        values = dict(trial_observed=int(trial["status"] != "MISSING"),
            trial_completed_updates=int(trial["status"] == "COMPLETED_UPDATES"),
            trial_updates=trial.get("updates"), trial_requested_updates=trial.get("requested_updates"),
            trial_parameters=trial.get("parameters"), trial_examples_seen=trial.get("examples_seen"),
            selected_train_physical_mse=selected.get("mse"), selected_train_physical_rms=selected.get("rms"),
            selected_train_sampled_maximum=selected.get("maximum"),
            selected_train_mean_absolute_error=selected.get("mean_absolute"),
            selected_train_centered_rms=selected.get("centered_rms"),
            selected_validation_physical_rms=validation.get("rms"),selected_validation_sampled_maximum=validation.get("maximum"),
            complete_call_median_seconds=timing.get("complete_call_median_seconds"),
            anchor_median_seconds=timing.get("anchor_median_seconds"),model_to_anchor_time_ratio=timing.get("model_to_anchor_time_ratio"),
            encoding_seconds=timing.get("encoding_seconds"),cached_decode_median_seconds=timing.get("cached_decode_median_seconds"),
            selected_train_initialization_mse_reduction=trial.get("initialization_mse_reduction"))
        records.append(metrics.metric_record(values, t=now, step=ordinal, phase=phase))
        lineage.append(dict(metric_ordinal=ordinal, trial_id=trial["trial_id"], phase=phase,
            source_path=trial.get("source_path"), source_sha256=trial.get("source_sha256"),
            timestamp_scope="Posthoc analysis publication epoch; step is registered-trial ordinal, not optimizer update"))
    (attempt/"metrics.jsonl").write_text("".join(json.dumps(record, allow_nan=False)+"\n" for record in records), encoding="utf-8")
    metrics.write_csv(output/"metric_lineage.csv", lineage)
    current = metrics.attempt_path(ctx).relative_to(project).as_posix()
    receipt = portable_export(project, Path(ctx.path)/BUNDLE, exclude_attempts=[current])
    (Path(ctx.path)/"raven_paths.txt").write_text(
        f"Raven project: ../raven\nPortable bundle: {BUNDLE}\n"
        f"Portable analysis: runs/{attempt.name}/outputs/index.html\n"
        f"Native command: :project /absolute/path/to/this/campaign/raven\n"
        f"Manual selection: :run select {attempt.name}\n", encoding="utf-8")
    mechanism = ctx.unit.get("mechanism", "M3")
    if mechanism not in ctx.protocol["mechanisms"]:
        mechanism = "M3" if "M3" in ctx.protocol["mechanisms"] else next(iter(ctx.protocol["mechanisms"]))
    ctx.record("diagnostic/portable-research-report", [mechanism],
        metrics=dict(data["summary"], portable_files=receipt["included_files"], figures=len(figures)),
        checks=[check("portable_inventory_hashes", 1, 1, "eq", category="correctness"),
                check("registered_trials_visible", len(data["trials"]), len(ctx.protocol.get("trials", {})), "eq", category="gap"),
                check("diagnostic_not_confirmation", None, None, category="math",
                      reason="Development diagnostics and complete reports do not establish a physical theorem or generalization")],
        evidence=["diagnostic_summary.json", BUNDLE, "raven_paths.txt"])
    return dict(raven_project="../raven", portable_bundle=BUNDLE, portable_sha256=receipt["sha256"],
        analysis_attempt=attempt.name, research_summary=data["summary"], eligible_for_development=False,
        report_scope="Relevant registered research metrics, full denominators and source hashes; no binary replay")
