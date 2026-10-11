"""Source-bound Response atlas: observations, missing evidence and separate claims.

This module never selects a checkpoint or changes a scientific result. Its
tables retain all declared trials and their exact JSON pointers. Display ranges
are observed ranges, not statistical intervals; missing measurements stay gaps.
"""
from __future__ import annotations

from collections import defaultdict
import gzip
import hashlib
import html
import io
import json
import math
from pathlib import Path

from tdn.analysis.frontier.core import check
from tdn.analysis.resolve.report import observed_ranges, quantitative_axis

SCHEMA = "tdn.response-atlas/v1"
GUIDANCE = ("Ours: triangles and thin solid lines; Theirs: circles and dashed lines; "
    "analytic/shared controls: squares and dash-dot lines. Lower error, time and memory are better. "
    "Higher accepted coverage is better. Continuous shaded learning envelopes show observed "
    "minimum–maximum ranges, not confidence intervals. Missing observations are never zero.")
FILES = {"summary.json", "rows.jsonl", "learning_curves.jsonl", "learning_curves.json",
    "training_results.json", "trial.json", "fixed_features.json", "gate.json", "selection.json",
    "endpoint_metrics.json", "rollout_metrics.json", "sensitivity_metrics.json", "timing.json",
    "reference_history.json", "dataset.json", "compression_results.json", "closure_results.json",
    "claims.json", "audit.json", "probes.json", "mathematical_audit.json", "budget.json",
    "costs.json", "mean_paths.json", "source_diagnostics.json", "reference_summary.json",
    "query_registry.json", "aggregate.json", "dense_query_metrics.json", "compression_diagnostics.json",
    "compression_learning.json", "numerical_controls.json", "performance.json", "spatial_examples.json",
    "fixed_encoder.json", "phase_twin_metrics.json", "numerical_control_summary.json", "estimands.json", "decomposition.json",
    "symmetry_metrics.json", "parameter_metrics.json", "spectral_stress.json",
    "accuracy_cost_frontiers.json", "amortization.json", "offline_costs.json", "model_resources.json",
    "diagnostic_teacher_costs.json", "timing_coverage.json", "profiling.json", "closure_attribution.json",
    "quadrature_refinement.json"}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def style(method):
    """Method ownership is explicit; an adapted FNO is not a paper reproduction."""
    import colorsys
    method = str(method)
    family = method.split("::",1)[0]
    if family in ("endpoint", "residual_fno", "endpoint_corrector", "shared_endpoint"):
        role, marker, line = "Shared control", "s", "-."
    elif "fno" in method:
        role, marker, line = "Theirs (adapted)", "o", "--"
    elif any(v in method for v in ("coupled", "temporal", "learned", "decoder_optimized", "noise_aware_observer")):
        role, marker, line = "Ours", "^", "-"
    else:
        role, marker, line = "Analytic/fitted control", "s", "-."
    hue = int(hashlib.sha256(method.encode()).hexdigest()[:8], 16) / 2**32
    return dict(label=f"{role}: {method}", marker=marker, linestyle=line, linewidth=.65,
                color=colorsys.hsv_to_rgb(hue, .7, .65), markersize=3)


def _read(base, name, expected):
    base, relative = Path(base), Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe Response atlas artifact")
    path = base / relative
    if path.is_symlink() or not path.resolve().is_relative_to(base.resolve()) or any(p.is_symlink() for p in path.parents):
        raise ValueError("Atlas reads require regular files inside their sealed source")
    if path.stat().st_size > 256 << 20:
        raise ValueError("Response atlas single-table read ceiling exceeded")
    raw = path.read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    if checksum != expected:
        raise ValueError(f"Sealed Response atlas source changed: {name}")
    def reject_constant(value):
        raise ValueError(f"Nonstandard JSON constant: {value}")
    parse = lambda s: json.loads(s, parse_constant=reject_constant)
    value = [parse(line) for line in raw.splitlines() if line] if name.endswith(".jsonl") else parse(raw)
    return value, dict(path=str(path), sha256=checksum, bytes=len(raw))


def collect(ctx):
    data = dict(schema=SCHEMA, profile=ctx.protocol["profile"], scope=ctx.protocol["scientific_scope"],
                sources=[], tables={}, stages=[], omissions=[], guidance=GUIDANCE)
    stages=ctx.protocol.get("stages",list(ctx.protocol["units"]))
    for stage in stages:
        unit=ctx.protocol["units"][stage]
        if stage == ctx.stage:
            continue
        base = ctx.prerequisites.get(stage)
        failure = getattr(ctx, "stage_failures", {}).get(stage, {})
        if not isinstance(failure, dict):
            failure = {"error": str(failure)}
        status = dict(stage=stage, kind=unit["kind"], status=failure.get("status", "NOT_MEASURED"),
                      seconds=failure.get("elapsed_seconds"), error=failure.get("error"),
                      timing_scope=failure.get("cost_scope","Sealed numerical stage body only"),
                      operational_source=failure.get("operational_source"))
        data["stages"].append(status)
        if base is None:
            data["omissions"].append(dict(stage=stage, reason="No verified completed prerequisite", **failure))
            continue
        base = Path(base)
        manifest_path = base / "science_manifest.json"
        if manifest_path.is_symlink():
            raise ValueError("Symlinked Response science manifest")
        manifest_raw = manifest_path.read_bytes()
        manifest = json.loads(manifest_raw)
        if manifest.get("schema") != "tdn.response-science/v1" or manifest.get("stage") != stage:
            raise ValueError("Response atlas source belongs to another program or stage")
        from tdn.research.protocol import digest
        if (manifest.get("protocol_sha256") != digest(ctx.protocol) or
                (base/"COMPLETED").read_text().strip() != digest(manifest)):
            raise ValueError("Response atlas requires its exact completed protocol seal")
        data["sources"].append(dict(source_id=f"{stage}/science_manifest.json", path=str(manifest_path),
            sha256=hashlib.sha256(manifest_raw).hexdigest(), bytes=len(manifest_raw)))
        for name, expected in manifest.get("artifacts", {}).items():
            # Explicit bounded tables only. Binary weights and arrays are not deserialized.
            if Path(name).name not in FILES:
                continue
            # Streamed curves are authoritative when both formats are declared.
            if name.endswith("learning_curves.json") and name + "l" in manifest["artifacts"]:
                continue
            value, source = _read(base, name, expected)
            key = f"{stage}/{name}"
            source["source_id"] = key
            data["sources"].append(source)
            data["tables"][key] = dict(stage=stage, kind=unit["kind"], filename=Path(name).name,
                target_kind=unit.get("target_kind"),grid=unit.get("grid"),payload=value)
            if name == "summary.json":
                status.update(status=value.get("status", "NOT_MEASURED"), seconds=value.get("elapsed_seconds"),
                    scientific_outcome=value.get("scientific_outcome"), experiments=value.get("experiment_count"))
    return data


def accounting_snapshot(ctx, data):
    """Snapshot scheduler costs separately; never sum allocation and step rows.

    Recovery origin allocations remain charged, including failed attempts.
    Snapshot files are operational evidence rather than numerical seals.
    Reporting-job accounting can lag and is visibly partial.
    """
    from .engine import ROOT
    base=ctx.path.parent.resolve();roots=[base]
    bridge=base/"recovery.json"
    if bridge.is_file() and not bridge.is_symlink() and bridge.stat().st_size<1<<20:
        raw=json.loads(bridge.read_text())
        for value in raw.get("origin_roots",[]):
            p=Path(value)
            if p.is_symlink() or not p.resolve().is_relative_to(ROOT.resolve()):
                raise ValueError("Accounting recovery origin must stay inside the project")
            roots.append(p.resolve())
    snapshots=[];rows=[];seen=set();missing=[]
    for root in dict.fromkeys(roots):
        candidates=sorted((root/"state").glob("scheduler-accounting-*.json"))
        if not candidates:missing.append(str(root));continue
        path=candidates[-1]
        if path.is_symlink() or path.stat().st_size>2<<20:
            raise ValueError("Unsafe or oversized accounting snapshot")
        raw=path.read_bytes();payload=json.loads(raw)
        if payload.get("schema")!="tdn.scheduler-accounting/v1":
            raise ValueError("Unexpected scheduler accounting schema")
        requested=set(payload.get("requested_job_ids",[]))
        for row in payload.get("records",[]):
            if row.get("record_kind")!="allocation":continue
            job=row["allocation_job_id"]
            if job not in requested or job in seen:
                if job in seen:continue
                raise ValueError("Accounting contains an undeclared allocation")
            seen.add(job);rows.append(dict(row,origin_run=str(root)))
        source=dict(source_id="accounting/"+root.name+"/"+path.name,path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
        data["sources"].append(source)
        snapshots.append(dict(source_id=source["source_id"],status=payload.get("status"),
            all_allocations_terminal=payload.get("all_allocations_terminal"),requested_job_ids=sorted(requested)))
    complete=bool(snapshots) and not missing and all(s["status"]=="RECORDED" and s["all_allocations_terminal"] for s in snapshots)
    elapsed=[r.get("elapsed_seconds") for r in rows]
    cpu=[r.get("allocated_cpu_seconds") for r in rows]
    cost=dict(schema="tdn.response-campaign-cost/v1",scope="Scheduler allocations, including origin attempts; steps excluded from totals",
        status="COMPLETE_SNAPSHOT" if complete else "PARTIAL_OR_NOT_MEASURED",snapshots=snapshots,rows=rows,
        missing_snapshots=missing,observed_allocation_seconds=sum(v for v in elapsed if finite(v)) if rows else None,
        complete_allocation_seconds=sum(elapsed) if complete and all(finite(v) for v in elapsed) else None,
        observed_allocated_cpu_seconds=sum(v for v in cpu if finite(v)) if rows else None,
        energy_joules=None,currency_cost=None,
        limitations="Allocation-seconds are not actual compute utilization or CPU+GPU hours. Current reporting allocation may be partial; later collect refreshes operational accounting, not this sealed snapshot.")
    return cost


def verify_sources(data):
    for source in data["sources"]:
        path = Path(source["path"])
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError(f"Response source changed during rendering: {source['source_id']}")


def records(data, names):
    """Yield source pointers with raw rows; no inferred independent sample count."""
    for source_id, table in data["tables"].items():
        if table["filename"] not in names:
            continue
        payload = table["payload"]
        if isinstance(payload, list):
            rows, prefix, singleton = payload, "", False
        else:
            key = next((k for k in ("rows", "trials", "results", "cases", "comparisons", "records") if isinstance(payload.get(k), list)), None)
            rows, prefix, singleton = (payload[key], "/" + key, False) if key else ([payload], "", True)
        for i, row in enumerate(rows):
            if isinstance(row, dict):
                yield dict(row=row, source_id=source_id, json_pointer="" if singleton else prefix+f"/{i}")


def _value(row, keys):
    for key in keys:
        if finite(row.get(key)):
            return row[key]
        if isinstance(row.get("metrics"), dict) and finite(row["metrics"].get(key)):
            return row["metrics"][key]
        for nested in ("reference", "selected_validation", "final_train", "cost", "deployed"):
            if isinstance(row.get(nested), dict) and finite(row[nested].get(key)):
                return row[nested][key]
    return None


def _method(row):
    if row.get("method_id") or row.get("method"):
        return str(row.get("method_id",row.get("method")))
    family=str(row.get("family",row.get("model_id","unspecified")))
    return family+"::"+str(row["variant_id"]) if row.get("variant_id") else family


def aggregate(ctx):
    """Execute frozen estimands, including absent predictions in denominators.

    The declared reference units supply expected parents and query roles. An
    incomplete evaluation cannot silently shrink a comparison to its survivors.
    A development signal is never relabeled as a confirmation result.
    """
    from .statistics import compare_endpoints, claims_payload
    from .evaluation import _role
    from tdn.runtime.metadata import write_json
    data = collect(ctx)
    tables = data["tables"]
    groups = {}
    missing = []
    excluded_panels = []
    for stage, unit in ctx.protocol["units"].items():
        if unit["kind"] != "evaluate":
            continue
        if unit.get("descriptive_only") or unit.get("panel") == "grid_relative_stress":
            excluded_panels.append(dict(stage=stage, panel=unit.get("panel", "declared_descriptive"),
                reason="Predeclared descriptive stress panel; excluded from primary and secondary formal parent-query estimands",
                raw_endpoint_table_present=stage+"/endpoint_metrics.json" in tables))
            continue
        target = unit["target_kind"]
        refs = [ctx.protocol["units"][name] for name in unit.get("data_stages", [])]
        for ref in refs:
            grid = ref.get("grid", ctx.protocol["train_grid"])
            grid = [grid]*ctx.protocol.get("dimension", 2) if isinstance(grid, int) else list(grid)
            key = (target, tuple(grid))
            group = groups.setdefault(key, dict(parents=set(), roles=set(), rows=[], stages=set(), verified=set()))
            group["parents"].update(ref.get("parent_ids", []))
            for h in ref.get("horizons", []):
                for steps in ctx.protocol["evaluation"]["frozen_schedules"]:
                    for phase in ref.get("phase_variants", ctx.protocol.get("phase_variants", ["original"])):
                        group["roles"].add(_role(dict(target_kind=target, grid=grid, horizon=h, phase_variant=phase), steps))
            group["stages"].add(stage)
        registry_table = tables.get(stage+"/query_registry.json")
        endpoints = tables.get(stage+"/endpoint_metrics.json")
        if registry_table is None or endpoints is None:
            missing.append(dict(stage=stage, reason="Missing verified query registry or endpoints"))
            continue
        registered = registry_table["payload"]
        if registered.get("target_kind") != target or not registered.get("frozen_before_evaluation"):
            raise ValueError("Evaluation registry target/freeze identity differs")
        if list(registered.get("seeds", [])) != ctx.protocol["seeds"]:
            raise ValueError("Evaluation registry changed the paired seed menu")
        intended_parents = {p for ref in refs for p in ref.get("parent_ids", [])}
        if set(registered.get("parents", [])) != intended_parents:
            raise ValueError("Evaluation registry changed the declared parent denominator")
        intended_roles = {role for group in groups.values() if stage in group["stages"] for role in group["roles"]}
        if set(registered.get("query_roles", [])) != intended_roles:
            raise ValueError("Evaluation registry changed the declared physical query-role menu")
        methods = set(registered.get("methods", []))
        if len(methods) != len(registered.get("methods", [])):
            raise ValueError("Evaluation registry repeats a method identity")
        expected_seed_map = registered.get("expected_seed_map", {})
        if set(expected_seed_map)-methods:
            raise ValueError("Evaluation seed registry contains an undeclared method")
        for method, declared_seeds in expected_seed_map.items():
            if len(set(declared_seeds)) != len(declared_seeds) or not set(declared_seeds) <= set(ctx.protocol["seeds"]):
                raise ValueError("Evaluation method seed registry is invalid")
        for pair in ctx.protocol["evaluation"].get("comparisons", []):
            if pair.get("targets") and target not in pair["targets"]:
                continue
            for method in (pair["candidate"], pair["comparator"]):
                wanted = pair.get("seeds", ctx.protocol["seeds"])
                if method not in methods or not set(wanted) <= set(expected_seed_map.get(method, [])):
                    missing.append(dict(stage=stage, method=method, reason="Declared comparison method/seed missing from pre-inference registry; denominator retained"))
        for key, group in groups.items():
            if stage in group["stages"]:
                group["verified"].add(stage)
        registered_keys = set()
        for row in endpoints["payload"].get("rows", []):
            key = (target, tuple(row["grid"]))
            if key not in groups or row.get("parent_id") not in groups[key]["parents"]:
                raise ValueError("Endpoint lies outside the declared equation/grid/parent scope")
            if row.get("query_role") not in registered.get("query_roles", []):
                raise ValueError("Endpoint lies outside its pre-inference query registry")
            if row.get("method") not in methods or row.get("seed") not in expected_seed_map.get(row.get("method"), []):
                raise ValueError("Endpoint method/seed lies outside its pre-inference registry")
            if row.get("target_kind") != target or row.get("norm") != registered.get("norm"):
                raise ValueError("Endpoint physical equation/norm differs from its registry")
            endpoint_key = (row["parent_id"], row["query_role"], row["method"], row["seed"])
            if endpoint_key in registered_keys:
                raise ValueError("Duplicate endpoint in the pre-inference registry")
            registered_keys.add(endpoint_key)
            groups[key]["rows"].append(dict(row, query_id=row["query_role"]))
        expected_count = len(intended_parents)*len(intended_roles)*sum(len(v) for v in expected_seed_map.values())
        if len(registered_keys) != expected_count:
            missing.append(dict(stage=stage, reason="Missing declared endpoint rows; failures must be explicit rows", expected_rows=expected_count, observed_rows=len(registered_keys)))
    comparisons = ctx.protocol["evaluation"].get("comparisons", [])
    if any(not isinstance(c, dict) or not c.get("candidate") or not c.get("comparator") for c in comparisons):
        raise ValueError("Primary comparisons must name explicit predeclared candidate/comparator methods")
    outcomes = []; estimands = []; coverage = []
    for (target, grid), group in sorted(groups.items()):
        registry = dict(parents=sorted(group["parents"]), query_ids=sorted(group["roles"]),
            seeds=ctx.protocol["seeds"], target_kind=target,
            norm="nodal_rms" if target == "fd_same_grid" else "trigonometric_interpolant_rms",
            primary_tolerance=ctx.protocol["evaluation"]["rms_target"],
            maximum_tolerance=ctx.protocol["evaluation"]["max_target"],
            minimum_parents=16, draws=2000, alpha=.05,
            analysis_kind="confirmation" if ctx.protocol["profile"] == "confirmation" else "development",
            frozen_before_evaluation=group["stages"] == group["verified"])
        if not registry["parents"] or not registry["query_ids"]:
            raise ValueError("Evaluation is missing its declared field or horizon menu")
        coverage.append(dict(target_kind=target, grid=list(grid), expected_parents=len(group["parents"]),
            expected_stages=len(group["stages"]), verified_stages=len(group["verified"]), observed_rows=len(group["rows"])))
        for pair in comparisons:
            if pair.get("targets") and target not in pair["targets"]:
                continue
            if pair.get("grids") and grid[0] not in pair["grids"]:
                continue
            # Missing confirmation units still produce explicit unavailable claims.
            usable_registry = dict(registry)
            primary_pair = pair.get("scope", "primary" if pair.get("primary", True) else "secondary") == "primary"
            if "scope" in pair and "primary" in pair and primary_pair != bool(pair["primary"]):
                raise ValueError("Comparison has conflicting primary/exploratory declarations")
            if pair.get("seeds"):
                if not set(pair["seeds"]) <= set(registry["seeds"]):
                    raise ValueError("Comparison seed menu is outside the protocol")
                usable_registry["seeds"] = list(pair["seeds"])
            if primary_pair and set(usable_registry["seeds"]) != set(registry["seeds"]):
                raise ValueError("Primary comparisons must retain the full paired seed menu; one-seed probes are exploratory")
            if registry["analysis_kind"] == "confirmation" and not registry["frozen_before_evaluation"]:
                usable_registry["frozen_before_evaluation"] = True
                usable_registry["incomplete_frozen_execution"] = True
            paired_rows = [r for r in group["rows"] if r["seed"] in usable_registry["seeds"]]
            result = compare_endpoints(paired_rows, pair["candidate"], pair["comparator"], usable_registry)
            result.update(grid=list(grid), primary=primary_pair,
                comparison_id=pair.get("id",pair["candidate"]+"/"+pair["comparator"]))
            outcomes.append(result); estimands.append(dict(comparison=pair, grid=list(grid), registry=usable_registry))
    primary = claims_payload([o for o in outcomes if o["primary"]])
    exploratory = claims_payload([o for o in outcomes if not o["primary"]])
    for claim in exploratory["comparisons"]:
        claim["inference_scope"] = "EXPLORATORY; excluded from primary claim family"
        claim["verdict"] = "EXPLORATORY_SIGNAL" if claim["composite_advantage_supported"] else "EXPLORATORY_UNRESOLVED_OR_NO_ADVANTAGE"
    primary["exploratory_comparisons"] = exploratory["comparisons"]
    complete = bool(groups) and not missing and all(g["stages"] == g["verified"] for g in groups.values())
    selections = [table["payload"] for table in tables.values() if table["filename"] == "selection.json"
                  and table["payload"].get("selection_contract")]
    selection = selections[-1] if selections else {}
    # Selection must cover the declared trial roster, including groups whose
    # training job never emitted an artifact. Observed groups cannot define it.
    def method_identity(item):
        return item.get("method_id", f"{item['family']}::{item.get('variant_id','base')}::n{item.get('data_count',ctx.protocol.get('parent_counts',{}).get('train',0))}")
    declared_trials = ctx.protocol.get("trials", {})
    declared_trials = list(declared_trials.values()) if isinstance(declared_trials, dict) else declared_trials
    expected_roster = {(t["target_kind"], method_identity(t), t["seed"]) for t in declared_trials if t.get("phase") in ("main", "data_efficiency")}
    selected_roster = {(t["target_kind"], method_identity(t), t["seed"]) for t in selection.get("selected_models", [])}
    missing_roster = sorted(expected_roster-selected_roster)
    readiness = ctx.protocol.get("selection_source") or {}
    eligible = (ctx.protocol["profile"] == "development" and complete and
                readiness.get("gate_decision") == "CONTINUE" and bool(selection.get("selected_models")) and
                not selection.get("missing") and bool(expected_roster) and not missing_roster)
    result = dict(schema="tdn.response-aggregate/v1", freeze_eligible=eligible,
        freeze_eligibility_scope="Complete registered development and credible trainability controls; favorable results are not required to freeze a claim",
        claims=primary, estimands=estimands, coverage=coverage, missing=missing, excluded_panels=excluded_panels,
        expected_selection_groups=len(expected_roster), missing_selection_groups=[dict(target_kind=t, method=m, seed=s) for t,m,s in missing_roster],
        scope=ctx.protocol["scientific_scope"], scientific_success=any(c["composite_advantage_supported"] for c in primary["comparisons"]),
        energy_joules=None, currency_cost=None, resource_cost_scope="Use measured timing plus scheduler accounting; absent instrumentation remains NA")
    verify_sources(data)
    ctx.path.mkdir(parents=True, exist_ok=True)
    write_json(ctx.path/"aggregate.json", result); write_json(ctx.path/"claims.json", primary)
    write_json(ctx.path/"estimands.json", estimands); write_json(ctx.path/"sources.json", data["sources"])
    ctx.record("aggregate/frozen-comparisons", ["M6" if ctx.protocol["profile"] != "confirmation" else "M7"],
        metrics=dict(primary_comparisons=len(primary["comparisons"]), registered_groups=len(groups), missing_evaluations=len(missing)),
        checks=[check("complete-evaluation-inventory", complete, True, "eq", category="correctness"),
            check("composite-advantage", result["scientific_success"] if primary["comparisons"] else None, True, "eq", category="gap",
                  reason="Every comparison retains physical target, reference uncertainty, tails, coverage and complete denominators")],
        evidence=["aggregate.json", "claims.json", "estimands.json", "sources.json"])
    return result


def _save(fig, root, name, caption, figures, points, scope):
    import matplotlib.pyplot as plt
    for ax in fig.axes:
        quantitative_axis(ax, "x"); quantitative_axis(ax, "y")
        ax.grid(alpha=.16, linewidth=.4)
    fig.suptitle(caption, fontsize=11)
    fig.tight_layout(rect=(0, .05, 1, .94))
    fig.text(.01, .012, scope + " · observed ranges ≠ confidence intervals · missing ≠ zero", fontsize=7)
    for suffix in ("png", "pdf"):
        fig.savefig(root/f"{name}.{suffix}", dpi=400, bbox_inches="tight")
    plt.close(fig)
    # Raw tables are stored once. Repeating complete endpoint/spectral records
    # in every chart would recreate the multi-gigabyte review export problem.
    references=[]
    for point in points:
        if "source_id" in point:
            row=point.get("row",{})
            references.append(dict(source_id=row.get("original_source_id",point["source_id"]),
                json_pointer=row.get("original_json_pointer",point.get("json_pointer","")),
                **{k:point[k] for k in ("array_path","array_sha256") if k in point}))
        else:references.append(point)
    figures.append(dict(name=name, png=f"{name}.png", pdf=f"{name}.pdf", caption=caption,
                        plotted_records=references, guidance=GUIDANCE))


def _missing(ax, message="NOT_MEASURED: no verified eligible observations"):
    ax.text(.5,.5,message,transform=ax.transAxes,ha="center",va="center",wrap=True,color="#666666")


def _architecture(root, figures, scope):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(13, 5)); ax.set_axis_off()
    boxes = [(0,.55,"Current state + known physics\nNo future truth at inference"),
        (.30,.80,"Declared physical reference\ncentered path a(t)"),
        (.30,.32,"Learned phase-sensitive encoder\ncache valid for this state only"),
        (.61,.64,"Learned mean path m(t)\nmean-dependent propagator"),
        (.61,.18,"Learned centered source\nrank J, temporal coefficients"),
        (.91,.45,"Integrated response + reference\nPredicted field at query time")]
    artists = []
    for x,y,label in boxes:
        artists.append(ax.text(x,y,label,ha="center",va="center",fontsize=9,
                bbox=dict(boxstyle="round,pad=.6",fc="#EDF5FA",ec="#557488",lw=.7)))
    fig.canvas.draw()
    for start,end in [(0,1),(0,2),(2,3),(2,4),(1,5),(3,5),(4,5)]:
        a,b=boxes[start],boxes[end]
        props = dict(arrowstyle="->",lw=.7,color="#557488",
                     patchA=artists[start].get_bbox_patch(), patchB=artists[end].get_bbox_patch())
        if (start,end)==(1,5):
            props["connectionstyle"]="angle,angleA=0,angleB=90,rad=8"
        ax.annotate("",xy=(b[0],b[1]),xytext=(a[0],a[1]),arrowprops=props)
    ax.set_xlim(-.14,1.08);ax.set_ylim(-.05,1)
    _save(fig,root,"architecture","Response information flow — architecture diagram, not a measured success",figures,[],scope)


def _learning(data, root, figures):
    import matplotlib.pyplot as plt
    grouped=defaultdict(list)
    for point in records(data,{"learning_curves.jsonl","learning_curves.json"}):
        row=point["row"]
        key=(str(row.get("target_kind",row.get("track","unknown"))),str(row.get("phase","unknown")),_method(row))
        grouped[key].append(point)
    if not grouped: grouped[("unavailable","unavailable","unavailable")]=[]
    for panel,(key,points) in enumerate(sorted(grouped.items())):
        fig,axes=plt.subplots(1,2,figsize=(13,5))
        trials=defaultdict(list)
        for p in points: trials[(p["source_id"],p["row"].get("trial_id",p["row"].get("model_id","unknown")))].append(p["row"])
        for _,rows in trials.items():
            sty=style(_method(rows[0])); col=sty["color"]
            for seg in observed_ranges(rows,"train_loss",bins=64):
                axes[0].fill_between(seg["x"],seg["lower"],seg["upper"],color=col,alpha=.12)
                axes[0].plot(seg["x"],seg["lower"],color=col,lw=.55,alpha=.9)
                axes[0].plot(seg["x"],seg["upper"],color=col,lw=.55,alpha=.9)
            measured=[r for r in rows if finite(r.get("validation_loss")) and finite(r.get("elapsed_seconds"))]
            if measured: axes[1].plot([r["elapsed_seconds"] for r in measured],[r["validation_loss"] for r in measured],**sty,alpha=.8)
        for ax,title,x in zip(axes,("Training loss: observed low–high envelope","Validation observations: no imputed measurements"),("Optimizer update","Elapsed trial seconds")):
            ax.set(title=title,xlabel=x,ylabel="Family-specific normalized objective · lower within same contract")
            if ax.has_data():ax.set_yscale("symlog",linthresh=1e-12)
            else:_missing(ax)
        handles,labels=axes[1].get_legend_handles_labels();unique=dict(zip(labels,handles))
        if unique:axes[1].legend(unique.values(),unique.keys(),fontsize=6)
        _save(fig,root,f"learning-{panel:02d}",f"{key[0]} / {key[1]} / {key[2]} — optimizer trials remain separate",figures,points,data["scope"])


def _scatter(data, root, figures, name, files, xkeys, ykeys, xlabel, ylabel, caption):
    import matplotlib.pyplot as plt
    allpoints=list(records(data,set(files)));scopes=defaultdict(list)
    for point in allpoints:
        row=point["row"]
        scope=(str(row.get("target_kind",row.get("track","not_declared"))),str(row.get("grid","not_declared")),str(row.get("panel","primary/development")))
        scopes[scope].append(point)
    if not scopes:scopes[("unavailable","unavailable","unavailable")]=[]
    for j,(scope,points) in enumerate(sorted(scopes.items())):
        series=defaultdict(list)
        for point in points:
            row=point["row"];x=_value(row,xkeys);y=_value(row,ykeys)
            if x is None or y is None:continue
            series[_method(row)].append((x,y,point))
        # Page large ablation sets by identity, never by the observed ranking.
        # Every page uses the same axes so visual comparisons remain valid.
        methods=sorted(series);coordinates=[v for rows in series.values() for v in rows]
        def limits(axis):
            values=[v[axis] for v in coordinates]
            if not values:return None
            lo,hi=min(values),max(values);pad=max((hi-lo)*.04,abs(hi)*1e-6,1e-14)
            return lo-pad,hi+pad
        for page,start in enumerate(range(0,max(1,len(methods)),16)):
            fig,ax=plt.subplots(figsize=(11,5));drawn=[]
            for method in methods[start:start+16]:
                xy=series[method];sty=style(method)
                ax.scatter([v[0] for v in xy],[v[1] for v in xy],marker=sty["marker"],s=13,color=sty["color"],alpha=.45,label=sty["label"])
                drawn.extend(v[2] for v in xy)
            ax.set(xlabel=xlabel,ylabel=ylabel)
            if drawn:
                ax.set_yscale("symlog",linthresh=1e-12)
                ax.set_xlim(limits(0));ax.set_ylim(limits(1))
                ax.legend(fontsize=5,loc="upper left",bbox_to_anchor=(1.,1.))
            else:_missing(ax)
            suffix=f"-{page:02d}" if len(methods)>16 else ""
            _save(fig,root,f"{name}-{j:02d}{suffix}",f"{caption} · {scope[0]} / grid {scope[1]} / {scope[2]} · identity page {page+1}",figures,drawn,data["scope"])


def _spectral(data, root, figures):
    """Flatten exact nested band measurements, retaining original pointers."""
    derived = dict(data, tables={})
    for point in records(data, {"endpoint_metrics.json"}):
        row = point["row"]
        if not row.get("spectral_bands"):
            continue
        derived["tables"][point["source_id"]+point["json_pointer"]] = dict(filename="spectral-view", payload={"rows":[
            dict(band, family=_method(row), target_kind=row.get("target_kind"), grid=row.get("grid"),
                panel=row.get("panel","primary/development"),
                original_source_id=point["source_id"], original_json_pointer=point["json_pointer"]+f"/spectral_bands/{i}")
            for i,band in enumerate(row["spectral_bands"])]})
    _scatter(derived,root,figures,"spectral-errors",["spectral-view"],["lower_cycles_per_length"],
        ["coefficient_error_energy"],"Band lower edge · physical cycles per length",
        "Squared coefficient error · lower is better","Spectral error bands; exact raw frequencies, no smoothing of measurements")


def _claims(data, root, figures):
    import matplotlib.pyplot as plt
    rows=list(records(data,{"claims.json"}))
    groups=defaultdict(list)
    for point in rows:
        row=point["row"]
        if row.get("candidate"):
            groups[(row.get("target_kind"),str(row.get("grid")))].append(point)
    if not groups:groups[("unavailable","unavailable")]=[]
    for i,(scope,points) in enumerate(sorted(groups.items())):
        fig,ax=plt.subplots(figsize=(13,max(4,len(points)*.45)))
        for j,point in enumerate(points):
            row=point["row"];value=row.get("rms_geometric_ratio");ci=row.get("rms_ratio_interval")
            if not finite(value):continue
            sty=style(row["candidate"])
            ax.scatter([value],[j],marker=sty["marker"],s=20,color=sty["color"])
            if isinstance(ci,list) and len(ci)==2 and all(finite(v) for v in ci):
                ax.plot(ci,[j,j],color=sty["color"],lw=.7)
        ax.axvline(1.,lw=.6,color="black",linestyle="--");ax.axvline(.9,lw=.6,color="#777",linestyle=":")
        ax.set_yticks(range(len(points)),[f"{p['row']['candidate']} / {p['row']['comparator']}\n"
            f"{p['row']['complete_paired_parents']}/{p['row']['attempted_parents']} parents · {p['row']['verdict']}" for p in points],fontsize=6)
        ax.set(xlabel="Paired RMS ratio · lower is better · 1 = equal, 0.9 = declared effect target")
        if not points:_missing(ax)
        _save(fig,root,f"claims-{i:02d}",f"{scope[0]} / {scope[1]} — parent-bootstrap intervals; Holm and tail/coverage decisions in claim table",figures,points,data["scope"])


def _training_outcomes(data, root, figures):
    import matplotlib.pyplot as plt
    # Continuation chunks can carry the same trial. Latest completed update is
    # the display row; every chunk and every attempt stays in chart-data.
    trials={}
    for point in records(data,{"training_results.json"}):
        row=point["row"];ident=row.get("trial_id")
        if ident and (ident not in trials or row.get("updates",0)>=trials[ident]["row"].get("updates",0)):
            trials[ident]=point
    groups=defaultdict(lambda:defaultdict(int))
    for point in trials.values():
        row=point["row"]
        label=f"{row.get('target_kind')} / {_method(row)}"
        groups[label][row.get("status","NOT_MEASURED")]+=1
    fig,ax=plt.subplots(figsize=(13,max(4,min(24,len(groups)*.28))))
    labels=sorted(groups);statuses=sorted({s for counts in groups.values() for s in counts});left=[0]*len(labels)
    for status in statuses:
        values=[groups[label].get(status,0) for label in labels]
        ax.barh(labels,values,left=left,label=status,alpha=.7,height=.65)
        left=[a+b for a,b in zip(left,values)]
    if labels:ax.legend(fontsize=6);ax.tick_params(axis="y",labelsize=5)
    else:_missing(ax)
    ax.set(xlabel="Distinct declared trials · completion alone is not scientific benefit")
    _save(fig,root,"training-outcomes","Completed, censored and failed learning opportunities; initialization selection remains in raw trial tables",figures,list(trials.values()),data["scope"])


def _coverage(data, root, figures):
    import matplotlib.pyplot as plt
    groups=defaultdict(list)
    for point in records(data,{"endpoint_metrics.json"}):
        row=point["row"]
        groups[(row.get("target_kind"),str(row.get("grid")),row.get("regime","unspecified"),row.get("panel","primary/development"))].append(point)
    if not groups:groups[("unavailable","unavailable","unavailable","unavailable")]=[]
    for i,(scope,points) in enumerate(sorted(groups.items())):
        bymethod=defaultdict(list)
        for point in points:bymethod[_method(point["row"])].append(point["row"])
        fig,ax=plt.subplots(figsize=(12,max(4,min(16,len(bymethod)*.3))))
        for j,(method,rows) in enumerate(sorted(bymethod.items())):
            accepted=sum(bool(r.get("reference_accepted")) for r in rows)
            qualified=sum(r.get("accuracy_qualification")=="QUALIFIED" for r in rows)
            ax.barh(j,qualified/len(rows),color=style(method)["color"],height=.55,alpha=.65)
            ax.text(0,j,f" {qualified}/{len(rows)} qualified; {accepted} accepted teachers",va="center",fontsize=6)
        ax.set_yticks(range(len(bymethod)),[style(m)["label"] for m in sorted(bymethod)],fontsize=5)
        ax.set(xlim=(0,1),xlabel="Qualified observed queries / all observed rows · higher is better; registry-wide missing counts in claims")
        if not points:_missing(ax)
        _save(fig,root,f"coverage-{i:02d}",f"{scope[0]} / {scope[1]} / {scope[2]} / {scope[3]} — descriptive repeated-query coverage, not independent samples",figures,points,data["scope"])


def _spatial(data, root, figures):
    """Bounded, hash-verified representatives; no pickle or interpolated truth."""
    import numpy as np
    import matplotlib.pyplot as plt
    sources={v["source_id"]:v for v in data["sources"]}
    selected=[];seen=set()
    for point in records(data,{"spatial_examples.json"}):
        row=point["row"];source=sources[point["source_id"]]
        path=Path(source["path"]).parent
        table=data["tables"][point["source_id"]]
        key=(_method(row).split("::",1)[0],table.get("target_kind"),str(table.get("grid")))
        if key in seen or len(selected)>=16:continue
        seen.add(key)
        name=row.get("arrays","");relative=Path(name)
        if not name or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe spatial example path")
        array_path=path/relative
        manifest=json.loads((path/"science_manifest.json").read_text())
        expected=manifest.get("artifacts",{}).get(name)
        if expected is None or expected!=row.get("sha256") or array_path.is_symlink() or not array_path.resolve().is_relative_to(path.resolve()):
            raise ValueError("Spatial example is not bound to its science seal")
        raw=array_path.read_bytes()
        if len(raw)>16<<20 or hashlib.sha256(raw).hexdigest()!=expected:
            raise ValueError("Spatial example changed or exceeds display budget")
        with np.load(io.BytesIO(raw),allow_pickle=False) as arrays:
            fields={k:arrays[k].squeeze() for k in ("input","reference","prediction","error")}
        if any(v.ndim!=2 or v.size>128**2 for v in fields.values()):
            data["omissions"].append(dict(source_id=point["source_id"],reason="Spatial image requires a 2D field of at most128²; raw example retained"));continue
        selected.append((point,fields,array_path,expected))
        data["sources"].append(dict(source_id=point["source_id"]+"/"+name,path=str(array_path),sha256=expected,bytes=len(raw)))
    for i,(point,fields,array_path,expected) in enumerate(selected):
        fig,axes=plt.subplots(1,4,figsize=(16,4))
        state_values=np.concatenate([fields[k].ravel() for k in ("input","reference","prediction")])
        finite_values=state_values[np.isfinite(state_values)]
        lo,hi=(float(finite_values.min()),float(finite_values.max())) if len(finite_values) else (0.,1.)
        for ax,(name,values) in zip(axes,fields.items()):
            if not np.isfinite(values).any():_missing(ax,"FAILED / no finite field");continue
            if name=="error":
                limit=max(float(np.nanmax(np.abs(values))),1e-15)
                img=ax.imshow(values,origin="lower",interpolation="nearest",cmap="RdBu_r",vmin=-limit,vmax=limit)
            else:img=ax.imshow(values,origin="lower",interpolation="nearest",cmap="viridis",vmin=lo,vmax=hi)
            fig.colorbar(img,ax=ax,fraction=.045);ax.set_title(name);ax.set_xlabel("Grid x index");ax.set_ylabel("Grid y index")
        point=dict(point,array_path=str(array_path),array_sha256=expected)
        _save(fig,root,f"spatial-{i:02d}",f"{style(_method(point['row']))['label']} · signed error closer to zero is better · original grid pixels, no solution smoothing",figures,[point],data["scope"])


def _decomposition(data, root, figures):
    """Signed error budgets and paired interventions; no new claim or pooling.

    Component/cross-term ranges are deterministic reference-sensitivity
    estimates, never confidence intervals. Phase plots compare two descendants
    of one parent; a translation-only intervention is not phase discrimination.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # One source/parent, one common norm per figure. Summing component norms or
    # merging FD/Galerkin inner products would conceal error cancellation.
    panels = defaultdict(list)
    for point in records(data, {"decomposition.json"}):
        row = point["row"]
        panels[(point["source_id"], row.get("norm_target", "unspecified"))].append(point)
    for index, ((source_id, norm), points) in enumerate(sorted(panels.items())):
        payload = data["tables"][source_id]["payload"]
        fig, axes = plt.subplots(1, len(points), figsize=(7*len(points), 5), squeeze=False)
        for ax, point in zip(axes.ravel(), points):
            row = point["row"]; metric = row.get("metrics", {})
            terms = metric.get("component_squared_norms", {})
            component_names = ("discrete_solver_error", "spatial_discretization", "unresolved_scale_closure")
            labels = ["Solver/temporal squared norm", "Spatial discretization squared norm", "Closure squared norm"]
            values = [terms.get(key) for key in component_names]
            sensitivities = row.get("component_uncertainty_sensitivity", {})
            ranges = []
            for key in component_names:
                sensitivity = sensitivities.get(key, {})
                lo, hi = sensitivity.get("lower"), sensitivity.get("upper")
                ranges.append((lo*lo, hi*hi) if finite(lo) and finite(hi) else None)
            short = dict(discrete_solver_error="solver", spatial_discretization="spatial", unresolved_scale_closure="closure")
            for cross in metric.get("cross_terms", []):
                labels.append("2 × inner("+short.get(cross.get("left"), "?")+", "+short.get(cross.get("right"), "?")+")")
                value = cross.get("signed_twice_inner_product"); values.append(value)
                uncertainty = cross.get("estimated_twice_inner_product_sensitivity")
                ranges.append((value-uncertainty, value+uncertainty) if finite(value) and finite(uncertainty) else None)
            labels.append("Actual total squared error"); total = metric.get("total_squared_norm"); values.append(total)
            epsilon = row.get("primary_total_uncertainty", {}).get("uncertainty_rms")
            ranges.append((max(0.,math.sqrt(max(total,0.))-epsilon)**2, (math.sqrt(max(total,0.))+epsilon)**2)
                          if finite(total) and finite(epsilon) else None)
            sty = style(row.get("method", "unspecified")); observed = []
            for j, (value, interval) in enumerate(zip(values, ranges)):
                if not finite(value):
                    ax.text(.02,j,"NOT_MEASURED",transform=ax.get_yaxis_transform(),fontsize=6,color="#777")
                    continue
                observed.append(abs(value))
                if j == len(values)-1:
                    ax.scatter([value],[j],marker=sty["marker"],s=22,color=sty["color"],zorder=4)
                else:
                    color = "#647D96" if j < 3 else "#8B6EB0" if value < 0 else "#BA814F"
                    ax.barh(j,value,height=.53,alpha=.55,color=color,edgecolor=color,linewidth=.45,
                            hatch="//" if not row.get("reference_accepted",False) else None)
                if interval is not None:
                    ax.plot(interval,[j,j],color="#303030",linewidth=.55,zorder=5)
            ax.axvline(0,color="#444",linewidth=.5)
            if observed:
                positive=[v for v in observed if v>0]
                ax.set_xscale("symlog",linthresh=max(min(positive)*.15,1e-20) if positive else 1e-12)
            else:_missing(ax)
            ax.set_yticks(range(len(labels)),labels,fontsize=7);ax.invert_yaxis()
            ax.tick_params(axis="x",labelrotation=35,labelsize=7)
            ax.set(xlabel="Signed squared-error contribution · total lower is better",
                   title=sty["label"]+("" if row.get("reference_accepted") else "\nUNRESOLVED references: diagnostic only"))
        caption=(f"Signed error identity · {norm} · grid {payload.get('grid')} · h={payload.get('horizon')}\n"
                 "Same input: solver + spatial discretization + closure; thin ranges are reference sensitivity, not CI\n"
                 "Negative cross terms cancel; positive terms reinforce. Cross terms alone do not measure quality.")
        _save(fig,root,f"signed-decomposition-{index:02d}",caption,figures,points,data["scope"])

    # Spectral stress keeps target, grid, field intervention and horizon apart.
    stress = defaultdict(list)
    for point in records(data, {"spectral_stress.json"}):
        row=point["row"]; metric=row.get("metrics", {})
        key=(point["source_id"],str(metric.get("target_kind")),str(metric.get("grid")),
             str(row.get("kind")),str(metric.get("horizon")),str(row.get("cluster_parent_id")))
        stress[key].append(point)
    for index, (key, points) in enumerate(sorted(stress.items())):
        fig,axes=plt.subplots(1,2,figsize=(13,4.6));by_method=defaultdict(list);plotted=[];unavailable=0
        for point in points:
            row=point["row"]; method=str(row.get("method", "unknown"));family,_,step=method.partition("/steps-")
            try:count=int(step)
            except ValueError:continue
            by_method[family].append((count,point))
        for family, entries in sorted(by_method.items()):
            entries.sort(key=lambda entry:entry[0]);sty=style(family);x=[];rms=[];maximum=[]
            for count,point in entries:
                metric=point["row"].get("metrics",{});eligible=bool(metric.get("reference_accepted")) and bool(metric.get("prediction_finite"))
                values=(metric.get("rms"),metric.get("maximum",metric.get("error_max")))
                if not eligible or not all(finite(value) for value in values):
                    values=(np.nan,np.nan);unavailable+=1
                else:plotted.append(point)
                x.append(count);rms.append(values[0]);maximum.append(values[1])
            axes[0].plot(x,rms,**sty);axes[1].plot(x,maximum,**sty)
        rms_label = "Nodal RMS" if key[1] == "fd_same_grid" else "Physical interpolant RMS"
        for ax,label in zip(axes,(rms_label+" · lower is better","Sampled maximum error · lower is better")):
            ax.set(xlabel="Macrosteps · more steps cost more work",ylabel=label)
            ax.set_xscale("log",base=2);ax.set_xticks([1,2,4,8],[1,2,4,8]);ax.set_yscale("symlog",linthresh=1e-12)
            if not plotted:_missing(ax,"NOT_MEASURED: no accepted teacher/prediction pairs")
            else:ax.legend(fontsize=6)
        axes[1].text(.02,.02,f"{len(plotted)} measured cells; {unavailable} missing/unresolved cells remain gaps",
                     transform=axes[1].transAxes,fontsize=7)
        caption=f"{key[1]} / grid {key[2]} / {key[3]} / h={key[4]} — paired spectral stress, not a new independent parent"
        _save(fig,root,f"spectral-stress-{index:02d}",caption,figures,plotted,data["scope"])

    # Equal-power relative-triad descendants. Signed mean-error agreement is
    # exactly error in the predicted phase-induced mean change. It does not
    # establish that the underlying true phase effect was reference-resolved.
    pairs=defaultdict(dict)
    for point in records(data, {"endpoint_metrics.json"}):
        row=point["row"]
        if not str(row.get("phase_scope","")).startswith("relative-triad-phase"):
            continue
        variant=row.get("phase_variant")
        if variant not in ("original","phase_twin"):continue
        seed=None if row.get("measurement_reused_across_seeds") else row.get("seed")
        key=(point["source_id"],str(row.get("target_kind")),str(row.get("grid")),
             str(row.get("regime")),str(row.get("panel","primary/development")),
             row.get("parent_id"),_method(row),seed,row.get("horizon"),row.get("steps",1))
        pairs[key][variant]=point
    groups=defaultdict(list)
    for key,pair in pairs.items():groups[(key[1],key[2],key[3],key[4])].append((key,pair))
    for index,(scope,entries) in enumerate(sorted(groups.items())):
        fig,ax=plt.subplots(figsize=(8.5,6));plotted=[];limits=[];seen=set();missing=0
        for key,pair in entries:
            if set(pair)!={"original","phase_twin"}:missing+=1;continue
            left,right=(pair[v] for v in ("original","phase_twin"));a,b=left["row"],right["row"]
            if not all(row.get("reference_accepted") and row.get("prediction_finite") and finite(row.get("mean_error")) for row in (a,b)):
                missing+=1;continue
            method=key[6];sty=style(method);label=sty["label"] if method not in seen else None;seen.add(method)
            ax.scatter([a["mean_error"]],[b["mean_error"]],marker=sty["marker"],s=18,color=sty["color"],alpha=.6,label=label)
            limits.extend([a["mean_error"],b["mean_error"]]);plotted.extend([left,right])
        if limits:
            extent=max(max(abs(v) for v in limits)*1.1,1e-12)
            ax.plot([-extent,extent],[-extent,extent],color="#777",linewidth=.55,linestyle=":",label="Equal signed errors: correct mean-change difference")
            ax.set_xlim(-extent,extent);ax.set_ylim(-extent,extent);ax.legend(fontsize=6,loc="upper left",bbox_to_anchor=(1,1))
        else:_missing(ax,"NOT_MEASURED: no complete accepted relative-triad pairs")
        ax.axhline(0,color="#777",linewidth=.4);ax.axvline(0,color="#777",linewidth=.4)
        ax.set(xlabel="Original signed mean error · closer to zero is better",
               ylabel="Phase-twin signed mean error · closer to zero is better")
        ax.text(.02,.02,f"{len(plotted)//2} paired descendants; {missing} incomplete/unresolved pairs\nNear diagonal: smaller phase-change residual; origin: both means correct",
                transform=ax.transAxes,fontsize=7)
        _save(fig,root,f"phase-discrimination-{index:02d}",
            f"{scope[0]} / grid {scope[1]} / {scope[2]} / {scope[3]}\nRelative-triad phase test; repeated schedules/seeds are paired, not independent; no true-effect resolution claim",
            figures,plotted,data["scope"])


def _source_behavior(data, root, figures):
    """Recorded source factors on declared states, with no causal interpretation."""
    import matplotlib.pyplot as plt
    points=list(records(data,{"source_diagnostics.json"}))
    representatives=[];seen=set()
    for point in points:
        row=point["row"];key=(_method(row),row.get("target_kind"),str(row.get("grid")))
        if key not in seen and row.get("times"):
            seen.add(key);representatives.append(point)
    for i,point in enumerate(representatives):
        row=point["row"];times=row["times"];x=[v["time"] for v in times]
        fig,axes=plt.subplots(1,3,figsize=(16,4));sty=style(_method(row))
        axes[0].plot(x,[v["learned_mean"] for v in times],label=sty["label"],marker=sty["marker"],ms=3,lw=.6,color=sty["color"])
        axes[0].plot(x,[v["anchor_mean"] for v in times],label="Analytic reference path mean",lw=.6,ls="--",color="#555")
        axes[0].set(xlabel="Physical time",ylabel="Mean · direction alone does not measure quality")
        count=max((len(v.get("effective_source_coefficients",[])) for v in times),default=0)
        for channel in range(count):
            y=[v.get("effective_source_coefficients",[])[channel] if channel<len(v.get("effective_source_coefficients",[])) else float("nan") for v in times]
            axes[1].plot(x,y,lw=.6,marker="^",ms=2,label=f"Ours: source factor {channel}")
        axes[1].set(xlabel="Physical time",ylabel="Source factor · field/time units; larger is not better")
        for key,label in (("source_field_rank","Encoded fields"),("active_final_rank","Final weighted fields"),("observed_source_time_rank","Observed time-source span")):
            values=row.get(key,{}).get("singular_values",[])
            if values:axes[2].plot(range(1,len(values)+1),values,lw=.6,marker="^",ms=3,label="Ours: "+label)
        axes[2].set(xlabel="Singular-value index",ylabel="Physical source singular value · neither direction alone is better")
        axes[2].set_yscale("symlog",linthresh=1e-12)
        for ax in axes:
            if ax.has_data():ax.legend(fontsize=5)
            else:_missing(ax)
        _save(fig,root,f"source-behavior-{i:03d}",f"{row.get('target_kind')} / {row.get('grid')} / {_method(row)} — one declared state; source factors are not identifiable physical components",figures,[point],data["scope"])
        fig, axes=plt.subplots(1,2,figsize=(12,4))
        for key,label,line in (("reconstructed_physical_variance","Reconstructed field variance","-"),
                               ("clock_implied_variance","Mean-clock implied variance","--")):
            values=[v.get(key) if finite(v.get(key)) else float("nan") for v in times]
            if any(finite(v.get(key)) for v in times):
                axes[0].plot(x,values,label=label,lw=.6,ls=line,marker="^",ms=3)
        for key,label,line in (("mean_ode_residual","Reconstructed mean balance","-"),
                               ("mean_path_ode_residual","Learned mean-path balance","--")):
            values=[v.get(key) if finite(v.get(key)) else float("nan") for v in times]
            if any(finite(v.get(key)) for v in times):
                axes[1].plot(x,values,label=label,lw=.6,ls=line,marker="^",ms=3)
        axes[0].set(xlabel="Physical time",ylabel="Variance · agreement, not magnitude, is desired")
        axes[1].set(xlabel="Physical time",ylabel="Signed mean-ODE residual · closer to zero is better")
        axes[1].set_yscale("symlog",linthresh=1e-10)
        for ax in axes:
            if ax.has_data():ax.legend(fontsize=7)
            else:_missing(ax)
        _save(fig,root,f"source-balance-{i:03d}",f"{sty['label']} / {row.get('target_kind')} / {row.get('grid')}\nClock variance is absent when inapplicable; consistency penalties do not enforce an identity",figures,[point],data["scope"])
    if not representatives:
        fig,ax=plt.subplots(figsize=(11,4));_missing(ax,"No completed coupled-source inspection; see source_diagnostics.json for reasons")
        _save(fig,root,"source-behavior-unavailable","Learned source behavior: unavailable measurements remain gaps",figures,points,data["scope"])


def _quadrature(data, root, figures):
    import matplotlib.pyplot as plt
    grouped=defaultdict(list)
    for point in records(data,{"quadrature_refinement.json"}):
        row=point["row"]
        grouped[(point["source_id"],_method(row),row.get("target_kind"),str(row.get("grid")),row.get("query_id"))].append(point)
    for index,(key,points) in enumerate(sorted(grouped.items())):
        row=points[0]["row"];sty=style(key[1]);fig,axes=plt.subplots(1,2,figsize=(12,4))
        labels=[str(p["row"].get("variant","unspecified")) for p in points]
        for ax,metric,ylabel in ((axes[0],"difference_rms_vs_finest","Physical RMS sensitivity · smaller is better"),
                                (axes[1],"difference_maximum_vs_finest","Sampled maximum sensitivity · smaller is better")):
            values=[p["row"].get(metric) if finite(p["row"].get(metric)) else float("nan") for p in points]
            ax.plot(range(len(points)),values,**sty)
            ax.set_xticks(range(len(points)),labels,rotation=35,ha="right",fontsize=7)
            ax.set(xlabel="Source / mean quadrature nodes · more nodes cost more work",ylabel=ylabel)
            ax.set_yscale("symlog",linthresh=1e-12)
            if not any(finite(p["row"].get(metric)) for p in points):_missing(ax)
        _save(fig,root,f"decoder-quadrature-{index:03d}",f"{sty['label']} / {key[2]} / {key[3]} / h={row.get('horizon')}\nFrozen weights in FP64; difference from finest completed joint rule, not a convergence certificate",figures,points,data["scope"])


def _cost_details(data, root, figures):
    import matplotlib.pyplot as plt
    derived=dict(data,tables={})
    for point in records(data,{"accuracy_cost_frontiers.json"}):
        row=point["row"];selected=row.get("selected")
        if not isinstance(selected,dict):continue
        derived["tables"][point["source_id"]+point["json_pointer"]]=dict(filename="frontier-view",payload={"rows":[
            dict(selected,tolerance=row["tolerance"],panel="posthoc/"+row["category"],
                original_source_id=point["source_id"],original_json_pointer=point["json_pointer"])]})
    _scatter(derived,root,figures,"tolerance-frontier",["frontier-view"],["tolerance"],["median_seconds"],
        "Joint error tolerance · smaller is stricter","Complete call seconds · lower at matched tolerance is better",
        "Representative reference-informed frontier; not a deployable schedule or full-cohort claim")
    profiles=defaultdict(list)
    for point in records(data,{"profiling.json"}):
        r=point["row"];profiles[(r.get("target_kind"),str(r.get("grid")),r.get("model_id"))].append(point)
    for i,(scope,points) in enumerate(sorted(profiles.items(),key=lambda item:str(item[0]))):
        fig,axes=plt.subplots(1,2,figsize=(13,4))
        for ax,key,label in zip(axes,("self_cpu_seconds","self_device_seconds"),("CPU","GPU")):
            categories=defaultdict(float)
            for point in points:
                r=point["row"]
                if finite(r.get(key)):categories[r.get("category","other")]+=r[key]
            if categories:ax.barh(list(categories),list(categories.values()),height=.6,color="#4C7A8C",alpha=.7)
            else:_missing(ax,"No measured "+label+" profiler events")
            ax.set_xlabel(label+" self-event seconds · instrumented attribution, not deployment latency")
            ax.tick_params(axis="y",labelsize=6)
        _save(fig,root,f"profile-{i:03d}",f"{scope[0]} / {scope[1]} / {scope[2]} — do not add CPU and GPU clocks",figures,points,data["scope"])


def evidence_review(data):
    """A factual review of recorded outcomes; no outcome-driven model selection."""
    from collections import Counter
    stages=Counter(row["status"] for row in data["stages"])
    trials={}
    for point in records(data,{"training_results.json"}):
        row=point["row"];key=row.get("trial_id")
        if key and (key not in trials or row.get("updates",0)>=trials[key]["row"].get("updates",0)):
            trials[key]=point
    selected_initial=sum(p["row"].get("selected_update")==0 for p in trials.values())
    claims=[p for p in records(data,{"claims.json"}) if p["row"].get("candidate")]
    outcomes=Counter(p["row"].get("verdict","NA") for p in claims)
    gates=list(records(data,{"gate.json"}));cost=data.get("campaign_cost",{})
    answers=[
        ("What ran?",f"Declared stage outcomes: {dict(stages)}. Missing or unverified stages: {len(data['omissions'])}. The coverage table retains each reason."),
        ("Is the implementation supported?","The mathematical audit, precision tests and reference-refinement records are separate evidence. Computational completion is not a theorem, a reference certificate, or a GPU validation claim."),
        ("Did the models train?",f"Distinct recorded trials: {len(trials)}; initialization selected: {selected_initial}. Outcomes: {dict(Counter(p['row'].get('status','NOT_MEASURED') for p in trials.values()))}. Inspect requested versus achieved work, per-family objective scales and the tiny-fit gate. Normalized losses with different objectives are not cross-family quality scores."),
        ("Which mechanisms changed the errors?","Use the registered paired comparisons and separate mean, centered, maximum, phase, spectral and rollout panels. Source-energy compression and propagated error answer different questions. Signed error cross terms can create or remove cancellation."),
        ("How do learned controls compare?",f"Registered claim rows: {len(claims)}; verdicts: {dict(outcomes)}. Local FNO adaptations are not published-benchmark reproductions. Shared endpoint/residual-FNO aliases are one control, not independent competitors. Missing seeds, parents or references retain their denominators."),
        ("What is the complete cost?",f"Scheduler accounting: {cost.get('status','NOT_MEASURED')}. Timed calls, instrumented profiles, science-body elapsed time and allocation time are different layers. The cost tables retain accuracy eligibility, all attempted offline work and unavailable break-even inputs; energy and monetary cost require actual accounting."),
        ("What should continue?","Follow the recorded trainability gate for expansion, then the frozen RMS, tail and coverage decisions. An unresolved gate is not success, and a negative protected exploration result remains useful. A positive development comparison still requires the separate fresh confirmation protocol. This report neither retunes nor selects a new winner."),
    ]
    return dict(schema="tdn.response-review/v1",scope=data["scope"],answers=[dict(question=q,answer=a) for q,a in answers],
        trial_sources=[dict(source_id=p["source_id"],json_pointer=p["json_pointer"]) for p in trials.values()],
        claim_sources=[dict(source_id=p["source_id"],json_pointer=p["json_pointer"]) for p in claims],
        gate_sources=[dict(source_id=p["source_id"],json_pointer=p["json_pointer"]) for p in gates])


def run(ctx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":8,"lines.linewidth":.65,"pdf.fonttype":42})
    ctx.path.mkdir(parents=True,exist_ok=True);data=collect(ctx);figures=[]
    data["campaign_cost"]=accounting_snapshot(ctx,data)
    (ctx.path/"campaign-cost.json").write_text(json.dumps(data["campaign_cost"],indent=2,allow_nan=False)+"\n")
    review=evidence_review(data)
    (ctx.path/"review.json").write_text(json.dumps(review,indent=2,allow_nan=False)+"\n")
    (ctx.path/"evidence-review.md").write_text("# Response evidence review\n\n"+data["scope"]+"\n\n"+
        "\n\n".join("**"+r["question"]+"** "+r["answer"] for r in review["answers"])+
        "\n\nExact table pointers: `review.json`; source hashes: `sources.json`.\n")
    _architecture(ctx.path,figures,data["scope"]);_learning(data,ctx.path,figures)
    _training_outcomes(data,ctx.path,figures);_claims(data,ctx.path,figures)
    _spectral(data,ctx.path,figures);_coverage(data,ctx.path,figures);_spatial(data,ctx.path,figures)
    _source_behavior(data,ctx.path,figures)
    _quadrature(data,ctx.path,figures)
    _cost_details(data,ctx.path,figures)
    _decomposition(data,ctx.path,figures)
    specs=[
        ("endpoint-errors",["endpoint_metrics.json"],["error_rms"],["error_max"],"RMS · lower is better","Maximum error · lower is better","Endpoint errors, including failures in raw tables"),
        ("mean-spatial",["endpoint_metrics.json"],["absolute_mean_error","mean_error_rms"],["centered_rms"],"Absolute mean error · lower is better","Centered RMS · lower is better","Mean versus spatial error"),
        ("rollout",["rollout_metrics.json"],["time","horizon","step"],["error_rms"],"Physical time / recorded step","RMS · lower is better","Autonomous rollout observations"),
        ("sensitivity",["sensitivity_metrics.json"],["perturbation","epsilon"],["tangent_error","error_rms"],"Perturbation magnitude","Error · lower is better","Sensitivity: retain linear-response validity limits"),
        ("compression",["compression_results.json"],["rank","J"],["propagated_error_rms"],"Retained rank · more is more work","Propagated error · lower is better","Source compression at equal rank"),
        ("compression-source",["compression_results.json"],["rank"],["source_rms"],"Retained rank","Source RMS · lower is better","Source norm alone does not determine endpoint error"),
        ("compression-maximum",["compression_results.json"],["rank"],["endpoint_maximum"],"Retained rank","Maximum error · lower is better","Compression: maximum error matters separately"),
        ("memory-noise",["closure_results.json"],["noise","sigma"],["error_rms","endpoint_rms"],"Observation noise standard deviation","Prediction RMS · lower is better","Protected closure/observer conditions"),
        ("closure-composition",["closure_attribution.json"],["prediction_seconds"],["endpoint_rms"],"Measured CPU prediction seconds · lower is cheaper","Endpoint RMS · lower is better","Matched temporal/closure components and combinations; signed interactions retained in raw cases"),
        ("accuracy-cost",["timing.json","costs.json"],["median_seconds","seconds"],["error_rms"],"Complete call seconds · lower is better","RMS · lower is better","Measured complete-call cost and error; missing cost stays unavailable"),
        ("reference-quality",["reference_history.json","dataset.json"],["horizon","h"],["uncertainty_rms","reference_uncertainty_rms"],"Physical horizon","Estimated uncertainty · lower is better","Reference acceptance must precede accuracy claims"),
        ("data-efficiency",["training_results.json"],["data_count","train_count"],["rms"],"Independent training parents","Validation RMS · lower is better","Nested training-data efficiency; same data opportunities remain paired"),
        ("optimization-gradients",["learning_curves.jsonl"],["update"],["gradient_norm","grad_norm","gradient_norm_before_clip"],"Optimizer update","Gradient norm · neither larger nor smaller is universally better","Optimization scales and clipping; not a model-quality score"),
        ("mean-variance",["endpoint_metrics.json"],["reference_variance"],["prediction_variance"],"Reference variance","Predicted variance · agreement with x is better","Variance response across physical regimes"),
        ("residual-alignment",["endpoint_metrics.json"],["target_defect_rms"],["residual_alignment"],"Anchor defect RMS","Cosine alignment · closer to +1 is better","Correction direction versus target defect"),
        ("reaction-generalization",["endpoint_metrics.json"],["reaction_rate"],["rms","error_rms"],"Reaction coefficient","Physical RMS · lower is better","Reaction-rate shifts without merging target equations"),
        ("diffusion-generalization",["endpoint_metrics.json"],["kappa"],["rms","error_rms"],"Diffusion coefficient","Physical RMS · lower is better","Diffusion-rate shifts without merging target equations"),
        ("neural-numerical",["numerical_controls.json"],["median_seconds","seconds","elapsed_seconds"],["rms","error_rms"],"Complete measured seconds · lower is better","Physical RMS · lower is better","Numerical controls: quadrature and physical backbone costs"),
        ("mean-path",["mean_paths.json"],["time","horizon"],["absolute_mean_error","mean_error_rms"],"Physical time","Absolute mean error · lower is better","Learned mean path; absent measurements remain gaps"),
        ("memory",["timing.json"],["batch_size"],["peak_allocated_bytes"],"Batch size · throughput opportunity","Peak allocated bytes · lower is better at matched work","Measured memory; host RAM is not dedicated VRAM"),
        ("checkpoint-size",["model_resources.json"],["total_parameters"],["checkpoint_bytes"],"Model parameters · not a quality score","Checkpoint bytes · smaller is cheaper to store","Static model and replay-artifact resources"),
        ("query-reuse",["performance.json"],["query_count"],["median_seconds"],"Queries from the same encoded state","Complete call seconds · lower at matched query accuracy is better","Amortized-query measurements; all queried times require accepted teachers"),
        ("parameter-sensitivity",["parameter_metrics.json"],["epsilon"],["tangent_error"],"Parameter perturbation magnitude","Derivative error · lower is better","Independent coefficient responses with refined finite differences"),
        ("symmetry",["symmetry_metrics.json"],["equivariance_rms"],["rms","error_rms"],"Symmetry defect · lower is better","Physical error · lower is better","Compatible rotations/reflections: generic learned symmetry is measured, not assumed"),
    ]
    for args in specs:ctx.budget.check();_scatter(data,ctx.path,figures,*args)
    measured=[r for r in data["stages"] if finite(r["seconds"])]
    for page,start in enumerate(range(0,max(1,len(measured)),40)):
        chunk=measured[start:start+40]
        fig,ax=plt.subplots(figsize=(12,max(4,len(chunk)*.16)))
        if chunk:
            ax.barh([r["stage"] for r in chunk],[r["seconds"] for r in chunk],color="#45758A",alpha=.75)
            ax.tick_params(axis="y",labelsize=6)
        else:_missing(ax)
        ax.set(xlabel="Measured seconds · timing layer specified per row; allocation timers are separate",ylabel="Declared stage")
        _save(fig,ctx.path,f"stage-costs-{page:02d}","Successful science-body and failed CLI costs; these timing layers must not be summed together",figures,chunk,data["scope"])
    allocations=data["campaign_cost"]["rows"]
    for page,start in enumerate(range(0,max(1,len(allocations)),40)):
        chunk=allocations[start:start+40];fig,ax=plt.subplots(figsize=(12,max(4,len(chunk)*.16)))
        measured=[r for r in chunk if finite(r.get("elapsed_seconds"))]
        if measured:
            ax.barh([str(r["allocation_job_id"])+" "+r.get("State","UNKNOWN") for r in measured],
                [r["elapsed_seconds"] for r in measured],height=.7,alpha=.7,color="#64768B")
        else:_missing(ax,"Scheduler accounting not measured in this environment")
        ax.set(xlabel="Observed allocation seconds · lower complete cost is better at matched accuracy",ylabel="Actual job ID / state")
        _save(fig,ctx.path,f"allocations-{page:02d}","Campaign allocation accounting, including failures and recovery origins; "+data["campaign_cost"]["status"],figures,chunk,data["scope"])
    verify_sources(data);data["figures"]=figures
    with gzip.open(ctx.path/"chart-data.json.gz","wt",encoding="utf-8",compresslevel=5) as stream:json.dump(data,stream,allow_nan=False,separators=(",",":"))
    for name,value in (("figures.json",figures),("sources.json",data["sources"]),("coverage.json",data["stages"])):
        (ctx.path/name).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    esc=html.escape
    stage_rows="".join(f"<tr><td>{esc(r['stage'])}</td><td>{esc(r['status'])}</td><td>{r['seconds'] if r['seconds'] is not None else 'NA'}</td><td>{esc(str(r.get('error') or ''))}</td></tr>" for r in data["stages"])
    panels="".join(f"<section><h2>{esc(f['name'])}</h2><p>{esc(f['caption'])}</p><a href='{f['png']}'><img loading='lazy' src='{f['png']}'></a><p><a href='{f['pdf']}'>Vector PDF</a></p></section>" for f in figures)
    panels="".join(f"<p><strong>{esc(r['question'])}</strong> {esc(r['answer'])}</p>" for r in review["answers"])+panels
    (ctx.path/"index.html").write_text("<!doctype html><meta charset='utf-8'><title>TDN Response atlas</title><style>body{font:15px system-ui;max-width:1500px;margin:2em auto;padding:1em}img{width:100%}td,th{padding:.3em;text-align:left;border-bottom:1px solid #ddd}section{margin:3em 0}</style><h1>TDN Response analytical atlas</h1><p>"+esc(data["scope"])+"</p><p>"+esc(GUIDANCE)+"</p><p>Computational completion, mathematical checks and scientific benefit are separate. This atlas does not select methods.</p><p><a href='chart-data.json.gz'>Exact plotted and source table data</a> · <a href='sources.json'>Source hashes</a></p><table><tr><th>Stage</th><th>Status</th><th>Measured seconds</th><th>Unavailable reason</th></tr>"+stage_rows+"</table>"+panels)
    ctx.record("report/source-bound-atlas",["M0"],metrics=dict(figures=len(figures),sources=len(data["sources"]),missing_stages=len(data["omissions"])),
        checks=[check("source-hashes-unchanged",True,True,"eq",category="math"),check("inventory-is-not-scientific-benefit",None,True,"eq",category="gap",reason="Consult frozen claim rows; a graph cannot establish benefit")],
        evidence=["sources.json","chart-data.json.gz","figures.json","coverage.json","campaign-cost.json","index.html"])
    return dict(figures=len(figures),report="index.html",sources=len(data["sources"]),missing_stages=len(data["omissions"]))
