"""Compact, source-bound analytical atlas for the independent advance program.

Reporting never selects a model, repairs a failed measurement, or opens a
checkpoint.  The engine supplies verified prerequisites; each consumed file is
hashed again.  Raw artifacts remain in their stages.  One compressed chart
projection contains the values displayed here, instead of replicated JSON in
thousands of CSV cells.  Missing evidence and partial cohorts remain explicit.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import hashlib
import html
import json
import math
import os
from pathlib import Path
import statistics

from tdn.analysis.frontier.report import _learning_ranges as _base_learning_ranges
from tdn.analysis.portfolio.report import (
    _layout_figure, amortization as _paired_economics, prototype_assessments,
)
from tdn.analysis.portfolio.statistics import paired_cluster_interval_fast

SCHEMA = "tdn.advance-atlas/v1"
MAX_SOURCE_BYTES = 512 << 20
TABLE_FILES = {
    "rows.jsonl": "experiments", "catalog.json": "catalog",
    "learning_curves.jsonl": "learning_curves", "endpoint_rows.json": "endpoints",
    "locked_frontiers.json": "locked_frontiers", "scaling_rows.json": "scaling",
    "diagnostic_rows.json": "diagnostics", "residual_rows.json": "residuals",
    "audit_rows.json": "audit", "invariance_rows.json": "invariance",
    "profile_rows.json": "profiling", "prototype_rows.json": "prototypes",
    "XH_history.json": "history", "XR_resolvent.json": "resolvent",
    "XT_tangent.json": "tangent", "fairness_diagnostics.json": "fairness",
    "response_rows.json": "responses", "architecture.json": "architectures",
    "residual_projections.json": "residuals", "composition_rows.json": "composition",
    "order_rows.json": "orders",
    "amplitude_rows.json": "amplitudes",
}
TABLES = tuple(dict.fromkeys(TABLE_FILES.values())) + (
    "partial_endpoints", "intermediates", "claims_accuracy", "claims_cost",
    "stage_status", "checks", "amortization", "learning_ranges", "fixed_bands",
)
OMIT_REPEATED = {"curves", "timing", "raw_timing", "rounds", "samples_seconds",
                 "raw_seconds", "initial_state", "target_state", "prediction_state"}
GUIDANCE = (
    "Ours: triangles and solid lines; Theirs: circles and dashed lines; "
    "analytic controls: squares and dash-dot lines. Thin lines identify methods. "
    "Observed low–high learning bands are not confidence intervals."
)
SCOPE = (
    "Computational completion, mathematical checks and scientific benefit are separate. "
    "Missing observations remain NA. Independent fields are the statistical units; "
    "repeated grids, schedules and seeds are paired. No universal leaderboard."
)


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _read(path):
    """Read a bounded regular source without following symlinks or changed bytes."""
    path = Path(path)
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError("Atlas sources cannot contain symlinks")
    if not path.is_file() or path.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError("Atlas source must be a bounded regular file")
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        raw = stream.read(MAX_SOURCE_BYTES + 1)
        after = os.fstat(stream.fileno())
    signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if (signature(before) != signature(after) or signature(after) != signature(path.lstat())
            or len(raw) != before.st_size):
        raise ValueError("Atlas source changed during read")
    def invalid(value):
        raise ValueError("Nonfinite JSON value: " + value)
    value = ([json.loads(line, parse_constant=invalid) for line in raw.splitlines() if line.strip()]
             if path.suffix == ".jsonl" else json.loads(raw, parse_constant=invalid))
    return value, hashlib.sha256(raw).hexdigest(), len(raw)


def _records(value):
    if isinstance(value, list):
        return [(f"/{i}", row) for i, row in enumerate(value) if isinstance(row, dict)]
    if isinstance(value, dict):
        for key in ("rows", "records", "experiments", "groups", "diagnostics"):
            if isinstance(value.get(key), list):
                return [(f"/{key}/{i}", row) for i, row in enumerate(value[key]) if isinstance(row, dict)]
        return [("", value)]
    return []


def _compact(row):
    """Projection, not an altered raw artifact; omitted arrays/timers stay at source."""
    if isinstance(row, dict):
        return {key: _compact(value) for key, value in row.items() if key not in OMIT_REPEATED}
    if isinstance(row, list):
        return [_compact(value) for value in row]
    return row


def view(row):
    result = {}
    for key in ("metrics", "effective_config", "costs"):
        if isinstance(row.get(key), dict):
            result.update(row[key])
    result.update(row)
    result.setdefault("family", row.get("method", row.get("model_id", "unclassified")))
    result.setdefault("final_time", result.get("horizon", result.get("h")))
    result.setdefault("cost_seconds", result.get("inference_seconds"))
    return result


def role(family, row=None):
    declared = (row or {}).get("role", (row or {}).get("ownership"))
    if declared:
        lower = str(declared).lower()
        if lower.startswith("ours"):
            return "Ours"
        if lower.startswith("theirs"):
            return "Theirs"
        if "analytic" in lower or "oracle" in lower:
            return "Analytic control"
        if "fitted control" in lower:
            return "Fitted control"
    try:
        from .models import MODEL_SPECS
        declared = MODEL_SPECS.get(family, {}).get("role")
    except ImportError:
        declared = None
    if declared:
        return role(family, {"role": declared})
    name = str(family).lower()
    if name in {"df", "rf", "etdrk4"} or name.startswith(("fno", "direct_fno", "classical")):
        return "Theirs"
    if name.endswith(("_fixed", "_full")) and "conditioned" not in name or name.startswith(("analytic", "gauss")):
        return "Analytic control"
    return "Unclassified"


def style(family, row=None):
    category = role(family, row)
    palettes = {"Ours": ("#0072B2", "#4393C3", "#2166AC", "#17A2A4"),
                "Theirs": ("#D55E00", "#E69F00", "#AA4499", "#CC6677"),
                "Analytic control": ("#009E73", "#447755", "#667744", "#777777"),
                "Fitted control": ("#8866AA", "#664488"),
                "Unclassified": ("#555555",)}
    index = int(hashlib.sha256(str(family).encode()).hexdigest()[:8], 16)
    # Every declared family has a stable, distinct color. A short hash palette
    # collided for fno_zero/fno_mean and made their curves indistinguishable.
    declared_colors = {
        "df": "#777777", "etdrk4": "#222222",
        "quad2_full": "#009E73", "quad4_full": "#668D3C",
        "analytic_quad_cubic": "#8C6D31", "quad2_conditioned": "#17A2A4",
        "commutator_raw": "#6A51A3", "commutator_cubic": "#56B4E9",
        "two_basis": "#0072B2", "fno_legacy": "#CC6677",
        "fno_zero": "#D55E00", "fno_scaled": "#E69F00", "fno_mean": "#AA4499",
    }
    return dict(color=declared_colors.get(family, palettes[category][index % len(palettes[category])]),
                marker={"Ours": "^", "Theirs": "o", "Analytic control": "s", "Fitted control": "D", "Unclassified": "D"}[category],
                linestyle={"Ours": "-", "Theirs": "--", "Analytic control": "-.", "Fitted control": ":", "Unclassified": ":"}[category],
                linewidth=.55)


def label(row):
    row = view(row)
    return f"{role(row['family'], row)}: {row['family']}"


def collect(ctx):
    """Only engine-verified prerequisites supply science; failures remain status rows."""
    data = {key: [] for key in TABLES}
    data.update(schema=SCHEMA, profile=ctx.protocol.get("profile", "unknown"),
                protocol=ctx.protocol, sources=[], omissions=[], auxiliary=[],
                device=str(ctx.device), projection_note="Raw arrays, checkpoints and timing rounds remain in hashed stage sources; this file contains display values and pointers, not a replay bundle.")
    units = ctx.protocol.get("units", {})
    if isinstance(units, list):
        units = {u["id"]: u for u in units}
    kind = lambda key: units.get(key, {}).get("kind", key.split("-")[0])
    has_aggregate = any(kind(key) == "aggregate" for key in ctx.prerequisites)
    has_freeze = any(kind(key) == "freeze" for key in ctx.prerequisites)
    for unit in sorted(set(units) | set(ctx.prerequisites)):
        if unit == ctx.stage:
            continue
        base = ctx.prerequisites.get(unit)
        failed = getattr(ctx, "stage_failures", {}).get(unit, {})
        status = dict(unit=unit, kind=kind(unit), verified=bool(base),
                      status="VERIFIED" if base else failed.get("status", "NA"),
                      elapsed_seconds=failed.get("elapsed_seconds"), error=failed.get("error"),
                      monetary_cost=None)
        data["stage_status"].append(status)
        if base is None:
            continue
        base = Path(base)
        for filename in ("summary.json", *TABLE_FILES, "claims.json", "offline_costs.json"):
            path = base / filename
            if not path.exists():
                continue
            # Canonical aggregate/freeze replaces source projections, not evidence.
            if filename == "catalog.json" and has_freeze and kind(unit) != "freeze":
                continue
            if filename == "endpoint_rows.json" and has_aggregate and kind(unit) != "aggregate":
                continue
            ctx.budget.check()
            try:
                value, sha, size = _read(path)
            except (OSError, ValueError, TypeError) as exc:
                data["omissions"].append(dict(unit=unit, path=str(path), reason=str(exc)))
                continue
            source = len(data["sources"])
            data["sources"].append(dict(unit=unit, path=str(path), sha256=sha, bytes=size,
                                        evidence_status="VERIFIED_PREREQUISITE_READ_AGAIN"))
            if filename == "summary.json":
                status.update(elapsed_seconds=value.get("elapsed_seconds"),
                              scientific_outcome=value.get("scientific_outcome"),
                              computational_status=value.get("status"), _source=dict(file=source, pointer=""))
                continue
            if filename == "offline_costs.json":
                data.setdefault("offline_costs", {}).update(value.get("models", value))
                continue
            if filename == "claims.json":
                for category in ("accuracy", "cost"):
                    data["claims_" + category].extend({**_compact(row), "_source": dict(file=source, pointer=f"/{category}/{i}")}
                                                     for i, row in enumerate(value.get(category, [])))
                continue
            table = TABLE_FILES[filename]
            if table == "endpoints" and kind(unit) != "aggregate":
                table = "partial_endpoints"
            for pointer, original in _records(value):
                row = _compact(original)
                row["_source"] = dict(file=source, pointer=pointer)
                # Retain timing summaries, not duplicate paired raw rounds in every row.
                timing = original.get("timing", original.get("raw_timing", {}))
                if isinstance(timing, dict):
                    for key in ("cold_seconds", "p95_seconds", "min_seconds", "max_seconds", "repeats", "warmup", "peak_allocated_bytes", "peak_reserved_bytes", "memory_scope", "tail_scope"):
                        if key in timing:
                            row.setdefault(key, timing[key])
                    if isinstance(timing.get("memory"), dict):
                        memory = timing["memory"]
                        row["memory_scope"] = memory.get("scope")
                        row["memory_status"] = memory.get("status")
                        for key in ("absolute_peak_allocated_bytes", "absolute_peak_reserved_bytes", "incremental_peak_allocated_bytes", "incremental_peak_reserved_bytes", "probe_seconds"):
                            row[key] = memory.get(key)
                    row["tail_statistics"] = timing.get("tail_statistics")
                if table in ("endpoints", "partial_endpoints"):
                    for i, middle in enumerate(row.pop("intermediates", [])):
                        data["intermediates"].append({**{k: row.get(k) for k in ("family", "model_id", "track", "seed", "train_count", "field_cluster", "parent_id", "regime", "grid", "schedule_id")},
                            **middle, "_source": dict(file=source, pointer=pointer + f"/intermediates/{i}")})
                    for i, band in enumerate(row.get("spectral_bands", [])):
                        data["fixed_bands"].append({**{k: row.get(k) for k in ("family", "model_id", "role", "track", "seed", "train_count", "field_cluster", "parent_id", "regime", "grid", "final_time", "primary")},
                            **band, "band": band.get("label", str(i)), "_source": dict(file=source, pointer=pointer + f"/spectral_bands/{i}")})
                data[table].append(row)
            if isinstance(value, dict):
                metadata = {key: _compact(item) for key, item in value.items() if key not in {"rows", "records", "experiments", "groups", "diagnostics"}}
                if metadata and any(isinstance(value.get(k), list) for k in ("rows", "records", "experiments", "groups", "diagnostics")):
                    data["auxiliary"].append(dict(unit=unit, filename=filename, value=metadata,
                                                   _source=dict(file=source, pointer="")))
    catalog = {row.get("model_id"): row for row in data["catalog"]}
    prototype_roles = {r.get("method"): r.get("role") for r in data["prototypes"] if r.get("method")}
    for table in TABLES:
        for row in data[table]:
            if row.get("method") in prototype_roles:
                row.setdefault("role", prototype_roles[row["method"]])
            if row.get("model_id") in catalog:
                for key in ("role", "seed", "train_count", "family", "track"):
                    if row.get(key) is None:
                        row[key] = catalog[row["model_id"]].get(key)
    for row in data["experiments"]:
        for i, check in enumerate(row.get("checks", [])):
            data["checks"].append({**check, "experiment_id": row.get("experiment_id"),
                "_source": {**row["_source"], "pointer": row["_source"]["pointer"] + f"/checks/{i}"}})
    for row in data["catalog"]:
        for i, probe in enumerate(row.get("response_probes", [])):
            data["responses"].append({**{k: row.get(k) for k in ("family", "model_id", "role", "track", "seed", "train_count")},
                **probe, "_source": {**row["_source"], "pointer": row["_source"]["pointer"] + f"/response_probes/{i}"}})
    try:
        data["amortization"] = amortization(data)
    except ValueError as exc:
        # A report after a partial/corrupt aggregation must still explain why
        # economics cannot be constructed; it must not repair duplicate rows.
        data["omissions"].append(dict(table="amortization", reason=str(exc)))
    return data


def amortization(data):
    """Use exact workload/seed joins; unknown charges and nonpositive margins stay NA."""
    families = sorted({r.get("family") for r in data.get("catalog", []) if role(r.get("family"), r) == "Ours"})
    comparators = sorted({r.get("family") for r in data.get("catalog", []) if role(r.get("family"), r) in ("Theirs", "Analytic control", "Fitted control")}
                         | set(data.get("protocol", {}).get("confirmation", {}).get("primary_controls", [])))
    results = []
    for family in families:
        selected = {**data, "amortization_comparators": comparators,
                    "protocol": {**data.get("protocol", {}), "selection": {"primary_family": family}}}
        # This corrected helper joins actual locked cost_seconds records. It
        # never chooses the best competing seed or silently changes a schedule.
        results.extend(_paired_economics(selected))
    return results


def matched_budget_views(rows):
    """Show a frozen control alongside every fitted data budget, without refitting.

    This is a display view only. Raw rows and source pointers remain unique in
    chart data; plot observations may reuse one analytic measurement in more
    than one data-budget facet. No repeated control is an independent field.
    """
    counts=sorted({r.get("train_count") for r in rows if isinstance(r.get("train_count"),int) and r["train_count"]>0})
    result=[]
    for row in rows:
        displayed=counts if row.get("train_count")==0 and counts else [row.get("train_count")]
        result.extend({**row,"comparison_train_count":count} for count in displayed)
    return result


def field_summary(rows, metric, *, logarithmic=False, repeats=1000):
    """Descriptive crossed-field interval; repeated queries are not extra fields."""
    values = []
    for raw in rows:
        row = view(raw)
        value = row.get(metric)
        if finite(value) and (not logarithmic or value > 0) and row.get("field_cluster") is not None:
            values.append(dict(field_cluster=row["field_cluster"], seed=row.get("seed"),
                               difference=math.log(value) if logarithmic else value))
    interval = paired_cluster_interval_fast(values, repeats=repeats, minimum_fields=5)
    if logarithmic:
        for key in ("mean", "lower", "upper"):
            if interval.get(key) is not None:
                interval[key] = math.exp(interval[key])
    return interval


def _blank(ax, reason="NA — no verified observations for this panel"):
    ax.text(.5, .5, reason, ha="center", va="center", transform=ax.transAxes, wrap=True, color="#555555", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def _learning_ranges(curves, x, y, *, max_windows=24):
    """Observed windows with explicit update-support segments.

    The shared window grid alone can hide a missing update *inside* a window.
    Split those windows at support gaps before drawing. Training losses have
    one-update cadence; sparse validation cadence is inferred conservatively
    from observed increments. Wall-clock windows retain their measured gaps.
    """
    ranges = _base_learning_ranges(curves, x, y, max_windows=max_windows)
    if x != "update":
        return ranges
    grouped = defaultdict(list)
    for row in curves:
        if finite(row.get(x)) and finite(row.get(y)):
            grouped[(row.get("family", row.get("model_id")), row.get("track"), row.get("phase"), row.get("train_count"))].append(row)
    supports = {}
    for key, rows in grouped.items():
        points = sorted({row[x] for row in rows})
        gaps = [b-a for a,b in zip(points, points[1:]) if b>a]
        cadence = 1. if y == "train_loss" else min(gaps, default=1.)
        segments = {}; bounds = {}; segment = 0
        for i, point in enumerate(points):
            if i and point-points[i-1] > cadence*(1+1e-9):
                segment += 1
            segments[point] = segment
            bounds.setdefault(segment, [point, point])[1] = point
        supports[key] = (rows, segments, bounds, cadence)
    output = []
    for window in ranges:
        if not window["observation_count"]:
            output.append({**window, "support_segment": None}); continue
        key = (window["family"], window.get("track"), window.get("phase"), window.get("train_count"))
        rows, segments, bounds, cadence = supports[key]
        parts = defaultdict(list)
        for row in rows:
            if window["x_min"] <= row[x] <= window["x_max"]:
                parts[segments[row[x]]].append(row)
        for segment, members in sorted(parts.items()):
            lo, hi = bounds[segment]
            output.append({**window, "support_segment": segment, "observed_update_cadence": cadence,
                "window_left": max(window["window_left"],lo), "window_right": min(window["window_right"],hi),
                "family_x_min": lo, "family_x_max": hi,
                "x_min": min(row[x] for row in members), "x_max": max(row[x] for row in members),
                "y_min": min(row[y] for row in members), "y_max": max(row[y] for row in members),
                "observation_count": len(members), "distinct_x_count": len({row[x] for row in members})})
    return output


def range_plot(ax, records, x, y):
    """Continuous observed bounds; empty windows break the curve, never bridge it."""
    groups = defaultdict(list)
    for record in records:
        groups[record["family"]].append(record)
    measured = []
    for family, rows in sorted(groups.items()):
        rows.sort(key=lambda row: (row["window_index"], row.get("x_min") if row.get("x_min") is not None else -math.inf))
        appearance = style(family, rows[0])
        measured.extend(r for r in rows if r["observation_count"])
        def draw(run):
            if not run:
                return
            if len(run)==1 and run[0]["distinct_x_count"]==1:
                row=run[0]
                ax.vlines(row["x_min"],row["y_min"],row["y_max"],color=appearance["color"],linewidth=.55)
                ax.plot([row["x_min"]]*2,[row["y_min"],row["y_max"]],linestyle="None",marker=appearance["marker"],color=appearance["color"],markersize=2)
                return
            bounds = [(max(r["window_left"], r["family_x_min"]), min(r["window_right"], r["family_x_max"])) for r in run]
            anchors = [(bounds[0][0],run[0]["y_min"],run[0]["y_max"])]
            anchors.extend((r["x_min"] if r["distinct_x_count"]==1 else (lo+hi)/2,r["y_min"],r["y_max"]) for r,(lo,hi) in zip(run,bounds))
            anchors.append((bounds[-1][1],run[-1]["y_min"],run[-1]["y_max"]))
            merged=[]
            for point,lower,upper in anchors:
                if merged and point==merged[-1][0]:
                    merged[-1]=(point,min(lower,merged[-1][1]),max(upper,merged[-1][2]))
                else: merged.append((point,lower,upper))
            xs,lower,upper=zip(*merged)
            ax.fill_between(xs, lower, upper, color=appearance["color"], alpha=.10, linewidth=0)
            for edge in (lower, upper):
                ax.plot(xs, edge, color=appearance["color"], linestyle=appearance["linestyle"], linewidth=.55, alpha=.95)
        run = []
        for row in rows:
            if not row["observation_count"]:
                draw(run); run = []
            else:
                if run and (row["window_index"] != run[-1]["window_index"] + 1
                            or row.get("support_segment") != run[-1].get("support_segment")):
                    draw(run); run = []
                run.append(row)
        draw(run)
        if any(r["observation_count"] for r in rows):
            ax.plot([], [], label=label(rows[0]), **appearance, markersize=4)
    if not measured:
        _blank(ax)
    elif all(r["y_min"] > 0 for r in measured):
        ax.set_yscale("log")
    ax.set_xlabel(x + " (measured progression)")
    ax.set_ylabel(y + " (lower is better)")
    return sum(row["observation_count"] for row in measured)


def _metric_draw(ax, rows, metric, *, log=True, higher=False):
    groups = defaultdict(list)
    for raw in rows:
        row = view(raw)
        if finite(row.get(metric)):
            groups[row["family"]].append(row)
    # Exact zeros are useful evidence. Use a linear panel if a zero occurs,
    # rather than dropping it or inventing an epsilon below the measurement.
    log = log and all(r[metric] > 0 for members in groups.values() for r in members)
    plotted = False
    count = 0
    for index, (family, members) in enumerate(sorted(groups.items())):
        interval = field_summary(members, metric, logarithmic=log)
        val = interval.get("mean")
        if val is None:
            # No field identity means no field uncertainty. A descriptive raw
            # median is displayed separately and never assigned a fake CI.
            finite_values = [r[metric] for r in members if finite(r.get(metric)) and (not log or r[metric] > 0)]
            val = statistics.median(finite_values) if finite_values else None
        if val is None:
            continue
        appearance = style(family, members[0])
        ax.plot(index, val, label=label(members[0]), linestyle="None", marker=appearance["marker"], color=appearance["color"], markersize=4)
        if interval.get("lower") is not None and interval.get("upper") is not None:
            # Percentile intervals need not contain the original estimate.
            ax.vlines(index, interval["lower"], interval["upper"], color=appearance["color"], linewidth=.6)
        plotted = True
        count += len(members)
    if not plotted:
        _blank(ax)
    else:
        ax.set_xticks(range(len(groups)), [label(v[0]) for _, v in sorted(groups.items())], rotation=55, ha="right", fontsize=6)
        if log:
            ax.set_yscale("log")
    ax.set_ylabel(metric + (" (higher is better)" if higher else " (lower is better)"))
    ax.set_xlabel("Methods within this target/workload only")
    return count


def _scatter(ax, rows, x, y, *, logx=False, logy=False, xlabel=None, ylabel=None):
    groups = defaultdict(list)
    for raw in rows:
        row = view(raw)
        if finite(row.get(x)) and finite(row.get(y)) and (not logx or row[x] > 0) and (not logy or row[y] > 0):
            groups[row["family"]].append(row)
    for family, members in sorted(groups.items()):
        appearance = style(family, members[0])
        ax.scatter([r[x] for r in members], [r[y] for r in members], marker=appearance["marker"], color=appearance["color"], s=13, alpha=.5, linewidths=.25, label=label(members[0]))
    if not groups:
        _blank(ax)
    else:
        if logx: ax.set_xscale("log")
        if logy: ax.set_yscale("log")
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    return sum(map(len, groups.values()))


def _eligible(row):
    """Plots do not silently reinterpret an unaccepted reference as accuracy."""
    if row.get("reference_accepted") is not True or row.get("finite") is False:
        return False
    if not all(finite(row.get(k)) for k in ("upper_rms", "upper_max", "cost_seconds")):
        return False
    return row["cost_seconds"] > 0 and row["upper_rms"] >= 0 and row["upper_max"] >= 0


def basis_gain_rows(catalog):
    """Project measured fixed coefficients without implying state dependence."""
    rows = []
    for row in catalog:
        for i, gain in enumerate(row.get("parameter_report", {}).get("effective_gains", [])):
            rows.append({**{k: row.get(k) for k in ("family", "role", "track", "seed", "train_count", "model_id")},
                "basis_index": i, "basis": ("Full GL2 interaction", "Cubic interaction")[i] if i < 2 else f"Basis {i}",
                "gain": gain, "_source": {**row.get("_source", {}),
                    "pointer": row.get("_source", {}).get("pointer", "") + f"/parameter_report/effective_gains/{i}"}})
    return rows


def _basis_gain_draw(ax, rows):
    count = _scatter(ax, rows, "basis_index", "gain", xlabel="Physical interaction basis",
                     ylabel="Bounded fitted gain (neither higher nor lower is universally better)")
    if count:
        ax.set_xticks([0, 1], ["Full GL2 interaction", "Cubic interaction"])
        ax.axhline(1., color="#555555", linestyle=":", linewidth=.5, alpha=.65)
    return count


def _write_json(path, value, *, compressed=False):
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":") if compressed else None,
                     indent=None if compressed else 2, allow_nan=False).encode()
    if compressed:
        with Path(path).open("wb") as output, gzip.GzipFile(fileobj=output, mode="wb", compresslevel=6, mtime=0) as stream:
            stream.write(raw)
    else:
        Path(path).write_bytes(raw + b"\n")


def build_figures(data, output, *, budget=None, dpi=300):
    """Render measured evidence and explicit missing panels at publication resolution."""
    if type(dpi) is not int or dpi < 300:
        raise ValueError("Advance atlas PNGs require at least 300 DPI")
    output = Path(output)
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("Unsafe atlas output directory")
    output.mkdir(parents=True, exist_ok=True)
    project = Path(__file__).resolve().parents[3]
    for name, suffix in (("MPLCONFIGDIR", "matplotlib"), ("XDG_CACHE_HOME", "xdg")):
        current = Path(os.environ.get(name, "/")).resolve()
        if current != project and project not in current.parents:
            current = project / "cache" / "advance-atlas" / suffix
            os.environ[name] = str(current)
        current.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import numpy as np

    panels = []
    plotted_data = {**data, "learning_ranges": []}
    with PdfPages(output / "advance-atlas.pdf") as pdf:
        def save(title, fig, guide, scope, count, facets=None):
            if budget: budget.check()
            _layout_figure(fig, title, guide, scope)
            name = f"{len(panels)+1:03d}-" + "".join(c if c.isalnum() else "-" for c in title.lower()).strip("-") + ".png"
            path = output / name
            if path.is_symlink(): raise ValueError("Symlink atlas output refused")
            fig.savefig(path, dpi=dpi)
            pdf.savefig(fig)  # Text, paths, markers and curves stay vector objects.
            plt.close(fig)
            panels.append(dict(title=title, path=name, reading_guide=guide, scope=scope,
                               observations=count, status="OBSERVED" if count else "NA",
                               facets=facets, dpi=dpi, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

        def single(title, draw, guide, *, scope=SCOPE, count=0):
            fig, ax = plt.subplots(figsize=(12, 5.5))
            measured = draw(ax)
            if isinstance(measured, int): count = measured
            save(title, fig, guide, scope, count)

        def facets(title, rows, keys, draw, guide, *, scope=SCOPE):
            groups = defaultdict(list)
            for raw in rows:
                row = view(raw)
                groups[tuple(str(row.get(k, "NA")) for k in keys)].append(row)
            groups = sorted(groups.items()) or [(tuple("NA" for _ in keys), [])]
            for offset in range(0, len(groups), 4):
                part = groups[offset:offset+4]
                cols = min(2, len(part)); nr = math.ceil(len(part)/cols)
                fig, axes = plt.subplots(nr, cols, figsize=(12, 4.3*nr), squeeze=False)
                count = 0
                for ax, (key, values) in zip(axes.flat, part):
                    measured = draw(ax, values)
                    count += measured if isinstance(measured, int) else len(values)
                    ax.set_title(" | ".join(f"{k}={v}" for k,v in zip(keys,key)), fontsize=7)
                for ax in list(axes.flat)[len(part):]: ax.axis("off")
                suffix = f" page {offset//4+1}" if len(groups)>4 else ""
                save(title+suffix, fig, guide, scope, count, keys)

        def architecture(ax):
            ax.axis("off")
            boxes = [(.02,.66,.23,.21,"Current field u\nh, physics, grid"),
                     (.35,.78,.28,.15,"Frozen physical backbone\nExact/controlled propagation"),
                     (.33,.45,.33,.21,"Full-output normalized GL2\n+ optional heat-filtered cubic defect"),
                     (.02,.14,.25,.22,"Two bounded basis gains\nOr zero-head/scaled FNO\nDeployable state + physics"),
                     (.75,.48,.23,.23,"Backbone + correction\nNext field / rollout")]
            for x,y,w,h,text in boxes:
                ax.add_patch(plt.Rectangle((x,y),w,h,facecolor="#edf4f8",edgecolor="#436784",linewidth=.65))
                ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=9)
            for a,b in [((.25,.8),(.35,.85)),((.25,.72),(.33,.58)),((.27,.25),(.42,.45)),((.63,.85),(.84,.71)),((.66,.55),(.75,.58))]:
                ax.annotate("",xy=b,xytext=a,arrowprops=dict(arrowstyle="->",lw=.65))
            ax.text(.5,.04,"XH: causal-history closure   |   XR: rational/resolvent representation   |   XT: tangent information",ha="center",fontsize=8)
        single("Architecture and information flow", architecture,
               "Arrows represent information and work, not a higher/lower performance scale.",
               scope="Conceptual interaction-correction graph. Exact implemented families/configurations and prototype formulations remain in catalog/architecture sources; a schematic does not prove temporal order.",count=len(data.get("catalog",[])))

        def architecture_inventory(ax):
            rows = data.get("catalog", [])
            seen = {}
            for row in rows:
                seen.setdefault(row.get("family", "unknown"), row)
            if not seen:
                _blank(ax); return 0
            ax.axis("off")
            descriptions = []
            try:
                from .models import MODEL_SPECS
            except ImportError:
                MODEL_SPECS = {}
            for family, row in sorted(seen.items()):
                description = MODEL_SPECS.get(family, {}).get("description", "See sealed family/configuration metadata")
                descriptions.append(f"{label(row)}: {description}")
            for i, text in enumerate(descriptions):
                ax.text(.01, 1-(i+.5)/max(len(descriptions),1), text, fontsize=8, va="center", wrap=True)
            return len(seen)
        single("Implemented family information inventory", architecture_inventory,
               "Architecture and available information are explanatory; no universal higher/lower direction.",
               scope="Frozen catalog is authoritative. Prior-art control adaptations are labeled Theirs; numerical formulas and learned components remain separate.")

        def stages(ax):
            rows=data.get("stage_status",[])
            if not rows: return _blank(ax)
            for i,r in enumerate(rows):
                value=r.get("elapsed_seconds")
                if finite(value): ax.barh(i,value,color="#009E73" if r["verified"] else "#999999",height=.55)
                else: ax.text(0,i,"NA",va="center",fontsize=7)
            ax.set_yticks(range(len(rows)),[r["unit"]+" — "+r["status"] for r in rows],fontsize=6)
            ax.set_xlabel("Recorded stage seconds (lower cost is better; completion is not scientific success)")
        single("Execution and missing evidence", stages,"Lower measured cost is better. Gray/missing entries retain failed or unavailable stages.",count=len(data.get("stage_status",[])))

        curves=[view(r) for r in data.get("learning_curves",[])]
        roles={r["family"]:r.get("role") for r in curves}
        for x,y,title in [("update","train_loss","Training loss by update"),
                          ("elapsed_seconds","train_loss","Training loss by wall compute"),
                          ("optimizer_seconds","train_loss","Training loss by optimizer compute"),
                          ("update","validation_loss","Validation objective by update"),
                          ("elapsed_seconds","validation_loss","Validation objective by wall compute")]:
            ranges=_learning_ranges(curves,x,y,max_windows=24)
            ranges=[{**r,"role":roles.get(r["family"])} for r in ranges]
            plotted_data["learning_ranges"].extend({"panel":title,**r} for r in ranges)
            facets(title,ranges,("track","phase","train_count"),lambda ax,rs,x=x,y=y:range_plot(ax,rs,x,y),
                   "Lower is better. Faint continuous low/high envelopes summarize observed values, not confidence intervals.",
                   scope="Linear connections interpolate observed windows only; gaps remain. Phases stay separate: tiny_overfit evaluates the SAME training example, not held-out validation or generalization. Equal-time diagnostics do not make the final validation-selected catalog an equal-training-compute comparison. All trial costs remain charged.")

        endpoints=data.get("endpoints",[])
        partial=data.get("partial_endpoints",[])
        if not endpoints and partial:
            endpoints=partial
        accepted=[view(r) for r in endpoints if r.get("reference_accepted") is True and r.get("finite") is not False]
        # Preserve exact query definitions instead of pooling grids, targets or
        # horizons into a universal family ranking.
        primary=[r for r in accepted if r.get("primary",True)]
        axes=("track","grid","final_time","comparison_train_count")
        matched_primary=matched_budget_views(primary)
        for metric,title in [("error_rms","Endpoint RMS error"),("error_max","Endpoint maximum error"),
                             ("mean_error","Evolving mean error"),("centered_rms","Centered spatial error"),
                             ("relative_rms","Relative RMS with declared normalization")]:
            facets(title,matched_primary,axes,lambda ax,rs,m=metric:_metric_draw(ax,rs,m),
                   "Lower is better. Points are field-balanced geometric means; whiskers are descriptive field/seed intervals when at least five independent fields are available.",
                   scope="Primary schedules only when labeled; repeated queries averaged within field/seed. Frozen controls are reused alongside each training budget without becoming new observations. Intervals are not multiplicity adjusted. Partial cohorts are not confirmation; exact zeros use a linear scale.")
        facets("Fixed physical-frequency error bands",matched_budget_views(data.get("fixed_bands",[])),(*axes,"band"),lambda ax,rs:_metric_draw(ax,rs,"error_rms"),
               "Lower error is better. Band boundaries use physical cycles per unit length, not a grid-dependent Nyquist fraction.")
        for metric,title in (("energy_absolute_error","Spatially consistent physical energy error"),("energy_increase","Observed energy increase"),("mean_balance_residual","Instantaneous mean-rate identity")):
            facets(title,matched_primary,axes,lambda ax,rs,m=metric:_metric_draw(ax,rs,m),
                   "Lower is better. Energy uses the declared FD or Nyquist-aware Galerkin spatial inner product; missing derivative measurements are NA, not zero.",
                   scope="Energy dissipation is an exact-flow identity in its specified spatial equation. Endpoint violations diagnose this step; observed compliance does not prove a universal stability theorem.")
        facets("Short-rollout intermediate errors",data.get("intermediates",[]),("track","grid","train_count"),
               lambda ax,rs:_scatter(ax,rs,"time","error_rms",logy=True,xlabel="Physical rollout time (later is a harder horizon, not a quality score)",ylabel="RMS error (lower is better)"),
               "Lower is better at the same physical time; missing teachers remain gaps. This is not a long-time stability proof.")
        facets("Data efficiency",primary,("track","grid","final_time"),
               lambda ax,rs:_scatter(ax,rs,"train_count","error_rms",logy=True,xlabel="Independent training fields (less data at matched error is better)",ylabel="RMS error (lower is better)"),
               "Lower error at equal independent-field data is better; more data may also change physics coverage. Paired seeds are retained.")

        def coverage(ax,rows):
            target=float(data.get("protocol",{}).get("primary_target",2e-5))
            cells=defaultdict(list)
            for r in rows:
                if r.get("field_cluster") is None: continue
                pass_= _eligible(r) and r["upper_rms"]<=float(r.get("rms_target",target)) and r["upper_max"]<=float(r.get("max_target",target))
                cells[(r["family"],r.get("regime","unspecified"),r["field_cluster"])].append(pass_)
            families=sorted({k[0] for k in cells});regimes=sorted({k[1] for k in cells})
            if not cells:
                _blank(ax); return 0
            arr=np.full((len(families),len(regimes)),np.nan)
            for i,f in enumerate(families):
                for j,g in enumerate(regimes):
                    values=[all(v) for (ff,gg,_),v in cells.items() if ff==f and gg==g]
                    if values:
                        arr[i,j]=sum(values)/len(values)
                        ax.text(j,i,f"{sum(values)}/{len(values)}",ha="center",va="center",fontsize=6,color="black")
            ax.imshow(np.ma.masked_invalid(arr),vmin=0,vmax=1,cmap="YlGn",aspect="auto")
            ax.set_yticks(range(len(families)),[label(next(r for r in rows if r['family']==f)) for f in families],fontsize=6)
            ax.set_xticks(range(len(regimes)),regimes,rotation=45,ha="right",fontsize=6)
            return len(cells)
        facets("Field coverage and failure map",matched_budget_views([view(r) for r in endpoints]),axes,coverage,
               "Higher is better; each field passes only when every included query passes both uncertainty-adjusted norms. Fractions show independent fields; blank is NA.")
        def violations(ax,rows):
            vals=[{**r,"violation_fraction":float(bool(r.get("physical_interval_violations")))} for r in rows if r.get("physical_interval_violations") is not None]
            return _metric_draw(ax,vals,"violation_fraction",log=False)
        facets("Physical interval failures",[view(r) for r in endpoints],("track","grid","train_count"),violations,
               "Lower violation frequency is better. No observed violation is not a positivity theorem; inspect initial-field bounds too.")

        qualified=[r for r in accepted if _eligible(r)]
        facets("Measured accuracy versus warm cost",matched_budget_views(qualified),axes,
               lambda ax,rs:_scatter(ax,rs,"upper_rms","cost_seconds",logx=True,logy=True,xlabel="RMS plus reference uncertainty (lower is better)",ylabel="Complete-call warm seconds (lower is better)"),
               "Lower-left is better at the same physical task. References must be accepted; maximum-norm qualification is still required for a solver claim.",
               scope="Measured tradeoff cloud, not reference-informed deployment selection. Raw timing rounds remain linked in source stages. No fabricated Pareto interpolation.")
        locked=[view(r) for r in data.get("locked_frontiers",[])]
        facets("Validation-locked qualified latency",matched_budget_views([r for r in locked if r.get("status")=="ELIGIBLE"]),(*axes,"rms_target","max_target"),
               lambda ax,rs:_metric_draw(ax,rs,"cost_seconds"),
               "Lower is better conditional on qualification. Coverage/failures must be read beside latency; no best seed or confirmation-selected schedule is chosen here.")
        facets("First invocation versus warmed latency",matched_budget_views(accepted),axes,
               lambda ax,rs:_scatter(ax,rs,"cost_seconds","cold_seconds",logx=True,logy=True,xlabel="Warm complete-call seconds (lower is better)",ylabel="First-in-group seconds (lower is better)"),
               "Lower-left is better. First invocation in a group is not guaranteed cold-process startup.")
        facets("Measured absolute memory with attribution scope",[view(r) for r in endpoints],("track","grid","memory_scope"),
               lambda ax,rs:_metric_draw(ax,rs,"absolute_peak_allocated_bytes"),
               "Lower is better at matched accuracy; memory scopes stay separate. Whole-group GPU memory is not assigned to individual models.")
        facets("Incremental memory above resident baseline",[view(r) for r in endpoints],("track","grid","memory_scope"),
               lambda ax,rs:_metric_draw(ax,rs,"incremental_peak_allocated_bytes"),
               "Lower is better. Incremental allocator peak excludes resident baseline and is not the total deployment footprint. CPU unavailable peaks remain NA.")
        scaling = [view(r) for r in data.get("scaling", [])]
        facets("Accuracy-qualified batch throughput",[r for r in scaling if r.get("accuracy_qualified") is True],("track","grid","batch_size","final_time"),
               lambda ax,rs:_metric_draw(ax,rs,"fields_per_second",higher=True),
               "Higher throughput is better only at the same accepted accuracy and batch size. Missing or inaccurate rows remain in the source chart data.")
        facets("Scaling complete-call latency",scaling,("track","grid","batch_size","final_time"),
               lambda ax,rs:_metric_draw(ax,rs,"cost_seconds"),
               "Lower is better, but accuracy-unqualified rows cannot establish an acceleration claim. Per-field references remain separately charged.")
        facets("Profiling overhead breakdown",[view(r) for r in data.get("profiling",[])],("track","grid","component"),
               lambda ax,rs:_metric_draw(ax,rs,"seconds",log=False),
               "Lower component time is better. Profiling is separate from deployment timing; missing transform/transport/product measurements remain NA.")
        facets("Known-offline amortization scenarios",[view(r) for r in data.get("amortization",[])],("track","final_time","comparator_family","train_count"),
               lambda ax,rs:_metric_draw(ax,rs,"known_offline_break_even_queries",log=True),
               "Lower break-even query count is better only with positive matched-accuracy savings. Missing offline components and monetary rates remain unknown.",
               scope="Exact parent/grid/track/horizon/target joins with paired seeds. Training-only break-even is an optimistic partial-cost scenario, not complete campaign economics; infeasible or nonpositive margins remain NA in chart data.")

        diagnostics=[view(r) for table in ("diagnostics","residuals","audit","invariance") for r in data.get(table,[])]
        for row in diagnostics:
            if "translation_max" in row:
                row.setdefault("translation_equivariance_error", row["translation_max"])
        residual_directions=[]
        for raw in data.get("residuals", []):
            row=view(raw)
            for i, cosine in enumerate(row.get("residual_basis_cosines", [])):
                residual_directions.append({**row,"family":f"basis-{i}","role":"Analytic control","residual_cosine":cosine})
        diagnostics.extend(residual_directions)
        order_rows=[]
        groups=defaultdict(list)
        for raw in data.get("orders", []):
            row=view(raw)
            if finite(row.get("horizon")) and finite(row.get("raw_remainder_rms")) and row["horizon"]>0 and row["raw_remainder_rms"]>0:
                groups[(row.get("track"),row.get("grid"),row.get("regime"))].append(row)
        for key, rows in groups.items():
            ordered=sorted(rows,key=lambda r:r["horizon"])
            for left,right in zip(ordered,ordered[1:]):
                if left["horizon"]!=right["horizon"]:
                    order_rows.append(dict(track=key[0],grid=key[1],regime=key[2],family="cubic-remainder",role="Ours numerical",
                        observed_order=math.log(right["raw_remainder_rms"]/left["raw_remainder_rms"])/math.log(right["horizon"]/left["horizon"]),
                        pair_sources=[left.get("_source"),right.get("_source")]))
        diagnostics.extend(order_rows)
        plotted_data["derived_order_rows"]=order_rows
        amplitude_rows=[]
        for raw in data.get("amplitudes", []):
            row=view(raw)
            for metric,family in (("base_error_rms","df"),("quad_error_rms","quad2_full"),("cubic_error_rms","commutator_cubic")):
                if finite(row.get(metric)):
                    amplitude_rows.append({**row,"family":family,"error_rms":row[metric]})
        facets("Amplitude expansion beyond quadrature",amplitude_rows,("track","grid","regime","horizon"),
               lambda ax,rs:_scatter(ax,rs,"amplitude","error_rms",logx=True,logy=True,
                   xlabel="Perturbation amplitude (not a quality ranking)",ylabel="Endpoint RMS error (lower is better)"),
               "Lower error is better at identical amplitude and horizon. Observed amplitude scaling does not prove uniform temporal order.")
        for metric,title,higher in [("residual_cosine","Residual direction alignment",True),
                                    ("oracle_error_rms","Residual representation ceiling",False),
                                    ("translation_equivariance_error","Translation equivariance audit",False),
                                    ("sensitivity_relative_error","Physical parameter sensitivity",False),
                                    ("observed_order","Observed temporal order",True)]:
            facets(title,diagnostics,("track","grid","regime"),lambda ax,rs,m=metric,hi=higher:_metric_draw(ax,rs,m,log=m not in ("residual_cosine","observed_order"),higher=hi),
                   ("Higher alignment is better; sign reversal can harm corrections." if metric=="residual_cosine" else
                    "Higher observed slope suggests faster asymptotic decay; it is not a proof of method order." if metric=="observed_order" else
                    "Lower is better. Oracle capacity is not a deployable learned result; missing diagnostics remain NA."))
        facets("Basis identifiability",[view(r) for r in data.get("residuals", [])],("track","grid","regime"),
               lambda ax,rs:_metric_draw(ax,rs,"condition_number"),
               "Lower conditioning is better. Collinear physical bases can produce nonidentifiable gains even when prediction is accurate; absent/infinite conditioning stays NA.")
        facets("Step composition and directional sensitivity",[view(r) for r in data.get("composition", [])],("track","grid","regime"),
               lambda ax,rs:_scatter(ax,rs,"physical_directional_amplification","composition_defect_rms",logy=True,
                   xlabel="Sampled directional amplification (compare with physical growth; not universally lower is better)",ylabel="One-step versus two-step RMS defect (lower is better)"),
               "Lower composition defect is better; a sampled directional derivative is not an operator norm or stability proof.")
        responses=[]
        for r in data.get("responses",[]):
            response=r.get("response")
            if isinstance(response,dict):responses.append({**r,**response})
            elif finite(response):responses.append({**r,"effective_gain":response})
        facets("Learned parameter response",responses,("track","train_count"),
               lambda ax,rs:_scatter(ax,rs,"horizon","effective_gain",xlabel="Requested physical horizon",ylabel="Effective gain (neither universally higher nor lower is better)"),
               "No universal better direction: compare constraints, identifiability and useful state dependence, not parameter magnitude.")
        gains=basis_gain_rows(data.get("catalog", []))
        plotted_data["basis_gain_rows"]=gains
        facets("Fitted basis coefficients",matched_budget_views(gains),("track","comparison_train_count"),_basis_gain_draw,
               "Fixed coefficients are not state-dependent response functions. Dotted unity is the analytic coefficient; magnitude alone is not success. Compare constraints, residual alignment and identifiability.")
        for table,title in (("history","XH causal history and ablations"),("resolvent","XR rational representation")):
            vals=[view(r) for r in data.get(table,[])]
            facets(title,vals,("scenario","track"),lambda ax,rs:_metric_draw(ax,rs,"error_rms"),
                   "Lower prediction error is better. Causal information, initialization/setup cost and failed ablations must remain visible.",
                   scope="Distinct exploratory task; do not mix prototype response/closure error with full PDE solver error. Negative outcomes are retained.")
        history=[view(r) for r in data.get("history", [])]
        facets("XH forward rollout and history-noise sensitivity",history,("scenario","history_noise"),
               lambda ax,rs:_scatter(ax,rs,"time","error_rms",logy=True,xlabel="Physical time after causal history acquisition",ylabel="Closure RMS error (lower is better)"),
               "Lower is better for the same history access/noise. History startup and teacher costs are distinct from inference.")
        facets("XH complete history-acquisition cost",history,("scenario","history_noise"),
               lambda ax,rs:_scatter(ax,rs,"error_rms","complete_seconds",logx=True,logy=True,xlabel="Closure RMS error (lower is better)",ylabel="Complete history plus inference seconds (lower is better)"),
               "Lower-left is better at a shared workload. The cost of obtaining forward history is included, not treated as free.")
        temporal_queries=[];responses=[]
        for source in data.get("auxiliary", []):
            if source.get("filename")!="XR_resolvent.json":continue
            for i,row in enumerate(source["value"].get("timing_rows", [])):
                temporal_queries.append({**row,"family":row.get("method"),"role":"Theirs" if row.get("method","").startswith("classical") else "Analytic control",
                    "complete_seconds":row.get("complete_build_and_queries",{}).get("median_seconds"),
                    "_source":{**source["_source"],"pointer":f"/timing_rows/{i}"}})
            for i,row in enumerate(source["value"].get("response_rows", [])):
                responses.append({**row,"family":row.get("method"),"role":"Analytic control",
                    "response_error":abs(row["exponential"]-row["exact_exponential"]) if finite(row.get("exponential")) and finite(row.get("exact_exponential")) else None,
                    "_source":{**source["_source"],"pointer":f"/response_rows/{i}"}})
        plotted_data["temporal_query_projection"]=temporal_queries
        facets("XR qualified encoding and repeated-query cost",[r for r in temporal_queries if r.get("quality_qualified") is True],(),
               lambda ax,rs:_scatter(ax,rs,"queries","complete_seconds",logx=True,logy=True,xlabel="Query count (workload size)",ylabel="Encoding plus all requested queries seconds (lower is better)"),
               "Lower total cost is better at the same query count and accepted accuracy. Setup is charged; classical dense output has equivalent reuse opportunity.")
        facets("XR response-kernel approximation",responses,(),
               lambda ax,rs:_scatter(ax,rs,"z","response_error",logy=True,xlabel="Dimensionless generator z",ylabel="Exponential response absolute error (lower is better)"),
               "Lower response error is better, but is not full nonlinear PDE error. Exact zero errors remain in canonical chart data.")
        tangents=[view(r) for r in data.get("tangent",[])]
        for metric,title in (("tangent_fd_relative_error","XT tangent directional derivative"),("semigroup_relative_error","XT tangent composition"),("mean_identity_error","XT physical mean identity")):
            facets(title,tangents,("track",),lambda ax,rs,m=metric:_metric_draw(ax,rs,m),
                   "Lower is better. Finite-difference and composition diagnostics do not establish a general differentiability/stability theorem.")
        def prototype_results(ax):
            rows=prototype_assessments(data.get("prototypes",[]))
            if not rows:return _blank(ax)
            categories=("math","gap","utility")
            mapping={"BAD":0.,"NA":.5,"GOOD":1.}
            arr=np.array([[mapping[r["display_science"][c]] for c in categories] for r in rows])
            ax.imshow(arr,vmin=0,vmax=1,cmap="RdYlGn",aspect="auto")
            for i,r in enumerate(rows):
                for j,c in enumerate(categories):ax.text(j,i,r["display_science"][c],ha="center",va="center",fontsize=7)
            ax.set_yticks(range(len(rows)),[r.get("experiment_id",r.get("prototype_id","NA")) for r in rows],fontsize=6)
            ax.set_xticks(range(3),categories)
        single("Prototype mathematics benefit and utility",prototype_results,
               "GOOD indicates the declared check passed; BAD failed; NA is unavailable. A mathematical check or completed job is not an accuracy/cost victory.",count=len(data.get("prototypes",[])))
        claims = [dict(row, family=row.get("candidate_family","unclassified")) for table in ("claims_accuracy","claims_cost") for row in data.get(table,[])]
        def claim_draw(ax, rows):
            if not rows:
                _blank(ax); return 0
            for i,row in enumerate(rows):
                ratio=row.get("ratio",row.get("geometric_control_over_candidate_rms",row.get("geometric_control_over_candidate_cost")))
                if finite(ratio):
                    appearance=style(row["family"],row)
                    ax.plot(ratio,i,marker=appearance["marker"],color=appearance["color"],linestyle="None",markersize=4)
                ax.text(.98,i,row.get("verdict","NA"),ha="right",va="center",transform=ax.get_yaxis_transform(),fontsize=7)
            ax.set_yticks(range(len(rows)),[r.get("control_family","unknown") for r in rows],fontsize=7)
            ax.axvline(1.,color="#333333",linewidth=.5,linestyle=":")
            ax.set_xlabel("Control / candidate ratio (higher favors candidate); verdict also charges coverage and uncertainty")
            return len(rows)
        facets("Predeclared scientific claims",claims,("claim","track","final_time"),claim_draw,
               "Higher effect ratio favors the candidate; GOOD/BAD/NA comes from the sealed decision, not the renderer. Scores/check completion are distinct from model quality.")

    _write_json(output/"chart-data.json.gz",plotted_data,compressed=True)
    manifest=dict(schema=SCHEMA,profile=data.get("profile"),panels=panels,
                  chart_data="chart-data.json.gz",chart_data_sha256=hashlib.sha256((output/"chart-data.json.gz").read_bytes()).hexdigest(),
                  pdf="advance-atlas.pdf",pdf_sha256=hashlib.sha256((output/"advance-atlas.pdf").read_bytes()).hexdigest(),
                  counts={key:len(data.get(key,[])) for key in TABLES},sources=data.get("sources",[]),
                  omissions=data.get("omissions",[]),guidance=GUIDANCE,scope=SCOPE,
                  renderer_establishes_scientific_success=False,raw_stage_artifacts_preserved=True,
                  uncertainty="Descriptive paired independent-field and crossed-training-seed intervals; at least five fields; no multiplicity correction. Observed-range learning bands are separate.")
    _write_json(output/"manifest.json",manifest)
    blocks="".join('<section><h2>'+html.escape(p['title'])+'</h2><p>'+html.escape(p['reading_guide'])+'</p><p>'+html.escape(p['scope'])+'</p><img loading="lazy" src="'+html.escape(p['path'])+'" alt="'+html.escape(p['title'])+'"><p>Status: '+p['status']+'; source observations: '+str(p['observations'])+'</p></section>' for p in panels)
    (output/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>TDN advance analytical atlas</title><style>body{font:16px system-ui;margin:2rem;max-width:1500px;color:#172437;background:#f5f7fa}section{background:white;padding:1rem;margin:2rem 0}img{max-width:100%;height:auto}a{color:#175c8f}</style><h1>TDN advance analytical atlas</h1><p>'+html.escape(GUIDANCE)+'</p><p>'+html.escape(SCOPE)+'</p><p><a href="advance-atlas.pdf">Vector PDF</a> · <a href="chart-data.json.gz">Canonical chart values and source pointers</a> · <a href="manifest.json">Provenance and omissions</a></p>'+blocks)
    return manifest


def run(ctx):
    data=collect(ctx)
    manifest=build_figures(data,Path(ctx.path)/"figures",budget=ctx.budget)
    summary=dict(panels=len(manifest["panels"]),chart_data="figures/chart-data.json.gz",
                 atlas="figures/advance-atlas.pdf",html="figures/index.html",
                 source_files=len(data["sources"]),omissions=len(data["omissions"]),
                 verified_stages=sum(s["verified"] for s in data["stage_status"]),
                 scientific_outcome="ANALYTICAL_REVIEW_ONLY",counts=manifest["counts"])
    _write_json(Path(ctx.path)/"analysis.json",summary)
    return summary
