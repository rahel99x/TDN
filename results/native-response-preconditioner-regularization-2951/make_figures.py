#!/usr/bin/env python3
"""Plot retained P1R job 2951 results, without fitting or model evaluation.

Use --run EXTRACTED_RUN --output FIGURES_DIRECTORY. JSON/JSONL observations
are the only inputs; all figures are descriptive previously inspected data.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from review_resources import guard_windows, read, read_lines


CELLS = ["p008-s08", "p032-s02", "p032-s08"]
CELL_LABELS = {"p008-s08": "8 parents × 8 directions", "p032-s02": "32 parents × 2 directions",
               "p032-s08": "32 parents × 8 directions"}
COLORS = {"original": "#777777", "coefficient_ridge": "#0072B2", "action_energy": "#D55E00", "projection": "#222222"}
MARKERS = {"original": "s", "coefficient_ridge": "o", "action_energy": "^", "projection": "D"}
LABELS = {"original": "Original P1", "coefficient_ridge": "Selected coefficient penalty",
          "action_energy": "Selected action-energy penalty", "projection": "Unregularized projection (diagnostic)"}
CELL_COLORS = dict(zip(CELLS, ["#0072B2", "#D55E00", "#009E73"]))
FAMILY_LABELS = {"smooth_spectral": "Smooth spectral", "localized_periodic": "Localized periodic",
                 "periodic_transition": "Periodic transition", "multiscale_discrete": "Multiscale discrete"}


def finish(plt, fig, output, name, caption):
    fig.savefig(output/f"{name}.png", dpi=200, metadata={"Description": caption})
    svg_path = output/f"{name}.svg"
    fig.savefig(svg_path, metadata={"Date": None, "Description": caption})
    # Normalize only this newly generated review artifact. Matplotlib emits
    # trailing XML indentation; removing it leaves labels and geometry intact.
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines())+"\n")
    plt.close(fig)


def accuracy(plt, np, groups, output):
    rows = {r["method"]: r for r in groups if r["split"] == "diagnostic"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6), gridspec_kw={"width_ratios": [1, 1.15]})
    fig.subplots_adjust(left=.07, right=.98, bottom=.26, top=.78, wspace=.61)
    fig.suptitle("Regularization helps the larger cohorts; classical accuracy remains ahead", x=.07, y=.97,
                 ha="left", fontsize=15, fontweight="bold")
    fig.text(.07, .90, "Eight diagnostic parents · same 5 × 5 periodic finite-difference grid · physical response RMS, lower is better", fontsize=10)
    for offset, kind in zip([-.24, -.08, .08, .24], COLORS):
        values = [rows[kind+"-"+cell]["rms"]*1e7 for cell in CELLS]
        axes[0].scatter(np.arange(3)+offset, values, s=55, marker=MARKERS[kind],
                        facecolors="none" if kind == "projection" else COLORS[kind],
                        edgecolors=COLORS[kind], linewidths=1.3, label=LABELS[kind], zorder=3)
    axes[0].axhline(rows["ordered_8"]["rms"]*1e7, color="#999999", ls="--", lw=1)
    axes[0].text(2.35, rows["ordered_8"]["rms"]*1e7+.045, "Ordered 8 panels", ha="right", fontsize=8, color="#555555")
    axes[0].set_xticks(range(3), ["8 × 8", "32 × 2", "32 × 8"])
    axes[0].set_xlim(-.45, 2.45)
    axes[0].set_ylim(.65, 2.2)
    axes[0].set_xlabel("Training parents × calibration directions/context")
    axes[0].set_ylabel("Physical RMS error (×10⁻⁷)")
    axes[0].set_title("a  Retained-data comparison", loc="left")
    selected = [kind+"-p032-s08" for kind in COLORS] + ["ordered_2", "ordered_8", "lawson_2", "lawson_4", "lawson_8"]
    names = ["Original P1", "Selected coefficient penalty", "Selected action-energy penalty", "Projection diagnostic",
             "Ordered 2 panels", "Ordered 8 panels", "Lawson 2 steps", "Lawson 4 steps", "Lawson 8 steps"]
    for index, method in enumerate(selected):
        kind = next((k for k in COLORS if method.startswith(k+"-")), "classical")
        axes[1].scatter(rows[method]["rms"], index, s=42,
                        marker=MARKERS.get(kind, "o"),
                        facecolors="none" if kind == "projection" else COLORS.get(kind, "#009E73"),
                        edgecolors=COLORS.get(kind, "#009E73"), zorder=3)
    axes[1].set_yticks(range(9), names)
    axes[1].invert_yaxis()
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Physical RMS error (logarithmic scale)")
    axes[1].set_title("b  32 × 8 cohort and classical controls", loc="left")
    uncertainty = rows["coefficient_ridge-p032-s08"]["reference_uncertainty_rms"]
    axes[1].axvline(uncertainty, color="#aaaaaa", ls=":", lw=1)
    axes[1].text(uncertainty*1.3, 1.0, "Pooled empirical\nreference uncertainty", fontsize=8, color="#666666")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(.06, .12), ncol=2, frameon=False, fontsize=9)
    fig.text(.07, .035, "RMS aggregates equal-parent mean squared error. Projection is an offline diagnostic, not a selected deployable result.\n"
             "Reference uncertainty is empirical, not certified. Reused development observations provide no confidence interval or new speedup claim.", fontsize=8.5)
    finish(plt, fig, output, "accuracy_comparison", "Three retained cohorts and same-bank classical diagnostic accuracy; no new inference timing.")


def paths(plt, np, cells, output):
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 5.7))
    fig.subplots_adjust(left=.065, right=.98, bottom=.25, top=.78, wspace=.30)
    fig.suptitle("Validation paths separate penalty strength from label-uncertainty sensitivity", x=.065,
                 y=.97, ha="left", fontsize=14, fontweight="bold")
    fig.text(.065, .90, "Eight validation parents · stars mark sealed validation selections · only retained candidates are shown", fontsize=10)
    for cell in cells:
        cid, color = cell["cell_id"], CELL_COLORS[cell["cell_id"]]
        for ax, family in zip(axes[:2], ["coefficient_ridge", "action_energy"]):
            candidates = sorted([c for c in cell["candidates"] if c["family"] == family], key=lambda c:c["ridge"])
            ax.plot([c["ridge"] for c in candidates], [np.sqrt(c["validation_mse"]) for c in candidates],
                    color=color, lw=1.6, marker="o", ms=4, label=CELL_LABELS[cid])
            selected = cell["selected"][family]
            ax.scatter(selected["ridge"], np.sqrt(selected["validation_mse"]), marker="*", s=130,
                       color=color, edgecolor="#222222", linewidth=.5, zorder=4)
        candidates = sorted([c for c in cell["candidates"] if c["family"] == "coefficient_ridge"], key=lambda c:c["ridge"])
        axes[2].plot([c["ridge"] for c in candidates],
                     [100*c["sensitivity"]["estimated_ball_prediction_to_validation_error_rms"] for c in candidates],
                     color=color, lw=1.6, marker="o", ms=4)
        selected = cell["selected"]["coefficient_ridge"]
        axes[2].scatter(selected["ridge"], 100*selected["sensitivity"]["estimated_ball_prediction_to_validation_error_rms"],
                        marker="*", s=130, color=color, edgecolor="#222222", linewidth=.5, zorder=4)
    for ax in axes:
        ax.set_xscale("log"); ax.set_yscale("log")
    for ax in (axes[0], axes[2]):
        ax.axvline(1e-10, color="#888888", lw=1., ls="--")
        ax.set_xticks([1e-18, 1e-14, 1e-10, 1e-6])
        ax.set_xlabel("Coefficient penalty λ")
    axes[1].set_xlabel("Action-energy penalty α")
    axes[1].set_xticks([1e-6, 1e-2, 1e2])
    axes[0].set_title("a  Coefficient penalty", loc="left")
    axes[1].set_title("b  Action-energy penalty", loc="left")
    axes[2].set_title("c  Estimated perturbation effect", loc="left")
    for ax in axes[:2]:
        ax.set_ylabel("Validation physical RMS error")
    axes[2].set_ylabel("Estimated prediction change / validation error (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(.055, .12), ncol=3, frameon=False)
    fig.text(.065, .035, "Dashed line: original λ = 10⁻¹⁰. λ and α define different penalties and are not interchangeable scales.\n"
             "Sensitivity uses the declared empirical training-label uncertainty ball at fixed design; it is neither a certificate nor a selection gate.", fontsize=8.5)
    finish(plt, fig, output, "validation_and_sensitivity", "Validation regularization paths and fixed-design empirical uncertainty sensitivity for three cohorts.")


def parent_changes(plt, np, groups, parents, output):
    rows = {g["method"]: g for g in groups if g["split"] == "diagnostic"}
    people = sorted([p for p in parents if p["split"] == "diagnostic"], key=lambda p:p["parent_index"])
    labels = [f"D{i+1:02d} · {FAMILY_LABELS[p['parent_family']].lower()}" for i, p in enumerate(people)]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 6.0), sharey=True)
    fig.subplots_adjust(left=.205, right=.98, bottom=.24, top=.77, wspace=.19)
    fig.suptitle("Larger-cohort gains are consistent across the eight diagnostic parents", x=.035,
                 y=.97, ha="left", fontsize=14, fontweight="bold")
    fig.text(.035, .90, "Each point compares the same parent with its original P1 model · negative change is better · horizontal scales differ", fontsize=10)
    for ax, cell in zip(axes, CELLS):
        original = {p["parent_id"]:p["rms"] for p in rows["original-"+cell]["parents"]}
        for offset, kind in zip([-.17, 0, .17], ["coefficient_ridge", "action_energy", "projection"]):
            changed = {p["parent_id"]:p["rms"] for p in rows[kind+"-"+cell]["parents"]}
            values = [100*(changed[p["parent_id"]]/original[p["parent_id"]]-1) for p in people]
            ax.scatter(values, np.arange(8)+offset, s=35, marker=MARKERS[kind],
                       facecolors="none" if kind == "projection" else COLORS[kind], edgecolors=COLORS[kind], label=LABELS[kind], zorder=3)
        ax.axvline(0, color="#555555", lw=1)
        ax.set_yticks(range(8), labels)
        ax.set_ylim(7.6, -.6)
        ax.set_xlabel("Change in parent RMS (%)")
        ax.set_title(CELL_LABELS[cell], loc="left")
        ax.grid(axis="y", visible=False)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower left", bbox_to_anchor=(.025, .11), ncol=3, frameon=False, fontsize=9)
    fig.text(.035, .035, "RMS change = 100 × (revised / original − 1). Parent values average six intervals and four withheld directions.\n"
             "Eight development parents, two per morphology family; no bootstrap interval or independent-confirmation claim.", fontsize=8.5)
    finish(plt, fig, output, "paired_parent_changes", "Paired diagnostic-parent RMS changes after retained-data regularization, compared with each cell's original P1 model.")


def concentration(plt, np, metrics, output):
    rows = [r for r in metrics if r["split"] == "diagnostic" and r["method"] == "coefficient_ridge-p032-s08"]
    assert len(rows) == 192 and len({r["parent_id"] for r in rows}) == 8
    assert all(sum(r["parent_id"] == p for r in rows) == 24 for p in {r["parent_id"] for r in rows})
    total = sum(r["mse"] for r in rows)
    partitions = [{}, {}, {}]
    for row in rows:
        keys = (f"[{row['start']:g}, {row['stop']:g}]", FAMILY_LABELS[row["parent_family"]],
                row["family"].replace("_", " ").capitalize())
        for partition, key in zip(partitions, keys):
            partition[key] = partition.get(key, 0.) + 100*row["mse"]/total
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
    fig.subplots_adjust(left=.08, right=.98, bottom=.23, top=.77, wspace=.58)
    fig.suptitle("The remaining error is concentrated in specific conditions", x=.08, y=.97,
                 ha="left", fontsize=15, fontweight="bold")
    fig.text(.08, .90, "Selected coefficient penalty, 32 parents × 8 directions · each panel partitions the same diagnostic mean squared error", fontsize=10)
    for ax, partition, title in zip(axes, partitions, ["a  Absolute-time interval", "b  Parent morphology", "c  Withheld source family"]):
        labels = sorted(partition, key=partition.get, reverse=True)
        values = [partition[label] for label in labels]
        ax.barh(range(len(labels)), values, height=.55, color="#0072B2")
        for y, value in enumerate(values):
            ax.text(value+1., y, f"{value:.2f}%", va="center", fontsize=9)
        ax.set_yticks(range(len(labels)), labels)
        ax.invert_yaxis(); ax.set_xlim(0, 114)
        ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xlabel("Share of total diagnostic MSE (%)")
        ax.set_title(title, loc="left"); ax.grid(axis="y", visible=False)
    fig.text(.08, .045, "Each parent has 24 equally weighted queries, so these contributions preserve equal parent weight.\n"
             "The panels are alternative partitions, not additive causes; concentration does not by itself identify the missing mechanism.", fontsize=8.5)
    finish(plt, fig, output, "error_concentration", "Descriptive interval, morphology and source-family contributions to largest selected-model diagnostic MSE.")


def resources(plt, np, run, output):
    status = read(run/"outputs/native/status.json")
    raven = read(run/"run.json")
    worker = read_lines(run/"outputs/native/telemetry/cpu-memory.jsonl")
    outer = read_lines(run/"telemetry/cpu-memory.jsonl")
    start = raven["start"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.8), sharex=True)
    fig.subplots_adjust(left=.09, right=.98, bottom=.20, top=.82, hspace=.35)
    fig.suptitle("Resource observations distinguish the worker from Raven's supervisor", x=.09,
                 y=.97, ha="left", fontsize=14, fontweight="bold")
    fig.text(.09, .91, "One-second process-tree snapshots · includes verification, retained-data audit and publication · no GPU measurements", fontsize=10)
    for ax in axes:
        for name, window in guard_windows(status).items():
            ax.axvspan(window["start"]-start, window["stop"]-start, alpha=.08,
                       color="#0072B2" if name == "verify" else "#009E73", zorder=0)
    axes[0].plot([r["timestamp"]-start for r in worker], [r["rss_bytes"]/2**30 for r in worker],
                 color="#0072B2", lw=1.6, label="Native worker tree")
    axes[0].plot([r["timestamp"]-start for r in outer], [r["rss_bytes"]/2**30 for r in outer],
                 color="#777777", ls="--", lw=1.4, label="Outer Raven supervisor tree")
    axes[0].set_ylabel("Summed sampled RSS (GiB)")
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    axes[0].set_ylim(-.08, 2.15)
    worker_peak = max(r["rss_bytes"] for r in worker)/2**30
    supervisor_peak = max(r["rss_bytes"] for r in outer)/2**20
    axes[0].text(.99, .92, f"Worker peak {worker_peak:.3f} GiB; outer supervisor {supervisor_peak:.2f} MiB",
                 ha="right", transform=axes[0].transAxes, fontsize=9)
    axes[1].plot([r["timestamp"]-start for r in worker],
                 [np.nan if r.get("cpu_utilization_percent") is None else r["cpu_utilization_percent"] for r in worker],
                 color="#0072B2", lw=1.4, marker=".", ms=3)
    axes[1].set_ylabel("Worker CPU utilization (%)\n100% = one logical CPU")
    axes[1].set_xlabel("Seconds since Raven attempt start")
    axes[1].set_ylim(0, 120)
    axes[1].text(.02, .10, "Gaps preserve unavailable CPU intervals; zero is not substituted.", transform=axes[1].transAxes, fontsize=9)
    fig.text(.09, .035, "Blue shading: verification pre/post identity guards. Green shading: study guards; these are not exact command start/stop times.\n"
             "RSS snapshots may repeat shared pages and miss transient peaks. Current process-tree CPU seconds exclude departed processes.\n"
             "The supervisor tree did not observe the Slurm worker. These are workload observations, not per-method inference costs or final scheduler accounting.", fontsize=8.5)
    finish(plt, fig, output, "resource_observations", "Native worker and Raven outer-supervisor process-tree resource snapshots, with unavailable CPU intervals retained.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output.resolve()/".matplotlib-cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(args.output.resolve()/".font-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 11,
                        "axes.spines.top": False, "axes.spines.right": False,
                        "axes.grid": True, "grid.alpha": .30, "grid.linewidth": .6,
                        "svg.fonttype": "none", "svg.hashsalt": "tdn-native-p1r-2951"})
    stage = args.run/"outputs/study"
    summary = read(stage/"summary.json")
    assert summary["status"] == "COMPLETED" and summary["measured_metrics"] == 19584
    assert summary["fresh_reference_solves"] == summary["new_timing_bundles"] == 0
    groups, selection = read(stage/"learning_curves.json"), read(stage/"selection.json")
    assert len([g for g in groups if g["split"] == "diagnostic"]) == 17
    assert all(g["complete"] and g["expected_parent_count"] == 8 for g in groups if g["split"] == "diagnostic")
    accuracy(plt, np, groups, args.output)
    paths(plt, np, selection["cells"], args.output)
    parent_changes(plt, np, groups, read(stage/"parents.json"), args.output)
    concentration(plt, np, read(stage/"metrics.json"), args.output)
    resources(plt, np, args.run, args.output)
    for path in sorted(args.output.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
