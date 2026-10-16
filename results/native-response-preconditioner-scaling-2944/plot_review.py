#!/usr/bin/env python3
"""Render the retained native P1 observations; never fit or evaluate a model.

Usage:
  .venv/bin/python results/native-response-preconditioner-scaling-2944/plot_review.py \
      --study path/to/outputs/study --output path/to/figures

Only JSON from the immutable study is read. PNG and SVG are exported together.
The figures are descriptive: one nested training cohort, eight diagnostic
parents, and a smaller four-parent timing subset. There are no confidence
intervals, population scaling fits, extrapolations, or new numerical solves.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


SOURCE_COLORS = {2: "#0072B2", 4: "#D55E00", 8: "#009E73"}
SOURCE_MARKERS = {2: "o", 4: "s", 8: "^"}
METHODS = ["ordered_2", "fit-p032-s08", "ordered_8", "lawson_2", "lawson_4", "lawson_8"]
METHOD_LABELS = {
    "ordered_2": "Ordered 2 panels", "ordered_8": "Ordered 8 panels",
    "fit-p032-s08": "Learned: 32 parents, 8 directions",
    "lawson_2": "Lawson 2 steps", "lawson_4": "Lawson 4 steps", "lawson_8": "Lawson 8 steps",
}
METHOD_COLORS = dict(zip(METHODS, ["#777777", "#0072B2", "#E69F00", "#D55E00", "#009E73", "#CC79A7"]))
FAMILY_LABELS = {"smooth_spectral": "smooth spectral", "localized_periodic": "localized periodic",
                 "periodic_transition": "periodic transition", "multiscale_discrete": "multiscale discrete"}


def read_json(study, name):
    return json.loads((study / f"{name}.json").read_text())


def save_figure(fig, output, stem, caption):
    fig.savefig(output / f"{stem}.png", dpi=220, facecolor="white",
                metadata={"Title": stem.replace("_", " "), "Description": caption})
    fig.savefig(output / f"{stem}.svg", facecolor="white",
                metadata={"Title": stem.replace("_", " "), "Description": caption, "Date": None})
    # Matplotlib emits trailing spaces in SVG path attributes. Keep retained
    # text clean without changing the vector drawing or any numerical values.
    svg = output / f"{stem}.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")


def learning_curves(plt, np, study, output, curves):
    rows = {row["method"]: row for row in curves if row["split"] == "diagnostic"}
    costs = {row["model_id"]: row for row in read_json(study, "costs")["cells"]}
    counts = [8, 16, 32]
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 5.6))
    fig.subplots_adjust(left=.067, right=.985, bottom=.26, top=.79, wspace=.32)
    fig.suptitle("P1: more independent parents help, but the classical accuracy gap remains", x=.067,
                 ha="left", fontsize=15, fontweight="bold", y=.98)
    fig.text(.067, .91, "Same nine sealed fits on eight diagnostic parents · 5 × 5 periodic FD grid · lower RMS is better",
             fontsize=10, color="#444444")
    for sources in [2, 4, 8]:
        cells = [f"fit-p{parents:03d}-s{sources:02d}" for parents in counts]
        error = np.array([rows[cell]["rms"] for cell in cells])
        style = dict(color=SOURCE_COLORS[sources], marker=SOURCE_MARKERS[sources], lw=1.8, ms=6,
                     label=f"{sources} calibration directions per context")
        axes[0].plot(counts, error, **style)
        axes[1].plot(counts, error * 1e7, **style)
        axes[2].plot([costs[cell]["training_reference_seconds"] for cell in cells], error * 1e7, **style)
        for cell, y in zip(cells, error * 1e7):
            axes[2].annotate(str(costs[cell]["training_parent_count"]),
                             (costs[cell]["training_reference_seconds"], y),
                             xytext=(4, 4), textcoords="offset points", fontsize=8,
                             color=SOURCE_COLORS[sources])
    controls = [("ordered_2", "--"), ("ordered_8", ":"), ("lawson_2", "-.")]
    for method, dash in controls:
        axes[0].axhline(rows[method]["rms"], color=METHOD_COLORS[method], linestyle=dash,
                       lw=1.6, label=METHOD_LABELS[method])
    axes[0].set_yscale("log")
    axes[0].set_ylim(1e-10, 2.5e-6)
    axes[0].set_ylabel("Physical response RMS error")
    axes[0].set_title("a  Classical context", loc="left")
    axes[1].set_title("b  Learned response detail", loc="left")
    axes[2].set_title("c  Training-label cost", loc="left")
    for ax in axes[:2]:
        ax.set_xticks(counts)
        ax.set_xlabel("Independent training parents")
    for ax in axes[1:]:
        ax.set_ylabel("Physical response RMS error (×10⁻⁷)")
        ax.set_ylim(1.62, 2.14)
    axes[2].set_xlabel("Observed training-reference acquisition (s)")
    axes[2].set_xscale("log")
    axes[2].set_xticks([5, 10, 20, 40, 80], labels=["5", "10", "20", "40", "80"])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(.06, .095), ncol=3,
               frameon=False, fontsize=9, columnspacing=2.0)
    fig.text(.067, .04, "RMS = square root of equally weighted parent MSE. One nested cohort; no confidence intervals.\n"
             "Cost labels give parent counts. Shared validation, design construction and fitting are excluded; overlapping cell costs are not additive.",
             fontsize=9, color="#444444", va="bottom")
    save_figure(fig, output, "learning_curves",
                "Retained P1 diagnostic learning curves, classical context, and overlapping training-reference acquisition costs.")
    plt.close(fig)


def paired_parents(plt, np, study, output, curves):
    rows = {row["method"]: row for row in curves if row["split"] == "diagnostic"}
    parents = sorted((p for p in read_json(study, "parents") if p["split"] == "diagnostic"),
                     key=lambda p: p["parent_index"])
    ids = [p["parent_id"] for p in parents]
    labels = [f"D{i + 1:02d} · {FAMILY_LABELS[p['parent_family']]}" for i, p in enumerate(parents)]
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.2), sharey=True)
    fig.subplots_adjust(left=.22, right=.98, bottom=.16, top=.70, wspace=.20)
    fig.suptitle("P1: parent-wise changes distinguish physical coverage from extra directions", x=.035,
                 ha="left", fontsize=14, fontweight="bold", y=.98)
    fig.text(.035, .91, "Eight diagnostic parents · all six intervals and four withheld directions per parent · negative values indicate improvement",
             fontsize=10, color="#444444")
    offsets = [-.20, 0., .20]
    for panel, levels in enumerate(([2, 4, 8], [8, 16, 32])):
        ax = axes[panel]
        for pos, fixed in enumerate(levels):
            before = f"fit-p008-s{fixed:02d}" if panel == 0 else f"fit-p{fixed:03d}-s02"
            after = f"fit-p032-s{fixed:02d}" if panel == 0 else f"fit-p{fixed:03d}-s08"
            a = {p["parent_id"]: p["rms"] for p in rows[before]["parents"]}
            b = {p["parent_id"]: p["rms"] for p in rows[after]["parents"]}
            changes = [100 * (b[p] / a[p] - 1) for p in ids]
            color = SOURCE_COLORS[[2, 4, 8][pos]]
            label = f"{fixed} directions/context" if panel == 0 else f"{fixed} training parents"
            ax.scatter(changes, np.arange(8) + offsets[pos], s=46, color=color,
                       marker=SOURCE_MARKERS[[2, 4, 8][pos]], label=label, zorder=3)
        ax.axvline(0, color="#333333", lw=1.1)
        ax.set_yticks(range(8), labels=labels)
        ax.set_ylim(7.6, -.6)
        ax.set_xlabel("Change in parent RMS error (%)")
        ax.grid(axis="x", color="#dddddd", lw=.7)
        ax.grid(axis="y", visible=False)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False, fontsize=9)
    axes[0].set_title("a  Training parents: 8 → 32", loc="left", y=1.24)
    axes[1].set_title("b  Calibration directions: 2 → 8", loc="left", y=1.24)
    fig.text(.035, .045, "Each point compares the same physical parent before and after scaling; each parent is one observation.\n"
             "RMS change = 100 × (after / before − 1). No confidence interval, repeated-cohort estimate or significance claim.",
             fontsize=9, color="#444444")
    save_figure(fig, output, "paired_parent_changes",
                "Descriptive paired percentage RMS changes for all eight diagnostic parents, with independent axes for parent and direction count.")
    plt.close(fig)


def work_precision(plt, np, study, output):
    work = read_json(study, "work_precision")
    timings = read_json(study, "timing")
    contexts = {row["context_id"] for row in timings}
    metrics = [row for row in read_json(study, "metrics")
               if row["context_id"] in contexts and row["method"] == "fit-p032-s08"]
    if len(contexts) != 24 or len(metrics) != 96 or len(work) != 144:
        raise ValueError("Expected the complete 24-context, 96-query, six-method timing subset")
    reference_uncertainty = np.sqrt(np.mean([row["reference_uncertainty_rms"] ** 2 for row in metrics]))
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.6), gridspec_kw={"width_ratios": [1.45, 1]})
    fig.subplots_adjust(left=.07, right=.985, bottom=.29, top=.78, wspace=.27)
    fig.suptitle("P1: learned accuracy remains behind Lawson 2 at similar observed action cost", x=.07,
                 ha="left", fontsize=14, fontweight="bold", y=.98)
    fig.text(.07, .91, "Timing subset: first diagnostic parent of each family · 24 contexts · five randomized paired rounds · CPU, one numerical thread",
             fontsize=9.5, color="#444444")
    for method in METHODS:
        rows = [row for row in work if row["method"] == method]
        if len(rows) != 24 or any(row["passed_rounds"] != 5 or row["failed_rounds"] for row in rows):
            raise ValueError(f"Incomplete timing observations for {method}")
        seconds = np.array([row["median_seconds"] for row in rows])
        errors = np.array([row["pooled_source_rms"] for row in rows])
        axes[0].scatter(seconds * 1e3, errors, s=19, color=METHOD_COLORS[method], alpha=.40,
                        linewidths=0, zorder=2)
        axes[0].scatter(np.median(seconds) * 1e3, np.sqrt(np.mean(errors ** 2)),
                        s=78, marker="D", color=METHOD_COLORS[method], edgecolor="white",
                        linewidth=.7, label=METHOD_LABELS[method], zorder=4)
    axes[0].axhline(reference_uncertainty, color="#444444", linestyle="--", lw=1.0, zorder=1)
    axes[0].text(.98, reference_uncertainty * 1.25, f"Pooled reference uncertainty ≈ {reference_uncertainty:.2e}",
                 transform=axes[0].get_yaxis_transform(), ha="right", va="bottom", fontsize=8,
                 color="#444444")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Median time for four-source bundle (ms)")
    axes[0].set_ylabel("Physical response RMS error")
    axes[0].set_title("a  Observed work–precision", loc="left")
    lookup = {(row["context_id"], row["method"]): row for row in work}
    timing_families = {row["context_id"]: row["parent_family"] for row in timings}
    family_markers = dict(zip(FAMILY_LABELS, ["o", "s", "^", "D"]))
    for family in FAMILY_LABELS:
        chosen = sorted(cid for cid in contexts if timing_families[cid] == family)
        ratios = [lookup[(cid, "fit-p032-s08")]["pooled_source_rms"] /
                  lookup[(cid, "lawson_2")]["pooled_source_rms"] for cid in chosen]
        times = [lookup[(cid, "fit-p032-s08")]["median_seconds"] /
                 lookup[(cid, "lawson_2")]["median_seconds"] for cid in chosen]
        axes[1].scatter(times, ratios, s=43, marker=family_markers[family], color="#0072B2",
                        alpha=.8, label=FAMILY_LABELS[family], edgecolor="white", linewidth=.4)
    axes[1].axvline(1., color="#555555", ls="--", lw=1.)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Time ratio: learned / Lawson 2")
    axes[1].set_ylabel("RMS error ratio: learned / Lawson 2")
    axes[1].set_title("b  Same-context comparison", loc="left")
    family_handles, family_labels = axes[1].get_legend_handles_labels()
    fig.legend(family_handles, family_labels, loc="lower left", bbox_to_anchor=(.64, .14),
               ncol=2, frameon=False, fontsize=8, columnspacing=.8, handletextpad=.4)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(.06, .14), ncol=3,
               frameon=False, fontsize=9, columnspacing=2.)
    fig.text(.07, .025, "Small points: individual contexts. Diamonds: RMS pooled across these 24 contexts; median of their five-round median times.\n"
             "Every source rebuilds path/features; no setup reuse. Offline reference/fitting cost is separate. Cold/warmup calls are excluded.\n"
             "The uncertainty line is pooled; context-specific uncertainties differ. These empirical estimates are not certificates.\n"
             "Comparisons near their reference-uncertainty scale are unresolved. No matched-tolerance speedup is established.",
             fontsize=8.7, color="#444444", va="bottom")
    save_figure(fig, output, "work_precision",
                "Retained 24-context work–precision points, equally weighted pooled summaries, and same-context learned/Lawson-2 ratios.")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output.resolve() / ".matplotlib-cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(args.output.resolve() / ".font-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .35, "grid.linewidth": .6,
                         "axes.titlesize": 11, "axes.titlepad": 11,
                         "svg.fonttype": "none", "svg.hashsalt": "tdn-p1-native-2944"})
    summary = read_json(args.study, "summary")
    if (summary.get("schema") != "tdn.response_preconditioner.scaling.v1"
            or summary.get("status") != "COMPLETED" or summary.get("selected_models") != 9
            or summary.get("measured_metrics") != 16128):
        raise ValueError("These figures require the completed, original P1 study")
    curves = read_json(args.study, "learning_curves")
    diagnostic = [row for row in curves if row["split"] == "diagnostic"]
    if len(diagnostic) != 14 or any(not row["complete"] or row["complete_parent_count"] != 8
                                  for row in diagnostic):
        raise ValueError("Expected all fourteen methods on all eight diagnostic parents")
    learning_curves(plt, np, args.study, args.output, curves)
    paired_parents(plt, np, args.study, args.output, curves)
    work_precision(plt, np, args.study, args.output)
    for stem in ["learning_curves", "paired_parent_changes", "work_precision"]:
        print(args.output / f"{stem}.png")
        print(args.output / f"{stem}.svg")


if __name__ == "__main__":
    main()
