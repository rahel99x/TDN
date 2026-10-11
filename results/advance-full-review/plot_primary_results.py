#!/usr/bin/env python3
"""Reproduce the review graphic without checkpoints, teachers, or model execution.

Run with the repository venv:
    .venv/bin/python results/advance-full-review/plot_primary_results.py

The adjacent JSON preserves the original claim records and source SHA-256s.
Neither conditional timing ratios nor individual seed/query rows are reaveraged.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO / ".cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


DATA = json.loads((BASE / "primary-results-data.json").read_text())
CONTROLS = ["commutator_cubic", "quad4_full", "quad2_conditioned", "fno_scaled", "etdrk4"]
SCENARIOS = [("discrete", .12), ("discrete", .24), ("continuum", .12), ("continuum", .24)]
LABELS = {
    "two_basis": "Ours: two fitted coefficients",
    "commutator_cubic": "Analytic (Ours): fixed cubic",
    "quad4_full": "Analytic: full four-node rule",
    "quad2_conditioned": "Ours: conditioned two-node",
    "fno_scaled": "Theirs: local scaled FNO",
    "etdrk4": "Theirs: ETDRK4",
}
COLORS = {
    "two_basis": "#0569a6", "commutator_cubic": "#a66b00", "quad4_full": "#8b4ba8",
    "quad2_conditioned": "#2b9eac", "fno_scaled": "#bf4969", "etdrk4": "#468454",
}
MARKERS = {m: ("s" if m in ["commutator_cubic", "quad4_full"] else
               "o" if m in ["fno_scaled", "etdrk4"] else "^") for m in LABELS}


def lookup(kind, track, time, control):
    rows = [r for r in DATA["claims"][kind] if r["track"] == track and
            r["final_time"] == time and r["control_family"] == control]
    assert len(rows) == 1
    return rows[0]


def no_validation_schedule(method, track, time):
    rows = [r for r in DATA["validation_frontier_primary_subset"] if
            r["model_id"].split("/")[1] == method and r["track"] == track and
            r["final_time"] == time]
    assert rows
    return all(r["status"] == "NO_FEASIBLE_VALIDATION_SCHEDULE" for r in rows)


def main():
    source = DATA["provenance"]["aggregate/claims.json"]
    assert hashlib.sha256((BASE / "claims.json").read_bytes()).hexdigest() == source["sha256"]
    assert DATA["claims"] == json.loads((BASE / "claims.json").read_bytes())
    assert Counter(r["verdict"] for r in DATA["claims"]["accuracy"]) == {"GOOD": 14, "BAD": 6}
    assert Counter(r["verdict"] for r in DATA["claims"]["cost"]) == {"BAD": 14, "NA": 6}

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.linewidth": .6, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(18, 13), facecolor="white")
    fig.text(.055, .965, "TDN Advance | promising corrections, no confirmed solver-cost advantage",
             fontsize=23, weight="bold", va="top", color="#16314a")
    fig.text(.055, .930, "Fresh confirmation: 64 independent fields • 3 paired training seeds • 32² and 64² grids • RTX 4090",
             fontsize=13, color="#3d4d59", va="top")

    fig.text(.055, .884, "A  Same-schedule RMS accuracy", fontsize=17, weight="bold", va="top")
    fig.text(.055, .858, "Control error / Ours two-basis error: higher favors Ours; 1 = equal. Color shows ratio, text shows the frozen claim verdict.",
             fontsize=11.5, va="top")
    ax = fig.add_axes([.155, .532, .795, .305])
    ratios = np.array([[lookup("accuracy", tr, t, c)["ratio"] for c in CONTROLS]
                       for tr, t in SCENARIOS])
    norm = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=4.25)
    im = ax.imshow(np.log2(ratios), cmap="RdBu", norm=norm, aspect="auto")
    ax.set_yticks(range(4), [f"{tr.capitalize()}\nt = {t:.2f}" for tr, t in SCENARIOS], fontsize=12)
    ax.set_xticks(range(5), ["Analytic (Ours)\nFixed cubic", "Analytic\nFull four-node",
                            "Ours\nConditioned two-node", "Theirs\nLocal scaled FNO",
                            "Theirs\nETDRK4"], fontsize=11.5)
    ax.tick_params(axis="both", length=0, pad=9)
    ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    for i, (track, time) in enumerate(SCENARIOS):
        for j, control in enumerate(CONTROLS):
            row = lookup("accuracy", track, time, control)
            lo, hi = (math.exp(row["interval"][k]) for k in ["lower", "upper"])
            color = "white" if abs(norm(math.log2(row["ratio"]))-.5) > .28 else "#102b40"
            ax.text(j, i-.22, f'{row["ratio"]:.2f}×', ha="center", va="center", weight="bold", fontsize=18, color=color)
            ax.text(j, i+.04, f"95% [{lo:.2f}, {hi:.2f}]", ha="center", va="center", fontsize=10.5, color=color)
            ax.text(j, i+.29, row["verdict"], ha="center", va="center", fontsize=11, weight="bold", color=color)
    cax = fig.add_axes([.962, .532, .010, .305])
    bar = fig.colorbar(im, cax=cax, ticks=[-2, -1, 0, 1, 2, 3, 4])
    bar.ax.set_yticklabels(["¼×", "½×", "1×", "2×", "4×", "8×", "16×"])
    bar.ax.tick_params(labelsize=10, length=2)
    fig.text(.155, .470, "GOOD: ratio ≥1.10, interval lower bound >1, ≤5 percentage-point pass-coverage loss. BAD: criterion unmet, not necessarily worse.",
             fontsize=10.6, va="top")
    fig.text(.155, .451, "Intervals: descriptive paired field/seed bootstrap, 1,000 replicates, unadjusted for multiple comparisons; repeated grids are not independent samples.",
             fontsize=10.6, va="top", color="#4a535a")

    fig.text(.055, .410, "B  Accuracy coverage under validation-locked schedules", fontsize=17, weight="bold", va="top")
    fig.text(.055, .384, "Higher coverage is better. A field must pass joint RMS/max ≤2×10⁻⁵ across its paired grids and training seeds; dashed line = 95% requirement.",
             fontsize=11.5, va="top")
    methods = ["two_basis"] + CONTROLS
    positions = [.260, .440, .620, .800]
    for k, (track, time) in enumerate(SCENARIOS):
        axis = fig.add_axes([positions[k], .146, .157, .202])
        axis.axvspan(95, 105, color="#eaf4ee", zorder=0)
        axis.axvline(95, color="#35503d", linestyle="--", linewidth=.8)
        rows = [lookup("cost", track, time, c) for c in CONTROLS]
        assert len({r["candidate_joint_field_coverage"] for r in rows}) == 1
        values = [rows[0]["candidate_joint_field_coverage"]] + [r["control_joint_field_coverage"] for r in rows]
        for y, (method, value) in enumerate(zip(methods, values)):
            pct = 100*value
            absent = no_validation_schedule(method, track, time)
            assert not absent or pct == 0
            axis.hlines(y, 0, pct, colors=COLORS[method], linewidth=.7, alpha=.45)
            axis.plot(pct, y, marker=MARKERS[method], markersize=7,
                      markerfacecolor="white" if absent else COLORS[method],
                      markeredgecolor=COLORS[method], markeredgewidth=.9, linestyle="none")
            text = "0%: no schedule" if absent else f"{pct:.1f}%"
            axis.annotate(text, (pct, y), xytext=(5 if pct < 80 else -6, 0), textcoords="offset points",
                          ha="left" if pct < 80 else "right", va="center", fontsize=9.5, color="#253b4c")
        axis.set_title(f"{track.capitalize()} · t={time:.2f}", fontsize=11.5, weight="bold", pad=10)
        axis.set_xlim(-4, 107)
        axis.set_ylim(5.65, -.65)
        axis.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100"])
        axis.set_yticks(range(6), [LABELS[m] for m in methods] if k == 0 else [""]*6, fontsize=11)
        axis.tick_params(axis="both", length=0, labelsize=10)
        axis.tick_params(axis="y", pad=10)
        axis.set_xlabel("Joint field coverage (%)", fontsize=10.5)
        axis.grid(axis="x", linewidth=.4, alpha=.20)
        axis.spines["left"].set_visible(False)
    fig.text(.055, .093, "Cost claims: 0 GOOD · 14 BAD · 6 NA. Conditional timing ratios cannot rescue insufficient coverage; no deployed-policy advantage is established.",
             fontsize=12, weight="bold", color="#16314a", va="top")
    fig.text(.055, .068, "Open markers denote no feasible schedule selected on validation (operational coverage 0%), not a measurement of all possible solver configurations.",
             fontsize=10.5, va="top", color="#4a535a")
    fig.text(.055, .048, "Triangles: Ours learned models. Squares: analytic controls (including our fixed cubic). Circles: Theirs. Local FNO control is not a published-paper reproduction.",
             fontsize=10.5, va="top", color="#4a535a")
    fig.text(.055, .027, "Source a764670 · fedora-advance-full-20261011T001715 · exact primary claims retained in primary-results-data.json · 128² belongs to separate scaling only",
             fontsize=9.5, va="top", color="#65717a")
    for suffix in ["png", "pdf"]:
        fig.savefig(BASE / f"advance-full-primary-results.{suffix}", dpi=300, facecolor="white")
    plt.close(fig)
    print("Verified 20 accuracy and 20 cost claim records; wrote PNG (5400×3900) and vector PDF.")


if __name__ == "__main__":
    main()
