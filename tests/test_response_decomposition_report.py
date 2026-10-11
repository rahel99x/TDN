"""Report tests protect signed cancellation, missing evidence and paired scope."""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from tdn.analysis.response import report


def _data(filename, rows, **metadata):
    source = "stage/" + filename
    return {"scope": "Unsealed test fixture", "tables": {
        source: {"filename": filename, "payload": {"rows": rows, **metadata}}
    }}


@pytest.fixture
def figures(monkeypatch):
    saved = []
    def save(fig, root, name, caption, output, points, scope):
        saved.append(dict(fig=fig, name=name, caption=caption, points=points))
    monkeypatch.setattr(report, "_save", save)
    yield saved
    for entry in saved:
        plt.close(entry["fig"])


def _decomposition_row():
    # Three collinear error vectors have large individual energies but nearly
    # cancel; plotting unsigned cross terms would completely invert the result.
    names = ["discrete_solver_error", "spatial_discretization", "unresolved_scale_closure"]
    coefficients = [.11, -.3, .2]
    squared = {key: value*value/2 for key, value in zip(names, coefficients)}
    cross = [dict(left=names[i], right=names[j],
                  signed_twice_inner_product=coefficients[i]*coefficients[j],
                  estimated_twice_inner_product_sensitivity=1e-6)
             for i in range(3) for j in range(i+1, 3)]
    total = sum(coefficients)**2/2
    assert sum(squared.values()) + sum(c["signed_twice_inner_product"] for c in cross) == pytest.approx(total)
    return dict(method="df", norm_target="galerkin_same_grid", reference_accepted=True,
        metrics=dict(component_squared_norms=squared, cross_terms=cross, total_squared_norm=total),
        primary_total_uncertainty={"uncertainty_rms": 1e-7})


def test_signed_error_budget_keeps_negative_cross_terms_and_actual_total(figures, tmp_path):
    row = _decomposition_row()
    report._decomposition(_data("decomposition.json", [row], grid=[8, 8], horizon=.04), tmp_path, [])
    assert len(figures) == 1
    ax = figures[0]["fig"].axes[0]
    widths = [bar.get_width() for bar in ax.patches]
    assert widths[3:] == pytest.approx([-.033, .022, -.06])
    assert ax.collections[0].get_offsets()[0, 0] == pytest.approx(row["metrics"]["total_squared_norm"])
    assert "galerkin_same_grid" in figures[0]["caption"]
    assert "not CI" in figures[0]["caption"]
    assert figures[0]["points"][0]["json_pointer"] == "/rows/0"
    assert "Cross terms alone do not measure quality" in figures[0]["caption"]


def test_spectral_stress_uses_gaps_and_never_pools_targets(figures, tmp_path):
    rows = []
    for target in ["fd_same_grid", "galerkin_same_grid"]:
        for steps in [1, 2, 4]:
            rows.append(dict(kind="grid_relative", method=f"df/steps-{steps}", cluster_parent_id="parent-a",
                metrics=dict(target_kind=target, grid=[8, 8], horizon=.04, rms=.02/steps,
                    maximum=.03/steps, reference_accepted=steps != 2, prediction_finite=True)))
    report._decomposition(_data("spectral_stress.json", rows), tmp_path, [])
    assert len(figures) == 2
    for entry in figures:
        ax = entry["fig"].axes[0]
        line = ax.lines[0]
        np.testing.assert_allclose(line.get_xdata(), [1, 2, 4])
        assert math.isnan(line.get_ydata()[1])
        assert line.get_linewidth() < 1
        assert len(entry["points"]) == 2
        assert len({point["row"]["metrics"]["target_kind"] for point in entry["points"]}) == 1
        assert "lower is better" in ax.get_ylabel()


def _phase(method, variant, error, **extra):
    return dict(method=method, phase_scope="relative-triad-phase intervention at identical continuous modal amplitudes",
        phase_variant=variant, mean_error=error, target_kind="galerkin_same_grid", grid=[8, 8],
        regime="smooth", parent_id="parent-a", seed=3, horizon=.04, steps=1,
        reference_accepted=True, prediction_finite=True, **extra)


def test_phase_pairs_preserve_ownership_and_exclude_translation_controls(figures, tmp_path):
    rows = [_phase("coupled", "original", .01), _phase("coupled", "phase_twin", .02),
            _phase("fno", "original", .03), _phase("fno", "phase_twin", -.01),
            _phase("df", "original", .4)]  # unmatched original is not zero-imputed
    translated = [_phase("translated", "original", 40), _phase("translated", "phase_twin", 40)]
    for row in translated:
        row["phase_scope"] = "translation-only or nonclosed-mode control; no higher-interaction claim"
    rows += translated
    report._decomposition(_data("endpoint_metrics.json", rows), tmp_path, [])
    assert len(figures) == 1
    entry = figures[0]; ax = entry["fig"].axes[0]
    np.testing.assert_allclose(ax.collections[0].get_offsets(), [[.01, .02]])
    np.testing.assert_allclose(ax.collections[1].get_offsets(), [[.03, -.01]])
    labels = ax.get_legend_handles_labels()[1]
    assert "Ours: coupled" in labels
    assert "Theirs (adapted): fno" in labels
    assert len(entry["points"]) == 4
    assert {p["json_pointer"] for p in entry["points"]} == {"/rows/0", "/rows/1", "/rows/2", "/rows/3"}
    assert any("1 incomplete/unresolved" in text.get_text() for text in ax.texts)
    # The first scatter marker is a triangle; the second has a circle's many
    # curved vertices, independently checking styling beyond label contents.
    assert len(ax.collections[0].get_paths()[0].vertices) == 4
    assert len(ax.collections[1].get_paths()[0].vertices) > 4


def test_signed_figure_writes_high_resolution_and_source_links(tmp_path):
    from PIL import Image
    output = []
    report._decomposition(_data("decomposition.json", [_decomposition_row()], grid=[8, 8], horizon=.04), tmp_path, output)
    assert len(output) == 1
    assert (tmp_path / output[0]["pdf"]).is_file()
    with Image.open(tmp_path / output[0]["png"]) as im:
        assert im.width > 2500
        assert im.info["dpi"][0] == pytest.approx(400, abs=.1)
    assert output[0]["plotted_records"] == [{"source_id": "stage/decomposition.json", "json_pointer": "/rows/0"}]
