"""Reporting must preserve missingness, raw scope and immutable provenance."""
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tdn.analysis.resolve import report as r


def source(base, files):
    base.mkdir(parents=True, exist_ok=True)
    manifest={"artifacts":{}}
    for name, value in files.items():
        text=("\n".join(json.dumps(x) for x in value)+"\n" if name.endswith(".jsonl") else json.dumps(value))
        (base/name).write_text(text)
        manifest["artifacts"][name]=hashlib.sha256((base/name).read_bytes()).hexdigest()
    (base/"science_manifest.json").write_text(json.dumps(manifest))
    return base


def context(tmp_path, **kwargs):
    values=dict(path=tmp_path/"report", stage="report",device="cpu",prerequisites={},stage_failures={},
        protocol={"profile":"smoke","units":{"fit-discrete":{"kind":"fit","track":"discrete"},
            "fairness-discrete":{"kind":"fairness","track":"discrete"},"memory":{"kind":"memory"},"report":{"kind":"report"}}},
        budget=SimpleNamespace(check=lambda:None),record=lambda *a,**kw:None)
    values.update(kwargs)
    return SimpleNamespace(**values)


def test_missing_cost_is_not_zero_and_failed_science_is_not_read(tmp_path):
    base=source(tmp_path/"fit",{"summary.json":{"elapsed_seconds":0.,"experiment_count":1},"fit_results.json":{"fits":{}}})
    # An unsealed sibling with plausible results must never enter the atlas.
    (tmp_path/"memory").mkdir();(tmp_path/"memory"/"memory_results.json").write_text('{"rows":[{"error_rms":0}]}')
    data=r.collect(context(tmp_path,prerequisites={"fit-discrete":base},
        stage_failures={"memory":{"status":"FAILED","elapsed_seconds":3.2,"error":"interrupted"}}))
    statuses={x["stage"]:x for x in data["stages"]}
    assert statuses["fit-discrete"]["elapsed_seconds"]==0.
    assert statuses["fairness-discrete"]["elapsed_seconds"] is None
    assert statuses["memory"]["elapsed_seconds"]==3.2
    assert not any("memory_results.json" in x for x in data["files"])


def test_sealed_source_change_and_during_render_change_are_rejected(tmp_path):
    base=source(tmp_path/"fit",{"fit_results.json":{"fits":{}}})
    ctx=context(tmp_path,prerequisites={"fit-discrete":base})
    data=r.collect(ctx)
    (base/"fit_results.json").write_text('{"fits":{"changed":1}}')
    with pytest.raises(ValueError,match="changed"):
        r.verify_sources(data)
    with pytest.raises(ValueError,match="sealed source changed"):
        r.collect(ctx)


def test_symlink_and_outside_source_paths_are_rejected(tmp_path):
    base=source(tmp_path/"fit",{"fit_results.json":{"fits":{}}})
    (base/"fit_results.json").unlink()
    outside=tmp_path/"outside.json";outside.write_text('{"fits":{}}')
    (base/"fit_results.json").symlink_to(outside)
    with pytest.raises(ValueError,match="symlink"):
        r.collect(context(tmp_path,prerequisites={"fit-discrete":base}))
    with pytest.raises(ValueError,match="Unsafe"):
        r._safe_path(base,"../outside.json")


def test_observed_bands_are_continuous_with_missing_measurement_gaps():
    rows=[dict(update=i,train_loss=(-1.)**i) for i in range(20)]
    rows[9]["train_loss"]=None
    bands=r.observed_ranges(rows,bins=2)
    assert len(bands)==2
    assert bands[0]["x"][0]==0 and bands[0]["x"][-1]==8
    assert bands[1]["x"][0]==10 and bands[1]["x"][-1]==19
    for b in bands:
        assert all(x<y for x,y in zip(b["x"],b["x"][1:]))
        assert min(b["lower"])==-1 and max(b["upper"])==1
        assert "not a confidence interval" in b["band_kind"]
    zeros=r.observed_ranges([dict(update=0,train_loss=0.)])
    assert zeros[0]["lower"]==[0.] and zeros[0]["upper"]==[0.]


def test_sparse_validation_is_not_bridged():
    rows=[dict(update=i,validation_loss=(1./(i+1) if i in (0,4,8) else None)) for i in range(9)]
    bands=r.observed_ranges(rows,"validation_loss")
    assert [b["x"] for b in bands]==[[0.],[4.],[8.]]


def test_compact_atlas_keeps_raw_loss_once_and_is_inspectable(tmp_path):
    curves=[dict(family="fno_scaled",track="discrete",trial_id="a",phase="equal_updates",seed=1,
        optimizer_control="unclipped",update=i,train_loss=1./(i+1),validation_loss=1./(i+1) if i in (0,7) else None)
        for i in range(8)]
    base=source(tmp_path/"fairness",{"summary.json":{"elapsed_seconds":1.},
        "fairness_results.json":{"records":[{"family":"fno_scaled","trials":[{"curves":curves}]}]},
        "learning_curves.json":{"rows":curves}})
    recorded=[]
    ctx=context(tmp_path,prerequisites={"fairness-discrete":base},record=lambda *a,**kw:recorded.append((a,kw)))
    result=r.run(ctx)
    assert result["figures"]==2 and result["report_bytes"]<3_000_000
    with gzip.open(ctx.path/"chart-data.json.gz","rt") as stream:data=json.load(stream)
    assert data["files"]["fairness-discrete/learning_curves.json"]["payload"]["rows"]==curves
    assert "curves" not in data["files"]["fairness-discrete/fairness_results.json"]["payload"]["records"][0]["trials"][0]
    assert all(set(f["source_ids"])<=set(s["source_id"] for s in data["sources"]) for f in data["figures"])
    page=(ctx.path/"index.html").read_text()
    assert "Missing evidence" in page and "confidence intervals" in page
    assert recorded[0][1]["checks"][1]["verdict"]=="NA"
    assert r.style("two_gain_direct")["marker"]=="^"
    assert r.style("fno_scaled")["marker"]=="o"
    assert r.style("quad4_full")["marker"]=="s"


def test_every_scientific_path_renders_separate_targets_and_costs(tmp_path):
    fit={"fits":{"two_gain_direct":{"gains":[1.1,.8],"elapsed_seconds":.1,
        "curves":[{"iteration":1,"objective":2.},{"iteration":2,"objective":1.}]}},
        "validation":{"two_gain_direct":{"validation_rms":1e-5}}}
    interaction={"track":"discrete","grid":8,"method":"quad4_full","case_id":"field-1",
        "diffusion_number":1.,"reaction_number":.3,"same_grid_metrics":{"error_rms":1e-5,"reference_accepted":True},
        "error_decomposition":{"component_rms":{"closure":1e-5,"discretization":2e-5}}}
    kernel={"method":"quad4_full","grid":8,"cost_seconds":.01,"error_rms":1e-5}
    profile={"method":"quad4_full","profile_seconds":.1,
        "invocation_counts":{"fft_dispatch":4,"products_and_scalar_multiplication":2}}
    memory=[{"reaction_multiplier":1.,"noise":0.,"method":"noise_aware_fitted_memory",
        "rollout_step":step,"error_rms":step*1e-5} for step in (1,2,3)]
    roots={"fit-discrete":source(tmp_path/"fit",{"fit_results.json":fit}),
        "interactions-discrete":source(tmp_path/"interactions",{"interaction_rows.json":{"rows":[interaction]}}),
        "kernels-discrete":source(tmp_path/"kernels",{"kernel_rows.json":{"rows":[kernel]},"profiling.json":{"profiles":[profile]}}),
        "memory":source(tmp_path/"memory",{"memory_results.json":{"rows":memory}})}
    ctx=context(tmp_path,prerequisites=roots)
    ctx.protocol["units"].update({"interactions-discrete":{"kind":"interactions","track":"discrete"},
        "kernels-discrete":{"kind":"kernels","track":"discrete"}})
    result=r.run(ctx)
    assert result["figures"]==6
    assert (ctx.path/"interactions-discrete-n8.png").exists()
    assert (ctx.path/"memory-rollouts.pdf").exists()


@pytest.mark.parametrize("values", [[.0062,.0064],[5.8e-8,7e-8],[3.7e-8,8.6e-8],[0.,0.],
    [0.,1e-8],[-1e-8,1e-8],[1e-10,1e-4]])
def test_numeric_axes_have_visible_ticks_for_small_wide_and_zero_ranges(values):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots()
    ax.plot([0,1],values);ax.set_yscale("symlog",linthresh=1e-12)
    original_y=np.array(ax.lines[0].get_ydata(),copy=True)
    r.quantitative_axis(ax)
    fig.canvas.draw()
    low,high=ax.get_ylim()
    visible=[label.get_text() for tick,label in zip(ax.get_yticks(),ax.get_yticklabels())
             if low<=tick<=high and label.get_visible() and label.get_text()]
    assert len(visible)>=2
    assert all(any(char.isdigit() for char in label) for label in visible)
    np.testing.assert_array_equal(ax.lines[0].get_ydata(),original_y)
    if min(values)>0 and max(values)/min(values)<10:
        assert ax.get_yscale()=="linear"
    plt.close(fig)


def test_attribution_palette_has_no_collisions():
    methods=["fixed_unit","quadratic_only_direct","quadratic_gain_direct","cubic_gain_direct",
             "two_gain_direct","two_gain_adam"]
    assert len({r.style(name)["color"] for name in methods})==len(methods)
