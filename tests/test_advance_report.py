"""Scientific reporting: provenance, denominator, missingness and readable ranges."""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tdn.analysis.advance import report


def context(tmp_path, **kwargs):
    args=dict(protocol={"profile":"smoke","primary_target":2e-5,"units":{
        "audit":{"kind":"audit"},"train-0":{"kind":"train"},"freeze":{"kind":"freeze"},
        "confirm-0":{"kind":"confirm"},"aggregate":{"kind":"aggregate"},"report":{"kind":"report"}}},
        stage="report",path=tmp_path/"report",prerequisites={},stage_failures={},device="cpu",
        budget=SimpleNamespace(check=lambda:None))
    args.update(kwargs)
    return SimpleNamespace(**args)


def put(path, filename, value):
    path.mkdir(parents=True,exist_ok=True)
    raw=("\n".join(json.dumps(row) for row in value)+"\n" if filename.endswith(".jsonl") else json.dumps(value))
    (path/filename).write_text(raw)


def test_canonical_sources_replace_repeated_rows_and_preserve_hashes(tmp_path):
    roots={name:tmp_path/name for name in ("train-0","freeze","confirm-0","aggregate")}
    model={"model_id":"m","family":"two_basis","role":"Ours","seed":11,"train_count":8}
    for unit in ("train-0","freeze"):put(roots[unit],"catalog.json",{"records":[model]})
    row={**model,"field_cluster":"f","upper_rms":1e-6,"reference_accepted":True,"intermediates":[{"time":.1,"error_rms":1e-6}],
         "raw_timing":{"cold_seconds":.02,"samples_seconds":[.1]*1000,"memory":{"scope":"separate probe","absolute_peak_allocated_bytes":123,"incremental_peak_allocated_bytes":12}}}
    for unit in ("confirm-0","aggregate"):put(roots[unit],"endpoint_rows.json",{"rows":[row]})
    data=report.collect(context(tmp_path,prerequisites=roots))
    assert len(data["catalog"])==len(data["endpoints"])==len(data["intermediates"])==1
    assert not data["partial_endpoints"]
    assert len(data["sources"])==2
    endpoint=data["endpoints"][0]
    assert "raw_timing" not in endpoint and "samples_seconds" not in endpoint
    assert endpoint["cold_seconds"]==.02 and endpoint["absolute_peak_allocated_bytes"]==123
    source=data["sources"][endpoint["_source"]["file"]]
    assert source["sha256"]==hashlib.sha256((roots["aggregate"]/"endpoint_rows.json").read_bytes()).hexdigest()
    assert endpoint["_source"]["pointer"]=="/rows/0"
    assert data["intermediates"][0]["_source"]["pointer"]=="/rows/0/intermediates/0"


def test_partial_failure_and_corrupt_source_remain_visible(tmp_path):
    partial=tmp_path/"confirm-0"
    put(partial,"endpoint_rows.json",{"rows":[{"family":"two_basis","error_rms":1.}]})
    (partial/"summary.json").write_text('{"elapsed_seconds":NaN}')
    data=report.collect(context(tmp_path,prerequisites={"confirm-0":partial},stage_failures={"aggregate":{"status":"FAILED","error":"missing partition","elapsed_seconds":3.}}))
    assert not data["endpoints"] and len(data["partial_endpoints"])==1
    assert data["omissions"] and "Nonfinite" in data["omissions"][0]["reason"]
    status=next(r for r in data["stage_status"] if r["unit"]=="aggregate")
    assert status["status"]=="FAILED" and not status["verified"] and status["monetary_cost"] is None


def test_source_symlinks_are_not_followed(tmp_path):
    source=tmp_path/"source.json";source.write_text('{}')
    link=tmp_path/"link.json";link.symlink_to(source)
    with pytest.raises(ValueError,match="symlink"):report._read(link)
    folder=tmp_path/"alias";folder.symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises(ValueError,match="symlink"):report._read(folder/"source.json")


def test_style_roles_and_actual_new_architectures():
    from tdn.analysis.advance.models import FAMILIES
    assert len({report.style(f)["color"] for f in FAMILIES})==len(FAMILIES)
    assert report.style("two_basis")["marker"]=="^"
    assert report.style("commutator_raw")["marker"]=="^"
    assert report.style("fno_scaled")["marker"]=="o"
    assert report.style("quad4_full")["marker"]=="s"
    assert report.role("unknown") == "Unclassified"
    assert all(report.style(f)["linewidth"]<1 for f in ("two_basis","fno_scaled","quad4_full"))
    assert report.style("two_basis")["linestyle"]!=report.style("fno_scaled")["linestyle"]


def test_fixed_basis_gains_are_traceable_without_fabricated_response():
    row=dict(family="two_basis",track="discrete",train_count=64,seed=11,
             parameter_report={"effective_gains":[.75,1.25]},
             _source={"file":2,"pointer":"/records/3"})
    before=copy.deepcopy(row)
    gains=report.basis_gain_rows([row])
    assert row==before
    assert [r["gain"] for r in gains]==[.75,1.25]
    assert [r["basis"] for r in gains]==["Full GL2 interaction","Cubic interaction"]
    assert gains[1]["_source"]=={"file":2,"pointer":"/records/3/parameter_report/effective_gains/1"}
    assert all("horizon" not in r and "response" not in r for r in gains)


def test_field_interval_uses_independent_fields_and_refuses_missing_crossed_cells():
    rows=[{"field_cluster":f"f{field}","seed":seed,"error_rms":float(field+1)} for field in range(6) for seed in (11,21) for _ in range(3)]
    summary=report.field_summary(rows,"error_rms",repeats=100)
    assert summary["independent_fields"]==6 and summary["training_seeds"]==2 and summary["mean"]==3.5
    assert summary["query_rows"]==36
    partial=[r for r in rows if not(r["field_cluster"]=="f0" and r["seed"]==21)]
    assert report.field_summary(partial,"error_rms",repeats=100)["status"]=="INCOMPLETE_CROSSED_DESIGN"
    assert report.field_summary(rows[:3],"error_rms",repeats=100)["lower"] is None


def test_frozen_control_shares_plot_with_each_data_budget_without_changing_source():
    rows=[dict(family="quad4_full",train_count=0,_source={"file":0,"pointer":"/rows/0"}),
          dict(family="two_basis",train_count=16),dict(family="two_basis",train_count=64)]
    original=copy.deepcopy(rows);display=report.matched_budget_views(rows)
    assert rows==original
    for count in (16,64):
        assert {r["family"] for r in display if r["comparison_train_count"]==count}=={"two_basis","quad4_full"}
    frozen=[r for r in display if r["family"]=="quad4_full"]
    assert frozen[0]["_source"]==frozen[1]["_source"]==original[0]["_source"]


def test_amortization_joins_real_frozen_workloads_and_does_not_credit_wrong_seed():
    common=dict(parent_id="p",field_cluster="f",grid=32,track="discrete",final_time=.1,rms_target=1e-5,max_target=1e-5,train_count=8,status="ELIGIBLE")
    candidate=dict(common,model_id="ours",family="two_basis",seed=11,cost_seconds=.01)
    control=dict(common,model_id="fno11",family="fno_scaled",seed=11,cost_seconds=.02)
    wrong=dict(common,model_id="fno21",family="fno_scaled",seed=21,cost_seconds=100.)
    data=dict(catalog=[dict(model_id="ours",family="two_basis",role="Ours",total_training_seconds=20.),dict(model_id="fno11",family="fno_scaled",role="Theirs")],locked_frontiers=[candidate,control,wrong])
    result=report.amortization(data)
    assert len(result)==1
    assert result[0]["inference_saving_seconds"]==pytest.approx(.01)
    assert result[0]["training_only_break_even_queries"]==2000
    assert result[0]["offline_cost_complete"] is False and result[0]["monetary_cost"] is None
    control["status"]="ACCURACY_INFEASIBLE"
    result=report.amortization(data)
    assert result[0]["known_offline_break_even_queries"] is None
    assert result[0]["status"]=="INELIGIBLE_COMPARISON"


def test_dense_range_is_continuous_without_filling_missing_windows():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    rows=[dict(family="two_basis",track="discrete",phase="final",train_count=8,seed=1,model_id="m",update=i,train_loss=(-1.)**i) for i in range(40) if not 15<=i<25]
    before=copy.deepcopy(rows)
    ranges=report._learning_ranges(rows,"update","train_loss",max_windows=8)
    fig,ax=plt.subplots()
    count=report.range_plot(ax,ranges,"update","train_loss")
    assert rows==before and count==30
    bands=[c for c in ax.collections if isinstance(c,PolyCollection)]
    assert len(bands)==2 and all(c.get_alpha()==.1 for c in bands)
    edges=[line for line in ax.lines if len(line.get_xdata())>2]
    assert len(edges)==4 and all(line.get_linewidth()<1 for line in edges)
    assert all(all(b>a for a,b in zip(line.get_xdata(),line.get_xdata()[1:])) for line in edges)
    assert min(min(line.get_ydata()) for line in edges)==-1
    assert max(max(line.get_ydata()) for line in edges)==1
    plt.close(fig)


def test_singleton_updates_connect_continuously_but_missing_updates_break_bands():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    rows=[dict(family="fno_zero",track="discrete",phase="tiny_overfit",train_count=2,
               seed=1,model_id="m",update=i,train_loss=float(i)) for i in (1,2,4,5)]
    for windows in (1,24):
        ranges=report._learning_ranges(rows,"update","train_loss",max_windows=windows)
        fig,ax=plt.subplots();report.range_plot(ax,ranges,"update","train_loss")
        bands=[c for c in ax.collections if isinstance(c,PolyCollection)]
        assert len(bands)==2
        paths=[band.get_paths()[0].vertices[:,0] for band in bands]
        assert [(float(xs.min()),float(xs.max())) for xs in paths]==[(1.,2.),(4.,5.)]
        assert sum(r["observation_count"] for r in ranges)==4
        plt.close(fig)
    contiguous=[dict(row,update=i,train_loss=float(i)) for i,row in enumerate(rows,1)]
    ranges=report._learning_ranges(contiguous,"update","train_loss")
    fig,ax=plt.subplots();report.range_plot(ax,ranges,"update","train_loss")
    lines=[line for line in ax.lines if len(line.get_xdata())>2]
    assert len(lines)==2 and all(list(line.get_xdata())==[1.,2.,3.,4.] for line in lines)
    plt.close(fig)


def test_zero_error_visible_missing_metric_has_zero_observation_count():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots()
    rows=[dict(family="quad4_full",field_cluster=f"f{i}",seed=None,error_rms=0.) for i in range(6)]
    assert report._metric_draw(ax,rows,"error_rms")==6
    assert ax.get_yscale()=="linear"
    assert report._metric_draw(ax,[dict(family="two_basis")],"error_rms")==0
    plt.close(fig)


def test_empty_hd_atlas_is_complete_explicit_na_with_one_chart_source(tmp_path):
    data=report.collect(context(tmp_path));before=copy.deepcopy(data)
    manifest=report.build_figures(data,tmp_path/"figures")
    assert data==before
    assert len(manifest["panels"])>=35
    assert all(p["dpi"]==300 and p["reading_guide"] and p["scope"] for p in manifest["panels"])
    missing=next(p for p in manifest["panels"] if p["title"]=="Physical parameter sensitivity")
    assert missing["status"]=="NA" and missing["observations"]==0
    with gzip.open(tmp_path/"figures/chart-data.json.gz","rt") as stream:payload=json.load(stream)
    assert payload["sources"]==[] and payload["learning_ranges"]==[]
    assert not list((tmp_path/"figures").glob("*.csv"))
    assert (tmp_path/"figures/advance-atlas.pdf").stat().st_size>1000
    assert "not confidence intervals" in (tmp_path/"figures/index.html").read_text()
    from PIL import Image
    with Image.open(tmp_path/"figures"/manifest["panels"][0]["path"]) as image:
        assert image.width>=3600 and image.height>=1650
    with pytest.raises(ValueError,match="300 DPI"):report.build_figures(data,tmp_path/"too-small",dpi=72)
