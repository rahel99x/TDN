"""Training/selection safety, equal schedule opportunities and trial recovery."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from tdn.analysis.portfolio.protocol import build_protocol
from tdn.analysis.portfolio.learning import (model_specs,shared_schedules,validation_selection,
    train,freeze,verify_frozen,confirm,aggregate,comparison_tables)
from tdn.analysis.portfolio.data import prepare,load_bank
from tdn.analysis.frontier.core import Context
from tdn.research.experiment import Budget


def tiny_protocol():
    p=build_protocol("smoke")
    p["models"]=["quad2_fixed","quad2_amplitude","quad2_linear","quad2_conditioned"]
    p["tracks"]=["discrete"];p["grids"]=[4];p["train_grid"]=4
    p["train_horizons"]=[.01];p["validation_horizons"]=[.01]
    p["evaluation_horizons"]=[.01];p["primary_schedules"]=[[.004,.006]];p["confirm_schedules"]=[[.004,.006],[.01],[.005,.005],[.0025]*4,[.00125]*8]
    p["training"].update(subset_sizes=[2],updates=2,tuning_updates=1,validation_every=1,learning_rates=[.001],ridge_candidates=[1e-6],batch_size=1)
    p["reference"] = dict(tolerance=1e-6,substeps=2,max_substeps=64)
    selected=[]
    for split,count in (("train",2),("validation",1),("confirmation",2)):
        candidates=[r for r in p["parents"] if r["split"]==split]
        for r in candidates[:count]:
            r.update(regime="smooth",field_terms=[{"mode":[1,0],"weight":1.,"phase":float(r["field_seed"])/1000}],mean=.3,variance=.0025,kappa=.01,reaction_rate=1.)
            selected.append(r)
    p["parents"]=selected
    p["units"]={"prepare":{"id":"prepare","kind":"prepare"},
      "train-A-000":{"id":"train-A-000","kind":"train","families":p["models"],"seeds":p["seeds"],"tracks":p["tracks"]},
      "freeze":{"id":"freeze","kind":"freeze"},
      "confirm-prepare-000":{"id":"confirm-prepare-000","kind":"confirm_prepare","parent_ids":[r["parent_id"] for r in selected if r["split"]=="confirmation"]},
      "confirm-000":{"id":"confirm-000","kind":"confirm","parent_ids":[r["parent_id"] for r in selected if r["split"]=="confirmation"]},
      "aggregate":{"id":"aggregate","kind":"aggregate"}}
    p["timing"]={"repeats":2,"warmup":1};p["bootstrap_replicates"]=10
    return p


def context(tmp_path,p,stage,prerequisites=None):
    return Context(p,stage,tmp_path/stage,prerequisites or {},"cpu",Budget(120))


def test_every_fitted_seed_and_data_subset_survives():
    p=build_protocol("full")
    all_specs=model_specs(p)
    unit_specs=[r for u in p["units"].values() if u["kind"]=="train" for r in model_specs(p,u)]
    assert len({r["model_id"] for r in unit_specs})==len(unit_specs)
    assert {r["model_id"] for r in unit_specs}=={r["model_id"] for r in all_specs}
    assert len([r for r in all_specs if r["family"]=="quad2_linear"])==12


def test_shared_schedules_retain_cheap_options_and_primary_identity():
    p=build_protocol("full");items=shared_schedules(p)
    for h in p["evaluation_horizons"]:
        for steps in [1,2,4,8]:
            assert any(r["schedule"]==[h/steps]*steps for r in items)
    assert sum(r["primary"] for r in items)==len(p["primary_schedules"])
    assert len({tuple(r["schedule"]) for r in items})==len(items)


def test_locked_validation_selection_cannot_drop_bad_fields():
    rows=[]
    for field in ["a","b"]:
        for sched,cost in [("cheap",1.),("expensive",2.)]:
            rows.append(dict(model_id="m",track="discrete",final_time=.1,field_cluster=field,schedule_id=sched,
                upper_rms=2e-4 if field=="b" and sched=="cheap" else 1e-6,upper_max=1e-6,
                cost_seconds=cost,reference_accepted=True))
    result=validation_selection(rows,[(1e-4,1e-4)])
    assert result[0]["schedule_id"]=="expensive"
    assert result[0]["independent_validation_fields"]==2


@pytest.fixture
def small_campaign(tmp_path,monkeypatch):
    # Unit fixtures hold a source identity while other parallel development agents
    # edit unrelated files. Production stages always use real live provenance.
    from tdn.analysis.portfolio import data
    snapshot=data.software_metadata()
    monkeypatch.setattr(data,"software_metadata",lambda:snapshot)
    torch.set_num_threads(1);p=tiny_protocol()
    prep=context(tmp_path,p,"prepare");prepare(prep)
    training=context(tmp_path,p,"train-A-000",{"prepare":prep.path});train(training)
    frozen=context(tmp_path,p,"freeze",{"train-A-000":training.path});freeze(frozen)
    references=context(tmp_path,p,"confirm-prepare-000",{"freeze":frozen.path});prepare(references)
    confirmation=context(tmp_path,p,"confirm-000",{"freeze":frozen.path,"confirm-prepare-000":references.path});confirm(confirmation)
    return p,{"prepare":prep,"train":training,"freeze":frozen,"references":references,"confirm":confirmation}


def test_numeric_bank_training_freeze_and_exact_confirmation(tmp_path,small_campaign):
    p,c=small_campaign
    bank=load_bank(c["train"],"prepare")
    assert len(bank)==3
    catalog,frozen=verify_frozen(c["confirm"])
    assert len(catalog["records"])==4
    assert all(r["checkpoint_validated"] for r in catalog["records"])
    assert all(r["fit_diagnostics"] is not None for r in catalog["records"] if r["family"] in ["quad2_amplitude","quad2_linear"])
    ctx=context(tmp_path,p,"aggregate",{"freeze":c["freeze"].path,"confirm-000":c["confirm"].path})
    result=aggregate(ctx)
    assert result["parents"]==2 and result["models"]==4
    assert result["endpoint_rows"]==2*4*len(shared_schedules(p))
    tables=json.loads((ctx.path/"comparisons.json").read_text())
    assert tables["locked_frontiers"] and tables["posthoc_frontiers"]
    assert all("final_time" in r and "regime" in r for r in tables["matched_configuration"])
    claim_rows=[r for r in ctx.rows if r["experiment_id"].startswith("claim/")]
    assert claim_rows
    assert not any(r["math_assessment"]["verdict"]=="BAD" for r in claim_rows)
    assert all(r["assessment"]["verdict"]=="NA" for r in claim_rows if r["metrics"]["verdict"]=="NA")


def test_recovery_reuses_completed_trials_and_groups_without_retiming(tmp_path,small_campaign,monkeypatch):
    p,c=small_campaign
    from tdn.analysis.portfolio import learning
    def fail(*a,**kw):raise AssertionError("Completed scientific work must not repeat")
    monkeypatch.setattr(learning,"_trial",fail);monkeypatch.setattr(learning,"measure_paired",fail)
    training=context(tmp_path,p,"train-A-000",{"prepare":c["prepare"].path});train(training)
    confirmation=context(tmp_path,p,"confirm-000",{"freeze":c["freeze"].path,"confirm-prepare-000":c["references"].path})
    result=confirm(confirmation)
    assert result["parents"]==2


def test_confirmation_aggregate_rejects_missing_projection(tmp_path,small_campaign):
    p,c=small_campaign;path=c["confirm"].path/"endpoint_rows.json"
    rows=json.loads(path.read_text());rows["rows"].pop();path.write_text(json.dumps(rows))
    ctx=context(tmp_path,p,"aggregate",{"freeze":c["freeze"].path,"confirm-000":c["confirm"].path})
    with pytest.raises(ValueError,match="projection"):
        aggregate(ctx)


def test_frozen_checkpoint_tampering_blocks_fresh_references(tmp_path,small_campaign):
    p,c=small_campaign;catalog,_=verify_frozen(c["confirm"])
    file=c["freeze"].path/catalog["records"][0]["checkpoint"]
    with file.open("ab") as handle:handle.write(b"tamper")
    with pytest.raises(ValueError,match="bytes changed"):
        verify_frozen(c["confirm"])


def test_fresh_confirmation_cannot_be_read_by_training(small_campaign):
    _,c=small_campaign
    c["train"].prerequisites.update({"freeze":c["freeze"].path,"confirm-prepare-000":c["references"].path})
    with pytest.raises(ValueError,match="forbidden during model selection"):
        load_bank(c["train"],"confirm-prepare-000")


def test_numeric_bank_tampering_is_not_recovered(small_campaign):
    _,c=small_campaign
    manifest=json.loads((c["prepare"].path/"data_manifest.json").read_text())
    file=c["prepare"].path/manifest["records"][0]["arrays"]
    with file.open("ab") as handle:handle.write(b"tamper")
    with pytest.raises(ValueError,match="Reference bytes changed"):
        load_bank(c["train"],"prepare")


def test_primary_claims_charge_joint_field_failures_and_reference_uncertainty():
    from tdn.analysis.portfolio.learning import primary_claims
    p=tiny_protocol();p["primary_target"]=1e-4
    rows=[];locked=[]
    for field in range(8):
        for seed in [11,21]:
            for family,cost,error in [("quad2_conditioned",1.,1e-6),("quad2_fixed",2.,2e-6)]:
                row=dict(family=family,track="discrete",train_count=2 if family=="quad2_conditioned" else 0,
                    seed=seed if family=="quad2_conditioned" else None,parent_id=str(field),field_cluster=str(field),
                    grid=4,schedule_id="one",final_time=.01,finite=True,error_rms=error,upper_rms=error,upper_max=error,
                    reference_accepted=True,reference_uncertainty_rms=1e-9,cost_seconds=cost)
                if family=="quad2_fixed" and seed==21:continue
                rows.append(row);locked.append({**row,"rms_target":1e-4,"max_target":1e-4,"status":"ELIGIBLE"})
    result=primary_claims(rows,{"locked_frontiers":locked},p)
    assert next(r for r in result["accuracy"] if r["control_family"]=="quad2_fixed")["verdict"]=="GOOD"
    assert next(r for r in result["cost"] if r["control_family"]=="quad2_fixed")["verdict"]=="GOOD"
    locked[0]["status"]="ACCURACY_INFEASIBLE"
    result=primary_claims(rows,{"locked_frontiers":locked},p)
    assert next(r for r in result["cost"] if r["control_family"]=="quad2_fixed")["verdict"]=="BAD"
    rows[0]["reference_uncertainty_rms"]=1e-5
    result=primary_claims(rows,{"locked_frontiers":locked},p)
    assert next(r for r in result["accuracy"] if r["control_family"]=="quad2_fixed")["verdict"]=="NA"
    locked[0]["status"]="REFERENCE_UNACCEPTED"
    result=primary_claims(rows,{"locked_frontiers":locked},p)
    assert next(r for r in result["cost"] if r["control_family"]=="quad2_fixed")["verdict"]=="NA"


def test_scaling_respects_frozen_step_count_and_primary_tolerance(tmp_path,monkeypatch):
    from tdn.analysis.portfolio import learning
    from tdn.analysis.frontier import data as teacher
    p=tiny_protocol();parent=copy.deepcopy(p['parents'][0]);parent.update(split='scaling',parent_id='scaling-only',field_cluster='scaling-only')
    p['parents'].append(parent);p['scaling'].update(grids=[4],batches=[1],maximum_cases=1,steps=2,horizon=.06,families=['quad2_conditioned'])
    p['primary_target']=2e-5;p['targets']=[2e-4,2e-5]
    p['units']['scaling']={'id':'scaling','kind':'scaling'}
    ctx=context(tmp_path,p,'scaling',{'freeze':tmp_path/'freeze'})
    record=dict(model_id='m',family='quad2_conditioned',seed=p['seeds'][0],track='discrete',train_count=2,checkpoint_validated=True)
    monkeypatch.setattr(learning,'verify_frozen',lambda ctx:({'records':[record]},{}))
    calls=[]
    class Model:
        family='quad2_conditioned'
        def __call__(self,u,h,eq,geom):calls.append(float(h));return u+5e-5
    monkeypatch.setattr(learning,'_load_model',lambda *a:Model())
    monkeypatch.setattr(teacher,'generate_reference',lambda parent,n,h,track,protocol,budget:dict(
        state=teacher.field_state(parent,n),accepted=True,uncertainty_rms=0.,uncertainty_max_bound=0.,teacher_seconds=0.))
    learning.scaling(ctx)
    result=json.loads((ctx.path/'scaling_rows.json').read_text())['rows'][0]
    assert result['schedule']==[.03,.03] and result['step_count']==2
    assert calls and all(x==.03 for x in calls)
    assert result['rms_target']==result['max_target']==2e-5
    assert result['status']=='ACCURACY_UNQUALIFIED'
    assert all(2e-5<r['upper_rms']<2e-4 for r in result['errors'])
