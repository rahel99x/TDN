"""Staged targets, query weighting, exact continuation and honest coverage."""
from copy import deepcopy
import json
import pytest
import torch

from tdn.analysis.response import numerics as n
from tdn.analysis.response.contracts import Physics
from tdn.numerics.types import Geometry
from tdn.analysis.tangent_response import training as tr
from tdn.analysis.tangent_response.tangent import AnchorPath, forced_response, response_action


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def samples(device="cpu", target="fd_same_grid"):
    geom = Geometry((3, 3), (1., 1.)); eq = Physics(.02, 3.)
    rows = []
    x = torch.arange(3, device=device)*2*torch.pi/3
    for split, parent, mean in (("train", "a", .4), ("validation", "v", .48)):
        u = (mean+.025*torch.cos(x)[:, None]+.018*torch.sin(x)[None, :])[None, None].float()
        h = .03; path = AnchorPath(u.double(), eq, geom, target)
        anchor = path(h); correction = forced_response(path, eq, geom, target, h, steps=8)
        probe = .2+n.center(u.double())
        action = response_action(path, eq, geom, target, probe, .005, h, steps=8)
        rows.append(dict(u0=u, teacher64=n.integrate(u.double(), h, eq, geom, target, substeps=16), smooth_anchor64=anchor,
            deployed_anchor32=n.anchor(u, h, eq, geom, target), oracle_endpoint64=anchor+correction,
            oracle_correction64=correction, oracle_metadata={"accepted":True, "uncertainty_rms":0., "test_fixture":True},
            record=dict(parent_id=parent, split=split, query_id=parent+"/.03", shape_sha256=parent,
                horizon=h, grid=[3, 3], lengths=[1., 1.], kappa=eq.kappa, reaction_rate=eq.reaction_rate,
                target_kind=target, reference={"accepted":True, "test_fixture":True}),
            action_labels=[dict(source=probe, s=.005, h=h, target_action=action, accepted=True, probe_id="probe")]))
    return rows


def config(family="learned_response", updates=4):
    return dict(id="fixture", family=family, seed=51, updates=updates, validation_every=2, total_seconds=120.,
        source_tree_sha256="unit-test-fixture", endpoint_target="oracle_tangent", action_fraction=.5,
        lr_schedule=dict(kind="cosine", minimum_ratio=.05, warmup_updates=1), head_warmup_updates=1,
        public_curve_every=2, live_every=2, conditional_readout=False,
        model_config=dict(width=4, depth=1, modes=1, rank=2, source_nodes=2, mean_nodes=2))


@pytest.mark.parametrize("family", ("learned_response", "residual_fno"))
def test_exact_staged_resume_model_optimizer_scheduler_rng_and_query_work(tmp_path, family):
    data = samples(); cfg = config(family)
    full = tr.fit(deepcopy(data), cfg, output_path=tmp_path/"full")
    part = tr.fit(deepcopy(data), cfg, output_path=tmp_path/"part", max_chunk_updates=2)
    assert part["status"] == "CONTINUATION_REQUIRED"
    resumed = tr.fit(deepcopy(data), cfg, output_path=tmp_path/"resumed", resume_from=tmp_path/"part")
    a = torch.load(tmp_path/"full/journal.pt", weights_only=True)
    b = torch.load(tmp_path/"resumed/journal.pt", weights_only=True)
    assert all(torch.equal(a["model"][key], b["model"][key]) for key in a["model"])
    assert torch.equal(a["rng"]["torch"], b["rng"]["torch"])
    assert a["scheduler_state"] == b["scheduler_state"]
    assert a["work"] == b["work"]
    for key,state in a["optimizer"]["state"].items():
        for name,value in state.items(): assert torch.equal(value, b["optimizer"]["state"][key][name])
    assert full["selected_state_sha256"] == resumed["selected_state_sha256"]
    assert [row.get("train_loss") for row in a["history"]] == [row.get("train_loss") for row in b["history"]]
    assert full["journal_sha256"] == tr.file_digest(tmp_path/"full/journal.pt")
    assert (tmp_path/"resumed/initial.pt").exists() and (tmp_path/"resumed/final.pt").exists()
    if family == "learned_response": assert (tmp_path/"resumed/action-stage.pt").exists()


def test_oracle_target_is_distinct_from_true_teacher_and_validation_is_separate(tmp_path):
    data = samples(); cfg = config(updates=2)
    for sample in data: sample["teacher64"] += .01
    result = tr.fit(data, cfg, output_path=tmp_path)
    assert result["endpoint_target"] == "oracle_tangent"
    row = result["initial_train"]["rows"][0]
    assert row["teacher_rms"] > row["rms"]+.009
    assert result["criterion_role"].endswith("diagnostic only")
    selected = torch.load(tmp_path/"selected.pt", weights_only=True)
    assert selected["checkpoint_role"] == "validation-selected deployment"


def test_training_normalizers_do_not_access_validation_labels():
    data = samples(); cfg = config()
    first = tr.normalization(data, cfg)
    data[-1]["teacher64"] += 1e6; data[-1]["oracle_endpoint64"] += 1e6
    data[-1]["action_labels"][0]["target_action"] += 1e6
    assert tr.normalization(data, cfg) == first
    data[-1]["record"]["shape_sha256"] = data[0]["record"]["shape_sha256"]
    with pytest.raises(ValueError, match="leakage"): tr.normalization(data, cfg)


def test_parent_weight_plan_and_missing_action_references_are_explicit(tmp_path):
    data = samples(); twin = deepcopy(data[0]); twin["record"]["query_id"] += "/phase"
    plan = tr._plan([data[0], twin, {**data[0], "record":{**data[0]["record"], "parent_id":"b", "query_id":"b"}}])
    assert [weight for _,weight in plan] == [.25, .25, .5]
    data[0]["action_labels"][0]["accepted"] = False
    result = tr.fit(data, config(), output_path=tmp_path)
    assert result["status"] == "UNRESOLVED_ACTION_LABELS" and result["updates"] == 0
    assert not (tmp_path/"selected.pt").exists()


def test_response_head_then_factor_gradients_and_component_accounting(tmp_path):
    result = tr.fit(samples(), config(updates=4), output_path=tmp_path)
    curves = [json.loads(line) for line in (tmp_path/"learning_curves.jsonl").read_text().splitlines()]
    assert result["status"] == "COMPLETED_UPDATES"
    assert result["work"]["component_gradient_calls"] == 8
    assert result["optimization_history_summary"]["metrics"]["head_gradient_norm"]["maximum"] > 0
    assert result["optimization_history_summary"]["metrics"]["trunk_gradient_norm"]["maximum"] > 0
    assert any(row.get("training_phase") == "action" for row in curves)
    assert any(row.get("training_phase") == "endpoint" for row in curves)


def test_conditional_readout_is_a_training_only_diagnostic(tmp_path):
    cfg = {**config(updates=2), "conditional_readout":True}
    result = tr.fit(samples(), cfg, output_path=tmp_path)
    control = result["conditional_control"]
    assert control["status"] == "COMPLETED"
    assert control["affine"] and not control["global_nonlinear_optimum"]
    assert control["training_parent_ids"] == ["a"]
    assert control["solve"]["physical_error"]["rms"] is not None


def test_changed_dataset_and_diagnostic_teacher_access_rejected(tmp_path):
    data = samples(); cfg = config()
    tr.fit(data, cfg, output_path=tmp_path/"first", max_chunk_updates=1)
    changed = deepcopy(data); changed[0]["oracle_endpoint64"] += 1e-6
    with pytest.raises(ValueError, match="identity changed"):
        tr.fit(changed, cfg, output_path=tmp_path/"bad", resume_from=tmp_path/"first")
    data[-1]["record"]["split"] = "diagnostic"
    with pytest.raises(ValueError, match="diagnostic/test"):
        tr.fit(data, cfg, output_path=tmp_path/"forbidden")



def test_interrupted_partial_query_is_rolled_back_and_actual_work_is_retained(tmp_path, monkeypatch):
    class InterruptOnce:
        active = False
        remaining = 3
        fired = False
        def check(self):
            if self.active and not self.fired:
                self.remaining -= 1
                if self.remaining == 0:
                    self.fired = True
                    raise InterruptedError("fixture interruption inside explicit anchor path")
    data = samples(); cfg = config()
    full = tr.fit(deepcopy(data),cfg,output_path=tmp_path/"full")
    budget = InterruptOnce(); original = tr._components
    def interrupted(*args,**kwargs):
        budget.active = True
        try:return original(*args,**kwargs)
        finally:budget.active = False
    monkeypatch.setattr(tr,"_components",interrupted)
    first = tr.fit(deepcopy(data),cfg,output_path=tmp_path/"first",budget=budget)
    assert first["updates"] == 0 and first["failure"]["type"] == "InterruptedError"
    assert first["work"]["query_attempts"] == 1
    assert first["work"]["anchor_evaluations"] > 0
    monkeypatch.setattr(tr,"_components",original)
    resumed = tr.fit(deepcopy(data),cfg,output_path=tmp_path/"next",resume_from=tmp_path/"first")
    assert resumed["final_state_sha256"] == full["final_state_sha256"]
    assert resumed["selected_state_sha256"] == full["selected_state_sha256"]
    assert resumed["work"]["committed_optimizer_updates"] == cfg["updates"]
    assert resumed["work"]["query_attempts"] == full["work"]["query_attempts"]+1
    assert resumed["attempts"][0]["work"]["query_attempts"] == 1


def test_scheduled_validation_interruption_keeps_pending_and_completed_observations(tmp_path, monkeypatch):
    cfg = config(); data = samples(); original = tr.evaluate; validation_calls = 0
    def interrupted(model, rows, *args, **kwargs):
        nonlocal validation_calls
        if rows[0]["record"]["split"] == "validation":
            validation_calls += 1
            if validation_calls == 2: raise InterruptedError("fixture scheduled validation interruption")
        return original(model, rows, *args, **kwargs)
    monkeypatch.setattr(tr,"evaluate",interrupted)
    first = tr.fit(deepcopy(data),cfg,output_path=tmp_path/"first")
    assert first["updates"] == 2
    old = [json.loads(line) for line in (tmp_path/"first/learning_curves.jsonl").read_text().splitlines()]
    assert old[-1]["observation_kind"] == "pending_validation"
    monkeypatch.setattr(tr,"evaluate",original)
    tr.fit(deepcopy(data),cfg,output_path=tmp_path/"next",resume_from=tmp_path/"first")
    new = [json.loads(line) for line in (tmp_path/"next/learning_curves.jsonl").read_text().splitlines()]
    completed = [row for row in new if row["update"] == 2 and "validation_loss" in row]
    assert len(completed) == 1 and completed[0].get("observation_kind", "default") != "pending_validation"
    assert completed[0]["train_loss"] == old[-1]["train_loss"]


@pytest.mark.parametrize("family", ("learned_response", "residual_fno"))
def test_selected_checkpoint_reload_in_an_independent_cpu_process(tmp_path, family):
    import os
    from pathlib import Path
    import subprocess
    import sys
    data = samples(); result = tr.fit(data,config(family,updates=2),output_path=tmp_path/"fit")
    model,metadata = tr.load_model(tmp_path/"fit")
    sample = data[-1]; eq,geom = tr.equation_geometry(sample)
    with torch.no_grad(): expected = model(sample["u0"],sample["record"]["horizon"],eq,geom,sample["record"]["target_kind"])
    torch.save({"u0":sample["u0"],"record":sample["record"]},tmp_path/"input.pt")
    script = '''import sys,torch
from tdn.analysis.tangent_response.training import load_model,equation_geometry
s=torch.load(sys.argv[2],weights_only=True);m,meta=load_model(sys.argv[1]);eq,g=equation_geometry(s)
with torch.no_grad():y=m(s["u0"],s["record"]["horizon"],eq,g,s["record"]["target_kind"])
torch.save({"output":y,"role":meta["checkpoint_role"]},sys.argv[3])
'''
    subprocess.run([sys.executable,"-c",script,str(tmp_path/"fit"),str(tmp_path/"input.pt"),str(tmp_path/"output.pt")],
        check=True,timeout=60,cwd=Path(__file__).resolve().parents[1],env={**os.environ,"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"})
    actual = torch.load(tmp_path/"output.pt",weights_only=True)
    torch.testing.assert_close(actual["output"],expected,atol=0,rtol=0)
    assert metadata["update"] == result["selected_update"] and actual["role"] == "validation-selected deployment"


def test_galerkin_validation_maximum_samples_physical_interpolant():
    data = samples(target="galerkin_same_grid")
    model = tr.create_model({**config()["model_config"],"normalization":tr.normalization(data,config())})
    result = tr.evaluate(model,data,config())
    assert all(row["maximum_kind"] == "sampled_physical_interpolant" and row["maximum_grid"] == [12,12] for row in result["rows"])


def test_complete_gain_uses_all_training_fit_probes_and_is_invariant_to_labels_and_validation():
    data = samples(); cfg = config(); norm = tr.normalization(data,cfg)
    base = deepcopy(data[0]["action_labels"][0]); base["role"] = "fit"
    tiny = deepcopy(base); tiny.update(probe_id="actual-tiny-residual",source=base["source"]*1e-10)
    unit = deepcopy(base); unit.update(probe_id="independent-order-one-direction",source=base["source"])
    rejected = deepcopy(base); rejected.update(probe_id="unaccepted-direction",accepted=False,source=base["source"]*1e20)
    heldout = deepcopy(base); heldout.update(probe_id="heldout-direction",role="heldout_direction",source=base["source"]*1e20)
    data[0]["action_labels"] = [tiny,unit,rejected,heldout]
    model = tr.create_model({**cfg["model_config"],"normalization":norm})
    deadline = tr.Deadline(tr.time.perf_counter(),0.,120.,120.)
    fitted = tr._gain(model,data,norm,cfg,deadline)
    receipt = fitted["response_gain_fit"]
    assert receipt["queried_probe_ids"] == ["actual-tiny-residual","independent-order-one-direction"]
    assert receipt["queries"] == 1 and receipt["probes"] == 2
    assert receipt["work"] == dict(gain_encoder_calls=1,gain_action_basis_calls=2,gain_physical_norm_calls=2)
    assert receipt["rows"][1]["physical_basis_rms"] > receipt["rows"][0]["physical_basis_rms"]*1e9
    assert fitted["response_gain"] == max(receipt["minimum_gain"],receipt["rows"][1]["physical_basis_rms"])
    # Gain fitting never reads target_action, validation inputs, or validation
    # source values. Poisoning them therefore cannot alter even the receipt.
    poisoned = deepcopy(data)
    for row in poisoned:
        for label in row["action_labels"]: label["target_action"] = torch.full_like(label["target_action"],float("nan"))
    poisoned[-1]["u0"] = torch.full_like(poisoned[-1]["u0"],float("nan"))
    for label in poisoned[-1]["action_labels"]: label["source"] = torch.full_like(label["source"],float("nan"))
    assert tr._gain(model,poisoned,norm,cfg,deadline) == fitted


def test_complete_gain_matches_parent_query_probe_weights_and_encodes_once_per_query():
    data = samples(); cfg = config(); norm = tr.normalization(data,cfg)
    first = deepcopy(data[0]); first["action_labels"] *= 2
    first["action_labels"][0]["probe_id"] = "a-first"
    first["action_labels"][1] = deepcopy(first["action_labels"][1]); first["action_labels"][1]["probe_id"] = "a-second"
    second = deepcopy(data[0]); second["record"]["query_id"] = "a/second-query"
    other = deepcopy(data[0]); other["record"].update(parent_id="b",query_id="b/first-query")
    model = tr.create_model({**cfg["model_config"],"normalization":norm})
    receipt = tr._gain(model,[first,second,other],norm,cfg,tr.Deadline(tr.time.perf_counter(),0.,120.,120.))["response_gain_fit"]
    assert [row["weight"] for row in receipt["rows"]] == [.125,.125,.25,.5]
    assert receipt["queries"] == 3 and receipt["probes"] == 4
    assert receipt["training_parent_ids"] == ["a","b"]
    assert receipt["work"]["gain_encoder_calls"] == 3
    assert receipt["work"]["gain_action_basis_calls"] == 4


@pytest.mark.parametrize("family", ("learned_response", "residual_fno"))
def test_actual_strict_inference_script_accepts_selected_fp32_checkpoint_and_checks_source(tmp_path,family):
    import os
    from pathlib import Path
    import subprocess
    import sys
    from tdn.analysis.tangent_response.evaluation import final_rounding_metrics
    cfg=config(family,updates=2);cfg["source_tree_sha256"]=tr.software_metadata()["source_tree_sha256"]
    data=samples();result=tr.fit(data,cfg,output_path=tmp_path/"fit")
    model,metadata=tr.load_model(tmp_path/"fit")
    sample=data[-1];eq,geom=tr.equation_geometry(sample)
    with torch.no_grad():parts=model.prediction_parts(model.encode(sample["u0"],eq,geom,sample["record"]["target_kind"]),sample["record"]["horizon"])
    assert parts["output"].dtype==torch.float32
    rounding=final_rounding_metrics(parts["output"],parts,geom,sample["record"]["target_kind"])
    assert rounding["final_rounding_available"] and rounding["final_rounding_rms"]>=0
    assert result["selected_validation"]["rows"][0]["final_rounding_rms"]==rounding["final_rounding_rms"]
    record={key:sample["record"][key] for key in ("query_id","parent_id","grid","lengths","horizon","target_kind","kappa","reaction_rate")}
    payload=tmp_path/"input.pt";torch.save({"u0":sample["u0"],"record":record},payload)
    command=[sys.executable,"scripts/tangent_response_infer.py","--checkpoint",str(tmp_path/"fit/selected.pt"),
        "--input",str(payload),"--device","cpu","--source-sha256",cfg["source_tree_sha256"]]
    kwargs=dict(capture_output=True,text=True,timeout=60,cwd=Path(__file__).resolve().parents[1],
        env={**os.environ,"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"})
    completed=subprocess.run(command,check=True,**kwargs)
    receipt=json.loads(completed.stdout)
    assert receipt["status"]=="COMPLETED" and receipt["device"]=="cpu"
    assert receipt["source_tree_sha256"]==cfg["source_tree_sha256"]==metadata["identity"]["source"]
    forged=command[:-1]+["0"*64]
    refused=subprocess.run(forged,check=False,**kwargs)
    assert refused.returncode==2 and "source changed" in refused.stderr
    checkpoint=torch.load(tmp_path/"fit/selected.pt",weights_only=True)
    checkpoint["identity"]["source"]="0"*64
    mismatched=tmp_path/"mismatched-selected.pt";torch.save(checkpoint,mismatched)
    checkpoint_command=list(command);checkpoint_command[checkpoint_command.index("--checkpoint")+1]=str(mismatched)
    refused=subprocess.run(checkpoint_command,check=False,**kwargs)
    assert refused.returncode==2 and "Checkpoint source differs" in refused.stderr
