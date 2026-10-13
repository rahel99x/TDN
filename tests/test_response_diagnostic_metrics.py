"""Research denominators, physical comparators and sealed-source integrity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tdn.analysis.response import diagnostic_metrics as dm
from tdn.research.protocol import digest
from tdn.runtime.metadata import write_json


def _context(tmp_path, trials=None):
    campaign = tmp_path/"response-diagnostic-example"
    path = campaign/"diagnostic-metrics"
    path.mkdir(parents=True)
    protocol = dict(profile="diagnostic", source_tree_sha256="a"*64,
        units={"train-one":{"kind":"train"}, "diagnostic-metrics":{"kind":"diagnostic_metrics", "mechanism":"M3"}},
        trials=trials or {"one":dict(id="one", family="temporal", target_kind="fd_same_grid", updates=10)},
        mechanisms={"M3":{}}, combinations={})
    return SimpleNamespace(protocol=protocol, stage="diagnostic-metrics", path=path,
        unit=protocol["units"]["diagnostic-metrics"], prerequisites={}, device="cpu")


def _seal(ctx, stage, files):
    base = ctx.path.parent/stage
    base.mkdir()
    artifacts = {}
    for name, payload in files.items():
        if name.endswith(".jsonl"):
            raw = "".join(json.dumps(row, allow_nan=False)+"\n" for row in payload).encode()
            (base/name).write_bytes(raw)
        else:
            write_json(base/name, payload)
            raw = (base/name).read_bytes()
        artifacts[name] = hashlib.sha256(raw).hexdigest()
    manifest = dict(schema="tdn.response-science/v1", stage=stage,
        kind=ctx.protocol["units"][stage]["kind"], source_tree_sha256="a"*64,
        protocol_sha256=digest(ctx.protocol), artifacts=artifacts)
    write_json(base/"science_manifest.json", manifest)
    (base/"COMPLETED").write_text(digest(manifest)+"\n")
    ctx.prerequisites[stage] = base
    return base


def _trial(ident="one"):
    endpoint = dict(query_id="q", parent_id="p", rms=.002, maximum=.003, mean_error=.001,
        centered_rms=.0017, mse=4e-6, target_defect_rms=.004, floor_limited=False,
        uncertainty_rms=1e-10, conservative_rms=.0020000001, reference_accepted=True)
    initial = dict(mse=.04, rms=.2, maximum=.3, rows=[dict(endpoint, mse=.04)])
    fit = dict(mse=4e-6, rms=.002, maximum=.003, rows=[endpoint])
    return dict(trial_id=ident, status="TIME_CENSORED", updates=7, requested_updates=10,
        initial_train=initial, selected_train=fit, final_train=fit, attainable_train=fit,
        criterion_train=fit, attempts=[dict(status="TIME_CENSORED", elapsed_seconds=2)],
        requested_training_queries=2, accepted_training_queries=1,
        requested_validation_queries=2, accepted_validation_queries=2,
        reference_coverage_rows=[dict(query_id="missing", reference_accepted=False, reason="UNRESOLVED")],
        total_parameters=808, examples_seen=7)


def test_registered_missing_failed_censored_and_reference_denominators(tmp_path):
    declared = {key:dict(id=key, family="temporal", updates=10) for key in ("one", "missing", "failed", "screen")}
    ctx = _context(tmp_path, declared)
    failed = dict(_trial("failed"), status="FAILED_NONFINITE", updates=1)
    screen = dict(trial_id="screen", status="UNRESOLVED_SCREEN", updates=0, requested_updates=10)
    _seal(ctx, "train-one", {"training_results.json":dict(trials=[_trial(),failed,screen])})
    data, _ = dm.collect(ctx)
    assert len(data["trials"]) == 4
    assert data["summary"]["registered_trials"] == 4
    assert data["summary"]["missing_trials"] == 1
    assert data["summary"]["time_censored_trials"] == 1
    assert data["summary"]["failed_trials"] == 1
    assert data["summary"]["no_headroom_trials"] == 1
    assert data["summary"]["achieved_updates"] == 8
    assert data["summary"]["complete_reference_coverage_trials"] == 0
    assert data["trials"][0]["reference_coverage_rows"][0]["query_id"] == "missing"
    assert data["no_development_authorization"] is True


def test_initialization_gain_is_distinct_from_common_anchor_gain(tmp_path):
    row, endpoints = dm.trial_metrics(_trial(), source_path="sources/a/training_results.json", source_sha256="b"*64)
    assert row["initialization_mse_reduction"] == pytest.approx(10000)
    assert endpoints[0]["anchor_mse_reduction"] == pytest.approx(4)
    assert endpoints[0]["anchor_rms_reduction"] == pytest.approx(2)
    assert endpoints[0]["source_record"] == "/trials/0/selected_train/rows/0"
    assert endpoints[0]["mean_error"] == .001
    assert endpoints[0]["centered_rms"] == .0017


def test_sealed_source_tampering_is_rejected(tmp_path):
    ctx = _context(tmp_path)
    base = _seal(ctx, "train-one", {"training_results.json":dict(trials=[_trial()])})
    (base/"training_results.json").write_text('{"trials":[]}')
    with pytest.raises(ValueError, match="science seal"):
        dm.collect(ctx)


def test_unregistered_trial_and_protocol_substitution_rejected(tmp_path):
    ctx = _context(tmp_path)
    _seal(ctx, "train-one", {"training_results.json":dict(trials=[_trial("not-registered")])})
    with pytest.raises(ValueError, match="unregistered"):
        dm.collect(ctx)
    ctx.protocol["profile"] = "changed"
    with pytest.raises(ValueError, match="exact completed science seal"):
        dm.collect(ctx)


def test_binary_checkpoints_are_not_loaded(tmp_path, monkeypatch):
    import torch
    ctx = _context(tmp_path)
    base = _seal(ctx, "train-one", {"training_results.json":dict(trials=[_trial()])})
    (base/"selected.pt").write_bytes(b"must not be deserialized")
    monkeypatch.setattr(torch, "load", lambda *a,**k: (_ for _ in ()).throw(AssertionError("Loaded a checkpoint")))
    data, tables = dm.collect(ctx)
    assert data["summary"]["observed_trials"] == 1
    assert all(table["name"] != "selected.pt" for table in tables)


def test_json_duplicates_nonfinite_and_symlinks_are_rejected(tmp_path):
    file = tmp_path/"metadata.json"
    for raw in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":1e400}'):
        file.write_text(raw)
        with pytest.raises(ValueError):
            dm.read_metadata(file)
    file.write_text('{"x":1}')
    link = tmp_path/"link.json"
    link.symlink_to(file)
    with pytest.raises(ValueError, match="symlink"):
        dm.read_metadata(link)
    child = tmp_path/"child"
    child.mkdir()
    with pytest.raises(ValueError,match="parent directories"):
        dm.read_metadata(child/".."/"metadata.json",root=child)


def test_matching_chunk_histories_deduplicate_with_all_citations(tmp_path):
    ctx = _context(tmp_path)
    ctx.protocol["units"]["train-two"] = {"kind":"train"}
    trial = _trial()
    curve = dict(trial_id="one", update=0, train_loss=.2)
    _seal(ctx, "train-one", {"training_results.json":dict(trials=[trial]), "learning_curves.jsonl":[curve]})
    _seal(ctx, "train-two", {"training_results.json":dict(trials=[trial]), "learning_curves.jsonl":[curve]})
    data, _ = dm.collect(ctx)
    assert len(data["learning_curves"]) == 1
    assert len(data["learning_curves"][0]["citations"]) == 2


def _joint_trial():
    regimes = ["a", "b"]
    rows = [dict(query_id="q-"+r,parent_id="p-"+r,tiny_regime=r,mse=1e-8,rms=1e-4,maximum=1e-4,
        conservative_rms=1e-4,conservative_maximum=1e-4,reference_accepted=True,floor_limited=False) for r in regimes]
    initial = [dict(row,mse=1e-4) for row in rows]
    coverage = [dict(row,split="train") for row in rows]
    return dict(trial_id="joint",phase="tiny_joint",status="COMPLETED_UPDATES",updates=10,requested_updates=10,
        initial_train={"rows":initial},selected_train={"rows":rows},attainable_train={"rows":rows},
        reference_coverage_rows=coverage)


def test_joint_regimes_use_same_checkpoint_and_preserve_missing_query_identity():
    trial = _joint_trial()
    cells = dm.assess_fit(trial,"attainable_train",regimes=["a","b"],rms_target=2e-4,max_target=2e-4,
                          expected_queries_per_regime=1)
    assert all(cell["passed"] for cell in cells)
    # A duplicated endpoint cannot stand in for a registered different query.
    changed = dict(trial,attainable_train={"rows":[trial["attainable_train"]["rows"][0],
        dict(trial["attainable_train"]["rows"][1],query_id="q-a")]})
    changed_cells = dm.assess_fit(changed,"attainable_train",regimes=["a","b"],rms_target=2e-4,max_target=2e-4,
                                expected_queries_per_regime=1)
    assert changed_cells[0]["passed"]
    assert changed_cells[1]["status"] == "UNRESOLVED"
    assert not changed_cells[1]["reference_complete"]


def test_physical_maximum_and_validation_are_separate_from_normalized_loss():
    trial = _joint_trial()
    fit = {"rows":[dict(row,conservative_maximum=.01,objective=1e-100) for row in trial["selected_train"]["rows"]]}
    trial["selected_train"] = fit
    cells = dm.assess_fit(trial,"selected_train",regimes=["a","b"],rms_target=2e-4,max_target=2e-4,
                          expected_queries_per_regime=1)
    assert all(cell["mse_reduction_passed"] for cell in cells)
    assert all(cell["absolute_accuracy_status"] == "FAILED" for cell in cells)
    # A good training diagnostic does not manufacture a validation result.
    validation = dm.assess_fit(trial,"selected_validation",regimes=["a","b"],rms_target=2e-4,max_target=2e-4,
                               expected_queries_per_regime=1)
    assert all(cell["absolute_accuracy_status"] == "UNRESOLVED" for cell in validation)


def test_floor_and_censoring_cannot_pass_even_with_small_absolute_errors():
    trial = _joint_trial()
    trial["attainable_train"] = {"rows":[dict(row,floor_limited=True) for row in trial["attainable_train"]["rows"]]}
    cells = dm.assess_fit(trial,"attainable_train",regimes=["a","b"],rms_target=2e-4,max_target=2e-4,
                          expected_queries_per_regime=1)
    assert all(cell["absolute_accuracy_status"] == "PASS" for cell in cells)
    assert all(cell["status"] == "UNRESOLVED" and not cell["passed"] for cell in cells)
    trial["status"] = "TIME_CENSORED"
    assert all(not cell["completed_requested_updates"] for cell in dm.assess_fit(trial,"attainable_train",
        regimes=["a","b"],rms_target=2e-4,max_target=2e-4,expected_queries_per_regime=1))


def test_accounting_sums_allocations_only_and_keeps_partial_state(tmp_path):
    ctx = _context(tmp_path)
    state = ctx.path.parent/"state"
    state.mkdir()
    payload = dict(schema="tdn.scheduler-accounting/v1",status="PENDING_ACCOUNTING",all_allocations_terminal=False,
        requested_job_ids=["100","101"],records=[
            dict(record_kind="allocation",allocation_job_id="100",elapsed_seconds=12,allocated_cpu_seconds=48),
            dict(record_kind="step",allocation_job_id="100",elapsed_seconds=12,allocated_cpu_seconds=48),
            dict(record_kind="allocation",allocation_job_id="101",elapsed_seconds=2,allocated_cpu_seconds=4)])
    write_json(state/"scheduler-accounting.json",payload)
    result,sources,tables = dm.accounting_sources(ctx)
    assert result["observed_allocation_seconds"] == 14
    assert result["observed_allocated_cpu_seconds"] == 52
    assert result["status"] == "PARTIAL_OR_NOT_AVAILABLE"
    assert len(result["allocation_rows"]) == 2
    assert result["monetary_cost"] is None
    assert sources[0]["path"].startswith("sources/accounting/")


def test_prospective_primary_readiness_requires_three_seeds_complete_six_regime_opportunities():
    regimes = ["a","b","c","d","e","f"]
    protocol = dict(training=dict(tiny_regimes=regimes,target_rms=2e-5,target_max=2e-5,
        tiny_squared_error_ratio=.01,tiny_required_regimes=5),train_horizons=[.1],validation_horizons=[.2],
        phase_variants=["original"],diagnostic=dict(primary_variants=["v"],full_primary_seeds=3,
            historical_pilot_decision="STOP_OR_UNRESOLVED"),trials={})
    trials = []
    for seed in (1,2,3):
        for regime in ["joint",*regimes]:
            ident = f"s{seed}-{regime}"
            selected_regimes = regimes if regime == "joint" else [regime]
            spec = dict(id=ident,family="temporal",variant_id="v",target_kind="fd_same_grid",seed=seed,
                regime=regime,phase="tiny_joint" if regime == "joint" else "tiny_individual",updates=10)
            protocol["trials"][ident] = spec
            rows = [dict(query_id=f"train-{r}",parent_id="parent-"+r,tiny_regime=r,split="train",
                reference_accepted=True,floor_limited=False,mse=1e-12,conservative_rms=1e-6,conservative_maximum=1e-6) for r in selected_regimes]
            val = [dict(row,query_id=row["query_id"].replace("train","validation"),split="validation") for row in rows]
            trial = dict(spec,trial_id=ident,status="COMPLETED_UPDATES",requested_updates=10,
                initial_train={"rows":[dict(r,mse=1e-6) for r in rows]},
                initial_validation={"rows":[dict(r,mse=1e-6) for r in val]},
                reference_coverage_rows=rows+val)
            trial.update({name:{"rows":rows} for name in ("selected_train","final_train","attainable_train","criterion_train")})
            trial["selected_validation"] = {"rows":val}
            trials.append(trial)
    result = dm.prospective_assessments(protocol,trials)
    outcome = result["variant_outcomes"][0]
    assert outcome["prospective_fresh_pilot_readiness"] is True
    assert outcome["eligible_for_development"] is False
    assert result["historical_gate"] == "STOP_OR_UNRESOLVED"
    # Five good regimes do not erase a missing declared teacher in the sixth.
    altered = [dict(t) for t in trials]
    altered[0]["attainable_train"] = {"rows":altered[0]["attainable_train"]["rows"][:-1]}
    after = dm.prospective_assessments(protocol,altered)
    assert after["variant_outcomes"][0]["prospective_fresh_pilot_readiness"] is False
    assert after["variant_outcomes"][0]["fit_status"] == "INSUFFICIENT_OPPORTUNITY"


def test_validation_query_initialization_ratio_uses_validation_initialization():
    trial = _trial()
    trial["initial_validation"] = {"mse":.008,"rows":[dict(query_id="validation-q",mse=.008)]}
    trial["selected_validation"] = {"mse":.002,"rows":[dict(query_id="validation-q",mse=.002,rms=.002**.5)]}
    row,queries = dm.trial_metrics(trial,source_path="sources/trial/training_results.json",source_sha256="a"*64)
    assert row["validation_initialization_mse_reduction"] == 4
    assert [q for q in queries if q["fit"] == "selected_validation"][0]["initialization_mse_reduction"] == 4


def test_compiled_initials_and_controls_keep_scalar_metrics_and_canonical_sources(tmp_path):
    trial = _trial()
    trial["learning_curve_export"]={"query_plan_file":"query_plan.json","query_plan_file_sha256":"a"*64}
    trial["optimization_history_summary"]={"committed_updates":7,"clipped_updates":1}
    row,queries=dm.trial_metrics(trial,source_path="sources/trial/training_results.json",source_sha256="b"*64)
    assert "rows" not in row["initial"]
    assert row["initial"]["mse"] == trial["initial_train"]["mse"]
    assert row["initial"]["endpoint_rows_count"] == 1
    assert row["initial"]["source_record"] == "/trials/0/initial_train"
    assert row["query_plan_source_path"] == "sources/trial/query_plan.json"
    assert row["query_plan_source_sha256"] == "a"*64
    assert row["optimization_history_summary"]["committed_updates"] == 7
    payload={"rows":[{"rms":.1}],"fp64_design_rows":[{"rms":.01}],"gram":[[1.,0.],[0.,1.]],
        "coefficients":[1.,2.],"condition_number":3.,"raw_coefficient_l2":5.}
    compact=dm.compact_control(payload)
    assert all(key not in compact for key in ("rows","fp64_design_rows","gram","coefficients"))
    assert compact["condition_number"] == 3.
    assert compact["gram_shape"] == [2,2]
    assert compact["endpoint_rows_count"] == compact["fp64_design_rows_count"] == 1
    assert compact["vector_sizes"]["coefficients"] == 2
    assert payload["rows"] == [{"rms":.1}]  # Canonical input is untouched.


def test_compact_json_publication_is_atomic_and_enforces_the_file_bound(tmp_path,monkeypatch):
    path=tmp_path/"compiled.json"
    dm.write_compact_json(path,{"measurements":[{"x":1.25}]*20})
    assert json.loads(path.read_text())["measurements"][0]["x"] == 1.25
    assert "\n  " not in path.read_text()
    original=path.read_bytes()
    monkeypatch.setattr(dm,"MAX_METADATA_BYTES",16)
    with pytest.raises(ValueError,match="bounded file size"):
        dm.write_compact_json(path,{"measurements":"x"*100})
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp-*"))


def test_public_curves_validate_query_plan_hashes_and_keep_source_links(tmp_path):
    ctx=_context(tmp_path)
    plan={"query_ids":["q"],"horizons":[.1],"parent_ids":["p"],"query_weights":[1.]}
    checksum=digest(plan)
    _seal(ctx,"train-one",{"training_results.json":{"trials":[_trial()]},
        "query_plan.json":{"plans":{checksum:plan}},
        "learning_curves.jsonl":[{"trial_id":"one","update":0,"query_plan_sha256":checksum}]})
    data,_=dm.collect(ctx)
    link=data["learning_curves"][0]["query_plan_sources"][0]
    assert link["source_path"] == "sources/train-one/query_plan.json"
    assert link["source_record"] == "/plans/"+checksum
    assert link["query_plan_sha256"] == checksum


def test_default_669_trial_public_metadata_shape_stays_bounded():
    # Measured fresh CPU record shapes are repeated solely for a storage-bound
    # fixture. These repeated values do not represent native scientific results.
    endpoint={'parent_id': 'response-diagnostic-train-low_mean-0-057c74316ea50464', 'query_id': '0cb6d4c5390b6ec4a30d1fdd', 'regime': 'isolated', 'tiny_regime': 'low_mean', 'target_kind': 'galerkin_same_grid', 'horizon': 0.04, 'split': 'train', 'rms': 5.701578660029475e-07, 'maximum': 9.570462732866325e-07, 'mean_error': -4.5794809107324586e-07, 'centered_rms': 3.3965208382608983e-07, 'sampled_rms': 5.701584464446385e-07, 'maximum_scope': 'sampled nodal maximum', 'norm_contract': 'physical interpolant L2; Nyquist mass weighting', 'mse': 3.25079992165035e-13, 'objective': 1.625399960825175e-13, 'anchor_rms': 5.701578660029475e-07, 'anchor_maximum': 9.570462732866325e-07, 'anchor_mse': 3.25079992165035e-13, 'candidate_over_anchor_rms': 1.0, 'physical_defect_rms': 5.654733408380476e-07, 'deployment_roundoff_rms': 1.8732489600579386e-08, 'deployment_roundoff_maximum': 3.279842557268875e-08, 'nearest_fp32_sampled_rms': 2.406674504310629e-09, 'fp32_resolution_scope': 'minimum nodal RMS for FP32 output; Galerkin physical norm is not a separable rounding bound', 'target_defect_rms': 5.701578660029475e-07, 'floor_limited': False, 'floor': 3.0461750255561526e-07, 'reference_accepted': True, 'uncertainty_rms': 2.202420403850391e-11, 'uncertainty_maximum': 2.5018875859927903e-11, 'conservative_rms': 5.70179890206986e-07, 'conservative_maximum': 9.570712921624924e-07, 'finite': True, 'trial_id': 'trial-0123456789abcdef', 'fit': 'selected_train', 'source_path': 'sources/trial-0123456789abcdef-000/training_results.json', 'source_sha256': 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', 'source_record': '/trials/0/selected_train/rows/0'}
    curve={'batch_mode': 'full_parent', 'clipped': False, 'consistency_gradient_norm': None, 'consistency_loss': None, 'data_count': 6, 'elapsed_seconds': 1.5874426549999043, 'endpoint_gradient_norm': None, 'endpoint_loss': 2.625762204602378e-11, 'examples_seen': 48, 'family': 'residual_fno', 'full_train_loss': 2.625806868236616e-11, 'full_train_maximum': 5.892011501207328e-05, 'full_train_mse': 5.2516137364732324e-11, 'full_train_rms': 4.698100716175067e-06, 'full_train_worst_tolerance_ratio': 2.946328809810827, 'gradient_norm': 4.59597443484812e-12, 'gradient_norm_after_clip': 4.59597443484812e-12, 'head_gradient_norm': 4.59597421608328e-12, 'head_only_warmup': False, 'learning_rate': 0.001, 'logical_updates': 2, 'model_id': 'trial-d5e3220d17d0cfad', 'objective_centered_scale': 1.0, 'objective_mean_scale': 1.0, 'objective_mode': 'physical', 'optimization_seconds': 0.6504448080086149, 'optimizer_gradient_norm': 4.59597443484812e-12, 'parameter_movement': 0.002042325598763205, 'phase': 'tiny_joint', 'rollout_gradient_norm': None, 'rollout_loss': None, 'scaled_objective': 2.625762204602378e-11, 'seed': 118010011, 'source_aux_loss': None, 'source_gradient_norm': None, 'target_kind': 'galerkin_same_grid', 'train_loss': 2.625762204602378e-11, 'trial_id': 'trial-d5e3220d17d0cfad', 'trunk_gradient_norm': 1.4180533054222897e-15, 'trunk_movement': 1.4169958488956604e-06, 'update': 2, 'validation_centered_rms': 1.955763166940818e-06, 'validation_loss': 4.3287803472485605e-12, 'validation_maximum': 2.3336096082848812e-05, 'validation_mean_absolute': 9.10128878505676e-07, 'validation_rms': 2.242753430933754e-06, 'variant_id': 'residual_numerical_remainder', 'query_plan_sha256': 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', 'citations': [{'source_path': 'sources/trial-0123456789abcdef-000/learning_curves.jsonl', 'source_sha256': 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', 'source_record': '/lines/51', 'stage': 'trial-0123456789abcdef-000'}], 'query_plan_sources': [{'source_path': 'sources/trial-0123456789abcdef-000/query_plan.json', 'source_sha256': 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', 'source_record': '/plans/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', 'query_plan_sha256': 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'}]}
    trial,_=dm.trial_metrics(_trial(),source_path="sources/trial/training_results.json",source_sha256="a"*64)
    data={"scope":"METADATA_SIZE_PROJECTION_ONLY", "trials":[trial]*669,
        "endpoint_queries":[endpoint]*(139*228+530*38),
        "learning_curves":[curve]*(669*52),
        "screening_cost_source_indices_safety_margin":"x"*(8<<20)}
    class Counter:
        bytes=0
        def write(self,value):self.bytes+=len(value.encode("utf-8"))
    stream=Counter()
    json.dump(data,stream,sort_keys=True,separators=(",",":"),allow_nan=False)
    assert stream.bytes < 200<<20
    assert stream.bytes < dm.MAX_METADATA_BYTES
