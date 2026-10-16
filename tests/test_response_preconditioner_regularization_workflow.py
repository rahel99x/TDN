"""Synthetic retained-array integration; never fit the real P1 bank locally."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tdn.analysis.response_preconditioner import regularization_workflow as w
from tdn.analysis.response_preconditioner.regularization_protocol import protocol,validate_config,_fixture_config,CONTROLS


class SyntheticEvidence:
    """Independent generated algebraic bank; no PDE integration or production seeds."""
    def __init__(self):
        self.identity={"scope":"synthetic implementation fixture, never native evidence"}
        self.parents=[dict(parent_id=f"{split}-{i:03}",split=split,parent_index=i,parent_family="synthetic")
            for split,count in (("train",2),("validation",1),("diagnostic",1)) for i in range(count)]
        self.queries=[];self.arrays={};self.access=[];self.released=False;self.closed=False
        rng=np.random.default_rng(137)
        self.truth=np.linspace(-.1,.2,40)
        for p in self.parents:
            for context in range(4):
                for role,count in (("calibration",2),("withheld",1)) if p["split"]=="train" else (("withheld",1),):
                    for source in range(count):
                        qid=f"{p['parent_id']}-{context}-{role}-{source}"
                        ref=dict(accepted=True,uncertainty_rms=1e-11,uncertainty_maximum=2e-11,
                                 matched_mean=dict(uncertainty_rms=1e-11,uncertainty_maximum=2e-11))
                        q=dict(query_id=qid,parent_id=p['parent_id'],parent_family=p['parent_family'],split=p['split'],role=role,
                            family="dense_random",source_index=source,source_id=f"{role}-{source}",context_id=f"{p['parent_id']}-{context}",
                            start=0.,stop=(context+1)*.01,status="ACCEPTED",reference=ref,reference_seconds=.01,design_seconds=.001)
                        f=np.zeros(8)
                        if p['split']=='train':f[p['parent_index']*4+context]=1.
                        else:f=rng.normal(size=8)
                        branches=rng.normal(size=(5,1,1,5,5))*1e-4
                        design=np.einsum('j...,k->jk...',branches,f).reshape(40,1,1,5,5)
                        base=rng.normal(size=(1,1,5,5))*.2
                        teacher=base+np.einsum('j,j...->...',self.truth,design)+rng.normal(size=base.shape)*1e-10
                        self.arrays[qid]=dict(design=design,base=base,teacher=teacher,matched_mean=base+.01,
                            refined=teacher+rng.normal(size=base.shape)*1e-12,features=f,branches=branches)
                        self.queries.append(q)

    def read_query(self,qid):
        q=next(q for q in self.queries if q['query_id']==qid)
        protected=q['split']=='diagnostic' or(q['split']=='train' and q['role']=='withheld')
        if protected and not self.released:raise AssertionError('Protected numeric data read before seal')
        self.access.append((qid,self.released))
        return dict(metadata=deepcopy(q),arrays=deepcopy(self.arrays[qid]))

    def read_original_selection(self):
        rows=[dict(metadata=q,arrays=self.arrays[q['query_id']]) for q in self.queries if q['role']=='calibration']
        s=w._stack(rows)
        augmented=np.vstack([s['Aw'],np.sqrt(1e-10)*np.eye(40)])
        rhs=np.concatenate([s['bw'],np.zeros(40)])
        self.original_coefficients=np.linalg.lstsq(augmented,rhs,rcond=None)[0]
        return dict(cells=[dict(model_id='fit-p002-s02',selected=dict(coefficients=self.original_coefficients.tolist()),
            selection_seal='synthetic-original',train_query_ids=[r['metadata']['query_id'] for r in rows],
            validation_query_ids=[q['query_id'] for q in self.queries if q['split']=='validation'])])

    def release_protected(self,path,sha):
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
        x=json.loads(Path(path).read_text());assert x['all_selections_sealed']
        assert len(x['cells'])==1 and len(x['cells'][0]['selected'])==2
        self.released=True

    def read_predictions(self,qid):
        assert self.released
        a=self.arrays[qid]
        return {'fit-p002-s02':a['base']+np.einsum('j,j...->...',self.original_coefficients,a['design']), 'ordered_2':a['base'], 'ordered_8':a['teacher']+1e-6,
            'lawson_2':a['teacher']+1e-9,'lawson_4':a['teacher']+1e-10,'lawson_8':a['teacher']+1e-11}

    def close(self):self.closed=True


def read(path):return json.loads(path.read_text())


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture(scope='module')
def completed(tmp_path_factory):
    path=tmp_path_factory.mktemp('regularization-synthetic-complete')
    bank=SyntheticEvidence()
    old=torch.get_num_threads();torch.set_num_threads(1)
    try:summary=w._study(path,_fixture_config(),evidence=bank,allow_fixture=True,plots=False)
    finally:torch.set_num_threads(old)
    return path,bank,summary


def test_frozen_protocol_counts_and_new_work_limits():
    p=protocol();validate_config(p)
    assert p['expected_candidates']==39 and p['expected_positive_fits']==33
    assert p['expected_metrics']==19584 and p['expected_selected_models']==6
    assert p['fresh_reference_solves']==p['new_timing_bundles']==0
    assert p['budget']['native_only'] and not p['budget']['automatic_expansion']
    assert not p['sensitivity']['selection_gate']


@pytest.mark.parametrize('key,value',[('coefficient_ridges',[1e-30]),('action_energies',[1e-20]),
    ('fresh_reference_solves',1),('expected_metrics',1),('budget',dict(study_seconds=99999))])
def test_changed_scientific_contract_rejected(key,value):
    p=protocol();p[key]=value
    with pytest.raises(ValueError):validate_config(p)


def test_private_fixture_forbidden_in_production():
    with pytest.raises(ValueError):validate_config(_fixture_config())
    with pytest.raises(ValueError):protocol('smoke')


def test_injected_evidence_cannot_release_production_work(tmp_path):
    with pytest.raises(ValueError,match='private'):
        w._study(tmp_path,protocol(),evidence=SyntheticEvidence())


def test_native_context_required_before_any_output(tmp_path):
    dest=tmp_path/'new'
    with pytest.raises(ValueError,match='native'):
        w.execute('study',dest)
    assert not dest.exists()
    with pytest.raises(ValueError,match='CPU'):
        w.execute('verify',dest,device='cuda')


def test_synthetic_complete_scope_and_all_selections_before_protected_reads(completed):
    output,bank,summary=completed
    assert summary['measured_metrics']==144 and summary['expected_metrics']==144
    assert summary['failed_metrics']==summary['missing_metrics']==summary['unresolved_metrics']==0
    assert summary['selected_models']==2 and summary['attempted_candidates']==13
    assert summary['read_input_queries']==32 and bank.closed
    for qid,released in bank.access:
        q=next(q for q in bank.queries if q['query_id']==qid)
        if q['split']=='diagnostic' or(q['split']=='train' and q['role']=='withheld'):assert released
    events=[json.loads(x) for x in (output/'journal.jsonl').read_text().splitlines()]
    frozen=next(i for i,e in enumerate(events) if e['event']=='all_selections_frozen')
    assert all(i>frozen for i,e in enumerate(events) if e['event']=='retained_query_read' and(e['split']=='diagnostic' or(e['split']=='train' and e['role']=='withheld')))


def test_validation_only_family_selection_preserves_all_candidates(completed):
    output,_,_=completed
    cell=read(output/'selection.json')['cells'][0]
    assert not cell['diagnostic_truth_accessed'] and not cell['training_withheld_truth_accessed']
    assert len(cell['candidates'])==13
    for family,chosen in cell['selected'].items():
        eligible=[c for c in cell['candidates'] if c['eligible'] and c['family'] in ('zero',family)]
        assert chosen==min(eligible,key=lambda c:c['validation_mse'])
        assert chosen['family']!='projection'
    projection=next(c for c in cell['candidates'] if c['family']=='projection')
    assert not projection['eligible']
    assert all(q.startswith('validation-') for q in cell['validation_query_ids'])
    assert all('calibration' in q and q.startswith('train-') for q in cell['train_query_ids'])
    assert cell['selection_seal']
    assert cell['original_strength_reproduction']['status']=='PASS'


def test_full_rank_action_energy_has_known_uniform_shrinkage(completed):
    output,_,_=completed
    cell=read(output/'selection.json')['cells'][0]
    assert len(cell['action_energy_projection_equivalence'])==5
    assert max(x['weighted_prediction_difference'] for x in cell['action_energy_projection_equivalence'])<1e-17
    assert all(c['diagnostics']['eligible'] for c in cell['candidates'] if c['family']=='action_energy')


def test_metric_replay_and_mean_spatial_identity(completed):
    output,bank,_=completed
    rows=read(output/'metrics.json')
    for row in rows:
        with np.load(output/'predictions'/(row['query_id']+'.npz'),allow_pickle=False) as a:prediction=a[row['method']]
        error=prediction-bank.arrays[row['query_id']]['teacher']
        assert row['mse']==float(np.mean(error**2))
        assert row['maximum']==float(np.max(np.abs(error)))
        assert row['mean_component_mse']==float(np.mean(error))**2
        assert row['spatial_centered_mse']==float(np.mean((error-error.mean())**2))
    groups=read(output/'learning_curves.json')
    for row in groups:
        assert row['mse']==pytest.approx(row['mean_component_mse']+row['spatial_centered_mse'],rel=2e-12,abs=1e-25)


def test_candidate_sensitivity_visible_without_becoming_a_gate(completed):
    output,_,_=completed
    rows=read(output/'sensitivity.json')
    assert len(rows)==12
    assert all(not r['scientific_eligibility_gate'] and not r['error_certificate'] for r in rows)
    assert all(r['prediction_operator_gain']>=0 for r in rows)
    assert all(r['estimated_ball_prediction_rms']>=0 for r in rows)


def test_costs_separate_reused_science_from_new_audit(completed):
    output,bank,_=completed
    c=read(output/'costs.json')
    assert c['fresh_reference_solves']==c['new_timing_bundles']==0
    assert c['original_reference_seconds']==sum(q['reference_seconds'] for q in bank.queries)
    assert c['reused_reference_pairs']==32
    assert c['input_read_seconds']>0 and c['fitting_seconds']>0 and c['evaluation_seconds']>0
    assert c['sensitivity_seconds']<=c['fitting_seconds']


def test_all_positive_fit_failures_still_keep_zero_and_projection(tmp_path,monkeypatch):
    def fail(*args,**kwargs):raise ArithmeticError('synthetic solver failure')
    monkeypatch.setattr(w,'fit_parent_balanced_tikhonov',fail)
    bank=SyntheticEvidence()
    summary=w._study(tmp_path,_fixture_config(),evidence=bank,allow_fixture=True,plots=False)
    assert summary['failed_candidates']==11
    cell=read(tmp_path/'selection.json')['cells'][0]
    assert all(c['name']=='zero_readout' for c in cell['selected'].values())
    assert summary['measured_metrics']==144


def test_timeout_retains_full_missing_denominator(tmp_path,monkeypatch):
    def timeout(self):raise TimeoutError('synthetic bounded deadline')
    monkeypatch.setattr(w.RegularizationStudy,'check',timeout)
    bank=SyntheticEvidence()
    with pytest.raises(TimeoutError):w._study(tmp_path,_fixture_config(),evidence=bank,allow_fixture=True,plots=False)
    assert read(tmp_path/'denominators.json')['missing_metrics']==144
    assert not read(tmp_path/'selection.json')['all_selections_sealed']
    assert bank.closed


def test_post_selection_missing_input_preserves_pending_rows(tmp_path):
    bank=SyntheticEvidence();original=bank.read_query
    def fail(qid):
        if qid.startswith('diagnostic-'):raise ValueError('retained input changed')
        return original(qid)
    bank.read_query=fail
    with pytest.raises(ValueError,match='changed'):
        w._study(tmp_path,_fixture_config(),evidence=bank,allow_fixture=True,plots=False)
    d=read(tmp_path/'denominators.json')
    assert d['measured_metrics']>0 and d['missing_metrics']>0
    assert read(tmp_path/'selection.json')['all_selections_sealed'] and bank.closed


def test_nonfinite_or_inconsistent_prediction_rejected():
    bank=SyntheticEvidence();a=next(iter(bank.arrays.values()))
    with pytest.raises(ValueError):w._prediction(a,np.full(40,np.nan))
    corrupted=deepcopy(a);corrupted['branches']*=2
    with pytest.raises(ArithmeticError):w._prediction(corrupted,bank.truth)


def test_parent_weighting_not_scalar_row_count():
    bank=SyntheticEvidence()
    a=[bank.read_query(q['query_id']) for q in bank.queries if q['role']=='calibration']
    selected=a[:2]+a[8:9]
    s=w._stack(selected)
    for pid in set(s['parent_ids']):
        assert s['omega'][np.asarray(s['parent_ids'])==pid].sum()==pytest.approx(.5)


def test_timeout_mid_query_keeps_every_measured_prediction(tmp_path,monkeypatch):
    original=w.RegularizationStudy.check
    def check(self):
        if any(row['status']=='MEASURED' for row in self.metrics.values()):
            raise TimeoutError('interrupted after first method')
        original(self)
    monkeypatch.setattr(w.RegularizationStudy,'check',check)
    with pytest.raises(TimeoutError):
        w._study(tmp_path,_fixture_config(),evidence=SyntheticEvidence(),allow_fixture=True,plots=False)
    rows=read(tmp_path/'metrics.json'); measured=[r for r in rows if r['status']=='MEASURED']
    assert len(measured)==1
    row=measured[0]
    with np.load(tmp_path/'predictions'/(row['query_id']+'.npz'),allow_pickle=False) as a:
        assert a.files==[row['method']]
    assert read(tmp_path/'denominators.json')['missing_metrics']==143


def test_reporting_time_is_included_in_study_deadline(tmp_path,monkeypatch):
    original=w.RegularizationStudy.finalize
    def finalize(self):
        original(self)
        self.deadline=-1.
    monkeypatch.setattr(w.RegularizationStudy,'finalize',finalize)
    with pytest.raises(TimeoutError):
        w._study(tmp_path,_fixture_config(),evidence=SyntheticEvidence(),allow_fixture=True,plots=False)
    assert read(tmp_path/'denominators.json')['measured_metrics']==144


def test_wrong_original_training_prefix_blocks_before_protected_read(tmp_path):
    bank=SyntheticEvidence();original=bank.read_original_selection
    def wrong():
        x=original();x['cells'][0]['train_query_ids'].reverse();return x
    bank.read_original_selection=wrong
    with pytest.raises(ValueError,match='prefixes'):
        w._study(tmp_path,_fixture_config(),evidence=bank,allow_fixture=True,plots=False)
    assert not bank.released and bank.closed


def test_primary_failure_survives_reporting_and_close_failures(tmp_path,monkeypatch):
    def timeout(self):raise TimeoutError('primary deadline')
    def reporting(self):raise OSError('report unavailable')
    bank=SyntheticEvidence()
    def close():
        bank.closed=True
        raise ValueError('input changed during close')
    bank.close=close
    monkeypatch.setattr(w.RegularizationStudy,'check',timeout)
    monkeypatch.setattr(w.RegularizationStudy,'finalize',reporting)
    with pytest.raises(TimeoutError,match='primary deadline') as caught:
        w._study(tmp_path,_fixture_config(),evidence=bank,allow_fixture=True,plots=False)
    failures=read(tmp_path/'finalization_failure.json')
    assert [x['stage'] for x in failures]==['reporting','input_close']
    assert bank.closed and len(caught.value.__notes__)==2


def test_close_failure_after_complete_metrics_is_not_success(tmp_path):
    bank=SyntheticEvidence()
    def close():raise ValueError('input changed during close')
    bank.close=close
    with pytest.raises(ValueError,match='input changed'):
        w._study(tmp_path,_fixture_config(),evidence=bank,allow_fixture=True,plots=False)
    assert read(tmp_path/'denominators.json')['measured_metrics']==144
    assert read(tmp_path/'finalization_failure.json')[0]['stage']=='input_close'


