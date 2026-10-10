"""Replay correctness, information boundaries and immutable-source visualization."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tdn.analysis.adjacent.fields import make_parent, sample_field
from tdn.analysis.adjacent.models import make_model
from tdn.analysis.adjacent import prediction_view as view
from tdn.numerics import Equation, Geometry


@pytest.fixture(scope="module", autouse=True)
def bounded_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def example():
    parent = make_parent(90531, .5, levels=2, rms=.04)
    initial = sample_field(parent, 8).numpy()
    eq, geom = Equation(.004, 3.), Geometry((8, 8), (1., 1.))
    torch.manual_seed(83017)
    model = make_model("channel_neural", "discrete", dict(modes=2, split_modes=1)).float()
    with torch.no_grad():
        model.conditioner[-1].weight.fill_(.03)
        model.conditioner[-1].bias.copy_(torch.tensor([.02, -.01, .04]))
        truth = model(torch.from_numpy(initial).float(), .06, eq, geom).double().numpy() + 1e-6
    return parent, initial, truth, model, eq, geom


def test_prediction_view_exact_channel_and_parameter_mapping(example):
    _, initial, truth, model, eq, geom = example
    info, arrays = view.compute_prediction_data(model, initial, truth, .06, eq, geom)
    reconstructed = arrays['baseline'] + np.sum(arrays['channels'] * arrays['gains'][..., None, None], axis=1, keepdims=True)
    np.testing.assert_allclose(reconstructed, arrays['prediction'], atol=6e-8, rtol=0)
    np.testing.assert_allclose(arrays['error'], arrays['prediction'].astype('float64') - truth, atol=0, rtol=0)
    first = arrays['features'] @ arrays['parameter__conditioner_0_weight'].T + arrays['parameter__conditioner_0_bias']
    hidden = first / (1 + np.exp(-first))
    logits = hidden @ arrays['parameter__conditioner_2_weight'].T + arrays['parameter__conditioner_2_bias']
    np.testing.assert_allclose(1+.75*np.tanh(logits), arrays['gains'], atol=1e-7, rtol=0)
    assert info['network']['trainable_parameters'] == 59
    assert info['network']['exact_gain_reconstruction']
    assert info['physical']['grid'] == [8, 8]
    assert info['error_rms'] == pytest.approx(1e-6, rel=1e-9)


def test_prediction_view_reference_cannot_influence_forward(example):
    _, initial, truth, model, eq, geom = example
    first, a = view.compute_prediction_data(model, initial, truth, .06, eq, geom)
    second, b = view.compute_prediction_data(model, initial, truth+.01, .06, eq, geom)
    for key in ('prediction','channels','gains','features','baseline','correction'):
        np.testing.assert_array_equal(a[key], b[key])
    assert first['error_rms'] != second['error_rms']
    assert not first['reference_used_for_inference']


def test_prediction_view_gain_sensitivity_is_not_network_weight_sensitivity(example):
    _, initial, truth, model, eq, geom = example
    info, arrays = view.compute_prediction_data(model, initial, truth, .06, eq, geom)
    for channel in range(3):
        for index, delta in enumerate(arrays['gain_perturbation_deltas']):
            np.testing.assert_allclose(arrays['gain_perturbation_changes'][channel,index],
                delta * arrays['gain_derivative'][:,channel:channel+1].astype('float64'), atol=7e-17, rtol=1e-7)
    assert info['sensitivity']['not_input_or_network_weight_derivative']
    with pytest.raises(ValueError, match='perturbation'):
        view.compute_prediction_data(model, initial, truth, .06, eq, geom, gain_delta=1.)


def test_prediction_view_detects_field_identity_mismatch(example):
    parent, initial, truth, _, _, _ = example
    row = dict(parent=parent.to_dict(), parent_index=0, grid=8, state_key='state', key='truth')
    bank = {'binding': {'parents': [parent.to_dict()]}}
    stored = {'state': initial.copy(), 'truth': truth.copy()}
    recovered, a, b = view._validate_bank_arrays(bank, stored, row)
    assert recovered.identity_sha256 == parent.identity_sha256
    np.testing.assert_array_equal(a, initial)
    np.testing.assert_array_equal(b, truth)
    stored['state'][0,0,0,0] += .001
    with pytest.raises(ValueError, match='continuous parent'):
        view._validate_bank_arrays(bank, stored, row)


def test_prediction_view_output_guard_before_source_access(tmp_path, monkeypatch):
    monkeypatch.setattr(view, 'ROOT', tmp_path)
    source = tmp_path/'source'; source.mkdir()
    with pytest.raises(ValueError, match='new output'):
        view.render_prediction_view(source, source/'graphics')
    with pytest.raises(ValueError, match='inside this project'):
        view._contained(tmp_path.parent/'outside')
    target = tmp_path/'target'; target.mkdir()
    linked = tmp_path/'link'; linked.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match='symlinks'):
        view._contained(linked/'new')
    with pytest.raises(ValueError, match='400'):
        view.render_prediction_view(source, tmp_path/'new', dpi=72)


def test_prediction_view_renders_actual_cells_at_small_test_dpi(example, tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.axes import Axes
    _, initial, truth, model, eq, geom = example
    info, arrays = view.compute_prediction_data(model, initial, truth, .06, eq, geom)
    info.update(selected_model={'family':'channel_neural','selection_status':'TEST_FIXTURE_ONLY',
                'selected_update':0,'checkpoint_sha256':'0'*64}, parent_id='unit-test-field')
    calls = []
    original = Axes.imshow
    def tracked(self, value, *args, **kwargs):
        calls.append((np.asarray(value).shape, kwargs.get('interpolation')))
        return original(self, value, *args, **kwargs)
    monkeypatch.setattr(Axes, 'imshow', tracked)
    paths = view._images(info, arrays, tmp_path, dpi=40)
    assert len(paths) == 6 and all((tmp_path/p).is_file() for p in paths)
    assert all(interpolation == 'nearest' for _, interpolation in calls)
    assert sum(shape == (8,8) for shape, _ in calls) >= 20


def test_prediction_view_actual_sealed_replay_if_available(tmp_path, monkeypatch):
    source = view.ROOT/'runs/adjacent-local-review-v2'
    if not source.exists():
        pytest.skip('Archived local development run is not distributed as a test fixture')
    bundle = view.load_bundle(source, family='channel_neural', track='discrete', parent_index=0)
    before = deepcopy(bundle['provenance']['source_artifacts'])
    info, _ = view.compute_prediction_data(bundle['model'], bundle['initial'], bundle['reference_state'],
        bundle['reference']['horizon'], bundle['equation'], bundle['geometry'])
    assert info['error_rms'] == pytest.approx(bundle['original_evaluation']['error_rms'], abs=1e-12)
    assert info['error_max'] == pytest.approx(bundle['original_evaluation']['error_max'], abs=1e-12)
    assert bundle['spec']['selection_status'] == 'FITTED_CHECKPOINT'
    assert bundle['spec']['selected_update'] == 2
    assert all(view.file_digest(view.ROOT/path) == expected for path,expected in before.items())
    # Exercise provenance and complete output sealing; plotting has a separate
    # actual-cell render test, so this avoids duplicating high-resolution work.
    monkeypatch.setattr(view, '_images', lambda *args, **kwargs: [])
    output = tmp_path / 'sealed-replay'
    result = view.render_prediction_view(source, output)
    manifest = json.loads((output/'manifest.json').read_text())
    assert result['replay']['status'] == 'SAME_CPU_METRIC_REPLAY'
    assert manifest['source_artifacts'] == before
    assert all(view.file_digest(output/path) == expected
               for path, expected in manifest['files'].items())
    assert all(view.file_digest(view.ROOT/path) == expected
               for path, expected in manifest['renderer']['files'].items())


def test_prediction_view_near_bound_fitted_gains_clip_probes(example):
    _, initial, truth, _, eq, geom = example
    model = make_model('channel_affine', 'discrete', dict(modes=2, split_modes=1)).float()
    with torch.no_grad():
        model.affine_coefficients[0].copy_(torch.tensor([-20., 20., 0.]))
    info, arrays = view.compute_prediction_data(model, initial, truth, .06, eq, geom)
    actual = arrays['gain_perturbation_actual_deltas']
    assert actual.shape == (3, 9)
    assert actual[0, 0] == 0 and actual[1, -1] == 0
    assert actual[2, 0] == pytest.approx(-.1) and actual[2, -1] == pytest.approx(.1)
    perturbed = arrays['gains'].reshape(3,1) + actual
    assert np.all(perturbed >= .25) and np.all(perturbed <= 1.75)
    for channel in range(3):
        for index, delta in enumerate(actual[channel]):
            np.testing.assert_allclose(arrays['gain_perturbation_changes'][channel,index],
                delta * arrays['gain_derivative'][:,channel:channel+1].astype('float64'), atol=7e-17, rtol=1e-7)
    assert info['sensitivity']['curves'][0]['clipped_to_admissible_range']
    assert info['sensitivity']['curves'][1]['clipped_to_admissible_range']
    assert not info['sensitivity']['curves'][2]['clipped_to_admissible_range']


def test_prediction_view_smoothing_is_display_only(example, tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.axes import Axes
    _, initial, truth, model, eq, geom = example
    info, arrays = view.compute_prediction_data(model, initial, truth, .06, eq, geom)
    info.update(selected_model={'family':'channel_neural','selection_status':'TEST_FIXTURE_ONLY',
                'selected_update':0,'checkpoint_sha256':'0'*64}, parent_id='unit-test-field')
    before_info = json.dumps(info, sort_keys=True)
    before_arrays = {key: value.copy() for key,value in arrays.items()}
    calls = []
    original = Axes.imshow
    def tracked(self, value, *args, **kwargs):
        calls.append((np.asarray(value).shape, kwargs.get('interpolation')))
        return original(self, value, *args, **kwargs)
    monkeypatch.setattr(Axes, 'imshow', tracked)
    paths = view._images(info, arrays, tmp_path, dpi=20, spatial_display='both')
    assert len(paths) == 12 and len(set(paths)) == 12
    assert all((tmp_path/p).is_file() for p in paths)
    assert 'prediction-architecture.png' in paths
    assert 'prediction-architecture-bicubic.png' in paths
    spatial_styles = {style for shape,style in calls if shape == (8,8)}
    assert spatial_styles == {'nearest', 'bicubic'}
    assert json.dumps(info, sort_keys=True) == before_info
    for key,value in arrays.items():
        np.testing.assert_array_equal(value, before_arrays[key])
    with pytest.raises(ValueError, match='Spatial display'):
        view._images(info, arrays, tmp_path, dpi=20, spatial_display='fabricated')


def test_prediction_view_native_declaration_and_recovery(tmp_path, monkeypatch):
    monkeypatch.setattr(view, 'ROOT', tmp_path)
    run = tmp_path/'run'; run.mkdir()
    origin = tmp_path/'origin'; origin.mkdir()
    (origin/'workflow-seal.json').write_text('original execution proof')
    protocol = {'profile':'resolution-smoke', 'units':{'freeze':{}}}
    declaration = {'scientific_protocol':protocol}
    (run/'protocol.json').write_text(json.dumps(declaration))
    bridge = {'stage_paths':{'freeze':str(origin)}, 'resume_paths':{},
              'source_sha256':'scientific-source', 'protocol_sha256':view.digest(declaration),
              'seals':{'freeze':view.file_digest(origin/'workflow-seal.json')}}
    (run/'recovery.json').write_text(json.dumps(bridge))
    workflow = {'profile':protocol['profile'], 'protocol_path':str(run/'protocol.json'),
        'protocol_sha256':view.digest(declaration), 'source_sha256':'scientific-source',
        'recovery':{'manifest_path':str(run/'recovery.json'),
                    'sha256':view.file_digest(run/'recovery.json'),
                    'stage_paths':bridge['stage_paths'], 'resume_paths':{}}}
    (run/'adjacent-workflow.json').write_text(json.dumps(workflow))
    recovered, paths, sources = view._coordinator(run)
    assert recovered == protocol and paths['freeze'] == origin
    assert set(path.name for path in sources) == {'protocol.json','adjacent-workflow.json','recovery.json'}
    (origin/'workflow-seal.json').write_text('changed execution proof')
    with pytest.raises(ValueError, match='seal changed'):
        view._coordinator(run)


@pytest.fixture
def resolution_replay(tmp_path, monkeypatch):
    """Synthetic sealed 64² fixture: validates artifact plumbing, never solver accuracy."""
    from tdn.analysis.adjacent import engine
    from tdn.analysis.adjacent.resolution_protocol import build_resolution_protocol
    from tdn.analysis.adjacent.resolution_diagnostics import (
        BANK_SCHEMA, resolution_parent, sample_resolution_parent)
    from tdn.analysis.adjacent.resolution_learning import SCHEMA as LEARNING_SCHEMA
    monkeypatch.setattr(view, 'ROOT', tmp_path)
    monkeypatch.setattr(engine, 'ROOT', tmp_path)
    # The synthetic fixture has no Git history; production still requires exact
    # scientific-source hashes. All file seals/checkpoints/fields below are real.
    monkeypatch.setattr(view, '_implementation_identity', lambda *args, **kwargs: {'fixture':'synthetic'})
    protocol = build_resolution_protocol('resolution-smoke')
    run = tmp_path/'resolution-run'; run.mkdir()
    def write(path, value):
        path.write_text(json.dumps(value))
    write(run/'local-protocol.json', protocol)
    train_name = next(name for name,unit in protocol['units'].items()
        if unit['kind']=='resolution_train' and unit['grid']==64 and unit['track']=='discrete' and unit['group']=='ours')
    eval_name = next(name for name,unit in protocol['units'].items()
        if unit['kind']=='resolution_evaluate' and unit['grid']==64 and unit['track']=='discrete' and 0 in unit['field_indices'])
    bank_name = protocol['units'][eval_name]['bank_units'][0]
    for name in (train_name,'freeze',bank_name,eval_name):
        (run/name).mkdir()
    train, frozen, bank, evaluation = (run/name for name in (train_name,'freeze',bank_name,eval_name))
    model = make_model('channel_neural','discrete',protocol['resolution']['model_config']).float().eval()
    with torch.no_grad():
        model.conditioner[-1].bias.copy_(torch.tensor([.03,-.02,.01]))
    (frozen/'checkpoints').mkdir()
    checkpoint = frozen/'checkpoints/selected.pt'
    torch.save({'state_dict':model.state_dict()},checkpoint)
    spec = dict(model_id='n64/discrete/channel_neural/seed871001',family='channel_neural',track='discrete',
        train_grid=64,grid=64,seed=871001,config=protocol['resolution']['model_config'],
        checkpoint='checkpoints/selected.pt',checkpoint_sha256=view.file_digest(checkpoint),
        selection_status='FITTED_CHECKPOINT',updates_selected=2,source_unit=train_name)
    write(frozen/'catalog.json',[spec]); write(train/'catalog.json',[spec])
    freeze = dict(schema=LEARNING_SCHEMA,protocol_sha256=view.digest(protocol),
        artifacts={'catalog.json':view.file_digest(frozen/'catalog.json'),spec['checkpoint']:spec['checkpoint_sha256']})
    write(frozen/'freeze.json', dict(freeze,freeze_sha256=view.digest(freeze)))
    parent = resolution_parent(protocol,'evaluation',0,64)
    initial = sample_resolution_parent(parent,64)
    h = protocol['resolution']['evaluation_horizons'][0]
    eq,geom = Equation(parent['kappa'],parent['reaction_rate']),Geometry((64,64),tuple(parent['domain']))
    with torch.no_grad():
        prediction = model(initial.float(),h,eq,geom).double()
    truth = prediction+1e-6
    np.savez(bank/'initial.npz',initial=initial.numpy())
    np.savez(bank/'reference.npz',reference=truth.numpy())
    meta = dict(parent,parent=parent,grid=64,N=64,track='discrete',horizon=h,
        initial_file='initial.npz',reference_file='reference.npz',
        initial_sha256=view.file_digest(bank/'initial.npz'),reference_sha256=view.file_digest(bank/'reference.npz'),
        accepted=True,uncertainty_rms=0.,uncertainty_max_bound=0.,scope='SYNTHETIC_TEST_FIXTURE')
    write(bank/'reference_bank.json',dict(schema=BANK_SCHEMA,status='COMPLETED',entries=[meta]))
    error = prediction-truth
    row = dict(model_id=spec['model_id'],checkpoint_sha256=spec['checkpoint_sha256'],parent_id=parent['parent_id'],
        track='discrete',grid=64,schedule=[h],final_time=h,error_rms=float(error.square().mean().sqrt()),
        error_max=float(error.abs().max()))
    # The full program also records each parent's batched throughput error.
    # Single-field prediction replay must not silently select that endpoint.
    write(evaluation/'evaluation-rows.json',[row,dict(row,batch_size=4,error_rms=123.,error_max=456.)])
    def seal(name):
        path=run/name; unit=protocol['units'][name]
        software={'source_tree_sha256':'fixture-source','git_commit':'0'*40}
        write(path/'protocol.json',protocol)
        write(path/'summary.json',dict(schema=protocol['schema'],status='COMPLETED',stage=name,
            protocol_sha256=view.digest(protocol),source_tree_sha256='fixture-source',device='cpu',experiment_count=0))
        (path/'rows.jsonl').write_text(''); write(path/'rows.json',{'rows':[]})
        for filename in ('review.csv','review.md','summary.txt'):
            (path/filename).write_text('Synthetic fixture, not scientific evidence.\n')
        prior={dependency:view.file_digest(run/dependency/'science_manifest.json')
            if (run/dependency/'science_manifest.json').is_file() else '0'*64 for dependency in unit['dependencies']}
        manifest=dict(schema='tdn.adjacent-science/v1',stage=name,kind=unit['kind'],
            protocol_sha256=view.digest(protocol),unit_sha256=view.digest(unit),source_tree_sha256='fixture-source',
            software=software,device='cpu',prerequisites=prior,artifacts=engine.inventory(path))
        write(path/'science_manifest.json',manifest);(path/'COMPLETED').write_text(view.digest(manifest))
        write(path/'stage.json',{'status':'COMPLETED'})
        write(path/'execution.json',{'stage':name,'software':software})
        write(path/'workflow-seal.json',dict(schema=protocol['schema'],protocol_sha256=view.digest(protocol),
            files={file:view.file_digest(path/file) for file in ('execution.json','protocol.json','stage.json','science_manifest.json')}))
    for name in (train_name,'freeze',bank_name,eval_name):
        seal(name)
    return run,initial.numpy(),truth.numpy(),frozen,bank


def test_prediction_view_resolution_checkpoint_and_field_replay(resolution_replay):
    run,initial,truth,_,_ = resolution_replay
    bundle=view.load_bundle(run,grid=64,train_grid=64,seed=871001)
    np.testing.assert_array_equal(bundle['initial'],initial)
    np.testing.assert_array_equal(bundle['reference_state'],truth)
    assert bundle['spec']['selected_update']==2 and bundle['geometry'].grid==(64,64)
    before=dict(bundle['provenance']['source_artifacts'])
    info,arrays=view.compute_prediction_data(bundle['model'],initial,truth,bundle['reference']['horizon'],
        bundle['equation'],bundle['geometry'])
    assert info['error_rms']==bundle['original_evaluation']['error_rms']
    assert info['error_max']==bundle['original_evaluation']['error_max']
    assert arrays['prediction'].shape==(1,1,64,64)
    assert all(view.file_digest(view.ROOT/path)==sha for path,sha in before.items())


def test_prediction_view_resolution_rejects_changed_checkpoint(resolution_replay):
    run,_,_,frozen,_=resolution_replay
    with (frozen/'checkpoints/selected.pt').open('ab') as output:
        output.write(b'changed')
    with pytest.raises(ValueError,match='inventory or bytes changed'):
        view.load_bundle(run,grid=64)


@pytest.fixture
def gallery(monkeypatch, tmp_path):
    import importlib.util
    filename=Path(__file__).resolve().parents[1]/'scripts/adjacent_resolution_images.py'
    spec=importlib.util.spec_from_file_location('tdn_test_resolution_images',filename)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(view,'ROOT',tmp_path)
    return module


def test_resolution_image_gallery_requires_completed_aggregate(resolution_replay, gallery):
    run,*_=resolution_replay
    with pytest.raises(FileNotFoundError):
        gallery.build_gallery_plan(run)


def test_resolution_image_gallery_declares_every_grid_track_before_errors(gallery, tmp_path, monkeypatch):
    run=tmp_path/'run';run.mkdir()
    protocol={'profile':'resolution-full','study':'adjacent-resolution','resolution':{
        'grids':[64,128],'tracks':['discrete','continuum'],'seeds':[871001,871011],
        'evaluation_horizons':[.04,.12]},'units':{'freeze':{'kind':'resolution_freeze','dependencies':[]}}}
    paths={'freeze':run/'freeze'}
    for grid in (64,128):
        for track in ('discrete','continuum'):
            name=f'evaluate-{grid}-{track}'
            protocol['units'][name]={'kind':'resolution_evaluate','dependencies':['freeze'],
                'grid':grid,'track':track,'field_indices':[5,0,1]}
            paths[name]=run/name
    protocol['units']['aggregate']={'kind':'resolution_aggregate','dependencies':list(protocol['units'])}
    paths['aggregate']=run/'aggregate'
    for name,path in paths.items():
        path.mkdir()
        for filename in ('science_manifest.json','workflow-seal.json','protocol.json'):
            (path/filename).write_text(name+filename)
    monkeypatch.setattr(view,'_coordinator',lambda _: (protocol,paths,[]))
    visited=[]
    def verify(_,path):
        name=path.name;visited.append(name)
        return {'source_tree_sha256':'fixture', 'prerequisites':{
            item:view.file_digest(paths[item]/'science_manifest.json')
            for item in protocol['units'][name]['dependencies']}}
    monkeypatch.setattr(view,'_verify_unit',verify)
    choices=[]
    def bundle(_,**choice):
        choices.append(choice)
        return {'reference':{'parent_id':'field-0'},'spec':{'model_id':str(choice),
            'checkpoint_sha256':'f'*64,'selection_status':'SELECTED_INITIALIZATION'},
            'provenance':{'source_artifacts':{}}}
    monkeypatch.setattr(view,'load_bundle',bundle)
    plan=gallery.build_gallery_plan(run)
    assert len(plan['selections'])==4 and set(visited)==set(paths)
    assert {(row['grid'],row['track']) for row in choices}=={
        (n,track) for n in (64,128) for track in ('discrete','continuum')}
    assert all(row['seed']==871001 and row['parent_index']==0 and row['horizon']==.04
        and row['family']=='channel_neural' and row['train_grid']==row['grid'] for row in choices)
    assert all(row['selection_status']=='SELECTED_INITIALIZATION' for row in plan['selections'])


def test_resolution_image_gallery_latest_uses_resolution_pointer(gallery, tmp_path, monkeypatch):
    from types import SimpleNamespace
    called=[]
    def resolve(value,**options):
        called.append((value,options));return tmp_path/'runs/new/adjacent-workflow.json'
    monkeypatch.setattr(gallery,'_workflow_api',lambda:SimpleNamespace(workflow_path=resolve,MANIFEST='adjacent-workflow.json'))
    assert gallery.resolve_run()==tmp_path/'runs/new'
    assert called==[('latest',{'resolution':True})]
    with pytest.raises(ValueError,match='inside this project'):
        gallery.resolve_run(str(tmp_path.parent/'external'))


def test_resolution_image_gallery_new_output_and_failure_retention(gallery, tmp_path, monkeypatch):
    run=tmp_path/'runs/scientific-run';run.mkdir(parents=True)
    original=run/'source.json';original.write_text('preserve scientific source')
    selection=dict(name='n64-discrete',family='channel_neural',track='discrete',grid=64,train_grid=64,
        seed=871001,parent_index=0,horizon=.04)
    plan=dict(run_dir=str(run),source_artifacts={str(original.relative_to(tmp_path)):view.file_digest(original)},
              selections=[selection])
    calls=[]
    def render(source,output,**options):
        calls.append((source,output,options));output.mkdir()
        (output/'manifest.json').write_text('fixture-render-proof')
        return dict(output_dir=str(output),images=[])
    monkeypatch.setattr(view,'render_prediction_view',render)
    first=gallery.export_gallery(plan);second=gallery.export_gallery(plan)
    assert first['output_dir']!=second['output_dir']
    assert first['status']=='COMPLETED' and all(call[2]['spatial_display']=='both' and call[2]['dpi']==400 for call in calls)
    assert all(Path(result['output_dir']).is_relative_to(tmp_path/'outputs') for result in (first,second))
    assert original.read_text()=='preserve scientific source'
    def fail(*args,**kwargs):
        raise RuntimeError('test fixture render interruption')
    monkeypatch.setattr(view,'render_prediction_view',fail)
    with pytest.raises(RuntimeError,match='render interruption'):
        gallery.export_gallery(plan)
    records=[json.loads(path.read_text()) for path in (tmp_path/'outputs').rglob('image-gallery.json')]
    failed=[row for row in records if row['status']=='FAILED']
    assert len(failed)==1 and failed[0]['partial_outputs_retained']
    assert original.read_text()=='preserve scientific source'
