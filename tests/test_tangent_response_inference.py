"""Complete-call adapters, source work, locked factors and same-path actions."""
import math

import pytest
import torch

from tdn.analysis.response import numerics as n
from tdn.analysis.response.contracts import Physics
from tdn.analysis.tangent_response.compression import fit_source_map
from tdn.analysis.tangent_response.inference import (
    NAMES, UnresolvedInference, make_methods, make_tangent_methods,
)
from tdn.analysis.tangent_response.models import create_model
from tdn.analysis.tangent_response.tangent import AnchorPath, response_action
from tdn.numerics.types import Geometry


def _fixture(target="fd_same_grid", *, reaction=1.):
    geom = Geometry((7,), (1.,)); eq = Physics(.004, reaction)
    x = torch.arange(7, dtype=torch.float64)/7
    u0 = (.4+.15*torch.cos(2*math.pi*x)).reshape(1, 1, 7).float()
    protocol = dict(profile="cpu-smoke", anchor=dict(name="df", reaction_substeps=8),
        mechanism=dict(initial_steps=4, max_steps=32, tolerance=1e-8, seconds_per_query=10),
        compression=dict(ranks=[2], source_nodes=2, t_ref=.12, temporal_channels=2),
        training=dict(source_nodes=2, mean_nodes=2, control_output_dtype="float32"))
    return u0, eq, geom, target, protocol


def _models():
    return {name: create_model(dict(family=name, ndim=1, width=4, depth=1, rank=2,
             temporal_channels=2, source_nodes=2, mean_nodes=2, modes=2, assembly="fp64_bridge",
             normalization=dict(mean_gain=1., centered_gain=1., response_gain=1.)))
            for name in ("learned_response", "residual_fno")}


def _factors(u0, eq, geom, target):
    path = AnchorPath(u0.double(), eq, geom, target)
    vectors, values, pairs = [], [], []
    for source in (torch.ones_like(u0).double(), n.center(u0.double())):
        for s, h in ((0., .01), (.003, .02), (.01, .04)):
            value = response_action(path, eq, geom, target, source, s, h, steps=16)-response_action(path, eq, geom, target, source, s, h, steps=16, mode="mean")
            vectors.append(source); values.append(value); pairs.append((s, h))
    factor = fit_source_map(torch.cat(vectors), torch.cat(values), geom, target, 2,
                            time_pairs=pairs, temporal_channels=2)
    key = (target, geom.grid, geom.lengths)
    return factor, [dict(identity=[list(key), "shared_response_action_pod_rank2"], context=list(key),
                        method="shared_response_action_pod_rank2", factors=factor.to_payload())]


@pytest.mark.parametrize("target", ["fd_same_grid", "galerkin_same_grid"])
def test_six_callbacks_are_complete_numerical_calls_without_dataset_labels(target):
    u0, eq, geom, target, protocol = _fixture(target)
    factor, records = _factors(u0, eq, geom, target)
    models = _models()
    # Poisonous labels cannot be read by a complete inference interface.
    samples = [dict(u0=u0, teacher64=object(), physical_defect=object(), action_labels=object())]
    methods = make_methods(samples, protocol, learned_models=models, private_factors=records)
    assert tuple(methods) == NAMES
    outputs = {name: method(u0, .02, eq, geom, target) for name, method in methods.items()}
    assert all(torch.isfinite(value).all() for value in outputs.values())
    assert outputs["anchor"].dtype == torch.float32
    assert outputs["G"].dtype == torch.float32
    assert outputs["fixed_shared_basis"].dtype == torch.float32
    assert outputs["U"].dtype == torch.float64
    assert torch.equal(outputs["G"], outputs["learned_response"].float())
    assert methods["G"].last_work["residual_evaluations"] >= 2
    assert methods["fixed_shared_basis"].last_work["factor_contractions"] == 2
    assert methods["U"].last_work["forced_solves"] >= 2
    assert methods["U"].metadata["deployable"] is False
    assert methods["learned_response"].last_parts["physical_prediction"].dtype == torch.float64
    assert not methods["fixed_shared_basis"].metadata["selection"]["diagnostic_labels_used"]
    descriptors = methods.performance_methods(deployable_only=True)
    assert "U" not in descriptors
    assert callable(descriptors["G"]["call"])


def test_missing_models_or_factors_are_unavailable_not_placeholder_predictions():
    u0, eq, geom, target, protocol = _fixture()
    methods = make_methods([], protocol)
    for name in ("fixed_shared_basis", "learned_response", "residual_fno"):
        assert methods[name].metadata["status"] == "UNAVAILABLE"
        with pytest.raises(ValueError, match="UNAVAILABLE"):
            methods[name](u0, .02, eq, geom, target)
    with pytest.raises(ValueError, match="FP32"):
        methods["G"](u0.double(), .02, eq, geom, target)
    with pytest.raises(ValueError, match="unsupported"):
        methods["anchor"](u0, .02, eq, geom, "galerkin_projected_fine")


def test_unresolved_oracle_does_not_return_unaccepted_last_iterate():
    u0, eq, geom, target, protocol = _fixture()
    protocol["mechanism"].update(initial_steps=1, max_steps=1)
    method = make_methods([], protocol)["U"]
    with pytest.raises(UnresolvedInference, match="not independently time resolved"):
        method(u0, .02, eq, geom, target)
    assert method.last_work["status"] == "UNRESOLVED"
    assert method.last_parts["oracle_metadata"]["accepted"] is False


def test_checkpoint_and_shared_selection_cannot_use_training_fit_or_diagnostics():
    u0, eq, geom, target, protocol = _fixture()
    model = _models()["learned_response"]
    with pytest.raises(ValueError, match="validation-selected"):
        make_methods([], protocol, learned_models={"learned_response":(model, {"checkpoint_role":"best training-fit diagnostic"})})
    with pytest.raises(ValueError, match="validation only"):
        make_methods([], protocol, compression_result={"shared_selection":dict(status="FROZEN", selection_scope="diagnostic_only", method="shared_response_action_pod_rank2", source_identity="wrong")})
    method = make_methods([], protocol, learned_models={"learned_response":(model, {"checkpoint_role":"validation-selected deployment", "seed":4})})["learned_response"]
    method(u0, .02, eq, geom, target)
    with torch.no_grad():
        model.coefficient_head.bias.add_(.1)
    with pytest.raises(ValueError, match="checkpoint/device/dtype changed"):
        method(u0, .02, eq, geom, target)


def test_state_oracle_factors_can_never_become_fixed_deployable_shared_method():
    u0, eq, geom, target, protocol = _fixture()
    factor, records = _factors(u0, eq, geom, target)
    factor.metadata.update(state_oracle=True, deployable=False)
    records[0]["factors"] = factor.to_payload()
    methods = make_methods([], protocol, private_factors=records)
    assert methods["fixed_shared_basis"].metadata["status"] == "UNAVAILABLE"


@pytest.mark.parametrize("null", ["time", "reaction", "constant"])
def test_complete_control_nulls_are_retained_separately_from_response_identities(null):
    u0, eq, geom, target, protocol = _fixture()
    _, factors = _factors(u0, eq, geom, target)
    if null == "reaction":
        eq = Physics(eq.kappa, 0.)
    if null == "constant":
        u0 = torch.full_like(u0, .4)
    h = 0. if null == "time" else .02
    methods = make_methods([], protocol, private_factors=factors, learned_models=_models())
    assert torch.equal(methods["G"](u0, h, eq, geom, target), methods["fixed_shared_basis"](u0, h, eq, geom, target))
    if null == "time":
        assert all(torch.equal(method(u0, h, eq, geom, target).float(), u0) for method in methods.values())


def test_same_path_actions_include_mean_signed_identity_and_complete_encoder_work():
    u0, eq, geom, target, protocol = _fixture()
    _, factors = _factors(u0, eq, geom, target)
    models = _models()
    methods = make_tangent_methods(u0, eq, geom, target, protocol, private_factors=factors, learned_models=models)
    probe = n.center(u0.double())
    assert methods.path.u0.dtype == torch.float64
    for name in ("G", "U", "fixed_shared_basis", "learned_response"):
        assert torch.equal(methods[name](probe, .01, .01), probe)
        positive = methods[name](probe, .002, .02)
        negative = methods[name](-probe, .002, .02)
        assert torch.allclose(positive, -negative, atol=2e-15)
    assert n.mean(methods["U"](probe, 0., .02)).abs().max() > 1e-6
    assert methods["learned_response"].last_work["encoder_calls"] == 1
    with pytest.raises(ValueError, match="Endpoint-only FNO"):
        methods["residual_fno"](probe, 0., .02)
    with pytest.raises(ValueError, match="Reversed"):
        methods["G"](probe, .03, .01)
    first = methods["U"](probe, 0., .01)
    composed = methods["U"](first, .01, .02)
    direct = methods["U"](probe, 0., .02)
    assert (composed-direct).abs().max() < 3e-8
