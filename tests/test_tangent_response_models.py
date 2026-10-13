"""Causal response identities, full-field factors and cache contracts."""
from copy import deepcopy
import pytest
import torch

from tdn.analysis.response import numerics as n
from tdn.analysis.response.contracts import Physics
from tdn.numerics.types import Geometry
from tdn.analysis.tangent_response.models import create_model


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def case(target="fd_same_grid", size=5, device="cpu"):
    x = torch.arange(size, device=device)*2*torch.pi/size
    u = (.42+.04*torch.cos(x)[:, None]+.025*torch.sin(x)[None, :])[None, None].float()
    z = (.2+torch.cos(x)[:, None]*torch.sin(x)[None, :])[None, None].double()
    return u, z, Physics(.01, 4.), Geometry((size, size), (1., 1.))


def config(family="learned_response", **extra):
    norm = dict(full_mean=.4, full_scale=.1, mean_scale=.01, centered_scale=.01, floor=1e-8,
                response_gain=.1, mean_gain=.01, centered_gain=.01)
    return dict(family=family, width=4, depth=1, modes=1, rank=2, source_nodes=2, mean_nodes=2,
                normalization=norm, **extra)


@pytest.mark.parametrize("target", ("fd_same_grid", "galerkin_same_grid"))
def test_initial_response_is_matched_mean_and_exact_causal_identity(target):
    u, z, eq, geom = case(target)
    model = create_model(config())
    encoded = model.encode(u, eq, geom, target)
    torch.testing.assert_close(model.response_action(encoded, z, .02, .02), z, atol=0, rtol=0)
    torch.testing.assert_close(model.response_action(encoded, z, .01, .04), model.matched_action(encoded, z, .01, .04), atol=0, rtol=0)
    with pytest.raises(ValueError, match="Reversed"):
        model.response_action(encoded, z, .04, .01)
    assert encoded.alpha.abs().sum() > 0 and encoded.beta.abs().sum() > 0


@pytest.mark.parametrize("target", ("fd_same_grid", "galerkin_same_grid"))
def test_physical_operator_nulls_do_not_mask_zero_diffusion(target):
    u, z, eq, geom = case(target)
    model = create_model(config())
    with torch.no_grad(): model.coefficient_head.bias.fill_(.2)
    for state, physics in ((u, Physics(eq.kappa, 0.)), (torch.full_like(u, .4), eq)):
        encoded = model.encode(state, physics, geom, target)
        torch.testing.assert_close(model.response_action(encoded, z, .01, .04), model.matched_action(encoded, z, .01, .04), atol=0, rtol=0)
    encoded = model.encode(u, Physics(0., eq.reaction_rate), geom, target)
    assert (model.response_action(encoded, z, .01, .04)-model.matched_action(encoded, z, .01, .04)).abs().max() > 0


@pytest.mark.parametrize("target", ("fd_same_grid", "galerkin_same_grid"))
def test_response_translation_covariance_and_mean_content(target):
    u, z, eq, geom = case(target, size=6)
    model = create_model(config())
    with torch.no_grad(): model.coefficient_head.bias.fill_(-.3)
    original = model.response_action(model.encode(u, eq, geom, target), z, .005, .03)
    shift = lambda value:torch.roll(value, (1, -2), (-2, -1))
    translated = model.response_action(model.encode(shift(u), eq, geom, target), shift(z), .005, .03)
    torch.testing.assert_close(translated, shift(original), atol=3e-7, rtol=3e-6)
    assert original.is_floating_point() and n.mean(original).abs().max() > 0


def test_conditional_action_design_matches_signed_readout_exactly():
    u, z, eq, geom = case()
    model = create_model(config())
    encoded = model.encode(u, eq, geom, "fd_same_grid")
    base, design = model.affine_action_design(encoded, z, .005, .04)
    coefficients = torch.linspace(-.04, .03, design.shape[0], dtype=torch.float64)
    expected = base+torch.einsum("k,k...->...", coefficients, design)
    model.set_affine_coefficients(coefficients)
    actual = model.response_action(model.encode(u, eq, geom, "fd_same_grid"), z, .005, .04)
    torch.testing.assert_close(actual, expected, atol=2e-8, rtol=2e-7)


def test_cache_rejects_input_physics_checkpoint_and_dtype_changes():
    u, z, _, geom = case()
    eq = Physics(torch.tensor(.01), torch.tensor(4.))
    model = create_model(config())
    encoded = model.encode(u, eq, geom, "fd_same_grid")
    with torch.no_grad(): u.add_(.001)
    with pytest.raises(ValueError, match="state changed"): model.decode(encoded, .03)
    encoded = model.encode(u, eq, geom, "fd_same_grid")
    with torch.no_grad(): eq.reaction_rate.add_(1.)
    with pytest.raises(ValueError, match="physics changed"): model.decode(encoded, .03)
    encoded = model.encode(u, eq, geom, "fd_same_grid")
    model.load_state_dict(deepcopy(model.state_dict()))
    with pytest.raises(ValueError, match="checkpoint"): model.decode(encoded, .03)
    encoded = model.encode(u, eq, geom, "fd_same_grid"); model.double()
    with pytest.raises(ValueError, match="dtype"): model.decode(encoded, .03)


@pytest.mark.parametrize("family", ("learned_response", "residual_fno"))
def test_full_call_cached_parity_and_charged_precision_bridge(family):
    u, _, eq, geom = case()
    model = create_model(config(family))
    encoded = model.encode(u, eq, geom, "fd_same_grid")
    cached = model.decode(encoded, .03)
    torch.testing.assert_close(cached, model(u, .03, eq, geom, "fd_same_grid"), atol=0, rtol=0)
    parts = model.prediction_parts(model.encode(u, eq, geom, "fd_same_grid"), .03)
    assert parts["work"]["anchor_evaluations"] > 0
    assert parts["work"]["deployed_anchor_evaluations"] == 1
    torch.testing.assert_close(parts["smooth_anchor64"], parts["deployed_anchor32"].double()+parts["arithmetic_bridge"], atol=1e-15, rtol=0)
    assert model.architecture_metadata()["inference_teacher_inputs"] is False


def test_unsupported_closure_and_unimplemented_compact_path_are_explicit():
    u, _, eq, geom = case()
    model = create_model(config())
    with pytest.raises(ValueError, match="UNSUPPORTED_TARGET"): model(u, .03, eq, geom, "galerkin_projected_fine")
    with pytest.raises(ValueError, match="Compact"): create_model(config(compact_path=True))



@pytest.mark.parametrize("family", ("learned_response", "residual_fno"))
@pytest.mark.parametrize("target", ("fd_same_grid", "galerkin_same_grid"))
@pytest.mark.parametrize("assembly", ("fp64_bridge", "ordinary_fp32"))
def test_deployment_is_fp32_with_separate_fp64_attribution(family,target,assembly):
    u,_,eq,geom = case(target)
    model = create_model(config(family,assembly=assembly)).eval()
    with torch.no_grad():
        parts = model.prediction_parts(model.encode(u,eq,geom,target),.03)
    assert parts["output"].dtype == torch.float32 and torch.isfinite(parts["output"]).all()
    assert parts["ordinary_deployed"].dtype == parts["deployed_anchor32"].dtype == torch.float32
    for key in ("physical_prediction","physical_correction","smooth_anchor64","arithmetic_bridge","assembly_difference","final_rounding"):
        assert parts[key].dtype == torch.float64
    expected = parts["physical_prediction"].float() if assembly == "fp64_bridge" else parts["ordinary_deployed"]
    torch.testing.assert_close(parts["output"],expected,atol=0,rtol=0)
    torch.testing.assert_close(parts["final_rounding"],parts["output"].double()-parts["physical_prediction"],atol=0,rtol=0)
    torch.testing.assert_close(parts["output"].double()-parts["ordinary_deployed"].double(),
                               parts["assembly_difference"]+parts["final_rounding"],atol=0,rtol=0)


@pytest.mark.parametrize("family", ("learned_response", "residual_fno"))
@pytest.mark.parametrize("target", ("fd_same_grid", "galerkin_same_grid"))
@pytest.mark.parametrize("assembly", ("fp64_bridge", "ordinary_fp32"))
def test_zero_time_identity_preserves_state_jacobian_and_right_time_derivative(family,target,assembly):
    u,_,eq,geom = case(target,size=3)
    model = create_model(config(family,assembly=assembly))
    u = u.requires_grad_(); h = torch.tensor(0.,requires_grad=True)
    weights = torch.arange(1,u.numel()+1,dtype=u.dtype).reshape_as(u)/u.numel()
    output = model(u,h,eq,geom,target)
    torch.testing.assert_close(output,u,atol=0,rtol=0)
    state_grad,time_grad = torch.autograd.grad((output*weights).sum(),(u,h))
    torch.testing.assert_close(state_grad,weights,atol=2e-6,rtol=2e-6)
    # Zero-initialized endpoint readouts give the physical RHS at the right
    # endpoint; an exact-value branch must not erase this nonzero derivative.
    expected = (n.rhs(u.detach().double(),eq,geom,target)*weights.double()).sum().float()
    torch.testing.assert_close(time_grad,expected,atol=2e-6,rtol=2e-6)
    assert time_grad.abs()>0


@pytest.mark.parametrize("family", ("learned_response", "residual_fno"))
@pytest.mark.parametrize("target", ("fd_same_grid", "galerkin_same_grid"))
@pytest.mark.parametrize("steps", (2,4,8))
def test_actual_autonomous_macrosteps_reencode_finite_fp32_states(family,target,steps):
    u,_,eq,geom = case(target,size=3)
    model = create_model(config(family)).eval()
    state = u
    with torch.no_grad():
        for _ in range(steps):
            # The returned deployment field goes straight into the next full
            # call. No caller cast conceals an incompatible model output.
            state = model(state,.03/steps,eq,geom,target)
            assert state.dtype == torch.float32 and torch.isfinite(state).all()
    assert not torch.equal(state,u)
