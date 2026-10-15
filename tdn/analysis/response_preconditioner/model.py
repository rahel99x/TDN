"""First affine, differentiable response-preconditioner implementation.

This fixed-feature prototype is an implementation diagnostic, not trained
science, an accepted adaptive solver, or a claim of superiority. Its reference
is docs/TDN_ARCHITECTURE_PROPOSAL_20261015.md. Only periodic same-grid FD is
supported. One linear defect correction is implemented; nonlinear second-stage
feedback and a learned encoder remain gated research work.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from tdn.analysis.response import numerics as n
from tdn.analysis.response.contracts import Physics
from tdn.analysis.roadmap.numerics import quadrature
from tdn.analysis.tangent_response.tangent import AnchorPath
from .numerics import commutator_actions, ordered_response, two_sided_heat

FEATURE_NAMES = (
    "constant", "mean_state", "reaction_time_reference", "diffusion_grid_time_reference",
    "start_time", "lag", "state_variance", "reaction_lag",
)
BRANCH_NAMES = ("LLV", "VLV", "LdotV", "two_sided_heat_1", "two_sided_heat_2")


def _count(work, key, amount=1):
    if work is not None:
        work[key] = work.get(key, 0) + amount


def _finite(value, label):
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} must be finite")
    return value


def _physics(eq, like):
    """Preserve differentiable tensor physics while checking its batch shape."""
    kap = _finite(n.scalar(eq.kappa, like), "diffusion")
    rate = _finite(n.scalar(eq.reaction_rate, like), "reaction")
    return Physics(kap, rate)


def _path_work(work, path):
    if work is not None:
        for key, value in path.work().items():
            if isinstance(value, int):
                _count(work, "path_" + key, value)


class FixedFeatureResponse(nn.Module):
    r"""A zero-initialized readout on five fixed response-action branches.

    Let v=u-mean(u), and Nv be the arithmetic average of the 2d nearest
    periodic neighbors (no grid derivative). The nonzero frozen factors are
    b1=exp(tanh(v)/4), c1=exp(-tanh(Nv)/4), b2=exp(tanh(Nv)/4),
    c2=exp(tanh(v)/4). They are input-dependent and are never detached.

    Eight bounded dimensionless features eta are [1, tanh(mean u), tanh(r*T),
    tanh(kappa*T*sum(dx^-2)), tanh(s/T), tanh(tau/T), tanh(mean(v^2)),
    tanh(r*tau)]. The positive reference time T is fixed model metadata.
    Each of the three tau^3 commutator actions and two (tau/T)^3 heat actions
    receives its own affine eta readout: 5*8=40 unconstrained signed weights.
    Features are bounded; finite fitted weights need not be uniformly small.

    The backbone is ordered_response with two panels by default. The endpoint
    is a(h)-h*sum_GL3 weight*K(h,s)*rho(s), exactly affine in all 40 weights.
    The default returned field is FP64; output_precision='input' explicitly
    rounds once to the input dtype. Design tensors always remain FP64 so
    fitting does not claim affinity of rounded deployment.

    Every public call builds a fresh graph-capable AnchorPath. There is no
    state cache, no future truth input, and no implicit nonlinear second stage.
    The complete forward derivative includes path, factors and coefficients.
    """

    coefficient_count = len(BRANCH_NAMES) * len(FEATURE_NAMES)

    def __init__(self, *, time_reference=1., panels=2, potential_nodes=2,
                 output_precision="float64"):
        super().__init__()
        if not math.isfinite(float(time_reference)) or float(time_reference) <= 0:
            raise ValueError("time_reference must be finite and positive")
        if type(panels) is not int or panels not in (1, 2, 4, 8):
            raise ValueError("panels must be one of 1, 2, 4, 8")
        if type(potential_nodes) is not int or potential_nodes not in (1, 2, 3, 4):
            raise ValueError("potential_nodes must be one of 1, 2, 3, 4")
        if output_precision not in ("float64", "input"):
            raise ValueError("output_precision must be float64 or input")
        self.time_reference = float(time_reference)
        self.panels, self.potential_nodes = panels, potential_nodes
        self.output_precision = output_precision
        self.readout = nn.Parameter(torch.zeros(self.coefficient_count, dtype=torch.float64))

    def architecture_metadata(self):
        return dict(schema="tdn.response_preconditioner.fixed_features.v1",
                    target="fd_same_grid", anchor="df", source_nodes=3,
                    response_panels=self.panels, potential_nodes=self.potential_nodes,
                    time_reference=self.time_reference, output_precision=self.output_precision,
                    internal_precision="float64", coefficient_count=self.coefficient_count,
                    feature_names=list(FEATURE_NAMES), branch_names=list(BRANCH_NAMES),
                    learned_encoder=False, correction_stages=1, cache=False,
                    work_counter_scope=("heat_actions/fft_transforms/laplacian_actions count explicit response "
                                        "branches only; path and feature calls are separately counted; "
                                        "path/autograd internal transforms are not instrumented; "
                                        "complete-call timings must include all work"),
                    inference_teacher_inputs=False, status="IMPLEMENTATION_ONLY")

    def set_readout(self, coefficients):
        value = torch.as_tensor(coefficients, dtype=torch.float64, device=self.readout.device)
        if value.shape != self.readout.shape:
            raise ValueError(f"Expected {self.coefficient_count} affine coefficients")
        _finite(value, "readout")
        with torch.no_grad():
            self.readout.copy_(value)
        return self

    def _prepare(self, u, eq, geom, target):
        if target != "fd_same_grid":
            raise ValueError("UNSUPPORTED_TARGET: response preconditioner requires fd_same_grid")
        n.validate(u, 0., geom, target)
        _finite(u, "initial state")
        if u.shape[0] < 1:
            raise ValueError("At least one initial state is required")
        if self.readout.dtype != torch.float64:
            raise ValueError("Readout must retain float64; do not call model.float()")
        if self.readout.device != u.device:
            raise ValueError("Model and input must use the same device")
        _finite(self.readout, "readout")
        initial = u.to(dtype=torch.float64)
        physics = _physics(eq, initial)
        path = AnchorPath(initial, physics, geom, target, create_graph=True)
        return initial, physics, path

    def _encode(self, initial, work):
        """Compute state-only factors once within this call's live graph."""
        _count(work, "fixed_feature_encodings")
        v = n.center(initial)
        neighbor = torch.zeros_like(v)
        for axis in n.axes(v):
            neighbor = neighbor + torch.roll(v, 1, axis) + torch.roll(v, -1, axis)
        neighbor = neighbor / (2 * (initial.ndim - 2))
        _count(work, "feature_neighbor_rolls", 2 * (initial.ndim - 2))
        factors = ((torch.exp(.25 * torch.tanh(v)), torch.exp(-.25 * torch.tanh(neighbor))),
                   (torch.exp(.25 * torch.tanh(neighbor)), torch.exp(.25 * torch.tanh(v))))
        return factors, _finite(n.mean(initial), "state mean"), _finite(n.mean(v.square()), "state variance")

    def _features(self, encoded, start, stop, eq, geom, work):
        _count(work, "temporal_feature_calls")
        factors, mu, variance = encoded
        tau, tref = stop - start, self.time_reference
        # Expand known scalar/per-parent physics to per-parent summary shape.
        one = torch.ones_like(mu)
        raw = (one, mu, one * eq.reaction_rate * tref,
               one * eq.kappa * tref * sum(dx ** -2 for dx in geom.dx),
               one * start / tref, one * tau / tref, variance,
               one * eq.reaction_rate * tau)
        eta = torch.stack((one, *(torch.tanh(x) for x in raw[1:])))
        return factors, eta

    def _action_design(self, initial, path, encoded, z, start, stop, eq, geom, target, work):
        n.validate(z, 0., geom, target)
        if z.shape != initial.shape or z.device != initial.device:
            raise ValueError("Source must match initial-state shape and device")
        z = _finite(z.to(dtype=torch.float64), "source")
        s, t = n.validate(initial, start, geom, target), n.validate(initial, stop, geom, target)
        if bool((t < s).any()):
            raise ValueError("Reversed response interval")
        tau = t - s
        factors, eta = self._features(encoded, s, t, eq, geom, work)
        base = ordered_response(path, z, s, t, eq, geom, panels=self.panels,
                                potential_nodes=self.potential_nodes, work=work, target=target)
        commutators = commutator_actions(path, z, s, eq, geom, work=work, target=target)
        actions = [tau ** 3 * value for value in commutators]
        actions += [(tau / self.time_reference) ** 3 * two_sided_heat(
            z, b, c, tau, eq, geom, work=work, target=target) for b, c in factors]
        # [5,8,B,1,*grid] -> branch-major [40,B,1,*grid].
        design = (torch.stack(actions).unsqueeze(1) * eta.unsqueeze(0)).flatten(0, 1)
        _count(work, "affine_action_designs")
        return _finite(base, "backbone response"), _finite(design, "action design")

    def action_design(self, u, z, start, stop, eq, geom, *, work=None, target="fd_same_grid"):
        initial, physics, path = self._prepare(u, eq, geom, target)
        encoded = self._encode(initial, work)
        result = self._action_design(initial, path, encoded, z, start, stop, physics, geom, target, work)
        _path_work(work, path)
        return result

    def endpoint_design(self, u, h, eq, geom, *, work=None, target="fd_same_grid"):
        initial, physics, path = self._prepare(u, eq, geom, target)
        encoded = self._encode(initial, work)
        step = n.validate(initial, h, geom, target)
        base = path(step)
        design = torch.zeros((self.coefficient_count, *initial.shape), dtype=torch.float64,
                             device=initial.device)
        for x, weight in zip(*quadrature(3)):
            start = step * x
            rho = path.evaluate(start)["residual"]
            response, columns = self._action_design(initial, path, encoded, rho, start, step,
                                                     physics, geom, target, work)
            base = base - step * weight * response
            design = design - (step * weight).unsqueeze(0) * columns
            _count(work, "source_quadrature_nodes")
        _path_work(work, path)
        _count(work, "endpoint_designs")
        return _finite(base, "classical corrected endpoint"), _finite(design, "endpoint design")

    def _apply_readout(self, base, design, input_dtype, work):
        result = base + torch.einsum("p,p...->...", self.readout, design)
        _finite(result, "response preconditioner output")
        _count(work, "readout_assemblies")
        if self.output_precision == "input":
            _count(work, "output_precision_casts")
            result = result.to(dtype=input_dtype)
            _finite(result, "rounded output")
        return result

    def response_action(self, u, z, start, stop, eq, geom, *, work=None, target="fd_same_grid"):
        base, design = self.action_design(u, z, start, stop, eq, geom, work=work, target=target)
        return self._apply_readout(base, design, z.dtype, work)

    def forward(self, u, h, eq, geom, *, work=None, target="fd_same_grid"):
        base, design = self.endpoint_design(u, h, eq, geom, work=work, target=target)
        return self._apply_readout(base, design, u.dtype, work)
