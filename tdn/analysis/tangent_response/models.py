"""State-conditioned full-field tangent response with charged explicit paths.

The learned operator corrects the matched anchor-mean response. No teacher,
action label, or future trajectory is accepted by an inference entry point.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import nn

from tdn.analysis.response import numerics as n
from tdn.analysis.response.features import FrozenNormalization, FEATURE_CHANNELS, build_features
from tdn.analysis.response.models import SpatialEncoder, create_model as response_model
from tdn.analysis.response.trajectory import mean_propagator
from tdn.analysis.roadmap.numerics import quadrature
from .tangent import AnchorPath, validate_target


def _version(value):
    try:
        return value._version
    except RuntimeError:
        return None


def _model_identity(model):
    return tuple((name, id(value), value.data_ptr(), _version(value), str(value.device), str(value.dtype))
                 for name, value in (*model.named_parameters(), *model.named_buffers()))


def _physics_identity(eq):
    return tuple((id(value), _version(value), str(value.device), str(value.dtype)) if torch.is_tensor(value)
                 else float(value) for value in (eq.kappa, eq.reaction_rate))


def _deployment_output(physical, ordinary, encoded, step, assembly):
    candidate = physical.to(encoded.u) if assembly == "fp64_bridge" else ordinary.to(encoded.u)
    # Preserve the literal state at zero without suppressing the computed
    # right time derivative or adding another copy of the state Jacobian.
    exact_initial = encoded.u.detach()+(candidate-candidate.detach())
    return torch.where(step == 0, exact_initial, candidate)


@dataclass
class Encoding:
    owner: int
    input_ref: torch.Tensor
    input_version: int | None
    input_snapshot: torch.Tensor
    u: torch.Tensor
    eq: object
    geom: object
    target: str
    model_identity: tuple
    physics_identity: tuple
    physics_snapshots: tuple
    path: AnchorPath
    latent: torch.Tensor | None = None
    pooled: torch.Tensor | None = None
    alpha: torch.Tensor | None = None
    beta: torch.Tensor | None = None
    coefficients: torch.Tensor | None = None
    gate: torch.Tensor | None = None
    residual_encoding: object = None
    geometry_signature: tuple = ()


class LearnedResponse(nn.Module):
    family = "learned_response"

    def __init__(self, config=None):
        super().__init__()
        self.config = dict(config or {})
        self.ndim = int(self.config.get("ndim", 2))
        self.width, self.depth = int(self.config.get("width", 16)), int(self.config.get("depth", 2))
        self.rank = int(self.config.get("rank", 4))
        self.temporal_channels = int(self.config.get("temporal_channels", 4))
        self.source_nodes = int(self.config.get("source_nodes", 8))
        self.mean_nodes = int(self.config.get("mean_nodes", 8))
        self.t_ref = float(self.config.get("t_ref", .12))
        self.anchor_name = self.config.get("anchor", "df")
        self.reaction_substeps = int(self.config.get("reaction_substeps", 8))
        self.assembly = self.config.get("assembly", "fp64_bridge")
        if self.ndim not in (1, 2) or min(self.width, self.depth, self.rank, self.temporal_channels, self.source_nodes, self.mean_nodes) <= 0:
            raise ValueError("Positive bounded spatial/time architecture dimensions are required")
        if self.rank not in (2, 4, 8) or self.temporal_channels > 6 or not math.isfinite(self.t_ref) or self.t_ref <= 0:
            raise ValueError("Declare rank 2/4/8, at most six temporal channels and a positive reference time")
        if self.assembly not in ("fp64_bridge", "ordinary_fp32"):
            raise ValueError("Assembly must be fp64_bridge or ordinary_fp32")
        if self.config.get("compact_path", False) or self.config.get("nonlinear_iterations", 0):
            raise ValueError("Compact path prediction and nonlinear learned iterations need separately implemented contracts")
        self.normalization = FrozenNormalization(self.config.get("normalization"))
        self.encoder = SpatialEncoder(FEATURE_CHANNELS, self.width, self.depth, int(self.config.get("modes", 4)), self.ndim)
        conv = nn.Conv1d if self.ndim == 1 else nn.Conv2d
        # Spatial factors are nonzero. Only the final corrective readout starts
        # at zero, so the initial model is exactly the matched mean response.
        self.alpha_factors = conv(self.width, self.rank, 1)
        self.beta_factors = conv(self.width, self.rank, 1)
        self.coefficient_head = nn.Linear(self.width, self.rank*self.temporal_channels)
        nn.init.zeros_(self.coefficient_head.weight); nn.init.zeros_(self.coefficient_head.bias)
        gain = float(self.config.get("normalization", {}).get("response_gain", 1.))
        if not math.isfinite(gain) or gain <= 0:
            raise ValueError("Frozen complete-response gain must be positive and finite")
        self.register_buffer("response_gain", torch.tensor(gain, dtype=torch.float64))

    def _check(self, encoded):
        if encoded.owner != id(self) or encoded.model_identity != _model_identity(self):
            raise ValueError("Encoding belongs to a different or changed checkpoint/device/dtype")
        if encoded.input_ref.device != encoded.u.device or encoded.input_ref.dtype != encoded.u.dtype:
            raise ValueError("Encoding input device/dtype changed")
        if (_version(encoded.input_ref) != encoded.input_version
            or encoded.input_version is None and not torch.equal(encoded.input_ref, encoded.input_snapshot)):
            raise ValueError("Encoding current state changed")
        if encoded.physics_identity != _physics_identity(encoded.eq):
            raise ValueError("Encoding physics changed")
        for value, snapshot in zip((encoded.eq.kappa, encoded.eq.reaction_rate), encoded.physics_snapshots):
            if torch.is_tensor(value) and not torch.equal(value.detach(), snapshot):
                raise ValueError("Encoding unversioned physics values changed")
        if tuple(encoded.u.shape[2:]) != tuple(encoded.geom.grid):
            raise ValueError("Encoding geometry changed")
        if encoded.geometry_signature != (tuple(encoded.geom.grid), tuple(encoded.geom.lengths)):
            raise ValueError("Encoding physical geometry changed")

    def encode(self, u0, eq, geom, target, *, budget=None):
        validate_target(target); n.validate(u0, 0., geom, target)
        parameter = next(self.parameters())
        if u0.device != parameter.device or u0.dtype != parameter.dtype:
            raise ValueError("State and network must have the same declared device/dtype")
        state = u0.clone()
        features, stats = build_features(state, eq, geom, target, self.normalization, t_ref=self.t_ref)
        latent = self.encoder(features)
        pooled = latent.mean(n.axes(latent))
        alpha, beta = self.alpha_factors(latent), self.beta_factors(latent)
        variance = stats["variance"]
        # Zero diffusion is deliberately absent: spatial reaction feedback
        # remains nonzero without diffusion on a varying anchor.
        gate = n.scalar(eq.reaction_rate, state)*self.t_ref*variance/(variance+self.normalization.floor.to(state).square())
        constant = state.amax(n.axes(state), keepdim=True) == state.amin(n.axes(state), keepdim=True)
        gate = torch.where(constant, torch.zeros_like(gate), gate)
        path = AnchorPath(state.double(), eq, geom, target, anchor_name=self.anchor_name,
            reaction_substeps=self.reaction_substeps, create_graph=torch.is_grad_enabled() and state.requires_grad, budget=budget)
        return Encoding(id(self), u0, _version(u0), u0.detach().clone(), state, eq, geom, target,
            _model_identity(self), _physics_identity(eq), tuple(value.detach().clone() if torch.is_tensor(value) else None
                for value in (eq.kappa, eq.reaction_rate)), path, latent, pooled, alpha, beta,
            self.coefficient_head(pooled).reshape(state.shape[0], self.rank, self.temporal_channels), gate,
            geometry_signature=(tuple(geom.grid), tuple(geom.lengths)))

    def temporal_basis(self, encoded, s, h):
        start, stop = n.validate(encoded.u, s, encoded.geom, encoded.target), n.validate(encoded.u, h, encoded.geom, encoded.target)
        if bool((stop < start).any()):
            raise ValueError("Reversed response intervals are invalid")
        delta, left, right = (stop-start)/self.t_ref, start/self.t_ref, stop/self.t_ref
        basis = (torch.ones_like(delta), left/(1+left), right/(1+right), delta/(1+delta),
                 torch.exp(-delta), delta/(1+delta).square())
        return torch.stack([value.expand(encoded.u.shape[0], 1, *([1]*self.ndim)) for value in basis[:self.temporal_channels]], dim=2), delta

    def matched_action(self, encoded, source, s, h):
        self._check(encoded); n.validate(source, 0., encoded.geom, encoded.target)
        start, stop = n.validate(source, s, encoded.geom, encoded.target), n.validate(source, h, encoded.geom, encoded.target)
        if bool((stop < start).any()):
            raise ValueError("Reversed response intervals are invalid")
        z = source.double()
        factor = mean_propagator(encoded.path.mean, start, stop, encoded.eq, z, nodes=self.mean_nodes)
        result = factor*n.heat(z, stop-start, encoded.eq, encoded.geom, encoded.target)
        # Exact value identity retains the computed time/physics derivative.
        exact_identity = result+(z-result).detach()
        return torch.where(stop == start, exact_identity, result)

    def action_basis(self, encoded, source, s, h):
        self._check(encoded)
        if source.shape != encoded.u.shape:
            raise ValueError("Response source must match the encoded full field")
        basis, delta = self.temporal_basis(encoded, s, h)
        beta = encoded.beta.double().reshape(-1, 1, *encoded.geom.grid)
        repeated = source.double().expand(-1, self.rank, *encoded.geom.grid).reshape_as(beta)
        projection = n.inner(beta, repeated, encoded.geom, encoded.target).reshape(encoded.u.shape[0], self.rank, *([1]*self.ndim))
        fields = encoded.alpha.double()*projection
        scalar = encoded.gate.double()*delta.double()/self.response_gain.double()
        return fields[:, :, None]*basis.double()*scalar[:, :, None]

    def response_action(self, encoded, source, s, h):
        base = self.matched_action(encoded, source, s, h)
        fields = self.action_basis(encoded, source, s, h)
        coefficients = encoded.coefficients.double().reshape(encoded.u.shape[0], self.rank, self.temporal_channels, *([1]*self.ndim))
        correction = (fields*coefficients).sum((1, 2))[:, None]
        result = base+correction
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("Nonfinite learned response; no output clipping")
        return result

    def affine_action_design(self, encoded, source, s, h):
        """Exact head design conditional on fixed spatial factors and path."""
        fields = self.action_basis(encoded, source, s, h).reshape(encoded.u.shape[0], self.rank*self.temporal_channels, *encoded.geom.grid)
        weighted = fields[:, :, None]*encoded.pooled.double().reshape(encoded.u.shape[0], 1, self.width, *([1]*self.ndim))
        weights = weighted.reshape(encoded.u.shape[0], -1, *encoded.geom.grid)
        complete = torch.cat((weights, fields), dim=1).movedim(1, 0)[:, :, None]
        return self.matched_action(encoded, source, s, h), complete

    def affine_endpoint_design(self, encoded, h):
        self._check(encoded)
        step = n.validate(encoded.u.double(), h, encoded.geom, encoded.target)
        count = self.rank*self.temporal_channels*(self.width+1)
        design = torch.zeros((count, *encoded.u.shape), dtype=torch.float64, device=encoded.u.device)
        base = encoded.path(step)
        for node, weight in zip(*quadrature(self.source_nodes)):
            source_time = step*node
            residual = encoded.path.evaluate(source_time)["residual"]
            mean, fields = self.affine_action_design(encoded, residual, source_time, step)
            base = base-step*weight*mean
            design = design-step*weight*fields
        return base, design

    def set_affine_coefficients(self, vector):
        value = torch.as_tensor(vector, dtype=self.coefficient_head.weight.dtype, device=self.coefficient_head.weight.device)
        count = self.coefficient_head.weight.numel()
        if value.numel() != count+self.coefficient_head.bias.numel() or not bool(torch.isfinite(value).all()):
            raise ValueError("Finite conditional readout vector has the wrong size")
        with torch.no_grad():
            self.coefficient_head.weight.copy_(value[:count].reshape_as(self.coefficient_head.weight))
            self.coefficient_head.bias.copy_(value[count:].reshape_as(self.coefficient_head.bias))

    def prediction_parts(self, encoded, h):
        self._check(encoded)
        step = n.validate(encoded.u.double(), h, encoded.geom, encoded.target)
        smooth = encoded.path(step)
        correction = torch.zeros_like(smooth)
        for node, weight in zip(*quadrature(self.source_nodes)):
            source_time = step*node
            residual = encoded.path.evaluate(source_time)["residual"]
            correction = correction-step*weight*self.response_action(encoded, residual, source_time, step)
        deployed = n.anchor(encoded.u, step.to(encoded.u), encoded.eq, encoded.geom, encoded.target,
                            name=self.anchor_name, reaction_substeps=self.reaction_substeps)
        physical = smooth+correction
        ordinary = deployed+correction.to(deployed)
        output = _deployment_output(physical, ordinary, encoded, step, self.assembly)
        return dict(output=output, physical_prediction=physical, ordinary_deployed=ordinary,
            smooth_anchor64=smooth, deployed_anchor32=deployed, physical_correction=correction,
            arithmetic_bridge=smooth-deployed.double(), assembly_difference=physical-ordinary.double(),
            final_rounding=output.double()-physical,
            diagnostics=dict(alpha_rms=encoded.alpha.double().square().mean().sqrt(),
                beta_rms=encoded.beta.double().square().mean().sqrt(),
                raw_readout_rms=encoded.coefficients.double().square().mean().sqrt(),
                complete_response_gain=self.response_gain,
                raw_units_scope="state-conditioned full-field factors and pre-temporal signed response coefficients"),
            work={**encoded.path.work(), "source_nodes":self.source_nodes, "mean_nodes":self.mean_nodes,
                  "deployed_anchor_evaluations":1, "response_action_calls":self.source_nodes})

    def decode(self, encoded, h):
        return self.prediction_parts(encoded, h)["output"]

    def forward(self, u0, h, eq, geom, target, *, budget=None):
        return self.decode(self.encode(u0, eq, geom, target, budget=budget), h)

    def step(self, u0, h, eq, geom, target):
        parts = self.prediction_parts(self.encode(u0, eq, geom, target), h)
        return parts["output"], parts["work"]

    def architecture_metadata(self):
        return dict(family=self.family, rank=self.rank, temporal_channels=self.temporal_channels,
            width=self.width, depth=self.depth, parameters=sum(value.numel() for value in self.parameters()),
            reusable_encoding=True, horizon_in_encoder=False, source_nodes=self.source_nodes, mean_nodes=self.mean_nodes,
            assembly=self.assembly, precision_bridge="FP64 smooth anchor and FP32 deployed anchor both charged",
            deployment_dtype="float32", physical_assembly_dtype="float64",
            final_rounding="deployed output minus FP64 physical assembly; ordinary assembly also includes its anchor/arithmetic difference",
            spatial_factors="full-field alpha/beta, including mean; signed time coefficients",
            normalization="positive train-frozen complete decoded gain; no per-example time/gate division",
            frozen_response_gain=float(self.response_gain),
            inference_teacher_inputs=False, dense_operator=False, compact_path=False,
            nulls="c(s,s)=0; reaction-zero and constant-anchor correction; diffusion-zero is not an operator null")


class MatchedResidualFNO(LearnedResponse):
    family = "residual_fno"

    def __init__(self, config=None):
        nn.Module.__init__(self)
        self.config = dict(config or {})
        self.ndim = int(self.config.get("ndim", 2)); self.rank = 0
        self.width, self.depth = int(self.config.get("width", 16)), int(self.config.get("depth", 2))
        self.anchor_name = self.config.get("anchor", "df")
        self.reaction_substeps = int(self.config.get("reaction_substeps", 8))
        self.assembly = self.config.get("assembly", "fp64_bridge")
        if self.assembly not in ("fp64_bridge", "ordinary_fp32"):
            raise ValueError("Declare a physical-bridge or ordinary FP32 assembly")
        self.backbone = response_model({**self.config, "family":"residual_fno", "assembly":"ordinary",
                                       "gain_normalization":self.config.get("gain_normalization", "frozen")})

    def encode(self, u0, eq, geom, target, *, budget=None):
        validate_target(target); n.validate(u0, 0., geom, target)
        if u0.dtype != next(self.parameters()).dtype or u0.device != next(self.parameters()).device:
            raise ValueError("State/network device and dtype must match")
        state = u0.clone()
        path = AnchorPath(state.double(), eq, geom, target, anchor_name=self.anchor_name,
            reaction_substeps=self.reaction_substeps, create_graph=torch.is_grad_enabled() and state.requires_grad, budget=budget)
        result = Encoding(id(self), u0, _version(u0), u0.detach().clone(), state, eq, geom, target,
            _model_identity(self), _physics_identity(eq), tuple(value.detach().clone() if torch.is_tensor(value) else None
                for value in (eq.kappa, eq.reaction_rate)), path)
        result.residual_encoding = self.backbone.encode(state, eq, geom, target)
        result.geometry_signature = (tuple(geom.grid), tuple(geom.lengths))
        return result

    def prediction_parts(self, encoded, h):
        self._check(encoded)
        step = n.validate(encoded.u, h, encoded.geom, encoded.target)
        parts = self.backbone._endpoint_parts(encoded.residual_encoding, step)
        correction = parts["physical"]+parts["numerical_remainder"]
        smooth, deployed = encoded.path(step), parts["base"]
        physical = smooth+correction.double(); ordinary = deployed+correction
        output = _deployment_output(physical, ordinary, encoded, step, self.assembly)
        return dict(output=output, physical_prediction=physical,
            ordinary_deployed=ordinary, smooth_anchor64=smooth, deployed_anchor32=deployed,
            physical_correction=correction.double(), arithmetic_bridge=smooth-deployed.double(),
            assembly_difference=physical-ordinary.double(),
            final_rounding=output.double()-physical,
            diagnostics=dict(raw_mean_rms=parts["raw_mean"].double().square().mean().sqrt(),
                raw_centered_rms=parts["raw_centered"].double().square().mean().sqrt(),
                raw_units_scope="residual FNO endpoint mean/centered readout before physical gain assembly"),
            work={**encoded.path.work(), "deployed_anchor_evaluations":1, "source_nodes":0, "response_action_calls":0})

    def response_action(self, *args, **kwargs):
        raise ValueError("Endpoint-only residual FNO has no learned response-action supervision contract")

    def affine_endpoint_design(self, encoded, h):
        self._check(encoded)
        _, design = self.backbone.affine_features(encoded.u, h, encoded.eq, encoded.geom, encoded.target)
        return encoded.path(h), design.double()

    def set_affine_coefficients(self, vector):
        self.backbone.set_affine_coefficients(vector)

    def architecture_metadata(self):
        return {**self.backbone.architecture_metadata(), "family":self.family, "rank":0,
            "temporal_channels":0, "parameters":sum(value.numel() for value in self.parameters()),
            "assembly":self.assembly, "reusable_encoding":False,
            "deployment_dtype":"float32", "physical_assembly_dtype":"float64",
            "final_rounding":"deployed output minus FP64 physical assembly; ordinary assembly also includes its anchor/arithmetic difference",
            "source_nodes":0, "mean_nodes":0, "inference_teacher_inputs":False,
            "precision_bridge":"Both smooth FP64 and deployed FP32 anchor computations charged",
            "supervision":"endpoint only; not matched extra action supervision"}


def create_model(config=None):
    config = dict(config or {})
    family = config.get("family", config.get("arm", "learned_response"))
    if family == "learned_response":
        return LearnedResponse(config)
    if family == "residual_fno":
        return MatchedResidualFNO(config)
    raise ValueError("Only learned_response and residual_fno are trainable arms; analytic/oracle controls live in the numerical modules")

