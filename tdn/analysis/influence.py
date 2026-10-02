"""Matrix-free discrete derivative and same-observation hidden-input audits."""
from __future__ import annotations

import torch

from tdn.features.local import extract_features
from tdn.numerics import Equation, Geometry, choose_substeps, reference_step, split_step, weighted_norm


def derivative_audit(u: torch.Tensor, equation: Equation, geometry: Geometry, h: float, *,
                     substeps: int, seed: int = 0) -> dict:
    """JVP/VJP adjoint identity plus a logarithmic central FD sweep."""
    if u.dtype != torch.float64:
        raise ValueError("Derivative audit requires FP64")
    generator = torch.Generator(device=u.device).manual_seed(seed)
    v = torch.randn(u.shape, dtype=u.dtype, device=u.device, generator=generator)
    w = torch.randn(u.shape, dtype=u.dtype, device=u.device, generator=generator)
    v = v / torch.linalg.vector_norm(v)
    w = w / torch.linalg.vector_norm(w)
    def full(state):
        return reference_step(state, h, equation, geometry, substeps=substeps)
    def defect(state):
        return full(state) - split_step(state, h, equation, geometry, differentiable=True)
    reports = {}
    for label, function in (("coupled_full_map", full), ("splitting_defect", defect)):
        _, jv = torch.autograd.functional.jvp(function, u, v, strict=True)
        _, jtw = torch.autograd.functional.vjp(function, u, w, strict=True)
        left = torch.sum(w * jv).item()
        right = torch.sum(v * jtw).item()
        jvp_norm = torch.linalg.vector_norm(jv).item()
        dot_absolute_error = abs(left - right)
        dot_error = abs(left - right) / max(abs(left), abs(right), 1e-14)
        sweep = []
        for epsilon in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
            fd = (function(u + epsilon * v) - function(u - epsilon * v)) / (2 * epsilon)
            absolute = torch.linalg.vector_norm(fd - jv).item()
            relative = absolute / max(jvp_norm, 1e-14)
            sweep.append({"epsilon": epsilon, "jvp_relative_error": relative, "jvp_absolute_error": absolute})
        reports[label] = {"dot_w_Jv": left, "dot_v_Jtranspose_w": right,
                          "adjoint_relative_error": dot_error, "adjoint_absolute_error": dot_absolute_error,
                          "jvp_l2_norm": jvp_norm, "relative_audit_roundoff_dominated": jvp_norm < 1e-9,
                          "finite_difference_sweep": sweep,
                          "pass_rule": "dot relative <1e-8 or absolute <1e-12; FD relative <1e-5 or absolute <1e-10",
                          "passed": (dot_error < 1e-8 or dot_absolute_error < 1e-12) and
                          (min(point["jvp_relative_error"] for point in sweep) < 1e-5 or
                           min(point["jvp_absolute_error"] for point in sweep) < 1e-10)}
    return {"scope": "FP64 tiny-grid matrix-free fixed-substep sensitivity; no uniform nonlinear guarantee",
            "fixed_substeps": substeps, "passed": all(x["passed"] for x in reports.values()), "maps": reports}


def same_observation_audit(u: torch.Tensor, equation: Equation, geometry: Geometry, h: float, *,
                           substeps: int, t_ref: float = 1., U_ref: float = 1.,
                           radius: int = 1, amplitude: float = .1) -> dict:
    """Exact equal-feature pair at one receiver with remote physical perturbation.

    The split FFT and coupled teacher are global single-domain computations. Only
    the pointwise neural observation has radius one; no finite diffusion cone or
    halo benefit follows from this audit.
    """
    if u.shape[0] != 1 or u.shape[1] != 1 or u.dtype != torch.float64:
        raise ValueError("Same-observation audit requires one scalar FP64 parent")
    if min(geometry.grid) < 2 * radius + 4:
        raise ValueError("Grid too small for an observation-invisible perturbation")
    target = tuple(n // 2 for n in geometry.grid)
    hidden = list(target)
    hidden[0] = (hidden[0] + radius + 2) % geometry.grid[0]
    hidden = tuple(hidden)
    perturbation = torch.zeros_like(u)
    perturbation[(0, 0, *hidden)] = amplitude
    plus, minus = u + perturbation, u - perturbation
    if plus.max().item() > 1 or minus.min().item() < 0:
        raise ValueError("Hidden pair left the validated logistic state interval")
    features_plus = extract_features(plus, equation, geometry, t_ref=t_ref, U_ref=U_ref)[(0, *target)]
    features_minus = extract_features(minus, equation, geometry, t_ref=t_ref, U_ref=U_ref)[(0, *target)]
    exact_equality = torch.equal(features_plus, features_minus)
    if not exact_equality:
        raise RuntimeError("Observation equality failed: feature support must be audited before using the bound")
    teacher_plus = reference_step(plus, h, equation, geometry, substeps=substeps)
    teacher_minus = reference_step(minus, h, equation, geometry, substeps=substeps)
    split_plus = split_step(plus, h, equation, geometry, differentiable=True)
    split_minus = split_step(minus, h, equation, geometry, differentiable=True)
    receiver = (0, 0, *target)
    full_difference = (teacher_plus - teacher_minus)[receiver].abs().item()
    defect_difference = ((teacher_plus - split_plus) - (teacher_minus - split_minus))[receiver].abs().item()
    # Individual hidden directions probe sensitivity tails, without a dense
    # Jacobian or an unsupported claim that a few probes give the operator norm.
    state = u.clone().requires_grad_(True)
    full_scalar = reference_step(state, h, equation, geometry, substeps=substeps)[receiver]
    defect_scalar = full_scalar - split_step(state, h, equation, geometry, differentiable=True)[receiver]
    full_gradient = torch.autograd.grad(full_scalar, state, retain_graph=True)[0]
    defect_gradient = torch.autograd.grad(defect_scalar, state)[0]
    mask = torch.ones_like(state, dtype=torch.bool)
    for dimension in range(geometry.ndim):
        for offset in range(-radius, radius + 1):
            index = list(target)
            index[dimension] = (index[dimension] + offset) % geometry.grid[dimension]
            mask[(0, 0, *index)] = False
    # Radius-one features use axial neighbors. Marking additional sites invisible
    # would be incorrect if feature construction later gained diagonal support.
    hidden_full_norm = torch.linalg.vector_norm(full_gradient[mask]).item()
    hidden_defect_norm = torch.linalg.vector_norm(defect_gradient[mask]).item()
    return {"scope": "single-device global periodic FFT prototype; local neural observation only",
            "target_index": list(target), "hidden_index": list(hidden), "feature_radius": radius,
            "feature_equality_exact": exact_equality,
            "feature_difference_max": (features_plus - features_minus).abs().max().item(),
            "hidden_pair_amplitude": amplitude, "horizon": h, "fixed_teacher_substeps": substeps,
            "full_map_receiver_difference": full_difference,
            "defect_receiver_difference": defect_difference,
            "deterministic_local_defect_worst_case_error_lower_bound": defect_difference / 2,
            "hidden_linear_full_receiver_l2_sensitivity": hidden_full_norm,
            "hidden_linear_defect_receiver_l2_sensitivity": hidden_defect_norm,
            "hidden_defect_to_full_sensitivity_ratio": hidden_defect_norm / max(hidden_full_norm, 1e-14),
            "bound_status": "exact same-feature nonlinear pair bound; matrix-free linear sensitivity is separate",
            "locality_certified": False,
            "blocker": "No application-specific permitted local prediction error is declared; no universal radius sufficiency claim"}
