"""Auditable pictures of one frozen adjacent-study prediction.

This is CPU postprocessing, not new training or deployment timing. Input,
checkpoint, reference and original evaluation errors must come from verified
immutable artifacts. The reference is used only for the last error panels.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import platform
from pathlib import Path
import re
import subprocess

import numpy as np
import torch

from tdn.analysis.roadmap.numerics import lowpass
from tdn.numerics import Equation, Geometry
from tdn.research.protocol import digest, file_digest
from .fields import FieldParent, sample_field
from .models import CHANNEL_NAMES, gain_bounds, make_model

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "tdn.adjacent-prediction-view/v1"
PROTECTED_IMPLEMENTATION = (
    "tdn/analysis/adjacent/models.py", "tdn/analysis/adjacent/fields.py",
    "tdn/analysis/adjacent/pilots.py", "tdn/analysis/portfolio/models.py",
    "tdn/analysis/frontier/models.py", "tdn/analysis/frontier/numerics.py",
    "tdn/analysis/roadmap/numerics.py", "tdn/analysis/agenda/physics.py",
    "tdn/numerics/operators.py", "tdn/numerics/subflows.py", "tdn/numerics/types.py",
)


def _contained(path):
    path = Path(path)
    path = path if path.is_absolute() else ROOT / path
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Prediction-view inputs and outputs must remain inside this project")
    for item in (path, *path.parents):
        if item.is_symlink():
            raise ValueError("Prediction-view paths must not contain symlinks")
        if item == ROOT:
            break
    return path.resolve()


def _read(path):
    return json.loads(Path(path).read_text())


def _array_digest(array):
    value = np.ascontiguousarray(array)
    return dict(dtype=str(value.dtype), shape=list(value.shape),
                sha256=hashlib.sha256(value.tobytes()).hexdigest())


def _implementation_identity(revision):
    if not isinstance(revision, str) or not re.fullmatch("[0-9a-f]{40}", revision):
        raise ValueError("A complete frozen source commit is required for prediction replay")
    identity = {}
    for relative in PROTECTED_IMPLEMENTATION:
        result = subprocess.run(["git", "show", f"{revision}:{relative}"], cwd=ROOT,
                                check=False, capture_output=True)
        if result.returncode:
            raise ValueError(f"Frozen implementation revision is unavailable: {revision}:{relative}")
        original = hashlib.sha256(result.stdout).hexdigest()
        if file_digest(ROOT / relative) != original:
            raise ValueError(f"Prediction implementation differs from the frozen experiment: {relative}")
        identity[relative] = original
    return identity


def _execution_protocol(protocol, pilot_unit):
    result = copy.deepcopy(protocol)
    result["pilot"]["tracks"] = protocol["units"][pilot_unit]["tracks"]
    result["pilot"]["max_seconds"] = protocol["units"][pilot_unit]["seconds"] - 15
    followers = [name for name, spec in protocol["units"].items() if spec.get("pilot_unit") == pilot_unit]
    if len(followers) != 1:
        raise ValueError("Exactly one frozen evaluation unit must follow the selected pilot")
    result["pilot"]["evaluation_seconds"] = protocol["units"][followers[0]]["seconds"] - 15
    return result, followers[0]


def _verify_unit(protocol, path):
    from .engine import verify_science
    manifest = verify_science(protocol, path)
    seal = _read(path / "workflow-seal.json")
    names = {"execution.json", "protocol.json", "stage.json", "science_manifest.json"}
    if seal.get("schema") != protocol["schema"] or seal.get("protocol_sha256") != digest(protocol) or set(seal.get("files", {})) != names:
        raise ValueError("Prediction source execution seal has changed scope")
    for name in names:
        if file_digest(path / name) != seal["files"][name]:
            raise ValueError("Prediction source execution evidence changed")
    execution, stage = _read(path / "execution.json"), _read(path / "stage.json")
    if stage.get("status") != "COMPLETED" or execution.get("stage") != manifest["stage"]:
        raise ValueError("Prediction source did not complete under its declared execution")
    if execution.get("software", {}).get("source_tree_sha256") != manifest["source_tree_sha256"]:
        raise ValueError("Prediction source scientific and execution identities disagree")
    return manifest


def _validate_bank_arrays(bank, arrays, row):
    parent = FieldParent.from_dict(row["parent"])
    declared = bank["binding"]["parents"][row["parent_index"]]
    if declared != row["parent"]:
        raise ValueError("Reference row and frozen parent declaration disagree")
    initial, truth = arrays[row["state_key"]], arrays[row["key"]]
    n = int(row["grid"])
    if initial.dtype != np.float64 or truth.dtype != np.float64 or initial.shape != (1, 1, n, n) or truth.shape != initial.shape:
        raise ValueError("Expected matching FP64 field and reference arrays")
    if not np.isfinite(initial).all() or not np.isfinite(truth).all():
        raise ValueError("Prediction source field and reference must be finite")
    regenerated = sample_field(parent, n).cpu().numpy()
    if not np.allclose(regenerated, initial, atol=2e-14, rtol=0):
        raise ValueError("Stored input does not match its continuous parent coefficients")
    return parent, initial, truth


def load_bundle(run_dir, *, family="channel_neural", track="discrete", parent_index=0,
                horizon=None, seed=None):
    """Read and verify source artifacts without modifying any historical file."""
    if family not in ("channel_neural", "channel_global", "channel_affine", "channel_fixed"):
        raise ValueError("Prediction view currently supports physical interaction-channel families")
    run = _contained(run_dir)
    protocols = [p for p in (run / "local-protocol.json", run / "protocol.json") if p.is_file()]
    if len(protocols) != 1:
        raise ValueError("Run must contain exactly one recognized frozen coordinator protocol")
    protocol = _read(protocols[0])
    candidates = [name for name, spec in protocol["units"].items()
                  if spec["kind"] == "train" and track in spec["tracks"]]
    if len(candidates) != 1:
        raise ValueError("Exactly one pilot unit must provide the requested target track")
    pilot_unit = candidates[0]
    execution, evaluation_unit = _execution_protocol(protocol, pilot_unit)
    pilot, evaluation = run / pilot_unit, run / evaluation_unit
    pilot_manifest = _verify_unit(protocol, pilot)
    evaluation_manifest = _verify_unit(protocol, evaluation)
    if evaluation_manifest["prerequisites"].get(pilot_unit) != file_digest(pilot / "science_manifest.json"):
        raise ValueError("Evaluation does not descend from the selected frozen pilot")
    from .pilots import verify_freeze, settings
    frozen = verify_freeze(execution, pilot)
    source_files = _implementation_identity(pilot_manifest["software"]["git_commit"])
    catalog = _read(pilot / "catalog.json")
    selected = [item for item in catalog if item["family"] == family and item["track"] == track
                and (seed is None or item["seed"] == seed)]
    if len(selected) != 1:
        raise ValueError("Choose an explicit seed when multiple frozen models match")
    spec = selected[0]
    checkpoint = _contained(pilot / spec["checkpoint"])
    if not checkpoint.is_relative_to(pilot) or file_digest(checkpoint) != spec["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint differs from its frozen catalog")
    bank = _read(evaluation / "reference-bank.json")
    data_path = evaluation / "reference-bank.npz"
    if file_digest(data_path) != bank["data_sha256"]:
        raise ValueError("Frozen reference-bank data changed")
    matching = [row for row in bank["references"] if row["track"] == track and row["parent_index"] == parent_index
                and (horizon is None or math.isclose(row["horizon"], horizon, rel_tol=0, abs_tol=1e-12))]
    if len(matching) != 1:
        raise ValueError("Choose a unique parent index and horizon from the frozen evaluation bank")
    reference = matching[0]
    with np.load(data_path, allow_pickle=False) as stored:
        arrays = {key: stored[key].copy() for key in (reference["state_key"], reference["key"])}
    parent, initial, truth = _validate_bank_arrays(bank, arrays, reference)
    model = make_model(family, track, spec["config"]).float().cpu().eval()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["state_dict"], strict=True)
    options = settings(execution)
    equation = Equation(options["kappa"], options["reaction_rate"])
    geometry = Geometry((reference["grid"],) * 2, (1., 1.))
    evaluation_rows = _read(evaluation / "evaluation-rows.json")
    original = [row for row in evaluation_rows if row["model_id"] == spec["model_id"]
                and row["parent_id"] == parent.parent_id and len(row["schedule"]) == 1
                and math.isclose(row["final_time"], reference["horizon"], abs_tol=1e-12)]
    if len(original) != 1:
        raise ValueError("Exactly one original one-step evaluation row is required for replay verification")
    sources = {str(path.relative_to(ROOT)): file_digest(path) for path in
        (protocols[0], pilot / "science_manifest.json", pilot / "workflow-seal.json", pilot / "freeze.json",
         pilot / "catalog.json", checkpoint, evaluation / "science_manifest.json", evaluation / "workflow-seal.json",
         evaluation / "reference-bank.json", data_path, evaluation / "evaluation-rows.json")}
    return dict(model=model, spec=spec, initial=initial, reference_state=truth, reference=reference,
        original_evaluation=original[0], original_device=evaluation_manifest["device"], source_profile=protocol["profile"], equation=equation, geometry=geometry,
        provenance=dict(source_run=str(run.relative_to(ROOT)), source_commit=pilot_manifest["software"]["git_commit"],
            source_tree_sha256=pilot_manifest["source_tree_sha256"], freeze_sha256=frozen["freeze_sha256"],
            pilot_manifest_sha256=file_digest(pilot / "science_manifest.json"),
            evaluation_manifest_sha256=file_digest(evaluation / "science_manifest.json"),
            verified_implementation_files=source_files, source_artifacts=sources,
            initial_fp64=_array_digest(initial), reference_fp64=_array_digest(truth),
            parent_identity_sha256=parent.identity_sha256, parent=parent.to_dict(),
            selection_rule="explicit family/track/parent/seed; never selected by endpoint error"))


@torch.no_grad()
def compute_prediction_data(model, initial, truth, h, equation, geometry, *, gain_delta=.1):
    """Replay FP32 inference and expose its actual internal response values."""
    if not 0 < gain_delta < .25:
        raise ValueError("Illustrative independent-gain perturbation must lie in (0,0.25)")
    u = torch.as_tensor(initial, dtype=torch.float32, device="cpu")
    reference = torch.as_tensor(truth, dtype=torch.float64, device="cpu")
    if u.shape != reference.shape or u.shape[0] != 1 or not torch.isfinite(reference).all():
        raise ValueError("One matching finite input/reference field is required")
    model = model.float().cpu().eval()
    mean = u.mean((-2, -1), keepdim=True)
    low = lowpass(u - mean, model.split_modes)
    high = u - mean - low
    raw_channels = model.channels(u, h, equation, geometry)
    projected = lowpass(raw_channels, model.modes) if model.output_compression else raw_channels
    response = model.response(u, h, equation, geometry)
    gains = response["gain"]
    parts = model.correction_components(u, h, equation, geometry)
    prediction = model(u, h, equation, geometry)
    manual = parts["base"] + (projected * gains[..., None, None]).sum(1, keepdim=True)
    # Moving the final linear projection across the sum can change roundoff.
    reconstruction_error = float((manual - prediction).abs().max())
    if reconstruction_error > 3e-7:
        raise ValueError("Exposed channels and gains do not reconstruct the actual prediction")
    unit_model = copy.deepcopy(model)
    if hasattr(unit_model, "conditioner"):
        unit_model.conditioner[-1].weight.zero_()
        unit_model.conditioner[-1].bias.zero_()
    elif hasattr(unit_model, "gain_logits"):
        unit_model.gain_logits.zero_()
    elif hasattr(unit_model, "affine_coefficients"):
        unit_model.affine_coefficients.zero_()
    unit_prediction = unit_model(u, h, equation, geometry)
    gain_change = (projected.double() * (gains.double() - 1)[..., None, None]).sum(1, keepdim=True)
    arrays = dict(initial_fp64=np.asarray(initial), input_fp32=u.numpy(), reference_fp64=reference.numpy(),
        mean=mean.numpy(), low_fluctuation=low.numpy(), high_fluctuation=high.numpy(),
        channels_before_output_projection=raw_channels.numpy(), channels=projected.numpy(),
        gains=gains.numpy(), features=response["features"].numpy(),
        weighted_channels=(projected * gains[..., None, None]).numpy(),
        correction=parts["increment"].numpy(), baseline=parts["base"].numpy(),
        prediction=prediction.numpy(), error=(prediction.double() - reference).numpy(),
        baseline_error=(parts["base"].double() - reference).numpy(),
        gain_derivative=projected.numpy(), unit_gain_prediction=unit_prediction.numpy(),
        learned_minus_unit_endpoint=(prediction.double() - unit_prediction.double()).numpy(),
        learned_minus_unit_analytic_correction=gain_change.numpy())
    params = {name: value.detach().cpu().numpy().copy() for name, value in model.state_dict().items()}
    arrays.update({"parameter__" + name.replace(".", "_"): value for name, value in params.items()})
    network = dict(kind=model.family, trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                   fitted_buffer_coefficients=model.parameter_report()["fitted_buffer_coefficients"], parameters={k: v.tolist() for k, v in params.items()})
    if hasattr(model, "conditioner"):
        features = response["features"]
        pre = model.conditioner[0](features)
        hidden = model.conditioner[1](pre)
        logits = model.conditioner[2](hidden)
        reproduced_gain = 1 + .75 * torch.tanh(logits)
        torch.testing.assert_close(reproduced_gain, gains, atol=0, rtol=0)
        arrays.update(hidden_preactivation=pre.numpy(), hidden_activation=hidden.numpy(), gain_logits=logits.numpy())
        network.update(layer_shapes=[3, model.hidden, 3], activation="SiLU(x)=x*sigmoid(x)",
                       exact_gain_reconstruction=True, hidden_activation=hidden.flatten().tolist(),
                       logits=logits.flatten().tolist())
    elif model.family == "channel_affine":
        design = torch.cat((torch.ones_like(response["features"][:, :1]), response["features"]), 1)
        logits = design @ model.affine_coefficients
        arrays["gain_logits"] = logits.numpy()
        network.update(logits=logits.flatten().tolist(), feature_intercept=True)
    elif model.family == "channel_global":
        arrays["gain_logits"] = model.gain_logits.numpy()
        network.update(logits=model.gain_logits.tolist())
    finite_changes, curves, actual_deltas = [], [], []
    # This changes independent effective gains. It does not claim to change a
    # network parameter independently, nor to represent an input derivative.
    requested_deltas = np.linspace(-gain_delta, gain_delta, 9)
    admissible = gain_bounds(model.family)
    for channel in range(3):
        actual_gain = float(gains[0, channel])
        minimum = max(-gain_delta, admissible[0] - actual_gain)
        maximum = min(gain_delta, admissible[1] - actual_gain)
        deltas = np.linspace(minimum, maximum, 9)
        # Keep the unperturbed prediction explicit, including asymmetric
        # near-bound intervals. Endpoints are floating-point range closures.
        if minimum < 0 < maximum and not np.any(deltas == 0):
            interior = 1 + int(np.argmin(abs(deltas[1:-1])))
            deltas[interior] = 0.
            deltas.sort()
        actual_deltas.append(deltas)
        delta_outputs = []
        error_curve = []
        for delta in deltas:
            hypothetical = prediction.double() + float(delta) * projected[:, channel:channel+1].double()
            change = hypothetical - prediction.double()
            delta_outputs.append(change.numpy())
            residual = hypothetical - reference
            error_curve.append(dict(delta=float(delta), effective_gain=float(gains[0, channel]) + float(delta),
                error_rms=float(residual.square().mean().sqrt()), error_max=float(residual.abs().max())))
        finite_changes.append(np.stack(delta_outputs))
        curves.append(dict(channel=CHANNEL_NAMES[channel], measurements=error_curve,
            actual_delta_interval=[float(deltas[0]), float(deltas[-1])],
            clipped_to_admissible_range=minimum != -gain_delta or maximum != gain_delta))
    arrays["gain_perturbation_deltas"] = requested_deltas
    arrays["gain_perturbation_actual_deltas"] = np.stack(actual_deltas)
    arrays["gain_perturbation_changes"] = np.stack(finite_changes)
    denominator = float(projected.square().sum(1).sqrt().norm())
    cancellation = float(projected.sum(1).norm()) / denominator if denominator > 0 else None
    info = dict(schema=SCHEMA, physical=dict(kappa=equation.kappa, reaction_rate=equation.reaction_rate,
            step=float(h), grid=list(geometry.grid), lengths=list(geometry.lengths), mean=float(mean),
            input_split_modes=model.split_modes, output_modes=model.modes,
            quadrature_nodes=model.quad_nodes, product="nodal FD" if model.track == "discrete" else "dealiased Galerkin"),
        network=network, features=response["features"].flatten().tolist(), gains=gains.flatten().tolist(),
        gain_formula="g = 1 (fixed)" if model.family == "channel_fixed" else "g = 1 + 0.75*tanh(logit)", channel_names=list(CHANNEL_NAMES),
        reconstruction_max_abs_error=reconstruction_error,
        error_rms=float((prediction.double() - reference).square().mean().sqrt()),
        error_max=float((prediction.double() - reference).abs().max()),
        baseline_error_rms=float((parts["base"].double() - reference).square().mean().sqrt()),
        channel_rms=[float(c.square().mean().sqrt()) for c in projected.unbind(1)],
        channel_combination_norm_over_root_sum_channel_norms=cancellation,
        sensitivity=dict(definition="d endpoint / d independent effective gain_c = P_K C_c; state, physics, other gains fixed",
            not_input_or_network_weight_derivative=True, perturbation_delta=gain_delta,
            admissible_gain_bounds=list(admissible), per_channel_deltas="gain_perturbation_actual_deltas in NPZ; clipped to attainable floating-point range closure",
            arithmetic="FP64 visualization of linear effective-gain perturbations of the actual FP32 endpoint", curves=curves),
        learned_vs_unit=dict(rounded_fp32_endpoint_max_change=float((prediction-unit_prediction).abs().max()),
            analytic_gain_mixing_max_change=float(gain_change.abs().max()),
            unit_gain_error_rms=float((unit_prediction.double()-reference).square().mean().sqrt()),
            interpretation="Actual rounded FP32 endpoint change and FP64 gain-mixing response are distinct quantities"),
        reference_used_for_inference=False, interpolation="none: nearest display of actual stored cells",
        execution="CPU replay and visualization; no new training, native GPU validation, or deployment timing")
    return info, arrays


def _images(info, arrays, output, *, dpi=400):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                         "lines.linewidth": .75, "pdf.fonttype": 42, "ps.fonttype": 42})
    def map_panel(fig, ax, values, title, *, symmetric=False, limit=None, units="state", value_range=None):
        values = np.asarray(values).squeeze()
        options = dict(origin="lower", interpolation="nearest", extent=(0, 1, 0, 1), aspect="equal")
        if symmetric:
            cap = limit or max(float(np.abs(values).max()), 1e-30)
            options.update(cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-cap, vcenter=0, vmax=cap))
        else:
            options.update(cmap="viridis")
            if value_range is not None:
                options.update(vmin=value_range[0], vmax=value_range[1])
        picture = ax.imshow(values.T, **options)
        ax.set(title=title, xlabel="x / Lx", ylabel="y / Ly")
        ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1])
        bar = fig.colorbar(picture, ax=ax, fraction=.045, pad=.025)
        bar.ax.tick_params(labelsize=7); bar.set_label(units, fontsize=7)
        return picture
    physical = info["physical"]
    status = info["selected_model"]["selection_status"]
    update = info["selected_model"]["selected_update"]
    title = f"TDN adjacent interaction prototype | Ours: {info['selected_model']['family']} | {physical['product']}"
    footer = (f"Actual {physical['grid'][0]} × {physical['grid'][1]} stored grid; nearest-cell display, no invented spatial detail. "
              f"{status}, selected update {update}; {info.get('source_profile', 'unspecified')} source, single-field illustration, not a superiority claim.\n"
              f"Checkpoint {info['selected_model']['checkpoint_sha256'][:16]} · Parent {info['parent_id']} · "
              "Reference appears only in evaluation panels. Red/blue means signed value; lower error is better.")
    paths = []
    def save(fig, name):
        fig.savefig(output / f"{name}.png", dpi=dpi, facecolor="white")
        fig.savefig(output / f"{name}.pdf", facecolor="white")
        plt.close(fig)
        paths.extend([f"{name}.png", f"{name}.pdf"])
    fig, axes = plt.subplots(4, 4, figsize=(16, 17))
    fig.subplots_adjust(left=.055, right=.97, top=.91, bottom=.085, wspace=.43, hspace=.49)
    fig.suptitle(title, fontsize=16, y=.975)
    fig.text(.5, .944, f"Physical inputs: h={physical['step']:g}, κ={physical['kappa']:g}, r={physical['reaction_rate']:g}, m={physical['mean']:.8f}  |  "
             f"Fixed cutoffs: input split {physical['input_split_modes']}, output {physical['output_modes']}; normalized GL{physical['quadrature_nodes']}", ha="center", fontsize=10)
    a = axes.ravel()
    state_range = (min(float(arrays[key].min()) for key in ("input_fp32", "baseline", "prediction", "reference_fp64")),
                   max(float(arrays[key].max()) for key in ("input_fp32", "baseline", "prediction", "reference_fp64")))
    fluctuation_limit = max(float(np.abs(arrays[key]).max()) for key in ("low_fluctuation", "high_fluctuation"))
    map_panel(fig, a[0], arrays["input_fp32"], "1. Observed state u(x,y)", value_range=state_range)
    map_panel(fig, a[1], arrays["low_fluctuation"], "2. Low input fluctuation vL", symmetric=True, limit=fluctuation_limit)
    map_panel(fig, a[2], arrays["high_fluctuation"], "3. High input fluctuation vH", symmetric=True, limit=fluctuation_limit)
    a[3].axis("off")
    features = info["features"]
    feature_text = "\n".join(f"{name} = {value:.9g}" for name, value in zip(
        ("mean(u)", "log(1+h r)", "log(1+h active diffusion)"), features))
    a[3].text(0, .99, "4. What the learned/fitted mixer sees", fontsize=11, va="top", weight="bold")
    network = info["network"]
    if network.get("layer_shapes"):
        architecture = (f"3 global features → {network['layer_shapes'][1]} SiLU units\n→ 3 effective channel gains\n\n"
                        f"{network['trainable_parameters']} neural weights/biases\n(no learned spatial convolution)")
    elif network["fitted_buffer_coefficients"]:
        architecture = f"{network['fitted_buffer_coefficients']} fitted coefficients\nNo hidden neural layers"
    else:
        architecture = "Fixed unit recombination\nNo fitted or learned parameters"
    a[3].text(0, .82, feature_text + "\n\n" + architecture +
        "\n\nThe physical branch sees the full field.\nThe reference is not an input.", va="top", linespacing=1.6)
    channel_limit = max(float(np.abs(arrays["channels"]).max()), 1e-30)
    for i, name in enumerate(CHANNEL_NAMES):
        map_panel(fig, a[4+i], arrays["channels"][0, i], f"5{i+1}. Projected signed {name}: P_K C{name}\nRMS {info['channel_rms'][i]:.3e}", symmetric=True, limit=channel_limit)
    gains = np.asarray(info["gains"])
    a[7].axhline(1, color="#777777", linestyle="--", linewidth=.7, label="Analytic control: gain 1")
    a[7].plot(range(3), gains, "^", color="#0072B2", markersize=7, label="Ours: selected effective gain")
    for i, value in enumerate(gains):
        a[7].annotate(f"{value:.8f}", (i, value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    pad = max(float(np.max(abs(gains-1))) * .8, .002)
    a[7].set(xlim=(-.4, 2.4), ylim=(min(1, gains.min())-pad, max(1, gains.max())+2*pad),
             xticks=range(3), xticklabels=CHANNEL_NAMES, title="6. Actual gains, initialized at 1", ylabel="gain (no universal better direction)")
    a[7].legend(loc="lower left", fontsize=6.5, frameon=False)
    weighted_limit = max(float(np.abs(arrays["weighted_channels"]).max()), 1e-30)
    for i, name in enumerate(CHANNEL_NAMES):
        map_panel(fig, a[8+i], arrays["weighted_channels"][0, i], f"7{i+1}. Actual g{name} × P_K C{name}", symmetric=True, limit=weighted_limit)
    map_panel(fig, a[11], arrays["correction"], "8. Sum: physical mixed correction", symmetric=True)
    map_panel(fig, a[12], arrays["baseline"], "9. Physical DF backbone S_h(u)", value_range=state_range)
    map_panel(fig, a[13], arrays["prediction"], "10. Ours: S_h(u) + correction", value_range=state_range)
    map_panel(fig, a[14], arrays["reference_fp64"], "11. Independent reference (evaluation)", value_range=state_range)
    map_panel(fig, a[15], arrays["error"], f"12. Endpoint error (lower |error| better)\nRMS {info['error_rms']:.3e}; max {info['error_max']:.3e}", symmetric=True)
    fig.text(.055, .032, footer, fontsize=8, va="bottom")
    save(fig, "prediction-architecture")

    # The matrices are actual learned global feature weights, not x/y filters.
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    fig.subplots_adjust(left=.075, right=.97, top=.86, bottom=.095, wspace=.4, hspace=.55)
    fig.suptitle("Actual parameter values and their meaning | " + info["selected_model"]["family"], fontsize=15, y=.98)
    fig.text(.5, .927, "These axes identify features, coefficients, hidden units, or LL/LH/HH responses—not spatial coordinates.\n"
             "Physical maps come from transport and signed products; the learned/fitted mixer supplies only three gains.", ha="center", fontsize=10)
    def matrix(ax, value, title, xlabel, ylabel, xlabels=None, ylabels=None):
        value = np.atleast_2d(value)
        cap = max(float(np.abs(value).max()), 1e-12)
        image = ax.imshow(value, interpolation="nearest", cmap="RdBu_r", vmin=-cap, vmax=cap, aspect="auto")
        for i in range(value.shape[0]):
            for j in range(value.shape[1]):
                ax.text(j, i, f"{value[i,j]:+.5f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(value[i,j]) > .6*cap else "black")
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.set_xticks(range(value.shape[1]), labels=xlabels if xlabels else range(value.shape[1]))
        ax.set_yticks(range(value.shape[0]), labels=ylabels if ylabels else range(value.shape[0]))
        fig.colorbar(image, ax=ax, fraction=.045, pad=.025).ax.tick_params(labelsize=7)
    if "parameter__conditioner_0_weight" in arrays:
        matrix(axes[0,0], arrays["parameter__conditioner_0_weight"], f"W1: {arrays['parameter__conditioner_0_weight'].shape[0]} × {arrays['parameter__conditioner_0_weight'].shape[1]} global-feature weights", "input feature", "hidden unit", ["mean", "log1p(hr)", "log1p(h rate)"])
        matrix(axes[0,1], arrays["parameter__conditioner_0_bias"][:,None], f"b1: {arrays['parameter__conditioner_0_bias'].size} hidden biases", "bias", "hidden unit")
        matrix(axes[0,2], arrays["hidden_activation"].T, "Actual SiLU(W1 f + b1)", "this state", "hidden unit")
        matrix(axes[1,0], arrays["parameter__conditioner_2_weight"], f"W2: {arrays['parameter__conditioner_2_weight'].shape[0]} × {arrays['parameter__conditioner_2_weight'].shape[1]} output weights", "hidden unit", "interaction origin", ylabels=CHANNEL_NAMES)
        matrix(axes[1,1], arrays["parameter__conditioner_2_bias"][:,None], f"b2: {arrays['parameter__conditioner_2_bias'].size} output biases", "bias", "interaction origin", ylabels=CHANNEL_NAMES)
    else:
        parameters = list(info["network"]["parameters"].items())
        for axis, (name, value) in zip(axes.ravel(), parameters):
            matrix(axis, np.asarray(value), name, "coefficient", "response")
        for axis in axes.ravel()[len(parameters):]:
            axis.axis("off")
    axes[1,2].axis("off")
    logits = info["network"].get("logits", [0,0,0])
    axes[1,2].text(0, .98, "Actual response mapping", fontsize=12, weight="bold", va="top")
    bounds = info["sensitivity"]["admissible_gain_bounds"]
    bound_label = f"Fixed gain: {bounds[0]:g}" if bounds[0] == bounds[1] else f"Admissible gain: ({bounds[0]:g}, {bounds[1]:g})"
    axes[1,2].text(0, .82, "\n\n".join(f"{name}: logit {raw:+.8f}\n       gain = {gain:.8f}" for name, raw, gain in zip(CHANNEL_NAMES, logits, gains))
        + f"\n\n{info['gain_formula']}\n{bound_label}\nBounded gain ≠ stability proof.", va="top", linespacing=1.4)
    for axis in axes[2]:
        axis.set_axis_on()
    change = info["learned_vs_unit"]
    map_panel(fig, axes[2,0], arrays["learned_minus_unit_endpoint"],
        f"Actual learned − unit-gain FP32 endpoint\nmax change {change['rounded_fp32_endpoint_max_change']:.3e}", symmetric=True)
    map_panel(fig, axes[2,1], arrays["learned_minus_unit_analytic_correction"],
        f"FP64 gain mix of stored FP32 channels\nmax change {change['analytic_gain_mixing_max_change']:.3e}", symmetric=True)
    axes[2,2].axis("off")
    axes[2,2].text(0, .98, "Do the gains visibly change this output?", weight="bold", va="top", fontsize=10)
    axes[2,2].text(0, .81, f"FP32 endpoint max change: {change['rounded_fp32_endpoint_max_change']:.9e}\n\n"
        f"Unit-gain RMS error: {change['unit_gain_error_rms']:.9e}\nSelected RMS error: {info['error_rms']:.9e}\n\n"
        "Small gain changes may fall below FP32\nendpoint rounding. A rescaled channel\nmap does not establish a visible or useful\ncomplete-solution improvement.\n\nLower error is better; this is one field.", va="top", linespacing=1.45, fontsize=8)
    fig.text(.075, .03, footer, fontsize=8, va="bottom")
    save(fig, "prediction-parameters")

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.subplots_adjust(left=.055, right=.965, top=.84, bottom=.11, wspace=.42, hspace=.48)
    fig.suptitle("What each effective channel gain changes | signed sensitivity", fontsize=16, y=.975)
    fig.text(.5, .925, "Analytic gain response before final FP32 rounding: ∂ endpoint / ∂ g_c = P_K C_c. Input and physics fixed.\n"
             "This is not a derivative with respect to an input pixel or neural weight. Gain probes are illustrative and clipped to admissible bounds.\n"
             "Signed response has no universal better direction; endpoint RMS error is lower-is-better.", ha="center", fontsize=10)
    for i, name in enumerate(CHANNEL_NAMES):
        derivative = arrays["gain_derivative"][0,i]
        changes = arrays["gain_perturbation_changes"][i]
        map_panel(fig, axes[i,0], derivative, f"{name}: analytic effective-gain derivative", symmetric=True, limit=channel_limit)
        actual_delta = arrays["gain_perturbation_actual_deltas"][i]
        map_panel(fig, axes[i,1], changes[-1], f"{name}: output change for {actual_delta[-1]:+.5g}", symmetric=True, limit=channel_limit*info["sensitivity"]["perturbation_delta"])
        map_panel(fig, axes[i,2], changes[0], f"{name}: output change for {actual_delta[0]:+.5g}", symmetric=True, limit=channel_limit*info["sensitivity"]["perturbation_delta"])
        rows = info["sensitivity"]["curves"][i]["measurements"]
        axes[i,3].plot([r["delta"] for r in rows], [r["error_rms"] for r in rows], "-^", color="#0072B2", markersize=4, linewidth=.75)
        axes[i,3].axvline(0, color="#777777", linewidth=.6, linestyle="--")
        axes[i,3].set(title=f"{name}: hypothetical gain perturbation", xlabel="change in effective gain", ylabel="endpoint RMS error ↓ better")
        axes[i,3].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        axes[i,3].grid(alpha=.18, linewidth=.4)
    fig.text(.055, .035, footer + "\nPerturbation curves use FP64 arithmetic around the actual FP32 endpoint; they are explanatory response curves, not new selected checkpoints.", fontsize=8, va="bottom")
    save(fig, "prediction-sensitivity")
    return paths


def render_prediction_view(run_dir, output_dir, *, family="channel_neural", track="discrete",
                           parent_index=0, horizon=None, seed=None, dpi=400):
    if type(dpi) is not int or dpi < 400 or dpi > 600:
        raise ValueError("Use 400–600 DPI for a legible high-resolution scientific view")
    run, output = _contained(run_dir), _contained(output_dir)
    if output.exists() or output.is_relative_to(run) or run.is_relative_to(output):
        raise ValueError("Use a new output directory outside the immutable source run")
    bundle = load_bundle(run, family=family, track=track, parent_index=parent_index, horizon=horizon, seed=seed)
    before = dict(bundle["provenance"]["source_artifacts"])
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        info, arrays = compute_prediction_data(bundle["model"], bundle["initial"], bundle["reference_state"],
            bundle["reference"]["horizon"], bundle["equation"], bundle["geometry"])
    finally:
        torch.set_num_threads(previous)
    original = bundle["original_evaluation"]
    replay_rms = abs(info["error_rms"] - original["error_rms"])
    replay_max = abs(info["error_max"] - original["error_max"])
    original_cpu = bundle["original_device"] == "cpu"
    replay_tolerance = 1e-12 if original_cpu else 3e-7
    if max(replay_rms, replay_max) > replay_tolerance:
        raise ValueError("CPU replay endpoint metrics differ from the recorded evaluation beyond the declared tolerance")
    renderer_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                     text=True, check=True).stdout.strip()
    renderer_files = {str(Path(__file__).relative_to(ROOT)): file_digest(Path(__file__)),
                      "scripts/adjacent_prediction_view.py": file_digest(ROOT / "scripts/adjacent_prediction_view.py")}
    info.update(selected_model={k: bundle["spec"][k] for k in ("family", "track", "model_id", "seed", "selection_status", "selected_update", "checkpoint_sha256", "config")},
        parent_id=bundle["reference"]["parent"]["parent_id"], parent_index=parent_index, source_profile=bundle["source_profile"],
        reference_metadata=bundle["reference"], provenance=bundle["provenance"],
        original_evaluation=original, replay=dict(original_device=bundle["original_device"], device="cpu", dtype="float32",
            absolute_rms_metric_difference=replay_rms, absolute_max_metric_difference=replay_max,
            relative_rms_metric_difference=replay_rms/max(abs(original["error_rms"]), 1e-30),
            relative_max_metric_difference=replay_max/max(abs(original["error_max"]), 1e-30),
            tolerance=replay_tolerance, comparison_scope="Endpoint metric parity only; historical full prediction array was not stored",
            status="SAME_CPU_METRIC_REPLAY" if original_cpu else "ALTERNATE_DEVICE_ILLUSTRATION_NOT_GPU_PARITY",
            no_new_timing_claim=True), renderer=dict(commit=renderer_commit, files=renderer_files,
                python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__), rendering=dict(dpi=dpi, interpolation="nearest", spatial_resolution_is_unchanged=True))
    output.mkdir(parents=True, exist_ok=False)
    (output / "prediction-data.json").write_text(json.dumps(info, indent=2, allow_nan=False) + "\n")
    with (output / "prediction-arrays.npz").open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    images = _images(info, arrays, output, dpi=dpi)
    for path, expected in before.items():
        if file_digest(ROOT / path) != expected:
            raise RuntimeError("Source artifact changed during prediction-view generation")
    manifest = dict(schema=SCHEMA, status="COMPLETED", output_scope="postprocessing only; immutable experiment untouched",
        renderer=info["renderer"], source_artifacts=before, source_commit=info["provenance"]["source_commit"],
        selected_model_id=info["selected_model"]["model_id"], parent_id=info["parent_id"],
        arrays={key: _array_digest(value) for key, value in arrays.items()},
        files={name: file_digest(output / name) for name in ["prediction-data.json", "prediction-arrays.npz", *images]})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return dict(output_dir=str(output), images=images, data="prediction-data.json", manifest="manifest.json",
                selected_model=info["selected_model"], parent_id=info["parent_id"], replay=info["replay"])
