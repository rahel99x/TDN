"""Bounded CPU implementation verification; no scientific promotion path.

The tiny development cohort is intentionally inadequate for scientific claims.
Selection finishes before any diagnostic parent is evaluated. All inference
features come from the actual FP32 initial state, promoted explicitly to FP64.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import uuid
import xml.etree.ElementTree as ET

import numpy as np
import scipy
from scipy.integrate import solve_ivp
import torch

from tdn.analysis.response import numerics as n
from tdn.analysis.tangent_response.oracle import NumpyEquation, independent_solve, refined_teacher
from tdn.analysis.tangent_response.tangent import AnchorPath, refined_action
from .model import FixedFeatureResponse
from .numerics import ordered_response

ROOT = Path(__file__).resolve().parents[3]
TARGET = "fd_same_grid"
SCHEMA = "tdn.response_preconditioner.implementation.v1"
VERIFY_TESTS = (
    "tests/test_response_preconditioner_numerics.py",
    "tests/test_response_preconditioner_model.py",
    "tests/test_response_preconditioner_fitting.py",
    "tests/test_response_preconditioner_audit.py",
    "tests/test_response_preconditioner_workflow.py",
    "tests/test_response_preconditioner_launch.py::test_shell_help_and_rejects_unallocated_execution",
    "tests/test_response_preconditioner_launch.py::test_native_script_syntax",
    "tests/test_response_preconditioner_launch.py::test_prepare_exclusive_identity_and_changed_source_detection",
    "tests/test_response_preconditioner_launch.py::test_source_identity_ignores_outputs_but_tracks_code_and_recipe",
    "tests/test_response_preconditioner_launch.py::test_allocation_resource_expansion_rejected",
    "tests/test_response_preconditioner_launch.py::test_shared_venv_lock_rejects_setup_writer",
    "tests/test_response_preconditioner_launch.py::test_metrics_preserve_split_method_and_task_without_aggregate",
    "tests/test_response_preconditioner_launch.py::test_stage_registers_exact_verification_xml_and_binary_artifacts",
    "tests/test_response_preconditioner_launch.py::test_stage_success_cannot_promote_science_or_hide_failed_summary",
    "tests/test_response_preconditioner_launch.py::test_failed_preflight_is_retained_without_child_work",
    "tests/test_response_preconditioner_launch.py::test_execution_keeps_stdout_stderr_and_child_exit",
    "tests/test_tangent_response_numerics.py",
    "tests/test_response_numerics.py",
)


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def source_snapshot(root=ROOT):
    root = Path(root)
    paths = set(root.glob("tdn/**/*.py"))
    paths.update(root.glob("scripts/response_preconditioner*"))
    paths.update(root.glob("tests/test_response_preconditioner*.py"))
    paths.update(root / item.split("::")[0] for item in VERIFY_TESTS)
    paths.add(root / "pyproject.toml")
    rows = [{"path": str(path.relative_to(root)),
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in sorted(paths) if path.is_file()]
    return {"algorithm": "sha256", "files": rows, "fingerprint": _digest(rows)}


def create_output(command, output=None, *, root=ROOT):
    root = Path(root).resolve()
    if command not in ("verify", "smoke"):
        raise ValueError("Only bounded verify and smoke commands are implemented")
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = root / "runs" / f"response-preconditioner-{command}-{stamp}-{uuid.uuid4().hex[:8]}"
    output = Path(output)
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    if output == root or not output.is_relative_to(root):
        raise ValueError("Output must remain inside this checkout")
    if output.exists():
        raise ValueError("Run directory exists; preserve it and select a fresh output")
    output.mkdir(parents=True, exist_ok=False)
    return output


def protocol(command):
    return dict(schema=SCHEMA, command=command, device="cpu", target=TARGET,
                scope="BOUNDED_CPU_IMPLEMENTATION", decision="IMPLEMENTATION_ONLY",
                scientific_promotion=False, historical_gate="B STOP unchanged",
                parent_counts={"train": 2, "validation": 1, "diagnostic": 1},
                parent_seeds={"train": [2026101501, 2026101502],
                              "validation": [2026101503], "diagnostic": [2026101504]},
                grid=[5, 5], lengths=[1., 1.], horizons=[.04, .08],
                source_directions_per_parent=2, source_interval_start=0.,
                input_dtype="float32", model_internal_dtype="float64", output_dtype="float64",
                teacher_dtype="float64", teacher_tolerance=1e-10,
                ridge_menu=[1e-10, 1e-6], include_zero_readout_selection=True,
                fit_scope="train parents only",
                loss="parent-balanced unnormalized physical squared error; equal weight per scalar row",
                selection="minimum validation combined endpoint/action MSE among initialization and numerically eligible fits; initialization first breaks ties",
                action_correction_denominator="physical RMS of (U-G)z on the identical anchor",
                correction_resolution_multiple=10.,
                diagnostic_scope="held-out development parents; never confirmation",
                baseline_steps=[1, 2, 4, 8], response_baseline_panels=[1, 2, 4, 8],
                fit_precision="float64", feature_count=40,
                verify_tests=list(VERIFY_TESTS), future_stages={
                    "learned_encoder": "BLOCKED_PENDING_REPRESENTATION_AND_UTILITY_EVIDENCE",
                    "nonlinear_second_correction": "BLOCKED_PENDING_MEASURED_REMAINDER",
                    "gpu_training": "NOT_IMPLEMENTED_OR_AUTHORIZED_BY_THIS_PROTOCOL",
                    "scientific_confirmation": "BLOCKED", "promotion": "BLOCKED"})


def _software():
    return dict(python=sys.version, executable=sys.executable, torch=torch.__version__,
                numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform(),
                device="cpu", threads=torch.get_num_threads(),
                execution_mode=os.environ.get("TDN_EXECUTION_MODE", "local-cpu"),
                slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                cuda_execution=False, tf32_used=False)


def _journal(output, event, **values):
    row = dict(event=event, utc=datetime.now(timezone.utc).isoformat(), **values)
    with (output / "journal.jsonl").open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _metrics(error):
    x = error.detach().double()
    return dict(rms=float(x.square().mean().sqrt()), maximum=float(x.abs().max()),
                mean_error=float(x.mean()), centered_rms=float((x-x.mean()).square().mean().sqrt()))


def parents(config, split):
    """Generate deterministic, independent parents without looking at labels."""
    for index, seed in enumerate(config["parent_seeds"][split]):
        rng = np.random.default_rng(seed)
        x = torch.arange(5, dtype=torch.float64) / 5
        phase = rng.uniform(0., 2*np.pi, size=3)
        initial = (.30 + .04*rng.random() + .09*torch.cos(2*torch.pi*x[:, None]+phase[0])
                   + .06*torch.sin(2*torch.pi*x[None, :]+phase[1])
                   + .025*torch.cos(2*torch.pi*(x[:, None]+x[None, :])+phase[2]))
        u = initial[None, None].float()
        directions = []
        for _ in range(config["source_directions_per_parent"]):
            z = torch.from_numpy(rng.normal(size=(1, 1, 5, 5))).double()
            directions.append(z/z.square().mean().sqrt())
        yield dict(parent_id=f"{split}-{index:03d}-{seed}", split=split, seed=seed, u=u,
                   eq=n.Physics(float(.009+.008*rng.random()), float(1.7+1.2*rng.random())),
                   geom=n.Geometry((5, 5), (1., 1.)), directions=directions)


def _acquire(parent, horizon, model, config):
    u, eq, geom = parent["u"], parent["eq"], parent["geom"]
    started = time.perf_counter()
    teacher, metadata = refined_teacher(u, horizon, eq, geom, TARGET, config=dict(
        tolerance=config["teacher_tolerance"], initial_steps=4, max_steps=512, max_seconds_per_query=20.))
    records = [dict(kind="endpoint", direction=None, teacher=teacher.detach(), reference=metadata)]
    for index, z in enumerate(parent["directions"]):
        path = AnchorPath(u.double(), eq, geom, TARGET)
        refined, refinement = refined_action(path, eq, geom, TARGET, z, 0., horizon,
                                             initial_steps=4, max_steps=256, tolerance=2.5e-11)
        independent, cross = independent_solve(u.double(), horizon, eq, geom, TARGET,
                                              kind="action", z=z, tolerance=1e-12)
        difference = _metrics(refined-independent) if refined is not None else dict(rms=None, maximum=None)
        combined = {key: refinement["uncertainty_"+key]+difference[key]
                    if refinement["uncertainty_"+key] is not None and difference[key] is not None else None
                    for key in ("rms", "maximum")}
        accepted = bool(refinement["accepted"] and all(v is not None and v <= config["teacher_tolerance"]
                                                        for v in combined.values()))
        mean_refined, mean_refinement = refined_action(AnchorPath(u.double(), eq, geom, TARGET), eq, geom,
            TARGET, z, 0., horizon, mode="mean", initial_steps=4, max_steps=256, tolerance=2.5e-11)
        mean_independent, mean_cross = independent_solve(u.double(), horizon, eq, geom, TARGET,
            kind="action", z=z, mode="mean", tolerance=1e-12)
        mean_difference = _metrics(mean_refined-mean_independent) if mean_refined is not None else dict(rms=None, maximum=None)
        mean_uncertainty = {key: mean_refinement["uncertainty_"+key]+mean_difference[key]
            if mean_refinement["uncertainty_"+key] is not None and mean_difference[key] is not None else None
            for key in ("rms", "maximum")}
        mean_accepted = bool(mean_refinement["accepted"] and all(v is not None and v <= config["teacher_tolerance"]
                                                                for v in mean_uncertainty.values()))
        records.append(dict(kind="action", direction=index, teacher=independent.detach(),
                            matched_mean=mean_independent.detach(),
                            reference=dict(accepted=accepted, status="ACCEPTED" if accepted else "UNRESOLVED",
                                refinement=refinement, independent=cross, cross_difference=difference,
                                uncertainty_rms=combined["rms"], uncertainty_maximum=combined["maximum"],
                                uncertainty_is_certificate=False,
                                matched_mean=dict(accepted=mean_accepted, refinement=mean_refinement,
                                    independent=mean_cross, cross_difference=mean_difference,
                                    uncertainty_rms=mean_uncertainty["rms"], uncertainty_maximum=mean_uncertainty["maximum"],
                                    uncertainty_is_certificate=False))))
    teacher_seconds = time.perf_counter()-started
    started = time.perf_counter()
    for record in records:
        work = {}
        if record["kind"] == "endpoint":
            base, design = model.endpoint_design(u, horizon, eq, geom, work=work)
        else:
            base, design = model.action_design(u, parent["directions"][record["direction"]],
                                               0., horizon, eq, geom, work=work)
        record.update(base=base.detach(), design=design.detach(), work=work, horizon=horizon,
                      parent_id=parent["parent_id"], split=parent["split"])
    return records, dict(teacher_seconds=teacher_seconds, feature_seconds=time.perf_counter()-started)


def _matrix(records):
    design = torch.cat([row["design"].flatten(1).T for row in records])
    target = torch.cat([(row["teacher"]-row["base"]).reshape(-1) for row in records])
    ids = [row["parent_id"] for row in records for _ in range(row["teacher"].numel())]
    return design, target, ids


def _prediction(row, coefficients):
    return row["base"]+torch.einsum("p,p...->...", coefficients, row["design"])


def _correction_metrics(record, prediction, config):
    if record["kind"] != "action":
        return {}
    reference, mean = record["reference"], record["reference"]["matched_mean"]
    correction_norm = _metrics(record["teacher"]-record["matched_mean"])["rms"]
    uncertainties = (reference["uncertainty_rms"], mean["uncertainty_rms"])
    uncertainty = sum(uncertainties) if all(x is not None for x in uncertainties) else None
    if not reference["accepted"] or not mean["accepted"] or uncertainty is None:
        status = "UNRESOLVED_REFERENCE"
    elif correction_norm <= config["correction_resolution_multiple"]*uncertainty:
        status = "UNRESOLVED_NEAR_NULL"
    else:
        status = "RESOLVED_NONTRIVIAL"
    return dict(action_correction_rms=correction_norm, correction_uncertainty_rms=uncertainty,
                correction_status=status, action_correction_relative_rms=(
                    _metrics(prediction-record["teacher"])["rms"]/correction_norm
                    if status == "RESOLVED_NONTRIVIAL" else None))


def _validation_loss(records, coefficients):
    groups = {}
    for row in records:
        groups.setdefault(row["parent_id"], []).append(float((_prediction(row, coefficients)-row["teacher"]).square().mean()))
    return float(np.mean([np.mean(values) for values in groups.values()]))


def _whole_flow_jvp(u, z, horizon, eq, geom):
    """Independent NumPy/SciPy variational solve of the complete nonlinear flow."""
    equation = NumpyEquation(eq, geom, TARGET)
    size = u.numel()
    initial = np.concatenate([u.double().numpy().reshape(-1), z.numpy().reshape(-1)])
    def rhs(_, state):
        primal = state[:size].reshape(geom.grid)
        tangent = state[size:].reshape(geom.grid)
        return np.concatenate([equation.rhs(primal).reshape(-1), equation.tangent(primal, tangent).reshape(-1)])
    solved = solve_ivp(rhs, (0., horizon), initial, method="DOP853", rtol=1e-12, atol=1e-14,
                       t_eval=[horizon])
    if not solved.success or not np.isfinite(solved.y).all():
        raise RuntimeError("Independent whole-flow variational solve failed")
    return torch.from_numpy(solved.y[size:, -1].reshape(u.shape)), dict(
        method="independent_numpy_DOP853_coupled_primal_variational", nfev=solved.nfev,
        uncertainty_is_certificate=False, tolerance_is_error_certificate=False)


def _diagnostic_rows(parent, records, model, config):
    rows, derivative_rows, work_rows = [], [], []
    u, eq, geom = parent["u"], parent["eq"], parent["geom"]
    for record in records:
        h = record["horizon"]
        common = {key: record[key] for key in ("parent_id", "split", "kind", "direction", "horizon")}
        common["reference_accepted"] = record["reference"]["accepted"]
        candidates = [("zero_readout", lambda record=record: record["base"], "cached_attribution_only")]
        work = {}
        if record["kind"] == "endpoint":
            candidates.append(("fixed_feature_fitted", lambda:model(u, h, eq, geom, work=work), "complete_uncached_call"))
            for orientation in ("df", "rf"):
                for steps in config["baseline_steps"]:
                    def split(orientation=orientation, steps=steps):
                        value = u.double()
                        for _ in range(steps):
                            value = n.anchor(value, h/steps, eq, geom, TARGET, name=orientation)
                        return value
                    candidates.append((f"strang_{orientation}_{steps}", split, "complete_call"))
        else:
            z = parent["directions"][record["direction"]]
            candidates.append(("fixed_feature_fitted", lambda:model.response_action(u, z, 0., h, eq, geom, work=work), "complete_uncached_call"))
            for panels in config["response_baseline_panels"]:
                def ordered(panels=panels):
                    path = AnchorPath(u.double(), eq, geom, TARGET)
                    return ordered_response(path, z, 0., h, eq, geom, panels=panels)
                candidates.append((f"ordered_response_{panels}", ordered, "complete_call"))
        for name, call, scope in candidates:
            started = time.perf_counter()
            value = call().detach()
            seconds = time.perf_counter()-started
            error = _metrics(value-record["teacher"])
            rows.append(dict(**common, method=name, **error, elapsed_seconds=seconds,
                             **_correction_metrics(record, value, config),
                             timing_scope=scope, timing_repetitions=1, speedup_claim=False,
                             output_dtype=str(value.dtype), accepted_error_certificate=False))
            if name == "fixed_feature_fitted":
                torch.testing.assert_close(value, _prediction(record, model.readout.detach()), rtol=1e-12, atol=1e-12)
                work_rows.append(dict(**common, work=work))
        if record["kind"] == "endpoint":
            z = parent["directions"][0]
            value = u.double().requires_grad_(True)
            _, derivative = torch.autograd.functional.jvp(lambda state:model(state, h, eq, geom), value, z)
            reference, metadata = _whole_flow_jvp(u, z, h, eq, geom)
            finite_differences = []
            for eps in (1e-3, 3e-4, 1e-4, 3e-5):
                estimate = (model(value+eps*z, h, eq, geom)-model(value-eps*z, h, eq, geom))/(2*eps)
                finite_differences.append(dict(epsilon=eps, **_metrics(estimate-derivative)))
            derivative_identity = {key: value for key, value in common.items() if key != "reference_accepted"}
            derivative_rows.append(dict(**derivative_identity,
                                        endpoint_reference_accepted=common["reference_accepted"],
                                        derivative_reference_status="INDEPENDENT_DIAGNOSTIC_UNCERTAINTY_UNQUALIFIED",
                                        whole_model_jvp_vs_flow=_metrics(derivative-reference),
                                        finite_difference_sweep=finite_differences, reference=metadata,
                                        internal_response_is_whole_model_jvp=False))
    return rows, derivative_rows, work_rows


def _smoke(output, config):
    from .fitting import fit_parent_balanced_ridge
    model = FixedFeatureResponse()
    _json(output / "architecture.json", model.architecture_metadata())
    _journal(output, "protocol_frozen", protocol_sha256=_digest(config))
    records, parent_records, timing, references = [], {}, [], []
    # Diagnostic generation and teacher acquisition happen only after selection.
    for split in ("train", "validation"):
        for parent in parents(config, split):
            parent_records[parent["parent_id"]] = parent
            for horizon in config["horizons"]:
                acquired, costs = _acquire(parent, horizon, model, config)
                records.extend(acquired)
                timing.append(dict(parent_id=parent["parent_id"], horizon=horizon, **costs))
                for row in acquired:
                    references.append({**{k: row[k] for k in ("parent_id", "split", "kind", "direction", "horizon")},
                                       "reference": row["reference"]})
                _json(output / "references.json", references)
                _json(output / "timing.json", timing)
                _journal(output, "parent_horizon_acquired", parent_id=parent["parent_id"], horizon=horizon)
    if not all(row["reference"]["accepted"] for row in records):
        raise RuntimeError("Unresolved train/validation references block fitting; all outcomes retained")
    training = [row for row in records if row["split"] == "train"]
    validation = [row for row in records if row["split"] == "validation"]
    design, target, ids = _matrix(training)
    fits = [dict(candidate="zero_readout", ridge=None, coefficients=torch.zeros(model.coefficient_count, dtype=torch.float64),
                 eligible=True, diagnostics=dict(status="INITIALIZATION_NO_OPTIMIZATION"),
                 validation_loss=_validation_loss(validation, torch.zeros_like(model.readout)))]
    started = time.perf_counter()
    for ridge in config["ridge_menu"]:
        try:
            fit = fit_parent_balanced_ridge(design, target, ids, ridge=ridge)
            coefficients = fit["coefficients"]
            eligible = bool(fit["diagnostics"]["optimality_verified"] and fit["diagnostics"]["baseline_nonregression_verified"])
            fits.append(dict(candidate=f"ridge_{ridge:g}", ridge=ridge, coefficients=coefficients.detach(),
                             eligible=eligible,
                             eligibility_reason="NUMERICAL_OPTIMALITY_AND_NONREGRESSION_VERIFIED" if eligible else "NUMERICAL_FIT_CHECK_FAILED",
                             diagnostics=fit["diagnostics"], validation_loss=_validation_loss(validation, coefficients)))
        except (ArithmeticError, ValueError, RuntimeError) as error:
            fits.append(dict(candidate=f"ridge_{ridge:g}", ridge=ridge, coefficients=None,
                             eligible=False, eligibility_reason="FITTING_FAILED",
                             diagnostics=dict(status="FAILED", error_type=type(error).__name__, error=str(error)),
                             validation_loss=None))
        _json(output / "fitting.json", [{k:v.tolist() if torch.is_tensor(v) else v
                                        for k,v in row.items()} for row in fits])
    fitting_seconds = time.perf_counter()-started
    selected = min((fit for fit in fits if fit["eligible"]), key=lambda fit:fit["validation_loss"])
    model.set_readout(selected["coefficients"])
    selection = dict(criterion=config["selection"], selected_ridge=selected["ridge"], selected_candidate=selected["candidate"],
                     coefficients=selected["coefficients"].tolist(),
                     candidates=[{k:v.tolist() if torch.is_tensor(v) else v for k,v in row.items()} for row in fits],
                     training_parent_ids=sorted(set(ids)),
                     validation_parent_ids=sorted({row["parent_id"] for row in validation}),
                     diagnostic_access_before_selection=False, fitting_seconds=fitting_seconds)
    _json(output / "selection.json", selection)
    _journal(output, "selection_frozen", selection_sha256=_digest(selection))
    metrics, derivatives, work = [], [], []
    for row in records:
        for name, coefficients in (("zero_readout", torch.zeros_like(model.readout)),
                                   ("fixed_feature_fitted", model.readout.detach())):
            metrics.append({**{key:row[key] for key in ("parent_id", "split", "kind", "direction", "horizon")},
                            "method": name, **_metrics(_prediction(row, coefficients)-row["teacher"]),
                            **_correction_metrics(row, _prediction(row, coefficients), config),
                            "reference_accepted": row["reference"]["accepted"],
                            "elapsed_seconds": None, "timing_scope": "cached_design_fitting_diagnostic",
                            "speedup_claim": False, "output_dtype": "torch.float64"})
    for parent in parents(config, "diagnostic"):
        parent_records[parent["parent_id"]] = parent
        for horizon in config["horizons"]:
            acquired, costs = _acquire(parent, horizon, model, config)
            records.extend(acquired)
            timing.append(dict(parent_id=parent["parent_id"], horizon=horizon, **costs))
            references.extend([{**{k: row[k] for k in ("parent_id", "split", "kind", "direction", "horizon")},
                                "reference": row["reference"]} for row in acquired])
            _json(output / "references.json", references)
            _json(output / "timing.json", timing)
            current, derivative, counts = _diagnostic_rows(parent, acquired, model, config)
            metrics.extend(current); derivatives.extend(derivative); work.extend(counts)
            _json(output / "metrics.json", metrics)
            _json(output / "derivatives.json", derivatives)
            _journal(output, "diagnostic_parent_horizon_complete", parent_id=parent["parent_id"], horizon=horizon)
            if horizon == config["horizons"][0]:
                from .audit import audit_cubic_fd_panel
                audit_rows = []
                initial = parent["u"].double()
                for method, endpoint in (("fixed_feature_fitted", model(initial, horizon, parent["eq"], parent["geom"]).detach()),
                                         ("strang_df_1", n.anchor(initial, horizon, parent["eq"], parent["geom"], TARGET))):
                    started = time.perf_counter()
                    audited = audit_cubic_fd_panel(initial[0, 0], endpoint[0, 0],
                        n.rhs(initial, parent["eq"], parent["geom"], TARGET)[0, 0],
                        n.rhs(endpoint, parent["eq"], parent["geom"], TARGET)[0, 0],
                        step=horizon, kappa=parent["eq"].kappa, reaction=parent["eq"].reaction_rate,
                        lengths=parent["geom"].lengths, true_initial_state=initial[0, 0])
                    audit_rows.append(dict(parent_id=parent["parent_id"], horizon=horizon, method=method,
                                           audit=audited, audit_elapsed_seconds=time.perf_counter()-started,
                                           scope="single_cubic_panel_same_grid_FD_only"))
                _json(output / "audit.json", audit_rows)
    _json(output / "work.json", work)
    fieldnames = sorted({key for row in metrics for key in row})
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(metrics)
    arrays = {}
    for index, row in enumerate(records):
        for key in ("teacher", "base", "design", "matched_mean"):
            if key not in row:
                continue
            arrays[f"query_{index:03d}_{key}"] = row[key].cpu().numpy()
    for index, parent in enumerate(parent_records.values()):
        arrays[f"parent_{index:03d}_input_fp32"] = parent["u"].numpy()
        arrays[f"parent_{index:03d}_directions"] = torch.stack(parent["directions"]).numpy()
    np.savez_compressed(output / "arrays.npz", **arrays)
    _json(output / "array_index.json", dict(
        queries=[{k:row[k] for k in ("parent_id", "split", "kind", "direction", "horizon")} for row in records],
        parents=[dict(parent_id=p["parent_id"], split=p["split"], seed=p["seed"],
                      kappa=p["eq"].kappa, reaction_rate=p["eq"].reaction_rate)
                 for p in parent_records.values()]))
    unresolved = [row for row in references if not row["reference"]["accepted"]]
    unresolved_means = [row for row in references if row["kind"] == "action"
                        and not row["reference"]["matched_mean"]["accepted"]]
    return dict(metrics_count=len(metrics), reference_count=len(references),
                unresolved_references=len(unresolved), selected_ridge=selected["ridge"],
                unresolved_matched_mean_references=len(unresolved_means),
                selected_candidate=selected["candidate"],
                ineligible_fit_count=sum(not row["eligible"] for row in fits),
                diagnostic_parent_count=1, training_parent_count=2, validation_parent_count=1,
                whole_model_derivative_cases=len(derivatives),
                numerical_certification="TWO_TINY_SINGLE_PANEL_FD_RESIDUAL_CERTIFICATES_ONLY",
                general_solver_certification=False, matched_accuracy_speedup_established=False)


def _verify(output, config):
    missing = [name for name in VERIFY_TESTS if not (ROOT/name.split("::")[0]).is_file()]
    if missing:
        raise RuntimeError("Declared verification tests missing: "+", ".join(missing))
    command = [sys.executable, "-m", "pytest", "-q", *VERIFY_TESTS,
               "--basetemp="+str(output/"pytest-temp"), "--junitxml="+str(output/"verification.xml")]
    _json(output / "command.json", dict(argv=command, cwd=str(ROOT)))
    with (output / "verification.log").open("w") as handle:
        completed = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                   env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    report = ET.parse(output / "verification.xml").getroot()
    cases = list(report.iter("testcase"))
    checks = dict(total=len(cases), failed=sum(case.find("failure") is not None for case in cases),
                  errors=sum(case.find("error") is not None for case in cases),
                  skipped=sum(case.find("skipped") is not None for case in cases))
    _json(output / "checks.json", checks)
    if completed.returncode or checks["failed"] or checks["errors"] or checks["skipped"] or not checks["total"]:
        raise RuntimeError("Bounded CPU verification did not pass every declared case; inspect verification.log")
    return dict(checks=checks)


def execute(command, output=None, *, device="cpu"):
    if device != "cpu":
        raise ValueError("This stage implements explicit CPU verification only")
    output = create_output(command, output)
    config = protocol(command)
    before = source_snapshot()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    started = time.perf_counter()
    summary = dict(schema=SCHEMA, command=command, status="RUNNING", decision="IMPLEMENTATION_ONLY",
                   scientific_promotion=False, native_gpu_readiness=False, historical_gate="B STOP unchanged",
                   output=str(output), device="cpu", execution_mode=os.environ.get("TDN_EXECUTION_MODE", "local-cpu"),
                   source_fingerprint=before["fingerprint"])
    _json(output / "protocol.json", config)
    _json(output / "source.json", before)
    _json(output / "software.json", _software())
    _json(output / "summary.json", summary)
    _journal(output, "run_started", command=command)
    try:
        summary.update((_verify if command == "verify" else _smoke)(output, config))
        after = source_snapshot()
        _json(output / "source_after.json", after)
        if before != after:
            raise RuntimeError("Source changed during execution; the run cannot seal")
        summary["status"] = "COMPLETED"
    except Exception as error:
        summary.update(status="FAILED", error_type=type(error).__name__, error=str(error))
        (output / "failure.txt").write_text(traceback.format_exc())
    finally:
        summary["elapsed_seconds"] = time.perf_counter()-started
        _json(output / "summary.json", summary)
        _journal(output, "run_finished", status=summary["status"])
        (output / "README.md").write_text(
            "# Response preconditioner implementation run\n\n"
            f"Status: **{summary['status']}**. Decision: **IMPLEMENTATION_ONLY**.\n\n"
            "This bounded CPU run preserves historical B STOP and grants no scientific promotion, "
            "GPU readiness, general solver certification or speedup claim. The four parents are development "
            "fixtures, with two train, one validation and one diagnostic parent. The source, protocol, "
            "software, failures and selection records are preserved. Diagnostic parents are acquired "
            "after readout selection.\n\n"
            "See `summary.json`, `protocol.json`, `source.json`, `software.json` and `journal.jsonl`. "
            "Smoke additionally writes `selection.json`, `references.json`, `metrics.csv`, "
            "`derivatives.json`, `work.json`, `audit.json`, `arrays.npz` and its index. The audit contains "
            "validated bounds for two tiny single cubic panels of the same-grid FD equation only. "
            "Timing is a single-call "
            "implementation diagnostic; it is not a controlled benchmark. FP64 output assembly is "
            "explicit. Teacher agreement is an uncertainty estimate, not a proof.\n\n"
            "Standing research reference: `docs/TDN_RESEARCH_PROGRAM.md`; original architecture "
            "derivation: `docs/TDN_ARCHITECTURE_PROPOSAL_20261015.md` in the source checkout.\n")
        artifacts = [{"path": str(path.relative_to(output)), "bytes": path.stat().st_size,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                     for path in sorted(output.rglob("*")) if path.is_file() and "pytest-temp" not in path.parts]
        _json(output / "artifacts.json", dict(schema=SCHEMA, source_fingerprint=before["fingerprint"],
                                             status=summary["status"], files=artifacts,
                                             self_excluded="artifacts.json"))
        torch.set_num_threads(previous_threads)
    return summary
