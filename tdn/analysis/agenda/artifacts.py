"""Strict, stdlib-only closure of the bounded research agenda's evidence.

Manifests detect accidental alteration; they are local hashes, not signatures.
Every science file is inventoried with its size and SHA256. Mutable lifecycle
records and sibling Tower reports remain outside that scientific closure.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

from .protocol import PROFILES, STAGES, digest, validate_protocol
from tdn.runtime.metadata import write_json

SCHEMA = "tdn.agenda-artifacts/v1"
_SHA = re.compile(r"[a-f0-9]{64}\Z")
_MUTABLE = {"stage.json", "manifest.json", "COMPLETED"}
_BASE = {"protocol.json", "summary.json", "execution.json", "rows.json"}
_TRAIN = {"catalog.json", "science_manifest.json", "trained_physical_limits.json"}
_REQUIRED = {
    "structure": _BASE | {"cases.json", "fields.npz", "summary.txt"},
    "prepare": _BASE | {"science_manifest.json", "data_manifest.json", "references.json", "normalization.json",
                         "train.pt", "validation.pt", "calibration.pt", "confirmation.pt"},
    **{stage: _BASE | _TRAIN for stage in ("controls", "optimize", "compression", "kernel")},
    "confirm": _BASE | {"science_manifest.json", "frozen_checkpoints.json", "candidates.json", "frontiers.json", "criteria.json"},
    "policy": _BASE | {"science_manifest.json", "calibration.json", "attempts.json", "policy_endpoints.json", "policy_summaries.json"},
}


def file_digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _labels(stage, profile, source):
    if stage not in STAGES or profile not in PROFILES:
        raise ValueError("Unknown agenda stage/profile")
    if not isinstance(source, str) or not _SHA.fullmatch(source):
        raise ValueError("Agenda source_tree_sha256 requires a SHA256 fingerprint")


def _artifact(root, name):
    if not isinstance(name, str) or not name:
        raise ValueError("Unsafe agenda artifact path")
    part = Path(name)
    if part.is_absolute() or ".." in part.parts or part.as_posix() != name:
        raise ValueError("Unsafe agenda artifact path")
    path = root / part
    if (root.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve())
            or any((root / Path(*part.parts[:index])).is_symlink() for index in range(1, len(part.parts) + 1))):
        raise ValueError(f"Missing or unsafe agenda artifact: {name}")
    return path


def _read(root, name):
    try:
        value = json.loads(_artifact(root, name).read_text())
        digest(value)
    except (OSError, json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"Invalid agenda JSON: {name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Agenda {name} must be a JSON object")
    return value


def _rows(root, name="rows.json"):
    rows = _read(root, name).get("rows")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Agenda {name} requires a row list")
    return rows


def _number(value, *, positive=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


def _identities(rows, keys):
    values = [tuple(row.get(key) for key in keys) for row in rows]
    try:
        unique = set(values)
    except TypeError as error:
        raise ValueError("Agenda table identities must be scalar") from error
    if len(unique) != len(values):
        raise ValueError("Agenda table contains duplicate identities")
    return unique


def _inventory(root):
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Agenda science root must be an existing real directory")
    files = {}
    for path in sorted(root.rglob("*")):
        name = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError("Agenda artifacts cannot be symlinks")
        if "tower" in path.relative_to(root).parts:
            raise ValueError("Tower reports must be outside agenda science directories")
        if name in _MUTABLE:
            continue
        if path.is_file():
            if ".partial" in path.suffixes:
                raise ValueError("Unfinished agenda artifact remains")
            files[name] = {"sha256": file_digest(path), "bytes": path.stat().st_size}
    return files


def _inner(root, files, name, protocol_hash, required):
    value = _read(root, name)
    inventory = value.get("artifacts")
    if value.get("protocol_sha256") != protocol_hash or not isinstance(inventory, dict) or not required <= inventory.keys():
        raise ValueError(f"Agenda {name} protocol/inventory is incomplete")
    for relative, expected in inventory.items():
        if not isinstance(expected, str) or not _SHA.fullmatch(expected):
            raise ValueError("Inner agenda fingerprint is not SHA256")
        if relative not in files or files[relative]["sha256"] != expected:
            raise ValueError(f"Inner agenda artifact differs: {relative}")
    return value


def _lineage(root, stage, profile, source, protocol_hash, execution, cache, stack):
    values = execution.get("prerequisites", {})
    expected = set(STAGES[:STAGES.index(stage)])
    if not isinstance(values, dict) or set(values) != expected:
        raise ValueError("Agenda prerequisite coverage differs from its declared chain")
    predecessors = {}
    for name in STAGES[:STAGES.index(stage)]:
        value = values[name]
        if (not isinstance(value, dict) or set(value) != {"run_dir", "manifest_sha256"}
                or not isinstance(value["run_dir"], str) or not Path(value["run_dir"]).is_absolute()
                or not _SHA.fullmatch(str(value["manifest_sha256"]))):
            raise ValueError("Agenda prerequisite identity is invalid")
        # A complete collected archive may move machines. Prefer its sibling
        # stage when present; the recorded original location remains evidence.
        sibling = root.parent / name
        directory = sibling if (sibling / "manifest.json").is_file() else Path(value["run_dir"])
        manifest_path = _artifact(directory, "manifest.json")
        if file_digest(manifest_path) != value["manifest_sha256"]:
            raise ValueError(f"Agenda prerequisite manifest changed: {name}")
        prior = _verify(directory, name, profile, source, cache, stack)
        if prior["protocol_sha256"] != protocol_hash:
            raise ValueError("Agenda prerequisite belongs to another protocol")
        previous_execution = _read(directory, "execution.json")
        if any(previous_execution.get(key) != execution.get(key) for key in
               ("execution_mode", "slurm_profile_sha256", "workflow_protocol_sha256")):
            raise ValueError("Agenda prerequisite execution environment differs")
        software_keys = ("python", "executable", "venv", "torch", "numpy", "scipy", "torch_cuda_runtime", "source_tree_sha256")
        if any(previous_execution.get("software", {}).get(key) != execution.get("software", {}).get(key) for key in software_keys):
            raise ValueError("Agenda prerequisite software environment differs")
        prefix = {key: values[key] for key in STAGES[:STAGES.index(name)]}
        if previous_execution.get("prerequisites", {}) != prefix:
            raise ValueError("Agenda prerequisite belongs to another lineage prefix")
        predecessors[name] = directory
    return predecessors


def _structure(root, protocol, summary, rows):
    spec = protocol["structural"]
    ids = _identities(rows, ("case_id",))
    if ids != {(name,) for name in spec["case_ids"]} or _rows(root, "cases.json") != rows:
        raise ValueError("Structural cases differ from frozen declared coverage")
    required = set(spec["required_case_ids"])
    if any(type(row.get("required")) is not bool or row["required"] != (row["case_id"] in required) for row in rows):
        raise ValueError("Structural required flags differ from the immutable plan")
    if any(row["required"] and row.get("outcome") != "PASS" for row in rows):
        raise ValueError("Required structural correctness checks did not pass")
    if (summary.get("expected_cases") != len(rows) or summary.get("reported_cases") != len(rows)
            or summary.get("required_failures") != [] or summary.get("unreported_case_ids") != []
            or summary.get("training_attempted") is not False):
        raise ValueError("Structural completion coverage differs")


def _reference_ids(protocol):
    result = set()
    for parent in protocol["parents"]:
        split = parent["split"]
        if split in ("train", "validation"):
            horizons = sorted(set(protocol[f"{split}_horizons"] + [2 * h for h in protocol[f"{split}_horizons"]]))
            grids = [protocol["train_grid"]]
        else:
            horizons = sorted({round(sum(schedule), 12) for schedule in protocol["confirm_schedules"]})
            grids = protocol["grids"]
        for n in grids:
            for horizon in horizons:
                result.add((parent["parent_id"], split, n, horizon, "discrete"))
                if parent["parent_id"] in protocol["continuum_parent_ids"]:
                    result.add((parent["parent_id"], split, n, horizon, "continuum"))
    return result


def _prepare(root, protocol, summary, rows, files):
    names = {f"{split}.pt" for split in ("train", "validation", "calibration", "confirmation")}
    inner = _inner(root, files, "data_manifest.json", digest(protocol), names | {"normalization.json", "references.json"})
    references = _rows(root, "references.json")
    actual = set()
    for row in references:
        grid = row.get("grid")
        if (not isinstance(grid, list) or len(grid) != 2 or grid[0] != grid[1]
                or row.get("accepted") is not True or not _SHA.fullmatch(str(row.get("state_sha256", "")))
                or not _number(row.get("uncertainty_rms")) or not _number(row.get("uncertainty_max_bound"))):
            raise ValueError("Prepared reference is unaccepted or malformed")
        identity = (row.get("parent_id"), row.get("split"), grid[0], row.get("horizon"), row.get("track"))
        if identity in actual:
            raise ValueError("Prepared reference identity is duplicated")
        actual.add(identity)
    if actual != _reference_ids(protocol):
        raise ValueError("Prepared reference coverage differs from the frozen parent/grid/horizon plan")
    projected = [{"record_type": "reference", "question_ids": ["Q4", "Q6"], **row} for row in references]
    if rows != projected:
        raise ValueError("Prepared reference canonical rows differ")
    counts = dict(Counter(parent["split"] for parent in protocol["parents"]))
    if inner.get("counts") != {"parents": counts, "accepted_references": len(references)}:
        raise ValueError("Prepared cohort counts differ")
    if summary.get("counts", {}).get("parents") != counts or summary["counts"].get("accepted_references") != len(references):
        raise ValueError("Prepared summary differs from its completed banks")
    normalization = _read(root, "normalization.json")
    expected = [p["parent_id"] for p in protocol["parents"] if p["split"] == "train"]
    if (normalization.get("split") != "train" or normalization.get("parent_ids") != expected
            or normalization.get("protocol_sha256") != digest(protocol)
            or len(normalization.get("mean", [])) != 12 or len(normalization.get("std", [])) != 12
            or not all(_number(value, positive=True) for value in normalization["std"])):
        raise ValueError("Normalization must identify its training-only fit")


def _catalog(root, protocol, stage, summary, rows, files, predecessors):
    catalog = _read(root, "catalog.json")
    records = catalog.get("records")
    if (catalog.get("protocol_sha256") != digest(protocol) or catalog.get("fresh_cohort_opened") is not False
            or catalog.get("checkpoint_selection") != "validation only including initialization"
            or not isinstance(records, list) or any(not isinstance(row, dict) for row in records)):
        raise ValueError("Training catalog selection/provenance differs")
    plans = {row["trial_id"]: row for row in protocol["stage_trials"][stage]}
    if _identities(records, ("trial_id",)) != {(name,) for name in plans}:
        raise ValueError("Training catalog does not retain every declared trial")
    outcome_rows = [row for row in rows if row.get("record_type") in ("training", "optimization", "kernel")]
    if _identities(outcome_rows, ("trial_id",)) != {(name,) for name in plans}:
        raise ValueError("Canonical training outcomes do not retain every declared trial")
    for record in records:
        plan = plans[record["trial_id"]]
        if any(record.get(key) != plan[key] for key in ("family", "seed", "updates", "width", "modes")):
            raise ValueError("Resolved training identity differs from the declared trial")
        if record.get("base_orientation") != record.get("orientation"):
            raise ValueError("Resolved physical base orientations disagree")
        status = record.get("status")
        if status == "NOT_PROMOTED":
            if stage != "kernel" or _read(predecessors["structure"], "summary.json").get("rank_promising") is not False:
                raise ValueError("Skipped kernel lacks the prerequisite rank decision")
            continue
        if status not in ("COMPLETED", "NUMERICAL_FAILURE") or record.get("fresh_cohort_opened") is not False:
            raise ValueError("Training attempt is incomplete or used fresh parents")
        if (record.get("training_parent_ids") != [p["parent_id"] for p in protocol["parents"] if p["split"] == "train"]
                or record.get("validation_parent_ids") != [p["parent_id"] for p in protocol["parents"] if p["split"] == "validation"]):
            raise ValueError("Training or validation cohort differs")
        detailed = _read(root, record.get("training_record"))
        if {key: value for key, value in detailed.items() if key not in ("history", "gradient_history")} != record:
            raise ValueError("Catalog differs from detailed training evidence")
        if status == "COMPLETED":
            if record.get("steps") != plan["updates"] or not _number(record.get("best_validation_loss")):
                raise ValueError("Completed training does not cover its bounded update plan")
            selected_step = record.get("selected_step")
            if type(selected_step) is not int or not 0 <= selected_step <= plan["updates"]:
                raise ValueError("Selected checkpoint step is invalid")
            if record.get("selection") != ("TRAINED_CHECKPOINT" if selected_step else "SELECTED_INITIALIZATION"):
                raise ValueError("Checkpoint selection label differs from its selected step")
            for suffix, key in (("selected", "checkpoint_sha256"), ("trained", "trained_checkpoint_sha256")):
                name = f"checkpoints/{record['trial_id']}-{suffix}.pt"
                if name not in files or files[name]["sha256"] != record.get(key):
                    raise ValueError("Selected/trained checkpoint differs from its catalog")
            if record.get("checkpoint") != f"checkpoints/{record['trial_id']}-selected.pt" or f"checkpoints/{record['trial_id']}-initial.pt" not in files:
                raise ValueError("Training initial/selected artifact closure is incomplete")
    limits = _rows(root, "trained_physical_limits.json")
    completed = {record["trial_id"]: record for record in records if record["status"] == "COMPLETED"}
    expected_limits = {(identifier, case) for identifier in completed
                       for case in ("constant", "zero_reaction", "zero_diffusion", "zero_time")}
    if _identities(limits, ("trial_id", "case")) != expected_limits:
        raise ValueError("Selected trained-checkpoint physical-limit coverage is incomplete")
    for row in limits:
        if (row.get("required") is not True or row.get("status") != "PASS"
                or row.get("after_checkpoint_freeze") is not True or row.get("fresh_cohort_opened") is not False
                or row.get("checkpoint_sha256") != completed[row["trial_id"]]["checkpoint_sha256"]
                or row.get("tolerance") != 2e-6 or not _number(row.get("absolute_error")) or row["absolute_error"] > row["tolerance"]):
            raise ValueError("Required trained-checkpoint physical limit failed or differs from its freeze")
    if [row for row in rows if row.get("record_type") == "case"] != limits:
        raise ValueError("Canonical trained-limit rows differ")
    if summary.get("declared_trials") != len(plans) or summary.get("completed_trials") != sum(r["status"] == "COMPLETED" for r in records):
        raise ValueError("Training summary coverage differs from its catalog")
    return records


def _selected_records(protocol, predecessors):
    catalog = []
    for stage in ("controls", "compression", "kernel"):
        for record in _read(predecessors[stage], "catalog.json")["records"]:
            if record["status"] == "COMPLETED":
                catalog.append({**record, "source_stage": stage})
    selected = []
    for slot in protocol["confirmation_selection"]:
        candidates = [row for row in catalog if row["source_stage"] == slot["stage"]
                      and (slot["family"] == "best_eligible_pair" or row["family"] == slot["family"])
                      and row["seed"] == slot["seed"]]
        if candidates:
            chosen = min(candidates, key=lambda row: (row["best_validation_loss"], row["trial_id"]))
            if chosen["trial_id"] not in {row["trial_id"] for row in selected}:
                selected.append(chosen)
    return selected[:protocol["maximum_confirmation_variants"]]


def _metrics(row):
    if not all(_number(row.get(key)) for key in ("error_rms", "error_max", "uncertainty_rms", "uncertainty_max", "upper_rms", "upper_max")):
        raise ValueError("Completed endpoint error evidence is invalid")
    for norm in ("rms", "max"):
        if not math.isclose(row[f"upper_{norm}"], row[f"error_{norm}"] + row[f"uncertainty_{norm}"], rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("Endpoint uncertainty upper error differs")


def _confirm(root, protocol, summary, rows, predecessors):
    freeze = _read(root, "frozen_checkpoints.json")
    expected = _selected_records(protocol, predecessors)
    actual = freeze.get("records")
    if (freeze.get("protocol_sha256") != digest(protocol) or freeze.get("confirmation_seen_during_selection") is not False
            or not isinstance(actual, list) or len(actual) != len(expected)):
        raise ValueError("Fresh confirmation was not preceded by checkpoint selection freeze")
    for record, wanted in zip(actual, expected):
        if {key: value for key, value in record.items() if key != "source_dir"} != wanted:
            raise ValueError("Frozen selection differs from validation-only prerequisite catalogs")
        source = predecessors[record["source_stage"]]
        if file_digest(_artifact(source, record["checkpoint"])) != record["checkpoint_sha256"]:
            raise ValueError("Frozen checkpoint differs from its prerequisite")
    candidates, frontiers = _rows(root, "candidates.json"), _rows(root, "frontiers.json")
    methods = protocol["classical"] + [record["trial_id"] for record in actual]
    expected_candidates = set()
    expected_frontiers = set()
    for parent in protocol["parents"]:
        if parent["split"] != "confirmation":
            continue
        tracks = ["discrete"] + (["continuum"] if parent["parent_id"] in protocol["continuum_parent_ids"] else [])
        for n in protocol["grids"]:
            for track in tracks:
                for method in methods:
                    for index in range(len(protocol["confirm_schedules"])):
                        expected_candidates.add((parent["parent_id"], n, track, method, index))
                    for horizon in {round(sum(s), 12) for s in protocol["confirm_schedules"]}:
                        for norm in protocol["norms"]:
                            for target in protocol["targets"]:
                                expected_frontiers.add((parent["parent_id"], n, track, method, horizon, norm, target))
    if _identities(candidates, ("parent_id", "grid_size", "track", "trial_id", "schedule_index")) != expected_candidates:
        raise ValueError("Fresh confirmation candidate coverage is incomplete")
    if _identities(frontiers, ("parent_id", "grid_size", "track", "trial_id", "horizon", "norm", "target")) != expected_frontiers:
        raise ValueError("Fresh confirmation frontier coverage is incomplete")
    fingerprint = file_digest(_artifact(root, "frozen_checkpoints.json"))
    for row in candidates:
        if row.get("checkpoint_freeze_sha256") != fingerprint or row.get("schedule") != protocol["confirm_schedules"][row["schedule_index"]]:
            raise ValueError("Fresh endpoint differs from the frozen catalog/schedule")
        if row.get("status") not in ("COMPLETED", "NUMERICAL_FAILURE"):
            raise ValueError("Fresh endpoint status is incomplete")
        if row["status"] == "COMPLETED":
            _metrics(row)
            if not _number(row.get("timing_seconds"), positive=True) or row.get("complete_rollout") is not True or row.get("preparation_included") is not True:
                raise ValueError("Fresh complete-solver timing evidence is invalid")
    criteria_file = _read(root, "criteria.json")
    criteria = criteria_file.get("rows")
    if not isinstance(criteria, list) or criteria_file.get("frozen_success_criteria") != protocol["success_criteria"]:
        raise ValueError("Fresh success criteria differ from their immutable declaration")
    per_norm = [row for row in criteria if row.get("row_kind") != "joint_criterion"]
    joint = [row for row in criteria if row.get("row_kind") == "joint_criterion"]
    expected_criteria = {(record["trial_id"], track, norm, classical) for record in actual
                         for track in ("discrete", "continuum") for norm in protocol["norms"]
                         for classical in ("etdrk4", "gl3_fused")}
    if _identities(per_norm, ("trial_id", "track", "norm", "classical")) != expected_criteria:
        raise ValueError("Per-norm/classical success-criterion coverage differs")
    if _identities(joint, ("trial_id", "track")) != {(record["trial_id"], track) for record in actual for track in ("discrete", "continuum")}:
        raise ValueError("Joint success-criterion coverage differs")
    for row in per_norm:
        if row.get("status") not in ("PASS", "FAIL", "INCONCLUSIVE") or row.get("criteria_frozen_before_confirmation") is not True:
            raise ValueError("Success criterion is incomplete or was not frozen")
        if row["status"] == "PASS" and not all(row.get("gates", {}).values()):
            raise ValueError("Passing success criterion contains a failed gate")
    for row in joint:
        if row.get("status") == "PASS" and (row.get("checkpoint_selection") != "TRAINED_CHECKPOINT" or row.get("criteria_complete") is not True):
            raise ValueError("Initialization/incomplete selection cannot pass the joint gate")
    ledger = freeze.get("planned_slots")
    if (not isinstance(ledger, list) or len(ledger) != len(protocol["confirmation_selection"])
            or any(row.get("slot_index") != index or row.get("slot") != protocol["confirmation_selection"][index]
                   or row.get("validation_only_selection") is not True for index, row in enumerate(ledger))):
        raise ValueError("Confirmation omitted a declared mechanism slot")
    if rows != candidates + frontiers + criteria + ledger or summary.get("candidates") != len(candidates) or summary.get("frontiers") != len(frontiers):
        raise ValueError("Fresh confirmation canonical rows/summary differ")


def _policy(root, protocol, summary, rows, predecessors):
    calibration = _read(root, "calibration.json")
    attempts, endpoints = _rows(root, "attempts.json"), _rows(root, "policy_endpoints.json")
    summaries = _rows(root, "policy_summaries.json")
    if (calibration.get("protocol_sha256") != digest(protocol)
            or calibration.get("fresh_confirmation_seen_during_calibration") is not False):
        raise ValueError("Acceptance calibration lacks a pre-fresh freeze")
    selected = _selected_records(protocol, predecessors)
    preferred = protocol["policy"].get("families", ["source", "source_time", "source_closure"])
    selection = [min((record for record in selected if record["family"] == family), key=lambda row: (row["best_validation_loss"], row["trial_id"]))
                 for family in preferred if any(record["family"] == family for record in selected)][:protocol["policy"].get("model_limit", 3)]
    if calibration.get("status") == "NO_ELIGIBLE_CHECKPOINT":
        if (selection or attempts or endpoints or summaries or summary.get("models") != 0
                or summary.get("policy_status") != "NO_ELIGIBLE_CHECKPOINT"
                or calibration.get("checkpoint_sha256") != {} or calibration.get("envelopes") != {}):
            raise ValueError("Ineligible policy cannot hide selected or accepted trajectories")
        return
    checkpoints = {row["trial_id"]: row["checkpoint_sha256"] for row in selection}
    if calibration.get("checkpoint_sha256") != checkpoints or set(calibration.get("envelopes", {})) != set(checkpoints):
        raise ValueError("Policy calibration checkpoint set differs from validation-only selection")
    calibrating = calibration.get("rows")
    if not isinstance(calibrating, list) or not calibrating:
        raise ValueError("Policy requires independent calibration observations")
    parents = [p["parent_id"] for p in protocol["parents"] if p["split"] == "calibration"]
    if calibration.get("calibration_parent_ids") != parents or any(row.get("parent_id") not in parents or row.get("cohort") != "calibration" for row in calibrating):
        raise ValueError("Policy calibration used another cohort")
    estimators = protocol["policy"]["estimators"]
    horizons = sorted({round(sum(schedule), 12) for schedule in protocol["confirm_schedules"]})
    sizes = protocol["policy"]["attempted_step_sizes"]
    calibration_ids = {(identifier, parent, n, index) for identifier in checkpoints
                       for parent in parents for n in protocol["grids"] for index in range(len(horizons) * len(sizes))}
    if _identities(calibrating, ("trial_id", "parent_id", "grid_size", "schedule_index")) != calibration_ids:
        raise ValueError("Policy calibration coverage is incomplete")
    for row in calibrating:
        _metrics(row)
        if set(row.get("estimates", {})) != set(estimators) or any(not _number(row["estimates"][estimator].get(norm)) for estimator in estimators for norm in protocol["norms"]):
            raise ValueError("Policy calibration estimator evidence is invalid")
    for identifier, envelopes in calibration["envelopes"].items():
        if set(envelopes) != set(estimators):
            raise ValueError("Policy estimator envelope coverage differs")
        own = [row for row in calibrating if row["trial_id"] == identifier]
        for estimator in estimators:
            for norm in protocol["norms"]:
                wanted = max(1., max(row[f"upper_{norm}"] / max(row["estimates"][estimator][norm], 1e-12) for row in own)) * protocol["policy"]["safety_factor"]
                if not _number(envelopes[estimator].get(norm), positive=True) or not math.isclose(envelopes[estimator][norm], wanted, rel_tol=1e-12):
                    raise ValueError("Frozen policy envelope differs from independent calibration")
    fingerprint = file_digest(_artifact(root, "calibration.json"))
    groups = defaultdict(list)
    for row in attempts:
        if row.get("calibration_sha256") != fingerprint or not _number(row.get("charged_seconds")):
            raise ValueError("Policy attempt freeze/cost differs")
        groups[tuple(row.get(key) for key in ("trial_id", "parent_id", "grid_size", "schedule_index", "estimator"))].append(row)
    fresh = [parent for parent in protocol["parents"] if parent["split"] == "confirmation"]
    expected_groups = {(identifier, parent["parent_id"], n, index, estimator) for identifier in checkpoints
                       for parent in fresh for n in protocol["grids"] for index in range(len(horizons)) for estimator in estimators}
    if set(groups) != expected_groups:
        raise ValueError("Policy fresh attempted-case coverage is incomplete")
    max_attempts = min(protocol["policy"]["max_attempts"], len(sizes))
    for key, values in groups.items():
        if not 1 <= len(values) <= max_attempts:
            raise ValueError("Policy exceeded or omitted its bounded attempts")
        cumulative = 0.
        for index, row in enumerate(values):
            cumulative += row["charged_seconds"]
            if row.get("attempt") != index or not math.isclose(row.get("cumulative_seconds", -1), cumulative, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Policy rejected attempt sequence/cost differs")
            if row.get("status") not in ("ACCEPTED", "REJECTED", "NUMERICAL_FAILURE") or any(v.get("status") == "ACCEPTED" for v in values[:index]):
                raise ValueError("Policy attempt sequence continued after acceptance")
            horizon = horizons[key[3]]
            count = max(1, math.ceil(round(horizon / sizes[index], 12)))
            if row.get("schedule") != [horizon / count] * count:
                raise ValueError("Policy attempted schedule differs from the bounded proposal")
            if row["status"] != "NUMERICAL_FAILURE":
                estimate = row.get("estimate")
                if not isinstance(estimate, dict) or not all(_number(estimate.get(norm)) for norm in protocol["norms"]):
                    raise ValueError("Policy attempted estimator is invalid")
                envelope = calibration["envelopes"][key[0]][key[4]]
                target = protocol["policy"]["primary_target"]
                decision = all(envelope[norm] * max(estimate[norm], 1e-12) <= target for norm in protocol["norms"])
                if (row["status"] == "ACCEPTED") != decision:
                    raise ValueError("Policy acceptance disagrees with its frozen joint-norm rule")
                costs = row.get("costs", {})
                if not all(_number(costs.get(name)) for name in ("coarse_seconds", "doubling_extra_seconds", "defect_extra_seconds")):
                    raise ValueError("Policy estimator work evidence is missing")
                wanted_charge = costs["coarse_seconds"]
                if key[4] in ("step_doubling", "validated_envelope"):
                    wanted_charge += costs["doubling_extra_seconds"]
                if key[4] in ("splitting_defect", "validated_envelope"):
                    wanted_charge += costs["defect_extra_seconds"]
                if not math.isclose(row["charged_seconds"], wanted_charge, rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError("Policy charge omits requested estimator work")
    endpoint_ids = {(identifier, parent["parent_id"], n, index, estimator, track)
                    for identifier in checkpoints for parent in fresh for n in protocol["grids"]
                    for index in range(len(horizons)) for estimator in estimators
                    for track in (["discrete", "continuum"] if parent["parent_id"] in protocol["continuum_parent_ids"] else ["discrete"])}
    if _identities(endpoints, ("trial_id", "parent_id", "grid_size", "schedule_index", "estimator", "track")) != endpoint_ids:
        raise ValueError("Policy independently audited endpoint coverage is incomplete")
    for row in endpoints:
        key = tuple(row.get(name) for name in ("trial_id", "parent_id", "grid_size", "schedule_index", "estimator"))
        values = groups.get(key)
        if not values or row.get("calibration_sha256") != fingerprint or type(row.get("fallback")) is not bool:
            raise ValueError("Policy endpoint lacks its frozen attempt evidence")
        _metrics(row)
        fallback = row["fallback"]
        extra = row.get("fallback_seconds")
        if not _number(extra) or not math.isclose(row.get("timing_seconds", -1), sum(v["charged_seconds"] for v in values) + extra, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Policy endpoint omits rejected or fallback work")
        if (row.get("status") != ("FALLBACK" if fallback else "ACCEPTED")
                or row.get("rejected_attempts") != sum(v["status"] != "ACCEPTED" for v in values)
                or (not fallback and (values[-1]["status"] != "ACCEPTED" or extra != 0))):
            raise ValueError("Policy endpoint acceptance/fallback status differs")
        false_rms = not fallback and row["upper_rms"] > row["target_rms"]
        false_max = not fallback and row["upper_max"] > row["target_max"]
        if (row.get("false_accept_rms") != false_rms or row.get("false_accept_max") != false_max
                or row.get("false_accept_joint") != (false_rms or false_max)):
            raise ValueError("Policy false-acceptance flags differ from the declared upper-error criterion")
        if not {"etdrk4", "gl3_fused"} <= row.get("classical_controls", {}).keys():
            raise ValueError("Policy endpoint lacks classical complete-solver costs")
    if _identities(summaries, ("trial_id", "estimator", "track")) != {(identifier, estimator, track) for identifier in checkpoints for estimator in estimators for track in ("discrete", "continuum")}:
        raise ValueError("Policy summary coverage differs from independently audited endpoints")
    if rows != calibrating + attempts + endpoints + summaries or summary.get("attempts") != len(attempts) or summary.get("policy_endpoints") != len(endpoints):
        raise ValueError("Policy canonical event rows/summary differ")


def _validate(root, manifest, cache, stack):
    stage, profile, source = (manifest[key] for key in ("stage", "profile", "source_tree_sha256"))
    protocol, summary, execution = (_read(root, name) for name in ("protocol.json", "summary.json", "execution.json"))
    validate_protocol(protocol)
    protocol_hash = digest(protocol)
    if protocol["profile"] != profile or protocol_hash != manifest["protocol_sha256"]:
        raise ValueError("Agenda protocol differs from its seal")
    expected = {"stage": stage, "profile": profile, "protocol_sha256": protocol_hash}
    if (any(execution.get(key) != value for key, value in expected.items())
            or execution.get("software", {}).get("source_tree_sha256") != source
            or manifest["prerequisites"] != execution.get("prerequisites", {})):
        raise ValueError("Agenda execution identity differs from its seal")
    mode, device = execution.get("execution_mode"), execution.get("device")
    if (mode not in ("local-cpu", "desktop-slurm") or device not in ("cpu", "cuda")
            or (stage in ("structure", "prepare") and device != "cpu")
            or (mode == "local-cpu" and (profile == "full" or device != "cpu"))
            or (mode == "desktop-slurm" and stage not in ("structure", "prepare") and device != "cuda")
            or (device == "cuda" and mode != "desktop-slurm")):
        raise ValueError("Agenda execution mode/device differs from its stage")
    if mode == "desktop-slurm" and not _SHA.fullmatch(str(execution.get("slurm_profile_sha256", ""))):
        raise ValueError("Agenda Slurm profile requires a fingerprint")
    if summary.get("status") != "COMPLETED" or summary.get("stage") != stage or summary.get("device") != device or summary.get("profile", profile) != profile:
        raise ValueError("Incomplete or foreign agenda stage cannot be sealed")
    rows = _rows(root)
    if (not rows or _read(root, "rows.json").get("stage") != stage
            or _read(root, "rows.json").get("schema") != "tdn.agenda-rows/v1"):
        raise ValueError("Agenda canonical rows are absent or belong to another stage")
    predecessors = _lineage(root, stage, profile, source, protocol_hash, execution, cache, stack)
    files = manifest["files"]
    if "science_manifest.json" in files:
        science_required = set(files) - {"science_manifest.json", "execution.json"}
        _inner(root, files, "science_manifest.json", protocol_hash, science_required)
    elif stage != "structure":
        raise ValueError("Agenda inner scientific manifest is absent")
    if stage == "structure":
        _structure(root, protocol, summary, rows)
    elif stage == "prepare":
        _prepare(root, protocol, summary, rows, files)
    elif stage in ("controls", "optimize", "compression", "kernel"):
        _catalog(root, protocol, stage, summary, rows, files, predecessors)
    elif stage == "confirm":
        _confirm(root, protocol, summary, rows, predecessors)
    else:
        _policy(root, protocol, summary, rows, predecessors)


def seal_stage(run_dir, *, stage, profile, source_tree_sha256):
    root = Path(run_dir)
    _labels(stage, profile, source_tree_sha256)
    if (root / "manifest.json").exists() or (root / "COMPLETED").exists():
        raise ValueError("Preserve existing agenda seals")
    files = _inventory(root)
    if not _REQUIRED[stage] <= files.keys():
        raise ValueError("Agenda artifact inventory is incomplete")
    manifest = {"schema": SCHEMA, "version": 1, "benchmark_suite": "agenda", "stage": stage, "profile": profile,
        "source_tree_sha256": source_tree_sha256, "protocol_sha256": digest(_read(root, "protocol.json")),
        "prerequisites": _read(root, "execution.json").get("prerequisites", {}), "files": files}
    _validate(root, manifest, {}, set())
    write_json(root / "manifest.json", manifest)
    (root / "COMPLETED").write_text(file_digest(root / "manifest.json") + "\n")
    return manifest


def _verify(root, stage, profile, source, cache, stack):
    identity = str(root.resolve())
    if identity in stack:
        raise ValueError("Agenda prerequisite lineage is cyclic")
    if identity in cache:
        manifest = cache[identity]
        for key, wanted in (("stage", stage), ("profile", profile), ("source_tree_sha256", source)):
            if wanted is not None and manifest.get(key) != wanted:
                raise ValueError(f"Agenda predecessor {key} differs")
        return manifest
    path = _artifact(root, "manifest.json")
    if _artifact(root, "COMPLETED").read_text().strip() != file_digest(path):
        raise ValueError("Agenda completion marker differs from its manifest")
    manifest = _read(root, "manifest.json")
    if (set(manifest) != {"schema", "version", "benchmark_suite", "stage", "profile", "source_tree_sha256", "protocol_sha256", "prerequisites", "files"}
            or manifest.get("schema") != SCHEMA or type(manifest.get("version")) is not int or manifest["version"] != 1 or manifest.get("benchmark_suite") != "agenda"):
        raise ValueError("Unsupported agenda manifest")
    _labels(manifest.get("stage"), manifest.get("profile"), manifest.get("source_tree_sha256"))
    for key, wanted in (("stage", stage), ("profile", profile), ("source_tree_sha256", source)):
        if wanted is not None and manifest.get(key) != wanted:
            raise ValueError(f"Agenda predecessor {key} differs")
    files = manifest.get("files")
    if not isinstance(files, dict) or not _REQUIRED[manifest["stage"]] <= files.keys():
        raise ValueError("Agenda artifact inventory is incomplete")
    if any(not isinstance(value, dict) or set(value) != {"sha256", "bytes"} or not _SHA.fullmatch(str(value["sha256"])) or type(value["bytes"]) is not int or value["bytes"] < 0 for value in files.values()):
        raise ValueError("Agenda artifact fingerprint/size is malformed")
    if _inventory(root) != files:
        raise ValueError("Agenda science inventory changed, including extra/missing files")
    stack.add(identity)
    try:
        _validate(root, manifest, cache, stack)
    finally:
        stack.remove(identity)
    cache[identity] = manifest
    return manifest


def verify_stage(run_dir, *, stage=None, profile=None, source_tree_sha256=None):
    return _verify(Path(run_dir), stage, profile, source_tree_sha256, {}, set())


__all__ = ["SCHEMA", "seal_stage", "verify_stage", "digest", "file_digest"]
