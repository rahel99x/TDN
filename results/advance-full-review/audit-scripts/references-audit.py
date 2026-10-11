"""Read-only audit of exact retained native Advance evidence; no model fitting."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_io as io


def rows(name, key="rows"):
    """Stream an array from the pretty-printed original JSON with stdlib only."""
    path = io.BASE / "objects" / (io.index()["files"][name]["sha256"] + ".gz")
    decoder = json.JSONDecoder()
    with gzip.open(path, "rt") as stream:
        buf = ""
        marker = json.dumps(key) + ": ["
        while marker not in buf:
            piece = stream.read(65536)
            if not piece:
                raise ValueError((name, key))
            buf += piece
        buf = buf.split(marker, 1)[1]
        while True:
            buf = buf.lstrip(" \n\r\t,")
            if buf.startswith("]"):
                return
            try:
                value, end = decoder.raw_decode(buf)
            except json.JSONDecodeError:
                piece = stream.read(65536)
                if not piece:
                    raise
                buf += piece
                continue
            buf = buf[end:]
            yield value


def describe(values):
    values = sorted(values)
    return dict(count=len(values), minimum=min(values), median=statistics.median(values),
                maximum=max(values)) if values else dict(count=0)


def audit():
    p = io.load("protocol.json")["scientific_protocol"]
    frozen = io.load("freeze/freeze.json")
    declarations = {r["parent_id"]: r for r in p["parents"]}
    banks = [n for n in io.names() if n.endswith("/data_manifest.json")]
    cohort = defaultdict(list)
    reference_groups = defaultdict(list)
    attempts = Counter()
    errors = []
    reflookup = {}
    for name in banks:
        manifest = io.load(name)
        stage = name.split("/")[0]
        if stage != "prepare" and manifest["frozen_training"] != frozen:
            errors.append([stage, "frozen training binding"])
        accepted = unresolved = 0
        for rec in manifest["records"]:
            metadata_name = stage + "/" + rec["metadata"]
            parent = io.load(metadata_name)
            if hashlib.sha256(io.raw(metadata_name)).hexdigest() != rec["hashes"][rec["metadata"]]:
                errors.append([metadata_name, "metadata digest"])
            if any(parent.get(k) != v for k, v in declarations[parent["parent_id"]].items()):
                errors.append([metadata_name, "declaration mismatch"])
            cohort[parent["split"]].append(parent)
            for key, r in parent["references"].items():
                reflookup[parent["parent_id"], r["track"], r["grid"], r["horizon"]] = r
                accepted += bool(r["accepted"])
                unresolved += not r["accepted"]
                reference_groups[parent["split"], r["track"], r["grid"]].append(r)
                if r["track"] == "discrete":
                    temporal = [r]
                else:
                    temporal = r["temporal_refinements"]
                    if r["spatial_reference_grids"] != [2*r["grid"], 4*r["grid"]]:
                        errors.append([parent["parent_id"], key, "spatial grids"])
                for t in temporal:
                    for a in t["temporal_refinements"]:
                        c = a["counts"]
                        if not (c[1] == 2*c[0] and c[2] == 4*c[0]):
                            errors.append([parent["parent_id"], key, "non-doubled refinement"])
                    a = t["temporal_refinements"][-1]
                    attempts[str(tuple(a["counts"]))] += 1
                    if t["accepted"] and not a["accepted"]:
                        errors.append([parent["parent_id"], key, "acceptance inconsistency"])
                if r["accepted"] and (not r["finite"] or not r["temporal_accepted"] or
                        r["uncertainty_rms"] > p["reference_tolerance"] or
                        r["uncertainty_max_bound"] > p["reference_tolerance"]):
                    errors.append([parent["parent_id"], key, "reference tolerance or finiteness"])
        if (accepted, unresolved) != (manifest["accepted_references"], manifest["unresolved_references"]):
            errors.append([stage, "reference totals"])

    identities = {}
    for split, parents in cohort.items():
        identities[split] = {
            key: {json.dumps(row[key], sort_keys=True) for row in parents}
            for key in ("parent_id", "field_seed", "field_cluster", "continuous_field_id", "field_terms")
        }
    overlaps = []
    for left in sorted(identities):
        for right in sorted(identities):
            if left >= right:
                continue
            for key in identities[left]:
                common = identities[left][key] & identities[right][key]
                if common:
                    overlaps.append(dict(left=left, right=right, key=key, count=len(common)))

    validation = list(rows("freeze/validation_frontier.json"))
    groups = defaultdict(list)
    validation_records = 0
    validation_ids = set()
    for r in rows("freeze/validation_frontier.json", "raw_validation_rows"):
        validation_records += 1
        validation_ids.add(r["parent_id"])
        groups[r["model_id"], r["track"], r["final_time"], r["schedule_id"]].append(
            dict(parent_id=r["parent_id"], cost=r["cost_seconds"],
                 finite=r["finite"], accepted=r["reference_accepted"],
                 rms=r["upper_rms"], maximum=r["upper_max"]))
    opportunity_sizes = Counter()
    for key, vals in groups.items():
        opportunity_sizes[len(vals)] += 1
        if len({v["parent_id"] for v in vals}) != len(vals):
            errors.append([list(key), "duplicate validation field"])
    mismatch = []
    for cell in validation:
        possible = []
        for key, vals in groups.items():
            if key[:3] != (cell["model_id"], cell["track"], cell["final_time"]):
                continue
            if all(v["finite"] and v["accepted"] and v["rms"] <= cell["rms_target"] and v["maximum"] <= cell["max_target"] for v in vals):
                possible.append((statistics.mean(v["cost"] for v in vals), key[3]))
        winner = min(possible) if possible else None
        if cell["schedule_id"] != (winner[1] if winner else None):
            mismatch.append(cell)

    endpoints = 0
    confirmation_ids = set()
    metadata_mismatches = []
    below_uncertainty = Counter()
    reference_error_ratios = defaultdict(list)
    primary_rows = Counter()
    endpoint_cells = set()
    model_schedules = defaultdict(set)
    for r in rows("aggregate/endpoint_rows.json"):
        endpoints += 1
        confirmation_ids.add(r["parent_id"])
        endpoint_cells.add((r["parent_id"], r["model_id"], r["grid"], r["schedule_id"]))
        model_schedules[r["model_id"]].add(r["schedule_id"])
        key = (r["parent_id"], r["track"], r["grid"], r["final_time"])
        ref = reflookup[key]
        if r["reference_accepted"] != ref["accepted"] or r["reference_uncertainty_rms"] != ref["uncertainty_rms"] or r["reference_uncertainty_max"] != ref["uncertainty_max_bound"]:
            metadata_mismatches.append(list(key))
        if r["error_rms"] <= ref["uncertainty_rms"]:
            below_uncertainty[r["track"]] += 1
        if r["error_rms"] > 0:
            reference_error_ratios[r["track"]].append(ref["uncertainty_rms"]/r["error_rms"])
        if r["primary"]:
            primary_rows[r["track"], r["model_id"]] += 1
    expected_locked = defaultdict(set)
    for cell in validation:
        if cell["schedule_id"] is not None:
            expected_locked[cell["model_id"]].add(cell["schedule_id"])
    menu_mismatches = []
    for model, schedules in model_schedules.items():
        expected = expected_locked[model] | {"primary-00", "primary-01"}
        if schedules != expected:
            menu_mismatches.append(dict(model=model, expected=sorted(expected), actual=sorted(schedules)))

    report = dict(
        schema="tdn.advance-review.references/v1", scientific_protocol_sha256=frozen["protocol_sha256"],
        source_tree_sha256=frozen["source_tree_sha256"],
        parent_count=sum(len(v) for v in cohort.values()),
        splits={split:dict(parents=len(vals), independent_fields=len({v["field_cluster"] for v in vals}),
                          regimes=dict(Counter(v["regime"] for v in vals)),
                          distributions=dict(Counter(v["distribution"] for v in vals)),
                          grids=sorted({int(g) for v in vals for g in v["states"]})) for split, vals in cohort.items()},
        cross_split_identity_overlaps=overlaps,
        reference_count=sum(len(v) for v in reference_groups.values()),
        reference_groups=[dict(split=k[0], track=k[1], grid=k[2], count=len(vals), accepted=sum(bool(v["accepted"]) for v in vals),
                               uncertainty_rms=describe([v["uncertainty_rms"] for v in vals]),
                               uncertainty_max=describe([v["uncertainty_max_bound"] for v in vals]),
                               spatial_difference_rms=describe([v["spatial_refinement"]["rms"] for v in vals if "spatial_refinement" in v]),
                               spatial_difference_max=describe([v["spatial_refinement"]["max"] for v in vals if "spatial_refinement" in v]))
                          for k, vals in sorted(reference_groups.items())],
        final_temporal_refinement_counts=dict(attempts),
        freeze_bindings_match_every_fresh_bank=not any(v[1] == "frozen training binding" for v in errors),
        validation=dict(raw_rows=validation_records, fields=len(validation_ids), expected_fields=32,
                        chosen_cells=len(validation), outcome_counts=dict(Counter(v["status"] for v in validation)),
                        opportunity_field_sizes=dict(opportunity_sizes),
                        selection_mismatch_count=len(mismatch),
                        per_track_outcomes={track:dict(Counter(v["status"] for v in validation if v["track"]==track)) for track in p["tracks"]}),
        confirmation=dict(raw_endpoints=endpoints, unique_cells=len(endpoint_cells), fields=len(confirmation_ids),
                          reference_metadata_mismatch_count=len(metadata_mismatches),
                          schedules_match_only_primary_and_validation_locked=not menu_mismatches,
                          endpoint_rms_at_or_below_uncertainty=dict(below_uncertainty),
                          uncertainty_to_error_ratio={k:describe(v) for k,v in reference_error_ratios.items()},
                          primary_rows_per_model=sorted(set(primary_rows.values()))),
        errors=errors, menu_mismatches=menu_mismatches,
        limitations=["Independent unit is continuous field, not grid/seed/schedule/repeated timing or grid point.",
                     "Reference uncertainty is a refinement estimate, not a rigorous certificate; raw arrays deliberately omitted in review export.",
                     "Endpoint references at .12 and .24 only; intermediate-state/long-rollout correctness is not covered.",
                     "Confirmation quarter shifts are unpaired and mix baseline physical coefficients and phases; not isolated shift interventions.",
                     "16-versus-64 training changes physical coverage as well as sample count.",
                     "near_nyquist label is relative to training N32 (mode14); same modes on N64/N128 are farther from their Nyquist limits.",
                     "Scaling has eight fresh fields and is not a new 64-field N128 confirmation cohort."])
    target=io.BASE/"references-audit.json"
    target.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("reference_groups", "limitations")},indent=2))


if __name__ == "__main__":
    audit()
