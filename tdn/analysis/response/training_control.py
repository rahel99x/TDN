"""Deterministic, work-indexed controls for bounded Response diagnostics."""
from __future__ import annotations

import math


def control_config(config):
    """Validate opt-in controls without changing legacy stochastic defaults."""
    reporting_control(config)
    updates = int(config["updates"])
    if updates <= 0 or int(config.get("validation_every", max(1, updates//10))) <= 0:
        raise ValueError("Training and validation update budgets must be positive")
    mode = config.get("batch_mode", "stochastic")
    if mode not in ("stochastic", "full_parent"):
        raise ValueError("batch_mode must be stochastic or full_parent")
    if int(config.get("batch_size", 1)) <= 0:
        raise ValueError("batch_size must be positive")
    schedule = dict(config.get("lr_schedule") or {"kind": "fixed"})
    kind = schedule.get("kind", "fixed")
    if kind not in ("fixed", "cosine"):
        raise ValueError("lr_schedule kind must be fixed or cosine")
    minimum = float(schedule.get("minimum_ratio", .05))
    warmup = int(schedule.get("warmup_updates", 0))
    head_warmup = int(config.get("head_warmup_updates", 0))
    if not math.isfinite(minimum) or not 0 <= minimum <= 1:
        raise ValueError("minimum_ratio must be finite and between zero and one")
    if not 0 <= warmup < updates or not 0 <= head_warmup <= updates:
        raise ValueError("Warmup must lie within the declared logical update budget")
    base = float(config["setting"]["lr"])
    if not math.isfinite(base) or base <= 0:
        raise ValueError("Learning rate must be finite and positive")
    if config.get("total_seconds") is not None and (not math.isfinite(float(config["total_seconds"])) or float(config["total_seconds"]) <= 0):
        raise ValueError("Declared trial time allowance must be finite and positive")
    return dict(batch_mode=mode, head_warmup_updates=head_warmup,
                lr_schedule=dict(kind=kind, minimum_ratio=minimum, warmup_updates=warmup),
                requested_updates=updates, base_learning_rate=base)


def learning_rate(control, completed):
    """Rate for the *next* optimizer commit, unaffected by allocation boundaries."""
    schedule = control["lr_schedule"]
    base = control["base_learning_rate"]
    if schedule["kind"] == "fixed":
        return base
    warmup, total = schedule["warmup_updates"], control["requested_updates"]
    if completed < warmup:
        return base*(completed+1)/warmup
    # First post-warmup update uses base; the last uses the frozen minimum.
    denominator = max(1, total-warmup-1)
    fraction = min(1., max(0., (completed-warmup)/denominator))
    ratio = schedule["minimum_ratio"]+(1-schedule["minimum_ratio"])*.5*(1+math.cos(math.pi*fraction))
    return base*ratio


def full_parent_plan(parent_samples):
    """Stable query order with equal total weight for each independent parent."""
    count = len(parent_samples)
    if not count:
        raise ValueError("Full-parent training needs independent parents")
    return [(index, 1./(count*len(parent_samples[parent])))
            for parent in sorted(parent_samples) for index in parent_samples[parent]]


def is_head(name):
    return any(word in name.lower() for word in ("head", "readout", "gate", "output", "amplitude"))


def reporting_control(config):
    """An opt-in reporting policy; never controls optimizer/checkpoint cadence."""
    keys = ("public_curve_every", "live_every", "compact_public_query_plan")
    every = config.get("public_curve_every", 1)
    live = config.get("live_every", 1)
    if type(every) is not int or every <= 0 or type(live) is not int or live <= 0:
        raise ValueError("Public curve/live cadence must be positive integer logical updates")
    compact = config.get("compact_public_query_plan", False)
    if type(compact) is not bool:
        raise ValueError("compact_public_query_plan must be boolean")
    return dict(enabled=any(key in config for key in keys), public_curve_every=every,
                live_every=live, compact_public_query_plan=compact)


def whole_history_summary(history):
    """Summarize committed optimization rows, never a sampled public curve."""
    rows = [row for row in history if int(row.get("update", 0)) > 0]
    fields = ("gradient_norm", "optimizer_gradient_norm", "gradient_norm_after_clip",
              "head_gradient_norm", "trunk_gradient_norm", "learning_rate",
              "parameter_movement", "trunk_movement", "train_loss", "endpoint_loss",
              "scaled_objective", "rollout_loss", "source_aux_loss", "consistency_loss")
    values = {}
    for name in fields:
        pairs = [(int(row["update"]), float(row[name])) for row in rows
                 if isinstance(row.get(name), (int, float)) and not isinstance(row.get(name), bool)
                 and math.isfinite(float(row[name]))]
        numbers = [number for _, number in pairs]
        maximum = max(pairs, key=lambda pair:pair[1]) if pairs else None
        magnitude = max((abs(value) for value in numbers), default=0.)
        values[name] = dict(count=len(numbers), missing_count=len(rows)-len(numbers),
            minimum=min(numbers) if numbers else None, maximum=maximum[1] if maximum else None,
            maximum_update=maximum[0] if maximum else None,
            mean=math.fsum(value/len(numbers) for value in numbers) if numbers else None,
            # Stable scaling avoids overflow for finite but large gradients.
            rms=(magnitude*math.sqrt(math.fsum((value/magnitude)**2 for value in numbers)/len(numbers)))
                if magnitude else (0. if numbers else None),
            first=numbers[0] if numbers else None, last=numbers[-1] if numbers else None)
    clipping = [row["clipped"] for row in rows if type(row.get("clipped")) is bool]
    return dict(schema="tdn.response-optimization-history-summary/v1",
        source="every committed optimizer-update row in the exact journal history; initialization excluded",
        committed_updates=len(rows), first_update=rows[0]["update"] if rows else None,
        last_update=rows[-1]["update"] if rows else None, metrics=values,
        clipping_observations=len(clipping), clipped_updates=sum(clipping),
        clipped_fraction=sum(clipping)/len(clipping) if clipping else None,
        head_only_warmup_updates=sum(bool(row.get("head_only_warmup")) for row in rows),
        committed_query_visits=sum(int(row.get("query_visits", 0)) for row in rows),
        final_examples_seen=int(rows[-1].get("examples_seen", 0)) if rows else 0,
        partial_work_scope="Uncommitted/failed work remains in trial.work counters and all-attempt costs, not committed-gradient summaries")


def sampled_curve_rows(history, policy, *, chunk_start_update, prior_rows=(), pending_validation=False):
    """Copy observations without changing their full journal query traces."""
    from tdn.research.protocol import digest
    plan_fields = ("parent_ids", "query_ids", "horizons", "phase_variants", "query_weights", "query_visits")
    plans = {}
    public = {}
    terminal = int(history[-1]["update"]) if history else None
    for row in history:
        copied = dict(row)
        if policy["compact_public_query_plan"] and any(name in row for name in plan_fields):
            plan = {name:row[name] for name in plan_fields if name in row}
            checksum = digest(plan)
            if checksum in plans and plans[checksum] != plan:
                raise ValueError("Immutable query plan digest collision")
            plans[checksum] = plan
            for name in plan_fields:
                copied.pop(name, None)
            copied["query_plan_sha256"] = checksum
        if pending_validation and int(row["update"]) == terminal and row.get("validation_loss") is None:
            # This is a distinct incomplete observation, never a changed copy of
            # a later canonical completed-validation point.
            copied["observation_kind"] = "pending_validation"
        public[int(row["update"])] = copied
    retained = {int(row["update"]):dict(row) for row in prior_rows}
    terminal = int(history[-1]["update"]) if history else None
    reasons = {}
    for row in history:
        update = int(row["update"])
        role = []
        if update == 0:
            role.append("initialization")
        if update > 0 and update % policy["public_curve_every"] == 0:
            role.append("declared_cadence")
        if row.get("validation_loss") is not None:
            role.append("completed_validation")
        if update == terminal:
            role.append("attempt_last_committed_update")
        if role and (update > chunk_start_update or chunk_start_update == 0 or update == terminal
                     or update == chunk_start_update and row.get("validation_loss") is not None):
            retained[update] = public[update]
            reasons[str(update)] = role
    # Reconstruct prior rows from authoritative journal observations so a resumed
    # same-directory export remains compact and references the available plan map.
    for update in retained:
        if update in public:
            retained[update] = public[update]
    rows = [retained[key] for key in sorted(retained)]
    contract = dict(schema="tdn.response-learning-curve-export/v1", **policy,
        journal_every_update=True, exact_query_trace="journal.pt",
        public_scope="observed committed updates; no interpolation or interval averaging",
        committed_update_count=sum(int(row.get("update", 0)) > 0 for row in history),
        exported_update_count=len(rows), exported_update_indices=[row["update"] for row in rows],
        retention_reasons=reasons, attempt_events="trial.attempts; canonical numeric observations carry no attempt-dependent status",
        query_plan_file="query_plan.json" if policy["compact_public_query_plan"] else None,
        query_plan_hash_contract="SHA256 of canonical plan payload via tdn.research.protocol.digest",
        whole_history_summary="trial.optimization_history_summary")
    plan_map = dict(schema="tdn.response-query-plans/v1", plans=plans,
                    source="exact journal history; immutable payload per digest")
    return rows, plan_map, contract
