"""Adjacent context; reuse the audited evidence/check schema unchanged."""
from __future__ import annotations

import json

from tdn.analysis.frontier.core import Context as BaseContext
from tdn.analysis.frontier.core import check, clean, score_checks, validate_row, write_reviews
from .protocol import digest


class Context(BaseContext):
    def __init__(self, protocol, stage, path, prerequisites, device, budget, *, resume=False,
                 stage_failures=None):
        super().__init__(protocol, stage, path, prerequisites, device, budget)
        self.unit = protocol["units"][stage]
        self.resume = resume
        self.stage_failures = stage_failures or {}
        if resume and (self.path / "rows.jsonl").exists():
            for line in (self.path / "rows.jsonl").read_text().splitlines():
                if not line:
                    continue
                row = validate_row(json.loads(line))
                if row["protocol_sha256"] != digest(protocol) or row["stage"] != stage:
                    raise ValueError("Interrupted ledger belongs to another protocol or unit")
                if row["experiment_id"] in self.identities:
                    raise ValueError("Interrupted ledger repeats an experiment identity")
                self.rows.append(row)
                self.identities.add(row["experiment_id"])
