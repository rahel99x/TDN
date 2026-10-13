# Exact old-batch cancellation helper

The user requested cancellation automation for the old Response batch containing
failed job 2080, while newer Tangent work and unrelated `samso`/interactive jobs
remain active. The supplied queue shows 74 old Response allocations still pending
or held. Actual native metadata and Slurm access are unavailable in this cloud
checkout; **no native job was cancelled here**.

The [standalone helper](../../scripts/cancel_response_batch.py) selects one exact
recorded Response run, checks every targeted live identity before any action,
issues only an explicit-ID `scancel`, and verifies terminal allocation accounting.
Original run artifacts and all non-selected jobs remain untouched. Source updates
do not prevent cancellation, and a detached `.runtime` copy avoids another source
update while newer jobs are running. A zero-target repeat reports
`NO_ACTIVE_JOBS`, without reclassifying historical science or missing accounting.

All **36 focused CPU tests passed**, with zero failures, errors or skips. They
use private scheduler recorders and exercise exact old-batch selection, unrelated
job preservation, full-batch ownership guards, malformed/duplicate metadata,
source updates, held/requeued jobs, idempotence, actual local controller locks,
post-cancellation accounting failures and original evidence preservation.
The shared lock covers preflight/cancellation and releases before polling.
Read-only independent review found no remaining blocking issue.

See the [original JUnit receipt](tests.xml), [verification record](verification.json)
and [native instructions](../../docs/RESPONSE_BATCH_CANCELLATION.md). These are
operational software checks, not actual Slurm/GPU or scientific validation.
Existing numerical/model/protocol sources and historical experiment outputs are
unchanged. Cancellation neither archives nor restarts the batch.
