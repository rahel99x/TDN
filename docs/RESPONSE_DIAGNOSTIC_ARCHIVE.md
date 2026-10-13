# Preserve a cancelled diagnostic before a fresh launch

The archive command tidies one Response diagnostic campaign by moving its
existing native calibration/full run roots and campaign receipt into a new
project-contained archive. It preserves every original byte, scientific
manifest, checkpoint, log, and failed or cancelled outcome. It never deletes
files, cancels jobs, rewrites scientific metadata, or resumes completed work.

Run it on the configured Fedora machine, outside a Slurm allocation, after
cancellation has become terminal in the actual scheduler **and** the campaign
coordinator has stopped. A cancellation request is not terminal evidence.
The command retains the existing checkout/user/profile/Conda policy checks.
Historical source fingerprints do not need to match today's code for this
metadata-only archival inspection.

First inspect the concrete move plan:

```bash
cd /home/rahel/TDN
bash scripts/response_diagnostic_archive.sh --plan
```

By default, this selects the most recently created campaign receipt under
`runs/.response-diagnostic-campaigns/`. Use its printed token to make the
execution selection explicit:

```bash
bash scripts/response_diagnostic_archive.sh --campaign CAMPAIGN_TOKEN --plan
bash scripts/response_diagnostic_archive.sh --campaign CAMPAIGN_TOKEN
```

Alternatively, `--run-id RESPONSE_DIAGNOSTIC_RUN_ID` resolves the unique campaign
receipt that records that native run. The receipt's exact smoke/full pair is the
scope; a missing full directory after partial submission is recorded as absent.
Manual runs without a coordinator receipt are not guessed into a campaign.
Both cancelled and completed evidence can be preserved; choose the intended
cancelled campaign explicitly when several campaigns exist.

No move occurs unless the receipt records a terminal coordinator status and
every recorded submitted allocation has an actual terminal scheduler state.
The helper checks live queue records, including campaign-tagged jobs accepted
before their IDs reached `jobs.json`. Missing accounting, unknown/pending/live
states, mismatching scheduler ownership, ambiguous untagged live Response jobs,
or a still-running coordinator block archival. It does not synthesize a
terminal receipt or use a stale status snapshot as proof.

Ownership checks use exact submitted allocation IDs, the actual user, original
project WorkDir, and expected unit job name. A nonempty scheduler comment must
match the frozen run/stage/profile identity. When accounting did not retain a
comment, a retained `scontrol` record can supply it; otherwise that coverage is
reported honestly as `NOT_RETAINED_BY_SCHEDULER`. Missing comment storage alone
does not require reconfiguring Slurm, but the remaining ownership and terminal
checks still apply. Batch/extern step records are not separate allocations.

The helper acquires the same controller lock used for native submission,
rechecks scheduler and metadata evidence immediately before moving, and uses
atomic no-overwrite renames into an exclusively created directory:

```text
runs/archived-response-diagnostics/<fresh-archive-id>/
  archive-manifest.json
  scheduler-proof.json
  native/<original-smoke-run-id>/...
  native/<original-full-run-id>/...       when submitted
  campaign/<original-campaign-token>/...
  pointers/.fedora-response-diagnostic-latest.json  when it names this pair
```

The bounded manifest maps every original path to its archived location and
records inspected metadata hashes, actual scheduler-state/comment coverage,
missing run roots, and retained Raven association. The diagnostic latest pointer
is moved only if it names this exact smoke/full pair. Other diagnostic pointers,
historical Response pointers, other programs' runs, review exports, caches,
the venv, and source files remain in place. The archive is a preservation
transaction, not a directory deletion or disk-space reclamation operation.

An exact `raven_launch` receipt binding additionally requires matching terminal
outer Raven `run.json` and `summary.json` records. Its local coordinator run,
launch intent, and immutable source snapshot remain at their original locations;
the archive manifest links that preserved association. An older direct-Bash
receipt without this binding does not cause any Raven run to be guessed or
moved. A recorded but missing, live, or mismatching association blocks archival.

Frozen native manifests retain their original absolute paths and calibration
identities after relocation. Inspect them as historical evidence; do not run
ordinary `status`, `recover`, or `resume` against archived paths. Internal
artifact symlinks, if any, are preserved without traversal and do not make the
archive a safe or complete replay package. A failed move retains the transaction
manifest and attempts no-overwrite restoration; any path that cannot be restored
stays recorded in the archive without replacing a newly created original path.

After archival, start the entire diagnostic with a **fresh calibration and new
run IDs** using the documented Raven launch command in the diagnostic runbook.
The new campaign never reuses the archived smoke's calibration or resumes an
old trial. Archival on the Fedora machine is established only by running this
command there and inspecting its `ARCHIVED` result; cloud shell tests cannot
establish that the user's local machine has been tidied.
