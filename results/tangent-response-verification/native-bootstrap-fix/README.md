# Native worker bootstrap repair

On 13 October 2026, the user reported immediate failures for Tangent Response
audit job **2158**, decision job **2448** and report job **2449**, all with
`00:00:00` elapsed and exit `1:0`. The older Response trial job **2080** ran for
14 minutes 37 seconds; its cause cannot be inferred from this scheduler table.

The Tangent wrapper had a reproducible bootstrap defect. `sbatch` copies a
submitted script into Slurm's spool directory before execution. The wrapper
derived its checkout root from that copy's `BASH_SOURCE` path and attempted to
source `<spool>/scripts/common.sh`. It failed before `srun`, Python, the native
CUDA roster or any science. The earlier transport tests covered Raven's captured
coordinator but missed Slurm's copied worker entry point.

Worker modes now use the controller's matching, canonical absolute
`TDN_PROJECT_ROOT` and `TDN_REPO_ROOT`. They source and invoke the original
checkout's scripts. Ordinary controller root selection is unchanged. Missing,
different, relative or symlink-aliased roots fail explicitly; actual job/step,
desktop-Slurm mode, source/software/profile and allocation checks remain intact.

## Verification

The [before/after receipt](reproduction.json) executes the original `9bbc5b8`
and patched wrapper copies in private shell transport fixtures:

- All three original audit/decision/report batch copies exit **1** with missing
  spool `scripts/common.sh`, matching the reported immediate failure signature.
- All three patched copies exit **0**, passing through `srun` and the original
  task path to a Python argument recorder.
- Invalid roots, missing jobs/steps and incorrect execution modes are rejected.
- **281 affected checks passed**: 185 launcher/workflow checks plus 96 shared
  runtime/storage/reporting checks, with no failures, errors or skips. Exact
  [launcher/workflow](workflow-tests.xml) and [shared-path](shared-tests.xml)
  JUnit receipts are retained.

The fixture's scheduler/Python targets are recorders. No production allocation
verifier, numerical engine or actual native CUDA workload runs in these
transport tests. Actual Fedora execution still needs a fresh campaign.
The previous numerical artifacts and CPU gates remain unchanged and
implementation-only; no new numerical experiments ran for this repair.

## Native next step

Inspect `status latest --json` and the exact Slurm logs. Wait until the failed
campaign's coordinator and submitted jobs are terminal before updating that
checkout. Preserve its directory; a source change prevents exact recovery.
After pulling the repair, use the [runbook](../../../docs/TANGENT_RESPONSE_RUNBOOK.md)
to launch a fresh native campaign. Omitting `--run-id` generates a new ID.
Previously submitted jobs retain their old Slurm script snapshots.

For the older Response trial, retain
`runs/<run-id>/logs/trial-9b2cb697421ddc9a-000-2080.{err,out}` and its worker state.
Its actual traceback is needed to distinguish training, checkpoint reload,
software identity, resource or other failures. This bootstrap repair does not
claim to diagnose or repair that separate trial.
