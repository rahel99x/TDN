# Advance research campaign: Fedora runbook

This is an independent, fresh-cohort campaign. It preserves portfolio runs and
uses `advance-workflow.json`, `fedora-advance-*` run identifiers and its own
`.fedora-advance-latest.json` pointer. Computational completion, mathematical
checks and scientific advantage remain separate outcomes.

The native target is `/home/rahel/TDN`, Python 3.13 in the existing project
`.venv`, a Ryzen 7800X3D, Slurm partition `local`, 110000 MiB scheduled host
memory and one RTX 4090 with 24 GiB dedicated VRAM. It reuses the configured
`.tdn/fedora-slurm.json`; no Conda, CARC account or Tower application change is
required. All files, temporary work and caches remain under this checkout.

## Submit and inspect

Do not update source, dependencies or the local Slurm profile while a workflow
is pending or running. Source and prerequisite fingerprints are frozen. Use
the project's venv through the launchers; activating it is optional.

```bash
cd /home/rahel/TDN
git pull --ff-only origin main

# Read-only resource/protocol plan. This never submits jobs.
bash scripts/fedora_advance.sh plan --smoke

# Small native end-to-end check, including real allocated CUDA tests.
smoke_id="fedora-advance-smoke-$(date -u +%Y%m%dT%H%M%S)"
bash scripts/fedora_advance.sh run --smoke --run-id "$smoke_id"
bash scripts/fedora_advance.sh status "$smoke_id"
bash scripts/fedora_advance.sh logs "$smoke_id" --lines 100
bash scripts/fedora_advance.sh paths "$smoke_id"
```

After all smoke jobs finish, run validation. It fails if any expected stage is
missing, incomplete or changed; it does not require a scientifically favorable
result. A scientifically BAD or NA experiment remains valid evidence.

```bash
bash scripts/fedora_advance.sh validate "$smoke_id"

# Only after successful validation: schedule the complete frozen experiment.
full_id="fedora-advance-full-$(date -u +%Y%m%dT%H%M%S)"
bash scripts/fedora_advance.sh plan --full --run-id "$full_id"
bash scripts/fedora_advance.sh validate "$smoke_id" && \
  bash scripts/fedora_advance.sh run --full --run-id "$full_id"

bash scripts/fedora_advance.sh status "$full_id"
bash scripts/fedora_advance.sh logs "$full_id" --lines 150
bash scripts/fedora_advance.sh paths "$full_id"
```

Keep the printed run ID for later shells. Explicit IDs avoid accidentally
collecting a later rerun. `latest` is also supported and selects only this
campaign. An optional `--development` profile is an additional bounded
development cohort, not a replacement for full confirmation.

The complete DAG schedules the audit; targeted numerical diagnostics;
training and validation references; three protected exploratory prototypes;
track/seed training units; a freeze; sixteen four-parent reference and
confirmation pairs; a whole-cohort aggregate; separately prepared scaling
references and measurements; and a final report. Science dependencies use
`afterok`. Independent branches wait for earlier allocations with `afterany`
to share the desktop without inheriting unrelated scientific failures. The
report runs after any terminal outcome and retains missing/failed branches.

Every GPU unit performs a preflight and the entire declared CUDA suite. Its
JUnit checker requires the exact unique case identities with no missing,
failed, duplicated or skipped cases. CPU tests and scheduler stubs do not
establish GPU readiness.

## Resource bounds and interpretation

| Profile | Jobs | GPU jobs | Longest allocation | Sum of allocation ceilings |
|---|---:|---:|---:|---:|
| Smoke | 15 | 4 | 10 minutes | 2 hours 30 minutes |
| Development | 29 | 11 | 20 minutes | 8 hours 10 minutes |
| Full | 49 | 23 | 35 minutes | 25 hours 50 minutes |

These sums are conservative reservation ceilings, **not predicted runtimes**;
actual elapsed work is recorded. No individual allocation exceeds 35 minutes
or the established 45-minute maximum. Numerical work is capped at 30 minutes
per full unit, leaving time for checks, sealing and reporting. CPU jobs use at
most eight CPUs and 48 GiB; GPU jobs use four CPUs, 32/48 GiB and one actual
allocated GPU. The existing 18 GiB/75% soft and 90% device-use hard VRAM
limits remain in force. Host RAM is not additional GPU VRAM. There is no
pending-job cap, automatic budget expansion or automatic retry.

Use `status` to distinguish runtime completion from scientific GOOD/BAD/NA.
Read the report's coverage and failure counts before comparing errors or
latency. A failed/missing method never becomes a zero-cost winner. Reference
uncertainty, missing intervals and unavailable monetary/energy rates remain
explicitly unknown.

## Analytics and review export

`paths` prints the report figures/tables and each allocation's exact
`TDN_TOWER_DIR` and `TDN_TOWER_METRICS` path. The generic Tower sidecar links
the original scientific artifacts and figure manifest; it does not duplicate
every scalar into enormous CSV tables. Tower itself is unchanged.

```bash
bash scripts/fedora_advance.sh paths "$full_id"
bash scripts/tower.sh list

# After all allocations finish: check science seals and record allocation costs.
bash scripts/fedora_advance.sh validate "$full_id"
bash scripts/fedora_advance.sh snapshot-accounting \
  --workflow "runs/$full_id/advance-workflow.json"

# Smaller, faster review package, with retained-file hashes and upload parts.
# Gzip level 1 is the default; avoid XZ-9 for an interactive export.
bash scripts/fedora_advance.sh compact "$full_id"
```

Inspect `runs/$full_id/state/scheduler-accounting.json`: complete accounting
requires `status` equal to `RECORDED` and `all_allocations_terminal` equal to
`true`. The snapshot command can exit successfully while `sacct` is unavailable
or incomplete; if accounting lags, repeat it later. Allocation and step rows
are distinct and must not be added together. For recovery, refresh the origin's
accounting too if its stored snapshot is incomplete. Keep exports available even
when validation fails, so the failure evidence can be reviewed.

Optional `bash scripts/fedora_advance.sh collect "$full_id"` retains checkpoints,
arrays, logs, references, failures and figures in a larger replay archive, and
also snapshots the exact submitted jobs with `sacct`. It is not required before
the compact export. `compact` is a read-only review
subset: it omits replay tensors/checkpoints and rendered images, while
retaining raw measurements, chart data, source inventories and omission
hashes. Keep the original run to reproduce predictions. For recovered runs,
both exporters include the verified origin chain and its failed work.

## Recover interrupted work without erasing it

Recovery requires all recorded origin jobs to be terminal and the same
source, protocol, software and hardware profile. It creates a new coordinator,
reuses only verified completed stages and copies compatible interrupted
train/confirmation journals to new stage directories. It preserves the origin
byte-for-byte. Unjournaled stages restart in the new coordinator. If those
identities changed, retain the old run and start a fresh experiment instead.

```bash
bash scripts/fedora_advance.sh recover "$full_id" --plan
recovery_id="fedora-advance-recovery-$(date -u +%Y%m%dT%H%M%S)"
bash scripts/fedora_advance.sh recover "$full_id" --run-id "$recovery_id"
bash scripts/fedora_advance.sh status "$recovery_id"
bash scripts/fedora_advance.sh logs "$recovery_id" --lines 150
```

## Local CPU implementation checks

On a separate cloud/development checkout, outside Slurm, the following runs
the small scientific DAG on CPU. It refuses full confirmation, CARC paths,
inherited scheduler identities and requested CUDA substitution.

```bash
bash scripts/advance_local.sh --smoke
# Optional bounded CPU development, with a separate data cohort:
bash scripts/advance_local.sh --development
```

Local CPU verification does not certify the desktop CUDA runtime, native
timings or scientific superiority. Native results require the actual Slurm
workflow above.
