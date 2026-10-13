# Cancel one old native Response batch

Use [the standalone helper](../scripts/cancel_response_batch.py) for a literal
Response run ID. It cancels only the active allocations belonging to that run,
preserves all original run/checkpoint/log/journal files, and records what happened
under `.runtime/response-cancellations/`. It does not restart, archive or reclassify
scientific work. Tangent Response, interactive allocations and unrelated programs
remain outside the selected batch.

The old run discussed in the job-2080 incident is
`response-diagnostic-smoke-20261013T032059-fd157c33d389`. The posted queue shows
74 remaining Response allocations pending or held; the newer Tangent batch is
progressing separately. Job-number ranges and truncated queue names are not
ownership evidence and are never used as cancellation filters.

## Run while another campaign remains active

Keep the active checkout and venv unchanged. Copy this independent,
standard-library helper into `.runtime`; it imports no TDN/Raven modules and
does not require today's source to match the historical batch. Do not pull the
repository merely to obtain the helper while other TDN jobs remain active.

These Bash commands fetch Git objects without checking out files, copy the
delivered helper into `.runtime`, and verify its exact bytes. Existing Git
authentication works for public and private repositories:

```bash
cd /home/rahel/TDN
mkdir -p .runtime
git fetch origin main
git show FETCH_HEAD:scripts/cancel_response_batch.py > .runtime/cancel_response_batch.py
printf '%s  %s\n' \
  9e9afc62496a76a6cda6e6b007810784a0644ab7b02de53b4baeb2f592c5cacc \
  .runtime/cancel_response_batch.py | sha256sum --check -
```

Stop if the fetch, copy or checksum verification fails. The checksum identifies
this delivered version; future helper edits need their own matching checksum.
A commit-pinned `git show` reference is provided with each published delivery.

Read-only preview:

```bash
.venv/bin/python .runtime/cancel_response_batch.py \
  --project "$PWD" \
  --run response-diagnostic-smoke-20261013T032059-fd157c33d389 --plan
```

Cancel that exact batch and wait up to 120 seconds for terminal accounting:

```bash
.venv/bin/python .runtime/cancel_response_batch.py \
  --project "$PWD" \
  --run response-diagnostic-smoke-20261013T032059-fd157c33d389 --execute
```

`--execute` needs no interactive confirmation. The default mode is a plan if
neither mode flag is supplied. Inspect the printed JSON and retained receipt.
Repetition checks live ownership again and sends no cancellation for jobs that
have already left the queue. With no active selected jobs it reports
`NO_ACTIVE_JOBS`, without inferring historical accounting or scientific success.

## What the helper verifies

- The exact native Response workflow, frozen profile, finished submission and
  unique stage/allocation registry belong to the selected root and actual user.
- Every selected live allocation has the exact recorded ID, username/UID,
  checkout working directory, complete stage name and frozen batch comment.
  All selected identities are checked before any cancellation.
- The shared controller lock excludes an in-progress submission during the
  brief preflight/cancellation window. The helper releases it before polling.
  A busy lock, unfinished submission, matching-run unrecorded allocation,
  metadata change, missing comment or ambiguous scheduler data aborts.
- One `scancel` call receives only those verified positive allocation IDs.
  It never uses user-wide, name-prefix or range cancellation.
- Requested IDs must disappear from the live queue and have exact terminal
  `sacct` allocation records before reporting `TERMINAL`. Steps are not mistaken
  for allocations. `SPECIAL_EXIT` is a requeued/held state and remains active.

A command failure, missing/nonterminal accounting or timeout leaves an
`UNVERIFIED` receipt and a nonzero exit; cancellation may already have occurred.
Inspect the outstanding IDs and scheduler before treating the batch as stopped.
There is no automatic retry or rollback of cancelled work.

The helper does not signal the outer Raven coordinator. An old diagnostic
coordinator waiting on this smoke cannot promote a batch with cancelled units;
its existing verification path exits with failure and retains available evidence.
An in-progress submission must finish or be stopped before cancellation.
Do not archive or pull while a related coordinator is still active. Other TDN
campaigns sharing the checkout must also be terminal before a source update.

Use the [diagnostic runbook](RESPONSE_DIAGNOSTIC_RUNBOOK.md) to evaluate exact-source
recovery afterward, or the [archive command](RESPONSE_DIAGNOSTIC_ARCHIVE.md) once
its actual terminal ownership checks pass. Cancellation itself performs neither.
CPU tests use private scheduler recorders; they do not constitute native Slurm
execution, GPU readiness or scientific evidence.
