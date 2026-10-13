# Response diagnostic runbook

This separate workflow investigates the completed Response pilot's unresolved
trainability result. It uses fresh native calibration and new immutable run IDs.
Preserve `response-pilot-20261012T075512819773079`, its original gate, and all
earlier campaigns. Read [the diagnostic research contract](RESPONSE_DIAGNOSTIC_RESEARCH.md)
before interpreting a result.

Native scientific work uses the configured Fedora Slurm profile, the project's
standalone Python `.venv`, actual RTX 4090 allocations, FP32 models with TF32 off,
and FP64 teachers. Each allocation remains at most 35 minutes; CPU work uses at
most eight cores and 48 GiB, GPU work at most four CPUs and 32/48 GiB with the
existing 18 GiB/75% soft VRAM and 90% hard thresholds. The controller freezes
requested updates and allowances from measured native workload. There is no
arbitrary summed campaign ceiling, pending-job cap, or cancellation of unrelated
projects.

## Run the native sequence with one command

From the configured Fedora checkout, use Raven's existing-script launch contract:

```bash
cd /home/rahel/TDN
raven launch scripts/response_diagnostic_campaign.sh \
  --local --project "$PWD" --recipe response-diagnostic
```

`--local` runs the lightweight foreground coordinator on the submission host.
It submits every scientific stage through the existing native Slurm workflow;
it does not run the numerical campaign locally. Raven's default launch mode is
an outer `sbatch` allocation, which is inappropriate for this coordinator and
is rejected. The opt-in recipe selects Experiment, Evidence and Artifacts,
without collecting coordinator GPU traces, CPU telemetry or unrelated research
panels. Explicit Raven CLI choices take precedence over the recipe.

Place campaign arguments after `--`, preserving Raven's argument boundary:

```bash
raven launch scripts/response_diagnostic_campaign.sh \
  --local --project "$PWD" --recipe response-diagnostic \
  -- --targets fd_same_grid galerkin_same_grid galerkin_projected_fine
```

Add `--check` before that boundary for Raven's read-only launch preview. The
preview does not execute the campaign or verify native CUDA readiness. Direct
`bash scripts/response_diagnostic_campaign.sh` remains supported.

Raven executes an immutable captured copy of the shell script. The campaign
validates Raven's original-script and launch-plan context to locate the actual
TDN checkout; it continues executing the captured bytes. Source and venv
fingerprints still guard the native scientific workers. Raven owns the outer
coordinator's start/finish records; TDN owns each allocated worker's lifecycle.
The outer `RAVEN_*` context is removed before native worker submission.

The coordinator retains its own stdout/stderr under the Raven attempt's
`logs/`, live infrastructure progress in `metrics.jsonl`, and small copies of
the campaign receipt, research paths and final decision under `outputs/`.
Its job-completion counters describe infrastructure, not scientific passing
results. Full scientific curves, failed/censored fits and query metrics remain
in the exact child research projects described below. The coordinator has no
invented Slurm job ID. Register `/home/rahel/TDN` in Raven and select the printed
`launch-...-a1` attempt to inspect this outer record.

The foreground coordinator prints two new run IDs and a project-contained
campaign receipt under `runs/.response-diagnostic-campaigns/`. It saves that
receipt before planning or submitting jobs, then performs this sequence:

1. Plan and submit `diagnostic-smoke`, including the declared variants' 64-update
   native calibration opportunities and mandatory native CUDA checks.
2. Poll actual scheduler/worker status every 30 seconds until submitted jobs are
   terminal, allowing the final `afterany` report to finish.
3. Validate sealed smoke artifacts and export its review evidence.
4. Plan and submit `diagnostic`, explicitly bound to that fresh smoke calibration.
5. Wait for terminal jobs, validate, export the review package, and save the
   separate scientific decision JSON.

With all three default targets, the declared roster contains **34 architecture
and training arms**. Five source/rollout/consistency arms apply only to same-grid
targets, producing **97 native calibration trials at 64 updates each**. Every
GPU allocation runs the mandatory **97-case CUDA suite**: the original 54 cases
plus 43 diagnostic cases. Completing these cases verifies their tested native
paths; it does not establish trainability or scientific superiority.

The default full diagnostic has **669 trials and 1,338,000 requested logical
updates**. Its seven primary arms each receive three paired seeds, six
individual-regime fits, and one joint fit per target. Exploratory arms receive
one seed and joint, low-mean, and large-amplitude fits. Each full trial defaults
to 2,000 updates. Full-parent updates process every declared training query with
equal parent weight, so example/forward/backward counts exceed logical-update
counts and are reported separately. Changing targets or work overrides changes
these totals; continuation job counts follow the measured allowances.

Keep this process in a persistent terminal session. Closing it does not cancel
submitted jobs. The receipt, plans, status snapshots, validation output, and
decision remain in the checkout. The script never automatically retries an
ambiguous or failed submission and never replaces native execution with CPU.
Source and venv changes invalidate the active calibration/recovery contracts.

Keep that checkout and venv unchanged until all related workers and the
coordinator are terminal. A running Python parent keeps imported code, while a
fresh checkpoint-reload child reads files from disk. Job 2080 demonstrated
unequal runtime schemas despite exact output/state/checkpoint parity; see the
[incident receipt and diagnosis](../results/response-runtime-reload-2080/README.md).
Reload now checks the engine's frozen source fingerprint in both processes and
rejects source changes explicitly. After a code fix, preserve the failed attempt
and launch a fresh campaign with new IDs and calibration. Do not repair a
receipt by filling missing settings, or rerun a probe in the original directory.

A negative scientific decision can finish with verified artifacts and a
`COMPLETED` coordinator receipt. This is complete evidence of that declared
experiment, not a scientific passing result. Failed jobs, missing units, invalid
seals, malformed status, or failed export produce a nonzero coordinator exit.
The review export is attempted after terminal jobs even when infrastructure
failed, preserving the available final report.

Choose a target subset or declare work overrides explicitly:

```bash
raven launch scripts/response_diagnostic_campaign.sh \
  --local --project "$PWD" --recipe response-diagnostic \
  -- --targets fd_same_grid galerkin_same_grid galerkin_projected_fine \
  --updates temporal=2000 \
  --trial-seconds temporal=7200 \
  --chunk-seconds 1500
```

These override values illustrate syntax, not recommended extra work. Both full
diagnostic plan and submission receive the same overrides; native calibration
retains its declared 64-update opportunities. The printed plan is the authority
for actual trial counts, variants, work, resource requests, and allowance ceilings.
The allowance is a ceiling, not a runtime prediction or evidence of optimality.

## Archive a cancelled attempt before a fresh launch

If a previous campaign was cancelled, allow its foreground coordinator to
finish or interrupt it with Ctrl-C, then use the separate
[archive command](RESPONSE_DIAGNOSTIC_ARCHIVE.md). It verifies actual terminal
jobs and preserves the exact runs and receipt in a project-contained archive.
It never deletes checkpoints, cancels jobs, or clears another program's runs.
The Raven launch above always creates new IDs and new calibration; partial
training from the archived attempt is not resumed or used for calibration.

## Inspect or run individual steps

The native wrapper defaults `plan`/`run` without a profile to
`--diagnostic-smoke`. It uses a separate latest pointer,
`runs/.fedora-response-diagnostic-latest.json`. Earlier Response pointers remain
separate. Choose fresh IDs for explicit launches:

```bash
bash scripts/fedora_response_diagnostic.sh doctor

bash scripts/fedora_response_diagnostic.sh plan --diagnostic-smoke \
  --run-id response-diagnostic-smoke-a \
  --targets fd_same_grid galerkin_same_grid galerkin_projected_fine
bash scripts/fedora_response_diagnostic.sh run --diagnostic-smoke \
  --run-id response-diagnostic-smoke-a \
  --targets fd_same_grid galerkin_same_grid galerkin_projected_fine

bash scripts/fedora_response_diagnostic.sh status response-diagnostic-smoke-a
bash scripts/fedora_response_diagnostic.sh validate response-diagnostic-smoke-a
```

After the smoke verifies, use its literal run ID:

```bash
bash scripts/fedora_response_diagnostic.sh plan --diagnostic \
  --calibration response-diagnostic-smoke-a --run-id response-diagnostic-a \
  --targets fd_same_grid galerkin_same_grid galerkin_projected_fine
bash scripts/fedora_response_diagnostic.sh run --diagnostic \
  --calibration response-diagnostic-smoke-a --run-id response-diagnostic-a \
  --targets fd_same_grid galerkin_same_grid galerkin_projected_fine

bash scripts/fedora_response_diagnostic.sh status response-diagnostic-a --json
bash scripts/fedora_response_diagnostic.sh validate response-diagnostic-a
bash scripts/fedora_response_diagnostic.sh decision response-diagnostic-a
bash scripts/fedora_response_diagnostic.sh compact response-diagnostic-a --mode review
```

`status --json` separates scheduler state, worker status, artifact validation,
terminal status, computational completion, report completion, and scientific
outcome. A scheduler `COMPLETED` observation alone does not establish a valid
sealed result. `validate` independently checks artifacts. `decision` reads the
diagnostic decision; the historical pilot's `gate` is not substituted for it.

After interruptions, preserve both attempts and inspect the exact-source recovery
plan. Do not delete an existing run directory to reuse its ID:

```bash
bash scripts/fedora_response_diagnostic.sh recover response-diagnostic-a \
  --run-id response-diagnostic-recovery-a --plan
bash scripts/fedora_response_diagnostic.sh recover response-diagnostic-a \
  --run-id response-diagnostic-recovery-a
```

Recovery requires terminal original jobs, matching source/software/data, and
valid inherited seals. Both attempts count toward cost. Resolved trials do not
repeat their training merely because another continuation allocation exists.

## Open the Raven project and portable report

The native report publishes a Raven project under the diagnostic run:

```text
/home/rahel/TDN/runs/<diagnostic-run-id>/raven/
  .raven/config.json
  .raven/contracts/outputs.v1.json
  runs/<diagnostic-run-id>-analysis/outputs/index.html
  project_catalog.json
  page-000/
    .raven/config.json
    runs/<campaign-unit>/
      run.json
      summary.json
      logs.json
      metrics.jsonl
      outputs/science/
  page-001/
    ...
```

In Raven on Fedora, register that exact project root:

```text
:project /home/rahel/TDN/runs/<diagnostic-run-id>/raven
:runs
```

Unit attempts retain their actual Slurm job identities. The aggregate analysis
attempt has no synthetic job ID; select `<diagnostic-run-id>-analysis` explicitly
to inspect the combined scientific report. Campaigns with more than 255 declared
units put exact unit attempts in deterministic standalone page projects, with
at most **255 unit attempts per page**; the root project contains the overview.
`project_catalog.json` maps every declared unit to its page and attempt ID.
Register a detail page, such as
`:project /home/rahel/TDN/runs/<diagnostic-run-id>/raven/page-000`, to inspect its
actual job attempts. Smaller smoke runs may keep unit attempts directly in the
root alongside the overview. No scientific methods or attempts are discarded
to satisfy Raven's discovery bounds.

The original native job WorkDir is the TDN checkout, so explicit registration
is the dependable path for these nested report projects. Generated metrics are
evidence of their measured scopes; missing GPU traces, energy, rates, or scheduler
predictions remain unavailable.

The report also writes:

```text
runs/<diagnostic-run-id>/report/response-diagnostic-raven.zip
runs/<diagnostic-run-id>/report/response-diagnostic-raven.manifest.json
runs/<diagnostic-run-id>/report/raven_paths.txt
runs/<diagnostic-run-id>/report/diagnostic_summary.json
```

Unzip the portable bundle into a fresh directory. Its root is the Raven project,
including all detail pages with relative paths; register that extracted root using `:project`, or open
`runs/<diagnostic-run-id>-analysis/outputs/index.html` in a browser. The analysis
exports trial, endpoint-query, learning-curve, screening-query, and correction
CSV tables plus a metric legend. Keep these raw tables with the plots.
Native learning curves use the declared 40-update cadence and retain
initialization, completed validation and attempt-end observations. Full journals
preserve every optimizer update; whole-history summaries preserve gradient,
clipping and work statistics independently of plot sampling. Public query plans
are stored once in `query_plan.json`, with hash-verified links from their curve
observations. Compiled tables reference the complete copied source JSON for
initial query rows and fitted control matrices.
Scientific plots distinguish model families and controls and preserve failed,
unresolved, censored, and initialization-selected outcomes.

Raven's default artifact inspection has a shared 8 MiB read budget. Large
structured metrics or CSV exports can therefore show `not_checked`/`incomplete`
even when their declared files and bounds are valid. This describes bounded
inspection coverage; inspect the individual check before interpreting it as
corruption or a scientific failure. Raven's table previews are also bounded.
The TDN `validate` command checks complete sealed artifacts independently, and
the portable manifest checks every included file. Use the HTML report and
copied CSV/JSON for the complete research data; Raven's sample inspection does
not certify the whole scientific campaign.

Native Raven attempts start after actual allocation verification, before software
checks and GPU preflight. Their terminal state and exit code follow the complete
worker, including failures outside the science command. Science-body status and
timing remain separate from worker duration and scheduler allocation accounting.
Worker stdout/stderr snapshots refresh at finalization; live research metrics
remain in each attempt's `metrics.jsonl` and `logs/research.log`.

For review in this chat, upload the compact package's printed `index.json` and
every numbered archive part. A compact review contains an omission inventory;
it is not the full binary replay package. The portable Raven ZIP is an additional
inspection artifact, not a checkpoint freeze or confirmation authorization.

Reporting follows Raven's pinned
[project standard](https://github.com/rahel99x/Raven/blob/c34f0892d7bddc685c5af10489ecb114fd6adab6/docs/PROJECT_STANDARD.md)
and [V2 integration guide](https://github.com/rahel99x/Raven/blob/c34f0892d7bddc685c5af10489ecb114fd6adab6/docs/PROJECT_REPORTING_V2.md). The application
produces these files without requiring Raven to be installed on compute nodes.
Only TDN reporting files are changed; Raven itself is not modified.

## Reduced CPU implementation smoke

This verifies the new integration paths on CPU, using a separate fresh directory:

```bash
bash scripts/response_diagnostic_local.sh --smoke \
  --run-dir runs/response-diagnostic-local-a \
  --targets fd_same_grid
```

The CPU wrapper rejects native/full profiles, CARC use, injected scheduler
identity, paths outside the checkout, and existing run paths. It calls the
separate reduced CPU runner without inventing Slurm variables. CPU smoke cannot
provide native calibration, satisfy mandatory CUDA readiness, or establish
trainability/scientific benefit.
