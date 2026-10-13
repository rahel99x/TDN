# Tangent-response native runbook

This separate gated program tests a response around a fixed smooth anchor. Read
[the research handout](TANGENT_RESPONSE_RESEARCH_HANDOUT.md) for its prospective
criteria and [the analysis guide](TANGENT_RESPONSE_ANALYSIS.md) for interpretation.
Preserve earlier Response experiments, source contracts, calibration and results.
Computational completion, a passing software test and a passing scientific gate
are separate observations.
See the [implementation overview](TANGENT_RESPONSE_RESEARCH.md) and
[verification record](../results/tangent-response-verification/README.md) for
the delivered interfaces, completed checks and pending native evidence.

The [uploaded native FD pilot](../results/native-tangent-compression-20261013/README.md)
has A `PASS`, B `STOP`, and two reporting failures. Preserve that original run
and its source-bound decisions. It does not authorize learned calibration or
training; reporting repair does not change a scientific gate or make an old
failed worker complete. Use the review's conditioning/state diagnostics before
proposing further numerical work.

The configured Fedora checkout uses its own standalone Python `.venv`, actual
Slurm allocations and the RTX 4090. CPU jobs request at most eight cores and
48 GiB; GPU jobs request four cores, 32/48 GiB and one GPU. Allocations are at most
35 minutes, with at most 30 minutes of declared science work plus cleanup
headroom. Keep the existing 18 GiB/75% soft VRAM and 90% hard limits, FP64
teachers/path/subtraction, FP32 models/deployment and TF32 off. No aggregate
campaign time ceiling, pending-job cap, driver change, automatic retry or silent
CPU fallback is introduced.

Keep the source and venv unchanged while related workers or their coordinator
are active. The source fingerprint covers executable code across TDN, including
the earlier Response program; updating another program can invalidate this
campaign too. Wait for terminal jobs before pulling fixes, preserve the old
attempt and use a fresh campaign when source changes. The
[Response reload incident](../results/response-runtime-reload-2080/README.md)
illustrates why running parents and newly started interpreters must share the
same frozen source.

## Verify the reduced CPU implementation

Run outside any Slurm allocation in a checkout with its own standalone `.venv`:

```bash
bash scripts/tangent_response_local.sh --profile cpu-smoke \
  --target fd_same_grid --run-id tangent-response-cpu-a \
  --worker-telemetry cpu,memory

bash scripts/tangent_response_local.sh --profile cpu-smoke \
  --target galerkin_same_grid --run-id tangent-response-cpu-galerkin-a \
  --worker-telemetry cpu,memory
```

Run them sequentially; the local lock prevents overlapping CPU rehearsals.
Each reserves a fresh directory and exercises the reduced complete program,
including both learned arms' two declared updates when labels are accepted.
`runs/<id>/local_status.json` records computational coverage; its separate
`decision/decision.json` retains scientific outcomes under `IMPLEMENTATION_ONLY`.
Open `runs/<id>/raven/runs/<id>-analysis/outputs/index.html`, or extract its
`report/tangent-response-raven.zip` for portable inspection. Existing directories
are preserved and cannot be reused for a fresh run. These commands cannot
establish native readiness or authorize native downstream stages.

Both local and native interfaces accept `--sampling-seconds 5` and
`--no-worker-telemetry`; the frozen protocol records the choice and any missing
coverage. The default native channels are CPU, memory and GPU.

## Launch a fresh native pilot

From the configured native checkout:

```bash
cd /home/rahel/TDN
bash scripts/fedora_tangent_response.sh doctor

bash scripts/fedora_tangent_response.sh plan --profile pilot \
  --target fd_same_grid --run-id tangent-response-pilot-a

raven launch scripts/tangent_response_campaign.sh \
  --local --project "$PWD" --recipe tangent-response \
  -- --profile pilot --target fd_same_grid \
  --run-id tangent-response-pilot-a \
  --worker-telemetry cpu,memory,gpu
```

The plan is read-only: it neither reserves an ID nor submits jobs. It prints the
initial phase's actual resource requests and identifies downstream allocations
as unavailable until their gates and measured calibration authorize them.
Choose a fresh campaign ID. Omitting `--run-id` generates one and prints it;
use that literal ID for later inspection. Each eligible phase receives a distinct
fresh child ID, written to the receipt before submission. The separate pointer
is `runs/.fedora-tangent-response-latest.json`.

Raven runs the lightweight coordinator locally. Its default outer Slurm launch
mode is inappropriate here and the shell rejects it. Campaign arguments follow
`--`; `--check` before that boundary previews the Raven launch without executing
science. Direct `bash scripts/tangent_response_campaign.sh --profile pilot
--target fd_same_grid` is supported. Raven executes its captured shell bytes;
the validated original project context selects the actual TDN checkout.
Native workers receive no outer `RAVEN_*` environment or synthetic job IDs.
Raven owns the outer attempt, while TDN owns each allocated worker.
Slurm executes a private copy of the batch script from its spool directory.
Worker modes therefore use the controller's matching, canonical absolute
`TDN_PROJECT_ROOT` and `TDN_REPO_ROOT`, then invoke the original checkout's
task script. Their actual allocation, step, workflow, source and software checks
remain mandatory. Controller commands still derive their root from the original
script location.

The additive `tangent-response` recipe in `.raven/automation.json` selects
Experiment, Evidence and Artifacts. It does not collect misleading coordinator
GPU/CPU traces. Worker telemetry is independently declared in the frozen TDN
protocol. Raven's explicit CLI settings take precedence over its recipe.

## Follow the gates

The receipt lives at
`runs/.tangent-response-campaigns/<campaign-id>/receipt.json`; an exact copy is
retained in the parent run. The foreground coordinator polls real scheduler and
worker status every 30 seconds. Keep it in a persistent terminal session.
Closing it preserves submitted jobs; it does not cancel them or authorize a
second coordinator to submit more work.

| Phase | Evidence and promotion requirement |
| --- | --- |
| `mechanism` | Native audit, fresh mechanism parents, independent references and matched mean/spatial controls; literal A `PASS` and eligibility are required for B. |
| `compression` | Declared source-direction actions, training-only response factors and held-out action/oracle-gain checks; literal B `PASS` is required for fresh fit work. |
| `fit_prepare` | Fresh, parent-disjoint fit candidates, accepted oracle labels and a visible screening inventory; literal C0 `PASS` is required for learned calibration. |
| `calibration` | Both declared learned arms execute 64 actual complete CUDA updates on the exact fit bank/layout/gradients; measured workload hashes and literal calibration `PASS` bind learning allowances. |
| `learning` | Exact committed work, validation selection and distinct best-training/initial/final checkpoints; literal C `PASS` authorizes utility. |
| `utility` | Actual complete inference and classical comparisons at declared tolerances; D reports the resulting eligible scientific outcome. |

Every GPU allocation runs this program's exact declared native CUDA roster.
The authoritative cases and files are in its sealed protocol; earlier Response
54/97-case counts are not substituted. Missing, skipped, duplicated or failing
required cases prevent native readiness. CPU smoke exercises the implementation
and records `IMPLEMENTATION_ONLY`; it cannot promote a native gate or provide
native calibration.

The default pilot trains **one paired seed, two learned arms and 300 committed
updates per arm**. Development uses three paired seeds and 1,000 updates per
arm, after a literal eligible pilot binding. A full-parent optimizer update
processes the declared physical objective; query/forward/backward counts and
allocated chunks differ from logical committed-update counts. Work-based chunks
continue the exact generated journal across allocations. Calibration freezes
allowance ceilings; it does not predict scientific success. The planned roster,
accepted/rejected slots, updates and all-attempt costs remain in the protocol
and reports.

`STOP` and `UNRESOLVED` retain a verified completed phase report and halt further
expansion. Unrun phases are explicitly listed in `not_submitted`. Failed jobs,
incomplete workers, invalid seals, ambiguous submission or failed export produce
a nonzero coordinator exit. The final report uses `afterany`, including after a
partially submitted phase, so available failure evidence remains visible.
Neither missing evidence nor a negative scientific result is relabeled as a
passing gate.

## Inspect, validate and export

```bash
bash scripts/fedora_tangent_response.sh status tangent-response-pilot-a --json
bash scripts/fedora_tangent_response.sh validate tangent-response-pilot-a
bash scripts/fedora_tangent_response.sh decision tangent-response-pilot-a
bash scripts/fedora_tangent_response.sh paths tangent-response-pilot-a
bash scripts/fedora_tangent_response.sh compact tangent-response-pilot-a --mode review
```

The parent resolves only its exact recorded children. The same commands accept
an exact child ID. Status distinguishes scheduler states, worker status, sealed
artifact presence, terminal jobs and computational completion. Validation
independently checks scientific bytes plus the final worker seal, source,
protocol, profile, software and telemetry/report inventory. A raw scheduler
`COMPLETED` state alone does not establish valid evidence. Historical read-only
status does not compare an old source fingerprint with today's source; execution,
calibration binding and recovery enforce the original source/software contract.

Each child has a standalone `runs/<child-id>/raven` project. Register its absolute
path with Raven using `:project /home/rahel/TDN/runs/<child-id>/raven`. The overview
attempt `<child-id>-analysis` contains source-traced tables, plots, definitions,
missing methods and negative gates. Larger rosters use `page-000`, `page-001`,
etc., each with at most 255 attempts. The root overview may contain no detailed
worker attempts; `project_catalog.json` maps every declared unit to its exact
page and attempt. Register a page to inspect the actual unit job binding.

The overview's `artifact_projects.json` separately lists `artifact-pages/page-000`,
etc. Register those exact relative projects to inspect lossless chunks of large
scientific tables and original source documents. Their completed artifact
attempts obey Raven's standard 8 MiB reader budget and project attempt limits.
`tangent_research.json` maps compiled rows and raw source segments to these
chunks; the HTML overview links them directly. Worker pages describe actual
allocations, while artifact pages carry data. Both page families are included
in the portable zip.

Declared compression aggregates larger than a page are verified against their
science seal and retained once as ordered source segments. Worker pages carry
small provenance descriptors that explicitly identify the original native-only
source; they do not promise a completed report. The producer page limit remains
7 MiB, Raven's default reader limit 8 MiB, generic metadata 256 MiB and portable
transport 2 GiB. The compression-only scientific payload limit is 1 GiB.
Incomplete attempts with missing declared content are preserved in the overview's
`incomplete_attempts` evidence section and excluded from live portable discovery.
Their original inventory bytes and native failure states remain visible.

The report unit produces `report/tangent-response-raven.zip` and
`report/tangent-response-raven.manifest.json`. Extract the zip to any directory
and register its root, or open `runs/<child-id>-analysis/outputs/index.html` in a
browser. Portable paths are relative. Exact teachers/checkpoints/journals remain
in the native run and the omission inventory retains their hashes. Compact
`review` export also preserves research metadata and source-traced inventories;
`replay` includes binary replay artifacts. Neither review transport nor a browser
report provides the omitted binary state needed for exact resume.

## Recover exact work or reconcile a stopped coordinator

Preserve all attempts. First inspect native jobs and the original coordinator.
Never start continuation while the original coordinator is alive. If it died
between phases, verify that all submitted jobs and the exact associated Raven
outer attempt are terminal, then reconcile its receipt:

```bash
bash scripts/fedora_tangent_response.sh reconcile tangent-response-pilot-a --plan
bash scripts/fedora_tangent_response.sh reconcile tangent-response-pilot-a
```

For a failure before Python starts, the Slurm `logs/<stage>-<job-id>.err` and
`.out` files are authoritative even when no worker state or Raven attempt exists.
A `00:00:00` failure is insufficient evidence of a model or CUDA problem.
The [worker bootstrap repair](../results/tangent-response-verification/native-bootstrap-fix/README.md)
documents the original copied-script path failure and its transport regression.
After any source fix, preserve the failed campaign and start a fresh campaign
once its submitted jobs and coordinator are terminal. Pulling updated code does
not replace Slurm's already captured script or authorize recovery under a changed
source fingerprint. Omit `--run-id` on the new Raven launch to generate a fresh
ID; the campaign's prospective scientific gates remain in force.

For the uploaded A PASS/B STOP pilot, the reviewed scientific measurements are
already sufficient to stop expansion. Its reporting-only repair does not require
a fresh numerical campaign, validate exact recovery under changed source, or
retroactively seal jobs 2891/2893. Read the saved review and verification receipts;
freeze a bounded compression diagnosis before proposing additional numerical work.

If exactly one child failed, the parent recovery command resolves that child and
prints the literal replacement phase. Its execute form has the same guards:

```bash
bash scripts/fedora_tangent_response.sh recover tangent-response-pilot-a \
  --run-id tangent-response-recovery-a --plan
bash scripts/fedora_tangent_response.sh recover tangent-response-pilot-a \
  --run-id tangent-response-recovery-a
```

For ambiguous or multiple failures, select the explicit failed child ID instead.
Recovery verifies terminal actual allocations and ownership, the unchanged
source/profile/software/protocol/data, valid completed-unit seals and any exact
interrupted training/calibration journal. It creates a fresh recovery root,
retains all origin paths and submits only eligible unfinished units plus new
decision/report observers. It never resumes a completed unit. A source change,
unsealed completed unit or ambiguous journal requires diagnosis and a fresh
campaign rather than a permissive resume.

After the recovery's real jobs finish and its artifacts validate, continue the
stopped parent explicitly, substituting the printed phase and exact recovery ID:

```bash
bash scripts/fedora_tangent_response.sh validate tangent-response-recovery-a
bash scripts/fedora_tangent_response.sh continue --profile pilot \
  --target fd_same_grid --run-id tangent-response-pilot-a \
  --replacement mechanism=tangent-response-recovery-a
```

Without a failed child, `continue` can reuse the completed recorded phases and
advance an interrupted coordinator. It verifies exact source and terminal
original jobs, records the new coordinator identity and preserves prior receipt
and Raven associations. It does not automatically retry a failed or ambiguous
native submission. Development launch requires an explicit passing pilot:

```bash
raven launch scripts/tangent_response_campaign.sh \
  --local --project "$PWD" --recipe tangent-response \
  -- --profile development --target fd_same_grid \
  --prerequisite pilot=tangent-response-pilot-a \
  --run-id tangent-response-development-a
```

The development campaign obtains its own exact native calibration after its
fresh fit bank is prepared. It does not reuse a pilot timing measurement for a
different workload. Fresh confirmation is outside this runnable program.

## Archive before a fresh launch

Once a cancelled campaign's submitted jobs and coordinator are actually
terminal, inspect the proposed moves before executing:

```bash
bash scripts/tangent_response_archive.sh --campaign tangent-response-pilot-a --plan
bash scripts/tangent_response_archive.sh --campaign tangent-response-pilot-a
```

Omitting `--campaign` discovers the most recent bounded tangent receipt inventory.
Archive checks the live scheduler census, exact retained job IDs/owner/root/unit
names, all observable comments, the stopped coordinator and any exact terminal
Raven association. An absent historical accounting comment is recorded honestly;
a nonempty mismatch or unknown state blocks moving files. It moves only the
selected campaign's native/recovery roots, parent root, receipt directory and a
matching tangent latest pointer under
`runs/archived-tangent-response/<fresh-id>/`. Atomic moves never overwrite or
delete; a bounded manifest maps original paths to preserved paths and records
the terminal proof. The outer Raven records, exports, unrelated runs and all
other program pointers stay in place. Frozen original absolute paths remain
unchanged as historical metadata; relocated evidence is not a resumable active
coordinator. Launching again creates fresh parents, IDs and calibration.

Raven contracts are pinned at
[PROJECT_STANDARD.md](https://github.com/rahel99x/Raven/blob/c34f0892d7bddc685c5af10489ecb114fd6adab6/docs/PROJECT_STANDARD.md),
[V2 reporting guide](https://github.com/rahel99x/Raven/blob/c34f0892d7bddc685c5af10489ecb114fd6adab6/docs/PROJECT_REPORTING_V2.md)
and [reporting/export guidance](https://github.com/rahel99x/Raven/blob/c34f0892d7bddc685c5af10489ecb114fd6adab6/docs/guides/automated-reporting-exports.md).

| Term | Meaning in this program |
| --- | --- |
| Anchor | Fixed discrete diffusion/reaction map with a smooth FP64 path and measured deployed FP32 counterpart. |
| Matched mean `G` | Mean-aware control evaluated on the same declared anchor and residual. |
| Spatial tangent `U` | Matrix-free linearized response to a declared source direction around that anchor. |
| Operator action / task residual | Response on independent source probes / response on a problem-specific residual; their ranks and claims differ. |
| Learned response / residual FNO | Gated response factors with separate mean/path/bridge terms / generic matched endpoint-residual control. |
| Arithmetic bridge | FP64 smooth anchor minus deployed FP32 anchor, separately computed and charged. |
| Parent / query | Independent initial-field draw / paired descendant by horizon, intervention, grid or source direction. Queries do not increase the independent-parent denominator. |
| Worker telemetry | Declared process/descendant CPU/RSS and assigned-device GPU counters; partial/unavailable coverage is explicit and never proves CUDA correctness. |
