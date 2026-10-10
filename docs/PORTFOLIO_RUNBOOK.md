# Run the TDN research portfolio

This is the new, separate A/B/C program described in [the research design](PORTFOLIO_RESEARCH.md). It preserves historical frontier/roadmap runs. Execute the native commands on **Fedora at `/home/rahel/TDN`**, outside an allocation. The wrappers select the project's `.venv`; no manual activation or `source scripts/common.sh` is needed.

## Update and check the existing desktop installation

```bash
cd /home/rahel/TDN
git pull --ff-only origin main
bash scripts/fedora_portfolio.sh doctor
```

The existing `.tdn/fedora-slurm.json` should already select the current user, root, CPU/GPU partition `local` and GRES `gpu:1`. If this is a new checkout without that profile/venv, configure and install once:

```bash
bash scripts/fedora_portfolio.sh configure --python python3.13 \
  --cpu-partition local --gpu-partition local --gpu-gres gpu:1
bash scripts/fedora_portfolio.sh setup
```

Do not rerun configuration with replacement flags simply to make a run pass. Fedora does not inherit CARC's `anakano_81` account. The current local account is optional and comes from the actual Slurm profile. Setup does not alter Slurm, the NVIDIA driver or Tower.

Do not change source, dependencies or the hardware profile while a campaign is running or awaiting exact-source recovery. Such changes require a new run; the old results remain intact.

## Start with native smoke

```bash
bash scripts/fedora_portfolio.sh plan --smoke
bash scripts/fedora_portfolio.sh run --smoke
```

`plan` prints the bounded job requests without submitting or computing. `run` submits the complete dependency graph. Native GPU stages each require preflight and the complete portfolio CUDA test suite; a CPU test or skipped CUDA test is not GPU verification.

Inspect progress at any time:

```bash
bash scripts/fedora_portfolio.sh status latest
bash scripts/fedora_portfolio.sh logs latest --lines 200
bash scripts/fedora_portfolio.sh paths latest
```

`paths` prints each stage's `TDN_TOWER_DIR` and `TDN_TOWER_METRICS`, plus generated analytical artifact paths. Before a stage starts, its Tower path may not exist. The final atlas is under:

```text
runs/<run-id>/report/figures/portfolio-atlas.pdf
runs/<run-id>/report/figures/index.html
runs/<run-id>/report/figures/chart-data.json.gz
runs/<run-id>/report/figures/learning-range-data.json.gz
runs/<run-id>/report/tables/
```

After all smoke jobs reach terminal states:

```bash
bash scripts/fedora_portfolio.sh validate latest
bash scripts/fedora_portfolio.sh collect latest
```

Validation checks sealed execution/artifact identity. It does not require every scientific hypothesis to succeed. A mathematical failure, a completed negative experiment and an unavailable measurement remain different outcomes.

## Development or full campaign

An intermediate native development run is available if smoke exposes a performance or numerical issue that merits a larger diagnostic before fresh full confirmation:

```bash
bash scripts/fedora_portfolio.sh plan --development
bash scripts/fedora_portfolio.sh run --development
```

After smoke completion and review, launch the frozen full campaign:

```bash
bash scripts/fedora_portfolio.sh plan --full
bash scripts/fedora_portfolio.sh run --full
```

Each launch creates a fresh UTC-stamped run directory and updates only the portfolio's latest pointer. To pin later commands to a specific run, replace `latest` with the printed run ID, such as `fedora-portfolio-...`. `--run-id` is supported on `plan`/`run` when an explicit fresh identifier is desired.

The full protocol currently contains **40 jobs: 21 CPU and 19 GPU**, including 12 two-parent confirmation-preparation units and 12 corresponding GPU confirmation units. All applicable models receive the same endpoint menus. Confirmation references are generated only after all selected models and settings freeze.

The full discretionary A/B/C science budget is **two hours**: A 66 minutes, B 30 minutes, C 24 minutes. This is **not the whole campaign estimate**. Shared teachers, diagnostics, confirmation, scaling and reporting have separate finite ceilings. Across the current full graph, science ceilings sum to **15 hours 40 minutes**, and requested allocation limits sum to **21 hours 38 minutes**. These are worst-case summed ceilings, not forecasts of actual use or billed charges. Actual elapsed/allocation costs are reported separately, including failures and recovery. Jobs finish when their work finishes.

No individual job exceeds 45 minutes; the current largest request is 40 minutes. Requests stay within eight physical CPU cores, 48 GiB host RAM and one RTX 4090; GPU stages use four CPUs. The Slurm node has 110000 MiB available host memory. VRAM retains the 18 GiB/75% soft and 90% hard policy. The launcher serializes this campaign's desktop resource use and queues its authorized graph without a pending-job-count cap. Other projects may still submit jobs subject to Slurm's real resource availability.

The three research paths remain scientifically independent where possible. A failed prerequisite blocks its scientific descendants. Resource sequencing uses terminal outcomes, so an unrelated prototype failure does not itself invalidate attribution. The report runs after any outcome and preserves missing/failed work. There are no unlimited retries or automatic budget increases.

## Collect everything needed for review

For a much smaller quantitative review upload from an existing finished run:

```bash
bash scripts/fedora_portfolio.sh compact latest
```

Upload its printed `index.json` and all listed archive/parts. This streams
deduplicated, solid XZ output, keeping all raw text measurements and chart data
exact while listing the hashes of omitted checkpoints, numerical arrays and
rendered figures. Use `--mode full` to retain those too. Original runs stay
unchanged; review mode is not a complete replay bundle. See
[compact review instructions](COMPACT_REVIEW.md) for verification, restoration,
resource use and measured reduction. It can read historical runs after an
exporter update without applying today's scientific-source guards.

After the jobs finish:

```bash
bash scripts/fedora_portfolio.sh status latest
bash scripts/fedora_portfolio.sh validate latest
bash scripts/fedora_portfolio.sh paths latest
bash scripts/fedora_portfolio.sh collect latest
```

If validation reports a missing or failed stage, still run `collect`; its logs and partial evidence are useful for diagnosis. Collection records `sacct` allocation/step rows without double-counting them, preserves unavailable accounting as unknown, and includes protocols, raw measurements, checkpoints, teachers, logs, failures and analytical outputs. Pytest temporary directories are excluded.

The command prints a `.tar.gz` and `.index.json`. If the archive is larger than the upload limit, it also creates ordered `.part001`, `.part002`, … files of at most 28 MiB. Upload the index and **all** parts; the index records individual and full-archive SHA-256 hashes. If there are no parts, upload the archive and index. Everything stays under the project root.

## Recover interrupted work

First preserve/inspect the failure:

```bash
bash scripts/fedora_portfolio.sh status latest
bash scripts/fedora_portfolio.sh logs latest --lines 300
bash scripts/fedora_portfolio.sh collect latest
```

When every origin job is terminal and the source/protocol/software/hardware profile is unchanged:

```bash
bash scripts/fedora_portfolio.sh recover latest --plan
bash scripts/fedora_portfolio.sh recover latest
```

Recovery creates a new workflow and an explicit bridge to the original. It reuses verified completed stages and resumes committed train/confirmation journals where supported. Completed trials or paired confirmation groups are retained; an unfinished group may need recomputation. Other incomplete stages restart in the new workflow. Original artifacts, interrupted attempts and costs remain available. The new report and collection include inherited evidence, with its original provenance.

Recovery deliberately rejects changed code/configuration/software or nonterminal origin jobs. Do not bypass fingerprints, copy stale checkpoints into new identities or overwrite completed runs. If a code fix is necessary, preserve the origin and start a new campaign under the revised source; decide explicitly which new comparison is development versus confirmation.

## Local CPU verification

Use a separate project checkout with a working venv and no inherited Slurm or execution-mode overrides. This path does not establish desktop GPU performance and rejects full confirmation:

```bash
cd /path/to/TDN
bash scripts/portfolio_local.sh --smoke
```

For a bounded development run:

```bash
bash scripts/portfolio_local.sh --development
```

An explicit fresh, project-contained output directory can be supplied:

```bash
bash scripts/portfolio_local.sh --smoke --run-dir runs/portfolio-local-check
```

The wrapper writes `local-protocol.json`, `local-state.json` and per-stage metadata/artifacts, and prints the final run path. Numerical progress is also printed to the terminal. It executes all feasible declared CPU stages and preserves blocked dependencies. Local reports are labeled CPU smoke/development; their timings cannot substitute for a Slurm-assigned RTX 4090 run. Native status/collect commands use the separate Fedora workflow manifest and latest pointer, so they do not inspect this local run.

## Read the result honestly

Start with the report's stage completeness, references and failure coverage. Then review matched analytic attribution, the fixed validation-selected schedules, accuracy–cost frontiers and exploratory prototypes separately. A 100% software pass rate does not establish a useful learned contribution. Dense learning bands show observed ranges, not confidence intervals; independent-field uncertainty is separately reported. Missing references or missing measurements remain NA. A useful normalized or fitted numerical rule without a network is an acceptable research outcome.
