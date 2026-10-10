# Working in this repository

Read `WINDOWS_START_HERE.md` and `docs/WINDOWS.md` for native Windows use. Read `docs/CARC.md` before changing or running the separate cluster workflow. `docs/IMPLEMENTATION_SPEC.md` is the original uploaded research specification; distinguish its environment examples and requested scientific extensions from implemented and measured behavior in `docs/IMPLEMENTATION_MAP.md` and the source.

## Desktop execution

- The user authorized a native Windows adaptation for an RTX 4090, 24 GB dedicated VRAM and 128 GB host RAM. Shared/host RAM is not additional dedicated VRAM.
- Use standalone 64-bit Python 3.11 or 3.12 in the repository's `.venv`, never Conda. Invoke `.venv\Scripts\python.exe` on Windows; activation is optional.
- Use `scripts/windows/Doctor.ps1`, `Setup.ps1` and `Run.ps1`. They set `TDN_EXECUTION_MODE=desktop` explicitly and root all generated files in this extracted checkout. Do not invent Slurm environment variables to pass a runtime check.
- Run doctor before choosing a CUDA wheel. Select the installed-driver-compatible official PyTorch distribution; do not infer wheel compatibility from CUDA toolkit availability or the GPU name. Do not install or alter GPU drivers automatically.
- Keep temporary files, caches, dependency caches, datasets, logs, checkpoints and reports within the repository. Do not redirect them to system temp, `/tmp`, `/scratch1` or external drives. Do not bring in a Linux venv from an archive.
- Start with `configs/desktop-smoke.yaml`, a fresh run identifier and a 12-step maximum. Keep eager FP32, TF32 off and FP64 teachers. Native Windows does not require Triton or `torch.compile`.
- Use CPU mode explicitly if CUDA is unavailable. Do not report CPU results as GPU results or silently downgrade a requested CUDA run.
- Preserve completed runs. Resume interrupted runs using the same configuration, dataset, source and software/device environment; use the launcher resume checks. Changing code/configuration invalidates prerequisite fingerprints and may invalidate exact resume.
- GPU VRAM policy in the desktop smoke uses an 18 GiB absolute soft cap, a 0.75 fraction of available memory and a 0.90 hard fraction. Respect lower available memory caused by the display or other programs. Do not increase those limits to hide an out-of-memory failure.

## Scientific scope and evidence

The implemented starting problem is periodic logistic reaction–diffusion with an exact discrete diffusion split, FP64 coupled teachers, six learned controls and classical accuracy/cost controls. Retain parent-disjoint splits, validation-only checkpoint selection, independent diagnostics and matched-tolerance comparisons.

Historical CPU screening in `results/gates.json` failed G2 and G4. The tiny smoke is a bounded development diagnostic with its declared permissive headroom setting. It does not authorize disabling `require_headroom` in a pilot or expanding into a confirmatory campaign. Optional Gray–Scott, advection, distributed HALO, KAN and oscillatory branches require separate implementation and validation.

Write factual reports: record actual commands, execution mode, device, versions and failures. Keep historical cloud CPU, actual desktop CPU/GPU and actual CARC A100 evidence distinguishable. A successful pipeline or unit test suite does not demonstrate a TDN efficiency advantage. GPU results must come from actual GPU execution; no mock or skipped test counts as GPU validation.

## CARC execution remains separate

The cluster workflow requires user `aadaniel`, charging account `anakano_81`, root `/home1/aadaniel/projects/TDN`, a project `.venv` and real Slurm allocations for substantial compute. Preserve those checks. Desktop mode must not bypass them on CARC or masquerade as an A100 allocation. All CARC project/cache/temp/run files remain under its fixed root, with no `/tmp` or `/scratch1` use.

## Fedora desktop Slurm

- The user moved the current premix experiment to `/home/rahel/TDN` on Fedora,
  with standalone Python 3.13.13, a Ryzen 7800X3D, 128 GB RAM and an RTX 4090
  with 24 GB dedicated VRAM. Read `docs/FEDORA_SLURM.md` for its launchers.
- Use `scripts/fedora_slurm.sh` and explicit `desktop-slurm` execution. The
  local profile records the checkout root, actual user, partitions and optional
  account. Never assume the CARC account or A100 constraints apply locally.
- Scientific work still requires real `sbatch`/`srun` allocations, checked against
  scheduler ownership, working directory and task-visible hardware. No invented
  scheduler variables or silent CPU fallback can establish GPU readiness.
- Keep all project outputs, caches and temporary files under this checkout and
  install its own `.venv`; do not copy the CARC or Windows venv. Retain FP32
  inference/training, TF32 off and FP64 teachers. GPU soft limits are 18 GiB and
  75% of initially free/total memory, with a 90% device-use hard threshold.
- CPU stages run in sequence on the eight-core desktop. The historical premix
  and consistency stages use 30-minute allocations; the research agenda uses
  explicitly frozen 20/30/45-minute stage limits. There is no pending-job cap or
  automatic expansion. Start with the allocated smoke before a full run.

## Bounded consistency program

- Read `docs/CONSISTENCY.md` before changing the new seven-arm experiment.
  Use `scripts/fedora_consistency.sh`; its audit → prepare → neural chain has
  separate run identifiers and does not overwrite the premix protocol/results.
- Keep the physical gate and mean/spatial treatment paired on premix and FNO.
  The gated correction must vanish for constant fields, zero reaction, zero
  diffusion and zero time even with arbitrary nonzero weights. Mean calibration
  targets a learned bounded mean; logistic reaction does not conserve mean.
- Full training is three paired seeds with 300 updates per arm. Keep all
  initialization selections, failures, unmet targets, same-step base regressions
  and independent mean/spatial errors visible. No scientific superiority follows
  from successful structural tests or computational completion.
- Fresh train/validation/diagnostic parents use the immutable consistency
  protocol. The previously inspected premix diagnostic bank is development
  evidence and cannot be relabeled as an untouched diagnostic cohort.
- The required structural audit and independent teacher preparation must seal
  before GPU work. All four constrained arms must pass the declared checks;
  the original arms' lack of those constraints is reported as an observation.

When editing, run tests appropriate to the change. Windows launcher syntax and actual NVIDIA execution need native-machine validation; Linux checks cannot substantiate those claims. Never delete results or overwrite a completed run to make a check pass.

## Full bounded research agenda

- Read `docs/RESEARCH_AGENDA.md`, `docs/AGENDA_REQUIREMENTS.json` and
  `docs/AGENDA_LITERATURE.md` before changing the new eight-stage program.
  `scripts/fedora_agenda.sh` keeps separate profiles, run IDs and latest pointers
  from premix and consistency. Implemented question IDs Q1–Q7 cover all six
  supplied literature questions and the deployment-policy extension.
- The current Slurm node exposes 110000 MiB host memory, not all 128 GB of
  physical RAM. Keep CPU stages at no more than eight physical cores and
  48 GiB, GPU stages at four CPUs and 32/48 GiB, one RTX 4090, and the existing
  18 GiB/75% VRAM soft limits. The eight stages run sequentially.
- Keep source-before-transport zero-preserving, gate-placement and nonlinear
  time controls separate, and phase/symmetry/reality intact in the pair kernel.
  An already-quadratic pair branch must not acquire another quadratic gate.
  Nominal learned rank and expanded separable terms are distinct quantities.
- Mean–variance closure is an approximate dynamical treatment with exact
  instantaneous identities, not the exact future mean/variance law. Retain the
  old endpoint mean head and analytic spectral-mean control. Strong-diffusion
  Strang orientation is an established classical control, not a novelty claim.
- Full confirmation uses eight continuous-field variants crossed with four
  physics choices and paired grids. The selected continuum subset is declared
  separately. Keep same-grid FD/nodal and dealiased continuum-estimate targets
  distinct. Reused parent/phase/seed/grid rows remain paired observations.
- Freeze checkpoints, selection, source/profile/software and success criteria
  before loading fresh confirmation. Smoke and development parent seed ranges
  are disjoint from full confirmation; local wrappers reject full confirmation.
- Policy fitting uses calibration only. Fresh references audit accepted
  accuracy and false acceptance; they must not make deployment decisions.
  Include rejected work, estimator work and measured classical fallback cost.
- Preserve negative, failed, infeasible and initialization-selected outcomes.
  Literal question-bearing executable evidence is required; documentation,
  passing tests or computational completion alone does not show superiority.

## Bounded M00–M23 roadmap program

- Read `docs/ROADMAP_EXPERIMENTS.md` and the dated research roadmap before
  changing `tdn/analysis/roadmap`. Use `scripts/fedora_roadmap.sh` for the
  current Fedora workflow; keep its ten stages, manifests and latest pointer
  separate from agenda, consistency and premix. The local wrapper supports
  CPU smoke/development only and must reject full fresh confirmation.
- The desktop scheduler exposes 110000 MiB RAM. CPU stages use at most eight
  physical cores and 48 GiB; GPU stages use four CPUs, 32/48 GiB and one actual
  RTX 4090. Preserve 18 GiB/75% soft VRAM and 90% device-use hard limits,
  project-contained files, a Python venv, FP64 teachers, FP32 models and TF32
  off. No stage exceeds 45 minutes; no pending cap or automatic expansion.
- Every GPU stage must run all 48 declared CUDA cases from the protocol:
  limits and gradients for all 17 families, four numerical parity cases,
  fractional physical scale, actual visible-allocation verification, and eight
  classical-controller/policy-fallback cases across discrete/continuum targets,
  FP32/FP64 controller precision and empirical/conformal decisions.
  The JUnit checker rejects missing, duplicated, skipped or failed cases.
  CPU tests or a filtered GPU suite cannot establish native readiness.
- Full training declares 64 tuning trials and 51 final trials across three
  paired seeds. Preserve the shared-optimizer loss comparator, rank-zero
  combined control, direct neural-only FNO, cheap/deep hybrid FNO, and
  strongest classical controls. Freeze every selected checkpoint and its
  effective settings before creating or reading fresh confirmation teachers.
- Keep independent fields, physics variants, grids, schedules and training
  seeds distinguishable. Main confirmation has 24 independent field clusters
  crossed with two physics choices; the declared continuum subset must retain
  balanced field regimes. Calibration fits must never inspect confirmation
  truth. Deployment reads deployable features and frozen calibration only.
- Preserve exact nulls, quadratic/cubic amplitude order, signed phase products,
  declared spatial target and one-time mean replacement. A midpoint-only DF
  quadratic defect is zero. Dynamic moment closure is approximate beyond its
  instantaneous identities. Explicit RK residual order is not A-stability.
- Every experiment log needs explicit math/gap evidence, including NA,
  effective parameters, costs and required check IDs. Scores are weighted
  evidence attainment, not model quality or theorem proofs; required NA
  checks retain denominator weight. Structural audit failures block successors;
  scientific BAD/NA results remain preserved and do not silently stop the
  other authorized mechanisms. The final report runs after any outcome.
- Use the true mechanism assessments in `report/mechanism_summary.json` or
  `experiment_summary.csv`; report inventory checks are infrastructure only.
  Include rejected/estimator/fallback work and offline amortization, separate
  numerical time from scheduler accounting, and keep unverified large-grid
  accuracy and unsupported statistical guarantees NA. Modify project Tower
  exports only; do not modify the Tower application.

## Five-gate physical interaction program

- Read `docs/FRONTIER_RESEARCH.md` and `docs/FRONTIER_RUNBOOK.md` before
  modifying `tdn/analysis/frontier`. Use `scripts/fedora_frontier.sh`; preserve
  historical roadmap runs and their original scores. The separate nine-stage
  program implements evaluation repair, spatial/headroom screening, focused
  neural comparison, accuracy-backed scaling and conditional deployment.
- Preserve the distinct neural-architecture and practical-solver claims. An
  inadequate trained FNO baseline remains unresolved; local adapted FNO
  controls do not establish reproduction or defeat of a published benchmark.
  Easy and stress cases, failed trials, initialized selections and missing
  references must remain visible. A priori stress labels are not universal
  worst-case claims.
- Match spatial operator, continuous field, physical final time, grid, RMS and
  maximum tolerance. Keep discrete FD/nodal and spatially refined Galerkin
  targets separate. Teachers must actually refine time and space; agreement
  is an uncertainty estimate, not an error certificate. Every prescribed
  intermediate confirmation time needs its own independently accepted teacher.
- Freeze all family/track/seed/data-fraction checkpoints before confirmation,
  scaling and policy banks are generated. Policy calibration and policy tests
  use distinct fields. Never choose a competitor seed, schedule or model using
  policy truth. Preserve explicit initialized and analytic attribution controls.
- Maintain the existing desktop allocation envelope: at most eight CPU cores,
  48 GiB host memory, four CPUs and one RTX 4090 for GPU stages, and at most
  45 minutes per job. Keep the 18 GiB/75% soft and 90% hard VRAM policy. Full
  execution requires actual desktop Slurm; local CPU smoke is separate. There
  is no pending-job cap or automatic resource expansion.
- All four GPU stages require the complete 45-case native suite, including
  operational policy sampling/resampling/fallback and different-field batching.
  Mocked scheduler tests and CPU-skipped CUDA cases are not GPU validation.
- Interleave repeated randomized timing, keep first invocation/warmup separate,
  count rejected and estimator work, and do not turn all-classical fallback into
  neural amortization. Block deployment by track when measured development
  headroom or confirmation cost margin is absent; preserve that scientific NA
  outcome while allowing the final `afterany` report to finish.
- Classical endpoint frontiers include uniform 1/2/4/8-step controls at each
  final time; duplicate workloads reuse their original measurement. Primary
  shared schedules retain separate intermediate-state teacher audits. G4 must
  score measured throughput utility as well as accuracy: accurate but slow is
  BAD utility, and unresolved references/timing remain NA.
- Publish complete loss, error, compute, timing and memory chart data with the
  graphs and Tower sidecars. Monetary cost remains unavailable without a rate.
  Do not silently replace original observations with revised comparisons.
- Confirmation executes as deterministic whole-parent parts (six in full), then
  a sealed whole-cohort aggregate. Preserve all models, schedules and paired raw
  timing rounds; a partition cannot satisfy the complete confirmation prerequisite.
  Use `fedora_frontier.sh recover` for an interrupted compatible run. Preserve
  original artifacts and the explicit recovery bridge; never disable source or
  software guards or mix unsealed partial measurements into new science. Reused
  training/reference costs and new part/merge allocations must remain distinguishable.
  Each native confirmation part and its aggregate require all 45 CUDA cases.

## Adjacent interaction study

- Read `docs/ADJACENT_INTERACTION_STUDY_PROMPT.md`, `ADJACENT_RESEARCH.md`,
  `ADJACENT_EVIDENCE.md` and `ADJACENT_RUNBOOK.md` before changing the separate
  `tdn.analysis.adjacent` program. Keep active portfolio source, protocols,
  budgets and historical evidence unchanged. Use an isolated checkout;
  promotion into the main architecture is a separate proposed decision.
- D01–D08 are development diagnostics. LL/LH/HH describe input origins, not a
  third spatial dimension. Fix the same background in every polarization term;
  normalized fixed recombination must match the two-node analytic correction.
- Preserve finite-field roughness semantics, actual gain bounds, signed phase,
  physical nulls, output-compression attribution, independent parent clusters,
  scalar/affine controls, strong cubic/classical controls and local FNO paths.
  Finite trigonometric fields are smooth; alpha=1 is a finite-band stress case.
- Keep the independent 20% exploration allocation, finite per-unit budgets,
  45-minute maximum, 110000 MiB scheduler ceiling and 4090 VRAM limits. Current
  adjacent requests are four CPUs/32 GiB and at most 38 minutes per job.
  Native GPU stages require all 92 declared CUDA cases, without skips.
- Freeze model selection and precision planning before generating fresh fields.
  Inadequate references, training, coverage or sample precision remain NA.
  Do not pool discrete and continuum claims or treat paired variants as fields.
  Read-only main-queue discovery defers adjacent work; never cancel main jobs.
- Use exact-source recovery and the separate latest pointer. Preserve failed
  work and costs, reference/timing journals, source-bound charts and Tower
  sidecars. Mathematical checks, job completion and scientific success differ.

## Paired adjacent resolution follow-up

- Read `docs/ADJACENT_RESOLUTION.md` for the separate 64²/128² program. Use
  `scripts/fedora_adjacent_resolution.sh`; preserve historical adjacent profiles,
  latest pointers and evidence. Every new profile, including `--full`, is a
  bounded development study, not a confirmatory superiority claim.
- Pair the same continuous parent across grids with fixed physical cutoffs.
  Resolution-relative bandwidth stress is a different, labeled workload.
  Same-grid FD/nodal and dealiased Galerkin references are distinct finite
  equations; selected spatial refinement must remain separately reported.
- Keep FP64 refined/cross-checked teachers, FP32 models with TF32 off, frozen
  validation selection before fresh evaluation, actual completed trial work,
  strong GL4 quadratic+cubic controls and paired endpoint opportunities.
- The new native protocol requires all 108 CUDA cases without skips. CPU
  checks or engineering runs cannot satisfy that requirement. Jobs request
  four CPUs/32 GiB and at most 23 minutes; retain the existing VRAM limits,
  recovery/source guards and protected 20% exploration allocation.
- Bicubic spatial views and PCHIP curves are display interpolation only.
  Retain raw arrays, raw views/points, visible missing-data gaps, direction
  guidance and source hashes. Observed-range bands are not confidence intervals.
