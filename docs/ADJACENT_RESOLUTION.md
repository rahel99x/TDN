# Paired 64² and 128² adjacent experiments

This is a new bounded **development** campaign on the isolated
`research/adjacent-interactions` branch. It preserves the historical adjacent
and main-portfolio artifacts. The prior CPU evidence suggested that interaction
order and output retention had more headroom than three separately fitted
quadratic gains. This campaign tests whether that diagnosis persists at useful
spatial resolutions, and whether learning or temporal reuse can repay its cost.
Successful Slurm jobs are not successful scientific claims.

## Questions and controls

| Question | Controlled experiment | What would matter |
|---|---|---|
| Does refining the grid improve the same physical problem? | Sample the same continuous parent at 64²/128² with fixed physical cutoffs; compare specified FD and Galerkin equations separately | Reduced error above teacher uncertainty, with the same parent and final time |
| What happens when genuinely finer input scales are added? | A separately named resolution-relative high-frequency pair | Changed workload is not mislabeled as fixed-field grid convergence |
| Is output compression the limitation? | Matched quadratic/cubic corrections with retained and full output; orthogonal L2 projection floor | Error attributable to discarded output modes rather than conditioner capacity |
| Do additional quadrature nodes help? | Normalized GL2, GL4 and refined quadrature with matched spatial treatment | Resolved quadratic quadrature error, distinguished from a missing cubic direction |
| Are input-origin gains useful or identifiable? | Bounded scalar, LL/LH/HH and output-band response spaces, fitted global/affine and neural controls | Useful residual reduction above uncertainty and stable inferred directions |
| Does learning improve a credible baseline? | Paired data, two seeds, validation-selected training recipes; small/larger local-path FNO controls | Adequate optimization and better accuracy–cost, not merely fewer parameters or equal update counts |
| Do benefits persist beyond one fitted endpoint? | Shared endpoint schedules, autonomous/unequal-step diagnostics and cross-grid checkpoint transfer | Early transient, maximum, mean and spectral errors remain acceptable |
| Does the work pay for itself? | Cold/warm complete calls, randomized paired timing, distinct-field batches, memory, preparation/training/failed-work accounting | Accuracy-qualified cost margin with all necessary work charged |
| Do earlier negative ideas change at larger grids? | Protected HR-E01 selective cubic and HR-E02 temporal reuse pilots | Predeclared accuracy and cost criteria against their strongest numerical controls |

The diagnostic cutoffs use physical integer frequencies, not automatically a
constant fraction of the grid. Primary parents have support strictly below the
64-grid Nyquist limit. They retain the same modes, phases, coefficients, mean,
RMS and physical parameters on both grids. Resolution-relative stress fields
have distinct identities and explicitly changed bandwidth. Finite Fourier
fields remain smooth; their alpha labels do not establish fractal dimension.

The learned candidate still combines physical LL/LH/HH correction fields using
three gains. A larger grid increases the physical work and resolves finer
structure; it does not automatically increase the conditioner's information.
The reference never becomes an inference input. All fitted coefficients,
initialization selections, incomplete optimization and failed cases remain in
the outputs.

## Reference and inference precision

Banks use FP64 Lawson time refinement with an independently implemented ETDRK4
cross-check and a declared tolerance of 1e-10. Failure to meet the reference
criterion remains visible. The recorded uncertainty is an empirical refinement
estimate, not a mathematical certificate.

Training/evaluation targets are explicitly the **same-grid FD/nodal equation**
or the **same-grid dealiased Galerkin equation**. The historical code string
`continuum` denotes the latter track; it does not turn that finite equation into
continuum truth. Selected spatial-refinement diagnostics separately compare
projected finer grids, bounded at 512². Time error, spatial discrepancy and
model error must not be merged into one attribution.

Models run in FP32 with TF32 disabled. Small learned changes may disappear after
endpoint rounding. Reports retain the actual complete endpoint and distinguish
it from a magnified or FP64 explanatory correction.

## Frozen development design

The full paired program uses 16 training, 8 validation and 12 fresh evaluation
parents, with paired 64²/128² views and separate physical equation tracks. It
uses two training seeds and a bounded 128-update recipe. Global/affine fits,
the three neural gain controls and two FNO capacities use the declared trial
menus, including the clipping/loss-scaling control. A 20-second trial ceiling
can stop a trial before 128 updates; actual updates, elapsed time, clipping and
selection status must be reviewed before interpreting a comparison.

All checkpoints and validation choices freeze before evaluation-bank jobs can
run. The evaluation remains development evidence: twelve parent fields do not
become hundreds of independent observations because we repeat grids, horizons,
schedules, models and seeds. Statistical intervals and reference uncertainty
are reported separately. No universal ranking or automatic FNO-paper claim is
authorized. Inspected evaluation data becomes development data for later work.

Both-grid profiles evaluate each grid's own trained checkpoints and separately
apply the frozen 64² checkpoints at 128². These transfer rows are exploratory
and are not pooled with models trained at 128². Single-grid profiles explicitly
disable cross-grid transfer. The primary endpoint eligibility requires both
RMS and maximum error, including recorded reference uncertainty, to be at most
2e-5. The 1e-4, 2e-5 and 1e-6 tolerance sweep provides separate descriptive
accuracy–cost frontiers; it does not add primary superiority claims.

The declared horizons are .04 and .12; shared endpoint schedules have 1/2/4/8
steps, with additional classical 16/32-step opportunities. Selected diagnostics
inspect earlier times and longer autonomous behavior. A missing point stays
missing; a method outside the fixed tested schedule menu is not proven
incapable of reaching a tolerance.

## Execution and recovery

Use `scripts/fedora_adjacent_resolution.sh` from `/home/rahel/TDN-adjacent`.
It has its own run IDs and latest pointer. `--smoke` uses actual 64² and 128²
fields but tiny data/training counts; `--full` runs the paired program. The
optional `--full --grid 64` or `--full --grid 128` selects one grid. Single-grid
runs cannot establish paired-grid or 64→128 transfer conclusions.

CPU reference banks are generated in recoverable parent shards before GPU
training. GPU train/evaluation work is separated into bounded units. Evaluation
requires the frozen selection. The final report runs after any outcome and
retains failures and missing prerequisites. Recovery requires identical source,
protocol, software and hardware profile, and preserves the original attempt.
Do not pull code or modify the venv during an active campaign.

Jobs request four CPUs and 32 GiB host RAM, with one RTX 4090 only for GPU
stages. The node cap remains 110000 MiB; host RAM is not additional GPU VRAM.
The 18 GiB/75% soft GPU limits and 90% hard policy remain unchanged. No new job
exceeds the 45-minute maximum; the current largest request is 23 minutes.
There is no pending-job cap. This campaign serializes its own resource usage
and defers to main-portfolio jobs discovered at submission.

The two exploration prototypes retain 20% of discretionary architecture-plus-
exploration time. References, diagnostics, evaluation and reporting have their
own finite limits. `plan` prints all jobs and budgets before submission.
Summed allocation limits are worst-case safety ceilings, not predicted runtime
or bills; actual scheduler usage and measured work are recorded separately.
The paired full declaration has 68 jobs: 15h53m summed science ceilings and
22h47m summed allocation ceilings (including checks and shutdown allowance).
Each single-grid full declaration has 36 jobs and an 11h45m allocation ceiling.
These deliberately conservative totals are not elapsed-runtime forecasts.

## Smooth scientific presentation

The atlas supplies raw nearest-cell views and clearly labeled bicubic display
companions. Interpolation makes a spatial plot visually continuous; it does not
add measurements, change the solver grid or alter errors. Bicubic interpolation
can overshoot between observed cells. Color ranges and raw plots remain
available, and extrema/metrics come from original arrays.

Suitable summary curves use shape-preserving PCHIP interpolation with original
points visible. Missing measurements break curves. Dense learning views retain
continuous translucent observed-range bands and thin boundaries; those bands
are **not confidence intervals**. Statistical intervals have separate labels.
Ours uses triangles, Theirs circles and analytic controls squares, with explicit
better/worse guidance. Raw chart tables, source hashes and plotted display
values accompany the high-resolution PNG, vector PDF and HTML atlas.

For a selected checkpoint's parameter-to-spatial-response illustrations use
`scripts/adjacent_prediction_view.py` with the resolution run, explicit grid,
method, seed and parent selection. See that command's `--help`; the raw export
remains available alongside display-only bicubic companions.

After complete aggregation, `bash scripts/adjacent_resolution_images.sh latest`
verifies the full source chain and exports all declared grid/track views at
400 DPI, with vector PDF companions. The first declared seed, field and horizon
are chosen before inspecting errors. Each export gets a fresh timestamped
directory under `outputs/`; `--plan` prints the selections without rendering.
Statistical parent-cluster confidence intervals are currently not computed by
this resolution program; the atlas marks them NA and retains descriptive
per-parent measurements. Display bands never substitute for those intervals.
