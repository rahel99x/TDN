# Resolution program: engineering readiness

The new program is implemented for actual 64² and 128² fields. Native Fedora
Slurm/CUDA execution remains pending; this directory contains CPU engineering
evidence, not a solver-superiority result.

- Final source `52567ab`: **322 CPU tests passed**. All **108 required native
  CUDA identities** were collected exactly; none was executed on this CPU host.
- Source `82bb067`: a complete **44-stage CPU smoke** produced 600 diagnostic
  rows, 296 endpoint cells, a 28-page 400-DPI atlas and verified scientific and
  execution seals. Its 28 selected models use a two-update smoke recipe, not
  established optimization convergence. There are only two independent
  evaluation fields; repeated grids/methods/schedules do not add fields.
- Actual saved-checkpoint replay exported all four grid/track views: 24 raw or
  display-smoothed PNG panels, up to 6400×6800 pixels, plus vector PDFs, arrays
  and source hashes. Source run and gallery paths are in `readiness.json`.
- The smoke exposed premature teacher stopping: Lawson had converged, but two
  independent ETDRK4 checks remained unresolved with unused refinement work.
  Source `52567ab` fixes joint refinement. **All 16 original bank cases were
  recomputed and accepted**, within the unchanged 1e-10 tolerance and 256-step
  ceiling. The two affected cases use 128 finest substeps. See
  `reference-recheck.json`, its original arrays and the matching NPZ hash.
- The original smoke, atlas and scores remain unchanged. They therefore still
  show their original 14/16 accepted bank entries and unresolved diagnostics.
  The recheck is separate development evidence, not retrospective relabeling.

Some CPU unit checks overlapped the integration run. Its raw timing data is
preserved but is not an isolated deployment benchmark, GPU performance result
or basis for a winner claim. Statistical parent-cluster confidence intervals
are not computed by this resolution atlas; they remain explicitly NA. Observed
range bands and display interpolation do not replace statistical uncertainty.

`budget-ceilings.json` lists safety limits, not expected run times or bills.
The paired full declaration has 68 jobs and no job longer than 23 minutes.
The existing three historical adjacent protocol dictionaries remain unchanged.

Run the allocated native smoke before the full program using
`bash scripts/fedora_adjacent_resolution.sh run --smoke` in the isolated
checkout. After all jobs finish and `validate latest` succeeds, `run --full`
submits the paired 64²/128² development study. Consult
`docs/ADJACENT_RESOLUTION.md` for scope, controls and resource constraints.
