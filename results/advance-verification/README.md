# Advance: local verification and bounded development evidence

Recorded 2026-10-10. This is **CPU development evidence**, not a native RTX 4090
experiment or a fresh full confirmation. It accompanies the research program
in [`ADVANCE_RESEARCH.md`](../../docs/ADVANCE_RESEARCH.md) and the native
[`runbook`](../../docs/ADVANCE_RUNBOOK.md).

## What was checked

| Check | Actual outcome | Scope |
|---|---|---|
| Relevant CPU suites | 307 cases passed across the selected suites | Numerical identities, models, metrics, data leakage, optimizer recovery, source/reference seals, workflow, report and exporter regressions |
| CUDA suite | 40 distinct cases collected; **not executed** | Native launchers require all cases, with no skipped or missing identities |
| Complete CPU smoke | 15/15 stages completed and sealed artifacts verified | Audit, diagnosis, reference preparation, three prototypes, both training tracks, freeze, confirmation, aggregation, scaling and atlas |
| Smoke confirmation | Two distinct fields; 420 endpoint rows; zero numerical failures | Too small for a scientific comparative claim; repeated methods/schedules are not independent fields |
| Smoke atlas | 119 panels: 104 observed, 15 unavailable | 300-DPI PNGs and vector PDF generated locally; missing metrics remain NA |
| Additional 16² numerical diagnostic | Audit and diagnosis completed; both seals verified | 256 endpoints, 32 oracle projections, 64 composition probes; zero mathematical-check failures |
| Independent formulation probes | XH, XR and XT executed on CPU with their full-sized development settings | Exploratory inspection, not untouched confirmation or native timing |

The CPU suite inventory is retained in `logs/advance-final-collection.txt.gz`.
The 307 cases are the union of the advance CPU suites plus compact-review and
portfolio learning/reporting regression suites, run in batches during
implementation. This is not a claim of 307 newly independent scientific tests.

The final smoke used Python 3.12.14, Torch 2.10.0+cpu, NumPy 2.2.6 and SciPy
1.15.3 in the project `.venv`. Its scientific source fingerprint was
`ae810a101203ddd4cb26dcd75253ef62ebb0cea19f5a36f15a8c347d81d01466`.
The original execution metadata records parent Git commit
`ca913975156fcf2da41c1f5c042d54bc4a680868` plus that working-tree fingerprint;
the implementation was committed after verification. No GPU timing, native
Fedora Python 3.13 execution or Slurm allocation was simulated as real evidence.

Local command used for the passing smoke:

```bash
bash scripts/advance_local.sh --smoke \
  --run-dir runs/advance-local-verified-20261010b
```

Use a fresh run directory to repeat it. Original local runs and failed attempts
remain intact. The first smoke's diagnosis failed on duplicate `track` metadata
while constructing an endpoint record. Its failure summary and log are retained
here; the metadata merge was corrected and a real-stage regression was added
before the successful new run. No failed observation was overwritten.

## What the numerical evidence supports

The 16² diagnosis crosses four **constructed fields** (low, high-pair,
broadband and near-Nyquist), two specified spatial equations and four time
steps. The 32 combinations are paired diagnostics, not 32 independent fields.
The continuum-labeled diagnosis targets the specified same-grid Galerkin
equation; its temporal reference refinement does not establish convergence
to the continuum PDE. The full campaign separately refines space.

Compared with correctly normalized two-node quadratic quadrature, the fixed
heat-filtered cubic correction reduced RMS error in 32/32 cases, by factors
between 1.17 and 35.34. Both errors exceeded the estimated reference
uncertainty in all 32 comparisons. The raw cubic correction worsened three
cases. The more expensive nested quadratic+cubic control was more accurate
than the filtered correction in this diagnostic. All three and normalized
two-node quadrature already passed the joint RMS/max target of 2e-5 in all
32 cases; the basic diffusion-first split passed 29/32. Reducing an already
acceptable error does not establish a useful cost advantage.

This supports testing the missing cubic **shape**, retaining the raw and
filtered controls, and measuring whether a larger usable step repays its
additional work. It does not establish that learning is necessary. The
`two_basis` model has two global fitted gains, not a neural conditioner. A
teacher-informed, unbounded, per-case projection reduced residual error by a
median factor of 16.97 relative to a one-basis projection. That is attainable
representation headroom, not evidence that two shared fitted parameters can
recover it on unseen fields. The frozen full protocol tests that distinction.

`diagnostic-findings.json` is derived from the exact published diagnostic
records by this standard-library-only command:

```bash
python results/advance-verification/derive.py
```

## What changed in the mathematical audit

The cubic expansion retains the nesting of the Galerkin product: spectral
projection makes that product nonassociative. The nodal simplification is
not substituted into the continuum track. The heat filter is explicitly a
fixed-grid heuristic, not a proof of uniform stiff order or stability.

For even grids, the physical Fourier-interpolant inner product weights a
Nyquist cosine by one half per Nyquist axis. The Galerkin energy law is checked
in this physical inner product, while sampled RMS errors retain their stated
grid norm. Using the ordinary nodal norm for both would produce a false
energy-law defect at the Nyquist modes. Random full-spectrum, odd/even-grid,
independent ODE and finite-difference derivative tests cover these distinctions.

## Exploration decisions

- **XH, causal unresolved-scale history:** fitted memory improved clean
  autonomous coarse rollouts, but small observation noise caused substantial
  regressions. Acquisition, fitting and rollout costs remain charged. Continue
  only as a noise-aware information/closure study; this is not a ready solver.
- **XR, rational/resolvent propagation:** exact exponentials, equal factor
  caching and classical dense output are retained. No consistent practical
  superiority was established. Rational propagation should not advance merely
  because its scalar stability function is well behaved.
- **XT, tangent and hidden-state probes:** derivative and composition checks
  passed; equal coarse-state/variance examples can still have different future
  mean behavior. This motivates richer causal information, without proving a
  learned tangent architecture or a universal impossibility theorem.

The `exploration-development/` records contain the exact source-file hashes,
seeds, response rows, selection records and limitations. Their `full: true`
setting denotes the size of the **development prototype**, not a native full
campaign or a locked confirmatory result. Those inspected parents remain
development evidence.

## What is deliberately still unknown

Native paired speed, peak GPU memory, meaningful tail latency, matched-accuracy
deployment margin, fresh full-cohort model comparisons and continuum spatial
generalization require the desktop run. Lower training loss alone is not a
fairness result. The FNO families are local physically informed adaptations,
not reproductions of a published FNO benchmark.

The atlas marks absent kernel-level profiles, CPU peak-memory measurements,
intermediate spatial maps and actual model endpoint time derivatives as NA.
Observed-range learning bands are not confidence intervals. First-call timing
in a warm process is not true process-cold latency. Money and energy remain
unknown without measured rates/power. Small regime samples and multiple
unadjusted comparisons limit inference even in the full campaign.

## Publication format and provenance

`publication.json` hashes every published file. Entries with `source` preserve
that file's exact bytes (usually inside gzip) and include its original SHA-256;
derived summaries and this explanation are labeled separately. This roughly
2.2 MB evidence subset omits large replay arrays, checkpoints and rendered
figures. It is not a standalone replay archive, and subset omission must not
be mistaken for corruption of an original science manifest.

The atlas chart data and manifest are retained; the compressed original HTML
is an inventory whose image/PDF links require the original run. Stage elapsed
times exclude some launcher and scheduler work. The desktop runbook captures
final `sacct` accounting separately, including failed/interrupted allocations;
allocation and task-step rows must not be summed together.
