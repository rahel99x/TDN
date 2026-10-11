# Resolve: implementation checks and cloud CPU development

These results are **development evidence**, not native RTX 4090 validation or
fresh confirmation. Original run artifacts remain intact. The new Fedora
program implements the five recommendations in the full Advance review.

## What was executed

- Complete 12-stage CPU smoke: `runs/resolve-cloud-smoke-20261011`.
- Complete 12-stage CPU development: `runs/resolve-cloud-development-20261011`.
- Complete final-source CPU smoke after the display fixes:
  `runs/resolve-cloud-final-smoke-20261011`.
- Every stage completed; both campaigns recorded zero required mathematical
  or correctness failures. Scientific BAD and NA outcomes remain in the rows.
- Development includes eight training and eight validation fields, two physical
  tracks, 16² paired mechanism diagnostics, 32² kernel timing, and six neural
  family/track comparisons. There are 27 optimizer trials per track, including
  independent long, equal-time and selection-ineligible tiny-fit opportunities.
- The original integrated Resolve test suite passed 169 tests; its 20 CUDA
  cases were skipped because this cloud environment has no actual CUDA.
  Shared exporter/Advance regressions passed 105 tests; publisher regressions
  passed 29 tests. Final display-fix checks are recorded in `validation.json`.
- Python 3.12 / CPU PyTorch 2.10 in the project `.venv`; these timings do not
  predict native Fedora Python 3.13 / RTX 4090 throughput.

The exact extracted JSON evidence is in the three compressed files named after
these runs. `capture_cpu_evidence.py` verifies scientific seals before exporting
their documents and source hashes. These excerpts omit binary teachers and
checkpoints and are **not resumable run archives**. The development atlas is a
separately labeled display revision; the original sealed figures are unchanged.
Open [the revised atlas](development-atlas/index.html) for all 16 figures and
[the validation record](validation.json) for commands, versions and run counts.

## Decisions supported by this small development run

### R1 — better optimization does not reliably improve generalization

The two-gain direct solve reached numerical objective gaps of 4.18e-7 on the
discrete track and 3.06e-7 on the Galerkin track, below its declared 1.01e-6
tolerance. These are floating-point optimization diagnostics, not certificates.

| Track | Fixed physical gains | Direct fitted gains | Direct training objective | Matched Adam objective |
|---|---|---|---:|---:|
| Discrete | (1, 1) | (0.99832, 1.30221) | 0.02142271 | 0.02142265 |
| Galerkin | (1, 1) | (1.11044, 1.75000) | 0.24329060 | 0.24516585 |

The discrete direct and Adam solutions are essentially equivalent on the
training objective. The Galerkin direct solution reaches the cubic coefficient
bound. Neither fact establishes that its physical correction is useful on new
fields.

| Track | Fixed parent validation RMS | Direct-fit validation RMS | GL4 validation RMS | Nested quadratic+cubic validation RMS |
|---|---:|---:|---:|---:|
| Discrete | 3.653e-6 | 4.104e-6 | 1.109e-6 | 3.366e-7 |
| Galerkin | 6.123e-6 | 5.888e-6 | 5.240e-6 | 4.405e-6 |

These are arithmetic field-mean RMS values over eight validation fields and
four queries per field. The normalized peak-sensitive objectives and worst
maximum errors have different rankings; the exact evidence retains both.
Training optimization alone is not a sufficient explanation of the remaining
gap. The full native protocol must test whether this persists across more
fields; do not expand the coefficient bounds after seeing these results and
call that the same preregistered test.

### R2 — finite-step quadrature is a more informative target

There are eight paired parameter/track cases on one deterministic continuous
field in the 16² diagnostic, not eight independent fields. All reference
targets were accepted; 56 method endpoints were finite, with no sampled
interval violations.

- GL4 plus the nested cubic term improves RMS over GL4 in 8/8 cases by
  4.19–48.63×, with the difference separated from estimated reference
  uncertainty. This is an accuracy observation, not a cost result.
- GL2 plus the filtered leading cubic term improves over GL4 in only 4/8.
  At diffusion number 1 and reaction number .1 it is 16.15–16.91× worse.
  Adding a leading cubic term does not fix inadequate finite-step quadrature.
- FD-versus-Galerkin differences are 1.218–1.839e-4 RMS, much larger than the
  corrected temporal error. A discrete-equation improvement cannot be relabeled
  as an equally large continuum improvement.
- The measured unresolved-scale discrepancy on this smooth template is below
  reference uncertainty. It supplies no resolved closure benefit.
- Quadratic/cubic cosine ranges from .432 to .754; normalized pair condition
  ranges from 1.59 to 2.67. The new diagnostic avoids artificial orthogonality.
- The vector decomposition residual is zero; its squared-RMS reconstruction
  differs by at most 6.62e-24. Cross terms are explicitly retained.

The independent 8² smoke field shows the same qualitative reversal. Two
deterministic fields do not establish a population effect. Full supplies more
regimes, phases and independent stiffness interventions.

### R3 — FNO training now has genuine alternatives, but is still unresolved

All four FNO family/track combinations select trained checkpoints, with the
lower-rate unclipped setting preferred in this bounded run. The long-trial tie
defect no longer removes optimizer alternatives. It does not follow that
clipping alone caused the old weakness: learning rate and objective scale also
vary, and timing-capped trials perform different numbers of updates.

| Track | Scaled FNO validation RMS | Mean-separated FNO validation RMS | Selected two-basis RMS | GL4 RMS |
|---|---:|---:|---:|---:|
| Discrete | 9.489e-6 | 9.327e-6 | 3.655e-6 | 1.109e-6 |
| Galerkin | 1.321e-5 | 1.279e-5 | 6.123e-6 | 5.240e-6 |

The Galerkin two-basis arm selects initialization. FNO still underperforms GL4
here; this local adapted, small-data comparator is not a reproduction or defeat
of the FNO paper. Inspect tiny-fit criteria, achieved update/time budgets,
maximum/mean errors and larger native trials before attributing a general
capacity limitation.

### R4/R5 — shared transforms deserve native evaluation

Across two fields and two schedules per track at 32², batched GL4 versus
sequential GL4 gives median paired complete-call speed ratios of **1.79–1.90×**
on the discrete track and **1.35–1.41×** on the Galerkin track. All eight paired
endpoints meet the declared RMS and maximum targets; maximum FP32 difference
is 8.94e-8 (discrete) and zero (Galerkin). Five timing rounds per case support
descriptive development ratios, not confidence intervals or GPU claims.

For the high-pair Galerkin field at one step, GL2 misses the maximum-error
target: RMS 1.823e-5, maximum 2.981e-5. Midpoint-elided GL3 passes at
1.619e-6 / 2.912e-6; batched GL4 is still more accurate at
1.208e-7 / 3.805e-7. No universal GL3 speed or accuracy advantage is established.

This supports evaluating exact GL4 execution reuse alongside GL3, not replacing
GL4 on the strength of node count. DF is an unfused attribution control in these
timing panels; no strongest-classical cost superiority follows. Paired grid
observations remain paired, even when their case identifiers include the grid.

### RX — retain the negative memory result

The separate full-sized CPU memory probe in
`memory-development-summary.json` used 48 training, 16 validation and 16
development-test fields, 32 autonomous steps and two paired noise draws.
Its explicitly recorded settings differ from the final native protocol; it is
a development probe with its own seed and hashes, not execution of native full.

Validation chose zero deterministic memory gain, recovering coarse RK4.
Clean-fitted memory improves clean errors but is badly noise-sensitive;
noise-aware fitted memory remains worse than RK4 in every tested noise/rate
scope. Retain the protected bounded replication, then stop escalation unless
fresh evidence changes the premise. More expressive memory is not justified
by this negative result alone.

## Next decision

Run the native smoke, then the declared full development portfolio unchanged.
Prioritize accuracy-qualified GL4 execution reuse and finite-step interaction
quality. Keep coefficient attribution and FNO fairness moving; protect the small
memory falsification budget. Only after these results should a smaller set of
methods and claims be frozen for new, untouched confirmation fields.
