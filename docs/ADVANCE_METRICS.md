# Reading the advance atlas

Each report produces separate 300-DPI PNG figures, a vector PDF atlas, and an
HTML index under `report/figures`. `chart-data.json.gz` is the single canonical
display-data projection; `manifest.json` records exact source hashes, JSON
pointers, source counts, omissions, and each figure's reading guide. Original
stage artifacts remain untouched. Large raw timing arrays, checkpoints and
spatial arrays stay in their source stages instead of being repeated in CSV
cells, JSON copies, and Tower tables.

The engine verifies stage inputs. The report hashes every consumed file again,
refuses symlinks or changing files, selects the aggregate/frozen catalog as the
canonical source when available, and explicitly identifies partial cohorts.
Missing or unparseable inputs become visible omissions and NA panels. A
successfully rendered atlas cannot establish scientific success.

## Visual semantics

- **Ours:** triangle markers and thin solid lines.
- **Theirs:** circle markers and thin dashed lines, including explicitly adapted
  FNO controls; this does not imply published benchmark reproduction.
- **Analytic controls:** square markers and thin dash-dot lines. Non-neural
  fitted controls have their own diamond marker; unknown roles stay unclassified.
- Every panel states which direction is desirable. Learned coefficient sizes,
  physical horizons and directional physical amplification do not have a
  universal higher/lower-is-better interpretation.
- Dense loss curves display continuous translucent observed low/high bands,
  with thin opaque bounds. Connections interpolate adjacent window summaries
  only. Empty windows remain gaps. These are **not confidence intervals**, and
  no smoothing changes a raw loss, endpoint, fit selection, or inference claim.

Learning phases remain separate. `tiny_overfit` evaluates the **same single
training example** as its diagnostic validation target; it tests optimization
capacity and cannot establish generalization. `equal_time` is a separately
bounded training diagnostic. Selecting the best validation checkpoint across
permitted trial phases does not make the final catalog an equal-training-compute
comparison; actual work and every tuning trial remain visible.

Accuracy and latency panels separate spatial equation, grid, physical endpoint
and training budget. Frozen controls are displayed alongside each fitted data
budget using their existing measurements, not a new independent sample. RMS,
maximum, evolving mean, centered spatial error, physical-frequency bands,
physical bounds and Lyapunov energy answer different questions. The energy
uses the specified discrete/Galerkin inner product, including Galerkin Nyquist
mass weights; sampled nodal L2 energy is not silently substituted for it.

Positive endpoint-error summaries use field-balanced geometric means with
descriptive crossed field/training-seed bootstrap intervals. When exact zeros
occur, linear-scale arithmetic summaries keep them visible. At least five
independent fields and a complete crossed design are needed for intervals.
Repeated grid points, schedules, seeds and horizons do not increase the field
denominator. Intervals are not adjusted for multiple comparisons. Fixed
physical-frequency bands retain their physical boundaries across grid sizes.

## Costs and limitations

Validation-locked latency includes only qualified comparisons and must be read
beside full field coverage and missing/failed schedules. An oracle-fit residual
ceiling is representational headroom, not a deployable result. Observed slopes,
equivariance checks, composition defects and sampled tangent directions do not
prove global method order, stability or an operator-norm bound.

First invocation is first in its timing group, not guaranteed process-cold
startup. Separate memory-probe absolute peaks include resident models/caches;
incremental peaks exclude the baseline and are not a complete deployment
footprint. CPU allocator peaks remain NA. Raw paired timing rounds remain at
their source; sparsely sampled latency tails are not population guarantees.

Amortization joins actual locked `cost_seconds` by parent, grid, spatial target,
physical endpoint, tolerances and paired seed/data count. It never chooses a
lucky competitor seed. Nonpositive margins and unqualified comparisons receive
no break-even estimate. A training-only estimate is explicitly a partial,
optimistic cost scenario: reference, preprocessing, validation, failures and
deployment expenses are not assumed to be zero. Money and energy charges need
measurements or declared rates and remain unknown otherwise.

XH causal-history, XR rational-response and XT tangent experiments retain their
own workload definitions and negative outcomes. Their response/closure errors
are not pooled into full-PDE solver scores. History acquisition and repeated
query encoding are charged where measured; repeated per-prediction teacher
metadata is never summed as distinct reference generation.
