# Protected resolution exploration

The 64²/128² follow-up keeps the previous adjacent CPU findings intact.
Historical E01's amplitude-only cubic selector and D07's temporal encoding
failed their relevant cost/accuracy gates. Historical E02's signed cubic
modulation also remains a negative finding; it is **not** renamed as the new
temporal experiment.

The new variants have separate IDs **HR-E01** and **HR-E02**. In the complete
resolution program, each grid/track pair receives 450 seconds of discretionary
exploration, for 1,800 seconds total. This is 20% of its 9,000-second
discretionary allocation, separate from reference generation and reporting.
No main-campaign budget is transferred. All four planned development parents
remain in denominators if a bounded stage runs out of time.

## HR-E01: interaction-informed cubic work selection

The changed computational premise is that nested cubic transport may become
more expensive at 64² and 128². The new selector also examines an interaction
that the solver already needs, rather than only fluctuation amplitude.

After computing the normalized four-node quadratic correction `Q`, form

\[
\eta=h r\|u-\bar u\|_\infty\|Q\|_{\mathrm{RMS}}.
\]

Compute the cubic term only when `eta > 0.1 * target`. This is a deterministic
heuristic, **not a bound on the cubic error**. The signed physical interactions
are formed before their norm is taken; phase cancellation can affect the
indicator, but the statistic does not reconstruct all cubic directions.
All selection work occurs before the expensive cubic function. A test passes
a deliberately failing cubic callback to verify that skipped work is absent.

Compare against both always-quadratic and always-quadratic+cubic. A speedup
over the expensive control alone is insufficient if quadratic alone is already
cheaper and meets the same RMS and maximum targets. The variant must retain
its frozen 5% relative RMS regression allowance against cubic, pass the
accuracy target, and beat the fastest accuracy-qualified contained component
by at least 1.10 on the tested parents. Reference uncertainty remains in these
checks. Near-unresolved control errors yield NA rather than an optimistic
ratio. A nonfinite prediction is a failure.

These are explicitly new development questions. The tolerance-related
heuristic does not retroactively change the earlier E01 gate or turn its
negative outcome into a success. No threshold is fit to the inspected parents.

## HR-E02: complete temporal reuse at larger grids

The encoding interpolates `Q(h)/h³` with a degree-six Chebyshev polynomial.
Its factor `h³` preserves the zero-time correction; numerical parity remains
necessary, and this factor is not a general stability or order proof.

Encoding samples are computed in FP64 on the declared device. This avoids
amplifying FP32 cancellation at small quadrature times. The interpolation fit
runs on the CPU; all transfers and preparation are charged. The stored
coefficients, physical backbone and decoder use FP32. Seven coefficient
fields are stored, without an explicit all-mode-pairs tensor. The polynomial
index is an information channel, not a spatial coordinate.

For each fixed parent, compare 1, 4 and 8 same-state queries against direct
four-node correction and classical DOP853 dense output for the complete
finite equation. The DOP853 control runs a CPU FP64 Torch/NumPy RHS, and its
query transfer to the same output device/precision is charged. It receives
the same opportunity to construct once and answer many queries. This adapter
is a disclosed comparator implementation, not a reproduction of an optimized
reference library.
The direct GL4 comparator also caches its diffusion multipliers at quadrature
nodes. Its state-dependent transported fields and products are recomputed;
operator caching never reuses a stale prediction. Formula parity is tested
against the original uncached correction in both equation tracks and precisions.

Reported complete cost is **measured preparation plus measured cost of the
whole query group**, not the number of queries times single-query latency.
Preparation is one measured build; its uncertainty is not hidden by treating
it as a repeated measurement. Query timings use randomized paired order,
separate first invocation, and CUDA synchronization when applicable. Stored
bytes and measured GPU allocator peaks have explicit scopes; the latter
include other resident comparator caches and are not isolated model footprints.

A separate two-step autonomous experiment refreshes the encoding on its own
predicted state at each step. This is charged as two new encodings and compared
with a two-step direct correction. The same state-dependent encoding cannot
be reused across a changed state, operator, grid, precision or valid interval.

## Reference scope and decisions

Inputs and accepted FP64 endpoint references come from verified sealed
**training-bank parents**, with the exact stored physics and domain for each
parent. They are development evidence and cannot become untouched evaluation
data. No expensive teacher bank is regenerated inside exploration.

The current bank contains `.04` and `.12` endpoints in the full program, and
`.04` in its smoke. It does **not** contain every requested interior time.
Consequently, an eight-query interpolant can have a measured cost and accepted
endpoint error while its complete-trajectory accuracy and utility remain
**NA**, whether its measured query group is faster or slower. An
endpoint-qualified comparator may itself miss an interior target. Neither
agreement with the direct quadratic approximation nor the classical control
alone creates independent interior truth. Endpoint failures and endpoint-only
cost outcomes remain explicit separate observations. A favorable timing
result could motivate a bounded interior-reference experiment; it cannot
authorize a temporal superiority claim, and a slower timing cannot reject
the full query workload before its competing methods are accuracy-qualified.

The finite FD/nodal and dealiased Galerkin equations remain separate. The
Galerkin label does not mean continuum accuracy is established. No result is
promoted into the main TDN architecture automatically.

Machine-readable outputs are `resolution_exploration_scope.json`,
`resolution_exploration_rows.json`, `resolution_exploration_costs.json`, and
`resolution_exploration_decisions.json`. They include parent identities,
physics, actual timings, reference status, missing evidence, resource/device
scope, and completion denominators. Energy and monetary cost remain NA when
measurements or rates are absent. CPU structural tests and collected CUDA case
names are never reported as actual GPU validation.
