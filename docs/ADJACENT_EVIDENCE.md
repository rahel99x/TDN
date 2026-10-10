# Evidence and mathematical boundary of the adjacent study

This study supports the original question in [the implementation specification](IMPLEMENTATION_SPEC.md): preserve reliable physical subsolvers, approximate their finite-time splitting defect, and determine whether an explicit temporal representation improves complete accuracy–cost. It does **not** replace that objective, alter the active [portfolio hypotheses](PORTFOLIO_HYPOTHESES.json), or authorize promotion into the main architecture. An interaction-origin channel is an information axis, not another spatial dimension.

The adjacent source was developed in `/workspace/TDN-adjacent`, outside the active `/workspace/TDN` source tree. It has separate protocols, budgets, field identities, run directories and reports. Old inspected confirmation is development evidence. The full portfolio's pending results are unavailable here; no conclusion below assumes their outcome.

## What the implementation actually contains

The inspected [portfolio models](../tdn/analysis/portfolio/models.py) retain the full physical field for a diffusion–reaction–diffusion backbone and a signed quadratic splitting correction. Physical transport, products, quadrature and most transforms are analytic. “Rank one” names one symmetric quadrature pair; it is not a matrix-rank theorem. Its small network conditions a response, not the complete spatial field evolution.

| Existing family | Learned or fitted component | Information and mathematical boundary |
|---|---|---|
| Historical `rank1_frozen` / `historical_half` | None at inference | Two weights initially sum to **0.5**. It remains a labeled historical control, not normalized GL2. The historical trainable total gain is `2 tanh(a) [1 + 0.5 tanh(network)]`, potentially in `(−3,3)`. |
| `quad2_fixed`, `quad4_fixed`; corresponding `_full` | None | Normalized Gauss weights sum to one. Fixed and full variants differ in output compression. Their physical quadratic calculation receives the full field. |
| `quad2_amplitude`, `quad2_nodes`, `quad2_joint` | Scalar gain, symmetric node pair, or both | Matched output support and spatial operators distinguish normalization, node placement and fitting. Gain is `1 + 0.75 tanh(a)`, in `(0.25,1.75)`. |
| `quad2_linear`, `quad2_conditioned` | Four-coefficient affine-feature response, or small neural conditioner plus global node/gain | Original features are mean, `log(1+hr)` and `log(1+h active_diffusion_rate)`. The active rate uses the **declared** spatial operator. Both effective gains are in `(0.25,1.75)`. The affine fit is non-neural. |
| `conditioned_rich` | Seven-feature scalar conditioner | Adds variance, mean absolute fluctuation, spectral-rate spread and nonlinear strength. It has already been implemented; more features are not automatically new information that matters. |
| `band_gain` | Three conditioned output-band gains | DC, lower output band and upper retained output band, with gains in `(0.5,1.5)`. These identify **where** an interaction lands, not which input bands produced it. |
| `residual_quad2` | Bounded learned scalar residual on the normalized rule | Gain in `(0.75,1.25)`; its admissible oracle differs from the main scalar conditioner. |
| `analytic_quad_cubic` | None | Analytic quadratic plus cubic defect. More work and more spatial directions; it is an essential strong control. |
| Local FNO controls | Full local and spectral learned paths | Fourier truncation does not erase all high-frequency information: the local path and nonlinear products remain. These are local adaptations, not verified reproductions of an FNO paper. |
| Portfolio X01 | Analytic kernel encoded in a Chebyshev temporal basis | Reuses a state-dependent signed-pair encoding. Its observed cost failure is a prior result, not a reason to suppress the comparator. |

Gain intervals above are **implementation bounds**, not stability proofs. Parameterizations can be non-identifiable: a global logit and conditioner bias may generate the same effective gain. Inspect effective responses and sensitivity before interpreting individual learned parameters.

## Compact evidence map

The [initial portfolio review](PORTFOLIO_INITIAL_REVIEW.md) and [historical evidence map](../results/portfolio-development/evidence-map.json) separate observed comparisons from their explanations. The key raw development files below were re-read and their SHA-256 values checked against the retained [publication index](../results/portfolio-development/development/publication-index.json).

| Status | Evidence and scope | Consequence for this adjacent study |
|---|---|---|
| Established finite implementation property | Prematurely removing high input modes removes their low-output quadratic products. This is an algebraic mechanism and has targeted measured examples. | Form physical interactions before output compression; D04 tests signed responses, including cancellations. It does not assert universal superiority over FNO. |
| Measured development observation | On one 16² high-pair Galerkin field at `κ=.004, r=3, h=.12`, GL4 versus GL16 quadratic difference is `6.600e−13` RMS, while the GL4 corrected endpoint error is `3.649e−6`. Quadratic+cubic gives `7.771e−8`. | D01 and D05 distinguish a missing spatial response from insufficient quadrature. This example weakens the “just add nodes” explanation. It is neither a population effect nor a matched-cost win. |
| Spatial limitation | On that field, same-grid Galerkin versus a finer projected solution differs by `4.691e−6`; the actual 2N/4N refinement difference is `1.686e−10`. FD/nodal versus continuum differs by `6.958e−3`. | A temporal correction to the specified finite equation cannot be credited with fixing continuum error. Keep the two targets explicit. Refinement differences estimate uncertainty; they do not certify it. |
| Confounded historical attribution | Historical frozen gain is half-normalized; stronger analytic comparisons also changed node count and output support. | Retain historical results, but use normalized, matched configurations to attribute new effects. |
| Unresolved neural fairness | The two-update CPU smoke selects many initializations; historical FNO runs showed pervasive gradient clipping. | Neither is a credible defeat of FNO. Training adequacy, local paths, loss scaling, data and measured compute require separate checks. |
| Negative cost result | Portfolio X01 needed a modeled **2,062** repeated queries to repay encoding, above its 32-query gate. X03 selectors met error tolerance but achieved full/selector speed ratios **0.691/0.615**, both slower than full work. | D07 must measure actual query-count curves and charge refresh. Selectors must save expensive work before it occurs. A new experiment needs a changed computational premise. |
| Narrow information result | X02 coarse twins share their coarse state but require different future values; its fitted memory rule performs well on a narrow synthetic target. Histories were generated backward at a non-negligible cost. | This is evidence of an information obstruction, not a deployable closure. Any memory follow-up must use causal forward history and charge startup. |
| Measurement defect retained | Original local v1 scaling did not honor its declared two-step schedule; v2 corrected it. Historical endpoint and cold-cost analyses have other documented repairs. | Use only compatible, corrected artifacts for those claims; do not overwrite historical results or relabel old costs. |
| Untested adjacent hypothesis | Separating LL/LH/HH may expose useful directions hidden by a common gain. | Fixed recombination must equal the original normalized correction. Improvement requires unequal responses and must beat scalar, output-band, fitted, analytic and learned controls at full cost. |
| Untested adjacent hypothesis | Fine-scale persistence, phases or physical spectral moments may predict difficulty better than a nominal roughness label. | Measure those properties and their evolution; do not interpret an initial “2.X-D” label as solver difficulty or neural-network dimension. |

The portfolio smoke's 2,024 endpoint rows came from **two** independent fields. All 48 primary scientific claim records were NA for inadequate independent sample size. Row counts, successful jobs, numerical checks and scientific success remain separate.

For reproducibility, the raw diagnostic file has SHA-256 `425bf456bfa1781697c23b7faf76f0537ab01d8fe728b6cf2fc91f28addfebc6`; X01's amortization record has `c5665ac690ec706440e7c5b98c26d6305afe2db17b9e7928a8ef7e9274b991d5`; X02's fit record has `1928d13595cb9f18def8d0fa6d6527f2ba7cb3b40f57339d87a33dfde41b2bc0`. These identify existing measurements, not new adjacent results.

## What the roughness labels mean

For `0<α<1`, let `H=1−α` and

\[
W_H(x)=\sum_{j\ge0}2^{-Hj}\cos(2\pi 2^j x).
\]

The classical cosine Weierstrass result applies to integer base `b≥2`, amplitude ratio `1/b<λ<1`, and this prescribed periodic generator. With `b=2`, `λ=2^{-H}`, its graph has Hausdorff dimension `2+log λ/log b=2−H`. Taking its product with an interval gives the ridge-surface dimension `3−H=2+α`. The theorem is about the infinite canonical construction, not arbitrary two-dimensional spectra, arbitrary phase families or a finite computation.

Every implemented field is a **finite smooth trigonometric polynomial**, hence its graph on a two-dimensional domain has dimension two. For `α=1`, the canonical `H=0` series loses the same convergence property; the code labels this a separate finite-bandwidth stress case. It makes no statement that dimension-three graphs are impossible.

The `ridge/structured` generator checks the canonical finite construction. `multiscale_2d` selects independent, non-collinear integer modes in each dyadic annulus and independently randomizes their phases. It has equal shell energy before the declared `2^{-Hj}` amplitude decay. It is genuinely two-dimensional, but the ridge dimension theorem is **not** transferred to it. Empirical structure-function slopes are finite-range measurements; the code reports no slope when fewer than three nonzero separation scales are available.

Parents store physical integer modes, phases and amplitudes once. Fourier orthogonality fixes continuous RMS before sampling. A conservative triangle bound certifies the full continuous physical range. Incompatible mean/RMS/range requests fail explicitly instead of clipping or silently changing amplitude. Exact sampled extrema need not match between phase controls; only the stated range guarantee, mean, RMS and spectral power are matched. Grid refinement samples the same polynomial below Nyquist; adding scales is labeled a changed workload even when total RMS stays fixed. Modes are cycles per domain length; gradient moments use angular physical frequencies `2πk/L`.

Related roughness, phase, bandwidth, translation and grid variants retain a common parent cluster. A global translation is a symmetry control, never a fresh statistical example. Canonical structured ridge checks are deterministic even when a seed argument changes. Independent confirmation requires new parent seeds, not many variants of the same field.

## Prior-art boundaries and remaining review work

This is a targeted conceptual review, **not a newly completed systematic literature search**. No literature-search connector was available in this environment. The handoff's arXiv links were not retrieved through an alternative route to bypass network policy. The repository's [retained source manifest](../results/portfolio-development/prior-art-source-manifest.json) records earlier authoritative implementation retrievals; those records do not establish publication priority for the present architecture.

| Relevant prior work | What is already established | Appropriate claim here |
|---|---|---|
| [Shen, *Hausdorff dimension of the graphs of the classical Weierstrass functions*](https://arxiv.org/abs/1505.03986) | Dimension of the canonical infinite cosine construction under the assumptions above. The supplied handoff cites this result; the primary manuscript was not newly retrieved here. | A validated finite benchmark derived from that construction; not a fractal neural architecture. |
| Bony's paraproduct calculus; bilinear and Volterra expansions | Low/high product decomposition and homogeneous nonlinear interaction structure are established mathematics. Three binary bands here are a finite operational partition, not the complete paraproduct theory. | LL/LH/HH decomposition is new to this tested TDN interface; algebraic decomposition itself is not novel. |
| Hochbruck & Ostermann, *Exponential integrators*, Acta Numerica (2010) | Variation of constants, exponential response functions and quadrature of nonlinear terms are established numerical approaches. | A particular signed splitting-defect channel, approximation or measured cost tradeoff; not invention of exponential propagation. |
| [*Multiwavelet-based Operator Learning for Differential Equations*](https://arxiv.org/abs/2109.13459) | Learned multiscale operator representations are prior art. A local multiscale diagnostic comparator is not an authoritative paper reproduction. | Any claimed benefit must survive a relevant multiscale comparator; generic “uses multiple scales” is not novelty. |
| FNO and DeepONet; retained authoritative implementation sources | Global spectral paths, local channel paths and separable continuous-query representations already exist. | Evaluate the specific information interface, initialization, training and complete cost; neither Fourier truncation nor an explicit query decoder alone establishes a gap. |
| Mori–Zwanzig / optimal prediction and numerical coarse-graining | Eliminating unresolved dynamics can create memory; instantaneous closures can lose information. | A carefully bounded causal closure experiment, with observed failure modes and acquisition cost. |
| Classical Gaussian quadrature and learned quadrature literature | Fitting nodes/weights is an established strategy. Our current project already fits symmetric nodes and bounded gains. | Compare normalized fixed quadrature and simple fits first. Literature-wide novelty of a trained rule remains unestablished. |

Before any publication-level novelty statement, retrieve and review the relevant primary texts and recent descendants, inspect authoritative comparator implementations, and record exact source versions. The adjacent hypotheses currently make no literature-wide novelty claim. Any positive pilot remains new project evidence until an independently frozen comparison supports the narrower stated contribution.
