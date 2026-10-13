# Staged tangent-response implementation

This program implements the [complete uploaded assignment](TANGENT_RESPONSE_RESEARCH_HANDOUT.md)
in `tdn.analysis.tangent_response`, alongside the existing Response and Response
Diagnostic experiments. The [native runbook](TANGENT_RESPONSE_RUNBOOK.md) provides
the implemented launch, inspection, export, continuation and archive commands.
The [verification record](../results/tangent-response-verification/README.md)
separates completed software checks from outstanding native and scientific
claims. The original handout remains unchanged.

The question is whether the spatially varying linearization along a fixed
smooth anchor offers useful physical correction beyond its matched mean
response, whether declared response actions admit useful compression, and
whether a learned approximation retains that gain at a lower complete cost.
A completed pipeline alone answers none of these questions.

## Implemented stages

| Stage | Implemented work | Evidence required before native expansion |
| --- | --- | --- |
| Mechanism | Matrix-free full-field tangent evolution; matched mean response on the same FP64 path/residual; independent NumPy/SciPy checks; first/second corrections, nonlinear error and strong classical controls. | Literal eligible A `PASS` on the primary same-grid FD diagnostic bank. |
| Compression | Predeclared residual, Fourier and signed-random sources; train-only source/response/endpoint bases; ranks 2/4/8; held-out sources and parents; per-state oracle headroom, conditioning and deployment sensitivity. | Literal B `PASS` for the declared reusable-action scope. Task-residual fits retain their narrower scope. |
| Fit preparation | A separate parent cohort and prospective candidate menu; independent references and labels; uncertainty, physical-signal and output-precision screening; all rejected slots retained. | Literal C0 `PASS` on the actual screened bank. |
| Calibration | Both learned arms execute the declared actual workload, including gradients and validation. Bank bytes, source, software, layout and work hashes bind measured allowances. | Literal native calibration `PASS`; CPU rates cannot substitute. |
| Learning | Action then endpoint objectives; deterministic full-parent weighting; train-frozen gain normalization; conditional affine-readout diagnostics; validation-only deployment selection; initialization, final and best-training checkpoints; exact journals. | C1 fitting and C2 held-out gain must both pass, with declared work and seeds complete. |
| Utility | Complete inference calls, cache creation/invalidation, resident and transfer timings, trusted fresh-process calls, validation-frozen classical schedules and RMS/maximum work-precision panels. | Source-bound native readiness and C `PASS`; D assesses measured accuracy/cost, including the strongest qualifying control. |

Native phases stop on `STOP`, `UNRESOLVED`, missing or invalid evidence. Decision
and report observers still finish after partial failures. A CPU smoke deliberately
exercises the bounded interfaces through all stages, but every resulting
decision is marked `IMPLEMENTATION_ONLY` and ineligible for native promotion.

## Architecture, data and precision

The learned response uses periodic full-field input/output factors and signed
causal temporal channels. Factors may carry a mean. The corrective response is
zero for constant anchors and zero reaction, while the full response is the
matched mean/heat evolution. Zero diffusion is not an operator null. Identity
at equal source/endpoint times, translation covariance, the physical wave-number
convention and first-update head gradients have explicit checks. Only the final
corrective readout starts at zero.

The minimal comparison has six named methods: anchor, matched mean response,
oracle tangent response, fixed shared basis, learned response and matched-anchor
residual FNO. Only the last two receive optimizer work. Native pilot defaults are
rank 4, width 16, depth 2 and 300 updates with one paired seed; development has
1,000 updates and three paired seeds after an eligible pilot binding. The
reduced CPU profile has rank 2, width 4, depth 1 and two updates. Conditional
compact path/residual encoders, nonlinear iterations, confirmation and broad
capacity sweeps are not automatically launched.

Mechanism and fit cohorts are independent. Horizons, shifts, grids and probes
are descendants of a parent and never increase the independent-parent count.
Same-grid FD is primary; same-grid dealiased Galerkin has separate physical
norms and scope. Projected-fine dynamics are rejected because this program has
no implemented closure for that target.

Teachers, smooth paths, residuals, tangent actions and defect subtraction use
FP64 from the actual FP32 initial input. Models and deployed outputs use FP32,
with TF32 off. Reports distinguish physical defect, arithmetic bridge, final
output rounding and deployment error. Both anchor evaluations are charged when
the decoder uses the FP64 bridge. Teacher/action labels are unavailable to the
deployment interface. Reference acceptance retains the prospective `1e-10`
combined uncertainty budget and uncertainty-separated effect requirements.

## Operations and research records

The native coordinator runs locally through Raven's captured launch contract.
Scientific work runs in real Fedora Slurm allocations of at most 35 minutes.
Workers verify actual allocation ownership and the sealed native CUDA roster,
strip the outer Raven environment, and collect actual CPU/RSS/assigned-GPU
telemetry. Existing CPU/host/VRAM limits remain in force. There is no automatic
retry, aggregate campaign ceiling, pending-job cap or silent CPU fallback.

Source/software/protocol/data hashes and exact optimizer/RNG/sampler journals
govern continuation. Recovery uses a fresh root, retains all original attempts
and costs, and resumes only compatible unfinished work. Archival requires
terminal jobs and a stopped coordinator, moves only the selected campaign, and
never deletes or overwrites evidence.

Portable Raven projects contain source-traced scientific rows, uncertainty and
gate definitions, actual resource graphs, omission inventories and relative
links. Large rosters/data use bounded pages and chunks that satisfy Raven's
standard reader limits. Energy and monetary cost remain unavailable without
measurements and rates. The [analysis](TANGENT_RESPONSE_ANALYSIS.md) derives the
identities and conditional bounds; the [literature audit](TANGENT_RESPONSE_LITERATURE.md)
states the established comparisons and limits any novelty claim. The handout's
[legend](TANGENT_RESPONSE_RESEARCH_HANDOUT.md#18-legend) defines the terms.

The next numerical action is the primary **native same-grid FD mechanism
pilot**, after doctor and the actual allocated CUDA checks pass. Its measured
A gate determines whether compression is justified. CPU completion does not
authorize a broader experiment or establish a learned speedup.
