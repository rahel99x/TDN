# Native Resolve kernel review

**Batched GL4 is a demonstrated execution improvement over the project’s sequential GL4 implementation. The cheaper GL3 representation does not establish an additional cost advantage on this RTX 4090.** This result strengthens the analytic control; it does not establish a learned-model advantage, superiority over published FNO, or an optimal numerical solver.

Source `f07f9d56ccaee85b73b8d090aec3bc22ac067cf9`. Four native CUDA stages (jobs 870–873), discrete and Galerkin tracks at 64² and 128². All stages completed with zero correctness failures. The four mandatory suites contain 80 CUDA case executions, 0 failures and 0 skips.

## Evidence and denominators

- 8 methods × 2 schedules × 4 field cases per stage × 4 stages = **256 endpoint rows** and 5,120 warm complete-call observations. There are 96 same-rule execution comparisons and 96 GL3 representation comparisons; each uses 20 randomized paired rounds.
- The 16 reference cases are repeated uses of **four independent phase-seed clusters**, two low-mode and two high-pair. Tracks and schedules reuse those phases. Low-mode physical fields are shared across grids; high-pair frequencies change with grid size. Do not treat the 32 conditions, 256 endpoints or timing repetitions as independent fields.
- The four draws vary phase only: amplitudes, mean, variance, physics and two-mode support are fixed within each regime. Each two-mode pair has independent wavevectors, so its phase choices describe translations of one continuous field shape. Galerkin translation-equivariant error norms are therefore nearly redundant; discrete sub-grid phases may change aliasing but do not add new amplitude or spectral regimes. This is a narrow execution panel, not broad statistical generalization.
- All 16 references passed actual 128/256/512-step refinement, with maximum empirical RMS uncertainty 1.57e-10 and maximum uncertainty 2.53e-10. These are same-grid finite-equation teachers, not spatially converged continuum truth.
- All 256 endpoints are finite, with no sampled interval violation or endpoint energy increase. **198/256 meet both RMS and maximum targets of 2e−5.** Mathematical correctness and practical feasibility remain separate.
- Independently reconstructed all recorded speed ratios, eligibility decisions, medians and raw randomized-round alignments: **zero mismatches**.

## Same-rule execution result

Ratio = sequential control time / batched candidate time; higher is better. Ranges below are the eight condition-specific median paired ratios, not confidence intervals.

| Track | Grid | GL4 speed ratio range | Accuracy-qualified conditions |
|---|---:|---:|---:|
| discrete | 64² | 2.029–2.256× | 8/8 |
| discrete | 128² | 2.017–2.121× | 6/8 |
| continuum | 64² | 1.481–1.507× | 8/8 |
| continuum | 128² | 1.473–1.483× | 8/8 |

GL4 was faster in **640/640 paired rounds**; 639/640 exceeded the declared 1.2× ratio. Every condition median exceeded 1.2×. Maximum endpoint disagreement between the two executions was **1.788×10−7**, well below the 3×10−6 parity threshold. This supports adopting the vectorized implementation as a frozen analytic baseline. It does not imply bitwise identity: easy cases at approximately 5×10−8 RMS are near the FP32 error floor.

GL2 batching also clears the speed threshold in all 32 condition medians (1.233–1.568×), but only 20 conditions pass accuracy. Midpoint-elided GL3 improves over sequential GL3 by 1.346–1.817×, with 24 eligible conditions. These are execution improvements, not proof that GL2 or GL3 solves the harder workload accurately.

## Representation, accuracy and practical choice

| Method | Eligible endpoints / 32 | Meaning |
|---|---:|---|
| df | 20/32 | Cheapest method when its accuracy suffices |
| quad2_batched | 20/32 | Two active quadrature nodes |
| quad3_midpoint_elided | 24/32 | Two active nodes plus an implicit zero-response midpoint |
| quad4_batched | 30/32 | Four active nodes with shared transforms |
| analytic_quad_cubic | 30/32 | Sequential GL4 quadratic plus nested cubic |

GL3 and batched GL4 have essentially the same runtime here: GL4/GL3 condition-median speed ratios range **0.981–1.042×**; none reaches the 1.2× threshold. GL3 has lower RMS on only 8/32 conditions, predominantly at the easy FP32 floor; GL4 can have **18.23× lower RMS**. GL3’s coverage is worse, so its smaller node count does not justify replacing batched GL4 on this hardware.

At 64² high-pair, one GL3 step fails the maximum-error target on both tracks, while one GL4 step passes. Two GL3 steps pass but take about twice the time of one GL4 step. At discrete 128² high-pair, GL3 fails even in two steps; GL4 needs two steps and passes. At Galerkin 128² high-pair, even DF passes, and its prediction is identical in the reported one-step errors to the more expensive corrections. This is an easy absolute-tolerance workload despite its large dimensionless diffusion.

An explicitly exploratory post-hoc 1/2-step menu selects DF in 10/16 field/track/grid cells (all eight low-mode cells and both Galerkin-128 high-pair cells), and batched GL4 in the other six. No GL3 winner emerges. These selections use test references and are **not a deployable policy or a validation-selected frontier**. DF receives the same ordinary step loop; the panel did not implement adjacent-heat fusion for two-step DF, so it cannot establish an optimized lower cost bound against all classical implementations.

Nested cubic is much slower and generally offers no useful extra error reduction on the high-pair fields at these settings. Its 30/32 coverage equals GL4. That finding is local to this two-mode panel; it is not a general rejection of cubic structure. GL3’s 4.66–7.68× advantage over the expensive nested control must not conceal its lack of advantage over batched GL4.

## What the math and profiler explain

For centered state v, define G(s)=E(h−s)P[(E(s)v)²]. The correction is −r h q(h) Σᵢ wᵢ q(sᵢ)[G(sᵢ)−G(h/2)]. GL3’s midpoint term is exactly zero, leaving nodes 1/2±√15/10 with weights 5/18 each; the implicit midpoint has weight 4/9. The complete rule remains normalized and degree-five polynomial-exact. This is not a fifth-order full solver claim.

Batching shares the centered-state FFT, evaluates node transports/products together, and combines the weighted transported spectra before the final inverse FFT. The same operator, precision, product, cutoff and physical formula are retained. It is an implementation change, not learned physics.

| One-step profile | Sequential GL4 | Batched GL4 | DF base |
|---|---:|---:|---:|
| Discrete FFT dispatch calls | 24 | 8 | 4 |
| Galerkin FFT dispatch calls | 276 | 212 | 196 |
| Discrete `_local_scalar_dense` calls | 20 | 10 | 12 |
| Galerkin `_local_scalar_dense` calls | 43 | 13 | 15 |
| Discrete allocation-related operator calls | 106 | 53 | 28 |
| Galerkin allocation-related operator calls | 1028 | 788 | 700 |

Counts are measured PyTorch invocations, not FLOPs or numbers of transformed scalar fields. Batched node transforms contain more fields per invocation. Allocation operators overlap; `scalar_reads` in the original profile adds wrapper `item` and backend `_local_scalar_dense`, so this table uses the latter rather than double-counting scalar reads. Copy-category counts also include wrappers and no-op conversion opportunities and must not be described as PCIe traffic.

The Galerkin backbone performs four projected RK4 reaction subdivisions, hence sixteen dealiased quadratic RHS evaluations before the correction. Its 196 FFT dispatch calls remain in the 212-call optimized GL4 path. The batch optimization removes much of the *additional* correction dispatch and synchronization overhead; changing from four to two active nodes then does little to end-to-end latency. This makes projected reaction/backbone overhead and static metadata validation promising profiling targets, while preserving matched numerical accuracy and comparable optimization opportunities for controls.

## Grid and stiffness interpretation

| Grid | High-pair modes | Discrete hλ range | Galerkin hλ range |
|---|---|---:|---:|
| 64² | [(29, 1), (30, 1)] | 7.714–7.808 | 15.956–17.074 |
| 128² | [(61, 1), (62, 1)] | 31.306–31.400 | 70.531–72.861 |

At h=.12, κ=.004 and r=3, hr=.36 on every kernel field. The diffusion symbols are the actual FD symbol 4κN²Σsin²(πk/N) and Galerkin κ(2π)²Σk². Two-step schedules halve these per-step numbers. The high-pair workload changes its physical frequencies with N, so its 64→128 comparison does not isolate resolution scaling or continuum generalization. Very strong damping moves the important interaction toward the start of the step; interior Gauss nodes may miss that narrow layer. The N128 Galerkin errors are smaller partly because the absolute nonlinear effect is smaller, not because generalization improved. A fixed 2e−5 tolerance can make the stiffest case operationally easier.

## Runtime, memory and scope

The four experiment bodies consumed **167.22 s**: 67.05 s reference generation, 69.93 s warm timed calls, 3.52 s first invocations, 3.51 s measurement warmups, 3.50 s memory probes, and 7.02 s profiling and its warmups. The remaining approximately 12.67 s covers other metric/report/orchestration work. These are experiment-body components; allocated Slurm time additionally includes preflight, required CUDA tests and launch overhead.

Maximum stage PyTorch allocated peak was **7.792 MiB**, reserved peak **30 MiB**, and maximum incremental isolated probe peak **6.227 MiB**. Maximum sampled physical-device use was 2.050 GiB and includes desktop/other resident allocations. The minimum permitted soft budget was 16.171 GiB. No memory-pressure or allocation-limit failure occurred; more VRAM is not the next bottleneck.

Timed complete calls start from resident device state and include the full scheduled solver, validation and cache use, but exclude caller-side input transfer, model construction and metric transfer. They are not complete service startup/transport costs. First-call values follow cache clearing but not a cold process: the profiler already exercised libraries, CUDA plans and the device. Twenty warm rounds support observed latency variation, not p95/p99 service guarantees. Profiling occurs separately and includes startup overhead (especially the first profile); do not rank methods using profiler durations. No power, energy, monetary rate, compiler optimization or cross-device throughput result was measured.

## Next decisive experiment

1. Preserve sequential GL4 as an exact attribution control and promote batched GL4 to the strong analytic baseline for subsequent learned comparisons. Expand parity tests to long repeated rollouts, non-square grids, and harder hλ/hr ranges before broader deployment.
2. Freeze a modest fresh panel with many independent phase/shape parents, independent diffusion/reaction sweeps, and both fixed-physical-frequency resolution tests and separately labeled grid-relative stress cases. Retain same-grid and spatially refined teacher tracks separately.
3. Test batched GL4/GL8 and a boundary-layer-aware or exponentially weighted interaction rule. At large hλ the missing early-time interaction is a more concrete mechanism than reducing GL4 to two active nodes. Keep a reference-informed oracle only as a diagnostic; any deployable rule must use state/physics features and independent validation.
4. Profile and optimize the Galerkin projected reaction/backbone and avoid repeated host scalar validation of immutable metadata, without changing the numerical equation. Give DF/nested/learned competitors the same caching, batching and fusion opportunities. Measure accurate one-, two- and several-step calls before any deployment-policy claim.
5. Deprioritize GL3 as a standalone RTX4090 latency claim: fewer nodes did not avoid enough launch/backbone work. Revisit only for larger spatial batches or different hardware where arithmetic or memory traffic, rather than host/dispatch overhead, becomes limiting.

All tables derive from `kernels-review.json`; `kernels-review.py` reconstructs them from verified canonical artifacts using `review_io.py`. No source or historical scientific artifact was edited.
