# Full native frontier review

Review date: **2026-10-08, America/Los_Angeles**. The archive and Slurm records use UTC identifiers.

**The recovery worked. The physical correction has a useful accuracy signal, but the present experiment does not establish practical solver superiority or defeat of the FNO paper. The next experiment should isolate learning beyond correctly normalized quadrature, repair the endpoint schedule comparison, and measure implementation overhead before increasing training.**

Origin: `fedora-frontier-20261008T092558622970Z`, jobs 447–455. Recovery: `fedora-frontier-20261009T005435342501Z`, jobs 456–465. This review reads the supplied data and trusted repository code. No uploaded code was executed, checkpoints deserialized, models retrained or new jobs submitted. Scientific observations and original scores are unchanged.

The [annotated 33-page atlas](../results/frontier-full-review/annotated/frontier-atlas.pdf) and [annotated HD overview](../results/frontier-full-review/annotated/frontier-overview.png) add metric directions and method identities to these results. Every page explains its axes, favorable direction or color, and when no better/worse ranking applies. Ours identifies the proposed learned family and its frozen/ablation controls; Theirs identifies established-method comparators implemented here. FNO labels explicitly say local, and untrained analytic attribution controls have a separate label. These are method roles, not claims of ownership or published-paper reproduction.

The original [33-page native atlas](../results/frontier-full-review/frontier-atlas.pdf) and [HD overview](../results/frontier-full-review/frontier-overview.png) remain unchanged. The annotated version uses byte-identical chart JSON observations; its [provenance and reading guides](../results/frontier-full-review/annotated/provenance.json) record the source and renderer hashes. The [evidence directory](../results/frontier-full-review/) contains archive, seal, coverage, execution, learning and gate audits.

The visual key is consistent across panels: our learned model uses blue upward triangles and a stronger solid line; our frozen control uses light-blue downward triangles and a dotted line; our input-compression ablation uses cyan leftward triangles and a dashed line. Other methods use circles and thinner patterned lines with fixed method colors. Model bars distinguish the main model's solid fill from hatched controls; accuracy heatmaps highlight its row without changing the accuracy colors. Paired comparisons use the candidate's shape, and partial-diagnostic colors retain their evidence-status meaning.

## What completed

All 20 parts and their combined SHA-256 match the index: `494f0b089174495012f535163725729832e295b33f46c7a943bfc5c172ef06ee`. All 8,951 archive files were streamed and hashed. The review verified 16 science/execution seals, 1,963 listed scientific artifacts, all 80 checkpoint byte hashes and recorded predecessor links. These counts include the original partial-outcome report as well as the complete recovery.

The six recovery parts each contain four parents and 3,648 endpoints. Their aggregate preserves exactly **24 independent confirmation fields, 80 model records, 21,888 endpoints and 1,152 paired timing groups**. Exact coverage, intermediate-reference inventories, randomized timing order, repeated timing arithmetic and verbatim merged rows verify. Original job 452's 11,376 partial observations remain excluded from new confirmation.

All ten new jobs completed. Each confirmation part took **11:18–11:57** of allocation time, within the 30-minute internal/45-minute allocation bounds. Every new GPU job ran all 45 mandatory CUDA cases: 405 passes over nine allocations. The original train and failed confirmation jobs each had another 45 passes. These are software-test executions, not independent scientific samples. CPU audit had 520 passes and one optional Tower-checkout integration skip.

There are **24,871 canonical scientific experiment rows and 161,846 checks**, plus one current report row with four infrastructure checks. Partition copies are not counted again as independent scientific evidence.

## Gate interpretation

| Gate | Recorded verdict | What the full measurements mean |
|---|---|---|
| G1: comparison/reference correctness | GOOD | Structural and measurement checks pass; finite tests are not general mathematical proofs. |
| G2: spatial consistency/headroom | BAD | Real temporal headroom exists in 56/144 primary conditions, but most conditions are easy and spatial-generator discrepancies are often larger than tolerance. |
| G3: learning and matched utility | BAD | Learning and precompression improve accuracy under this budget; current rank1 schedules do not beat matched classical cost. |
| G4: accuracy-backed scaling | BAD | All 60 workloads are accurate; only 1/48 paired comparisons clears the 1.2× practical-speed threshold. |
| G5: operational deployment | NA | Correctly blocked before calibration because the frozen candidate lacks classical cost savings. |

The weighted 1–100 scores are check attainment, not model quality percentages. For example, G2 includes intentionally coarse solver candidates that fail accuracy; its BAD verdict does not mean there is no promising regime.

## Learning and precompression: a real, bounded positive result

All **60 trainable final models selected trained checkpoints**, with 12,800 optimizer updates including tuning. The remaining records are 12 frozen rank controls and eight analytic/classical controls. No trainable final model selected initialization.

The table uses the **32-field training subset**, all three seeds, both 32²/64² confirmation grids and five identical primary schedules. There are 720 paired observations per track, but only **24 independent fields**. RMS ratios are competitor error divided by rank1 error, averaged geometrically within each field and then across fields. Larger than one favors rank1. These are descriptive comparisons conditional on the registered seeds.

| Method | Discrete joint-target passes | Continuum joint-target passes | Discrete RMS ratio | Continuum RMS ratio |
|---|---:|---:|---:|---:|
| Learned rank1, 43 parameters | 668/720 | 668/720 | 1 | 1 |
| Frozen rank1 | 570/720 | 576/720 | 2.991 | 2.693 |
| Input-compression ablation | 549/720 | 558/720 | 2.299 | 2.143 |
| Small hybrid FNO, 47,057 parameters | 313/720 | 372/720 | 24.804 | 15.095 |
| Standard hybrid FNO, 1,258,977 parameters | 337/720 | 367/720 | 19.119 | 17.314 |
| Direct FNO | 0/720 | 0/720 | 50,590 | 44,382 |
| Analytic quadratic, four nodes | 235/240 | 234/240 | 0.356 | 0.391 |
| Analytic quadratic + cubic | 239/240 | 238/240 | 0.223 | 0.285 |

Joint accuracy requires RMS and maximum error, including declared teacher uncertainty, at most **2e-5**. Analytic controls have 240 unique primary endpoints per track; they are repeated only when constructing seed-paired ratios. Their smaller denominators do not indicate omitted fields.

The precompression advantage is concentrated where hypothesized: error ratios against the input-compression ablation are about **6.7–8.5× on high-pair fields, 7.0–9.2× on rough fields and 10.6–12.3× near Nyquist**, with only three independent fields in each regime. The ablation is modestly more accurate on low-frequency, mixed and reaction-stiff fields. Precompression is useful here, not uniformly superior.

The mechanism is consistent with

\[
P_K(u^2) \ne P_K[(P_Ku)^2].
\]

Two high-frequency factors can produce a retained low-frequency mode. Forming the nonlinear product first preserves that contribution. This numerical evidence supports the representation choice; it does not imply that every FNO loses those interactions, since the tested FNOs retain a full-grid local path.

## The main attribution gap: the frozen control is half-normalized

The current pair uses nodes

\[
a=\frac12-\frac{1}{2\sqrt3},\qquad 1-a,
\]

which are Gauss–Legendre's two nodes on [0,1]. The branch computes equal weights

\[
w_1=w_2=\tanh(\alpha)\,g_\theta(u,h),\quad
\alpha_0=\operatorname{atanh}(0.25),\quad g_{\theta_0}=1.
\]

Therefore its frozen weights are **0.25 each, totaling 0.5**. Normalized Gauss-2 weights are **0.5 each, totaling one**. Improvement over this initialization is real, but does not isolate learning beyond a correctly normalized same-cost analytic rule. The saved JSON does not expose learned parameter values, so the review does not claim that training merely doubled the amplitude.

There is a second confound: the existing analytic quadratic control uses **four normalized nodes and no output cutoff**, whereas rank1 uses two learned nodes and a four-mode output cutoff. The analytic quadratic's paired RMS is already **2.81× lower in discrete and 2.56× lower in continuum**. At identical schedules it is only about 7%/13% slower. The cubic control is still more accurate, at additional work.

Our next attribution test should cross these factors explicitly: half versus normalized two-node weights; two versus four nodes; matched cutoff versus no cutoff; scalar-amplitude-only fit versus the full conditioner. Until those controls run, a unique advantage from learned conditioning is unresolved.

## Cost: distinguish the declared frontier from the best possible frontier

At the preregistered policy cell—32², final time 0.12, target 2e-5, 32-field model and fixed seed 3600011—rank1 is feasible on all 24 fields in each track. The strongest accuracy-qualified classical endpoint frontier has these geometric cost ratios:

| Track | Classical / rank1 cost | Field-cluster 95% interval | Rank1 slowdown |
|---|---:|---:|---:|
| Discrete | 0.1208 | [0.0979, 0.1484] | 8.28× |
| Continuum | 0.1461 | [0.1087, 0.1959] | 6.84× |

The analytic quadratic alone is **2.54×/2.25× faster** on its declared feasible endpoint frontier. It usually qualifies in one step; rank1's measured frontier selects a three-step primary schedule. Hybrid FNOs are generally faster on mutually feasible fields, but qualify on fewer fields: small FNO 18/24 discrete and 19/24 continuum; standard FNO 19/24 and 20/24. Their failures must remain in coverage, not disappear from a speed-only ranking.

**The schedule inventory is asymmetric.** Classical controls received extra uniform one-/two-/four-/eight-step endpoint options; neural families received the primary schedules. Neural one-/two-step alternatives at this full horizon were not measured. Same-output comparisons remain valid observations, but are not an exhaustive rank1 work–precision frontier. Every family should receive the same endpoint step grid next, with out-of-training step sizes labeled explicitly and intermediate-state claims kept separate.

The measured shortfall is large enough that adding a policy estimator now would be premature. G5 is blocked by cost, not absent development headroom. No calibration, operational coverage, false-acceptance probability, reliability guarantee or amortization benefit was measured.

## Regimes and scale

At primary tolerance, development headroom appears in **56/144 conditions on 7/12 independent fields**: high-pair 22/24, rough 17/24, near-Nyquist 6/12, reaction-stiff 6/12 and mixed 5/24. Low-frequency, boundary-mean and diffusion-stiff conditions have none. Headroom means one bare diffusion-first step fails; ETDRK4 can still solve several of these cells in one cheap step.

FD versus refined Galerkin RMS disagreement exceeds tolerance in **34/72 spatial cells**. This is a difference between target spatial equations. A temporal correction cannot remove it; discrete and continuum-track results must remain separate.

Scaling evaluates four distinct fields, one frozen seed, 64²/128²/256², batches one/four and T=0.06 with two 0.03 steps. All **60 workload/model rows pass accuracy**. Only one comparison clears 1.2×:

- Discrete 256², batch four: rank1 **3.947 ms**, standard FNO **6.021 ms**, a **1.525×** advantage.
- At that same workload, small FNO takes **2.778 ms**, DF **0.680 ms**, and ETDRK4 **0.917 ms**. Rank1 is not the best measured option.
- Rank1 loses all 12 small-FNO and all 24 classical comparisons. The one crossover is a useful scaling clue, not a general FNO victory or long-rollout result.

Rank1's nearly flat latency across these sizes motivates profiling. Static source inspection finds repeated diffusion-symbol construction and tensor-to-Python condition checks in its generic defect path, while classical paths have coefficient caches. CPU synchronization, transform reuse and dispatch overhead are hypotheses to measure, not established explanations. Optimize controls equivalently and require output parity before attributing a speedup to architecture.

## Baseline and training limitations

Hybrid FNOs pass the modest local adequacy gate: every largest-subset seed improves validation at least 5%, and some confirmation endpoints meet the primary tolerance. This does not establish convergence or reproduce the FNO paper. The direct FNO's zero accurate primary endpoints makes its enormous error ratio unsuitable as a literature-level victory claim.

Final training is 200 updates at batch four, with two learning rates tested for 40 tuning updates each. All rank1 and compression-ablation trials choose the last update; convergence is unresolved. Increasing training fields from eight to 32 at fixed updates does not improve rank1 consistently, so these observations are not a sample-complexity law.

FNO's preclip gradient norms exceed the common threshold on every update, with very different scales from rank1. This motivates validation-only loss/optimizer conditioning controls. It does not prove clipping caused the gap: Adam is approximately scale invariant. Fairer optimization and convergence checks must retain the same data splits and include measured training cost.

## Bounded next decision

| Experiment | Question | Essential controls and stopping condition |
|---|---|---|
| Normalized quadrature attribution | Does learning add anything beyond known quadrature at comparable cost? | Cross two/four nodes, half/full normalization and cutoff; compare fixed, scalar-only and fully conditioned variants. Retain learned benefit only if it survives same-cost analytic controls on new held-out fields. |
| Matched endpoint step grids | Was a feasible cheap neural schedule omitted? | Give every family 1/2/4/8 steps at identical final times and tolerances; preserve all infeasible and extrapolative cases. Keep endpoint and intermediate-output claims separate. |
| Frozen-model implementation audit | Is current latency dominated by avoidable implementation work? | Profile kernels, FFTs, synchronization and allocation; cache invariant coefficients, reuse spectra and batch node transforms. Require FP32 parity, physical-null/phase checks and equal baseline optimization before paired retiming. |
| Stronger local FNO optimization | Does the accuracy advantage survive a credible converged control? | Tune loss scaling, clipping and learning rate on validation only; report convergence curves and equal-data plus measured-compute frontiers. Do not retune on these 24 confirmation fields. |

Do these before another broad architecture or training sweep. The current full cohort is now inspected evidence; new confirmatory claims require new held-out fields. Preserve easy controls while enriching high-pair/rough/near-Nyquist cases, and require demonstrated matched-cost savings before re-enabling deployment.

## Resource accounting

The ten recovery allocations total **84m41s**, including **82m17s of GPU allocation**. The six parts contribute 70m37s of allocation and 62m1.4s of numerical work. Verification, readiness tests, aggregation and reporting are real research cost, distinct from model inference latency. Maximum current task RSS is about 3.06 GiB; this failure/recovery was not a host-memory problem.

The original completed prerequisites cost 25m42s, and the failed confirmation cost 30m27s. Known terminal GPU allocation across both attempts totals **2h0m3s**. Original report job 455 still has a running accounting snapshot in the archive, so a fully archived terminal total for both workflows is unavailable. Current report job 465's final 144 seconds is present in the later collection snapshot. Allocation and task-step records are not summed together, and dollars are unavailable without a supplied rate.
