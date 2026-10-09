# TDN model and comparison review

Reviewed **2026-10-08, America/Los_Angeles**. Scope: the retained project reports from initial CPU development through CARC research, Fedora premix/consistency/agenda/roadmap and the complete native frontier recovery. This is a synthesis of recorded results, not new training, numerical simulation, a new literature search or an independent replication of published papers.

**Expert assessment: TDN has evolved into a promising compact, physics-informed interaction correction. It has convincing bounded evidence for accuracy gains over several local neural controls, especially on high-frequency interaction cases. It does not yet demonstrate a generally better practical solver, an advantage attributable uniquely to learned conditioning over equally equipped analytic quadrature, or superiority to the published FNO method.**

The strongest next question is: **does a small learned, phase-aware correction improve accuracy–cost over a correctly normalized analytic correction with the same nodes, spectral support, physical core and implementation quality?** The current evidence motivates that experiment; it does not answer it.

## How to read this review

“Ours” denotes a project proposal or its ablation. “Theirs” denotes an established architecture or classical method implemented locally. Analytic attribution controls are identified separately: they are neither trained models nor a reproduced external paper. Established ingredients inside our models remain established ingredients.

A family, a configuration, a trained seed and a checkpoint are different objects. The frontier's **80 final records are ten families**, not 80 novel architectures. Names also recur across studies with changed implementations; roadmap `rank1` and frontier `rank1`, or early `fno_split` and later `fno_standard`, must not be treated as identical checkpoints or configurations. Parameter counts below belong to their stated study.

Comparisons have four separate meanings:

1. **Representation/structural diagnostic:** can a mechanism express the correct interaction or satisfy a physical identity? Oracle fits are not learned generalization.
2. **Same-schedule accuracy:** which method has lower error with identical requested steps and outputs? This does not establish equal-cost performance.
3. **Accuracy-qualified cost:** how fast is each method on schedules that actually meet the same target? Coverage and failures must remain visible alongside conditional speed.
4. **Deployment:** can a frozen policy choose its work without reference truth and win after estimation, rejection, fallback and setup costs? Post-hoc reference-informed frontiers are not deployment policies.

Unless explicitly stated otherwise, error ratios in the frontier discussion are **comparator error / candidate error** and speed ratios are **comparator time / candidate time**; greater than one favors the candidate. Historical ledgers retain their own explicit conventions. Do not compare numerical losses across studies with different normalization, tolerances, horizons, grids, hardware or spatial targets. BAD/NA mechanism scores measure declared check attainment, not model-quality percentages or theorem proofs.

## The architectural progression and its mathematical meaning

The core problem is periodic scalar logistic reaction–diffusion,

\[
\partial_tu=Au+B(u),\qquad A=\kappa\Delta,\quad B(u)=r u(1-u).
\]

Initial TDN used a physical split plus a learned temporal defect,

\[
\widehat\Phi_h(u)=S_h(u)+\sum_j a_j(\mathcal F(u))\,\psi_j(h;\rho_j(\mathcal F(u))).
\]

The encoder did not receive the queried step size; an analytic decoder supplied the time dependence and cubic small-step behavior. That is a real architectural restriction relative to an ordinary time-conditioned MLP, but a restriction is valuable only when it improves accuracy, cost or generalization. The early tests did not establish such a general advantage.

The later program increasingly models **nonlinear spatial interactions before compression**. For projection onto retained Fourier modes,

\[
P_K(u^2)\ne P_K[(P_Ku)^2],\qquad
\widehat{u^2}_k=\sum_{p+q=k}\widehat u_p\widehat u_q.
\]

Two discarded high-frequency modes can contribute to a retained low-frequency mode. Their phases and individual diffusion rates matter. This is stronger motivation for a physical pair response than a time basis indexed only by output frequency. It is not proof that every FNO loses those interactions: our FNO implementations retain a full-grid local path and nonlinearities.

The final compact branch learns a symmetric physical node pair, its amplitude and a small conditioner, with **43 real parameters**. It forms signed physical products before output projection. Its declared “rank one” is a physical node-pair convention, not matrix rank one or a Tucker-rank claim. Its correction vanishes for constant states, zero reaction, zero diffusion and zero time. Those identities are useful correctness constraints; they do not guarantee positivity, nonlinear stability or finite-step accuracy.

Reaction-first and diffusion-first Strang composition are established controls:

\[
S_h^{RF}=R_{h/2}D_hR_{h/2},\qquad
S_h^{DF}=D_{h/2}R_hD_{h/2}.
\]

Their different errors can dominate a learned correction's apparent gain. Likewise a same-grid finite-difference target and a dealiased spectral continuum estimate are different equations at finite resolution. Reducing time-integration error cannot by itself remove their spatial discrepancy.

## Study-by-study inventory and comparisons

The sections below enumerate the distinct implemented arms and preserve unsuccessful and initialization-selected outcomes. Optional or structural-only branches are labeled explicitly. Seeds, grids and tuning choices are configurations of these arms, not additional architecture inventions.

### 1. Original temporal TDN and local learned controls

All six original families actually ran the same **12-update CPU smoke**, with only **two independent diagnostic parents** and one timing repeat per parent/sequence. They were feasible at the smoke's tolerance. This is implementation evidence, not a credible architecture leaderboard.

| Family | Ownership/role and mechanism | Parameters in smoke | Worst weighted error | Median solve time |
|---|---|---:|---:|---:|
| `temporal_mlp` | Ours: learned amplitudes/rates with horizon-independent encoder and analytic time decoder | 1,736 | 7.573e-4 | 1.863 ms |
| `fixed_rate` | Ours ablation: fixed temporal-rate dictionary, learned amplitudes | 1,604 | 7.567e-4 | 1.779 ms |
| `taylor` | Established-inspired local control: learned cubic coefficient | 1,505 | 3.607e-4 | 1.507 ms |
| `polynomial` | Local control: cubic prefactor times learned polynomial in time | 1,604 | 6.807e-4 | 1.618 ms |
| `rational` | Local control: learned rational function with positive denominator | 1,637 | 6.168e-4 | 1.652 ms |
| `generic_mlp` | Theirs/local: ordinary horizon-conditioned MLP with cubic prefactor | 1,537 | 6.136e-4 | 1.516 ms |

The smoke did not favor elaborate temporal decoding. Initial development gates failed numerical headroom and temporal advantage. A later six-case light screen found five temporal-oracle advantages but **zero numerical-headroom cases**. A teacher-informed oracle fit cannot establish a deployable learned advantage. Optional KAN branches were rejected/unimplemented in this starting pipeline, not tested models.

Classical comparisons included physical Strang (`split`/`strang`, reaction-first in these early studies), `richardson_split`, `adaptive_split` and `coupled_rk4`. The initial learned solver was slower than the simple split on the tiny CPU diagnostic. None of this is native A100 or full-confirmation evidence.

### 2. Confluent, transported and reaction-coordinate research models

The next development program used eight TDN-related families and then seven local neural comparators. The three main proposals were confluent temporal responses, transport of an early correction plus a late bypass, and a bounded reaction clock. The fixed dictionaries and additive coordinate ablation were essential controls.

| Family | Ownership/role | Mechanism and observed status |
|---|---|---|
| `confluent_decay` | Ours; 277 parameters | Repeated-rate exponential moments with equilibrium decay; trained, without a sustained advantage over simpler dictionaries |
| `fixed_decay_r` | Ours control; 276 | Ordinary fixed rates with the same reaction decay envelope |
| `fixed_decay` | Ours control; 276 | Fixed rates with twice the reaction decay |
| `fixed_undamped` | Ours control; 276 | Fixed rates without equilibrium damping; often competitive with more elaborate temporal choices |
| `reaction_clock` | Ours; 242 | Signed cubic/quartic perturbation inside final logistic reaction; strongest early small neural lead |
| `reaction_additive` | Ours ablation; 242 | Same time basis/amplitudes as an additive state correction; weaker coverage and retained inadmissibility |
| `transport` | Ours; 295 | Discrete commutator-anchored early correction transported through diffusion, plus late bypass; failed training/validation, no eligible learned checkpoint |
| `temporal_mlp` | Ours original control | Original temporal decoder; initialization selected in the first native eight-family run |
| `reaction_polynomial` | Ours optional | Unsaturated polynomial clock; implemented/diagnosed, not a default trained native comparison arm |
| `reaction_hybrid` | Ours optional | Clock/capacity boundary blend; implemented/diagnosed, not a demonstrated trained winner |
| `generic_mlp` | Theirs/local; 1,537 | Shared physical split plus a time-conditioned local MLP |
| `residual_cnn`, `residual_cnn_split` | Theirs/local; 14,161 each | Direct stepper versus split-correction version of residual CNN |
| `unet`, `unet_split` | Theirs/local; 126,913 each | Direct versus split-correction U-Net |
| `fno`, `fno_split` | Theirs/local; 67,137 each | Direct versus split-correction FNO, with local paths retained |

The first native eight-family run allowed 96 updates and achieved **zero speed wins in 49 eligible comparisons against the best classical control**. The separate cloud development record is not mislabeled as that native run. The analytic `e3_anchor` joined the classical controls; it is an untrained cubic commutator correction and retained invalid trajectories.

The subsequent native A100 benchmark reused bounded CPU-selected checkpoints from a 96-update screen with nine training, nine validation and nine diagnostic parents. Parameter counts were unequal and convergence was not established. It has **56 declared TDN-to-neural pairings**. All are preserved in the [early comparison ledger](../results/model-comparison-review/early-neural-comparisons.json). The direct CNN/U-Net/FNO controls had no feasible diagnostic rollout, so matched-tolerance speed comparisons against them are unavailable, not automatic TDN wins.

For the reaction clock, conditional A100 median speedups against generic MLP, split-CNN, split-U-Net and split-FNO were **1.050×, 1.125×, 3.014× and 1.853×**. Eligible-parent speed wins were 5/7, 5/8, 6/8 and 5/7. Yet the clock beat the best classical method on **0/9 parents**. It had some feasible step on all nine parents but passed only 20/36 rollout horizons and 7/18 one-step off-grid queries. “Some accurate endpoint exists” is weaker than transient accuracy.

Original temporal MLP versus generic MLP lost all seven eligible speed comparisons and 26/36 same-step rollout-error comparisons, while winning 13/18 one-step and 10/18 two-step off-grid error comparisons. The result is a tradeoff, not superiority. Two isolated ≥20% speed wins over the best classical choice—one split-CNN case and one fixed-undamped case—did not establish broad utility.

Three-seed replication retained **the same 27 diagnostic parents**, not 81 independent parents:

| Family | Parents with some feasible rollout, by seed /27 | Parents passing joint off-grid one/two-step checks, by seed /27 |
|---|---|---|
| Reaction clock | 27 / 27 / 27 | 6 / 6 / 5 |
| Generic MLP | 21 / 21 / 21 | 6 / 6 / 6 |
| Residual CNN + split | 24 / 27 / 21 | 5 / 0 / 6 |
| U-Net + split | 24 / 27 / 15 | 3 / 3 / 0 |
| FNO + split | 21 / 21 / 21 | 9 / 10 / 9 |

The clock had broader endpoint coverage; **FNO had better joint transient coverage**. Generic MLP selected initialization in seed74011 and update8 in the other two seeds, so the baseline was not established as converged. Clock beat the best classical choice on 0/27, 0/27 and 1/27 parents by seed; the isolated positive case remains part of the record. Clock lost all 36 mixed-frequency rollout-error comparisons against MLP in each seed, and no learned model jointly passed boundary parents. Native CPU conditional speedups do not transfer numerically to A100; the later user-provided A100 log agrees with the transient pattern, but is not a separate archived, independently reviewed GPU replication dataset.

The coordinate mechanism has a specific limitation: near a zero receiver with diffused state O(h), an O(h³) clock change produces only an O(h⁴) state correction, while the desired defect can be O(h³). More optimization cannot remove that architectural mismatch.

### 3. Analytic interaction, work–precision and compact-spatial controls

These experiments trained **no neural models**. They tested the numerical mechanisms that motivated later neural architectures.

| Program | Complete control/variant inventory | Main conclusion |
|---|---|---|
| Mechanism audit | Scalar temporal, transported temporal, output-frequency phi and pair-aware oracle fits; local/global coordinate gates; capacity/boundary controls; phase, aliasing, factorization, composition and wrong-generator probes | Pair-aware oracle best in 21/24 response cases; it uses privileged generating information and is not deployable learned evidence |
| Interaction | `strang`, `etdrk2`, `etdrk4`; `scalar_cubic`, `output_phi`; `gl3_additive`, `gl3_capacity`, `gl3_projected`, `gl5_additive`, `gl5_capacity`, `gl5_projected`, `gl3_zero_mean`, `gl3_mean_only` | GL3/GL5 analytic quadratic responses improve accuracy strongly but add substantial work |
| Prepared work–precision | `strang`, `etdrk2`, `etdrk4`, `gl3`, `gl5`, `gl3_mean_full`, `gl3_mean_spectral` | Strang dominates loose targets, ETDRK4 tight targets; extra quadrature and mean-only shortcuts rarely repay their cost |
| Compact spatial | `strang`, `etdrk4`, `gl3`, `gl5`, `gl3_mean_spectral`, `gl3_fused`, `compact_gl3_m2`, `compact_gl3_m4`, `compact_gl3_m8`, `compact_gl3_m4_no_mean` | Fusion useful; input compression destroys retained interactions and loses all common-feasible accuracy-bank cost comparisons |
| Numerical premix extensions | Output-projected GL3 K4/K8, input-truncated controls, chunked/fused GL3 and direct selected-output convolution | Preserve nonlinear products before projecting outputs; implementation speedups do not alone establish practical solver utility |

GL3/GL5 denote our untrained interaction correction evaluated using established Gauss–Legendre quadrature. They are not learned FNO competitors or merely bare quadrature integrators. The prepared/fused versions are implementation variants of related mathematics, not independent neural inventions.

On the 32-case interaction screen, GL3 additive reduced error versus Strang in 32/32 cases, with median error ratio **0.02742**; it beat ETDRK4 accuracy in 23/32. But it cost **5.846× Strang and 1.789× ETDRK4** per step. GL5 cost another 1.236× for almost the same median error. All Strang RMS endpoints already passed the loose 0.002 target. Mean-only and zero-mean GL3 had median error/Strang of 0.27664 and 0.96346: neither reproduced the full interaction gain.

Across 184 accepted prepared work–precision endpoint/norm/target cells, warmed fastest-method counts were **Strang 102, ETDRK4 73, GL3 eight, GL5 one**. Including setup changed them to 135/40/8/1. ETDRK2 and both mean-only variants won none. One reference remained rejected; its entries were excluded explicitly, not counted as passes. The spectral mean identity was correct to approximately 1e-15, but it lost all 174 common-feasible cost comparisons to Strang. Quadratic-amplitude fidelity did not yield fourth-order time integration: observed late time order was about 1.88 for GL3 versus 3.98 for ETDRK4 on resolved subsets.

The expanded compact bank found Strang/ETDRK4/fused-GL3 fastest in **177/130/13 of 320 warmed accuracy cells**; Strang won all 144 scaling cells. Fusion gave a 1.587× median same-step gain with approximately 3e-15 maximum FP64 disagreement, yet was slower in one large batch regime. K2/K4/K8 input truncation lost all 314/316/316 mutually feasible comparisons to fused GL3 and cost roughly 1.89× as much; spatial-only K4 cost 2.143× and missed 13 targets. Modes 9 and 10 producing mode 1 illustrate why retaining only low inputs is not equivalent to retaining low outputs.

Output projection in the later numerical premix study improved four-step RMS over K4 input truncation by median **107× near Nyquist**, 1.53× for cutoff pairs and 1.57× for broadband fields. Nevertheless it remained about 4.11× slower than the best feasible classical choice in accuracy tests. Direct selected convolution was 2.03×/10.06× slower than FFT for the declared 1D/2D cases. These are useful representation and implementation findings, not a neural win.

### 4. Native Fedora premix: five families

Premix multiplies full-state latent features before spectral output selection; precompression projects inputs first; Premix-local adds a local high-frequency bypass. The 15 trials used three paired seeds and 300 updates each. Fourteen selected trained checkpoints; one Premix-local selection retained initialization.

Each numerator below is out of **48 diagnostic parents**; seed entries refer to repeated evaluation of the same bank. RMS and maximum are separate criteria here.

| Family | Role | Parameters | RMS ≤2e-4 by seed | Maximum ≤2e-4 by seed |
|---|---|---:|---|---|
| `premix` | Ours | 23,825 | 16 / 12 / 12 | 12 / 4 / 8 |
| `premix_local` | Ours + local bypass | 23,841 | 16 / 36* / 16 | 12 / 36* / 8 |
| `precompress` | Ours input-compression ablation | 23,825 | 12 / 20 / 36 | 12 / 12 / 20 |
| `fno` | Theirs, adapted hybrid | 67,137 | 32 / 24 / 16 | 28 / 20 / 12 |
| `cnn` | Theirs, adapted hybrid | 14,161 | 16 / 24 / 28 | 16 / 16 / 26 |
| Strang | Established classical | 0 | 36 | 36 |
| ETDRK4 | Established classical | 0 | 48 | 48 |
| Fused GL3 | Analytic control | 0 | 48 | 48 |

*Initialization, hence physical-base behavior rather than a learned bypass success. At RMS 2e-6 every trained neural selection has zero coverage; Strang/ETDRK4/GL3 cover 12/32/44 of 48.

The six declared matchups compare Premix and Premix-local separately with FNO, precompression and CNN. Premix/FNO conditional speed ratios were **0.865/0.847/0.860**, so Premix was slower as well as less broadly accurate. Premix was approximately 1.10–1.13× faster than its precompression ablation. Premix-local/FNO was 1.482×, unavailable for the initialized seed, then 0.758×. This does not establish a consistent backbone advantage.

Premix improved rough-field same-step RMS in 36/36 comparisons, but learned mean drift dominated many smooth/boundary/long-time failures. A cubic time factor and bounded output were insufficient to protect near-exact cases.

### 5. Consistency and mean controls: seven arms

The next experiment retained raw `premix`, `precompress`, `fno` and added `premix_gated`, `fno_gated`, `premix_moment`, `fno_moment`. Physical gating enforces commuting/constant limits; moment versions add a learned mean target and bounded redistribution. These are ablations, not seven unrelated architecture discoveries.

All 12 constrained family/seed trials selected trained checkpoints. Of 21 total trials, 14 trained selections and seven initialization selections remained; raw Premix and precompression selected initialization in every seed.

| Family | RMS ≤2e-4 coverage /48, by seed | RMS ≤2e-6 coverage /48 |
|---|---|---|
| Premix | 36 / 36 / 36, all initialization | 12 / 12 / 12 |
| Precompression | 36 / 36 / 36, all initialization | 12 / 12 / 12 |
| FNO | 36 / 36 / 36, one initialization | 12 / 12 / 12 |
| Gated Premix | 40 / 36 / 40 | 17 / 15 / 16 |
| Gated FNO | 40 / 40 / 40 | 16 / 16 / 16 |
| Moment Premix | 40 / 40 / 40 | 22 / 28 / 12 |
| Moment FNO | 40 / 40 / 40 | 24 / 28 / 12 |

All neural arms still have **36/48 maximum-error coverage** at 2e-4. Four extra RMS passes do not establish maximum-error improvement. Moment Premix and Moment FNO had identical primary RMS coverage, with Premix **1.511–1.522× faster** conditionally. Gated Premix was 1.610–1.638× faster than gated FNO but had worse coverage in one seed. Moment treatment itself added roughly 22% same-step cost to Premix and 13.5% to FNO.

Moment Premix/FNO had 23,842/67,154 parameters, different depths and Fourier supports. The result supports a cheaper compact implementation under this design, not matched-capacity superiority. Diagnostic step sizes had already appeared in development, and the grid-transfer cases changed fields as well as grid. Ten declared pairings, including raw→gated→moment and Premix→precompression, are retained in the comparison annex.

### 6. Full research agenda: source placement, time, closure and rank

All **17 family IDs** below participated somewhere in 69 trials; only a subset entered confirmation. There were 64 trained and five initialized selections.

| Families | Role / mathematical change | Evidence scope |
|---|---|---|
| `local_gate`, `local_gate_time`, `local_endpoint` | Ours controls: local receiver gate, optional time decoder or endpoint-mean treatment | Minimal-control training; local gating can suppress needed remote response |
| `source`, `source_time`, `source_endpoint` | Ours: physical commutator-source features before zero-preserving nonlocal transport; time/mean variants | Source confirmed in three seeds; source-time confirmation selected DF initialization |
| `source_closure`, `source_time_closure` | Ours: approximate dynamic mean–variance closure | Closure confirmation selected RF initialization; no learned closure benefit established |
| `precompress_source` | Ours compression ablation | Three-seed training and confirmation; physical bypass remained, limiting pure-bottleneck attribution |
| `fno_source`, `fno_source_depth4`, `fno_source_matched`, `fno_anchor` | Theirs/local: cheap one-layer, equal-width deep, parameter-matched deep and historical-mask FNO | All trained in compression study; matched deep FNO confirmed in three seeds |
| `pair_rank2`, `pair_rank4`, `pair_rank8`, `pair_rank4_closure` | Ours: signed phase-preserving physical pair kernels, optional closure | One-seed kernel study; only rank2 confirmed |

The biggest validation improvement was **classical DF orientation: 23.684× lower normalized objective than RF before learning**. In the compression study, Source (23,824 parameters) beat input precompression and parameter-matched deep FNO (23,480) in two of three validation seeds. However, **cheap one-layer FNO (23,792) had similar cost and beat Source validation loss in two of three seeds**. Omitting that control would materially exaggerate our result.

Kernel rank2/rank4/rank8/rank4+closure had 120/176/288/241 parameters and validation objectives 3.282e-5/3.665e-5/3.490e-5/3.760e-5. Rank2 was best under that one-seed budget; higher rank or closure did not help. Rank2 required nine correction transforms versus Source's two: parameter count did not represent inference work.

In discrete confirmation, Source covered **574/576** RMS cells at 2e-4, matched FNO **576/576**, and rank2 **192/192**. At maximum error 2e-5, counts were **477/576, 533/576 and 176/192**. Source/rank2 were about 1.74×/1.68× faster than the deep matched FNO, but the bare DF base met the primary target everywhere and was about **3.9× faster than Source/rank2**. No trained candidate passed the complete preregistered solver gate.

The apparent sample count requires care: eight controlled field variants shared **one underlying phase-seed cluster**. They were not 32 independent random fields. The reverse unseen step schedule produced joint passes of Source 193/288, matched FNO 213/288, rank2 86/96 and bare DF 90/96. Source's worst maximum error reached 0.04098 where its paired base had only 3.53e-5 error. Rank2 was substantially more robust, but still incomplete and represented by one seed.

Continuum coverage was nearly the same across methods, with roughly fourfold error decrease when resolution doubled—consistent with shared finite-difference bias. Policy calibration was target-specific: zero false accepts among 1,179 discrete accepted endpoints did **not** transfer to continuum; 306/410 accepted continuum endpoints failed. Source+step-doubling had a favorable median attributed cost against ETDRK4 but **6.168 s total versus 4.415 s**, about 40% more work. These are component-attributed costs against fixed refined controls, not independently timed deployments against the fastest classical frontier. Rejection tails matter.

### 7. M00–M23 roadmap: every neural family and combination

The roadmap ran **64 tuning trials plus 51 final trials**, 22,980 updates, with 42 trained and nine initialized final selections. Seventeen neural families plus five untrained endpoint controls yielded 56 evaluated instances. There were 24 independent discrete field clusters and 12 continuum clusters; the 40,320 endpoints were repeated observations over them.

The following table uses **corrected identical final time 0.27 and joint RMS/maximum tolerance 2e-5**. D/C denominators are 288/144 repeated model-seed/field/grid cells. The last column is strongest-control time / candidate time on mutually feasible discrete cells; below one means the candidate is slower. It is separate from coverage.

| Family | Ownership / change | Parameters | Trained /3 | D /288 | C /144 | Classical / candidate time |
|---|---|---:|---:|---:|---:|---:|
| `source` | Ours, source before compression + DF | 12,996 | 3 | 192 | 34 | .322 |
| `rank1` | Ours, one symmetric physical pair | 43 | 3 | 288 | 48 | .217 |
| `rank2` | Ours, two physical pairs | 54 | 3 | 288 | 48 | .163 |
| `source_postcompression` | Ours input-compression ablation | 12,996 | 3 | 192 | 36 | .286 |
| `source_trust` | Ours stiffness/time trust envelope | 12,996 | 3 | 192 | 34 | .250 |
| `source_consistency` | Ours temporal/generator supervision | 12,996 | 3 | 192 | 34 | .322 |
| `source_df_loss` | Ours regime/peak/base-aware loss | 12,996 | 3 | 192 | 34 | .319 |
| `source_df_loss_shared` | Same loss, shared optimizer choice | 12,996 | 3 | 192 | 34 | .321 |
| `source_multiband` | Ours fractional/multiband features | 13,021 | 2 | 192 | 33 | .231 |
| `cheap_fno` | Theirs/local one-layer source hybrid | 13,140 | **0** | 192 | 33 | .314 |
| `deep_fno` | Theirs/local deeper matched control | 13,122 | 3 | 192 | 36 | .169 |
| `direct_fno` | Theirs/local neural-only evolution | 52,489 | 3 | 0 | 0 | NA |
| `c1_rank0` | Ours analytic quadratic + source | 12,996 | 3 | 288 | 36 | .256 |
| `c1_rank1` | C1 + learned rank1 | 13,039 | 2 | 288 | 34 | .153 |
| `c1_rank2` | C1 + learned rank2 | 13,050 | 2 | 288 | 34 | .129 |
| `c2_rank2` | C1 rank2 + trust/consistency | 13,050 | 3 | 288 | 36 | .101 |
| `source_embedded` | Ours learned residual on classical subdivided RK4 | 12,996 | **0** | 288 | 30 | .085 |

The five additional endpoint controls were `df_base`, `rf_base`, `etdrk4`, `gl3_fused` and `df_quad2`. The last is an analytic quadratic DF correction, not a trained model. The combined classical set covered 288/288 discrete and 42/144 repeated continuum cells at this target.

Rank1 reduced discrete same-schedule RMS versus Cheap FNO in **1,395/1,440 pairs**, median 2.344×, while costing about 1.82× more. Rank2 cost another 1.35× without a convincing additional benefit. Source versus postcompression showed only a 1.00215× median discrete RMS ratio and approximately 1.0 continuum ratio. Thus the roadmap established the mechanism more strongly than a broad learned advantage.

C1 ranks 0/1/2 had median RMS ratio **1.0 versus untrained `df_quad2`** while costing 1.39×/2.34×/2.79× as much. Embedded's strong accuracy belonged to its **initialization-selected RK4 backbone**. Trust, consistency, loss and multiband arms did not materially improve coverage. Shared and independent DF-loss tuning selected the same settings, leaving tuning attribution unresolved.

The historical matched-frontier report mixed endpoints at 0.27 and 0.81; **1,412/22,032 original rows had a horizon mismatch**. The corrected review retained 44,064 separate-horizon comparisons. Only one of 28,275 jointly feasible neural/classical comparisons favored the neural method on latency, by 0.7%, an isolated margin rather than a practical speed claim.

All M00–M23 and C0–C4 IDs executed. They are research mechanisms and evidence checks, not 29 additional neural architectures:

| ID | Implemented mechanism / combination | Recorded verdict | Check score /100 |
|---|---|---|---:|
| M00 | Headroom map and fused classical reference | BAD | 61 |
| M01 | Consistent spatial target and two-resolution discrepancy | BAD | 68 |
| M02 | Quadrature approximation to the finite-step quadratic DF coefficient | BAD | 61 |
| M03 | Physical factors plus learned rank-1/rank-2 remainder | BAD | 60 |
| M04 | Signed nonlinear sources before output compression | BAD | 60 |
| M05 | Product-consistent anti-aliasing and grid-scaled filters | BAD | 82 |
| M06 | Stiffness-aware trust envelope for the correction | BAD | 59 |
| M07 | Conditional solver bank with a classical safe path | BAD | 68 |
| M08 | Step-doubling and stability-weighted residual estimators | BAD | 68 |
| M09 | Function-level calibrated total-error and selective risk | BAD | 68 |
| M10 | Unequal-step generator consistency and intermediate supervision | BAD | 61 |
| M11 | DF-tuned defect loss with regime and maximum-error control | BAD | 60 |
| M12 | Dynamic feasible mean–variance closure | NA | 93 |
| M13 | Dimensionally consistent multi-band/fractional features | BAD | 59 |
| M14 | Geometry-aware and anisotropic transport | BAD | 82 |
| M15 | Cost-aware algebraic reuse, fusion and batching | BAD | 67 |
| M16 | Local FNO controls and strongest classical comparison | BAD | 44 |
| M17 | Factor-isolated stress and coefficient-transfer bank | BAD | 84 |
| M18 | Fresh independent fields and hierarchical inference | BAD | 69 |
| M19 | PDE and boundary portability ladder | BAD | 82 |
| M20 | RK4 backbone with order-constrained learned residual | BAD | 65 |
| M21 | Physical monotonicity and growth-envelope diagnostics | BAD | 81 |
| M22 | Effective configuration, provenance and reproducible measurement | NA | 93 |
| M23 | Dedicated cubic Volterra remainder | BAD | 87 |
| C0 | Prerequisite screen | NA | 74 |
| C1 | Primary compact finite-step residual | BAD | 61 |
| C2 | Operational error/cost-controlled hybrid | BAD | 62 |
| C3 | Finite-amplitude side branch | BAD | 94 |
| C4 | Later transfer/geometry branch | BAD | 82 |

The most revealing combination failure was C3. Quadratic+cubic reduced RMS versus DF by about **3,742×/428×/103×** across three amplitudes. Adding approximate mean replacement reduced these gains to **1.59×/1.68×/1.92×**, at roughly double its already high cost. Cubic-only produced little benefit. A component being mathematically motivated does not imply its combination is beneficial.

C4's unfiltered Richardson correction improved Gray–Scott/Burgers numerical errors, but fixed filtering spoiled much of the cancellation. Guards never activated, and the historical reference “refinement” kept the same 196 RHS evaluations, so near-roundoff gains were not adequately independently resolved. These were numerical transfer tests, not trained TDN cross-PDE generalization.

Roadmap policy had 84 false accepts, all continuum cases without spatial checks. Spatial checks removed these observed failures but cost enough that the cheapest accurate variant was still 1.192× classical total cost; Source was 1.202× and C2 4.162×. Estimators consumed **91.8%** of deployment time. The reported 49,015-query break-even was a cold-first-call confound in a fallback-only policy, not neural amortization.

Large-grid roadmap rank1 throughput crossed over against Cheap FNO at 512²/batch16 and 1024²/batch16, but those rows had **accuracy unavailable** and repeated one engineering field. They motivated the later accuracy-backed frontier study; they were not solver victories.

### 8. Final frontier: ten families, 80 final model records

The latest native recovery provides the strongest current comparison: 24 independent confirmation fields, both discrete/continuum target tracks, 32²/64² confirmation grids, three paired training seeds, 8/32 training-field subsets and 21,888 endpoints. There are **five trainable families × two tracks × two training subsets × three seeds = 60 trained records**, plus 12 frozen records and eight analytic/classical records. All 60 trainable final records selected trained checkpoints; 20 separate tuning trials give a 100-record catalogue.

| Family | Role | Parameters | Final records |
|---|---|---:|---:|
| `rank1` | Ours: DF core + conditioned symmetric quadratic pair, output compression | 43 | 12 |
| `rank1_postcompression` | Ours ablation: compress correction input first | 43 | 12 |
| `rank1_frozen` | Ours untrained initialization control | 43 stored, zero trainable | 12 |
| `fno_small` | Theirs/local hybrid: width16, depth2, modes4 | 47,057 | 12 |
| `fno_standard` | Theirs/local hybrid: width32, depth4, modes8 | 1,258,977 | 12 |
| `direct_fno` | Theirs/local neural-only residual evolution; same large FNO dimensions | 1,258,977 | 12 |
| `analytic_quad` | Analytic control: normalized four-node quadratic defect | 0 | 2 |
| `analytic_quad_cubic` | Analytic control: quadratic + cubic defect | 0 | 2 |
| `df` | Theirs: diffusion-first Strang | 0 | 2 |
| `etdrk4` | Theirs: exponential RK4 | 0 | 2 |

“Strongest classical” is a reference-informed envelope of DF, ETDRK4 and the two analytic controls, **not an eleventh architecture**.

At **32 training fields**, the five identical primary schedules give 720 neural observations per track, but only 24 independent fields. Classical/analytic controls have 240 unique observations; repeated seed pairing does not increase their independent sample count. Passes require both RMS and maximum error, including recorded reference uncertainty, ≤2e-5.

| Method | Discrete passes | Continuum passes | Comparator RMS / rank1 RMS, D / C | Comparator time / rank1 time, D / C |
|---|---:|---:|---|---|
| Rank1 | **668/720** | **668/720** | 1 / 1 | 1 / 1 |
| Frozen rank1 | 570/720 | 576/720 | 2.991 / 2.693 | .999 / 1.000 |
| Input-compression ablation | 549/720 | 558/720 | 2.299 / 2.143 | 1.063 / 1.013 |
| Small hybrid FNO | 313/720 | 372/720 | 24.804 / 15.095 | .644 / .735 |
| Standard hybrid FNO | 337/720 | 367/720 | 19.119 / 17.314 | .925 / .791 |
| Direct FNO | 0/720 | 0/720 | 50,590 / 44,382 | .709 / .142 |
| Analytic quadratic | 235/240 | 234/240 | .356 / .391 | 1.073 / 1.132 |
| Analytic quadratic+cubic | 239/240 | 238/240 | .223 / .285 | 3.696 / 3.621 |
| DF | 160/240 | 162/240 | 4.499 / 4.091 | .154 / .633 |
| ETDRK4 | 147/240 | 144/240 | 4.947 / 5.447 | .228 / .190 |

Ratios are field-cluster geometric means on **the same primary schedules**, not matched-tolerance deployment speedups. Rank1 is more accurate but slower than the local hybrid FNOs at these schedules. Analytic quadratic is **2.81×/2.56× more accurate** and only about 7%/13% slower; cubic adds still more accuracy and much more work. DF/ETDRK4 have lower same-step accuracy here, but their cheaper refined schedules often win at matched tolerance.

The learning gain over frozen rank1 is real, but attribution is incomplete: frozen pair weights are **0.25+0.25**, whereas normalized Gauss-2 weights are **0.5+0.5**. The analytic comparator additionally differs in node count and output cutoff. We cannot conclude that the conditioner beats a properly normalized same-cost analytic rule, or that training merely rescaled amplitude; saved JSON does not reveal the selected parameter values.

The most convincing precompression gains occur exactly where high-mode interactions matter:

| Regime | Input-compressed error / rank1 error, discrete | Continuum |
|---|---:|---:|
| Low frequency | .868 | .919 |
| Mixed | .798 | .755 |
| High pair | 6.729 | 8.534 |
| Rough | 9.206 | 6.956 |
| Near Nyquist | 12.305 | 10.648 |
| Boundary mean | 1.080 | 1.047 |
| Reaction stiff | .697 | .700 |
| Diffusion stiff | 1.962 | 1.383 |

There are only **three independent fields per regime**. Compression ablation is modestly better in low/mixed/reaction-stiff cases; no universal advantage follows.

At the preregistered 32², T=0.12, 32-field, fixed-seed policy cell, rank1 qualifies on all 24 fields in both tracks. On the declared accuracy-qualified endpoint menus:

| Comparator | Feasible D / C | Comparator time / rank1 time, D / C |
|---|---|---|
| Analytic quadratic | 24/24 / 24/24 | .394 / .444 |
| Analytic quadratic+cubic | 24/24 / 24/24 | 1.300 / 1.326 |
| Small FNO | 18/24 / 19/24 | .728 / .784 |
| Standard FNO | 19/24 / 20/24 | .941 / .805 |
| Strongest classical envelope | 24/24 / 24/24 | **.1208 / .1461** |

Rank1 is thus **8.28×/6.84× slower than the observed classical frontier** at this cell. These are valid results of the declared menu, but the classical menu included extra 1/2/4/8-step endpoints that were not measured for neural families. An exhaustive equal-menu neural work–precision conclusion remains open. Direct FNO is infeasible and cannot establish a speed victory by being omitted.

Accuracy-backed scaling tested four fields, one frozen seed, grids64²/128²/256², batch1/4 and two steps to T=0.06. All 60 workload/model rows pass accuracy. Rank1 loses all 12 small-FNO and all 24 classical comparisons. One of 48 comparisons exceeds 1.2×: at discrete256²/batch4, rank1 takes **3.947 ms** versus standard FNO **6.021 ms** (1.525×). At that same workload, small FNO takes **2.778 ms**, DF **0.680 ms**, ETDRK4 **0.917 ms**. This is a crossover against one comparator, not the fastest solver.

Training remained bounded: 200 final updates at batch4, with two 40-update learning-rate trials. Every rank1/ablation seed selected its final update; convergence is unresolved. Increasing training fields8→32 changed rank1 passes667→668 discrete and660→668 continuum, but did not consistently improve aggregate RMS. No sample-complexity law follows. Catalogue training time was416.15 s (331.38 s optimizer work); teacher generation, evaluation and allocations are additional costs.

The retained five-gate verdicts are **G1 GOOD, G2 BAD, G3 BAD, G4 BAD, G5 NA**. G5 was correctly blocked; no deployment calibration, risk guarantee or amortization benefit was measured. Known terminal GPU allocation across the original attempt and recovery was2h0m3s, including failure/recovery; this is not per-model inference time and has no dollar conversion without a rate.


## What the complete comparison record does and does not contain

This was a sequence of controlled studies, **not an exhaustive tournament among every model ever created**. A dedicated paired result is absent for some within-study pairs, even when common-condition raw observations exist; deriving such a comparison would be a separately labeled post-hoc analysis. No scientifically valid result exists for an unmeasured cross-study pair merely because both models have a number in a report. In particular, reaction-clock versus the final 43-parameter frontier model has not been measured on a common frozen cohort and budget.

The companion [comparison evidence directory](../results/model-comparison-review/README.md) exposes the retained pair tables and their source conventions. It complements the narrative, including unfavorable and infeasible cases. The roadmap's [44,064 corrected endpoint-frontier records](../results/roadmap-full-review/fixed-horizon-comparisons.json.gz) remain available separately. Repeated schedules, grids, seeds and physics variants are not independent evidence.

| Study | Comparison coverage | Missing inference |
|---|---|---|
| Initial controls | Six learned families share one tiny CPU smoke design; classical split/adaptive/Richardson/RK4 evaluated | No converged neural ranking or native GPU conclusion |
| Early native neural benchmark | Eight TDN-family controls crossed with seven generic/CNN/U-Net/FNO controls: 56 matchup aggregates, 504 parent comparisons | Not every neural-baseline-to-baseline pair |
| Three-seed replication | Reaction clock versus generic MLP, residual CNN, U-Net and FNO, with paired classical checks | No conclusion about all possible temporal or neural-operator architectures |
| Premix | Premix and Premix-local each versus precompression, FNO and CNN | No complete pairwise tournament |
| Consistency/agenda | Declared physical-gate, mean, source, temporal, rank, loss and FNO ablations; see ledgers | No universal model ranking across different cohorts or spatial targets |
| Roadmap | 420 stored directional family-pair/track rows: 370 measured and 50 explicitly unavailable; 165 distinct unordered pairs have evidence in at least one direction; corrected fixed-horizon frontiers separately | Not all 231 unordered pairs among 22 families/controls |
| Frontier | Rank1 versus nine individual controls in same-schedule attribution; six comparator categories in recorded work–precision frontiers; four comparators in scaling | No complete all-pairs matrix, and not every family received the same endpoint step inventory |

## Expert review: strongest findings

**1. The physical interaction hypothesis is supported more strongly than a generic claim about temporal neural networks.** Independent mechanism probes expose the information lost by premature projection, the insufficiency of output-frequency-only conditioning and the need to preserve signed phase products. The final learned precompression advantage is concentrated on high-pair, rough and near-Nyquist cases, which matches that mechanism. Some smooth and reaction-stiff cases favor the ablation, so this is a regime-dependent advantage.

**2. Rank1 is the best current learned research lead, not the best established solver.** It is small, trainable, physically structured and substantially more accurate than the tested hybrid FNOs at matched primary schedules. Yet analytic quadratic/cubic corrections are more accurate, and optimized classical choices often solve the declared endpoint more cheaply. Adding rank, networks or safeguards has repeatedly added more cost than useful accuracy.

**3. Several apparent gains belong to classical structure.** Diffusion-first orientation, exact subflows, analytic quadratic response and an initialization-selected RK4 backbone account for major improvements. These are valuable engineering choices. Crediting all of their gain to learned TDN would be incorrect.

**4. Engineering optimization is a demonstrated research contribution when its claim is stated narrowly.** Fusion and coefficient reuse produced verified same-output speedups. Parameter efficiency alone did not predict solver speed: a 43-parameter method can still lose to a much larger network because parameter count omits transforms, quadrature, memory traffic and launch/synchronization costs. Profiling is still needed to establish which costs dominate here.

**5. The reporting and provenance are unusually useful for diagnosing failure.** Preserved initialization selections, failures, raw endpoint measurements, source/teacher seals and costs made it possible to find horizon mismatches, target confusion and poor combination attribution. Passing the software suite supports implementation reliability; it does not turn negative scientific gates into positive findings.

## Shortcomings that materially limit the conclusion

| Issue | Evidence and consequence | Required repair |
|---|---|---|
| Unequal analytic attribution controls | Final frozen pair weights sum to 0.5, while normalized Gauss-2 sums to 1. Analytic comparator uses four nodes/no cutoff versus learned two nodes/four-mode cutoff. | Cross normalization, node count, cutoff, scalar-only fitting and full conditioner at comparable cost. |
| Incomplete neural endpoint frontier | Classical methods received extra 1/2/4/8-step endpoint choices; neural models received primary schedules. | Give all families the same final-time step grid; mark step-size extrapolation. |
| FNO optimization remains limited | Early Cheap FNO sometimes selected initialization; Direct FNO repeatedly fails tolerance; frontier FNO gradients clip every update. | Validation-only loss/optimizer conditioning and convergence checks; report equal-data and measured-compute budgets. Clipping frequency alone is not proof of causation. |
| No published-paper reproduction | Known RD physics, exact propagators and specialized corrections differ from a general operator-learning benchmark. | Reproduce an official benchmark and credible baseline training before a paper-level superiority claim. |
| Historical endpoint mismatch | Roadmap originally mixed final times 0.27 and 0.81; hundreds of apparent neural speed wins involved different targets. | Use corrected same-horizon tables here; later frontier repairs do not retroactively validate original numbers. |
| Spatial target mismatch | Agenda/roadmap temporal improvements often hit a common continuum grid-error floor. | Keep track-consistent operators/references, paired grid refinement and separate spatial uncertainty. |
| Weak easy-case headroom | Bare DF solves most original roadmap screening cells cheaply. | Retain easy controls but enrich genuinely costly high-pair/rough/near-Nyquist cases on fresh fields. |
| Too little independent evidence for broad risk claims | Final frontier has 24 independent fields, only three per stress regime; other studies have smaller effective cohorts. | Use fresh independently drawn fields and field-cluster inference; do not count repeated endpoints as independent samples. |
| Mean/closure combinations can destroy a strong solver | Approximate mean replacement severely damages quadratic+cubic accuracy. | Compare every combination with its best contained component, not only a weak base; audit mean and centered errors separately. |
| Invariance does not imply correct dynamics | Identity/wrong-generator controls can have tiny composition or estimator discrepancies while being wrong. | Independent PDE residual/accuracy checks and explicit estimator-bias assessment. |
| Safeguards can dominate runtime | Roadmap estimation takes 91.8% of policy time; rejections and fallbacks erase favorable medians. | Measure whole-policy total/tail cost with frozen decisions and the strongest accurate classical policy. |
| Historical cold-call artifact | A fallback-only policy appeared to amortize because of a slow first classical call. | Charge setup separately and use repeated randomized warm/cold comparisons. Later repairs must be used consistently. |
| Scaling evidence is narrow | Roadmap large-grid speedups had accuracy NA; frontier adds accuracy but covers four fields/one seed and short horizons. | Expand only after useful accuracy-qualified crossovers survive; no universal large-scale conclusion yet. |
| No settled convergence or sample-complexity result | Final rank1/ablation selects update 200; more training fields at fixed updates do not consistently improve results. | Separate optimization saturation from data exposure and architecture; avoid extrapolated scaling laws. |
| Transfer is mostly numerical, not learned | Gray–Scott/Burgers/boundary tests concern numerical corrections; historical teacher refinement was insufficient. | A new teacher-verified learned transfer study is needed before claiming cross-PDE generalization. |

For logistic RD, mean conservation is the wrong constraint. The relevant balance includes variance:

\[
\frac{d\bar u}{dt}=r\{\bar u(1-\bar u)-\operatorname{Var}(u)\}.
\]

Matching that instantaneous identity does not provide an exact future closure. Also,

\[
\operatorname{RMS}(e)^2=\bar e^2+\operatorname{RMS}(e-\bar e)^2.
\]

A bad replacement mean therefore imposes an error floor even when the spatial correction is excellent. This explains why physically motivated combinations must be tested for harm against their strongest component.

## Recommended decisions

| Keep / investigate | Simplify / defer | Promotion criterion |
|---|---|---|
| Final rank1 physical pair and pure input-compression ablation | Higher rank and stacked C1/C2 branches | Benefit beyond normalized same-cost analytic controls on fresh fields |
| Analytic quadratic and quadratic+cubic controls | Mandatory approximate mean replacement and fixed filtering | Combination beats its best component at equal accuracy and measured total cost |
| Track-consistent DF and optimized ETDRK4 | Claims based on weak/untrained FNO or smooth long-time endpoints | Credible FNO optimization plus identical outputs, schedules and physical targets |
| High-pair/rough/near-Nyquist stress tests alongside easy cases | Broad training expansion before attribution is resolved | Replicated regime benefit with enough independent fields and a positive cost margin |
| Frozen-model profiling, coefficient reuse and transform sharing | Deployment estimator stacks while base method is slower | Output parity and fair end-to-end speedup against equally optimized controls |
| Strict provenance and complete failed/infeasible coverage | Pooled test counts or weighted scores as “percent better” | Separate correctness, empirical accuracy, utility and uncertainty |

The next bounded experiment should be **small and decisive**: fixed normalized Gauss-2, frozen half-normalized control, scalar-amplitude-only fitting, full 43-parameter conditioning, four-node analytic quadratic, quadratic+cubic, credible small/standard FNO, DF and ETDRK4. Cross the essential node/cutoff factors, give all families the same endpoint schedules, and use fresh fields. Preserve existing checkpoints for diagnostic profiling; any new selection using the inspected cohort is development, not untouched confirmation.

My assessment is that **the project should continue as a focused study of learned corrections to physically derived interaction quadrature**. The current evidence supports that research direction. It does not justify marketing TDN as a general FNO replacement or an already superior adaptive solver.

## Evidence trail

- [Initial learned-control evidence](../results/learned_controls.json), [development gates](../results/gates.json), [early architecture definitions](RESEARCH_ARCHITECTURES.md).
- [Native CARC neural benchmark review](../results/carc-gpu-benchmark-review.json), [three-seed replication review](../results/carc-neural-replication-review.json).
- [Mechanism review](MECHANISM_REVIEW.md), [interaction review](INTERACTION_REVIEW.md), [work–precision review](WORK_PRECISION_REVIEW.md), [compact spatial review](COMPACT_SPATIAL_REVIEW.md).
- [Full agenda review](AGENDA_REVIEW.md) and [machine-readable evidence](../results/fedora-agenda-review.json).
- [Full roadmap review](ROADMAP_FULL_REVIEW.md), [neural detail](../results/roadmap-full-review/neural-review.md), [numerical detail](../results/roadmap-full-review/numerical-review.md), [policy detail](../results/roadmap-full-review/policy-review.md).
- [Full frontier review](FRONTIER_FULL_REVIEW.md), [learning/attribution data](../results/frontier-full-review/learning-review.json), [annotated 33-page atlas](../results/frontier-full-review/annotated/frontier-atlas.pdf).
- [Complete comparison annex and extraction provenance](../results/model-comparison-review/README.md). Existing reports are preserved; review conclusions do not change recorded measurements or historical scores.
