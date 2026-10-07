# Mechanism-directed literature and experimental questions

This ledger turns every search direction in the uploaded research audit into
an executable question. It is a mechanism agenda, not a claim to reproduce the
FNO paper or to have established a new publication result.

The six exact supplied links are retained below. Attempts to retrieve their
arXiv/SIAM pages in this cloud environment failed with an HTTPS proxy `403`
response on 2026-10-07. Article text, bibliographic metadata, and current version
were therefore **not independently verified here**. The labels below describe
the starting points supplied by the user audit; they are not fabricated paper
summaries. The implementation tests mathematical statements directly against
the declared logistic reaction–diffusion problem.

| ID | Supplied search direction and starting reference | Experimental question | Executable stages |
| --- | --- | --- | --- |
| Q1 | Non-asymptotic splitting, large gradients and stiff order reduction: [Descombes et al., supplied arXiv link](https://arxiv.org/abs/1402.1828) | When does cubic defect scaling stop being an adequate finite-step model? | `structure`, `controls`, `confirm` |
| Q2 | Exponential integrators and stiff order conditions: [Hochbruck and Ostermann, supplied SIAM DOI](https://epubs.siam.org/doi/10.1137/040611434) | Which input/output diffusion scales must the learned kernel retain? | `structure`, `kernel`, `confirm` |
| Q3 | Bilinear/Volterra operators, separable interaction kernels and learned exponential integrators; search “mode-pair” and “operator splitting defect” together | Can the pair kernel be compressed without losing phase-dependent response? | `structure`, `kernel`, `confirm` |
| Q4 | Operator aliasing/discretization consistency: [representation-equivalent operators, supplied arXiv link](https://arxiv.org/abs/2305.19913), [FNO discretization error, supplied arXiv link](https://arxiv.org/abs/2405.02221) | Which rough-field failures arise from representation mismatch rather than temporal error? | `structure`, `compression`, `confirm` |
| Q5 | Defect-based error estimators: [Auzinger, Hofstätter and Koch, supplied arXiv link](https://arxiv.org/abs/1806.07771) | Can a validated estimator eventually replace reference-informed step selection? | `prepare`, `policy` |
| Q6 | Variable-step operator learning: [Deep-OSG, supplied arXiv link](https://arxiv.org/abs/2302.03358) | Can unequal unseen steps remain accurate, rather than merely self-consistent? | `structure`, `confirm`, `policy` |
| Q7 | The audit's deployment extension; no specific paper supplied | Does an audited acceptance policy improve complete-solver accuracy and cost after rejected work and classical fallback? | `policy` |

## Established ideas, implementation deductions and hypotheses

**Established ingredients used as controls:** reaction-first and diffusion-first
Strang splitting; ETDRK4 and fused GL3; discrete Fourier eigenvalues;
phase-preserving bilinear interactions; rank/separable approximations;
step-doubling estimates; dealiased products; the instantaneous mean and variance
identities of the logistic equation. Giving these ingredients new names does
not establish novelty.

**Deductions from the inspected TDN implementation:** at quadratic amplitude,
a local quadratic commutator gate times a regular translation-equivariant
backbone has only a scalar multiple of the gate as its leading spatial
response. The historical moment head adds a constant direction. The historical
premix network has an affine time input and a single quadratic interaction,
restricting its proposed cubic-prefactor increment to a low-degree time
polynomial. These deductions are checked using the actual output capacity map;
they are not assumed to explain every observed failure.

**Hypotheses tested by this program:** zero-preserving transport of physical
sources can remove a quadratic spatial obstruction; nonlinear time decoding can
improve finite-step response; a compact eigenvalue-conditioned phase-aware
pair kernel can retain the relevant response economically; bounded
mean–variance evolution can improve mean accuracy without concealing spatial
damage. None of these hypotheses is accepted merely because structural tests
or training complete.

## What the questions measure

Q1 separates asymptotic local order from finite-step representability. Positive
and negative perturbations around an interior background isolate the quadratic
response. Reference error is amplified by division by perturbation amplitude
squared and remains in the reported uncertainty. The temporal panel compares
actual bounded outputs, rather than fitting only an unconstrained raw increment.

Q2 independently crosses reaction, diffusion, step size, mean and amplitude.
Mode pairs sharing an output frequency can have different input diffusion
rates; input-only and output-only ablations test whether all three eigenvalue
scales are needed. The highest eigenvalue describes short-wave stiffness;
the nonzero spectral gap describes global homogenization. They are distinct.

Q3 fits ranks 2, 4 and 8 before interpreting a learned kernel. Phase-varied
mixed modes, real-output checks and pair symmetry prevent a low-rank fit from
hiding phase loss. A nominal kernel rank can expand into several separable
terms once a physical source is included; both counts and every added transform
are reported. A low-rank model that costs more than the classical solver still
needs an independently demonstrated benefit.

Q4 uses the same continuous Fourier coefficients on multiple grids. Native
finite-difference/nodal products define the same-grid discrete target; a
separate dealiased spectral refinement defines the continuum-estimate track.
Spatial refinement and time refinement are both required for the latter.
Changing product conventions changes the target. Support masks and full-state
source bypasses are recorded when comparing compression order.

Q5 fits its empirical estimator envelope on calibration parents only, freezes
it, and evaluates accepted predictions against fresh independent references.
It reports both RMS and maximum-error false acceptance. This is an audited
empirical policy, not a certified bound.

Q6 compares genuinely unseen unequal-step schedules with independent endpoints
and reports composition agreement separately. Two wrong predictions can agree;
small composition error alone does not establish accurate evolution.

Q7 includes all proposal/estimator work, rejected trials, fallback calls and
complete fallback cost. A post-hoc reference-informed frontier is kept distinct
from this deployable policy. Direct ETDRK4 and fused GL3 remain the practical
cost controls.

The machine-readable mapping is [AGENDA_REQUIREMENTS.json](AGENDA_REQUIREMENTS.json).
The implementation, budgets and runbook are in [RESEARCH_AGENDA.md](RESEARCH_AGENDA.md).
