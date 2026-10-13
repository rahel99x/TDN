# Tangent response: related work and claim limits

This is a related-work audit for the separate Tangent Response implementation.
It identifies established ingredients and comparison obligations. It does not
establish novelty, an exhaustive prior-art search, learned accuracy, or utility.
The [research handout](TANGENT_RESPONSE_RESEARCH_HANDOUT.md) specifies the experiment;
the [analysis](TANGENT_RESPONSE_ANALYSIS.md) separates exact identities from
conditional bounds and measured approximations.

## Verification record

Titles, authors, publication years and identifiers for references [1–9] below
were checked against the Crossref REST records for their literal DOIs, using the
session's inherited HTTPS proxy and TLS verification. Reference [10] was checked
against its official JMLR page and the authors' NeuralOperator repository.
Reference [11] was checked against its official PMLR publication page, including
its abstract and citation metadata. The raw metadata, fetched pages and hashes
are retained in `.runtime/tangent-response-literature/verification.json`; the
portable [bibliographic verification receipt](../results/tangent-response-verification/literature-verification.json)
records the verified identities and hashes.

The first DEIM metadata request received HTTP 429; a later bounded retry returned
the expected record. The attempted NeuralOperator `CITATION.cff` returned 404;
the actual `README.rst` and official journal page provided the citation. These
are bibliographic and accessible abstract/source checks, not a claim to have
audited every theorem or implementation in the complete papers. No proxy,
certificate verification, or destination policy was bypassed.

## Comparison matrix

| Established area and references | Assumptions and information | Causal structure and analysis | Complete-cost comparison required here | Potential Tangent Response distinction to test |
| --- | --- | --- | --- | --- |
| Exponential and Rosenbrock methods [1,2] | Known finite RHS/Jacobian; exponential actions or Jacobian-based stages | Variational propagation, stiff-order and discretization questions are established; a frozen Jacobian differs from a time-varying anchor Jacobian | RHS/JVP evaluations, matrix-function actions, adaptive steps, transfers and error qualification | A state-conditioned approximation of time-varying response actions on specified residuals, if it is accurate and cheaper; adding a neural encoder alone is insufficient |
| Defect/deferred correction [3] | Numerical trajectory, residual and quadrature rules | Correcting the failure of a path to satisfy its equation is established; correction iterations require their own convergence conditions | Path/residual construction, source quadrature and all correction iterations | Quantified response-action approximation and nonlinear-remainder attribution on the declared finite equation |
| Variational/tangent equations [2] | A differentiable RHS and a fixed trajectory/path | The Jacobian evolution and variation-of-constants identity are established mathematics | Integration of homogeneous and forced tangent equations, independent refinement | Learned approximation of relevant full-field actions; neither the identity nor the tangent solve itself is novel |
| Krylov matrix-exponential actions [4] | Operator actions without a complete dense exponential | Approximation of an exponential acting on a vector; finite basis dimension is not a uniformly low-rank complete evolution operator | Basis construction, JVP/matvecs, orthogonalization, restarts and tolerances | Nonautonomous, state-conditioned residual-family actions with explicit held-out forcing directions; compare to an action-oriented numerical alternative |
| POD/DEIM and nonlinear reduction [5,6] | Training snapshots, reduced bases and sampled nonlinear evaluations | Source reconstruction, output sensitivity and nonlinear interpolation are distinct objectives | Snapshot acquisition, basis/Gram solves and actual online nonlinear evaluation | Response-weighted endpoint/action error and mean coupling; Response already has source-energy/response-weighted compression, so neither ingredient alone is a contribution |
| Goal/output-sensitive reduction and GNAT [6,7] | Training trajectories, retained output/residual directions and a declared reduced nonlinear system | Output/least-squares residual weighting matters; sampled residual minimization is not a full physical error certificate | Offline trajectories, reduced solves, sampling and complete residual reconstruction | A paired same-anchor comparison and measured response action/rank/cost tradeoff, rather than relabeling goal-sensitive reduction |
| Variable projection [8] | A linear coefficient subproblem conditional on nonlinear features | Stable conditional least squares does not globally optimize the nonlinear feature/path problem | SVD/ridge/cutoff work, factor conditioning and deployment precision | Attribution separating readout optimization from learned-factor limitation; the conditional solve is an established technique |
| Neural reduced models and operators [9,10] | Data/snapshots, frozen training/selection distributions, known or learned spatial representations | Learned reduced dynamics or maps between function spaces; empirical generalization depends on the actual information and cohort | Label acquisition, training, encoder/decoder, transfers, rollout and inference | Approximation of a declared tangent-response action plus explicit residual/path decoder and nonlinear-remainder evidence; compare supervision rather than calling unlike data budgets matched |
| Learned solver components/preconditioners [11] | Discretized equations and learned prolongation/operator components | A learned component can improve solver convergence; that is distinct from predicting a solution or finite-time propagator | Training, component construction, residual evaluation, iterations and fallback | Response factors must show physical accuracy and complete utility against qualifying numerical solvers; solver-component learning is not a new category |

The closest conceptual predecessors are variational propagation, defect
correction, and output-sensitive reduction. Their combination requires a precise
experimental claim. A possible claim is: on a frozen, disclosed family of
same-grid reaction–diffusion parents, a compact state-conditioned response action
retains a declared fraction of the spatial-tangent oracle improvement at an
explicit rank, precision and complete cost. This remains a hypothesis until
the mechanism, compression, fitting, held-out and utility decisions are measured.

Do not describe fixed-state oracle factors as deployable, one endpoint vector
as evidence of an operator's rank, or an inadequate FNO adaptation as defeat of
a published baseline. A comparison needs the same finite equation, actual input,
geometry, final time, physical RMS/maximum target, allowed information and
complete execution scope. Confirmatory claims require a later frozen protocol
and fresh uninspected parents.

## Verified references

1. Marlis Hochbruck and Alexander Ostermann (2010), **Exponential integrators**,
   *Acta Numerica*. DOI: [10.1017/S0962492910000048](https://doi.org/10.1017/S0962492910000048).
2. Ernst Hairer and Gerhard Wanner (1996), **Solving Ordinary Differential
   Equations II**, *Springer Series in Computational Mathematics*.
   DOI: [10.1007/978-3-642-05221-7](https://doi.org/10.1007/978-3-642-05221-7).
3. Alok Dutt, Leslie Greengard and Vladimir Rokhlin (2000), **Spectral Deferred
   Correction Methods for Ordinary Differential Equations**, *BIT Numerical
   Mathematics*. DOI: [10.1023/A:1022338906936](https://doi.org/10.1023/A:1022338906936).
4. Yousef Saad (1992), **Analysis of Some Krylov Subspace Approximations to the
   Matrix Exponential Operator**, *SIAM Journal on Numerical Analysis*.
   DOI: [10.1137/0729014](https://doi.org/10.1137/0729014).
5. Saifon Chaturantabut and Danny C. Sorensen (2010), **Nonlinear Model
   Reduction via Discrete Empirical Interpolation**, *SIAM Journal on Scientific
   Computing*. DOI: [10.1137/090766498](https://doi.org/10.1137/090766498).
6. Karen Willcox and Jaime Peraire (2002), **Balanced Model Reduction via the
   Proper Orthogonal Decomposition**, *AIAA Journal*.
   DOI: [10.2514/2.1570](https://doi.org/10.2514/2.1570).
7. Kevin Carlberg, Charbel Farhat, Julien Cortial and David Amsallem (2013),
   **The GNAT method for nonlinear model reduction: Effective implementation
   and application to computational fluid dynamics and turbulent flows**,
   *Journal of Computational Physics*.
   DOI: [10.1016/j.jcp.2013.02.028](https://doi.org/10.1016/j.jcp.2013.02.028).
8. Gene Golub and Victor Pereyra (2003), **Separable nonlinear least squares:
   the variable projection method and its applications**, *Inverse Problems*.
   DOI: [10.1088/0266-5611/19/2/201](https://doi.org/10.1088/0266-5611/19/2/201).
9. Jan S. Hesthaven and Stefano Ubbiali (2018), **Non-intrusive reduced order
   modeling of nonlinear problems using neural networks**, *Journal of
   Computational Physics*. DOI: [10.1016/j.jcp.2018.02.037](https://doi.org/10.1016/j.jcp.2018.02.037).
10. Nikola Kovachki, Zongyi Li, Burigede Liu, Kamyar Azizzadenesheli, Kaushik
    Bhattacharya, Andrew Stuart and Anima Anandkumar (2023), **Neural Operator:
    Learning Maps Between Function Spaces With Applications to PDEs**,
    *JMLR* 24(89):1–97. [Official publication](https://jmlr.org/papers/v24/21-1524.html);
    [authors' source](https://github.com/neuraloperator/neuraloperator).
11. Daniel Greenfeld, Meirav Galun, Ronen Basri, Irad Yavneh and Ron Kimmel
    (2019), **Learning to Optimize Multigrid PDE Solvers**, *ICML*, PMLR
    97:2415–2423. [Official publication and abstract](https://proceedings.mlr.press/v97/greenfeld19a.html).
