# Protected exploration: three distinct falsifiable prototypes

These experiments are the portfolio's separately budgeted path C. They run on
CPU after the mathematical audit and do not wait for neural training. The frozen
portfolio allocates 20% of discretionary scientific capacity to this path; unused
time is not evidence of a scientific failure and is not silently transferred to
another path. Actual time is reported separately from the allocation. Every unit
has its own bounded deadline, source/protocol seal and recorded outcome.

They are **new to TDN, not asserted to be novel in the literature**. They change
the representation, the information available to a coarse model, and the order in
which work is selected. None is merely a larger rank1 neural network.

| ID | Challenged assumption | Potential payoff | Main risk | Cheapest falsifier | Frozen advance criterion |
|---|---|---|---|---|---|
| X01 | A new horizon requires repeating all transported pair interactions | Encode a state once, query a continuous time interval cheaply | Encoding and storage overwhelm saved work; reuse invalid after state/operator changes | Closed-form pair response versus a compact temporal encoding, cached exact rule, FFT quadrature and classical dense output | Maximum query relative error ≤0.001; independent quadrature agreement ≤1e-10; cold-inclusive break-even against cached exact pairs ≤32 queries |
| X02 | Instantaneous resolved state can specify missing high-to-low forcing | Establish when short history supplies information that a larger instantaneous network cannot recover | Backward-generated smooth histories are too favorable; acquisition and causal rollout error erase benefit | Indistinguishable coarse twin states with opposite unresolved cross-products | Resolved difference ≤1e-12, future target difference >1e-8; heldout memory RMSE ≤half instantaneous-fit RMSE |
| X03 | Expensive interactions must be formed before deciding their importance | Skip products and transport kernels before paying for them | Cheap features miss phase cancellation; sorting and masking cost more than dense vectorized work | Fit a tiny rank rule on development fields and compare with deterministic tail selection | Every heldout field relative error ≤1%; measured geometric mean end-to-end speedup ≥1.10× |

The criteria are declared in `protocol.py` and checked against the implementation
before execution. A completed job, an accurate representation, and a useful
end-to-end solver are different outcomes. A failed cost gate remains failed even
when mathematical checks pass.

## X01: encode the continuous interaction response

For the quadratic semilinear problem

\[
v_t=Lv+qv^2,\qquad L=\nu\partial_{xx}+\rho,
\]

consider only the second-Picard term in its variation-of-constants expansion:

\[
Q_h(v)=\int_0^h e^{(h-s)L}(e^{sL}v)^2\,ds.
\]

The coefficient of output mode \(k=p+r\) is a sum of signed complex products
\(\widehat v_p\widehat v_r\) weighted by

\[
K_h(a,b)=\int_0^h e^{(h-s)a+sb}\,ds
=\begin{cases}
 (e^{hb}-e^{ha})/(b-a),&a\ne b,\\
 h e^{ha},&a=b,
\end{cases}
\quad a=\lambda_k,\quad b=\lambda_p+\lambda_r.
\]

This identity is established mathematics. Stable `expm1` evaluation avoids a
removable singularity and avoids multiplying overflow by underflow for large
negative rates. Integer mode sums are non-aliased; output compression is applied
after interactions. The bounded prototype uses input modes −15…15 and output
modes −6…6, so no large all-pairs tensor is allocated.

The new-to-project representation is

\[
K_h(a,b)/h\approx\sum_{j=0}^{12} c_j(a,b)T_j(2h/H-1),\qquad
B_{jk}(v)=\sum_{p+r=k}\widehat v_p\widehat v_r c_j(\lambda_k,\lambda_p+\lambda_r).
\]

At query time the decoder evaluates
\(Q_h(v)\approx h\sum_j B_{jk}(v)T_j(2h/H-1)\).
The factor \(h\) makes the zero-horizon response exact. This is a compact,
continuous-time physical interaction encoding, fitted to a known mathematical
kernel, **not a neural model trained against PDE solutions**. Coefficients are
computed once for a fixed spatial operator and sealed interval \([0,H]\).
The implementation rejects extrapolation instead of silently assuming validity.

Encoding costs include all pair products and basis contractions. Changing the
state requires a fresh state encoding; changing physical parameters, modes or the
interval requires a new operator basis. Reusing an encoding through a nonlinear
rollout without refreshing it is invalid. The first experiment tests one
synthetic state and many paired horizons, not many independent fields.

The comparator set receives equivalent reuse opportunities:

- Exact closed-form pairs, both with and without cached state products.
- Two- and four-node normalized Gauss rules with cached quadrature nodes.
- Batched, zero-padded FFT four-node quadrature, avoiding the all-pairs approach.
- DOP853 dense output on a lifted triangular ODE that computes exactly the same
  quadratic response. Its integration/build cost is included. It does **not**
  solve a different full nonlinear task and then compete against an easier task.

Query order is randomized within repeated rounds. First invocation and warm
samples are distinct. `temporal_amortization.json` reports measured build cost
plus query-count times measured median latency, clearly labeled as a cost model,
not a measured batch benchmark. It includes the separate scenario where the
operator basis already exists and only the state encoding is refreshed. Passing
the primary break-even gate would justify a larger repeated-query pilot, not a
claim against the best classical full solver.

Artifacts: `temporal_query_rows.json`, `temporal_amortization.json`,
`temporal_encoding.json`. The latter retains all basis/encoding coefficients and
explicit array bytes; array size is not asserted to be process peak memory.

## X02: prove an information limitation, then test history

Let the resolved cutoff be two, and construct current states

\[
u_\pm(x)=m+A\cos(9x+\phi)\pm B\cos(10x+\psi).
\]

They have identical resolved coefficients and identical variance. However,
their nonlinear product has opposite mode-one forcing, because

\[
2\cos(9x+\phi)\cos(10x+\psi)
=\cos(x+\psi-\phi)+\cos(19x+\psi+\phi).
\]

For logistic reaction the instantaneous unresolved contribution to complex
mode one is \(\mp\beta AB e^{i(\psi-\phi)}/2\). Any deterministic predictor
receiving only the identical resolved state must return the same output for both
twins. For two targets \(y_+,y_-\), its best possible pair RMSE is at least
\(\|y_+-y_-\|/2\). Adding network capacity cannot remove this information
limitation. It is a statement about the specified information interface, not a
claim that every coarse closure needs memory.

The experiment independently integrates the finite non-aliased Galerkin logistic
equation with modes −15…15. It records a future time \(h=0.01\) and a short
backward-generated history \(\delta=0.002\). Actual refinement with fixed-step
RK4 at 8 and 16 steps cross-checks adaptive DOP853 on the declared four-state
subset. This verifies a discrete equation; it does not establish continuum
convergence or justify long backward integration of diffusion.

Fit and heldout-development parents are distinct; both twins of a parent stay
in the same split. Comparators are:

1. A non-neural least-squares instantaneous map from mean and variance features.
2. Deterministic memory persistence, \(\widehat u_1(h)\approx h d_\mathrm{past}\).
3. A two-coefficient fitted memory rule,
   \(\widehat u_1(h)\approx (a+bm)d_\mathrm{past}\).
4. A full-state instantaneous oracle, using the unavailable signed high-mode
   product. This is an information control, not a deployable coarse competitor.

Real and imaginary components share the fitted coefficients, preserving phase
rotation behavior. The model is non-neural. History generation is timed and
reported separately from prediction. One may consider a scenario with existing
history, but it is invalid to present newly computed fine-scale history as free.

A favorable result justifies a **causal, forward-history rollout pilot** against
deterministic memory rules. It does not justify deploying a universal learned
closure. The next pilot must vary unresolved spectra, use naturally generated
trajectories, include history noise, count startup cost, and measure closure
drift over repeated coarse steps. The existing information limit should not be
confused with proof that the proposed memory features are sufficient.

Artifacts: `closure_rows.json`, `closure_fit.json`. Raw current Fourier states,
history features, future targets, fits, parent identities, acquisition time and
independent cross-checks are retained. Smoke has eight independent heldout twin
parents; development/full prototyping has twenty. The full profile remains a
development prototype and never accesses confirmation.

## X03: select work before creating complex products

The full response uses all non-aliased pairs feeding output modes −8…8 from
input modes −63…63. Selection retains conjugate mode pairs together, protecting
real-valued outputs. It chooses modes before computing complex pair products or
transport exponentials, although the common pair-index geometry remains cached
for every method.

The deterministic selector uses an inexpensive L1 tail proxy. If \(S\) is the
retained input set and \(M=\sum_p|\widehat u_p|\), then the sum of omitted
absolute products is

\[
M^2-\left(\sum_{p\in S}|\widehat u_p|\right)^2.
\]

Multiplying by a bound on \(|K_h|\) bounds absolute discarded contributions.
The normalized tail proxy is **not a relative output error guarantee** when
signed interactions cancel. Heldout relative error is therefore measured rather
than inferred from the proxy.

The competing fitted selector uses ordinary least squares with four features:
bias, spectral entropy, energy-weighted frequency and scaled L1²/L2². Its target
is the smallest candidate rank achieving 1% error on its fit fields. It is a
tiny distilled rule, not an oracle at deployment. Candidate ranks are frozen at
2, 4, 8, 16, 32 and 63 positive modes. It cannot consult the reference, the
heldout optimal rank, or heldout error while selecting work.

Smooth, sparse and rough spectra occur in both disjoint development splits.
Output retains regime labels and per-field failures. End-to-end randomized
timings include feature extraction, sorting/prediction, mask construction,
products, kernels and output summation. Selection-only timing is diagnostic and
is not substituted for total cost. The dense vectorized exact rule receives the
same cached geometry. Slower selection is a useful negative result: operation
counts alone cannot establish a benefit on modern hardware.

An additional phase audit holds all spectral magnitudes fixed and varies eight
phase configurations. Its shared features cannot encode changing cancellations.
This is one correlated amplitude parent and is excluded from independent-field
timing/error aggregates. Its gap verdict is explicitly NA: the audit records
errors without introducing a post-hoc success threshold.

Artifacts: `selection_rows.json`, `selection_fit.json`,
`selection_states.json`, `selection_phase_audit.json`. All exact targets, candidate errors, predicted ranks,
raw Fourier states and timing rounds are retained. A successful CPU pilot would
still require a larger-workload and actual-GPU test, including launch/masking
overhead and complete solver accuracy. No such claim follows automatically.

## Prior art and limits on novelty claims

The following authoritative implementations and their stated references were
read for this implementation. This is a focused prior-art check, not an
exhaustive search or a publication-priority determination. The underlying
techniques predate this project; merely combining them with TDN is insufficient
to establish novelty.

| Topic | Verified source | Consequence for our interpretation |
|---|---|---|
| Exponential propagation and semilinear PDE integrators | [Exponax implementation and references](https://github.com/Ceyron/exponax); its references include Cox–Matthews (2002), Kassam–Trefethen (2005), and Montanelli–Bootland (2020) | Variation of constants, exponential treatment of stiffness and Fourier implementations are established controls, not TDN inventions. Exponax also identifies its connection to the NeurIPS 2024 APEBench benchmark. |
| Classical dense output | [SciPy RK/DOP853 implementation](https://github.com/scipy/scipy/blob/main/scipy/integrate/_ivp/rk.py) | Amortized temporal queries must compete with existing dense-output reuse. The prototype uses SciPy's actual implementation. |
| Continuous-time learned systems | [torchdiffeq and Neural ODE references](https://github.com/rtqichen/torchdiffeq) | Continuous-time querying alone is not a novel capability. Our proposed distinction is a reusable physical interaction representation with measured build/query economics. |
| Data-driven polynomial reduced dynamics | [Operator Inference](https://github.com/operator-inference/opinf), linking Peherstorfer–Willcox (2016) | Fitting reduced quadratic structure is established. Our twin experiment diagnoses whether its chosen state variables contain enough information. |
| Distillation into concise equations | [PySINDy](https://github.com/dynamicslab/pysindy), citing sparse identification work of Brunton et al. | Replacing a learned behavior with a short fitted rule is a legitimate numerical outcome, but not itself a novel research claim. |

The Mori–Zwanzig projection perspective also motivates the distinction between
an instantaneous closure and a history-dependent closure. This implementation
does not approximate a Mori–Zwanzig memory kernel or claim equivalence to that
formalism: it tests an explicit finite information counterexample and one cheap
history statistic. A literature-novel closure would require a separate targeted
review and materially broader evidence.

## Validation and interpretation

`tests/test_portfolio_prototypes.py` exercises the equal-rate kernel limit,
high-order independent quadrature, non-aliased FFT agreement, continuous-query
validity and state refresh, classical dense output on the same task, twin
indistinguishability, logistic constant-state behavior, conjugate-preserving
selection, independent parent splits and complete execution with raw artifacts.
No unit test asserts a machine-dependent speedup. `prototype_rows.json` and the
normal stage log preserve all successes, failures and NA checks.

The decision sequence is deliberately asymmetric: inexpensive mathematical
diagnostics can identify a promising representation or an impossibility, while
practical utility must earn a separate measured cost result. Even a negative
prototype can advance the research by ruling out a proposed saving before a
large GPU campaign is launched.
