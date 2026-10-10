"""D07: measured reuse, with kernel and complete finite-PDE tasks kept separate.

These CPU diagnostics are development experiments, not GPU or continuum claims.
The old X01 kernel construction is reused explicitly, including its cost failure;
actual build-and-many-query measurements replace extrapolation-only evidence.
"""
from __future__ import annotations

import hashlib
import math
import statistics
import time

import numpy as np
import torch
from numpy.polynomial.chebyshev import chebfit, chebval
from scipy.integrate import solve_ivp

from tdn.analysis.frontier.core import check
from tdn.analysis.frontier import numerics
from tdn.analysis.frontier.data import lawson_reference
from tdn.analysis.portfolio import prototypes as previous
from tdn.analysis.portfolio.diagnostics import _state
from tdn.analysis.roadmap.numerics import quadratic_df_defect
from tdn.numerics import Equation, Geometry
from tdn.runtime.metadata import write_json


def nominal_break_even(encoding_seconds, direct_seconds, query_seconds):
    """Smallest positive integer Q with E+Qq<Qd; unavailable when d<=q."""
    values = (encoding_seconds, direct_seconds, query_seconds)
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Cost model requires finite nonnegative times")
    if direct_seconds <= query_seconds:
        return None
    return max(1, math.floor(encoding_seconds / (direct_seconds-query_seconds))+1)


def state_fingerprint(u, equation, geometry, track, horizon):
    """Strict validity identity. Its validation overhead is measured, not free."""
    if u.device.type != "cpu":
        raise ValueError("This diagnostic measures CPU; no implicit GPU transfer")
    identity = repr((tuple(u.shape), str(u.dtype), equation.kappa, equation.reaction_rate,
                     geometry.grid, geometry.lengths, track, float(horizon))).encode()
    return hashlib.sha256(identity + u.detach().contiguous().numpy().tobytes()).hexdigest()


class CorrectionEncoding:
    """Chebyshev interpolation of Q(h)/h^3, retaining exact zero at h=0.

    The stored object contains one immutable state's physical correction, not a
    neural solution operator. A different state requires a complete refresh.
    Decoding still pays for the DF backbone and physical-field reconstruction.
    """
    def __init__(self, u, equation, geometry, track, horizon, *, degree=8, nodes=4):
        if horizon <= 0 or not math.isfinite(horizon) or degree < 1 or nodes < 1:
            raise ValueError("Positive interval, polynomial degree and quadrature count required")
        numerics.validate(u, 0., geometry, track)
        self.identity = state_fingerprint(u, equation, geometry, track, horizon)
        self.state = u.detach().clone()
        self.equation, self.geometry, self.track = equation, geometry, track
        self.horizon, self.degree, self.nodes = float(horizon), degree, nodes
        self.cache = numerics.CoefficientCache()
        x = np.cos(np.pi*(np.arange(degree+1)+.5)/(degree+1))
        times = horizon*(x+1)/2
        with torch.no_grad():
            values = np.stack([(quadratic_df_defect(u, float(h), equation, geometry,
                target=track, nodes=nodes)/float(h)**3).numpy().ravel() for h in times])
        self.coefficients = chebfit(x, values, degree)

    def validate(self, u, equation, geometry, track, horizon):
        if state_fingerprint(u, equation, geometry, track, horizon) != self.identity:
            raise ValueError("Changed state/operator/grid/precision/interval requires encoding refresh")

    def correction(self, h):
        if not math.isfinite(h) or not 0 <= h <= self.horizon:
            raise ValueError("Query outside encoded interval")
        values = h**3 * chebval(2*h/self.horizon-1, self.coefficients)
        return torch.from_numpy(values.copy()).reshape_as(self.state).to(self.state.dtype)

    def query(self, h):
        return numerics.df_step(self.state, h, self.equation, self.geometry, self.track,
                                cache=self.cache) + self.correction(h)

    @property
    def storage_bytes(self):
        return self.coefficients.nbytes + self.state.numel()*self.state.element_size()


def dense_pde(u, equation, geometry, track, horizon, *, rtol=2e-11, atol=2e-13):
    """DOP853 build with dense output for exactly the same finite equation."""
    if u.device.type != "cpu" or u.dtype != torch.float64:
        raise ValueError("Dense reference and cost control require a CPU FP64 state")
    def rhs(_, values):
        field = torch.from_numpy(values.copy()).reshape_as(u)
        return numerics.rhs(field, equation, geometry, track).numpy().ravel()
    result = solve_ivp(rhs, (0., horizon), u.numpy().ravel(), method="DOP853",
                       dense_output=True, rtol=rtol, atol=atol)
    if not result.success:
        raise RuntimeError("DOP853 dense build failed: " + result.message)
    return result


def paired_timing(calls, *, repeats, rng, budget):
    """Randomized CPU end-to-end wall time; first invocation kept separately."""
    raw = {name: [] for name in calls}; first = {}
    for name in rng.permutation(list(calls)):
        budget.check(); start = time.perf_counter(); calls[name]()
        first[name] = time.perf_counter()-start
    for repeat in range(repeats):
        for order, name in enumerate(rng.permutation(list(calls))):
            budget.check(); start = time.perf_counter(); calls[name]()
            raw[name].append(dict(repeat=repeat, order=order, seconds=time.perf_counter()-start))
    return {name: dict(first_invocation_seconds=first[name],
                      median_seconds=statistics.median(r["seconds"] for r in rows),
                      raw=rows, device="cpu", synchronization="CPU synchronous calls",
                      first_invocation_scope="warm process, not cold process") for name, rows in raw.items()}


def _measure(call):
    start = time.perf_counter(); answer = call()
    return answer, time.perf_counter()-start


def _error(actual, expected):
    delta = np.asarray(actual)-np.asarray(expected)
    return dict(error_rms=float(np.sqrt(np.mean(abs(delta)**2))), error_max=float(np.max(abs(delta))))


def _emit(ctx, rows, identifier, metrics, checks, **configuration):
    row = dict(schema="tdn.adjacent-temporal/v1", diagnostic_id="D07", experiment_id=identifier,
               metrics=metrics, checks=checks, config=configuration, status="COMPLETED")
    rows.append(row)
    ctx.record(identifier, ["D07"], metrics=metrics, checks=checks, config=configuration,
               evidence=["temporal_rows.json", "temporal_cost_rows.json"])


def _kernel(ctx, rows, costs, rng, options):
    geometry = previous.pair_geometry(7 if options["smoke"] else 15, 3, diffusion=.03)
    field = previous._field(geometry, rng)
    horizon, degree = .2, 10
    basis, preparation = _measure(lambda: previous.temporal_basis(geometry, horizon, degree))
    encoded, encoding = _measure(lambda: previous.encode_temporal(field, geometry, basis))
    products, product_build = _measure(lambda: field[geometry.p]*field[geometry.q])
    dense, dense_build = _measure(lambda: previous.dense_quadratic_solution(field, geometry, horizon))
    def calls(times, rebuild=False):
        def compact():
            e = previous.encode_temporal(field, geometry,
                previous.temporal_basis(geometry, horizon, degree)) if rebuild else encoded
            return np.stack([previous.temporal_query(e, float(h)) for h in times])
        def exact():
            p = field[geometry.p]*field[geometry.q] if rebuild else products
            return np.stack([previous.pair_query(field, geometry, float(h), products=p) for h in times])
        def classical():
            d = previous.dense_quadratic_solution(field, geometry, horizon) if rebuild else dense
            return d.sol(times)[len(field):].T.copy()
        return {"compact_kernel": compact, "cached_exact_kernel": exact,
                "fft_gl4": lambda: np.stack([previous.fft_quadrature(field, geometry, float(h)) for h in times]),
                "dop853_lifted_dense": classical}
    for count in options["query_counts"]:
        times = np.linspace(horizon/count, horizon, count)
        methods = calls(times); reference = methods["cached_exact_kernel"]()
        warm = paired_timing(methods, repeats=options["repeats"], rng=rng, budget=ctx.budget)
        complete = paired_timing(calls(times, True), repeats=options["repeats"], rng=rng, budget=ctx.budget)
        for method, call in methods.items():
            error = _error(call(), reference)
            build = {"compact_kernel": preparation+encoding, "cached_exact_kernel": product_build,
                     "dop853_lifted_dense": dense_build, "fft_gl4": 0.}[method]
            costs.append(dict(task="quadratic_kernel_only", workload="same_state_queries", method=method,
                query_count=count, measured_query_seconds=warm[method]["median_seconds"],
                measured_complete_seconds=complete[method]["median_seconds"],
                modeled_complete_seconds=build+warm[method]["median_seconds"], build_seconds=build,
                timing=complete[method], query_timing=warm[method], **error))
            _emit(ctx, rows, f"D07/kernel/{count}/{method}", dict(error, query_count=count,
                complete_seconds=complete[method]["median_seconds"], storage_bytes=(encoded["coefficients"].nbytes
                +basis["coefficients"].nbytes) if method=="compact_kernel" else None),
                [check("same-kernel", error["error_max"], 1e-8, category="math"),
                 check("full-pde-accuracy", None, None, reason="This task is only a lifted quadratic response")],
                task="quadratic_kernel_only", workload="same_state_queries", method=method,
                role="Ours" if method=="compact_kernel" else "Theirs" if "dop853" in method else "Analytic control")
    return dict(operator_preparation_seconds=preparation, encoding_seconds=encoding,
                classical_build_seconds=dense_build, prior_cost_failure="X01 exceeded realistic query budget",
                changed_premise="Actual finite query groups and full build-and-query costs are now measured")


def _pde(ctx, rows, costs, rng, options):
    n, horizon = (8 if options["smoke"] else 16), .12
    u = _state(n, "high_pair", options["seed"]); equation=Equation(.004, 3.)
    geometry = Geometry((n,n),(1.,1.)); cache_results = []; build_rows = []
    for track in options["tracks"]:
        ctx.budget.check()
        encoded, encoding = _measure(lambda: CorrectionEncoding(u, equation, geometry, track, horizon))
        dense, dense_build = _measure(lambda: dense_pde(u, equation, geometry, track, horizon))
        direct_cache = numerics.CoefficientCache()
        def direct(h):
            return numerics.df_step(u, float(h), equation, geometry, track, cache=direct_cache) + quadratic_df_defect(
                u, float(h), equation, geometry, target=track, nodes=4)
        _, validation_seconds = _measure(lambda: encoded.validate(u,equation,geometry,track,horizon))
        _, reconstruction_seconds = _measure(lambda: encoded.correction(horizon))
        reference_start=time.perf_counter()
        tight_dense=dense_pde(u,equation,geometry,track,horizon,rtol=2e-13,atol=2e-15)
        teacher = lawson_reference(u,horizon,equation,geometry,128,track,check_budget=ctx.budget.check)
        teacher2 = lawson_reference(u,horizon,equation,geometry,256,track,check_budget=ctx.budget.check)
        reference_seconds=time.perf_counter()-reference_start
        dense_endpoint = dense.sol(horizon).reshape(u.shape)
        reference_uncertainty = max(float((teacher2-teacher).abs().max()),
                                    float(np.max(abs(dense_endpoint-teacher2.numpy()))),1e-12)
        build_rows.append(dict(track=track, encoding_seconds=encoding, classical_build_seconds=dense_build,
            validation_seconds=validation_seconds, query_reconstruction_seconds=reconstruction_seconds,
            storage_bytes=encoded.storage_bytes, compilation_seconds=None, transfer_seconds=0.,
            reference_generation_seconds=reference_seconds,
            transfer_note="CPU-only; no host/device copy, NumPy reconstruction included in query",
            reference_uncertainty_max=reference_uncertainty, reference_is_certificate=False))
        for field, nequation, ngeom, ntrack, interval, change in [
            (u+1e-6,equation,geometry,track,horizon,"state"),
            (u,Equation(.005,3.),geometry,track,horizon,"parameter"),
            (u,equation,Geometry((n,n),(2.,1.)),track,horizon,"operator_domain"),
            (u.float(),equation,geometry,track,horizon,"precision"),
            (u,equation,geometry,"continuum" if track=="discrete" else "discrete",horizon,"spatial_equation"),
            (u,equation,geometry,track,horizon*2,"interval")]:
            try: encoded.validate(field,nequation,ngeom,ntrack,interval); rejected=False
            except ValueError: rejected=True
            cache_results.append(dict(track=track,change=change,rejected=rejected))
        for count in options["query_counts"]:
            times=np.linspace(horizon/count,horizon,count)
            def compact_queries():
                encoded.validate(u,equation,geometry,track,horizon)
                return np.stack([encoded.query(float(h)).numpy() for h in times])
            def compact_complete():
                e=CorrectionEncoding(u,equation,geometry,track,horizon)
                e.validate(u,equation,geometry,track,horizon)
                return np.stack([e.query(float(h)).numpy() for h in times])
            def dense_complete():
                d=dense_pde(u,equation,geometry,track,horizon)
                return d.sol(times).T.reshape((count,)+tuple(u.shape)).copy()
            def direct_complete():
                direct_cache.clear()
                return np.stack([direct(h).numpy() for h in times])
            calls={"cached_correction":compact_queries,
                   "direct_gl4":lambda:np.stack([direct(h).numpy() for h in times]),
                   "dop853_dense":lambda:dense.sol(times).T.reshape((count,)+tuple(u.shape)).copy()}
            complete_calls={"cached_correction":compact_complete,"direct_gl4":direct_complete,"dop853_dense":dense_complete}
            warm=paired_timing(calls,repeats=options["repeats"],rng=rng,budget=ctx.budget)
            complete=paired_timing(complete_calls,repeats=options["repeats"],rng=rng,budget=ctx.budget)
            reference=tight_dense.sol(times).T.reshape((count,)+tuple(u.shape))
            time_query_uncertainty=max(reference_uncertainty,
                float(np.max(abs(reference-calls["dop853_dense"]()))))
            direct_answer=calls["direct_gl4"]()
            for method,call in calls.items():
                answer=call(); error=_error(answer,reference)
                interpolation=_error(answer,direct_answer) if method=="cached_correction" else None
                for workload in ("same_state_queries","trajectory_dense_output"):
                    # Same measured computation, two explicitly linked operational interpretations.
                    costs.append(dict(task="complete_finite_pde",track=track,workload=workload,method=method,
                        query_count=count,measurement_id=f"pde/{track}/{count}/{method}",
                        shared_measurement=True,measured_query_seconds=warm[method]["median_seconds"],
                        measured_complete_seconds=complete[method]["median_seconds"],
                        modeled_complete_seconds=({"cached_correction":encoding,"direct_gl4":0.,"dop853_dense":dense_build}[method]
                                                  +warm[method]["median_seconds"]),
                        timing=complete[method],query_timing=warm[method],**error))
                _emit(ctx,rows,f"D07/pde/{track}/{count}/{method}",dict(error,query_count=count,
                    complete_seconds=complete[method]["median_seconds"],reference_uncertainty_max=time_query_uncertainty,
                    correction_interpolation_error=interpolation),
                    [check("dense-independent-crosscheck",time_query_uncertainty,2e-8,category="math"),
                     check("complete-pde-target",error["error_max"],2e-5),
                     check("reuse-over-direct",complete[method]["median_seconds"],complete["direct_gl4"]["median_seconds"],category="utility")],
                    task="complete_finite_pde",track=track,method=method,
                    role="Ours" if method=="cached_correction" else "Theirs" if method=="dop853_dense" else "Analytic control",
                    interpretation="One initial state, many horizons; not composition of predicted states")
        # Parameter queries invalidate both learned/analytic state encodings and classical dense builds.
        parameter_equations=[Equation(equation.kappa*factor,3.) for factor in (.8,1.,1.2)]
        def parameter_compact():
            return torch.stack([CorrectionEncoding(u,e,geometry,track,horizon).query(horizon) for e in parameter_equations])
        def parameter_direct():
            return torch.stack([numerics.df_step(u,horizon,e,geometry,track)+quadratic_df_defect(
                u,horizon,e,geometry,target=track,nodes=4) for e in parameter_equations])
        def parameter_dense():
            return np.stack([dense_pde(u,e,geometry,track,horizon).sol(horizon) for e in parameter_equations])
        pt=paired_timing({"cached_correction":parameter_compact,"direct_gl4":parameter_direct,"dop853_dense":parameter_dense},
                        repeats=options["repeats"],rng=rng,budget=ctx.budget)
        parameter_reference=parameter_dense().reshape((3,)+tuple(u.shape))
        for method,call in {"cached_correction":parameter_compact,"direct_gl4":parameter_direct,"dop853_dense":parameter_dense}.items():
            answer=np.asarray(call()).reshape(parameter_reference.shape)
            costs.append(dict(task="complete_finite_pde",track=track,workload="parameter_queries",method=method,
                query_count=3,refresh_count=3,measured_complete_seconds=pt[method]["median_seconds"],timing=pt[method],
                **_error(answer,parameter_reference)))
        # Every autonomous step must encode the *predicted* state, with all refresh work charged.
        steps=(.02,.03,.01,.04)
        def rollout(method):
            state=u.clone()
            for dt in steps:
                if method=="cached_correction":
                    state=CorrectionEncoding(state,equation,geometry,track,dt).query(dt)
                elif method=="direct_gl4":
                    state=numerics.df_step(state,dt,equation,geometry,track)+quadratic_df_defect(
                        state,dt,equation,geometry,target=track,nodes=4)
                else:
                    d=dense_pde(state,equation,geometry,track,dt)
                    state=torch.from_numpy(d.sol(dt).copy()).reshape_as(state)
            return state.numpy()
        rc={m:(lambda m=m:rollout(m)) for m in ("cached_correction","direct_gl4","dop853_dense")}
        rt=paired_timing(rc,repeats=options["repeats"],rng=rng,budget=ctx.budget)
        rollout_reference=tight_dense.sol(sum(steps)).reshape(u.shape)
        for method,call in rc.items():
            costs.append(dict(task="complete_finite_pde",track=track,workload="autonomous_rollout",method=method,
                query_count=len(steps),refresh_count=len(steps),steps=list(steps),final_time=sum(steps),
                measured_complete_seconds=rt[method]["median_seconds"],timing=rt[method],**_error(call(),rollout_reference)))
    write_json(ctx.path/"temporal_cache_checks.json",cache_results)
    return build_rows


def run(ctx):
    if str(getattr(ctx,"device","cpu"))!="cpu":
        raise ValueError("D07 explicitly measures CPU, not native GPU")
    smoke=ctx.protocol.get("profile")=="smoke"
    options=dict(smoke=smoke,seed=934071,tracks=ctx.protocol.get("tracks",["discrete","continuum"]),
                 repeats=2 if smoke else 5,query_counts=[1,4,16] if smoke else [1,4,16,32,64])
    options.update(ctx.protocol.get("temporal",{}))
    rows=[];costs=[];rng=np.random.default_rng(options["seed"])
    with torch.no_grad():
        kernel=_kernel(ctx,rows,costs,rng,options)
        builds=_pde(ctx,rows,costs,rng,options)
    # Nominal counts are secondary; only measured, accurate finite Q can pass.
    decisions=[]
    for track in options["tracks"]:
        points=[r for r in costs if r.get("track")==track and r["workload"]=="same_state_queries"]
        queries=sorted({r["query_count"] for r in points})
        eligible=[]
        for count in queries:
            group={r["method"]:r for r in points if r["query_count"]==count}
            ours=group["cached_correction"]
            if ours["error_max"]<=2e-5 and all(ours["measured_complete_seconds"]<r["measured_complete_seconds"]
                for name,r in group.items() if name!="cached_correction"):
                eligible.append(count)
        group={r["method"]:r for r in points if r["query_count"]==queries[-1]}
        build=next(r for r in builds if r["track"]==track)
        nominal=nominal_break_even(build["encoding_seconds"],group["direct_gl4"]["measured_query_seconds"]/queries[-1],
                                  group["cached_correction"]["measured_query_seconds"]/queries[-1])
        decisions.append(dict(track=track,nominal_break_even_queries=nominal,
            measured_accuracy_qualified_break_even_queries=min(eligible) if eligible else None,
            utility="GOOD" if eligible and min(eligible)<=32 else "BAD",
            decision="FOCUSED_REUSE_PILOT" if eligible and min(eligible)<=32 else "STOP_CURRENT_COST_CLAIM",
            qualification="Single independent development field; no confidence interval or GPU claim"))
    write_json(ctx.path/"temporal_rows.json",rows)
    write_json(ctx.path/"temporal_cost_rows.json",costs)
    write_json(ctx.path/"temporal_build_costs.json",dict(kernel=kernel,pde=builds,decisions=decisions))
    write_json(ctx.path/"temporal_applicability.json",{
        "historical_temporal_mlp": {"status":"NA",
            "reason":"No compatible sealed trained checkpoint is supplied to this adjacent development stage; random initialization is not a temporal competitor"},
        "published_dense_output_reproduction": {"status":"NA",
            "reason":"SciPy DOP853 uses the same finite equation through a Torch/NumPy CPU RHS adapter; this is an implementation control, not an optimized reference-library benchmark"},
        "continuum_solution": {"status":"NA",
            "reason":"The continuum-labeled operator is the finite Galerkin equation here; D07 does not perform spatial refinement"},
        "timing_policy":"Cached competitors receive the same fixed-state interval. Complete timings rebuild every representation; query-only timings reuse all representations."})
    return dict(status="COMPLETED",diagnostic_id="D07",row_count=len(rows),cost_rows=len(costs),
                decisions=decisions,device="cpu",energy_joules=None,monetary_cost=None,
                interpretation="Kernel representation and complete finite-equation accuracy are distinct results")
