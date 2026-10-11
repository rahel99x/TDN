"""Saved independent Appendix A probes; numerical evidence, never a proof.

NumPy/SciPy one-dimensional controls do not call the Torch operator under test.
All designed interventions and both shape templates remain explicitly labeled.
"""
from __future__ import annotations
import hashlib
import itertools
import json
import math
import platform
import time

import numpy as np
import scipy
from scipy.integrate import solve_ivp, quad
from scipy.signal import resample as sp_resample
import torch

from tdn.runtime.metadata import write_json
from tdn.numerics.types import Geometry
from .contracts import Physics
from .core import check, clean
from . import numerics as n, kernels as k, trajectory as tr


def _rms(x):
    return float(np.sqrt(np.mean(np.abs(np.asarray(x))**2)))


def _array(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def _tensor(x):
    return torch.from_numpy(np.asarray(x,dtype=np.float64).copy())[None,None,:]


def _np_product(a,b):
    N=len(a)
    return sp_resample(sp_resample(a,3*N)*sp_resample(b,3*N),N)


def _np_rhs(u,kappa,r,target):
    N=len(u)
    if target == 'fd_same_grid':
        diffusion=kappa*N*N*(np.roll(u,1)-2*u+np.roll(u,-1));square=u*u
    else:
        symbol=-kappa*(2*np.pi*np.fft.fftfreq(N,d=1/N))**2
        diffusion=np.fft.ifft(np.fft.fft(u)*symbol).real;square=_np_product(u,u)
    return diffusion+r*(u-square)


def _teacher(u,h,kappa,r,target,*,fine=False,dense=False,budget=None):
    def fun(t,x):
        if budget is not None:budget.check()
        return _np_rhs(x,kappa,r,target)
    solution=solve_ivp(fun,(0,h),u,method='DOP853',rtol=3e-14 if fine else 2.5e-13,
                       atol=1e-15 if fine else 1e-14,dense_output=dense)
    if not solution.success or not np.isfinite(solution.y).all():
        raise RuntimeError('Independent Appendix teacher did not converge: '+solution.message)
    return solution


def _np_logistic(u,t,r):
    return u/(u+(1-u)*np.exp(-r*t))


def _np_df_path(u,h,kappa,r):
    """Independent analytic nodal Strang path derivative, Appendix A7."""
    N=len(u);lam=-4*kappa*N*N*np.sin(np.pi*np.arange(N)/N)**2
    H=lambda x,s:np.fft.ifft(np.fft.fft(x)*np.exp(lam*s)).real
    L=lambda x:np.fft.ifft(np.fft.fft(x)*lam).real
    y=H(u,h/2);z=_np_logistic(y,h,r);a=H(z,h/2)
    q=np.exp(-r*h)/(y+(1-y)*np.exp(-r*h))**2
    derivative=.5*L(a)+H(r*z*(1-z)+.5*q*L(y),h/2)
    return a,derivative,derivative-_np_rhs(a,kappa,r,'fd_same_grid')


def _np_response(u,h,kappa,r,nodes):
    positions,weights=np.polynomial.legendre.leggauss(nodes)
    positions=(positions+1)/2;weights=weights/2
    c=float(u.mean());N=len(u);lam=-4*kappa*N*N*np.sin(np.pi*np.arange(N)/N)**2
    def q(s):return np.exp(-r*s)/(c+(1-c)*np.exp(-r*s))**2
    value=np.zeros_like(u)
    for x,w in zip(positions,weights):
        s=h*x;rho=_np_df_path(u,s,kappa,r)[2]
        value-=h*w*q(h)/q(s)*np.fft.ifft(np.fft.fft(rho)*np.exp(lam*(h-s))).real
    return value


def run(ctx):
    started=time.perf_counter();rows=[];inputs={}
    def remember(name,value):
        array=np.ascontiguousarray(_array(value));inputs[name]=array
        return dict(input_key=name,sha256=hashlib.sha256(array.tobytes()).hexdigest(),shape=list(array.shape),dtype=str(array.dtype))
    def record(name,metrics,checks,config=None):
        ctx.budget.check()
        row=dict(probe=name,metrics=clean(metrics),config=config or {},checks=checks)
        rows.append(row)
        ctx.record(name,['M1'],metrics=metrics,checks=checks,config=config,
                   evidence=['Appendix A designed numerical probe; not independent-cohort confirmation'])
        write_json(ctx.path/'probes.json',dict(schema='tdn.response-math-probes/v1',rows=rows,complete=False))
    def numerical(name,value,tol=2e-10):
        return check(name,float(value),tol,category='correctness')
    targets=('fd_same_grid','galerkin_same_grid')
    eq=Physics(.006,3.)
    # A1: exact transformed equation plus the Nyquist-mass negative control.
    for N,target in itertools.product((15,16),targets):
        ctx.budget.check();g=Geometry((N,N),(1.,1.))
        w=.025*torch.randn(1,1,N,N,dtype=torch.float64,generator=torch.Generator().manual_seed(83041))
        c0=torch.full((1,1,1,1),.37,dtype=torch.float64);bg=k.background(c0,.12,3.)
        u=bg['c']+bg['q']*w
        transformed=bg['c_derivative']+bg['q_derivative']*w+bg['q']*(n.generator(w,eq,g,target)-3*bg['q']*n.product(w,w,g,target))
        discrepancy=float((n.rhs(u,eq,g,target)-transformed).abs().max())
        moments=tr.moment_rhs(u,eq,g,target);mean_error=float((n.mean(n.rhs(u,eq,g,target))-moments['mean_derivative']).abs().max())
        wrong=float((n.mean(n.center(u).square())-moments['variance']).abs().max())
        checks=[numerical('transformed_rhs_identity',discrepancy),numerical('mean_identity',mean_error)]
        if N%2==0 and target=='galerkin_same_grid':checks.append(check('wrong_nodal_mass_detected',wrong,1e-7,'gt',category='correctness'))
        record(f'A1/{target}/{N}',dict(rhs_max_discrepancy=discrepancy,mean_discrepancy=mean_error,nodal_vs_physical_variance=wrong),checks,
               dict(seed=83041,h=.12,kappa=.006,reaction_rate=3,**remember(f'A1_{N}',w)))
    # A2: full physical product versus an independently reduced spectral sum.
    for N,target in itertools.product((15,16),targets):
        ctx.budget.check();g=Geometry((N,N),(1.,1.))
        u=.37+.04*torch.randn(1,1,N,N,dtype=torch.float64,generator=torch.Generator().manual_seed(80231))
        full=n.mean(n.quadratic_defect(u,.12,eq,g,target,nodes=64))
        spectral=k.mean_quadratic_spectral(u,.12,eq,g,target,nodes=64)
        refined=k.mean_quadratic_spectral(u,.12,eq,g,target,nodes=96)
        parity=float((full-spectral).abs().max());refinement=float((refined-spectral).abs().max())
        record(f'A2/{target}/{N}',dict(mean_reduction_discrepancy=parity,GL64_GL96_mean_disagreement=refinement,
            scope='Algebraic reduction and mean-only quadrature refinement; not full PDE accuracy'),
            [numerical('mean_reduction_parity',parity,1e-14),numerical('mean_quadrature_refinement',refinement,1e-13)],
            dict(seed=80231,**remember(f'A2_{N}',u)))
    # A3: asymptotic coefficient conventions, explicitly not a finite-step guarantee.
    for target in targets:
        g=Geometry((10,10),(1.,1.));eq3=Physics(.02,3.)
        u=.4+.08*torch.randn(1,1,10,10,dtype=torch.float64,generator=torch.Generator().manual_seed(92841))
        errors=[];disagreements=[]
        for h in (.002,.001,.0005,.00025):
            ctx.budget.check()
            lead=k.leading_cubic_defect(u,h,eq3,g,target)
            c8=k.nested_cubic_defect(u,h,eq3,g,target,nodes=8)
            c12=k.nested_cubic_defect(u,h,eq3,g,target,nodes=12)
            errors.append(_rms(_array(c12-lead)));disagreements.append(_rms(_array(c12-c8)))
        orders=[math.log(errors[j]/errors[j+1],2) for j in range(3)]
        record(f'A3/{target}',dict(remainders=errors,remainder_orders=orders,nested_GL8_GL12_disagreement=disagreements,
            interpretation='Small-h coefficient check at fixed grid; no stiff order or finite-amplitude claim'),
            [check('leading_coefficient_remainder_order',min(orders),3.7,'gt',category='math'),numerical('nested_quadrature_refinement',max(disagreements),1e-16)],
            dict(h=[.002,.001,.0005,.00025],**remember('A3',u)))
    # A4: fine exact reaction versus independently integrated coarse Galerkin.
    N=17;fineN=257;r=3.;xf=np.arange(fineN)/fineN
    uf=.4+.15*np.cos(2*np.pi*6*xf);uc=sp_resample(uf,N)
    square_unresolved=uf*uf-sp_resample(sp_resample(uf*uf,N),fineN)
    coefficient=r*r*sp_resample(uf*square_unresolved,N)
    closures=[];relative=[];teacher_eps=[];fine_eps=[]
    for h in (.0025,.005,.01,.02,.04):
        ctx.budget.check();coarse=_teacher(uc,h,0,r,'galerkin_same_grid',budget=ctx.budget).y[:,-1]
        refined=_teacher(uc,h,0,r,'galerkin_same_grid',fine=True,budget=ctx.budget).y[:,-1]
        truth=sp_resample(_np_logistic(uf,h,r),N)
        uf2=.4+.15*np.cos(2*np.pi*6*np.arange(514)/514)
        fine_eps.append(_rms(truth-sp_resample(_np_logistic(uf2,h,r),N)))
        difference=truth-coarse;closures.append(_rms(difference));teacher_eps.append(_rms(coarse-refined))
        relative.append(_rms(difference/h**2-coefficient)/_rms(coefficient))
    orders=[math.log(closures[j+1]/closures[j],2) for j in range(4)]
    record('A4/resolved_reaction_only_closure',dict(closure_rms=closures,doubling_orders=orders,coefficient_relative_errors=relative,
        reference_refinement=teacher_eps,spatial_refinement=fine_eps,coefficient_rms=_rms(coefficient)),
        [check('closure_nonzero',closures[1],1e-8,'gt',category='math'),check('quadratic_closure_order',min(orders),1.9,'gt',category='math'),
         check('derived_coefficient_small_h',relative[0],.01,'lt',category='correctness'),numerical('teacher_refinement',max(teacher_eps+fine_eps),1e-11)],
        dict(h=[.0025,.005,.01,.02,.04],coarse_grid=N,fine_grid=fineN,**remember('A4_resolved_fine',uf)))
    hidden=uf+.1*np.cos(2*np.pi*11*xf);observed=sp_resample(hidden,N)
    initial_forcing=sp_resample(r*(hidden-hidden*hidden),N)-r*(observed-_np_product(observed,observed))
    hidden_errors=[]
    for h in (.001,.0005):
        truth=sp_resample(_np_logistic(hidden,h,r),N)
        coarse=_teacher(observed,h,0,r,'galerkin_same_grid',fine=True,budget=ctx.budget).y[:,-1]
        hidden_errors.append(_rms((truth-coarse)/h-initial_forcing)/_rms(initial_forcing))
    record('A4/hidden_initial_mode',dict(relative_first_order_forcing_error=hidden_errors,forcing_rms=_rms(initial_forcing)),
        [check('hidden_mode_first_order_coefficient',hidden_errors[-1],.02,'lt',category='correctness')],
        dict(coarse_observation='Fourier projection, never nodal sampling',**remember('A4_hidden_fine',hidden)))
    # A5: complex coefficient convolution independently resolves phase information.
    phase_rows=[];power=[];N=31;mode=np.arange(-15,16);lam=-.03*(2*np.pi*mode)**2;h=.24;r=8.
    for sign in (1,-1):
        u=.4+.22*np.cos(2*np.pi*3*np.arange(N)/N)+sign*.15*np.cos(2*np.pi*6*np.arange(N)/N)
        coeff=np.fft.fftshift(np.fft.fft(u)/N);power.append(np.abs(coeff)**2)
        def evolve(coeff,diffusion,rtol,atol):
            def fun(t,x):
                ctx.budget.check()
                return diffusion*x+r*(x-np.convolve(x,x)[15:46])
            result=solve_ivp(fun,(0,h),coeff,method='DOP853',rtol=rtol,atol=atol)
            if not result.success:raise RuntimeError(result.message)
            return result.y[:,-1]
        truth=evolve(coeff,lam,2.5e-13,1e-14);truth2=evolve(coeff,lam,3e-14,1e-15)
        midpoint=coeff*np.exp(lam*h/2)
        df=evolve(midpoint,0,2.5e-13,1e-14)*np.exp(lam*h/2)
        df2=evolve(midpoint,0,3e-14,1e-15)*np.exp(lam*h/2)
        c=u.mean();q=lambda t:np.exp(-r*t)/(c+(1-c)*np.exp(-r*t))**2
        energy=np.abs(coeff)**2;energy[15]=0
        integrand=lambda s:q(s)*np.sum(energy*(np.exp(2*lam*s)-np.exp(lam*h)))
        integral,quad_error=quad(integrand,0,h,epsabs=1e-14,epsrel=1e-12)
        qm=-r*q(h)*integral;remaining=float((truth[15]-df[15]).real-qm)
        phase_rows.append(dict(sign=sign,target_mean=float(truth[15].real),df_mean=float(df[15].real),quadratic_mean=qm,
            remaining_mean_defect=remaining,teacher_refinement=float(np.max(np.abs(truth-truth2))),
            anchor_refinement=float(np.max(np.abs(df-df2))),quadrature_error_estimate=quad_error,**remember(f'A5_{sign}',u)))
    record('A5/phase_twins',dict(fields=phase_rows,power_discrepancy=float(np.max(np.abs(power[0]-power[1]))),
        interpretation='Opposite remaining mean corrections after the same converged analytic quadratic control'),
        [numerical('identical_power',np.max(np.abs(power[0]-power[1])),1e-14),
         check('opposite_residual_mean_signs',phase_rows[0]['remaining_mean_defect']*phase_rows[1]['remaining_mean_defect'],0,'lt',category='math'),
         numerical('independent_teacher_refinement',max(x['teacher_refinement'] for x in phase_rows),1e-11)])
    # A6: rational scalar identity and well-resolved finite-amplitude counterexamples.
    c=.1;v=.4;z=3.;D=c+(1-c)*np.exp(-z);q=np.exp(-z)/D**2;beta=-np.expm1(-z)/D
    exact=_np_logistic(c+v,1,z);base=_np_logistic(c,1,z)
    trunc=[base+q*sum((-beta)**j*v**(j+1) for j in range(order)) for order in range(1,5)]
    rational=base+q*v/(1+beta*v)
    record('A6/scalar_amplitude_radius',dict(chi=beta*abs(v),exact=exact,truncations=trunc,rational=rational),
        [numerical('rational_identity',abs(exact-rational),1e-13),check('valid_input_outside_series_radius',beta*abs(v),1,'gt',category='math')])
    N=32;x=np.arange(N)/N;bell=np.exp(5*(np.cos(2*np.pi*x)-1));bell-=bell.mean();g=Geometry((N,),(1.,));eq6=Physics(.003,25.)
    baseline_rows=[]
    for amplitude in (.05,.2,.5,.75):
        ctx.budget.check();u=.15+amplitude*bell;ut=_tensor(u);h=.12
        start=time.perf_counter();sol=_teacher(u,h,.003,25.,'fd_same_grid',budget=ctx.budget);truth=sol.y[:,-1]
        refined=_teacher(u,h,.003,25.,'fd_same_grid',fine=True,budget=ctx.budget).y[:,-1];teacher_seconds=time.perf_counter()-start
        start=time.perf_counter();a=n.anchor(ut,h,eq6,g,'fd_same_grid');a_seconds=time.perf_counter()-start
        Q=n.quadratic_defect(ut,h,eq6,g,'fd_same_grid',nodes=32)
        Q2=n.quadratic_defect(ut,h,eq6,g,'fd_same_grid',nodes=64)
        C=k.nested_cubic_defect(ut,h,eq6,g,'fd_same_grid',nodes=12)
        C2=k.nested_cubic_defect(ut,h,eq6,g,'fd_same_grid',nodes=20)
        mean0=u.mean();chi=float(k.background(n.mean(ut),h,25.)['beta'])*np.max(np.abs(u-mean0))
        start=time.perf_counter();response=_np_response(u,h,.003,25.,16);response_seconds=time.perf_counter()-start
        response_refined=_np_response(u,h,.003,25.,32)
        row=dict(amplitude=amplitude,chi=chi,min_initial=float(u.min()),max_initial=float(u.max()),
            df_rms=_rms(truth-_array(a).ravel()),quadratic_rms=_rms(truth-_array(a+Q).ravel()),
            quadratic_cubic_rms=_rms(truth-_array(a+Q+C).ravel()),residual_response_rms=_rms(truth-_array(a).ravel()-response),
            teacher_disagreement=_rms(truth-refined),quadratic_GL32_GL64=_rms(_array(Q-Q2)),cubic_GL12_GL20=_rms(_array(C-C2)),
            response_GL16_GL32=_rms(response-response_refined),teacher_wall_seconds=teacher_seconds,
            torch_df_single_wall_seconds=a_seconds,numpy_response_single_wall_seconds=response_seconds,
            timing_scope='Observed probe timers, different implementations; not paired deployment timing',**remember(f'A6_{amplitude}',u))
        baseline_rows.append(row)
        record(f'A6/fd_amplitude_{amplitude:g}',row,[numerical('reference_refinement',row['teacher_disagreement'],1e-10),
            numerical('cubic_quadrature_refinement',row['cubic_GL12_GL20'],1e-9),
            numerical('response_quadrature_refinement',row['response_GL16_GL32'],1e-10)])
    record('A6/finite_amplitude_counterexample',dict(cubic_worsens_count=sum(x['quadratic_cubic_rms']>x['df_rms'] for x in baseline_rows),denominator=4),
        [check('counterexample_resolved',sum(x['quadratic_cubic_rms']>x['df_rms'] for x in baseline_rows),1,'ge',category='math')])
    # A7: designed full panel, individually refined response and teacher.
    panel=[]
    for (mean0,sign),rate,kap,amplitude in itertools.product(((.15,1),(.85,-1)),(3.,25.,60.),(.0003,.003,.03),(.05,.5,.75)):
        ctx.budget.check();u=mean0+sign*amplitude*bell;h=.12
        start=time.perf_counter();truth=_teacher(u,h,kap,rate,'fd_same_grid',budget=ctx.budget).y[:,-1]
        refined=_teacher(u,h,kap,rate,'fd_same_grid',fine=True,budget=ctx.budget).y[:,-1];teacher_seconds=time.perf_counter()-start
        start=time.perf_counter();response=_np_response(u,h,kap,rate,24);response_seconds=time.perf_counter()-start
        response2=_np_response(u,h,kap,rate,40)
        approx=[];step_seconds=[]
        for count in (1,2,4):
            start=time.perf_counter();value=u.copy()
            for _ in range(count):value=_np_df_path(value,h/count,kap,rate)[0]
            approx.append(value);step_seconds.append(time.perf_counter()-start)
        errors=[_rms(truth-value) for value in approx];corrected=_rms(truth-approx[0]-response)
        row=dict(mean=mean0,sign=sign,r=rate,kappa=kap,amplitude=amplitude,df_errors=errors,response_rms=corrected,
            ratios=[corrected/error for error in errors],teacher_disagreement=_rms(truth-refined),
            response_GL24_GL40=_rms(response-response2),teacher_wall_seconds=teacher_seconds,
            response_single_wall_seconds=response_seconds,df_path_single_wall_seconds=step_seconds,
            timing_scope='Unpaired CPU diagnostic timers including unused path derivative; not deployment latency',
            independent_shape_template='peak' if sign==1 else 'hole')
        panel.append(row)
        record(f'A7/{sign}/{rate:g}/{kap:g}/{amplitude:g}',row,
               [numerical('teacher_refinement',row['teacher_disagreement'],1e-10),numerical('response_GL24_GL40',row['response_GL24_GL40'],1e-9)])
    sample=.15+.5*bell;ut=_tensor(sample);pt=tr.anchor_path(ut,.12,eq6,g,'fd_same_grid',create_graph=False)
    np_path=_np_df_path(sample,.12,.003,25.)
    derivative_error=_rms(_array(pt['derivative']).ravel()-np_path[1])
    central=(_np_df_path(sample,.120001,.003,25.)[0]-_np_df_path(sample,.119999,.003,25.)[0])/(2e-6)
    central_error=_rms(central-np_path[1])
    record('A7/designed_panel_summary',dict(designed_cases=len(panel),independent_shape_templates=2,
        wins_vs_df_1_2_4=[sum(row['ratios'][i]<1 for row in panel) for i in range(3)],
        median_ratios_vs_df_1_2_4=[float(np.median([row['ratios'][i] for row in panel])) for i in range(3)],
        worst_ratio_vs_df=max(row['ratios'][0] for row in panel),analytic_vs_autograd_derivative=derivative_error,
        analytic_vs_central_derivative=central_error,claim='Designed numerical control; no trained model and no speedup claim'),
        [numerical('analytic_vs_autograd_path_derivative',derivative_error,1e-11),numerical('path_finite_difference',central_error,1e-8)])
    # A8: true variance clock, not an inferred latent variable.
    dense=_teacher(sample,.12,.003,25.,'fd_same_grid',fine=True,dense=True,budget=ctx.budget)
    def clock_integrand(t):
        ctx.budget.check();value=dense.sol(t);m=value.mean();V=np.mean((value-m)**2)
        return V/(m*(1-m))
    zint,zerror=quad(clock_integrand,0,.12,epsabs=1e-13,epsrel=1e-11)
    m0=sample.mean();clock=1/(1+np.exp(-(np.log(m0/(1-m0))+25*(.12-zint))))
    true_mean=float(dense.y[:,-1].mean());clock_error=abs(clock-true_mean)
    record('A8/true_variance_clock',dict(integral_z=zint,quadrature_error_estimate=zerror,effective_time=.12-zint,
        teacher_mean=true_mean,clock_mean=clock,discrepancy=clock_error,scope='Uses future true variance; not available at deployment'),
        [numerical('conditional_clock_identity',clock_error,1e-11)])
    # A9: actual model initialization is additionally covered by model-agent tests.
    source=torch.tensor(2.,dtype=torch.float64)
    def grads(values):
        p=[torch.tensor(v,dtype=torch.float64,requires_grad=True) for v in values]
        loss=(p[0]*(p[1]*source)*(p[2]*source)-1)**2
        return [float(x) for x in torch.autograd.grad(loss,p)]
    dead=grads((0.,0.,0.));gate=grads((0.,1.,1.));awake=grads((.01,1.,1.))
    record('A9/initialization',dict(all_zero_gradients=dead,final_gate_zero_gradients=gate,post_gate_update_gradients=awake),
        [check('dead_product_detected',max(abs(x) for x in dead),0,'eq',category='correctness'),
         check('gate_live',abs(gate[0]),0,'gt',category='correctness'),
         check('trunk_awakes',min(abs(x) for x in awake[1:]),0,'gt',category='correctness')])
    np.savez_compressed(ctx.path/'probe_inputs.npz',**inputs)
    summary=dict(schema='tdn.response-math-audit/v1',complete=True,probe_rows=len(rows),
        appendix_sections=['A'+str(j) for j in range(1,10)],designed_A7_cases=len(panel),
        independent_A7_shape_templates=2,wall_seconds=time.perf_counter()-started,
        libraries=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__),
        arithmetic='CPU FP64/complex128; Torch autograd and independently coded NumPy/SciPy DOP853/convolution',
        scope='Implementation and mathematical diagnostics, not a neural win, novelty claim or solver speedup',
        inputs='probe_inputs.npz',source_files=['audit.py','contracts.py','numerics.py','trajectory.py','kernels.py'])
    write_json(ctx.path/'probes.json',dict(schema='tdn.response-math-probes/v1',rows=rows,complete=True))
    write_json(ctx.path/'audit.json',summary)
    return summary
